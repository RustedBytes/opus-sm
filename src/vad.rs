use std::fs::{self, File};
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow_array::{RecordBatch, RecordBatchReader};
use parquet::arrow::arrow_writer::ArrowWriter;
use rayon::prelude::*;

use crate::analyze::{AnalysisOptions, DecisionOptions, SegmentKind, segment};
use crate::audio::{
    ChunkOutputFormat, analyze_audio, chunk_output_sample_rate, decode_audio, default_chunk_output_path,
    encode_audio_with_format, rewrite_chunk_output_path, speech_chunks,
};
use crate::cli::VadArgs;
use crate::parquet_io::{
    AudioColumns, analysis_options, binary_value, collect_parquet_files, decision_options,
    empty_like, map_output_path, open_reader, rebuild_batch_from_indices_with_overrides,
    string_value, vad_output_schema, validate_optional_seconds,
};

const MAX_BINARY_PAYLOAD_PER_BATCH: usize = 1_500_000_000;
const MAX_STRING_PAYLOAD_PER_BATCH: usize = 1_500_000_000;

#[derive(Debug, Clone)]
pub struct VadChunk {
    pub index: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub duration_seconds: f64,
    pub sampling_rate: u32,
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct VadOptions<'a> {
    pub fade_ms: f32,
    pub min_speech_seconds: Option<f32>,
    pub max_speech_seconds: Option<f32>,
    pub chunk_format: ChunkOutputFormat,
    pub source_path: Option<&'a str>,
}

pub fn run_vad(args: &VadArgs) -> Result<()> {
    let analysis = analysis_options(&args.analysis);
    let decision = decision_options(&args.decision)?;
    validate_optional_seconds(args.min_speech_seconds, "min-speech-seconds")?;
    validate_optional_seconds(args.max_speech_seconds, "max-speech-seconds")?;
    if let (Some(min), Some(max)) = (args.min_speech_seconds, args.max_speech_seconds)
        && min > max
    {
        bail!("min-speech-seconds must be <= max-speech-seconds");
    }

    for input_path in collect_parquet_files(&args.input.input)? {
        let output_path = map_output_path(&args.input.input, &args.output, &input_path)?;
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }

        let mut reader = open_reader(&input_path, args.input.batch_size)?;
        let schema = vad_output_schema(reader.schema().as_ref())?;
        let output = File::create(&output_path)
            .with_context(|| format!("create output file {}", output_path.display()))?;
        let mut writer = ArrowWriter::try_new(output, schema, None)
            .with_context(|| format!("open {}", output_path.display()))?;

        for batch in &mut reader {
            let batch =
                batch.with_context(|| format!("read batch from {}", input_path.display()))?;
            let chunked_batches = vad_batches(
                &batch,
                analysis,
                decision,
                args.strip.fade_ms,
                args.min_speech_seconds,
                args.max_speech_seconds,
                args.chunk_format,
            )
            .with_context(|| format!("run vad in {}", input_path.display()))?;
            for chunked in chunked_batches {
                if chunked.num_rows() == 0 {
                    continue;
                }
                writer
                    .write(&chunked)
                    .with_context(|| format!("write batch to {}", output_path.display()))?;
            }
        }

        writer
            .close()
            .with_context(|| format!("close {}", output_path.display()))?;
    }

    Ok(())
}

pub fn vad_bytes(
    bytes: &[u8],
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    options: VadOptions<'_>,
) -> Result<Vec<VadChunk>> {
    validate_optional_seconds(options.min_speech_seconds, "min-speech-seconds")?;
    validate_optional_seconds(options.max_speech_seconds, "max-speech-seconds")?;
    if let (Some(min), Some(max)) = (options.min_speech_seconds, options.max_speech_seconds)
        && min > max
    {
        bail!("min-speech-seconds must be <= max-speech-seconds");
    }

    let decoded = decode_audio(bytes)?;
    let probabilities = analyze_audio(&decoded, analysis)?;
    let segments = segment(&probabilities, decision);
    let speech_segments = segments
        .into_iter()
        .filter(|segment| segment.kind == SegmentKind::Speech)
        .collect::<Vec<_>>();
    let chunk_audio = speech_chunks(
        &decoded,
        decision,
        analysis,
        options.fade_ms,
        options.min_speech_seconds,
        options.max_speech_seconds,
    )?;
    if chunk_audio.is_empty() {
        return Ok(Vec::new());
    }

    let base_path = options
        .source_path
        .map(ToOwned::to_owned)
        .map(|path| rewrite_chunk_output_path(&path, &decoded, options.chunk_format))
        .transpose()?
        .unwrap_or(default_chunk_output_path(&decoded, options.chunk_format)?);

    let mut chunks = Vec::new();
    let mut chunk_cursor = 0usize;
    let chunk_sample_rate = chunk_output_sample_rate(&decoded, options.chunk_format)?;
    for speech_segment in speech_segments {
        let segment_duration = speech_segment.end_seconds - speech_segment.start_seconds;
        let max_seconds = options.max_speech_seconds.map(|value| value as f64);
        let pieces = match max_seconds {
            Some(max) if max > 0.0 => (segment_duration / max).ceil().max(1.0) as usize,
            _ => 1,
        };
        for piece in 0..pieces {
            if chunk_cursor >= chunk_audio.len() {
                break;
            }
            let start_seconds = match max_seconds {
                Some(max) => speech_segment.start_seconds + piece as f64 * max,
                None => speech_segment.start_seconds,
            };
            let end_seconds = match max_seconds {
                Some(max) => (speech_segment.start_seconds + (piece + 1) as f64 * max)
                    .min(speech_segment.end_seconds),
                None => speech_segment.end_seconds,
            };
            let chunk = &chunk_audio[chunk_cursor];
            let bytes = encode_audio_with_format(&decoded, chunk, options.chunk_format)?;
            chunks.push(VadChunk {
                index: chunk_cursor,
                start_seconds,
                end_seconds,
                duration_seconds: end_seconds - start_seconds,
                sampling_rate: chunk_sample_rate,
                path: rewrite_chunk_path(&base_path, chunk_cursor),
                bytes,
            });
            chunk_cursor += 1;
        }
    }

    Ok(chunks)
}

fn vad_batches(
    batch: &RecordBatch,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
    chunk_format: ChunkOutputFormat,
) -> Result<Vec<RecordBatch>> {
    let audio = AudioColumns::new(batch)?;
    let schema = vad_output_schema(batch.schema().as_ref())?;
    let chunked_rows = (0..batch.num_rows())
        .into_par_iter()
        .map(|row| {
            process_vad_row(
                &audio,
                row,
                analysis,
                decision,
                VadOptions {
                    fade_ms,
                    min_speech_seconds,
                    max_speech_seconds,
                    chunk_format,
                    source_path: None,
                },
            )
        })
        .collect::<Vec<_>>();

    let mut source_indices = Vec::new();
    let mut bytes_out = Vec::new();
    let mut sampling_rate_out = Vec::new();
    let mut path_out = Vec::new();
    let mut duration_out = Vec::new();
    let mut batches = Vec::new();
    let mut binary_bytes = 0usize;
    let mut string_bytes = 0usize;

    for (row_idx, outputs) in chunked_rows.into_iter().enumerate() {
        for chunk in outputs? {
            let chunk_binary = chunk.bytes.len();
            let chunk_string = chunk.path.len();
            let should_flush = !source_indices.is_empty()
                && (binary_bytes.saturating_add(chunk_binary) > MAX_BINARY_PAYLOAD_PER_BATCH
                    || string_bytes.saturating_add(chunk_string) > MAX_STRING_PAYLOAD_PER_BATCH);
            if should_flush {
                batches.push(rebuild_batch_from_indices_with_overrides(
                    &schema,
                    batch,
                    &audio,
                    &source_indices,
                    &bytes_out,
                    &sampling_rate_out,
                    &path_out,
                    Some(&duration_out),
                    Some("-"),
                )?);
                source_indices.clear();
                bytes_out.clear();
                sampling_rate_out.clear();
                path_out.clear();
                duration_out.clear();
                binary_bytes = 0;
                string_bytes = 0;
            }
            source_indices.push(row_idx as u32);
            bytes_out.push(Some(chunk.bytes));
            sampling_rate_out.push(chunk.sampling_rate);
            path_out.push(Some(chunk.path));
            duration_out.push(chunk.duration_seconds);
            binary_bytes += chunk_binary;
            string_bytes += chunk_string;
        }
    }

    if source_indices.is_empty() {
        if batches.is_empty() {
            return Ok(vec![empty_like(batch)?]);
        }
        return Ok(batches);
    }

    batches.push(rebuild_batch_from_indices_with_overrides(
        &schema,
        batch,
        &audio,
        &source_indices,
        &bytes_out,
        &sampling_rate_out,
        &path_out,
        Some(&duration_out),
        Some("-"),
    )?);

    Ok(batches)
}

fn process_vad_row<'a>(
    audio: &AudioColumns<'a>,
    row: usize,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    mut options: VadOptions<'a>,
) -> Result<Vec<VadChunk>> {
    let Some(bytes) = binary_value(audio.bytes.as_ref(), row)? else {
        return Ok(Vec::new());
    };
    options.source_path = string_value(audio.path.as_ref(), row)?;
    vad_bytes(bytes, analysis, decision, options).with_context(|| format!("extract speech chunks row {row}"))
}

pub(crate) fn rewrite_chunk_path(path: &str, chunk_index: usize) -> String {
    let input = Path::new(path);
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    let ext = input
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let file_name = if ext.is_empty() {
        format!("{stem}_chunk{}", chunk_index + 1)
    } else {
        format!("{stem}_chunk{}.{}", chunk_index + 1, ext)
    };

    if let Some(parent) = input.parent() {
        if parent.as_os_str().is_empty() {
            file_name
        } else {
            parent.join(file_name).to_string_lossy().into_owned()
        }
    } else {
        file_name
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{VadOptions, vad_bytes};
    use crate::analyze::{AnalysisOptions, DecisionOptions, RowDecisionMode};
    use crate::audio::ChunkOutputFormat;
    use crate::parquet_io::{AudioColumns, binary_value, open_reader, string_value};

    #[test]
    fn vad_accepts_mp3_rows_from_radio_free_dataset() {
        let input = Path::new("testdata/radio-free/train-00384-of-00385.parquet");
        let mut reader = open_reader(input, 1).expect("open parquet reader");
        let batch = reader
            .next()
            .expect("first batch exists")
            .expect("first batch is valid");
        let audio = AudioColumns::new(&batch).expect("extract audio columns");
        let bytes = binary_value(audio.bytes.as_ref(), 0)
            .expect("read bytes")
            .expect("row 0 has audio bytes");
        let source_path = string_value(audio.path.as_ref(), 0)
            .expect("read path")
            .expect("row 0 has audio path");

        let chunks = vad_bytes(
            bytes,
            AnalysisOptions { smooth_window: 1 },
            DecisionOptions {
                threshold: 0.8,
                low_threshold: 0.8,
                high_threshold: 0.8,
                min_music_frames: 0,
                min_speech_frames: 0,
                row_decision: RowDecisionMode::Max,
                row_fraction: 0.5,
            },
            VadOptions {
                fade_ms: 0.0,
                min_speech_seconds: Some(0.1),
                max_speech_seconds: Some(30.0),
                chunk_format: ChunkOutputFormat::Auto,
                source_path: Some(source_path),
            },
        )
        .expect("run vad on embedded mp3 bytes");

        assert!(chunks.iter().all(|chunk| chunk.path.ends_with(".wav")));
    }
}
