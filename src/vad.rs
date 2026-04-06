use std::fs::{self, File};
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow_array::{RecordBatch, RecordBatchReader};
use parquet::arrow::arrow_writer::ArrowWriter;
use rayon::prelude::*;

use crate::analyze::{AnalysisOptions, DecisionOptions, SegmentKind, segment};
use crate::audio::{analyze_audio, decode_audio, encode_audio, rewrite_encoded_audio_path, speech_chunks};
use crate::cli::VadArgs;
use crate::parquet_io::{
    AudioColumns, analysis_options, binary_value, collect_parquet_files, decision_options,
    empty_like, map_output_path, open_reader, rebuild_batch_with_overrides, string_value, take_rows,
    validate_optional_seconds,
};

#[derive(Debug, Clone)]
pub struct VadChunk {
    pub index: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub duration_seconds: f64,
    pub path: String,
    pub bytes: Vec<u8>,
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
        let schema = reader.schema().clone();
        let output = File::create(&output_path)
            .with_context(|| format!("create output file {}", output_path.display()))?;
        let mut writer = ArrowWriter::try_new(output, schema, None)
            .with_context(|| format!("open {}", output_path.display()))?;

        for batch in &mut reader {
            let batch =
                batch.with_context(|| format!("read batch from {}", input_path.display()))?;
            let chunked = vad_batch(
                &batch,
                analysis,
                decision,
                args.strip.fade_ms,
                args.min_speech_seconds,
                args.max_speech_seconds,
            )
            .with_context(|| format!("run vad in {}", input_path.display()))?;
            if chunked.num_rows() > 0 {
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
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
    source_path: Option<&str>,
) -> Result<Vec<VadChunk>> {
    validate_optional_seconds(min_speech_seconds, "min-speech-seconds")?;
    validate_optional_seconds(max_speech_seconds, "max-speech-seconds")?;
    if let (Some(min), Some(max)) = (min_speech_seconds, max_speech_seconds)
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
        fade_ms,
        min_speech_seconds,
        max_speech_seconds,
    )?;
    if chunk_audio.is_empty() {
        return Ok(Vec::new());
    }

    let base_path = source_path
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| default_chunk_basename(decoded.format));

    let mut chunks = Vec::new();
    let mut chunk_cursor = 0usize;
    for speech_segment in speech_segments {
        let segment_duration = speech_segment.end_seconds - speech_segment.start_seconds;
        let max_seconds = max_speech_seconds.map(|value| value as f64);
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
            let bytes = encode_audio(&decoded, chunk)?;
            chunks.push(VadChunk {
                index: chunk_cursor,
                start_seconds,
                end_seconds,
                duration_seconds: end_seconds - start_seconds,
                path: rewrite_chunk_path(&base_path, decoded.format, chunk_cursor),
                bytes,
            });
            chunk_cursor += 1;
        }
    }

    Ok(chunks)
}

fn vad_batch(
    batch: &RecordBatch,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
) -> Result<RecordBatch> {
    let audio = AudioColumns::new(batch)?;
    let chunked_rows = (0..batch.num_rows())
        .into_par_iter()
        .map(|row| process_vad_row(&audio, row, analysis, decision, fade_ms, min_speech_seconds, max_speech_seconds))
        .collect::<Vec<_>>();

    let mut source_indices = Vec::new();
    let mut bytes_out = Vec::new();
    let mut path_out = Vec::new();
    let mut duration_out = Vec::new();

    for (row_idx, outputs) in chunked_rows.into_iter().enumerate() {
        for chunk in outputs? {
            source_indices.push(row_idx as u32);
            bytes_out.push(Some(chunk.bytes));
            path_out.push(Some(chunk.path));
            duration_out.push(chunk.duration_seconds);
        }
    }

    if source_indices.is_empty() {
        return empty_like(batch);
    }

    let base = take_rows(batch, &source_indices)?;
    let base_audio = AudioColumns::new(&base)?;
    rebuild_batch_with_overrides(
        &base,
        &base_audio,
        &bytes_out,
        &path_out,
        Some(&duration_out),
        Some("-"),
    )
}

fn process_vad_row(
    audio: &AudioColumns<'_>,
    row: usize,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
) -> Result<Vec<VadChunk>> {
    let Some(bytes) = binary_value(audio.bytes.as_ref(), row)? else {
        return Ok(Vec::new());
    };
    let source_path = string_value(audio.path.as_ref(), row)?;
    vad_bytes(
        bytes,
        analysis,
        decision,
        fade_ms,
        min_speech_seconds,
        max_speech_seconds,
        source_path,
    )
    .with_context(|| format!("extract speech chunks row {row}"))
}

pub(crate) fn rewrite_chunk_path(
    path: &str,
    format: crate::audio::AudioFormat,
    chunk_index: usize,
) -> String {
    let rewritten = rewrite_encoded_audio_path(path, format);
    let input = Path::new(&rewritten);
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    let ext = input.extension().and_then(|value| value.to_str()).unwrap_or("");
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

pub(crate) fn default_chunk_basename(format: crate::audio::AudioFormat) -> String {
    match format {
        crate::audio::AudioFormat::Wav(_) => "audio.wav".to_owned(),
        crate::audio::AudioFormat::OggOpus => "audio.opus".to_owned(),
        crate::audio::AudioFormat::Mp3 => "audio.wav".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::vad_bytes;
    use crate::analyze::{AnalysisOptions, DecisionOptions, RowDecisionMode};
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
            0.0,
            Some(0.1),
            Some(30.0),
            Some(source_path),
        )
        .expect("run vad on embedded mp3 bytes");

        assert!(chunks.iter().all(|chunk| chunk.path.ends_with(".wav")));
    }
}
