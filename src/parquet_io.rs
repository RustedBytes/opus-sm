use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, LargeBinaryArray,
    LargeStringArray, RecordBatch, RecordBatchReader, StringArray, StringViewArray, StructArray,
    UInt32Array,
};
use arrow_schema::DataType;
use arrow_select::take::take;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use rayon::prelude::*;

use crate::analyze::{AnalysisOptions, DecisionOptions, SegmentKind, row_is_music, segment};
use crate::audio::{analyze_audio, decode_audio, encode_audio, rewrite_encoded_audio_path, strip_music};
use crate::cli::{AnalyzeArgs, DecisionArgs, SegmentArgs, SeparateSmArgs, StripMusicArgs};

pub fn run_analyze(args: &AnalyzeArgs) -> Result<()> {
    let analysis = analysis_options(&args.analysis);

    for input_path in collect_parquet_files(&args.input.input)? {
        let mut reader = open_reader(&input_path, args.input.batch_size)?;
        for batch in &mut reader {
            let batch =
                batch.with_context(|| format!("read batch from {}", input_path.display()))?;
            let audio = AudioColumns::new(&batch)?;
            let rows = analyze_batch_rows(&audio, analysis, &input_path)?;
            for probabilities in rows.into_iter().flatten() {
                print_probabilities(&input_path, probabilities.row, &probabilities.frames);
            }
        }
    }
    Ok(())
}

pub fn run_segment(args: &SegmentArgs) -> Result<()> {
    let analysis = analysis_options(&args.analysis);
    let decision = decision_options(&args.decision)?;

    for input_path in collect_parquet_files(&args.input.input)? {
        let mut reader = open_reader(&input_path, args.input.batch_size)?;
        for batch in &mut reader {
            let batch =
                batch.with_context(|| format!("read batch from {}", input_path.display()))?;
            let audio = AudioColumns::new(&batch)?;
            let rows = analyze_batch_rows(&audio, analysis, &input_path)?;
            for row in rows {
                let Some(probabilities) = row else {
                    continue;
                };
                let row = probabilities.row;
                println!("FILE\t{}\tROW\t{}", input_path.display(), row);
                print_probabilities(&input_path, row, &probabilities.frames);
                for segment in segment(&probabilities.frames, decision) {
                    let label = if segment.kind == SegmentKind::Music {
                        "music"
                    } else {
                        "speech"
                    };
                    println!(
                        "SEGMENT\t{}\t{}\t{}\t{:.3}\t{:.3}",
                        label,
                        segment.start_frame,
                        segment.end_frame,
                        segment.start_seconds,
                        segment.end_seconds
                    );
                }
            }
        }
    }
    Ok(())
}

pub fn run_strip_music(args: &StripMusicArgs) -> Result<()> {
    let analysis = analysis_options(&args.analysis);
    let decision = decision_options(&args.decision)?;

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
            let stripped = strip_batch(&batch, analysis, decision, args.strip.fade_ms)
                .with_context(|| format!("strip music in {}", input_path.display()))?;
            writer
                .write(&stripped)
                .with_context(|| format!("write batch to {}", output_path.display()))?;
        }

        writer
            .close()
            .with_context(|| format!("close {}", output_path.display()))?;
    }
    Ok(())
}

pub fn run_separate_sm(args: &SeparateSmArgs) -> Result<()> {
    let analysis = analysis_options(&args.analysis);
    let decision = decision_options(&args.decision)?;

    for input_path in collect_parquet_files(&args.input.input)? {
        let speech_path = map_output_path(&args.input.input, &args.speech_output, &input_path)?;
        let music_path = map_output_path(&args.input.input, &args.music_output, &input_path)?;

        if let Some(parent) = speech_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        if let Some(parent) = music_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }

        let mut reader = open_reader(&input_path, args.input.batch_size)?;
        let schema = reader.schema().clone();
        let mut speech_writer = ArrowWriter::try_new(
            File::create(&speech_path)
                .with_context(|| format!("create {}", speech_path.display()))?,
            schema.clone(),
            None,
        )
        .with_context(|| format!("open {}", speech_path.display()))?;
        let mut music_writer = ArrowWriter::try_new(
            File::create(&music_path)
                .with_context(|| format!("create {}", music_path.display()))?,
            schema,
            None,
        )
        .with_context(|| format!("open {}", music_path.display()))?;

        for batch in &mut reader {
            let batch =
                batch.with_context(|| format!("read batch from {}", input_path.display()))?;
            let (speech_batch, music_batch) = classify_batch(&batch, analysis, decision)
                .with_context(|| format!("classify rows in {}", input_path.display()))?;

            if speech_batch.num_rows() > 0 {
                speech_writer
                    .write(&speech_batch)
                    .with_context(|| format!("write speech batch to {}", speech_path.display()))?;
            }
            if music_batch.num_rows() > 0 {
                music_writer
                    .write(&music_batch)
                    .with_context(|| format!("write music batch to {}", music_path.display()))?;
            }
        }

        speech_writer
            .close()
            .with_context(|| format!("close {}", speech_path.display()))?;
        music_writer
            .close()
            .with_context(|| format!("close {}", music_path.display()))?;
    }

    Ok(())
}

fn strip_batch(
    batch: &RecordBatch,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    fade_ms: f32,
) -> Result<RecordBatch> {
    let audio = AudioColumns::new(batch)?;
    let row_outputs = (0..batch.num_rows())
        .into_par_iter()
        .map(|row| process_strip_row(&audio, row, analysis, decision, fade_ms))
        .collect::<Vec<_>>();

    let mut bytes_out = Vec::with_capacity(batch.num_rows());
    let mut path_out = Vec::with_capacity(batch.num_rows());
    for output in row_outputs {
        let output = output?;
        bytes_out.push(output.bytes);
        path_out.push(output.path);
    }

    rebuild_batch(batch, &audio, &bytes_out, &path_out)
}

fn classify_batch(
    batch: &RecordBatch,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
) -> Result<(RecordBatch, RecordBatch)> {
    let audio = AudioColumns::new(batch)?;
    let row_kinds = (0..batch.num_rows())
        .into_par_iter()
        .map(|row| classify_row(&audio, row, analysis, decision))
        .collect::<Vec<_>>();

    let mut speech_indices = Vec::new();
    let mut music_indices = Vec::new();
    for (row, kind) in row_kinds.into_iter().enumerate() {
        match kind? {
            RowKind::Speech => speech_indices.push(row as u32),
            RowKind::Music => music_indices.push(row as u32),
        }
    }

    Ok((
        take_rows(batch, &speech_indices)?,
        take_rows(batch, &music_indices)?,
    ))
}

fn analyze_batch_rows(
    audio: &AudioColumns<'_>,
    analysis: AnalysisOptions,
    input_path: &Path,
) -> Result<Vec<Option<RowAnalysis>>> {
    (0..audio.audio_struct.len())
        .into_par_iter()
        .map(|row| analyze_row(audio, row, analysis, input_path))
        .collect()
}

fn analyze_row(
    audio: &AudioColumns<'_>,
    row: usize,
    analysis: AnalysisOptions,
    input_path: &Path,
) -> Result<Option<RowAnalysis>> {
    let Some(bytes) = binary_value(audio.bytes.as_ref(), row)? else {
        return Ok(None);
    };
    let decoded = decode_audio(bytes).with_context(|| format!("decode audio row {row}"))?;
    let frames = analyze_audio(&decoded, analysis)
        .with_context(|| format!("analyze audio row {row} in {}", input_path.display()))?;
    Ok(Some(RowAnalysis { row, frames }))
}

fn process_strip_row(
    audio: &AudioColumns<'_>,
    row: usize,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
    fade_ms: f32,
) -> Result<StripRowOutput> {
    let original_path = string_value(audio.path.as_ref(), row)?.map(ToOwned::to_owned);
    let Some(bytes) = binary_value(audio.bytes.as_ref(), row)? else {
        return Ok(StripRowOutput {
            bytes: None,
            path: original_path,
        });
    };

    let decoded = decode_audio(bytes).with_context(|| format!("decode audio row {row}"))?;
    let stripped = strip_music(&decoded, decision, analysis, fade_ms)
        .with_context(|| format!("strip music row {row}"))?;
    let encoded = encode_audio(&decoded, &stripped).with_context(|| format!("encode row {row}"))?;
    Ok(StripRowOutput {
        bytes: Some(encoded),
        path: original_path.map(|path| rewrite_encoded_audio_path(&path, decoded.format)),
    })
}

fn classify_row(
    audio: &AudioColumns<'_>,
    row: usize,
    analysis: AnalysisOptions,
    decision: DecisionOptions,
) -> Result<RowKind> {
    let Some(bytes) = binary_value(audio.bytes.as_ref(), row)? else {
        return Ok(RowKind::Speech);
    };

    let decoded = decode_audio(bytes).with_context(|| format!("decode audio row {row}"))?;
    let probabilities =
        analyze_audio(&decoded, analysis).with_context(|| format!("analyze audio row {row}"))?;
    Ok(if row_is_music(&probabilities, decision) {
        RowKind::Music
    } else {
        RowKind::Speech
    })
}

pub(crate) fn take_rows(batch: &RecordBatch, indices: &[u32]) -> Result<RecordBatch> {
    let indices = UInt32Array::from(indices.to_vec());
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None).context("take batch rows"))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(batch.schema(), columns).context("build filtered batch")
}

fn rebuild_batch(
    batch: &RecordBatch,
    audio: &AudioColumns<'_>,
    bytes_out: &[Option<Vec<u8>>],
    path_out: &[Option<String>],
) -> Result<RecordBatch> {
    rebuild_batch_with_overrides(batch, audio, bytes_out, path_out, None, None)
}

pub(crate) fn rebuild_batch_with_overrides(
    batch: &RecordBatch,
    audio: &AudioColumns<'_>,
    bytes_out: &[Option<Vec<u8>>],
    path_out: &[Option<String>],
    duration_out: Option<&[f64]>,
    transcription_override: Option<&str>,
) -> Result<RecordBatch> {
    let schema = batch.schema();
    let new_audio = rebuild_audio_struct(
        audio.audio_struct,
        build_binary_array(audio.bytes_type, bytes_out)?,
        build_string_array(audio.path_type, path_out)?,
    )?;
    let mut columns = batch.columns().to_vec();
    columns[audio.audio_index] = Arc::new(new_audio);

    if let Some(duration_out) = duration_out
        && let Some(duration_index) = batch
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == "duration")
        {
            let data_type = schema.field(duration_index).data_type();
            columns[duration_index] = build_duration_array(data_type, duration_out)?;
        }

    if let Some(transcription_override) = transcription_override
        && let Some(transcription_index) = batch
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == "transcription")
        {
            let data_type = schema.field(transcription_index).data_type();
            let values = vec![Some(transcription_override.to_owned()); batch.num_rows()];
            columns[transcription_index] = build_string_array(data_type, &values)?;
        }

    RecordBatch::try_new(batch.schema(), columns).context("build output batch")
}

pub(crate) fn rebuild_batch_from_indices_with_overrides(
    batch: &RecordBatch,
    audio: &AudioColumns<'_>,
    source_indices: &[u32],
    bytes_out: &[Option<Vec<u8>>],
    path_out: &[Option<String>],
    duration_out: Option<&[f64]>,
    transcription_override: Option<&str>,
) -> Result<RecordBatch> {
    let schema = batch.schema();
    let indices = UInt32Array::from(source_indices.to_vec());
    let new_audio = rebuild_audio_struct_from_indices(
        audio.audio_struct,
        &indices,
        build_binary_array(audio.bytes_type, bytes_out)?,
        build_string_array(audio.path_type, path_out)?,
    )?;

    let mut columns = Vec::with_capacity(batch.num_columns());
    for (column_index, field) in schema.fields().iter().enumerate() {
        let column = if column_index == audio.audio_index {
            Arc::new(new_audio.clone()) as ArrayRef
        } else if let Some(duration_out) = duration_out
            && field.name() == "duration"
        {
            build_duration_array(field.data_type(), duration_out)?
        } else if let Some(transcription_override) = transcription_override
            && field.name() == "transcription"
        {
            let values = vec![Some(transcription_override.to_owned()); source_indices.len()];
            build_string_array(field.data_type(), &values)?
        } else {
            take(batch.column(column_index).as_ref(), &indices, None)
                .with_context(|| format!("take column {}", field.name()))?
        };
        columns.push(column);
    }

    RecordBatch::try_new(schema, columns).context("build output batch")
}

fn print_probabilities(
    input_path: &Path,
    row: usize,
    probabilities: &[crate::analyze::FrameProbability],
) {
    for frame in probabilities {
        println!(
            "FRAME\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.6}\t{:.6}",
            input_path.display(),
            row,
            frame.index,
            frame.start_seconds,
            frame.end_seconds,
            frame.music_probability,
            frame.activity_probability
        );
    }
}

#[derive(Debug)]
struct RowAnalysis {
    row: usize,
    frames: Vec<crate::analyze::FrameProbability>,
}

#[derive(Debug)]
struct StripRowOutput {
    bytes: Option<Vec<u8>>,
    path: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum RowKind {
    Speech,
    Music,
}

pub(crate) fn analysis_options(args: &crate::cli::AnalysisArgs) -> AnalysisOptions {
    AnalysisOptions {
        smooth_window: args.smooth_window.max(1),
    }
}

pub(crate) fn decision_options(args: &DecisionArgs) -> Result<DecisionOptions> {
    validate_threshold(args.threshold)?;
    let low_threshold = args.low_threshold.unwrap_or(args.threshold);
    let high_threshold = args.high_threshold.unwrap_or(args.threshold);
    validate_threshold(low_threshold)?;
    validate_threshold(high_threshold)?;
    if low_threshold > high_threshold {
        bail!("low-threshold must be <= high-threshold");
    }
    if !(0.0..=1.0).contains(&args.row_fraction) {
        bail!("row-fraction must be between 0.0 and 1.0");
    }

    Ok(DecisionOptions {
        threshold: args.threshold,
        low_threshold,
        high_threshold,
        min_music_frames: args.min_music_frames,
        min_speech_frames: args.min_speech_frames,
        row_decision: args.row_decision,
        row_fraction: args.row_fraction,
    })
}

pub(crate) fn open_reader(
    input_path: &Path,
    batch_size: usize,
) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
    let input = File::open(input_path).with_context(|| format!("open {}", input_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input)
        .with_context(|| format!("open parquet reader for {}", input_path.display()))?
        .with_batch_size(batch_size);
    builder
        .build()
        .with_context(|| format!("build parquet batch reader for {}", input_path.display()))
}

pub(crate) fn collect_parquet_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        bail!("input path does not exist: {}", input.display());
    }

    let mut files = Vec::new();
    collect_parquet_files_recursive(input, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_parquet_files_recursive(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root).with_context(|| format!("read dir {}", root.display()))? {
        let entry = entry.with_context(|| format!("read entry in {}", root.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_parquet_files_recursive(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "parquet") {
            files.push(path);
        }
    }
    Ok(())
}

pub(crate) fn map_output_path(input_root: &Path, output_root: &Path, input_path: &Path) -> Result<PathBuf> {
    if input_root.is_file() {
        return Ok(output_root.to_path_buf());
    }
    let relative = input_path.strip_prefix(input_root).with_context(|| {
        format!(
            "strip prefix {} from {}",
            input_root.display(),
            input_path.display()
        )
    })?;
    Ok(output_root.join(relative))
}

fn validate_threshold(threshold: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&threshold) {
        bail!("threshold must be between 0.0 and 1.0");
    }
    Ok(())
}

pub(crate) fn validate_optional_seconds(value: Option<f32>, name: &str) -> Result<()> {
    if let Some(value) = value
        && value <= 0.0 {
            bail!("{name} must be > 0");
        }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct AudioColumns<'a> {
    pub(crate) audio_index: usize,
    pub(crate) audio_struct: &'a StructArray,
    pub(crate) bytes: &'a ArrayRef,
    pub(crate) path: &'a ArrayRef,
    pub(crate) bytes_type: &'a DataType,
    pub(crate) path_type: &'a DataType,
}

impl<'a> AudioColumns<'a> {
    pub(crate) fn new(batch: &'a RecordBatch) -> Result<Self> {
        let audio_index = batch
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == "audio")
            .ok_or_else(|| anyhow!("missing audio column"))?;

        let audio_struct = batch.column(audio_index).as_ref().as_struct();
        let bytes = audio_struct
            .column_by_name("bytes")
            .ok_or_else(|| anyhow!("missing audio.bytes field"))?;
        let path = audio_struct
            .column_by_name("path")
            .ok_or_else(|| anyhow!("missing audio.path field"))?;
        let bytes_type = audio_struct
            .fields()
            .iter()
            .find(|field| field.name() == "bytes")
            .ok_or_else(|| anyhow!("missing audio.bytes field metadata"))?
            .data_type();
        let path_type = audio_struct
            .fields()
            .iter()
            .find(|field| field.name() == "path")
            .ok_or_else(|| anyhow!("missing audio.path field metadata"))?
            .data_type();

        Ok(Self {
            audio_index,
            audio_struct,
            bytes,
            path,
            bytes_type,
            path_type,
        })
    }
}

fn rebuild_audio_struct(
    original: &StructArray,
    new_bytes: ArrayRef,
    new_path: ArrayRef,
) -> Result<StructArray> {
    let mut columns = Vec::with_capacity(original.num_columns());
    for field in original.fields() {
        let column = match field.name().as_str() {
            "bytes" => new_bytes.clone(),
            "path" => new_path.clone(),
            name => original
                .column_by_name(name)
                .ok_or_else(|| anyhow!("missing audio field {}", name))?
                .clone(),
        };
        columns.push(column);
    }

    Ok(StructArray::new(
        original.fields().clone(),
        columns,
        original.nulls().cloned(),
    ))
}

fn rebuild_audio_struct_from_indices(
    original: &StructArray,
    indices: &UInt32Array,
    new_bytes: ArrayRef,
    new_path: ArrayRef,
) -> Result<StructArray> {
    let mut columns = Vec::with_capacity(original.num_columns());
    for field in original.fields() {
        let column = match field.name().as_str() {
            "bytes" => new_bytes.clone(),
            "path" => new_path.clone(),
            name => {
                let source = original
                    .column_by_name(name)
                    .ok_or_else(|| anyhow!("missing audio field {}", name))?;
                take(source.as_ref(), indices, None)
                    .with_context(|| format!("take audio field {}", name))?
            }
        };
        columns.push(column);
    }

    let nulls = match original.nulls() {
        Some(_) => {
            let taken = take(original, indices, None).context("take audio struct nulls")?;
            taken.as_struct().nulls().cloned()
        }
        None => None,
    };

    Ok(StructArray::new(original.fields().clone(), columns, nulls))
}

fn build_binary_array(data_type: &DataType, values: &[Option<Vec<u8>>]) -> Result<ArrayRef> {
    let array: ArrayRef = match data_type {
        DataType::Binary => Arc::new(BinaryArray::from_iter(
            values.iter().map(|value| value.as_deref()),
        )),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from_iter(
            values.iter().map(|value| value.as_deref()),
        )),
        DataType::BinaryView => Arc::new(BinaryViewArray::from_iter(
            values.iter().map(|value| value.as_deref()),
        )),
        other => bail!("unsupported audio bytes type: {other:?}"),
    };
    Ok(array)
}

fn build_string_array(data_type: &DataType, values: &[Option<String>]) -> Result<ArrayRef> {
    let array: ArrayRef = match data_type {
        DataType::Utf8 => Arc::new(StringArray::from_iter(
            values.iter().map(|value| value.as_deref()),
        )),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from_iter(
            values.iter().map(|value| value.as_deref()),
        )),
        DataType::Utf8View => Arc::new(StringViewArray::from_iter(
            values.iter().map(|value| value.as_deref()),
        )),
        other => bail!("unsupported audio path type: {other:?}"),
    };
    Ok(array)
}

fn build_duration_array(data_type: &DataType, values: &[f64]) -> Result<ArrayRef> {
    let array: ArrayRef = match data_type {
        DataType::Float32 => Arc::new(Float32Array::from_iter_values(
            values.iter().copied().map(|value| value as f32),
        )),
        DataType::Float64 => Arc::new(Float64Array::from_iter_values(values.iter().copied())),
        other => bail!("unsupported duration type: {other:?}"),
    };
    Ok(array)
}

pub(crate) fn empty_like(batch: &RecordBatch) -> Result<RecordBatch> {
    let empty_indices = UInt32Array::from(Vec::<u32>::new());
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &empty_indices, None).context("build empty batch"))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(batch.schema(), columns).context("build empty output batch")
}

pub(crate) fn binary_value(array: &dyn Array, row: usize) -> Result<Option<&[u8]>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Binary => Ok(Some(array.as_binary::<i32>().value(row))),
        DataType::LargeBinary => Ok(Some(array.as_binary::<i64>().value(row))),
        DataType::BinaryView => Ok(Some(array.as_binary_view().value(row))),
        other => bail!("unsupported audio bytes type: {other:?}"),
    }
}

pub(crate) fn string_value(array: &dyn Array, row: usize) -> Result<Option<&str>> {
    if array.is_null(row) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Utf8 => Ok(Some(array.as_string::<i32>().value(row))),
        DataType::LargeUtf8 => Ok(Some(array.as_string::<i64>().value(row))),
        DataType::Utf8View => Ok(Some(array.as_string_view().value(row))),
        other => bail!("unsupported audio path type: {other:?}"),
    }
}
