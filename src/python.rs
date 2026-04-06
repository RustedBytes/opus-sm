#![cfg(feature = "python")]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::analyze::{AnalysisOptions, DecisionOptions, RowDecisionMode};
use crate::api;
use crate::{AudioFormat, ChunkOutputFormat, SegmentKind};

#[pyclass(module = "opus_sm", get_all)]
#[derive(Clone)]
struct AudioInfo {
    sample_rate: u32,
    channels: usize,
    format: String,
    num_samples: usize,
    num_frames: usize,
}

#[pymethods]
impl AudioInfo {
    fn __repr__(&self) -> String {
        format!(
            "AudioInfo(sample_rate={}, channels={}, format={:?}, num_samples={}, num_frames={})",
            self.sample_rate, self.channels, self.format, self.num_samples, self.num_frames
        )
    }
}

#[pyclass(module = "opus_sm", get_all)]
#[derive(Clone)]
struct FrameProbability {
    index: usize,
    start_seconds: f64,
    end_seconds: f64,
    music_probability: f32,
    activity_probability: f32,
}

#[pymethods]
impl FrameProbability {
    fn __repr__(&self) -> String {
        format!(
            "FrameProbability(index={}, start_seconds={:?}, end_seconds={:?}, music_probability={:?}, activity_probability={:?})",
            self.index,
            self.start_seconds,
            self.end_seconds,
            self.music_probability,
            self.activity_probability
        )
    }
}

#[pyclass(module = "opus_sm", get_all)]
#[derive(Clone)]
struct Segment {
    kind: String,
    start_frame: usize,
    end_frame: usize,
    start_seconds: f64,
    end_seconds: f64,
}

#[pymethods]
impl Segment {
    fn __repr__(&self) -> String {
        format!(
            "Segment(kind={:?}, start_frame={}, end_frame={}, start_seconds={:?}, end_seconds={:?})",
            self.kind, self.start_frame, self.end_frame, self.start_seconds, self.end_seconds
        )
    }
}

#[pyclass(module = "opus_sm", get_all)]
#[derive(Clone)]
struct AnalysisResult {
    sample_rate: u32,
    channels: usize,
    format: String,
    probabilities: Vec<FrameProbability>,
}

#[pymethods]
impl AnalysisResult {
    fn __repr__(&self) -> String {
        format!(
            "AnalysisResult(sample_rate={}, channels={}, format={:?}, probabilities=<{} frames>)",
            self.sample_rate,
            self.channels,
            self.format,
            self.probabilities.len()
        )
    }
}

#[pyclass(module = "opus_sm", get_all)]
#[derive(Clone)]
struct SegmentationResult {
    analysis: AnalysisResult,
    segments: Vec<Segment>,
}

#[pymethods]
impl SegmentationResult {
    fn __repr__(&self) -> String {
        format!(
            "SegmentationResult(analysis=<{} frames>, segments=<{} segments>)",
            self.analysis.probabilities.len(),
            self.segments.len()
        )
    }
}

#[pyclass(module = "opus_sm", get_all)]
#[derive(Clone)]
struct VadChunk {
    index: usize,
    start_seconds: f64,
    end_seconds: f64,
    duration_seconds: f64,
    path: String,
}

#[pymethods]
impl VadChunk {
    fn __repr__(&self) -> String {
        format!(
            "VadChunk(index={}, start_seconds={:?}, end_seconds={:?}, duration_seconds={:?}, path={:?})",
            self.index, self.start_seconds, self.end_seconds, self.duration_seconds, self.path
        )
    }
}

#[pyfunction(name = "analyze_bytes", signature = (audio_bytes, smooth_window=1))]
fn analyze_bytes_py(audio_bytes: &[u8], smooth_window: usize) -> PyResult<AnalysisResult> {
    let analysis =
        api::analyze_bytes(audio_bytes, AnalysisOptions { smooth_window }).map_err(to_py_err)?;
    Ok(to_analysis_result(&analysis))
}

#[pyfunction(
    name = "segment_bytes",
    signature = (
        audio_bytes,
        threshold,
        smooth_window=1,
        high_threshold=None,
        low_threshold=None,
        min_music_frames=0,
        min_speech_frames=0,
        row_decision="max",
        row_fraction=0.5
    )
)]
fn segment_bytes_py(
    audio_bytes: &[u8],
    threshold: f32,
    smooth_window: usize,
    high_threshold: Option<f32>,
    low_threshold: Option<f32>,
    min_music_frames: usize,
    min_speech_frames: usize,
    row_decision: &str,
    row_fraction: f32,
) -> PyResult<SegmentationResult> {
    let segmentation = api::segment_bytes(
        audio_bytes,
        AnalysisOptions { smooth_window },
        decision_options(
            threshold,
            high_threshold,
            low_threshold,
            min_music_frames,
            min_speech_frames,
            row_decision,
            row_fraction,
        )?,
    )
    .map_err(to_py_err)?;

    Ok(SegmentationResult {
        analysis: to_analysis_result(&segmentation.analysis),
        segments: segmentation
            .segments
            .iter()
            .map(|segment| Segment {
                kind: if segment.kind == SegmentKind::Music {
                    "music".to_owned()
                } else {
                    "speech".to_owned()
                },
                start_frame: segment.start_frame,
                end_frame: segment.end_frame,
                start_seconds: segment.start_seconds,
                end_seconds: segment.end_seconds,
            })
            .collect(),
    })
}

#[pyfunction(
    name = "classify_bytes",
    signature = (
        audio_bytes,
        threshold,
        smooth_window=1,
        high_threshold=None,
        low_threshold=None,
        min_music_frames=0,
        min_speech_frames=0,
        row_decision="max",
        row_fraction=0.5
    )
)]
fn classify_bytes_py(
    audio_bytes: &[u8],
    threshold: f32,
    smooth_window: usize,
    high_threshold: Option<f32>,
    low_threshold: Option<f32>,
    min_music_frames: usize,
    min_speech_frames: usize,
    row_decision: &str,
    row_fraction: f32,
) -> PyResult<bool> {
    api::classify_bytes(
        audio_bytes,
        AnalysisOptions { smooth_window },
        decision_options(
            threshold,
            high_threshold,
            low_threshold,
            min_music_frames,
            min_speech_frames,
            row_decision,
            row_fraction,
        )?,
    )
    .map_err(to_py_err)
}

#[pyfunction(
    name = "music_score_bytes",
    signature = (
        audio_bytes,
        threshold,
        smooth_window=1,
        high_threshold=None,
        low_threshold=None,
        min_music_frames=0,
        min_speech_frames=0,
        row_decision="max",
        row_fraction=0.5
    )
)]
fn music_score_bytes_py(
    audio_bytes: &[u8],
    threshold: f32,
    smooth_window: usize,
    high_threshold: Option<f32>,
    low_threshold: Option<f32>,
    min_music_frames: usize,
    min_speech_frames: usize,
    row_decision: &str,
    row_fraction: f32,
) -> PyResult<f32> {
    api::music_score_bytes(
        audio_bytes,
        AnalysisOptions { smooth_window },
        decision_options(
            threshold,
            high_threshold,
            low_threshold,
            min_music_frames,
            min_speech_frames,
            row_decision,
            row_fraction,
        )?,
    )
    .map_err(to_py_err)
}

#[pyfunction(
    name = "strip_music_bytes",
    signature = (
        audio_bytes,
        threshold,
        smooth_window=1,
        high_threshold=None,
        low_threshold=None,
        min_music_frames=0,
        min_speech_frames=0,
        row_decision="max",
        row_fraction=0.5,
        fade_ms=0.0
    )
)]
fn strip_music_bytes_py<'py>(
    py: Python<'py>,
    audio_bytes: &[u8],
    threshold: f32,
    smooth_window: usize,
    high_threshold: Option<f32>,
    low_threshold: Option<f32>,
    min_music_frames: usize,
    min_speech_frames: usize,
    row_decision: &str,
    row_fraction: f32,
    fade_ms: f32,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = api::strip_music_bytes(
        audio_bytes,
        AnalysisOptions { smooth_window },
        decision_options(
            threshold,
            high_threshold,
            low_threshold,
            min_music_frames,
            min_speech_frames,
            row_decision,
            row_fraction,
        )?,
        fade_ms,
    )
    .map_err(to_py_err)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction(
    name = "vad_bytes",
    signature = (
        audio_bytes,
        threshold,
        smooth_window=1,
        high_threshold=None,
        low_threshold=None,
        min_music_frames=0,
        min_speech_frames=0,
        row_decision="max",
        row_fraction=0.5,
        fade_ms=0.0,
        min_speech_seconds=None,
        max_speech_seconds=None,
        chunk_format="auto",
        source_path=None
    )
)]
fn vad_bytes_py(
    audio_bytes: &[u8],
    threshold: f32,
    smooth_window: usize,
    high_threshold: Option<f32>,
    low_threshold: Option<f32>,
    min_music_frames: usize,
    min_speech_frames: usize,
    row_decision: &str,
    row_fraction: f32,
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
    chunk_format: &str,
    source_path: Option<&str>,
) -> PyResult<Vec<VadChunk>> {
    let chunks = api::vad_chunks_bytes(
        audio_bytes,
        AnalysisOptions { smooth_window },
        decision_options(
            threshold,
            high_threshold,
            low_threshold,
            min_music_frames,
            min_speech_frames,
            row_decision,
            row_fraction,
        )?,
        fade_ms,
        min_speech_seconds,
        max_speech_seconds,
        parse_chunk_format(chunk_format)?,
        source_path,
    )
    .map_err(to_py_err)?;

    Ok(chunks
        .into_iter()
        .map(|chunk| VadChunk {
            index: chunk.index,
            start_seconds: chunk.start_seconds,
            end_seconds: chunk.end_seconds,
            duration_seconds: chunk.duration_seconds,
            path: chunk.path,
        })
        .collect())
}

#[pyfunction(name = "decode_info")]
fn decode_info_py(audio_bytes: &[u8]) -> PyResult<AudioInfo> {
    let decoded = api::decode(audio_bytes).map_err(to_py_err)?;
    Ok(AudioInfo {
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
        format: format_name(decoded.format),
        num_samples: decoded.samples.len(),
        num_frames: decoded.samples.len() / decoded.channels.max(1),
    })
}

#[pymodule]
fn opus_sm(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AudioInfo>()?;
    m.add_class::<FrameProbability>()?;
    m.add_class::<Segment>()?;
    m.add_class::<AnalysisResult>()?;
    m.add_class::<SegmentationResult>()?;
    m.add_class::<VadChunk>()?;

    m.add_function(wrap_pyfunction!(analyze_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(segment_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(classify_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(music_score_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(strip_music_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(vad_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_info_py, m)?)?;

    m.add("ROW_DECISION_MAX", "max")?;
    m.add("ROW_DECISION_MEAN", "mean")?;
    m.add("ROW_DECISION_MEDIAN", "median")?;
    m.add("ROW_DECISION_FRACTION", "fraction")?;
    m.add("CHUNK_FORMAT_AUTO", "auto")?;
    m.add("CHUNK_FORMAT_WAV", "wav")?;
    m.add("CHUNK_FORMAT_OGG_OPUS", "ogg-opus")?;
    Ok(())
}

fn to_analysis_result(analysis: &api::AudioAnalysis) -> AnalysisResult {
    AnalysisResult {
        sample_rate: analysis.decoded.sample_rate,
        channels: analysis.decoded.channels,
        format: format_name(analysis.decoded.format),
        probabilities: analysis
            .probabilities
            .iter()
            .map(|frame| FrameProbability {
                index: frame.index,
                start_seconds: frame.start_seconds,
                end_seconds: frame.end_seconds,
                music_probability: frame.music_probability,
                activity_probability: frame.activity_probability,
            })
            .collect(),
    }
}

fn format_name(format: AudioFormat) -> String {
    match format {
        AudioFormat::Wav(_) => "wav".to_owned(),
        AudioFormat::OggOpus => "ogg_opus".to_owned(),
        AudioFormat::Mp3 => "mp3".to_owned(),
    }
}

fn decision_options(
    threshold: f32,
    high_threshold: Option<f32>,
    low_threshold: Option<f32>,
    min_music_frames: usize,
    min_speech_frames: usize,
    row_decision: &str,
    row_fraction: f32,
) -> PyResult<DecisionOptions> {
    validate_threshold(threshold)?;
    let high_threshold = high_threshold.unwrap_or(threshold);
    let low_threshold = low_threshold.unwrap_or(threshold);
    validate_threshold(high_threshold)?;
    validate_threshold(low_threshold)?;
    if low_threshold > high_threshold {
        return Err(PyValueError::new_err(
            "low_threshold must be less than or equal to high_threshold",
        ));
    }
    if !(0.0..=1.0).contains(&row_fraction) {
        return Err(PyValueError::new_err(
            "row_fraction must be between 0.0 and 1.0",
        ));
    }

    Ok(DecisionOptions {
        threshold,
        low_threshold,
        high_threshold,
        min_music_frames,
        min_speech_frames,
        row_decision: parse_row_decision(row_decision)?,
        row_fraction,
    })
}

fn parse_row_decision(value: &str) -> PyResult<RowDecisionMode> {
    match value {
        "max" => Ok(RowDecisionMode::Max),
        "mean" => Ok(RowDecisionMode::Mean),
        "median" => Ok(RowDecisionMode::Median),
        "fraction" => Ok(RowDecisionMode::Fraction),
        _ => Err(PyValueError::new_err(
            "row_decision must be one of: max, mean, median, fraction",
        )),
    }
}

fn parse_chunk_format(value: &str) -> PyResult<ChunkOutputFormat> {
    match value {
        "auto" => Ok(ChunkOutputFormat::Auto),
        "wav" => Ok(ChunkOutputFormat::Wav),
        "ogg-opus" => Ok(ChunkOutputFormat::OggOpus),
        _ => Err(PyValueError::new_err(
            "chunk_format must be one of: auto, wav, ogg-opus",
        )),
    }
}

fn validate_threshold(threshold: f32) -> PyResult<()> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err(
            "threshold must be between 0.0 and 1.0",
        ));
    }
    Ok(())
}

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
