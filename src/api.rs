use anyhow::Result;

use crate::analyze::{
    AnalysisOptions, DecisionOptions, FrameProbability, Segment, music_score, row_is_music, segment,
};
use crate::audio::{
    ChunkOutputFormat, DecodedAudio, analyze_audio, decode_audio, encode_audio, strip_music,
};
use crate::vad::{VadChunk, vad_bytes};

#[derive(Debug, Clone)]
pub struct AudioAnalysis {
    pub decoded: DecodedAudio,
    pub probabilities: Vec<FrameProbability>,
}

#[derive(Debug, Clone)]
pub struct AudioSegmentation {
    pub analysis: AudioAnalysis,
    pub segments: Vec<Segment>,
}

pub fn decode(bytes: &[u8]) -> Result<DecodedAudio> {
    decode_audio(bytes)
}

pub fn analyze_bytes(bytes: &[u8], options: AnalysisOptions) -> Result<AudioAnalysis> {
    let decoded = decode_audio(bytes)?;
    let probabilities = analyze_audio(&decoded, options)?;
    Ok(AudioAnalysis {
        decoded,
        probabilities,
    })
}

pub fn analyze_decoded(
    audio: &DecodedAudio,
    options: AnalysisOptions,
) -> Result<Vec<FrameProbability>> {
    analyze_audio(audio, options)
}

pub fn segment_bytes(
    bytes: &[u8],
    analysis_options: AnalysisOptions,
    decision_options: DecisionOptions,
) -> Result<AudioSegmentation> {
    let analysis = analyze_bytes(bytes, analysis_options)?;
    let segments = segment(&analysis.probabilities, decision_options);
    Ok(AudioSegmentation { analysis, segments })
}

pub fn classify_bytes(
    bytes: &[u8],
    analysis_options: AnalysisOptions,
    decision_options: DecisionOptions,
) -> Result<bool> {
    let analysis = analyze_bytes(bytes, analysis_options)?;
    Ok(row_is_music(&analysis.probabilities, decision_options))
}

pub fn music_score_bytes(
    bytes: &[u8],
    analysis_options: AnalysisOptions,
    decision_options: DecisionOptions,
) -> Result<f32> {
    let analysis = analyze_bytes(bytes, analysis_options)?;
    Ok(music_score(&analysis.probabilities, decision_options))
}

pub fn strip_music_bytes(
    bytes: &[u8],
    analysis_options: AnalysisOptions,
    decision_options: DecisionOptions,
    fade_ms: f32,
) -> Result<Vec<u8>> {
    let decoded = decode_audio(bytes)?;
    let stripped = strip_music(&decoded, decision_options, analysis_options, fade_ms)?;
    encode_audio(&decoded, &stripped)
}

pub fn vad_chunks_bytes(
    bytes: &[u8],
    analysis_options: AnalysisOptions,
    decision_options: DecisionOptions,
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
    chunk_format: ChunkOutputFormat,
    source_path: Option<&str>,
) -> Result<Vec<VadChunk>> {
    vad_bytes(
        bytes,
        analysis_options,
        decision_options,
        fade_ms,
        min_speech_seconds,
        max_speech_seconds,
        chunk_format,
        source_path,
    )
}
