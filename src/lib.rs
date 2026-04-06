pub mod analyze;
pub mod api;
pub mod audio;
pub mod cli;
pub mod parquet_io;
#[cfg(feature = "python")]
pub mod python;

pub use analyze::{
    AnalysisOptions, DecisionOptions, FrameProbability, RowDecisionMode, Segment, SegmentKind,
    music_score, row_is_music, segment, speech_ranges,
};
pub use api::{
    AudioAnalysis, AudioSegmentation, analyze_bytes, analyze_decoded, classify_bytes, decode,
    music_score_bytes, segment_bytes, strip_music_bytes,
};
pub use audio::{AudioFormat, DecodedAudio, WavEncoding, decode_audio, encode_audio, strip_music};
