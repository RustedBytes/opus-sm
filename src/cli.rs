use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::analyze::RowDecisionMode;
use crate::audio::ChunkOutputFormat;

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Analyze(AnalyzeArgs),
    Segment(SegmentArgs),
    StripMusic(StripMusicArgs),
    SeparateSm(SeparateSmArgs),
    Vad(VadArgs),
}

#[derive(Debug, Clone, Args)]
pub struct InputArgs {
    #[arg(long)]
    pub input: PathBuf,

    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
}

#[derive(Debug, Clone, Args)]
pub struct AnalysisArgs {
    #[arg(long, default_value_t = 1)]
    pub smooth_window: usize,
}

#[derive(Debug, Clone, Args)]
pub struct DecisionArgs {
    #[arg(long)]
    pub threshold: f32,

    #[arg(long)]
    pub high_threshold: Option<f32>,

    #[arg(long)]
    pub low_threshold: Option<f32>,

    #[arg(long, default_value_t = 0)]
    pub min_music_frames: usize,

    #[arg(long, default_value_t = 0)]
    pub min_speech_frames: usize,

    #[arg(long, value_enum, default_value_t = RowDecisionMode::Max)]
    pub row_decision: RowDecisionMode,

    #[arg(long, default_value_t = 0.5)]
    pub row_fraction: f32,
}

#[derive(Debug, Clone, Args)]
pub struct StripArgs {
    #[arg(long, default_value_t = 0.0)]
    pub fade_ms: f32,
}

#[derive(Debug, Clone, Args)]
pub struct AnalyzeArgs {
    #[command(flatten)]
    pub input: InputArgs,

    #[command(flatten)]
    pub analysis: AnalysisArgs,
}

#[derive(Debug, Clone, Args)]
pub struct SegmentArgs {
    #[command(flatten)]
    pub input: InputArgs,

    #[command(flatten)]
    pub analysis: AnalysisArgs,

    #[command(flatten)]
    pub decision: DecisionArgs,
}

#[derive(Debug, Clone, Args)]
pub struct StripMusicArgs {
    #[command(flatten)]
    pub input: InputArgs,

    #[arg(long)]
    pub output: PathBuf,

    #[command(flatten)]
    pub analysis: AnalysisArgs,

    #[command(flatten)]
    pub decision: DecisionArgs,

    #[command(flatten)]
    pub strip: StripArgs,
}

#[derive(Debug, Clone, Args)]
pub struct SeparateSmArgs {
    #[command(flatten)]
    pub input: InputArgs,

    #[arg(long)]
    pub speech_output: PathBuf,

    #[arg(long)]
    pub music_output: PathBuf,

    #[command(flatten)]
    pub analysis: AnalysisArgs,

    #[command(flatten)]
    pub decision: DecisionArgs,
}

#[derive(Debug, Clone, Args)]
pub struct VadArgs {
    #[command(flatten)]
    pub input: InputArgs,

    #[arg(long)]
    pub output: PathBuf,

    #[command(flatten)]
    pub analysis: AnalysisArgs,

    #[command(flatten)]
    pub decision: DecisionArgs,

    #[command(flatten)]
    pub strip: StripArgs,

    #[arg(long)]
    pub min_speech_seconds: Option<f32>,

    #[arg(long)]
    pub max_speech_seconds: Option<f32>,

    #[arg(long, value_enum, default_value_t = ChunkOutputFormat::Auto)]
    pub chunk_format: ChunkOutputFormat,
}
