use anyhow::Result;
use clap::Parser;

use opus_sm::cli::{Cli, Commands};
use opus_sm::parquet_io;

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze(args) => parquet_io::run_analyze(&args),
        Commands::Segment(args) => parquet_io::run_segment(&args),
        Commands::StripMusic(args) => parquet_io::run_strip_music(&args),
        Commands::SeparateSm(args) => parquet_io::run_separate_sm(&args),
        Commands::Vad(args) => parquet_io::run_vad(&args),
    }
}
