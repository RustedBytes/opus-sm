# opus-sm

`opus-sm` is a Rust CLI for speech-vs-music analysis on Parquet datasets with embedded audio.
It uses the built-in speech/music discriminator from the open source Opus codec to compute
framewise music probabilities, emit speech/music segmentation, strip detected music regions,
and split Parquet rows into speech and music outputs.

## What It Does

- reads Parquet files containing embedded audio rows
- supports embedded audio stored as:
  - WAV bytes
  - Ogg Opus bytes
  - MP3 bytes
- runs the Opus speech/music discriminator frame by frame
- prints framewise music probabilities
- supports smoothed probabilities before segmentation
- supports hysteresis segmentation with separate high/low thresholds
- supports minimum run lengths for speech/music segments
- can remove music regions from each row and write updated Parquet output
- can split rows into separate speech and music Parquet files
- parallelizes row-level work within each Parquet batch using Rayon

## Expected Parquet Layout

The tool follows the same Arrow/Parquet access pattern as `opusify` and expects an `audio`
struct column containing at least:

- `audio.bytes`
- `audio.path`

Other fields in the schema are preserved when writing output.

For `strip-music`, the output Parquet keeps the same schema and rewrites only:

- `audio.bytes`
- `audio.path`

For `separate-sm`, the speech and music output files keep the same schema and row structure as
the input, with rows routed by the configured row-decision rule.

## Build

Requirements:

- Rust toolchain
- C toolchain for building the small Opus analysis shim

Build:

```bash
cargo build --release
```

## Usage

Top-level help:

```bash
cargo run -- --help
```

## Commands

### Analyze

Print framewise probabilities for every audio row.

```bash
cargo run -- analyze \
  --input data/input.parquet
```

Directory input:

```bash
cargo run -- analyze \
  --input data/parquet-dir \
  --smooth-window 5
```

Output format:

- `FRAME <file> <row> <frame_index> <start_seconds> <end_seconds> <music_probability> <activity_probability>`

### Segment

Print framewise probabilities and thresholded speech/music segments.

```bash
cargo run -- segment \
  --input data/parquet-dir \
  --threshold 0.6 \
  --smooth-window 5 \
  --low-threshold 0.45 \
  --high-threshold 0.65 \
  --min-music-frames 3 \
  --min-speech-frames 3
```

Additional output format:

- `SEGMENT <speech|music> <start_frame> <end_frame> <start_seconds> <end_seconds>`

### Strip Music

Remove music segments from each row and write a new Parquet file or tree with the same schema.

```bash
cargo run -- strip-music \
  --input data/parquet-dir \
  --output data/speech-only \
  --threshold 0.6 \
  --low-threshold 0.45 \
  --high-threshold 0.65 \
  --smooth-window 5 \
  --fade-ms 15
```

Behavior:

- rows are preserved
- only detected music spans are removed from the audio payload
- optional fades can be applied at strip boundaries to reduce clicks
- output audio is written back in the same container family as the input:
  - WAV in, WAV out
  - Ogg Opus in, Ogg Opus out
  - MP3 in, WAV out
- WAV output preserves the original WAV sample format and bit depth when supported

### VAD

Segment each row into speech-only chunks and emit one output row per chunk.

```bash
cargo run -- vad \
  --input data/parquet-dir \
  --output data/vad-chunks \
  --threshold 0.8 \
  --min-speech-seconds 0.5 \
  --max-speech-seconds 30.0
```

Behavior:

- one output row is written for each detected speech chunk
- `audio.bytes` is replaced with the chunk audio payload
- `audio.path` is rewritten as `..._chunkN.ext`
- chunk `duration` is recomputed when a top-level `duration` column exists
- top-level `transcription` is replaced with `"-"` when that column exists
- for MP3 input rows, chunk audio is written as WAV and chunk paths end in `.wav`

### Separate Speech/Music

Split rows into two Parquet outputs based on a configurable row-level decision rule.

Max-score routing:

```bash
cargo run -- separate-sm \
  --input data/parquet-dir \
  --speech-output data/speech \
  --music-output data/music \
  --threshold 0.6 \
  --row-decision max
```

Fraction-based routing:

```bash
cargo run -- separate-sm \
  --input data/parquet-dir \
  --speech-output data/speech \
  --music-output data/music \
  --threshold 0.6 \
  --row-decision fraction \
  --row-fraction 0.25
```

Supported row-decision modes:

- `max`
- `mean`
- `median`
- `fraction`

For `fraction`, the row is classified as music when the fraction of frames with
`music_probability >= threshold` is at least `row-fraction`.

## Library API

`opus-sm` can also be used as a Rust library.

Main exported convenience functions:

- `decode`
- `analyze_bytes`
- `analyze_decoded`
- `segment_bytes`
- `classify_bytes`
- `music_score_bytes`
- `strip_music_bytes`
- `vad_chunks_bytes`

Main exported analysis types:

- `AnalysisOptions`
- `DecisionOptions`
- `FrameProbability`
- `Segment`
- `SegmentKind`
- `RowDecisionMode`
- `DecodedAudio`

Minimal example:

```rust
use anyhow::Result;
use opus_sm::{AnalysisOptions, DecisionOptions, RowDecisionMode, analyze_bytes, segment_bytes};

fn run(audio_bytes: &[u8]) -> Result<()> {
    let analysis = analyze_bytes(audio_bytes, AnalysisOptions { smooth_window: 5 })?;

    let segmented = segment_bytes(
        audio_bytes,
        AnalysisOptions { smooth_window: 5 },
        DecisionOptions {
            threshold: 0.6,
            low_threshold: 0.45,
            high_threshold: 0.65,
            min_music_frames: 3,
            min_speech_frames: 3,
            row_decision: RowDecisionMode::Max,
            row_fraction: 0.5,
        },
    )?;

    println!("frames: {}", analysis.probabilities.len());
    println!("segments: {}", segmented.segments.len());
    Ok(())
}
```

## CLI Reference

### Shared Input Flags

- `--input <PATH>`: input Parquet file or directory
- `--batch-size <N>`: Arrow batch size for the Parquet reader, default `256`

### Analysis Flags

- `--smooth-window <N>`: moving-average smoothing window over frame probabilities, default `1`

### Decision Flags

- `--threshold <FLOAT>`: base probability threshold in `[0.0, 1.0]`
- `--high-threshold <FLOAT>`: hysteresis switch-to-music threshold
- `--low-threshold <FLOAT>`: hysteresis switch-to-speech threshold
- `--min-music-frames <N>`: suppress short music runs shorter than `N` frames
- `--min-speech-frames <N>`: suppress short speech runs shorter than `N` frames
- `--row-decision <MODE>`: row-level classifier for `separate-sm`, one of `max`, `mean`, `median`, `fraction`
- `--row-fraction <FLOAT>`: fraction cutoff for `row-decision=fraction`, default `0.5`

### `strip-music`

- `--output <PATH>`: output file or root directory
- `--fade-ms <FLOAT>`: fade-in/fade-out duration applied at strip boundaries

### `separate-sm`

- `--speech-output <PATH>`: destination file or root directory for speech rows
- `--music-output <PATH>`: destination file or root directory for music rows

### `vad`

- `--output <PATH>`: output file or root directory for chunk rows
- `--fade-ms <FLOAT>`: fade-in/fade-out duration applied at chunk boundaries
- `--min-speech-seconds <FLOAT>`: drop detected speech chunks shorter than this
- `--max-speech-seconds <FLOAT>`: split long speech chunks to at most this duration

## Implementation Notes

- frame analysis uses Opus’s internal `music_prob` analysis path via a small C shim
- analysis runs on 20 ms frames at 48 kHz
- non-48 kHz input is resampled before analysis
- multichannel inputs above stereo are downmixed to mono for analysis
- row-level decode/analyze/transform work is parallelized per batch using Rayon
- output ordering remains deterministic and follows original row order

## Audio Handling Notes

- Ogg Opus decode uses `pre_skip` and also trims the decoded tail using the final granule position
- Ogg Opus output writes final-page granule positions based on the true output sample count
- MP3 decode uses `symphonia`
- MP3 input is analyzed directly, but rewritten audio is emitted as WAV because the tool does not encode MP3 output
- WAV output preserves the original WAV container format where supported:
  - 32-bit float
  - 8-bit PCM
  - 16-bit PCM
  - 24-bit PCM
  - 32-bit PCM

## Current Limits

- Ogg Opus decode/encode currently supports mono and stereo only
- audio above stereo is analyzed by mono downmix rather than per-channel classification
- MP3 input is supported for decoding only; rewritten MP3 rows are emitted as WAV

## Verification

Checked with:

```bash
cargo check
cargo run -- --help
cargo run -- segment --help
cargo run -- strip-music --help
cargo test vad_accepts_mp3_rows_from_radio_free_dataset -- --nocapture
```
