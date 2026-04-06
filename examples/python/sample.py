#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import opus_sm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example usage of the opus_sm Python bindings."
    )
    parser.add_argument("audio", type=Path, help="Path to a WAV or Ogg Opus file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Music threshold in [0.0, 1.0]",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving average smoothing window",
    )
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=15.0,
        help="Fade duration for stripped output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("speech_only.out"),
        help="Path for stripped output bytes",
    )
    args = parser.parse_args()

    audio_bytes = args.audio.read_bytes()

    info = opus_sm.decode_info(audio_bytes)
    print("decode info:")
    print(info)
    print(info.sample_rate, info.channels, info.format)
    print()

    analysis = opus_sm.analyze_bytes(
        audio_bytes,
        smooth_window=args.smooth_window,
    )
    print(f"frame count: {len(analysis.probabilities)}")
    print("first 5 frames:")
    for frame in analysis.probabilities[:5]:
        print(frame)
    print()

    segmentation = opus_sm.segment_bytes(
        audio_bytes,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
        low_threshold=max(0.0, args.threshold - 0.15),
        high_threshold=min(1.0, args.threshold + 0.05),
        min_music_frames=3,
        min_speech_frames=3,
    )
    print(f"segment count: {len(segmentation.segments)}")
    print("segments:")
    for segment in segmentation.segments:
        print(segment)
    print()

    is_music = opus_sm.classify_bytes(
        audio_bytes,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
    )
    score = opus_sm.music_score_bytes(
        audio_bytes,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
    )
    print(f"classified as music: {is_music}")
    print(f"music score: {score:.6f}")
    print()

    stripped = opus_sm.strip_music_bytes(
        audio_bytes,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
        fade_ms=args.fade_ms,
    )
    args.output.write_bytes(bytes(stripped))
    print(f"wrote stripped output to {args.output}")


if __name__ == "__main__":
    main()
