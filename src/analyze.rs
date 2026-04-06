use std::ffi::c_void;

use anyhow::{Result, anyhow, bail};
use clap::ValueEnum;

const FRAME_SAMPLES: usize = 960;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RowDecisionMode {
    Max,
    Mean,
    Median,
    Fraction,
}

#[derive(Debug, Clone, Copy)]
pub struct FrameProbability {
    pub index: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub music_probability: f32,
    pub activity_probability: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Speech,
    Music,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub kind: SegmentKind,
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AnalysisOptions {
    pub smooth_window: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DecisionOptions {
    pub threshold: f32,
    pub low_threshold: f32,
    pub high_threshold: f32,
    pub min_music_frames: usize,
    pub min_speech_frames: usize,
    pub row_decision: RowDecisionMode,
    pub row_fraction: f32,
}

pub fn analyze_interleaved(
    samples: &[f32],
    channels: usize,
    sample_rate: u32,
    options: AnalysisOptions,
) -> Result<Vec<FrameProbability>> {
    if sample_rate != 48_000 {
        bail!("analyzer expects 48 kHz input, got {}", sample_rate);
    }
    if channels == 0 {
        bail!("channel count must be positive");
    }

    let (analysis_samples, analysis_channels) = if channels <= 2 {
        (samples.to_vec(), channels)
    } else {
        (downmix_to_mono(samples, channels), 1)
    };

    let frame_stride = FRAME_SAMPLES * analysis_channels;
    let frame_count = analysis_samples.len().div_ceil(frame_stride);
    let mut probabilities = Vec::with_capacity(frame_count);

    let analyzer = Analyzer::new(48_000)?;
    for frame_index in 0..frame_count {
        let start = frame_index * frame_stride;
        let available = analysis_samples
            .len()
            .saturating_sub(start)
            .min(frame_stride);
        let mut frame = vec![0.0_f32; frame_stride];
        if available > 0 {
            frame[..available].copy_from_slice(&analysis_samples[start..start + available]);
        }

        let (music_probability, activity_probability) =
            analyzer.process(&frame, FRAME_SAMPLES as i32, analysis_channels as i32)?;
        probabilities.push(FrameProbability {
            index: frame_index,
            start_seconds: start as f64 / (analysis_channels as f64 * 48_000.0),
            end_seconds: ((start / analysis_channels) + FRAME_SAMPLES) as f64 / 48_000.0,
            music_probability,
            activity_probability,
        });
    }

    if options.smooth_window > 1 {
        smooth_probabilities(&mut probabilities, options.smooth_window);
    }

    Ok(probabilities)
}

pub fn segment(probabilities: &[FrameProbability], options: DecisionOptions) -> Vec<Segment> {
    if probabilities.is_empty() {
        return Vec::new();
    }

    let labels = classify_frames(probabilities, options);
    build_segments(probabilities, &labels)
}

pub fn music_score(probabilities: &[FrameProbability], options: DecisionOptions) -> f32 {
    if probabilities.is_empty() {
        return 0.0;
    }

    let mut values = probabilities
        .iter()
        .map(|frame| frame.music_probability)
        .collect::<Vec<_>>();
    match options.row_decision {
        RowDecisionMode::Max => values.iter().copied().fold(0.0_f32, f32::max),
        RowDecisionMode::Mean => values.iter().copied().sum::<f32>() / values.len() as f32,
        RowDecisionMode::Median => {
            values.sort_by(|a, b| a.total_cmp(b));
            values[values.len() / 2]
        }
        RowDecisionMode::Fraction => {
            let count = values
                .iter()
                .filter(|&&value| value >= options.threshold)
                .count();
            count as f32 / values.len() as f32
        }
    }
}

pub fn row_is_music(probabilities: &[FrameProbability], options: DecisionOptions) -> bool {
    let score = music_score(probabilities, options);
    match options.row_decision {
        RowDecisionMode::Fraction => score >= options.row_fraction,
        _ => score >= options.threshold,
    }
}

pub fn speech_ranges(
    probabilities: &[FrameProbability],
    options: DecisionOptions,
) -> Vec<(f64, f64)> {
    segment(probabilities, options)
        .into_iter()
        .filter(|segment| segment.kind == SegmentKind::Speech)
        .map(|segment| (segment.start_seconds, segment.end_seconds))
        .collect()
}

fn classify_frames(
    probabilities: &[FrameProbability],
    options: DecisionOptions,
) -> Vec<SegmentKind> {
    let mut labels = Vec::with_capacity(probabilities.len());
    let mut current = if probabilities[0].music_probability >= options.high_threshold {
        SegmentKind::Music
    } else {
        SegmentKind::Speech
    };
    labels.push(current);

    for frame in probabilities.iter().skip(1) {
        current = match current {
            SegmentKind::Speech if frame.music_probability >= options.high_threshold => {
                SegmentKind::Music
            }
            SegmentKind::Music if frame.music_probability <= options.low_threshold => {
                SegmentKind::Speech
            }
            _ => current,
        };
        labels.push(current);
    }

    if options.min_music_frames > 0 || options.min_speech_frames > 0 {
        merge_short_runs(&mut labels, options);
    }

    labels
}

fn merge_short_runs(labels: &mut [SegmentKind], options: DecisionOptions) {
    if labels.is_empty() {
        return;
    }

    let mut start = 0usize;
    while start < labels.len() {
        let current = labels[start];
        let mut end = start + 1;
        while end < labels.len() && labels[end] == current {
            end += 1;
        }

        let len = end - start;
        let min_len = match current {
            SegmentKind::Music => options.min_music_frames,
            SegmentKind::Speech => options.min_speech_frames,
        };

        if min_len > 0 && len < min_len {
            let replacement = if start == 0 {
                if end < labels.len() {
                    labels[end]
                } else {
                    current
                }
            } else {
                labels[start - 1]
            };
            labels[start..end].fill(replacement);
        }

        start = end;
    }
}

fn build_segments(probabilities: &[FrameProbability], labels: &[SegmentKind]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut current = labels[0];

    for index in 1..labels.len() {
        if labels[index] != current {
            segments.push(Segment {
                kind: current,
                start_frame: start,
                end_frame: index,
                start_seconds: probabilities[start].start_seconds,
                end_seconds: probabilities[index - 1].end_seconds,
            });
            start = index;
            current = labels[index];
        }
    }

    segments.push(Segment {
        kind: current,
        start_frame: start,
        end_frame: labels.len(),
        start_seconds: probabilities[start].start_seconds,
        end_seconds: probabilities[labels.len() - 1].end_seconds,
    });
    segments
}

fn smooth_probabilities(probabilities: &mut [FrameProbability], window: usize) {
    let radius = window / 2;
    let original = probabilities.to_vec();
    for (index, frame) in probabilities.iter_mut().enumerate() {
        let start = index.saturating_sub(radius);
        let end = (index + radius + 1).min(original.len());
        let slice = &original[start..end];
        let sum_music = slice.iter().map(|item| item.music_probability).sum::<f32>();
        let sum_activity = slice
            .iter()
            .map(|item| item.activity_probability)
            .sum::<f32>();
        frame.music_probability = sum_music / slice.len() as f32;
        frame.activity_probability = sum_activity / slice.len() as f32;
    }
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

struct Analyzer {
    raw: *mut c_void,
}

impl Analyzer {
    fn new(sample_rate: i32) -> Result<Self> {
        let raw = unsafe { opus_sm_analyzer_create(sample_rate) };
        if raw.is_null() {
            return Err(anyhow!("failed to create Opus speech/music analyzer"));
        }
        Ok(Self { raw })
    }

    fn process(&self, pcm: &[f32], frame_size: i32, channels: i32) -> Result<(f32, f32)> {
        let mut music_probability = 0.0_f32;
        let mut activity_probability = 0.0_f32;
        let _valid = unsafe {
            opus_sm_analyzer_process(
                self.raw,
                pcm.as_ptr(),
                frame_size,
                channels,
                &mut music_probability,
                &mut activity_probability,
            )
        };
        Ok((music_probability, activity_probability))
    }
}

impl Drop for Analyzer {
    fn drop(&mut self) {
        unsafe { opus_sm_analyzer_destroy(self.raw) };
    }
}

unsafe extern "C" {
    fn opus_sm_analyzer_create(sample_rate: i32) -> *mut c_void;
    fn opus_sm_analyzer_destroy(analyzer: *mut c_void);
    fn opus_sm_analyzer_process(
        analyzer: *mut c_void,
        pcm: *const f32,
        frame_size: i32,
        channels: i32,
        music_prob: *mut f32,
        activity_prob: *mut f32,
    ) -> i32;
}
