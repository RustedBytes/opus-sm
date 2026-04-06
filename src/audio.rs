use std::io::Cursor;

use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use ogg::PacketReader;
use opus::{Application, Channels, Decoder, Encoder, Signal as OpusSignal};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use crate::analyze::{
    AnalysisOptions, DecisionOptions, FrameProbability, analyze_interleaved, speech_ranges,
};

const OPUS_SAMPLE_RATE: u32 = 48_000;
const OPUS_FRAME_SAMPLES: usize = 960;
const MAX_OPUS_PACKET_SIZE: usize = 4_000;
const OGG_SERIAL: u32 = 0x4F50_5553;

#[derive(Debug, Clone, Copy)]
pub struct WavEncoding {
    pub spec: WavSpec,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioFormat {
    Wav(WavEncoding),
    OggOpus,
    Mp3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChunkOutputFormat {
    Auto,
    Wav,
    #[value(name = "ogg-opus")]
    OggOpus,
}

#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
    pub format: AudioFormat,
}

pub fn decode_audio(bytes: &[u8]) -> Result<DecodedAudio> {
    if bytes.starts_with(b"RIFF") {
        decode_wav(bytes)
    } else if bytes.starts_with(b"OggS") {
        decode_ogg_opus(bytes)
    } else if bytes.starts_with(b"ID3") || is_mpeg_audio_frame(bytes) {
        decode_mp3(bytes)
    } else {
        bail!("unsupported embedded audio format (expected RIFF/WAV, OggS/Opus, or MP3/ID3)");
    }
}

pub fn analyze_audio(
    audio: &DecodedAudio,
    options: AnalysisOptions,
) -> Result<Vec<FrameProbability>> {
    let analysis_samples = if audio.sample_rate == OPUS_SAMPLE_RATE {
        audio.samples.clone()
    } else {
        resample_linear_interleaved(
            &audio.samples,
            audio.channels,
            audio.sample_rate,
            OPUS_SAMPLE_RATE,
        )
    };
    analyze_interleaved(&analysis_samples, audio.channels, OPUS_SAMPLE_RATE, options)
}

pub fn strip_music(
    audio: &DecodedAudio,
    decision: DecisionOptions,
    analysis: AnalysisOptions,
    fade_ms: f32,
) -> Result<Vec<f32>> {
    let probabilities = analyze_audio(audio, analysis)?;
    let speech_ranges = speech_ranges(&probabilities, decision);
    if speech_ranges.is_empty() {
        return Ok(Vec::new());
    }

    let total_frames = audio.samples.len() / audio.channels;
    let fade_frames = ((fade_ms.max(0.0) / 1_000.0) * audio.sample_rate as f32).round() as usize;
    let mut stripped = Vec::new();

    for (start_seconds, end_seconds) in speech_ranges {
        let start_frame = (start_seconds * audio.sample_rate as f64).round() as usize;
        let end_frame = (end_seconds * audio.sample_rate as f64).round() as usize;
        let start_frame = start_frame.min(total_frames);
        let end_frame = end_frame.min(total_frames);
        if end_frame <= start_frame {
            continue;
        }

        let start = start_frame * audio.channels;
        let end = end_frame * audio.channels;
        let segment = &audio.samples[start..end];
        let boundary_left = start_frame > 0;
        let boundary_right = end_frame < total_frames;
        append_with_fades(
            &mut stripped,
            segment,
            audio.channels,
            fade_frames,
            boundary_left,
            boundary_right,
        );
    }

    Ok(stripped)
}

pub fn speech_chunks(
    audio: &DecodedAudio,
    decision: DecisionOptions,
    analysis: AnalysisOptions,
    fade_ms: f32,
    min_speech_seconds: Option<f32>,
    max_speech_seconds: Option<f32>,
) -> Result<Vec<Vec<f32>>> {
    let probabilities = analyze_audio(audio, analysis)?;
    let speech_ranges = speech_ranges(&probabilities, decision);
    if speech_ranges.is_empty() {
        return Ok(Vec::new());
    }

    let total_frames = audio.samples.len() / audio.channels;
    let fade_frames = ((fade_ms.max(0.0) / 1_000.0) * audio.sample_rate as f32).round() as usize;
    let min_speech_frames = min_speech_seconds
        .map(|seconds| seconds.max(0.0))
        .map(|seconds| (seconds * audio.sample_rate as f32).round() as usize)
        .unwrap_or(0);
    let max_speech_frames = max_speech_seconds
        .map(|seconds| seconds.max(0.0))
        .map(|seconds| (seconds * audio.sample_rate as f32).round() as usize)
        .filter(|frames| *frames > 0);
    let mut chunks = Vec::with_capacity(speech_ranges.len());

    for (start_seconds, end_seconds) in speech_ranges {
        let start_frame = (start_seconds * audio.sample_rate as f64).round() as usize;
        let end_frame = (end_seconds * audio.sample_rate as f64).round() as usize;
        let start_frame = start_frame.min(total_frames);
        let end_frame = end_frame.min(total_frames);
        if end_frame <= start_frame {
            continue;
        }

        let chunk_len = end_frame.saturating_sub(start_frame);
        if chunk_len < min_speech_frames {
            continue;
        }
        if let Some(limit) = max_speech_frames
            && chunk_len > limit
        {
            continue;
        }

        let start = start_frame * audio.channels;
        let end = end_frame * audio.channels;
        let segment = &audio.samples[start..end];
        let boundary_left = start_frame > 0;
        let boundary_right = end_frame < total_frames;
        let mut chunk = Vec::with_capacity(segment.len());
        append_with_fades(
            &mut chunk,
            segment,
            audio.channels,
            fade_frames,
            boundary_left,
            boundary_right,
        );
        chunks.push(chunk);
    }

    Ok(chunks)
}

pub fn encode_audio(audio: &DecodedAudio, samples: &[f32]) -> Result<Vec<u8>> {
    match audio.format {
        AudioFormat::Wav(wav) => encode_wav(samples, wav.spec),
        AudioFormat::OggOpus => encode_ogg_opus(samples, audio.channels),
        AudioFormat::Mp3 => encode_wav(
            samples,
            default_wav_spec(audio.sample_rate, audio.channels)?,
        ),
    }
}

pub fn rewrite_encoded_audio_path(path: &str, format: AudioFormat) -> String {
    let target_ext = match format {
        AudioFormat::Wav(_) | AudioFormat::Mp3 => "wav",
        AudioFormat::OggOpus => "opus",
    };
    rewrite_path_extension(path, target_ext)
}

pub fn encode_audio_with_format(
    audio: &DecodedAudio,
    samples: &[f32],
    output_format: ChunkOutputFormat,
) -> Result<Vec<u8>> {
    let prepared = prepare_chunk_output_samples(audio, samples, output_format)?;
    match resolve_chunk_audio_format(audio, output_format)? {
        AudioFormat::Wav(wav) => encode_wav(&prepared, wav.spec),
        AudioFormat::OggOpus => encode_ogg_opus(&prepared, audio.channels),
        AudioFormat::Mp3 => unreachable!("chunk output format does not resolve to MP3"),
    }
}

pub fn rewrite_chunk_output_path(
    path: &str,
    audio: &DecodedAudio,
    output_format: ChunkOutputFormat,
) -> Result<String> {
    Ok(rewrite_encoded_audio_path(
        path,
        resolve_chunk_audio_format(audio, output_format)?,
    ))
}

pub fn default_chunk_output_path(
    audio: &DecodedAudio,
    output_format: ChunkOutputFormat,
) -> Result<String> {
    let file_name = match resolve_chunk_audio_format(audio, output_format)? {
        AudioFormat::Wav(_) => "audio.wav",
        AudioFormat::OggOpus => "audio.opus",
        AudioFormat::Mp3 => unreachable!("chunk output format does not resolve to MP3"),
    };
    Ok(file_name.to_owned())
}

pub fn chunk_output_sample_rate(
    audio: &DecodedAudio,
    output_format: ChunkOutputFormat,
) -> Result<u32> {
    Ok(match resolve_chunk_audio_format(audio, output_format)? {
        AudioFormat::OggOpus => OPUS_SAMPLE_RATE,
        AudioFormat::Wav(_) => audio.sample_rate,
        AudioFormat::Mp3 => unreachable!("chunk output format does not resolve to MP3"),
    })
}

pub fn chunk_output_duration_seconds(
    audio: &DecodedAudio,
    samples: &[f32],
    output_format: ChunkOutputFormat,
) -> Result<f64> {
    let prepared = prepare_chunk_output_samples(audio, samples, output_format)?;
    let sample_rate = chunk_output_sample_rate(audio, output_format)?;
    let frames = prepared.len().div_ceil(audio.channels);
    Ok(frames as f64 / sample_rate as f64)
}

fn decode_wav(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut reader = WavReader::new(Cursor::new(bytes)).context("open wav payload")?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("wav has zero channels");
    }

    let mut samples = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                samples.push(sample.context("read wav float sample")?);
            }
        }
        SampleFormat::Int => {
            let scale = pcm_scale(spec.bits_per_sample)?;
            for sample in reader.samples::<i32>() {
                samples
                    .push((sample.context("read wav int sample")? as f32 / scale).clamp(-1.0, 1.0));
            }
        }
    }

    Ok(DecodedAudio {
        samples,
        sample_rate: spec.sample_rate,
        channels,
        format: AudioFormat::Wav(WavEncoding { spec }),
    })
}

fn decode_ogg_opus(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut reader = PacketReader::new(Cursor::new(bytes));
    let head_packet = reader
        .read_packet()
        .context("read opus head packet")?
        .ok_or_else(|| anyhow!("missing OpusHead packet"))?;
    let head = parse_opus_head(&head_packet.data)?;
    if head.channels == 0 || head.channels > 2 {
        bail!("only mono and stereo Ogg Opus are supported");
    }

    let _tags_packet = reader
        .read_packet()
        .context("read opus tags packet")?
        .ok_or_else(|| anyhow!("missing OpusTags packet"))?;

    let mut decoder = Decoder::new(
        OPUS_SAMPLE_RATE,
        if head.channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        },
    )
    .context("create opus decoder")?;

    let mut decoded = Vec::new();
    let mut final_granule = None;
    while let Some(packet) = reader.read_packet().context("read opus audio packet")? {
        if packet.data.is_empty() {
            continue;
        }
        if packet.last_in_stream() {
            final_granule = Some(packet.absgp_page());
        }

        let sample_count = decoder
            .get_nb_samples(&packet.data)
            .context("query opus packet sample count")?;
        let mut buffer = vec![0.0_f32; sample_count * head.channels as usize];
        let decoded_samples = decoder
            .decode_float(&packet.data, &mut buffer, false)
            .context("decode opus packet")?;
        decoded.extend_from_slice(&buffer[..decoded_samples * head.channels as usize]);
    }

    let skip = head.pre_skip as usize * head.channels as usize;
    if skip < decoded.len() {
        decoded.drain(..skip);
    } else {
        decoded.clear();
    }

    if let Some(granule) = final_granule {
        let expected_frames = granule.saturating_sub(head.pre_skip as u64) as usize;
        let expected_len = expected_frames.saturating_mul(head.channels as usize);
        if expected_len < decoded.len() {
            decoded.truncate(expected_len);
        }
    }

    Ok(DecodedAudio {
        samples: decoded,
        sample_rate: OPUS_SAMPLE_RATE,
        channels: head.channels as usize,
        format: AudioFormat::OggOpus,
    })
}

fn decode_mp3(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let source = Cursor::new(bytes.to_vec());
    let stream = MediaSourceStream::new(Box::new(source), MediaSourceStreamOptions::default());
    let mut probed = get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probe mp3 payload")?;
    let track = probed
        .format
        .default_track()
        .ok_or_else(|| anyhow!("missing default audio track"))?;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("create mp3 decoder")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("mp3 track is missing sample rate"))?;
    let channels = track
        .codec_params
        .channels
        .map(|value| value.count())
        .ok_or_else(|| anyhow!("mp3 track is missing channel count"))?;
    if channels == 0 {
        bail!("mp3 track has zero channels");
    }

    let mut samples = Vec::new();
    loop {
        let packet = match probed.format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("mp3 decoder reset is not supported");
            }
            Err(error) => return Err(error).context("read mp3 packet"),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                bail!("mp3 decoder reset is not supported");
            }
            Err(error) => return Err(error).context("decode mp3 packet"),
        };

        match decoded {
            AudioBufferRef::F32(buffer) => {
                append_planar_samples(
                    &mut samples,
                    buffer.spec().channels.count(),
                    buffer.frames(),
                    |channel, frame| *buffer.chan(channel).get(frame).unwrap_or(&0.0),
                );
            }
            other => {
                let spec = *other.spec();
                let duration = other.capacity() as u64;
                let mut sample_buffer = SampleBuffer::<f32>::new(duration, spec);
                sample_buffer.copy_interleaved_ref(other);
                samples.extend_from_slice(sample_buffer.samples());
            }
        }
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
        format: AudioFormat::Mp3,
    })
}

fn encode_wav(samples: &[f32], spec: WavSpec) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).context("create wav writer")?;
        match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => {
                for &sample in samples {
                    writer
                        .write_sample(sample.clamp(-1.0, 1.0))
                        .context("write wav float sample")?;
                }
            }
            (SampleFormat::Int, 8) => {
                for &sample in samples {
                    let pcm = (sample.clamp(-1.0, 1.0) * i8::MAX as f32).round() as i8;
                    writer.write_sample(pcm).context("write wav 8-bit sample")?;
                }
            }
            (SampleFormat::Int, 16) => {
                for &sample in samples {
                    let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
                    writer
                        .write_sample(pcm)
                        .context("write wav 16-bit sample")?;
                }
            }
            (SampleFormat::Int, 24) => {
                for &sample in samples {
                    let pcm = (sample.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
                    writer
                        .write_sample(pcm)
                        .context("write wav 24-bit sample")?;
                }
            }
            (SampleFormat::Int, 32) => {
                for &sample in samples {
                    let pcm = (sample.clamp(-1.0, 1.0) * i32::MAX as f32).round() as i32;
                    writer
                        .write_sample(pcm)
                        .context("write wav 32-bit sample")?;
                }
            }
            _ => bail!(
                "unsupported wav output format: {:?} {}",
                spec.sample_format,
                spec.bits_per_sample
            ),
        }
        writer.finalize().context("finalize wav payload")?;
    }
    Ok(cursor.into_inner())
}

fn default_wav_spec(sample_rate: u32, channels: usize) -> Result<WavSpec> {
    let channels = u16::try_from(channels).context("wav output channel count exceeds u16")?;
    Ok(WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    })
}

fn resolve_chunk_audio_format(
    audio: &DecodedAudio,
    output_format: ChunkOutputFormat,
) -> Result<AudioFormat> {
    match output_format {
        ChunkOutputFormat::Auto => match audio.format {
            AudioFormat::Wav(wav) => Ok(AudioFormat::Wav(wav)),
            AudioFormat::OggOpus => Ok(AudioFormat::OggOpus),
            AudioFormat::Mp3 => Ok(AudioFormat::Wav(WavEncoding {
                spec: default_wav_spec(audio.sample_rate, audio.channels)?,
            })),
        },
        ChunkOutputFormat::Wav => Ok(AudioFormat::Wav(WavEncoding {
            spec: match audio.format {
                AudioFormat::Wav(wav) => wav.spec,
                _ => default_wav_spec(audio.sample_rate, audio.channels)?,
            },
        })),
        ChunkOutputFormat::OggOpus => Ok(AudioFormat::OggOpus),
    }
}

fn prepare_chunk_output_samples(
    audio: &DecodedAudio,
    samples: &[f32],
    output_format: ChunkOutputFormat,
) -> Result<Vec<f32>> {
    Ok(match resolve_chunk_audio_format(audio, output_format)? {
        AudioFormat::Wav(_) => samples.to_vec(),
        AudioFormat::OggOpus => {
            if audio.sample_rate == OPUS_SAMPLE_RATE {
                samples.to_vec()
            } else {
                resample_linear_interleaved(
                    samples,
                    audio.channels,
                    audio.sample_rate,
                    OPUS_SAMPLE_RATE,
                )
            }
        }
        AudioFormat::Mp3 => unreachable!("chunk output format does not resolve to MP3"),
    })
}

fn encode_ogg_opus(samples: &[f32], channels: usize) -> Result<Vec<u8>> {
    if channels == 0 || channels > 2 {
        bail!("Opus output supports mono and stereo only");
    }

    let opus_channels = if channels == 1 {
        Channels::Mono
    } else {
        Channels::Stereo
    };
    let mut encoder = Encoder::new(OPUS_SAMPLE_RATE, opus_channels, Application::Audio)
        .context("create opus encoder")?;
    encoder
        .set_signal(OpusSignal::Voice)
        .context("set opus signal hint")?;

    let frame_len = OPUS_FRAME_SAMPLES * channels;
    let mut frame = vec![0.0_f32; frame_len];
    let mut packets = Vec::new();
    let mut offset = 0usize;
    while offset < samples.len() {
        let remaining = (samples.len() - offset).min(frame_len);
        frame.fill(0.0);
        frame[..remaining].copy_from_slice(&samples[offset..offset + remaining]);
        let packet = encoder
            .encode_vec_float(&frame, MAX_OPUS_PACKET_SIZE)
            .context("encode opus frame")?;
        packets.push(packet);
        offset += remaining;
    }

    if packets.is_empty() {
        let packet = encoder
            .encode_vec_float(&frame, MAX_OPUS_PACKET_SIZE)
            .context("encode empty opus frame")?;
        packets.push(packet);
    }

    let total_frames = samples.len().div_ceil(channels);
    let mut output = Vec::new();
    pack_ogg_opus(&packets, opus_channels, total_frames, &mut output);
    Ok(output)
}

pub fn resample_linear_interleaved(
    samples: &[f32],
    channels: usize,
    input_rate: u32,
    output_rate: u32,
) -> Vec<f32> {
    if input_rate == output_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let input_frames = samples.len() / channels;
    let output_frames = (input_frames as u128 * output_rate as u128).div_ceil(input_rate as u128);
    let output_frames = output_frames as usize;
    let mut output = vec![0.0_f32; output_frames * channels];

    for frame_index in 0..output_frames {
        let position = frame_index as f64 * input_rate as f64 / output_rate as f64;
        let left = position.floor() as usize;
        let right = left
            .min(input_frames.saturating_sub(1))
            .saturating_add(1)
            .min(input_frames.saturating_sub(1));
        let alpha = (position - left as f64) as f32;

        for channel in 0..channels {
            let left_sample = samples[left * channels + channel];
            let right_sample = samples[right * channels + channel];
            output[frame_index * channels + channel] =
                left_sample + (right_sample - left_sample) * alpha;
        }
    }

    output
}

fn append_with_fades(
    output: &mut Vec<f32>,
    segment: &[f32],
    channels: usize,
    fade_frames: usize,
    fade_in: bool,
    fade_out: bool,
) {
    let segment_frames = segment.len() / channels;
    let fade_frames = fade_frames.min(segment_frames / 2);
    let start = output.len();
    output.extend_from_slice(segment);
    let slice = &mut output[start..];

    if fade_frames == 0 {
        return;
    }

    if fade_in {
        for frame_index in 0..fade_frames {
            let gain = frame_index as f32 / fade_frames as f32;
            for channel in 0..channels {
                slice[frame_index * channels + channel] *= gain;
            }
        }
    }

    if fade_out {
        for frame_index in 0..fade_frames {
            let gain = (fade_frames - frame_index) as f32 / fade_frames as f32;
            let base = (segment_frames - fade_frames + frame_index) * channels;
            for channel in 0..channels {
                slice[base + channel] *= gain;
            }
        }
    }
}

fn rewrite_path_extension(path: &str, ext: &str) -> String {
    let input = std::path::Path::new(path);
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    let file_name = format!("{stem}.{ext}");
    match input.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            parent.join(file_name).to_string_lossy().into_owned()
        }
        _ => file_name,
    }
}

fn is_mpeg_audio_frame(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0
}

fn append_planar_samples(
    output: &mut Vec<f32>,
    channels: usize,
    frames: usize,
    mut sample_at: impl FnMut(usize, usize) -> f32,
) {
    output.reserve(frames.saturating_mul(channels));
    for frame in 0..frames {
        for channel in 0..channels {
            output.push(sample_at(channel, frame));
        }
    }
}

fn pcm_scale(bits_per_sample: u16) -> Result<f32> {
    match bits_per_sample {
        8 => Ok(128.0),
        16 => Ok(32_768.0),
        24 => Ok(8_388_608.0),
        32 => Ok(2_147_483_648.0),
        other => bail!("unsupported PCM depth: {other}"),
    }
}

fn parse_opus_head(packet: &[u8]) -> Result<OpusHead> {
    if packet.len() < 19 || &packet[..8] != b"OpusHead" {
        bail!("invalid OpusHead packet");
    }
    Ok(OpusHead {
        channels: packet[9],
        pre_skip: u16::from_le_bytes([packet[10], packet[11]]),
    })
}

#[derive(Debug, Clone, Copy)]
struct OpusHead {
    channels: u8,
    pre_skip: u16,
}

fn pack_ogg_opus(
    packets: &[Vec<u8>],
    channels: Channels,
    total_frames: usize,
    output: &mut Vec<u8>,
) {
    let mut sequence = 0_u32;
    let mut opus_head = Vec::with_capacity(19);
    opus_head.extend_from_slice(b"OpusHead");
    opus_head.push(1);
    opus_head.push(if matches!(channels, Channels::Mono) {
        1
    } else {
        2
    });
    opus_head.extend_from_slice(&0_u16.to_le_bytes());
    opus_head.extend_from_slice(&OPUS_SAMPLE_RATE.to_le_bytes());
    opus_head.extend_from_slice(&0_i16.to_le_bytes());
    opus_head.push(0);
    write_ogg_page(output, 0x02, 0, sequence, &[opus_head]);
    sequence += 1;

    let mut opus_tags = Vec::new();
    opus_tags.extend_from_slice(b"OpusTags");
    opus_tags.extend_from_slice(&(7_u32).to_le_bytes());
    opus_tags.extend_from_slice(b"opus-sm");
    opus_tags.extend_from_slice(&0_u32.to_le_bytes());
    write_ogg_page(output, 0x00, 0, sequence, &[opus_tags]);
    sequence += 1;

    for (index, packet) in packets.iter().enumerate() {
        let is_last = index + 1 == packets.len();
        let packet_frames = ((index + 1) * OPUS_FRAME_SAMPLES).min(total_frames);
        write_ogg_page(
            output,
            if is_last { 0x04 } else { 0x00 },
            packet_frames as u64,
            sequence,
            std::slice::from_ref(packet),
        );
        sequence += 1;
    }
}

fn write_ogg_page(
    output: &mut Vec<u8>,
    header_type: u8,
    granule_position: u64,
    sequence: u32,
    packets: &[Vec<u8>],
) {
    let mut lacing = Vec::new();
    let mut body = Vec::new();
    for packet in packets {
        let mut remaining = packet.len();
        while remaining >= 255 {
            lacing.push(255);
            remaining -= 255;
        }
        lacing.push(remaining as u8);
        body.extend_from_slice(packet);
    }

    let header_len = 27 + lacing.len();
    let page_start = output.len();
    output.resize(page_start + header_len + body.len(), 0);
    let page = &mut output[page_start..];

    page[0..4].copy_from_slice(b"OggS");
    page[4] = 0;
    page[5] = header_type;
    page[6..14].copy_from_slice(&granule_position.to_le_bytes());
    page[14..18].copy_from_slice(&OGG_SERIAL.to_le_bytes());
    page[18..22].copy_from_slice(&sequence.to_le_bytes());
    page[26] = lacing.len() as u8;
    page[27..27 + lacing.len()].copy_from_slice(&lacing);
    page[27 + lacing.len()..27 + lacing.len() + body.len()].copy_from_slice(&body);

    let checksum = ogg_crc32(page);
    page[22..26].copy_from_slice(&checksum.to_le_bytes());
}

fn ogg_crc32(data: &[u8]) -> u32 {
    let mut crc = 0_u32;
    for &byte in data {
        crc ^= (byte as u32) << 24;
        for _ in 0..8 {
            if (crc & 0x8000_0000) != 0 {
                crc = (crc << 1) ^ 0x04C1_1DB7;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}
