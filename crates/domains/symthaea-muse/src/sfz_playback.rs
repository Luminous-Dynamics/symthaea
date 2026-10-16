// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing one-shot playback for the supported SFZ mapping subset.
//!
//! This is not a complete SFZ voice engine. It decodes one selected sample,
//! converts it to mono f32, and changes pitch with deterministic linear
//! interpolation. Envelopes, loops, release triggers, velocity gain/crossfades,
//! filters, and modulation are not applied. The receipt records these limits.

use super::{inspect_sample_reference, plan_sfz_zone_selection, SfzMappingManifestV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{Cursor, ErrorKind};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const MAX_SOURCE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_DECODED_FRAMES: usize = 10_000_000;
const MAX_RENDERED_FRAMES: usize = 20_000_000;
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 384_000;
const MAX_CHANNELS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SfzPlaybackError {
    InvalidOutputSampleRate(u32),
    SelectionUnresolved(String),
    AssetMissingOrUnsafe(String),
    AssetTooLarge(u64),
    AssetChangedDuringRead,
    AssetReadFailed(String),
    ProbeFailed(String),
    NoDefaultTrack,
    DecoderUnavailable(String),
    DecodeFailed(String),
    InvalidAudioMetadata(String),
    SampleTooLong,
    NoDecodedFrames,
    InvalidResamplingRatio,
    OutputTooLong,
}

impl fmt::Display for SfzPlaybackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputSampleRate(rate) => write!(f, "output sample rate {rate} is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"),
            Self::SelectionUnresolved(decision) => write!(f, "SFZ selection is not executable: {decision}"),
            Self::AssetMissingOrUnsafe(path) => write!(f, "sample asset is missing or rejected as unsafe: {path}"),
            Self::AssetTooLarge(size) => write!(f, "sample asset is {size} bytes; limit is {MAX_SOURCE_BYTES}"),
            Self::AssetChangedDuringRead => write!(f, "sample asset changed while its bytes were read"),
            Self::AssetReadFailed(message) => write!(f, "sample read failed: {message}"),
            Self::ProbeFailed(message) => write!(f, "audio format probe failed: {message}"),
            Self::NoDefaultTrack => write!(f, "audio container has no default track"),
            Self::DecoderUnavailable(message) => write!(f, "audio decoder unavailable: {message}"),
            Self::DecodeFailed(message) => write!(f, "audio decode failed: {message}"),
            Self::InvalidAudioMetadata(message) => write!(f, "invalid audio metadata or samples: {message}"),
            Self::SampleTooLong => write!(f, "decoded sample exceeds frame safety limit"),
            Self::NoDecodedFrames => write!(f, "audio stream yielded no decoded frames"),
            Self::InvalidResamplingRatio => write!(f, "pitch/resampling ratio is invalid"),
            Self::OutputTooLong => write!(f, "rendered sample exceeds frame safety limit"),
        }
    }
}

impl std::error::Error for SfzPlaybackError {}

/// Receipt binds SFZ choice, exact sample bytes, codec metadata, and output hash.
/// The evidence fields explicitly exclude unsupported instrument semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfzPlaybackReceiptV1 {
    pub schema: String,
    pub sfz_sha256: String,
    pub requested_key: u8,
    pub requested_velocity: u8,
    pub round_robin_index: u64,
    pub selection_decision: String,
    pub selected_region_index: usize,
    pub sample_path: String,
    pub sample_sha256: String,
    pub sample_byte_len: u64,
    pub source_codec: String,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub decoded_source_frames: usize,
    pub frequency_ratio: f64,
    pub output_sample_rate: u32,
    pub output_frames: usize,
    pub output_sha256: String,
    pub peak_abs: f64,
    pub rms: f64,
    pub unexpected_silence: bool,
    pub unhandled_sfz_opcodes: Vec<String>,
    pub amplitude_envelope: String,
    pub velocity_amplitude_scaling: String,
    pub loop_and_release_semantics: String,
    pub resampler: String,
    pub audio_decode_evidence: String,
    pub rendering_evidence: String,
}

struct DecodedSample {
    mono: Vec<f32>,
    sample_rate: u32,
    channels: usize,
    codec: String,
    sha256: String,
    byte_len: u64,
}

/// Select, decode, and resample one SFZ zone to mono f32.
/// The returned receipt does not claim full SFZ playback or acoustic fidelity.
pub fn render_sfz_note_once(
    manifest: &SfzMappingManifestV1,
    sample_root: &Path,
    requested_key: u8,
    requested_velocity: u8,
    round_robin_index: u64,
    output_sample_rate: u32,
) -> Result<(SfzPlaybackReceiptV1, Vec<f32>), SfzPlaybackError> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&output_sample_rate) {
        return Err(SfzPlaybackError::InvalidOutputSampleRate(output_sample_rate));
    }

    let selection = plan_sfz_zone_selection(manifest, requested_key, requested_velocity, round_robin_index);
    let region_index = selection.selected_region_index
        .ok_or_else(|| SfzPlaybackError::SelectionUnresolved(selection.decision.clone()))?;
    let sample_path = selection.selected_sample_path.as_deref()
        .ok_or_else(|| SfzPlaybackError::SelectionUnresolved(selection.decision.clone()))?;
    let ratio = selection.frequency_ratio
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or(SfzPlaybackError::InvalidResamplingRatio)?;
    let decoded = decode_checked_sample(sample_root, sample_path)?;
    if decoded.mono.len() < 2 {
        return Err(SfzPlaybackError::NoDecodedFrames);
    }

    // Linear interpolation is the first executable baseline, not a high-quality
    // band-limited resampler. The receipt lets benchmark suites distinguish it.
    let source_step = ratio * f64::from(decoded.sample_rate) / f64::from(output_sample_rate);
    if !source_step.is_finite() || source_step <= 0.0 {
        return Err(SfzPlaybackError::InvalidResamplingRatio);
    }
    let output_frames = (((decoded.mono.len() - 1) as f64) / source_step).floor() as usize + 1;
    if output_frames == 0 || output_frames > MAX_RENDERED_FRAMES {
        return Err(SfzPlaybackError::OutputTooLong);
    }

    let mut samples = Vec::with_capacity(output_frames);
    let last = decoded.mono.len() - 1;
    for output_index in 0..output_frames {
        let source_position = (output_index as f64 * source_step).min(last as f64);
        let lower = source_position.floor() as usize;
        let upper = (lower + 1).min(last);
        let fraction = (source_position - lower as f64) as f32;
        let sample = decoded.mono[lower] + (decoded.mono[upper] - decoded.mono[lower]) * fraction;
        if !sample.is_finite() {
            return Err(SfzPlaybackError::InvalidAudioMetadata("resampling produced a non-finite sample".to_owned()));
        }
        samples.push(sample);
    }

    let mut hasher = Sha256::new();
    let mut peak_abs = 0.0_f64;
    let mut squares = 0.0_f64;
    for sample in &samples {
        hasher.update(sample.to_le_bytes());
        let value = f64::from(*sample);
        peak_abs = peak_abs.max(value.abs());
        squares += value * value;
    }
    let rms = (squares / samples.len() as f64).sqrt();
    let receipt = SfzPlaybackReceiptV1 {
        schema: "muse-sfz-one-shot-playback-receipt-v1".to_owned(),
        sfz_sha256: manifest.sfz_sha256.clone(),
        requested_key,
        requested_velocity,
        round_robin_index,
        selection_decision: selection.decision,
        selected_region_index: region_index,
        sample_path: sample_path.to_owned(),
        sample_sha256: decoded.sha256,
        sample_byte_len: decoded.byte_len,
        source_codec: decoded.codec,
        source_sample_rate: decoded.sample_rate,
        source_channels: decoded.channels,
        decoded_source_frames: decoded.mono.len(),
        frequency_ratio: ratio,
        output_sample_rate,
        output_frames: samples.len(),
        output_sha256: format!("{:x}", hasher.finalize()),
        peak_abs,
        rms,
        unexpected_silence: peak_abs <= 1.0e-12,
        unhandled_sfz_opcodes: manifest.unhandled_opcodes.clone(),
        amplitude_envelope: "not_applied".to_owned(),
        velocity_amplitude_scaling: "not_applied; velocity selects a zone only".to_owned(),
        loop_and_release_semantics: "not_applied".to_owned(),
        resampler: "linear_interpolation_unfiltered_v1".to_owned(),
        audio_decode_evidence: "decoded_to_f32_and_validated".to_owned(),
        rendering_evidence: "deterministic_one_shot_sample_resampling_not_full_sfz_playback".to_owned(),
    };
    Ok((receipt, samples))
}

fn decode_checked_sample(root: &Path, relative_path: &str) -> Result<DecodedSample, SfzPlaybackError> {
    let inspected = inspect_sample_reference(root, relative_path, false)
        .map_err(|error| SfzPlaybackError::AssetReadFailed(error.to_string()))?;
    let Some((inspected_len, _)) = inspected else {
        return Err(SfzPlaybackError::AssetMissingOrUnsafe(relative_path.to_owned()));
    };
    if inspected_len > MAX_SOURCE_BYTES {
        return Err(SfzPlaybackError::AssetTooLarge(inspected_len));
    }
    let bytes = std::fs::read(root.join(relative_path))
        .map_err(|error| SfzPlaybackError::AssetReadFailed(error.to_string()))?;
    if bytes.len() as u64 != inspected_len {
        return Err(SfzPlaybackError::AssetChangedDuringRead);
    }
    let sample_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let post_read = inspect_sample_reference(root, relative_path, false)
        .map_err(|error| SfzPlaybackError::AssetReadFailed(error.to_string()))?;
    if !matches!(post_read, Some((len, _)) if len == inspected_len) {
        return Err(SfzPlaybackError::AssetChangedDuringRead);
    }

    let mut hint = Hint::new();
    if let Some(extension) = Path::new(relative_path).extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe().format(
        &hint, source, &FormatOptions::default(), &MetadataOptions::default()
    ).map_err(|error| SfzPlaybackError::ProbeFailed(error.to_string()))?;
    let mut format = probed.format;
    let track = format.default_track().ok_or(SfzPlaybackError::NoDefaultTrack)?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let codec = format!("{:?}", codec_params.codec);
    let mut decoder = symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())
        .map_err(|error| SfzPlaybackError::DecoderUnavailable(error.to_string()))?;

    let mut mono = Vec::<f32>::new();
    let mut source_rate = None;
    let mut source_channels = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(ref error)) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(SfzPlaybackError::DecodeFailed(error.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)
            .map_err(|error| SfzPlaybackError::DecodeFailed(error.to_string()))?;
        let spec = *decoded.spec();
        let channels = spec.channels.count();
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(SfzPlaybackError::InvalidAudioMetadata(format!("channel count {channels} is outside 1..={MAX_CHANNELS}")));
        }
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&spec.rate) {
            return Err(SfzPlaybackError::InvalidAudioMetadata(format!("sample rate {} is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz", spec.rate)));
        }
        if let Some(rate) = source_rate {
            if rate != spec.rate || source_channels != Some(channels) {
                return Err(SfzPlaybackError::InvalidAudioMetadata("sample rate or channel layout changed mid-stream".to_owned()));
            }
        } else {
            source_rate = Some(spec.rate);
            source_channels = Some(channels);
        }
        if decoded.capacity() as usize > MAX_DECODED_FRAMES
            || mono.len().saturating_add(decoded.capacity() as usize) > MAX_DECODED_FRAMES {
            return Err(SfzPlaybackError::SampleTooLong);
        }
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        let interleaved = buffer.samples();
        if interleaved.len() % channels != 0 {
            return Err(SfzPlaybackError::InvalidAudioMetadata("interleaved sample count is not divisible by channel count".to_owned()));
        }
        for frame in interleaved.chunks_exact(channels) {
            if !frame.iter().all(|sample| sample.is_finite()) {
                return Err(SfzPlaybackError::InvalidAudioMetadata("decoder returned a non-finite sample".to_owned()));
            }
            mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
            if mono.len() > MAX_DECODED_FRAMES {
                return Err(SfzPlaybackError::SampleTooLong);
            }
        }
    }
    if mono.is_empty() {
        return Err(SfzPlaybackError::NoDecodedFrames);
    }
    Ok(DecodedSample {
        mono,
        sample_rate: source_rate.ok_or(SfzPlaybackError::NoDecodedFrames)?,
        channels: source_channels.ok_or(SfzPlaybackError::NoDecodedFrames)?,
        codec,
        sha256: sample_sha256,
        byte_len: inspected_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfz_manifest::parse_sfz_mapping;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct FixtureRoot(std::path::PathBuf);
    impl FixtureRoot {
        fn new() -> Self {
            let nonce = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("symthaea-muse-sfz-{}-{nonce}", std::process::id()));
            std::fs::create_dir_all(path.join("samples")).unwrap();
            Self(path)
        }
        fn root(&self) -> &Path { &self.0 }
        fn write_wave(&self, relative: &str, frames: usize) -> std::path::PathBuf {
            let path = self.0.join(relative);
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).unwrap(); }
            let mut writer = hound::WavWriter::create(&path, hound::WavSpec {
                channels: 1, sample_rate: 44_100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
            }).unwrap();
            for index in 0..frames {
                let angle = std::f64::consts::TAU * 440.0 * index as f64 / 44_100.0;
                writer.write_sample((angle.sin() * 16_000.0).round() as i16).unwrap();
            }
            writer.finalize().unwrap();
            path
        }
    }
    impl Drop for FixtureRoot {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    fn manifest() -> SfzMappingManifestV1 {
        parse_sfz_mapping("<group> lokey=60 hikey=72 pitch_keycenter=60\n<region> sample=samples/test.wav").unwrap()
    }

    #[test]
    fn renders_deterministically_and_binds_source_and_output_hashes() {
        let fixture = FixtureRoot::new();
        fixture.write_wave("samples/test.wav", 1_024);
        let mapping = manifest();
        let (base_receipt, base) = render_sfz_note_once(&mapping, fixture.root(), 60, 96, 0, 44_100).unwrap();
        let (up_receipt, up) = render_sfz_note_once(&mapping, fixture.root(), 72, 96, 0, 44_100).unwrap();
        let (repeat_receipt, repeat) = render_sfz_note_once(&mapping, fixture.root(), 72, 96, 0, 44_100).unwrap();
        assert_eq!(base.len(), 1_024);
        assert_eq!(up.len(), 512);
        assert_eq!(base_receipt.frequency_ratio, 1.0);
        assert!((up_receipt.frequency_ratio - 2.0).abs() < 1.0e-12);
        assert_eq!(up, repeat);
        assert_eq!(up_receipt.output_sha256, repeat_receipt.output_sha256);
        assert_eq!(up_receipt.audio_decode_evidence, "decoded_to_f32_and_validated");
        assert!(!up_receipt.unexpected_silence);
        assert!(up.iter().all(|sample| sample.is_finite()));
        let input = std::fs::read(fixture.root().join("samples/test.wav")).unwrap();
        assert_eq!(up_receipt.sample_sha256, format!("{:x}", Sha256::digest(input)));
    }

    #[test]
    fn refuses_ambiguous_zones_before_opening_assets() {
        let fixture = FixtureRoot::new();
        let mapping = parse_sfz_mapping("<group> key=60\n<region> sample=samples/a.wav\n<region> sample=samples/b.wav").unwrap();
        assert!(matches!(
            render_sfz_note_once(&mapping, fixture.root(), 60, 96, 0, 44_100),
            Err(SfzPlaybackError::SelectionUnresolved(ref decision)) if decision == "ambiguous_zones"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_sample_assets() {
        use std::os::unix::fs::symlink;
        let fixture = FixtureRoot::new();
        let nonce = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let outside = std::env::temp_dir().join(format!("sfz-outside-{}-{nonce}.wav", std::process::id()));
        let mut writer = hound::WavWriter::create(&outside, hound::WavSpec {
            channels: 1, sample_rate: 44_100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
        }).unwrap();
        writer.write_sample(123_i16).unwrap();
        writer.write_sample(456_i16).unwrap();
        writer.finalize().unwrap();
        symlink(&outside, fixture.root().join("samples/alias.wav")).unwrap();
        let mapping = parse_sfz_mapping("<region> key=60 sample=samples/alias.wav").unwrap();
        assert!(matches!(
            render_sfz_note_once(&mapping, fixture.root(), 60, 96, 0, 44_100),
            Err(SfzPlaybackError::AssetMissingOrUnsafe(_))
        ));
        let _ = std::fs::remove_file(outside);
    }

    #[test]
    fn rejects_invalid_output_rates_and_bad_audio_bytes() {
        let fixture = FixtureRoot::new();
        fixture.write_wave("samples/test.wav", 128);
        let mapping = manifest();
        assert!(matches!(
            render_sfz_note_once(&mapping, fixture.root(), 60, 96, 0, 0),
            Err(SfzPlaybackError::InvalidOutputSampleRate(0))
        ));
        std::fs::write(fixture.root().join("samples/test.wav"), b"not audio").unwrap();
        assert!(matches!(
            render_sfz_note_once(&mapping, fixture.root(), 60, 96, 0, 44_100),
            Err(SfzPlaybackError::ProbeFailed(_))
        ));
    }
}
