// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Artifact-bound receipts for successful local render executions.
//!
//! A receipt binds the exact bytes and metadata that the adapter observed. It
//! is deliberately not a renderer attestation: an untrusted process can lie
//! about which binary/configuration it used, and unkeyed hashes do not defeat
//! a malicious host. Callers must hash executable/assets from the exact files
//! they invoked, supply the exact bytes passed to the renderer, and preserve
//! the receipt next to the output artifact.
//!
//! decoded_pcm_f32_le_sha256 hashes a canonical stream of decoded samples
//! represented as little-endian f32: integer PCM is normalized by its signed
//! full-scale value, float PCM is preserved. It is different from the hash of
//! the WAV container bytes and makes sample-level comparisons possible.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

pub const RENDER_EXECUTION_RECEIPT_VERSION: &str = "mel-mr-eval-015-render-execution-receipt-v0.1";
const MAX_RENDER_ID_BYTES: usize = 256;
const MAX_CHANNELS: u16 = 32;
const MAX_SAMPLE_RATE_HZ: u32 = 768_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererBackend {
    NativeMuse,
    FluidSynth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodedSampleFormat {
    PcmInteger,
    PcmFloat,
}

/// Facts parsed from the exact output WAV, never caller-declared metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioArtifactSummary {
    pub wav_sha256: String,
    pub decoded_pcm_f32_le_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_format: DecodedSampleFormat,
    /// Complete interleaved sample frames, not scalar samples.
    pub frame_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RendererExecutionReceipt {
    pub receipt_version: String,
    pub render_id: String,
    pub backend: RendererBackend,
    /// Digest of the exact serialized Score supplied to the render adapter.
    pub score_sha256: String,
    /// Exactly one of input_midi_sha256 and performed_schedule_sha256 is set.
    /// FluidSynth consumes MIDI; the native renderer consumes its in-memory
    /// performed schedule and must not invent a MIDI input artifact.
    pub input_midi_sha256: Option<String>,
    pub performed_schedule_sha256: Option<String>,
    /// Digests supplied by the adapter after hashing the executable and assets
    /// actually selected by the local runtime. These are not independently
    /// attested merely because they occur in this receipt.
    pub renderer_executable_sha256: Option<String>,
    pub soundfont_sha256: Option<String>,
    pub render_config_sha256: String,
    pub process_exit_code: Option<i32>,
    pub performed_event_map_sha256: Option<String>,
    pub output: AudioArtifactSummary,
    /// SHA-256 over this receipt with receipt_sha256 blank.
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RendererReceiptError {
    InvalidRenderId,
    InvalidSha256(&'static str),
    InvalidInputArtifacts,
    InvalidExecutionStatus,
    InvalidWav(&'static str),
    NonFinitePcm,
    PcmSampleCountOverflow,
    SerializationFailure,
    InvalidReceipt,
    ArtifactMismatch(&'static str),
    FileReadFailure,
}

impl std::fmt::Display for RendererReceiptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRenderId => write!(f, "invalid render ID"),
            Self::InvalidSha256(field) => write!(f, "invalid SHA-256 field: {field}"),
            Self::InvalidInputArtifacts => {
                write!(f, "renderer backend requires a mutually consistent input artifact set")
            }
            Self::InvalidExecutionStatus => {
                write!(f, "receipt may only be created for a successful render execution")
            }
            Self::InvalidWav(reason) => write!(f, "invalid output WAV: {reason}"),
            Self::NonFinitePcm => write!(f, "decoded PCM contains a non-finite sample"),
            Self::PcmSampleCountOverflow => write!(f, "PCM sample count overflow"),
            Self::SerializationFailure => write!(f, "could not serialize renderer receipt"),
            Self::InvalidReceipt => write!(f, "renderer receipt digest does not verify"),
            Self::ArtifactMismatch(field) => write!(f, "render artifact mismatch: {field}"),
            Self::FileReadFailure => write!(f, "could not read an artifact file"),
        }
    }
}

impl std::error::Error for RendererReceiptError {}

/// Exact locally observed input artifacts for a successful render.
///
/// For FluidSynth, input MIDI bytes, executable digest, SoundFont digest and
/// process_exit_code == Some(0) are required. For NativeMuse, performed_schedule_bytes
/// is required, MIDI must be absent, and process_exit_code must be None because
/// the renderer executes in-process.
pub struct RendererReceiptInputs<'a> {
    pub render_id: &'a str,
    pub backend: RendererBackend,
    pub score_bytes: &'a [u8],
    pub render_config_bytes: &'a [u8],
    pub input_midi_bytes: Option<&'a [u8]>,
    pub performed_schedule_bytes: Option<&'a [u8]>,
    pub renderer_executable_sha256: Option<&'a str>,
    pub soundfont_sha256: Option<&'a str>,
    pub process_exit_code: Option<i32>,
    pub performed_event_map_sha256: Option<&'a str>,
    pub output_wav_bytes: &'a [u8],
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Hash an exact artifact file. Callers should hash the resolved executable
/// and SoundFont paths used by the adapter, not a similarly named default.
pub fn sha256_file(path: &Path) -> Result<String, RendererReceiptError> {
    let mut file = std::fs::File::open(path).map_err(|_| RendererReceiptError::FileReadFailure)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| RendererReceiptError::FileReadFailure)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_render_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_RENDER_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn optional_digest_is_valid(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), RendererReceiptError> {
    if let Some(value) = value
        && !is_lower_hex_sha256(value)
    {
        return Err(RendererReceiptError::InvalidSha256(field));
    }
    Ok(())
}

fn validate_inputs(inputs: &RendererReceiptInputs<'_>) -> Result<(), RendererReceiptError> {
    if !valid_render_id(inputs.render_id) {
        return Err(RendererReceiptError::InvalidRenderId);
    }
    optional_digest_is_valid(inputs.renderer_executable_sha256, "renderer_executable_sha256")?;
    optional_digest_is_valid(inputs.soundfont_sha256, "soundfont_sha256")?;
    optional_digest_is_valid(
        inputs.performed_event_map_sha256,
        "performed_event_map_sha256",
    )?;

    match inputs.backend {
        RendererBackend::FluidSynth => {
            if inputs.input_midi_bytes.is_none()
                || inputs.performed_schedule_bytes.is_some()
                || inputs.renderer_executable_sha256.is_none()
                || inputs.soundfont_sha256.is_none()
                || inputs.process_exit_code != Some(0)
            {
                return Err(RendererReceiptError::InvalidInputArtifacts);
            }
        }
        RendererBackend::NativeMuse => {
            if inputs.input_midi_bytes.is_some()
                || inputs.performed_schedule_bytes.is_none()
                || inputs.soundfont_sha256.is_some()
                || inputs.process_exit_code.is_some()
            {
                return Err(RendererReceiptError::InvalidInputArtifacts);
            }
        }
    }
    Ok(())
}

impl RendererExecutionReceipt {
    /// Create a canonical receipt for a successful local rendering operation.
    /// The output WAV header and decoded PCM facts are measured from bytes.
    pub fn from_successful_render(
        inputs: RendererReceiptInputs<'_>,
    ) -> Result<Self, RendererReceiptError> {
        validate_inputs(&inputs)?;
        let output = summarize_wav(inputs.output_wav_bytes)?;
        let mut receipt = Self {
            receipt_version: RENDER_EXECUTION_RECEIPT_VERSION.into(),
            render_id: inputs.render_id.into(),
            backend: inputs.backend,
            score_sha256: sha256_hex(inputs.score_bytes),
            input_midi_sha256: inputs.input_midi_bytes.map(sha256_hex),
            performed_schedule_sha256: inputs.performed_schedule_bytes.map(sha256_hex),
            renderer_executable_sha256: inputs.renderer_executable_sha256.map(str::to_owned),
            soundfont_sha256: inputs.soundfont_sha256.map(str::to_owned),
            render_config_sha256: sha256_hex(inputs.render_config_bytes),
            process_exit_code: inputs.process_exit_code,
            performed_event_map_sha256: inputs.performed_event_map_sha256.map(str::to_owned),
            output,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.compute_sha256()?;
        Ok(receipt)
    }

    /// Rebuild the expected receipt from the exact same artifacts and context.
    /// Any changed Score, MIDI/schedule, config, executable/SoundFont digest,
    /// map receipt, WAV byte, decoded sample, or execution status rejects.
    pub fn verify_artifacts(
        &self,
        inputs: RendererReceiptInputs<'_>,
    ) -> Result<(), RendererReceiptError> {
        if !is_lower_hex_sha256(&self.receipt_sha256)
            || self.compute_sha256()? != self.receipt_sha256
        {
            return Err(RendererReceiptError::InvalidReceipt);
        }
        let expected = Self::from_successful_render(inputs)?;
        if self != &expected {
            return Err(RendererReceiptError::ArtifactMismatch("receipt context or output"));
        }
        Ok(())
    }

    fn compute_sha256(&self) -> Result<String, RendererReceiptError> {
        let mut canonical = self.clone();
        canonical.receipt_sha256.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|_| RendererReceiptError::SerializationFailure)?;
        let mut hasher = Sha256::new();
        hasher.update(b"luminous-dynamics-render-execution-receipt-v1\0");
        hasher.update(bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

/// Parse the actual WAV bytes and hash decoded PCM as little-endian f32.
/// Integer samples are divided by 2^(bits_per_sample - 1); float samples are
/// hashed without peak normalization or clamping.
pub fn summarize_wav(wav_bytes: &[u8]) -> Result<AudioArtifactSummary, RendererReceiptError> {
    summarize_wav_reader(std::io::Cursor::new(wav_bytes), sha256_hex(wav_bytes))
}

fn summarize_wav_reader<R: std::io::Read + std::io::Seek>(
    reader: R,
    wav_sha256: String,
) -> Result<AudioArtifactSummary, RendererReceiptError> {
    let mut reader = hound::WavReader::new(reader)
        .map_err(|_| RendererReceiptError::InvalidWav("header or container"))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.channels > MAX_CHANNELS {
        return Err(RendererReceiptError::InvalidWav("unsupported channel count"));
    }
    if spec.sample_rate == 0 || spec.sample_rate > MAX_SAMPLE_RATE_HZ {
        return Err(RendererReceiptError::InvalidWav("unsupported sample rate"));
    }
    if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
        return Err(RendererReceiptError::InvalidWav("unsupported bit depth"));
    }

    let mut pcm_hasher = Sha256::new();
    let mut sample_count = 0_u64;
    let sample_format = match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                return Err(RendererReceiptError::InvalidWav("only 32-bit float PCM is supported"));
            }
            for sample in reader.samples::<f32>() {
                let sample = sample.map_err(|_| RendererReceiptError::InvalidWav("float PCM data"))?;
                if !sample.is_finite() {
                    return Err(RendererReceiptError::NonFinitePcm);
                }
                pcm_hasher.update(sample.to_bits().to_le_bytes());
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or(RendererReceiptError::PcmSampleCountOverflow)?;
            }
            DecodedSampleFormat::PcmFloat
        }
        hound::SampleFormat::Int => {
            let scale = 2_f64.powi(i32::from(spec.bits_per_sample) - 1);
            for sample in reader.samples::<i32>() {
                let sample = sample.map_err(|_| RendererReceiptError::InvalidWav("integer PCM data"))?;
                let normalized = (f64::from(sample) / scale) as f32;
                if !normalized.is_finite() {
                    return Err(RendererReceiptError::NonFinitePcm);
                }
                pcm_hasher.update(normalized.to_bits().to_le_bytes());
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or(RendererReceiptError::PcmSampleCountOverflow)?;
            }
            DecodedSampleFormat::PcmInteger
        }
    };
    if sample_count == 0 || sample_count % u64::from(spec.channels) != 0 {
        return Err(RendererReceiptError::InvalidWav("empty or partial sample frame"));
    }
    Ok(AudioArtifactSummary {
        wav_sha256,
        decoded_pcm_f32_le_sha256: format!("{:x}", pcm_hasher.finalize()),
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        bits_per_sample: spec.bits_per_sample,
        sample_format,
        frame_count: sample_count / u64::from(spec.channels),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const DIGEST_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DIGEST_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const DIGEST_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const DIGEST_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const DIGEST_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn test_wav(samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for sample in samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    fn fluid_inputs<'a>(wav: &'a [u8]) -> RendererReceiptInputs<'a> {
        RendererReceiptInputs {
            render_id: "render-test-1",
            backend: RendererBackend::FluidSynth,
            score_bytes: br#"{"score":"fixture"}"#,
            render_config_bytes: br#"{"gain":0.8,"sample_rate_hz":48000}"#,
            input_midi_bytes: Some(b"MThd-fixture-midi"),
            performed_schedule_bytes: None,
            renderer_executable_sha256: Some(DIGEST_C),
            soundfont_sha256: Some(DIGEST_D),
            process_exit_code: Some(0),
            performed_event_map_sha256: Some(DIGEST_E),
            output_wav_bytes: wav,
        }
    }

    #[test]
    fn parsed_wav_summary_counts_frames_and_binds_container_and_pcm() {
        let wav = test_wav(&[0, 16384, -16384, 8192, 4096, -4096]);
        let summary = summarize_wav(&wav).unwrap();
        assert_eq!(summary.sample_rate_hz, 48_000);
        assert_eq!(summary.channels, 2);
        assert_eq!(summary.bits_per_sample, 16);
        assert_eq!(summary.sample_format, DecodedSampleFormat::PcmInteger);
        assert_eq!(summary.frame_count, 3);
        assert_eq!(summary.wav_sha256, sha256_hex(&wav));
        assert!(is_lower_hex_sha256(&summary.decoded_pcm_f32_le_sha256));
    }

    #[test]
    fn exact_artifact_set_replays_the_same_receipt() {
        let wav = test_wav(&[0, 0, 16384, -16384]);
        let receipt = RendererExecutionReceipt::from_successful_render(fluid_inputs(&wav)).unwrap();
        receipt.verify_artifacts(fluid_inputs(&wav)).unwrap();
        assert_eq!(receipt.output.frame_count, 2);
        assert_eq!(receipt.process_exit_code, Some(0));
    }

    #[test]
    fn changed_wav_bytes_and_decoded_pcm_are_rejected() {
        let original = test_wav(&[0, 0, 16384, -16384]);
        let changed = test_wav(&[0, 0, 8192, -8192]);
        let receipt = RendererExecutionReceipt::from_successful_render(fluid_inputs(&original)).unwrap();
        assert_eq!(
            receipt.verify_artifacts(fluid_inputs(&changed)),
            Err(RendererReceiptError::ArtifactMismatch("receipt context or output"))
        );
    }

    #[test]
    fn changed_midi_and_configuration_are_rejected() {
        let wav = test_wav(&[0, 0, 16384, -16384]);
        let receipt = RendererExecutionReceipt::from_successful_render(fluid_inputs(&wav)).unwrap();
        let mut changed_midi = fluid_inputs(&wav);
        changed_midi.input_midi_bytes = Some(b"MThd-substituted-midi");
        assert_eq!(
            receipt.verify_artifacts(changed_midi),
            Err(RendererReceiptError::ArtifactMismatch("receipt context or output"))
        );
        let mut changed_config = fluid_inputs(&wav);
        changed_config.render_config_bytes = br#"{"gain":0.7,"sample_rate_hz":48000}"#;
        assert_eq!(
            receipt.verify_artifacts(changed_config),
            Err(RendererReceiptError::ArtifactMismatch("receipt context or output"))
        );
    }

    #[test]
    fn score_executable_soundfont_and_event_map_digests_are_bound() {
        let wav = test_wav(&[0, 0, 16384, -16384]);
        let receipt = RendererExecutionReceipt::from_successful_render(fluid_inputs(&wav)).unwrap();
        let mut changed_score = fluid_inputs(&wav);
        changed_score.score_bytes = br#"{"score":"different"}"#;
        assert!(receipt.verify_artifacts(changed_score).is_err());
        let mut changed_binary = fluid_inputs(&wav);
        changed_binary.renderer_executable_sha256 = Some(DIGEST_F);
        assert!(receipt.verify_artifacts(changed_binary).is_err());
        let mut changed_soundfont = fluid_inputs(&wav);
        changed_soundfont.soundfont_sha256 = Some(DIGEST_1);
        assert!(receipt.verify_artifacts(changed_soundfont).is_err());
        let mut changed_map = fluid_inputs(&wav);
        changed_map.performed_event_map_sha256 = Some(DIGEST_2);
        assert!(receipt.verify_artifacts(changed_map).is_err());
    }

    #[test]
    fn fluid_receipt_fails_closed_without_exact_input_or_success_status() {
        let wav = test_wav(&[0, 0, 1, -1]);
        let mut no_midi = fluid_inputs(&wav);
        no_midi.input_midi_bytes = None;
        assert_eq!(
            RendererExecutionReceipt::from_successful_render(no_midi).unwrap_err(),
            RendererReceiptError::InvalidInputArtifacts
        );
        let mut failed = fluid_inputs(&wav);
        failed.process_exit_code = Some(1);
        assert_eq!(
            RendererExecutionReceipt::from_successful_render(failed).unwrap_err(),
            RendererReceiptError::InvalidInputArtifacts
        );
    }

    #[test]
    fn native_receipt_binds_schedule_without_inventing_midi() {
        let wav = test_wav(&[0, 0, 1, -1]);
        let inputs = RendererReceiptInputs {
            render_id: "native-render",
            backend: RendererBackend::NativeMuse,
            score_bytes: br#"{"score":"fixture"}"#,
            render_config_bytes: br#"{"reverb":"default"}"#,
            input_midi_bytes: None,
            performed_schedule_bytes: Some(b"native-performed-event-schedule-v1"),
            renderer_executable_sha256: Some(DIGEST_F),
            soundfont_sha256: None,
            process_exit_code: None,
            performed_event_map_sha256: None,
            output_wav_bytes: &wav,
        };
        let receipt = RendererExecutionReceipt::from_successful_render(inputs).unwrap();
        assert!(receipt.input_midi_sha256.is_none());
        assert!(receipt.performed_schedule_sha256.is_some());
        assert_eq!(receipt.backend, RendererBackend::NativeMuse);
    }

    #[test]
    fn malformed_wav_and_non_finite_float_pcm_are_rejected() {
        assert!(summarize_wav(b"not a WAV").is_err());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            writer.write_sample(f32::NAN).unwrap();
            writer.finalize().unwrap();
        }
        assert_eq!(
            summarize_wav(&cursor.into_inner()),
            Err(RendererReceiptError::NonFinitePcm)
        );
    }

    #[test]
    fn receipt_tampering_is_rejected_even_when_artifacts_are_unchanged() {
        let wav = test_wav(&[0, 0, 16384, -16384]);
        let mut receipt = RendererExecutionReceipt::from_successful_render(fluid_inputs(&wav)).unwrap();
        receipt.process_exit_code = Some(7);
        assert_eq!(
            receipt.verify_artifacts(fluid_inputs(&wav)),
            Err(RendererReceiptError::InvalidReceipt)
        );
    }
}
