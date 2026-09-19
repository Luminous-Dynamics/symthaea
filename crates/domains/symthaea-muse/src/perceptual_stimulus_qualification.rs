// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1CR: executable byte/sample qualification for P1C stimuli.
//!
//! P1C freezes the stimulus manifest. This layer consumes the exact source and
//! participant-facing WAV bytes and independently proves that the only signal
//! transformation is the registered attenuation-only constant gain.
//!
//! Two identities remain separate:
//! - exact RIFF/WAVE file bytes presented to the participant;
//! - canonical decoded stereo PCM16 sample content.
//!
//! This module deliberately accepts only one canonical container profile so
//! byte reconstruction is deterministic: RIFF/WAVE PCM16, stereo, 44.1 kHz,
//! the minimal 44-byte header, one `fmt ` chunk and one `data` chunk, with no
//! metadata/ancillary chunks. Signal gain is applied in a normalized f64
//! domain using `10^(dB/20)`, with no dither, no limiter, no resampling, no
//! channel remix, and truncating Rust float-to-i16 quantization after scaling
//! by `i16::MAX`.

use crate::{
    auto_master::measure_lufs,
    evidence_digest::{
        canonical_json_sha256, sha256_hex,
        perceptual_stimulus_pack::{
            FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
            LOUDNESS_REMEASUREMENT_TOLERANCE_LU, MAX_RESIDUAL_PAIR_DELTA_LU_V1,
            NUMERIC_TOLERANCE_DB, REQUIRED_CHANNEL_COUNT, REQUIRED_SAMPLE_RATE_HZ,
            StimulusArmV1, StimulusAudioAssetV1,
        },
        perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
    },
};
use serde::{Deserialize, Serialize};

pub const STIMULUS_BYTE_QUALIFICATION_VERSION: &str =
    "mel003-perceptual-stimulus-byte-qualification-v1";
pub const CANONICAL_WAV_PROFILE_V1: &str =
    "riff-wave-pcm16le-stereo-44100-minimal44-v1";
pub const CANONICAL_PCM_DIGEST_PROFILE_V1: &str =
    "pcm16le-stereo-44100-domain-separated-v1";
pub const ATTENUATION_SAMPLE_TRANSFORM_PROFILE_V1: &str =
    "pcm16-normalized-f64-db20-no-dither-trunc-i16max-v1";

const WAV_HEADER_BYTES: usize = 44;
const PCM_BYTES_PER_FRAME: usize = 4;
const AUDIO_FORMAT_PCM: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;
const BLOCK_ALIGN: u16 = 4;
const BYTE_RATE: u32 = REQUIRED_SAMPLE_RATE_HZ * BLOCK_ALIGN as u32;
const CLIP_EPSILON: f64 = 1.0e-12;

/// Runtime-only bytes supplied to the qualifier. Raw audio is intentionally
/// not serialized into the evidence receipt.
#[derive(Debug, Clone, Copy)]
pub struct StimulusByteInputV1<'a> {
    pub item_id: &'a str,
    pub seed: u64,
    pub arm: StimulusArmV1,
    pub source_wav: &'a [u8],
    pub output_wav: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanonicalWavIssueV1 {
    TooShort,
    WrongRiffTag,
    WrongWaveTag,
    WrongRiffLength,
    WrongFmtTag,
    WrongFmtChunkSize,
    WrongAudioFormat,
    WrongChannelCount,
    WrongSampleRate,
    WrongByteRate,
    WrongBlockAlign,
    WrongBitsPerSample,
    WrongDataTag,
    WrongDataLength,
    NonIntegralFrameCount,
    FrameCountOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StimulusByteQualificationIssueV1 {
    InvalidProtocol,
    InvalidRenderSubjectBinding,
    InvalidStimulusPack,
    ProtocolSerializationFailed,
    RenderSubjectBindingSerializationFailed,
    StimulusPackSerializationFailed,
    WrongInputCount { found: usize, expected: usize },
    MissingInput { item_id: String, arm: StimulusArmV1 },
    DuplicateInput { item_id: String, arm: StimulusArmV1 },
    InputSeedMismatch { item_id: String, arm: StimulusArmV1 },
    SourceFileDigestMismatch { item_id: String, arm: StimulusArmV1 },
    OutputFileDigestMismatch { item_id: String, arm: StimulusArmV1 },
    InvalidSourceWav {
        item_id: String,
        arm: StimulusArmV1,
        issue: CanonicalWavIssueV1,
    },
    InvalidOutputWav {
        item_id: String,
        arm: StimulusArmV1,
        issue: CanonicalWavIssueV1,
    },
    SourceGeometryMismatch { item_id: String, arm: StimulusArmV1 },
    OutputGeometryMismatch { item_id: String, arm: StimulusArmV1 },
    NonFiniteGain { item_id: String, arm: StimulusArmV1 },
    PositiveGain { item_id: String, arm: StimulusArmV1 },
    UnexpectedClipping { item_id: String, arm: StimulusArmV1 },
    UnchangedArmBytesDiffer { item_id: String, arm: StimulusArmV1 },
    ReconstructedOutputMismatch { item_id: String, arm: StimulusArmV1 },
    NonFiniteMeasuredLoudness { item_id: String, arm: StimulusArmV1 },
    InitialLoudnessMismatch { item_id: String, arm: StimulusArmV1 },
    FinalLoudnessMismatch { item_id: String, arm: StimulusArmV1 },
    PairDeltaMismatch { item_id: String },
    PairDeltaExceedsTolerance { item_id: String },
    QualificationSerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedStimulusAssetV1 {
    pub item_id: String,
    pub seed: u64,
    pub arm: StimulusArmV1,
    pub source_file_sha256: String,
    pub source_pcm_sha256: String,
    pub output_file_sha256: String,
    pub output_pcm_sha256: String,
    pub reconstructed_output_file_sha256: String,
    pub reconstructed_output_pcm_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    pub frame_count: usize,
    pub applied_gain_db: f64,
    pub measured_initial_lufs: f64,
    pub measured_final_lufs: f64,
    pub source_output_file_identical: bool,
    pub reconstructed_output_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedStimulusPairV1 {
    pub item_id: String,
    pub seed: u64,
    pub baseline: QualifiedStimulusAssetV1,
    pub intervention: QualifiedStimulusAssetV1,
    pub independently_measured_pair_delta_lu: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPerceptualStimulusByteQualificationV1 {
    pub qualification_version: String,
    pub protocol_sha256: String,
    pub render_subject_binding_sha256: String,
    pub stimulus_pack_sha256: String,
    pub canonical_wav_profile: String,
    pub canonical_pcm_digest_profile: String,
    pub attenuation_sample_transform_profile: String,
    pub pairs: Vec<QualifiedStimulusPairV1>,
    pub qualification_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalPcm16Stereo {
    samples: Vec<[i16; 2]>,
}

impl CanonicalPcm16Stereo {
    fn frame_count(&self) -> usize {
        self.samples.len()
    }

    fn as_f32_frames(&self) -> Vec<[f32; 2]> {
        const SCALE: f32 = 1.0 / 32768.0;
        self.samples
            .iter()
            .map(|frame| [frame[0] as f32 * SCALE, frame[1] as f32 * SCALE])
            .collect()
    }

    fn pcm_sha256(&self) -> String {
        let mut bytes = Vec::with_capacity(64 + self.samples.len() * PCM_BYTES_PER_FRAME);
        bytes.extend_from_slice(CANONICAL_PCM_DIGEST_PROFILE_V1.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&REQUIRED_SAMPLE_RATE_HZ.to_le_bytes());
        bytes.push(REQUIRED_CHANNEL_COUNT);
        bytes.extend_from_slice(&(self.samples.len() as u64).to_le_bytes());
        for frame in &self.samples {
            bytes.extend_from_slice(&frame[0].to_le_bytes());
            bytes.extend_from_slice(&frame[1].to_le_bytes());
        }
        sha256_hex(&bytes)
    }
}

pub fn qualify_perceptual_stimulus_bytes(
    protocol: &FrozenPerceptualStudyProtocolV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    pack: &FrozenPerceptualStimulusPackV1,
    inputs: &[StimulusByteInputV1<'_>],
) -> Result<FrozenPerceptualStimulusByteQualificationV1, Vec<StimulusByteQualificationIssueV1>> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(StimulusByteQualificationIssueV1::InvalidProtocol);
    }
    if !render_binding.validate(protocol).is_empty() {
        issues.push(StimulusByteQualificationIssueV1::InvalidRenderSubjectBinding);
    }
    if !pack.validate(protocol, render_binding).is_empty() {
        issues.push(StimulusByteQualificationIssueV1::InvalidStimulusPack);
    }

    let protocol_sha256 = match canonical_json_sha256(protocol) {
        Ok(value) => value,
        Err(_) => {
            issues.push(StimulusByteQualificationIssueV1::ProtocolSerializationFailed);
            String::new()
        }
    };
    let render_subject_binding_sha256 = match canonical_json_sha256(render_binding) {
        Ok(value) => value,
        Err(_) => {
            issues.push(
                StimulusByteQualificationIssueV1::RenderSubjectBindingSerializationFailed,
            );
            String::new()
        }
    };
    let stimulus_pack_sha256 = match canonical_json_sha256(pack) {
        Ok(value) => value,
        Err(_) => {
            issues.push(StimulusByteQualificationIssueV1::StimulusPackSerializationFailed);
            String::new()
        }
    };

    let expected_input_count = pack.items.len().saturating_mul(2);
    if inputs.len() != expected_input_count {
        issues.push(StimulusByteQualificationIssueV1::WrongInputCount {
            found: inputs.len(),
            expected: expected_input_count,
        });
    }

    let mut pairs = Vec::with_capacity(pack.items.len());
    for pair in &pack.items {
        let baseline_input = unique_input(inputs, &pair.item_id, pair.seed, StimulusArmV1::Baseline, &mut issues);
        let intervention_input = unique_input(
            inputs,
            &pair.item_id,
            pair.seed,
            StimulusArmV1::Intervention,
            &mut issues,
        );
        let (Some(baseline_input), Some(intervention_input)) = (baseline_input, intervention_input)
        else {
            continue;
        };

        let baseline = qualify_asset(
            &pair.item_id,
            pair.seed,
            StimulusArmV1::Baseline,
            &pair.baseline,
            pair.attenuated_arm,
            baseline_input,
            &mut issues,
        );
        let intervention = qualify_asset(
            &pair.item_id,
            pair.seed,
            StimulusArmV1::Intervention,
            &pair.intervention,
            pair.attenuated_arm,
            intervention_input,
            &mut issues,
        );
        let (Some(baseline), Some(intervention)) = (baseline, intervention) else {
            continue;
        };

        let pair_delta = (baseline.measured_final_lufs - intervention.measured_final_lufs).abs();
        if !same_number(pair_delta, pair.post_match_pair_delta_lu, LOUDNESS_REMEASUREMENT_TOLERANCE_LU)
        {
            issues.push(StimulusByteQualificationIssueV1::PairDeltaMismatch {
                item_id: pair.item_id.clone(),
            });
        }
        if !pair_delta.is_finite() || pair_delta > MAX_RESIDUAL_PAIR_DELTA_LU_V1 {
            issues.push(StimulusByteQualificationIssueV1::PairDeltaExceedsTolerance {
                item_id: pair.item_id.clone(),
            });
        }

        pairs.push(QualifiedStimulusPairV1 {
            item_id: pair.item_id.clone(),
            seed: pair.seed,
            baseline,
            intervention,
            independently_measured_pair_delta_lu: pair_delta,
        });
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let mut qualification = FrozenPerceptualStimulusByteQualificationV1 {
        qualification_version: STIMULUS_BYTE_QUALIFICATION_VERSION.into(),
        protocol_sha256,
        render_subject_binding_sha256,
        stimulus_pack_sha256,
        canonical_wav_profile: CANONICAL_WAV_PROFILE_V1.into(),
        canonical_pcm_digest_profile: CANONICAL_PCM_DIGEST_PROFILE_V1.into(),
        attenuation_sample_transform_profile: ATTENUATION_SAMPLE_TRANSFORM_PROFILE_V1.into(),
        pairs,
        qualification_sha256: String::new(),
    };
    qualification.qualification_sha256 = qualification_commitment(&qualification).map_err(|_| {
        vec![StimulusByteQualificationIssueV1::QualificationSerializationFailed]
    })?;
    Ok(qualification)
}

pub fn qualification_commitment(
    qualification: &FrozenPerceptualStimulusByteQualificationV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = qualification.clone();
    unsigned.qualification_sha256.clear();
    canonical_json_sha256(&unsigned)
}

fn unique_input<'a>(
    inputs: &'a [StimulusByteInputV1<'a>],
    item_id: &str,
    seed: u64,
    arm: StimulusArmV1,
    issues: &mut Vec<StimulusByteQualificationIssueV1>,
) -> Option<&'a StimulusByteInputV1<'a>> {
    let mut matches = inputs
        .iter()
        .filter(|input| input.item_id == item_id && input.arm == arm);
    let first = matches.next();
    if matches.next().is_some() {
        issues.push(StimulusByteQualificationIssueV1::DuplicateInput {
            item_id: item_id.into(),
            arm,
        });
        return None;
    }
    let Some(input) = first else {
        issues.push(StimulusByteQualificationIssueV1::MissingInput {
            item_id: item_id.into(),
            arm,
        });
        return None;
    };
    if input.seed != seed {
        issues.push(StimulusByteQualificationIssueV1::InputSeedMismatch {
            item_id: item_id.into(),
            arm,
        });
        return None;
    }
    Some(input)
}

fn qualify_asset(
    item_id: &str,
    seed: u64,
    arm: StimulusArmV1,
    manifest: &StimulusAudioAssetV1,
    attenuated_arm: Option<StimulusArmV1>,
    input: &StimulusByteInputV1<'_>,
    issues: &mut Vec<StimulusByteQualificationIssueV1>,
) -> Option<QualifiedStimulusAssetV1> {
    let source_file_sha256 = sha256_hex(input.source_wav);
    let output_file_sha256 = sha256_hex(input.output_wav);
    if source_file_sha256 != manifest.source_sha256 {
        issues.push(StimulusByteQualificationIssueV1::SourceFileDigestMismatch {
            item_id: item_id.into(),
            arm,
        });
    }
    if output_file_sha256 != manifest.output_sha256 {
        issues.push(StimulusByteQualificationIssueV1::OutputFileDigestMismatch {
            item_id: item_id.into(),
            arm,
        });
    }

    let source = match decode_canonical_wav(input.source_wav) {
        Ok(value) => value,
        Err(issue) => {
            issues.push(StimulusByteQualificationIssueV1::InvalidSourceWav {
                item_id: item_id.into(),
                arm,
                issue,
            });
            return None;
        }
    };
    let output = match decode_canonical_wav(input.output_wav) {
        Ok(value) => value,
        Err(issue) => {
            issues.push(StimulusByteQualificationIssueV1::InvalidOutputWav {
                item_id: item_id.into(),
                arm,
                issue,
            });
            return None;
        }
    };

    if source.frame_count() != manifest.frame_count
        || manifest.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ
        || manifest.channel_count != REQUIRED_CHANNEL_COUNT
    {
        issues.push(StimulusByteQualificationIssueV1::SourceGeometryMismatch {
            item_id: item_id.into(),
            arm,
        });
    }
    if output.frame_count() != manifest.frame_count
        || output.frame_count() != source.frame_count()
        || manifest.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ
        || manifest.channel_count != REQUIRED_CHANNEL_COUNT
    {
        issues.push(StimulusByteQualificationIssueV1::OutputGeometryMismatch {
            item_id: item_id.into(),
            arm,
        });
    }

    if !manifest.applied_gain_db.is_finite() {
        issues.push(StimulusByteQualificationIssueV1::NonFiniteGain {
            item_id: item_id.into(),
            arm,
        });
        return None;
    }
    if manifest.applied_gain_db > NUMERIC_TOLERANCE_DB {
        issues.push(StimulusByteQualificationIssueV1::PositiveGain {
            item_id: item_id.into(),
            arm,
        });
    }

    let source_frames = source.as_f32_frames();
    let output_frames = output.as_f32_frames();
    let measured_initial_lufs = measure_lufs(&source_frames, REQUIRED_SAMPLE_RATE_HZ).integrated as f64;
    let measured_final_lufs = measure_lufs(&output_frames, REQUIRED_SAMPLE_RATE_HZ).integrated as f64;
    if !measured_initial_lufs.is_finite() || !measured_final_lufs.is_finite() {
        issues.push(StimulusByteQualificationIssueV1::NonFiniteMeasuredLoudness {
            item_id: item_id.into(),
            arm,
        });
    } else {
        if !same_number(
            measured_initial_lufs,
            manifest.initial_integrated_lufs,
            LOUDNESS_REMEASUREMENT_TOLERANCE_LU,
        ) {
            issues.push(StimulusByteQualificationIssueV1::InitialLoudnessMismatch {
                item_id: item_id.into(),
                arm,
            });
        }
        if !same_number(
            measured_final_lufs,
            manifest.final_integrated_lufs,
            LOUDNESS_REMEASUREMENT_TOLERANCE_LU,
        ) {
            issues.push(StimulusByteQualificationIssueV1::FinalLoudnessMismatch {
                item_id: item_id.into(),
                arm,
            });
        }
    }

    let should_attenuate = attenuated_arm == Some(arm);
    let expected_output = if should_attenuate {
        match apply_registered_gain(&source, manifest.applied_gain_db) {
            Ok(value) => value,
            Err(()) => {
                issues.push(StimulusByteQualificationIssueV1::UnexpectedClipping {
                    item_id: item_id.into(),
                    arm,
                });
                return None;
            }
        }
    } else {
        source.clone()
    };
    let reconstructed_output_wav = encode_canonical_wav(&expected_output);
    let reconstructed_output_file_sha256 = sha256_hex(&reconstructed_output_wav);
    let reconstructed_output_pcm_sha256 = expected_output.pcm_sha256();
    let source_output_file_identical = input.source_wav == input.output_wav;
    let reconstructed_output_exact = reconstructed_output_wav == input.output_wav;

    if should_attenuate {
        if !reconstructed_output_exact {
            issues.push(StimulusByteQualificationIssueV1::ReconstructedOutputMismatch {
                item_id: item_id.into(),
                arm,
            });
        }
    } else if !source_output_file_identical {
        issues.push(StimulusByteQualificationIssueV1::UnchangedArmBytesDiffer {
            item_id: item_id.into(),
            arm,
        });
    }

    Some(QualifiedStimulusAssetV1 {
        item_id: item_id.into(),
        seed,
        arm,
        source_file_sha256,
        source_pcm_sha256: source.pcm_sha256(),
        output_file_sha256,
        output_pcm_sha256: output.pcm_sha256(),
        reconstructed_output_file_sha256,
        reconstructed_output_pcm_sha256,
        sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
        channel_count: REQUIRED_CHANNEL_COUNT,
        frame_count: source.frame_count(),
        applied_gain_db: manifest.applied_gain_db,
        measured_initial_lufs,
        measured_final_lufs,
        source_output_file_identical,
        reconstructed_output_exact,
    })
}

fn decode_canonical_wav(bytes: &[u8]) -> Result<CanonicalPcm16Stereo, CanonicalWavIssueV1> {
    if bytes.len() < WAV_HEADER_BYTES {
        return Err(CanonicalWavIssueV1::TooShort);
    }
    if &bytes[0..4] != b"RIFF" {
        return Err(CanonicalWavIssueV1::WrongRiffTag);
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(CanonicalWavIssueV1::WrongWaveTag);
    }
    let riff_size = le_u32(bytes, 4) as usize;
    if riff_size.checked_add(8) != Some(bytes.len()) {
        return Err(CanonicalWavIssueV1::WrongRiffLength);
    }
    if &bytes[12..16] != b"fmt " {
        return Err(CanonicalWavIssueV1::WrongFmtTag);
    }
    if le_u32(bytes, 16) != 16 {
        return Err(CanonicalWavIssueV1::WrongFmtChunkSize);
    }
    if le_u16(bytes, 20) != AUDIO_FORMAT_PCM {
        return Err(CanonicalWavIssueV1::WrongAudioFormat);
    }
    if le_u16(bytes, 22) != REQUIRED_CHANNEL_COUNT as u16 {
        return Err(CanonicalWavIssueV1::WrongChannelCount);
    }
    if le_u32(bytes, 24) != REQUIRED_SAMPLE_RATE_HZ {
        return Err(CanonicalWavIssueV1::WrongSampleRate);
    }
    if le_u32(bytes, 28) != BYTE_RATE {
        return Err(CanonicalWavIssueV1::WrongByteRate);
    }
    if le_u16(bytes, 32) != BLOCK_ALIGN {
        return Err(CanonicalWavIssueV1::WrongBlockAlign);
    }
    if le_u16(bytes, 34) != BITS_PER_SAMPLE {
        return Err(CanonicalWavIssueV1::WrongBitsPerSample);
    }
    if &bytes[36..40] != b"data" {
        return Err(CanonicalWavIssueV1::WrongDataTag);
    }
    let data_bytes = le_u32(bytes, 40) as usize;
    if data_bytes.checked_add(WAV_HEADER_BYTES) != Some(bytes.len()) {
        return Err(CanonicalWavIssueV1::WrongDataLength);
    }
    if data_bytes % PCM_BYTES_PER_FRAME != 0 {
        return Err(CanonicalWavIssueV1::NonIntegralFrameCount);
    }

    let mut samples = Vec::with_capacity(data_bytes / PCM_BYTES_PER_FRAME);
    for chunk in bytes[WAV_HEADER_BYTES..].chunks_exact(PCM_BYTES_PER_FRAME) {
        samples.push([
            i16::from_le_bytes([chunk[0], chunk[1]]),
            i16::from_le_bytes([chunk[2], chunk[3]]),
        ]);
    }
    Ok(CanonicalPcm16Stereo { samples })
}

fn encode_canonical_wav(pcm: &CanonicalPcm16Stereo) -> Vec<u8> {
    let data_bytes = pcm
        .samples
        .len()
        .checked_mul(PCM_BYTES_PER_FRAME)
        .and_then(|value| u32::try_from(value).ok())
        .expect("P1 stimulus PCM must fit a canonical RIFF/WAVE data chunk");
    let riff_size = 36u32
        .checked_add(data_bytes)
        .expect("P1 stimulus RIFF/WAVE length must fit u32");
    let mut bytes = Vec::with_capacity(WAV_HEADER_BYTES + data_bytes as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&AUDIO_FORMAT_PCM.to_le_bytes());
    bytes.extend_from_slice(&(REQUIRED_CHANNEL_COUNT as u16).to_le_bytes());
    bytes.extend_from_slice(&REQUIRED_SAMPLE_RATE_HZ.to_le_bytes());
    bytes.extend_from_slice(&BYTE_RATE.to_le_bytes());
    bytes.extend_from_slice(&BLOCK_ALIGN.to_le_bytes());
    bytes.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for frame in &pcm.samples {
        bytes.extend_from_slice(&frame[0].to_le_bytes());
        bytes.extend_from_slice(&frame[1].to_le_bytes());
    }
    bytes
}

fn apply_registered_gain(
    source: &CanonicalPcm16Stereo,
    gain_db: f64,
) -> Result<CanonicalPcm16Stereo, ()> {
    if !gain_db.is_finite() || gain_db > NUMERIC_TOLERANCE_DB {
        return Err(());
    }
    let gain = 10.0f64.powf(gain_db / 20.0);
    let mut samples = Vec::with_capacity(source.samples.len());
    for frame in &source.samples {
        let mut output = [0i16; 2];
        for channel in 0..2 {
            let normalized = frame[channel] as f64 / 32768.0;
            let transformed = normalized * gain;
            if transformed.abs() > 1.0 + CLIP_EPSILON {
                return Err(());
            }
            output[channel] = (transformed.clamp(-1.0, 1.0) * i16::MAX as f64) as i16;
        }
        samples.push(output);
    }
    Ok(CanonicalPcm16Stereo { samples })
}

fn same_number(left: f64, right: f64, tolerance: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= tolerance
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_pcm() -> CanonicalPcm16Stereo {
        CanonicalPcm16Stereo {
            samples: vec![
                [0, 0],
                [10_000, -10_000],
                [20_000, -20_000],
                [i16::MAX, -32_000],
            ],
        }
    }

    #[test]
    fn canonical_wav_round_trip_preserves_exact_pcm_and_file_bytes() {
        let pcm = fixture_pcm();
        let bytes = encode_canonical_wav(&pcm);
        let decoded = decode_canonical_wav(&bytes).unwrap();
        assert_eq!(decoded, pcm);
        assert_eq!(encode_canonical_wav(&decoded), bytes);
        assert_eq!(decoded.pcm_sha256(), pcm.pcm_sha256());
    }

    #[test]
    fn canonical_wav_rejects_appended_or_ancillary_bytes() {
        let mut bytes = encode_canonical_wav(&fixture_pcm());
        bytes.push(0);
        assert_eq!(
            decode_canonical_wav(&bytes),
            Err(CanonicalWavIssueV1::WrongRiffLength)
        );
    }

    #[test]
    fn registered_negative_gain_reconstructs_exact_samples_without_dither() {
        let source = fixture_pcm();
        let output = apply_registered_gain(&source, -6.0).unwrap();
        let gain = 10.0f64.powf(-6.0 / 20.0);
        for (source_frame, output_frame) in source.samples.iter().zip(&output.samples) {
            for channel in 0..2 {
                let expected =
                    ((source_frame[channel] as f64 / 32768.0) * gain * i16::MAX as f64) as i16;
                assert_eq!(output_frame[channel], expected);
            }
        }
    }

    #[test]
    fn one_sample_change_changes_both_pcm_and_file_identity() {
        let source = fixture_pcm();
        let source_bytes = encode_canonical_wav(&source);
        let mut changed = source.clone();
        changed.samples[1][0] += 1;
        let changed_bytes = encode_canonical_wav(&changed);
        assert_ne!(source.pcm_sha256(), changed.pcm_sha256());
        assert_ne!(sha256_hex(&source_bytes), sha256_hex(&changed_bytes));
    }

    #[test]
    fn independent_loudness_measurement_tracks_registered_constant_gain() {
        let sample_count = 22_050usize;
        let samples = (0..sample_count)
            .map(|index| {
                let phase = std::f64::consts::TAU * 440.0 * index as f64
                    / REQUIRED_SAMPLE_RATE_HZ as f64;
                let sample = (phase.sin() * 0.2 * i16::MAX as f64) as i16;
                [sample, sample]
            })
            .collect();
        let source = CanonicalPcm16Stereo { samples };
        let output = apply_registered_gain(&source, -6.0).unwrap();
        let source_lufs = measure_lufs(&source.as_f32_frames(), REQUIRED_SAMPLE_RATE_HZ).integrated;
        let output_lufs = measure_lufs(&output.as_f32_frames(), REQUIRED_SAMPLE_RATE_HZ).integrated;
        assert!(((output_lufs - source_lufs) + 6.0).abs() < 0.02);
    }
}
