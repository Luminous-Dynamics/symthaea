// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed identity for acoustic hypervector spaces.
//!
//! An HV's bit width does not describe what its bits mean. This module makes
//! the feature frontend and projection semantics part of the contract carried
//! beside persisted vectors and prototype artifacts.
//!
//! This is an integrity primitive: it does not itself improve recognition,
//! migrate legacy prototypes, or prove that two encoders are numerically
//! equivalent. Callers must persist the descriptor and require exact
//! compatibility before comparing or consuming vectors.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Current wire/schema version for AudioEncoderSpace.
pub const AUDIO_ENCODER_SPACE_SCHEMA_VERSION: u16 = 1;

/// Acoustic features represented by an encoder space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticFeatureFamily {
    /// Log-mel filterbank energies.
    LogMel,
    /// Log-mel energies, first derivatives, and second derivatives.
    LogMelWithDeltas,
    /// Legacy energy, zero-crossing-rate, and coarse band heuristics.
    EnergyZcrBands,
}

/// Temporal semantics applied after the per-frame feature encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ProjectionSemantics {
    /// One vector represents one acoustic frame.
    DirectFrame,
    /// One vector represents a centered, position-encoded temporal window.
    CenteredWindow {
        /// Number of frames on either side of the center frame.
        radius_frames: u16,
        /// Rotation step assigned to adjacent temporal positions.
        rotation_step_bits: i32,
    },
    /// Legacy LTC-state serialization followed by a hash-based HV mapping.
    LtcStateHash,
}

/// A complete, versioned identity for the semantics of an acoustic HV.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioEncoderSpace {
    /// Descriptor serialization version.
    pub schema_version: u16,
    /// Stable producer identifier, not a human-facing model nickname.
    pub encoder_id: String,
    /// Immutable algorithm revision or source revision identifier.
    pub algorithm_revision: String,
    /// Canonical sample rate presented to the frontend after resampling.
    pub sample_rate_hz: u32,
    /// Frame length in samples at the canonical sample rate.
    pub frame_len_samples: u32,
    /// Frame hop in samples at the canonical sample rate.
    pub hop_len_samples: u32,
    /// Feature family before vector projection.
    pub feature_family: AcousticFeatureFamily,
    /// Number of mel channels; absent only for non-mel feature families.
    pub mel_channels: Option<u16>,
    /// FFT length; absent only for non-FFT feature families.
    pub fft_size: Option<u32>,
    /// Minimum mel-filter frequency in millihertz.
    pub f_min_millihz: Option<u32>,
    /// Maximum mel-filter frequency in millihertz.
    pub f_max_millihz: Option<u32>,
    /// Exact normalization/feature transform identifier and revision.
    pub normalization_id: String,
    /// Explicit channel layout/downmix policy identifier.
    pub channel_policy_id: String,
    /// Explicit resampler implementation/configuration identifier.
    pub resampler_id: String,
    /// Projection and temporal-context semantics.
    pub projection: ProjectionSemantics,
    /// Output vector width in bits.
    pub vector_width_bits: u32,
}

impl AudioEncoderSpace {
    /// Validate invariants required for an encoder-space descriptor.
    ///
    /// Errors are stable field names so callers can report them in receipts
    /// without depending on prose wording.
    pub fn validate(&self) -> Result<(), Vec<&'static str>> {
        let mut invalid = Vec::new();

        if self.schema_version != AUDIO_ENCODER_SPACE_SCHEMA_VERSION {
            invalid.push("schema_version");
        }
        if self.encoder_id.trim().is_empty() {
            invalid.push("encoder_id");
        }
        if self.algorithm_revision.trim().is_empty() {
            invalid.push("algorithm_revision");
        }
        if self.sample_rate_hz == 0 {
            invalid.push("sample_rate_hz");
        }
        if self.frame_len_samples == 0 {
            invalid.push("frame_len_samples");
        }
        if self.hop_len_samples == 0 || self.hop_len_samples > self.frame_len_samples {
            invalid.push("hop_len_samples");
        }
        if self.vector_width_bits == 0 {
            invalid.push("vector_width_bits");
        }
        if self.normalization_id.trim().is_empty() {
            invalid.push("normalization_id");
        }
        if self.channel_policy_id.trim().is_empty() {
            invalid.push("channel_policy_id");
        }
        if self.resampler_id.trim().is_empty() {
            invalid.push("resampler_id");
        }

        match self.feature_family {
            AcousticFeatureFamily::LogMel | AcousticFeatureFamily::LogMelWithDeltas => {
                if self.mel_channels.is_none_or(|value| value == 0) {
                    invalid.push("mel_channels");
                }
                if self.fft_size.is_none_or(|value| value == 0) {
                    invalid.push("fft_size");
                }
                if self.f_min_millihz.is_none() {
                    invalid.push("f_min_millihz");
                }
                if self.f_max_millihz.is_none() {
                    invalid.push("f_max_millihz");
                }
                if matches!(
                    (self.f_min_millihz, self.f_max_millihz),
                    (Some(min), Some(max)) if min >= max
                ) {
                    invalid.push("frequency_range");
                }
            }
            AcousticFeatureFamily::EnergyZcrBands => {
                if self.mel_channels.is_some()
                    || self.fft_size.is_some()
                    || self.f_min_millihz.is_some()
                    || self.f_max_millihz.is_some()
                {
                    invalid.push("non_mel_frontend_fields");
                }
            }
        }

        if let ProjectionSemantics::CenteredWindow {
            radius_frames,
            rotation_step_bits,
        } = self.projection
            && (radius_frames == 0 || rotation_step_bits == 0)
        {
            invalid.push("temporal_window");
        }

        if invalid.is_empty() {
            Ok(())
        } else {
            Err(invalid)
        }
    }

    /// Return a deterministic BLAKE3 fingerprint of the canonical struct
    /// serialization. Struct field order and enum tags are part of schema v1.
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        let canonical = serde_json::to_vec(self)?;
        Ok(blake3::hash(&canonical).to_hex().to_string())
    }

    /// List every semantic field that differs from another descriptor.
    ///
    /// This intentionally compares the complete descriptor, not just the
    /// output width or crate version.
    pub fn mismatch_fields(&self, other: &Self) -> Vec<&'static str> {
        let mut mismatches = Vec::new();

        macro_rules! compare_field {
            ($field:ident) => {
                if self.$field != other.$field {
                    mismatches.push(stringify!($field));
                }
            };
        }

        compare_field!(schema_version);
        compare_field!(encoder_id);
        compare_field!(algorithm_revision);
        compare_field!(sample_rate_hz);
        compare_field!(frame_len_samples);
        compare_field!(hop_len_samples);
        compare_field!(feature_family);
        compare_field!(mel_channels);
        compare_field!(fft_size);
        compare_field!(f_min_millihz);
        compare_field!(f_max_millihz);
        compare_field!(normalization_id);
        compare_field!(channel_policy_id);
        compare_field!(resampler_id);
        compare_field!(projection);
        compare_field!(vector_width_bits);

        mismatches
    }

    /// Require both descriptors to be valid and semantically identical.
    pub fn require_compatible(&self, actual: &Self) -> Result<(), SpaceCompatibilityError> {
        let expected_invalid = self.validate().err().unwrap_or_default();
        let actual_invalid = actual.validate().err().unwrap_or_default();
        let mismatches = self.mismatch_fields(actual);

        if expected_invalid.is_empty() && actual_invalid.is_empty() && mismatches.is_empty() {
            return Ok(());
        }

        let expected_digest = self.digest().ok();
        let actual_digest = actual.digest().ok();
        Err(SpaceCompatibilityError {
            expected_invalid,
            actual_invalid,
            mismatches,
            expected_digest,
            actual_digest,
        })
    }
}

/// Why two acoustic encoder spaces cannot be used interchangeably.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceCompatibilityError {
    /// Invalid fields in the expected descriptor.
    pub expected_invalid: Vec<&'static str>,
    /// Invalid fields in the actual descriptor.
    pub actual_invalid: Vec<&'static str>,
    /// Semantic fields that differ.
    pub mismatches: Vec<&'static str>,
    /// Digest of the expected descriptor when serialization succeeds.
    pub expected_digest: Option<String>,
    /// Digest of the actual descriptor when serialization succeeds.
    pub actual_digest: Option<String>,
}

impl fmt::Display for SpaceCompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "incompatible acoustic encoder spaces (mismatches: {:?}, expected invalid: {:?}, actual invalid: {:?})",
            self.mismatches, self.expected_invalid, self.actual_invalid
        )
    }
}

impl std::error::Error for SpaceCompatibilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        feature_family: AcousticFeatureFamily,
        projection: ProjectionSemantics,
    ) -> AudioEncoderSpace {
        let (mel_channels, fft_size, f_min_millihz, f_max_millihz) =
            if feature_family == AcousticFeatureFamily::EnergyZcrBands {
                (None, None, None, None)
            } else {
                (Some(40), Some(512), Some(80_000), Some(7_600_000))
            };

        AudioEncoderSpace {
            schema_version: AUDIO_ENCODER_SPACE_SCHEMA_VERSION,
            encoder_id: "symthaea-stt.direct-mel".to_string(),
            algorithm_revision: "git:fixture-revision".to_string(),
            sample_rate_hz: 16_000,
            frame_len_samples: 400,
            hop_len_samples: 160,
            feature_family,
            mel_channels,
            fft_size,
            f_min_millihz,
            f_max_millihz,
            normalization_id: "log-mel-ln-v1".to_string(),
            channel_policy_id: "mono-explicit-v1".to_string(),
            resampler_id: "sinc-polyphase-v1".to_string(),
            projection,
            vector_width_bits: 16_384,
        }
    }

    #[test]
    fn identical_descriptors_have_stable_digests_and_are_compatible() {
        let left = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::DirectFrame,
        );
        let right: AudioEncoderSpace =
            serde_json::from_str(&serde_json::to_string(&left).unwrap()).unwrap();

        assert_eq!(left.digest().unwrap(), right.digest().unwrap());
        assert!(left.require_compatible(&right).is_ok());
    }

    #[test]
    fn raw_and_temporal_spaces_are_not_compatible() {
        let raw = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::DirectFrame,
        );
        let temporal = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::CenteredWindow {
                radius_frames: 3,
                rotation_step_bits: 256,
            },
        );

        let error = raw.require_compatible(&temporal).unwrap_err();
        assert_eq!(error.mismatches, vec!["projection"]);
        assert_ne!(raw.digest().unwrap(), temporal.digest().unwrap());
    }

    #[test]
    fn legacy_ltc_hash_is_not_direct_log_mel() {
        let direct = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::DirectFrame,
        );
        let legacy = descriptor(
            AcousticFeatureFamily::EnergyZcrBands,
            ProjectionSemantics::LtcStateHash,
        );

        let error = direct.require_compatible(&legacy).unwrap_err();
        assert!(error.mismatches.contains(&"feature_family"));
        assert!(error.mismatches.contains(&"projection"));
        assert!(error.mismatches.contains(&"mel_channels"));
    }

    #[test]
    fn signal_geometry_changes_encoder_identity() {
        let left = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::DirectFrame,
        );
        let mut right = left.clone();
        right.sample_rate_hz = 48_000;

        let error = left.require_compatible(&right).unwrap_err();
        assert_eq!(error.mismatches, vec!["sample_rate_hz"]);
    }

    #[test]
    fn incomplete_descriptors_fail_closed() {
        let mut invalid = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::DirectFrame,
        );
        invalid.encoder_id.clear();
        invalid.mel_channels = None;

        let error = invalid.require_compatible(&invalid).unwrap_err();
        assert!(error.expected_invalid.contains(&"encoder_id"));
        assert!(error.expected_invalid.contains(&"mel_channels"));
        assert!(error.actual_invalid.contains(&"encoder_id"));
    }

    #[test]
    fn zero_width_temporal_window_is_rejected() {
        let mut invalid = descriptor(
            AcousticFeatureFamily::LogMel,
            ProjectionSemantics::CenteredWindow {
                radius_frames: 0,
                rotation_step_bits: 256,
            },
        );
        invalid.frame_len_samples = 0;

        let errors = invalid.validate().unwrap_err();
        assert!(errors.contains(&"temporal_window"));
        assert!(errors.contains(&"frame_len_samples"));
    }
}
