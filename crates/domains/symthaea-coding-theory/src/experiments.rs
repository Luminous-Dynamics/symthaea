// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Seeded end-to-end experiments connecting channel errors to decoder outcomes.

use std::fmt;

use crate::{
    channel::{
        BinarySymmetricChannel, ChannelError, DeterministicRng,
        FixedCountErasureChannel, FixedCountErrataChannel, SymbolErasureChannel,
        Transmission,
    },
    hamming::{
        Hamming84Status, hamming84_decode, hamming84_encode_checked,
    },
    interoperability::ReedSolomonProfile,
    reed_solomon::{
        ReedSolomon, ReedSolomonConfig, ReedSolomonError,
    },
};

/// Error while configuring or running a coding experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    Channel(ChannelError),
    ReedSolomon(ReedSolomonError),
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Channel(error) => error.fmt(f),
            Self::ReedSolomon(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ExperimentError {}

impl From<ChannelError> for ExperimentError {
    fn from(value: ChannelError) -> Self {
        Self::Channel(value)
    }
}

impl From<ReedSolomonError> for ExperimentError {
    fn from(value: ReedSolomonError) -> Self {
        Self::ReedSolomon(value)
    }
}

/// Erasure process used by a Reed-Solomon evidence run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolErasureModel {
    Independent(SymbolErasureChannel),
    FixedCount(FixedCountErasureChannel),
}

impl SymbolErasureModel {
    fn transmit(
        self,
        symbols: &[u8],
        rng: &mut DeterministicRng,
    ) -> Result<Transmission<Vec<u8>>, ChannelError> {
        match self {
            Self::Independent(channel) => Ok(channel.transmit(symbols, rng)),
            Self::FixedCount(channel) => channel.transmit(symbols, rng),
        }
    }
}

/// Configuration for a reproducible Reed-Solomon erasure experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedSolomonErasureExperiment {
    pub frames: usize,
    pub message_symbols: usize,
    pub seed: u64,
    pub config: ReedSolomonConfig,
    pub channel: SymbolErasureModel,
}

/// Aggregate evidence from a Reed-Solomon erasure experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedSolomonErasureExperimentReport {
    pub frames: usize,
    pub message_symbols_per_frame: usize,
    pub parity_symbols: usize,
    pub codeword_symbols_per_frame: usize,
    pub seed: u64,
    pub channel: SymbolErasureModel,
    pub source_symbols: usize,
    pub transmitted_symbols: usize,
    pub channel_erasures: usize,
    pub clean_frames: usize,
    pub recovered_frames: usize,
    pub over_capacity_frames: usize,
    pub verification_failure_frames: usize,
    pub other_decoder_failure_frames: usize,
    pub wrong_message_frames: usize,
}

impl ReedSolomonErasureExperimentReport {
    #[must_use]
    pub fn erasure_rate(self) -> f64 {
        if self.transmitted_symbols == 0 {
            return 0.0;
        }
        self.channel_erasures as f64 / self.transmitted_symbols as f64
    }

    #[must_use]
    pub fn recovery_rate(self) -> f64 {
        let erased_frames = self.frames.saturating_sub(self.clean_frames);
        if erased_frames == 0 {
            return 1.0;
        }
        self.recovered_frames as f64 / erased_frames as f64
    }

    #[must_use]
    pub const fn accounted_frames(self) -> usize {
        self.clean_frames
            + self.recovered_frames
            + self.over_capacity_frames
            + self.verification_failure_frames
            + self.other_decoder_failure_frames
    }
}

/// Run a deterministic known-erasure Reed-Solomon experiment.
pub fn run_reed_solomon_erasure_experiment(
    experiment: ReedSolomonErasureExperiment,
) -> Result<ReedSolomonErasureExperimentReport, ExperimentError> {
    let codec = ReedSolomon::new(experiment.config)?;
    let codeword_symbols = codec.encoded_len(experiment.message_symbols)?;
    let mut rng = DeterministicRng::new(experiment.seed);
    let mut report = ReedSolomonErasureExperimentReport {
        frames: experiment.frames,
        message_symbols_per_frame: experiment.message_symbols,
        parity_symbols: experiment.config.parity_symbols,
        codeword_symbols_per_frame: codeword_symbols,
        seed: experiment.seed,
        channel: experiment.channel,
        source_symbols: experiment.frames.saturating_mul(experiment.message_symbols),
        transmitted_symbols: experiment.frames.saturating_mul(codeword_symbols),
        channel_erasures: 0,
        clean_frames: 0,
        recovered_frames: 0,
        over_capacity_frames: 0,
        verification_failure_frames: 0,
        other_decoder_failure_frames: 0,
        wrong_message_frames: 0,
    };

    for _ in 0..experiment.frames {
        let message = (0..experiment.message_symbols)
            .map(|_| rng.next_u8())
            .collect::<Vec<_>>();
        let codeword = codec.encode(&message)?;
        let transmission = experiment.channel.transmit(&codeword, &mut rng)?;
        report.channel_erasures += transmission.corrupted_positions.len();

        match codec.decode_erasures(
            &transmission.received,
            &transmission.corrupted_positions,
        ) {
            Ok(decoded) => {
                if transmission.corrupted_positions.is_empty() {
                    report.clean_frames += 1;
                } else {
                    report.recovered_frames += 1;
                }
                if decoded.message != message {
                    report.wrong_message_frames += 1;
                }
            }
            Err(ReedSolomonError::TooManyErasures { .. }) => {
                report.over_capacity_frames += 1;
            }
            Err(ReedSolomonError::CorrectionVerificationFailed) => {
                report.verification_failure_frames += 1;
            }
            Err(_) => {
                report.other_decoder_failure_frames += 1;
            }
        }
    }

    Ok(report)
}

/// Configuration for an exact-count mixed Reed-Solomon errata campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedSolomonErrataExperiment {
    pub frames: usize,
    pub message_symbols: usize,
    pub seed: u64,
    pub config: ReedSolomonConfig,
    pub channel: FixedCountErrataChannel,
}

impl ReedSolomonErrataExperiment {
    /// Stable one-line preregistration manifest for logs and evidence bundles.
    pub fn manifest(self) -> Result<String, ReedSolomonError> {
        let codec = ReedSolomon::new(self.config)?;
        let codeword_symbols = codec.encoded_len(self.message_symbols)?;
        Ok(format!(
            "symthaea-coding-evidence-v1;kind=rs-fixed-errata;profile={};k={};n={};frames={};seed={:016x};errors={};erasures={};placeholder={:02x}",
            ReedSolomonProfile::aes(self.config).identifier(),
            self.message_symbols,
            codeword_symbols,
            self.frames,
            self.seed,
            self.channel.error_count(),
            self.channel.erasure_count(),
            self.channel.erasure_placeholder(),
        ))
    }
}

/// Aggregate evidence from an exact-count mixed errata campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedSolomonErrataExperimentReport {
    pub frames: usize,
    pub message_symbols_per_frame: usize,
    pub parity_symbols: usize,
    pub codeword_symbols_per_frame: usize,
    pub seed: u64,
    pub channel: FixedCountErrataChannel,
    pub within_guaranteed_capacity: bool,
    pub source_symbols: usize,
    pub transmitted_symbols: usize,
    pub channel_errors: usize,
    pub channel_erasures: usize,
    pub exact_recovery_frames: usize,
    pub wrong_message_frames: usize,
    pub rejected_too_many_errata_frames: usize,
    pub verification_failure_frames: usize,
    pub other_decoder_failure_frames: usize,
}

impl ReedSolomonErrataExperimentReport {
    #[must_use]
    pub const fn accounted_frames(self) -> usize {
        self.exact_recovery_frames
            + self.wrong_message_frames
            + self.rejected_too_many_errata_frames
            + self.verification_failure_frames
            + self.other_decoder_failure_frames
    }

    #[must_use]
    pub fn exact_recovery_rate(self) -> f64 {
        if self.frames == 0 {
            return 1.0;
        }
        self.exact_recovery_frames as f64 / self.frames as f64
    }
}

/// Run a deterministic exact-count mixed error/erasure campaign.
pub fn run_reed_solomon_errata_experiment(
    experiment: ReedSolomonErrataExperiment,
) -> Result<ReedSolomonErrataExperimentReport, ExperimentError> {
    let codec = ReedSolomon::new(experiment.config)?;
    let codeword_symbols = codec.encoded_len(experiment.message_symbols)?;
    if experiment
        .channel
        .error_count()
        .checked_add(experiment.channel.erasure_count())
        .map_or(true, |count| count > codeword_symbols)
    {
        return Err(ChannelError::TooManyRequestedErrata {
            errors: experiment.channel.error_count(),
            erasures: experiment.channel.erasure_count(),
            symbols: codeword_symbols,
        }
        .into());
    }

    let errata_weight = experiment
        .channel
        .error_count()
        .saturating_mul(2)
        .saturating_add(experiment.channel.erasure_count());
    let mut rng = DeterministicRng::new(experiment.seed);
    let mut report = ReedSolomonErrataExperimentReport {
        frames: experiment.frames,
        message_symbols_per_frame: experiment.message_symbols,
        parity_symbols: experiment.config.parity_symbols,
        codeword_symbols_per_frame: codeword_symbols,
        seed: experiment.seed,
        channel: experiment.channel,
        within_guaranteed_capacity: errata_weight <= experiment.config.parity_symbols,
        source_symbols: experiment.frames.saturating_mul(experiment.message_symbols),
        transmitted_symbols: experiment.frames.saturating_mul(codeword_symbols),
        channel_errors: 0,
        channel_erasures: 0,
        exact_recovery_frames: 0,
        wrong_message_frames: 0,
        rejected_too_many_errata_frames: 0,
        verification_failure_frames: 0,
        other_decoder_failure_frames: 0,
    };

    for _ in 0..experiment.frames {
        let message = (0..experiment.message_symbols)
            .map(|_| rng.next_u8())
            .collect::<Vec<_>>();
        let codeword = codec.encode(&message)?;
        let transmission = experiment.channel.transmit(&codeword, &mut rng)?;
        report.channel_errors += transmission.error_positions.len();
        report.channel_erasures += transmission.erasure_positions.len();

        match codec.decode_with_erasures(
            &transmission.received,
            &transmission.erasure_positions,
        ) {
            Ok(decoded) => {
                if decoded.message == message && decoded.corrected_codeword == codeword {
                    report.exact_recovery_frames += 1;
                } else {
                    report.wrong_message_frames += 1;
                }
            }
            Err(ReedSolomonError::TooManyErrata { .. }) => {
                report.rejected_too_many_errata_frames += 1;
            }
            Err(ReedSolomonError::CorrectionVerificationFailed) => {
                report.verification_failure_frames += 1;
            }
            Err(_) => {
                report.other_decoder_failure_frames += 1;
            }
        }
    }

    Ok(report)
}

/// Configuration for a reproducible Hamming(8,4) channel experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hamming84Experiment {
    /// Number of independently generated four-bit frames.
    pub frames: usize,
    /// Initial PRNG seed recorded in the evidence report.
    pub seed: u64,
    /// Binary symmetric channel applied to each eight-bit codeword.
    pub channel: BinarySymmetricChannel,
}

/// Aggregate evidence from a Hamming(8,4) channel experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hamming84ExperimentReport {
    /// Number of source frames.
    pub frames: usize,
    /// Seed that fully determines source words and channel draws.
    pub seed: u64,
    /// Total source payload bits.
    pub source_bits: usize,
    /// Number of physical codeword bits flipped by the channel.
    pub channel_bit_errors: usize,
    /// Frames accepted without correction.
    pub clean_frames: usize,
    /// Frames from which one bit was corrected.
    pub corrected_frames: usize,
    /// Frames rejected after SECDED detected a double-error pattern.
    pub detected_uncorrectable_frames: usize,
    /// Frames that returned a payload different from the source payload.
    pub wrong_payload_frames: usize,
    /// Differing payload bits among frames that returned a payload.
    pub wrong_payload_bits: usize,
}

impl Hamming84ExperimentReport {
    /// Observed channel bit-error rate over transmitted codeword bits.
    #[must_use]
    pub fn channel_bit_error_rate(self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        self.channel_bit_errors as f64 / self.frames.saturating_mul(8) as f64
    }

    /// Observed wrong-payload frame rate.
    #[must_use]
    pub fn wrong_payload_frame_rate(self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        self.wrong_payload_frames as f64 / self.frames as f64
    }
}

/// Run a deterministic end-to-end Hamming(8,4) experiment.
pub fn run_hamming84_experiment(
    experiment: Hamming84Experiment,
) -> Result<Hamming84ExperimentReport, ChannelError> {
    let mut rng = DeterministicRng::new(experiment.seed);
    let mut report = Hamming84ExperimentReport {
        frames: experiment.frames,
        seed: experiment.seed,
        source_bits: experiment.frames.saturating_mul(4),
        channel_bit_errors: 0,
        clean_frames: 0,
        corrected_frames: 0,
        detected_uncorrectable_frames: 0,
        wrong_payload_frames: 0,
        wrong_payload_bits: 0,
    };

    for _ in 0..experiment.frames {
        let source = nibble_to_bits(rng.next_u8() & 0x0F);
        let codeword = hamming84_encode_checked(source)
            .expect("nibble_to_bits always returns binary symbols");
        let transmission = experiment.channel.transmit(&codeword, &mut rng)?;
        report.channel_bit_errors += transmission.corrupted_positions.len();

        let decoded = hamming84_decode(array8(&transmission.received))
            .expect("binary symmetric channels preserve the binary alphabet");
        match decoded.status {
            Hamming84Status::Clean => report.clean_frames += 1,
            Hamming84Status::Corrected { .. } | Hamming84Status::CorrectedOverallParity => {
                report.corrected_frames += 1;
            }
            Hamming84Status::DetectedDoubleError => {
                report.detected_uncorrectable_frames += 1;
            }
        }

        if let Some(payload) = decoded.data {
            let wrong_bits = payload
                .iter()
                .zip(source)
                .filter(|(decoded_bit, source_bit)| **decoded_bit != *source_bit)
                .count();
            if wrong_bits != 0 {
                report.wrong_payload_frames += 1;
                report.wrong_payload_bits += wrong_bits;
            }
        }
    }

    Ok(report)
}

fn nibble_to_bits(nibble: u8) -> [u8; 4] {
    [
        nibble & 1,
        (nibble >> 1) & 1,
        (nibble >> 2) & 1,
        (nibble >> 3) & 1,
    ]
}

fn array8(bits: &[u8]) -> [u8; 8] {
    bits.try_into()
        .expect("Hamming(8,4) channels preserve codeword length")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::Probability;

    #[test]
    fn reed_solomon_recovers_exactly_through_erasure_capacity() {
        let parity_symbols = 12;
        let report = run_reed_solomon_erasure_experiment(
            ReedSolomonErasureExperiment {
                frames: 500,
                message_symbols: 64,
                seed: 0xE2A5_5EED,
                config: ReedSolomonConfig::aes(parity_symbols),
                channel: SymbolErasureModel::FixedCount(
                    FixedCountErasureChannel::new(parity_symbols, 0),
                ),
            },
        )
        .unwrap();

        assert_eq!(report.clean_frames, 0);
        assert_eq!(report.recovered_frames, report.frames);
        assert_eq!(report.over_capacity_frames, 0);
        assert_eq!(report.verification_failure_frames, 0);
        assert_eq!(report.other_decoder_failure_frames, 0);
        assert_eq!(report.wrong_message_frames, 0);
        assert_eq!(report.accounted_frames(), report.frames);
        assert_eq!(report.channel_erasures, report.frames * parity_symbols);
    }

    #[test]
    fn reed_solomon_reports_every_over_capacity_frame() {
        let parity_symbols = 8;
        let report = run_reed_solomon_erasure_experiment(
            ReedSolomonErasureExperiment {
                frames: 200,
                message_symbols: 32,
                seed: 99,
                config: ReedSolomonConfig::aes(parity_symbols),
                channel: SymbolErasureModel::FixedCount(
                    FixedCountErasureChannel::new(parity_symbols + 1, 0xFF),
                ),
            },
        )
        .unwrap();

        assert_eq!(report.over_capacity_frames, report.frames);
        assert_eq!(report.recovered_frames, 0);
        assert_eq!(report.wrong_message_frames, 0);
        assert_eq!(report.accounted_frames(), report.frames);
    }

    #[test]
    fn reed_solomon_erasure_evidence_is_seed_reproducible() {
        let experiment = ReedSolomonErasureExperiment {
            frames: 2_000,
            message_symbols: 48,
            seed: 0xC0DE_E2A5,
            config: ReedSolomonConfig::aes(10),
            channel: SymbolErasureModel::Independent(
                SymbolErasureChannel::new(Probability::new(1, 100).unwrap(), 0),
            ),
        };
        let first = run_reed_solomon_erasure_experiment(experiment).unwrap();
        let second = run_reed_solomon_erasure_experiment(experiment).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.accounted_frames(), first.frames);
        assert_eq!(first.wrong_message_frames, 0);
        assert!(first.channel_erasures > 0);
        assert!(first.recovered_frames > 0);
    }

    #[test]
    fn reed_solomon_experiment_rejects_impossible_fixed_count() {
        let result = run_reed_solomon_erasure_experiment(
            ReedSolomonErasureExperiment {
                frames: 1,
                message_symbols: 2,
                seed: 1,
                config: ReedSolomonConfig::aes(2),
                channel: SymbolErasureModel::FixedCount(
                    FixedCountErasureChannel::new(5, 0),
                ),
            },
        );
        assert_eq!(
            result,
            Err(ExperimentError::Channel(
                ChannelError::TooManyRequestedErasures {
                    requested: 5,
                    symbols: 4,
                }
            ))
        );
    }

    #[test]
    fn mixed_errata_campaigns_recover_every_guaranteed_partition() {
        let parity_symbols = 10;
        for erasures in 0..=parity_symbols {
            for errors in 0..=(parity_symbols - erasures) / 2 {
                let report = run_reed_solomon_errata_experiment(
                    ReedSolomonErrataExperiment {
                        frames: 40,
                        message_symbols: 48,
                        seed: 0xE22A_0000 ^ ((errors as u64) << 8) ^ erasures as u64,
                        config: ReedSolomonConfig::aes(parity_symbols),
                        channel: FixedCountErrataChannel::new(errors, erasures, 0),
                    },
                )
                .unwrap();

                assert!(report.within_guaranteed_capacity);
                assert_eq!(report.exact_recovery_frames, report.frames);
                assert_eq!(report.wrong_message_frames, 0);
                assert_eq!(report.accounted_frames(), report.frames);
                assert_eq!(report.channel_errors, report.frames * errors);
                assert_eq!(report.channel_erasures, report.frames * erasures);
            }
        }
    }

    #[test]
    fn mixed_errata_evidence_and_manifest_are_seed_stable() {
        let experiment = ReedSolomonErrataExperiment {
            frames: 250,
            message_symbols: 32,
            seed: 0x51A7_E22A_5EED,
            config: ReedSolomonConfig {
                parity_symbols: 8,
                primitive_element: crate::reed_solomon::AES_PRIMITIVE_ELEMENT,
                first_root: 17,
            },
            channel: FixedCountErrataChannel::new(2, 4, 0xEE),
        };
        let first = run_reed_solomon_errata_experiment(experiment).unwrap();
        let second = run_reed_solomon_errata_experiment(experiment).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.exact_recovery_frames, first.frames);
        assert_eq!(
            experiment.manifest().unwrap(),
            "symthaea-coding-evidence-v1;kind=rs-fixed-errata;profile=rs-gf256-p11b-g03-fcr11-msb-systematic-nsym8;k=32;n=40;frames=250;seed=000051a7e22a5eed;errors=2;erasures=4;placeholder=ee"
        );
    }

    #[test]
    fn mixed_errata_campaign_accounts_for_unguaranteed_outcomes() {
        let report = run_reed_solomon_errata_experiment(
            ReedSolomonErrataExperiment {
                frames: 400,
                message_symbols: 40,
                seed: 0xBAD0_E22A,
                config: ReedSolomonConfig::aes(6),
                channel: FixedCountErrataChannel::new(2, 3, 0),
            },
        )
        .unwrap();

        assert!(!report.within_guaranteed_capacity);
        assert_eq!(report.accounted_frames(), report.frames);
        assert_eq!(report.channel_errors, report.frames * 2);
        assert_eq!(report.channel_erasures, report.frames * 3);
    }

    #[test]
    fn mixed_errata_experiment_rejects_impossible_distinct_locations() {
        let result = run_reed_solomon_errata_experiment(
            ReedSolomonErrataExperiment {
                frames: 0,
                message_symbols: 2,
                seed: 1,
                config: ReedSolomonConfig::aes(2),
                channel: FixedCountErrataChannel::new(3, 2, 0),
            },
        );
        assert_eq!(
            result,
            Err(ExperimentError::Channel(ChannelError::TooManyRequestedErrata {
                errors: 3,
                erasures: 2,
                symbols: 4,
            }))
        );
    }

    #[test]
    fn no_noise_recovers_every_payload_cleanly() {
        let report = run_hamming84_experiment(Hamming84Experiment {
            frames: 10_000,
            seed: 0xC0DE,
            channel: BinarySymmetricChannel::new(Probability::new(0, 1).unwrap()),
        })
        .unwrap();

        assert_eq!(report.channel_bit_errors, 0);
        assert_eq!(report.clean_frames, report.frames);
        assert_eq!(report.corrected_frames, 0);
        assert_eq!(report.detected_uncorrectable_frames, 0);
        assert_eq!(report.wrong_payload_frames, 0);
        assert_eq!(report.wrong_payload_bits, 0);
    }

    #[test]
    fn evidence_is_reproducible_from_seed_and_exact_probability() {
        let experiment = Hamming84Experiment {
            frames: 25_000,
            seed: 0x51A7_1C5E_5EED,
            channel: BinarySymmetricChannel::new(Probability::new(1, 100).unwrap()),
        };
        let first = run_hamming84_experiment(experiment).unwrap();
        let second = run_hamming84_experiment(experiment).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.clean_frames + first.corrected_frames + first.detected_uncorrectable_frames,
            first.frames
        );
        assert!(first.channel_bit_errors > 0);
        assert!(first.corrected_frames > 0);
    }

    #[test]
    fn different_seeds_produce_different_evidence() {
        let channel = BinarySymmetricChannel::new(Probability::new(1, 20).unwrap());
        let first = run_hamming84_experiment(Hamming84Experiment {
            frames: 2_000,
            seed: 1,
            channel,
        })
        .unwrap();
        let second = run_hamming84_experiment(Hamming84Experiment {
            frames: 2_000,
            seed: 2,
            channel,
        })
        .unwrap();
        assert_ne!(first, second);
    }
}
