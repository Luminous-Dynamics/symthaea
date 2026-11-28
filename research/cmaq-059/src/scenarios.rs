#![forbid(unsafe_code)]

//! Reproducible paired conditional-stream generators for CMAQ benchmarks.
//!
//! Each replicate is seeded by (experiment, scenario, replicate), never by
//! algorithm arm. All arms therefore consume identical source paths when run
//! with the same manifest. The seed digest is fed to a pinned ChaCha20 RNG;
//! the generated stream receives its own canonical SHA-256 digest.

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use sha2::{Digest, Sha256};

use crate::{paired_stream_seed, seed_hex, CalibrationError};

const STREAM_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-059/SYNTHETIC-PAIRED-STREAM/V1\0";
const MAX_COVARIATE_DIMENSION: usize = 256;
const MAX_HORIZON: u64 = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenarioKind {
    EqualConditionalIid,
    EqualConditionalHeteroskedastic,
    EqualConditionalSerial,
    EqualConditionalCovariateDrift,
    EqualMarginalConditionalPairingShift,
    SparseConditionalShift,
    HighFrequencyConditionalShift,
    LateAbruptShift,
    TemporaryShift,
    PeriodicConditionalShift,
    ConditionalScaleShift,
}

impl ScenarioKind {
    pub const ALL: [Self; 11] = [
        Self::EqualConditionalIid,
        Self::EqualConditionalHeteroskedastic,
        Self::EqualConditionalSerial,
        Self::EqualConditionalCovariateDrift,
        Self::EqualMarginalConditionalPairingShift,
        Self::SparseConditionalShift,
        Self::HighFrequencyConditionalShift,
        Self::LateAbruptShift,
        Self::TemporaryShift,
        Self::PeriodicConditionalShift,
        Self::ConditionalScaleShift,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::EqualConditionalIid => "equal-conditional-iid",
            Self::EqualConditionalHeteroskedastic => "equal-conditional-heteroskedastic",
            Self::EqualConditionalSerial => "equal-conditional-serial",
            Self::EqualConditionalCovariateDrift => "equal-conditional-covariate-drift",
            Self::EqualMarginalConditionalPairingShift => "equal-marginal-conditional-pairing-shift",
            Self::SparseConditionalShift => "sparse-conditional-shift",
            Self::HighFrequencyConditionalShift => "high-frequency-conditional-shift",
            Self::LateAbruptShift => "late-abrupt-shift",
            Self::TemporaryShift => "temporary-shift",
            Self::PeriodicConditionalShift => "periodic-conditional-shift",
            Self::ConditionalScaleShift => "conditional-scale-shift",
        }
    }

    pub fn is_null(self) -> bool {
        matches!(
            self,
            Self::EqualConditionalIid
                | Self::EqualConditionalHeteroskedastic
                | Self::EqualConditionalSerial
                | Self::EqualConditionalCovariateDrift
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioConfig {
    pub kind: ScenarioKind,
    pub horizon: u64,
    pub covariate_dimension: usize,
    pub amplitude: f64,
    /// Number of pre-change observations for late and temporary alternatives.
    pub change_point: u64,
    /// Final sequence with a nonzero effect for the temporary-shift scenario.
    pub change_end: u64,
    /// Period, in observations, for the periodic-shift alternative.
    pub period: u64,
}

impl ScenarioConfig {
    pub fn standard(kind: ScenarioKind, horizon: u64) -> Self {
        Self {
            kind,
            horizon,
            covariate_dimension: 10,
            amplitude: 0.35,
            change_point: horizon / 2,
            change_end: horizon.saturating_mul(3) / 4,
            period: 64,
        }
    }

    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.horizon == 0
            || self.horizon > MAX_HORIZON
            || self.covariate_dimension == 0
            || self.covariate_dimension > MAX_COVARIATE_DIMENSION
            || !self.amplitude.is_finite()
            || self.amplitude < 0.0
            || self.period == 0
        {
            return Err(CalibrationError::InvalidScenarioConfig);
        }
        if matches!(
            self.kind,
            ScenarioKind::LateAbruptShift
                | ScenarioKind::TemporaryShift
                | ScenarioKind::PeriodicConditionalShift
                | ScenarioKind::ConditionalScaleShift
        ) && self.change_point >= self.horizon
        {
            return Err(CalibrationError::InvalidScenarioConfig);
        }
        if self.kind == ScenarioKind::TemporaryShift
            && (self.change_end <= self.change_point || self.change_end > self.horizon)
        {
            return Err(CalibrationError::InvalidScenarioConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedObservation {
    pub sequence: u64,
    pub covariate: Vec<f64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedStream {
    pub experiment_id: String,
    pub scenario_id: String,
    pub replicate_id: u64,
    pub horizon: u64,
    pub seed_sha256: String,
    pub stream_sha256: String,
    pub observations: Vec<PairedObservation>,
}

/// Generates one complete paired source stream.
///
/// Under the four null scenarios, X_t and Y_t have exactly the same
/// conditional law given the current covariate and the full previous stream.
/// In the serial scenario, their common conditional mean depends on a state
/// updated only from previous pairs. In the heteroskedastic scenario, both
/// responses share the same predictable, covariate-dependent scale.
///
/// The equal-marginal conditional-pairing alternative uses symmetric
/// covariates and opposite conditional mean effects, so the unconditional
/// marginal distributions of X and Y remain equal while their conditional
/// laws differ.
pub fn generate_stream(
    experiment_id: &str,
    config: &ScenarioConfig,
    replicate_id: u64,
) -> Result<GeneratedStream, CalibrationError> {
    config.validate()?;
    let scenario_id = config.kind.id();
    let seed = paired_stream_seed(experiment_id, scenario_id, replicate_id)?;
    let seed_sha256 = seed_hex(&seed);
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut stream_hasher = Sha256::new();
    stream_hasher.update(STREAM_DOMAIN);
    update_length_prefixed(&mut stream_hasher, experiment_id.as_bytes());
    update_length_prefixed(&mut stream_hasher, scenario_id.as_bytes());
    stream_hasher.update(replicate_id.to_be_bytes());
    stream_hasher.update(config.horizon.to_be_bytes());
    stream_hasher.update((config.covariate_dimension as u64).to_be_bytes());
    stream_hasher.update(config.amplitude.to_bits().to_be_bytes());
    stream_hasher.update(config.change_point.to_be_bytes());
    stream_hasher.update(config.change_end.to_be_bytes());
    stream_hasher.update(config.period.to_be_bytes());

    let mut observations = Vec::with_capacity(config.horizon as usize);
    let mut serial_state = 0.0_f64;

    for sequence in 1..=config.horizon {
        let mut covariate = Vec::with_capacity(config.covariate_dimension);
        for dimension in 0..config.covariate_dimension {
            let uniform: f64 = rng.random();
            let base = 2.0 * uniform - 1.0;
            let drift = if config.kind == ScenarioKind::EqualConditionalCovariateDrift {
                0.55 * ((sequence as f64 + dimension as f64 * 13.0) / 73.0).sin()
            } else {
                0.0
            };
            covariate.push(base + drift);
        }

        let context = covariate.iter().sum::<f64>()
            / (config.covariate_dimension as f64).sqrt();

        let serial_mean = if config.kind == ScenarioKind::EqualConditionalSerial {
            serial_state
        } else {
            0.0
        };
        let common_context_mean = if config.kind == ScenarioKind::EqualMarginalConditionalPairingShift {
            0.0
        } else {
            0.25 * context
        };
        let common_mean = common_context_mean + serial_mean;

        let common_scale = if config.kind == ScenarioKind::EqualConditionalHeteroskedastic {
            0.5 + 0.3 * context.abs().min(2.0)
        } else {
            1.0
        };

        let epsilon_x: f64 = StandardNormal.sample(&mut rng);
        let epsilon_y: f64 = StandardNormal.sample(&mut rng);
        let mut x_mean = common_mean;
        let mut y_mean = common_mean;
        let mut x_scale = common_scale;
        let mut y_scale = common_scale;

        match config.kind {
            ScenarioKind::EqualMarginalConditionalPairingShift => {
                x_mean += config.amplitude * context;
                y_mean -= config.amplitude * context;
            }
            ScenarioKind::SparseConditionalShift => {
                if covariate.first().copied().unwrap_or(0.0) > 0.75 {
                    y_mean += config.amplitude;
                }
            }
            ScenarioKind::HighFrequencyConditionalShift => {
                y_mean += config.amplitude * (4.0 * context).sin();
            }
            ScenarioKind::LateAbruptShift => {
                if sequence > config.change_point {
                    y_mean += config.amplitude;
                }
            }
            ScenarioKind::TemporaryShift => {
                if sequence > config.change_point && sequence <= config.change_end {
                    y_mean += config.amplitude;
                }
            }
            ScenarioKind::PeriodicConditionalShift => {
                if sequence > config.change_point {
                    y_mean += config.amplitude
                        * (std::f64::consts::TAU * sequence as f64 / config.period as f64).sin();
                }
            }
            ScenarioKind::ConditionalScaleShift => {
                if sequence > config.change_point {
                    y_scale *= config.amplitude.exp();
                }
            }
            ScenarioKind::EqualConditionalIid
            | ScenarioKind::EqualConditionalHeteroskedastic
            | ScenarioKind::EqualConditionalSerial
            | ScenarioKind::EqualConditionalCovariateDrift => {}
        }

        let x_value = x_mean + x_scale * epsilon_x;
        let y_value = y_mean + y_scale * epsilon_y;
        if !x_value.is_finite() || !y_value.is_finite() {
            return Err(CalibrationError::InvalidScenarioConfig);
        }

        // State is updated only after both current responses are generated.
        // At t+1 it is therefore part of the observable past.
        if config.kind == ScenarioKind::EqualConditionalSerial {
            serial_state = 0.72 * serial_state + 0.14 * (x_value + y_value);
        }

        let observation = PairedObservation {
            sequence,
            covariate,
            x: vec![x_value],
            y: vec![y_value],
        };
        stream_hasher.update(sequence.to_be_bytes());
        for vector in [
            observation.covariate.as_slice(),
            observation.x.as_slice(),
            observation.y.as_slice(),
        ] {
            stream_hasher.update((vector.len() as u64).to_be_bytes());
            for value in vector {
                stream_hasher.update(value.to_bits().to_be_bytes());
            }
        }
        observations.push(observation);
    }

    let stream_digest = stream_hasher.finalize();
    let stream_sha256 = hex_digest(&stream_digest);
    Ok(GeneratedStream {
        experiment_id: experiment_id.to_string(),
        scenario_id: scenario_id.to_string(),
        replicate_id,
        horizon: config.horizon,
        seed_sha256,
        stream_sha256,
        observations,
    })
}

fn update_length_prefixed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_generation_is_reproducible_and_arm_independent_by_seed_contract() {
        let config = ScenarioConfig::standard(ScenarioKind::EqualConditionalIid, 128);
        let first = generate_stream("cmaq-059-stream-v1", &config, 11).unwrap();
        let repeated = generate_stream("cmaq-059-stream-v1", &config, 11).unwrap();
        let next = generate_stream("cmaq-059-stream-v1", &config, 12).unwrap();

        assert_eq!(first.seed_sha256, repeated.seed_sha256);
        assert_eq!(first.stream_sha256, repeated.stream_sha256);
        assert_ne!(first.stream_sha256, next.stream_sha256);
        assert_eq!(first.observations.len(), 128);
    }

    #[test]
    fn generated_null_scenarios_are_finite_and_identified_as_null() {
        for kind in [
            ScenarioKind::EqualConditionalIid,
            ScenarioKind::EqualConditionalHeteroskedastic,
            ScenarioKind::EqualConditionalSerial,
            ScenarioKind::EqualConditionalCovariateDrift,
        ] {
            let config = ScenarioConfig::standard(kind, 256);
            let stream = generate_stream("cmaq-059-null-v1", &config, 3).unwrap();
            assert!(kind.is_null());
            assert!(stream.observations.iter().all(|observation| {
                observation.covariate.iter().all(|value| value.is_finite())
                    && observation.x.iter().all(|value| value.is_finite())
                    && observation.y.iter().all(|value| value.is_finite())
            }));
        }
    }

    #[test]
    fn equal_marginal_conditional_alternative_is_symmetric_by_construction() {
        let config = ScenarioConfig::standard(
            ScenarioKind::EqualMarginalConditionalPairingShift,
            128,
        );
        assert!(!config.kind.is_null());
        let stream = generate_stream("cmaq-059-alt-v1", &config, 5).unwrap();
        assert_eq!(stream.observations.len(), 128);
        assert!(stream.observations.iter().all(|observation| {
            observation.covariate.iter().all(|value| value.is_finite())
                && observation.x[0].is_finite()
                && observation.y[0].is_finite()
        }));
    }

    #[test]
    fn invalid_scenario_horizons_and_dimensions_fail_closed() {
        let mut config = ScenarioConfig::standard(ScenarioKind::EqualConditionalIid, 16);
        config.horizon = 0;
        assert_eq!(
            generate_stream("cmaq-059-invalid", &config, 0),
            Err(CalibrationError::InvalidScenarioConfig)
        );
        config = ScenarioConfig::standard(ScenarioKind::TemporaryShift, 16);
        config.change_end = config.change_point;
        assert_eq!(
            generate_stream("cmaq-059-invalid", &config, 0),
            Err(CalibrationError::InvalidScenarioConfig)
        );
    }
}
