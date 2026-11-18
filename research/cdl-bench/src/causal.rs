//! Deterministic finite-state and structural-causal generators for CDL-Bench.
//!
//! Ground-truth latent states and SCM records are returned separately from the
//! observation byte stream. Evaluators should provide only the observation
//! stream and declared intervention manifest to a learner; retain latent state
//! and generator parameters in the sealed scoring manifest.

use std::fmt;

const QMAX: i32 = i16::MAX as i32;
const SCM_MAGIC: &[u8; 4] = b"SCM1";
const SCM_VERSION: u16 = 1;
const SCM_HEADER_LEN: usize = 33;

/// Invalid configuration for a deterministic causal benchmark generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratorError {
    StateCountOutOfRange,
    AlphabetSizeOutOfRange,
    InterventionVariableOutOfRange,
    InterventionWindowInvalid,
    LengthOverflow,
    InvalidMechanismConfiguration,
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StateCountOutOfRange => write!(f, "FSM state count must be in 2..=16"),
            Self::AlphabetSizeOutOfRange => write!(f, "FSM alphabet size must be in 2..=16"),
            Self::InterventionVariableOutOfRange => {
                write!(f, "SCM intervention variable must be in 0..=4")
            }
            Self::InterventionWindowInvalid => {
                write!(f, "SCM intervention window must satisfy 0 <= start < end <= steps")
            }
            Self::LengthOverflow => write!(f, "SCM stream length overflows this platform"),
            Self::InvalidMechanismConfiguration => write!(f, "SCM mechanism coefficients are outside their valid ranges"),
        }
    }
}

impl std::error::Error for GeneratorError {}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Observations and withheld ground truth for a finite-state source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsmTrace {
    /// Byte observations presented to a compression/prediction system.
    pub observations: Vec<u8>,
    /// Hidden state at each timestep; evaluator-only ground truth.
    pub latent_states: Vec<u8>,
    pub state_count: u8,
    pub alphabet_size: u8,
}

/// Generate a partially observed finite-state process.
///
/// The transition law is sticky (70% self-transition), with adjacent and
/// random jumps. Emissions alias states modulo alphabet_size and include a
/// 4% observation-noise channel. State and emission randomness are separate
/// draws from one deterministic, seed-pinned generator.
pub fn generate_fsm(
    len: usize,
    state_count: u8,
    alphabet_size: u8,
    seed: u64,
) -> Result<FsmTrace, GeneratorError> {
    if !(2..=16).contains(&state_count) {
        return Err(GeneratorError::StateCountOutOfRange);
    }
    if !(2..=16).contains(&alphabet_size) {
        return Err(GeneratorError::AlphabetSizeOutOfRange);
    }

    let mut rng = SplitMix64::new(seed);
    let mut state = (rng.next() % state_count as u64) as u8;
    let mut observations = Vec::with_capacity(len);
    let mut latent_states = Vec::with_capacity(len);

    for _ in 0..len {
        latent_states.push(state);
        let noisy = rng.next() % 100 < 4;
        let observed = if noisy {
            (rng.next() % alphabet_size as u64) as u8
        } else {
            state % alphabet_size
        };
        observations.push(observed);

        let transition = rng.next() % 100;
        state = match transition {
            0..=69 => state,
            70..=84 => (state + 1) % state_count,
            85..=94 => (state + state_count - 1) % state_count,
            _ => (rng.next() % state_count as u64) as u8,
        };
    }

    Ok(FsmTrace {
        observations,
        latent_states,
        state_count,
        alphabet_size,
    })
}

/// A hard intervention, active on the half-open interval [start_step, end_step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScmIntervention {
    /// Variable index: 0=x0, 1=x1, 2=x2, 3=x3, 4=x4.
    pub variable: u8,
    pub start_step: usize,
    pub end_step: usize,
    /// Fixed-point Q15 value, clamped to the signed i16 range.
    pub value: i16,
}

/// Nonlinear response family for the x3 structural mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScmResponse {
    Linear = 0,
    Saturating = 1,
}

/// Factorized mechanism parameters for E3 combination/transfer experiments.
///
/// Coefficients are integer percentages so the generator remains deterministic
/// and cross-platform. These values are evaluator-side ground truth and are
/// deliberately not serialized into the learner-visible SCM stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScmMechanismConfig {
    /// Percent of x1[t] driven by x0[t]; remainder is x1 autoregression.
    pub x1_parent_gain: u8,
    /// Percent of x2[t] driven by x1[t].
    pub x2_parent_gain: u8,
    /// Percent of x2[t] counter-coupled to x0[t].
    pub x2_x0_counter_gain: u8,
    /// Response family mapping x2 and prior x3 into x3.
    pub x3_response: ScmResponse,
    /// Percent of x4[t] driven by x1[t].
    pub x4_x1_gain: u8,
    /// Percent of x4[t] driven by x3[t].
    pub x4_x3_gain: u8,
}

impl Default for ScmMechanismConfig {
    fn default() -> Self {
        Self {
            x1_parent_gain: 70,
            x2_parent_gain: 80,
            x2_x0_counter_gain: 10,
            x3_response: ScmResponse::Saturating,
            x4_x1_gain: 40,
            x4_x3_gain: 40,
        }
    }
}

impl ScmMechanismConfig {
    fn validate(self) -> Result<Self, GeneratorError> {
        if self.x1_parent_gain > 100
            || self.x2_parent_gain > 100
            || self.x2_x0_counter_gain > 100
            || self.x4_x1_gain > 80
            || self.x4_x3_gain > 80
            || self.x4_x1_gain as u16 + self.x4_x3_gain as u16 > 80
        {
            return Err(GeneratorError::InvalidMechanismConfiguration);
        }
        Ok(self)
    }
}

/// Fixed training and held-out mechanism combinations for E3.
///
/// Each component value appears in training, but the held-out tuples do not.
/// This enables tests of compositional mechanism reuse rather than seed-only
/// generalization. The tuples are protocol fixtures, not an assertion that
/// any learner has already recovered the mechanisms.
pub fn e3_mechanism_splits() -> (Vec<ScmMechanismConfig>, Vec<ScmMechanismConfig>) {
    let training = vec![
        ScmMechanismConfig {
            x1_parent_gain: 30,
            x2_parent_gain: 70,
            x2_x0_counter_gain: 10,
            x3_response: ScmResponse::Linear,
            x4_x1_gain: 20,
            x4_x3_gain: 60,
        },
        ScmMechanismConfig {
            x1_parent_gain: 70,
            x2_parent_gain: 30,
            x2_x0_counter_gain: 30,
            x3_response: ScmResponse::Saturating,
            x4_x1_gain: 60,
            x4_x3_gain: 20,
        },
        ScmMechanismConfig {
            x1_parent_gain: 30,
            x2_parent_gain: 30,
            x2_x0_counter_gain: 10,
            x3_response: ScmResponse::Saturating,
            x4_x1_gain: 20,
            x4_x3_gain: 20,
        },
    ];
    let held_out = vec![
        ScmMechanismConfig {
            x1_parent_gain: 70,
            x2_parent_gain: 70,
            x2_x0_counter_gain: 30,
            x3_response: ScmResponse::Linear,
            x4_x1_gain: 60,
            x4_x3_gain: 20,
        },
        ScmMechanismConfig {
            x1_parent_gain: 30,
            x2_parent_gain: 70,
            x2_x0_counter_gain: 30,
            x3_response: ScmResponse::Saturating,
            x4_x1_gain: 60,
            x4_x3_gain: 20,
        },
    ];
    (training, held_out)
}

/// Five observed variables from one generated structural-causal time series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScmTrace {
    /// Observed values at each timestep, in variable order x0..x4.
    pub observations: Vec<[i16; 5]>,
    /// Binary stream: SCM1 header, intervention manifest, then little-endian i16 records.
    pub stream: Vec<u8>,
    /// Intervention supplied to the generator (also represented in stream header).
    pub intervention: Option<ScmIntervention>,
    /// Hidden generator parameters, reserved for evaluator-side scoring.
    pub mechanisms: ScmMechanismConfig,
}

fn clip_q15(value: i64) -> i16 {
    value.clamp(-(QMAX as i64), QMAX as i64) as i16
}

fn noise(rng: &mut SplitMix64) -> i64 {
    (rng.next() % 8193) as i64 - 4096
}

fn apply_do(
    natural_value: i16,
    variable: usize,
    step: usize,
    intervention: Option<ScmIntervention>,
) -> i16 {
    match intervention {
        Some(do_op)
            if do_op.variable as usize == variable
                && step >= do_op.start_step
                && step < do_op.end_step =>
        {
            do_op.value
        }
        _ => natural_value,
    }
}

/// Generate a fixed-point five-variable structural causal time series.
///
/// Structural dependencies are:
/// - x0[t] <- x0[t-1] + exogenous noise
/// - x1[t] <- x0[t], x1[t-1], noise
/// - x2[t] <- x0[t], x1[t], x2[t-1], noise
/// - x3[t] <- nonlinear(x2[t], x3[t-1]) + noise
/// - x4[t] <- x1[t], x3[t], x4[t-1], noise
///
/// Values use signed Q15-style integer scaling, avoiding non-deterministic
/// floating-point transcendental functions in the source generator. An active
/// do(variable=value) is applied after that variable's natural equation and
/// before downstream variables are evaluated.
pub fn generate_scm(
    steps: usize,
    seed: u64,
    intervention: Option<ScmIntervention>,
) -> Result<ScmTrace, GeneratorError> {
    generate_scm_with_config(steps, seed, ScmMechanismConfig::default(), intervention)
}

/// Generate an SCM trace using an explicit factorized mechanism configuration.
pub fn generate_scm_with_config(
    steps: usize,
    seed: u64,
    mechanisms: ScmMechanismConfig,
    intervention: Option<ScmIntervention>,
) -> Result<ScmTrace, GeneratorError> {
    let mechanisms = mechanisms.validate()?;
    if let Some(do_op) = intervention {
        if do_op.variable > 4 {
            return Err(GeneratorError::InterventionVariableOutOfRange);
        }
        if do_op.start_step >= do_op.end_step || do_op.end_step > steps {
            return Err(GeneratorError::InterventionWindowInvalid);
        }
    }

    let steps_u64 = u64::try_from(steps).map_err(|_| GeneratorError::LengthOverflow)?;
    let stream_capacity = steps
        .checked_mul(5 * std::mem::size_of::<i16>())
        .and_then(|payload| SCM_HEADER_LEN.checked_add(payload))
        .ok_or(GeneratorError::LengthOverflow)?;

    let mut rng = SplitMix64::new(seed);
    let mut previous = [0i16; 5];
    let mut observations = Vec::with_capacity(steps);

    for step in 0..steps {
        let n0 = noise(&mut rng);
        let n1 = noise(&mut rng);
        let n2 = noise(&mut rng);
        let n3 = noise(&mut rng);
        let n4 = noise(&mut rng);

        // Topological evaluation order is explicit: interventions on upstream
        // variables affect descendants in the same timestep.
        let x0_natural = clip_q15((previous[0] as i64 * 3) / 4 + n0);
        let x0 = apply_do(x0_natural, 0, step, intervention);

        let x1_parent = mechanisms.x1_parent_gain as i64;
        let x1_natural = clip_q15(
            (previous[1] as i64 * (100 - x1_parent)) / 100
                + (x0 as i64 * x1_parent) / 100
                + n1,
        );
        let x1 = apply_do(x1_natural, 1, step, intervention);

        let x2_parent = mechanisms.x2_parent_gain as i64;
        let x2_counter = mechanisms.x2_x0_counter_gain as i64;
        let x2_natural = clip_q15(
            (previous[2] as i64 * 2) / 10
                + (x1 as i64 * x2_parent) / 100
                - (x0 as i64 * x2_counter) / 100
                + n2,
        );
        let x2 = apply_do(x2_natural, 2, step, intervention);

        let nonlinear_input = x2 as i64 + (previous[3] as i64 * 2) / 10;
        let response = match mechanisms.x3_response {
            ScmResponse::Linear => nonlinear_input,
            ScmResponse::Saturating => {
                (nonlinear_input * QMAX as i64)
                    / (QMAX as i64 + nonlinear_input.abs())
            }
        };
        let x3_natural = clip_q15(response + n3);
        let x3 = apply_do(x3_natural, 3, step, intervention);

        let x4_natural = clip_q15(
            (previous[4] as i64 * 2) / 10
                + (x1 as i64 * mechanisms.x4_x1_gain as i64) / 100
                + (x3 as i64 * mechanisms.x4_x3_gain as i64) / 100
                + n4,
        );
        let x4 = apply_do(x4_natural, 4, step, intervention);

        let current = [x0, x1, x2, x3, x4];
        observations.push(current);
        previous = current;
    }

    let mut stream = Vec::with_capacity(stream_capacity);
    stream.extend_from_slice(SCM_MAGIC);
    stream.extend_from_slice(&SCM_VERSION.to_le_bytes());
    match intervention {
        Some(do_op) => {
            stream.push(do_op.variable);
            stream.extend_from_slice(
                &u64::try_from(do_op.start_step)
                    .map_err(|_| GeneratorError::LengthOverflow)?
                    .to_le_bytes(),
            );
            stream.extend_from_slice(
                &u64::try_from(do_op.end_step)
                    .map_err(|_| GeneratorError::LengthOverflow)?
                    .to_le_bytes(),
            );
            stream.extend_from_slice(&do_op.value.to_le_bytes());
        }
        None => {
            stream.push(u8::MAX);
            stream.extend_from_slice(&0u64.to_le_bytes());
            stream.extend_from_slice(&0u64.to_le_bytes());
            stream.extend_from_slice(&0i16.to_le_bytes());
        }
    }
    stream.extend_from_slice(&steps_u64.to_le_bytes());

    for record in &observations {
        for value in record {
            stream.extend_from_slice(&value.to_le_bytes());
        }
    }

    debug_assert_eq!(stream.len(), SCM_HEADER_LEN + steps * 10);
    Ok(ScmTrace {
        observations,
        stream,
        intervention,
        mechanisms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsm_generation_is_deterministic_and_keeps_truth_separate() {
        let a = generate_fsm(10_000, 8, 4, 123).unwrap();
        let b = generate_fsm(10_000, 8, 4, 123).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.observations.len(), 10_000);
        assert_eq!(a.latent_states.len(), 10_000);
        assert!(a.observations.iter().all(|&v| v < 4));
        assert!(a.latent_states.iter().all(|&v| v < 8));
    }

    #[test]
    fn fsm_rejects_invalid_state_or_alphabet_sizes() {
        assert_eq!(
            generate_fsm(10, 1, 4, 1).unwrap_err(),
            GeneratorError::StateCountOutOfRange
        );
        assert_eq!(
            generate_fsm(10, 4, 1, 1).unwrap_err(),
            GeneratorError::AlphabetSizeOutOfRange
        );
    }

    #[test]
    fn scm_stream_is_deterministic_and_length_delimited() {
        let a = generate_scm(1000, 55, None).unwrap();
        let b = generate_scm(1000, 55, None).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.stream.len(), SCM_HEADER_LEN + 1000 * 10);
        assert_eq!(&a.stream[..4], SCM_MAGIC);
    }

    #[test]
    fn intervention_changes_descendants_without_changing_upstream_cause() {
        let baseline = generate_scm(80, 99, None).unwrap();
        let intervention = ScmIntervention {
            variable: 1,
            start_step: 10,
            end_step: 30,
            value: 20_000,
        };
        let treated = generate_scm(80, 99, Some(intervention)).unwrap();

        for step in 0..80 {
            assert_eq!(baseline.observations[step][0], treated.observations[step][0]);
        }
        for step in 10..30 {
            assert_eq!(treated.observations[step][1], 20_000);
        }
        assert_ne!(
            baseline.observations[10..30]
                .iter()
                .map(|row| row[2])
                .collect::<Vec<_>>(),
            treated.observations[10..30]
                .iter()
                .map(|row| row[2])
                .collect::<Vec<_>>()
        );
        assert_ne!(
            baseline.observations[10..30]
                .iter()
                .map(|row| row[4])
                .collect::<Vec<_>>(),
            treated.observations[10..30]
                .iter()
                .map(|row| row[4])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn configured_scm_is_deterministic_and_returns_hidden_mechanism_truth() {
        let (training, held_out) = e3_mechanism_splits();
        assert!(!training.is_empty());
        assert!(!held_out.is_empty());
        for config in training.iter().chain(held_out.iter()) {
            let a = generate_scm_with_config(256, 72, *config, None).unwrap();
            let b = generate_scm_with_config(256, 72, *config, None).unwrap();
            assert_eq!(a, b);
            assert_eq!(a.mechanisms, *config);
            assert_eq!(a.stream.len(), SCM_HEADER_LEN + 256 * 10);
        }
        for test in &held_out {
            assert!(!training.contains(test), "held-out tuple duplicated a training tuple");
        }
    }

    #[test]
    fn e3_splits_reuse_component_values_but_hold_out_combinations() {
        let (training, held_out) = e3_mechanism_splits();
        let seen_x1: Vec<_> = training.iter().map(|c| c.x1_parent_gain).collect();
        let seen_x2: Vec<_> = training.iter().map(|c| c.x2_parent_gain).collect();
        let seen_response: Vec<_> = training.iter().map(|c| c.x3_response).collect();
        let seen_x4: Vec<_> = training
            .iter()
            .flat_map(|c| [c.x4_x1_gain, c.x4_x3_gain])
            .collect();
        for test in &held_out {
            assert!(seen_x1.contains(&test.x1_parent_gain));
            assert!(seen_x2.contains(&test.x2_parent_gain));
            assert!(seen_response.contains(&test.x3_response));
            assert!(seen_x4.contains(&test.x4_x1_gain));
            assert!(seen_x4.contains(&test.x4_x3_gain));
        }
    }

    #[test]
    fn e3_heldout_intervention_pairs_preserve_upstream_and_change_descendants() {
        let (_, held_out) = e3_mechanism_splits();
        for (world_index, config) in held_out.iter().enumerate() {
            let seed = 8_000 + world_index as u64;
            let observational =
                generate_scm_with_config(128, seed, *config, None).unwrap();
            let treated = generate_scm_with_config(
                128,
                seed,
                *config,
                Some(ScmIntervention {
                    variable: 1,
                    start_step: 32,
                    end_step: 96,
                    value: 20_000,
                }),
            )
            .unwrap();

            for step in 0..128 {
                assert_eq!(
                    observational.observations[step][0],
                    treated.observations[step][0],
                    "upstream x0 changed in held-out world {world_index}"
                );
            }
            for step in 32..96 {
                assert_eq!(treated.observations[step][1], 20_000);
            }

            let x2_changed = (32..96)
                .any(|step| observational.observations[step][2] != treated.observations[step][2]);
            let x4_changed = (32..96)
                .any(|step| observational.observations[step][4] != treated.observations[step][4]);
            assert!(x2_changed, "x1 intervention did not reach x2 in world {world_index}");
            assert!(x4_changed, "x1 intervention did not reach x4 in world {world_index}");
        }
    }

    #[test]
    fn invalid_mechanism_coefficients_reject() {
        let invalid = ScmMechanismConfig {
            x1_parent_gain: 101,
            ..ScmMechanismConfig::default()
        };
        assert_eq!(
            generate_scm_with_config(10, 1, invalid, None).unwrap_err(),
            GeneratorError::InvalidMechanismConfiguration
        );
    }

    #[test]
    fn scm_rejects_invalid_interventions() {
        assert_eq!(
            generate_scm(
                10,
                1,
                Some(ScmIntervention {
                    variable: 5,
                    start_step: 0,
                    end_step: 1,
                    value: 0,
                })
            )
            .unwrap_err(),
            GeneratorError::InterventionVariableOutOfRange
        );
        assert_eq!(
            generate_scm(
                10,
                1,
                Some(ScmIntervention {
                    variable: 1,
                    start_step: 7,
                    end_step: 6,
                    value: 0,
                })
            )
            .unwrap_err(),
            GeneratorError::InterventionWindowInvalid
        );
    }
}
