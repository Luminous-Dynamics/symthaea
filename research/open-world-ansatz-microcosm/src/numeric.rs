//! Coefficient-bearing numeric extension of the finite ansatz microcosm.
//!
//! Unlike the binary v1 control, this model family contains unknown numerical
//! gain/decay parameters. A0 supplies the mechanism but must recover its
//! parameters. A2 must recover both mechanism and parameters. Both arms consume
//! the exact same intervention and observation log.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const INPUT_LEVELS: [i64; 3] = [-1, 0, 1];
/// Closed candidate-grammar gains; this is deliberately narrower than the world generator.
pub const GAIN_VALUES: [i64; 3] = [1, 2, 3];
/// Maximum gain accepted by the reference world generator, including out-of-catalogue controls.
pub const MAX_WORLD_GAIN: i64 = 16;
pub const DECAY_VALUES: [i64; 3] = [0, 1, 2];
pub const INITIAL_STATES: [i64; 3] = [-1, 0, 1];
pub const MAX_ORACLE_HORIZON: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mechanism {
    /// Output is a memoryless response to the current intervention.
    DirectResponse,
    /// Output is the current hidden state; interventions change its future.
    LeakyMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorMap {
    Identity,
    Negate,
    Double,
    /// An uninformative channel that maps every latent value to zero.
    ConstantZero,
}

impl SensorMap {
    fn apply(self, latent: i64) -> i64 {
        match self {
            Self::Identity => latent,
            Self::Negate => -latent,
            Self::Double => 2 * latent,
            Self::ConstantZero => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "mechanism", rename_all = "snake_case")]
pub enum StructureAccess {
    /// A0: mechanism supplied, numerical parameters and sensor withheld.
    Supplied(Mechanism),
    /// A2: mechanism, parameters and sensor all withheld.
    Withheld,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelHypothesis {
    pub mechanism: Mechanism,
    pub gain: i64,
    /// Present only for LeakyMemory.
    pub decay: Option<i64>,
    /// Present only for LeakyMemory.
    pub initial_state: Option<i64>,
    pub sensor: SensorMap,
}

impl ModelHypothesis {
    pub fn validate(&self) -> Result<(), NumericError> {
        if !(1..=MAX_WORLD_GAIN).contains(&self.gain) {
            return Err(NumericError::InvalidGain);
        }
        match (self.mechanism, self.decay, self.initial_state) {
            (Mechanism::DirectResponse, None, None) => Ok(()),
            (Mechanism::LeakyMemory, Some(decay), Some(initial))
                if DECAY_VALUES.contains(&decay) && INITIAL_STATES.contains(&initial) =>
            {
                Ok(())
            }
            _ => Err(NumericError::InvalidParameterBinding),
        }
    }

    pub fn predict(&self, interventions: &[i64]) -> Result<Vec<i64>, NumericError> {
        self.predict_continuation(&[], interventions)
    }

    /// Predict future observations after an already-consumed input history.
    pub fn predict_continuation(
        &self,
        history: &[i64],
        future: &[i64],
    ) -> Result<Vec<i64>, NumericError> {
        self.validate()?;
        validate_inputs(history)?;
        validate_inputs(future)?;

        let mut state = self.initial_state;
        if self.mechanism == Mechanism::LeakyMemory {
            let decay = self.decay.ok_or(NumericError::InvalidParameterBinding)?;
            for input in history {
                state = Some(
                    decay
                        .checked_mul(state.ok_or(NumericError::InvalidParameterBinding)?)
                        .and_then(|value| {
                            self.gain
                                .checked_mul(*input)
                                .and_then(|drive| value.checked_add(drive))
                        })
                        .ok_or(NumericError::ArithmeticOverflow)?,
                );
            }
        }

        let mut output = Vec::with_capacity(future.len());
        for input in future {
            let latent = match self.mechanism {
                Mechanism::DirectResponse => self
                    .gain
                    .checked_mul(*input)
                    .ok_or(NumericError::ArithmeticOverflow)?,
                Mechanism::LeakyMemory => state.ok_or(NumericError::InvalidParameterBinding)?,
            };
            output.push(self.sensor.apply(latent));

            if self.mechanism == Mechanism::LeakyMemory {
                let decay = self.decay.ok_or(NumericError::InvalidParameterBinding)?;
                state = Some(
                    decay
                        .checked_mul(state.ok_or(NumericError::InvalidParameterBinding)?)
                        .and_then(|value| {
                            self.gain
                                .checked_mul(*input)
                                .and_then(|drive| value.checked_add(drive))
                        })
                        .ok_or(NumericError::ArithmeticOverflow)?,
                );
            }
        }
        Ok(output)
    }
}

fn validate_inputs(inputs: &[i64]) -> Result<(), NumericError> {
    if inputs.iter().any(|value| !INPUT_LEVELS.contains(value)) {
        return Err(NumericError::InterventionOutOfRange);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationLog {
    pub interventions: Vec<i64>,
    pub measurements: Vec<i64>,
}

impl ObservationLog {
    pub fn append(&mut self, other: &Self) {
        self.interventions.extend_from_slice(&other.interventions);
        self.measurements.extend_from_slice(&other.measurements);
    }

    fn validate(&self) -> Result<(), NumericError> {
        if self.interventions.is_empty() {
            return Err(NumericError::EmptyExperiment);
        }
        if self.interventions.len() != self.measurements.len() {
            return Err(NumericError::MismatchedObservationLengths);
        }
        validate_inputs(&self.interventions)
    }
}

/// The generating model is evaluator-side state. run() returns only the chosen
/// intervention and measured value.
pub struct NumericWorld {
    truth: ModelHypothesis,
    state: Option<i64>,
}

impl NumericWorld {
    pub fn new(truth: ModelHypothesis) -> Result<Self, NumericError> {
        truth.validate()?;
        Ok(Self {
            truth,
            state: truth.initial_state,
        })
    }

    pub fn run(&mut self, interventions: &[i64]) -> Result<ObservationLog, NumericError> {
        validate_inputs(interventions)?;
        let mut measurements = Vec::with_capacity(interventions.len());

        for input in interventions {
            let latent = match self.truth.mechanism {
                Mechanism::DirectResponse => self
                    .truth
                    .gain
                    .checked_mul(*input)
                    .ok_or(NumericError::ArithmeticOverflow)?,
                Mechanism::LeakyMemory => {
                    self.state.ok_or(NumericError::InvalidParameterBinding)?
                }
            };
            measurements.push(self.truth.sensor.apply(latent));

            if self.truth.mechanism == Mechanism::LeakyMemory {
                let decay = self
                    .truth
                    .decay
                    .ok_or(NumericError::InvalidParameterBinding)?;
                self.state = Some(
                    decay
                        .checked_mul(self.state.ok_or(NumericError::InvalidParameterBinding)?)
                        .and_then(|value| {
                            self.truth
                                .gain
                                .checked_mul(*input)
                                .and_then(|drive| value.checked_add(drive))
                        })
                        .ok_or(NumericError::ArithmeticOverflow)?,
                );
            }
        }

        Ok(ObservationLog {
            interventions: interventions.to_vec(),
            measurements,
        })
    }
}

pub fn candidate_models(access: StructureAccess) -> Vec<ModelHypothesis> {
    let mechanisms = match access {
        StructureAccess::Supplied(mechanism) => vec![mechanism],
        StructureAccess::Withheld => {
            vec![Mechanism::DirectResponse, Mechanism::LeakyMemory]
        }
    };
    let sensors = [
        SensorMap::Identity,
        SensorMap::Negate,
        SensorMap::Double,
        SensorMap::ConstantZero,
    ];
    let mut candidates = Vec::new();

    for mechanism in mechanisms {
        for gain in GAIN_VALUES {
            for sensor in sensors {
                match mechanism {
                    Mechanism::DirectResponse => candidates.push(ModelHypothesis {
                        mechanism,
                        gain,
                        decay: None,
                        initial_state: None,
                        sensor,
                    }),
                    Mechanism::LeakyMemory => {
                        for decay in DECAY_VALUES {
                            for initial_state in INITIAL_STATES {
                                candidates.push(ModelHypothesis {
                                    mechanism,
                                    gain,
                                    decay: Some(decay),
                                    initial_state: Some(initial_state),
                                    sensor,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    candidates.sort();
    candidates
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryDisposition {
    NoCompatibleModel,
    UniqueModel,
    StructureIdentifiedParametersAmbiguous,
    StructureAmbiguous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub access: StructureAccess,
    pub compatible_models: Vec<ModelHypothesis>,
    pub compatible_structures: Vec<Mechanism>,
    pub compatible_gains: Vec<i64>,
    pub compatible_decays: Vec<i64>,
    pub compatible_initial_states: Vec<i64>,
    pub compatible_sensors: Vec<SensorMap>,
    pub structure_identified: bool,
    pub coefficients_identified: bool,
    pub initial_state_identified: bool,
    pub measurement_identified: bool,
    pub model_identified: bool,
    pub disposition: DiscoveryDisposition,
}

pub fn discover(
    log: &ObservationLog,
    access: StructureAccess,
) -> Result<DiscoveryReport, NumericError> {
    log.validate()?;
    let mut compatible_models = Vec::new();

    for model in candidate_models(access) {
        if model.predict(&log.interventions)? == log.measurements {
            compatible_models.push(model);
        }
    }
    compatible_models.sort();

    let compatible_structures: Vec<Mechanism> = compatible_models
        .iter()
        .map(|model| model.mechanism)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let compatible_gains: Vec<i64> = compatible_models
        .iter()
        .map(|model| model.gain)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let compatible_decays: Vec<i64> = compatible_models
        .iter()
        .filter_map(|model| model.decay)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let compatible_initial_states: Vec<i64> = compatible_models
        .iter()
        .filter_map(|model| model.initial_state)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let compatible_sensors: Vec<SensorMap> = compatible_models
        .iter()
        .map(|model| model.sensor)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let structure_identified = !compatible_models.is_empty() && compatible_structures.len() == 1;
    let coefficient_tuples: BTreeSet<(i64, Option<i64>)> = compatible_models
        .iter()
        .map(|model| (model.gain, model.decay))
        .collect();
    let coefficients_identified =
        structure_identified && !compatible_models.is_empty() && coefficient_tuples.len() == 1;
    let initial_state_identified = structure_identified
        && (compatible_models
            .iter()
            .all(|model| model.mechanism == Mechanism::DirectResponse)
            || compatible_initial_states.len() == 1);
    let measurement_identified = !compatible_models.is_empty() && compatible_sensors.len() == 1;
    let model_identified = compatible_models.len() == 1;
    let disposition = match (model_identified, structure_identified) {
        (false, false) if compatible_models.is_empty() => DiscoveryDisposition::NoCompatibleModel,
        (true, true) => DiscoveryDisposition::UniqueModel,
        (false, true) => DiscoveryDisposition::StructureIdentifiedParametersAmbiguous,
        _ => DiscoveryDisposition::StructureAmbiguous,
    };

    Ok(DiscoveryReport {
        access,
        compatible_models,
        compatible_structures,
        compatible_gains,
        compatible_decays,
        compatible_initial_states,
        compatible_sensors,
        structure_identified,
        coefficients_identified,
        initial_state_identified,
        measurement_identified,
        model_identified,
        disposition,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleEquivalenceClass {
    pub members: Vec<ModelHypothesis>,
}

/// Exhaustive structural and parameter equivalence oracle over every input
/// sequence in {-1,0,1} through the requested finite horizon.
pub fn oracle_equivalence_classes(
    horizon: usize,
) -> Result<Vec<OracleEquivalenceClass>, NumericError> {
    validate_horizon(horizon)?;
    let sequence_count = 3usize
        .checked_pow(horizon as u32)
        .ok_or(NumericError::OracleHorizonOutOfRange)?;
    let mut classes: BTreeMap<Vec<i64>, Vec<ModelHypothesis>> = BTreeMap::new();

    for model in candidate_models(StructureAccess::Withheld) {
        let signature = closed_form_signature(&model, horizon, sequence_count)?;
        classes.entry(signature).or_default().push(model);
    }

    Ok(classes
        .into_values()
        .map(|mut members| {
            members.sort();
            OracleEquivalenceClass { members }
        })
        .collect())
}

/// Decode a sequence in lexicographic base-3 order.
fn decode_sequence(mut code: usize, horizon: usize) -> Vec<i64> {
    let mut sequence = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let digit = code % 3;
        sequence.push(INPUT_LEVELS[digit]);
        code /= 3;
    }
    sequence
}

/// Closed form for LeakyMemory:
/// h_t = decay^t * h_0 + gain * sum_{j=0}^{t-1}(decay^(t-1-j) * u_j).
/// This deliberately does not call the simulator or ModelHypothesis::predict.
fn closed_form_signature(
    model: &ModelHypothesis,
    horizon: usize,
    sequence_count: usize,
) -> Result<Vec<i64>, NumericError> {
    let mut signature = Vec::with_capacity(sequence_count * horizon);

    for code in 0..sequence_count {
        let inputs = decode_sequence(code, horizon);
        for time in 0..horizon {
            let latent = match model.mechanism {
                Mechanism::DirectResponse => model
                    .gain
                    .checked_mul(inputs[time])
                    .ok_or(NumericError::ArithmeticOverflow)?,
                Mechanism::LeakyMemory => {
                    let decay = model.decay.ok_or(NumericError::InvalidParameterBinding)?;
                    let initial = model
                        .initial_state
                        .ok_or(NumericError::InvalidParameterBinding)?;
                    let mut value = decay
                        .checked_pow(time as u32)
                        .and_then(|power| power.checked_mul(initial))
                        .ok_or(NumericError::ArithmeticOverflow)?;
                    for (index, input) in inputs.iter().enumerate().take(time) {
                        let exponent = (time - index - 1) as u32;
                        let contribution = decay
                            .checked_pow(exponent)
                            .and_then(|power| model.gain.checked_mul(power))
                            .and_then(|weighted_gain| weighted_gain.checked_mul(*input))
                            .ok_or(NumericError::ArithmeticOverflow)?;
                        value = value
                            .checked_add(contribution)
                            .ok_or(NumericError::ArithmeticOverflow)?;
                    }
                    value
                }
            };
            let measured = match model.sensor {
                SensorMap::Identity => latent,
                SensorMap::Negate => -latent,
                SensorMap::Double => latent * 2,
                SensorMap::ConstantZero => 0,
            };
            signature.push(measured);
        }
    }
    Ok(signature)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeRecommendation {
    pub interventions: Vec<i64>,
    pub outcome_partition_count: usize,
    pub worst_case_remaining_models: usize,
    pub candidate_model_count: usize,
    pub query_cost: usize,
    pub guarantees_unique_model: bool,
}

/// Exact finite partition baseline. It maximizes outcome partitions, then
/// minimizes the largest remaining model class, then minimizes query count.
/// It is not EVSI, expected utility, or an optimal adaptive policy.
pub fn recommend_discriminating_probe(
    log: &ObservationLog,
    report: &DiscoveryReport,
    max_horizon: usize,
) -> Result<Option<ProbeRecommendation>, NumericError> {
    log.validate()?;
    validate_horizon(max_horizon)?;
    if report.compatible_models.is_empty() {
        return Ok(None);
    }

    let mut best: Option<ProbeRecommendation> = None;
    for horizon in 1..=max_horizon {
        let sequence_count = 3usize
            .checked_pow(horizon as u32)
            .ok_or(NumericError::OracleHorizonOutOfRange)?;
        for code in 0..sequence_count {
            let interventions = decode_sequence(code, horizon);
            let mut partitions: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
            for model in &report.compatible_models {
                let prediction = model.predict_continuation(&log.interventions, &interventions)?;
                *partitions.entry(prediction).or_insert(0) += 1;
            }
            let partition_count = partitions.len();
            let worst = partitions.values().copied().max().unwrap_or(0);
            let candidate = ProbeRecommendation {
                query_cost: interventions.len(),
                guarantees_unique_model: worst <= 1,
                interventions,
                outcome_partition_count: partition_count,
                worst_case_remaining_models: worst,
                candidate_model_count: report.compatible_models.len(),
            };
            let replace = match &best {
                None => true,
                Some(current) => {
                    candidate.outcome_partition_count > current.outcome_partition_count
                        || (candidate.outcome_partition_count == current.outcome_partition_count
                            && candidate.worst_case_remaining_models
                                < current.worst_case_remaining_models)
                        || (candidate.outcome_partition_count == current.outcome_partition_count
                            && candidate.worst_case_remaining_models
                                == current.worst_case_remaining_models
                            && candidate.query_cost < current.query_cost)
                        || (candidate.outcome_partition_count == current.outcome_partition_count
                            && candidate.worst_case_remaining_models
                                == current.worst_case_remaining_models
                            && candidate.query_cost == current.query_cost
                            && candidate.interventions < current.interventions)
                }
            };
            if replace {
                best = Some(candidate);
            }
        }
    }
    Ok(best)
}

fn validate_horizon(horizon: usize) -> Result<(), NumericError> {
    if horizon == 0 || horizon > MAX_ORACLE_HORIZON {
        return Err(NumericError::OracleHorizonOutOfRange);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericError {
    InvalidGain,
    InvalidParameterBinding,
    InterventionOutOfRange,
    MismatchedObservationLengths,
    EmptyExperiment,
    ArithmeticOverflow,
    OracleHorizonOutOfRange,
}

impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGain => write!(
                f,
                "gain must be within the world-generator range 1..={MAX_WORLD_GAIN}"
            ),
            Self::InvalidParameterBinding => write!(
                f,
                "direct models have no decay or initial state; leaky-memory models require both"
            ),
            Self::InterventionOutOfRange => {
                write!(f, "interventions must be one of -1, 0 or 1")
            }
            Self::MismatchedObservationLengths => {
                write!(f, "intervention and measurement lengths differ")
            }
            Self::EmptyExperiment => write!(f, "at least one observation is required"),
            Self::ArithmeticOverflow => write!(f, "numeric world arithmetic overflowed"),
            Self::OracleHorizonOutOfRange => write!(
                f,
                "oracle horizon must be between 1 and {MAX_ORACLE_HORIZON}"
            ),
        }
    }
}

impl std::error::Error for NumericError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaky_truth() -> ModelHypothesis {
        ModelHypothesis {
            mechanism: Mechanism::LeakyMemory,
            gain: 2,
            decay: Some(1),
            initial_state: Some(1),
            sensor: SensorMap::Identity,
        }
    }

    fn direct_truth() -> ModelHypothesis {
        ModelHypothesis {
            mechanism: Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: SensorMap::Identity,
        }
    }

    #[test]
    fn leaky_world_emits_state_before_applying_current_intervention() {
        let mut world = NumericWorld::new(leaky_truth()).unwrap();
        let log = world.run(&[1, -1, 1, 0]).unwrap();
        assert_eq!(log.measurements, vec![1, 3, 1, 3]);
    }

    #[test]
    fn a0_recovers_numeric_coefficients_when_structure_is_supplied() {
        let mut world = NumericWorld::new(leaky_truth()).unwrap();
        let log = world.run(&[1, -1, 1, 0]).unwrap();
        let report = discover(&log, StructureAccess::Supplied(Mechanism::LeakyMemory)).unwrap();

        assert_eq!(report.disposition, DiscoveryDisposition::UniqueModel);
        assert!(report.coefficients_identified);
        assert_eq!(report.compatible_models, vec![leaky_truth()]);
    }

    #[test]
    fn a2_recovers_structure_coefficients_and_sensor_from_same_trace() {
        let mut world = NumericWorld::new(leaky_truth()).unwrap();
        let log = world.run(&[1, -1, 1, 0]).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();

        assert_eq!(report.disposition, DiscoveryDisposition::UniqueModel);
        assert!(report.structure_identified);
        assert!(report.coefficients_identified);
        assert!(report.measurement_identified);
        assert_eq!(report.compatible_models, vec![leaky_truth()]);
    }

    #[test]
    fn direct_gain_and_sensor_scale_remain_an_explicit_parameter_equivalence() {
        let truth = ModelHypothesis {
            mechanism: Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: SensorMap::Identity,
        };
        let mut world = NumericWorld::new(truth).unwrap();
        let log = world.run(&[-1, 0, 1]).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();

        assert!(report.structure_identified);
        assert!(!report.coefficients_identified);
        assert!(!report.measurement_identified);
        assert!(report.compatible_models.contains(&ModelHypothesis {
            mechanism: Mechanism::DirectResponse,
            gain: 1,
            decay: None,
            initial_state: None,
            sensor: SensorMap::Double,
        }));
    }

    #[test]
    fn constant_sensor_preserves_cross_structure_non_identifiability() {
        let truth = ModelHypothesis {
            sensor: SensorMap::ConstantZero,
            ..leaky_truth()
        };
        let mut world = NumericWorld::new(truth).unwrap();
        let log = world.run(&[-1, 0, 1, 1, -1]).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();

        assert_eq!(report.disposition, DiscoveryDisposition::StructureAmbiguous);
        assert_eq!(
            report.compatible_structures,
            vec![Mechanism::DirectResponse, Mechanism::LeakyMemory]
        );
        assert_eq!(report.compatible_sensors, vec![SensorMap::ConstantZero]);
    }

    #[test]
    fn partition_probe_separates_six_remaining_numeric_models() {
        let mut world = NumericWorld::new(leaky_truth()).unwrap();
        let mut log = world.run(&[0, 0]).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.compatible_models.len(), 6);

        let probe = recommend_discriminating_probe(&log, &report, 3)
            .unwrap()
            .unwrap();
        assert_eq!(probe.outcome_partition_count, 6);
        assert_eq!(probe.worst_case_remaining_models, 1);
        assert!(probe.guarantees_unique_model);

        log.append(&world.run(&probe.interventions).unwrap());
        let updated = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(updated.compatible_models, vec![leaky_truth()]);
    }

    #[test]
    fn exhaustive_oracle_keeps_direct_gain_sensor_equivalence_together() {
        let classes = oracle_equivalence_classes(3).unwrap();
        let identity_gain_two = ModelHypothesis {
            mechanism: Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: SensorMap::Identity,
        };
        let double_gain_one = ModelHypothesis {
            mechanism: Mechanism::DirectResponse,
            gain: 1,
            decay: None,
            initial_state: None,
            sensor: SensorMap::Double,
        };

        assert!(classes.iter().any(|class| {
            class.members.contains(&identity_gain_two) && class.members.contains(&double_gain_one)
        }));
    }

    #[test]
    fn exhaustive_oracle_separates_identifiable_direct_and_memory_models() {
        let classes = oracle_equivalence_classes(3).unwrap();
        let direct = ModelHypothesis {
            mechanism: Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: SensorMap::Identity,
        };
        let memory = ModelHypothesis {
            mechanism: Mechanism::LeakyMemory,
            gain: 2,
            decay: Some(1),
            initial_state: Some(1),
            sensor: SensorMap::Identity,
        };
        assert!(classes.iter().any(|class| class.members.contains(&direct)));
        assert!(classes.iter().any(|class| class.members.contains(&memory)));
        assert!(
            !classes.iter().any(|class| {
                class.members.contains(&direct) && class.members.contains(&memory)
            })
        );
    }

    #[test]
    fn closed_form_oracle_matches_simulator_for_every_model_and_input_sequence() {
        let horizon = 4;
        let sequence_count = 3usize.pow(horizon as u32);
        for model in candidate_models(StructureAccess::Withheld) {
            let signature = closed_form_signature(&model, horizon, sequence_count).unwrap();
            let mut offset = 0;
            for code in 0..sequence_count {
                let inputs = decode_sequence(code, horizon);
                let predicted = model.predict(&inputs).unwrap();
                assert_eq!(
                    &signature[offset..offset + horizon],
                    predicted.as_slice(),
                    "oracle mismatch for model {model:?} and inputs {inputs:?}"
                );
                offset += horizon;
            }
        }
    }

    #[test]
    fn invalid_inputs_and_parameter_bindings_fail_closed() {
        assert!(
            NumericWorld::new(ModelHypothesis {
                gain: 0,
                ..leaky_truth()
            })
            .is_err()
        );
        assert!(
            NumericWorld::new(ModelHypothesis {
                mechanism: Mechanism::DirectResponse,
                gain: 1,
                decay: Some(1),
                initial_state: None,
                sensor: SensorMap::Identity,
            })
            .is_err()
        );
        let mut world = NumericWorld::new(leaky_truth()).unwrap();
        assert!(world.run(&[2]).is_err());
    }

    #[test]
    fn world_generator_accepts_bounded_gains_outside_candidate_catalogue() {
        let out_of_catalogue = ModelHypothesis {
            gain: 7,
            ..direct_truth()
        };
        let mut world = NumericWorld::new(out_of_catalogue).unwrap();
        let log = world.run(&[1]).unwrap();
        assert_eq!(log.measurements, vec![7]);
        assert!(!candidate_models(StructureAccess::Withheld).contains(&out_of_catalogue));
        assert!(
            ModelHypothesis {
                gain: MAX_WORLD_GAIN + 1,
                ..direct_truth()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn impossible_observations_do_not_force_a_nearest_model() {
        let log = ObservationLog {
            interventions: vec![-1, 0, 1],
            measurements: vec![999, 999, 999],
        };
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::NoCompatibleModel);
        assert!(report.compatible_models.is_empty());
        assert!(
            recommend_discriminating_probe(&log, &report, 2)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn oracle_horizon_is_bounded() {
        assert!(oracle_equivalence_classes(0).is_err());
        assert!(oracle_equivalence_classes(MAX_ORACLE_HORIZON + 1).is_err());
    }
}
