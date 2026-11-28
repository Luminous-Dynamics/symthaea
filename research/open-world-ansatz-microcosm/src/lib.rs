//! Executable finite-state reference world for ansatz-free discovery.
//! The simulator uses an explicit state transition. The oracle uses a separate
//! closed-form trace construction; this is a finite synthetic reference, not
//! a general equation-discovery engine or physical-world simulator.

pub mod campaign;
pub mod finite_campaign;
pub mod interaction;
pub mod numeric;
pub mod policy;
pub mod process;
pub mod protocol;
pub mod strategies;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_ORACLE_HORIZON: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mechanism {
    Direct,
    HiddenLag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementMap {
    Identity,
    Invert,
    ConstantZero,
}

impl MeasurementMap {
    fn apply(self, latent: bool) -> bool {
        match self {
            Self::Identity => latent,
            Self::Invert => !latent,
            Self::ConstantZero => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "mechanism", rename_all = "snake_case")]
pub enum StructureAccess {
    Supplied(Mechanism),
    Withheld,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationLog {
    pub interventions: Vec<bool>,
    pub measurements: Vec<bool>,
}

impl ObservationLog {
    pub fn new() -> Self {
        Self {
            interventions: Vec::new(),
            measurements: Vec::new(),
        }
    }

    pub fn append(&mut self, other: &Self) {
        self.interventions.extend_from_slice(&other.interventions);
        self.measurements.extend_from_slice(&other.measurements);
    }

    fn validate(&self) -> Result<(), MicrocosmError> {
        if self.interventions.is_empty() {
            return Err(MicrocosmError::EmptyExperiment);
        }
        if self.interventions.len() != self.measurements.len() {
            return Err(MicrocosmError::MismatchedObservationLengths);
        }
        Ok(())
    }
}

impl Default for ObservationLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Generating truth is private. run() returns only interventions and sensor
/// readings, never the mechanism, sensor map or hidden-state value.
pub struct HiddenWorld {
    mechanism: Mechanism,
    measurement_map: MeasurementMap,
    hidden_state: Option<bool>,
}

impl HiddenWorld {
    pub fn new(
        mechanism: Mechanism,
        measurement_map: MeasurementMap,
        initial_hidden_state: Option<bool>,
    ) -> Result<Self, MicrocosmError> {
        validate_state_binding(mechanism, initial_hidden_state)?;
        Ok(Self {
            mechanism,
            measurement_map,
            hidden_state: initial_hidden_state,
        })
    }

    /// Execute interventions while preserving hidden state across calls.
    pub fn run(&mut self, interventions: &[bool]) -> ObservationLog {
        let mut measurements = Vec::with_capacity(interventions.len());
        for intervention in interventions {
            let latent = match self.mechanism {
                Mechanism::Direct => *intervention,
                Mechanism::HiddenLag => self.hidden_state.expect("validated hidden state"),
            };
            measurements.push(self.measurement_map.apply(latent));
            if self.mechanism == Mechanism::HiddenLag {
                self.hidden_state = Some(*intervention);
            }
        }
        ObservationLog {
            interventions: interventions.to_vec(),
            measurements,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Hypothesis {
    pub mechanism: Mechanism,
    pub measurement_map: MeasurementMap,
    pub initial_hidden_state: Option<bool>,
}

impl Hypothesis {
    pub fn new(
        mechanism: Mechanism,
        measurement_map: MeasurementMap,
        initial_hidden_state: Option<bool>,
    ) -> Result<Self, MicrocosmError> {
        validate_state_binding(mechanism, initial_hidden_state)?;
        Ok(Self {
            mechanism,
            measurement_map,
            initial_hidden_state,
        })
    }

    pub fn predict(&self, interventions: &[bool]) -> Result<Vec<bool>, MicrocosmError> {
        self.predict_continuation(&[], interventions)
    }

    /// Predict a future intervention sequence after an observed input history.
    pub fn predict_continuation(
        &self,
        history: &[bool],
        future: &[bool],
    ) -> Result<Vec<bool>, MicrocosmError> {
        validate_state_binding(self.mechanism, self.initial_hidden_state)?;
        let mut hidden = self.initial_hidden_state;
        if self.mechanism == Mechanism::HiddenLag {
            for input in history {
                hidden = Some(*input);
            }
        }

        let mut output = Vec::with_capacity(future.len());
        for input in future {
            let latent = match self.mechanism {
                Mechanism::Direct => *input,
                Mechanism::HiddenLag => hidden.ok_or(MicrocosmError::InvalidStateBinding)?,
            };
            output.push(self.measurement_map.apply(latent));
            if self.mechanism == Mechanism::HiddenLag {
                hidden = Some(*input);
            }
        }
        Ok(output)
    }
}

fn validate_state_binding(
    mechanism: Mechanism,
    initial_hidden_state: Option<bool>,
) -> Result<(), MicrocosmError> {
    match (mechanism, initial_hidden_state) {
        (Mechanism::Direct, None) | (Mechanism::HiddenLag, Some(_)) => Ok(()),
        _ => Err(MicrocosmError::InvalidStateBinding),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryDisposition {
    NoCompatibleModel,
    FullyIdentified,
    StructureAndMeasurementIdentifiedInitialStateAmbiguous,
    StructureIdentifiedMeasurementAmbiguous,
    StructureAmbiguous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub access: StructureAccess,
    pub compatible_hypotheses: Vec<Hypothesis>,
    pub compatible_structures: Vec<Mechanism>,
    pub compatible_measurement_maps: Vec<MeasurementMap>,
    pub disposition: DiscoveryDisposition,
}

pub fn candidate_hypotheses(access: StructureAccess) -> Vec<Hypothesis> {
    let mechanisms = match access {
        StructureAccess::Supplied(mechanism) => vec![mechanism],
        StructureAccess::Withheld => vec![Mechanism::Direct, Mechanism::HiddenLag],
    };
    let maps = [
        MeasurementMap::Identity,
        MeasurementMap::Invert,
        MeasurementMap::ConstantZero,
    ];
    let mut candidates = Vec::new();

    for mechanism in mechanisms {
        for measurement_map in maps {
            match mechanism {
                Mechanism::Direct => candidates.push(Hypothesis {
                    mechanism,
                    measurement_map,
                    initial_hidden_state: None,
                }),
                Mechanism::HiddenLag => {
                    for initial_hidden_state in [false, true] {
                        candidates.push(Hypothesis {
                            mechanism,
                            measurement_map,
                            initial_hidden_state: Some(initial_hidden_state),
                        });
                    }
                }
            }
        }
    }
    candidates.sort();
    candidates
}

/// Exact consistency filtering over the small finite hypothesis set.
pub fn discover(
    log: &ObservationLog,
    access: StructureAccess,
) -> Result<DiscoveryReport, MicrocosmError> {
    log.validate()?;
    let mut compatible_hypotheses = Vec::new();
    for hypothesis in candidate_hypotheses(access) {
        if hypothesis.predict(&log.interventions)? == log.measurements {
            compatible_hypotheses.push(hypothesis);
        }
    }
    compatible_hypotheses.sort();

    let compatible_structures: Vec<Mechanism> = compatible_hypotheses
        .iter()
        .map(|hypothesis| hypothesis.mechanism)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let compatible_measurement_maps: Vec<MeasurementMap> = compatible_hypotheses
        .iter()
        .map(|hypothesis| hypothesis.measurement_map)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let disposition = match (
        compatible_hypotheses.len(),
        compatible_structures.len(),
        compatible_measurement_maps.len(),
    ) {
        (0, _, _) => DiscoveryDisposition::NoCompatibleModel,
        (1, 1, 1) => DiscoveryDisposition::FullyIdentified,
        (_, 1, 1) => DiscoveryDisposition::StructureAndMeasurementIdentifiedInitialStateAmbiguous,
        (_, 1, _) => DiscoveryDisposition::StructureIdentifiedMeasurementAmbiguous,
        _ => DiscoveryDisposition::StructureAmbiguous,
    };

    Ok(DiscoveryReport {
        access,
        compatible_hypotheses,
        compatible_structures,
        compatible_measurement_maps,
        disposition,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeRecommendation {
    pub interventions: Vec<bool>,
    pub outcome_partition_count: usize,
    pub worst_case_remaining_hypotheses: usize,
    pub candidate_hypothesis_count: usize,
    pub step_cost: usize,
    pub guarantees_unique_hypothesis: bool,
}

/// Finite partition-maximizing baseline. It is not EVSI, expected utility or
/// a globally optimal adaptive experimental policy.
pub fn recommend_discriminating_probe(
    log: &ObservationLog,
    report: &DiscoveryReport,
    max_horizon: usize,
) -> Result<Option<ProbeRecommendation>, MicrocosmError> {
    log.validate()?;
    validate_horizon(max_horizon)?;
    if report.compatible_hypotheses.is_empty() {
        return Ok(None);
    }

    let mut best: Option<ProbeRecommendation> = None;
    for horizon in 1..=max_horizon {
        let count = 1usize
            .checked_shl(horizon as u32)
            .ok_or(MicrocosmError::OracleHorizonOutOfRange)?;
        for mask in 0..count {
            let interventions: Vec<bool> = (0..horizon)
                .map(|bit| (mask & (1usize << bit)) != 0)
                .collect();
            let mut partitions: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
            for hypothesis in &report.compatible_hypotheses {
                let prediction =
                    hypothesis.predict_continuation(&log.interventions, &interventions)?;
                *partitions.entry(prediction).or_insert(0) += 1;
            }
            let partition_count = partitions.len();
            let worst = partitions.values().copied().max().unwrap_or(0);
            let candidate = ProbeRecommendation {
                step_cost: interventions.len(),
                guarantees_unique_hypothesis: worst <= 1,
                interventions,
                outcome_partition_count: partition_count,
                worst_case_remaining_hypotheses: worst,
                candidate_hypothesis_count: report.compatible_hypotheses.len(),
            };
            let replace = match &best {
                None => true,
                Some(current) => {
                    candidate.outcome_partition_count > current.outcome_partition_count
                        || (candidate.outcome_partition_count == current.outcome_partition_count
                            && candidate.worst_case_remaining_hypotheses
                                < current.worst_case_remaining_hypotheses)
                        || (candidate.outcome_partition_count == current.outcome_partition_count
                            && candidate.worst_case_remaining_hypotheses
                                == current.worst_case_remaining_hypotheses
                            && candidate.step_cost < current.step_cost)
                        || (candidate.outcome_partition_count == current.outcome_partition_count
                            && candidate.worst_case_remaining_hypotheses
                                == current.worst_case_remaining_hypotheses
                            && candidate.step_cost == current.step_cost
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleEquivalenceClass {
    pub members: Vec<Hypothesis>,
}

/// Enumerate all binary intervention sequences and group models with identical
/// complete input/output signatures over that intervention language.
pub fn oracle_equivalence_classes(
    horizon: usize,
) -> Result<Vec<OracleEquivalenceClass>, MicrocosmError> {
    validate_horizon(horizon)?;
    let sequence_count = 1usize
        .checked_shl(horizon as u32)
        .ok_or(MicrocosmError::OracleHorizonOutOfRange)?;
    let mut classes: BTreeMap<Vec<bool>, Vec<Hypothesis>> = BTreeMap::new();

    for hypothesis in candidate_hypotheses(StructureAccess::Withheld) {
        let signature = oracle_signature(&hypothesis, horizon, sequence_count);
        classes.entry(signature).or_default().push(hypothesis);
    }
    Ok(classes
        .into_values()
        .map(|mut members| {
            members.sort();
            OracleEquivalenceClass { members }
        })
        .collect())
}

/// Independent closed-form oracle: y[0]=h[0], y[t]=u[t-1] for hidden lag.
/// This deliberately does not call the simulator or Hypothesis::predict.
fn oracle_signature(hypothesis: &Hypothesis, horizon: usize, sequence_count: usize) -> Vec<bool> {
    let mut signature = Vec::with_capacity(sequence_count * horizon);
    for mask in 0..sequence_count {
        for time in 0..horizon {
            let current_input = (mask & (1usize << time)) != 0;
            let latent = match hypothesis.mechanism {
                Mechanism::Direct => current_input,
                Mechanism::HiddenLag => {
                    if time == 0 {
                        hypothesis
                            .initial_hidden_state
                            .expect("valid hidden-lag hypothesis")
                    } else {
                        (mask & (1usize << (time - 1))) != 0
                    }
                }
            };
            let measured = match hypothesis.measurement_map {
                MeasurementMap::Identity => latent,
                MeasurementMap::Invert => !latent,
                MeasurementMap::ConstantZero => false,
            };
            signature.push(measured);
        }
    }
    signature
}

fn validate_horizon(horizon: usize) -> Result<(), MicrocosmError> {
    if horizon == 0 || horizon > MAX_ORACLE_HORIZON {
        return Err(MicrocosmError::OracleHorizonOutOfRange);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrocosmError {
    InvalidStateBinding,
    MismatchedObservationLengths,
    EmptyExperiment,
    OracleHorizonOutOfRange,
}

impl fmt::Display for MicrocosmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStateBinding => write!(
                f,
                "direct mechanisms require no hidden initial state; hidden-lag mechanisms require one"
            ),
            Self::MismatchedObservationLengths => {
                write!(f, "intervention and measurement lengths differ")
            }
            Self::EmptyExperiment => write!(f, "at least one observation is required"),
            Self::OracleHorizonOutOfRange => write!(
                f,
                "oracle horizon must be between 1 and {MAX_ORACLE_HORIZON}"
            ),
        }
    }
}
impl std::error::Error for MicrocosmError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_for(
        mechanism: Mechanism,
        map: MeasurementMap,
        initial: Option<bool>,
        inputs: &[bool],
    ) -> ObservationLog {
        let mut world = HiddenWorld::new(mechanism, map, initial).unwrap();
        world.run(inputs)
    }

    #[test]
    fn hidden_lag_world_returns_previous_intervention() {
        let log = log_for(
            Mechanism::HiddenLag,
            MeasurementMap::Identity,
            Some(false),
            &[true, false, false],
        );
        assert_eq!(log.measurements, vec![false, true, false]);
    }

    #[test]
    fn a2_recovers_structure_and_sensor_after_discriminating_intervention() {
        let log = log_for(
            Mechanism::HiddenLag,
            MeasurementMap::Identity,
            Some(false),
            &[true, false, false],
        );
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::FullyIdentified);
        assert_eq!(report.compatible_structures, vec![Mechanism::HiddenLag]);
        assert_eq!(
            report.compatible_measurement_maps,
            vec![MeasurementMap::Identity]
        );
        assert_eq!(
            report.compatible_hypotheses,
            vec![
                Hypothesis::new(Mechanism::HiddenLag, MeasurementMap::Identity, Some(false))
                    .unwrap()
            ]
        );
    }

    #[test]
    fn a0_and_a2_share_exact_observations_and_change_only_structure_access() {
        let log = log_for(
            Mechanism::HiddenLag,
            MeasurementMap::Invert,
            Some(false),
            &[true, false, false],
        );
        let a0 = discover(&log, StructureAccess::Supplied(Mechanism::HiddenLag)).unwrap();
        let a2 = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(a0.disposition, DiscoveryDisposition::FullyIdentified);
        assert_eq!(a2.disposition, DiscoveryDisposition::FullyIdentified);
        assert_eq!(a0.compatible_hypotheses[0], a2.compatible_hypotheses[0]);
    }

    #[test]
    fn passive_constant_input_does_not_identify_mechanism() {
        let log = log_for(
            Mechanism::HiddenLag,
            MeasurementMap::Identity,
            Some(false),
            &[false, false, false],
        );
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::StructureAmbiguous);
        assert_eq!(
            report.compatible_structures,
            vec![Mechanism::Direct, Mechanism::HiddenLag]
        );
    }

    #[test]
    fn exact_partition_probe_resolves_informative_hidden_lag_world() {
        let mut world =
            HiddenWorld::new(Mechanism::HiddenLag, MeasurementMap::Identity, Some(false)).unwrap();
        let mut log = world.run(&[false, false, false]);
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        let probe = recommend_discriminating_probe(&log, &report, 3)
            .unwrap()
            .unwrap();
        assert_eq!(probe.interventions, vec![true, false]);
        assert_eq!(probe.outcome_partition_count, 3);
        assert_eq!(probe.worst_case_remaining_hypotheses, 3);
        assert!(!probe.guarantees_unique_hypothesis);
        log.append(&world.run(&probe.interventions));
        let updated = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(updated.disposition, DiscoveryDisposition::FullyIdentified);
        assert_eq!(updated.compatible_structures, vec![Mechanism::HiddenLag]);
    }

    #[test]
    fn uninformative_sensor_preserves_structural_ambiguity() {
        let log = log_for(
            Mechanism::HiddenLag,
            MeasurementMap::ConstantZero,
            Some(false),
            &[true, false, true],
        );
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::StructureAmbiguous);
        assert_eq!(
            report.compatible_measurement_maps,
            vec![MeasurementMap::ConstantZero]
        );
        assert_eq!(
            report.compatible_structures,
            vec![Mechanism::Direct, Mechanism::HiddenLag]
        );
    }

    #[test]
    fn exhaustive_oracle_keeps_constant_sensor_models_in_one_equivalence_class() {
        let classes = oracle_equivalence_classes(3).unwrap();
        let class = classes
            .iter()
            .find(|class| {
                class.members.contains(
                    &Hypothesis::new(
                        Mechanism::HiddenLag,
                        MeasurementMap::ConstantZero,
                        Some(false),
                    )
                    .unwrap(),
                )
            })
            .unwrap();
        assert!(class.members.contains(
            &Hypothesis::new(Mechanism::Direct, MeasurementMap::ConstantZero, None).unwrap()
        ));
        assert!(
            class.members.contains(
                &Hypothesis::new(
                    Mechanism::HiddenLag,
                    MeasurementMap::ConstantZero,
                    Some(true)
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn distorted_measurement_map_is_not_mislabeled_as_world_state() {
        let log = log_for(
            Mechanism::Direct,
            MeasurementMap::Invert,
            None,
            &[true, false, false],
        );
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::FullyIdentified);
        assert_eq!(report.compatible_structures, vec![Mechanism::Direct]);
        assert_eq!(
            report.compatible_measurement_maps,
            vec![MeasurementMap::Invert]
        );
    }

    #[test]
    fn inconsistent_evidence_does_not_force_explanation() {
        let log = ObservationLog {
            interventions: vec![true, false, true],
            measurements: vec![true, true, true],
        };
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::NoCompatibleModel);
        assert!(report.compatible_hypotheses.is_empty());
    }

    #[test]
    fn oracle_separates_direct_and_hidden_lag_under_complete_binary_probe_set() {
        let classes = oracle_equivalence_classes(3).unwrap();
        let direct = Hypothesis::new(Mechanism::Direct, MeasurementMap::Identity, None).unwrap();
        let delayed =
            Hypothesis::new(Mechanism::HiddenLag, MeasurementMap::Identity, Some(false)).unwrap();
        assert!(classes.iter().any(|class| class.members.contains(&direct)));
        assert!(classes.iter().any(|class| class.members.contains(&delayed)));
        assert!(
            !classes
                .iter()
                .any(|class| class.members.contains(&direct) && class.members.contains(&delayed))
        );
    }

    #[test]
    fn invalid_hidden_state_bindings_are_rejected() {
        assert!(
            HiddenWorld::new(Mechanism::Direct, MeasurementMap::Identity, Some(false)).is_err()
        );
        assert!(HiddenWorld::new(Mechanism::HiddenLag, MeasurementMap::Identity, None).is_err());
    }

    #[test]
    fn oracle_rejects_zero_and_excessive_horizons() {
        assert!(oracle_equivalence_classes(0).is_err());
        assert!(oracle_equivalence_classes(MAX_ORACLE_HORIZON + 1).is_err());
    }
}
