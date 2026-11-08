//! Two-input structural-discovery microcosm for variable relevance and
//! interaction discovery. This complements the hidden-state numeric world:
//! it tests whether a researcher can discover omitted variables and cross terms.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const INPUT_LEVELS: [i64; 3] = [-1, 0, 1];
pub const BIAS_VALUES: [i64; 3] = [-1, 0, 1];
pub const COEFFICIENT_VALUES: [i64; 4] = [-2, -1, 1, 2];
pub const FACTORIAL_DESIGN: [[i64; 2]; 9] = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 0],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Structure {
    /// Only the first controllable variable matters.
    SingleInputU,
    /// Only the second controllable variable matters.
    SingleInputV,
    /// Both variables matter, but their effects are separable.
    AdditiveTwoInput,
    /// The effect of one variable depends on the value of the other.
    BilinearInteraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "structure", rename_all = "snake_case")]
pub enum StructureAccess {
    Supplied(Structure),
    Withheld,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelHypothesis {
    pub structure: Structure,
    pub bias: i64,
    pub weight_u: i64,
    /// Zero only when the second variable is absent.
    pub weight_v: i64,
    /// Zero unless the structure is BilinearInteraction.
    pub interaction_weight: i64,
}

impl ModelHypothesis {
    pub fn validate(&self) -> Result<(), InteractionError> {
        if !BIAS_VALUES.contains(&self.bias) {
            return Err(InteractionError::InvalidCoefficient);
        }
        match self.structure {
            Structure::SingleInputU
                if COEFFICIENT_VALUES.contains(&self.weight_u)
                    && self.weight_v == 0
                    && self.interaction_weight == 0 =>
            {
                Ok(())
            }
            Structure::SingleInputV
                if self.weight_u == 0
                    && COEFFICIENT_VALUES.contains(&self.weight_v)
                    && self.interaction_weight == 0 =>
            {
                Ok(())
            }
            Structure::AdditiveTwoInput
                if COEFFICIENT_VALUES.contains(&self.weight_u)
                    && COEFFICIENT_VALUES.contains(&self.weight_v)
                    && self.interaction_weight == 0 =>
            {
                Ok(())
            }
            Structure::BilinearInteraction
                if COEFFICIENT_VALUES.contains(&self.weight_u)
                    && COEFFICIENT_VALUES.contains(&self.weight_v)
                    && COEFFICIENT_VALUES.contains(&self.interaction_weight) =>
            {
                Ok(())
            }
            _ => Err(InteractionError::InvalidStructureBinding),
        }
    }

    pub fn predict(&self, input: [i64; 2]) -> Result<i64, InteractionError> {
        self.validate()?;
        validate_input(input)?;
        let [u, v] = input;
        let u_term = self
            .weight_u
            .checked_mul(u)
            .ok_or(InteractionError::ArithmeticOverflow)?;
        let v_term = self
            .weight_v
            .checked_mul(v)
            .ok_or(InteractionError::ArithmeticOverflow)?;
        let interaction = self
            .interaction_weight
            .checked_mul(u)
            .and_then(|value| value.checked_mul(v))
            .ok_or(InteractionError::ArithmeticOverflow)?;

        self.bias
            .checked_add(u_term)
            .and_then(|value| value.checked_add(v_term))
            .and_then(|value| value.checked_add(interaction))
            .ok_or(InteractionError::ArithmeticOverflow)
    }
}

fn validate_input(input: [i64; 2]) -> Result<(), InteractionError> {
    if input.iter().any(|value| !INPUT_LEVELS.contains(value)) {
        return Err(InteractionError::InterventionOutOfRange);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationLog {
    pub interventions: Vec<[i64; 2]>,
    pub measurements: Vec<i64>,
}

impl ObservationLog {
    pub fn append(&mut self, other: &Self) {
        self.interventions.extend_from_slice(&other.interventions);
        self.measurements.extend_from_slice(&other.measurements);
    }

    fn validate(&self) -> Result<(), InteractionError> {
        if self.interventions.is_empty() {
            return Err(InteractionError::EmptyExperiment);
        }
        if self.interventions.len() != self.measurements.len() {
            return Err(InteractionError::MismatchedObservationLengths);
        }
        for input in &self.interventions {
            validate_input(*input)?;
        }
        Ok(())
    }
}

/// Generating coefficients remain evaluator-side; run() reveals only inputs
/// and measured outputs.
pub struct InteractionWorld {
    truth: ModelHypothesis,
}

impl InteractionWorld {
    pub fn new(truth: ModelHypothesis) -> Result<Self, InteractionError> {
        truth.validate()?;
        Ok(Self { truth })
    }

    pub fn run(&self, interventions: &[[i64; 2]]) -> Result<ObservationLog, InteractionError> {
        let mut measurements = Vec::with_capacity(interventions.len());
        for input in interventions {
            measurements.push(self.truth.predict(*input)?);
        }
        Ok(ObservationLog {
            interventions: interventions.to_vec(),
            measurements,
        })
    }
}

pub fn candidate_models(access: StructureAccess) -> Vec<ModelHypothesis> {
    let structures = match access {
        StructureAccess::Supplied(structure) => vec![structure],
        StructureAccess::Withheld => vec![
            Structure::SingleInputU,
            Structure::SingleInputV,
            Structure::AdditiveTwoInput,
            Structure::BilinearInteraction,
        ],
    };
    let mut candidates = Vec::new();

    for structure in structures {
        for bias in BIAS_VALUES {
            match structure {
                Structure::SingleInputU => {
                    for weight_u in COEFFICIENT_VALUES {
                        candidates.push(ModelHypothesis {
                            structure,
                            bias,
                            weight_u,
                            weight_v: 0,
                            interaction_weight: 0,
                        });
                    }
                }
                Structure::SingleInputV => {
                    for weight_v in COEFFICIENT_VALUES {
                        candidates.push(ModelHypothesis {
                            structure,
                            bias,
                            weight_u: 0,
                            weight_v,
                            interaction_weight: 0,
                        });
                    }
                }
                Structure::AdditiveTwoInput => {
                    for weight_u in COEFFICIENT_VALUES {
                        for weight_v in COEFFICIENT_VALUES {
                            candidates.push(ModelHypothesis {
                                structure,
                                bias,
                                weight_u,
                                weight_v,
                                interaction_weight: 0,
                            });
                        }
                    }
                }
                Structure::BilinearInteraction => {
                    for weight_u in COEFFICIENT_VALUES {
                        for weight_v in COEFFICIENT_VALUES {
                            for interaction_weight in COEFFICIENT_VALUES {
                                candidates.push(ModelHypothesis {
                                    structure,
                                    bias,
                                    weight_u,
                                    weight_v,
                                    interaction_weight,
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
    pub compatible_structures: Vec<Structure>,
    pub structure_identified: bool,
    pub coefficients_identified: bool,
    pub model_identified: bool,
    pub disposition: DiscoveryDisposition,
}

pub fn discover(
    log: &ObservationLog,
    access: StructureAccess,
) -> Result<DiscoveryReport, InteractionError> {
    log.validate()?;
    let mut compatible_models = Vec::new();

    for model in candidate_models(access) {
        let mut matches = true;
        for (input, measurement) in log.interventions.iter().zip(&log.measurements) {
            if model.predict(*input)? != *measurement {
                matches = false;
                break;
            }
        }
        if matches {
            compatible_models.push(model);
        }
    }
    compatible_models.sort();

    let compatible_structures: Vec<Structure> = compatible_models
        .iter()
        .map(|model| model.structure)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let structure_identified = !compatible_models.is_empty() && compatible_structures.len() == 1;
    let model_identified = compatible_models.len() == 1;
    let coefficients_identified = structure_identified && model_identified;
    let disposition = match (
        compatible_models.is_empty(),
        structure_identified,
        model_identified,
    ) {
        (true, _, _) => DiscoveryDisposition::NoCompatibleModel,
        (false, true, true) => DiscoveryDisposition::UniqueModel,
        (false, true, false) => DiscoveryDisposition::StructureIdentifiedParametersAmbiguous,
        _ => DiscoveryDisposition::StructureAmbiguous,
    };

    Ok(DiscoveryReport {
        access,
        compatible_models,
        compatible_structures,
        structure_identified,
        coefficients_identified,
        model_identified,
        disposition,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleEquivalenceClass {
    pub members: Vec<ModelHypothesis>,
}

/// Enumerate each model's response over all nine legal input pairs and group
/// models with identical complete response signatures.
pub fn oracle_equivalence_classes() -> Result<Vec<OracleEquivalenceClass>, InteractionError> {
    let mut classes: BTreeMap<Vec<i64>, Vec<ModelHypothesis>> = BTreeMap::new();
    for model in candidate_models(StructureAccess::Withheld) {
        let signature = closed_form_signature(&model)?;
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

/// Independent algebraic response construction; it does not call predict().
fn closed_form_signature(model: &ModelHypothesis) -> Result<Vec<i64>, InteractionError> {
    let mut signature = Vec::with_capacity(FACTORIAL_DESIGN.len());
    for [u, v] in FACTORIAL_DESIGN {
        let u_term = model
            .weight_u
            .checked_mul(u)
            .ok_or(InteractionError::ArithmeticOverflow)?;
        let v_term = model
            .weight_v
            .checked_mul(v)
            .ok_or(InteractionError::ArithmeticOverflow)?;
        let interaction = model
            .interaction_weight
            .checked_mul(u)
            .and_then(|value| value.checked_mul(v))
            .ok_or(InteractionError::ArithmeticOverflow)?;
        let value = model
            .bias
            .checked_add(u_term)
            .and_then(|value| value.checked_add(v_term))
            .and_then(|value| value.checked_add(interaction))
            .ok_or(InteractionError::ArithmeticOverflow)?;
        signature.push(value);
    }
    Ok(signature)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeRecommendation {
    pub intervention: [i64; 2],
    pub outcome_partition_count: usize,
    pub worst_case_remaining_models: usize,
    pub candidate_model_count: usize,
    pub guarantees_unique_model: bool,
}

/// One-step exhaustive probe baseline for the finite two-input action space.
/// It is not EVSI or a globally optimal multi-step policy.
pub fn recommend_discriminating_probe(
    log: &ObservationLog,
    report: &DiscoveryReport,
) -> Result<Option<ProbeRecommendation>, InteractionError> {
    log.validate()?;
    if report.compatible_models.len() <= 1 {
        return Ok(None);
    }

    let mut best: Option<ProbeRecommendation> = None;
    for intervention in FACTORIAL_DESIGN {
        if log.interventions.contains(&intervention) {
            continue;
        }
        let mut partitions: BTreeMap<i64, usize> = BTreeMap::new();
        for model in &report.compatible_models {
            *partitions.entry(model.predict(intervention)?).or_insert(0) += 1;
        }
        let count = partitions.len();
        let worst = partitions.values().copied().max().unwrap_or(0);
        let candidate = ProbeRecommendation {
            intervention,
            outcome_partition_count: count,
            worst_case_remaining_models: worst,
            candidate_model_count: report.compatible_models.len(),
            guarantees_unique_model: worst <= 1,
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
                        && candidate.intervention < current.intervention)
            }
        };
        if replace {
            best = Some(candidate);
        }
    }
    Ok(best)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionError {
    InvalidCoefficient,
    InvalidStructureBinding,
    InterventionOutOfRange,
    MismatchedObservationLengths,
    EmptyExperiment,
    ArithmeticOverflow,
}

impl fmt::Display for InteractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCoefficient => {
                write!(f, "bias/coefficients are outside the declared search space")
            }
            Self::InvalidStructureBinding => write!(
                f,
                "coefficient presence must match the supplied structural family"
            ),
            Self::InterventionOutOfRange => write!(f, "each input must be one of -1, 0 or 1"),
            Self::MismatchedObservationLengths => {
                write!(f, "intervention and measurement lengths differ")
            }
            Self::EmptyExperiment => write!(f, "at least one intervention is required"),
            Self::ArithmeticOverflow => write!(f, "interaction world arithmetic overflowed"),
        }
    }
}

impl std::error::Error for InteractionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn bilinear_truth() -> ModelHypothesis {
        ModelHypothesis {
            structure: Structure::BilinearInteraction,
            bias: 0,
            weight_u: 1,
            weight_v: 1,
            interaction_weight: 1,
        }
    }

    #[test]
    fn bilinear_world_emits_the_cross_term() {
        let world = InteractionWorld::new(bilinear_truth()).unwrap();
        let log = world.run(&[[1, 1], [1, -1]]).unwrap();
        assert_eq!(log.measurements, vec![3, -1]);
    }

    #[test]
    fn a0_recovers_coefficients_when_bilinear_structure_is_supplied() {
        let world = InteractionWorld::new(bilinear_truth()).unwrap();
        let log = world.run(&FACTORIAL_DESIGN).unwrap();
        let report = discover(
            &log,
            StructureAccess::Supplied(Structure::BilinearInteraction),
        )
        .unwrap();

        assert_eq!(report.disposition, DiscoveryDisposition::UniqueModel);
        assert!(report.coefficients_identified);
        assert_eq!(report.compatible_models, vec![bilinear_truth()]);
    }

    #[test]
    fn a2_discovers_variable_relevance_and_bilinear_interaction() {
        let world = InteractionWorld::new(bilinear_truth()).unwrap();
        let log = world.run(&FACTORIAL_DESIGN).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();

        assert_eq!(report.disposition, DiscoveryDisposition::UniqueModel);
        assert_eq!(
            report.compatible_structures,
            vec![Structure::BilinearInteraction]
        );
        assert_eq!(report.compatible_models, vec![bilinear_truth()]);
    }

    #[test]
    fn a2_can_identify_that_only_the_second_variable_matters() {
        let truth = ModelHypothesis {
            structure: Structure::SingleInputV,
            bias: 1,
            weight_u: 0,
            weight_v: 2,
            interaction_weight: 0,
        };
        let world = InteractionWorld::new(truth).unwrap();
        let log = world.run(&FACTORIAL_DESIGN).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();

        assert_eq!(report.disposition, DiscoveryDisposition::UniqueModel);
        assert_eq!(report.compatible_structures, vec![Structure::SingleInputV]);
        assert_eq!(report.compatible_models, vec![truth]);
    }

    #[test]
    fn collinear_interventions_alias_bias_and_interaction_structure() {
        let world = InteractionWorld::new(bilinear_truth()).unwrap();
        let log = world.run(&[[1, 1], [-1, -1]]).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();

        assert_eq!(report.compatible_models.len(), 5);
        assert_eq!(
            report.compatible_structures,
            vec![
                Structure::SingleInputU,
                Structure::SingleInputV,
                Structure::AdditiveTwoInput,
                Structure::BilinearInteraction
            ]
        );
        assert!(!report.structure_identified);
    }

    #[test]
    fn adaptive_factorial_probes_break_collinear_alias_without_repeating_queries() {
        let world = InteractionWorld::new(bilinear_truth()).unwrap();
        let mut log = world.run(&[[1, 1], [-1, -1]]).unwrap();
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        let first_probe = recommend_discriminating_probe(&log, &report)
            .unwrap()
            .unwrap();

        assert_eq!(first_probe.outcome_partition_count, 4);
        assert_eq!(first_probe.worst_case_remaining_models, 2);
        assert!(!first_probe.guarantees_unique_model);
        assert!(!log.interventions.contains(&first_probe.intervention));

        log.append(&world.run(&[first_probe.intervention]).unwrap());
        let intermediate = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(intermediate.compatible_models.len(), 2);

        let second_probe = recommend_discriminating_probe(&log, &intermediate)
            .unwrap()
            .unwrap();
        assert!(!log.interventions.contains(&second_probe.intervention));
        assert_eq!(second_probe.outcome_partition_count, 2);
        assert_eq!(second_probe.worst_case_remaining_models, 1);
        assert!(second_probe.guarantees_unique_model);

        log.append(&world.run(&[second_probe.intervention]).unwrap());
        let updated = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(updated.compatible_models, vec![bilinear_truth()]);
    }

    #[test]
    fn full_factorial_oracle_identifies_every_declared_candidate_and_variable_role() {
        let classes = oracle_equivalence_classes().unwrap();
        assert_eq!(
            classes.len(),
            candidate_models(StructureAccess::Withheld).len()
        );
        assert!(classes.iter().all(|class| class.members.len() == 1));
    }

    #[test]
    fn closed_form_oracle_matches_predictor_for_every_candidate() {
        for model in candidate_models(StructureAccess::Withheld) {
            assert_eq!(
                closed_form_signature(&model).unwrap(),
                FACTORIAL_DESIGN
                    .iter()
                    .map(|input| model.predict(*input).unwrap())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn invalid_coefficients_and_interventions_fail_closed() {
        assert!(
            InteractionWorld::new(ModelHypothesis {
                weight_v: 1,
                ..ModelHypothesis {
                    structure: Structure::SingleInputU,
                    bias: 0,
                    weight_u: 1,
                    weight_v: 0,
                    interaction_weight: 0,
                }
            })
            .is_err()
        );
        let world = InteractionWorld::new(bilinear_truth()).unwrap();
        assert!(world.run(&[[2, 0]]).is_err());
    }

    #[test]
    fn impossible_observations_do_not_force_a_nearest_structure() {
        let log = ObservationLog {
            interventions: vec![[-1, 0], [0, 0], [1, 0]],
            measurements: vec![999, 999, 999],
        };
        let report = discover(&log, StructureAccess::Withheld).unwrap();
        assert_eq!(report.disposition, DiscoveryDisposition::NoCompatibleModel);
        assert!(report.compatible_models.is_empty());
        assert!(
            recommend_discriminating_probe(&log, &report)
                .unwrap()
                .is_none()
        );
    }
}
