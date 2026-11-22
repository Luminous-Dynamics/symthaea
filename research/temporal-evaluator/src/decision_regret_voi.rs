//! Evaluator-side robust value of an observation under decision regret.
//!
//! This module composes the finite hidden-world/action-loss contract in
//! decision_observability.rs. It adds no candidate sensor policy, probability
//! calibrator, expected-value oracle, or acquisition executor.
//!
//! A support manifest enumerates every possible candidate-visible outcome for
//! each hidden world. Outcomes available after the decision deadline collapse
//! into one no-result class. The evaluator then calculates the worst-case
//! minimax-regret floor over every possible visible outcome.
//!
//! Probabilistic posterior and expected Bayes-risk calculations belong to the
//! information-channel owner (#12567 / #12551). This module reports only the
//! complementary robust support bound. A noisy full-support sensor can have
//! positive expected value while providing zero worst-case guarantee.
//!
//! Hidden worlds, outcome support, and action losses are evaluator-only.
//! Package separation is not an OS isolation boundary; #9437 remains required.

use crate::decision_observability::{
    CounterfactualWorld, DecisionObservabilityError, ObservationalEquivalenceClass,
    evaluate_decision_observability,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_OUTCOME_ROWS: usize = 65_536;
const MAX_REGRET_PPM: u32 = 1_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SensorOutcomeSupport {
    /// Evaluator-only world identity from the base equivalence class.
    pub evaluator_world_id: u64,
    /// SHA-256 of the complete candidate-visible transcript after appending
    /// the result, including event and availability times, status and masks.
    /// Identical transcripts must have identical digests across worlds.
    pub candidate_visible_outcome_sha256: [u8; 32],
    /// Absolute episode-clock time when the complete result is accessible to
    /// the candidate policy. None means no result is available.
    pub available_at_us: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationSupportModel {
    /// Stable identity of the exact sensor model/configuration.
    pub sensor_model_sha256: [u8; 32],
    /// Commitment to a separately reviewed complete possible-outcome support
    /// artifact. Nonzero binding does not itself prove support completeness.
    pub support_evidence_sha256: [u8; 32],
    /// Deadline on the same clock as candidate-visible availability.
    pub decision_deadline_us: u64,
    /// Task-declared maximum acceptable worst-case regret, in PPM.
    /// This is a decision-quality threshold, not a safety/authority gate.
    pub decision_regret_tolerance_ppm: u32,
    /// Rows enumerate possible outcomes, not probabilities. Probabilities and
    /// posterior qualification remain owned by #12567/#12551.
    pub possible_outcomes: Vec<SensorOutcomeSupport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum CandidateVisibleOutcome {
    Delivered {
        transcript_sha256: [u8; 32],
        available_at_us: u64,
    },
    NoResultByDeadline,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationOutcomeRegretBound {
    /// None denotes the single candidate-visible no-result-by-deadline class.
    pub candidate_visible_outcome_sha256: Option<[u8; 32]>,
    /// Complete candidate-visible availability time; None means no result.
    pub available_at_us: Option<u64>,
    pub possible_world_count: u64,
    /// Evaluator-only minimax-regret-optimal action after this visible outcome.
    pub minimax_regret_action_id: String,
    pub worst_case_regret_ppm: u32,
    pub information_set_sufficient: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RobustDecisionRegretObservationCertificate {
    pub equivalence_class_id: u64,
    pub sensor_model_sha256: [u8; 32],
    pub support_evidence_sha256: [u8; 32],
    pub decision_deadline_us: u64,
    pub decision_regret_tolerance_ppm: u32,
    pub baseline_worst_case_regret_ppm: u32,
    pub post_observation_worst_case_regret_ppm: u32,
    /// Whether some deterministic action is within tolerance for the entire
    /// current information set; this does not qualify the candidate's chosen action.
    pub baseline_information_set_sufficient: bool,
    /// Whether every possible observation result leaves an information set
    /// with at least one deterministic action within the declared tolerance.
    pub post_observation_information_set_sufficient: bool,
    /// Guaranteed reduction across every outcome in the declared support.
    pub guaranteed_regret_reduction_ppm: u32,
    pub outcomes: Vec<ObservationOutcomeRegretBound>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RobustObservationVoiError {
    EmptySensorModelIdentity,
    EmptySupportEvidenceCommitment,
    RegretToleranceOutOfRange { tolerance_ppm: u32 },
    TooManyOutcomeRows,
    OutcomeWorldUnknown { world_id: u64 },
    WorldHasNoPossibleOutcome { world_id: u64 },
    EmptyDeliveredOutcomeDigest { world_id: u64 },
    OutcomeDigestAvailabilityMismatch {
        candidate_visible_outcome_sha256: [u8; 32],
        first_available_at_us: u64,
        conflicting_available_at_us: u64,
    },
    ArithmeticOverflow,
    DecisionObservability(DecisionObservabilityError),
}

impl fmt::Display for RobustObservationVoiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySensorModelIdentity => write!(formatter, "sensor-model digest must be nonzero"),
            Self::EmptySupportEvidenceCommitment => write!(formatter, "outcome-support evidence digest must be nonzero"),
            Self::RegretToleranceOutOfRange { tolerance_ppm } => write!(formatter, "decision-regret tolerance {tolerance_ppm} exceeds the one-million PPM scale"),
            Self::TooManyOutcomeRows => write!(formatter, "sensor outcome rows exceed the bounded evaluator limit"),
            Self::OutcomeWorldUnknown { world_id } => write!(formatter, "sensor outcome references unknown world {world_id}"),
            Self::WorldHasNoPossibleOutcome { world_id } => write!(formatter, "world {world_id} has no declared possible sensor outcome"),
            Self::EmptyDeliveredOutcomeDigest { world_id } => write!(formatter, "delivered outcome for world {world_id} has an empty candidate-visible transcript digest"),
            Self::OutcomeDigestAvailabilityMismatch {
                first_available_at_us,
                conflicting_available_at_us,
                ..
            } => write!(
                formatter,
                "candidate-visible transcript digest binds conflicting availability times {first_available_at_us} and {conflicting_available_at_us}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "robust observation-regret arithmetic overflowed"),
            Self::DecisionObservability(error) => write!(formatter, "base decision-observability qualification failed: {error}"),
        }
    }
}

impl std::error::Error for RobustObservationVoiError {}

impl From<DecisionObservabilityError> for RobustObservationVoiError {
    fn from(value: DecisionObservabilityError) -> Self {
        Self::DecisionObservability(value)
    }
}

fn is_zero_digest(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn normalize_visible_outcome(
    row: &SensorOutcomeSupport,
    deadline_us: u64,
) -> Result<CandidateVisibleOutcome, RobustObservationVoiError> {
    match row.available_at_us {
        Some(available_at_us) if available_at_us <= deadline_us => {
            if is_zero_digest(&row.candidate_visible_outcome_sha256) {
                return Err(RobustObservationVoiError::EmptyDeliveredOutcomeDigest {
                    world_id: row.evaluator_world_id,
                });
            }
            Ok(CandidateVisibleOutcome::Delivered {
                transcript_sha256: row.candidate_visible_outcome_sha256,
                available_at_us,
            })
        }
        Some(_) | None => Ok(CandidateVisibleOutcome::NoResultByDeadline),
    }
}

fn validate_and_partition_support(
    class: &ObservationalEquivalenceClass,
    model: &ObservationSupportModel,
) -> Result<
    BTreeMap<CandidateVisibleOutcome, BTreeSet<u64>>,
    RobustObservationVoiError,
> {
    if model.possible_outcomes.len() > MAX_OUTCOME_ROWS {
        return Err(RobustObservationVoiError::TooManyOutcomeRows);
    }

    let world_ids: BTreeSet<u64> = class
        .alternatives
        .iter()
        .map(|world| world.evaluator_world_id)
        .collect();
    let mut per_world: BTreeMap<u64, BTreeSet<CandidateVisibleOutcome>> =
        BTreeMap::new();
    // The transcript digest includes availability time. A digest collision
    // across different times is therefore malformed input, not a second
    // candidate-visible outcome with the same transcript identity.
    let mut availability_by_digest: BTreeMap<[u8; 32], u64> = BTreeMap::new();

    for row in &model.possible_outcomes {
        if !world_ids.contains(&row.evaluator_world_id) {
            return Err(RobustObservationVoiError::OutcomeWorldUnknown {
                world_id: row.evaluator_world_id,
            });
        }
        let outcome = normalize_visible_outcome(row, model.decision_deadline_us)?;
        if let CandidateVisibleOutcome::Delivered {
            transcript_sha256,
            available_at_us,
        } = outcome
        {
            if let Some(first_available_at_us) =
                availability_by_digest.insert(transcript_sha256, available_at_us)
            {
                if first_available_at_us != available_at_us {
                    return Err(
                        RobustObservationVoiError::OutcomeDigestAvailabilityMismatch {
                            candidate_visible_outcome_sha256: transcript_sha256,
                            first_available_at_us,
                            conflicting_available_at_us: available_at_us,
                        },
                    );
                }
            }
        }
        per_world
            .entry(row.evaluator_world_id)
            .or_default()
            .insert(outcome);
    }

    let mut by_outcome: BTreeMap<CandidateVisibleOutcome, BTreeSet<u64>> =
        BTreeMap::new();
    for world_id in world_ids {
        let outcomes = per_world
            .get(&world_id)
            .ok_or(RobustObservationVoiError::WorldHasNoPossibleOutcome {
                world_id,
            })?;
        if outcomes.is_empty() {
            return Err(RobustObservationVoiError::WorldHasNoPossibleOutcome {
                world_id,
            });
        }
        for outcome in outcomes {
            by_outcome
                .entry(*outcome)
                .or_default()
                .insert(world_id);
        }
    }

    Ok(by_outcome)
}

fn evaluate_outcome_regret_floor(
    class: &ObservationalEquivalenceClass,
    possible_worlds: &BTreeSet<u64>,
) -> Result<(String, u32), RobustObservationVoiError> {
    let reduced = ObservationalEquivalenceClass {
        equivalence_class_id: class.equivalence_class_id,
        alternatives: class
            .alternatives
            .iter()
            .filter(|world| possible_worlds.contains(&world.evaluator_world_id))
            .cloned()
            .collect::<Vec<CounterfactualWorld>>(),
    };
    let certificate = evaluate_decision_observability(&reduced)?;
    Ok((
        certificate.minimax_regret_action_id,
        certificate.unavoidable_worst_case_regret_ppm,
    ))
}

/// Compute the finite-support robust value of one sensor observation.
///
/// The support artifact must conservatively enumerate every candidate-visible
/// outcome with nonzero possibility in each hidden world. If support is
/// incomplete or unknown, this finite-set certificate is not a valid
/// worst-case guarantee. Probability calibration and expected-value analysis
/// are intentionally delegated to #12567/#12551.
pub fn evaluate_robust_decision_regret_observation(
    class: &ObservationalEquivalenceClass,
    model: &ObservationSupportModel,
) -> Result<RobustDecisionRegretObservationCertificate, RobustObservationVoiError> {
    if is_zero_digest(&model.sensor_model_sha256) {
        return Err(RobustObservationVoiError::EmptySensorModelIdentity);
    }
    if is_zero_digest(&model.support_evidence_sha256) {
        return Err(RobustObservationVoiError::EmptySupportEvidenceCommitment);
    }
    if model.decision_regret_tolerance_ppm > MAX_REGRET_PPM {
        return Err(RobustObservationVoiError::RegretToleranceOutOfRange {
            tolerance_ppm: model.decision_regret_tolerance_ppm,
        });
    }

    let baseline = evaluate_decision_observability(class)?;
    let partitions = validate_and_partition_support(class, model)?;

    let mut post_worst_case_regret = 0_u32;
    let mut outcome_bounds = Vec::with_capacity(partitions.len());

    for (outcome, possible_worlds) in partitions {
        let (minimax_action, worst_case_regret) =
            evaluate_outcome_regret_floor(class, &possible_worlds)?;
        post_worst_case_regret = post_worst_case_regret.max(worst_case_regret);

        let (candidate_visible_outcome_sha256, available_at_us) = match outcome {
            CandidateVisibleOutcome::Delivered {
                transcript_sha256,
                available_at_us,
            } => (Some(transcript_sha256), Some(available_at_us)),
            CandidateVisibleOutcome::NoResultByDeadline => (None, None),
        };
        let possible_world_count = u64::try_from(possible_worlds.len())
            .map_err(|_| RobustObservationVoiError::ArithmeticOverflow)?;

        outcome_bounds.push(ObservationOutcomeRegretBound {
            candidate_visible_outcome_sha256,
            available_at_us,
            possible_world_count,
            minimax_regret_action_id: minimax_action,
            worst_case_regret_ppm: worst_case_regret,
            information_set_sufficient: worst_case_regret <= model.decision_regret_tolerance_ppm,
        });
    }

    let guaranteed_regret_reduction_ppm = baseline
        .unavoidable_worst_case_regret_ppm
        .checked_sub(post_worst_case_regret)
        .ok_or(RobustObservationVoiError::ArithmeticOverflow)?;

    Ok(RobustDecisionRegretObservationCertificate {
        equivalence_class_id: class.equivalence_class_id,
        sensor_model_sha256: model.sensor_model_sha256,
        support_evidence_sha256: model.support_evidence_sha256,
        decision_deadline_us: model.decision_deadline_us,
        decision_regret_tolerance_ppm: model.decision_regret_tolerance_ppm,
        baseline_worst_case_regret_ppm: baseline.unavoidable_worst_case_regret_ppm,
        post_observation_worst_case_regret_ppm: post_worst_case_regret,
        baseline_information_set_sufficient: baseline.unavoidable_worst_case_regret_ppm
            <= model.decision_regret_tolerance_ppm,
        post_observation_information_set_sufficient: post_worst_case_regret
            <= model.decision_regret_tolerance_ppm,
        guaranteed_regret_reduction_ppm,
        outcomes: outcome_bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_observability::CandidateViewIdentity;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn view() -> CandidateViewIdentity {
        CandidateViewIdentity {
            observation_history_sha256: digest(1),
            recurrent_state_sha256: digest(2),
            model_configuration_sha256: digest(3),
            policy_seed: 17,
        }
    }

    fn world(
        world_id: u64,
        candidate_action: &str,
        losses: &[(&str, u32)],
    ) -> CounterfactualWorld {
        CounterfactualWorld {
            evaluator_world_id: world_id,
            hidden_world_sha256: digest(if world_id == 1 { 11 } else { 12 }),
            candidate_view: view(),
            candidate_action_id: candidate_action.to_owned(),
            action_losses: losses
                .iter()
                .map(|(action, loss)| crate::decision_observability::ActionLoss {
                    action_id: (*action).to_owned(),
                    loss_ppm: *loss,
                })
                .collect(),
        }
    }

    fn opposed_worlds() -> ObservationalEquivalenceClass {
        ObservationalEquivalenceClass {
            equivalence_class_id: 77,
            alternatives: vec![
                world(1, "left", &[("left", 0), ("right", 1_000_000)]),
                world(2, "left", &[("left", 1_000_000), ("right", 0)]),
            ],
        }
    }

    fn model(rows: Vec<SensorOutcomeSupport>, deadline: u64) -> ObservationSupportModel {
        ObservationSupportModel {
            sensor_model_sha256: digest(4),
            support_evidence_sha256: digest(5),
            decision_deadline_us: deadline,
            decision_regret_tolerance_ppm: 0,
            possible_outcomes: rows,
        }
    }

    fn row(
        world_id: u64,
        outcome_digest: u8,
        available_at_us: Option<u64>,
    ) -> SensorOutcomeSupport {
        SensorOutcomeSupport {
            evaluator_world_id: world_id,
            candidate_visible_outcome_sha256: digest(outcome_digest),
            available_at_us,
        }
    }

    #[test]
    fn perfect_on_time_observation_removes_decision_relevant_ambiguity() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, Some(100)), row(2, 20, Some(100))],
            100,
        );
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.baseline_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 0);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 1_000_000);
        assert!(!result.baseline_information_set_sufficient);
        assert!(result.post_observation_information_set_sufficient);
        assert_eq!(result.outcomes.len(), 2);
    }

    #[test]
    fn full_support_noisy_sensor_has_zero_worst_case_guarantee() {
        let class = opposed_worlds();
        let model = model(
            vec![
                row(1, 10, Some(50)),
                row(1, 20, Some(50)),
                row(2, 10, Some(50)),
                row(2, 20, Some(50)),
            ],
            100,
        );
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.baseline_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 0);
        assert!(!result.post_observation_information_set_sufficient);
        assert_eq!(result.outcomes.len(), 2);
        assert!(result.outcomes.iter().all(|outcome| outcome.possible_world_count == 2));
    }

    #[test]
    fn decision_irrelevant_hidden_variation_has_zero_regret_value() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 78,
            alternatives: vec![
                world(1, "continue", &[("continue", 0), ("abstain", 100_000)]),
                world(2, "continue", &[("continue", 0), ("abstain", 200_000)]),
            ],
        };
        let model = model(
            vec![row(1, 30, Some(40)), row(2, 40, Some(40))],
            100,
        );
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.baseline_worst_case_regret_ppm, 0);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 0);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 0);
        assert!(result.baseline_information_set_sufficient);
        assert!(result.post_observation_information_set_sufficient);
    }

    #[test]
    fn late_observations_collapse_to_no_result_by_deadline() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, Some(101)), row(2, 20, Some(101))],
            100,
        );
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].candidate_visible_outcome_sha256, None);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 0);
        assert!(!result.post_observation_information_set_sufficient);
    }

    #[test]
    fn delivery_exactly_at_deadline_is_candidate_visible() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, Some(100)), row(2, 20, Some(100))],
            100,
        );
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.post_observation_worst_case_regret_ppm, 0);
        assert!(result.outcomes.iter().all(|outcome| outcome.available_at_us == Some(100)));
    }

    #[test]
    fn absent_observation_is_an_explicit_outcome() {
        let class = opposed_worlds();
        let model = model(vec![row(1, 0, None), row(2, 0, None)], 100);
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].candidate_visible_outcome_sha256, None);
        assert_eq!(result.outcomes[0].possible_world_count, 2);
    }

    #[test]
    fn duplicate_rows_with_same_visible_transcript_are_deduplicated() {
        let class = opposed_worlds();
        let model = model(
            vec![
                row(1, 10, Some(50)),
                row(1, 10, Some(50)),
                row(2, 20, Some(50)),
                row(2, 20, Some(50)),
            ],
            100,
        );
        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.outcomes.len(), 2);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 1_000_000);
    }

    #[test]
    fn rejects_same_transcript_digest_with_conflicting_availability_times() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, Some(40)), row(2, 10, Some(80))],
            100,
        );
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::OutcomeDigestAvailabilityMismatch {
                candidate_visible_outcome_sha256: digest(10),
                first_available_at_us: 40,
                conflicting_available_at_us: 80,
            })
        );
    }

    #[test]
    fn rejects_unknown_world_ids() {
        let class = opposed_worlds();
        let model = model(vec![row(1, 10, Some(50)), row(3, 20, Some(50))], 100);
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::OutcomeWorldUnknown { world_id: 3 })
        );
    }

    #[test]
    fn rejects_worlds_without_declared_outcomes() {
        let class = opposed_worlds();
        let model = model(vec![row(1, 10, Some(50))], 100);
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::WorldHasNoPossibleOutcome { world_id: 2 })
        );
    }

    #[test]
    fn epsilon_regret_sufficiency_accepts_near_optimal_common_action() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 79,
            alternatives: vec![
                world(1, "left", &[("left", 0), ("right", 100_000)]),
                world(2, "left", &[("left", 100_000), ("right", 0)]),
            ],
        };
        let mut model = model(
            vec![row(1, 10, Some(50)), row(2, 10, Some(50))],
            100,
        );
        model.decision_regret_tolerance_ppm = 100_000;

        let exact = evaluate_decision_observability(&class).unwrap();
        assert!(!exact.decision_identifiable);
        assert_eq!(exact.unavoidable_worst_case_regret_ppm, 100_000);

        let result = evaluate_robust_decision_regret_observation(&class, &model).unwrap();
        assert!(result.baseline_information_set_sufficient);
        assert!(result.post_observation_information_set_sufficient);
        assert!(result.outcomes.iter().all(|outcome| outcome.information_set_sufficient));
    }

    #[test]
    fn rejects_regret_tolerance_outside_declared_scale() {
        let class = opposed_worlds();
        let mut model = model(vec![row(1, 10, Some(50)), row(2, 20, Some(50))], 100);
        model.decision_regret_tolerance_ppm = 1_000_001;
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::RegretToleranceOutOfRange {
                tolerance_ppm: 1_000_001
            })
        );
    }

    #[test]
    fn rejects_empty_sensor_or_support_commitments_and_delivered_digests() {
        let class = opposed_worlds();
        let mut model = model(vec![row(1, 10, Some(50)), row(2, 20, Some(50))], 100);
        model.sensor_model_sha256 = [0; 32];
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::EmptySensorModelIdentity)
        );
        model.sensor_model_sha256 = digest(4);
        model.support_evidence_sha256 = [0; 32];
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::EmptySupportEvidenceCommitment)
        );
        model.support_evidence_sha256 = digest(5);
        model.possible_outcomes[0].candidate_visible_outcome_sha256 = [0; 32];
        assert_eq!(
            evaluate_robust_decision_regret_observation(&class, &model),
            Err(RobustObservationVoiError::EmptyDeliveredOutcomeDigest { world_id: 1 })
        );
    }
}
