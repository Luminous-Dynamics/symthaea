//! Exact expected-regret evaluation for randomized policies in hidden-world classes.
//!
//! The deterministic decision-observability oracle intentionally requires one
//! emitted action across worlds with the same candidate-visible view. Stochastic
//! policies instead emit one frozen action distribution for that view. This
//! module evaluates that distribution without sampling it and without leaking
//! evaluator-only worlds or action losses to the candidate.
//!
//! Probabilities use integer parts-per-million and must sum to exactly
//! 1_000_000. Expected-loss and expected-regret numerators are retained exactly;
//! reported PPM values are rounded upward. The maximum regret of any action
//! with positive support is reported separately from expected regret. It is a
//! support bound, not a probability-calibrated tail-risk estimate.
//!
//! The deterministic minimax-regret value is included as a comparator only.
//! It is not a lower bound for randomized policies: mixing can improve the
//! worst-case expected regret. No randomized minimax optimizer is implemented
//! here. Candidate/evaluator OS isolation under #9437 remains mandatory.

use crate::decision_observability::{
    ActionLoss, CandidateViewIdentity, CounterfactualWorld, DecisionObservabilityError,
    ObservationalEquivalenceClass, evaluate_decision_observability,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const RANDOMIZED_POLICY_MASS_PPM: u32 = 1_000_000;
const MAX_RANDOMIZED_ALTERNATIVES: usize = 4_096;
const MAX_RANDOMIZED_ACTIONS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateActionProbability {
    pub action_id: String,
    /// Probability mass on the one-million PPM scale. Zero-mass rows are
    /// omitted; every listed action must have strictly positive mass.
    pub probability_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizedPolicyWorld {
    /// Evaluator-only identity. Never expose it to the candidate.
    pub evaluator_world_id: u64,
    pub hidden_world_sha256: [u8; 32],
    pub candidate_view: CandidateViewIdentity,
    /// Evaluator-only loss table for the common admissible action set.
    pub action_losses: Vec<ActionLoss>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizedDecisionClass {
    pub equivalence_class_id: u64,
    /// Opaque identity digest for the candidate policy artifact. This is not
    /// a digest of the action-probability rows and does not authenticate the
    /// candidate's execution or the provenance of this supplied distribution.
    pub candidate_policy_artifact_sha256: [u8; 32],
    pub action_probabilities: Vec<CandidateActionProbability>,
    pub alternatives: Vec<RandomizedPolicyWorld>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizedWorldRegret {
    pub evaluator_world_id: u64,
    pub best_action_loss_ppm: u32,
    /// Exact sum(probability_ppm * action_loss_ppm), denominator 1_000_000.
    pub expected_loss_numerator: u128,
    /// Exact sum(probability_ppm * per-world action regret), denominator 1_000_000.
    pub expected_regret_numerator: u128,
    /// Conservatively rounded-up expected loss on the PPM loss scale.
    pub expected_loss_ppm_ceil: u32,
    /// Conservatively rounded-up expected regret on the PPM loss scale.
    pub expected_regret_ppm_ceil: u32,
    /// Maximum regret among actions assigned nonzero probability in this world.
    /// This does not describe how likely the maximum-regret action is.
    pub worst_supported_action_regret_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizedDecisionRegretCertificate {
    pub equivalence_class_id: u64,
    /// Opaque candidate policy artifact identity; does not hash the distribution rows.
    pub candidate_policy_artifact_sha256: [u8; 32],
    pub policy_probability_mass_ppm: u32,
    pub policy_support_action_count: u64,
    pub alternative_world_count: u64,
    /// Deterministic minimax-regret benchmark from the shared canonical oracle.
    /// This is not a randomized-policy lower bound or randomized optimum.
    pub deterministic_minimax_regret_comparator_ppm: u32,
    pub deterministic_minimax_action_id: String,
    /// Maximum expected regret over hidden worlds for the supplied mixture.
    pub worst_case_expected_regret_numerator: u128,
    pub worst_case_expected_regret_ppm_ceil: u32,
    /// Exact sum of per-world expected-regret numerators.
    pub total_expected_regret_numerator: u128,
    /// Equal-weight mean across supplied worlds, rounded upward.
    pub mean_expected_regret_ppm_ceil: u32,
    /// Maximum regret of any positive-probability action in any supplied world.
    /// Kept distinct from expected regret; this is a support, not a tail, bound.
    pub worst_supported_action_regret_ppm: u32,
    pub worlds: Vec<RandomizedWorldRegret>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RandomizedDecisionError {
    EmptyAlternatives,
    TooManyAlternatives,
    EmptyPolicyDigest,
    EmptyPolicyDistribution,
    TooManyPolicyActions,
    EmptyActionId,
    UnsortedOrDuplicatePolicyAction,
    ZeroProbability {
        action_id: String,
    },
    ProbabilityOutOfRange {
        action_id: String,
        probability_ppm: u32,
    },
    ProbabilityMassMismatch {
        actual_ppm: u64,
    },
    EmptyHiddenWorldDigest {
        world_id: u64,
    },
    PolicyActionUnknown {
        world_id: u64,
        action_id: String,
    },
    ArithmeticOverflow,
    DecisionObservability(DecisionObservabilityError),
}

impl fmt::Display for RandomizedDecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAlternatives => write!(
                formatter,
                "randomized decision class requires at least one hidden world"
            ),
            Self::TooManyAlternatives => write!(
                formatter,
                "randomized decision class exceeds the bounded hidden-world limit"
            ),
            Self::EmptyPolicyDigest => write!(formatter, "candidate policy digest must be nonzero"),
            Self::EmptyPolicyDistribution => {
                write!(formatter, "candidate action distribution must not be empty")
            }
            Self::TooManyPolicyActions => {
                write!(
                    formatter,
                    "candidate action distribution exceeds the action limit"
                )
            }
            Self::EmptyActionId => write!(formatter, "candidate action ID must not be empty"),
            Self::UnsortedOrDuplicatePolicyAction => write!(
                formatter,
                "candidate action probability rows must be strictly sorted by unique action ID"
            ),
            Self::ZeroProbability { action_id } => write!(
                formatter,
                "candidate action {action_id} has zero probability; omit zero-mass rows"
            ),
            Self::ProbabilityOutOfRange {
                action_id,
                probability_ppm,
            } => write!(
                formatter,
                "candidate action {action_id} has out-of-range probability {probability_ppm} PPM"
            ),
            Self::ProbabilityMassMismatch { actual_ppm } => write!(
                formatter,
                "candidate action probabilities sum to {actual_ppm} PPM, expected {RANDOMIZED_POLICY_MASS_PPM}"
            ),
            Self::EmptyHiddenWorldDigest { world_id } => {
                write!(
                    formatter,
                    "hidden world {world_id} has an empty identity digest"
                )
            }
            Self::PolicyActionUnknown {
                world_id,
                action_id,
            } => write!(
                formatter,
                "candidate policy assigns positive probability to inadmissible action {action_id} in world {world_id}"
            ),
            Self::ArithmeticOverflow => {
                write!(
                    formatter,
                    "randomized decision-regret arithmetic overflowed"
                )
            }
            Self::DecisionObservability(error) => write!(
                formatter,
                "shared decision-observability validation failed: {error}"
            ),
        }
    }
}

impl std::error::Error for RandomizedDecisionError {}

impl From<DecisionObservabilityError> for RandomizedDecisionError {
    fn from(value: DecisionObservabilityError) -> Self {
        Self::DecisionObservability(value)
    }
}

fn digest_is_zero(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn ceil_div(numerator: u128, denominator: u128) -> Result<u128, RandomizedDecisionError> {
    if denominator == 0 {
        return Err(RandomizedDecisionError::ArithmeticOverflow);
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    quotient
        .checked_add(if remainder != 0 { 1 } else { 0 })
        .ok_or(RandomizedDecisionError::ArithmeticOverflow)
}

fn to_u32(value: u128) -> Result<u32, RandomizedDecisionError> {
    u32::try_from(value).map_err(|_| RandomizedDecisionError::ArithmeticOverflow)
}

fn validate_policy_distribution(
    class: &RandomizedDecisionClass,
) -> Result<(), RandomizedDecisionError> {
    if digest_is_zero(&class.candidate_policy_artifact_sha256) {
        return Err(RandomizedDecisionError::EmptyPolicyDigest);
    }
    if class.action_probabilities.is_empty() {
        return Err(RandomizedDecisionError::EmptyPolicyDistribution);
    }
    if class.action_probabilities.len() > MAX_RANDOMIZED_ACTIONS {
        return Err(RandomizedDecisionError::TooManyPolicyActions);
    }

    let mut mass = 0_u64;
    let mut previous_action: Option<&str> = None;
    for probability in &class.action_probabilities {
        if probability.action_id.is_empty() {
            return Err(RandomizedDecisionError::EmptyActionId);
        }
        if previous_action.is_some_and(|previous| previous >= probability.action_id.as_str()) {
            return Err(RandomizedDecisionError::UnsortedOrDuplicatePolicyAction);
        }
        if probability.probability_ppm == 0 {
            return Err(RandomizedDecisionError::ZeroProbability {
                action_id: probability.action_id.clone(),
            });
        }
        if probability.probability_ppm > RANDOMIZED_POLICY_MASS_PPM {
            return Err(RandomizedDecisionError::ProbabilityOutOfRange {
                action_id: probability.action_id.clone(),
                probability_ppm: probability.probability_ppm,
            });
        }
        mass = mass
            .checked_add(u64::from(probability.probability_ppm))
            .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
        previous_action = Some(probability.action_id.as_str());
    }
    if mass != u64::from(RANDOMIZED_POLICY_MASS_PPM) {
        return Err(RandomizedDecisionError::ProbabilityMassMismatch { actual_ppm: mass });
    }
    Ok(())
}

/// Evaluate one frozen randomized action distribution over a deterministic
/// policy's observational-equivalence class.
///
/// The evaluator reuses evaluate_decision_observability for world/action-set
/// validation and the deterministic minimax comparator. It does not sample
/// actions or infer probabilities. For each world, expected regret is computed
/// exactly from the supplied integer action probabilities and evaluator loss
/// table. A candidate distribution that puts any positive mass on an
/// inadmissible action is rejected.
pub fn evaluate_randomized_decision_regret(
    class: &RandomizedDecisionClass,
) -> Result<RandomizedDecisionRegretCertificate, RandomizedDecisionError> {
    validate_policy_distribution(class)?;
    if class.alternatives.is_empty() {
        return Err(RandomizedDecisionError::EmptyAlternatives);
    }
    if class.alternatives.len() > MAX_RANDOMIZED_ALTERNATIVES {
        return Err(RandomizedDecisionError::TooManyAlternatives);
    }

    let first_policy_action = &class.action_probabilities[0].action_id;
    let mut seen_world_ids = BTreeSet::new();
    let mut base_worlds = Vec::with_capacity(class.alternatives.len());

    for world in &class.alternatives {
        if !seen_world_ids.insert(world.evaluator_world_id) {
            return Err(RandomizedDecisionError::DecisionObservability(
                DecisionObservabilityError::DuplicateWorldId {
                    world_id: world.evaluator_world_id,
                },
            ));
        }
        if digest_is_zero(&world.hidden_world_sha256) {
            return Err(RandomizedDecisionError::EmptyHiddenWorldDigest {
                world_id: world.evaluator_world_id,
            });
        }

        let admissible: BTreeSet<&str> = world
            .action_losses
            .iter()
            .map(|loss| loss.action_id.as_str())
            .collect();
        for probability in &class.action_probabilities {
            if !admissible.contains(probability.action_id.as_str()) {
                return Err(RandomizedDecisionError::PolicyActionUnknown {
                    world_id: world.evaluator_world_id,
                    action_id: probability.action_id.clone(),
                });
            }
        }

        base_worlds.push(CounterfactualWorld {
            evaluator_world_id: world.evaluator_world_id,
            hidden_world_sha256: world.hidden_world_sha256,
            candidate_view: world.candidate_view.clone(),
            // The base oracle is used only for validation and the deterministic
            // comparator. The first supported action is a valid placeholder.
            candidate_action_id: first_policy_action.clone(),
            action_losses: world.action_losses.clone(),
        });
    }

    let equivalence_class = ObservationalEquivalenceClass {
        equivalence_class_id: class.equivalence_class_id,
        alternatives: base_worlds,
    };
    let deterministic = evaluate_decision_observability(&equivalence_class)?;

    let mut world_results = Vec::with_capacity(class.alternatives.len());
    let mut worst_expected_regret_numerator = 0_u128;
    let mut total_expected_regret_numerator = 0_u128;
    let mut worst_supported_action_regret_ppm = 0_u32;

    for world in &class.alternatives {
        let losses: BTreeMap<&str, u32> = world
            .action_losses
            .iter()
            .map(|loss| (loss.action_id.as_str(), loss.loss_ppm))
            .collect();
        let best_action_loss_ppm = world
            .action_losses
            .iter()
            .map(|loss| loss.loss_ppm)
            .min()
            .ok_or(RandomizedDecisionError::DecisionObservability(
                DecisionObservabilityError::EmptyActionSet {
                    world_id: world.evaluator_world_id,
                },
            ))?;

        let mut expected_loss_numerator = 0_u128;
        let mut expected_regret_numerator = 0_u128;
        let mut world_worst_supported_regret_ppm = 0_u32;

        for probability in &class.action_probabilities {
            let loss_ppm = *losses.get(probability.action_id.as_str()).ok_or_else(|| {
                RandomizedDecisionError::PolicyActionUnknown {
                    world_id: world.evaluator_world_id,
                    action_id: probability.action_id.clone(),
                }
            })?;
            let regret_ppm = loss_ppm
                .checked_sub(best_action_loss_ppm)
                .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
            let weighted_loss = u128::from(probability.probability_ppm)
                .checked_mul(u128::from(loss_ppm))
                .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
            let weighted_regret = u128::from(probability.probability_ppm)
                .checked_mul(u128::from(regret_ppm))
                .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
            expected_loss_numerator = expected_loss_numerator
                .checked_add(weighted_loss)
                .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
            expected_regret_numerator = expected_regret_numerator
                .checked_add(weighted_regret)
                .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
            world_worst_supported_regret_ppm = world_worst_supported_regret_ppm.max(regret_ppm);
        }

        let expected_loss_ppm_ceil = to_u32(ceil_div(
            expected_loss_numerator,
            u128::from(RANDOMIZED_POLICY_MASS_PPM),
        )?)?;
        let expected_regret_ppm_ceil = to_u32(ceil_div(
            expected_regret_numerator,
            u128::from(RANDOMIZED_POLICY_MASS_PPM),
        )?)?;

        worst_expected_regret_numerator =
            worst_expected_regret_numerator.max(expected_regret_numerator);
        total_expected_regret_numerator = total_expected_regret_numerator
            .checked_add(expected_regret_numerator)
            .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
        worst_supported_action_regret_ppm =
            worst_supported_action_regret_ppm.max(world_worst_supported_regret_ppm);

        world_results.push(RandomizedWorldRegret {
            evaluator_world_id: world.evaluator_world_id,
            best_action_loss_ppm,
            expected_loss_numerator,
            expected_regret_numerator,
            expected_loss_ppm_ceil,
            expected_regret_ppm_ceil,
            worst_supported_action_regret_ppm: world_worst_supported_regret_ppm,
        });
    }

    let world_count = u128::try_from(class.alternatives.len())
        .map_err(|_| RandomizedDecisionError::ArithmeticOverflow)?;
    let mean_denominator = u128::from(RANDOMIZED_POLICY_MASS_PPM)
        .checked_mul(world_count)
        .ok_or(RandomizedDecisionError::ArithmeticOverflow)?;
    let worst_case_expected_regret_ppm_ceil = to_u32(ceil_div(
        worst_expected_regret_numerator,
        u128::from(RANDOMIZED_POLICY_MASS_PPM),
    )?)?;
    let mean_expected_regret_ppm_ceil =
        to_u32(ceil_div(total_expected_regret_numerator, mean_denominator)?)?;

    Ok(RandomizedDecisionRegretCertificate {
        equivalence_class_id: class.equivalence_class_id,
        candidate_policy_artifact_sha256: class.candidate_policy_artifact_sha256,
        policy_probability_mass_ppm: RANDOMIZED_POLICY_MASS_PPM,
        policy_support_action_count: u64::try_from(class.action_probabilities.len())
            .map_err(|_| RandomizedDecisionError::ArithmeticOverflow)?,
        alternative_world_count: u64::try_from(class.alternatives.len())
            .map_err(|_| RandomizedDecisionError::ArithmeticOverflow)?,
        deterministic_minimax_regret_comparator_ppm: deterministic
            .unavoidable_worst_case_regret_ppm,
        deterministic_minimax_action_id: deterministic.minimax_regret_action_id,
        worst_case_expected_regret_numerator: worst_expected_regret_numerator,
        worst_case_expected_regret_ppm_ceil,
        total_expected_regret_numerator,
        mean_expected_regret_ppm_ceil,
        worst_supported_action_regret_ppm,
        worlds: world_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn view() -> CandidateViewIdentity {
        CandidateViewIdentity {
            observation_history_sha256: digest(1),
            recurrent_state_sha256: digest(2),
            model_configuration_sha256: digest(3),
            policy_seed: 19,
        }
    }

    fn world(world_id: u64, left_loss: u32, right_loss: u32) -> RandomizedPolicyWorld {
        RandomizedPolicyWorld {
            evaluator_world_id: world_id,
            hidden_world_sha256: digest(if world_id == 1 { 11 } else { 12 }),
            candidate_view: view(),
            action_losses: vec![
                ActionLoss {
                    action_id: "left".to_owned(),
                    loss_ppm: left_loss,
                },
                ActionLoss {
                    action_id: "right".to_owned(),
                    loss_ppm: right_loss,
                },
            ],
        }
    }

    fn class(probabilities: Vec<CandidateActionProbability>) -> RandomizedDecisionClass {
        RandomizedDecisionClass {
            equivalence_class_id: 77,
            candidate_policy_artifact_sha256: digest(9),
            action_probabilities: probabilities,
            alternatives: vec![world(1, 0, 1_000_000), world(2, 1_000_000, 0)],
        }
    }

    fn mixture() -> Vec<CandidateActionProbability> {
        vec![
            CandidateActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: 500_000,
            },
            CandidateActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: 500_000,
            },
        ]
    }

    #[test]
    fn balanced_randomized_policy_halves_worst_case_expected_regret() {
        let result = evaluate_randomized_decision_regret(&class(mixture())).unwrap();
        assert_eq!(result.worst_case_expected_regret_ppm_ceil, 500_000);
        assert_eq!(result.mean_expected_regret_ppm_ceil, 500_000);
        assert_eq!(
            result.deterministic_minimax_regret_comparator_ppm,
            1_000_000
        );
        assert_eq!(result.deterministic_minimax_action_id, "left");
        assert_eq!(result.policy_probability_mass_ppm, 1_000_000);
        assert_eq!(result.policy_support_action_count, 2);
        assert_eq!(result.worlds.len(), 2);
    }

    #[test]
    fn expected_regret_does_not_hide_worst_supported_action_regret() {
        let result = evaluate_randomized_decision_regret(&class(mixture())).unwrap();
        assert_eq!(result.worst_case_expected_regret_ppm_ceil, 500_000);
        assert_eq!(result.worst_supported_action_regret_ppm, 1_000_000);
        assert!(
            result.worst_supported_action_regret_ppm > result.worst_case_expected_regret_ppm_ceil
        );
    }

    #[test]
    fn degenerate_distribution_matches_deterministic_candidate_regret() {
        let deterministic = class(vec![CandidateActionProbability {
            action_id: "left".to_owned(),
            probability_ppm: 1_000_000,
        }]);
        let result = evaluate_randomized_decision_regret(&deterministic).unwrap();
        assert_eq!(result.worst_case_expected_regret_ppm_ceil, 1_000_000);
        assert_eq!(result.worlds[0].expected_regret_ppm_ceil, 0);
        assert_eq!(result.worlds[1].expected_regret_ppm_ceil, 1_000_000);
    }

    #[test]
    fn policy_probability_mass_must_sum_exactly_to_one_million() {
        let invalid = class(vec![
            CandidateActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: 499_999,
            },
            CandidateActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: 500_000,
            },
        ]);
        assert_eq!(
            evaluate_randomized_decision_regret(&invalid),
            Err(RandomizedDecisionError::ProbabilityMassMismatch {
                actual_ppm: 999_999
            })
        );
    }

    #[test]
    fn policy_rows_must_be_canonical_and_zero_mass_rows_are_rejected() {
        let unsorted = class(vec![
            CandidateActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: 500_000,
            },
            CandidateActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: 500_000,
            },
        ]);
        assert_eq!(
            evaluate_randomized_decision_regret(&unsorted),
            Err(RandomizedDecisionError::UnsortedOrDuplicatePolicyAction)
        );

        let zero = class(vec![
            CandidateActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: 1_000_000,
            },
            CandidateActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: 0,
            },
        ]);
        assert_eq!(
            evaluate_randomized_decision_regret(&zero),
            Err(RandomizedDecisionError::ZeroProbability {
                action_id: "right".to_owned()
            })
        );
    }

    #[test]
    fn policy_cannot_assign_mass_to_an_inadmissible_action() {
        let invalid = class(vec![CandidateActionProbability {
            action_id: "teleport".to_owned(),
            probability_ppm: 1_000_000,
        }]);
        assert_eq!(
            evaluate_randomized_decision_regret(&invalid),
            Err(RandomizedDecisionError::PolicyActionUnknown {
                world_id: 1,
                action_id: "teleport".to_owned()
            })
        );
    }

    #[test]
    fn same_hidden_worlds_must_have_identical_candidate_views() {
        let mut invalid = class(mixture());
        invalid.alternatives[1]
            .candidate_view
            .recurrent_state_sha256 = digest(88);
        assert_eq!(
            evaluate_randomized_decision_regret(&invalid),
            Err(RandomizedDecisionError::DecisionObservability(
                DecisionObservabilityError::CandidateViewsDiffer
            ))
        );
    }

    #[test]
    fn duplicate_world_identity_is_rejected() {
        let mut invalid = class(mixture());
        invalid.alternatives[1].evaluator_world_id = 1;
        assert!(matches!(
            evaluate_randomized_decision_regret(&invalid),
            Err(RandomizedDecisionError::DecisionObservability(
                DecisionObservabilityError::DuplicateWorldId { world_id: 1 }
            ))
        ));
    }

    #[test]
    fn zero_policy_commitment_and_zero_hidden_world_digest_are_rejected() {
        let mut invalid_policy = class(mixture());
        invalid_policy.candidate_policy_artifact_sha256 = [0; 32];
        assert_eq!(
            evaluate_randomized_decision_regret(&invalid_policy),
            Err(RandomizedDecisionError::EmptyPolicyDigest)
        );

        let mut invalid_world = class(mixture());
        invalid_world.alternatives[0].hidden_world_sha256 = [0; 32];
        assert_eq!(
            evaluate_randomized_decision_regret(&invalid_world),
            Err(RandomizedDecisionError::EmptyHiddenWorldDigest { world_id: 1 })
        );
    }

    #[test]
    fn empty_classes_and_distributions_fail_closed() {
        let empty_distribution = class(Vec::new());
        assert_eq!(
            evaluate_randomized_decision_regret(&empty_distribution),
            Err(RandomizedDecisionError::EmptyPolicyDistribution)
        );

        let mut empty_alternatives = class(mixture());
        empty_alternatives.alternatives.clear();
        assert_eq!(
            evaluate_randomized_decision_regret(&empty_alternatives),
            Err(RandomizedDecisionError::EmptyAlternatives)
        );
    }

    #[test]
    fn probability_bounds_and_empty_action_ids_fail_closed() {
        let out_of_range = class(vec![CandidateActionProbability {
            action_id: "left".to_owned(),
            probability_ppm: 1_000_001,
        }]);
        assert_eq!(
            evaluate_randomized_decision_regret(&out_of_range),
            Err(RandomizedDecisionError::ProbabilityOutOfRange {
                action_id: "left".to_owned(),
                probability_ppm: 1_000_001,
            })
        );

        let empty_id = class(vec![CandidateActionProbability {
            action_id: String::new(),
            probability_ppm: 1_000_000,
        }]);
        assert_eq!(
            evaluate_randomized_decision_regret(&empty_id),
            Err(RandomizedDecisionError::EmptyActionId)
        );
    }

    #[test]
    fn action_and_world_limits_are_enforced_before_iteration() {
        let too_many_actions = class(
            (0..=MAX_RANDOMIZED_ACTIONS)
                .map(|index| CandidateActionProbability {
                    action_id: format!("action-{index:03}"),
                    probability_ppm: 1,
                })
                .collect(),
        );
        assert_eq!(
            evaluate_randomized_decision_regret(&too_many_actions),
            Err(RandomizedDecisionError::TooManyPolicyActions)
        );

        let mut too_many_worlds = class(mixture());
        too_many_worlds.alternatives = vec![world(1, 0, 1_000_000); MAX_RANDOMIZED_ALTERNATIVES + 1];
        assert_eq!(
            evaluate_randomized_decision_regret(&too_many_worlds),
            Err(RandomizedDecisionError::TooManyAlternatives)
        );
    }

    #[test]
    fn inconsistent_action_sets_and_out_of_range_losses_are_rejected() {
        let mut mismatched = class(mixture());
        mismatched.alternatives[1].action_losses.pop();
        assert!(matches!(
            evaluate_randomized_decision_regret(&mismatched),
            Err(RandomizedDecisionError::DecisionObservability(
                DecisionObservabilityError::ActionSetMismatch { world_id: 2 }
            ))
        ));

        let mut invalid_loss = class(mixture());
        invalid_loss.alternatives[0].action_losses[0].loss_ppm = 1_000_001;
        assert!(matches!(
            evaluate_randomized_decision_regret(&invalid_loss),
            Err(RandomizedDecisionError::DecisionObservability(
                DecisionObservabilityError::LossOutOfRange {
                    world_id: 1,
                    action_id,
                    ..
                }
            )) if action_id == "left"
        ));
    }

    #[test]
    fn asymmetric_mixture_is_scored_from_its_actual_probability_mass() {
        let asymmetric = class(vec![
            CandidateActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: 200_000,
            },
            CandidateActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: 800_000,
            },
        ]);
        let result = evaluate_randomized_decision_regret(&asymmetric).unwrap();
        assert_eq!(result.worlds[0].expected_regret_ppm_ceil, 800_000);
        assert_eq!(result.worlds[1].expected_regret_ppm_ceil, 200_000);
        assert_eq!(result.worst_case_expected_regret_ppm_ceil, 800_000);
        assert_eq!(result.mean_expected_regret_ppm_ceil, 500_000);
    }

    #[test]
    fn upper_rounding_is_conservative_for_fractional_expected_regret() {
        let mut fractional = class(vec![
            CandidateActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: 333_333,
            },
            CandidateActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: 666_667,
            },
        ]);
        fractional.alternatives = vec![world(1, 0, 3)];
        let result = evaluate_randomized_decision_regret(&fractional).unwrap();
        assert_eq!(result.worlds[0].expected_regret_numerator, 2_000_001);
        assert_eq!(result.worlds[0].expected_regret_ppm_ceil, 3);
        assert_eq!(result.worst_case_expected_regret_ppm_ceil, 3);
    }
}
