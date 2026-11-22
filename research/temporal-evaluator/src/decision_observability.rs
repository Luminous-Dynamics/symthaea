//! Decision-relevant observability bounds for evaluator-side counterfactual worlds.
//!
//! Two hidden worlds may produce the same candidate-visible observation history,
//! recurrent candidate state, model/configuration identity and policy RNG seed.
//! A deterministic causal candidate must then take the same action in both.
//!
//! This module computes a finite, task-relative lower bound on decision regret
//! over that observational equivalence class. It separates:
//! - irreducible worst-case regret caused by indistinguishable worlds;
//! - avoidable candidate regret above the best deterministic common action;
//! - state observability from decision identifiability (different concepts).
//!
//! It is evaluator-only. Do not link this module into candidate policy code.
//! The loss table and hidden-world identifiers must never be exposed to a
//! candidate. Randomized policies require a separate mixed-strategy evaluator;
//! v0.1 qualifies deterministic policies with a frozen RNG seed.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DECISION_OBSERVABILITY_PPM: u32 = 1_000_000;
const MAX_ALTERNATIVES: usize = 4_096;
const MAX_ACTIONS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateViewIdentity {
    /// SHA-256 of exact candidate-visible observations, including event time,
    /// availability time, values, masks, modality and canonical encoding.
    pub observation_history_sha256: [u8; 32],
    /// SHA-256 of the candidate's recurrent/internal state before this decision.
    pub recurrent_state_sha256: [u8; 32],
    /// SHA-256 of the exact candidate model and configuration.
    pub model_configuration_sha256: [u8; 32],
    /// Frozen candidate-owned randomness seed for deterministic paired replay.
    pub policy_seed: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionLoss {
    pub action_id: String,
    /// Evaluator-only normalized loss in parts per million [0, 1_000_000].
    pub loss_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterfactualWorld {
    /// Evaluator-only identity; never visible to the candidate.
    pub evaluator_world_id: u64,
    /// SHA-256 identity of this hidden world / latent trajectory.
    pub hidden_world_sha256: [u8; 32],
    /// Must be byte-identical across all worlds in one equivalence class.
    pub candidate_view: CandidateViewIdentity,
    /// Candidate's actual emitted action in this paired replay.
    pub candidate_action_id: String,
    /// Evaluator-only loss of each common admissible action in this world.
    pub action_losses: Vec<ActionLoss>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationalEquivalenceClass {
    pub equivalence_class_id: u64,
    pub alternatives: Vec<CounterfactualWorld>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionObservabilityCertificate {
    pub equivalence_class_id: u64,
    pub observation_history_sha256: [u8; 32],
    pub recurrent_state_sha256: [u8; 32],
    pub model_configuration_sha256: [u8; 32],
    pub policy_seed: u64,
    pub alternative_world_count: u64,
    pub admissible_action_count: u64,
    pub candidate_action_id: String,
    pub decision_identifiable: bool,
    /// Actions optimal in every indistinguishable world. Empty means no single
    /// deterministic action is optimal across the whole equivalence class.
    pub common_optimal_action_ids: Vec<String>,
    /// Best deterministic action minimizing the maximum raw loss.
    pub minimax_loss_action_id: String,
    pub minimax_worst_case_loss_ppm: u32,
    /// Best deterministic action minimizing maximum regret vs per-world oracle.
    pub minimax_regret_action_id: String,
    pub unavoidable_worst_case_regret_ppm: u32,
    pub unavoidable_mean_regret_ppm: u32,
    pub candidate_worst_case_loss_ppm: u32,
    pub candidate_mean_loss_ppm: u32,
    pub candidate_worst_case_regret_ppm: u32,
    pub candidate_mean_regret_ppm: u32,
    /// Candidate regret above the best deterministic worst-case-regret policy.
    pub avoidable_worst_case_regret_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionObservabilityError {
    EmptyEquivalenceClass,
    TooManyAlternatives,
    TooManyActions,
    CandidateViewsDiffer,
    CandidateActionsDiffer,
    DuplicateWorldId { world_id: u64 },
    EmptyActionSet { world_id: u64 },
    EmptyActionId { world_id: u64 },
    DuplicateActionId { world_id: u64, action_id: String },
    ActionSetMismatch { world_id: u64 },
    CandidateActionNotAdmissible { world_id: u64, action_id: String },
    LossOutOfRange { world_id: u64, action_id: String, loss_ppm: u32 },
    ArithmeticOverflow,
}

impl fmt::Display for DecisionObservabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEquivalenceClass => write!(formatter, "equivalence class requires at least one world"),
            Self::TooManyAlternatives => write!(formatter, "equivalence class exceeds the alternative-world limit"),
            Self::TooManyActions => write!(formatter, "equivalence class exceeds the admissible-action limit"),
            Self::CandidateViewsDiffer => write!(formatter, "counterfactual worlds do not have identical candidate-visible view/state/configuration/seed"),
            Self::CandidateActionsDiffer => write!(formatter, "same candidate view and frozen seed produced different actions across paired worlds"),
            Self::DuplicateWorldId { world_id } => write!(formatter, "duplicate evaluator world ID {world_id}"),
            Self::EmptyActionSet { world_id } => write!(formatter, "world {world_id} has no admissible actions"),
            Self::EmptyActionId { world_id } => write!(formatter, "world {world_id} has an empty action ID"),
            Self::DuplicateActionId { world_id, action_id } => write!(formatter, "world {world_id} repeats action ID {action_id}"),
            Self::ActionSetMismatch { world_id } => write!(formatter, "world {world_id} has a different admissible action set"),
            Self::CandidateActionNotAdmissible { world_id, action_id } => write!(formatter, "candidate action {action_id} is not admissible in world {world_id}"),
            Self::LossOutOfRange { world_id, action_id, loss_ppm } => write!(formatter, "world {world_id} action {action_id} has out-of-range loss {loss_ppm}"),
            Self::ArithmeticOverflow => write!(formatter, "decision-observability arithmetic overflow"),
        }
    }
}

impl std::error::Error for DecisionObservabilityError {}

#[derive(Clone, Debug)]
struct PreparedWorld {
    world_id: u64,
    candidate_loss: u32,
    losses: BTreeMap<String, u32>,
    regret_by_action: BTreeMap<String, u32>,
}

fn mean_floor(sum: u64, count: usize) -> Result<u32, DecisionObservabilityError> {
    let count = u64::try_from(count).map_err(|_| DecisionObservabilityError::ArithmeticOverflow)?;
    if count == 0 {
        return Err(DecisionObservabilityError::EmptyEquivalenceClass);
    }
    u32::try_from(sum / count).map_err(|_| DecisionObservabilityError::ArithmeticOverflow)
}

/// Compute a finite deterministic-policy decision lower bound for one class.
///
/// A valid class binds every alternative to the same observation-history digest,
/// candidate recurrent-state digest, model/config digest and policy seed. The
/// candidate action must also be the same across alternatives. A violation is
/// rejected instead of being interpreted as model error or hidden-world insight.
pub fn evaluate_decision_observability(
    class: &ObservationalEquivalenceClass,
) -> Result<DecisionObservabilityCertificate, DecisionObservabilityError> {
    if class.alternatives.is_empty() {
        return Err(DecisionObservabilityError::EmptyEquivalenceClass);
    }
    if class.alternatives.len() > MAX_ALTERNATIVES {
        return Err(DecisionObservabilityError::TooManyAlternatives);
    }

    let reference_view = &class.alternatives[0].candidate_view;
    let reference_action = &class.alternatives[0].candidate_action_id;
    let mut world_ids = BTreeSet::new();
    let mut action_universe: Option<BTreeSet<String>> = None;
    let mut prepared = Vec::with_capacity(class.alternatives.len());

    for world in &class.alternatives {
        if &world.candidate_view != reference_view {
            return Err(DecisionObservabilityError::CandidateViewsDiffer);
        }
        if world.candidate_action_id.as_str() != reference_action.as_str() {
            return Err(DecisionObservabilityError::CandidateActionsDiffer);
        }
        if !world_ids.insert(world.evaluator_world_id) {
            return Err(DecisionObservabilityError::DuplicateWorldId {
                world_id: world.evaluator_world_id,
            });
        }
        if world.action_losses.is_empty() {
            return Err(DecisionObservabilityError::EmptyActionSet {
                world_id: world.evaluator_world_id,
            });
        }
        if world.action_losses.len() > MAX_ACTIONS {
            return Err(DecisionObservabilityError::TooManyActions);
        }

        let mut losses = BTreeMap::new();
        for item in &world.action_losses {
            if item.action_id.trim().is_empty() {
                return Err(DecisionObservabilityError::EmptyActionId {
                    world_id: world.evaluator_world_id,
                });
            }
            if item.loss_ppm > DECISION_OBSERVABILITY_PPM {
                return Err(DecisionObservabilityError::LossOutOfRange {
                    world_id: world.evaluator_world_id,
                    action_id: item.action_id.clone(),
                    loss_ppm: item.loss_ppm,
                });
            }
            if losses.insert(item.action_id.clone(), item.loss_ppm).is_some() {
                return Err(DecisionObservabilityError::DuplicateActionId {
                    world_id: world.evaluator_world_id,
                    action_id: item.action_id.clone(),
                });
            }
        }
        let actions: BTreeSet<String> = losses.keys().cloned().collect();
        if let Some(reference_actions) = &action_universe {
            if &actions != reference_actions {
                return Err(DecisionObservabilityError::ActionSetMismatch {
                    world_id: world.evaluator_world_id,
                });
            }
        } else {
            action_universe = Some(actions);
        }
        if !losses.contains_key(reference_action) {
            return Err(DecisionObservabilityError::CandidateActionNotAdmissible {
                world_id: world.evaluator_world_id,
                action_id: reference_action.clone(),
            });
        }

        let best_loss = losses
            .values()
            .copied()
            .min()
            .ok_or(DecisionObservabilityError::EmptyActionSet {
                world_id: world.evaluator_world_id,
            })?;
        let regret_by_action = losses
            .iter()
            .map(|(action, loss)| (action.clone(), loss.saturating_sub(best_loss)))
            .collect();
        let candidate_loss = *losses.get(reference_action).ok_or(
            DecisionObservabilityError::CandidateActionNotAdmissible {
                world_id: world.evaluator_world_id,
                action_id: reference_action.clone(),
            },
        )?;
        prepared.push(PreparedWorld {
            world_id: world.evaluator_world_id,
            candidate_loss,
            losses,
            regret_by_action,
        });
    }

    let actions = action_universe.ok_or(DecisionObservabilityError::TooManyActions)?;
    let mut common_optimal = actions.clone();
    for world in &prepared {
        let best = world
            .losses
            .values()
            .copied()
            .min()
            .ok_or(DecisionObservabilityError::EmptyActionSet {
                world_id: world.world_id,
            })?;
        let optimal_here: BTreeSet<String> = world
            .losses
            .iter()
            .filter_map(|(action, loss)| (*loss == best).then_some(action.clone()))
            .collect();
        common_optimal = common_optimal.intersection(&optimal_here).cloned().collect();
    }

    let mut candidate_worst_loss = 0_u32;
    let mut candidate_worst_regret = 0_u32;
    let mut candidate_loss_sum = 0_u64;
    let mut candidate_regret_sum = 0_u64;
    for world in &prepared {
        candidate_worst_loss = candidate_worst_loss.max(world.candidate_loss);
        candidate_loss_sum = candidate_loss_sum
            .checked_add(u64::from(world.candidate_loss))
            .ok_or(DecisionObservabilityError::ArithmeticOverflow)?;
        let regret = *world
            .regret_by_action
            .get(reference_action)
            .ok_or(DecisionObservabilityError::CandidateActionNotAdmissible {
                world_id: world.world_id,
                action_id: reference_action.clone(),
            })?;
        candidate_worst_regret = candidate_worst_regret.max(regret);
        candidate_regret_sum = candidate_regret_sum
            .checked_add(u64::from(regret))
            .ok_or(DecisionObservabilityError::ArithmeticOverflow)?;
    }

    let mut minimax_loss_action: Option<(u32, String)> = None;
    let mut minimax_regret_action: Option<(u32, String)> = None;
    for action in &actions {
        let mut worst_loss = 0_u32;
        let mut worst_regret = 0_u32;
        for world in &prepared {
            let loss = *world.losses.get(action).ok_or(
                DecisionObservabilityError::ActionSetMismatch {
                    world_id: world.world_id,
                },
            )?;
            let regret = *world.regret_by_action.get(action).ok_or(
                DecisionObservabilityError::ActionSetMismatch {
                    world_id: world.world_id,
                },
            )?;
            worst_loss = worst_loss.max(loss);
            worst_regret = worst_regret.max(regret);
        }

        let loss_candidate = (worst_loss, action.clone());
        if minimax_loss_action
            .as_ref()
            .is_none_or(|current| loss_candidate.cmp(current).is_lt())
        {
            minimax_loss_action = Some(loss_candidate);
        }
        let regret_candidate = (worst_regret, action.clone());
        if minimax_regret_action
            .as_ref()
            .is_none_or(|current| regret_candidate.cmp(current).is_lt())
        {
            minimax_regret_action = Some(regret_candidate);
        }
    }

    let (minimax_worst_case_loss_ppm, minimax_loss_action_id) =
        minimax_loss_action.ok_or(DecisionObservabilityError::TooManyActions)?;
    let (unavoidable_worst_case_regret_ppm, minimax_regret_action_id) =
        minimax_regret_action.ok_or(DecisionObservabilityError::TooManyActions)?;
    let minimax_regret_sum = prepared.iter().try_fold(0_u64, |sum, world| {
        let regret = *world.regret_by_action.get(&minimax_regret_action_id).ok_or(
            DecisionObservabilityError::ActionSetMismatch {
                world_id: world.world_id,
            },
        )?;
        sum.checked_add(u64::from(regret))
            .ok_or(DecisionObservabilityError::ArithmeticOverflow)
    })?;

    Ok(DecisionObservabilityCertificate {
        equivalence_class_id: class.equivalence_class_id,
        observation_history_sha256: reference_view.observation_history_sha256,
        recurrent_state_sha256: reference_view.recurrent_state_sha256,
        model_configuration_sha256: reference_view.model_configuration_sha256,
        policy_seed: reference_view.policy_seed,
        alternative_world_count: u64::try_from(prepared.len())
            .map_err(|_| DecisionObservabilityError::ArithmeticOverflow)?,
        admissible_action_count: u64::try_from(actions.len())
            .map_err(|_| DecisionObservabilityError::ArithmeticOverflow)?,
        candidate_action_id: reference_action.clone(),
        decision_identifiable: !common_optimal.is_empty(),
        common_optimal_action_ids: common_optimal.into_iter().collect(),
        minimax_loss_action_id,
        minimax_worst_case_loss_ppm,
        minimax_regret_action_id,
        unavoidable_worst_case_regret_ppm,
        unavoidable_mean_regret_ppm: mean_floor(minimax_regret_sum, prepared.len())?,
        candidate_worst_case_loss_ppm: candidate_worst_loss,
        candidate_mean_loss_ppm: mean_floor(candidate_loss_sum, prepared.len())?,
        candidate_worst_case_regret_ppm: candidate_worst_regret,
        candidate_mean_regret_ppm: mean_floor(candidate_regret_sum, prepared.len())?,
        avoidable_worst_case_regret_ppm: candidate_worst_regret
            .saturating_sub(unavoidable_worst_case_regret_ppm),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn world(
        world_id: u64,
        view: CandidateViewIdentity,
        candidate_action: &str,
        losses: &[(&str, u32)],
    ) -> CounterfactualWorld {
        CounterfactualWorld {
            evaluator_world_id: world_id,
            hidden_world_sha256: digest(world_id as u8),
            candidate_view: view,
            candidate_action_id: candidate_action.to_owned(),
            action_losses: losses
                .iter()
                .map(|(action, loss)| ActionLoss {
                    action_id: (*action).to_owned(),
                    loss_ppm: *loss,
                })
                .collect(),
        }
    }

    fn view() -> CandidateViewIdentity {
        CandidateViewIdentity {
            observation_history_sha256: digest(1),
            recurrent_state_sha256: digest(2),
            model_configuration_sha256: digest(3),
            policy_seed: 17,
        }
    }

    #[test]
    fn shared_optimal_action_means_decision_identifiable() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 1,
            alternatives: vec![
                world(1, view(), "continue", &[("continue", 0), ("abstain", 100_000)]),
                world(2, view(), "continue", &[("continue", 0), ("abstain", 200_000)]),
            ],
        };
        let result = evaluate_decision_observability(&class).unwrap();
        assert!(result.decision_identifiable);
        assert_eq!(result.common_optimal_action_ids, vec!["continue"]);
        assert_eq!(result.unavoidable_worst_case_regret_ppm, 0);
        assert_eq!(result.avoidable_worst_case_regret_ppm, 0);
    }

    #[test]
    fn indistinguishable_worlds_can_have_no_common_optimal_action() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 2,
            alternatives: vec![
                world(1, view(), "north", &[("north", 0), ("south", 1_000_000)]),
                world(2, view(), "north", &[("north", 1_000_000), ("south", 0)]),
            ],
        };
        let result = evaluate_decision_observability(&class).unwrap();
        assert!(!result.decision_identifiable);
        assert!(result.common_optimal_action_ids.is_empty());
        assert_eq!(result.unavoidable_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.candidate_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.avoidable_worst_case_regret_ppm, 0);
    }

    #[test]
    fn distinguishes_irreducible_ambiguity_from_avoidable_candidate_error() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 3,
            alternatives: vec![
                world(1, view(), "north", &[("north", 0), ("south", 1_000_000), ("safe", 400_000)]),
                world(2, view(), "north", &[("north", 1_000_000), ("south", 0), ("safe", 400_000)]),
            ],
        };
        let result = evaluate_decision_observability(&class).unwrap();
        assert!(!result.decision_identifiable);
        assert_eq!(result.minimax_regret_action_id, "safe");
        assert_eq!(result.unavoidable_worst_case_regret_ppm, 400_000);
        assert_eq!(result.candidate_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.avoidable_worst_case_regret_ppm, 600_000);
    }

    #[test]
    fn robust_loss_and_robust_regret_are_distinct_objectives() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 4,
            alternatives: vec![
                world(1, view(), "fast", &[("fast", 0), ("safe", 300_000)]),
                world(2, view(), "fast", &[("fast", 900_000), ("safe", 400_000)]),
            ],
        };
        let result = evaluate_decision_observability(&class).unwrap();
        assert_eq!(result.minimax_loss_action_id, "safe");
        assert_eq!(result.minimax_regret_action_id, "safe");
        assert_eq!(result.minimax_worst_case_loss_ppm, 400_000);
        assert_eq!(result.unavoidable_worst_case_regret_ppm, 300_000);
    }

    #[test]
    fn candidate_views_must_match_including_recurrent_state_and_seed() {
        let mut other = view();
        other.recurrent_state_sha256 = digest(99);
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 5,
            alternatives: vec![
                world(1, view(), "continue", &[("continue", 0)]),
                world(2, other, "continue", &[("continue", 0)]),
            ],
        };
        assert_eq!(
            evaluate_decision_observability(&class),
            Err(DecisionObservabilityError::CandidateViewsDiffer)
        );
    }

    #[test]
    fn candidate_actions_must_match_under_identical_view_and_seed() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 6,
            alternatives: vec![
                world(1, view(), "north", &[("north", 0), ("south", 1_000_000)]),
                world(2, view(), "south", &[("north", 1_000_000), ("south", 0)]),
            ],
        };
        assert_eq!(
            evaluate_decision_observability(&class),
            Err(DecisionObservabilityError::CandidateActionsDiffer)
        );
    }

    #[test]
    fn action_sets_must_be_identical_across_worlds() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 7,
            alternatives: vec![
                world(1, view(), "north", &[("north", 0), ("south", 1_000_000)]),
                world(2, view(), "north", &[("north", 1_000_000), ("west", 0)]),
            ],
        };
        assert!(matches!(
            evaluate_decision_observability(&class),
            Err(DecisionObservabilityError::ActionSetMismatch { world_id: 2 })
        ));
    }

    #[test]
    fn rejects_duplicate_world_ids_and_action_ids() {
        let duplicate_worlds = ObservationalEquivalenceClass {
            equivalence_class_id: 8,
            alternatives: vec![
                world(1, view(), "continue", &[("continue", 0)]),
                world(1, view(), "continue", &[("continue", 0)]),
            ],
        };
        assert!(matches!(
            evaluate_decision_observability(&duplicate_worlds),
            Err(DecisionObservabilityError::DuplicateWorldId { world_id: 1 })
        ));

        let duplicate_actions = ObservationalEquivalenceClass {
            equivalence_class_id: 9,
            alternatives: vec![world(
                1,
                view(),
                "continue",
                &[("continue", 0), ("continue", 100)],
            )],
        };
        assert!(matches!(
            evaluate_decision_observability(&duplicate_actions),
            Err(DecisionObservabilityError::DuplicateActionId { .. })
        ));
    }

    #[test]
    fn rejects_empty_and_out_of_range_loss_records() {
        let empty = ObservationalEquivalenceClass {
            equivalence_class_id: 10,
            alternatives: Vec::new(),
        };
        assert_eq!(
            evaluate_decision_observability(&empty),
            Err(DecisionObservabilityError::EmptyEquivalenceClass)
        );

        let bad = ObservationalEquivalenceClass {
            equivalence_class_id: 11,
            alternatives: vec![world(1, view(), "continue", &[("continue", 1_000_001)])],
        };
        assert!(matches!(
            evaluate_decision_observability(&bad),
            Err(DecisionObservabilityError::LossOutOfRange { .. })
        ));
    }

    #[test]
    fn single_world_is_decision_identifiable() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 12,
            alternatives: vec![world(1, view(), "stop", &[("stop", 0), ("continue", 500_000)])],
        };
        let result = evaluate_decision_observability(&class).unwrap();
        assert!(result.decision_identifiable);
        assert_eq!(result.unavoidable_worst_case_regret_ppm, 0);
    }
}
