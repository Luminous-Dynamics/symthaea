//! Decision-preserving qualification for compressed recurrent-state representations.
//!
//! This evaluator checks whether a candidate-provided compressed-state digest
//! merges full recurrent-state information sets that require materially
//! different decisions. It reuses decision_observability as the sole regret
//! oracle; it does not create another action-loss or safety authority.
//!
//! The evaluator receives full-state and compressed-state identities plus
//! evaluator-owned hidden-world/action-loss records. Candidate code must never
//! receive those records or this module. Package separation is not an OS
//! isolation boundary; #9437 remains required.
//!
//! A digest is an identity commitment, not proof that the candidate actually
//! used the compressed representation or achieved a memory/energy reduction.
//! Those claims require source-bound execution and measured resource receipts.

use crate::decision_observability::{
    CandidateViewIdentity, CounterfactualWorld, DecisionObservabilityCertificate,
    DecisionObservabilityError, ObservationalEquivalenceClass,
    evaluate_decision_observability,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_REPRESENTATION_ALTERNATIVES: usize = 4_096;
const MAX_REGRET_PPM: u32 = 1_000_000;

/// One evaluator-side counterfactual alternative.
///
/// The full-state identity is taken from world.candidate_view. The compressed
/// identity is the digest of the representation that the candidate is
/// supposed to use at decision time. The other candidate-view fields must be
/// unchanged: observations, model/configuration and policy seed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationAlternative {
    pub world: CounterfactualWorld,
    pub compressed_recurrent_state_sha256: [u8; 32],
}

/// Frozen tolerances supplied by the task contract before evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationCompressionConfig {
    /// Maximum worst-case regret for an information set to be decision-sufficient.
    pub epsilon_regret_tolerance_ppm: u32,
    /// Maximum additional minimax-regret floor introduced by compression.
    pub maximum_added_regret_ppm: u32,
}

/// Result for one compressed information set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompressedRepresentationClassCertificate {
    pub compressed_recurrent_state_sha256: [u8; 32],
    pub hidden_world_count: u64,
    pub source_full_state_class_count: u64,
    /// Worst information-set regret floor before compression, across the
    /// distinct full-state classes merged into this compressed class.
    pub baseline_worst_case_regret_ppm: u32,
    /// Minimax-regret floor after all merged worlds become indistinguishable.
    pub compressed_worst_case_regret_ppm: u32,
    /// Non-negative increase caused by merging the source information sets.
    pub added_worst_case_regret_ppm: u32,
    pub all_source_classes_epsilon_sufficient: bool,
    pub compressed_information_set_epsilon_sufficient: bool,
    /// This qualifies the emitted action, not merely the information set.
    pub candidate_action_within_epsilon: bool,
    pub exact_decision_identifiable: bool,
    pub common_optimal_action_ids: Vec<String>,
    pub minimax_regret_action_id: String,
    pub candidate_worst_case_regret_ppm: u32,
    /// True only when the source states were already epsilon-sufficient,
    /// compression remains epsilon-sufficient, and the added-regret budget
    /// is respected. This is not a safety or deployment certificate.
    pub representation_decision_preserved: bool,
}

/// Aggregate evaluator certificate for one representation/compression profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationCompressionCertificate {
    pub alternative_count: u64,
    pub full_state_class_count: u64,
    pub compressed_state_class_count: u64,
    pub epsilon_regret_tolerance_ppm: u32,
    pub maximum_added_regret_ppm: u32,
    pub worst_added_regret_ppm: u32,
    pub all_compressed_classes_decision_preserving: bool,
    pub all_candidate_actions_within_epsilon: bool,
    pub classes: Vec<CompressedRepresentationClassCertificate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RepresentationCompressionError {
    EmptyAlternatives,
    TooManyAlternatives,
    EmptyCompressedStateDigest { world_id: u64 },
    DuplicateWorldId { world_id: u64 },
    InconsistentCompressionMapping {
        full_recurrent_state_sha256: [u8; 32],
        first_compressed_state_sha256: [u8; 32],
        conflicting_compressed_state_sha256: [u8; 32],
    },
    RegretToleranceOutOfRange { tolerance_ppm: u32 },
    AddedRegretBudgetOutOfRange { budget_ppm: u32 },
    CompressedRegretFloorBelowSourceFloor {
        compressed_floor_ppm: u32,
        source_floor_ppm: u32,
    },
    ArithmeticOverflow,
    DecisionObservability(DecisionObservabilityError),
}

impl fmt::Display for RepresentationCompressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAlternatives => write!(formatter, "representation qualification requires at least one alternative"),
            Self::TooManyAlternatives => write!(formatter, "representation alternatives exceed the bounded evaluator limit"),
            Self::EmptyCompressedStateDigest { world_id } => write!(formatter, "world {world_id} has an empty compressed-state digest"),
            Self::DuplicateWorldId { world_id } => write!(formatter, "duplicate evaluator world ID {world_id}"),
            Self::InconsistentCompressionMapping { .. } => write!(formatter, "one full recurrent state maps to multiple compressed-state digests"),
            Self::RegretToleranceOutOfRange { tolerance_ppm } => write!(formatter, "epsilon-regret tolerance {tolerance_ppm} exceeds the one-million PPM scale"),
            Self::AddedRegretBudgetOutOfRange { budget_ppm } => write!(formatter, "added-regret budget {budget_ppm} exceeds the one-million PPM scale"),
            Self::CompressedRegretFloorBelowSourceFloor { compressed_floor_ppm, source_floor_ppm } => write!(formatter, "compressed regret floor {compressed_floor_ppm} is below source floor {source_floor_ppm}; the compression partition is not a coarsening or the evaluator contract is inconsistent"),
            Self::ArithmeticOverflow => write!(formatter, "representation-compression arithmetic overflowed"),
            Self::DecisionObservability(error) => write!(formatter, "base decision-observability qualification failed: {error}"),
        }
    }
}

impl std::error::Error for RepresentationCompressionError {}

impl From<DecisionObservabilityError> for RepresentationCompressionError {
    fn from(value: DecisionObservabilityError) -> Self {
        Self::DecisionObservability(value)
    }
}

/// Canonical grouping key. The evaluator groups by the exact candidate-visible
/// identity, not by a candidate-supplied class number or row order.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct CandidateViewKey {
    observation_history_sha256: [u8; 32],
    recurrent_state_sha256: [u8; 32],
    model_configuration_sha256: [u8; 32],
    policy_seed: u64,
}

impl From<&CandidateViewIdentity> for CandidateViewKey {
    fn from(view: &CandidateViewIdentity) -> Self {
        Self {
            observation_history_sha256: view.observation_history_sha256,
            recurrent_state_sha256: view.recurrent_state_sha256,
            model_configuration_sha256: view.model_configuration_sha256,
            policy_seed: view.policy_seed,
        }
    }
}

fn is_zero_digest(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

/// Qualify whether replacing full recurrent-state identity with a compressed
/// state preserves epsilon-qualified decisions.
///
/// The evaluator constructs two partitions:
/// - the baseline partition, keyed by full recurrent-state identity;
/// - the compressed partition, keyed by the compressed-state digest.
///
/// Every compressed class must be a union of complete baseline classes. A
/// deterministic compression function cannot split one identical full state
/// across multiple compressed identities. The base evaluator validates the
/// candidate action is identical within each resulting information set.
///
/// For each compressed class, the function reports:
/// - the maximum baseline regret floor across its source full-state classes;
/// - the regret floor after those classes are merged;
/// - the extra irreducible regret caused by compression;
/// - epsilon-sufficiency and candidate-action regret as separate properties.
///
/// The monotonicity check enforces the information-ordering invariant:
/// merging information sets cannot lower their minimax-regret floor.
pub fn qualify_representation_compression(
    alternatives: &[RepresentationAlternative],
    config: RepresentationCompressionConfig,
) -> Result<RepresentationCompressionCertificate, RepresentationCompressionError> {
    if alternatives.is_empty() {
        return Err(RepresentationCompressionError::EmptyAlternatives);
    }
    if alternatives.len() > MAX_REPRESENTATION_ALTERNATIVES {
        return Err(RepresentationCompressionError::TooManyAlternatives);
    }
    if config.epsilon_regret_tolerance_ppm > MAX_REGRET_PPM {
        return Err(RepresentationCompressionError::RegretToleranceOutOfRange {
            tolerance_ppm: config.epsilon_regret_tolerance_ppm,
        });
    }
    if config.maximum_added_regret_ppm > MAX_REGRET_PPM {
        return Err(RepresentationCompressionError::AddedRegretBudgetOutOfRange {
            budget_ppm: config.maximum_added_regret_ppm,
        });
    }

    let mut world_ids = BTreeSet::new();
    let mut full_groups: BTreeMap<CandidateViewKey, Vec<CounterfactualWorld>> =
        BTreeMap::new();
    let mut compressed_groups: BTreeMap<CandidateViewKey, Vec<CounterfactualWorld>> =
        BTreeMap::new();
    let mut compressed_to_full: BTreeMap<CandidateViewKey, BTreeSet<CandidateViewKey>> =
        BTreeMap::new();
    let mut compressed_by_full: BTreeMap<CandidateViewKey, [u8; 32]> =
        BTreeMap::new();

    for alternative in alternatives {
        let world_id = alternative.world.evaluator_world_id;
        if !world_ids.insert(world_id) {
            return Err(RepresentationCompressionError::DuplicateWorldId { world_id });
        }
        if is_zero_digest(&alternative.compressed_recurrent_state_sha256) {
            return Err(RepresentationCompressionError::EmptyCompressedStateDigest {
                world_id,
            });
        }

        let full_key = CandidateViewKey::from(&alternative.world.candidate_view);
        let mut compressed_world = alternative.world.clone();
        compressed_world.candidate_view.recurrent_state_sha256 =
            alternative.compressed_recurrent_state_sha256;
        let compressed_key =
            CandidateViewKey::from(&compressed_world.candidate_view);

        if let Some(first_digest) = compressed_by_full.get(&full_key) {
            if first_digest != &alternative.compressed_recurrent_state_sha256 {
                return Err(
                    RepresentationCompressionError::InconsistentCompressionMapping {
                        full_recurrent_state_sha256: full_key.recurrent_state_sha256,
                        first_compressed_state_sha256: *first_digest,
                        conflicting_compressed_state_sha256: alternative
                            .compressed_recurrent_state_sha256,
                    },
                );
            }
        } else {
            compressed_by_full.insert(
                full_key.clone(),
                alternative.compressed_recurrent_state_sha256,
            );
        }

        full_groups
            .entry(full_key.clone())
            .or_default()
            .push(alternative.world.clone());
        compressed_groups
            .entry(compressed_key.clone())
            .or_default()
            .push(compressed_world);
        compressed_to_full
            .entry(compressed_key)
            .or_default()
            .insert(full_key);
    }

    let mut full_certificates: BTreeMap<CandidateViewKey, DecisionObservabilityCertificate> =
        BTreeMap::new();
    for (index, (key, worlds)) in full_groups.iter().enumerate() {
        let class_id =
            u64::try_from(index).map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?;
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: class_id,
            alternatives: worlds.clone(),
        };
        full_certificates.insert(key.clone(), evaluate_decision_observability(&class)?);
    }

    let mut certificates = Vec::with_capacity(compressed_groups.len());
    let mut worst_added_regret_ppm = 0_u32;
    let mut all_compressed_classes_decision_preserving = true;
    let mut all_candidate_actions_within_epsilon = true;

    for (index, (compressed_key, worlds)) in compressed_groups.iter().enumerate() {
        let class_id =
            u64::try_from(index).map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?;
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: class_id,
            alternatives: worlds.clone(),
        };
        let compressed = evaluate_decision_observability(&class)?;
        let source_keys = compressed_to_full
            .get(compressed_key)
            .ok_or(RepresentationCompressionError::ArithmeticOverflow)?;

        let mut baseline_worst_case_regret_ppm = 0_u32;
        let mut all_source_classes_epsilon_sufficient = true;
        for source_key in source_keys {
            let source = full_certificates
                .get(source_key)
                .ok_or(RepresentationCompressionError::ArithmeticOverflow)?;
            baseline_worst_case_regret_ppm =
                baseline_worst_case_regret_ppm.max(source.unavoidable_worst_case_regret_ppm);
            all_source_classes_epsilon_sufficient &=
                source.unavoidable_worst_case_regret_ppm
                    <= config.epsilon_regret_tolerance_ppm;
        }

        let added_worst_case_regret_ppm = compressed
            .unavoidable_worst_case_regret_ppm
            .checked_sub(baseline_worst_case_regret_ppm)
            .ok_or(
                RepresentationCompressionError::CompressedRegretFloorBelowSourceFloor {
                    compressed_floor_ppm: compressed.unavoidable_worst_case_regret_ppm,
                    source_floor_ppm: baseline_worst_case_regret_ppm,
                },
            )?;
        let compressed_information_set_epsilon_sufficient =
            compressed.unavoidable_worst_case_regret_ppm
                <= config.epsilon_regret_tolerance_ppm;
        let candidate_action_within_epsilon =
            compressed.candidate_worst_case_regret_ppm
                <= config.epsilon_regret_tolerance_ppm;
        let representation_decision_preserved =
            all_source_classes_epsilon_sufficient
                && compressed_information_set_epsilon_sufficient
                && added_worst_case_regret_ppm <= config.maximum_added_regret_ppm;

        worst_added_regret_ppm =
            worst_added_regret_ppm.max(added_worst_case_regret_ppm);
        all_compressed_classes_decision_preserving &=
            representation_decision_preserved;
        all_candidate_actions_within_epsilon &= candidate_action_within_epsilon;

        certificates.push(CompressedRepresentationClassCertificate {
            compressed_recurrent_state_sha256: compressed_key.recurrent_state_sha256,
            hidden_world_count: u64::try_from(worlds.len())
                .map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?,
            source_full_state_class_count: u64::try_from(source_keys.len())
                .map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?,
            baseline_worst_case_regret_ppm,
            compressed_worst_case_regret_ppm:
                compressed.unavoidable_worst_case_regret_ppm,
            added_worst_case_regret_ppm,
            all_source_classes_epsilon_sufficient,
            compressed_information_set_epsilon_sufficient,
            candidate_action_within_epsilon,
            exact_decision_identifiable: compressed.decision_identifiable,
            common_optimal_action_ids: compressed.common_optimal_action_ids,
            minimax_regret_action_id: compressed.minimax_regret_action_id,
            candidate_worst_case_regret_ppm: compressed.candidate_worst_case_regret_ppm,
            representation_decision_preserved,
        });
    }

    Ok(RepresentationCompressionCertificate {
        alternative_count: u64::try_from(alternatives.len())
            .map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?,
        full_state_class_count: u64::try_from(full_groups.len())
            .map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?,
        compressed_state_class_count: u64::try_from(compressed_groups.len())
            .map_err(|_| RepresentationCompressionError::ArithmeticOverflow)?,
        epsilon_regret_tolerance_ppm: config.epsilon_regret_tolerance_ppm,
        maximum_added_regret_ppm: config.maximum_added_regret_ppm,
        worst_added_regret_ppm,
        all_compressed_classes_decision_preserving,
        all_candidate_actions_within_epsilon,
        classes: certificates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_observability::ActionLoss;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn alternative(
        world_id: u64,
        full_state: u8,
        compressed_state: u8,
        candidate_action: &str,
        left_loss: u32,
        right_loss: u32,
    ) -> RepresentationAlternative {
        RepresentationAlternative {
            world: CounterfactualWorld {
                evaluator_world_id: world_id,
                hidden_world_sha256: digest(world_id as u8 + 1),
                candidate_view: CandidateViewIdentity {
                    observation_history_sha256: digest(1),
                    recurrent_state_sha256: digest(full_state),
                    model_configuration_sha256: digest(2),
                    policy_seed: 7,
                },
                candidate_action_id: candidate_action.to_owned(),
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
            },
            compressed_recurrent_state_sha256: digest(compressed_state),
        }
    }

    fn config(epsilon: u32, added: u32) -> RepresentationCompressionConfig {
        RepresentationCompressionConfig {
            epsilon_regret_tolerance_ppm: epsilon,
            maximum_added_regret_ppm: added,
        }
    }

    #[test]
    fn opposing_decisions_merged_by_compression_are_rejected() {
        let rows = vec![
            alternative(1, 11, 99, "left", 0, 1_000_000),
            alternative(2, 12, 99, "left", 1_000_000, 0),
        ];
        let result = qualify_representation_compression(&rows, config(100_000, 100_000))
            .expect("valid counterfactual fixture");
        let class = &result.classes[0];

        assert_eq!(result.full_state_class_count, 2);
        assert_eq!(result.compressed_state_class_count, 1);
        assert_eq!(class.baseline_worst_case_regret_ppm, 0);
        assert_eq!(class.compressed_worst_case_regret_ppm, 1_000_000);
        assert_eq!(class.added_worst_case_regret_ppm, 1_000_000);
        assert!(!class.compressed_information_set_epsilon_sufficient);
        assert!(!class.candidate_action_within_epsilon);
        assert!(!class.representation_decision_preserved);
        assert!(!result.all_compressed_classes_decision_preserving);
    }

    #[test]
    fn epsilon_sufficient_compression_can_preserve_a_near_optimal_action() {
        let rows = vec![
            alternative(1, 11, 99, "left", 0, 50_000),
            alternative(2, 12, 99, "left", 50_000, 0),
        ];
        let result = qualify_representation_compression(&rows, config(100_000, 50_000))
            .expect("valid epsilon-sufficient fixture");
        let class = &result.classes[0];

        assert!(!class.exact_decision_identifiable);
        assert!(class.common_optimal_action_ids.is_empty());
        assert_eq!(class.compressed_worst_case_regret_ppm, 50_000);
        assert_eq!(class.added_worst_case_regret_ppm, 50_000);
        assert!(class.compressed_information_set_epsilon_sufficient);
        assert!(class.candidate_action_within_epsilon);
        assert!(class.representation_decision_preserved);
    }

    #[test]
    fn identity_representation_preserves_the_baseline_partition() {
        let rows = vec![
            alternative(1, 11, 11, "left", 0, 1_000_000),
            alternative(2, 12, 12, "right", 1_000_000, 0),
        ];
        let result = qualify_representation_compression(&rows, config(0, 0))
            .expect("identity mapping");
        assert_eq!(result.full_state_class_count, 2);
        assert_eq!(result.compressed_state_class_count, 2);
        assert_eq!(result.worst_added_regret_ppm, 0);
        assert!(result.all_compressed_classes_decision_preserving);
        assert!(result.all_candidate_actions_within_epsilon);
    }

    #[test]
    fn one_full_state_cannot_map_to_two_compressed_digests() {
        let rows = vec![
            alternative(1, 11, 91, "left", 0, 100),
            alternative(2, 11, 92, "left", 0, 100),
        ];
        let error = qualify_representation_compression(&rows, config(100, 100))
            .expect_err("one full state must have one compressed identity");
        assert!(matches!(
            error,
            RepresentationCompressionError::InconsistentCompressionMapping { .. }
        ));
    }

    #[test]
    fn same_compressed_view_must_emit_the_same_action() {
        let rows = vec![
            alternative(1, 11, 99, "left", 0, 100),
            alternative(2, 12, 99, "right", 100, 0),
        ];
        let error = qualify_representation_compression(&rows, config(100, 100))
            .expect_err("same compressed view cannot produce two deterministic actions");
        assert!(matches!(
            error,
            RepresentationCompressionError::DecisionObservability(
                DecisionObservabilityError::CandidateActionsDiffer
            )
        ));
    }

    #[test]
    fn duplicate_world_identity_is_rejected_across_partitions() {
        let rows = vec![
            alternative(7, 11, 91, "left", 0, 100),
            alternative(7, 12, 92, "right", 100, 0),
        ];
        let error = qualify_representation_compression(&rows, config(100, 100))
            .expect_err("world IDs must be unique globally");
        assert!(matches!(
            error,
            RepresentationCompressionError::DuplicateWorldId { world_id: 7 }
        ));
    }

    #[test]
    fn out_of_range_tolerance_is_rejected() {
        let rows = vec![alternative(1, 11, 11, "left", 0, 100)];
        let error = qualify_representation_compression(&rows, config(1_000_001, 0))
            .expect_err("tolerance must use the bounded PPM scale");
        assert!(matches!(
            error,
            RepresentationCompressionError::RegretToleranceOutOfRange {
                tolerance_ppm: 1_000_001
            }
        ));
    }

    #[test]
    fn compression_budget_is_independent_from_decision_tolerance() {
        let rows = vec![
            alternative(1, 11, 99, "left", 0, 50_000),
            alternative(2, 12, 99, "left", 50_000, 0),
        ];
        let result = qualify_representation_compression(&rows, config(100_000, 10_000))
            .expect("valid but over compression regret budget");
        assert!(result.classes[0].compressed_information_set_epsilon_sufficient);
        assert!(!result.classes[0].representation_decision_preserved);
        assert_eq!(result.classes[0].added_worst_case_regret_ppm, 50_000);
    }
}
