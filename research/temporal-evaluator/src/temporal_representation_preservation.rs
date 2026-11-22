//! Temporal decision-preservation profiles for compressed recurrent states.
//!
//! This module composes the single-epoch decision-sufficient representation
//! certificate across a frozen sequence of future decision epochs. It reuses
//! decision_sufficient_representation as the only representation/regret
//! evaluator. It does not compute the selected-action sufficiency horizon
//! owned by #11646, calibrate loss intervals owned by #11655, or authorize
//! actions.
//!
//! Every epoch must contain the same evaluator hidden-world identities. This
//! prevents a compression profile from appearing better by dropping the
//! counterfactual worlds that become difficult later in the episode. The
//! model/configuration identity and deterministic policy seed are frozen
//! across the temporal profile; observation and recurrent-state identities
//! are expected to evolve with time.
//!
//! The result is a representation-preservation profile, not a physical action
//! commitment bound and not a substitute for the calibrated action-horizon
//! reference. Candidate/evaluator isolation under #9437 remains required.

use crate::decision_sufficient_representation::{
    RepresentationAlternative, RepresentationCompressionCertificate,
    RepresentationCompressionConfig, RepresentationCompressionError,
    qualify_representation_compression,
};
use std::collections::BTreeMap;
use std::fmt;

const MAX_TEMPORAL_EPOCHS: usize = 4_096;

/// One paired evaluator snapshot at a future decision epoch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalRepresentationEpoch {
    /// Contiguous decision-epoch index in the frozen episode profile.
    pub epoch_index: u64,
    /// Exactly one evaluator alternative per hidden world in the frozen cohort.
    pub alternatives: Vec<RepresentationAlternative>,
}

/// Frozen temporal qualification contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalRepresentationConfig {
    /// First expected decision epoch. Epochs are required to be contiguous.
    pub first_epoch_index: u64,
    /// Number of epochs declared before confirmatory evaluation.
    pub expected_epoch_count: usize,
    /// Per-epoch decision and compression tolerances.
    pub compression: RepresentationCompressionConfig,
}

/// Per-epoch interpretation of the underlying representation certificate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalRepresentationEpochCertificate {
    pub epoch_index: u64,
    pub source_information_sufficient: bool,
    pub compressed_information_sufficient: bool,
    pub compression_regret_within_budget: bool,
    pub candidate_actions_within_epsilon: bool,
    pub representation_preserved: bool,
    pub worst_added_regret_ppm: u32,
    /// Exact single-epoch certificate from the canonical evaluator.
    pub certificate: RepresentationCompressionCertificate,
}

/// Temporal representation qualification over a frozen consecutive horizon.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalRepresentationCertificate {
    pub first_epoch_index: u64,
    pub epoch_count: u64,
    /// Longest prefix, starting at the first epoch, for which every compressed
    /// information class remains epsilon-sufficient and within the added-regret
    /// budget. This is not #11646's selected-action decision-sufficiency horizon.
    pub representation_preservation_prefix_epochs: u64,
    pub all_epochs_representation_preserved: bool,
    pub all_source_information_sufficient: bool,
    pub all_compressed_information_sufficient: bool,
    pub all_compression_regret_within_budget: bool,
    /// Whether the candidate's actually emitted action remains within epsilon
    /// in every class at every epoch. Kept separate from information sufficiency.
    pub all_candidate_actions_within_epsilon: bool,
    pub maximum_added_regret_ppm: u32,
    pub first_representation_failure_epoch: Option<u64>,
    pub first_source_insufficiency_epoch: Option<u64>,
    pub first_compressed_insufficiency_epoch: Option<u64>,
    pub first_compression_budget_failure_epoch: Option<u64>,
    pub first_candidate_action_failure_epoch: Option<u64>,
    pub epochs: Vec<TemporalRepresentationEpochCertificate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalRepresentationError {
    EmptyEpochs,
    TooManyEpochs {
        requested: usize,
        maximum: usize,
    },
    EpochCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonContiguousEpoch {
        expected: u64,
        actual: u64,
    },
    EmptyEpochAlternatives {
        epoch_index: u64,
    },
    DuplicateWorldId {
        epoch_index: u64,
        world_id: u64,
    },
    EmptyHiddenWorldDigest {
        epoch_index: u64,
        world_id: u64,
    },
    DuplicateHiddenWorldDigest {
        epoch_index: u64,
        world_id: u64,
        first_world_id: u64,
    },
    HiddenWorldSetMismatch {
        epoch_index: u64,
    },
    HiddenWorldIdentityChanged {
        epoch_index: u64,
        world_id: u64,
    },
    ModelConfigurationChanged {
        epoch_index: u64,
        world_id: u64,
    },
    PolicySeedChanged {
        epoch_index: u64,
        world_id: u64,
    },
    EpochIndexOverflow,
    ArithmeticOverflow,
    Representation(RepresentationCompressionError),
}

impl fmt::Display for TemporalRepresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEpochs => write!(
                formatter,
                "temporal representation profile requires at least one epoch"
            ),
            Self::TooManyEpochs { requested, maximum } => write!(
                formatter,
                "temporal profile has {requested} epochs; maximum is {maximum}"
            ),
            Self::EpochCountMismatch { expected, actual } => write!(
                formatter,
                "temporal profile declares {expected} epochs but supplies {actual}"
            ),
            Self::NonContiguousEpoch { expected, actual } => write!(
                formatter,
                "expected decision epoch {expected}, received {actual}"
            ),
            Self::EmptyEpochAlternatives { epoch_index } => write!(
                formatter,
                "decision epoch {epoch_index} has no hidden-world alternatives"
            ),
            Self::DuplicateWorldId {
                epoch_index,
                world_id,
            } => write!(
                formatter,
                "decision epoch {epoch_index} repeats hidden-world ID {world_id}"
            ),
            Self::EmptyHiddenWorldDigest {
                epoch_index,
                world_id,
            } => write!(
                formatter,
                "decision epoch {epoch_index} has an empty hidden-world digest for world {world_id}"
            ),
            Self::DuplicateHiddenWorldDigest {
                epoch_index,
                world_id,
                first_world_id,
            } => write!(
                formatter,
                "decision epoch {epoch_index} assigns the same hidden-world digest to worlds {first_world_id} and {world_id}"
            ),
            Self::HiddenWorldSetMismatch { epoch_index } => write!(
                formatter,
                "decision epoch {epoch_index} does not contain the frozen hidden-world set"
            ),
            Self::HiddenWorldIdentityChanged {
                epoch_index,
                world_id,
            } => write!(
                formatter,
                "hidden-world digest changed for world {world_id} at epoch {epoch_index}"
            ),
            Self::ModelConfigurationChanged {
                epoch_index,
                world_id,
            } => write!(
                formatter,
                "model/configuration identity changed for world {world_id} at epoch {epoch_index}"
            ),
            Self::PolicySeedChanged {
                epoch_index,
                world_id,
            } => write!(
                formatter,
                "deterministic policy seed changed for world {world_id} at epoch {epoch_index}"
            ),
            Self::EpochIndexOverflow => {
                write!(formatter, "temporal decision-epoch index overflowed")
            }
            Self::ArithmeticOverflow => write!(
                formatter,
                "temporal representation certificate arithmetic overflowed"
            ),
            Self::Representation(error) => write!(
                formatter,
                "single-epoch representation qualification failed: {error}"
            ),
        }
    }
}

impl std::error::Error for TemporalRepresentationError {}

impl From<RepresentationCompressionError> for TemporalRepresentationError {
    fn from(value: RepresentationCompressionError) -> Self {
        Self::Representation(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WorldIdentity {
    hidden_world_sha256: [u8; 32],
    model_configuration_sha256: [u8; 32],
    policy_seed: u64,
}

fn world_identities(
    epoch: &TemporalRepresentationEpoch,
) -> Result<BTreeMap<u64, WorldIdentity>, TemporalRepresentationError> {
    if epoch.alternatives.is_empty() {
        return Err(TemporalRepresentationError::EmptyEpochAlternatives {
            epoch_index: epoch.epoch_index,
        });
    }

    let mut identities = BTreeMap::new();
    let mut hidden_digests = BTreeMap::new();
    for alternative in &epoch.alternatives {
        let world_id = alternative.world.evaluator_world_id;
        let hidden_digest = alternative.world.hidden_world_sha256;
        if hidden_digest.iter().all(|byte| *byte == 0) {
            return Err(TemporalRepresentationError::EmptyHiddenWorldDigest {
                epoch_index: epoch.epoch_index,
                world_id,
            });
        }
        if let Some(first_world_id) = hidden_digests.insert(hidden_digest, world_id) {
            return Err(TemporalRepresentationError::DuplicateHiddenWorldDigest {
                epoch_index: epoch.epoch_index,
                world_id,
                first_world_id,
            });
        }
        let identity = WorldIdentity {
            hidden_world_sha256: alternative.world.hidden_world_sha256,
            model_configuration_sha256: alternative.world.candidate_view.model_configuration_sha256,
            policy_seed: alternative.world.candidate_view.policy_seed,
        };
        if identities.insert(world_id, identity).is_some() {
            return Err(TemporalRepresentationError::DuplicateWorldId {
                epoch_index: epoch.epoch_index,
                world_id,
            });
        }
    }
    Ok(identities)
}

fn first_epoch_failure(current: Option<u64>, failed: bool, epoch_index: u64) -> Option<u64> {
    if current.is_some() {
        current
    } else if failed {
        Some(epoch_index)
    } else {
        None
    }
}

/// Evaluate a sequence of decision epochs under one frozen compression profile.
///
/// The same hidden-world cohort, hidden-world identity, model/configuration and
/// deterministic policy seed must be present at every epoch. Candidate-visible
/// observations and recurrent-state identities may evolve normally.
///
/// At each epoch the function delegates to
/// qualify_representation_compression(). It then reports the longest contiguous
/// representation-preservation prefix and first failure locations. It does not
/// compute the calibrated selected-action horizon from #11646 or perform
/// simultaneous calibration from #11655.
pub fn qualify_temporal_representation_compression(
    epochs: &[TemporalRepresentationEpoch],
    config: TemporalRepresentationConfig,
) -> Result<TemporalRepresentationCertificate, TemporalRepresentationError> {
    if epochs.is_empty() || config.expected_epoch_count == 0 {
        return Err(TemporalRepresentationError::EmptyEpochs);
    }
    if config.expected_epoch_count > MAX_TEMPORAL_EPOCHS {
        return Err(TemporalRepresentationError::TooManyEpochs {
            requested: config.expected_epoch_count,
            maximum: MAX_TEMPORAL_EPOCHS,
        });
    }
    if epochs.len() != config.expected_epoch_count {
        return Err(TemporalRepresentationError::EpochCountMismatch {
            expected: config.expected_epoch_count,
            actual: epochs.len(),
        });
    }

    let expected_first_epoch = config.first_epoch_index;
    let first_identities = world_identities(&epochs[0])?;
    let reference_identity = first_identities.values().next().copied().ok_or(
        TemporalRepresentationError::EmptyEpochAlternatives {
            epoch_index: epochs[0].epoch_index,
        },
    )?;

    for (index, epoch) in epochs.iter().enumerate() {
        let offset =
            u64::try_from(index).map_err(|_| TemporalRepresentationError::EpochIndexOverflow)?;
        let expected_epoch = expected_first_epoch
            .checked_add(offset)
            .ok_or(TemporalRepresentationError::EpochIndexOverflow)?;
        if epoch.epoch_index != expected_epoch {
            return Err(TemporalRepresentationError::NonContiguousEpoch {
                expected: expected_epoch,
                actual: epoch.epoch_index,
            });
        }

        let identities = world_identities(epoch)?;
        if identities.len() != first_identities.len()
            || identities.keys().ne(first_identities.keys())
        {
            return Err(TemporalRepresentationError::HiddenWorldSetMismatch {
                epoch_index: epoch.epoch_index,
            });
        }

        for (world_id, first_identity) in &first_identities {
            let current = identities.get(world_id).ok_or(
                TemporalRepresentationError::HiddenWorldSetMismatch {
                    epoch_index: epoch.epoch_index,
                },
            )?;
            if current.hidden_world_sha256 != first_identity.hidden_world_sha256 {
                return Err(TemporalRepresentationError::HiddenWorldIdentityChanged {
                    epoch_index: epoch.epoch_index,
                    world_id: *world_id,
                });
            }
            if current.model_configuration_sha256 != reference_identity.model_configuration_sha256 {
                return Err(TemporalRepresentationError::ModelConfigurationChanged {
                    epoch_index: epoch.epoch_index,
                    world_id: *world_id,
                });
            }
            if current.policy_seed != reference_identity.policy_seed {
                return Err(TemporalRepresentationError::PolicySeedChanged {
                    epoch_index: epoch.epoch_index,
                    world_id: *world_id,
                });
            }
        }
    }

    let mut epoch_certificates = Vec::with_capacity(epochs.len());
    let mut representation_preservation_prefix_epochs = 0_u64;
    let mut prefix_open = true;
    let mut all_epochs_representation_preserved = true;
    let mut all_source_information_sufficient = true;
    let mut all_compressed_information_sufficient = true;
    let mut all_compression_regret_within_budget = true;
    let mut all_candidate_actions_within_epsilon = true;
    let mut maximum_added_regret_ppm = 0_u32;
    let mut first_representation_failure_epoch = None;
    let mut first_source_insufficiency_epoch = None;
    let mut first_compressed_insufficiency_epoch = None;
    let mut first_compression_budget_failure_epoch = None;
    let mut first_candidate_action_failure_epoch = None;

    for epoch in epochs {
        let certificate =
            qualify_representation_compression(&epoch.alternatives, config.compression)?;

        let source_information_sufficient = certificate
            .classes
            .iter()
            .all(|class| class.all_source_classes_epsilon_sufficient);
        let compressed_information_sufficient = certificate
            .classes
            .iter()
            .all(|class| class.compressed_information_set_epsilon_sufficient);
        let compression_regret_within_budget = certificate.classes.iter().all(|class| {
            class.added_worst_case_regret_ppm <= config.compression.maximum_added_regret_ppm
        });
        let candidate_actions_within_epsilon = certificate.all_candidate_actions_within_epsilon;
        let representation_preserved = certificate.all_compressed_classes_decision_preserving;

        all_epochs_representation_preserved &= representation_preserved;
        all_source_information_sufficient &= source_information_sufficient;
        all_compressed_information_sufficient &= compressed_information_sufficient;
        all_compression_regret_within_budget &= compression_regret_within_budget;
        all_candidate_actions_within_epsilon &= candidate_actions_within_epsilon;
        maximum_added_regret_ppm = maximum_added_regret_ppm.max(certificate.worst_added_regret_ppm);

        first_representation_failure_epoch = first_epoch_failure(
            first_representation_failure_epoch,
            !representation_preserved,
            epoch.epoch_index,
        );
        first_source_insufficiency_epoch = first_epoch_failure(
            first_source_insufficiency_epoch,
            !source_information_sufficient,
            epoch.epoch_index,
        );
        first_compressed_insufficiency_epoch = first_epoch_failure(
            first_compressed_insufficiency_epoch,
            !compressed_information_sufficient,
            epoch.epoch_index,
        );
        first_compression_budget_failure_epoch = first_epoch_failure(
            first_compression_budget_failure_epoch,
            !compression_regret_within_budget,
            epoch.epoch_index,
        );
        first_candidate_action_failure_epoch = first_epoch_failure(
            first_candidate_action_failure_epoch,
            !candidate_actions_within_epsilon,
            epoch.epoch_index,
        );

        if prefix_open && representation_preserved {
            representation_preservation_prefix_epochs = representation_preservation_prefix_epochs
                .checked_add(1)
                .ok_or(TemporalRepresentationError::ArithmeticOverflow)?;
        } else {
            prefix_open = false;
        }

        epoch_certificates.push(TemporalRepresentationEpochCertificate {
            epoch_index: epoch.epoch_index,
            source_information_sufficient,
            compressed_information_sufficient,
            compression_regret_within_budget,
            candidate_actions_within_epsilon,
            representation_preserved,
            worst_added_regret_ppm: certificate.worst_added_regret_ppm,
            certificate,
        });
    }

    Ok(TemporalRepresentationCertificate {
        first_epoch_index: config.first_epoch_index,
        epoch_count: u64::try_from(epochs.len())
            .map_err(|_| TemporalRepresentationError::ArithmeticOverflow)?,
        representation_preservation_prefix_epochs,
        all_epochs_representation_preserved,
        all_source_information_sufficient,
        all_compressed_information_sufficient,
        all_compression_regret_within_budget,
        all_candidate_actions_within_epsilon,
        maximum_added_regret_ppm,
        first_representation_failure_epoch,
        first_source_insufficiency_epoch,
        first_compressed_insufficiency_epoch,
        first_compression_budget_failure_epoch,
        first_candidate_action_failure_epoch,
        epochs: epoch_certificates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_observability::{ActionLoss, CandidateViewIdentity, CounterfactualWorld};

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn alternative(
        epoch_index: u64,
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
                hidden_world_sha256: digest(world_id as u8 + 10),
                candidate_view: CandidateViewIdentity {
                    observation_history_sha256: digest(epoch_index as u8 + 30),
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

    fn epoch(
        epoch_index: u64,
        first_compressed: u8,
        second_compressed: u8,
        first_action: &str,
        second_action: &str,
        first_losses: (u32, u32),
        second_losses: (u32, u32),
    ) -> TemporalRepresentationEpoch {
        TemporalRepresentationEpoch {
            epoch_index,
            alternatives: vec![
                alternative(
                    epoch_index,
                    1,
                    11 + epoch_index as u8,
                    first_compressed,
                    first_action,
                    first_losses.0,
                    first_losses.1,
                ),
                alternative(
                    epoch_index,
                    2,
                    21 + epoch_index as u8,
                    second_compressed,
                    second_action,
                    second_losses.0,
                    second_losses.1,
                ),
            ],
        }
    }

    fn config(
        first_epoch_index: u64,
        count: usize,
        epsilon: u32,
        added: u32,
    ) -> TemporalRepresentationConfig {
        TemporalRepresentationConfig {
            first_epoch_index,
            expected_epoch_count: count,
            compression: RepresentationCompressionConfig {
                epsilon_regret_tolerance_ppm: epsilon,
                maximum_added_regret_ppm: added,
            },
        }
    }

    #[test]
    fn identity_compression_preserves_every_epoch() {
        let epochs = vec![
            epoch(5, 16, 26, "left", "right", (0, 100_000), (100_000, 0)),
            epoch(6, 17, 27, "left", "right", (0, 100_000), (100_000, 0)),
            epoch(7, 18, 28, "left", "right", (0, 100_000), (100_000, 0)),
        ];
        let result = qualify_temporal_representation_compression(&epochs, config(5, 3, 0, 0))
            .expect("identity compression should preserve decisions");

        assert_eq!(result.representation_preservation_prefix_epochs, 3);
        assert!(result.all_epochs_representation_preserved);
        assert!(result.all_candidate_actions_within_epsilon);
        assert_eq!(result.maximum_added_regret_ppm, 0);
        assert_eq!(result.first_representation_failure_epoch, None);
    }

    #[test]
    fn current_decision_can_be_preserved_while_a_future_epoch_fails() {
        let epochs = vec![
            // At epoch 5, both hidden worlds prefer LEFT, so merging them is harmless.
            epoch(5, 99, 99, "left", "left", (0, 100_000), (0, 100_000)),
            // At epoch 6, the worlds require opposed actions; the same compression
            // collision now creates a full one-million-PPM regret floor.
            epoch(6, 98, 98, "left", "left", (0, 1_000_000), (1_000_000, 0)),
            epoch(7, 97, 97, "left", "left", (0, 1_000_000), (1_000_000, 0)),
        ];
        let result =
            qualify_temporal_representation_compression(&epochs, config(5, 3, 100_000, 100_000))
                .expect("the finite temporal fixture is well-formed");

        assert!(result.epochs[0].representation_preserved);
        assert!(!result.epochs[1].representation_preserved);
        assert_eq!(result.representation_preservation_prefix_epochs, 1);
        assert_eq!(result.first_representation_failure_epoch, Some(6));
        assert_eq!(result.first_compressed_insufficiency_epoch, Some(6));
        assert_eq!(result.first_candidate_action_failure_epoch, Some(6));
        assert_eq!(result.maximum_added_regret_ppm, 1_000_000);
    }

    #[test]
    fn recovery_after_a_failed_epoch_does_not_restore_the_contiguous_prefix() {
        let epochs = vec![
            epoch(0, 11, 21, "left", "right", (0, 100_000), (100_000, 0)),
            epoch(1, 99, 99, "left", "left", (0, 1_000_000), (1_000_000, 0)),
            epoch(2, 13, 23, "left", "right", (0, 100_000), (100_000, 0)),
        ];
        let result =
            qualify_temporal_representation_compression(&epochs, config(0, 3, 100_000, 100_000))
                .expect("the frozen temporal cohort is valid");

        assert!(result.epochs[0].representation_preserved);
        assert!(!result.epochs[1].representation_preserved);
        assert!(result.epochs[2].representation_preserved);
        assert_eq!(result.representation_preservation_prefix_epochs, 1);
        assert_eq!(result.first_representation_failure_epoch, Some(1));
        assert!(!result.all_epochs_representation_preserved);
    }

    #[test]
    fn hidden_world_digest_change_is_rejected() {
        let first = epoch(5, 11, 21, "left", "right", (0, 100_000), (100_000, 0));
        let mut second = epoch(6, 12, 22, "left", "right", (0, 100_000), (100_000, 0));
        second.alternatives[1].world.hidden_world_sha256 = digest(99);
        let error =
            qualify_temporal_representation_compression(&[first, second], config(5, 2, 0, 0))
                .expect_err("a world ID cannot change its hidden-world identity");
        assert!(matches!(
            error,
            TemporalRepresentationError::HiddenWorldIdentityChanged {
                epoch_index: 6,
                world_id: 2
            }
        ));
    }

    #[test]
    fn candidate_action_failure_is_separate_from_information_preservation() {
        let epochs = vec![
            // The compressed state keeps the two worlds distinct, but the candidate
            // emits LEFT in both; the second world actually requires RIGHT.
            epoch(0, 11, 21, "left", "left", (0, 100_000), (100_000, 0)),
        ];
        let result = qualify_temporal_representation_compression(&epochs, config(0, 1, 0, 0))
            .expect("information partitions are well-formed");

        assert!(result.all_epochs_representation_preserved);
        assert!(!result.all_candidate_actions_within_epsilon);
        assert_eq!(result.first_representation_failure_epoch, None);
        assert_eq!(result.first_candidate_action_failure_epoch, Some(0));
    }

    #[test]
    fn non_contiguous_epochs_are_rejected() {
        let epochs = vec![
            epoch(5, 11, 21, "left", "right", (0, 100_000), (100_000, 0)),
            epoch(7, 12, 22, "left", "right", (0, 100_000), (100_000, 0)),
        ];
        let error = qualify_temporal_representation_compression(&epochs, config(5, 2, 0, 0))
            .expect_err("epoch gap must be rejected");
        assert!(matches!(
            error,
            TemporalRepresentationError::NonContiguousEpoch {
                expected: 6,
                actual: 7
            }
        ));
    }

    #[test]
    fn hidden_world_drop_is_rejected_instead_of_improving_the_score() {
        let first = epoch(5, 11, 21, "left", "right", (0, 100_000), (100_000, 0));
        let mut second = epoch(6, 12, 22, "left", "right", (0, 100_000), (100_000, 0));
        let _ = second.alternatives.pop();
        let epochs = vec![first, second];
        let error = qualify_temporal_representation_compression(&epochs, config(5, 2, 0, 0))
            .expect_err("every epoch must retain the frozen hidden-world cohort");
        assert!(matches!(
            error,
            TemporalRepresentationError::HiddenWorldSetMismatch { epoch_index: 6 }
        ));
    }

    #[test]
    fn duplicate_hidden_world_digests_are_rejected() {
        let mut only_epoch = epoch(0, 11, 21, "left", "right", (0, 100_000), (100_000, 0));
        only_epoch.alternatives[1].world.hidden_world_sha256 =
            only_epoch.alternatives[0].world.hidden_world_sha256;
        let error = qualify_temporal_representation_compression(&[only_epoch], config(0, 1, 0, 0))
            .expect_err("two world IDs cannot alias one hidden-world digest");
        assert!(matches!(
            error,
            TemporalRepresentationError::DuplicateHiddenWorldDigest {
                epoch_index: 0,
                world_id: 2,
                first_world_id: 1
            }
        ));
    }

    #[test]
    fn model_configuration_change_is_rejected() {
        let first = epoch(5, 11, 21, "left", "right", (0, 100_000), (100_000, 0));
        let mut second = epoch(6, 12, 22, "left", "right", (0, 100_000), (100_000, 0));
        second.alternatives[1]
            .world
            .candidate_view
            .model_configuration_sha256 = digest(9);
        let epochs = vec![first, second];
        let error = qualify_temporal_representation_compression(&epochs, config(5, 2, 0, 0))
            .expect_err("model identity must be fixed across the temporal profile");
        assert!(matches!(
            error,
            TemporalRepresentationError::ModelConfigurationChanged {
                epoch_index: 6,
                world_id: 2
            }
        ));
    }

    #[test]
    fn policy_seed_change_is_rejected() {
        let first = epoch(5, 11, 21, "left", "right", (0, 100_000), (100_000, 0));
        let mut second = epoch(6, 12, 22, "left", "right", (0, 100_000), (100_000, 0));
        second.alternatives[0].world.candidate_view.policy_seed = 99;
        let epochs = vec![first, second];
        let error = qualify_temporal_representation_compression(&epochs, config(5, 2, 0, 0))
            .expect_err("policy seed must be fixed across the temporal profile");
        assert!(matches!(
            error,
            TemporalRepresentationError::PolicySeedChanged {
                epoch_index: 6,
                world_id: 1
            }
        ));
    }

    #[test]
    fn epoch_count_mismatch_is_rejected() {
        let epochs = vec![epoch(
            5,
            11,
            21,
            "left",
            "right",
            (0, 100_000),
            (100_000, 0),
        )];
        let error = qualify_temporal_representation_compression(&epochs, config(5, 2, 0, 0))
            .expect_err("declared profile length must match supplied epochs");
        assert!(matches!(
            error,
            TemporalRepresentationError::EpochCountMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }
}
