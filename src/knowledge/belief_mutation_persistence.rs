// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Export/validation contract for restart-stable epistemic-support persistence.
//!
//! EKM-029 makes public-path authorization consumption reconstructable from the
//! mutation history by requiring idempotent replay to reuse the original
//! authorization ID. This module captures that closed state into a deterministic
//! versioned capsule and validates the complete revision chain.
//!
//! This module deliberately does **not** hydrate an [`EpistemicSupportStore`]
//! from a capsule and does not perform file/database I/O. Export + structural
//! validation are qualified separately before restore authority is introduced.

use super::belief_mutation_firewall::{
    BeliefMutationReceipt, BeliefMutationReceiptId, EpistemicSupportStore,
};
use super::belief_revision_receipt::BeliefRevisionReceiptId;
use super::claim_evidence::ClaimId;
use super::knowledge_weight_routing::BoundedWeight;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefMutationPersistenceVersion {
    V1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedEpistemicSupportStateV1 {
    pub claim_id: ClaimId,
    /// Support before the first persisted mutation, or current support when the
    /// claim has never been revised.
    pub baseline_support: BoundedWeight,
    pub current_support: BoundedWeight,
    pub revision: u64,
    pub initialized_at_cycle: u64,
    pub last_updated_cycle: u64,
    pub last_mutation_id: Option<BeliefMutationReceiptId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedBeliefMutationV1 {
    pub id: BeliefMutationReceiptId,
    pub source_revision_receipt_id: BeliefRevisionReceiptId,
    pub claim_id: ClaimId,
    pub proposed_delta: f32,
    pub support_before: BoundedWeight,
    pub support_after: BoundedWeight,
    pub state_revision_before: u64,
    pub state_revision_after: u64,
    pub authorization_id: String,
    pub authority_label: String,
    pub authorized_at_cycle: u64,
    pub applied_at_cycle: u64,
}

impl PersistedBeliefMutationV1 {
    fn from_receipt(receipt: &BeliefMutationReceipt) -> Self {
        Self {
            id: receipt.id(),
            source_revision_receipt_id: receipt.source_revision_receipt_id(),
            claim_id: receipt.claim_id(),
            proposed_delta: receipt.proposed_delta(),
            support_before: receipt.support_before(),
            support_after: receipt.support_after(),
            state_revision_before: receipt.state_revision_before(),
            state_revision_after: receipt.state_revision_after(),
            authorization_id: receipt.authorization_id().to_string(),
            authority_label: receipt.authority_label().to_string(),
            authorized_at_cycle: receipt.authorized_at_cycle(),
            applied_at_cycle: receipt.applied_at_cycle(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationPersistenceCapsuleV1 {
    version: BeliefMutationPersistenceVersion,
    captured_at_cycle: u64,
    states: Vec<PersistedEpistemicSupportStateV1>,
    mutations: Vec<PersistedBeliefMutationV1>,
    consumed_authorization_count: usize,
}

impl BeliefMutationPersistenceCapsuleV1 {
    /// Capture a complete declared support-store inventory.
    ///
    /// `claim_ids` is explicit because EKM-026 intentionally does not expose a
    /// public state iterator. The capture fails unless the supplied unique IDs
    /// account for every registered support state.
    pub fn capture(
        store: &EpistemicSupportStore,
        claim_ids: &[ClaimId],
        captured_at_cycle: u64,
    ) -> Result<Self, BeliefMutationPersistenceError> {
        let mut declared = HashSet::new();
        for claim_id in claim_ids {
            if !declared.insert(*claim_id) {
                return Err(BeliefMutationPersistenceError::DuplicateDeclaredClaim(
                    *claim_id,
                ));
            }
        }
        if declared.len() != store.len() {
            return Err(BeliefMutationPersistenceError::StateInventoryCountMismatch {
                declared: declared.len(),
                actual: store.len(),
            });
        }
        for claim_id in &declared {
            if store.state(*claim_id).is_none() {
                return Err(BeliefMutationPersistenceError::MissingDeclaredState(
                    *claim_id,
                ));
            }
        }

        let mutations = store
            .history()
            .iter()
            .map(PersistedBeliefMutationV1::from_receipt)
            .collect::<Vec<_>>();

        validate_global_history(
            &mutations,
            &declared,
            store.consumed_authorization_count(),
            captured_at_cycle,
        )?;

        let mut by_claim: HashMap<ClaimId, Vec<&PersistedBeliefMutationV1>> = HashMap::new();
        for mutation in &mutations {
            by_claim.entry(mutation.claim_id).or_default().push(mutation);
        }

        let mut states = Vec::with_capacity(declared.len());
        for claim_id in declared {
            let live = store
                .state(claim_id)
                .ok_or(BeliefMutationPersistenceError::MissingDeclaredState(
                    claim_id,
                ))?;
            if live.last_updated_cycle() > captured_at_cycle {
                return Err(BeliefMutationPersistenceError::CapturePredatesState {
                    claim_id,
                    captured_at_cycle,
                    state_cycle: live.last_updated_cycle(),
                });
            }

            let mut claim_mutations = by_claim.remove(&claim_id).unwrap_or_default();
            claim_mutations.sort_by_key(|mutation| mutation.state_revision_after);

            let baseline_support = if claim_mutations.is_empty() {
                if live.revision() != 0 {
                    return Err(BeliefMutationPersistenceError::UnmutatedStateRevisionNonZero {
                        claim_id,
                        revision: live.revision(),
                    });
                }
                if live.last_mutation_id().is_some() {
                    return Err(BeliefMutationPersistenceError::UnmutatedStateHasMutationId {
                        claim_id,
                        mutation_id: live.last_mutation_id(),
                    });
                }
                if live.last_updated_cycle() != live.initialized_at_cycle() {
                    return Err(BeliefMutationPersistenceError::UnmutatedStateCycleMismatch {
                        claim_id,
                        initialized_at_cycle: live.initialized_at_cycle(),
                        last_updated_cycle: live.last_updated_cycle(),
                    });
                }
                live.support()
            } else {
                validate_claim_chain(claim_id, live, &claim_mutations)?;
                claim_mutations[0].support_before
            };

            states.push(PersistedEpistemicSupportStateV1 {
                claim_id,
                baseline_support,
                current_support: live.support(),
                revision: live.revision(),
                initialized_at_cycle: live.initialized_at_cycle(),
                last_updated_cycle: live.last_updated_cycle(),
                last_mutation_id: live.last_mutation_id(),
            });
        }
        states.sort_by_key(|state| state.claim_id);

        if !by_claim.is_empty() {
            let mutation = by_claim
                .values()
                .flat_map(|records| records.iter())
                .next()
                .expect("non-empty map has at least one mutation");
            return Err(BeliefMutationPersistenceError::MutationForUndeclaredState(
                mutation.claim_id,
            ));
        }

        Ok(Self {
            version: BeliefMutationPersistenceVersion::V1,
            captured_at_cycle,
            states,
            mutations,
            consumed_authorization_count: store.consumed_authorization_count(),
        })
    }

    pub fn version(&self) -> BeliefMutationPersistenceVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn states(&self) -> &[PersistedEpistemicSupportStateV1] {
        &self.states
    }

    pub fn mutations(&self) -> &[PersistedBeliefMutationV1] {
        &self.mutations
    }

    pub fn consumed_authorization_count(&self) -> usize {
        self.consumed_authorization_count
    }

    pub fn claim_ids(&self) -> Vec<ClaimId> {
        self.states.iter().map(|state| state.claim_id).collect()
    }

    /// Check whether the live store is still exactly represented by this capsule.
    ///
    /// A later observation cycle is allowed; state/history/authorization semantics
    /// must remain identical.
    pub fn validate_live(
        &self,
        store: &EpistemicSupportStore,
        observed_at_cycle: u64,
    ) -> Result<(), BeliefMutationPersistenceError> {
        if observed_at_cycle < self.captured_at_cycle {
            return Err(BeliefMutationPersistenceError::ObservationPredatesCapsule {
                observed_at_cycle,
                captured_at_cycle: self.captured_at_cycle,
            });
        }
        let live = Self::capture(store, &self.claim_ids(), observed_at_cycle)?;
        if live.states != self.states {
            return Err(BeliefMutationPersistenceError::LiveStateMismatch);
        }
        if live.mutations != self.mutations {
            return Err(BeliefMutationPersistenceError::LiveMutationHistoryMismatch);
        }
        if live.consumed_authorization_count != self.consumed_authorization_count {
            return Err(
                BeliefMutationPersistenceError::LiveAuthorizationCountMismatch {
                    expected: self.consumed_authorization_count,
                    actual: live.consumed_authorization_count,
                },
            );
        }
        Ok(())
    }
}

fn validate_global_history(
    mutations: &[PersistedBeliefMutationV1],
    declared_claims: &HashSet<ClaimId>,
    consumed_authorization_count: usize,
    captured_at_cycle: u64,
) -> Result<(), BeliefMutationPersistenceError> {
    if consumed_authorization_count != mutations.len() {
        return Err(
            BeliefMutationPersistenceError::AuthorizationHistoryNotClosed {
                consumed_authorizations: consumed_authorization_count,
                mutation_receipts: mutations.len(),
            },
        );
    }

    let mut mutation_ids = HashSet::new();
    let mut source_receipts = HashSet::new();
    let mut authorization_ids = HashSet::new();
    let mut previous_mutation_id = None;

    for mutation in mutations {
        if !declared_claims.contains(&mutation.claim_id) {
            return Err(BeliefMutationPersistenceError::MutationForUndeclaredState(
                mutation.claim_id,
            ));
        }
        if !mutation_ids.insert(mutation.id) {
            return Err(BeliefMutationPersistenceError::DuplicateMutationId(
                mutation.id,
            ));
        }
        if !source_receipts.insert(mutation.source_revision_receipt_id) {
            return Err(
                BeliefMutationPersistenceError::DuplicateSourceRevisionReceipt(
                    mutation.source_revision_receipt_id,
                ),
            );
        }
        if !authorization_ids.insert(mutation.authorization_id.clone()) {
            return Err(BeliefMutationPersistenceError::DuplicateAuthorizationId(
                mutation.authorization_id.clone(),
            ));
        }
        if mutation.authorized_at_cycle > mutation.applied_at_cycle {
            return Err(BeliefMutationPersistenceError::MutationPredatesAuthorization {
                mutation_id: mutation.id,
                authorized_at_cycle: mutation.authorized_at_cycle,
                applied_at_cycle: mutation.applied_at_cycle,
            });
        }
        if mutation.applied_at_cycle > captured_at_cycle {
            return Err(BeliefMutationPersistenceError::CapturePredatesMutation {
                mutation_id: mutation.id,
                captured_at_cycle,
                applied_at_cycle: mutation.applied_at_cycle,
            });
        }
        if let Some(previous) = previous_mutation_id {
            if mutation.id.0 <= previous {
                return Err(BeliefMutationPersistenceError::MutationIdsNotMonotonic {
                    previous: BeliefMutationReceiptId(previous),
                    current: mutation.id,
                });
            }
        }
        previous_mutation_id = Some(mutation.id.0);
    }
    Ok(())
}

fn validate_claim_chain(
    claim_id: ClaimId,
    live: &super::belief_mutation_firewall::EpistemicSupportState,
    mutations: &[&PersistedBeliefMutationV1],
) -> Result<(), BeliefMutationPersistenceError> {
    let mut expected_revision = 0u64;
    let mut previous_support_after = None;

    for mutation in mutations {
        if mutation.state_revision_before != expected_revision
            || mutation.state_revision_after != expected_revision + 1
        {
            return Err(BeliefMutationPersistenceError::RevisionChainBroken {
                claim_id,
                expected_before: expected_revision,
                actual_before: mutation.state_revision_before,
                actual_after: mutation.state_revision_after,
            });
        }
        if let Some(previous) = previous_support_after {
            if mutation.support_before != previous {
                return Err(BeliefMutationPersistenceError::SupportChainBroken {
                    claim_id,
                    expected: previous.get(),
                    actual: mutation.support_before.get(),
                });
            }
        }
        let expected_after = mutation.support_before.get() + mutation.proposed_delta;
        if (mutation.support_after.get() - expected_after).abs() > 1e-6 {
            return Err(BeliefMutationPersistenceError::MutationDeltaMismatch {
                mutation_id: mutation.id,
                expected_after,
                actual_after: mutation.support_after.get(),
            });
        }
        expected_revision = mutation.state_revision_after;
        previous_support_after = Some(mutation.support_after);
    }

    let latest = mutations
        .last()
        .expect("claim chain validation requires at least one mutation");
    if live.revision() != latest.state_revision_after {
        return Err(BeliefMutationPersistenceError::FinalRevisionMismatch {
            claim_id,
            expected: latest.state_revision_after,
            actual: live.revision(),
        });
    }
    if live.support() != latest.support_after {
        return Err(BeliefMutationPersistenceError::FinalSupportMismatch {
            claim_id,
            expected: latest.support_after.get(),
            actual: live.support().get(),
        });
    }
    if live.last_mutation_id() != Some(latest.id) {
        return Err(BeliefMutationPersistenceError::FinalMutationIdMismatch {
            claim_id,
            expected: latest.id,
            actual: live.last_mutation_id(),
        });
    }
    if live.last_updated_cycle() != latest.applied_at_cycle {
        return Err(BeliefMutationPersistenceError::FinalUpdateCycleMismatch {
            claim_id,
            expected: latest.applied_at_cycle,
            actual: live.last_updated_cycle(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationPersistenceError {
    DuplicateDeclaredClaim(ClaimId),
    StateInventoryCountMismatch {
        declared: usize,
        actual: usize,
    },
    MissingDeclaredState(ClaimId),
    MutationForUndeclaredState(ClaimId),
    DuplicateMutationId(BeliefMutationReceiptId),
    DuplicateSourceRevisionReceipt(BeliefRevisionReceiptId),
    DuplicateAuthorizationId(String),
    AuthorizationHistoryNotClosed {
        consumed_authorizations: usize,
        mutation_receipts: usize,
    },
    MutationIdsNotMonotonic {
        previous: BeliefMutationReceiptId,
        current: BeliefMutationReceiptId,
    },
    MutationPredatesAuthorization {
        mutation_id: BeliefMutationReceiptId,
        authorized_at_cycle: u64,
        applied_at_cycle: u64,
    },
    CapturePredatesMutation {
        mutation_id: BeliefMutationReceiptId,
        captured_at_cycle: u64,
        applied_at_cycle: u64,
    },
    CapturePredatesState {
        claim_id: ClaimId,
        captured_at_cycle: u64,
        state_cycle: u64,
    },
    UnmutatedStateRevisionNonZero {
        claim_id: ClaimId,
        revision: u64,
    },
    UnmutatedStateHasMutationId {
        claim_id: ClaimId,
        mutation_id: Option<BeliefMutationReceiptId>,
    },
    UnmutatedStateCycleMismatch {
        claim_id: ClaimId,
        initialized_at_cycle: u64,
        last_updated_cycle: u64,
    },
    RevisionChainBroken {
        claim_id: ClaimId,
        expected_before: u64,
        actual_before: u64,
        actual_after: u64,
    },
    SupportChainBroken {
        claim_id: ClaimId,
        expected: f32,
        actual: f32,
    },
    MutationDeltaMismatch {
        mutation_id: BeliefMutationReceiptId,
        expected_after: f32,
        actual_after: f32,
    },
    FinalRevisionMismatch {
        claim_id: ClaimId,
        expected: u64,
        actual: u64,
    },
    FinalSupportMismatch {
        claim_id: ClaimId,
        expected: f32,
        actual: f32,
    },
    FinalMutationIdMismatch {
        claim_id: ClaimId,
        expected: BeliefMutationReceiptId,
        actual: Option<BeliefMutationReceiptId>,
    },
    FinalUpdateCycleMismatch {
        claim_id: ClaimId,
        expected: u64,
        actual: u64,
    },
    ObservationPredatesCapsule {
        observed_at_cycle: u64,
        captured_at_cycle: u64,
    },
    LiveStateMismatch,
    LiveMutationHistoryMismatch,
    LiveAuthorizationCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for BeliefMutationPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief mutation persistence capsule invalid: {self:?}")
    }
}

impl Error for BeliefMutationPersistenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthority, BeliefMutationAuthorization,
        BeliefMutationAuthorizationDecision, BeliefMutationFirewall, BeliefRevisionHistory,
        BeliefRevisionPolicy, ClaimKind, EpistemicLedger, EpistemicRevisionProposal,
        EvidenceKind, EvidencePolarity,
    };

    struct Fixture {
        ledger: EpistemicLedger,
        prepared: crate::knowledge::PreparedBeliefMutation,
        store: EpistemicSupportStore,
        authorization: BeliefMutationAuthorization,
        claim: ClaimId,
    }

    fn fixture() -> Fixture {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();
        let authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        Fixture {
            ledger,
            prepared,
            store,
            authorization,
            claim,
        }
    }

    #[test]
    fn closed_public_history_captures_baseline_and_current_state() {
        let mut fixture = fixture();
        let mut authority = BeliefMutationAuthority::new();
        authority
            .apply(
                &fixture.ledger,
                &mut fixture.store,
                &fixture.prepared,
                &fixture.authorization,
                5,
            )
            .unwrap();

        let capsule = BeliefMutationPersistenceCapsuleV1::capture(
            &fixture.store,
            &[fixture.claim],
            6,
        )
        .unwrap();
        assert_eq!(capsule.version(), BeliefMutationPersistenceVersion::V1);
        assert_eq!(capsule.states().len(), 1);
        assert_eq!(capsule.mutations().len(), 1);
        assert_eq!(capsule.consumed_authorization_count(), 1);
        assert!((capsule.states()[0].baseline_support.get() - 0.50).abs() < 1e-6);
        assert!((capsule.states()[0].current_support.get() - 0.60).abs() < 1e-6);
        assert_eq!(capsule.states()[0].revision, 1);
        capsule.validate_live(&fixture.store, 7).unwrap();
    }

    #[test]
    fn capture_requires_complete_unique_state_inventory() {
        let fixture = fixture();
        assert_eq!(
            BeliefMutationPersistenceCapsuleV1::capture(
                &fixture.store,
                &[fixture.claim, fixture.claim],
                3,
            )
            .unwrap_err(),
            BeliefMutationPersistenceError::DuplicateDeclaredClaim(fixture.claim)
        );
        assert_eq!(
            BeliefMutationPersistenceCapsuleV1::capture(&fixture.store, &[], 3).unwrap_err(),
            BeliefMutationPersistenceError::StateInventoryCountMismatch {
                declared: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn live_validation_detects_store_inventory_drift() {
        let mut fixture = fixture();
        let capsule = BeliefMutationPersistenceCapsuleV1::capture(
            &fixture.store,
            &[fixture.claim],
            3,
        )
        .unwrap();
        let other = fixture
            .ledger
            .add_claim("other", ClaimKind::Descriptive, None, None, 4);
        fixture
            .store
            .register_claim(
                &fixture.ledger,
                other,
                BoundedWeight::new(0.40).unwrap(),
                4,
            )
            .unwrap();
        assert!(matches!(
            capsule.validate_live(&fixture.store, 5),
            Err(BeliefMutationPersistenceError::StateInventoryCountMismatch { .. })
        ));
    }

    #[test]
    fn raw_alternate_replay_authorization_makes_history_non_persistable() {
        let mut fixture = fixture();
        let mut authority = BeliefMutationAuthority::new();
        authority
            .apply(
                &fixture.ledger,
                &mut fixture.store,
                &fixture.prepared,
                &fixture.authorization,
                5,
            )
            .unwrap();

        // Exercise the lower-level writer from inside crate tests to prove why
        // EKM-029 forbids alternate replay authorization IDs publicly.
        let alternate = BeliefMutationAuthorization::new(
            "auth-hidden-replay",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            6,
            fixture.prepared.receipt(),
            fixture.store.state(fixture.claim).unwrap(),
        )
        .unwrap();
        let mut raw = BeliefMutationFirewall::new();
        raw.apply(
            &fixture.ledger,
            &mut fixture.store,
            fixture.prepared.receipt(),
            &alternate,
            7,
        )
        .unwrap();
        assert_eq!(fixture.store.history().len(), 1);
        assert_eq!(fixture.store.consumed_authorization_count(), 2);
        assert_eq!(
            BeliefMutationPersistenceCapsuleV1::capture(
                &fixture.store,
                &[fixture.claim],
                8,
            )
            .unwrap_err(),
            BeliefMutationPersistenceError::AuthorizationHistoryNotClosed {
                consumed_authorizations: 2,
                mutation_receipts: 1,
            }
        );
    }
}
