// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea::knowledge::belief_mutation_seal_persistence::{
    BeliefMutationEvidenceSealCapsuleV1, BeliefMutationSealPersistenceError,
    PersistedBeliefMutationEvidenceSealV1,
};
use symthaea::knowledge::{
    BeliefMutationAuthority, BeliefMutationAuthorization, BeliefMutationAuthorizationDecision,
    BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory, BeliefRevisionPolicy,
    BoundedWeight, ClaimKind, EpistemicLedger, EpistemicRevisionProposal, EpistemicSupportStore,
    EvidenceKind, EvidencePolarity,
};

fn applied_fixture() -> (
    EpistemicLedger,
    EpistemicSupportStore,
    PersistedBeliefMutationEvidenceSealV1,
    symthaea::knowledge::ClaimId,
    symthaea::knowledge::ProvenanceId,
) {
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
            Some("fixture".into()),
            Some("protocol-v1".into()),
        )
        .unwrap();

    let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
        .unwrap();
    let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
    let mut history = BeliefRevisionHistory::new();
    let mut authority = BeliefMutationAuthority::new();
    let prepared = authority
        .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
        .unwrap();

    let mut store = EpistemicSupportStore::new();
    store
        .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
        .unwrap();
    let authorization = BeliefMutationAuthorization::new(
        "auth-ekm056",
        "test-authority",
        BeliefMutationAuthorizationDecision::Approved,
        4,
        prepared.receipt(),
        store.state(claim).unwrap(),
    )
    .unwrap();
    let outcome = authority
        .apply(&ledger, &mut store, &prepared, &authorization, 5)
        .unwrap();
    assert!(outcome.verified());

    let seal = PersistedBeliefMutationEvidenceSealV1::capture(
        &prepared,
        outcome.mutation(),
    )
    .unwrap();
    (ledger, store, seal, claim, provenance)
}

#[test]
fn persisted_mutation_seal_survives_later_evidence_growth() {
    let (mut ledger, store, seal, claim, provenance) = applied_fixture();

    // This later evidence makes direct replay against the final ledger fail the
    // EKM-026 post-decision-evidence rule. The EKM-028 seal must nevertheless
    // retain the exact earlier census instead of being invalidated by normal
    // append-only knowledge growth.
    ledger
        .add_evidence(
            claim,
            EvidenceKind::Observation,
            EvidencePolarity::Contextualizes,
            provenance,
            6,
            Some("later context".into()),
            Some("follow-up".into()),
        )
        .unwrap();

    let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 7).unwrap();
    let capsule = BeliefMutationEvidenceSealCapsuleV1::capture(
        &[seal],
        &mutations,
        &ledger,
        7,
    )
    .unwrap();

    assert_eq!(capsule.records().len(), 1);
    assert_eq!(capsule.records()[0].seal().evidence().len(), 1);
    capsule.verify(&mutations, &ledger, 8).unwrap();
}

#[test]
fn complete_mutation_history_requires_complete_seal_history() {
    let (ledger, store, _seal, claim, _) = applied_fixture();
    let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 6).unwrap();

    assert_eq!(
        BeliefMutationEvidenceSealCapsuleV1::capture(&[], &mutations, &ledger, 6).unwrap_err(),
        BeliefMutationSealPersistenceError::MutationSealCountMismatch {
            mutations: 1,
            seals: 0,
        }
    );
}
