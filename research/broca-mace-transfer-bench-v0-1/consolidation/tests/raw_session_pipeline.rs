// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_consolidation::{
    verify_and_release_promoted_experiences, ConsolidationPolicyV1,
    ConsolidationProposalV1, ConsolidationReceiptContextV1,
    ConsolidationReceiptProposalV1, ConsolidationReceiptVerifierV1,
    HoldReason, VerifiedConsolidationReceiptV1,
};
use broca_mace_ingress::{
    build_consolidation_evidence_pack, ingest_as_known, ingest_batch_as_known, AsKnownQuery,
    AuthorityStatus, EvidenceVerifier, RawInteractionV1, SourceLifecycleStatus,
    SourceTrustStatus, VerifiedCausalAssessment, VerifiedLineage, VerifiedMeasurement,
    VerifiedOutcome, VerifiedRawEvent,
};
use broca_mace_policy::{decide, Action, Experience, Objective, Policy, Query, Regime, Scope};

#[derive(Default)]
fn fixture_causal_identification(
    seed: u32,
) -> broca_mace_policy::CausalIdentificationV1 {
    let mut design = [0_u8; 32];
    let mut estimand = [0_u8; 32];
    design[..4].copy_from_slice(&seed.to_be_bytes());
    estimand[..4].copy_from_slice(&seed.wrapping_add(1).to_be_bytes());
    broca_mace_policy::CausalIdentificationV1::new(
        broca_mace_policy::IdentificationMethodV1::RandomizedControlledTrial,
        broca_mace_policy::EvidenceDigest256::from_bytes(design),
        broca_mace_policy::EvidenceDigest256::from_bytes(estimand),
        5_000,
        5_000,
        50_000,
        true,
    )
}

struct FixtureVerifier {
    retracted_event_ids: BTreeSet<u32>,
    outcome_overrides: BTreeMap<u32, f64>,
}

impl EvidenceVerifier for FixtureVerifier {
    fn verify_raw_event(
        &self,
        _receipt_id: &str,
        raw: &RawInteractionV1,
        _known_at_ms: i64,
    ) -> Option<VerifiedRawEvent> {
        Some(VerifiedRawEvent {
            event_id: raw.event_id,
            session_id: raw.session_id.clone(),
            task_id: raw.task_id.clone(),
            payload_sha256: raw.payload_sha256.clone(),
            canonical_event_hash: "c".repeat(64),
            authority_status: AuthorityStatus::ReceiptAttested,
            authority_receipt_digest: "b".repeat(64),
            valid_from_ms: raw.valid_from_ms,
            valid_until_ms: raw.valid_until_ms,
            occurred_at_ms: raw.occurred_at_ms,
            received_at_ms: raw.occurred_at_ms.checked_add(50)?,
            receipt_digest: "a".repeat(64),
        })
    }

    fn verify_lineage(
        &self,
        _receipt_id: &str,
        event_id: u32,
        _scope: Scope,
        _known_at_ms: i64,
    ) -> Option<VerifiedLineage> {
        Some(VerifiedLineage {
            source_family: u16::try_from(event_id).ok()?,
            source_root_id: 10_000 + event_id,
            source_root_digest: fixture_root_digest(10_000 + event_id),
            trust_status: SourceTrustStatus::TrustedForScope,
            trust_policy_digest: "8".repeat(64),
            acquisition_mode: "fixture-ledger".into(),
            upstream_source_id: format!("fixture:source:{event_id}"),
            source_artifact_digest: "e".repeat(64),
            transformation_digest: "f".repeat(64),
            issuer_id: "did:key:fixture-source-ledger".into(),
            receipt_digest: "d".repeat(64),
        })
    }

    fn verify_measurement(
        &self,
        _receipt_id: &str,
        expected_unit: &str,
        _scope: Scope,
        _observed_at_ms: i64,
        _known_at_ms: i64,
    ) -> Option<VerifiedMeasurement> {
        Some(VerifiedMeasurement {
            absolute_error_bound: 0.02,
            unit: expected_unit.into(),
            instrument_id: "fixture-instrument-1".into(),
            method_id: "fixture-method-v1".into(),
            calibration_revision: "fixture-calibration-r1".into(),
            valid_from_ms: 100,
            valid_until_ms: Some(300),
            calibration_digest: "1".repeat(64),
        })
    }

    fn verify_causal_assessment(
        &self,
        _receipt_id: &str,
        _event_id: u32,
        _scope: Scope,
        _known_at_ms: i64,
    ) -> Option<VerifiedCausalAssessment> {
        Some(VerifiedCausalAssessment {
            intervention_confirmed: true,
            prediction_pre_registered: true,
            confounders_controlled: true,
            delayed_horizon_complete: true,
            identification: fixture_causal_identification(event_id),
            assessment_digest: "2".repeat(64),
        })
    }

    fn verify_outcome(
        &self,
        _receipt_id: &str,
        event_id: u32,
        _observed_at_ms: i64,
        _known_at_ms: i64,
    ) -> Option<VerifiedOutcome> {
        let observed_utility = self
            .outcome_overrides
            .get(&event_id)
            .copied()
            .or_else(|| match event_id {
                100 => Some(0.91),
                101 => Some(0.90),
                102 => Some(0.42),
                103 => Some(0.41),
                104 => Some(0.40),
                105 => Some(0.39),
                _ => None,
            })?;
        Some(VerifiedOutcome {
            reconciled: true,
            success: true,
            observed_utility,
            utility_unit: "normalized_utility".into(),
            outcome_digest: "3".repeat(64),
        })
    }

    fn lifecycle_status_as_of(
        &self,
        event_id: u32,
        known_at_ms: i64,
    ) -> Option<SourceLifecycleStatus> {
        if self.retracted_event_ids.contains(&event_id) && known_at_ms >= 260 {
            Some(SourceLifecycleStatus::Retracted)
        } else {
            Some(SourceLifecycleStatus::Active)
        }
    }
}

struct FixtureConsolidationVerifier {
    verified_at_ms: Option<i64>,
}

impl ConsolidationReceiptVerifierV1 for FixtureConsolidationVerifier {
    fn verify(
        &self,
        proposal: &ConsolidationReceiptProposalV1,
    ) -> Option<VerifiedConsolidationReceiptV1> {
        Some(VerifiedConsolidationReceiptV1 {
            proposal_digest: proposal.canonical_digest(),
            verification_receipt_digest: "d".repeat(64),
            verifier_id: "did:key:fixture-independent-verifier".into(),
            verifier_role: "independent_test_verifier".into(),
            verified_at_ms: self
                .verified_at_ms
                .unwrap_or(proposal.as_known.known_at_ms),
            authority_scope: proposal.target_scope,
        })
    }
}

fn receipt_context(action: Action) -> ConsolidationReceiptContextV1 {
    ConsolidationReceiptContextV1 {
        consolidation_run_id: format!("session-a-consolidation-{}", action.name()),
        principal_id: "did:key:fixture-consolidation-principal".into(),
        capability_receipt_digest: "a".repeat(64),
        policy_digest: "9".repeat(64),
        destination_abstraction_id: format!("procedure:quality:{}", action.name()),
        output_artifact_digest: if action == Action::A {
            "b".repeat(64)
        } else {
            "c".repeat(64)
        },
        retention_policy_id: "retention:source-linked:v1".into(),
    }
}

fn fixture_root_digest(seed: u32) -> broca_mace_policy::SourceRootDigest {
    let mut bytes = [0_u8; 32];
    bytes[..4].copy_from_slice(&seed.to_be_bytes());
    broca_mace_policy::SourceRootDigest::from_bytes(bytes)
}

fn target_scope() -> Scope {
    Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    }
}

fn raw_event(event_id: u32, action: Action, utility: f64) -> RawInteractionV1 {
    RawInteractionV1 {
        event_id,
        session_id: "session-a".into(),
        task_id: format!("task-{event_id}"),
        payload_sha256: format!("{event_id:064x}"),
        raw_event_receipt_id: format!("raw-receipt-{event_id}"),
        scope: target_scope(),
        action,
        observed_utility: utility,
        utility_unit: "normalized_utility".into(),
        valid_from_ms: 100,
        valid_until_ms: Some(300),
        occurred_at_ms: 170,
        prediction_registered_at_ms: Some(150),
        action_dispatched_at_ms: Some(170),
        outcome_observed_at_ms: 200,
        lineage_receipt_id: format!("lineage-{event_id}"),
        calibration_receipt_id: format!("calibration-{event_id}"),
        causal_assessment_receipt_id: format!("causal-{event_id}"),
        outcome_receipt_id: format!("outcome-{event_id}"),
    }
}

fn as_known(known_at_ms: i64) -> AsKnownQuery {
    as_known_at(180, known_at_ms)
}

fn as_known_at(valid_at_ms: i64, known_at_ms: i64) -> AsKnownQuery {
    AsKnownQuery {
        valid_at_ms,
        known_at_ms,
    }
}

fn raw_event_at(
    event_id: u32,
    session_id: &str,
    scope: Scope,
    action: Action,
    utility: f64,
    occurred_at_ms: u64,
) -> RawInteractionV1 {
    let mut event = raw_event(event_id, action, utility);
    event.session_id = session_id.to_owned();
    event.scope = scope;
    event.valid_from_ms = occurred_at_ms.saturating_sub(20) as i64;
    event.valid_until_ms = Some(300);
    event.occurred_at_ms = occurred_at_ms as i64;
    event.prediction_registered_at_ms =
        Some(occurred_at_ms.saturating_sub(40) as i64);
    event.action_dispatched_at_ms = Some(occurred_at_ms as i64);
    event.outcome_observed_at_ms = (occurred_at_ms + 20) as i64;
    event
}

#[test]
fn raw_session_a_governed_proposal_session_b_decision_session_c_retraction() {
    // Session A raw interactions: three independently rooted observations per action.
    let raw_events = vec![
        raw_event(100, Action::A, 0.91),
        raw_event(101, Action::A, 0.90),
        raw_event(102, Action::A, 0.89),
        raw_event(103, Action::B, 0.42),
        raw_event(104, Action::B, 0.41),
        raw_event(105, Action::B, 0.40),
    ];
    let event_ids = [100, 101, 102, 103, 104, 105];
    let verifier_a = FixtureVerifier::default();
    let batch_a = ingest_batch_as_known(&raw_events, as_known(250), &verifier_a)
        .expect("Session A events must pass receipt-gated ingress");
    let pack_a = build_consolidation_evidence_pack(&batch_a, &event_ids)
        .expect("Session A evidence pack must preserve all source receipts");

    let consolidation = ConsolidationPolicyV1::default();
    let support_a = pack_a.summarize_action_for_scope(target_scope(), Action::A);
    let support_b = pack_a.summarize_action_for_scope(target_scope(), Action::B);
    let proposal_a = consolidation.decide_procedure_promotion(&support_a);
    let proposal_b = consolidation.decide_procedure_promotion(&support_b);

    assert!(matches!(
        proposal_a,
        ConsolidationProposalV1::ProposeProcedure(ref proposal)
            if proposal.evidence_event_ids == vec![100, 101, 102]
                && proposal.independent_source_root_count == 3
    ));
    assert!(matches!(
        proposal_b,
        ConsolidationProposalV1::ProposeProcedure(ref proposal)
            if proposal.evidence_event_ids == vec![103, 104, 105]
                && proposal.independent_source_root_count == 3
    ));

    let ConsolidationProposalV1::ProposeProcedure(proposal_a) = proposal_a else {
        panic!("A-side promotion proposal required");
    };
    let ConsolidationProposalV1::ProposeProcedure(proposal_b) = proposal_b else {
        panic!("B-side promotion proposal required");
    };
    let receipt_proposal_a = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &proposal_a,
        &pack_a,
        receipt_context(Action::A),
    )
    .expect("A proposal must bind exact source revisions");
    let receipt_proposal_b = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &proposal_b,
        &pack_a,
        receipt_context(Action::B),
    )
    .expect("B proposal must bind exact source revisions");

    let attestor = FixtureConsolidationVerifier {
        verified_at_ms: None,
    };
    let (_receipt_a, mut memory_b) = verify_and_release_promoted_experiences(
        &receipt_proposal_a,
        &pack_a,
        250,
        &attestor,
    )
    .expect("independent verifier must attest A-side proposal");
    let (_receipt_b, memory_b_side) = verify_and_release_promoted_experiences(
        &receipt_proposal_b,
        &pack_a,
        250,
        &attestor,
    )
    .expect("independent verifier must attest B-side proposal");
    memory_b.extend(memory_b_side);
    let query_b = Query {
        scope: target_scope(),
    };
    assert_eq!(
        decide(Policy::CausalQualified, &memory_b, query_b).action,
        Some(Action::A)
    );

    // Session C learns that the entire A-side lineage was retracted.
    let verifier_c = FixtureVerifier {
        retracted_event_ids: [100, 101, 102].into_iter().collect(),
    };
    for event in &raw_events[..3] {
        assert!(matches!(
            ingest_as_known(event, as_known(270), &verifier_c),
            Err(broca_mace_ingress::IngestReject::RetractedAsKnown)
        ));
    }

    // The source ledger's as-known view contains only still-active raw events.
    let batch_c = ingest_batch_as_known(&raw_events[3..], as_known(270), &verifier_c)
        .expect("non-retracted Session A observations remain eligible");
    let pack_c = build_consolidation_evidence_pack(&batch_c, &[103, 104, 105])
        .expect("Session C pack preserves surviving B-side receipts");
    let support_a_c = pack_c.summarize_action_for_scope(target_scope(), Action::A);
    let support_b_c = pack_c.summarize_action_for_scope(target_scope(), Action::B);

    assert_eq!(
        consolidation.decide_procedure_promotion(&support_a_c),
        ConsolidationProposalV1::RetainProvisional(HoldReason::NoTargetScopeEvidence)
    );
    assert!(matches!(
        consolidation.decide_procedure_promotion(&support_b_c),
        ConsolidationProposalV1::ProposeProcedure(_)
    ));

    let ConsolidationProposalV1::ProposeProcedure(proposal_b_c) =
        consolidation.decide_procedure_promotion(&support_b_c)
    else {
        panic!("surviving B-side evidence must propose promotion");
    };
    let receipt_proposal_b_c = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &proposal_b_c,
        &pack_c,
        receipt_context(Action::B),
    )
    .expect("Session C proposal must bind surviving source revisions");
    let (_receipt_c, memory_c) = verify_and_release_promoted_experiences(
        &receipt_proposal_b_c,
        &pack_c,
        270,
        &FixtureConsolidationVerifier {
            verified_at_ms: None,
        },
    )
    .expect("independent verifier must attest Session C proposal");
    assert_eq!(decide(Policy::CausalQualified, &memory_c, query_b).action, None);
    assert!(decide(Policy::ActiveProbe, &memory_c, query_b)
        .probe
        .is_some());
}

#[test]
fn consolidation_receipt_rejects_source_substitution_and_late_attestation() {
    let raw_events = vec![
        raw_event(100, Action::A, 0.91),
        raw_event(101, Action::A, 0.90),
        raw_event(102, Action::A, 0.89),
    ];
    let batch = ingest_batch_as_known(
        &raw_events,
        as_known(250),
        &FixtureVerifier::default(),
    )
    .unwrap();
    let pack = build_consolidation_evidence_pack(&batch, &[100, 101, 102]).unwrap();
    let evidence = pack.summarize_action_for_scope(target_scope(), Action::A);
    let ConsolidationProposalV1::ProposeProcedure(procedure) =
        ConsolidationPolicyV1::default().decide_procedure_promotion(&evidence)
    else {
        panic!("evidence should propose promotion");
    };
    let proposal = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &procedure,
        &pack,
        receipt_context(Action::A),
    )
    .unwrap();
    let verifier = FixtureConsolidationVerifier {
        verified_at_ms: None,
    };
    let (verified, _) =
        verify_and_release_promoted_experiences(&proposal, &pack, 260, &verifier).unwrap();

    let mut substituted = proposal.clone();
    substituted.source_bindings[0].receipts.source_artifact_digest = "f".repeat(64);
    assert_eq!(
        verify_and_release_promoted_experiences(
            &substituted,
            &pack,
            260,
            &verifier,
        )
        .unwrap_err(),
        broca_mace_consolidation::ReceiptUseReject::SourceBindingsChanged
    );

    let late_verifier = FixtureConsolidationVerifier {
        verified_at_ms: Some(280),
    };
    assert_eq!(
        verify_and_release_promoted_experiences(
            &proposal,
            &pack,
            270,
            &late_verifier,
        )
        .unwrap_err(),
        broca_mace_consolidation::ReceiptUseReject::ReceiptNotKnownAtTransfer
    );
    assert_eq!(verified.proposal_digest, proposal.canonical_digest());
}

#[test]
fn consolidation_proposal_digest_binds_output_and_policy_identity() {
    let raw_events = vec![
        raw_event(100, Action::A, 0.91),
        raw_event(101, Action::A, 0.90),
        raw_event(102, Action::A, 0.89),
    ];
    let batch = ingest_batch_as_known(
        &raw_events,
        as_known(250),
        &FixtureVerifier::default(),
    )
    .unwrap();
    let pack = build_consolidation_evidence_pack(&batch, &[100, 101, 102]).unwrap();
    let evidence = pack.summarize_action_for_scope(target_scope(), Action::A);
    let ConsolidationProposalV1::ProposeProcedure(procedure) =
        ConsolidationPolicyV1::default().decide_procedure_promotion(&evidence)
    else {
        panic!("evidence should propose promotion");
    };
    let original = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &procedure,
        &pack,
        receipt_context(Action::A),
    )
    .unwrap();
    let mut changed = receipt_context(Action::A);
    changed.output_artifact_digest = "e".repeat(64);
    let changed = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &procedure,
        &pack,
        changed,
    )
    .unwrap();
    assert_ne!(original.canonical_digest(), changed.canonical_digest());
}


#[test]
fn promotion_receipt_cannot_be_reused_after_the_knowledge_snapshot_advances() {
    let raw_events = vec![
        raw_event(100, Action::A, 0.91),
        raw_event(101, Action::A, 0.90),
        raw_event(102, Action::A, 0.89),
    ];
    let batch = ingest_batch_as_known(
        &raw_events,
        as_known(250),
        &FixtureVerifier::default(),
    )
    .unwrap();
    let pack = build_consolidation_evidence_pack(&batch, &[100, 101, 102]).unwrap();
    let evidence = pack.summarize_action_for_scope(target_scope(), Action::A);
    let ConsolidationProposalV1::ProposeProcedure(procedure) =
        ConsolidationPolicyV1::default().decide_procedure_promotion(&evidence)
    else {
        panic!("evidence should propose promotion");
    };
    let proposal = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &procedure,
        &pack,
        receipt_context(Action::A),
    )
    .unwrap();
    let verifier = FixtureConsolidationVerifier {
        verified_at_ms: None,
    };
    assert_eq!(
        verify_and_release_promoted_experiences(
            &proposal,
            &pack,
            260,
            &verifier,
        )
        .unwrap_err(),
        broca_mace_consolidation::ReceiptUseReject::KnowledgeSnapshotMismatch
    );
}


#[test]
fn raw_session_a_promoted_effect_transports_through_common_mode_shifted_session_b() {
    let baseline = Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    };
    let shifted = Scope {
        regime: Regime::Shifted,
        objective: Objective::Quality,
    };

    let source_events = vec![
        raw_event_at(200, "session-a", baseline, Action::A, 0.80, 170),
        raw_event_at(201, "session-a", baseline, Action::A, 0.80, 170),
        raw_event_at(202, "session-a", baseline, Action::A, 0.80, 170),
        raw_event_at(203, "session-a", baseline, Action::B, 0.40, 170),
        raw_event_at(204, "session-a", baseline, Action::B, 0.40, 170),
        raw_event_at(205, "session-a", baseline, Action::B, 0.40, 170),
    ];
    let mut source_values = BTreeMap::new();
    source_values.extend((200..=202).map(|id| (id, 0.80)));
    source_values.extend((203..=205).map(|id| (id, 0.40)));
    let source_verifier = FixtureVerifier {
        outcome_overrides: source_values,
        ..FixtureVerifier::default()
    };
    let source_batch =
        ingest_batch_as_known(&source_events, as_known(250), &source_verifier)
            .expect("Session A raw outcomes must be receipt-qualified");
    let source_pack =
        build_consolidation_evidence_pack(&source_batch, &[200, 201, 202, 203, 204, 205])
            .expect("Session A evidence pack preserves source lineage");

    let consolidation = ConsolidationPolicyV1::default();
    let attestor = FixtureConsolidationVerifier {
        verified_at_ms: None,
    };
    let mut promoted_memory = Vec::new();
    for action in [Action::A, Action::B] {
        let summary = source_pack.summarize_action_for_scope(baseline, action);
        let ConsolidationProposalV1::ProposeProcedure(proposal) =
            consolidation.decide_procedure_promotion(&summary)
        else {
            panic!("Session A action {action:?} should meet the promotion gate");
        };
        let receipt_proposal = ConsolidationReceiptProposalV1::from_procedure_promotion(
            &proposal,
            &source_pack,
            receipt_context(action),
        )
        .expect("source-linked proposal must bind exact Session A receipts");
        let (_receipt, experiences) = verify_and_release_promoted_experiences(
            &receipt_proposal,
            &source_pack,
            250,
            &attestor,
        )
        .expect("independent verifier must attest the exact promotion");
        promoted_memory.extend(experiences);
    }

    // Session B has one receipt-verified witness per action. Both raw outcomes
    // share a +0.15 baseline shift, so the A-B treatment contrast remains 0.40.
    let target_events = vec![
        raw_event_at(300, "session-b", shifted, Action::A, 0.95, 240),
        raw_event_at(301, "session-b", shifted, Action::B, 0.55, 240),
    ];
    let target_verifier = FixtureVerifier {
        outcome_overrides: BTreeMap::from([(300, 0.95), (301, 0.55)]),
        ..FixtureVerifier::default()
    };
    let target_batch =
        ingest_batch_as_known(&target_events, as_known_at(260, 350), &target_verifier)
            .expect("Session B observations must pass as-known receipt checks");
    let mut candidate_memory = promoted_memory;
    candidate_memory.extend(target_batch.admitted().iter().map(|item| *item.experience()));

    let query = Query { scope: shifted };
    let exact = decide(Policy::CausalQualified, &candidate_memory, query);
    let transported = decide(Policy::InvariantTransport, &candidate_memory, query);
    assert_eq!(exact.action, None);
    assert_eq!(transported.action, Some(Action::A));
    assert!(transported.causal_claim.is_some());
    assert_eq!(transported.evidence_ids.len(), 8);
    assert!(transported.probe.is_none());
}

