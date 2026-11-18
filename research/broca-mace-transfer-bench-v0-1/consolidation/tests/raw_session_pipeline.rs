// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use broca_mace_consolidation::{
    ConsolidationPolicyV1, ConsolidationProposalV1, HoldReason,
};
use broca_mace_ingress::{
    build_consolidation_evidence_pack, ingest_as_known, ingest_batch_as_known, AsKnownQuery,
    AuthorityStatus, EvidenceVerifier, RawInteractionV1, SourceLifecycleStatus,
    SourceTrustStatus, VerifiedCausalAssessment, VerifiedLineage, VerifiedMeasurement,
    VerifiedOutcome, VerifiedRawEvent,
};
use broca_mace_policy::{decide, Action, Experience, Objective, Policy, Query, Regime, Scope};

#[derive(Default)]
struct FixtureVerifier {
    retracted_event_ids: BTreeSet<u32>,
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
            received_at_ms: 220,
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
            assessment_digest: "2".repeat(64),
        })
    }

    fn verify_outcome(
        &self,
        _receipt_id: &str,
        _event_id: u32,
        _observed_at_ms: i64,
        _known_at_ms: i64,
    ) -> Option<VerifiedOutcome> {
        Some(VerifiedOutcome {
            reconciled: true,
            success: true,
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
    AsKnownQuery {
        valid_at_ms: 180,
        known_at_ms,
    }
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

    // Session B consumes the pack projection, not the original raw/admitted batch.
    let memory_b: Vec<Experience> = pack_a
        .events()
        .iter()
        .map(|event| event.experience())
        .collect();
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

    let memory_c: Vec<Experience> = pack_c
        .events()
        .iter()
        .map(|event| event.experience())
        .collect();
    assert_eq!(decide(Policy::CausalQualified, &memory_c, query_b).action, None);
    assert!(decide(Policy::ActiveProbe, &memory_c, query_b)
        .probe
        .is_some());
}
