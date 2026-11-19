// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Cross-crate integration: verified raw Session-A events -> consolidation
// promotion receipts -> MACE Session-B decision -> Session-C correction.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_consolidation::{
    verify_and_release_promoted_experiences, ConsolidationPolicyV1,
    ConsolidationProposalV1, ConsolidationReceiptContextV1,
    ConsolidationReceiptProposalV1, ConsolidationReceiptVerifierV1,
    VerifiedConsolidationReceiptV1, ReceiptUseReject,
};
use broca_mace_ingress::{
    build_consolidation_evidence_pack, ingest_as_known, ingest_batch_as_known,
    Action, AsKnownQuery, AuthorityStatus, EvidenceVerifier, RawInteractionV1,
    SourceLifecycleStatus, SourceTrustStatus, VerifiedCausalAssessment,
    VerifiedLineage, VerifiedMeasurement, VerifiedOutcome, VerifiedRawEvent,
};
use broca_mace_policy::{
    decide, Objective, Policy, Query, Regime, Scope, SourceRootDigest,
};

const UTILITY_UNIT: &str = "normalized_utility";

fn hex_digest(seed: u32) -> String {
    format!("{seed:064x}")
}

fn root_digest(seed: u32) -> SourceRootDigest {
    let mut bytes = [0_u8; 32];
    bytes[..4].copy_from_slice(&seed.to_be_bytes());
    SourceRootDigest::from_bytes(bytes)
}

fn scope() -> Scope {
    Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    }
}

fn as_known(known_at_ms: i64) -> AsKnownQuery {
    AsKnownQuery {
        valid_at_ms: 180,
        known_at_ms,
    }
}

fn raw_event(event_id: u32, action: Action, utility: f64) -> RawInteractionV1 {
    RawInteractionV1 {
        event_id,
        session_id: "session-a".into(),
        task_id: format!("task-{event_id}"),
        payload_sha256: hex_digest(10_000 + event_id),
        raw_event_receipt_id: format!("raw-{event_id}"),
        scope: scope(),
        action,
        observed_utility: utility,
        utility_unit: UTILITY_UNIT.into(),
        valid_from_ms: 100,
        valid_until_ms: Some(300),
        occurred_at_ms: 170,
        prediction_registered_at_ms: Some(150),
        action_dispatched_at_ms: Some(180),
        outcome_observed_at_ms: 200,
        lineage_receipt_id: format!("lineage-{event_id}"),
        calibration_receipt_id: format!("calibration-{event_id}"),
        causal_assessment_receipt_id: format!("causal-{event_id}"),
        outcome_receipt_id: format!("outcome-{event_id}"),
    }
}

#[derive(Clone)]
struct FixtureVerifier {
    outcomes: BTreeMap<u32, f64>,
    retracted_events: BTreeSet<u32>,
    retracted_at_ms: i64,
}

impl EvidenceVerifier for FixtureVerifier {
    fn verify_raw_event(
        &self,
        receipt_id: &str,
        raw: &RawInteractionV1,
        _known_at_ms: i64,
    ) -> Option<VerifiedRawEvent> {
        if receipt_id != format!("raw-{}", raw.event_id) {
            return None;
        }
        Some(VerifiedRawEvent {
            event_id: raw.event_id,
            session_id: raw.session_id.clone(),
            task_id: raw.task_id.clone(),
            payload_sha256: raw.payload_sha256.clone(),
            canonical_event_hash: hex_digest(20_000 + raw.event_id),
            authority_status: AuthorityStatus::ReceiptAttested,
            authority_receipt_digest: hex_digest(30_000 + raw.event_id),
            valid_from_ms: raw.valid_from_ms,
            valid_until_ms: raw.valid_until_ms,
            occurred_at_ms: raw.occurred_at_ms,
            received_at_ms: 220,
            receipt_digest: hex_digest(40_000 + raw.event_id),
        })
    }

    fn verify_lineage(
        &self,
        receipt_id: &str,
        event_id: u32,
        _scope: Scope,
        _known_at_ms: i64,
    ) -> Option<VerifiedLineage> {
        if receipt_id != format!("lineage-{event_id}") {
            return None;
        }
        Some(VerifiedLineage {
            source_family: u16::try_from(event_id).ok()?,
            source_root_id: event_id,
            source_root_digest: root_digest(event_id),
            trust_status: SourceTrustStatus::TrustedForScope,
            trust_policy_digest: hex_digest(50_000 + event_id),
            acquisition_mode: "fixture_signed_release".into(),
            upstream_source_id: format!("upstream-{event_id}"),
            source_artifact_digest: hex_digest(60_000 + event_id),
            transformation_digest: hex_digest(70_000 + event_id),
            issuer_id: "fixture-independent-source-ledger".into(),
            receipt_digest: hex_digest(80_000 + event_id),
        })
    }

    fn verify_measurement(
        &self,
        receipt_id: &str,
        expected_unit: &str,
        _scope: Scope,
        _observed_at_ms: i64,
        _known_at_ms: i64,
    ) -> Option<VerifiedMeasurement> {
        let event_id = receipt_id.strip_prefix("calibration")?;
        if event_id.is_empty() || expected_unit != UTILITY_UNIT {
            return None;
        }
        Some(VerifiedMeasurement {
            absolute_error_bound: 0.01,
            unit: UTILITY_UNIT.into(),
            instrument_id: "fixture-utility-measurement".into(),
            method_id: "fixture-calibration-method-v1".into(),
            calibration_revision: "fixture-calibration-r1".into(),
            valid_from_ms: 100,
            valid_until_ms: Some(300),
            calibration_digest: hex_digest(90_000 + event_id.parse::<u32>().ok()?),
        })
    }

    fn verify_causal_assessment(
        &self,
        receipt_id: &str,
        event_id: u32,
        _scope: Scope,
        _known_at_ms: i64,
    ) -> Option<VerifiedCausalAssessment> {
        if receipt_id != format!("causal-{event_id}") {
            return None;
        }
        Some(VerifiedCausalAssessment {
            intervention_confirmed: true,
            prediction_pre_registered: true,
            confounders_controlled: true,
            delayed_horizon_complete: true,
            assessment_digest: hex_digest(100_000 + event_id),
        })
    }

    fn verify_outcome(
        &self,
        receipt_id: &str,
        event_id: u32,
        _observed_at_ms: i64,
        _known_at_ms: i64,
    ) -> Option<VerifiedOutcome> {
        if receipt_id != format!("outcome-{event_id}") {
            return None;
        }
        let observed_utility = *self.outcomes.get(&event_id)?;
        Some(VerifiedOutcome {
            reconciled: true,
            success: observed_utility >= 0.5,
            observed_utility,
            utility_unit: UTILITY_UNIT.into(),
            outcome_digest: hex_digest(110_000 + event_id),
        })
    }

    fn lifecycle_status_as_of(
        &self,
        event_id: u32,
        known_at_ms: i64,
    ) -> Option<SourceLifecycleStatus> {
        if self.retracted_events.contains(&event_id)
            && known_at_ms >= self.retracted_at_ms
        {
            Some(SourceLifecycleStatus::Retracted)
        } else {
            Some(SourceLifecycleStatus::Active)
        }
    }
}

struct FixtureConsolidationVerifier;

impl ConsolidationReceiptVerifierV1 for FixtureConsolidationVerifier {
    fn verify(
        &self,
        proposal: &ConsolidationReceiptProposalV1,
    ) -> Option<VerifiedConsolidationReceiptV1> {
        Some(VerifiedConsolidationReceiptV1 {
            proposal_digest: proposal.canonical_digest(),
            verification_receipt_digest: hex_digest(120_000),
            verifier_id: "fixture-independent-consolidation-verifier".into(),
            verifier_role: "independent_consolidation_qualifier".into(),
            verified_at_ms: proposal.as_known.known_at_ms,
            authority_scope: proposal.target_scope,
        })
    }
}

fn promotion_context(action: Action) -> ConsolidationReceiptContextV1 {
    ConsolidationReceiptContextV1 {
        consolidation_run_id: format!("session-a-consolidation-{}", action.name()),
        principal_id: "fixture-principal".into(),
        capability_receipt_digest: hex_digest(130_000),
        policy_digest: hex_digest(140_000),
        destination_abstraction_id: format!("procedure-{}", action.name()),
        output_artifact_digest: hex_digest(150_000),
        retention_policy_id: "retain-source-links-v1".into(),
    }
}

fn propose_and_release(
    action: Action,
    pack: &broca_mace_ingress::ConsolidationEvidencePackV1,
) -> Vec<broca_mace_policy::Experience> {
    let evidence = pack.summarize_action_for_scope(scope(), action);
    let decision = ConsolidationPolicyV1::default().decide_procedure_promotion(&evidence);
    let ConsolidationProposalV1::ProposeProcedure(proposal) = decision else {
        panic!("three qualified independent roots should propose {action:?}");
    };
    let receipt_proposal = ConsolidationReceiptProposalV1::from_procedure_promotion(
        &proposal,
        pack,
        promotion_context(action),
    )
    .expect("proposal should bind the exact admitted evidence");
    let (_, released) = verify_and_release_promoted_experiences(
        &receipt_proposal,
        pack,
        pack.as_known().known_at_ms,
        &FixtureConsolidationVerifier,
    )
    .expect("independent receipt should release source-linked experiences");
    assert_eq!(released.len(), 3);
    released
}

#[test]
fn raw_session_to_governed_consolidation_to_mace_decision_then_correction() {
    let events = vec![
        raw_event(1, Action::A, 0.91),
        raw_event(2, Action::A, 0.89),
        raw_event(3, Action::A, 0.90),
        raw_event(4, Action::B, 0.42),
        raw_event(5, Action::B, 0.40),
        raw_event(6, Action::B, 0.41),
    ];
    let verifier = FixtureVerifier {
        outcomes: BTreeMap::from([
            (1, 0.91),
            (2, 0.89),
            (3, 0.90),
            (4, 0.42),
            (5, 0.40),
            (6, 0.41),
        ]),
        retracted_events: BTreeSet::new(),
        retracted_at_ms: 260,
    };
    let session_a = as_known(250);
    let event_ids = [1, 2, 3, 4, 5, 6];

    let batch_a = ingest_batch_as_known(&events, session_a, &verifier)
        .expect("all Session-A evidence should pass receipt checks");
    let pack_a = build_consolidation_evidence_pack(&batch_a, &event_ids)
        .expect("consolidation pack should preserve source receipt lineage");
    assert_eq!(pack_a.trusted_event_count(), 6);
    assert_eq!(pack_a.source_root_count(), 6);

    let mut memory_b = propose_and_release(Action::A, &pack_a);
    memory_b.extend(propose_and_release(Action::B, &pack_a));
    let query_b = Query { scope: scope() };
    let decision_b = decide(Policy::CausalQualified, &memory_b, query_b);
    assert_eq!(decision_b.action, Some(Action::A));
    assert_eq!(decision_b.causal_claim, Some(Action::A));

    // Session C learns that two of the three Action-A outcomes were retracted.
    // The old consolidation receipt remains auditable but cannot be replayed
    // against a newer as-known snapshot.
    let corrected_verifier = FixtureVerifier {
        retracted_events: BTreeSet::from([1, 2]),
        ..verifier.clone()
    };
    assert_eq!(
        ingest_as_known(&events[0], as_known(270), &corrected_verifier).unwrap_err(),
        broca_mace_ingress::IngestReject::RetractedAsKnown
    );
    assert_eq!(
        ingest_as_known(&events[1], as_known(270), &corrected_verifier).unwrap_err(),
        broca_mace_ingress::IngestReject::RetractedAsKnown
    );

    let batch_c = ingest_batch_as_known(&events[2..], as_known(270), &corrected_verifier)
        .expect("unretracted evidence remains available at the new knowledge time");
    let pack_c = build_consolidation_evidence_pack(&batch_c, &[3, 4, 5, 6])
        .expect("corrected pack must bind only non-retracted observations");
    assert_eq!(pack_c.events().len(), 4);

    let stale_promotion = propose_and_release(Action::B, &pack_a);
    assert_eq!(stale_promotion.len(), 3);
    assert_eq!(
        verify_and_release_promoted_experiences(
            &ConsolidationReceiptProposalV1::from_procedure_promotion(
                &match ConsolidationPolicyV1::default()
                    .decide_procedure_promotion(
                        &pack_a.summarize_action_for_scope(scope(), Action::A)
                    )
                {
                    ConsolidationProposalV1::ProposeProcedure(value) => value,
                    _ => panic!("original Action-A evidence should promote"),
                },
                &pack_a,
                promotion_context(Action::A),
            )
            .unwrap(),
            &pack_c,
            270,
            &FixtureConsolidationVerifier,
        ),
        Err(ReceiptUseReject::KnowledgeSnapshotMismatch)
    );

    let corrected_memory: Vec<_> = pack_c
        .events()
        .iter()
        .map(broca_mace_ingress::ConsolidationEvidenceRefV1::experience)
        .collect();
    let corrected_decision = decide(Policy::CausalQualified, &corrected_memory, query_b);
    assert_eq!(corrected_decision.action, None);
    assert!(corrected_decision.evidence_ids.is_empty());
    assert!(decide(Policy::ActiveProbe, &corrected_memory, query_b)
        .probe
        .is_some());
}
