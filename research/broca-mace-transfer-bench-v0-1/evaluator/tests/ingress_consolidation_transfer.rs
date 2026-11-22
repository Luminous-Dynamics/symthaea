// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Cross-crate integration: verified raw Session-A events -> consolidation
// promotion receipts -> MACE Session-B decision -> Session-C correction.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_consolidation::{
    verify_and_release_promoted_experiences, ConsolidationPolicyV1,
    ConsolidationProposalV1, ConsolidationReceiptContextV1, HoldReason,
    ConsolidationReceiptProposalV1, ConsolidationReceiptVerifierV1,
    VerifiedConsolidationReceiptV1, ReceiptUseReject,
};
use broca_mace_ingress::{
    build_consolidation_evidence_pack, ingest_as_known, ingest_batch_as_known,
    AsKnownQuery, AuthorityStatus, EvidenceVerifier, RawInteractionV1,
    SourceLifecycleStatus, SourceTrustStatus, VerifiedCausalAssessment,
    VerifiedLineage, VerifiedMeasurement, VerifiedOutcome, VerifiedRawEvent,
};
use broca_mace_policy::{
    decide, Action, Objective, Policy, Query, Regime, Scope, SourceRootDigest,
};
use broca_mace_consolidation::memory_budget::{
    fit_model_memory_budget, qualify_model_memory_budget,
    select_for_qualified_model_budget, MemoryDoseTrialReceiptV1,
    MemoryDoseTrialVerifierV1, MemoryDoseTrialV1, ModelBudgetSurfacingReject,
};
use broca_mace_consolidation::reobservation::{
    enumerate_memory_representation_options, plan_memory_representation,
    verify_reobservation_profile, MemoryObjectKindV1, ReobservationAvailabilityV1,
    ReobservationPolicyV1, ReobservationProposalV1, ReobservationReceiptV1,
    ReobservationVerifierV1, ReobservationDispositionV1, MemoryAuthorityScopeV1,
    MemoryPurposeV1,
};
use broca_mace_consolidation::portfolio::{
    optimize_memory_portfolio_exact, optimize_memory_portfolio_pairwise,
    verify_portfolio_interactions, MemoryPortfolioBudgetV1, MemoryPortfolioOptionV1,
    PairwiseInteractionEstimateV1, PortfolioInteractionCalibrationReceiptV1,
    PortfolioInteractionContextV1, PortfolioInteractionProposalV1,
    PortfolioInteractionVerifierV1,
};
use sha2::{Digest, Sha256};
use broca_mace_consolidation::salience::{
    future_use_prediction_evidence_digest, verify_future_use_prediction,
    FutureUsePredictionV1, MemoryAccessTelemetryV1, MemorySurfacingCandidateV1,
    MemorySurfacingModeV1, SalienceCalibrationReceiptV1, SaliencePredictionVerifierV1,
};

fn fixture_semantic_estimand() -> broca_mace_policy::SemanticEstimandV1 {
    broca_mace_policy::SemanticEstimandV1::new(
        1,
        broca_mace_policy::EvidenceDigest256::from_bytes([1; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([2; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([3; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([4; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([5; 32]),
        30_000,
        broca_mace_policy::EvidenceDigest256::from_bytes([6; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([7; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([8; 32]),
    )
}

fn fixture_causal_identification(
    seed: u32,
) -> broca_mace_policy::CausalIdentificationV1 {
    let mut design = [0_u8; 32];
    design[..4].copy_from_slice(&seed.to_be_bytes());
    broca_mace_policy::CausalIdentificationV1::new(
        broca_mace_policy::IdentificationMethodV1::RandomizedControlledTrial,
        broca_mace_policy::EvidenceDigest256::from_bytes(design),
        fixture_semantic_estimand().canonical_digest(),
        5_000,
        5_000,
        50_000,
        true,
    )
    .with_semantic_estimand(fixture_semantic_estimand())
}

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
        let event_id = receipt_id.strip_prefix("calibration-")?;
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
            identification: fixture_causal_identification(event_id),
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
            success: true,
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

const MEMORY_DOSE_BUDGETS: [u32; 4] = [0, 128, 512, 1_024];
const DOSE_AS_KNOWN_AT_MS: i64 = 250;

struct FixtureMemoryDoseVerifier;

impl MemoryDoseTrialVerifierV1 for FixtureMemoryDoseVerifier {
    fn verify_trial(
        &self,
        trial: &MemoryDoseTrialV1,
        as_known_at_ms: i64,
    ) -> Option<MemoryDoseTrialReceiptV1> {
        let mut receipt = MemoryDoseTrialReceiptV1 {
            trial_sha256: trial.canonical_digest(),
            episode_sha256: trial.episode_sha256.clone(),
            environment_sha256: trial.environment_sha256.clone(),
            evaluation_protocol_sha256: trial.evaluation_protocol_sha256.clone(),
            target_model_sha256: trial.target_model_sha256.clone(),
            memory_pipeline_sha256: trial.memory_pipeline_sha256.clone(),
            verified_at_ms: as_known_at_ms.saturating_sub(1),
            valid_until_ms: as_known_at_ms.saturating_add(10_000),
            verifier_id: "fixture-independent-dose-verifier".into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.canonical_digest();
        Some(receipt)
    }
}

fn memory_dose_rows(
    target_model_sha256: &str,
    memory_pipeline_sha256: &str,
    cluster_start: u64,
    event_start: u32,
    family_start: u32,
    root_start: u32,
    uplifts: [i64; 4],
) -> Vec<MemoryDoseTrialV1> {
    let mut rows = Vec::with_capacity(48 * MEMORY_DOSE_BUDGETS.len());
    for cluster_offset in 0..48_u64 {
        let cluster_id = cluster_start + cluster_offset + 1;
        let task_family_id = family_start + cluster_offset as u32;
        let source_root = root_digest(root_start + cluster_offset as u32 + 1);
        let episode_sha256 = hex_digest(200_000 + cluster_id as u32);
        let baseline = 500_000 + (cluster_offset % 5) as i64 * 500;
        for (arm, context_budget_tokens) in MEMORY_DOSE_BUDGETS.iter().enumerate() {
            rows.push(MemoryDoseTrialV1 {
                event_id: event_start
                    + (cluster_offset as u32) * MEMORY_DOSE_BUDGETS.len() as u32
                    + arm as u32
                    + 1,
                cluster_id,
                task_family_id,
                source_root,
                episode_sha256: episode_sha256.clone(),
                environment_sha256: hex_digest(300_001),
                evaluation_protocol_sha256: hex_digest(300_002),
                target_model_sha256: target_model_sha256.to_owned(),
                memory_pipeline_sha256: memory_pipeline_sha256.to_owned(),
                context_budget_tokens: *context_budget_tokens,
                net_utility_micros: baseline + uplifts[arm],
            });
        }
    }
    rows
}

fn memory_authority_scope() -> MemoryAuthorityScopeV1 {
    MemoryAuthorityScopeV1 {
        principal_sha256: hex_digest(510_006),
        capability_sha256: hex_digest(510_007),
        audience_sha256: hex_digest(510_008),
        resource_sha256: hex_digest(510_009),
        policy_revision_sha256: hex_digest(510_010),
        purpose: MemoryPurposeV1::EnvironmentState,
    }
}

struct FixtureReobservationVerifier;

impl ReobservationVerifierV1 for FixtureReobservationVerifier {
    fn verify_profile(
        &self,
        proposal: &ReobservationProposalV1,
        _: Scope,
        authority_scope: &MemoryAuthorityScopeV1,
        as_known: AsKnownQuery,
    ) -> Option<ReobservationReceiptV1> {
        if &proposal.authority_scope != authority_scope {
            return None;
        }
        let mut receipt = ReobservationReceiptV1 {
            proposal_sha256: proposal.canonical_digest(),
            verified_at_ms: as_known.known_at_ms,
            valid_until_ms: as_known.known_at_ms + 2_000,
            verifier_id: "fixture-reobservation-verifier".into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.canonical_digest();
        Some(receipt)
    }
}

struct FixtureSalienceVerifier;

impl SaliencePredictionVerifierV1 for FixtureSalienceVerifier {
    fn verify_prediction(
        &self,
        prediction: &FutureUsePredictionV1,
        target_scope: Scope,
        as_known: AsKnownQuery,
    ) -> Option<SalienceCalibrationReceiptV1> {
        let mut receipt = SalienceCalibrationReceiptV1 {
            prediction_sha256: prediction.prediction_sha256.clone(),
            evidence_sha256: prediction.evidence_sha256.clone(),
            target_model_sha256: prediction.target_model_sha256.clone(),
            target_scope,
            as_known,
            model_id: prediction.model_id.clone(),
            model_version: prediction.model_version.clone(),
            model_config_sha256: prediction.model_config_sha256.clone(),
            training_dataset_sha256: hex_digest(310_001),
            calibration_dataset_sha256: hex_digest(310_002),
            heldout_evaluation_sha256: hex_digest(310_003),
            task_family_split_sha256: hex_digest(310_004),
            source_root_split_sha256: hex_digest(310_005),
            training_cluster_count: 32,
            calibration_cluster_count: 16,
            heldout_cluster_count: 16,
            brier_score_micros: 100_000,
            base_rate_brier_score_micros: 200_000,
            expected_calibration_error_micros: 20_000,
            worst_group_calibration_error_micros: 30_000,
            task_families_disjoint: true,
            source_roots_disjoint: true,
            verified_at_ms: as_known.known_at_ms.saturating_sub(1),
            valid_until_ms: as_known.known_at_ms.saturating_add(10_000),
            verifier_id: "fixture-independent-salience-verifier".into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.canonical_digest();
        Some(receipt)
    }
}

fn verified_salience_candidate(
    evidence: &broca_mace_ingress::ConsolidationEvidenceRefV1,
    target_model_sha256: &str,
    known_at: AsKnownQuery,
) -> MemorySurfacingCandidateV1 {
    let ingress_as_known = AsKnownQuery {
        valid_at_ms: known_at.valid_at_ms,
        known_at_ms: known_at.known_at_ms,
    };
    let prediction = FutureUsePredictionV1::new(
        evidence.event_id(),
        target_model_sha256,
        evidence.scope(),
        known_at.known_at_ms.saturating_sub(5),
        known_at.known_at_ms.saturating_add(500),
        9_500,
        200_000,
        100,
        "fixture-salience-model",
        "1",
        hex_digest(320_001),
        hex_digest(320_002),
        future_use_prediction_evidence_digest(evidence),
    );
    let verified_prediction = verify_future_use_prediction(
        prediction,
        evidence.scope(),
        known_at,
        &FixtureSalienceVerifier,
    )
    .expect("fixture forecast must pass independent calibration");
    MemorySurfacingCandidateV1::from_verified_evidence(
        evidence,
        MemoryAccessTelemetryV1 {
            context_token_cost: 64,
            last_accessed_at_ms: known_at.known_at_ms.saturating_sub(20),
            historical_access_count: 0,
        },
        Some(verified_prediction),
        ingress_as_known,
    )
    .expect("candidate must retain the admitted evidence receipt")
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


fn build_portfolio_interaction_option(
    evidence: &broca_mace_ingress::ConsolidationEvidenceRefV1,
    target_model: &str,
    expected_gain_micros: u32,
) -> MemoryPortfolioOptionV1 {
    let known = AsKnownQuery {
        valid_at_ms: 180,
        known_at_ms: 300,
    };
    let prediction = FutureUsePredictionV1::new(
        evidence.event_id(),
        target_model,
        evidence.scope(),
        295,
        1_200,
        10_000,
        expected_gain_micros,
        0,
        "portfolio-interaction-salience",
        "v1",
        hex_digest(810_001),
        hex_digest(810_002),
        future_use_prediction_evidence_digest(evidence),
    );
    let verified_prediction = verify_future_use_prediction(
        prediction,
        evidence.scope(),
        known,
        &FixtureSalienceVerifier,
    )
    .expect("future-use forecast must be independently calibrated");

    let authority = memory_authority_scope();
    let profile_proposal = ReobservationProposalV1 {
        event_id: evidence.event_id(),
        evidence_sha256: future_use_prediction_evidence_digest(evidence),
        scope: evidence.scope(),
        authority_scope: authority.clone(),
        as_known: known,
        object_kind: MemoryObjectKindV1::CurrentState,
        availability: ReobservationAvailabilityV1::Available {
            available_at_ms: 300,
            locator_sha256: hex_digest(820_000 + evidence.event_id()),
        },
        availability_success_probability_bps: 9_900,
        reobservation_failure_penalty_micros: 20_000,
        reobservation_latency_ms: 20,
        reobservation_cost_micros: 5_000,
        snapshot_context_tokens: 100,
        snapshot_valid_until_ms: Some(1_200),
        source_profile_sha256: hex_digest(830_000 + evidence.event_id()),
    };
    let verified_profile = verify_reobservation_profile(
        profile_proposal,
        evidence.scope(),
        &authority,
        known,
        &FixtureReobservationVerifier,
    )
    .expect("source profile must be independently checked");

    let policy = ReobservationPolicyV1 {
        as_known: known,
        authority_scope: authority,
        decision_at_ms: 300,
        decision_deadline_ms: 800,
        context_budget_tokens: 512,
        context_cost_per_token_micros: 100,
        locator_context_tokens: 8,
        latency_cost_per_ms_micros: 10,
        maximum_reobservation_cost_micros: 20_000,
        minimum_net_value_micros: 0,
    };
    let alternatives = enumerate_memory_representation_options(
        &verified_prediction,
        &verified_profile,
        policy.clone(),
    )
    .expect("representation alternatives must be re-derived");
    let plan = alternatives
        .iter()
        .find(|option| option.disposition() == ReobservationDispositionV1::StoreLocator)
        .expect("locator representation must be available");

    MemoryPortfolioOptionV1::from_verified_plan(
        plan,
        &verified_prediction,
        &verified_profile,
        &policy,
    )
    .expect("portfolio option must be re-derived from verified inputs")
}

fn interaction_option_set_digest(options: &[MemoryPortfolioOptionV1]) -> String {
    let mut digests: Vec<String> = options
        .iter()
        .map(|option| option.option_sha256().to_owned())
        .collect();
    digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"mace:portfolio-interaction-option-set:v1\0");
    for field in digests {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct FixturePortfolioInteractionVerifier {
    receipt: PortfolioInteractionCalibrationReceiptV1,
}

impl PortfolioInteractionVerifierV1 for FixturePortfolioInteractionVerifier {
    fn verify(
        &self,
        _proposal: &PortfolioInteractionProposalV1,
    ) -> Option<PortfolioInteractionCalibrationReceiptV1> {
        Some(self.receipt.clone())
    }
}

fn fixture_interaction_calibration_receipt(
    proposal: &PortfolioInteractionProposalV1,
) -> PortfolioInteractionCalibrationReceiptV1 {
    let mut receipt = PortfolioInteractionCalibrationReceiptV1 {
        proposal_sha256: proposal.proposal_sha256.clone(),
        target_model_sha256: proposal.target_model_sha256.clone(),
        baseline_policy_sha256: proposal.baseline_policy_sha256.clone(),
        task_protocol_sha256: proposal.task_protocol_sha256.clone(),
        interaction_model_sha256: proposal.interaction_model_sha256.clone(),
        training_dataset_sha256: proposal.training_dataset_sha256.clone(),
        calibration_dataset_sha256: proposal.calibration_dataset_sha256.clone(),
        heldout_evaluation_sha256: proposal.heldout_evaluation_sha256.clone(),
        task_family_split_sha256: proposal.task_family_split_sha256.clone(),
        source_root_split_sha256: proposal.source_root_split_sha256.clone(),
        training_pair_count: 512,
        calibration_pair_count: 128,
        heldout_pair_count: 128,
        independent_seed_count: 5,
        task_families_disjoint: true,
        source_roots_disjoint: true,
        calibration_mae_micros: 80,
        base_rate_mae_micros: 120,
        heldout_mae_micros: 85,
        heldout_base_rate_mae_micros: 130,
        heldout_mae_improvement_lower_95_micros: 12,
        heldout_multi_event_sample_count: 128,
        heldout_higher_order_residual_micros: 2_500,
        maximum_allowed_higher_order_residual_micros: 5_000,
        verified_at_ms: 250,
        valid_until_ms: 500,
        verifier_id: "fixture-only-interaction-verifier".into(),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt.canonical_digest();
    receipt
}

fn evaluator_hidden_portfolio_utility(selected_event_ids: &[u32]) -> i64 {
    let singleton_value: i64 = selected_event_ids
        .iter()
        .map(|event_id| match event_id {
            801 => 94_000,
            802 | 803 => 74_000,
            _ => panic!("hidden evaluator does not know this event"),
        })
        .sum();
    if selected_event_ids.contains(&802) && selected_event_ids.contains(&803) {
        singleton_value + 120_000
    } else {
        singleton_value
    }
}

#[test]
fn qualified_memory_dose_and_salience_compose_over_verified_session_a_evidence() {
    let events = vec![
        raw_event(51, Action::A, 0.91),
        raw_event(52, Action::A, 0.89),
        raw_event(53, Action::A, 0.90),
        raw_event(54, Action::B, 0.42),
        raw_event(55, Action::B, 0.40),
        raw_event(56, Action::B, 0.41),
    ];
    let verifier = FixtureVerifier {
        outcomes: BTreeMap::from([
            (51, 0.91),
            (52, 0.89),
            (53, 0.90),
            (54, 0.42),
            (55, 0.40),
            (56, 0.41),
        ]),
        retracted_events: BTreeSet::new(),
        retracted_at_ms: 260,
    };
    let known = as_known(DOSE_AS_KNOWN_AT_MS);
    let batch = ingest_batch_as_known(&events, known, &verifier)
        .expect("Session-A events must pass raw-ingress receipts");
    let pack = build_consolidation_evidence_pack(&batch, &[51, 52, 53, 54, 55, 56])
        .expect("evidence pack must preserve verified source lineage");
    let admitted = pack.events().first().expect("admitted evidence reference");

    let target_model = hex_digest(400_001);
    let memory_pipeline = hex_digest(400_002);
    let calibration_rows = memory_dose_rows(
        &target_model,
        &memory_pipeline,
        0,
        500_000,
        10_000,
        20_000,
        [0, 10_000, 30_000, 5_000],
    );
    let heldout_rows = memory_dose_rows(
        &target_model,
        &memory_pipeline,
        10_000,
        600_000,
        30_000,
        40_000,
        [0, 9_000, 29_000, 4_000],
    );
    let profile = fit_model_memory_budget(
        &calibration_rows,
        &MEMORY_DOSE_BUDGETS,
        DOSE_AS_KNOWN_AT_MS,
        &FixtureMemoryDoseVerifier,
    )
    .expect("dose profile should fit independent calibration clusters");
    assert_eq!(profile.selected_context_budget_tokens, 512);
    let qualified_budget = qualify_model_memory_budget(
        &profile,
        &calibration_rows,
        &heldout_rows,
        DOSE_AS_KNOWN_AT_MS,
        &FixtureMemoryDoseVerifier,
    )
    .expect("512-token arm should beat zero-memory on held-out clusters");
    assert_eq!(qualified_budget.context_budget_tokens(), 512);
    assert!(qualified_budget.heldout_ci95_lower_micros() > 0);

    let known_for_policy = AsKnownQuery {
        valid_at_ms: known.valid_at_ms,
        known_at_ms: known.known_at_ms,
    };
    let candidate = verified_salience_candidate(admitted, &target_model, known_for_policy);
    let expected_evidence_digest = candidate.evidence_sha256().to_owned();
    let plan = select_for_qualified_model_budget(
        &[candidate],
        MemorySurfacingModeV1::PredictedFutureValue,
        scope(),
        known_for_policy,
        &target_model,
        &qualified_budget,
    )
    .expect("matching model budget and calibrated salience should compose");
    assert_eq!(plan.plan.context_budget_tokens, 512);
    assert_eq!(plan.plan.context_tokens_used, 64);
    assert_eq!(plan.plan.surfaced.len(), 1);
    assert_eq!(plan.plan.surfaced[0].event_id, admitted.event_id());
    assert_eq!(plan.plan.surfaced[0].evidence_sha256, expected_evidence_digest);
    assert_eq!(plan.target_model_sha256, target_model);

    let other_model = hex_digest(400_003);
    let mismatched = verified_salience_candidate(admitted, &other_model, known_for_policy);
    assert!(matches!(
        select_for_qualified_model_budget(
            &[mismatched],
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            known_for_policy,
            &target_model,
            &qualified_budget,
        ),
        Err(ModelBudgetSurfacingReject::SalienceTargetModelMismatch)
    ));
}

#[test]
fn receipt_qualified_session_a_evidence_selects_reobservation_aware_memory() {
    let events = vec![
        raw_event(81, Action::A, 0.91),
        raw_event(82, Action::A, 0.89),
        raw_event(83, Action::A, 0.90),
        raw_event(84, Action::B, 0.42),
        raw_event(85, Action::B, 0.40),
        raw_event(86, Action::B, 0.41),
    ];
    let verifier = FixtureVerifier {
        outcomes: BTreeMap::from([
            (81, 0.91),
            (82, 0.89),
            (83, 0.90),
            (84, 0.42),
            (85, 0.40),
            (86, 0.41),
        ]),
        retracted_events: BTreeSet::new(),
        retracted_at_ms: 800,
    };
    let known = as_known(300);
    let batch = ingest_batch_as_known(&events, known, &verifier)
        .expect("Session-A evidence must pass independent ingress receipts");
    let pack = build_consolidation_evidence_pack(&batch, &[81, 82, 83, 84, 85, 86])
        .expect("evidence pack must preserve source lineage");
    let evidence = pack
        .events()
        .iter()
        .find(|item| item.event_id() == 81)
        .expect("source evidence exists");

    let target_model = hex_digest(510_001);
    let evidence_digest = future_use_prediction_evidence_digest(evidence);
    let prediction = FutureUsePredictionV1::new(
        evidence.event_id(),
        target_model,
        scope(),
        290,
        1_000,
        9_000,
        50_000,
        1_000,
        "future-use-model",
        "v1",
        hex_digest(510_002),
        hex_digest(510_003),
        evidence_digest.clone(),
    );
    let policy_as_known = AsKnownQuery {
        valid_at_ms: known.valid_at_ms,
        known_at_ms: known.known_at_ms,
    };
    let verified_prediction = verify_future_use_prediction(
        prediction,
        scope(),
        policy_as_known,
        &FixtureSalienceVerifier,
    )
    .expect("future-use prediction requires independent calibration");

    let proposal = ReobservationProposalV1 {
        event_id: evidence.event_id(),
        evidence_sha256: evidence_digest,
        scope: evidence.scope(),
        authority_scope: memory_authority_scope(),
        as_known: known,
        object_kind: MemoryObjectKindV1::CurrentState,
        availability: ReobservationAvailabilityV1::Available {
            available_at_ms: 300,
            locator_sha256: hex_digest(510_004),
        },
        availability_success_probability_bps: 9_900,
        reobservation_failure_penalty_micros: 20_000,
        reobservation_latency_ms: 20,
        reobservation_cost_micros: 5_000,
        snapshot_context_tokens: 100,
        snapshot_valid_until_ms: Some(1_200),
        source_profile_sha256: hex_digest(510_005),
    };
    let verified_profile = verify_reobservation_profile(
        proposal,
        scope(),
        &memory_authority_scope(),
        known,
        &FixtureReobservationVerifier,
    )
    .expect("source profile must be independently verified");

    let plan = plan_memory_representation(
        &verified_prediction,
        &verified_profile,
        ReobservationPolicyV1 {
            as_known: known,
            authority_scope: memory_authority_scope(),
            decision_at_ms: known.known_at_ms,
            decision_deadline_ms: 800,
            context_budget_tokens: 512,
            context_cost_per_token_micros: 100,
            locator_context_tokens: 8,
            latency_cost_per_ms_micros: 10,
            maximum_reobservation_cost_micros: 20_000,
            minimum_net_value_micros: 0,
        },
    )
    .expect("verified evidence and profile should produce a representation plan");

    assert_eq!(plan.event_id, 81);
    assert_eq!(plan.disposition, ReobservationDispositionV1::StoreLocator);
    assert_eq!(plan.context_tokens_reserved, 8);
    assert!(plan.expected_failure_loss_micros > 0);
    assert!(plan.locator_sha256.is_some());
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

    let corrected_a_promotion = ConsolidationPolicyV1::default()
        .decide_procedure_promotion(
            &pack_c.summarize_action_for_scope(scope(), Action::A),
        );
    assert_eq!(
        corrected_a_promotion,
        ConsolidationProposalV1::RetainProvisional(
            HoldReason::InsufficientQualifiedEvents
        )
    );

    let _verified_action_b = propose_and_release(Action::B, &pack_a);
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


#[test]
fn interaction_aware_portfolio_beats_additive_selection_on_hidden_joint_utility() {
    let events = vec![
        raw_event(801, Action::A, 0.91),
        raw_event(802, Action::A, 0.89),
        raw_event(803, Action::A, 0.90),
    ];
    let verifier = FixtureVerifier {
        outcomes: BTreeMap::from([(801, 0.91), (802, 0.89), (803, 0.90)]),
        retracted_events: BTreeSet::new(),
        retracted_at_ms: 900,
    };
    let batch = ingest_batch_as_known(&events, as_known(300), &verifier)
        .expect("raw Session-A events must pass receipt-gated ingress");
    let pack = build_consolidation_evidence_pack(&batch, &[801, 802, 803])
        .expect("admitted events must retain source evidence");
    let target_model = hex_digest(840_001);
    let gains = BTreeMap::from([(801, 100_000), (802, 80_000), (803, 80_000)]);
    let options: Vec<_> = pack
        .events()
        .iter()
        .map(|evidence| {
            build_portfolio_interaction_option(
                evidence,
                &target_model,
                *gains.get(&evidence.event_id()).unwrap(),
            )
        })
        .collect();

    let budget = MemoryPortfolioBudgetV1 {
        context_tokens: 16,
        reobservation_calls: 2,
        latency_work_ms: 100,
        estimated_cost_micros: 100_000,
        expected_failure_loss_micros: 10_000,
    };
    let additive = optimize_memory_portfolio_exact(&options, budget)
        .expect("additive exact baseline must produce a feasible plan");
    let mut pairs = Vec::new();
    for left in 0..options.len() {
        for right in (left + 1)..options.len() {
            let left_option = &options[left];
            let right_option = &options[right];
            let (left_sha, right_sha) = if left_option.option_sha256()
                < right_option.option_sha256()
            {
                (
                    left_option.option_sha256().to_owned(),
                    right_option.option_sha256().to_owned(),
                )
            } else {
                (
                    right_option.option_sha256().to_owned(),
                    left_option.option_sha256().to_owned(),
                )
            };
            let ids = (
                left_option.event_id().min(right_option.event_id()),
                left_option.event_id().max(right_option.event_id()),
            );
            let interaction = if ids == (802, 803) { 120_000 } else { 0 };
            pairs.push(PairwiseInteractionEstimateV1 {
                left_option_sha256: left_sha,
                right_option_sha256: right_sha,
                interaction_micros: interaction,
                absolute_error_bound_micros: 10_000,
            });
        }
    }
    pairs.sort_by(|left, right| {
        left.left_option_sha256
            .cmp(&right.left_option_sha256)
            .then_with(|| left.right_option_sha256.cmp(&right.right_option_sha256))
    });

    let context = PortfolioInteractionContextV1 {
        scope: scope(),
        as_known: as_known(300),
        budget,
        target_model_sha256: target_model.clone(),
        baseline_policy_sha256: hex_digest(840_002),
        task_protocol_sha256: hex_digest(840_003),
    };
    let mut proposal = PortfolioInteractionProposalV1 {
        scope: context.scope,
        as_known: context.as_known,
        budget,
        target_model_sha256: context.target_model_sha256.clone(),
        baseline_policy_sha256: context.baseline_policy_sha256.clone(),
        task_protocol_sha256: context.task_protocol_sha256.clone(),
        interaction_model_sha256: hex_digest(840_004),
        option_set_sha256: interaction_option_set_digest(&options),
        training_dataset_sha256: hex_digest(840_005),
        calibration_dataset_sha256: hex_digest(840_006),
        heldout_evaluation_sha256: hex_digest(840_007),
        task_family_split_sha256: hex_digest(840_008),
        source_root_split_sha256: hex_digest(840_009),
        pairs,
        proposal_sha256: String::new(),
    };
    proposal.proposal_sha256 = proposal.canonical_digest();
    let receipt = fixture_interaction_calibration_receipt(&proposal);
    let verified = verify_portfolio_interactions(
        proposal,
        &options,
        &context,
        &FixturePortfolioInteractionVerifier { receipt },
    )
    .expect("interaction model must pass the held-out calibration contract");
    let pairwise = optimize_memory_portfolio_pairwise(&options, budget, &verified)
        .expect("pairwise selector must return a bounded feasible plan");

    let additive_ids: Vec<_> = additive.selected.iter().map(|item| item.event_id).collect();
    let pairwise_ids: Vec<_> = pairwise
        .selection
        .selected
        .iter()
        .map(|item| item.event_id)
        .collect();
    let additive_utility = evaluator_hidden_portfolio_utility(&additive_ids);
    let pairwise_utility = evaluator_hidden_portfolio_utility(&pairwise_ids);
    let oracle_utility = evaluator_hidden_portfolio_utility(&[802, 803]);

    assert!(additive_ids.contains(&801));
    assert!(pairwise_ids.contains(&802) && pairwise_ids.contains(&803));
    assert!(pairwise.is_admissible_with(&verified));
    assert_eq!(pairwise_utility, oracle_utility);
    assert!(pairwise_utility > additive_utility);
}

