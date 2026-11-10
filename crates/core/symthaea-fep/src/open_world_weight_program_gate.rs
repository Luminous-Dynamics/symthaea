// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Connects open-world predictive adequacy to sequential WeightProgram evaluation.
//!
//! A candidate evaluation stream may begin only when the open-world gate permits
//! detached candidate evaluation. The exact adequacy report and gate decision
//! are hashed and bound into the statistical verifier receipt. This does not
//! prove that the model family is adequate: NoDetectedInadequacy means only
//! that the frozen thresholds did not detect a problem.
//!
//! This adapter never accepts a candidate, executes an action, or mutates a
//! live model. The independent signed verifier remains mandatory.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use symthaea_core::sequential_weight_verification::{
    ConfirmatoryEpisodeManifest, SequentialWeightVerification,
    SequentialWeightVerificationCampaign, SequentialWeightVerificationError,
};

use super::open_world_robust_decision::{
    ModelAdequacyReport, OpenWorldDecisionGateReport, OpenWorldGateDisposition,
    evaluate_open_world_decision_gate,
};

/// Detached candidate evaluation stream bound to its open-world precondition.
#[derive(Debug)]
pub struct OpenWorldBoundCandidateEvaluation {
    /// The model-adequacy decision that permitted detached evaluation.
    pub gate_report: OpenWorldDecisionGateReport,
    /// SHA-256 digest of the adequacy report and gate report.
    pub precondition_gate_digest: String,
    /// Statistical verifier for the frozen baseline/candidate/shard triple.
    pub verification: SequentialWeightVerification,
}

/// Failure to establish the open-world precondition for candidate evaluation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OpenWorldWeightProgramGateError {
    #[error("open-world gate does not permit detached candidate evaluation: {0:?}")]
    CandidateEvaluationNotPermitted(OpenWorldGateDisposition),
    #[error("open-world gate authority fields are internally inconsistent")]
    InvalidGateAuthority,
    #[error("failed to serialize the model-adequacy precondition")]
    ReportSerialization,
    #[error(transparent)]
    Verification(#[from] SequentialWeightVerificationError),
}

/// Start a candidate's confirmatory stream only after open-world adequacy permits it.
///
/// The caller supplies the fresh candidate-specific shard digest. The campaign
/// rejects shard reuse and assigns the candidate ordinal under one family-wise
/// alpha budget. The gate digest is carried into every statistical report and
/// can then be included in the independent verifier's signed attestation.
pub fn begin_open_world_candidate_evaluation(
    adequacy: &ModelAdequacyReport,
    prequential_evidence_crossed_at: Option<usize>,
    campaign: &mut SequentialWeightVerificationCampaign,
    baseline_model_digest: String,
    candidate_model_digest: String,
    confirmatory_shard_digest: String,
    confirmatory_episode_manifest: ConfirmatoryEpisodeManifest,
) -> Result<OpenWorldBoundCandidateEvaluation, OpenWorldWeightProgramGateError> {
    let gate_report =
        evaluate_open_world_decision_gate(adequacy, prequential_evidence_crossed_at);

    if gate_report.disposition != OpenWorldGateDisposition::DetachedCandidateEvaluationPermitted
        || !gate_report.detached_candidate_evaluation_permitted
        || !gate_report.weight_program_generation_permitted
    {
        return Err(
            OpenWorldWeightProgramGateError::CandidateEvaluationNotPermitted(
                gate_report.disposition,
            ),
        );
    }
    if !gate_report.independent_verifier_required
        || gate_report.candidate_acceptance_authorized
        || gate_report.live_model_mutation_authorized
    {
        return Err(OpenWorldWeightProgramGateError::InvalidGateAuthority);
    }

    let encoded = serde_json::to_vec(&(adequacy, &gate_report))
        .map_err(|_| OpenWorldWeightProgramGateError::ReportSerialization)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-open-world-weight-program-precondition-v1\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    let precondition_gate_digest = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let verification = campaign.start_candidate(
        baseline_model_digest,
        candidate_model_digest,
        confirmatory_shard_digest,
        confirmatory_episode_manifest,
        precondition_gate_digest.clone(),
    )?;

    Ok(OpenWorldBoundCandidateEvaluation {
        gate_report,
        precondition_gate_digest,
        verification,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_world_robust_decision::{
        ModelAdequacyStatus, ModelAdequacyThresholds,
    };

    fn adequacy(status: ModelAdequacyStatus) -> ModelAdequacyReport {
        ModelAdequacyReport {
            status,
            observation_count: 100,
            zero_support_count: 0,
            low_probability_count: 0,
            low_probability_rate: 0.0,
            mean_log_loss: Some(0.3),
            mean_multiclass_brier_score: Some(0.1),
            minimum_observed_probability: Some(0.2),
            thresholds: ModelAdequacyThresholds {
                minimum_observations: 50,
                minimum_predictive_probability: 0.01,
                maximum_mean_log_loss: 1.0,
                maximum_mean_brier_score: 0.5,
                maximum_low_probability_rate: 0.1,
            },
        }
    }

    fn episode_manifest(marker: char) -> ConfirmatoryEpisodeManifest {
        ConfirmatoryEpisodeManifest::new(
            (0..1_000)
                .map(|index| format!("open-world-{marker}-{index}"))
                .collect(),
        )
        .expect("valid confirmatory manifest")
    }

    fn campaign() -> SequentialWeightVerificationCampaign {
        SequentialWeightVerificationCampaign::new(
            symthaea_core::sequential_weight_verification::SequentialWeightVerificationConfig {
                campaign_alpha: 0.05,
                minimum_improvement: 0.05,
                minimum_observations: 10,
                maximum_observations: 1_000,
            },
            "c".repeat(64),
        )
        .expect("valid campaign")
    }

    #[test]
    fn adequate_but_not_proven_model_permits_only_detached_candidate_evaluation() {
        let mut campaign = campaign();
        let started = begin_open_world_candidate_evaluation(
            &adequacy(ModelAdequacyStatus::NoDetectedInadequacy),
            None,
            &mut campaign,
            "a".repeat(64),
            "b".repeat(64),
            "d".repeat(64),
            episode_manifest('d'),
        )
        .expect("detached candidate evaluation permitted");

        assert!(started.gate_report.detached_candidate_evaluation_permitted);
        assert!(started.gate_report.independent_verifier_required);
        assert!(!started.gate_report.candidate_acceptance_authorized);
        assert!(!started.gate_report.live_model_mutation_authorized);
        assert_eq!(started.precondition_gate_digest.len(), 64);
        let report = started.verification.report().expect("statistical report");
        assert_eq!(
            report.binding.precondition_gate_digest,
            started.precondition_gate_digest
        );
    }

    #[test]
    fn insufficient_or_inadequate_predictive_models_cannot_start_candidate_evaluation() {
        for (status, crossed_at, expected) in [
            (
                ModelAdequacyStatus::InsufficientEvidence,
                None,
                OpenWorldGateDisposition::EvidenceCollectionOnly,
            ),
            (
                ModelAdequacyStatus::NoDetectedInadequacy,
                Some(7),
                OpenWorldGateDisposition::ModelRevisionRequired,
            ),
            (
                ModelAdequacyStatus::Inadequate,
                None,
                OpenWorldGateDisposition::ModelRevisionRequired,
            ),
        ] {
            let mut campaign = campaign();
            assert!(matches!(
                begin_open_world_candidate_evaluation(
                    &adequacy(status),
                    crossed_at,
                    &mut campaign,
                    "a".repeat(64),
                    "b".repeat(64),
                    "d".repeat(64),
                    episode_manifest('d'),
                ),
                Err(OpenWorldWeightProgramGateError::CandidateEvaluationNotPermitted(
                    disposition
                )) if disposition == expected
            ));
            assert_eq!(campaign.next_candidate_index(), 1);
            assert_eq!(campaign.used_confirmatory_shard_count(), 0);
        }
    }

    #[test]
    fn changed_adequacy_evidence_changes_the_precondition_binding() {
        let mut first_campaign = campaign();
        let first = begin_open_world_candidate_evaluation(
            &adequacy(ModelAdequacyStatus::NoDetectedInadequacy),
            None,
            &mut first_campaign,
            "a".repeat(64),
            "b".repeat(64),
            "d".repeat(64),
            episode_manifest('d'),
        )
        .expect("first precondition");

        let mut second_adequacy = adequacy(ModelAdequacyStatus::NoDetectedInadequacy);
        second_adequacy.low_probability_count = 1;
        second_adequacy.low_probability_rate = 0.01;
        let mut second_campaign = campaign();
        let second = begin_open_world_candidate_evaluation(
            &second_adequacy,
            None,
            &mut second_campaign,
            "a".repeat(64),
            "b".repeat(64),
            "e".repeat(64),
            episode_manifest('e'),
        )
        .expect("second precondition");
        assert_ne!(
            first.precondition_gate_digest,
            second.precondition_gate_digest
        );
    }
}
