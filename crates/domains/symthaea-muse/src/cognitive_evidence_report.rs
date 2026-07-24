// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Claim-safe evidence reporting across mechanism, prediction, and usefulness.
//!
//! There is deliberately no single overall cognition score. Each claim is
//! supported only by the experiment designed to answer it.

use crate::adaptive_experiment::AdaptiveExperimentConclusion;
use crate::confirmatory_analysis::ConfirmatoryStudyConclusion;
use crate::experiment_manifest::ConfirmatoryEndpoint;
use crate::temporal_confirmatory::TemporalConfirmatoryConclusion;
use serde::{Deserialize, Serialize};

pub const COGNITIVE_EVIDENCE_REPORT_VERSION: &str = "symthaea-muse-cognitive-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceStatus {
    Supported,
    NotSupported,
    NotEvaluated,
    InvalidEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveEvidenceClaim {
    TemporalStateInfluencesFep,
    AdaptiveWorldModelImprovesPrediction,
    TheoryValidSonataOperation,
    BetterThanSimpleBaselines,
    NonInferiorToHandAuthoredHeuristic,
    BlindedListenerBenefit,
    ArtistWorkflowBenefit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimAssessment {
    pub claim: CognitiveEvidenceClaim,
    pub status: EvidenceStatus,
    pub evidence_source: String,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveEvidenceReport {
    pub report_version: String,
    pub mechanism: Option<TemporalConfirmatoryConclusion>,
    pub prediction: Option<AdaptiveExperimentConclusion>,
    pub usefulness: Option<ConfirmatoryStudyConclusion>,
    pub claims: Vec<ClaimAssessment>,
    pub explicit_non_claims: Vec<String>,
}

pub fn build_cognitive_evidence_report(
    mechanism: Option<TemporalConfirmatoryConclusion>,
    prediction: Option<AdaptiveExperimentConclusion>,
    usefulness: Option<ConfirmatoryStudyConclusion>,
) -> CognitiveEvidenceReport {
    let mut claims = Vec::new();
    claims.push(mechanism_claim(mechanism.as_ref()));
    claims.push(prediction_claim(prediction.as_ref()));
    claims.extend(usefulness_claims(usefulness.as_ref()));
    CognitiveEvidenceReport {
        report_version: COGNITIVE_EVIDENCE_REPORT_VERSION.into(),
        mechanism,
        prediction,
        usefulness,
        claims,
        explicit_non_claims: vec![
            "No result in this report establishes consciousness, sentience, or subjective musical experience.".into(),
            "Mechanistic influence does not imply musical benefit.".into(),
            "Prediction calibration does not imply preference or compositional quality.".into(),
            "A Sonata result does not generalize to other forms, cultures, renderers, or artists.".into(),
            "Null and negative results must be retained without changing frozen thresholds after observation.".into(),
        ],
    }
}

fn mechanism_claim(conclusion: Option<&TemporalConfirmatoryConclusion>) -> ClaimAssessment {
    match conclusion {
        None => assessment(
            CognitiveEvidenceClaim::TemporalStateInfluencesFep,
            EvidenceStatus::NotEvaluated,
            "temporal-confirmatory",
            "the confirmatory temporal ablation has not been supplied",
        ),
        Some(value) if !value.issues.is_empty() => assessment(
            CognitiveEvidenceClaim::TemporalStateInfluencesFep,
            EvidenceStatus::InvalidEvidence,
            "temporal-confirmatory",
            "the temporal evidence failed validation",
        ),
        Some(value) if value.success => assessment(
            CognitiveEvidenceClaim::TemporalStateInfluencesFep,
            EvidenceStatus::Supported,
            "temporal-confirmatory",
            "paired confirmatory evidence cleared the preregistered sensory and action-influence gates",
        ),
        Some(_) => assessment(
            CognitiveEvidenceClaim::TemporalStateInfluencesFep,
            EvidenceStatus::NotSupported,
            "temporal-confirmatory",
            "the frozen temporal mechanism gate did not pass",
        ),
    }
}

fn prediction_claim(conclusion: Option<&AdaptiveExperimentConclusion>) -> ClaimAssessment {
    match conclusion {
        None => assessment(
            CognitiveEvidenceClaim::AdaptiveWorldModelImprovesPrediction,
            EvidenceStatus::NotEvaluated,
            "adaptive-holdout",
            "the prequential adaptive holdout has not been supplied",
        ),
        Some(value) if !value.issues.is_empty() => assessment(
            CognitiveEvidenceClaim::AdaptiveWorldModelImprovesPrediction,
            EvidenceStatus::InvalidEvidence,
            "adaptive-holdout",
            "the adaptive holdout failed its leakage or validity checks",
        ),
        Some(value) if value.success => assessment(
            CognitiveEvidenceClaim::AdaptiveWorldModelImprovesPrediction,
            EvidenceStatus::Supported,
            "adaptive-holdout",
            "held-out symbolic prediction error improved by the frozen practical margin",
        ),
        Some(_) => assessment(
            CognitiveEvidenceClaim::AdaptiveWorldModelImprovesPrediction,
            EvidenceStatus::NotSupported,
            "adaptive-holdout",
            "the frozen prequential calibration gate did not pass",
        ),
    }
}

fn usefulness_claims(conclusion: Option<&ConfirmatoryStudyConclusion>) -> Vec<ClaimAssessment> {
    let claims = [
        CognitiveEvidenceClaim::TheoryValidSonataOperation,
        CognitiveEvidenceClaim::BetterThanSimpleBaselines,
        CognitiveEvidenceClaim::NonInferiorToHandAuthoredHeuristic,
        CognitiveEvidenceClaim::BlindedListenerBenefit,
        CognitiveEvidenceClaim::ArtistWorkflowBenefit,
    ];
    let Some(value) = conclusion else {
        return claims
            .into_iter()
            .map(|claim| {
                assessment(
                    claim,
                    EvidenceStatus::NotEvaluated,
                    "four-arm-confirmatory-study",
                    "the blinded confirmatory Sonata study has not been supplied",
                )
            })
            .collect();
    };
    if !value.issues.is_empty() {
        return claims
            .into_iter()
            .map(|claim| {
                assessment(
                    claim,
                    EvidenceStatus::InvalidEvidence,
                    "four-arm-confirmatory-study",
                    "the confirmatory study failed validation",
                )
            })
            .collect();
    }

    let listener = value.passing_primary_endpoints.iter().any(|endpoint| {
        matches!(
            endpoint,
            ConfirmatoryEndpoint::ReturnRecognition
                | ConfirmatoryEndpoint::EarnedRecapitulation
                | ConfirmatoryEndpoint::Preference
        )
    });
    let workflow = value.passing_primary_endpoints.iter().any(|endpoint| {
        matches!(
            endpoint,
            ConfirmatoryEndpoint::KeepRate | ConfirmatoryEndpoint::LowerTimeToCommit
        )
    });
    vec![
        assessment(
            CognitiveEvidenceClaim::TheoryValidSonataOperation,
            supported(value.structural_gate_passed),
            "four-arm-confirmatory-study",
            if value.structural_gate_passed {
                "the Symthaea arm preserved every frozen structural requirement"
            } else {
                "the Symthaea arm did not clear the frozen structural gate"
            },
        ),
        assessment(
            CognitiveEvidenceClaim::BetterThanSimpleBaselines,
            supported(value.analysis_gate_passed),
            "four-arm-confirmatory-study",
            if value.analysis_gate_passed {
                "at least the preregistered number of primary endpoints cleared fixed and random-valid superiority gates"
            } else {
                "the preregistered confirmatory usefulness gate did not pass"
            },
        ),
        assessment(
            CognitiveEvidenceClaim::NonInferiorToHandAuthoredHeuristic,
            supported(value.analysis_gate_passed),
            "four-arm-confirmatory-study",
            if value.analysis_gate_passed {
                "every passing endpoint also cleared the frozen heuristic non-inferiority comparison"
            } else {
                "heuristic non-inferiority was not established on the required primary endpoints"
            },
        ),
        assessment(
            CognitiveEvidenceClaim::BlindedListenerBenefit,
            supported(listener),
            "four-arm-confirmatory-study",
            if listener {
                "a preregistered blinded-listener endpoint passed all corrected comparisons"
            } else {
                "no preregistered blinded-listener endpoint passed all corrected comparisons"
            },
        ),
        assessment(
            CognitiveEvidenceClaim::ArtistWorkflowBenefit,
            supported(workflow),
            "four-arm-confirmatory-study",
            if workflow {
                "a preregistered artist-workflow endpoint passed all corrected comparisons"
            } else {
                "no preregistered artist-workflow endpoint passed all corrected comparisons"
            },
        ),
    ]
}

fn supported(value: bool) -> EvidenceStatus {
    if value {
        EvidenceStatus::Supported
    } else {
        EvidenceStatus::NotSupported
    }
}

fn assessment(
    claim: CognitiveEvidenceClaim,
    status: EvidenceStatus,
    evidence_source: &str,
    rationale: &str,
) -> ClaimAssessment {
    ClaimAssessment {
        claim,
        status,
        evidence_source: evidence_source.into(),
        rationale: rationale.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_evidence_never_becomes_a_negative_or_positive_result() {
        let report = build_cognitive_evidence_report(None, None, None);
        assert!(
            report
                .claims
                .iter()
                .all(|claim| claim.status == EvidenceStatus::NotEvaluated)
        );
    }

    #[test]
    fn report_has_no_single_overall_success_field() {
        let report = build_cognitive_evidence_report(None, None, None);
        let json = serde_json::to_value(report).unwrap();
        assert!(json.get("success").is_none());
        assert!(json.get("claims").is_some());
    }
}
