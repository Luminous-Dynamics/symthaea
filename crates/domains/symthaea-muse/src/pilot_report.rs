// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Claim-limited pilot review and confirmatory sample-size authority.

use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::ConfirmatoryEndpoint;
use crate::pilot_monitoring::{PilotOperationalDecision, PilotOperationalSnapshot};
use crate::pilot_protocol::{FrozenPilotProtocol, PilotAmendmentLedger};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PILOT_REVIEW_REPORT_VERSION: &str = "symthaea-muse-pilot-review-report-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotVarianceEstimate {
    pub endpoint: ConfirmatoryEndpoint,
    /// Pooled across blinded conditions; no arm-specific means or effects are
    /// part of the pilot report.
    pub pooled_standard_deviation: f64,
    pub participant_intraclass_correlation: f64,
    pub family_intraclass_correlation: f64,
    pub estimated_exclusion_rate: f64,
    pub observation_count: usize,
    pub participant_count: usize,
    pub family_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatorySampleSizeRecommendation {
    pub power_analysis_version: String,
    pub simulation_source_sha256: String,
    pub simulation_environment_sha256: String,
    pub simulation_input_sha256: String,
    pub target_power: f64,
    pub alpha: f64,
    pub practical_margin: f64,
    pub recommended_confirmatory_families: usize,
    pub recommended_participants_per_fixture: usize,
    pub maximum_planned_exclusion_fraction: f64,
    pub simulation_replicates: usize,
    pub recommendation_sha256: String,
}

#[derive(Serialize)]
struct SampleSizeCommitment<'a> {
    power_analysis_version: &'a str,
    simulation_source_sha256: &'a str,
    simulation_environment_sha256: &'a str,
    simulation_input_sha256: &'a str,
    target_power: f64,
    alpha: f64,
    practical_margin: f64,
    recommended_confirmatory_families: usize,
    recommended_participants_per_fixture: usize,
    maximum_planned_exclusion_fraction: f64,
    simulation_replicates: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotReviewReport {
    pub report_version: String,
    pub pilot_protocol_sha256: String,
    pub amendment_ledger_sha256: String,
    pub final_operational_snapshot_sha256: String,
    pub sealed_pilot_collection_sha256: String,
    pub pilot_closed_at_utc: String,
    pub operational_decision: PilotOperationalDecision,
    pub objectives_completed: Vec<String>,
    pub unresolved_operational_risks: Vec<String>,
    pub variance_estimates: Vec<PilotVarianceEstimate>,
    pub sample_size_recommendation: ConfirmatorySampleSizeRecommendation,
    pub instrument_changes_required: bool,
    pub confirmatory_manifest_must_be_refrozen: bool,
    pub pilot_data_excluded_from_confirmatory_analysis: bool,
    pub confirmatory_quality_claim_made: bool,
    pub external_receipt_uri: String,
    pub external_receipt_sha256: String,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct PilotReportCommitment<'a> {
    report_version: &'a str,
    pilot_protocol_sha256: &'a str,
    amendment_ledger_sha256: &'a str,
    final_operational_snapshot_sha256: &'a str,
    sealed_pilot_collection_sha256: &'a str,
    pilot_closed_at_utc: &'a str,
    operational_decision: PilotOperationalDecision,
    objectives_completed: &'a [String],
    unresolved_operational_risks: &'a [String],
    variance_estimates: &'a [PilotVarianceEstimate],
    sample_size_recommendation: &'a ConfirmatorySampleSizeRecommendation,
    instrument_changes_required: bool,
    confirmatory_manifest_must_be_refrozen: bool,
    pilot_data_excluded_from_confirmatory_analysis: bool,
    confirmatory_quality_claim_made: bool,
    external_receipt_uri: &'a str,
    external_receipt_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotReviewIssue {
    WrongVersion {
        found: String,
    },
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    EmptyField {
        field: String,
    },
    PilotNotReadyToClose,
    EmptyObjectiveCompletion,
    EmptyVarianceRegistry,
    DuplicateVarianceEndpoint {
        endpoint: ConfirmatoryEndpoint,
    },
    InvalidVariance {
        endpoint: ConfirmatoryEndpoint,
        field: String,
    },
    ZeroVarianceEvidence {
        endpoint: ConfirmatoryEndpoint,
        field: String,
    },
    InvalidPowerParameter {
        field: String,
    },
    TooFewSimulationReplicates {
        found: usize,
        required: usize,
    },
    PilotDataAllowedInConfirmation,
    ConfirmatoryQualityClaimMade,
    InstrumentChangeWithoutRefreeze,
    SampleSizeDigestMismatch,
    ReportDigestMismatch,
}

pub fn sample_size_recommendation_commitment(
    recommendation: &ConfirmatorySampleSizeRecommendation,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SampleSizeCommitment {
        power_analysis_version: &recommendation.power_analysis_version,
        simulation_source_sha256: &recommendation.simulation_source_sha256,
        simulation_environment_sha256: &recommendation.simulation_environment_sha256,
        simulation_input_sha256: &recommendation.simulation_input_sha256,
        target_power: recommendation.target_power,
        alpha: recommendation.alpha,
        practical_margin: recommendation.practical_margin,
        recommended_confirmatory_families: recommendation.recommended_confirmatory_families,
        recommended_participants_per_fixture: recommendation.recommended_participants_per_fixture,
        maximum_planned_exclusion_fraction: recommendation.maximum_planned_exclusion_fraction,
        simulation_replicates: recommendation.simulation_replicates,
    })
}

pub fn seal_sample_size_recommendation(
    recommendation: &mut ConfirmatorySampleSizeRecommendation,
) -> Result<(), serde_json::Error> {
    recommendation.recommendation_sha256 = sample_size_recommendation_commitment(recommendation)?;
    Ok(())
}

pub fn pilot_review_report_commitment(
    report: &PilotReviewReport,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PilotReportCommitment {
        report_version: &report.report_version,
        pilot_protocol_sha256: &report.pilot_protocol_sha256,
        amendment_ledger_sha256: &report.amendment_ledger_sha256,
        final_operational_snapshot_sha256: &report.final_operational_snapshot_sha256,
        sealed_pilot_collection_sha256: &report.sealed_pilot_collection_sha256,
        pilot_closed_at_utc: &report.pilot_closed_at_utc,
        operational_decision: report.operational_decision,
        objectives_completed: &report.objectives_completed,
        unresolved_operational_risks: &report.unresolved_operational_risks,
        variance_estimates: &report.variance_estimates,
        sample_size_recommendation: &report.sample_size_recommendation,
        instrument_changes_required: report.instrument_changes_required,
        confirmatory_manifest_must_be_refrozen: report.confirmatory_manifest_must_be_refrozen,
        pilot_data_excluded_from_confirmatory_analysis: report
            .pilot_data_excluded_from_confirmatory_analysis,
        confirmatory_quality_claim_made: report.confirmatory_quality_claim_made,
        external_receipt_uri: &report.external_receipt_uri,
        external_receipt_sha256: &report.external_receipt_sha256,
    })
}

pub fn seal_pilot_review_report(report: &mut PilotReviewReport) -> Result<(), serde_json::Error> {
    report
        .variance_estimates
        .sort_by_key(|estimate| estimate.endpoint);
    report.report_sha256 = pilot_review_report_commitment(report)?;
    Ok(())
}

pub fn validate_pilot_review_report(
    protocol: &FrozenPilotProtocol,
    amendments: &PilotAmendmentLedger,
    snapshot: &PilotOperationalSnapshot,
    report: &PilotReviewReport,
) -> Vec<PilotReviewIssue> {
    let mut issues = Vec::new();
    if report.report_version != PILOT_REVIEW_REPORT_VERSION {
        issues.push(PilotReviewIssue::WrongVersion {
            found: report.report_version.clone(),
        });
    }
    verify_digest(
        protocol,
        &report.pilot_protocol_sha256,
        "pilot_protocol_sha256",
        &mut issues,
    );
    verify_digest(
        amendments,
        &report.amendment_ledger_sha256,
        "amendment_ledger_sha256",
        &mut issues,
    );
    verify_digest(
        snapshot,
        &report.final_operational_snapshot_sha256,
        "final_operational_snapshot_sha256",
        &mut issues,
    );
    for (field, digest) in [
        (
            "sealed_pilot_collection_sha256",
            report.sealed_pilot_collection_sha256.as_str(),
        ),
        (
            "external_receipt_sha256",
            report.external_receipt_sha256.as_str(),
        ),
        ("report_sha256", report.report_sha256.as_str()),
        (
            "sample_size.simulation_source_sha256",
            report
                .sample_size_recommendation
                .simulation_source_sha256
                .as_str(),
        ),
        (
            "sample_size.simulation_environment_sha256",
            report
                .sample_size_recommendation
                .simulation_environment_sha256
                .as_str(),
        ),
        (
            "sample_size.simulation_input_sha256",
            report
                .sample_size_recommendation
                .simulation_input_sha256
                .as_str(),
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(PilotReviewIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("pilot_closed_at_utc", report.pilot_closed_at_utc.as_str()),
        ("external_receipt_uri", report.external_receipt_uri.as_str()),
        (
            "power_analysis_version",
            report
                .sample_size_recommendation
                .power_analysis_version
                .as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            issues.push(PilotReviewIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    if report.operational_decision != PilotOperationalDecision::ReadyToClosePilot
        || snapshot.decision != PilotOperationalDecision::ReadyToClosePilot
    {
        issues.push(PilotReviewIssue::PilotNotReadyToClose);
    }
    if report.objectives_completed.is_empty()
        || report
            .objectives_completed
            .iter()
            .any(|value| value.trim().is_empty())
    {
        issues.push(PilotReviewIssue::EmptyObjectiveCompletion);
    }
    if report.variance_estimates.is_empty() {
        issues.push(PilotReviewIssue::EmptyVarianceRegistry);
    }
    let mut endpoints = BTreeSet::new();
    for estimate in &report.variance_estimates {
        if !endpoints.insert(estimate.endpoint) {
            issues.push(PilotReviewIssue::DuplicateVarianceEndpoint {
                endpoint: estimate.endpoint,
            });
        }
        for (field, value) in [
            (
                "pooled_standard_deviation",
                estimate.pooled_standard_deviation,
            ),
            (
                "participant_intraclass_correlation",
                estimate.participant_intraclass_correlation,
            ),
            (
                "family_intraclass_correlation",
                estimate.family_intraclass_correlation,
            ),
            (
                "estimated_exclusion_rate",
                estimate.estimated_exclusion_rate,
            ),
        ] {
            let valid = match field {
                "pooled_standard_deviation" => value.is_finite() && value >= 0.0,
                _ => value.is_finite() && (0.0..=1.0).contains(&value),
            };
            if !valid {
                issues.push(PilotReviewIssue::InvalidVariance {
                    endpoint: estimate.endpoint,
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            ("observation_count", estimate.observation_count),
            ("participant_count", estimate.participant_count),
            ("family_count", estimate.family_count),
        ] {
            if value == 0 {
                issues.push(PilotReviewIssue::ZeroVarianceEvidence {
                    endpoint: estimate.endpoint,
                    field: field.into(),
                });
            }
        }
    }
    let recommendation = &report.sample_size_recommendation;
    for (field, value, minimum, maximum) in [
        ("target_power", recommendation.target_power, 0.5, 0.999),
        ("alpha", recommendation.alpha, 0.0001, 0.10),
        (
            "maximum_planned_exclusion_fraction",
            recommendation.maximum_planned_exclusion_fraction,
            0.0,
            0.5,
        ),
    ] {
        if !value.is_finite() || value < minimum || value > maximum {
            issues.push(PilotReviewIssue::InvalidPowerParameter {
                field: field.into(),
            });
        }
    }
    if !recommendation.practical_margin.is_finite() || recommendation.practical_margin <= 0.0 {
        issues.push(PilotReviewIssue::InvalidPowerParameter {
            field: "practical_margin".into(),
        });
    }
    if recommendation.recommended_confirmatory_families < 8 {
        issues.push(PilotReviewIssue::InvalidPowerParameter {
            field: "recommended_confirmatory_families".into(),
        });
    }
    if recommendation.recommended_participants_per_fixture < 12
        || recommendation.recommended_participants_per_fixture % 4 != 0
    {
        issues.push(PilotReviewIssue::InvalidPowerParameter {
            field: "recommended_participants_per_fixture".into(),
        });
    }
    if recommendation.simulation_replicates < 10_000 {
        issues.push(PilotReviewIssue::TooFewSimulationReplicates {
            found: recommendation.simulation_replicates,
            required: 10_000,
        });
    }
    match sample_size_recommendation_commitment(recommendation) {
        Ok(value) if value == recommendation.recommendation_sha256 => {}
        Ok(_) => issues.push(PilotReviewIssue::SampleSizeDigestMismatch),
        Err(_) => issues.push(PilotReviewIssue::SerializationFailed {
            field: "sample_size_recommendation".into(),
        }),
    }
    if !report.pilot_data_excluded_from_confirmatory_analysis {
        issues.push(PilotReviewIssue::PilotDataAllowedInConfirmation);
    }
    if report.confirmatory_quality_claim_made {
        issues.push(PilotReviewIssue::ConfirmatoryQualityClaimMade);
    }
    if report.instrument_changes_required && !report.confirmatory_manifest_must_be_refrozen {
        issues.push(PilotReviewIssue::InstrumentChangeWithoutRefreeze);
    }
    match pilot_review_report_commitment(report) {
        Ok(value) if value == report.report_sha256 => {}
        Ok(_) => issues.push(PilotReviewIssue::ReportDigestMismatch),
        Err(_) => issues.push(PilotReviewIssue::SerializationFailed {
            field: "pilot_review_report".into(),
        }),
    }
    issues
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<PilotReviewIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(PilotReviewIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(PilotReviewIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_report_cannot_make_confirmatory_claim() {
        assert!(matches!(
            PilotReviewIssue::ConfirmatoryQualityClaimMade,
            PilotReviewIssue::ConfirmatoryQualityClaimMade
        ));
    }
}
