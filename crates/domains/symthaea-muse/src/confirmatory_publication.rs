// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complete confirmatory result disclosure and claim-limiting publication gate.
//!
//! Every frozen endpoint must appear, including null, negative, or
//! non-estimable outcomes. A failed crosscheck or any analysis deviation forces
//! descriptive-only language; a confirmatory negative result remains a valid
//! confirmatory result and must be published as such.

use crate::confirmatory_analysis_execution::{
    ConfirmatoryAnalysisExecutionRecord, ConfirmatoryClaimStatus,
    confirmatory_analysis_execution_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{ConfirmatoryEndpoint, FrozenStudyManifest};
use crate::methodology_plan::{EndpointRole, FrozenMethodologyPlan};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONFIRMATORY_PUBLICATION_VERSION: &str = "symthaea-muse-confirmatory-publication-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointAnalysisStatus {
    Completed,
    NotEstimable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryEndpointDisclosure {
    pub endpoint: ConfirmatoryEndpoint,
    pub role: EndpointRole,
    pub analysis_status: EndpointAnalysisStatus,
    pub estimate: Option<f64>,
    pub confidence_interval: Option<[f64; 2]>,
    pub adjusted_p_value: Option<f64>,
    pub practical_margin: Option<f64>,
    pub gate_passed: Option<bool>,
    pub non_estimable_reason: Option<String>,
    pub analysis_output_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryPrimaryConclusion {
    ConfirmedBenefit,
    DidNotConfirmBenefit,
    DescriptiveOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryDisclosureStatement {
    pub statement_id: String,
    pub text: String,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryPublicationRecord {
    pub publication_version: String,
    pub study_id: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub analysis_execution_sha256: String,
    pub primary_conclusion: ConfirmatoryPrimaryConclusion,
    pub endpoint_disclosures: Vec<ConfirmatoryEndpointDisclosure>,
    pub participant_flow_statement: ConfirmatoryDisclosureStatement,
    pub exclusions_statement: ConfirmatoryDisclosureStatement,
    pub deviations_statement: ConfirmatoryDisclosureStatement,
    pub adverse_events_statement: ConfirmatoryDisclosureStatement,
    pub conflicts_and_funding_statement: ConfirmatoryDisclosureStatement,
    pub limitations: Vec<String>,
    pub data_and_code_availability_uri: String,
    pub preregistration_uri: String,
    pub published_at_utc: String,
    pub record_sha256: String,
}

#[derive(Serialize)]
struct PublicationCommitment<'a> {
    publication_version: &'a str,
    study_id: &'a str,
    manifest_sha256: &'a str,
    methodology_sha256: &'a str,
    analysis_execution_sha256: &'a str,
    primary_conclusion: ConfirmatoryPrimaryConclusion,
    endpoint_disclosures: &'a [ConfirmatoryEndpointDisclosure],
    participant_flow_statement: &'a ConfirmatoryDisclosureStatement,
    exclusions_statement: &'a ConfirmatoryDisclosureStatement,
    deviations_statement: &'a ConfirmatoryDisclosureStatement,
    adverse_events_statement: &'a ConfirmatoryDisclosureStatement,
    conflicts_and_funding_statement: &'a ConfirmatoryDisclosureStatement,
    limitations: &'a [String],
    data_and_code_availability_uri: &'a str,
    preregistration_uri: &'a str,
    published_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryPublicationIssue {
    InvalidManifest,
    InvalidMethodology,
    InvalidAnalysisExecution,
    WrongVersion {
        found: String,
    },
    EmptyField {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    DuplicateEndpoint {
        endpoint: ConfirmatoryEndpoint,
    },
    MissingEndpoint {
        endpoint: ConfirmatoryEndpoint,
    },
    UnexpectedEndpoint {
        endpoint: ConfirmatoryEndpoint,
    },
    EndpointRoleMismatch {
        endpoint: ConfirmatoryEndpoint,
    },
    InvalidCompletedEndpoint {
        endpoint: ConfirmatoryEndpoint,
        field: String,
    },
    InvalidNotEstimableEndpoint {
        endpoint: ConfirmatoryEndpoint,
        field: String,
    },
    PrimaryConclusionMismatch,
    MissingDisclosureStatement {
        statement_id: String,
    },
    InvalidDisclosureDigest {
        statement_id: String,
    },
    EmptyLimitation,
    SerializationFailed,
    RecordDigestMismatch,
}

pub fn confirmatory_publication_commitment(
    record: &ConfirmatoryPublicationRecord,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PublicationCommitment {
        publication_version: &record.publication_version,
        study_id: &record.study_id,
        manifest_sha256: &record.manifest_sha256,
        methodology_sha256: &record.methodology_sha256,
        analysis_execution_sha256: &record.analysis_execution_sha256,
        primary_conclusion: record.primary_conclusion,
        endpoint_disclosures: &record.endpoint_disclosures,
        participant_flow_statement: &record.participant_flow_statement,
        exclusions_statement: &record.exclusions_statement,
        deviations_statement: &record.deviations_statement,
        adverse_events_statement: &record.adverse_events_statement,
        conflicts_and_funding_statement: &record.conflicts_and_funding_statement,
        limitations: &record.limitations,
        data_and_code_availability_uri: &record.data_and_code_availability_uri,
        preregistration_uri: &record.preregistration_uri,
        published_at_utc: &record.published_at_utc,
    })
}

pub fn seal_confirmatory_publication(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    analysis: &ConfirmatoryAnalysisExecutionRecord,
    record: &mut ConfirmatoryPublicationRecord,
) -> Result<(), Vec<ConfirmatoryPublicationIssue>> {
    record
        .endpoint_disclosures
        .sort_by_key(|disclosure| disclosure.endpoint);
    record.limitations.sort();
    record.record_sha256 = confirmatory_publication_commitment(record)
        .map_err(|_| vec![ConfirmatoryPublicationIssue::SerializationFailed])?;
    let issues = validate_confirmatory_publication(manifest, methodology, analysis, record);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_confirmatory_publication(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    analysis: &ConfirmatoryAnalysisExecutionRecord,
    record: &ConfirmatoryPublicationRecord,
) -> Vec<ConfirmatoryPublicationIssue> {
    let mut issues = Vec::new();
    if !manifest.validate().is_empty() {
        issues.push(ConfirmatoryPublicationIssue::InvalidManifest);
    }
    if !methodology.validate(manifest).is_empty() {
        issues.push(ConfirmatoryPublicationIssue::InvalidMethodology);
    }
    match confirmatory_analysis_execution_commitment(analysis) {
        Ok(found)
            if found == analysis.record_sha256 && found == record.analysis_execution_sha256 => {}
        _ => issues.push(ConfirmatoryPublicationIssue::InvalidAnalysisExecution),
    }
    if record.publication_version != CONFIRMATORY_PUBLICATION_VERSION {
        issues.push(ConfirmatoryPublicationIssue::WrongVersion {
            found: record.publication_version.clone(),
        });
    }
    for (field, value) in [
        ("study_id", record.study_id.as_str()),
        (
            "data_and_code_availability_uri",
            record.data_and_code_availability_uri.as_str(),
        ),
        ("preregistration_uri", record.preregistration_uri.as_str()),
        ("published_at_utc", record.published_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryPublicationIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    verify_digest(
        "manifest_sha256",
        canonical_json_sha256(manifest),
        &record.manifest_sha256,
        &mut issues,
    );
    verify_digest(
        "methodology_sha256",
        canonical_json_sha256(methodology),
        &record.methodology_sha256,
        &mut issues,
    );
    if !is_sha256(&record.analysis_execution_sha256) {
        issues.push(ConfirmatoryPublicationIssue::InvalidDigest {
            field: "analysis_execution_sha256".into(),
        });
    }

    let declarations: BTreeMap<_, _> = methodology
        .endpoints
        .iter()
        .map(|declaration| (declaration.endpoint, declaration))
        .collect();
    let mut disclosed = BTreeSet::new();
    for endpoint in &record.endpoint_disclosures {
        if !disclosed.insert(endpoint.endpoint) {
            issues.push(ConfirmatoryPublicationIssue::DuplicateEndpoint {
                endpoint: endpoint.endpoint,
            });
        }
        match declarations.get(&endpoint.endpoint) {
            None => issues.push(ConfirmatoryPublicationIssue::UnexpectedEndpoint {
                endpoint: endpoint.endpoint,
            }),
            Some(declaration) if declaration.role != endpoint.role => {
                issues.push(ConfirmatoryPublicationIssue::EndpointRoleMismatch {
                    endpoint: endpoint.endpoint,
                });
            }
            Some(_) => {}
        }
        validate_endpoint(endpoint, &mut issues);
    }
    for endpoint in declarations.keys() {
        if !disclosed.contains(endpoint) {
            issues.push(ConfirmatoryPublicationIssue::MissingEndpoint {
                endpoint: *endpoint,
            });
        }
    }
    let expected_conclusion = match analysis.claim_status {
        ConfirmatoryClaimStatus::DescriptiveOnly => ConfirmatoryPrimaryConclusion::DescriptiveOnly,
        ConfirmatoryClaimStatus::Confirmatory if analysis.primary_success => {
            ConfirmatoryPrimaryConclusion::ConfirmedBenefit
        }
        ConfirmatoryClaimStatus::Confirmatory => {
            ConfirmatoryPrimaryConclusion::DidNotConfirmBenefit
        }
    };
    if record.primary_conclusion != expected_conclusion {
        issues.push(ConfirmatoryPublicationIssue::PrimaryConclusionMismatch);
    }
    for statement in [
        &record.participant_flow_statement,
        &record.exclusions_statement,
        &record.deviations_statement,
        &record.adverse_events_statement,
        &record.conflicts_and_funding_statement,
    ] {
        validate_statement(statement, &mut issues);
    }
    if record
        .limitations
        .iter()
        .any(|value| value.trim().is_empty())
    {
        issues.push(ConfirmatoryPublicationIssue::EmptyLimitation);
    }
    match confirmatory_publication_commitment(record) {
        Ok(found) if found == record.record_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryPublicationIssue::RecordDigestMismatch),
        Err(_) => issues.push(ConfirmatoryPublicationIssue::SerializationFailed),
    }
    issues
}

fn validate_endpoint(
    endpoint: &ConfirmatoryEndpointDisclosure,
    issues: &mut Vec<ConfirmatoryPublicationIssue>,
) {
    if !is_sha256(&endpoint.analysis_output_sha256) {
        issues.push(ConfirmatoryPublicationIssue::InvalidDigest {
            field: format!("endpoint.{:?}.analysis_output_sha256", endpoint.endpoint),
        });
    }
    match endpoint.analysis_status {
        EndpointAnalysisStatus::Completed => {
            for (field, present) in [
                ("estimate", endpoint.estimate.is_some_and(f64::is_finite)),
                (
                    "confidence_interval",
                    endpoint.confidence_interval.is_some_and(|interval| {
                        interval.iter().all(|value| value.is_finite()) && interval[0] <= interval[1]
                    }),
                ),
                (
                    "adjusted_p_value",
                    endpoint
                        .adjusted_p_value
                        .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value)),
                ),
                (
                    "practical_margin",
                    endpoint.practical_margin.is_some_and(f64::is_finite),
                ),
                ("gate_passed", endpoint.gate_passed.is_some()),
                (
                    "non_estimable_reason",
                    endpoint.non_estimable_reason.is_none(),
                ),
            ] {
                if !present {
                    issues.push(ConfirmatoryPublicationIssue::InvalidCompletedEndpoint {
                        endpoint: endpoint.endpoint,
                        field: field.into(),
                    });
                }
            }
        }
        EndpointAnalysisStatus::NotEstimable => {
            for (field, valid) in [
                ("estimate", endpoint.estimate.is_none()),
                (
                    "confidence_interval",
                    endpoint.confidence_interval.is_none(),
                ),
                ("adjusted_p_value", endpoint.adjusted_p_value.is_none()),
                ("practical_margin", endpoint.practical_margin.is_none()),
                ("gate_passed", endpoint.gate_passed.is_none()),
                (
                    "non_estimable_reason",
                    endpoint
                        .non_estimable_reason
                        .as_deref()
                        .is_some_and(|value| !value.trim().is_empty()),
                ),
            ] {
                if !valid {
                    issues.push(ConfirmatoryPublicationIssue::InvalidNotEstimableEndpoint {
                        endpoint: endpoint.endpoint,
                        field: field.into(),
                    });
                }
            }
        }
    }
}

fn validate_statement(
    statement: &ConfirmatoryDisclosureStatement,
    issues: &mut Vec<ConfirmatoryPublicationIssue>,
) {
    if statement.statement_id.trim().is_empty() || statement.text.trim().is_empty() {
        issues.push(ConfirmatoryPublicationIssue::MissingDisclosureStatement {
            statement_id: statement.statement_id.clone(),
        });
    }
    if !is_sha256(&statement.evidence_sha256) {
        issues.push(ConfirmatoryPublicationIssue::InvalidDisclosureDigest {
            statement_id: statement.statement_id.clone(),
        });
    }
}

fn verify_digest(
    field: &str,
    expected: Result<String, serde_json::Error>,
    found: &str,
    issues: &mut Vec<ConfirmatoryPublicationIssue>,
) {
    match expected {
        Ok(value) if value == found => {}
        Ok(_) => issues.push(ConfirmatoryPublicationIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ConfirmatoryPublicationIssue::SerializationFailed),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_confirmatory_result_is_not_descriptive() {
        let conclusion = match (ConfirmatoryClaimStatus::Confirmatory, false) {
            (ConfirmatoryClaimStatus::DescriptiveOnly, _) => {
                ConfirmatoryPrimaryConclusion::DescriptiveOnly
            }
            (ConfirmatoryClaimStatus::Confirmatory, true) => {
                ConfirmatoryPrimaryConclusion::ConfirmedBenefit
            }
            (ConfirmatoryClaimStatus::Confirmatory, false) => {
                ConfirmatoryPrimaryConclusion::DidNotConfirmBenefit
            }
        };
        assert_eq!(
            conclusion,
            ConfirmatoryPrimaryConclusion::DidNotConfirmBenefit
        );
    }
}
