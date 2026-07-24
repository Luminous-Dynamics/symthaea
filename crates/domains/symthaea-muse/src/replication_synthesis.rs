// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-site synthesis for independently executed replication studies.
//!
//! The synthesis uses inverse-variance random-effects aggregation, reports
//! heterogeneity and attenuation relative to the published source estimate,
//! and derives a conservative conclusion from the frozen practical margin.

use crate::evidence_digest::canonical_json_sha256;
use crate::replication_execution::{
    ReplicationSiteExecutionRecord, SiteReplicationConclusion, replication_execution_commitment,
};
use crate::replication_protocol::{
    FavorableDirection, FrozenReplicationProtocol, replication_protocol_commitment,
};
use crate::replication_site_registry::{
    ReplicationSiteRegistry, ReplicationSiteStatus, replication_site_registry_commitment,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REPLICATION_SYNTHESIS_VERSION: &str = "symthaea-muse-replication-synthesis-v1";
const NORMAL_95_Z: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishedSourcePrimaryResult {
    pub source_final_release_sha256: String,
    pub endpoint_id: String,
    pub estimate: f64,
    pub standard_error: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicationMetaAnalysis {
    pub site_count: u32,
    pub participant_count: u32,
    pub family_count: u32,
    pub fixed_effect_estimate: f64,
    pub random_effect_estimate: f64,
    pub random_effect_standard_error: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub cochran_q: f64,
    pub tau_squared: f64,
    pub i_squared_percent: f64,
    pub source_attenuation_ratio: Option<f64>,
    pub direction_concordant_with_source: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationSynthesisConclusion {
    IndependentlyReplicated,
    MixedEvidence,
    DidNotReplicate,
    DescriptiveOnly,
    InsufficientEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicationSynthesisRecord {
    pub synthesis_version: String,
    pub replication_id: String,
    pub protocol_sha256: String,
    pub site_registry_sha256: String,
    pub source_result: PublishedSourcePrimaryResult,
    pub site_execution_sha256: Vec<String>,
    pub quantitative_site_ids: Vec<String>,
    pub excluded_site_ids: Vec<String>,
    pub meta_analysis: Option<ReplicationMetaAnalysis>,
    pub conclusion: ReplicationSynthesisConclusion,
    pub analysis_implementation_sha256: String,
    pub independent_analysis_sha256: String,
    pub crosscheck_sha256: String,
    pub public_release_uri: String,
    pub completed_at_utc: String,
    pub synthesis_sha256: String,
}

#[derive(Serialize)]
struct SynthesisCommitment<'a> {
    synthesis_version: &'a str,
    replication_id: &'a str,
    protocol_sha256: &'a str,
    site_registry_sha256: &'a str,
    source_result: &'a PublishedSourcePrimaryResult,
    site_execution_sha256: &'a [String],
    quantitative_site_ids: &'a [String],
    excluded_site_ids: &'a [String],
    meta_analysis: &'a Option<ReplicationMetaAnalysis>,
    conclusion: ReplicationSynthesisConclusion,
    analysis_implementation_sha256: &'a str,
    independent_analysis_sha256: &'a str,
    crosscheck_sha256: &'a str,
    public_release_uri: &'a str,
    completed_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationSynthesisIssue {
    WrongVersion {
        found: String,
    },
    InvalidProtocol,
    InvalidSiteRegistry,
    AuthorityMismatch,
    InvalidSourceResult {
        field: String,
    },
    SourceEndpointMismatch,
    InvalidExecutionRecord {
        site_id: String,
    },
    DuplicateSite {
        site_id: String,
    },
    UnknownOrInactiveSite {
        site_id: String,
    },
    MissingRegisteredSite {
        site_id: String,
    },
    InvalidDigest {
        field: String,
    },
    EmptyField {
        field: String,
    },
    DerivedFieldMismatch {
        field: String,
    },
    ConclusionMismatch {
        expected: ReplicationSynthesisConclusion,
        found: ReplicationSynthesisConclusion,
    },
    SerializationFailed,
    SynthesisDigestMismatch,
}

pub fn replication_synthesis_commitment(
    record: &ReplicationSynthesisRecord,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SynthesisCommitment {
        synthesis_version: &record.synthesis_version,
        replication_id: &record.replication_id,
        protocol_sha256: &record.protocol_sha256,
        site_registry_sha256: &record.site_registry_sha256,
        source_result: &record.source_result,
        site_execution_sha256: &record.site_execution_sha256,
        quantitative_site_ids: &record.quantitative_site_ids,
        excluded_site_ids: &record.excluded_site_ids,
        meta_analysis: &record.meta_analysis,
        conclusion: record.conclusion,
        analysis_implementation_sha256: &record.analysis_implementation_sha256,
        independent_analysis_sha256: &record.independent_analysis_sha256,
        crosscheck_sha256: &record.crosscheck_sha256,
        public_release_uri: &record.public_release_uri,
        completed_at_utc: &record.completed_at_utc,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn synthesize_replications(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    source_result: PublishedSourcePrimaryResult,
    records: &[ReplicationSiteExecutionRecord],
    analysis_implementation_sha256: String,
    independent_analysis_sha256: String,
    crosscheck_sha256: String,
    public_release_uri: String,
    completed_at_utc: String,
) -> Result<ReplicationSynthesisRecord, Vec<ReplicationSynthesisIssue>> {
    let mut ordered = records.to_vec();
    ordered.sort_by(|left, right| left.site_id.cmp(&right.site_id));
    let quantitative = ordered
        .iter()
        .filter(|record| {
            matches!(
                record.conclusion,
                SiteReplicationConclusion::SupportsReplication
                    | SiteReplicationConclusion::DoesNotSupportReplication
                    | SiteReplicationConclusion::Inconclusive
            ) && record.primary_result.is_some()
        })
        .collect::<Vec<_>>();
    let excluded = ordered
        .iter()
        .filter(|record| {
            matches!(
                record.conclusion,
                SiteReplicationConclusion::DescriptiveOnly
                    | SiteReplicationConclusion::NonEstimable
            )
        })
        .map(|record| record.site_id.clone())
        .collect::<Vec<_>>();
    let meta_analysis = compute_meta_analysis(&source_result, &quantitative);
    let conclusion =
        derive_synthesis_conclusion(protocol, registry, &ordered, meta_analysis.as_ref());
    let mut record = ReplicationSynthesisRecord {
        synthesis_version: REPLICATION_SYNTHESIS_VERSION.into(),
        replication_id: protocol.replication_id.clone(),
        protocol_sha256: protocol.protocol_sha256.clone(),
        site_registry_sha256: registry.registry_sha256.clone(),
        source_result,
        site_execution_sha256: ordered
            .iter()
            .map(|item| item.record_sha256.clone())
            .collect(),
        quantitative_site_ids: quantitative
            .iter()
            .map(|item| item.site_id.clone())
            .collect(),
        excluded_site_ids: excluded,
        meta_analysis,
        conclusion,
        analysis_implementation_sha256,
        independent_analysis_sha256,
        crosscheck_sha256,
        public_release_uri,
        completed_at_utc,
        synthesis_sha256: String::new(),
    };
    record.synthesis_sha256 = replication_synthesis_commitment(&record)
        .map_err(|_| vec![ReplicationSynthesisIssue::SerializationFailed])?;
    let issues = validate_replication_synthesis(protocol, registry, &ordered, &record);
    if issues.is_empty() {
        Ok(record)
    } else {
        Err(issues)
    }
}

pub fn validate_replication_synthesis(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    records: &[ReplicationSiteExecutionRecord],
    synthesis: &ReplicationSynthesisRecord,
) -> Vec<ReplicationSynthesisIssue> {
    let mut issues = Vec::new();
    if synthesis.synthesis_version != REPLICATION_SYNTHESIS_VERSION {
        issues.push(ReplicationSynthesisIssue::WrongVersion {
            found: synthesis.synthesis_version.clone(),
        });
    }
    let protocol_digest = match replication_protocol_commitment(protocol) {
        Ok(value) if value == protocol.protocol_sha256 => value,
        _ => {
            issues.push(ReplicationSynthesisIssue::InvalidProtocol);
            String::new()
        }
    };
    let registry_digest = match replication_site_registry_commitment(registry) {
        Ok(value) if value == registry.registry_sha256 => value,
        _ => {
            issues.push(ReplicationSynthesisIssue::InvalidSiteRegistry);
            String::new()
        }
    };
    if synthesis.replication_id != protocol.replication_id
        || synthesis.protocol_sha256 != protocol_digest
        || synthesis.site_registry_sha256 != registry_digest
    {
        issues.push(ReplicationSynthesisIssue::AuthorityMismatch);
    }
    validate_source_result(protocol, &synthesis.source_result, &mut issues);
    let registered = registry
        .sites
        .iter()
        .filter(|site| site.site_status == ReplicationSiteStatus::Registered)
        .map(|site| site.site_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for record in records {
        if !seen.insert(record.site_id.as_str()) {
            issues.push(ReplicationSynthesisIssue::DuplicateSite {
                site_id: record.site_id.clone(),
            });
        }
        if !registered.contains(record.site_id.as_str()) {
            issues.push(ReplicationSynthesisIssue::UnknownOrInactiveSite {
                site_id: record.site_id.clone(),
            });
        }
        match replication_execution_commitment(record) {
            Ok(digest) if digest == record.record_sha256 => {}
            _ => issues.push(ReplicationSynthesisIssue::InvalidExecutionRecord {
                site_id: record.site_id.clone(),
            }),
        }
    }
    for site_id in registered {
        if !seen.contains(site_id) {
            issues.push(ReplicationSynthesisIssue::MissingRegisteredSite {
                site_id: site_id.into(),
            });
        }
    }
    for (field, digest) in [
        ("protocol_sha256", synthesis.protocol_sha256.as_str()),
        (
            "site_registry_sha256",
            synthesis.site_registry_sha256.as_str(),
        ),
        (
            "source_result.source_final_release_sha256",
            synthesis.source_result.source_final_release_sha256.as_str(),
        ),
        (
            "analysis_implementation_sha256",
            synthesis.analysis_implementation_sha256.as_str(),
        ),
        (
            "independent_analysis_sha256",
            synthesis.independent_analysis_sha256.as_str(),
        ),
        ("crosscheck_sha256", synthesis.crosscheck_sha256.as_str()),
        ("synthesis_sha256", synthesis.synthesis_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationSynthesisIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for digest in &synthesis.site_execution_sha256 {
        if !is_sha256(digest) {
            issues.push(ReplicationSynthesisIssue::InvalidDigest {
                field: "site_execution_sha256".into(),
            });
        }
    }
    for (field, value) in [
        ("replication_id", synthesis.replication_id.as_str()),
        ("public_release_uri", synthesis.public_release_uri.as_str()),
        ("completed_at_utc", synthesis.completed_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ReplicationSynthesisIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    validate_derived_fields(protocol, registry, records, synthesis, &mut issues);
    match replication_synthesis_commitment(synthesis) {
        Ok(digest) if digest == synthesis.synthesis_sha256 => {}
        Ok(_) => issues.push(ReplicationSynthesisIssue::SynthesisDigestMismatch),
        Err(_) => issues.push(ReplicationSynthesisIssue::SerializationFailed),
    }
    issues
}

fn compute_meta_analysis(
    source: &PublishedSourcePrimaryResult,
    records: &[&ReplicationSiteExecutionRecord],
) -> Option<ReplicationMetaAnalysis> {
    if records.is_empty() {
        return None;
    }
    let values = records
        .iter()
        .filter_map(|record| record.primary_result.as_ref())
        .map(|result| {
            (
                result.estimate,
                result.standard_error,
                result.participant_count,
                result.family_count,
            )
        })
        .collect::<Vec<_>>();
    if values.len() != records.len()
        || values
            .iter()
            .any(|(estimate, se, _, _)| !estimate.is_finite() || !se.is_finite() || *se <= 0.0)
    {
        return None;
    }
    let fixed_weights = values
        .iter()
        .map(|(_, se, _, _)| 1.0 / (se * se))
        .collect::<Vec<_>>();
    let sum_w = fixed_weights.iter().sum::<f64>();
    let fixed_effect_estimate = values
        .iter()
        .zip(&fixed_weights)
        .map(|((estimate, _, _, _), weight)| estimate * weight)
        .sum::<f64>()
        / sum_w;
    let cochran_q = values
        .iter()
        .zip(&fixed_weights)
        .map(|((estimate, _, _, _), weight)| weight * (estimate - fixed_effect_estimate).powi(2))
        .sum::<f64>();
    let degrees_freedom = values.len().saturating_sub(1) as f64;
    let sum_w_squared = fixed_weights
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>();
    let c = sum_w - sum_w_squared / sum_w;
    let tau_squared = if c > 0.0 {
        ((cochran_q - degrees_freedom) / c).max(0.0)
    } else {
        0.0
    };
    let random_weights = values
        .iter()
        .map(|(_, se, _, _)| 1.0 / (se * se + tau_squared))
        .collect::<Vec<_>>();
    let random_sum_w = random_weights.iter().sum::<f64>();
    let random_effect_estimate = values
        .iter()
        .zip(&random_weights)
        .map(|((estimate, _, _, _), weight)| estimate * weight)
        .sum::<f64>()
        / random_sum_w;
    let random_effect_standard_error = (1.0 / random_sum_w).sqrt();
    let confidence_lower = random_effect_estimate - NORMAL_95_Z * random_effect_standard_error;
    let confidence_upper = random_effect_estimate + NORMAL_95_Z * random_effect_standard_error;
    let i_squared_percent = if cochran_q > 0.0 {
        ((cochran_q - degrees_freedom) / cochran_q).max(0.0) * 100.0
    } else {
        0.0
    };
    let source_attenuation_ratio = if source.estimate.abs() > f64::EPSILON {
        Some(random_effect_estimate / source.estimate)
    } else {
        None
    };
    Some(ReplicationMetaAnalysis {
        site_count: values.len() as u32,
        participant_count: values
            .iter()
            .map(|(_, _, participants, _)| participants)
            .sum(),
        family_count: values.iter().map(|(_, _, _, families)| families).sum(),
        fixed_effect_estimate,
        random_effect_estimate,
        random_effect_standard_error,
        confidence_lower,
        confidence_upper,
        cochran_q,
        tau_squared,
        i_squared_percent,
        source_attenuation_ratio,
        direction_concordant_with_source: random_effect_estimate.signum()
            == source.estimate.signum(),
    })
}

fn derive_synthesis_conclusion(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    records: &[ReplicationSiteExecutionRecord],
    meta: Option<&ReplicationMetaAnalysis>,
) -> ReplicationSynthesisConclusion {
    let registered_count = registry
        .sites
        .iter()
        .filter(|site| site.site_status == ReplicationSiteStatus::Registered)
        .count() as u32;
    if registered_count < protocol.required_site_count
        || (records.len() as u32) < protocol.required_site_count
    {
        return ReplicationSynthesisConclusion::InsufficientEvidence;
    }
    if records.iter().any(|record| {
        matches!(
            record.conclusion,
            SiteReplicationConclusion::DescriptiveOnly | SiteReplicationConclusion::NonEstimable
        )
    }) {
        return ReplicationSynthesisConclusion::DescriptiveOnly;
    }
    let Some(meta) = meta else {
        return ReplicationSynthesisConclusion::InsufficientEvidence;
    };
    let support_count = records
        .iter()
        .filter(|record| record.conclusion == SiteReplicationConclusion::SupportsReplication)
        .count();
    let margin = protocol.primary_endpoint.practical_margin;
    match protocol.primary_endpoint.favorable_direction {
        FavorableDirection::Higher => {
            if meta.confidence_lower >= margin && support_count >= 2 {
                ReplicationSynthesisConclusion::IndependentlyReplicated
            } else if meta.confidence_upper < margin && support_count == 0 {
                ReplicationSynthesisConclusion::DidNotReplicate
            } else {
                ReplicationSynthesisConclusion::MixedEvidence
            }
        }
        FavorableDirection::Lower => {
            let threshold = -margin;
            if meta.confidence_upper <= threshold && support_count >= 2 {
                ReplicationSynthesisConclusion::IndependentlyReplicated
            } else if meta.confidence_lower > threshold && support_count == 0 {
                ReplicationSynthesisConclusion::DidNotReplicate
            } else {
                ReplicationSynthesisConclusion::MixedEvidence
            }
        }
    }
}

fn validate_source_result(
    protocol: &FrozenReplicationProtocol,
    source: &PublishedSourcePrimaryResult,
    issues: &mut Vec<ReplicationSynthesisIssue>,
) {
    if source.endpoint_id != protocol.primary_endpoint.endpoint_id {
        issues.push(ReplicationSynthesisIssue::SourceEndpointMismatch);
    }
    for (field, valid) in [
        (
            "source_final_release_sha256",
            is_sha256(&source.source_final_release_sha256)
                && source.source_final_release_sha256 == protocol.source_final_release_sha256,
        ),
        ("estimate", source.estimate.is_finite()),
        (
            "standard_error",
            source.standard_error.is_finite() && source.standard_error > 0.0,
        ),
        ("confidence_lower", source.confidence_lower.is_finite()),
        ("confidence_upper", source.confidence_upper.is_finite()),
        (
            "confidence_order",
            source.confidence_lower <= source.estimate
                && source.estimate <= source.confidence_upper,
        ),
    ] {
        if !valid {
            issues.push(ReplicationSynthesisIssue::InvalidSourceResult {
                field: field.into(),
            });
        }
    }
}

fn validate_derived_fields(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    records: &[ReplicationSiteExecutionRecord],
    synthesis: &ReplicationSynthesisRecord,
    issues: &mut Vec<ReplicationSynthesisIssue>,
) {
    let mut ordered = records.to_vec();
    ordered.sort_by(|left, right| left.site_id.cmp(&right.site_id));
    let quantitative = ordered
        .iter()
        .filter(|record| {
            matches!(
                record.conclusion,
                SiteReplicationConclusion::SupportsReplication
                    | SiteReplicationConclusion::DoesNotSupportReplication
                    | SiteReplicationConclusion::Inconclusive
            ) && record.primary_result.is_some()
        })
        .collect::<Vec<_>>();
    let expected_digests = ordered
        .iter()
        .map(|record| record.record_sha256.clone())
        .collect::<Vec<_>>();
    let expected_quantitative = quantitative
        .iter()
        .map(|record| record.site_id.clone())
        .collect::<Vec<_>>();
    let expected_excluded = ordered
        .iter()
        .filter(|record| {
            matches!(
                record.conclusion,
                SiteReplicationConclusion::DescriptiveOnly
                    | SiteReplicationConclusion::NonEstimable
            )
        })
        .map(|record| record.site_id.clone())
        .collect::<Vec<_>>();
    if synthesis.site_execution_sha256 != expected_digests {
        issues.push(ReplicationSynthesisIssue::DerivedFieldMismatch {
            field: "site_execution_sha256".into(),
        });
    }
    if synthesis.quantitative_site_ids != expected_quantitative {
        issues.push(ReplicationSynthesisIssue::DerivedFieldMismatch {
            field: "quantitative_site_ids".into(),
        });
    }
    if synthesis.excluded_site_ids != expected_excluded {
        issues.push(ReplicationSynthesisIssue::DerivedFieldMismatch {
            field: "excluded_site_ids".into(),
        });
    }
    let expected_meta = compute_meta_analysis(&synthesis.source_result, &quantitative);
    if synthesis.meta_analysis != expected_meta {
        issues.push(ReplicationSynthesisIssue::DerivedFieldMismatch {
            field: "meta_analysis".into(),
        });
    }
    let expected_conclusion =
        derive_synthesis_conclusion(protocol, registry, &ordered, expected_meta.as_ref());
    if synthesis.conclusion != expected_conclusion {
        issues.push(ReplicationSynthesisIssue::ConclusionMismatch {
            expected: expected_conclusion,
            found: synthesis.conclusion,
        });
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execution(site: &str, estimate: f64, standard_error: f64) -> ReplicationSiteExecutionRecord {
        ReplicationSiteExecutionRecord {
            record_version: "symthaea-muse-replication-execution-v1".into(),
            replication_id: "rep".into(),
            site_id: site.into(),
            protocol_sha256: "a".repeat(64),
            site_registry_sha256: "b".repeat(64),
            site_package_sha256: "c".repeat(64),
            package_receipt_sha256: "d".repeat(64),
            collection_close_sha256: "e".repeat(64),
            dataset_sha256: "f".repeat(64),
            analysis_plan_sha256: "1".repeat(64),
            primary_analysis_sha256: "2".repeat(64),
            independent_analysis_sha256: "3".repeat(64),
            analysis_crosscheck_sha256: "4".repeat(64),
            execution_environment_sha256: "5".repeat(64),
            source_outcome_access_audit_sha256: "6".repeat(64),
            deviations: Vec::new(),
            primary_result: Some(crate::replication_execution::SitePrimaryResult {
                endpoint_id: "primary".into(),
                estimate,
                standard_error,
                confidence_lower: estimate - NORMAL_95_Z * standard_error,
                confidence_upper: estimate + NORMAL_95_Z * standard_error,
                p_value: 0.01,
                participant_count: 48,
                family_count: 24,
            }),
            conclusion: SiteReplicationConclusion::SupportsReplication,
            all_frozen_commands_succeeded: true,
            collection_blinded_until_close: true,
            source_outcomes_withheld_until_close: true,
            public_release_uri: "uri".into(),
            completed_at_utc: "now".into(),
            record_sha256: "7".repeat(64),
        }
    }

    #[test]
    fn identical_sites_have_zero_heterogeneity() {
        let source = PublishedSourcePrimaryResult {
            source_final_release_sha256: "a".repeat(64),
            endpoint_id: "primary".into(),
            estimate: 0.10,
            standard_error: 0.02,
            confidence_lower: 0.06,
            confidence_upper: 0.14,
        };
        let left = execution("a", 0.10, 0.02);
        let right = execution("b", 0.10, 0.02);
        let meta = compute_meta_analysis(&source, &[&left, &right]).unwrap();
        assert!((meta.random_effect_estimate - 0.10).abs() < 1e-12);
        assert!(meta.tau_squared.abs() < 1e-12);
        assert!(meta.i_squared_percent.abs() < 1e-12);
    }

    #[test]
    fn conflicting_sites_report_heterogeneity() {
        let source = PublishedSourcePrimaryResult {
            source_final_release_sha256: "a".repeat(64),
            endpoint_id: "primary".into(),
            estimate: 0.10,
            standard_error: 0.02,
            confidence_lower: 0.06,
            confidence_upper: 0.14,
        };
        let left = execution("a", 0.15, 0.02);
        let right = execution("b", -0.05, 0.02);
        let meta = compute_meta_analysis(&source, &[&left, &right]).unwrap();
        assert!(meta.tau_squared > 0.0);
        assert!(meta.i_squared_percent > 0.0);
    }
}
