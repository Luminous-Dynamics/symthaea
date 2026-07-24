// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent primary-analysis agreement gate.
//!
//! A result cannot enter the release as confirmatory evidence unless the Rust
//! implementation and a separately maintained external implementation agree on
//! the frozen input, comparator estimates, intervals, adjusted p-values, and
//! final gates within preregistered numerical tolerances.

use crate::confirmatory_analysis::ConfirmatoryComparator;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::ConfirmatoryEndpoint;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ANALYSIS_CROSSCHECK_VERSION: &str = "symthaea-muse-analysis-crosscheck-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnalysisEngineKind {
    RustPrimary,
    IndependentExternal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedComparatorResult {
    pub comparator: ConfirmatoryComparator,
    pub estimate: f64,
    pub confidence_interval: [f64; 2],
    pub required_margin: f64,
    pub raw_one_sided_p: f64,
    pub adjusted_p: f64,
    pub margin_gate_passed: bool,
    pub inferential_gate_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedPrimaryAnalysis {
    pub engine_kind: AnalysisEngineKind,
    pub engine_name: String,
    pub engine_version: String,
    pub source_sha256: String,
    pub environment_sha256: String,
    pub input_sha256: String,
    pub analysis_plan_sha256: String,
    pub endpoint: ConfirmatoryEndpoint,
    pub alpha: f64,
    pub comparisons: Vec<NormalizedComparatorResult>,
    pub success: bool,
    pub output_sha256: String,
}

#[derive(Serialize)]
struct AnalysisCommitment<'a> {
    engine_kind: AnalysisEngineKind,
    engine_name: &'a str,
    engine_version: &'a str,
    source_sha256: &'a str,
    environment_sha256: &'a str,
    input_sha256: &'a str,
    analysis_plan_sha256: &'a str,
    endpoint: ConfirmatoryEndpoint,
    alpha: f64,
    comparisons: &'a [NormalizedComparatorResult],
    success: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisAgreementTolerance {
    pub estimate_absolute: f64,
    pub interval_absolute: f64,
    pub margin_absolute: f64,
    pub p_value_absolute: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparatorAgreement {
    pub comparator: ConfirmatoryComparator,
    pub estimate_difference: f64,
    pub lower_interval_difference: f64,
    pub upper_interval_difference: f64,
    pub margin_difference: f64,
    pub raw_p_difference: f64,
    pub adjusted_p_difference: f64,
    pub gates_match: bool,
    pub within_tolerance: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisCrosscheckReport {
    pub crosscheck_version: String,
    pub rust_output_sha256: String,
    pub external_output_sha256: String,
    pub tolerance: AnalysisAgreementTolerance,
    pub agreements: Vec<ComparatorAgreement>,
    pub exact_identity_fields_match: bool,
    pub success_decision_matches: bool,
    pub passed: bool,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct CrosscheckCommitment<'a> {
    crosscheck_version: &'a str,
    rust_output_sha256: &'a str,
    external_output_sha256: &'a str,
    tolerance: &'a AnalysisAgreementTolerance,
    agreements: &'a [ComparatorAgreement],
    exact_identity_fields_match: bool,
    success_decision_matches: bool,
    passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisCrosscheckIssue {
    WrongEngineKind {
        expected: AnalysisEngineKind,
        found: AnalysisEngineKind,
    },
    EmptyEngineField {
        engine: AnalysisEngineKind,
        field: String,
    },
    InvalidDigest {
        engine: AnalysisEngineKind,
        field: String,
    },
    InvalidNumericValue {
        engine: AnalysisEngineKind,
        field: String,
    },
    InvalidTolerance {
        field: String,
    },
    DuplicateComparator {
        engine: AnalysisEngineKind,
        comparator: ConfirmatoryComparator,
    },
    MissingComparator {
        engine: AnalysisEngineKind,
        comparator: ConfirmatoryComparator,
    },
    OutputDigestMismatch {
        engine: AnalysisEngineKind,
    },
    SerializationFailed {
        field: String,
    },
    ReportDigestMismatch,
    ReportContentMismatch,
}

pub fn normalized_primary_analysis_commitment(
    analysis: &NormalizedPrimaryAnalysis,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&AnalysisCommitment {
        engine_kind: analysis.engine_kind,
        engine_name: &analysis.engine_name,
        engine_version: &analysis.engine_version,
        source_sha256: &analysis.source_sha256,
        environment_sha256: &analysis.environment_sha256,
        input_sha256: &analysis.input_sha256,
        analysis_plan_sha256: &analysis.analysis_plan_sha256,
        endpoint: analysis.endpoint,
        alpha: analysis.alpha,
        comparisons: &analysis.comparisons,
        success: analysis.success,
    })
}

pub fn seal_normalized_primary_analysis(
    analysis: &mut NormalizedPrimaryAnalysis,
) -> Result<(), serde_json::Error> {
    analysis
        .comparisons
        .sort_by_key(|comparison| comparison.comparator);
    analysis.output_sha256 = normalized_primary_analysis_commitment(analysis)?;
    Ok(())
}

pub fn analysis_crosscheck_commitment(
    report: &AnalysisCrosscheckReport,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&CrosscheckCommitment {
        crosscheck_version: &report.crosscheck_version,
        rust_output_sha256: &report.rust_output_sha256,
        external_output_sha256: &report.external_output_sha256,
        tolerance: &report.tolerance,
        agreements: &report.agreements,
        exact_identity_fields_match: report.exact_identity_fields_match,
        success_decision_matches: report.success_decision_matches,
        passed: report.passed,
    })
}

pub fn crosscheck_primary_analyses(
    rust: &NormalizedPrimaryAnalysis,
    external: &NormalizedPrimaryAnalysis,
    tolerance: AnalysisAgreementTolerance,
) -> Result<AnalysisCrosscheckReport, Vec<AnalysisCrosscheckIssue>> {
    let mut issues = validate_analysis(rust, AnalysisEngineKind::RustPrimary);
    issues.extend(validate_analysis(
        external,
        AnalysisEngineKind::IndependentExternal,
    ));
    for (field, value) in [
        ("estimate_absolute", tolerance.estimate_absolute),
        ("interval_absolute", tolerance.interval_absolute),
        ("margin_absolute", tolerance.margin_absolute),
        ("p_value_absolute", tolerance.p_value_absolute),
    ] {
        if !value.is_finite() || value < 0.0 {
            issues.push(AnalysisCrosscheckIssue::InvalidTolerance {
                field: field.into(),
            });
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    let rust_by_comparator: BTreeMap<_, _> = rust
        .comparisons
        .iter()
        .map(|comparison| (comparison.comparator, comparison))
        .collect();
    let external_by_comparator: BTreeMap<_, _> = external
        .comparisons
        .iter()
        .map(|comparison| (comparison.comparator, comparison))
        .collect();
    let mut agreements = Vec::new();
    for comparator in ConfirmatoryComparator::ALL {
        let left = rust_by_comparator[&comparator];
        let right = external_by_comparator[&comparator];
        let estimate_difference = (left.estimate - right.estimate).abs();
        let lower_interval_difference =
            (left.confidence_interval[0] - right.confidence_interval[0]).abs();
        let upper_interval_difference =
            (left.confidence_interval[1] - right.confidence_interval[1]).abs();
        let margin_difference = (left.required_margin - right.required_margin).abs();
        let raw_p_difference = (left.raw_one_sided_p - right.raw_one_sided_p).abs();
        let adjusted_p_difference = (left.adjusted_p - right.adjusted_p).abs();
        let gates_match = left.margin_gate_passed == right.margin_gate_passed
            && left.inferential_gate_passed == right.inferential_gate_passed;
        let within_tolerance = estimate_difference <= tolerance.estimate_absolute
            && lower_interval_difference <= tolerance.interval_absolute
            && upper_interval_difference <= tolerance.interval_absolute
            && margin_difference <= tolerance.margin_absolute
            && raw_p_difference <= tolerance.p_value_absolute
            && adjusted_p_difference <= tolerance.p_value_absolute
            && gates_match;
        agreements.push(ComparatorAgreement {
            comparator,
            estimate_difference,
            lower_interval_difference,
            upper_interval_difference,
            margin_difference,
            raw_p_difference,
            adjusted_p_difference,
            gates_match,
            within_tolerance,
        });
    }
    let exact_identity_fields_match = rust.input_sha256 == external.input_sha256
        && rust.analysis_plan_sha256 == external.analysis_plan_sha256
        && rust.endpoint == external.endpoint
        && rust.alpha.to_bits() == external.alpha.to_bits();
    let success_decision_matches = rust.success == external.success;
    let passed = exact_identity_fields_match
        && success_decision_matches
        && agreements
            .iter()
            .all(|agreement| agreement.within_tolerance);
    let mut report = AnalysisCrosscheckReport {
        crosscheck_version: ANALYSIS_CROSSCHECK_VERSION.into(),
        rust_output_sha256: rust.output_sha256.clone(),
        external_output_sha256: external.output_sha256.clone(),
        tolerance,
        agreements,
        exact_identity_fields_match,
        success_decision_matches,
        passed,
        report_sha256: String::new(),
    };
    report.report_sha256 = analysis_crosscheck_commitment(&report).map_err(|_| {
        vec![AnalysisCrosscheckIssue::SerializationFailed {
            field: "crosscheck_report".into(),
        }]
    })?;
    Ok(report)
}

pub fn validate_analysis_crosscheck(
    rust: &NormalizedPrimaryAnalysis,
    external: &NormalizedPrimaryAnalysis,
    report: &AnalysisCrosscheckReport,
) -> Vec<AnalysisCrosscheckIssue> {
    let mut issues = validate_analysis(rust, AnalysisEngineKind::RustPrimary);
    issues.extend(validate_analysis(
        external,
        AnalysisEngineKind::IndependentExternal,
    ));
    if report.crosscheck_version != ANALYSIS_CROSSCHECK_VERSION {
        issues.push(AnalysisCrosscheckIssue::SerializationFailed {
            field: "crosscheck_version".into(),
        });
    }
    if report.rust_output_sha256 != rust.output_sha256
        || report.external_output_sha256 != external.output_sha256
    {
        issues.push(AnalysisCrosscheckIssue::SerializationFailed {
            field: "analysis_output_binding".into(),
        });
    }
    match crosscheck_primary_analyses(rust, external, report.tolerance.clone()) {
        Ok(rebuilt) if rebuilt == *report => {}
        Ok(_) => issues.push(AnalysisCrosscheckIssue::ReportContentMismatch),
        Err(mut found) => issues.append(&mut found),
    }
    match analysis_crosscheck_commitment(report) {
        Ok(value) if value == report.report_sha256 => {}
        Ok(_) => issues.push(AnalysisCrosscheckIssue::ReportDigestMismatch),
        Err(_) => issues.push(AnalysisCrosscheckIssue::SerializationFailed {
            field: "crosscheck_report".into(),
        }),
    }
    issues
}

fn validate_analysis(
    analysis: &NormalizedPrimaryAnalysis,
    expected_kind: AnalysisEngineKind,
) -> Vec<AnalysisCrosscheckIssue> {
    let mut issues = Vec::new();
    if analysis.engine_kind != expected_kind {
        issues.push(AnalysisCrosscheckIssue::WrongEngineKind {
            expected: expected_kind,
            found: analysis.engine_kind,
        });
    }
    for (field, value) in [
        ("engine_name", analysis.engine_name.as_str()),
        ("engine_version", analysis.engine_version.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(AnalysisCrosscheckIssue::EmptyEngineField {
                engine: analysis.engine_kind,
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("source_sha256", analysis.source_sha256.as_str()),
        ("environment_sha256", analysis.environment_sha256.as_str()),
        ("input_sha256", analysis.input_sha256.as_str()),
        (
            "analysis_plan_sha256",
            analysis.analysis_plan_sha256.as_str(),
        ),
        ("output_sha256", analysis.output_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(AnalysisCrosscheckIssue::InvalidDigest {
                engine: analysis.engine_kind,
                field: field.into(),
            });
        }
    }
    if !analysis.alpha.is_finite() || analysis.alpha <= 0.0 || analysis.alpha > 0.10 {
        issues.push(AnalysisCrosscheckIssue::InvalidNumericValue {
            engine: analysis.engine_kind,
            field: "alpha".into(),
        });
    }
    let mut comparators = BTreeSet::new();
    for comparison in &analysis.comparisons {
        if !comparators.insert(comparison.comparator) {
            issues.push(AnalysisCrosscheckIssue::DuplicateComparator {
                engine: analysis.engine_kind,
                comparator: comparison.comparator,
            });
        }
        for (field, value) in [
            ("estimate", comparison.estimate),
            (
                "confidence_interval.lower",
                comparison.confidence_interval[0],
            ),
            (
                "confidence_interval.upper",
                comparison.confidence_interval[1],
            ),
            ("required_margin", comparison.required_margin),
            ("raw_one_sided_p", comparison.raw_one_sided_p),
            ("adjusted_p", comparison.adjusted_p),
        ] {
            let invalid = !value.is_finite()
                || ((field == "raw_one_sided_p" || field == "adjusted_p")
                    && !(0.0..=1.0).contains(&value));
            if invalid {
                issues.push(AnalysisCrosscheckIssue::InvalidNumericValue {
                    engine: analysis.engine_kind,
                    field: format!("{:?}.{field}", comparison.comparator),
                });
            }
        }
    }
    for comparator in ConfirmatoryComparator::ALL {
        if !comparators.contains(&comparator) {
            issues.push(AnalysisCrosscheckIssue::MissingComparator {
                engine: analysis.engine_kind,
                comparator,
            });
        }
    }
    match normalized_primary_analysis_commitment(analysis) {
        Ok(value) if value == analysis.output_sha256 => {}
        Ok(_) => issues.push(AnalysisCrosscheckIssue::OutputDigestMismatch {
            engine: analysis.engine_kind,
        }),
        Err(_) => issues.push(AnalysisCrosscheckIssue::SerializationFailed {
            field: format!("analysis.{:?}", analysis.engine_kind),
        }),
    }
    issues
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis(kind: AnalysisEngineKind, estimate_shift: f64) -> NormalizedPrimaryAnalysis {
        let mut analysis = NormalizedPrimaryAnalysis {
            engine_kind: kind,
            engine_name: format!("{kind:?}"),
            engine_version: "1".into(),
            source_sha256: "a".repeat(64),
            environment_sha256: "b".repeat(64),
            input_sha256: "c".repeat(64),
            analysis_plan_sha256: "d".repeat(64),
            endpoint: ConfirmatoryEndpoint::Preference,
            alpha: 0.05,
            comparisons: ConfirmatoryComparator::ALL
                .into_iter()
                .map(|comparator| NormalizedComparatorResult {
                    comparator,
                    estimate: 0.1 + estimate_shift,
                    confidence_interval: [0.05 + estimate_shift, 0.15 + estimate_shift],
                    required_margin: 0.02,
                    raw_one_sided_p: 0.01,
                    adjusted_p: 0.03,
                    margin_gate_passed: true,
                    inferential_gate_passed: true,
                })
                .collect(),
            success: true,
            output_sha256: String::new(),
        };
        seal_normalized_primary_analysis(&mut analysis).unwrap();
        analysis
    }

    #[test]
    fn crosscheck_rejects_material_numeric_disagreement() {
        let rust = analysis(AnalysisEngineKind::RustPrimary, 0.0);
        let external = analysis(AnalysisEngineKind::IndependentExternal, 0.1);
        let report = crosscheck_primary_analyses(
            &rust,
            &external,
            AnalysisAgreementTolerance {
                estimate_absolute: 1e-9,
                interval_absolute: 1e-9,
                margin_absolute: 1e-9,
                p_value_absolute: 1e-9,
            },
        )
        .unwrap();
        assert!(!report.passed);
    }
}
