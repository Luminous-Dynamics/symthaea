// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Release closure for causal evidence, study quality, semantic invariance, and
//! metric lifecycle governance.

use crate::{
    CausalAuthority, CausalError, CausalReport, IntegrityError, MetricLifecycleDecision,
    MetricLifecycleError, MetricLifecycleReport, MetricRegistry, MetricState,
    SemanticComparability, SemanticDriftError, SemanticDriftReport, StudyQualityError,
    StudyQualityOutcome, StudyQualityReport, digest_json, is_stable_digest, load_json,
    save_json_atomic,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const SCIENTIFIC_VALIDATION_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const SCIENTIFIC_VALIDATION_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScientificClaimKind {
    CausalBenefit,
    QuasiExperimentalBenefit,
    AssociationalSignal,
    CrossContextComparability,
    OperationalMetricUse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificClaim {
    pub claim_id: String,
    pub metric_id: String,
    pub kind: ScientificClaimKind,
    pub causal_study_id: Option<String>,
    pub semantic_baseline_set_id: Option<String>,
    pub semantic_candidate_set_id: Option<String>,
}

impl ScientificClaim {
    fn validate(&self) -> Result<(), ScientificValidationError> {
        validate_identifier(&self.claim_id, "claim id")?;
        validate_identifier(&self.metric_id, "metric id")?;
        if let Some(id) = &self.causal_study_id {
            validate_identifier(id, "causal study id")?;
        }
        if let Some(id) = &self.semantic_baseline_set_id {
            validate_identifier(id, "semantic baseline set id")?;
        }
        if let Some(id) = &self.semantic_candidate_set_id {
            validate_identifier(id, "semantic candidate set id")?;
        }
        match self.kind {
            ScientificClaimKind::CausalBenefit
            | ScientificClaimKind::QuasiExperimentalBenefit
            | ScientificClaimKind::AssociationalSignal
                if self.causal_study_id.is_none() =>
            {
                Err(ScientificValidationError::MissingClaimEvidence(
                    self.claim_id.clone(),
                ))
            }
            ScientificClaimKind::CrossContextComparability
                if self.semantic_baseline_set_id.is_none()
                    || self.semantic_candidate_set_id.is_none() =>
            {
                Err(ScientificValidationError::MissingClaimEvidence(
                    self.claim_id.clone(),
                ))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificValidationBundle {
    pub schema_version: u32,
    pub bundle_id: String,
    pub release_id: String,
    pub prior_epistemic_release_id: String,
    pub prior_epistemic_release_digest: String,
    pub metric_registry: MetricRegistry,
    pub metric_reports: Vec<MetricLifecycleReport>,
    pub causal_reports: Vec<CausalReport>,
    pub study_quality_reports: Vec<StudyQualityReport>,
    pub semantic_drift_reports: Vec<SemanticDriftReport>,
    pub claims: Vec<ScientificClaim>,
}

impl ScientificValidationBundle {
    pub fn validate(&self) -> Result<(), ScientificValidationError> {
        if self.schema_version == 0
            || self.schema_version > SCIENTIFIC_VALIDATION_BUNDLE_SCHEMA_VERSION
        {
            return Err(ScientificValidationError::UnsupportedBundleSchema(
                self.schema_version,
            ));
        }
        validate_identifier(&self.bundle_id, "bundle id")?;
        validate_identifier(&self.release_id, "release id")?;
        validate_identifier(&self.prior_epistemic_release_id, "prior epistemic release id")?;
        if !is_stable_digest(&self.prior_epistemic_release_digest) {
            return Err(ScientificValidationError::InvalidPriorDigest);
        }
        self.metric_registry
            .validate()
            .map_err(ScientificValidationError::Metric)?;
        if self.claims.is_empty() {
            return Err(ScientificValidationError::EmptyClaims);
        }

        let mut metric_reports = BTreeMap::new();
        for report in &self.metric_reports {
            report
                .validate(&self.metric_registry)
                .map_err(ScientificValidationError::Metric)?;
            if metric_reports
                .insert(report.metric.metric_id.as_str(), report)
                .is_some()
            {
                return Err(ScientificValidationError::DuplicateMetricReport(
                    report.metric.metric_id.clone(),
                ));
            }
        }
        let mut causal = BTreeMap::new();
        for report in &self.causal_reports {
            report.validate().map_err(ScientificValidationError::Causal)?;
            if causal
                .insert(report.study.study_id.as_str(), report)
                .is_some()
            {
                return Err(ScientificValidationError::DuplicateCausalStudy(
                    report.study.study_id.clone(),
                ));
            }
        }
        let mut quality = BTreeMap::new();
        for report in &self.study_quality_reports {
            report
                .validate()
                .map_err(ScientificValidationError::StudyQuality)?;
            if quality
                .insert(report.execution.study_id.as_str(), report)
                .is_some()
            {
                return Err(ScientificValidationError::DuplicateQualityStudy(
                    report.execution.study_id.clone(),
                ));
            }
        }
        for (study_id, causal_report) in &causal {
            let quality_report = quality
                .get(study_id)
                .copied()
                .ok_or_else(|| ScientificValidationError::MissingStudyQuality((*study_id).to_owned()))?;
            if causal_report.study.preregistration_id
                != quality_report.preregistration.preregistration_id
            {
                return Err(ScientificValidationError::PreregistrationBindingMismatch(
                    (*study_id).to_owned(),
                ));
            }
            if causal_report.study.collection_started_unix_ms
                != quality_report.execution.collection_started_unix_ms
                || causal_report.study.collection_ended_unix_ms
                    != quality_report.execution.collection_ended_unix_ms
            {
                return Err(ScientificValidationError::CollectionBindingMismatch(
                    (*study_id).to_owned(),
                ));
            }
        }
        if quality.keys().any(|study_id| !causal.contains_key(study_id)) {
            return Err(ScientificValidationError::OrphanStudyQuality);
        }

        let mut semantic = BTreeMap::new();
        for report in &self.semantic_drift_reports {
            report
                .validate()
                .map_err(ScientificValidationError::Semantic)?;
            let key = (
                report.baseline.set_id.as_str(),
                report.candidate.set_id.as_str(),
            );
            if semantic.insert(key, report).is_some() {
                return Err(ScientificValidationError::DuplicateSemanticComparison(
                    report.baseline.set_id.clone(),
                    report.candidate.set_id.clone(),
                ));
            }
        }

        let mut claim_ids = BTreeSet::new();
        let mut referenced_metrics = BTreeSet::new();
        let mut referenced_causal = BTreeSet::new();
        let mut referenced_semantic = BTreeSet::new();
        for claim in &self.claims {
            claim.validate()?;
            if !claim_ids.insert(claim.claim_id.as_str()) {
                return Err(ScientificValidationError::DuplicateClaim(
                    claim.claim_id.clone(),
                ));
            }
            if self.metric_registry.metric(&claim.metric_id).is_none() {
                return Err(ScientificValidationError::UnknownClaimMetric(
                    claim.metric_id.clone(),
                ));
            }
            if !metric_reports.contains_key(claim.metric_id.as_str()) {
                return Err(ScientificValidationError::MissingMetricReport(
                    claim.metric_id.clone(),
                ));
            }
            referenced_metrics.insert(claim.metric_id.as_str());
            if let Some(study_id) = &claim.causal_study_id {
                let report = causal.get(study_id.as_str()).copied().ok_or_else(|| {
                    ScientificValidationError::UnknownCausalStudy(study_id.clone())
                })?;
                referenced_causal.insert(study_id.as_str());
                if report.study.intervention_metric_id != claim.metric_id
                    && report.study.outcome_metric_id != claim.metric_id
                {
                    return Err(ScientificValidationError::ClaimMetricBindingMismatch(
                        claim.claim_id.clone(),
                    ));
                }
            }
            if let (Some(baseline), Some(candidate)) = (
                claim.semantic_baseline_set_id.as_deref(),
                claim.semantic_candidate_set_id.as_deref(),
            ) {
                let report = semantic.get(&(baseline, candidate)).copied().ok_or_else(|| {
                    ScientificValidationError::UnknownSemanticComparison(
                        baseline.to_owned(),
                        candidate.to_owned(),
                    )
                })?;
                referenced_semantic.insert((baseline, candidate));
                if report.baseline.metric_id != claim.metric_id {
                    return Err(ScientificValidationError::ClaimMetricBindingMismatch(
                        claim.claim_id.clone(),
                    ));
                }
            }
        }
        if metric_reports.keys().any(|metric_id| !referenced_metrics.contains(metric_id)) {
            return Err(ScientificValidationError::OrphanMetricEvidence);
        }
        if causal.keys().any(|study_id| !referenced_causal.contains(study_id)) {
            return Err(ScientificValidationError::OrphanCausalEvidence);
        }
        if semantic.keys().any(|comparison| !referenced_semantic.contains(comparison)) {
            return Err(ScientificValidationError::OrphanSemanticEvidence);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ScientificValidationError> {
        self.validate()?;
        digest_json(self).map_err(ScientificValidationError::Integrity)
    }

    pub fn save(&self, path: &Path) -> Result<(), ScientificValidationError> {
        self.validate()?;
        save_json_atomic::<_, ScientificValidationError>(path, self)
    }

    pub fn load(path: &Path) -> Result<Self, ScientificValidationError> {
        let bundle: Self = load_json::<_, ScientificValidationError>(path)?;
        bundle.validate()?;
        Ok(bundle)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScientificValidationCriteria {
    pub require_high_quality_for_causal_claims: bool,
    pub allow_quasi_experimental_claims: bool,
    pub allow_drifted_but_comparable_semantics: bool,
    pub allow_shadow_metrics: bool,
    pub block_deprecated_metrics: bool,
}

impl ScientificValidationCriteria {
    pub const fn production() -> Self {
        Self {
            require_high_quality_for_causal_claims: true,
            allow_quasi_experimental_claims: false,
            allow_drifted_but_comparable_semantics: true,
            allow_shadow_metrics: false,
            block_deprecated_metrics: true,
        }
    }
}

impl Default for ScientificValidationCriteria {
    fn default() -> Self { Self::production() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScientificValidationFindingCode {
    PriorEpistemicBinding,
    EvidenceBinding,
    StudyQuality,
    CausalAuthority,
    SemanticComparability,
    MetricValidity,
    MetricMigration,
}

impl ScientificValidationFindingCode {
    pub const ALL: [Self; 7] = [
        Self::PriorEpistemicBinding,
        Self::EvidenceBinding,
        Self::StudyQuality,
        Self::CausalAuthority,
        Self::SemanticComparability,
        Self::MetricValidity,
        Self::MetricMigration,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScientificValidationFindingStatus { Pass, Warning, Fail }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificValidationFinding {
    pub code: ScientificValidationFindingCode,
    pub status: ScientificValidationFindingStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScientificValidationOutcome { Ready, HumanReview, Blocked }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificValidationReport {
    pub schema_version: u32,
    pub report_id: String,
    pub bundle: ScientificValidationBundle,
    pub criteria: ScientificValidationCriteria,
    pub outcome: ScientificValidationOutcome,
    pub findings: Vec<ScientificValidationFinding>,
}

impl ScientificValidationReport {
    pub fn validate(&self) -> Result<(), ScientificValidationError> {
        if self.schema_version == 0
            || self.schema_version > SCIENTIFIC_VALIDATION_REPORT_SCHEMA_VERSION
        {
            return Err(ScientificValidationError::UnsupportedReportSchema(
                self.schema_version,
            ));
        }
        let expected = evaluate_scientific_validation(
            self.report_id.clone(),
            self.bundle.clone(),
            self.criteria,
        )?;
        if self != &expected {
            return Err(ScientificValidationError::DerivedReportMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ScientificValidationError> {
        self.validate()?;
        digest_json(self).map_err(ScientificValidationError::Integrity)
    }
}

pub fn evaluate_scientific_validation(
    report_id: impl Into<String>,
    bundle: ScientificValidationBundle,
    criteria: ScientificValidationCriteria,
) -> Result<ScientificValidationReport, ScientificValidationError> {
    let report_id = report_id.into();
    validate_identifier(&report_id, "report id")?;
    bundle.validate()?;

    let causal_by_id = bundle
        .causal_reports
        .iter()
        .map(|report| (report.study.study_id.as_str(), report))
        .collect::<BTreeMap<_, _>>();
    let quality_by_id = bundle
        .study_quality_reports
        .iter()
        .map(|report| (report.execution.study_id.as_str(), report))
        .collect::<BTreeMap<_, _>>();
    let semantic_by_pair = bundle
        .semantic_drift_reports
        .iter()
        .map(|report| ((report.baseline.set_id.as_str(), report.candidate.set_id.as_str()), report))
        .collect::<BTreeMap<_, _>>();
    let metric_by_id = bundle
        .metric_reports
        .iter()
        .map(|report| (report.metric.metric_id.as_str(), report))
        .collect::<BTreeMap<_, _>>();

    let mut causal_fail = 0usize;
    let mut causal_warn = 0usize;
    let mut quality_fail = 0usize;
    let mut quality_warn = 0usize;
    let mut semantic_fail = 0usize;
    let mut semantic_warn = 0usize;
    let mut metric_fail = 0usize;
    let mut metric_warn = 0usize;
    let mut migrations = 0usize;
    for claim in &bundle.claims {
        if let Some(study_id) = &claim.causal_study_id {
            let causal = causal_by_id[study_id.as_str()];
            let quality = quality_by_id[study_id.as_str()];
            match claim.kind {
                ScientificClaimKind::CausalBenefit => {
                    if causal.authority != CausalAuthority::Causal {
                        causal_fail += 1;
                    }
                    if criteria.require_high_quality_for_causal_claims
                        && quality.outcome != StudyQualityOutcome::High
                    {
                        quality_fail += 1;
                    }
                }
                ScientificClaimKind::QuasiExperimentalBenefit => {
                    if causal.authority == CausalAuthority::Causal {
                        // stronger evidence is acceptable
                    } else if causal.authority == CausalAuthority::QuasiExperimental
                        && criteria.allow_quasi_experimental_claims
                    {
                        causal_warn += 1;
                    } else {
                        causal_fail += 1;
                    }
                    if matches!(quality.outcome, StudyQualityOutcome::Low | StudyQualityOutcome::Invalid) {
                        quality_fail += 1;
                    } else if quality.outcome == StudyQualityOutcome::Moderate {
                        quality_warn += 1;
                    }
                }
                ScientificClaimKind::AssociationalSignal => {
                    if causal.authority == CausalAuthority::Unsupported {
                        causal_fail += 1;
                    } else if causal.authority != CausalAuthority::Causal {
                        causal_warn += 1;
                    }
                    if quality.outcome == StudyQualityOutcome::Invalid {
                        quality_fail += 1;
                    } else if quality.outcome != StudyQualityOutcome::High {
                        quality_warn += 1;
                    }
                }
                _ => {}
            }
        }
        if let (Some(baseline), Some(candidate)) = (
            claim.semantic_baseline_set_id.as_deref(),
            claim.semantic_candidate_set_id.as_deref(),
        ) {
            let semantic = semantic_by_pair[&(baseline, candidate)];
            match semantic.comparability {
                SemanticComparability::Invariant => {}
                SemanticComparability::DriftedButComparable
                    if criteria.allow_drifted_but_comparable_semantics => semantic_warn += 1,
                SemanticComparability::DriftedButComparable
                | SemanticComparability::HumanReview => semantic_fail += 1,
                SemanticComparability::NonComparable => semantic_fail += 1,
            }
        }
        let metric = metric_by_id[claim.metric_id.as_str()];
        match metric.metric.state {
            MetricState::Active => {}
            MetricState::Shadow if criteria.allow_shadow_metrics => metric_warn += 1,
            MetricState::Candidate | MetricState::Shadow => metric_fail += 1,
            MetricState::Deprecated if criteria.block_deprecated_metrics => metric_fail += 1,
            MetricState::Deprecated => metric_warn += 1,
            MetricState::Retired => metric_fail += 1,
        }
        if matches!(
            metric.decision,
            MetricLifecycleDecision::Deprecate
                | MetricLifecycleDecision::Retire
                | MetricLifecycleDecision::Blocked
        ) {
            metric_fail += 1;
        }
        migrations += usize::from(metric.requires_migration());
    }
    let findings = vec![
        finding(ScientificValidationFindingCode::PriorEpistemicBinding, ScientificValidationFindingStatus::Pass, format!("prior release {} is bound by {}", bundle.prior_epistemic_release_id, bundle.prior_epistemic_release_digest)),
        finding(ScientificValidationFindingCode::EvidenceBinding, ScientificValidationFindingStatus::Pass, format!("{} claims are bound to registered evidence", bundle.claims.len())),
        finding(ScientificValidationFindingCode::StudyQuality, aggregate(quality_fail, quality_warn), format!("{quality_fail} study-quality failures; {quality_warn} warnings")),
        finding(ScientificValidationFindingCode::CausalAuthority, aggregate(causal_fail, causal_warn), format!("{causal_fail} causal-authority failures; {causal_warn} warnings")),
        finding(ScientificValidationFindingCode::SemanticComparability, aggregate(semantic_fail, semantic_warn), format!("{semantic_fail} semantic failures; {semantic_warn} warnings")),
        finding(ScientificValidationFindingCode::MetricValidity, aggregate(metric_fail, metric_warn), format!("{metric_fail} metric failures; {metric_warn} warnings")),
        finding(ScientificValidationFindingCode::MetricMigration, if migrations == 0 { ScientificValidationFindingStatus::Pass } else { ScientificValidationFindingStatus::Warning }, format!("{migrations} claim metrics require consumer migration")),
    ];
    let failures = findings.iter().filter(|finding| finding.status == ScientificValidationFindingStatus::Fail).count();
    let warnings = findings.iter().filter(|finding| finding.status == ScientificValidationFindingStatus::Warning).count();
    let outcome = if failures > 0 {
        ScientificValidationOutcome::Blocked
    } else if warnings > 0 {
        ScientificValidationOutcome::HumanReview
    } else {
        ScientificValidationOutcome::Ready
    };
    Ok(ScientificValidationReport {
        schema_version: SCIENTIFIC_VALIDATION_REPORT_SCHEMA_VERSION,
        report_id,
        bundle,
        criteria,
        outcome,
        findings,
    })
}

fn aggregate(failures: usize, warnings: usize) -> ScientificValidationFindingStatus {
    if failures > 0 { ScientificValidationFindingStatus::Fail }
    else if warnings > 0 { ScientificValidationFindingStatus::Warning }
    else { ScientificValidationFindingStatus::Pass }
}
fn finding(code: ScientificValidationFindingCode, status: ScientificValidationFindingStatus, detail: String) -> ScientificValidationFinding {
    ScientificValidationFinding { code, status, detail }
}
fn validate_identifier(value: &str, field: &str) -> Result<(), ScientificValidationError> {
    if value.trim().is_empty() { Err(ScientificValidationError::InvalidIdentifier(field.to_owned())) } else { Ok(()) }
}

#[derive(Debug)]
pub enum ScientificValidationError {
    UnsupportedBundleSchema(u32),
    UnsupportedReportSchema(u32),
    InvalidIdentifier(String),
    InvalidPriorDigest,
    EmptyClaims,
    DuplicateMetricReport(String),
    DuplicateCausalStudy(String),
    DuplicateQualityStudy(String),
    DuplicateSemanticComparison(String, String),
    DuplicateClaim(String),
    MissingStudyQuality(String),
    OrphanStudyQuality,
    PreregistrationBindingMismatch(String),
    CollectionBindingMismatch(String),
    UnknownClaimMetric(String),
    MissingMetricReport(String),
    OrphanMetricEvidence,
    OrphanCausalEvidence,
    OrphanSemanticEvidence,
    UnknownCausalStudy(String),
    UnknownSemanticComparison(String, String),
    MissingClaimEvidence(String),
    ClaimMetricBindingMismatch(String),
    DerivedReportMismatch,
    Causal(CausalError),
    StudyQuality(StudyQualityError),
    Semantic(SemanticDriftError),
    Metric(MetricLifecycleError),
    Integrity(IntegrityError),
    Persistence(String),
}

impl std::fmt::Display for ScientificValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBundleSchema(version) => write!(formatter, "unsupported scientific-validation bundle schema {version}"),
            Self::UnsupportedReportSchema(version) => write!(formatter, "unsupported scientific-validation report schema {version}"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidPriorDigest => formatter.write_str("prior epistemic release digest is not a stable digest"),
            Self::EmptyClaims => formatter.write_str("scientific validation bundle requires claims"),
            Self::DuplicateMetricReport(id) => write!(formatter, "duplicate metric report for {id}"),
            Self::DuplicateCausalStudy(id) => write!(formatter, "duplicate causal study {id}"),
            Self::DuplicateQualityStudy(id) => write!(formatter, "duplicate study-quality report {id}"),
            Self::DuplicateSemanticComparison(left, right) => write!(formatter, "duplicate semantic comparison {left} -> {right}"),
            Self::DuplicateClaim(id) => write!(formatter, "duplicate scientific claim {id}"),
            Self::MissingStudyQuality(id) => write!(formatter, "causal study {id} lacks study-quality evidence"),
            Self::OrphanStudyQuality => formatter.write_str("study-quality evidence is not bound to a causal study"),
            Self::PreregistrationBindingMismatch(id) => write!(formatter, "study {id} has mismatched preregistration evidence"),
            Self::CollectionBindingMismatch(id) => write!(formatter, "study {id} has mismatched collection windows"),
            Self::UnknownClaimMetric(id) => write!(formatter, "claim references unknown metric {id}"),
            Self::MissingMetricReport(id) => write!(formatter, "claim metric {id} lacks lifecycle evidence"),
            Self::OrphanMetricEvidence => formatter.write_str("metric lifecycle evidence is not referenced by a claim"),
            Self::OrphanCausalEvidence => formatter.write_str("causal evidence is not referenced by a claim"),
            Self::OrphanSemanticEvidence => formatter.write_str("semantic evidence is not referenced by a claim"),
            Self::UnknownCausalStudy(id) => write!(formatter, "claim references unknown causal study {id}"),
            Self::UnknownSemanticComparison(left, right) => write!(formatter, "claim references unknown semantic comparison {left} -> {right}"),
            Self::MissingClaimEvidence(id) => write!(formatter, "claim {id} lacks required evidence references"),
            Self::ClaimMetricBindingMismatch(id) => write!(formatter, "claim {id} metric does not match referenced evidence"),
            Self::DerivedReportMismatch => formatter.write_str("scientific validation report does not match recomputed evidence"),
            Self::Causal(error) => write!(formatter, "causal evidence failed: {error}"),
            Self::StudyQuality(error) => write!(formatter, "study quality failed: {error}"),
            Self::Semantic(error) => write!(formatter, "semantic evidence failed: {error}"),
            Self::Metric(error) => write!(formatter, "metric lifecycle failed: {error}"),
            Self::Integrity(error) => write!(formatter, "scientific-validation integrity failed: {error}"),
            Self::Persistence(detail) => formatter.write_str(detail),
        }
    }
}
impl std::error::Error for ScientificValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Causal(error) => Some(error),
            Self::StudyQuality(error) => Some(error),
            Self::Semantic(error) => Some(error),
            Self::Metric(error) => Some(error),
            Self::Integrity(error) => Some(error),
            _ => None,
        }
    }
}
impl From<std::io::Error> for ScientificValidationError { fn from(error: std::io::Error) -> Self { Self::Persistence(error.to_string()) } }
impl From<serde_json::Error> for ScientificValidationError { fn from(error: serde_json::Error) -> Self { Self::Persistence(error.to_string()) } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CausalDesign, CausalPolicy, CausalStudy, Estimand, MetricDefinition,
        MetricEvidenceSnapshot, MetricLifecyclePolicy, MultiplicityCorrection,
        ProtocolDeviation, RegisteredHypothesis, SemanticAnchor, SemanticAnchorSet,
        SemanticContext, SemanticDriftPolicy, StudyArm, StudyExecution,
        StudyPreregistration, StudyQualityPolicy, evaluate_causal_study,
        evaluate_metric_lifecycle, evaluate_semantic_drift, evaluate_study_quality,
    };
    use std::collections::BTreeMap;

    fn bundle() -> ScientificValidationBundle {
        let registry = MetricRegistry {
            schema_version: crate::METRIC_REGISTRY_SCHEMA_VERSION,
            registry_id: "metrics-1".to_owned(),
            metrics: vec![MetricDefinition { metric_id: "metric-order".to_owned(), version: "1.0.0".to_owned(), owner: "team".to_owned(), purpose: "ordering signal".to_owned(), state: MetricState::Active, introduced_unix_ms: 1, replacement_metric_id: None, decision_dependencies: vec!["ranking".to_owned()] }],
        };
        let metric_report = evaluate_metric_lifecycle(&registry, MetricEvidenceSnapshot { snapshot_id: "snap".to_owned(), metric_id: "metric-order".to_owned(), observed_unix_ms: 2, predictive_validity: 0.8, calibration_quality: 0.8, out_of_distribution_reliability: 0.8, grounded_evidence_fraction: 0.9, saturation_fraction: 0.1, proxy_divergence: 0.1, documented_harm_incidents: 0, decision_volume: 1000, active_consumer_count: 0 }, MetricLifecyclePolicy::production()).expect("metric report");
        let study = CausalStudy { schema_version: crate::CAUSAL_STUDY_SCHEMA_VERSION, study_id: "study-1".to_owned(), preregistration_id: "pre-1".to_owned(), intervention_metric_id: "metric-order".to_owned(), outcome_metric_id: "preference".to_owned(), design: CausalDesign::RandomizedControlled, estimand: Estimand::IntentionToTreat, assignment_seed: Some(1), arms: vec![StudyArm { arm_id: "control".to_owned(), is_control: true, intervention_dose: 0.0, assigned: 100, completed: 95, baseline_mean: Some(0.5), outcome_mean: 0.5, outcome_variance: 0.04 }, StudyArm { arm_id: "treatment".to_owned(), is_control: false, intervention_dose: 0.2, assigned: 100, completed: 95, baseline_mean: Some(0.5), outcome_mean: 0.65, outcome_variance: 0.04 }], measured_confounders: vec!["expertise".to_owned()], known_unmeasured_confounders: Vec::new(), collection_started_unix_ms: 30, collection_ended_unix_ms: 40 };
        let causal = evaluate_causal_study(study, CausalPolicy::production()).expect("causal");
        let preregistration = StudyPreregistration { schema_version: crate::PREREGISTRATION_SCHEMA_VERSION, preregistration_id: "pre-1".to_owned(), study_id: "study-1".to_owned(), registered_unix_ms: 10, planned_collection_start_unix_ms: 20, hypotheses: vec![RegisteredHypothesis { hypothesis_id: "h1".to_owned(), statement: "order changes preference".to_owned(), primary: true, directional: true }], primary_outcomes: vec!["preference".to_owned()], exclusion_rules: Vec::new(), analysis_plan_digest: "plan:v1".to_owned(), planned_sample_size: 95, planned_multiplicity_correction: MultiplicityCorrection::Holm };
        let execution = StudyExecution { study_id: "study-1".to_owned(), collection_started_unix_ms: 30, collection_ended_unix_ms: 40, enrolled: 100, completed: 95, excluded_after_collection: 0, hypotheses_tested: 1, reported_primary_outcomes: vec!["preference".to_owned()], participant_blinded: true, assessor_blinded: true, analyst_blinded_until_lock: true, manipulation_check_pass_rate: Some(0.9), applied_multiplicity_correction: MultiplicityCorrection::None, deviations: Vec::<ProtocolDeviation>::new(), analysis_plan_digest: "plan:v1".to_owned() };
        let quality = evaluate_study_quality(preregistration, execution, StudyQualityPolicy::production()).expect("quality");
        let make_set = |id: &str, culture: &str| SemanticAnchorSet { schema_version: crate::SEMANTIC_ANCHOR_SET_SCHEMA_VERSION, set_id: id.to_owned(), metric_id: "metric-order".to_owned(), context: SemanticContext { culture: culture.to_owned(), language: "en".to_owned(), population: "adult".to_owned(), period_start_unix_ms: 1, period_end_unix_ms: 2 }, anchors: vec![SemanticAnchor { anchor_id: "balanced".to_owned(), label: "balanced".to_owned(), dimensions: BTreeMap::from([("order".to_owned(), 0.5)]), sample_count: 100, reliability: 0.9 }, SemanticAnchor { anchor_id: "chaotic".to_owned(), label: "chaotic".to_owned(), dimensions: BTreeMap::from([("order".to_owned(), -0.5)]), sample_count: 100, reliability: 0.9 }] };
        let semantic = evaluate_semantic_drift(make_set("set-a", "A"), make_set("set-b", "B"), SemanticDriftPolicy::production()).expect("semantic");
        ScientificValidationBundle { schema_version: SCIENTIFIC_VALIDATION_BUNDLE_SCHEMA_VERSION, bundle_id: "science-1".to_owned(), release_id: "release-1".to_owned(), prior_epistemic_release_id: "epistemic-1".to_owned(), prior_epistemic_release_digest: crate::digest_bytes(b"epistemic"), metric_registry: registry, metric_reports: vec![metric_report], causal_reports: vec![causal], study_quality_reports: vec![quality], semantic_drift_reports: vec![semantic], claims: vec![ScientificClaim { claim_id: "claim-1".to_owned(), metric_id: "metric-order".to_owned(), kind: ScientificClaimKind::CausalBenefit, causal_study_id: Some("study-1".to_owned()), semantic_baseline_set_id: Some("set-a".to_owned()), semantic_candidate_set_id: Some("set-b".to_owned()) }] }
    }

    #[test]
    fn complete_bundle_can_be_ready() {
        let report = evaluate_scientific_validation("report-1", bundle(), ScientificValidationCriteria::production()).expect("report");
        assert_eq!(report.outcome, ScientificValidationOutcome::Ready);
        report.validate().expect("valid report");
    }

    #[test]
    fn substituted_preregistration_is_rejected() {
        let mut bundle = bundle();
        bundle.causal_reports[0].study.preregistration_id = "other".to_owned();
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn forged_ready_outcome_is_rejected() {
        let mut report = evaluate_scientific_validation("report-1", bundle(), ScientificValidationCriteria::production()).expect("report");
        report.outcome = ScientificValidationOutcome::Blocked;
        assert!(matches!(report.validate(), Err(ScientificValidationError::DerivedReportMismatch)));
    }
}
