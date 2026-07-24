// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lifecycle governance for aesthetic metrics.
//!
//! Metrics are hypotheses with operational consequences, not permanent truths.
//! This module supports shadowing, deprecation, retirement, replacement binding,
//! and evidence-based refusal to keep a metric active after validity or safety
//! has deteriorated.

use crate::{IntegrityError, digest_json, load_json, save_json_atomic};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const METRIC_REGISTRY_SCHEMA_VERSION: u32 = 1;
pub const METRIC_LIFECYCLE_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MetricState {
    Candidate,
    Shadow,
    Active,
    Deprecated,
    Retired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricDefinition {
    pub metric_id: String,
    pub version: String,
    pub owner: String,
    pub purpose: String,
    pub state: MetricState,
    pub introduced_unix_ms: u64,
    pub replacement_metric_id: Option<String>,
    pub decision_dependencies: Vec<String>,
}

impl MetricDefinition {
    fn validate(&self) -> Result<(), MetricLifecycleError> {
        validate_identifier(&self.metric_id, "metric id")?;
        validate_identifier(&self.version, "metric version")?;
        validate_identifier(&self.owner, "metric owner")?;
        validate_identifier(&self.purpose, "metric purpose")?;
        if self.introduced_unix_ms == 0 {
            return Err(MetricLifecycleError::InvalidDefinition(
                "metric introduction timestamp must be nonzero".to_owned(),
            ));
        }
        if let Some(replacement) = &self.replacement_metric_id {
            validate_identifier(replacement, "replacement metric id")?;
            if replacement == &self.metric_id {
                return Err(MetricLifecycleError::SelfReplacement(self.metric_id.clone()));
            }
        }
        validate_unique(&self.decision_dependencies, "decision dependency")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricEvidenceSnapshot {
    pub snapshot_id: String,
    pub metric_id: String,
    pub observed_unix_ms: u64,
    pub predictive_validity: f32,
    pub calibration_quality: f32,
    pub out_of_distribution_reliability: f32,
    pub grounded_evidence_fraction: f32,
    pub saturation_fraction: f32,
    pub proxy_divergence: f32,
    pub documented_harm_incidents: u64,
    pub decision_volume: u64,
    pub active_consumer_count: usize,
}

impl MetricEvidenceSnapshot {
    fn validate(&self) -> Result<(), MetricLifecycleError> {
        validate_identifier(&self.snapshot_id, "snapshot id")?;
        validate_identifier(&self.metric_id, "metric id")?;
        if self.observed_unix_ms == 0 {
            return Err(MetricLifecycleError::InvalidEvidence(
                "observation timestamp must be nonzero".to_owned(),
            ));
        }
        for (value, field) in [
            (self.predictive_validity, "predictive validity"),
            (self.calibration_quality, "calibration quality"),
            (self.out_of_distribution_reliability, "OOD reliability"),
            (self.grounded_evidence_fraction, "grounded evidence fraction"),
            (self.saturation_fraction, "saturation fraction"),
            (self.proxy_divergence, "proxy divergence"),
        ] {
            validate_unit(value, field)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricRegistry {
    pub schema_version: u32,
    pub registry_id: String,
    pub metrics: Vec<MetricDefinition>,
}

impl MetricRegistry {
    pub fn validate(&self) -> Result<(), MetricLifecycleError> {
        if self.schema_version == 0 || self.schema_version > METRIC_REGISTRY_SCHEMA_VERSION {
            return Err(MetricLifecycleError::UnsupportedRegistrySchema(self.schema_version));
        }
        validate_identifier(&self.registry_id, "registry id")?;
        if self.metrics.is_empty() {
            return Err(MetricLifecycleError::EmptyRegistry);
        }
        let mut ids = BTreeSet::new();
        for metric in &self.metrics {
            metric.validate()?;
            if !ids.insert(metric.metric_id.as_str()) {
                return Err(MetricLifecycleError::DuplicateMetric(metric.metric_id.clone()));
            }
        }
        for metric in &self.metrics {
            if let Some(replacement) = &metric.replacement_metric_id {
                if !ids.contains(replacement.as_str()) {
                    return Err(MetricLifecycleError::UnknownReplacement(replacement.clone()));
                }
                let replacement_metric = self
                    .metrics
                    .iter()
                    .find(|candidate| candidate.metric_id == *replacement)
                    .ok_or_else(|| MetricLifecycleError::UnknownReplacement(replacement.clone()))?;
                if replacement_metric.state == MetricState::Retired {
                    return Err(MetricLifecycleError::RetiredReplacement(replacement.clone()));
                }
            }
        }
        detect_replacement_cycles(&self.metrics)?;
        Ok(())
    }

    pub fn metric(&self, metric_id: &str) -> Option<&MetricDefinition> {
        self.metrics.iter().find(|metric| metric.metric_id == metric_id)
    }

    pub fn digest(&self) -> Result<String, MetricLifecycleError> {
        self.validate()?;
        digest_json(self).map_err(MetricLifecycleError::Integrity)
    }

    pub fn save(&self, path: &Path) -> Result<(), MetricLifecycleError> {
        self.validate()?;
        save_json_atomic::<_, MetricLifecycleError>(path, self)
    }

    pub fn load(path: &Path) -> Result<Self, MetricLifecycleError> {
        let registry: Self = load_json::<_, MetricLifecycleError>(path)?;
        registry.validate()?;
        Ok(registry)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricLifecyclePolicy {
    pub minimum_predictive_validity: f32,
    pub minimum_calibration_quality: f32,
    pub minimum_ood_reliability: f32,
    pub minimum_grounded_fraction: f32,
    pub maximum_saturation_fraction: f32,
    pub maximum_proxy_divergence: f32,
    pub maximum_harm_rate_per_thousand: f32,
    pub require_replacement_before_retirement_with_consumers: bool,
}

impl MetricLifecyclePolicy {
    pub const fn production() -> Self {
        Self {
            minimum_predictive_validity: 0.55,
            minimum_calibration_quality: 0.65,
            minimum_ood_reliability: 0.50,
            minimum_grounded_fraction: 0.70,
            maximum_saturation_fraction: 0.35,
            maximum_proxy_divergence: 0.25,
            maximum_harm_rate_per_thousand: 1.0,
            require_replacement_before_retirement_with_consumers: true,
        }
    }

    pub fn validate(&self) -> Result<(), MetricLifecycleError> {
        for (value, field) in [
            (self.minimum_predictive_validity, "minimum predictive validity"),
            (self.minimum_calibration_quality, "minimum calibration quality"),
            (self.minimum_ood_reliability, "minimum OOD reliability"),
            (self.minimum_grounded_fraction, "minimum grounded fraction"),
            (self.maximum_saturation_fraction, "maximum saturation fraction"),
            (self.maximum_proxy_divergence, "maximum proxy divergence"),
        ] {
            validate_unit(value, field)?;
        }
        if !self.maximum_harm_rate_per_thousand.is_finite()
            || self.maximum_harm_rate_per_thousand < 0.0
        {
            return Err(MetricLifecycleError::InvalidPolicy(
                "maximum harm rate must be finite and nonnegative".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for MetricLifecyclePolicy {
    fn default() -> Self { Self::production() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MetricLifecycleFindingCode {
    RegistryBinding,
    PredictiveValidity,
    Calibration,
    DistributionShift,
    Grounding,
    Saturation,
    ProxyDivergence,
    HarmRate,
    ConsumerMigration,
}

impl MetricLifecycleFindingCode {
    pub const ALL: [Self; 9] = [
        Self::RegistryBinding,
        Self::PredictiveValidity,
        Self::Calibration,
        Self::DistributionShift,
        Self::Grounding,
        Self::Saturation,
        Self::ProxyDivergence,
        Self::HarmRate,
        Self::ConsumerMigration,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricLifecycleFindingStatus { Pass, Warning, Fail }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricLifecycleFinding {
    pub code: MetricLifecycleFindingCode,
    pub status: MetricLifecycleFindingStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricLifecycleDecision {
    PromoteToShadow,
    KeepShadow,
    PromoteToActive,
    KeepActive,
    Deprecate,
    Retire,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricLifecycleReport {
    pub schema_version: u32,
    pub registry_id: String,
    pub metric: MetricDefinition,
    pub evidence: MetricEvidenceSnapshot,
    pub policy: MetricLifecyclePolicy,
    pub harm_rate_per_thousand: f32,
    pub recommended_state: MetricState,
    pub decision: MetricLifecycleDecision,
    pub findings: Vec<MetricLifecycleFinding>,
}

impl MetricLifecycleReport {
    pub fn validate(&self, registry: &MetricRegistry) -> Result<(), MetricLifecycleError> {
        if self.schema_version == 0 || self.schema_version > METRIC_LIFECYCLE_REPORT_SCHEMA_VERSION {
            return Err(MetricLifecycleError::UnsupportedReportSchema(self.schema_version));
        }
        let expected = evaluate_metric_lifecycle(
            registry,
            self.evidence.clone(),
            self.policy,
        )?;
        if self != &expected {
            return Err(MetricLifecycleError::DerivedReportMismatch);
        }
        Ok(())
    }

    pub fn requires_migration(&self) -> bool {
        matches!(self.decision, MetricLifecycleDecision::Deprecate | MetricLifecycleDecision::Retire)
            && self.evidence.active_consumer_count > 0
    }

    pub fn digest(&self, registry: &MetricRegistry) -> Result<String, MetricLifecycleError> {
        self.validate(registry)?;
        digest_json(self).map_err(MetricLifecycleError::Integrity)
    }
}

pub fn evaluate_metric_lifecycle(
    registry: &MetricRegistry,
    evidence: MetricEvidenceSnapshot,
    policy: MetricLifecyclePolicy,
) -> Result<MetricLifecycleReport, MetricLifecycleError> {
    registry.validate()?;
    evidence.validate()?;
    policy.validate()?;
    let metric = registry
        .metric(&evidence.metric_id)
        .cloned()
        .ok_or_else(|| MetricLifecycleError::UnknownMetric(evidence.metric_id.clone()))?;
    if evidence.observed_unix_ms < metric.introduced_unix_ms {
        return Err(MetricLifecycleError::EvidencePredatesMetric(
            metric.metric_id.clone(),
        ));
    }
    let harm_rate_per_thousand = if evidence.decision_volume == 0 {
        if evidence.documented_harm_incidents == 0 { 0.0 } else { f32::INFINITY }
    } else {
        evidence.documented_harm_incidents as f32 * 1000.0 / evidence.decision_volume as f32
    };
    let checks = [
        evidence.predictive_validity >= policy.minimum_predictive_validity,
        evidence.calibration_quality >= policy.minimum_calibration_quality,
        evidence.out_of_distribution_reliability >= policy.minimum_ood_reliability,
        evidence.grounded_evidence_fraction >= policy.minimum_grounded_fraction,
        evidence.saturation_fraction <= policy.maximum_saturation_fraction,
        evidence.proxy_divergence <= policy.maximum_proxy_divergence,
        harm_rate_per_thousand <= policy.maximum_harm_rate_per_thousand,
    ];
    let failures = checks.into_iter().filter(|value| !*value).count();
    let severe = evidence.predictive_validity < policy.minimum_predictive_validity * 0.75
        || evidence.proxy_divergence > (policy.maximum_proxy_divergence * 1.5).min(1.0)
        || harm_rate_per_thousand > policy.maximum_harm_rate_per_thousand * 2.0;
    let recommended_state = match metric.state {
        MetricState::Candidate if failures == 0 => MetricState::Shadow,
        MetricState::Shadow if failures == 0 => MetricState::Active,
        MetricState::Active if severe || failures >= 3 => MetricState::Deprecated,
        MetricState::Deprecated if severe || failures >= 3 => MetricState::Retired,
        state => state,
    };
    let migration_ready = evidence.active_consumer_count == 0
        || !policy.require_replacement_before_retirement_with_consumers
        || metric.replacement_metric_id.is_some();
    let decision = match (metric.state, recommended_state) {
        (MetricState::Candidate, MetricState::Shadow) => MetricLifecycleDecision::PromoteToShadow,
        (MetricState::Shadow, MetricState::Active) => MetricLifecycleDecision::PromoteToActive,
        (MetricState::Active, MetricState::Deprecated) => MetricLifecycleDecision::Deprecate,
        (MetricState::Deprecated, MetricState::Retired) if migration_ready => MetricLifecycleDecision::Retire,
        (MetricState::Deprecated, MetricState::Retired) => MetricLifecycleDecision::Blocked,
        (MetricState::Active, MetricState::Active) => MetricLifecycleDecision::KeepActive,
        (MetricState::Shadow, MetricState::Shadow) | (MetricState::Candidate, MetricState::Candidate) => MetricLifecycleDecision::KeepShadow,
        (MetricState::Retired, MetricState::Retired) => MetricLifecycleDecision::Blocked,
        _ => MetricLifecycleDecision::Blocked,
    };
    let findings = vec![
        finding(MetricLifecycleFindingCode::RegistryBinding, MetricLifecycleFindingStatus::Pass, format!("metric {} is registered in {}", metric.metric_id, registry.registry_id)),
        finding(MetricLifecycleFindingCode::PredictiveValidity, pass_fail(checks[0]), format!("predictive validity {:.4}", evidence.predictive_validity)),
        finding(MetricLifecycleFindingCode::Calibration, pass_fail(checks[1]), format!("calibration quality {:.4}", evidence.calibration_quality)),
        finding(MetricLifecycleFindingCode::DistributionShift, pass_fail(checks[2]), format!("OOD reliability {:.4}", evidence.out_of_distribution_reliability)),
        finding(MetricLifecycleFindingCode::Grounding, pass_fail(checks[3]), format!("grounded evidence fraction {:.4}", evidence.grounded_evidence_fraction)),
        finding(MetricLifecycleFindingCode::Saturation, pass_fail(checks[4]), format!("saturation fraction {:.4}", evidence.saturation_fraction)),
        finding(MetricLifecycleFindingCode::ProxyDivergence, pass_fail(checks[5]), format!("proxy divergence {:.4}", evidence.proxy_divergence)),
        finding(MetricLifecycleFindingCode::HarmRate, pass_fail(checks[6]), format!("harm rate per thousand {:.4}", harm_rate_per_thousand)),
        finding(MetricLifecycleFindingCode::ConsumerMigration, if migration_ready { MetricLifecycleFindingStatus::Pass } else { MetricLifecycleFindingStatus::Fail }, format!("{} active consumers; replacement {:?}", evidence.active_consumer_count, metric.replacement_metric_id)),
    ];
    Ok(MetricLifecycleReport {
        schema_version: METRIC_LIFECYCLE_REPORT_SCHEMA_VERSION,
        registry_id: registry.registry_id.clone(),
        metric,
        evidence,
        policy,
        harm_rate_per_thousand,
        recommended_state,
        decision,
        findings,
    })
}

fn detect_replacement_cycles(metrics: &[MetricDefinition]) -> Result<(), MetricLifecycleError> {
    let by_id = metrics.iter().map(|metric| (metric.metric_id.as_str(), metric)).collect::<BTreeMap<_, _>>();
    for metric in metrics {
        let mut seen = BTreeSet::new();
        let mut current = Some(metric.metric_id.as_str());
        while let Some(id) = current {
            if !seen.insert(id) {
                return Err(MetricLifecycleError::ReplacementCycle(metric.metric_id.clone()));
            }
            current = by_id.get(id).and_then(|entry| entry.replacement_metric_id.as_deref());
        }
    }
    Ok(())
}

fn pass_fail(pass: bool) -> MetricLifecycleFindingStatus {
    if pass { MetricLifecycleFindingStatus::Pass } else { MetricLifecycleFindingStatus::Fail }
}
fn finding(code: MetricLifecycleFindingCode, status: MetricLifecycleFindingStatus, detail: String) -> MetricLifecycleFinding {
    MetricLifecycleFinding { code, status, detail }
}
fn validate_identifier(value: &str, field: &str) -> Result<(), MetricLifecycleError> {
    if value.trim().is_empty() { Err(MetricLifecycleError::InvalidIdentifier(field.to_owned())) } else { Ok(()) }
}
fn validate_unique(values: &[String], field: &str) -> Result<(), MetricLifecycleError> {
    let mut unique = BTreeSet::new();
    for value in values {
        validate_identifier(value, field)?;
        if !unique.insert(value.as_str()) { return Err(MetricLifecycleError::DuplicateDependency(value.clone())); }
    }
    Ok(())
}
fn validate_unit(value: f32, field: &str) -> Result<(), MetricLifecycleError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) { Err(MetricLifecycleError::InvalidNumber(field.to_owned())) } else { Ok(()) }
}

#[derive(Debug)]
pub enum MetricLifecycleError {
    UnsupportedRegistrySchema(u32),
    UnsupportedReportSchema(u32),
    InvalidIdentifier(String),
    InvalidNumber(String),
    InvalidDefinition(String),
    InvalidEvidence(String),
    InvalidPolicy(String),
    EmptyRegistry,
    DuplicateMetric(String),
    DuplicateDependency(String),
    UnknownMetric(String),
    UnknownReplacement(String),
    RetiredReplacement(String),
    SelfReplacement(String),
    ReplacementCycle(String),
    EvidencePredatesMetric(String),
    DerivedReportMismatch,
    Integrity(IntegrityError),
    Persistence(String),
}

impl std::fmt::Display for MetricLifecycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRegistrySchema(version) => write!(formatter, "unsupported metric-registry schema {version}"),
            Self::UnsupportedReportSchema(version) => write!(formatter, "unsupported metric-lifecycle schema {version}"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidNumber(field) => write!(formatter, "{field} must be finite and in [0, 1]"),
            Self::InvalidDefinition(detail) | Self::InvalidEvidence(detail) | Self::InvalidPolicy(detail) | Self::Persistence(detail) => formatter.write_str(detail),
            Self::EmptyRegistry => formatter.write_str("metric registry must not be empty"),
            Self::DuplicateMetric(id) => write!(formatter, "duplicate metric {id}"),
            Self::DuplicateDependency(id) => write!(formatter, "duplicate decision dependency {id}"),
            Self::UnknownMetric(id) => write!(formatter, "unknown metric {id}"),
            Self::UnknownReplacement(id) => write!(formatter, "unknown replacement metric {id}"),
            Self::RetiredReplacement(id) => write!(formatter, "replacement metric {id} is retired"),
            Self::SelfReplacement(id) => write!(formatter, "metric {id} cannot replace itself"),
            Self::ReplacementCycle(id) => write!(formatter, "replacement chain for {id} contains a cycle"),
            Self::EvidencePredatesMetric(id) => write!(formatter, "evidence for metric {id} predates its introduction"),
            Self::DerivedReportMismatch => formatter.write_str("metric-lifecycle report does not match recomputed evidence"),
            Self::Integrity(error) => write!(formatter, "metric lifecycle integrity failed: {error}"),
        }
    }
}
impl std::error::Error for MetricLifecycleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { match self { Self::Integrity(error) => Some(error), _ => None } }
}
impl From<std::io::Error> for MetricLifecycleError { fn from(error: std::io::Error) -> Self { Self::Persistence(error.to_string()) } }
impl From<serde_json::Error> for MetricLifecycleError { fn from(error: serde_json::Error) -> Self { Self::Persistence(error.to_string()) } }

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(state: MetricState, replacement: Option<&str>) -> MetricRegistry {
        let mut metrics = vec![MetricDefinition {
            metric_id: "legacy".to_owned(), version: "1.0.0".to_owned(), owner: "aesthetic".to_owned(),
            purpose: "legacy proxy".to_owned(), state, introduced_unix_ms: 1,
            replacement_metric_id: replacement.map(str::to_owned), decision_dependencies: vec!["ranking".to_owned()],
        }];
        if replacement.is_some() {
            metrics.push(MetricDefinition { metric_id: "replacement".to_owned(), version: "1.0.0".to_owned(), owner: "aesthetic".to_owned(), purpose: "grounded replacement".to_owned(), state: MetricState::Active, introduced_unix_ms: 2, replacement_metric_id: None, decision_dependencies: vec!["ranking".to_owned()] });
        }
        MetricRegistry { schema_version: METRIC_REGISTRY_SCHEMA_VERSION, registry_id: "registry-1".to_owned(), metrics }
    }
    fn evidence(consumers: usize) -> MetricEvidenceSnapshot {
        MetricEvidenceSnapshot { snapshot_id: "snapshot-1".to_owned(), metric_id: "legacy".to_owned(), observed_unix_ms: 10, predictive_validity: 0.20, calibration_quality: 0.30, out_of_distribution_reliability: 0.20, grounded_evidence_fraction: 0.30, saturation_fraction: 0.80, proxy_divergence: 0.70, documented_harm_incidents: 5, decision_volume: 1000, active_consumer_count: consumers }
    }

    #[test]
    fn failing_active_metric_is_deprecated() {
        let report = evaluate_metric_lifecycle(&registry(MetricState::Active, Some("replacement")), evidence(3), MetricLifecyclePolicy::production()).expect("report");
        assert_eq!(report.decision, MetricLifecycleDecision::Deprecate);
    }

    #[test]
    fn retirement_requires_replacement_for_active_consumers() {
        let report = evaluate_metric_lifecycle(&registry(MetricState::Deprecated, None), evidence(3), MetricLifecyclePolicy::production()).expect("report");
        assert_eq!(report.decision, MetricLifecycleDecision::Blocked);
    }

    #[test]
    fn replacement_cycles_are_rejected() {
        let mut registry = registry(MetricState::Deprecated, Some("replacement"));
        registry.metrics[1].replacement_metric_id = Some("legacy".to_owned());
        assert!(matches!(registry.validate(), Err(MetricLifecycleError::ReplacementCycle(_))));
    }
}
