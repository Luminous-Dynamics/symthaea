//! Versioned benchmark records and conservative release gates.

use crate::{CapabilityLevel, CommunicationEvidence};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BENCHMARK_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub id: String,
    pub uri: String,
    pub revision: String,
    pub manifest_hash: String,
    pub license_id: String,
    pub split: String,
    #[serde(default)]
    pub sample_ids: BTreeSet<String>,
    #[serde(default)]
    pub identity_ids: BTreeSet<String>,
    #[serde(default)]
    pub site_ids: BTreeSet<String>,
}

impl DatasetManifest {
    pub fn computed_manifest_hash(&self) -> String {
        let mut bytes = Vec::new();
        for value in &self.sample_ids {
            bytes.extend_from_slice(value.as_bytes());
            bytes.push(b'\n');
        }
        crate::content_hash(&bytes)
    }
}

pub fn validate_split_separation(datasets: &[DatasetManifest]) -> Result<(), String> {
    for (index, left) in datasets.iter().enumerate() {
        if left.sample_ids.is_empty() || left.computed_manifest_hash() != left.manifest_hash {
            return Err(format!(
                "dataset {} sample manifest is empty or hash-mismatched",
                left.id
            ));
        }
        for right in &datasets[index + 1..] {
            if left.split != right.split && !left.sample_ids.is_disjoint(&right.sample_ids) {
                return Err(format!(
                    "sample leakage between {} and {}",
                    left.id, right.id
                ));
            }
            if left.split != right.split
                && !left.identity_ids.is_empty()
                && !right.identity_ids.is_empty()
                && !left.identity_ids.is_disjoint(&right.identity_ids)
            {
                return Err(format!(
                    "identity leakage between {} and {}",
                    left.id, right.id
                ));
            }
            if left.split != right.split
                && !left.site_ids.is_empty()
                && !right.site_ids.is_empty()
                && !left.site_ids.is_disjoint(&right.site_ids)
            {
                return Err(format!("site leakage between {} and {}", left.id, right.id));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationPlan {
    pub schema_version: u32,
    pub id: String,
    pub provider_manifest: String,
    pub scopes: BTreeSet<String>,
    pub required_metrics: BTreeSet<String>,
    pub datasets: Vec<DatasetManifest>,
    pub maximum_relative_regression: f64,
    #[serde(default = "default_minimum_samples")]
    pub minimum_sample_count: u64,
    #[serde(default)]
    pub require_calibration: bool,
    #[serde(default)]
    pub require_thresholds: bool,
    #[serde(default)]
    pub require_hardware: bool,
}

fn default_minimum_samples() -> u64 {
    1
}

impl EvaluationPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != BENCHMARK_SCHEMA_VERSION || self.id.is_empty() {
            return Err("invalid evaluation plan identity or version".into());
        }
        if self.scopes.is_empty() || self.required_metrics.is_empty() || self.datasets.is_empty() {
            return Err("scopes, metrics, and pinned datasets are required".into());
        }
        if !(0.0..1.0).contains(&self.maximum_relative_regression) {
            return Err("maximum relative regression must be in [0, 1)".into());
        }
        if self.datasets.iter().any(|dataset| {
            dataset.id.is_empty()
                || dataset.uri.is_empty()
                || dataset.revision.is_empty()
                || dataset.manifest_hash.len() < 16
                || dataset.license_id.is_empty()
                || dataset.split.is_empty()
                || dataset.sample_ids.is_empty()
        }) {
            return Err(
                "every dataset must have URI, revision, manifest hash, license, and split".into(),
            );
        }
        validate_split_separation(&self.datasets)?;
        Ok(())
    }

    pub fn release_gate(&self) -> ReleaseGate {
        ReleaseGate {
            required_scopes: self.scopes.clone(),
            required_metrics: self.required_metrics.clone(),
            minimum_sample_count: self.minimum_sample_count,
            require_calibration: self.require_calibration,
            require_thresholds: self.require_thresholds,
            require_hardware: self.require_hardware,
            expected_provider: None,
            expected_model_hash: None,
            expected_dataset_hashes: self
                .datasets
                .iter()
                .map(|dataset| dataset.manifest_hash.clone())
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderStatus {
    Active,
    FeatureGated,
    Placeholder,
    Disconnected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricResult {
    pub name: String,
    pub value: f64,
    pub sample_count: u64,
    pub threshold: Option<f64>,
    pub higher_is_better: bool,
}

impl MetricResult {
    pub fn passes(&self) -> bool {
        self.value.is_finite()
            && self.sample_count > 0
            && self.threshold.is_none_or(|threshold| {
                if self.higher_is_better {
                    self.value >= threshold
                } else {
                    self.value <= threshold
                }
            })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScopeResult {
    /// Language, species, site, protocol, or modality identifier.
    pub scope: String,
    pub metrics: Vec<MetricResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub provider: String,
    pub provider_status: ProviderStatus,
    pub claimed_capability: CapabilityLevel,
    pub evidence: Vec<CommunicationEvidence>,
    pub scopes: Vec<ScopeResult>,
    pub hardware: BTreeMap<String, String>,
    pub feature_flags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateFailure {
    WrongSchemaVersion,
    InactiveProvider,
    MissingEvidence,
    ExperimentalEvidence,
    EmptyHash,
    MissingScope(String),
    DuplicateScope(String),
    MissingMetrics(String),
    MissingRequiredMetric { scope: String, metric: String },
    MetricFailed { scope: String, metric: String },
    UnsupportedGroundedClaim,
    MissingCalibration,
    SampleCountTooSmall { scope: String, metric: String },
    ProviderMismatch,
    ModelHashMismatch,
    DatasetHashMismatch,
    InvalidCalibration,
    EvidenceIdentityMismatch,
    MissingThreshold { scope: String, metric: String },
    MissingHardware,
}

#[derive(Clone, Debug, Default)]
pub struct ReleaseGate {
    pub required_scopes: BTreeSet<String>,
    pub required_metrics: BTreeSet<String>,
    pub minimum_sample_count: u64,
    pub require_calibration: bool,
    pub require_thresholds: bool,
    pub require_hardware: bool,
    pub expected_provider: Option<String>,
    pub expected_model_hash: Option<String>,
    pub expected_dataset_hashes: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricRegression {
    pub scope: String,
    pub metric: String,
    pub baseline: f64,
    pub candidate: f64,
    pub relative_regression: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ComparisonFailure {
    MissingScope(String),
    MissingMetric { scope: String, metric: String },
    DirectionChanged { scope: String, metric: String },
    Regression(MetricRegression),
}

pub fn compare_reports(
    baseline: &BenchmarkReport,
    candidate: &BenchmarkReport,
    maximum_relative_regression: f64,
) -> Vec<ComparisonFailure> {
    let candidate_metrics: BTreeMap<(&str, &str), &MetricResult> = candidate
        .scopes
        .iter()
        .flat_map(|scope| {
            scope
                .metrics
                .iter()
                .map(move |metric| ((scope.scope.as_str(), metric.name.as_str()), metric))
        })
        .collect();
    let candidate_scopes: BTreeSet<_> = candidate
        .scopes
        .iter()
        .map(|scope| scope.scope.as_str())
        .collect();
    let mut failures = Vec::new();
    for scope in &baseline.scopes {
        if !candidate_scopes.contains(scope.scope.as_str()) {
            failures.push(ComparisonFailure::MissingScope(scope.scope.clone()));
            continue;
        }
        for baseline in &scope.metrics {
            let Some(metric) =
                candidate_metrics.get(&(scope.scope.as_str(), baseline.name.as_str()))
            else {
                failures.push(ComparisonFailure::MissingMetric {
                    scope: scope.scope.clone(),
                    metric: baseline.name.clone(),
                });
                continue;
            };
            if metric.higher_is_better != baseline.higher_is_better {
                failures.push(ComparisonFailure::DirectionChanged {
                    scope: scope.scope.clone(),
                    metric: baseline.name.clone(),
                });
                continue;
            }
            let denominator = baseline.value.abs().max(f64::EPSILON);
            let regression = if metric.higher_is_better {
                (baseline.value - metric.value) / denominator
            } else {
                (metric.value - baseline.value) / denominator
            };
            if regression > maximum_relative_regression {
                failures.push(ComparisonFailure::Regression(MetricRegression {
                    scope: scope.scope.clone(),
                    metric: metric.name.clone(),
                    baseline: baseline.value,
                    candidate: metric.value,
                    relative_regression: regression,
                }));
            }
        }
    }
    failures
}

impl ReleaseGate {
    pub fn evaluate(&self, report: &BenchmarkReport) -> Result<(), Vec<GateFailure>> {
        let mut failures = Vec::new();
        if report.schema_version != BENCHMARK_SCHEMA_VERSION {
            failures.push(GateFailure::WrongSchemaVersion);
        }
        if report.provider_status != ProviderStatus::Active {
            failures.push(GateFailure::InactiveProvider);
        }
        if self.require_hardware && report.hardware.is_empty() {
            failures.push(GateFailure::MissingHardware);
        }
        if self
            .expected_provider
            .as_ref()
            .is_some_and(|provider| provider != &report.provider)
        {
            failures.push(GateFailure::ProviderMismatch);
        }
        if self
            .expected_model_hash
            .as_ref()
            .is_some_and(|hash| report.evidence.iter().any(|e| &e.model_hash != hash))
        {
            failures.push(GateFailure::ModelHashMismatch);
        }
        if !self.expected_dataset_hashes.is_empty()
            && report
                .evidence
                .iter()
                .any(|e| !self.expected_dataset_hashes.contains(&e.dataset_hash))
        {
            failures.push(GateFailure::DatasetHashMismatch);
        }
        if self.require_calibration && report.evidence.iter().any(|e| e.calibration.is_empty()) {
            failures.push(GateFailure::MissingCalibration);
        }
        if report.evidence.iter().any(|e| {
            !e.calibration.is_empty()
                && crate::metrics::expected_calibration_error(&e.calibration).is_none()
        }) {
            failures.push(GateFailure::InvalidCalibration);
        }
        if report
            .evidence
            .iter()
            .any(|e| e.computed_id().is_err() || e.computed_id().is_ok_and(|id| id != e.id))
        {
            failures.push(GateFailure::EvidenceIdentityMismatch);
        }
        if report.evidence.is_empty() {
            failures.push(GateFailure::MissingEvidence);
        }
        if report.evidence.iter().any(|e| e.experimental) {
            failures.push(GateFailure::ExperimentalEvidence);
        }
        if report
            .evidence
            .iter()
            .any(|e| e.dataset_hash.is_empty() || e.model_hash.is_empty())
        {
            failures.push(GateFailure::EmptyHash);
        }
        if report.claimed_capability >= CapabilityLevel::Reference
            && crate::requires_grounded_intervention(&report.evidence)
            && !crate::has_grounded_intervention(&report.evidence)
        {
            failures.push(GateFailure::UnsupportedGroundedClaim);
        }
        let mut seen = BTreeSet::new();
        for scope in &report.scopes {
            if !seen.insert(scope.scope.clone()) {
                failures.push(GateFailure::DuplicateScope(scope.scope.clone()));
            }
            if scope.metrics.is_empty() {
                failures.push(GateFailure::MissingMetrics(scope.scope.clone()));
            }
            let metric_names: BTreeSet<_> = scope
                .metrics
                .iter()
                .map(|metric| metric.name.as_str())
                .collect();
            for required in &self.required_metrics {
                if !metric_names.contains(required.as_str()) {
                    failures.push(GateFailure::MissingRequiredMetric {
                        scope: scope.scope.clone(),
                        metric: required.clone(),
                    });
                }
            }
            for metric in &scope.metrics {
                if self.require_thresholds && metric.threshold.is_none() {
                    failures.push(GateFailure::MissingThreshold {
                        scope: scope.scope.clone(),
                        metric: metric.name.clone(),
                    });
                }
                if metric.sample_count < self.minimum_sample_count {
                    failures.push(GateFailure::SampleCountTooSmall {
                        scope: scope.scope.clone(),
                        metric: metric.name.clone(),
                    });
                }
                if !metric.passes() {
                    failures.push(GateFailure::MetricFailed {
                        scope: scope.scope.clone(),
                        metric: metric.name.clone(),
                    });
                }
            }
        }
        for required in &self.required_scopes {
            if !seen.contains(required) {
                failures.push(GateFailure::MissingScope(required.clone()));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DatasetSplit, EvidenceDomain, ReplicationStatus};

    fn evidence(experimental: bool) -> CommunicationEvidence {
        let mut evidence = CommunicationEvidence {
            id: String::new(),
            dataset_uri: "local:test".into(),
            dataset_hash: "dataset-hash".into(),
            model_hash: "model-hash".into(),
            lineage: vec![],
            split: DatasetSplit::default(),
            evidence_records: vec![],
            preregistration_uri: None,
            replication: ReplicationStatus::Unreplicated,
            calibration: vec![],
            experimental,
            domain: EvidenceDomain::HumanLanguage,
        };
        evidence.id = evidence.computed_id().unwrap();
        evidence
    }

    fn report() -> BenchmarkReport {
        BenchmarkReport {
            schema_version: BENCHMARK_SCHEMA_VERSION,
            benchmark_id: "v1".into(),
            provider: "test".into(),
            provider_status: ProviderStatus::Active,
            claimed_capability: CapabilityLevel::Structure,
            evidence: vec![evidence(false)],
            scopes: vec![ScopeResult {
                scope: "en".into(),
                metrics: vec![MetricResult {
                    name: "lid_macro_f1".into(),
                    value: 0.96,
                    sample_count: 100,
                    threshold: Some(0.95),
                    higher_is_better: true,
                }],
            }],
            hardware: BTreeMap::new(),
            feature_flags: vec![],
        }
    }

    #[test]
    fn experimental_evidence_never_passes_release_gate() {
        let mut r = report();
        r.evidence[0].experimental = true;
        assert!(
            ReleaseGate::default()
                .evaluate(&r)
                .unwrap_err()
                .contains(&GateFailure::ExperimentalEvidence)
        );
    }

    #[test]
    fn every_declared_scope_must_have_measured_metrics() {
        let gate = ReleaseGate {
            required_scopes: BTreeSet::from(["en".into(), "zu".into()]),
            required_metrics: BTreeSet::new(),
            ..ReleaseGate::default()
        };
        assert!(
            gate.evaluate(&report())
                .unwrap_err()
                .contains(&GateFailure::MissingScope("zu".into()))
        );
    }

    #[test]
    fn detects_relative_regression_in_error_metrics() {
        let baseline = report();
        let mut candidate = baseline.clone();
        candidate.scopes[0].metrics[0].value = 0.8;
        assert_eq!(compare_reports(&baseline, &candidate, 0.05).len(), 1);
    }

    #[test]
    fn missing_candidate_metric_is_a_failure() {
        let baseline = report();
        let mut candidate = baseline.clone();
        candidate.scopes[0].metrics.clear();
        assert!(matches!(
            compare_reports(&baseline, &candidate, 0.05).as_slice(),
            [ComparisonFailure::MissingMetric { .. }]
        ));
    }

    #[test]
    fn split_overlap_is_rejected() {
        let make = |id: &str, split: &str, samples: &[&str]| {
            let mut manifest = DatasetManifest {
                id: id.into(),
                uri: "local:test".into(),
                revision: "1".into(),
                manifest_hash: String::new(),
                license_id: "CC0".into(),
                split: split.into(),
                sample_ids: samples.iter().map(|value| (*value).into()).collect(),
                identity_ids: BTreeSet::new(),
                site_ids: BTreeSet::new(),
            };
            manifest.manifest_hash = manifest.computed_manifest_hash();
            manifest
        };
        assert!(
            validate_split_separation(&[
                make("train", "train", &["a", "b"]),
                make("test", "test", &["b", "c"])
            ])
            .is_err()
        );
    }
}
