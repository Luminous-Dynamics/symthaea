// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, explicit metric-policy manifests for regression comparison.
//!
//! This layer owns policy *identity and coverage*, not policy semantics. The
//! direction-aware comparator remains the single authority for deciding whether
//! a policy's numeric contents are valid and how it adjudicates a metric.
//!
//! ```text
//! structurally valid manifest
//!     != semantically valid metric policy
//!     != successful regression comparison
//! ```
//!
//! No metric direction is inferred from benchmark or metric names.

use crate::harness::snapshot::{RegressionSnapshot, SNAPSHOT_SCHEMA_VERSION};
use crate::regression_contract::{
    ComparisonSetupError, MetricComparisonPolicy, MetricKey, MetricObjective, MetricPolicyRegistry,
    PolicyAwareRegressionReport, RegressionThresholds, compare_snapshots,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REGRESSION_POLICY_MANIFEST_SCHEMA_VERSION: &str =
    "psych-regression-policy-manifest-v1";

/// One explicit policy assignment to one stable benchmark metric identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPolicyEntry {
    pub key: MetricKey,
    pub policy: MetricComparisonPolicy,
}

impl MetricPolicyEntry {
    pub fn new(key: MetricKey, policy: MetricComparisonPolicy) -> Self {
        Self { key, policy }
    }
}

/// Versioned policy catalog. There is intentionally no default policy and no
/// name-based direction inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPolicyManifest {
    pub schema_version: String,
    pub manifest_id: String,
    pub revision: u32,
    pub entries: Vec<MetricPolicyEntry>,
}

impl MetricPolicyManifest {
    pub fn new(
        manifest_id: impl Into<String>,
        revision: u32,
        entries: Vec<MetricPolicyEntry>,
    ) -> Self {
        Self {
            schema_version: REGRESSION_POLICY_MANIFEST_SCHEMA_VERSION.to_string(),
            manifest_id: manifest_id.into(),
            revision,
            entries,
        }
    }

    /// Validate manifest structure only.
    ///
    /// Objective bounds and thresholds remain the comparator's semantic
    /// authority; this function deliberately does not duplicate that logic.
    pub fn validate_structure(&self) -> Result<(), MetricPolicyManifestError> {
        if self.schema_version != REGRESSION_POLICY_MANIFEST_SCHEMA_VERSION {
            return Err(MetricPolicyManifestError::UnsupportedSchema);
        }
        if self.manifest_id.trim().is_empty() {
            return Err(MetricPolicyManifestError::EmptyManifestId);
        }
        if self.revision == 0 {
            return Err(MetricPolicyManifestError::InvalidRevision);
        }
        if self.entries.is_empty() {
            return Err(MetricPolicyManifestError::EmptyEntries);
        }

        let mut seen = BTreeSet::new();
        for entry in &self.entries {
            if entry.key.benchmark.trim().is_empty() || entry.key.metric.trim().is_empty() {
                return Err(MetricPolicyManifestError::EmptyMetricIdentity);
            }
            if !seen.insert(entry.key.clone()) {
                return Err(MetricPolicyManifestError::DuplicateMetricKey(
                    entry.key.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Build the exact registry consumed by the qualified comparator.
    pub fn to_registry(&self) -> Result<MetricPolicyRegistry, MetricPolicyManifestError> {
        self.validate_structure()?;
        Ok(self
            .entries
            .iter()
            .map(|entry| (entry.key.clone(), entry.policy.clone()))
            .collect::<BTreeMap<_, _>>())
    }

    /// Deterministic manifest identity independent of entry ordering.
    pub fn digest_hex(&self) -> Result<String, MetricPolicyManifestError> {
        self.validate_structure()?;

        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.key.cmp(&right.key));

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.regression-policy-manifest.v1\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.manifest_id.as_bytes());
        hasher.update(&self.revision.to_le_bytes());
        hasher.update(&(entries.len() as u64).to_le_bytes());

        for entry in entries {
            hash_field(&mut hasher, entry.key.benchmark.as_bytes());
            hash_field(&mut hasher, entry.key.metric.as_bytes());
            hash_field(&mut hasher, entry.policy.policy_id.as_bytes());
            hasher.update(&entry.policy.version.to_le_bytes());
            hash_objective(&mut hasher, &entry.policy.objective);
        }

        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Compare manifest coverage to the exact metric census of one baseline.
    pub fn coverage_for_snapshot(
        &self,
        snapshot: &RegressionSnapshot,
    ) -> Result<MetricPolicyCoverage, MetricPolicyManifestError> {
        self.validate_structure()?;
        if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
            return Err(MetricPolicyManifestError::SnapshotSchemaMismatch {
                found: snapshot.schema_version.clone(),
                required: SNAPSHOT_SCHEMA_VERSION.to_string(),
            });
        }

        let registry = self.to_registry()?;
        let baseline_keys = snapshot
            .metrics
            .iter()
            .flat_map(|(benchmark, metrics)| {
                metrics
                    .keys()
                    .map(move |metric| MetricKey::new(benchmark, metric))
            })
            .collect::<BTreeSet<_>>();
        let manifest_keys = registry.keys().cloned().collect::<BTreeSet<_>>();

        let missing = baseline_keys
            .difference(&manifest_keys)
            .cloned()
            .collect::<Vec<_>>();
        let extra = manifest_keys
            .difference(&baseline_keys)
            .cloned()
            .collect::<Vec<_>>();

        Ok(MetricPolicyCoverage {
            baseline_required: baseline_keys.len(),
            covered: baseline_keys.len() - missing.len(),
            missing,
            extra,
        })
    }

    /// Require every baseline metric to have an explicit policy before numeric
    /// comparison begins. Extra manifest policies remain visible but are allowed.
    pub fn require_complete_for_snapshot(
        &self,
        snapshot: &RegressionSnapshot,
    ) -> Result<MetricPolicyCoverage, MetricPolicyManifestError> {
        let coverage = self.coverage_for_snapshot(snapshot)?;
        if !coverage.missing.is_empty() {
            return Err(MetricPolicyManifestError::MissingBaselinePolicies(
                coverage.missing.clone(),
            ));
        }
        Ok(coverage)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricPolicyCoverage {
    pub baseline_required: usize,
    pub covered: usize,
    pub missing: Vec<MetricKey>,
    pub extra: Vec<MetricKey>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricPolicyManifestError {
    UnsupportedSchema,
    EmptyManifestId,
    InvalidRevision,
    EmptyEntries,
    EmptyMetricIdentity,
    DuplicateMetricKey(MetricKey),
    SnapshotSchemaMismatch { found: String, required: String },
    MissingBaselinePolicies(Vec<MetricKey>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManifestComparisonError {
    Manifest(MetricPolicyManifestError),
    Comparison(ComparisonSetupError),
}

impl From<MetricPolicyManifestError> for ManifestComparisonError {
    fn from(value: MetricPolicyManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<ComparisonSetupError> for ManifestComparisonError {
    fn from(value: ComparisonSetupError) -> Self {
        Self::Comparison(value)
    }
}

/// Fail closed on missing policy coverage, then delegate all numeric and policy
/// semantics to the qualified direction-aware comparator.
pub fn compare_snapshots_with_manifest(
    baseline: &RegressionSnapshot,
    current: &RegressionSnapshot,
    manifest: &MetricPolicyManifest,
    thresholds: RegressionThresholds,
) -> Result<PolicyAwareRegressionReport, ManifestComparisonError> {
    manifest.require_complete_for_snapshot(baseline)?;
    let registry = manifest.to_registry()?;
    Ok(compare_snapshots(
        baseline,
        current,
        &registry,
        thresholds,
    )?)
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_objective(hasher: &mut blake3::Hasher, objective: &MetricObjective) {
    match objective {
        MetricObjective::HigherIsBetter => {
            hasher.update(&[1]);
        }
        MetricObjective::LowerIsBetter => {
            hasher.update(&[2]);
        }
        MetricObjective::TargetRange { min, max } => {
            hasher.update(&[3]);
            hasher.update(&min.to_bits().to_le_bytes());
            hasher.update(&max.to_bits().to_le_bytes());
        }
        MetricObjective::TwoSidedStable {
            warning_abs,
            critical_abs,
        } => {
            hasher.update(&[4]);
            hasher.update(&warning_abs.to_bits().to_le_bytes());
            hasher.update(&critical_abs.to_bits().to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::MetricValue;
    use crate::regression_contract::ComparisonDisposition;

    fn value(mean: f64) -> MetricValue {
        MetricValue {
            mean,
            std_dev: 0.01,
            n: 20,
            ci_lower: mean - 0.02,
            ci_upper: mean + 0.02,
        }
    }

    fn snapshot(name: &str, entries: &[(&str, &str, f64)]) -> RegressionSnapshot {
        let mut metrics = BTreeMap::new();
        for (benchmark, metric, mean) in entries {
            metrics
                .entry((*benchmark).to_string())
                .or_insert_with(BTreeMap::new)
                .insert((*metric).to_string(), value(*mean));
        }
        RegressionSnapshot {
            name: name.to_string(),
            timestamp: "2026-09-14T00:00:00Z".to_string(),
            git_hash: None,
            config_summary: "test".to_string(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics,
        }
    }

    fn entry(
        benchmark: &str,
        metric: &str,
        policy: MetricComparisonPolicy,
    ) -> MetricPolicyEntry {
        MetricPolicyEntry::new(MetricKey::new(benchmark, metric), policy)
    }

    #[test]
    fn manifest_digest_is_independent_of_entry_order() {
        let a = entry(
            "Bench",
            "accuracy",
            MetricComparisonPolicy::higher("accuracy-v1", 1),
        );
        let b = entry(
            "Bench",
            "error",
            MetricComparisonPolicy::lower("error-v1", 1),
        );
        let first = MetricPolicyManifest::new("battery-v1", 1, vec![a.clone(), b.clone()]);
        let second = MetricPolicyManifest::new("battery-v1", 1, vec![b, a]);
        assert_eq!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn duplicate_metric_keys_fail_closed() {
        let key = MetricKey::new("Bench", "accuracy");
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![
                MetricPolicyEntry::new(
                    key.clone(),
                    MetricComparisonPolicy::higher("a", 1),
                ),
                MetricPolicyEntry::new(key.clone(), MetricComparisonPolicy::lower("b", 1)),
            ],
        );
        assert_eq!(
            manifest.validate_structure(),
            Err(MetricPolicyManifestError::DuplicateMetricKey(key))
        );
    }

    #[test]
    fn coverage_reports_missing_and_extra_policies_explicitly() {
        let baseline = snapshot(
            "baseline",
            &[("Bench", "accuracy", 0.9), ("Bench", "error", 0.1)],
        );
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![
                entry(
                    "Bench",
                    "accuracy",
                    MetricComparisonPolicy::higher("accuracy-v1", 1),
                ),
                entry(
                    "Other",
                    "latency",
                    MetricComparisonPolicy::lower("latency-v1", 1),
                ),
            ],
        );
        let coverage = manifest.coverage_for_snapshot(&baseline).unwrap();
        assert_eq!(coverage.baseline_required, 2);
        assert_eq!(coverage.covered, 1);
        assert_eq!(coverage.missing, vec![MetricKey::new("Bench", "error")]);
        assert_eq!(coverage.extra, vec![MetricKey::new("Other", "latency")]);
        assert_eq!(
            manifest.require_complete_for_snapshot(&baseline),
            Err(MetricPolicyManifestError::MissingBaselinePolicies(vec![
                MetricKey::new("Bench", "error"),
            ]))
        );
    }

    #[test]
    fn complete_manifest_drives_qualified_comparator_without_name_inference() {
        let baseline = snapshot("baseline", &[("Bench", "error", 0.50)]);
        let current = snapshot("current", &[("Bench", "error", 0.40)]);
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "error",
                MetricComparisonPolicy::lower("explicit-error-policy", 1),
            )],
        );
        let report = compare_snapshots_with_manifest(
            &baseline,
            &current,
            &manifest,
            RegressionThresholds::new(0.05, 0.10).unwrap(),
        )
        .unwrap();
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].disposition, ComparisonDisposition::Pass);
        assert_eq!(
            report.results[0].policy.as_ref().unwrap().policy_id,
            "explicit-error-policy"
        );
    }

    #[test]
    fn missing_policy_blocks_before_numeric_comparison() {
        let baseline = snapshot(
            "baseline",
            &[("Bench", "accuracy", 0.9), ("Bench", "error", 0.1)],
        );
        let current = snapshot(
            "current",
            &[("Bench", "accuracy", 0.9), ("Bench", "error", 0.1)],
        );
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "accuracy",
                MetricComparisonPolicy::higher("accuracy-v1", 1),
            )],
        );
        assert_eq!(
            compare_snapshots_with_manifest(
                &baseline,
                &current,
                &manifest,
                RegressionThresholds::new(0.05, 0.10).unwrap(),
            ),
            Err(ManifestComparisonError::Manifest(
                MetricPolicyManifestError::MissingBaselinePolicies(vec![MetricKey::new(
                    "Bench", "error"
                )])
            ))
        );
    }

    #[test]
    fn policy_semantics_remain_comparator_authority() {
        let baseline = snapshot("baseline", &[("Bench", "score", 0.5)]);
        let current = snapshot("current", &[("Bench", "score", 0.5)]);
        let invalid_policy = MetricComparisonPolicy::target_range("bad-range", 1, 1.0, 0.0);
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry("Bench", "score", invalid_policy)],
        );

        // Structural manifest validation deliberately does not duplicate the
        // comparator's numeric policy-validation authority.
        assert!(manifest.validate_structure().is_ok());
        let report = compare_snapshots_with_manifest(
            &baseline,
            &current,
            &manifest,
            RegressionThresholds::new(0.05, 0.10).unwrap(),
        )
        .unwrap();
        assert_eq!(
            report.results[0].disposition,
            ComparisonDisposition::InvalidPolicy
        );
    }

    #[test]
    fn policy_identity_changes_manifest_digest_even_when_objective_matches() {
        let first = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "metric",
                MetricComparisonPolicy::higher("metric-policy", 1),
            )],
        );
        let second = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "metric",
                MetricComparisonPolicy::higher("metric-policy", 2),
            )],
        );
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn objective_change_changes_manifest_digest() {
        let higher = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "metric",
                MetricComparisonPolicy::higher("metric-v1", 1),
            )],
        );
        let lower = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "metric",
                MetricComparisonPolicy::lower("metric-v1", 1),
            )],
        );
        assert_ne!(higher.digest_hex().unwrap(), lower.digest_hex().unwrap());
    }

    #[test]
    fn unsupported_snapshot_schema_fails_before_coverage() {
        let mut baseline = snapshot("baseline", &[("Bench", "metric", 1.0)]);
        baseline.schema_version = "legacy".to_string();
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "metric",
                MetricComparisonPolicy::higher("metric-v1", 1),
            )],
        );
        assert_eq!(
            manifest.coverage_for_snapshot(&baseline),
            Err(MetricPolicyManifestError::SnapshotSchemaMismatch {
                found: "legacy".to_string(),
                required: SNAPSHOT_SCHEMA_VERSION.to_string(),
            })
        );
    }

    #[test]
    fn serialized_manifest_omission_fails_closed() {
        let manifest = MetricPolicyManifest::new(
            "battery-v1",
            1,
            vec![entry(
                "Bench",
                "metric",
                MetricComparisonPolicy::higher("metric-v1", 1),
            )],
        );
        let mut json = serde_json::to_value(&manifest).unwrap();
        json.as_object_mut().unwrap().remove("entries");
        assert!(serde_json::from_value::<MetricPolicyManifest>(json).is_err());
    }
}
