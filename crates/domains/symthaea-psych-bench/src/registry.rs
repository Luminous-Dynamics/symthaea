// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, fail-closed access to the PsychBench pilot registry.
//!
//! The bundled registry is intentionally incomplete. It describes selected
//! anchor tasks and PCE, not the complete benchmark crate inventory. Do not use
//! its entry count to publish a total benchmark count.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const REGISTRY_SCHEMA_VERSION: u32 = 1;
pub const PILOT_REGISTRY_JSON: &str =
    include_str!("../registry/psych_bench_registry_v0.1.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryMaturity {
    PilotIncomplete,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    SyntheticMechanism,
    ExternalTask,
    HumanNormed,
    EcologicalInteractive,
    TheoryProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkValidationStatus {
    MechanicsOnly,
    Unvalidated,
    ExternallyScored,
    HumanAnchored,
    CausallySupported,
    IndependentlyReplicated,
    TheoryProbeOnly,
    EnvironmentSmokeUnvalidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Higher,
    Lower,
    Target,
    ConstructDiagnostic,
    CalibrationDiagnostic,
    DiagnosticOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricValidationStatus {
    VerifiedInCode,
    FeatureGated,
    SemanticsNeedsAudit,
    NotImplemented,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlStatus {
    NotQualified,
    Partial,
    SmokeControlsOnly,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PsychBenchRegistry {
    pub registry_id: String,
    pub registry_version: String,
    pub schema_version: u32,
    pub maturity: RegistryMaturity,
    pub generated_at: String,
    pub completeness_note: String,
    pub evidence_classes: Vec<EvidenceClass>,
    pub comparison_policy: ComparisonPolicy,
    pub metric_policy: MetricPolicy,
    pub benchmarks: Vec<BenchmarkDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonPolicy {
    pub same_protocol_inter_system: String,
    pub human_normative: String,
    pub consciousness: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricPolicy {
    pub bounds: String,
    pub missing: String,
    pub unqualified: String,
    pub aggregation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDefinition {
    pub id: String,
    pub version: String,
    pub benchmark_name: String,
    pub domain: String,
    pub construct_card: ConstructCard,
    pub implementation_path: String,
    pub evidence_classes: Vec<EvidenceClass>,
    pub validation_status: BenchmarkValidationStatus,
    pub stimulus_provenance: StimulusProvenance,
    pub paradigm_sources: Vec<ParadigmSource>,
    pub metrics: Vec<MetricDefinition>,
    pub controls: Controls,
    pub comparison_eligibility: ComparisonEligibility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstructCard {
    pub construct: String,
    pub operationalization: String,
    pub exclusions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StimulusProvenance {
    pub kind: String,
    pub generator: String,
    pub external_dataset: Option<String>,
    pub licence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParadigmSource {
    pub citation: String,
    pub doi: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricDefinition {
    pub id: String,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub key_pattern: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Map<String, serde_json::Value>>,
    pub unit: String,
    pub bounds: MetricBounds,
    pub direction: MetricDirection,
    pub estimand: String,
    pub comparison_status: String,
    pub validation_status: MetricValidationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricBounds {
    pub lower: f64,
    pub upper: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Controls {
    pub status: ControlStatus,
    pub positive_controls: Vec<String>,
    pub negative_controls: Vec<String>,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonEligibility {
    pub inter_system_same_protocol: String,
    pub human_normative: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryValidationError {
    pub message: String,
}

impl RegistryValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RegistryValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RegistryValidationError {}

impl PsychBenchRegistry {
    /// Parse and validate the supplied JSON registry.
    pub fn from_json(json: &str) -> Result<Self, RegistryValidationError> {
        let registry: Self = serde_json::from_str(json)
            .map_err(|error| RegistryValidationError::new(format!("registry JSON: {error}")))?;
        registry.validate()?;
        Ok(registry)
    }

    /// Parse the bundled pilot registry and apply the same validation rules.
    pub fn pilot() -> Result<Self, RegistryValidationError> {
        Self::from_json(PILOT_REGISTRY_JSON)
    }

    /// Fail-closed structural and claim-boundary validation.
    pub fn validate(&self) -> Result<(), RegistryValidationError> {
        if self.registry_id != "org.luminousdynamics.symthaea.psychbench" {
            return Err(RegistryValidationError::new(
                "unexpected registry_id; refusing to interpret an unknown registry",
            ));
        }
        if self.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err(RegistryValidationError::new(format!(
                "unsupported schema_version {}; expected {}",
                self.schema_version, REGISTRY_SCHEMA_VERSION
            )));
        }
        if !valid_semver(&self.registry_version) {
            return Err(RegistryValidationError::new(
                "registry_version must be numeric major.minor.patch",
            ));
        }
        if self.generated_at.len() != 10
            || self.generated_at.as_bytes().get(4) != Some(&b'-')
            || self.generated_at.as_bytes().get(7) != Some(&b'-')
        {
            return Err(RegistryValidationError::new(
                "generated_at must use YYYY-MM-DD format",
            ));
        }
        if self.completeness_note.trim().is_empty() {
            return Err(RegistryValidationError::new(
                "completeness_note is required to prevent silent inventory overclaims",
            ));
        }
        if self.evidence_classes.is_empty() {
            return Err(RegistryValidationError::new(
                "registry must declare its supported evidence classes",
            ));
        }

        let declared_classes: BTreeSet<_> = self.evidence_classes.iter().copied().collect();
        if declared_classes.len() != self.evidence_classes.len() {
            return Err(RegistryValidationError::new(
                "registry evidence_classes contains duplicates",
            ));
        }

        let mut benchmark_ids = BTreeSet::new();
        for benchmark in &self.benchmarks {
            if !benchmark_ids.insert(benchmark.id.as_str()) {
                return Err(RegistryValidationError::new(format!(
                    "duplicate benchmark id: {}",
                    benchmark.id
                )));
            }
            validate_benchmark(benchmark, &declared_classes)?;
        }
        if self.benchmarks.is_empty() {
            return Err(RegistryValidationError::new(
                "registry must contain at least one benchmark definition",
            ));
        }

        Ok(())
    }
}

fn validate_benchmark(
    benchmark: &BenchmarkDefinition,
    declared_classes: &BTreeSet<EvidenceClass>,
) -> Result<(), RegistryValidationError> {
    if !valid_stable_id(&benchmark.id) {
        return Err(RegistryValidationError::new(format!(
            "invalid stable benchmark id: {}",
            benchmark.id
        )));
    }
    if !valid_semver(&benchmark.version) {
        return Err(RegistryValidationError::new(format!(
            "benchmark {} has invalid version",
            benchmark.id
        )));
    }
    if benchmark.benchmark_name.trim().is_empty()
        || benchmark.domain.trim().is_empty()
        || benchmark.construct_card.construct.trim().is_empty()
        || benchmark.construct_card.operationalization.trim().is_empty()
    {
        return Err(RegistryValidationError::new(format!(
            "benchmark {} has incomplete construct metadata",
            benchmark.id
        )));
    }
    if !benchmark
        .implementation_path
        .starts_with("crates/domains/symthaea-psych-bench/")
        || benchmark
            .implementation_path
            .split('/')
            .any(|segment| segment == "..")
    {
        return Err(RegistryValidationError::new(format!(
            "benchmark {} has unsafe or out-of-crate implementation_path",
            benchmark.id
        )));
    }
    if benchmark.evidence_classes.is_empty()
        || benchmark
            .evidence_classes
            .iter()
            .any(|class| !declared_classes.contains(class))
    {
        return Err(RegistryValidationError::new(format!(
            "benchmark {} uses missing or undeclared evidence classes",
            benchmark.id
        )));
    }
    if benchmark.metrics.is_empty() {
        return Err(RegistryValidationError::new(format!(
            "benchmark {} has no metric definitions",
            benchmark.id
        )));
    }
    if !benchmark
        .evidence_classes
        .contains(&EvidenceClass::HumanNormed)
        && !benchmark
            .comparison_eligibility
            .human_normative
            .contains("not_eligible")
    {
        return Err(RegistryValidationError::new(format!(
            "benchmark {} claims human-norm eligibility without human_normed evidence",
            benchmark.id
        )));
    }

    let mut metric_ids = BTreeSet::new();
    for metric in &benchmark.metrics {
        if metric.id.trim().is_empty() || !metric_ids.insert(metric.id.as_str()) {
            return Err(RegistryValidationError::new(format!(
                "benchmark {} has an empty or duplicate metric id: {}",
                benchmark.id, metric.id
            )));
        }
        if metric.key.is_some() == metric.key_pattern.is_some() {
            return Err(RegistryValidationError::new(format!(
                "metric {} in {} must define exactly one of key or key_pattern",
                metric.id, benchmark.id
            )));
        }
        if metric.unit.trim().is_empty() || metric.estimand.trim().is_empty() {
            return Err(RegistryValidationError::new(format!(
                "metric {} in {} has missing unit/estimand",
                metric.id, benchmark.id
            )));
        }
        if !metric.bounds.lower.is_finite()
            || metric.bounds.upper.is_some_and(|upper| {
                !upper.is_finite() || upper < metric.bounds.lower
            })
        {
            return Err(RegistryValidationError::new(format!(
                "metric {} in {} has invalid bounds",
                metric.id, benchmark.id
            )));
        }
        if metric.validation_status == MetricValidationStatus::SemanticsNeedsAudit
            && !metric.comparison_status.contains("not_eligible")
        {
            return Err(RegistryValidationError::new(format!(
                "metric {} in {} needs audit but is not excluded from comparison",
                metric.id, benchmark.id
            )));
        }
    }

    Ok(())
}

fn valid_semver(value: &str) -> bool {
    let parts: Vec<_> = value.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

fn valid_stable_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte))
        && value.as_bytes()[0].is_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_pilot_registry_parses_and_validates() {
        let registry = PsychBenchRegistry::pilot().expect("pilot registry should validate");
        assert_eq!(registry.schema_version, REGISTRY_SCHEMA_VERSION);
        assert_eq!(registry.maturity, RegistryMaturity::PilotIncomplete);
        assert_eq!(registry.benchmarks.len(), 7);
    }

    #[test]
    fn pilot_implementation_paths_exist_in_this_crate() {
        let registry = PsychBenchRegistry::pilot().expect("pilot registry");
        let crate_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for benchmark in &registry.benchmarks {
            let relative = benchmark
                .implementation_path
                .strip_prefix("crates/domains/symthaea-psych-bench/")
                .expect("path prefix checked by registry validator");
            assert!(
                crate_root.join(relative).is_file(),
                "missing implementation for {}: {}",
                benchmark.id,
                benchmark.implementation_path
            );
        }
    }

    #[test]
    fn registry_rejects_duplicate_benchmark_ids() {
        let mut registry = PsychBenchRegistry::pilot().expect("pilot registry");
        let duplicate = registry.benchmarks[0].clone();
        registry.benchmarks.push(duplicate);
        let error = registry.validate().expect_err("duplicate ID must fail closed");
        assert!(error.message.contains("duplicate benchmark id"));
    }

    #[test]
    fn registry_rejects_invalid_metric_bounds() {
        let mut registry = PsychBenchRegistry::pilot().expect("pilot registry");
        registry.benchmarks[0].metrics[0].bounds = MetricBounds {
            lower: 1.0,
            upper: Some(0.0),
        };
        let error = registry.validate().expect_err("reversed bounds must fail");
        assert!(error.message.contains("invalid bounds"));
    }

    #[test]
    fn registry_rejects_human_norm_claim_without_human_evidence() {
        let mut registry = PsychBenchRegistry::pilot().expect("pilot registry");
        registry.benchmarks[0].comparison_eligibility.human_normative =
            "eligible".to_string();
        let error = registry
            .validate()
            .expect_err("unsupported human-norm claim must fail");
        assert!(error.message.contains("human_normed evidence"));
    }

    #[test]
    fn registry_rejects_metric_with_both_or_neither_selectors() {
        let mut registry = PsychBenchRegistry::pilot().expect("pilot registry");
        registry.benchmarks[0].metrics[0].key_pattern = None;
        let error = registry
            .validate()
            .expect_err("metric selector ambiguity must fail");
        assert!(error.message.contains("exactly one"));
    }
}
