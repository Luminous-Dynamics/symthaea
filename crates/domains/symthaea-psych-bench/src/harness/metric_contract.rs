// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned metric semantics for psychological benchmark comparisons.
//!
//! This contract does not infer meaning from metric names. Every comparison-qualified
//! metric must declare its estimand, unit/domain, direction, sample unit, uncertainty
//! treatment, and comparison rule. Missing semantics remain missing.

use crate::harness::report::MetricValue;
use crate::harness::snapshot::{RegressionSnapshot, SNAPSHOT_SCHEMA_VERSION};
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const METRIC_CONTRACT_SCHEMA_VERSION: &str = "symthaea-psych-metric-contract-v1";
const METRIC_CONTRACT_DOMAIN_SEPARATOR: &[u8] = b"symthaea:psych-bench-metric-contract:v1\n";

/// Numeric support/domain of a metric's reported values. Units are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricValueDomainV1 {
    Real,
    NonNegative,
    UnitInterval,
    IntegerCount,
}

/// Direction is explicit even for metrics that are diagnostic-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetricDirectionV1 {
    HigherIsBetter,
    LowerIsBetter,
    TargetRange { lower: f64, upper: f64 },
    DiagnosticOnly,
}

/// Declared uncertainty treatment, including deterministic outputs and missing estimates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UncertaintyPolicyV1 {
    Empirical {
        method: String,
        confidence_level: f64,
    },
    Deterministic {
        verification_method: String,
    },
    NotEstimated {
        reason: String,
    },
}

/// Per-metric comparison rule, with thresholds in the metric's declared scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComparisonRuleV1 {
    AbsoluteDelta {
        warning_delta: f64,
        critical_delta: f64,
    },
    RelativeDelta {
        warning_fraction: f64,
        critical_fraction: f64,
    },
    NoClassification {
        rationale: String,
    },
}

/// Versioned interpretation for one benchmark/metric key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricDefinitionV1 {
    pub benchmark: String,
    pub metric: String,
    pub definition_version: u32,
    pub estimand: String,
    pub unit: String,
    pub value_domain: MetricValueDomainV1,
    pub direction: MetricDirectionV1,
    /// The sample unit represented by MetricValue.n for empirical uncertainty.
    pub sample_unit: String,
    pub uncertainty: UncertaintyPolicyV1,
    pub comparison: ComparisonRuleV1,
}

impl MetricDefinitionV1 {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for (field, value) in [
            ("benchmark", self.benchmark.as_str()),
            ("metric", self.metric.as_str()),
            ("estimand", self.estimand.as_str()),
            ("unit", self.unit.as_str()),
            ("sample_unit", self.sample_unit.as_str()),
        ] {
            if value.trim().is_empty() {
                errors.push(format!("{field} must not be empty"));
            }
            if value.chars().any(char::is_control) {
                errors.push(format!("{field} must not contain control characters"));
            }
        }
        if self.definition_version == 0 {
            errors.push("definition_version must be greater than zero".to_string());
        }

        if let MetricDirectionV1::TargetRange { lower, upper } = &self.direction {
            if !lower.is_finite() || !upper.is_finite() || *lower > *upper {
                errors.push(
                    "target-range bounds must be finite and lower must not exceed upper"
                        .to_string(),
                );
            }
        }

        match &self.uncertainty {
            UncertaintyPolicyV1::Empirical {
                method,
                confidence_level,
            } => {
                if method.trim().is_empty() {
                    errors.push("empirical uncertainty method must not be empty".to_string());
                }
                if !confidence_level.is_finite()
                    || *confidence_level <= 0.0
                    || *confidence_level >= 1.0
                {
                    errors.push(
                        "empirical confidence_level must be finite and strictly between zero and one"
                            .to_string(),
                    );
                }
            }
            UncertaintyPolicyV1::Deterministic {
                verification_method,
            } => {
                if verification_method.trim().is_empty() {
                    errors.push(
                        "deterministic verification_method must not be empty".to_string(),
                    );
                }
            }
            UncertaintyPolicyV1::NotEstimated { reason } => {
                if reason.trim().is_empty() {
                    errors.push("not-estimated uncertainty requires a reason".to_string());
                }
                if !matches!(&self.comparison, ComparisonRuleV1::NoClassification { .. }) {
                    errors.push(
                        "metrics without an uncertainty estimate must use NoClassification"
                            .to_string(),
                    );
                }
            }
        }

        match &self.comparison {
            ComparisonRuleV1::AbsoluteDelta {
                warning_delta,
                critical_delta,
            } => {
                if !warning_delta.is_finite()
                    || !critical_delta.is_finite()
                    || *warning_delta < 0.0
                    || *critical_delta <= *warning_delta
                {
                    errors.push(
                        "absolute thresholds must be finite/non-negative and critical_delta must exceed warning_delta"
                            .to_string(),
                    );
                }
                if self.value_domain == MetricValueDomainV1::UnitInterval
                    && *critical_delta > 1.0
                {
                    errors.push(
                        "unit-interval absolute critical_delta must not exceed one".to_string(),
                    );
                }
            }
            ComparisonRuleV1::RelativeDelta {
                warning_fraction,
                critical_fraction,
            } => {
                if !warning_fraction.is_finite()
                    || !critical_fraction.is_finite()
                    || *warning_fraction < 0.0
                    || *critical_fraction <= *warning_fraction
                {
                    errors.push(
                        "relative thresholds must be finite/non-negative and critical_fraction must exceed warning_fraction"
                            .to_string(),
                    );
                }
            }
            ComparisonRuleV1::NoClassification { rationale } => {
                if rationale.trim().is_empty() {
                    errors.push("NoClassification requires a non-empty rationale".to_string());
                }
            }
        }

        if matches!(&self.direction, MetricDirectionV1::DiagnosticOnly)
            && !matches!(&self.comparison, ComparisonRuleV1::NoClassification { .. })
        {
            errors.push("DiagnosticOnly metrics must use NoClassification".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// A complete policy for one exact benchmark/metric population. Entries are
/// canonicalized by key order and duplicate keys are rejected.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricContractV1 {
    schema_version: String,
    definitions: Vec<MetricDefinitionV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetricContractV1Wire {
    schema_version: String,
    definitions: Vec<MetricDefinitionV1>,
}

impl<'de> Deserialize<'de> for MetricContractV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MetricContractV1Wire::deserialize(deserializer)?;
        if wire.schema_version != METRIC_CONTRACT_SCHEMA_VERSION {
            return Err(D::Error::custom(format!(
                "unsupported metric contract schema {:?}; expected {:?}",
                wire.schema_version, METRIC_CONTRACT_SCHEMA_VERSION
            )));
        }
        Self::new(wire.definitions).map_err(|errors| D::Error::custom(errors.join("; ")))
    }
}

impl MetricContractV1 {
    pub fn new(
        mut definitions: Vec<MetricDefinitionV1>,
    ) -> Result<Self, Vec<String>> {
        definitions.sort_by(|left, right| {
            (&left.benchmark, &left.metric).cmp(&(&right.benchmark, &right.metric))
        });
        let contract = Self {
            schema_version: METRIC_CONTRACT_SCHEMA_VERSION.to_string(),
            definitions,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn definitions(&self) -> &[MetricDefinitionV1] {
        &self.definitions
    }

    pub fn definition(&self, benchmark: &str, metric: &str) -> Option<&MetricDefinitionV1> {
        self.definitions
            .binary_search_by(|item| {
                (item.benchmark.as_str(), item.metric.as_str()).cmp(&(benchmark, metric))
            })
            .ok()
            .map(|index| &self.definitions[index])
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.schema_version != METRIC_CONTRACT_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported metric contract schema {:?}; expected {:?}",
                self.schema_version, METRIC_CONTRACT_SCHEMA_VERSION
            ));
        }
        if self.definitions.is_empty() {
            errors.push("metric contract contains no definitions".to_string());
        }
        let mut seen = BTreeSet::new();
        for definition in &self.definitions {
            if !seen.insert((definition.benchmark.clone(), definition.metric.clone())) {
                errors.push(format!(
                    "duplicate metric definition for {}/{}",
                    definition.benchmark, definition.metric
                ));
            }
            if let Err(definition_errors) = definition.validate() {
                errors.extend(definition_errors.into_iter().map(|error| {
                    format!("{}/{}: {error}", definition.benchmark, definition.metric)
                }));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Fingerprint the canonical, validated policy bytes using a domain-separated
    /// BLAKE3 digest. This binds the chosen contract, not its scientific adequacy.
    pub fn fingerprint(&self) -> Result<String, Vec<String>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| vec![format!("cannot serialize metric contract: {error}")])?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(METRIC_CONTRACT_DOMAIN_SEPARATOR);
        hasher.update(&bytes);
        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Require exact equality between this contract's keys and every key in a
    /// snapshot. Empirical uncertainty requires at least two sample units; an
    /// n=1 metric must be explicitly deterministic or descriptive-only.
    pub fn validate_snapshot_population(
        &self,
        snapshot: &RegressionSnapshot,
    ) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if let Err(contract_errors) = self.validate() {
            errors.extend(contract_errors);
        }
        if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
            errors.push(format!(
                "snapshot schema {:?} is not supported; expected {:?}",
                snapshot.schema_version, SNAPSHOT_SCHEMA_VERSION
            ));
        }
        if let Err(snapshot_errors) = snapshot.validate() {
            errors.extend(
                snapshot_errors
                    .into_iter()
                    .map(|error| format!("snapshot: {error}")),
            );
        }

        let contract_keys: BTreeSet<(String, String)> = self
            .definitions
            .iter()
            .map(|definition| {
                (definition.benchmark.clone(), definition.metric.clone())
            })
            .collect();
        let snapshot_keys: BTreeSet<(String, String)> = snapshot
            .metrics
            .iter()
            .flat_map(|(benchmark, metrics)| {
                metrics
                    .keys()
                    .map(move |metric| (benchmark.clone(), metric.clone()))
            })
            .collect();

        for key in snapshot_keys.difference(&contract_keys) {
            errors.push(format!(
                "missing metric definition for snapshot key {}/{}",
                key.0, key.1
            ));
        }
        for key in contract_keys.difference(&snapshot_keys) {
            errors.push(format!(
                "metric definition has no snapshot value for {}/{}",
                key.0, key.1
            ));
        }

        for definition in &self.definitions {
            if let Some(value) = snapshot
                .metrics
                .get(&definition.benchmark)
                .and_then(|metrics| metrics.get(&definition.metric))
            {
                if matches!(&definition.uncertainty, UncertaintyPolicyV1::Empirical { .. })
                    && value.n < 2
                {
                    errors.push(format!(
                        "{}/{}: empirical uncertainty requires n >= 2; found n={}",
                        definition.benchmark, definition.metric, value.n
                    ));
                }
                validate_value_domain(definition, value, &mut errors);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn validate_value_domain(
    definition: &MetricDefinitionV1,
    value: &MetricValue,
    errors: &mut Vec<String>,
) {
    let valid_domain = |number: f64| match definition.value_domain {
        MetricValueDomainV1::Real => number.is_finite(),
        MetricValueDomainV1::NonNegative | MetricValueDomainV1::IntegerCount => {
            number.is_finite() && number >= 0.0
        }
        MetricValueDomainV1::UnitInterval => {
            number.is_finite() && (0.0..=1.0).contains(&number)
        }
    };

    if !valid_domain(value.mean) {
        errors.push(format!(
            "{}/{}: mean {} violates declared {:?} domain",
            definition.benchmark, definition.metric, value.mean, definition.value_domain
        ));
    }
    if definition.value_domain == MetricValueDomainV1::UnitInterval
        && (!valid_domain(value.ci_lower) || !valid_domain(value.ci_upper))
    {
        errors.push(format!(
            "{}/{}: confidence bounds violate the declared unit-interval domain",
            definition.benchmark, definition.metric
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::MetricValue;

    fn definition(
        benchmark: &str,
        metric: &str,
        uncertainty: UncertaintyPolicyV1,
        comparison: ComparisonRuleV1,
    ) -> MetricDefinitionV1 {
        MetricDefinitionV1 {
            benchmark: benchmark.to_string(),
            metric: metric.to_string(),
            definition_version: 1,
            estimand: "mean_response_accuracy".to_string(),
            unit: "proportion".to_string(),
            value_domain: MetricValueDomainV1::UnitInterval,
            direction: MetricDirectionV1::HigherIsBetter,
            sample_unit: "independent_seed_run".to_string(),
            uncertainty,
            comparison,
        }
    }

    fn empirical_definition(benchmark: &str, metric: &str) -> MetricDefinitionV1 {
        definition(
            benchmark,
            metric,
            UncertaintyPolicyV1::Empirical {
                method: "bootstrap_percentile_95_v1".to_string(),
                confidence_level: 0.95,
            },
            ComparisonRuleV1::RelativeDelta {
                warning_fraction: 0.05,
                critical_fraction: 0.10,
            },
        )
    }

    fn snapshot(values: &[(&str, &str, f64, usize)]) -> RegressionSnapshot {
        let mut metrics: BTreeMap<String, BTreeMap<String, MetricValue>> = BTreeMap::new();
        for (benchmark, metric, mean, n) in values {
            metrics.entry((*benchmark).to_string()).or_default().insert(
                (*metric).to_string(),
                MetricValue {
                    mean: *mean,
                    std_dev: 0.01,
                    n: *n,
                    ci_lower: (*mean - 0.02).max(0.0),
                    ci_upper: (*mean + 0.02).min(1.0),
                },
            );
        }
        RegressionSnapshot {
            name: "fixture".to_string(),
            timestamp: "2026-10-12T00:00:00Z".to_string(),
            git_hash: Some("0123456789012345678901234567890123456789".to_string()),
            config_summary: "fixture-only".to_string(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics,
        }
    }

    #[test]
    fn contract_fingerprint_is_deterministic_and_roundtrips() {
        let first = MetricContractV1::new(vec![
            empirical_definition("B", "accuracy"),
            empirical_definition("A", "latency"),
        ])
        .unwrap();
        let second = MetricContractV1::new(vec![
            empirical_definition("A", "latency"),
            empirical_definition("B", "accuracy"),
        ])
        .unwrap();
        assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());

        let json = serde_json::to_string(&first).unwrap();
        let loaded: MetricContractV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.fingerprint().unwrap(), first.fingerprint().unwrap());
        assert_eq!(loaded.schema_version(), METRIC_CONTRACT_SCHEMA_VERSION);
    }

    #[test]
    fn duplicate_metric_definitions_are_rejected() {
        let error = MetricContractV1::new(vec![
            empirical_definition("B", "accuracy"),
            empirical_definition("B", "accuracy"),
        ])
        .unwrap_err();
        assert!(error.iter().any(|message| message.contains("duplicate metric definition")));
    }

    #[test]
    fn exact_snapshot_population_is_required() {
        let contract = MetricContractV1::new(vec![
            empirical_definition("B", "accuracy"),
            empirical_definition("A", "latency"),
        ])
        .unwrap();
        let exact = snapshot(&[
            ("B", "accuracy", 0.8, 10),
            ("A", "latency", 0.4, 10),
        ]);
        assert!(contract.validate_snapshot_population(&exact).is_ok());

        let partial = snapshot(&[("B", "accuracy", 0.8, 10)]);
        let errors = contract.validate_snapshot_population(&partial).unwrap_err();
        assert!(errors.iter().any(|message| message.contains("metric definition has no snapshot value for A/latency")));
        assert!(!errors.iter().any(|message| message.contains("missing metric definition for snapshot key")));
    }

    #[test]
    fn snapshot_metric_without_definition_is_rejected() {
        let contract = MetricContractV1::new(vec![empirical_definition("B", "accuracy")]).unwrap();
        let expanded = snapshot(&[
            ("B", "accuracy", 0.8, 10),
            ("B", "latency", 0.4, 10),
        ]);
        let errors = contract.validate_snapshot_population(&expanded).unwrap_err();
        assert!(errors.iter().any(|message| message.contains("missing metric definition for snapshot key B/latency")));
    }

    #[test]
    fn empirical_policy_rejects_single_sample_and_bad_interval() {
        let contract = MetricContractV1::new(vec![empirical_definition("B", "accuracy")]).unwrap();
        let one_sample = snapshot(&[("B", "accuracy", 0.8, 1)]);
        let errors = contract.validate_snapshot_population(&one_sample).unwrap_err();
        assert!(errors.iter().any(|message| message.contains("empirical uncertainty requires n >= 2")));

        let invalid = definition(
            "B",
            "accuracy",
            UncertaintyPolicyV1::Empirical {
                method: "bootstrap".to_string(),
                confidence_level: 1.0,
            },
            ComparisonRuleV1::RelativeDelta {
                warning_fraction: 0.05,
                critical_fraction: 0.10,
            },
        );
        assert!(invalid.validate().unwrap_err().iter().any(|message| message.contains("confidence_level")));
    }

    #[test]
    fn diagnostic_or_not_estimated_metrics_cannot_receive_regression_thresholds() {
        let mut diagnostic = empirical_definition("B", "effect");
        diagnostic.direction = MetricDirectionV1::DiagnosticOnly;
        assert!(diagnostic.validate().unwrap_err().iter().any(|message| message.contains("DiagnosticOnly")));

        let not_estimated = definition(
            "B",
            "diagnostic",
            UncertaintyPolicyV1::NotEstimated {
                reason: "Single deterministic fixture with no sampling model".to_string(),
            },
            ComparisonRuleV1::RelativeDelta {
                warning_fraction: 0.05,
                critical_fraction: 0.10,
            },
        );
        assert!(not_estimated.validate().unwrap_err().iter().any(|message| message.contains("must use NoClassification")));
    }

    #[test]
    fn unknown_direction_cannot_deserialize() {
        let json = r#"{
            "benchmark":"B",
            "metric":"accuracy",
            "definition_version":1,
            "estimand":"mean_response_accuracy",
            "unit":"proportion",
            "value_domain":"unit_interval",
            "direction":{"kind":"unknown"},
            "sample_unit":"independent_seed_run",
            "uncertainty":{"kind":"empirical","method":"bootstrap","confidence_level":0.95},
            "comparison":{"kind":"relative_delta","warning_fraction":0.05,"critical_fraction":0.10}
        }"#;
        assert!(serde_json::from_str::<MetricDefinitionV1>(json).is_err());
    }

    #[test]
    fn unsupported_contract_version_cannot_deserialize() {
        let json = r#"{"schema_version":"symthaea-psych-metric-contract-v999","definitions":[]}"#;
        assert!(serde_json::from_str::<MetricContractV1>(json).is_err());
    }
}
