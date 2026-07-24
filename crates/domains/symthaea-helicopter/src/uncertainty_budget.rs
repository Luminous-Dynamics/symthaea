// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative uncertainty-budget aggregation for flight decisions.
//!
//! Contributions in the same declared correlation group are summed linearly;
//! independent groups are combined by root-sum-square. Missing evidence and
//! missing required source classes produce an explicit `Incomplete` result
//! rather than optimistic zero uncertainty.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UncertaintySource {
    Calibration,
    Sensor,
    Estimator,
    AerodynamicModel,
    Environment,
    Timing,
    Actuation,
    Guidance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyContribution {
    pub contribution_id: String,
    pub source: UncertaintySource,
    /// One-standard-deviation uncertainty in the contribution's native units.
    pub sigma: f64,
    /// Local output sensitivity to one native unit of the contribution.
    pub sensitivity: f64,
    /// Contributions with the same group are treated as fully correlated.
    pub correlation_group: Option<String>,
    pub evidence_id: Option<String>,
}

impl UncertaintyContribution {
    pub fn output_sigma(&self) -> f64 {
        (self.sigma * self.sensitivity).abs()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyBudgetConfig {
    pub schema_version: String,
    pub budget_id: String,
    pub confidence_sigma_multiplier: f64,
    pub maximum_total_sigma: f64,
    pub minimum_remaining_margin: f64,
    pub maximum_single_contribution_fraction: f64,
    pub required_sources: Vec<UncertaintySource>,
    pub require_evidence_ids: bool,
}

impl Default for UncertaintyBudgetConfig {
    fn default() -> Self {
        Self {
            schema_version: "symthaea.helicopter.uncertainty-budget.v1".into(),
            budget_id: "default-flight-decision-budget".into(),
            confidence_sigma_multiplier: 3.0,
            maximum_total_sigma: 1.0,
            minimum_remaining_margin: 0.0,
            maximum_single_contribution_fraction: 0.7,
            required_sources: vec![
                UncertaintySource::Calibration,
                UncertaintySource::Sensor,
                UncertaintySource::AerodynamicModel,
            ],
            require_evidence_ids: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyBudgetStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UncertaintyBudgetIssue {
    MissingRequiredSource(UncertaintySource),
    MissingEvidence(String),
    TotalSigmaExceeded {
        observed: f64,
        maximum: f64,
    },
    RemainingMarginInsufficient {
        observed: f64,
        minimum: f64,
    },
    DominantContribution {
        contribution_id: String,
        fraction: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrelationGroupEvidence {
    pub group_id: String,
    pub contribution_ids: Vec<String>,
    pub linearly_combined_sigma: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyBudgetReport {
    pub schema_version: String,
    pub budget_id: String,
    pub status: UncertaintyBudgetStatus,
    pub nominal_margin: f64,
    pub total_sigma: f64,
    pub protected_margin: f64,
    pub dominant_contribution_id: Option<String>,
    pub dominant_fraction: f64,
    pub groups: Vec<CorrelationGroupEvidence>,
    pub issues: Vec<UncertaintyBudgetIssue>,
}

impl UncertaintyBudgetReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, UncertaintyBudgetError> {
        let mut canonical = self.clone();
        canonical.groups.sort_by(|a, b| a.group_id.cmp(&b.group_id));
        for group in &mut canonical.groups {
            group.contribution_ids.sort();
        }
        canonical.issues.sort_by_key(issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| UncertaintyBudgetError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, UncertaintyBudgetError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyBudgetError {
    InvalidConfiguration,
    InvalidNominalMargin,
    EmptyContributions,
    DuplicateContributionId(String),
    InvalidContribution(String),
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct UncertaintyBudgetEvaluator {
    config: UncertaintyBudgetConfig,
}

impl UncertaintyBudgetEvaluator {
    pub fn new(config: UncertaintyBudgetConfig) -> Result<Self, UncertaintyBudgetError> {
        let required: BTreeSet<_> = config.required_sources.iter().copied().collect();
        if config.schema_version.trim().is_empty()
            || config.budget_id.trim().is_empty()
            || !config.confidence_sigma_multiplier.is_finite()
            || config.confidence_sigma_multiplier <= 0.0
            || !config.maximum_total_sigma.is_finite()
            || config.maximum_total_sigma < 0.0
            || !config.minimum_remaining_margin.is_finite()
            || !config.maximum_single_contribution_fraction.is_finite()
            || !(0.0..=1.0).contains(&config.maximum_single_contribution_fraction)
            || required.len() != config.required_sources.len()
        {
            return Err(UncertaintyBudgetError::InvalidConfiguration);
        }
        Ok(Self { config })
    }

    pub fn evaluate(
        &self,
        nominal_margin: f64,
        contributions: &[UncertaintyContribution],
    ) -> Result<UncertaintyBudgetReport, UncertaintyBudgetError> {
        if !nominal_margin.is_finite() {
            return Err(UncertaintyBudgetError::InvalidNominalMargin);
        }
        if contributions.is_empty() {
            return Err(UncertaintyBudgetError::EmptyContributions);
        }

        let mut ids = BTreeSet::new();
        let mut observed_sources = BTreeSet::new();
        let mut groups: BTreeMap<String, (Vec<String>, f64)> = BTreeMap::new();
        let mut missing_evidence = Vec::new();
        let mut linear_total = 0.0;
        let mut dominant_id = None;
        let mut dominant_sigma = -1.0f64;

        for contribution in contributions {
            if contribution.contribution_id.trim().is_empty()
                || !contribution.sigma.is_finite()
                || contribution.sigma < 0.0
                || !contribution.sensitivity.is_finite()
                || contribution
                    .correlation_group
                    .as_deref()
                    .is_some_and(|group| group.trim().is_empty())
            {
                return Err(UncertaintyBudgetError::InvalidContribution(
                    contribution.contribution_id.clone(),
                ));
            }
            if !ids.insert(contribution.contribution_id.clone()) {
                return Err(UncertaintyBudgetError::DuplicateContributionId(
                    contribution.contribution_id.clone(),
                ));
            }

            observed_sources.insert(contribution.source);
            if self.config.require_evidence_ids
                && contribution
                    .evidence_id
                    .as_deref()
                    .is_none_or(|value| value.trim().is_empty())
            {
                missing_evidence.push(contribution.contribution_id.clone());
            }

            let sigma = contribution.output_sigma();
            linear_total += sigma;
            if sigma > dominant_sigma
                || (sigma == dominant_sigma
                    && dominant_id
                        .as_ref()
                        .is_none_or(|id| contribution.contribution_id < *id))
            {
                dominant_sigma = sigma;
                dominant_id = Some(contribution.contribution_id.clone());
            }
            let group_id = contribution
                .correlation_group
                .clone()
                .unwrap_or_else(|| format!("independent:{}", contribution.contribution_id));
            let entry = groups.entry(group_id).or_default();
            entry.0.push(contribution.contribution_id.clone());
            entry.1 += sigma;
        }

        let total_sigma = groups
            .values()
            .map(|(_, sigma)| sigma * sigma)
            .sum::<f64>()
            .sqrt();
        let protected_margin =
            nominal_margin - self.config.confidence_sigma_multiplier * total_sigma;
        let dominant_fraction = if linear_total > 0.0 {
            dominant_sigma.max(0.0) / linear_total
        } else {
            0.0
        };

        let mut issues = Vec::new();
        for source in &self.config.required_sources {
            if !observed_sources.contains(source) {
                issues.push(UncertaintyBudgetIssue::MissingRequiredSource(*source));
            }
        }
        for contribution_id in missing_evidence {
            issues.push(UncertaintyBudgetIssue::MissingEvidence(contribution_id));
        }
        if total_sigma > self.config.maximum_total_sigma {
            issues.push(UncertaintyBudgetIssue::TotalSigmaExceeded {
                observed: total_sigma,
                maximum: self.config.maximum_total_sigma,
            });
        }
        if protected_margin < self.config.minimum_remaining_margin {
            issues.push(UncertaintyBudgetIssue::RemainingMarginInsufficient {
                observed: protected_margin,
                minimum: self.config.minimum_remaining_margin,
            });
        }
        if dominant_fraction > self.config.maximum_single_contribution_fraction {
            issues.push(UncertaintyBudgetIssue::DominantContribution {
                contribution_id: dominant_id.clone().unwrap_or_default(),
                fraction: dominant_fraction,
            });
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                UncertaintyBudgetIssue::MissingRequiredSource(_)
                    | UncertaintyBudgetIssue::MissingEvidence(_)
            )
        });
        let failed = issues.iter().any(|issue| {
            matches!(
                issue,
                UncertaintyBudgetIssue::TotalSigmaExceeded { .. }
                    | UncertaintyBudgetIssue::RemainingMarginInsufficient { .. }
                    | UncertaintyBudgetIssue::DominantContribution { .. }
            )
        });
        let status = if failed {
            UncertaintyBudgetStatus::Fail
        } else if incomplete {
            UncertaintyBudgetStatus::Incomplete
        } else {
            UncertaintyBudgetStatus::Pass
        };

        Ok(UncertaintyBudgetReport {
            schema_version: self.config.schema_version.clone(),
            budget_id: self.config.budget_id.clone(),
            status,
            nominal_margin,
            total_sigma,
            protected_margin,
            dominant_contribution_id: dominant_id,
            dominant_fraction,
            groups: groups
                .into_iter()
                .map(|(group_id, (contribution_ids, linearly_combined_sigma))| {
                    CorrelationGroupEvidence {
                        group_id,
                        contribution_ids,
                        linearly_combined_sigma,
                    }
                })
                .collect(),
            issues,
        })
    }
}

fn issue_sort_key(issue: &UncertaintyBudgetIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contribution(
        id: &str,
        source: UncertaintySource,
        sigma: f64,
        group: Option<&str>,
    ) -> UncertaintyContribution {
        UncertaintyContribution {
            contribution_id: id.into(),
            source,
            sigma,
            sensitivity: 1.0,
            correlation_group: group.map(str::to_owned),
            evidence_id: Some(format!("evidence:{id}")),
        }
    }

    #[test]
    fn correlated_terms_sum_before_rss() {
        let evaluator = UncertaintyBudgetEvaluator::new(UncertaintyBudgetConfig {
            maximum_total_sigma: 10.0,
            minimum_remaining_margin: -10.0,
            ..Default::default()
        })
        .unwrap();
        let report = evaluator
            .evaluate(
                10.0,
                &[
                    contribution("cal-a", UncertaintySource::Calibration, 0.3, Some("cal")),
                    contribution("cal-b", UncertaintySource::Calibration, 0.4, Some("cal")),
                    contribution("sensor", UncertaintySource::Sensor, 0.5, None),
                    contribution("model", UncertaintySource::AerodynamicModel, 0.0, None),
                ],
            )
            .unwrap();
        let expected = (0.7f64.powi(2) + 0.5f64.powi(2)).sqrt();
        assert!((report.total_sigma - expected).abs() < 1e-12);
        assert_eq!(report.status, UncertaintyBudgetStatus::Pass);
    }

    #[test]
    fn missing_evidence_is_incomplete() {
        let evaluator =
            UncertaintyBudgetEvaluator::new(UncertaintyBudgetConfig::default()).unwrap();
        let mut items = vec![
            contribution("cal", UncertaintySource::Calibration, 0.01, None),
            contribution("sensor", UncertaintySource::Sensor, 0.01, None),
            contribution("model", UncertaintySource::AerodynamicModel, 0.01, None),
        ];
        items[1].evidence_id = None;
        let report = evaluator.evaluate(5.0, &items).unwrap();
        assert_eq!(report.status, UncertaintyBudgetStatus::Incomplete);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            UncertaintyBudgetIssue::MissingEvidence(id) if id == "sensor"
        )));
    }

    #[test]
    fn threshold_breach_fails() {
        let evaluator = UncertaintyBudgetEvaluator::new(UncertaintyBudgetConfig {
            maximum_total_sigma: 0.1,
            minimum_remaining_margin: 0.0,
            ..Default::default()
        })
        .unwrap();
        let report = evaluator
            .evaluate(
                1.0,
                &[
                    contribution("cal", UncertaintySource::Calibration, 0.2, None),
                    contribution("sensor", UncertaintySource::Sensor, 0.2, None),
                    contribution("model", UncertaintySource::AerodynamicModel, 0.2, None),
                ],
            )
            .unwrap();
        assert_eq!(report.status, UncertaintyBudgetStatus::Fail);
    }

    #[test]
    fn digest_is_order_stable() {
        let evaluator = UncertaintyBudgetEvaluator::new(UncertaintyBudgetConfig {
            maximum_total_sigma: 10.0,
            minimum_remaining_margin: -10.0,
            ..Default::default()
        })
        .unwrap();
        let mut items = vec![
            contribution("cal", UncertaintySource::Calibration, 0.1, None),
            contribution("sensor", UncertaintySource::Sensor, 0.1, None),
            contribution("model", UncertaintySource::AerodynamicModel, 0.1, None),
        ];
        let first = evaluator
            .evaluate(2.0, &items)
            .unwrap()
            .digest_fnv1a64()
            .unwrap();
        items.reverse();
        let second = evaluator
            .evaluate(2.0, &items)
            .unwrap()
            .digest_fnv1a64()
            .unwrap();
        assert_eq!(first, second);
    }
}
