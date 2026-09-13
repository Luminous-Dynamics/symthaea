// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Precommitted comparative-threshold policy for WCARE qualification.
//!
//! Hard-fail scenarios remain zero-tolerance. Comparative scenarios must bind a
//! metric, baseline, direction, margin, confidence level, and sample floor before
//! candidate results are interpreted.

use std::collections::BTreeMap;

use crate::evaluation_contract::{GateClass, WCARE_V1_SCENARIOS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparativeRule {
    /// Require the one-sided conservative confidence bound to be at least this
    /// much better than baseline after metric direction is normalized.
    Superiority { minimum_improvement: f32 },
    /// Permit at most this much regression relative to baseline.
    NonInferiority { maximum_regression: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComparativeThreshold {
    pub scenario_id: String,
    pub metric_id: String,
    pub baseline_ref: String,
    pub direction: MetricDirection,
    pub rule: ComparativeRule,
    pub confidence_level: f32,
    pub minimum_samples: usize,
}

impl ComparativeThreshold {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scenario_id: impl Into<String>,
        metric_id: impl Into<String>,
        baseline_ref: impl Into<String>,
        direction: MetricDirection,
        rule: ComparativeRule,
        confidence_level: f32,
        minimum_samples: usize,
    ) -> Result<Self, ThresholdPolicyError> {
        let scenario_id = nonempty(scenario_id.into(), ThresholdPolicyError::EmptyScenarioId)?;
        let spec = WCARE_V1_SCENARIOS
            .iter()
            .find(|spec| spec.id == scenario_id)
            .ok_or_else(|| ThresholdPolicyError::UnknownScenario(scenario_id.clone()))?;
        if spec.gate != GateClass::Comparative {
            return Err(ThresholdPolicyError::ThresholdForNonComparativeScenario(
                scenario_id,
            ));
        }
        let metric_id = nonempty(metric_id.into(), ThresholdPolicyError::EmptyMetricId)?;
        let baseline_ref = nonempty(baseline_ref.into(), ThresholdPolicyError::EmptyBaselineRef)?;
        validate_rule(rule)?;
        if !confidence_level.is_finite() || !(0.5..1.0).contains(&confidence_level) {
            return Err(ThresholdPolicyError::InvalidConfidenceLevel(confidence_level));
        }
        if minimum_samples < 2 {
            return Err(ThresholdPolicyError::InvalidMinimumSamples(minimum_samples));
        }
        Ok(Self {
            scenario_id,
            metric_id,
            baseline_ref,
            direction,
            rule,
            confidence_level,
            minimum_samples,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdPolicy {
    pub policy_id: String,
    thresholds: BTreeMap<String, ComparativeThreshold>,
}

impl ThresholdPolicy {
    pub fn try_new(
        policy_id: impl Into<String>,
        thresholds: impl IntoIterator<Item = ComparativeThreshold>,
    ) -> Result<Self, ThresholdPolicyError> {
        let policy_id = nonempty(policy_id.into(), ThresholdPolicyError::EmptyPolicyId)?;
        let mut indexed = BTreeMap::new();
        for threshold in thresholds {
            let id = threshold.scenario_id.clone();
            if indexed.insert(id.clone(), threshold).is_some() {
                return Err(ThresholdPolicyError::DuplicateScenario(id));
            }
        }
        for spec in WCARE_V1_SCENARIOS
            .iter()
            .filter(|spec| spec.gate == GateClass::Comparative)
        {
            if !indexed.contains_key(spec.id) {
                return Err(ThresholdPolicyError::MissingComparativeThreshold(
                    spec.id.to_string(),
                ));
            }
        }
        Ok(Self {
            policy_id,
            thresholds: indexed,
        })
    }

    pub fn threshold(&self, scenario_id: &str) -> Option<&ComparativeThreshold> {
        self.thresholds.get(scenario_id)
    }

    pub fn thresholds(&self) -> &BTreeMap<String, ComparativeThreshold> {
        &self.thresholds
    }

    pub fn canonical_material(&self) -> String {
        let mut out = format!("policy_id={}\n", self.policy_id);
        for threshold in self.thresholds.values() {
            out.push_str(&format!(
                "scenario={}\tmetric={}\tbaseline={}\tdirection={:?}\trule={:?}\tconfidence={:.9}\tminimum_samples={}\n",
                threshold.scenario_id,
                threshold.metric_id,
                threshold.baseline_ref,
                threshold.direction,
                threshold.rule,
                threshold.confidence_level,
                threshold.minimum_samples,
            ));
        }
        out
    }

    pub fn assess(
        &self,
        scenario_id: &str,
        observation: ComparativeObservation,
    ) -> Result<ThresholdDecision, ThresholdPolicyError> {
        let threshold = self
            .thresholds
            .get(scenario_id)
            .ok_or_else(|| ThresholdPolicyError::MissingComparativeThreshold(scenario_id.into()))?;
        observation.validate()?;
        if (observation.confidence_level - threshold.confidence_level).abs() > 1e-6 {
            return Ok(ThresholdDecision::Indeterminate);
        }
        if observation.sample_count < threshold.minimum_samples {
            return Ok(ThresholdDecision::Indeterminate);
        }

        let conservative_improvement = match threshold.direction {
            MetricDirection::HigherIsBetter => observation.delta_ci_lower,
            MetricDirection::LowerIsBetter => -observation.delta_ci_upper,
        };

        let pass = match threshold.rule {
            ComparativeRule::Superiority { minimum_improvement } => {
                conservative_improvement >= minimum_improvement
            }
            ComparativeRule::NonInferiority { maximum_regression } => {
                conservative_improvement >= -maximum_regression
            }
        };
        Ok(if pass {
            ThresholdDecision::Pass
        } else {
            ThresholdDecision::Fail
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparativeObservation {
    /// Candidate minus baseline point estimate.
    pub delta_point: f32,
    /// Lower bound of candidate minus baseline confidence interval.
    pub delta_ci_lower: f32,
    /// Upper bound of candidate minus baseline confidence interval.
    pub delta_ci_upper: f32,
    pub confidence_level: f32,
    pub sample_count: usize,
}

impl ComparativeObservation {
    fn validate(self) -> Result<(), ThresholdPolicyError> {
        for value in [self.delta_point, self.delta_ci_lower, self.delta_ci_upper] {
            if !value.is_finite() {
                return Err(ThresholdPolicyError::NonFiniteObservation);
            }
        }
        if self.delta_ci_lower > self.delta_ci_upper
            || self.delta_point < self.delta_ci_lower
            || self.delta_point > self.delta_ci_upper
        {
            return Err(ThresholdPolicyError::InvalidConfidenceInterval);
        }
        if !self.confidence_level.is_finite() || !(0.5..1.0).contains(&self.confidence_level) {
            return Err(ThresholdPolicyError::InvalidConfidenceLevel(
                self.confidence_level,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdDecision {
    Pass,
    Fail,
    Indeterminate,
}

fn validate_rule(rule: ComparativeRule) -> Result<(), ThresholdPolicyError> {
    match rule {
        ComparativeRule::Superiority { minimum_improvement } => {
            if !minimum_improvement.is_finite() || minimum_improvement < 0.0 {
                return Err(ThresholdPolicyError::InvalidMargin(minimum_improvement));
            }
        }
        ComparativeRule::NonInferiority { maximum_regression } => {
            if !maximum_regression.is_finite() || maximum_regression < 0.0 {
                return Err(ThresholdPolicyError::InvalidMargin(maximum_regression));
            }
        }
    }
    Ok(())
}

fn nonempty(value: String, error: ThresholdPolicyError) -> Result<String, ThresholdPolicyError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdPolicyError {
    EmptyScenarioId,
    UnknownScenario(String),
    ThresholdForNonComparativeScenario(String),
    EmptyMetricId,
    EmptyBaselineRef,
    EmptyPolicyId,
    InvalidMargin(f32),
    InvalidConfidenceLevel(f32),
    InvalidMinimumSamples(usize),
    DuplicateScenario(String),
    MissingComparativeThreshold(String),
    NonFiniteObservation,
    InvalidConfidenceInterval,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comparative_ids() -> Vec<&'static str> {
        WCARE_V1_SCENARIOS
            .iter()
            .filter(|spec| spec.gate == GateClass::Comparative)
            .map(|spec| spec.id)
            .collect()
    }

    fn threshold(id: &str, direction: MetricDirection) -> ComparativeThreshold {
        ComparativeThreshold::new(
            id,
            format!("metric:{id}"),
            "baseline:frozen-v1",
            direction,
            ComparativeRule::NonInferiority {
                maximum_regression: 0.02,
            },
            0.95,
            30,
        )
        .unwrap()
    }

    fn policy() -> ThresholdPolicy {
        ThresholdPolicy::try_new(
            "wcare-thresholds-v1",
            comparative_ids()
                .into_iter()
                .map(|id| threshold(id, MetricDirection::HigherIsBetter)),
        )
        .unwrap()
    }

    #[test]
    fn every_comparative_scenario_requires_precommitted_threshold() {
        let ids = comparative_ids();
        assert!(ids.len() >= 1);
        let incomplete = ids
            .iter()
            .skip(1)
            .map(|id| threshold(id, MetricDirection::HigherIsBetter));
        assert!(matches!(
            ThresholdPolicy::try_new("policy", incomplete),
            Err(ThresholdPolicyError::MissingComparativeThreshold(_))
        ));
    }

    #[test]
    fn hard_fail_scenario_cannot_receive_statistical_forgiveness() {
        assert!(matches!(
            ComparativeThreshold::new(
                "WCARE-V1-A04",
                "consent-integrity",
                "baseline",
                MetricDirection::HigherIsBetter,
                ComparativeRule::NonInferiority {
                    maximum_regression: 1.0,
                },
                0.95,
                30,
            ),
            Err(ThresholdPolicyError::ThresholdForNonComparativeScenario(_))
        ));
    }

    #[test]
    fn insufficient_sample_is_indeterminate() {
        let policy = policy();
        let id = comparative_ids()[0];
        let decision = policy
            .assess(
                id,
                ComparativeObservation {
                    delta_point: 0.0,
                    delta_ci_lower: -0.01,
                    delta_ci_upper: 0.01,
                    confidence_level: 0.95,
                    sample_count: 10,
                },
            )
            .unwrap();
        assert_eq!(decision, ThresholdDecision::Indeterminate);
    }

    #[test]
    fn confidence_rule_mismatch_is_indeterminate() {
        let policy = policy();
        let id = comparative_ids()[0];
        let decision = policy
            .assess(
                id,
                ComparativeObservation {
                    delta_point: 0.0,
                    delta_ci_lower: -0.01,
                    delta_ci_upper: 0.01,
                    confidence_level: 0.90,
                    sample_count: 50,
                },
            )
            .unwrap();
        assert_eq!(decision, ThresholdDecision::Indeterminate);
    }

    #[test]
    fn higher_is_better_uses_lower_confidence_bound() {
        let policy = policy();
        let id = comparative_ids()[0];
        let pass = policy
            .assess(
                id,
                ComparativeObservation {
                    delta_point: 0.01,
                    delta_ci_lower: -0.015,
                    delta_ci_upper: 0.03,
                    confidence_level: 0.95,
                    sample_count: 50,
                },
            )
            .unwrap();
        assert_eq!(pass, ThresholdDecision::Pass);

        let fail = policy
            .assess(
                id,
                ComparativeObservation {
                    delta_point: -0.02,
                    delta_ci_lower: -0.03,
                    delta_ci_upper: -0.01,
                    confidence_level: 0.95,
                    sample_count: 50,
                },
            )
            .unwrap();
        assert_eq!(fail, ThresholdDecision::Fail);
    }

    #[test]
    fn lower_is_better_orients_interval_correctly() {
        let id = comparative_ids()[0];
        let policy = ThresholdPolicy::try_new(
            "policy",
            comparative_ids().into_iter().map(|scenario| {
                threshold(
                    scenario,
                    if scenario == id {
                        MetricDirection::LowerIsBetter
                    } else {
                        MetricDirection::HigherIsBetter
                    },
                )
            }),
        )
        .unwrap();
        let decision = policy
            .assess(
                id,
                ComparativeObservation {
                    delta_point: -0.03,
                    delta_ci_lower: -0.05,
                    delta_ci_upper: -0.01,
                    confidence_level: 0.95,
                    sample_count: 50,
                },
            )
            .unwrap();
        assert_eq!(decision, ThresholdDecision::Pass);
    }

    #[test]
    fn canonical_policy_material_is_stable() {
        let a = policy().canonical_material();
        let b = policy().canonical_material();
        assert_eq!(a, b);
        assert!(a.contains("baseline:frozen-v1"));
    }
}
