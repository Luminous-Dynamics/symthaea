// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded rare-event campaign assessment.
//!
//! Importance sampling can improve coverage of dangerous scenarios, but only
//! when proposal and target probabilities are declared and weights remain
//! numerically credible. This module refuses unsupported probability claims.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RareEventOutcome {
    Safe,
    Unsafe,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RareEventSample {
    pub sample_id: String,
    pub family_id: String,
    pub seed: u64,
    pub target_probability: f64,
    pub proposal_probability: f64,
    pub outcome: RareEventOutcome,
    pub evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RareEventCampaignPolicy {
    pub required_families: BTreeSet<String>,
    pub minimum_samples_per_family: usize,
    pub minimum_effective_sample_size: f64,
    pub maximum_normalized_weight: f64,
    pub maximum_indeterminate_fraction: f64,
    pub maximum_estimated_unsafe_probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RareEventCampaignStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RareEventCampaignIssue {
    EmptyIdentity,
    DuplicateSample(String),
    MissingRequiredFamily(String),
    InsufficientFamilySamples {
        family_id: String,
        required: usize,
        observed: usize,
    },
    InvalidProbability(String),
    MissingEvidence(String),
    DuplicateSeed {
        family_id: String,
        seed: u64,
    },
    ExcessiveWeight {
        sample_id: String,
        normalized_weight: f64,
        maximum: f64,
    },
    InsufficientEffectiveSampleSize {
        observed: f64,
        required: f64,
    },
    ExcessiveIndeterminateFraction {
        observed: f64,
        maximum: f64,
    },
    UnsafeProbabilityExceeded {
        observed: f64,
        maximum: f64,
    },
    ZeroTotalWeight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RareEventFamilyReport {
    pub family_id: String,
    pub samples: usize,
    pub effective_sample_size: f64,
    pub estimated_unsafe_probability: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RareEventCampaignReport {
    pub status: RareEventCampaignStatus,
    pub total_samples: usize,
    pub effective_sample_size: f64,
    pub estimated_unsafe_probability: Option<f64>,
    pub indeterminate_fraction: f64,
    pub family_reports: Vec<RareEventFamilyReport>,
    pub issues: Vec<RareEventCampaignIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RareEventCampaignError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct RareEventCampaignAssessor {
    policy: RareEventCampaignPolicy,
}

impl RareEventCampaignAssessor {
    pub fn new(policy: RareEventCampaignPolicy) -> Result<Self, RareEventCampaignError> {
        if policy.required_families.is_empty()
            || policy.minimum_samples_per_family == 0
            || !policy.minimum_effective_sample_size.is_finite()
            || policy.minimum_effective_sample_size <= 0.0
            || !in_unit_interval(policy.maximum_normalized_weight)
            || policy.maximum_normalized_weight == 0.0
            || !in_unit_interval(policy.maximum_indeterminate_fraction)
            || !in_unit_interval(policy.maximum_estimated_unsafe_probability)
        {
            return Err(RareEventCampaignError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, samples: &[RareEventSample]) -> RareEventCampaignReport {
        let mut issues = Vec::new();
        let mut sample_ids = BTreeSet::new();
        let mut family_seeds = BTreeSet::new();
        let mut by_family = BTreeMap::<&str, Vec<&RareEventSample>>::new();
        let mut weights = Vec::with_capacity(samples.len());

        for sample in samples {
            if sample.sample_id.trim().is_empty()
                || sample.family_id.trim().is_empty()
                || sample.evidence_id.trim().is_empty()
            {
                if sample.evidence_id.trim().is_empty() {
                    issues.push(RareEventCampaignIssue::MissingEvidence(
                        sample.sample_id.clone(),
                    ));
                } else {
                    issues.push(RareEventCampaignIssue::EmptyIdentity);
                }
            }
            if !sample_ids.insert(sample.sample_id.as_str()) {
                issues.push(RareEventCampaignIssue::DuplicateSample(
                    sample.sample_id.clone(),
                ));
            }
            if !family_seeds.insert((sample.family_id.as_str(), sample.seed)) {
                issues.push(RareEventCampaignIssue::DuplicateSeed {
                    family_id: sample.family_id.clone(),
                    seed: sample.seed,
                });
            }
            if !sample.target_probability.is_finite()
                || !sample.proposal_probability.is_finite()
                || sample.target_probability < 0.0
                || sample.target_probability > 1.0
                || sample.proposal_probability <= 0.0
                || sample.proposal_probability > 1.0
            {
                issues.push(RareEventCampaignIssue::InvalidProbability(
                    sample.sample_id.clone(),
                ));
                weights.push(f64::NAN);
            } else {
                weights.push(sample.target_probability / sample.proposal_probability);
            }
            by_family
                .entry(sample.family_id.as_str())
                .or_default()
                .push(sample);
        }

        for family in &self.policy.required_families {
            match by_family.get(family.as_str()) {
                None => issues.push(RareEventCampaignIssue::MissingRequiredFamily(
                    family.clone(),
                )),
                Some(entries) if entries.len() < self.policy.minimum_samples_per_family => {
                    issues.push(RareEventCampaignIssue::InsufficientFamilySamples {
                        family_id: family.clone(),
                        required: self.policy.minimum_samples_per_family,
                        observed: entries.len(),
                    });
                }
                Some(_) => {}
            }
        }

        let finite_weights = weights
            .iter()
            .copied()
            .filter(|weight| weight.is_finite() && *weight >= 0.0)
            .collect::<Vec<_>>();
        let total_weight: f64 = finite_weights.iter().sum();
        let sum_sq: f64 = finite_weights.iter().map(|weight| weight * weight).sum();
        let effective_sample_size = if sum_sq > 0.0 {
            total_weight * total_weight / sum_sq
        } else {
            0.0
        };
        if total_weight <= 0.0 {
            issues.push(RareEventCampaignIssue::ZeroTotalWeight);
        } else {
            for (sample, weight) in samples.iter().zip(weights.iter().copied()) {
                if weight.is_finite() && weight >= 0.0 {
                    let normalized = weight / total_weight;
                    if normalized > self.policy.maximum_normalized_weight {
                        issues.push(RareEventCampaignIssue::ExcessiveWeight {
                            sample_id: sample.sample_id.clone(),
                            normalized_weight: normalized,
                            maximum: self.policy.maximum_normalized_weight,
                        });
                    }
                }
            }
        }
        if effective_sample_size < self.policy.minimum_effective_sample_size {
            issues.push(RareEventCampaignIssue::InsufficientEffectiveSampleSize {
                observed: effective_sample_size,
                required: self.policy.minimum_effective_sample_size,
            });
        }

        let indeterminate = samples
            .iter()
            .filter(|sample| sample.outcome == RareEventOutcome::Indeterminate)
            .count();
        let indeterminate_fraction = if samples.is_empty() {
            1.0
        } else {
            indeterminate as f64 / samples.len() as f64
        };
        if indeterminate_fraction > self.policy.maximum_indeterminate_fraction {
            issues.push(RareEventCampaignIssue::ExcessiveIndeterminateFraction {
                observed: indeterminate_fraction,
                maximum: self.policy.maximum_indeterminate_fraction,
            });
        }

        let estimated_unsafe_probability = weighted_unsafe_probability(samples, &weights);
        if estimated_unsafe_probability
            .is_some_and(|estimate| estimate > self.policy.maximum_estimated_unsafe_probability)
        {
            issues.push(RareEventCampaignIssue::UnsafeProbabilityExceeded {
                observed: estimated_unsafe_probability.unwrap_or(0.0),
                maximum: self.policy.maximum_estimated_unsafe_probability,
            });
        }

        let mut family_reports = Vec::new();
        for (family, entries) in by_family {
            let family_weights = entries
                .iter()
                .map(|sample| sample.target_probability / sample.proposal_probability)
                .collect::<Vec<_>>();
            let family_total: f64 = family_weights.iter().sum();
            let family_sq: f64 = family_weights.iter().map(|weight| weight * weight).sum();
            let family_ess = if family_sq > 0.0 {
                family_total * family_total / family_sq
            } else {
                0.0
            };
            let family_samples = entries
                .iter()
                .map(|sample| (**sample).clone())
                .collect::<Vec<_>>();
            family_reports.push(RareEventFamilyReport {
                family_id: family.to_string(),
                samples: entries.len(),
                effective_sample_size: family_ess,
                estimated_unsafe_probability: weighted_unsafe_probability(
                    &family_samples,
                    &family_weights,
                ),
            });
        }
        family_reports.sort_by(|left, right| left.family_id.cmp(&right.family_id));

        let status = if issues.iter().any(is_failure) {
            RareEventCampaignStatus::Fail
        } else if issues.is_empty() {
            RareEventCampaignStatus::Pass
        } else {
            RareEventCampaignStatus::Incomplete
        };
        RareEventCampaignReport {
            status,
            total_samples: samples.len(),
            effective_sample_size,
            estimated_unsafe_probability,
            indeterminate_fraction,
            family_reports,
            issues,
        }
    }
}

fn weighted_unsafe_probability(samples: &[RareEventSample], weights: &[f64]) -> Option<f64> {
    if samples.len() != weights.len()
        || samples
            .iter()
            .any(|sample| sample.outcome == RareEventOutcome::Indeterminate)
    {
        return None;
    }
    let mut total = 0.0;
    let mut unsafe_weight = 0.0;
    for (sample, weight) in samples.iter().zip(weights.iter().copied()) {
        if !weight.is_finite() || weight < 0.0 {
            return None;
        }
        total += weight;
        if sample.outcome == RareEventOutcome::Unsafe {
            unsafe_weight += weight;
        }
    }
    (total > 0.0).then_some(unsafe_weight / total)
}

fn in_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn is_failure(issue: &RareEventCampaignIssue) -> bool {
    matches!(
        issue,
        RareEventCampaignIssue::InvalidProbability(_)
            | RareEventCampaignIssue::ExcessiveWeight { .. }
            | RareEventCampaignIssue::InsufficientEffectiveSampleSize { .. }
            | RareEventCampaignIssue::UnsafeProbabilityExceeded { .. }
            | RareEventCampaignIssue::ZeroTotalWeight
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RareEventCampaignPolicy {
        RareEventCampaignPolicy {
            required_families: BTreeSet::from(["engine-out".into()]),
            minimum_samples_per_family: 3,
            minimum_effective_sample_size: 2.5,
            maximum_normalized_weight: 0.5,
            maximum_indeterminate_fraction: 0.0,
            maximum_estimated_unsafe_probability: 0.1,
        }
    }

    fn sample(id: &str, seed: u64, outcome: RareEventOutcome) -> RareEventSample {
        RareEventSample {
            sample_id: id.into(),
            family_id: "engine-out".into(),
            seed,
            target_probability: 0.01,
            proposal_probability: 0.1,
            outcome,
            evidence_id: format!("evidence-{id}"),
        }
    }

    #[test]
    fn balanced_safe_campaign_passes() {
        let report = RareEventCampaignAssessor::new(policy()).unwrap().assess(&[
            sample("a", 1, RareEventOutcome::Safe),
            sample("b", 2, RareEventOutcome::Safe),
            sample("c", 3, RareEventOutcome::Safe),
        ]);
        assert_eq!(report.status, RareEventCampaignStatus::Pass);
        assert_eq!(report.estimated_unsafe_probability, Some(0.0));
    }

    #[test]
    fn indeterminate_campaign_is_incomplete() {
        let report = RareEventCampaignAssessor::new(policy()).unwrap().assess(&[
            sample("a", 1, RareEventOutcome::Safe),
            sample("b", 2, RareEventOutcome::Indeterminate),
            sample("c", 3, RareEventOutcome::Safe),
        ]);
        assert_eq!(report.status, RareEventCampaignStatus::Incomplete);
        assert!(report.estimated_unsafe_probability.is_none());
    }

    #[test]
    fn observed_unsafe_rate_fails_gate() {
        let report = RareEventCampaignAssessor::new(policy()).unwrap().assess(&[
            sample("a", 1, RareEventOutcome::Unsafe),
            sample("b", 2, RareEventOutcome::Safe),
            sample("c", 3, RareEventOutcome::Safe),
        ]);
        assert_eq!(report.status, RareEventCampaignStatus::Fail);
    }
}
