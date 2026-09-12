// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zero-event statistical assurance over perception-crucible evidence.
//!
//! A campaign with zero observed false events can support an upper confidence
//! bound; it cannot establish a literal zero true event probability.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_statistics::{
    bernoulli_zero_event_upper_bound, poisson_zero_event_upper_rate,
};
use symthaea_perception_crucible::{CrucibleStatus, PerceptionCrucibleReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroEventMetric {
    FalsePositiveObservation,
    FalseTrack,
    IdentitySwitch,
}

impl ZeroEventMetric {
    pub fn observed_count(self, report: &PerceptionCrucibleReport) -> usize {
        match self {
            Self::FalsePositiveObservation => report.total_false_positives,
            Self::FalseTrack => report.total_false_tracks,
            Self::IdentitySwitch => report.total_identity_switches,
        }
    }
}

/// Exposure assumptions are supplied explicitly by a reviewed validation campaign.
/// Raw frame count is intentionally not an exposure model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ZeroEventExposureModel {
    IndependentBernoulli {
        independent_exposures: u64,
        exposure_definition: String,
    },
    Poisson {
        exposure: f64,
        exposure_unit: String,
    },
}

impl ZeroEventExposureModel {
    pub fn validate(&self) -> bool {
        match self {
            Self::IndependentBernoulli {
                independent_exposures,
                exposure_definition,
            } => *independent_exposures > 0 && !exposure_definition.trim().is_empty(),
            Self::Poisson {
                exposure,
                exposure_unit,
            } => exposure.is_finite() && *exposure > 0.0 && !exposure_unit.trim().is_empty(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroEventCampaignEvidence {
    pub campaign_id: String,
    pub exposure_model: ZeroEventExposureModel,
    /// Evidence justifying why the supplied exposures/model are appropriate.
    pub exposure_basis_ref: String,
    /// Dataset/run/crucible/evidence-lineage references.
    pub evidence_refs: Vec<String>,
}

impl ZeroEventCampaignEvidence {
    pub fn validate(&self) -> bool {
        !self.campaign_id.trim().is_empty()
            && self.exposure_model.validate()
            && !self.exposure_basis_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ZeroEventRequirement {
    BernoulliUpperProbabilityAtMost(f64),
    PoissonUpperRateAtMost {
        maximum_rate_per_exposure_unit: f64,
        exposure_unit: String,
    },
}

impl ZeroEventRequirement {
    pub fn validate(&self) -> bool {
        match self {
            Self::BernoulliUpperProbabilityAtMost(target) => {
                target.is_finite() && *target > 0.0 && *target < 1.0
            }
            Self::PoissonUpperRateAtMost {
                maximum_rate_per_exposure_unit,
                exposure_unit,
            } => {
                maximum_rate_per_exposure_unit.is_finite()
                    && *maximum_rate_per_exposure_unit > 0.0
                    && !exposure_unit.trim().is_empty()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroEventAssurancePolicy {
    pub policy_id: String,
    pub metric: ZeroEventMetric,
    pub confidence: f64,
    pub requirement: ZeroEventRequirement,
}

impl ZeroEventAssurancePolicy {
    pub fn validate(&self) -> bool {
        !self.policy_id.trim().is_empty()
            && self.confidence.is_finite()
            && self.confidence > 0.0
            && self.confidence < 1.0
            && self.requirement.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroEventAssuranceStatus {
    Supported,
    NotSupported,
    ObservedEvents,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ZeroEventAssuranceIssue {
    InvalidPolicy,
    InvalidCampaign,
    CrucibleDidNotPass,
    ExposureRequirementMismatch,
    ExposureUnitMismatch {
        campaign_unit: String,
        policy_unit: String,
    },
    ObservedEvents(usize),
    StatisticalBoundUnavailable,
    UpperBoundExceedsTarget {
        observed_upper_bound: f64,
        target_upper_bound: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroEventAssuranceReport {
    pub policy_id: String,
    pub campaign_id: String,
    pub crucible_policy_id: String,
    pub metric: ZeroEventMetric,
    pub observed_events: usize,
    pub confidence: f64,
    /// Probability for Bernoulli campaigns; event-rate per exposure unit for Poisson.
    pub upper_bound: Option<f64>,
    pub target_upper_bound: Option<f64>,
    pub bound_unit: String,
    pub status: ZeroEventAssuranceStatus,
    pub issues: Vec<ZeroEventAssuranceIssue>,
    pub evidence_refs: Vec<String>,
}

impl ZeroEventAssuranceReport {
    /// Statistical evidence can support a safety claim but cannot authorize action.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_zero_event_campaign(
    crucible: &PerceptionCrucibleReport,
    campaign: &ZeroEventCampaignEvidence,
    policy: &ZeroEventAssurancePolicy,
) -> ZeroEventAssuranceReport {
    let observed_events = policy.metric.observed_count(crucible);
    let mut evidence_refs = campaign.evidence_refs.clone();
    if !campaign.exposure_basis_ref.trim().is_empty() {
        evidence_refs.push(campaign.exposure_basis_ref.clone());
    }

    let mut report = ZeroEventAssuranceReport {
        policy_id: policy.policy_id.clone(),
        campaign_id: campaign.campaign_id.clone(),
        crucible_policy_id: crucible.policy_id.clone(),
        metric: policy.metric,
        observed_events,
        confidence: policy.confidence,
        upper_bound: None,
        target_upper_bound: None,
        bound_unit: String::new(),
        status: ZeroEventAssuranceStatus::Invalid,
        issues: Vec::new(),
        evidence_refs,
    };

    if !policy.validate() {
        report.issues.push(ZeroEventAssuranceIssue::InvalidPolicy);
        return report;
    }
    if !campaign.validate() {
        report.issues.push(ZeroEventAssuranceIssue::InvalidCampaign);
        return report;
    }
    if crucible.status != CrucibleStatus::Pass {
        report.status = ZeroEventAssuranceStatus::Blocked;
        report
            .issues
            .push(ZeroEventAssuranceIssue::CrucibleDidNotPass);
        return report;
    }
    if observed_events > 0 {
        report.status = ZeroEventAssuranceStatus::ObservedEvents;
        report
            .issues
            .push(ZeroEventAssuranceIssue::ObservedEvents(observed_events));
        return report;
    }

    let (upper, target, unit) = match (&campaign.exposure_model, &policy.requirement) {
        (
            ZeroEventExposureModel::IndependentBernoulli {
                independent_exposures,
                ..
            },
            ZeroEventRequirement::BernoulliUpperProbabilityAtMost(target),
        ) => {
            let Some(bound) =
                bernoulli_zero_event_upper_bound(*independent_exposures, policy.confidence)
            else {
                report
                    .issues
                    .push(ZeroEventAssuranceIssue::StatisticalBoundUnavailable);
                return report;
            };
            (bound.upper_event_probability, *target, "probability".to_string())
        }
        (
            ZeroEventExposureModel::Poisson {
                exposure,
                exposure_unit,
            },
            ZeroEventRequirement::PoissonUpperRateAtMost {
                maximum_rate_per_exposure_unit,
                exposure_unit: policy_unit,
            },
        ) => {
            if exposure_unit != policy_unit {
                report.issues.push(ZeroEventAssuranceIssue::ExposureUnitMismatch {
                    campaign_unit: exposure_unit.clone(),
                    policy_unit: policy_unit.clone(),
                });
                return report;
            }
            let Some(bound) = poisson_zero_event_upper_rate(*exposure, policy.confidence) else {
                report
                    .issues
                    .push(ZeroEventAssuranceIssue::StatisticalBoundUnavailable);
                return report;
            };
            (
                bound.upper_event_rate_per_exposure_unit,
                *maximum_rate_per_exposure_unit,
                format!("events/{exposure_unit}"),
            )
        }
        _ => {
            report
                .issues
                .push(ZeroEventAssuranceIssue::ExposureRequirementMismatch);
            return report;
        }
    };

    report.upper_bound = Some(upper);
    report.target_upper_bound = Some(target);
    report.bound_unit = unit;
    if upper <= target {
        report.status = ZeroEventAssuranceStatus::Supported;
    } else {
        report.status = ZeroEventAssuranceStatus::NotSupported;
        report
            .issues
            .push(ZeroEventAssuranceIssue::UpperBoundExceedsTarget {
                observed_upper_bound: upper,
                target_upper_bound: target,
            });
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_perception_crucible::PerceptionCrucibleReport;

    fn crucible(status: CrucibleStatus, fp: usize, false_tracks: usize) -> PerceptionCrucibleReport {
        PerceptionCrucibleReport {
            policy_id: "crucible-v1".into(),
            status,
            assessed_scenarios: 4,
            total_frames: 10_000,
            total_negative_frames: 9_000,
            total_false_positives: fp,
            total_false_tracks: false_tracks,
            total_identity_switches: 0,
            classification_trials: 100,
            ood_trials: 50,
            ood_abstentions: 50,
            incomplete_classifications: 0,
            issues: Vec::new(),
        }
    }

    fn bernoulli_campaign(n: u64) -> ZeroEventCampaignEvidence {
        ZeroEventCampaignEvidence {
            campaign_id: "negative-scenes-v1".into(),
            exposure_model: ZeroEventExposureModel::IndependentBernoulli {
                independent_exposures: n,
                exposure_definition: "independently sampled negative operational scenes".into(),
            },
            exposure_basis_ref: "review:independence-analysis-v1".into(),
            evidence_refs: vec!["crucible-receipt:abc".into()],
        }
    }

    fn bernoulli_policy(target: f64) -> ZeroEventAssurancePolicy {
        ZeroEventAssurancePolicy {
            policy_id: "false-positive-zero-event-v1".into(),
            metric: ZeroEventMetric::FalsePositiveObservation,
            confidence: 0.95,
            requirement: ZeroEventRequirement::BernoulliUpperProbabilityAtMost(target),
        }
    }

    #[test]
    fn sufficiently_large_zero_event_campaign_supports_target() {
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Pass, 0, 0),
            &bernoulli_campaign(3_000),
            &bernoulli_policy(0.001),
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::Supported);
        assert!(report.upper_bound.unwrap() <= 0.001);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn small_zero_event_campaign_does_not_overclaim() {
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Pass, 0, 0),
            &bernoulli_campaign(100),
            &bernoulli_policy(0.001),
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::NotSupported);
        assert!(report.upper_bound.unwrap() > 0.001);
    }

    #[test]
    fn observed_false_positive_disables_zero_event_reasoning() {
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Pass, 1, 0),
            &bernoulli_campaign(10_000),
            &bernoulli_policy(0.001),
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::ObservedEvents);
        assert!(report.upper_bound.is_none());
    }

    #[test]
    fn nonpassing_crucible_blocks_statistical_claim() {
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Incomplete, 0, 0),
            &bernoulli_campaign(10_000),
            &bernoulli_policy(0.001),
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::Blocked);
    }

    #[test]
    fn missing_exposure_basis_is_invalid() {
        let mut campaign = bernoulli_campaign(10_000);
        campaign.exposure_basis_ref.clear();
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Pass, 0, 0),
            &campaign,
            &bernoulli_policy(0.001),
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::Invalid);
    }

    #[test]
    fn incompatible_exposure_model_and_requirement_is_invalid() {
        let campaign = ZeroEventCampaignEvidence {
            campaign_id: "hours-v1".into(),
            exposure_model: ZeroEventExposureModel::Poisson {
                exposure: 10_000.0,
                exposure_unit: "hours".into(),
            },
            exposure_basis_ref: "review:poisson-model".into(),
            evidence_refs: vec!["crucible-receipt:abc".into()],
        };
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Pass, 0, 0),
            &campaign,
            &bernoulli_policy(0.001),
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::Invalid);
        assert!(report
            .issues
            .contains(&ZeroEventAssuranceIssue::ExposureRequirementMismatch));
    }

    #[test]
    fn poisson_campaign_preserves_named_exposure_unit() {
        let campaign = ZeroEventCampaignEvidence {
            campaign_id: "operating-hours-v1".into(),
            exposure_model: ZeroEventExposureModel::Poisson {
                exposure: 100_000.0,
                exposure_unit: "hours".into(),
            },
            exposure_basis_ref: "review:poisson-operating-hours".into(),
            evidence_refs: vec!["crucible-receipt:def".into()],
        };
        let policy = ZeroEventAssurancePolicy {
            policy_id: "false-track-rate-v1".into(),
            metric: ZeroEventMetric::FalseTrack,
            confidence: 0.99,
            requirement: ZeroEventRequirement::PoissonUpperRateAtMost {
                maximum_rate_per_exposure_unit: 0.0001,
                exposure_unit: "hours".into(),
            },
        };
        let report = assess_zero_event_campaign(
            &crucible(CrucibleStatus::Pass, 0, 0),
            &campaign,
            &policy,
        );
        assert_eq!(report.status, ZeroEventAssuranceStatus::Supported);
        assert_eq!(report.bound_unit, "events/hours");
    }
}
