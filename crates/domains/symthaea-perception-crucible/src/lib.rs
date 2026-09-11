// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-first perception crucibles for false alarms, identity instability,
//! explicit abstention, OOD behavior, and perception/authority separation.
//!
//! A crucible does not prove a system can never fail. It records whether a
//! reviewed set of stressful scenarios satisfies explicit release criteria.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use symthaea_selective_classification::{
    ClassificationAssessment, ClassificationDisposition,
};
use symthaea_tracking_eval::TrackingEvaluationReport;

/// Broad stress families. Deployments should populate these with environment-
/// specific recorded/simulated datasets rather than treating the enum itself as evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StressFamily {
    QuietBackground,
    Biology,
    Weather,
    OpticalArtifact,
    MaritimeEnvironment,
    SensorFault,
    CalibrationOrTimingFault,
    OutOfDistribution,
    SensorDisagreement,
    Mixed,
}

/// One classification trial whose expected OOD/non-OOD condition is known from
/// the scenario fixture rather than inferred from the classifier itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassificationTrial {
    pub trial_id: String,
    pub expected_ood: bool,
    pub assessment: ClassificationAssessment,
}

impl ClassificationTrial {
    pub fn validate(&self) -> bool {
        !self.trial_id.trim().is_empty()
    }
}

/// Evidence from one stress scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionScenarioEvidence {
    pub scenario_id: String,
    pub family: StressFamily,
    /// When true, the tracking ground truth must contain no relevant tracked objects.
    /// This is how long "boring sky/ocean" sequences become explicit evidence.
    pub expected_background_only: bool,
    pub tracking: TrackingEvaluationReport,
    pub classification_trials: Vec<ClassificationTrial>,
    /// Count supplied by an integration harness that checks whether any perception
    /// result bypassed the independent authority boundary. This is a hard invariant:
    /// any non-zero value fails regardless of release thresholds.
    pub perception_to_authority_boundary_violations: usize,
    pub evidence_refs: Vec<String>,
}

impl PerceptionScenarioEvidence {
    pub fn validate(&self) -> bool {
        !self.scenario_id.trim().is_empty()
            && self.tracking.evaluated_frames > 0
            && self.classification_trials.iter().all(ClassificationTrial::validate)
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
            && (!self.expected_background_only || self.tracking.negative_only_sequence)
    }
}

/// Explicit release criteria. There are intentionally no safety-critical defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionCruciblePolicy {
    pub policy_id: String,
    pub required_families: Vec<StressFamily>,
    pub minimum_frames_per_scenario: usize,
    pub minimum_negative_frames_per_background_scenario: usize,
    pub maximum_false_positives_per_negative_frame: f64,
    pub minimum_clean_negative_frame_fraction: f64,
    pub maximum_false_track_fraction: f64,
    pub maximum_identity_switches_per_true_positive: f64,
    pub minimum_ood_abstention_rate: f64,
    pub maximum_incomplete_classification_rate: f64,
}

impl PerceptionCruciblePolicy {
    pub fn validate(&self) -> bool {
        let required = self.required_families.iter().copied().collect::<BTreeSet<_>>();
        !self.policy_id.trim().is_empty()
            && !self.required_families.is_empty()
            && required.len() == self.required_families.len()
            && self.minimum_frames_per_scenario > 0
            && self.minimum_negative_frames_per_background_scenario > 0
            && finite_nonnegative(self.maximum_false_positives_per_negative_frame)
            && unit_interval(self.minimum_clean_negative_frame_fraction)
            && unit_interval(self.maximum_false_track_fraction)
            && finite_nonnegative(self.maximum_identity_switches_per_true_positive)
            && unit_interval(self.minimum_ood_abstention_rate)
            && unit_interval(self.maximum_incomplete_classification_rate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrucibleStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrucibleIssue {
    InvalidPolicy,
    InvalidScenario(String),
    MissingRequiredFamily(StressFamily),
    InsufficientFrames {
        scenario_id: String,
        observed: usize,
        required: usize,
    },
    InsufficientNegativeFrames {
        scenario_id: String,
        observed: usize,
        required: usize,
    },
    FalsePositiveRateExceeded {
        scenario_id: String,
        observed: f64,
        maximum: f64,
    },
    CleanNegativeFractionBelowMinimum {
        scenario_id: String,
        observed: f64,
        minimum: f64,
    },
    FalseTrackFractionExceeded {
        scenario_id: String,
        observed: f64,
        maximum: f64,
    },
    IdentitySwitchRateExceeded {
        scenario_id: String,
        observed: f64,
        maximum: f64,
    },
    OodAbstentionBelowMinimum {
        observed: f64,
        minimum: f64,
    },
    IncompleteClassificationRateExceeded {
        observed: f64,
        maximum: f64,
    },
    PerceptionAuthorityBoundaryViolation {
        scenario_id: String,
        count: usize,
    },
    NoOodTrials,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionCrucibleReport {
    pub policy_id: String,
    pub status: CrucibleStatus,
    pub assessed_scenarios: usize,
    pub total_frames: usize,
    pub total_negative_frames: usize,
    pub total_false_positives: usize,
    pub total_false_tracks: usize,
    pub total_identity_switches: usize,
    pub classification_trials: usize,
    pub ood_trials: usize,
    pub ood_abstentions: usize,
    pub incomplete_classifications: usize,
    pub issues: Vec<CrucibleIssue>,
}

impl PerceptionCrucibleReport {
    /// A release-assurance report cannot authorize physical action.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_perception_crucible(
    scenarios: &[PerceptionScenarioEvidence],
    policy: &PerceptionCruciblePolicy,
) -> PerceptionCrucibleReport {
    if !policy.validate() {
        return PerceptionCrucibleReport {
            policy_id: policy.policy_id.clone(),
            status: CrucibleStatus::Incomplete,
            assessed_scenarios: 0,
            total_frames: 0,
            total_negative_frames: 0,
            total_false_positives: 0,
            total_false_tracks: 0,
            total_identity_switches: 0,
            classification_trials: 0,
            ood_trials: 0,
            ood_abstentions: 0,
            incomplete_classifications: 0,
            issues: vec![CrucibleIssue::InvalidPolicy],
        };
    }

    let mut issues = Vec::new();
    let present_families = scenarios.iter().map(|scenario| scenario.family).collect::<BTreeSet<_>>();
    for family in &policy.required_families {
        if !present_families.contains(family) {
            issues.push(CrucibleIssue::MissingRequiredFamily(*family));
        }
    }

    let mut total_frames = 0usize;
    let mut total_negative_frames = 0usize;
    let mut total_false_positives = 0usize;
    let mut total_false_tracks = 0usize;
    let mut total_identity_switches = 0usize;
    let mut classification_trials = 0usize;
    let mut ood_trials = 0usize;
    let mut ood_abstentions = 0usize;
    let mut incomplete_classifications = 0usize;

    for scenario in scenarios {
        if !scenario.validate() {
            issues.push(CrucibleIssue::InvalidScenario(scenario.scenario_id.clone()));
            continue;
        }

        let tracking = &scenario.tracking;
        total_frames = total_frames.saturating_add(tracking.evaluated_frames);
        total_negative_frames = total_negative_frames.saturating_add(tracking.negative_frames);
        total_false_positives = total_false_positives.saturating_add(tracking.false_positives);
        total_false_tracks = total_false_tracks.saturating_add(tracking.false_track_count);
        total_identity_switches = total_identity_switches.saturating_add(tracking.identity_switches);

        if tracking.evaluated_frames < policy.minimum_frames_per_scenario {
            issues.push(CrucibleIssue::InsufficientFrames {
                scenario_id: scenario.scenario_id.clone(),
                observed: tracking.evaluated_frames,
                required: policy.minimum_frames_per_scenario,
            });
        }

        if scenario.expected_background_only {
            if tracking.negative_frames < policy.minimum_negative_frames_per_background_scenario {
                issues.push(CrucibleIssue::InsufficientNegativeFrames {
                    scenario_id: scenario.scenario_id.clone(),
                    observed: tracking.negative_frames,
                    required: policy.minimum_negative_frames_per_background_scenario,
                });
            }
            let fp_per_negative_frame = if tracking.negative_frames == 0 {
                f64::INFINITY
            } else {
                tracking.false_positives as f64 / tracking.negative_frames as f64
            };
            if fp_per_negative_frame > policy.maximum_false_positives_per_negative_frame {
                issues.push(CrucibleIssue::FalsePositiveRateExceeded {
                    scenario_id: scenario.scenario_id.clone(),
                    observed: fp_per_negative_frame,
                    maximum: policy.maximum_false_positives_per_negative_frame,
                });
            }
            if let Some(clean_fraction) = tracking.clean_negative_frame_fraction {
                if clean_fraction < policy.minimum_clean_negative_frame_fraction {
                    issues.push(CrucibleIssue::CleanNegativeFractionBelowMinimum {
                        scenario_id: scenario.scenario_id.clone(),
                        observed: clean_fraction,
                        minimum: policy.minimum_clean_negative_frame_fraction,
                    });
                }
            }
        }

        if let Some(false_track_fraction) = tracking.false_track_fraction {
            if false_track_fraction > policy.maximum_false_track_fraction {
                issues.push(CrucibleIssue::FalseTrackFractionExceeded {
                    scenario_id: scenario.scenario_id.clone(),
                    observed: false_track_fraction,
                    maximum: policy.maximum_false_track_fraction,
                });
            }
        }

        let identity_switch_rate = if tracking.true_positives == 0 {
            0.0
        } else {
            tracking.identity_switches as f64 / tracking.true_positives as f64
        };
        if identity_switch_rate > policy.maximum_identity_switches_per_true_positive {
            issues.push(CrucibleIssue::IdentitySwitchRateExceeded {
                scenario_id: scenario.scenario_id.clone(),
                observed: identity_switch_rate,
                maximum: policy.maximum_identity_switches_per_true_positive,
            });
        }

        if scenario.perception_to_authority_boundary_violations > 0 {
            issues.push(CrucibleIssue::PerceptionAuthorityBoundaryViolation {
                scenario_id: scenario.scenario_id.clone(),
                count: scenario.perception_to_authority_boundary_violations,
            });
        }

        for trial in &scenario.classification_trials {
            classification_trials = classification_trials.saturating_add(1);
            if trial.assessment.disposition == ClassificationDisposition::Incomplete {
                incomplete_classifications = incomplete_classifications.saturating_add(1);
            }
            if trial.expected_ood {
                ood_trials = ood_trials.saturating_add(1);
                if trial.assessment.disposition == ClassificationDisposition::Abstain {
                    ood_abstentions = ood_abstentions.saturating_add(1);
                }
            }
        }
    }

    if ood_trials == 0 {
        issues.push(CrucibleIssue::NoOodTrials);
    } else {
        let rate = ood_abstentions as f64 / ood_trials as f64;
        if rate < policy.minimum_ood_abstention_rate {
            issues.push(CrucibleIssue::OodAbstentionBelowMinimum {
                observed: rate,
                minimum: policy.minimum_ood_abstention_rate,
            });
        }
    }

    if classification_trials > 0 {
        let rate = incomplete_classifications as f64 / classification_trials as f64;
        if rate > policy.maximum_incomplete_classification_rate {
            issues.push(CrucibleIssue::IncompleteClassificationRateExceeded {
                observed: rate,
                maximum: policy.maximum_incomplete_classification_rate,
            });
        }
    }

    let incomplete = issues.iter().any(|issue| {
        matches!(
            issue,
            CrucibleIssue::InvalidPolicy
                | CrucibleIssue::InvalidScenario(_)
                | CrucibleIssue::MissingRequiredFamily(_)
                | CrucibleIssue::InsufficientFrames { .. }
                | CrucibleIssue::InsufficientNegativeFrames { .. }
                | CrucibleIssue::NoOodTrials
        )
    });
    let failed = issues.iter().any(|issue| {
        matches!(
            issue,
            CrucibleIssue::FalsePositiveRateExceeded { .. }
                | CrucibleIssue::CleanNegativeFractionBelowMinimum { .. }
                | CrucibleIssue::FalseTrackFractionExceeded { .. }
                | CrucibleIssue::IdentitySwitchRateExceeded { .. }
                | CrucibleIssue::OodAbstentionBelowMinimum { .. }
                | CrucibleIssue::IncompleteClassificationRateExceeded { .. }
                | CrucibleIssue::PerceptionAuthorityBoundaryViolation { .. }
        )
    });

    PerceptionCrucibleReport {
        policy_id: policy.policy_id.clone(),
        status: if incomplete {
            CrucibleStatus::Incomplete
        } else if failed {
            CrucibleStatus::Fail
        } else {
            CrucibleStatus::Pass
        },
        assessed_scenarios: scenarios.len(),
        total_frames,
        total_negative_frames,
        total_false_positives,
        total_false_tracks,
        total_identity_switches,
        classification_trials,
        ood_trials,
        ood_abstentions,
        incomplete_classifications,
        issues,
    }
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_selective_classification::{ClassificationAssessment, ClassificationIssue};
    use symthaea_tracking_eval::TrackingEvaluationReport;

    fn policy() -> PerceptionCruciblePolicy {
        PerceptionCruciblePolicy {
            policy_id: "release-v1".into(),
            required_families: vec![StressFamily::QuietBackground, StressFamily::OutOfDistribution],
            minimum_frames_per_scenario: 100,
            minimum_negative_frames_per_background_scenario: 100,
            maximum_false_positives_per_negative_frame: 0.01,
            minimum_clean_negative_frame_fraction: 0.99,
            maximum_false_track_fraction: 0.05,
            maximum_identity_switches_per_true_positive: 0.01,
            minimum_ood_abstention_rate: 0.95,
            maximum_incomplete_classification_rate: 0.10,
        }
    }

    fn negative_tracking(frames: usize, false_positives: usize, false_tracks: usize) -> TrackingEvaluationReport {
        TrackingEvaluationReport {
            evaluated_frames: frames,
            ground_truth_points: 0,
            prediction_points: false_positives,
            true_positives: 0,
            false_positives,
            false_negatives: 0,
            identity_switches: 0,
            false_track_count: false_tracks,
            unique_prediction_tracks: false_tracks,
            frames_with_false_positives: false_positives.min(frames),
            negative_frames: frames,
            negative_frames_with_false_positives: false_positives.min(frames),
            precision: (false_positives > 0).then_some(0.0),
            recall: None,
            f1: None,
            mean_match_distance_norm: None,
            false_positives_per_frame: false_positives as f64 / frames as f64,
            clean_negative_frame_fraction: Some(
                (frames - false_positives.min(frames)) as f64 / frames as f64,
            ),
            false_track_fraction: (false_tracks > 0).then_some(1.0),
            negative_only_sequence: true,
            frames: Vec::new(),
        }
    }

    fn ood_trial(id: &str, abstain: bool) -> ClassificationTrial {
        ClassificationTrial {
            trial_id: id.into(),
            expected_ood: true,
            assessment: ClassificationAssessment {
                disposition: if abstain {
                    ClassificationDisposition::Abstain
                } else {
                    ClassificationDisposition::EvidenceUsable
                },
                issues: if abstain {
                    vec![ClassificationIssue::OutOfDistribution]
                } else {
                    Vec::new()
                },
            },
        }
    }

    fn quiet_scenario() -> PerceptionScenarioEvidence {
        PerceptionScenarioEvidence {
            scenario_id: "quiet-1000".into(),
            family: StressFamily::QuietBackground,
            expected_background_only: true,
            tracking: negative_tracking(1_000, 0, 0),
            classification_trials: Vec::new(),
            perception_to_authority_boundary_violations: 0,
            evidence_refs: vec!["dataset:quiet-1000".into()],
        }
    }

    fn ood_scenario() -> PerceptionScenarioEvidence {
        PerceptionScenarioEvidence {
            scenario_id: "ood-100".into(),
            family: StressFamily::OutOfDistribution,
            expected_background_only: false,
            tracking: TrackingEvaluationReport {
                evaluated_frames: 100,
                ground_truth_points: 100,
                prediction_points: 100,
                true_positives: 100,
                false_positives: 0,
                false_negatives: 0,
                identity_switches: 0,
                false_track_count: 0,
                unique_prediction_tracks: 1,
                frames_with_false_positives: 0,
                negative_frames: 0,
                negative_frames_with_false_positives: 0,
                precision: Some(1.0),
                recall: Some(1.0),
                f1: Some(1.0),
                mean_match_distance_norm: Some(0.0),
                false_positives_per_frame: 0.0,
                clean_negative_frame_fraction: None,
                false_track_fraction: Some(0.0),
                negative_only_sequence: false,
                frames: Vec::new(),
            },
            classification_trials: (0..100)
                .map(|i| ood_trial(&format!("ood-{i}"), true))
                .collect(),
            perception_to_authority_boundary_violations: 0,
            evidence_refs: vec!["dataset:ood-100".into()],
        }
    }

    #[test]
    fn complete_clean_suite_passes() {
        let report = assess_perception_crucible(&[quiet_scenario(), ood_scenario()], &policy());
        assert_eq!(report.status, CrucibleStatus::Pass);
        assert!(report.issues.is_empty());
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn perception_authority_violation_is_hard_failure() {
        let mut quiet = quiet_scenario();
        quiet.perception_to_authority_boundary_violations = 1;
        let report = assess_perception_crucible(&[quiet, ood_scenario()], &policy());
        assert_eq!(report.status, CrucibleStatus::Fail);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            CrucibleIssue::PerceptionAuthorityBoundaryViolation { count: 1, .. }
        )));
    }

    #[test]
    fn missing_required_stress_family_is_incomplete() {
        let report = assess_perception_crucible(&[quiet_scenario()], &policy());
        assert_eq!(report.status, CrucibleStatus::Incomplete);
        assert!(report
            .issues
            .contains(&CrucibleIssue::MissingRequiredFamily(StressFamily::OutOfDistribution)));
    }

    #[test]
    fn noisy_negative_background_fails_release() {
        let mut quiet = quiet_scenario();
        quiet.tracking = negative_tracking(1_000, 50, 50);
        let report = assess_perception_crucible(&[quiet, ood_scenario()], &policy());
        assert_eq!(report.status, CrucibleStatus::Fail);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            CrucibleIssue::FalsePositiveRateExceeded { .. }
        )));
    }

    #[test]
    fn ood_that_does_not_abstain_fails_release() {
        let mut ood = ood_scenario();
        for trial in ood.classification_trials.iter_mut().take(20) {
            *trial = ood_trial(&trial.trial_id, false);
        }
        let report = assess_perception_crucible(&[quiet_scenario(), ood], &policy());
        assert_eq!(report.status, CrucibleStatus::Fail);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            CrucibleIssue::OodAbstentionBelowMinimum { .. }
        )));
    }
}