// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen four-arm protocol for the first symbolic cognition experiment.
//!
//! This module defines records, comparability checks, descriptive summaries,
//! and the pre-registered success gate. It deliberately does not manufacture
//! statistical significance or collapse structural, perceptual, and workflow
//! evidence into one score.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Policy arms frozen before the first experiment is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CognitivePolicyArm {
    Fixed,
    RandomValid,
    Heuristic,
    Symthaea,
}

impl CognitivePolicyArm {
    pub const ALL: [Self; 4] = [
        Self::Fixed,
        Self::RandomValid,
        Self::Heuristic,
        Self::Symthaea,
    ];
}

/// Identity shared by all four arms for one frozen musical input.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrozenTrialKey {
    pub fixture_id: String,
    pub seed: u64,
}

/// Structural measurements that must remain separate from preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralTrialOutcome {
    pub hard_constraints_valid: bool,
    pub obligations_total: usize,
    pub obligations_fulfilled: usize,
    pub voice_leading_violations: usize,
    pub motif_return_similarity: Option<f32>,
    pub tonic_returned: bool,
}

impl StructuralTrialOutcome {
    pub fn obligation_fulfilment_rate(&self) -> f32 {
        if self.obligations_total == 0 {
            0.0
        } else {
            self.obligations_fulfilled as f32 / self.obligations_total as f32
        }
    }
}

/// Blinded listener evidence. Every field may remain absent until collected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualTrialOutcome {
    pub listener_count: usize,
    /// Fraction recognizing the intended return, in [0, 1].
    pub return_recognition_rate: Option<f32>,
    /// Mean rating that the development was less stable, in [0, 1].
    pub development_instability: Option<f32>,
    /// Mean rating that the recapitulation felt earned, in [0, 1].
    pub earned_recapitulation: Option<f32>,
    /// Fraction preferring this arm in its blinded comparison, in [0, 1].
    pub preference_rate: Option<f32>,
}

/// Artist-workflow evidence kept independent from listener preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowTrialOutcome {
    pub kept: bool,
    pub edited: bool,
    pub rejected: bool,
    pub time_to_commit_seconds: Option<u64>,
}

/// One arm's result for one frozen input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveTrialRecord {
    pub key: FrozenTrialKey,
    pub arm: CognitivePolicyArm,
    /// Digest of subject pair, key, meter, orchestration, renderer, soundfont,
    /// seed set, and hard constraints. All four arms must match exactly.
    pub frozen_input_sha256: String,
    pub policy_version: String,
    pub structural: StructuralTrialOutcome,
    pub perceptual: Option<PerceptualTrialOutcome>,
    pub workflow: Option<WorkflowTrialOutcome>,
}

/// Invalid or incomparable experiment data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentIssue {
    NoRecords,
    InvalidFrozenDigest {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    DuplicateArm {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    MissingArm {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    FrozenInputMismatch {
        key: FrozenTrialKey,
    },
    InvalidRate {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
        field: String,
    },
    NonFiniteMeasurement {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
        field: String,
    },
    EmptyPolicyVersion {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    NoStructuralObligations {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    MissingMotifReturnSimilarity {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    ObligationCountMismatch {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
        total: usize,
        fulfilled: usize,
    },
    PerceptualDataWithoutListeners {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    IncompletePerceptualOutcome {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    PerceptualAvailabilityMismatch {
        key: FrozenTrialKey,
    },
    PerceptualCompletenessMismatch {
        key: FrozenTrialKey,
    },
    ListenerCountMismatch {
        key: FrozenTrialKey,
    },
    MissingWorkflowDisposition {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    ContradictoryWorkflowDisposition {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    WorkflowAvailabilityMismatch {
        key: FrozenTrialKey,
    },
}

/// Validate frozen pairing and bounded measurements before analysis.
pub fn validate_experiment(records: &[CognitiveTrialRecord]) -> Vec<ExperimentIssue> {
    let mut issues = Vec::new();
    if records.is_empty() {
        issues.push(ExperimentIssue::NoRecords);
        return issues;
    }

    let mut groups: BTreeMap<FrozenTrialKey, Vec<&CognitiveTrialRecord>> = BTreeMap::new();
    for record in records {
        groups.entry(record.key.clone()).or_default().push(record);
        if !is_sha256(&record.frozen_input_sha256) {
            issues.push(ExperimentIssue::InvalidFrozenDigest {
                key: record.key.clone(),
                arm: record.arm,
            });
        }
        if record.policy_version.trim().is_empty() {
            issues.push(ExperimentIssue::EmptyPolicyVersion {
                key: record.key.clone(),
                arm: record.arm,
            });
        }
        if record.structural.obligations_total == 0 {
            issues.push(ExperimentIssue::NoStructuralObligations {
                key: record.key.clone(),
                arm: record.arm,
            });
        }
        if record.structural.motif_return_similarity.is_none() {
            issues.push(ExperimentIssue::MissingMotifReturnSimilarity {
                key: record.key.clone(),
                arm: record.arm,
            });
        }
        if record.structural.obligations_fulfilled > record.structural.obligations_total {
            issues.push(ExperimentIssue::ObligationCountMismatch {
                key: record.key.clone(),
                arm: record.arm,
                total: record.structural.obligations_total,
                fulfilled: record.structural.obligations_fulfilled,
            });
        }
        for (field, value) in rates(record) {
            if !value.is_finite() {
                issues.push(ExperimentIssue::NonFiniteMeasurement {
                    key: record.key.clone(),
                    arm: record.arm,
                    field: field.into(),
                });
            } else if !(0.0..=1.0).contains(&value) {
                issues.push(ExperimentIssue::InvalidRate {
                    key: record.key.clone(),
                    arm: record.arm,
                    field: field.into(),
                });
            }
        }
        if let Some(perceptual) = &record.perceptual {
            let present = perceptual_field_count(perceptual);
            if present > 0 && perceptual.listener_count == 0 {
                issues.push(ExperimentIssue::PerceptualDataWithoutListeners {
                    key: record.key.clone(),
                    arm: record.arm,
                });
            }
            if present != 0 && present != 4 {
                issues.push(ExperimentIssue::IncompletePerceptualOutcome {
                    key: record.key.clone(),
                    arm: record.arm,
                });
            }
        }
        if let Some(workflow) = &record.workflow {
            if !workflow.kept && !workflow.edited && !workflow.rejected {
                issues.push(ExperimentIssue::MissingWorkflowDisposition {
                    key: record.key.clone(),
                    arm: record.arm,
                });
            }
            if workflow.rejected && (workflow.kept || workflow.edited) {
                issues.push(ExperimentIssue::ContradictoryWorkflowDisposition {
                    key: record.key.clone(),
                    arm: record.arm,
                });
            }
        }
    }

    for (key, group) in groups {
        let mut seen = BTreeSet::new();
        for record in &group {
            if !seen.insert(record.arm) {
                issues.push(ExperimentIssue::DuplicateArm {
                    key: key.clone(),
                    arm: record.arm,
                });
            }
        }
        for arm in CognitivePolicyArm::ALL {
            if !seen.contains(&arm) {
                issues.push(ExperimentIssue::MissingArm {
                    key: key.clone(),
                    arm,
                });
            }
        }
        let digests: BTreeSet<&str> = group
            .iter()
            .map(|record| record.frozen_input_sha256.as_str())
            .collect();
        if digests.len() > 1 {
            issues.push(ExperimentIssue::FrozenInputMismatch { key: key.clone() });
        }

        let perceptual_presence: BTreeSet<bool> = group
            .iter()
            .map(|record| record.perceptual.is_some())
            .collect();
        if perceptual_presence.len() > 1 {
            issues.push(ExperimentIssue::PerceptualAvailabilityMismatch { key: key.clone() });
        }
        let perceptual_completeness: BTreeSet<usize> = group
            .iter()
            .filter_map(|record| record.perceptual.as_ref().map(perceptual_field_count))
            .collect();
        if perceptual_completeness.len() > 1 {
            issues.push(ExperimentIssue::PerceptualCompletenessMismatch { key: key.clone() });
        }
        let listener_counts: BTreeSet<usize> = group
            .iter()
            .filter_map(|record| record.perceptual.as_ref().map(|value| value.listener_count))
            .collect();
        if listener_counts.len() > 1 {
            issues.push(ExperimentIssue::ListenerCountMismatch { key: key.clone() });
        }

        let workflow_presence: BTreeSet<bool> = group
            .iter()
            .map(|record| record.workflow.is_some())
            .collect();
        if workflow_presence.len() > 1 {
            issues.push(ExperimentIssue::WorkflowAvailabilityMismatch { key });
        }
    }

    issues
}

fn rates(record: &CognitiveTrialRecord) -> Vec<(&'static str, f32)> {
    let mut values = Vec::new();
    if let Some(value) = record.structural.motif_return_similarity {
        values.push(("motif_return_similarity", value));
    }
    if let Some(perceptual) = &record.perceptual {
        for (field, value) in [
            (
                "return_recognition_rate",
                perceptual.return_recognition_rate,
            ),
            (
                "development_instability",
                perceptual.development_instability,
            ),
            ("earned_recapitulation", perceptual.earned_recapitulation),
            ("preference_rate", perceptual.preference_rate),
        ] {
            if let Some(value) = value {
                values.push((field, value));
            }
        }
    }
    values
}

fn perceptual_field_count(outcome: &PerceptualTrialOutcome) -> usize {
    [
        outcome.return_recognition_rate,
        outcome.development_instability,
        outcome.earned_recapitulation,
        outcome.preference_rate,
    ]
    .into_iter()
    .filter(Option::is_some)
    .count()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Descriptive arm summary. No field implies statistical significance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveArmSummary {
    pub arm: CognitivePolicyArm,
    pub trials: usize,
    #[serde(default)]
    pub perceptual_trials: usize,
    #[serde(default)]
    pub workflow_trials: usize,
    #[serde(default)]
    pub time_to_commit_trials: usize,
    pub structural_validity_rate: f32,
    pub mean_obligation_fulfilment: f32,
    pub mean_voice_leading_violations: f32,
    #[serde(default)]
    pub tonic_return_rate: f32,
    pub mean_motif_return_similarity: Option<f32>,
    pub mean_return_recognition: Option<f32>,
    pub mean_development_instability: Option<f32>,
    pub mean_earned_recapitulation: Option<f32>,
    pub mean_preference_rate: Option<f32>,
    pub keep_rate: Option<f32>,
    pub edit_rate: Option<f32>,
    pub rejection_rate: Option<f32>,
    pub mean_time_to_commit_seconds: Option<f32>,
}

/// Summarize each policy arm after validation.
pub fn summarize_experiment(records: &[CognitiveTrialRecord]) -> Vec<CognitiveArmSummary> {
    CognitivePolicyArm::ALL
        .into_iter()
        .map(|arm| summarize_arm(arm, records.iter().filter(|record| record.arm == arm)))
        .collect()
}

fn summarize_arm<'a>(
    arm: CognitivePolicyArm,
    records: impl Iterator<Item = &'a CognitiveTrialRecord>,
) -> CognitiveArmSummary {
    let records: Vec<&CognitiveTrialRecord> = records.collect();
    let trials = records.len();
    let denominator = trials.max(1) as f32;
    let workflow: Vec<&WorkflowTrialOutcome> = records
        .iter()
        .filter_map(|record| record.workflow.as_ref())
        .collect();

    CognitiveArmSummary {
        arm,
        trials,
        perceptual_trials: records
            .iter()
            .filter(|record| {
                record.perceptual.as_ref().is_some_and(|value| {
                    value.listener_count > 0 && perceptual_field_count(value) == 4
                })
            })
            .count(),
        workflow_trials: workflow
            .iter()
            .filter(|value| {
                (value.kept || value.edited || value.rejected)
                    && !(value.rejected && (value.kept || value.edited))
            })
            .count(),
        time_to_commit_trials: workflow
            .iter()
            .filter(|value| {
                value.time_to_commit_seconds.is_some()
                    && (value.kept || value.edited || value.rejected)
                    && !(value.rejected && (value.kept || value.edited))
            })
            .count(),
        structural_validity_rate: records
            .iter()
            .filter(|record| record.structural.hard_constraints_valid)
            .count() as f32
            / denominator,
        mean_obligation_fulfilment: records
            .iter()
            .map(|record| record.structural.obligation_fulfilment_rate())
            .sum::<f32>()
            / denominator,
        mean_voice_leading_violations: records
            .iter()
            .map(|record| record.structural.voice_leading_violations as f32)
            .sum::<f32>()
            / denominator,
        tonic_return_rate: records
            .iter()
            .filter(|record| record.structural.tonic_returned)
            .count() as f32
            / denominator,
        mean_motif_return_similarity: mean_option(
            records
                .iter()
                .filter_map(|record| record.structural.motif_return_similarity),
        ),
        mean_return_recognition: mean_option(records.iter().filter_map(|record| {
            record
                .perceptual
                .as_ref()
                .and_then(|value| value.return_recognition_rate)
        })),
        mean_development_instability: mean_option(records.iter().filter_map(|record| {
            record
                .perceptual
                .as_ref()
                .and_then(|value| value.development_instability)
        })),
        mean_earned_recapitulation: mean_option(records.iter().filter_map(|record| {
            record
                .perceptual
                .as_ref()
                .and_then(|value| value.earned_recapitulation)
        })),
        mean_preference_rate: mean_option(records.iter().filter_map(|record| {
            record
                .perceptual
                .as_ref()
                .and_then(|value| value.preference_rate)
        })),
        keep_rate: fraction(&workflow, |value| value.kept),
        edit_rate: fraction(&workflow, |value| value.edited),
        rejection_rate: fraction(&workflow, |value| value.rejected),
        mean_time_to_commit_seconds: mean_option(
            workflow
                .iter()
                .filter_map(|value| value.time_to_commit_seconds.map(|seconds| seconds as f32)),
        ),
    }
}

fn mean_option(values: impl Iterator<Item = f32>) -> Option<f32> {
    let values: Vec<f32> = values.collect();
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

fn fraction(
    values: &[&WorkflowTrialOutcome],
    predicate: impl Fn(&WorkflowTrialOutcome) -> bool,
) -> Option<f32> {
    (!values.is_empty()).then(|| {
        values.iter().filter(|value| predicate(value)).count() as f32 / values.len() as f32
    })
}

/// Pre-registered channels on which the Symthaea arm may demonstrate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImprovementChannel {
    ReturnRecognition,
    EarnedRecapitulation,
    Preference,
    KeepRate,
    LowerTimeToCommit,
}

/// Minimum paired inputs required before the descriptive gate may pass.
pub const MIN_TRIALS_PER_ARM: usize = 8;
/// Minimum mean structural identity similarity for the Symthaea arm.
pub const MIN_MOTIF_RETURN_SIMILARITY: f32 = 0.95;
/// Minimum absolute improvement on bounded perceptual/workflow rates.
pub const MIN_RATE_EFFECT: f32 = 0.05;
/// Symthaea may be slightly below the heuristic, but not meaningfully worse.
pub const HEURISTIC_NONINFERIORITY_MARGIN: f32 = 0.02;
/// Minimum reduction in artist decision time relative to simple baselines.
pub const MIN_TIME_EFFECT_SECONDS: f32 = 10.0;
/// Maximum descriptive time disadvantage permitted relative to the heuristic.
pub const HEURISTIC_TIME_MARGIN_SECONDS: f32 = 5.0;

/// Result of the deliberately narrow first descriptive success gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentConclusion {
    pub success: bool,
    pub structural_gate_passed: bool,
    #[serde(default)]
    pub sample_gate_passed: bool,
    #[serde(default)]
    pub heuristic_gate_passed: bool,
    pub improvements: Vec<ImprovementChannel>,
    pub rationale: Vec<String>,
}

/// Apply the frozen descriptive success rule without claiming significance.
///
/// Symthaea must preserve every hard structural constraint over at least
/// `MIN_TRIALS_PER_ARM` paired inputs. A qualifying improvement must exceed
/// both simple baselines by a pre-registered practical margin and remain
/// non-inferior to the hand-authored heuristic. Each channel also requires
/// complete paired observations for all four arms.
pub fn conclude_first_experiment(summaries: &[CognitiveArmSummary]) -> ExperimentConclusion {
    let get = |arm| summaries.iter().find(|summary| summary.arm == arm);
    let (Some(symthaea), Some(fixed), Some(random), Some(heuristic)) = (
        get(CognitivePolicyArm::Symthaea),
        get(CognitivePolicyArm::Fixed),
        get(CognitivePolicyArm::RandomValid),
        get(CognitivePolicyArm::Heuristic),
    ) else {
        return ExperimentConclusion {
            success: false,
            structural_gate_passed: false,
            sample_gate_passed: false,
            heuristic_gate_passed: false,
            improvements: Vec::new(),
            rationale: vec!["all four required arm summaries are not present".into()],
        };
    };

    let trial_counts = [
        symthaea.trials,
        fixed.trials,
        random.trials,
        heuristic.trials,
    ];
    let sample_gate_passed = trial_counts
        .iter()
        .all(|trials| *trials >= MIN_TRIALS_PER_ARM)
        && trial_counts.iter().all(|trials| *trials == trial_counts[0]);
    let structural_gate_passed = sample_gate_passed
        && symthaea.structural_validity_rate == 1.0
        && symthaea.mean_obligation_fulfilment == 1.0
        && symthaea.mean_voice_leading_violations == 0.0
        && symthaea.tonic_return_rate == 1.0
        && symthaea
            .mean_motif_return_similarity
            .is_some_and(|value| value >= MIN_MOTIF_RETURN_SIMILARITY);

    let perceptual_coverage = [symthaea, fixed, random, heuristic]
        .iter()
        .all(|summary| summary.perceptual_trials == summary.trials);
    let workflow_coverage = [symthaea, fixed, random, heuristic]
        .iter()
        .all(|summary| summary.workflow_trials == summary.trials);
    let time_coverage = [symthaea, fixed, random, heuristic]
        .iter()
        .all(|summary| summary.time_to_commit_trials == summary.trials);

    let mut improvements = Vec::new();
    if perceptual_coverage
        && rate_improves_over_baselines_and_meets_heuristic(
            symthaea.mean_return_recognition,
            fixed.mean_return_recognition,
            random.mean_return_recognition,
            heuristic.mean_return_recognition,
        )
    {
        improvements.push(ImprovementChannel::ReturnRecognition);
    }
    if perceptual_coverage
        && rate_improves_over_baselines_and_meets_heuristic(
            symthaea.mean_earned_recapitulation,
            fixed.mean_earned_recapitulation,
            random.mean_earned_recapitulation,
            heuristic.mean_earned_recapitulation,
        )
    {
        improvements.push(ImprovementChannel::EarnedRecapitulation);
    }
    if perceptual_coverage
        && rate_improves_over_baselines_and_meets_heuristic(
            symthaea.mean_preference_rate,
            fixed.mean_preference_rate,
            random.mean_preference_rate,
            heuristic.mean_preference_rate,
        )
    {
        improvements.push(ImprovementChannel::Preference);
    }
    if workflow_coverage
        && rate_improves_over_baselines_and_meets_heuristic(
            symthaea.keep_rate,
            fixed.keep_rate,
            random.keep_rate,
            heuristic.keep_rate,
        )
    {
        improvements.push(ImprovementChannel::KeepRate);
    }
    if time_coverage
        && time_improves_over_baselines_and_meets_heuristic(
            symthaea.mean_time_to_commit_seconds,
            fixed.mean_time_to_commit_seconds,
            random.mean_time_to_commit_seconds,
            heuristic.mean_time_to_commit_seconds,
        )
    {
        improvements.push(ImprovementChannel::LowerTimeToCommit);
    }

    let heuristic_gate_passed = !improvements.is_empty();
    let success = structural_gate_passed && heuristic_gate_passed;
    let mut rationale = vec![format!(
        "sample gate {} (minimum {MIN_TRIALS_PER_ARM} paired trials per arm)",
        if sample_gate_passed {
            "passed"
        } else {
            "failed"
        }
    )];
    rationale.push(format!(
        "structural gate {}",
        if structural_gate_passed {
            "passed"
        } else {
            "failed"
        }
    ));
    if improvements.is_empty() {
        rationale.push(
            "no fully observed channel cleared both practical baseline margins and the heuristic non-inferiority gate"
                .into(),
        );
    } else {
        rationale.push(format!(
            "descriptive improvements clearing the heuristic gate observed on {:?}",
            improvements
        ));
    }
    rationale.push(
        "this gate does not replace inferential statistics, blinded-listener analysis, or multiplicity control"
            .into(),
    );

    ExperimentConclusion {
        success,
        structural_gate_passed,
        sample_gate_passed,
        heuristic_gate_passed,
        improvements,
        rationale,
    }
}

fn rate_improves_over_baselines_and_meets_heuristic(
    value: Option<f32>,
    fixed: Option<f32>,
    random: Option<f32>,
    heuristic: Option<f32>,
) -> bool {
    matches!(
        (value, fixed, random, heuristic),
        (Some(value), Some(fixed), Some(random), Some(heuristic))
            if value - fixed >= MIN_RATE_EFFECT
                && value - random >= MIN_RATE_EFFECT
                && value + HEURISTIC_NONINFERIORITY_MARGIN >= heuristic
    )
}

fn time_improves_over_baselines_and_meets_heuristic(
    value: Option<f32>,
    fixed: Option<f32>,
    random: Option<f32>,
    heuristic: Option<f32>,
) -> bool {
    matches!(
        (value, fixed, random, heuristic),
        (Some(value), Some(fixed), Some(random), Some(heuristic))
            if fixed - value >= MIN_TIME_EFFECT_SECONDS
                && random - value >= MIN_TIME_EFFECT_SECONDS
                && value <= heuristic + HEURISTIC_TIME_MARGIN_SECONDS
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn record(
        seed: u64,
        arm: CognitivePolicyArm,
        kept: bool,
        recognition: f32,
        time_to_commit_seconds: u64,
    ) -> CognitiveTrialRecord {
        CognitiveTrialRecord {
            key: FrozenTrialKey {
                fixture_id: format!("sonata-{seed}"),
                seed,
            },
            arm,
            frozen_input_sha256: DIGEST.into(),
            policy_version: "v1".into(),
            structural: StructuralTrialOutcome {
                hard_constraints_valid: true,
                obligations_total: 6,
                obligations_fulfilled: 6,
                voice_leading_violations: 0,
                motif_return_similarity: Some(0.98),
                tonic_returned: true,
            },
            perceptual: Some(PerceptualTrialOutcome {
                listener_count: 8,
                return_recognition_rate: Some(recognition),
                development_instability: Some(0.7),
                earned_recapitulation: Some(recognition),
                preference_rate: Some(recognition),
            }),
            workflow: Some(WorkflowTrialOutcome {
                kept,
                edited: false,
                rejected: !kept,
                time_to_commit_seconds: Some(time_to_commit_seconds),
            }),
        }
    }

    fn complete_trials(count: usize) -> Vec<CognitiveTrialRecord> {
        let mut records = Vec::new();
        for seed in 0..count as u64 {
            records.extend([
                record(seed, CognitivePolicyArm::Fixed, false, 0.40, 120),
                record(seed, CognitivePolicyArm::RandomValid, false, 0.30, 130),
                record(seed, CognitivePolicyArm::Heuristic, true, 0.75, 65),
                record(seed, CognitivePolicyArm::Symthaea, true, 0.80, 60),
            ]);
        }
        records
    }

    #[test]
    fn validation_requires_all_four_arms() {
        let issues = validate_experiment(&[record(42, CognitivePolicyArm::Fixed, false, 0.4, 120)]);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ExperimentIssue::MissingArm {
                arm: CognitivePolicyArm::Symthaea,
                ..
            }
        )));
    }

    #[test]
    fn frozen_complete_records_validate() {
        let records = complete_trials(1);
        assert!(validate_experiment(&records).is_empty());
    }

    #[test]
    fn validation_rejects_empty_obligations_and_bad_collection_states() {
        let mut records = complete_trials(1);
        records[0].structural.obligations_total = 0;
        records[0].structural.obligations_fulfilled = 1;
        records[1].perceptual.as_mut().unwrap().listener_count = 0;
        records[1].perceptual.as_mut().unwrap().preference_rate = None;
        records[2].workflow.as_mut().unwrap().rejected = true;

        let issues = validate_experiment(&records);
        assert!(
            issues
                .iter()
                .any(|issue| matches!(issue, ExperimentIssue::NoStructuralObligations { .. }))
        );
        assert!(
            issues
                .iter()
                .any(|issue| matches!(issue, ExperimentIssue::ObligationCountMismatch { .. }))
        );
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ExperimentIssue::PerceptualDataWithoutListeners { .. }
        )));
        assert!(
            issues
                .iter()
                .any(|issue| matches!(issue, ExperimentIssue::IncompletePerceptualOutcome { .. }))
        );
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ExperimentIssue::ContradictoryWorkflowDisposition { .. }
        )));
    }

    #[test]
    fn one_trial_cannot_pass_the_gate() {
        let records = complete_trials(1);
        let conclusion = conclude_first_experiment(&summarize_experiment(&records));
        assert!(!conclusion.sample_gate_passed);
        assert!(!conclusion.success);
    }

    #[test]
    fn symthaea_can_pass_the_practical_and_heuristic_gates() {
        let records = complete_trials(MIN_TRIALS_PER_ARM);
        let conclusion = conclude_first_experiment(&summarize_experiment(&records));
        assert!(conclusion.sample_gate_passed);
        assert!(conclusion.structural_gate_passed);
        assert!(conclusion.heuristic_gate_passed);
        assert!(conclusion.success);
        assert!(
            conclusion
                .improvements
                .contains(&ImprovementChannel::ReturnRecognition)
        );
    }

    #[test]
    fn beating_simple_baselines_but_losing_to_the_heuristic_is_not_success() {
        let mut records = complete_trials(MIN_TRIALS_PER_ARM);
        for record in &mut records {
            let perceptual = record.perceptual.as_mut().unwrap();
            let workflow = record.workflow.as_mut().unwrap();
            workflow.kept = true;
            workflow.rejected = false;
            workflow.time_to_commit_seconds = Some(60);
            match record.arm {
                CognitivePolicyArm::Fixed => set_recognition(perceptual, 0.40),
                CognitivePolicyArm::RandomValid => set_recognition(perceptual, 0.30),
                CognitivePolicyArm::Heuristic => set_recognition(perceptual, 0.90),
                CognitivePolicyArm::Symthaea => set_recognition(perceptual, 0.70),
            }
        }

        let conclusion = conclude_first_experiment(&summarize_experiment(&records));
        assert!(conclusion.structural_gate_passed);
        assert!(!conclusion.heuristic_gate_passed);
        assert!(!conclusion.success);
    }

    fn set_recognition(outcome: &mut PerceptualTrialOutcome, value: f32) {
        outcome.return_recognition_rate = Some(value);
        outcome.earned_recapitulation = Some(value);
        outcome.preference_rate = Some(value);
    }
}
