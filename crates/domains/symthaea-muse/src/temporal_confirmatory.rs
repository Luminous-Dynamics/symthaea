// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Confirmatory inference for the paired HDC/CfC temporal ablation.
//!
//! Pilot and confirmatory families are separated before analysis. The result
//! remains a mechanism claim only: temporal state influenced the FEP trajectory.

use crate::cognitive_session::CognitiveSensoryVector;
use crate::cognitive_session_experiment::{
    MIN_TEMPORAL_SENSORY_DELTA, TemporalAblationIssue, TemporalAblationRecord,
    validate_temporal_ablation,
};
use crate::experiment_manifest::StudySplit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const TEMPORAL_CONFIRMATORY_VERSION: &str = "symthaea-muse-temporal-confirmatory-v1";
pub const MIN_TEMPORAL_CONFIRMATORY_PAIRS: usize = 24;
pub const MIN_ACTION_DIVERGENCE_RATE: f64 = 0.10;
pub const MIN_TEMPORAL_BOOTSTRAP_REPLICATES: usize = 2_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenTemporalRecord {
    pub family_id: String,
    pub split: StudySplit,
    pub record: TemporalAblationRecord,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalConfirmatoryPlan {
    pub analysis_version: String,
    pub alpha: f64,
    pub minimum_pairs: usize,
    pub bootstrap_replicates: usize,
    pub rng_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalConfirmatorySummary {
    pub confirmatory_pairs: usize,
    pub mean_sensory_delta: f64,
    pub sensory_delta_ci_95: [f64; 2],
    pub mean_action_divergence_rate: f64,
    pub action_divergence_ci_95: [f64; 2],
    pub terminal_action_divergence_rate: f64,
    pub terminal_divergence_ci_95: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalConfirmatoryConclusion {
    pub success: bool,
    pub evidence_gate_passed: bool,
    pub sample_gate_passed: bool,
    pub sensory_gate_passed: bool,
    pub action_gate_passed: bool,
    pub issues: Vec<TemporalConfirmatoryIssue>,
    pub summary: Option<TemporalConfirmatorySummary>,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalConfirmatoryIssue {
    WrongAnalysisVersion { found: String },
    InvalidAlpha,
    TooFewRequiredPairs { found: usize, required: usize },
    TooFewBootstrapReplicates { found: usize, required: usize },
    EmptyFamilyId { trial_id: String },
    FamilyCrossesSplits { family_id: String },
    InnerAblation { issue: TemporalAblationIssue },
}

pub fn validate_temporal_confirmatory(
    records: &[FrozenTemporalRecord],
    plan: &TemporalConfirmatoryPlan,
) -> Vec<TemporalConfirmatoryIssue> {
    let mut issues = Vec::new();
    if plan.analysis_version != TEMPORAL_CONFIRMATORY_VERSION {
        issues.push(TemporalConfirmatoryIssue::WrongAnalysisVersion {
            found: plan.analysis_version.clone(),
        });
    }
    if !plan.alpha.is_finite() || !(0.0..=0.10).contains(&plan.alpha) || plan.alpha == 0.0 {
        issues.push(TemporalConfirmatoryIssue::InvalidAlpha);
    }
    if plan.minimum_pairs < MIN_TEMPORAL_CONFIRMATORY_PAIRS {
        issues.push(TemporalConfirmatoryIssue::TooFewRequiredPairs {
            found: plan.minimum_pairs,
            required: MIN_TEMPORAL_CONFIRMATORY_PAIRS,
        });
    }
    if plan.bootstrap_replicates < MIN_TEMPORAL_BOOTSTRAP_REPLICATES {
        issues.push(TemporalConfirmatoryIssue::TooFewBootstrapReplicates {
            found: plan.bootstrap_replicates,
            required: MIN_TEMPORAL_BOOTSTRAP_REPLICATES,
        });
    }
    let mut family_splits: BTreeMap<&str, StudySplit> = BTreeMap::new();
    for record in records {
        if record.family_id.trim().is_empty() {
            issues.push(TemporalConfirmatoryIssue::EmptyFamilyId {
                trial_id: record.record.trial_id.clone(),
            });
        }
        if let Some(previous) = family_splits.insert(&record.family_id, record.split) {
            if previous != record.split {
                issues.push(TemporalConfirmatoryIssue::FamilyCrossesSplits {
                    family_id: record.family_id.clone(),
                });
            }
        }
    }
    let inner: Vec<_> = records.iter().map(|record| record.record.clone()).collect();
    for issue in validate_temporal_ablation(&inner) {
        issues.push(TemporalConfirmatoryIssue::InnerAblation { issue });
    }
    issues
}

pub fn analyze_temporal_confirmatory(
    records: &[FrozenTemporalRecord],
    plan: &TemporalConfirmatoryPlan,
) -> TemporalConfirmatoryConclusion {
    let issues = validate_temporal_confirmatory(records, plan);
    let confirmatory: Vec<_> = records
        .iter()
        .filter(|record| record.split == StudySplit::Confirmatory)
        .collect();
    let sample_gate_passed = confirmatory.len() >= plan.minimum_pairs;
    let evidence_gate_passed = issues.is_empty();
    let summary = summarize(&confirmatory, plan);
    let sensory_gate_passed = summary.as_ref().is_some_and(|summary| {
        summary.mean_sensory_delta >= f64::from(MIN_TEMPORAL_SENSORY_DELTA)
            && summary.sensory_delta_ci_95[0] > 0.0
    });
    let action_gate_passed = summary.as_ref().is_some_and(|summary| {
        summary.mean_action_divergence_rate >= MIN_ACTION_DIVERGENCE_RATE
            && summary.action_divergence_ci_95[0] > 0.0
    });
    let success =
        sample_gate_passed && evidence_gate_passed && sensory_gate_passed && action_gate_passed;
    let mut rationale = vec![format!(
        "sample gate {} ({} confirmatory pairs required)",
        pass_fail(sample_gate_passed),
        plan.minimum_pairs
    )];
    if let Some(summary) = &summary {
        rationale.push(format!(
            "temporal sensory influence {:.6}, 95% paired-bootstrap interval [{:.6}, {:.6}]",
            summary.mean_sensory_delta,
            summary.sensory_delta_ci_95[0],
            summary.sensory_delta_ci_95[1]
        ));
        rationale.push(format!(
            "mean paired action divergence {:.2}%, interval [{:.2}%, {:.2}%]",
            100.0 * summary.mean_action_divergence_rate,
            100.0 * summary.action_divergence_ci_95[0],
            100.0 * summary.action_divergence_ci_95[1]
        ));
    }
    rationale.push(
        "a pass establishes confirmatory mechanistic influence only; it does not show better prediction, music, workflow, or listener preference"
            .into(),
    );
    TemporalConfirmatoryConclusion {
        success,
        evidence_gate_passed,
        sample_gate_passed,
        sensory_gate_passed,
        action_gate_passed,
        issues,
        summary,
        rationale,
    }
}

fn summarize(
    records: &[&FrozenTemporalRecord],
    plan: &TemporalConfirmatoryPlan,
) -> Option<TemporalConfirmatorySummary> {
    if records.is_empty() {
        return None;
    }
    let sensory: Vec<_> = records
        .iter()
        .map(|record| pair_sensory_delta(&record.record))
        .collect();
    let action: Vec<_> = records
        .iter()
        .map(|record| pair_action_divergence(&record.record))
        .collect();
    let terminal: Vec<_> = records
        .iter()
        .map(|record| {
            if record
                .record
                .no_temporal_influence
                .terminal_inference
                .action
                != record.record.temporal_influence.terminal_inference.action
            {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Some(TemporalConfirmatorySummary {
        confirmatory_pairs: records.len(),
        mean_sensory_delta: mean(&sensory),
        sensory_delta_ci_95: bootstrap_interval(
            &sensory,
            plan.bootstrap_replicates,
            plan.rng_seed ^ 0x51A5_0A11,
        ),
        mean_action_divergence_rate: mean(&action),
        action_divergence_ci_95: bootstrap_interval(
            &action,
            plan.bootstrap_replicates,
            plan.rng_seed ^ 0xAC71_0A11,
        ),
        terminal_action_divergence_rate: mean(&terminal),
        terminal_divergence_ci_95: bootstrap_interval(
            &terminal,
            plan.bootstrap_replicates,
            plan.rng_seed ^ 0x7E2A_1A11,
        ),
    })
}

fn pair_sensory_delta(record: &TemporalAblationRecord) -> f64 {
    let mut total = 0.0;
    let mut channels = 0usize;
    for (control, treatment) in record
        .no_temporal_influence
        .frames
        .iter()
        .zip(&record.temporal_influence.frames)
    {
        total += f64::from(sensory_l1(
            control.temporal_sensory,
            treatment.temporal_sensory,
        ));
        channels += 6;
    }
    if channels == 0 {
        0.0
    } else {
        total / channels as f64
    }
}

fn pair_action_divergence(record: &TemporalAblationRecord) -> f64 {
    let frames = record.no_temporal_influence.frames.len();
    if frames == 0 {
        return 0.0;
    }
    record
        .no_temporal_influence
        .frames
        .iter()
        .zip(&record.temporal_influence.frames)
        .filter(|(control, treatment)| control.inference.action != treatment.inference.action)
        .count() as f64
        / frames as f64
}

fn sensory_l1(left: CognitiveSensoryVector, right: CognitiveSensoryVector) -> f32 {
    (left.spectral_centroid - right.spectral_centroid).abs()
        + (left.spectral_flux - right.spectral_flux).abs()
        + (left.rhythm_entropy - right.rhythm_entropy).abs()
        + (left.harmonic_tension - right.harmonic_tension).abs()
        + (left.rms_energy - right.rms_energy).abs()
        + (left.zero_crossing_rate - right.zero_crossing_rate).abs()
}

fn bootstrap_interval(values: &[f64], replicates: usize, seed: u64) -> [f64; 2] {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.gen_range(0..values.len())];
        }
        samples.push(total / values.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    [
        samples[percentile_index(samples.len(), 0.025)],
        samples[percentile_index(samples.len(), 0.975)],
    ]
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len.saturating_sub(1)) as f64 * percentile).round() as usize).min(len.saturating_sub(1))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn pass_fail(value: bool) -> &'static str {
    if value { "passed" } else { "failed" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MusicalState;
    use crate::cognitive_session::CognitiveSessionConfig;
    use crate::cognitive_session_experiment::run_temporal_ablation_pair;
    use symthaea_music_theory::{MusicalIntent, PitchClass, Style, compose_sonata_with_plan};

    fn record(seed: u64, split: StudySplit) -> FrozenTemporalRecord {
        let intent = MusicalIntent {
            seed,
            tonic: PitchClass::C,
            ..MusicalIntent::default()
        };
        let realization = compose_sonata_with_plan(&intent, &Style::Sonata.spec()).unwrap();
        FrozenTemporalRecord {
            family_id: format!("family-{seed}"),
            split,
            record: run_temporal_ablation_pair(
                format!("trial-{seed}"),
                format!("{seed:064x}"),
                &realization,
                &MusicalState::default(),
                seed,
                CognitiveSessionConfig::default(),
            )
            .unwrap(),
        }
    }

    fn plan() -> TemporalConfirmatoryPlan {
        TemporalConfirmatoryPlan {
            analysis_version: TEMPORAL_CONFIRMATORY_VERSION.into(),
            alpha: 0.05,
            minimum_pairs: MIN_TEMPORAL_CONFIRMATORY_PAIRS,
            bootstrap_replicates: MIN_TEMPORAL_BOOTSTRAP_REPLICATES,
            rng_seed: 42,
        }
    }

    #[test]
    fn family_split_leakage_is_rejected() {
        let mut records = vec![
            record(101, StudySplit::Pilot),
            record(103, StudySplit::Confirmatory),
        ];
        records[1].family_id = records[0].family_id.clone();
        assert!(
            validate_temporal_confirmatory(&records, &plan())
                .iter()
                .any(|issue| matches!(
                    issue,
                    TemporalConfirmatoryIssue::FamilyCrossesSplits { .. }
                ))
        );
    }

    #[test]
    fn empty_confirmatory_set_cannot_pass() {
        let conclusion = analyze_temporal_confirmatory(&[], &plan());
        assert!(!conclusion.success);
        assert!(conclusion.summary.is_none());
    }

    #[test]
    fn zero_temporal_treatment_cannot_clear_action_gate() {
        let mut value = record(107, StudySplit::Confirmatory);
        value.record.temporal_influence = value.record.no_temporal_influence.clone();
        let summary = summarize(&[&value], &plan()).unwrap();
        assert_eq!(summary.mean_action_divergence_rate, 0.0);
    }
}
