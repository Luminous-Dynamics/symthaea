// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen paired ablation for the temporal HDC/CfC contribution.
//!
//! Each trial runs the same score, seed, FEP RNG stream, goals, observation
//! windows, and state feedback twice. The control keeps the HDC/CfC network in
//! the execution path but sets `temporal_blend` to zero, so its latent state
//! cannot alter FEP observations. The treatment uses the frozen non-zero blend.
//!
//! A positive result means only that temporal modulation measurably influenced
//! the observed FEP trajectory. It does not establish better music, listener
//! preference, or superiority over a hand-authored policy.

use crate::MusicalState;
use crate::cognitive_session::{
    CognitiveSensoryVector, CognitiveSessionConfig, CognitiveSessionError, CognitiveSessionTrace,
    run_sonata_cognitive_session,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_music_theory::SonataRealization;

/// Frozen minimum for the first mechanistic temporal ablation.
pub const MIN_TEMPORAL_ABLATION_PAIRS: usize = 16;
/// Minimum average per-channel sensory change required before temporal influence
/// is considered practically non-zero.
pub const MIN_TEMPORAL_SENSORY_DELTA: f32 = 0.005;

/// One paired control/treatment run over an identical frozen Sonata input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalAblationRecord {
    pub trial_id: String,
    pub frozen_input_sha256: String,
    pub seed: u64,
    /// HDC/CfC executes, but its output has zero weight in the FEP observation.
    pub no_temporal_influence: CognitiveSessionTrace,
    /// Identical run except for the frozen non-zero temporal blend.
    pub temporal_influence: CognitiveSessionTrace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalAblationIssue {
    NoRecords,
    TooFewRecords { found: usize, required: usize },
    EmptyTrialId { index: usize },
    DuplicateTrialId { trial_id: String },
    InvalidFrozenDigest { trial_id: String },
    DuplicateFrozenInput { frozen_input_sha256: String },
    DuplicateSeed { seed: u64 },
    RecordSeedMismatch { trial_id: String },
    InvalidControlSession { trial_id: String },
    InvalidTreatmentSession { trial_id: String },
    ControlBlendNotZero { trial_id: String },
    TreatmentBlendNotPositive { trial_id: String },
    NonBlendConfigurationMismatch { trial_id: String },
    FepSeedMismatch { trial_id: String },
    GoalMismatch { trial_id: String },
    FrameCountMismatch { trial_id: String },
    FrameRegionMismatch { trial_id: String, sequence: u32 },
    RawSensoryMismatch { trial_id: String, sequence: u32 },
    NonFiniteMeasurement { trial_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TemporalAblationSummary {
    pub pairs: usize,
    pub total_paired_frames: usize,
    pub divergent_action_frames: usize,
    pub action_divergence_rate: f32,
    pub divergent_terminal_actions: usize,
    pub terminal_action_divergence_rate: f32,
    pub mean_temporal_sensory_delta: f32,
    pub mean_free_energy_delta: f64,
    pub mean_prediction_error_delta: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalAblationConclusion {
    pub success: bool,
    pub sample_gate_passed: bool,
    pub evidence_gate_passed: bool,
    pub sensory_influence_gate_passed: bool,
    pub action_influence_gate_passed: bool,
    pub issues: Vec<TemporalAblationIssue>,
    pub summary: Option<TemporalAblationSummary>,
    pub rationale: Vec<String>,
}

/// Execute one paired ablation with all variables except `temporal_blend` held
/// fixed. The treatment blend is bounded by `CognitiveSessionConfig` and must
/// remain strictly positive after bounding.
/// # What this harness CAN and CANNOT show (audited 2026-07-30)
///
/// It is a well-built causal harness — matched seed, matched input, frozen input
/// digest, a control arm with `temporal_blend = 0.0`. But its record holds two
/// [`CognitiveSessionTrace`]s, **not two scores and not two audio buffers**.
///
/// So it can only ever demonstrate that HDC/CfC state differs between arms. It
/// **cannot** detect whether that difference reaches the music — and on the live
/// `muse_studio` path it does not: every state field
/// [`CognitiveSession::bridge_observation`] writes has zero readers, and the one
/// consumed output is pinned by a hardcoded caller argument. See that method's
/// doc comment for the verified detail.
///
/// A positive result here is therefore **not** evidence that temporal influence
/// shapes the output. To test that, the record would need a digest of the
/// RENDERED SCORE per arm; the expectation on today's code is that the two
/// digests would be identical.
pub fn run_temporal_ablation_pair(
    trial_id: impl Into<String>,
    frozen_input_sha256: impl Into<String>,
    realization: &SonataRealization,
    initial_state: &MusicalState,
    seed: u64,
    treatment_config: CognitiveSessionConfig,
) -> Result<TemporalAblationRecord, CognitiveSessionError> {
    let mut control_config = treatment_config.clone();
    control_config.temporal_blend = 0.0;
    let no_temporal_influence =
        run_sonata_cognitive_session(realization, initial_state, seed, control_config)?;
    let temporal_influence =
        run_sonata_cognitive_session(realization, initial_state, seed, treatment_config)?;
    Ok(TemporalAblationRecord {
        trial_id: trial_id.into(),
        frozen_input_sha256: frozen_input_sha256.into(),
        seed,
        no_temporal_influence,
        temporal_influence,
    })
}

pub fn validate_temporal_ablation(
    records: &[TemporalAblationRecord],
) -> Vec<TemporalAblationIssue> {
    let mut issues = Vec::new();
    if records.is_empty() {
        issues.push(TemporalAblationIssue::NoRecords);
        return issues;
    }
    if records.len() < MIN_TEMPORAL_ABLATION_PAIRS {
        issues.push(TemporalAblationIssue::TooFewRecords {
            found: records.len(),
            required: MIN_TEMPORAL_ABLATION_PAIRS,
        });
    }

    let mut ids = BTreeSet::new();
    let mut digests = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        if record.trial_id.trim().is_empty() {
            issues.push(TemporalAblationIssue::EmptyTrialId { index });
        } else if !ids.insert(record.trial_id.clone()) {
            issues.push(TemporalAblationIssue::DuplicateTrialId {
                trial_id: record.trial_id.clone(),
            });
        }
        if !is_sha256(&record.frozen_input_sha256) {
            issues.push(TemporalAblationIssue::InvalidFrozenDigest {
                trial_id: record.trial_id.clone(),
            });
        } else if !digests.insert(record.frozen_input_sha256.clone()) {
            issues.push(TemporalAblationIssue::DuplicateFrozenInput {
                frozen_input_sha256: record.frozen_input_sha256.clone(),
            });
        }
        if !seeds.insert(record.seed) {
            issues.push(TemporalAblationIssue::DuplicateSeed { seed: record.seed });
        }

        let control = &record.no_temporal_influence;
        let treatment = &record.temporal_influence;
        if control.seed != record.seed || treatment.seed != record.seed {
            issues.push(TemporalAblationIssue::RecordSeedMismatch {
                trial_id: record.trial_id.clone(),
            });
        }
        if !control.is_valid() {
            issues.push(TemporalAblationIssue::InvalidControlSession {
                trial_id: record.trial_id.clone(),
            });
        }
        if !treatment.is_valid() {
            issues.push(TemporalAblationIssue::InvalidTreatmentSession {
                trial_id: record.trial_id.clone(),
            });
        }
        if control.config.temporal_blend != 0.0 {
            issues.push(TemporalAblationIssue::ControlBlendNotZero {
                trial_id: record.trial_id.clone(),
            });
        }
        if treatment.config.temporal_blend <= 0.0 {
            issues.push(TemporalAblationIssue::TreatmentBlendNotPositive {
                trial_id: record.trial_id.clone(),
            });
        }
        if config_without_blend(&control.config) != config_without_blend(&treatment.config) {
            issues.push(TemporalAblationIssue::NonBlendConfigurationMismatch {
                trial_id: record.trial_id.clone(),
            });
        }
        if control.fep_rng_seed != treatment.fep_rng_seed {
            issues.push(TemporalAblationIssue::FepSeedMismatch {
                trial_id: record.trial_id.clone(),
            });
        }
        if control.fep_goal_preferences != treatment.fep_goal_preferences
            || control.fep_goal_precision != treatment.fep_goal_precision
        {
            issues.push(TemporalAblationIssue::GoalMismatch {
                trial_id: record.trial_id.clone(),
            });
        }
        if control.frames.len() != treatment.frames.len() {
            issues.push(TemporalAblationIssue::FrameCountMismatch {
                trial_id: record.trial_id.clone(),
            });
            continue;
        }
        for (left, right) in control.frames.iter().zip(&treatment.frames) {
            if left.sequence != right.sequence
                || left.section != right.section
                || left.start != right.start
                || left.end != right.end
            {
                issues.push(TemporalAblationIssue::FrameRegionMismatch {
                    trial_id: record.trial_id.clone(),
                    sequence: left.sequence,
                });
            }
            if left.symbolic_profile != right.symbolic_profile
                || left.raw_sensory != right.raw_sensory
                || left.input_state_fingerprint != right.input_state_fingerprint
            {
                issues.push(TemporalAblationIssue::RawSensoryMismatch {
                    trial_id: record.trial_id.clone(),
                    sequence: left.sequence,
                });
            }
        }
        if !record_measurements_finite(record) {
            issues.push(TemporalAblationIssue::NonFiniteMeasurement {
                trial_id: record.trial_id.clone(),
            });
        }
    }
    issues
}

pub fn summarize_temporal_ablation(
    records: &[TemporalAblationRecord],
) -> Option<TemporalAblationSummary> {
    if records.is_empty() {
        return None;
    }
    let mut summary = TemporalAblationSummary {
        pairs: records.len(),
        ..TemporalAblationSummary::default()
    };
    let mut sensory_delta = 0.0f32;
    let mut sensory_channels = 0usize;
    let mut free_energy_delta = 0.0f64;
    let mut prediction_error_delta = 0.0f64;

    for record in records {
        let control = &record.no_temporal_influence;
        let treatment = &record.temporal_influence;
        if control.terminal_inference.action != treatment.terminal_inference.action {
            summary.divergent_terminal_actions += 1;
        }
        for (left, right) in control.frames.iter().zip(&treatment.frames) {
            summary.total_paired_frames += 1;
            if left.inference.action != right.inference.action {
                summary.divergent_action_frames += 1;
            }
            sensory_delta += sensory_l1(left.temporal_sensory, right.temporal_sensory);
            sensory_channels += 6;
            free_energy_delta += right.inference.free_energy - left.inference.free_energy;
            prediction_error_delta +=
                right.inference.prediction_error - left.inference.prediction_error;
        }
    }

    if summary.total_paired_frames > 0 {
        let frames = summary.total_paired_frames as f32;
        summary.action_divergence_rate = summary.divergent_action_frames as f32 / frames;
        summary.mean_free_energy_delta /= f64::from(frames);
        summary.mean_prediction_error_delta /= f64::from(frames);
    }
    if summary.pairs > 0 {
        summary.terminal_action_divergence_rate =
            summary.divergent_terminal_actions as f32 / summary.pairs as f32;
    }
    if sensory_channels > 0 {
        summary.mean_temporal_sensory_delta = sensory_delta / sensory_channels as f32;
    }
    Some(summary)
}

/// Apply the frozen V7 mechanistic gate.
///
/// Success means that the paired evidence is valid, the temporal state changed
/// FEP observations by a practical amount, and at least one paired action path
/// diverged. It is deliberately not a musical-quality gate.
pub fn conclude_temporal_ablation(
    records: &[TemporalAblationRecord],
) -> TemporalAblationConclusion {
    let issues = validate_temporal_ablation(records);
    let sample_gate_passed = records.len() >= MIN_TEMPORAL_ABLATION_PAIRS;
    let evidence_gate_passed = issues.is_empty();
    let summary = summarize_temporal_ablation(records);
    let sensory_influence_gate_passed = summary
        .as_ref()
        .is_some_and(|value| value.mean_temporal_sensory_delta >= MIN_TEMPORAL_SENSORY_DELTA);
    let action_influence_gate_passed = summary
        .as_ref()
        .is_some_and(|value| value.divergent_action_frames > 0);
    let success = sample_gate_passed
        && evidence_gate_passed
        && sensory_influence_gate_passed
        && action_influence_gate_passed;
    let mut rationale = vec![format!(
        "sample gate {} ({} unique paired seeds required)",
        pass_fail(sample_gate_passed),
        MIN_TEMPORAL_ABLATION_PAIRS
    )];
    rationale.push(format!(
        "paired evidence gate {}",
        pass_fail(evidence_gate_passed)
    ));
    if let Some(value) = &summary {
        rationale.push(format!(
            "mean per-channel temporal sensory delta {:.6} (minimum {:.6})",
            value.mean_temporal_sensory_delta, MIN_TEMPORAL_SENSORY_DELTA
        ));
        rationale.push(format!(
            "{} of {} paired frame actions diverged ({:.2}%)",
            value.divergent_action_frames,
            value.total_paired_frames,
            100.0 * value.action_divergence_rate
        ));
    }
    rationale.push(
        "a pass establishes mechanistic influence only; usefulness, preference, and musical quality require separate frozen evaluations"
            .into(),
    );
    TemporalAblationConclusion {
        success,
        sample_gate_passed,
        evidence_gate_passed,
        sensory_influence_gate_passed,
        action_influence_gate_passed,
        issues,
        summary,
        rationale,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConfigWithoutBlend {
    genesis_namespace: String,
    windows_per_section: u8,
    state_feedback_strength_bits: u32,
    hdc_dimension: usize,
    cfc_layer_sizes: Vec<usize>,
}

fn config_without_blend(config: &CognitiveSessionConfig) -> ConfigWithoutBlend {
    ConfigWithoutBlend {
        genesis_namespace: config.genesis_namespace.clone(),
        windows_per_section: config.windows_per_section,
        state_feedback_strength_bits: config.state_feedback_strength.to_bits(),
        hdc_dimension: config.hdc_dimension,
        cfc_layer_sizes: config.cfc_layer_sizes.clone(),
    }
}

fn sensory_l1(left: CognitiveSensoryVector, right: CognitiveSensoryVector) -> f32 {
    (left.spectral_centroid - right.spectral_centroid).abs()
        + (left.spectral_flux - right.spectral_flux).abs()
        + (left.rhythm_entropy - right.rhythm_entropy).abs()
        + (left.harmonic_tension - right.harmonic_tension).abs()
        + (left.rms_energy - right.rms_energy).abs()
        + (left.zero_crossing_rate - right.zero_crossing_rate).abs()
}

fn record_measurements_finite(record: &TemporalAblationRecord) -> bool {
    [&record.no_temporal_influence, &record.temporal_influence]
        .into_iter()
        .flat_map(|trace| trace.frames.iter())
        .all(|frame| {
            sensory_values(frame.temporal_sensory)
                .into_iter()
                .all(f32::is_finite)
                && frame.inference.free_energy.is_finite()
                && frame.inference.prediction_error.is_finite()
        })
}

fn sensory_values(value: CognitiveSensoryVector) -> [f32; 6] {
    [
        value.spectral_centroid,
        value.spectral_flux,
        value.rhythm_entropy,
        value.harmonic_tension,
        value.rms_energy,
        value.zero_crossing_rate,
    ]
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn pass_fail(value: bool) -> &'static str {
    if value { "passed" } else { "failed" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{MusicalIntent, PitchClass, Style, compose_sonata_with_plan};

    fn realization(seed: u64) -> SonataRealization {
        let intent = MusicalIntent {
            seed,
            tonic: PitchClass::C,
            ..MusicalIntent::default()
        };
        compose_sonata_with_plan(&intent, &Style::Sonata.spec()).unwrap()
    }

    fn record(seed: u64) -> TemporalAblationRecord {
        run_temporal_ablation_pair(
            format!("trial-{seed}"),
            format!("{seed:064x}"),
            &realization(seed),
            &MusicalState::default(),
            seed,
            CognitiveSessionConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn paired_ablation_holds_raw_stream_and_fep_seed_fixed() {
        let record = record(101);
        assert_eq!(record.no_temporal_influence.config.temporal_blend, 0.0);
        assert!(record.temporal_influence.config.temporal_blend > 0.0);
        assert_eq!(
            record.no_temporal_influence.fep_rng_seed,
            record.temporal_influence.fep_rng_seed
        );
        assert!(
            record
                .no_temporal_influence
                .frames
                .iter()
                .zip(&record.temporal_influence.frames)
                .all(|(left, right)| left.raw_sensory == right.raw_sensory)
        );
        let issues = validate_temporal_ablation(&[record]);
        assert_eq!(
            issues,
            vec![TemporalAblationIssue::TooFewRecords {
                found: 1,
                required: MIN_TEMPORAL_ABLATION_PAIRS,
            }]
        );
    }

    #[test]
    fn configuration_drift_is_rejected() {
        let mut record = record(103);
        record.temporal_influence.config.windows_per_section += 1;
        record.temporal_influence.session_fingerprint.clear();
        let issues = validate_temporal_ablation(&[record]);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            TemporalAblationIssue::NonBlendConfigurationMismatch { .. }
        )));
        assert!(
            issues.iter().any(|issue| matches!(
                issue,
                TemporalAblationIssue::InvalidTreatmentSession { .. }
            ))
        );
    }

    #[test]
    fn summary_keeps_mechanism_separate_from_quality() {
        let record = record(107);
        let summary = summarize_temporal_ablation(&[record]).unwrap();
        assert!(summary.mean_temporal_sensory_delta > 0.0);
        let conclusion = conclude_temporal_ablation(&[]);
        assert!(!conclusion.success);
        assert!(
            conclusion
                .rationale
                .iter()
                .any(|line| line.contains("musical quality"))
        );
    }
}
