// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independently calibrated candidate-local transient contrast.
//!
//! The existing rendered-attack gate asks whether baseline/candidate *difference*
//! energy appears locally around a predeclared performed attack. That is useful
//! but weaker than showing that the candidate waveform itself contains a new
//! local activity rise. This module keeps those propositions separate.
//!
//! V1 operates only on the already-recorded time-domain first-difference
//! activity descriptors in `RenderedAttackEvidenceV1`. It is not a spectral
//! onset detector and it makes no audibility or perceptual-salience claim.

use crate::evidence_digest::rendered_attack_evidence::{
    RENDERED_ATTACK_EVIDENCE_VERSION, RenderedAttackEvidenceV1,
};
use serde::{Deserialize, Serialize};

pub const RENDERED_TRANSIENT_CONTRAST_VERSION: &str = "rendered-transient-contrast-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderedTransientContrastConfigV1 {
    /// Denominator floor for pre/post and candidate/baseline growth ratios.
    pub activity_floor: f64,
    /// Candidate post-window first-difference activity must exceed this floor.
    pub minimum_candidate_post_activity: f64,
    /// Candidate activity must rise by at least this factor from pre to post.
    pub minimum_candidate_growth_ratio: f64,
    /// Candidate pre/post growth must exceed matched baseline growth by this factor.
    pub minimum_growth_excess_over_baseline: f64,
}

impl Default for RenderedTransientContrastConfigV1 {
    fn default() -> Self {
        Self {
            activity_floor: 1.0e-6,
            minimum_candidate_post_activity: 1.0e-4,
            minimum_candidate_growth_ratio: 2.0,
            minimum_growth_excess_over_baseline: 2.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedTransientContrastResultV1 {
    pub contrast_version: String,
    pub source_evidence_version: String,
    pub config: RenderedTransientContrastConfigV1,

    pub baseline_pre_activity: f64,
    pub baseline_post_activity: f64,
    pub baseline_growth_ratio: f64,

    pub candidate_pre_activity: f64,
    pub candidate_post_activity: f64,
    pub candidate_growth_ratio: f64,

    pub candidate_growth_excess_over_baseline: f64,

    pub candidate_post_activity_requirement_met: bool,
    pub candidate_growth_requirement_met: bool,
    pub growth_excess_requirement_met: bool,
    pub candidate_local_transient_contrast_detected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedTransientContrastErrorV1 {
    UnsupportedEvidenceVersion,
    InvalidConfig,
    NonFiniteEvidence,
}

pub fn evaluate_rendered_transient_contrast(
    evidence: &RenderedAttackEvidenceV1,
    config: RenderedTransientContrastConfigV1,
) -> Result<RenderedTransientContrastResultV1, RenderedTransientContrastErrorV1> {
    if evidence.evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION {
        return Err(RenderedTransientContrastErrorV1::UnsupportedEvidenceVersion);
    }
    if !valid_config(config) {
        return Err(RenderedTransientContrastErrorV1::InvalidConfig);
    }

    let baseline_pre_activity = evidence.baseline_pre.mean_absolute_first_difference;
    let baseline_post_activity = evidence.baseline_post.mean_absolute_first_difference;
    let candidate_pre_activity = evidence.candidate_pre.mean_absolute_first_difference;
    let candidate_post_activity = evidence.candidate_post.mean_absolute_first_difference;

    if ![
        baseline_pre_activity,
        baseline_post_activity,
        candidate_pre_activity,
        candidate_post_activity,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        return Err(RenderedTransientContrastErrorV1::NonFiniteEvidence);
    }

    let baseline_growth_ratio =
        baseline_post_activity / baseline_pre_activity.max(config.activity_floor);
    let candidate_growth_ratio =
        candidate_post_activity / candidate_pre_activity.max(config.activity_floor);
    let candidate_growth_excess_over_baseline =
        candidate_growth_ratio / baseline_growth_ratio.max(config.activity_floor);

    if !baseline_growth_ratio.is_finite()
        || !candidate_growth_ratio.is_finite()
        || !candidate_growth_excess_over_baseline.is_finite()
    {
        return Err(RenderedTransientContrastErrorV1::NonFiniteEvidence);
    }

    let candidate_post_activity_requirement_met =
        candidate_post_activity >= config.minimum_candidate_post_activity;
    let candidate_growth_requirement_met =
        candidate_growth_ratio >= config.minimum_candidate_growth_ratio;
    let growth_excess_requirement_met =
        candidate_growth_excess_over_baseline >= config.minimum_growth_excess_over_baseline;

    Ok(RenderedTransientContrastResultV1 {
        contrast_version: RENDERED_TRANSIENT_CONTRAST_VERSION.into(),
        source_evidence_version: evidence.evidence_version.clone(),
        config,
        baseline_pre_activity,
        baseline_post_activity,
        baseline_growth_ratio,
        candidate_pre_activity,
        candidate_post_activity,
        candidate_growth_ratio,
        candidate_growth_excess_over_baseline,
        candidate_post_activity_requirement_met,
        candidate_growth_requirement_met,
        growth_excess_requirement_met,
        candidate_local_transient_contrast_detected: candidate_post_activity_requirement_met
            && candidate_growth_requirement_met
            && growth_excess_requirement_met,
    })
}

fn valid_config(config: RenderedTransientContrastConfigV1) -> bool {
    config.activity_floor.is_finite()
        && config.activity_floor > 0.0
        && config.minimum_candidate_post_activity.is_finite()
        && config.minimum_candidate_post_activity >= 0.0
        && config.minimum_candidate_growth_ratio.is_finite()
        && config.minimum_candidate_growth_ratio > 1.0
        && config.minimum_growth_excess_over_baseline.is_finite()
        && config.minimum_growth_excess_over_baseline > 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::rendered_attack_evidence::measure_rendered_attack;

    const SAMPLE_RATE: u32 = 4_000;
    const ATTACK_TIME: f32 = 0.5;
    const WINDOW: f32 = 0.1;

    fn zero_audio() -> Vec<[f32; 2]> {
        vec![[0.0, 0.0]; SAMPLE_RATE as usize]
    }

    fn sine_audio(amplitude: f32, frequency_hz: f32, start_sample: usize) -> Vec<[f32; 2]> {
        let mut audio = zero_audio();
        let rate = SAMPLE_RATE as f32;
        for (index, frame) in audio.iter_mut().enumerate().skip(start_sample) {
            let t = index as f32 / rate;
            let sample = amplitude * (std::f32::consts::TAU * frequency_hz * t).sin();
            *frame = [sample, sample];
        }
        audio
    }

    fn evidence(
        baseline: &[[f32; 2]],
        candidate: &[[f32; 2]],
    ) -> RenderedAttackEvidenceV1 {
        measure_rendered_attack(
            baseline,
            candidate,
            SAMPLE_RATE,
            ATTACK_TIME,
            WINDOW,
            WINDOW,
        )
        .unwrap()
    }

    fn evaluate(
        baseline: &[[f32; 2]],
        candidate: &[[f32; 2]],
    ) -> RenderedTransientContrastResultV1 {
        evaluate_rendered_transient_contrast(
            &evidence(baseline, candidate),
            RenderedTransientContrastConfigV1::default(),
        )
        .unwrap()
    }

    #[test]
    fn frozen_negative_controls_do_not_trigger_candidate_local_transient_contrast() {
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;
        let silence = zero_audio();

        let identical = evaluate(&silence, &silence);
        assert!(!identical.candidate_local_transient_contrast_detected);

        // Candidate activity exists on both sides of the marked time, so no
        // candidate-local emergence should be inferred.
        let continuous_candidate = sine_audio(0.1, 100.0, 0);
        let continuous = evaluate(&silence, &continuous_candidate);
        assert!(continuous.candidate_post_activity_requirement_met);
        assert!(!continuous.candidate_growth_requirement_met);
        assert!(!continuous.candidate_local_transient_contrast_detected);

        // Both arms begin the same tone at the same marked time. There is a
        // real acoustic transient, but it is not candidate-specific.
        let shared_onset = sine_audio(0.1, 100.0, center);
        let shared = evaluate(&shared_onset, &shared_onset);
        assert!(shared.candidate_growth_requirement_met);
        assert!(!shared.growth_excess_requirement_met);
        assert!(!shared.candidate_local_transient_contrast_detected);
    }

    #[test]
    fn frozen_positive_controls_trigger_candidate_local_transient_contrast() {
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;
        let silence = zero_audio();

        // Positive A: candidate-only tone begins exactly at the marked time.
        let candidate_tone = sine_audio(0.1, 100.0, center);
        let tone = evaluate(&silence, &candidate_tone);
        assert!(tone.candidate_post_activity_requirement_met);
        assert!(tone.candidate_growth_requirement_met);
        assert!(tone.growth_excess_requirement_met);
        assert!(tone.candidate_local_transient_contrast_detected);

        // Positive B: both arms contain a continuous low-level tone, but only
        // the candidate gains a short high-frequency burst at the marked time.
        let baseline = sine_audio(0.02, 100.0, 0);
        let mut candidate = baseline.clone();
        for (offset, frame) in candidate[center..center + 80].iter_mut().enumerate() {
            let envelope = 1.0 - offset as f32 / 80.0;
            let burst = if offset % 2 == 0 { 0.2 } else { -0.2 } * envelope;
            frame[0] += burst;
            frame[1] -= burst;
        }
        let burst = evaluate(&baseline, &candidate);
        assert!(burst.candidate_post_activity_requirement_met);
        assert!(burst.candidate_growth_requirement_met);
        assert!(burst.growth_excess_requirement_met);
        assert!(burst.candidate_local_transient_contrast_detected);
    }

    #[test]
    fn invalid_configuration_and_non_finite_evidence_fail_closed() {
        let audio = zero_audio();
        let evidence = evidence(&audio, &audio);
        let invalid = RenderedTransientContrastConfigV1 {
            minimum_candidate_growth_ratio: 1.0,
            ..RenderedTransientContrastConfigV1::default()
        };
        assert_eq!(
            evaluate_rendered_transient_contrast(&evidence, invalid),
            Err(RenderedTransientContrastErrorV1::InvalidConfig)
        );

        let mut malformed = evidence;
        malformed.candidate_post.mean_absolute_first_difference = f64::NAN;
        assert_eq!(
            evaluate_rendered_transient_contrast(
                &malformed,
                RenderedTransientContrastConfigV1::default(),
            ),
            Err(RenderedTransientContrastErrorV1::NonFiniteEvidence)
        );
    }
}
