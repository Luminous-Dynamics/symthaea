// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predeclared gate for localized A/B audio change around a performed attack.
//!
//! The gate is deliberately narrower than an onset detector. It asks whether
//! the baseline/candidate *difference signal* is materially stronger after a
//! predeclared performed attack time than before it. Synthetic controls freeze
//! the v1 operating point before the gate is applied to Sonata evidence.

use crate::evidence_digest::rendered_attack_evidence::{
    RENDERED_ATTACK_EVIDENCE_VERSION, RenderedAttackEvidenceV1,
};
use serde::{Deserialize, Serialize};

pub const RENDERED_ATTACK_GATE_VERSION: &str = "rendered-attack-gate-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackGateConfigV1 {
    /// Floor used only in the pre/post ratio denominator.
    pub ratio_floor_rms: f64,
    /// Minimum absolute post-attack A/B-difference RMS.
    pub minimum_post_difference_rms: f64,
    /// Minimum post/pre A/B-difference RMS ratio.
    pub minimum_post_to_pre_ratio: f64,
}

impl Default for RenderedAttackGateConfigV1 {
    fn default() -> Self {
        Self {
            ratio_floor_rms: 1.0e-6,
            minimum_post_difference_rms: 1.0e-4,
            minimum_post_to_pre_ratio: 2.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackGateResultV1 {
    pub gate_version: String,
    pub source_evidence_version: String,
    pub config: RenderedAttackGateConfigV1,
    pub pre_difference_rms: f64,
    pub post_difference_rms: f64,
    pub post_to_pre_ratio: f64,
    pub post_energy_requirement_met: bool,
    pub growth_requirement_met: bool,
    pub localized_change_detected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackGateErrorV1 {
    UnsupportedEvidenceVersion,
    InvalidConfig,
    NonFiniteEvidence,
}

pub fn evaluate_rendered_attack_gate(
    evidence: &RenderedAttackEvidenceV1,
    config: RenderedAttackGateConfigV1,
) -> Result<RenderedAttackGateResultV1, RenderedAttackGateErrorV1> {
    if evidence.evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION {
        return Err(RenderedAttackGateErrorV1::UnsupportedEvidenceVersion);
    }
    if !valid_config(config) {
        return Err(RenderedAttackGateErrorV1::InvalidConfig);
    }

    let pre_difference_rms = evidence.difference_pre.rms;
    let post_difference_rms = evidence.difference_post.rms;
    if !pre_difference_rms.is_finite() || !post_difference_rms.is_finite() {
        return Err(RenderedAttackGateErrorV1::NonFiniteEvidence);
    }

    let denominator = pre_difference_rms.max(config.ratio_floor_rms);
    let post_to_pre_ratio = post_difference_rms / denominator;
    if !post_to_pre_ratio.is_finite() {
        return Err(RenderedAttackGateErrorV1::NonFiniteEvidence);
    }

    let post_energy_requirement_met =
        post_difference_rms >= config.minimum_post_difference_rms;
    let growth_requirement_met = post_to_pre_ratio >= config.minimum_post_to_pre_ratio;

    Ok(RenderedAttackGateResultV1 {
        gate_version: RENDERED_ATTACK_GATE_VERSION.into(),
        source_evidence_version: evidence.evidence_version.clone(),
        config,
        pre_difference_rms,
        post_difference_rms,
        post_to_pre_ratio,
        post_energy_requirement_met,
        growth_requirement_met,
        localized_change_detected: post_energy_requirement_met && growth_requirement_met,
    })
}

fn valid_config(config: RenderedAttackGateConfigV1) -> bool {
    config.ratio_floor_rms.is_finite()
        && config.ratio_floor_rms > 0.0
        && config.minimum_post_difference_rms.is_finite()
        && config.minimum_post_difference_rms >= 0.0
        && config.minimum_post_to_pre_ratio.is_finite()
        && config.minimum_post_to_pre_ratio > 1.0
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

    #[test]
    fn frozen_negative_controls_do_not_trigger() {
        let baseline = zero_audio();
        let identical = evidence(&baseline, &baseline);
        let identical_result =
            evaluate_rendered_attack_gate(&identical, RenderedAttackGateConfigV1::default())
                .unwrap();
        assert!(!identical_result.localized_change_detected);

        // A continuous 100 Hz difference exists for the full clip, including
        // both matched windows. Because the A/B difference did not begin at
        // the marked attack, its pre/post RMS ratio should stay near unity.
        let continuous = sine_audio(0.1, 100.0, 0);
        let continuous_evidence = evidence(&baseline, &continuous);
        let continuous_result = evaluate_rendered_attack_gate(
            &continuous_evidence,
            RenderedAttackGateConfigV1::default(),
        )
        .unwrap();
        assert!(continuous_result.post_energy_requirement_met);
        assert!(!continuous_result.growth_requirement_met);
        assert!(!continuous_result.localized_change_detected);
        assert!(continuous_result.post_to_pre_ratio < 1.1);
    }

    #[test]
    fn frozen_positive_controls_trigger() {
        let baseline = zero_audio();
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;

        // Positive control A: a new 100 Hz tone begins exactly at the marked
        // attack and persists through the post window.
        let new_tone = sine_audio(0.1, 100.0, center);
        let tone_result = evaluate_rendered_attack_gate(
            &evidence(&baseline, &new_tone),
            RenderedAttackGateConfigV1::default(),
        )
        .unwrap();
        assert!(tone_result.localized_change_detected);

        // Positive control B: a short alternating burst begins at the attack.
        let mut burst = zero_audio();
        for (offset, frame) in burst[center..center + 80].iter_mut().enumerate() {
            let envelope = 1.0 - offset as f32 / 80.0;
            let sample = if offset % 2 == 0 { 0.2 } else { -0.2 } * envelope;
            *frame = [sample, -sample];
        }
        let burst_result = evaluate_rendered_attack_gate(
            &evidence(&baseline, &burst),
            RenderedAttackGateConfigV1::default(),
        )
        .unwrap();
        assert!(burst_result.localized_change_detected);
    }

    #[test]
    fn invalid_configuration_fails_closed() {
        let baseline = zero_audio();
        let evidence = evidence(&baseline, &baseline);
        let invalid = RenderedAttackGateConfigV1 {
            minimum_post_to_pre_ratio: 1.0,
            ..RenderedAttackGateConfigV1::default()
        };
        assert_eq!(
            evaluate_rendered_attack_gate(&evidence, invalid),
            Err(RenderedAttackGateErrorV1::InvalidConfig)
        );
    }
}
