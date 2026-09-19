// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind two distinct acoustic propositions to one exact rendered-attack
//! evidence record.
//!
//! `RenderedAttackGateV1` asks whether baseline/candidate difference energy
//! emerges locally at a predeclared performed attack. `RenderedTransientContrastV1`
//! asks whether the candidate waveform itself shows more local activity growth
//! than the matched baseline. Neither proposition implies the other, and neither
//! is a perceptual or musical-quality claim.

use crate::evidence_digest::{
    canonical_json_sha256,
    rendered_attack_evidence::{RENDERED_ATTACK_EVIDENCE_VERSION, RenderedAttackEvidenceV1},
    rendered_attack_gate::{
        RENDERED_ATTACK_GATE_VERSION, RenderedAttackGateConfigV1, RenderedAttackGateErrorV1,
        RenderedAttackGateResultV1, evaluate_rendered_attack_gate,
    },
    rendered_transient_contrast::{
        RENDERED_TRANSIENT_CONTRAST_VERSION, RenderedTransientContrastConfigV1,
        RenderedTransientContrastErrorV1, RenderedTransientContrastResultV1,
        evaluate_rendered_transient_contrast,
    },
};
use serde::{Deserialize, Serialize};

pub const RENDERED_ATTACK_CROSSCHECK_VERSION: &str = "rendered-attack-crosscheck-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackCrosscheckClassV1 {
    /// Both independent acoustic propositions are satisfied.
    LocalizedDifferenceAndCandidateTransient,
    /// A/B difference localizes, but candidate-local transient contrast does not.
    LocalizedDifferenceOnly,
    /// Candidate-local transient contrast is present, but the A/B localized gate rejects.
    CandidateTransientOnly,
    /// Neither acoustic proposition is satisfied.
    Neither,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackCrosscheckV1 {
    pub crosscheck_version: String,
    pub source_evidence_version: String,
    pub source_evidence_sha256: String,
    /// Exact matched-window geometry retained so downstream dependency analysis
    /// can detect overlapping/pseudoreplicated acoustic observations.
    pub sample_rate: u32,
    pub attack_time_seconds: f32,
    pub center_sample: usize,
    pub pre_samples: usize,
    pub post_samples: usize,
    pub localized_difference: RenderedAttackGateResultV1,
    pub candidate_transient: RenderedTransientContrastResultV1,
    pub class: RenderedAttackCrosscheckClassV1,
}

impl RenderedAttackCrosscheckV1 {
    /// Half-open exact sample interval used by the matched-window evidence.
    pub fn sample_interval(&self) -> Option<(usize, usize)> {
        let start = self.center_sample.checked_sub(self.pre_samples)?;
        let end = self.center_sample.checked_add(self.post_samples)?;
        Some((start, end))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackCrosscheckErrorV1 {
    UnsupportedCrosscheckVersion,
    UnsupportedEvidenceVersion,
    EvidenceDigestFailed,
    InvalidEvidenceDigest,
    InvalidWindowGeometry,
    InconsistentDerivedEvidence,
    LocalizedDifferenceGate(RenderedAttackGateErrorV1),
    CandidateTransientGate(RenderedTransientContrastErrorV1),
}

pub fn evaluate_rendered_attack_crosscheck(
    evidence: &RenderedAttackEvidenceV1,
    difference_config: RenderedAttackGateConfigV1,
    transient_config: RenderedTransientContrastConfigV1,
) -> Result<RenderedAttackCrosscheckV1, RenderedAttackCrosscheckErrorV1> {
    if evidence.evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION {
        return Err(RenderedAttackCrosscheckErrorV1::UnsupportedEvidenceVersion);
    }
    if evidence.center_sample.checked_sub(evidence.pre_samples).is_none()
        || evidence.center_sample.checked_add(evidence.post_samples).is_none()
    {
        return Err(RenderedAttackCrosscheckErrorV1::InvalidWindowGeometry);
    }

    let source_evidence_sha256 = canonical_json_sha256(evidence)
        .map_err(|_| RenderedAttackCrosscheckErrorV1::EvidenceDigestFailed)?;
    let localized_difference = evaluate_rendered_attack_gate(evidence, difference_config)
        .map_err(RenderedAttackCrosscheckErrorV1::LocalizedDifferenceGate)?;
    let candidate_transient = evaluate_rendered_transient_contrast(evidence, transient_config)
        .map_err(RenderedAttackCrosscheckErrorV1::CandidateTransientGate)?;

    let class = classify(
        localized_difference.localized_change_detected,
        candidate_transient.candidate_local_transient_contrast_detected,
    );

    Ok(RenderedAttackCrosscheckV1 {
        crosscheck_version: RENDERED_ATTACK_CROSSCHECK_VERSION.into(),
        source_evidence_version: evidence.evidence_version.clone(),
        source_evidence_sha256,
        sample_rate: evidence.sample_rate,
        attack_time_seconds: evidence.attack_time_seconds,
        center_sample: evidence.center_sample,
        pre_samples: evidence.pre_samples,
        post_samples: evidence.post_samples,
        localized_difference,
        candidate_transient,
        class,
    })
}

/// Re-derive every deterministic field that can be checked without the source
/// waveform itself. This is intended for downstream consumers of serialized
/// cross-check evidence: constructor validity is not assumed merely because a
/// record deserialized successfully.
pub fn verify_rendered_attack_crosscheck(
    result: &RenderedAttackCrosscheckV1,
) -> Result<(), RenderedAttackCrosscheckErrorV1> {
    if result.crosscheck_version != RENDERED_ATTACK_CROSSCHECK_VERSION {
        return Err(RenderedAttackCrosscheckErrorV1::UnsupportedCrosscheckVersion);
    }
    if result.source_evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION
        || result.localized_difference.source_evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION
        || result.candidate_transient.source_evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION
        || result.localized_difference.gate_version != RENDERED_ATTACK_GATE_VERSION
        || result.candidate_transient.contrast_version != RENDERED_TRANSIENT_CONTRAST_VERSION
    {
        return Err(RenderedAttackCrosscheckErrorV1::UnsupportedEvidenceVersion);
    }
    if result.source_evidence_sha256.len() != 64
        || !result
            .source_evidence_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(RenderedAttackCrosscheckErrorV1::InvalidEvidenceDigest);
    }
    if result.sample_rate == 0
        || !result.attack_time_seconds.is_finite()
        || result.attack_time_seconds < 0.0
        || result.pre_samples == 0
        || result.post_samples == 0
        || result.sample_interval().is_none()
    {
        return Err(RenderedAttackCrosscheckErrorV1::InvalidWindowGeometry);
    }
    let center = result.attack_time_seconds * result.sample_rate as f32;
    if !center.is_finite() || center.round() as usize != result.center_sample {
        return Err(RenderedAttackCrosscheckErrorV1::InvalidWindowGeometry);
    }

    if !verify_localized_difference(&result.localized_difference)
        || !verify_candidate_transient(&result.candidate_transient)
        || result.class
            != classify(
                result.localized_difference.localized_change_detected,
                result
                    .candidate_transient
                    .candidate_local_transient_contrast_detected,
            )
    {
        return Err(RenderedAttackCrosscheckErrorV1::InconsistentDerivedEvidence);
    }

    Ok(())
}

fn classify(
    localized_difference: bool,
    candidate_transient: bool,
) -> RenderedAttackCrosscheckClassV1 {
    match (localized_difference, candidate_transient) {
        (true, true) => RenderedAttackCrosscheckClassV1::LocalizedDifferenceAndCandidateTransient,
        (true, false) => RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly,
        (false, true) => RenderedAttackCrosscheckClassV1::CandidateTransientOnly,
        (false, false) => RenderedAttackCrosscheckClassV1::Neither,
    }
}

fn verify_localized_difference(result: &RenderedAttackGateResultV1) -> bool {
    let config = result.config;
    if !config.ratio_floor_rms.is_finite()
        || config.ratio_floor_rms <= 0.0
        || !config.minimum_post_difference_rms.is_finite()
        || config.minimum_post_difference_rms < 0.0
        || !config.minimum_post_to_pre_ratio.is_finite()
        || config.minimum_post_to_pre_ratio <= 1.0
        || !result.pre_difference_rms.is_finite()
        || !result.post_difference_rms.is_finite()
        || !result.post_to_pre_ratio.is_finite()
    {
        return false;
    }
    let ratio = result.post_difference_rms / result.pre_difference_rms.max(config.ratio_floor_rms);
    let post_met = result.post_difference_rms >= config.minimum_post_difference_rms;
    let growth_met = ratio >= config.minimum_post_to_pre_ratio;
    result.post_to_pre_ratio == ratio
        && result.post_energy_requirement_met == post_met
        && result.growth_requirement_met == growth_met
        && result.localized_change_detected == (post_met && growth_met)
}

fn verify_candidate_transient(result: &RenderedTransientContrastResultV1) -> bool {
    let config = result.config;
    if !config.activity_floor.is_finite()
        || config.activity_floor <= 0.0
        || !config.minimum_candidate_post_activity.is_finite()
        || config.minimum_candidate_post_activity < 0.0
        || !config.minimum_candidate_growth_ratio.is_finite()
        || config.minimum_candidate_growth_ratio <= 1.0
        || !config.minimum_growth_excess_over_baseline.is_finite()
        || config.minimum_growth_excess_over_baseline <= 1.0
        || ![
            result.baseline_pre_activity,
            result.baseline_post_activity,
            result.baseline_growth_ratio,
            result.candidate_pre_activity,
            result.candidate_post_activity,
            result.candidate_growth_ratio,
            result.candidate_growth_excess_over_baseline,
        ]
        .into_iter()
        .all(f64::is_finite)
    {
        return false;
    }

    let baseline_growth =
        result.baseline_post_activity / result.baseline_pre_activity.max(config.activity_floor);
    let candidate_growth =
        result.candidate_post_activity / result.candidate_pre_activity.max(config.activity_floor);
    let growth_excess = candidate_growth / baseline_growth.max(config.activity_floor);
    let post_met = result.candidate_post_activity >= config.minimum_candidate_post_activity;
    let growth_met = candidate_growth >= config.minimum_candidate_growth_ratio;
    let excess_met = growth_excess >= config.minimum_growth_excess_over_baseline;

    result.baseline_growth_ratio == baseline_growth
        && result.candidate_growth_ratio == candidate_growth
        && result.candidate_growth_excess_over_baseline == growth_excess
        && result.candidate_post_activity_requirement_met == post_met
        && result.candidate_growth_requirement_met == growth_met
        && result.growth_excess_requirement_met == excess_met
        && result.candidate_local_transient_contrast_detected == (post_met && growth_met && excess_met)
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

    fn classify_audio(
        baseline: &[[f32; 2]],
        candidate: &[[f32; 2]],
    ) -> RenderedAttackCrosscheckV1 {
        evaluate_rendered_attack_crosscheck(
            &evidence(baseline, candidate),
            RenderedAttackGateConfigV1::default(),
            RenderedTransientContrastConfigV1::default(),
        )
        .unwrap()
    }

    #[test]
    fn crosscheck_retains_all_four_joint_outcome_classes() {
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;
        let silence = zero_audio();

        let mut both = silence.clone();
        for (offset, frame) in both[center..center + 200].iter_mut().enumerate() {
            let sample = 0.1
                * (std::f32::consts::TAU * 100.0 * offset as f32 / SAMPLE_RATE as f32).sin();
            *frame = [sample, sample];
        }
        assert_eq!(
            classify_audio(&silence, &both).class,
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceAndCandidateTransient
        );

        let mut localized_only = silence.clone();
        for frame in &mut localized_only[center..] {
            *frame = [0.1, 0.1];
        }
        assert_eq!(
            classify_audio(&silence, &localized_only).class,
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly
        );

        let mut transient_only = silence.clone();
        for (offset, frame) in transient_only[center..center + 200].iter_mut().enumerate() {
            let sample = if offset % 2 == 0 { 8.0e-5 } else { -8.0e-5 };
            *frame = [sample, -sample];
        }
        assert_eq!(
            classify_audio(&silence, &transient_only).class,
            RenderedAttackCrosscheckClassV1::CandidateTransientOnly
        );

        assert_eq!(
            classify_audio(&silence, &silence).class,
            RenderedAttackCrosscheckClassV1::Neither
        );
    }

    #[test]
    fn exact_source_evidence_identity_and_window_geometry_are_bound() {
        let baseline = zero_audio();
        let mut candidate = baseline.clone();
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;
        for (offset, frame) in candidate[center..center + 100].iter_mut().enumerate() {
            let sample = if offset % 2 == 0 { 0.1 } else { -0.1 };
            *frame = [sample, -sample];
        }
        let evidence = evidence(&baseline, &candidate);
        let result = evaluate_rendered_attack_crosscheck(
            &evidence,
            RenderedAttackGateConfigV1::default(),
            RenderedTransientContrastConfigV1::default(),
        )
        .unwrap();
        assert_eq!(
            result.source_evidence_sha256,
            canonical_json_sha256(&evidence).unwrap()
        );
        assert_eq!(result.source_evidence_sha256.len(), 64);
        assert_eq!(result.sample_rate, evidence.sample_rate);
        assert_eq!(result.center_sample, evidence.center_sample);
        assert_eq!(result.pre_samples, evidence.pre_samples);
        assert_eq!(result.post_samples, evidence.post_samples);
        assert_eq!(
            result.sample_interval(),
            Some((
                evidence.center_sample - evidence.pre_samples,
                evidence.center_sample + evidence.post_samples
            ))
        );
        assert_eq!(verify_rendered_attack_crosscheck(&result), Ok(()));
    }

    #[test]
    fn forged_derived_fields_are_rejected() {
        let audio = zero_audio();
        let mut result = classify_audio(&audio, &audio);
        result.class = RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly;
        assert_eq!(
            verify_rendered_attack_crosscheck(&result),
            Err(RenderedAttackCrosscheckErrorV1::InconsistentDerivedEvidence)
        );

        let mut result = classify_audio(&audio, &audio);
        result.center_sample += 1;
        assert_eq!(
            verify_rendered_attack_crosscheck(&result),
            Err(RenderedAttackCrosscheckErrorV1::InvalidWindowGeometry)
        );

        let mut result = classify_audio(&audio, &audio);
        result.source_evidence_sha256 = "not-a-digest".into();
        assert_eq!(
            verify_rendered_attack_crosscheck(&result),
            Err(RenderedAttackCrosscheckErrorV1::InvalidEvidenceDigest)
        );
    }
}
