// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Homogeneous replication panels for rendered-attack gate evidence.
//!
//! This layer summarizes repeated applications of the already-calibrated
//! rendered-attack gate without converting them into a musical-quality score
//! or silently requiring every subject to pass. Individual failures remain
//! first-class evidence.

use crate::evidence_digest::rendered_attack_evidence::RENDERED_ATTACK_EVIDENCE_VERSION;
use crate::evidence_digest::rendered_attack_gate::{
    RENDERED_ATTACK_GATE_VERSION, RenderedAttackGateConfigV1, RenderedAttackGateResultV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RENDERED_ATTACK_PANEL_VERSION: &str = "rendered-attack-panel-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackPanelSampleV1 {
    /// Stable panel-local subject identity, e.g. `sonata-seed-79`.
    pub subject_id: String,
    /// Stable ordinal among introduced attacks for the same subject.
    pub attack_ordinal: usize,
    /// Performed attack time supplied to the matched-window measurement.
    pub attack_time_secs: f32,
    /// Frozen gate result for this attack.
    pub gate: RenderedAttackGateResultV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackSubjectSummaryV1 {
    pub subject_id: String,
    pub attack_count: usize,
    pub localized_change_count: usize,
    pub rejected_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackMetricSummaryV1 {
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackPanelV1 {
    pub panel_version: String,
    pub source_gate_version: String,
    pub source_evidence_version: String,
    pub config: RenderedAttackGateConfigV1,
    pub subject_count: usize,
    pub attack_count: usize,
    pub localized_change_count: usize,
    pub rejected_count: usize,
    pub post_energy_requirement_met_count: usize,
    pub growth_requirement_met_count: usize,
    pub pre_difference_rms: RenderedAttackMetricSummaryV1,
    pub post_difference_rms: RenderedAttackMetricSummaryV1,
    pub post_to_pre_ratio: RenderedAttackMetricSummaryV1,
    pub subjects: Vec<RenderedAttackSubjectSummaryV1>,
    /// Every individual result remains available; the panel is a summary,
    /// never a replacement for the underlying observations.
    pub samples: Vec<RenderedAttackPanelSampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackPanelErrorV1 {
    EmptyPanel,
    EmptySubjectId,
    DuplicateAttackIdentity,
    InvalidAttackTime,
    InvalidGateConfig,
    UnsupportedGateVersion,
    UnsupportedEvidenceVersion,
    MixedGateConfig,
    NonFiniteGateEvidence,
    InconsistentGateEvidence,
}

/// Summarize repeated applications of one exact rendered-attack gate.
///
/// The panel is intentionally descriptive. It retains pass/fail counts and
/// metric distributions but emits no aggregate success, quality, or product-
/// authority judgment.
pub fn summarize_rendered_attack_panel(
    samples: &[RenderedAttackPanelSampleV1],
) -> Result<RenderedAttackPanelV1, RenderedAttackPanelErrorV1> {
    let Some(first) = samples.first() else {
        return Err(RenderedAttackPanelErrorV1::EmptyPanel);
    };
    if first.gate.gate_version != RENDERED_ATTACK_GATE_VERSION {
        return Err(RenderedAttackPanelErrorV1::UnsupportedGateVersion);
    }
    if first.gate.source_evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION {
        return Err(RenderedAttackPanelErrorV1::UnsupportedEvidenceVersion);
    }
    let config = first.gate.config;
    if !valid_gate_config(config) {
        return Err(RenderedAttackPanelErrorV1::InvalidGateConfig);
    }

    let mut identities = BTreeSet::new();
    let mut subjects: BTreeMap<String, RenderedAttackSubjectSummaryV1> = BTreeMap::new();
    let mut pre = Vec::with_capacity(samples.len());
    let mut post = Vec::with_capacity(samples.len());
    let mut ratios = Vec::with_capacity(samples.len());
    let mut localized_change_count = 0usize;
    let mut post_energy_requirement_met_count = 0usize;
    let mut growth_requirement_met_count = 0usize;

    for sample in samples {
        if sample.subject_id.trim().is_empty() {
            return Err(RenderedAttackPanelErrorV1::EmptySubjectId);
        }
        if !sample.attack_time_secs.is_finite() || sample.attack_time_secs < 0.0 {
            return Err(RenderedAttackPanelErrorV1::InvalidAttackTime);
        }
        if !identities.insert((sample.subject_id.clone(), sample.attack_ordinal)) {
            return Err(RenderedAttackPanelErrorV1::DuplicateAttackIdentity);
        }
        if sample.gate.gate_version != RENDERED_ATTACK_GATE_VERSION {
            return Err(RenderedAttackPanelErrorV1::UnsupportedGateVersion);
        }
        if sample.gate.source_evidence_version != RENDERED_ATTACK_EVIDENCE_VERSION {
            return Err(RenderedAttackPanelErrorV1::UnsupportedEvidenceVersion);
        }
        if sample.gate.config != config {
            return Err(RenderedAttackPanelErrorV1::MixedGateConfig);
        }
        if !sample.gate.pre_difference_rms.is_finite()
            || !sample.gate.post_difference_rms.is_finite()
            || !sample.gate.post_to_pre_ratio.is_finite()
        {
            return Err(RenderedAttackPanelErrorV1::NonFiniteGateEvidence);
        }

        // Re-derive the frozen gate semantics instead of trusting booleans in
        // the serialized record. This catches stale/corrupt/hand-built samples.
        let expected_post =
            sample.gate.post_difference_rms >= config.minimum_post_difference_rms;
        let expected_growth =
            sample.gate.post_to_pre_ratio >= config.minimum_post_to_pre_ratio;
        if sample.gate.post_energy_requirement_met != expected_post
            || sample.gate.growth_requirement_met != expected_growth
            || sample.gate.localized_change_detected != (expected_post && expected_growth)
        {
            return Err(RenderedAttackPanelErrorV1::InconsistentGateEvidence);
        }

        pre.push(sample.gate.pre_difference_rms);
        post.push(sample.gate.post_difference_rms);
        ratios.push(sample.gate.post_to_pre_ratio);
        localized_change_count += sample.gate.localized_change_detected as usize;
        post_energy_requirement_met_count += sample.gate.post_energy_requirement_met as usize;
        growth_requirement_met_count += sample.gate.growth_requirement_met as usize;

        let entry = subjects
            .entry(sample.subject_id.clone())
            .or_insert_with(|| RenderedAttackSubjectSummaryV1 {
                subject_id: sample.subject_id.clone(),
                attack_count: 0,
                localized_change_count: 0,
                rejected_count: 0,
            });
        entry.attack_count += 1;
        if sample.gate.localized_change_detected {
            entry.localized_change_count += 1;
        } else {
            entry.rejected_count += 1;
        }
    }

    let attack_count = samples.len();
    Ok(RenderedAttackPanelV1 {
        panel_version: RENDERED_ATTACK_PANEL_VERSION.into(),
        source_gate_version: RENDERED_ATTACK_GATE_VERSION.into(),
        source_evidence_version: RENDERED_ATTACK_EVIDENCE_VERSION.into(),
        config,
        subject_count: subjects.len(),
        attack_count,
        localized_change_count,
        rejected_count: attack_count - localized_change_count,
        post_energy_requirement_met_count,
        growth_requirement_met_count,
        pre_difference_rms: summarize_metric(&pre),
        post_difference_rms: summarize_metric(&post),
        post_to_pre_ratio: summarize_metric(&ratios),
        subjects: subjects.into_values().collect(),
        samples: samples.to_vec(),
    })
}

fn valid_gate_config(config: RenderedAttackGateConfigV1) -> bool {
    config.ratio_floor_rms.is_finite()
        && config.ratio_floor_rms > 0.0
        && config.minimum_post_difference_rms.is_finite()
        && config.minimum_post_difference_rms >= 0.0
        && config.minimum_post_to_pre_ratio.is_finite()
        && config.minimum_post_to_pre_ratio > 1.0
}

fn summarize_metric(values: &[f64]) -> RenderedAttackMetricSummaryV1 {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    RenderedAttackMetricSummaryV1 {
        minimum,
        maximum,
        mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(detected: bool, pre: f64, post: f64) -> RenderedAttackGateResultV1 {
        let config = RenderedAttackGateConfigV1::default();
        let ratio = post / pre.max(config.ratio_floor_rms);
        let post_met = post >= config.minimum_post_difference_rms;
        let growth_met = ratio >= config.minimum_post_to_pre_ratio;
        assert_eq!(detected, post_met && growth_met);
        RenderedAttackGateResultV1 {
            gate_version: RENDERED_ATTACK_GATE_VERSION.into(),
            source_evidence_version: RENDERED_ATTACK_EVIDENCE_VERSION.into(),
            config,
            pre_difference_rms: pre,
            post_difference_rms: post,
            post_to_pre_ratio: ratio,
            post_energy_requirement_met: post_met,
            growth_requirement_met: growth_met,
            localized_change_detected: detected,
        }
    }

    fn sample(
        subject: &str,
        ordinal: usize,
        gate: RenderedAttackGateResultV1,
    ) -> RenderedAttackPanelSampleV1 {
        RenderedAttackPanelSampleV1 {
            subject_id: subject.into(),
            attack_ordinal: ordinal,
            attack_time_secs: 0.5 + ordinal as f32 * 0.1,
            gate,
        }
    }

    #[test]
    fn panel_retains_failures_instead_of_turning_them_into_a_winner() {
        let samples = vec![
            sample("seed-1", 0, gate(true, 1.0e-5, 5.0e-4)),
            sample("seed-1", 1, gate(false, 2.0e-4, 2.5e-4)),
            sample("seed-2", 0, gate(true, 1.0e-6, 3.0e-4)),
        ];
        let panel = summarize_rendered_attack_panel(&samples).unwrap();
        assert_eq!(panel.subject_count, 2);
        assert_eq!(panel.attack_count, 3);
        assert_eq!(panel.localized_change_count, 2);
        assert_eq!(panel.rejected_count, 1);
        assert_eq!(panel.samples.len(), 3);
        assert!(panel.pre_difference_rms.minimum <= panel.pre_difference_rms.mean);
        assert!(panel.pre_difference_rms.mean <= panel.pre_difference_rms.maximum);
        assert_eq!(panel.subjects[0].subject_id, "seed-1");
        assert_eq!(panel.subjects[0].rejected_count, 1);
    }

    #[test]
    fn mixed_gate_configuration_is_rejected() {
        let mut second = gate(true, 1.0e-5, 5.0e-4);
        second.config.minimum_post_to_pre_ratio = 3.0;
        let error = summarize_rendered_attack_panel(&[
            sample("seed-1", 0, gate(true, 1.0e-5, 5.0e-4)),
            sample("seed-2", 0, second),
        ]);
        assert_eq!(error, Err(RenderedAttackPanelErrorV1::MixedGateConfig));
    }

    #[test]
    fn invalid_gate_configuration_is_rejected() {
        let mut invalid = gate(true, 1.0e-5, 5.0e-4);
        invalid.config.minimum_post_to_pre_ratio = 1.0;
        assert_eq!(
            summarize_rendered_attack_panel(&[sample("seed-1", 0, invalid)]),
            Err(RenderedAttackPanelErrorV1::InvalidGateConfig)
        );
    }

    #[test]
    fn inconsistent_serialized_gate_result_is_rejected() {
        let mut forged = gate(true, 1.0e-5, 5.0e-4);
        forged.localized_change_detected = false;
        let error = summarize_rendered_attack_panel(&[sample("seed-1", 0, forged)]);
        assert_eq!(
            error,
            Err(RenderedAttackPanelErrorV1::InconsistentGateEvidence)
        );
    }

    #[test]
    fn duplicate_attack_identity_is_rejected() {
        let one = sample("seed-1", 0, gate(true, 1.0e-5, 5.0e-4));
        assert_eq!(
            summarize_rendered_attack_panel(&[one.clone(), one]),
            Err(RenderedAttackPanelErrorV1::DuplicateAttackIdentity)
        );
    }
}
