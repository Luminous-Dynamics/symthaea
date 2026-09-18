// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired descriptive contrast between introduced-event and real-music control
//! rendered-attack panels.
//!
//! This layer deliberately emits rates and rate differences, not a winner,
//! significance claim, perceptual conclusion, or product-authority verdict.

use crate::evidence_digest::rendered_attack_panel::{
    RenderedAttackPanelV1, summarize_rendered_attack_panel,
};
use crate::evidence_digest::rendered_attack_provenance::{
    BoundRenderedAttackPanelV1, bind_rendered_attack_panel,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RENDERED_ATTACK_CONTRAST_VERSION: &str = "rendered-attack-contrast-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackSubjectContrastV1 {
    pub subject_id: String,
    pub introduced_attack_count: usize,
    pub control_attack_count: usize,
    pub introduced_localized_change_count: usize,
    pub control_localized_change_count: usize,
    pub introduced_detection_rate: f64,
    pub control_detection_rate: f64,
    /// Introduced minus control detection rate for this subject.
    pub detection_rate_difference: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackContrastV1 {
    pub contrast_version: String,
    pub introduced_binding_sha256: String,
    pub control_binding_sha256: String,
    pub shared_provenance_sha256: String,
    pub subject_count: usize,
    pub introduced_attack_count: usize,
    pub control_attack_count: usize,
    pub introduced_localized_change_count: usize,
    pub control_localized_change_count: usize,
    pub introduced_event_detection_rate: f64,
    pub control_event_detection_rate: f64,
    /// Event-weighted introduced minus control detection rate.
    pub event_detection_rate_difference: f64,
    pub introduced_post_energy_rate: f64,
    pub control_post_energy_rate: f64,
    pub post_energy_rate_difference: f64,
    pub introduced_growth_rate: f64,
    pub control_growth_rate: f64,
    pub growth_rate_difference: f64,
    /// Equal-weight mean of each subject's detection rate.
    pub introduced_macro_subject_detection_rate: f64,
    pub control_macro_subject_detection_rate: f64,
    pub macro_subject_detection_rate_difference: f64,
    pub subjects: Vec<RenderedAttackSubjectContrastV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackContrastErrorV1 {
    InvalidIntroducedPanel,
    InvalidControlPanel,
    IntroducedBindingMismatch,
    ControlBindingMismatch,
    ProvenanceMismatch,
    GateConfigMismatch,
    SourceVersionMismatch,
    SubjectSetMismatch,
}

/// Compare two already-bound panels only after revalidating their raw samples
/// and cryptographic bindings.
///
/// The two panels must share the exact same provenance digest, which binds the
/// same subject IDs, seeds, score hashes, audio hashes, renderer, sample rate,
/// and window geometry. This prevents a seemingly favorable treatment/control
/// contrast from comparing different underlying music or render conditions.
pub fn contrast_rendered_attack_panels(
    introduced: &BoundRenderedAttackPanelV1,
    controls: &BoundRenderedAttackPanelV1,
) -> Result<RenderedAttackContrastV1, RenderedAttackContrastErrorV1> {
    validate_bound_panel(introduced, true)?;
    validate_bound_panel(controls, false)?;

    if introduced.provenance_sha256 != controls.provenance_sha256
        || introduced.provenance != controls.provenance
    {
        return Err(RenderedAttackContrastErrorV1::ProvenanceMismatch);
    }
    if introduced.panel.config != controls.panel.config {
        return Err(RenderedAttackContrastErrorV1::GateConfigMismatch);
    }
    if introduced.panel.source_gate_version != controls.panel.source_gate_version
        || introduced.panel.source_evidence_version != controls.panel.source_evidence_version
    {
        return Err(RenderedAttackContrastErrorV1::SourceVersionMismatch);
    }

    let introduced_subjects: BTreeMap<_, _> = introduced
        .panel
        .subjects
        .iter()
        .map(|subject| (subject.subject_id.clone(), subject))
        .collect();
    let control_subjects: BTreeMap<_, _> = controls
        .panel
        .subjects
        .iter()
        .map(|subject| (subject.subject_id.clone(), subject))
        .collect();
    if introduced_subjects.keys().ne(control_subjects.keys()) {
        return Err(RenderedAttackContrastErrorV1::SubjectSetMismatch);
    }

    let mut subjects = Vec::with_capacity(introduced_subjects.len());
    for (subject_id, introduced_subject) in &introduced_subjects {
        let control_subject = control_subjects
            .get(subject_id)
            .expect("subject sets were checked equal");
        let introduced_rate = rate(
            introduced_subject.localized_change_count,
            introduced_subject.attack_count,
        );
        let control_rate = rate(
            control_subject.localized_change_count,
            control_subject.attack_count,
        );
        subjects.push(RenderedAttackSubjectContrastV1 {
            subject_id: subject_id.clone(),
            introduced_attack_count: introduced_subject.attack_count,
            control_attack_count: control_subject.attack_count,
            introduced_localized_change_count: introduced_subject.localized_change_count,
            control_localized_change_count: control_subject.localized_change_count,
            introduced_detection_rate: introduced_rate,
            control_detection_rate: control_rate,
            detection_rate_difference: introduced_rate - control_rate,
        });
    }

    let introduced_event_detection_rate = rate(
        introduced.panel.localized_change_count,
        introduced.panel.attack_count,
    );
    let control_event_detection_rate = rate(
        controls.panel.localized_change_count,
        controls.panel.attack_count,
    );
    let introduced_post_energy_rate = rate(
        introduced.panel.post_energy_requirement_met_count,
        introduced.panel.attack_count,
    );
    let control_post_energy_rate = rate(
        controls.panel.post_energy_requirement_met_count,
        controls.panel.attack_count,
    );
    let introduced_growth_rate = rate(
        introduced.panel.growth_requirement_met_count,
        introduced.panel.attack_count,
    );
    let control_growth_rate = rate(
        controls.panel.growth_requirement_met_count,
        controls.panel.attack_count,
    );
    let introduced_macro_subject_detection_rate =
        subjects.iter().map(|subject| subject.introduced_detection_rate).sum::<f64>()
            / subjects.len() as f64;
    let control_macro_subject_detection_rate =
        subjects.iter().map(|subject| subject.control_detection_rate).sum::<f64>()
            / subjects.len() as f64;

    Ok(RenderedAttackContrastV1 {
        contrast_version: RENDERED_ATTACK_CONTRAST_VERSION.into(),
        introduced_binding_sha256: introduced.binding_sha256.clone(),
        control_binding_sha256: controls.binding_sha256.clone(),
        shared_provenance_sha256: introduced.provenance_sha256.clone(),
        subject_count: subjects.len(),
        introduced_attack_count: introduced.panel.attack_count,
        control_attack_count: controls.panel.attack_count,
        introduced_localized_change_count: introduced.panel.localized_change_count,
        control_localized_change_count: controls.panel.localized_change_count,
        introduced_event_detection_rate,
        control_event_detection_rate,
        event_detection_rate_difference: introduced_event_detection_rate
            - control_event_detection_rate,
        introduced_post_energy_rate,
        control_post_energy_rate,
        post_energy_rate_difference: introduced_post_energy_rate - control_post_energy_rate,
        introduced_growth_rate,
        control_growth_rate,
        growth_rate_difference: introduced_growth_rate - control_growth_rate,
        introduced_macro_subject_detection_rate,
        control_macro_subject_detection_rate,
        macro_subject_detection_rate_difference: introduced_macro_subject_detection_rate
            - control_macro_subject_detection_rate,
        subjects,
    })
}

fn validate_bound_panel(
    bound: &BoundRenderedAttackPanelV1,
    introduced: bool,
) -> Result<(), RenderedAttackContrastErrorV1> {
    let rebuilt_panel = summarize_rendered_attack_panel(&bound.panel.samples).map_err(|_| {
        if introduced {
            RenderedAttackContrastErrorV1::InvalidIntroducedPanel
        } else {
            RenderedAttackContrastErrorV1::InvalidControlPanel
        }
    })?;
    if rebuilt_panel != bound.panel {
        return Err(if introduced {
            RenderedAttackContrastErrorV1::InvalidIntroducedPanel
        } else {
            RenderedAttackContrastErrorV1::InvalidControlPanel
        });
    }

    let rebound = bind_rendered_attack_panel(&bound.panel, &bound.provenance).map_err(|_| {
        if introduced {
            RenderedAttackContrastErrorV1::IntroducedBindingMismatch
        } else {
            RenderedAttackContrastErrorV1::ControlBindingMismatch
        }
    })?;
    if rebound.binding_version != bound.binding_version
        || rebound.panel_sha256 != bound.panel_sha256
        || rebound.provenance_sha256 != bound.provenance_sha256
        || rebound.binding_sha256 != bound.binding_sha256
    {
        return Err(if introduced {
            RenderedAttackContrastErrorV1::IntroducedBindingMismatch
        } else {
            RenderedAttackContrastErrorV1::ControlBindingMismatch
        });
    }
    Ok(())
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    debug_assert!(denominator > 0);
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::rendered_attack_evidence::RENDERED_ATTACK_EVIDENCE_VERSION;
    use crate::evidence_digest::rendered_attack_gate::{
        RENDERED_ATTACK_GATE_VERSION, RenderedAttackGateConfigV1, RenderedAttackGateResultV1,
    };
    use crate::evidence_digest::rendered_attack_panel::{
        RenderedAttackPanelSampleV1, summarize_rendered_attack_panel,
    };
    use crate::evidence_digest::rendered_attack_provenance::{
        RENDERED_ATTACK_PROVENANCE_VERSION, RenderedAttackPanelProvenanceV1,
        RenderedAttackSubjectProvenanceV1,
    };

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn gate(detected: bool) -> RenderedAttackGateResultV1 {
        let config = RenderedAttackGateConfigV1::default();
        let (pre, post) = if detected {
            (1.0e-5, 5.0e-4)
        } else {
            (2.0e-4, 2.5e-4)
        };
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

    fn provenance() -> RenderedAttackPanelProvenanceV1 {
        RenderedAttackPanelProvenanceV1 {
            provenance_version: RENDERED_ATTACK_PROVENANCE_VERSION.into(),
            renderer_id: "native".into(),
            sample_rate: 44_100,
            pre_window_seconds: 0.005,
            post_window_seconds: 0.050,
            subjects: vec![
                RenderedAttackSubjectProvenanceV1 {
                    subject_id: "seed-1".into(),
                    seed: 1,
                    baseline_score_sha256: digest('1'),
                    candidate_score_sha256: digest('2'),
                    baseline_audio_sha256: digest('3'),
                    candidate_audio_sha256: digest('4'),
                },
                RenderedAttackSubjectProvenanceV1 {
                    subject_id: "seed-2".into(),
                    seed: 2,
                    baseline_score_sha256: digest('5'),
                    candidate_score_sha256: digest('6'),
                    baseline_audio_sha256: digest('a'),
                    candidate_audio_sha256: digest('b'),
                },
            ],
        }
    }

    fn bound(pattern: &[(bool, bool)]) -> BoundRenderedAttackPanelV1 {
        let mut samples = Vec::new();
        for (subject_index, detections) in pattern.iter().enumerate() {
            let subject_id = format!("seed-{}", subject_index + 1);
            for (attack_ordinal, detected) in [detections.0, detections.1].into_iter().enumerate() {
                samples.push(RenderedAttackPanelSampleV1 {
                    subject_id: subject_id.clone(),
                    attack_ordinal,
                    attack_time_secs: 1.0 + attack_ordinal as f32,
                    gate: gate(detected),
                });
            }
        }
        let panel = summarize_rendered_attack_panel(&samples).unwrap();
        bind_rendered_attack_panel(&panel, &provenance()).unwrap()
    }

    #[test]
    fn contrast_reports_both_event_weighted_and_equal_subject_rates_without_a_winner() {
        let introduced = bound(&[(true, true), (true, false)]);
        let controls = bound(&[(false, false), (true, false)]);
        let contrast = contrast_rendered_attack_panels(&introduced, &controls).unwrap();
        assert_eq!(contrast.subject_count, 2);
        assert_eq!(contrast.introduced_attack_count, 4);
        assert_eq!(contrast.control_attack_count, 4);
        assert_eq!(contrast.introduced_localized_change_count, 3);
        assert_eq!(contrast.control_localized_change_count, 1);
        assert_eq!(contrast.introduced_event_detection_rate, 0.75);
        assert_eq!(contrast.control_event_detection_rate, 0.25);
        assert_eq!(contrast.event_detection_rate_difference, 0.5);
        assert_eq!(contrast.macro_subject_detection_rate_difference, 0.5);
        assert_eq!(contrast.subjects[0].detection_rate_difference, 1.0);
        assert_eq!(contrast.subjects[1].detection_rate_difference, 0.0);
    }

    #[test]
    fn mismatched_provenance_is_rejected_even_when_panel_shapes_match() {
        let introduced = bound(&[(true, true), (true, false)]);
        let mut controls = bound(&[(false, false), (true, false)]);
        controls.provenance.subjects[0].candidate_audio_sha256 = digest('c');
        let controls = bind_rendered_attack_panel(&controls.panel, &controls.provenance).unwrap();
        assert_eq!(
            contrast_rendered_attack_panels(&introduced, &controls),
            Err(RenderedAttackContrastErrorV1::ProvenanceMismatch)
        );
    }

    #[test]
    fn forged_panel_summary_is_rejected_before_contrast() {
        let introduced = bound(&[(true, true), (true, false)]);
        let mut controls = bound(&[(false, false), (true, false)]);
        controls.panel.localized_change_count += 1;
        assert_eq!(
            contrast_rendered_attack_panels(&introduced, &controls),
            Err(RenderedAttackContrastErrorV1::InvalidControlPanel)
        );
    }
}
