// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance binding for rendered-attack replication panels.
//!
//! A numerically valid panel is not sufficient scientific evidence if its
//! subjects can be detached from the exact scores and audio that produced it.
//! This module binds one panel to explicit per-subject score/audio identities
//! plus the renderer/window contract used for every measurement.

use crate::evidence_digest::{
    canonical_json_sha256,
    rendered_attack_panel::{RENDERED_ATTACK_PANEL_VERSION, RenderedAttackPanelV1},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const RENDERED_ATTACK_PROVENANCE_VERSION: &str = "rendered-attack-provenance-v1";
pub const RENDERED_ATTACK_PANEL_BINDING_VERSION: &str = "rendered-attack-panel-binding-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedAttackSubjectProvenanceV1 {
    pub subject_id: String,
    pub seed: u64,
    pub baseline_score_sha256: String,
    pub candidate_score_sha256: String,
    pub baseline_audio_sha256: String,
    pub candidate_audio_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackPanelProvenanceV1 {
    pub provenance_version: String,
    pub renderer_id: String,
    pub sample_rate: u32,
    pub pre_window_seconds: f32,
    pub post_window_seconds: f32,
    pub subjects: Vec<RenderedAttackSubjectProvenanceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundRenderedAttackPanelV1 {
    pub binding_version: String,
    pub panel_sha256: String,
    pub provenance_sha256: String,
    pub binding_sha256: String,
    pub panel: RenderedAttackPanelV1,
    pub provenance: RenderedAttackPanelProvenanceV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackProvenanceErrorV1 {
    UnsupportedPanelVersion,
    UnsupportedProvenanceVersion,
    EmptyRendererId,
    NonCanonicalRendererId,
    InvalidSampleRate,
    InvalidWindowGeometry,
    EmptySubjectId,
    NonCanonicalSubjectId,
    DuplicateSubjectId,
    InvalidSha256,
    SubjectSetMismatch,
    CanonicalizationFailed,
}

pub fn bind_rendered_attack_panel(
    panel: &RenderedAttackPanelV1,
    provenance: &RenderedAttackPanelProvenanceV1,
) -> Result<BoundRenderedAttackPanelV1, RenderedAttackProvenanceErrorV1> {
    if panel.panel_version != RENDERED_ATTACK_PANEL_VERSION {
        return Err(RenderedAttackProvenanceErrorV1::UnsupportedPanelVersion);
    }
    if provenance.provenance_version != RENDERED_ATTACK_PROVENANCE_VERSION {
        return Err(RenderedAttackProvenanceErrorV1::UnsupportedProvenanceVersion);
    }
    if provenance.renderer_id.trim().is_empty() {
        return Err(RenderedAttackProvenanceErrorV1::EmptyRendererId);
    }
    if provenance.renderer_id.trim() != provenance.renderer_id {
        return Err(RenderedAttackProvenanceErrorV1::NonCanonicalRendererId);
    }
    if provenance.sample_rate == 0 {
        return Err(RenderedAttackProvenanceErrorV1::InvalidSampleRate);
    }
    if !provenance.pre_window_seconds.is_finite()
        || !provenance.post_window_seconds.is_finite()
        || provenance.pre_window_seconds <= 0.0
        || provenance.post_window_seconds <= 0.0
    {
        return Err(RenderedAttackProvenanceErrorV1::InvalidWindowGeometry);
    }

    let mut provenance_ids = BTreeSet::new();
    for subject in &provenance.subjects {
        if subject.subject_id.trim().is_empty() {
            return Err(RenderedAttackProvenanceErrorV1::EmptySubjectId);
        }
        if subject.subject_id.trim() != subject.subject_id {
            return Err(RenderedAttackProvenanceErrorV1::NonCanonicalSubjectId);
        }
        if !provenance_ids.insert(subject.subject_id.clone()) {
            return Err(RenderedAttackProvenanceErrorV1::DuplicateSubjectId);
        }
        for digest in [
            &subject.baseline_score_sha256,
            &subject.candidate_score_sha256,
            &subject.baseline_audio_sha256,
            &subject.candidate_audio_sha256,
        ] {
            if !is_lower_hex_sha256(digest) {
                return Err(RenderedAttackProvenanceErrorV1::InvalidSha256);
            }
        }
    }

    let panel_ids: BTreeSet<_> = panel
        .subjects
        .iter()
        .map(|subject| subject.subject_id.clone())
        .collect();
    if panel_ids != provenance_ids {
        return Err(RenderedAttackProvenanceErrorV1::SubjectSetMismatch);
    }

    let panel_sha256 = canonical_json_sha256(panel)
        .map_err(|_| RenderedAttackProvenanceErrorV1::CanonicalizationFailed)?;
    let provenance_sha256 = canonical_json_sha256(provenance)
        .map_err(|_| RenderedAttackProvenanceErrorV1::CanonicalizationFailed)?;
    let binding_sha256 = canonical_json_sha256(&(
        RENDERED_ATTACK_PANEL_BINDING_VERSION,
        &panel_sha256,
        &provenance_sha256,
    ))
    .map_err(|_| RenderedAttackProvenanceErrorV1::CanonicalizationFailed)?;

    Ok(BoundRenderedAttackPanelV1 {
        binding_version: RENDERED_ATTACK_PANEL_BINDING_VERSION.into(),
        panel_sha256,
        provenance_sha256,
        binding_sha256,
        panel: panel.clone(),
        provenance: provenance.clone(),
    })
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::rendered_attack_evidence::RENDERED_ATTACK_EVIDENCE_VERSION;
    use crate::evidence_digest::rendered_attack_gate::{
        RENDERED_ATTACK_GATE_VERSION, RenderedAttackGateConfigV1,
    };
    use crate::evidence_digest::rendered_attack_panel::{
        RenderedAttackMetricSummaryV1, RenderedAttackPanelSampleV1,
        RenderedAttackSubjectSummaryV1,
    };

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn panel() -> RenderedAttackPanelV1 {
        let config = RenderedAttackGateConfigV1::default();
        let sample = RenderedAttackPanelSampleV1 {
            subject_id: "sonata-seed-79".into(),
            attack_ordinal: 0,
            attack_time_secs: 1.0,
            gate: crate::evidence_digest::rendered_attack_gate::RenderedAttackGateResultV1 {
                gate_version: RENDERED_ATTACK_GATE_VERSION.into(),
                source_evidence_version: RENDERED_ATTACK_EVIDENCE_VERSION.into(),
                config,
                pre_difference_rms: 1.0e-5,
                post_difference_rms: 5.0e-4,
                post_to_pre_ratio: 50.0,
                post_energy_requirement_met: true,
                growth_requirement_met: true,
                localized_change_detected: true,
            },
        };
        RenderedAttackPanelV1 {
            panel_version: RENDERED_ATTACK_PANEL_VERSION.into(),
            source_gate_version: RENDERED_ATTACK_GATE_VERSION.into(),
            source_evidence_version: RENDERED_ATTACK_EVIDENCE_VERSION.into(),
            config,
            subject_count: 1,
            attack_count: 1,
            localized_change_count: 1,
            rejected_count: 0,
            post_energy_requirement_met_count: 1,
            growth_requirement_met_count: 1,
            pre_difference_rms: RenderedAttackMetricSummaryV1 {
                minimum: 1.0e-5,
                maximum: 1.0e-5,
                mean: 1.0e-5,
            },
            post_difference_rms: RenderedAttackMetricSummaryV1 {
                minimum: 5.0e-4,
                maximum: 5.0e-4,
                mean: 5.0e-4,
            },
            post_to_pre_ratio: RenderedAttackMetricSummaryV1 {
                minimum: 50.0,
                maximum: 50.0,
                mean: 50.0,
            },
            subjects: vec![RenderedAttackSubjectSummaryV1 {
                subject_id: "sonata-seed-79".into(),
                attack_count: 1,
                localized_change_count: 1,
                rejected_count: 0,
            }],
            samples: vec![sample],
        }
    }

    fn provenance() -> RenderedAttackPanelProvenanceV1 {
        RenderedAttackPanelProvenanceV1 {
            provenance_version: RENDERED_ATTACK_PROVENANCE_VERSION.into(),
            renderer_id: "theory_realize::realize_with_spec/native".into(),
            sample_rate: 44_100,
            pre_window_seconds: 0.005,
            post_window_seconds: 0.050,
            subjects: vec![RenderedAttackSubjectProvenanceV1 {
                subject_id: "sonata-seed-79".into(),
                seed: 79,
                baseline_score_sha256: digest('1'),
                candidate_score_sha256: digest('2'),
                baseline_audio_sha256: digest('3'),
                candidate_audio_sha256: digest('4'),
            }],
        }
    }

    #[test]
    fn binding_commits_panel_and_underlying_subject_provenance() {
        let bound = bind_rendered_attack_panel(&panel(), &provenance()).unwrap();
        assert_eq!(bound.panel_sha256.len(), 64);
        assert_eq!(bound.provenance_sha256.len(), 64);
        assert_eq!(bound.binding_sha256.len(), 64);
        assert_ne!(bound.panel_sha256, bound.provenance_sha256);
    }

    #[test]
    fn subject_set_mismatch_is_rejected() {
        let mut provenance = provenance();
        provenance.subjects[0].subject_id = "sonata-seed-97".into();
        assert_eq!(
            bind_rendered_attack_panel(&panel(), &provenance),
            Err(RenderedAttackProvenanceErrorV1::SubjectSetMismatch)
        );
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let mut provenance = provenance();
        provenance.subjects[0].baseline_audio_sha256 = "ABC".into();
        assert_eq!(
            bind_rendered_attack_panel(&panel(), &provenance),
            Err(RenderedAttackProvenanceErrorV1::InvalidSha256)
        );
    }
}
