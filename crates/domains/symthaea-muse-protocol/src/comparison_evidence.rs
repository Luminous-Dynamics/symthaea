// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Platform-neutral V1 wire schema for artifact-bound blind-comparison evidence.
//!
//! This module deliberately contains data shapes only. Playback, blinding,
//! browser callbacks, structural validation/promotion, IndexedDB persistence,
//! consent, and research authority remain owned by their respective layers.

use serde::{Deserialize, Serialize};

use crate::ArtifactIdentity;

pub const ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION: u32 = 1;
pub const TRIAL_INSTANCE_ID_HEX_LEN: usize = 32;
pub const MAX_COMPARISON_NOTE_UTF8_BYTES: usize = 4096;
pub const MAX_ASSIGNMENT_METHOD_UTF8_BYTES: usize = 256;
pub const MAX_ASSIGNMENT_REFERENCE_UTF8_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSideV1 {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealedAssignmentEvidenceV1 {
    pub visible_a_side: EvidenceSideV1,
    pub visible_b_side: EvidenceSideV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MusicalAnchorEvidenceV1 {
    pub bar_index: u32,
    pub beat_offset: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExposurePolicyEvidenceV1 {
    pub minimum_seconds_per_side: f64,
    pub max_contiguous_step_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurationEvidenceV1 {
    pub visible_a_seconds: f64,
    pub visible_b_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListeningExposureEvidenceV1 {
    pub visible_a_auditions: u32,
    pub visible_b_auditions: u32,
}

/// The response the listener actually made while identities were still hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlindHumanComparisonChoiceV1 {
    PreferVisibleA,
    PreferVisibleB,
    NoPreference,
}

/// The same response resolved through the revealed assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedHumanComparisonChoiceV1 {
    PreferSideA,
    PreferSideB,
    NoPreference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanComparisonJudgmentV1 {
    pub blind_choice: BlindHumanComparisonChoiceV1,
    pub resolved_choice: ResolvedHumanComparisonChoiceV1,
    pub listening_exposure: ListeningExposureEvidenceV1,
    pub self_reported_confidence: Option<f32>,
    pub note: String,
}

/// Descriptive provenance for an externally supplied blind assignment.
///
/// This is data about the assignment method. It does not prove randomness,
/// preregistration, allocation concealment, or inferential-study validity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlindAssignmentProvenanceV1 {
    Unspecified,
    ExternallySupplied {
        method: String,
        reference: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonIdentityRelationV1 {
    AlternateRendition,
    SameScoreDifferentComposition,
    DifferentComposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonEvidenceAuthorityV1 {
    HumanSelfReport,
}

/// Restart-stable V1 evidence for one exact artifact-bound blind trial.
///
/// This is an ordinary wire DTO. Deserialization alone does not establish that
/// its contents are structurally valid, authenticated, attentive, randomized,
/// causal, representative, or evidence of musical quality.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBoundBlindComparisonEvidenceV1 {
    pub schema_version: u32,
    pub trial_id: u64,
    pub recorded_at_unix_ms: u64,
    pub side_a_identity: ArtifactIdentity,
    pub side_b_identity: ArtifactIdentity,
    pub revealed_assignment: RevealedAssignmentEvidenceV1,
    pub anchor: MusicalAnchorEvidenceV1,
    pub exposure_policy: ExposurePolicyEvidenceV1,
    pub duration_evidence_at_judgment: DurationEvidenceV1,
    pub judgment: HumanComparisonJudgmentV1,
    pub identity_relation: ComparisonIdentityRelationV1,
    pub assignment_provenance: BlindAssignmentProvenanceV1,
    pub authority: ComparisonEvidenceAuthorityV1,
}

/// Durable storage/export envelope around one V1 comparison record.
///
/// `trial_instance_id` is a namespace token. This DTO does not itself prove
/// canonical syntax, uniqueness, randomness, or producer authenticity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBoundBlindComparisonEnvelopeV1 {
    pub schema_version: u32,
    pub trial_instance_id: String,
    pub evidence: ArtifactBoundBlindComparisonEvidenceV1,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId};

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    fn fixture() -> ArtifactBoundBlindComparisonEnvelopeV1 {
        ArtifactBoundBlindComparisonEnvelopeV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION,
            trial_instance_id: "0123456789abcdef0123456789abcdef".into(),
            evidence: ArtifactBoundBlindComparisonEvidenceV1 {
                schema_version: ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION,
                trial_id: 202,
                recorded_at_unix_ms: 1_800_000_000_000,
                side_a_identity: identity('a', 'b', 'c'),
                side_b_identity: identity('d', 'e', 'f'),
                revealed_assignment: RevealedAssignmentEvidenceV1 {
                    visible_a_side: EvidenceSideV1::B,
                    visible_b_side: EvidenceSideV1::A,
                },
                anchor: MusicalAnchorEvidenceV1 {
                    bar_index: 2,
                    beat_offset: 0.5,
                },
                exposure_policy: ExposurePolicyEvidenceV1 {
                    minimum_seconds_per_side: 0.5,
                    max_contiguous_step_seconds: 0.6,
                },
                duration_evidence_at_judgment: DurationEvidenceV1 {
                    visible_a_seconds: 0.5,
                    visible_b_seconds: 0.5,
                },
                judgment: HumanComparisonJudgmentV1 {
                    blind_choice: BlindHumanComparisonChoiceV1::PreferVisibleA,
                    resolved_choice: ResolvedHumanComparisonChoiceV1::PreferSideB,
                    listening_exposure: ListeningExposureEvidenceV1 {
                        visible_a_auditions: 1,
                        visible_b_auditions: 1,
                    },
                    self_reported_confidence: Some(0.75),
                    note: "blind report".into(),
                },
                identity_relation: ComparisonIdentityRelationV1::DifferentComposition,
                assignment_provenance: BlindAssignmentProvenanceV1::Unspecified,
                authority: ComparisonEvidenceAuthorityV1::HumanSelfReport,
            },
        }
    }

    #[test]
    fn v1_json_shape_matches_the_pre_migration_ui_contract() {
        let encoded = serde_json::to_value(fixture()).unwrap();
        let expected = serde_json::json!({
            "schema_version": 1,
            "trial_instance_id": "0123456789abcdef0123456789abcdef",
            "evidence": {
                "schema_version": 1,
                "trial_id": 202,
                "recorded_at_unix_ms": 1_800_000_000_000_u64,
                "side_a_identity": {
                    "score_content": "a".repeat(64),
                    "composition": "b".repeat(64),
                    "rendition": "c".repeat(64),
                },
                "side_b_identity": {
                    "score_content": "d".repeat(64),
                    "composition": "e".repeat(64),
                    "rendition": "f".repeat(64),
                },
                "revealed_assignment": {
                    "visible_a_side": "b",
                    "visible_b_side": "a",
                },
                "anchor": {
                    "bar_index": 2,
                    "beat_offset": 0.5,
                },
                "exposure_policy": {
                    "minimum_seconds_per_side": 0.5,
                    "max_contiguous_step_seconds": 0.6,
                },
                "duration_evidence_at_judgment": {
                    "visible_a_seconds": 0.5,
                    "visible_b_seconds": 0.5,
                },
                "judgment": {
                    "blind_choice": "prefer_visible_a",
                    "resolved_choice": "prefer_side_b",
                    "listening_exposure": {
                        "visible_a_auditions": 1,
                        "visible_b_auditions": 1,
                    },
                    "self_reported_confidence": 0.75,
                    "note": "blind report",
                },
                "identity_relation": "different_composition",
                "assignment_provenance": {
                    "kind": "unspecified",
                },
                "authority": "human_self_report",
            },
        });
        assert_eq!(encoded, expected);
    }

    #[test]
    fn v1_wire_round_trip_preserves_the_complete_record() {
        let original = fixture();
        let json = serde_json::to_string(&original).unwrap();
        let decoded: ArtifactBoundBlindComparisonEnvelopeV1 =
            serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn assignment_provenance_keeps_the_existing_tagged_shape() {
        let encoded = serde_json::to_value(BlindAssignmentProvenanceV1::ExternallySupplied {
            method: "precomputed-list".into(),
            reference: Some("trial-set-7".into()),
        })
        .unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({
                "kind": "externally_supplied",
                "method": "precomputed-list",
                "reference": "trial-set-7",
            })
        );
    }
}
