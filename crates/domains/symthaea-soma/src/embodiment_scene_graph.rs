// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Versioned, renderer-facing scene graph with explicit source-bound relations.
//!
//! Scene schema v1 contains flat glyphs and is preserved for replay. This v2
//! graph gives frame-local nodes stable IDs and separates semantic relations
//! from renderer-only layout attachments. No edge is inferred from array order,
//! proximity, or a renderer's visual composition.

use super::embodiment_event::{EpistemicStatus, EventKind, PrivacyClass, TurnState};
use super::embodiment_policy::{
    canonical_cue_digest, MotionMode, OutputChannel, OutputDestination, PresentationPermit,
};
use super::embodiment_projection::{CanonicalCue, EmbodimentForm, SemanticCue};
use super::embodiment_scene::{
    compile_scene, SceneCompileError, SceneGlyph, SceneMotion, SceneRole,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const SCENE_GRAPH_SCHEMA_V2: &str = "symthaea.scene-graph.v2";
const SCENE_GRAPH_DIGEST_CONTEXT: &str = "symthaea.scene-graph.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LayoutAnchor {
    Core,
    Periphery { slot: u16 },
    AttachedTo { node_id: u16, slot: u16 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneNodeV2 {
    pub id: u16,
    pub role: SceneRole,
    pub glyph: SceneGlyph,
    pub weight: Option<f32>,
    pub motion: SceneMotion,
    pub layout_anchor: LayoutAnchor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneRelationKind {
    AttentionTo,
    HandoffTo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneRelationSourceField {
    #[serde(rename = "attention_focus.target")]
    AttentionFocusTarget,
    #[serde(rename = "turn_state.target")]
    TurnStateTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneRelationV2 {
    pub id: u16,
    pub source_node_id: u16,
    pub target_node_id: u16,
    pub kind: SceneRelationKind,
    pub source_field: SceneRelationSourceField,
}

/// Renderer-only placement hint. It is deliberately not a semantic edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SceneLayoutAttachmentV2 {
    pub source_node_id: u16,
    pub target_node_id: u16,
    pub slot: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneGraphV2 {
    pub schema: String,
    pub session_id: String,
    pub clock_domain_id: String,
    pub event_id: String,
    pub event_sequence: u64,
    pub event_kind: EventKind,
    pub semantic_key: String,
    pub canonical_cue_digest: String,
    pub scene_graph_digest: String,
    pub form: EmbodimentForm,
    pub channel: OutputChannel,
    pub privacy_class: PrivacyClass,
    pub destination: OutputDestination,
    pub epistemic_status: EpistemicStatus,
    pub effective_intensity: f32,
    pub effective_permit_expires_at_monotonic_ns: String,
    pub nodes: Vec<SceneNodeV2>,
    pub relations: Vec<SceneRelationV2>,
    pub layout_attachments: Vec<SceneLayoutAttachmentV2>,
    pub accessible_description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneGraphV2Error {
    SceneCompileRejected(SceneCompileError),
    CanonicalCueDigestMismatch,
    SceneGraphDigestMismatch,
    MetadataMismatch,
    InvalidIntensity,
    MissingCoreNode,
    InvalidNodeSet,
    DuplicateNodeId,
    DuplicateRelationId,
    DuplicateLayoutAttachment,
    DanglingRelationEndpoint,
    DanglingLayoutEndpoint,
    SelfRelation,
    SelfAttachment,
    InvalidCoreAnchor,
    InvalidLayoutAnchor,
    InvalidRelationForSemanticCue,
    MissingRequiredRelation,
    DuplicateSemanticRelation,
    InvalidSemanticTarget,
    TooManyNodes,
    TooManyRelations,
    TooManyLayoutAttachments,
    SerializationFailure,
}

/// Compile one authorized canonical cue to a versioned graph.
///
/// The call delegates the policy/cue binding checks to compile_scene. The new
/// graph adds explicit relation endpoints, while keeping raw semantic targets
/// out of the renderer-facing nodes and relations. A relation's exact source
/// field remains bound by canonical_cue_digest.
pub fn compile_scene_graph_v2(
    cue: &CanonicalCue,
    permit: &PresentationPermit,
) -> Result<SceneGraphV2, SceneGraphV2Error> {
    let frame = compile_scene(cue, permit)
        .map_err(SceneGraphV2Error::SceneCompileRejected)?;
    let (core_glyph, marker_glyph) = glyphs_for_form(permit.form());
    let dynamic = permit.motion_mode() == MotionMode::Dynamic;

    let mut nodes = vec![SceneNodeV2 {
        id: 0,
        role: SceneRole::Core,
        glyph: core_glyph,
        weight: None,
        motion: if dynamic { SceneMotion::Pulse } else { SceneMotion::Static },
        layout_anchor: LayoutAnchor::Core,
    }];
    let mut relations = Vec::new();
    let mut layout_attachments = Vec::new();

    let child_role = match &cue.semantic {
        SemanticCue::Listening { .. } | SemanticCue::Speaking { .. } => None,
        SemanticCue::TurnState { .. } => Some(SceneRole::TurnState),
        SemanticCue::Backchannel { .. } => Some(SceneRole::Backchannel),
        SemanticCue::AttentionFocus { .. } => Some(SceneRole::Attention),
        SemanticCue::ExpressionIntent { .. } => Some(SceneRole::Expression),
        SemanticCue::Uncertainty { .. } => Some(SceneRole::Alternative),
        SemanticCue::EvidenceStatus { .. } => Some(SceneRole::Evidence),
        SemanticCue::UserControl { .. } => Some(SceneRole::UserControl),
    };

    if let Some(role) = child_role {
        let child_id = 1_u16;
        let glyph = if role == SceneRole::Alternative {
            SceneGlyph::Branch
        } else {
            marker_glyph
        };
        nodes.push(SceneNodeV2 {
            id: child_id,
            role,
            glyph,
            weight: None,
            motion: if dynamic { child_motion(&cue.semantic) } else { SceneMotion::Static },
            layout_anchor: LayoutAnchor::AttachedTo {
                node_id: 0,
                slot: 0,
            },
        });
        layout_attachments.push(SceneLayoutAttachmentV2 {
            source_node_id: 0,
            target_node_id: child_id,
            slot: 0,
        });

        match &cue.semantic {
            SemanticCue::AttentionFocus { target, .. } => {
                if target.trim().is_empty() {
                    return Err(SceneGraphV2Error::InvalidSemanticTarget);
                }
                relations.push(SceneRelationV2 {
                    id: 0,
                    source_node_id: 0,
                    target_node_id: child_id,
                    kind: SceneRelationKind::AttentionTo,
                    source_field: SceneRelationSourceField::AttentionFocusTarget,
                });
            }
            SemanticCue::TurnState {
                state: TurnState::Yield,
                target: Some(target),
            } => {
                if target.trim().is_empty() {
                    return Err(SceneGraphV2Error::InvalidSemanticTarget);
                }
                relations.push(SceneRelationV2 {
                    id: 0,
                    source_node_id: 0,
                    target_node_id: child_id,
                    kind: SceneRelationKind::HandoffTo,
                    source_field: SceneRelationSourceField::TurnStateTarget,
                });
            }
            _ => {}
        }
    }

    let mut graph = SceneGraphV2 {
        schema: SCENE_GRAPH_SCHEMA_V2.to_owned(),
        session_id: frame.session_id().to_owned(),
        clock_domain_id: frame.clock_domain_id().to_owned(),
        event_id: frame.event_id().to_owned(),
        event_sequence: frame.sequence(),
        event_kind: frame.event_kind(),
        semantic_key: frame.semantic_key().to_owned(),
        canonical_cue_digest: frame.canonical_cue_digest().to_owned(),
        scene_graph_digest: String::new(),
        form: frame.form(),
        channel: OutputChannel::Visual,
        privacy_class: frame.privacy_class(),
        destination: permit.destination(),
        epistemic_status: frame.epistemic_status(),
        effective_intensity: frame.effective_intensity(),
        effective_permit_expires_at_monotonic_ns: frame
            .effective_permit_expires_at_monotonic_ns()
            .to_owned(),
        nodes,
        relations,
        layout_attachments,
        accessible_description: frame.accessible_description().to_owned(),
    };
    graph.scene_graph_digest = compute_scene_graph_v2_digest(&graph)?;
    verify_scene_graph_v2(&graph, cue, permit)?;
    Ok(graph)
}

/// Recompute the scene-graph digest using a fixed field order and a
/// domain-separated BLAKE3 context. The digest excludes itself.
pub fn compute_scene_graph_v2_digest(
    graph: &SceneGraphV2,
) -> Result<String, SceneGraphV2Error> {
    let mut material = graph.clone();
    material.scene_graph_digest.clear();
    let bytes =
        serde_json::to_vec(&material).map_err(|_| SceneGraphV2Error::SerializationFailure)?;
    let mut hasher = blake3::Hasher::new_derive_key(SCENE_GRAPH_DIGEST_CONTEXT);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Verify structural graph invariants, the exact canonical cue binding, the
/// permit binding, and every semantic relation's source-field provenance.
pub fn verify_scene_graph_v2(
    graph: &SceneGraphV2,
    cue: &CanonicalCue,
    permit: &PresentationPermit,
) -> Result<(), SceneGraphV2Error> {
    let frame = compile_scene(cue, permit)
        .map_err(SceneGraphV2Error::SceneCompileRejected)?;
    let expected_cue_digest =
        canonical_cue_digest(cue).map_err(|_| SceneGraphV2Error::CanonicalCueDigestMismatch)?;

    if graph.canonical_cue_digest != expected_cue_digest {
        return Err(SceneGraphV2Error::CanonicalCueDigestMismatch);
    }
    if graph.schema != SCENE_GRAPH_SCHEMA_V2
        || graph.session_id != cue.session_id
        || graph.clock_domain_id != cue.clock_domain_id
        || graph.event_id != cue.event_id
        || graph.event_sequence != cue.sequence
        || graph.event_kind != cue.event_kind
        || graph.semantic_key != cue.semantic.semantic_key()
        || graph.form != permit.form()
        || graph.channel != OutputChannel::Visual
        || graph.privacy_class != cue.privacy_class
        || graph.destination != permit.destination()
        || graph.epistemic_status != cue.epistemic_status
        || graph.effective_permit_expires_at_monotonic_ns
            != permit.expires_at_monotonic_ns()
        || graph.accessible_description != frame.accessible_description()
        || graph.effective_intensity.to_bits() != permit.effective_intensity().to_bits()
    {
        return Err(SceneGraphV2Error::MetadataMismatch);
    }
    if !graph.effective_intensity.is_finite()
        || !(0.0..=1.0).contains(&graph.effective_intensity)
    {
        return Err(SceneGraphV2Error::InvalidIntensity);
    }
    if graph.nodes.is_empty() || graph.nodes.len() > 64 {
        return Err(SceneGraphV2Error::TooManyNodes);
    }
    if graph.relations.len() > 64 {
        return Err(SceneGraphV2Error::TooManyRelations);
    }
    if graph.layout_attachments.len() > 128 {
        return Err(SceneGraphV2Error::TooManyLayoutAttachments);
    }

    let mut node_ids = HashSet::with_capacity(graph.nodes.len());
    let mut core_count = 0_usize;
    for node in &graph.nodes {
        if !node_ids.insert(node.id) {
            return Err(SceneGraphV2Error::DuplicateNodeId);
        }
        if let Some(weight) = node.weight {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(SceneGraphV2Error::InvalidIntensity);
            }
        }
        if node.role == SceneRole::Core {
            core_count += 1;
            if node.id != 0 || node.layout_anchor != LayoutAnchor::Core {
                return Err(SceneGraphV2Error::InvalidCoreAnchor);
            }
        }
    }
    if core_count != 1 || !node_ids.contains(&0) {
        return Err(SceneGraphV2Error::MissingCoreNode);
    }

    let (expected_core_glyph, expected_marker_glyph) = glyphs_for_form(graph.form);
    let expected_child_role = match &cue.semantic {
        SemanticCue::Listening { .. } | SemanticCue::Speaking { .. } => None,
        SemanticCue::TurnState { .. } => Some(SceneRole::TurnState),
        SemanticCue::Backchannel { .. } => Some(SceneRole::Backchannel),
        SemanticCue::AttentionFocus { .. } => Some(SceneRole::Attention),
        SemanticCue::ExpressionIntent { .. } => Some(SceneRole::Expression),
        SemanticCue::Uncertainty { .. } => Some(SceneRole::Alternative),
        SemanticCue::EvidenceStatus { .. } => Some(SceneRole::Evidence),
        SemanticCue::UserControl { .. } => Some(SceneRole::UserControl),
    };
    let core = graph.nodes.iter().find(|node| node.id == 0)
        .ok_or(SceneGraphV2Error::MissingCoreNode)?;
    if core.glyph != expected_core_glyph {
        return Err(SceneGraphV2Error::InvalidNodeSet);
    }
    match expected_child_role {
        None => {
            if graph.nodes.len() != 1 || !graph.layout_attachments.is_empty() {
                return Err(SceneGraphV2Error::InvalidNodeSet);
            }
        }
        Some(role) => {
            if graph.nodes.len() != 2 {
                return Err(SceneGraphV2Error::InvalidNodeSet);
            }
            let child = graph.nodes.iter().find(|node| node.id == 1)
                .ok_or(SceneGraphV2Error::InvalidNodeSet)?;
            let expected_glyph = if role == SceneRole::Alternative {
                SceneGlyph::Branch
            } else {
                expected_marker_glyph
            };
            if child.role != role
                || child.glyph != expected_glyph
                || child.layout_anchor != LayoutAnchor::AttachedTo { node_id: 0, slot: 0 }
                || graph.layout_attachments.as_slice() != [SceneLayoutAttachmentV2 {
                    source_node_id: 0,
                    target_node_id: 1,
                    slot: 0,
                }]
            {
                return Err(SceneGraphV2Error::InvalidNodeSet);
            }
        }
    }

    let mut relation_ids = HashSet::with_capacity(graph.relations.len());
    for relation in &graph.relations {
        if !relation_ids.insert(relation.id) {
            return Err(SceneGraphV2Error::DuplicateRelationId);
        }
        if relation.source_node_id == relation.target_node_id {
            return Err(SceneGraphV2Error::SelfRelation);
        }
        let source = graph.nodes.iter().find(|node| node.id == relation.source_node_id);
        let target = graph.nodes.iter().find(|node| node.id == relation.target_node_id);
        let (Some(source), Some(target)) = (source, target) else {
            return Err(SceneGraphV2Error::DanglingRelationEndpoint);
        };
        if source.role != SceneRole::Core {
            return Err(SceneGraphV2Error::InvalidRelationForSemanticCue);
        }
        match (relation.kind, relation.source_field, &cue.semantic, target.role) {
            (
                SceneRelationKind::AttentionTo,
                SceneRelationSourceField::AttentionFocusTarget,
                SemanticCue::AttentionFocus { target: value, .. },
                SceneRole::Attention,
            ) if !value.trim().is_empty() => {}
            (
                SceneRelationKind::HandoffTo,
                SceneRelationSourceField::TurnStateTarget,
                SemanticCue::TurnState {
                    state: TurnState::Yield,
                    target: Some(value),
                },
                SceneRole::TurnState,
            ) if !value.trim().is_empty() => {}
            _ => return Err(SceneGraphV2Error::InvalidRelationForSemanticCue),
        }
    }

    let expected_relation_count = match &cue.semantic {
        SemanticCue::AttentionFocus { target, .. } if !target.trim().is_empty() => 1,
        SemanticCue::TurnState {
            state: TurnState::Yield,
            target: Some(target),
        } if !target.trim().is_empty() => 1,
        _ => 0,
    };
    if expected_relation_count == 1 && graph.relations.is_empty() {
        return Err(SceneGraphV2Error::MissingRequiredRelation);
    }
    if expected_relation_count > 1 || graph.relations.len() > expected_relation_count {
        return Err(SceneGraphV2Error::DuplicateSemanticRelation);
    }
    if expected_relation_count == 1
        && (graph.relations[0].id != 0
            || graph.relations[0].source_node_id != 0
            || graph.relations[0].target_node_id != 1)
    {
        return Err(SceneGraphV2Error::InvalidRelationForSemanticCue);
    }

    let mut attachment_keys = HashSet::with_capacity(graph.layout_attachments.len());
    for attachment in &graph.layout_attachments {
        if attachment.source_node_id == attachment.target_node_id {
            return Err(SceneGraphV2Error::SelfAttachment);
        }
        if !node_ids.contains(&attachment.source_node_id)
            || !node_ids.contains(&attachment.target_node_id)
        {
            return Err(SceneGraphV2Error::DanglingLayoutEndpoint);
        }
        if !attachment_keys.insert(*attachment) {
            return Err(SceneGraphV2Error::DuplicateLayoutAttachment);
        }
        if attachment.source_node_id != 0 {
            return Err(SceneGraphV2Error::InvalidLayoutAnchor);
        }
    }
    for node in &graph.nodes {
        if let LayoutAnchor::AttachedTo { node_id, .. } = node.layout_anchor {
            if !node_ids.contains(&node_id) {
                return Err(SceneGraphV2Error::DanglingLayoutEndpoint);
            }
            if node_id == node.id || node_id != 0 {
                return Err(SceneGraphV2Error::InvalidLayoutAnchor);
            }
        }
    }

    if compute_scene_graph_v2_digest(graph)? != graph.scene_graph_digest {
        return Err(SceneGraphV2Error::SceneGraphDigestMismatch);
    }
    Ok(())
}

fn child_motion(semantic: &SemanticCue) -> SceneMotion {
    match semantic {
        SemanticCue::Listening { .. } | SemanticCue::Speaking { .. } => SceneMotion::Static,
        SemanticCue::TurnState { state, .. } => match state {
            TurnState::Wait => SceneMotion::Static,
            TurnState::Respond => SceneMotion::Pulse,
            TurnState::Continue => SceneMotion::Flow,
            TurnState::Yield => SceneMotion::Radiate,
        },
        SemanticCue::Backchannel { .. } | SemanticCue::AttentionFocus { .. }
        | SemanticCue::ExpressionIntent { .. } => SceneMotion::Pulse,
        SemanticCue::Uncertainty { .. } | SemanticCue::EvidenceStatus { .. }
        | SemanticCue::UserControl { .. } => SceneMotion::Static,
    }
}

fn glyphs_for_form(form: EmbodimentForm) -> (SceneGlyph, SceneGlyph) {
    match form {
        EmbodimentForm::LivingConstellation => (SceneGlyph::StarNode, SceneGlyph::Orbit),
        EmbodimentForm::PuppetTheatre => {
            (SceneGlyph::MarionetteJoint, SceneGlyph::LabelAnchor)
        }
        EmbodimentForm::MycelialBeing => (SceneGlyph::HyphaNode, SceneGlyph::Branch),
        EmbodimentForm::Swarm => (SceneGlyph::Particle, SceneGlyph::Marker),
        EmbodimentForm::LivingInstrument => (SceneGlyph::Resonator, SceneGlyph::Marker),
        EmbodimentForm::HearthCompanion => (SceneGlyph::Ember, SceneGlyph::Marker),
        EmbodimentForm::CognitiveMandala => (SceneGlyph::MandalaCenter, SceneGlyph::Marker),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_event::{EventSource, PrivacyClass};
    use crate::embodiment_policy::{ConsentGrant, PresentationPolicy, PresentationRequest};

    const SESSION: &str = "scene-graph-session";
    const CLOCK: &str = "clock:scene-graph-epoch-1";

    fn fixture(
        form: EmbodimentForm,
        privacy: PrivacyClass,
        semantic: SemanticCue,
    ) -> (CanonicalCue, PresentationPermit) {
        let event_kind = match &semantic {
            SemanticCue::Listening { .. } => EventKind::ConversationListening,
            SemanticCue::Speaking { .. } => EventKind::ConversationSpeaking,
            SemanticCue::TurnState { .. } => EventKind::ConversationTurnState,
            SemanticCue::Backchannel { .. } => EventKind::ConversationBackchannel,
            SemanticCue::AttentionFocus { .. } => EventKind::AttentionFocus,
            SemanticCue::ExpressionIntent { .. } => EventKind::ExpressionIntent,
            SemanticCue::Uncertainty { .. } => EventKind::CognitionUncertainty,
            SemanticCue::EvidenceStatus { .. } => EventKind::EvidenceStatus,
            SemanticCue::UserControl { .. } => EventKind::UserControl,
        };
        let cue = CanonicalCue {
            session_id: SESSION.to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            event_id: "event-scene-graph-1".to_owned(),
            sequence: 1,
            event_kind,
            semantic,
            source: EventSource {
                component: "dialogue".to_owned(),
                instance_id: "dialogue-1".to_owned(),
                source_event_id: None,
                build_id: Some("build-1".to_owned()),
            },
            observed_at_monotonic_ns: "100".to_owned(),
            expires_at_monotonic_ns: "9000".to_owned(),
            epistemic_status: EpistemicStatus::Measured,
            confidence: None,
            evidence_refs: Vec::new(),
            privacy_class: privacy,
        };
        let destination = if privacy == PrivacyClass::LocalSensitive {
            OutputDestination::LocalPrivate
        } else {
            OutputDestination::LocalShared
        };
        let mut policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        policy.grant(
            ConsentGrant {
                grant_id: "grant-scene-graph-v1".to_owned(),
                session_id: SESSION.to_owned(),
                clock_domain_id: CLOCK.to_owned(),
                channel: OutputChannel::Visual,
                destination,
                allowed_forms: vec![form],
                expires_at_monotonic_ns: "8000".to_owned(),
                max_intensity: 0.6,
            },
            "150",
            CLOCK,
        ).unwrap();
        policy.set_paused(false);
        let permit = policy.evaluate(
            &cue,
            &PresentationRequest {
                form,
                channel: OutputChannel::Visual,
                destination,
                requested_intensity: 0.9,
            },
            "200",
            CLOCK,
        ).unwrap();
        (cue, permit)
    }

    #[test]
    fn attention_relation_is_compiled_and_preserved_across_all_forms() {
        for form in EmbodimentForm::ALL {
            let (cue, permit) = fixture(
                form,
                PrivacyClass::Public,
                SemanticCue::AttentionFocus {
                    target: "shared-object-1".to_owned(),
                    strength: None,
                },
            );
            let graph = compile_scene_graph_v2(&cue, &permit).unwrap();
            assert_eq!(graph.relations.len(), 1);
            assert_eq!(graph.relations[0].kind, SceneRelationKind::AttentionTo);
            assert_eq!(graph.relations[0].target_node_id, 1);
            verify_scene_graph_v2(&graph, &cue, &permit).unwrap();
        }
    }

    #[test]
    fn yield_relation_is_emitted_only_for_explicit_yield_target() {
        let (cue, permit) = fixture(
            EmbodimentForm::PuppetTheatre,
            PrivacyClass::Public,
            SemanticCue::TurnState {
                state: TurnState::Yield,
                target: Some("participant-2".to_owned()),
            },
        );
        let graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        assert_eq!(graph.relations[0].kind, SceneRelationKind::HandoffTo);
        verify_scene_graph_v2(&graph, &cue, &permit).unwrap();

        let (wait_cue, wait_permit) = fixture(
            EmbodimentForm::PuppetTheatre,
            PrivacyClass::Public,
            SemanticCue::TurnState {
                state: TurnState::Wait,
                target: None,
            },
        );
        let wait_graph = compile_scene_graph_v2(&wait_cue, &wait_permit).unwrap();
        assert!(wait_graph.relations.is_empty());
    }

    #[test]
    fn listening_scene_has_core_without_invented_semantic_edges() {
        let (cue, permit) = fixture(
            EmbodimentForm::CognitiveMandala,
            PrivacyClass::Public,
            SemanticCue::Listening { active: true },
        );
        let graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.relations.is_empty());
        assert!(graph.layout_attachments.is_empty());
    }

    #[test]
    fn relation_kind_cannot_be_changed_without_matching_source_cue() {
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: "target-a".to_owned(),
                strength: None,
            },
        );
        let mut graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        graph.relations[0].kind = SceneRelationKind::HandoffTo;
        graph.relations[0].source_field = SceneRelationSourceField::TurnStateTarget;
        graph.scene_graph_digest = compute_scene_graph_v2_digest(&graph).unwrap();
        assert_eq!(
            verify_scene_graph_v2(&graph, &cue, &permit),
            Err(SceneGraphV2Error::InvalidRelationForSemanticCue)
        );
    }

    #[test]
    fn duplicate_node_ids_are_rejected() {
        let (cue, permit) = fixture(
            EmbodimentForm::Swarm,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: "target-a".to_owned(),
                strength: None,
            },
        );
        let mut graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        graph.nodes[1].id = 0;
        graph.scene_graph_digest = compute_scene_graph_v2_digest(&graph).unwrap();
        assert_eq!(
            verify_scene_graph_v2(&graph, &cue, &permit),
            Err(SceneGraphV2Error::DuplicateNodeId)
        );
    }

    #[test]
    fn dangling_relation_endpoint_is_rejected() {
        let (cue, permit) = fixture(
            EmbodimentForm::MycelialBeing,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: "target-a".to_owned(),
                strength: None,
            },
        );
        let mut graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        graph.relations[0].target_node_id = 42;
        graph.scene_graph_digest = compute_scene_graph_v2_digest(&graph).unwrap();
        assert_eq!(
            verify_scene_graph_v2(&graph, &cue, &permit),
            Err(SceneGraphV2Error::DanglingRelationEndpoint)
        );
    }

    #[test]
    fn semantic_target_text_is_not_copied_into_graph_or_description_when_sensitive() {
        let secret_target = "private-person-very-sensitive";
        let (cue, permit) = fixture(
            EmbodimentForm::HearthCompanion,
            PrivacyClass::LocalSensitive,
            SemanticCue::AttentionFocus {
                target: secret_target.to_owned(),
                strength: None,
            },
        );
        let graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        let serialized = serde_json::to_string(&graph).unwrap();
        assert!(!serialized.contains(secret_target));
        assert!(!graph.accessible_description.contains(secret_target));
        verify_scene_graph_v2(&graph, &cue, &permit).unwrap();
    }

    #[test]
    fn orphaned_child_node_is_rejected_even_when_digest_is_recomputed() {
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: "target-a".to_owned(),
                strength: None,
            },
        );
        let mut graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        graph.nodes.push(SceneNodeV2 {
            id: 2,
            role: SceneRole::Evidence,
            glyph: SceneGlyph::Marker,
            weight: None,
            motion: SceneMotion::Static,
            layout_anchor: LayoutAnchor::Periphery { slot: 1 },
        });
        graph.scene_graph_digest = compute_scene_graph_v2_digest(&graph).unwrap();
        assert_eq!(
            verify_scene_graph_v2(&graph, &cue, &permit),
            Err(SceneGraphV2Error::InvalidNodeSet)
        );
    }

    #[test]
    fn missing_layout_attachment_is_rejected() {
        let (cue, permit) = fixture(
            EmbodimentForm::PuppetTheatre,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: "target-a".to_owned(),
                strength: None,
            },
        );
        let mut graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        graph.layout_attachments.clear();
        graph.scene_graph_digest = compute_scene_graph_v2_digest(&graph).unwrap();
        assert_eq!(
            verify_scene_graph_v2(&graph, &cue, &permit),
            Err(SceneGraphV2Error::InvalidNodeSet)
        );
    }

    #[test]
    fn graph_digest_tampering_is_rejected() {
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::Listening { active: true },
        );
        let mut graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        graph.scene_graph_digest = "0".repeat(64);
        assert_eq!(
            verify_scene_graph_v2(&graph, &cue, &permit),
            Err(SceneGraphV2Error::SceneGraphDigestMismatch)
        );
    }

    #[test]
    fn raw_semantic_target_is_not_exported_for_public_relations_either() {
        let target = "target-string-not-for-renderer";
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: target.to_owned(),
                strength: None,
            },
        );
        let graph = compile_scene_graph_v2(&cue, &permit).unwrap();
        let serialized = serde_json::to_string(&graph).unwrap();
        assert!(!serialized.contains(target));
    }
}
