// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Authorized Bevy submission path for typed SceneGraphV2.
//!
//! The graph is verified against its canonical cue, opaque permit and
//! Authorized receipt before it can enter the queue. It is re-authorized
//! against the live policy and source clock immediately before ECS creation
//! and on every update. ECS acceptance is not compositor observation.

use super::{
    color_for_role, is_before_expiry, mesh_for_glyph, render_amplitude,
    ActiveScene, ActiveSceneTransition, AdapterError, AdapterSubmissionStatus,
    LatestSceneSubmissionAck, PresentationPolicyState, RenderedSceneEntity,
    SceneClock, SceneElementPresentation, SceneFrameMetadata, SceneGroupId,
};
use bevy::color::Alpha;
use bevy::sprite_render::AlphaMode2d;
use bevy::prelude::*;
use symthaea_soma::embodiment_policy::{
    DenialReason, OutputChannel, PresentationPermit,
};
use symthaea_soma::embodiment_projection::CanonicalCue;
use symthaea_soma::embodiment_receipt::{
    PresentationReceipt, ReceiptPhase,
};
use symthaea_soma::embodiment_scene::{SceneGlyph, SceneMotion, SceneRole};
use symthaea_soma::embodiment_scene_graph::{
    verify_scene_graph_v2, SceneGraphV2, SceneGraphV2Error, SceneRelationKind,
};

#[derive(Debug, Clone)]
pub struct AuthorizedSceneGraphSubmission {
    graph: SceneGraphV2,
    permit: PresentationPermit,
    authorization_receipt: PresentationReceipt,
}

impl AuthorizedSceneGraphSubmission {
    /// Verify the graph, original cue, opaque live permit and Authorized receipt.
    /// This is a local binding check, not a cryptographic signature.
    pub fn new(
        graph: SceneGraphV2,
        cue: &CanonicalCue,
        permit: PresentationPermit,
        authorization_receipt: PresentationReceipt,
    ) -> Result<Self, AdapterError> {
        if authorization_receipt.phase() != ReceiptPhase::Authorized {
            return Err(AdapterError::ReceiptNotAuthorized);
        }
        if permit.channel() != OutputChannel::Visual {
            return Err(AdapterError::WrongOutputChannel);
        }
        verify_scene_graph_v2(&graph, cue, &permit)
            .map_err(AdapterError::SceneGraphVerificationFailed)?;

        if graph.session_id != permit.session_id()
            || graph.clock_domain_id != permit.clock_domain_id()
            || authorization_receipt.decision_clock().clock_domain_id != graph.clock_domain_id
            || authorization_receipt.transition_clock().clock_domain_id != graph.clock_domain_id
        {
            return Err(AdapterError::WrongClockDomain);
        }
        if graph.event_id != permit.event_id()
            || authorization_receipt.event_id() != graph.event_id
        {
            return Err(AdapterError::EventIdentityMismatch);
        }
        if graph.event_sequence != permit.sequence()
            || graph.event_kind != permit.event_kind()
            || graph.semantic_key != permit.semantic_key()
        {
            return Err(AdapterError::FramePermitMismatch);
        }
        if graph.canonical_cue_digest != permit.cue_digest()
            || authorization_receipt.canonical_cue_digest() != graph.canonical_cue_digest
        {
            return Err(AdapterError::CueDigestMismatch);
        }
        if graph.form != permit.form()
            || authorization_receipt.form() != graph.form
        {
            return Err(AdapterError::FormMismatch);
        }
        if graph.channel != permit.channel()
            || authorization_receipt.channel() != graph.channel
            || authorization_receipt.destination() != permit.destination()
        {
            return Err(AdapterError::WrongOutputChannel);
        }
        if graph.privacy_class != permit.privacy_class()
            || authorization_receipt.privacy_class() != graph.privacy_class
        {
            return Err(AdapterError::PrivacyMismatch);
        }
        if graph.destination != permit.destination()
            || authorization_receipt.semantic_key() != graph.semantic_key
        {
            return Err(AdapterError::ReceiptPresentationMismatch);
        }
        if graph.effective_intensity.to_bits() != permit.effective_intensity().to_bits()
            || authorization_receipt.effective_intensity().to_bits()
                != graph.effective_intensity.to_bits()
            || authorization_receipt.motion_mode() != permit.motion_mode()
        {
            return Err(AdapterError::IntensityMismatch);
        }
        if authorization_receipt.event_id() != permit.event_id()
            || authorization_receipt.event_sequence() != permit.sequence()
            || authorization_receipt.event_kind() != permit.event_kind()
            || authorization_receipt.grant_id() != permit.grant_id()
            || authorization_receipt.event_expires_at_monotonic_ns()
                != permit.event_expires_at_monotonic_ns()
            || authorization_receipt.grant_expires_at_monotonic_ns()
                != permit.grant_expires_at_monotonic_ns()
            || authorization_receipt.effective_permit_expires_at_monotonic_ns()
                != permit.expires_at_monotonic_ns()
            || graph.effective_permit_expires_at_monotonic_ns
                != permit.expires_at_monotonic_ns()
        {
            return Err(AdapterError::EffectiveExpiryMismatch);
        }

        Ok(Self { graph, permit, authorization_receipt })
    }

    pub fn graph(&self) -> &SceneGraphV2 { &self.graph }
    pub fn permit(&self) -> &PresentationPermit { &self.permit }
    pub fn authorization_receipt(&self) -> &PresentationReceipt {
        &self.authorization_receipt
    }
}

#[derive(Resource, Default)]
pub struct SceneGraphSubmissionQueue {
    pending: Option<AuthorizedSceneGraphSubmission>,
    revision: u64,
}

impl SceneGraphSubmissionQueue {
    pub fn enqueue(
        &mut self,
        submission: AuthorizedSceneGraphSubmission,
    ) -> Result<u64, AdapterError> {
        self.revision = self.revision.checked_add(1)
            .ok_or(AdapterError::QueueRevisionExhausted)?;
        self.pending = Some(submission);
        Ok(self.revision)
    }

    pub fn clear(&mut self) -> Result<u64, AdapterError> {
        self.revision = self.revision.checked_add(1)
            .ok_or(AdapterError::QueueRevisionExhausted)?;
        self.pending = None;
        Ok(self.revision)
    }

    pub fn revision(&self) -> u64 { self.revision }
}

#[derive(Debug, Clone)]
pub struct SceneGraphSubmissionAck {
    pub status: AdapterSubmissionStatus,
    pub event_id: String,
    pub canonical_cue_digest: String,
    pub grant_id: String,
    pub node_count: usize,
    pub relation_count: usize,
    permit: Option<PresentationPermit>,
}

impl SceneGraphSubmissionAck {
    pub fn accepted_permit(&self) -> Option<&PresentationPermit> {
        self.permit.as_ref()
    }
}

#[derive(Resource, Default)]
pub struct LatestSceneGraphSubmissionAck {
    latest: Option<SceneGraphSubmissionAck>,
}

impl LatestSceneGraphSubmissionAck {
    pub fn latest(&self) -> Option<&SceneGraphSubmissionAck> {
        self.latest.as_ref()
    }
}

#[derive(Resource, Default)]
struct ActiveSceneGraph {
    submission: Option<AuthorizedSceneGraphSubmission>,
    group_id: u64,
}

#[derive(Component)]
struct SceneGraphRelationEntity;

fn render_pending_scene_graph(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    queue: Res<SceneGraphSubmissionQueue>,
    clock: Res<SceneClock>,
    policy: Option<Res<PresentationPolicyState>>,
    mut latest_ack: ResMut<LatestSceneGraphSubmissionAck>,
    mut active_graph: ResMut<ActiveSceneGraph>,
    mut active_scene: ResMut<ActiveScene>,
    mut transition: ResMut<ActiveSceneTransition>,
    existing: Query<(Entity, &SceneGroupId), With<RenderedSceneEntity>>,
    mut last_revision: Local<u64>,
) {
    if queue.revision() == *last_revision {
        return;
    }
    *last_revision = queue.revision();
    latest_ack.latest = None;

    // Graph and legacy flat-scene submissions are mutually exclusive. A new
    // graph request hard-cuts the previous output rather than blending two
    // independently authored scene representations.
    for (entity, _) in &existing {
        commands.entity(entity).despawn();
    }
    active_graph.submission = None;
    active_scene.submission = None;
    transition.source_submission = None;
    transition.plan = None;

    let Some(submission) = queue.pending.as_ref() else { return; };
    let Some(policy) = policy.as_ref() else {
        latest_ack.latest = Some(rejected_graph_ack(
            submission, AdapterError::PolicyUnavailable,
        ));
        return;
    };

    let refreshed = match recheck_graph_submission(
        submission, &clock, clock.is_changed(), policy,
    ) {
        Ok(value) => value,
        Err(error) => {
            latest_ack.latest = Some(rejected_graph_ack(submission, error));
            return;
        }
    };

    let layout = match super::layout_scene_graph_v2(submission.graph()) {
        Ok(value) => value,
        Err(error) => {
            latest_ack.latest = Some(rejected_graph_ack(
                submission, AdapterError::SceneGraphLayoutFailed(error),
            ));
            return;
        }
    };

    let group_id = queue.revision();
    let graph = submission.graph();
    commands.spawn((
        RenderedSceneEntity,
        SceneGroupId(group_id),
        SceneFrameMetadata {
            event_id: graph.event_id.clone(),
            semantic_key: graph.semantic_key.clone(),
            canonical_cue_digest: graph.canonical_cue_digest.clone(),
            accessible_description: graph.accessible_description.clone(),
        },
    ));

    for node in &layout.nodes {
        let amplitude = render_amplitude(node.weight, graph.effective_intensity);
        let base_scale = Vec3::splat(amplitude);
        let base_translation = Vec3::new(node.position.x, node.position.y, 0.0);
        let rotation = Quat::from_rotation_z(node.rotation_radians);
        let mesh = meshes.add(mesh_for_glyph(node.glyph));
        let base_color = color_for_role(node.role);
        let material = materials.add(ColorMaterial {
            color: base_color,
            alpha_mode: AlphaMode2d::Blend,
            ..default()
        });
        commands.spawn((
            Mesh2d(mesh),
            MeshMaterial2d(material),
            Transform {
                translation: base_translation,
                rotation,
                scale: base_scale,
            },
            if amplitude > 0.0 { Visibility::Visible } else { Visibility::Hidden },
            RenderedSceneEntity,
            SceneGroupId(group_id),
            SceneElementPresentation {
                role: node.role,
                glyph: node.glyph,
                motion: node.motion,
                weight: node.weight,
                authorized_intensity: graph.effective_intensity,
                base_scale,
                base_translation,
            },
        ));
    }

    let mut relation_count = 0_usize;
    for relation in &layout.semantic_paths {
        let color = relation_color(relation.kind);
        let material = materials.add(ColorMaterial {
            color,
            alpha_mode: AlphaMode2d::Blend,
            ..default()
        });
        let points = quadratic_points(
            relation.start, relation.control, relation.end, 12,
        );
        for pair in points.windows(2) {
            let start = pair[0];
            let end = pair[1];
            let delta = end - start;
            let length = delta.length();
            if !length.is_finite() || length <= f32::EPSILON {
                continue;
            }
            let mesh = meshes.add(Mesh::from(Rectangle::new(length, 2.0)));
            let transform = Transform {
                translation: Vec3::new((start.x + end.x) * 0.5, (start.y + end.y) * 0.5, -0.1),
                rotation: Quat::from_rotation_z(delta.y.atan2(delta.x)),
                scale: Vec3::ONE,
            };
            commands.spawn((
                Mesh2d(mesh),
                MeshMaterial2d(material.clone()),
                transform,
                Visibility::Visible,
                RenderedSceneEntity,
                SceneGroupId(group_id),
                SceneGraphRelationEntity,
            ));
        }
        relation_count += 1;
    }

    latest_ack.latest = Some(SceneGraphSubmissionAck {
        status: AdapterSubmissionStatus::Accepted,
        event_id: graph.event_id.clone(),
        canonical_cue_digest: graph.canonical_cue_digest.clone(),
        grant_id: submission.permit().grant_id().to_owned(),
        node_count: layout.nodes.len(),
        relation_count,
        permit: Some(refreshed),
    });
    active_graph.submission = Some(submission.clone());
    active_graph.group_id = group_id;
}

fn enforce_active_scene_graph(
    mut commands: Commands,
    clock: Res<SceneClock>,
    policy: Option<Res<PresentationPolicyState>>,
    mut active: ResMut<ActiveSceneGraph>,
    rendered: Query<(Entity, &SceneGroupId), With<RenderedSceneEntity>>,
) {
    let Some(submission) = active.submission.clone() else { return; };
    let valid = policy.as_ref().is_some_and(|state| {
        recheck_graph_submission(&submission, &clock, clock.is_changed(), state).is_ok()
    });
    if valid { return; }

    for (entity, group) in &rendered {
        if group.0 == active.group_id {
            commands.entity(entity).despawn();
        }
    }
    active.submission = None;
}

fn recheck_graph_submission(
    submission: &AuthorizedSceneGraphSubmission,
    clock: &SceneClock,
    clock_fresh: bool,
    policy: &PresentationPolicyState,
) -> Result<PresentationPermit, AdapterError> {
    let graph = submission.graph();
    if !clock_fresh
        || clock.clock_domain_id() != Some(graph.clock_domain_id.as_str())
    {
        return Err(AdapterError::ClockUnavailable);
    }
    let now = clock.now_monotonic_ns().ok_or(AdapterError::ClockUnavailable)?;
    let expires = graph.effective_permit_expires_at_monotonic_ns
        .parse::<u64>().map_err(|_| AdapterError::ClockUnavailable)?;
    if !is_before_expiry(now, expires) {
        return Err(AdapterError::PolicyDenied(
            DenialReason::PermitExpired,
        ));
    }
    let refreshed = policy.policy()
        .recheck_permit(
            submission.permit(),
            &now.to_string(),
            graph.clock_domain_id.as_str(),
        )
        .map_err(AdapterError::PolicyDenied)?;
    if &refreshed != submission.permit() {
        return Err(AdapterError::PermitChangedAfterAuthorization);
    }
    Ok(refreshed)
}

fn rejected_graph_ack(
    submission: &AuthorizedSceneGraphSubmission,
    error: AdapterError,
) -> SceneGraphSubmissionAck {
    SceneGraphSubmissionAck {
        status: AdapterSubmissionStatus::Rejected(error),
        event_id: submission.graph().event_id.clone(),
        canonical_cue_digest: submission.graph().canonical_cue_digest.clone(),
        grant_id: submission.permit().grant_id().to_owned(),
        node_count: 0,
        relation_count: 0,
        permit: None,
    }
}

fn quadratic_points(
    start: Vec2,
    control: Vec2,
    end: Vec2,
    segments: usize,
) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(segments + 1);
    for index in 0..=segments {
        let t = index as f32 / segments as f32;
        let one_minus_t = 1.0 - t;
        let point = start * (one_minus_t * one_minus_t)
            + control * (2.0 * one_minus_t * t)
            + end * (t * t);
        points.push(point);
    }
    points
}

fn relation_color(kind: SceneRelationKind) -> Color {
    match kind {
        SceneRelationKind::AttentionTo => Color::srgb(0.86, 0.74, 0.48),
        SceneRelationKind::HandoffTo => Color::srgb(0.68, 0.78, 0.92),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_path_has_stable_endpoints_and_finite_interior() {
        let points = quadratic_points(
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(10.0, 0.0),
            12,
        );
        assert_eq!(points.len(), 13);
        assert_eq!(points[0], Vec2::ZERO);
        assert_eq!(points[12], Vec2::new(10.0, 0.0));
        assert!(points.iter().all(|point| point.is_finite()));
    }

    #[test]
    fn quadratic_path_is_reproducible() {
        let a = quadratic_points(Vec2::ZERO, Vec2::new(2.0, 2.0), Vec2::X, 8);
        let b = quadratic_points(Vec2::ZERO, Vec2::new(2.0, 2.0), Vec2::X, 8);
        assert_eq!(a, b);
    }

    #[test]
    fn relation_kinds_have_explicit_non_default_rendering_grammar() {
        assert_ne!(
            relation_color(SceneRelationKind::AttentionTo),
            relation_color(SceneRelationKind::HandoffTo),
        );
    }
}
