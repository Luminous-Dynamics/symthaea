// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Isolated Bevy renderer adapter for immutable Symthaea Soma scene frames.
//!
//! This crate is intentionally outside the root Cargo workspace. It pins Bevy
//! exactly and does not add a graphics dependency to symthaea-soma.
//!
//! The host must:
//! 1. insert PresentationPolicyState with the live default-deny policy;
//! 2. update SceneClock from the same monotonic clock domain as the cue on
//!    every app update;
//! 3. obtain an Authorized receipt from PresentationReceiptLedger;
//! 4. construct AuthorizedSceneSubmission and enqueue it;
//! 5. record Submitted only from the adapter acknowledgement;
//! 6. use a qualified compositor/device observer for Observed and Stopped.
//!
//! A stale/missing clock, revoked grant, changed policy, expired permit, or
//! reduced-motion change hides the active scene. The adapter never uses Bevy's
//! wall/game clock as a substitute for the source monotonic clock.

use bevy::prelude::*;
use symthaea_soma::embodiment_policy::{
    DenialReason, OutputChannel, PresentationPermit, PresentationPolicy,
};
use symthaea_soma::embodiment_receipt::{
    PresentationReceipt, ReceiptPhase,
};
use symthaea_soma::embodiment_scene::{
    EmbodimentSceneFrame, SceneGlyph, SceneMotion, SceneRole,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    ReceiptNotAuthorized,
    WrongOutputChannel,
    WrongClockDomain,
    EventIdentityMismatch,
    GrantIdentityMismatch,
    CueDigestMismatch,
    FormMismatch,
    MotionModeMismatch,
    IntensityMismatch,
    PrivacyMismatch,
    EventExpiryMismatch,
    GrantExpiryMismatch,
    EffectiveExpiryMismatch,
    ReceiptPresentationMismatch,
    InvalidClockDomain,
    ClockRegression,
    ClockUnavailable,
    PolicyUnavailable,
    PermitChangedAfterAuthorization,
    PolicyDenied(DenialReason),
    QueueRevisionExhausted,
}

#[derive(Debug, Clone)]
pub struct AuthorizedSceneSubmission {
    frame: EmbodimentSceneFrame,
    permit: PresentationPermit,
    authorization_receipt: PresentationReceipt,
}

impl AuthorizedSceneSubmission {
    /// Bind the immutable renderer frame, opaque permit, and recorded final
    /// authorization. This is a local consistency check, not a signature.
    pub fn new(
        frame: EmbodimentSceneFrame,
        permit: PresentationPermit,
        authorization_receipt: PresentationReceipt,
    ) -> Result<Self, AdapterError> {
        if authorization_receipt.phase() != ReceiptPhase::Authorized {
            return Err(AdapterError::ReceiptNotAuthorized);
        }
        if permit.channel() != OutputChannel::Visual {
            return Err(AdapterError::WrongOutputChannel);
        }
        if frame.session_id() != permit.session_id()
            || frame.clock_domain_id() != permit.clock_domain_id()
            || authorization_receipt.decision_clock().clock_domain_id != frame.clock_domain_id()
        {
            return Err(AdapterError::WrongClockDomain);
        }
        if frame.event_id() != permit.event_id()
            || authorization_receipt.event_id() != frame.event_id()
        {
            return Err(AdapterError::EventIdentityMismatch);
        }
        if frame.grant_id() != permit.grant_id()
            || authorization_receipt.grant_id() != frame.grant_id()
        {
            return Err(AdapterError::GrantIdentityMismatch);
        }
        if frame.canonical_cue_digest() != permit.cue_digest()
            || authorization_receipt.canonical_cue_digest() != frame.canonical_cue_digest()
        {
            return Err(AdapterError::CueDigestMismatch);
        }
        if frame.form() != permit.form() {
            return Err(AdapterError::FormMismatch);
        }
        if frame.motion_mode() != permit.motion_mode() {
            return Err(AdapterError::MotionModeMismatch);
        }
        if (frame.effective_intensity() - permit.effective_intensity()).abs() > f32::EPSILON {
            return Err(AdapterError::IntensityMismatch);
        }
        if frame.privacy_class() != permit.privacy_class() {
            return Err(AdapterError::PrivacyMismatch);
        }
        if authorization_receipt.event_sequence() != frame.sequence()
            || authorization_receipt.event_kind() != frame.event_kind()
            || authorization_receipt.form() != frame.form()
            || authorization_receipt.channel() != permit.channel()
            || authorization_receipt.destination() != permit.destination()
            || authorization_receipt.motion_mode() != frame.motion_mode()
            || authorization_receipt.privacy_class() != frame.privacy_class()
            || (authorization_receipt.effective_intensity() - frame.effective_intensity()).abs() > f32::EPSILON
            || authorization_receipt.semantic_key() != frame.semantic_key()
        {
            return Err(AdapterError::ReceiptPresentationMismatch);
        }
        if frame.event_expires_at_monotonic_ns() != permit.event_expires_at_monotonic_ns()
            || authorization_receipt.event_expires_at_monotonic_ns() != frame.event_expires_at_monotonic_ns()
        {
            return Err(AdapterError::EventExpiryMismatch);
        }
        if frame.grant_expires_at_monotonic_ns() != permit.grant_expires_at_monotonic_ns()
            || authorization_receipt.grant_expires_at_monotonic_ns() != frame.grant_expires_at_monotonic_ns()
        {
            return Err(AdapterError::GrantExpiryMismatch);
        }
        if frame.effective_permit_expires_at_monotonic_ns() != permit.expires_at_monotonic_ns()
            || authorization_receipt.effective_permit_expires_at_monotonic_ns()
                != frame.effective_permit_expires_at_monotonic_ns()
        {
            return Err(AdapterError::EffectiveExpiryMismatch);
        }

        Ok(Self { frame, permit, authorization_receipt })
    }

    pub fn frame(&self) -> &EmbodimentSceneFrame { &self.frame }
    pub fn permit(&self) -> &PresentationPermit { &self.permit }
    pub fn authorization_receipt(&self) -> &PresentationReceipt { &self.authorization_receipt }
}

#[derive(Resource, Clone)]
pub struct PresentationPolicyState {
    policy: PresentationPolicy,
}

impl PresentationPolicyState {
    pub fn new(policy: PresentationPolicy) -> Self { Self { policy } }
    pub fn policy(&self) -> &PresentationPolicy { &self.policy }
    pub fn policy_mut(&mut self) -> &mut PresentationPolicy { &mut self.policy }
    pub fn replace(&mut self, policy: PresentationPolicy) { self.policy = policy; }
}

/// Host-supplied monotonic time. This must be refreshed before every Bevy
/// update using the same clock domain as the source event. If it is not
/// refreshed, the renderer hides its scene rather than assuming time.
#[derive(Resource, Debug, Clone, Default)]
pub struct SceneClock {
    clock_domain_id: Option<String>,
    now_monotonic_ns: Option<u64>,
}

impl SceneClock {
    pub fn update(
        &mut self,
        clock_domain_id: impl Into<String>,
        now_monotonic_ns: u64,
    ) -> Result<(), AdapterError> {
        let clock_domain_id = clock_domain_id.into();
        if clock_domain_id.is_empty()
            || clock_domain_id.len() > 128
            || !clock_domain_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || b"._:-".contains(&byte)
            })
        {
            return Err(AdapterError::InvalidClockDomain);
        }
        if self.clock_domain_id.as_deref() == Some(clock_domain_id.as_str())
            && self.now_monotonic_ns.is_some_and(|previous| now_monotonic_ns < previous)
        {
            return Err(AdapterError::ClockRegression);
        }
        self.clock_domain_id = Some(clock_domain_id);
        self.now_monotonic_ns = Some(now_monotonic_ns);
        Ok(())
    }

    pub fn clock_domain_id(&self) -> Option<&str> { self.clock_domain_id.as_deref() }
    pub fn now_monotonic_ns(&self) -> Option<u64> { self.now_monotonic_ns }

    fn permits_frame_clock_id(&self, clock_domain_id: &str) -> bool {
        self.clock_domain_id.as_deref() == Some(clock_domain_id)
    }

    fn permits_frame(&self, frame: &EmbodimentSceneFrame) -> bool {
        if !self.permits_frame_clock_id(frame.clock_domain_id()) {
            return false;
        }
        let Some(now) = self.now_monotonic_ns else { return false; };
        let Ok(expires) = frame.effective_permit_expires_at_monotonic_ns().parse::<u64>() else {
            return false;
        };
        is_before_expiry(now, expires)
    }
}

fn is_before_expiry(now_monotonic_ns: u64, expires_at_monotonic_ns: u64) -> bool {
    now_monotonic_ns < expires_at_monotonic_ns
}

/// Scene weights are output amplitudes, not semantic confidence. Explicit
/// scene weights are already bounded by the policy intensity; an unspecified
/// weight uses the authorized global intensity instead of a renderer constant.
fn render_amplitude(scene_weight: Option<f32>, authorized_intensity: f32) -> f32 {
    let authorized_intensity = authorized_intensity.clamp(0.0, 1.0);
    scene_weight
        .unwrap_or(authorized_intensity)
        .clamp(0.0, authorized_intensity)
}

#[derive(Resource, Default)]
pub struct SceneSubmissionQueue {
    pending: Option<AuthorizedSceneSubmission>,
    revision: u64,
}

impl SceneSubmissionQueue {
    pub fn enqueue(
        &mut self,
        submission: AuthorizedSceneSubmission,
    ) -> Result<u64, AdapterError> {
        self.revision = self.revision.checked_add(1).ok_or(AdapterError::QueueRevisionExhausted)?;
        self.pending = Some(submission);
        Ok(self.revision)
    }

    pub fn clear(&mut self) -> Result<u64, AdapterError> {
        self.revision = self.revision.checked_add(1).ok_or(AdapterError::QueueRevisionExhausted)?;
        self.pending = None;
        Ok(self.revision)
    }

    pub fn revision(&self) -> u64 { self.revision }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterSubmissionStatus {
    Accepted,
    Rejected(AdapterError),
}

/// This acknowledgement means the Bevy adapter accepted and queued the scene
/// command. It does not mean a compositor displayed it or that the user saw it.
#[derive(Debug, Clone)]
pub struct SceneSubmissionAck {
    pub status: AdapterSubmissionStatus,
    pub event_id: String,
    pub canonical_cue_digest: String,
    pub grant_id: String,
    pub element_count: usize,
    permit: Option<PresentationPermit>,
}

impl SceneSubmissionAck {
    pub fn accepted_permit(&self) -> Option<&PresentationPermit> { self.permit.as_ref() }
}

#[derive(Resource, Default)]
pub struct LatestSceneSubmissionAck {
    latest: Option<SceneSubmissionAck>,
}

impl LatestSceneSubmissionAck {
    pub fn latest(&self) -> Option<&SceneSubmissionAck> { self.latest.as_ref() }
}

#[derive(Resource, Default)]
struct ActiveScene {
    submission: Option<AuthorizedSceneSubmission>,
}

#[derive(Component)]
struct RenderedSceneEntity;

#[derive(Component, Clone)]
struct SceneFrameGuard {
    clock_domain_id: String,
    effective_expires_at_monotonic_ns: String,
}

#[derive(Component, Clone)]
pub struct SceneFrameMetadata {
    event_id: String,
    semantic_key: String,
    canonical_cue_digest: String,
    accessible_description: String,
}

impl SceneFrameMetadata {
    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn semantic_key(&self) -> &str { &self.semantic_key }
    pub fn canonical_cue_digest(&self) -> &str { &self.canonical_cue_digest }
    pub fn accessible_description(&self) -> &str { &self.accessible_description }
}

#[derive(Component, Clone)]
pub struct SceneElementPresentation {
    role: SceneRole,
    glyph: SceneGlyph,
    motion: SceneMotion,
    weight: Option<f32>,
    base_scale: Vec3,
    base_translation: Vec3,
}

impl SceneElementPresentation {
    pub fn role(&self) -> SceneRole { self.role }
    pub fn glyph(&self) -> SceneGlyph { self.glyph }
    pub fn motion(&self) -> SceneMotion { self.motion }
    pub fn weight(&self) -> Option<f32> { self.weight }
}

pub struct SymthaeaSceneRendererPlugin;

impl Plugin for SymthaeaSceneRendererPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneClock>()
            .init_resource::<SceneSubmissionQueue>()
            .init_resource::<LatestSceneSubmissionAck>()
            .init_resource::<ActiveScene>()
            .add_systems(Startup, setup_camera)
            .add_systems(
                Update,
                (render_pending_scene, enforce_active_scene, animate_scene_elements).chain(),
            );
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn render_pending_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    queue: Res<SceneSubmissionQueue>,
    clock: Res<SceneClock>,
    policy: Option<Res<PresentationPolicyState>>,
    mut latest_ack: ResMut<LatestSceneSubmissionAck>,
    mut active: ResMut<ActiveScene>,
    existing: Query<Entity, With<RenderedSceneEntity>>,
    mut last_revision: Local<u64>,
) {
    if queue.revision() == *last_revision {
        return;
    }
    *last_revision = queue.revision();

    for entity in &existing {
        commands.entity(entity).despawn();
    }
    active.submission = None;
    latest_ack.latest = None;

    let Some(submission) = queue.pending.as_ref() else { return; };
    let frame = submission.frame();

    let Some(policy) = policy else {
        latest_ack.latest = Some(rejected_ack(submission, AdapterError::PolicyUnavailable));
        return;
    };

    if !clock.is_changed() || !clock.permits_frame(frame) {
        latest_ack.latest = Some(rejected_ack(submission, AdapterError::ClockUnavailable));
        return;
    }
    let Some(now) = clock.now_monotonic_ns() else {
        latest_ack.latest = Some(rejected_ack(submission, AdapterError::ClockUnavailable));
        return;
    };
    let Some(clock_id) = clock.clock_domain_id() else {
        latest_ack.latest = Some(rejected_ack(submission, AdapterError::ClockUnavailable));
        return;
    };

    let refreshed = match policy.policy().recheck_permit(
        submission.permit(),
        &now.to_string(),
        clock_id,
    ) {
        Ok(permit) => permit,
        Err(reason) => {
            latest_ack.latest = Some(rejected_ack(submission, AdapterError::PolicyDenied(reason)));
            return;
        }
    };
    if &refreshed != submission.permit() {
        latest_ack.latest = Some(rejected_ack(submission, AdapterError::PermitChangedAfterAuthorization));
        return;
    }

    let expires = frame.effective_permit_expires_at_monotonic_ns().to_owned();
    let clock_domain_id = frame.clock_domain_id().to_owned();
    commands.spawn((
        RenderedSceneEntity,
        SceneFrameGuard {
            clock_domain_id: clock_domain_id.clone(),
            effective_expires_at_monotonic_ns: expires.clone(),
        },
        SceneFrameMetadata {
            event_id: frame.event_id().to_owned(),
            semantic_key: frame.semantic_key().to_owned(),
            canonical_cue_digest: frame.canonical_cue_digest().to_owned(),
            accessible_description: frame.accessible_description().to_owned(),
        },
    ));

    let count = frame.elements().len();
    for (index, element) in frame.elements().iter().enumerate() {
        let angle = index as f32 * std::f32::consts::TAU / count.max(1) as f32;
        let radius = if element.role == SceneRole::Core { 0.0 } else { 62.0 };
        let base_translation = Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0);
        let amplitude = render_amplitude(element.weight, frame.effective_intensity());
        let base_scale = Vec3::splat(amplitude);
        let rotation = Quat::from_rotation_z(angle);
        let mesh = meshes.add(mesh_for_glyph(element.glyph));
        let material = materials.add(ColorMaterial::from_color(color_for_role(element.role)));
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
            SceneFrameGuard {
                clock_domain_id: clock_domain_id.clone(),
                effective_expires_at_monotonic_ns: expires.clone(),
            },
            SceneElementPresentation {
                role: element.role,
                glyph: element.glyph,
                motion: element.motion,
                weight: element.weight,
                base_scale,
                base_translation,
            },
        ));
    }

    latest_ack.latest = Some(SceneSubmissionAck {
        status: AdapterSubmissionStatus::Accepted,
        event_id: frame.event_id().to_owned(),
        canonical_cue_digest: frame.canonical_cue_digest().to_owned(),
        grant_id: frame.grant_id().to_owned(),
        element_count: count,
        permit: Some(refreshed),
    });
    active.submission = Some(submission.clone());
}

fn enforce_active_scene(
    mut commands: Commands,
    clock: Res<SceneClock>,
    policy: Option<Res<PresentationPolicyState>>,
    mut active: ResMut<ActiveScene>,
    rendered: Query<(Entity, &SceneFrameGuard), With<RenderedSceneEntity>>,
) {
    let Some(submission) = active.submission.as_ref() else { return; };
    let frame = submission.frame();

    let Some(policy) = policy else {
        for (entity, _) in &rendered {
            commands.entity(entity).despawn();
        }
        active.submission = None;
        return;
    };

    let valid_clock = clock.is_changed() && clock.permits_frame(frame);
    let policy_still_allows = match (clock.now_monotonic_ns(), clock.clock_domain_id()) {
        (Some(now), Some(clock_id)) => policy.policy().recheck_permit(
            submission.permit(),
            &now.to_string(),
            clock_id,
        ).is_ok_and(|permit| &permit == submission.permit()),
        _ => false,
    };

    if valid_clock && policy_still_allows {
        return;
    }

    for (entity, guard) in &rendered {
        if guard.clock_domain_id == frame.clock_domain_id() {
            commands.entity(entity).despawn();
        }
    }
    active.submission = None;
}

fn animate_scene_elements(
    time: Res<Time>,
    mut elements: Query<(&SceneElementPresentation, &mut Transform)>,
) {
    let seconds = time.elapsed_secs();
    for (element, mut transform) in &mut elements {
        let phase = seconds * 2.6;
        match element.motion {
            SceneMotion::Static => {
                transform.scale = element.base_scale;
                transform.translation = element.base_translation;
            }
            SceneMotion::Pulse => {
                let scale = 1.0 + 0.06 * phase.sin();
                transform.scale = element.base_scale * scale;
                transform.translation = element.base_translation;
            }
            SceneMotion::Flow => {
                transform.scale = element.base_scale;
                transform.translation = element.base_translation
                    + Vec3::new((phase + element.base_translation.y).sin() * 6.0, 0.0, 0.0);
            }
            SceneMotion::Radiate => {
                let scale = 1.0 + 0.08 * (phase.sin() * 0.5 + 0.5);
                transform.scale = element.base_scale * scale;
                transform.translation = element.base_translation;
            }
        }
    }
}

fn rejected_ack(
    submission: &AuthorizedSceneSubmission,
    reason: AdapterError,
) -> SceneSubmissionAck {
    SceneSubmissionAck {
        status: AdapterSubmissionStatus::Rejected(reason),
        event_id: submission.frame().event_id().to_owned(),
        canonical_cue_digest: submission.frame().canonical_cue_digest().to_owned(),
        grant_id: submission.frame().grant_id().to_owned(),
        element_count: 0,
        permit: None,
    }
}

fn mesh_for_glyph(glyph: SceneGlyph) -> Mesh {
    match glyph {
        SceneGlyph::StarNode => Mesh::from(RegularPolygon::new(20.0, 5)),
        SceneGlyph::Filament | SceneGlyph::PuppetThread | SceneGlyph::Hypha
        | SceneGlyph::ParticleTrail | SceneGlyph::StringLine | SceneGlyph::HearthArc
        | SceneGlyph::RadialLine => Mesh::from(Rectangle::new(44.0, 3.5)),
        SceneGlyph::Orbit | SceneGlyph::Marker | SceneGlyph::LabelAnchor => {
            Mesh::from(Circle::new(16.0))
        }
        SceneGlyph::MarionetteJoint | SceneGlyph::HyphaNode | SceneGlyph::Particle
        | SceneGlyph::Resonator | SceneGlyph::Ember | SceneGlyph::MandalaCenter => {
            Mesh::from(Circle::new(13.0))
        }
        SceneGlyph::Branch => Mesh::from(RegularPolygon::new(18.0, 3)),
    }
}

fn color_for_role(role: SceneRole) -> Color {
    match role {
        SceneRole::Core => Color::srgb(0.24, 0.72, 0.84),
        SceneRole::Connection => Color::srgb(0.50, 0.72, 0.84),
        SceneRole::TurnState => Color::srgb(0.80, 0.84, 0.90),
        SceneRole::Backchannel => Color::srgb(0.62, 0.76, 0.88),
        SceneRole::Attention => Color::srgb(0.86, 0.74, 0.48),
        SceneRole::Expression => Color::srgb(0.72, 0.68, 0.86),
        SceneRole::Alternative => Color::srgb(0.70, 0.64, 0.82),
        // Evidence status is not conveyed as "green = verified"; textual and
        // shape semantics remain authoritative.
        SceneRole::Evidence => Color::srgb(0.75, 0.78, 0.82),
        SceneRole::UserControl => Color::srgb(0.90, 0.72, 0.42),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_clock_rejects_regression_in_same_domain() {
        let mut clock = SceneClock::default();
        clock.update("clock:scene-1", 100).unwrap();
        assert_eq!(clock.update("clock:scene-1", 99), Err(AdapterError::ClockRegression));
    }

    #[test]
    fn scene_clock_denies_exact_effective_expiry() {
        assert!(is_before_expiry(99, 100));
        assert!(!is_before_expiry(100, 100));
        assert!(!is_before_expiry(101, 100));
    }

    #[test]
    fn unweighted_primitives_use_authorized_intensity_not_a_renderer_default() {
        assert!((render_amplitude(None, 0.2) - 0.2).abs() < f32::EPSILON);
        assert!((render_amplitude(None, 0.0) - 0.0).abs() < f32::EPSILON);
        assert!((render_amplitude(None, 1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn explicit_scene_weights_never_exceed_the_authorized_intensity() {
        assert!((render_amplitude(Some(0.8), 0.3) - 0.3).abs() < f32::EPSILON);
        assert!((render_amplitude(Some(0.1), 0.3) - 0.1).abs() < f32::EPSILON);
        assert!((render_amplitude(Some(1.5), 0.3) - 0.3).abs() < f32::EPSILON);
        assert!((render_amplitude(Some(0.5), 0.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn scene_clock_exposes_host_clock_without_substituting_game_time() {
        let mut clock = SceneClock::default();
        clock.update("clock:scene-1", 100).unwrap();
        assert_eq!(clock.clock_domain_id(), Some("clock:scene-1"));
        assert_eq!(clock.now_monotonic_ns(), Some(100));
        assert!(!clock.permits_frame_clock_id("clock:other"));
    }
}
