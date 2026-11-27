// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Renderer-independent continuity planning across semantic updates and forms.
//!
//! This module plans a visual handoff; it does not render or animate it. It
//! preserves the identity of the canonical cue across visual forms and never
//! allows the old or new frame to outlive its own effective presentation
//! permission.

use super::embodiment_policy::MotionMode;
use super::embodiment_projection::EmbodimentForm;
use super::embodiment_scene::EmbodimentSceneFrame;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    InvalidIdentifier,
    InvalidDigest,
    InvalidTimestamp,
    InvalidIntensity,
    InvalidPolicy,
    WrongClockDomain,
    WrongSession,
    FutureTarget,
    TargetExpired,
    SequenceRegression,
    EventContentMismatch,
    ClockRegression,
}

/// Minimal renderer-facing identity and expiry descriptor. It contains no raw
/// semantic payload or evidence-reference identifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneContinuityDescriptor {
    session_id: String,
    clock_domain_id: String,
    event_id: String,
    canonical_cue_digest: String,
    sequence: u64,
    form: EmbodimentForm,
    motion_mode: MotionMode,
    effective_intensity: f32,
    observed_at_monotonic_ns: String,
    effective_expires_at_monotonic_ns: String,
}

impl SceneContinuityDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: impl Into<String>,
        clock_domain_id: impl Into<String>,
        event_id: impl Into<String>,
        canonical_cue_digest: impl Into<String>,
        sequence: u64,
        form: EmbodimentForm,
        motion_mode: MotionMode,
        effective_intensity: f32,
        observed_at_monotonic_ns: impl Into<String>,
        effective_expires_at_monotonic_ns: impl Into<String>,
    ) -> Result<Self, TransitionError> {
        let session_id = session_id.into();
        let clock_domain_id = clock_domain_id.into();
        let event_id = event_id.into();
        let canonical_cue_digest = canonical_cue_digest.into();
        let observed_at_monotonic_ns = observed_at_monotonic_ns.into();
        let effective_expires_at_monotonic_ns = effective_expires_at_monotonic_ns.into();
        validate_identifier(&session_id)?;
        validate_identifier(&clock_domain_id)?;
        validate_identifier(&event_id)?;
        if canonical_cue_digest.len() != 64
            || !canonical_cue_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(TransitionError::InvalidDigest);
        }
        if !effective_intensity.is_finite() || !(0.0..=1.0).contains(&effective_intensity) {
            return Err(TransitionError::InvalidIntensity);
        }
        let observed = parse_monotonic_ns(&observed_at_monotonic_ns)?;
        let expires = parse_monotonic_ns(&effective_expires_at_monotonic_ns)?;
        if expires <= observed {
            return Err(TransitionError::TargetExpired);
        }
        Ok(Self {
            session_id,
            clock_domain_id,
            event_id,
            canonical_cue_digest: canonical_cue_digest.to_ascii_lowercase(),
            sequence,
            form,
            motion_mode,
            effective_intensity,
            observed_at_monotonic_ns,
            effective_expires_at_monotonic_ns,
        })
    }

    pub fn from_frame(frame: &EmbodimentSceneFrame) -> Result<Self, TransitionError> {
        Self::new(
            frame.session_id(),
            frame.clock_domain_id(),
            frame.event_id(),
            frame.canonical_cue_digest(),
            frame.sequence(),
            frame.form(),
            frame.motion_mode(),
            frame.effective_intensity(),
            frame.observed_at_monotonic_ns(),
            frame.effective_permit_expires_at_monotonic_ns(),
        )
    }

    pub fn session_id(&self) -> &str { &self.session_id }
    pub fn clock_domain_id(&self) -> &str { &self.clock_domain_id }
    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn canonical_cue_digest(&self) -> &str { &self.canonical_cue_digest }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn form(&self) -> EmbodimentForm { self.form }
    pub fn motion_mode(&self) -> MotionMode { self.motion_mode }
    pub fn effective_intensity(&self) -> f32 { self.effective_intensity }
    pub fn observed_at_monotonic_ns(&self) -> &str { &self.observed_at_monotonic_ns }
    pub fn effective_expires_at_monotonic_ns(&self) -> &str { &self.effective_expires_at_monotonic_ns }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionPolicy {
    same_form_duration_ns: u64,
    cross_form_duration_ns: u64,
    max_duration_ns: u64,
}

impl TransitionPolicy {
    pub fn new(
        same_form_duration_ns: u64,
        cross_form_duration_ns: u64,
        max_duration_ns: u64,
    ) -> Result<Self, TransitionError> {
        if max_duration_ns == 0
            || same_form_duration_ns > max_duration_ns
            || cross_form_duration_ns > max_duration_ns
        {
            return Err(TransitionError::InvalidPolicy);
        }
        Ok(Self {
            same_form_duration_ns,
            cross_form_duration_ns,
            max_duration_ns,
        })
    }

    pub fn same_form_duration_ns(self) -> u64 { self.same_form_duration_ns }
    pub fn cross_form_duration_ns(self) -> u64 { self.cross_form_duration_ns }
    pub fn max_duration_ns(self) -> u64 { self.max_duration_ns }
}

impl Default for TransitionPolicy {
    fn default() -> Self {
        Self {
            same_form_duration_ns: 90_000_000,
            cross_form_duration_ns: 180_000_000,
            max_duration_ns: 250_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    Initial,
    NoChange,
    HardCut,
    SameFormUpdate,
    CrossFormSameCue,
    CrossFormNewCue,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionWeights {
    pub source_opacity: f32,
    pub target_opacity: f32,
    pub progress: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneTransitionPlan {
    kind: TransitionKind,
    source_event_id: Option<String>,
    target_event_id: String,
    source_digest: Option<String>,
    target_digest: String,
    source_form: Option<EmbodimentForm>,
    target_form: EmbodimentForm,
    clock_domain_id: String,
    started_at_monotonic_ns: u64,
    duration_ns: u64,
    source_expires_at_monotonic_ns: Option<u64>,
    target_expires_at_monotonic_ns: u64,
}

impl SceneTransitionPlan {
    pub fn kind(&self) -> TransitionKind { self.kind }
    pub fn source_event_id(&self) -> Option<&str> { self.source_event_id.as_deref() }
    pub fn target_event_id(&self) -> &str { &self.target_event_id }
    pub fn source_digest(&self) -> Option<&str> { self.source_digest.as_deref() }
    pub fn target_digest(&self) -> &str { &self.target_digest }
    pub fn source_form(&self) -> Option<EmbodimentForm> { self.source_form }
    pub fn target_form(&self) -> EmbodimentForm { self.target_form }
    pub fn duration_ns(&self) -> u64 { self.duration_ns }
    pub fn started_at_monotonic_ns(&self) -> u64 { self.started_at_monotonic_ns }

    /// Return opacity weights at a host-supplied timestamp in the exact scene
    /// clock domain. At exact effective expiry, the target is no longer valid.
    pub fn weights_at(
        &self,
        clock_domain_id: &str,
        now_monotonic_ns: u64,
    ) -> Result<TransitionWeights, TransitionError> {
        if clock_domain_id != self.clock_domain_id {
            return Err(TransitionError::WrongClockDomain);
        }
        if now_monotonic_ns < self.started_at_monotonic_ns {
            return Err(TransitionError::ClockRegression);
        }
        if now_monotonic_ns >= self.target_expires_at_monotonic_ns {
            return Err(TransitionError::TargetExpired);
        }
        if self.kind == TransitionKind::Initial || self.kind == TransitionKind::NoChange
            || self.kind == TransitionKind::HardCut || self.duration_ns == 0
        {
            return Ok(TransitionWeights {
                source_opacity: 0.0,
                target_opacity: 1.0,
                progress: 1.0,
            });
        }
        let elapsed = now_monotonic_ns.saturating_sub(self.started_at_monotonic_ns);
        let progress = (elapsed as f64 / self.duration_ns as f64).clamp(0.0, 1.0) as f32;
        let source_still_permitted = self.source_expires_at_monotonic_ns
            .is_some_and(|expires| now_monotonic_ns < expires);
        let source_opacity = if source_still_permitted { 1.0 - progress } else { 0.0 };
        let target_opacity = if source_still_permitted { progress } else { 1.0 };
        Ok(TransitionWeights {
            source_opacity,
            target_opacity,
            progress,
        })
    }
}

pub fn plan_transition(
    source: Option<&SceneContinuityDescriptor>,
    target: &SceneContinuityDescriptor,
    policy: TransitionPolicy,
    now_clock_domain_id: &str,
    now_monotonic_ns: &str,
) -> Result<SceneTransitionPlan, TransitionError> {
    validate_identifier(now_clock_domain_id)?;
    let now = parse_monotonic_ns(now_monotonic_ns)?;
    if now_clock_domain_id != target.clock_domain_id {
        return Err(TransitionError::WrongClockDomain);
    }
    let target_observed = parse_monotonic_ns(&target.observed_at_monotonic_ns)?;
    let target_expires = parse_monotonic_ns(&target.effective_expires_at_monotonic_ns)?;
    if target_observed > now {
        return Err(TransitionError::FutureTarget);
    }
    if target_expires <= now {
        return Err(TransitionError::TargetExpired);
    }

    let Some(source) = source else {
        return Ok(SceneTransitionPlan {
            kind: TransitionKind::Initial,
            source_event_id: None,
            target_event_id: target.event_id.clone(),
            source_digest: None,
            target_digest: target.canonical_cue_digest.clone(),
            source_form: None,
            target_form: target.form,
            clock_domain_id: target.clock_domain_id.clone(),
            started_at_monotonic_ns: now,
            duration_ns: 0,
            source_expires_at_monotonic_ns: None,
            target_expires_at_monotonic_ns: target_expires,
        });
    };

    if source.session_id != target.session_id {
        return Err(TransitionError::WrongSession);
    }
    if source.clock_domain_id != target.clock_domain_id {
        return Err(TransitionError::WrongClockDomain);
    }
    if source.event_id == target.event_id && source.canonical_cue_digest != target.canonical_cue_digest {
        return Err(TransitionError::EventContentMismatch);
    }
    let same_event = source.event_id == target.event_id;
    if same_event {
        if source.sequence != target.sequence {
            return Err(TransitionError::SequenceRegression);
        }
    } else if target.sequence <= source.sequence {
        return Err(TransitionError::SequenceRegression);
    }

    let source_expires = parse_monotonic_ns(&source.effective_expires_at_monotonic_ns)?;
    let source_expired = source_expires <= now;
    let same_form = source.form == target.form;
    let same_cue = source.canonical_cue_digest == target.canonical_cue_digest;
    let no_change = same_event && same_cue && same_form;
    let requires_hard_cut = source_expired
        || source.motion_mode == MotionMode::Static
        || target.motion_mode == MotionMode::Static
        || source.effective_intensity <= 0.0
        || target.effective_intensity <= 0.0;

    let kind = if no_change {
        TransitionKind::NoChange
    } else if requires_hard_cut {
        TransitionKind::HardCut
    } else if same_form {
        TransitionKind::SameFormUpdate
    } else if same_cue {
        TransitionKind::CrossFormSameCue
    } else {
        TransitionKind::CrossFormNewCue
    };

    let requested_duration = if same_form {
        policy.same_form_duration_ns
    } else {
        policy.cross_form_duration_ns
    };
    let source_remaining = source_expires.saturating_sub(now);
    let target_remaining = target_expires.saturating_sub(now);
    let duration = if matches!(kind, TransitionKind::HardCut | TransitionKind::NoChange) {
        0
    } else {
        requested_duration
            .min(policy.max_duration_ns)
            .min(source_remaining)
            .min(target_remaining)
    };

    Ok(SceneTransitionPlan {
        kind,
        source_event_id: Some(source.event_id.clone()),
        target_event_id: target.event_id.clone(),
        source_digest: Some(source.canonical_cue_digest.clone()),
        target_digest: target.canonical_cue_digest.clone(),
        source_form: Some(source.form),
        target_form: target.form,
        clock_domain_id: target.clock_domain_id.clone(),
        started_at_monotonic_ns: now,
        duration_ns: duration,
        source_expires_at_monotonic_ns: Some(source_expires),
        target_expires_at_monotonic_ns: target_expires,
    })
}

fn validate_identifier(value: &str) -> Result<(), TransitionError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(TransitionError::InvalidIdentifier);
    }
    Ok(())
}

fn parse_monotonic_ns(value: &str) -> Result<u64, TransitionError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(TransitionError::InvalidTimestamp);
    }
    u64::from_str(value).map_err(|_| TransitionError::InvalidTimestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = "transition-session";
    const CLOCK: &str = "clock:transition-epoch-1";
    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn descriptor(
        id: &str,
        sequence: u64,
        digest: &str,
        form: EmbodimentForm,
        motion: MotionMode,
        intensity: f32,
        expires: &str,
    ) -> SceneContinuityDescriptor {
        SceneContinuityDescriptor::new(
            SESSION, CLOCK, id, digest, sequence, form, motion, intensity, "100", expires,
        ).unwrap()
    }

    #[test]
    fn initial_frame_appears_without_a_source_fade() {
        let target = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let plan = plan_transition(None, &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::Initial);
        let weights = plan.weights_at(CLOCK, 200).unwrap();
        assert_eq!(weights.source_opacity, 0.0);
        assert_eq!(weights.target_opacity, 1.0);
    }

    #[test]
    fn same_cue_cross_form_transition_preserves_digest() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::CrossFormSameCue);
        assert_eq!(plan.source_digest(), Some(DIGEST_A));
        assert_eq!(plan.target_digest(), DIGEST_A);
    }

    #[test]
    fn cross_form_new_cue_is_distinguished_from_same_cue_morph() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::MycelialBeing, MotionMode::Dynamic, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::CrossFormNewCue);
    }

    #[test]
    fn same_form_semantic_update_uses_shorter_transition() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::SameFormUpdate);
        assert_eq!(plan.duration_ns(), 90_000_000);
    }

    #[test]
    fn session_and_clock_mismatch_are_rejected() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let wrong_session = SceneContinuityDescriptor::new(
            "other-session", CLOCK, "event-2", DIGEST_B, 2, EmbodimentForm::Swarm, MotionMode::Dynamic, 0.8, "200", "1200",
        ).unwrap();
        assert_eq!(plan_transition(Some(&source), &wrong_session, TransitionPolicy::default(), CLOCK, "250"), Err(TransitionError::WrongSession));
        let wrong_clock = SceneContinuityDescriptor::new(
            SESSION, "clock:other", "event-2", DIGEST_B, 2, EmbodimentForm::Swarm, MotionMode::Dynamic, 0.8, "200", "1200",
        ).unwrap();
        assert_eq!(plan_transition(Some(&source), &wrong_clock, TransitionPolicy::default(), "clock:other", "250"), Err(TransitionError::WrongClockDomain));
    }

    #[test]
    fn same_event_id_cannot_change_content_digest() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-1", 1, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "1200");
        assert_eq!(plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200"), Err(TransitionError::EventContentMismatch));
    }

    #[test]
    fn target_sequence_must_advance_for_a_new_event() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-2", 1, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "1200");
        assert_eq!(plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200"), Err(TransitionError::SequenceRegression));
    }

    #[test]
    fn static_or_zero_intensity_scene_uses_hard_cut() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Static, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::HardCut);
        assert_eq!(plan.duration_ns(), 0);
    }

    #[test]
    fn transition_duration_is_bounded_by_source_expiry() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "250");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.duration_ns(), 50);
    }

    #[test]
    fn transition_weights_are_linear_and_reach_target_before_expiry() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        let start = plan.weights_at(CLOCK, 200).unwrap();
        let middle = plan.weights_at(CLOCK, 290_000_200).unwrap();
        let end = plan.weights_at(CLOCK, 380_000_200).unwrap();
        assert_eq!(start.source_opacity, 1.0);
        assert_eq!(start.target_opacity, 0.0);
        assert!((middle.source_opacity - 0.5).abs() < 0.0001);
        assert!((middle.target_opacity - 0.5).abs() < 0.0001);
        assert_eq!(end.source_opacity, 0.0);
        assert_eq!(end.target_opacity, 1.0);
    }

    #[test]
    fn expired_source_is_never_kept_visible() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "150");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "1200");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::HardCut);
        let weights = plan.weights_at(CLOCK, 200).unwrap();
        assert_eq!(weights.source_opacity, 0.0);
        assert_eq!(weights.target_opacity, 1.0);
    }

    #[test]
    fn target_expiry_and_clock_regression_fail_closed() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let target = descriptor("event-2", 2, DIGEST_B, EmbodimentForm::PuppetTheatre, MotionMode::Dynamic, 0.8, "300");
        let plan = plan_transition(Some(&source), &target, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.weights_at(CLOCK, 199), Err(TransitionError::ClockRegression));
        assert_eq!(plan.weights_at(CLOCK, 300), Err(TransitionError::TargetExpired));
    }

    #[test]
    fn exact_same_frame_is_a_no_change_not_a_fade() {
        let source = descriptor("event-1", 1, DIGEST_A, EmbodimentForm::LivingConstellation, MotionMode::Dynamic, 0.8, "1000");
        let plan = plan_transition(Some(&source), &source, TransitionPolicy::default(), CLOCK, "200").unwrap();
        assert_eq!(plan.kind(), TransitionKind::NoChange);
        assert_eq!(plan.duration_ns(), 0);
    }
}
