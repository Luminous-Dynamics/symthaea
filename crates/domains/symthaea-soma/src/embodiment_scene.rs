// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic, renderer-independent scene compilation for Symthaea.
//!
//! The scene compiler translates an authorized canonical cue into declarative
//! visual primitives. It has no Bevy, GPU, windowing, sensor, audio, network,
//! or actuator dependency. A later renderer may animate these primitives but
//! must not rewrite their semantic key, provenance, or accessible description.

use super::embodiment_event::{EventSource, EpistemicStatus, EventKind, PrivacyClass, TurnState};
use super::embodiment_policy::{canonical_cue_digest, MotionMode, OutputChannel, OutputDestination, PresentationPermit};
use super::embodiment_projection::{CanonicalCue, EmbodimentForm, SemanticCue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneRole {
    Core,
    Connection,
    TurnState,
    Backchannel,
    Attention,
    Expression,
    Alternative,
    Evidence,
    UserControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneGlyph {
    StarNode,
    Filament,
    Orbit,
    MarionetteJoint,
    PuppetThread,
    HyphaNode,
    Hypha,
    Particle,
    ParticleTrail,
    Resonator,
    StringLine,
    Ember,
    HearthArc,
    MandalaCenter,
    RadialLine,
    Marker,
    Branch,
    LabelAnchor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneMotion {
    Static,
    Pulse,
    Flow,
    Radiate,
}

/// A declarative shape hint. Weight is None when the source supplied no
/// calibrated/authorized amplitude; renderers must not invent confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneElement {
    pub role: SceneRole,
    pub glyph: SceneGlyph,
    pub weight: Option<f32>,
    pub motion: SceneMotion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentSceneFrame {
    pub schema: String,
    pub session_id: String,
    pub clock_domain_id: String,
    pub event_id: String,
    pub canonical_cue_digest: String,
    pub grant_id: String,
    pub sequence: u64,
    pub event_kind: EventKind,
    pub semantic_key: String,
    pub form: EmbodimentForm,
    pub motion_mode: MotionMode,
    pub epistemic_status: EpistemicStatus,
    pub privacy_class: PrivacyClass,
    pub source: EventSource,
    pub observed_at_monotonic_ns: String,
    pub expires_at_monotonic_ns: String,
    /// Privacy-aware, non-colour-only textual equivalent of the essential cue.
    pub accessible_description: String,
    pub elements: Vec<SceneElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneCompileError {
    WrongOutputChannel,
    WrongSession,
    WrongClockDomain,
    WrongEventIdentity,
    WrongSequence,
    WrongEventKind,
    WrongSemanticKey,
    WrongSource,
    WrongPrivacyClass,
    WrongEventExpiry,
    CueContentMismatch,
    RestrictedPrivacy,
    SensitiveDestinationNotPrivate,
    InvalidPermitIntensity,
}

/// Compile an authorized visual permit and its canonical cue into a
/// deterministic scene frame. This function does not perform the final policy
/// recheck: the caller must recheck immediately before renderer submission and
/// record the Authorized and Submitted receipt states. The returned frame is
/// data, not proof that pixels were displayed.
pub fn compile_scene(
    cue: &CanonicalCue,
    permit: &PresentationPermit,
) -> Result<EmbodimentSceneFrame, SceneCompileError> {
    if permit.channel() != OutputChannel::Visual {
        return Err(SceneCompileError::WrongOutputChannel);
    }
    if cue.session_id != permit.session_id() {
        return Err(SceneCompileError::WrongSession);
    }
    if cue.clock_domain_id != permit.clock_domain_id() {
        return Err(SceneCompileError::WrongClockDomain);
    }
    if cue.event_id != permit.event_id() {
        return Err(SceneCompileError::WrongEventIdentity);
    }
    if cue.sequence != permit.sequence() {
        return Err(SceneCompileError::WrongSequence);
    }
    if cue.event_kind != permit.event_kind() {
        return Err(SceneCompileError::WrongEventKind);
    }
    if cue.semantic.semantic_key() != permit.semantic_key() {
        return Err(SceneCompileError::WrongSemanticKey);
    }
    let cue_digest = canonical_cue_digest(cue)
        .map_err(|_| SceneCompileError::CueContentMismatch)?;
    if cue_digest != permit.cue_digest() {
        return Err(SceneCompileError::CueContentMismatch);
    }
    if cue.source != *permit.source() {
        return Err(SceneCompileError::WrongSource);
    }
    if cue.privacy_class != permit.privacy_class() {
        return Err(SceneCompileError::WrongPrivacyClass);
    }
    if cue.expires_at_monotonic_ns != permit.event_expires_at_monotonic_ns() {
        return Err(SceneCompileError::WrongEventExpiry);
    }
    if cue.privacy_class == PrivacyClass::Restricted {
        return Err(SceneCompileError::RestrictedPrivacy);
    }
    if cue.privacy_class == PrivacyClass::LocalSensitive
        && permit.destination() != OutputDestination::LocalPrivate
    {
        return Err(SceneCompileError::SensitiveDestinationNotPrivate);
    }
    let intensity = permit.effective_intensity();
    if !intensity.is_finite() || !(0.0..=1.0).contains(&intensity) {
        return Err(SceneCompileError::InvalidPermitIntensity);
    }

    let dynamic = permit.motion_mode() == MotionMode::Dynamic;
    let (core, connection, marker) = glyphs_for_form(permit.form());
    let mut elements = Vec::with_capacity(3);

    match &cue.semantic {
        SemanticCue::Listening { active } => {
            push_element(
                &mut elements, dynamic, SceneRole::Core, core,
                Some(if *active { intensity * 0.25 } else { 0.08 }),
                SceneMotion::Pulse,
            );
            if *active {
                push_element(&mut elements, dynamic, SceneRole::Connection, connection, None, SceneMotion::Flow);
            }
        }
        SemanticCue::Speaking { active, .. } => {
            push_element(
                &mut elements, dynamic, SceneRole::Core, core,
                Some(if *active { intensity } else { 0.08 }),
                SceneMotion::Pulse,
            );
            if *active {
                push_element(&mut elements, dynamic, SceneRole::Connection, connection, None, SceneMotion::Flow);
            }
        }
        SemanticCue::TurnState { state, .. } => match state {
            TurnState::Wait => {
                push_element(&mut elements, dynamic, SceneRole::TurnState, marker, Some(intensity * 0.15), SceneMotion::Static);
            }
            TurnState::Respond => {
                push_element(&mut elements, dynamic, SceneRole::TurnState, marker, Some(intensity), SceneMotion::Pulse);
                push_element(&mut elements, dynamic, SceneRole::Core, core, Some(intensity * 0.8), SceneMotion::Pulse);
            }
            TurnState::Continue => {
                push_element(&mut elements, dynamic, SceneRole::TurnState, connection, Some(intensity * 0.7), SceneMotion::Flow);
                push_element(&mut elements, dynamic, SceneRole::Core, core, Some(intensity * 0.5), SceneMotion::Pulse);
            }
            TurnState::Yield => {
                push_element(&mut elements, dynamic, SceneRole::TurnState, connection, Some(intensity * 0.7), SceneMotion::Radiate);
                push_element(&mut elements, dynamic, SceneRole::Attention, marker, None, SceneMotion::Radiate);
            }
        },
        SemanticCue::Backchannel { intensity: signal_intensity, .. } => {
            push_element(&mut elements, dynamic, SceneRole::Backchannel, marker, Some(intensity.min(*signal_intensity)), SceneMotion::Pulse);
        }
        SemanticCue::AttentionFocus { strength, .. } => {
            push_element(&mut elements, dynamic, SceneRole::Attention, marker, strength.map(|value| value * intensity), SceneMotion::Pulse);
            push_element(&mut elements, dynamic, SceneRole::Connection, connection, None, SceneMotion::Flow);
        }
        SemanticCue::ExpressionIntent { intensity: expression_intensity, .. } => {
            push_element(&mut elements, dynamic, SceneRole::Expression, marker, Some(intensity.min(*expression_intensity)), SceneMotion::Pulse);
        }
        SemanticCue::Uncertainty { .. } => {
            // Branches communicate alternatives, not a probability distribution.
            push_element(&mut elements, dynamic, SceneRole::Alternative, SceneGlyph::Branch, None, SceneMotion::Static);
            push_element(&mut elements, dynamic, SceneRole::Core, core, None, SceneMotion::Static);
        }
        SemanticCue::EvidenceStatus { .. } => {
            // The status is communicated by text/shape; colour or glow is not proof.
            push_element(&mut elements, dynamic, SceneRole::Evidence, marker, None, SceneMotion::Static);
            push_element(&mut elements, dynamic, SceneRole::Core, core, None, SceneMotion::Static);
        }
        SemanticCue::UserControl { .. } => {
            push_element(&mut elements, dynamic, SceneRole::UserControl, marker, None, SceneMotion::Static);
        }
    }

    if elements.is_empty() {
        push_element(&mut elements, dynamic, SceneRole::Core, core, None, SceneMotion::Static);
    }

    Ok(EmbodimentSceneFrame {
        schema: "symthaea.embodiment-scene.v1".to_owned(),
        session_id: cue.session_id.clone(),
        clock_domain_id: cue.clock_domain_id.clone(),
        event_id: cue.event_id.clone(),
        canonical_cue_digest: cue_digest,
        grant_id: permit.grant_id().to_owned(),
        sequence: cue.sequence,
        event_kind: cue.event_kind,
        semantic_key: cue.semantic.semantic_key().to_owned(),
        form: permit.form(),
        motion_mode: permit.motion_mode(),
        epistemic_status: cue.epistemic_status,
        privacy_class: cue.privacy_class,
        source: cue.source.clone(),
        observed_at_monotonic_ns: cue.observed_at_monotonic_ns.clone(),
        expires_at_monotonic_ns: cue.expires_at_monotonic_ns.clone(),
        accessible_description: cue.accessible_description(),
        elements,
    })
}

fn push_element(
    elements: &mut Vec<SceneElement>,
    dynamic: bool,
    role: SceneRole,
    glyph: SceneGlyph,
    weight: Option<f32>,
    animated_motion: SceneMotion,
) {
    elements.push(SceneElement {
        role,
        glyph,
        weight: weight.map(|value| value.clamp(0.0, 1.0)),
        motion: if dynamic { animated_motion } else { SceneMotion::Static },
    });
}

fn glyphs_for_form(form: EmbodimentForm) -> (SceneGlyph, SceneGlyph, SceneGlyph) {
    match form {
        EmbodimentForm::LivingConstellation => (SceneGlyph::StarNode, SceneGlyph::Filament, SceneGlyph::Orbit),
        EmbodimentForm::PuppetTheatre => (SceneGlyph::MarionetteJoint, SceneGlyph::PuppetThread, SceneGlyph::LabelAnchor),
        EmbodimentForm::MycelialBeing => (SceneGlyph::HyphaNode, SceneGlyph::Hypha, SceneGlyph::Branch),
        EmbodimentForm::Swarm => (SceneGlyph::Particle, SceneGlyph::ParticleTrail, SceneGlyph::Marker),
        EmbodimentForm::LivingInstrument => (SceneGlyph::Resonator, SceneGlyph::StringLine, SceneGlyph::Marker),
        EmbodimentForm::HearthCompanion => (SceneGlyph::Ember, SceneGlyph::HearthArc, SceneGlyph::Marker),
        EmbodimentForm::CognitiveMandala => (SceneGlyph::MandalaCenter, SceneGlyph::RadialLine, SceneGlyph::Marker),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_event::{EvidenceState, EventSource};
    use crate::embodiment_policy::{ConsentGrant, PresentationPolicy, PresentationRequest};

    const SESSION: &str = "scene-session";
    const CLOCK: &str = "clock:scene-epoch-1";

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
        let canonical = CanonicalCue {
            session_id: SESSION.to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            event_id: "event-1".to_owned(),
            sequence: 0,
            event_kind,
            semantic: semantic.clone(),
            source: EventSource {
                component: "dialogue".to_owned(),
                instance_id: "dialogue-1".to_owned(),
                source_event_id: None,
                build_id: Some("build-1".to_owned()),
            },
            observed_at_monotonic_ns: "100".to_owned(),
            expires_at_monotonic_ns: "900".to_owned(),
            epistemic_status: EpistemicStatus::Measured,
            confidence: None,
            privacy_class: privacy,
        };
        let mut policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        policy.grant(
            ConsentGrant {
                grant_id: "grant-v1".to_owned(),
                session_id: SESSION.to_owned(),
                clock_domain_id: CLOCK.to_owned(),
                channel: OutputChannel::Visual,
                destination: if privacy == PrivacyClass::LocalSensitive {
                    OutputDestination::LocalPrivate
                } else {
                    OutputDestination::LocalShared
                },
                allowed_forms: vec![form],
                expires_at_monotonic_ns: "800".to_owned(),
                max_intensity: 0.6,
            },
            "150",
            CLOCK,
        ).unwrap();
        policy.set_paused(false);
        let permit = policy.evaluate(
            &canonical,
            &PresentationRequest {
                form,
                channel: OutputChannel::Visual,
                destination: if privacy == PrivacyClass::LocalSensitive {
                    OutputDestination::LocalPrivate
                } else {
                    OutputDestination::LocalShared
                },
                requested_intensity: 0.9,
            },
            "200",
            CLOCK,
        ).unwrap();
        (canonical, permit)
    }

    #[test]
    fn scene_compilation_is_deterministic_for_identical_inputs() {
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::Listening { active: true },
        );
        assert_eq!(compile_scene(&cue, &permit).unwrap(), compile_scene(&cue, &permit).unwrap());
    }

    #[test]
    fn all_seven_forms_preserve_event_identity_and_semantic_key() {
        let semantic = SemanticCue::Listening { active: true };
        let frames: Vec<_> = EmbodimentForm::ALL.iter().map(|form| {
            let (canonical, permit) = fixture(*form, PrivacyClass::Public, semantic.clone());
            compile_scene(&canonical, &permit).unwrap()
        }).collect();
        assert_eq!(frames.len(), 7);
        for frame in &frames {
            assert_eq!(frame.event_id, "event-1");
            assert_eq!(frame.semantic_key, "conversation.listening.active");
            assert_eq!(frame.sequence, 0);
            assert_eq!(frame.privacy_class, PrivacyClass::Public);
        }
        assert_eq!(frames.iter().map(|frame| frame.elements[0].glyph).collect::<std::collections::HashSet<_>>().len(), 7);
    }

    #[test]
    fn reduced_motion_forces_all_scene_elements_static() {
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::Speaking { active: true, segment_id: "segment-1".to_owned() },
        );
        assert_eq!(permit.motion_mode(), MotionMode::Static);
        let frame = compile_scene(&cue, &permit).unwrap();
        assert!(frame.elements.iter().all(|element| element.motion == SceneMotion::Static));
    }

    #[test]
    fn local_sensitive_accessibility_text_is_redacted() {
        let (cue, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::LocalSensitive,
            SemanticCue::AttentionFocus { target: "private-person".to_owned(), strength: Some(0.9) },
        );
        let frame = compile_scene(&cue, &permit).unwrap();
        assert!(!frame.accessible_description.contains("private-person"));
        assert!(frame.accessible_description.contains("Sensitive interaction cue"));
    }

    #[test]
    fn uncertainty_is_a_branch_not_a_probability_gauge() {
        let (cue, permit) = fixture(
            EmbodimentForm::MycelialBeing,
            PrivacyClass::Public,
            SemanticCue::Uncertainty {
                subject: "hypothesis-A".to_owned(),
                level: 0.99,
                calibration_ref: None,
            },
        );
        let frame = compile_scene(&cue, &permit).unwrap();
        assert!(frame.elements.iter().any(|element| element.role == SceneRole::Alternative && element.glyph == SceneGlyph::Branch && element.weight.is_none()));
    }

    #[test]
    fn evidence_status_marker_does_not_claim_verification() {
        let (cue, permit) = fixture(
            EmbodimentForm::CognitiveMandala,
            PrivacyClass::Public,
            SemanticCue::EvidenceStatus {
                subject_id: "record-1".to_owned(),
                status: EvidenceState::Supported,
                verifier_receipt_ref: Some("receipt:1".to_owned()),
            },
        );
        let frame = compile_scene(&cue, &permit).unwrap();
        assert!(frame.elements.iter().any(|element| element.role == SceneRole::Evidence && element.weight.is_none()));
        assert!(frame.accessible_description.contains("Evidence status reported"));
        assert!(frame.accessible_description.contains("verifier receipt reference is present"));
    }

    #[test]
    fn payload_mutation_with_the_same_semantic_key_is_rejected() {
        let (canonical, permit) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::AttentionFocus {
                target: "original-target".to_owned(),
                strength: Some(0.4),
            },
        );
        let mut altered = canonical.clone();
        if let SemanticCue::AttentionFocus { target, .. } = &mut altered.semantic {
            *target = "tampered-target".to_owned();
        }
        assert_eq!(
            compile_scene(&altered, &permit),
            Err(SceneCompileError::CueContentMismatch)
        );
    }

    #[test]
    fn audio_permit_cannot_compile_a_visual_scene() {
        let (canonical, _) = fixture(
            EmbodimentForm::LivingConstellation,
            PrivacyClass::Public,
            SemanticCue::Listening { active: true },
        );
        let mut policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        policy.grant(
            ConsentGrant {
                grant_id: "grant-audio".to_owned(),
                session_id: SESSION.to_owned(),
                clock_domain_id: CLOCK.to_owned(),
                channel: OutputChannel::Audio,
                destination: OutputDestination::LocalShared,
                allowed_forms: vec![EmbodimentForm::LivingConstellation],
                expires_at_monotonic_ns: "800".to_owned(),
                max_intensity: 0.6,
            },
            "150",
            CLOCK,
        ).unwrap();
        policy.set_paused(false);
        let audio_permit = policy.evaluate(
            &canonical,
            &PresentationRequest {
                form: EmbodimentForm::LivingConstellation,
                channel: OutputChannel::Audio,
                destination: OutputDestination::LocalShared,
                requested_intensity: 0.4,
            },
            "200",
            CLOCK,
        ).unwrap();
        assert_eq!(compile_scene(&canonical, &audio_permit), Err(SceneCompileError::WrongOutputChannel));
    }
}
