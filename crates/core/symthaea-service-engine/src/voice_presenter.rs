// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Presentation gate between a completed voice cognition turn and TTS.
//!
//! This module deliberately owns no cognition and mints no semantic authority. Its
//! job is narrower: consume the voice presentation token before handing text to a
//! renderer. For the live backend it additionally binds the generation to the
//! race-safe active-presentation control and uses the same token as a monotonic
//! per-frame cancellation probe.

#[cfg(feature = "live-voice")]
use std::fmt;
#[cfg(feature = "live-voice")]
use std::sync::Arc;

#[cfg(feature = "live-voice")]
use crate::voice_control::{
    VoicePresentationControl, VoiceRegistrationError, VoiceStopCapability,
};
use crate::voice_session::VoicePresentationToken;
use crate::wire::{ServiceWireOutcome, ServiceWireResponse};

/// Minimal renderer contract used by the service presentation gate.
///
/// Keeping this trait tiny makes the cancellation policy testable without requiring
/// audio hardware and lets future renderers (vocal tract, Kokoro, remote speech,
/// accessibility output) share the same preflight rule.
pub trait VoiceRenderer {
    type Error;

    fn speak(&mut self, text: &str) -> Result<(), Self::Error>;
}

#[cfg(feature = "voice-tts")]
impl VoiceRenderer for symthaea::voice::VoiceConversation {
    type Error = anyhow::Error;

    fn speak(&mut self, text: &str) -> Result<(), Self::Error> {
        symthaea::voice::VoiceConversation::speak(self, text)
    }
}

/// Truthful result of one presentation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePresentationDisposition {
    /// TTS completed without observing cancellation.
    Presented,
    /// Presentation had already been cancelled before the renderer was invoked.
    SuppressedCancelled,
    /// Live rendering began but cancellation was observed while it was active.
    Interrupted,
    /// The runtime outcome does not contain non-empty query response text.
    NoSpeakableResponse,
}

fn speakable_content(outcome: &ServiceWireOutcome) -> Option<&str> {
    match &outcome.response {
        ServiceWireResponse::QueryResponse { content, .. } if !content.trim().is_empty() => {
            Some(content)
        }
        _ => None,
    }
}

/// Present a completed runtime outcome only when its voice generation is still
/// eligible.
///
/// This compatibility path guarantees that an already-cancelled turn is never
/// handed to TTS. Renderers that need race-safe mid-stream barge-in should use the
/// live controlled path below or provide an equivalent stop capability.
pub fn present_voice_outcome<R: VoiceRenderer>(
    renderer: &mut R,
    presentation: &VoicePresentationToken,
    outcome: &ServiceWireOutcome,
) -> Result<VoicePresentationDisposition, R::Error> {
    if presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let Some(content) = speakable_content(outcome) else {
        return Ok(VoicePresentationDisposition::NoSpeakableResponse);
    };

    if presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    renderer.speak(content)?;
    Ok(VoicePresentationDisposition::Presented)
}

/// Failure of the race-safe live presentation path.
#[cfg(feature = "live-voice")]
#[derive(Debug)]
pub enum LiveVoicePresentationError {
    Registration(VoiceRegistrationError),
    Renderer(anyhow::Error),
}

#[cfg(feature = "live-voice")]
impl fmt::Display for LiveVoicePresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registration(error) => write!(f, "live voice registration failed: {error}"),
            Self::Renderer(error) => write!(f, "live voice rendering failed: {error}"),
        }
    }
}

#[cfg(feature = "live-voice")]
impl std::error::Error for LiveVoicePresentationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Registration(error) => Some(error),
            Self::Renderer(error) => Some(error.as_ref()),
        }
    }
}

/// Present one response through the real-time live backend with generation-safe
/// registration and monotonic cancellation.
///
/// The active registration lease is held for the complete `speak_cancellable`
/// call. Barge-in may therefore request stop through [`VoicePresentationControl`]
/// without borrowing mutable voice state. The presentation token is also passed as
/// the live backend's monotonic cancellation probe, closing the stop-before-start
/// race even if cancellation wins immediately before backend activation.
#[cfg(feature = "live-voice")]
pub fn present_live_voice_outcome(
    renderer: &mut symthaea::voice::LiveVoice,
    control: &VoicePresentationControl,
    presentation: &VoicePresentationToken,
    outcome: &ServiceWireOutcome,
) -> Result<VoicePresentationDisposition, LiveVoicePresentationError> {
    if presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let Some(content) = speakable_content(outcome) else {
        return Ok(VoicePresentationDisposition::NoSpeakableResponse);
    };

    let generation = presentation.generation();
    let stop: Arc<dyn VoiceStopCapability> = Arc::new(renderer.stop_handle());
    let _lease = match control.try_register(generation, stop) {
        Ok(lease) => lease,
        Err(VoiceRegistrationError::Cancelled { .. }) => {
            return Ok(VoicePresentationDisposition::SuppressedCancelled);
        }
        Err(error) => return Err(LiveVoicePresentationError::Registration(error)),
    };

    // Cancellation can race between the first eligibility check and registration.
    // Record it in the shared control plane as well as relying on the monotonic probe.
    if presentation.is_cancelled() {
        let _ = control.interrupt_through(generation);
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let result = renderer
        .speak_cancellable(content, || presentation.is_cancelled())
        .map_err(LiveVoicePresentationError::Renderer)?;

    match result {
        symthaea::voice::live_voice::LiveVoiceSpeakOutcome::Completed => {
            Ok(VoicePresentationDisposition::Presented)
        }
        symthaea::voice::live_voice::LiveVoiceSpeakOutcome::Cancelled => {
            Ok(VoicePresentationDisposition::Interrupted)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::voice_session::{VoiceIngress, VoiceSession};
    use crate::wire::ServiceWireDiagnostics;
    use symthaea_interface_types::SessionId;

    #[derive(Default)]
    struct RecordingRenderer {
        spoken: Vec<String>,
    }

    impl VoiceRenderer for RecordingRenderer {
        type Error = Infallible;

        fn speak(&mut self, text: &str) -> Result<(), Self::Error> {
            self.spoken.push(text.to_string());
            Ok(())
        }
    }

    fn query_outcome(content: &str) -> ServiceWireOutcome {
        ServiceWireOutcome {
            response: ServiceWireResponse::QueryResponse {
                content: content.to_string(),
                confidence: 0.9,
                safe: true,
                phi: 0.4,
                phi_dyad: 0.0,
                steps_to_emergence: 0,
                processing_time_ms: 1,
                creative_artifact: None,
            },
            diagnostics: ServiceWireDiagnostics::default(),
            bridge_telemetry: None,
        }
    }

    fn voice_turn() -> (VoiceSession, crate::voice_session::VoiceTurn) {
        let session = VoiceSession::with_session_id(SessionId::new("voice-session:test").unwrap());
        let turn = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        (session, turn)
    }

    #[test]
    fn pre_cancelled_generation_never_reaches_renderer() {
        let (session, turn) = voice_turn();
        let token = turn.presentation_token().clone();
        session.interrupt_presentation();

        let mut renderer = RecordingRenderer::default();
        let disposition = present_voice_outcome(&mut renderer, &token, &query_outcome("hello"))
            .unwrap();

        assert_eq!(
            disposition,
            VoicePresentationDisposition::SuppressedCancelled
        );
        assert!(renderer.spoken.is_empty());
    }

    #[test]
    fn eligible_query_response_is_presented_verbatim() {
        let (_session, turn) = voice_turn();
        let mut renderer = RecordingRenderer::default();

        let disposition = present_voice_outcome(
            &mut renderer,
            turn.presentation_token(),
            &query_outcome("hello Symthaea"),
        )
        .unwrap();

        assert_eq!(disposition, VoicePresentationDisposition::Presented);
        assert_eq!(renderer.spoken, vec!["hello Symthaea"]);
    }

    #[test]
    fn non_query_or_empty_response_never_becomes_speech() {
        let (_session, turn) = voice_turn();
        let mut renderer = RecordingRenderer::default();
        let empty = query_outcome("   ");

        assert_eq!(
            present_voice_outcome(&mut renderer, turn.presentation_token(), &empty).unwrap(),
            VoicePresentationDisposition::NoSpeakableResponse
        );

        let status = ServiceWireOutcome {
            response: ServiceWireResponse::Status {
                uptime_seconds: 1,
                requests_processed: 1,
                consciousness_level: 0.2,
                memory_count: 0,
                sleep_cycles: 0,
            },
            diagnostics: ServiceWireDiagnostics::default(),
            bridge_telemetry: None,
        };
        assert_eq!(
            present_voice_outcome(&mut renderer, turn.presentation_token(), &status).unwrap(),
            VoicePresentationDisposition::NoSpeakableResponse
        );
        assert!(renderer.spoken.is_empty());
    }
}
