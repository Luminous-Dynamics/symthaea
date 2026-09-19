// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Presentation gate between a completed voice cognition turn and TTS.
//!
//! This module deliberately owns no cognition and mints no semantic authority. Its
//! job is narrower: consume the voice presentation token before handing text to a
//! renderer, so a turn cancelled before presentation cannot accidentally begin TTS.
//! Mid-stream cancellation requires an interruptible renderer and is a separate
//! capability; this gate does not claim to close the check-to-play race by itself.

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

impl VoiceRenderer for symthaea::voice::VoiceConversation {
    type Error = anyhow::Error;

    fn speak(&mut self, text: &str) -> Result<(), Self::Error> {
        symthaea::voice::VoiceConversation::speak(self, text)
    }
}

/// Truthful result of one presentation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePresentationDisposition {
    /// TTS was invoked for the query response.
    Presented,
    /// Presentation had already been cancelled before the renderer was invoked.
    SuppressedCancelled,
    /// The runtime outcome does not contain non-empty query response text.
    NoSpeakableResponse,
}

/// Present a completed runtime outcome only when its voice generation is still
/// eligible.
///
/// The token is checked before response extraction and again immediately before the
/// renderer call. This guarantees that an already-cancelled turn is never handed to
/// TTS. A cancellation racing with a renderer that has already started requires the
/// renderer-side stop capability introduced by the next seam.
pub fn present_voice_outcome<R: VoiceRenderer>(
    renderer: &mut R,
    presentation: &VoicePresentationToken,
    outcome: &ServiceWireOutcome,
) -> Result<VoicePresentationDisposition, R::Error> {
    if presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let content = match &outcome.response {
        ServiceWireResponse::QueryResponse { content, .. } if !content.trim().is_empty() => content,
        _ => return Ok(VoicePresentationDisposition::NoSpeakableResponse),
    };

    if presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    renderer.speak(content)?;
    Ok(VoicePresentationDisposition::Presented)
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
