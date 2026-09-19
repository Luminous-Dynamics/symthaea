// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Presentation gates between completed voice cognition and TTS.
//!
//! This module deliberately owns no cognition and mints no semantic authority. The
//! basic gate prevents an already-cancelled generation from entering TTS. The
//! interruptible gate additionally registers a narrow backend stop capability for
//! the exact presentation generation and keeps that registration alive for the
//! complete render call.

use std::fmt;
use std::sync::Arc;

use crate::voice_control::{
    VoicePresentationControl, VoiceRegistrationError, VoiceStopCapability,
};
use crate::voice_session::VoicePresentationToken;
use crate::wire::{ServiceWireOutcome, ServiceWireResponse};

/// Minimal renderer contract used by the compatibility presentation gate.
///
/// Keeping this trait tiny makes cancellation policy testable without requiring
/// audio hardware and lets non-interruptible renderers retain the conservative
/// preflight-only contract.
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

/// Renderer-side completion for a presentation that supports a monotonic
/// cancellation probe as well as a separate stop capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceRenderCompletion {
    Completed,
    Cancelled,
}

/// Renderer contract required to close both pre-start and in-flight cancellation
/// races without sharing mutable renderer state with the service control path.
pub trait InterruptibleVoiceRenderer {
    type Error;

    /// Narrow stop authority for the renderer that is about to become active.
    fn stop_capability(&self) -> Arc<dyn VoiceStopCapability>;

    /// Render while consulting the supplied monotonic cancellation probe.
    fn speak_cancellable(
        &mut self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<VoiceRenderCompletion, Self::Error>;
}

/// Truthful result of one presentation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePresentationDisposition {
    /// TTS completed for the query response.
    Presented,
    /// Presentation had already been cancelled before renderer activation.
    SuppressedCancelled,
    /// Rendering began but cancellation was observed before normal completion.
    InterruptedDuringPresentation,
    /// The runtime outcome does not contain non-empty query response text.
    NoSpeakableResponse,
}

/// Failure specific to registering or invoking an interruptible renderer.
#[derive(Debug)]
pub enum InterruptiblePresentationError<E> {
    Registration(VoiceRegistrationError),
    Renderer(E),
}

impl<E: fmt::Display> fmt::Display for InterruptiblePresentationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registration(error) => write!(f, "voice presentation registration failed: {error}"),
            Self::Renderer(error) => write!(f, "voice renderer failed: {error}"),
        }
    }
}

impl<E> std::error::Error for InterruptiblePresentationError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Registration(error) => Some(error),
            Self::Renderer(error) => Some(error),
        }
    }
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
/// This compatibility path closes preflight cancellation only. A renderer already
/// executing after the final token check requires [`present_interruptible_voice_outcome`].
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

/// Present one generation through an interruptible renderer.
///
/// Registration and barge-in cancellation are serialized by
/// [`VoicePresentationControl`]. If cancellation wins before registration, this
/// returns `SuppressedCancelled` without invoking the renderer. If registration wins,
/// an RAII lease remains active for the complete render call and a concurrent barge-in
/// can invoke the registered stop capability. The renderer additionally receives the
/// monotonic presentation token, closing the stop-before-activation race even if its
/// own transient "speaking" state is later set active.
pub fn present_interruptible_voice_outcome<R: InterruptibleVoiceRenderer>(
    renderer: &mut R,
    control: &VoicePresentationControl,
    presentation: &VoicePresentationToken,
    outcome: &ServiceWireOutcome,
) -> Result<VoicePresentationDisposition, InterruptiblePresentationError<R::Error>> {
    if presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let Some(content) = speakable_content(outcome) else {
        return Ok(VoicePresentationDisposition::NoSpeakableResponse);
    };

    let generation = presentation.generation();
    let stop = renderer.stop_capability();
    let lease = match control.try_register(generation, stop) {
        Ok(lease) => lease,
        Err(VoiceRegistrationError::Cancelled { .. }) => {
            return Ok(VoicePresentationDisposition::SuppressedCancelled);
        }
        Err(error) => return Err(InterruptiblePresentationError::Registration(error)),
    };

    if presentation.is_cancelled() {
        drop(lease);
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let completion = renderer
        .speak_cancellable(content, &|| presentation.is_cancelled())
        .map_err(InterruptiblePresentationError::Renderer)?;

    drop(lease);
    Ok(match completion {
        VoiceRenderCompletion::Completed => VoicePresentationDisposition::Presented,
        VoiceRenderCompletion::Cancelled => {
            VoicePresentationDisposition::InterruptedDuringPresentation
        }
    })
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

    struct FakeStop {
        speaking: AtomicBool,
        stops: AtomicUsize,
    }

    impl FakeStop {
        fn active() -> Arc<Self> {
            Arc::new(Self {
                speaking: AtomicBool::new(true),
                stops: AtomicUsize::new(0),
            })
        }
    }

    impl VoiceStopCapability for FakeStop {
        fn stop(&self) {
            self.stops.fetch_add(1, Ordering::SeqCst);
            self.speaking.store(false, Ordering::SeqCst);
        }

        fn is_speaking(&self) -> bool {
            self.speaking.load(Ordering::SeqCst)
        }
    }

    struct InterruptibleRecordingRenderer {
        spoken: Vec<String>,
        stop: Arc<FakeStop>,
        completion: VoiceRenderCompletion,
    }

    impl InterruptibleRecordingRenderer {
        fn new(completion: VoiceRenderCompletion) -> Self {
            Self {
                spoken: Vec::new(),
                stop: FakeStop::active(),
                completion,
            }
        }
    }

    impl InterruptibleVoiceRenderer for InterruptibleRecordingRenderer {
        type Error = Infallible;

        fn stop_capability(&self) -> Arc<dyn VoiceStopCapability> {
            let capability: Arc<dyn VoiceStopCapability> = self.stop.clone();
            capability
        }

        fn speak_cancellable(
            &mut self,
            text: &str,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<VoiceRenderCompletion, Self::Error> {
            self.spoken.push(text.to_string());
            if cancelled() {
                Ok(VoiceRenderCompletion::Cancelled)
            } else {
                Ok(self.completion)
            }
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

    #[test]
    fn interruptible_registration_rejects_generation_cancelled_before_activation() {
        let (_session, turn) = voice_turn();
        let control = VoicePresentationControl::default();
        assert_eq!(
            control.interrupt_through(turn.presentation_token().generation()),
            crate::voice_control::VoiceStopOutcome::NoActivePresentation
        );

        let mut renderer = InterruptibleRecordingRenderer::new(VoiceRenderCompletion::Completed);
        let disposition = present_interruptible_voice_outcome(
            &mut renderer,
            &control,
            turn.presentation_token(),
            &query_outcome("do not speak"),
        )
        .unwrap();

        assert_eq!(
            disposition,
            VoicePresentationDisposition::SuppressedCancelled
        );
        assert!(renderer.spoken.is_empty());
        assert_eq!(control.active_generation(), None);
    }

    #[test]
    fn interruptible_renderer_lease_is_cleared_after_completion() {
        let (_session, turn) = voice_turn();
        let control = VoicePresentationControl::default();
        let mut renderer = InterruptibleRecordingRenderer::new(VoiceRenderCompletion::Completed);

        let disposition = present_interruptible_voice_outcome(
            &mut renderer,
            &control,
            turn.presentation_token(),
            &query_outcome("hello live voice"),
        )
        .unwrap();

        assert_eq!(disposition, VoicePresentationDisposition::Presented);
        assert_eq!(renderer.spoken, vec!["hello live voice"]);
        assert_eq!(control.active_generation(), None);
    }

    #[test]
    fn renderer_reported_cancellation_is_not_mislabeled_as_completed() {
        let (_session, turn) = voice_turn();
        let control = VoicePresentationControl::default();
        let mut renderer = InterruptibleRecordingRenderer::new(VoiceRenderCompletion::Cancelled);

        let disposition = present_interruptible_voice_outcome(
            &mut renderer,
            &control,
            turn.presentation_token(),
            &query_outcome("partially spoken"),
        )
        .unwrap();

        assert_eq!(
            disposition,
            VoicePresentationDisposition::InterruptedDuringPresentation
        );
        assert_eq!(control.active_generation(), None);
    }
}
