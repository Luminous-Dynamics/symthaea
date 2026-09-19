// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Presentation gate between a completed voice cognition turn and TTS.
//!
//! This module deliberately owns no cognition and mints no semantic authority. It
//! combines monotonic presentation eligibility with optional renderer-side stop
//! authority, while keeping mutable synthesis state behind a narrow renderer trait.

use std::fmt;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::voice_control::{
    VoicePresentationControl, VoiceRegistrationError, VoiceStopCapability,
};
use crate::voice_session::VoicePresentationToken;
use crate::wire::{ServiceWireOutcome, ServiceWireResponse};

const PLAYBACK_DRAIN_POLL: Duration = Duration::from_millis(2);
const PLAYBACK_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Minimal renderer contract used by the simple presentation gate.
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

/// Renderer-local result for a monotonic cancellation-aware speech operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellableRenderOutcome {
    Completed,
    Cancelled,
}

/// Stronger renderer contract for a backend that can both expose a narrow stop
/// capability and continuously consult a monotonic cancellation probe.
pub trait CancellableVoiceRenderer {
    type Error;

    fn stop_capability(&self) -> Arc<dyn VoiceStopCapability>;

    fn speak_cancellable(
        &mut self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<CancellableRenderOutcome, Self::Error>;
}

/// Truthful result of one presentation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePresentationDisposition {
    /// TTS completed and software-buffered playback drained.
    Presented,
    /// Presentation had already been cancelled before the renderer was invoked.
    SuppressedCancelled,
    /// Cancellation was observed after active-renderer registration.
    CancelledDuringPresentation,
    /// The runtime outcome does not contain non-empty query response text.
    NoSpeakableResponse,
}

/// Failure specific to composing active-renderer registration with TTS execution.
#[derive(Debug)]
pub enum InterruptiblePresentationError<E> {
    Registration(VoiceRegistrationError),
    Renderer(E),
    PlaybackDrainTimeout { generation: u64 },
}

impl<E: fmt::Display> fmt::Display for InterruptiblePresentationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registration(error) => write!(f, "voice presentation registration failed: {error}"),
            Self::Renderer(error) => write!(f, "voice renderer failed: {error}"),
            Self::PlaybackDrainTimeout { generation } => write!(
                f,
                "voice generation {generation} did not drain software playback within {:?}",
                PLAYBACK_DRAIN_TIMEOUT
            ),
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
            Self::PlaybackDrainTimeout { .. } => None,
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
/// This compatibility path guarantees that an already-cancelled turn is never
/// handed to TTS, but cannot stop a renderer once `speak()` has begun. Prefer
/// [`present_interruptible_voice_outcome`] when the backend supports it.
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

/// Present through a renderer that supports both monotonic cancellation and a
/// cross-thread stop capability.
///
/// Registration and barge-in are serialized by [`VoicePresentationControl`]. The
/// active-generation lease remains alive across synthesis *and* software-buffered
/// playback drain. The presentation token is rechecked after registration, probed
/// continuously by the renderer, and checked while draining, closing both the
/// stop-before-start race and the synthesis-finished/playback-still-audible gap.
///
/// This function is intentionally synchronous because the renderer contract is
/// synchronous. Production callers should execute it on a dedicated presentation
/// worker rather than a Tokio core worker. A later tranche will make that scheduling
/// boundary explicit.
pub fn present_interruptible_voice_outcome<R: CancellableVoiceRenderer>(
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
    let playback = Arc::clone(&stop);
    let _lease = match control.try_register(generation, stop) {
        Ok(lease) => lease,
        Err(VoiceRegistrationError::Cancelled { .. }) => {
            return Ok(VoicePresentationDisposition::SuppressedCancelled);
        }
        Err(error) => return Err(InterruptiblePresentationError::Registration(error)),
    };

    if presentation.is_cancelled() {
        playback.stop();
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let cancelled = || presentation.is_cancelled();
    match renderer
        .speak_cancellable(content, &cancelled)
        .map_err(InterruptiblePresentationError::Renderer)?
    {
        CancellableRenderOutcome::Cancelled => {
            playback.stop();
            return Ok(VoicePresentationDisposition::CancelledDuringPresentation);
        }
        CancellableRenderOutcome::Completed => {}
    }

    let drain_started = Instant::now();
    while playback.is_speaking() {
        if presentation.is_cancelled() {
            playback.stop();
            return Ok(VoicePresentationDisposition::CancelledDuringPresentation);
        }
        if drain_started.elapsed() >= PLAYBACK_DRAIN_TIMEOUT {
            playback.stop();
            return Err(InterruptiblePresentationError::PlaybackDrainTimeout {
                generation,
            });
        }
        thread::sleep(PLAYBACK_DRAIN_POLL);
    }

    Ok(VoicePresentationDisposition::Presented)
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::voice_control::VoiceStopOutcome;
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
    }

    impl FakeStop {
        fn active() -> Arc<Self> {
            Arc::new(Self {
                speaking: AtomicBool::new(true),
            })
        }
    }

    impl VoiceStopCapability for FakeStop {
        fn stop(&self) {
            self.speaking.store(false, Ordering::SeqCst);
        }

        fn is_speaking(&self) -> bool {
            self.speaking.load(Ordering::SeqCst)
        }
    }

    struct BlockingRenderer {
        stop: Arc<FakeStop>,
        entered: Arc<Barrier>,
    }

    impl CancellableVoiceRenderer for BlockingRenderer {
        type Error = Infallible;

        fn stop_capability(&self) -> Arc<dyn VoiceStopCapability> {
            let capability: Arc<dyn VoiceStopCapability> = self.stop.clone();
            capability
        }

        fn speak_cancellable(
            &mut self,
            _text: &str,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<CancellableRenderOutcome, Self::Error> {
            self.entered.wait();
            while !cancelled() && self.stop.is_speaking() {
                thread::yield_now();
            }
            Ok(CancellableRenderOutcome::Cancelled)
        }
    }

    struct CompletedThenDrainingRenderer {
        stop: Arc<FakeStop>,
        synthesized: Arc<Barrier>,
    }

    impl CancellableVoiceRenderer for CompletedThenDrainingRenderer {
        type Error = Infallible;

        fn stop_capability(&self) -> Arc<dyn VoiceStopCapability> {
            let capability: Arc<dyn VoiceStopCapability> = self.stop.clone();
            capability
        }

        fn speak_cancellable(
            &mut self,
            _text: &str,
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<CancellableRenderOutcome, Self::Error> {
            self.synthesized.wait();
            Ok(CancellableRenderOutcome::Completed)
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
    fn control_cancellation_before_registration_suppresses_interruptible_renderer() {
        let (_session, turn) = voice_turn();
        let control = VoicePresentationControl::default();
        assert_eq!(
            control.interrupt_through(turn.presentation_token().generation()),
            VoiceStopOutcome::NoActivePresentation
        );
        let stop = FakeStop::active();
        let barrier = Arc::new(Barrier::new(1));
        let mut renderer = BlockingRenderer {
            stop,
            entered: barrier,
        };

        let disposition = present_interruptible_voice_outcome(
            &mut renderer,
            &control,
            turn.presentation_token(),
            &query_outcome("must not start"),
        )
        .unwrap();

        assert_eq!(
            disposition,
            VoicePresentationDisposition::SuppressedCancelled
        );
        assert_eq!(control.active_generation(), None);
    }

    #[test]
    fn in_flight_barge_in_stops_registered_renderer_and_clears_lease() {
        let (session, turn) = voice_turn();
        let token = turn.presentation_token().clone();
        let control = VoicePresentationControl::default();
        let control_for_thread = control.clone();
        let stop = FakeStop::active();
        let stop_for_thread = stop.clone();
        let entered = Arc::new(Barrier::new(2));
        let entered_for_thread = entered.clone();

        let worker = thread::spawn(move || {
            let mut renderer = BlockingRenderer {
                stop: stop_for_thread,
                entered: entered_for_thread,
            };
            present_interruptible_voice_outcome(
                &mut renderer,
                &control_for_thread,
                &token,
                &query_outcome("long response"),
            )
            .unwrap()
        });

        entered.wait();
        assert_eq!(control.active_generation(), Some(1));

        let receipt = session.interrupt_presentation();
        assert_eq!(
            control.apply_interruption(&receipt),
            VoiceStopOutcome::StopRequested {
                generation: 1,
                was_speaking: true,
            }
        );

        let disposition = worker.join().unwrap();
        assert_eq!(
            disposition,
            VoicePresentationDisposition::CancelledDuringPresentation
        );
        assert!(!stop.is_speaking());
        assert_eq!(control.active_generation(), None);
    }

    #[test]
    fn active_generation_remains_registered_until_playback_drains() {
        let (_session, turn) = voice_turn();
        let token = turn.presentation_token().clone();
        let control = VoicePresentationControl::default();
        let control_for_thread = control.clone();
        let stop = FakeStop::active();
        let stop_for_thread = stop.clone();
        let synthesized = Arc::new(Barrier::new(2));
        let synthesized_for_thread = synthesized.clone();

        let worker = thread::spawn(move || {
            let mut renderer = CompletedThenDrainingRenderer {
                stop: stop_for_thread,
                synthesized: synthesized_for_thread,
            };
            present_interruptible_voice_outcome(
                &mut renderer,
                &control_for_thread,
                &token,
                &query_outcome("queued audio remains"),
            )
            .unwrap()
        });

        synthesized.wait();
        assert_eq!(control.active_generation(), Some(1));
        stop.speaking.store(false, Ordering::SeqCst);

        assert_eq!(worker.join().unwrap(), VoicePresentationDisposition::Presented);
        assert_eq!(control.active_generation(), None);
    }
}
