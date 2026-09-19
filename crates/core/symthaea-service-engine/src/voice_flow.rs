// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Daemon-facing composition of one voice session, cognition admission, semantic
//! interruption, and bounded presentation scheduling.
//!
//! The completion path and interruption path are deliberately independent:
//!
//! - completion: STT text -> bounded cognition owner -> bounded presentation worker;
//! - interruption: session token + semantic event + active-renderer stop, with no
//!   wait on either bounded queue.
//!
//! This gives service code one duplex control surface without moving microphone,
//! STT, TTS, or mutable cognition ownership into this module.

use std::fmt;

use symthaea_interface_types::{SessionId, TurnId};

use crate::host::ServiceRuntimeHost;
use crate::protocol::ServiceProtocolCore;
use crate::protocol_error::ServiceProtocolFailure;
use crate::voice_session::{
    VoiceCancellationCapability, VoiceCognitionTicket, VoiceIngress, VoiceInterruptionReceipt,
    VoiceSession, VoiceSessionError,
};
use crate::voice_session_control::SessionVoiceStopOutcome;
use crate::voice_worker::{
    VoicePresentationJob, VoicePresentationSubmitError, VoicePresentationTicket,
    VoicePresentationWorkerHandle,
};

/// One cloneable service-level voice session plus presentation scheduler.
pub struct VoiceServiceFlow<E> {
    session: VoiceSession,
    presentation: VoicePresentationWorkerHandle<E>,
}

impl<E> Clone for VoiceServiceFlow<E> {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            presentation: self.presentation.clone(),
        }
    }
}

impl<E> VoiceServiceFlow<E> {
    pub fn new(session: VoiceSession, presentation: VoicePresentationWorkerHandle<E>) -> Self {
        Self {
            session,
            presentation,
        }
    }

    pub fn session(&self) -> &VoiceSession {
        &self.session
    }

    pub fn session_id(&self) -> &SessionId {
        self.session.session_id()
    }

    pub fn cancellation_capability(&self) -> VoiceCancellationCapability {
        self.session.cancellation_capability()
    }
}

#[derive(Debug)]
pub enum VoiceFlowAdmissionError {
    Session(VoiceSessionError),
    Cognition(ServiceProtocolFailure),
}

impl fmt::Display for VoiceFlowAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(error) => write!(f, "voice turn construction failed: {error}"),
            Self::Cognition(error) => write!(f, "voice cognition admission failed: {error}"),
        }
    }
}

impl std::error::Error for VoiceFlowAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Session(error) => Some(error),
            Self::Cognition(error) => Some(error),
        }
    }
}

impl From<VoiceSessionError> for VoiceFlowAdmissionError {
    fn from(value: VoiceSessionError) -> Self {
        Self::Session(value)
    }
}

impl From<ServiceProtocolFailure> for VoiceFlowAdmissionError {
    fn from(value: ServiceProtocolFailure) -> Self {
        Self::Cognition(value)
    }
}

/// Stable control capability retained by UI/VAD while cognition completion is
/// awaited elsewhere.
pub struct VoiceFlowTurnControl<E> {
    session: VoiceSession,
    turn_id: TurnId,
    presentation: VoicePresentationWorkerHandle<E>,
}

impl<E> Clone for VoiceFlowTurnControl<E> {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            turn_id: self.turn_id.clone(),
            presentation: self.presentation.clone(),
        }
    }
}

impl<E> VoiceFlowTurnControl<E> {
    pub fn session_id(&self) -> &SessionId {
        self.session.session_id()
    }

    pub fn turn_id(&self) -> &TurnId {
        &self.turn_id
    }
}

/// Combined truth from one fast barge-in operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceFlowInterruptionReceipt {
    pub voice: VoiceInterruptionReceipt,
    pub presentation_stop: SessionVoiceStopOutcome,
}

impl<E: Send + 'static> VoiceFlowTurnControl<E> {
    /// Cancel presentation eligibility, attempt authoritative semantic interruption,
    /// and directly stop any matching active renderer.
    ///
    /// None of these operations waits on the cognition mailbox or presentation job
    /// queue. Semantic interruption may be rejected if `ResponseStarted` has not yet
    /// been observed; presentation cancellation remains effective regardless.
    pub fn interrupt(&self, host: &ServiceRuntimeHost) -> VoiceFlowInterruptionReceipt {
        let mut voice = self.session.interrupt_presentation();
        match host.voice_interrupted(
            Some(self.session.session_id().clone()),
            self.turn_id.clone(),
        ) {
            Ok(cursor) => {
                voice.semantic_event_emitted = true;
                voice.semantic_event_cursor = Some(cursor);
            }
            Err(error) => {
                voice.semantic_event_error = Some(error.to_string());
            }
        }
        let presentation_stop = self.presentation.apply_interruption(&voice);
        VoiceFlowInterruptionReceipt {
            voice,
            presentation_stop,
        }
    }

    /// Presentation-only fast path for a caller that intentionally does not publish
    /// a semantic interruption event (for example, a local preflight abort).
    pub fn interrupt_presentation_only(&self) -> VoiceFlowInterruptionReceipt {
        let voice = self.session.interrupt_presentation();
        let presentation_stop = self.presentation.apply_interruption(&voice);
        VoiceFlowInterruptionReceipt {
            voice,
            presentation_stop,
        }
    }
}

/// One admitted cognition turn whose result can later be submitted to presentation.
pub struct VoiceFlowCompletion<E> {
    cognition: VoiceCognitionTicket,
    presentation: VoicePresentationWorkerHandle<E>,
}

#[derive(Debug)]
pub enum VoiceFlowCompletionError {
    Cognition(ServiceProtocolFailure),
    Presentation(VoicePresentationSubmitError),
}

impl fmt::Display for VoiceFlowCompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cognition(error) => write!(f, "voice cognition failed: {error}"),
            Self::Presentation(VoicePresentationSubmitError::Busy(job)) => write!(
                f,
                "voice presentation queue is full for {} generation {}",
                job.identity().session_id,
                job.generation()
            ),
            Self::Presentation(VoicePresentationSubmitError::Closed(job)) => write!(
                f,
                "voice presentation worker is closed for {} generation {}",
                job.identity().session_id,
                job.generation()
            ),
        }
    }
}

impl std::error::Error for VoiceFlowCompletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cognition(error) => Some(error),
            Self::Presentation(_) => None,
        }
    }
}

impl<E: Send + 'static> VoiceFlowCompletion<E> {
    /// Resolve cognition, preserve its exact session/utterance/turn correlation, then
    /// non-blockingly admit presentation. A full TTS queue never backpressures the
    /// cognition owner or the caller's async executor.
    pub async fn resolve_and_submit(
        self,
    ) -> Result<VoicePresentationTicket<E>, VoiceFlowCompletionError> {
        let outcome = self
            .cognition
            .resolve()
            .await
            .map_err(VoiceFlowCompletionError::Cognition)?;
        self.presentation
            .try_submit(VoicePresentationJob::from_cognition(outcome))
            .map_err(VoiceFlowCompletionError::Presentation)
    }
}

/// Result of admitting one recognized utterance to cognition.
pub struct VoiceFlowAdmission<E> {
    pub control: VoiceFlowTurnControl<E>,
    pub completion: VoiceFlowCompletion<E>,
}

impl<E: Send + 'static> VoiceServiceFlow<E> {
    /// Begin one voice turn and admit its recognized semantic text to the bounded
    /// cognition owner without waiting for cognition or TTS to complete.
    pub fn admit_transcript(
        &self,
        core: &ServiceProtocolCore,
        host: &ServiceRuntimeHost,
        ingress: VoiceIngress,
        transcript: impl Into<String>,
    ) -> Result<VoiceFlowAdmission<E>, VoiceFlowAdmissionError> {
        let turn = self.session.begin_turn(ingress)?;
        let cognition = turn.admit_transcript(core, host, transcript)?;
        let turn_id = cognition.turn_id().clone();

        Ok(VoiceFlowAdmission {
            control: VoiceFlowTurnControl {
                session: self.session.clone(),
                turn_id,
                presentation: self.presentation.clone(),
            },
            completion: VoiceFlowCompletion {
                cognition,
                presentation: self.presentation.clone(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_and_completion_errors_are_distinct_categories() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<VoiceFlowAdmissionError>();
        assert_error::<VoiceFlowCompletionError>();
    }
}
