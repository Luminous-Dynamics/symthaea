// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Correlation and cancellation contract for service voice turns.
//!
//! Audio capture, STT, TTS, and playback remain outside this module. This layer
//! gives those replaceable surfaces one shared session/utterance identity and routes
//! semantic text through the sole cognition owner with the correct [`ProcessOrigin`].
//!
//! Cancellation is deliberately truthful. The current owner executes an admitted
//! command exactly once and does not yet expose cooperative query cancellation, and
//! the legacy blocking speaker cannot necessarily stop an already submitted audio
//! buffer. A barge-in therefore cancels **presentation eligibility** immediately for
//! the current/older voice generations. Cooperative cognition/audio-device abort is
//! a later capability and must not be inferred from this token.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use symthaea_interface_types::{IdError, SessionId, TurnId, UtteranceId};
use symthaea_service_runtime::ProcessOrigin;
use uuid::Uuid;

use crate::host::ServiceRuntimeHost;
use crate::protocol::{CorrelatedQueryTicket, ServiceProtocolCore};
use crate::protocol_error::ServiceProtocolFailure;
use crate::wire::ServiceWireOutcome;

/// How semantic text entered one voice turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceIngress {
    /// Live listen -> speech recognition -> cognition.
    LiveMicrophone,
    /// Existing audio/WAV -> speech recognition -> cognition.
    TranscribedAudio,
}

impl VoiceIngress {
    pub fn process_origin(self) -> ProcessOrigin {
        match self {
            Self::LiveMicrophone => ProcessOrigin::VoiceTurn,
            Self::TranscribedAudio => ProcessOrigin::VoiceTranscription,
        }
    }
}

/// What a voice barge-in can currently guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceCancellationCapability {
    /// Future presentation work can observe cancellation before enqueue/playback.
    /// Already-admitted cognition and a blocking audio backend are not guaranteed to
    /// abort yet.
    PresentationEligibilityOnly,
}

/// Voice-session construction/admission failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceSessionError {
    Identity(IdError),
    GenerationExhausted,
}

impl fmt::Display for VoiceSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(error) => write!(f, "voice identity is invalid: {error}"),
            Self::GenerationExhausted => write!(f, "voice session generation counter exhausted"),
        }
    }
}

impl std::error::Error for VoiceSessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(error) => Some(error),
            Self::GenerationExhausted => None,
        }
    }
}

impl From<IdError> for VoiceSessionError {
    fn from(value: IdError) -> Self {
        Self::Identity(value)
    }
}

#[derive(Debug)]
struct VoiceCancellationState {
    current_generation: AtomicU64,
    cancelled_through: AtomicU64,
}

impl Default for VoiceCancellationState {
    fn default() -> Self {
        Self {
            current_generation: AtomicU64::new(0),
            cancelled_through: AtomicU64::new(0),
        }
    }
}

fn interrupt_started_presentations(cancellation: &VoiceCancellationState) -> u64 {
    let through = cancellation.current_generation.load(Ordering::Acquire);
    cancellation
        .cancelled_through
        .fetch_max(through, Ordering::AcqRel);
    through
}

/// Cloneable presentation token for one voice generation.
///
/// Streaming TTS/audio code should check this before enqueueing each clause/chunk.
/// A legacy blocking `speak()` call can at minimum check it before starting output.
#[derive(Debug, Clone)]
pub struct VoicePresentationToken {
    generation: u64,
    cancellation: Arc<VoiceCancellationState>,
}

impl VoicePresentationToken {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.cancelled_through.load(Ordering::Acquire) >= self.generation
    }

    fn interrupt_started(&self) -> u64 {
        interrupt_started_presentations(&self.cancellation)
    }
}

/// Immutable correlation identity for one recognized user utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceTurnIdentity {
    pub session_id: SessionId,
    pub utterance_id: UtteranceId,
    pub ingress: VoiceIngress,
    pub presentation_generation: u64,
}

/// One voice utterance before cognition admission.
#[derive(Debug, Clone)]
pub struct VoiceTurn {
    identity: VoiceTurnIdentity,
    presentation: VoicePresentationToken,
}

impl VoiceTurn {
    pub fn identity(&self) -> &VoiceTurnIdentity {
        &self.identity
    }

    pub fn presentation_token(&self) -> &VoicePresentationToken {
        &self.presentation
    }

    /// Admit recognized semantic text to the sole cognition owner without waiting
    /// for cognition to finish. The returned ticket already knows the exact semantic
    /// `TurnId`, enabling later lifecycle-aware barge-in while it is unresolved.
    pub fn admit_transcript(
        &self,
        core: &ServiceProtocolCore,
        host: &ServiceRuntimeHost,
        transcript: impl Into<String>,
    ) -> Result<VoiceCognitionTicket, ServiceProtocolFailure> {
        let inner = core.try_query_correlated_with_origin(
            host,
            transcript,
            self.identity.ingress.process_origin(),
        )?;
        let turn_id = inner.turn_id().clone();
        Ok(VoiceCognitionTicket {
            identity: self.identity.clone(),
            turn_id,
            presentation: self.presentation.clone(),
            inner,
        })
    }

    /// Convenience path for callers that do not need in-flight control.
    pub async fn process_transcript(
        &self,
        core: &ServiceProtocolCore,
        host: &ServiceRuntimeHost,
        transcript: impl Into<String>,
    ) -> Result<VoiceCognitionOutcome, ServiceProtocolFailure> {
        self.admit_transcript(core, host, transcript)?.resolve().await
    }
}

/// One voice turn already admitted to cognition but not necessarily completed.
///
/// Dropping this ticket does not cancel the admitted owner command. It only drops
/// the caller's completion observation, matching the runtime-owner contract.
pub struct VoiceCognitionTicket {
    identity: VoiceTurnIdentity,
    turn_id: TurnId,
    presentation: VoicePresentationToken,
    inner: CorrelatedQueryTicket,
}

impl VoiceCognitionTicket {
    pub fn identity(&self) -> &VoiceTurnIdentity {
        &self.identity
    }

    pub fn turn_id(&self) -> &TurnId {
        &self.turn_id
    }

    pub fn presentation_token(&self) -> &VoicePresentationToken {
        &self.presentation
    }

    /// Fast presentation-only barge-in available immediately after admission.
    ///
    /// Semantic interruption is intentionally not emitted here yet: admission can
    /// precede owner-side `ResponseStarted` publication. The next lifecycle-aware
    /// tranche will emit `VoiceInterrupted` only after that turn is proven active.
    pub fn interrupt_presentation(&self) -> VoiceInterruptionReceipt {
        let through = self.presentation.interrupt_started();
        VoiceInterruptionReceipt {
            session_id: self.identity.session_id.clone(),
            cancelled_through_generation: through,
            capability: VoiceCancellationCapability::PresentationEligibilityOnly,
            semantic_event_emitted: false,
            cognition_cancelled: false,
        }
    }

    pub async fn resolve(self) -> Result<VoiceCognitionOutcome, ServiceProtocolFailure> {
        let correlated = self.inner.resolve().await?;
        if correlated.turn_id.as_ref() != Some(&self.turn_id) {
            return Err(ServiceProtocolFailure::runtime_turn_mismatch());
        }

        Ok(VoiceCognitionOutcome {
            identity: self.identity,
            turn_id: correlated.turn_id,
            presentation: self.presentation,
            outcome: correlated.outcome,
        })
    }
}

/// Correlated result of a voice utterance entering cognition.
#[derive(Debug)]
pub struct VoiceCognitionOutcome {
    pub identity: VoiceTurnIdentity,
    pub turn_id: Option<TurnId>,
    pub presentation: VoicePresentationToken,
    pub outcome: ServiceWireOutcome,
}

impl VoiceCognitionOutcome {
    /// Whether presentation should be suppressed before starting/continuing TTS.
    pub fn presentation_cancelled(&self) -> bool {
        self.presentation.is_cancelled()
    }
}

/// Receipt for one presentation-level barge-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceInterruptionReceipt {
    pub session_id: SessionId,
    pub cancelled_through_generation: u64,
    pub capability: VoiceCancellationCapability,
    /// Remains false in this tranche because an admitted query may not yet have
    /// published `ResponseStarted`; emitting earlier would invert semantic order.
    pub semantic_event_emitted: bool,
    /// The runtime owner does not yet support cooperative cancellation of an
    /// already-admitted `Symthaea::process()` call.
    pub cognition_cancelled: bool,
}

/// One persistent voice conversation identity plus presentation-cancellation state.
#[derive(Debug, Clone)]
pub struct VoiceSession {
    session_id: SessionId,
    cancellation: Arc<VoiceCancellationState>,
}

impl VoiceSession {
    /// Mint an opaque voice session identity independent of wall-clock time.
    pub fn new() -> Result<Self, VoiceSessionError> {
        let id = SessionId::new(format!("voice-session:{}", Uuid::new_v4().simple()))?;
        Ok(Self::with_session_id(id))
    }

    /// Construct with an externally established semantic session identity.
    pub fn with_session_id(session_id: SessionId) -> Self {
        Self {
            session_id,
            cancellation: Arc::new(VoiceCancellationState::default()),
        }
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn cancellation_capability(&self) -> VoiceCancellationCapability {
        VoiceCancellationCapability::PresentationEligibilityOnly
    }

    /// Begin one utterance and mint a role-safe utterance identity.
    pub fn begin_turn(&self, ingress: VoiceIngress) -> Result<VoiceTurn, VoiceSessionError> {
        let previous = self
            .cancellation
            .current_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| VoiceSessionError::GenerationExhausted)?;
        let generation = previous + 1;
        let utterance_id =
            UtteranceId::new(format!("voice-utt:{}", Uuid::new_v4().simple()))?;
        let identity = VoiceTurnIdentity {
            session_id: self.session_id.clone(),
            utterance_id,
            ingress,
            presentation_generation: generation,
        };
        Ok(VoiceTurn {
            identity,
            presentation: VoicePresentationToken {
                generation,
                cancellation: Arc::clone(&self.cancellation),
            },
        })
    }

    /// Cancel presentation eligibility for every voice generation that has already
    /// begun. Future turns remain eligible.
    ///
    /// This operation is synchronous/lock-free and can therefore live on a fast
    /// barge-in path even while the cognition owner is occupied.
    pub fn interrupt_presentation(&self) -> VoiceInterruptionReceipt {
        let through = interrupt_started_presentations(&self.cancellation);

        VoiceInterruptionReceipt {
            session_id: self.session_id.clone(),
            cancelled_through_generation: through,
            capability: VoiceCancellationCapability::PresentationEligibilityOnly,
            semantic_event_emitted: false,
            cognition_cancelled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> VoiceSession {
        VoiceSession::with_session_id(SessionId::new("voice-session:test").unwrap())
    }

    #[test]
    fn voice_ingress_preserves_process_origin() {
        assert_eq!(
            VoiceIngress::LiveMicrophone.process_origin(),
            ProcessOrigin::VoiceTurn
        );
        assert_eq!(
            VoiceIngress::TranscribedAudio.process_origin(),
            ProcessOrigin::VoiceTranscription
        );
    }

    #[test]
    fn interruption_cancels_current_and_older_presentations_not_future_turns() {
        let session = session();
        let first = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let second = session.begin_turn(VoiceIngress::TranscribedAudio).unwrap();
        assert!(!first.presentation_token().is_cancelled());
        assert!(!second.presentation_token().is_cancelled());

        let receipt = session.interrupt_presentation();
        assert_eq!(receipt.cancelled_through_generation, 2);
        assert!(!receipt.semantic_event_emitted);
        assert!(!receipt.cognition_cancelled);
        assert!(first.presentation_token().is_cancelled());
        assert!(second.presentation_token().is_cancelled());

        let third = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        assert_eq!(third.presentation_token().generation(), 3);
        assert!(!third.presentation_token().is_cancelled());
    }

    #[test]
    fn every_turn_gets_role_safe_independent_utterance_identity() {
        let session = session();
        let first = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let second = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();

        assert_eq!(first.identity().session_id, second.identity().session_id);
        assert_ne!(first.identity().utterance_id, second.identity().utterance_id);
        assert_ne!(
            first.identity().presentation_generation,
            second.identity().presentation_generation
        );
    }

    #[test]
    fn turn_level_interrupt_uses_shared_session_cancellation_state() {
        let session = session();
        let turn = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let token = turn.presentation_token().clone();
        assert!(!token.is_cancelled());

        let through = token.interrupt_started();
        assert_eq!(through, 1);
        assert!(token.is_cancelled());
    }
}
