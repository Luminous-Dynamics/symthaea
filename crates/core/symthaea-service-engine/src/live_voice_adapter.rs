// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Concrete binding between the service interruptible-presentation contract and
//! Symthaea's real-time [`LiveVoice`] backend.
//!
//! This module is feature-gated by `live-voice`. Batch/fallback voice paths remain
//! outside this adapter so callers can distinguish true mid-synthesis cancellation
//! capability from preflight-only presentation suppression.

use std::sync::Arc;

use symthaea::voice::live_voice::{LiveVoice, LiveVoiceSpeakOutcome, LiveVoiceStopHandle};

use crate::host::ServiceRuntimeHost;
use crate::voice_control::{VoicePresentationControl, VoiceStopCapability, VoiceStopOutcome};
use crate::voice_presenter::{InterruptibleVoiceRenderer, VoiceRenderCompletion};
use crate::voice_session::{
    VoiceCognitionTicket, VoiceInterruptionReceipt, VoiceSession,
};

impl VoiceStopCapability for LiveVoiceStopHandle {
    fn stop(&self) {
        LiveVoiceStopHandle::stop(self);
    }

    fn is_speaking(&self) -> bool {
        LiveVoiceStopHandle::is_speaking(self)
    }
}

impl InterruptibleVoiceRenderer for LiveVoice {
    type Error = anyhow::Error;

    fn stop_capability(&self) -> Arc<dyn VoiceStopCapability> {
        let capability: Arc<dyn VoiceStopCapability> = Arc::new(self.stop_handle());
        capability
    }

    fn speak_cancellable(
        &mut self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<VoiceRenderCompletion, Self::Error> {
        match LiveVoice::speak_cancellable(self, text, || cancelled())? {
            LiveVoiceSpeakOutcome::Completed => Ok(VoiceRenderCompletion::Completed),
            LiveVoiceSpeakOutcome::Cancelled => Ok(VoiceRenderCompletion::Cancelled),
        }
    }
}

/// Combined result of one live-voice barge-in.
///
/// Presentation eligibility is always cancelled first by the voice-session layer.
/// Semantic interruption and the active renderer stop are reported independently;
/// neither field implies cooperative cancellation of the admitted cognition turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveVoiceBargeInReceipt {
    pub interruption: VoiceInterruptionReceipt,
    pub presentation_stop: VoiceStopOutcome,
}

impl LiveVoiceBargeInReceipt {
    pub fn renderer_stop_requested(&self) -> bool {
        matches!(
            self.presentation_stop,
            VoiceStopOutcome::StopRequested { .. }
        )
    }

    pub fn cognition_cancelled(&self) -> bool {
        self.interruption.cognition_cancelled
    }
}

/// Barge in on one already-admitted voice cognition turn.
///
/// Ordering is intentional:
/// 1. `ticket.interrupt(host)` monotonically cancels presentation eligibility;
/// 2. it attempts the authoritative `VoiceInterrupted` semantic event;
/// 3. the exact cancelled generation range is applied to the active renderer.
///
/// The renderer stop is attempted even when semantic publication degrades, because
/// semantic observability must not resurrect speech that the human interrupted.
pub fn interrupt_live_voice_ticket(
    ticket: &VoiceCognitionTicket,
    host: &ServiceRuntimeHost,
    control: &VoicePresentationControl,
) -> LiveVoiceBargeInReceipt {
    let interruption = ticket.interrupt(host);
    let presentation_stop = control.apply_interruption(&interruption);
    LiveVoiceBargeInReceipt {
        interruption,
        presentation_stop,
    }
}

/// Presentation-only barge-in before or without an admitted cognition ticket.
///
/// This cancels every voice presentation generation that has already begun and
/// applies the same generation range to the active live renderer. Future voice turns
/// remain eligible. No semantic `VoiceInterrupted` event is invented because there
/// is no exact admitted `TurnId` to bind it to.
pub fn interrupt_live_voice_session(
    session: &VoiceSession,
    control: &VoicePresentationControl,
) -> LiveVoiceBargeInReceipt {
    let interruption = session.interrupt_presentation();
    let presentation_stop = control.apply_interruption(&interruption);
    LiveVoiceBargeInReceipt {
        interruption,
        presentation_stop,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::voice_session::VoiceIngress;
    use symthaea_interface_types::SessionId;

    fn assert_stop_capability<T: VoiceStopCapability>() {}
    fn assert_interruptible_renderer<T: InterruptibleVoiceRenderer<Error = anyhow::Error>>() {}

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

    #[test]
    fn concrete_live_voice_types_satisfy_service_capability_contracts() {
        assert_stop_capability::<LiveVoiceStopHandle>();
        assert_interruptible_renderer::<LiveVoice>();
    }

    #[test]
    fn session_barge_in_cancels_token_and_stops_matching_renderer() {
        let session = VoiceSession::with_session_id(
            SessionId::new("voice-session:live-barge-in-test").unwrap(),
        );
        let turn = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let control = VoicePresentationControl::default();
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop.clone();
        let _lease = control
            .try_register(turn.presentation_token().generation(), capability)
            .unwrap();

        let receipt = interrupt_live_voice_session(&session, &control);

        assert!(turn.presentation_token().is_cancelled());
        assert!(receipt.renderer_stop_requested());
        assert!(!receipt.cognition_cancelled());
        assert_eq!(stop.stops.load(Ordering::SeqCst), 1);
        assert!(!stop.is_speaking());
        assert!(!receipt.interruption.semantic_event_emitted);
        assert!(receipt.interruption.semantic_event_cursor.is_none());
    }
}
