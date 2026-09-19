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

use crate::voice_control::VoiceStopCapability;
use crate::voice_presenter::{InterruptibleVoiceRenderer, VoiceRenderCompletion};

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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_stop_capability<T: VoiceStopCapability>() {}
    fn assert_interruptible_renderer<T: InterruptibleVoiceRenderer<Error = anyhow::Error>>() {}

    #[test]
    fn concrete_live_voice_types_satisfy_service_capability_contracts() {
        assert_stop_capability::<LiveVoiceStopHandle>();
        assert_interruptible_renderer::<LiveVoice>();
    }
}
