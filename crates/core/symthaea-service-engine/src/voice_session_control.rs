// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Session-keyed active-presentation control for service voice.
//!
//! `VoicePresentationToken` remains the monotonic source of presentation
//! cancellation truth for one session. This control plane owns only the *currently
//! active renderer* and serializes registration against barge-in stop selection.
//! It intentionally retains no `cancelled_through` history of its own, avoiding
//! cross-session generation bleed when a new `VoiceSession` restarts at generation 1.

use std::fmt;
use std::sync::{Arc, Mutex};

use symthaea_interface_types::SessionId;

use crate::voice_control::VoiceStopCapability;
use crate::voice_session::{VoiceInterruptionReceipt, VoicePresentationToken, VoiceTurnIdentity};

/// Exact identity of one presentation generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoicePresentationKey {
    session_id: SessionId,
    generation: u64,
}

impl VoicePresentationKey {
    pub fn from_turn(
        identity: &VoiceTurnIdentity,
        presentation: &VoicePresentationToken,
    ) -> Result<Self, SessionVoiceRegistrationError> {
        if identity.presentation_generation != presentation.generation() {
            return Err(SessionVoiceRegistrationError::IdentityMismatch {
                identity_generation: identity.presentation_generation,
                token_generation: presentation.generation(),
            });
        }
        Ok(Self {
            session_id: identity.session_id.clone(),
            generation: presentation.generation(),
        })
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

struct ActivePresentation {
    key: VoicePresentationKey,
    stop: Arc<dyn VoiceStopCapability>,
}

#[derive(Default)]
struct ControlState {
    active: Option<ActivePresentation>,
}

/// Cloneable fast-path control capability for at most one active renderer.
#[derive(Clone, Default)]
pub struct SessionVoicePresentationControl {
    state: Arc<Mutex<ControlState>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionVoiceRegistrationError {
    Cancelled {
        key: VoicePresentationKey,
    },
    IdentityMismatch {
        identity_generation: u64,
        token_generation: u64,
    },
    AlreadyActive {
        active: VoicePresentationKey,
    },
    Poisoned,
}

impl fmt::Display for SessionVoiceRegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { key } => write!(
                f,
                "voice presentation {} generation {} is already cancelled",
                key.session_id, key.generation
            ),
            Self::IdentityMismatch {
                identity_generation,
                token_generation,
            } => write!(
                f,
                "voice turn/presentation generation mismatch: identity={identity_generation}, token={token_generation}"
            ),
            Self::AlreadyActive { active } => write!(
                f,
                "voice presentation {} generation {} is already active",
                active.session_id, active.generation
            ),
            Self::Poisoned => write!(f, "session voice presentation control mutex is poisoned"),
        }
    }
}

impl std::error::Error for SessionVoiceRegistrationError {}

/// Result of applying a session-scoped interruption receipt to the active renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionVoiceStopOutcome {
    NoActivePresentation,
    ActiveSessionDifferent {
        active_session_id: SessionId,
        interrupted_session_id: SessionId,
    },
    ActiveGenerationNewer {
        session_id: SessionId,
        active_generation: u64,
        cancelled_through: u64,
    },
    StopRequested {
        session_id: SessionId,
        generation: u64,
        was_speaking: bool,
    },
    ControlPoisoned,
}

/// RAII registration for exactly one `(SessionId, generation)`.
pub struct SessionVoicePresentationLease {
    control: SessionVoicePresentationControl,
    key: VoicePresentationKey,
}

impl SessionVoicePresentationLease {
    pub fn key(&self) -> &VoicePresentationKey {
        &self.key
    }
}

impl Drop for SessionVoicePresentationLease {
    fn drop(&mut self) {
        let Ok(mut state) = self.control.state.lock() else {
            return;
        };
        if state.active.as_ref().map(|active| &active.key) == Some(&self.key) {
            state.active = None;
        }
    }
}

impl SessionVoicePresentationControl {
    /// Register the renderer for this exact turn/presentation pair.
    ///
    /// Cancellation history is not copied into this control. The session-owned
    /// presentation token is checked under the same mutex that installs `active`, so
    /// a token already cancelled before registration can never become active here.
    pub fn try_register(
        &self,
        identity: &VoiceTurnIdentity,
        presentation: &VoicePresentationToken,
        stop: Arc<dyn VoiceStopCapability>,
    ) -> Result<SessionVoicePresentationLease, SessionVoiceRegistrationError> {
        let key = VoicePresentationKey::from_turn(identity, presentation)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| SessionVoiceRegistrationError::Poisoned)?;

        if presentation.is_cancelled() {
            return Err(SessionVoiceRegistrationError::Cancelled { key });
        }
        if let Some(active) = state.active.as_ref() {
            return Err(SessionVoiceRegistrationError::AlreadyActive {
                active: active.key.clone(),
            });
        }

        state.active = Some(ActivePresentation {
            key: key.clone(),
            stop,
        });
        Ok(SessionVoicePresentationLease {
            control: self.clone(),
            key,
        })
    }

    /// Apply an interruption only to an active presentation from the same session.
    ///
    /// Selection occurs under the registration mutex, but the backend stop callback
    /// runs after the lock is released. A receipt from another voice session can
    /// never stop or poison the active session's generation namespace.
    pub fn apply_interruption(&self, receipt: &VoiceInterruptionReceipt) -> SessionVoiceStopOutcome {
        let target = {
            let Ok(state) = self.state.lock() else {
                return SessionVoiceStopOutcome::ControlPoisoned;
            };
            let Some(active) = state.active.as_ref() else {
                return SessionVoiceStopOutcome::NoActivePresentation;
            };

            if active.key.session_id != receipt.session_id {
                return SessionVoiceStopOutcome::ActiveSessionDifferent {
                    active_session_id: active.key.session_id.clone(),
                    interrupted_session_id: receipt.session_id.clone(),
                };
            }
            if active.key.generation > receipt.cancelled_through_generation {
                return SessionVoiceStopOutcome::ActiveGenerationNewer {
                    session_id: active.key.session_id.clone(),
                    active_generation: active.key.generation,
                    cancelled_through: receipt.cancelled_through_generation,
                };
            }

            Some((active.key.clone(), Arc::clone(&active.stop)))
        };

        let Some((key, stop)) = target else {
            return SessionVoiceStopOutcome::NoActivePresentation;
        };
        let was_speaking = stop.is_speaking();
        stop.stop();
        SessionVoiceStopOutcome::StopRequested {
            session_id: key.session_id,
            generation: key.generation,
            was_speaking,
        }
    }

    pub fn active_key(&self) -> Option<VoicePresentationKey> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.active.as_ref().map(|active| active.key.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::voice_session::{VoiceIngress, VoiceSession};

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

    fn session(id: &str) -> VoiceSession {
        VoiceSession::with_session_id(SessionId::new(id).unwrap())
    }

    #[test]
    fn pre_cancelled_session_token_cannot_register() {
        let session = session("voice-session:a");
        let turn = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let receipt = session.interrupt_presentation();
        assert_eq!(receipt.cancelled_through_generation, 1);

        let control = SessionVoicePresentationControl::default();
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop;
        let error = control
            .try_register(turn.identity(), turn.presentation_token(), capability)
            .err()
            .unwrap();
        assert!(matches!(error, SessionVoiceRegistrationError::Cancelled { .. }));
        assert!(control.active_key().is_none());
    }

    #[test]
    fn interruption_from_another_session_never_stops_active_renderer() {
        let session_a = session("voice-session:a");
        let session_b = session("voice-session:b");
        let turn_a = session_a.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let _turn_b = session_b.begin_turn(VoiceIngress::LiveMicrophone).unwrap();

        let control = SessionVoicePresentationControl::default();
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop.clone();
        let _lease = control
            .try_register(turn_a.identity(), turn_a.presentation_token(), capability)
            .unwrap();

        let receipt_b = session_b.interrupt_presentation();
        assert_eq!(
            control.apply_interruption(&receipt_b),
            SessionVoiceStopOutcome::ActiveSessionDifferent {
                active_session_id: SessionId::new("voice-session:a").unwrap(),
                interrupted_session_id: SessionId::new("voice-session:b").unwrap(),
            }
        );
        assert_eq!(stop.stops.load(Ordering::SeqCst), 0);
        assert!(stop.is_speaking());
    }

    #[test]
    fn cancelled_generation_in_old_session_does_not_poison_new_session_generation_one() {
        let session_a = session("voice-session:a");
        for _ in 0..5 {
            session_a.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        }
        let receipt_a = session_a.interrupt_presentation();
        assert_eq!(receipt_a.cancelled_through_generation, 5);

        let session_b = session("voice-session:b");
        let turn_b = session_b.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        assert_eq!(turn_b.presentation_token().generation(), 1);
        assert!(!turn_b.presentation_token().is_cancelled());

        let control = SessionVoicePresentationControl::default();
        assert_eq!(
            control.apply_interruption(&receipt_a),
            SessionVoiceStopOutcome::NoActivePresentation
        );
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop;
        let _lease = control
            .try_register(turn_b.identity(), turn_b.presentation_token(), capability)
            .unwrap();
        assert_eq!(control.active_key().unwrap().generation(), 1);
    }

    #[test]
    fn same_session_interruption_stops_matching_active_generation() {
        let session = session("voice-session:a");
        let turn = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let control = SessionVoicePresentationControl::default();
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop.clone();
        let _lease = control
            .try_register(turn.identity(), turn.presentation_token(), capability)
            .unwrap();

        let receipt = session.interrupt_presentation();
        assert_eq!(
            control.apply_interruption(&receipt),
            SessionVoiceStopOutcome::StopRequested {
                session_id: SessionId::new("voice-session:a").unwrap(),
                generation: 1,
                was_speaking: true,
            }
        );
        assert_eq!(stop.stops.load(Ordering::SeqCst), 1);
        assert!(!stop.is_speaking());
    }

    #[test]
    fn old_lease_cannot_clear_different_session_active_key() {
        let session_a = session("voice-session:a");
        let session_b = session("voice-session:b");
        let turn_a = session_a.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let turn_b = session_b.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        let control = SessionVoicePresentationControl::default();

        let stop_a = FakeStop::active();
        let capability_a: Arc<dyn VoiceStopCapability> = stop_a;
        let lease_a = control
            .try_register(turn_a.identity(), turn_a.presentation_token(), capability_a)
            .unwrap();
        drop(lease_a);

        let stop_b = FakeStop::active();
        let capability_b: Arc<dyn VoiceStopCapability> = stop_b;
        let _lease_b = control
            .try_register(turn_b.identity(), turn_b.presentation_token(), capability_b)
            .unwrap();

        assert_eq!(
            control.active_key(),
            Some(VoicePresentationKey {
                session_id: SessionId::new("voice-session:b").unwrap(),
                generation: 1,
            })
        );
    }
}
