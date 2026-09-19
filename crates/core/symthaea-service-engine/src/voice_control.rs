// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Race-safe control plane for an actively presented voice generation.
//!
//! Presentation eligibility and backend stop authority are separate capabilities.
//! This module joins them without sharing mutable synthesis state: registration of
//! the active generation and barge-in cancellation are serialized through one small
//! mutex, while the actual stop capability is invoked after the mutex is released.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::voice_session::VoiceInterruptionReceipt;

/// Minimal cross-thread authority required to stop one active renderer.
///
/// Implementations should make `stop()` prompt and non-blocking. The service layer
/// deliberately cannot start speech or access renderer internals through this trait.
pub trait VoiceStopCapability: Send + Sync + 'static {
    fn stop(&self);
    fn is_speaking(&self) -> bool;
}

struct ActivePresentation {
    generation: u64,
    stop: Arc<dyn VoiceStopCapability>,
}

#[derive(Default)]
struct ControlState {
    cancelled_through: u64,
    active: Option<ActivePresentation>,
}

/// Cloneable control capability shared by the presentation path and barge-in path.
#[derive(Clone, Default)]
pub struct VoicePresentationControl {
    state: Arc<Mutex<ControlState>>,
}

/// Why a presentation generation could not become the active renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceRegistrationError {
    Cancelled {
        generation: u64,
        cancelled_through: u64,
    },
    AlreadyActive {
        active_generation: u64,
    },
    Poisoned,
}

impl fmt::Display for VoiceRegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled {
                generation,
                cancelled_through,
            } => write!(
                f,
                "voice generation {generation} is already cancelled through {cancelled_through}"
            ),
            Self::AlreadyActive { active_generation } => {
                write!(f, "voice generation {active_generation} is already active")
            }
            Self::Poisoned => write!(f, "voice presentation control mutex is poisoned"),
        }
    }
}

impl std::error::Error for VoiceRegistrationError {}

/// Result of applying a barge-in cancellation to the active renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceStopOutcome {
    /// No renderer was active when the cancellation was recorded.
    NoActivePresentation,
    /// An active renderer existed, but it belongs to a newer generation than the
    /// cancellation range and must not be stopped.
    ActiveGenerationNewer {
        active_generation: u64,
        cancelled_through: u64,
    },
    /// Stop was requested for the matching active generation.
    StopRequested {
        generation: u64,
        was_speaking: bool,
    },
    /// Control state could not be inspected. Presentation eligibility remains
    /// separately cancelled by `VoicePresentationToken`; this reports only failure
    /// to apply the backend stop request.
    ControlPoisoned,
}

/// RAII registration for exactly one active presentation generation.
///
/// Dropping an old lease cannot clear a newer active generation because clearing is
/// generation-checked under the control mutex.
pub struct ActiveVoicePresentationLease {
    control: VoicePresentationControl,
    generation: u64,
}

impl ActiveVoicePresentationLease {
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for ActiveVoicePresentationLease {
    fn drop(&mut self) {
        let Ok(mut state) = self.control.state.lock() else {
            return;
        };
        if state.active.as_ref().map(|active| active.generation) == Some(self.generation) {
            state.active = None;
        }
    }
}

impl VoicePresentationControl {
    /// Atomically register one renderer as active if this generation has not already
    /// been cancelled and no other renderer is active.
    pub fn try_register(
        &self,
        generation: u64,
        stop: Arc<dyn VoiceStopCapability>,
    ) -> Result<ActiveVoicePresentationLease, VoiceRegistrationError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| VoiceRegistrationError::Poisoned)?;

        if generation <= state.cancelled_through {
            return Err(VoiceRegistrationError::Cancelled {
                generation,
                cancelled_through: state.cancelled_through,
            });
        }
        if let Some(active) = state.active.as_ref() {
            return Err(VoiceRegistrationError::AlreadyActive {
                active_generation: active.generation,
            });
        }

        state.active = Some(ActivePresentation { generation, stop });
        Ok(ActiveVoicePresentationLease {
            control: self.clone(),
            generation,
        })
    }

    /// Record cancellation through `generation` and request stop if the currently
    /// active renderer falls within that cancelled range.
    ///
    /// Selection of the target occurs under the same mutex used by registration,
    /// which closes the register-vs-interrupt race. The stop callback itself runs
    /// after releasing the mutex so a backend cannot block control-state progress.
    pub fn interrupt_through(&self, generation: u64) -> VoiceStopOutcome {
        let target = {
            let Ok(mut state) = self.state.lock() else {
                return VoiceStopOutcome::ControlPoisoned;
            };
            state.cancelled_through = state.cancelled_through.max(generation);

            match state.active.as_ref() {
                None => return VoiceStopOutcome::NoActivePresentation,
                Some(active) if active.generation > state.cancelled_through => {
                    return VoiceStopOutcome::ActiveGenerationNewer {
                        active_generation: active.generation,
                        cancelled_through: state.cancelled_through,
                    };
                }
                Some(active) => Some((active.generation, Arc::clone(&active.stop))),
            }
        };

        let Some((active_generation, stop)) = target else {
            return VoiceStopOutcome::NoActivePresentation;
        };
        let was_speaking = stop.is_speaking();
        stop.stop();
        VoiceStopOutcome::StopRequested {
            generation: active_generation,
            was_speaking,
        }
    }

    /// Apply the exact presentation-generation range already recorded by a voice
    /// interruption receipt.
    pub fn apply_interruption(&self, receipt: &VoiceInterruptionReceipt) -> VoiceStopOutcome {
        self.interrupt_through(receipt.cancelled_through_generation)
    }

    pub fn cancelled_through(&self) -> Option<u64> {
        self.state.lock().ok().map(|state| state.cancelled_through)
    }

    pub fn active_generation(&self) -> Option<u64> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.active.as_ref().map(|active| active.generation))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

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
    fn cancellation_before_registration_rejects_generation() {
        let control = VoicePresentationControl::default();
        assert_eq!(
            control.interrupt_through(4),
            VoiceStopOutcome::NoActivePresentation
        );

        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop;
        let error = control.try_register(4, capability).err().unwrap();
        assert_eq!(
            error,
            VoiceRegistrationError::Cancelled {
                generation: 4,
                cancelled_through: 4,
            }
        );
        assert_eq!(control.active_generation(), None);
    }

    #[test]
    fn registration_before_interruption_requests_stop() {
        let control = VoicePresentationControl::default();
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop.clone();
        let lease = control.try_register(5, capability).unwrap();

        assert_eq!(control.active_generation(), Some(5));
        assert_eq!(
            control.interrupt_through(5),
            VoiceStopOutcome::StopRequested {
                generation: 5,
                was_speaking: true,
            }
        );
        assert_eq!(stop.stops.load(Ordering::SeqCst), 1);
        assert!(!stop.is_speaking());

        drop(lease);
        assert_eq!(control.active_generation(), None);
    }

    #[test]
    fn older_cancellation_does_not_stop_newer_active_generation() {
        let control = VoicePresentationControl::default();
        let stop = FakeStop::active();
        let capability: Arc<dyn VoiceStopCapability> = stop.clone();
        let _lease = control.try_register(9, capability).unwrap();

        assert_eq!(
            control.interrupt_through(8),
            VoiceStopOutcome::ActiveGenerationNewer {
                active_generation: 9,
                cancelled_through: 8,
            }
        );
        assert_eq!(stop.stops.load(Ordering::SeqCst), 0);
        assert!(stop.is_speaking());
    }

    #[test]
    fn only_one_active_renderer_can_register() {
        let control = VoicePresentationControl::default();
        let first = FakeStop::active();
        let second = FakeStop::active();
        let first_capability: Arc<dyn VoiceStopCapability> = first;
        let _lease = control.try_register(1, first_capability).unwrap();

        let second_capability: Arc<dyn VoiceStopCapability> = second;
        let error = control.try_register(2, second_capability).err().unwrap();
        assert_eq!(
            error,
            VoiceRegistrationError::AlreadyActive {
                active_generation: 1,
            }
        );
    }
}
