// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotic material handling finite-state machine.
//!
//! Models the pick-transport-place cycle common to automated fabrication
//! cells. The FSM enforces valid transitions and accumulates per-session
//! statistics.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Coordinates and metadata for a material pick location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PickTarget {
    /// XYZ position in millimetres.
    pub location: [f32; 3],
    /// Descriptive material identifier (e.g. "PLA", "PETG", "aluminium").
    pub material_type: String,
    /// Mass of the material in grams.
    pub weight_grams: f32,
}

/// Coordinates and orientation for a material placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceTarget {
    /// XYZ position in millimetres.
    pub location: [f32; 3],
    /// Roll/pitch/yaw orientation in radians.
    pub orientation: [f32; 3],
}

/// States of the material-handling FSM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialHandlerState {
    /// Waiting for a pick command.
    Idle,
    /// Moving toward the pick location.
    Approaching { target: PickTarget },
    /// End-effector is closing on the material.
    Gripping { target: PickTarget },
    /// Carrying material from source to destination.
    Transporting { from: PickTarget, to: PlaceTarget },
    /// Releasing the material at the destination.
    Placing { target: PlaceTarget },
    /// Returning to the home/idle position.
    Returning,
    /// An unrecoverable fault has occurred.
    Error { reason: String },
}

/// Events that drive state transitions.
#[derive(Debug, Clone)]
pub enum HandlerEvent {
    /// Begin a pick-and-place cycle.
    StartPick(PickTarget, PlaceTarget),
    /// Grip sensor confirmed material acquisition.
    GripConfirmed,
    /// Transport motion completed.
    TransportComplete,
    /// Place sensor confirmed release.
    PlaceConfirmed,
    /// Return motion completed.
    ReturnComplete,
    /// An external fault was detected.
    Fault(String),
    /// Reset from error state back to idle.
    Reset,
}

/// Errors produced by invalid FSM transitions.
#[derive(Debug, Clone)]
pub enum HandlerError {
    /// The requested event is not valid from the current state.
    InvalidTransition { from_state: String, event: String },
    /// A `StartPick` was issued while the handler is already busy.
    AlreadyBusy,
    /// A `Reset` was issued while not in the `Error` state.
    NotInErrorState,
}

impl fmt::Display for HandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { from_state, event } => {
                write!(f, "invalid transition: {event} from {from_state}")
            }
            Self::AlreadyBusy => write!(f, "handler is already busy"),
            Self::NotInErrorState => {
                write!(f, "reset requires error state")
            }
        }
    }
}

impl std::error::Error for HandlerError {}

/// Cumulative statistics for a handler session.
#[derive(Debug, Clone, Default)]
pub struct HandlerStats {
    /// Number of successfully completed place operations.
    pub total_handled: u64,
    /// Cumulative weight placed in grams.
    pub total_weight_grams: f64,
    /// Number of times the handler entered the error state.
    pub error_count: u32,
    /// Total number of events processed (successful or not).
    pub cycle_count: u64,
}

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------

/// Material-handling robot controller.
pub struct MaterialHandler {
    state: MaterialHandlerState,
    current_pick: Option<PickTarget>,
    current_place: Option<PlaceTarget>,
    stats: HandlerStats,
}

impl MaterialHandler {
    /// Create a new handler in the [`Idle`](MaterialHandlerState::Idle) state.
    pub fn new() -> Self {
        Self {
            state: MaterialHandlerState::Idle,
            current_pick: None,
            current_place: None,
            stats: HandlerStats::default(),
        }
    }

    /// Process an event and return the resulting state, or an error if the
    /// transition is invalid.
    pub fn process_event(
        &mut self,
        event: HandlerEvent,
    ) -> Result<&MaterialHandlerState, HandlerError> {
        self.stats.cycle_count += 1;

        // Fault from any state.
        if let HandlerEvent::Fault(ref reason) = event {
            self.stats.error_count += 1;
            self.state = MaterialHandlerState::Error {
                reason: reason.clone(),
            };
            return Ok(&self.state);
        }

        match (&self.state, &event) {
            // Idle + StartPick → Approaching
            (MaterialHandlerState::Idle, HandlerEvent::StartPick(pick, place)) => {
                self.current_pick = Some(pick.clone());
                self.current_place = Some(place.clone());
                self.state = MaterialHandlerState::Approaching {
                    target: pick.clone(),
                };
                Ok(&self.state)
            }

            // Approaching + GripConfirmed → Gripping → Transporting (auto)
            (MaterialHandlerState::Approaching { target }, HandlerEvent::GripConfirmed) => {
                let pick = target.clone();
                let place = self
                    .current_place
                    .clone()
                    .expect("place target set on StartPick");
                // Brief grip state, then automatically begin transport.
                self.state = MaterialHandlerState::Transporting {
                    from: pick,
                    to: place,
                };
                Ok(&self.state)
            }

            // Transporting + TransportComplete → Placing
            (MaterialHandlerState::Transporting { to, .. }, HandlerEvent::TransportComplete) => {
                self.state = MaterialHandlerState::Placing { target: to.clone() };
                Ok(&self.state)
            }

            // Placing + PlaceConfirmed → Returning (update stats)
            (MaterialHandlerState::Placing { .. }, HandlerEvent::PlaceConfirmed) => {
                self.stats.total_handled += 1;
                if let Some(ref pick) = self.current_pick {
                    self.stats.total_weight_grams += pick.weight_grams as f64;
                }
                self.state = MaterialHandlerState::Returning;
                Ok(&self.state)
            }

            // Returning + ReturnComplete → Idle
            (MaterialHandlerState::Returning, HandlerEvent::ReturnComplete) => {
                self.current_pick = None;
                self.current_place = None;
                self.state = MaterialHandlerState::Idle;
                Ok(&self.state)
            }

            // Error + Reset → Idle
            (MaterialHandlerState::Error { .. }, HandlerEvent::Reset) => {
                self.current_pick = None;
                self.current_place = None;
                self.state = MaterialHandlerState::Idle;
                Ok(&self.state)
            }

            // Reset from non-error is a specific error.
            (_, HandlerEvent::Reset) => Err(HandlerError::NotInErrorState),

            // StartPick while busy.
            (_, HandlerEvent::StartPick(_, _)) => Err(HandlerError::AlreadyBusy),

            // Everything else is an invalid transition.
            _ => Err(HandlerError::InvalidTransition {
                from_state: format!("{:?}", self.state),
                event: format!("{:?}", event),
            }),
        }
    }

    /// Current state of the handler.
    pub fn state(&self) -> &MaterialHandlerState {
        &self.state
    }

    /// Returns `true` when the handler is mid-cycle (not idle or errored).
    pub fn is_busy(&self) -> bool {
        !matches!(
            self.state,
            MaterialHandlerState::Idle | MaterialHandlerState::Error { .. }
        )
    }

    /// Cumulative session statistics.
    pub fn stats(&self) -> &HandlerStats {
        &self.stats
    }
}

impl Default for MaterialHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pick() -> PickTarget {
        PickTarget {
            location: [100.0, 200.0, 50.0],
            material_type: "PLA".into(),
            weight_grams: 12.5,
        }
    }

    fn sample_place() -> PlaceTarget {
        PlaceTarget {
            location: [300.0, 200.0, 10.0],
            orientation: [0.0, 0.0, 1.57],
        }
    }

    /// Drive the handler through a complete happy-path cycle.
    fn run_full_cycle(h: &mut MaterialHandler) {
        h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
            .unwrap();
        h.process_event(HandlerEvent::GripConfirmed).unwrap();
        h.process_event(HandlerEvent::TransportComplete).unwrap();
        h.process_event(HandlerEvent::PlaceConfirmed).unwrap();
        h.process_event(HandlerEvent::ReturnComplete).unwrap();
    }

    #[test]
    fn happy_path_cycle() {
        let mut h = MaterialHandler::new();
        assert!(matches!(h.state(), MaterialHandlerState::Idle));

        run_full_cycle(&mut h);

        assert!(matches!(h.state(), MaterialHandlerState::Idle));
        assert_eq!(h.stats().total_handled, 1);
    }

    #[test]
    fn invalid_transitions() {
        let mut h = MaterialHandler::new();
        // GripConfirmed from Idle is invalid.
        let err = h.process_event(HandlerEvent::GripConfirmed).unwrap_err();
        assert!(matches!(err, HandlerError::InvalidTransition { .. }));

        // TransportComplete from Idle is invalid.
        let err = h
            .process_event(HandlerEvent::TransportComplete)
            .unwrap_err();
        assert!(matches!(err, HandlerError::InvalidTransition { .. }));

        // StartPick while busy → AlreadyBusy.
        h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
            .unwrap();
        let err = h
            .process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
            .unwrap_err();
        assert!(matches!(err, HandlerError::AlreadyBusy));
    }

    #[test]
    fn error_recovery() {
        let mut h = MaterialHandler::new();
        h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
            .unwrap();
        h.process_event(HandlerEvent::Fault("gripper jam".into()))
            .unwrap();
        assert!(matches!(h.state(), MaterialHandlerState::Error { .. }));
        assert!(!h.is_busy());

        h.process_event(HandlerEvent::Reset).unwrap();
        assert!(matches!(h.state(), MaterialHandlerState::Idle));
    }

    #[test]
    fn stats_tracking() {
        let mut h = MaterialHandler::new();
        run_full_cycle(&mut h);
        assert_eq!(h.stats().total_handled, 1);
        assert!((h.stats().total_weight_grams - 12.5).abs() < f64::EPSILON);
        assert_eq!(h.stats().error_count, 0);
        assert_eq!(h.stats().cycle_count, 5);
    }

    #[test]
    fn busy_detection() {
        let mut h = MaterialHandler::new();
        assert!(!h.is_busy());

        h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
            .unwrap();
        assert!(h.is_busy());

        // Complete the cycle.
        h.process_event(HandlerEvent::GripConfirmed).unwrap();
        assert!(h.is_busy());
        h.process_event(HandlerEvent::TransportComplete).unwrap();
        assert!(h.is_busy());
        h.process_event(HandlerEvent::PlaceConfirmed).unwrap();
        assert!(h.is_busy()); // Returning
        h.process_event(HandlerEvent::ReturnComplete).unwrap();
        assert!(!h.is_busy());
    }

    #[test]
    fn multi_cycle_stats() {
        let mut h = MaterialHandler::new();
        run_full_cycle(&mut h);
        run_full_cycle(&mut h);
        run_full_cycle(&mut h);
        assert_eq!(h.stats().total_handled, 3);
        assert!((h.stats().total_weight_grams - 37.5).abs() < f64::EPSILON);
        assert_eq!(h.stats().cycle_count, 15);
    }

    #[test]
    fn serde_roundtrip() {
        let pick = sample_pick();
        let json = serde_json::to_string(&pick).unwrap();
        let back: PickTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(back.material_type, "PLA");
        assert!((back.weight_grams - 12.5).abs() < f32::EPSILON);

        let state = MaterialHandlerState::Transporting {
            from: sample_pick(),
            to: sample_place(),
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: MaterialHandlerState = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, MaterialHandlerState::Transporting { .. }));
    }

    #[test]
    fn fault_from_any_state() {
        // Fault should work from every non-error state.
        let states_to_try = vec![
            "idle",
            "approaching",
            "transporting",
            "placing",
            "returning",
        ];

        for label in states_to_try {
            let mut h = MaterialHandler::new();
            match label {
                "idle" => {}
                "approaching" => {
                    h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
                        .unwrap();
                }
                "transporting" => {
                    h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
                        .unwrap();
                    h.process_event(HandlerEvent::GripConfirmed).unwrap();
                }
                "placing" => {
                    h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
                        .unwrap();
                    h.process_event(HandlerEvent::GripConfirmed).unwrap();
                    h.process_event(HandlerEvent::TransportComplete).unwrap();
                }
                "returning" => {
                    h.process_event(HandlerEvent::StartPick(sample_pick(), sample_place()))
                        .unwrap();
                    h.process_event(HandlerEvent::GripConfirmed).unwrap();
                    h.process_event(HandlerEvent::TransportComplete).unwrap();
                    h.process_event(HandlerEvent::PlaceConfirmed).unwrap();
                }
                _ => unreachable!(),
            }

            let result = h.process_event(HandlerEvent::Fault(format!("fault from {label}")));
            assert!(result.is_ok(), "Fault should succeed from {label}");
            assert!(
                matches!(h.state(), MaterialHandlerState::Error { .. }),
                "should be in Error from {label}"
            );
        }
    }

    #[test]
    fn reset_from_non_error_fails() {
        let mut h = MaterialHandler::new();
        let err = h.process_event(HandlerEvent::Reset).unwrap_err();
        assert!(matches!(err, HandlerError::NotInErrorState));
    }

    #[test]
    fn handler_error_display() {
        let e = HandlerError::InvalidTransition {
            from_state: "Idle".into(),
            event: "GripConfirmed".into(),
        };
        assert!(e.to_string().contains("GripConfirmed"));
        assert!(e.to_string().contains("Idle"));

        let e = HandlerError::AlreadyBusy;
        assert!(e.to_string().contains("busy"));

        let e = HandlerError::NotInErrorState;
        assert!(e.to_string().contains("error state"));
    }
}
