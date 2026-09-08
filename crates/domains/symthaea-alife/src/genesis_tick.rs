// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed tick authority for the flat Genesis behavioral event stream.
//!
//! `Population::step_social` historically used an unchecked `current_tick += 1`. Besides allowing
//! behavioral tick reuse after integer wrap, that increment happened after a potentially stateful
//! resource callback had already executed. A caller could therefore advance external environment
//! state and only then discover that no distinct behavioral tick remained representable.
//!
//! This module defines a narrow transition contract for that boundary. A social step first
//! [`GenesisTickCursorV1::preflight`]s the next tick without mutating cursor state. Only after any
//! external inputs needed for the step have been obtained does the caller [`GenesisTickCursorV1::commit`]
//! the non-serializable reservation. Exhaustion therefore fails before external side effects, and
//! a failed/stale commit never changes the cursor.
//!
//! This is a clock-transition primitive only. Its snapshot is semantic state for a future higher
//! execution capsule; it is not by itself proof that a persisted behavioral event prefix is
//! complete or authentic.

use serde::{Deserialize, Serialize};

/// Persistable semantic boundary for the next social tick.
///
/// The field is intentionally private. A deserialized value is not standalone behavioral-history
/// authority; future population/execution restore must bind it to qualified event-continuation
/// evidence before recreating a live cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisTickCursorSnapshotV1 {
    next_tick: u64,
}

impl GenesisTickCursorSnapshotV1 {
    pub fn next_tick(self) -> u64 {
        self.next_tick
    }
}

/// Ephemeral, non-serializable authorization for one social tick transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenesisTickReservationV1 {
    tick: u64,
    next_tick: u64,
}

impl GenesisTickReservationV1 {
    /// Tick number to stamp on all behavioral events produced by this social step.
    pub fn tick(self) -> u64 {
        self.tick
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenesisTickCursorErrorV1 {
    /// `u64::MAX` is a terminal *next-tick boundary*: emitting it would leave no representable
    /// successor boundary for exact continuation, so the step must fail before external input.
    Exhausted { next_tick: u64 },
    /// A reservation was already consumed or belongs to another cursor state.
    ReservationMismatch {
        expected_tick: u64,
        reserved_tick: u64,
    },
}

/// Monotonic social/Genesis tick authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GenesisTickCursorV1 {
    next_tick: u64,
}

impl GenesisTickCursorV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> GenesisTickCursorSnapshotV1 {
        GenesisTickCursorSnapshotV1 {
            next_tick: self.next_tick,
        }
    }

    /// Validate that one more social tick can be represented without changing state.
    ///
    /// Call this **before** invoking a stateful resource/environment provider. At the terminal
    /// boundary this returns `Exhausted` and leaves the cursor untouched.
    pub fn preflight(&self) -> Result<GenesisTickReservationV1, GenesisTickCursorErrorV1> {
        let next_tick = self.next_tick.checked_add(1).ok_or(
            GenesisTickCursorErrorV1::Exhausted {
                next_tick: self.next_tick,
            },
        )?;
        Ok(GenesisTickReservationV1 {
            tick: self.next_tick,
            next_tick,
        })
    }

    /// Commit one previously preflighted social tick.
    ///
    /// Reservations are intentionally ephemeral and non-serializable. Reusing a consumed/stale
    /// reservation fails without changing state.
    pub fn commit(
        &mut self,
        reservation: GenesisTickReservationV1,
    ) -> Result<u64, GenesisTickCursorErrorV1> {
        if reservation.tick != self.next_tick {
            return Err(GenesisTickCursorErrorV1::ReservationMismatch {
                expected_tick: self.next_tick,
                reserved_tick: reservation.tick,
            });
        }
        self.next_tick = reservation.next_tick;
        Ok(reservation.tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_is_side_effect_free_until_commit() {
        let mut cursor = GenesisTickCursorV1::new();
        let reservation = cursor.preflight().expect("tick zero available");
        assert_eq!(reservation.tick(), 0);
        assert_eq!(cursor.snapshot().next_tick(), 0);

        assert_eq!(cursor.commit(reservation), Ok(0));
        assert_eq!(cursor.snapshot().next_tick(), 1);
    }

    #[test]
    fn ordinary_ticks_are_monotonic_and_never_reused() {
        let mut cursor = GenesisTickCursorV1::new();
        for expected in 0..1_000u64 {
            let reservation = cursor.preflight().expect("tick available");
            assert_eq!(reservation.tick(), expected);
            assert_eq!(cursor.commit(reservation), Ok(expected));
        }
        assert_eq!(cursor.snapshot().next_tick(), 1_000);
    }

    #[test]
    fn stale_reservation_fails_without_advancing_state() {
        let mut cursor = GenesisTickCursorV1::new();
        let reservation = cursor.preflight().expect("reservation");
        assert_eq!(cursor.commit(reservation), Ok(0));
        let before = cursor.snapshot();
        assert_eq!(
            cursor.commit(reservation),
            Err(GenesisTickCursorErrorV1::ReservationMismatch {
                expected_tick: 1,
                reserved_tick: 0,
            })
        );
        assert_eq!(cursor.snapshot(), before);
    }

    #[test]
    fn terminal_boundary_fails_closed_before_emitting_an_uncontinuable_tick() {
        let cursor = GenesisTickCursorV1 {
            next_tick: u64::MAX,
        };
        let before = cursor.snapshot();
        assert_eq!(
            cursor.preflight(),
            Err(GenesisTickCursorErrorV1::Exhausted {
                next_tick: u64::MAX,
            })
        );
        assert_eq!(cursor.snapshot(), before);
    }

    #[test]
    fn penultimate_tick_is_valid_and_commits_to_terminal_boundary() {
        let mut cursor = GenesisTickCursorV1 {
            next_tick: u64::MAX - 1,
        };
        let reservation = cursor.preflight().expect("last representable committed step");
        assert_eq!(reservation.tick(), u64::MAX - 1);
        assert_eq!(cursor.commit(reservation), Ok(u64::MAX - 1));
        assert_eq!(cursor.snapshot().next_tick(), u64::MAX);
        assert!(matches!(
            cursor.preflight(),
            Err(GenesisTickCursorErrorV1::Exhausted { .. })
        ));
    }
}
