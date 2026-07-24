// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prospective musical memory: promises made by a composition and due later.
//!
//! Long-range form is not only a sequence of sections. It is a set of
//! obligations: return an opening subject, resolve an altered tone, arrive in
//! a destination key, reserve a voice until the climax, or restore an eroded
//! identity. This module records those obligations independently from any one
//! form constructor so a planner, Studio edit, or Symthaea cognitive loop can
//! inspect and fulfil them explicitly.

use crate::harmony::Key;
use crate::rhythm::Duration;
use crate::spelling::AlteredDegree;
use serde::{Deserialize, Serialize};

/// How a returning motif is expected to relate to its earlier identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnTransformation {
    Literal,
    Transposed,
    Inverted,
    Augmented,
    Diminished,
    Fragmented,
    Restored,
}

/// A long-range musical promise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObligationKind {
    /// Bring back named thematic material by the due point.
    ReturnMotif {
        motif_id: String,
        transformation: ReturnTransformation,
    },
    /// Establish a destination key or tonal center.
    ReachKey { key: Key },
    /// Produce a cadence whose arrival degree is explicit.
    Cadence { arrival_degree: AlteredDegree },
    /// Resolve a chromatically altered tendency tone.
    ResolveAlteredDegree { degree: AlteredDegree },
    /// Give a named voice or role its promised entrance.
    EnterVoice { voice_id: String },
    /// Reach the planned large-scale climax.
    ReachClimax,
    /// Re-establish an identity that was intentionally weakened or eroded.
    RestoreIdentity { identity_id: String },
    /// An application-defined promise retained in the same auditable ledger.
    Custom { label: String },
}

/// Current state of a compositional obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObligationStatus {
    Pending,
    Fulfilled,
    Waived,
}

/// One explicit prospective-memory item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionalObligation {
    pub id: u64,
    pub created_at: Duration,
    pub due_by: Duration,
    /// Relative importance in [0, 1].
    pub priority: f32,
    pub kind: ObligationKind,
    pub status: ObligationStatus,
    /// Optional evidence or explanation attached when the state changes.
    pub resolution_note: Option<String>,
}

impl CompositionalObligation {
    pub fn new(
        id: u64,
        created_at: Duration,
        due_by: Duration,
        priority: f32,
        kind: ObligationKind,
    ) -> Self {
        assert!(
            due_by.beats() >= created_at.beats(),
            "an obligation cannot be due before it is created"
        );
        Self {
            id,
            created_at,
            due_by,
            priority: priority.clamp(0.0, 1.0),
            kind,
            status: ObligationStatus::Pending,
            resolution_note: None,
        }
    }

    pub fn is_pending(&self) -> bool {
        self.status == ObligationStatus::Pending
    }

    pub fn is_due_at(&self, now: Duration) -> bool {
        self.is_pending() && now.beats() >= self.due_by.beats()
    }

    pub fn fulfil(&mut self, note: impl Into<String>) {
        self.status = ObligationStatus::Fulfilled;
        self.resolution_note = Some(note.into());
    }

    pub fn waive(&mut self, reason: impl Into<String>) {
        self.status = ObligationStatus::Waived;
        self.resolution_note = Some(reason.into());
    }
}

/// Aggregate due-state for a cognitive planner.
///
/// `weighted_pressure` is the maximum priority-weighted progress toward a
/// pending obligation's deadline. It is an urgency signal, not a quality
/// score: overdue high-priority promises reach 1.0, while newly created or
/// low-priority promises exert less pressure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObligationPressure {
    pub pending_count: usize,
    pub overdue_count: usize,
    pub weighted_pressure: f32,
    pub next_due_in_beats: Option<f64>,
}

/// A serializable ledger of long-range musical obligations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ObligationLedger {
    items: Vec<CompositionalObligation>,
}

impl ObligationLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn items(&self) -> &[CompositionalObligation] {
        &self.items
    }

    pub fn add(&mut self, obligation: CompositionalObligation) {
        assert!(
            !self.items.iter().any(|item| item.id == obligation.id),
            "obligation identifiers must be unique"
        );
        self.items.push(obligation);
    }

    pub fn get(&self, id: u64) -> Option<&CompositionalObligation> {
        self.items.iter().find(|item| item.id == id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut CompositionalObligation> {
        self.items.iter_mut().find(|item| item.id == id)
    }

    /// Pending obligations ordered by priority (highest first), then due time.
    pub fn pending(&self) -> Vec<&CompositionalObligation> {
        let mut pending: Vec<_> = self.items.iter().filter(|item| item.is_pending()).collect();
        pending.sort_by(|a, b| {
            b.priority
                .total_cmp(&a.priority)
                .then_with(|| a.due_by.beats().total_cmp(&b.due_by.beats()))
        });
        pending
    }

    /// Pending obligations whose due point has been reached.
    pub fn overdue_at(&self, now: Duration) -> Vec<&CompositionalObligation> {
        self.pending()
            .into_iter()
            .filter(|item| item.is_due_at(now))
            .collect()
    }

    /// Summarize how strongly pending promises should influence a planner now.
    pub fn pressure_at(&self, now: Duration) -> ObligationPressure {
        let pending = self.pending();
        let now_beats = now.beats();
        let overdue_count = pending.iter().filter(|item| item.is_due_at(now)).count();
        let weighted_pressure = pending
            .iter()
            .map(|item| {
                let created = item.created_at.beats();
                let due = item.due_by.beats();
                let progress = if now_beats >= due {
                    1.0
                } else if due <= created {
                    1.0
                } else {
                    ((now_beats - created) / (due - created)).clamp(0.0, 1.0)
                };
                item.priority * progress as f32
            })
            .fold(0.0_f32, f32::max);
        let next_due_in_beats = pending
            .iter()
            .map(|item| (item.due_by.beats() - now_beats).max(0.0))
            .min_by(|left, right| left.total_cmp(right));

        ObligationPressure {
            pending_count: pending.len(),
            overdue_count,
            weighted_pressure,
            next_due_in_beats,
        }
    }

    /// Obligations that have been explicitly fulfilled.
    pub fn fulfilled(&self) -> Vec<&CompositionalObligation> {
        self.items
            .iter()
            .filter(|item| item.status == ObligationStatus::Fulfilled)
            .collect()
    }

    /// Obligations that remain pending or were waived rather than fulfilled.
    pub fn unresolved(&self) -> Vec<&CompositionalObligation> {
        self.items
            .iter()
            .filter(|item| item.status != ObligationStatus::Fulfilled)
            .collect()
    }

    /// True only when every recorded promise was explicitly fulfilled.
    pub fn all_fulfilled(&self) -> bool {
        !self.items.is_empty()
            && self
                .items
                .iter()
                .all(|item| item.status == ObligationStatus::Fulfilled)
    }

    pub fn fulfil(&mut self, id: u64, note: impl Into<String>) -> bool {
        if let Some(item) = self.get_mut(id) {
            item.fulfil(note);
            true
        } else {
            false
        }
    }

    pub fn waive(&mut self, id: u64, reason: impl Into<String>) -> bool {
        if let Some(item) = self.get_mut(id) {
            item.waive(reason);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    #[test]
    fn ledger_orders_pending_promises_by_priority() {
        let mut ledger = ObligationLedger::new();
        ledger.add(CompositionalObligation::new(
            1,
            Duration::zero(),
            Duration::new(16, 1),
            0.5,
            ObligationKind::ReachKey {
                key: Key::major(PitchClass::G),
            },
        ));
        ledger.add(CompositionalObligation::new(
            2,
            Duration::zero(),
            Duration::new(32, 1),
            0.9,
            ObligationKind::ReturnMotif {
                motif_id: "opening".into(),
                transformation: ReturnTransformation::Literal,
            },
        ));

        let pending = ledger.pending();
        assert_eq!(pending[0].id, 2);
        assert_eq!(pending[1].id, 1);
    }

    #[test]
    fn due_promises_remain_visible_until_resolved() {
        let mut ledger = ObligationLedger::new();
        ledger.add(CompositionalObligation::new(
            7,
            Duration::zero(),
            Duration::new(8, 1),
            1.0,
            ObligationKind::ReachClimax,
        ));

        assert!(ledger.overdue_at(Duration::new(7, 1)).is_empty());
        assert_eq!(ledger.overdue_at(Duration::new(8, 1))[0].id, 7);
        assert!(ledger.fulfil(7, "climax reached in the upper voice"));
        assert!(ledger.overdue_at(Duration::new(9, 1)).is_empty());
    }

    #[test]
    fn priority_is_bounded() {
        let high = CompositionalObligation::new(
            1,
            Duration::zero(),
            Duration::quarter(),
            4.0,
            ObligationKind::Custom {
                label: "test".into(),
            },
        );
        let low = CompositionalObligation::new(
            2,
            Duration::zero(),
            Duration::quarter(),
            -2.0,
            ObligationKind::Custom {
                label: "test".into(),
            },
        );
        assert_eq!(high.priority, 1.0);
        assert_eq!(low.priority, 0.0);
    }
    #[test]
    fn pressure_rises_as_a_high_priority_deadline_approaches() {
        let mut ledger = ObligationLedger::new();
        ledger.add(CompositionalObligation::new(
            12,
            Duration::zero(),
            Duration::new(8, 1),
            0.8,
            ObligationKind::ReturnMotif {
                motif_id: "opening".into(),
                transformation: ReturnTransformation::Literal,
            },
        ));

        let early = ledger.pressure_at(Duration::new(2, 1));
        let late = ledger.pressure_at(Duration::new(7, 1));
        let overdue = ledger.pressure_at(Duration::new(9, 1));

        assert!(late.weighted_pressure > early.weighted_pressure);
        assert_eq!(overdue.weighted_pressure, 0.8);
        assert_eq!(overdue.overdue_count, 1);
        assert_eq!(overdue.next_due_in_beats, Some(0.0));
    }

    #[test]
    fn fulfilled_promises_exert_no_pressure() {
        let mut ledger = ObligationLedger::new();
        ledger.add(CompositionalObligation::new(
            2,
            Duration::zero(),
            Duration::quarter(),
            1.0,
            ObligationKind::ReachClimax,
        ));
        assert!(ledger.fulfil(2, "climax reached"));

        let pressure = ledger.pressure_at(Duration::new(4, 1));
        assert_eq!(pressure.pending_count, 0);
        assert_eq!(pressure.overdue_count, 0);
        assert_eq!(pressure.weighted_pressure, 0.0);
        assert_eq!(pressure.next_due_in_beats, None);
    }

    #[test]
    fn ledger_reports_resolution_state() {
        let mut ledger = ObligationLedger::new();
        ledger.add(CompositionalObligation::new(
            1,
            Duration::zero(),
            Duration::quarter(),
            1.0,
            ObligationKind::ReachClimax,
        ));

        assert!(!ledger.all_fulfilled());
        assert_eq!(ledger.unresolved().len(), 1);
        assert!(ledger.fulfil(1, "structural peak reached"));
        assert!(ledger.all_fulfilled());
        assert_eq!(ledger.fulfilled().len(), 1);
        assert!(ledger.unresolved().is_empty());
    }
}
