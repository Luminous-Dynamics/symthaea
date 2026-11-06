//! In-memory event processor with an intentionally missing deduplication guard.
//!
//! This is a deterministic development seed for duplicate delivery. It does not
//! model persistence, restart, atomic acknowledgement, or concurrent workers.

use ceb_event_replay_contract::{AppliedEffect, DeliveryOutcome, Event};

#[derive(Debug, Default)]
pub struct EventProcessor {
    effects: Vec<AppliedEffect>,
}

impl EventProcessor {
    /// Apply every delivery as a new effect.
    ///
    /// The defect is intentional: replaying the same event identity appends a
    /// second effect. A future task specification may require an idempotent
    /// identity/payload decision, but no such repair is included here.
    pub fn deliver(&mut self, event: Event) -> DeliveryOutcome {
        let effect = AppliedEffect {
            event_id: event.event_id,
            payload: event.payload,
            ordinal: self.effects.len(),
        };
        self.effects.push(effect.clone());
        DeliveryOutcome::Applied(effect)
    }

    /// Current in-memory effects, in application order.
    pub fn effects(&self) -> &[AppliedEffect] {
        &self.effects
    }
}
