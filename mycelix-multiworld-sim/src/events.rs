// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civilization event types and logging.

use serde::{Deserialize, Serialize};

/// A discrete event in the civilization timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivEvent {
    /// Simulation tick when this event occurred.
    pub tick: u32,
    /// World where the event occurred (None for interworld events).
    pub world_id: Option<u32>,
    /// Event classification.
    pub event_type: CivEventType,
    /// Human-readable description.
    pub description: String,
}

/// Classification of civilization events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CivEventType {
    Birth,
    Death,
    Migration,
    EpochTransition,
    ConstitutionalAmendment,
    TradeEstablished,
    EpidemicStart,
    EpidemicEnd,
    ResourceCrisis,
    OppressionAlert,
    CulturalShift,
    WorldFounded,
    InnovationBreakthrough,
    GeneticAlert,
    HarmonyMilestone,
    EmergencyDeclared,
    /// Thrill-seeking health risk event (adrenaline extraction).
    ThrillIncident,
    /// Rogue bio-hacking triggered a novel pathogen.
    BiohackIncident,
    /// Peer-to-peer teaching interaction completed.
    TeachingInteraction,
    /// Community skill crisis: mean sector skill below survival threshold.
    SkillCrisis,
}

impl CivEvent {
    /// Create a new event.
    pub fn new(
        tick: u32,
        world_id: Option<u32>,
        event_type: CivEventType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            tick,
            world_id,
            event_type,
            description: description.into(),
        }
    }
}

impl std::fmt::Display for CivEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_logs_correctly() {
        let e = CivEvent::new(
            42,
            Some(1),
            CivEventType::Birth,
            "Agent 100 born on Moon",
        );
        assert_eq!(e.tick, 42);
        assert_eq!(e.world_id, Some(1));
        assert_eq!(e.event_type, CivEventType::Birth);
        assert!(e.description.contains("Moon"));
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(format!("{}", CivEventType::EpochTransition), "EpochTransition");
    }
}
