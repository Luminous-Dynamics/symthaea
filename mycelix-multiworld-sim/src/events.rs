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
    /// A political faction emerged from inequality or instability.
    FactionEmerged,
    /// A faction dissolved due to declining strength.
    FactionDissolved,
    /// Armed or political conflict between factions.
    FactionConflict,
    /// A faction conflict was resolved (peacefully or by attrition).
    FactionResolution,
    /// Innovation stagnation detected (tech growth flatlined).
    InnovationStagnation,
    /// Conflict or tension between worlds.
    InterWorldConflict,
    /// An outer system colony was established.
    OuterSystemColony,
    /// Constitutional calcification: governance rigidity exceeds threshold.
    ConstitutionalCalcification,
    /// Genetic speciation divergence between worlds.
    GeneticSpeciation,
    /// Collective trauma accumulation in a world.
    TraumaAccumulation,
    /// Governance model transition (e.g., authoritarian -> democratic).
    GovernanceTransition,
    /// Non-linear cascade failure: 3+ simultaneous disasters amplify each other.
    SystemicCascadeFailure,
    /// A critical economic sector has zero skilled workers (knowledge bottleneck).
    SkillGapCrisis,
    /// Inter-generational knowledge loss: too few skilled workers to maintain tech level.
    KnowledgeLoss,
    /// A critical resource has been below 20% capacity for 12+ consecutive ticks.
    ResourceDepletionCrisis,
    /// Low or high morale is spreading through the population (social epidemic).
    MoraleContagion,
    /// Population has exceeded the dynamic carrying capacity of the world.
    CarryingCapacityExceeded,
    /// Resontia vault or maglev corridor construction completed.
    ResontiaInfrastructureBuilt,
    /// Resontia evacuation event (population moved to vaults).
    ResontiaEvacuation,
    /// Moral dilemma: disaster forced an ethical choice (save many vs protect vulnerable).
    MoralDilemma,
    /// Ethical crisis: sustained conflict between dominant ethical orientations.
    EthicalCrisis,
    /// Ethical synthesis: opposing frameworks resolved into a new hybrid orientation.
    EthicalSynthesis,
    /// Ethical fork: population split along ethical lines (triggers world split consideration).
    EthicalFork,
    /// Ethics shift: population mean ethical orientation drifted significantly from founding.
    EthicsShift,
    /// Moral revival: accumulated outrage + guilt triggered a collective return to virtue/duty.
    MoralRevival,
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
        let e = CivEvent::new(42, Some(1), CivEventType::Birth, "Agent 100 born on Moon");
        assert_eq!(e.tick, 42);
        assert_eq!(e.world_id, Some(1));
        assert_eq!(e.event_type, CivEventType::Birth);
        assert!(e.description.contains("Moon"));
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(
            format!("{}", CivEventType::EpochTransition),
            "EpochTransition"
        );
    }
}
