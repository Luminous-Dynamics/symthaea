// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix Bridge: re-exports production types from bridge-common.
//!
//! The simulation uses the SAME types that Mycelix governance runs on
//! Holochain. This means parameter extraction from the sim produces
//! values that deploy directly to production without translation.
//!
//! bridge-common is imported with `default-features = false` — no HDK,
//! no Holochain runtime. Pure Rust + serde + blake3.

// Re-export the types the sim needs
pub use mycelix_bridge_common::consciousness_profile::ConsciousnessProfile;
pub use mycelix_bridge_common::earth_colony_protocol::PlanetaryBody;
pub use mycelix_bridge_common::sovereign_gate::CivicTier;
// consciousness_sync is behind the "federated" feature gate in bridge-common.
// Stub types for downstream compatibility until federated module is restored.
#[derive(Debug, Clone, Default)]
pub struct ConsciousnessSynchronizer;
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessMetricPacket {
    pub source_world: String,
    pub phi: f64,
    pub coherence: f64,
    pub timestamp: u32,
}
#[derive(Debug, Clone, Default)]
pub struct SyncStrategy;
#[derive(Debug, Clone)]
pub struct SyncEvent {
    pub source: String,
    pub target: String,
    pub delta_phi: f64,
}
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PlanetaryConsciousnessState {
    pub body: String,
    pub collective_phi: f64,
    pub population: usize,
    pub coherence: f64,
}
pub use mycelix_bridge_common::collective_phi::{
    AgentConsciousnessVector, CollectivePhiEngine, CollectivePhiResult,
};
pub use mycelix_bridge_common::planetary_governance::TendEquivalence;

/// Convert a location string (used in V1) to a PlanetaryBody.
/// This is the migration bridge from string-based to enum-based locations.
pub fn location_to_body(location: &str) -> PlanetaryBody {
    match location {
        "Earth" => PlanetaryBody::Earth,
        "Moon" => PlanetaryBody::Moon,
        "Mars" => PlanetaryBody::Mars,
        "Europa" => PlanetaryBody::Europa,
        "Titan" => PlanetaryBody::Titan,
        _ => PlanetaryBody::Earth, // Default fallback
    }
}

/// Get the communication latency penalty for governance coherence.
/// Distant colonies can't participate in real-time deliberation.
/// Returns [0, 1] where 0 = real-time and 1 = autonomous.
pub fn governance_latency_penalty(body: &PlanetaryBody) -> f64 {
    let delay = body.light_delay_to_earth_secs();
    // 0s = 0.0 penalty, 750s (Mars) = 0.15, 4758s (Titan) = 0.50
    (delay / 10000.0).min(0.5)
}

/// Compute a ConsciousnessProfile from sim agent state.
/// Maps the sim's per-agent fields to the 4D Mycelix profile.
pub fn agent_to_profile(
    education_level: f64,
    skill_total: f64,
    partner_count: u32,
    faction_active: bool,
    work_hours: f64,
) -> ConsciousnessProfile {
    ConsciousnessProfile {
        // Identity: education + skill mastery (how well you know yourself)
        identity: (education_level * 0.5 + (skill_total / 8.0).min(1.0) * 0.5).min(1.0),
        // Reputation: skill-based (what you contribute)
        reputation: (skill_total / 4.0).min(1.0),
        // Community: social connections + faction participation
        community: ((partner_count as f64 * 0.3) + if faction_active { 0.3 } else { 0.0 }).min(1.0),
        // Engagement: productive work hours
        engagement: (work_hours / 160.0).min(1.0), // 160 hr/month = full engagement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_to_body() {
        assert_eq!(location_to_body("Mars"), PlanetaryBody::Mars);
        assert_eq!(location_to_body("Europa"), PlanetaryBody::Europa);
        assert_eq!(location_to_body("Moon"), PlanetaryBody::Moon);
    }

    #[test]
    fn test_planetary_body_data() {
        let mars = PlanetaryBody::Mars;
        assert!((mars.gravity_fraction() - 0.38).abs() < 0.01);
        assert!((mars.solar_flux_fraction() - 0.43).abs() < 0.01);
        assert!(mars.light_delay_to_earth_secs() > 100.0);
        assert!(mars.solar_viable()); // 0.43 > 0.1 threshold
    }

    #[test]
    fn test_consciousness_profile_tiers() {
        // Low engagement agent = Observer
        let low = agent_to_profile(0.1, 0.8, 0, false, 20.0);
        assert_eq!(
            CivicTier::from_score(low.combined_score()),
            CivicTier::Observer
        );

        // High engagement agent = Citizen or above
        let high = agent_to_profile(0.8, 4.0, 1, true, 160.0);
        let tier = CivicTier::from_score(high.combined_score());
        assert!(
            tier as u8 >= CivicTier::Citizen as u8,
            "High agent should be Citizen+, got {:?} (score: {:.3})",
            tier,
            high.combined_score()
        );
    }

    #[test]
    fn test_governance_latency() {
        assert!(governance_latency_penalty(&PlanetaryBody::Earth) < 0.01);
        assert!(governance_latency_penalty(&PlanetaryBody::Mars) > 0.05);
        assert!(governance_latency_penalty(&PlanetaryBody::Titan) > 0.3);
    }
}
