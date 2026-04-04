// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix ecosystem data flows for Sol Atlas visualization.
//! Types and simulation for all cross-cluster connections.

use serde::{Deserialize, Serialize};

// ─── TEND Time Exchange ─────────────────────────────────────────

/// A TEND time-banking exchange between two communities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TendFlow {
    pub from_lat: f64,
    pub from_lon: f64,
    pub from_name: String,
    pub to_lat: f64,
    pub to_lon: f64,
    pub to_name: String,
    pub amount_hours: f64,
    pub service_type: String,
}

// ─── Carbon Accounting ──────────────────────────────────────────

/// Live carbon accounting entry from mycelix-climate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonEntry {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    pub co2_offset_mt: f64,
    pub project_type: String,
    pub verified: bool,
}

// ─── Governance Activity ────────────────────────────────────────

/// Governance participation pulse from mycelix-governance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernancePulse {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    /// Participation rate (0.0-1.0).
    pub participation: f32,
    /// Active proposals count.
    pub active_proposals: u32,
    /// Total voters in region.
    pub voter_count: u32,
}

// ─── Supply Chain Provenance ────────────────────────────────────

/// Verified supply chain flow from mycelix-supplychain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceFlow {
    pub from_lat: f64,
    pub from_lon: f64,
    pub to_lat: f64,
    pub to_lon: f64,
    pub goods: String,
    pub verified: bool,
    pub carbon_footprint_kg: f64,
}

// ─── Resource Mesh ──────────────────────────────────────────────

/// Resource sharing flow from mycelix-commons (water, food, transport).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFlow {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    pub resource_type: String,
    pub capacity_utilization: f32,
    pub sharing_active: bool,
}

// ─── Knowledge Propagation ──────────────────────────────────────

/// Knowledge claim propagation from mycelix-knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    pub claims_count: u32,
    pub verification_score: f32,
}

// ─── Simulations ────────────────────────────────────────────────

/// Simulate TEND time-banking flows between major community hubs.
pub fn simulate_tend_flows() -> Vec<TendFlow> {
    vec![
        TendFlow { from_lat: -26.2, from_lon: 28.0, from_name: "Johannesburg".into(), to_lat: -33.9, to_lon: 18.4, to_name: "Cape Town".into(), amount_hours: 45.0, service_type: "CareWork".into() },
        TendFlow { from_lat: -33.9, from_lon: 18.4, from_name: "Cape Town".into(), to_lat: -26.2, to_lon: 28.0, to_name: "Johannesburg".into(), amount_hours: 38.0, service_type: "Education".into() },
        TendFlow { from_lat: 51.5, from_lon: -0.1, from_name: "London".into(), to_lat: 48.9, to_lon: 2.3, to_name: "Paris".into(), amount_hours: 22.0, service_type: "TechSupport".into() },
        TendFlow { from_lat: 40.7, from_lon: -74.0, from_name: "New York".into(), to_lat: 34.0, to_lon: -118.2, to_name: "Los Angeles".into(), amount_hours: 31.0, service_type: "Creative".into() },
        TendFlow { from_lat: -23.5, from_lon: -46.6, from_name: "São Paulo".into(), to_lat: -22.9, to_lon: -43.2, to_name: "Rio de Janeiro".into(), amount_hours: 28.0, service_type: "Wellness".into() },
        TendFlow { from_lat: 35.7, from_lon: 139.7, from_name: "Tokyo".into(), to_lat: 37.6, to_lon: 127.0, to_name: "Seoul".into(), amount_hours: 19.0, service_type: "Gardening".into() },
        TendFlow { from_lat: 19.4, from_lon: -99.1, from_name: "Mexico City".into(), to_lat: 4.7, to_lon: -74.1, to_name: "Bogotá".into(), amount_hours: 15.0, service_type: "HomeServices".into() },
        TendFlow { from_lat: 6.5, from_lon: 3.4, from_name: "Lagos".into(), to_lat: 5.6, to_lon: -0.2, to_name: "Accra".into(), amount_hours: 12.0, service_type: "FoodServices".into() },
    ]
}

/// Simulate governance participation pulses.
pub fn simulate_governance_pulses() -> Vec<GovernancePulse> {
    vec![
        GovernancePulse { lat: -26.2, lon: 28.0, name: "Johannesburg DAO".into(), participation: 0.72, active_proposals: 5, voter_count: 340 },
        GovernancePulse { lat: 51.5, lon: -0.1, name: "London Commons".into(), participation: 0.45, active_proposals: 3, voter_count: 890 },
        GovernancePulse { lat: 40.7, lon: -74.0, name: "NYC Cooperative".into(), participation: 0.38, active_proposals: 7, voter_count: 1200 },
        GovernancePulse { lat: -33.9, lon: 18.4, name: "Cape Town Assembly".into(), participation: 0.81, active_proposals: 4, voter_count: 210 },
        GovernancePulse { lat: 48.9, lon: 2.3, name: "Paris Collective".into(), participation: 0.55, active_proposals: 2, voter_count: 560 },
        GovernancePulse { lat: 35.7, lon: 139.7, name: "Tokyo Network".into(), participation: 0.33, active_proposals: 1, voter_count: 780 },
        GovernancePulse { lat: -1.3, lon: 36.8, name: "Nairobi Hub".into(), participation: 0.67, active_proposals: 6, voter_count: 150 },
        GovernancePulse { lat: 13.75, lon: 100.5, name: "Bangkok Exchange".into(), participation: 0.42, active_proposals: 3, voter_count: 290 },
    ]
}

/// Color for governance participation: dim → bright based on participation rate.
pub fn governance_color(participation: f32) -> [f32; 3] {
    // Mycelix lime with intensity based on participation
    [0.486 * participation, 0.988 * participation, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tend_flows_bidirectional() {
        let flows = simulate_tend_flows();
        assert!(flows.len() >= 8);
        // SA has bidirectional flows
        let sa_flows: Vec<_> = flows.iter()
            .filter(|f| f.from_name.contains("Johannesburg") || f.to_name.contains("Johannesburg"))
            .collect();
        assert!(sa_flows.len() >= 2);
    }

    #[test]
    fn test_governance_pulses() {
        let pulses = simulate_governance_pulses();
        assert_eq!(pulses.len(), 8);
        // Cape Town should have highest participation
        let ct = pulses.iter().find(|p| p.name.contains("Cape Town")).unwrap();
        assert!(ct.participation > 0.7);
    }

    #[test]
    fn test_governance_color_scales() {
        let dim = governance_color(0.1);
        let bright = governance_color(0.9);
        assert!(bright[0] > dim[0]);
        assert!(bright[1] > dim[1]);
    }
}
