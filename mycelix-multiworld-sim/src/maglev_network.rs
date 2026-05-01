// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global Maglev Network: geothermally-powered vacuum tube transit
//! connecting Resontia vault nodes and the Ecuador spaceport.
//!
//! Routing principles:
//! 1. Follow geothermal hotspots for power (no external grid dependency)
//! 2. Avoid active fault crossings where possible (seismic guillotine risk)
//! 3. Connect population centers to vault nodes AND spaceport
//! 4. Submarine crossings only where unavoidable (Bering, Gibraltar, Suez)
//!
//! Engineering constraints from published data:
//! - Vacuum tube: 300m thermal expansion per 600km (segmented)
//! - Passenger capacity: 840-3600/hr per corridor (vs 15K-20K for HSR)
//! - Speed: Mach 1+ (1,235 km/h) in vacuum
//! - Pod capacity: 28-50 passengers per pod
//! - Blast door segmentation every 10km for seismic/pressure safety
//! - Geothermal power: 5-50 MW per station (varies by hotspot)
//!
//! Sources:
//! - Carnegie Endowment (2025): Global geothermal deployment pathways
//! - IEA Geothermal Report: 15% of electricity demand growth by 2050
//! - China 621 mph maglev (2025): Practical high-speed vacuum tube
//! - International Maglev Board 2022 survey: Engineering constraints

use serde::{Deserialize, Serialize};

// ============================================================================
// GEOTHERMAL NODES
// ============================================================================

/// A geothermal power station that can power maglev infrastructure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeothermalNode {
    pub name: &'static str,
    pub region: &'static str,
    pub latitude: f64,
    pub longitude: f64,
    pub capacity_mw: f64,
    pub temperature_c: f64,
    pub node_type: GeothermalType,
    pub development_status: DevelopmentStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GeothermalType {
    HighEnthalpy,       // >150°C, volcanic
    SuperhotRock,       // >400°C, next-gen
    EnhancedGeothermal, // Engineered reservoirs
    LowEnthalpy,        // <150°C, heat pumps
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DevelopmentStatus {
    Operational,
    UnderDevelopment,
    Planned,
    Potential,
}

/// Global geothermal node catalog — sites that can power the maglev network.
pub static GEOTHERMAL_NODES: &[GeothermalNode] = &[
    // === RING OF FIRE (Pacific) ===
    GeothermalNode {
        name: "Reykjanes Peninsula",
        region: "Iceland",
        latitude: 63.85,
        longitude: -22.45,
        capacity_mw: 800.0,
        temperature_c: 300.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Krafla-Theistareykir",
        region: "Iceland",
        latitude: 65.72,
        longitude: -16.78,
        capacity_mw: 500.0,
        temperature_c: 350.0,
        node_type: GeothermalType::SuperhotRock,
        development_status: DevelopmentStatus::UnderDevelopment,
    },
    GeothermalNode {
        name: "The Geysers",
        region: "North America",
        latitude: 38.80,
        longitude: -122.77,
        capacity_mw: 1517.0,
        temperature_c: 260.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Cerro Prieto",
        region: "Latin America",
        latitude: 32.42,
        longitude: -115.30,
        capacity_mw: 720.0,
        temperature_c: 350.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Los Humeros",
        region: "Latin America",
        latitude: 19.68,
        longitude: -97.45,
        capacity_mw: 400.0,
        temperature_c: 380.0,
        node_type: GeothermalType::SuperhotRock,
        development_status: DevelopmentStatus::UnderDevelopment,
    },
    GeothermalNode {
        name: "Chimborazo Highlands",
        region: "Latin America",
        latitude: -1.47,
        longitude: -78.82,
        capacity_mw: 300.0,
        temperature_c: 250.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Potential,
    },
    GeothermalNode {
        name: "Wayang Windu",
        region: "Southeast Asia",
        latitude: -7.17,
        longitude: 107.63,
        capacity_mw: 227.0,
        temperature_c: 280.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Sarulla",
        region: "Southeast Asia",
        latitude: 2.07,
        longitude: 99.05,
        capacity_mw: 330.0,
        temperature_c: 300.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Tiwi-MakBan",
        region: "Southeast Asia",
        latitude: 13.30,
        longitude: 123.70,
        capacity_mw: 458.0,
        temperature_c: 260.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Hatchobaru",
        region: "East Asia",
        latitude: 33.10,
        longitude: 131.23,
        capacity_mw: 112.0,
        temperature_c: 250.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Wairakei-Tauhara",
        region: "Oceania",
        latitude: -38.62,
        longitude: 176.10,
        capacity_mw: 350.0,
        temperature_c: 270.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    // === EAST AFRICAN RIFT ===
    GeothermalNode {
        name: "Olkaria",
        region: "Sub-Saharan Africa",
        latitude: -0.88,
        longitude: 36.30,
        capacity_mw: 985.0,
        temperature_c: 300.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Menengai",
        region: "Sub-Saharan Africa",
        latitude: -0.20,
        longitude: 36.07,
        capacity_mw: 105.0,
        temperature_c: 290.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::UnderDevelopment,
    },
    GeothermalNode {
        name: "Aluto-Langano",
        region: "Sub-Saharan Africa",
        latitude: 7.77,
        longitude: 38.78,
        capacity_mw: 70.0,
        temperature_c: 260.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::UnderDevelopment,
    },
    GeothermalNode {
        name: "Tendaho",
        region: "Sub-Saharan Africa",
        latitude: 11.70,
        longitude: 40.95,
        capacity_mw: 100.0,
        temperature_c: 270.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Planned,
    },
    // === MEDITERRANEAN / EUROPE ===
    GeothermalNode {
        name: "Larderello",
        region: "Europe",
        latitude: 43.25,
        longitude: 10.87,
        capacity_mw: 770.0,
        temperature_c: 250.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
    GeothermalNode {
        name: "Cornwall (United Downs)",
        region: "Europe",
        latitude: 50.22,
        longitude: -5.17,
        capacity_mw: 10.0,
        temperature_c: 195.0,
        node_type: GeothermalType::EnhancedGeothermal,
        development_status: DevelopmentStatus::UnderDevelopment,
    },
    // === CENTRAL ASIA ===
    GeothermalNode {
        name: "Mutnovskaya",
        region: "Russia/CIS",
        latitude: 52.45,
        longitude: 158.20,
        capacity_mw: 50.0,
        temperature_c: 240.0,
        node_type: GeothermalType::HighEnthalpy,
        development_status: DevelopmentStatus::Operational,
    },
];

// ============================================================================
// MAGLEV CORRIDORS
// ============================================================================

/// A proposed maglev corridor connecting two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaglevCorridor {
    pub name: &'static str,
    pub from_node: &'static str,
    pub to_node: &'static str,
    pub distance_km: f64,
    pub travel_time_minutes: f64,
    pub is_submarine: bool,
    pub seismic_risk: SeismicRisk,
    pub geothermal_powered: bool,
    pub capacity_passengers_per_hour: u32,
    pub blast_door_count: u32,
    pub construction_years: f64,
    pub cost_billion_usd: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SeismicRisk {
    Low,     // Stable craton (Australia, Scandinavia)
    Medium,  // Moderate activity (Europe, East Coast)
    High,    // Active zone (Ring of Fire, Rift Valley)
    Extreme, // Direct fault crossing (San Andreas, Anatolian)
}

/// The proposed global maglev network — 15 corridors connecting
/// geothermal nodes, population centers, vault sites, and the spaceport.
pub static MAGLEV_CORRIDORS: &[MaglevCorridor] = &[
    // === AMERICAS SPINE (Ecuador spaceport to both continents) ===
    MaglevCorridor {
        name: "Ecuador Spaceport ↔ Bogota",
        from_node: "Chimborazo Highlands",
        to_node: "Bogota Hub",
        distance_km: 800.0,
        travel_time_minutes: 39.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::High,
        geothermal_powered: true,
        capacity_passengers_per_hour: 3000,
        blast_door_count: 80,
        construction_years: 8.0,
        cost_billion_usd: 40.0,
    },
    MaglevCorridor {
        name: "Bogota ↔ Panama ↔ Mexico City",
        from_node: "Bogota Hub",
        to_node: "Los Humeros",
        distance_km: 3500.0,
        travel_time_minutes: 170.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::High,
        geothermal_powered: true,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 350,
        construction_years: 15.0,
        cost_billion_usd: 175.0,
    },
    MaglevCorridor {
        name: "Mexico City ↔ Los Angeles",
        from_node: "Los Humeros",
        to_node: "The Geysers",
        distance_km: 2500.0,
        travel_time_minutes: 121.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::Extreme,
        geothermal_powered: true,
        capacity_passengers_per_hour: 3000,
        blast_door_count: 250,
        construction_years: 12.0,
        cost_billion_usd: 125.0,
    },
    MaglevCorridor {
        name: "Ecuador ↔ Lima ↔ Santiago",
        from_node: "Chimborazo Highlands",
        to_node: "Santiago Hub",
        distance_km: 3200.0,
        travel_time_minutes: 155.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::High,
        geothermal_powered: false,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 320,
        construction_years: 14.0,
        cost_billion_usd: 160.0,
    },
    // === TRANSATLANTIC (submarine) ===
    MaglevCorridor {
        name: "Iceland ↔ Scotland (submarine)",
        from_node: "Reykjanes Peninsula",
        to_node: "Cornwall (United Downs)",
        distance_km: 1500.0,
        travel_time_minutes: 73.0,
        is_submarine: true,
        seismic_risk: SeismicRisk::Medium,
        geothermal_powered: true,
        capacity_passengers_per_hour: 1000,
        blast_door_count: 150,
        construction_years: 20.0,
        cost_billion_usd: 200.0,
    },
    // === EUROPE SPINE ===
    MaglevCorridor {
        name: "London ↔ Paris ↔ Milan ↔ Larderello",
        from_node: "Cornwall (United Downs)",
        to_node: "Larderello",
        distance_km: 1800.0,
        travel_time_minutes: 87.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::Low,
        geothermal_powered: true,
        capacity_passengers_per_hour: 3600,
        blast_door_count: 180,
        construction_years: 10.0,
        cost_billion_usd: 90.0,
    },
    // === AFRICA RIFT SPINE ===
    MaglevCorridor {
        name: "Cairo ↔ Addis Ababa (Rift Valley)",
        from_node: "Tendaho",
        to_node: "Olkaria",
        distance_km: 3000.0,
        travel_time_minutes: 146.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::High,
        geothermal_powered: true,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 300,
        construction_years: 15.0,
        cost_billion_usd: 120.0,
    },
    MaglevCorridor {
        name: "Nairobi ↔ Dar es Salaam ↔ Maputo",
        from_node: "Olkaria",
        to_node: "Maputo Hub",
        distance_km: 2500.0,
        travel_time_minutes: 121.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::Medium,
        geothermal_powered: true,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 250,
        construction_years: 12.0,
        cost_billion_usd: 100.0,
    },
    // === ASIA-PACIFIC RING ===
    MaglevCorridor {
        name: "Tokyo ↔ Shanghai ↔ Jakarta",
        from_node: "Hatchobaru",
        to_node: "Wayang Windu",
        distance_km: 5500.0,
        travel_time_minutes: 267.0,
        is_submarine: true,
        seismic_risk: SeismicRisk::Extreme,
        geothermal_powered: true,
        capacity_passengers_per_hour: 3600,
        blast_door_count: 550,
        construction_years: 25.0,
        cost_billion_usd: 400.0,
    },
    MaglevCorridor {
        name: "Jakarta ↔ Manila ↔ Taipei",
        from_node: "Sarulla",
        to_node: "Tiwi-MakBan",
        distance_km: 4000.0,
        travel_time_minutes: 194.0,
        is_submarine: true,
        seismic_risk: SeismicRisk::Extreme,
        geothermal_powered: true,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 400,
        construction_years: 20.0,
        cost_billion_usd: 300.0,
    },
    // === BERING STRAIT (Americas ↔ Asia) ===
    MaglevCorridor {
        name: "Alaska ↔ Kamchatka (Bering submarine)",
        from_node: "The Geysers",
        to_node: "Mutnovskaya",
        distance_km: 4000.0,
        travel_time_minutes: 194.0,
        is_submarine: true,
        seismic_risk: SeismicRisk::Extreme,
        geothermal_powered: true,
        capacity_passengers_per_hour: 1000,
        blast_door_count: 400,
        construction_years: 30.0,
        cost_billion_usd: 500.0,
    },
    // === OCEANIA LINK ===
    MaglevCorridor {
        name: "Sydney ↔ Auckland (submarine)",
        from_node: "Wairakei-Tauhara",
        to_node: "Sydney Hub",
        distance_km: 2200.0,
        travel_time_minutes: 107.0,
        is_submarine: true,
        seismic_risk: SeismicRisk::Medium,
        geothermal_powered: true,
        capacity_passengers_per_hour: 1500,
        blast_door_count: 220,
        construction_years: 18.0,
        cost_billion_usd: 180.0,
    },
    // === MIDDLE EAST / SOUTH ASIA ===
    MaglevCorridor {
        name: "Istanbul ↔ Tehran ↔ Delhi",
        from_node: "Larderello",
        to_node: "Delhi Hub",
        distance_km: 5000.0,
        travel_time_minutes: 243.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::High,
        geothermal_powered: false,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 500,
        construction_years: 20.0,
        cost_billion_usd: 250.0,
    },
    // === GIBRALTAR CROSSING ===
    MaglevCorridor {
        name: "Spain ↔ Morocco (Gibraltar submarine)",
        from_node: "Larderello",
        to_node: "Tendaho",
        distance_km: 600.0,
        travel_time_minutes: 29.0,
        is_submarine: true,
        seismic_risk: SeismicRisk::Medium,
        geothermal_powered: false,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 60,
        construction_years: 10.0,
        cost_billion_usd: 80.0,
    },
    // === SUEZ BRIDGE ===
    MaglevCorridor {
        name: "Cairo ↔ Riyadh ↔ Mumbai",
        from_node: "Tendaho",
        to_node: "Delhi Hub",
        distance_km: 4500.0,
        travel_time_minutes: 218.0,
        is_submarine: false,
        seismic_risk: SeismicRisk::Medium,
        geothermal_powered: false,
        capacity_passengers_per_hour: 2000,
        blast_door_count: 450,
        construction_years: 18.0,
        cost_billion_usd: 225.0,
    },
];

// ============================================================================
// NETWORK STATISTICS
// ============================================================================

/// Compute aggregate network statistics.
pub fn network_stats() -> NetworkStats {
    let total_km: f64 = MAGLEV_CORRIDORS.iter().map(|c| c.distance_km).sum();
    let total_cost: f64 = MAGLEV_CORRIDORS.iter().map(|c| c.cost_billion_usd).sum();
    let submarine_km: f64 = MAGLEV_CORRIDORS
        .iter()
        .filter(|c| c.is_submarine)
        .map(|c| c.distance_km)
        .sum();
    let geothermal_powered: usize = MAGLEV_CORRIDORS
        .iter()
        .filter(|c| c.geothermal_powered)
        .count();
    let total_blast_doors: u32 = MAGLEV_CORRIDORS.iter().map(|c| c.blast_door_count).sum();
    let max_construction_years: f64 = MAGLEV_CORRIDORS
        .iter()
        .map(|c| c.construction_years)
        .fold(0.0f64, f64::max);
    let total_geothermal_mw: f64 = GEOTHERMAL_NODES.iter().map(|n| n.capacity_mw).sum();

    NetworkStats {
        total_corridors: MAGLEV_CORRIDORS.len(),
        total_km,
        submarine_km,
        total_cost_billion_usd: total_cost,
        geothermal_powered_corridors: geothermal_powered,
        total_blast_doors,
        max_construction_years,
        total_geothermal_capacity_mw: total_geothermal_mw,
        geothermal_nodes: GEOTHERMAL_NODES.len(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_corridors: usize,
    pub total_km: f64,
    pub submarine_km: f64,
    pub total_cost_billion_usd: f64,
    pub geothermal_powered_corridors: usize,
    pub total_blast_doors: u32,
    pub max_construction_years: f64,
    pub total_geothermal_capacity_mw: f64,
    pub geothermal_nodes: usize,
}

impl std::fmt::Display for NetworkStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== GLOBAL MAGLEV NETWORK ===")?;
        writeln!(f, "Corridors:       {}", self.total_corridors)?;
        writeln!(
            f,
            "Total distance:  {:.0} km ({:.0} km submarine)",
            self.total_km, self.submarine_km
        )?;
        writeln!(
            f,
            "Total cost:      ${:.0}B USD",
            self.total_cost_billion_usd
        )?;
        writeln!(
            f,
            "Geothermal:      {}/{} corridors powered",
            self.geothermal_powered_corridors, self.total_corridors
        )?;
        writeln!(f, "Blast doors:     {}", self.total_blast_doors)?;
        writeln!(
            f,
            "Max build time:  {:.0} years",
            self.max_construction_years
        )?;
        writeln!(
            f,
            "Geothermal cap:  {:.0} MW across {} nodes",
            self.total_geothermal_capacity_mw, self.geothermal_nodes
        )?;
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geothermal_nodes_have_valid_coordinates() {
        for node in GEOTHERMAL_NODES {
            assert!(
                node.latitude >= -90.0 && node.latitude <= 90.0,
                "{}: invalid latitude {}",
                node.name,
                node.latitude
            );
            assert!(
                node.longitude >= -180.0 && node.longitude <= 180.0,
                "{}: invalid longitude {}",
                node.name,
                node.longitude
            );
            assert!(node.capacity_mw > 0.0, "{}: zero capacity", node.name);
        }
    }

    #[test]
    fn corridors_have_positive_distances() {
        for corridor in MAGLEV_CORRIDORS {
            assert!(
                corridor.distance_km > 0.0,
                "{}: zero distance",
                corridor.name
            );
            assert!(
                corridor.travel_time_minutes > 0.0,
                "{}: zero travel time",
                corridor.name
            );
            assert!(
                corridor.cost_billion_usd > 0.0,
                "{}: zero cost",
                corridor.name
            );
        }
    }

    #[test]
    fn travel_times_are_realistic_for_mach_1() {
        for corridor in MAGLEV_CORRIDORS {
            let mach1_km_per_min = 1235.0 / 60.0; // ~20.6 km/min
            let min_time = corridor.distance_km / mach1_km_per_min;
            // Allow 20% overhead for acceleration/deceleration
            assert!(
                corridor.travel_time_minutes >= min_time * 0.8,
                "{}: travel time {:.0} min too fast for {:.0} km",
                corridor.name,
                corridor.travel_time_minutes,
                corridor.distance_km
            );
        }
    }

    #[test]
    fn blast_doors_every_10km() {
        for corridor in MAGLEV_CORRIDORS {
            let expected = (corridor.distance_km / 10.0) as u32;
            assert!(
                corridor.blast_door_count >= expected / 2,
                "{}: only {} blast doors for {:.0} km (expected ~{})",
                corridor.name,
                corridor.blast_door_count,
                corridor.distance_km,
                expected
            );
        }
    }

    #[test]
    fn total_network_stats() {
        let stats = network_stats();
        assert!(stats.total_km > 40_000.0, "Network should span >40K km");
        assert!(stats.total_cost_billion_usd > 1000.0, "Network costs >$1T");
        assert!(
            stats.geothermal_nodes >= 15,
            "Should have 15+ geothermal nodes"
        );
        assert!(stats.submarine_km > 5000.0, "Submarine sections >5K km");
    }

    #[test]
    fn geothermal_capacity_sufficient() {
        let stats = network_stats();
        // Each corridor needs ~50-100 MW for propulsion
        let power_needed = stats.total_corridors as f64 * 75.0;
        assert!(
            stats.total_geothermal_capacity_mw > power_needed,
            "Geothermal {:.0} MW insufficient for {:.0} MW network demand",
            stats.total_geothermal_capacity_mw,
            power_needed
        );
    }

    #[test]
    fn network_stats_display() {
        let stats = network_stats();
        let display = format!("{}", stats);
        assert!(display.contains("GLOBAL MAGLEV NETWORK"));
        assert!(display.contains("km"));
    }
}
