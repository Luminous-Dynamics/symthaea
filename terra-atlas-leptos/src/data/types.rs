// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ─── Layers ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    Energy,
    ResontiaVaults,
    Maglev,
    Geothermal,
    TerraLumina,
    Regions,
    SupplyChain,
    Climate,
    Emergency,
    Health,
    Robotics,
}

impl Layer {
    pub fn all() -> HashSet<Self> {
        [Self::Energy, Self::Geothermal, Self::Maglev, Self::ResontiaVaults, Self::TerraLumina,
         Self::Regions, Self::SupplyChain, Self::Climate, Self::Emergency, Self::Health, Self::Robotics]
            .into_iter()
            .collect()
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Energy => "Energy Sites",
            Self::ResontiaVaults => "Resontia Vaults",
            Self::Maglev => "Maglev Corridors",
            Self::Geothermal => "Geothermal Nodes",
            Self::TerraLumina => "Terra Lumina Sites",
            Self::Regions => "Earth Regions",
            Self::SupplyChain => "Supply Chain",
            Self::Climate => "Climate Projects",
            Self::Emergency => "Emergency Shelters",
            Self::Health => "Health Facilities",
            Self::Robotics => "Robotics Dispatch",
        }
    }

    pub fn css_color(&self) -> &'static str {
        match self {
            Self::Energy => "#06b6d4",
            Self::ResontiaVaults => "#34d399",
            Self::Maglev => "#fbbf24",
            Self::Geothermal => "#ef4444",
            Self::TerraLumina => "#a78bfa",
            Self::Regions => "#3b82f6",
            Self::SupplyChain => "#f59e0b",
            Self::Climate => "#10b981",
            Self::Emergency => "#ef4444",
            Self::Health => "#ec4899",
            Self::Robotics => "#8b5cf6",
        }
    }
}

// ─── Energy ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnergyType {
    Solar,
    Wind,
    Hydro,
    Nuclear,
    Geothermal,
    Battery,
    Hybrid,
}

impl EnergyType {
    /// RGB color (0.0-1.0) matching TerraGlobeWithSites.tsx palette.
    pub fn color_rgb(&self) -> (f32, f32, f32) {
        match self {
            Self::Solar => (0.984, 0.749, 0.141),       // #fbbf24
            Self::Wind => (0.024, 0.714, 0.831),        // #06b6d4
            Self::Hydro => (0.231, 0.510, 0.965),       // #3b82f6
            Self::Nuclear => (0.659, 0.333, 0.969),     // #a855f7
            Self::Geothermal => (0.937, 0.267, 0.267),  // #ef4444
            Self::Battery => (0.659, 0.333, 0.969),     // #a855f7
            Self::Hybrid => (0.925, 0.282, 0.600),      // #ec4899
        }
    }

    pub fn hex_color(&self) -> &'static str {
        match self {
            Self::Solar => "#fbbf24",
            Self::Wind => "#06b6d4",
            Self::Hydro => "#3b82f6",
            Self::Nuclear => "#a855f7",
            Self::Geothermal => "#ef4444",
            Self::Battery => "#a855f7",
            Self::Hybrid => "#ec4899",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Solar => "Solar",
            Self::Wind => "Wind",
            Self::Hydro => "Hydro",
            Self::Nuclear => "Nuclear",
            Self::Geothermal => "Geothermal",
            Self::Battery => "Battery",
            Self::Hybrid => "Hybrid",
        }
    }
}

// ─── Site ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Site {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub energy_type: EnergyType,
    pub capacity_mw: f64,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub country: String,
}

// ─── Geothermal Node ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeothermalNode {
    pub name: String,
    pub region: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_mw: f64,
    pub temperature_c: u32,
    pub node_type: String,
    pub status: String,
}

// ─── Maglev Corridor ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaglevCorridor {
    pub name: String,
    pub from_name: String,
    pub from_lat: f64,
    pub from_lon: f64,
    pub to_name: String,
    pub to_lat: f64,
    pub to_lon: f64,
    pub distance_km: f64,
    pub travel_time_min: f64,
    pub submarine: bool,
    pub seismic_risk: String,
    pub cost_billion_usd: f64,
    pub capacity_pax_hr: u32,
    pub geothermal_powered: bool,
}

// ─── Resontia Vault ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResontiaVault {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_people: u32,
    pub heat_rejection_mw: f64,
    pub blast_doors: u32,
    pub status: String,
    pub terra_lumina_id: String,
}

// ─── Terra Lumina Site ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerraLuminaSite {
    pub id: String,
    pub name: String,
    pub country: String,
    pub lat: f64,
    pub lon: f64,
    pub score: u32,
    pub tier: String,
    pub geothermal_gw: f64,
    pub solar_gw: f64,
    pub hydro_gw: f64,
    pub total_renewable_gw: f64,
    pub phase1_billion_eur: f64,
    pub total_billion_eur: f64,
    pub irr_percent: f64,
}

// ─── Supply Chain Route ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyRoute {
    pub from: String,
    pub from_lat: f64,
    pub from_lon: f64,
    pub to: String,
    pub to_lat: f64,
    pub to_lon: f64,
    pub mode: String,
    pub goods: Vec<String>,
    pub capacity: f64,
}

// ─── Climate Project ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateProject {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    #[serde(rename = "type")]
    pub project_type: String,
    pub status: String,
    pub co2_offset_mt: f64,
    pub country: String,
}

// ─── Emergency + Health + Robotics ───────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyShelter {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity: u32,
    #[serde(rename = "type")]
    pub shelter_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthFacility {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    #[serde(rename = "type")]
    pub facility_type: String,
    pub beds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoboticsDispatch {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    #[serde(rename = "type")]
    pub platform_type: String,
    pub units: u32,
}

// ─── Earth Region ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarthRegion {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub population_m: f64,
    pub gdp_per_capita: f64,
    pub education_index: f64,
    pub phi_mean: f64,
    pub climate_vulnerability: f64,
    pub infrastructure: f64,
    pub spaceport: bool,
}

// ─── Loaded Data Bundle ──────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct LoadedData {
    pub sites: Vec<Site>,
    pub geothermal_nodes: Vec<GeothermalNode>,
    pub maglev_corridors: Vec<MaglevCorridor>,
    pub resontia_vaults: Vec<ResontiaVault>,
    pub terra_lumina_sites: Vec<TerraLuminaSite>,
    pub earth_regions: Vec<EarthRegion>,
    pub supply_routes: Vec<SupplyRoute>,
    pub climate_projects: Vec<ClimateProject>,
    pub emergency_shelters: Vec<EmergencyShelter>,
    pub health_facilities: Vec<HealthFacility>,
    pub robotics_dispatch: Vec<RoboticsDispatch>,
}

// ─── Hover/Selection Info ────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum HoverInfo {
    Site(Site),
    GeothermalNode(GeothermalNode),
    MaglevCorridor(MaglevCorridor),
    ResontiaVault(ResontiaVault),
    TerraLuminaSite(TerraLuminaSite),
    EarthRegion(EarthRegion),
}

#[derive(Debug, Clone)]
pub enum SelectedItem {
    Site(Site),
    GeothermalNode(GeothermalNode),
    MaglevCorridor(MaglevCorridor),
    ResontiaVault(ResontiaVault),
    TerraLuminaSite(TerraLuminaSite),
    EarthRegion(EarthRegion),
}
