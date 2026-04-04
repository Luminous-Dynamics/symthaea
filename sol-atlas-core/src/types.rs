// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Data types shared across Sol Atlas renderers.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ─── Layers ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    FossilDeposits,
    Nuclear,
    Earthquakes,
    Fires,
    Storms,
    Volcanoes,
}

impl Layer {
    pub fn all() -> HashSet<Self> {
        [
            Self::Energy, Self::Geothermal, Self::Maglev, Self::ResontiaVaults,
            Self::TerraLumina, Self::Regions, Self::SupplyChain, Self::Climate,
            Self::Emergency, Self::Health, Self::Robotics, Self::FossilDeposits, Self::Nuclear,
            Self::Earthquakes, Self::Fires, Self::Storms, Self::Volcanoes,
        ]
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
            Self::FossilDeposits => "Fossil Deposits",
            Self::Nuclear => "Nuclear Sites",
            Self::Earthquakes => "Earthquakes",
            Self::Fires => "Active Fires",
            Self::Storms => "Storms",
            Self::Volcanoes => "Volcanoes",
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
            Self::FossilDeposits => "#b8860b",
            Self::Nuclear => "#a855f7",
            Self::Earthquakes => "#e53e3e",
            Self::Fires => "#ed8936",
            Self::Storms => "#4299e1",
            Self::Volcanoes => "#e53e3e",
        }
    }

    /// RGB color as `[f32; 3]` (0.0-1.0) for GPU use.
    pub fn rgb(&self) -> [f32; 3] {
        match self {
            Self::Energy => [0.024, 0.714, 0.831],
            Self::ResontiaVaults => [0.204, 0.827, 0.600],
            Self::Maglev => [0.984, 0.749, 0.141],
            Self::Geothermal => [0.937, 0.267, 0.267],
            Self::TerraLumina => [0.655, 0.545, 0.984],
            Self::Regions => [0.231, 0.510, 0.965],
            Self::SupplyChain => [0.961, 0.620, 0.043],
            Self::Climate => [0.063, 0.725, 0.506],
            Self::Emergency => [0.937, 0.267, 0.267],
            Self::Health => [0.925, 0.282, 0.600],
            Self::Robotics => [0.545, 0.361, 0.965],
            Self::FossilDeposits => [0.72, 0.53, 0.04],
            Self::Nuclear => [0.659, 0.333, 0.969],
            Self::Earthquakes => [0.9, 0.15, 0.1],
            Self::Fires => [0.95, 0.5, 0.1],
            Self::Storms => [0.1, 0.7, 0.9],
            Self::Volcanoes => [0.9, 0.3, 0.05],
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
    pub fn rgb(&self) -> [f32; 3] {
        match self {
            Self::Solar => [0.984, 0.749, 0.141],
            Self::Wind => [0.024, 0.714, 0.831],
            Self::Hydro => [0.231, 0.510, 0.965],
            Self::Nuclear => [0.659, 0.333, 0.969],
            Self::Geothermal => [0.937, 0.267, 0.267],
            Self::Battery => [0.659, 0.333, 0.969],
            Self::Hybrid => [0.925, 0.282, 0.600],
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

// ─── Fuel types (fossil deposits) ────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuelType {
    Oil,
    Gas,
    Coal,
    TarSands,
}

impl FuelType {
    pub fn rgb(&self) -> [f32; 3] {
        match self {
            Self::Oil => [0.72, 0.53, 0.04],       // dark goldenrod
            Self::Gas => [0.50, 0.50, 0.50],        // gray
            Self::Coal => [0.29, 0.29, 0.29],       // dark gray
            Self::TarSands => [0.42, 0.26, 0.15],   // dark brown
        }
    }

    pub fn hex_color(&self) -> &'static str {
        match self {
            Self::Oil => "#b8860b",
            Self::Gas => "#808080",
            Self::Coal => "#4a4a4a",
            Self::TarSands => "#6b4226",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Oil => "Oil",
            Self::Gas => "Natural Gas",
            Self::Coal => "Coal",
            Self::TarSands => "Tar Sands",
        }
    }
}

// ─── Reactor types (nuclear sites) ───────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ReactorType {
    #[serde(alias = "pwr")]
    PWR,   // Pressurized Water Reactor
    #[serde(alias = "bwr")]
    BWR,   // Boiling Water Reactor
    #[serde(alias = "phwr")]
    PHWR,  // Pressurized Heavy Water (CANDU)
    #[serde(alias = "htgr")]
    HTGR,  // High Temperature Gas-cooled
    #[serde(alias = "smr")]
    SMR,   // Small Modular Reactor
    #[serde(alias = "fbr")]
    FBR,   // Fast Breeder Reactor
    #[serde(alias = "other")]
    Other,
}

impl ReactorType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::PWR => "Pressurized Water",
            Self::BWR => "Boiling Water",
            Self::PHWR => "Heavy Water (CANDU)",
            Self::HTGR => "High Temp Gas",
            Self::SMR => "Small Modular",
            Self::FBR => "Fast Breeder",
            Self::Other => "Other",
        }
    }

    pub fn is_smr(&self) -> bool {
        matches!(self, Self::SMR)
    }
}

// ─── Marker types (for SDF selection on GPU) ────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MarkerShape {
    Energy = 0,
    Geothermal = 1,
    Vault = 2,
    TerraLumina = 3,
    Region = 4,
    Climate = 5,
    Emergency = 6,
    Health = 7,
    Robotics = 8,
    FossilDeposit = 9,
    Nuclear = 10,
}

impl MarkerShape {
    pub fn as_f32(self) -> f32 {
        self as u8 as f32
    }
}

// ─── Data types ─────────────────────────────────────────────────

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

// ─── Fossil Deposit ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FossilDeposit {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub fuel_type: FuelType,
    pub proven_reserves_mboe: f64,
    pub annual_production_mboe: f64,
    pub status: String,
    pub country: String,
    pub discovery_year: u32,
    /// Extraction cost per barrel of oil equivalent (USD). None if unknown.
    #[serde(default)]
    pub extraction_cost_per_boe: Option<f64>,
    /// Decommissioning/remediation cost (millions USD). None if unknown.
    #[serde(default)]
    pub decommission_cost_m: Option<f64>,
    /// Energy Return on Investment (energy out / energy in). None if unknown.
    #[serde(default)]
    pub eroi: Option<f64>,
}

// ─── Nuclear Site ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearSite {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub reactor_type: ReactorType,
    pub capacity_mw: f64,
    pub status: String,
    pub operator: String,
    pub country: String,
    pub commission_year: u32,
}

// ─── Loaded Data Bundle ─────────────────────────────────────────

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
    pub fossil_deposits: Vec<FossilDeposit>,
    pub nuclear_sites: Vec<NuclearSite>,
    pub natural_events: Vec<NaturalEvent>,
}

/// Natural event from USGS, NASA EONET, FIRMS, or volcanoes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalEvent {
    pub lat: f64,
    pub lon: f64,
    pub event_type: NaturalEventType,
    pub magnitude: f64,
    pub name: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum NaturalEventType {
    Earthquake,
    Fire,
    Storm,
    Volcano,
}

// ─── Hover/Selection ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum HoverInfo {
    Site(Site),
    GeothermalNode(GeothermalNode),
    MaglevCorridor(MaglevCorridor),
    ResontiaVault(ResontiaVault),
    TerraLuminaSite(TerraLuminaSite),
    EarthRegion(EarthRegion),
    FossilDeposit(FossilDeposit),
    NuclearSite(NuclearSite),
}

#[derive(Debug, Clone)]
pub enum SelectedItem {
    Site(Site),
    GeothermalNode(GeothermalNode),
    MaglevCorridor(MaglevCorridor),
    ResontiaVault(ResontiaVault),
    TerraLuminaSite(TerraLuminaSite),
    EarthRegion(EarthRegion),
    FossilDeposit(FossilDeposit),
    NuclearSite(NuclearSite),
}

// ─── Marker instance data (renderer-agnostic) ───────────────────

/// Per-marker instance data for GPU instanced rendering.
/// Layout: position [f32; 3], color [f32; 3], size f32, marker_type f32 = 8 floats.
#[derive(Debug, Clone, Copy)]
pub struct MarkerInstance {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub size: f32,
    pub marker_type: f32,
}

impl MarkerInstance {
    /// Pack into flat [f32; 8] for GPU buffer upload.
    pub fn as_f32_array(&self) -> [f32; 8] {
        [
            self.position[0], self.position[1], self.position[2],
            self.color[0], self.color[1], self.color[2],
            self.size,
            self.marker_type,
        ]
    }
}
