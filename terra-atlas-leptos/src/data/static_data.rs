// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::types::*;

const MAGLEV_JSON: &str = include_str!("../../assets/data/maglev-network.json");
const VAULTS_JSON: &str = include_str!("../../assets/data/resontia-vaults.json");
const TERRA_LUMINA_JSON: &str = include_str!("../../assets/data/terra-lumina-sites.json");
const SITES_JSON: &str = include_str!("../../assets/data/sites-clustered.json");
const REGIONS_JSON: &str = include_str!("../../assets/data/earth-regions.json");
const SUPPLY_ROUTES_JSON: &str = include_str!("../../assets/data/supply-routes.json");
const CLIMATE_JSON: &str = include_str!("../../assets/data/climate-projects.json");
const INFRA_JSON: &str = include_str!("../../assets/data/infrastructure.json");

#[derive(serde::Deserialize)]
struct MaglevNetwork {
    geothermal_nodes: Vec<GeothermalNode>,
    maglev_corridors: Vec<MaglevCorridor>,
}

/// Raw site entry from clustered JSON (has extra `count` field we drop).
#[derive(serde::Deserialize)]
struct RawSite {
    id: String,
    name: String,
    lat: f64,
    lon: f64,
    energy_type: EnergyType,
    capacity_mw: f64,
    #[serde(default)]
    status: String,
    #[serde(default)]
    country: String,
}

pub fn load_all() -> LoadedData {
    let network: MaglevNetwork =
        serde_json::from_str(MAGLEV_JSON).expect("Failed to parse maglev-network.json");

    let vaults: Vec<ResontiaVault> =
        serde_json::from_str(VAULTS_JSON).expect("Failed to parse resontia-vaults.json");

    let terra_lumina: Vec<TerraLuminaSite> =
        serde_json::from_str(TERRA_LUMINA_JSON).expect("Failed to parse terra-lumina-sites.json");

    let raw_sites: Vec<RawSite> =
        serde_json::from_str(SITES_JSON).expect("Failed to parse sites-clustered.json");

    let sites = raw_sites
        .into_iter()
        .map(|r| Site {
            id: r.id,
            name: r.name,
            lat: r.lat,
            lon: r.lon,
            energy_type: r.energy_type,
            capacity_mw: r.capacity_mw,
            status: r.status,
            country: r.country,
        })
        .collect();

    let earth_regions: Vec<EarthRegion> =
        serde_json::from_str(REGIONS_JSON).expect("Failed to parse earth-regions.json");

    let supply_routes: Vec<SupplyRoute> =
        serde_json::from_str(SUPPLY_ROUTES_JSON).expect("Failed to parse supply-routes.json");

    let climate_projects: Vec<ClimateProject> =
        serde_json::from_str(CLIMATE_JSON).expect("Failed to parse climate-projects.json");

    let infra: serde_json::Value =
        serde_json::from_str(INFRA_JSON).expect("Failed to parse infrastructure.json");
    let emergency_shelters: Vec<EmergencyShelter> =
        serde_json::from_value(infra["emergency_shelters"].clone()).unwrap_or_default();
    let health_facilities: Vec<HealthFacility> =
        serde_json::from_value(infra["health_facilities"].clone()).unwrap_or_default();
    let robotics_dispatch: Vec<RoboticsDispatch> =
        serde_json::from_value(infra["robotics_dispatch"].clone()).unwrap_or_default();

    LoadedData {
        sites,
        geothermal_nodes: network.geothermal_nodes,
        maglev_corridors: network.maglev_corridors,
        resontia_vaults: vaults,
        terra_lumina_sites: terra_lumina,
        earth_regions,
        supply_routes,
        climate_projects,
        emergency_shelters,
        health_facilities,
        robotics_dispatch,
    }
}
