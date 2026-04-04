// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON data loading for Sol Atlas datasets.
//!
//! Consumers provide JSON strings (from `include_str!`, file reads, or network).
//! This module handles deserialization into terra-atlas-core types.

use crate::types::*;
use serde::Deserialize;

/// Intermediate type for maglev-network.json (two arrays in one object).
#[derive(Deserialize)]
struct MaglevNetwork {
    geothermal_nodes: Vec<GeothermalNode>,
    maglev_corridors: Vec<MaglevCorridor>,
}

/// Raw site entry from clustered JSON (has extra fields we drop).
#[derive(Deserialize)]
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

/// Intermediate type for infrastructure.json (three arrays in one object).
#[derive(Deserialize)]
struct InfrastructureBundle {
    #[serde(default)]
    emergency_shelters: Vec<EmergencyShelter>,
    #[serde(default)]
    health_facilities: Vec<HealthFacility>,
    #[serde(default)]
    robotics_dispatch: Vec<RoboticsDispatch>,
}

/// Parse energy sites from sites-clustered.json.
pub fn parse_sites(json: &str) -> Result<Vec<Site>, serde_json::Error> {
    let raw: Vec<RawSite> = serde_json::from_str(json)?;
    Ok(raw
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
        .collect())
}

/// Parse maglev-network.json → (geothermal_nodes, maglev_corridors).
pub fn parse_maglev_network(
    json: &str,
) -> Result<(Vec<GeothermalNode>, Vec<MaglevCorridor>), serde_json::Error> {
    let network: MaglevNetwork = serde_json::from_str(json)?;
    Ok((network.geothermal_nodes, network.maglev_corridors))
}

/// Parse resontia-vaults.json.
pub fn parse_vaults(json: &str) -> Result<Vec<ResontiaVault>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Parse terra-lumina-sites.json.
pub fn parse_terra_lumina(json: &str) -> Result<Vec<TerraLuminaSite>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Parse earth-regions.json.
pub fn parse_regions(json: &str) -> Result<Vec<EarthRegion>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Parse supply-routes.json.
pub fn parse_supply_routes(json: &str) -> Result<Vec<SupplyRoute>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Parse climate-projects.json.
pub fn parse_climate_projects(json: &str) -> Result<Vec<ClimateProject>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Parse infrastructure.json → (emergency_shelters, health_facilities, robotics_dispatch).
pub fn parse_infrastructure(
    json: &str,
) -> Result<(Vec<EmergencyShelter>, Vec<HealthFacility>, Vec<RoboticsDispatch>), serde_json::Error>
{
    let bundle: InfrastructureBundle = serde_json::from_str(json)?;
    Ok((
        bundle.emergency_shelters,
        bundle.health_facilities,
        bundle.robotics_dispatch,
    ))
}

/// Parse nuclear-sites.json.
pub fn parse_nuclear_sites(json: &str) -> Result<Vec<NuclearSite>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Parse fossil-deposits.json.
pub fn parse_fossil_deposits(json: &str) -> Result<Vec<FossilDeposit>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Load all datasets from their respective JSON strings into a single `LoadedData`.
pub fn load_all(
    sites_json: &str,
    maglev_json: &str,
    vaults_json: &str,
    terra_lumina_json: &str,
    regions_json: &str,
    supply_routes_json: &str,
    climate_json: &str,
    infrastructure_json: &str,
    fossil_deposits_json: &str,
    nuclear_sites_json: &str,
    earthquakes_json: &str,
    fires_json: &str,
    storms_json: &str,
    volcanoes_json: &str,
) -> LoadedData {
    let sites = parse_sites(sites_json).unwrap_or_default();
    let (geothermal_nodes, maglev_corridors) =
        parse_maglev_network(maglev_json).unwrap_or_default();
    let resontia_vaults = parse_vaults(vaults_json).unwrap_or_default();
    let terra_lumina_sites = parse_terra_lumina(terra_lumina_json).unwrap_or_default();
    let earth_regions = parse_regions(regions_json).unwrap_or_default();
    let supply_routes = parse_supply_routes(supply_routes_json).unwrap_or_default();
    let climate_projects = parse_climate_projects(climate_json).unwrap_or_default();
    let (emergency_shelters, health_facilities, robotics_dispatch) =
        parse_infrastructure(infrastructure_json).unwrap_or_default();
    let fossil_deposits = parse_fossil_deposits(fossil_deposits_json).unwrap_or_default();
    let nuclear_sites = parse_nuclear_sites(nuclear_sites_json).unwrap_or_default();

    LoadedData {
        sites,
        geothermal_nodes,
        maglev_corridors,
        resontia_vaults,
        terra_lumina_sites,
        earth_regions,
        supply_routes,
        climate_projects,
        emergency_shelters,
        health_facilities,
        robotics_dispatch,
        fossil_deposits,
        nuclear_sites,
        natural_events: parse_natural_events(earthquakes_json, fires_json, storms_json, volcanoes_json),
    }
}

/// Parse simplified shipping lane routes (Vec of Vec of [lon, lat]).
pub fn parse_shipping_lanes(json: &str) -> Vec<Vec<[f64; 2]>> {
    serde_json::from_str(json).unwrap_or_default()
}

/// Parse GeoJSON natural events from USGS, NASA EONET, FIRMS, and volcano data.
fn parse_natural_events(
    earthquakes: &str,
    fires: &str,
    storms: &str,
    volcanoes: &str,
) -> Vec<NaturalEvent> {
    let mut events = Vec::new();

    // Parse GeoJSON FeatureCollections (all share same structure)
    fn parse_geojson(json: &str, event_type: NaturalEventType) -> Vec<NaturalEvent> {
        #[derive(Deserialize)]
        struct FeatureCollection { features: Vec<Feature> }
        #[derive(Deserialize)]
        struct Feature { properties: Properties, geometry: Geometry }
        #[derive(Deserialize)]
        struct Properties {
            #[serde(default)]
            magnitude: Option<f64>,
            #[serde(default)]
            brightness: Option<f64>,
            #[serde(default)]
            title: Option<String>,
            #[serde(default)]
            place: Option<String>,
            #[serde(default)]
            name: Option<String>,
            #[serde(rename = "type", default)]
            event_kind: Option<String>,
        }
        #[derive(Deserialize)]
        struct Geometry { coordinates: Vec<f64> }

        let Ok(fc) = serde_json::from_str::<FeatureCollection>(json) else { return vec![] };
        fc.features.iter().filter_map(|f| {
            let coords = &f.geometry.coordinates;
            if coords.len() < 2 { return None; }
            let lon = coords[0];
            let lat = coords[1];
            if lat.abs() > 90.0 || lon.abs() > 180.0 { return None; }
            let magnitude = f.properties.magnitude
                .or(f.properties.brightness.map(|b| b / 100.0))
                .unwrap_or(1.0);
            let name = f.properties.title
                .as_deref()
                .or(f.properties.place.as_deref())
                .or(f.properties.name.as_deref())
                .unwrap_or("Unknown")
                .to_string();
            Some(NaturalEvent { lat, lon, event_type, magnitude, name })
        }).collect()
    }

    events.extend(parse_geojson(earthquakes, NaturalEventType::Earthquake));
    events.extend(parse_geojson(fires, NaturalEventType::Fire));
    events.extend(parse_geojson(storms, NaturalEventType::Storm));
    events.extend(parse_geojson(volcanoes, NaturalEventType::Volcano));
    events
}
