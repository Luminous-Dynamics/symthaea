// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;

// ─── Entry Types ─────────────────────────────────────────────────

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GeothermalNode {
    pub name: String,
    pub region: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_mw: f64,
    pub temperature_c: u32,
    pub node_type: String, // HighEnthalpy, SuperhotRock, EnhancedGeothermal
    pub status: String,    // Operational, UnderDevelopment, Planned, Potential
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
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
    pub seismic_risk: String, // Low, Medium, High, Extreme
    pub cost_billion_usd: f64,
    pub capacity_pax_hr: u32,
    pub geothermal_powered: bool,
    pub blast_doors: u32,
    pub build_years: u32,
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResontiaVault {
    pub vault_id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_people: u32,
    pub heat_rejection_mw: f64,
    pub blast_doors: u32,
    pub status: String, // Planned, UnderConstruction, Operational, Dormant
    pub terra_lumina_id: String,
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TerraLuminaSite {
    pub site_id: String,
    pub name: String,
    pub country: String,
    pub lat: f64,
    pub lon: f64,
    pub score: u32,
    pub tier: String, // Ultimate, Premium, Standard
    pub geothermal_gw: f64,
    pub solar_gw: f64,
    pub hydro_gw: f64,
    pub total_renewable_gw: f64,
    pub phase1_billion_eur: f64,
    pub total_billion_eur: f64,
    pub irr_percent: f64,
    pub created: Timestamp,
}

// ─── Entry & Link Registration ──────────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    GeothermalNode(GeothermalNode),
    MaglevCorridor(MaglevCorridor),
    ResontiaVault(ResontiaVault),
    TerraLuminaSite(TerraLuminaSite),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllGeothermalNodes,
    AllCorridors,
    AllVaults,
    AllTerraLuminaSites,
    NodeToCorridors,
    VaultToCorridors,
    RegionToInfrastructure,
}

// ─── Validation ──────────────────────────────────────────────────

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::GeothermalNode(node) => validate_coordinates(node.lat, node.lon),
                EntryTypes::MaglevCorridor(corridor) => {
                    validate_coordinates(corridor.from_lat, corridor.from_lon)?;
                    validate_coordinates(corridor.to_lat, corridor.to_lon)?;
                    if corridor.distance_km <= 0.0 {
                        return Ok(ValidateCallbackResult::Invalid("Distance must be positive".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::ResontiaVault(vault) => {
                    validate_coordinates(vault.lat, vault.lon)?;
                    if vault.capacity_people == 0 {
                        return Ok(ValidateCallbackResult::Invalid("Vault must have capacity > 0".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::TerraLuminaSite(site) => {
                    validate_coordinates(site.lat, site.lon)?;
                    if site.score > 100 {
                        return Ok(ValidateCallbackResult::Invalid("Score must be 0-100".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_coordinates(lat: f64, lon: f64) -> ExternResult<ValidateCallbackResult> {
    if lat < -90.0 || lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if lon < -180.0 || lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}
