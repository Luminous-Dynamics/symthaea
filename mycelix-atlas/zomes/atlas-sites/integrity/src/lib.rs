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
pub struct EnergySite {
    pub site_id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub energy_type: String, // solar, wind, hydro, nuclear, geothermal, battery, hybrid
    pub capacity_mw: f64,
    pub status: String, // proposed, construction, operational, decommissioned
    pub country: String,
    pub state: Option<String>,
    pub developer: Option<String>,
    pub scorecard: Option<SiteScorecard>,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SiteScorecard {
    pub total_investment: f64,
    pub payback_years: f64,
    pub capacity_factor: f64,
    pub co2_avoided_tons_yr: f64,
    pub risk_score: f64,
    pub grid_connection: String, // easy, moderate, difficult
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SiteCluster {
    pub cluster_id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub energy_type: String,
    pub total_capacity_mw: f64,
    pub site_count: u32,
    pub zoom_level: String, // world, regional, local
}

// ─── Entry & Link Type Registration ─────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    EnergySite(EnergySite),
    SiteScorecard(SiteScorecard),
    SiteCluster(SiteCluster),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllSites,
    TypeToSites,
    RegionToSites,
    StateToSites,
    AllClusters,
    ZoomLevelToClusters,
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
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::EnergySite(site) => validate_create_energy_site(site),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::EnergySite(site) => validate_create_energy_site(site),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_energy_site(site: EnergySite) -> ExternResult<ValidateCallbackResult> {
    if site.lat < -90.0 || site.lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if site.lon < -180.0 || site.lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    if site.capacity_mw < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Capacity must be non-negative".into()));
    }
    if site.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Name must not be empty".into()));
    }
    let valid_types = ["solar", "wind", "hydro", "nuclear", "geothermal", "battery", "hybrid"];
    if !valid_types.contains(&site.energy_type.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid energy type: {}. Must be one of: {:?}", site.energy_type, valid_types),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
