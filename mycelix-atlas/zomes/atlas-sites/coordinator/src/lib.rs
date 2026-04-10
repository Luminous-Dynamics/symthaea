// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use atlas_sites_integrity::*;

// ─── Input Types ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterSiteInput {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub energy_type: String,
    pub capacity_mw: f64,
    pub status: String,
    pub country: String,
    pub state: Option<String>,
    pub developer: Option<String>,
    pub scorecard: Option<SiteScorecard>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchByRegionInput {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

// ─── Anchor Helper ───────────────────────────────────────────────

fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}

// ─── CRUD Operations ─────────────────────────────────────────────

#[hdk_extern]
pub fn register_site(input: RegisterSiteInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let site = EnergySite {
        site_id: format!("site:{}:{}", input.name.replace(' ', "_"), now.as_micros()),
        name: input.name.clone(),
        lat: input.lat,
        lon: input.lon,
        energy_type: input.energy_type.clone(),
        capacity_mw: input.capacity_mw,
        status: input.status,
        country: input.country.clone(),
        state: input.state.clone(),
        developer: input.developer,
        scorecard: input.scorecard,
        created: now,
        updated: now,
    };

    let action_hash = create_entry(&EntryTypes::EnergySite(site.clone()))?;

    // Index: all sites
    create_link(
        anchor_hash("all_sites")?,
        action_hash.clone(),
        LinkTypes::AllSites,
        (),
    )?;

    // Index: by energy type
    create_link(
        anchor_hash(&format!("type:{}", input.energy_type))?,
        action_hash.clone(),
        LinkTypes::TypeToSites,
        (),
    )?;

    // Index: by geo bucket (1-degree grid)
    let geo_key = format!("geo:{}:{}", input.lat as i64, input.lon as i64);
    create_link(
        anchor_hash(&geo_key)?,
        action_hash.clone(),
        LinkTypes::RegionToSites,
        (),
    )?;

    // Index: by state (if US)
    if let Some(ref state) = input.state {
        create_link(
            anchor_hash(&format!("state:{}", state))?,
            action_hash.clone(),
            LinkTypes::StateToSites,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Site not found after creation".into())))
}

#[hdk_extern]
pub fn get_site(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn list_sites(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_sites")?, LinkTypes::AllSites)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

#[hdk_extern]
pub fn list_sites_by_type(energy_type: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("type:{}", energy_type))?,
            LinkTypes::TypeToSites,
        )?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

#[hdk_extern]
pub fn search_by_region(input: SearchByRegionInput) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();

    // Search all geo buckets that overlap the bounding box
    let min_lat = input.min_lat as i64;
    let max_lat = input.max_lat as i64;
    let min_lon = input.min_lon as i64;
    let max_lon = input.max_lon as i64;

    for lat in min_lat..=max_lat {
        for lon in min_lon..=max_lon {
            let geo_key = format!("geo:{}:{}", lat, lon);
            if let Ok(links) = get_links(
                LinkQuery::try_new(
                    anchor_hash(&geo_key)?,
                    LinkTypes::RegionToSites,
                )?,
                GetStrategy::default(),
            ) {
                for link in links {
                    if let Some(target) = link.target.into_action_hash() {
                        if let Some(record) = get(target, GetOptions::default())? {
                            records.push(record);
                        }
                    }
                }
            }
        }
    }

    Ok(records)
}
