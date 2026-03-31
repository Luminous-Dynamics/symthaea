// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use atlas_infrastructure_integrity::*;

// ─── Input Types ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterGeothermalNodeInput {
    pub name: String,
    pub region: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_mw: f64,
    pub temperature_c: u32,
    pub node_type: String,
    pub status: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterCorridorInput {
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
    pub blast_doors: u32,
    pub build_years: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterVaultInput {
    pub vault_id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_people: u32,
    pub heat_rejection_mw: f64,
    pub blast_doors: u32,
    pub status: String,
    pub terra_lumina_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterTerraLuminaInput {
    pub site_id: String,
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

// ─── Anchor Helper ───────────────────────────────────────────────

fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

// ─── Geothermal Nodes ────────────────────────────────────────────

#[hdk_extern]
pub fn register_geothermal_node(input: RegisterGeothermalNodeInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let node = GeothermalNode {
        name: input.name,
        region: input.region,
        lat: input.lat,
        lon: input.lon,
        capacity_mw: input.capacity_mw,
        temperature_c: input.temperature_c,
        node_type: input.node_type,
        status: input.status,
        created: now,
    };

    let hash = create_entry(&EntryTypes::GeothermalNode(node))?;
    create_link(anchor_hash("all_geothermal_nodes")?, hash.clone(), LinkTypes::AllGeothermalNodes, ())?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Node not found".into())))
}

#[hdk_extern]
pub fn list_geothermal_nodes(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_geothermal_nodes")?, LinkTypes::AllGeothermalNodes)?,
        GetStrategy::default(),
    )?;
    links_to_records(links)
}

// ─── Maglev Corridors ────────────────────────────────────────────

#[hdk_extern]
pub fn register_corridor(input: RegisterCorridorInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let corridor = MaglevCorridor {
        name: input.name,
        from_name: input.from_name,
        from_lat: input.from_lat,
        from_lon: input.from_lon,
        to_name: input.to_name,
        to_lat: input.to_lat,
        to_lon: input.to_lon,
        distance_km: input.distance_km,
        travel_time_min: input.travel_time_min,
        submarine: input.submarine,
        seismic_risk: input.seismic_risk,
        cost_billion_usd: input.cost_billion_usd,
        capacity_pax_hr: input.capacity_pax_hr,
        geothermal_powered: input.geothermal_powered,
        blast_doors: input.blast_doors,
        build_years: input.build_years,
        created: now,
    };

    let hash = create_entry(&EntryTypes::MaglevCorridor(corridor))?;
    create_link(anchor_hash("all_corridors")?, hash.clone(), LinkTypes::AllCorridors, ())?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Corridor not found".into())))
}

#[hdk_extern]
pub fn list_corridors(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_corridors")?, LinkTypes::AllCorridors)?,
        GetStrategy::default(),
    )?;
    links_to_records(links)
}

// ─── Resontia Vaults ─────────────────────────────────────────────

#[hdk_extern]
pub fn register_vault(input: RegisterVaultInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let vault = ResontiaVault {
        vault_id: input.vault_id,
        name: input.name,
        lat: input.lat,
        lon: input.lon,
        capacity_people: input.capacity_people,
        heat_rejection_mw: input.heat_rejection_mw,
        blast_doors: input.blast_doors,
        status: input.status,
        terra_lumina_id: input.terra_lumina_id,
        created: now,
    };

    let hash = create_entry(&EntryTypes::ResontiaVault(vault))?;
    create_link(anchor_hash("all_vaults")?, hash.clone(), LinkTypes::AllVaults, ())?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Vault not found".into())))
}

#[hdk_extern]
pub fn list_vaults(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_vaults")?, LinkTypes::AllVaults)?,
        GetStrategy::default(),
    )?;
    links_to_records(links)
}

// ─── Terra Lumina Sites ──────────────────────────────────────────

#[hdk_extern]
pub fn register_terra_lumina_site(input: RegisterTerraLuminaInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let site = TerraLuminaSite {
        site_id: input.site_id,
        name: input.name,
        country: input.country,
        lat: input.lat,
        lon: input.lon,
        score: input.score,
        tier: input.tier,
        geothermal_gw: input.geothermal_gw,
        solar_gw: input.solar_gw,
        hydro_gw: input.hydro_gw,
        total_renewable_gw: input.total_renewable_gw,
        phase1_billion_eur: input.phase1_billion_eur,
        total_billion_eur: input.total_billion_eur,
        irr_percent: input.irr_percent,
        created: now,
    };

    let hash = create_entry(&EntryTypes::TerraLuminaSite(site))?;
    create_link(anchor_hash("all_terra_lumina_sites")?, hash.clone(), LinkTypes::AllTerraLuminaSites, ())?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Site not found".into())))
}

#[hdk_extern]
pub fn list_terra_lumina_sites(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_terra_lumina_sites")?, LinkTypes::AllTerraLuminaSites)?,
        GetStrategy::default(),
    )?;
    links_to_records(links)
}

// ─── Fossil Deposits ────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterFossilDepositInput {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub fuel_type: String,
    pub proven_reserves_mboe: f64,
    pub annual_production_mboe: f64,
    pub status: String,
    pub country: String,
    pub discovery_year: u32,
}

#[hdk_extern]
pub fn register_fossil_deposit(input: RegisterFossilDepositInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let deposit = FossilDeposit {
        name: input.name,
        lat: input.lat,
        lon: input.lon,
        fuel_type: input.fuel_type.clone(),
        proven_reserves_mboe: input.proven_reserves_mboe,
        annual_production_mboe: input.annual_production_mboe,
        status: input.status,
        country: input.country,
        discovery_year: input.discovery_year,
        created: now,
    };

    let hash = create_entry(&EntryTypes::FossilDeposit(deposit))?;
    create_link(anchor_hash("all_fossil_deposits")?, hash.clone(), LinkTypes::AllFossilDeposits, ())?;
    create_link(
        anchor_hash(&format!("fossil_type:{}", input.fuel_type))?,
        hash.clone(),
        LinkTypes::TypeToFossilDeposits,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Deposit not found".into())))
}

#[hdk_extern]
pub fn list_fossil_deposits(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_fossil_deposits")?, LinkTypes::AllFossilDeposits)?,
        GetStrategy::default(),
    )?;
    links_to_records(links)
}

// ─── Helper ──────────────────────────────────────────────────────

fn links_to_records(links: Vec<Link>) -> ExternResult<Vec<Record>> {
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
