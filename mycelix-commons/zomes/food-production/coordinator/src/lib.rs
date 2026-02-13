//! Food Production Coordinator Zome
//! Business logic for plot management, crop tracking, and harvest recording.

use food_production_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// PLOT MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn register_plot(plot: Plot) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Plot(plot.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_plots".to_string())))?;
    create_link(anchor_hash("all_plots")?, action_hash.clone(), LinkTypes::AllPlots, ())?;
    create_link(plot.steward, action_hash.clone(), LinkTypes::StewardToPlot, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created plot".into())))
}

#[hdk_extern]
pub fn get_plot(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_all_plots(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_plots")?, LinkTypes::AllPlots)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CROP MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn plant_crop(crop: Crop) -> ExternResult<Record> {
    // Verify plot exists
    let _plot = get(crop.plot_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Plot not found".into())))?;

    let action_hash = create_entry(&EntryTypes::Crop(crop.clone()))?;
    create_link(crop.plot_hash, action_hash.clone(), LinkTypes::PlotToCrop, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created crop".into())))
}

#[hdk_extern]
pub fn get_plot_crops(plot_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(plot_hash, LinkTypes::PlotToCrop)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// HARVEST / YIELD
// ============================================================================

#[hdk_extern]
pub fn record_harvest(yr: YieldRecord) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Verify crop exists
    let _crop = get(yr.crop_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Crop not found".into())))?;

    let action_hash = create_entry(&EntryTypes::YieldRecord(yr.clone()))?;
    create_link(yr.crop_hash, action_hash.clone(), LinkTypes::CropToYield, ())?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToYield, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created yield record".into())))
}

#[hdk_extern]
pub fn get_crop_yields(crop_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(crop_hash, LinkTypes::CropToYield)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// SEASON PLANNING
// ============================================================================

#[hdk_extern]
pub fn create_season_plan(plan: SeasonPlan) -> ExternResult<Record> {
    let _plot = get(plan.plot_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Plot not found".into())))?;

    let action_hash = create_entry(&EntryTypes::SeasonPlan(plan.clone()))?;
    create_link(plan.plot_hash, action_hash.clone(), LinkTypes::PlotToSeasonPlan, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created season plan".into())))
}

#[hdk_extern]
pub fn get_season_plans(plot_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(plot_hash, LinkTypes::PlotToSeasonPlan)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
