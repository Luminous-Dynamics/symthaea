//! Food Production Integrity Zome
//! Entry types and validation for community food growing operations.
//!
//! Manages plots, crops, yield records, and season planning for
//! local food sovereignty.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// PLOT
// ============================================================================

/// Soil classification for a plot
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SoilType {
    Clay,
    Sandy,
    Loam,
    Silt,
    Peat,
    Chalk,
    Mixed,
}

/// Operational status of a plot
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PlotStatus {
    Active,
    Fallow,
    Preparing,
    Retired,
}

/// A registered growing plot
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Plot {
    pub id: String,
    pub name: String,
    pub area_sqm: f64,
    pub soil_type: SoilType,
    pub location_lat: f64,
    pub location_lon: f64,
    pub steward: AgentPubKey,
    pub status: PlotStatus,
}

// ============================================================================
// CROP
// ============================================================================

/// Lifecycle status of a crop
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CropStatus {
    Planned,
    Planted,
    Growing,
    Ready,
    Harvested,
    Failed,
}

/// A crop planted in a plot
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Crop {
    pub plot_hash: ActionHash,
    pub name: String,
    pub variety: String,
    pub planted_at: u64,
    pub expected_harvest: u64,
    pub status: CropStatus,
}

// ============================================================================
// YIELD RECORD
// ============================================================================

/// Quality grade for harvested produce
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum QualityGrade {
    Premium,
    Standard,
    Processing,
    Compost,
}

/// Record of a harvest from a crop
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct YieldRecord {
    pub crop_hash: ActionHash,
    pub quantity_kg: f64,
    pub quality_grade: QualityGrade,
    pub harvested_at: u64,
    pub notes: Option<String>,
}

// ============================================================================
// SEASON PLAN
// ============================================================================

/// Seasonal growing plan for a plot
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SeasonPlan {
    pub plot_hash: ActionHash,
    pub year: u32,
    pub season: String,
    pub planned_crops: Vec<String>,
    pub rotation_notes: Option<String>,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Plot(Plot),
    Crop(Crop),
    YieldRecord(YieldRecord),
    SeasonPlan(SeasonPlan),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllPlots,
    StewardToPlot,
    PlotToCrop,
    CropToYield,
    PlotToSeasonPlan,
    AgentToYield,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Plot(plot) => validate_plot(plot),
                EntryTypes::Crop(crop) => validate_crop(crop),
                EntryTypes::YieldRecord(yr) => validate_yield(yr),
                EntryTypes::SeasonPlan(sp) => validate_season_plan(sp),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Plot(plot) => validate_plot(plot),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_plot(plot: Plot) -> ExternResult<ValidateCallbackResult> {
    if plot.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Plot ID cannot be empty".into()));
    }
    if plot.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Plot name cannot be empty".into()));
    }
    if plot.area_sqm <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Area must be positive".into()));
    }
    if plot.location_lat < -90.0 || plot.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if plot.location_lon < -180.0 || plot.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_crop(crop: Crop) -> ExternResult<ValidateCallbackResult> {
    if crop.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Crop name cannot be empty".into()));
    }
    if crop.variety.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Crop variety cannot be empty".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_yield(yr: YieldRecord) -> ExternResult<ValidateCallbackResult> {
    if yr.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Yield quantity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_season_plan(sp: SeasonPlan) -> ExternResult<ValidateCallbackResult> {
    if sp.season.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Season cannot be empty".into()));
    }
    if sp.planned_crops.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Must plan at least one crop".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}
