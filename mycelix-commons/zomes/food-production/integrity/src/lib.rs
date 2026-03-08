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

/// Type of growing plot
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PlotType {
    Garden,
    FoodForest,
    Orchard,
    Greenhouse,
    Raised,
    Rooftop,
}

/// A registered growing plot
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Plot {
    pub id: String,
    pub name: String,
    pub area_sqm: f64,
    pub soil_type: SoilType,
    pub plot_type: PlotType,
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
    pub allergen_flags: Vec<String>,
    pub organic_certified: bool,
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
// GARDEN MEMBERSHIP
// ============================================================================

/// Role within a community garden
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum GardenRole {
    Steward,
    Volunteer,
    Member,
}

/// Membership record linking an agent to a plot
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GardenMembership {
    pub plot_hash: ActionHash,
    pub member: AgentPubKey,
    pub role: GardenRole,
    pub joined_at: u64,
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
    GardenMembership(GardenMembership),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllPlots,
    StewardToPlot,
    PlotToCrop,
    CropToYield,
    PlotToSeasonPlan,
    AgentToYield,
    PlotToMembers,
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
                EntryTypes::GardenMembership(m) => validate_garden_membership(m),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Plot(plot) => validate_plot(plot),
                EntryTypes::GardenMembership(m) => validate_garden_membership(m),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::AllPlots => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllPlots link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::StewardToPlot => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "StewardToPlot link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PlotToCrop => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PlotToCrop link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CropToYield => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CropToYield link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PlotToSeasonPlan => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PlotToSeasonPlan link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToYield => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToYield link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PlotToMembers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PlotToMembers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_plot(plot: Plot) -> ExternResult<ValidateCallbackResult> {
    if plot.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plot ID cannot be empty".into(),
        ));
    }
    if plot.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plot ID must be 256 characters or fewer".into(),
        ));
    }
    if plot.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Plot name cannot be empty".into(),
        ));
    }
    if plot.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Plot name must be 256 characters or fewer".into(),
        ));
    }
    if !plot.area_sqm.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Area must be a finite number".into(),
        ));
    }
    if plot.area_sqm <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Area must be positive".into(),
        ));
    }
    if !plot.location_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be a finite number".into(),
        ));
    }
    if plot.location_lat < -90.0 || plot.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !plot.location_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be a finite number".into(),
        ));
    }
    if plot.location_lon < -180.0 || plot.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_crop(crop: Crop) -> ExternResult<ValidateCallbackResult> {
    if crop.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Crop name cannot be empty".into(),
        ));
    }
    if crop.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Crop name must be 256 characters or fewer".into(),
        ));
    }
    if crop.variety.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Crop variety cannot be empty".into(),
        ));
    }
    if crop.variety.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Crop variety must be 128 characters or fewer".into(),
        ));
    }
    if crop.allergen_flags.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 allergen flags".into(),
        ));
    }
    for flag in &crop.allergen_flags {
        if flag.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Allergen flag cannot be empty".into(),
            ));
        }
        if flag.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Allergen flag too long (max 128 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_yield(yr: YieldRecord) -> ExternResult<ValidateCallbackResult> {
    if !yr.quantity_kg.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a finite number".into(),
        ));
    }
    if yr.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Yield quantity must be positive".into(),
        ));
    }
    if let Some(ref notes) = yr.notes {
        if notes.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Yield notes must be 4096 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_season_plan(sp: SeasonPlan) -> ExternResult<ValidateCallbackResult> {
    if sp.season.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Season cannot be empty".into(),
        ));
    }
    if sp.season.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Season must be 64 characters or fewer".into(),
        ));
    }
    if sp.planned_crops.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Must plan at least one crop".into(),
        ));
    }
    if sp.planned_crops.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot plan more than 100 crops".into(),
        ));
    }
    for crop_name in &sp.planned_crops {
        if crop_name.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each planned crop name must be 256 characters or fewer".into(),
            ));
        }
    }
    if let Some(ref notes) = sp.rotation_notes {
        if notes.len() > 2048 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rotation notes must be 2048 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_garden_membership(m: GardenMembership) -> ExternResult<ValidateCallbackResult> {
    if m.joined_at == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "GardenMembership joined_at cannot be zero".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }
    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn valid_plot() -> Plot {
        Plot {
            id: "plot-1".into(),
            name: "Community Garden".into(),
            area_sqm: 100.0,
            soil_type: SoilType::Loam,
            plot_type: PlotType::Garden,
            location_lat: 32.95,
            location_lon: -96.73,
            steward: fake_agent(),
            status: PlotStatus::Active,
        }
    }

    fn valid_crop() -> Crop {
        Crop {
            plot_hash: fake_action_hash(),
            name: "Tomato".into(),
            variety: "Cherokee Purple".into(),
            planted_at: 1700000000,
            expected_harvest: 1708000000,
            status: CropStatus::Planted,
            allergen_flags: vec![],
            organic_certified: false,
        }
    }

    fn valid_garden_membership() -> GardenMembership {
        GardenMembership {
            plot_hash: fake_action_hash(),
            member: fake_agent(),
            role: GardenRole::Member,
            joined_at: 1700000000,
        }
    }

    fn valid_yield_record() -> YieldRecord {
        YieldRecord {
            crop_hash: fake_action_hash(),
            quantity_kg: 25.5,
            quality_grade: QualityGrade::Premium,
            harvested_at: 1708000000,
            notes: Some("Great harvest".into()),
        }
    }

    fn valid_season_plan() -> SeasonPlan {
        SeasonPlan {
            plot_hash: fake_action_hash(),
            year: 2026,
            season: "Spring".into(),
            planned_crops: vec!["Tomato".into(), "Basil".into()],
            rotation_notes: None,
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_soil_type() {
        let types = vec![
            SoilType::Clay,
            SoilType::Sandy,
            SoilType::Loam,
            SoilType::Silt,
            SoilType::Peat,
            SoilType::Chalk,
            SoilType::Mixed,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: SoilType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, t);
        }
    }

    #[test]
    fn serde_roundtrip_plot_type() {
        let types = vec![
            PlotType::Garden,
            PlotType::FoodForest,
            PlotType::Orchard,
            PlotType::Greenhouse,
            PlotType::Raised,
            PlotType::Rooftop,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: PlotType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, t);
        }
    }

    #[test]
    fn serde_roundtrip_plot_status() {
        let statuses = vec![
            PlotStatus::Active,
            PlotStatus::Fallow,
            PlotStatus::Preparing,
            PlotStatus::Retired,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: PlotStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_crop_status() {
        let statuses = vec![
            CropStatus::Planned,
            CropStatus::Planted,
            CropStatus::Growing,
            CropStatus::Ready,
            CropStatus::Harvested,
            CropStatus::Failed,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: CropStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_quality_grade() {
        let grades = vec![
            QualityGrade::Premium,
            QualityGrade::Standard,
            QualityGrade::Processing,
            QualityGrade::Compost,
        ];
        for g in &grades {
            let json = serde_json::to_string(g).unwrap();
            let back: QualityGrade = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, g);
        }
    }

    #[test]
    fn serde_roundtrip_plot() {
        let p = valid_plot();
        let json = serde_json::to_string(&p).unwrap();
        let back: Plot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn serde_roundtrip_crop() {
        let c = valid_crop();
        let json = serde_json::to_string(&c).unwrap();
        let back: Crop = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn serde_roundtrip_yield_record() {
        let yr = valid_yield_record();
        let json = serde_json::to_string(&yr).unwrap();
        let back: YieldRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, yr);
    }

    #[test]
    fn serde_roundtrip_season_plan() {
        let sp = valid_season_plan();
        let json = serde_json::to_string(&sp).unwrap();
        let back: SeasonPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sp);
    }

    // ── validate_plot: id ───────────────────────────────────────────────

    #[test]
    fn valid_plot_passes() {
        assert_valid(validate_plot(valid_plot()));
    }

    #[test]
    fn plot_empty_id_rejected() {
        let mut p = valid_plot();
        p.id = String::new();
        assert_invalid(validate_plot(p), "Plot ID cannot be empty");
    }

    #[test]
    fn plot_whitespace_id_rejected() {
        let mut p = valid_plot();
        p.id = " ".into();
        assert_invalid(validate_plot(p), "Plot ID cannot be empty");
    }

    // ── validate_plot: name ─────────────────────────────────────────────

    #[test]
    fn plot_empty_name_rejected() {
        let mut p = valid_plot();
        p.name = String::new();
        assert_invalid(validate_plot(p), "Plot name cannot be empty");
    }

    #[test]
    fn plot_whitespace_name_rejected() {
        let mut p = valid_plot();
        p.name = "  ".into();
        assert_invalid(validate_plot(p), "Plot name cannot be empty");
    }

    // ── validate_plot: area_sqm ─────────────────────────────────────────

    #[test]
    fn plot_zero_area_rejected() {
        let mut p = valid_plot();
        p.area_sqm = 0.0;
        assert_invalid(validate_plot(p), "Area must be positive");
    }

    #[test]
    fn plot_negative_area_rejected() {
        let mut p = valid_plot();
        p.area_sqm = -10.0;
        assert_invalid(validate_plot(p), "Area must be positive");
    }

    #[test]
    fn plot_barely_positive_area_valid() {
        let mut p = valid_plot();
        p.area_sqm = 0.01;
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_large_area_valid() {
        let mut p = valid_plot();
        p.area_sqm = 100000.0;
        assert_valid(validate_plot(p));
    }

    // ── validate_plot: latitude ─────────────────────────────────────────

    #[test]
    fn plot_lat_too_low_rejected() {
        let mut p = valid_plot();
        p.location_lat = -91.0;
        assert_invalid(validate_plot(p), "Latitude must be between -90 and 90");
    }

    #[test]
    fn plot_lat_too_high_rejected() {
        let mut p = valid_plot();
        p.location_lat = 91.0;
        assert_invalid(validate_plot(p), "Latitude must be between -90 and 90");
    }

    #[test]
    fn plot_lat_at_boundary_neg90_valid() {
        let mut p = valid_plot();
        p.location_lat = -90.0;
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_lat_at_boundary_pos90_valid() {
        let mut p = valid_plot();
        p.location_lat = 90.0;
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_lat_zero_valid() {
        let mut p = valid_plot();
        p.location_lat = 0.0;
        assert_valid(validate_plot(p));
    }

    // ── validate_plot: longitude ────────────────────────────────────────

    #[test]
    fn plot_lon_too_low_rejected() {
        let mut p = valid_plot();
        p.location_lon = -181.0;
        assert_invalid(validate_plot(p), "Longitude must be between -180 and 180");
    }

    #[test]
    fn plot_lon_too_high_rejected() {
        let mut p = valid_plot();
        p.location_lon = 181.0;
        assert_invalid(validate_plot(p), "Longitude must be between -180 and 180");
    }

    #[test]
    fn plot_lon_at_boundary_neg180_valid() {
        let mut p = valid_plot();
        p.location_lon = -180.0;
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_lon_at_boundary_pos180_valid() {
        let mut p = valid_plot();
        p.location_lon = 180.0;
        assert_valid(validate_plot(p));
    }

    // ── validate_plot: soil type and status variants ────────────────────

    #[test]
    fn plot_all_soil_types_valid() {
        for st in [
            SoilType::Clay,
            SoilType::Sandy,
            SoilType::Loam,
            SoilType::Silt,
            SoilType::Peat,
            SoilType::Chalk,
            SoilType::Mixed,
        ] {
            let mut p = valid_plot();
            p.soil_type = st;
            assert_valid(validate_plot(p));
        }
    }

    #[test]
    fn plot_all_plot_types_valid() {
        for pt in [
            PlotType::Garden,
            PlotType::FoodForest,
            PlotType::Orchard,
            PlotType::Greenhouse,
            PlotType::Raised,
            PlotType::Rooftop,
        ] {
            let mut p = valid_plot();
            p.plot_type = pt;
            assert_valid(validate_plot(p));
        }
    }

    #[test]
    fn plot_all_statuses_valid() {
        for status in [
            PlotStatus::Active,
            PlotStatus::Fallow,
            PlotStatus::Preparing,
            PlotStatus::Retired,
        ] {
            let mut p = valid_plot();
            p.status = status;
            assert_valid(validate_plot(p));
        }
    }

    // ── validate_plot: combined invalid ─────────────────────────────────

    #[test]
    fn plot_empty_id_rejects_before_empty_name() {
        let mut p = valid_plot();
        p.id = String::new();
        p.name = String::new();
        assert_invalid(validate_plot(p), "Plot ID cannot be empty");
    }

    #[test]
    fn plot_empty_name_rejects_before_zero_area() {
        let mut p = valid_plot();
        p.name = String::new();
        p.area_sqm = 0.0;
        assert_invalid(validate_plot(p), "Plot name cannot be empty");
    }

    // ── validate_crop: name ─────────────────────────────────────────────

    #[test]
    fn valid_crop_passes() {
        assert_valid(validate_crop(valid_crop()));
    }

    #[test]
    fn crop_empty_name_rejected() {
        let mut c = valid_crop();
        c.name = String::new();
        assert_invalid(validate_crop(c), "Crop name cannot be empty");
    }

    #[test]
    fn crop_whitespace_name_rejected() {
        let mut c = valid_crop();
        c.name = " ".into();
        assert_invalid(validate_crop(c), "Crop name cannot be empty");
    }

    // ── validate_crop: variety ──────────────────────────────────────────

    #[test]
    fn crop_empty_variety_rejected() {
        let mut c = valid_crop();
        c.variety = String::new();
        assert_invalid(validate_crop(c), "Crop variety cannot be empty");
    }

    #[test]
    fn crop_whitespace_variety_rejected() {
        let mut c = valid_crop();
        c.variety = "  ".into();
        assert_invalid(validate_crop(c), "Crop variety cannot be empty");
    }

    // ── validate_crop: status variants ──────────────────────────────────

    #[test]
    fn crop_all_statuses_valid() {
        for status in [
            CropStatus::Planned,
            CropStatus::Planted,
            CropStatus::Growing,
            CropStatus::Ready,
            CropStatus::Harvested,
            CropStatus::Failed,
        ] {
            let mut c = valid_crop();
            c.status = status;
            assert_valid(validate_crop(c));
        }
    }

    // ── validate_crop: combined invalid ─────────────────────────────────

    #[test]
    fn crop_empty_name_with_empty_variety_rejects_name_first() {
        let mut c = valid_crop();
        c.name = String::new();
        c.variety = String::new();
        assert_invalid(validate_crop(c), "Crop name cannot be empty");
    }

    // ── validate_yield: quantity_kg ─────────────────────────────────────

    #[test]
    fn valid_yield_passes() {
        assert_valid(validate_yield(valid_yield_record()));
    }

    #[test]
    fn yield_zero_quantity_rejected() {
        let mut yr = valid_yield_record();
        yr.quantity_kg = 0.0;
        assert_invalid(validate_yield(yr), "Yield quantity must be positive");
    }

    #[test]
    fn yield_negative_quantity_rejected() {
        let mut yr = valid_yield_record();
        yr.quantity_kg = -5.0;
        assert_invalid(validate_yield(yr), "Yield quantity must be positive");
    }

    #[test]
    fn yield_barely_positive_quantity_valid() {
        let mut yr = valid_yield_record();
        yr.quantity_kg = 0.001;
        assert_valid(validate_yield(yr));
    }

    #[test]
    fn yield_large_quantity_valid() {
        let mut yr = valid_yield_record();
        yr.quantity_kg = 99999.0;
        assert_valid(validate_yield(yr));
    }

    // ── validate_yield: quality grade variants ──────────────────────────

    #[test]
    fn yield_all_quality_grades_valid() {
        for grade in [
            QualityGrade::Premium,
            QualityGrade::Standard,
            QualityGrade::Processing,
            QualityGrade::Compost,
        ] {
            let mut yr = valid_yield_record();
            yr.quality_grade = grade;
            assert_valid(validate_yield(yr));
        }
    }

    // ── validate_yield: optional notes ──────────────────────────────────

    #[test]
    fn yield_no_notes_valid() {
        let mut yr = valid_yield_record();
        yr.notes = None;
        assert_valid(validate_yield(yr));
    }

    #[test]
    fn yield_empty_notes_string_valid() {
        let mut yr = valid_yield_record();
        yr.notes = Some(String::new());
        assert_valid(validate_yield(yr));
    }

    // ── validate_season_plan: season ────────────────────────────────────

    #[test]
    fn valid_season_plan_passes() {
        assert_valid(validate_season_plan(valid_season_plan()));
    }

    #[test]
    fn season_plan_empty_season_rejected() {
        let mut sp = valid_season_plan();
        sp.season = String::new();
        assert_invalid(validate_season_plan(sp), "Season cannot be empty");
    }

    #[test]
    fn season_plan_whitespace_season_rejected() {
        let mut sp = valid_season_plan();
        sp.season = " ".into();
        assert_invalid(validate_season_plan(sp), "Season cannot be empty");
    }

    // ── validate_season_plan: planned_crops ─────────────────────────────

    #[test]
    fn season_plan_no_crops_rejected() {
        let mut sp = valid_season_plan();
        sp.planned_crops = vec![];
        assert_invalid(validate_season_plan(sp), "Must plan at least one crop");
    }

    #[test]
    fn season_plan_one_crop_valid() {
        let mut sp = valid_season_plan();
        sp.planned_crops = vec!["Corn".into()];
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_many_crops_valid() {
        let mut sp = valid_season_plan();
        sp.planned_crops = (0..50).map(|i| format!("crop_{i}")).collect();
        assert_valid(validate_season_plan(sp));
    }

    // ── validate_season_plan: optional fields ───────────────────────────

    #[test]
    fn season_plan_with_rotation_notes_valid() {
        let mut sp = valid_season_plan();
        sp.rotation_notes = Some("Rotate legumes after heavy feeders".into());
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_year_zero_valid() {
        let mut sp = valid_season_plan();
        sp.year = 0;
        assert_valid(validate_season_plan(sp));
    }

    // ── validate_season_plan: combined invalid ──────────────────────────

    #[test]
    fn season_plan_empty_season_with_no_crops_rejects_season_first() {
        let mut sp = valid_season_plan();
        sp.season = String::new();
        sp.planned_crops = vec![];
        assert_invalid(validate_season_plan(sp), "Season cannot be empty");
    }

    // ── Anchor test ─────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_plots".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ── validate_plot: string length limits ─────────────────────────────

    #[test]
    fn plot_id_too_long_rejected() {
        let mut p = valid_plot();
        p.id = "x".repeat(257);
        assert_invalid(validate_plot(p), "Plot ID must be 256 characters or fewer");
    }

    #[test]
    fn plot_id_exactly_64_chars_accepted() {
        let mut p = valid_plot();
        p.id = "x".repeat(64);
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_id_63_chars_accepted() {
        let mut p = valid_plot();
        p.id = "x".repeat(63);
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_name_too_long_rejected() {
        let mut p = valid_plot();
        p.name = "x".repeat(257);
        assert_invalid(
            validate_plot(p),
            "Plot name must be 256 characters or fewer",
        );
    }

    #[test]
    fn plot_name_exactly_256_chars_accepted() {
        let mut p = valid_plot();
        p.name = "x".repeat(256);
        assert_valid(validate_plot(p));
    }

    #[test]
    fn plot_name_255_chars_accepted() {
        let mut p = valid_plot();
        p.name = "x".repeat(255);
        assert_valid(validate_plot(p));
    }

    // ── validate_crop: string length limits ─────────────────────────────

    #[test]
    fn crop_name_too_long_rejected() {
        let mut c = valid_crop();
        c.name = "x".repeat(257);
        assert_invalid(
            validate_crop(c),
            "Crop name must be 256 characters or fewer",
        );
    }

    #[test]
    fn crop_name_exactly_256_chars_accepted() {
        let mut c = valid_crop();
        c.name = "x".repeat(256);
        assert_valid(validate_crop(c));
    }

    #[test]
    fn crop_name_255_chars_accepted() {
        let mut c = valid_crop();
        c.name = "x".repeat(255);
        assert_valid(validate_crop(c));
    }

    #[test]
    fn crop_variety_too_long_rejected() {
        let mut c = valid_crop();
        c.variety = "x".repeat(129);
        assert_invalid(
            validate_crop(c),
            "Crop variety must be 128 characters or fewer",
        );
    }

    #[test]
    fn crop_variety_exactly_128_chars_accepted() {
        let mut c = valid_crop();
        c.variety = "x".repeat(128);
        assert_valid(validate_crop(c));
    }

    #[test]
    fn crop_variety_127_chars_accepted() {
        let mut c = valid_crop();
        c.variety = "x".repeat(127);
        assert_valid(validate_crop(c));
    }

    // ── validate_yield: notes string length limit ───────────────────────

    #[test]
    fn yield_notes_too_long_rejected() {
        let mut yr = valid_yield_record();
        yr.notes = Some("x".repeat(4097));
        assert_invalid(
            validate_yield(yr),
            "Yield notes must be 4096 characters or fewer",
        );
    }

    #[test]
    fn yield_notes_exactly_4096_chars_accepted() {
        let mut yr = valid_yield_record();
        yr.notes = Some("x".repeat(4096));
        assert_valid(validate_yield(yr));
    }

    #[test]
    fn yield_notes_4095_chars_accepted() {
        let mut yr = valid_yield_record();
        yr.notes = Some("x".repeat(4095));
        assert_valid(validate_yield(yr));
    }

    // ── validate_season_plan: string length limits ──────────────────────

    #[test]
    fn season_plan_season_too_long_rejected() {
        let mut sp = valid_season_plan();
        sp.season = "x".repeat(65);
        assert_invalid(
            validate_season_plan(sp),
            "Season must be 64 characters or fewer",
        );
    }

    #[test]
    fn season_plan_season_exactly_64_chars_accepted() {
        let mut sp = valid_season_plan();
        sp.season = "x".repeat(64);
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_season_63_chars_accepted() {
        let mut sp = valid_season_plan();
        sp.season = "x".repeat(63);
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_rotation_notes_too_long_rejected() {
        let mut sp = valid_season_plan();
        sp.rotation_notes = Some("x".repeat(2049));
        assert_invalid(
            validate_season_plan(sp),
            "Rotation notes must be 2048 characters or fewer",
        );
    }

    #[test]
    fn season_plan_rotation_notes_exactly_2048_chars_accepted() {
        let mut sp = valid_season_plan();
        sp.rotation_notes = Some("x".repeat(2048));
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_rotation_notes_2047_chars_accepted() {
        let mut sp = valid_season_plan();
        sp.rotation_notes = Some("x".repeat(2047));
        assert_valid(validate_season_plan(sp));
    }

    // ── validate_season_plan: Vec length limits ─────────────────────────

    #[test]
    fn season_plan_too_many_planned_crops_rejected() {
        let mut sp = valid_season_plan();
        sp.planned_crops = (0..101).map(|i| format!("crop_{i}")).collect();
        assert_invalid(validate_season_plan(sp), "Cannot plan more than 100 crops");
    }

    #[test]
    fn season_plan_exactly_100_planned_crops_accepted() {
        let mut sp = valid_season_plan();
        sp.planned_crops = (0..100).map(|i| format!("crop_{i}")).collect();
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_99_planned_crops_accepted() {
        let mut sp = valid_season_plan();
        sp.planned_crops = (0..99).map(|i| format!("crop_{i}")).collect();
        assert_valid(validate_season_plan(sp));
    }

    // ── validate_season_plan: individual crop name length ───────────────

    #[test]
    fn season_plan_planned_crop_name_too_long_rejected() {
        let mut sp = valid_season_plan();
        sp.planned_crops = vec!["x".repeat(257)];
        assert_invalid(
            validate_season_plan(sp),
            "Each planned crop name must be 256 characters or fewer",
        );
    }

    #[test]
    fn season_plan_planned_crop_name_exactly_256_chars_accepted() {
        let mut sp = valid_season_plan();
        sp.planned_crops = vec!["x".repeat(256)];
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_planned_crop_name_255_chars_accepted() {
        let mut sp = valid_season_plan();
        sp.planned_crops = vec!["x".repeat(255)];
        assert_valid(validate_season_plan(sp));
    }

    #[test]
    fn season_plan_second_crop_name_too_long_rejected() {
        let mut sp = valid_season_plan();
        sp.planned_crops = vec![
            "Valid crop".to_string(),
            "x".repeat(257),
            "Another valid".to_string(),
        ];
        assert_invalid(
            validate_season_plan(sp),
            "Each planned crop name must be 256 characters or fewer",
        );
    }

    // ── Link tag length validation tests ────────────────────────────────

    fn validate_link_tag(link_type: &LinkTypes, tag_len: usize) -> ValidateCallbackResult {
        let tag = LinkTag(vec![0u8; tag_len]);
        let max = match link_type {
            LinkTypes::AllPlots
            | LinkTypes::StewardToPlot
            | LinkTypes::PlotToCrop
            | LinkTypes::CropToYield
            | LinkTypes::PlotToSeasonPlan
            | LinkTypes::AgentToYield
            | LinkTypes::PlotToMembers => 256,
        };
        let name = match link_type {
            LinkTypes::AllPlots => "AllPlots",
            LinkTypes::StewardToPlot => "StewardToPlot",
            LinkTypes::PlotToCrop => "PlotToCrop",
            LinkTypes::CropToYield => "CropToYield",
            LinkTypes::PlotToSeasonPlan => "PlotToSeasonPlan",
            LinkTypes::AgentToYield => "AgentToYield",
            LinkTypes::PlotToMembers => "PlotToMembers",
        };
        if tag.0.len() > max {
            ValidateCallbackResult::Invalid(format!(
                "{} link tag too long (max {} bytes)",
                name, max
            ))
        } else {
            ValidateCallbackResult::Valid
        }
    }

    #[test]
    fn test_link_all_plots_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllPlots, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_plots_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllPlots, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_steward_to_plot_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::StewardToPlot, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_steward_to_plot_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::StewardToPlot, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_plot_to_crop_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::PlotToCrop, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_plot_to_crop_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::PlotToCrop, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_crop_to_yield_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::CropToYield, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_crop_to_yield_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::CropToYield, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_plot_to_season_plan_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::PlotToSeasonPlan, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_plot_to_season_plan_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::PlotToSeasonPlan, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_agent_to_yield_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToYield, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_agent_to_yield_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToYield, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── PlotToMembers link tag tests ────────────────────────────────────

    #[test]
    fn test_link_plot_to_members_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::PlotToMembers, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_plot_to_members_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::PlotToMembers, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Serde roundtrip: GardenRole ─────────────────────────────────────

    #[test]
    fn serde_roundtrip_garden_role_all_variants() {
        let roles = vec![
            GardenRole::Steward,
            GardenRole::Volunteer,
            GardenRole::Member,
        ];
        for r in &roles {
            let json = serde_json::to_string(r).unwrap();
            let back: GardenRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, r);
        }
    }

    // ── Serde roundtrip: GardenMembership ───────────────────────────────

    #[test]
    fn serde_roundtrip_garden_membership() {
        let m = valid_garden_membership();
        let json = serde_json::to_string(&m).unwrap();
        let back: GardenMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    // ── Serde roundtrip: Crop with allergens ────────────────────────────

    #[test]
    fn serde_roundtrip_crop_with_allergens() {
        let mut c = valid_crop();
        c.allergen_flags = vec!["gluten".into(), "soy".into()];
        c.organic_certified = true;
        let json = serde_json::to_string(&c).unwrap();
        let back: Crop = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    // ── validate_crop: allergen flags ───────────────────────────────────

    #[test]
    fn crop_with_allergen_flags_valid() {
        let mut c = valid_crop();
        c.allergen_flags = vec!["gluten".into(), "nuts".into()];
        assert_valid(validate_crop(c));
    }

    #[test]
    fn crop_too_many_allergens_rejected() {
        let mut c = valid_crop();
        c.allergen_flags = (0..51).map(|i| format!("allergen_{i}")).collect();
        assert_invalid(validate_crop(c), "Cannot have more than 50 allergen flags");
    }

    #[test]
    fn crop_empty_allergen_flag_rejected() {
        let mut c = valid_crop();
        c.allergen_flags = vec!["".into()];
        assert_invalid(validate_crop(c), "Allergen flag cannot be empty");
    }

    #[test]
    fn crop_allergen_flag_too_long_rejected() {
        let mut c = valid_crop();
        c.allergen_flags = vec!["x".repeat(129)];
        assert_invalid(validate_crop(c), "Allergen flag too long (max 128 chars)");
    }

    #[test]
    fn crop_exactly_50_allergen_flags_accepted() {
        let mut c = valid_crop();
        c.allergen_flags = (0..50).map(|i| format!("allergen_{i}")).collect();
        assert_valid(validate_crop(c));
    }

    #[test]
    fn crop_whitespace_allergen_flag_rejected() {
        let mut c = valid_crop();
        c.allergen_flags = vec!["  ".into()];
        assert_invalid(validate_crop(c), "Allergen flag cannot be empty");
    }

    #[test]
    fn crop_organic_true_valid() {
        let mut c = valid_crop();
        c.organic_certified = true;
        assert_valid(validate_crop(c));
    }

    // ── validate_garden_membership ──────────────────────────────────────

    #[test]
    fn garden_membership_valid() {
        assert_valid(validate_garden_membership(valid_garden_membership()));
    }

    #[test]
    fn garden_membership_zero_joined_at_rejected() {
        let mut m = valid_garden_membership();
        m.joined_at = 0;
        assert_invalid(
            validate_garden_membership(m),
            "GardenMembership joined_at cannot be zero",
        );
    }

    // ── NaN/Infinity bypass hardening tests ────────────────────────────

    #[test]
    fn plot_nan_area_rejected() {
        let mut p = valid_plot();
        p.area_sqm = f64::NAN;
        assert_invalid(validate_plot(p), "Area must be a finite number");
    }

    #[test]
    fn plot_infinity_area_rejected() {
        let mut p = valid_plot();
        p.area_sqm = f64::INFINITY;
        assert_invalid(validate_plot(p), "Area must be a finite number");
    }

    #[test]
    fn plot_neg_infinity_area_rejected() {
        let mut p = valid_plot();
        p.area_sqm = f64::NEG_INFINITY;
        assert_invalid(validate_plot(p), "Area must be a finite number");
    }

    #[test]
    fn plot_nan_lat_rejected() {
        let mut p = valid_plot();
        p.location_lat = f64::NAN;
        assert_invalid(validate_plot(p), "Latitude must be a finite number");
    }

    #[test]
    fn plot_infinity_lat_rejected() {
        let mut p = valid_plot();
        p.location_lat = f64::INFINITY;
        assert_invalid(validate_plot(p), "Latitude must be a finite number");
    }

    #[test]
    fn plot_nan_lon_rejected() {
        let mut p = valid_plot();
        p.location_lon = f64::NAN;
        assert_invalid(validate_plot(p), "Longitude must be a finite number");
    }

    #[test]
    fn plot_infinity_lon_rejected() {
        let mut p = valid_plot();
        p.location_lon = f64::INFINITY;
        assert_invalid(validate_plot(p), "Longitude must be a finite number");
    }

    #[test]
    fn yield_nan_quantity_rejected() {
        let mut yr = valid_yield_record();
        yr.quantity_kg = f64::NAN;
        assert_invalid(validate_yield(yr), "Quantity must be a finite number");
    }

    #[test]
    fn yield_infinity_quantity_rejected() {
        let mut yr = valid_yield_record();
        yr.quantity_kg = f64::INFINITY;
        assert_invalid(validate_yield(yr), "Quantity must be a finite number");
    }

    #[test]
    fn garden_membership_all_roles_valid() {
        for role in [
            GardenRole::Steward,
            GardenRole::Volunteer,
            GardenRole::Member,
        ] {
            let mut m = valid_garden_membership();
            m.role = role;
            assert_valid(validate_garden_membership(m));
        }
    }
}
