//! Food Production Coordinator Zome
//! Business logic for plot management, crop tracking, and harvest recording.

use food_production_integrity::*;
use hdk::prelude::*;

// ============================================================================
// BRIDGE SIGNAL (for cross-domain UI notification)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub source_zome: String,
    pub payload: String,
}

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
    create_link(yr.crop_hash.clone(), action_hash.clone(), LinkTypes::CropToYield, ())?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToYield, ())?;

    // Emit bridge signal so the UI can update with new harvest data
    let _ = emit_signal(&BridgeEventSignal {
        event_type: "harvest_recorded".to_string(),
        source_zome: "food_production".to_string(),
        payload: format!(
            r#"{{"yield_hash":"{}","crop_hash":"{}","quantity_kg":{}}}"#,
            action_hash, yr.crop_hash, yr.quantity_kg,
        ),
    });

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            event_type: "harvest_recorded".to_string(),
            source_zome: "food_production".to_string(),
            payload: r#"{"yield_hash":"abc","crop_hash":"def","quantity_kg":42.5}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type, "harvest_recorded");
        assert_eq!(decoded.source_zome, "food_production");
        assert!(decoded.payload.contains("42.5"));
    }

    #[test]
    fn bridge_event_signal_empty_payload() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            source_zome: "food_production".to_string(),
            payload: String::new(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payload, "");
    }

    #[test]
    fn bridge_event_signal_json_structure() {
        let signal = BridgeEventSignal {
            event_type: "harvest_recorded".to_string(),
            source_zome: "food_production".to_string(),
            payload: "{}".to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("\"event_type\""));
        assert!(json.contains("\"source_zome\""));
        assert!(json.contains("\"payload\""));
    }

    // ========================================================================
    // SoilType enum serde roundtrip
    // ========================================================================

    #[test]
    fn soil_type_all_variants_serde_roundtrip() {
        let variants = vec![
            SoilType::Clay,
            SoilType::Sandy,
            SoilType::Loam,
            SoilType::Silt,
            SoilType::Peat,
            SoilType::Chalk,
            SoilType::Mixed,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: SoilType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // PlotStatus enum serde roundtrip
    // ========================================================================

    #[test]
    fn plot_status_all_variants_serde_roundtrip() {
        let variants = vec![
            PlotStatus::Active,
            PlotStatus::Fallow,
            PlotStatus::Preparing,
            PlotStatus::Retired,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: PlotStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // CropStatus enum serde roundtrip
    // ========================================================================

    #[test]
    fn crop_status_all_variants_serde_roundtrip() {
        let variants = vec![
            CropStatus::Planned,
            CropStatus::Planted,
            CropStatus::Growing,
            CropStatus::Ready,
            CropStatus::Harvested,
            CropStatus::Failed,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CropStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // QualityGrade enum serde roundtrip
    // ========================================================================

    #[test]
    fn quality_grade_all_variants_serde_roundtrip() {
        let variants = vec![
            QualityGrade::Premium,
            QualityGrade::Standard,
            QualityGrade::Processing,
            QualityGrade::Compost,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: QualityGrade = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Plot struct serde roundtrip
    // ========================================================================

    #[test]
    fn plot_serde_roundtrip() {
        let plot = Plot {
            id: "plot-42".to_string(),
            name: "Sunrise Garden".to_string(),
            area_sqm: 250.5,
            soil_type: SoilType::Loam,
            location_lat: 32.95,
            location_lon: -96.73,
            steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
            status: PlotStatus::Active,
        };
        let json = serde_json::to_string(&plot).unwrap();
        let decoded: Plot = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "plot-42");
        assert_eq!(decoded.name, "Sunrise Garden");
        assert_eq!(decoded.area_sqm, 250.5);
        assert_eq!(decoded.soil_type, SoilType::Loam);
        assert_eq!(decoded.location_lat, 32.95);
        assert_eq!(decoded.location_lon, -96.73);
        assert_eq!(decoded.status, PlotStatus::Active);
    }

    #[test]
    fn plot_serde_all_soil_and_status_combinations() {
        let soils = [SoilType::Clay, SoilType::Peat, SoilType::Chalk];
        let statuses = [PlotStatus::Fallow, PlotStatus::Preparing, PlotStatus::Retired];
        for (soil, status) in soils.iter().zip(statuses.iter()) {
            let plot = Plot {
                id: "p-combo".to_string(),
                name: "Combo Plot".to_string(),
                area_sqm: 10.0,
                soil_type: soil.clone(),
                location_lat: 0.0,
                location_lon: 0.0,
                steward: AgentPubKey::from_raw_36(vec![0xab; 36]),
                status: status.clone(),
            };
            let json = serde_json::to_string(&plot).unwrap();
            let decoded: Plot = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.soil_type, *soil);
            assert_eq!(decoded.status, *status);
        }
    }

    // ========================================================================
    // Crop struct serde roundtrip
    // ========================================================================

    #[test]
    fn crop_serde_roundtrip() {
        let crop = Crop {
            plot_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            name: "Tomato".to_string(),
            variety: "Cherokee Purple".to_string(),
            planted_at: 1700000000,
            expected_harvest: 1708000000,
            status: CropStatus::Growing,
        };
        let json = serde_json::to_string(&crop).unwrap();
        let decoded: Crop = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Tomato");
        assert_eq!(decoded.variety, "Cherokee Purple");
        assert_eq!(decoded.planted_at, 1700000000);
        assert_eq!(decoded.expected_harvest, 1708000000);
        assert_eq!(decoded.status, CropStatus::Growing);
    }

    #[test]
    fn crop_serde_all_statuses() {
        for status in [
            CropStatus::Planned,
            CropStatus::Planted,
            CropStatus::Growing,
            CropStatus::Ready,
            CropStatus::Harvested,
            CropStatus::Failed,
        ] {
            let crop = Crop {
                plot_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                name: "Basil".to_string(),
                variety: "Genovese".to_string(),
                planted_at: 1700000000,
                expected_harvest: 1705000000,
                status: status.clone(),
            };
            let json = serde_json::to_string(&crop).unwrap();
            let decoded: Crop = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ========================================================================
    // YieldRecord struct serde roundtrip
    // ========================================================================

    #[test]
    fn yield_record_serde_roundtrip_with_notes() {
        let yr = YieldRecord {
            crop_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            quantity_kg: 42.5,
            quality_grade: QualityGrade::Premium,
            harvested_at: 1708000000,
            notes: Some("Excellent season".to_string()),
        };
        let json = serde_json::to_string(&yr).unwrap();
        let decoded: YieldRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity_kg, 42.5);
        assert_eq!(decoded.quality_grade, QualityGrade::Premium);
        assert_eq!(decoded.harvested_at, 1708000000);
        assert_eq!(decoded.notes, Some("Excellent season".to_string()));
    }

    #[test]
    fn yield_record_serde_roundtrip_without_notes() {
        let yr = YieldRecord {
            crop_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            quantity_kg: 3.2,
            quality_grade: QualityGrade::Compost,
            harvested_at: 1709000000,
            notes: None,
        };
        let json = serde_json::to_string(&yr).unwrap();
        let decoded: YieldRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity_kg, 3.2);
        assert_eq!(decoded.quality_grade, QualityGrade::Compost);
        assert_eq!(decoded.notes, None);
    }

    #[test]
    fn yield_record_serde_all_grades() {
        for grade in [
            QualityGrade::Premium,
            QualityGrade::Standard,
            QualityGrade::Processing,
            QualityGrade::Compost,
        ] {
            let yr = YieldRecord {
                crop_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                quantity_kg: 10.0,
                quality_grade: grade.clone(),
                harvested_at: 1708000000,
                notes: None,
            };
            let json = serde_json::to_string(&yr).unwrap();
            let decoded: YieldRecord = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.quality_grade, grade);
        }
    }

    // ========================================================================
    // SeasonPlan struct serde roundtrip
    // ========================================================================

    #[test]
    fn season_plan_serde_roundtrip_with_rotation_notes() {
        let sp = SeasonPlan {
            plot_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            year: 2026,
            season: "Spring".to_string(),
            planned_crops: vec!["Tomato".to_string(), "Basil".to_string(), "Pepper".to_string()],
            rotation_notes: Some("Follow legumes with brassicas".to_string()),
        };
        let json = serde_json::to_string(&sp).unwrap();
        let decoded: SeasonPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.year, 2026);
        assert_eq!(decoded.season, "Spring");
        assert_eq!(decoded.planned_crops.len(), 3);
        assert_eq!(decoded.planned_crops[0], "Tomato");
        assert_eq!(decoded.rotation_notes, Some("Follow legumes with brassicas".to_string()));
    }

    #[test]
    fn season_plan_serde_roundtrip_without_rotation_notes() {
        let sp = SeasonPlan {
            plot_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            year: 2027,
            season: "Fall".to_string(),
            planned_crops: vec!["Garlic".to_string()],
            rotation_notes: None,
        };
        let json = serde_json::to_string(&sp).unwrap();
        let decoded: SeasonPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.year, 2027);
        assert_eq!(decoded.season, "Fall");
        assert_eq!(decoded.planned_crops, vec!["Garlic"]);
        assert_eq!(decoded.rotation_notes, None);
    }
}
