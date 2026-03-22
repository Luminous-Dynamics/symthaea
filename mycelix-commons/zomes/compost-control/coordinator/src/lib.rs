// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compost Control Coordinator Zome
//! Sensor-driven composting automation: batch management, threshold evaluation, action recommendations

use hdk::prelude::*;
use mycelix_bridge_common::requirement_for_basic;
use mycelix_zome_helpers::records_from_links;
use compost_control_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn require_consciousness(
    requirement: &mycelix_bridge_common::GovernanceRequirement,
    action_name: &str,
) -> ExternResult<mycelix_bridge_common::GovernanceEligibility> {
    mycelix_zome_helpers::require_consciousness("commons_bridge", requirement, action_name)
}

// ============================================================================
// COMPOST BATCHES
// ============================================================================

/// Create a new compost batch
#[hdk_extern]
pub fn create_compost_batch(batch: CompostBatch) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "create_compost_batch")?;

    // Validate inputs
    if batch.inputs.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Compost batch must have at least one input".into()
        )));
    }
    if !batch.carbon_nitrogen_ratio.is_finite() || batch.carbon_nitrogen_ratio <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "C:N ratio must be a positive finite number".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CompostBatch(batch.clone()))?;

    // Link to all batches anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_compost_batches".to_string())))?;
    create_link(
        anchor_hash("all_compost_batches")?,
        action_hash.clone(),
        LinkTypes::AllBatches,
        (),
    )?;

    // Link facility to batch
    create_link(
        batch.facility_hash.clone(),
        action_hash.clone(),
        LinkTypes::FacilityToBatches,
        (),
    )?;

    // Link to status anchor
    let status_anchor = format!("compost_status:{:?}", batch.status);
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    create_link(
        anchor_hash(&status_anchor)?,
        action_hash.clone(),
        LinkTypes::StatusToBatches,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "compost_batch_created",
        "source_zome": "compost_control",
        "payload": {
            "batch_id": batch.id,
            "method": format!("{:?}", batch.method),
            "cn_ratio": batch.carbon_nitrogen_ratio,
            "input_count": batch.inputs.len(),
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get a compost batch by hash
#[hdk_extern]
pub fn get_compost_batch(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all compost batches
#[hdk_extern]
pub fn get_all_batches(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_compost_batches")?, LinkTypes::AllBatches)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get compost batches for a facility
#[hdk_extern]
pub fn get_batches_by_facility(facility_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(facility_hash, LinkTypes::FacilityToBatches)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// SENSOR READINGS
// ============================================================================

/// Record a sensor reading from a compost batch
#[hdk_extern]
pub fn record_compost_reading(reading: CompostReading) -> ExternResult<Record> {
    let _eligibility =
        require_consciousness(&requirement_for_basic(), "record_compost_reading")?;

    // Validate ranges
    if !reading.temperature_c.is_finite()
        || reading.temperature_c < -20.0
        || reading.temperature_c > 90.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Temperature must be between -20 and 90 C".into()
        )));
    }
    if !reading.moisture_pct.is_finite()
        || reading.moisture_pct < 0.0
        || reading.moisture_pct > 100.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Moisture must be between 0 and 100 percent".into()
        )));
    }
    if !reading.oxygen_pct.is_finite()
        || reading.oxygen_pct < 0.0
        || reading.oxygen_pct > 100.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Oxygen must be between 0 and 100 percent".into()
        )));
    }
    if !reading.ph.is_finite() || reading.ph < 0.0 || reading.ph > 14.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "pH must be between 0 and 14".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CompostReading(reading.clone()))?;

    // Link batch to reading
    create_link(
        reading.batch_hash.clone(),
        action_hash.clone(),
        LinkTypes::BatchToReadings,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get all readings for a compost batch
#[hdk_extern]
pub fn get_batch_readings(batch_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(batch_hash, LinkTypes::BatchToReadings)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CONTROL LOOP: evaluate batch status and recommend actions
// ============================================================================

/// Result of evaluating a compost batch's sensor readings
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchEvaluation {
    /// Current batch status assessment
    pub current_phase: String,
    /// Recommended actions based on sensor data
    pub recommended_actions: Vec<RecommendedAction>,
    /// Whether a phase transition is recommended
    pub phase_transition: Option<String>,
    /// Summary of conditions
    pub temperature_status: String,
    pub moisture_status: String,
    pub oxygen_status: String,
    pub ph_status: String,
}

/// A single recommended action
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecommendedAction {
    pub action_type: CompostActionType,
    pub reason: String,
    pub urgency: String, // "low", "medium", "high"
}

/// Evaluate a compost batch and recommend actions based on latest sensor data
#[hdk_extern]
pub fn evaluate_batch_status(batch_hash: ActionHash) -> ExternResult<BatchEvaluation> {
    // Get the batch
    let batch_record = get(batch_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let batch: CompostBatch = batch_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode batch: {}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch entry not found".into())))?;

    // Get latest readings
    let reading_links = get_links(
        LinkQuery::try_new(batch_hash, LinkTypes::BatchToReadings)?,
        GetStrategy::default(),
    )?;
    let reading_records = records_from_links(reading_links)?;

    // Find most recent reading
    let latest_reading: Option<CompostReading> = reading_records
        .iter()
        .filter_map(|r| r.entry().to_app_option::<CompostReading>().ok().flatten())
        .max_by_key(|r| r.timestamp_us);

    let Some(reading) = latest_reading else {
        return Ok(BatchEvaluation {
            current_phase: format!("{:?}", batch.status),
            recommended_actions: vec![],
            phase_transition: None,
            temperature_status: "No readings".to_string(),
            moisture_status: "No readings".to_string(),
            oxygen_status: "No readings".to_string(),
            ph_status: "No readings".to_string(),
        });
    };

    let mut actions: Vec<RecommendedAction> = Vec::new();

    // Evaluate temperature
    let temperature_status = evaluate_temperature(reading.temperature_c, &batch.status);

    // Evaluate moisture
    let moisture_status = if reading.moisture_pct < MOISTURE_MIN_PCT {
        actions.push(RecommendedAction {
            action_type: CompostActionType::AddWater,
            reason: format!(
                "Moisture {:.1}% below minimum {:.0}%",
                reading.moisture_pct, MOISTURE_MIN_PCT
            ),
            urgency: if reading.moisture_pct < 30.0 {
                "high".to_string()
            } else {
                "medium".to_string()
            },
        });
        format!("Low ({:.1}%)", reading.moisture_pct)
    } else if reading.moisture_pct > MOISTURE_MAX_PCT {
        actions.push(RecommendedAction {
            action_type: CompostActionType::AddBulking,
            reason: format!(
                "Moisture {:.1}% above maximum {:.0}% — add carbon-rich bulking agent",
                reading.moisture_pct, MOISTURE_MAX_PCT
            ),
            urgency: "medium".to_string(),
        });
        format!("High ({:.1}%)", reading.moisture_pct)
    } else {
        format!("OK ({:.1}%)", reading.moisture_pct)
    };

    // Evaluate oxygen
    let oxygen_status = if reading.oxygen_pct < OXYGEN_MIN_PCT {
        actions.push(RecommendedAction {
            action_type: CompostActionType::Turn,
            reason: format!(
                "Oxygen {:.1}% below minimum {:.0}% — pile is going anaerobic",
                reading.oxygen_pct, OXYGEN_MIN_PCT
            ),
            urgency: "high".to_string(),
        });
        format!("Critical ({:.1}%)", reading.oxygen_pct)
    } else if reading.oxygen_pct < 10.0 {
        actions.push(RecommendedAction {
            action_type: CompostActionType::AdjustAeration,
            reason: format!(
                "Oxygen {:.1}% is low — increase aeration",
                reading.oxygen_pct
            ),
            urgency: "low".to_string(),
        });
        format!("Low ({:.1}%)", reading.oxygen_pct)
    } else {
        format!("OK ({:.1}%)", reading.oxygen_pct)
    };

    // Evaluate pH
    let ph_status = if reading.ph < PH_MIN || reading.ph > PH_MAX {
        format!("Out of range ({:.1})", reading.ph)
    } else {
        format!("OK ({:.1})", reading.ph)
    };

    // Determine phase transition
    let phase_transition = evaluate_phase_transition(&batch.status, &reading);

    // If ready to harvest
    if matches!(batch.status, CompostBatchStatus::Curing)
        && reading.temperature_c < CURING_COMPLETE_C
    {
        actions.push(RecommendedAction {
            action_type: CompostActionType::HarvestScreen,
            reason: "Curing complete — temperature stable below ambient".to_string(),
            urgency: "low".to_string(),
        });
    }

    // Emit signal if actions recommended
    if !actions.is_empty() {
        let _ = emit_signal(&serde_json::json!({
            "event_type": "compost_actions_recommended",
            "source_zome": "compost_control",
            "payload": {
                "action_count": actions.len(),
                "temperature_c": reading.temperature_c,
                "moisture_pct": reading.moisture_pct,
                "oxygen_pct": reading.oxygen_pct,
            }
        }));
    }

    Ok(BatchEvaluation {
        current_phase: format!("{:?}", batch.status),
        recommended_actions: actions,
        phase_transition,
        temperature_status,
        moisture_status,
        oxygen_status,
        ph_status,
    })
}

/// Evaluate temperature status relative to current composting phase
fn evaluate_temperature(temp_c: f64, status: &CompostBatchStatus) -> String {
    match status {
        CompostBatchStatus::Mixing => {
            format!("Mixing phase ({:.1}C)", temp_c)
        }
        CompostBatchStatus::ActiveComposting => {
            if temp_c < THERMOPHILIC_MIN_C {
                format!("Below thermophilic range ({:.1}C < {:.0}C)", temp_c, THERMOPHILIC_MIN_C)
            } else if temp_c > THERMOPHILIC_MAX_C {
                format!("Above thermophilic range ({:.1}C > {:.0}C) — risk of microbe die-off", temp_c, THERMOPHILIC_MAX_C)
            } else {
                format!("Thermophilic OK ({:.1}C)", temp_c)
            }
        }
        CompostBatchStatus::Curing => {
            if temp_c > MESOPHILIC_MAX_C {
                format!("Still too hot for curing ({:.1}C > {:.0}C)", temp_c, MESOPHILIC_MAX_C)
            } else if temp_c < CURING_COMPLETE_C {
                format!("Curing near complete ({:.1}C)", temp_c)
            } else {
                format!("Curing OK ({:.1}C)", temp_c)
            }
        }
        _ => format!("{:.1}C", temp_c),
    }
}

/// Determine if a phase transition is warranted based on sensor data
fn evaluate_phase_transition(
    current: &CompostBatchStatus,
    reading: &CompostReading,
) -> Option<String> {
    match current {
        CompostBatchStatus::Mixing => {
            if reading.temperature_c >= THERMOPHILIC_MIN_C {
                Some("Transition to ActiveComposting: temperature reached thermophilic range".to_string())
            } else {
                None
            }
        }
        CompostBatchStatus::ActiveComposting => {
            // After sustained thermophilic phase, if temp drops below mesophilic max → curing
            if reading.temperature_c < MESOPHILIC_MAX_C {
                Some("Transition to Curing: temperature dropped below thermophilic range".to_string())
            } else {
                None
            }
        }
        CompostBatchStatus::Curing => {
            if reading.temperature_c < CURING_COMPLETE_C {
                Some("Transition to Screening: curing complete, ambient temperature reached".to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

// ============================================================================
// ACTIONS
// ============================================================================

/// Record a compost action (recommended or manually created)
#[hdk_extern]
pub fn record_compost_action(action: CompostAction) -> ExternResult<Record> {
    let _eligibility =
        require_consciousness(&requirement_for_basic(), "record_compost_action")?;

    let action_hash = create_entry(&EntryTypes::CompostAction(action.clone()))?;

    // Link batch to action
    create_link(
        action.batch_hash.clone(),
        action_hash.clone(),
        LinkTypes::BatchToActions,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "compost_action_recorded",
        "source_zome": "compost_control",
        "payload": {
            "action_type": format!("{:?}", action.action_type),
            "recommended_by": format!("{:?}", action.recommended_by),
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get all actions for a compost batch
#[hdk_extern]
pub fn get_batch_actions(batch_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(batch_hash, LinkTypes::BatchToActions)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// NUTRIENT ESTIMATION
// ============================================================================

/// Estimated macronutrient profile of a compost batch (from food-production literature values)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NutrientEstimate {
    /// Total batch weight (kg)
    pub total_kg: f64,
    /// Estimated nitrogen percentage (weighted average of inputs)
    pub nitrogen_pct: f64,
    /// Estimated phosphorus percentage
    pub phosphorus_pct: f64,
    /// Estimated potassium percentage
    pub potassium_pct: f64,
}

/// Estimate the nutrient profile of a compost batch based on its inputs
#[hdk_extern]
pub fn get_batch_nutrient_estimate(batch_hash: ActionHash) -> ExternResult<NutrientEstimate> {
    let batch_record = get(batch_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let batch: CompostBatch = batch_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode batch: {}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch entry not found".into())))?;

    let mut total_kg: f64 = 0.0;
    let mut total_n: f64 = 0.0;
    let mut total_p: f64 = 0.0;
    let mut total_k: f64 = 0.0;

    for input in &batch.inputs {
        let (n, p, k) = estimate_npk(&input.resource_type);
        let kg = input.quantity_kg;
        total_kg += kg;
        total_n += n * kg;
        total_p += p * kg;
        total_k += k * kg;
    }

    let (nitrogen_pct, phosphorus_pct, potassium_pct) = if total_kg > 0.0 {
        (total_n / total_kg, total_p / total_kg, total_k / total_kg)
    } else {
        (0.0, 0.0, 0.0)
    };

    Ok(NutrientEstimate {
        total_kg,
        nitrogen_pct,
        phosphorus_pct,
        potassium_pct,
    })
}

/// NPK estimates from composting literature (matches food-production NutrientProfile::estimate)
fn estimate_npk(resource_type: &str) -> (f64, f64, f64) {
    match resource_type {
        "KitchenWaste" => (1.5, 0.5, 1.0),
        "GreenWaste" => (2.0, 0.3, 1.5),
        "Biochar" => (0.5, 0.2, 1.0),
        "Vermicompost" => (2.5, 1.5, 1.5),
        "Digestate" => (3.0, 0.8, 2.0),
        "Manure" => (2.0, 1.0, 2.0),
        "Mulch" => (0.5, 0.1, 0.3),
        "CoverCrop" => (3.0, 0.4, 2.5),
        "Inoculant" => (0.1, 0.1, 0.1),
        _ => (1.0, 0.3, 0.5), // Generic organic waste fallback
    }
}

// ============================================================================
// CARBON ATTRIBUTION (Phase 4)
// ============================================================================

/// EPA estimate: composting 1 tonne of food waste avoids ~0.06 tonnes CO2e
/// (methane avoidance from landfill diversion)
const COMPOST_CO2E_AVOIDED_PER_TONNE: f64 = 0.06;

/// Result of carbon attribution for a completed batch
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CarbonAttribution {
    /// Total waste diverted from landfill (kg)
    pub waste_diverted_kg: f64,
    /// Estimated avoided CO2 equivalent (tonnes)
    pub co2e_avoided_tonnes: f64,
    /// Method used for calculation
    pub methodology: String,
}

/// Calculate carbon attribution for a completed compost batch.
///
/// This estimates the greenhouse gas emissions avoided by composting
/// rather than landfilling the organic waste. Uses EPA's factor of
/// 0.06 tCO2e avoided per tonne of food waste composted.
#[hdk_extern]
pub fn calculate_carbon_attribution(batch_hash: ActionHash) -> ExternResult<CarbonAttribution> {
    let batch_record = get(batch_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let batch: CompostBatch = batch_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode batch: {}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch entry not found".into())))?;

    // Sum all input weights
    let waste_diverted_kg: f64 = batch.inputs.iter().map(|i| i.quantity_kg).sum();
    let waste_diverted_tonnes = waste_diverted_kg / 1000.0;

    // Calculate avoided emissions (EPA: 0.06 tCO2e per tonne food waste composted)
    let co2e_avoided_tonnes = waste_diverted_tonnes * COMPOST_CO2E_AVOIDED_PER_TONNE;

    // Emit signal for climate integration
    let _ = emit_signal(&serde_json::json!({
        "event_type": "carbon_attribution_calculated",
        "source_zome": "compost_control",
        "payload": {
            "batch_id": batch.id,
            "waste_diverted_kg": waste_diverted_kg,
            "co2e_avoided_tonnes": co2e_avoided_tonnes,
            "methodology": "EPA_WARM_model",
        }
    }));

    Ok(CarbonAttribution {
        waste_diverted_kg,
        co2e_avoided_tonnes,
        methodology: "EPA WARM model: 0.06 tCO2e/tonne food waste composted".to_string(),
    })
}

/// Calculate the diversion rate for a facility's compost batches
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FacilityDiversionSummary {
    /// Total batches processed
    pub total_batches: u32,
    /// Completed batches
    pub completed_batches: u32,
    /// Total waste composted (kg)
    pub total_composted_kg: f64,
    /// Total CO2e avoided (tonnes)
    pub total_co2e_avoided_tonnes: f64,
}

/// Get a summary of composting activity and carbon savings for a facility
#[hdk_extern]
pub fn get_facility_diversion_summary(
    facility_hash: ActionHash,
) -> ExternResult<FacilityDiversionSummary> {
    let links = get_links(
        LinkQuery::try_new(facility_hash, LinkTypes::FacilityToBatches)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    let mut total_batches: u32 = 0;
    let mut completed_batches: u32 = 0;
    let mut total_composted_kg: f64 = 0.0;

    for record in records {
        let batch: CompostBatch = match record.entry().to_app_option() {
            Ok(Some(b)) => b,
            _ => continue,
        };
        total_batches += 1;
        if batch.status == CompostBatchStatus::Complete {
            completed_batches += 1;
            let batch_kg: f64 = batch.inputs.iter().map(|i| i.quantity_kg).sum();
            total_composted_kg += batch_kg;
        }
    }

    let total_co2e_avoided_tonnes =
        (total_composted_kg / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;

    Ok(FacilityDiversionSummary {
        total_batches,
        completed_batches,
        total_composted_kg,
        total_co2e_avoided_tonnes,
    })
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_temperature_thermophilic_ok() {
        let status = evaluate_temperature(60.0, &CompostBatchStatus::ActiveComposting);
        assert!(status.contains("OK"), "60C should be OK during active composting: {}", status);
    }

    #[test]
    fn test_evaluate_temperature_too_cold_for_thermophilic() {
        let status = evaluate_temperature(45.0, &CompostBatchStatus::ActiveComposting);
        assert!(status.contains("Below"), "45C should be below thermophilic: {}", status);
    }

    #[test]
    fn test_evaluate_temperature_too_hot() {
        let status = evaluate_temperature(70.0, &CompostBatchStatus::ActiveComposting);
        assert!(status.contains("Above"), "70C should be above thermophilic: {}", status);
    }

    #[test]
    fn test_evaluate_temperature_curing_complete() {
        let status = evaluate_temperature(22.0, &CompostBatchStatus::Curing);
        assert!(status.contains("near complete"), "22C during curing should be near complete: {}", status);
    }

    #[test]
    fn test_phase_transition_mixing_to_active() {
        let reading = CompostReading {
            batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            sensor_id: "S1".to_string(),
            temperature_c: 56.0,
            moisture_pct: 55.0,
            oxygen_pct: 15.0,
            ph: 7.0,
            timestamp_us: 100,
        };
        let result = evaluate_phase_transition(&CompostBatchStatus::Mixing, &reading);
        assert!(result.is_some(), "56C should trigger transition from Mixing");
        assert!(result.unwrap().contains("ActiveComposting"));
    }

    #[test]
    fn test_phase_transition_active_to_curing() {
        let reading = CompostReading {
            batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            sensor_id: "S1".to_string(),
            temperature_c: 35.0,
            moisture_pct: 50.0,
            oxygen_pct: 15.0,
            ph: 7.0,
            timestamp_us: 100,
        };
        let result = evaluate_phase_transition(&CompostBatchStatus::ActiveComposting, &reading);
        assert!(result.is_some(), "35C should trigger transition from ActiveComposting");
        assert!(result.unwrap().contains("Curing"));
    }

    #[test]
    fn test_phase_transition_curing_to_screening() {
        let reading = CompostReading {
            batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            sensor_id: "S1".to_string(),
            temperature_c: 22.0,
            moisture_pct: 40.0,
            oxygen_pct: 18.0,
            ph: 7.0,
            timestamp_us: 100,
        };
        let result = evaluate_phase_transition(&CompostBatchStatus::Curing, &reading);
        assert!(result.is_some(), "22C should trigger transition from Curing");
        assert!(result.unwrap().contains("Screening"));
    }

    #[test]
    fn test_no_transition_when_temperature_in_range() {
        let reading = CompostReading {
            batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            sensor_id: "S1".to_string(),
            temperature_c: 60.0,
            moisture_pct: 55.0,
            oxygen_pct: 15.0,
            ph: 7.0,
            timestamp_us: 100,
        };
        let result = evaluate_phase_transition(&CompostBatchStatus::ActiveComposting, &reading);
        assert!(result.is_none(), "60C should not trigger transition during ActiveComposting");
    }

    #[test]
    fn test_npk_kitchen_waste() {
        let (n, p, k) = estimate_npk("KitchenWaste");
        assert!((n - 1.5).abs() < 0.001);
        assert!((p - 0.5).abs() < 0.001);
        assert!((k - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_npk_vermicompost() {
        let (n, p, k) = estimate_npk("Vermicompost");
        assert!((n - 2.5).abs() < 0.001);
        assert!((p - 1.5).abs() < 0.001);
        assert!((k - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_npk_unknown_fallback() {
        let (n, p, k) = estimate_npk("UnknownMaterial");
        assert!(n > 0.0 && p > 0.0 && k > 0.0, "Unknown material should use fallback");
    }

    #[test]
    fn test_npk_weighted_average() {
        // 50kg kitchen waste + 50kg green waste
        let total_kg: f64 = 100.0;
        let (n1, p1, k1) = estimate_npk("KitchenWaste");
        let (n2, p2, k2) = estimate_npk("GreenWaste");
        let avg_n = (n1 * 50.0 + n2 * 50.0) / total_kg;
        let avg_p = (p1 * 50.0 + p2 * 50.0) / total_kg;
        let avg_k = (k1 * 50.0 + k2 * 50.0) / total_kg;
        // Kitchen: N=1.5, Green: N=2.0 → avg=1.75
        assert!((avg_n - 1.75).abs() < 0.001);
        assert!((avg_p - 0.4).abs() < 0.001);
        assert!((avg_k - 1.25).abs() < 0.001);
    }

    #[test]
    fn test_all_resource_types_have_npk() {
        let types = [
            "KitchenWaste", "GreenWaste", "Biochar", "Vermicompost",
            "Digestate", "Manure", "Mulch", "CoverCrop", "Inoculant",
        ];
        for rt in types {
            let (n, p, k) = estimate_npk(rt);
            assert!(n >= 0.0, "{} nitrogen should be non-negative", rt);
            assert!(p >= 0.0, "{} phosphorus should be non-negative", rt);
            assert!(k >= 0.0, "{} potassium should be non-negative", rt);
        }
    }

    // -- Carbon attribution tests --

    #[test]
    fn test_carbon_factor_positive() {
        assert!(COMPOST_CO2E_AVOIDED_PER_TONNE > 0.0);
    }

    #[test]
    fn test_carbon_calculation_1_tonne() {
        let waste_kg: f64 = 1000.0;
        let waste_tonnes = waste_kg / 1000.0;
        let co2e = waste_tonnes * COMPOST_CO2E_AVOIDED_PER_TONNE;
        assert!((co2e - 0.06).abs() < 0.001, "1 tonne should avoid 0.06 tCO2e, got {}", co2e);
    }

    #[test]
    fn test_carbon_calculation_zero_waste() {
        let waste_kg: f64 = 0.0;
        let waste_tonnes = waste_kg / 1000.0;
        let co2e = waste_tonnes * COMPOST_CO2E_AVOIDED_PER_TONNE;
        assert_eq!(co2e, 0.0, "Zero waste should avoid zero emissions");
    }

    #[test]
    fn test_carbon_scales_linearly() {
        let co2e_1t: f64 = (1000.0 / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;
        let co2e_10t: f64 = (10000.0 / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;
        assert!((co2e_10t / co2e_1t - 10.0).abs() < 0.001, "Carbon should scale linearly");
    }

    #[test]
    fn test_carbon_non_negative() {
        // For any non-negative waste, CO2e avoided should be non-negative
        for kg in [0.0, 1.0, 100.0, 10000.0, 1_000_000.0_f64] {
            let co2e = (kg / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;
            assert!(co2e >= 0.0, "CO2e should be non-negative for {}kg", kg);
        }
    }
}
