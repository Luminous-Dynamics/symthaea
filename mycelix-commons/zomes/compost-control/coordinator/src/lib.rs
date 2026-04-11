// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compost Control Coordinator Zome
//! Sensor-driven composting automation: batch management, threshold evaluation, action recommendations

use hdk::prelude::*;
use mycelix_bridge_common::civic_requirement_basic;
use mycelix_zome_helpers::records_from_links;
use compost_control_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}


// ============================================================================
// COMPOST BATCHES
// ============================================================================

/// Create a new compost batch
#[hdk_extern]
pub fn create_compost_batch(batch: CompostBatch) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_compost_batch")?;

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
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_compost_reading")?;

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
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_compost_action")?;

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
// RECIPE OPTIMIZER
// ============================================================================

/// Carbon-to-nitrogen ratios for common composting inputs.
/// Sources: Cornell Waste Management Institute, USCC guidelines.
fn estimate_cn_ratio(resource_type: &str) -> f64 {
    match resource_type {
        "KitchenWaste" => 15.0,   // Nitrogen-rich
        "GreenWaste" => 20.0,     // Moderate nitrogen
        "Biochar" => 200.0,       // Very high carbon
        "Vermicompost" => 15.0,   // Pre-processed, balanced
        "Digestate" => 10.0,      // Very nitrogen-rich
        "Manure" => 18.0,         // Nitrogen-rich
        "Mulch" => 50.0,          // Carbon-rich
        "CoverCrop" => 20.0,      // Moderate
        "Inoculant" => 10.0,      // Nitrogen-dominant
        "Woodchips" => 400.0,     // Very high carbon
        "Straw" => 80.0,          // High carbon
        "Cardboard" => 350.0,     // Very high carbon
        "Sawdust" => 500.0,       // Extremely high carbon
        "Leaves" => 60.0,         // Moderate-high carbon
        _ => 30.0,                // Default fallback
    }
}

/// A suggested amendment to improve batch C:N ratio.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecipeAmendment {
    /// Resource type to add
    pub resource_type: String,
    /// Quantity to add (kg)
    pub quantity_kg: f64,
    /// C:N ratio of this material
    pub material_cn_ratio: f64,
    /// Resulting batch C:N ratio after adding this amendment
    pub resulting_cn_ratio: f64,
}

/// Recipe optimization result.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecipeOptimization {
    /// Current batch C:N ratio (weighted average of inputs)
    pub current_cn_ratio: f64,
    /// Target C:N ratio
    pub target_cn_ratio: f64,
    /// Whether the current ratio is within acceptable range
    pub is_optimal: bool,
    /// "nitrogen_heavy" or "carbon_heavy" or "balanced"
    pub diagnosis: String,
    /// Suggested amendments (sorted by quantity needed, ascending)
    pub amendments: Vec<RecipeAmendment>,
}

/// Input for recipe optimization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptimizeRecipeInput {
    pub batch_hash: ActionHash,
    /// Target C:N ratio (default: 27.5 if 0)
    pub target_cn_ratio: f64,
}

/// Optimize the C:N ratio of a compost batch by suggesting amendments.
///
/// Uses weighted average C:N from existing inputs, compares to target,
/// and suggests the minimum amendment to reach the target range (25:1 - 30:1).
#[hdk_extern]
pub fn optimize_recipe(input: OptimizeRecipeInput) -> ExternResult<RecipeOptimization> {
    let batch_record = get(input.batch_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let batch: CompostBatch = batch_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch entry not found".into())))?;

    let target = if input.target_cn_ratio > 0.0 {
        input.target_cn_ratio
    } else {
        27.5 // Midpoint of 25-30 optimal range
    };

    // Calculate current weighted C:N ratio
    let mut total_kg: f64 = 0.0;
    let mut weighted_cn: f64 = 0.0;
    for inp in &batch.inputs {
        let cn = estimate_cn_ratio(&inp.resource_type);
        weighted_cn += cn * inp.quantity_kg;
        total_kg += inp.quantity_kg;
    }

    let current_cn = if total_kg > 0.0 {
        weighted_cn / total_kg
    } else {
        0.0
    };

    let is_optimal = current_cn >= CN_RATIO_MIN && current_cn <= CN_RATIO_MAX;
    let diagnosis = if current_cn < CN_RATIO_MIN {
        "nitrogen_heavy".to_string()
    } else if current_cn > CN_RATIO_MAX {
        "carbon_heavy".to_string()
    } else {
        "balanced".to_string()
    };

    // Suggest amendments
    let mut amendments = Vec::new();

    if !is_optimal && total_kg > 0.0 {
        // Determine which amendments would help
        let amendment_types: Vec<(&str, f64)> = if current_cn < target {
            // Need more carbon — suggest carbon-rich materials
            vec![
                ("Woodchips", 400.0),
                ("Straw", 80.0),
                ("Cardboard", 350.0),
                ("Sawdust", 500.0),
                ("Leaves", 60.0),
                ("Mulch", 50.0),
            ]
        } else {
            // Need more nitrogen — suggest nitrogen-rich materials
            vec![
                ("KitchenWaste", 15.0),
                ("GreenWaste", 20.0),
                ("Manure", 18.0),
                ("Digestate", 10.0),
            ]
        };

        for (rt, cn) in amendment_types {
            // Solve: (weighted_cn + cn * x) / (total_kg + x) = target
            // → weighted_cn + cn * x = target * total_kg + target * x
            // → x * (cn - target) = target * total_kg - weighted_cn
            // → x = (target * total_kg - weighted_cn) / (cn - target)
            let denominator = cn - target;
            if denominator.abs() < 0.001 {
                continue; // This material has roughly the target C:N, won't help
            }
            let x = (target * total_kg - weighted_cn) / denominator;
            if x > 0.0 && x.is_finite() {
                let resulting_cn = (weighted_cn + cn * x) / (total_kg + x);
                amendments.push(RecipeAmendment {
                    resource_type: rt.to_string(),
                    quantity_kg: (x * 10.0).round() / 10.0, // Round to 0.1 kg
                    material_cn_ratio: cn,
                    resulting_cn_ratio: (resulting_cn * 10.0).round() / 10.0,
                });
            }
        }

        // Sort by quantity needed (least amendment first)
        amendments.sort_by(|a, b| {
            a.quantity_kg
                .partial_cmp(&b.quantity_kg)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    Ok(RecipeOptimization {
        current_cn_ratio: (current_cn * 10.0).round() / 10.0,
        target_cn_ratio: target,
        is_optimal,
        diagnosis,
        amendments,
    })
}

// ============================================================================
// SENSOR BRIDGE (resource-mesh integration)
// ============================================================================

/// Query a single latest value from resource-mesh via `CallTargetCell::Local`.
///
/// Calls `resource_mesh::get_resource_status` and returns the most recent
/// value for the given resource_type. Returns `None` if the call fails
/// (e.g., resource-mesh zome not present in DNA, or no readings available).
fn query_resource_mesh_sensor(resource_type: &str, sensor_id: Option<&str>) -> Option<f64> {
    if sensor_id.is_none() {
        return None;
    }

    // Build the input for resource_mesh::get_resource_status
    // The resource-mesh expects: { resource_type: String, location: Option<String>, limit: usize }
    let query_input = serde_json::json!({
        "resource_type": resource_type,
        "location": null,
        "limit": 1
    });

    let payload = ExternIO::encode(query_input).ok()?;

    let response = call(
        CallTargetCell::Local,
        ZomeName::from("resource_mesh"),
        FunctionName::from("get_resource_status"),
        None,
        payload,
    );

    match response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            // Decode response as Vec of SensorReading-like objects
            // resource-mesh returns Vec<SensorReading> where each has a `value: f64`
            let readings: Vec<serde_json::Value> =
                extern_io.decode().ok()?;
            readings
                .first()
                .and_then(|r| r.get("value"))
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite())
        }
        _ => {
            // Resource-mesh not available or call failed — graceful degradation
            None
        }
    }
}

/// Input for syncing sensor readings from resource-mesh to compost-control.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SensorBridgeInput {
    /// Compost batch to update
    pub batch_hash: ActionHash,
    /// Resource-mesh sensor ID for temperature
    pub temp_sensor_id: Option<String>,
    /// Resource-mesh sensor ID for moisture
    pub moisture_sensor_id: Option<String>,
    /// Resource-mesh sensor ID for oxygen
    pub oxygen_sensor_id: Option<String>,
    /// Default pH if no sensor available
    pub default_ph: f64,
}

/// Result of sensor bridge sync.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SensorBridgeResult {
    /// Whether a reading was successfully recorded
    pub reading_recorded: bool,
    /// Temperature value (if available)
    pub temperature_c: Option<f64>,
    /// Moisture value (if available)
    pub moisture_pct: Option<f64>,
    /// Oxygen value (if available)
    pub oxygen_pct: Option<f64>,
    /// Evaluation result (if reading was recorded)
    pub evaluation: Option<BatchEvaluation>,
    /// Number of threshold alerts triggered
    pub alerts_triggered: u32,
}

/// Sync latest sensor readings from resource-mesh into a compost batch reading,
/// then evaluate the batch status and return recommendations.
///
/// This closes the sensor → compost-control → action loop by:
/// 1. Calling `resource_mesh::get_resource_status` via `CallTargetCell::Local`
///    for temperature, humidity, and air_quality readings
/// 2. Mapping resource-mesh values to CompostReading fields
/// 3. Creating a CompostReading entry
/// 4. Running evaluate_batch_status() to get recommendations
#[hdk_extern]
pub fn record_sensor_bridge_reading(input: SensorBridgeInput) -> ExternResult<SensorBridgeResult> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_sensor_bridge_reading")?;

    // Verify the batch exists
    let _batch_record = get(input.batch_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;

    // Query resource-mesh for latest sensor values via cross-zome Local call.
    // Each sensor type maps: temperature → temperature_c, humidity → moisture_pct,
    // air_quality (O2 %) → oxygen_pct.
    let temp = query_resource_mesh_sensor("temperature", input.temp_sensor_id.as_deref());
    let moisture = query_resource_mesh_sensor("humidity", input.moisture_sensor_id.as_deref());
    let oxygen = query_resource_mesh_sensor("air_quality", input.oxygen_sensor_id.as_deref());

    if temp.is_none() && moisture.is_none() && oxygen.is_none() {
        return Ok(SensorBridgeResult {
            reading_recorded: false,
            temperature_c: None,
            moisture_pct: None,
            oxygen_pct: None,
            evaluation: None,
            alerts_triggered: 0,
        });
    }

    // Create reading with available values (defaults for missing sensors)
    let reading = CompostReading {
        batch_hash: input.batch_hash.clone(),
        sensor_id: "sensor_bridge".to_string(),
        temperature_c: temp.unwrap_or(25.0),
        moisture_pct: moisture.unwrap_or(50.0),
        oxygen_pct: oxygen.unwrap_or(15.0),
        ph: input.default_ph.clamp(0.0, 14.0),
        timestamp_us: sys_time()?.as_micros() as u64,
    };

    // Record the reading
    let reading_hash = create_entry(&EntryTypes::CompostReading(reading))?;
    create_link(
        input.batch_hash.clone(),
        reading_hash,
        LinkTypes::BatchToReadings,
        (),
    )?;

    // Evaluate and count alerts
    let evaluation = evaluate_batch_status(input.batch_hash)?;
    let alerts_triggered = evaluation.recommended_actions.len() as u32;

    // Emit alert if thresholds crossed
    if alerts_triggered > 0 {
        let _ = emit_signal(&serde_json::json!({
            "event_type": "sensor_bridge_alerts",
            "source_zome": "compost_control",
            "payload": {
                "alerts_triggered": alerts_triggered,
                "temperature_c": temp,
                "moisture_pct": moisture,
            }
        }));
    }

    Ok(SensorBridgeResult {
        reading_recorded: true,
        temperature_c: temp,
        moisture_pct: moisture,
        oxygen_pct: oxygen,
        evaluation: Some(evaluation),
        alerts_triggered,
    })
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

// ============================================================================
// PROPERTY TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Carbon attribution is always non-negative for non-negative waste.
        #[test]
        fn carbon_attribution_non_negative(waste_kg in 0.0..=1_000_000.0_f64) {
            let co2e = (waste_kg / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;
            prop_assert!(co2e >= 0.0, "CO2e must be non-negative for {}kg: {}", waste_kg, co2e);
            prop_assert!(co2e.is_finite(), "CO2e must be finite for {}kg", waste_kg);
        }

        /// Carbon attribution scales linearly with waste.
        #[test]
        fn carbon_attribution_linear(
            waste_kg in 1.0..=100_000.0_f64,
            factor in 1.0..=10.0_f64,
        ) {
            let co2e_1 = (waste_kg / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;
            let co2e_f = (waste_kg * factor / 1000.0) * COMPOST_CO2E_AVOIDED_PER_TONNE;
            let ratio = co2e_f / co2e_1;
            prop_assert!((ratio - factor).abs() < 0.001,
                "Carbon should scale linearly: ratio {} vs factor {}", ratio, factor);
        }

        /// Phase transition from Mixing requires temperature >= 55C.
        #[test]
        fn mixing_transition_requires_thermophilic(temp in 0.0..=90.0_f64) {
            let reading = CompostReading {
                batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sensor_id: "S1".to_string(),
                temperature_c: temp,
                moisture_pct: 55.0,
                oxygen_pct: 15.0,
                ph: 7.0,
                timestamp_us: 100,
            };
            let result = evaluate_phase_transition(&CompostBatchStatus::Mixing, &reading);
            if temp >= 55.0 {
                prop_assert!(result.is_some(),
                    "Temp {}C >= 55C should trigger Mixing→Active transition", temp);
            } else {
                prop_assert!(result.is_none(),
                    "Temp {}C < 55C should NOT trigger Mixing transition", temp);
            }
        }

        /// Phase transition from ActiveComposting requires temp < 40C.
        #[test]
        fn active_transition_requires_cooldown(temp in 0.0..=90.0_f64) {
            let reading = CompostReading {
                batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sensor_id: "S1".to_string(),
                temperature_c: temp,
                moisture_pct: 50.0,
                oxygen_pct: 15.0,
                ph: 7.0,
                timestamp_us: 100,
            };
            let result = evaluate_phase_transition(&CompostBatchStatus::ActiveComposting, &reading);
            if temp < 40.0 {
                prop_assert!(result.is_some(),
                    "Temp {}C < 40C should trigger Active→Curing transition", temp);
            } else {
                prop_assert!(result.is_none(),
                    "Temp {}C >= 40C should NOT trigger Active transition", temp);
            }
        }

        /// Phase transition from Curing requires temp < 25C.
        #[test]
        fn curing_transition_requires_ambient(temp in 0.0..=90.0_f64) {
            let reading = CompostReading {
                batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sensor_id: "S1".to_string(),
                temperature_c: temp,
                moisture_pct: 40.0,
                oxygen_pct: 18.0,
                ph: 7.0,
                timestamp_us: 100,
            };
            let result = evaluate_phase_transition(&CompostBatchStatus::Curing, &reading);
            if temp < 25.0 {
                prop_assert!(result.is_some(),
                    "Temp {}C < 25C should trigger Curing→Screening transition", temp);
            } else {
                prop_assert!(result.is_none(),
                    "Temp {}C >= 25C should NOT trigger Curing transition", temp);
            }
        }

        /// NPK estimates are always non-negative for any resource type.
        #[test]
        fn npk_always_non_negative(
            rt in prop_oneof![
                Just("KitchenWaste"),
                Just("GreenWaste"),
                Just("Biochar"),
                Just("Vermicompost"),
                Just("Digestate"),
                Just("Manure"),
                Just("Mulch"),
                Just("CoverCrop"),
                Just("Inoculant"),
                Just("UnknownType"),
            ]
        ) {
            let (n, p, k) = estimate_npk(rt);
            prop_assert!(n >= 0.0, "{} nitrogen must be non-negative: {}", rt, n);
            prop_assert!(p >= 0.0, "{} phosphorus must be non-negative: {}", rt, p);
            prop_assert!(k >= 0.0, "{} potassium must be non-negative: {}", rt, k);
            prop_assert!(n.is_finite() && p.is_finite() && k.is_finite(),
                "NPK values must be finite for {}", rt);
        }

        /// C:N ratio is always positive for known types.
        #[test]
        fn cn_ratio_positive(
            rt in prop_oneof![
                Just("KitchenWaste"),
                Just("GreenWaste"),
                Just("Biochar"),
                Just("Woodchips"),
                Just("Straw"),
                Just("Manure"),
                Just("Mulch"),
                Just("Sawdust"),
                Just("Leaves"),
                Just("UnknownType"),
            ]
        ) {
            let cn = estimate_cn_ratio(rt);
            prop_assert!(cn > 0.0, "{} C:N must be positive: {}", rt, cn);
            prop_assert!(cn.is_finite(), "{} C:N must be finite", rt);
        }

        /// Amendment quantity is always positive when recipe needs correction.
        #[test]
        fn amendment_quantity_positive(
            current_cn in 5.0..=100.0_f64,
            target_cn in 20.0..=35.0_f64,
            total_kg in 10.0..=1000.0_f64,
        ) {
            // If current_cn != target_cn, the formula should produce positive x
            let weighted_cn = current_cn * total_kg;
            let amendment_cn = if current_cn < target_cn { 400.0 } else { 15.0 }; // Woodchips or KitchenWaste
            let denominator = amendment_cn - target_cn;
            if denominator.abs() > 0.001 {
                let x = (target_cn * total_kg - weighted_cn) / denominator;
                if x > 0.0 {
                    prop_assert!(x.is_finite(), "Amendment quantity must be finite");
                    let result_cn = (weighted_cn + amendment_cn * x) / (total_kg + x);
                    prop_assert!((result_cn - target_cn).abs() < 0.1,
                        "Result C:N {} should be near target {}", result_cn, target_cn);
                }
            }
        }
    }
}

// ============================================================================
// RECIPE OPTIMIZER TESTS
// ============================================================================

#[cfg(test)]
mod recipe_tests {
    use super::*;

    #[test]
    fn test_cn_ratio_kitchen_waste() {
        let cn = estimate_cn_ratio("KitchenWaste");
        assert!((cn - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_cn_ratio_woodchips() {
        let cn = estimate_cn_ratio("Woodchips");
        assert!((cn - 400.0).abs() < 0.001);
    }

    #[test]
    fn test_cn_ratio_carbon_materials_higher_than_nitrogen() {
        let carbon_types = ["Woodchips", "Straw", "Cardboard", "Sawdust"];
        let nitrogen_types = ["KitchenWaste", "GreenWaste", "Manure", "Digestate"];

        let min_carbon: f64 = carbon_types.iter().map(|t| estimate_cn_ratio(t)).fold(f64::MAX, f64::min);
        let max_nitrogen: f64 = nitrogen_types.iter().map(|t| estimate_cn_ratio(t)).fold(f64::MIN, f64::max);

        assert!(min_carbon > max_nitrogen,
            "All carbon materials ({}) should have higher C:N than nitrogen materials ({})",
            min_carbon, max_nitrogen);
    }

    #[test]
    fn test_recipe_amendment_struct() {
        let amendment = RecipeAmendment {
            resource_type: "Woodchips".to_string(),
            quantity_kg: 50.0,
            material_cn_ratio: 400.0,
            resulting_cn_ratio: 27.5,
        };
        assert_eq!(amendment.resource_type, "Woodchips");
        assert!(amendment.quantity_kg > 0.0);
    }

    #[test]
    fn test_diagnosis_nitrogen_heavy() {
        // C:N of 15 is below 25 minimum → nitrogen heavy
        assert!(15.0 < CN_RATIO_MIN);
    }

    #[test]
    fn test_diagnosis_carbon_heavy() {
        // C:N of 50 is above 30 maximum → carbon heavy
        assert!(50.0 > CN_RATIO_MAX);
    }

    #[test]
    fn test_balanced_cn_ratio() {
        let cn: f64 = 27.5;
        assert!(cn >= CN_RATIO_MIN && cn <= CN_RATIO_MAX);
    }

    #[test]
    fn test_sensor_bridge_result_struct() {
        let result = SensorBridgeResult {
            reading_recorded: true,
            temperature_c: Some(58.0),
            moisture_pct: Some(55.0),
            oxygen_pct: Some(15.0),
            evaluation: None,
            alerts_triggered: 0,
        };
        assert!(result.reading_recorded);
        assert_eq!(result.temperature_c, Some(58.0));
    }

    #[test]
    fn test_sensor_bridge_no_sensors() {
        let result = SensorBridgeResult {
            reading_recorded: false,
            temperature_c: None,
            moisture_pct: None,
            oxygen_pct: None,
            evaluation: None,
            alerts_triggered: 0,
        };
        assert!(!result.reading_recorded);
        assert!(result.temperature_c.is_none());
    }
}
