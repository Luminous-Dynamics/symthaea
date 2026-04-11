// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waste Registry Coordinator Zome
//! Business logic for waste stream tracking, classification, and facility routing

use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;
use waste_registry_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}


// ============================================================================
// WASTE STREAMS
// ============================================================================

/// Register a new waste stream for tracking
#[hdk_extern]
pub fn register_waste_stream(stream: WasteStream) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "register_waste_stream")?;

    // Validate quantity is positive and finite
    if !stream.quantity_kg.is_finite() || stream.quantity_kg <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Quantity must be a positive finite number".into()
        )));
    }

    // Validate GPS coordinates
    if !stream.location_lat.is_finite()
        || stream.location_lat < -90.0
        || stream.location_lat > 90.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Latitude must be between -90 and 90".into()
        )));
    }
    if !stream.location_lon.is_finite()
        || stream.location_lon < -180.0
        || stream.location_lon > 180.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Longitude must be between -180 and 180".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::WasteStream(stream.clone()))?;

    // Link to all streams anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_streams".to_string())))?;
    create_link(
        anchor_hash("all_streams")?,
        action_hash.clone(),
        LinkTypes::AllStreams,
        (),
    )?;

    // Link to category anchor
    let category_anchor = format!("waste_category:{:?}", stream.waste_category);
    create_entry(&EntryTypes::Anchor(Anchor(category_anchor.clone())))?;
    create_link(
        anchor_hash(&category_anchor)?,
        action_hash.clone(),
        LinkTypes::CategoryToStreams,
        (),
    )?;

    // Link to status anchor
    let status_anchor = format!("waste_status:{:?}", stream.status);
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    create_link(
        anchor_hash(&status_anchor)?,
        action_hash.clone(),
        LinkTypes::StatusToStreams,
        (),
    )?;

    // Link agent to stream
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::AgentToStreams,
        (),
    )?;

    // Emit signal for UI
    let _ = emit_signal(&serde_json::json!({
        "event_type": "waste_stream_registered",
        "source_zome": "waste_registry",
        "payload": {
            "stream_id": stream.id,
            "category": format!("{:?}", stream.waste_category),
            "quantity_kg": stream.quantity_kg,
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get a waste stream by its action hash
#[hdk_extern]
pub fn get_waste_stream(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all registered waste streams
#[hdk_extern]
pub fn get_all_streams(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_streams")?, LinkTypes::AllStreams)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get waste streams by category
#[hdk_extern]
pub fn get_streams_by_category(category: WasteCategory) -> ExternResult<Vec<Record>> {
    let category_anchor = format!("waste_category:{:?}", category);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&category_anchor)?, LinkTypes::CategoryToStreams)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get waste streams by status
#[hdk_extern]
pub fn get_streams_by_status(status: WasteStreamStatus) -> ExternResult<Vec<Record>> {
    let status_anchor = format!("waste_status:{:?}", status);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&status_anchor)?, LinkTypes::StatusToStreams)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CLASSIFICATION
// ============================================================================

/// Classify a waste stream (human or AI)
#[hdk_extern]
pub fn classify_waste_stream(classification: WasteClassification) -> ExternResult<Record> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "classify_waste_stream")?;

    // Validate confidence
    if !classification.confidence.is_finite()
        || classification.confidence < 0.0
        || classification.confidence > 1.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Confidence must be between 0.0 and 1.0".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::WasteClassification(classification.clone()))?;

    // Link stream to classification
    create_link(
        classification.stream_hash.clone(),
        action_hash.clone(),
        LinkTypes::StreamToClassification,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "waste_stream_classified",
        "source_zome": "waste_registry",
        "payload": {
            "category": format!("{:?}", classification.category),
            "confidence": classification.confidence,
            "method": format!("{:?}", classification.method),
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get classifications for a waste stream
#[hdk_extern]
pub fn get_classifications(stream_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(stream_hash, LinkTypes::StreamToClassification)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CONTAMINATION FEEDBACK LOOP
// ============================================================================

/// Result of contamination assessment for a waste stream
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContaminationFeedback {
    /// Was contamination detected?
    pub contamination_detected: bool,
    /// Severity level (0 = clean, 3 = hazardous)
    pub severity: u8,
    /// Contaminants found (from classification mismatch)
    pub contaminants: Vec<String>,
    /// Alert sent to source?
    pub alert_emitted: bool,
}

/// Check a classification for contamination relative to the original stream category.
/// If contamination is found, emit an alert to the source and update stream status.
///
/// This closes the feedback loop: classify → detect contamination → alert source.
#[hdk_extern]
pub fn check_contamination_feedback(
    classification_hash: ActionHash,
) -> ExternResult<ContaminationFeedback> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "check_contamination_feedback")?;

    // Get the classification
    let class_record = get(classification_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Classification not found".into())))?;
    let classification: WasteClassification = class_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode classification: {}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Classification entry not found".into()
        )))?;

    // Get the original waste stream
    let stream_record = get(classification.stream_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Waste stream not found".into())))?;
    let stream: WasteStream = stream_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode stream: {}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Stream entry not found".into()
        )))?;

    // Compare classification contamination level against original
    let class_severity = classification.contamination_level.severity();
    let stream_severity = stream.contamination_level.severity();

    let contamination_detected = class_severity > stream_severity;
    let mut contaminants = Vec::new();

    // If the classified category differs from the original, record the mismatch
    if format!("{:?}", classification.category) != format!("{:?}", stream.waste_category) {
        contaminants.push(format!(
            "Expected {:?}, found {:?}",
            stream.waste_category, classification.category
        ));
    }

    // Higher contamination level detected
    if contamination_detected {
        contaminants.push(format!(
            "Contamination level escalated from {:?} to {:?}",
            stream.contamination_level, classification.contamination_level
        ));
    }

    let alert_emitted = !contaminants.is_empty();

    if alert_emitted {
        // Emit contamination alert signal for UI + upstream notification
        let _ = emit_signal(&serde_json::json!({
            "event_type": "contamination_alert",
            "source_zome": "waste_registry",
            "payload": {
                "stream_id": stream.id,
                "source_did": stream.source_did,
                "severity": class_severity,
                "contaminants": contaminants,
                "confidence": classification.confidence,
                "method": format!("{:?}", classification.method),
            }
        }));
    }

    Ok(ContaminationFeedback {
        contamination_detected,
        severity: class_severity,
        contaminants,
        alert_emitted,
    })
}

// ============================================================================
// FACILITIES
// ============================================================================

/// Register a waste processing facility (requires proposal-level consciousness)
#[hdk_extern]
pub fn register_facility(facility: WasteFacility) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "register_facility")?;

    // Validate capacity
    if !facility.capacity_kg_per_day.is_finite() || facility.capacity_kg_per_day <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Capacity must be a positive finite number".into()
        )));
    }

    // Must accept at least one category
    if facility.accepts.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Facility must accept at least one waste category".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::WasteFacility(facility.clone()))?;

    // Link to all facilities anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_facilities".to_string())))?;
    create_link(
        anchor_hash("all_facilities")?,
        action_hash.clone(),
        LinkTypes::AllFacilities,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "facility_registered",
        "source_zome": "waste_registry",
        "payload": {
            "facility_id": facility.id,
            "facility_type": format!("{:?}", facility.facility_type),
            "capacity_kg_per_day": facility.capacity_kg_per_day,
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get a facility by its action hash
#[hdk_extern]
pub fn get_facility(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all registered facilities
#[hdk_extern]
pub fn get_all_facilities(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_facilities")?, LinkTypes::AllFacilities)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// ROUTING (waste stream → facility matching)
// ============================================================================

/// Haversine distance between two GPS points in kilometers
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0; // Earth radius in km
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

/// Maximum routing distance in km (facilities beyond this score 0 for proximity)
const MAX_ROUTING_DISTANCE_KM: f64 = 100.0;

/// Minimum match score threshold for route acceptance
const MIN_ROUTE_SCORE: f32 = 0.3;

/// Input for route_waste_stream
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RouteWasteStreamInput {
    pub stream_hash: ActionHash,
}

/// Route a waste stream to the best matching facility
#[hdk_extern]
pub fn route_waste_stream(input: RouteWasteStreamInput) -> ExternResult<Option<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "route_waste_stream")?;

    // Get the waste stream
    let stream_record = get(input.stream_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Waste stream not found".into())))?;
    let stream: WasteStream = stream_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode stream: {}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Stream entry not found".into())))?;

    // Get all facilities
    let facility_links = get_links(
        LinkQuery::try_new(anchor_hash("all_facilities")?, LinkTypes::AllFacilities)?,
        GetStrategy::default(),
    )?;
    let facility_records = records_from_links(facility_links)?;

    // Score each facility
    let mut best_score: f32 = 0.0;
    let mut best_factors = None;
    let mut best_facility_hash = None;

    for record in &facility_records {
        let facility: WasteFacility = match record.entry().to_app_option() {
            Ok(Some(f)) => f,
            _ => continue,
        };

        // Skip closed or at-capacity facilities
        if facility.status == FacilityStatus::Closed {
            continue;
        }

        // Category match (hard gate)
        let category_match = if facility.accepts.contains(&stream.waste_category) {
            1.0_f32
        } else {
            0.0
        };

        // Proximity score (inverse distance, clamped)
        let distance = haversine_km(
            stream.location_lat,
            stream.location_lon,
            facility.location_lat,
            facility.location_lon,
        );
        let proximity_score = if distance >= MAX_ROUTING_DISTANCE_KM {
            0.0_f32
        } else {
            (1.0 - distance / MAX_ROUTING_DISTANCE_KM) as f32
        };

        // Capacity score
        let available = facility.capacity_kg_per_day - facility.current_load_kg;
        let capacity_score = if available <= 0.0 {
            0.0_f32
        } else {
            (available / facility.capacity_kg_per_day).min(1.0) as f32
        };

        // Contamination compatibility
        let contamination_compatibility = match stream.contamination_level {
            ContaminationLevel::Clean => 1.0_f32,
            ContaminationLevel::LightlyContaminated => 0.8,
            ContaminationLevel::Contaminated => {
                // Only MRFs and chemical recyclers handle contaminated streams
                match facility.facility_type {
                    FacilityType::MRF | FacilityType::ChemicalRecycler => 0.6,
                    _ => 0.2,
                }
            }
            ContaminationLevel::Hazardous => {
                // Only specialized facilities handle hazardous
                match facility.facility_type {
                    FacilityType::ChemicalRecycler => 0.4,
                    _ => 0.0,
                }
            }
        };

        let factors = WasteRouteFactors {
            category_match,
            proximity_score,
            capacity_score,
            contamination_compatibility,
        };

        let score = factors.overall_score();
        if score > best_score {
            best_score = score;
            best_factors = Some(factors);
            best_facility_hash = Some(record.action_address().clone());
        }
    }

    // Check minimum threshold
    if best_score < MIN_ROUTE_SCORE {
        return Ok(None);
    }

    let facility_hash = best_facility_hash
        .ok_or(wasm_error!(WasmErrorInner::Guest("No matching facility found".into())))?;
    let factors = best_factors
        .ok_or(wasm_error!(WasmErrorInner::Guest("No matching factors found".into())))?;

    // Create route entry
    let route = WasteRoute {
        stream_hash: input.stream_hash.clone(),
        facility_hash: facility_hash.clone(),
        factors,
        overall_score: best_score,
        routed_at: sys_time()?.as_micros() as u64,
    };

    let route_hash = create_entry(&EntryTypes::WasteRoute(route))?;

    // Link stream to route
    create_link(
        input.stream_hash,
        route_hash.clone(),
        LinkTypes::StreamToRoute,
        (),
    )?;

    // Link facility to stream
    create_link(
        facility_hash,
        route_hash.clone(),
        LinkTypes::FacilityToStreams,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "waste_stream_routed",
        "source_zome": "waste_registry",
        "payload": {
            "overall_score": best_score,
        }
    }));

    let record = get(route_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get route record".into())))?;
    Ok(Some(record))
}

// ============================================================================
// DIVERSION METRICS
// ============================================================================

/// Input for calculating diversion rate
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiversionRateInput {
    pub period_start_us: u64,
    pub period_end_us: u64,
}

/// Diversion rate result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiversionRateResult {
    pub total_kg: f64,
    pub composted_kg: f64,
    pub recycled_kg: f64,
    pub landfilled_kg: f64,
    pub diversion_rate_pct: f64,
}

/// Calculate waste diversion rate for a given period
#[hdk_extern]
pub fn calculate_diversion_rate(input: DiversionRateInput) -> ExternResult<DiversionRateResult> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_streams")?, LinkTypes::AllStreams)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    let mut total_kg = 0.0_f64;
    let mut composted_kg = 0.0_f64;
    let mut recycled_kg = 0.0_f64;
    let mut landfilled_kg = 0.0_f64;

    for record in records {
        let stream: WasteStream = match record.entry().to_app_option() {
            Ok(Some(s)) => s,
            _ => continue,
        };

        // Filter by period
        if stream.created_at < input.period_start_us || stream.created_at > input.period_end_us {
            continue;
        }

        // Only count completed streams
        if stream.status != WasteStreamStatus::Completed {
            continue;
        }

        total_kg += stream.quantity_kg;

        match stream.recommended_eol {
            EndOfLifeStrategy::IndustrialCompost | EndOfLifeStrategy::Biodegradable => {
                composted_kg += stream.quantity_kg;
            }
            EndOfLifeStrategy::MechanicalRecycling
            | EndOfLifeStrategy::ChemicalRecycling
            | EndOfLifeStrategy::Downcycle => {
                recycled_kg += stream.quantity_kg;
            }
            EndOfLifeStrategy::Landfill => {
                landfilled_kg += stream.quantity_kg;
            }
        }
    }

    let diverted = composted_kg + recycled_kg;
    let diversion_rate_pct = if total_kg > 0.0 {
        (diverted / total_kg) * 100.0
    } else {
        0.0
    };

    Ok(DiversionRateResult {
        total_kg,
        composted_kg,
        recycled_kg,
        landfilled_kg,
        diversion_rate_pct,
    })
}

// ============================================================================
// HISTORICAL TREND ANALYTICS
// ============================================================================

/// A single data point in a time series.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrendPoint {
    /// Period start (Unix microseconds)
    pub period_start_us: u64,
    /// Period end (Unix microseconds)
    pub period_end_us: u64,
    /// Value for this period
    pub value: f64,
}

/// Input for trend queries.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrendQueryInput {
    /// Start of the analysis window (Unix microseconds)
    pub window_start_us: u64,
    /// End of the analysis window (Unix microseconds)
    pub window_end_us: u64,
    /// Number of buckets to divide the window into
    pub num_buckets: u32,
}

/// Waste volume trend over time (total kg per bucket).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VolumeTrendResult {
    /// Time series of waste volumes
    pub trend: Vec<TrendPoint>,
    /// Total across all buckets
    pub total_kg: f64,
    /// Average per bucket
    pub average_kg_per_bucket: f64,
}

/// Diversion rate trend over time.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiversionTrendResult {
    /// Time series of diversion rates (0-100%)
    pub trend: Vec<TrendPoint>,
    /// Overall diversion rate across the window
    pub overall_rate_pct: f64,
}

/// Category breakdown for a time period.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CategoryBreakdown {
    pub category: String,
    pub quantity_kg: f64,
    pub percentage: f64,
}

/// Category distribution result.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CategoryDistributionResult {
    pub breakdown: Vec<CategoryBreakdown>,
    pub total_kg: f64,
}

/// Get waste volume trend over time (sparkline-compatible).
#[hdk_extern]
pub fn get_volume_trend(input: TrendQueryInput) -> ExternResult<VolumeTrendResult> {
    if input.num_buckets == 0 || input.num_buckets > 365 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "num_buckets must be between 1 and 365".into()
        )));
    }
    if input.window_start_us >= input.window_end_us {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "window_start must be before window_end".into()
        )));
    }

    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_streams")?, LinkTypes::AllStreams)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    let bucket_width = (input.window_end_us - input.window_start_us) / input.num_buckets as u64;
    let mut buckets = vec![0.0_f64; input.num_buckets as usize];

    for record in &records {
        let stream: WasteStream = match record.entry().to_app_option() {
            Ok(Some(s)) => s,
            _ => continue,
        };
        if stream.created_at < input.window_start_us || stream.created_at >= input.window_end_us {
            continue;
        }
        let bucket_idx =
            ((stream.created_at - input.window_start_us) / bucket_width).min(input.num_buckets as u64 - 1) as usize;
        buckets[bucket_idx] += stream.quantity_kg;
    }

    let total_kg: f64 = buckets.iter().sum();
    let average_kg_per_bucket = total_kg / input.num_buckets as f64;

    let trend: Vec<TrendPoint> = buckets
        .iter()
        .enumerate()
        .map(|(i, &val)| TrendPoint {
            period_start_us: input.window_start_us + i as u64 * bucket_width,
            period_end_us: input.window_start_us + (i + 1) as u64 * bucket_width,
            value: (val * 10.0).round() / 10.0,
        })
        .collect();

    Ok(VolumeTrendResult {
        trend,
        total_kg: (total_kg * 10.0).round() / 10.0,
        average_kg_per_bucket: (average_kg_per_bucket * 10.0).round() / 10.0,
    })
}

/// Get category distribution for a time period.
#[hdk_extern]
pub fn get_category_distribution(input: TrendQueryInput) -> ExternResult<CategoryDistributionResult> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_streams")?, LinkTypes::AllStreams)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    let mut category_totals: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut total_kg: f64 = 0.0;

    for record in &records {
        let stream: WasteStream = match record.entry().to_app_option() {
            Ok(Some(s)) => s,
            _ => continue,
        };
        if stream.created_at < input.window_start_us || stream.created_at >= input.window_end_us {
            continue;
        }
        let category = format!("{:?}", stream.waste_category);
        *category_totals.entry(category).or_insert(0.0) += stream.quantity_kg;
        total_kg += stream.quantity_kg;
    }

    let mut breakdown: Vec<CategoryBreakdown> = category_totals
        .into_iter()
        .map(|(category, quantity_kg)| {
            let percentage = if total_kg > 0.0 {
                (quantity_kg / total_kg * 100.0 * 10.0).round() / 10.0
            } else {
                0.0
            };
            CategoryBreakdown {
                category,
                quantity_kg: (quantity_kg * 10.0).round() / 10.0,
                percentage,
            }
        })
        .collect();

    // Sort by quantity descending
    breakdown.sort_by(|a, b| {
        b.quantity_kg
            .partial_cmp(&a.quantity_kg)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(CategoryDistributionResult {
        breakdown,
        total_kg: (total_kg * 10.0).round() / 10.0,
    })
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_same_point() {
        let d = haversine_km(32.9483, -96.7299, 32.9483, -96.7299);
        assert!(d < 0.001, "Same point should have ~0 distance, got {}", d);
    }

    #[test]
    fn test_haversine_known_distance() {
        // Dallas to Austin ~290 km
        let d = haversine_km(32.7767, -96.7970, 30.2672, -97.7431);
        assert!(d > 250.0 && d < 320.0, "Dallas-Austin should be ~290km, got {}", d);
    }

    #[test]
    fn test_haversine_antipodal() {
        // North pole to south pole ~20,000 km
        let d = haversine_km(90.0, 0.0, -90.0, 0.0);
        assert!(d > 19_000.0 && d < 21_000.0, "Pole-to-pole should be ~20,000km, got {}", d);
    }

    #[test]
    fn test_min_route_score_threshold() {
        assert!(MIN_ROUTE_SCORE > 0.0 && MIN_ROUTE_SCORE < 1.0);
    }

    #[test]
    fn test_max_routing_distance() {
        assert!(MAX_ROUTING_DISTANCE_KM > 0.0);
    }

    #[test]
    fn test_diversion_rate_empty() {
        let result = DiversionRateResult {
            total_kg: 0.0,
            composted_kg: 0.0,
            recycled_kg: 0.0,
            landfilled_kg: 0.0,
            diversion_rate_pct: 0.0,
        };
        assert_eq!(result.diversion_rate_pct, 0.0);
    }

    #[test]
    fn test_diversion_rate_all_diverted() {
        let total: f64 = 100.0;
        let composted: f64 = 60.0;
        let recycled: f64 = 40.0;
        let diverted = composted + recycled;
        let rate = (diverted / total) * 100.0;
        assert!((rate - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_diversion_rate_partial() {
        let total: f64 = 100.0;
        let composted: f64 = 30.0;
        let recycled: f64 = 20.0;
        let landfilled: f64 = 50.0;
        assert!((total - composted - recycled - landfilled).abs() < 0.001);
        let diverted = composted + recycled;
        let rate = (diverted / total) * 100.0;
        assert!((rate - 50.0).abs() < 0.001);
    }

    // -- Contamination feedback tests --

    #[test]
    fn test_contamination_feedback_struct() {
        let feedback = ContaminationFeedback {
            contamination_detected: true,
            severity: 2,
            contaminants: vec!["Plastic in organic stream".to_string()],
            alert_emitted: true,
        };
        assert!(feedback.contamination_detected);
        assert_eq!(feedback.severity, 2);
        assert!(!feedback.contaminants.is_empty());
        assert!(feedback.alert_emitted);
    }

    #[test]
    fn test_clean_stream_no_feedback() {
        let feedback = ContaminationFeedback {
            contamination_detected: false,
            severity: 0,
            contaminants: vec![],
            alert_emitted: false,
        };
        assert!(!feedback.contamination_detected);
        assert!(feedback.contaminants.is_empty());
        assert!(!feedback.alert_emitted);
    }

    // -- Analytics tests --

    #[test]
    fn test_trend_point_struct() {
        let point = TrendPoint {
            period_start_us: 1711100000_000000,
            period_end_us: 1711200000_000000,
            value: 150.0,
        };
        assert!(point.period_start_us < point.period_end_us);
        assert!(point.value >= 0.0);
    }

    #[test]
    fn test_volume_trend_result_averages() {
        let result = VolumeTrendResult {
            trend: vec![
                TrendPoint { period_start_us: 0, period_end_us: 100, value: 50.0 },
                TrendPoint { period_start_us: 100, period_end_us: 200, value: 150.0 },
            ],
            total_kg: 200.0,
            average_kg_per_bucket: 100.0,
        };
        assert!((result.total_kg - 200.0).abs() < 0.001);
        assert!((result.average_kg_per_bucket - 100.0).abs() < 0.001);
        assert_eq!(result.trend.len(), 2);
    }

    #[test]
    fn test_category_breakdown_percentages_sum() {
        let breakdown = vec![
            CategoryBreakdown { category: "Organic".to_string(), quantity_kg: 60.0, percentage: 60.0 },
            CategoryBreakdown { category: "Recyclable".to_string(), quantity_kg: 30.0, percentage: 30.0 },
            CategoryBreakdown { category: "Mixed".to_string(), quantity_kg: 10.0, percentage: 10.0 },
        ];
        let total_pct: f64 = breakdown.iter().map(|b| b.percentage).sum();
        assert!((total_pct - 100.0).abs() < 0.1, "Percentages should sum to ~100: {}", total_pct);
    }

    #[test]
    fn test_category_distribution_empty() {
        let result = CategoryDistributionResult {
            breakdown: vec![],
            total_kg: 0.0,
        };
        assert!(result.breakdown.is_empty());
        assert_eq!(result.total_kg, 0.0);
    }
}
