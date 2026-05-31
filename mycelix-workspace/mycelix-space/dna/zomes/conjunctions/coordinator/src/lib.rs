// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Conjunctions Coordinator Zome
//!
//! Functions for managing conjunction events, CDMs, and avoidance maneuvers.
//! Includes real-time alert signals for high-risk conjunctions.

use conjunctions_integrity::*;
use hdk::prelude::*;
use mycelix_space_shared::{
    AlertPriority, AlertType, ConjunctionAlert, ConjunctionAssessment, ConjunctionDataMessage,
    ManeuverAlert, ManeuverType as SharedManeuverType, ScreenInput, SpaceSignal, SpaceTimestamp,
};
use orbital_mechanics::conjunction::ConjunctionAnalyzer;
use orbital_mechanics::propagator::Propagator;
use orbital_mechanics::tle::TwoLineElement;

/// Signal types for this zome (used by Holochain's emit_signal)
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub enum ConjunctionSignal {
    /// Alert for conjunction detection or risk change
    Alert(SpaceSignal),
    /// CDM update notification
    CdmUpdate { event_id: String, version: u32 },
    /// Maneuver announcement
    ManeuverAnnounced { event_id: String, norad_id: u32 },
}

/// Risk level threshold for automatic alerts
const ALERT_THRESHOLD: RiskLevel = RiskLevel::Medium;

/// Create a conjunction event
///
/// When `compute_details` is true and TLEs are provided, the collision probability,
/// miss distance, and risk level are computed using SGP4 propagation and Alfano 2D
/// analysis. Otherwise, the manually-provided values are used (backward compatible).
#[hdk_extern]
pub fn create_conjunction_event(input: CreateEventInput) -> ExternResult<ActionHash> {
    if input.event_id.is_empty() || input.event_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event ID must be 1-256 characters".into()
        )));
    }
    if input.primary_norad_id == input.secondary_norad_id {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Primary and secondary NORAD IDs must be different".into()
        )));
    }
    if !input.miss_distance_km.is_finite() || input.miss_distance_km < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Miss distance must be a non-negative finite number".into()
        )));
    }
    if !input.max_pc.is_finite() || input.max_pc < 0.0 || input.max_pc > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Max Pc must be between 0.0 and 1.0".into()
        )));
    }
    let (miss_distance_km, max_pc, risk_level) = if input.compute_details {
        // Compute values from TLEs using orbital-mechanics library
        let (primary_lines, secondary_lines) = match (&input.primary_tle, &input.secondary_tle) {
            (Some(p), Some(s)) => (p, s),
            _ => {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "primary_tle and secondary_tle are required when compute_details is true"
                        .to_string()
                )));
            }
        };

        let primary_tle = TwoLineElement::parse_lines(None, &primary_lines.0, &primary_lines.1)
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Invalid primary TLE: {}", e)))
            })?;
        let secondary_tle =
            TwoLineElement::parse_lines(None, &secondary_lines.0, &secondary_lines.1).map_err(
                |e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Invalid secondary TLE: {}",
                        e
                    )))
                },
            )?;

        let primary_prop = Propagator::from_tle(&primary_tle).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Primary propagation init failed: {}",
                e
            )))
        })?;
        let secondary_prop = Propagator::from_tle(&secondary_tle).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Secondary propagation init failed: {}",
                e
            )))
        })?;

        let tca_dt = input.tca.to_datetime();
        let primary_state = primary_prop.propagate_to(tca_dt).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Primary propagation failed: {}",
                e
            )))
        })?;
        let secondary_state = secondary_prop.propagate_to(tca_dt).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Secondary propagation failed: {}",
                e
            )))
        })?;

        let analyzer = ConjunctionAnalyzer::new();
        let assessment = analyzer.assess(&primary_state, &secondary_state);

        (
            assessment.miss_distance_km,
            assessment.collision_probability.pc,
            convert_lib_risk_to_integrity(assessment.risk_level),
        )
    } else {
        // Use manually-provided values (backward compatible)
        (input.miss_distance_km, input.max_pc, input.risk_level)
    };

    let event = ConjunctionEvent {
        event_id: input.event_id.clone(),
        primary_norad_id: input.primary_norad_id,
        secondary_norad_id: input.secondary_norad_id,
        tca: input.tca.clone(),
        miss_distance_km,
        max_pc,
        risk_level,
        status: EventStatus::Screening,
        created_at: SpaceTimestamp::now(),
        updated_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::ConjunctionEvent(event.clone()))?;

    // Emit signal if risk level is at or above threshold
    if risk_level >= ALERT_THRESHOLD {
        emit_conjunction_alert(&event)?;
    }

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateEventInput {
    pub event_id: String,
    pub primary_norad_id: u32,
    pub secondary_norad_id: u32,
    pub tca: SpaceTimestamp,
    pub miss_distance_km: f64,
    pub max_pc: f64,
    pub risk_level: RiskLevel,
    /// If true and TLEs are provided, compute collision_probability and miss_distance
    /// using SGP4 propagation and Alfano 2D analysis instead of using manual values.
    #[serde(default)]
    pub compute_details: bool,
    /// Primary TLE (line1, line2) - required when compute_details is true
    pub primary_tle: Option<(String, String)>,
    /// Secondary TLE (line1, line2) - required when compute_details is true
    pub secondary_tle: Option<(String, String)>,
}

/// Update conjunction event risk level (with signal emission)
#[hdk_extern]
pub fn update_conjunction_risk(input: UpdateRiskInput) -> ExternResult<ActionHash> {
    // Get the current event
    let record = get(input.event_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Conjunction event not found".to_string())
    ))?;

    let mut event: ConjunctionEvent = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Entry is not a ConjunctionEvent".to_string()
        )))?;

    let previous_risk = event.risk_level;

    // Update the event
    event.risk_level = input.new_risk_level;
    event.max_pc = input.new_pc;
    event.miss_distance_km = input.new_miss_distance_km;
    event.updated_at = SpaceTimestamp::now();

    let action_hash = update_entry(input.event_hash, &event)?;

    // Emit escalation/de-escalation signal if crossing threshold
    if previous_risk < ALERT_THRESHOLD && input.new_risk_level >= ALERT_THRESHOLD {
        // Risk escalated above threshold
        emit_risk_escalation_alert(&event, previous_risk)?;
    } else if input.new_risk_level >= ALERT_THRESHOLD {
        // Already above threshold, emit update
        emit_conjunction_alert(&event)?;
    }

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateRiskInput {
    pub event_hash: ActionHash,
    pub new_risk_level: RiskLevel,
    pub new_pc: f64,
    pub new_miss_distance_km: f64,
}

/// Submit a CDM for a conjunction event
#[hdk_extern]
pub fn submit_cdm(input: SubmitCdmInput) -> ExternResult<ActionHash> {
    if input.cdm.conjunction_id.is_empty() || input.cdm.conjunction_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Conjunction ID must be 1-256 characters".into()
        )));
    }
    let cdm_entry = CdmEntry {
        cdm: input.cdm.clone(),
        version: input.version,
        supersedes: input.supersedes.clone(),
    };

    let action_hash = create_entry(&EntryTypes::Cdm(cdm_entry))?;

    // Emit CDM update signal
    let signal = ConjunctionSignal::CdmUpdate {
        event_id: input.cdm.conjunction_id.clone(),
        version: input.version,
    };
    emit_signal(signal)?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitCdmInput {
    pub cdm: ConjunctionDataMessage,
    pub version: u32,
    pub supersedes: Option<ActionHash>,
}

/// Announce an avoidance maneuver
#[hdk_extern]
pub fn announce_maneuver(input: AnnounceManeuverInput) -> ExternResult<ActionHash> {
    if input.event_id.is_empty() || input.event_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event ID must be 1-256 characters".into()
        )));
    }
    if !input.delta_v_ms.is_finite() || input.delta_v_ms <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Delta-V must be a positive finite number".into()
        )));
    }
    for &component in &input.direction {
        if !component.is_finite() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Direction components must be finite numbers".into()
            )));
        }
    }
    let agent = agent_info()?.agent_initial_pubkey;

    let maneuver = AvoidanceManeuver {
        event_id: input.event_id.clone(),
        norad_id: input.norad_id,
        operator: agent,
        burn_time: input.burn_time.clone(),
        delta_v_ms: input.delta_v_ms,
        direction: input.direction,
        status: ManeuverStatus::Announced,
        created_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::AvoidanceManeuver(maneuver))?;

    // Emit maneuver announcement signal
    let maneuver_alert = ManeuverAlert {
        alert_type: AlertType::ManeuverRequired,
        priority: AlertPriority::High,
        norad_id: input.norad_id,
        conjunction_primary: None, // Could be populated from event lookup
        conjunction_secondary: None,
        maneuver_type: SharedManeuverType::Combined, // Determine from direction
        delta_v_ms: input.delta_v_ms,
        execution_time: input.burn_time.to_datetime(),
        generated_at: chrono::Utc::now(),
        message: Some(format!(
            "Avoidance maneuver announced for event {}",
            input.event_id
        )),
    };

    let signal = ConjunctionSignal::Alert(SpaceSignal::Maneuver(maneuver_alert));
    emit_signal(signal)?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnounceManeuverInput {
    pub event_id: String,
    pub norad_id: u32,
    pub burn_time: SpaceTimestamp,
    pub delta_v_ms: f64,
    pub direction: [f64; 3],
}

/// Mark maneuver as executed
#[hdk_extern]
pub fn mark_maneuver_executed(input: ManeuverExecutedInput) -> ExternResult<ActionHash> {
    // Get the current maneuver
    let record = get(input.maneuver_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Maneuver not found".to_string())
    ))?;

    let mut maneuver: AvoidanceManeuver = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Entry is not an AvoidanceManeuver".to_string()
        )))?;

    // Update status
    maneuver.status = ManeuverStatus::Executed;

    let action_hash = update_entry(input.maneuver_hash, &maneuver)?;

    // Emit executed signal
    let maneuver_alert = ManeuverAlert {
        alert_type: AlertType::ManeuverExecuted,
        priority: AlertPriority::Medium,
        norad_id: maneuver.norad_id,
        conjunction_primary: None,
        conjunction_secondary: None,
        maneuver_type: SharedManeuverType::Combined,
        delta_v_ms: maneuver.delta_v_ms,
        execution_time: maneuver.burn_time.to_datetime(),
        generated_at: chrono::Utc::now(),
        message: Some(format!("Maneuver executed for event {}", maneuver.event_id)),
    };

    let signal = ConjunctionSignal::Alert(SpaceSignal::Maneuver(maneuver_alert));
    emit_signal(signal)?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManeuverExecutedInput {
    pub maneuver_hash: ActionHash,
}

/// Get all high-risk conjunctions
#[hdk_extern]
pub fn get_high_risk_conjunctions(_: ()) -> ExternResult<Vec<ConjunctionEvent>> {
    // Query all conjunction events with risk level >= High
    // In a real implementation, this would use a path or link query
    // For now, return empty - would need proper indexing
    Ok(vec![])
}

/// Query conjunctions by object
#[hdk_extern]
pub fn get_conjunctions_for_object(_norad_id: u32) -> ExternResult<Vec<ConjunctionEvent>> {
    // Would query by link or path anchored on the NORAD ID
    Ok(vec![])
}

/// Screen a primary object against secondaries using SGP4 propagation
///
/// Fetches TLEs from the DHT, propagates all objects to the screening time,
/// and returns assessments for pairs exceeding the Pc threshold.
///
/// Max 100 secondaries per call (SGP4 in WASM is expensive).
#[hdk_extern]
pub fn screen_conjunction(input: ScreenInput) -> ExternResult<Vec<ConjunctionAssessment>> {
    // Bound secondary count to avoid excessive WASM computation
    if input.secondary_norad_ids.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum 100 secondary objects per screening call".to_string()
        )));
    }

    if input.hours_ahead <= 0.0 || input.hours_ahead > 720.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "hours_ahead must be between 0 and 720 (30 days)".to_string()
        )));
    }

    // Collect all NORAD IDs we need
    let mut all_ids: Vec<u32> = vec![input.primary_norad_id];
    for id in &input.secondary_norad_ids {
        all_ids.push(*id);
    }

    // Cross-zome call to orbital_objects to fetch latest TLEs
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("orbital_objects"),
        FunctionName::new("get_latest_tles"),
        None,
        all_ids,
    )?;

    let response_bytes = match response {
        ZomeCallResponse::Ok(bytes) => bytes,
        ZomeCallResponse::Unauthorized(..) => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Unauthorized cross-zome call to orbital_objects".to_string()
            )));
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cross-zome call to orbital_objects failed".to_string()
            )));
        }
    };

    let tle_lines: Vec<TleLinesResponse> = response_bytes.decode().map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to decode TLE response: {:?}",
            e
        )))
    })?;

    // Build a lookup by NORAD ID
    let tle_map: std::collections::HashMap<u32, &TleLinesResponse> =
        tle_lines.iter().map(|t| (t.norad_id, t)).collect();

    // Get primary TLE
    let primary_tle_data = tle_map.get(&input.primary_norad_id).ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "No TLE found for primary object {}",
            input.primary_norad_id
        )))
    })?;

    // Delegate to the TLE-based screening function
    let secondary_tles: Vec<(String, String)> = input
        .secondary_norad_ids
        .iter()
        .filter_map(|id| tle_map.get(id).map(|t| (t.line1.clone(), t.line2.clone())))
        .collect();

    let from_tles_input = ScreenFromTlesInput {
        primary_tle: (
            primary_tle_data.line1.clone(),
            primary_tle_data.line2.clone(),
        ),
        secondary_tles,
        hours_ahead: input.hours_ahead,
        pc_threshold: input.pc_threshold,
    };

    screen_conjunction_from_tles(from_tles_input)
}

/// Screen conjunction with TLE data provided directly
///
/// This avoids cross-zome DHT queries by accepting TLE lines as input.
/// The caller is responsible for fetching the latest TLEs.
#[hdk_extern]
pub fn screen_conjunction_from_tles(
    input: ScreenFromTlesInput,
) -> ExternResult<Vec<ConjunctionAssessment>> {
    if input.secondary_tles.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum 100 secondary objects per screening call".to_string()
        )));
    }

    if input.hours_ahead <= 0.0 || input.hours_ahead > 720.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "hours_ahead must be between 0 and 720 (30 days)".to_string()
        )));
    }

    // Parse primary TLE
    let primary_tle = TwoLineElement::parse_lines(None, &input.primary_tle.0, &input.primary_tle.1)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid primary TLE: {}", e))))?;

    let primary_prop = Propagator::from_tle(&primary_tle).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Primary propagation init failed: {}",
            e
        )))
    })?;

    // Parse secondary TLEs
    let mut secondary_props: Vec<(u32, Propagator)> = Vec::new();
    for (line1, line2) in &input.secondary_tles {
        match TwoLineElement::parse_lines(None, line1, line2) {
            Ok(tle) => {
                let norad_id = tle.norad_id;
                match Propagator::from_tle(&tle) {
                    Ok(prop) => secondary_props.push((norad_id, prop)),
                    Err(_) => continue, // Skip objects that fail to initialize
                }
            }
            Err(_) => continue, // Skip malformed TLEs
        }
    }

    // Sweep the time window to find TCA (time of closest approach) for each pair.
    // Step size: 10 minutes for coarse scan, then refine around the minimum.
    let total_seconds = (input.hours_ahead * 3600.0) as i64;
    let coarse_step_secs: i64 = 600; // 10 minutes
    let fine_step_secs: i64 = 10; // 10 seconds for refinement
    let num_coarse_steps = (total_seconds / coarse_step_secs).max(1);

    let analyzer = ConjunctionAnalyzer::new();
    let mut results: Vec<ConjunctionAssessment> = Vec::new();

    for (_norad_id, sec_prop) in &secondary_props {
        // Coarse scan: find the time step with minimum miss distance
        let mut best_distance = f64::MAX;
        let mut best_step: i64 = 0;

        for i in 0..=num_coarse_steps {
            let t = primary_tle.epoch + chrono::Duration::seconds(i * coarse_step_secs);
            let p_state = match primary_prop.propagate_to(t) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let s_state = match sec_prop.propagate_to(t) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let dist = distance_between(&p_state, &s_state);
            if dist < best_distance {
                best_distance = dist;
                best_step = i * coarse_step_secs;
            }
        }

        // Fine scan: refine around the best coarse step
        let fine_start = (best_step - coarse_step_secs).max(0);
        let fine_end = (best_step + coarse_step_secs).min(total_seconds);
        let mut tca_offset = best_step;
        let mut tca_distance = best_distance;

        let mut t_secs = fine_start;
        while t_secs <= fine_end {
            let t = primary_tle.epoch + chrono::Duration::seconds(t_secs);
            let p_state = match primary_prop.propagate_to(t) {
                Ok(s) => s,
                Err(_) => {
                    t_secs += fine_step_secs;
                    continue;
                }
            };
            let s_state = match sec_prop.propagate_to(t) {
                Ok(s) => s,
                Err(_) => {
                    t_secs += fine_step_secs;
                    continue;
                }
            };
            let dist = distance_between(&p_state, &s_state);
            if dist < tca_distance {
                tca_distance = dist;
                tca_offset = t_secs;
            }
            t_secs += fine_step_secs;
        }

        // Assess at TCA
        let tca_time = primary_tle.epoch + chrono::Duration::seconds(tca_offset);
        let p_state = match primary_prop.propagate_to(tca_time) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let s_state = match sec_prop.propagate_to(tca_time) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let assessment = analyzer.assess(&p_state, &s_state);

        if assessment.collision_probability.pc >= input.pc_threshold {
            let shared_assessment = ConjunctionAssessment {
                primary_norad_id: assessment.primary_norad_id,
                secondary_norad_id: assessment.secondary_norad_id,
                tca: assessment.tca,
                miss_distance_km: assessment.miss_distance_km,
                relative_velocity_kms: assessment.relative_velocity_kms,
                collision_probability: assessment.collision_probability.pc,
                pc_method: mycelix_space_shared::PcMethod::Alfano2D,
                risk_level: convert_lib_risk_level(assessment.risk_level),
                hard_body_radius_m: assessment.hard_body_radius_m,
                screening_volume_km: 5.0,
            };
            results.push(shared_assessment);
        }
    }

    // Sort by collision probability descending (highest risk first)
    results.sort_by(|a, b| {
        b.collision_probability
            .partial_cmp(&a.collision_probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// TLE lines response from cross-zome call to orbital_objects
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TleLinesResponse {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
}

/// Input for screening with TLE data provided directly
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreenFromTlesInput {
    /// Primary object TLE (line1, line2)
    pub primary_tle: (String, String),
    /// Secondary object TLEs (line1, line2) pairs
    pub secondary_tles: Vec<(String, String)>,
    /// Hours to propagate ahead
    pub hours_ahead: f64,
    /// Minimum Pc to include in results
    pub pc_threshold: f64,
}

/// Euclidean distance between two orbital states (km)
fn distance_between(
    a: &orbital_mechanics::state::OrbitalState,
    b: &orbital_mechanics::state::OrbitalState,
) -> f64 {
    a.state.distance_to(&b.state)
}

/// Convert orbital-mechanics RiskLevel to shared RiskLevel
fn convert_lib_risk_level(
    level: orbital_mechanics::conjunction::RiskLevel,
) -> mycelix_space_shared::RiskLevel {
    match level {
        orbital_mechanics::conjunction::RiskLevel::Negligible => {
            mycelix_space_shared::RiskLevel::Negligible
        }
        orbital_mechanics::conjunction::RiskLevel::Low => mycelix_space_shared::RiskLevel::Low,
        orbital_mechanics::conjunction::RiskLevel::Medium => {
            mycelix_space_shared::RiskLevel::Medium
        }
        orbital_mechanics::conjunction::RiskLevel::High => mycelix_space_shared::RiskLevel::High,
        orbital_mechanics::conjunction::RiskLevel::Emergency => {
            mycelix_space_shared::RiskLevel::Emergency
        }
    }
}

/// Convert orbital-mechanics RiskLevel to integrity RiskLevel
fn convert_lib_risk_to_integrity(level: orbital_mechanics::conjunction::RiskLevel) -> RiskLevel {
    match level {
        orbital_mechanics::conjunction::RiskLevel::Negligible => RiskLevel::Negligible,
        orbital_mechanics::conjunction::RiskLevel::Low => RiskLevel::Low,
        orbital_mechanics::conjunction::RiskLevel::Medium => RiskLevel::Medium,
        orbital_mechanics::conjunction::RiskLevel::High => RiskLevel::High,
        orbital_mechanics::conjunction::RiskLevel::Emergency => RiskLevel::Emergency,
    }
}

// =============================================================================
// Internal Helper Functions
// =============================================================================

/// Emit a conjunction alert signal
fn emit_conjunction_alert(event: &ConjunctionEvent) -> ExternResult<()> {
    let assessment = ConjunctionAssessment {
        primary_norad_id: event.primary_norad_id,
        secondary_norad_id: event.secondary_norad_id,
        tca: event.tca.to_datetime(),
        miss_distance_km: event.miss_distance_km,
        relative_velocity_kms: 10.0, // Would come from actual data
        collision_probability: event.max_pc,
        pc_method: mycelix_space_shared::PcMethod::Alfano2D,
        risk_level: convert_risk_level(event.risk_level),
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::new_conjunction(&assessment);
    let signal = ConjunctionSignal::Alert(SpaceSignal::Conjunction(alert));
    emit_signal(signal)?;

    Ok(())
}

/// Emit a risk escalation alert signal
fn emit_risk_escalation_alert(
    event: &ConjunctionEvent,
    previous_risk: RiskLevel,
) -> ExternResult<()> {
    let assessment = ConjunctionAssessment {
        primary_norad_id: event.primary_norad_id,
        secondary_norad_id: event.secondary_norad_id,
        tca: event.tca.to_datetime(),
        miss_distance_km: event.miss_distance_km,
        relative_velocity_kms: 10.0,
        collision_probability: event.max_pc,
        pc_method: mycelix_space_shared::PcMethod::Alfano2D,
        risk_level: convert_risk_level(event.risk_level),
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::risk_escalation(&assessment, convert_risk_level(previous_risk));
    let signal = ConjunctionSignal::Alert(SpaceSignal::Conjunction(alert));
    emit_signal(signal)?;

    Ok(())
}

/// Convert integrity RiskLevel to shared RiskLevel
fn convert_risk_level(level: RiskLevel) -> mycelix_space_shared::RiskLevel {
    match level {
        RiskLevel::Negligible => mycelix_space_shared::RiskLevel::Negligible,
        RiskLevel::Low => mycelix_space_shared::RiskLevel::Low,
        RiskLevel::Medium => mycelix_space_shared::RiskLevel::Medium,
        RiskLevel::High => mycelix_space_shared::RiskLevel::High,
        RiskLevel::Emergency => mycelix_space_shared::RiskLevel::Emergency,
    }
}
