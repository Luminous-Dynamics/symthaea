// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compost Control Integrity Zome
//! Entry types for composting process management: batches, sensor readings, and actions

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// COMPOST BATCH
// ============================================================================

/// Composting method
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CompostMethod {
    /// Open-air rows, turned periodically
    Windrow,
    /// Forced aeration with static piles
    StaticAerated,
    /// Enclosed vessel with controlled conditions
    InVessel,
    /// Worm-mediated decomposition
    Vermicompost,
    /// Anaerobic fermentation (Japanese method)
    Bokashi,
}

/// Lifecycle status of a compost batch
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CompostBatchStatus {
    /// Initial mixing of inputs
    Mixing,
    /// Thermophilic phase (55-65C target)
    ActiveComposting,
    /// Cooling and maturation (25-40C)
    Curing,
    /// Final screening and grading
    Screening,
    /// Finished, ready for use or sale
    Complete,
    /// Process failed (contamination, stall, etc.)
    Failed,
}

/// A phase transition record
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PhaseTransition {
    /// Previous status
    pub from: CompostBatchStatus,
    /// New status
    pub to: CompostBatchStatus,
    /// When the transition occurred (Unix microseconds)
    pub timestamp_us: u64,
    /// What triggered the transition
    pub trigger: String,
}

/// An input material contributing to the compost batch
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CompostInput {
    /// Link to waste stream entry
    pub stream_hash: ActionHash,
    /// Weight contributed (kg)
    pub quantity_kg: f64,
    /// Type of resource (maps to food-production ResourceType names)
    pub resource_type: String,
}

/// A composting batch — a tracked pile of organic material being processed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CompostBatch {
    /// Unique identifier
    pub id: String,
    /// Link to the waste facility managing this batch
    pub facility_hash: ActionHash,
    /// Composting method
    pub method: CompostMethod,
    /// Input materials
    pub inputs: Vec<CompostInput>,
    /// Estimated carbon-to-nitrogen ratio (target 25:1 - 30:1)
    pub carbon_nitrogen_ratio: f64,
    /// Current lifecycle status
    pub status: CompostBatchStatus,
    /// When the batch was created (Unix microseconds)
    pub started_at: u64,
    /// Target completion date (Unix microseconds)
    pub target_complete: u64,
    /// History of phase transitions
    pub phase_history: Vec<PhaseTransition>,
}

// ============================================================================
// COMPOST READING (sensor data)
// ============================================================================

/// A sensor reading from a compost batch
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CompostReading {
    /// The batch this reading is for
    pub batch_hash: ActionHash,
    /// Sensor identifier
    pub sensor_id: String,
    /// Core temperature in Celsius
    pub temperature_c: f64,
    /// Moisture content as percentage (0-100)
    pub moisture_pct: f64,
    /// Oxygen percentage (0-100)
    pub oxygen_pct: f64,
    /// pH level (0-14)
    pub ph: f64,
    /// When the reading was taken (Unix microseconds)
    pub timestamp_us: u64,
}

// ============================================================================
// COMPOST ACTION (recommended or executed)
// ============================================================================

/// Type of action to take on a compost batch
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CompostActionType {
    /// Turn/aerate the pile
    Turn,
    /// Add water to increase moisture
    AddWater,
    /// Add carbon-rich bulking agent (woodchips, straw)
    AddBulking,
    /// Adjust mechanical aeration rate
    AdjustAeration,
    /// Screen and harvest finished compost
    HarvestScreen,
}

/// Who or what recommended the action
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ActionRecommender {
    /// Automated threshold-based recommendation from sensor data
    Sensor,
    /// AI-driven recommendation (Symthaea)
    AI,
    /// Human operator decision
    Manual,
}

/// A recommended or executed action on a compost batch
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CompostAction {
    /// The batch this action is for
    pub batch_hash: ActionHash,
    /// Type of action
    pub action_type: CompostActionType,
    /// Who recommended it
    pub recommended_by: ActionRecommender,
    /// Who executed it (None if not yet executed)
    pub executed_by: Option<AgentPubKey>,
    /// When executed (None if pending)
    pub executed_at: Option<u64>,
    /// Additional notes
    pub notes: String,
}

// ============================================================================
// THRESHOLD CONSTANTS (from composting science literature)
// ============================================================================

/// Thermophilic phase minimum temperature (C) — EPA 40 CFR 503
pub const THERMOPHILIC_MIN_C: f64 = 55.0;
/// Thermophilic phase maximum temperature (C) — above this, beneficial microbes die
pub const THERMOPHILIC_MAX_C: f64 = 65.0;
/// Mesophilic phase temperature range (C) — curing phase
pub const MESOPHILIC_MIN_C: f64 = 25.0;
pub const MESOPHILIC_MAX_C: f64 = 40.0;
/// Curing complete threshold (C)
pub const CURING_COMPLETE_C: f64 = 25.0;
/// Target moisture range (%) — USCC guidelines
pub const MOISTURE_MIN_PCT: f64 = 40.0;
pub const MOISTURE_MAX_PCT: f64 = 65.0;
/// Optimal moisture for maximum microbial activity
pub const MOISTURE_OPTIMAL_PCT: f64 = 55.0;
/// Minimum oxygen for aerobic decomposition (%)
pub const OXYGEN_MIN_PCT: f64 = 5.0;
/// Optimal C:N ratio range — Cornell Waste Management Institute
pub const CN_RATIO_MIN: f64 = 25.0;
pub const CN_RATIO_MAX: f64 = 30.0;
/// pH range for healthy composting
pub const PH_MIN: f64 = 5.5;
pub const PH_MAX: f64 = 8.5;

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CompostBatch(CompostBatch),
    CompostReading(CompostReading),
    CompostAction(CompostAction),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All compost batches (global anchor)
    AllBatches,
    /// Facility to its compost batches
    FacilityToBatches,
    /// Batch to its sensor readings
    BatchToReadings,
    /// Batch to its actions
    BatchToActions,
    /// Status anchor to batches with that status
    StatusToBatches,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CompostBatch(batch) => validate_create_batch(action, batch),
                EntryTypes::CompostReading(reading) => validate_create_reading(action, reading),
                EntryTypes::CompostAction(action_entry) => {
                    validate_create_action(action, action_entry)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CompostBatch(_) | EntryTypes::CompostAction(_) => {
                    let original = must_get_action(original_action_hash)?;
                    Ok(check_author_match(
                        original.action().author(),
                        &action.author,
                        "update",
                    ))
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            match link_type {
                LinkTypes::AllBatches
                | LinkTypes::FacilityToBatches
                | LinkTypes::BatchToReadings
                | LinkTypes::BatchToActions
                | LinkTypes::StatusToBatches => {
                    if tag.0.len() > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            format!("{:?} link tag too long (max 512 bytes)", link_type),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original_action.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_batch(
    _action: Create,
    batch: CompostBatch,
) -> ExternResult<ValidateCallbackResult> {
    if batch.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Batch ID cannot be empty".into(),
        ));
    }
    if batch.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Batch ID too long (max 256 chars)".into(),
        ));
    }
    if batch.inputs.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Compost batch must have at least one input".into(),
        ));
    }
    if !batch.carbon_nitrogen_ratio.is_finite() || batch.carbon_nitrogen_ratio <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "C:N ratio must be a positive finite number".into(),
        ));
    }
    // Sanity check C:N ratio (extremes are suspicious)
    if batch.carbon_nitrogen_ratio > 500.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "C:N ratio exceeds maximum (500)".into(),
        ));
    }
    for input in &batch.inputs {
        if !input.quantity_kg.is_finite() || input.quantity_kg <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Input quantity must be a positive finite number".into(),
            ));
        }
    }
    if batch.started_at >= batch.target_complete {
        return Ok(ValidateCallbackResult::Invalid(
            "Target complete must be after start time".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_reading(
    _action: Create,
    reading: CompostReading,
) -> ExternResult<ValidateCallbackResult> {
    if !reading.temperature_c.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Temperature must be a finite number".into(),
        ));
    }
    // Temperature sanity: -20C (frozen outdoor) to 90C (very hot pile)
    if reading.temperature_c < -20.0 || reading.temperature_c > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Temperature must be between -20 and 90 C".into(),
        ));
    }
    if !reading.moisture_pct.is_finite()
        || reading.moisture_pct < 0.0
        || reading.moisture_pct > 100.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Moisture must be between 0 and 100 percent".into(),
        ));
    }
    if !reading.oxygen_pct.is_finite()
        || reading.oxygen_pct < 0.0
        || reading.oxygen_pct > 100.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Oxygen must be between 0 and 100 percent".into(),
        ));
    }
    if !reading.ph.is_finite() || reading.ph < 0.0 || reading.ph > 14.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "pH must be between 0 and 14".into(),
        ));
    }
    if reading.sensor_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Sensor ID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_action(
    _action: Create,
    compost_action: CompostAction,
) -> ExternResult<ValidateCallbackResult> {
    if compost_action.notes.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Action notes too long (max 2048 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_constants_valid() {
        assert!(THERMOPHILIC_MIN_C < THERMOPHILIC_MAX_C);
        assert!(MESOPHILIC_MIN_C < MESOPHILIC_MAX_C);
        assert!(MOISTURE_MIN_PCT < MOISTURE_MAX_PCT);
        assert!(MOISTURE_MIN_PCT < MOISTURE_OPTIMAL_PCT);
        assert!(MOISTURE_OPTIMAL_PCT < MOISTURE_MAX_PCT);
        assert!(OXYGEN_MIN_PCT > 0.0);
        assert!(CN_RATIO_MIN < CN_RATIO_MAX);
        assert!(PH_MIN < PH_MAX);
    }

    #[test]
    fn test_thermophilic_range() {
        assert!(THERMOPHILIC_MIN_C >= 55.0, "EPA 40 CFR 503 requires >= 55C");
        assert!(THERMOPHILIC_MAX_C <= 70.0, "Above ~65-70C kills beneficial microbes");
    }

    #[test]
    fn test_optimal_cn_ratio() {
        assert!(CN_RATIO_MIN >= 20.0, "Below 20:1 causes ammonia loss");
        assert!(CN_RATIO_MAX <= 35.0, "Above 35:1 slows decomposition");
    }

    #[test]
    fn test_compost_methods_exhaustive() {
        let methods = vec![
            CompostMethod::Windrow,
            CompostMethod::StaticAerated,
            CompostMethod::InVessel,
            CompostMethod::Vermicompost,
            CompostMethod::Bokashi,
        ];
        assert_eq!(methods.len(), 5);
    }

    #[test]
    fn test_batch_status_lifecycle() {
        let statuses = vec![
            CompostBatchStatus::Mixing,
            CompostBatchStatus::ActiveComposting,
            CompostBatchStatus::Curing,
            CompostBatchStatus::Screening,
            CompostBatchStatus::Complete,
            CompostBatchStatus::Failed,
        ];
        assert_eq!(statuses.len(), 6);
    }

    #[test]
    fn test_action_types_exhaustive() {
        let types = vec![
            CompostActionType::Turn,
            CompostActionType::AddWater,
            CompostActionType::AddBulking,
            CompostActionType::AdjustAeration,
            CompostActionType::HarvestScreen,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_recommender_types_exhaustive() {
        let types = vec![
            ActionRecommender::Sensor,
            ActionRecommender::AI,
            ActionRecommender::Manual,
        ];
        assert_eq!(types.len(), 3);
    }

    fn make_valid_batch() -> CompostBatch {
        CompostBatch {
            id: "CB-001".to_string(),
            facility_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            method: CompostMethod::Windrow,
            inputs: vec![CompostInput {
                stream_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                quantity_kg: 100.0,
                resource_type: "KitchenWaste".to_string(),
            }],
            carbon_nitrogen_ratio: 28.0,
            status: CompostBatchStatus::Mixing,
            started_at: 1711100000_000000,
            target_complete: 1714100000_000000,
            phase_history: vec![],
        }
    }

    fn make_valid_reading() -> CompostReading {
        CompostReading {
            batch_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            sensor_id: "SENSOR-001".to_string(),
            temperature_c: 58.0,
            moisture_pct: 55.0,
            oxygen_pct: 12.0,
            ph: 7.0,
            timestamp_us: 1711200000_000000,
        }
    }

    #[test]
    fn test_valid_batch() {
        let batch = make_valid_batch();
        assert!(!batch.id.is_empty());
        assert!(!batch.inputs.is_empty());
        assert!(batch.carbon_nitrogen_ratio > 0.0);
        assert!(batch.started_at < batch.target_complete);
    }

    #[test]
    fn test_batch_empty_inputs_invalid() {
        let mut batch = make_valid_batch();
        batch.inputs = vec![];
        assert!(batch.inputs.is_empty());
    }

    #[test]
    fn test_batch_cn_ratio_must_be_positive() {
        let mut batch = make_valid_batch();
        batch.carbon_nitrogen_ratio = 0.0;
        assert!(batch.carbon_nitrogen_ratio <= 0.0);
    }

    #[test]
    fn test_reading_temperature_bounds() {
        let reading = make_valid_reading();
        assert!(reading.temperature_c >= -20.0 && reading.temperature_c <= 90.0);
    }

    #[test]
    fn test_reading_moisture_bounds() {
        let reading = make_valid_reading();
        assert!(reading.moisture_pct >= 0.0 && reading.moisture_pct <= 100.0);
    }

    #[test]
    fn test_reading_oxygen_bounds() {
        let reading = make_valid_reading();
        assert!(reading.oxygen_pct >= 0.0 && reading.oxygen_pct <= 100.0);
    }

    #[test]
    fn test_reading_ph_bounds() {
        let reading = make_valid_reading();
        assert!(reading.ph >= 0.0 && reading.ph <= 14.0);
    }

    #[test]
    fn test_nan_temperature_rejected() {
        let mut reading = make_valid_reading();
        reading.temperature_c = f64::NAN;
        assert!(!reading.temperature_c.is_finite());
    }

    #[test]
    fn test_inf_moisture_rejected() {
        let mut reading = make_valid_reading();
        reading.moisture_pct = f64::INFINITY;
        assert!(!reading.moisture_pct.is_finite());
    }

    #[test]
    fn test_phase_transition_record() {
        let transition = PhaseTransition {
            from: CompostBatchStatus::Mixing,
            to: CompostBatchStatus::ActiveComposting,
            timestamp_us: 1711200000_000000,
            trigger: "Temperature reached 55C".to_string(),
        };
        assert_ne!(transition.from, transition.to);
    }
}
