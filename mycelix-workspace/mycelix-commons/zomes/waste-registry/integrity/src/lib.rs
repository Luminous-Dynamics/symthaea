// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waste Registry Integrity Zome
//! Entry types, link types, and validation for waste stream tracking and facility matching

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// WASTE STREAM
// ============================================================================

/// Category of waste material
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WasteCategory {
    /// Food scraps, yard waste, wood, paper
    Organic,
    /// Plastics, metals, glass, cardboard
    Recyclable,
    /// Toxic chemicals, batteries, medical waste
    Hazardous,
    /// Computers, phones, circuit boards
    Electronic,
    /// Unsorted or multi-material waste
    Mixed,
    /// Construction debris, ceramics, non-reactive
    Inert,
}

/// Contamination level of a waste stream
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub enum ContaminationLevel {
    /// < 0.5% contaminant by weight
    Clean,
    /// 0.5–3% contaminant
    LightlyContaminated,
    /// 3–10% contaminant
    Contaminated,
    /// > 10% or toxic contaminants present
    Hazardous,
}

impl ContaminationLevel {
    /// Numeric ordering for comparison (higher = worse)
    pub fn severity(&self) -> u8 {
        match self {
            Self::Clean => 0,
            Self::LightlyContaminated => 1,
            Self::Contaminated => 2,
            Self::Hazardous => 3,
        }
    }
}

/// End-of-life strategy (mirrors fabrication_common::EndOfLifeStrategy)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EndOfLifeStrategy {
    MechanicalRecycling,
    ChemicalRecycling,
    Biodegradable,
    IndustrialCompost,
    Downcycle,
    Landfill,
}

/// Lifecycle status of a waste stream
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WasteStreamStatus {
    /// Newly registered, awaiting classification
    Registered,
    /// Classified by human or AI
    Classified,
    /// Matched to a facility
    Routed,
    /// Picked up by collection vehicle
    Collected,
    /// Being processed at facility
    Processing,
    /// Fully processed (recycled, composted, etc.)
    Completed,
}

/// Method used to classify waste
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClassificationMethod {
    /// Human visual inspection
    Manual,
    /// IoT sensor data (weight, spectral, etc.)
    SensorBased,
    /// Symthaea HDC vision classification
    VisionAI,
    /// Read from fabrication MaterialPassport
    MaterialPassport,
}

/// A waste stream entry — a batch of waste material tracked through its lifecycle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WasteStream {
    /// Unique identifier
    pub id: String,
    /// DID of the source (household, business, farm)
    pub source_did: String,
    /// Optional link to fabrication MaterialPassport
    pub material_passport_hash: Option<ActionHash>,
    /// Primary waste category
    pub waste_category: WasteCategory,
    /// More specific description (e.g. "PET bottles", "food scraps")
    pub subcategory: String,
    /// Weight in kilograms
    pub quantity_kg: f64,
    /// Contamination assessment
    pub contamination_level: ContaminationLevel,
    /// Recommended end-of-life strategy
    pub recommended_eol: EndOfLifeStrategy,
    /// GPS latitude
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// Current lifecycle status
    pub status: WasteStreamStatus,
    /// When the stream was registered (Unix microseconds)
    pub created_at: u64,
}

/// A classification result for a waste stream
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WasteClassification {
    /// The waste stream being classified
    pub stream_hash: ActionHash,
    /// DID of the classifier (human or AI agent)
    pub classifier_did: String,
    /// Determined category
    pub category: WasteCategory,
    /// Specific subcategory
    pub subcategory: String,
    /// Assessed contamination
    pub contamination_level: ContaminationLevel,
    /// Classification confidence (0.0–1.0)
    pub confidence: f32,
    /// How the classification was performed
    pub method: ClassificationMethod,
    /// When classified (Unix microseconds)
    pub classified_at: u64,
}

// ============================================================================
// WASTE FACILITY
// ============================================================================

/// Type of waste processing facility
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FacilityType {
    /// Materials Recovery Facility (mechanical sorting)
    MRF,
    /// Composting operation (windrow, in-vessel, etc.)
    Composter,
    /// Anaerobic digestion (biogas + digestate)
    AnaerobicDigester,
    /// Chemical recycling (depolymerization, pyrolysis)
    ChemicalRecycler,
    /// Sanitary landfill (last resort)
    Landfill,
    /// Transfer station (consolidation point)
    TransferStation,
}

/// Operating status of a facility
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FacilityStatus {
    Active,
    Maintenance,
    AtCapacity,
    Closed,
}

/// A waste processing facility
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WasteFacility {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Type of processing
    pub facility_type: FacilityType,
    /// Which waste categories this facility accepts
    pub accepts: Vec<WasteCategory>,
    /// Maximum throughput (kg per day)
    pub capacity_kg_per_day: f64,
    /// Current daily load (kg)
    pub current_load_kg: f64,
    /// GPS latitude
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// DID of the facility operator
    pub operator_did: String,
    /// Operating status
    pub status: FacilityStatus,
}

// ============================================================================
// WASTE ROUTE (result of matching)
// ============================================================================

/// Scoring factors for waste-to-facility matching
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WasteRouteFactors {
    /// Does the facility accept this waste category? (0.0 or 1.0)
    pub category_match: f32,
    /// Inverse distance score (closer = higher, 0.0–1.0)
    pub proximity_score: f32,
    /// Available capacity fraction (0.0–1.0)
    pub capacity_score: f32,
    /// Contamination compatibility (0.0–1.0)
    pub contamination_compatibility: f32,
}

impl WasteRouteFactors {
    /// Compute overall match score (weighted average)
    pub fn overall_score(&self) -> f32 {
        // Category match is a hard gate
        if self.category_match < 0.5 {
            return 0.0;
        }
        // Weighted: proximity 40%, capacity 35%, contamination 25%
        0.40 * self.proximity_score
            + 0.35 * self.capacity_score
            + 0.25 * self.contamination_compatibility
    }
}

/// A routing decision linking a waste stream to a facility
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WasteRoute {
    /// The waste stream being routed
    pub stream_hash: ActionHash,
    /// The matched facility
    pub facility_hash: ActionHash,
    /// Match quality factors
    pub factors: WasteRouteFactors,
    /// Overall match score
    pub overall_score: f32,
    /// When the route was computed (Unix microseconds)
    pub routed_at: u64,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    WasteStream(WasteStream),
    WasteClassification(WasteClassification),
    WasteFacility(WasteFacility),
    WasteRoute(WasteRoute),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All waste streams (global anchor)
    AllStreams,
    /// All facilities (global anchor)
    AllFacilities,
    /// Facility to streams it has been matched with
    FacilityToStreams,
    /// Agent to their registered streams
    AgentToStreams,
    /// Category anchor to streams of that category
    CategoryToStreams,
    /// Stream to its classifications
    StreamToClassification,
    /// Stream to its route
    StreamToRoute,
    /// Status anchor to streams with that status
    StatusToStreams,
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
                EntryTypes::WasteStream(stream) => validate_create_stream(action, stream),
                EntryTypes::WasteClassification(classification) => {
                    validate_create_classification(action, classification)
                }
                EntryTypes::WasteFacility(facility) => {
                    validate_create_facility(action, facility)
                }
                EntryTypes::WasteRoute(route) => validate_create_route(action, route),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WasteStream(_)
                | EntryTypes::WasteFacility(_)
                | EntryTypes::WasteRoute(_) => {
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
                LinkTypes::AllStreams
                | LinkTypes::AllFacilities
                | LinkTypes::AgentToStreams
                | LinkTypes::CategoryToStreams
                | LinkTypes::StatusToStreams
                | LinkTypes::StreamToClassification
                | LinkTypes::StreamToRoute
                | LinkTypes::FacilityToStreams => {
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

fn validate_create_stream(
    _action: Create,
    stream: WasteStream,
) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if stream.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Waste stream ID cannot be empty".into(),
        ));
    }
    if stream.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Waste stream ID too long (max 256 chars)".into(),
        ));
    }

    // Quantity must be positive and finite
    if !stream.quantity_kg.is_finite() || stream.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a positive finite number".into(),
        ));
    }
    if stream.quantity_kg > 1_000_000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity exceeds maximum (1,000,000 kg)".into(),
        ));
    }

    // Location must be valid GPS coordinates
    if !stream.location_lat.is_finite()
        || stream.location_lat < -90.0
        || stream.location_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !stream.location_lon.is_finite()
        || stream.location_lon < -180.0
        || stream.location_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }

    // Subcategory length limit
    if stream.subcategory.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Subcategory too long (max 256 chars)".into(),
        ));
    }

    // Source DID must not be empty
    if stream.source_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source DID cannot be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_classification(
    _action: Create,
    classification: WasteClassification,
) -> ExternResult<ValidateCallbackResult> {
    // Confidence must be in [0, 1]
    if !classification.confidence.is_finite()
        || classification.confidence < 0.0
        || classification.confidence > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Classification confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // Classifier DID must not be empty
    if classification.classifier_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Classifier DID cannot be empty".into(),
        ));
    }

    // Subcategory length limit
    if classification.subcategory.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Subcategory too long (max 256 chars)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_facility(
    _action: Create,
    facility: WasteFacility,
) -> ExternResult<ValidateCallbackResult> {
    // ID and name must not be empty
    if facility.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Facility ID cannot be empty".into(),
        ));
    }
    if facility.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Facility ID too long (max 256 chars)".into(),
        ));
    }
    if facility.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Facility name cannot be empty".into(),
        ));
    }
    if facility.name.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Facility name too long (max 512 chars)".into(),
        ));
    }

    // Capacity must be positive and finite
    if !facility.capacity_kg_per_day.is_finite() || facility.capacity_kg_per_day <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Capacity must be a positive finite number".into(),
        ));
    }

    // Current load must be non-negative and finite
    if !facility.current_load_kg.is_finite() || facility.current_load_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Current load must be a non-negative finite number".into(),
        ));
    }

    // Location must be valid GPS coordinates
    if !facility.location_lat.is_finite()
        || facility.location_lat < -90.0
        || facility.location_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !facility.location_lon.is_finite()
        || facility.location_lon < -180.0
        || facility.location_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }

    // Must accept at least one category
    if facility.accepts.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Facility must accept at least one waste category".into(),
        ));
    }

    // Operator DID must not be empty
    if facility.operator_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Operator DID cannot be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_route(
    _action: Create,
    route: WasteRoute,
) -> ExternResult<ValidateCallbackResult> {
    // Overall score must be in [0, 1]
    if !route.overall_score.is_finite() || route.overall_score < 0.0 || route.overall_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Overall score must be between 0.0 and 1.0".into(),
        ));
    }

    // Factor scores must be in [0, 1]
    let f = &route.factors;
    for (name, val) in [
        ("category_match", f.category_match),
        ("proximity_score", f.proximity_score),
        ("capacity_score", f.capacity_score),
        ("contamination_compatibility", f.contamination_compatibility),
    ] {
        if !val.is_finite() || val < 0.0 || val > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be between 0.0 and 1.0", name),
            ));
        }
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
    fn test_contamination_level_severity_ordering() {
        assert!(ContaminationLevel::Clean.severity() < ContaminationLevel::LightlyContaminated.severity());
        assert!(ContaminationLevel::LightlyContaminated.severity() < ContaminationLevel::Contaminated.severity());
        assert!(ContaminationLevel::Contaminated.severity() < ContaminationLevel::Hazardous.severity());
    }

    #[test]
    fn test_waste_route_factors_category_gate() {
        let factors = WasteRouteFactors {
            category_match: 0.0,
            proximity_score: 1.0,
            capacity_score: 1.0,
            contamination_compatibility: 1.0,
        };
        assert_eq!(factors.overall_score(), 0.0, "No category match should yield zero score");
    }

    #[test]
    fn test_waste_route_factors_perfect_match() {
        let factors = WasteRouteFactors {
            category_match: 1.0,
            proximity_score: 1.0,
            capacity_score: 1.0,
            contamination_compatibility: 1.0,
        };
        let score = factors.overall_score();
        assert!((score - 1.0).abs() < 0.001, "Perfect match should score ~1.0, got {}", score);
    }

    #[test]
    fn test_waste_route_factors_weighted_scoring() {
        let factors = WasteRouteFactors {
            category_match: 1.0,
            proximity_score: 0.5,
            capacity_score: 0.5,
            contamination_compatibility: 0.5,
        };
        let score = factors.overall_score();
        assert!((score - 0.5).abs() < 0.001, "Half scores should yield ~0.5, got {}", score);
    }

    #[test]
    fn test_waste_route_factors_proximity_weighted_highest() {
        let near = WasteRouteFactors {
            category_match: 1.0,
            proximity_score: 1.0,
            capacity_score: 0.5,
            contamination_compatibility: 0.5,
        };
        let far = WasteRouteFactors {
            category_match: 1.0,
            proximity_score: 0.0,
            capacity_score: 0.5,
            contamination_compatibility: 0.5,
        };
        assert!(near.overall_score() > far.overall_score());
    }

    fn make_valid_stream() -> WasteStream {
        WasteStream {
            id: "WS-001".to_string(),
            source_did: "did:example:household1".to_string(),
            material_passport_hash: None,
            waste_category: WasteCategory::Organic,
            subcategory: "food scraps".to_string(),
            quantity_kg: 25.0,
            contamination_level: ContaminationLevel::Clean,
            recommended_eol: EndOfLifeStrategy::IndustrialCompost,
            location_lat: 32.9483,
            location_lon: -96.7299,
            status: WasteStreamStatus::Registered,
            created_at: 1711100000_000000,
        }
    }

    fn make_valid_facility() -> WasteFacility {
        WasteFacility {
            id: "FAC-001".to_string(),
            name: "Community Composting Hub".to_string(),
            facility_type: FacilityType::Composter,
            accepts: vec![WasteCategory::Organic],
            capacity_kg_per_day: 500.0,
            current_load_kg: 200.0,
            location_lat: 32.9500,
            location_lon: -96.7300,
            operator_did: "did:example:operator1".to_string(),
            status: FacilityStatus::Active,
        }
    }

    fn make_valid_classification() -> WasteClassification {
        WasteClassification {
            stream_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            classifier_did: "did:example:ai_classifier".to_string(),
            category: WasteCategory::Organic,
            subcategory: "food scraps".to_string(),
            contamination_level: ContaminationLevel::Clean,
            confidence: 0.92,
            method: ClassificationMethod::VisionAI,
            classified_at: 1711100001_000000,
        }
    }

    fn make_valid_route() -> WasteRoute {
        WasteRoute {
            stream_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            facility_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            factors: WasteRouteFactors {
                category_match: 1.0,
                proximity_score: 0.85,
                capacity_score: 0.60,
                contamination_compatibility: 1.0,
            },
            overall_score: 0.79,
            routed_at: 1711100002_000000,
        }
    }

    #[test]
    fn test_valid_stream_passes_validation() {
        let stream = make_valid_stream();
        assert!(!stream.id.is_empty());
        assert!(stream.quantity_kg > 0.0);
        assert!((-90.0..=90.0).contains(&stream.location_lat));
        assert!((-180.0..=180.0).contains(&stream.location_lon));
    }

    #[test]
    fn test_stream_quantity_must_be_positive() {
        let mut stream = make_valid_stream();
        stream.quantity_kg = 0.0;
        assert!(stream.quantity_kg <= 0.0, "Zero quantity should fail validation");

        stream.quantity_kg = -1.0;
        assert!(stream.quantity_kg <= 0.0, "Negative quantity should fail validation");
    }

    #[test]
    fn test_stream_quantity_max_limit() {
        let mut stream = make_valid_stream();
        stream.quantity_kg = 1_000_001.0;
        assert!(stream.quantity_kg > 1_000_000.0, "Exceeds max");
    }

    #[test]
    fn test_stream_invalid_latitude() {
        let mut stream = make_valid_stream();
        stream.location_lat = 91.0;
        assert!(stream.location_lat > 90.0, "Invalid latitude");
    }

    #[test]
    fn test_facility_must_accept_categories() {
        let mut facility = make_valid_facility();
        facility.accepts = vec![];
        assert!(facility.accepts.is_empty(), "Empty accepts should fail");
    }

    #[test]
    fn test_facility_capacity_must_be_positive() {
        let mut facility = make_valid_facility();
        facility.capacity_kg_per_day = 0.0;
        assert!(facility.capacity_kg_per_day <= 0.0, "Zero capacity should fail");
    }

    #[test]
    fn test_classification_confidence_bounds() {
        let mut c = make_valid_classification();
        c.confidence = 0.0;
        assert!(c.confidence >= 0.0 && c.confidence <= 1.0);

        c.confidence = 1.0;
        assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
    }

    #[test]
    fn test_classification_confidence_out_of_bounds() {
        let mut c = make_valid_classification();
        c.confidence = 1.1;
        assert!(c.confidence > 1.0, "Confidence > 1.0 should fail");

        c.confidence = -0.1;
        assert!(c.confidence < 0.0, "Confidence < 0.0 should fail");
    }

    #[test]
    fn test_route_score_bounds() {
        let route = make_valid_route();
        assert!(route.overall_score >= 0.0 && route.overall_score <= 1.0);
        assert!(route.factors.category_match >= 0.0 && route.factors.category_match <= 1.0);
        assert!(route.factors.proximity_score >= 0.0 && route.factors.proximity_score <= 1.0);
        assert!(route.factors.capacity_score >= 0.0 && route.factors.capacity_score <= 1.0);
        assert!(route.factors.contamination_compatibility >= 0.0 && route.factors.contamination_compatibility <= 1.0);
    }

    #[test]
    fn test_nan_latitude_rejected() {
        let mut stream = make_valid_stream();
        stream.location_lat = f64::NAN;
        assert!(!stream.location_lat.is_finite(), "NaN should be rejected");
    }

    #[test]
    fn test_nan_quantity_rejected() {
        let mut stream = make_valid_stream();
        stream.quantity_kg = f64::NAN;
        assert!(!stream.quantity_kg.is_finite(), "NaN should be rejected");
    }

    #[test]
    fn test_inf_capacity_rejected() {
        let mut facility = make_valid_facility();
        facility.capacity_kg_per_day = f64::INFINITY;
        assert!(!facility.capacity_kg_per_day.is_finite(), "Infinity should be rejected");
    }

    #[test]
    fn test_all_waste_categories_exhaustive() {
        let categories = vec![
            WasteCategory::Organic,
            WasteCategory::Recyclable,
            WasteCategory::Hazardous,
            WasteCategory::Electronic,
            WasteCategory::Mixed,
            WasteCategory::Inert,
        ];
        assert_eq!(categories.len(), 6);
    }

    #[test]
    fn test_all_facility_types_exhaustive() {
        let types = vec![
            FacilityType::MRF,
            FacilityType::Composter,
            FacilityType::AnaerobicDigester,
            FacilityType::ChemicalRecycler,
            FacilityType::Landfill,
            FacilityType::TransferStation,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_eol_strategy_exhaustive() {
        let strategies = vec![
            EndOfLifeStrategy::MechanicalRecycling,
            EndOfLifeStrategy::ChemicalRecycling,
            EndOfLifeStrategy::Biodegradable,
            EndOfLifeStrategy::IndustrialCompost,
            EndOfLifeStrategy::Downcycle,
            EndOfLifeStrategy::Landfill,
        ];
        assert_eq!(strategies.len(), 6);
    }
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_contamination_level() -> impl Strategy<Value = ContaminationLevel> {
        prop_oneof![
            Just(ContaminationLevel::Clean),
            Just(ContaminationLevel::LightlyContaminated),
            Just(ContaminationLevel::Contaminated),
            Just(ContaminationLevel::Hazardous),
        ]
    }

    fn arb_route_factors() -> impl Strategy<Value = WasteRouteFactors> {
        (0.0..=1.0_f32, 0.0..=1.0_f32, 0.0..=1.0_f32, 0.0..=1.0_f32).prop_map(
            |(cat, prox, cap, contam)| WasteRouteFactors {
                category_match: cat,
                proximity_score: prox,
                capacity_score: cap,
                contamination_compatibility: contam,
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Route score is always in [0.0, 1.0].
        #[test]
        fn route_score_bounded(factors in arb_route_factors()) {
            let score = factors.overall_score();
            prop_assert!(score >= 0.0, "Score must be non-negative: {}", score);
            prop_assert!(score <= 1.0, "Score must not exceed 1.0: {}", score);
        }

        /// Contamination severity is strictly monotonic.
        #[test]
        fn contamination_severity_monotonic(
            a in arb_contamination_level(),
            b in arb_contamination_level(),
        ) {
            if a.severity() < b.severity() {
                prop_assert!(a.severity() < b.severity());
            }
            // Reflexive: same level → same severity
            prop_assert_eq!(a.severity(), a.severity());
        }

        /// Higher contamination_compatibility always yields >= score
        /// (with all other factors fixed and category_match = 1.0).
        #[test]
        fn higher_contamination_compat_higher_score(
            prox in 0.0..=1.0_f32,
            cap in 0.0..=1.0_f32,
            low_contam in 0.0..=0.5_f32,
            high_contam in 0.5..=1.0_f32,
        ) {
            let low = WasteRouteFactors {
                category_match: 1.0,
                proximity_score: prox,
                capacity_score: cap,
                contamination_compatibility: low_contam,
            };
            let high = WasteRouteFactors {
                category_match: 1.0,
                proximity_score: prox,
                capacity_score: cap,
                contamination_compatibility: high_contam,
            };
            prop_assert!(
                high.overall_score() >= low.overall_score(),
                "Higher contamination compat should yield >= score: {} vs {}",
                high.overall_score(), low.overall_score()
            );
        }

        /// No category match always yields zero score regardless of other factors.
        #[test]
        fn no_category_match_zero_score(
            prox in 0.0..=1.0_f32,
            cap in 0.0..=1.0_f32,
            contam in 0.0..=1.0_f32,
        ) {
            let factors = WasteRouteFactors {
                category_match: 0.0,
                proximity_score: prox,
                capacity_score: cap,
                contamination_compatibility: contam,
            };
            prop_assert_eq!(factors.overall_score(), 0.0);
        }

        /// GPS coordinates must be within valid ranges.
        #[test]
        fn gps_bounds_validation(
            lat in -90.0..=90.0_f64,
            lon in -180.0..=180.0_f64,
        ) {
            prop_assert!(lat >= -90.0 && lat <= 90.0);
            prop_assert!(lon >= -180.0 && lon <= 180.0);
        }

        /// Quantity is always positive when valid.
        #[test]
        fn quantity_positive(qty in 0.001..=1_000_000.0_f64) {
            prop_assert!(qty > 0.0);
            prop_assert!(qty.is_finite());
        }
    }
}
