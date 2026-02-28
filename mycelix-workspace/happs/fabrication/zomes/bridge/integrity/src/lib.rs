//! Bridge Integrity Zome
//!
//! Defines entry types for cross-hApp integration including:
//! - Anticipatory Repair Loop (Property -> Knowledge -> Fabrication)
//! - Marketplace integration for design trading
//! - Supply Chain integration for material sourcing

use hdi::prelude::*;
use fabrication_common::*;
use fabrication_common::validation;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    RepairPrediction(RepairPredictionEntry),
    #[entry_type(visibility = "public")]
    RepairWorkflow(RepairWorkflowEntry),
    #[entry_type(visibility = "public")]
    FabricationQuery(FabricationQueryEntry),
    #[entry_type(visibility = "public")]
    FabricationEvent(FabricationEventEntry),
    #[entry_type(visibility = "public")]
    MarketplaceListing(MarketplaceListingEntry),
    #[entry_type(visibility = "public")]
    SupplyChainLink(SupplyChainLinkEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    AssetToPredictions,
    PredictionToWorkflow,
    DesignToListings,
    MaterialToSuppliers,
    RecentEvents,
    ActiveWorkflows,
    RateLimitBucket,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RepairPredictionEntry {
    pub prediction: RepairPrediction,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RepairWorkflowEntry {
    pub prediction_hash: ActionHash,
    pub status: RepairWorkflowStatus,
    pub design_hash: Option<ActionHash>,
    pub printer_hash: Option<ActionHash>,
    pub hearth_funding_hash: Option<ActionHash>,
    pub print_job_hash: Option<ActionHash>,
    pub property_installation_hash: Option<ActionHash>,
    pub created_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FabricationQueryEntry {
    pub query_type: FabQueryType,
    pub source_happ: String,
    pub parameters: String,
    pub queried_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FabricationEventEntry {
    pub event_type: FabEventType,
    pub design_id: Option<ActionHash>,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MarketplaceListingEntry {
    pub design_hash: ActionHash,
    pub marketplace_listing_hash: Option<ActionHash>,
    pub price: Option<u64>,
    pub listing_type: ListingType,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SupplyChainLinkEntry {
    pub material_hash: ActionHash,
    pub supplychain_item_hash: Option<ActionHash>,
    pub supplier_did: String,
    pub created_at: Timestamp,
}

// =============================================================================
// PER-ENTRY VALIDATION
// =============================================================================

fn validate_repair_prediction(p: &RepairPredictionEntry) -> ExternResult<ValidateCallbackResult> {
    check!(validation::require_in_range(
        p.prediction.failure_probability,
        0.0,
        1.0,
        "failure_probability"
    ));
    check!(validation::require_non_empty(
        &p.prediction.asset_model,
        "asset_model"
    ));
    check!(validation::require_max_len(
        &p.prediction.asset_model,
        256,
        "asset_model"
    ));
    check!(validation::require_non_empty(
        &p.prediction.predicted_failure_component,
        "predicted_failure_component"
    ));
    check!(validation::require_max_len(
        &p.prediction.predicted_failure_component,
        256,
        "predicted_failure_component"
    ));
    check!(validation::require_max_len(
        &p.prediction.sensor_data_summary,
        4096,
        "sensor_data_summary"
    ));
    if p.prediction.confidence_interval_days > 3650 {
        return Ok(ValidateCallbackResult::Invalid(
            "confidence_interval_days cannot exceed 3650".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_repair_workflow(_w: &RepairWorkflowEntry) -> ExternResult<ValidateCallbackResult> {
    // Status enum is self-validating via serde deserialization
    Ok(ValidateCallbackResult::Valid)
}

fn validate_fabrication_query(q: &FabricationQueryEntry) -> ExternResult<ValidateCallbackResult> {
    check!(validation::require_non_empty(
        &q.source_happ,
        "source_happ"
    ));
    check!(validation::require_max_len(
        &q.source_happ,
        256,
        "source_happ"
    ));
    check!(validation::require_max_len(
        &q.parameters,
        32768,
        "parameters"
    ));
    Ok(ValidateCallbackResult::Valid)
}

fn validate_fabrication_event(e: &FabricationEventEntry) -> ExternResult<ValidateCallbackResult> {
    check!(validation::require_max_len(&e.payload, 32768, "payload"));
    check!(validation::require_non_empty(
        &e.source_happ,
        "source_happ"
    ));
    check!(validation::require_max_len(
        &e.source_happ,
        256,
        "source_happ"
    ));
    Ok(ValidateCallbackResult::Valid)
}

fn validate_marketplace_listing(
    ml: &MarketplaceListingEntry,
) -> ExternResult<ValidateCallbackResult> {
    if let Some(price) = ml.price {
        if price > 1_000_000_000 {
            return Ok(ValidateCallbackResult::Invalid(
                "price cannot exceed 1000000000".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_supply_chain_link(s: &SupplyChainLinkEntry) -> ExternResult<ValidateCallbackResult> {
    check!(validation::require_non_empty(
        &s.supplier_did,
        "supplier_did"
    ));
    check!(validation::require_max_len(
        &s.supplier_did,
        256,
        "supplier_did"
    ));
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// VALIDATION DISPATCH
// =============================================================================

fn validate_entry(app_entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match app_entry {
        EntryTypes::RepairPrediction(p) => validate_repair_prediction(&p),
        EntryTypes::RepairWorkflow(w) => validate_repair_workflow(&w),
        EntryTypes::FabricationQuery(q) => validate_fabrication_query(&q),
        EntryTypes::FabricationEvent(e) => validate_fabrication_event(&e),
        EntryTypes::MarketplaceListing(ml) => validate_marketplace_listing(&ml),
        EntryTypes::SupplyChainLink(s) => validate_supply_chain_link(&s),
    }
}

#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn make_timestamp() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn valid_prediction() -> RepairPredictionEntry {
        RepairPredictionEntry {
            prediction: RepairPrediction {
                property_asset_hash: make_action_hash(),
                asset_model: "Model-X100".to_string(),
                predicted_failure_component: "bearing-assembly".to_string(),
                failure_probability: 0.75,
                estimated_failure_date: make_timestamp(),
                confidence_interval_days: 30,
                sensor_data_summary: "Vibration trending up".to_string(),
                recommended_action: RepairAction::PrintReplacement,
                created_at: make_timestamp(),
            },
        }
    }

    #[test]
    fn valid_prediction_passes() {
        let p = valid_prediction();
        let result = validate_repair_prediction(&p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn nan_failure_probability_rejected() {
        let mut p = valid_prediction();
        p.prediction.failure_probability = f32::NAN;
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn probability_above_one_rejected() {
        let mut p = valid_prediction();
        p.prediction.failure_probability = 1.01;
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn empty_asset_model_rejected() {
        let mut p = valid_prediction();
        p.prediction.asset_model = "".to_string();
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn sensor_summary_too_long_rejected() {
        let mut p = valid_prediction();
        p.prediction.sensor_data_summary = "x".repeat(4097);
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn confidence_interval_too_large_rejected() {
        let mut p = valid_prediction();
        p.prediction.confidence_interval_days = 3651;
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn valid_workflow_passes() {
        let w = RepairWorkflowEntry {
            prediction_hash: make_action_hash(),
            status: RepairWorkflowStatus::Predicted,
            design_hash: None,
            printer_hash: None,
            hearth_funding_hash: None,
            print_job_hash: None,
            property_installation_hash: None,
            created_at: make_timestamp(),
            completed_at: None,
        };
        let result = validate_repair_workflow(&w).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn query_source_happ_empty_rejected() {
        let q = FabricationQueryEntry {
            query_type: FabQueryType::GetDesign,
            source_happ: "".to_string(),
            parameters: "{}".to_string(),
            queried_at: make_timestamp(),
        };
        let result = validate_fabrication_query(&q).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn query_params_too_long_rejected() {
        let q = FabricationQueryEntry {
            query_type: FabQueryType::GetDesign,
            source_happ: "property".to_string(),
            parameters: "x".repeat(32769),
            queried_at: make_timestamp(),
        };
        let result = validate_fabrication_query(&q).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn event_payload_too_long_rejected() {
        let e = FabricationEventEntry {
            event_type: FabEventType::DesignPublished,
            design_id: None,
            payload: "x".repeat(32769),
            source_happ: "fabrication".to_string(),
            timestamp: make_timestamp(),
        };
        let result = validate_fabrication_event(&e).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn valid_listing_passes() {
        let ml = MarketplaceListingEntry {
            design_hash: make_action_hash(),
            marketplace_listing_hash: None,
            price: Some(500),
            listing_type: ListingType::DesignSale,
            created_at: make_timestamp(),
        };
        let result = validate_marketplace_listing(&ml).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn listing_price_too_high_rejected() {
        let ml = MarketplaceListingEntry {
            design_hash: make_action_hash(),
            marketplace_listing_hash: None,
            price: Some(1_000_000_001),
            listing_type: ListingType::DesignSale,
            created_at: make_timestamp(),
        };
        let result = validate_marketplace_listing(&ml).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn supplier_did_empty_rejected() {
        let s = SupplyChainLinkEntry {
            material_hash: make_action_hash(),
            supplychain_item_hash: None,
            supplier_did: "".to_string(),
            created_at: make_timestamp(),
        };
        let result = validate_supply_chain_link(&s).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn supplier_did_too_long_rejected() {
        let s = SupplyChainLinkEntry {
            material_hash: make_action_hash(),
            supplychain_item_hash: None,
            supplier_did: "x".repeat(257),
            created_at: make_timestamp(),
        };
        let result = validate_supply_chain_link(&s).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // Edge case: infinity, whitespace-only, boundary precision
    // =========================================================================

    #[test]
    fn infinity_failure_probability_rejected() {
        let mut p = valid_prediction();
        p.prediction.failure_probability = f32::INFINITY;
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn neg_infinity_failure_probability_rejected() {
        let mut p = valid_prediction();
        p.prediction.failure_probability = f32::NEG_INFINITY;
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn failure_probability_at_zero_passes() {
        let mut p = valid_prediction();
        p.prediction.failure_probability = 0.0;
        let result = validate_repair_prediction(&p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn failure_probability_at_one_passes() {
        let mut p = valid_prediction();
        p.prediction.failure_probability = 1.0;
        let result = validate_repair_prediction(&p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn whitespace_only_asset_model_rejected() {
        let mut p = valid_prediction();
        p.prediction.asset_model = "  \t\n  ".to_string();
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("asset_model")));
    }

    #[test]
    fn sensor_summary_at_max_length_passes() {
        let mut p = valid_prediction();
        p.prediction.sensor_data_summary = "x".repeat(4096);
        let result = validate_repair_prediction(&p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn listing_price_zero_passes() {
        let ml = MarketplaceListingEntry {
            design_hash: make_action_hash(),
            marketplace_listing_hash: None,
            price: Some(0),
            listing_type: ListingType::DesignSale,
            created_at: make_timestamp(),
        };
        let result = validate_marketplace_listing(&ml).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn whitespace_only_supplier_did_rejected() {
        let s = SupplyChainLinkEntry {
            material_hash: make_action_hash(),
            supplychain_item_hash: None,
            supplier_did: "   ".to_string(),
            created_at: make_timestamp(),
        };
        let result = validate_supply_chain_link(&s).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn confidence_interval_at_max_passes() {
        let mut p = valid_prediction();
        p.prediction.confidence_interval_days = 3650;
        let result = validate_repair_prediction(&p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn confidence_interval_over_max_rejected() {
        let mut p = valid_prediction();
        p.prediction.confidence_interval_days = 3651;
        let result = validate_repair_prediction(&p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
