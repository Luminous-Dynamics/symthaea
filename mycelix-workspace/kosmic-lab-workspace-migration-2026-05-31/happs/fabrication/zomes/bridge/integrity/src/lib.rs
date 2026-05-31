// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bridge Integrity Zome
//!
//! Defines entry types for cross-hApp integration including:
//! - Anticipatory Repair Loop (Property → Knowledge → Fabrication)
//! - Marketplace integration for design trading
//! - Supply Chain integration for material sourcing

use fabrication_common::*;
use hdi::prelude::*;

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

#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::RepairPrediction(p) => {
                    if p.prediction.failure_probability < 0.0
                        || p.prediction.failure_probability > 1.0
                    {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Invalid probability".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
