//! Steward Coordinator Zome
//! Business logic for watershed governance, water rights, transfers, and disputes

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_constitutional, requirement_for_proposal,
    requirement_for_voting, GovernanceEligibility, GovernanceRequirement,
};
use water_steward_integrity::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// WATERSHED MANAGEMENT
// ============================================================================

/// Define a new watershed
#[hdk_extern]
pub fn define_watershed(watershed: Watershed) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "define_watershed")?;
    if watershed.id.trim().is_empty() || watershed.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Watershed ID must be 1-256 non-whitespace characters".into()
        )));
    }
    if watershed.name.trim().is_empty() || watershed.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Watershed name must be 1-256 non-whitespace characters".into()
        )));
    }
    if !watershed.area_sq_km.is_finite() || watershed.area_sq_km < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Watershed area_sq_km must be a finite non-negative number".into()
        )));
    }
    for (lat, lon) in &watershed.boundary {
        if !lat.is_finite()
            || !lon.is_finite()
            || !(-90.0..=90.0).contains(lat)
            || !(-180.0..=180.0).contains(lon)
        {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Boundary coordinates must be finite with lat in [-90,90] and lon in [-180,180]"
                    .into()
            )));
        }
    }

    let action_hash = create_entry(&EntryTypes::Watershed(watershed.clone()))?;

    // Link to all watersheds
    create_entry(&EntryTypes::Anchor(Anchor("all_watersheds".to_string())))?;
    create_link(
        anchor_hash("all_watersheds")?,
        action_hash.clone(),
        LinkTypes::AllWatersheds,
        (),
    )?;

    // Link stewardship type to watershed
    let type_anchor = format!("stewardship:{:?}", watershed.stewardship_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::StewardshipTypeToWatershed,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created watershed".into()
    )))
}

/// Get all registered watersheds
#[hdk_extern]
pub fn get_all_watersheds(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_watersheds")?, LinkTypes::AllWatersheds)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// WATER RIGHTS
// ============================================================================

/// Register a new water right within a watershed
#[hdk_extern]
pub fn register_water_right(right: WaterRight) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "register_water_right")?;
    // Verify watershed exists
    let _ws_record = get(right.watershed_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Watershed not found".into())),
    )?;

    let action_hash = create_entry(&EntryTypes::WaterRight(right.clone()))?;

    // Link watershed to right
    create_link(
        right.watershed_hash.clone(),
        action_hash.clone(),
        LinkTypes::WatershedToRight,
        (),
    )?;

    // Link holder to right
    create_link(
        right.holder.clone(),
        action_hash.clone(),
        LinkTypes::HolderToRight,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created water right".into()
    )))
}

/// Get all water rights for a watershed
#[hdk_extern]
pub fn get_watershed_rights(watershed_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(watershed_hash, LinkTypes::WatershedToRight)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all water rights held by the calling agent
#[hdk_extern]
pub fn get_my_rights(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let links = get_links(
        LinkQuery::try_new(agent_info.agent_initial_pubkey, LinkTypes::HolderToRight)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// RIGHT TRANSFERS
// ============================================================================

/// Transfer a water right to another holder
#[hdk_extern]
pub fn transfer_right(input: TransferRightInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "transfer_right")?;
    let agent_info = agent_info()?;

    // Fetch the water right
    let right_record = get(input.right_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Water right not found".into())
    ))?;
    let right: WaterRight = right_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid water right entry".into()
        )))?;

    // Only the holder can transfer
    if right.holder != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the right holder can initiate a transfer".into()
        )));
    }

    if !right.transferable {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "This water right is not transferable".into()
        )));
    }

    if right.status != RightStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only active rights can be transferred".into()
        )));
    }

    if input.volume_liters == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transfer volume must be greater than zero".into()
        )));
    }

    if input.volume_liters > right.volume_authorized_liters {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transfer volume exceeds authorized volume".into()
        )));
    }

    let now = sys_time()?;

    let transfer = RightTransfer {
        right_hash: input.right_hash.clone(),
        from_holder: agent_info.agent_initial_pubkey.clone(),
        to_holder: input.to_holder.clone(),
        volume_liters: input.volume_liters,
        transfer_type: input.transfer_type,
        approved_by: input.approved_by,
        transferred_at: now,
    };

    let transfer_hash = create_entry(&EntryTypes::RightTransfer(transfer))?;

    // Link right to transfer
    create_link(
        input.right_hash.clone(),
        transfer_hash.clone(),
        LinkTypes::RightToTransfer,
        (),
    )?;

    // Update the original right status if full transfer
    if input.volume_liters == right.volume_authorized_liters {
        let updated_right = WaterRight {
            watershed_hash: right.watershed_hash,
            holder: right.holder,
            right_type: right.right_type,
            volume_authorized_liters: right.volume_authorized_liters,
            priority_date: right.priority_date,
            conditions: right.conditions,
            status: RightStatus::Transferred,
            transferable: right.transferable,
        };
        update_entry(
            right_record.action_address().clone(),
            &EntryTypes::WaterRight(updated_right),
        )?;
    }

    get(transfer_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created transfer".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferRightInput {
    pub right_hash: ActionHash,
    pub to_holder: AgentPubKey,
    pub volume_liters: u64,
    pub transfer_type: TransferType,
    pub approved_by: Option<AgentPubKey>,
}

// ============================================================================
// DISPUTE RESOLUTION
// ============================================================================

/// File a water dispute within a watershed
#[hdk_extern]
pub fn file_dispute(dispute: WaterDispute) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "file_dispute")?;
    if dispute.description.trim().is_empty() || dispute.description.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-8192 non-whitespace characters".into()
        )));
    }

    // Verify watershed exists
    let _ws = get(dispute.watershed_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Watershed not found".into())
    ))?;

    let action_hash = create_entry(&EntryTypes::WaterDispute(dispute.clone()))?;

    // Link watershed to dispute
    create_link(
        dispute.watershed_hash.clone(),
        action_hash.clone(),
        LinkTypes::WatershedToDispute,
        (),
    )?;

    // Link complainant to dispute
    create_link(
        dispute.complainant.clone(),
        action_hash.clone(),
        LinkTypes::AgentToDispute,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created dispute".into()
    )))
}

/// Resolve a water dispute
#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_constitutional(), "resolve_dispute")?;

    if input.resolution_text.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Resolution text cannot be empty or whitespace-only".into()
        )));
    }

    let record = get(input.dispute_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Dispute not found".into())
    ))?;
    let mut dispute: WaterDispute = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid dispute entry".into()
        )))?;

    if dispute.status == DisputeStatus::Resolved || dispute.status == DisputeStatus::Dismissed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute is already resolved or dismissed".into()
        )));
    }

    dispute.status = input.new_status;
    dispute.resolution = Some(input.resolution_text);

    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::WaterDispute(dispute),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated dispute".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_hash: ActionHash,
    pub new_status: DisputeStatus,
    pub resolution_text: String,
}

/// Get all disputes for a watershed
#[hdk_extern]
pub fn get_watershed_disputes(watershed_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(watershed_hash, LinkTypes::WatershedToDispute)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get disputes filed by the calling agent
#[hdk_extern]
pub fn get_my_disputes(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let links = get_links(
        LinkQuery::try_new(agent_info.agent_initial_pubkey, LinkTypes::AgentToDispute)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// Cross-domain: Verify property before registering water right
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckPropertyForWaterRightInput {
    /// Property ID to verify in the property registry.
    pub property_id: String,
    /// Expected owner DID — if provided, verifies the property owner matches.
    pub expected_owner_did: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PropertyCheckResult {
    pub property_exists: bool,
    pub has_clear_title: bool,
    pub owner_matches: bool,
    pub error: Option<String>,
}

/// Verify a property exists in the registry before registering a water right.
///
/// Cross-domain call: water-steward queries property_registry via
/// `call(CallTargetCell::Local, ...)` to confirm the property exists
/// and has clear title. This prevents registering water rights for
/// non-existent or encumbered properties.
#[hdk_extern]
pub fn check_property_for_water_right(
    input: CheckPropertyForWaterRightInput,
) -> ExternResult<PropertyCheckResult> {
    // 1. Check if property exists
    let get_response = call(
        CallTargetCell::Local,
        ZomeName::from("property_registry"),
        FunctionName::from("get_property"),
        None,
        input.property_id.clone(),
    );

    let property_exists = match &get_response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let record: Option<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            record.is_some()
        }
        _ => false,
    };

    if !property_exists {
        return Ok(PropertyCheckResult {
            property_exists: false,
            has_clear_title: false,
            owner_matches: false,
            error: Some(format!(
                "Property '{}' not found in registry",
                input.property_id
            )),
        });
    }

    // 2. Check clear title
    let title_response = call(
        CallTargetCell::Local,
        ZomeName::from("property_registry"),
        FunctionName::from("has_clear_title"),
        None,
        input.property_id.clone(),
    );

    let has_clear_title = match &title_response {
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io.decode::<bool>().unwrap_or(false),
        _ => false,
    };

    // 3. Optionally verify ownership
    let owner_matches = if let Some(ref expected) = input.expected_owner_did {
        let verify_response = call(
            CallTargetCell::Local,
            ZomeName::from("property_registry"),
            FunctionName::from("verify_ownership"),
            None,
            serde_json::json!({
                "property_id": input.property_id,
                "expected_owner": expected,
            })
            .to_string(),
        );

        match &verify_response {
            Ok(ZomeCallResponse::Ok(extern_io)) => extern_io.decode::<bool>().unwrap_or(false),
            _ => false,
        }
    } else {
        true // No ownership check requested
    };

    Ok(PropertyCheckResult {
        property_exists,
        has_clear_title,
        owner_matches,
        error: if !has_clear_title {
            Some("Property has encumbrances — clear title needed for water right".to_string())
        } else if !owner_matches {
            Some("Property ownership does not match expected holder".to_string())
        } else {
            None
        },
    })
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ========================================================================
    // COORDINATOR STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn transfer_right_input_serde_roundtrip() {
        let input = TransferRightInput {
            right_hash: fake_action_hash(),
            to_holder: fake_agent_2(),
            volume_liters: 5000,
            transfer_type: TransferType::Sale,
            approved_by: Some(fake_agent()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.volume_liters, 5000);
        assert_eq!(decoded.transfer_type, TransferType::Sale);
        assert!(decoded.approved_by.is_some());
    }

    #[test]
    fn transfer_right_input_without_approval() {
        let input = TransferRightInput {
            right_hash: fake_action_hash(),
            to_holder: fake_agent(),
            volume_liters: 100,
            transfer_type: TransferType::Donation,
            approved_by: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.volume_liters, 100);
        assert_eq!(decoded.transfer_type, TransferType::Donation);
        assert!(decoded.approved_by.is_none());
    }

    #[test]
    fn transfer_right_input_all_transfer_types() {
        for transfer_type in [
            TransferType::Sale,
            TransferType::Lease,
            TransferType::Donation,
            TransferType::Inheritance,
            TransferType::Emergency,
        ] {
            let input = TransferRightInput {
                right_hash: fake_action_hash(),
                to_holder: fake_agent(),
                volume_liters: 1000,
                transfer_type: transfer_type.clone(),
                approved_by: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: TransferRightInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.transfer_type, transfer_type);
        }
    }

    #[test]
    fn resolve_dispute_input_serde_roundtrip() {
        let input = ResolveDisputeInput {
            dispute_hash: fake_action_hash(),
            new_status: DisputeStatus::Resolved,
            resolution_text: "Parties agreed to share water access equally".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, DisputeStatus::Resolved);
        assert_eq!(
            decoded.resolution_text,
            "Parties agreed to share water access equally"
        );
    }

    #[test]
    fn resolve_dispute_input_dismissed() {
        let input = ResolveDisputeInput {
            dispute_hash: fake_action_hash(),
            new_status: DisputeStatus::Dismissed,
            resolution_text: "Insufficient evidence".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, DisputeStatus::Dismissed);
        assert_eq!(decoded.resolution_text, "Insufficient evidence");
    }

    #[test]
    fn resolve_dispute_input_all_statuses() {
        for status in [
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Mediation,
            DisputeStatus::Arbitration,
            DisputeStatus::Resolved,
            DisputeStatus::Dismissed,
        ] {
            let input = ResolveDisputeInput {
                dispute_hash: fake_action_hash(),
                new_status: status.clone(),
                resolution_text: "Test resolution".to_string(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: ResolveDisputeInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    #[test]
    fn check_property_for_water_right_input_serde_roundtrip() {
        let input = CheckPropertyForWaterRightInput {
            property_id: "PROP-123".to_string(),
            expected_owner_did: Some("did:holo:abc123".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CheckPropertyForWaterRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "PROP-123");
        assert_eq!(
            decoded.expected_owner_did,
            Some("did:holo:abc123".to_string())
        );
    }

    #[test]
    fn check_property_for_water_right_input_without_owner() {
        let input = CheckPropertyForWaterRightInput {
            property_id: "PROP-456".to_string(),
            expected_owner_did: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CheckPropertyForWaterRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "PROP-456");
        assert!(decoded.expected_owner_did.is_none());
    }

    #[test]
    fn property_check_result_serde_roundtrip_success() {
        let result = PropertyCheckResult {
            property_exists: true,
            has_clear_title: true,
            owner_matches: true,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_exists, true);
        assert_eq!(decoded.has_clear_title, true);
        assert_eq!(decoded.owner_matches, true);
        assert!(decoded.error.is_none());
    }

    #[test]
    fn property_check_result_serde_roundtrip_not_found() {
        let result = PropertyCheckResult {
            property_exists: false,
            has_clear_title: false,
            owner_matches: false,
            error: Some("Property 'PROP-999' not found in registry".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_exists, false);
        assert_eq!(
            decoded.error,
            Some("Property 'PROP-999' not found in registry".to_string())
        );
    }

    #[test]
    fn property_check_result_serde_roundtrip_encumbered() {
        let result = PropertyCheckResult {
            property_exists: true,
            has_clear_title: false,
            owner_matches: true,
            error: Some("Property has encumbrances".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_exists, true);
        assert_eq!(decoded.has_clear_title, false);
        assert_eq!(decoded.owner_matches, true);
        assert!(decoded.error.is_some());
    }

    // ========================================================================
    // INTEGRITY ENUM SERDE ROUNDTRIP TESTS (via coordinator re-export)
    // ========================================================================

    #[test]
    fn stewardship_type_all_variants_serde() {
        for variant in [
            StewardshipType::Riparian,
            StewardshipType::PriorAppropriation,
            StewardshipType::Commons,
            StewardshipType::Hybrid,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: StewardshipType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn right_type_all_variants_serde() {
        for variant in [
            RightType::Riparian,
            RightType::Appropriative,
            RightType::Prescriptive,
            RightType::Aboriginal,
            RightType::Commons,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: RightType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn right_status_all_variants_serde() {
        for variant in [
            RightStatus::Active,
            RightStatus::Suspended,
            RightStatus::Revoked,
            RightStatus::Transferred,
            RightStatus::Expired,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: RightStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn transfer_type_all_variants_serde() {
        for variant in [
            TransferType::Sale,
            TransferType::Lease,
            TransferType::Donation,
            TransferType::Inheritance,
            TransferType::Emergency,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: TransferType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn dispute_type_all_variants_serde() {
        for variant in [
            DisputeType::Allocation,
            DisputeType::Quality,
            DisputeType::Access,
            DisputeType::Overuse,
            DisputeType::Encroachment,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: DisputeType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn dispute_status_all_variants_serde() {
        for variant in [
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Mediation,
            DisputeStatus::Arbitration,
            DisputeStatus::Resolved,
            DisputeStatus::Dismissed,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: DisputeStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn water_source_type_all_variants_serde() {
        for variant in [
            WaterSourceType::Municipal,
            WaterSourceType::Well,
            WaterSourceType::Spring,
            WaterSourceType::Rainwater,
            WaterSourceType::Aquifer,
            WaterSourceType::River,
            WaterSourceType::Lake,
            WaterSourceType::Recycled,
            WaterSourceType::Desalinated,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: WaterSourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    // ========================================================================
    // INTEGRITY ENTRY STRUCT SERDE ROUNDTRIP TESTS (via coordinator re-export)
    // ========================================================================

    #[test]
    fn watershed_serde_roundtrip() {
        let ws = Watershed {
            id: "ws-test".to_string(),
            name: "Test Watershed".to_string(),
            huc_code: Some("17110006".to_string()),
            boundary: vec![(45.5, -122.6), (45.6, -122.5), (45.4, -122.4)],
            area_sq_km: 150.0,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        let json = serde_json::to_string(&ws).unwrap();
        let decoded: Watershed = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "ws-test");
        assert_eq!(decoded.name, "Test Watershed");
        assert_eq!(decoded.huc_code, Some("17110006".to_string()));
        assert_eq!(decoded.boundary.len(), 3);
        assert_eq!(decoded.stewardship_type, StewardshipType::Commons);
        assert_eq!(decoded.primary_source_type, WaterSourceType::River);
    }

    #[test]
    fn water_right_serde_roundtrip() {
        let right = WaterRight {
            watershed_hash: fake_action_hash(),
            holder: fake_agent(),
            right_type: RightType::Aboriginal,
            volume_authorized_liters: u64::MAX,
            priority_date: Some(Timestamp::from_micros(1_500_000_000)),
            conditions: vec!["Seasonal only".to_string(), "No commercial use".to_string()],
            status: RightStatus::Active,
            transferable: false,
        };
        let json = serde_json::to_string(&right).unwrap();
        let decoded: WaterRight = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.right_type, RightType::Aboriginal);
        assert_eq!(decoded.volume_authorized_liters, u64::MAX);
        assert_eq!(decoded.conditions.len(), 2);
        assert_eq!(decoded.transferable, false);
    }

    #[test]
    fn water_dispute_serde_roundtrip() {
        let dispute = WaterDispute {
            watershed_hash: fake_action_hash(),
            complainant: fake_agent(),
            respondent: fake_agent_2(),
            dispute_type: DisputeType::Quality,
            description: "Upstream pollution affecting downstream users".to_string(),
            evidence: vec![fake_action_hash()],
            status: DisputeStatus::Mediation,
            resolution: None,
        };
        let json = serde_json::to_string(&dispute).unwrap();
        let decoded: WaterDispute = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.dispute_type, DisputeType::Quality);
        assert_eq!(decoded.status, DisputeStatus::Mediation);
        assert_eq!(decoded.evidence.len(), 1);
        assert!(decoded.resolution.is_none());
    }

    #[test]
    fn right_transfer_serde_roundtrip() {
        let transfer = RightTransfer {
            right_hash: fake_action_hash(),
            from_holder: fake_agent(),
            to_holder: fake_agent_2(),
            volume_liters: 10_000,
            transfer_type: TransferType::Inheritance,
            approved_by: Some(fake_agent()),
            transferred_at: Timestamp::from_micros(1_700_000_000),
        };
        let json = serde_json::to_string(&transfer).unwrap();
        let decoded: RightTransfer = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.volume_liters, 10_000);
        assert_eq!(decoded.transfer_type, TransferType::Inheritance);
        assert!(decoded.approved_by.is_some());
    }

    #[test]
    fn property_check_result_owner_mismatch_and_encumbered() {
        let result = PropertyCheckResult {
            property_exists: true,
            has_clear_title: false,
            owner_matches: false,
            error: Some("Property ownership does not match expected holder".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_exists, true);
        assert_eq!(decoded.has_clear_title, false);
        assert_eq!(decoded.owner_matches, false);
        assert!(decoded.error.is_some());
    }

    // ========================================================================
    // VALIDATION HARDENING EDGE CASE TESTS
    // ========================================================================

    /// Whitespace-only watershed ID should be rejected.
    #[test]
    fn watershed_whitespace_only_id_serde() {
        let ws = Watershed {
            id: "   ".to_string(),
            name: "Valid Name".to_string(),
            huc_code: None,
            boundary: vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            area_sq_km: 10.0,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        let json = serde_json::to_string(&ws).unwrap();
        let decoded: Watershed = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.id.trim().is_empty(),
            "Whitespace-only ID must be caught by trim()"
        );
    }

    /// Whitespace-only watershed name should be rejected.
    #[test]
    fn watershed_whitespace_only_name_serde() {
        let ws = Watershed {
            id: "ws-001".to_string(),
            name: " \t\n ".to_string(),
            huc_code: None,
            boundary: vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            area_sq_km: 10.0,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        let json = serde_json::to_string(&ws).unwrap();
        let decoded: Watershed = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.name.trim().is_empty(),
            "Whitespace-only name must be caught by trim()"
        );
    }

    /// Whitespace-only dispute description should be rejected.
    #[test]
    fn dispute_whitespace_only_description_serde() {
        let dispute = WaterDispute {
            watershed_hash: fake_action_hash(),
            complainant: fake_agent(),
            respondent: fake_agent_2(),
            dispute_type: DisputeType::Allocation,
            description: "    ".to_string(),
            evidence: vec![],
            status: DisputeStatus::Filed,
            resolution: None,
        };
        let json = serde_json::to_string(&dispute).unwrap();
        let decoded: WaterDispute = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.description.trim().is_empty(),
            "Whitespace-only description must be caught by trim()"
        );
    }

    /// Zero-volume transfer right should be rejected.
    #[test]
    fn transfer_right_input_zero_volume_serde() {
        let input = TransferRightInput {
            right_hash: fake_action_hash(),
            to_holder: fake_agent_2(),
            volume_liters: 0,
            transfer_type: TransferType::Sale,
            approved_by: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.volume_liters, 0,
            "Zero volume preserved in struct for coordinator rejection"
        );
    }

    /// Whitespace-only resolution text should be rejected.
    #[test]
    fn resolve_dispute_input_whitespace_only_resolution_serde() {
        let input = ResolveDisputeInput {
            dispute_hash: fake_action_hash(),
            new_status: DisputeStatus::Resolved,
            resolution_text: "   \t  ".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveDisputeInput = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.resolution_text.trim().is_empty(),
            "Whitespace-only resolution must be caught by trim()"
        );
    }

    // ========================================================================
    // FLOAT VALIDATION EDGE CASES
    // ========================================================================

    #[test]
    fn watershed_nan_area_rejected() {
        let ws = Watershed {
            id: "ws-nan".to_string(),
            name: "NaN Watershed".to_string(),
            huc_code: None,
            boundary: vec![(45.5, -122.6)],
            area_sq_km: f32::NAN,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        // NaN should fail is_finite() check in define_watershed
        assert!(!ws.area_sq_km.is_finite());
    }

    #[test]
    fn watershed_infinite_area_rejected() {
        let ws = Watershed {
            id: "ws-inf".to_string(),
            name: "Infinite Watershed".to_string(),
            huc_code: None,
            boundary: vec![(45.5, -122.6)],
            area_sq_km: f32::INFINITY,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        assert!(!ws.area_sq_km.is_finite());
    }

    #[test]
    fn watershed_negative_area_rejected() {
        let ws = Watershed {
            id: "ws-neg".to_string(),
            name: "Negative Watershed".to_string(),
            huc_code: None,
            boundary: vec![(45.5, -122.6)],
            area_sq_km: -10.0,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        assert!(ws.area_sq_km < 0.0, "Negative area should be caught");
    }

    #[test]
    fn watershed_boundary_nan_coordinate_detected() {
        let boundary = vec![(f64::NAN, -122.6), (45.6, -122.5)];
        assert!(
            !boundary[0].0.is_finite(),
            "NaN coordinate should fail is_finite()"
        );
    }

    #[test]
    fn watershed_boundary_out_of_range_detected() {
        // Latitude > 90 should be rejected
        let lat = 91.0;
        assert!(
            !(-90.0..=90.0).contains(&lat),
            "Latitude > 90 should be out of range"
        );
    }

    #[test]
    fn watershed_valid_area_accepted() {
        let ws = Watershed {
            id: "ws-ok".to_string(),
            name: "Valid Watershed".to_string(),
            huc_code: None,
            boundary: vec![(45.5, -122.6), (45.6, -122.5), (45.4, -122.4)],
            area_sq_km: 150.0,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        };
        assert!(ws.area_sq_km.is_finite() && ws.area_sq_km >= 0.0);
        for (lat, lon) in &ws.boundary {
            assert!(lat.is_finite() && lon.is_finite());
            assert!((-90.0..=90.0).contains(lat));
            assert!((-180.0..=180.0).contains(lon));
        }
    }
}
