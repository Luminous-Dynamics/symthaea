//! Steward Integrity Zome
//! Watershed governance, water rights, transfers, and dispute resolution

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// WATER SOURCE TYPE (re-exported for cross-zome reference)
// ============================================================================

/// Type of water source (mirrored from flow for steward context)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WaterSourceType {
    Municipal,
    Well,
    Spring,
    Rainwater,
    Aquifer,
    River,
    Lake,
    Recycled,
    Desalinated,
}

// ============================================================================
// WATERSHED
// ============================================================================

/// Type of water stewardship governance
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum StewardshipType {
    /// Land-adjacent rights
    Riparian,
    /// First-in-time, first-in-right
    PriorAppropriation,
    /// Community-managed commons
    Commons,
    /// Mixed governance
    Hybrid,
}

/// A watershed under community stewardship
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Watershed {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Optional HUC (Hydrologic Unit Code) identifier
    pub huc_code: Option<String>,
    /// Boundary polygon as (lat, lon) pairs
    pub boundary: Vec<(f64, f64)>,
    /// Area in square kilometers
    pub area_sq_km: f32,
    /// Governance model for this watershed
    pub stewardship_type: StewardshipType,
    /// Optional link to governing body (e.g., governance hApp proposal)
    pub governing_body: Option<ActionHash>,
    /// Primary water source type in this watershed
    pub primary_source_type: WaterSourceType,
}

// ============================================================================
// WATER RIGHTS
// ============================================================================

/// Type of water right
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RightType {
    /// Based on land adjacency
    Riparian,
    /// Based on historical use
    Appropriative,
    /// Established through long use
    Prescriptive,
    /// Indigenous / first nations rights
    Aboriginal,
    /// Community commons right
    Commons,
}

/// Status of a water right
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RightStatus {
    Active,
    Suspended,
    Revoked,
    Transferred,
    Expired,
}

/// A legally recognized water right within a watershed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterRight {
    /// Watershed this right belongs to
    pub watershed_hash: ActionHash,
    /// Agent holding this right
    pub holder: AgentPubKey,
    /// Type of right
    pub right_type: RightType,
    /// Maximum authorized volume in liters
    pub volume_authorized_liters: u64,
    /// Priority date (for appropriative rights)
    pub priority_date: Option<Timestamp>,
    /// Conditions attached to this right
    pub conditions: Vec<String>,
    /// Current status of the right
    pub status: RightStatus,
    /// Whether this right can be transferred
    pub transferable: bool,
}

// ============================================================================
// RIGHT TRANSFERS
// ============================================================================

/// Type of water right transfer
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferType {
    Sale,
    Lease,
    Donation,
    Inheritance,
    Emergency,
}

/// A transfer of a water right from one holder to another
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RightTransfer {
    /// The water right being transferred
    pub right_hash: ActionHash,
    /// Current holder
    pub from_holder: AgentPubKey,
    /// New holder
    pub to_holder: AgentPubKey,
    /// Volume being transferred in liters
    pub volume_liters: u64,
    /// Type of transfer
    pub transfer_type: TransferType,
    /// Agent who approved the transfer (e.g., watershed governance)
    pub approved_by: Option<AgentPubKey>,
    /// When the transfer was executed
    pub transferred_at: Timestamp,
}

// ============================================================================
// WATER DISPUTES
// ============================================================================

/// Type of water dispute
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeType {
    /// Disagreement over allocation amounts
    Allocation,
    /// Disagreement over water quality responsibility
    Quality,
    /// Access to water sources
    Access,
    /// Excessive water use
    Overuse,
    /// Physical or legal encroachment
    Encroachment,
}

/// Status of a water dispute
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    Filed,
    UnderReview,
    Mediation,
    Arbitration,
    Resolved,
    Dismissed,
}

/// A water dispute between parties in a watershed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterDispute {
    /// Watershed where the dispute occurs
    pub watershed_hash: ActionHash,
    /// Agent filing the complaint
    pub complainant: AgentPubKey,
    /// Agent being complained about
    pub respondent: AgentPubKey,
    /// Type of dispute
    pub dispute_type: DisputeType,
    /// Description of the dispute
    pub description: String,
    /// Evidence action hashes
    pub evidence: Vec<ActionHash>,
    /// Current status
    pub status: DisputeStatus,
    /// Resolution text (when resolved)
    pub resolution: Option<String>,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Watershed(Watershed),
    WaterRight(WaterRight),
    RightTransfer(RightTransfer),
    WaterDispute(WaterDispute),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all watersheds
    AllWatersheds,
    /// Watershed to its water rights
    WatershedToRight,
    /// Holder to their water rights
    HolderToRight,
    /// Right to its transfers
    RightToTransfer,
    /// Watershed to disputes
    WatershedToDispute,
    /// Agent to disputes they filed
    AgentToDispute,
    /// Stewardship type to watersheds
    StewardshipTypeToWatershed,
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
                EntryTypes::Watershed(ws) => validate_create_watershed(action, ws),
                EntryTypes::WaterRight(right) => validate_create_water_right(action, right),
                EntryTypes::RightTransfer(transfer) => {
                    validate_create_right_transfer(action, transfer)
                }
                EntryTypes::WaterDispute(dispute) => validate_create_water_dispute(action, dispute),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Watershed(ws) => validate_update_watershed(ws, original_action_hash),
                EntryTypes::WaterRight(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WaterDispute(_) => Ok(ValidateCallbackResult::Valid),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllWatersheds => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllWatersheds link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::WatershedToRight => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "WatershedToRight link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::HolderToRight => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "HolderToRight link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RightToTransfer => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "RightToTransfer link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::WatershedToDispute => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "WatershedToDispute link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToDispute => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToDispute link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::StewardshipTypeToWatershed => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "StewardshipTypeToWatershed link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_watershed(
    _action: Create,
    ws: Watershed,
) -> ExternResult<ValidateCallbackResult> {
    if ws.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed ID cannot be empty".into(),
        ));
    }
    if ws.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed ID must be 256 characters or fewer".into(),
        ));
    }
    if ws.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed name cannot be empty".into(),
        ));
    }
    if ws.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed name must be 256 characters or fewer".into(),
        ));
    }
    if let Some(ref huc) = ws.huc_code {
        if huc.len() > 64 {
            return Ok(ValidateCallbackResult::Invalid(
                "HUC code must be 64 characters or fewer".into(),
            ));
        }
    }
    if ws.boundary.len() < 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed boundary must have at least 3 points".into(),
        ));
    }
    if !ws.area_sq_km.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Area must be a finite number".into(),
        ));
    }
    if ws.area_sq_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed area must be positive".into(),
        ));
    }
    // Validate boundary coordinates
    for (lat, lon) in &ws.boundary {
        if *lat < -90.0 || *lat > 90.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary latitude must be between -90 and 90".into(),
            ));
        }
        if *lon < -180.0 || *lon > 180.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary longitude must be between -180 and 180".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_watershed(
    ws: Watershed,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_ws: Watershed = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original watershed not found".into()
        )))?;
    if ws.id != original_ws.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change watershed ID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_water_right(
    _action: Create,
    right: WaterRight,
) -> ExternResult<ValidateCallbackResult> {
    if right.volume_authorized_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Authorized volume must be greater than zero".into(),
        ));
    }
    if right.conditions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 conditions".into(),
        ));
    }
    for condition in &right.conditions {
        if condition.trim().is_empty() || condition.len() > 1024 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each condition must be 1-1024 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_right_transfer(
    _action: Create,
    transfer: RightTransfer,
) -> ExternResult<ValidateCallbackResult> {
    if transfer.volume_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transfer volume must be greater than zero".into(),
        ));
    }
    if transfer.from_holder == transfer.to_holder {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transfer a right to the same holder".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_water_dispute(
    _action: Create,
    dispute: WaterDispute,
) -> ExternResult<ValidateCallbackResult> {
    if dispute.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description cannot be empty".into(),
        ));
    }
    if dispute.description.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description must be at most 8192 characters".into(),
        ));
    }
    if dispute.complainant == dispute.respondent {
        return Ok(ValidateCallbackResult::Invalid(
            "Complainant and respondent cannot be the same agent".into(),
        ));
    }
    if dispute.evidence.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 evidence items".into(),
        ));
    }
    if let Some(ref res) = dispute.resolution {
        if res.len() > 8192 {
            return Ok(ValidateCallbackResult::Invalid(
                "Resolution must be 8192 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    fn make_watershed() -> Watershed {
        Watershed {
            id: "ws-001".into(),
            name: "Cedar Creek Watershed".into(),
            huc_code: Some("17110006".into()),
            boundary: vec![(45.5, -122.6), (45.6, -122.5), (45.4, -122.4)],
            area_sq_km: 150.0,
            stewardship_type: StewardshipType::Commons,
            governing_body: None,
            primary_source_type: WaterSourceType::River,
        }
    }

    fn make_water_right() -> WaterRight {
        WaterRight {
            watershed_hash: fake_action_hash(),
            holder: fake_agent(),
            right_type: RightType::Commons,
            volume_authorized_liters: 10_000,
            priority_date: None,
            conditions: vec!["No irrigation during drought".into()],
            status: RightStatus::Active,
            transferable: true,
        }
    }

    fn make_right_transfer() -> RightTransfer {
        RightTransfer {
            right_hash: fake_action_hash(),
            from_holder: fake_agent(),
            to_holder: fake_agent_2(),
            volume_liters: 5_000,
            transfer_type: TransferType::Lease,
            approved_by: None,
            transferred_at: Timestamp::from_micros(0),
        }
    }

    fn make_dispute() -> WaterDispute {
        WaterDispute {
            watershed_hash: fake_action_hash(),
            complainant: fake_agent(),
            respondent: fake_agent_2(),
            dispute_type: DisputeType::Allocation,
            description: "Excessive water withdrawal upstream".into(),
            evidence: vec![],
            status: DisputeStatus::Filed,
            resolution: None,
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_water_source_type() {
        let variants = vec![
            WaterSourceType::Municipal,
            WaterSourceType::Well,
            WaterSourceType::Spring,
            WaterSourceType::Rainwater,
            WaterSourceType::Aquifer,
            WaterSourceType::River,
            WaterSourceType::Lake,
            WaterSourceType::Recycled,
            WaterSourceType::Desalinated,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: WaterSourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_stewardship_type() {
        let variants = vec![
            StewardshipType::Riparian,
            StewardshipType::PriorAppropriation,
            StewardshipType::Commons,
            StewardshipType::Hybrid,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: StewardshipType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_right_type() {
        let variants = vec![
            RightType::Riparian,
            RightType::Appropriative,
            RightType::Prescriptive,
            RightType::Aboriginal,
            RightType::Commons,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: RightType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_right_status() {
        let variants = vec![
            RightStatus::Active,
            RightStatus::Suspended,
            RightStatus::Revoked,
            RightStatus::Transferred,
            RightStatus::Expired,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: RightStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_transfer_type() {
        let variants = vec![
            TransferType::Sale,
            TransferType::Lease,
            TransferType::Donation,
            TransferType::Inheritance,
            TransferType::Emergency,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: TransferType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_dispute_type() {
        let variants = vec![
            DisputeType::Allocation,
            DisputeType::Quality,
            DisputeType::Access,
            DisputeType::Overuse,
            DisputeType::Encroachment,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: DisputeType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_dispute_status() {
        let variants = vec![
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Mediation,
            DisputeStatus::Arbitration,
            DisputeStatus::Resolved,
            DisputeStatus::Dismissed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: DisputeStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_anchor() {
        let anchor = Anchor("all_watersheds".into());
        let bytes = holochain_serialized_bytes::encode(&anchor).unwrap();
        let back: Anchor = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn serde_roundtrip_watershed() {
        let ws = make_watershed();
        let bytes = holochain_serialized_bytes::encode(&ws).unwrap();
        let back: Watershed = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(ws, back);
    }

    #[test]
    fn serde_roundtrip_water_right() {
        let right = make_water_right();
        let bytes = holochain_serialized_bytes::encode(&right).unwrap();
        let back: WaterRight = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(right, back);
    }

    #[test]
    fn serde_roundtrip_right_transfer() {
        let transfer = make_right_transfer();
        let bytes = holochain_serialized_bytes::encode(&transfer).unwrap();
        let back: RightTransfer = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(transfer, back);
    }

    #[test]
    fn serde_roundtrip_water_dispute() {
        let dispute = make_dispute();
        let bytes = holochain_serialized_bytes::encode(&dispute).unwrap();
        let back: WaterDispute = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(dispute, back);
    }

    #[test]
    fn serde_roundtrip_watershed_all_optional_fields() {
        let ws = Watershed {
            id: "ws-full".into(),
            name: "Full Watershed".into(),
            huc_code: None,
            boundary: vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            area_sq_km: 1.0,
            stewardship_type: StewardshipType::PriorAppropriation,
            governing_body: Some(fake_action_hash()),
            primary_source_type: WaterSourceType::Desalinated,
        };
        let bytes = holochain_serialized_bytes::encode(&ws).unwrap();
        let back: Watershed = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(ws, back);
    }

    #[test]
    fn serde_roundtrip_water_right_with_priority_date() {
        let mut right = make_water_right();
        right.priority_date = Some(Timestamp::from_micros(1_000_000));
        let bytes = holochain_serialized_bytes::encode(&right).unwrap();
        let back: WaterRight = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(right, back);
    }

    #[test]
    fn serde_roundtrip_right_transfer_with_approval() {
        let mut transfer = make_right_transfer();
        transfer.approved_by = Some(fake_agent_2());
        let bytes = holochain_serialized_bytes::encode(&transfer).unwrap();
        let back: RightTransfer = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(transfer, back);
    }

    #[test]
    fn serde_roundtrip_dispute_with_resolution() {
        let mut dispute = make_dispute();
        dispute.resolution = Some("Parties agreed to split allocation 60/40".into());
        dispute.status = DisputeStatus::Resolved;
        let bytes = holochain_serialized_bytes::encode(&dispute).unwrap();
        let back: WaterDispute = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(dispute, back);
    }

    // ========================================================================
    // WATERSHED VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_watershed_passes() {
        let result = validate_create_watershed(fake_create(), make_watershed());
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_empty_id_rejected() {
        let mut ws = make_watershed();
        ws.id = "".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_empty_id_error_message() {
        let mut ws = make_watershed();
        ws.id = "".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(invalid_msg(&result), "Watershed ID cannot be empty");
    }

    #[test]
    fn watershed_empty_name_rejected() {
        let mut ws = make_watershed();
        ws.name = "".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_empty_name_error_message() {
        let mut ws = make_watershed();
        ws.name = "".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(invalid_msg(&result), "Watershed name cannot be empty");
    }

    #[test]
    fn watershed_two_points_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(45.5, -122.6), (45.6, -122.5)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_one_point_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(45.5, -122.6)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_zero_points_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_boundary_error_message() {
        let mut ws = make_watershed();
        ws.boundary = vec![(0.0, 0.0), (1.0, 1.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(
            invalid_msg(&result),
            "Watershed boundary must have at least 3 points"
        );
    }

    #[test]
    fn watershed_three_points_accepted() {
        let ws = make_watershed(); // Already has 3 points
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_many_points_accepted() {
        let mut ws = make_watershed();
        ws.boundary = (0..100).map(|i| (i as f64 * 0.5, i as f64 * 0.5)).collect();
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_zero_area_rejected() {
        let mut ws = make_watershed();
        ws.area_sq_km = 0.0;
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_negative_area_rejected() {
        let mut ws = make_watershed();
        ws.area_sq_km = -1.0;
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_area_error_message() {
        let mut ws = make_watershed();
        ws.area_sq_km = -5.0;
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(invalid_msg(&result), "Watershed area must be positive");
    }

    #[test]
    fn watershed_tiny_area_accepted() {
        let mut ws = make_watershed();
        ws.area_sq_km = 0.001;
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_very_large_area_accepted() {
        let mut ws = make_watershed();
        ws.area_sq_km = f32::MAX;
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_lat_over_90_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(91.0, 0.0), (45.0, 0.0), (44.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_lat_under_neg90_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(-91.0, 0.0), (45.0, 0.0), (44.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_lat_error_message() {
        let mut ws = make_watershed();
        ws.boundary = vec![(91.0, 0.0), (45.0, 0.0), (44.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(
            invalid_msg(&result),
            "Boundary latitude must be between -90 and 90"
        );
    }

    #[test]
    fn watershed_lon_over_180_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(45.0, 181.0), (45.0, 0.0), (44.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_lon_under_neg180_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(45.0, -181.0), (45.0, 0.0), (44.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_lon_error_message() {
        let mut ws = make_watershed();
        ws.boundary = vec![(45.0, 200.0), (45.0, 0.0), (44.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(
            invalid_msg(&result),
            "Boundary longitude must be between -180 and 180"
        );
    }

    #[test]
    fn watershed_boundary_lat_90_accepted() {
        let mut ws = make_watershed();
        ws.boundary = vec![(90.0, 0.0), (-90.0, 0.0), (0.0, 180.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_boundary_lon_180_accepted() {
        let mut ws = make_watershed();
        ws.boundary = vec![(0.0, 180.0), (0.0, -180.0), (0.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_lat_invalid_in_middle_point_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(10.0, 10.0), (95.0, 10.0), (10.0, 20.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_lon_invalid_in_last_point_rejected() {
        let mut ws = make_watershed();
        ws.boundary = vec![(10.0, 10.0), (20.0, 10.0), (10.0, -200.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
    }

    #[test]
    fn watershed_unicode_id_accepted() {
        let mut ws = make_watershed();
        ws.id = "\u{6c34}\u{7cfb}".into(); // Chinese for "watershed"
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_unicode_name_accepted() {
        let mut ws = make_watershed();
        ws.name = "\u{00d6}resunds\u{00e5}n V\u{00e4}stra".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_single_char_id_accepted() {
        let mut ws = make_watershed();
        ws.id = "x".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_single_char_name_accepted() {
        let mut ws = make_watershed();
        ws.name = "W".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_no_huc_code_accepted() {
        let mut ws = make_watershed();
        ws.huc_code = None;
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_with_governing_body_accepted() {
        let mut ws = make_watershed();
        ws.governing_body = Some(fake_action_hash());
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_all_stewardship_types_accepted() {
        for st in [
            StewardshipType::Riparian,
            StewardshipType::PriorAppropriation,
            StewardshipType::Commons,
            StewardshipType::Hybrid,
        ] {
            let mut ws = make_watershed();
            ws.stewardship_type = st;
            let result = validate_create_watershed(fake_create(), ws);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn watershed_all_source_types_accepted() {
        for src in [
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
            let mut ws = make_watershed();
            ws.primary_source_type = src;
            let result = validate_create_watershed(fake_create(), ws);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // WATER RIGHT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_water_right_passes() {
        let result = validate_create_water_right(fake_create(), make_water_right());
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_zero_volume_rejected() {
        let mut right = make_water_right();
        right.volume_authorized_liters = 0;
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_right_zero_volume_error_message() {
        let mut right = make_water_right();
        right.volume_authorized_liters = 0;
        let result = validate_create_water_right(fake_create(), right);
        assert_eq!(
            invalid_msg(&result),
            "Authorized volume must be greater than zero"
        );
    }

    #[test]
    fn water_right_one_liter_accepted() {
        let mut right = make_water_right();
        right.volume_authorized_liters = 1;
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_max_u64_volume_accepted() {
        let mut right = make_water_right();
        right.volume_authorized_liters = u64::MAX;
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_50_conditions_accepted() {
        let mut right = make_water_right();
        right.conditions = (0..50).map(|i| format!("Condition {}", i)).collect();
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_51_conditions_rejected() {
        let mut right = make_water_right();
        right.conditions = (0..51).map(|i| format!("Condition {}", i)).collect();
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_right_51_conditions_error_message() {
        let mut right = make_water_right();
        right.conditions = (0..51).map(|i| format!("Condition {}", i)).collect();
        let result = validate_create_water_right(fake_create(), right);
        assert_eq!(invalid_msg(&result), "Cannot have more than 50 conditions");
    }

    #[test]
    fn water_right_empty_condition_rejected() {
        let mut right = make_water_right();
        right.conditions = vec!["Valid condition".into(), "".into()];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_right_empty_condition_error_message() {
        let mut right = make_water_right();
        right.conditions = vec!["".into()];
        let result = validate_create_water_right(fake_create(), right);
        assert_eq!(
            invalid_msg(&result),
            "Each condition must be 1-1024 characters"
        );
    }

    #[test]
    fn water_right_condition_over_1024_rejected() {
        let mut right = make_water_right();
        right.conditions = vec!["x".repeat(1025)];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_right_condition_over_1024_error_message() {
        let mut right = make_water_right();
        right.conditions = vec!["x".repeat(1025)];
        let result = validate_create_water_right(fake_create(), right);
        assert_eq!(
            invalid_msg(&result),
            "Each condition must be 1-1024 characters"
        );
    }

    #[test]
    fn water_right_condition_at_1024_accepted() {
        let mut right = make_water_right();
        right.conditions = vec!["x".repeat(1024)];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_condition_single_char_accepted() {
        let mut right = make_water_right();
        right.conditions = vec!["a".into()];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_no_conditions_accepted() {
        let mut right = make_water_right();
        right.conditions = vec![];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_unicode_condition_accepted() {
        let mut right = make_water_right();
        right.conditions = vec!["\u{6c34}\u{6743} restricted during monsoon".into()];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_all_right_types_accepted() {
        for rt in [
            RightType::Riparian,
            RightType::Appropriative,
            RightType::Prescriptive,
            RightType::Aboriginal,
            RightType::Commons,
        ] {
            let mut right = make_water_right();
            right.right_type = rt;
            let result = validate_create_water_right(fake_create(), right);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_right_all_statuses_accepted() {
        for status in [
            RightStatus::Active,
            RightStatus::Suspended,
            RightStatus::Revoked,
            RightStatus::Transferred,
            RightStatus::Expired,
        ] {
            let mut right = make_water_right();
            right.status = status;
            let result = validate_create_water_right(fake_create(), right);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_right_not_transferable_accepted() {
        let mut right = make_water_right();
        right.transferable = false;
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_with_priority_date_accepted() {
        let mut right = make_water_right();
        right.priority_date = Some(Timestamp::from_micros(1_700_000_000_000_000));
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_right_first_empty_condition_in_list_rejected() {
        let mut right = make_water_right();
        right.conditions = vec!["".into(), "Valid condition".into()];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_right_oversized_condition_in_middle_rejected() {
        let mut right = make_water_right();
        right.conditions = vec!["Short".into(), "x".repeat(1025), "Also short".into()];
        let result = validate_create_water_right(fake_create(), right);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // RIGHT TRANSFER VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_right_transfer_passes() {
        let result = validate_create_right_transfer(fake_create(), make_right_transfer());
        assert!(is_valid(&result));
    }

    #[test]
    fn right_transfer_zero_volume_rejected() {
        let mut transfer = make_right_transfer();
        transfer.volume_liters = 0;
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert!(is_invalid(&result));
    }

    #[test]
    fn right_transfer_zero_volume_error_message() {
        let mut transfer = make_right_transfer();
        transfer.volume_liters = 0;
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert_eq!(
            invalid_msg(&result),
            "Transfer volume must be greater than zero"
        );
    }

    #[test]
    fn right_transfer_one_liter_accepted() {
        let mut transfer = make_right_transfer();
        transfer.volume_liters = 1;
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert!(is_valid(&result));
    }

    #[test]
    fn right_transfer_max_u64_volume_accepted() {
        let mut transfer = make_right_transfer();
        transfer.volume_liters = u64::MAX;
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert!(is_valid(&result));
    }

    #[test]
    fn right_transfer_same_holder_rejected() {
        let mut transfer = make_right_transfer();
        transfer.to_holder = transfer.from_holder.clone();
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert!(is_invalid(&result));
    }

    #[test]
    fn right_transfer_same_holder_error_message() {
        let mut transfer = make_right_transfer();
        transfer.to_holder = transfer.from_holder.clone();
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert_eq!(
            invalid_msg(&result),
            "Cannot transfer a right to the same holder"
        );
    }

    #[test]
    fn right_transfer_all_transfer_types_accepted() {
        for tt in [
            TransferType::Sale,
            TransferType::Lease,
            TransferType::Donation,
            TransferType::Inheritance,
            TransferType::Emergency,
        ] {
            let mut transfer = make_right_transfer();
            transfer.transfer_type = tt;
            let result = validate_create_right_transfer(fake_create(), transfer);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn right_transfer_with_approval_accepted() {
        let mut transfer = make_right_transfer();
        transfer.approved_by = Some(fake_agent());
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert!(is_valid(&result));
    }

    #[test]
    fn right_transfer_without_approval_accepted() {
        let transfer = make_right_transfer(); // approved_by is None by default
        assert!(transfer.approved_by.is_none());
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // WATER DISPUTE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_water_dispute_passes() {
        let result = validate_create_water_dispute(fake_create(), make_dispute());
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_empty_description_rejected() {
        let mut dispute = make_dispute();
        dispute.description = "".into();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_dispute_empty_description_error_message() {
        let mut dispute = make_dispute();
        dispute.description = "".into();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(invalid_msg(&result), "Dispute description cannot be empty");
    }

    #[test]
    fn water_dispute_single_char_description_accepted() {
        let mut dispute = make_dispute();
        dispute.description = "X".into();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_description_at_limit_accepted() {
        let mut dispute = make_dispute();
        dispute.description = "x".repeat(8192);
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_description_over_limit_rejected() {
        let mut dispute = make_dispute();
        dispute.description = "x".repeat(8193);
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_dispute_description_over_limit_error_message() {
        let mut dispute = make_dispute();
        dispute.description = "x".repeat(8193);
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(
            invalid_msg(&result),
            "Dispute description must be at most 8192 characters"
        );
    }

    #[test]
    fn water_dispute_unicode_description_accepted() {
        let mut dispute = make_dispute();
        dispute.description = "\u{6c34}\u{4e89}\u{8bae}: upstream pollutant discharge \u{2014} \u{00e9}coulement toxique".into();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_same_parties_rejected() {
        let mut dispute = make_dispute();
        dispute.respondent = dispute.complainant.clone();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_dispute_same_parties_error_message() {
        let mut dispute = make_dispute();
        dispute.respondent = dispute.complainant.clone();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(
            invalid_msg(&result),
            "Complainant and respondent cannot be the same agent"
        );
    }

    #[test]
    fn water_dispute_100_evidence_accepted() {
        let mut dispute = make_dispute();
        dispute.evidence = (0..100).map(|_| fake_action_hash()).collect();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_101_evidence_rejected() {
        let mut dispute = make_dispute();
        dispute.evidence = (0..101).map(|_| fake_action_hash()).collect();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_invalid(&result));
    }

    #[test]
    fn water_dispute_101_evidence_error_message() {
        let mut dispute = make_dispute();
        dispute.evidence = (0..101).map(|_| fake_action_hash()).collect();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(
            invalid_msg(&result),
            "Cannot have more than 100 evidence items"
        );
    }

    #[test]
    fn water_dispute_no_evidence_accepted() {
        let dispute = make_dispute(); // Default has empty evidence
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_single_evidence_accepted() {
        let mut dispute = make_dispute();
        dispute.evidence = vec![fake_action_hash()];
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_all_dispute_types_accepted() {
        for dt in [
            DisputeType::Allocation,
            DisputeType::Quality,
            DisputeType::Access,
            DisputeType::Overuse,
            DisputeType::Encroachment,
        ] {
            let mut dispute = make_dispute();
            dispute.dispute_type = dt;
            let result = validate_create_water_dispute(fake_create(), dispute);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_dispute_all_statuses_accepted() {
        for status in [
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Mediation,
            DisputeStatus::Arbitration,
            DisputeStatus::Resolved,
            DisputeStatus::Dismissed,
        ] {
            let mut dispute = make_dispute();
            dispute.status = status;
            let result = validate_create_water_dispute(fake_create(), dispute);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_dispute_with_resolution_text_accepted() {
        let mut dispute = make_dispute();
        dispute.resolution = Some("Resolved via mediation: 50/50 split".into());
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_with_empty_resolution_accepted() {
        let mut dispute = make_dispute();
        dispute.resolution = Some("".into());
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn watershed_id_at_limit_accepted() {
        let mut ws = make_watershed();
        ws.id = "A".repeat(64);
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_id_over_limit_rejected() {
        let mut ws = make_watershed();
        ws.id = "A".repeat(257);
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Watershed ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn watershed_name_at_limit_accepted() {
        let mut ws = make_watershed();
        ws.name = "A".repeat(256);
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_name_over_limit_rejected() {
        let mut ws = make_watershed();
        ws.name = "A".repeat(257);
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Watershed name must be 256 characters or fewer"
        );
    }

    #[test]
    fn watershed_huc_code_at_limit_accepted() {
        let mut ws = make_watershed();
        ws.huc_code = Some("H".repeat(64));
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_valid(&result));
    }

    #[test]
    fn watershed_huc_code_over_limit_rejected() {
        let mut ws = make_watershed();
        ws.huc_code = Some("H".repeat(65));
        let result = validate_create_watershed(fake_create(), ws);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "HUC code must be 64 characters or fewer"
        );
    }

    #[test]
    fn water_dispute_resolution_at_limit_accepted() {
        let mut dispute = make_dispute();
        dispute.resolution = Some("R".repeat(8192));
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_dispute_resolution_over_limit_rejected() {
        let mut dispute = make_dispute();
        dispute.resolution = Some("R".repeat(8193));
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Resolution must be 8192 characters or fewer"
        );
    }

    // ========================================================================
    // LINK TAG LENGTH VALIDATION TESTS
    // ========================================================================

    fn validate_link_tag_for(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllWatersheds
            | LinkTypes::WatershedToRight
            | LinkTypes::HolderToRight
            | LinkTypes::WatershedToDispute
            | LinkTypes::AgentToDispute => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RightToTransfer | LinkTypes::StewardshipTypeToWatershed => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 512 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn link_all_watersheds_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AllWatersheds, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_all_watersheds_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AllWatersheds, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_watershed_to_right_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::WatershedToRight, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_watershed_to_right_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::WatershedToRight, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_holder_to_right_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::HolderToRight, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_holder_to_right_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::HolderToRight, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_right_to_transfer_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::RightToTransfer, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_right_to_transfer_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::RightToTransfer, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_watershed_to_dispute_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::WatershedToDispute, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_watershed_to_dispute_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::WatershedToDispute, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_agent_to_dispute_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AgentToDispute, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_agent_to_dispute_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AgentToDispute, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_stewardship_type_to_watershed_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::StewardshipTypeToWatershed, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_stewardship_type_to_watershed_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::StewardshipTypeToWatershed, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // CLONE TESTS (derive correctness)
    // ========================================================================

    #[test]
    fn watershed_clone_is_equal() {
        let ws = make_watershed();
        let cloned = ws.clone();
        assert_eq!(ws, cloned);
    }

    #[test]
    fn water_right_clone_is_equal() {
        let right = make_water_right();
        let cloned = right.clone();
        assert_eq!(right, cloned);
    }

    #[test]
    fn right_transfer_clone_is_equal() {
        let transfer = make_right_transfer();
        let cloned = transfer.clone();
        assert_eq!(transfer, cloned);
    }

    #[test]
    fn water_dispute_clone_is_equal() {
        let dispute = make_dispute();
        let cloned = dispute.clone();
        assert_eq!(dispute, cloned);
    }

    #[test]
    fn anchor_clone_is_equal() {
        let anchor = Anchor("test".into());
        let cloned = anchor.clone();
        assert_eq!(anchor, cloned);
    }

    // ========================================================================
    // VALIDATION PRIORITY ORDER TESTS
    // ========================================================================

    #[test]
    fn watershed_empty_id_checked_before_empty_name() {
        let mut ws = make_watershed();
        ws.id = "".into();
        ws.name = "".into();
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(invalid_msg(&result), "Watershed ID cannot be empty");
    }

    #[test]
    fn watershed_empty_name_checked_before_boundary() {
        let mut ws = make_watershed();
        ws.name = "".into();
        ws.boundary = vec![];
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(invalid_msg(&result), "Watershed name cannot be empty");
    }

    #[test]
    fn watershed_boundary_checked_before_area() {
        let mut ws = make_watershed();
        ws.boundary = vec![(0.0, 0.0)];
        ws.area_sq_km = -1.0;
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(
            invalid_msg(&result),
            "Watershed boundary must have at least 3 points"
        );
    }

    #[test]
    fn watershed_area_checked_before_coords() {
        let mut ws = make_watershed();
        ws.area_sq_km = 0.0;
        ws.boundary = vec![(91.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let result = validate_create_watershed(fake_create(), ws);
        assert_eq!(invalid_msg(&result), "Watershed area must be positive");
    }

    #[test]
    fn water_right_volume_checked_before_conditions_count() {
        let mut right = make_water_right();
        right.volume_authorized_liters = 0;
        right.conditions = (0..51).map(|i| format!("C{}", i)).collect();
        let result = validate_create_water_right(fake_create(), right);
        assert_eq!(
            invalid_msg(&result),
            "Authorized volume must be greater than zero"
        );
    }

    #[test]
    fn water_right_conditions_count_checked_before_content() {
        let mut right = make_water_right();
        right.conditions = (0..51).map(|_| "".into()).collect();
        let result = validate_create_water_right(fake_create(), right);
        assert_eq!(invalid_msg(&result), "Cannot have more than 50 conditions");
    }

    #[test]
    fn transfer_volume_checked_before_same_holder() {
        let mut transfer = make_right_transfer();
        transfer.volume_liters = 0;
        transfer.to_holder = transfer.from_holder.clone();
        let result = validate_create_right_transfer(fake_create(), transfer);
        assert_eq!(
            invalid_msg(&result),
            "Transfer volume must be greater than zero"
        );
    }

    #[test]
    fn dispute_description_empty_checked_before_same_parties() {
        let mut dispute = make_dispute();
        dispute.description = "".into();
        dispute.respondent = dispute.complainant.clone();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(invalid_msg(&result), "Dispute description cannot be empty");
    }

    #[test]
    fn dispute_description_length_checked_before_same_parties() {
        let mut dispute = make_dispute();
        dispute.description = "x".repeat(8193);
        dispute.respondent = dispute.complainant.clone();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(
            invalid_msg(&result),
            "Dispute description must be at most 8192 characters"
        );
    }

    #[test]
    fn dispute_same_parties_checked_before_evidence() {
        let mut dispute = make_dispute();
        dispute.respondent = dispute.complainant.clone();
        dispute.evidence = (0..101).map(|_| fake_action_hash()).collect();
        let result = validate_create_water_dispute(fake_create(), dispute);
        assert_eq!(
            invalid_msg(&result),
            "Complainant and respondent cannot be the same agent"
        );
    }
}
