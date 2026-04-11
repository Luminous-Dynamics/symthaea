// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property Disputes Coordinator Zome
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_proposal, civic_requirement_voting};
use mycelix_zome_helpers::get_latest_record;
use property_disputes_integrity::*;


/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    // Create the anchor entry - if it already exists, this is idempotent
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    // Return the deterministic entry hash
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn file_dispute(input: FileDisputeInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "file_dispute")?;
    let now = sys_time()?;
    let dispute = PropertyDispute {
        id: format!("dispute:{}:{}", input.property_id, now.as_micros()),
        property_id: input.property_id.clone(),
        dispute_type: input.dispute_type,
        claimant_did: input.claimant_did.clone(),
        respondent_did: input.respondent_did,
        description: input.description,
        evidence_ids: input.evidence_ids,
        status: DisputeStatus::Filed,
        justice_case_id: None,
        filed: now,
        resolved: None,
    };

    let action_hash = create_entry(&EntryTypes::PropertyDispute(dispute))?;
    create_link(
        anchor_hash(&input.property_id)?,
        action_hash.clone(),
        LinkTypes::PropertyToDisputes,
        (),
    )?;
    create_link(
        anchor_hash(&input.claimant_did)?,
        action_hash.clone(),
        LinkTypes::ClaimantToDisputes,
        (),
    )?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FileDisputeInput {
    pub property_id: String,
    pub dispute_type: DisputeType,
    pub claimant_did: String,
    pub respondent_did: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
}

#[hdk_extern]
pub fn file_ownership_claim(input: FileClaimInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "file_ownership_claim")?;
    let now = sys_time()?;
    let claim = OwnershipClaim {
        id: format!("claim:{}:{}", input.property_id, now.as_micros()),
        property_id: input.property_id.clone(),
        claimant_did: input.claimant_did.clone(),
        claim_basis: input.claim_basis,
        supporting_documents: input.supporting_documents,
        status: ClaimStatus::Pending,
        filed: now,
    };

    let action_hash = create_entry(&EntryTypes::OwnershipClaim(claim))?;
    create_link(
        anchor_hash(&input.property_id)?,
        action_hash.clone(),
        LinkTypes::PropertyToClaims,
        (),
    )?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FileClaimInput {
    pub property_id: String,
    pub claimant_did: String,
    pub claim_basis: ClaimBasis,
    pub supporting_documents: Vec<String>,
}

#[hdk_extern]
pub fn escalate_to_justice(input: EscalateInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "escalate_to_justice")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == input.dispute_id {
                let updated = PropertyDispute {
                    status: DisputeStatus::Arbitration,
                    justice_case_id: Some(input.justice_case_id.clone()),
                    ..dispute
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::PropertyDispute(updated),
                )?;
                create_link(
                    action_hash.clone(),
                    anchor_hash(&input.justice_case_id)?,
                    LinkTypes::DisputeToJustice,
                    (),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EscalateInput {
    pub dispute_id: String,
    pub justice_case_id: String,
}

#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "resolve_dispute")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == input.dispute_id {
                let now = sys_time()?;
                let updated = PropertyDispute {
                    status: if input.dismissed {
                        DisputeStatus::Dismissed
                    } else {
                        DisputeStatus::Resolved
                    },
                    resolved: Some(now),
                    ..dispute
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::PropertyDispute(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_id: String,
    pub dismissed: bool,
}

#[hdk_extern]
pub fn get_property_disputes(property_id: String) -> ExternResult<Vec<Record>> {
    let mut disputes = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&property_id)?, LinkTypes::PropertyToDisputes)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            disputes.push(record);
        }
    }
    Ok(disputes)
}

/// Get a specific dispute by ID
#[hdk_extern]
pub fn get_dispute(dispute_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == dispute_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all disputes filed by a claimant
#[hdk_extern]
pub fn get_claimant_disputes(claimant_did: String) -> ExternResult<Vec<Record>> {
    let mut disputes = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&claimant_did)?, LinkTypes::ClaimantToDisputes)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            disputes.push(record);
        }
    }
    Ok(disputes)
}

/// Get all ownership claims for a property
#[hdk_extern]
pub fn get_property_claims(property_id: String) -> ExternResult<Vec<Record>> {
    let mut claims = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&property_id)?, LinkTypes::PropertyToClaims)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            claims.push(record);
        }
    }
    Ok(claims)
}

/// Update dispute status
#[hdk_extern]
pub fn update_dispute_status(input: UpdateDisputeStatusInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "update_dispute_status")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == input.dispute_id {
                let updated = PropertyDispute {
                    status: input.new_status,
                    ..dispute
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::PropertyDispute(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDisputeStatusInput {
    pub dispute_id: String,
    pub new_status: DisputeStatus,
}

/// Update ownership claim status
#[hdk_extern]
pub fn update_claim_status(input: UpdateClaimStatusInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "update_claim_status")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::OwnershipClaim,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(claim) = record
            .entry()
            .to_app_option::<OwnershipClaim>()
            .ok()
            .flatten()
        {
            if claim.id == input.claim_id {
                let updated = OwnershipClaim {
                    status: input.new_status,
                    ..claim
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::OwnershipClaim(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Claim not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateClaimStatusInput {
    pub claim_id: String,
    pub new_status: ClaimStatus,
}

/// Add evidence to a dispute
#[hdk_extern]
pub fn add_dispute_evidence(input: AddEvidenceInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "add_dispute_evidence")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == input.dispute_id {
                // Only parties can add evidence
                if dispute.claimant_did != input.submitter_did
                    && dispute.respondent_did != input.submitter_did
                {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only parties can add evidence".into()
                    )));
                }

                let mut evidence_ids = dispute.evidence_ids.clone();
                evidence_ids.push(input.evidence_id);

                let updated = PropertyDispute {
                    evidence_ids,
                    ..dispute
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::PropertyDispute(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddEvidenceInput {
    pub dispute_id: String,
    pub evidence_id: String,
    pub submitter_did: String,
}

/// Get disputes by status
#[hdk_extern]
pub fn get_disputes_by_status(status: DisputeStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.status == status {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Get a specific ownership claim by ID
#[hdk_extern]
pub fn get_ownership_claim(claim_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::OwnershipClaim,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(claim) = record
            .entry()
            .to_app_option::<OwnershipClaim>()
            .ok()
            .flatten()
        {
            if claim.id == claim_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Add supporting document to ownership claim
#[hdk_extern]
pub fn add_claim_document(input: AddDocumentInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "add_claim_document")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::OwnershipClaim,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(claim) = record
            .entry()
            .to_app_option::<OwnershipClaim>()
            .ok()
            .flatten()
        {
            if claim.id == input.claim_id {
                // Only claimant can add documents
                if claim.claimant_did != input.submitter_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only claimant can add documents".into()
                    )));
                }

                let mut docs = claim.supporting_documents.clone();
                docs.push(input.document_id);

                let updated = OwnershipClaim {
                    supporting_documents: docs,
                    ..claim
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::OwnershipClaim(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Claim not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddDocumentInput {
    pub claim_id: String,
    pub document_id: String,
    pub submitter_did: String,
}

/// Get disputes where DID is respondent
#[hdk_extern]
pub fn get_respondent_disputes(respondent_did: String) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PropertyDispute,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<PropertyDispute>()
            .ok()
            .flatten()
        {
            if dispute.respondent_did == respondent_did {
                results.push(record);
            }
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_dispute_input_serde_roundtrip() {
        let input = FileDisputeInput {
            property_id: "prop-001".to_string(),
            dispute_type: DisputeType::Boundary,
            claimant_did: "did:key:z6Mk001".to_string(),
            respondent_did: "did:key:z6Mk002".to_string(),
            description: "Fence encroachment on north side".to_string(),
            evidence_ids: vec!["survey-001".to_string(), "photo-001".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-001");
        assert_eq!(decoded.evidence_ids.len(), 2);
    }

    #[test]
    fn file_claim_input_serde_roundtrip() {
        let input = FileClaimInput {
            property_id: "prop-001".to_string(),
            claimant_did: "did:key:z6Mk003".to_string(),
            claim_basis: ClaimBasis::Inheritance,
            supporting_documents: vec!["will.pdf".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileClaimInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.claimant_did, "did:key:z6Mk003");
        assert_eq!(decoded.supporting_documents.len(), 1);
    }

    #[test]
    fn escalate_input_serde_roundtrip() {
        let input = EscalateInput {
            dispute_id: "dispute-001".to_string(),
            justice_case_id: "case-001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: EscalateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.justice_case_id, "case-001");
    }

    #[test]
    fn resolve_dispute_input_serde_roundtrip_resolved() {
        let input = ResolveDisputeInput {
            dispute_id: "dispute-001".to_string(),
            dismissed: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveDisputeInput = serde_json::from_str(&json).unwrap();
        assert!(!decoded.dismissed);
    }

    #[test]
    fn resolve_dispute_input_serde_roundtrip_dismissed() {
        let input = ResolveDisputeInput {
            dispute_id: "dispute-001".to_string(),
            dismissed: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveDisputeInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.dismissed);
    }

    #[test]
    fn update_dispute_status_input_serde_roundtrip() {
        let input = UpdateDisputeStatusInput {
            dispute_id: "dispute-001".to_string(),
            new_status: DisputeStatus::Mediation,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateDisputeStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, DisputeStatus::Mediation);
    }

    #[test]
    fn update_claim_status_input_serde_roundtrip() {
        let input = UpdateClaimStatusInput {
            claim_id: "claim-001".to_string(),
            new_status: ClaimStatus::UnderInvestigation,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateClaimStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, ClaimStatus::UnderInvestigation);
    }

    #[test]
    fn add_evidence_input_serde_roundtrip() {
        let input = AddEvidenceInput {
            dispute_id: "dispute-001".to_string(),
            evidence_id: "photo-002".to_string(),
            submitter_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEvidenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.evidence_id, "photo-002");
    }

    #[test]
    fn add_document_input_serde_roundtrip() {
        let input = AddDocumentInput {
            claim_id: "claim-001".to_string(),
            document_id: "deed-scan.pdf".to_string(),
            submitter_did: "did:key:z6Mk003".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddDocumentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.document_id, "deed-scan.pdf");
    }

    #[test]
    fn dispute_type_all_variants_serialize() {
        let types = vec![
            DisputeType::Boundary,
            DisputeType::Ownership,
            DisputeType::Encumbrance,
            DisputeType::Easement,
            DisputeType::Trespass,
            DisputeType::Damage,
            DisputeType::Other("Nuisance".to_string()),
        ];
        for dt in types {
            let json = serde_json::to_string(&dt).unwrap();
            let decoded: DisputeType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, dt);
        }
    }

    #[test]
    fn dispute_status_all_variants_serialize() {
        let statuses = vec![
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Mediation,
            DisputeStatus::Arbitration,
            DisputeStatus::Resolved,
            DisputeStatus::Dismissed,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: DisputeStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn claim_basis_all_variants_serialize() {
        let bases = vec![
            ClaimBasis::PriorOwnership,
            ClaimBasis::Inheritance,
            ClaimBasis::AdversePossession,
            ClaimBasis::FraudulentTransfer,
            ClaimBasis::DocumentaryEvidence,
            ClaimBasis::Other("Custom basis".to_string()),
        ];
        for basis in bases {
            let json = serde_json::to_string(&basis).unwrap();
            let decoded: ClaimBasis = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, basis);
        }
    }

    #[test]
    fn claim_status_all_variants_serialize() {
        let statuses = vec![
            ClaimStatus::Pending,
            ClaimStatus::UnderInvestigation,
            ClaimStatus::Validated,
            ClaimStatus::Rejected,
            ClaimStatus::Superseded,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: ClaimStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    // ========================================================================
    // Boundary condition tests
    // ========================================================================

    #[test]
    fn file_dispute_input_empty_description() {
        let input = FileDisputeInput {
            property_id: "prop-001".to_string(),
            dispute_type: DisputeType::Boundary,
            claimant_did: "did:key:z6Mk001".to_string(),
            respondent_did: "did:key:z6Mk002".to_string(),
            description: "".to_string(),
            evidence_ids: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description, "");
        assert!(decoded.evidence_ids.is_empty());
    }

    #[test]
    fn file_dispute_input_very_long_description() {
        let long_desc = "D".repeat(10_000);
        let input = FileDisputeInput {
            property_id: "prop-001".to_string(),
            dispute_type: DisputeType::Damage,
            claimant_did: "did:key:z6Mk001".to_string(),
            respondent_did: "did:key:z6Mk002".to_string(),
            description: long_desc.clone(),
            evidence_ids: vec!["e1".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description.len(), 10_000);
    }

    #[test]
    fn file_dispute_input_many_evidence_ids() {
        let evidence: Vec<String> = (0..100).map(|i| format!("evidence-{}", i)).collect();
        let input = FileDisputeInput {
            property_id: "prop-001".to_string(),
            dispute_type: DisputeType::Ownership,
            claimant_did: "did:key:z6Mk001".to_string(),
            respondent_did: "did:key:z6Mk002".to_string(),
            description: "Many evidence items".to_string(),
            evidence_ids: evidence.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.evidence_ids.len(), 100);
    }

    #[test]
    fn file_claim_input_empty_supporting_documents() {
        let input = FileClaimInput {
            property_id: "prop-001".to_string(),
            claimant_did: "did:key:z6Mk001".to_string(),
            claim_basis: ClaimBasis::PriorOwnership,
            supporting_documents: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileClaimInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.supporting_documents.is_empty());
    }

    #[test]
    fn file_claim_input_many_documents() {
        let docs: Vec<String> = (0..50).map(|i| format!("doc-{}.pdf", i)).collect();
        let input = FileClaimInput {
            property_id: "prop-001".to_string(),
            claimant_did: "did:key:z6Mk001".to_string(),
            claim_basis: ClaimBasis::DocumentaryEvidence,
            supporting_documents: docs.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileClaimInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.supporting_documents.len(), 50);
    }

    // ========================================================================
    // Dispute type Other variant with different strings
    // ========================================================================

    #[test]
    fn dispute_type_other_empty_string() {
        let dt = DisputeType::Other("".to_string());
        let json = serde_json::to_string(&dt).unwrap();
        let decoded: DisputeType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, DisputeType::Other("".to_string()));
    }

    #[test]
    fn claim_basis_other_empty_string() {
        let cb = ClaimBasis::Other("".to_string());
        let json = serde_json::to_string(&cb).unwrap();
        let decoded: ClaimBasis = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ClaimBasis::Other("".to_string()));
    }

    // ========================================================================
    // Full struct field access tests
    // ========================================================================

    #[test]
    fn file_dispute_input_all_fields_accessed() {
        let input = FileDisputeInput {
            property_id: "prop-full".to_string(),
            dispute_type: DisputeType::Easement,
            claimant_did: "did:key:z6MkAlpha".to_string(),
            respondent_did: "did:key:z6MkBeta".to_string(),
            description: "Access easement dispute".to_string(),
            evidence_ids: vec!["deed-copy".to_string(), "map.pdf".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileDisputeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-full");
        assert_eq!(decoded.dispute_type, DisputeType::Easement);
        assert_eq!(decoded.claimant_did, "did:key:z6MkAlpha");
        assert_eq!(decoded.respondent_did, "did:key:z6MkBeta");
        assert_eq!(decoded.description, "Access easement dispute");
        assert_eq!(decoded.evidence_ids.len(), 2);
    }

    #[test]
    fn file_claim_input_all_fields_accessed() {
        let input = FileClaimInput {
            property_id: "prop-claim-full".to_string(),
            claimant_did: "did:key:z6MkClaimer".to_string(),
            claim_basis: ClaimBasis::AdversePossession,
            supporting_documents: vec![
                "affidavit.pdf".to_string(),
                "tax-records.pdf".to_string(),
                "photos.zip".to_string(),
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FileClaimInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "prop-claim-full");
        assert_eq!(decoded.claimant_did, "did:key:z6MkClaimer");
        assert_eq!(decoded.claim_basis, ClaimBasis::AdversePossession);
        assert_eq!(decoded.supporting_documents.len(), 3);
    }

    // ========================================================================
    // All dispute type variants for FileDisputeInput
    // ========================================================================

    #[test]
    fn file_dispute_input_all_dispute_types() {
        let types = vec![
            DisputeType::Boundary,
            DisputeType::Ownership,
            DisputeType::Encumbrance,
            DisputeType::Easement,
            DisputeType::Trespass,
            DisputeType::Damage,
            DisputeType::Other("Custom".to_string()),
        ];
        for dt in types {
            let input = FileDisputeInput {
                property_id: "prop-001".to_string(),
                dispute_type: dt.clone(),
                claimant_did: "did:key:z6Mk001".to_string(),
                respondent_did: "did:key:z6Mk002".to_string(),
                description: "test".to_string(),
                evidence_ids: vec![],
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: FileDisputeInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.dispute_type, dt);
        }
    }

    // ========================================================================
    // All claim basis variants for FileClaimInput
    // ========================================================================

    #[test]
    fn file_claim_input_all_claim_bases() {
        let bases = vec![
            ClaimBasis::PriorOwnership,
            ClaimBasis::Inheritance,
            ClaimBasis::AdversePossession,
            ClaimBasis::FraudulentTransfer,
            ClaimBasis::DocumentaryEvidence,
            ClaimBasis::Other("Treaty".to_string()),
        ];
        for basis in bases {
            let input = FileClaimInput {
                property_id: "prop-001".to_string(),
                claimant_did: "did:key:z6Mk001".to_string(),
                claim_basis: basis.clone(),
                supporting_documents: vec![],
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: FileClaimInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.claim_basis, basis);
        }
    }

    // ========================================================================
    // UpdateDisputeStatusInput all status variants
    // ========================================================================

    #[test]
    fn update_dispute_status_all_variants() {
        let statuses = vec![
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Mediation,
            DisputeStatus::Arbitration,
            DisputeStatus::Resolved,
            DisputeStatus::Dismissed,
        ];
        for status in statuses {
            let input = UpdateDisputeStatusInput {
                dispute_id: "dispute-001".to_string(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateDisputeStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    // ========================================================================
    // UpdateClaimStatusInput all status variants
    // ========================================================================

    #[test]
    fn update_claim_status_all_variants() {
        let statuses = vec![
            ClaimStatus::Pending,
            ClaimStatus::UnderInvestigation,
            ClaimStatus::Validated,
            ClaimStatus::Rejected,
            ClaimStatus::Superseded,
        ];
        for status in statuses {
            let input = UpdateClaimStatusInput {
                claim_id: "claim-001".to_string(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateClaimStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    // ========================================================================
    // Empty string boundary tests
    // ========================================================================

    #[test]
    fn escalate_input_empty_dispute_id() {
        let input = EscalateInput {
            dispute_id: "".to_string(),
            justice_case_id: "case-001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: EscalateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.dispute_id, "");
    }

    #[test]
    fn add_evidence_input_empty_evidence_id() {
        let input = AddEvidenceInput {
            dispute_id: "dispute-001".to_string(),
            evidence_id: "".to_string(),
            submitter_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddEvidenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.evidence_id, "");
    }

    #[test]
    fn add_document_input_empty_document_id() {
        let input = AddDocumentInput {
            claim_id: "claim-001".to_string(),
            document_id: "".to_string(),
            submitter_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddDocumentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.document_id, "");
    }
}
