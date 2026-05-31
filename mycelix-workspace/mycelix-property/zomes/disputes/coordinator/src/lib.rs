// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Disputes Coordinator Zome
use disputes_integrity::*;
use hdk::prelude::*;

/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    // Create the anchor entry - if it already exists, this is idempotent
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    // Return the deterministic entry hash
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn file_dispute(input: FileDisputeInput) -> ExternResult<Record> {
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
                return get(action_hash, GetOptions::default())?
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
                return get(action_hash, GetOptions::default())?
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
                return get(action_hash, GetOptions::default())?
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
                return get(action_hash, GetOptions::default())?
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
                return get(action_hash, GetOptions::default())?
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
                return get(action_hash, GetOptions::default())?
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
