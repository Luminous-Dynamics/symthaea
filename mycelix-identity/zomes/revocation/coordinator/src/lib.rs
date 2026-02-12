//! Revocation Registry Coordinator Zome
//! Business logic for credential revocation and status checking
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use revocation_integrity::*;

/// Create a deterministic entry hash from a string identifier
/// This is used for link bases when we need to link from string IDs
fn string_to_entry_hash(s: &str) -> EntryHash {
    EntryHash::from_raw_36(
        holo_hash::blake2b_256(s.as_bytes())
            .into_iter()
            .chain([0u8; 4])
            .collect::<Vec<u8>>()
            .try_into()
            .expect("36 bytes"),
    )
}

/// Revoke a credential
#[hdk_extern]
pub fn revoke_credential(input: RevokeInput) -> ExternResult<Record> {
    // Input validation
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential ID must be 1-256 characters".into())));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer DID must be 1-256 characters".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }

    let now = sys_time()?;

    let entry = RevocationEntry {
        credential_id: input.credential_id.clone(),
        issuer: input.issuer_did.clone(),
        status: RevocationStatus::Revoked,
        reason: input.reason,
        effective_from: input.effective_from.unwrap_or(now),
        recorded_at: now,
        suspension_end: None,
    };

    let action_hash = create_entry(&EntryTypes::RevocationEntry(entry))?;

    // Link credential to revocation using deterministic hash
    let credential_hash = string_to_entry_hash(&input.credential_id);
    create_link(
        credential_hash,
        action_hash.clone(),
        LinkTypes::CredentialToRevocation,
        (),
    )?;

    // Link issuer to revocation
    let issuer_hash = string_to_entry_hash(&input.issuer_did);
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::IssuerToRevocation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find revocation entry".into()
        )))
}

/// Input for revoking a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeInput {
    pub credential_id: String,
    pub issuer_did: String,
    pub reason: String,
    pub effective_from: Option<Timestamp>,
}

/// Suspend a credential temporarily
#[hdk_extern]
pub fn suspend_credential(input: SuspendInput) -> ExternResult<Record> {
    // Input validation
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential ID must be 1-256 characters".into())));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer DID must be 1-256 characters".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }

    let now = sys_time()?;

    let entry = RevocationEntry {
        credential_id: input.credential_id.clone(),
        issuer: input.issuer_did.clone(),
        status: RevocationStatus::Suspended,
        reason: input.reason,
        effective_from: now,
        recorded_at: now,
        suspension_end: Some(input.suspension_end),
    };

    let action_hash = create_entry(&EntryTypes::RevocationEntry(entry))?;

    // Link credential to revocation
    let credential_hash = string_to_entry_hash(&input.credential_id);
    create_link(
        credential_hash,
        action_hash.clone(),
        LinkTypes::CredentialToRevocation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find suspension entry".into()
        )))
}

/// Input for suspending a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct SuspendInput {
    pub credential_id: String,
    pub issuer_did: String,
    pub reason: String,
    pub suspension_end: Timestamp,
}

/// Reinstate a suspended credential
#[hdk_extern]
pub fn reinstate_credential(input: ReinstateInput) -> ExternResult<Record> {
    // Input validation
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential ID must be 1-256 characters".into())));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer DID must be 1-256 characters".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }

    // Find the current revocation entry
    let credential_hash = string_to_entry_hash(&input.credential_id);
    let links = get_links(
        LinkQuery::try_new(credential_hash, LinkTypes::CredentialToRevocation)?,
        GetStrategy::default(),
    )?;

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    let current_action_hash = latest_link
        .map(|l| ActionHash::try_from(l.target))
        .transpose()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "No revocation entry found".into()
        )))?;

    let current_record = get(current_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Revocation entry not found".into()
        )))?;

    let current_entry: RevocationEntry = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid revocation entry".into()
        )))?;

    // Can only reinstate suspended credentials
    if current_entry.status != RevocationStatus::Suspended {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only reinstate suspended credentials".into()
        )));
    }

    let now = sys_time()?;

    let reinstated = RevocationEntry {
        credential_id: current_entry.credential_id,
        issuer: current_entry.issuer,
        status: RevocationStatus::Active,
        reason: input.reason,
        effective_from: now,
        recorded_at: now,
        suspension_end: None,
    };

    let action_hash = update_entry(current_action_hash, &EntryTypes::RevocationEntry(reinstated))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find reinstated entry".into()
        )))
}

/// Input for reinstating a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct ReinstateInput {
    pub credential_id: String,
    pub issuer_did: String,
    pub reason: String,
}

/// Check revocation status of a credential
#[hdk_extern]
pub fn check_revocation_status(credential_id: String) -> ExternResult<RevocationCheckResult> {
    let now = sys_time()?;

    let credential_hash = string_to_entry_hash(&credential_id);
    let links = get_links(
        LinkQuery::try_new(credential_hash, LinkTypes::CredentialToRevocation)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(RevocationCheckResult {
            credential_id,
            status: RevocationStatus::Active,
            reason: None,
            checked_at: now,
        });
    }

    // Get the most recent revocation entry
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            let entry: RevocationEntry = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid revocation entry".into()
                )))?;

            // Check if suspension has expired
            let status = if entry.status == RevocationStatus::Suspended {
                if let Some(end) = entry.suspension_end {
                    if now >= end {
                        RevocationStatus::Active
                    } else {
                        RevocationStatus::Suspended
                    }
                } else {
                    RevocationStatus::Suspended
                }
            } else {
                entry.status.clone()
            };

            return Ok(RevocationCheckResult {
                credential_id,
                status,
                reason: Some(entry.reason),
                checked_at: now,
            });
        }
    }

    Ok(RevocationCheckResult {
        credential_id,
        status: RevocationStatus::Active,
        reason: None,
        checked_at: now,
    })
}

/// Batch check multiple credentials
#[hdk_extern]
pub fn batch_check_revocation(credential_ids: Vec<String>) -> ExternResult<Vec<RevocationCheckResult>> {
    // Input validation
    if credential_ids.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest("Batch check must not exceed 100 credential IDs".into())));
    }

    let mut results = Vec::new();
    for id in credential_ids {
        results.push(check_revocation_status(id)?);
    }
    Ok(results)
}

/// Get all revocations by issuer
#[hdk_extern]
pub fn get_revocations_by_issuer(issuer_did: String) -> ExternResult<Vec<Record>> {
    let issuer_hash = string_to_entry_hash(&issuer_did);
    let links = get_links(
        LinkQuery::try_new(issuer_hash, LinkTypes::IssuerToRevocation)?,
        GetStrategy::default(),
    )?;

    let mut revocations = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            revocations.push(record);
        }
    }

    Ok(revocations)
}

// ==================== REVOCATION LIST MANAGEMENT ====================

/// Create a new revocation list for an issuer
#[hdk_extern]
pub fn create_revocation_list(input: CreateRevocationListInput) -> ExternResult<Record> {
    if input.id.is_empty() || input.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("List ID must be 1-256 characters".into())));
    }
    if input.issuer_did.is_empty() || !input.issuer_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer must be a valid DID".into())));
    }

    let now = sys_time()?;

    let list = RevocationList {
        id: input.id.clone(),
        issuer: input.issuer_did.clone(),
        revoked: Vec::new(),
        updated: now,
        version: 1,
    };

    let action_hash = create_entry(&EntryTypes::RevocationList(list))?;

    // Link issuer to their revocation list
    let issuer_hash = string_to_entry_hash(&input.issuer_did);
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::IssuerToRevocationList,
        input.id.as_bytes().to_vec(),
    )?;

    // Also link by list ID for direct lookup
    let list_hash = string_to_entry_hash(&input.id);
    create_link(
        list_hash,
        action_hash.clone(),
        LinkTypes::IssuerToRevocationList,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find revocation list".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRevocationListInput {
    pub id: String,
    pub issuer_did: String,
}

/// Get a revocation list by its ID
#[hdk_extern]
pub fn get_revocation_list(list_id: String) -> ExternResult<Option<Record>> {
    let list_hash = string_to_entry_hash(&list_id);
    let links = get_links(
        LinkQuery::try_new(list_hash, LinkTypes::IssuerToRevocationList)?,
        GetStrategy::default(),
    )?;

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Get all revocation lists for an issuer
#[hdk_extern]
pub fn get_issuer_revocation_lists(issuer_did: String) -> ExternResult<Vec<Record>> {
    let issuer_hash = string_to_entry_hash(&issuer_did);
    let links = get_links(
        LinkQuery::try_new(issuer_hash, LinkTypes::IssuerToRevocationList)?,
        GetStrategy::default(),
    )?;

    let mut lists = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            lists.push(record);
        }
    }

    Ok(lists)
}

// ==================== BATCH OPERATIONS ====================

/// Batch revoke multiple credentials, updating the revocation list
#[hdk_extern]
pub fn batch_revoke_credentials(input: BatchRevokeInput) -> ExternResult<BatchRevokeResult> {
    if input.credential_ids.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Must provide at least one credential ID".into())));
    }
    if input.credential_ids.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest("Batch revoke must not exceed 100 credentials".into())));
    }
    if input.issuer_did.is_empty() || !input.issuer_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer must be a valid DID".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }

    let now = sys_time()?;
    let mut revoked = Vec::new();
    let mut failed: Vec<BatchItemError> = Vec::new();

    for credential_id in &input.credential_ids {
        let result = revoke_credential(RevokeInput {
            credential_id: credential_id.clone(),
            issuer_did: input.issuer_did.clone(),
            reason: input.reason.clone(),
            effective_from: input.effective_from,
        });

        match result {
            Ok(_record) => revoked.push(credential_id.clone()),
            Err(e) => failed.push(BatchItemError {
                credential_id: credential_id.clone(),
                error: format!("{}", e),
            }),
        }
    }

    // Update revocation list if specified
    if let Some(ref list_id) = input.revocation_list_id {
        let _ = add_to_revocation_list(list_id, &input.issuer_did, &revoked, now);
    }

    Ok(BatchRevokeResult {
        revoked_count: revoked.len() as u32,
        failed_count: failed.len() as u32,
        revoked,
        failed,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BatchRevokeInput {
    pub credential_ids: Vec<String>,
    pub issuer_did: String,
    pub reason: String,
    pub effective_from: Option<Timestamp>,
    pub revocation_list_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BatchRevokeResult {
    pub revoked_count: u32,
    pub failed_count: u32,
    pub revoked: Vec<String>,
    pub failed: Vec<BatchItemError>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BatchItemError {
    pub credential_id: String,
    pub error: String,
}

/// Batch suspend multiple credentials
#[hdk_extern]
pub fn batch_suspend_credentials(input: BatchSuspendInput) -> ExternResult<BatchRevokeResult> {
    if input.credential_ids.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Must provide at least one credential ID".into())));
    }
    if input.credential_ids.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest("Batch suspend must not exceed 100 credentials".into())));
    }
    if input.issuer_did.is_empty() || !input.issuer_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer must be a valid DID".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }

    let mut revoked = Vec::new();
    let mut failed: Vec<BatchItemError> = Vec::new();

    for credential_id in &input.credential_ids {
        let result = suspend_credential(SuspendInput {
            credential_id: credential_id.clone(),
            issuer_did: input.issuer_did.clone(),
            reason: input.reason.clone(),
            suspension_end: input.suspension_end,
        });

        match result {
            Ok(_record) => revoked.push(credential_id.clone()),
            Err(e) => failed.push(BatchItemError {
                credential_id: credential_id.clone(),
                error: format!("{}", e),
            }),
        }
    }

    Ok(BatchRevokeResult {
        revoked_count: revoked.len() as u32,
        failed_count: failed.len() as u32,
        revoked,
        failed,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BatchSuspendInput {
    pub credential_ids: Vec<String>,
    pub issuer_did: String,
    pub reason: String,
    pub suspension_end: Timestamp,
}

/// Internal: add credential IDs to an existing revocation list
fn add_to_revocation_list(
    list_id: &str,
    issuer_did: &str,
    credential_ids: &[String],
    now: Timestamp,
) -> ExternResult<()> {
    let list_hash = string_to_entry_hash(list_id);
    let links = get_links(
        LinkQuery::try_new(list_hash, LinkTypes::IssuerToRevocationList)?,
        GetStrategy::default(),
    )?;

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    let action_hash = match latest_link {
        Some(link) => ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?,
        None => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Revocation list '{}' not found", list_id)
        ))),
    };

    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Revocation list record not found".into())))?;

    let mut list: RevocationList = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid revocation list entry".into())))?;

    // Verify issuer owns this list
    if list.issuer != issuer_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the list owner can update it".into()
        )));
    }

    // Add new credential IDs (avoid duplicates)
    for id in credential_ids {
        if !list.revoked.contains(id) {
            list.revoked.push(id.clone());
        }
    }
    list.updated = now;
    list.version += 1;

    update_entry(action_hash, &EntryTypes::RevocationList(list))?;
    Ok(())
}
