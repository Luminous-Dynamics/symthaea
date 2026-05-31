// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Revocation Registry Coordinator Zome
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential ID must be 1-256 characters".into()
        )));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-4096 characters".into()
        )));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential ID must be 1-256 characters".into()
        )));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-4096 characters".into()
        )));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential ID must be 1-256 characters".into()
        )));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-4096 characters".into()
        )));
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

    let current_record = get(current_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Revocation entry not found".into())),
    )?;

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

    let action_hash = update_entry(
        current_action_hash,
        &EntryTypes::RevocationEntry(reinstated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
pub fn batch_check_revocation(
    credential_ids: Vec<String>,
) -> ExternResult<Vec<RevocationCheckResult>> {
    // Input validation
    if credential_ids.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Batch check must not exceed 100 credential IDs".into()
        )));
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
