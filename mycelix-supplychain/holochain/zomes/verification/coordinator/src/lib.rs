// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim Verification Coordinator Zome
//!
//! Provides CRUD operations for claim verification.

use hdk::prelude::*;
use mycelix_zome_helpers as _;
use verification_integrity::*;

// NOTE: We create links to claims using a cross-zome pattern
// The claims zome owns the ClaimToVerifications link type

/// Input for creating a claim verification
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateVerificationInput {
    pub claim_hash: ActionHash,
    pub status: VerificationStatus,
}

/// Create a new claim verification
///
/// Authorization: `verifier` is *not* a caller-supplied field. Historically this
/// function accepted an arbitrary `verifier: String` in `CreateVerificationInput`,
/// which meant any agent could attribute a verification to any other identity
/// (unbound free text — see `verification/integrity/src/lib.rs`'s `ClaimVerification`
/// entry, which only checks the string is non-empty, not that it corresponds to the
/// calling agent). The verifier is now always derived from `agent_info()`, matching
/// the pattern already used by `submit_proof` below.
#[hdk_extern]
pub fn create_verification(input: CreateVerificationInput) -> ExternResult<Record> {
    let verifier = agent_info()?.agent_initial_pubkey.to_string();

    // Verify claim exists
    if get(input.claim_hash.clone(), GetOptions::default())?.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Claim not found".to_string()
        )));
    }

    let now = sys_time()?;

    let verification = ClaimVerification {
        claim_hash: input.claim_hash.clone(),
        verifier: verifier.clone(),
        status: input.status,
        verified_at: now.as_micros() as u64,
    };

    let action_hash = create_entry(EntryTypes::ClaimVerification(verification.clone()))?;

    // Link from verifier to verification
    let verifier_hash = hash_identifier(&verifier)?;
    create_link(
        verifier_hash,
        action_hash.clone(),
        LinkTypes::VerifierToVerifications,
        (),
    )?;

    // Link to all verifications anchor
    let all_anchor = all_verifications_anchor()?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllVerifications,
        (),
    )?;

    // Cross-zome: link this verification to the claim in the claims zome.
    let link_input = serde_json::json!({
        "claim_hash": input.claim_hash,
        "verification_hash": action_hash.clone(),
    });
    let _ = call(
        CallTargetCell::Local,
        ZomeName::from("claims_coordinator"),
        FunctionName::from("link_verification_to_claim"),
        None,
        ExternIO::encode(link_input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    );

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created verification".to_string()
    )))
}

/// Get a verification by its action hash
#[hdk_extern]
pub fn get_verification(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all verifications for a claim by calling the claims coordinator cross-zome.
///
/// The claims zome owns the `ClaimToVerifications` link type and returns
/// a `(Record, Vec<Record>)` tuple from `get_claim_with_verifications`.
/// We extract just the verifications (index 1 of the tuple).
#[hdk_extern]
pub fn get_verifications_for_claim(claim_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let payload = ExternIO::encode(claim_hash)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let response = call(
        CallTargetCell::Local,
        ZomeName::from("claims_coordinator"),
        FunctionName::from("get_claim_with_verifications"),
        None,
        payload,
    )?;

    match response {
        ZomeCallResponse::Ok(data) => {
            // get_claim_with_verifications returns (Record, Vec<Record>)
            // Decode as a JSON value; the second element is the verification list.
            let value: serde_json::Value = data
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

            // The tuple is serialized as a two-element JSON array.
            if let Some(arr) = value.as_array() {
                if arr.len() == 2 {
                    // The second element is the Vec<Record>; we re-encode it and
                    // decode as Vec<Record> to get properly typed records.
                    let verifications_json = arr[1].clone();
                    let verifications: Vec<Record> = serde_json::from_value(verifications_json)
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
                    return Ok(verifications);
                }
            }
            Ok(Vec::new())
        }
        // Claim not found or claims zome unavailable — return empty list gracefully.
        _ => Ok(Vec::new()),
    }
}

/// Get all verifications by a verifier
#[hdk_extern]
pub fn get_verifications_by_verifier(verifier: String) -> ExternResult<Vec<Record>> {
    let verifier_hash = hash_identifier(&verifier)?;
    let links = get_links(
        LinkQuery::try_new(verifier_hash, LinkTypes::VerifierToVerifications)?,
        GetStrategy::default(),
    )?;

    let mut verifications = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                verifications.push(record);
            }
        }
    }

    Ok(verifications)
}

/// Get all verifications
#[hdk_extern]
pub fn get_all_verifications(limit: u32) -> ExternResult<Vec<Record>> {
    if limit == 0 || limit > 1000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Limit must be between 1 and 1000".to_string()
        )));
    }

    let anchor = all_verifications_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllVerifications)?,
        GetStrategy::default(),
    )?;

    let mut verifications = Vec::new();
    for link in links.into_iter().take(limit as usize) {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                verifications.push(record);
            }
        }
    }

    Ok(verifications)
}

/// Submit proof of verification
#[hdk_extern]
pub fn submit_proof(input: (ActionHash, String)) -> ExternResult<ActionHash> {
    let (claim_hash, proof_data) = input;

    if proof_data.is_empty() || proof_data.len() > 10_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proof data must be 1-10000 characters".to_string()
        )));
    }

    let verification_input = CreateVerificationInput {
        claim_hash,
        status: VerificationStatus::Verified,
    };

    let record = create_verification(verification_input)?;
    let action_hash = record.action_address().clone();
    Ok(action_hash)
}

/// Verify a proof
#[hdk_extern]
pub fn verify_proof(verification_hash: ActionHash) -> ExternResult<bool> {
    let record = get(verification_hash, GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Verification not found".to_string())))?;

    let verification: ClaimVerification = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "Invalid verification entry".to_string()
            ))
        })?;

    Ok(verification.status == VerificationStatus::Verified)
}

/// Get verification status for a claim
#[hdk_extern]
pub fn get_verification_status(claim_hash: ActionHash) -> ExternResult<String> {
    let verifications = get_verifications_for_claim(claim_hash)?;

    if verifications.is_empty() {
        return Ok("Unverified".to_string());
    }

    let verified_count = verifications
        .iter()
        .filter(|r| {
            if let Some(v) = r
                .entry()
                .to_app_option::<ClaimVerification>()
                .ok()
                .flatten()
            {
                v.status == VerificationStatus::Verified
            } else {
                false
            }
        })
        .count();

    let rejected_count = verifications
        .iter()
        .filter(|r| {
            if let Some(v) = r
                .entry()
                .to_app_option::<ClaimVerification>()
                .ok()
                .flatten()
            {
                v.status == VerificationStatus::Rejected
            } else {
                false
            }
        })
        .count();

    if rejected_count > 0 {
        Ok("Rejected".to_string())
    } else if verified_count > 0 {
        Ok(format!("Verified ({} verifications)", verified_count))
    } else {
        Ok("Pending".to_string())
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create a deterministic hash from a string identifier
fn hash_identifier(identifier: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", identifier).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get the anchor for all verifications
fn all_verifications_anchor() -> ExternResult<EntryHash> {
    hash_identifier("all_verifications")
}
