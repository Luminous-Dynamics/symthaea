//! Claim Verification Coordinator Zome
//!
//! Provides CRUD operations for claim verification.

use hdk::prelude::*;
use verification_integrity::*;

// NOTE: We create links to claims using a cross-zome pattern
// The claims zome owns the ClaimToVerifications link type

/// Input for creating a claim verification
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateVerificationInput {
    pub claim_hash: ActionHash,
    pub verifier: String,
    pub status: VerificationStatus,
}

/// Create a new claim verification
#[hdk_extern]
pub fn create_verification(input: CreateVerificationInput) -> ExternResult<Record> {
    // Input validation
    if input.verifier.is_empty() || input.verifier.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verifier must be 1-200 characters".to_string()
        )));
    }

    // Verify claim exists
    if get(input.claim_hash.clone(), GetOptions::default())?.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Claim not found".to_string()
        )));
    }

    let now = sys_time()?;

    let verification = ClaimVerification {
        claim_hash: input.claim_hash.clone(),
        verifier: input.verifier.clone(),
        status: input.status,
        verified_at: now.as_micros() as u64,
    };

    let action_hash = create_entry(EntryTypes::ClaimVerification(verification.clone()))?;

    // Link from verifier to verification
    let verifier_hash = hash_identifier(&input.verifier)?;
    create_link(
        verifier_hash,
        action_hash.clone(),
        LinkTypes::VerifierToVerifications,
        (),
    )?;

    // NOTE: Link from claim to verification would use claims zome's LinkTypes
    // For now, we'll only create the verifier-to-verification link
    // In production, would use call() to create the claim link in the claims zome

    // Link to all verifications anchor
    let all_anchor = all_verifications_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllVerifications, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created verification".to_string()
    )))
}

/// Get a verification by its action hash
#[hdk_extern]
pub fn get_verification(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all verifications for a claim
/// NOTE: This is a simplified version. In production, would query via claims zome
#[hdk_extern]
pub fn get_verifications_for_claim(_claim_hash: ActionHash) -> ExternResult<Vec<Record>> {
    // Placeholder - in production, would call claims zome to get verifications
    // For now, return empty list
    Ok(Vec::new())
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

    let verifier = agent_info()?.agent_initial_pubkey.to_string();

    let verification_input = CreateVerificationInput {
        claim_hash,
        verifier,
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
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "Verification not found".to_string()
        )))?;

    let verification: ClaimVerification = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "Invalid verification entry".to_string()
        )))?;

    Ok(verification.status == VerificationStatus::Verified)
}

/// Get verification status for a claim
#[hdk_extern]
pub fn get_verification_status(claim_hash: ActionHash) -> ExternResult<String> {
    let verifications = get_verifications_for_claim(claim_hash)?;

    if verifications.is_empty() {
        return Ok("Unverified".to_string());
    }

    let verified_count = verifications.iter()
        .filter(|r| {
            if let Some(v) = r.entry().to_app_option::<ClaimVerification>().ok().flatten() {
                v.status == VerificationStatus::Verified
            } else {
                false
            }
        })
        .count();

    let rejected_count = verifications.iter()
        .filter(|r| {
            if let Some(v) = r.entry().to_app_option::<ClaimVerification>().ok().flatten() {
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
        format!("anchor:{}", identifier).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get the anchor for all verifications
fn all_verifications_anchor() -> ExternResult<EntryHash> {
    hash_identifier("all_verifications")
}
