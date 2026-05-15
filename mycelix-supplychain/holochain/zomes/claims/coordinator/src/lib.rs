// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Supply Chain Claims Coordinator Zome
//!
//! Provides CRUD operations for supply chain claims and provider profiles.

use claims_integrity::*;
use hdk::prelude::*;
use mycelix_zome_helpers as _;

/// Input for creating a supply chain claim
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateClaimInput {
    pub item_id: String,
    pub claim_type: String,
    pub data: String,
    pub issuer: String,
    pub previous_claim: Option<ActionHash>,
}

/// Create a new supply chain claim with provenance chain linking
#[hdk_extern]
pub fn create_claim(input: CreateClaimInput) -> ExternResult<Record> {
    // Input validation
    if input.item_id.is_empty() || input.item_id.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Item ID must be 1-200 characters".to_string()
        )));
    }
    if input.claim_type.is_empty() || input.claim_type.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Claim type must be 1-100 characters".to_string()
        )));
    }
    if input.data.len() > 10_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Claim data cannot exceed 10KB".to_string()
        )));
    }
    if input.issuer.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer is required".to_string()
        )));
    }

    let now = sys_time()?;

    let claim = SupplyChainClaim {
        item_id: input.item_id.clone(),
        claim_type: input.claim_type,
        data: input.data,
        issuer: input.issuer.clone(),
        timestamp: now.as_micros() as u64,
    };

    let action_hash = create_entry(EntryTypes::SupplyChainClaim(claim.clone()))?;

    // Link from issuer to claim
    let issuer_hash = hash_identifier(&input.issuer)?;
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::ProviderToClaims,
        (),
    )?;

    // Link from item_id to claim
    let item_hash = hash_identifier(&input.item_id)?;
    create_link(item_hash, action_hash.clone(), LinkTypes::ItemToClaims, ())?;

    // Link to all claims anchor
    let all_anchor = all_claims_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllClaims, ())?;

    // PROVENANCE CHAIN: Link to previous claim if provided
    if let Some(prev_hash) = input.previous_claim {
        // Verify previous claim exists
        if get(prev_hash.clone(), GetOptions::default())?.is_none() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Previous claim not found".to_string()
            )));
        }

        // Create provenance chain link
        create_link(
            prev_hash,
            action_hash.clone(),
            LinkTypes::ClaimToVerifications,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created claim".to_string()
    )))
}

/// Get a claim by its action hash
#[hdk_extern]
pub fn get_claim(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all claims for an item
#[hdk_extern]
pub fn get_claims_by_item(item_id: String) -> ExternResult<Vec<Record>> {
    let item_hash = hash_identifier(&item_id)?;
    let links = get_links(
        LinkQuery::try_new(item_hash, LinkTypes::ItemToClaims)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                claims.push(record);
            }
        }
    }

    Ok(claims)
}

/// Get all claims by a provider
#[hdk_extern]
pub fn get_claims_by_provider(provider_did: String) -> ExternResult<Vec<Record>> {
    let provider_hash = hash_identifier(&provider_did)?;
    let links = get_links(
        LinkQuery::try_new(provider_hash, LinkTypes::ProviderToClaims)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                claims.push(record);
            }
        }
    }

    Ok(claims)
}

/// Get all claims
#[hdk_extern]
pub fn get_all_claims(limit: u32) -> ExternResult<Vec<Record>> {
    let anchor = all_claims_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllClaims)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links.into_iter().take(limit as usize) {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                claims.push(record);
            }
        }
    }

    Ok(claims)
}

/// Input for creating a provider profile
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProviderInput {
    pub provider_did: String,
    pub name: String,
    pub certifications: Vec<String>,
}

/// Create a provider profile
#[hdk_extern]
pub fn create_provider_profile(input: CreateProviderInput) -> ExternResult<Record> {
    let profile = ProviderProfile {
        provider_did: input.provider_did,
        name: input.name,
        certifications: input.certifications,
    };

    let action_hash = create_entry(EntryTypes::ProviderProfile(profile))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created provider profile".to_string()
    )))
}

/// Get a provider profile by action hash
#[hdk_extern]
pub fn get_provider_profile(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get provenance chain for an item (all claims in chronological order)
#[hdk_extern]
pub fn get_item_provenance_chain(item_id: String) -> ExternResult<Vec<Record>> {
    if item_id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Item ID is required".to_string()
        )));
    }

    let all_claims = get_claims_by_item(item_id)?;

    // Sort claims by timestamp to build chronological chain
    let mut sorted_claims: Vec<(Record, u64)> = Vec::new();
    for claim_record in all_claims {
        if let Some(claim) = claim_record
            .entry()
            .to_app_option::<SupplyChainClaim>()
            .map_err(|e| wasm_error!(e))?
        {
            sorted_claims.push((claim_record, claim.timestamp));
        }
    }

    sorted_claims.sort_by_key(|(_, timestamp)| *timestamp);
    let chain: Vec<Record> = sorted_claims
        .into_iter()
        .map(|(record, _)| record)
        .collect();

    Ok(chain)
}

/// Verify claim authenticity by checking issuer signature (placeholder)
#[hdk_extern]
pub fn verify_claim_authenticity(claim_hash: ActionHash) -> ExternResult<bool> {
    // Verify claim exists
    let record = get(claim_hash, GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    // Get claim data
    let claim: SupplyChainClaim = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid claim entry".to_string())))?;

    // In production, would verify signature against issuer DID
    // For now, return true if claim has valid structure
    Ok(!claim.issuer.is_empty() && !claim.item_id.is_empty())
}

/// Input for linking a verification back to its parent claim.
#[derive(Serialize, Deserialize, Debug)]
pub struct LinkVerificationInput {
    pub claim_hash: ActionHash,
    pub verification_hash: ActionHash,
}

/// Called by the verification coordinator via cross-zome `call()` to register
/// a newly-created verification under the claim it belongs to.
#[hdk_extern]
pub fn link_verification_to_claim(input: LinkVerificationInput) -> ExternResult<ActionHash> {
    // Verify the claim exists before creating the link
    if get(input.claim_hash.clone(), GetOptions::default())?.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Claim not found".to_string()
        )));
    }

    create_link(
        input.claim_hash,
        input.verification_hash,
        LinkTypes::ClaimToVerifications,
        (),
    )
}

/// Get claim with its verification records
#[hdk_extern]
pub fn get_claim_with_verifications(claim_hash: ActionHash) -> ExternResult<(Record, Vec<Record>)> {
    let claim_record = get(claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    // Get verification records
    let links = get_links(
        LinkQuery::try_new(claim_hash, LinkTypes::ClaimToVerifications)?,
        GetStrategy::default(),
    )?;

    let mut verifications = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(verification) = get(hash, GetOptions::default())? {
                verifications.push(verification);
            }
        }
    }

    Ok((claim_record, verifications))
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

/// Get the anchor for all claims
fn all_claims_anchor() -> ExternResult<EntryHash> {
    hash_identifier("all_claims")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_verification_input_serde() {
        let input = LinkVerificationInput {
            claim_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            verification_hash: ActionHash::from_raw_36(vec![1u8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: LinkVerificationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.claim_hash, input.claim_hash);
        assert_eq!(back.verification_hash, input.verification_hash);
    }

    #[test]
    fn test_link_verification_input_hashes_differ() {
        let claim = ActionHash::from_raw_36(vec![0u8; 36]);
        let verification = ActionHash::from_raw_36(vec![1u8; 36]);
        assert_ne!(claim, verification);
    }

    #[test]
    fn test_create_claim_input_serde() {
        let input = CreateClaimInput {
            item_id: "ITEM-123".to_string(),
            claim_type: "PRODUCED".to_string(),
            data: r#"{"factory":"XYZ"}"#.to_string(),
            issuer: "did:example:producer".to_string(),
            previous_claim: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateClaimInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.item_id, "ITEM-123");
        assert_eq!(back.claim_type, "PRODUCED");
        assert!(back.previous_claim.is_none());
    }
}
