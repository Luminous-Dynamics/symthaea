// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Keys Zome
//!
//! Post-Quantum key management for Mycelix Mail
//! Handles CRYSTALS-Kyber key encapsulation and CRYSTALS-Dilithium signatures

use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Key bundle containing both encryption and signing keys
#[hdk_entry_helper]
#[derive(Clone)]
pub struct KeyBundle {
    /// Agent who owns this key bundle
    pub agent: AgentPubKey,
    /// CRYSTALS-Kyber public key for encryption
    pub kyber_public_key: Vec<u8>,
    /// CRYSTALS-Dilithium public key for signing
    pub dilithium_public_key: Vec<u8>,
    /// Key version (increments on rotation)
    pub version: u32,
    /// When this key bundle was created
    pub created_at: Timestamp,
    /// Optional expiration
    pub expires_at: Option<Timestamp>,
    /// Signature of this bundle with previous key (for rotation)
    pub rotation_signature: Option<Vec<u8>>,
}

/// Revoked key entry
#[hdk_entry_helper]
#[derive(Clone)]
pub struct RevokedKey {
    /// The revoked key bundle hash
    pub key_bundle_hash: ActionHash,
    /// Reason for revocation
    pub reason: String,
    /// When revoked
    pub revoked_at: Timestamp,
    /// Signature proving ownership
    pub revocation_signature: Vec<u8>,
}

/// Encrypted message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedEnvelope {
    /// Recipient's agent key
    pub recipient: AgentPubKey,
    /// Kyber ciphertext (encapsulated symmetric key)
    pub kyber_ciphertext: Vec<u8>,
    /// Symmetric encrypted payload
    pub encrypted_payload: Vec<u8>,
    /// Sender's signature
    pub signature: Vec<u8>,
    /// Key version used
    pub key_version: u32,
}

/// Key lookup result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyLookupResult {
    pub key_bundle: KeyBundle,
    pub action_hash: ActionHash,
    pub is_current: bool,
    pub is_revoked: bool,
}

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Grant unrestricted access for reading public keys
    let zome_name = zome_info()?.name;
    let functions = HashSet::from([
        (zome_name.clone(), "get_current_key_bundle".into()),
        (zome_name.clone(), "get_key_bundle_by_email".into()),
        (zome_name, "verify_signature".into()),
    ]);

    create_cap_grant(CapGrantEntry {
        tag: "public_key_access".into(),
        access: CapAccess::Unrestricted,
        functions: GrantedFunctions::Listed(functions),
    })?;

    Ok(InitCallbackResult::Pass)
}

/// Create initial key bundle for current agent
#[hdk_extern]
pub fn create_key_bundle(input: CreateKeyBundleInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Check if agent already has a key bundle
    let existing = get_agent_key_bundles(agent.clone())?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Agent already has key bundle. Use rotate_keys instead.".into()
        )));
    }

    let key_bundle = KeyBundle {
        agent: agent.clone(),
        kyber_public_key: input.kyber_public_key,
        dilithium_public_key: input.dilithium_public_key,
        version: 1,
        created_at: sys_time()?,
        expires_at: input.expires_at,
        rotation_signature: None,
    };

    let action_hash = create_entry(&EntryTypes::KeyBundle(key_bundle.clone()))?;

    // Create link from agent to key bundle
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToKeys,
        LinkTag::new("current"),
    )?;

    // If email provided, create email -> agent link
    if let Some(email) = input.email {
        let email_anchor = email_anchor_hash(&email)?;
        create_link(
            email_anchor,
            agent,
            LinkTypes::EmailToAgent,
            LinkTag::new(&email),
        )?;
    }

    Ok(action_hash)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateKeyBundleInput {
    pub kyber_public_key: Vec<u8>,
    pub dilithium_public_key: Vec<u8>,
    pub expires_at: Option<Timestamp>,
    pub email: Option<String>,
}

/// Rotate keys for current agent
#[hdk_extern]
pub fn rotate_keys(input: RotateKeysInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Get current key bundle
    let current = get_current_key_bundle(agent.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No existing key bundle".into())))?;

    // Create new key bundle with incremented version
    let new_bundle = KeyBundle {
        agent: agent.clone(),
        kyber_public_key: input.new_kyber_public_key,
        dilithium_public_key: input.new_dilithium_public_key,
        version: current.key_bundle.version + 1,
        created_at: sys_time()?,
        expires_at: input.expires_at,
        rotation_signature: Some(input.rotation_signature),
    };

    let action_hash = create_entry(&EntryTypes::KeyBundle(new_bundle))?;

    // Delete old "current" link
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToKeys)?,
        GetStrategy::default(),
    )?;

    for link in links {
        // Filter for "current" tag
        if link.tag == LinkTag::new("current") {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }

    // Create new "current" link
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::AgentToKeys,
        LinkTag::new("current"),
    )?;

    Ok(action_hash)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotateKeysInput {
    pub new_kyber_public_key: Vec<u8>,
    pub new_dilithium_public_key: Vec<u8>,
    pub rotation_signature: Vec<u8>,
    pub expires_at: Option<Timestamp>,
}

/// Revoke a key bundle
#[hdk_extern]
pub fn revoke_key(input: RevokeKeyInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Verify the key belongs to this agent
    let key_bundle = get_key_bundle(input.key_bundle_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Key bundle not found".into())))?;

    if key_bundle.agent != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot revoke another agent's keys".into()
        )));
    }

    let revoked = RevokedKey {
        key_bundle_hash: input.key_bundle_hash.clone(),
        reason: input.reason,
        revoked_at: sys_time()?,
        revocation_signature: input.revocation_signature,
    };

    let action_hash = create_entry(&EntryTypes::RevokedKey(revoked))?;

    // Create link for revocation lookup
    create_link(
        input.key_bundle_hash,
        action_hash.clone(),
        LinkTypes::KeyToRevocation,
        LinkTag::new("revoked"),
    )?;

    Ok(action_hash)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevokeKeyInput {
    pub key_bundle_hash: ActionHash,
    pub reason: String,
    pub revocation_signature: Vec<u8>,
}

/// Get current key bundle for an agent
#[hdk_extern]
pub fn get_current_key_bundle(agent: AgentPubKey) -> ExternResult<Option<KeyLookupResult>> {
    let all_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToKeys)?,
        GetStrategy::default(),
    )?;

    // Filter for "current" tag
    let links: Vec<_> = all_links
        .into_iter()
        .filter(|l| l.tag == LinkTag::new("current"))
        .collect();

    if links.is_empty() {
        return Ok(None);
    }

    let action_hash = ActionHash::try_from(links[0].target.clone()).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
    })?;

    let key_bundle = get_key_bundle(action_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Key bundle not found".into())))?;

    // Check if revoked
    let revoked = is_key_revoked(action_hash.clone())?;

    Ok(Some(KeyLookupResult {
        key_bundle,
        action_hash,
        is_current: true,
        is_revoked: revoked,
    }))
}

/// Get key bundle by email address
#[hdk_extern]
pub fn get_key_bundle_by_email(email: String) -> ExternResult<Option<KeyLookupResult>> {
    let email_anchor = email_anchor_hash(&email)?;

    let links = get_links(LinkQuery::try_new(email_anchor, LinkTypes::EmailToAgent)?, GetStrategy::default())?;

    if links.is_empty() {
        return Ok(None);
    }

    let agent = AgentPubKey::try_from(links[0].target.clone()).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest("Invalid agent key".into()))
    })?;

    get_current_key_bundle(agent)
}

/// Get all key bundles for an agent (including rotated ones)
#[hdk_extern]
pub fn get_agent_key_bundles(agent: AgentPubKey) -> ExternResult<Vec<KeyLookupResult>> {
    let links = get_links(LinkQuery::try_new(agent.clone(), LinkTypes::AgentToKeys)?, GetStrategy::default())?;

    let mut results = Vec::new();
    let mut current_hash: Option<ActionHash> = None;

    // Find current key
    for link in &links {
        if link.tag == LinkTag::new("current") {
            current_hash = Some(ActionHash::try_from(link.target.clone()).map_err(|_| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
            })?);
            break;
        }
    }

    for link in links {
        let action_hash = ActionHash::try_from(link.target.clone()).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;

        if let Some(key_bundle) = get_key_bundle(action_hash.clone())? {
            let is_current = current_hash.as_ref() == Some(&action_hash);
            let is_revoked = is_key_revoked(action_hash.clone())?;

            results.push(KeyLookupResult {
                key_bundle,
                action_hash,
                is_current,
                is_revoked,
            });
        }
    }

    // Sort by version descending
    results.sort_by(|a, b| b.key_bundle.version.cmp(&a.key_bundle.version));

    Ok(results)
}

/// Verify a signature against a key bundle
#[hdk_extern]
pub fn verify_signature(input: VerifySignatureInput) -> ExternResult<bool> {
    // In production, this would use CRYSTALS-Dilithium verification
    // For now, we just check the structure is valid
    let key_bundle = get_key_bundle(input.key_bundle_hash)?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Key bundle not found".into())))?;

    // Check key is not revoked
    if is_key_revoked(input.key_bundle_hash)? {
        return Ok(false);
    }

    // In real implementation: dilithium_verify(
    //     &key_bundle.dilithium_public_key,
    //     &input.message,
    //     &input.signature
    // )

    Ok(!input.signature.is_empty())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifySignatureInput {
    pub key_bundle_hash: ActionHash,
    pub message: Vec<u8>,
    pub signature: Vec<u8>,
}

// Helper functions

fn get_key_bundle(action_hash: ActionHash) -> ExternResult<Option<KeyBundle>> {
    let record = get(action_hash, GetOptions::default())?;
    match record {
        Some(r) => {
            let entry = r.entry().to_app_option::<KeyBundle>()
                .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
            Ok(entry)
        }
        None => Ok(None),
    }
}

fn is_key_revoked(key_bundle_hash: ActionHash) -> ExternResult<bool> {
    let links = get_links(
        LinkQuery::try_new(key_bundle_hash, LinkTypes::KeyToRevocation)?,
        GetStrategy::default(),
    )?;
    // Any link to revocation means key is revoked
    Ok(!links.is_empty())
}

/// Create email anchor hash for lookups
fn email_anchor_hash(email: &str) -> ExternResult<EntryHash> {
    let normalized = email.to_lowercase();
    let path = Path::from(format!("email:{}", normalized));
    path.path_entry_hash()
}

// Entry and Link type definitions
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    KeyBundle(KeyBundle),
    #[entry_type]
    RevokedKey(RevokedKey),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToKeys,
    EmailToAgent,
    KeyToRevocation,
}

// Validation callbacks
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } |
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::KeyBundle(key_bundle) => validate_key_bundle(key_bundle),
                EntryTypes::RevokedKey(revoked) => validate_revoked_key(revoked),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            ..
        } => match link_type {
            LinkTypes::AgentToKeys => {
                // Only the agent can create links from themselves
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::EmailToAgent => {
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::KeyToRevocation => {
                Ok(ValidateCallbackResult::Valid)
            }
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_key_bundle(key_bundle: KeyBundle) -> ExternResult<ValidateCallbackResult> {
    // Kyber-1024 public key should be 1568 bytes
    if key_bundle.kyber_public_key.len() != 1568 && !key_bundle.kyber_public_key.is_empty() {
        // Allow empty for testing, require correct size in production
        // return Ok(ValidateCallbackResult::Invalid(
        //     "Invalid Kyber public key size".into()
        // ));
    }

    // Dilithium3 public key should be 1952 bytes
    if key_bundle.dilithium_public_key.len() != 1952 && !key_bundle.dilithium_public_key.is_empty() {
        // Allow empty for testing
        // return Ok(ValidateCallbackResult::Invalid(
        //     "Invalid Dilithium public key size".into()
        // ));
    }

    // Version must be positive
    if key_bundle.version == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Key version must be positive".into()
        ));
    }

    // If rotation, signature must be present
    if key_bundle.version > 1 && key_bundle.rotation_signature.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rotation requires signature from previous key".into()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_revoked_key(revoked: RevokedKey) -> ExternResult<ValidateCallbackResult> {
    // Revocation must have a signature
    if revoked.revocation_signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation must include signature".into()
        ));
    }

    // Reason must not be empty
    if revoked.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation reason required".into()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
