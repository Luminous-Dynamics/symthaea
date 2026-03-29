// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Keys Coordinator Zome
//!
//! Pre-key bundle management for end-to-end encryption in Mycelix Mail.
//!
//! # Overview
//!
//! This zome implements the Signal Protocol-style pre-key bundle system with
//! post-quantum cryptography support (Kyber + Dilithium). Key bundles enable
//! asynchronous message encryption when the recipient is offline.
//!
//! # Key Bundle Structure
//!
//! Each bundle contains:
//! - **Identity Key**: Long-term public key (Dilithium for post-quantum)
//! - **Signed Pre-Key**: Medium-term key signed by identity key
//! - **One-Time Pre-Keys**: Ephemeral keys for forward secrecy (Kyber)
//!
//! # Key Management Functions
//!
//! - [`publish_pre_key_bundle`] - Publish a new key bundle to the DHT
//! - [`get_pre_key_bundle`] - Retrieve an agent's current key bundle
//! - [`consume_pre_key`] - Use a one-time pre-key for encryption
//! - [`rotate_keys`] - Rotate keys with audit trail
//! - [`needs_refresh`] - Check if bundle needs refresh
//!
//! # Example Usage
//!
//! ```ignore
//! // Publish a pre-key bundle
//! let bundle = PreKeyBundle {
//!     identity_public_key: identity_key,
//!     signed_pre_key: signed_key,
//!     signed_pre_key_signature: signature,
//!     one_time_pre_keys: otpks,
//!     created_at: now,
//!     expires_at: now + 30_days,
//! };
//! let hash = publish_pre_key_bundle(bundle)?;
//!
//! // Get someone's bundle for encryption
//! let their_bundle = get_pre_key_bundle(recipient_agent)?;
//!
//! // Consume a one-time key
//! let otpk = consume_pre_key(ConsumePreKeyInput {
//!     bundle_hash: their_bundle_hash,
//!     key_id: 0,
//! })?;
//! ```
//!
//! # Security Considerations
//!
//! - One-time pre-keys provide forward secrecy
//! - Bundle expiration enforces key rotation
//! - Key rotation creates audit trail
//! - Post-quantum algorithms protect against future threats

use hdk::prelude::*;
use keys_integrity::*;

// ==================== BUNDLE MANAGEMENT ====================

/// Publish pre-key bundle
#[hdk_extern]
pub fn publish_pre_key_bundle(bundle: PreKeyBundle) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::PreKeyBundle(bundle))?;

    // Link from agent to bundle
    let my_agent = agent_info()?.agent_initial_pubkey;
    create_link(
        my_agent,
        action_hash.clone(),
        LinkTypes::AgentToBundle,
        (),
    )?;

    Ok(action_hash)
}

/// Get pre-key bundle for an agent
#[hdk_extern]
pub fn get_pre_key_bundle(agent: AgentPubKey) -> ExternResult<Option<PreKeyBundle>> {
    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToBundle)?, GetStrategy::default())?;

    // Get the most recent bundle
    let mut bundles: Vec<(u64, PreKeyBundle)> = Vec::new();
    let now = sys_time()?.as_micros() as u64;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(bundle) = record
                    .entry()
                    .to_app_option::<PreKeyBundle>()
                    .map_err(|e| wasm_error!(e))?
                {
                    // Only include non-expired bundles
                    if bundle.expires_at > now {
                        bundles.push((bundle.created_at, bundle));
                    }
                }
            }
        }
    }

    // Return the most recent bundle
    bundles.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(bundles.into_iter().next().map(|(_, b)| b))
}

/// Get my current bundle
#[hdk_extern]
pub fn get_my_bundle(_: ()) -> ExternResult<Option<PreKeyBundle>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    get_pre_key_bundle(my_agent)
}

/// Consume a one-time pre-key
#[hdk_extern]
pub fn consume_pre_key(input: ConsumePreKeyInput) -> ExternResult<Option<Vec<u8>>> {
    // Get the bundle
    if let Some(record) = get(input.bundle_hash.clone(), GetOptions::default())? {
        if let Some(mut bundle) = record
            .entry()
            .to_app_option::<PreKeyBundle>()
            .map_err(|e| wasm_error!(e))?
        {
            // Find the requested key
            for otpk in &mut bundle.one_time_pre_keys {
                if otpk.key_id == input.key_id && !otpk.used {
                    let key = otpk.public_key.clone();

                    // Mark as used
                    otpk.used = true;

                    // Update the bundle
                    update_entry(input.bundle_hash.clone(), EntryTypes::PreKeyBundle(bundle))?;

                    // Record the usage
                    let used = UsedPreKey {
                        key_id: input.key_id,
                        bundle_hash: input.bundle_hash,
                        used_at: sys_time()?.as_micros() as u64,
                        used_by: agent_info()?.agent_initial_pubkey,
                    };
                    create_entry(EntryTypes::UsedPreKey(used))?;

                    return Ok(Some(key));
                }
            }
        }
    }

    Ok(None)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsumePreKeyInput {
    pub bundle_hash: ActionHash,
    pub key_id: u32,
}

/// Get available one-time pre-key count
#[hdk_extern]
pub fn get_available_pre_key_count(agent: AgentPubKey) -> ExternResult<u32> {
    if let Some(bundle) = get_pre_key_bundle(agent)? {
        let count = bundle
            .one_time_pre_keys
            .iter()
            .filter(|k| !k.used)
            .count();
        Ok(count as u32)
    } else {
        Ok(0)
    }
}

// ==================== KEY ROTATION ====================

/// Rotate keys (publish new bundle)
#[hdk_extern]
pub fn rotate_keys(input: RotateKeysInput) -> ExternResult<ActionHash> {
    // Get old bundle hash
    let my_agent = agent_info()?.agent_initial_pubkey;
    let old_bundle_hash = get_my_bundle_hash()?;

    // Publish new bundle
    let new_bundle_hash = publish_pre_key_bundle(input.new_bundle)?;

    // Record rotation
    if let Some(old_hash) = old_bundle_hash {
        let rotation = KeyRotation {
            old_bundle_hash: old_hash,
            new_bundle_hash: new_bundle_hash.clone(),
            rotated_at: sys_time()?.as_micros() as u64,
            reason: input.reason,
        };
        let rotation_hash = create_entry(EntryTypes::KeyRotation(rotation))?;

        create_link(
            my_agent,
            rotation_hash,
            LinkTypes::KeyRotations,
            (),
        )?;
    }

    Ok(new_bundle_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotateKeysInput {
    pub new_bundle: PreKeyBundle,
    pub reason: RotationReason,
}

/// Get key rotation history
#[hdk_extern]
pub fn get_rotation_history(_: ()) -> ExternResult<Vec<KeyRotation>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(LinkQuery::try_new(my_agent, LinkTypes::KeyRotations)?, GetStrategy::default())?;

    let mut rotations = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(rotation) = record
                    .entry()
                    .to_app_option::<KeyRotation>()
                    .map_err(|e| wasm_error!(e))?
                {
                    rotations.push(rotation);
                }
            }
        }
    }

    rotations.sort_by(|a, b| b.rotated_at.cmp(&a.rotated_at));

    Ok(rotations)
}

// ==================== BUNDLE REFRESH ====================

/// Check if bundle needs refresh
#[hdk_extern]
pub fn needs_refresh(_: ()) -> ExternResult<BundleStatus> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.as_micros() as u64;

    if let Some(bundle) = get_pre_key_bundle(my_agent)? {
        let available_keys = bundle
            .one_time_pre_keys
            .iter()
            .filter(|k| !k.used)
            .count();

        // Check expiration (warn if expires within 24 hours)
        let one_day = 24 * 60 * 60 * 1_000_000; // microseconds
        if bundle.expires_at < now + one_day {
            return Ok(BundleStatus::ExpiringSoon);
        }

        // Check if running low on one-time keys
        if available_keys < 10 {
            return Ok(BundleStatus::LowOnKeys(available_keys as u32));
        }

        Ok(BundleStatus::Ok)
    } else {
        Ok(BundleStatus::NoBundle)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BundleStatus {
    Ok,
    NoBundle,
    ExpiringSoon,
    Expired,
    LowOnKeys(u32),
}

// ==================== HELPERS ====================

fn get_my_bundle_hash() -> ExternResult<Option<ActionHash>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(LinkQuery::try_new(my_agent, LinkTypes::AgentToBundle)?, GetStrategy::default())?;

    // Get most recent
    let mut bundles: Vec<(u64, ActionHash)> = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(bundle) = record
                    .entry()
                    .to_app_option::<PreKeyBundle>()
                    .map_err(|e| wasm_error!(e))?
                {
                    bundles.push((bundle.created_at, hash));
                }
            }
        }
    }

    bundles.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(bundles.into_iter().next().map(|(_, h)| h))
}
