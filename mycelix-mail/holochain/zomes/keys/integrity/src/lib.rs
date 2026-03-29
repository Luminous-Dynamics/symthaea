// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Keys Integrity Zome
//!
//! Pre-key bundles for E2E encryption key exchange.

use hdi::prelude::*;

/// Pre-key bundle for X3DH key exchange
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PreKeyBundle {
    /// Long-term identity key (public)
    pub identity_key: Vec<u8>,
    /// Signed pre-key (public)
    pub signed_pre_key: Vec<u8>,
    /// Signed pre-key ID
    pub signed_pre_key_id: u32,
    /// Signature of signed pre-key
    pub signed_pre_key_signature: Vec<u8>,
    /// One-time pre-keys (public)
    pub one_time_pre_keys: Vec<OneTimePreKey>,
    /// When the bundle was created
    pub created_at: u64,
    /// When the bundle expires
    pub expires_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OneTimePreKey {
    pub key_id: u32,
    pub public_key: Vec<u8>,
    pub used: bool,
}

/// Used pre-key record (to prevent reuse)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsedPreKey {
    pub key_id: u32,
    pub bundle_hash: ActionHash,
    pub used_at: u64,
    pub used_by: AgentPubKey,
}

/// Key rotation record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KeyRotation {
    pub old_bundle_hash: ActionHash,
    pub new_bundle_hash: ActionHash,
    pub rotated_at: u64,
    pub reason: RotationReason,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RotationReason {
    Scheduled,
    Compromised,
    PreKeysExhausted,
    Manual,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PreKeyBundle(PreKeyBundle),
    UsedPreKey(UsedPreKey),
    KeyRotation(KeyRotation),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToBundle,
    BundleToUsedKeys,
    KeyRotations,
}

/// Validate pre-key bundle
fn validate_create_pre_key_bundle(
    _action: Create,
    bundle: PreKeyBundle,
) -> ExternResult<ValidateCallbackResult> {
    // Validate identity key length (32 bytes for X25519)
    if bundle.identity_key.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Identity key must be 32 bytes".to_string(),
        ));
    }

    // Validate signed pre-key length
    if bundle.signed_pre_key.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Signed pre-key must be 32 bytes".to_string(),
        ));
    }

    // Validate signature length (64 bytes for Ed25519)
    if bundle.signed_pre_key_signature.len() != 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature must be 64 bytes".to_string(),
        ));
    }

    // Validate at least some one-time pre-keys
    if bundle.one_time_pre_keys.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Bundle must have at least one one-time pre-key".to_string(),
        ));
    }

    // Validate one-time pre-key lengths
    for otpk in &bundle.one_time_pre_keys {
        if otpk.public_key.len() != 32 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("One-time pre-key {} must be 32 bytes", otpk.key_id),
            ));
        }
    }

    // Validate expiration is after creation
    if bundle.expires_at <= bundle.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Expiration must be after creation time".to_string(),
        ));
    }

    // Note: Expiration check against current time is done in coordinator zome
    // since sys_time() is not available in integrity zomes

    Ok(ValidateCallbackResult::Valid)
}

/// Validate used pre-key record
fn validate_create_used_pre_key(
    _action: Create,
    used: UsedPreKey,
) -> ExternResult<ValidateCallbackResult> {
    // Basic validation
    if used.key_id == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Key ID cannot be 0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate key rotation
fn validate_create_key_rotation(
    _action: Create,
    rotation: KeyRotation,
) -> ExternResult<ValidateCallbackResult> {
    // Old and new bundles must be different
    if rotation.old_bundle_hash == rotation.new_bundle_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Old and new bundle hashes must be different".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::PreKeyBundle(bundle) => {
                    validate_create_pre_key_bundle(action, bundle)
                }
                EntryTypes::UsedPreKey(used) => validate_create_used_pre_key(action, used),
                EntryTypes::KeyRotation(rotation) => {
                    validate_create_key_rotation(action, rotation)
                }
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::PreKeyBundle(bundle) => {
                    // Updates are allowed (for marking keys as used)
                    if bundle.identity_key.len() != 32 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Identity key must be 32 bytes".to_string(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
