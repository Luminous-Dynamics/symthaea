// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Identity Vault Integrity Zome
//!
//! Defines entry types and validation for the agent's private identity data.
//! All entries are stored on the source chain only (private by default).

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// Agent profile stored on the private source chain.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Profile {
    /// Display name (not necessarily unique).
    pub display_name: String,
    /// Optional avatar URL or base64-encoded image.
    pub avatar: Option<String>,
    /// Optional biographical text.
    pub bio: Option<String>,
    /// Key-value metadata (e.g., preferred language, timezone).
    pub metadata: std::collections::HashMap<String, String>,
    /// Timestamp of last profile update.
    pub updated_at: Timestamp,
}

/// Master key entry for key management.
///
/// Stores a reference to a cryptographic key used for signing,
/// encryption, or credential issuance. The actual key material
/// is stored externally (e.g., in lair-keystore); this entry
/// tracks the key's purpose and status.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MasterKey {
    /// Human-readable label for this key.
    pub label: String,
    /// Purpose: "signing", "encryption", "credential_issuance".
    pub purpose: String,
    /// Public key bytes (hex-encoded).
    pub public_key_hex: String,
    /// Whether this key is currently active.
    pub active: bool,
    /// Timestamp of key creation.
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Profile(Profile),
    MasterKey(MasterKey),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToProfile,
    AgentToKeys,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::Profile(profile) => validate_profile(&profile),
            EntryTypes::MasterKey(key) => validate_master_key(&key),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Profile(profile) => validate_profile(&profile),
            EntryTypes::MasterKey(key) => validate_master_key(&key),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_profile(profile: &Profile) -> ExternResult<ValidateCallbackResult> {
    if profile.display_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Profile display_name cannot be empty".into(),
        ));
    }
    if profile.display_name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Profile display_name must be <= 256 characters".into(),
        ));
    }
    if let Some(ref bio) = profile.bio {
        if bio.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Profile bio must be <= 4096 characters".into(),
            ));
        }
    }
    if profile.metadata.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Profile metadata must have <= 50 entries".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_master_key(key: &MasterKey) -> ExternResult<ValidateCallbackResult> {
    if key.label.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "MasterKey label cannot be empty".into(),
        ));
    }
    let valid_purposes = ["signing", "encryption", "credential_issuance"];
    if !valid_purposes.contains(&key.purpose.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "MasterKey purpose '{}' not in {:?}",
            key.purpose, valid_purposes
        )));
    }
    if key.public_key_hex.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "MasterKey public_key_hex cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(name: &str) -> Profile {
        Profile {
            display_name: name.into(),
            avatar: None,
            bio: None,
            metadata: std::collections::HashMap::new(),
            updated_at: Timestamp::from_micros(0),
        }
    }

    fn make_key(label: &str, purpose: &str) -> MasterKey {
        MasterKey {
            label: label.into(),
            purpose: purpose.into(),
            public_key_hex: "abcdef1234567890".into(),
            active: true,
            created_at: Timestamp::from_micros(0),
        }
    }

    #[test]
    fn valid_profile_passes() {
        let p = make_profile("Alice");
        assert!(matches!(
            validate_profile(&p).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn empty_display_name_rejected() {
        let p = make_profile("");
        match validate_profile(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn oversized_display_name_rejected() {
        let p = make_profile(&"x".repeat(257));
        match validate_profile(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn oversized_bio_rejected() {
        let mut p = make_profile("Alice");
        p.bio = Some("x".repeat(4097));
        match validate_profile(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn too_many_metadata_rejected() {
        let mut p = make_profile("Alice");
        for i in 0..51 {
            p.metadata.insert(format!("key_{}", i), "val".into());
        }
        match validate_profile(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("50")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn valid_signing_key_passes() {
        let k = make_key("primary", "signing");
        assert!(matches!(
            validate_master_key(&k).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn valid_encryption_key_passes() {
        let k = make_key("backup", "encryption");
        assert!(matches!(
            validate_master_key(&k).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn valid_credential_issuance_key_passes() {
        let k = make_key("issuer", "credential_issuance");
        assert!(matches!(
            validate_master_key(&k).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn invalid_purpose_rejected() {
        let k = make_key("bad", "decryption");
        match validate_master_key(&k).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("decryption")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn empty_label_rejected() {
        let k = make_key("", "signing");
        match validate_master_key(&k).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn empty_public_key_rejected() {
        let mut k = make_key("test", "signing");
        k.public_key_hex = String::new();
        match validate_master_key(&k).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn profile_serde_roundtrip() {
        let mut p = make_profile("Bob");
        p.bio = Some("Developer".into());
        p.avatar = Some("https://example.com/avatar.png".into());
        let json = serde_json::to_string(&p).unwrap();
        let back: Profile = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn master_key_serde_roundtrip() {
        let k = make_key("primary", "signing");
        let json = serde_json::to_string(&k).unwrap();
        let back: MasterKey = serde_json::from_str(&json).unwrap();
        assert_eq!(back, k);
    }
}
