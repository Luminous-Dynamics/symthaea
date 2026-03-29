// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ── Entry Types ──────────────────────────────────────────────────────

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DependencyEcosystem {
    RustCrate,
    NpmPackage,
    PythonPackage,
    NixFlake,
    GoModule,
    RubyGem,
    MavenPackage,
    Other,
}

impl std::fmt::Display for DependencyEcosystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RustCrate => write!(f, "rust_crate"),
            Self::NpmPackage => write!(f, "npm_package"),
            Self::PythonPackage => write!(f, "python_package"),
            Self::NixFlake => write!(f, "nix_flake"),
            Self::GoModule => write!(f, "go_module"),
            Self::RubyGem => write!(f, "ruby_gem"),
            Self::MavenPackage => write!(f, "maven_package"),
            Self::Other => write!(f, "other"),
        }
    }
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DependencyIdentity {
    pub id: String,
    pub name: String,
    pub ecosystem: DependencyEcosystem,
    pub maintainer_did: String,
    pub repository_url: Option<String>,
    pub license: Option<String>,
    pub description: String,
    pub version: Option<String>,
    pub registered_at: Timestamp,
    pub verified: bool,
}

// ── Entry & Link Enums ───────────────────────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    DependencyIdentity(DependencyIdentity),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllDependencies,
    EcosystemToDependency,
    MaintainerToDependency,
    DependencyById,
}

// ── Pure Validation Functions ────────────────────────────────────────

pub fn validate_create_dependency(
    _action: Create,
    dep: DependencyIdentity,
) -> ExternResult<ValidateCallbackResult> {
    if dep.id.is_empty() || dep.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dependency id must be 1-256 characters".into(),
        ));
    }
    if dep.name.is_empty() || dep.name.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dependency name must be 1-128 characters".into(),
        ));
    }
    if !dep.maintainer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "maintainer_did must start with 'did:'".into(),
        ));
    }
    if let Some(ref url) = dep.repository_url {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ValidateCallbackResult::Invalid(
                "repository_url must be http(s)://".into(),
            ));
        }
    }
    if dep.description.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be at most 500 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_update_dependency(
    _action: Update,
    dep: DependencyIdentity,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Same field validation as create
    if dep.id.is_empty() || dep.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dependency id must be 1-256 characters".into(),
        ));
    }
    if dep.name.is_empty() || dep.name.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dependency name must be 1-128 characters".into(),
        ));
    }
    if !dep.maintainer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "maintainer_did must start with 'did:'".into(),
        ));
    }
    if let Some(ref url) = dep.repository_url {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ValidateCallbackResult::Invalid(
                "repository_url must be http(s)://".into(),
            ));
        }
    }
    if dep.description.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be at most 500 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ── HDI Validation Callback ──────────────────────────────────────────

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::DependencyIdentity(dep) => validate_create_dependency(action, dep),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::DependencyIdentity(dep) => {
                    validate_update_dependency(action, dep, original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllDependencies
            | LinkTypes::EcosystemToDependency
            | LinkTypes::MaintainerToDependency
            | LinkTypes::DependencyById => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllDependencies
            | LinkTypes::EcosystemToDependency
            | LinkTypes::MaintainerToDependency
            | LinkTypes::DependencyById => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

// ── Unit Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_action() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::now(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex::from(0),
                0.into(),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn valid_dep() -> DependencyIdentity {
        DependencyIdentity {
            id: "crate:serde:1.0".into(),
            name: "serde".into(),
            ecosystem: DependencyEcosystem::RustCrate,
            maintainer_did: "did:mycelix:abc123".into(),
            repository_url: Some("https://github.com/serde-rs/serde".into()),
            license: Some("MIT OR Apache-2.0".into()),
            description: "A serialization framework for Rust".into(),
            version: Some("1.0.219".into()),
            registered_at: Timestamp::now(),
            verified: false,
        }
    }

    #[test]
    fn test_valid_dependency() {
        let result = validate_create_dependency(test_action(), valid_dep()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_id_rejected() {
        let mut dep = valid_dep();
        dep.id = String::new();
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_id_too_long_rejected() {
        let mut dep = valid_dep();
        dep.id = "x".repeat(257);
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_name_rejected() {
        let mut dep = valid_dep();
        dep.name = String::new();
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_name_too_long_rejected() {
        let mut dep = valid_dep();
        dep.name = "x".repeat(129);
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_did_rejected() {
        let mut dep = valid_dep();
        dep.maintainer_did = "not-a-did".into();
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_repo_url_rejected() {
        let mut dep = valid_dep();
        dep.repository_url = Some("ftp://example.com".into());
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_none_repo_url_valid() {
        let mut dep = valid_dep();
        dep.repository_url = None;
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_description_too_long_rejected() {
        let mut dep = valid_dep();
        dep.description = "x".repeat(501);
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_all_ecosystems_serialize() {
        let ecosystems = vec![
            DependencyEcosystem::RustCrate,
            DependencyEcosystem::NpmPackage,
            DependencyEcosystem::PythonPackage,
            DependencyEcosystem::NixFlake,
            DependencyEcosystem::GoModule,
            DependencyEcosystem::RubyGem,
            DependencyEcosystem::MavenPackage,
            DependencyEcosystem::Other,
        ];
        for eco in ecosystems {
            let json = serde_json::to_string(&eco).unwrap();
            let back: DependencyEcosystem = serde_json::from_str(&json).unwrap();
            assert_eq!(eco, back);
        }
    }

    #[test]
    fn test_ecosystem_display() {
        assert_eq!(DependencyEcosystem::RustCrate.to_string(), "rust_crate");
        assert_eq!(DependencyEcosystem::NpmPackage.to_string(), "npm_package");
        assert_eq!(DependencyEcosystem::NixFlake.to_string(), "nix_flake");
        assert_eq!(DependencyEcosystem::GoModule.to_string(), "go_module");
        assert_eq!(DependencyEcosystem::RubyGem.to_string(), "ruby_gem");
        assert_eq!(
            DependencyEcosystem::MavenPackage.to_string(),
            "maven_package"
        );
    }

    #[test]
    fn test_boundary_id_length() {
        let mut dep = valid_dep();
        dep.id = "x".repeat(256);
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_boundary_name_length_128() {
        let mut dep = valid_dep();
        dep.name = "x".repeat(128);
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_boundary_description_500() {
        let mut dep = valid_dep();
        dep.description = "x".repeat(500);
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_http_repo_url_valid() {
        let mut dep = valid_dep();
        dep.repository_url = Some("http://example.com/repo".into());
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_ssh_repo_url_rejected() {
        let mut dep = valid_dep();
        dep.repository_url = Some("git@github.com:user/repo.git".into());
        let result = validate_create_dependency(test_action(), dep).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Update validation tests ─────────────────────────────────────

    fn test_update_action() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::now(),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0u8; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex::from(0),
                0.into(),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    #[test]
    fn test_valid_update_dependency() {
        let result = validate_update_dependency(
            test_update_action(),
            valid_dep(),
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_empty_id_rejected() {
        let mut dep = valid_dep();
        dep.id = String::new();
        let result = validate_update_dependency(
            test_update_action(),
            dep,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_invalid_did_rejected() {
        let mut dep = valid_dep();
        dep.maintainer_did = "not-a-did".into();
        let result = validate_update_dependency(
            test_update_action(),
            dep,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_invalid_repo_url_rejected() {
        let mut dep = valid_dep();
        dep.repository_url = Some("ftp://bad.url".into());
        let result = validate_update_dependency(
            test_update_action(),
            dep,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_description_too_long_rejected() {
        let mut dep = valid_dep();
        dep.description = "x".repeat(501);
        let result = validate_update_dependency(
            test_update_action(),
            dep,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── DependencyIdentity serde ────────────────────────────────────

    #[test]
    fn test_dependency_identity_serde_roundtrip() {
        let dep = valid_dep();
        let json = serde_json::to_string(&dep).unwrap();
        let back: DependencyIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(dep.id, back.id);
        assert_eq!(dep.name, back.name);
        assert_eq!(dep.ecosystem, back.ecosystem);
        assert_eq!(dep.maintainer_did, back.maintainer_did);
        assert_eq!(dep.verified, back.verified);
    }

    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = Anchor("test_anchor".into());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn test_ecosystem_display_python_and_other() {
        assert_eq!(
            DependencyEcosystem::PythonPackage.to_string(),
            "python_package"
        );
        assert_eq!(DependencyEcosystem::Other.to_string(), "other");
    }
}
