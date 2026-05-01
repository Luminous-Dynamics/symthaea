#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extension Registry Integrity — entry types and validation for community extensions.
//!
//! Community members publish extension manifests as DHT entries. Other users
//! browse, verify, and install. Consciousness gating: Steward+ to publish,
//! anyone to browse.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// A published community extension manifest.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ExtensionManifest {
    /// Unique extension identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Version string (semver).
    pub version: String,
    /// Author agent public key (publisher).
    pub author: AgentPubKey,
    /// Biological metaphor name.
    pub bio_name: String,
    /// Primary CSS color.
    pub color_primary: String,
    /// Glow/accent CSS color.
    pub color_glow: String,
    /// Minimum consciousness tier name ("Observer", "Participant", etc.).
    pub min_tier: String,
    /// URL where the frontend is hosted.
    pub frontend_url: String,
    /// Clusters this extension requires.
    pub required_clusters: Vec<String>,
    /// Clusters this extension can optionally use.
    pub optional_clusters: Vec<String>,
    /// Short description.
    pub description: String,
    /// Timestamp of publication.
    pub published_at: Timestamp,
}

/// A verification vote on an extension (governance-gated).
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ExtensionVerification {
    /// Hash of the ExtensionManifest being verified.
    pub extension_hash: ActionHash,
    /// Whether the verifier approves this extension.
    pub approved: bool,
    /// Optional review comment.
    pub comment: String,
    /// Verifier's consciousness tier at time of verification.
    pub verifier_tier: String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    ExtensionManifest(ExtensionManifest),
    #[entry_type]
    ExtensionVerification(ExtensionVerification),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All extensions → anchor for browsing
    AllExtensions,
    /// Extension → its verifications
    ExtensionToVerifications,
    /// Author → their published extensions
    AuthorToExtensions,
}

/// Validate extension manifest — basic checks.
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::ExtensionManifest(manifest) => validate_manifest(&manifest),
                EntryTypes::ExtensionVerification(verification) => {
                    validate_verification(&verification)
                }
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::ExtensionManifest(manifest) => validate_manifest(&manifest),
                EntryTypes::ExtensionVerification(verification) => {
                    validate_verification(&verification)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { tag, .. } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_manifest(manifest: &ExtensionManifest) -> ExternResult<ValidateCallbackResult> {
    if manifest.id.trim().is_empty() || manifest.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "id and name are required".into(),
        ));
    }
    if manifest.id.len() > 128 || manifest.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "id or name too long".into(),
        ));
    }
    if manifest.version.trim().is_empty() || manifest.version.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "version is required and must be <= 64 chars".into(),
        ));
    }
    if manifest.frontend_url.trim().is_empty() || manifest.frontend_url.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "frontend_url is required and must be <= 2048 chars".into(),
        ));
    }
    if manifest.description.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "description too long (max 2048 chars)".into(),
        ));
    }
    if manifest.required_clusters.len() > 64 || manifest.optional_clusters.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many clusters listed (max 64 each)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_verification(
    verification: &ExtensionVerification,
) -> ExternResult<ValidateCallbackResult> {
    if verification.comment.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "comment too long (max 1024 chars)".into(),
        ));
    }
    if verification.verifier_tier.trim().is_empty()
        || verification.verifier_tier.len() > 64
    {
        return Ok(ValidateCallbackResult::Invalid(
            "verifier_tier is required and must be <= 64 chars".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
