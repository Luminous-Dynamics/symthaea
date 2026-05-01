#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extension Registry Coordinator — publish, browse, verify community extensions.
//!
//! Consciousness gating:
//! - Browse: anyone (Observer+)
//! - Publish: Steward+ (prevents spam)
//! - Verify: Citizen+ (community review)

use hdk::prelude::*;
use extension_registry_integrity::*;

const ALL_EXTENSIONS_ANCHOR: &str = "all_extensions";

/// Publish a new extension manifest.
///
/// Requires Steward+ consciousness tier (enforced by caller/bridge gate).
/// Creates the entry and links it to the browsing anchor.
#[hdk_extern]
pub fn publish_extension(manifest: ExtensionManifest) -> ExternResult<ActionHash> {
    // Validate required fields
    if manifest.id.is_empty() || manifest.name.is_empty() || manifest.frontend_url.is_empty() {
        return Err(wasm_error!("Extension must have id, name, and frontend_url"));
    }

    let action_hash = create_entry(EntryTypes::ExtensionManifest(manifest.clone()))?;

    // Link to browsing anchor
    let anchor = anchor(ALL_EXTENSIONS_ANCHOR)?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::AllExtensions,
        (),
    )?;

    // Link to author
    let author = agent_info()?.agent_latest_pubkey;
    create_link(
        author,
        action_hash.clone(),
        LinkTypes::AuthorToExtensions,
        (),
    )?;

    Ok(action_hash)
}

/// Browse all published extensions.
#[hdk_extern]
pub fn list_extensions(_: ()) -> ExternResult<Vec<(ActionHash, ExtensionManifest)>> {
    let anchor = anchor(ALL_EXTENSIONS_ANCHOR)?;
    let links = get_links(
        GetLinksInputBuilder::try_new(anchor, LinkTypes::AllExtensions)?.build(),
    )?;

    let mut extensions = Vec::new();
    for link in links {
        let target: ActionHash = link.target.into_action_hash()
            .ok_or(wasm_error!("Invalid link target"))?;
        if let Some(record) = get(target.clone(), GetOptions::default())? {
            if let Some(manifest) = record.entry().to_app_option::<ExtensionManifest>()
                .map_err(|e| wasm_error!("Deserialize error: {}", e))? {
                extensions.push((target, manifest));
            }
        }
    }

    Ok(extensions)
}

/// Submit a verification vote on an extension.
///
/// Requires Citizen+ consciousness tier.
#[hdk_extern]
pub fn verify_extension(verification: ExtensionVerification) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::ExtensionVerification(verification.clone()))?;

    // Link from extension to this verification
    create_link(
        verification.extension_hash,
        action_hash.clone(),
        LinkTypes::ExtensionToVerifications,
        (),
    )?;

    Ok(action_hash)
}

/// Get all verifications for an extension.
#[hdk_extern]
pub fn get_verifications(extension_hash: ActionHash) -> ExternResult<Vec<ExtensionVerification>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(extension_hash, LinkTypes::ExtensionToVerifications)?.build(),
    )?;

    let mut verifications = Vec::new();
    for link in links {
        let target: ActionHash = link.target.into_action_hash()
            .ok_or(wasm_error!("Invalid link target"))?;
        if let Some(record) = get(target, GetOptions::default())? {
            if let Some(v) = record.entry().to_app_option::<ExtensionVerification>()
                .map_err(|e| wasm_error!("Deserialize error: {}", e))? {
                verifications.push(v);
            }
        }
    }

    Ok(verifications)
}

/// Get my published extensions.
#[hdk_extern]
pub fn my_extensions(_: ()) -> ExternResult<Vec<(ActionHash, ExtensionManifest)>> {
    let author = agent_info()?.agent_latest_pubkey;
    let links = get_links(
        GetLinksInputBuilder::try_new(author, LinkTypes::AuthorToExtensions)?.build(),
    )?;

    let mut extensions = Vec::new();
    for link in links {
        let target: ActionHash = link.target.into_action_hash()
            .ok_or(wasm_error!("Invalid link target"))?;
        if let Some(record) = get(target.clone(), GetOptions::default())? {
            if let Some(manifest) = record.entry().to_app_option::<ExtensionManifest>()
                .map_err(|e| wasm_error!("Deserialize error: {}", e))? {
                extensions.push((target, manifest));
            }
        }
    }

    Ok(extensions)
}

/// Create a deterministic anchor entry for linking.
fn anchor(text: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = text.as_bytes().to_vec();
    let hash = hash_entry(Entry::App(
        AppEntryBytes::try_from(SerializedBytes::from(UnsafeBytes::from(anchor_bytes)))
            .map_err(|e| wasm_error!("Anchor error: {}", e))?,
    ))?;
    Ok(hash)
}
