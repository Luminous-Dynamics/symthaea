// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Music Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Music cluster.
//! Entry struct definitions come from `mycelix-bridge-entry-types` (shared
//! with Commons and Civic bridges). The `EntryTypes` enum and validation are local.

use hdi::prelude::*;
pub use mycelix_bridge_entry_types::CachedCredentialEntry;
use mycelix_bridge_entry_types::{
    check_author_match, check_link_author_match, validate_cached_credential,
    validate_event_fields, validate_query_fields, BridgeEventEntry, BridgeQueryEntry,
};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A consciousness-driven composition stored on the DHT.
///
/// Created when Symthaea's cognitive loop triggers music generation.
/// The harmony_activations, valence/arousal, and phi_score record the
/// exact cognitive state that produced this composition.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousCompositionEntry {
    /// Schema version for forward-compatibility
    pub schema_version: u8,
    /// The agent whose consciousness triggered this composition
    pub composer_agent: AgentPubKey,
    /// Eight Harmonies activation vector at composition time [0,1]^8
    pub harmony_activations: Vec<f32>,
    /// Valence at composition time [-1, 1]
    pub valence: f32,
    /// Arousal at composition time [0, 1]
    pub arousal: f32,
    /// Integrated Information (Phi) score at composition time [0, 1]
    pub phi_score: f32,
    /// Narrative tags from the active episode (e.g. ["discovery", "rising"])
    pub narrative_tags: Vec<String>,
    /// Derived tempo BPM (for playback reference)
    pub tempo_bpm: f32,
    /// Derived scale name (e.g. "minor", "maqam_rast")
    pub scale_name: String,
    /// Total duration in seconds
    pub duration_secs: f32,
    /// Number of notes generated
    pub note_count: u32,
    /// Composite aesthetic quality score [0, 1]
    pub quality_score: f32,
    /// Timestamp of composition
    pub composed_at: Timestamp,
}

/// Type alias for bridge query entries
pub type MusicQueryEntry = BridgeQueryEntry;

/// Type alias for bridge event entries
pub type MusicEventEntry = BridgeEventEntry;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Query(BridgeQueryEntry),
    Event(BridgeEventEntry),
    CachedCredential(CachedCredentialEntry),
    ConsciousComposition(ConsciousCompositionEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllQueries,
    AgentToQuery,
    DomainToQuery,
    AllEvents,
    EventTypeToEvent,
    AgentToEvent,
    DomainToEvent,
    /// Rate limit tracking: agent → anchor per dispatch call
    DispatchRateLimit,
    /// Agent → cached consciousness credential
    AgentToCredentialCache,
    /// All consciousness-driven compositions (global index)
    AllConsciousCompositions,
    /// Agent → their consciousness compositions
    AgentToConsciousComposition,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

const VALID_DOMAINS: &[&str] = &["catalog", "plays", "balances", "trust"];

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
            EntryTypes::ConsciousComposition(comp) => validate_conscious_composition(&comp),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
            EntryTypes::ConsciousComposition(comp) => validate_conscious_composition(&comp),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_query(query: &BridgeQueryEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_query_fields(query, VALID_DOMAINS) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

fn validate_event(event: &BridgeEventEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_event_fields(event, VALID_DOMAINS) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

fn validate_credential_cache(cred: &CachedCredentialEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_cached_credential(cred) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

fn validate_conscious_composition(
    comp: &ConsciousCompositionEntry,
) -> ExternResult<ValidateCallbackResult> {
    if comp.harmony_activations.len() != 8 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "harmony_activations must have 8 elements, got {}",
            comp.harmony_activations.len()
        )));
    }
    for (i, &v) in comp.harmony_activations.iter().enumerate() {
        if !(0.0..=1.0).contains(&v) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "harmony_activations[{}] = {} is out of [0, 1]",
                i, v
            )));
        }
    }
    if !(-1.0..=1.0).contains(&comp.valence) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "valence {} is out of [-1, 1]",
            comp.valence
        )));
    }
    if !(0.0..=1.0).contains(&comp.arousal) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "arousal {} is out of [0, 1]",
            comp.arousal
        )));
    }
    if !(0.0..=1.0).contains(&comp.phi_score) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "phi_score {} is out of [0, 1]",
            comp.phi_score
        )));
    }
    if comp.duration_secs <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "duration_secs must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
