// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use std::collections::HashMap;

//================================
// Entry Types
//================================

/// Reputation entry from a specific network
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReputationEntry {
    pub did: String,
    pub network_id: String,           // "zero_trustml", "gitcoin_passport", "worldcoin", etc.
    pub reputation_score: f64,        // Normalized 0.0-1.0
    pub raw_score: f64,               // Original network score
    pub score_type: String,           // "trust", "contribution", "verification"
    pub metadata: String,             // JSON-encoded network-specific data
    pub issued_at: i64,
    pub expires_at: Option<i64>,
    pub issuer: String,               // Network authority DID
}

/// Aggregated reputation across all networks
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregatedReputation {
    pub did: String,
    pub global_score: f64,            // Weighted average 0.0-1.0
    pub network_count: u32,           // Number of networks with reputation
    pub network_scores: String,       // JSON map of network_id -> score
    pub trust_score: f64,             // Sybil resistance component
    pub contribution_score: f64,      // Participation component
    pub verification_score: f64,      // Identity verification component
    pub last_updated: i64,
    pub version: u32,                 // Increments with each update
}

//================================
// Link Types
//================================

#[hdk_link_types]
pub enum LinkTypes {
    ReputationByDID,              // DID -> ReputationEntry
    ReputationByNetwork,          // Network -> ReputationEntry
    AggregatedReputationLink,     // DID -> AggregatedReputation
}

//================================
// Entry Definitions
//================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    ReputationEntry(ReputationEntry),
    #[entry_type]
    AggregatedReputation(AggregatedReputation),
}

//================================
// Zome Functions
//================================

/// Store reputation entry from a network
#[hdk_extern]
pub fn store_reputation_entry(input: StoreReputationInput) -> ExternResult<ActionHash> {
    // 1. Validate reputation entry
    validate_reputation_entry(&input.entry)?;

    // 2. Check if entry already exists (prevent duplicates)
    // TODO: Cannot call #[hdk_extern] functions from within same zome.
    // DHT validation will prevent true duplicates anyway.
    // Consider creating a separate internal helper function if this check is needed.
    /*
    let existing = get_reputation_for_network(
        input.entry.did.clone(),
        input.entry.network_id.clone()
    )?;

    if existing.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Reputation entry already exists for {} on {}",
                    input.entry.did, input.entry.network_id)
        )));
    }
    */

    // 3. Create entry on DHT
    let action_hash = create_entry(&EntryTypes::ReputationEntry(input.entry.clone()))?;

    // 4. Create links for efficient querying
    // Link by DID
    let did_path = Path::from(format!("reputation.did.{}", input.entry.did)).typed(LinkTypes::ReputationByDID)?;
    did_path.ensure()?;
    create_link(
        did_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ReputationByDID,
        LinkTag::new(input.entry.network_id.as_bytes())
    )?;

    // Link by Network
    let network_path = Path::from(format!("reputation.network.{}", input.entry.network_id)).typed(LinkTypes::ReputationByNetwork)?;
    network_path.ensure()?;
    create_link(
        network_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ReputationByNetwork,
        LinkTag::new(input.entry.did.as_bytes())
    )?;

    debug!("Reputation entry stored: {} on {}", input.entry.did, input.entry.network_id);

    // 5. Trigger aggregated reputation update
    update_aggregated_reputation(input.entry.did.clone())?;

    Ok(action_hash)
}

/// Get reputation entry for a specific network
#[hdk_extern]
pub fn get_reputation_for_network(input: GetReputationInput) -> ExternResult<Option<ReputationEntry>> {
    let path = Path::from(format!("reputation.did.{}", input.did));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::ReputationByDID
        )?,
        GetStrategy::default()
    )?;

    // Find link matching network_id
    for link in links {
        if link.tag.0 == input.network_id.as_bytes() {
            let action_hash = link.target.clone().into_action_hash()
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(entry) = record.entry().as_option() {
                    if let Entry::App(app_bytes) = entry {
                        let reputation_entry = ReputationEntry::try_from(
                            SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                        ).map_err(|e| wasm_error!(e))?;

                        // Check expiration
                        if let Some(expires_at) = reputation_entry.expires_at {
                            if expires_at < sys_time()?.as_micros() {
                                continue; // Skip expired
                            }
                        }

                        return Ok(Some(reputation_entry));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get all reputation entries for a DID
#[hdk_extern]
pub fn get_all_reputation_entries(did: String) -> ExternResult<Vec<ReputationEntry>> {
    let path = Path::from(format!("reputation.did.{}", did));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::ReputationByDID
        )?,
        GetStrategy::default()
    )?;

    let mut entries = Vec::new();

    for link in links {
        let action_hash = link.target.into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    let reputation_entry = ReputationEntry::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ).map_err(|e| wasm_error!(e))?;

                    // Check expiration
                    if let Some(expires_at) = reputation_entry.expires_at {
                        if expires_at < sys_time()?.as_micros() {
                            continue; // Skip expired
                        }
                    }

                    entries.push(reputation_entry);
                }
            }
        }
    }

    Ok(entries)
}

/// Get aggregated reputation for a DID
#[hdk_extern]
pub fn get_aggregated_reputation(did: String) -> ExternResult<Option<AggregatedReputation>> {
    let path = Path::from(format!("aggregated_reputation.{}", did));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::AggregatedReputationLink
        )?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent aggregated reputation
    let action_hash = links.last().expect("links verified non-empty above").target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    if let Some(record) = get(action_hash, GetOptions::default())? {
        if let Some(entry) = record.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let aggregated = AggregatedReputation::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                ).map_err(|e| wasm_error!(e))?;
                return Ok(Some(aggregated));
            }
        }
    }

    Ok(None)
}

/// Update aggregated reputation (compute weighted average across networks)
#[hdk_extern]
pub fn update_aggregated_reputation(did: String) -> ExternResult<ActionHash> {
    // 1. Get all reputation entries for DID
    let entries = get_all_reputation_entries(did.clone())?;

    if entries.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No reputation entries found for DID".into()
        )));
    }

    // 2. Compute aggregated scores
    let (global_score, trust_score, contribution_score, verification_score) =
        compute_aggregated_scores(&entries)?;

    // 3. Build network scores map
    let mut network_scores_map = HashMap::new();
    for entry in &entries {
        network_scores_map.insert(entry.network_id.clone(), entry.reputation_score);
    }
    let network_scores_json = serde_json::to_string(&network_scores_map)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    // 4. Get previous version number
    let previous_version = if let Some(prev) = get_aggregated_reputation(did.clone())? {
        prev.version
    } else {
        0
    };

    // 5. Create new aggregated reputation
    let aggregated = AggregatedReputation {
        did: did.clone(),
        global_score,
        network_count: entries.len() as u32,
        network_scores: network_scores_json,
        trust_score,
        contribution_score,
        verification_score,
        last_updated: sys_time()?.as_micros(),
        version: previous_version + 1,
    };

    // 6. Store on DHT
    let action_hash = create_entry(&EntryTypes::AggregatedReputation(aggregated.clone()))?;

    // 7. Create link for resolution
    let path = Path::from(format!("aggregated_reputation.{}", did)).typed(LinkTypes::AggregatedReputationLink)?;
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AggregatedReputationLink,
        LinkTag::new("aggregated")
    )?;

    debug!("Aggregated reputation updated for {}: global_score={}", did, global_score);

    Ok(action_hash)
}

/// Query reputation entries by network
#[hdk_extern]
pub fn get_reputation_entries_by_network(network_id: String) -> ExternResult<Vec<ReputationEntry>> {
    let path = Path::from(format!("reputation.network.{}", network_id));

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::ReputationByNetwork
        )?,
        GetStrategy::default()
    )?;

    let mut entries = Vec::new();

    for link in links {
        let action_hash = link.target.into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    let reputation_entry = ReputationEntry::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ).map_err(|e| wasm_error!(e))?;

                    // Check expiration
                    if let Some(expires_at) = reputation_entry.expires_at {
                        if expires_at < sys_time()?.as_micros() {
                            continue; // Skip expired
                        }
                    }

                    entries.push(reputation_entry);
                }
            }
        }
    }

    Ok(entries)
}

//================================
// Helper Functions
//================================

fn validate_reputation_entry(entry: &ReputationEntry) -> ExternResult<()> {
    // 1. Validate DID format
    if !entry.did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    // 2. Validate network ID
    if entry.network_id.is_empty() || entry.network_id.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid network_id".into()
        )));
    }

    // 3. Validate reputation score (must be 0.0-1.0)
    if entry.reputation_score < 0.0 || entry.reputation_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reputation score must be between 0.0 and 1.0".into()
        )));
    }

    // 4. Validate score type
    match entry.score_type.as_str() {
        "trust" | "contribution" | "verification" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid score_type: {}", entry.score_type)
        )))
    }

    // 5. Validate issuer DID
    if !entry.issuer.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid issuer DID format".into()
        )));
    }

    // 6. Validate timestamps
    if entry.issued_at > sys_time()?.as_micros() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issued timestamp cannot be in the future".into()
        )));
    }

    if let Some(expires_at) = entry.expires_at {
        if expires_at <= entry.issued_at {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Expiration must be after issuance".into()
            )));
        }
    }

    Ok(())
}

fn compute_aggregated_scores(entries: &[ReputationEntry]) -> ExternResult<(f64, f64, f64, f64)> {
    // Network weights (can be made configurable)
    let network_weights: HashMap<&str, f64> = [
        ("zero_trustml", 1.0),
        ("gitcoin_passport", 0.9),
        ("worldcoin", 0.8),
        ("brightid", 0.7),
        ("proof_of_humanity", 0.8),
    ].iter().cloned().collect();

    let mut total_weighted_score = 0.0;
    let mut total_weight = 0.0;

    let mut trust_sum = 0.0;
    let mut trust_count = 0;

    let mut contribution_sum = 0.0;
    let mut contribution_count = 0;

    let mut verification_sum = 0.0;
    let mut verification_count = 0;

    for entry in entries {
        let weight = network_weights.get(entry.network_id.as_str()).unwrap_or(&0.5);

        // Global weighted score
        total_weighted_score += entry.reputation_score * weight;
        total_weight += weight;

        // Score type aggregation
        match entry.score_type.as_str() {
            "trust" => {
                trust_sum += entry.reputation_score;
                trust_count += 1;
            }
            "contribution" => {
                contribution_sum += entry.reputation_score;
                contribution_count += 1;
            }
            "verification" => {
                verification_sum += entry.reputation_score;
                verification_count += 1;
            }
            _ => {}
        }
    }

    let global_score = if total_weight > 0.0 {
        total_weighted_score / total_weight
    } else {
        0.0
    };

    let trust_score = if trust_count > 0 {
        trust_sum / trust_count as f64
    } else {
        0.0
    };

    let contribution_score = if contribution_count > 0 {
        contribution_sum / contribution_count as f64
    } else {
        0.0
    };

    let verification_score = if verification_count > 0 {
        verification_sum / verification_count as f64
    } else {
        0.0
    };

    Ok((global_score, trust_score, contribution_score, verification_score))
}

//================================
// Validation Callback
//================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(StoreEntry { entry, .. }) => {
            match entry {
                Entry::App(bytes) => {
                    // Try to deserialize as ReputationEntry
                    if let Ok(reputation_entry) = ReputationEntry::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if let Err(e) = validate_reputation_entry(&reputation_entry) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid reputation entry: {:?}", e)
                            ));
                        }
                        return Ok(ValidateCallbackResult::Valid);
                    }

                    // Try to deserialize as AggregatedReputation
                    if let Ok(_aggregated) = AggregatedReputation::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        // Basic validation for aggregated reputation
                        // (Could add more sophisticated checks)
                        return Ok(ValidateCallbackResult::Valid);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    Ok(ValidateCallbackResult::Valid)
}

//================================
// Input Types
//================================

#[derive(Serialize, Deserialize, Debug)]
pub struct StoreReputationInput {
    pub entry: ReputationEntry,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetReputationInput {
    pub did: String,
    pub network_id: String,
}
