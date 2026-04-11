// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Attestation Zome for Mycelix Mail
//!
//! Implements decentralized trust attestations on Holochain DHT

use hdk::prelude::*;

/// Trust attestation entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustAttestation {
    /// The agent being attested
    pub subject: AgentPubKey,
    /// Trust level (0.0 to 1.0)
    pub trust_level: f64,
    /// Context for the trust (e.g., "professional", "personal", "verified")
    pub context: String,
    /// Optional expiration timestamp
    pub expires_at: Option<Timestamp>,
    /// Post-quantum signature (Dilithium)
    pub pq_signature: Vec<u8>,
    /// Additional metadata
    pub metadata: Option<String>,
}

/// Trust attestation with author info for queries
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrustAttestationWithAuthor {
    pub attestation: TrustAttestation,
    pub author: AgentPubKey,
    pub created_at: Timestamp,
    pub action_hash: ActionHash,
}

/// Input for creating a trust attestation
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateAttestationInput {
    pub subject: AgentPubKey,
    pub trust_level: f64,
    pub context: String,
    pub expires_at: Option<Timestamp>,
    pub pq_signature: Vec<u8>,
    pub metadata: Option<String>,
}

/// Input for querying attestations
#[derive(Serialize, Deserialize, Debug)]
pub struct GetAttestationsInput {
    pub subject: AgentPubKey,
    pub context: Option<String>,
    pub include_expired: bool,
}

/// Aggregated trust score
#[derive(Serialize, Deserialize, Debug)]
pub struct TrustScore {
    pub subject: AgentPubKey,
    pub score: f64,
    pub attestation_count: u32,
    pub contexts: Vec<String>,
    pub last_updated: Timestamp,
}

/// Trust path between agents
#[derive(Serialize, Deserialize, Debug)]
pub struct TrustPath {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub path: Vec<AgentPubKey>,
    pub path_trust: f64,
}

// Entry definitions
entry_defs![
    PathEntry::entry_def(),
    TrustAttestation::entry_def()
];

/// Link types for the trust zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from agent to their attestations (as subject)
    AgentToAttestations,
    /// Links from agent to attestations they've made (as author)
    AuthorToAttestations,
    /// Links for context-based indexing
    ContextToAttestations,
    /// Links for trust graph traversal
    TrustConnection,
}

// ============================================================================
// Zome Functions
// ============================================================================

/// Create a new trust attestation
#[hdk_extern]
pub fn create_attestation(input: CreateAttestationInput) -> ExternResult<ActionHash> {
    // Validate trust level
    if input.trust_level < 0.0 || input.trust_level > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust level must be between 0.0 and 1.0".to_string()
        )));
    }

    // Validate context
    if input.context.is_empty() || input.context.len() > 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Context must be 1-50 characters".to_string()
        )));
    }

    // Cannot attest to yourself
    let author = agent_info()?.agent_initial_pubkey;
    if author == input.subject {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot create attestation for yourself".to_string()
        )));
    }

    // Create the attestation
    let attestation = TrustAttestation {
        subject: input.subject.clone(),
        trust_level: input.trust_level,
        context: input.context.clone(),
        expires_at: input.expires_at,
        pq_signature: input.pq_signature,
        metadata: input.metadata,
    };

    let action_hash = create_entry(&attestation)?;

    // Create links for indexing
    // Link from subject agent to this attestation
    create_link(
        input.subject.clone(),
        action_hash.clone(),
        LinkTypes::AgentToAttestations,
        input.context.as_bytes().to_vec(),
    )?;

    // Link from author to this attestation
    create_link(
        author.clone(),
        action_hash.clone(),
        LinkTypes::AuthorToAttestations,
        (),
    )?;

    // Link for trust graph - author trusts subject
    create_link(
        author,
        input.subject,
        LinkTypes::TrustConnection,
        action_hash.clone().get_raw_39().to_vec(),
    )?;

    Ok(action_hash)
}

/// Get all attestations for an agent
#[hdk_extern]
pub fn get_attestations(input: GetAttestationsInput) -> ExternResult<Vec<TrustAttestationWithAuthor>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(input.subject.clone(), LinkTypes::AgentToAttestations)?
            .build(),
    )?;

    let now = sys_time()?;
    let mut attestations = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid action hash: {:?}", e)))
        })?;

        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(attestation) = record
                .entry()
                .to_app_option::<TrustAttestation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
            {
                // Filter by context if specified
                if let Some(ref ctx) = input.context {
                    if attestation.context != *ctx {
                        continue;
                    }
                }

                // Filter expired unless requested
                if !input.include_expired {
                    if let Some(expires) = attestation.expires_at {
                        if expires < now {
                            continue;
                        }
                    }
                }

                let author = record.action().author().clone();
                let created_at = record.action().timestamp();

                attestations.push(TrustAttestationWithAuthor {
                    attestation,
                    author,
                    created_at,
                    action_hash,
                });
            }
        }
    }

    Ok(attestations)
}

/// Calculate aggregated trust score for an agent
#[hdk_extern]
pub fn get_trust_score(subject: AgentPubKey) -> ExternResult<TrustScore> {
    let attestations = get_attestations(GetAttestationsInput {
        subject: subject.clone(),
        context: None,
        include_expired: false,
    })?;

    if attestations.is_empty() {
        return Ok(TrustScore {
            subject,
            score: 0.5, // Default neutral score
            attestation_count: 0,
            contexts: vec![],
            last_updated: sys_time()?,
        });
    }

    // Calculate weighted average
    let total_trust: f64 = attestations.iter().map(|a| a.attestation.trust_level).sum();
    let score = total_trust / attestations.len() as f64;

    // Collect unique contexts
    let mut contexts: Vec<String> = attestations
        .iter()
        .map(|a| a.attestation.context.clone())
        .collect();
    contexts.sort();
    contexts.dedup();

    Ok(TrustScore {
        subject,
        score,
        attestation_count: attestations.len() as u32,
        contexts,
        last_updated: sys_time()?,
    })
}

/// Find trust path between two agents
#[hdk_extern]
pub fn find_trust_path(input: (AgentPubKey, AgentPubKey)) -> ExternResult<Option<TrustPath>> {
    let (from, to) = input;

    // BFS to find shortest trust path
    let mut visited: Vec<AgentPubKey> = vec![from.clone()];
    let mut queue: Vec<(AgentPubKey, Vec<AgentPubKey>)> = vec![(from.clone(), vec![from.clone()])];
    let max_depth = 6; // Limit search depth

    while let Some((current, path)) = queue.pop() {
        if path.len() > max_depth {
            continue;
        }

        // Get trust connections from current agent
        let links = get_links(
            GetLinksInputBuilder::try_new(current.clone(), LinkTypes::TrustConnection)?
                .build(),
        )?;

        for link in links {
            let next = AgentPubKey::try_from(link.target).map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Invalid agent key: {:?}", e)))
            })?;

            if next == to {
                // Found path!
                let mut final_path = path.clone();
                final_path.push(to.clone());

                // Calculate path trust (product of trust levels)
                let path_trust = calculate_path_trust(&final_path)?;

                return Ok(Some(TrustPath {
                    from,
                    to,
                    path: final_path,
                    path_trust,
                }));
            }

            if !visited.contains(&next) {
                visited.push(next.clone());
                let mut new_path = path.clone();
                new_path.push(next.clone());
                queue.insert(0, (next, new_path));
            }
        }
    }

    Ok(None)
}

/// Revoke an attestation
#[hdk_extern]
pub fn revoke_attestation(action_hash: ActionHash) -> ExternResult<ActionHash> {
    // Verify caller is the author
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Attestation not found".to_string())))?;

    let author = record.action().author();
    let caller = agent_info()?.agent_initial_pubkey;

    if *author != caller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the author can revoke an attestation".to_string()
        )));
    }

    // Delete the entry
    delete_entry(action_hash)
}

/// Get attestations made by an agent
#[hdk_extern]
pub fn get_my_attestations(_: ()) -> ExternResult<Vec<TrustAttestationWithAuthor>> {
    let author = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        GetLinksInputBuilder::try_new(author, LinkTypes::AuthorToAttestations)?
            .build(),
    )?;

    let mut attestations = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid action hash: {:?}", e)))
        })?;

        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(attestation) = record
                .entry()
                .to_app_option::<TrustAttestation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
            {
                attestations.push(TrustAttestationWithAuthor {
                    attestation,
                    author: record.action().author().clone(),
                    created_at: record.action().timestamp(),
                    action_hash,
                });
            }
        }
    }

    Ok(attestations)
}

// ============================================================================
// Helper Functions
// ============================================================================

fn calculate_path_trust(path: &[AgentPubKey]) -> ExternResult<f64> {
    if path.len() < 2 {
        return Ok(1.0);
    }

    let mut trust = 1.0;

    for i in 0..path.len() - 1 {
        let from = &path[i];
        let to = &path[i + 1];

        // Get attestation from 'from' to 'to'
        let links = get_links(
            GetLinksInputBuilder::try_new(from.clone(), LinkTypes::TrustConnection)?
                .build(),
        )?;

        let mut found_trust = 0.5; // Default if not found

        for link in links {
            let target = AgentPubKey::try_from(link.target.clone());
            if let Ok(t) = target {
                if t == *to {
                    // Get the attestation to find trust level
                    if let Ok(action_hash) = ActionHash::try_from(link.tag.into_inner()) {
                        if let Some(record) = get(action_hash, GetOptions::default())? {
                            if let Some(att) = record
                                .entry()
                                .to_app_option::<TrustAttestation>()
                                .ok()
                                .flatten()
                            {
                                found_trust = att.trust_level;
                                break;
                            }
                        }
                    }
                }
            }
        }

        trust *= found_trust;
    }

    Ok(trust)
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<TrustAttestation, ()>()? {
        FlatOp::StoreEntry(store_entry) => {
            match store_entry {
                OpEntry::CreateEntry { entry, .. } => {
                    if let Entry::App(app_entry) = entry {
                        let attestation: TrustAttestation = app_entry.into_sb().try_into()?;
                        validate_attestation(&attestation)
                    } else {
                        Ok(ValidateCallbackResult::Valid)
                    }
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_attestation(attestation: &TrustAttestation) -> ExternResult<ValidateCallbackResult> {
    // Trust level must be 0-1
    if attestation.trust_level < 0.0 || attestation.trust_level > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust level must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Context must be non-empty
    if attestation.context.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Context cannot be empty".to_string(),
        ));
    }

    // PQ signature must be present
    if attestation.pq_signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Post-quantum signature required".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
