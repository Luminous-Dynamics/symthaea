// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Trust DNA
//!
//! Holochain DNA for decentralized trust attestations in Mycelix Mail.
//!
//! This DNA provides:
//! - Trust attestation creation and management
//! - Trust path discovery
//! - Aggregate trust score calculation
//! - Introduction requests
//! - Trust network queries

use hdk::prelude::*;

mod trust_attestation;
mod trust_path;
mod introduction;
mod profile;

pub use trust_attestation::*;
pub use trust_path::*;
pub use introduction::*;
pub use profile::*;

// ============================================================================
// Entry Types
// ============================================================================

#[hdk_entry_defs]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_def(name = "trust_attestation", visibility = "public")]
    TrustAttestation(TrustAttestation),

    #[entry_def(name = "profile", visibility = "public")]
    Profile(Profile),

    #[entry_def(name = "introduction_request", visibility = "private")]
    IntroductionRequest(IntroductionRequest),
}

#[hdk_link_types]
pub enum LinkTypes {
    // Agent -> TrustAttestation (attestations I've made)
    AgentToAttestation,
    // Agent -> TrustAttestation (attestations about me)
    AgentAttestations,
    // Agent -> Profile
    AgentToProfile,
    // TrustAttestation -> Agent (subject of attestation)
    AttestationToSubject,
    // Profile -> Profile (trust connections)
    TrustConnection,
    // Agent -> IntroductionRequest
    PendingIntroductions,
}

// ============================================================================
// Trust Attestation
// ============================================================================

/// A trust attestation from one agent to another
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustAttestation {
    /// The agent being attested to
    pub subject: AgentPubKey,
    /// Trust score (0-100)
    pub score: u8,
    /// Type of trust relationship
    pub trust_type: TrustType,
    /// Optional context/reason for the attestation
    pub context: Option<String>,
    /// Claims/evidence supporting the attestation
    pub claims: Vec<String>,
    /// When this attestation was created
    pub created_at: Timestamp,
    /// When this attestation expires (optional)
    pub expires_at: Option<Timestamp>,
    /// Whether this attestation is still active
    pub is_active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TrustType {
    Personal,
    Professional,
    Organizational,
    Transactional,
    Referral,
}

// ============================================================================
// Profile
// ============================================================================

/// User profile in the trust network
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Profile {
    /// Display name
    pub name: String,
    /// Email addresses associated with this profile
    pub emails: Vec<String>,
    /// Avatar URL or hash
    pub avatar: Option<String>,
    /// Bio/description
    pub bio: Option<String>,
    /// Public key for encryption (separate from Holochain agent key)
    pub encryption_key: Option<String>,
    /// When the profile was last updated
    pub updated_at: Timestamp,
}

// ============================================================================
// Introduction Request
// ============================================================================

/// Request for introduction between two agents
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IntroductionRequest {
    /// The agent requesting introduction
    pub requester: AgentPubKey,
    /// The target agent to be introduced to
    pub target: AgentPubKey,
    /// The intermediary (mutual connection)
    pub introducer: AgentPubKey,
    /// Message from requester
    pub message: Option<String>,
    /// Status of the request
    pub status: IntroductionStatus,
    /// When the request was created
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum IntroductionStatus {
    Pending,
    Accepted,
    Declined,
    Expired,
}

// ============================================================================
// Zome Functions - Trust Attestations
// ============================================================================

/// Create a new trust attestation
#[hdk_extern]
pub fn create_attestation(input: CreateAttestationInput) -> ExternResult<ActionHash> {
    let attestation = TrustAttestation {
        subject: input.subject.clone(),
        score: input.score.min(100), // Cap at 100
        trust_type: input.trust_type,
        context: input.context,
        claims: input.claims,
        created_at: sys_time()?,
        expires_at: input.expires_at,
        is_active: true,
    };

    let action_hash = create_entry(&EntryTypes::TrustAttestation(attestation.clone()))?;

    // Create links for querying
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Link from me to my attestation
    create_link(
        my_agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToAttestation,
        (),
    )?;

    // Link from subject to this attestation
    create_link(
        input.subject.clone(),
        action_hash.clone(),
        LinkTypes::AgentAttestations,
        (),
    )?;

    // Link attestation to subject for reverse lookup
    create_link(
        action_hash.clone(),
        input.subject,
        LinkTypes::AttestationToSubject,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateAttestationInput {
    pub subject: AgentPubKey,
    pub score: u8,
    pub trust_type: TrustType,
    pub context: Option<String>,
    pub claims: Vec<String>,
    pub expires_at: Option<Timestamp>,
}

/// Get attestations made by an agent
#[hdk_extern]
pub fn get_attestations_by(agent: AgentPubKey) -> ExternResult<Vec<TrustAttestationWithHash>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(agent, LinkTypes::AgentToAttestation)?.build(),
    )?;

    let mut attestations = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(attestation) = record.entry().to_app_option::<TrustAttestation>()? {
                    attestations.push(TrustAttestationWithHash {
                        hash,
                        attestation,
                    });
                }
            }
        }
    }

    Ok(attestations)
}

/// Get attestations about an agent
#[hdk_extern]
pub fn get_attestations_about(agent: AgentPubKey) -> ExternResult<Vec<TrustAttestationWithHash>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(agent, LinkTypes::AgentAttestations)?.build(),
    )?;

    let mut attestations = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(attestation) = record.entry().to_app_option::<TrustAttestation>()? {
                    // Only include active attestations
                    if attestation.is_active {
                        attestations.push(TrustAttestationWithHash {
                            hash,
                            attestation,
                        });
                    }
                }
            }
        }
    }

    Ok(attestations)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrustAttestationWithHash {
    pub hash: ActionHash,
    pub attestation: TrustAttestation,
}

/// Calculate aggregate trust score for an agent
#[hdk_extern]
pub fn calculate_trust_score(agent: AgentPubKey) -> ExternResult<TrustScore> {
    let attestations = get_attestations_about(agent.clone())?;

    if attestations.is_empty() {
        return Ok(TrustScore {
            agent,
            score: 0,
            attestation_count: 0,
            weighted_score: 0.0,
        });
    }

    // Simple weighted average based on attestor's own trust score
    // In production, would use more sophisticated algorithms
    let total_score: u32 = attestations
        .iter()
        .map(|a| a.attestation.score as u32)
        .sum();

    let avg_score = (total_score / attestations.len() as u32) as u8;
    let weighted = total_score as f64 / attestations.len() as f64;

    Ok(TrustScore {
        agent,
        score: avg_score,
        attestation_count: attestations.len(),
        weighted_score: weighted,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrustScore {
    pub agent: AgentPubKey,
    pub score: u8,
    pub attestation_count: usize,
    pub weighted_score: f64,
}

/// Revoke an attestation
#[hdk_extern]
pub fn revoke_attestation(hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Attestation not found".into())))?;

    let mut attestation: TrustAttestation = record
        .entry()
        .to_app_option()?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid attestation".into())))?;

    // Verify caller is the creator
    if record.action().author() != &agent_info()?.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the creator can revoke an attestation".into()
        )));
    }

    attestation.is_active = false;

    update_entry(hash, &attestation)
}

// ============================================================================
// Zome Functions - Trust Paths
// ============================================================================

/// Find trust paths between two agents
#[hdk_extern]
pub fn find_trust_paths(input: FindTrustPathsInput) -> ExternResult<Vec<TrustPath>> {
    let mut paths = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    // BFS to find paths
    queue.push_back(TrustPath {
        nodes: vec![input.from.clone()],
        total_score: 100,
        path_length: 0,
    });

    while let Some(current_path) = queue.pop_front() {
        let current_agent = current_path.nodes.last().unwrap();

        if current_agent == &input.to {
            paths.push(current_path);
            continue;
        }

        if current_path.path_length >= input.max_depth {
            continue;
        }

        if visited.contains(current_agent) {
            continue;
        }
        visited.insert(current_agent.clone());

        // Get attestations made by current agent
        let attestations = get_attestations_by(current_agent.clone())?;

        for attestation in attestations {
            let subject = attestation.attestation.subject;
            if !visited.contains(&subject) {
                let mut new_path = current_path.clone();
                new_path.nodes.push(subject);
                // Decay score along the path
                new_path.total_score = (new_path.total_score * attestation.attestation.score as u32) / 100;
                new_path.path_length += 1;
                queue.push_back(new_path);
            }
        }
    }

    // Sort by score (highest first)
    paths.sort_by(|a, b| b.total_score.cmp(&a.total_score));

    // Return top paths
    Ok(paths.into_iter().take(input.max_paths).collect())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FindTrustPathsInput {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub max_depth: u8,
    pub max_paths: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrustPath {
    pub nodes: Vec<AgentPubKey>,
    pub total_score: u32,
    pub path_length: u8,
}

// ============================================================================
// Zome Functions - Profiles
// ============================================================================

/// Create or update profile
#[hdk_extern]
pub fn set_profile(input: Profile) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Check for existing profile
    let links = get_links(
        GetLinksInputBuilder::try_new(my_agent.clone(), LinkTypes::AgentToProfile)?.build(),
    )?;

    let profile = Profile {
        name: input.name,
        emails: input.emails,
        avatar: input.avatar,
        bio: input.bio,
        encryption_key: input.encryption_key,
        updated_at: sys_time()?,
    };

    let action_hash = if let Some(link) = links.first() {
        // Update existing profile
        if let Some(hash) = link.target.clone().into_action_hash() {
            update_entry(hash, &profile)?
        } else {
            create_entry(&EntryTypes::Profile(profile))?
        }
    } else {
        // Create new profile
        let hash = create_entry(&EntryTypes::Profile(profile))?;
        create_link(my_agent, hash.clone(), LinkTypes::AgentToProfile, ())?;
        hash
    };

    Ok(action_hash)
}

/// Get profile for an agent
#[hdk_extern]
pub fn get_profile(agent: AgentPubKey) -> ExternResult<Option<Profile>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(agent, LinkTypes::AgentToProfile)?.build(),
    )?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                return Ok(record.entry().to_app_option()?);
            }
        }
    }

    Ok(None)
}

/// Get my profile
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<Profile>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    get_profile(my_agent)
}

// ============================================================================
// Zome Functions - Introduction Requests
// ============================================================================

/// Request an introduction
#[hdk_extern]
pub fn request_introduction(input: RequestIntroductionInput) -> ExternResult<ActionHash> {
    let requester = agent_info()?.agent_initial_pubkey;

    let request = IntroductionRequest {
        requester: requester.clone(),
        target: input.target,
        introducer: input.introducer.clone(),
        message: input.message,
        status: IntroductionStatus::Pending,
        created_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::IntroductionRequest(request))?;

    // Link to introducer for notification
    create_link(
        input.introducer,
        hash.clone(),
        LinkTypes::PendingIntroductions,
        (),
    )?;

    Ok(hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestIntroductionInput {
    pub target: AgentPubKey,
    pub introducer: AgentPubKey,
    pub message: Option<String>,
}

/// Get pending introduction requests for me to handle
#[hdk_extern]
pub fn get_pending_introductions(_: ()) -> ExternResult<Vec<IntroductionRequestWithHash>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        GetLinksInputBuilder::try_new(my_agent, LinkTypes::PendingIntroductions)?.build(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(request) = record.entry().to_app_option::<IntroductionRequest>()? {
                    if matches!(request.status, IntroductionStatus::Pending) {
                        requests.push(IntroductionRequestWithHash { hash, request });
                    }
                }
            }
        }
    }

    Ok(requests)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IntroductionRequestWithHash {
    pub hash: ActionHash,
    pub request: IntroductionRequest,
}

/// Respond to an introduction request
#[hdk_extern]
pub fn respond_to_introduction(input: RespondToIntroductionInput) -> ExternResult<ActionHash> {
    let record = get(input.request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;

    let mut request: IntroductionRequest = record
        .entry()
        .to_app_option()?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid request".into())))?;

    // Verify caller is the introducer
    if request.introducer != agent_info()?.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the introducer can respond".into()
        )));
    }

    request.status = if input.accept {
        IntroductionStatus::Accepted
    } else {
        IntroductionStatus::Declined
    };

    update_entry(input.request_hash, &request)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RespondToIntroductionInput {
    pub request_hash: ActionHash,
    pub accept: bool,
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { entry_type, entry, .. } => match entry_type {
                EntryTypes::TrustAttestation(attestation) => {
                    validate_trust_attestation(attestation)
                }
                EntryTypes::Profile(profile) => validate_profile(profile),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_trust_attestation(attestation: TrustAttestation) -> ExternResult<ValidateCallbackResult> {
    // Score must be 0-100
    if attestation.score > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score must be between 0 and 100".into(),
        ));
    }

    // Can't attest to yourself
    // Note: Would need access to the author here for full validation

    Ok(ValidateCallbackResult::Valid)
}

fn validate_profile(profile: Profile) -> ExternResult<ValidateCallbackResult> {
    // Name must not be empty
    if profile.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Profile name cannot be empty".into(),
        ));
    }

    // Name length check
    if profile.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Profile name too long".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
