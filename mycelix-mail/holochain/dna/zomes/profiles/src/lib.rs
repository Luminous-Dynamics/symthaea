// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Profiles Zome for Mycelix Mail
//!
//! Manages user profiles on the Holochain DHT

use hdk::prelude::*;

/// User profile entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Profile {
    /// Display name
    pub display_name: String,
    /// Email address (verified off-chain)
    pub email: String,
    /// Avatar URL or data URI
    pub avatar: Option<String>,
    /// Bio/description
    pub bio: Option<String>,
    /// Post-quantum public key (Dilithium)
    pub pq_public_key: Vec<u8>,
    /// Key exchange public key (Kyber)
    pub pq_kem_public_key: Vec<u8>,
    /// Additional profile fields
    pub fields: Option<String>, // JSON string of custom fields
}

/// Profile update input
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateProfileInput {
    pub display_name: Option<String>,
    pub avatar: Option<String>,
    pub bio: Option<String>,
    pub fields: Option<String>,
}

/// Profile with agent info
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProfileWithAgent {
    pub profile: Profile,
    pub agent: AgentPubKey,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

/// Search result
#[derive(Serialize, Deserialize, Debug)]
pub struct ProfileSearchResult {
    pub profiles: Vec<ProfileWithAgent>,
    pub total: u32,
}

entry_defs![
    PathEntry::entry_def(),
    Profile::entry_def()
];

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from agent to their profile
    AgentToProfile,
    /// Link from email to agent (for lookup)
    EmailToAgent,
    /// Link for name-based search
    NameIndex,
    /// All profiles anchor
    AllProfiles,
}

// ============================================================================
// Zome Functions
// ============================================================================

/// Create or update profile
#[hdk_extern]
pub fn create_profile(profile: Profile) -> ExternResult<ActionHash> {
    // Validate profile
    validate_profile(&profile)?;

    let agent = agent_info()?.agent_initial_pubkey;

    // Check if profile already exists
    let existing = get_my_profile(())?;

    let action_hash = if let Some(existing_profile) = existing {
        // Update existing profile
        let links = get_links(
            GetLinksInputBuilder::try_new(agent.clone(), LinkTypes::AgentToProfile)?
                .build(),
        )?;

        if let Some(link) = links.first() {
            let old_hash = ActionHash::try_from(link.target.clone()).map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
            })?;

            // Update entry
            update_entry(old_hash, &profile)?
        } else {
            create_entry(&profile)?
        }
    } else {
        // Create new profile
        create_entry(&profile)?
    };

    // Update links
    // Agent -> Profile
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToProfile,
        (),
    )?;

    // Email -> Agent (for lookup)
    let email_path = Path::from(format!("emails/{}", profile.email.to_lowercase()));
    email_path.ensure()?;
    create_link(
        email_path.path_entry_hash()?,
        agent.clone(),
        LinkTypes::EmailToAgent,
        (),
    )?;

    // Name index for search
    let name_prefix = profile.display_name.to_lowercase().chars().take(3).collect::<String>();
    let name_path = Path::from(format!("names/{}", name_prefix));
    name_path.ensure()?;
    create_link(
        name_path.path_entry_hash()?,
        agent,
        LinkTypes::NameIndex,
        profile.display_name.as_bytes().to_vec(),
    )?;

    Ok(action_hash)
}

/// Get current agent's profile
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<ProfileWithAgent>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_profile_for_agent(agent)
}

/// Get profile for a specific agent
#[hdk_extern]
pub fn get_profile_for_agent(agent: AgentPubKey) -> ExternResult<Option<ProfileWithAgent>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(agent.clone(), LinkTypes::AgentToProfile)?
            .build(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
        })?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(profile) = record
                .entry()
                .to_app_option::<Profile>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?
            {
                return Ok(Some(ProfileWithAgent {
                    profile,
                    agent,
                    created_at: record.action().timestamp(),
                    updated_at: record.action().timestamp(),
                }));
            }
        }
    }

    Ok(None)
}

/// Get agent by email
#[hdk_extern]
pub fn get_agent_by_email(email: String) -> ExternResult<Option<AgentPubKey>> {
    let email_path = Path::from(format!("emails/{}", email.to_lowercase()));

    let links = get_links(
        GetLinksInputBuilder::try_new(email_path.path_entry_hash()?, LinkTypes::EmailToAgent)?
            .build(),
    )?;

    if let Some(link) = links.first() {
        let agent = AgentPubKey::try_from(link.target.clone()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid agent: {:?}", e)))
        })?;
        return Ok(Some(agent));
    }

    Ok(None)
}

/// Search profiles by name
#[hdk_extern]
pub fn search_profiles(query: String) -> ExternResult<ProfileSearchResult> {
    let query_lower = query.to_lowercase();
    let prefix = query_lower.chars().take(3).collect::<String>();

    if prefix.len() < 2 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Search query must be at least 2 characters".to_string()
        )));
    }

    let name_path = Path::from(format!("names/{}", prefix));

    let links = get_links(
        GetLinksInputBuilder::try_new(name_path.path_entry_hash()?, LinkTypes::NameIndex)?
            .build(),
    )?;

    let mut profiles = Vec::new();

    for link in links {
        // Filter by full name match
        let name = String::from_utf8_lossy(&link.tag.into_inner()).to_lowercase();
        if !name.contains(&query_lower) {
            continue;
        }

        let agent = AgentPubKey::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid agent: {:?}", e)))
        })?;

        if let Some(profile) = get_profile_for_agent(agent)? {
            profiles.push(profile);
        }
    }

    let total = profiles.len() as u32;

    Ok(ProfileSearchResult { profiles, total })
}

/// Update profile fields
#[hdk_extern]
pub fn update_profile(input: UpdateProfileInput) -> ExternResult<ActionHash> {
    let current = get_my_profile(())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No profile exists".to_string())))?;

    let updated = Profile {
        display_name: input.display_name.unwrap_or(current.profile.display_name),
        email: current.profile.email, // Cannot change email
        avatar: input.avatar.or(current.profile.avatar),
        bio: input.bio.or(current.profile.bio),
        pq_public_key: current.profile.pq_public_key,
        pq_kem_public_key: current.profile.pq_kem_public_key,
        fields: input.fields.or(current.profile.fields),
    };

    create_profile(updated)
}

/// Get PQ public keys for an agent
#[hdk_extern]
pub fn get_public_keys(agent: AgentPubKey) -> ExternResult<Option<(Vec<u8>, Vec<u8>)>> {
    if let Some(profile) = get_profile_for_agent(agent)? {
        Ok(Some((
            profile.profile.pq_public_key,
            profile.profile.pq_kem_public_key,
        )))
    } else {
        Ok(None)
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn validate_profile(profile: &Profile) -> ExternResult<()> {
    if profile.display_name.is_empty() || profile.display_name.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Display name must be 1-100 characters".to_string()
        )));
    }

    if profile.email.is_empty() || !profile.email.contains('@') {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid email address".to_string()
        )));
    }

    if profile.pq_public_key.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "PQ public key required".to_string()
        )));
    }

    if profile.pq_kem_public_key.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "PQ KEM public key required".to_string()
        )));
    }

    Ok(())
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<Profile, ()>()? {
        FlatOp::StoreEntry(store_entry) => {
            match store_entry {
                OpEntry::CreateEntry { entry, .. } => {
                    if let Entry::App(app_entry) = entry {
                        let profile: Profile = app_entry.into_sb().try_into()?;

                        if profile.display_name.is_empty() {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Display name required".to_string(),
                            ));
                        }

                        if profile.pq_public_key.is_empty() {
                            return Ok(ValidateCallbackResult::Invalid(
                                "PQ public key required".to_string(),
                            ));
                        }
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
