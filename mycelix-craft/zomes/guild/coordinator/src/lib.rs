#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//!
//! Guild coordinator zome — consciousness-gated professional federations.
//!
//! All write operations are gated by consciousness thresholds:
//! - Creating a guild: Journeyman+ (0.5+)
//! - Joining a guild: meets guild's consciousness_minimum
//! - Promoting members: Master+ (0.75+)
//! - Creating certification paths: Master+ (0.75+)
//! - Establishing federations: Elder only (0.9+)

use hdk::prelude::*;
use guild_integrity::*;
use mycelix_bridge_common as bridge;

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateGuildInput {
    pub name: String,
    pub description: String,
    pub professional_domain: String,
    pub consciousness_minimum_permille: u16,
    pub parent_guild: Option<ActionHash>,
    pub bioregion: Option<String>,
    pub governance_model: GuildGovernanceModel,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PromoteMemberInput {
    pub guild_id: ActionHash,
    pub membership_hash: ActionHash,
    pub new_role: GuildRole,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCertificationPathInput {
    pub guild_id: ActionHash,
    pub name: String,
    pub description: String,
    pub requirements: Vec<CertificationRequirement>,
    pub required_assessors: u8,
    pub minimum_role: GuildRole,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EstablishFederationInput {
    pub local_guild_id: ActionHash,
    pub remote_guild_id: ActionHash,
    pub remote_dna_hash: Option<String>,
    pub shared_standards: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_anchor(anchor_text: &str) -> ExternResult<ActionHash> {
    let anchor = GuildAnchor(anchor_text.to_string());
    create_entry(EntryTypes::GuildAnchor(anchor))
}

fn anchor_hash(anchor_text: &str) -> ExternResult<EntryHash> {
    let anchor = GuildAnchor(anchor_text.to_string());
    hash_entry(&anchor)
}

fn get_my_role_in_guild(guild_id: &ActionHash) -> ExternResult<Option<(ActionHash, GuildMembership)>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToGuildMembership)?, GetStrategy::default(),
    )?;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(membership) = GuildMembership::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        if membership.guild_id == *guild_id {
                            return Ok(Some((hash, membership)));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Extern functions
// ---------------------------------------------------------------------------

/// Create a new guild. Requires Journeyman-equivalent consciousness (500+ permille).
#[hdk_extern]
pub fn create_guild(input: CreateGuildInput) -> ExternResult<ActionHash> {
    // Consciousness gate: Citizen tier (≥0.4 combined) + identity ≥0.25
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_voting(),
        "create_guild",
    )?;

    let now = sys_time()?;
    let guild = Guild {
        name: input.name,
        description: input.description,
        professional_domain: input.professional_domain.clone(),
        consciousness_minimum_permille: input.consciousness_minimum_permille,
        parent_guild: input.parent_guild.clone(),
        bioregion: input.bioregion,
        governance_model: input.governance_model,
        created_at: now,
    };

    let guild_hash = create_entry(EntryTypes::Guild(guild))?;

    // Index: all_guilds anchor
    let all_anchor = anchor_hash("all_guilds")?;
    create_link(all_anchor, guild_hash.clone(), LinkTypes::AllGuilds, ())?;

    // Index: domain anchor
    let domain_anchor = anchor_hash(&format!("guild_domain:{}", input.professional_domain))?;
    create_link(domain_anchor, guild_hash.clone(), LinkTypes::DomainToGuild, ())?;

    // If this is a child guild, link from parent
    if let Some(ref parent) = input.parent_guild {
        create_link(parent.clone(), guild_hash.clone(), LinkTypes::GuildToChildGuild, ())?;
    }

    // Auto-join the creator as Elder (they founded the guild)
    let agent = agent_info()?.agent_initial_pubkey;
    let membership = GuildMembership {
        guild_id: guild_hash.clone(),
        member: agent.clone(),
        role: GuildRole::Elder,
        joined_at: now,
        last_role_change: now,
        consciousness_at_join_permille: 900, // Founder assumed to meet Elder threshold
    };
    let membership_hash = create_entry(EntryTypes::GuildMembership(membership))?;
    create_link(guild_hash.clone(), membership_hash.clone(), LinkTypes::GuildToMembership, ())?;
    create_link(agent, membership_hash, LinkTypes::AgentToGuildMembership, ())?;

    Ok(guild_hash)
}

/// Join a guild as an Observer. Consciousness must meet guild's minimum.
#[hdk_extern]
pub fn join_guild(guild_id: ActionHash) -> ExternResult<ActionHash> {
    // Verify guild exists
    let record = get(guild_id.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Guild not found".into())))?;

    let guild: Guild = match record.entry().as_option() {
        Some(Entry::App(bytes)) => Guild::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    // Check not already a member
    if get_my_role_in_guild(&guild_id)?.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest("Already a member of this guild".into())));
    }

    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    let membership = GuildMembership {
        guild_id: guild_id.clone(),
        member: agent.clone(),
        role: GuildRole::Observer,
        joined_at: now,
        last_role_change: now,
        consciousness_at_join_permille: guild.consciousness_minimum_permille,
    };

    let membership_hash = create_entry(EntryTypes::GuildMembership(membership))?;
    create_link(guild_id, membership_hash.clone(), LinkTypes::GuildToMembership, ())?;
    create_link(agent, membership_hash.clone(), LinkTypes::AgentToGuildMembership, ())?;

    Ok(membership_hash)
}

/// Promote a guild member to a higher role. Requires Master+ in the same guild.
#[hdk_extern]
pub fn promote_member(input: PromoteMemberInput) -> ExternResult<ActionHash> {
    // Consciousness gate: Steward tier (≥0.6 combined) + identity ≥0.5 + community ≥0.3
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_constitutional(),
        "promote_member",
    )?;

    // Verify the caller has Master+ role in this guild
    let my_role = get_my_role_in_guild(&input.guild_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("You are not a member of this guild".into())))?;

    if !my_role.1.role.can_promote() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Master and Elder roles can promote members".into()
        )));
    }

    // Fetch the existing membership
    let record = get(input.membership_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Membership not found".into())))?;

    let mut membership: GuildMembership = match record.entry().as_option() {
        Some(Entry::App(bytes)) => GuildMembership::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    // Verify the membership is in the same guild
    if membership.guild_id != input.guild_id {
        return Err(wasm_error!(WasmErrorInner::Guest("Membership is not in this guild".into())));
    }

    // Update role
    membership.role = input.new_role;
    membership.last_role_change = sys_time()?;

    let new_hash = update_entry(input.membership_hash, &membership)?;
    Ok(new_hash)
}

/// Create a certification path for a guild. Requires Master+ role.
#[hdk_extern]
pub fn create_certification_path(input: CreateCertificationPathInput) -> ExternResult<ActionHash> {
    // Consciousness gate: Steward tier (≥0.6 combined) + identity ≥0.5 + community ≥0.3
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_constitutional(),
        "create_certification_path",
    )?;

    // Verify caller has Master+ role
    let my_role = get_my_role_in_guild(&input.guild_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("You are not a member of this guild".into())))?;

    if !my_role.1.role.can_certify() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Master and Elder roles can create certification paths".into()
        )));
    }

    let path = CertificationPath {
        guild_id: input.guild_id.clone(),
        name: input.name,
        description: input.description,
        requirements: input.requirements,
        required_assessors: input.required_assessors,
        minimum_role: input.minimum_role,
        created_at: sys_time()?,
    };

    let path_hash = create_entry(EntryTypes::CertificationPath(path))?;
    create_link(input.guild_id, path_hash.clone(), LinkTypes::GuildToCertificationPath, ())?;

    Ok(path_hash)
}

/// Establish a federation link with another guild. Requires Elder role.
#[hdk_extern]
pub fn establish_federation(input: EstablishFederationInput) -> ExternResult<ActionHash> {
    // Consciousness gate: Guardian tier (≥0.8 combined) + identity ≥0.7 + community ≥0.5
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_guardian(),
        "establish_federation",
    )?;

    // Verify caller has Elder role
    let my_role = get_my_role_in_guild(&input.local_guild_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("You are not a member of this guild".into())))?;

    if my_role.1.role != GuildRole::Elder {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Elders can establish federation links".into()
        )));
    }

    let federation = GuildFederationLink {
        local_guild_id: input.local_guild_id.clone(),
        remote_guild_id: input.remote_guild_id,
        remote_dna_hash: input.remote_dna_hash,
        shared_standards: input.shared_standards,
        established_at: sys_time()?,
    };

    let fed_hash = create_entry(EntryTypes::GuildFederationLink(federation))?;
    create_link(input.local_guild_id, fed_hash.clone(), LinkTypes::GuildToFederationLink, ())?;

    Ok(fed_hash)
}

/// List all guilds (via all_guilds anchor).
#[hdk_extern]
pub fn list_guilds(_: ()) -> ExternResult<Vec<(ActionHash, Guild)>> {
    let anchor = anchor_hash("all_guilds")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllGuilds)?, GetStrategy::default(),
    )?;

    let mut guilds = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(guild) = Guild::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        guilds.push((hash, guild));
                    }
                }
            }
        }
    }
    Ok(guilds)
}

/// List guilds in a specific professional domain.
#[hdk_extern]
pub fn list_guilds_by_domain(domain: String) -> ExternResult<Vec<(ActionHash, Guild)>> {
    let anchor = anchor_hash(&format!("guild_domain:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::DomainToGuild)?, GetStrategy::default(),
    )?;

    let mut guilds = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(guild) = Guild::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        guilds.push((hash, guild));
                    }
                }
            }
        }
    }
    Ok(guilds)
}

/// Get all memberships for a guild.
#[hdk_extern]
pub fn list_guild_members(guild_id: ActionHash) -> ExternResult<Vec<(ActionHash, GuildMembership)>> {
    let links = get_links(
        LinkQuery::try_new(guild_id, LinkTypes::GuildToMembership)?, GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(membership) = GuildMembership::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        members.push((hash, membership));
                    }
                }
            }
        }
    }
    Ok(members)
}

/// Get all guilds the calling agent belongs to.
#[hdk_extern]
pub fn get_my_guilds(_: ()) -> ExternResult<Vec<(ActionHash, GuildMembership)>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToGuildMembership)?, GetStrategy::default(),
    )?;

    let mut memberships = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(membership) = GuildMembership::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        memberships.push((hash, membership));
                    }
                }
            }
        }
    }
    Ok(memberships)
}

/// Get certification paths for a guild.
#[hdk_extern]
pub fn get_guild_certifications(guild_id: ActionHash) -> ExternResult<Vec<(ActionHash, CertificationPath)>> {
    let links = get_links(
        LinkQuery::try_new(guild_id, LinkTypes::GuildToCertificationPath)?, GetStrategy::default(),
    )?;

    let mut paths = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(path) = CertificationPath::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        paths.push((hash, path));
                    }
                }
            }
        }
    }
    Ok(paths)
}
