// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Profiles Coordinator Zome for Mycelix Mail
use hdk::prelude::*;
use mail_profiles_integrity::*;

#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<Profile>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        LinkQuery::try_new(my_agent, LinkTypes::AgentToProfile)?,
        GetStrategy::default(),
    )?;

    // Get the most recent profile link
    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                return record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
            }
        }
    }

    Ok(None)
}

#[hdk_extern]
pub fn set_profile(profile: Profile) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Delete old profile links
    let old_links = get_links(
        LinkQuery::try_new(my_agent.clone(), LinkTypes::AgentToProfile)?,
        GetStrategy::default(),
    )?;
    for link in old_links {
        delete_link(link.create_link_hash, GetOptions::default())?;
    }

    // Create new profile entry
    let hash = create_entry(EntryTypes::Profile(profile))?;

    // Link agent to profile
    create_link(my_agent, hash.clone(), LinkTypes::AgentToProfile, ())?;

    Ok(hash)
}
