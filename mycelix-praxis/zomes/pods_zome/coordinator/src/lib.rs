// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Pods Coordinator Zome
//!
//! Updated for Type 1 Civilization Substrate (Praxis-0.2)
//!
//! Manages "Pods" (autonomous learning cooperatives) and peer endorsements.

use hdk::prelude::*;
use mycelix_zome_helpers as _;
use pods_integrity::*;

#[hdk_extern]
pub fn create_pod(pod: LearningPod) -> ExternResult<ActionHash> {
    create_entry(EntryTypes::LearningPod(pod))
}

#[hdk_extern]
pub fn get_pod(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn create_endorsement(input: Endorsement) -> ExternResult<ActionHash> {
    create_entry(EntryTypes::Endorsement(input))
}

#[hdk_extern]
pub fn get_endorsements_for_agent(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::new(agent, LinkTypes::AgentToEndorsements.try_into_filter()?),
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

#[hdk_extern]
pub fn get_scholarship_pots(_: ()) -> ExternResult<Vec<Record>> {
    let path = Path::from("scholarship_pots");
    let links = get_links(
        LinkQuery::new(
            path.path_entry_hash()?,
            LinkTypes::AllPods.try_into_filter()?,
        ), // TODO: Add specific link type for pots
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}
