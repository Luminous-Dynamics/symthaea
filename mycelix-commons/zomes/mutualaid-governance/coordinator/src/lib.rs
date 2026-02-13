//! Mutual Aid Governance Coordinator Zome
//! Democratic decision-making for mutual aid circles.

use hdk::prelude::*;
use mutualaid_governance_integrity::*;
use mutualaid_common::{Proposal, Vote};

/// Create a governance proposal
#[hdk_extern]
pub fn create_proposal(proposal: Proposal) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Proposal(proposal))?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created proposal".into())))
}

/// Get all proposals
#[hdk_extern]
pub fn get_all_proposals(_: ()) -> ExternResult<Vec<Record>> {
    Ok(vec![])
}

/// Cast a vote on a proposal
#[hdk_extern]
pub fn cast_vote(vote: Vote) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Vote(vote))?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created vote".into())))
}
