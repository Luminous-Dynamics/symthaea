use hdk::prelude::*;
use mesh_time_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

/// Record a time anchor on the DHT (Participant+).
#[hdk_extern]
pub fn record_time_anchor(anchor: TimeAnchor) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "record_time_anchor")?;

    let action_hash = create_entry(&EntryTypes::TimeAnchor(anchor.clone()))?;

    // Link from AllTimeAnchors anchor
    let all_anchor = ensure_anchor("all_time_anchors")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllTimeAnchors,
        (),
    )?;

    // Link from agent
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::AgentToAnchors,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

/// Get recent mesh time anchors for consensus.
#[hdk_extern]
pub fn get_mesh_time(_: ()) -> ExternResult<Vec<TimeAnchor>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_time_anchors")?, LinkTypes::AllTimeAnchors)?,
        GetStrategy::default(),
    )?;

    let mut anchors = Vec::new();
    for link in links.into_iter().rev().take(50) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(anchor) = record.entry().to_app_option::<TimeAnchor>().ok().flatten() {
                    anchors.push(anchor);
                }
            }
        }
    }
    Ok(anchors)
}

/// Dispute a skewed time anchor (Participant+).
#[hdk_extern]
pub fn dispute_time_anchor(dispute: TimeDispute) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "dispute_time_anchor")?;

    let action_hash = create_entry(&EntryTypes::TimeDispute(dispute.clone()))?;

    create_link(
        dispute.anchor_hash,
        action_hash.clone(),
        LinkTypes::AnchorToDisputes,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}
