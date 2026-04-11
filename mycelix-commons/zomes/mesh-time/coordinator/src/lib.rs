// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use mesh_time_integrity::*;
use mycelix_bridge_common::civic_requirement_basic;

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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_time_anchor")?;

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

/// Get per-agent network resilience score [0.0, 1.0].
///
/// Counts the agent's time anchors and weights by stratum quality.
/// Score = min(1.0, anchor_count / 720) × avg_stratum_quality
/// where stratum_quality = (16 - stratum) / 16.
///
/// Used by the 8D Sovereign Profile (D2: Network Resilience).
#[hdk_extern]
pub fn get_agent_resilience_score(_: ()) -> ExternResult<f64> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToAnchors)?,
        GetStrategy::Local,
    )?;

    if links.is_empty() {
        return Ok(0.0);
    }

    let count = links.len();
    let mut total_quality = 0.0_f64;
    let mut valid = 0u32;

    for link in links.into_iter().rev().take(100) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(anchor) = record.entry().to_app_option::<TimeAnchor>().ok().flatten() {
                    // Lower stratum = more reliable. Stratum 0 = reference clock.
                    let quality = (16.0 - anchor.stratum as f64) / 16.0;
                    total_quality += quality.clamp(0.0, 1.0);
                    valid += 1;
                }
            }
        }
    }

    let avg_quality = if valid > 0 { total_quality / valid as f64 } else { 0.5 };
    let saturation = (count as f64 / 720.0).min(1.0); // 720 anchors ≈ 30 days of hourly
    Ok((saturation * avg_quality).clamp(0.0, 1.0))
}

/// Dispute a skewed time anchor (Participant+).
#[hdk_extern]
pub fn dispute_time_anchor(dispute: TimeDispute) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "dispute_time_anchor")?;

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
