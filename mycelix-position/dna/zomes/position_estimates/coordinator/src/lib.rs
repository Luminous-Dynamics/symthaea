// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Position estimates coordinator: compute and query fused positions.
//!
//! Uses the `positioning` library's trilateration to compute positions
//! from range measurements and anchor locations stored in sibling zomes.

use hdk::prelude::*;
use mycelix_position_shared::{PositionEstimateEntry, PositionQuality, PositionTimestamp};
use position_estimates_integrity::*;

fn anchor_for_node(node_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("estimates.node.{}", node_id));
    let typed = path.typed(LinkTypes::EstimatesByNode)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Store a computed position estimate. Requires Citizen tier.
#[hdk_extern]
pub fn store_position_estimate(input: PositionEstimateEntry) -> ExternResult<ActionHash> {
    let action_hash = create_entry(&EntryTypes::PositionEstimate(input.clone()))?;

    let node_anchor = anchor_for_node(&input.node_id)?;
    create_link(
        node_anchor,
        action_hash.clone(),
        LinkTypes::EstimatesByNode,
        LinkTag::new(format!("est:{}", input.node_id)),
    )?;

    Ok(action_hash)
}

/// Get the latest position estimate for a node.
#[hdk_extern]
pub fn get_latest_position(node_id: String) -> ExternResult<Option<PositionEstimateEntry>> {
    let node_anchor = anchor_for_node(&node_id)?;
    let links = get_links(
        LinkQuery::try_new(node_anchor, LinkTypes::EstimatesByNode)?,
        GetStrategy::Network,
    )?;

    let mut latest: Option<(PositionEstimateEntry, Timestamp)> = None;

    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        let timestamp = record.action().timestamp();
        if let Some(est) = record
            .entry()
            .to_app_option::<PositionEstimateEntry>()
            .ok()
            .flatten()
        {
            match &latest {
                Some((_, t)) if timestamp > *t => {
                    latest = Some((est, timestamp));
                }
                None => {
                    latest = Some((est, timestamp));
                }
                _ => {}
            }
        }
    }

    Ok(latest.map(|(est, _)| est))
}

/// Get position history for a node (all estimates, newest first).
#[hdk_extern]
pub fn get_position_history(node_id: String) -> ExternResult<Vec<PositionEstimateEntry>> {
    let node_anchor = anchor_for_node(&node_id)?;
    let links = get_links(
        LinkQuery::try_new(node_anchor, LinkTypes::EstimatesByNode)?,
        GetStrategy::Network,
    )?;

    let mut results = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(est) = record
            .entry()
            .to_app_option::<PositionEstimateEntry>()
            .ok()
            .flatten()
        {
            results.push(est);
        }
    }
    Ok(results)
}
