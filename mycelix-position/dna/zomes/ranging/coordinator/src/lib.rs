// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ranging coordinator: ingest and query range measurements.

use hdk::prelude::*;
use mycelix_position_shared::{
    PositionQuality, PositionTimestamp, RangeMeasurement, RangingMethod,
};
use ranging_integrity::*;

fn anchor_for_node(node_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("ranges.node.{}", node_id));
    let typed = path.typed(LinkTypes::RangesByNode)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitRangeInput {
    pub from_node: String,
    pub to_node: String,
    pub range_m: f64,
    pub sigma_m: f64,
    pub method: RangingMethod,
    pub quality: u8,
}

/// Submit a range measurement. Requires Participant tier.
#[hdk_extern]
pub fn submit_range(input: SubmitRangeInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let measurement = RangeMeasurement {
        from_node: input.from_node.clone(),
        to_node: input.to_node.clone(),
        range_m: input.range_m,
        sigma_m: input.sigma_m,
        method: input.method,
        quality: PositionQuality::new(input.quality),
        measured_by: agent,
        measured_at: PositionTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::RangeMeasurement(measurement))?;

    // Index by from_node
    let from_anchor = anchor_for_node(&input.from_node)?;
    create_link(
        from_anchor,
        action_hash.clone(),
        LinkTypes::RangesByNode,
        LinkTag::new(format!("range:{}→{}", input.from_node, input.to_node)),
    )?;

    // Index by to_node
    let to_anchor = anchor_for_node(&input.to_node)?;
    create_link(
        to_anchor,
        action_hash.clone(),
        LinkTypes::RangesByNode,
        LinkTag::new(format!("range:{}→{}", input.from_node, input.to_node)),
    )?;

    Ok(action_hash)
}

/// Get all range measurements involving a node.
#[hdk_extern]
pub fn get_ranges_for_node(node_id: String) -> ExternResult<Vec<RangeMeasurement>> {
    let node_anchor = anchor_for_node(&node_id)?;
    let links = get_links(
        LinkQuery::try_new(node_anchor, LinkTypes::RangesByNode)?,
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
        if let Some(m) = record
            .entry()
            .to_app_option::<RangeMeasurement>()
            .ok()
            .flatten()
        {
            results.push(m);
        }
    }
    Ok(results)
}
