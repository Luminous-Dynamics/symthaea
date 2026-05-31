// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Circles Coordinator Zome
//! Business logic for care circle creation, membership, and discovery.

use circles_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Create a new care circle. The creator automatically becomes an Organizer member.
#[hdk_extern]
pub fn create_circle(circle: CareCircle) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CareCircle(circle.clone()))?;

    // Link to all circles
    let all_anchor = ensure_anchor("all_circles")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllCircles, ())?;

    // Link by type
    let type_anchor = ensure_anchor(&format!("circle_type:{}", circle.circle_type.anchor_key()))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::TypeToCircle,
        (),
    )?;

    // Link creator to circle
    let creator_anchor = ensure_anchor(&format!("agent_created_circles:{}", circle.created_by))?;
    create_link(
        creator_anchor,
        action_hash.clone(),
        LinkTypes::AgentToCreatedCircle,
        (),
    )?;

    // Auto-join creator as Organizer
    let now = sys_time()?;
    let membership = CircleMembership {
        circle_hash: action_hash.clone(),
        member: circle.created_by.clone(),
        role: MemberRole::Organizer,
        joined_at: now,
        active: true,
    };
    let membership_hash = create_entry(&EntryTypes::CircleMembership(membership))?;

    // Link circle to membership
    let circle_members_anchor = ensure_anchor(&format!("circle_members:{}", action_hash))?;
    create_link(
        circle_members_anchor,
        membership_hash.clone(),
        LinkTypes::CircleToMembership,
        (),
    )?;

    // Link agent to membership
    let agent_membership_anchor =
        ensure_anchor(&format!("agent_memberships:{}", circle.created_by))?;
    create_link(
        agent_membership_anchor,
        membership_hash,
        LinkTypes::AgentToMembership,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created circle".into()
    )))
}

/// Input for joining a circle
#[derive(Serialize, Deserialize, Debug)]
pub struct JoinCircleInput {
    pub circle_hash: ActionHash,
    pub role: MemberRole,
}

/// Join an existing care circle
#[hdk_extern]
pub fn join_circle(input: JoinCircleInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify circle exists
    let circle_record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Circle not found".into())),
    )?;

    let circle: CareCircle = circle_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    if !circle.active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot join an inactive circle".into()
        )));
    }

    // Check member count
    let circle_members_anchor = anchor_hash(&format!("circle_members:{}", input.circle_hash))?;
    let existing_links = get_links(
        LinkQuery::try_new(circle_members_anchor.clone(), LinkTypes::CircleToMembership)?,
        GetStrategy::default(),
    )?;

    // Count active members
    let mut active_count = 0u32;
    for link in &existing_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah, GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.active {
                    if m.member == caller {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Already a member of this circle".into()
                        )));
                    }
                    active_count += 1;
                }
            }
        }
    }

    if active_count >= circle.max_members {
        return Err(wasm_error!(WasmErrorInner::Guest("Circle is full".into())));
    }

    // Cannot self-assign Organizer role
    let role = if input.role == MemberRole::Organizer {
        MemberRole::Member
    } else {
        input.role
    };

    let now = sys_time()?;
    let membership = CircleMembership {
        circle_hash: input.circle_hash.clone(),
        member: caller.clone(),
        role,
        joined_at: now,
        active: true,
    };

    let membership_hash = create_entry(&EntryTypes::CircleMembership(membership))?;

    let cm_anchor = ensure_anchor(&format!("circle_members:{}", input.circle_hash))?;
    create_link(
        cm_anchor,
        membership_hash.clone(),
        LinkTypes::CircleToMembership,
        (),
    )?;

    let am_anchor = ensure_anchor(&format!("agent_memberships:{}", caller))?;
    create_link(
        am_anchor,
        membership_hash.clone(),
        LinkTypes::AgentToMembership,
        (),
    )?;

    get(membership_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created membership".into()
    )))
}

/// Leave a care circle by deactivating membership
#[hdk_extern]
pub fn leave_circle(circle_hash: ActionHash) -> ExternResult<bool> {
    let caller = agent_info()?.agent_initial_pubkey;

    let cm_anchor = anchor_hash(&format!("circle_members:{}", circle_hash))?;
    let links = get_links(
        LinkQuery::try_new(cm_anchor, LinkTypes::CircleToMembership)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah.clone(), GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.member == caller && m.active {
                    let updated = CircleMembership {
                        circle_hash: m.circle_hash,
                        member: m.member,
                        role: m.role,
                        joined_at: m.joined_at,
                        active: false,
                    };
                    update_entry(ah, &EntryTypes::CircleMembership(updated))?;
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Get circles the calling agent is a member of
#[hdk_extern]
pub fn get_my_circles(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let am_anchor = anchor_hash(&format!("agent_memberships:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(am_anchor, LinkTypes::AgentToMembership)?,
        GetStrategy::default(),
    )?;

    let mut circles = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah, GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.active {
                    if let Some(circle_rec) = get(m.circle_hash, GetOptions::default())? {
                        circles.push(circle_rec);
                    }
                }
            }
        }
    }

    Ok(circles)
}

/// Get all members of a circle
#[hdk_extern]
pub fn get_circle_members(circle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let cm_anchor = anchor_hash(&format!("circle_members:{}", circle_hash))?;
    let links = get_links(
        LinkQuery::try_new(cm_anchor, LinkTypes::CircleToMembership)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah, GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.active {
                    members.push(rec);
                }
            }
        }
    }

    Ok(members)
}

/// Get all circles
#[hdk_extern]
pub fn get_all_circles(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_circles")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllCircles)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get circles by type
#[hdk_extern]
pub fn get_circles_by_type(circle_type: CircleType) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("circle_type:{}", circle_type.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::TypeToCircle)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
