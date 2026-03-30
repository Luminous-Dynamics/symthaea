// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use atlas_bridge_integrity::*;

// ─── Bridge to mycelix-energy ────────────────────────────────────

/// Emit a bridge event when new infrastructure is registered.
#[hdk_extern]
pub fn emit_bridge_event(input: EmitEventInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let event = BridgeEvent {
        event_type: input.event_type.clone(),
        source_dna: "mycelix_atlas".to_string(),
        payload: input.payload,
        created: now,
    };

    let hash = create_entry(&EntryTypes::BridgeEvent(event))?;

    // Index by type
    let anchor = Anchor(format!("event_type:{}", input.event_type));
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    let anchor_hash = hash_entry(&anchor)?;
    create_link(anchor_hash, hash.clone(), LinkTypes::EventTypeToEvents, ())?;

    // Index all events
    let all_anchor = Anchor("all_bridge_events".to_string());
    let _ = create_entry(&EntryTypes::Anchor(all_anchor.clone()));
    let all_hash = hash_entry(&all_anchor)?;
    create_link(all_hash, hash.clone(), LinkTypes::AllEvents, ())?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[hdk_extern]
pub fn list_bridge_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = Anchor("all_bridge_events".to_string());
    let _ = create_entry(&EntryTypes::Anchor(all_anchor.clone()));
    let all_hash = hash_entry(&all_anchor)?;

    let links = get_links(
        LinkQuery::try_new(all_hash, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ─── Input Types ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
pub struct EmitEventInput {
    pub event_type: String,
    pub payload: String,
}

// TODO: Add cross-hApp bridge calls to mycelix-energy when
// the unified hApp includes both DNAs:
//
// #[hdk_extern]
// pub fn sync_energy_projects(_: ()) -> ExternResult<Vec<Record>> {
//     call(
//         CallTargetCell::OtherRole("mycelix_energy".into()),
//         "projects",
//         "list_all_projects".into(),
//         None,
//         (),
//     )
// }
