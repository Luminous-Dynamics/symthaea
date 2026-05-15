// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use hdk::prelude::*;
use restorative_justice_integrity::*;

#[hdk_extern]
pub fn initiate_wound(input: WoundRecord) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::WoundRecord(input.clone()))?;
    create_link(
        input.agent,
        action_hash.clone(),
        LinkTypes::AgentToWound,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn resolve_wound(wound_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(wound_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Wound record not found".into())
    ))?;

    let mut wound: WoundRecord = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid entry type".into()
        )))?;

    // Must be in Remodeling to resolve
    if wound.phase != WoundPhase::Remodeling {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Wound must be in Remodeling phase to resolve".into()
        )));
    }

    wound.phase = WoundPhase::Maturation;
    let new_action_hash = update_entry(wound_hash, &EntryTypes::WoundRecord(wound.clone()))?;

    // --- RESTORE REPUTATION (Cross-hApp) ---
    // Successfully completing maturation restores resonance score to Identity.
    call(
        CallTargetCell::OtherRole("identity".into()),
        "reputation_aggregator".into(),
        "report_domain_score".into(),
        None,
        serde_json::json!({
            "agent_pubkey_b64": wound.agent.to_string(),
            "cluster": "symthaea",
            "score": wound.resolution_score
        }),
    )?;

    Ok(new_action_hash)
}
