// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tool Library Coordinator Zome
//!
//! Community tool lending library — register tools, borrow, return,
//! and discover nearby tools via geohash indexing.

use hdk::prelude::*;
use tool_library_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal,
};

/// Helper to get an anchor entry hash.
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash.
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

/// Simple geohash encode (precision 6 ~ 1.2km cells).
fn geohash_encode(lat: f64, lon: f64, precision: u8) -> String {
    let precision = precision.max(1).min(12) as usize;
    let base32 = b"0123456789bcdefghjkmnpqrstuvwxyz";
    let mut lat_range = (-90.0_f64, 90.0_f64);
    let mut lon_range = (-180.0_f64, 180.0_f64);
    let mut hash = String::with_capacity(precision);
    let mut is_lon = true;
    let mut bit = 0u8;
    let mut ch = 0u8;

    while hash.len() < precision {
        let (range, val) = if is_lon {
            (&mut lon_range, lon)
        } else {
            (&mut lat_range, lat)
        };
        let mid = (range.0 + range.1) / 2.0;
        if val >= mid {
            ch |= 1 << (4 - bit);
            range.0 = mid;
        } else {
            range.1 = mid;
        }
        is_lon = !is_lon;
        bit += 1;
        if bit == 5 {
            hash.push(base32[ch as usize] as char);
            bit = 0;
            ch = 0;
        }
    }
    hash
}

// ============================================================================
// TOOL REGISTRATION & MANAGEMENT
// ============================================================================

/// Register a new tool in the community library.
#[hdk_extern]
pub fn register_tool(tool: Tool) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "register_tool")?;

    let action_hash = create_entry(&EntryTypes::Tool(tool.clone()))?;

    // Link from all tools anchor
    let all_anchor = ensure_anchor("all_tools")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllTools, ())?;

    // Link from category anchor
    let cat_str = format!("{:?}", tool.category);
    let cat_anchor = ensure_anchor(&format!("tools/category/{}", cat_str))?;
    create_link(cat_anchor, action_hash.clone(), LinkTypes::ToolsByCategory, ())?;

    // Link from agent (owner)
    let agent_anchor = ensure_anchor(&format!("agent/{}/tools", tool.owner_did))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToTool, ())?;

    // Geo-index if location provided
    if let Some(ref loc) = tool.location {
        let gh = geohash_encode(loc.latitude, loc.longitude, 6);
        let geo_anchor = ensure_anchor(&format!("geo/{}", gh))?;
        create_link(geo_anchor, action_hash.clone(), LinkTypes::GeoIndex, ())?;
    }

    Ok(action_hash)
}

/// Get a tool record by its action hash.
#[hdk_extern]
pub fn get_tool(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all registered tools.
#[hdk_extern]
pub fn get_all_tools(_: ()) -> ExternResult<Vec<Record>> {
    let base = anchor_hash("all_tools")?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::AllTools)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(200) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Get tools filtered by category.
#[hdk_extern]
pub fn get_tools_by_category(category: String) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_tools_by_category")?;

    let base = anchor_hash(&format!("tools/category/{}", category))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::ToolsByCategory)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(200) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Get all tools owned by the calling agent.
#[hdk_extern]
pub fn get_my_tools(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let agent_did = format!("{}", agent_info.agent_initial_pubkey);
    let base = anchor_hash(&format!("agent/{}/tools", agent_did))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::AgentToTool)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(200) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ============================================================================
// BORROWING & RETURNING
// ============================================================================

/// Input for borrowing a tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct BorrowInput {
    pub tool_hash: ActionHash,
    pub due_at: Timestamp,
}

/// Borrow a tool from the library.
#[hdk_extern]
pub fn borrow_tool(input: BorrowInput) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "borrow_tool")?;

    // Get the tool to verify it exists and is available
    let record = get(input.tool_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Tool not found".into())))?;
    let tool: Tool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid tool entry".into())))?;

    if !tool.available {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tool is not currently available".into()
        )));
    }

    let agent_info = agent_info()?;
    let borrower_did = format!("{}", agent_info.agent_initial_pubkey);

    let lending = Lending {
        tool_id: tool.tool_id.clone(),
        borrower_did: borrower_did.clone(),
        lender_did: tool.owner_did.clone(),
        borrowed_at: sys_time()?,
        due_at: input.due_at,
        returned_at: None,
        condition_on_return: None,
        status: LendingStatus::Active,
    };

    let lending_hash = create_entry(&EntryTypes::Lending(lending))?;

    // Link lending to tool
    create_link(
        input.tool_hash.clone(),
        lending_hash.clone(),
        LinkTypes::ToolToLending,
        (),
    )?;

    // Link lending to borrower
    let borrower_anchor = ensure_anchor(&format!("agent/{}/lendings", borrower_did))?;
    create_link(borrower_anchor, lending_hash.clone(), LinkTypes::AgentToLending, ())?;

    // Mark tool as unavailable
    let mut updated_tool = tool;
    updated_tool.available = false;
    update_entry(input.tool_hash, &updated_tool)?;

    Ok(lending_hash)
}

/// Input for returning a tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct ReturnInput {
    pub lending_hash: ActionHash,
    pub tool_hash: ActionHash,
    pub condition_on_return: Option<String>,
}

/// Return a borrowed tool.
#[hdk_extern]
pub fn return_tool(input: ReturnInput) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "return_tool")?;

    // Get the lending record
    let lending_record = get(input.lending_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Lending not found".into())))?;
    let mut lending: Lending = lending_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid lending entry".into())))?;

    if lending.status != LendingStatus::Active && lending.status != LendingStatus::Overdue {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Lending is not in a returnable state".into()
        )));
    }

    lending.returned_at = Some(sys_time()?);
    lending.condition_on_return = input.condition_on_return;
    lending.status = LendingStatus::Returned;

    let updated_hash = update_entry(input.lending_hash, &lending)?;

    // Mark tool as available again
    let tool_record = get(input.tool_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Tool not found".into())))?;
    let mut tool: Tool = tool_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid tool entry".into())))?;

    tool.available = true;
    update_entry(input.tool_hash, &tool)?;

    Ok(updated_hash)
}

// ============================================================================
// HISTORY & QUERIES
// ============================================================================

/// Get lending history for a specific tool.
#[hdk_extern]
pub fn get_tool_history(tool_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_tool_history")?;

    let links = get_links(
        LinkQuery::try_new(tool_hash, LinkTypes::ToolToLending)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(100) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Get all lendings for the calling agent (as borrower).
#[hdk_extern]
pub fn get_my_lendings(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let agent_did = format!("{}", agent_info.agent_initial_pubkey);
    let base = anchor_hash(&format!("agent/{}/lendings", agent_did))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::AgentToLending)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(100) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Input for nearby tools query.
#[derive(Serialize, Deserialize, Debug)]
pub struct NearbyQuery {
    pub latitude: f64,
    pub longitude: f64,
    /// Geohash precision (default 6 ~ 1.2km). Lower = wider area.
    pub precision: Option<u8>,
}

/// Get tools near a geographic location.
#[hdk_extern]
pub fn get_nearby_tools(input: NearbyQuery) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_nearby_tools")?;

    let precision = input.precision.unwrap_or(6);
    let gh = geohash_encode(input.latitude, input.longitude, precision);
    let base = anchor_hash(&format!("geo/{}", gh))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::GeoIndex)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(200) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ============================================================================
// AVAILABILITY MANAGEMENT
// ============================================================================

/// Mark a tool as unavailable (owner only).
#[hdk_extern]
pub fn mark_unavailable(tool_hash: ActionHash) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "mark_unavailable")?;

    let record = get(tool_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Tool not found".into())))?;
    let mut tool: Tool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid tool entry".into())))?;

    tool.available = false;
    let updated = update_entry(tool_hash, &tool)?;
    Ok(updated)
}

/// Mark a tool as available (owner only).
#[hdk_extern]
pub fn mark_available(tool_hash: ActionHash) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "mark_available")?;

    let record = get(tool_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Tool not found".into())))?;
    let mut tool: Tool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid tool entry".into())))?;

    tool.available = true;
    let updated = update_entry(tool_hash, &tool)?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tool_condition_variants() {
        use tool_library_integrity::ToolCondition;
        let variants = vec![
            ToolCondition::New,
            ToolCondition::Good,
            ToolCondition::Fair,
            ToolCondition::NeedsRepair,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ToolCondition = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn test_lending_status_flow() {
        use tool_library_integrity::LendingStatus;
        // Verify all variants serialize/deserialize and represent valid states
        let flow = vec![
            LendingStatus::Active,
            LendingStatus::Returned,
            LendingStatus::Overdue,
            LendingStatus::Disputed,
        ];
        for status in &flow {
            let json = serde_json::to_string(status).unwrap();
            let back: LendingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, back);
        }
        // Verify the typical happy-path transition is representable
        assert_ne!(LendingStatus::Active, LendingStatus::Returned);
        // Verify overdue is distinct from disputed
        assert_ne!(LendingStatus::Overdue, LendingStatus::Disputed);
    }

    #[test]
    fn test_tool_category_variants() {
        use tool_library_integrity::ToolCategory;
        let variants = vec![
            ToolCategory::HandTool,
            ToolCategory::PowerTool,
            ToolCategory::Garden,
            ToolCategory::Kitchen,
            ToolCategory::Workshop,
            ToolCategory::Electronics,
            ToolCategory::Other,
        ];
        assert_eq!(variants.len(), 7);
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ToolCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }
}
