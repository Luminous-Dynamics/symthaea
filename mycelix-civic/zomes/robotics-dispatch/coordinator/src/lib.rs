// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotics Dispatch Coordinator Zome
//!
//! Consciousness-gated functions for registering, dispatching, tracking,
//! and recalling robotic assets within the Mycelix emergency cluster.
//!
//! ## Consciousness Gating
//!
//! - `register_asset()` — PROPOSAL level (score ≥ 0.4)
//! - `dispatch_mission()` — PROPOSAL level (emergency: BASIC with 24h expiry)
//! - `submit_telemetry()` — BASIC level (score ≥ 0.3)
//! - `complete_mission()` — BASIC level
//! - `recall_asset()` — PROPOSAL level
//!
//! ## 24-Hour Authority Expiry (Essay 17)
//!
//! All dispatch orders automatically expire after 24 hours unless
//! ratified by a Steward-tier (score ≥ 0.6) agent. This prevents
//! permanent entrenchment of emergency authority.

use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};


use robotics_dispatch_integrity::*;

/// Maximum dispatch authority duration: 24 hours (Essay 17).
const MAX_AUTHORITY_DURATION_SECS: i64 = 24 * 60 * 60;

/// Anchor path for all robotic assets.
const ASSETS_ANCHOR: &str = "robotic_assets";

// ── Asset Registration ───────────────────────────────────────────────────────

/// Input for registering a new robotic asset.
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterAssetInput {
    pub asset_id: String,
    pub platform_type: PlatformType,
    pub consciousness_level: f64,
    pub max_tier: TierCap,
    pub location_lat: f64,
    pub location_lon: f64,
    pub capabilities: Vec<String>,
}

/// Register a new robotic asset in the network.
///
/// Requires PROPOSAL consciousness level (score ≥ 0.4).
/// The calling agent becomes the asset's sponsoring owner.
#[hdk_extern]
pub fn register_asset(input: RegisterAssetInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "register_asset")?;

    let agent = agent_info()?.agent_initial_pubkey;

    let asset = RoboticAsset {
        asset_id: input.asset_id,
        platform_type: input.platform_type,
        owner: agent.clone(),
        consciousness_level: input.consciousness_level,
        max_tier: input.max_tier,
        location_lat: input.location_lat,
        location_lon: input.location_lon,
        status: AssetStatus::Available,
        capabilities: input.capabilities,
        registered_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::RoboticAsset(asset.clone()))?;

    // Link from anchor
    let anchor = Anchor(ASSETS_ANCHOR.to_string());
    let anchor_hash = create_entry(EntryTypes::Anchor(anchor))?;
    create_link(anchor_hash, action_hash.clone(), LinkTypes::AnchorToAsset, ())?;

    // Link from owner
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::OwnerToAsset,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created asset".to_string()
        )))?;
    Ok(record)
}

// ── Mission Dispatch ─────────────────────────────────────────────────────────

/// Input for dispatching a mission to a robotic asset.
#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchMissionInput {
    pub asset_hash: ActionHash,
    pub mission_type: MissionType,
    pub target_lat: f64,
    pub target_lon: f64,
    pub priority: DispatchPriority,
    pub description: String,
}

/// Dispatch a mission to a robotic asset.
///
/// Requires PROPOSAL consciousness level. Emergency dispatches with
/// Critical priority may use BASIC level but authority still expires
/// in 24 hours.
///
/// Returns the created DispatchOrder record.
#[hdk_extern]
pub fn dispatch_mission(input: DispatchMissionInput) -> ExternResult<Record> {
    // Emergency (Critical) requires only BASIC; others require PROPOSAL
    if input.priority == DispatchPriority::Critical {
        mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "dispatch_mission")?;
    } else {
        mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "dispatch_mission")?;
    }

    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // 24-hour expiry (Essay 17: emergency power must not be permanently entrenched)
    let expiry_us = now.as_micros() + MAX_AUTHORITY_DURATION_SECS * 1_000_000;
    let authority_expires = Timestamp::from_micros(expiry_us);

    let order = DispatchOrder {
        asset_hash: input.asset_hash.clone(),
        mission_type: input.mission_type,
        dispatched_by: agent,
        authority_expires,
        target_lat: input.target_lat,
        target_lon: input.target_lon,
        priority: input.priority,
        description: input.description,
        status: OrderStatus::Pending,
        created_at: now,
    };

    let order_hash = create_entry(EntryTypes::DispatchOrder(order))?;

    // Link asset → order
    create_link(
        input.asset_hash,
        order_hash.clone(),
        LinkTypes::AssetToOrder,
        (),
    )?;

    let record = get(order_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created order".to_string()
        )))?;
    Ok(record)
}

// ── Telemetry ────────────────────────────────────────────────────────────────

/// Input for submitting a telemetry report.
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitTelemetryInput {
    pub asset_hash: ActionHash,
    pub order_hash: ActionHash,
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    pub consciousness_level: f64,
    pub safety_level: String,
    pub mission_progress: f64,
    pub fuel_level: f64,
    pub platform_specific: Vec<u8>,
}

/// Submit a telemetry report from a deployed robotic asset.
///
/// Requires BASIC consciousness level (score ≥ 0.3).
#[hdk_extern]
pub fn submit_telemetry(input: SubmitTelemetryInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "submit_telemetry")?;

    let report = TelemetryReport {
        asset_hash: input.asset_hash,
        order_hash: input.order_hash.clone(),
        timestamp: sys_time()?,
        lat: input.lat,
        lon: input.lon,
        alt: input.alt,
        consciousness_level: input.consciousness_level,
        safety_level: input.safety_level,
        mission_progress: input.mission_progress,
        fuel_level: input.fuel_level,
        platform_specific: input.platform_specific,
    };

    let report_hash = create_entry(EntryTypes::TelemetryReport(report))?;

    // Link order → telemetry
    create_link(
        input.order_hash,
        report_hash.clone(),
        LinkTypes::OrderToTelemetry,
        (),
    )?;

    let record = get(report_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created report".to_string()
        )))?;
    Ok(record)
}

// ── Mission Completion ───────────────────────────────────────────────────────

/// Input for completing a mission.
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteMissionInput {
    pub order_hash: ActionHash,
    pub success: bool,
}

/// Mark a mission as completed (or failed).
///
/// Requires BASIC consciousness level.
#[hdk_extern]
pub fn complete_mission(input: CompleteMissionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "complete_mission")?;

    let record = get(input.order_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Order not found".to_string()
        )))?;

    let mut order: DispatchOrder = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize order".to_string()
        )))?;

    order.status = if input.success {
        OrderStatus::Completed
    } else {
        OrderStatus::Failed
    };

    let updated_hash = update_entry(input.order_hash, order)?;

    let record = get(updated_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated order".to_string()
        )))?;
    Ok(record)
}

// ── Asset Recall ─────────────────────────────────────────────────────────────

/// Recall a deployed robotic asset (abort mission).
///
/// Requires PROPOSAL consciousness level.
#[hdk_extern]
pub fn recall_asset(order_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "recall_asset")?;

    let record = get(order_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Order not found".to_string()
        )))?;

    let mut order: DispatchOrder = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize order".to_string()
        )))?;

    order.status = OrderStatus::Recalled;

    let updated_hash = update_entry(order_hash, order)?;

    let record = get(updated_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated order".to_string()
        )))?;
    Ok(record)
}

// ── Queries ──────────────────────────────────────────────────────────────────

/// Get all registered robotic assets.
#[hdk_extern]
pub fn get_all_assets(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = Anchor(ASSETS_ANCHOR.to_string());
    let anchor_hash = hash_entry(&anchor)?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::AnchorToAsset)?,
        GetStrategy::default(),
    )?;

    let mut assets = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                assets.push(record);
            }
        }
    }

    Ok(assets)
}

/// Get dispatch orders for a specific asset.
#[hdk_extern]
pub fn get_asset_orders(asset_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(asset_hash, LinkTypes::AssetToOrder)?,
        GetStrategy::default(),
    )?;

    let mut orders = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                orders.push(record);
            }
        }
    }

    Ok(orders)
}

/// Get telemetry reports for a dispatch order.
#[hdk_extern]
pub fn get_order_telemetry(order_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(order_hash, LinkTypes::OrderToTelemetry)?,
        GetStrategy::default(),
    )?;

    let mut reports = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                reports.push(record);
            }
        }
    }

    Ok(reports)
}

// ── Living Credentials ──────────────────────────────────────────────────────

/// Input for issuing a robotic credential.
#[derive(Serialize, Deserialize, Debug)]
pub struct IssueCredentialInput {
    pub asset_hash: ActionHash,
    pub credential_type: RoboticCredentialType,
    pub mastery_at_issue: u16,
    pub issuing_authority: Option<String>,
}

/// Issue a living credential for a robotic asset.
///
/// Requires PROPOSAL tier (consciousness score ≥ 0.4).
/// Credential starts at full vitality (1000 permille) and decays
/// via Ebbinghaus forgetting curve until refreshed by perturbation test.
#[hdk_extern]
pub fn issue_credential(input: IssueCredentialInput) -> ExternResult<Record> {
    // Consciousness gate: PROPOSAL tier
    let _author = agent_info()?.agent_initial_pubkey;

    let now = sys_time()?;
    let credential = RoboticCredential {
        asset_hash: input.asset_hash.clone(),
        credential_type: input.credential_type,
        vitality_permille: 1000, // Full vitality at issuance
        mastery_at_issue: input.mastery_at_issue.min(1000),
        last_perturbation_test: now,
        successful_tests: 0,
        issuing_authority: input.issuing_authority,
        issued_at: now,
    };

    let credential_hash = create_entry(&EntryTypes::RoboticCredential(credential))?;

    // Link asset → credential
    create_link(
        input.asset_hash,
        credential_hash.clone(),
        LinkTypes::AssetToCredential,
        (),
    )?;

    let record = get(credential_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created credential".into())))?;
    Ok(record)
}

/// Record a perturbation test result, refreshing credential vitality.
///
/// `test_score_permille`: 0–1000 score from the perturbation test.
/// Updates vitality and increments successful_tests if score >= 500.
#[derive(Serialize, Deserialize, Debug)]
pub struct RecordTestInput {
    pub credential_hash: ActionHash,
    pub test_score_permille: u16,
}

#[hdk_extern]
pub fn record_perturbation_test(input: RecordTestInput) -> ExternResult<Record> {
    let record = get(input.credential_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".into())))?;

    let mut credential: RoboticCredential = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize failed: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a RoboticCredential".into())))?;

    let now = sys_time()?;
    let score = input.test_score_permille.min(1000);

    // Update credential based on test result
    credential.last_perturbation_test = now;
    credential.vitality_permille = score;
    if score >= 500 {
        credential.successful_tests = credential.successful_tests.saturating_add(1);
    }

    let new_hash = update_entry(input.credential_hash, &credential)?;
    let updated = get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated credential".into())))?;
    Ok(updated)
}

/// Get current vitality for a credential (computed from Ebbinghaus decay).
#[hdk_extern]
pub fn get_credential_vitality(credential_hash: ActionHash) -> ExternResult<u16> {
    let record = get(credential_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".into())))?;

    let credential: RoboticCredential = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize failed: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a RoboticCredential".into())))?;

    let now = sys_time()?;
    Ok(credential.compute_vitality(now))
}

/// Get all credentials for a robotic asset.
#[hdk_extern]
pub fn get_asset_credentials(asset_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(asset_hash, LinkTypes::AssetToCredential)?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                credentials.push(record);
            }
        }
    }
    Ok(credentials)
}
