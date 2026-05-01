// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anchor registry coordinator: manage known-position nodes.
//!
//! Anchors are nodes with surveyed or GPS-determined positions that
//! serve as reference points for cooperative trilateration.

use anchor_registry_integrity::*;
use hdk::prelude::*;
use mycelix_position_shared::{
    AnchorCertification, AnchorNode, PositionError, PositionErrorCode, PositionTimestamp,
    SurveyMethod,
};

// ============================================================================
// GEOHASH (simplified 3-char prefix for spatial indexing)
// ============================================================================

/// Compute a 3-character geohash prefix for spatial DHT indexing.
/// Groups nearby anchors for efficient regional queries.
fn geohash_prefix(lat: f64, lon: f64) -> String {
    // Simple grid: 10° lat × 10° lon cells → ~1100km × ~1100km at equator
    let lat_idx = ((lat + 90.0) / 10.0).floor() as i32;
    let lon_idx = ((lon + 180.0) / 10.0).floor() as i32;
    format!("{:02}{:02}", lat_idx.clamp(0, 17), lon_idx.clamp(0, 35))
}

// ============================================================================
// DHT ANCHORS
// ============================================================================

fn anchor_by_region(geohash: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("anchors.region.{}", geohash));
    let typed = path.typed(LinkTypes::AnchorsByRegion)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

fn anchor_all() -> ExternResult<AnyLinkableHash> {
    let path = Path::from("anchors.all");
    let typed = path.typed(LinkTypes::AllAnchors)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

// ============================================================================
// INPUT TYPES
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterAnchorInput {
    pub node_id: String,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    pub accuracy_m: f64,
    pub survey_method: SurveyMethod,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertifyAnchorInput {
    pub anchor_node_id: String,
    pub verified_accuracy_m: f64,
    pub certification_method: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionQueryInput {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
}

// ============================================================================
// EXTERN FUNCTIONS
// ============================================================================

/// Register a new anchor node. Requires Participant tier (identity >= 0.15).
#[hdk_extern]
pub fn register_anchor(input: RegisterAnchorInput) -> ExternResult<ActionHash> {
    // TODO: gate_position_operation (requires identity cluster wiring)
    // For standalone deployment, allow all registrations.

    let agent = agent_info()?.agent_initial_pubkey;

    let node = AnchorNode {
        node_id: input.node_id.clone(),
        latitude_deg: input.latitude_deg,
        longitude_deg: input.longitude_deg,
        altitude_m: input.altitude_m,
        accuracy_m: input.accuracy_m,
        survey_method: input.survey_method,
        certified_by: None,
        registered_by: agent,
        registered_at: PositionTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::AnchorNode(node))?;

    // Index by region (geohash)
    let geohash = geohash_prefix(input.latitude_deg, input.longitude_deg);
    let region_anchor = anchor_by_region(&geohash)?;
    create_link(
        region_anchor,
        action_hash.clone(),
        LinkTypes::AnchorsByRegion,
        LinkTag::new(format!("anchor:{}", input.node_id)),
    )?;

    // Global index
    let all_anchor = anchor_all()?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllAnchors,
        LinkTag::new(format!("anchor:{}", input.node_id)),
    )?;

    Ok(action_hash)
}

/// Certify an anchor's accuracy. Requires Steward tier (identity >= 0.50).
#[hdk_extern]
pub fn certify_anchor(input: CertifyAnchorInput) -> ExternResult<ActionHash> {
    // TODO: gate_position_operation with Steward requirement

    let agent = agent_info()?.agent_initial_pubkey;

    let cert = AnchorCertification {
        anchor_node_id: input.anchor_node_id.clone(),
        certifier: agent,
        verified_accuracy_m: input.verified_accuracy_m,
        certification_method: input.certification_method,
        certified_at: PositionTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::AnchorCertification(cert))?;
    Ok(action_hash)
}

/// Get all anchors in a region (same 10°×10° cell).
#[hdk_extern]
pub fn get_anchors_in_region(input: RegionQueryInput) -> ExternResult<Vec<AnchorNode>> {
    let geohash = geohash_prefix(input.latitude_deg, input.longitude_deg);
    let region_anchor = anchor_by_region(&geohash)?;

    let links = get_links(
        LinkQuery::try_new(region_anchor, LinkTypes::AnchorsByRegion)?,
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
        if let Some(node) = record.entry().to_app_option::<AnchorNode>().ok().flatten() {
            results.push(node);
        }
    }
    Ok(results)
}

/// Get all registered anchors (global query).
#[hdk_extern]
pub fn get_all_anchors(_: ()) -> ExternResult<Vec<AnchorNode>> {
    let all_anchor = anchor_all()?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllAnchors)?,
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
        if let Some(node) = record.entry().to_app_option::<AnchorNode>().ok().flatten() {
            results.push(node);
        }
    }
    Ok(results)
}
