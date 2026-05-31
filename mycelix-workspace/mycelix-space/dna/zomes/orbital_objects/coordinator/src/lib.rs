// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Orbital Objects Coordinator Zome
//!
//! Provides the callable functions for managing the orbital object catalog:
//! - Register new objects
//! - Submit TLE updates
//! - Claim operator status
//! - Query catalog

use hdk::prelude::*;
use mycelix_space_shared::{DataSourceType, QualityScore, SpaceTimestamp};
use orbital_mechanics::tle::TwoLineElement;
use orbital_objects_integrity::*;

/// Register a new orbital object in the catalog
#[hdk_extern]
pub fn register_object(input: RegisterObjectInput) -> ExternResult<ActionHash> {
    if input.intl_designator.is_empty() || input.intl_designator.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "International designator must be 1-256 characters".into()
        )));
    }
    if input.name.is_empty() || input.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Name must be 1-256 characters".into()
        )));
    }
    if let Some(ref country) = input.country {
        if country.is_empty() || country.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Country must be 1-256 characters".into()
            )));
        }
    }
    let agent = agent_info()?.agent_initial_pubkey;

    let object = OrbitalObject {
        norad_id: input.norad_id,
        intl_designator: input.intl_designator,
        name: input.name,
        object_type: input.object_type,
        country: input.country,
        launch_date: input.launch_date,
        decay_date: None,
        status: input.status.unwrap_or(OperationalStatus::Unknown),
        created_at: SpaceTimestamp::now(),
        created_by: agent,
    };

    let action_hash = create_entry(&EntryTypes::OrbitalObject(object))?;

    Ok(action_hash)
}

/// Input for registering a new object
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterObjectInput {
    pub norad_id: u32,
    pub intl_designator: String,
    pub name: String,
    pub object_type: ObjectType,
    pub country: Option<String>,
    pub launch_date: Option<SpaceTimestamp>,
    pub status: Option<OperationalStatus>,
}

/// Submit a TLE for an object
#[hdk_extern]
pub fn submit_tle(input: SubmitTleInput) -> ExternResult<ActionHash> {
    if input.line1.is_empty() || input.line1.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "TLE line 1 must be 1-256 characters".into()
        )));
    }
    if input.line2.is_empty() || input.line2.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "TLE line 2 must be 1-256 characters".into()
        )));
    }
    let agent = agent_info()?.agent_initial_pubkey;

    // Parse TLE using orbital-mechanics library for validation and epoch extraction
    let parsed_tle = TwoLineElement::parse_lines(None, &input.line1, &input.line2)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid TLE: {}", e))))?;

    // Verify NORAD ID in TLE matches the submitted ID
    if parsed_tle.norad_id != input.norad_id {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "NORAD ID mismatch: input {} vs TLE {}",
            input.norad_id, parsed_tle.norad_id
        ))));
    }

    let epoch = SpaceTimestamp::from_datetime(parsed_tle.epoch);

    let tle = TleRecord {
        norad_id: input.norad_id,
        line1: input.line1,
        line2: input.line2,
        epoch,
        source: input.source.unwrap_or(DataSourceType::SpaceTrack),
        quality: input.quality.unwrap_or_default(),
        submitted_at: SpaceTimestamp::now(),
        submitted_by: agent,
    };

    let action_hash = create_entry(&EntryTypes::TleRecord(tle))?;

    // Create a link from a NORAD ID anchor to this TLE for DHT discoverability
    let norad_anchor = anchor_for_norad_id(input.norad_id)?;
    create_link(
        norad_anchor,
        action_hash.clone(),
        LinkTypes::ObjectToTles,
        LinkTag::new(format!("tle:{}", input.norad_id)),
    )?;

    Ok(action_hash)
}

/// Deterministic anchor hash for a NORAD ID, used for DHT link indexing.
/// Uses the TypedPath primitive for a stable, well-known anchor on the DHT.
fn anchor_for_norad_id(norad_id: u32) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("norad_tles.{}", norad_id));
    let typed = path.typed(LinkTypes::ObjectToTles)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Input for submitting a TLE
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitTleInput {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub source: Option<DataSourceType>,
    pub quality: Option<QualityScore>,
}

/// Claim operator status for an object
#[hdk_extern]
pub fn claim_operator(input: ClaimOperatorInput) -> ExternResult<ActionHash> {
    if input.organization.is_empty() || input.organization.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Organization must be 1-256 characters".into()
        )));
    }
    if let Some(ref contact) = input.contact {
        if contact.is_empty() || contact.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Contact must be 1-256 characters".into()
            )));
        }
    }
    let agent = agent_info()?.agent_initial_pubkey;

    let claim = OperatorClaim {
        norad_id: input.norad_id,
        operator: agent,
        organization: input.organization,
        contact: input.contact,
        claimed_at: SpaceTimestamp::now(),
        verified: false, // Requires separate verification process
        verification_hash: input.verification_hash,
    };

    let action_hash = create_entry(&EntryTypes::OperatorClaim(claim))?;

    Ok(action_hash)
}

/// Input for claiming operator status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaimOperatorInput {
    pub norad_id: u32,
    pub organization: String,
    pub contact: Option<String>,
    pub verification_hash: Option<[u8; 32]>,
}

/// Input for listing objects
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListObjectsInput {
    pub object_type: Option<ObjectType>,
    pub status: Option<OperationalStatus>,
    pub limit: Option<u32>,
}

/// Input for getting TLE history
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetTleHistoryInput {
    pub norad_id: u32,
    pub start_time: Option<SpaceTimestamp>,
    pub end_time: Option<SpaceTimestamp>,
    pub limit: Option<u32>,
}

/// Get the latest TLE lines for a set of NORAD IDs.
///
/// Queries the DHT via get_links for TleRecord entries submitted by any agent,
/// then returns the most recent TLE (by epoch) for each requested NORAD ID.
#[hdk_extern]
pub fn get_latest_tles(norad_ids: Vec<u32>) -> ExternResult<Vec<TleLines>> {
    if norad_ids.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one NORAD ID is required".into()
        )));
    }
    if norad_ids.len() > 1000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum 1000 NORAD IDs per query".into()
        )));
    }
    let mut latest: std::collections::HashMap<u32, TleRecord> = std::collections::HashMap::new();

    for norad_id in &norad_ids {
        let anchor = anchor_for_norad_id(*norad_id)?;
        let links = get_links(
            LinkQuery::try_new(anchor, LinkTypes::ObjectToTles)?,
            GetStrategy::Network,
        )?;

        for link in links {
            let Some(target) = link.target.into_action_hash() else {
                continue;
            };
            let Some(record) = get(target, GetOptions::default())? else {
                continue;
            };
            if let Some(tle) = record.entry().to_app_option::<TleRecord>().ok().flatten() {
                let dominated = latest
                    .get(&tle.norad_id)
                    .map(|existing| tle.epoch.micros > existing.epoch.micros)
                    .unwrap_or(true);
                if dominated {
                    latest.insert(tle.norad_id, tle);
                }
            }
        }
    }

    Ok(latest
        .into_values()
        .map(|tle| TleLines {
            norad_id: tle.norad_id,
            line1: tle.line1,
            line2: tle.line2,
        })
        .collect())
}

/// Minimal TLE output for cross-zome consumption
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TleLines {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
}
