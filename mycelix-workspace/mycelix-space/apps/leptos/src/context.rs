// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Space.

use leptos::prelude::*;
use space_leptos_types::*;

#[derive(Clone)]
pub struct SpaceCtx {
    pub objects: RwSignal<Vec<OrbitalObjectView>>,
    pub conjunctions: RwSignal<Vec<ConjunctionEventView>>,
    pub bounties: RwSignal<Vec<DebrisBountyView>>,
    pub proposals: RwSignal<Vec<ManeuverProposalView>>,
}

pub fn provide_space_context() {
    let state = SpaceCtx {
        objects: RwSignal::new(vec![
            OrbitalObjectView { norad_id: 25544, name: "ISS (ZARYA)".into(), object_type: ObjectType::Payload, operator_did: Some("did:mycelix:nasa".into()), inclination_deg: 51.6, period_min: 92.9, apogee_km: 422.0, perigee_km: 418.0, tle_epoch: 1714521600 },
            OrbitalObjectView { norad_id: 48274, name: "CSS (TIANHE)".into(), object_type: ObjectType::Payload, operator_did: Some("did:mycelix:cnsa".into()), inclination_deg: 41.5, period_min: 91.5, apogee_km: 389.0, perigee_km: 382.0, tle_epoch: 1714500000 },
            OrbitalObjectView { norad_id: 43205, name: "COSMOS 2251 DEB".into(), object_type: ObjectType::Debris, operator_did: None, inclination_deg: 74.0, period_min: 99.8, apogee_km: 850.0, perigee_km: 720.0, tle_epoch: 1714400000 },
            OrbitalObjectView { norad_id: 39084, name: "STARLINK-1007".into(), object_type: ObjectType::Payload, operator_did: Some("did:mycelix:spacex".into()), inclination_deg: 53.0, period_min: 95.7, apogee_km: 550.0, perigee_km: 545.0, tle_epoch: 1714521600 },
        ]),
        conjunctions: RwSignal::new(vec![
            ConjunctionEventView { id: "CJ-001".into(), primary_norad: 25544, primary_name: "ISS".into(), secondary_norad: 43205, secondary_name: "COSMOS 2251 DEB".into(), tca_epoch: 1714600000, miss_distance_km: 2.3, collision_probability: 1.2e-4, risk_level: RiskLevel::High },
            ConjunctionEventView { id: "CJ-002".into(), primary_norad: 48274, primary_name: "CSS".into(), secondary_norad: 39084, secondary_name: "STARLINK-1007".into(), tca_epoch: 1714700000, miss_distance_km: 51.6, collision_probability: 4.7e-7, risk_level: RiskLevel::Low },
        ]),
        bounties: RwSignal::new(vec![
            DebrisBountyView { id: "BNT-001".into(), target_norad: 43205, target_name: "COSMOS 2251 DEB".into(), reward_amount: 50000.0, currency: "USD".into(), status: BountyStatus::Open, creator_did: "did:mycelix:esa".into() },
        ]),
        proposals: RwSignal::new(vec![
            ManeuverProposalView { id: "MP-001".into(), conjunction_id: "CJ-001".into(), proposer_did: "did:mycelix:nasa".into(), description: "Raise ISS orbit by 1.2 km to increase miss distance".into(), votes_for: 2, votes_against: 0, quorum: 3, deadline: 1714590000 },
        ]),
    };
    provide_context(state.clone());

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[Space] Conductor connected — loading real data...".into());

            // Load orbital objects
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "orbital_objects", "get_all_objects", &()
            ).await {
                let objects: Vec<OrbitalObjectView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !objects.is_empty() {
                    web_sys::console::log_1(&format!("[Space] Loaded {} orbital objects", objects.len()).into());
                    state.objects.set(objects);
                }
            }

            // Load conjunction events
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "conjunctions", "get_active_conjunctions", &()
            ).await {
                let conjunctions: Vec<ConjunctionEventView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !conjunctions.is_empty() {
                    web_sys::console::log_1(&format!("[Space] Loaded {} conjunctions", conjunctions.len()).into());
                    state.conjunctions.set(conjunctions);
                }
            }

            // Load debris bounties
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "debris_bounties", "get_active_bounties", &()
            ).await {
                let bounties: Vec<DebrisBountyView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !bounties.is_empty() {
                    web_sys::console::log_1(&format!("[Space] Loaded {} bounties", bounties.len()).into());
                    state.bounties.set(bounties);
                }
            }

            // Load maneuver proposals
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "traffic_control", "get_active_proposals", &()
            ).await {
                let proposals: Vec<ManeuverProposalView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !proposals.is_empty() {
                    web_sys::console::log_1(&format!("[Space] Loaded {} proposals", proposals.len()).into());
                    state.proposals.set(proposals);
                }
            }
        }
    });
}

/// Extract the entry from a Holochain Record JSON value.
/// Records have structure: { "entry": { "Present": { ...fields... } } }
fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_space_context() -> SpaceCtx {
    expect_context::<SpaceCtx>()
}
