// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hearth context: current hearth, member list, user's role, bonds, and all domain data.
//!
//! On init: tries to load from conductor via real zome calls.
//! Falls back to mock data if conductor is unavailable.
//! Uses `RwSignal` so pages can both read and mutate.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use hearth_leptos_types::*;
use crate::mock_data;
use mycelix_leptos_core::holochain_provider::use_holochain;
use crate::record_bridge::{self, WireRecord};

/// The active hearth context shared across pages.
#[derive(Clone)]
pub struct HearthCtx {
    pub current_hearth: RwSignal<Option<HearthView>>,
    pub members: RwSignal<Vec<MemberView>>,
    pub my_role: RwSignal<Option<MemberRole>>,
    pub bonds: RwSignal<Vec<BondView>>,
    pub care_schedules: RwSignal<Vec<CareScheduleView>>,
    pub decisions: RwSignal<Vec<DecisionView>>,
    pub votes: RwSignal<Vec<VoteView>>,
    pub gratitude: RwSignal<Vec<GratitudeExpressionView>>,
    pub stories: RwSignal<Vec<StoryView>>,
    pub rhythms: RwSignal<Vec<RhythmView>>,
    pub presence: RwSignal<Vec<PresenceView>>,
    pub emergency_alerts: RwSignal<Vec<EmergencyAlertView>>,
    pub resources: RwSignal<Vec<ResourceView>>,
    pub milestones: RwSignal<Vec<MilestoneView>>,
    pub autonomy_profiles: RwSignal<Vec<AutonomyProfileView>>,
    pub my_agent: RwSignal<String>,
}

/// Initialize hearth context.
///
/// Starts with mock data immediately (so the UI renders instantly),
/// then attempts to load real data from the conductor in the background.
/// If the conductor responds, mock data is replaced with real data.
pub fn provide_hearth_context() -> HearthCtx {
    let ctx = HearthCtx {
        current_hearth: RwSignal::new(Some(mock_data::mock_hearth())),
        members: RwSignal::new(mock_data::mock_members()),
        my_role: RwSignal::new(Some(MemberRole::Adult)),
        bonds: RwSignal::new(mock_data::mock_bonds()),
        care_schedules: RwSignal::new(mock_data::mock_care_schedules()),
        decisions: RwSignal::new(mock_data::mock_decisions()),
        votes: RwSignal::new(mock_data::mock_votes()),
        gratitude: RwSignal::new(mock_data::mock_gratitude()),
        stories: RwSignal::new(mock_data::mock_stories()),
        rhythms: RwSignal::new(mock_data::mock_rhythms()),
        presence: RwSignal::new(mock_data::mock_presence()),
        emergency_alerts: RwSignal::new(mock_data::mock_emergency_alerts()),
        resources: RwSignal::new(mock_data::mock_resources()),
        milestones: RwSignal::new(mock_data::mock_milestones()),
        autonomy_profiles: RwSignal::new(mock_data::mock_autonomy_profiles()),
        my_agent: RwSignal::new("agent_rowan".into()),
    };

    provide_context(ctx.clone());

    // Feed homeostasis detection
    if let Some(set_care) = use_context::<WriteSignal<u32>>() {
        let care_count = ctx.care_schedules.get_untracked()
            .iter()
            .filter(|c| c.status == CareScheduleStatus::Active)
            .count() as u32;
        set_care.set(care_count);
    }

    // Try to load real data from conductor (async, non-blocking)
    let ctx_for_load = ctx.clone();
    spawn_local(async move {
        // Wait a moment for the conductor connection to establish
        gloo_timers::future::TimeoutFuture::new(4000).await;
        try_load_real_data(ctx_for_load).await;
    });

    ctx
}

/// Attempt to load real data from the Holochain conductor.
/// Replaces mock signals with real data on success.
/// Fails silently on error (mock data remains).
async fn try_load_real_data(ctx: HearthCtx) {
    let hc = use_holochain();
    if hc.is_mock() {
        web_sys::console::log_1(&"[Hearth] Mock mode — using simulated data".into());
        return;
    }

    web_sys::console::log_1(&"[Hearth] Connected — loading real data...".into());

    // Step 1: Get my hearths
    match hc.call_zome_default::<(), Vec<WireRecord>>("hearth_kinship", "get_my_hearths", &()).await {
        Ok(hearth_records) => {
            web_sys::console::log_1(
                &format!("[Hearth] Found {} hearths", hearth_records.len()).into()
            );

            if let Some(first_hearth) = hearth_records.first() {
                let hearth_hash = first_hearth.action_hash_b64();

                // Step 2: Get members for this hearth
                match hc.call_zome_default::<String, Vec<WireRecord>>(
                    "hearth_kinship", "get_hearth_members", &hearth_hash
                ).await {
                    Ok(member_records) => {
                        let members = record_bridge::records_to_members(&member_records);
                        web_sys::console::log_1(
                            &format!("[Hearth] Loaded {} members", members.len()).into()
                        );
                        if !members.is_empty() {
                            ctx.members.set(members);
                        }
                    }
                    Err(e) => web_sys::console::log_1(
                        &format!("[Hearth] get_hearth_members failed: {e}").into()
                    ),
                }

                // Step 3: Get bonds
                match hc.call_zome_default::<String, Vec<WireRecord>>(
                    "hearth_kinship", "get_kinship_graph", &hearth_hash
                ).await {
                    Ok(bond_records) => {
                        let bonds = record_bridge::records_to_bonds(&bond_records);
                        web_sys::console::log_1(
                            &format!("[Hearth] Loaded {} bonds", bonds.len()).into()
                        );
                        if !bonds.is_empty() {
                            ctx.bonds.set(bonds);
                        }
                    }
                    Err(e) => web_sys::console::log_1(
                        &format!("[Hearth] get_kinship_graph failed: {e}").into()
                    ),
                }

                // Step 4: Get gratitude
                match hc.call_zome_default::<String, Vec<WireRecord>>(
                    "hearth_gratitude", "get_gratitude_stream", &hearth_hash
                ).await {
                    Ok(grat_records) => {
                        let gratitude = record_bridge::records_to_gratitude(&grat_records);
                        web_sys::console::log_1(
                            &format!("[Hearth] Loaded {} gratitude expressions", gratitude.len()).into()
                        );
                        if !gratitude.is_empty() {
                            ctx.gratitude.set(gratitude);
                        }
                    }
                    Err(e) => web_sys::console::log_1(
                        &format!("[Hearth] get_gratitude_stream failed: {e}").into()
                    ),
                }
            }
        }
        Err(e) => {
            web_sys::console::log_1(
                &format!("[Hearth] get_my_hearths failed: {e} — staying in mock mode").into()
            );
        }
    }
}

pub fn use_hearth() -> HearthCtx {
    expect_context::<HearthCtx>()
}

/// Look up a member's display name by agent key.
pub fn member_name(members: &[MemberView], agent: &str) -> String {
    members.iter()
        .find(|m| m.agent == agent)
        .map(|m| m.display_name.clone())
        .unwrap_or_else(|| {
            // For base64 agent keys, show first 8 chars
            if agent.len() > 8 {
                format!("{}...", &agent[..8])
            } else {
                agent.to_string()
            }
        })
}

/// Simple mock timestamp (seconds since epoch, approximate).
pub fn mock_now() -> i64 {
    1_774_934_400 // ~2026-03-30
}
