// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Governance context: proposals, votes, councils, constitution.
//!
//! Starts with mock data immediately (instant render), then replaces
//! with real conductor data if connected.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use governance_leptos_types::*;
use crate::mock_data;
use mycelix_leptos_core::holochain_provider::use_holochain;

#[derive(Clone)]
pub struct GovernanceCtx {
    pub proposals: RwSignal<Vec<ProposalView>>,
    pub my_votes: RwSignal<Vec<PhiVoteView>>,
    pub councils: RwSignal<Vec<CouncilView>>,
    pub charter: RwSignal<Option<CharterView>>,
    pub timelocks: RwSignal<Vec<TimelockView>>,
    pub budget_cycles: RwSignal<Vec<BudgetCycleView>>,
    pub delegations: RwSignal<Vec<DelegationView>>,
    pub thresholds: RwSignal<ConsciousnessThresholds>,
    pub my_agent_did: RwSignal<String>,
}

pub fn provide_governance_context() -> GovernanceCtx {
    let ctx = GovernanceCtx {
        proposals: RwSignal::new(mock_data::mock_proposals()),
        my_votes: RwSignal::new(mock_data::mock_votes()),
        councils: RwSignal::new(mock_data::mock_councils()),
        charter: RwSignal::new(Some(mock_data::mock_charter())),
        timelocks: RwSignal::new(Vec::new()),
        budget_cycles: RwSignal::new(Vec::new()),
        delegations: RwSignal::new(Vec::new()),
        thresholds: RwSignal::new(ConsciousnessThresholds {
            basic: 0.2,
            proposal_submission: 0.3,
            voting: 0.4,
            constitutional: 0.6,
        }),
        my_agent_did: RwSignal::new("did:mycelix:mock-citizen".into()),
    };

    provide_context(ctx.clone());

    // Update homeostasis with active proposal count
    if let Some(set_pending) = use_context::<WriteSignal<u32>>() {
        let active = ctx.proposals.get_untracked()
            .iter()
            .filter(|p| p.status.is_active())
            .count() as u32;
        set_pending.set(active);
    }

    // Try to load real data from conductor
    let ctx_for_load = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4000).await;
        try_load_real_data(ctx_for_load).await;
    });

    ctx
}

async fn try_load_real_data(ctx: GovernanceCtx) {
    let hc = use_holochain();
    if hc.is_mock() {
        web_sys::console::log_1(&"[Civic] Running in mock mode — using simulated governance data.".into());
        return;
    }

    web_sys::console::log_1(&"[Civic] Loading real governance data from conductor…".into());

    // Load proposals — conductor returns Vec<Record> decoded to JSON values.
    match hc.call_zome::<(), Vec<serde_json::Value>>("governance", "proposals", "get_active_proposals", &()).await {
        Ok(records) => {
            let proposals: Vec<ProposalView> = records
                .iter()
                .filter_map(|record| {
                    let entry = record.get("entry")?.get("Present")?;
                    serde_json::from_value::<ProposalView>(entry.clone()).ok()
                })
                .collect();
            web_sys::console::log_1(&format!("[Civic] Loaded {} proposals from conductor", proposals.len()).into());
            if !proposals.is_empty() {
                ctx.proposals.set(proposals);
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Civic] Could not load proposals: {e}").into());
        }
    }

    // Load consciousness thresholds
    match hc.call_zome::<(), ConsciousnessThresholds>("governance", "bridge", "get_consciousness_thresholds", &()).await {
        Ok(thresholds) => {
            ctx.thresholds.set(thresholds);
            web_sys::console::log_1(&"[Civic] Loaded consciousness thresholds".into());
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Civic] Could not load thresholds: {e}").into());
        }
    }
}

pub fn use_governance() -> GovernanceCtx {
    expect_context::<GovernanceCtx>()
}
