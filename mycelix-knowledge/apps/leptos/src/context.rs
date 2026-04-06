// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Knowledge.

use leptos::prelude::*;
use knowledge_leptos_types::*;
use mycelix_mock_data::knowledge as mock;

#[derive(Clone)]
pub struct KnowledgeCtx {
    pub claims: RwSignal<Vec<ClaimView>>,
    pub fact_checks: RwSignal<Vec<FactCheckResultView>>,
    pub inferences: RwSignal<Vec<InferenceView>>,
    pub graph_stats: RwSignal<GraphStatsView>,
}

pub fn provide_knowledge_context() {
    let state = KnowledgeCtx {
        claims: RwSignal::new(mock::claims()),
        fact_checks: RwSignal::new(mock::fact_checks()),
        inferences: RwSignal::new(mock::inferences()),
        graph_stats: RwSignal::new(mock::graph_stats()),
    };
    provide_context(state.clone());

    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            let _ = &state;
        }
    });
}

pub fn use_knowledge_context() -> KnowledgeCtx {
    expect_context::<KnowledgeCtx>()
}
