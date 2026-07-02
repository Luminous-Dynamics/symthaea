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

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[Knowledge] Conductor connected — loading real data...".into());

            // Load claims
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "claims", "get_claims_by_tag", &"".to_string()
            ).await {
                let claims: Vec<ClaimView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !claims.is_empty() {
                    web_sys::console::log_1(&format!("[Knowledge] Loaded {} claims", claims.len()).into());
                    state.claims.set(claims);
                }
            }

            // Load fact checks
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "factcheck", "get_recent_fact_checks", &()
            ).await {
                let fact_checks: Vec<FactCheckResultView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !fact_checks.is_empty() {
                    web_sys::console::log_1(&format!("[Knowledge] Loaded {} fact checks", fact_checks.len()).into());
                    state.fact_checks.set(fact_checks);
                }
            }

            // Load inferences
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "inference", "get_recent_inferences", &()
            ).await {
                let inferences: Vec<InferenceView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !inferences.is_empty() {
                    web_sys::console::log_1(&format!("[Knowledge] Loaded {} inferences", inferences.len()).into());
                    state.inferences.set(inferences);
                }
            }

            // Load graph stats
            if let Ok(stats) = hc.call_zome_default::<(), GraphStatsView>(
                "graph", "get_graph_stats", &()
            ).await {
                state.graph_stats.set(stats);
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

pub fn use_knowledge_context() -> KnowledgeCtx {
    expect_context::<KnowledgeCtx>()
}
