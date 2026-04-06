// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Finance.
//!
//! Mock-first pattern: initializes with mock data for instant render,
//! then attempts real data from the Holochain conductor after 4 seconds.

use leptos::prelude::*;
use mycelix_leptos_core::holochain_provider;

/// Domain state shared via Leptos context.
#[derive(Clone)]
pub struct FinanceCtx {
    pub items: RwSignal<Vec<String>>,
    // Add your domain-specific signals here
}

pub fn provide_finance_context() {
    let state = FinanceCtx {
        items: RwSignal::new(vec!["Mock item 1".into(), "Mock item 2".into()]),
    };
    provide_context(state.clone());

    // Attempt real data load after 4s
    let hc = holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            // Replace with your actual zome call:
            // if let Ok(real_items) = hc.call_zome::<(), Vec<String>>(
            //     "finance", "your_zome", "list_items", &()
            // ).await {
            //     state.items.set(real_items);
            // }
            let _ = &state; // suppress unused warning
        }
    });
}

pub fn use_finance_context() -> FinanceCtx {
    expect_context::<FinanceCtx>()
}
