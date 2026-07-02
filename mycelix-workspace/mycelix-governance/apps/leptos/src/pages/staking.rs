// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Staking summary — full view lives in the Finance cluster.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn StakingPage() -> impl IntoView {
    let fin = use_finance();

    let total_staked = move || {
        let stakes = fin.stakes.get();
        let total: u64 = stakes.iter().map(|s| s.sap_amount).sum();
        total as f64 / 1_000_000.0
    };

    view! {
        <div class="summary-page">
            <h1>"Staking"</h1>
            <div class="summary-card">
                <span class="summary-balance">
                    {move || format!("{:.2} SAP", total_staked())}
                </span>
                <span class="summary-label">
                    {move || format!("{} active positions", fin.stakes.get().len())}
                </span>
            </div>
            <a href="http://localhost:8109/staking" class="cross-cluster-link">
                "View full staking \u{2192} Finance"
            </a>
        </div>
    }
}
