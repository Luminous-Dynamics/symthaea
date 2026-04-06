// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! TEND summary — full exchange interface lives in the Finance cluster.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn TendPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="summary-page">
            <h1>"Mutual Care (TEND)"</h1>
            <div class="summary-card">
                <span class="summary-balance">
                    {move || format!("{:+} hours", fin.tend_balance.get().balance)}
                </span>
                <span class="summary-label">
                    {move || fin.tend_balance.get().equilibrium_label()}
                </span>
                <div class="summary-stats">
                    <span>{move || format!("{:.1}h given", fin.tend_balance.get().total_provided)}</span>
                    <span>{move || format!("{:.1}h received", fin.tend_balance.get().total_received)}</span>
                    <span>{move || format!("{} exchanges", fin.tend_balance.get().exchange_count)}</span>
                </div>
            </div>
            <a href="http://localhost:8109/tend" class="cross-cluster-link">
                "View full TEND exchange \u{2192} Finance"
            </a>
        </div>
    }
}
