// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Treasury summary — full view lives in the Finance cluster.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn TreasuryPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="summary-page">
            <h1>"Commons Treasury"</h1>
            {move || fin.treasury.get().map(|t| view! {
                <div class="summary-card">
                    <span class="summary-balance">
                        {format!("{:.2} SAP", t.display_balance())}
                    </span>
                    <span class="summary-label">
                        {format!("Reserve: {:.0}% health", t.reserve_health() * 100.0)}
                    </span>
                </div>
            })}
            <a href="http://localhost:8109/treasury" class="cross-cluster-link">
                "View full treasury \u{2192} Finance"
            </a>
        </div>
    }
}
