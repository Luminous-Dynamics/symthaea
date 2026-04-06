// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Oracle summary — full view lives in the Finance cluster.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn OraclePage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="summary-page">
            <h1>"Oracle"</h1>
            <div class="summary-card">
                <span class="summary-balance">
                    {move || fin.oracle_state.get().vitality.to_string()}
                </span>
                <span class="summary-label">
                    {move || format!("Tier: {} | Limit: {} TEND",
                        fin.oracle_state.get().tier.label(),
                        fin.oracle_state.get().tier.credit_limit()
                    )}
                </span>
            </div>
            <a href="http://localhost:8109/oracle" class="cross-cluster-link">
                "View full oracle \u{2192} Finance"
            </a>
        </div>
    }
}
