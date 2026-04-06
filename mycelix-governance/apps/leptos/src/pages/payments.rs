// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! SAP summary — full wallet lives in the Finance cluster.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn PaymentsPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="summary-page">
            <h1>"SAP Wallet"</h1>
            <div class="summary-card">
                <span class="summary-balance">
                    {move || format!("{:.2} SAP", fin.sap_balance.get().display_balance())}
                </span>
                <span class="summary-label">"transferable with 2%/yr demurrage"</span>
            </div>
            <a href="http://localhost:8109/sap" class="cross-cluster-link">
                "View full SAP wallet \u{2192} Finance"
            </a>
        </div>
    }
}
