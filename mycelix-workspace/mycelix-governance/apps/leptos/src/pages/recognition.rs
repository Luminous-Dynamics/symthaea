// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MYCEL summary — full reputation view lives in the Finance cluster.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn RecognitionPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="summary-page">
            <h1>"MYCEL Reputation"</h1>
            <div class="summary-card">
                <span class="summary-balance">
                    {move || format!("{:.2}", fin.mycel_score.get().score)}
                </span>
                <span class="summary-label">
                    {move || fin.mycel_score.get().tier.label()}
                </span>
            </div>
            <a href="http://localhost:8109/mycel" class="cross-cluster-link">
                "View full MYCEL breakdown \u{2192} Finance"
            </a>
        </div>
    }
}
