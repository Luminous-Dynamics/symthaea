// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn StakingPage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-staking">
            <h1>"Staking"</h1>
            <p class="subtitle">"Stake SAP to increase your governance weight."</p>

            <div class="stake-list">
                {move || ctx.stakes.get().iter().map(|s| {
                    let amount = s.sap_amount as f64 / 1_000_000.0;
                    let weight = s.stake_weight;
                    let status = s.status.label();
                    view! {
                        <div class="stake-card">
                            <div class="stake-header">
                                <span class="stake-amount">{format!("{amount:.2} SAP")}</span>
                                <span class="badge">{status}</span>
                            </div>
                            <span class="stake-weight">{format!("Weight: {weight:.2}")}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
