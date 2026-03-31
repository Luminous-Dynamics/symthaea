// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Treasury page: the circulatory system of the commons.
//!
//! The 25% inalienable reserve is the heart — it never empties.
//! Contributions flow in (veins), allocations flow out (arteries).
//! Demurrage compost is the nutrient return cycle.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn TreasuryPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="treasury-page" data-page="treasury" role="main">
            <h1 class="page-title">"Treasury"</h1>
            <p class="page-subtitle">"the circulatory system of the commons"</p>

            {move || {
                fin.treasury.get().map(|t| {
                    let reserve_pct = (t.reserve_health() * 100.0).min(100.0);
                    let available_pct = 100.0 - reserve_pct;
                    let balance_display = t.display_balance();
                    let reserve_display = t.display_reserve();
                    let available_display = t.display_available();

                    view! {
                        <section
                            class="treasury-organism"
                            aria-label="treasury health"
                            data-section="treasury-overview"
                            data-balance=t.balance.to_string()
                            data-reserve=t.inalienable_reserve.to_string()
                            data-available=t.available.to_string()
                            data-reserve-ratio=format!("{:.2}", t.reserve_ratio)
                        >
                            // The heart (inalienable reserve)
                            <div class="treasury-heart" data-component="reserve">
                                <div class="heart-label">"inalienable reserve"</div>
                                <div class="heart-value" data-metric="reserve">
                                    {format!("{reserve_display:.0} SAP")}
                                </div>
                                <div class="heart-ratio">
                                    {format!("{reserve_pct:.0}% of total — this can never be withdrawn")}
                                </div>
                                <div class="heart-pulse-ring"></div>
                            </div>

                            // Circulatory visualization
                            <div class="treasury-flow" data-component="circulation">
                                <div class="flow-total">
                                    <span class="flow-label">"total balance"</span>
                                    <span class="flow-value" data-metric="total-balance">
                                        {format!("{balance_display:.0} SAP")}
                                    </span>
                                </div>
                                <div class="flow-bar">
                                    <div
                                        class="flow-reserve"
                                        style=format!("width: {reserve_pct:.1}%")
                                        title="inalienable reserve"
                                    ></div>
                                    <div
                                        class="flow-available"
                                        style=format!("width: {available_pct:.1}%")
                                        title="available for allocation"
                                    ></div>
                                </div>
                                <div class="flow-available-section">
                                    <span class="flow-label">"available for allocation"</span>
                                    <span class="flow-value" data-metric="available">
                                        {format!("{available_display:.0} SAP")}
                                    </span>
                                </div>
                            </div>

                            // Demurrage return cycle
                            <div class="treasury-compost" data-component="demurrage-cycle">
                                <div class="compost-label">"demurrage composting"</div>
                                <p class="compost-desc">
                                    "SAP naturally decomposes at 2%/yr. The compost returns to the commons: "
                                    "70% to your local pool, 20% regional, 10% global."
                                </p>
                                <div class="compost-split">
                                    <span class="compost-local" data-split="local">"70% local"</span>
                                    <span class="compost-regional" data-split="regional">"20% regional"</span>
                                    <span class="compost-global" data-split="global">"10% global"</span>
                                </div>
                            </div>
                        </section>
                    }
                })
            }}
        </div>
    }
}
