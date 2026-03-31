// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Staking page: lock SAP as collateral, earn governance weight.
//! Stake weight = 1.0 + MYCEL score (range 1.0-2.0).
//! 21-day unbonding period. Slashing for Byzantine behavior.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;
use finance_leptos_types::*;

#[component]
pub fn StakingPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="staking-page" data-page="staking" role="main">
            <h1 class="page-title">"Staking"</h1>
            <p class="page-subtitle">"lock SAP as collateral — your MYCEL amplifies your weight"</p>

            // MYCEL weight modifier
            <section class="stake-weight-section" aria-label="stake weight" data-section="stake-weight">
                {move || {
                    let mycel = fin.mycel_score.get();
                    let weight = 1.0 + mycel.score;
                    view! {
                        <div
                            class="stake-weight-card"
                            data-mycel-score=format!("{:.2}", mycel.score)
                            data-stake-weight=format!("{:.2}", weight)
                        >
                            <span class="weight-formula">
                                {format!("stake weight = 1.0 + {:.2} MYCEL = {weight:.2}×", mycel.score)}
                            </span>
                            <span class="weight-explanation">
                                "higher MYCEL means your staked SAP carries more governance weight"
                            </span>
                        </div>
                    }
                }}
            </section>

            // Active stakes
            <section class="stakes-section" aria-label="your stakes" data-section="stakes">
                <h2 class="section-title">"Your Stakes"</h2>
                <div class="stakes-list" role="list">
                    {move || {
                        let stakes = fin.stakes.get();
                        if stakes.is_empty() {
                            view! {
                                <div class="empty-state-card">
                                    <p class="empty-state">"no active stakes"</p>
                                    <p class="empty-state-sub">
                                        "staking SAP increases your governance influence. "
                                        "21-day unbonding period protects the commons."
                                    </p>
                                </div>
                            }.into_any()
                        } else {
                            stakes.into_iter().map(|s| {
                                let amount_sap = s.sap_amount as f64 / 1_000_000.0;
                                let status_label = s.status.label().to_string();
                                let status_label2 = status_label.clone();
                                view! {
                                    <div
                                        class=format!("stake-card stake-{}", status_label.to_lowercase())
                                        data-stake-id=s.id.clone()
                                        data-amount=s.sap_amount.to_string()
                                        data-status=status_label
                                        data-weight=format!("{:.2}", s.stake_weight)
                                        role="listitem"
                                    >
                                        <div class="stake-header">
                                            <span class="stake-amount" data-metric="amount">
                                                {format!("{amount_sap:.0} SAP")}
                                            </span>
                                            <span class="stake-status">{status_label2}</span>
                                        </div>
                                        <div class="stake-weight" data-metric="weight">
                                            {format!("weight: {:.2}×", s.stake_weight)}
                                        </div>
                                    </div>
                                }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>
        </div>
    }
}
