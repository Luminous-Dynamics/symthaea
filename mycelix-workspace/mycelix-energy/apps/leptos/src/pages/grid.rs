// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_energy_context;

#[component]
pub fn GridPage() -> impl IntoView {
    let ctx = use_energy_context();

    view! {
        <div class="page-grid">
            <h1>"Grid Trading"</h1>
            <p class="subtitle">"Peer-to-peer energy trading marketplace."</p>

            <h2>"Active Offers"</h2>
            <div class="offer-list">
                {move || ctx.offers.get().iter().map(|offer| {
                    let kwh = offer.amount_kwh;
                    let price = offer.price_per_kwh;
                    let currency = offer.currency.clone();
                    let status = offer.status.label();
                    let project = offer.project_id.clone().unwrap_or_else(|| "Independent".into());
                    view! {
                        <div class="offer-card">
                            <div class="offer-header">
                                <span class="offer-kwh">{format!("{kwh:.0} kWh")}</span>
                                <span class="offer-price">{format!("{currency} {price:.2}/kWh")}</span>
                            </div>
                            <div class="offer-meta">
                                <span>{format!("Source: {project}")}</span>
                                <span class="badge">{status}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>

            <h2>"Recent Trades"</h2>
            {move || {
                let trades = ctx.trades.get();
                if trades.is_empty() {
                    view! { <p class="no-data">"No trades yet."</p> }.into_any()
                } else {
                    trades.iter().map(|t| {
                        let kwh = t.amount_kwh;
                        let price = t.total_price;
                        let settled = if t.settled { "Settled" } else { "Pending" };
                        view! {
                            <div class="trade-card">
                                <span>{format!("{kwh:.0} kWh for R {price:.0}")}</span>
                                <span class="badge">{settled}</span>
                            </div>
                        }
                    }).collect_view().into_any()
                }
            }}
        </div>
    }
}
