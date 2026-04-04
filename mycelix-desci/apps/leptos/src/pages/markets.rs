// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prediction markets page.

use leptos::prelude::*;
use crate::types::*;
use crate::components::market_card::MarketCard;

fn demo_markets() -> Vec<PredictionMarket> {
    vec![
        PredictionMarket {
            claim_id: "island-stability-z120".into(),
            question: "Will Element 120 (A=294) be synthesized and show enhanced stability (t½ > 1s) by 2030?".into(),
            current_probability: 0.23,
            total_staked: 4500.0,
            positions: 18,
            status: "Open".into(),
            resolution_method: "Oracle".into(),
        },
        PredictionMarket {
            claim_id: "lk99-successor".into(),
            question: "Will a room-temperature, ambient-pressure superconductor be confirmed by 2028?".into(),
            current_probability: 0.08,
            total_staked: 12000.0,
            positions: 67,
            status: "Open".into(),
            resolution_method: "Consensus".into(),
        },
        PredictionMarket {
            claim_id: "fusion-q10".into(),
            question: "Will any facility achieve Q>10 fusion energy gain by 2030?".into(),
            current_probability: 0.41,
            total_staked: 8700.0,
            positions: 42,
            status: "Open".into(),
            resolution_method: "Oracle".into(),
        },
        PredictionMarket {
            claim_id: "quantum-advantage".into(),
            question: "Will quantum advantage for a practical problem be unambiguously demonstrated by 2027?".into(),
            current_probability: 0.55,
            total_staked: 6200.0,
            positions: 31,
            status: "Open".into(),
            resolution_method: "Consensus".into(),
        },
    ]
}

#[component]
pub fn MarketsPage() -> impl IntoView {
    let markets = demo_markets();
    let total_staked: f64 = markets.iter().map(|m| m.total_staked).sum();
    let total_positions: usize = markets.iter().map(|m| m.positions).sum();

    view! {
        <div class="page-container">
            <h1 class="page-title">"Prediction Markets"</h1>

            <div class="stat-grid" style="margin-bottom: 1.5rem;">
                <div class="glass-panel stat-card">
                    <div class="stat-value">{markets.len()}</div>
                    <div class="stat-label">"Active Markets"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{format!("{:.0}", total_staked)}</div>
                    <div class="stat-label">"Total Reputation Staked"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{total_positions}</div>
                    <div class="stat-label">"Total Positions"</div>
                </div>
            </div>

            <div class="claim-grid">
                {markets.into_iter().map(|m| view! { <MarketCard market=m /> }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
