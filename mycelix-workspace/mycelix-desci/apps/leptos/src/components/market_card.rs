// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::types::PredictionMarket;

#[component]
pub fn MarketCard(market: PredictionMarket) -> impl IntoView {
    let prob_pct = (market.current_probability * 100.0) as u32;
    let status_color = match market.status.as_str() {
        "Open" => "var(--accent-emerald)",
        "Resolved" => "var(--accent-indigo)",
        _ => "var(--text-secondary)",
    };
    view! {
        <div class="glass-panel" style="cursor: pointer;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style=format!("font-size: 0.75rem; color: {};", status_color)>{market.status.clone()}</span>
                <span style="font-size: 1.5rem; font-weight: 700; color: var(--accent-indigo);">{format!("{}%", prob_pct)}</span>
            </div>
            <p style="font-size: 0.875rem; line-height: 1.4; margin-bottom: 0.5rem;">{market.question.clone()}</p>
            <div style="font-size: 0.75rem; color: var(--text-secondary); display: flex; gap: 1rem;">
                <span>{format!("{} positions", market.positions)}</span>
                <span>{format!("{:.0} staked", market.total_staked)}</span>
            </div>
        </div>
    }
}
