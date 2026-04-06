// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search result card with epistemic overlay and claim actions.

use leptos::prelude::*;
use prism_common::{EmpiricalLevel, SearchResult};
use crate::components::claim_actions::ClaimActions;

#[component]
pub fn ResultCard(result: SearchResult, rank: usize) -> impl IntoView {
    let (e_class, e_label) = match result.empirical_level {
        EmpiricalLevel::E4 => ("e-badge e4", "E4"),
        EmpiricalLevel::E3 => ("e-badge e3", "E3"),
        EmpiricalLevel::E2 => ("e-badge e2", "E2"),
        EmpiricalLevel::E1 => ("e-badge e1", "E1"),
        EmpiricalLevel::E0 => ("e-badge e0", "E0"),
    };

    let score_pct = (result.rank_score() * 100.0) as u32;
    let source = result.sources.first().cloned().unwrap_or_else(|| "—".to_string());
    let tags = result.tags.join(", ");
    let result_for_actions = result.clone();

    view! {
        <div class="result-card">
            <div class="result-header">
                <span style="color:var(--content-text-secondary); font-size:14px; font-weight:600;">{rank}"."</span>
                <span class=e_class>{e_label}</span>
                <span class="result-score">{score_pct}"% match"</span>
            </div>
            <div class="result-content">{result.content.clone()}</div>
            <div class="result-footer">
                <div class="result-meta">
                    <span class="result-source">{source}</span>
                    <span class="result-tags">{tags}</span>
                </div>
                <ClaimActions result=result_for_actions />
            </div>
        </div>
    }
}
