// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::types::TrustScore;

#[component]
pub fn TrustCard(trust: TrustScore) -> impl IntoView {
    let score_pct = (trust.score * 100.0) as u32;
    let conf_pct = (trust.confidence * 100.0) as u32;
    view! {
        <div class="glass-panel">
            <h4 style="font-size: 1rem; margin-bottom: 0.5rem;">{trust.participant.clone()}</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.875rem;">
                <div><span style="color: var(--text-secondary);">"Trust: "</span><span style="color: var(--accent-emerald); font-weight: 600;">{format!("{:.0}%", trust.score * 100.0)}</span></div>
                <div><span style="color: var(--text-secondary);">"Confidence: "</span><span>{format!("{:.0}%", trust.confidence * 100.0)}</span></div>
                <div><span style="color: var(--text-secondary);">"Interactions: "</span><span>{trust.interaction_count}</span></div>
                <div><span style="color: var(--text-secondary);">"Domains: "</span><span>{trust.expertise_domains.len()}</span></div>
            </div>
            <div style="margin-top: 0.5rem; display: flex; gap: 0.25rem; flex-wrap: wrap;">
                {trust.expertise_domains.iter().map(|d| view! {
                    <span style="font-size: 0.7rem; padding: 0.125rem 0.375rem; background: var(--bg-secondary); border-radius: 9999px; color: var(--text-secondary);">{d.clone()}</span>
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
