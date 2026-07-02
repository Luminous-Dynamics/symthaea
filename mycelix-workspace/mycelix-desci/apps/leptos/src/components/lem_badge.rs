// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::types::EpistemicTier;

/// Color-coded badge for epistemic tier.
#[component]
pub fn LemBadge(tier: EpistemicTier) -> impl IntoView {
    let class = format!("lem-badge {}", tier.css_class());
    view! {
        <span class=class>{tier.label()}</span>
    }
}

/// Verification count indicator.
#[component]
pub fn VerificationCount(count: usize) -> impl IntoView {
    let label = if count == 1 { "verification" } else { "verifications" };
    view! {
        <span class="verification-count">{count} " " {label}</span>
    }
}
