// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness tier gate component.

use leptos::prelude::*;
use mycelix_leptos_core::use_consciousness;
use personal_leptos_types::TrustTier;

/// Gate content behind a minimum consciousness tier.
///
/// Renders both children and locked message; toggles visibility reactively.
#[component]
pub fn TierGate(
    min_tier: TrustTier,
    #[prop(default = "access this")]
    action_label: &'static str,
    children: Children,
) -> impl IntoView {
    let consciousness = use_consciousness();

    let needed = min_tier.label();
    let locked_msg = format!("Reach {needed} tier to {action_label}");

    let child_view = children();

    let show_style = move || {
        if consciousness.tier.get() >= min_tier {
            "display: block"
        } else {
            "display: none"
        }
    };

    let hide_style = move || {
        if consciousness.tier.get() >= min_tier {
            "display: none"
        } else {
            "display: flex"
        }
    };

    view! {
        <div class="tier-gate-content" style=show_style>
            {child_view}
        </div>
        <div class="tier-locked" style=hide_style>
            <span class="tier-locked-icon">"\u{1f512}"</span>
            <span class="tier-locked-msg">{locked_msg}</span>
        </div>
    }
}
