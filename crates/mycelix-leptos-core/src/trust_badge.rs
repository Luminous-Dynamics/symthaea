// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trust tier badge component.
//!
//! Displays a user's consciousness-gated trust tier as a colored badge.
//! The five tiers map to the Mycelix bridge-common consciousness gating
//! system: Observer -> Participant -> Citizen -> Steward -> Guardian.

use leptos::prelude::*;

/// Badge color and label configuration for each trust tier.
struct TierStyle {
    background: &'static str,
    color: &'static str,
    border: &'static str,
    label: &'static str,
}

fn tier_style(tier: &str) -> TierStyle {
    match tier.to_lowercase().as_str() {
        "observer" => TierStyle {
            background: "#f3f4f6", // gray-100
            color: "#4b5563",      // gray-600
            border: "#d1d5db",     // gray-300
            label: "Observer",
        },
        "participant" => TierStyle {
            background: "#dbeafe", // blue-100
            color: "#2563eb",      // blue-600
            border: "#93c5fd",     // blue-300
            label: "Participant",
        },
        "citizen" => TierStyle {
            background: "#dcfce7", // green-100
            color: "#16a34a",      // green-600
            border: "#86efac",     // green-300
            label: "Citizen",
        },
        "steward" => TierStyle {
            background: "#f3e8ff", // purple-100
            color: "#9333ea",      // purple-600
            border: "#c084fc",     // purple-300
            label: "Steward",
        },
        "guardian" => TierStyle {
            background: "#fef3c7", // amber-100
            color: "#b45309",      // amber-700
            border: "#fbbf24",     // amber-400
            label: "Guardian",
        },
        _ => TierStyle {
            background: "#f3f4f6",
            color: "#4b5563",
            border: "#d1d5db",
            label: "Unknown",
        },
    }
}

/// Displays a trust tier as a styled badge.
///
/// The `tier` string is matched case-insensitively against the five
/// Mycelix trust tiers. Unknown values render a gray "Unknown" badge.
///
/// # Props
///
/// * `tier` — Trust tier name (e.g. "Observer", "guardian", "Citizen").
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::TrustBadge;
/// // view! { <TrustBadge tier="Guardian".to_string() /> }
/// ```
#[component]
pub fn TrustBadge(tier: String) -> impl IntoView {
    let style = tier_style(&tier);
    let label = style.label.to_string();

    view! {
        <span
            style=format!(
                "display: inline-flex; align-items: center; padding: 2px 10px; \
                 border-radius: 9999px; font-size: 0.75rem; font-weight: 600; \
                 font-family: system-ui, sans-serif; line-height: 1.25rem; \
                 background-color: {}; color: {}; border: 1px solid {};",
                style.background, style.color, style.border,
            )
        >
            {label}
        </span>
    }
}
