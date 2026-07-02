// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Loading skeleton placeholder component.

use leptos::prelude::*;

/// A pulsing skeleton placeholder for loading states.
#[component]
pub fn LoadingSkeleton(
    #[prop(default = "100%".to_string())] width: String,
    #[prop(default = "1rem".to_string())] height: String,
    #[prop(default = "4px".to_string())] border_radius: String,
) -> impl IntoView {
    view! {
        <div
            class="mycelix-skeleton"
            style=format!(
                "width: {width}; height: {height}; border-radius: {border_radius};",
            )
            aria-hidden="true"
        >
            <style>
                ".mycelix-skeleton { \
                    background: linear-gradient(90deg, \
                        var(--bg-raised, var(--bg-surface, #1a1a2e)) 25%, \
                        var(--surface, var(--border, rgba(148, 163, 184, 0.12))) 50%, \
                        var(--bg-raised, var(--bg-surface, #1a1a2e)) 75%); \
                    background-size: 200% 100%; \
                    animation: mycelix-skeleton-pulse 1.5s ease-in-out infinite; \
                } \
                @keyframes mycelix-skeleton-pulse { \
                    0% { background-position: 200% 0; } \
                    100% { background-position: -200% 0; } \
                }"
            </style>
        </div>
    }
}
