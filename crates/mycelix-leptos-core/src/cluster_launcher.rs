// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-cluster navigation launcher.
//!
//! Provides a floating menu linking to other Mycelix cluster apps.
//! Each cluster app includes this component with its own cluster list.

use leptos::prelude::*;

/// A link to another Mycelix cluster app.
#[derive(Clone, Debug)]
pub struct ClusterLink {
    pub name: &'static str,
    pub icon: &'static str,
    pub href: &'static str,
}

/// Default cluster links for the Mycelix ecosystem.
pub fn default_clusters() -> Vec<ClusterLink> {
    vec![
        ClusterLink { name: "Climate", icon: "🌍", href: "http://localhost:8103" },
        ClusterLink { name: "Knowledge", icon: "🧠", href: "http://localhost:8114" },
        ClusterLink { name: "Energy", icon: "⚡", href: "http://localhost:8108" },
        ClusterLink { name: "Finance", icon: "💰", href: "http://localhost:8109" },
        ClusterLink { name: "Governance", icon: "⚖", href: "http://localhost:8110" },
        ClusterLink { name: "Health", icon: "💚", href: "http://localhost:8111" },
        ClusterLink { name: "Hearth", icon: "🏠", href: "http://localhost:8112" },
        ClusterLink { name: "Commons", icon: "🤲", href: "http://localhost:8104" },
        ClusterLink { name: "Music", icon: "🎵", href: "http://localhost:8121" },
        ClusterLink { name: "Praxis", icon: "📚", href: "http://localhost:8107" },
        ClusterLink { name: "Portal", icon: "🌀", href: "http://localhost:8124" },
    ]
}

/// Floating cluster launcher button + dropdown.
///
/// Shows a button that expands to reveal links to other clusters.
/// Excludes the current cluster (matched by name).
#[component]
pub fn ClusterLauncher(
    #[prop(into)] current: String,
    #[prop(optional, into)] clusters: Option<Vec<ClusterLink>>,
) -> impl IntoView {
    let (open, set_open) = signal(false);
    let all = clusters.unwrap_or_else(default_clusters);
    let current_name = current.clone();
    let filtered: Vec<ClusterLink> = all
        .into_iter()
        .filter(|c| c.name != current_name)
        .collect();

    view! {
        <div class="cluster-launcher">
            <button
                class="cluster-launcher-btn"
                on:click=move |_| set_open.update(|o| *o = !*o)
                aria-label="Switch cluster"
                aria-expanded=move || open.get().to_string()
            >
                "🌐"
            </button>
            <div
                class="cluster-launcher-menu"
                style=move || if open.get() { "display: flex" } else { "display: none" }
                role="menu"
            >
                {filtered.into_iter().map(|cluster| {
                    view! {
                        <a
                            class="cluster-launcher-item"
                            href=cluster.href
                            role="menuitem"
                            on:click=move |_| set_open.set(false)
                        >
                            <span class="cluster-launcher-icon">{cluster.icon}</span>
                            <span class="cluster-launcher-name">{cluster.name}</span>
                        </a>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
