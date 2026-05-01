// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-cluster navigation launcher.
//!
//! Detects whether running on `*.mycelix.net` or localhost and
//! generates appropriate links for the environment.

use leptos::prelude::*;

/// A link to another Mycelix cluster app.
#[derive(Clone, Debug)]
pub struct ClusterLink {
    pub name: &'static str,
    pub icon: &'static str,
    pub href: String,
}

/// Default cluster links — auto-detects localhost vs public deployment.
pub fn default_clusters() -> Vec<ClusterLink> {
    if is_public_domain() {
        public_clusters()
    } else {
        localhost_clusters()
    }
}

fn is_public_domain() -> bool {
    web_sys::window()
        .and_then(|w| w.location().hostname().ok())
        .map(|h| h.contains("mycelix.net") || h.contains("luminousdynamics.io"))
        .unwrap_or(false)
}

fn public_clusters() -> Vec<ClusterLink> {
    vec![
        ClusterLink {
            name: "Climate",
            icon: "🌍",
            href: "https://climate.mycelix.net".into(),
        },
        ClusterLink {
            name: "Knowledge",
            icon: "🧠",
            href: "https://knowledge.mycelix.net".into(),
        },
        ClusterLink {
            name: "Energy",
            icon: "⚡",
            href: "https://energy.mycelix.net".into(),
        },
        ClusterLink {
            name: "Finance",
            icon: "💰",
            href: "https://finance.mycelix.net".into(),
        },
        ClusterLink {
            name: "Governance",
            icon: "⚖",
            href: "https://governance.mycelix.net".into(),
        },
        ClusterLink {
            name: "Health",
            icon: "💚",
            href: "https://health.mycelix.net".into(),
        },
        ClusterLink {
            name: "Hearth",
            icon: "🏠",
            href: "https://hearth.mycelix.net".into(),
        },
        ClusterLink {
            name: "Commons",
            icon: "🤲",
            href: "https://commons.mycelix.net".into(),
        },
        ClusterLink {
            name: "Music",
            icon: "🎵",
            href: "https://music.mycelix.net".into(),
        },
        ClusterLink {
            name: "Praxis",
            icon: "📚",
            href: "https://praxis.mycelix.net".into(),
        },
        ClusterLink {
            name: "Sensorium",
            icon: "🌀",
            href: "https://sensorium.mycelix.net".into(),
        },
        ClusterLink {
            name: "Attribution",
            icon: "🏷",
            href: "https://attribution.mycelix.net".into(),
        },
        ClusterLink {
            name: "Space",
            icon: "🛰",
            href: "https://space.mycelix.net".into(),
        },
        ClusterLink {
            name: "Supply Chain",
            icon: "📦",
            href: "https://supplychain.mycelix.net".into(),
        },
    ]
}

fn localhost_clusters() -> Vec<ClusterLink> {
    vec![
        ClusterLink {
            name: "Climate",
            icon: "🌍",
            href: "http://localhost:8103".into(),
        },
        ClusterLink {
            name: "Knowledge",
            icon: "🧠",
            href: "http://localhost:8114".into(),
        },
        ClusterLink {
            name: "Energy",
            icon: "⚡",
            href: "http://localhost:8108".into(),
        },
        ClusterLink {
            name: "Finance",
            icon: "💰",
            href: "http://localhost:8109".into(),
        },
        ClusterLink {
            name: "Governance",
            icon: "⚖",
            href: "http://localhost:8110".into(),
        },
        ClusterLink {
            name: "Health",
            icon: "💚",
            href: "http://localhost:8111".into(),
        },
        ClusterLink {
            name: "Hearth",
            icon: "🏠",
            href: "http://localhost:8112".into(),
        },
        ClusterLink {
            name: "Commons",
            icon: "🤲",
            href: "http://localhost:8104".into(),
        },
        ClusterLink {
            name: "Music",
            icon: "🎵",
            href: "http://localhost:8121".into(),
        },
        ClusterLink {
            name: "Praxis",
            icon: "📚",
            href: "http://localhost:8107".into(),
        },
        ClusterLink {
            name: "Portal",
            icon: "🌀",
            href: "http://localhost:8124".into(),
        },
        ClusterLink {
            name: "Attribution",
            icon: "🏷",
            href: "http://localhost:8101".into(),
        },
        ClusterLink {
            name: "Space",
            icon: "🛰",
            href: "http://localhost:8126".into(),
        },
        ClusterLink {
            name: "Supply Chain",
            icon: "📦",
            href: "http://localhost:8127".into(),
        },
    ]
}

/// Floating cluster launcher button + dropdown.
#[component]
pub fn ClusterLauncher(
    #[prop(into)] current: String,
    #[prop(optional, into)] clusters: Option<Vec<ClusterLink>>,
) -> impl IntoView {
    let (open, set_open) = signal(false);
    let all = clusters.unwrap_or_else(default_clusters);
    let current_name = current.clone();
    let filtered: Vec<ClusterLink> = all.into_iter().filter(|c| c.name != current_name).collect();

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
                    let href = cluster.href.clone();
                    view! {
                        <a
                            class="cluster-launcher-item"
                            href=href
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
