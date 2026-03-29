// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sidebar navigation — domain-aware, consciousness-gated.

use leptos::prelude::*;
use leptos_router::hooks::use_location;
use portal_domain_trait::NavItem;
use crate::identity::VaultState;

#[component]
pub fn Sidebar(
    nav_items: Vec<NavItem>,
    tier_label: Memo<String>,
    vault_state: RwSignal<VaultState>,
) -> impl IntoView {
    let location = use_location();

    let vault_status = move || match vault_state.get() {
        VaultState::NoVault => "No Identity",
        VaultState::Locked => "Locked",
        VaultState::Unlocked => "Active",
    };

    view! {
        <nav class="sidebar" role="navigation" aria-label="Domain navigation">
            <div class="sidebar-header">
                <span class="sidebar-title">"Mycelix"</span>
                <span class="sidebar-tier">{tier_label}</span>
            </div>

            <div class="sidebar-vault">
                <span class="vault-dot" />
                <span class="vault-label">{vault_status}</span>
            </div>

            <div class="sidebar-domains">
                // Portal home
                {
                    let is_home = move || location.pathname.get() == "/";
                    view! {
                        <a
                            href="/"
                            class=move || if is_home() { "sidebar-item active" } else { "sidebar-item" }
                        >
                            <span class="sidebar-bio">"Portal"</span>
                            <span class="sidebar-label">"Home"</span>
                        </a>
                    }
                }

                // Domain nav items
                {nav_items.into_iter().map(|item| {
                    let path = item.path.to_string();
                    let path2 = path.clone();
                    let is_active = move || {
                        let current = location.pathname.get();
                        current == path || current.starts_with(&format!("{}/", path))
                    };
                    view! {
                        <a
                            href=item.path
                            class=move || if is_active() { "sidebar-item active" } else { "sidebar-item" }
                        >
                            <span class="sidebar-bio">{item.bio_label}</span>
                            <span class="sidebar-label">{item.label}</span>
                        </a>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </nav>
    }
}
