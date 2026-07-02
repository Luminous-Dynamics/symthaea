// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain-aware AppShell wrapper for Mycelix clusters.

use crate::consciousness::use_consciousness;
use crate::holochain_provider::ConnectionBadge;
use leptos::prelude::*;
use leptos_router::components::A;
use mycelix_leptos_ui::app_shell as ui;

// Re-export types from UI crate
pub use mycelix_leptos_ui::{NavLink, NavTab};

/// App shell: wraps nav + main content area.
///
/// Injects Holochain connection status and Consciousness tier badge
/// into the UI-layer AppShell.
#[component]
pub fn AppShell(
    #[prop(into)] brand_name: String,
    #[prop(into)] brand_icon: String,
    #[prop(into)] nav_links: Vec<NavLink>,
    #[prop(optional, into)] mobile_tabs: Option<Vec<NavTab>>,
    children: Children,
) -> impl IntoView {
    let consciousness = use_consciousness();

    // The "meta_view" slot contains the Holochain-specific elements
    let meta_view = move || {
        view! {
            <A href="/profile" attr:class="nav-profile-link">
                <span class=move || {
                    let tier = consciousness.tier.get();
                    format!("tier-badge tier-{}", tier.css_class())
                }>
                    {move || consciousness.tier.get().label()}
                </span>
            </A>
            <ConnectionBadge />
        }
        .into_any()
    };

    view! {
        <ui::AppShell
            brand_name=brand_name
            brand_icon=brand_icon
            nav_links=nav_links
            mobile_tabs=mobile_tabs
            meta_view=Some(meta_view().into_any())
        >
            {children()}
        </ui::AppShell>
    }
}

// Internal components are re-exported if needed, but AppShell is the primary entry point.
pub use ui::{AppNav, MobileBottomNav};
