// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! App shell — nav + main layout wrapper for all Mycelix cluster frontends.
//!
//! Provides a consistent navigation skeleton with brand, links,
//! tier badge, and connection status across all apps.

use leptos::prelude::*;
use leptos_router::components::A;
use crate::holochain_provider::ConnectionBadge;
use crate::consciousness::use_consciousness;

/// A navigation link definition.
#[derive(Clone, Debug)]
pub struct NavLink {
    pub href: &'static str,
    pub label: &'static str,
    pub icon: Option<&'static str>,
}

/// A mobile bottom nav tab definition.
#[derive(Clone, Debug)]
pub struct NavTab {
    pub href: &'static str,
    pub icon: &'static str,
    pub label: &'static str,
}

/// App shell: wraps nav + main content area.
///
/// Place at the top level inside your Router:
/// ```ignore
/// <AppShell brand_name="Civic" brand_icon="⚖" nav_links=links>
///     <Routes .../>
/// </AppShell>
/// ```
#[component]
pub fn AppShell(
    #[prop(into)] brand_name: String,
    #[prop(into)] brand_icon: String,
    #[prop(into)] nav_links: Vec<NavLink>,
    #[prop(optional, into)] mobile_tabs: Option<Vec<NavTab>>,
    children: Children,
) -> impl IntoView {
    view! {
        <a class="skip-link" href="#main-content">"Skip to content"</a>
        <AppNav brand_name=brand_name.clone() brand_icon=brand_icon.clone() links=nav_links />
        <main id="main-content" class="app-main">
            {children()}
        </main>
        {mobile_tabs.map(|tabs| view! { <MobileBottomNav tabs /> })}
    }
}

/// Navigation bar with brand, links, tier badge, and connection status.
#[component]
pub fn AppNav(
    #[prop(into)] brand_name: String,
    #[prop(into)] brand_icon: String,
    #[prop(into)] links: Vec<NavLink>,
) -> impl IntoView {
    let consciousness = use_consciousness();

    view! {
        <nav class="app-nav" role="navigation" aria-label="main navigation">
            <div class="nav-brand">
                <A href="/" attr:class="nav-logo" attr:aria-label="home">
                    <span class="logo-glyph">{brand_icon}</span>
                    <span class="logo-text">{brand_name}</span>
                </A>
            </div>

            <div class="nav-links">
                {links.into_iter().map(|link| {
                    let icon = link.icon.unwrap_or("");
                    view! {
                        <A href=link.href attr:class="nav-link">
                            {if !icon.is_empty() {
                                Some(view! { <span class="nav-icon">{icon}</span> })
                            } else {
                                None
                            }}
                            {link.label}
                        </A>
                    }
                }).collect_view()}
            </div>

            <div class="nav-meta">
                <A href="/profile" attr:class="nav-profile-link">
                    <span class=move || {
                        let tier = consciousness.tier.get();
                        format!("tier-badge tier-{}", tier.css_class())
                    }>
                        {move || consciousness.tier.get().label()}
                    </span>
                </A>
                <ConnectionBadge />
            </div>
        </nav>
    }
}

/// Mobile bottom navigation bar.
#[component]
pub fn MobileBottomNav(
    #[prop(into)] tabs: Vec<NavTab>,
) -> impl IntoView {
    view! {
        <nav class="mobile-bottom-nav" role="navigation" aria-label="mobile navigation">
            {tabs.into_iter().map(|tab| {
                view! {
                    <A href=tab.href attr:class="bottom-tab">
                        <span class="bottom-tab-icon">{tab.icon}</span>
                        <span class="bottom-tab-label">{tab.label}</span>
                    </A>
                }
            }).collect_view()}
        </nav>
    }
}
