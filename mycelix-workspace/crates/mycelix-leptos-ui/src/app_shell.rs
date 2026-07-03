// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! App shell — nav + main layout wrapper for Mycelix frontends.

use leptos::prelude::*;
use leptos_router::components::A;

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

/// Pure App shell: wraps nav + main content area.
#[component]
pub fn AppShell(
    #[prop(into)] brand_name: String,
    #[prop(into)] brand_icon: String,
    #[prop(into)] nav_links: Vec<NavLink>,
    #[prop(optional, into)] mobile_tabs: Option<Vec<NavTab>>,
    #[prop(optional, into)] meta_view: Option<AnyView>,
    children: Children,
) -> impl IntoView {
    view! {
        <a class="skip-link" href="#main-content">"Skip to content"</a>
        <AppNav
            brand_name=brand_name.clone()
            brand_icon=brand_icon.clone()
            links=nav_links
            meta_view=meta_view
        />
        <main id="main-content" class="app-main">
            {children()}
        </main>
        {mobile_tabs.map(|tabs| view! { <MobileBottomNav tabs /> })}
    }
}

/// Navigation bar with brand, links, and optional metadata slot.
#[component]
pub fn AppNav(
    #[prop(into)] brand_name: String,
    #[prop(into)] brand_icon: String,
    #[prop(into)] links: Vec<NavLink>,
    // No `#[prop(...)]` attribute at all: Leptos's `optional` prop modifier
    // means "the caller may omit this and get `None`, or pass the *inner*
    // type and get it auto-wrapped in `Some`" — it does NOT mean "accepts an
    // already-built `Option<AnyView>` as-is". `AppShell` (the only caller)
    // forwards its own `meta_view: Option<AnyView>` prop straight through,
    // so this needs to be a plain required prop of type `Option<AnyView>`
    // (caller always passes `Some(view)` or `None` explicitly) — not
    // `optional` (which would need a bare `AnyView`) and not `optional,
    // into` (which would need `Option<AnyView>: Into<AnyView>`, which
    // doesn't exist).
    meta_view: Option<AnyView>,
) -> impl IntoView {
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
                {meta_view}
            </div>
        </nav>
    }
}

/// Mobile bottom navigation bar.
#[component]
pub fn MobileBottomNav(#[prop(into)] tabs: Vec<NavTab>) -> impl IntoView {
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
