// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::{Router, Routes, Route};
use leptos_router::path;

use mycelix_leptos_core::{
    HolochainProviderAuto, HolochainProviderConfig,
    provide_theme_context, provide_thermodynamic_context,
    provide_consciousness_context, provide_toast_context,
    provide_homeostasis_context, init_consciousness_ui,
    AppShell, NavLink, ToastContainer, EmptyState,
};
use mycelix_leptos_i18n::provide_i18n;

use crate::context::provide_knowledge_context;
use crate::pages;

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AppThemeVariant {
    Dark,
    Light,
}

impl mycelix_leptos_core::AppTheme for AppThemeVariant {
    fn label(&self) -> &'static str {
        match self {
            AppThemeVariant::Dark => "dark",
            AppThemeVariant::Light => "light",
        }
    }
    fn all() -> &'static [Self] {
        &[AppThemeVariant::Dark, AppThemeVariant::Light]
    }
    fn next(&self) -> Self {
        match self {
            AppThemeVariant::Dark => AppThemeVariant::Light,
            AppThemeVariant::Light => AppThemeVariant::Dark,
        }
    }
    fn is_light(&self) -> bool {
        matches!(self, AppThemeVariant::Light)
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[component]
pub fn App() -> impl IntoView {
    let config = HolochainProviderConfig {
        app_id: "mycelix-unified".into(),
        default_role: Some("knowledge".into()),
        log_prefix: "[Knowledge]",
        connect_strategy: mycelix_leptos_core::ConnectStrategy::WebSocket,
        status_labels: None,
    };

    view! {
        <HolochainProviderAuto config=config>
            <AppInner />
        </HolochainProviderAuto>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    // Provider stack (order matters)
    provide_theme_context("knowledge-theme", AppThemeVariant::Dark);
    provide_thermodynamic_context();
    provide_consciousness_context();
    provide_toast_context();
    provide_homeostasis_context(2, "--homeostasis");
    provide_i18n("knowledge-lang");
    provide_knowledge_context();
    init_consciousness_ui();

    let nav_links = vec![
        NavLink { href: "/", label: "Home", icon: None },
        // Add your cluster-specific nav links here
    ];

    view! {
        <Router>
            <AppShell
                brand_name="Knowledge"
                brand_icon="🧠"
                nav_links=nav_links
            >
                <Routes fallback=|| view! {
                    <EmptyState icon="?" title="Page not found" />
                }>
                    <Route path=path!("/") view=pages::HomePage />
                </Routes>
            </AppShell>
            <ToastContainer />
        </Router>
    }
}
