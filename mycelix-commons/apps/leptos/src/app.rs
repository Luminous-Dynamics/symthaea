// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use crate::components::Nav;
use crate::contexts::commons_context::provide_commons_context;
use crate::contexts::commons_actions::provide_commons_actions;
use crate::pages::*;

use mycelix_leptos_core::{
    HolochainProviderAuto, HolochainProviderConfig, ConnectStrategy,
    provide_consciousness_context, provide_thermodynamic_context,
    provide_homeostasis_context, init_consciousness_ui,
    provide_toast_context, ToastContainer,
    ConnectionStatusIndicator,
};

#[component]
pub fn App() -> impl IntoView {
    let config = HolochainProviderConfig {
        app_id: "mycelix-unified".into(),
        default_role: Some("commons".into()),
        log_prefix: "[Commons]",
        connect_strategy: ConnectStrategy::WebSocket,
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
    // Initialize providers in dependency order
    crate::themes::provide_theme_context();
    provide_thermodynamic_context();
    provide_consciousness_context();
    provide_toast_context();
    provide_homeostasis_context(2, "--homeostasis");
    provide_commons_context();
    provide_commons_actions();

    // Wire consciousness + thermodynamic state to CSS custom properties
    init_consciousness_ui();

    view! {
        <Router>
            <Nav />
            <main class="main-content">
                <Routes fallback=|| view! { <div class="page"><h1>"404 — Page not found"</h1></div> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/property") view=PropertyPage />
                    <Route path=path!("/housing") view=HousingPage />
                    <Route path=path!("/care") view=CarePage />
                    <Route path=path!("/resources") view=ResourcesPage />
                    <Route path=path!("/transport") view=TransportPage />
                </Routes>
            </main>
            <footer class="footer">
                <ConnectionStatusIndicator />
                <span class="footer-text">"Mycelix Commons — Community resources on Holochain"</span>
            </footer>
            <ToastContainer />
        </Router>
    }
}
