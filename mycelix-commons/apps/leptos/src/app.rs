// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use mycelix_leptos_client::MockTransport;
use mycelix_leptos_core::{ConnectionStatusIndicator, HolochainProvider};

use crate::components::Nav;
use crate::pages::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <HolochainProvider transport=MockTransport::new()>
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
            </Router>
        </HolochainProvider>
    }
}
