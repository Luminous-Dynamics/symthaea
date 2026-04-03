// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes, A},
    path,
};

use crate::pages::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <a href="#main-content" class="skip-link">"Skip to main content"</a>
            <nav class="navbar" aria-label="Primary navigation">
                <a href="/" class="logo">"Mycelix Professional"</a>
                <div class="nav-links">
                    <A href="/jobs">"Jobs"</A>
                    <A href="/network">"Network"</A>
                    <A href="/applications">"Applications"</A>
                    <A href="/profile">"Profile"</A>
                </div>
                <span class="connection-tag">"Local Mode"</span>
            </nav>
            <main id="main-content">
                <Routes fallback=|| view! { <p>"Page not found"</p> }>
                    <Route path=path!("/") view=DashboardPage />
                    <Route path=path!("/jobs") view=JobsPage />
                    <Route path=path!("/network") view=NetworkPage />
                    <Route path=path!("/applications") view=ApplicationsPage />
                    <Route path=path!("/profile") view=ProfilePage />
                </Routes>
            </main>
        </Router>
    }
}
