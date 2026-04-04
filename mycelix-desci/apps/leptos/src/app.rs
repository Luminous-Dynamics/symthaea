// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use crate::components::nav::NavBar;
use crate::pages::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <NavBar />
            <Routes fallback=|| view! { <div class="page-container"><h1>"404 — Page Not Found"</h1></div> }>
                <Route path=path!("/") view=HomePage />
                <Route path=path!("/browse") view=BrowsePage />
                <Route path=path!("/submit") view=SubmitPage />
                <Route path=path!("/discovery") view=DiscoveryPage />
                <Route path=path!("/about") view=AboutPage />
            </Routes>
        </Router>
    }
}
