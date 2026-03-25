// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

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
            <nav class="navbar">
                <a href="/" class="logo">"EduNet"</a>
                <div class="nav-links">
                    <A href="/courses">"Courses"</A>
                    <A href="/review">"Review"</A>
                    <A href="/dashboard">"Dashboard"</A>
                    <A href="/governance">"Governance"</A>
                    <A href="/credentials">"Credentials"</A>
                </div>
            </nav>
            <main>
                <Routes fallback=|| view! { <p>"Page not found"</p> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/courses") view=CoursesPage />
                    <Route path=path!("/review") view=ReviewPage />
                    <Route path=path!("/dashboard") view=DashboardPage />
                    <Route path=path!("/governance") view=GovernancePage />
                    <Route path=path!("/credentials") view=CredentialsPage />
                </Routes>
            </main>
        </Router>
    }
}
