// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes, A},
    path,
};

use mycelix_leptos_core::{
    holochain_provider::HolochainProviderAuto,
    connection_status::ConnectionStatusIndicator,
    theme::provide_theme_context,
    consciousness::provide_consciousness_context,
    toasts::{provide_toast_context, ToastContainer},
    empty_state::EmptyState,
};

use crate::pages::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <HolochainProviderAuto
            app_id="mycelix-craft"
            role_name="craft"
            ws_url="ws://localhost:8888"
        >
            <AppInner />
        </HolochainProviderAuto>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    provide_theme_context("craft-theme", mycelix_leptos_core::theme::AppThemeVariant::Dark);
    provide_consciousness_context();
    provide_toast_context();

    view! {
        <Router>
            <a href="#main-content" class="skip-link">"Skip to main content"</a>
            <nav class="navbar" aria-label="Primary navigation">
                <a href="/" class="logo">"Mycelix Craft"</a>
                <div class="nav-links">
                    <A href="/employer">"Employer"</A>
                    <A href="/jobs">"Jobs"</A>
                    <A href="/network">"Network"</A>
                    <A href="/applications">"Applications"</A>
                    <A href="/credentials">"Credentials"</A>
                    <A href="/profile">"Profile"</A>
                </div>
                <ConnectionStatusIndicator />
            </nav>
            <main id="main-content">
                <Routes fallback=|| view! { <EmptyState icon="?" title="Page not found" /> }>
                    <Route path=path!("/") view=DashboardPage />
                    <Route path=path!("/employer") view=EmployerDashboard />
                    <Route path=path!("/jobs") view=JobsPage />
                    <Route path=path!("/network") view=NetworkPage />
                    <Route path=path!("/applications") view=ApplicationsPage />
                    <Route path=path!("/credentials") view=CredentialsPage />
                    <Route path=path!("/profile") view=ProfilePage />
                </Routes>
            </main>
            <ToastContainer />
        </Router>
    }
}
