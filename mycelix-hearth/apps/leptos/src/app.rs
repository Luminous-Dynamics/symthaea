// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use crate::components::{Nav, DevPanel};
use crate::hearth_context::provide_hearth_context;
use crate::pages::*;
use crate::pages::personal;
use mycelix_leptos_core::{
    HolochainProviderAuto, HolochainProviderConfig, ConnectStrategy,
    provide_consciousness_context, init_consciousness_ui,
    provide_thermodynamic_context, provide_homeostasis_context,
    provide_toast_context, ToastContainer,
};

#[component]
pub fn App() -> impl IntoView {
    let config = HolochainProviderConfig {
        app_id: "mycelix-unified".into(),
        default_role: Some("hearth".into()),
        log_prefix: "[Hearth]",
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
    crate::circadian::provide_circadian_context();
    crate::ambient_sound::provide_ambient_sound_context();
    provide_thermodynamic_context();
    provide_consciousness_context();
    provide_toast_context();
    provide_homeostasis_context(2, "--homeostasis");
    provide_hearth_context();
    crate::hearth_prefs::provide_hearth_prefs();

    // Action dispatch layer (real zome calls when connected, mock when not)
    crate::hearth_actions::provide_hearth_actions();

    // Wire consciousness + circadian → CSS custom properties
    init_consciousness_ui();

    // Real-time signals from conductor (when connected)
    crate::signal_listener::start_signal_listener();

    // Simulated family life (only activates in mock mode)
    crate::simulated_life::start_simulated_life();

    view! {
        <Router>
            <a href="#main-content" class="skip-to-content">"skip to content"</a>
            <Nav />
            <DevPanel />
            // Ambient campfire — persistent warmth from the founding ceremony
            <crate::components::HearthFlame mode=crate::components::FlameMode::Ambient />
            <main id="main-content" class="main-content heartbeat" role="main" aria-label="hearth content">
                <Routes fallback=|| view! { <p class="not-found">"you’ve wandered off the path"</p> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/found") view=crate::pages::found::FoundingCeremony />
                    <Route path=path!("/settings") view=crate::pages::settings::SettingsPage />
                    <Route path=path!("/kinship") view=KinshipPage />
                    <Route path=path!("/care") view=CarePage />
                    <Route path=path!("/decisions") view=DecisionsPage />
                    <Route path=path!("/gratitude") view=GratitudePage />
                    <Route path=path!("/stories") view=StoriesPage />
                    <Route path=path!("/milestones") view=MilestonesPage />
                    <Route path=path!("/rhythms") view=RhythmsPage />
                    <Route path=path!("/emergency") view=EmergencyPage />
                    <Route path=path!("/resources") view=ResourcesPage />
                    <Route path=path!("/autonomy") view=AutonomyPage />
                    <Route path=path!("/personal/profile") view=personal::ProfilePage />
                    <Route path=path!("/personal/health") view=personal::HealthPage />
                    <Route path=path!("/personal/credentials") view=personal::CredentialsPage />
                    <Route path=path!("/personal/disclosure") view=personal::DisclosurePage />
                </Routes>
            </main>
            <ToastContainer />
            <crate::onboarding::OnboardingOverlay />
        </Router>
    }
}
