// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Identity app shell — provider stack + routing.
//!
//! Provider ordering is critical: IdentityContext sits ABOVE ConsciousnessContext
//! because consciousness gating derives from identity data (reputation, MFA assurance).

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use mycelix_leptos_core::{
    HolochainProviderAuto, HolochainProviderConfig, ConnectStrategy,
    provide_toast_context, ToastContainer,
};
use crate::identity_context::provide_identity_context;
use crate::pages::*;
use crate::components::Nav;

#[component]
pub fn App() -> impl IntoView {
    let config = HolochainProviderConfig {
        app_id: "mycelix-unified".into(),
        default_role: Some("identity".into()),
        log_prefix: "[Identity]",
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
    // Provider stack — order matters (dependencies first):
    // 1. Identity context (loads DID, MFA, credentials from conductor)
    provide_identity_context();
    // 2. Toast notifications
    provide_toast_context();
    // NOTE: ConsciousnessContext would go here, deriving from identity data.
    // For MVP, consciousness coupling is done via CSS variables directly.

    view! {
        <Router>
            <a href="#main-content" class="skip-link">"Skip to main content"</a>
            <Nav />
            <main id="main-content">
                <Routes fallback=|| view! { <div class="page"><h1>"404 — Page not found"</h1></div> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/did") view=DidPage />
                    <Route path=path!("/mfa") view=MfaPage />
                    <Route path=path!("/credentials") view=CredentialsPage />
                    <Route path=path!("/recovery") view=RecoveryPage />
                    <Route path=path!("/trust") view=TrustPage />
                </Routes>
            </main>
            <ToastContainer />
        </Router>
    }
}
