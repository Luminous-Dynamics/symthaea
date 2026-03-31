// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use crate::components::Nav;
use crate::consciousness_provider::provide_consciousness_context;
use crate::contexts::governance_context::provide_governance_context;
use crate::contexts::finance_context::provide_finance_context;
use crate::holochain::HolochainProvider;
use crate::homeostasis::provide_homeostasis_context;
use crate::thermodynamic::provide_thermodynamic_context;
use crate::toasts::{provide_toast_context, ToastContainer};
use crate::pages::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <HolochainProvider>
            <AppInner />
        </HolochainProvider>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    // Initialize providers in dependency order
    crate::themes::provide_theme_context();
    provide_thermodynamic_context();
    provide_consciousness_context();
    provide_toast_context();
    provide_homeostasis_context();
    provide_governance_context();
    provide_finance_context();

    // Action dispatch layer
    crate::contexts::civic_actions::provide_civic_actions();

    // Wire consciousness + thermodynamic → CSS custom properties
    crate::consciousness_ui::init_consciousness_ui();

    view! {
        <Router>
            <a href="#main-content" class="skip-to-content">"skip to content"</a>
            <Nav />
            <main id="main-content" class="main-content civic-pulse" role="main" aria-label="civic content">
                <Routes fallback=|| view! { <p class="not-found">"you have wandered beyond the commons"</p> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/proposals") view=ProposalListPage />
                    <Route path=path!("/proposals/new") view=CreateProposalPage />
                    <Route path=path!("/proposals/:id") view=ProposalDetailPage />
                    <Route path=path!("/voting") view=VotingPage />
                    <Route path=path!("/councils") view=CouncilsPage />
                    <Route path=path!("/tend") view=TendPage />
                    <Route path=path!("/payments") view=PaymentsPage />
                    <Route path=path!("/recognition") view=RecognitionPage />
                    <Route path=path!("/treasury") view=TreasuryPage />
                    <Route path=path!("/budgeting") view=BudgetingPage />
                    <Route path=path!("/constitution") view=ConstitutionPage />
                    <Route path=path!("/execution") view=ExecutionPage />
                    <Route path=path!("/staking") view=StakingPage />
                    <Route path=path!("/oracle") view=OraclePage />
                    <Route path=path!("/profile") view=ProfilePage />
                </Routes>
            </main>
            <ToastContainer />
        </Router>
    }
}
