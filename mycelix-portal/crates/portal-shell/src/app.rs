// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! App shell — domain registry, identity provider, consciousness-gated routing.

use leptos::prelude::*;
use leptos_router::components::*;
use leptos_router::path;
use portal_domain_trait::{DomainRegistry, DomainModule, ConsciousnessTier};

use crate::identity::{PortalIdentity, VaultState};
use crate::nav::Sidebar;

/// Build the domain registry from feature-gated crates.
fn build_registry() -> DomainRegistry {
    let mut registry = DomainRegistry::new();

    #[cfg(feature = "health")]
    registry.register(Box::new(domain_health::HealthDomain));

    // Future domains:
    // #[cfg(feature = "governance")]
    // registry.register(Box::new(domain_governance::GovernanceDomain));
    // #[cfg(feature = "edunet")]
    // registry.register(Box::new(domain_edunet::EduNetDomain));
    // #[cfg(feature = "finance")]
    // registry.register(Box::new(domain_finance::FinanceDomain));

    registry
}

#[component]
pub fn App() -> impl IntoView {
    let identity = PortalIdentity::new();
    provide_context(identity.clone());

    let registry = build_registry();
    let domain_count = registry.domains.len();

    // Collect nav items from all visible domains
    let identity_clone = identity.clone();
    let nav_items: Vec<_> = registry.domains.iter()
        .filter(|d| d.min_tier() <= ConsciousnessTier::from_score(identity_clone.consciousness_score.get_untracked()))
        .flat_map(|d| d.nav_items())
        .collect();

    let tier_label = Memo::new(move |_| {
        identity.tier.get().label().to_string()
    });

    view! {
        <div class="portal-root">
            <div class="portal-content">
                <Router>
                    <Sidebar
                        nav_items=nav_items
                        tier_label=tier_label
                        vault_state=identity.vault
                    />
                    <main class="portal-main">
                        <Routes fallback=|| view! {
                            <div class="page">
                                <p class="not-found">"This pathway does not exist."</p>
                            </div>
                        }>
                            <Route path=path!("/") view=move || {
                                view! { <PortalHome vault=identity.vault tier_label=tier_label domain_count=domain_count /> }
                            } />
                            <Route path=path!("/welcome") view=|| view! {
                                <div class="page">
                                    <h1 class="bio-title">"First Breath"</h1>
                                    <p>"Onboarding flow — generates your Mycelix identity."</p>
                                    <p>"Your master key derives domain-specific keys for Health, Education, Governance, and every other Mycelix cluster."</p>
                                </div>
                            } />
                        </Routes>
                    </main>
                </Router>
            </div>
        </div>
    }
}

/// Portal home — shows identity status and available domains.
#[component]
fn PortalHome(
    vault: RwSignal<VaultState>,
    tier_label: Memo<String>,
    domain_count: usize,
) -> impl IntoView {
    let has_vault = move || vault.get() != VaultState::NoVault;

    view! {
        <div class="page portal-home">
            <Show when=move || !has_vault()>
                <div class="first-breath-container">
                    <a href="/welcome" class="first-breath-seed">
                        <div class="seed-pulse" />
                        <div class="seed-core" />
                    </a>
                    <h2 class="first-breath-title">"Take Your First Breath"</h2>
                    <p class="first-breath-text">
                        "Generate your Mycelix identity. One key unlocks Health, "
                        "Education, Governance, and every domain in the ecosystem."
                    </p>
                </div>
            </Show>

            <Show when=has_vault>
                <header class="page-header">
                    <h1 class="bio-title">"Mycelix"</h1>
                    <p class="bio-subtitle">
                        "Consciousness tier: " {tier_label}
                        " · " {domain_count.to_string()} " domains available"
                    </p>
                </header>

                <div class="domain-grid">
                    <a href="/health" class="domain-card" data-domain="health">
                        <span class="domain-name">"Health"</span>
                        <span class="domain-bio">"Biological Sovereignty"</span>
                    </a>
                    <div class="domain-card locked">
                        <span class="domain-name">"Governance"</span>
                        <span class="domain-bio">"Coming soon"</span>
                    </div>
                    <div class="domain-card locked">
                        <span class="domain-name">"Education"</span>
                        <span class="domain-bio">"Coming soon"</span>
                    </div>
                    <div class="domain-card locked">
                        <span class="domain-name">"Finance"</span>
                        <span class="domain-bio">"Coming soon"</span>
                    </div>
                </div>
            </Show>
        </div>
    }
}
