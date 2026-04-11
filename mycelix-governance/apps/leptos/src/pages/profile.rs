// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civic profile: unified identity across governance + finance.
//! Shows consciousness tier, all balances, council memberships, delegations.
//! AI-interactable: data-agent-did, data-tier, all balances as data-* attributes.

use leptos::prelude::*;
use mycelix_leptos_core::{use_consciousness, SovereignRadar, SovereignRadarSize};
use crate::contexts::governance_context::use_governance;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let gov = use_governance();
    let fin = use_finance();

    view! {
        <div class="profile-page" data-page="profile" role="main">
            <h1 class="page-title">"Your Civic Identity"</h1>

            // Agent identity + consciousness
            <section class="profile-identity" aria-label="identity" data-section="identity">
                {move || {
                    let did = gov.my_agent_did.get();
                    let profile = consciousness.profile.get();
                    let tier = consciousness.tier.get();
                    let short_did = did.split(':').last().unwrap_or(&did).to_string();
                    view! {
                        <div
                            class="identity-card"
                            data-agent-did=did.clone()
                            data-tier=tier.label().to_string()
                            data-consciousness=format!("{:.3}", profile.combined_score())
                        >
                            <div class="identity-header">
                                <span class="identity-did" data-field="did">{short_did}</span>
                                <span class=format!("tier-badge tier-{}", tier.css_class())>
                                    {tier.label()}
                                </span>
                            </div>
                            <SovereignRadar size=SovereignRadarSize::Medium />
                        </div>
                    }
                }}
            </section>

            // Financial summary
            <section class="profile-finance" aria-label="financial summary" data-section="finance-summary">
                <h2 class="section-title">"Balances"</h2>
                <div class="balance-grid">
                    // TEND
                    {move || {
                        let tend = fin.tend_balance.get();
                        view! {
                            <div
                                class=format!("balance-card balance-tend {}", tend.balance_class())
                                data-currency="TEND"
                                data-balance=tend.balance.to_string()
                            >
                                <span class="balance-currency">"TEND"</span>
                                <span class="balance-value">{format!("{:+}h", tend.balance)}</span>
                                <span class="balance-subtitle">{tend.equilibrium_label()}</span>
                            </div>
                        }
                    }}
                    // SAP
                    {move || {
                        let sap = fin.sap_balance.get();
                        view! {
                            <div
                                class="balance-card balance-sap"
                                data-currency="SAP"
                                data-balance=sap.balance.to_string()
                            >
                                <span class="balance-currency">"SAP"</span>
                                <span class="balance-value">{format!("{:.2}", sap.display_balance())}</span>
                                <span class="balance-subtitle">"stored value"</span>
                            </div>
                        }
                    }}
                    // MYCEL
                    {move || {
                        let mycel = fin.mycel_score.get();
                        view! {
                            <div
                                class=format!("balance-card balance-mycel {}", mycel.tier.css_class())
                                data-currency="MYCEL"
                                data-score=format!("{:.3}", mycel.score)
                            >
                                <span class="balance-currency">"MYCEL"</span>
                                <span class="balance-value">{format!("{:.2}", mycel.score)}</span>
                                <span class="balance-subtitle">{mycel.tier.label()}</span>
                            </div>
                        }
                    }}
                </div>
            </section>

            // Governance summary
            <section class="profile-governance" aria-label="governance activity" data-section="governance-summary">
                <h2 class="section-title">"Governance"</h2>
                <div class="governance-stats">
                    {move || {
                        let vote_count = gov.my_votes.get().len();
                        let council_count = gov.councils.get().len(); // TODO: filter to my memberships
                        let delegation_count = gov.delegations.get().len();
                        view! {
                            <div class="gov-stat" data-metric="votes-cast">
                                <span class="stat-value">{vote_count.to_string()}</span>
                                <span class="stat-label">"votes cast"</span>
                            </div>
                            <div class="gov-stat" data-metric="councils">
                                <span class="stat-value">{council_count.to_string()}</span>
                                <span class="stat-label">"councils"</span>
                            </div>
                            <div class="gov-stat" data-metric="delegations">
                                <span class="stat-value">{delegation_count.to_string()}</span>
                                <span class="stat-label">"delegations"</span>
                            </div>
                        }
                    }}
                </div>
            </section>
        </div>
    }
}
