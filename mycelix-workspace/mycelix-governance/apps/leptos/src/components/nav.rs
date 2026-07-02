// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Navigation — the civic spine.
//! AI-interactable: data-nav for each link, data-component="navigation".

use leptos::prelude::*;
use leptos_router::components::A;
use mycelix_leptos_core::{ConnectionBadge, use_consciousness};

#[component]
pub fn Nav() -> impl IntoView {
    let consciousness = use_consciousness();

    view! {
        <nav class="civic-nav" role="navigation" aria-label="civic navigation" data-component="navigation">
            <div class="nav-brand">
                <A href="/" attr:class="nav-logo" attr:aria-label="civic home">
                    <span class="logo-glyph">"⚖"</span>
                    <span class="logo-text">"Civic"</span>
                </A>
            </div>

            <div class="nav-links">
                // Governance
                <A href="/proposals" attr:class="nav-link" attr:data-nav="proposals">"Proposals"</A>
                <A href="/voting" attr:class="nav-link" attr:data-nav="voting">"Voting"</A>
                <A href="/councils" attr:class="nav-link" attr:data-nav="councils">"Councils"</A>
                <A href="/constitution" attr:class="nav-link" attr:data-nav="constitution">"Constitution"</A>
                // Finance
                <A href="/tend" attr:class="nav-link nav-finance" attr:data-nav="tend">"TEND"</A>
                <A href="/payments" attr:class="nav-link nav-finance" attr:data-nav="payments">"SAP"</A>
                <A href="/recognition" attr:class="nav-link nav-finance" attr:data-nav="recognition">"MYCEL"</A>
                <A href="/treasury" attr:class="nav-link nav-finance" attr:data-nav="treasury">"Treasury"</A>
            </div>

            <div class="nav-meta">
                <A href="/profile" attr:class="nav-profile-link" attr:data-nav="profile">
                    <span class=move || {
                        let tier = consciousness.tier.get();
                        format!("tier-badge tier-{}", tier.css_class())
                    }>
                        {move || consciousness.tier.get().label()}
                    </span>
                </A>
                <ConnectionBadge />
            </div>
        </nav>
    }
}
