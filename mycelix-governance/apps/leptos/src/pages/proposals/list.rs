// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proposal list: living organisms, not a table.

use leptos::prelude::*;
use leptos_router::components::A;
use crate::contexts::governance_context::use_governance;
use crate::components::ProposalCard;

#[component]
pub fn ProposalListPage() -> impl IntoView {
    let gov = use_governance();

    view! {
        <div class="proposals-page">
            <div class="page-header">
                <h1 class="page-title">"Proposals"</h1>
                <A href="/proposals/new" attr:class="create-proposal-link">
                    "plant a new proposal"
                </A>
            </div>

            <div class="proposal-grid">
                {move || {
                    let proposals = gov.proposals.get();
                    if proposals.is_empty() {
                        view! {
                            <p class="empty-state">"the soil is quiet — no proposals are growing"</p>
                        }.into_any()
                    } else {
                        proposals.into_iter().map(|p| {
                            view! { <ProposalCard proposal=p /> }
                        }).collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}
