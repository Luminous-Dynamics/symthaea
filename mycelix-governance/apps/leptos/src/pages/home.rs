// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civic home: a living overview of the commons' health.
//!
//! Not stat cards. A breathing organism that shows:
//! - How many proposals are growing (discussion)
//! - How many are deciding (voting)
//! - Whether the commons is at rest (homeostasis)
//! - The constitutional heartbeat

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;
use mycelix_leptos_core::use_consciousness;
use crate::components::ProposalCard;
use governance_leptos_types::*;

#[component]
pub fn HomePage() -> impl IntoView {
    let gov = use_governance();
    let consciousness = use_consciousness();

    let active_proposals = Memo::new(move |_| {
        gov.proposals.get()
            .into_iter()
            .filter(|p| p.status.is_active())
            .collect::<Vec<_>>()
    });

    let growing = Memo::new(move |_| {
        gov.proposals.get()
            .iter()
            .filter(|p| p.status == ProposalStatus::Draft)
            .count()
    });

    let deciding = Memo::new(move |_| {
        gov.proposals.get()
            .iter()
            .filter(|p| p.status == ProposalStatus::Active)
            .count()
    });

    view! {
        <div class="home-page">
            <section class="civic-breath" aria-label="commons health">
                <div class="breath-state">
                    {move || {
                        let g = growing.get();
                        let d = deciding.get();
                        if g == 0 && d == 0 {
                            view! {
                                <p class="breath-message homeostatic">
                                    "the commons is at rest"
                                </p>
                            }.into_any()
                        } else {
                            view! {
                                <div class="breath-activity">
                                    {(g > 0).then(|| view! {
                                        <p class="breath-growing">
                                            {format!("{g} proposal{} growing", if g == 1 { "" } else { "s" })}
                                        </p>
                                    })}
                                    {(d > 0).then(|| view! {
                                        <p class="breath-deciding">
                                            {format!("{d} decision{} in tension", if d == 1 { "" } else { "s" })}
                                        </p>
                                    })}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
                <div class="breath-tier">
                    <span class="tier-label">"your voice carries as "</span>
                    <span class=move || {
                        format!("tier-badge tier-{}", consciousness.tier.get().css_class())
                    }>
                        {move || consciousness.tier.get().label()}
                    </span>
                </div>
            </section>

            <section class="active-proposals" aria-label="active proposals">
                <h2 class="section-title">"Living proposals"</h2>
                <div class="proposal-grid">
                    {move || {
                        let proposals = active_proposals.get();
                        if proposals.is_empty() {
                            view! {
                                <p class="empty-state">"no proposals are growing right now"</p>
                            }.into_any()
                        } else {
                            proposals.into_iter().map(|p| {
                                view! { <ProposalCard proposal=p /> }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>

            {move || {
                gov.charter.get().map(|charter| view! {
                    <section class="charter-pulse" aria-label="constitutional heartbeat">
                        <h2 class="section-title">"Constitutional heartbeat"</h2>
                        <blockquote class="charter-preamble">
                            {charter.preamble}
                        </blockquote>
                        <div class="charter-rights">
                            {charter.rights.into_iter().map(|right| {
                                view! { <span class="right-badge">{right}</span> }
                            }).collect_view()}
                        </div>
                    </section>
                })
            }}
        </div>
    }
}
