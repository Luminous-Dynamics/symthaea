// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Voting hub: delegation management, active votes, voice credit balance.
//! AI-interactable via data-* attributes on all actionable elements.

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;
use mycelix_leptos_core::use_consciousness;
use governance_leptos_types::*;

#[component]
pub fn VotingPage() -> impl IntoView {
    let gov = use_governance();
    let consciousness = use_consciousness();

    let active_votes = Memo::new(move |_| {
        gov.proposals.get()
            .into_iter()
            .filter(|p| p.status == ProposalStatus::Active)
            .collect::<Vec<_>>()
    });

    let my_vote_count = Memo::new(move |_| gov.my_votes.get().len());

    view! {
        <div class="voting-page" data-page="voting" role="main">
            <h1 class="page-title">"Your Voice"</h1>

            // Consciousness weight display (AI-readable)
            <section
                class="voice-weight-section"
                aria-label="voting weight"
                data-section="voice-weight"
            >
                <div class="voice-weight-card" data-component="voice-weight">
                    <div class="weight-visual">
                        <span
                            class="weight-value"
                            data-weight=move || format!("{:.2}", consciousness.profile.get().combined_score())
                        >
                            {move || format!("{:.2}", consciousness.profile.get().combined_score())}
                        </span>
                        <span class="weight-label">"effective weight"</span>
                    </div>
                    <div class="weight-breakdown" data-component="weight-breakdown">
                        {move || {
                            let p = consciousness.profile.get();
                            view! {
                                <div class="weight-dim" data-dimension="identity" data-value=format!("{:.2}", p.identity)>
                                    <span class="dim-label">"identity"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill" style=format!("width: {}%", p.identity * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", p.identity)}</span>
                                </div>
                                <div class="weight-dim" data-dimension="reputation" data-value=format!("{:.2}", p.reputation)>
                                    <span class="dim-label">"reputation"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill" style=format!("width: {}%", p.reputation * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", p.reputation)}</span>
                                </div>
                                <div class="weight-dim" data-dimension="community" data-value=format!("{:.2}", p.community)>
                                    <span class="dim-label">"community"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill" style=format!("width: {}%", p.community * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", p.community)}</span>
                                </div>
                                <div class="weight-dim" data-dimension="engagement" data-value=format!("{:.2}", p.engagement)>
                                    <span class="dim-label">"engagement"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill" style=format!("width: {}%", p.engagement * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", p.engagement)}</span>
                                </div>
                            }
                        }}
                    </div>
                </div>
            </section>

            // Active votes needing attention
            <section
                class="active-votes-section"
                aria-label="proposals awaiting your vote"
                data-section="active-votes"
                data-count=move || active_votes.get().len().to_string()
            >
                <h2 class="section-title">"Decisions in tension"</h2>
                {move || {
                    let votes = active_votes.get();
                    if votes.is_empty() {
                        view! {
                            <p class="empty-state" data-state="empty">"no decisions await your voice"</p>
                        }.into_any()
                    } else {
                        votes.into_iter().map(|p| {
                            let pid = p.id.clone();
                            let pid2 = pid.clone();
                            let pid3 = pid.clone();
                            let title = p.title.clone();
                            let hash = p.hash.clone();
                            view! {
                                <div
                                    class="vote-pending-card"
                                    data-proposal-id=pid.clone()
                                    data-proposal-hash=hash
                                    data-status="awaiting-vote"
                                >
                                    <a href=format!("/proposals/{pid2}") class="vote-pending-link">
                                        <span class="vote-pending-id">{pid3}</span>
                                        <span class="vote-pending-title">{title}</span>
                                    </a>
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </section>

            // My vote history
            <section
                class="vote-history-section"
                aria-label="your voting history"
                data-section="vote-history"
                data-count=move || my_vote_count.get().to_string()
            >
                <h2 class="section-title">"Voices cast"</h2>
                {move || {
                    let votes = gov.my_votes.get();
                    if votes.is_empty() {
                        view! {
                            <p class="empty-state">"you have not yet spoken"</p>
                        }.into_any()
                    } else {
                        votes.into_iter().map(|v| {
                            let choice_class = v.choice.css_class().to_string();
                            let choice_label = v.choice.label().to_string();
                            let choice_str = format!("{:?}", v.choice);
                            let pid = v.proposal_id.clone();
                            let pid_display = pid.clone();
                            let weight = v.effective_weight;
                            view! {
                                <div
                                    class=format!("vote-record {choice_class}")
                                    data-proposal-id=pid.clone()
                                    data-choice=choice_str
                                    data-weight=format!("{:.2}", weight)
                                    role="listitem"
                                >
                                    <span class="vote-record-proposal">{pid_display}</span>
                                    <span class="vote-record-choice">{choice_label}</span>
                                    <span class="vote-record-weight">
                                        {format!("weight {weight:.2}")}
                                    </span>
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </section>

            // Delegations
            <section
                class="delegations-section"
                aria-label="vote delegations"
                data-section="delegations"
            >
                <h2 class="section-title">"Delegations"</h2>
                {move || {
                    let delegations = gov.delegations.get();
                    if delegations.is_empty() {
                        view! {
                            <p class="empty-state">"you carry your own voice — no delegations active"</p>
                        }.into_any()
                    } else {
                        delegations.into_iter().map(|d| {
                            let delegate = d.delegate_did.split(':').last()
                                .unwrap_or(&d.delegate_did).to_string();
                            let did_clone = d.delegate_did.clone();
                            let domain_data = d.domain.clone().unwrap_or_default();
                            let domain_display = d.domain.clone();
                            view! {
                                <div
                                    class="delegation-card"
                                    data-delegate-did=did_clone
                                    data-domain=domain_data
                                >
                                    <span class="delegation-to">{format!("delegated to {delegate}")}</span>
                                    {domain_display.map(|domain| view! {
                                        <span class="delegation-domain">{format!("({domain})")}</span>
                                    })}
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </section>
        </div>
    }
}
