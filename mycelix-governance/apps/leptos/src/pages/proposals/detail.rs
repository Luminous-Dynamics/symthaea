// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proposal detail: the full lifecycle view of a single proposal.

use leptos::prelude::*;
use leptos_router::hooks::use_params_map;
use crate::contexts::governance_context::use_governance;
use crate::contexts::civic_actions;
use mycelix_leptos_core::use_consciousness;
use governance_leptos_types::*;

#[component]
pub fn ProposalDetailPage() -> impl IntoView {
    let gov = use_governance();
    let consciousness = use_consciousness();
    let params = use_params_map();

    let proposal = Memo::new(move |_| {
        let id = params.get().get("id").unwrap_or_default();
        gov.proposals.get()
            .into_iter()
            .find(|p| p.hash == id || p.id == id)
    });

    let my_vote = Memo::new(move |_| {
        proposal.get().and_then(|p| {
            gov.my_votes.get()
                .into_iter()
                .find(|v| v.proposal_id == p.id)
        })
    });

    view! {
        <div class="proposal-detail-page">
            {move || {
                match proposal.get() {
                    None => view! {
                        <p class="not-found">"this proposal has returned to the earth"</p>
                    }.into_any(),
                    Some(p) => {
                        let phase = p.status.phase();
                        let can_vote = p.status == ProposalStatus::Active;
                        let thresholds = gov.thresholds.get();
                        let score = consciousness.profile.get().combined_score();
                        let meets_voting_gate = score >= thresholds.voting;
                        let proposal_id = p.id.clone();

                        view! {
                            <article class=format!("proposal-organism phase-{phase}")>
                                <header class="proposal-header-detail">
                                    <span class="proposal-id-large">{p.id.clone()}</span>
                                    <span class=format!("proposal-type-badge {}", p.proposal_type.css_class())>
                                        {p.proposal_type.label()}
                                    </span>
                                    <span class="proposal-status-badge">
                                        {p.status.label()}
                                    </span>
                                </header>

                                <h1 class="proposal-title-large">{p.title}</h1>

                                <div class="proposal-body">
                                    <p>{p.description}</p>
                                </div>

                                <div class="proposal-author-detail">
                                    {"proposed by "}
                                    <span class="author-did">
                                        {p.author.split(':').last().unwrap_or(&p.author).to_string()}
                                    </span>
                                </div>

                                // Voting section
                                {can_vote.then(move || {
                                    let pid_for = proposal_id.clone();
                                    let pid_against = proposal_id.clone();
                                    let pid_abstain = proposal_id.clone();

                                    view! {
                                        <section class="voting-section" aria-label="cast your voice">
                                            <h2 class="section-subtitle">"Your voice"</h2>

                                            {move || my_vote.get().map(|v| view! {
                                                <div class=format!("current-vote {}", v.choice.css_class())>
                                                    {"you voted "}
                                                    <strong>{v.choice.label()}</strong>
                                                    {format!(" (weight: {:.2})", v.effective_weight)}
                                                </div>
                                            })}

                                            {if meets_voting_gate {
                                                view! {
                                                    <div class="vote-actions">
                                                        <button
                                                            class="vote-btn vote-for"
                                                            on:click=move |_| {
                                                                civic_actions::cast_vote(
                                                                    pid_for.clone(),
                                                                    VoteChoice::For,
                                                                    None,
                                                                );
                                                            }
                                                        >"For"</button>
                                                        <button
                                                            class="vote-btn vote-against"
                                                            on:click=move |_| {
                                                                civic_actions::cast_vote(
                                                                    pid_against.clone(),
                                                                    VoteChoice::Against,
                                                                    None,
                                                                );
                                                            }
                                                        >"Against"</button>
                                                        <button
                                                            class="vote-btn vote-abstain"
                                                            on:click=move |_| {
                                                                civic_actions::cast_vote(
                                                                    pid_abstain.clone(),
                                                                    VoteChoice::Abstain,
                                                                    None,
                                                                );
                                                            }
                                                        >"Abstain"</button>
                                                    </div>
                                                }.into_any()
                                            } else {
                                                view! {
                                                    <div class="gate-notice">
                                                        {format!(
                                                            "voting requires consciousness level {:.2} — you are at {:.2}",
                                                            thresholds.voting, score,
                                                        )}
                                                    </div>
                                                }.into_any()
                                            }}
                                        </section>
                                    }
                                })}

                                <div class=format!("proposal-pulse-large phase-{phase}-pulse")></div>
                            </article>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}
