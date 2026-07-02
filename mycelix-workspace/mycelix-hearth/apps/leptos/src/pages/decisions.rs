// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::hearth_actions;
use mycelix_leptos_core::use_consciousness;
use crate::components::TierGate;
use hearth_leptos_types::*;
use personal_leptos_types::TrustTier;

#[component]
pub fn DecisionsPage() -> impl IntoView {
    let hearth = use_hearth();
    let consciousness = use_consciousness();

    let (voting_for, set_voting_for) = signal::<Option<String>>(None);
    let (selected_choice, set_selected_choice) = signal::<Option<u32>>(None);
    let (vote_reasoning, set_vote_reasoning) = signal(String::new());

    let do_cast_vote = move |decision_hash: String| {
        let choice = match selected_choice.get() {
            Some(c) => c,
            None => return,
        };
        let role = hearth.my_role.get().unwrap_or(MemberRole::Guest);
        let tier_bp = ((consciousness.profile.get().combined_score(&mycelix_leptos_core::consciousness::DimensionWeights::governance()) * 10000.0) as u32).min(10000);
        let weight = effective_vote_weight(&role, tier_bp);
        let reasoning = vote_reasoning.get();
        let reasoning = if reasoning.is_empty() { None } else { Some(reasoning) };

        hearth_actions::cast_vote(decision_hash, choice, reasoning, weight);

        set_voting_for.set(None);
        set_selected_choice.set(None);
        set_vote_reasoning.set(String::new());
    };

    view! {
        <div class="page decisions-page">
            <h1 class="page-title">"decision chamber"</h1>
            <p class="page-subtitle">"consciousness-weighted governance"</p>

            // Consciousness weight info
            <div class="consciousness-info">
                <span>"Your tier: "</span>
                <span class=move || format!("tier-label {}", consciousness.tier.get().css_class())>
                    {move || consciousness.tier.get().label()}
                </span>
                {move || {
                    let role = hearth.my_role.get().unwrap_or(MemberRole::Guest);
                    let role_bp = role.default_vote_weight_bp();
                    let tier_bp = ((consciousness.profile.get().combined_score(&mycelix_leptos_core::consciousness::DimensionWeights::governance()) * 10000.0) as u32).min(10000);
                    let effective = effective_vote_weight(&role, tier_bp);
                    view! {
                        <span class="weight-display">
                            {format!(" | Weight: {:.0}% (role {}% x tier {}%)",
                                effective as f64 / 100.0,
                                role_bp / 100,
                                tier_bp / 100
                            )}
                        </span>
                    }
                }}
            </div>

            <TierGate min_tier=TrustTier::Standard action_label="create decisions">
                <button class="action-btn">"+ New Decision"</button>
            </TierGate>

            // Decision cards
            {move || {
                let members = hearth.members.get();
                let decisions = hearth.decisions.get();
                let all_votes = hearth.votes.get();
                let my_agent = hearth.my_agent.get();
                let current_voting = voting_for.get();

                if decisions.is_empty() {
                    view! {
                        <div class="empty-state">"no crossroads at the moment. the path is clear."</div>
                    }.into_any()
                } else {
                    view! {
                        <div class="decision-list">
                            {decisions.iter().map(|d| {
                                let title = d.title.clone();
                                let desc = d.description.clone();
                                let proposer = member_name(&members, &d.created_by);
                                let decision_votes: Vec<_> = all_votes.iter()
                                    .filter(|v| v.decision_hash == d.hash)
                                    .cloned().collect();
                                let status_label = format!("{:?}", d.status);
                                let status_class = match d.status {
                                    DecisionStatus::Open => "decision-open",
                                    DecisionStatus::Closed => "decision-closed",
                                    DecisionStatus::Finalized => "decision-finalized",
                                };
                                let dtype = format!("{:?}", d.decision_type);
                                let options = d.options.clone();
                                let hash = d.hash.clone();
                                let is_open = d.status == DecisionStatus::Open;
                                let already_voted = decision_votes.iter().any(|v| v.voter == my_agent);
                                let is_voting = current_voting.as_ref() == Some(&hash);

                                view! {
                                    <div class=format!("decision-card {status_class}")>
                                        <div class="decision-header">
                                            <h3>{title}</h3>
                                            <div class="decision-badges">
                                                <span class="decision-type-badge">{dtype}</span>
                                                <span class="decision-status-badge">{status_label}</span>
                                            </div>
                                        </div>
                                        <p class="decision-desc">{desc}</p>
                                        <p class="decision-proposer">"Proposed by " {proposer}</p>

                                        // Options with tallies
                                        <div class="decision-options">
                                            {options.iter().enumerate().map(|(i, opt)| {
                                                let votes_for: u64 = decision_votes.iter()
                                                    .filter(|v| v.choice == i as u32)
                                                    .map(|v| v.weight_bp as u64)
                                                    .sum();
                                                let opt_text = opt.clone();
                                                let voters: Vec<_> = decision_votes.iter()
                                                    .filter(|v| v.choice == i as u32)
                                                    .map(|v| member_name(&members, &v.voter))
                                                    .collect();
                                                let choice_idx = i as u32;
                                                let is_selected = selected_choice.get_untracked() == Some(choice_idx) && is_voting;
                                                let can_select = is_voting;
                                                view! {
                                                    <div
                                                        class=format!("option-row {}", if is_selected { "option-selected" } else { "" })
                                                        on:click=move |_| {
                                                            if can_select {
                                                                set_selected_choice.set(Some(choice_idx));
                                                            }
                                                        }
                                                    >
                                                        {can_select.then(|| view! {
                                                            <span class="option-radio">
                                                                {if is_selected { "{25c9}" } else { "{25cb}" }}
                                                            </span>
                                                        })}
                                                        <span class="option-text">{opt_text}</span>
                                                        <div class="option-bar-container">
                                                            <div
                                                                class="option-bar"
                                                                style=format!("width: {}%", (votes_for as f64 / 100.0).min(100.0))
                                                            ></div>
                                                        </div>
                                                        <span class="option-pct">{format!("{:.0}%", votes_for as f64 / 100.0)}</span>
                                                        {(!voters.is_empty()).then(|| {
                                                            let vstr = voters.join(", ");
                                                            view! { <span class="option-voters">{vstr}</span> }
                                                        })}
                                                    </div>
                                                }
                                            }).collect_view()}
                                        </div>

                                        // Vote / voting UI
                                        {if is_open {
                                            let hash_for_btn = hash.clone();
                                            let hash_for_cast = hash.clone();
                                            if is_voting {
                                                view! {
                                                    <div class="vote-form">
                                                        <div class="form-row">
                                                            <label>"Reasoning (optional)"</label>
                                                            <input
                                                                type="text"
                                                                placeholder="Why this choice?"
                                                                prop:value=move || vote_reasoning.get()
                                                                on:input=move |ev| {
                                                                    use wasm_bindgen::JsCast;
                                                                    if let Some(input) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlInputElement>().cloned()) {
                                                                        set_vote_reasoning.set(input.value());
                                                                    }
                                                                }
                                                            />
                                                        </div>
                                                        <div class="vote-actions">
                                                            <button
                                                                class="action-btn"
                                                                disabled=move || selected_choice.get().is_none()
                                                                on:click=move |_| do_cast_vote(hash_for_cast.clone())
                                                            >
                                                                "Cast Vote"
                                                            </button>
                                                            <button
                                                                class="cancel-btn"
                                                                on:click=move |_| set_voting_for.set(None)
                                                            >
                                                                "Cancel"
                                                            </button>
                                                        </div>
                                                    </div>
                                                }.into_any()
                                            } else {
                                                view! {
                                                    <button
                                                        class=format!("vote-btn {}", if already_voted { "voted" } else { "" })
                                                        on:click=move |_| {
                                                            set_voting_for.set(Some(hash_for_btn.clone()));
                                                            set_selected_choice.set(None);
                                                        }
                                                    >
                                                        {if already_voted { "Change Vote" } else { "Vote" }}
                                                    </button>
                                                }.into_any()
                                            }
                                        } else {
                                            view! { <span></span> }.into_any()
                                        }}

                                        <div class="decision-meta">
                                            {format!("{} vote{} cast", decision_votes.len(), if decision_votes.len() != 1 { "s" } else { "" })}
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
