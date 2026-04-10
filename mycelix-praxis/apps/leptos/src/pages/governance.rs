// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Suggest Topics page — simplified governance for teachers, parents, and students.
//! Advanced governance features (voting modes, proposal types, quadratic voting)
//! are accessible via an "Advanced view" toggle.

use leptos::prelude::*;

use crate::holochain::use_holochain;

// ---------------------------------------------------------------------------
// Data types (mirror dao_zome integrity types for the UI layer)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProposalView {
    pub proposal_id: String,
    pub title: String,
    pub description: String,
    pub proposer: String,
    pub proposal_type: String,
    pub category: String,
    pub status: String,
    pub for_votes: u32,
    pub against_votes: u32,
    pub abstain_votes: u32,
    pub weighted_for: u64,
    pub weighted_against: u64,
    pub voting_mode: String,
    pub voting_deadline: String,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Mock data (teacher/parent friendly examples)
// ---------------------------------------------------------------------------

fn real_proposals() -> Vec<ProposalView> {
    let store = crate::persistence::GovernanceStore::load();
    let mut proposals: Vec<ProposalView> = store.proposals.iter().map(|p| ProposalView {
        proposal_id: p.id.clone(),
        title: p.title.clone(),
        description: p.description.clone(),
        proposer: p.proposer.clone(),
        proposal_type: "Normal".into(),
        category: p.category.clone(),
        status: p.status.clone(),
        for_votes: p.for_votes,
        against_votes: p.against_votes,
        abstain_votes: 0,
        weighted_for: 0,
        weighted_against: 0,
        voting_mode: "Simple".into(),
        voting_deadline: if p.status == "Active" { "Ongoing".into() } else { "Closed".into() },
        created_at: p.created_at.clone(),
    }).collect();

    // Add seed proposals if store is empty (first-time experience)
    if proposals.is_empty() {
        proposals = vec![
            ProposalView {
                proposal_id: "seed_001".into(),
                title: "Add Gr11 Life Sciences content".into(),
                description: "Life Sciences is a popular matric subject but has no content yet. Adding Gr11 LS would help students build the foundation for Gr12.".into(),
                proposer: "Community".into(),
                proposal_type: "Normal".into(),
                category: "Curriculum".into(),
                status: "Active".into(),
                for_votes: 3,
                against_votes: 0,
                abstain_votes: 0,
                weighted_for: 0,
                weighted_against: 0,
                voting_mode: "Simple".into(),
                voting_deadline: "Ongoing".into(),
                created_at: "2026-03-31".into(),
            },
            ProposalView {
                proposal_id: "seed_002".into(),
                title: "Add isiZulu and Afrikaans language content".into(),
                description: "SA has 11 official languages. Adding basic language learning for isiZulu and Afrikaans would serve millions of students.".into(),
                proposer: "Community".into(),
                proposal_type: "Normal".into(),
                category: "Curriculum".into(),
                status: "Active".into(),
                for_votes: 5,
                against_votes: 0,
                abstain_votes: 0,
                weighted_for: 0,
                weighted_against: 0,
                voting_mode: "Simple".into(),
                voting_deadline: "Ongoing".into(),
                created_at: "2026-03-31".into(),
            },
        ];
    }
    proposals
}

// ---------------------------------------------------------------------------
// Friendly category label
// ---------------------------------------------------------------------------

fn friendly_category(cat: &str) -> &str {
    match cat {
        "Curriculum" => "Lessons & Content",
        "Policy" => "Rules & Guidelines",
        "Community" => "Community",
        "Finance" => "Resources",
        "Technical" => "Tools & Technology",
        _ => cat,
    }
}

fn category_css(cat: &str) -> &str {
    match cat.to_lowercase().as_str() {
        "mathematics" => "cat-curriculum",
        "science" => "cat-community",
        "computer science" => "cat-technical",
        "lessons & content" | "curriculum" => "cat-curriculum",
        "rules & guidelines" | "policy" => "cat-policy",
        "community" => "cat-community",
        "resources" | "finance" => "cat-finance",
        "tools & technology" | "technical" => "cat-technical",
        _ => "cat-curriculum",
    }
}

// ---------------------------------------------------------------------------
// Suggest Topics page (layout)
// ---------------------------------------------------------------------------

#[component]
pub fn GovernancePage() -> impl IntoView {
    let hc = use_holochain();

    let proposals = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc
                .call_zome_default::<(), Vec<ProposalView>>("dao", "list_active_proposals", &())
                .await
            {
                Ok(p) => p,
                Err(_) => real_proposals(),
            }
        }
    });

    let (selected, set_selected) = signal::<Option<usize>>(None);
    let (show_form, set_show_form) = signal(false);

    view! {
        <div class="governance-page">
            <div class="governance-header">
                <div>
                    <h2>"Suggest Topics"</h2>
                    <p class="governance-subtitle">
                        "Have an idea for something we should learn? Suggest it here and vote on ideas from others."
                    </p>
                </div>
                <button
                    class="btn-primary"
                    on:click=move |_| set_show_form.update(|v| *v = !*v)
                >
                    {move || if show_form.get() { "Cancel" } else { "Suggest a Topic" }}
                </button>
            </div>

            {move || {
                if show_form.get() {
                    Some(view! { <CreateProposalForm on_close=move || set_show_form.set(false) /> })
                } else {
                    None
                }
            }}

            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    proposals.get().map(|data| {
                        let data: Vec<ProposalView> = data.clone();
                        let selected_val = selected.get();

                        if let Some(idx) = selected_val {
                            let proposal = data[idx].clone();
                            let (show_advanced, set_show_advanced) = signal(false);
                            view! {
                                <div>
                                    <button
                                        class="btn-back"
                                        on:click=move |_| set_selected.set(None)
                                    >
                                        "< Back to suggestions"
                                    </button>
                                    <SimpleTopicDetail proposal=proposal.clone() />

                                    // Advanced view toggle
                                    <div class="advanced-toggle-section">
                                        <button
                                            class="advanced-toggle-btn"
                                            on:click=move |_| set_show_advanced.update(|v| *v = !*v)
                                        >
                                            {move || if show_advanced.get() {
                                                "Hide advanced view"
                                            } else {
                                                "Advanced view"
                                            }}
                                        </button>
                                        {move || {
                                            if show_advanced.get() {
                                                Some(view! { <ProposalDetail proposal=proposal.clone() /> })
                                            } else {
                                                None
                                            }
                                        }}
                                    </div>
                                </div>
                            }
                            .into_any()
                        } else {
                            let cards = data
                                .into_iter()
                                .enumerate()
                                .map(|(idx, proposal)| {
                                    view! {
                                        <SimpleProposalCard
                                            proposal=proposal
                                            on_select=move || set_selected.set(Some(idx))
                                        />
                                    }
                                })
                                .collect_view();

                            view! {
                                <div class="proposals-section">
                                    <h3 class="section-label">"Current Suggestions"</h3>
                                    <div class="proposals-list">{cards}</div>
                                </div>
                            }
                            .into_any()
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// SimpleProposalCard — teacher/parent friendly
// ---------------------------------------------------------------------------

#[component]
fn SimpleProposalCard(proposal: ProposalView, on_select: impl Fn() + 'static) -> impl IntoView {
    let total = proposal.for_votes + proposal.against_votes;
    let for_pct = if total > 0 {
        (proposal.for_votes as f64 / total as f64 * 100.0) as u32
    } else {
        50
    };

    let cat_css = category_css(&proposal.category);
    let category_display = proposal.category.clone();
    let proposer = proposal.proposer.clone();
    let for_v = proposal.for_votes;
    let against_v = proposal.against_votes;

    view! {
        <div class="proposal-card" on:click=move |_| on_select()>
            <h4 class="proposal-title">{proposal.title}</h4>
            <p class="proposal-byline">
                "by " {proposer}
                " · "
                <span class=format!("category-badge {}", cat_css)>{category_display}</span>
            </p>
            <VoteBar for_pct=for_pct for_count=for_v against_count=against_v />
        </div>
    }
}

// ---------------------------------------------------------------------------
// SimpleTopicDetail — friendly detail view with simple voting
// ---------------------------------------------------------------------------

#[component]
fn SimpleTopicDetail(proposal: ProposalView) -> impl IntoView {
    let (user_vote, set_user_vote) = signal::<Option<String>>(None);
    let for_v = proposal.for_votes;
    let against_v = proposal.against_votes;
    let total = for_v + against_v;
    let for_pct = if total > 0 {
        (for_v as f64 / total as f64 * 100.0) as u32
    } else {
        50
    };
    let cat_css = category_css(&proposal.category);
    let category_display = proposal.category.clone();

    view! {
        <div class="simple-topic-detail">
            <h3>{proposal.title.clone()}</h3>
            <p class="proposal-byline">
                "Suggested by " {proposal.proposer.clone()}
                " · "
                <span class=format!("category-badge {}", cat_css)>{category_display}</span>
            </p>
            <p class="topic-description">{proposal.description.clone()}</p>

            <VoteBar for_pct=for_pct for_count=for_v against_count=against_v />

            <div class="simple-vote-section">
                {move || {
                    if let Some(ref choice) = user_vote.get() {
                        view! {
                            <div class="vote-confirmation">
                                <span>"You voted: "</span>
                                <strong>{choice.clone()}</strong>
                            </div>
                        }
                        .into_any()
                    } else {
                        view! {
                            <div class="simple-vote-buttons">
                                <button
                                    class="vote-btn vote-btn-for"
                                    on:click={
                                        let pid = proposal.proposal_id.clone();
                                        move |_| {
                                            let mut store = crate::persistence::GovernanceStore::load();
                                            store.vote(&pid, true);
                                            set_user_vote.set(Some("Yes — recorded!".into()));
                                        }
                                    }
                                >
                                    "Yes, add this"
                                </button>
                                <button
                                    class="vote-btn vote-btn-against"
                                    on:click={
                                        let pid = proposal.proposal_id.clone();
                                        move |_| {
                                            let mut store = crate::persistence::GovernanceStore::load();
                                            store.vote(&pid, false);
                                            set_user_vote.set(Some("No — recorded!".into()));
                                        }
                                    }
                                >
                                    "Not needed"
                                </button>
                            </div>
                        }
                        .into_any()
                    }
                }}
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// VoteBar — simplified labels
// ---------------------------------------------------------------------------

#[component]
fn VoteBar(for_pct: u32, for_count: u32, against_count: u32) -> impl IntoView {
    view! {
        <div class="vote-bar-wrap">
            <div class="vote-bar-labels">
                <span class="vote-for-label">"Yes " {for_count}</span>
                <span class="vote-against-label">"No " {against_count}</span>
            </div>
            <div class="vote-bar-container">
                <div class="vote-bar-for" style=format!("width: {}%", for_pct)></div>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// VotingModeIndicator (kept for Advanced view)
// ---------------------------------------------------------------------------

#[component]
fn VotingModeIndicator(mode: String) -> impl IntoView {
    let (css, label) = match mode.as_str() {
        "Quadratic" => ("mode-quadratic", "QV"),
        "Conviction" => ("mode-conviction", "CV"),
        _ => ("mode-simple", "1p1v"),
    };

    view! {
        <span class=format!("voting-mode-badge {}", css)>{label}</span>
    }
}

// ---------------------------------------------------------------------------
// ProposalDetail (full advanced view, hidden by default)
// ---------------------------------------------------------------------------

#[component]
fn ProposalDetail(proposal: ProposalView) -> impl IntoView {
    let total = proposal.for_votes + proposal.against_votes + proposal.abstain_votes;
    let for_pct = if total > 0 {
        (proposal.for_votes as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };
    let against_pct = if total > 0 {
        (proposal.against_votes as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };
    let abstain_pct = if total > 0 {
        (proposal.abstain_votes as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };

    let is_quadratic = proposal.voting_mode == "Quadratic";

    view! {
        <div class="proposal-detail">
            <div class="proposal-detail-header">
                <h3>{proposal.title.clone()}</h3>
                {
                    let cat = proposal.category.clone();
                    let cat_lower = cat.to_lowercase();
                    let ptype = proposal.proposal_type.clone();
                    let pt_lower = ptype.to_lowercase();
                    let vm = proposal.voting_mode.clone();
                    view! {
                        <div class="proposal-badges">
                            <span class=format!("category-badge cat-{}", cat_lower)>{cat}</span>
                            <span class=format!("type-badge type-{}", pt_lower)>{ptype}</span>
                            <VotingModeIndicator mode=vm />
                        </div>
                    }
                }
            </div>

            <div class="proposal-detail-body">
                <p>{proposal.description.clone()}</p>

                <div class="detail-meta">
                    <div class="meta-row">
                        <span class="meta-label">"Proposer"</span>
                        <span class="meta-value">{proposal.proposer.clone()}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">"Created"</span>
                        <span class="meta-value">{proposal.created_at.clone()}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">"Deadline"</span>
                        <span class="meta-value deadline-text">{proposal.voting_deadline.clone()}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">"Voting Mode"</span>
                        <span class="meta-value">{proposal.voting_mode.clone()}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">"Proposal Track"</span>
                        <span class="meta-value">{proposal.proposal_type.clone()}</span>
                    </div>
                </div>

                <div class="vote-breakdown">
                    <h4>"Vote Breakdown"</h4>
                    <div class="breakdown-bars">
                        <div class="breakdown-row">
                            <span class="breakdown-label vote-for-label">"For"</span>
                            <div class="breakdown-bar-bg">
                                <div
                                    class="breakdown-bar-fill vote-for-fill"
                                    style=format!("width: {}%", for_pct)
                                ></div>
                            </div>
                            <span class="breakdown-count">{proposal.for_votes} " (" {for_pct} "%)"</span>
                        </div>
                        <div class="breakdown-row">
                            <span class="breakdown-label vote-against-label">"Against"</span>
                            <div class="breakdown-bar-bg">
                                <div
                                    class="breakdown-bar-fill vote-against-fill"
                                    style=format!("width: {}%", against_pct)
                                ></div>
                            </div>
                            <span class="breakdown-count">{proposal.against_votes} " (" {against_pct} "%)"</span>
                        </div>
                        <div class="breakdown-row">
                            <span class="breakdown-label vote-abstain-label">"Abstain"</span>
                            <div class="breakdown-bar-bg">
                                <div
                                    class="breakdown-bar-fill vote-abstain-fill"
                                    style=format!("width: {}%", abstain_pct)
                                ></div>
                            </div>
                            <span class="breakdown-count">{proposal.abstain_votes} " (" {abstain_pct} "%)"</span>
                        </div>
                    </div>

                    {if is_quadratic {
                        Some(view! {
                            <div class="weighted-tallies">
                                <h4>"Weighted Tallies (Quadratic)"</h4>
                                <div class="tally-row">
                                    <span class="vote-for-label">"Weighted For: " {proposal.weighted_for}</span>
                                    <span class="vote-against-label">"Weighted Against: " {proposal.weighted_against}</span>
                                </div>
                            </div>
                        })
                    } else {
                        None
                    }}
                </div>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// CreateProposalForm — simplified for teachers/parents
// ---------------------------------------------------------------------------

#[component]
fn CreateProposalForm(on_close: impl Fn() + Send + Sync + 'static) -> impl IntoView {
    let on_close = std::sync::Arc::new(on_close);
    let (title, set_title) = signal(String::new());
    let (description, set_description) = signal(String::new());
    let (category, set_category) = signal("Mathematics".to_string());
    let (submitted, set_submitted) = signal(false);

    view! {
        <div class="create-proposal-form">
            <h3>"Suggest a Topic"</h3>

            {move || {
                let on_close_done = on_close.clone();
                let on_close_cancel = on_close.clone();
                if submitted.get() {
                    view! {
                        <div class="form-success">
                            <p>"Your suggestion has been submitted! Others can now vote on it."</p>
                            <button class="btn-primary" on:click=move |_| on_close_done()>
                                "Done"
                            </button>
                        </div>
                    }
                    .into_any()
                } else {
                    view! {
                        <div class="form-fields">
                            <div class="form-group">
                                <label>"What would you like to add?"</label>
                                <input
                                    type="text"
                                    placeholder="e.g. More fraction practice problems"
                                    maxlength="200"
                                    prop:value=move || title.get()
                                    on:input=move |ev| set_title.set(event_target_value(&ev))
                                />
                            </div>

                            <div class="form-group">
                                <label>"Tell us more about your idea"</label>
                                <textarea
                                    placeholder="Why is this important? How would it help students learn?"
                                    rows="4"
                                    maxlength="10000"
                                    prop:value=move || description.get()
                                    on:input=move |ev| set_description.set(event_target_value(&ev))
                                ></textarea>
                            </div>

                            <div class="form-group">
                                <label>"Subject Area"</label>
                                <select
                                    prop:value=move || category.get()
                                    on:change=move |ev| set_category.set(event_target_value(&ev))
                                >
                                    <option value="Mathematics">"Mathematics"</option>
                                    <option value="English Language Arts">"English Language Arts"</option>
                                    <option value="Science">"Science"</option>
                                    <option value="Social Studies">"Social Studies"</option>
                                    <option value="Computer Science">"Computer Science"</option>
                                </select>
                            </div>

                            <div class="form-actions">
                                <button
                                    class="btn-primary"
                                    on:click=move |_| {
                                        if !title.get().is_empty() && !description.get().is_empty() {
                                            let profile = crate::persistence::load::<crate::student_profile::StudentProfile>("praxis_profile");
                                            let name = profile.map(|p| p.name).unwrap_or_else(|| "Anonymous".into());
                                            let mut store = crate::persistence::GovernanceStore::load();
                                            store.create_proposal(
                                                title.get_untracked(),
                                                description.get_untracked(),
                                                category.get_untracked(),
                                                name,
                                            );
                                            set_submitted.set(true);
                                        }
                                    }
                                >
                                    "Submit Suggestion"
                                </button>
                                <button class="btn-secondary" on:click=move |_| on_close_cancel()>
                                    "Cancel"
                                </button>
                            </div>
                        </div>
                    }
                    .into_any()
                }
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Shared loading skeleton
// ---------------------------------------------------------------------------

#[component]
fn CardLoading() -> impl IntoView {
    view! {
        <div class="card-loading">
            <div class="skeleton-line wide"></div>
            <div class="skeleton-line medium"></div>
            <div class="skeleton-line narrow"></div>
        </div>
    }
}
