// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Governance page — proposal creation, voting, and community decision-making.

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
// Mock data
// ---------------------------------------------------------------------------

fn mock_proposals() -> Vec<ProposalView> {
    vec![
        ProposalView {
            proposal_id: "prop_001".into(),
            title: "Add Rust Programming Track".into(),
            description: "Propose a comprehensive Rust programming track covering ownership, \
                          borrowing, lifetimes, async, and systems programming. The track would \
                          include 12 modules with spaced repetition cards and hands-on projects."
                .into(),
            proposer: "did:key:z6Mkf5rG...".into(),
            proposal_type: "Normal".into(),
            category: "Curriculum".into(),
            status: "Active".into(),
            for_votes: 12,
            against_votes: 3,
            abstain_votes: 2,
            weighted_for: 450,
            weighted_against: 120,
            voting_mode: "Quadratic".into(),
            voting_deadline: "3 days remaining".into(),
            created_at: "2026-03-20".into(),
        },
        ProposalView {
            proposal_id: "prop_002".into(),
            title: "Extend Review Session Limit to 200 Cards".into(),
            description: "Currently the daily review session cap is 100 cards. Power users \
                          report wanting more. Proposal to raise the limit to 200 while keeping \
                          the default at 100."
                .into(),
            proposer: "did:key:z6MkpT9...".into(),
            proposal_type: "Fast".into(),
            category: "Policy".into(),
            status: "Active".into(),
            for_votes: 8,
            against_votes: 5,
            abstain_votes: 1,
            weighted_for: 0,
            weighted_against: 0,
            voting_mode: "Simple".into(),
            voting_deadline: "18 hours remaining".into(),
            created_at: "2026-03-24".into(),
        },
        ProposalView {
            proposal_id: "prop_003".into(),
            title: "Community Mentorship Program".into(),
            description: "Establish a peer mentorship program where experienced learners \
                          can guide newcomers. Mentors earn reputation and XP. Includes \
                          matching algorithm based on skill domains and availability."
                .into(),
            proposer: "did:key:z6MkqR7...".into(),
            proposal_type: "Slow".into(),
            category: "Community".into(),
            status: "Active".into(),
            for_votes: 25,
            against_votes: 2,
            abstain_votes: 4,
            weighted_for: 0,
            weighted_against: 0,
            voting_mode: "Simple".into(),
            voting_deadline: "12 days remaining".into(),
            created_at: "2026-03-18".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Governance page (layout)
// ---------------------------------------------------------------------------

#[component]
pub fn GovernancePage() -> impl IntoView {
    let hc = use_holochain();

    let proposals = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc
                .call_zome::<(), Vec<ProposalView>>("dao", "list_active_proposals", &())
                .await
            {
                Ok(p) => p,
                Err(_) => mock_proposals(),
            }
        }
    });

    let (selected, set_selected) = signal::<Option<usize>>(None);
    let (show_form, set_show_form) = signal(false);

    view! {
        <div class="governance-page">
            <div class="governance-header">
                <div>
                    <h2>"Community Governance"</h2>
                    <p class="governance-subtitle">
                        "Participate in curriculum decisions through democratic voting."
                    </p>
                </div>
                <button
                    class="btn-primary"
                    on:click=move |_| set_show_form.update(|v| *v = !*v)
                >
                    {move || if show_form.get() { "Cancel" } else { "New Proposal" }}
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
                        let data: Vec<ProposalView> = (*data).clone();
                        let selected_val = selected.get();

                        if let Some(idx) = selected_val {
                            let proposal = data[idx].clone();
                            view! {
                                <div>
                                    <button
                                        class="btn-back"
                                        on:click=move |_| set_selected.set(None)
                                    >
                                        "< Back to proposals"
                                    </button>
                                    <ProposalDetail proposal=proposal />
                                </div>
                            }
                            .into_any()
                        } else {
                            let cards = data
                                .into_iter()
                                .enumerate()
                                .map(|(idx, proposal)| {
                                    view! {
                                        <ProposalCard
                                            proposal=proposal
                                            on_select=move || set_selected.set(Some(idx))
                                        />
                                    }
                                })
                                .collect_view();

                            view! {
                                <div class="proposals-section">
                                    <h3 class="section-label">"Active Proposals"</h3>
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
// ProposalCard
// ---------------------------------------------------------------------------

#[component]
fn ProposalCard(proposal: ProposalView, on_select: impl Fn() + 'static) -> impl IntoView {
    let total = proposal.for_votes + proposal.against_votes;
    let for_pct = if total > 0 {
        (proposal.for_votes as f64 / total as f64 * 100.0) as u32
    } else {
        50
    };

    let category = proposal.category.clone();
    let category_lower = category.to_lowercase();
    let category_display = category.clone();
    let voting_mode = proposal.voting_mode.clone();
    let proposal_type = proposal.proposal_type.clone();
    let ptype_lower = proposal_type.to_lowercase();
    let ptype_display = proposal_type.clone();
    let deadline = proposal.voting_deadline.clone();
    let for_v = proposal.for_votes;
    let against_v = proposal.against_votes;

    view! {
        <div class="proposal-card" on:click=move |_| on_select()>
            <div class="proposal-card-header">
                <h4 class="proposal-title">{proposal.title}</h4>
                <div class="proposal-badges">
                    <span class=format!("category-badge cat-{}", category_lower)>
                        {category_display}
                    </span>
                    <span class=format!("type-badge type-{}", ptype_lower)>
                        {ptype_display}
                    </span>
                    <VotingModeIndicator mode=voting_mode />
                </div>
            </div>
            <p class="proposal-excerpt">{proposal.description}</p>
            <VoteBar for_pct=for_pct for_count=for_v against_count=against_v />
            <div class="proposal-card-footer">
                <span class="deadline-text">{deadline}</span>
                <span class="total-votes">{total} " votes"</span>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// VoteBar
// ---------------------------------------------------------------------------

#[component]
fn VoteBar(for_pct: u32, for_count: u32, against_count: u32) -> impl IntoView {
    view! {
        <div class="vote-bar-wrap">
            <div class="vote-bar-labels">
                <span class="vote-for-label">{for_count} " For"</span>
                <span class="vote-against-label">{against_count} " Against"</span>
            </div>
            <div class="vote-bar-container">
                <div class="vote-bar-for" style=format!("width: {}%", for_pct)></div>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// VotingModeIndicator
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
// ProposalDetail
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
    let (user_vote, set_user_vote) = signal::<Option<String>>(None);
    let (rep_alloc, set_rep_alloc) = signal(10_u32);

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

                <div class="cast-vote-section">
                    <h4>"Cast Your Vote"</h4>

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
                            let is_q = is_quadratic;
                            view! {
                                <div class="vote-actions">
                                    <div class="vote-buttons">
                                        <button
                                            class="vote-btn vote-btn-for"
                                            on:click=move |_| set_user_vote.set(Some("For".into()))
                                        >
                                            "For"
                                        </button>
                                        <button
                                            class="vote-btn vote-btn-against"
                                            on:click=move |_| set_user_vote.set(Some("Against".into()))
                                        >
                                            "Against"
                                        </button>
                                        <button
                                            class="vote-btn vote-btn-abstain"
                                            on:click=move |_| set_user_vote.set(Some("Abstain".into()))
                                        >
                                            "Abstain"
                                        </button>
                                    </div>

                                    {if is_q {
                                        Some(view! {
                                            <div class="rep-allocation">
                                                <label>"Reputation to allocate: " {move || rep_alloc.get()}</label>
                                                <input
                                                    type="range"
                                                    min="1"
                                                    max="100"
                                                    prop:value=move || rep_alloc.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_rep_alloc.set(v);
                                                        }
                                                    }
                                                />
                                                <p class="rep-note">
                                                    "Quadratic voting: your vote weight = sqrt(reputation). "
                                                    "Allocating " {move || rep_alloc.get()} " rep gives "
                                                    {move || format!("{:.1}", (rep_alloc.get() as f64).sqrt())}
                                                    " vote weight."
                                                </p>
                                            </div>
                                        })
                                    } else {
                                        None
                                    }}
                                </div>
                            }
                            .into_any()
                        }
                    }}
                </div>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// CreateProposalForm
// ---------------------------------------------------------------------------

#[component]
fn CreateProposalForm(on_close: impl Fn() + Send + Sync + 'static) -> impl IntoView {
    let on_close = std::sync::Arc::new(on_close);
    let (title, set_title) = signal(String::new());
    let (description, set_description) = signal(String::new());
    let (category, set_category) = signal("Curriculum".to_string());
    let (proposal_type, set_proposal_type) = signal("Normal".to_string());
    let (voting_mode, set_voting_mode) = signal("Simple".to_string());
    let (submitted, set_submitted) = signal(false);

    view! {
        <div class="create-proposal-form">
            <h3>"Create New Proposal"</h3>

            {move || {
                let on_close_done = on_close.clone();
                let on_close_cancel = on_close.clone();
                if submitted.get() {
                    view! {
                        <div class="form-success">
                            <p>"Proposal submitted successfully (mock)."</p>
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
                                <label>"Title"</label>
                                <input
                                    type="text"
                                    placeholder="Proposal title (max 200 characters)"
                                    maxlength="200"
                                    prop:value=move || title.get()
                                    on:input=move |ev| set_title.set(event_target_value(&ev))
                                />
                            </div>

                            <div class="form-group">
                                <label>"Description"</label>
                                <textarea
                                    placeholder="Describe your proposal in detail..."
                                    rows="5"
                                    maxlength="10000"
                                    prop:value=move || description.get()
                                    on:input=move |ev| set_description.set(event_target_value(&ev))
                                ></textarea>
                            </div>

                            <div class="form-row">
                                <div class="form-group">
                                    <label>"Category"</label>
                                    <select
                                        prop:value=move || category.get()
                                        on:change=move |ev| set_category.set(event_target_value(&ev))
                                    >
                                        <option value="Curriculum">"Curriculum"</option>
                                        <option value="Policy">"Policy"</option>
                                        <option value="Community">"Community"</option>
                                        <option value="Finance">"Finance"</option>
                                        <option value="Technical">"Technical"</option>
                                    </select>
                                </div>

                                <div class="form-group">
                                    <label>"Proposal Track"</label>
                                    <select
                                        prop:value=move || proposal_type.get()
                                        on:change=move |ev| {
                                            set_proposal_type.set(event_target_value(&ev))
                                        }
                                    >
                                        <option value="Fast">"Fast (24-72h)"</option>
                                        <option value="Normal">"Normal (3-14 days)"</option>
                                        <option value="Slow">"Slow (14-30 days)"</option>
                                    </select>
                                </div>

                                <div class="form-group">
                                    <label>"Voting Mode"</label>
                                    <select
                                        prop:value=move || voting_mode.get()
                                        on:change=move |ev| {
                                            set_voting_mode.set(event_target_value(&ev))
                                        }
                                    >
                                        <option value="Simple">"Simple (1 person = 1 vote)"</option>
                                        <option value="Quadratic">
                                            "Quadratic (sqrt reputation)"
                                        </option>
                                    </select>
                                </div>
                            </div>

                            <div class="form-actions">
                                <button
                                    class="btn-primary"
                                    on:click=move |_| {
                                        if !title.get().is_empty() && !description.get().is_empty() {
                                            set_submitted.set(true);
                                        }
                                    }
                                >
                                    "Submit Proposal"
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
