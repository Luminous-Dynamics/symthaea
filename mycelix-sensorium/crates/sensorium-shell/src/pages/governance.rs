// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance domain — the first "excellent" domain module.
//!
//! Features:
//! - Intra-domain routing (Proposals / Councils / Constitution / Budget)
//! - Signal-based state store (proposals, councils loaded once, reactively shared)
//! - Proposal detail view with vote history
//! - Vote button that updates tally in real-time
//! - Create proposal form with conductor call + mock fallback
//! - Proper loading states via Suspense

use domain_governance::types::*;
use leptos::prelude::*;
use sensorium_viz::{bar_chart::Bar, BarChart, VoteTallyBar};

use crate::ai::{ai_state_json, AiState};
use crate::identity::{ConductorStatus, SensoriumIdentity};

// ── Governance sub-views ──
#[derive(Clone, Copy, Debug, PartialEq)]
enum GovView {
    Proposals,
    Councils,
    Constitution,
    Budget,
}

// ── State store ──

/// Governance state — loaded once, shared across all sub-views via context.
#[derive(Clone)]
struct GovState {
    proposals: RwSignal<Vec<ProposalSummary>>,
    councils: RwSignal<Vec<Council>>,
    loading: RwSignal<bool>,
    selected_proposal: RwSignal<Option<String>>,
    show_create_form: RwSignal<bool>,
}

impl GovState {
    fn new() -> Self {
        Self {
            proposals: RwSignal::new(Vec::new()),
            councils: RwSignal::new(Vec::new()),
            loading: RwSignal::new(true),
            selected_proposal: RwSignal::new(None),
            show_create_form: RwSignal::new(false),
        }
    }
}

// ── Mock data ──

fn mock_proposals() -> Vec<ProposalSummary> {
    vec![
        ProposalSummary {
            proposal: Proposal {
                id: "MIP-042".into(), title: "Establish community solar garden on Block 7".into(),
                description: "Convert the abandoned lot on Block 7 into a shared solar installation. Revenue shared via TEND to all Block 7 residents proportional to their energy contribution. Expected ROI: 3 years. Community benefit: 15kW capacity serving 12 households.".into(),
                proposal_type: ProposalType::Standard, author: "did:mycelix:uhCAkSolar".into(),
                status: ProposalStatus::Active, actions: "[]".into(), discussion_url: Some("https://forum.mycelix.net/p/42".into()),
                voting_starts: 1711900800, voting_ends: 1712505600,
                created: 1711814400, updated: 1711900800, version: 1,
            },
            tally: VoteTally { votes_for: 34, votes_against: 8, votes_abstain: 3, total_weight_for: 28.5, total_weight_against: 5.2, quorum_reached: true, approved: false },
            my_vote: Some(VoteChoice::For),
        },
        ProposalSummary {
            proposal: Proposal {
                id: "MIP-041".into(), title: "Increase care worker TEND allocation by 15%".into(),
                description: "Care workers are undercompensated relative to their contribution. This proposal increases their base TEND allocation from 100 to 115 SAP/month. Funded by reducing administrative overhead in the treasury.".into(),
                proposal_type: ProposalType::Funding, author: "did:mycelix:uhCAkCare".into(),
                status: ProposalStatus::Active, actions: "[]".into(), discussion_url: None,
                voting_starts: 1711814400, voting_ends: 1712419200,
                created: 1711728000, updated: 1711814400, version: 1,
            },
            tally: VoteTally { votes_for: 52, votes_against: 4, votes_abstain: 7, total_weight_for: 45.1, total_weight_against: 2.8, quorum_reached: true, approved: false },
            my_vote: None,
        },
        ProposalSummary {
            proposal: Proposal {
                id: "MIP-039".into(), title: "Adopt water stewardship protocol for Commons watershed".into(),
                description: "Implement tiered water allocation with emergency priority and seasonal adjustment. Based on 6 months of flow data from 8 water sources.".into(),
                proposal_type: ProposalType::Constitutional, author: "did:mycelix:uhCAkWater".into(),
                status: ProposalStatus::Executed, actions: "[]".into(), discussion_url: None,
                voting_starts: 1710604800, voting_ends: 1713196800,
                created: 1710518400, updated: 1713283200, version: 2,
            },
            tally: VoteTally { votes_for: 89, votes_against: 3, votes_abstain: 1, total_weight_for: 78.2, total_weight_against: 1.5, quorum_reached: true, approved: true },
            my_vote: Some(VoteChoice::For),
        },
    ]
}

fn mock_councils() -> Vec<Council> {
    vec![
        Council {
            id: "c-root".into(),
            name: "Root Council".into(),
            purpose: "Top-level governance and constitutional stewardship".into(),
            council_type: CouncilType::Root,
            parent_council_id: None,
            phi_threshold: 0.4,
            quorum: 0.51,
            supermajority: 0.67,
            status: CouncilStatus::Active,
            created_at: 1700000000,
        },
        Council {
            id: "c-energy".into(),
            name: "Energy Working Group".into(),
            purpose: "Solar, wind, and geothermal project coordination".into(),
            council_type: CouncilType::WorkingGroup {
                focus: "renewable energy".into(),
                expires: Some(1743350400),
            },
            parent_council_id: Some("c-root".into()),
            phi_threshold: 0.3,
            quorum: 0.4,
            supermajority: 0.6,
            status: CouncilStatus::Active,
            created_at: 1711000000,
        },
        Council {
            id: "c-care".into(),
            name: "Care Coordination Council".into(),
            purpose: "Coordinate care services across the community".into(),
            council_type: CouncilType::Domain {
                domain: "commons".into(),
            },
            parent_council_id: Some("c-root".into()),
            phi_threshold: 0.3,
            quorum: 0.4,
            supermajority: 0.6,
            status: CouncilStatus::Active,
            created_at: 1705000000,
        },
    ]
}

fn status_color(s: &ProposalStatus) -> &'static str {
    match s {
        ProposalStatus::Active => "#22c55e",
        ProposalStatus::Draft => "#6b7280",
        ProposalStatus::Approved | ProposalStatus::Executed => "#e8c547",
        ProposalStatus::Rejected | ProposalStatus::Failed => "#ef4444",
        _ => "#a78bfa",
    }
}

fn type_badge_color(t: &ProposalType) -> &'static str {
    match t {
        ProposalType::Standard => "#7C3AED",
        ProposalType::Emergency => "#ef4444",
        ProposalType::Constitutional => "#22c55e",
        ProposalType::Funding => "#FBBF24",
        ProposalType::Parameter => "#6b7280",
    }
}

// ── Main component ──

#[component]
pub fn GovernanceOverview() -> impl IntoView {
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");
    let (active_view, set_active_view) = signal(GovView::Proposals);
    // selected_proposal is in GovState (shared via context)

    // State store
    let state = GovState::new();
    provide_context(state.clone());

    // Load data
    let identity_for_load = identity.clone();
    let state_for_load = state.clone();
    wasm_bindgen_futures::spawn_local(async move {
        // Try conductor first
        let proposals = if identity_for_load.conductor_status.get() == ConductorStatus::Connected {
            identity_for_load
                .call_zome::<(), Vec<ProposalSummary>>(
                    "governance",
                    "proposals",
                    "list_active_proposals",
                    &(),
                )
                .await
                .unwrap_or_else(|_| mock_proposals())
        } else {
            mock_proposals()
        };
        state_for_load.proposals.set(proposals);

        let councils = if identity_for_load.conductor_status.get() == ConductorStatus::Connected {
            identity_for_load
                .call_zome::<(), Vec<Council>>("governance", "councils", "list_councils", &())
                .await
                .unwrap_or_else(|_| mock_councils())
        } else {
            mock_councils()
        };
        state_for_load.councils.set(councils);
        state_for_load.loading.set(false);
    });

    view! {
        <div class="governance-content" data-domain="governance" role="region" aria-label="Governance domain">
            // AI-readable state (updated reactively)
            {move || {
                let proposals = state.proposals.get();
                let json = ai_state_json("governance", &serde_json::json!({
                    "view": format!("{:?}", active_view.get()),
                    "proposal_count": proposals.len(),
                    "active_proposals": proposals.iter().filter(|p| p.proposal.status == ProposalStatus::Active).count(),
                    "selected": state.selected_proposal.get(),
                    "councils": state.councils.get().len(),
                }));
                view! { <AiState json=json /> }
            }}

            // Navigation tabs
            <div class="governance-nav" role="tablist" aria-label="Governance sections">
                <button class=move || if active_view.get() == GovView::Proposals { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| { set_active_view.set(GovView::Proposals); state.selected_proposal.set(None); }>"Proposals"</button>
                <button class=move || if active_view.get() == GovView::Councils { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(GovView::Councils)>"Councils"</button>
                <button class=move || if active_view.get() == GovView::Constitution { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(GovView::Constitution)>"Constitution"</button>
                <button class=move || if active_view.get() == GovView::Budget { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(GovView::Budget)>"Budget"</button>
            </div>

            // Loading state
            <Show when=move || state.loading.get()>
                <div class="domain-coming-soon">
                    <p class="coming-soon-text">"Loading governance data..."</p>
                </div>
            </Show>

            // Sub-views
            <Show when=move || !state.loading.get() && active_view.get() == GovView::Proposals>
                {move || {
                    if let Some(ref pid) = state.selected_proposal.get() {
                        // Detail view
                        let pid = pid.clone();
                        let proposal = state.proposals.get().iter().find(|p| p.proposal.id == pid).cloned();
                        if let Some(ps) = proposal {
                            view! { <ProposalDetail summary=ps /> }.into_any()
                        } else {
                            view! { <p>"Proposal not found"</p> }.into_any()
                        }
                    } else {
                        // List view
                        view! { <ProposalList /> }.into_any()
                    }
                }}
            </Show>

            <Show when=move || !state.loading.get() && active_view.get() == GovView::Councils>
                <CouncilList />
            </Show>

            <Show when=move || !state.loading.get() && active_view.get() == GovView::Constitution>
                <div class="thought-card">
                    <h3 class="section-title">"Constitutional Charter"</h3>
                    <p style="font-size: 0.85rem; line-height: 1.6; color: var(--text-primary)">
                        "We, the members of the Mycelix network, establish this charter to govern "
                        "our shared resources, protect individual sovereignty, and nurture collective "
                        "wellbeing through consciousness-aware decision-making."
                    </p>
                    <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.75rem">
                        "Version 2 \u{00b7} Ratified by 89 members \u{00b7} Last amended: Water Stewardship Protocol"
                    </p>
                </div>
            </Show>

            <Show when=move || !state.loading.get() && active_view.get() == GovView::Budget>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #FBBF24">"Q1 2026"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"50,000"</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"SAP total budget"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"PHASE"</div>
                        <p class="thought-content" style="font-size: 1.2rem; font-weight: 700">"Voting"</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"12 projects submitted"</p>
                    </div>
                </div>
                <h3 class="section-title">"Allocation by Domain"</h3>
                <BarChart data=vec![
                    Bar { label: "Care".into(), value: 18000.0, color: "#ec4899".into() },
                    Bar { label: "Energy".into(), value: 12000.0, color: "#22c55e".into() },
                    Bar { label: "Education".into(), value: 8000.0, color: "#2563EB".into() },
                    Bar { label: "Infra".into(), value: 7000.0, color: "#059669".into() },
                    Bar { label: "Reserve".into(), value: 5000.0, color: "#6b7280".into() },
                ] width=350.0 height=180.0 />
            </Show>
        </div>
    }
}

// ── Proposal List ──

#[component]
fn ProposalList() -> impl IntoView {
    let state = use_context::<GovState>().expect("GovState");

    view! {
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 0.7rem; color: var(--text-muted)">
                    {move || format!("{} proposals", state.proposals.get().len())}
                </span>
                <button class="domain-nav-btn" data-action="create-proposal" aria-label="Create new proposal"
                    on:click=move |_| state.show_create_form.update(|v| *v = !*v)>
                    {move || if state.show_create_form.get() { "- Close" } else { "+ New Proposal" }}
                </button>
            </div>

            <Show when=move || state.show_create_form.get()>
                <CreateProposalForm />
            </Show>

            <div class="proposals-list">
                {move || { state.proposals.get().iter().map(|ps| {
                    let id = ps.proposal.id.clone();
                    let title = ps.proposal.title.clone();
                    let ptype = ps.proposal.proposal_type.label().to_string();
                    let status_label = ps.proposal.status.label().to_string();
                    let s_color = status_color(&ps.proposal.status).to_string();
                    let t_color = type_badge_color(&ps.proposal.proposal_type).to_string();
                    let v_for = ps.tally.votes_for;
                    let v_against = ps.tally.votes_against;
                    let v_abstain = ps.tally.votes_abstain;
                    let my_vote = ps.my_vote.clone();
                    let id_for_click = id.clone();

                    view! {
                        <div class="proposal-card"
                            data-action="select-proposal"
                            data-id=id.clone()
                            role="button"
                            aria-label=format!("View proposal {}", id.clone())
                            style=format!("border-left: 3px solid {s_color}; cursor: pointer;")
                            on:click=move |_| state.selected_proposal.set(Some(id_for_click.clone()))
                        >
                            <div class="proposal-header">
                                <span class="proposal-id">{id.clone()}</span>
                                <span class="proposal-type-badge" style=format!("color: {t_color}")>{ptype}</span>
                                <span class="proposal-status" style=format!("color: {s_color}")>{status_label}</span>
                            </div>
                            <h3 class="proposal-title">{title}</h3>
                            <VoteTallyBar votes_for=v_for votes_against=v_against votes_abstain=v_abstain />
                            {my_vote.map(|v| {
                                let label = match v { VoteChoice::For => "You voted For", VoteChoice::Against => "You voted Against", VoteChoice::Abstain => "You abstained" };
                                view! { <span class="my-vote-badge">{label}</span> }
                            })}
                        </div>
                    }
                }).collect::<Vec<_>>() }}
            </div>
        </div>
    }
}

// ── Proposal Detail ──

#[component]
fn ProposalDetail(summary: ProposalSummary) -> impl IntoView {
    let state = use_context::<GovState>().expect("GovState");
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");
    let p = summary.proposal.clone();
    let tally = RwSignal::new(summary.tally.clone());
    let my_vote = RwSignal::new(summary.my_vote.clone());
    let (vote_submitting, set_vote_submitting) = signal(false);

    let proposal_id_for_vote = p.id.clone();
    let identity_for_vote = identity.clone();
    let cast_vote = move |choice: VoteChoice| {
        set_vote_submitting.set(true);
        let choice_clone = choice.clone();
        let identity = identity_for_vote.clone();
        let pid = proposal_id_for_vote.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let input =
                serde_json::json!({ "proposal_id": pid, "choice": format!("{:?}", choice_clone) });
            let _ = identity
                .call_zome::<serde_json::Value, serde_json::Value>(
                    "governance",
                    "voting",
                    "cast_vote",
                    &input,
                )
                .await;

            // Optimistic update
            my_vote.set(Some(choice_clone.clone()));
            tally.update(|t| match choice_clone {
                VoteChoice::For => {
                    t.votes_for += 1;
                    t.total_weight_for += 1.0;
                }
                VoteChoice::Against => {
                    t.votes_against += 1;
                    t.total_weight_against += 1.0;
                }
                VoteChoice::Abstain => {
                    t.votes_abstain += 1;
                }
            });
            set_vote_submitting.set(false);
        });
    };

    let s_color = status_color(&p.status).to_string();
    let t_color = type_badge_color(&p.proposal_type).to_string();
    view! {
        <div class="proposal-detail">
            <button class="domain-nav-btn" style="margin-bottom: 1rem;" on:click=move |_| state.selected_proposal.set(None)>
                "\u{2190} Back to proposals"
            </button>

            <div class="thought-card" style=format!("border-left: 3px solid {s_color};")>
                <div class="proposal-header" style="margin-bottom: 0.5rem;">
                    <span class="proposal-id">{p.id.clone()}</span>
                    <span class="proposal-type-badge" style=format!("color: {t_color}")>{p.proposal_type.label()}</span>
                    <span class="proposal-status" style=format!("color: {s_color}")>{p.status.label()}</span>
                </div>
                <h2 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--text-primary)">{p.title.clone()}</h2>
                <p style="font-size: 0.85rem; line-height: 1.6; color: var(--text-secondary); margin-bottom: 1rem;">
                    {p.description.clone()}
                </p>

                // Vote tally (reactive)
                {move || {
                    let t = tally.get();
                    view! { <VoteTallyBar votes_for=t.votes_for votes_against=t.votes_against votes_abstain=t.votes_abstain /> }
                }}

                // Vote buttons
                <Show when=move || my_vote.get().is_none() && p.status == ProposalStatus::Active>
                    <div class="vote-buttons" style="display: flex; gap: 0.5rem; margin-top: 0.75rem;">
                        <button class="form-submit" style="background: #22c55e;"
                            disabled=move || vote_submitting.get()
                            on:click={
                                let cast = cast_vote.clone();
                                move |_| cast(VoteChoice::For)
                            }>"Vote For"</button>
                        <button class="form-submit" style="background: #ef4444;"
                            disabled=move || vote_submitting.get()
                            on:click={
                                let cast = cast_vote.clone();
                                move |_| cast(VoteChoice::Against)
                            }>"Vote Against"</button>
                        <button class="form-submit" style="background: #6b7280;"
                            disabled=move || vote_submitting.get()
                            on:click={
                                let cast = cast_vote.clone();
                                move |_| cast(VoteChoice::Abstain)
                            }>"Abstain"</button>
                    </div>
                </Show>

                <Show when=move || my_vote.get().is_some()>
                    <p class="my-vote-badge" style="margin-top: 0.75rem;">
                        {move || match my_vote.get() {
                            Some(VoteChoice::For) => "You voted For",
                            Some(VoteChoice::Against) => "You voted Against",
                            Some(VoteChoice::Abstain) => "You abstained",
                            None => "",
                        }}
                    </p>
                </Show>

                // Metadata
                <div style="margin-top: 1rem; font-size: 0.7rem; color: var(--text-muted); display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span>"Author: " {p.author.chars().take(20).collect::<String>()}"..."</span>
                    <span>"Voting: " {p.proposal_type.voting_days()}" days"</span>
                    {p.discussion_url.clone().map(|url| view! {
                        <span>"Discussion: " {url}</span>
                    })}
                </div>
            </div>
        </div>
    }
}

// ── Council List ──

#[component]
fn CouncilList() -> impl IntoView {
    let state = use_context::<GovState>().expect("GovState");

    view! {
        <div class="councils-list">
            {move || { state.councils.get().iter().map(|c| {
                let name = c.name.clone();
                let purpose = c.purpose.clone();
                let ctype = match &c.council_type {
                    CouncilType::Root => "Root".to_string(),
                    CouncilType::Domain { domain } => format!("Domain: {domain}"),
                    CouncilType::WorkingGroup { focus, .. } => format!("WG: {focus}"),
                    CouncilType::Regional { region } => format!("Regional: {region}"),
                    CouncilType::Advisory => "Advisory".to_string(),
                    CouncilType::Emergency { .. } => "Emergency".to_string(),
                };
                let status = match c.status {
                    CouncilStatus::Active => ("Active", "#22c55e"),
                    CouncilStatus::Dormant => ("Dormant", "#6b7280"),
                    CouncilStatus::Dissolved => ("Dissolved", "#ef4444"),
                    CouncilStatus::Suspended => ("Suspended", "#f59e0b"),
                };

                view! {
                    <div class="thought-card" style=format!("border-left: 3px solid {};", status.1)>
                        <div class="thought-meta">
                            <span class="thought-type" style=format!("color: {}", status.1)>{status.0}</span>
                            <span class="thought-domain">{ctype}</span>
                        </div>
                        <h3 style="font-size: 0.95rem; font-weight: 600; margin-bottom: 0.3rem;">{name}</h3>
                        <p style="font-size: 0.8rem; color: var(--text-secondary)">{purpose}</p>
                        <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.4rem;">
                            {format!("Phi threshold: {:.1} \u{00b7} Quorum: {:.0}% \u{00b7} Supermajority: {:.0}%", c.phi_threshold, c.quorum * 100.0, c.supermajority * 100.0)}
                        </div>
                    </div>
                }
            }).collect::<Vec<_>>() }}
        </div>
    }
}

// ── Create Proposal Form ──

#[component]
fn CreateProposalForm() -> impl IntoView {
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");
    let state = use_context::<GovState>().expect("GovState");
    let (title, set_title) = signal(String::new());
    let (description, set_description) = signal(String::new());
    let (proposal_type, set_proposal_type) = signal("Standard".to_string());
    let (error_msg, set_error) = signal(None::<String>);
    let (submitting, set_submitting) = signal(false);

    let identity_for_submit = identity.clone();
    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        let t = title.get();
        let d = description.get();
        if t.trim().is_empty() || d.trim().is_empty() {
            set_error.set(Some("Title and description required".into()));
            return;
        }
        set_error.set(None);
        set_submitting.set(true);

        let identity = identity_for_submit.clone();
        let pt = proposal_type.get();
        let title_val = t.trim().to_string();
        let desc_val = d.trim().to_string();

        wasm_bindgen_futures::spawn_local(async move {
            let input = serde_json::json!({ "title": title_val, "description": desc_val, "proposal_type": pt });
            let _ = identity
                .call_zome::<serde_json::Value, serde_json::Value>(
                    "governance",
                    "proposals",
                    "create_proposal",
                    &input,
                )
                .await;

            // Optimistic: add to local state
            let new_proposal = ProposalSummary {
                proposal: Proposal {
                    id: format!("MIP-{:03}", state.proposals.get().len() + 43),
                    title: title_val,
                    description: desc_val,
                    proposal_type: match pt.as_str() {
                        "Emergency" => ProposalType::Emergency,
                        "Constitutional" => ProposalType::Constitutional,
                        "Funding" => ProposalType::Funding,
                        "Parameter" => ProposalType::Parameter,
                        _ => ProposalType::Standard,
                    },
                    author: "did:mycelix:uhCAkYou".into(),
                    status: ProposalStatus::Active,
                    actions: "[]".into(),
                    discussion_url: None,
                    voting_starts: 0,
                    voting_ends: 0,
                    created: 0,
                    updated: 0,
                    version: 1,
                },
                tally: VoteTally::default(),
                my_vote: None,
            };
            state.proposals.update(|p| p.insert(0, new_proposal));
            set_title.set(String::new());
            set_description.set(String::new());
            set_submitting.set(false);
            state.show_create_form.set(false);
        });
    };

    view! {
        <form class="proposal-form" on:submit=on_submit>
            <input type="text" class="form-input" placeholder="Proposal title..."
                prop:value=move || title.get()
                on:input=move |ev| set_title.set(event_target_value(&ev)) />
            <textarea class="form-textarea" placeholder="Describe what you're proposing and why..." rows="4"
                prop:value=move || description.get()
                on:input=move |ev| set_description.set(event_target_value(&ev)) />
            <div class="form-row">
                <select class="form-select" on:change=move |ev| set_proposal_type.set(event_target_value(&ev))>
                    <option value="Standard">"Standard (7 days)"</option>
                    <option value="Emergency">"Emergency (24 hours)"</option>
                    <option value="Constitutional">"Constitutional (30 days)"</option>
                    <option value="Funding">"Funding Request (14 days)"</option>
                    <option value="Parameter">"Parameter Change (14 days)"</option>
                </select>
                <button type="submit" class="form-submit" disabled=move || submitting.get()>
                    {move || if submitting.get() { "Submitting..." } else { "Submit Proposal" }}
                </button>
            </div>
            {move || error_msg.get().map(|e| view! { <p class="form-error">{e}</p> })}
        </form>
    }
}
