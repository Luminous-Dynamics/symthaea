// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance domain pages — proposals, voting, councils.

use leptos::prelude::*;
use domain_governance::types::*;
use portal_viz::{VoteTallyBar, BarChart, bar_chart::Bar};

use crate::identity::{ConductorStatus, PortalIdentity};

/// Mock proposals for demo mode.
fn mock_proposals() -> Vec<ProposalSummary> {
    vec![
        ProposalSummary {
            proposal: Proposal {
                id: "MIP-042".into(),
                title: "Establish community solar garden on Block 7".into(),
                description: "Convert the abandoned lot on Block 7 into a shared solar installation. Revenue shared via TEND.".into(),
                proposal_type: ProposalType::Standard,
                author: "did:mycelix:uhCAkSolar".into(),
                status: ProposalStatus::Active,
                actions: "[]".into(),
                discussion_url: None,
                voting_starts: 1711900800,
                voting_ends: 1712505600,
                created: 1711814400,
                updated: 1711900800,
                version: 1,
            },
            tally: VoteTally {
                votes_for: 34,
                votes_against: 8,
                votes_abstain: 3,
                total_weight_for: 28.5,
                total_weight_against: 5.2,
                quorum_reached: true,
                approved: false,
            },
            my_vote: Some(VoteChoice::For),
        },
        ProposalSummary {
            proposal: Proposal {
                id: "MIP-041".into(),
                title: "Increase care worker TEND allocation by 15%".into(),
                description: "Care workers are undercompensated relative to their contribution. This proposal increases their base TEND allocation.".into(),
                proposal_type: ProposalType::Funding,
                author: "did:mycelix:uhCAkCare".into(),
                status: ProposalStatus::Active,
                actions: "[]".into(),
                discussion_url: None,
                voting_starts: 1711814400,
                voting_ends: 1712419200,
                created: 1711728000,
                updated: 1711814400,
                version: 1,
            },
            tally: VoteTally {
                votes_for: 52,
                votes_against: 4,
                votes_abstain: 7,
                total_weight_for: 45.1,
                total_weight_against: 2.8,
                quorum_reached: true,
                approved: false,
            },
            my_vote: None,
        },
        ProposalSummary {
            proposal: Proposal {
                id: "MIP-039".into(),
                title: "Adopt water stewardship protocol for Commons watershed".into(),
                description: "Implement tiered water allocation with emergency priority and seasonal adjustment.".into(),
                proposal_type: ProposalType::Constitutional,
                author: "did:mycelix:uhCAkWater".into(),
                status: ProposalStatus::Executed,
                actions: "[]".into(),
                discussion_url: None,
                voting_starts: 1710604800,
                voting_ends: 1713196800,
                created: 1710518400,
                updated: 1713283200,
                version: 2,
            },
            tally: VoteTally {
                votes_for: 89,
                votes_against: 3,
                votes_abstain: 1,
                total_weight_for: 78.2,
                total_weight_against: 1.5,
                quorum_reached: true,
                approved: true,
            },
            my_vote: Some(VoteChoice::For),
        },
    ]
}

/// Governance overview — proposals list with vote tallies.
#[component]
pub fn GovernanceOverview() -> impl IntoView {
    let identity = use_context::<PortalIdentity>().expect("PortalIdentity");
    let (selected, set_selected) = signal(None::<String>);

    // Try real conductor, fall back to mock data
    let proposals_resource = LocalResource::new(move || {
        let identity = identity.clone();
        async move {
            if identity.conductor_status.get() == ConductorStatus::Connected {
                // Try real zome call
                match identity.call_zome::<(), Vec<ProposalSummary>>(
                    "governance", "proposals", "list_active_proposals", &()
                ).await {
                    Ok(proposals) => return proposals,
                    Err(e) => {
                        web_sys::console::log_1(
                            &format!("[Governance] Zome call failed, using mock: {e}").into()
                        );
                    }
                }
            }
            mock_proposals()
        }
    });

    let proposals = move || {
        proposals_resource.get().unwrap_or_else(|| mock_proposals())
    };

    let status_color = |s: &ProposalStatus| -> &'static str {
        match s {
            ProposalStatus::Active => "#22c55e",
            ProposalStatus::Draft => "#6b7280",
            ProposalStatus::Approved | ProposalStatus::Executed => "#e8c547",
            ProposalStatus::Rejected | ProposalStatus::Failed => "#ef4444",
            _ => "#a78bfa",
        }
    };

    view! {
        <div class="governance-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Proposals"</button>
                <button class="domain-nav-btn">"Councils"</button>
                <button class="domain-nav-btn">"Constitution"</button>
                <button class="domain-nav-btn">"Budget"</button>
            </div>

            <div class="proposals-list">
                {move || { proposals().iter().map(|ps| {
                    let id = ps.proposal.id.clone();
                    let title = ps.proposal.title.clone();
                    let ptype = ps.proposal.proposal_type.label().to_string();
                    let status = ps.proposal.status.label().to_string();
                    let s_color = status_color(&ps.proposal.status).to_string();
                    let v_for = ps.tally.votes_for;
                    let v_against = ps.tally.votes_against;
                    let v_abstain = ps.tally.votes_abstain;
                    let my_vote = ps.my_vote.clone();
                    let is_selected = {
                        let id = id.clone();
                        move || selected.get().as_deref() == Some(&id)
                    };
                    let id_click = id.clone();
                    let id_display = id.clone();

                    view! {
                        <div
                            class="proposal-card"
                            class:selected=is_selected
                            on:click=move |_| {
                                if selected.get().as_deref() == Some(&id_click) {
                                    set_selected.set(None);
                                } else {
                                    set_selected.set(Some(id_click.clone()));
                                }
                            }
                            style=format!("border-left: 3px solid {s_color};")
                        >
                            <div class="proposal-header">
                                <span class="proposal-id">{id_display}</span>
                                <span class="proposal-type-badge" style=format!("color: {s_color}")>{ptype}</span>
                                <span class="proposal-status" style=format!("color: {s_color}")>{status}</span>
                            </div>
                            <h3 class="proposal-title">{title}</h3>
                            <VoteTallyBar votes_for=v_for votes_against=v_against votes_abstain=v_abstain />
                            {my_vote.map(|v| {
                                let label = match v {
                                    VoteChoice::For => "You voted For",
                                    VoteChoice::Against => "You voted Against",
                                    VoteChoice::Abstain => "You abstained",
                                };
                                view! { <span class="my-vote-badge">{label}</span> }
                            })}
                        </div>
                    }
                }).collect::<Vec<_>>() }}
            </div>

            // Create proposal form
            <CreateProposalForm />

            // Participation chart
            <div class="governance-chart">
                <h3 class="chart-title">"Participation by Type"</h3>
                <BarChart
                    data=vec![
                        Bar { label: "Standard".into(), value: 42.0, color: "#7C3AED".into() },
                        Bar { label: "Funding".into(), value: 28.0, color: "#FBBF24".into() },
                        Bar { label: "Constitutional".into(), value: 8.0, color: "#22c55e".into() },
                        Bar { label: "Emergency".into(), value: 3.0, color: "#ef4444".into() },
                        Bar { label: "Parameter".into(), value: 12.0, color: "#6b7280".into() },
                    ]
                    width=350.0
                    height=180.0
                />
            </div>
        </div>
    }
}

/// Form to create a new governance proposal.
#[component]
fn CreateProposalForm() -> impl IntoView {
    let identity = use_context::<PortalIdentity>().expect("PortalIdentity");
    let (title, set_title) = signal(String::new());
    let (description, set_description) = signal(String::new());
    let (proposal_type, set_proposal_type) = signal("Standard".to_string());
    let (submitted, set_submitted) = signal(false);
    let (error_msg, set_error) = signal(None::<String>);
    let (expanded, set_expanded) = signal(false);

    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        let t = title.get();
        let d = description.get();
        if t.trim().is_empty() || d.trim().is_empty() {
            set_error.set(Some("Title and description required".into()));
            return;
        }
        set_error.set(None);

        let identity = identity.clone();
        let pt = proposal_type.get();
        wasm_bindgen_futures::spawn_local(async move {
            let input = serde_json::json!({
                "title": t.trim(),
                "description": d.trim(),
                "proposal_type": pt,
            });
            match identity.call_zome::<serde_json::Value, serde_json::Value>(
                "governance", "proposals", "create_proposal", &input
            ).await {
                Ok(_) => {
                    set_submitted.set(true);
                    set_title.set(String::new());
                    set_description.set(String::new());
                }
                Err(e) => {
                    if e.contains("Mock mode") {
                        set_submitted.set(true);
                        set_title.set(String::new());
                        set_description.set(String::new());
                    } else {
                        set_error.set(Some(e));
                    }
                }
            }
        });
    };

    view! {
        <div class="create-proposal-section">
            <button
                class="domain-nav-btn"
                on:click=move |_| set_expanded.update(|e| *e = !*e)
            >
                {move || if expanded.get() { "- Close" } else { "+ New Proposal" }}
            </button>

            <Show when=move || expanded.get()>
                <form class="proposal-form" on:submit=on_submit>
                    <input
                        type="text"
                        class="form-input"
                        placeholder="Proposal title..."
                        prop:value=move || title.get()
                        on:input=move |ev| set_title.set(event_target_value(&ev))
                    />
                    <textarea
                        class="form-textarea"
                        placeholder="Describe what you're proposing and why..."
                        rows="4"
                        prop:value=move || description.get()
                        on:input=move |ev| set_description.set(event_target_value(&ev))
                    />
                    <div class="form-row">
                        <select
                            class="form-select"
                            on:change=move |ev| set_proposal_type.set(event_target_value(&ev))
                        >
                            <option value="Standard">"Standard (7 days)"</option>
                            <option value="Emergency">"Emergency (24 hours)"</option>
                            <option value="Constitutional">"Constitutional (30 days)"</option>
                            <option value="Funding">"Funding Request (14 days)"</option>
                            <option value="Parameter">"Parameter Change (14 days)"</option>
                        </select>
                        <button type="submit" class="form-submit">"Submit Proposal"</button>
                    </div>
                    {move || error_msg.get().map(|e| view! {
                        <p class="form-error">{e}</p>
                    })}
                    <Show when=move || submitted.get()>
                        <p class="form-success">"Proposal submitted!"</p>
                    </Show>
                </form>
            </Show>
        </div>
    }
}
