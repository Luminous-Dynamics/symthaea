// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Employer Dashboard — stake on pathways, track apprentices, verify credentials.
//!
//! The employer experience:
//! 1. Create an apprenticeship stake (SAP bounty on a curriculum pathway)
//! 2. See cohort mastery distribution (FL intelligence, privacy-preserving)
//! 3. Track applicant pipeline (Draft → Submitted → Interview → Offered)
//! 4. Verify candidate credentials (PoL score + Living Credential vitality)
//! 5. Interview graduates who meet the PoL threshold

use leptos::prelude::*;

use crate::persistence;

const STAKES_KEY: &str = "craft_employer_stakes";

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct LocalStake {
    pathway: String,
    organization: String,
    stake_sap: u32,
    max_apprentices: u16,
    current_apprentices: u16,
    required_pol_permille: u16,
    interview_guarantee: bool,
    required_skills: Vec<String>,
    status: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct EmployerState {
    stakes: Vec<LocalStake>,
}

impl EmployerState {
    fn load() -> Self {
        persistence::load::<EmployerState>(STAKES_KEY).unwrap_or_default()
    }
    fn save(&self) {
        persistence::save(STAKES_KEY, self);
    }
}

#[component]
pub fn EmployerDashboard() -> impl IntoView {
    let (state, set_state) = signal(EmployerState::load());
    let (new_pathway, set_new_pathway) = signal(String::new());
    let (new_sap, set_new_sap) = signal(1000u32);
    let (new_slots, set_new_slots) = signal(5u16);
    let (new_pol, set_new_pol) = signal(700u16);

    // Persist on change
    Effect::new(move |_| {
        state.get().save();
    });

    view! {
        <div class="page employer-page">
            <h1>"Employer Dashboard"</h1>
            <p class="text-secondary">"Invest in talent pipelines. Fund learning journeys. Hire graduates with cryptographic proof."</p>

            // Active Stakes
            <div class="dash-section">
                <h2>"Active Apprenticeship Stakes"</h2>
                {move || {
                    let stakes = state.get().stakes;
                    if stakes.is_empty() {
                        view! {
                            <p class="empty-state">"No active stakes. Create one below to start funding a talent pipeline."</p>
                        }.into_any()
                    } else {
                        view! {
                            <div class="stakes-grid">
                                {stakes.iter().map(|s| {
                                    let filled_pct = if s.max_apprentices > 0 {
                                        (s.current_apprentices as u32 * 100 / s.max_apprentices as u32) as u16
                                    } else { 0 };
                                    view! {
                                        <div class="stake-card">
                                            <h3>{s.pathway.clone()}</h3>
                                            <div class="stake-meta">
                                                <span class="stake-sap">{s.stake_sap}" SAP staked"</span>
                                                <span class="stake-slots">{s.current_apprentices}"/"
                                                    {s.max_apprentices}" slots filled"</span>
                                            </div>
                                            <div class="progress-bar-container"
                                                role="progressbar"
                                                aria-valuenow=filled_pct
                                                aria-valuemin="0"
                                                aria-valuemax="100">
                                                <div class="progress-bar"
                                                    style=format!("width: {}%", filled_pct)>
                                                </div>
                                            </div>
                                            <div class="stake-details">
                                                <span>"Min PoL: "{s.required_pol_permille / 10}"%"</span>
                                                {if s.interview_guarantee {
                                                    view! { <span class="guarantee-badge">"Guaranteed Interview"</span> }.into_any()
                                                } else {
                                                    view! { <span></span> }.into_any()
                                                }}
                                            </div>
                                            <div class="stake-skills">
                                                {s.required_skills.iter().map(|skill| {
                                                    view! { <span class="skill-tag">{skill.clone()}</span> }
                                                }).collect_view()}
                                            </div>
                                        </div>
                                    }
                                }).collect_view()}
                            </div>
                        }.into_any()
                    }
                }}
            </div>

            // Create New Stake
            <div class="dash-section create-stake">
                <h2>"Create Apprenticeship Stake"</h2>
                <p class="text-secondary">"Fund a learning pathway. Graduates who meet your PoL threshold get guaranteed interviews."</p>

                <div class="form-row">
                    <div class="form-group">
                        <label>"Pathway"</label>
                        <select
                            prop:value=move || new_pathway.get()
                            on:change=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlSelectElement>().unwrap().value();
                                set_new_pathway.set(val);
                            }
                        >
                            <option value="">"Select a pathway..."</option>
                            <option value="Rust Development">"Rust Development"</option>
                            <option value="Data Science">"Data Science"</option>
                            <option value="Cybersecurity">"Cybersecurity"</option>
                            <option value="Holochain Development">"Holochain Development"</option>
                            <option value="CAPS Mathematics">"CAPS Mathematics (SA)"</option>
                            <option value="Financial Literacy">"Financial Literacy"</option>
                            <option value="Decentralized Systems">"Decentralized Systems"</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>"SAP Stake"</label>
                        <input type="number" min="100" max="10000" step="100"
                            prop:value=move || new_sap.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_new_sap.set(val.parse().unwrap_or(1000));
                            }
                        />
                    </div>

                    <div class="form-group">
                        <label>"Max Apprentices"</label>
                        <input type="number" min="1" max="50"
                            prop:value=move || new_slots.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_new_slots.set(val.parse().unwrap_or(5));
                            }
                        />
                    </div>

                    <div class="form-group">
                        <label>"Min PoL Score (%)"</label>
                        <input type="number" min="50" max="100"
                            prop:value=move || new_pol.get() / 10
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val: u16 = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse().unwrap_or(70);
                                set_new_pol.set(val * 10);
                            }
                        />
                    </div>
                </div>

                <button class="btn-primary"
                    on:click=move |_| {
                        let pathway = new_pathway.get();
                        if pathway.is_empty() { return; }
                        set_state.update(|s| {
                            s.stakes.push(LocalStake {
                                pathway: pathway.clone(),
                                organization: "Your Organization".to_string(),
                                stake_sap: new_sap.get(),
                                max_apprentices: new_slots.get(),
                                current_apprentices: 0,
                                required_pol_permille: new_pol.get(),
                                interview_guarantee: true,
                                required_skills: vec![pathway.to_lowercase()],
                                status: "Active".to_string(),
                            });
                        });
                        set_new_pathway.set(String::new());
                    }
                >"Create Stake"</button>
            </div>

            // Applicant Pipeline
            <div class="dash-section">
                <h2>"Applicant Pipeline"</h2>
                <div class="pipeline" role="list" aria-label="Hiring pipeline">
                    <div class="pipeline-stage" role="listitem">
                        <div class="stage-label">"Applied"</div>
                        <div class="stage-count">"0"</div>
                    </div>
                    <div class="pipeline-arrow">"→"</div>
                    <div class="pipeline-stage" role="listitem">
                        <div class="stage-label">"Under Review"</div>
                        <div class="stage-count">"0"</div>
                    </div>
                    <div class="pipeline-arrow">"→"</div>
                    <div class="pipeline-stage" role="listitem">
                        <div class="stage-label">"Interview"</div>
                        <div class="stage-count">"0"</div>
                    </div>
                    <div class="pipeline-arrow">"→"</div>
                    <div class="pipeline-stage" role="listitem">
                        <div class="stage-label">"Offered"</div>
                        <div class="stage-count">"0"</div>
                    </div>
                    <div class="pipeline-arrow">"→"</div>
                    <div class="pipeline-stage highlight" role="listitem">
                        <div class="stage-label">"Hired"</div>
                        <div class="stage-count">"0"</div>
                    </div>
                </div>
                <p class="text-secondary" style="margin-top: 1rem">
                    "Pipeline populates when apprentices graduate and apply. "
                    "Each applicant shows their PoL score, credential vitality, and peer endorsements."
                </p>
            </div>

            // Cohort Intelligence
            <div class="dash-section">
                <h2>"Cohort Intelligence"</h2>
                <p class="text-secondary">
                    "Privacy-preserving mastery distribution from Federated Learning. "
                    "You see aggregate statistics, not individual learner data."
                </p>
                <div class="intelligence-placeholder">
                    <p>"Connect to the Holochain conductor to see live cohort mastery data."</p>
                    <p class="text-secondary">"The FL pipeline aggregates mastery buckets (None/Low/Medium/High) "
                        "without exposing individual scores."</p>
                </div>
            </div>
        </div>
    }
}
