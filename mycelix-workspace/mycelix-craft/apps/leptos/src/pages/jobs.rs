// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Jobs page — search, browse, and post job opportunities.
//! Job matching runs client-side using the matching engine.
//!
//! When connected to the Holochain conductor, posting jobs calls
//! `job_postings_coordinator::create_job_posting`.

use leptos::prelude::*;
use std::collections::HashSet;
use wasm_bindgen::JsCast;

use mycelix_leptos_core::{
    holochain_provider::use_holochain,
    toasts::{use_toasts, ToastKind},
};

use crate::matching::{rank_jobs, JobMatch, JobView};

/// Return a set of mock job postings spanning diverse skill domains.
fn mock_jobs() -> Vec<JobView> {
    vec![
        JobView {
            title: "Rust / Holochain Engineer".into(),
            organization: "Luminous Dynamics".into(),
            required_skills: vec!["rust".into(), "holochain".into(), "wasm".into()],
            preferred_skills: vec!["leptos".into(), "nix".into()],
            education_level: Some("Bachelor's".into()),
            remote_ok: true,
            vitality_minimum: Some(700),
        },
        JobView {
            title: "Data Science Lead".into(),
            organization: "Mycelix Foundation".into(),
            required_skills: vec!["python".into(), "data-science".into(), "machine-learning".into()],
            preferred_skills: vec!["rust".into(), "sql".into()],
            education_level: Some("Master's".into()),
            remote_ok: true,
            vitality_minimum: None,
        },
        JobView {
            title: "Cybersecurity Analyst".into(),
            organization: "NixForHumanity".into(),
            required_skills: vec!["cybersecurity".into(), "linux".into(), "networking".into()],
            preferred_skills: vec!["nix".into(), "rust".into(), "python".into()],
            education_level: None,
            remote_ok: false,
            vitality_minimum: None,
        },
        JobView {
            title: "Full-Stack Web Developer".into(),
            organization: "Resontia Labs".into(),
            required_skills: vec!["typescript".into(), "react".into(), "node".into()],
            preferred_skills: vec!["rust".into(), "wasm".into(), "holochain".into()],
            education_level: None,
            remote_ok: true,
            vitality_minimum: None,
        },
        JobView {
            title: "Distributed Systems Researcher".into(),
            organization: "Symthaea Institute".into(),
            required_skills: vec!["rust".into(), "distributed-systems".into()],
            preferred_skills: vec![
                "holochain".into(),
                "cryptography".into(),
                "python".into(),
                "data-science".into(),
            ],
            education_level: Some("PhD".into()),
            remote_ok: true,
            vitality_minimum: Some(700),
        },
    ]
}

/// Lightweight input for the `create_job_posting` zome call.
#[derive(Clone, Debug, serde::Serialize)]
struct CreateJobPostingInput {
    title: String,
    organization: String,
    required_skills: Vec<String>,
    preferred_skills: Vec<String>,
    remote_ok: bool,
}

#[component]
pub fn JobsPage() -> impl IntoView {
    let hc = use_holochain();
    let toasts = use_toasts();

    let (search_query, set_search_query) = signal(String::new());
    let (skills_input, set_skills_input) = signal(String::from("rust, holochain"));

    // Parse comma-separated skills into a HashSet, updated reactively.
    let agent_skills = Memo::new(move |_| {
        skills_input
            .get()
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect::<HashSet<String>>()
    });

    let jobs = mock_jobs();

    // Rank jobs whenever agent_skills changes.
    let matched_jobs = Memo::new(move |_| {
        let skills = agent_skills.get();
        let query = search_query.get().to_lowercase();
        // TODO: compute avg vitality from CraftCtx credentials when available
        let mut ranked = rank_jobs(&jobs, &skills, None, None, 0.0, 20);
        // Additionally filter by free-text search if present.
        if !query.is_empty() {
            ranked.retain(|m| {
                m.job.title.to_lowercase().contains(&query)
                    || m.job.organization.to_lowercase().contains(&query)
                    || m.job
                        .required_skills
                        .iter()
                        .any(|s| s.contains(&query))
                    || m.job
                        .preferred_skills
                        .iter()
                        .any(|s| s.contains(&query))
            });
        }
        ranked
    });

    // Clone hc/toasts for each closure that captures them.
    let hc_for_post = hc.clone();
    let toasts_for_post = toasts.clone();
    let hc_for_status = hc.clone();
    let hc_for_disabled = hc.clone();
    let hc_for_title = hc.clone();

    // Post-job handler
    let on_post_job = move |_| {
        let hc = hc_for_post.clone();
        let toasts = toasts_for_post.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if hc.is_mock() {
                toasts.push(
                    "Connect to the Holochain conductor to post jobs.",
                    ToastKind::Info,
                );
                return;
            }
            let input = CreateJobPostingInput {
                title: "New Job Posting".into(),
                organization: "My Organization".into(),
                required_skills: vec![],
                preferred_skills: vec![],
                remote_ok: true,
            };
            match hc
                .call_zome_default::<CreateJobPostingInput, ()>(
                    "job_postings_coordinator",
                    "create_job_posting",
                    &input,
                )
                .await
            {
                Ok(_) => toasts.success("Job posting created on the network!"),
                Err(e) => toasts.error(format!("Failed to create posting: {e}")),
            }
        });
    };

    view! {
        <div class="page jobs-page">
            <h1>"Job Opportunities"</h1>

            // Conductor status
            <div class="conductor-status">
                {move || if hc_for_status.is_mock() {
                    view! { <span class="status-indicator mock">"Local demo -- matching engine active"</span> }.into_any()
                } else {
                    view! { <span class="status-indicator connected">"Connected to conductor"</span> }.into_any()
                }}
            </div>

            // Skills input for the matching engine
            <div class="skills-input-section">
                <label for="agent-skills"><strong>"Your Skills"</strong>" (comma-separated)"</label>
                <input
                    id="agent-skills"
                    type="text"
                    placeholder="e.g. rust, holochain, python, data-science"
                    class="search-input"
                    aria-label="Your skills for job matching"
                    prop:value=move || skills_input.get()
                    on:input=move |ev| {
                        let input: web_sys::HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
                        set_skills_input.set(input.value());
                    }
                />
                <p class="text-secondary">
                    {move || {
                        let count = agent_skills.get().len();
                        format!("{count} skill{} active", if count == 1 { "" } else { "s" })
                    }}
                </p>
            </div>

            // Free-text search bar
            <div class="search-bar">
                <input
                    type="text"
                    placeholder="Filter by keyword (title, org, skill)..."
                    class="search-input"
                    aria-label="Search jobs by keyword"
                    prop:value=move || search_query.get()
                    on:input=move |ev| {
                        let input: web_sys::HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
                        set_search_query.set(input.value());
                    }
                />
            </div>

            // Matched jobs grid
            <div class="jobs-grid">
                {move || {
                    let matches = matched_jobs.get();
                    if matches.is_empty() {
                        view! {
                            <div class="placeholder-card">
                                <p>"No matching jobs found. Try adding more skills above."</p>
                            </div>
                        }
                        .into_any()
                    } else {
                        view! {
                            <div class="matched-jobs-list">
                                {matches
                                    .into_iter()
                                    .map(|m| job_card_view(m, agent_skills.get()))
                                    .collect::<Vec<_>>()}
                            </div>
                        }
                        .into_any()
                    }
                }}
            </div>

            // Post a job
            <div class="post-job-section">
                <h3>"Post a Job"</h3>
                <p class="text-secondary">
                    "Create a job posting visible to the entire craft network."
                </p>
                <button
                    class="btn-primary"
                    on:click=on_post_job
                    disabled=move || hc_for_disabled.is_mock()
                    title=move || if hc_for_title.is_mock() { "Connect to Holochain conductor to post jobs" } else { "Create a new job posting" }
                >
                    "Create Job Posting"
                </button>
            </div>
        </div>
    }
}

/// Render a single matched job card with score bar and highlighted tags.
fn job_card_view(m: JobMatch, agent_skills: HashSet<String>) -> impl IntoView {
    let pct = (m.score * 100.0).round() as u32;
    let pct_label = format!("{}% match", pct);
    let bar_width = format!("{}%", pct);
    let location = if m.job.remote_ok { "Remote" } else { "On-site" };

    view! {
        <div class="job-card">
            <div class="job-header">
                <h3>{m.job.title.clone()}</h3>
                <span class="match-badge">{pct_label}</span>
            </div>
            <p class="job-org">{m.job.organization.clone()}</p>

            // Score bar
            <div
                class="score-bar-bg"
                role="progressbar"
                aria-valuenow=pct
                aria-valuemin=0
                aria-valuemax=100
            >
                <div class="score-bar-fill" style:width=bar_width></div>
            </div>

            // Required skills
            <div class="job-skills">
                <span class="skill-label">"Required: "</span>
                {m.job
                    .required_skills
                    .iter()
                    .map(|s| {
                        let matched = agent_skills.contains(s);
                        let cls = if matched { "skill-tag matched" } else { "skill-tag" };
                        view! { <span class=cls>{s.clone()}</span> }
                    })
                    .collect::<Vec<_>>()}
            </div>

            // Preferred skills (if any)
            {if !m.job.preferred_skills.is_empty() {
                Some(
                    view! {
                        <div class="job-skills">
                            <span class="skill-label">"Preferred: "</span>
                            {m.job
                                .preferred_skills
                                .iter()
                                .map(|s| {
                                    let matched = agent_skills.contains(s);
                                    let cls = if matched {
                                        "skill-tag preferred matched"
                                    } else {
                                        "skill-tag preferred"
                                    };
                                    view! { <span class=cls>{s.clone()}</span> }
                                })
                                .collect::<Vec<_>>()}
                        </div>
                    },
                )
            } else {
                None
            }}

            <div class="job-meta">
                <span>{location}</span>
                {m.job
                    .education_level
                    .as_ref()
                    .map(|ed| {
                        view! { <span>{format!("Edu: {ed}")}</span> }
                    })}
                {if m.credential_bonus {
                    Some(view! { <span class="credential-ok">"Credential match"</span> })
                } else {
                    None
                }}
            </div>

            <button class="btn-primary" disabled title="Connect to Holochain to apply">
                "Apply"
            </button>
        </div>
    }
}
