// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Applications page — track job application lifecycle.
//!
//! When connected to the Holochain conductor the Withdraw button calls
//! `applications_coordinator::withdraw_application`.  In mock mode the
//! withdrawal happens client-side only.

use leptos::prelude::*;
use mycelix_leptos_core::{
    holochain_provider::use_holochain,
    toasts::{use_toasts, ToastKind},
};

// ---------------------------------------------------------------------------
// Mock data types
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
enum AppStatus {
    Draft,
    Submitted,
    UnderReview,
    Interview,
    Offered,
}

impl AppStatus {
    fn label(&self) -> &'static str {
        match self {
            Self::Draft => "Draft",
            Self::Submitted => "Submitted",
            Self::UnderReview => "Under Review",
            Self::Interview => "Interview",
            Self::Offered => "Offered",
        }
    }

    fn css_class(&self) -> &'static str {
        match self {
            Self::Draft => "badge-draft",
            Self::Submitted => "badge-submitted",
            Self::UnderReview => "badge-review",
            Self::Interview => "badge-interview",
            Self::Offered => "badge-offered",
        }
    }

    /// Whether the application can still be withdrawn.
    fn withdrawable(&self) -> bool {
        matches!(self, Self::Submitted | Self::UnderReview | Self::Interview)
    }
}

#[derive(Clone)]
struct Application {
    id: u32,
    job_title: String,
    organization: String,
    status: AppStatus,
    submitted_date: String,
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

fn mock_applications() -> Vec<Application> {
    vec![
        Application {
            id: 1,
            job_title: "Rust / Holochain Engineer".into(),
            organization: "Luminous Dynamics".into(),
            status: AppStatus::Submitted,
            submitted_date: "2026-03-28".into(),
        },
        Application {
            id: 2,
            job_title: "Data Science Lead".into(),
            organization: "Mycelix Foundation".into(),
            status: AppStatus::UnderReview,
            submitted_date: "2026-03-15".into(),
        },
    ]
}

/// Count how many applications are in each pipeline stage.
fn stage_counts(apps: &[Application]) -> [usize; 5] {
    let mut counts = [0usize; 5];
    for app in apps {
        match app.status {
            AppStatus::Draft => counts[0] += 1,
            AppStatus::Submitted => counts[1] += 1,
            AppStatus::UnderReview => counts[2] += 1,
            AppStatus::Interview => counts[3] += 1,
            AppStatus::Offered => counts[4] += 1,
        }
    }
    counts
}

/// Input for the withdraw zome call.
#[derive(Clone, Debug, serde::Serialize)]
struct WithdrawInput {
    application_id: u32,
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn ApplicationsPage() -> impl IntoView {
    let hc = use_holochain();
    let toasts = use_toasts();

    let (applications, set_applications) = signal(mock_applications());

    // Derive pipeline counts reactively.
    let counts = Memo::new(move |_| stage_counts(&applications.get()));

    let stage_names = ["Draft", "Submitted", "Under Review", "Interview", "Offered"];

    // Clone hc for each closure that captures it.
    let hc_for_status = hc.clone();
    let hc_for_list = hc.clone();

    view! {
        <div class="page applications-page">
            <h1>"My Applications"</h1>

            // Conductor status indicator
            <div class="conductor-status">
                {move || if hc_for_status.is_mock() {
                    view! { <span class="status-indicator mock">"Local demo -- withdrawals are client-side only"</span> }.into_any()
                } else {
                    view! { <span class="status-indicator connected">"Connected to conductor"</span> }.into_any()
                }}
            </div>

            // Pipeline stages with live counts
            <div class="app-pipeline" role="list" aria-label="Application pipeline">
                {stage_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        view! {
                            <div class="pipeline-stage" role="listitem">
                                <h3>{*name}</h3>
                                <p class="stage-count">{move || counts.get()[i].to_string()}</p>
                            </div>
                        }
                    })
                    .collect::<Vec<_>>()}
            </div>

            // Application cards
            <div class="applications-list">
                {move || {
                    let apps = applications.get();
                    if apps.is_empty() {
                        view! {
                            <p class="empty-state">"No applications yet. Browse jobs to apply."</p>
                        }
                        .into_any()
                    } else {
                        let hc = hc_for_list.clone();
                        let toasts = toasts.clone();
                        view! {
                            <div class="app-cards">
                                {apps
                                    .iter()
                                    .map(|app| {
                                        let app_id = app.id;
                                        let badge_cls = format!("status-badge {}", app.status.css_class());
                                        let can_withdraw = app.status.withdrawable();
                                        let hc = hc.clone();
                                        let toasts = toasts.clone();
                                        view! {
                                            <div class="application-card">
                                                <div class="app-card-header">
                                                    <h3>{app.job_title.clone()}</h3>
                                                    <span class=badge_cls>{app.status.label()}</span>
                                                </div>
                                                <p class="app-org">{app.organization.clone()}</p>
                                                <p class="app-date">
                                                    {format!("Submitted {}", app.submitted_date)}
                                                </p>
                                                {if can_withdraw {
                                                    let set_apps = set_applications;
                                                    let hc = hc.clone();
                                                    let toasts = toasts.clone();
                                                    Some(
                                                        view! {
                                                            <button
                                                                class="btn-danger btn-sm"
                                                                on:click=move |_| {
                                                                    let hc = hc.clone();
                                                                    let toasts = toasts.clone();
                                                                    // Always remove locally
                                                                    set_apps
                                                                        .update(|apps| {
                                                                            apps.retain(|a| a.id != app_id);
                                                                        });
                                                                    // If conductor connected, also withdraw on-chain
                                                                    if !hc.is_mock() {
                                                                        let input = WithdrawInput { application_id: app_id };
                                                                        wasm_bindgen_futures::spawn_local(async move {
                                                                            match hc
                                                                                .call_zome_default::<WithdrawInput, ()>(
                                                                                    "applications_coordinator",
                                                                                    "withdraw_application",
                                                                                    &input,
                                                                                )
                                                                                .await
                                                                            {
                                                                                Ok(_) => toasts.success("Application withdrawn from the network."),
                                                                                Err(e) => toasts.error(format!("Conductor withdrawal failed: {e}")),
                                                                            }
                                                                        });
                                                                    } else {
                                                                        toasts.push(
                                                                            "Application withdrawn locally. Connect conductor to sync.",
                                                                            ToastKind::Info,
                                                                        );
                                                                    }
                                                                }
                                                            >
                                                                "Withdraw"
                                                            </button>
                                                        },
                                                    )
                                                } else {
                                                    None
                                                }}
                                            </div>
                                        }
                                    })
                                    .collect::<Vec<_>>()}
                            </div>
                        }
                        .into_any()
                    }
                }}
            </div>
        </div>
    }
}
