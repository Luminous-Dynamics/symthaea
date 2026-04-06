// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network page — connections, recommendations, and endorsements.
//!
//! Uses `CraftCtx::connections` from the conductor when available and falls
//! back to mock data otherwise.  Action buttons (Send Request, Write
//! Recommendation) notify via toast and are disabled in mock mode.

use leptos::prelude::*;
use mycelix_leptos_core::{
    holochain_provider::use_holochain,
    toasts::{use_toasts, ToastKind},
};

use crate::context::use_craft;

// ---------------------------------------------------------------------------
// Mock data types
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Connection {
    display_name: String,
    headline: String,
    connected_since: String,
}

#[derive(Clone)]
struct PendingRequest {
    display_name: String,
    headline: String,
    received: String,
}

#[derive(Clone)]
struct Recommendation {
    from_name: String,
    text: String,
    date: String,
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

fn mock_connections() -> Vec<Connection> {
    vec![
        Connection {
            display_name: "Nia Modise".into(),
            headline: "Holochain hApp Developer".into(),
            connected_since: "2026-01-15".into(),
        },
        Connection {
            display_name: "Erik Lindqvist".into(),
            headline: "NixOS Infrastructure Engineer".into(),
            connected_since: "2026-02-03".into(),
        },
        Connection {
            display_name: "Amara Osei".into(),
            headline: "Data Science & ML Researcher".into(),
            connected_since: "2026-03-22".into(),
        },
    ]
}

fn mock_pending() -> Vec<PendingRequest> {
    vec![PendingRequest {
        display_name: "Jun Tanaka".into(),
        headline: "Cybersecurity Consultant".into(),
        received: "2026-04-02".into(),
    }]
}

fn mock_recommendations() -> Vec<Recommendation> {
    vec![Recommendation {
        from_name: "Nia Modise".into(),
        text: "Exceptional systems thinker. Built a full Holochain cluster in two weeks and documented every design decision. Highly recommended for distributed systems work.".into(),
        date: "2026-03-10".into(),
    }]
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn NetworkPage() -> impl IntoView {
    let craft = use_craft();
    let hc = use_holochain();
    let toasts = use_toasts();

    let connections = mock_connections();
    let pending = mock_pending();
    let recommendations = mock_recommendations();

    // Use conductor connection count when available, else mock count.
    let conn_count = Memo::new(move |_| {
        let conductor_conns = craft.connections.get();
        if conductor_conns.is_empty() {
            3usize // mock count
        } else {
            conductor_conns.len()
        }
    });
    let pending_count = pending.len();
    let rec_count = recommendations.len();

    // Send connection request handler
    let on_send_request = {
        let hc = hc.clone();
        let toasts = toasts.clone();
        move |_| {
            let hc = hc.clone();
            let toasts = toasts.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if hc.is_mock() {
                    toasts.push(
                        "Connect to the Holochain conductor to send connection requests.",
                        ToastKind::Info,
                    );
                    return;
                }
                match hc
                    .call_zome_default::<(), ()>(
                        "connection_graph_coordinator",
                        "send_connection_request",
                        &(),
                    )
                    .await
                {
                    Ok(_) => toasts.success("Connection request sent!"),
                    Err(e) => toasts.error(format!("Failed to send request: {e}")),
                }
            });
        }
    };

    // Write recommendation handler
    let on_write_recommendation = {
        let hc = hc.clone();
        let toasts = toasts.clone();
        move |_| {
            let hc = hc.clone();
            let toasts = toasts.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if hc.is_mock() {
                    toasts.push(
                        "Connect to the Holochain conductor to write recommendations.",
                        ToastKind::Info,
                    );
                    return;
                }
                match hc
                    .call_zome_default::<(), ()>(
                        "connection_graph_coordinator",
                        "create_recommendation",
                        &(),
                    )
                    .await
                {
                    Ok(_) => toasts.success("Recommendation published!"),
                    Err(e) => toasts.error(format!("Failed to publish: {e}")),
                }
            });
        }
    };

    view! {
        <div class="page network-page">
            <h1>"Craft Network"</h1>

            // Summary stats
            <div class="stat-cards">
                <div class="stat-card">
                    <p class="stat-value">{move || conn_count.get()}</p>
                    <p class="stat-label">"Connections"</p>
                </div>
                <div class="stat-card">
                    <p class="stat-value">{pending_count}</p>
                    <p class="stat-label">"Pending"</p>
                </div>
                <div class="stat-card">
                    <p class="stat-value">{rec_count}</p>
                    <p class="stat-label">"Recommendations"</p>
                </div>
                <div class="stat-card">
                    <p class="stat-value">{move || craft.endorsement_count.get()}</p>
                    <p class="stat-label">"Endorsements"</p>
                </div>
            </div>

            // Action buttons
            <div class="network-actions">
                <button
                    class="btn-primary"
                    on:click=on_send_request
                    disabled=move || hc.is_mock()
                    title=move || if hc.is_mock() { "Connect to conductor first" } else { "Send a connection request" }
                >
                    "Send Request"
                </button>
                <button
                    class="btn-secondary"
                    on:click=on_write_recommendation
                    disabled=move || hc.is_mock()
                    title=move || if hc.is_mock() { "Connect to conductor first" } else { "Write a recommendation" }
                >
                    "Write Recommendation"
                </button>
                {move || hc.is_mock().then(|| view! {
                    <span class="status-indicator mock">"Local demo"</span>
                })}
                {move || (!hc.is_mock()).then(|| view! {
                    <span class="status-indicator connected">"Connected to conductor"</span>
                })}
            </div>

            <div class="network-grid">
                // Connections
                <div class="network-section">
                    <h3>"Connections"</h3>
                    <div class="connections-list">
                        {connections
                            .into_iter()
                            .map(|c| {
                                view! {
                                    <div class="connection-card">
                                        <div class="connection-info">
                                            <p class="connection-name">{c.display_name}</p>
                                            <p class="connection-headline">{c.headline}</p>
                                        </div>
                                        <p class="connection-date">
                                            {format!("Connected {}", c.connected_since)}
                                        </p>
                                    </div>
                                }
                            })
                            .collect::<Vec<_>>()}
                    </div>
                </div>

                // Pending requests
                <div class="network-section">
                    <h3>"Pending Requests"</h3>
                    <p class="text-secondary">"Incoming connection requests."</p>
                    <div class="requests-list">
                        {if pending.is_empty() {
                            view! { <p class="empty-state">"No pending requests."</p> }.into_any()
                        } else {
                            view! {
                                <div>
                                    {pending
                                        .into_iter()
                                        .map(|r| {
                                            view! {
                                                <div class="request-card">
                                                    <div class="request-info">
                                                        <p class="connection-name">
                                                            {r.display_name}
                                                        </p>
                                                        <p class="connection-headline">
                                                            {r.headline}
                                                        </p>
                                                        <p class="connection-date">
                                                            {format!("Received {}", r.received)}
                                                        </p>
                                                    </div>
                                                    <div class="request-actions">
                                                        <button class="btn-primary btn-sm">
                                                            "Accept"
                                                        </button>
                                                        <button class="btn-secondary btn-sm">
                                                            "Decline"
                                                        </button>
                                                    </div>
                                                </div>
                                            }
                                        })
                                        .collect::<Vec<_>>()}
                                </div>
                            }
                            .into_any()
                        }}
                    </div>
                </div>

                // Recommendations
                <div class="network-section">
                    <h3>"Recommendations"</h3>
                    <p class="text-secondary">"Recommendations from your network."</p>
                    <div class="recommendations-list">
                        {if recommendations.is_empty() {
                            view! { <p class="empty-state">"No recommendations yet."</p> }
                                .into_any()
                        } else {
                            view! {
                                <div>
                                    {recommendations
                                        .into_iter()
                                        .map(|r| {
                                            view! {
                                                <div class="recommendation-card">
                                                    <blockquote>{r.text}</blockquote>
                                                    <p class="rec-attribution">
                                                        {format!("-- {}, {}", r.from_name, r.date)}
                                                    </p>
                                                </div>
                                            }
                                        })
                                        .collect::<Vec<_>>()}
                                </div>
                            }
                            .into_any()
                        }}
                    </div>
                </div>

                // Endorsements (still empty — data comes from Holochain)
                <div class="network-section">
                    <h3>"Skill Endorsements"</h3>
                    <p class="text-secondary">"Peer attestations of your skills."</p>
                    <div class="endorsements-list">
                        <p class="empty-state">
                            "No endorsements yet. Share your agent pubkey with peers."
                        </p>
                    </div>
                </div>
            </div>
        </div>
    }
}
