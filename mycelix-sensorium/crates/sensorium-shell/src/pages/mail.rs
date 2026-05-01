// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail domain page — inbox summary from Mail REST API.
//! Connects to the Mail React app's backend at localhost:3001.

use domain_mail::types::*;
use leptos::prelude::*;
use mycelix_leptos_core::{
    ActivityFeed, ActivityFeedItem, AvailabilityState, AvailabilityStateKind, FreshnessBadge,
    FreshnessLevel,
};
use sensorium_viz::{bar_chart::Bar, BarChart};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

const MAIL_API: &str = "http://localhost:3001";
const PULSE_SUMMARY_STORAGE_KEY: &str = "mycelix:pulse:inbox_summary:v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MailPageMode {
    Api,
    LocalBridge,
    Mock,
}

/// Fetch inbox summary from Mail REST API.
async fn fetch_inbox() -> Result<InboxSummary, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_value =
        JsFuture::from(window.fetch_with_str(&format!("{MAIL_API}/api/inbox/summary")))
            .await
            .map_err(|e| format!("fetch: {:?}", e))?;
    let resp: web_sys::Response = resp_value.dyn_into().map_err(|_| "not a Response")?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let json = JsFuture::from(resp.json().map_err(|_| "json() failed")?)
        .await
        .map_err(|e| format!("json: {:?}", e))?;
    serde_wasm_bindgen::from_value(json).map_err(|e| format!("deserialize: {e}"))
}

fn load_local_inbox() -> Option<InboxSummary> {
    web_sys::window()
        .and_then(|window| window.local_storage().ok().flatten())
        .and_then(|storage| storage.get_item(PULSE_SUMMARY_STORAGE_KEY).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
}

fn mock_inbox() -> InboxSummary {
    InboxSummary {
        unread_count: 12,
        total_count: 247,
        urgent_count: 1,
        high_priority_count: 4,
        queued_actions: 0,
        recent_threads: vec![
            ThreadSummary {
                id: "t-001".into(),
                subject: "Re: Council meeting agenda for Thursday".into(),
                from_name: "Elena".into(),
                from_email: "elena@mycelix.local".into(),
                preview: "I've added the water allocation item to the agenda.".into(),
                trust_tier: TrustTier::High,
                timestamp: 1711954920,
                is_read: false,
            },
            ThreadSummary {
                id: "t-002".into(),
                subject: "FL round 47 results \u{2014} 94.2% accuracy".into(),
                from_name: "Coordinator".into(),
                from_email: "fl@mycelix.local".into(),
                preview: "Your gradient contribution improved the model by 0.3%.".into(),
                trust_tier: TrustTier::High,
                timestamp: 1711949700,
                is_read: false,
            },
            ThreadSummary {
                id: "t-003".into(),
                subject: "Solar garden construction timeline".into(),
                from_name: "Block 7 Committee".into(),
                from_email: "block7@mycelix.local".into(),
                preview: "Foundation work starts next Monday.".into(),
                trust_tier: TrustTier::Medium,
                timestamp: 1711868400,
                is_read: true,
            },
        ],
        trust_health: TrustHealth {
            trusted_contacts: 89,
            quarantined: 3,
            introductions_pending: 2,
            average_trust_score: 0.78,
        },
        source: InboxSummarySource::Mock,
        updated_at: Some(1_711_954_920_000_000),
    }
}

fn tier_color(tier: &TrustTier) -> &'static str {
    match tier {
        TrustTier::High => "#22c55e",
        TrustTier::Medium => "#f59e0b",
        TrustTier::Low => "#ef4444",
        TrustTier::Unknown => "#6b7280",
    }
}
fn tier_label(tier: &TrustTier) -> &'static str {
    match tier {
        TrustTier::High => "TRUSTED",
        TrustTier::Medium => "MEDIUM",
        TrustTier::Low => "LOW",
        TrustTier::Unknown => "UNKNOWN",
    }
}

#[component]
pub fn MailOverview() -> impl IntoView {
    let inbox_resource = LocalResource::new(move || async move {
        fetch_inbox()
            .await
            .map(|inbox| (inbox, MailPageMode::Api))
            .unwrap_or_else(|e| {
                if let Some(inbox) = load_local_inbox() {
                    web_sys::console::log_1(
                        &format!("[Mail] API unavailable: {e}. Using local Pulse bridge.").into(),
                    );
                    (inbox, MailPageMode::LocalBridge)
                } else {
                    web_sys::console::log_1(
                        &format!("[Mail] API unavailable: {e}. Mock mode.").into(),
                    );
                    (mock_inbox(), MailPageMode::Mock)
                }
            })
    });
    let inbox = move || {
        inbox_resource
            .get()
            .unwrap_or_else(|| (mock_inbox(), MailPageMode::Mock))
    };
    let feed_items = move || {
        inbox()
            .0
            .recent_threads
            .into_iter()
            .take(6)
            .map(|thread| ActivityFeedItem {
                id: thread.id,
                domain_label: format!("{} • {}", thread.from_name, tier_label(&thread.trust_tier)),
                description: thread.subject,
                emphasis_class: Some(
                    if !thread.is_read {
                        "activity-feed-warning"
                    } else {
                        "activity-feed-live"
                    }
                    .into(),
                ),
            })
            .collect::<Vec<_>>()
    };
    let freshness = move || {
        let (summary, mode) = inbox();
        if mode == MailPageMode::Mock {
            (FreshnessLevel::Unknown, "Mock Pulse posture".to_string())
        } else if let Some(updated_at) = summary.updated_at {
            (
                freshness_from_micros(updated_at),
                format!("Summary {}", format_relative_micros(updated_at)),
            )
        } else if let Some(latest) = summary
            .recent_threads
            .iter()
            .map(|thread| thread.timestamp)
            .max()
        {
            (
                freshness_from_secs(latest),
                format!("Threads {}", format_relative_secs(latest)),
            )
        } else {
            (FreshnessLevel::Unknown, "No Pulse threads yet".to_string())
        }
    };

    view! {
        <div class="mail-content">
            <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                {move || {
                    let (level, detail) = freshness();
                    view! { <FreshnessBadge level detail /> }
                }}
            </div>

            {move || {
                let (summary, mode) = inbox();
                if mode == MailPageMode::Mock {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Mock
                            title="Mock Pulse Posture"
                            description="The Mail backend is unavailable, so Signal Stream is currently rendering from mock inbox and trust data."
                            action={None}
                        />
                    }.into_any()
                } else if mode == MailPageMode::LocalBridge {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Degraded
                            title="Pulse Local Bridge"
                            description="The Mail REST backend is unavailable, but Signal Stream is rendering the live summary exported by the Pulse shell."
                            action={None}
                        />
                    }.into_any()
                } else if summary.total_count == 0 && summary.recent_threads.is_empty() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Empty
                            title="Pulse Connected, Inbox Empty"
                            description="Signal Stream is reachable, but there are no current threads or unread messages in this inbox posture."
                            action={None}
                        />
                    }.into_any()
                } else {
                    view! { <></> }.into_any()
                }
            }}

            <div class="governance-nav">
                <button class="domain-nav-btn active">"Inbox"</button>
                <button class="domain-nav-btn">"Trust Network"</button>
                <button class="domain-nav-btn">"Compose"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #DC2626">"UNREAD"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || inbox().0.unread_count.to_string()}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">{move || format!("of {} total", inbox().0.total_count)}</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"TRUSTED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || inbox().0.trust_health.trusted_contacts.to_string()}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Contacts"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"QUARANTINED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || inbox().0.trust_health.quarantined.to_string()}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Low-trust held"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #8b5cf6">"TRUST"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{:.0}%", inbox().0.trust_health.average_trust_score * 100.0)}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Network avg"</p>
                </div>
            </div>

            <h3 class="section-title">"Recent Threads"</h3>
            <ActivityFeed items=feed_items() />
            <div class="thought-list">
                {move || { inbox().0.recent_threads.iter().map(|t| {
                    let subject = t.subject.clone();
                    let from = t.from_name.clone();
                    let preview = t.preview.clone();
                    let color = tier_color(&t.trust_tier);
                    let label = tier_label(&t.trust_tier);
                    let unread = !t.is_read;
                    view! {
                        <div class="thought-card" style=move || if unread { "border-left: 3px solid var(--domain-glow, #FCA5A5);" } else { "" }>
                            <div class="thought-meta">
                                <span class="thought-type" style=format!("color: {color}")>{label}</span>
                                <span class="thought-domain">{from}</span>
                            </div>
                            <p class="thought-content" style=move || if unread { "font-weight: 600" } else { "" }>{subject}</p>
                            <p style="font-size: 0.75rem; color: var(--text-muted)">{preview}</p>
                        </div>
                    }
                }).collect::<Vec<_>>() }}
            </div>

            <h3 class="section-title" style="margin-top: 1rem">"Trust Distribution"</h3>
            <BarChart data=vec![
                Bar { label: "High".into(), value: 67.0, color: "#22c55e".into() },
                Bar { label: "Medium".into(), value: 18.0, color: "#f59e0b".into() },
                Bar { label: "Low".into(), value: 3.0, color: "#ef4444".into() },
                Bar { label: "Unknown".into(), value: 1.0, color: "#6b7280".into() },
            ] width=300.0 height=140.0 />
        </div>
    }
}

fn freshness_from_secs(timestamp_secs: i64) -> FreshnessLevel {
    let now_secs = (js_sys::Date::now() / 1000.0) as i64;
    let age_minutes = now_secs.saturating_sub(timestamp_secs) / 60;
    if age_minutes <= 5 {
        FreshnessLevel::Fresh
    } else if age_minutes <= 60 {
        FreshnessLevel::Aging
    } else {
        FreshnessLevel::Stale
    }
}

fn freshness_from_micros(timestamp_micros: i64) -> FreshnessLevel {
    let now_micros = (js_sys::Date::now() * 1000.0) as i64;
    let age_minutes = now_micros.saturating_sub(timestamp_micros) / 60_000_000;
    if age_minutes <= 5 {
        FreshnessLevel::Fresh
    } else if age_minutes <= 60 {
        FreshnessLevel::Aging
    } else {
        FreshnessLevel::Stale
    }
}

fn format_relative_secs(timestamp_secs: i64) -> String {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(
        (timestamp_secs * 1000) as f64,
    ));
    date.to_locale_string("en-US", &wasm_bindgen::JsValue::UNDEFINED)
        .as_string()
        .unwrap_or_else(|| "recently".into())
}

fn format_relative_micros(timestamp_micros: i64) -> String {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(
        (timestamp_micros / 1000) as f64,
    ));
    date.to_locale_string("en-US", &wasm_bindgen::JsValue::UNDEFINED)
        .as_string()
        .unwrap_or_else(|| "recently".into())
}
