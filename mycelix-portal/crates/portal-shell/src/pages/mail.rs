// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail domain page — inbox summary from Mail REST API.
//! Connects to the Mail React app's backend at localhost:3001.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use domain_mail::types::*;
use portal_viz::{BarChart, bar_chart::Bar};

const MAIL_API: &str = "http://localhost:3001";

/// Fetch inbox summary from Mail REST API.
async fn fetch_inbox() -> Result<InboxSummary, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_value = JsFuture::from(window.fetch_with_str(&format!("{MAIL_API}/api/inbox/summary")))
        .await
        .map_err(|e| format!("fetch: {:?}", e))?;
    let resp: web_sys::Response = resp_value.dyn_into()
        .map_err(|_| "not a Response")?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let json = JsFuture::from(resp.json().map_err(|_| "json() failed")?)
        .await
        .map_err(|e| format!("json: {:?}", e))?;
    serde_wasm_bindgen::from_value(json).map_err(|e| format!("deserialize: {e}"))
}

fn mock_inbox() -> InboxSummary {
    InboxSummary {
        unread_count: 12, total_count: 247,
        recent_threads: vec![
            ThreadSummary { id: "t-001".into(), subject: "Re: Council meeting agenda for Thursday".into(), from_name: "Elena".into(), from_email: "elena@mycelix.local".into(), preview: "I've added the water allocation item to the agenda.".into(), trust_tier: TrustTier::High, timestamp: 1711954920, is_read: false },
            ThreadSummary { id: "t-002".into(), subject: "FL round 47 results \u{2014} 94.2% accuracy".into(), from_name: "Coordinator".into(), from_email: "fl@mycelix.local".into(), preview: "Your gradient contribution improved the model by 0.3%.".into(), trust_tier: TrustTier::High, timestamp: 1711949700, is_read: false },
            ThreadSummary { id: "t-003".into(), subject: "Solar garden construction timeline".into(), from_name: "Block 7 Committee".into(), from_email: "block7@mycelix.local".into(), preview: "Foundation work starts next Monday.".into(), trust_tier: TrustTier::Medium, timestamp: 1711868400, is_read: true },
        ],
        trust_health: TrustHealth { trusted_contacts: 89, quarantined: 3, introductions_pending: 2, average_trust_score: 0.78 },
    }
}

fn tier_color(tier: &TrustTier) -> &'static str {
    match tier { TrustTier::High => "#22c55e", TrustTier::Medium => "#f59e0b", TrustTier::Low => "#ef4444", TrustTier::Unknown => "#6b7280" }
}
fn tier_label(tier: &TrustTier) -> &'static str {
    match tier { TrustTier::High => "TRUSTED", TrustTier::Medium => "MEDIUM", TrustTier::Low => "LOW", TrustTier::Unknown => "UNKNOWN" }
}

#[component]
pub fn MailOverview() -> impl IntoView {
    let inbox_resource = LocalResource::new(move || async move {
        fetch_inbox().await.unwrap_or_else(|e| {
            web_sys::console::log_1(&format!("[Mail] API unavailable: {e}. Mock mode.").into());
            mock_inbox()
        })
    });
    let inbox = move || inbox_resource.get().unwrap_or_else(mock_inbox);

    view! {
        <div class="mail-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Inbox"</button>
                <button class="domain-nav-btn">"Trust Network"</button>
                <button class="domain-nav-btn">"Compose"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #DC2626">"UNREAD"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || inbox().unread_count.to_string()}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">{move || format!("of {} total", inbox().total_count)}</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"TRUSTED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || inbox().trust_health.trusted_contacts.to_string()}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Contacts"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"QUARANTINED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || inbox().trust_health.quarantined.to_string()}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Low-trust held"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #8b5cf6">"TRUST"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{:.0}%", inbox().trust_health.average_trust_score * 100.0)}</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Network avg"</p>
                </div>
            </div>

            <h3 class="section-title">"Recent Threads"</h3>
            <div class="thought-list">
                {move || { inbox().recent_threads.iter().map(|t| {
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
