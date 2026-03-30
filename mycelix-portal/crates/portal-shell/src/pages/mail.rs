// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail domain page — inbox summary from Mail REST API.

use leptos::prelude::*;
use portal_viz::{BarChart, bar_chart::Bar};

#[component]
pub fn MailOverview() -> impl IntoView {
    // TODO: fetch from Mail REST API (http://localhost:3001/api/emails)
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
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"12"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Messages awaiting attention"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"TRUSTED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"89"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Contacts in trust network"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"QUARANTINED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"3"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Low-trust messages held"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #8b5cf6">"INTRODUCTIONS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"2"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Pending trust introductions"</p>
                </div>
            </div>

            // Recent threads
            <h3 class="section-title">"Recent Threads"</h3>
            <div class="thought-list">
                <div class="thought-card">
                    <div class="thought-meta">
                        <span class="thought-type" style="color: #22c55e">"TRUSTED"</span>
                        <span class="thought-confidence">"10:42 AM"</span>
                    </div>
                    <p class="thought-content" style="font-weight: 600">"Re: Council meeting agenda for Thursday"</p>
                    <p style="font-size: 0.75rem; color: var(--text-muted)">"Elena \u{2014} I've added the water allocation item to the agenda. Can you bring the latest flow data?"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-meta">
                        <span class="thought-type" style="color: #22c55e">"TRUSTED"</span>
                        <span class="thought-confidence">"9:15 AM"</span>
                    </div>
                    <p class="thought-content" style="font-weight: 600">"FL round 47 results — 94.2% accuracy"</p>
                    <p style="font-size: 0.75rem; color: var(--text-muted)">"Coordinator \u{2014} Your gradient contribution improved the diabetes study model by 0.3%."</p>
                </div>
                <div class="thought-card">
                    <div class="thought-meta">
                        <span class="thought-type" style="color: #f59e0b">"MEDIUM"</span>
                        <span class="thought-confidence">"Yesterday"</span>
                    </div>
                    <p class="thought-content" style="font-weight: 600">"Solar garden construction timeline"</p>
                    <p style="font-size: 0.75rem; color: var(--text-muted)">"Block 7 Committee \u{2014} Foundation work starts next Monday. Volunteer signup attached."</p>
                </div>
            </div>

            <h3 class="section-title" style="margin-top: 1rem">"Trust by Source"</h3>
            <BarChart
                data=vec![
                    Bar { label: "High".into(), value: 67.0, color: "#22c55e".into() },
                    Bar { label: "Medium".into(), value: 18.0, color: "#f59e0b".into() },
                    Bar { label: "Low".into(), value: 3.0, color: "#ef4444".into() },
                    Bar { label: "Unknown".into(), value: 1.0, color: "#6b7280".into() },
                ]
                width=300.0
                height=140.0
            />
        </div>
    }
}
