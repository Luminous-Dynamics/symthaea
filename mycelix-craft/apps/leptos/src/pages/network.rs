// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network page — connections, recommendations, and endorsements.

use leptos::prelude::*;

#[component]
pub fn NetworkPage() -> impl IntoView {
    view! {
        <div class="page network-page">
            <h1>"Professional Network"</h1>

            <div class="network-grid">
                <div class="network-section">
                    <h3>"Connections"</h3>
                    <p class="text-secondary">"Your professional connections appear here."</p>
                    <div class="connections-list">
                        <p class="empty-state">"No connections yet. Send a connection request to get started."</p>
                    </div>
                </div>

                <div class="network-section">
                    <h3>"Pending Requests"</h3>
                    <p class="text-secondary">"Incoming connection requests."</p>
                    <div class="requests-list">
                        <p class="empty-state">"No pending requests."</p>
                    </div>
                </div>

                <div class="network-section">
                    <h3>"Recommendations"</h3>
                    <p class="text-secondary">"Professional recommendations from your network."</p>
                    <div class="recommendations-list">
                        <p class="empty-state">"No recommendations yet."</p>
                    </div>
                </div>

                <div class="network-section">
                    <h3>"Skill Endorsements"</h3>
                    <p class="text-secondary">"Peer attestations of your skills."</p>
                    <div class="endorsements-list">
                        <p class="empty-state">"No endorsements yet. Share your agent pubkey with peers."</p>
                    </div>
                </div>
            </div>
        </div>
    }
}
