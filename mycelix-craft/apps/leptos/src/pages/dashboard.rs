// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Craft Dashboard — overview of profile, credentials, guilds, connections, and jobs.

use leptos::prelude::*;

#[component]
pub fn DashboardPage() -> impl IntoView {
    view! {
        <div class="page dashboard-page">
            <h1>"Craft Dashboard"</h1>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <h3>"Living Credentials"</h3>
                    <p class="dash-stat">"0"</p>
                    <p class="text-secondary">"Published from Praxis with vitality tracking"</p>
                    <a href="/credentials" class="btn-secondary">"View Credentials"</a>
                </div>
                <div class="dash-card">
                    <h3>"Guild Memberships"</h3>
                    <p class="dash-stat">"0"</p>
                    <p class="text-secondary">"Consciousness-gated professional federations"</p>
                </div>
                <div class="dash-card">
                    <h3>"Connections"</h3>
                    <p class="dash-stat">"0"</p>
                    <p class="text-secondary">"Peer-verified network"</p>
                    <a href="/network" class="btn-secondary">"View Network"</a>
                </div>
                <div class="dash-card">
                    <h3>"Endorsements"</h3>
                    <p class="dash-stat">"0"</p>
                    <p class="text-secondary">"Skill attestations from peers"</p>
                </div>
            </div>

            <div class="dashboard-grid" style="margin-top: 1.5rem;">
                <div class="dash-card">
                    <h3>"Quick Actions"</h3>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <a href="/profile" class="btn-primary">"Edit Profile"</a>
                        <a href="/jobs" class="btn-secondary">"Browse Jobs"</a>
                        <a href="/applications" class="btn-secondary">"My Applications"</a>
                        <a href="/employer" class="btn-secondary">"Employer Dashboard"</a>
                    </div>
                </div>

                <div class="dash-card">
                    <h3>"What makes Craft different?"</h3>
                    <ul class="feature-list">
                        <li>"Credentials verified via Praxis Proof of Learning — not self-reported"</li>
                        <li>"Living credentials decay via Ebbinghaus curve — mastery must be proven"</li>
                        <li>"Guilds gate access via consciousness scores — not admin roles"</li>
                        <li>"Job matching runs locally on your device — data never leaves"</li>
                        <li>"Your profile lives on your Holochain agent — no platform owns it"</li>
                    </ul>
                </div>
            </div>
        </div>
    }
}
