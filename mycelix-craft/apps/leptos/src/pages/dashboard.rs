// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Craft Dashboard — live stats from domain context, conductor-connected or mock.

use leptos::prelude::*;
use crate::context::use_craft;

#[component]
pub fn DashboardPage() -> impl IntoView {
    let craft = use_craft();

    view! {
        <div class="page dashboard-page">
            <h1>"Craft Dashboard"</h1>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <h3>"Living Credentials"</h3>
                    <p class="dash-stat">{move || craft.credentials.get().len()}</p>
                    <p class="text-secondary">"Published from Praxis with vitality tracking"</p>
                    <a href="/credentials" class="btn-secondary">"View Credentials"</a>
                </div>
                <div class="dash-card">
                    <h3>"Guild Memberships"</h3>
                    <p class="dash-stat">{move || craft.guilds.get().len()}</p>
                    <p class="text-secondary">"Consciousness-gated professional federations"</p>
                </div>
                <div class="dash-card">
                    <h3>"Connections"</h3>
                    <p class="dash-stat">{move || craft.connections.get().len()}</p>
                    <p class="text-secondary">"Peer-verified network"</p>
                    <a href="/network" class="btn-secondary">"View Network"</a>
                </div>
                <div class="dash-card">
                    <h3>"Endorsements"</h3>
                    <p class="dash-stat">{move || craft.endorsement_count.get()}</p>
                    <p class="text-secondary">"Skill attestations from peers"</p>
                </div>
            </div>

            {move || craft.loading.get().then(|| view! {
                <div class="loading-bar" style="margin-top: 1rem;">
                    <p class="text-secondary">"Loading data from conductor..."</p>
                </div>
            })}

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
                        <li>"Credentials verified via Praxis Proof of Learning"</li>
                        <li>"Living credentials decay via Ebbinghaus curve"</li>
                        <li>"Guilds gate access via consciousness scores"</li>
                        <li>"Job matching runs locally on your device"</li>
                        <li>"Your profile lives on your Holochain agent"</li>
                    </ul>
                </div>
            </div>
        </div>
    }
}
