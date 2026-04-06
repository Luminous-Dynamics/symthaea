// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Craft Dashboard — overview of profile, credentials, guilds, connections, and jobs.
//! Mock-first: shows static cards immediately, will load real data when conductor connects.

use leptos::prelude::*;
use mycelix_leptos_core::stat_card::StatCard;

#[component]
pub fn DashboardPage() -> impl IntoView {
    // Mock stats — will be replaced by conductor queries when connected
    let credential_count = signal(0u32);
    let guild_count = signal(0u32);
    let connection_count = signal(0u32);
    let endorsement_count = signal(0u32);

    view! {
        <div class="page dashboard-page">
            <h1>"Craft Dashboard"</h1>

            <div class="dashboard-grid">
                <StatCard
                    title="Living Credentials"
                    value=move || format!("{}", credential_count.0.get())
                    subtitle="Published from Praxis with vitality tracking"
                    href="/profile"
                />
                <StatCard
                    title="Guild Memberships"
                    value=move || format!("{}", guild_count.0.get())
                    subtitle="Consciousness-gated professional federations"
                    href="/profile"
                />
                <StatCard
                    title="Connections"
                    value=move || format!("{}", connection_count.0.get())
                    subtitle="Peer-verified network"
                    href="/network"
                />
                <StatCard
                    title="Endorsements"
                    value=move || format!("{}", endorsement_count.0.get())
                    subtitle="Skill attestations from peers"
                    href="/profile"
                />
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
                        <li>"Credentials verified via Praxis Proof of Learning \u{2014} not self-reported"</li>
                        <li>"Living credentials decay via Ebbinghaus curve \u{2014} mastery must be proven"</li>
                        <li>"Guilds gate access via consciousness scores \u{2014} not admin roles"</li>
                        <li>"Job matching runs locally on your device \u{2014} data never leaves"</li>
                        <li>"Your profile lives on your Holochain agent \u{2014} no platform owns it"</li>
                    </ul>
                </div>
            </div>
        </div>
    }
}
