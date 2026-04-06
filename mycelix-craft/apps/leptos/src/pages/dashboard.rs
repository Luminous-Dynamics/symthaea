// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Craft Dashboard — overview of profile, connections, and job matches.

use leptos::prelude::*;

#[component]
pub fn DashboardPage() -> impl IntoView {
    view! {
        <div class="page dashboard-page">
            <h1>"Craft Dashboard"</h1>

            <div class="dashboard-grid">
                <div class="dash-card">
                    <h3>"Profile"</h3>
                    <p class="dash-stat">"Set up your craft profile to get started."</p>
                    <a href="/profile" class="btn-primary">"Edit Profile"</a>
                </div>

                <div class="dash-card">
                    <h3>"Connections"</h3>
                    <p class="dash-stat">"Build your craft network."</p>
                    <a href="/network" class="btn-secondary">"View Network"</a>
                </div>

                <div class="dash-card">
                    <h3>"Job Matches"</h3>
                    <p class="dash-stat">"Find opportunities matching your verified skills."</p>
                    <a href="/jobs" class="btn-secondary">"Browse Jobs"</a>
                </div>

                <div class="dash-card">
                    <h3>"Applications"</h3>
                    <p class="dash-stat">"Track your application progress."</p>
                    <a href="/applications" class="btn-secondary">"View Applications"</a>
                </div>
            </div>

            <div class="dash-section">
                <h3>"What makes this different?"</h3>
                <ul class="feature-list">
                    <li>"Your credentials are cryptographically verified via EduNet Proof of Learning"</li>
                    <li>"Skill endorsements come from real peers, not self-reported"</li>
                    <li>"Job matching runs locally on your device \u{2014} your data never leaves"</li>
                    <li>"No platform owns your profile \u{2014} it lives on your Holochain agent"</li>
                </ul>
            </div>
        </div>
    }
}
