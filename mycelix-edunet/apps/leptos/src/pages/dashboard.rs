// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

#[component]
pub fn DashboardPage() -> impl IntoView {
    view! {
        <div class="stub-page">
            <h2>"Learner Dashboard"</h2>
            <p>
                "Track your learning progress across all enrolled courses. "
                "View personalized recommendations based on your VARK learning style "
                "and BKT mastery estimates."
            </p>
            <div class="planned-features">
                <h3>"Planned Features"</h3>
                <ul>
                    <li>"Course progress tracking with mastery indicators"</li>
                    <li>"BKT knowledge state visualization"</li>
                    <li>"VARK learning style profile"</li>
                    <li>"ZPD-aware next-step recommendations"</li>
                    <li>"XP, badges, and streak tracking (gamification)"</li>
                    <li>"Federated learning round participation history"</li>
                </ul>
            </div>
        </div>
    }
}
