// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

#[component]
pub fn ReviewPage() -> impl IntoView {
    view! {
        <div class="stub-page">
            <h2>"Spaced Repetition Review"</h2>
            <p>
                "Review items using the SM-2 algorithm for optimal long-term retention. "
                "Cards are scheduled based on your performance history, with intervals "
                "adapting to each item's difficulty."
            </p>
            <div class="planned-features">
                <h3>"Planned Features"</h3>
                <ul>
                    <li>"SM-2 scheduling with easiness factor tracking"</li>
                    <li>"Review queue with priority ordering"</li>
                    <li>"Performance statistics and retention curves"</li>
                    <li>"Cross-course review consolidation"</li>
                    <li>"Holochain-backed progress persistence"</li>
                </ul>
            </div>
        </div>
    }
}
