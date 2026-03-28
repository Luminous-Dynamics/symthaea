// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::components::EarningsCalculator;

#[component]
pub fn ArtistPage() -> impl IntoView {
    view! {
        <div class="page artist-page">
            <h1>"For Artists"</h1>
            <p class="page-subtitle">"Own your music. Choose your economics. Get paid for every play."</p>

            <section class="comparison">
                <h2>"Platform Comparison"</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>"Feature"</th>
                            <th>"Spotify"</th>
                            <th>"Mycelix"</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>"Per-play rate"</td>
                            <td>"$0.003"</td>
                            <td>"$0.007+ (70% of $0.01)"</td>
                        </tr>
                        <tr>
                            <td>"Payment model"</td>
                            <td>"One-size-fits-all"</td>
                            <td>"10+ models (artist choice)"</td>
                        </tr>
                        <tr>
                            <td>"Platform fee"</td>
                            <td>"~30%"</td>
                            <td>"3%"</td>
                        </tr>
                        <tr>
                            <td>"Data ownership"</td>
                            <td>"Platform owns"</td>
                            <td>"Artist owns (source chain)"</td>
                        </tr>
                        <tr>
                            <td>"Listener cost"</td>
                            <td>"$10.99/mo"</td>
                            <td>"Near-zero (micro-payments)"</td>
                        </tr>
                    </tbody>
                </table>
            </section>

            <EarningsCalculator />

            <section class="cta-section">
                <a href="/upload" class="btn btn-primary">"Upload Your First Track"</a>
            </section>
        </div>
    }
}
