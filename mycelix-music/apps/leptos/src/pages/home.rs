// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::components::EarningsCalculator;

#[component]
pub fn HomePage() -> impl IntoView {
    view! {
        <div class="page home-page">
            <section class="hero">
                <h1>"Zero-Cost Music Streaming"</h1>
                <p class="hero-subtitle">
                    "Artists get paid for every play. Listeners pay near-zero fees. "
                    "Powered by Holochain."
                </p>
                <div class="hero-cta">
                    <a href="/discover" class="btn btn-primary">"Start Listening"</a>
                    <a href="/upload" class="btn btn-secondary">"Upload Music"</a>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How It Works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-number">"1"</span>
                        <h3>"Listen for Free"</h3>
                        <p>"Each play is recorded on your local chain — zero blockchain fees."</p>
                    </div>
                    <div class="step">
                        <span class="step-number">"2"</span>
                        <h3>"Plays Accumulate"</h3>
                        <p>"Micro-payments are calculated per play based on the artist's chosen model."</p>
                    </div>
                    <div class="step">
                        <span class="step-number">"3"</span>
                        <h3>"Batched Settlement"</h3>
                        <p>"Plays are batched and settled on-chain periodically — amortized cost."</p>
                    </div>
                </div>
            </section>

            <section class="models">
                <h2>"Choose Your Model"</h2>
                <div class="model-grid">
                    <div class="model-card">
                        <h3>"Pay Per Stream"</h3>
                        <p>"$0.01 per full play. Artists earn 70%."</p>
                    </div>
                    <div class="model-card">
                        <h3>"Gift Economy"</h3>
                        <p>"Free listening. Optional tips. Community rewards."</p>
                    </div>
                    <div class="model-card">
                        <h3>"Patronage"</h3>
                        <p>"Monthly support. Unlimited streaming. Direct artist connection."</p>
                    </div>
                    <div class="model-card">
                        <h3>"Freemium"</h3>
                        <p>"Free 128kbps. Premium for lossless FLAC/24-96kHz."</p>
                    </div>
                </div>
            </section>

            <EarningsCalculator />
        </div>
    }
}
