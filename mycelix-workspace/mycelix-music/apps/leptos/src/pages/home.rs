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
                <h1>"What Does Your Consciousness Sound Like?"</h1>
                <p class="hero-subtitle">
                    "Mycelix Music is a consciousness instrument. "
                    "Your internal state \u{2014} arousal, valence, integration \u{2014} "
                    "becomes sound in real time."
                </p>
                <div class="hero-cta">
                    <a href="/" class="btn btn-primary">"Begin Resonance"</a>
                    <a href="/discover" class="btn btn-secondary">"Discover Others"</a>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"The Consciousness Loop"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-number">"1"</span>
                        <h3>"Set Your State"</h3>
                        <p>"Adjust \u{03A6} (integration), arousal, and valence \u{2014} or let them flow from your Symthaea consciousness engine."</p>
                    </div>
                    <div class="step">
                        <span class="step-number">"2"</span>
                        <h3>"Sound Emerges"</h3>
                        <p>"Your state drives real-time synthesis. Harmonic structure shifts with coherence. Dissonance maps to prediction error."</p>
                    </div>
                    <div class="step">
                        <span class="step-number">"3"</span>
                        <h3>"Feedback Resonates"</h3>
                        <p>"The sound feeds back into the visualization. Consciousness and music become a closed loop."</p>
                    </div>
                </div>
            </section>

            <section class="models">
                <h2>"Artist Economics"</h2>
                <p style="color: var(--text-secondary, #94a3b8); margin-bottom: 1.5rem; max-width: 600px">
                    "Every play is recorded on your local Holochain chain. Artists earn from the first listen."
                </p>
                <div class="model-grid">
                    <div class="model-card">
                        <h3>"Gift Economy"</h3>
                        <p>"Free listening. Optional tips. Community rewards."</p>
                    </div>
                    <div class="model-card">
                        <h3>"Patronage"</h3>
                        <p>"Monthly support. Unlimited streaming. Direct artist connection."</p>
                    </div>
                    <div class="model-card">
                        <h3>"Pay Per Stream"</h3>
                        <p>"$0.01 per full play. Artists earn 70%."</p>
                    </div>
                </div>
            </section>

            <EarningsCalculator />
        </div>
    }
}
