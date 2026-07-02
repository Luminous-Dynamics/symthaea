// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Home dashboard — stats, recent claims, discovery highlights.

use leptos::prelude::*;

/// Dashboard page with system overview.
///
/// Attempts to fetch real stats from the DeSci REST API.
/// Falls back to static demo data if the API is unavailable.
#[component]
pub fn HomePage() -> impl IntoView {
    let stats = leptos::prelude::LocalResource::new(|| async {
        crate::api::fetch_stats().await.ok()
    });

    let total_claims = move || {
        stats.get().flatten().map(|s| s.total_claims).unwrap_or(50)
    };
    let total_categories = move || {
        stats.get().flatten().map(|s| s.total_categories).unwrap_or(20)
    };
    let api_connected = move || stats.get().flatten().is_some();
    let total_equations = 208;

    view! {
        <div class="page-container">
            // ── Hero Section ──
            <div style="text-align: center; padding: 2rem 0 3rem;">
                <h1 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0.75rem; background: linear-gradient(135deg, var(--accent-indigo), var(--accent-emerald)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    "Mycelix DeSci"
                </h1>
                <p style="font-size: 1.25rem; color: var(--text-secondary); max-width: 700px; margin: 0 auto 1.5rem; line-height: 1.6;">
                    "The world's first epistemic verification engine for science. "
                    "Search 208 physics equations, verify claims against known physics, "
                    "and track reproducibility — all powered by hyperdimensional computing."
                </p>
                <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                    <a href="/discovery" class="btn btn-primary" style="text-decoration: none; padding: 0.75rem 1.5rem; font-size: 1rem;">"Search 208 Equations"</a>
                    <a href="/case-studies" class="btn btn-emerald" style="text-decoration: none; padding: 0.75rem 1.5rem; font-size: 1rem;">"View Case Studies"</a>
                    <a href="/submit" class="btn" style="text-decoration: none; padding: 0.75rem 1.5rem; font-size: 1rem; background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border-glass);">"Submit a Claim"</a>
                </div>
            </div>

            <div class="stat-grid">
                <div class="glass-panel stat-card">
                    <div class="stat-value">{total_equations}</div>
                    <div class="stat-label">"Physics Equations"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{move || total_claims()}</div>
                    <div class="stat-label">"Epistemic Claims"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{move || total_categories()}</div>
                    <div class="stat-label">"Physics Domains"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">"E0-E4"</div>
                    <div class="stat-label">"LEM Cube Axes"</div>
                </div>
            </div>

            // Connection status
            <div style="text-align: center; margin-bottom: 1.5rem; font-size: 0.75rem;">
                {move || if api_connected() {
                    view! { <span style="color: var(--accent-emerald);">"● API Connected (live data)"</span> }.into_any()
                } else {
                    view! { <span style="color: var(--tier-e1);">"○ Demo Mode (API offline)"</span> }.into_any()
                }}
            </div>

            // ── Featured Story: Same Physics, Opposite Outcomes ──
            <div style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; text-align: center; margin-bottom: 0.5rem; color: var(--text-secondary);">"Same Physics. Different Outcomes."</h2>
                <p style="font-size: 0.875rem; text-align: center; color: var(--text-secondary); margin-bottom: 1rem;">"We track the difference."</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    // Cold Fusion — debunked
                    <div class="glass-panel" style="border-left: 3px solid var(--tier-e0);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <h3 style="font-size: 1rem;">"Cold Fusion (1989)"</h3>
                            <span class="lem-badge e0">"E0: Unverified"</span>
                        </div>
                        <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 0.5rem;">
                            "Claimed deuterium fusion at room temperature. Gamow tunneling factor gives probability ~10"<sup>"-2700"</sup>". Never replicated by any lab worldwide."
                        </p>
                        <div style="font-size: 0.7rem; color: var(--tier-e0);">"Verdict: Physically impossible. Calorimetry errors."</div>
                    </div>
                    // NIF Ignition — verified
                    <div class="glass-panel" style="border-left: 3px solid var(--tier-e4);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <h3 style="font-size: 1rem;">"NIF Fusion Ignition (2022-2025)"</h3>
                            <span class="lem-badge e4">"E4: Reproducible"</span>
                        </div>
                        <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 0.5rem;">
                            "Same Gamow physics at correct temperatures. Q ratio: 1.54 → 1.89 → 2.36 → 4.13. Eight successful ignition shots."
                        </p>
                        <div style="font-size: 0.7rem; color: var(--tier-e4);">"Verdict: Independently verified. Progressive improvement."</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 0.75rem;">
                    <a href="/case-studies" style="font-size: 0.8rem; color: var(--accent-indigo); text-decoration: none;">"See all 10 case studies in 5 paired comparisons →"</a>
                </div>
            </div>

            // ── More Highlights ──
            <div class="glass-panel" style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"More Discoveries"</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--accent-indigo); margin-bottom: 0.25rem;">"Art's Parts Waveguide"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                            "0.915 match to textbook waveguide dispersion. Physics is real; ORNL sample fails."
                        </p>
                    </div>
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--accent-emerald); margin-bottom: 0.25rem;">"Superheavy Island"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                            "Z=115-120, N=180. Element 120 deepest shell correction (-22.81 MeV)."
                        </p>
                    </div>
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--tier-e1); margin-bottom: 0.25rem;">"Lazar Gravity-A"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                            "Nearest: Yukawa (0.60), Schwarzschild (0.58). E0/N2/M0."
                        </p>
                    </div>
                </div>
            </div>

            <div class="glass-panel">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"How It Works"</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--accent-indigo);">"1. Simulate"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary);">
                            "Symthaea runs computational physics: nuclear structure, metamaterials, metric engineering"
                        </p>
                    </div>
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--accent-indigo);">"2. Search"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary);">
                            "HDC structural search finds nearest equations in the 93-equation catalog"
                        </p>
                    </div>
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--accent-indigo);">"3. Classify"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary);">
                            "Discovery bridge assigns LEM Cube (E/N/M) epistemic classification"
                        </p>
                    </div>
                    <div>
                        <h3 style="font-size: 0.875rem; color: var(--accent-indigo);">"4. Verify"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary);">
                            "Claims enter the knowledge graph for peer review and prediction markets"
                        </p>
                    </div>
                </div>
            </div>
        </div>
    }
}
