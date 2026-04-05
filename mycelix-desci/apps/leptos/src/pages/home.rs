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

            <div class="glass-panel" style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Discovery Highlights"</h2>

                <div style="margin-bottom: 1rem;">
                    <h3 style="font-size: 1rem; color: var(--accent-indigo);">"Lazar Gravity-A Structural Analysis"</h3>
                    <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6;">
                        "Nearest neighbors: Yukawa Potential (0.603), Schwarzschild Metric (0.583), "
                        "cosmological metrics (0.58). Classification: E0/N2/M0 — zero empirical confidence, "
                        "moderate structural analog in known physics, ephemeral domain."
                    </p>
                </div>

                <div style="margin-bottom: 1rem;">
                    <h3 style="font-size: 1rem; color: var(--accent-emerald);">"Art's Parts THz Waveguide"</h3>
                    <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6;">
                        "Nearest neighbor: Waveguide Dispersion (0.915). Classification: E0/N3/M1 — "
                        "the physics is axiomatic (textbook optics), but the physical sample fails "
                        "(ORNL 2022: terrestrial isotopes, impure Bi layers)."
                    </p>
                </div>

                <div>
                    <h3 style="font-size: 1rem; color: var(--tier-e3);">"Superheavy Island of Stability"</h3>
                    <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6;">
                        "SEMF + shell model sweep (Z=110-120, N=170-190): stability island centered "
                        "at Z=115/N=180. Element 120 (A=294) shows deepest shell correction (-22.81 MeV). "
                        "Consistent with Moller/Oganessian predictions."
                    </p>
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
