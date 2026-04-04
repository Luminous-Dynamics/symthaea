// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Home dashboard — stats, recent claims, discovery highlights.

use leptos::prelude::*;

/// Dashboard page with system overview.
#[component]
pub fn HomePage() -> impl IntoView {
    // Static demo data — will connect to API when backend is running
    let total_claims = 93; // Matches our equation catalog size
    let total_categories = 16;
    let total_equations = 93;

    view! {
        <div class="page-container">
            <h1 class="page-title">"Epistemic Verification Engine"</h1>

            <div class="stat-grid">
                <div class="glass-panel stat-card">
                    <div class="stat-value">{total_equations}</div>
                    <div class="stat-label">"Physics Equations"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{total_claims}</div>
                    <div class="stat-label">"Epistemic Claims"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{total_categories}</div>
                    <div class="stat-label">"Physics Domains"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">"E0-E4"</div>
                    <div class="stat-label">"LEM Cube Axes"</div>
                </div>
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
