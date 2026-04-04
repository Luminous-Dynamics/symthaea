// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics Discovery page — catalog browser, structural search, case studies.

use leptos::prelude::*;
use crate::types::DiscoveryResult;
use crate::components::nuclear_chart::NuclearChart;

/// Physics equation domains and counts (from the 148-equation catalog).
fn catalog_domains() -> Vec<(&'static str, usize)> {
    vec![
        ("Classical Mechanics", 12),
        ("Electromagnetism", 12),
        ("General Relativity", 12),
        ("Quantum Mechanics", 13),
        ("Quantum Field Theory", 3),
        ("Statistical Mechanics", 7),
        ("Fluid Dynamics", 7),
        ("Cosmology", 6),
        ("Nuclear Physics", 9),
        ("Thermodynamics", 15),
        ("Optics", 7),
        ("Condensed Matter", 5),
        ("Particle Physics", 4),
        ("Modified Gravity", 3),
        ("Information Theory", 11),
        ("Biophysics", 6),
        ("Acoustics", 3),
        ("Plasma Physics", 3),
        ("Mathematics", 4),
    ]
}

/// Pre-computed Lazar analysis results.
fn lazar_results() -> Vec<DiscoveryResult> {
    vec![
        DiscoveryResult { name: "Heat Equation".into(), domain: "Thermodynamics".into(), score: 0.629 },
        DiscoveryResult { name: "Yukawa Potential".into(), domain: "Nuclear Physics".into(), score: 0.603 },
        DiscoveryResult { name: "Schwarzschild Metric".into(), domain: "General Relativity".into(), score: 0.583 },
        DiscoveryResult { name: "de Sitter Metric".into(), domain: "Cosmology".into(), score: 0.581 },
        DiscoveryResult { name: "FLRW Metric".into(), domain: "Cosmology".into(), score: 0.579 },
    ]
}

/// Pre-computed Art's Parts analysis results.
fn arts_parts_results() -> Vec<DiscoveryResult> {
    vec![
        DiscoveryResult { name: "Waveguide Dispersion Relation".into(), domain: "Optics".into(), score: 0.915 },
        DiscoveryResult { name: "Friedmann First Equation".into(), domain: "Cosmology".into(), score: 0.469 },
        DiscoveryResult { name: "One-Pion Exchange Potential".into(), domain: "Nuclear Physics".into(), score: 0.393 },
        DiscoveryResult { name: "Yukawa Potential".into(), domain: "Nuclear Physics".into(), score: 0.352 },
        DiscoveryResult { name: "Casimir Force".into(), domain: "Particle Physics".into(), score: 0.334 },
    ]
}

/// Discovery page component.
#[component]
pub fn DiscoveryPage() -> impl IntoView {
    let (active_case, set_active_case) = signal("lazar".to_string());

    let case_results = move || {
        match active_case.get().as_str() {
            "arts-parts" => arts_parts_results(),
            _ => lazar_results(),
        }
    };

    view! {
        <div class="page-container">
            <h1 class="page-title">"Physics Discovery Engine"</h1>

            // Catalog overview
            <div class="glass-panel" style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Equation Catalog — 148 Equations across 19 Domains"</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.5rem;">
                    {catalog_domains().into_iter().map(|(name, count)| {
                        view! {
                            <div style="display: flex; justify-content: space-between; padding: 0.375rem 0.75rem; background: var(--bg-secondary); border-radius: 0.375rem;">
                                <span style="font-size: 0.8rem;">{name}</span>
                                <span style="font-size: 0.8rem; color: var(--accent-indigo); font-weight: 600;">{count}</span>
                            </div>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>

            // Case study selector
            <div class="glass-panel" style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Structural Analysis Case Studies"</h2>
                <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                    <button
                        class="btn btn-primary"
                        style=move || if active_case.get() == "lazar" { "opacity: 1;" } else { "opacity: 0.5;" }
                        on:click=move |_| set_active_case.set("lazar".to_string())
                    >"Lazar Gravity-A"</button>
                    <button
                        class="btn btn-emerald"
                        style=move || if active_case.get() == "arts-parts" { "opacity: 1;" } else { "opacity: 0.5;" }
                        on:click=move |_| set_active_case.set("arts-parts".to_string())
                    >"Art's Parts Waveguide"</button>
                </div>

                <div style="margin-bottom: 1rem;">
                    {move || {
                        if active_case.get() == "lazar" {
                            view! {
                                <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.5rem;">
                                    <strong style="color: var(--text-primary);">"Query: "</strong>
                                    "Extended strong nuclear force via Element 115 proton bombardment "
                                    "produces accessible gravity wave. Structural analog of Yukawa potential "
                                    "with range parameter mu approaching 0 (infinite range)."
                                </p>
                                <p style="font-size: 0.75rem; color: var(--tier-e0);">
                                    "LEM Classification: E0 (zero confidence) / N2 (structural analog exists) / M0 (ephemeral)"
                                </p>
                            }.into_any()
                        } else {
                            view! {
                                <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.5rem;">
                                    <strong style="color: var(--text-primary);">"Query: "</strong>
                                    "Alternating Bismuth/Magnesium layers act as terahertz waveguide "
                                    "for anti-gravity propulsion. ORNL (2022): terrestrial isotopes, "
                                    "impure Bi layers disrupt waveguide function."
                                </p>
                                <p style="font-size: 0.75rem; color: var(--tier-e4);">
                                    "LEM Classification: E0 (sample fails) / N3 (physics is axiomatic) / M1 (optics = temporal)"
                                </p>
                            }.into_any()
                        }
                    }}
                </div>

                <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">"Nearest Structural Neighbors"</h3>
                {move || case_results().into_iter().enumerate().map(|(i, r)| {
                    let bar_width = format!("{}%", (r.score * 100.0) as u32);
                    view! {
                        <div class="discovery-result">
                            <div>
                                <span style="font-weight: 600; margin-right: 0.5rem;">
                                    {format!("{}.", i + 1)}
                                </span>
                                <span>{r.name}</span>
                                <span class="domain">{format!(" ({})", r.domain)}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div style="width: 100px; height: 6px; background: var(--bg-secondary); border-radius: 3px;">
                                    <div style=format!("width: {}; height: 100%; background: var(--accent-emerald); border-radius: 3px;", bar_width)></div>
                                </div>
                                <span class="score">{format!("{:.3}", r.score)}</span>
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>

            // Superheavy discovery
            <div class="glass-panel">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Superheavy Island of Stability"</h2>
                <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem;">
                    "Systematic SEMF + Woods-Saxon shell model sweep of Z=110-120, N=170-190 (231 isotopes). "
                    "All elements show stability pockets with shell corrections -17 to -23 MeV."
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0.5rem;">
                    {vec![
                        ("Fl-114 (A=296)", "-18.67 MeV"),
                        ("Mc-115 (A=300)", "-19.20 MeV"),
                        ("Lv-116 (A=287)", "-19.73 MeV"),
                        ("Og-118 (A=295)", "-21.14 MeV"),
                        ("E120 (A=294)", "-22.81 MeV"),
                    ].into_iter().map(|(elem, corr)| {
                        view! {
                            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0.75rem; background: var(--bg-secondary); border-radius: 0.375rem;">
                                <span style="font-size: 0.875rem; font-weight: 600;">{elem}</span>
                                <span style="font-size: 0.875rem; color: var(--accent-emerald);">{corr}</span>
                            </div>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>

            // Interactive Nuclear Chart
            <NuclearChart z_min=100 z_max=120 n_min=140 n_max=190 />
        </div>
    }
}
