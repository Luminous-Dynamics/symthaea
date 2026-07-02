// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Case Studies page — 10 studies in 5 paired comparisons.

use leptos::prelude::*;
use crate::components::lem_badge::LemBadge;
use crate::types::EpistemicTier;

/// Case study data for display.
#[derive(Clone)]
struct CaseStudyView {
    title: String,
    year: String,
    category: String,
    claim: String,
    outcome: String,
    lesson: String,
    lem_e: u8,
    lem_n: u8,
    lem_m: u8,
    equations: Vec<String>,
}

fn paired_studies() -> Vec<(CaseStudyView, CaseStudyView, &'static str)> {
    vec![
        (
            CaseStudyView {
                title: "Cold Fusion (Pons & Fleischmann)".into(), year: "1989".into(),
                category: "Replication Failure".into(),
                claim: "Deuterium fusion at room temperature in palladium, producing excess heat.".into(),
                outcome: "Never replicated. Gamow factor gives tunneling probability ~10^-2700. Calorimetry errors.".into(),
                lesson: "The Gamow factor alone should have flagged this as E0.".into(),
                lem_e: 0, lem_n: 0, lem_m: 0,
                equations: vec!["Gamow Tunneling Factor".into(), "Lawson Criterion".into(), "Coulomb Law".into()],
            },
            CaseStudyView {
                title: "NIF Fusion Ignition".into(), year: "2022-2025".into(),
                category: "Verification Success".into(),
                claim: "Inertial confinement fusion achieving Q > 1, progressing to Q > 4.".into(),
                outcome: "Verified. Q: 1.54 → 1.89 → 2.36 → 4.13. Eight successful shots.".into(),
                lesson: "Same physics as cold fusion but at correct temperatures and densities.".into(),
                lem_e: 4, lem_n: 3, lem_m: 3,
                equations: vec!["Lawson Criterion".into(), "Gamow Peak Integral".into(), "Bremsstrahlung Radiation".into()],
            },
            "Same Physics, Opposite Outcomes"
        ),
        (
            CaseStudyView {
                title: "LK-99 Superconductor".into(), year: "2023".into(),
                category: "Replication Failure".into(),
                claim: "Lead-copper phosphate apatite is a room-temperature superconductor.".into(),
                outcome: "Debunked in 3 weeks. Cu₂S impurity phase transition, not superconductivity.".into(),
                lesson: "Fastest verification cycle in modern physics. Global replication effort was exemplary.".into(),
                lem_e: 0, lem_n: 0, lem_m: 0,
                equations: vec!["BCS Gap Equation".into(), "Meissner Effect".into(), "Curie-Weiss Law".into()],
            },
            CaseStudyView {
                title: "Nickelate Superconductors".into(), year: "2023-2025".into(),
                category: "Verification Success".into(),
                claim: "Bilayer nickelate La₃Ni₂O₇ exhibits ~80K superconductivity.".into(),
                outcome: "Multiple groups reproduced. SLAC stabilized at ambient pressure (Feb 2025).".into(),
                lesson: "The anti-LK-99: steady E1→E3 progression with complementary techniques.".into(),
                lem_e: 3, lem_n: 2, lem_m: 2,
                equations: vec!["BCS Gap Equation".into(), "Eliashberg Equation".into(), "Ginzburg-Landau Equation".into()],
            },
            "Same Domain, Opposite Trajectories"
        ),
        (
            CaseStudyView {
                title: "Hendrik Schön (Bell Labs)".into(), year: "2002".into(),
                category: "Fraud".into(),
                claim: "Organic semiconductor transistors with extraordinary mobility.".into(),
                outcome: "Fabricated data: identical noise patterns across independent experiments. 25+ retractions.".into(),
                lesson: "Hash-based data provenance (BLAKE3) would have caught identical-noise immediately.".into(),
                lem_e: 0, lem_n: 2, lem_m: 0,
                equations: vec!["Ohm Law".into(), "BCS Gap Equation".into(), "Drude Conductivity Model".into()],
            },
            CaseStudyView {
                title: "Ranga Dias (Rochester)".into(), year: "2020-2023".into(),
                category: "Fraud".into(),
                claim: "Room-temperature superconductivity in sulfur hydride and lutetium hydride.".into(),
                outcome: "Same fraud pattern as Schön, 20 years later. 5 papers retracted.".into(),
                lesson: "Traditional peer review cannot detect data fabrication. Automated provenance is necessary.".into(),
                lem_e: 0, lem_n: 2, lem_m: 0,
                equations: vec!["BCS Gap Equation".into(), "London Penetration Depth".into(), "Eliashberg Equation".into()],
            },
            "Peer Review Failure (20 Years Apart)"
        ),
        (
            CaseStudyView {
                title: "LIGO GW150914".into(), year: "2015".into(),
                category: "Verification Gold Standard".into(),
                claim: "Direct detection of gravitational waves from binary black hole merger.".into(),
                outcome: "SNR=24, false alarm < 1/203,000 years, two detectors 3,002 km apart. Nobel 2017.".into(),
                lesson: "The gold standard: independent detectors, quantified false alarm rate, theory confirmed.".into(),
                lem_e: 4, lem_n: 3, lem_m: 3,
                equations: vec!["Gravitational Wave Strain".into(), "Geodesic Equation".into(), "Einstein Field Equations".into()],
            },
            CaseStudyView {
                title: "Higgs Boson Discovery".into(), year: "2012".into(),
                category: "Verification Gold Standard".into(),
                claim: "Scalar boson at ~125 GeV consistent with the Higgs mechanism.".into(),
                outcome: "ATLAS + CMS (independent detectors), multiple decay channels, 5.9σ. Nobel 2013.".into(),
                lesson: "Two experiments see the same particle through different methods — N3 consensus.".into(),
                lem_e: 4, lem_n: 3, lem_m: 3,
                equations: vec!["Higgs Potential".into(), "QCD Running Coupling".into(), "Dirac Equation".into()],
            },
            "Verification Gold Standards"
        ),
        (
            CaseStudyView {
                title: "Hubble Tension".into(), year: "2019-2026".into(),
                category: "Active Controversy".into(),
                claim: "Local H₀ ≈ 73 km/s/Mpc disagrees with CMB H₀ ≈ 67 at 5σ.".into(),
                outcome: "Unresolved. Both sides E4-verified. JWST confirmed Cepheids. DESI suggests dynamical dark energy.".into(),
                lesson: "Genuine paradigm tension: E4 evidence on both sides, N2 consensus it's real, no N3 resolution.".into(),
                lem_e: 4, lem_n: 2, lem_m: 3,
                equations: vec!["Friedmann First Equation".into(), "Hubble Law".into(), "CMB Temperature Redshift".into()],
            },
            CaseStudyView {
                title: "EMDrive (Impossible Thruster)".into(), year: "2001-2021".into(),
                category: "Replication Failure".into(),
                claim: "Microwave cavity produces thrust without propellant.".into(),
                outcome: "Dead. TU Dresden found all thrust was magnetic cable interaction. Properly shielded: thrust = 0.".into(),
                lesson: "Conservation law violation = automatic epistemic red flag. Structural search finds zero analog.".into(),
                lem_e: 0, lem_n: 0, lem_m: 0,
                equations: vec!["Newton Second Law".into(), "Angular Momentum".into()],
            },
            "Legitimate Controversy vs Pseudoscience"
        ),
    ]
}

fn tier_from_e(e: u8) -> EpistemicTier {
    match e {
        0 => EpistemicTier::E0, 1 => EpistemicTier::E1, 2 => EpistemicTier::E2,
        3 => EpistemicTier::E3, _ => EpistemicTier::E4,
    }
}

#[component]
pub fn CaseStudiesPage() -> impl IntoView {
    let pairs = paired_studies();

    view! {
        <div class="page-container">
            <h1 class="page-title">"Case Studies: Why Epistemic Verification Matters"</h1>
            <p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 2rem; line-height: 1.6;">
                "10 high-profile scientific cases organized in 5 paired comparisons. Each pair contrasts \
                 a verification failure with a verification success using the same physics — demonstrating \
                 why multi-dimensional epistemic classification (LEM Cube) is essential."
            </p>

            {pairs.into_iter().map(|(left, right, pair_title)| {
                view! {
                    <div style="margin-bottom: 2rem;">
                        <h2 style="font-size: 1.25rem; color: var(--accent-indigo); margin-bottom: 1rem; text-align: center;">
                            {pair_title}
                        </h2>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            {render_study(left)}
                            {render_study(right)}
                        </div>
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}

fn render_study(study: CaseStudyView) -> impl IntoView {
    let tier = tier_from_e(study.lem_e);
    let border_color = match study.category.as_str() {
        "Replication Failure" | "Fraud" => "border-color: var(--tier-e0);",
        "Verification Success" | "Verification Gold Standard" => "border-color: var(--tier-e4);",
        _ => "border-color: var(--tier-e2);",
    };

    view! {
        <div class="glass-panel" style=format!("border-left: 3px solid; {};", border_color)>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 style="font-size: 1rem;">{study.title}</h3>
                <LemBadge tier=tier />
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                {study.year} " | " {study.category}
            </div>
            <div style="margin-bottom: 0.5rem;">
                <p style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.25rem;">"Claim:"</p>
                <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">{study.claim}</p>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <p style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.25rem;">"Outcome:"</p>
                <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">{study.outcome}</p>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <p style="font-size: 0.8rem; font-weight: 600; color: var(--accent-indigo); margin-bottom: 0.25rem;">"Lesson:"</p>
                <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4; font-style: italic;">{study.lesson}</p>
            </div>
            <div style="font-size: 0.7rem; color: var(--text-secondary);">
                "LEM: E" {study.lem_e} "/N" {study.lem_n} "/M" {study.lem_m}
                " | Equations: " {study.equations.join(", ")}
            </div>
        </div>
    }
}
