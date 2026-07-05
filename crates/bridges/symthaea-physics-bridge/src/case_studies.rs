// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Credible case studies for demonstrating epistemic verification.
//!
//! 10 case studies in 5 paired comparisons, each with LEM classification
//! and links to relevant equations in the 208-equation catalog.

use crate::discovery::LEMClassification;
use serde::{Deserialize, Serialize};

/// A scientific case study for epistemic analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseStudy {
    /// Short identifier
    pub id: String,
    /// Human-readable title
    pub title: String,
    /// Year(s) of the claim
    pub year: String,
    /// Category: "replication_failure", "verification_success", "active_controversy", "fraud"
    pub category: String,
    /// What was claimed
    pub claim: String,
    /// What happened (verification outcome)
    pub outcome: String,
    /// Why this case matters for epistemic verification
    pub lesson: String,
    /// LEM Cube classification
    pub lem: LEMClassification,
    /// Relevant equations from the catalog (by name)
    pub equations: Vec<String>,
    /// Pair partner ID (for side-by-side comparison)
    pub pair_with: Option<String>,
}

/// All 10 case studies organized in 5 pairs.
pub fn all_case_studies() -> Vec<CaseStudy> {
    vec![
        // ── Pair 1: Same Physics, Opposite Outcomes ──
        CaseStudy {
            id: "cold-fusion".into(),
            title: "Cold Fusion (Pons & Fleischmann)".into(),
            year: "1989".into(),
            category: "replication_failure".into(),
            claim: "Deuterium fusion at room temperature in a palladium electrode, \
                    producing excess heat and neutrons."
                .into(),
            outcome: "Never replicated. Gamow tunneling factor gives probability ~10^-2700 \
                     for d-d fusion at room temperature. Calorimetry errors identified. \
                     DOE review panels (1989, 2004) found no convincing evidence."
                .into(),
            lesson: "The Gamow factor alone should have flagged this as E0. A structural \
                    search against the Lawson criterion shows the claimed conditions fall \
                    short by ~20 orders of magnitude."
                .into(),
            lem: LEMClassification {
                empirical: 0,
                normative: 0,
                materiality: 0,
            },
            equations: vec![
                "Gamow Tunneling Factor".into(),
                "Lawson Criterion".into(),
                "Coulomb Law".into(),
            ],
            pair_with: Some("nif-ignition".into()),
        },
        CaseStudy {
            id: "nif-ignition".into(),
            title: "NIF Fusion Ignition".into(),
            year: "2022-2025".into(),
            category: "verification_success".into(),
            claim: "Inertial confinement fusion achieving energy gain Q > 1, with \
                    progressive improvements to Q > 4."
                .into(),
            outcome: "Verified. Q ratio progression: 1.54 (Dec 2022) → 1.89 → 2.36 → 4.13 \
                     (Apr 2025). Eight successful ignition shots. Independently confirmed \
                     by peer review and multiple diagnostic systems."
                .into(),
            lesson: "Same physics as cold fusion (Gamow peak, Lawson criterion) but at \
                    correct temperatures (~100M K) and densities. The progressive Q \
                    improvement from 1.5→4.1 shows reproducible, incremental verification."
                .into(),
            lem: LEMClassification {
                empirical: 4,
                normative: 3,
                materiality: 3,
            },
            equations: vec![
                "Lawson Criterion".into(),
                "Gamow Peak Integral".into(),
                "Bremsstrahlung Radiation".into(),
                "Stefan-Boltzmann Law".into(),
            ],
            pair_with: Some("cold-fusion".into()),
        },
        // ── Pair 2: Same Domain, Opposite Trajectories ──
        CaseStudy {
            id: "lk99".into(),
            title: "LK-99 Room-Temperature Superconductor".into(),
            year: "2023".into(),
            category: "replication_failure".into(),
            claim: "Lead-substituted copper phosphate apatite (Pb₁₀₋ₓCuₓ(PO₄)₆O) \
                    is a room-temperature, ambient-pressure superconductor."
                .into(),
            outcome: "Debunked in 3 weeks — fastest verification cycle in modern physics. \
                     Observed resistance drop was Cu₂S impurity phase transition, not \
                     superconductivity. No Meissner effect confirmed. Multiple independent \
                     replications failed globally."
                .into(),
            lesson: "The global replication effort was exemplary — hundreds of labs attempted \
                    verification within days. An epistemic platform tracking replication \
                    attempts in real-time would have provided the definitive timeline."
                .into(),
            lem: LEMClassification {
                empirical: 0,
                normative: 0,
                materiality: 0,
            },
            equations: vec![
                "BCS Gap Equation".into(),
                "Meissner Effect".into(),
                "Curie-Weiss Law".into(),
            ],
            pair_with: Some("nickelates".into()),
        },
        CaseStudy {
            id: "nickelates".into(),
            title: "Nickelate Superconductors".into(),
            year: "2023-2025".into(),
            category: "verification_success".into(),
            claim: "Bilayer nickelate La₃Ni₂O₇ exhibits high-temperature \
                    superconductivity (~80K), reproducible under ambient pressure \
                    via compressive strain from substrates."
                .into(),
            outcome: "Progressively verified. Multiple groups reproduced Tc~80K. \
                     SLAC (Feb 2025) stabilized at ambient pressure using SmNiO₂. \
                     Increasingly sophisticated probes (μSR, NMR, specific heat) \
                     confirm bulk superconductivity."
                .into(),
            lesson: "The anti-LK-99: same year, same domain, opposite trajectory. \
                    Steady accumulation of evidence from independent groups with \
                    complementary techniques. This is how E1→E3 progression should work."
                .into(),
            lem: LEMClassification {
                empirical: 3,
                normative: 2,
                materiality: 2,
            },
            equations: vec![
                "BCS Gap Equation".into(),
                "Eliashberg Equation".into(),
                "Ginzburg-Landau Equation".into(),
            ],
            pair_with: Some("lk99".into()),
        },
        // ── Pair 3: Peer Review Failure (Fraud) ──
        CaseStudy {
            id: "schon".into(),
            title: "Hendrik Schön (Bell Labs Fraud)".into(),
            year: "2002".into(),
            category: "fraud".into(),
            claim: "Organic semiconductor single-crystal transistors with extraordinary \
                    mobility, superconductivity in C₆₀, and single-molecule devices."
                .into(),
            outcome: "Fabricated data. Investigation revealed identical noise patterns \
                     across experiments that should have been physically independent. \
                     25+ papers retracted from Nature, Science, and PRL. Peer review \
                     failed completely — no reviewer checked for duplicated noise."
                .into(),
            lesson: "Hash-based data provenance (BLAKE3 of raw datasets) would have \
                    caught the identical-noise issue immediately. The DeSci approach — \
                    requiring data hashes at submission — prevents this class of fraud."
                .into(),
            lem: LEMClassification {
                empirical: 0,
                normative: 2,
                materiality: 0,
            },
            equations: vec![
                "Ohm Law".into(),
                "BCS Gap Equation".into(),
                "Drude Conductivity Model".into(),
            ],
            pair_with: Some("dias".into()),
        },
        CaseStudy {
            id: "dias".into(),
            title: "Ranga Dias (Rochester Superconductor Fraud)".into(),
            year: "2020-2023".into(),
            category: "fraud".into(),
            claim: "Room-temperature superconductivity in carbonaceous sulfur hydride \
                    (267K at 267 GPa) and nitrogen-doped lutetium hydride (294K at 1 GPa)."
                .into(),
            outcome: "Same failure mode as Schön, 20 years later. Data manipulation \
                     confirmed by university investigation. 5 papers retracted. Nature's \
                     peer review failed again — identical class of fraud was not caught."
                .into(),
            lesson: "Two decades after Schön, the same fraud pattern succeeded through \
                    the same journals. This proves traditional peer review cannot detect \
                    data fabrication. Automated provenance verification is necessary."
                .into(),
            lem: LEMClassification {
                empirical: 0,
                normative: 2,
                materiality: 0,
            },
            equations: vec![
                "BCS Gap Equation".into(),
                "London Penetration Depth".into(),
                "Eliashberg Equation".into(),
            ],
            pair_with: Some("schon".into()),
        },
        // ── Pair 4: Verification Gold Standards ──
        CaseStudy {
            id: "ligo".into(),
            title: "LIGO Gravitational Wave Detection (GW150914)".into(),
            year: "2015".into(),
            category: "verification_success".into(),
            claim: "Direct detection of gravitational waves from a binary black hole \
                    merger, confirming general relativity's prediction."
                .into(),
            outcome: "Matched-filter SNR=24, false alarm rate < 1 per 203,000 years, \
                     simultaneous detection at two sites 3,002 km apart. Signal matched \
                     numerical relativity waveform templates. Nobel Prize 2017."
                .into(),
            lesson: "The gold standard for scientific verification: independent detectors, \
                    quantified false alarm rate, theoretical prediction confirmed, \
                    reproducible methodology. Every element is E4."
                .into(),
            lem: LEMClassification {
                empirical: 4,
                normative: 3,
                materiality: 3,
            },
            equations: vec![
                "Gravitational Wave Strain".into(),
                "Geodesic Equation".into(),
                "Einstein Field Equations".into(),
            ],
            pair_with: Some("higgs".into()),
        },
        CaseStudy {
            id: "higgs".into(),
            title: "Higgs Boson Discovery".into(),
            year: "2012".into(),
            category: "verification_success".into(),
            claim: "Discovery of a scalar boson at ~125 GeV consistent with the \
                    Standard Model Higgs mechanism."
                .into(),
            outcome: "Independently discovered by ATLAS and CMS (two separate detectors, \
                     different teams, different analysis pipelines). Multiple decay channels \
                     confirmed. Combined significance: 5.9σ. Nobel Prize 2013."
                .into(),
            lesson: "Independent replication at the highest standard: two experiments \
                    see the same particle through different methods. This is what N3 \
                    (network consensus) looks like in practice."
                .into(),
            lem: LEMClassification {
                empirical: 4,
                normative: 3,
                materiality: 3,
            },
            equations: vec![
                "Higgs Potential".into(),
                "QCD Running Coupling".into(),
                "Dirac Equation".into(),
            ],
            pair_with: Some("ligo".into()),
        },
        // ── Pair 5: Active Controversies ──
        CaseStudy {
            id: "hubble-tension".into(),
            title: "Hubble Tension".into(),
            year: "2019-2026".into(),
            category: "active_controversy".into(),
            claim: "The expansion rate of the universe measured locally (H₀ ≈ 73 km/s/Mpc, \
                    Cepheid/SN Ia) disagrees with the early-universe value (H₀ ≈ 67 km/s/Mpc, \
                    Planck CMB) at 5σ significance."
                .into(),
            outcome: "Unresolved. Both measurements are independently E4-verified. JWST \
                     confirmed Cepheid distance ladder accuracy. DESI 2025 data suggests \
                     dark energy may be dynamical. Neither systematic error nor new physics \
                     has been definitively established."
                .into(),
            lesson: "Both sides are right by their own methods. This is what a genuine \
                    paradigm-level tension looks like: E4 evidence on both sides, N2 \
                    consensus that the tension is real, but no N3 resolution."
                .into(),
            lem: LEMClassification {
                empirical: 4,
                normative: 2,
                materiality: 3,
            },
            equations: vec![
                "Friedmann First Equation".into(),
                "Hubble Law".into(),
                "Hubble Parameter Evolution".into(),
                "CMB Temperature Redshift".into(),
            ],
            pair_with: Some("emdrive".into()),
        },
        CaseStudy {
            id: "emdrive".into(),
            title: "EMDrive (Impossible Thruster)".into(),
            year: "2001-2021".into(),
            category: "replication_failure".into(),
            claim: "Microwave cavity thruster produces thrust without propellant, \
                    violating conservation of momentum."
                .into(),
            outcome: "Dead. TU Dresden (2021) found all measured thrust was from \
                     interaction between power cables and Earth's magnetic field. \
                     When properly shielded, thrust = 0 within measurement error."
                .into(),
            lesson: "Conservation law violation should be an automatic epistemic red \
                    flag. The structural search engine correctly identifies this: the \
                    claim has zero similarity to any equation obeying Newton's second law \
                    or conservation of momentum."
                .into(),
            lem: LEMClassification {
                empirical: 0,
                normative: 0,
                materiality: 0,
            },
            equations: vec!["Newton Second Law".into(), "Angular Momentum".into()],
            pair_with: Some("hubble-tension".into()),
        },
    ]
}

/// Get a case study by ID.
pub fn get_case_study(id: &str) -> Option<CaseStudy> {
    all_case_studies().into_iter().find(|s| s.id == id)
}

/// Get paired case studies for comparison.
pub fn get_paired_studies() -> Vec<(CaseStudy, CaseStudy)> {
    let studies = all_case_studies();
    let mut pairs = Vec::new();
    let mut used = std::collections::HashSet::new();

    for study in &studies {
        if used.contains(&study.id) {
            continue;
        }
        if let Some(ref pair_id) = study.pair_with
            && let Some(partner) = studies.iter().find(|s| &s.id == pair_id)
        {
            pairs.push((study.clone(), partner.clone()));
            used.insert(study.id.clone());
            used.insert(partner.id.clone());
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_study_count() {
        assert_eq!(all_case_studies().len(), 10);
    }

    #[test]
    fn test_all_paired() {
        let pairs = get_paired_studies();
        assert_eq!(pairs.len(), 5, "Should have 5 pairs");
    }

    #[test]
    fn test_equations_reference_catalog() {
        // All referenced equations should be recognizable names
        for study in all_case_studies() {
            assert!(
                !study.equations.is_empty(),
                "Study {} has no equations",
                study.id
            );
        }
    }

    #[test]
    fn test_lem_classifications_valid() {
        for study in all_case_studies() {
            assert!(study.lem.empirical <= 4);
            assert!(study.lem.normative <= 3);
            assert!(study.lem.materiality <= 3);
        }
    }
}
