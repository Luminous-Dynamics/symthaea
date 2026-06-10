// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Isotope Property Characterization Engine
//!
//! Computes comprehensive nuclear properties using the DZ10+RF predictor:
//!
//! - **Decay modes**: alpha, beta±, spontaneous fission, proton/neutron emission
//! - **Half-life estimates**: Geiger-Nuttall (alpha), Sargent (beta), WKB (fission)
//! - **Separation energies**: S_n, S_p, S_2n, S_2p, S_alpha
//! - **Drip line detection**: where separation energies go negative
//! - **Astrophysical relevance**: r-process path, rp-process, s-process branch points
//! - **Medical/industrial interest**: isotope production potential
//!
//! ## References
//!
//! - Geiger & Nuttall (1911). alpha half-life vs Q-value.
//! - Sargent (1933). beta decay ft values.
//! - Viola & Seaborg (1966). Spontaneous fission systematics.
//! - Möller et al. (2003). r-process path calculations.

use crate::deformation::frdm_deformation;
use crate::duflo_zuker::dz_binding_energy;
use crate::fission_barrier::compute_fission_barrier;
use crate::ml_mass::MlMassPredictor;
use serde::{Deserialize, Serialize};

/// Hydrogen atom mass excess (MeV)
const M_H: f64 = 7.28897;
/// Neutron mass excess (MeV)
const M_N: f64 = 8.07132;
/// He-4 binding energy (MeV)
const BE_HE4: f64 = 28.296;
/// Electron mass (MeV)
const M_E: f64 = 0.511;

/// Dominant decay mode for an isotope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayMode {
    Stable,
    AlphaDecay,
    BetaMinus,
    BetaPlus,
    ProtonEmission,
    NeutronEmission,
    SpontaneousFission,
    DoubleAlpha,
}

impl std::fmt::Display for DecayMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecayMode::Stable => write!(f, "stable"),
            DecayMode::AlphaDecay => write!(f, "α"),
            DecayMode::BetaMinus => write!(f, "β⁻"),
            DecayMode::BetaPlus => write!(f, "β⁺/EC"),
            DecayMode::ProtonEmission => write!(f, "p"),
            DecayMode::NeutronEmission => write!(f, "n"),
            DecayMode::SpontaneousFission => write!(f, "SF"),
            DecayMode::DoubleAlpha => write!(f, "2α"),
        }
    }
}

/// Astrophysical relevance classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AstroRelevance {
    /// On the r-process path (neutron-rich, short β⁻ half-life)
    RProcess,
    /// s-process branch point (slow neutron capture)
    SProcessBranch,
    /// rp-process (proton-rich, X-ray bursts)
    RpProcess,
    /// p-process (proton-rich, supernovae)
    PProcess,
    /// No special astrophysical role
    None,
}

/// Application interest classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplicationInterest {
    /// Medical isotope (imaging, therapy)
    Medical,
    /// Nuclear energy (fission/fusion fuel, transmutation target)
    Energy,
    /// Industrial (radiography, gauging, tracers)
    Industrial,
    /// Fundamental research
    Research,
    /// No specific application identified
    None,
}

/// Comprehensive property characterization of an isotope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotopeProperties {
    pub z: u16,
    pub n: u16,
    pub a: u16,

    // Binding
    pub binding_energy: f64,
    pub ba: f64,
    pub ml_correction: f64,
    pub uncertainty: f64,

    // Separation energies (MeV)
    pub s_n: f64,
    pub s_p: f64,
    pub s_2n: f64,
    pub s_2p: f64,
    pub s_alpha: f64,

    // Decay
    pub q_alpha: f64,
    pub q_beta_minus: f64,
    pub q_beta_plus: f64,
    pub dominant_decay: DecayMode,
    pub alpha_half_life_s: Option<f64>,
    pub beta_half_life_s: Option<f64>,
    pub fission_barrier: f64,

    // Deformation
    pub beta2: f64,

    // Classification
    pub astro_relevance: AstroRelevance,
    pub application: ApplicationInterest,
    pub is_bound: bool,
    pub beyond_drip_line: bool,
}

/// The isotope property engine.
pub struct IsotopePropertyEngine {
    predictor: MlMassPredictor,
}

impl IsotopePropertyEngine {
    pub fn new() -> Self {
        Self {
            predictor: MlMassPredictor::new(),
        }
    }

    /// Predict binding energy using the ML model.
    fn be(&self, z: u16, n: u16) -> f64 {
        if z < 1 || n < 1 {
            return 0.0;
        }
        self.predictor.predict(z, n).binding_energy
    }

    /// Compute full properties for a single isotope.
    pub fn characterize(&self, z: u16, n: u16) -> IsotopeProperties {
        let pred = self.predictor.predict(z, n);
        let a = z + n;
        let be = pred.binding_energy;

        // ── Separation energies ──
        let s_n = if n >= 1 {
            be - self.be(z, n - 1)
        } else {
            -99.0
        };
        let s_p = if z >= 1 {
            be - self.be(z - 1, n)
        } else {
            -99.0
        };
        let s_2n = if n >= 2 {
            be - self.be(z, n - 2)
        } else {
            -99.0
        };
        let s_2p = if z >= 2 {
            be - self.be(z - 2, n)
        } else {
            -99.0
        };
        let s_alpha = if z >= 2 && n >= 2 {
            be - self.be(z - 2, n - 2) - BE_HE4
        } else {
            -99.0
        };

        // ── Q-values ──
        // Q_alpha = BE(parent) - BE(daughter) - BE(He-4)
        let q_alpha = if z >= 2 && n >= 2 {
            BE_HE4 + self.be(z - 2, n - 2) - be
        } else {
            -99.0
        };

        // Q_beta- = M(Z,A) - M(Z+1,A) = [BE(Z+1,N-1) - BE(Z,N)] + (M_n - M_H)
        let q_beta_minus = if n >= 1 {
            self.be(z + 1, n - 1) - be + (M_N - M_H)
        } else {
            -99.0
        };

        // Q_beta+ = M(Z,A) - M(Z-1,A) - 2*m_e = [BE(Z-1,N+1) - BE(Z,N)] - (M_N - M_H) - 2*M_E
        let q_beta_plus = if z >= 2 {
            self.be(z - 1, n + 1) - be - (M_N - M_H) - 2.0 * M_E
        } else {
            -99.0
        };

        // ── Fission barrier ──
        let fb = compute_fission_barrier(z, n);
        let fission_barrier = fb.total_barrier;

        // ── Alpha half-life (Geiger-Nuttall) ──
        let alpha_half_life_s = if q_alpha > 0.0 && z >= 2 {
            Some(geiger_nuttall_half_life(z, q_alpha))
        } else {
            None
        };

        // ── Beta half-life (Sargent rule approximation) ──
        let beta_half_life_s = if q_beta_minus > 0.0 {
            Some(sargent_beta_half_life(q_beta_minus))
        } else if q_beta_plus > 0.0 {
            Some(sargent_beta_half_life(q_beta_plus))
        } else {
            None
        };

        // ── Dominant decay mode ──
        let dominant_decay = determine_dominant_decay(
            s_n,
            s_p,
            q_alpha,
            q_beta_minus,
            q_beta_plus,
            fission_barrier,
            alpha_half_life_s,
            beta_half_life_s,
            z,
            n,
        );

        // ── Deformation ──
        let (beta2, _) = frdm_deformation(z, n);

        // ── Drip line ──
        let beyond_drip_line = s_n < 0.0 || s_p < 0.0;
        let is_bound = be > 0.0 && s_n > -5.0 && s_p > -5.0;

        // ── Astrophysical relevance ──
        let astro_relevance = classify_astro(z, n, s_n, q_beta_minus, beta_half_life_s);

        // ── Application interest ──
        let application = classify_application(z, n, a, dominant_decay);

        IsotopeProperties {
            z,
            n,
            a: z + n,
            binding_energy: be,
            ba: pred.ba,
            ml_correction: pred.ml_correction,
            uncertainty: pred.uncertainty,
            s_n,
            s_p,
            s_2n,
            s_2p,
            s_alpha,
            q_alpha,
            q_beta_minus,
            q_beta_plus,
            dominant_decay,
            alpha_half_life_s,
            beta_half_life_s,
            fission_barrier,
            beta2,
            astro_relevance,
            application,
            is_bound,
            beyond_drip_line,
        }
    }

    /// Scan a rectangular region of the nuclear chart.
    pub fn scan_region(
        &self,
        z_min: u16,
        z_max: u16,
        n_min: u16,
        n_max: u16,
    ) -> Vec<IsotopeProperties> {
        let mut results = Vec::new();
        for z in z_min..=z_max {
            for n in n_min..=n_max {
                let props = self.characterize(z, n);
                if props.is_bound {
                    results.push(props);
                }
            }
        }
        results
    }
}

impl Default for IsotopePropertyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Geiger-Nuttall alpha-decay half-life estimate.
///
/// log₁₀(t½/s) = a·Z/√(Q_α) + b
/// Using Viola-Seaborg-Swiatecki parametrization.
fn geiger_nuttall_half_life(z: u16, q_alpha: f64) -> f64 {
    if q_alpha <= 0.0 {
        return f64::INFINITY;
    }
    let z_f = z as f64;
    let z_d = z_f - 2.0; // Daughter Z

    // Viola-Seaborg: log₁₀(t½/s) = (aZ + b)/√Q + (cZ + d)
    let a = 1.66175;
    let b = -8.5166;
    let c = -0.20228;
    let d = -33.9069;

    let log10_t = (a * z_d + b) / q_alpha.sqrt() + c * z_d + d;
    10.0_f64.powf(log10_t)
}

/// Sargent rule approximation for beta-decay half-life.
///
/// t½ ∝ Q⁻⁵ (allowed transitions).
/// Calibrated to known decays.
fn sargent_beta_half_life(q_beta: f64) -> f64 {
    if q_beta <= 0.0 {
        return f64::INFINITY;
    }
    // For allowed transitions: log₁₀(ft) ≈ 3.0-6.0
    // t½ ≈ C / Q^5 where C calibrated to give ~1s for Q≈5 MeV
    let ft_value = 1000.0; // log(ft) ≈ 3 for superallowed
    ft_value / q_beta.powi(5)
}

/// Determine dominant decay mode from all available Q-values and barriers.
fn determine_dominant_decay(
    s_n: f64,
    s_p: f64,
    q_alpha: f64,
    q_beta_minus: f64,
    q_beta_plus: f64,
    fission_barrier: f64,
    alpha_hl: Option<f64>,
    beta_hl: Option<f64>,
    z: u16,
    _n: u16,
) -> DecayMode {
    // Particle emission is fastest if separation energy is negative
    if s_n < -2.0 {
        return DecayMode::NeutronEmission;
    }
    if s_p < -2.0 {
        return DecayMode::ProtonEmission;
    }

    // Spontaneous fission dominates for very heavy nuclei with low barriers
    if z > 100 && fission_barrier < 3.0 {
        return DecayMode::SpontaneousFission;
    }

    // Compare alpha and beta half-lives
    let alpha_possible = q_alpha > 0.0;
    let beta_minus_possible = q_beta_minus > 0.0;
    let beta_plus_possible = q_beta_plus > 0.0;

    // If nothing decays, it's stable
    if !alpha_possible && !beta_minus_possible && !beta_plus_possible {
        return DecayMode::Stable;
    }

    // Pick the fastest decay
    let alpha_t = alpha_hl.unwrap_or(f64::INFINITY);
    let beta_t = beta_hl.unwrap_or(f64::INFINITY);

    if alpha_possible && alpha_t < beta_t {
        DecayMode::AlphaDecay
    } else if beta_minus_possible && q_beta_minus > q_beta_plus.max(0.0) {
        DecayMode::BetaMinus
    } else if beta_plus_possible {
        DecayMode::BetaPlus
    } else if alpha_possible {
        DecayMode::AlphaDecay
    } else {
        DecayMode::Stable
    }
}

/// Classify astrophysical relevance.
fn classify_astro(
    z: u16,
    n: u16,
    s_n: f64,
    q_beta_minus: f64,
    beta_hl: Option<f64>,
) -> AstroRelevance {
    let n_over_z = n as f64 / z.max(1) as f64;

    // r-process: very neutron-rich, S_n ~ 2-4 MeV (waiting point), fast beta decay
    if n_over_z > 1.4 && s_n > 0.0 && s_n < 4.0 && q_beta_minus > 2.0 {
        if let Some(t) = beta_hl {
            if t < 10.0 {
                // Less than 10 seconds
                return AstroRelevance::RProcess;
            }
        }
    }

    // r-process: any neutron-rich isotope near the drip line
    if n_over_z > 1.5 && s_n > 0.0 && s_n < 3.0 {
        return AstroRelevance::RProcess;
    }

    // s-process branch: near stability, moderate S_n, Z in s-process range
    if n_over_z > 1.0 && n_over_z < 1.4 && s_n > 5.0 && s_n < 9.0 {
        // Known s-process branch points near specific elements
        if matches!(z, 35 | 36 | 56 | 57 | 63 | 64 | 79 | 80) {
            return AstroRelevance::SProcessBranch;
        }
    }

    // rp-process: proton-rich, fast β⁺ decay
    if n_over_z < 0.9 && z > 10 {
        return AstroRelevance::RpProcess;
    }

    // p-process: proton-rich, heavier elements
    if n_over_z < 1.0 && z > 34 {
        return AstroRelevance::PProcess;
    }

    AstroRelevance::None
}

/// Classify application interest.
fn classify_application(z: u16, n: u16, a: u16, decay: DecayMode) -> ApplicationInterest {
    // Known medical isotopes and their neighbors
    let medical = matches!(
        (z, a),
        (42, 99)   | // Mo-99 → Tc-99m (diagnostic imaging)
        (43, 99)   | // Tc-99m
        (53, 131)  | // I-131 (thyroid)
        (53, 125)  | // I-125 (brachytherapy)
        (71, 177)  | // Lu-177 (PRRT therapy)
        (89, 225)  | // Ac-225 (TAT alpha therapy)
        (39, 90)   | // Y-90 (microspheres)
        (64, 153)  | // Gd-153 (bone density)
        (31, 68)   | // Ga-68 (PET)
        (38, 89)   | // Sr-89 (bone pain)
        (29, 64)   | // Cu-64 (PET/therapy)
        (29, 67)   | // Cu-67 (therapy)
        (82, 212)  | // Pb-212 (alpha therapy)
        (83, 213)  | // Bi-213 (alpha therapy)
        (88, 223) // Ra-223 (bone metastases)
    );
    if medical {
        return ApplicationInterest::Medical;
    }

    // Nuclear energy: actinides and near neighbors
    if z >= 89 && z <= 98 {
        return ApplicationInterest::Energy;
    }

    // Transmutation targets for medical isotope production
    // (isotopes that produce medical isotopes via neutron capture or decay)
    let production_parent = matches!(
        (z, a),
        (42, 98)  | // Mo-98 + n → Mo-99
        (70, 176) | // Yb-176 + n → Lu-177
        (88, 226) | // Ra-226 → Ac-225 chain
        (52, 130) // Te-130 for I-131 production
    );
    if production_parent {
        return ApplicationInterest::Medical;
    }

    // Industrial: Co-60, Cs-137, Ir-192, Am-241
    let industrial = matches!(
        (z, a),
        (27, 60)  | // Co-60 (radiography, sterilization)
        (55, 137) | // Cs-137 (gauging)
        (77, 192) | // Ir-192 (radiography)
        (95, 241) // Am-241 (smoke detectors, gauging)
    );
    if industrial {
        return ApplicationInterest::Industrial;
    }

    ApplicationInterest::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_characterize_pb208() {
        let engine = IsotopePropertyEngine::new();
        let props = engine.characterize(82, 126);
        eprintln!(
            "Pb-208: BE={:.2}, B/A={:.4}, S_n={:.2}, S_2n={:.2}, Q_α={:.2}, decay={}",
            props.binding_energy,
            props.ba,
            props.s_n,
            props.s_2n,
            props.q_alpha,
            props.dominant_decay
        );
        assert!(props.binding_energy > 1600.0);
        assert!(props.is_bound);
        assert!(!props.beyond_drip_line);
    }

    #[test]
    fn test_characterize_fe56() {
        let engine = IsotopePropertyEngine::new();
        let props = engine.characterize(26, 30);
        eprintln!("Fe-56: B/A={:.4}, decay={}", props.ba, props.dominant_decay);
        assert!((props.ba - 8.79).abs() < 0.5);
    }

    /// Scan the r-process path: neutron-rich nuclei critical for heavy element nucleosynthesis.
    ///
    /// The r-process in neutron star mergers creates roughly half of all elements
    /// heavier than iron. The path runs along the neutron drip line where S_n ~ 2-4 MeV.
    /// Most nuclei on this path have never been measured — our model predicts their
    /// properties for the first time.
    #[test]
    fn test_rprocess_path_scan() {
        let engine = IsotopePropertyEngine::new();

        eprintln!("\n=== r-Process Path Scan (Neutron Star Merger Nucleosynthesis) ===");
        eprintln!(
            "Looking for neutron-rich nuclei with S_n = 2-4 MeV (r-process waiting points)\n"
        );

        let mut rprocess_nuclei = Vec::new();

        // Scan neutron-rich region: Z=26-82, look for S_n ~ 2-4 MeV
        for z in (26..=82).step_by(2) {
            // For each Z, scan N from stability outward until drip line
            let n_start = (z as f64 * 1.3) as u16;
            let n_end = (z as f64 * 2.2).min(200.0) as u16;

            for n in n_start..=n_end {
                let props = engine.characterize(z, n);
                if props.s_n > 1.5 && props.s_n < 4.5 && props.q_beta_minus > 1.0 {
                    rprocess_nuclei.push(props);
                    break; // Found the r-process waiting point for this Z
                }
                if props.s_n < 0.0 {
                    break;
                } // Past drip line
            }
        }

        eprintln!(
            "{:>4} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Z", "N", "A", "B/A", "S_n", "S_2n", "Q_β⁻", "t½_β(s)", "β₂", "σ"
        );
        eprintln!("{}", "-".repeat(88));

        for p in &rprocess_nuclei {
            let beta_t = p
                .beta_half_life_s
                .map(|t| {
                    if t < 0.001 {
                        format!("{:.1e}", t)
                    } else if t < 1.0 {
                        format!("{:.3}", t)
                    } else {
                        format!("{:.1}", t)
                    }
                })
                .unwrap_or_else(|| "∞".to_string());

            eprintln!(
                "{:>4} {:>4} {:>5} {:>8.4} {:>8.2} {:>8.2} {:>8.2} {:>8} {:>8.3} {:>8.2}",
                p.z, p.n, p.a, p.ba, p.s_n, p.s_2n, p.q_beta_minus, beta_t, p.beta2, p.uncertainty
            );
        }

        eprintln!(
            "\nFound {} r-process waiting points from Z=26 to Z=82",
            rprocess_nuclei.len()
        );
        assert!(!rprocess_nuclei.is_empty(), "Should find r-process nuclei");
    }

    /// Scan for medical isotope production opportunities.
    ///
    /// Identifies unmeasured or poorly-known isotopes near established medical
    /// isotopes that could serve as production targets or novel therapeutics.
    #[test]
    fn test_medical_isotope_scan() {
        let engine = IsotopePropertyEngine::new();

        eprintln!("\n=== Medical Isotope Region Scan ===\n");

        let medical_targets: &[(u16, u16, &str)] = &[
            (42, 57, "Mo-99/Tc-99m (diagnostic)"),
            (53, 78, "I-131 (thyroid therapy)"),
            (71, 106, "Lu-177 (PRRT therapy)"),
            (89, 136, "Ac-225 (targeted alpha)"),
            (29, 35, "Cu-64 (PET/therapy theranostic)"),
            (31, 37, "Ga-68 (PET imaging)"),
        ];

        for &(z_center, n_center, name) in medical_targets {
            eprintln!("--- {} (Z={}, N={}) ---", name, z_center, n_center);
            eprintln!(
                "{:>4} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
                "Z", "N", "A", "B/A", "S_n", "Q_β⁻", "Q_α", "σ", "Decay"
            );

            for dz in -2_i16..=2 {
                for dn in -3_i16..=3 {
                    let z = (z_center as i16 + dz) as u16;
                    let n = (n_center as i16 + dn) as u16;
                    if z < 1 || n < 1 {
                        continue;
                    }
                    let props = engine.characterize(z, n);
                    if !props.is_bound {
                        continue;
                    }

                    let marker = if dz == 0 && dn == 0 { " <<<" } else { "" };
                    eprintln!(
                        "{:>4} {:>4} {:>5} {:>8.4} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>6}{}",
                        z,
                        n,
                        props.a,
                        props.ba,
                        props.s_n,
                        props.q_beta_minus,
                        props.q_alpha,
                        props.uncertainty,
                        props.dominant_decay,
                        marker
                    );
                }
            }
            eprintln!();
        }
    }

    /// Superheavy element characterization with full decay properties.
    #[test]
    fn test_superheavy_properties() {
        let engine = IsotopePropertyEngine::new();

        eprintln!("\n=== Superheavy Element Full Characterization ===");
        eprintln!(
            "Properties beyond just binding energy: decay chains, fission barriers, half-lives\n"
        );

        // Focus on the predicted island of stability: Z=112-120
        let mut candidates = Vec::new();

        for z in 112..=120 {
            // Find the most stable isotope for each Z
            let mut best: Option<IsotopeProperties> = None;
            for n in 160..=190 {
                let props = engine.characterize(z, n);
                if props.s_2n > 0.0 && props.s_2p > 0.0 {
                    if best.as_ref().map(|b| props.ba > b.ba).unwrap_or(true) {
                        best = Some(props);
                    }
                }
            }
            if let Some(b) = best {
                candidates.push(b);
            }
        }

        eprintln!(
            "{:>4} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8} {:>6} {:>8}",
            "Z", "N", "A", "B/A", "Q_α", "B_f", "β₂", "t½_α(s)", "S_2n", "Decay", "σ"
        );
        eprintln!("{}", "-".repeat(102));

        for p in &candidates {
            let alpha_t = p
                .alpha_half_life_s
                .map(|t| {
                    if t < 1e-6 {
                        format!("{:.1e}", t)
                    } else if t < 1.0 {
                        format!("{:.3}", t)
                    } else if t < 3.15e7 {
                        format!("{:.1}", t)
                    } else {
                        format!("{:.1e}", t)
                    }
                })
                .unwrap_or_else(|| "∞".to_string());

            eprintln!(
                "{:>4} {:>4} {:>5} {:>8.4} {:>8.2} {:>8.2} {:>8.3} {:>10} {:>8.2} {:>6} {:>8.2}",
                p.z,
                p.n,
                p.a,
                p.ba,
                p.q_alpha,
                p.fission_barrier,
                p.beta2,
                alpha_t,
                p.s_2n,
                p.dominant_decay,
                p.uncertainty
            );
        }

        eprintln!("\n=== Decay Chain from Z=118, N=176 (Oganesson-294) ===");
        // Follow the alpha-decay chain
        let mut z = 118_u16;
        let mut n = 176_u16;
        let mut step = 0;
        while z >= 82 && step < 15 {
            let props = engine.characterize(z, n);
            let alpha_t = props
                .alpha_half_life_s
                .map(|t| {
                    if t < 1e-6 {
                        format!("{:.1e}s", t)
                    } else if t < 1.0 {
                        format!("{:.3}s", t)
                    } else if t < 60.0 {
                        format!("{:.1}s", t)
                    } else if t < 3600.0 {
                        format!("{:.1}min", t / 60.0)
                    } else if t < 86400.0 {
                        format!("{:.1}hr", t / 3600.0)
                    } else if t < 3.15e7 {
                        format!("{:.1}d", t / 86400.0)
                    } else {
                        format!("{:.2e}yr", t / 3.15e7)
                    }
                })
                .unwrap_or_else(|| "stable".to_string());

            eprintln!(
                "  {} Z={:>3} N={:>3} A={:>3} | Q_α={:>6.2} | t½={:>12} | {}",
                if step == 0 { "→" } else { "↓" },
                z,
                n,
                z + n,
                props.q_alpha,
                alpha_t,
                props.dominant_decay
            );

            if props.dominant_decay != DecayMode::AlphaDecay || props.q_alpha <= 0.0 {
                break;
            }
            z -= 2;
            n -= 2;
            step += 1;
        }

        assert!(!candidates.is_empty());
    }

    /// Map the neutron drip line — the boundary beyond which nuclei
    /// can't hold another neutron (S_n < 0).
    #[test]
    fn test_drip_line_mapping() {
        let engine = IsotopePropertyEngine::new();

        eprintln!("\n=== Neutron Drip Line (S_n = 0 boundary) ===");
        eprintln!("Where nuclei fall apart — defines the edge of the nuclear landscape\n");

        eprintln!(
            "{:>4} {:>4} {:>5} {:>8} {:>8} {:>8}",
            "Z", "N", "A", "S_n", "S_2n", "B/A"
        );

        let mut drip_line = Vec::new();

        for z in (8..=82).step_by(2) {
            let n_start = z; // Start from N=Z
            let n_max = (z as f64 * 2.5).min(250.0) as u16;

            let mut last_bound_n = n_start;
            for n in n_start..=n_max {
                let props = engine.characterize(z, n);
                if props.s_n > 0.0 {
                    last_bound_n = n;
                } else {
                    break;
                }
            }

            let props = engine.characterize(z, last_bound_n);
            eprintln!(
                "{:>4} {:>4} {:>5} {:>8.2} {:>8.2} {:>8.4}",
                z,
                last_bound_n,
                z + last_bound_n,
                props.s_n,
                props.s_2n,
                props.ba
            );

            drip_line.push((z, last_bound_n));
        }

        eprintln!("\nDrip line mapped for {} elements", drip_line.len());
        assert!(drip_line.len() > 10);
    }
}
