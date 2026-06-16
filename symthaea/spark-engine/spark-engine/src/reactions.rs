// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Fusion Reactions Library
//!
//! Comprehensive library of fusion reactions beyond D-D:
//! - D-T: Deuterium-Tritium (highest cross-section, 14.1 MeV neutrons)
//! - p-B11: Proton-Boron-11 (aneutronic, 3 alpha particles)
//! - D-He3: Deuterium-Helium-3 (aneutronic primary branch)
//! - Muon-Catalyzed: μ-catalyzed D-D and D-T
//!
//! Each reaction includes:
//! - Cross-section parameterization
//! - Gamow integration
//! - Q-value and product energies
//! - Practical considerations

use serde::{Deserialize, Serialize};

/// Fusion reaction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionReaction {
    /// D + D → He3 + n (2.45 MeV) or T + p (3.02 MeV)
    DD,
    /// D + T → He4 + n (14.1 MeV)
    DT,
    /// D + He3 → He4 + p (14.7 MeV proton)
    DHe3,
    /// p + B11 → 3 He4 (8.7 MeV total)
    PB11,
    /// p + Li6 → He4 + He3
    PLi6,
    /// p + Li7 → 2 He4
    PLi7,
    /// He3 + He3 → He4 + 2p
    He3He3,
}

impl FusionReaction {
    /// Q-value (energy release) in MeV.
    pub fn q_value_mev(&self) -> f64 {
        match self {
            Self::DD => 3.65,      // Average of n and p branches
            Self::DT => 17.6,     // He4 (3.5 MeV) + n (14.1 MeV)
            Self::DHe3 => 18.3,   // He4 (3.6 MeV) + p (14.7 MeV)
            Self::PB11 => 8.7,    // 3 × He4
            Self::PLi6 => 4.0,    // He4 + He3
            Self::PLi7 => 17.3,   // 2 × He4
            Self::He3He3 => 12.9, // He4 + 2p
        }
    }

    /// Astrophysical S-factor at zero energy S(0) in keV·barn.
    pub fn s_factor_kev_barn(&self) -> f64 {
        match self {
            Self::DD => 55.0,
            Self::DT => 1.1e4,    // Much higher - resonance enhanced
            Self::DHe3 => 5.9e3,
            Self::PB11 => 2.0e5,  // Very high but needs high energy
            Self::PLi6 => 3.0e3,
            Self::PLi7 => 60.0,
            Self::He3He3 => 5.0e3,
        }
    }

    /// Gamow energy E_G in keV (determines tunneling probability).
    pub fn gamow_energy_kev(&self) -> f64 {
        // E_G = (π α Z1 Z2)² × 2 μ c² where α ≈ 1/137
        // Simplified: E_G = 986.1 × Z1² × Z2² × μ (keV, with μ in amu)
        let (z1, z2, mu) = self.charges_and_mass();
        986.1 * ((z1 * z2) as f64).powi(2) * mu
    }

    /// Charges and reduced mass.
    fn charges_and_mass(&self) -> (u32, u32, f64) {
        match self {
            Self::DD => (1, 1, 1.0),         // D-D: Z=1,1, μ=1
            Self::DT => (1, 1, 1.2),         // D-T: Z=1,1, μ=1.2
            Self::DHe3 => (1, 2, 1.2),       // D-He3: Z=1,2, μ=1.2
            Self::PB11 => (1, 5, 0.92),      // p-B11: Z=1,5, μ=0.92
            Self::PLi6 => (1, 3, 0.86),      // p-Li6: Z=1,3, μ=0.86
            Self::PLi7 => (1, 3, 0.88),      // p-Li7: Z=1,3, μ=0.88
            Self::He3He3 => (2, 2, 1.5),     // He3-He3: Z=2,2, μ=1.5
        }
    }

    /// Whether this reaction produces neutrons.
    pub fn produces_neutrons(&self) -> bool {
        match self {
            Self::DD => true,   // 50% branch
            Self::DT => true,   // Primary product
            Self::DHe3 => false, // Aneutronic (primary)
            Self::PB11 => false, // Aneutronic
            Self::PLi6 => false,
            Self::PLi7 => false,
            Self::He3He3 => false,
        }
    }

    /// Primary neutron energy if applicable (MeV).
    pub fn neutron_energy_mev(&self) -> Option<f64> {
        match self {
            Self::DD => Some(2.45),
            Self::DT => Some(14.1),
            _ => None,
        }
    }

    /// Description of reaction products.
    pub fn products(&self) -> &'static str {
        match self {
            Self::DD => "He3 (0.82 MeV) + n (2.45 MeV) OR T (1.01 MeV) + p (3.02 MeV)",
            Self::DT => "He4 (3.5 MeV) + n (14.1 MeV)",
            Self::DHe3 => "He4 (3.6 MeV) + p (14.7 MeV)",
            Self::PB11 => "3 × He4 (2.9 MeV each)",
            Self::PLi6 => "He4 + He3",
            Self::PLi7 => "2 × He4",
            Self::He3He3 => "He4 + 2p",
        }
    }

    /// Practical considerations for this reaction.
    pub fn considerations(&self) -> Vec<&'static str> {
        match self {
            Self::DD => vec![
                "Fuel is abundant (seawater)",
                "Low cross-section at achievable temperatures",
                "Produces neutrons (radiation/activation)",
                "Tritium breeding from T branch",
            ],
            Self::DT => vec![
                "Highest cross-section of all fusion reactions",
                "Ignition possible at ~10 keV",
                "Requires tritium breeding (scarce fuel)",
                "14.1 MeV neutrons require heavy shielding",
                "Basis for ITER and most fusion reactor designs",
            ],
            Self::DHe3 => vec![
                "Aneutronic primary branch (reduced shielding)",
                "He3 is extremely scarce on Earth",
                "Potential lunar He3 mining",
                "Higher ignition temperature than D-T",
                "Side D-D reactions produce some neutrons",
            ],
            Self::PB11 => vec![
                "Truly aneutronic (no neutron shielding needed)",
                "Abundant fuels (hydrogen, boron)",
                "Very high ignition temperature (~300 keV)",
                "Low power density compared to D-T",
                "Direct energy conversion possible",
                "Bremsstrahlung losses significant",
            ],
            Self::PLi6 | Self::PLi7 => vec![
                "Aneutronic",
                "Lithium is abundant",
                "Lower cross-section than p-B11",
            ],
            Self::He3He3 => vec![
                "Aneutronic",
                "He3 scarcity limits practicality",
                "Important in stellar nucleosynthesis",
            ],
        }
    }
}

/// Detailed fusion reaction calculation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionResult {
    /// Reaction type
    pub reaction: FusionReaction,
    /// Temperature (keV)
    pub temperature_kev: f64,
    /// Reactivity <σv> (cm³/s)
    pub sigma_v_cm3_s: f64,
    /// Power density (W/cm³) at n = 10²⁰/cm³
    pub power_density_w_cm3: f64,
    /// Lawson criterion nτ (s/cm³) for ignition
    pub lawson_nt: f64,
    /// Estimated Q factor at given conditions
    pub q_estimate: f64,
    /// Whether ignition is possible at this temperature
    pub ignition_possible: bool,
}

/// Fusion reaction calculator.
pub struct ReactionCalculator;

impl ReactionCalculator {
    /// Calculate reaction rate for given temperature.
    pub fn calculate(reaction: FusionReaction, temperature_kev: f64) -> ReactionResult {
        let sigma_v = Self::reactivity(reaction, temperature_kev);

        // Power density at n = 10²⁰/cm³ (typical fusion plasma)
        let n = 1e20; // particles/cm³
        let q_mev = reaction.q_value_mev();
        let power_density = 0.25 * n * n * sigma_v * q_mev * 1.602e-13; // W/cm³

        // Lawson criterion: nτ_E > 3kT / (<σv> × Q)
        // Simplified estimate
        let lawson_nt = 3.0 * temperature_kev / (sigma_v * q_mev * 1e3);

        // Q estimate (very rough - depends on confinement)
        // Assume τ_E ~ 1s, n ~ 10²⁰
        let fusion_power = power_density;
        let loss_power = 3.0 * n * temperature_kev * 1.602e-16; // Thermal losses
        let q_estimate = if loss_power > 0.0 { fusion_power / loss_power } else { 0.0 };

        // Ignition threshold (Q → ∞, self-sustaining)
        let ignition_possible = q_estimate > 5.0; // Rough threshold

        ReactionResult {
            reaction,
            temperature_kev,
            sigma_v_cm3_s: sigma_v,
            power_density_w_cm3: power_density,
            lawson_nt,
            q_estimate,
            ignition_possible,
        }
    }

    /// Calculate reactivity <σv> using Bosch-Hale parameterization.
    pub fn reactivity(reaction: FusionReaction, t_kev: f64) -> f64 {
        match reaction {
            FusionReaction::DD => Self::sigma_v_dd(t_kev),
            FusionReaction::DT => Self::sigma_v_dt(t_kev),
            FusionReaction::DHe3 => Self::sigma_v_dhe3(t_kev),
            FusionReaction::PB11 => Self::sigma_v_pb11(t_kev),
            _ => Self::sigma_v_generic(reaction, t_kev),
        }
    }

    /// D-D reactivity (Bosch-Hale parameterization).
    fn sigma_v_dd(t_kev: f64) -> f64 {
        if t_kev < 0.001 { return 0.0; }

        // Bosch-Hale coefficients for D(d,n)He3
        let bg = 31.3970;
        let mc2 = 937814.0; // keV

        let theta = t_kev / (1.0 - t_kev * (0.0 + t_kev * (0.0 + t_kev * 0.0)) / mc2);
        let xi = (bg * bg / (4.0 * theta)).powf(1.0 / 3.0);

        let c = [
            5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0, -2.96400e-6, 0.0, 0.0
        ];

        let sigma_v = c[0] * theta.sqrt() * (-3.0 * xi).exp() / (mc2 * t_kev.powi(3))
            * (c[1] + t_kev * (c[2] + t_kev * (c[3] + t_kev * (c[4] + t_kev * (c[5] + t_kev * c[6])))));

        // Fallback to simpler formula if parameterization fails
        if sigma_v.is_nan() || sigma_v <= 0.0 {
            return Self::sigma_v_generic(FusionReaction::DD, t_kev);
        }

        sigma_v
    }

    /// D-T reactivity (Bosch-Hale parameterization).
    fn sigma_v_dt(t_kev: f64) -> f64 {
        if t_kev < 0.001 { return 0.0; }

        // Bosch-Hale coefficients for D(t,n)He4
        let bg = 34.3827;
        let mc2 = 1124656.0; // keV

        let theta = t_kev / (1.0 - t_kev * (0.0 + t_kev * (0.0 + t_kev * 0.0)) / mc2);
        let xi = (bg * bg / (4.0 * theta)).powf(1.0 / 3.0);

        let c = [
            1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5
        ];

        let sigma_v = c[0] * theta.sqrt() * (-3.0 * xi).exp() / (mc2 * t_kev.powi(3))
            * (c[1] + t_kev * (c[2] + t_kev * (c[3] + t_kev * (c[4] + t_kev * (c[5] + t_kev * c[6])))));

        if sigma_v.is_nan() || sigma_v <= 0.0 {
            return Self::sigma_v_generic(FusionReaction::DT, t_kev);
        }

        sigma_v
    }

    /// D-He3 reactivity.
    fn sigma_v_dhe3(t_kev: f64) -> f64 {
        if t_kev < 0.001 { return 0.0; }

        // Simplified parameterization
        let bg = 68.7508; // Higher Gamow energy due to Z=2
        let s0 = 5.9e3; // keV·barn

        let xi = (bg * bg / (4.0 * t_kev)).powf(1.0 / 3.0);
        let sigma_v = 3.7e-12 * s0 / t_kev.powf(2.0/3.0) * (-3.0 * xi).exp();

        sigma_v.max(0.0)
    }

    /// p-B11 reactivity.
    fn sigma_v_pb11(t_kev: f64) -> f64 {
        if t_kev < 1.0 { return 0.0; } // Needs high temperature

        // p-B11 has a resonance at ~150 keV
        // Simplified formula with resonance contribution
        let bg = 213.0; // High Gamow energy (Z=5)

        // Non-resonant contribution
        let xi = (bg * bg / (4.0 * t_kev)).powf(1.0 / 3.0);
        let non_res = 3.7e-12 * 2.0e5 / t_kev.powf(2.0/3.0) * (-3.0 * xi).exp();

        // Resonance at 148 keV (Breit-Wigner approximation)
        let e_res: f64 = 148.0; // keV
        let gamma: f64 = 10.0; // keV width
        let res_strength: f64 = 1.0e-15;
        let resonance = res_strength * gamma.powi(2) /
            ((t_kev - e_res).powi(2) + gamma.powi(2) / 4.0) *
            (-(e_res / t_kev).abs()).exp();

        (non_res + resonance).max(0.0)
    }

    /// Generic reactivity using Gamow formula.
    fn sigma_v_generic(reaction: FusionReaction, t_kev: f64) -> f64 {
        if t_kev < 0.001 { return 0.0; }

        let s0 = reaction.s_factor_kev_barn();
        let e_g = reaction.gamow_energy_kev();

        // Gamow peak: E_0 = (E_G × kT² / 4)^(1/3)
        let e0 = (e_g * t_kev * t_kev / 4.0).powf(1.0 / 3.0);

        // Width: Δ = 4/√3 × √(E_0 × kT)
        let delta = 4.0 / 3.0_f64.sqrt() * (e0 * t_kev).sqrt();

        // <σv> ≈ (8/πm)^(1/2) × S(E_0) × Δ/kT × exp(-3E_0/kT)
        // Simplified: <σv> ≈ 3.7×10⁻¹² × S₀ × τ⁻² × exp(-3τ) where τ = (E_G/4kT)^(1/3)
        let tau = (e_g / (4.0 * t_kev)).powf(1.0 / 3.0);

        3.7e-12 * s0 / t_kev.powf(2.0/3.0) * (-3.0 * tau).exp()
    }

    /// Find optimal temperature for a reaction (max <σv>/T² for ignition).
    pub fn optimal_temperature(reaction: FusionReaction) -> f64 {
        // Search for maximum reactivity
        let mut best_t = 10.0;
        let mut best_sv = 0.0;

        for t in (1..1000).map(|i| i as f64 * 0.5) {
            let sv = Self::reactivity(reaction, t);
            if sv > best_sv {
                best_sv = sv;
                best_t = t;
            }
        }

        best_t
    }

    /// Compare all reactions at a given temperature.
    pub fn compare_all(temperature_kev: f64) -> Vec<ReactionResult> {
        vec![
            Self::calculate(FusionReaction::DD, temperature_kev),
            Self::calculate(FusionReaction::DT, temperature_kev),
            Self::calculate(FusionReaction::DHe3, temperature_kev),
            Self::calculate(FusionReaction::PB11, temperature_kev),
        ]
    }
}

// ============================================================================
// Muon-Catalyzed Fusion
// ============================================================================

/// Muon-catalyzed fusion calculator.
///
/// In μCF, a muon replaces an electron in a hydrogen molecule,
/// bringing the nuclei ~200× closer (muon mass ≈ 207 × electron mass).
/// This dramatically increases tunneling probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuonCatalyzedFusion {
    /// Muon lifetime (μs)
    pub muon_lifetime_us: f64,
    /// Fusions per muon (cycling rate × sticking fraction)
    pub fusions_per_muon: f64,
    /// Energy per fusion (MeV)
    pub energy_per_fusion_mev: f64,
    /// Total energy per muon (MeV)
    pub energy_per_muon_mev: f64,
    /// Energy to produce muon (MeV)
    pub muon_production_cost_mev: f64,
    /// Net energy gain possible?
    pub energy_gain_possible: bool,
    /// Reaction type (D-D or D-T)
    pub reaction: FusionReaction,
}

impl MuonCatalyzedFusion {
    /// Calculate μCF for D-T (most favorable).
    pub fn dt() -> Self {
        let muon_lifetime_us = 2.2; // μs

        // μCF cycle time ~10⁻⁸ s in D-T mixture
        let cycle_time_us: f64 = 0.01;
        let max_cycles: f64 = muon_lifetime_us / cycle_time_us;

        // Sticking probability: muon sticks to alpha ~0.5% per cycle
        let sticking_prob: f64 = 0.005;
        let effective_cycles: f64 = 1.0 / sticking_prob; // ~200 cycles before sticking

        let fusions_per_muon = max_cycles.min(effective_cycles);
        let energy_per_fusion_mev = 17.6; // D-T Q-value
        let energy_per_muon_mev = fusions_per_muon * energy_per_fusion_mev;

        // Muon production requires ~5 GeV proton beam, ~3% efficiency
        let muon_production_cost_mev = 5000.0 / 0.03; // ~167 GeV effective

        Self {
            muon_lifetime_us,
            fusions_per_muon,
            energy_per_fusion_mev,
            energy_per_muon_mev,
            muon_production_cost_mev,
            energy_gain_possible: energy_per_muon_mev > muon_production_cost_mev,
            reaction: FusionReaction::DT,
        }
    }

    /// Calculate μCF for D-D.
    pub fn dd() -> Self {
        let muon_lifetime_us = 2.2;

        // D-D has lower sticking (no alpha produced initially)
        // but also lower cycle rate
        let cycle_time_us: f64 = 0.1; // Slower than D-T
        let max_cycles: f64 = muon_lifetime_us / cycle_time_us;

        // Lower sticking for D-D products
        let sticking_prob: f64 = 0.001;
        let effective_cycles: f64 = 1.0 / sticking_prob;

        let fusions_per_muon = max_cycles.min(effective_cycles);
        let energy_per_fusion_mev = 3.65;
        let energy_per_muon_mev = fusions_per_muon * energy_per_fusion_mev;

        let muon_production_cost_mev = 5000.0 / 0.03;

        Self {
            muon_lifetime_us,
            fusions_per_muon,
            energy_per_fusion_mev,
            energy_per_muon_mev,
            muon_production_cost_mev,
            energy_gain_possible: energy_per_muon_mev > muon_production_cost_mev,
            reaction: FusionReaction::DD,
        }
    }

    /// Analysis of μCF as energy source.
    pub fn energy_analysis(&self) -> MuonCatalysisAnalysis {
        let q_factor = self.energy_per_muon_mev / self.muon_production_cost_mev;

        let breakeven_fusions = (self.muon_production_cost_mev / self.energy_per_fusion_mev).ceil() as u32;

        let required_sticking_reduction = if q_factor < 1.0 {
            1.0 / q_factor
        } else {
            1.0
        };

        MuonCatalysisAnalysis {
            q_factor,
            breakeven_fusions,
            current_fusions: self.fusions_per_muon as u32,
            deficit_factor: if q_factor < 1.0 { 1.0 / q_factor } else { 1.0 },
            required_sticking_reduction,
            assessment: if q_factor > 1.0 {
                "Energy gain achieved!".to_string()
            } else if q_factor > 0.5 {
                format!("Close to breakeven - need {:.0}× more fusions/muon", 1.0/q_factor)
            } else {
                format!("Far from breakeven - need {:.0}× improvement", 1.0/q_factor)
            },
        }
    }
}

/// Analysis of muon-catalyzed fusion viability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuonCatalysisAnalysis {
    /// Q factor (fusion energy / production cost)
    pub q_factor: f64,
    /// Fusions needed per muon for breakeven
    pub breakeven_fusions: u32,
    /// Current fusions per muon
    pub current_fusions: u32,
    /// How far from breakeven
    pub deficit_factor: f64,
    /// Required reduction in sticking probability
    pub required_sticking_reduction: f64,
    /// Overall assessment
    pub assessment: String,
}

// ============================================================================
// Fusion Reactor Comparison
// ============================================================================

/// Compare different fusion approaches.
#[derive(Debug, Clone)]
pub struct FusionComparison {
    pub approaches: Vec<FusionApproach>,
    pub recommendation: String,
}

#[derive(Debug, Clone)]
pub struct FusionApproach {
    pub name: String,
    pub reaction: Option<FusionReaction>,
    pub temperature_kev: f64,
    pub sigma_v_cm3_s: f64,
    pub q_achievable: bool,
    pub fuel_availability: &'static str,
    pub neutron_hazard: &'static str,
    pub technology_readiness: u8, // 1-9 TRL
    pub challenges: Vec<&'static str>,
}

impl FusionComparison {
    /// Build comprehensive comparison.
    pub fn build() -> Self {
        let approaches = vec![
            // Magnetic confinement D-T (ITER approach)
            FusionApproach {
                name: "Tokamak D-T (ITER)".to_string(),
                reaction: Some(FusionReaction::DT),
                temperature_kev: 15.0,
                sigma_v_cm3_s: ReactionCalculator::reactivity(FusionReaction::DT, 15.0),
                q_achievable: true,
                fuel_availability: "Tritium scarce (breed from Li)",
                neutron_hazard: "Severe (14.1 MeV)",
                technology_readiness: 6,
                challenges: vec![
                    "Plasma instabilities",
                    "First wall materials",
                    "Tritium breeding ratio",
                    "Magnet technology",
                ],
            },
            // Inertial confinement (NIF approach)
            FusionApproach {
                name: "Inertial Confinement D-T (NIF)".to_string(),
                reaction: Some(FusionReaction::DT),
                temperature_kev: 50.0,
                sigma_v_cm3_s: ReactionCalculator::reactivity(FusionReaction::DT, 50.0),
                q_achievable: true,
                fuel_availability: "Tritium scarce",
                neutron_hazard: "Severe (14.1 MeV)",
                technology_readiness: 5,
                challenges: vec![
                    "Driver efficiency",
                    "Target fabrication",
                    "Repetition rate",
                    "Chamber survival",
                ],
            },
            // Aneutronic p-B11
            FusionApproach {
                name: "Aneutronic p-B11".to_string(),
                reaction: Some(FusionReaction::PB11),
                temperature_kev: 300.0,
                sigma_v_cm3_s: ReactionCalculator::reactivity(FusionReaction::PB11, 300.0),
                q_achievable: false, // Very difficult
                fuel_availability: "Abundant (H, B)",
                neutron_hazard: "Minimal",
                technology_readiness: 2,
                challenges: vec![
                    "Extreme temperature required",
                    "Bremsstrahlung losses",
                    "Low power density",
                    "No demonstrated ignition path",
                ],
            },
            // LCF (anomalous)
            FusionApproach {
                name: "Lattice Confinement (LCF)".to_string(),
                reaction: Some(FusionReaction::DD),
                temperature_kev: 0.026, // Room temperature
                sigma_v_cm3_s: 1e-50, // Essentially zero without anomaly
                q_achievable: false,
                fuel_availability: "Abundant (D from water)",
                neutron_hazard: "Moderate (2.45 MeV)",
                technology_readiness: 2,
                challenges: vec![
                    "40 orders of magnitude gap vs observations",
                    "Mechanism unknown",
                    "Reproducibility issues",
                    "No independent replication of NASA results",
                ],
            },
            // Muon-catalyzed
            FusionApproach {
                name: "Muon-Catalyzed Fusion".to_string(),
                reaction: Some(FusionReaction::DT),
                temperature_kev: 0.026, // Room temperature
                sigma_v_cm3_s: 1e-15, // Enhanced by muon
                q_achievable: false, // Sticking problem
                fuel_availability: "D-T mixture needed",
                neutron_hazard: "Severe (14.1 MeV)",
                technology_readiness: 4,
                challenges: vec![
                    "Alpha sticking limits cycles to ~150",
                    "Muon production costs ~5 GeV",
                    "Need ~300 fusions/muon for breakeven",
                ],
            },
        ];

        let recommendation = "D-T magnetic confinement (tokamak) remains the most \
            viable path to fusion energy, despite challenges. ITER aims to \
            demonstrate Q > 10 by 2035. Aneutronic approaches are attractive \
            but face severe physics barriers.".to_string();

        Self {
            approaches,
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dt_higher_than_dd() {
        let dt = ReactionCalculator::reactivity(FusionReaction::DT, 10.0);
        let dd = ReactionCalculator::reactivity(FusionReaction::DD, 10.0);
        assert!(dt > dd * 10.0); // D-T should be much higher
    }

    #[test]
    fn test_reactivity_increases_with_temp() {
        let sv_10 = ReactionCalculator::reactivity(FusionReaction::DT, 10.0);
        let sv_20 = ReactionCalculator::reactivity(FusionReaction::DT, 20.0);
        assert!(sv_20 > sv_10);
    }

    #[test]
    fn test_aneutronic_reactions() {
        assert!(!FusionReaction::PB11.produces_neutrons());
        assert!(!FusionReaction::DHe3.produces_neutrons());
        assert!(FusionReaction::DT.produces_neutrons());
    }

    #[test]
    fn test_muon_catalyzed() {
        let mcf_dt = MuonCatalyzedFusion::dt();
        assert!(mcf_dt.fusions_per_muon > 100.0);
        assert!(mcf_dt.fusions_per_muon < 300.0);

        let analysis = mcf_dt.energy_analysis();
        assert!(analysis.q_factor < 1.0); // Currently not viable
    }

    #[test]
    fn test_fusion_comparison() {
        let comparison = FusionComparison::build();
        assert!(comparison.approaches.len() >= 4);
    }

    #[test]
    fn test_q_values() {
        assert!(FusionReaction::DT.q_value_mev() > FusionReaction::DD.q_value_mev());
        assert!(FusionReaction::DHe3.q_value_mev() > FusionReaction::DT.q_value_mev());
    }
}
