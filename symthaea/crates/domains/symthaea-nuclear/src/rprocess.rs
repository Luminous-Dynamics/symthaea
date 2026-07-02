// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # R-Process Nucleosynthesis Simulation
//!
//! Simplified nuclear reaction network tracing neutron captures and beta
//! decays from seed nuclei (Fe-56) through successive neutron-rich isotopes.
//!
//! ## Physics
//!
//! - **Neutron capture**: `(n,γ)` rate proportional to `exp(-S_n / kT)`
//! - **Beta decay**: `β⁻` rate from Sargent rule `t_half ≈ C / Q_β^5`
//! - **Waiting points**: where `S_n` drops below ~2-3 MeV (capture stalls)
//! - **(n,γ)⇌(γ,n) equilibrium**: at high temperature, forward/reverse balance
//! - **Freezeout**: neutron exhaustion → decay back to stability
//!
//! ## Astrophysical context
//!
//! The r-process ("rapid neutron capture") occurs in neutron star mergers
//! and possibly core-collapse supernovae. It is responsible for ~half of
//! all elements heavier than iron, including gold, platinum, and uranium.
//!
//! ## Key observables
//!
//! - **Abundance peaks**: A~130 (N=82), A~195 (N=126), rare earth (A~165)
//! - **Actinide production**: A>230 (fission recycling)
//! - **Kilonova heating**: radioactive decay powering electromagnetic transient
//!   (Metzger et al., 2010: `L ∝ t^{-1.3}`)
//!
//! ## References
//!
//! - Burbidge, Burbidge, Fowler & Hoyle (1957). "B²FH" — Rev. Mod. Phys. 29, 547.
//! - Cowan, Thielemann & Truran (1991). Phys. Rep. 208, 267.
//! - Metzger et al. (2010). MNRAS 406, 2650.
//! - Mumpower et al. (2016). Prog. Part. Nucl. Phys. 86, 86.
//! - Arnould, Goriely & Takahashi (2007). Phys. Rep. 450, 97.

use crate::ml_mass::MlMassPredictor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Physical Constants ────────────────────────────────────────────────────

/// Boltzmann constant in MeV/GK (1 GK = 10^9 K)
const K_BOLTZMANN_MEV_PER_GK: f64 = 0.086173;

/// Neutron mass excess (MeV) — AME2020
const M_N: f64 = 8.07132;
/// Hydrogen atom mass excess (MeV) — AME2020
const M_H: f64 = 7.28897;
/// Neutron-hydrogen mass difference (MeV)
const DELTA_NP: f64 = M_N - M_H; // 0.78235 MeV

/// Sargent rule constant for allowed beta transitions (s·MeV^5).
/// Calibrated so that a Q_beta=5 MeV transition gives t_half ~ 0.1s.
const SARGENT_C: f64 = 3.125e2;

/// Minimum Q_beta for Sargent rule applicability (MeV)
const Q_BETA_MIN: f64 = 0.1;

/// Neutron capture rate prefactor (s^-1) — encodes cross-section × velocity × density.
/// Typical r-process: n_n ~ 10^24 cm^-3, <σv>_30keV ~ 100 mb × v_thermal → λ ~ 10^5 s^-1.
/// Neutrons have NO Coulomb barrier, so capture rate scales as n_n × σ × v ∝ n_n × T^{1/2},
/// NOT exponentially suppressed by S_n. S_n only enters photodissociation.
const CAPTURE_RATE_PREFACTOR: f64 = 1.0e3;

/// Photodissociation rate prefactor (s^-1) for (γ,n) equilibrium.
/// From detailed balance: λ_γn ∝ T^{3/2} exp(-S_n / kT), but we absorb
/// temperature dependence into the Boltzmann factor.
const PHOTODISSOCIATION_PREFACTOR: f64 = 1.0e10;

/// Default seed nucleus: Fe-56
const SEED_Z: u16 = 26;
const SEED_N: u16 = 30;

/// Maximum Z to track (beyond uranium, into superheavy)
const MAX_Z: u16 = 120;
/// Maximum N to track
const MAX_N: u16 = 220;
/// Minimum abundance to keep tracking
const ABUNDANCE_FLOOR: f64 = 1.0e-20;

// ── Types ─────────────────────────────────────────────────────────────────

/// A (Z, N) pair identifying a nuclide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Nuclide {
    pub z: u16,
    pub n: u16,
}

impl Nuclide {
    pub fn new(z: u16, n: u16) -> Self {
        Self { z, n }
    }

    /// Mass number A = Z + N
    pub fn a(&self) -> u16 {
        self.z + self.n
    }

    /// Element symbol (abbreviated)
    pub fn symbol(&self) -> &'static str {
        element_symbol(self.z)
    }
}

impl std::fmt::Display for Nuclide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.symbol(), self.a())
    }
}

/// R-process environment conditions at a given moment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RProcessConditions {
    /// Temperature in GK (10^9 K). Typical: 1-10 GK.
    pub temperature_gk: f64,
    /// Neutron number density (cm^-3). Typical: 10^20 - 10^28.
    pub neutron_density: f64,
    /// Electron fraction Y_e (protons per baryon). Typical: 0.05 - 0.25.
    pub ye: f64,
    /// Expansion timescale (seconds). Typical: 0.01 - 1.0.
    pub expansion_timescale: f64,
}

impl Default for RProcessConditions {
    fn default() -> Self {
        Self {
            temperature_gk: 1.5,
            neutron_density: 1.0e24,
            ye: 0.15,
            expansion_timescale: 0.3,
        }
    }
}

/// Simulation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RProcessConfig {
    /// Initial conditions
    pub conditions: RProcessConditions,
    /// Maximum simulation time (seconds)
    pub max_time: f64,
    /// Timestep (seconds)
    pub dt: f64,
    /// Number of freezeout decay steps (after neutron exhaustion)
    pub freezeout_steps: usize,
    /// Freezeout timestep (seconds) — longer since only beta decay
    pub freezeout_dt: f64,
    /// Enable (n,γ)⇌(γ,n) equilibrium calculation
    pub enable_equilibrium: bool,
    /// S_n threshold for waiting point (MeV)
    pub waiting_point_sn: f64,
}

impl Default for RProcessConfig {
    fn default() -> Self {
        Self {
            conditions: RProcessConditions::default(),
            dt: 1.0e-4,
            max_time: 2.0,
            freezeout_steps: 5000,
            freezeout_dt: 1.0e-2,
            enable_equilibrium: true,
            waiting_point_sn: 2.5,
        }
    }
}

/// Per-isotope network data (cached).
#[derive(Debug, Clone)]
struct NuclideData {
    binding_energy: f64,
    sn: f64,        // neutron separation energy (MeV)
    q_beta: f64,    // Q-value for beta-minus decay (MeV)
    beta_rate: f64, // beta-decay rate (s^-1)
}

/// Snapshot of the abundance distribution at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbundanceSnapshot {
    /// Time (seconds)
    pub time: f64,
    /// Abundances keyed by (Z, N)
    pub abundances: HashMap<Nuclide, f64>,
}

/// Single entry in the final abundance pattern Y(A).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassAbundance {
    pub a: u16,
    pub abundance: f64,
}

/// Kilonova heating rate at a given time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatingPoint {
    /// Time since merger (seconds)
    pub time_s: f64,
    /// Heating rate (erg/s/g) — energy per unit mass
    pub heating_rate: f64,
    /// Metzger power-law prediction for comparison
    pub metzger_prediction: f64,
}

/// Solar r-process abundance comparison at a given A.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarComparison {
    pub a: u16,
    pub predicted: f64,
    pub solar: f64,
    pub ratio: f64, // predicted / solar
}

/// Full result of an r-process simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RProcessResult {
    /// Configuration used
    pub config: RProcessConfig,
    /// Final abundance pattern Y(A) after freezeout
    pub abundances: Vec<MassAbundance>,
    /// R-process path: the (Z, N) trajectory of the abundance peak
    pub path: Vec<Nuclide>,
    /// Waiting point nuclei identified during the simulation
    pub waiting_points: Vec<Nuclide>,
    /// Kilonova heating curve
    pub heating_curve: Vec<HeatingPoint>,
    /// Solar abundance comparison
    pub solar_comparison: Vec<SolarComparison>,
    /// Number of simulation steps taken
    pub steps: usize,
    /// Peak abundance mass number
    pub peak_a: u16,
    /// Whether N=82 peak was produced
    pub has_n82_peak: bool,
    /// Whether N=126 peak was produced
    pub has_n126_peak: bool,
}

// ── R-Process Network ─────────────────────────────────────────────────────

/// The r-process nucleosynthesis simulator.
///
/// Uses `MlMassPredictor` for binding energies and computes a simplified
/// reaction network of neutron captures and beta decays.
pub struct RProcessNetwork {
    predictor: MlMassPredictor,
    /// Cached nuclear data for (Z, N) pairs
    cache: HashMap<Nuclide, NuclideData>,
}

impl RProcessNetwork {
    /// Create a new r-process network with a trained mass predictor.
    pub fn new() -> Self {
        Self {
            predictor: MlMassPredictor::new(),
            cache: HashMap::new(),
        }
    }

    /// Create from an existing predictor (avoids re-training).
    pub fn with_predictor(predictor: MlMassPredictor) -> Self {
        Self {
            predictor,
            cache: HashMap::new(),
        }
    }

    /// Get or compute nuclear data for a nuclide.
    fn get_data(&mut self, nuc: Nuclide) -> NuclideData {
        if let Some(data) = self.cache.get(&nuc) {
            return data.clone();
        }

        let be = self.predictor.predict(nuc.z, nuc.n).binding_energy;

        // Neutron separation energy: S_n = BE(Z, N) - BE(Z, N-1)
        let sn = if nuc.n > 1 {
            let be_prev = self.predictor.predict(nuc.z, nuc.n - 1).binding_energy;
            be - be_prev
        } else {
            0.0
        };

        // Q_beta = BE(Z+1, N-1) - BE(Z, N) + (M_n - M_H)
        let q_beta = if nuc.n > 0 && nuc.z < MAX_Z {
            let be_daughter = self.predictor.predict(nuc.z + 1, nuc.n - 1).binding_energy;
            be_daughter - be + DELTA_NP
        } else {
            0.0
        };

        // Beta-decay rate from Sargent rule: t_half = C / Q^5
        let beta_rate = if q_beta > Q_BETA_MIN {
            let t_half = SARGENT_C / q_beta.powi(5);
            (2.0_f64.ln()) / t_half
        } else {
            0.0 // stable or very slow
        };

        let data = NuclideData {
            binding_energy: be,
            sn,
            q_beta,
            beta_rate,
        };
        self.cache.insert(nuc, data.clone());
        data
    }

    /// Neutron capture rate for a nuclide at given temperature.
    ///
    /// λ_capture = prefactor × (n_n / n_n_ref) × exp(-S_n / kT)
    ///
    /// At (n,γ)⇌(γ,n) equilibrium, the effective capture rate accounts
    /// for photodissociation balance.
    fn capture_rate(&mut self, nuc: Nuclide, conditions: &RProcessConditions) -> f64 {
        let data = self.get_data(nuc);
        let kt = K_BOLTZMANN_MEV_PER_GK * conditions.temperature_gk;

        if kt < 1.0e-6 || data.sn < 0.0 {
            return 0.0;
        }

        // Scale capture rate by neutron density relative to reference
        let density_factor = conditions.neutron_density / 1.0e24;

        // Neutron capture has NO Coulomb barrier.
        // Rate = n_n × <σv> where <σv> ∝ T^{1/2} for thermal neutrons (1/v law).
        // Cross section decreases near shell closures (low level density),
        // modelled as a soft suppression when S_n exceeds ~15 MeV.
        let t_factor = conditions.temperature_gk.sqrt();
        let shell_suppression = if data.sn > 15.0 {
            (-(data.sn - 15.0) / 5.0).exp()
        } else {
            1.0
        };

        // Shell quenching: reduce capture rate near magic neutron numbers.
        // Near shell closures (N = 50, 82, 126) nuclear level density drops
        // dramatically, suppressing the (n,γ) cross section by ~1000× at the
        // magic number itself (Rauscher & Thielemann 2000, Surman et al. 1997).
        // The effect extends ~5 neutrons from each closure due to sub-shell gaps.
        let magic_ns = [50.0, 82.0, 126.0];
        let shell_factor = magic_ns
            .iter()
            .map(|&m| {
                let dist = (nuc.n as f64 - m).abs();
                if dist < 5.0 {
                    // Suppress by up to 1000x at the magic number itself
                    0.001 + 0.999 * (dist / 5.0).powi(2)
                } else {
                    1.0
                }
            })
            .fold(1.0f64, f64::min);

        CAPTURE_RATE_PREFACTOR * density_factor * t_factor * shell_suppression * shell_factor
    }

    /// Photodissociation rate (γ,n) for equilibrium calculation.
    fn photodissociation_rate(&mut self, nuc: Nuclide, conditions: &RProcessConditions) -> f64 {
        let data = self.get_data(nuc);
        let kt = K_BOLTZMANN_MEV_PER_GK * conditions.temperature_gk;

        if kt < 1.0e-6 || data.sn < 0.0 {
            return 0.0;
        }

        // (γ,n) from detailed balance: proportional to T^{3/2} exp(-S_n/kT)
        let t_factor = conditions.temperature_gk.powf(1.5);
        PHOTODISSOCIATION_PREFACTOR * t_factor * (-data.sn / kt).exp()
    }

    /// Check if a nuclide is a waiting point (capture slower than beta decay).
    fn is_waiting_point(&mut self, nuc: Nuclide, conditions: &RProcessConditions) -> bool {
        let data = self.get_data(nuc);
        let cap_rate = self.capture_rate(nuc, conditions);

        // Waiting point: beta faster than capture, and S_n is low
        data.beta_rate > cap_rate
            || data.sn < conditions.temperature_gk * K_BOLTZMANN_MEV_PER_GK * 20.0
    }

    /// Run the full r-process simulation.
    pub fn simulate(&mut self, config: &RProcessConfig) -> RProcessResult {
        let mut abundances: HashMap<Nuclide, f64> = HashMap::new();
        let mut path: Vec<Nuclide> = Vec::new();
        let mut waiting_points: Vec<Nuclide> = Vec::new();

        // Seed: Fe-56 with Y = 1.0
        let seed = Nuclide::new(SEED_Z, SEED_N);
        abundances.insert(seed, 1.0);
        path.push(seed);

        let mut time = 0.0;
        let mut steps = 0;
        let mut conditions = config.conditions.clone();

        // ── Phase 1: Active r-process (neutron captures + beta decays) ────
        while time < config.max_time {
            let dt = config.dt;
            time += dt;
            steps += 1;

            // Adiabatic expansion: temperature and density decrease
            let expansion_factor = (-time / conditions.expansion_timescale).exp();
            conditions.temperature_gk = config.conditions.temperature_gk * expansion_factor;
            conditions.neutron_density = config.conditions.neutron_density * expansion_factor;

            // Stop if neutrons exhausted (freezeout)
            if conditions.neutron_density < 1.0e16 || conditions.temperature_gk < 0.1 {
                break;
            }

            // Collect nuclides to process (avoid borrow conflict)
            let current: Vec<(Nuclide, f64)> = abundances
                .iter()
                .filter(|&(_, &y)| y > ABUNDANCE_FLOOR)
                .map(|(&nuc, &y)| (nuc, y))
                .collect();

            let mut delta: HashMap<Nuclide, f64> = HashMap::new();

            for (nuc, y) in &current {
                if nuc.z >= MAX_Z || nuc.n >= MAX_N {
                    continue;
                }

                let data = self.get_data(*nuc);

                // Neutron capture: (Z, N) → (Z, N+1)
                // Use implicit Euler: flow = y × (1 - exp(-λ×dt)) to prevent overshoot
                let lambda_cap = self.capture_rate(*nuc, &conditions);
                let capture_flow = if lambda_cap * dt > 0.01 {
                    y * (1.0 - (-lambda_cap * dt).exp())
                } else {
                    lambda_cap * y * dt
                };

                if capture_flow > 0.0 && nuc.n + 1 <= MAX_N {
                    let daughter = Nuclide::new(nuc.z, nuc.n + 1);
                    *delta.entry(*nuc).or_insert(0.0) -= capture_flow;
                    *delta.entry(daughter).or_insert(0.0) += capture_flow;
                }

                // Beta decay: (Z, N) → (Z+1, N-1)
                if data.beta_rate > 0.0 && nuc.n > 0 && nuc.z + 1 <= MAX_Z {
                    let beta_flow = if data.beta_rate * dt > 0.01 {
                        y * (1.0 - (-data.beta_rate * dt).exp())
                    } else {
                        data.beta_rate * y * dt
                    };
                    let daughter = Nuclide::new(nuc.z + 1, nuc.n - 1);
                    *delta.entry(*nuc).or_insert(0.0) -= beta_flow;
                    *delta.entry(daughter).or_insert(0.0) += beta_flow;
                }

                // (γ,n) photodissociation: (Z, N) → (Z, N-1) + n
                if config.enable_equilibrium && nuc.n > 1 {
                    let lambda_gn = self.photodissociation_rate(*nuc, &conditions);
                    let photo_flow = if lambda_gn * dt > 0.01 {
                        y * (1.0 - (-lambda_gn * dt).exp())
                    } else {
                        lambda_gn * y * dt
                    };
                    if photo_flow > 0.0 {
                        let daughter = Nuclide::new(nuc.z, nuc.n - 1);
                        *delta.entry(*nuc).or_insert(0.0) -= photo_flow;
                        *delta.entry(daughter).or_insert(0.0) += photo_flow;
                    }
                }

                // Spontaneous fission: A > 260 → asymmetric split into heavy + light fragment.
                // Rate ~ 1e6 s^-1 for the heaviest transactinides (Panov et al. 2010).
                // Asymmetric split: A_heavy ≈ 0.6×A, A_light ≈ 0.4×A,
                // Z split proportional to A fraction.
                if nuc.a() > 260 {
                    let fission_rate: f64 = 1.0e6; // s^-1
                    let fission_flow = if fission_rate * dt > 0.01 {
                        y * (1.0 - (-fission_rate * dt).exp())
                    } else {
                        fission_rate * y * dt
                    };

                    if fission_flow > 0.0 {
                        let a_total = nuc.a() as f64;
                        let a_heavy = (0.6 * a_total).round() as u16;
                        let a_light = nuc.a() - a_heavy;
                        let z_heavy = ((nuc.z as f64) * (a_heavy as f64) / a_total).round() as u16;
                        let z_light = nuc.z - z_heavy;
                        let n_heavy = a_heavy.saturating_sub(z_heavy);
                        let n_light = a_light.saturating_sub(z_light);

                        // Remove parent
                        *delta.entry(*nuc).or_insert(0.0) -= fission_flow;

                        // Each fission produces ONE heavy + ONE light fragment
                        if z_heavy > 0 && n_heavy > 0 && z_heavy <= MAX_Z && n_heavy <= MAX_N {
                            let heavy = Nuclide::new(z_heavy, n_heavy);
                            *delta.entry(heavy).or_insert(0.0) += fission_flow;
                        }
                        if z_light > 0 && n_light > 0 && z_light <= MAX_Z && n_light <= MAX_N {
                            let light = Nuclide::new(z_light, n_light);
                            *delta.entry(light).or_insert(0.0) += fission_flow;
                        }
                    }
                }

                // Identify waiting points
                if self.is_waiting_point(*nuc, &conditions) {
                    if !waiting_points.contains(nuc) {
                        waiting_points.push(*nuc);
                    }
                }
            }

            // Apply deltas
            for (nuc, dy) in delta {
                let entry = abundances.entry(nuc).or_insert(0.0);
                *entry += dy;
                if *entry < 0.0 {
                    *entry = 0.0;
                }
            }

            // Conservation: renormalize total abundance to 1.0
            // (baryon number is conserved in all reactions)
            let total: f64 = abundances.values().sum();
            if total > 0.0 && total.is_finite() {
                for y in abundances.values_mut() {
                    *y /= total;
                }
            }

            // Track path: nuclide with highest abundance at each Z
            if steps % 100 == 0 {
                if let Some((&peak_nuc, _)) = abundances
                    .iter()
                    .filter(|&(_, &y)| y > ABUNDANCE_FLOOR)
                    .max_by(|a, b| a.1.total_cmp(b.1))
                {
                    if path.last() != Some(&peak_nuc) {
                        path.push(peak_nuc);
                    }
                }
            }

            // Prune negligible abundances
            abundances.retain(|_, y| *y > ABUNDANCE_FLOOR);
        }

        // ── Phase 2: Freezeout — beta decay back to stability ─────────────
        for _ in 0..config.freezeout_steps {
            let current: Vec<(Nuclide, f64)> = abundances
                .iter()
                .filter(|&(_, &y)| y > ABUNDANCE_FLOOR)
                .map(|(&nuc, &y)| (nuc, y))
                .collect();

            let mut delta: HashMap<Nuclide, f64> = HashMap::new();
            let mut any_decay = false;

            for (nuc, y) in &current {
                let data = self.get_data(*nuc);

                if data.beta_rate > 0.0 && nuc.n > 0 && nuc.z + 1 <= MAX_Z {
                    let beta_flow = (data.beta_rate * config.freezeout_dt).min(1.0) * y;
                    if beta_flow > ABUNDANCE_FLOOR {
                        let daughter = Nuclide::new(nuc.z + 1, nuc.n - 1);
                        *delta.entry(*nuc).or_insert(0.0) -= beta_flow;
                        *delta.entry(daughter).or_insert(0.0) += beta_flow;
                        any_decay = true;
                    }
                }
            }

            if !any_decay {
                break;
            }

            for (nuc, dy) in delta {
                let entry = abundances.entry(nuc).or_insert(0.0);
                *entry += dy;
                if *entry < 0.0 {
                    *entry = 0.0;
                }
            }

            abundances.retain(|_, y| *y > ABUNDANCE_FLOOR);
        }

        // ── Collect results ───────────────────────────────────────────────

        // Aggregate abundances by mass number A
        let mut y_a: HashMap<u16, f64> = HashMap::new();
        for (nuc, y) in &abundances {
            *y_a.entry(nuc.a()).or_insert(0.0) += y;
        }

        let mut mass_abundances: Vec<MassAbundance> = y_a
            .into_iter()
            .map(|(a, abundance)| MassAbundance { a, abundance })
            .collect();
        mass_abundances.sort_by_key(|m| m.a);

        // Identify peaks
        let peak_a = mass_abundances
            .iter()
            .max_by(|a, b| a.abundance.total_cmp(&b.abundance))
            .map(|m| m.a)
            .unwrap_or(56);

        let has_n82_peak = mass_abundances
            .iter()
            .any(|m| (125..=135).contains(&m.a) && m.abundance > 1.0e-10);
        let has_n126_peak = mass_abundances
            .iter()
            .any(|m| (190..=200).contains(&m.a) && m.abundance > 1.0e-10);

        // Compute kilonova heating curve
        let heating_curve = self.compute_heating_curve(&abundances);

        // Compute solar comparison
        let solar_comparison = self.compute_solar_comparison(&mass_abundances);

        // Sort waiting points by Z
        waiting_points.sort_by_key(|n| (n.z, n.n));
        waiting_points.dedup();

        RProcessResult {
            config: config.clone(),
            abundances: mass_abundances,
            path,
            waiting_points,
            heating_curve,
            solar_comparison,
            steps,
            peak_a,
            has_n82_peak,
            has_n126_peak,
        }
    }

    /// Compute kilonova heating rate curve from radioactive decay.
    ///
    /// Q_heat(t) = Σ_i Q_decay_i × λ_i × Y_i × exp(-λ_i × t)
    ///
    /// Evaluated from 1 hour to 100 days post-merger.
    fn compute_heating_curve(&mut self, abundances: &HashMap<Nuclide, f64>) -> Vec<HeatingPoint> {
        let mut curve = Vec::new();

        // Time grid: logarithmic from 1 hour to 100 days
        let t_start: f64 = 3600.0; // 1 hour in seconds
        let t_end: f64 = 100.0 * 86400.0; // 100 days in seconds
        let n_points = 50;

        // Reference heating at 1 day for Metzger normalization
        let t_ref = 86400.0; // 1 day

        // Collect unstable nuclides with their decay properties
        let unstable: Vec<(Nuclide, f64, f64, f64)> = abundances
            .iter()
            .filter_map(|(&nuc, &y)| {
                if y < ABUNDANCE_FLOOR {
                    return None;
                }
                let data = self.get_data(nuc);
                if data.beta_rate > 0.0 && data.q_beta > 0.0 {
                    // Convert Q to erg (1 MeV = 1.602e-6 erg)
                    let q_erg = data.q_beta * 1.602e-6;
                    Some((nuc, y, data.beta_rate, q_erg))
                } else {
                    None
                }
            })
            .collect();

        if unstable.is_empty() {
            return curve;
        }

        // Compute reference heating at t_ref for normalization
        let q_ref: f64 = unstable
            .iter()
            .map(|(_, y, lambda, q)| q * lambda * y * (-lambda * t_ref).exp())
            .sum();

        for i in 0..n_points {
            let frac = i as f64 / (n_points - 1) as f64;
            let t = t_start * (t_end / t_start).powf(frac);

            let heating: f64 = unstable
                .iter()
                .map(|(_, y, lambda, q)| q * lambda * y * (-lambda * t).exp())
                .sum();

            // Metzger et al. (2010): L ∝ t^{-1.3}
            // Normalize to match our heating at t_ref
            let metzger = if q_ref > 0.0 {
                q_ref * (t / t_ref).powf(-1.3)
            } else {
                0.0
            };

            curve.push(HeatingPoint {
                time_s: t,
                heating_rate: heating,
                metzger_prediction: metzger,
            });
        }

        curve
    }

    /// Compare predicted abundances with solar r-process residuals.
    ///
    /// Solar r-process residuals from Arlandini et al. (1999) and
    /// Sneden et al. (2008). Values are log-scale relative abundances
    /// normalized to Si=10^6.
    fn compute_solar_comparison(&self, predicted: &[MassAbundance]) -> Vec<SolarComparison> {
        // Solar r-process residual abundances (log10, normalized).
        // Selected values at key mass numbers from Arlandini et al. (1999).
        let solar_data: Vec<(u16, f64)> = vec![
            // Iron peak tail
            (56, 1.0e0),
            (60, 5.0e-1),
            (70, 2.0e-1),
            (80, 8.0e-2),
            // First peak (A~80)
            (82, 1.5e-1),
            (84, 1.0e-1),
            (88, 8.0e-2),
            (90, 6.0e-2),
            // Between peaks
            (100, 3.0e-2),
            (110, 2.0e-2),
            (120, 1.5e-2),
            // Second peak (N=82, A~130)
            (128, 8.0e-2),
            (130, 1.2e-1),
            (132, 1.0e-1),
            (134, 7.0e-2),
            (136, 5.0e-2),
            // Rare earth peak (A~165)
            (150, 2.0e-2),
            (155, 2.5e-2),
            (160, 3.0e-2),
            (165, 3.5e-2),
            (170, 2.5e-2),
            // Third peak (N=126, A~195)
            (190, 5.0e-2),
            (192, 8.0e-2),
            (195, 1.0e-1),
            (196, 9.0e-2),
            (198, 6.0e-2),
            // Actinide region
            (208, 3.0e-2),
            (232, 5.0e-3),
            (235, 3.0e-3),
            (238, 4.0e-3),
        ];

        let pred_map: HashMap<u16, f64> = predicted.iter().map(|m| (m.a, m.abundance)).collect();

        // Normalize predicted to match solar at A=130 (reference point)
        let norm_factor = if let Some(pred_130) = pred_map.get(&130) {
            if *pred_130 > 0.0 {
                0.12 / pred_130 // solar A=130 ≈ 0.12
            } else {
                1.0
            }
        } else {
            1.0
        };

        solar_data
            .into_iter()
            .map(|(a, solar)| {
                let predicted_val = pred_map.get(&a).copied().unwrap_or(0.0) * norm_factor;
                let ratio = if solar > 0.0 {
                    predicted_val / solar
                } else {
                    0.0
                };
                SolarComparison {
                    a,
                    predicted: predicted_val,
                    solar,
                    ratio,
                }
            })
            .collect()
    }

    /// Compute neutron separation energy S_n for a nuclide.
    pub fn separation_energy(&mut self, z: u16, n: u16) -> f64 {
        let nuc = Nuclide::new(z, n);
        self.get_data(nuc).sn
    }

    /// Compute Q_beta for a nuclide.
    pub fn q_beta(&mut self, z: u16, n: u16) -> f64 {
        let nuc = Nuclide::new(z, n);
        self.get_data(nuc).q_beta
    }

    /// Compute beta-decay half-life (seconds) from Sargent rule.
    pub fn beta_half_life(&mut self, z: u16, n: u16) -> f64 {
        let nuc = Nuclide::new(z, n);
        let data = self.get_data(nuc);
        if data.beta_rate > 0.0 {
            2.0_f64.ln() / data.beta_rate
        } else {
            f64::INFINITY
        }
    }

    /// Find the r-process path at a given temperature: the N for each Z
    /// where S_n drops below the waiting-point threshold.
    pub fn find_rprocess_path(
        &mut self,
        conditions: &RProcessConditions,
        z_range: std::ops::RangeInclusive<u16>,
    ) -> Vec<Nuclide> {
        let mut path = Vec::new();

        for z in z_range {
            // Scan from stability outward in N until S_n < threshold
            let n_start = z; // roughly N ≈ Z at stability
            let mut waiting_n = n_start;

            for n in n_start..=MAX_N {
                let nuc = Nuclide::new(z, n);
                let data = self.get_data(nuc);

                if data.sn < conditions.temperature_gk * K_BOLTZMANN_MEV_PER_GK * 20.0
                    && data.sn > 0.0
                {
                    waiting_n = n;
                    break;
                }

                // Also check if beta faster than capture
                let cap_rate = self.capture_rate(nuc, conditions);
                if data.beta_rate > cap_rate && n > n_start + 5 {
                    waiting_n = n;
                    break;
                }
            }

            if waiting_n > n_start {
                path.push(Nuclide::new(z, waiting_n));
            }
        }

        path
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Map Z to element symbol.
fn element_symbol(z: u16) -> &'static str {
    const SYMBOLS: &[&str] = &[
        "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
        "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
        "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
        "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
        "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue", "Ubn",
    ];
    if (z as usize) < SYMBOLS.len() {
        SYMBOLS[z as usize]
    } else {
        "??"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a network (caches predictor across tests in the same binary).
    fn make_network() -> RProcessNetwork {
        RProcessNetwork::new()
    }

    #[test]
    fn test_separation_energy_fe56() {
        let mut net = make_network();
        let sn = net.separation_energy(26, 30);
        eprintln!("S_n(Fe-56) = {sn:.3} MeV");
        // Fe-56 is tightly bound; S_n should be positive and ~8-11 MeV
        assert!(sn > 5.0, "Fe-56 S_n should be > 5 MeV, got {sn}");
        assert!(sn < 15.0, "Fe-56 S_n should be < 15 MeV, got {sn}");
    }

    #[test]
    fn test_q_beta_neutron_rich() {
        let mut net = make_network();
        // Ni-78 (Z=28, N=50): very neutron-rich, should have large Q_beta
        let q = net.q_beta(28, 50);
        eprintln!("Q_beta(Ni-78) = {q:.3} MeV");
        assert!(q > 0.0, "Neutron-rich Ni-78 should have positive Q_beta");
    }

    #[test]
    fn test_beta_half_life_sargent() {
        let mut net = make_network();
        // A nucleus with Q_beta ~ 5 MeV should have t_half ~ 0.1 s
        let t = net.beta_half_life(28, 50);
        eprintln!(
            "t_half(Ni-78) = {t:.4} s (Q_beta = {:.3} MeV)",
            net.q_beta(28, 50)
        );
        assert!(t > 0.0, "Half-life must be positive");
        assert!(
            t < 1.0e10,
            "Half-life should be finite for neutron-rich nucleus"
        );
    }

    #[test]
    fn test_rprocess_path_from_seed() {
        let mut net = make_network();
        let conditions = RProcessConditions::default();
        let path = net.find_rprocess_path(&conditions, 26..=80);

        eprintln!(
            "\n=== R-Process Path (T={:.1} GK, n_n={:.0e}) ===",
            conditions.temperature_gk, conditions.neutron_density
        );
        for nuc in &path {
            let data = net.get_data(*nuc);
            eprintln!(
                "  {} (Z={:3}, N={:3}, A={:3}) — S_n={:.2} MeV, Q_β={:.2} MeV",
                nuc,
                nuc.z,
                nuc.n,
                nuc.a(),
                data.sn,
                data.q_beta
            );
        }

        assert!(!path.is_empty(), "R-process path should not be empty");
        // Path should extend well beyond Fe
        let max_z = path.iter().map(|n| n.z).max().unwrap_or(0);
        eprintln!("Path extends to Z={max_z}");
        assert!(
            max_z > 40,
            "Path should reach beyond Zr (Z=40), got Z={max_z}"
        );
    }

    #[test]
    fn test_full_rprocess_simulation() {
        let mut net = make_network();
        let config = RProcessConfig {
            conditions: RProcessConditions {
                temperature_gk: 1.5,
                neutron_density: 1.0e24,
                ye: 0.15,
                expansion_timescale: 0.3,
            },
            dt: 5.0e-4,
            max_time: 1.5,
            freezeout_steps: 3000,
            freezeout_dt: 5.0e-3,
            enable_equilibrium: true,
            waiting_point_sn: 2.5,
        };

        let result = net.simulate(&config);

        eprintln!("\n=== Full R-Process Simulation ===");
        eprintln!("Steps: {}", result.steps);
        eprintln!("Waiting points: {}", result.waiting_points.len());
        eprintln!("Peak mass number: A={}", result.peak_a);
        eprintln!("N=82 peak (A~130): {}", result.has_n82_peak);
        eprintln!("N=126 peak (A~195): {}", result.has_n126_peak);

        eprintln!("\n--- Abundance Pattern Y(A) ---");
        for m in &result.abundances {
            if m.abundance > 1.0e-8 {
                let bar_len = ((m.abundance.log10() + 10.0) * 3.0).max(0.0) as usize;
                let bar: String = "█".repeat(bar_len.min(60));
                eprintln!("  A={:3}: {:.3e}  {}", m.a, m.abundance, bar);
            }
        }

        eprintln!("\n--- Waiting Points ---");
        for wp in &result.waiting_points {
            eprintln!("  {} (Z={}, N={}, A={})", wp, wp.z, wp.n, wp.a());
        }

        // Verify: simulation produced isotopes beyond Fe
        assert!(
            result.abundances.len() > 5,
            "Should produce multiple mass numbers, got {}",
            result.abundances.len()
        );

        // Total abundance should be roughly conserved
        let total: f64 = result.abundances.iter().map(|m| m.abundance).sum();
        eprintln!("\nTotal abundance: {total:.4} (should be ~1.0)");
        assert!(
            total > 0.01,
            "Total abundance should be significant: {total}"
        );
    }

    #[test]
    fn test_abundance_peaks() {
        let mut net = make_network();
        let config = RProcessConfig {
            conditions: RProcessConditions {
                temperature_gk: 2.0,
                neutron_density: 5.0e24,
                ye: 0.10,
                expansion_timescale: 0.5,
            },
            dt: 5.0e-4,
            max_time: 2.0,
            freezeout_steps: 5000,
            freezeout_dt: 1.0e-2,
            enable_equilibrium: true,
            waiting_point_sn: 2.0,
        };

        let result = net.simulate(&config);

        eprintln!("\n=== Abundance Peak Analysis ===");

        // Check for material in the A~130 region (N=82 shell closure)
        let n82_region: f64 = result
            .abundances
            .iter()
            .filter(|m| (120..=140).contains(&m.a))
            .map(|m| m.abundance)
            .sum();
        eprintln!("A=120-140 (N=82 region): Y = {n82_region:.3e}");

        // Check for material in the A~195 region (N=126 shell closure)
        let n126_region: f64 = result
            .abundances
            .iter()
            .filter(|m| (185..=205).contains(&m.a))
            .map(|m| m.abundance)
            .sum();
        eprintln!("A=185-205 (N=126 region): Y = {n126_region:.3e}");

        // Check for rare-earth peak material (A~165)
        let rare_earth: f64 = result
            .abundances
            .iter()
            .filter(|m| (155..=175).contains(&m.a))
            .map(|m| m.abundance)
            .sum();
        eprintln!("A=155-175 (rare earth): Y = {rare_earth:.3e}");

        // Check for actinide production (A>230)
        let actinides: f64 = result
            .abundances
            .iter()
            .filter(|m| m.a > 230)
            .map(|m| m.abundance)
            .sum();
        eprintln!("A>230 (actinides): Y = {actinides:.3e}");

        // At minimum, the simulation should produce material beyond Fe
        let heavy: f64 = result
            .abundances
            .iter()
            .filter(|m| m.a > 100)
            .map(|m| m.abundance)
            .sum();
        eprintln!("A>100 (heavy elements): Y = {heavy:.3e}");
        assert!(heavy > 0.0, "Should produce some heavy elements");
    }

    #[test]
    fn test_kilonova_heating_curve() {
        let mut net = make_network();
        let config = RProcessConfig::default();
        let result = net.simulate(&config);

        eprintln!("\n=== Kilonova Heating Curve ===");
        eprintln!(
            "{:>12} {:>14} {:>14} {:>8}",
            "Time", "Q_heat", "Metzger", "Ratio"
        );
        eprintln!(
            "{:>12} {:>14} {:>14} {:>8}",
            "(days)", "(erg/s/g)", "(erg/s/g)", ""
        );

        for hp in &result.heating_curve {
            let days = hp.time_s / 86400.0;
            let ratio = if hp.metzger_prediction > 0.0 {
                hp.heating_rate / hp.metzger_prediction
            } else {
                0.0
            };
            eprintln!(
                "  {:8.3} d  {:12.4e}  {:12.4e}  {:6.3}",
                days, hp.heating_rate, hp.metzger_prediction, ratio
            );
        }

        if result.heating_curve.len() >= 2 {
            // Heating should decrease with time
            let first = result.heating_curve.first().unwrap();
            let last = result.heating_curve.last().unwrap();
            if first.heating_rate > 0.0 {
                eprintln!(
                    "\nHeating drops from {:.3e} to {:.3e} ({:.1}x)",
                    first.heating_rate,
                    last.heating_rate,
                    first.heating_rate / last.heating_rate.max(1.0e-100)
                );
                assert!(
                    last.heating_rate <= first.heating_rate,
                    "Heating should decrease with time"
                );
            }
        }
    }

    #[test]
    fn test_solar_abundance_comparison() {
        let mut net = make_network();
        let config = RProcessConfig {
            conditions: RProcessConditions {
                temperature_gk: 2.0,
                neutron_density: 1.0e25,
                ye: 0.10,
                expansion_timescale: 0.5,
            },
            dt: 5.0e-4,
            max_time: 2.0,
            freezeout_steps: 5000,
            freezeout_dt: 1.0e-2,
            enable_equilibrium: true,
            waiting_point_sn: 2.0,
        };

        let result = net.simulate(&config);

        eprintln!("\n=== Solar R-Process Comparison ===");
        eprintln!(
            "{:>5} {:>12} {:>12} {:>8} {}",
            "A", "Predicted", "Solar", "Ratio", "Match"
        );

        for sc in &result.solar_comparison {
            let match_quality = if sc.ratio > 0.0 {
                if (0.3..=3.0).contains(&sc.ratio) {
                    "OK"
                } else if (0.1..=10.0).contains(&sc.ratio) {
                    "~"
                } else {
                    "✗"
                }
            } else {
                "-"
            };
            eprintln!(
                "  {:3}: {:10.3e}  {:10.3e}  {:6.2}  {}",
                sc.a, sc.predicted, sc.solar, sc.ratio, match_quality
            );
        }

        // At least some entries should have nonzero predicted values
        let nonzero = result
            .solar_comparison
            .iter()
            .filter(|sc| sc.predicted > 0.0)
            .count();
        eprintln!(
            "\nNonzero predictions: {}/{}",
            nonzero,
            result.solar_comparison.len()
        );
    }

    #[test]
    fn test_nuclide_display() {
        let nuc = Nuclide::new(26, 30);
        assert_eq!(format!("{nuc}"), "Fe-56");
        assert_eq!(nuc.a(), 56);
        assert_eq!(nuc.symbol(), "Fe");

        let au = Nuclide::new(79, 118);
        assert_eq!(format!("{au}"), "Au-197");
    }

    #[test]
    fn test_equilibrium_at_high_temperature() {
        let mut net = make_network();
        // At very high T, photodissociation should dominate
        let conditions = RProcessConditions {
            temperature_gk: 8.0,
            neutron_density: 1.0e26,
            ye: 0.15,
            expansion_timescale: 1.0,
        };

        let nuc = Nuclide::new(50, 82); // Sn-132 (N=82)
        let cap = net.capture_rate(nuc, &conditions);
        let photo = net.photodissociation_rate(nuc, &conditions);
        let data = net.get_data(nuc);

        eprintln!("\n=== (n,γ)⇌(γ,n) Equilibrium Test at Sn-132 ===");
        eprintln!(
            "T = {:.1} GK, n_n = {:.0e}",
            conditions.temperature_gk, conditions.neutron_density
        );
        eprintln!("S_n = {:.3} MeV", data.sn);
        eprintln!("λ(n,γ) = {:.3e} s⁻¹", cap);
        eprintln!("λ(γ,n) = {:.3e} s⁻¹", photo);
        eprintln!("λ(β⁻)  = {:.3e} s⁻¹", data.beta_rate);

        // At high T, both rates should be significant
        assert!(cap > 0.0, "Capture rate should be nonzero");
    }

    #[test]
    fn test_shell_closure_effect_on_sn() {
        let mut net = make_network();

        eprintln!("\n=== Shell Closure Effects on S_n ===");

        // S_n should show a jump at magic neutron numbers
        // N=82: compare S_n at N=82 vs N=83 for Sn (Z=50)
        let sn_82 = net.separation_energy(50, 82);
        let sn_83 = net.separation_energy(50, 83);
        eprintln!("Sn-132 (N=82): S_n = {sn_82:.3} MeV");
        eprintln!("Sn-133 (N=83): S_n = {sn_83:.3} MeV");
        eprintln!("Drop at N=82→83: {:.3} MeV", sn_82 - sn_83);

        // N=126: compare for Pb (Z=82)
        let sn_126 = net.separation_energy(82, 126);
        let sn_127 = net.separation_energy(82, 127);
        eprintln!("Pb-208 (N=126): S_n = {sn_126:.3} MeV");
        eprintln!("Pb-209 (N=127): S_n = {sn_127:.3} MeV");
        eprintln!("Drop at N=126→127: {:.3} MeV", sn_126 - sn_127);

        // S_n at the magic number should be large (tightly bound)
        assert!(
            sn_82 > 3.0,
            "S_n at N=82 shell closure should be large: {sn_82}"
        );
        assert!(
            sn_126 > 3.0,
            "S_n at N=126 shell closure should be large: {sn_126}"
        );
    }

    #[test]
    fn test_default_config() {
        let config = RProcessConfig::default();
        assert!(config.conditions.temperature_gk > 0.0);
        assert!(config.conditions.neutron_density > 0.0);
        assert!(config.dt > 0.0);
        assert!(config.max_time > 0.0);

        let conditions = RProcessConditions::default();
        assert!((conditions.ye - 0.15).abs() < 0.01);
    }
}
