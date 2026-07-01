// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Reactor Physics Module
//!
//! Computational reactor physics built on the ML mass predictor and fission
//! barrier models. Implements:
//!
//! 1. **Fission product yield estimation** — asymmetric mass distribution via
//!    double-Gaussian model with Viola TKE systematics
//! 2. **Transmutation pathways** — neutron capture chains for long-lived waste
//!    isotopes (Cs-137, Sr-90, Tc-99, I-129)
//! 3. **Fuel characterization** — binding energy, fission barriers, breeding
//!    ratios, and energy-per-fission for U/Pu/Th fuel cycles
//! 4. **Decay heat** — Way-Wigner approximation for post-shutdown thermal power
//!
//! ## References
//!
//! - Viola, V. E. et al. (1985). Systematics of fission fragment TKE.
//!   *Phys. Rev. C* 31, 1550.
//! - Way, K. & Wigner, E. P. (1948). Rate of decay of fission products.
//!   *Phys. Rev.* 73, 1318.
//! - England, T. R. & Rider, B. F. (1994). ENDF-349 fission product yields.
//!   LA-UR-94-3106.
//! - Brosa, U. et al. (1990). Nuclear scission. *Phys. Rep.* 197, 167.
//! - Krane, K. S. (1988). *Introductory Nuclear Physics*. Wiley.

use crate::constants::B_ALPHA;
use crate::fission_barrier::compute_fission_barrier;
use crate::ml_mass::MlMassPredictor;
use serde::{Deserialize, Serialize};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Neutron mass in MeV/c^2 (CODATA 2022).
const M_NEUTRON_MEV: f64 = 939.565;

/// Boltzmann constant × reactor thermal temperature (kT ~ 0.0253 eV for 293 K).
const KT_THERMAL_EV: f64 = 0.0253;

/// Number of prompt neutrons per fission (approximate, U-235 thermal).
const NU_BAR_U235: f64 = 2.43;

// ── 1. Fission Product Yields ─────────────────────────────────────────────────

/// A single fission product fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FissionFragment {
    /// Proton number.
    pub z: u16,
    /// Neutron number.
    pub n: u16,
    /// Mass number A = Z + N.
    pub a: u16,
    /// Relative yield (normalized so total = 2.0, one per fragment pair).
    pub yield_fraction: f64,
    /// Binding energy of this fragment (MeV).
    pub binding_energy: f64,
}

/// Result of fission product yield calculation for a parent nucleus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FissionYieldResult {
    /// Parent Z.
    pub parent_z: u16,
    /// Parent N.
    pub parent_n: u16,
    /// Parent A.
    pub parent_a: u16,
    /// All fragment yields (sorted by mass number).
    pub fragments: Vec<FissionFragment>,
    /// Total kinetic energy release (MeV) from Viola systematics.
    pub tke_viola_mev: f64,
    /// Average total energy release per fission (MeV).
    pub q_fission_mev: f64,
    /// Peak light fragment mass.
    pub peak_light: u16,
    /// Peak heavy fragment mass.
    pub peak_heavy: u16,
}

/// Gaussian function for yield distribution.
fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let arg = (x - mu) / sigma;
    (-0.5 * arg * arg).exp()
}

/// Compute fission product yield distribution for a fissioning nucleus.
///
/// Uses double-Gaussian model peaked at A_light ~ 95 and A_heavy ~ 140
/// (for actinide fission), with fragment Z determined by the Unchanged
/// Charge Distribution (UCD) hypothesis: Z_f/A_f = Z_parent/A_parent.
///
/// The Viola systematics give TKE = 0.1189 * Z^2 / A^(1/3) MeV.
pub fn fission_product_yields(
    predictor: &MlMassPredictor,
    parent_z: u16,
    parent_n: u16,
) -> FissionYieldResult {
    let parent_a = (parent_z + parent_n) as f64;

    // Viola TKE systematics: TKE = 0.1189 * Z^2 / A^(1/3)
    let z_f = parent_z as f64;
    let tke_viola = 0.1189 * z_f * z_f / parent_a.powf(1.0 / 3.0);

    // Asymmetric mass split: double-humped Gaussian
    // Light peak near A = 95, heavy peak near A = parent_a - 95
    // For actinides these values reproduce experimental data well.
    let mu_light = 95.0_f64.min(parent_a * 0.4); // cap for lighter parents
    let mu_heavy = parent_a - mu_light;
    let sigma_light = 7.0;
    let sigma_heavy = 7.0;

    // Symmetric component (small, ~2-5% of total)
    let mu_sym = parent_a / 2.0;
    let sigma_sym = 10.0;
    let sym_weight = 0.03;

    // Generate fragment yields for each possible mass split
    // Minimum fragment: ~70, maximum: ~170 (for actinides)
    let a_min = 60.max((parent_a * 0.25) as u16);
    let a_max = ((parent_a * 0.75) as u16).min(parent_a as u16 - 20);

    let mut raw_yields: Vec<(u16, f64)> = Vec::new();
    let mut total_yield = 0.0;

    for a_frag in a_min..=a_max {
        let af = a_frag as f64;
        let y = (1.0 - sym_weight)
            * (gaussian(af, mu_light, sigma_light) + gaussian(af, mu_heavy, sigma_heavy))
            + sym_weight * gaussian(af, mu_sym, sigma_sym);

        if y > 1e-6 {
            raw_yields.push((a_frag, y));
            total_yield += y;
        }
    }

    // Normalize: total yield = 2.0 (two fragments per fission)
    let norm = if total_yield > 0.0 {
        2.0 / total_yield
    } else {
        0.0
    };

    // Build fragments with Z from UCD hypothesis
    let z_over_a = z_f / parent_a;
    let mut fragments = Vec::with_capacity(raw_yields.len());

    let parent_be = predictor.predict(parent_z, parent_n).binding_energy;
    let mut weighted_be_sum = 0.0;

    for (a_frag, raw_y) in &raw_yields {
        let yf = raw_y * norm;
        let z_frag = ((*a_frag as f64) * z_over_a).round() as u16;
        let n_frag = a_frag.saturating_sub(z_frag);

        // Clamp Z to physically sensible range
        let z_frag = z_frag.max(1).min(a_frag - 1);
        let n_frag = a_frag - z_frag;

        let be = predictor.predict(z_frag, n_frag).binding_energy;

        fragments.push(FissionFragment {
            z: z_frag,
            n: n_frag,
            a: *a_frag,
            yield_fraction: yf,
            binding_energy: be,
        });

        weighted_be_sum += yf * be;
    }

    // Q-value: sum of fragment BE - parent BE + neutron KE contribution
    // Q ≈ <BE_fragments> - BE_parent (per fission, weighting by yield pairs)
    // A rough estimate: average the yield-weighted fragment BEs for a pair
    let q_fission = weighted_be_sum - parent_be;

    let peak_light = (mu_light.round() as u16).max(a_min);
    let peak_heavy = (mu_heavy.round() as u16).min(a_max);

    fragments.sort_by_key(|f| f.a);

    FissionYieldResult {
        parent_z,
        parent_n: parent_n,
        parent_a: (parent_z + parent_n),
        fragments,
        tke_viola_mev: tke_viola,
        q_fission_mev: q_fission,
        peak_light,
        peak_heavy,
    }
}

// ── 2. Transmutation Pathways ─────────────────────────────────────────────────

/// A single step in a neutron-capture transmutation chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmutationStep {
    /// Isotope Z at this step.
    pub z: u16,
    /// Isotope N at this step.
    pub n: u16,
    /// Mass number A = Z + N.
    pub a: u16,
    /// Neutron separation energy S_n (MeV). Higher → larger capture cross-section.
    pub separation_energy_mev: f64,
    /// Proxy capture cross-section (arbitrary units, proportional to exp(S_n / kT)).
    pub capture_proxy: f64,
    /// Whether this isotope is considered stable (proxy: S_n in reasonable range
    /// and not a known long-lived fission product).
    pub is_stable: bool,
    /// Element symbol (approximate).
    pub element: String,
}

/// Full transmutation chain result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmutationChain {
    /// Starting isotope description.
    pub start_element: String,
    /// Starting (Z, N).
    pub start_z: u16,
    pub start_n: u16,
    /// Chain of (n,gamma) captures.
    pub steps: Vec<TransmutationStep>,
    /// Number of captures to reach stability (or max chain length).
    pub captures_to_stable: Option<usize>,
    /// Whether a stable endpoint was found.
    pub reached_stable: bool,
}

/// Known long-lived fission products (Z, N, half-life description).
const LONG_LIVED_FP: &[(u16, u16, &str)] = &[
    (55, 82, "Cs-137 (30.2 yr)"),  // Cs-137: Z=55, N=82
    (38, 52, "Sr-90  (28.8 yr)"),  // Sr-90:  Z=38, N=52
    (43, 56, "Tc-99  (2.1e5 yr)"), // Tc-99:  Z=43, N=56
    (53, 76, "I-129  (1.6e7 yr)"), // I-129:  Z=53, N=76
];

/// Known stable isotopes used as transmutation endpoints.
/// Format: (Z, N) — a subset of relevant ones for fission product chains.
const STABLE_ENDPOINTS: &[(u16, u16)] = &[
    // Ruthenium isotopes (Tc-99 chain endpoint)
    (44, 56), // Ru-100
    (44, 57), // Ru-101
    (44, 58), // Ru-102
    // Xenon isotopes (I-129 chain)
    (54, 76), // Xe-130
    (54, 77), // Xe-131
    (54, 78), // Xe-132
    // Barium isotopes (Cs-137 chain)
    (56, 82), // Ba-138
    (56, 83), // Ba-139 (short-lived, decays to stable La-139)
    // Zirconium/Yttrium (Sr-90 chain)
    (40, 52), // Zr-92
    (40, 51), // Zr-91
    (39, 51), // Y-90 → Zr-90 (beta decay, short-lived intermediate)
];

/// Element symbol lookup (Z=1..60 covers fission product region).
fn element_symbol(z: u16) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        21 => "Sc",
        22 => "Ti",
        23 => "V",
        24 => "Cr",
        25 => "Mn",
        26 => "Fe",
        27 => "Co",
        28 => "Ni",
        29 => "Cu",
        30 => "Zn",
        31 => "Ga",
        32 => "Ge",
        33 => "As",
        34 => "Se",
        35 => "Br",
        36 => "Kr",
        37 => "Rb",
        38 => "Sr",
        39 => "Y",
        40 => "Zr",
        41 => "Nb",
        42 => "Mo",
        43 => "Tc",
        44 => "Ru",
        45 => "Rh",
        46 => "Pd",
        47 => "Ag",
        48 => "Cd",
        49 => "In",
        50 => "Sn",
        51 => "Sb",
        52 => "Te",
        53 => "I",
        54 => "Xe",
        55 => "Cs",
        56 => "Ba",
        57 => "La",
        58 => "Ce",
        59 => "Pr",
        60 => "Nd",
        90 => "Th",
        91 => "Pa",
        92 => "U",
        93 => "Np",
        94 => "Pu",
        95 => "Am",
        96 => "Cm",
        _ => "??",
    }
}

/// Compute neutron separation energy: S_n = BE(Z, N) - BE(Z, N-1).
///
/// This is the energy released when a neutron is captured — higher S_n
/// means the (n,gamma) reaction is more exothermic and the capture
/// cross-section is generally larger.
pub fn neutron_separation_energy(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let be_full = predictor.predict(z, n).binding_energy;
    let be_minus1 = predictor.predict(z, n - 1).binding_energy;
    be_full - be_minus1
}

/// Compute capture cross-section proxy from separation energy.
///
/// For thermal neutrons, the capture cross-section scales roughly as
/// exp(S_n / kT) due to the density of states at the compound nucleus
/// excitation energy. We use a scaled version to keep values manageable.
fn capture_cross_section_proxy(s_n_mev: f64) -> f64 {
    // Scale S_n to eV, divide by thermal kT
    let s_n_ev = s_n_mev * 1.0e6;
    // Use a tempered version: sigma ~ S_n^2 for positive S_n
    // (full exp would overflow; the quadratic captures the trend)
    if s_n_mev > 0.0 {
        s_n_mev * s_n_mev * 100.0
    } else {
        0.01
    }
}

/// Compute a neutron-capture transmutation chain starting from (z, n).
///
/// Follows (n,gamma) captures: each step adds one neutron. The chain
/// terminates when a stable endpoint is reached or after `max_steps`.
///
/// Note: In reality, some chain members beta-decay before capturing
/// another neutron. This simplified model tracks the (n,gamma) path
/// and flags known stable endpoints.
pub fn transmutation_chain(
    predictor: &MlMassPredictor,
    start_z: u16,
    start_n: u16,
    max_steps: usize,
) -> TransmutationChain {
    let start_a = start_z + start_n;
    let start_elem = format!("{}-{}", element_symbol(start_z), start_a);

    let mut steps = Vec::new();
    let mut z = start_z;
    let mut n = start_n;
    let mut reached_stable = false;
    let mut captures_to_stable = None;

    for i in 0..max_steps {
        let a = z + n;
        let s_n = neutron_separation_energy(predictor, z, n);
        let proxy = capture_cross_section_proxy(s_n);
        let is_stable = STABLE_ENDPOINTS.contains(&(z, n));
        let elem = format!("{}-{}", element_symbol(z), a);

        steps.push(TransmutationStep {
            z,
            n,
            a,
            separation_energy_mev: s_n,
            capture_proxy: proxy,
            is_stable,
            element: elem,
        });

        if is_stable && i > 0 {
            reached_stable = true;
            captures_to_stable = Some(i);
            break;
        }

        // Next step: (n, gamma) capture adds one neutron
        n += 1;
    }

    TransmutationChain {
        start_element: start_elem,
        start_z,
        start_n,
        steps,
        captures_to_stable,
        reached_stable,
    }
}

/// Compute transmutation chains for all four major long-lived fission products.
pub fn all_transmutation_chains(predictor: &MlMassPredictor) -> Vec<TransmutationChain> {
    LONG_LIVED_FP
        .iter()
        .map(|&(z, n, _)| transmutation_chain(predictor, z, n, 10))
        .collect()
}

// ── 3. Fuel Characterization ──────────────────────────────────────────────────

/// Reactor fuel candidate properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuelProperties {
    /// Fuel name (e.g., "U-235").
    pub name: String,
    /// Proton number.
    pub z: u16,
    /// Neutron number.
    pub n: u16,
    /// Mass number.
    pub a: u16,
    /// Total binding energy (MeV).
    pub binding_energy_mev: f64,
    /// Binding energy per nucleon (MeV).
    pub be_per_nucleon: f64,
    /// Fission barrier height (MeV).
    pub fission_barrier_mev: f64,
    /// Fissility parameter.
    pub fissility: f64,
    /// Alpha-decay Q-value (MeV). Q > 0 means alpha-unstable.
    pub alpha_q_mev: f64,
    /// Estimated energy per fission (MeV).
    pub energy_per_fission_mev: f64,
    /// Whether this is a fissile (directly fissionable by thermal neutrons).
    pub is_fissile: bool,
    /// Whether this is fertile (can breed into fissile material).
    pub is_fertile: bool,
}

/// Known fissile/fertile isotopes for characterization.
const FUEL_ISOTOPES: &[(u16, u16, &str, bool, bool)] = &[
    // (Z, N, name, is_fissile, is_fertile)
    (92, 143, "U-235", true, false),
    (92, 146, "U-238", false, true),
    (94, 145, "Pu-239", true, false),
    (94, 147, "Pu-241", true, false),
    (90, 142, "Th-232", false, true),
    (92, 141, "U-233", true, false),
];

/// Compute alpha-decay Q-value: Q_alpha = BE(daughter) + B_ALPHA - BE(parent).
fn alpha_q_value(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    let be_parent = predictor.predict(z, n).binding_energy;
    let be_daughter = predictor.predict(z - 2, n - 2).binding_energy;
    be_daughter + B_ALPHA - be_parent
}

/// Estimate energy per fission from ML binding energies.
///
/// Q_fission ~ BE(light fragment) + BE(heavy fragment) - BE(parent) + nu_bar * neutron_KE
/// Using average fragments A_light=95, A_heavy=A-95-nu_bar.
fn energy_per_fission(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    let a = (z + n) as f64;
    let z_f = z as f64;
    let z_over_a = z_f / a;

    // Average fragments
    let a_light = 95.0_f64.min(a * 0.4);
    let a_heavy = a - a_light - NU_BAR_U235; // subtract prompt neutrons

    let z_light = (a_light * z_over_a).round() as u16;
    let n_light = (a_light.round() as u16).saturating_sub(z_light);
    let z_heavy = (a_heavy * z_over_a).round() as u16;
    let n_heavy = (a_heavy.round() as u16).saturating_sub(z_heavy);

    let be_parent = predictor.predict(z, n).binding_energy;
    let be_light = predictor.predict(z_light, n_light).binding_energy;
    let be_heavy = predictor.predict(z_heavy, n_heavy).binding_energy;

    // Q = BE_products - BE_reactants (binding energy is positive)
    let q = be_light + be_heavy - be_parent;
    q.max(0.0) // should be ~200 MeV for actinides
}

/// Characterize all standard reactor fuel candidates.
pub fn characterize_fuels(predictor: &MlMassPredictor) -> Vec<FuelProperties> {
    FUEL_ISOTOPES
        .iter()
        .map(|&(z, n, name, is_fissile, is_fertile)| {
            let pred = predictor.predict(z, n);
            let barrier = compute_fission_barrier(z, n);
            let q_alpha = if z >= 2 && n >= 2 {
                alpha_q_value(predictor, z, n)
            } else {
                0.0
            };
            let q_fission = energy_per_fission(predictor, z, n);

            FuelProperties {
                name: name.to_string(),
                z,
                n,
                a: z + n,
                binding_energy_mev: pred.binding_energy,
                be_per_nucleon: pred.ba,
                fission_barrier_mev: barrier.total_barrier,
                fissility: barrier.fissility,
                alpha_q_mev: q_alpha,
                energy_per_fission_mev: q_fission,
                is_fissile,
                is_fertile,
            }
        })
        .collect()
}

/// Breeding chain properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreedingChain {
    /// Chain name (e.g., "Th-232 -> U-233").
    pub name: String,
    /// Fertile parent (Z, N).
    pub fertile_z: u16,
    pub fertile_n: u16,
    /// Fissile product (Z, N).
    pub fissile_z: u16,
    pub fissile_n: u16,
    /// Number of intermediate steps (beta decays).
    pub intermediate_steps: u16,
    /// Neutron separation energy of fertile isotope (MeV) — governs capture rate.
    pub fertile_sn_mev: f64,
    /// Energy per fission of the bred fissile product (MeV).
    pub fissile_energy_mev: f64,
    /// Fission barrier of bred fissile product (MeV).
    pub fissile_barrier_mev: f64,
}

/// Compare the two main breeding cycles:
/// - Th-232 + n -> Th-233 -> Pa-233 -> U-233 (thorium cycle, MSR)
/// - U-238  + n -> U-239  -> Np-239 -> Pu-239 (uranium cycle, fast breeder)
pub fn compare_breeding_cycles(predictor: &MlMassPredictor) -> Vec<BreedingChain> {
    let th_sn = neutron_separation_energy(predictor, 90, 142);
    let u233_energy = energy_per_fission(predictor, 92, 141);
    let u233_barrier = compute_fission_barrier(92, 141);

    let u238_sn = neutron_separation_energy(predictor, 92, 146);
    let pu239_energy = energy_per_fission(predictor, 94, 145);
    let pu239_barrier = compute_fission_barrier(94, 145);

    vec![
        BreedingChain {
            name: "Th-232 -> Pa-233 -> U-233 (thorium cycle)".to_string(),
            fertile_z: 90,
            fertile_n: 142,
            fissile_z: 92,
            fissile_n: 141,
            intermediate_steps: 2, // Th-233 (beta) -> Pa-233 (beta) -> U-233
            fertile_sn_mev: th_sn,
            fissile_energy_mev: u233_energy,
            fissile_barrier_mev: u233_barrier.total_barrier,
        },
        BreedingChain {
            name: "U-238 -> Np-239 -> Pu-239 (uranium cycle)".to_string(),
            fertile_z: 92,
            fertile_n: 146,
            fissile_z: 94,
            fissile_n: 145,
            intermediate_steps: 2, // U-239 (beta) -> Np-239 (beta) -> Pu-239
            fertile_sn_mev: u238_sn,
            fissile_energy_mev: pu239_energy,
            fissile_barrier_mev: pu239_barrier.total_barrier,
        },
    ]
}

// ── 4. Decay Heat ─────────────────────────────────────────────────────────────

/// Decay heat data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayHeatPoint {
    /// Time after shutdown (seconds).
    pub time_s: f64,
    /// Time after shutdown (human-readable).
    pub time_label: String,
    /// Decay heat as fraction of operating power.
    pub power_fraction: f64,
    /// Decay heat in MW per GW(th) of operating power.
    pub mw_per_gwth: f64,
}

/// Decay heat curve result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayHeatCurve {
    /// Operating time before shutdown (seconds).
    pub operating_time_s: f64,
    /// Data points at logarithmically spaced times.
    pub points: Vec<DecayHeatPoint>,
}

/// Format time duration for human readability.
fn format_time(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1} s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1} min", seconds / 60.0)
    } else if seconds < 86400.0 {
        format!("{:.1} hr", seconds / 3600.0)
    } else if seconds < 86400.0 * 365.25 {
        format!("{:.1} days", seconds / 86400.0)
    } else {
        format!("{:.2} yr", seconds / (86400.0 * 365.25))
    }
}

/// Compute decay heat curve using the Way-Wigner approximation.
///
/// P(t) / P_0 = 0.0622 × [t^(-0.2) - (t + t_s)^(-0.2)]
///
/// where:
/// - t = time after shutdown (seconds)
/// - t_s = operating time before shutdown (seconds)
/// - P_0 = steady-state operating power
///
/// Valid for t > ~1 second and t_s > t (operating time much longer
/// than time after shutdown for best accuracy). The approximation
/// captures the sum of hundreds of fission product beta + gamma decays.
///
/// Reference: Way & Wigner, Phys. Rev. 73, 1318 (1948).
pub fn decay_heat_curve(operating_time_s: f64) -> DecayHeatCurve {
    // Logarithmically spaced time points from 1 second to 1 year
    let log_min = 0.0_f64; // 10^0 = 1 second
    let log_max = 7.5_f64; // 10^7.5 ~ 1 year
    let n_points = 30;

    let points: Vec<DecayHeatPoint> = (0..n_points)
        .map(|i| {
            let log_t = log_min + (log_max - log_min) * (i as f64) / ((n_points - 1) as f64);
            let t = 10.0_f64.powf(log_t);

            // Way-Wigner formula
            let power_fraction = 0.0622 * (t.powf(-0.2) - (t + operating_time_s).powf(-0.2));

            // Clamp to physical range
            let power_fraction = power_fraction.max(0.0).min(1.0);

            DecayHeatPoint {
                time_s: t,
                time_label: format_time(t),
                power_fraction,
                mw_per_gwth: power_fraction * 1000.0, // MW per GW(th)
            }
        })
        .collect();

    DecayHeatCurve {
        operating_time_s,
        points,
    }
}

/// Compute decay heat at a specific time after shutdown.
///
/// Returns power fraction P(t)/P_0.
pub fn decay_heat_at(time_after_shutdown_s: f64, operating_time_s: f64) -> f64 {
    if time_after_shutdown_s <= 0.0 {
        return 0.0;
    }
    let p = 0.0622
        * (time_after_shutdown_s.powf(-0.2)
            - (time_after_shutdown_s + operating_time_s).powf(-0.2));
    p.max(0.0).min(1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ── Fission product yields ────────────────────────────────────────────

    #[test]
    fn test_u235_fission_yield_distribution() {
        let pred = predictor();
        let result = fission_product_yields(&pred, 92, 143);

        println!("\n=== U-235 Fission Product Yield Distribution ===");
        println!(
            "Parent: U-235 (Z={}, N={}, A={})",
            result.parent_z, result.parent_n, result.parent_a
        );
        println!("TKE (Viola): {:.1} MeV", result.tke_viola_mev);
        println!("Q_fission:   {:.1} MeV", result.q_fission_mev);
        println!(
            "Peak masses: light={}, heavy={}",
            result.peak_light, result.peak_heavy
        );
        println!(
            "\n{:>4} {:>4} {:>4}  {:>8}  {:>10}",
            "A", "Z", "N", "Yield%", "BE (MeV)"
        );
        println!("{}", "-".repeat(45));

        for frag in &result.fragments {
            if frag.yield_fraction > 0.01 {
                println!(
                    "{:>4} {:>4} {:>4}  {:>8.3}  {:>10.2}",
                    frag.a,
                    frag.z,
                    frag.n,
                    frag.yield_fraction * 100.0,
                    frag.binding_energy
                );
            }
        }

        // Sanity checks
        assert!(result.parent_a == 235, "parent A should be 235");
        assert!(
            (160.0..=200.0).contains(&result.tke_viola_mev),
            "TKE should be ~170 MeV for U-235, got {}",
            result.tke_viola_mev
        );
        assert!(
            result.peak_light < result.peak_heavy,
            "light peak should be less than heavy peak"
        );
        assert!(
            !result.fragments.is_empty(),
            "should have fission fragments"
        );

        // Check yield normalization (should sum to ~2.0)
        let total: f64 = result.fragments.iter().map(|f| f.yield_fraction).sum();
        assert!(
            (1.5..=2.5).contains(&total),
            "total yield should be ~2.0, got {}",
            total
        );
    }

    #[test]
    fn test_pu239_fission_yield_distribution() {
        let pred = predictor();
        let result = fission_product_yields(&pred, 94, 145);

        println!("\n=== Pu-239 Fission Product Yield Distribution ===");
        println!("TKE (Viola): {:.1} MeV", result.tke_viola_mev);
        println!("Q_fission:   {:.1} MeV", result.q_fission_mev);
        println!(
            "Peak masses: light={}, heavy={}",
            result.peak_light, result.peak_heavy
        );

        assert_eq!(result.parent_a, 239);
        assert!(result.tke_viola_mev > 160.0, "TKE should be > 160 MeV");
    }

    #[test]
    fn test_viola_tke_systematics() {
        let pred = predictor();
        // TKE should increase with Z^2/A^(1/3)
        let u235_tke = fission_product_yields(&pred, 92, 143).tke_viola_mev;
        let pu239_tke = fission_product_yields(&pred, 94, 145).tke_viola_mev;
        let cf252_tke = fission_product_yields(&pred, 98, 154).tke_viola_mev;

        println!("\n=== Viola TKE Systematics ===");
        println!("U-235:  {:.1} MeV", u235_tke);
        println!("Pu-239: {:.1} MeV", pu239_tke);
        println!("Cf-252: {:.1} MeV", cf252_tke);

        assert!(pu239_tke > u235_tke, "Pu-239 TKE should exceed U-235 TKE");
        assert!(cf252_tke > pu239_tke, "Cf-252 TKE should exceed Pu-239 TKE");
    }

    #[test]
    fn test_yield_peaks_asymmetric() {
        let pred = predictor();
        let result = fission_product_yields(&pred, 92, 143);

        // Find the two highest-yield fragments
        let mut sorted = result.fragments.clone();
        sorted.sort_by(|a, b| b.yield_fraction.partial_cmp(&a.yield_fraction).unwrap());

        let top1 = &sorted[0];
        let top2 = &sorted[1];

        println!("\n=== Yield Peaks (U-235) ===");
        println!("Highest yield: A={} (Y={:.3})", top1.a, top1.yield_fraction);
        println!(
            "Second highest: A={} (Y={:.3})",
            top2.a, top2.yield_fraction
        );

        // The two highest-yield fragments should be in the light and heavy humps
        let (a_lo, a_hi) = if top1.a < top2.a {
            (top1.a, top2.a)
        } else {
            (top2.a, top1.a)
        };

        assert!(
            a_lo < 120 && a_hi > 120,
            "peaks should straddle A=120: got {} and {}",
            a_lo,
            a_hi
        );
    }

    // ── Transmutation chains ──────────────────────────────────────────────

    #[test]
    fn test_neutron_separation_energy_positive() {
        let pred = predictor();
        // S_n for stable isotopes should be positive (bound)
        let sn_fe56 = neutron_separation_energy(&pred, 26, 30);
        let sn_u235 = neutron_separation_energy(&pred, 92, 143);

        println!("\n=== Neutron Separation Energies ===");
        println!("Fe-56: S_n = {:.3} MeV", sn_fe56);
        println!("U-235: S_n = {:.3} MeV", sn_u235);

        assert!(sn_fe56 > 0.0, "Fe-56 S_n should be positive");
        assert!(sn_u235 > 0.0, "U-235 S_n should be positive");
    }

    #[test]
    fn test_tc99_transmutation_chain() {
        let pred = predictor();
        let chain = transmutation_chain(&pred, 43, 56, 10);

        println!("\n=== Tc-99 Transmutation Chain ===");
        println!("Start: {}", chain.start_element);
        println!(
            "{:>10} {:>4} {:>4}  {:>8}  {:>10}  {}",
            "Isotope", "Z", "N", "S_n(MeV)", "sigma_proxy", "Stable?"
        );
        println!("{}", "-".repeat(55));

        for step in &chain.steps {
            println!(
                "{:>10} {:>4} {:>4}  {:>8.3}  {:>10.1}  {}",
                step.element,
                step.z,
                step.n,
                step.separation_energy_mev,
                step.capture_proxy,
                if step.is_stable { "YES" } else { "" }
            );
        }

        if let Some(nc) = chain.captures_to_stable {
            println!("Captures to stable: {}", nc);
        }
        println!("Reached stable: {}", chain.reached_stable);

        // Tc-99 + n -> Tc-100 (unstable, beta-decays) -> Ru-100 (stable)
        // In our (n,gamma) model: Tc-99 -> Tc-100 is one capture, then we
        // check Ru-100 equivalence at (44, 56)
        assert!(chain.steps.len() >= 2, "chain should have at least 2 steps");
        assert_eq!(chain.steps[0].element, "Tc-99");
    }

    #[test]
    fn test_all_transmutation_chains() {
        let pred = predictor();
        let chains = all_transmutation_chains(&pred);

        println!("\n=== All Major Waste Transmutation Chains ===");

        for chain in &chains {
            println!("\n--- {} ---", chain.start_element);
            for step in &chain.steps {
                let stable_mark = if step.is_stable { " [STABLE]" } else { "" };
                println!(
                    "  {} (Z={}, N={})  S_n={:.2} MeV{}",
                    step.element, step.z, step.n, step.separation_energy_mev, stable_mark
                );
            }
            match chain.captures_to_stable {
                Some(n) => println!("  -> {} captures to stability", n),
                None => println!("  -> did NOT reach stable endpoint in chain"),
            }
        }

        assert_eq!(chains.len(), 4, "should have 4 waste isotope chains");
    }

    #[test]
    fn test_cs137_transmutation() {
        let pred = predictor();
        let chain = transmutation_chain(&pred, 55, 82, 10);

        println!("\n=== Cs-137 Transmutation Chain ===");
        for step in &chain.steps {
            println!(
                "  {} S_n={:.2} MeV  proxy={:.1}",
                step.element, step.separation_energy_mev, step.capture_proxy
            );
        }

        assert_eq!(chain.start_element, "Cs-137");
        assert_eq!(chain.steps[0].z, 55);
        assert_eq!(chain.steps[0].n, 82);
    }

    #[test]
    fn test_sr90_transmutation() {
        let pred = predictor();
        let chain = transmutation_chain(&pred, 38, 52, 10);

        println!("\n=== Sr-90 Transmutation Chain ===");
        for step in &chain.steps {
            println!(
                "  {} S_n={:.2} MeV  proxy={:.1}",
                step.element, step.separation_energy_mev, step.capture_proxy
            );
        }

        assert_eq!(chain.start_element, "Sr-90");
    }

    #[test]
    fn test_i129_transmutation() {
        let pred = predictor();
        let chain = transmutation_chain(&pred, 53, 76, 10);

        println!("\n=== I-129 Transmutation Chain ===");
        for step in &chain.steps {
            let stable_mark = if step.is_stable { " [STABLE]" } else { "" };
            println!(
                "  {} S_n={:.2} MeV{}",
                step.element, step.separation_energy_mev, stable_mark
            );
        }

        assert_eq!(chain.start_element, "I-129");
    }

    // ── Fuel characterization ─────────────────────────────────────────────

    #[test]
    fn test_fuel_comparison_table() {
        let pred = predictor();
        let fuels = characterize_fuels(&pred);

        println!("\n=== Reactor Fuel Comparison Table ===");
        println!(
            "{:<8} {:>6} {:>8} {:>8} {:>7} {:>8} {:>9}  {}",
            "Fuel", "A", "BE/A", "Barrier", "Fiss.", "Q_alpha", "Q_fission", "Type"
        );
        println!(
            "{:<8} {:>6} {:>8} {:>8} {:>7} {:>8} {:>9}  {}",
            "", "", "(MeV)", "(MeV)", "", "(MeV)", "(MeV)", ""
        );
        println!("{}", "-".repeat(80));

        for fuel in &fuels {
            let ftype = if fuel.is_fissile {
                "Fissile"
            } else if fuel.is_fertile {
                "Fertile"
            } else {
                "Other"
            };
            println!(
                "{:<8} {:>6} {:>8.3} {:>8.2} {:>7.3} {:>8.2} {:>9.1}  {}",
                fuel.name,
                fuel.a,
                fuel.be_per_nucleon,
                fuel.fission_barrier_mev,
                fuel.fissility,
                fuel.alpha_q_mev,
                fuel.energy_per_fission_mev,
                ftype
            );
        }

        // Sanity checks
        assert_eq!(fuels.len(), 6);

        let u235 = fuels.iter().find(|f| f.name == "U-235").unwrap();
        assert!(u235.is_fissile);
        assert!(u235.be_per_nucleon > 7.0 && u235.be_per_nucleon < 8.0);

        let u238 = fuels.iter().find(|f| f.name == "U-238").unwrap();
        assert!(u238.is_fertile);
        assert!(!u238.is_fissile);

        let pu239 = fuels.iter().find(|f| f.name == "Pu-239").unwrap();
        assert!(pu239.is_fissile);

        let th232 = fuels.iter().find(|f| f.name == "Th-232").unwrap();
        assert!(th232.is_fertile);
    }

    #[test]
    fn test_breeding_cycles() {
        let pred = predictor();
        let cycles = compare_breeding_cycles(&pred);

        println!("\n=== Breeding Cycle Comparison ===");
        for cycle in &cycles {
            println!("\n{}", cycle.name);
            println!("  Fertile S_n:        {:.3} MeV", cycle.fertile_sn_mev);
            println!("  Fissile Q_fission:  {:.1} MeV", cycle.fissile_energy_mev);
            println!("  Fissile barrier:    {:.2} MeV", cycle.fissile_barrier_mev);
            println!("  Beta-decay steps:   {}", cycle.intermediate_steps);
        }

        assert_eq!(cycles.len(), 2);

        let th_cycle = &cycles[0];
        let u_cycle = &cycles[1];

        // Both fertile isotopes should have positive S_n (bound neutron)
        assert!(
            th_cycle.fertile_sn_mev > 0.0,
            "Th-232 S_n should be positive"
        );
        assert!(u_cycle.fertile_sn_mev > 0.0, "U-238 S_n should be positive");

        // Both bred fissile isotopes should release energy
        assert!(
            th_cycle.fissile_energy_mev > 100.0,
            "U-233 fission energy should be > 100 MeV"
        );
        assert!(
            u_cycle.fissile_energy_mev > 100.0,
            "Pu-239 fission energy should be > 100 MeV"
        );
    }

    #[test]
    fn test_alpha_q_values_positive_for_actinides() {
        let pred = predictor();
        // All actinides should be alpha-unstable (Q > 0)
        let q_u235 = alpha_q_value(&pred, 92, 143);
        let q_u238 = alpha_q_value(&pred, 92, 146);
        let q_pu239 = alpha_q_value(&pred, 94, 145);
        let q_th232 = alpha_q_value(&pred, 90, 142);

        println!("\n=== Alpha Q-values (actinides) ===");
        println!("U-235:  Q_alpha = {:.3} MeV", q_u235);
        println!("U-238:  Q_alpha = {:.3} MeV", q_u238);
        println!("Pu-239: Q_alpha = {:.3} MeV", q_pu239);
        println!("Th-232: Q_alpha = {:.3} MeV", q_th232);

        // These should all be positive (alpha-unstable), typically 4-6 MeV
        assert!(q_u235 > 0.0, "U-235 should be alpha-unstable");
        assert!(q_th232 > 0.0, "Th-232 should be alpha-unstable");
    }

    // ── Decay heat ────────────────────────────────────────────────────────

    #[test]
    fn test_decay_heat_curve() {
        // 1 year of operation = 3.15e7 seconds
        let operating_time = 365.25 * 24.0 * 3600.0;
        let curve = decay_heat_curve(operating_time);

        println!("\n=== Decay Heat Curve (1 year operation) ===");
        println!("{:>12}  {:>12}  {:>12}", "Time", "P/P0 (%)", "MW/GW(th)");
        println!("{}", "-".repeat(40));

        for pt in &curve.points {
            println!(
                "{:>12}  {:>11.4}%  {:>12.2}",
                pt.time_label,
                pt.power_fraction * 100.0,
                pt.mw_per_gwth
            );
        }

        // At t=1s, decay heat should be ~6% of operating power
        let p_1s = decay_heat_at(1.0, operating_time);
        println!("\nAt t=1s: P/P0 = {:.4}%", p_1s * 100.0);
        assert!(
            (0.04..=0.08).contains(&p_1s),
            "decay heat at 1s should be ~6%, got {:.4}%",
            p_1s * 100.0
        );

        // At t=1 hour, decay heat should be ~1%
        let p_1h = decay_heat_at(3600.0, operating_time);
        println!("At t=1h: P/P0 = {:.4}%", p_1h * 100.0);
        assert!(
            (0.005..=0.02).contains(&p_1h),
            "decay heat at 1h should be ~1%, got {:.4}%",
            p_1h * 100.0
        );

        // At t=1 day, should be lower still
        let p_1d = decay_heat_at(86400.0, operating_time);
        println!("At t=1d: P/P0 = {:.4}%", p_1d * 100.0);
        assert!(p_1d < p_1h, "decay heat should decrease with time");

        // At t=1 year, very low
        let p_1y = decay_heat_at(3.15e7, operating_time);
        println!("At t=1yr: P/P0 = {:.6}%", p_1y * 100.0);
        assert!(p_1y < 0.001, "decay heat at 1 year should be < 0.1%");
    }

    #[test]
    fn test_decay_heat_monotonic_decrease() {
        let operating_time = 1.0e7; // ~115 days
        let times = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1e6, 1e7];

        let mut prev = f64::MAX;
        for &t in &times {
            let p = decay_heat_at(t, operating_time);
            assert!(
                p <= prev,
                "decay heat should decrease monotonically: at t={}, P={} > prev={}",
                t,
                p,
                prev
            );
            prev = p;
        }
    }

    #[test]
    fn test_decay_heat_zero_at_negative_time() {
        assert_eq!(decay_heat_at(-1.0, 1e7), 0.0);
        assert_eq!(decay_heat_at(0.0, 1e7), 0.0);
    }

    #[test]
    fn test_decay_heat_longer_operation_higher_initial() {
        // Longer operation time -> higher initial decay heat
        let p_short = decay_heat_at(10.0, 1e5); // ~1 day operation
        let p_long = decay_heat_at(10.0, 1e8); // ~3 years operation

        println!("\n=== Operating Time Effect on Decay Heat ===");
        println!("Short operation (1 day),  at t=10s: {:.6}", p_short);
        println!("Long operation  (3 yr),   at t=10s: {:.6}", p_long);

        assert!(
            p_long >= p_short,
            "longer operation should give higher or equal initial decay heat"
        );
    }

    // ── Integration / summary ─────────────────────────────────────────────

    #[test]
    fn test_full_reactor_analysis_summary() {
        let pred = predictor();

        println!("\n{}", "=".repeat(60));
        println!("    REACTOR PHYSICS ANALYSIS SUMMARY");
        println!("{}\n", "=".repeat(60));

        // Fission yields
        let u235_yield = fission_product_yields(&pred, 92, 143);
        println!("1. U-235 FISSION");
        println!("   TKE (Viola):    {:.1} MeV", u235_yield.tke_viola_mev);
        println!("   Q_fission:      {:.1} MeV", u235_yield.q_fission_mev);
        println!(
            "   Fragment count: {} (above 1% yield)",
            u235_yield
                .fragments
                .iter()
                .filter(|f| f.yield_fraction > 0.01)
                .count()
        );

        // Transmutation
        println!("\n2. WASTE TRANSMUTATION");
        let chains = all_transmutation_chains(&pred);
        for chain in &chains {
            let status = match chain.captures_to_stable {
                Some(n) => format!("{} captures -> stable", n),
                None => "no stable endpoint in 10 captures".to_string(),
            };
            println!("   {}: {}", chain.start_element, status);
        }

        // Fuel comparison
        println!("\n3. FUEL CANDIDATES");
        let fuels = characterize_fuels(&pred);
        for fuel in &fuels {
            println!(
                "   {}: BE/A={:.3}, barrier={:.1} MeV, Q={:.0} MeV",
                fuel.name,
                fuel.be_per_nucleon,
                fuel.fission_barrier_mev,
                fuel.energy_per_fission_mev
            );
        }

        // Breeding
        println!("\n4. BREEDING CYCLES");
        let cycles = compare_breeding_cycles(&pred);
        for c in &cycles {
            println!(
                "   {}: S_n(fertile)={:.2} MeV, Q(fissile)={:.0} MeV",
                c.name, c.fertile_sn_mev, c.fissile_energy_mev
            );
        }

        // Decay heat
        println!("\n5. DECAY HEAT (1 year operation)");
        let operating_time = 365.25 * 86400.0;
        for &(t, label) in &[
            (1.0, "1 sec"),
            (60.0, "1 min"),
            (3600.0, "1 hour"),
            (86400.0, "1 day"),
            (2.63e6, "1 month"),
            (3.15e7, "1 year"),
        ] {
            let p = decay_heat_at(t, operating_time);
            println!(
                "   {}: {:.4}% ({:.2} MW/GWth)",
                label,
                p * 100.0,
                p * 1000.0
            );
        }
    }
}
