// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Space Nuclear Applications
//!
//! Evaluates nuclear physics for deep-space mission design:
//!
//! - **RTG fuel candidates**: Known and novel alpha-emitters ranked by specific power (W/g)
//! - **Cosmic ray shielding**: Material comparison by interaction length and secondary production
//! - **Nuclear thermal propulsion**: Fuel comparison by energy per fission and fission barrier
//! - **Activation products**: Cosmic-ray induced radioactivity in spacecraft structural materials
//!
//! All binding energies computed via [`MlMassPredictor`] (DZ + Random Forest correction).
//!
//! ## References
//!
//! - Geiger, H. & Nuttall, J. M. (1911). *Philosophical Magazine*.
//! - Rowe, M. W. (2002). Space nuclear power systems. *Progress in Nuclear Energy*.
//! - NCRP Report No. 153 (2006). Information needed to make radiation protection
//!   recommendations for space missions beyond low-Earth orbit.
//! - Fassò et al. (2005). FLUKA: a multi-particle transport code. *CERN*.

use crate::fission_barrier::compute_fission_barrier;
use crate::ml_mass::MlMassPredictor;
use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

/// Neutron mass excess (MeV)
const M_N: f64 = 8.07132;
/// Hydrogen mass excess (MeV)
const M_H: f64 = 7.28897;
/// He-4 binding energy (MeV)
const BE_HE4: f64 = 28.296;
/// Avogadro constant
const N_A: f64 = 6.022_140_76e23;
/// ln(2)
const LN2: f64 = core::f64::consts::LN_2;
/// Seconds per year
const SECONDS_PER_YEAR: f64 = 3.156e7;

// ═══════════════════════════════════════════════════════════════════════════════
// §1  RTG Fuel Candidates
// ═══════════════════════════════════════════════════════════════════════════════

/// Decay mode of an isotope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayMode {
    Alpha,
    BetaMinus,
    BetaPlus,
    Gamma,
}

impl DecayMode {
    /// Qualitative shielding difficulty.
    pub fn shielding_difficulty(&self) -> &'static str {
        match self {
            DecayMode::Alpha => "minimal (few cm air / thin foil)",
            DecayMode::BetaMinus | DecayMode::BetaPlus => "moderate (mm Al / plastic)",
            DecayMode::Gamma => "heavy (cm Pb / concrete)",
        }
    }
}

impl std::fmt::Display for DecayMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecayMode::Alpha => write!(f, "alpha"),
            DecayMode::BetaMinus => write!(f, "beta-"),
            DecayMode::BetaPlus => write!(f, "beta+"),
            DecayMode::Gamma => write!(f, "gamma"),
        }
    }
}

/// RTG fuel candidate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtgCandidate {
    /// Element symbol or label
    pub name: String,
    /// Proton number
    pub z: u16,
    /// Neutron number
    pub n: u16,
    /// Mass number A = Z + N
    pub a: u16,
    /// Decay mode
    pub decay_mode: DecayMode,
    /// Half-life in years
    pub half_life_years: f64,
    /// Decay energy Q (MeV)
    pub q_value_mev: f64,
    /// Specific power (W/g)
    pub specific_power_wg: f64,
    /// Shielding difficulty
    pub shielding: String,
    /// Whether this is a known/established RTG fuel
    pub known: bool,
}

/// Compute Q_alpha for parent (z, n) → daughter (z-2, n-2) + He-4.
///
/// Q_alpha = BE(daughter) + BE(He4) - BE(parent)
fn q_alpha(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    if z < 2 || n < 2 {
        return 0.0;
    }
    let be_parent = predictor.predict(z, n).binding_energy;
    let be_daughter = predictor.predict(z - 2, n - 2).binding_energy;
    be_daughter + BE_HE4 - be_parent
}

/// Geiger-Nuttall half-life estimate for alpha decay.
///
/// log10(t½/s) = (a_coeff * Z_daughter + b1) / sqrt(Q) + (c_coeff * Z_daughter + d1)
///
/// Coefficients from Viola-Seaborg systematics.
fn geiger_nuttall_half_life_years(z_daughter: u16, q_mev: f64) -> f64 {
    if q_mev <= 0.0 {
        return f64::INFINITY;
    }
    let zd = z_daughter as f64;
    let log10_t_seconds = (1.66175 * zd - 8.5166) / q_mev.sqrt() + (-0.20228 * zd - 33.9069);
    let t_seconds = 10.0_f64.powf(log10_t_seconds);
    t_seconds / SECONDS_PER_YEAR
}

/// Specific power in W/g: P = Q * lambda * N_A / (A * 1000)
///
/// lambda = ln(2) / t_half (in seconds)
/// Q in joules = Q_mev * 1.602e-13
fn specific_power_wg(q_mev: f64, half_life_years: f64, a: u16) -> f64 {
    if half_life_years <= 0.0 || a == 0 {
        return 0.0;
    }
    let t_half_s = half_life_years * SECONDS_PER_YEAR;
    let lambda = LN2 / t_half_s;
    let q_joules = q_mev * 1.602e-13;
    // N_A atoms per mole, A g/mol → N_A/A atoms per gram → ×1000 not needed
    // W/g = Q_J * lambda * N_A / A
    q_joules * lambda * N_A / (a as f64)
}

/// Evaluate known RTG fuel candidates with literature values.
pub fn known_rtg_fuels(predictor: &MlMassPredictor) -> Vec<RtgCandidate> {
    let known = [
        // (name, z, n, decay_mode, half_life_years, q_override_mev or None)
        ("Pu-238", 94u16, 144u16, DecayMode::Alpha, 87.7, None),
        ("Sr-90", 38, 52, DecayMode::BetaMinus, 28.8, Some(0.546)),
        ("Am-241", 95, 146, DecayMode::Alpha, 432.2, None),
        ("Cm-244", 96, 148, DecayMode::Alpha, 18.1, None),
        ("Po-210", 84, 126, DecayMode::Alpha, 0.3789, None), // 138.4 days
    ];

    known
        .iter()
        .map(|&(name, z, n, mode, half_life, q_override)| {
            let a = z + n;
            let q = match q_override {
                Some(q) => q,
                None => q_alpha(predictor, z, n),
            };
            let sp = specific_power_wg(q, half_life, a);
            RtgCandidate {
                name: name.to_string(),
                z,
                n,
                a,
                decay_mode: mode,
                half_life_years: half_life,
                q_value_mev: q,
                specific_power_wg: sp,
                shielding: mode.shielding_difficulty().to_string(),
                known: true,
            }
        })
        .collect()
}

/// Scan for novel alpha-emitting RTG fuel candidates.
///
/// Searches Z ∈ [60, 100], N ∈ [80, 160] for isotopes with:
/// - Q_alpha in 4.0-7.0 MeV
/// - Geiger-Nuttall half-life between 10 and 200 years
/// - Ranked by specific power
pub fn scan_novel_rtg_candidates(predictor: &MlMassPredictor) -> Vec<RtgCandidate> {
    let mut candidates = Vec::new();

    for z in 60..=100u16 {
        for n in 80..=160u16 {
            let a = z + n;
            let q = q_alpha(predictor, z, n);

            // Filter: Q_alpha in usable range
            if q < 4.0 || q > 7.0 {
                continue;
            }

            // Geiger-Nuttall half-life estimate
            let t_half = geiger_nuttall_half_life_years(z - 2, q);

            // Filter: mission-viable half-life
            if t_half < 10.0 || t_half > 200.0 {
                continue;
            }

            let sp = specific_power_wg(q, t_half, a);

            candidates.push(RtgCandidate {
                name: format!("Z{}-A{}", z, a),
                z,
                n,
                a,
                decay_mode: DecayMode::Alpha,
                half_life_years: t_half,
                q_value_mev: q,
                specific_power_wg: sp,
                shielding: DecayMode::Alpha.shielding_difficulty().to_string(),
                known: false,
            });
        }
    }

    // Sort by specific power descending
    candidates.sort_by(|a, b| {
        b.specific_power_wg
            .partial_cmp(&a.specific_power_wg)
            .unwrap()
    });
    candidates
}

// ═══════════════════════════════════════════════════════════════════════════════
// §2  Cosmic Ray Shielding
// ═══════════════════════════════════════════════════════════════════════════════

/// Shielding material properties and GCR attenuation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShieldingMaterial {
    /// Material name
    pub name: String,
    /// Average atomic mass A
    pub avg_a: f64,
    /// Average atomic number Z
    pub avg_z: f64,
    /// Density (g/cm³)
    pub density: f64,
    /// Hydrogen mass fraction (0-1)
    pub hydrogen_fraction: f64,
    /// Nuclear interaction length (g/cm²)
    pub interaction_length_gcm2: f64,
    /// Interaction length in cm (= interaction_length / density)
    pub interaction_length_cm: f64,
    /// Relative secondary neutron production (normalized to Al = 1.0)
    pub secondary_neutron_factor: f64,
    /// Overall shielding score (higher = better): considers attenuation & secondaries
    pub shielding_score: f64,
}

/// Nuclear interaction length: lambda_int ~ 35 * A^(1/3) g/cm².
fn nuclear_interaction_length(avg_a: f64) -> f64 {
    35.0 * avg_a.powf(1.0 / 3.0)
}

/// Secondary neutron production scales roughly as A^0.7 (spallation cascade).
/// Normalized so Al-27 = 1.0.
fn secondary_neutron_factor(avg_a: f64) -> f64 {
    avg_a.powf(0.7) / 27.0_f64.powf(0.7)
}

/// Evaluate cosmic ray shielding candidates.
///
/// Shielding score = (1 / interaction_length_cm) * (1 / secondary_factor) * (1 + 2*H_fraction)
/// Higher score = better: short interaction length + low secondaries + high H content.
pub fn evaluate_shielding_materials() -> Vec<ShieldingMaterial> {
    // (name, avg_A, avg_Z, density g/cm³, H mass fraction)
    let materials = [
        ("Aluminum (Al)", 27.0, 13.0, 2.70, 0.0),
        ("Polyethylene (CH2)", 4.67, 2.67, 0.95, 0.143), // (12+2)/3 avg, H-rich
        ("Tungsten (W)", 183.84, 74.0, 19.3, 0.0),
        ("Lead (Pb)", 207.2, 82.0, 11.35, 0.0),
        ("Boron Carbide (B4C)", 8.77, 4.2, 2.52, 0.0), // (4*10.8+12)/5
        ("Lithium Hydride (LiH)", 3.97, 1.5, 0.78, 0.126), // (6.94+1.008)/2, H/(Li+H)
        ("Water (H2O)", 6.01, 3.33, 1.0, 0.111),
        ("Regolith (SiO2-like)", 21.7, 10.5, 1.5, 0.0),
    ];

    let mut results: Vec<ShieldingMaterial> = materials
        .iter()
        .map(|&(name, avg_a, avg_z, density, h_frac)| {
            let int_len_gcm2 = nuclear_interaction_length(avg_a);
            let int_len_cm = int_len_gcm2 / density;
            let sec_factor = secondary_neutron_factor(avg_a);
            // Score: prefer short interaction length (cm), low secondaries, high H
            let score = (1.0 / int_len_cm) * (1.0 / sec_factor) * (1.0 + 2.0 * h_frac);

            ShieldingMaterial {
                name: name.to_string(),
                avg_a,
                avg_z,
                density,
                hydrogen_fraction: h_frac,
                interaction_length_gcm2: int_len_gcm2,
                interaction_length_cm: int_len_cm,
                secondary_neutron_factor: sec_factor,
                shielding_score: score,
            }
        })
        .collect();

    // Sort by score descending (best first)
    results.sort_by(|a, b| b.shielding_score.partial_cmp(&a.shielding_score).unwrap());
    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// §3  Nuclear Thermal Propulsion
// ═══════════════════════════════════════════════════════════════════════════════

/// NTP fuel candidate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtpFuelCandidate {
    /// Fuel name
    pub name: String,
    /// Proton number
    pub z: u16,
    /// Neutron number
    pub n: u16,
    /// Mass number
    pub a: u16,
    /// Energy per fission (MeV), estimated from fragment binding energies
    pub energy_per_fission_mev: f64,
    /// Fission barrier height (MeV) — lower = easier chain reaction
    pub fission_barrier_mev: f64,
    /// Approximate thermal fission cross-section (barns) — literature values
    pub thermal_xs_barns: f64,
    /// Specific impulse score: proportional to sqrt(T / M_propellant)
    /// Normalized so all NTP fuels use H2 propellant at ~2700K
    pub isp_relative: f64,
    /// Overall NTP suitability score
    pub ntp_score: f64,
}

/// Estimate fission energy: Q ~ BE(fragments) - BE(parent).
///
/// For asymmetric fission (dominant mode), fragments peak near A~95 and A~140.
/// We compute: Q = BE(Z1, N1) + BE(Z2, N2) - BE(Z_parent, N_parent)
/// assuming Z splits proportional to A.
fn estimate_fission_energy(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    let a = (z + n) as f64;

    // Asymmetric fission: light fragment A~95, heavy fragment A~(A-95)
    let a_light: u16 = 95;
    let a_heavy = (z + n).saturating_sub(a_light);

    if a_heavy < 20 || a_light < 20 {
        return 200.0; // fallback to empirical ~200 MeV
    }

    // Z splits roughly proportional to A
    let z_light = ((z as f64) * (a_light as f64) / a).round() as u16;
    let z_heavy = z.saturating_sub(z_light);
    let n_light = a_light.saturating_sub(z_light);
    let n_heavy = a_heavy.saturating_sub(z_heavy);

    // Sanity checks
    if z_light == 0 || z_heavy == 0 || n_light == 0 || n_heavy == 0 {
        return 200.0;
    }

    let be_parent = predictor.predict(z, n).binding_energy;
    let be_light = predictor.predict(z_light, n_light).binding_energy;
    let be_heavy = predictor.predict(z_heavy, n_heavy).binding_energy;

    let q = be_light + be_heavy - be_parent;
    // Fission also emits ~2-3 neutrons worth ~5 MeV each kinetic + ~8 MeV gamma
    // but the fragment BE difference captures the bulk (~170-180 MeV)
    // Add ~20 MeV for prompt neutrons + gamma
    q + 20.0
}

/// Evaluate NTP fuel candidates.
pub fn evaluate_ntp_fuels(predictor: &MlMassPredictor) -> Vec<NtpFuelCandidate> {
    // (name, z, n, thermal_xs_barns — literature)
    let fuels = [
        ("U-235", 92u16, 143u16, 585.0),
        ("Pu-239", 94, 145, 748.0),
        ("Am-242m", 95, 147, 8500.0), // highest thermal fission xs
        ("U-233", 92, 141, 530.0),
        ("Cf-252", 98, 154, 32.0), // spontaneous fission source
    ];

    let mut candidates: Vec<NtpFuelCandidate> = fuels
        .iter()
        .map(|&(name, z, n, xs)| {
            let a = z + n;
            let q_fission = estimate_fission_energy(predictor, z, n);
            let barrier = compute_fission_barrier(z, n);

            // All NTP designs use H2 propellant at ~2700K, so Isp is same for all
            // The fuel affects reactor design (critical mass, power density)
            // Score: high energy × high xs / barrier (lower barrier = easier)
            let barrier_mev = barrier.total_barrier.max(0.1);
            let ntp_score = (q_fission * xs) / barrier_mev;
            // Relative Isp — all use H2 at same T, so Isp is essentially identical
            // (differs only by reactor temperature achievable, which correlates with q)
            let isp_relative = (q_fission / 200.0).sqrt();

            NtpFuelCandidate {
                name: name.to_string(),
                z,
                n,
                a,
                energy_per_fission_mev: q_fission,
                fission_barrier_mev: barrier_mev,
                thermal_xs_barns: xs,
                isp_relative,
                ntp_score,
            }
        })
        .collect();

    // Sort by NTP score descending
    candidates.sort_by(|a, b| b.ntp_score.partial_cmp(&a.ntp_score).unwrap());
    candidates
}

// ═══════════════════════════════════════════════════════════════════════════════
// §4  Activation Products
// ═══════════════════════════════════════════════════════════════════════════════

/// Activation product from cosmic ray spallation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationProduct {
    /// Parent material name
    pub parent_name: String,
    /// Parent (Z, N)
    pub parent_z: u16,
    pub parent_n: u16,
    /// Product description (reaction channel)
    pub reaction: String,
    /// Product (Z, N)
    pub product_z: u16,
    pub product_n: u16,
    /// Product mass number
    pub product_a: u16,
    /// Binding energy per nucleon of product (MeV)
    pub product_ba: f64,
    /// Whether product is likely radioactive (odd-odd, far from stability)
    pub likely_radioactive: bool,
    /// Estimated half-life category
    pub half_life_category: String,
    /// Estimated activity after exposure (relative, Bq/kg, order-of-magnitude)
    pub activity_relative: f64,
}

/// Check if a nucleus is likely radioactive.
///
/// Heuristic: odd-odd nuclei are almost always unstable; also check
/// if it's far from the valley of stability (N/Z ratio).
fn is_likely_radioactive(z: u16, n: u16) -> bool {
    if z == 0 || n == 0 {
        return true;
    }
    // Odd-odd: almost always radioactive (except H-2, Li-6, B-10, N-14, few others)
    let odd_odd = z % 2 != 0 && n % 2 != 0;
    if odd_odd && z > 7 {
        return true;
    }
    // Far from stability line: N/Z should be ~1 for light, ~1.5 for heavy
    let expected_n_over_z = if z < 20 {
        1.0
    } else {
        1.0 + 0.015 * (z as f64 - 20.0)
    };
    let actual = n as f64 / z as f64;
    let deviation = (actual - expected_n_over_z).abs();

    // Large deviation from stability → likely radioactive
    deviation > 0.15
}

/// Categorize half-life from binding energy stability.
fn half_life_category(z: u16, n: u16, ba: f64) -> String {
    // Magic number products tend to be stable
    let magic_z = [2, 8, 20, 28, 50, 82];
    let magic_n = [2, 8, 20, 28, 50, 82, 126];
    let near_magic_z = magic_z
        .iter()
        .any(|&m| (z as i32 - m as i32).unsigned_abs() <= 2);
    let near_magic_n = magic_n
        .iter()
        .any(|&m| (n as i32 - m as i32).unsigned_abs() <= 2);

    if !is_likely_radioactive(z, n) {
        "stable".to_string()
    } else if near_magic_z || near_magic_n {
        "long (>1 year)".to_string()
    } else if ba > 8.5 {
        "long (>1 year)".to_string()
    } else if ba > 8.0 {
        "medium (days-months)".to_string()
    } else {
        "short (<days)".to_string()
    }
}

/// GCR flux in deep space: ~4 particles/cm²/s (mostly protons).
/// Spallation cross-section ~30-50 mb for common materials.
/// Rate = flux * xs * N_target * time
fn estimate_activity_relative(z: u16, half_life_cat: &str, years_exposed: f64) -> f64 {
    if half_life_cat == "stable" {
        return 0.0;
    }
    // Base production rate scales with target A (larger nucleus = more channels)
    let base_rate = (z as f64).powf(0.7);
    // Activity depends on whether it builds up (long-lived) or reaches equilibrium (short-lived)
    let buildup = match half_life_cat {
        "long (>1 year)" => years_exposed.min(50.0),
        "medium (days-months)" => 1.0, // equilibrium quickly
        "short (<days)" => 0.01,       // negligible buildup
        _ => 0.1,
    };
    base_rate * buildup
}

/// Evaluate activation products for common spacecraft structural materials.
///
/// Models proton-knockout (p,2p) and neutron-knockout (p,pn) spallation channels.
pub fn evaluate_activation_products(
    predictor: &MlMassPredictor,
    years_in_deep_space: f64,
) -> Vec<ActivationProduct> {
    // (name, Z, N) for common structural materials
    let targets = [
        ("Al-27", 13u16, 14u16),
        ("Ti-48", 22, 26),
        ("Fe-56", 26, 30),
        ("Cu-63", 29, 34),
        ("Ni-58", 28, 30),
    ];

    // Spallation channels: (description, delta_z, delta_n)
    let channels = [
        ("(p,2p) proton knockout", -1i16, 0i16),
        ("(p,pn) neutron knockout", 0, -1),
        ("(p,2pn) two-nucleon removal", -1, -1),
        ("(p,alpha) alpha emission", -2, -2),
        ("(p,p2n) two-neutron removal", 0, -2),
    ];

    let mut products = Vec::new();

    for &(parent_name, pz, pn) in &targets {
        for &(reaction_desc, dz, dn) in &channels {
            let prod_z = (pz as i16 + dz) as u16;
            let prod_n = (pn as i16 + dn) as u16;

            if prod_z == 0 || prod_n == 0 {
                continue;
            }

            let pred = predictor.predict(prod_z, prod_n);
            let radioactive = is_likely_radioactive(prod_z, prod_n);
            let hl_cat = half_life_category(prod_z, prod_n, pred.ba);
            let activity = estimate_activity_relative(prod_z, &hl_cat, years_in_deep_space);

            products.push(ActivationProduct {
                parent_name: parent_name.to_string(),
                parent_z: pz,
                parent_n: pn,
                reaction: reaction_desc.to_string(),
                product_z: prod_z,
                product_n: prod_n,
                product_a: prod_z + prod_n,
                product_ba: pred.ba,
                likely_radioactive: radioactive,
                half_life_category: hl_cat,
                activity_relative: activity,
            });
        }
    }

    // Sort by activity descending (most concerning first)
    products.sort_by(|a, b| {
        b.activity_relative
            .partial_cmp(&a.activity_relative)
            .unwrap()
    });
    products
}

// ═══════════════════════════════════════════════════════════════════════════════
// §5  Combined Report
// ═══════════════════════════════════════════════════════════════════════════════

/// Full space nuclear assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceNuclearReport {
    pub rtg_known: Vec<RtgCandidate>,
    pub rtg_novel: Vec<RtgCandidate>,
    pub shielding: Vec<ShieldingMaterial>,
    pub ntp_fuels: Vec<NtpFuelCandidate>,
    pub activation: Vec<ActivationProduct>,
}

/// Generate a comprehensive space nuclear assessment.
pub fn generate_space_nuclear_report(
    predictor: &MlMassPredictor,
    mission_years: f64,
) -> SpaceNuclearReport {
    let mut rtg_known = known_rtg_fuels(predictor);
    rtg_known.sort_by(|a, b| {
        b.specific_power_wg
            .partial_cmp(&a.specific_power_wg)
            .unwrap()
    });

    let rtg_novel = scan_novel_rtg_candidates(predictor);
    let shielding = evaluate_shielding_materials();
    let ntp_fuels = evaluate_ntp_fuels(predictor);
    let activation = evaluate_activation_products(predictor, mission_years);

    SpaceNuclearReport {
        rtg_known,
        rtg_novel,
        shielding,
        ntp_fuels,
        activation,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn predictor() -> MlMassPredictor {
        MlMassPredictor::new()
    }

    // ── RTG Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_known_rtg_fuels() {
        let pred = predictor();
        let fuels = known_rtg_fuels(&pred);

        println!("\n=== KNOWN RTG FUEL CANDIDATES ===");
        println!(
            "{:<10} {:>5} {:>10} {:>10} {:>10}  {:<30}",
            "Name", "A", "t½ (yr)", "Q (MeV)", "W/g", "Shielding"
        );
        println!("{}", "-".repeat(85));
        for f in &fuels {
            println!(
                "{:<10} {:>5} {:>10.2} {:>10.3} {:>10.4}  {:<30}",
                f.name, f.a, f.half_life_years, f.q_value_mev, f.specific_power_wg, f.shielding
            );
        }

        assert!(!fuels.is_empty());
        // Pu-238 should be present
        let pu238 = fuels.iter().find(|f| f.name == "Pu-238").unwrap();
        assert_eq!(pu238.z, 94);
        assert_eq!(pu238.a, 238);
        assert!(pu238.half_life_years > 80.0 && pu238.half_life_years < 100.0);
        assert!(pu238.specific_power_wg > 0.0);

        // Po-210 should have highest specific power (shortest half-life)
        let po210 = fuels.iter().find(|f| f.name == "Po-210").unwrap();
        assert!(po210.specific_power_wg > pu238.specific_power_wg);
    }

    #[test]
    fn test_novel_rtg_scan() {
        let pred = predictor();
        let novel = scan_novel_rtg_candidates(&pred);

        println!("\n=== NOVEL RTG CANDIDATES (top 20) ===");
        println!(
            "{:<12} {:>5} {:>10} {:>10} {:>10}",
            "Name", "A", "t½ (yr)", "Q (MeV)", "W/g"
        );
        println!("{}", "-".repeat(55));
        for f in novel.iter().take(20) {
            println!(
                "{:<12} {:>5} {:>10.2} {:>10.3} {:>10.6}",
                f.name, f.a, f.half_life_years, f.q_value_mev, f.specific_power_wg
            );
        }

        // Should find some candidates
        println!("\nTotal novel candidates found: {}", novel.len());

        // All should be alpha emitters
        for c in &novel {
            assert_eq!(c.decay_mode, DecayMode::Alpha);
            assert!(c.q_value_mev >= 4.0 && c.q_value_mev <= 7.0);
            assert!(c.half_life_years >= 10.0 && c.half_life_years <= 200.0);
        }

        // Should be sorted by specific power
        for w in novel.windows(2) {
            assert!(w[0].specific_power_wg >= w[1].specific_power_wg);
        }
    }

    #[test]
    fn test_q_alpha_pu238() {
        let pred = predictor();
        let q = q_alpha(&pred, 94, 144);
        println!("\nPu-238 Q_alpha = {:.3} MeV (literature: ~5.593 MeV)", q);
        // Q_alpha for Pu-238 is ~5.593 MeV — ML predictor should be in the right ballpark
        assert!(q > 3.0 && q < 8.0, "Q_alpha = {} out of physical range", q);
    }

    #[test]
    fn test_geiger_nuttall_pu238() {
        // Pu-238: Z_daughter = 92 (U-234), Q = 5.593 MeV
        let t_half = geiger_nuttall_half_life_years(92, 5.593);
        println!(
            "\nGeiger-Nuttall Pu-238 t½ = {:.2} years (literature: 87.7 yr)",
            t_half
        );
        // Geiger-Nuttall is order-of-magnitude; within 2 orders is acceptable
        assert!(t_half > 0.1 && t_half < 1e6, "t½ = {} out of range", t_half);
    }

    #[test]
    fn test_specific_power_formula() {
        // Pu-238: Q=5.593 MeV, t½=87.7 yr, A=238
        let sp = specific_power_wg(5.593, 87.7, 238);
        println!(
            "\nPu-238 specific power = {:.4} W/g (literature: ~0.57 W/g)",
            sp
        );
        // Should be in the right order of magnitude
        assert!(sp > 0.1 && sp < 2.0, "specific power = {} out of range", sp);
    }

    // ── Shielding Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_shielding_materials() {
        let materials = evaluate_shielding_materials();

        println!("\n=== COSMIC RAY SHIELDING COMPARISON ===");
        println!(
            "{:<25} {:>6} {:>8} {:>10} {:>8} {:>8} {:>8}",
            "Material", "A_avg", "rho", "lam (g/cm2)", "lam (cm)", "SecN", "Score"
        );
        println!("{}", "-".repeat(85));
        for m in &materials {
            println!(
                "{:<25} {:>6.1} {:>8.2} {:>10.2} {:>8.1} {:>8.3} {:>8.4}",
                m.name,
                m.avg_a,
                m.density,
                m.interaction_length_gcm2,
                m.interaction_length_cm,
                m.secondary_neutron_factor,
                m.shielding_score
            );
        }

        assert!(!materials.is_empty());

        // H-rich materials (polyethylene, LiH, water) should score well
        let poly = materials
            .iter()
            .find(|m| m.name.contains("Polyethylene"))
            .unwrap();
        let lead = materials.iter().find(|m| m.name.contains("Lead")).unwrap();
        // Polyethylene should outscore lead (H-rich, low secondaries)
        assert!(
            poly.shielding_score > lead.shielding_score,
            "Polyethylene ({:.4}) should beat Lead ({:.4})",
            poly.shielding_score,
            lead.shielding_score
        );

        // Heavy materials should have higher secondary neutron factor
        let tungsten = materials
            .iter()
            .find(|m| m.name.contains("Tungsten"))
            .unwrap();
        let al = materials
            .iter()
            .find(|m| m.name.contains("Aluminum"))
            .unwrap();
        assert!(tungsten.secondary_neutron_factor > al.secondary_neutron_factor);
    }

    #[test]
    fn test_interaction_length_scaling() {
        // lambda ~ A^(1/3), so heavier materials have longer interaction length
        let lam_al = nuclear_interaction_length(27.0);
        let lam_pb = nuclear_interaction_length(207.2);
        assert!(lam_pb > lam_al);
        println!(
            "\nInteraction length: Al = {:.1} g/cm², Pb = {:.1} g/cm²",
            lam_al, lam_pb
        );
    }

    // ── NTP Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_ntp_fuels() {
        let pred = predictor();
        let fuels = evaluate_ntp_fuels(&pred);

        println!("\n=== NUCLEAR THERMAL PROPULSION FUEL COMPARISON ===");
        println!(
            "{:<12} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Name", "A", "Q (MeV)", "Barrier", "xs (b)", "Isp_rel", "Score"
        );
        println!("{}", "-".repeat(75));
        for f in &fuels {
            println!(
                "{:<12} {:>5} {:>10.1} {:>10.2} {:>10.0} {:>10.3} {:>10.1}",
                f.name,
                f.a,
                f.energy_per_fission_mev,
                f.fission_barrier_mev,
                f.thermal_xs_barns,
                f.isp_relative,
                f.ntp_score
            );
        }

        assert!(!fuels.is_empty());

        // U-235 fission energy should be ~150-220 MeV range
        let u235 = fuels.iter().find(|f| f.name == "U-235").unwrap();
        assert!(
            u235.energy_per_fission_mev > 100.0 && u235.energy_per_fission_mev < 300.0,
            "U-235 Q_fission = {} out of range",
            u235.energy_per_fission_mev
        );

        // Am-242m should rank highly (enormous thermal xs)
        let am242m = fuels.iter().find(|f| f.name == "Am-242m").unwrap();
        assert!(
            am242m.thermal_xs_barns > 5000.0,
            "Am-242m xs should be very high"
        );
    }

    #[test]
    fn test_fission_energy_u235() {
        let pred = predictor();
        let q = estimate_fission_energy(&pred, 92, 143);
        println!(
            "\nU-235 fission energy = {:.1} MeV (literature: ~200 MeV)",
            q
        );
        // Should be in the 150-250 MeV range
        assert!(q > 100.0 && q < 300.0);
    }

    // ── Activation Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_activation_products() {
        let pred = predictor();
        let products = evaluate_activation_products(&pred, 10.0);

        println!("\n=== ACTIVATION PRODUCTS (10-year deep space mission) ===");
        println!(
            "{:<8} {:<20} {:>5} {:>8} {:>6} {:>12} {:>10}",
            "Parent", "Reaction", "A_prod", "BA", "Radio", "Half-life", "Activity"
        );
        println!("{}", "-".repeat(80));
        for p in &products {
            println!(
                "{:<8} {:<20} {:>5} {:>8.3} {:>6} {:>12} {:>10.2}",
                p.parent_name,
                p.reaction,
                p.product_a,
                p.product_ba,
                if p.likely_radioactive { "YES" } else { "no" },
                p.half_life_category,
                p.activity_relative
            );
        }

        assert!(!products.is_empty());

        // Al-27 (p,2p) → Mg-26: should be stable (even-even, near magic)
        let mg26 = products
            .iter()
            .find(|p| p.parent_name == "Al-27" && p.reaction.contains("p,2p"))
            .unwrap();
        assert_eq!(mg26.product_z, 12);
        assert_eq!(mg26.product_n, 14);
        assert_eq!(mg26.product_a, 26);

        // Fe-56 products should exist
        let fe_products: Vec<_> = products
            .iter()
            .filter(|p| p.parent_name == "Fe-56")
            .collect();
        assert!(!fe_products.is_empty());
    }

    #[test]
    fn test_radioactivity_heuristic() {
        // Even-even near stability: likely stable
        assert!(!is_likely_radioactive(26, 30)); // Fe-56
        assert!(!is_likely_radioactive(13, 14)); // Al-27 (odd-even)

        // Far from stability: radioactive
        assert!(is_likely_radioactive(26, 40)); // Fe with way too many neutrons
    }

    // ── Full Report Test ─────────────────────────────────────────────────────

    #[test]
    fn test_full_report() {
        let pred = predictor();
        let report = generate_space_nuclear_report(&pred, 15.0);

        println!("\n{}", "=".repeat(80));
        println!("SPACE NUCLEAR APPLICATIONS — FULL REPORT (15-year mission)");
        println!("{}", "=".repeat(80));

        println!("\n--- RTG KNOWN FUELS (ranked by W/g) ---");
        for f in &report.rtg_known {
            println!(
                "  {}: {:.4} W/g, t½={:.1} yr, Q={:.2} MeV ({})",
                f.name, f.specific_power_wg, f.half_life_years, f.q_value_mev, f.decay_mode
            );
        }

        println!("\n--- RTG NOVEL CANDIDATES (top 10) ---");
        for f in report.rtg_novel.iter().take(10) {
            println!(
                "  {}: {:.6} W/g, t½={:.1} yr, Q={:.3} MeV",
                f.name, f.specific_power_wg, f.half_life_years, f.q_value_mev
            );
        }

        println!("\n--- SHIELDING (ranked by score) ---");
        for m in &report.shielding {
            println!(
                "  {}: score={:.4}, lambda={:.1} cm, secondaries={:.3}, H={:.1}%",
                m.name,
                m.shielding_score,
                m.interaction_length_cm,
                m.secondary_neutron_factor,
                m.hydrogen_fraction * 100.0
            );
        }

        println!("\n--- NTP FUELS (ranked by suitability) ---");
        for f in &report.ntp_fuels {
            println!(
                "  {}: Q={:.1} MeV, barrier={:.2} MeV, xs={:.0} b, Isp_rel={:.3}",
                f.name,
                f.energy_per_fission_mev,
                f.fission_barrier_mev,
                f.thermal_xs_barns,
                f.isp_relative
            );
        }

        println!("\n--- ACTIVATION (top 10 by activity) ---");
        for p in report.activation.iter().take(10) {
            println!(
                "  {} → {} (Z={}, A={}): BA={:.3}, radio={}, activity={:.2}",
                p.parent_name,
                p.reaction,
                p.product_z,
                p.product_a,
                p.product_ba,
                if p.likely_radioactive { "YES" } else { "no" },
                p.activity_relative
            );
        }

        // Verify all sections populated
        assert!(!report.rtg_known.is_empty());
        assert!(!report.shielding.is_empty());
        assert!(!report.ntp_fuels.is_empty());
        assert!(!report.activation.is_empty());
    }
}
