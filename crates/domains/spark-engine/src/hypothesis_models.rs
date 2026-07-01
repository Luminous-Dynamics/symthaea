// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Detailed Hypothesis Models (Option A)
//!
//! Physics simulations for each anomaly hypothesis to generate testable predictions.

use crate::constants::*;
use crate::physics::GamowIntegration;
use serde::{Deserialize, Serialize};

/// Hypothesis type for classifying anomaly explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HypothesisType {
    /// Transient hot spots from X-ray absorption
    HotSpots,
    /// Coherent phonon cascade enhancement
    PhononCascade,
    /// Enhanced electron screening beyond measured values
    SuperScreening,
    /// Lattice-modified nuclear interaction (speculative)
    LatticeNuclear,
    /// Systematic measurement error
    MeasurementError,
}

// ============================================================================
// Hypothesis 1: Transient Hot Spots
// ============================================================================

/// Model for transient hot spots from X-ray energy deposition.
///
/// When X-rays are absorbed, they deposit energy in a localized region.
/// If thermalization is slow, this creates a "hot spot" with T >> T_bulk
/// for a brief time, dramatically increasing local <σv>.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSpotModel {
    /// X-ray energy (keV)
    pub xray_energy_kev: f64,
    /// X-ray flux (photons/cm²/s)
    pub xray_flux: f64,
    /// Photoelectron range (nm)
    pub electron_range_nm: f64,
    /// Hot spot radius (nm)
    pub hot_spot_radius_nm: f64,
    /// Peak temperature (K)
    pub peak_temperature_k: f64,
    /// Cooling time constant (ps)
    pub cooling_time_ps: f64,
    /// Effective <σv> during hot phase
    pub effective_sigma_v: f64,
    /// Predicted neutron rate (n/s/cm²)
    pub predicted_rate: f64,
}

impl HotSpotModel {
    /// Compute hot spot model for given X-ray conditions.
    pub fn compute(xray_energy_kev: f64, xray_flux: f64) -> Self {
        // Photoelectron energy ≈ X-ray energy - binding energy
        // Pd K-edge is 24.35 keV, L-edge ~3 keV
        let electron_energy_kev = if xray_energy_kev > 24.35 {
            xray_energy_kev - 24.35
        } else if xray_energy_kev > 3.0 {
            xray_energy_kev - 3.0
        } else {
            xray_energy_kev * 0.9
        };

        // Electron range in Pd (nm) - CSDA approximation
        // R ≈ 40 × E^1.67 nm for E in keV (very rough)
        let electron_range_nm = 40.0 * electron_energy_kev.powf(1.67);

        // Hot spot radius ≈ electron range / 3 (cascade spreads)
        let hot_spot_radius_nm = electron_range_nm / 3.0;

        // Energy deposited per hot spot (J)
        let energy_j = xray_energy_kev * 1000.0 * 1.602e-19;

        // Hot spot volume (cm³)
        let radius_cm = hot_spot_radius_nm * 1e-7;
        let volume_cm3 = (4.0 / 3.0) * std::f64::consts::PI * radius_cm.powi(3);

        // Heat capacity of Pd: C = 25.98 J/mol/K, ρ = 12.02 g/cm³, M = 106.42 g/mol
        // C_vol = 25.98 × 12.02 / 106.42 = 2.93 J/cm³/K
        let heat_capacity = 2.93;
        let _mass_volume = volume_cm3 * 12.02; // grams in hot spot

        // Temperature rise: ΔT = E / (m × c)
        // Using volumetric: ΔT = E / (V × C_vol)
        let delta_t = energy_j / (volume_cm3 * heat_capacity);
        let peak_temperature_k = 300.0 + delta_t;

        // Cooling time: τ ≈ r² / α where α is thermal diffusivity
        // Pd: α ≈ 0.25 cm²/s = 2.5e7 nm²/ps
        let alpha_nm2_ps = 2.5e7;
        let cooling_time_ps = hot_spot_radius_nm.powi(2) / alpha_nm2_ps;

        // Compute <σv> at peak temperature
        let gamow_hot = GamowIntegration::dd_rate(peak_temperature_k, SCREENING_PD_EV, 0);
        let _gamow_cold = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, 0);
        let effective_sigma_v = gamow_hot.sigma_v_cm3_s;

        // Neutron rate calculation
        // Number of D atoms in hot spot
        let pd_atoms = volume_cm3 * 12.02 * 6.022e23 / 106.42;
        let d_atoms = pd_atoms * 0.7; // D/Pd = 0.7

        // Reaction rate during hot phase (per hot spot)
        let reaction_rate = d_atoms.powi(2) * effective_sigma_v / 4.0;

        // Integrate over hot phase (approximate as triangular pulse)
        let hot_duration_s = cooling_time_ps * 1e-12;
        let reactions_per_hot_spot = reaction_rate * hot_duration_s / 2.0;

        // Hot spots per second per cm² = X-ray absorption rate
        // Absorption probability ≈ 1 - exp(-μρx) ≈ μρx for thin targets
        // For 12 keV in Pd: μ ≈ 100 cm²/g, x = 25 μm = 0.0025 cm
        let absorption_prob = 0.9; // Most X-rays absorbed in 25 μm
        let hot_spots_per_s_cm2 = xray_flux * absorption_prob;

        // Neutrons per hot spot (50% go to n channel)
        let neutrons_per_hot_spot = reactions_per_hot_spot * 0.5;

        let predicted_rate = hot_spots_per_s_cm2 * neutrons_per_hot_spot;

        Self {
            xray_energy_kev,
            xray_flux,
            electron_range_nm,
            hot_spot_radius_nm,
            peak_temperature_k,
            cooling_time_ps,
            effective_sigma_v,
            predicted_rate,
        }
    }

    /// Compare prediction to NASA observation.
    pub fn gap_from_nasa(&self) -> f64 {
        NASA_OBSERVED_RATE / self.predicted_rate
    }

    /// Temperature needed to explain NASA rate.
    pub fn required_temperature_for_nasa(&self) -> f64 {
        // Binary search for temperature that gives NASA rate
        let mut t_low = 300.0;
        let mut t_high = 1e9;

        for _ in 0..100 {
            let t_mid = (t_low + t_high) / 2.0;
            // Note: we use self.hot_spot_radius_nm etc. for geometry, just vary temperature

            // Recompute rate at this temperature
            let gamow = GamowIntegration::dd_rate(t_mid, SCREENING_PD_EV, 0);
            let volume_cm3 =
                (4.0 / 3.0) * std::f64::consts::PI * (self.hot_spot_radius_nm * 1e-7).powi(3);
            let pd_atoms = volume_cm3 * 12.02 * 6.022e23 / 106.42;
            let d_atoms = pd_atoms * 0.7;
            let reaction_rate = d_atoms.powi(2) * gamow.sigma_v_cm3_s / 4.0;
            let hot_duration_s = self.cooling_time_ps * 1e-12;
            let reactions_per_hot_spot = reaction_rate * hot_duration_s / 2.0;
            let predicted = self.xray_flux * 0.9 * reactions_per_hot_spot * 0.5;

            if predicted > NASA_OBSERVED_RATE {
                t_high = t_mid;
            } else {
                t_low = t_mid;
            }

            if (t_high - t_low) < 100.0 {
                break;
            }
        }

        t_high
    }
}

// ============================================================================
// Hypothesis 2: Coherent Phonon Enhancement
// ============================================================================

/// Model for coherent phonon cascade enhancement.
///
/// Multiple phonons coherently add their energy to the D-D center-of-mass,
/// effectively boosting the collision energy by N × ℏω.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhononCascadeModel {
    /// Base phonon energy (meV)
    pub phonon_energy_mev: f64,
    /// Number of coherent phonons
    pub n_phonons: u32,
    /// Coherence probability (per D-D pair)
    pub coherence_prob: f64,
    /// Effective CM energy boost (keV)
    pub energy_boost_kev: f64,
    /// Enhancement factor over thermal
    pub enhancement_factor: f64,
    /// Predicted rate (n/s/cm³)
    pub predicted_rate_per_cm3: f64,
}

impl PhononCascadeModel {
    /// Compute phonon cascade model.
    ///
    /// # Arguments
    /// * `n_phonons` - Number of coherent phonon modes
    /// * `coherence_prob` - Probability of coherent state per D-D pair
    pub fn compute(n_phonons: u32, coherence_prob: f64) -> Self {
        let phonon_energy_mev = 56.0; // PdD optical phonon
        let phonon_energy_kev = phonon_energy_mev / 1000.0;
        let energy_boost_kev = (n_phonons as f64) * phonon_energy_kev;

        // Compute rate with phonon enhancement
        let gamow_enhanced = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, n_phonons);
        let gamow_base = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, 0);

        let enhancement_factor = if gamow_base.sigma_v_cm3_s > 0.0 {
            gamow_enhanced.sigma_v_cm3_s / gamow_base.sigma_v_cm3_s
        } else {
            1.0
        };

        // D number density
        let n_d: f64 = 0.7 * 12.02 * 6.022e23 / 106.42;

        // Reaction rate with coherence probability
        // Only fraction of D-D pairs are in coherent state
        let effective_sigma_v = gamow_enhanced.sigma_v_cm3_s * coherence_prob;
        let predicted_rate_per_cm3 = n_d.powi(2) * effective_sigma_v / 4.0 * 0.5; // 50% neutron branch

        Self {
            phonon_energy_mev,
            n_phonons,
            coherence_prob,
            energy_boost_kev,
            enhancement_factor,
            predicted_rate_per_cm3,
        }
    }

    /// Compute phonons needed to explain NASA rate.
    pub fn phonons_needed_for_nasa() -> u32 {
        // NASA: ~10³ n/s from ~0.01 cm³ → ~10⁵ n/s/cm³
        let target_rate = 1e5;

        for n in 1..100 {
            let model = Self::compute(n, 1.0); // Assume full coherence
            if model.predicted_rate_per_cm3 >= target_rate {
                return n;
            }
        }
        100 // Can't reach with reasonable phonon count
    }
}

// ============================================================================
// Hypothesis 3: Super-Screening
// ============================================================================

/// Model for enhanced screening beyond measured values.
///
/// At high D/Pd loading, screening might exceed the ~309 eV measured
/// at lower loading due to collective electronic effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperScreeningModel {
    /// Base screening energy (eV) at D/Pd = 0.7
    pub base_screening_ev: f64,
    /// Loading ratio D/Pd
    pub loading_ratio: f64,
    /// Effective screening energy (eV)
    pub effective_screening_ev: f64,
    /// Enhancement over base screening
    pub screening_enhancement: f64,
    /// Reaction rate enhancement
    pub rate_enhancement: f64,
    /// Predicted rate (n/s/cm³)
    pub predicted_rate_per_cm3: f64,
}

impl SuperScreeningModel {
    /// Compute super-screening model.
    ///
    /// Hypothesis: Ue increases with loading as Ue = Ue_base × (1 + k × (D/Pd - 0.7))
    /// where k is an unknown enhancement factor.
    pub fn compute(loading_ratio: f64, enhancement_factor: f64) -> Self {
        let base_screening_ev = SCREENING_PD_EV;

        // Enhanced screening (hypothetical)
        let effective_screening_ev =
            base_screening_ev * (1.0 + enhancement_factor * (loading_ratio - 0.7).max(0.0));

        // Compute rates with different screening
        let gamow_base = GamowIntegration::dd_rate(300.0, base_screening_ev, 2);
        let gamow_enhanced = GamowIntegration::dd_rate(300.0, effective_screening_ev, 2);

        let rate_enhancement = if gamow_base.sigma_v_cm3_s > 0.0 {
            gamow_enhanced.sigma_v_cm3_s / gamow_base.sigma_v_cm3_s
        } else {
            1.0
        };

        let screening_enhancement = effective_screening_ev / base_screening_ev;

        // D number density (scales with loading)
        let n_d = loading_ratio * 12.02 * 6.022e23 / 106.42;

        // Predicted rate
        let predicted_rate_per_cm3 = n_d.powi(2) * gamow_enhanced.sigma_v_cm3_s / 4.0 * 0.5;

        Self {
            base_screening_ev,
            loading_ratio,
            effective_screening_ev,
            screening_enhancement,
            rate_enhancement,
            predicted_rate_per_cm3,
        }
    }

    /// Compute screening needed to explain NASA rate.
    pub fn screening_needed_for_nasa() -> f64 {
        let target_rate = 1e5; // n/s/cm³

        let mut ue_low = SCREENING_PD_EV;
        let mut ue_high = 1e6; // 1 MeV max

        for _ in 0..100 {
            let ue_mid = (ue_low + ue_high) / 2.0;
            let gamow = GamowIntegration::dd_rate(300.0, ue_mid, 2);
            let n_d: f64 = 0.7 * 12.02 * 6.022e23 / 106.42;
            let rate = n_d.powi(2) * gamow.sigma_v_cm3_s / 4.0 * 0.5;

            if rate > target_rate {
                ue_high = ue_mid;
            } else {
                ue_low = ue_mid;
            }

            if (ue_high - ue_low) < 10.0 {
                break;
            }
        }

        ue_high
    }
}

// ============================================================================
// Hypothesis 4: Oppenheimer-Phillips Effect
// ============================================================================

/// Model for Oppenheimer-Phillips (stripping) reactions.
///
/// In the O-P mechanism, the neutron in a deuteron can be captured by
/// a target nucleus before the Coulomb barrier is fully penetrated,
/// because the neutron experiences no Coulomb repulsion. This effectively
/// reduces the barrier for certain reactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OppenheimerPhillipsModel {
    /// Target nucleus (typically Pd)
    pub target_z: u32,
    /// Target mass number
    pub target_a: u32,
    /// Effective barrier reduction (keV)
    pub barrier_reduction_kev: f64,
    /// Enhancement factor over direct D-D
    pub enhancement_factor: f64,
    /// Predicted neutron rate per cm³
    pub predicted_rate_per_cm3: f64,
    /// Expected products
    pub products: String,
}

impl OppenheimerPhillipsModel {
    /// Compute O-P model for D + Pd stripping.
    ///
    /// D + ¹⁰⁶Pd → n + ¹⁰⁷Ag (stripping)
    /// The neutron from D is captured by Pd, releasing the proton.
    pub fn compute_pd() -> Self {
        let target_z = 46; // Pd
        let target_a = 106;

        // Coulomb barrier for d + Pd: V_c = Z_d * Z_Pd * e² / R
        // R ≈ 1.2 * (A_d^(1/3) + A_Pd^(1/3)) fm
        // For d + Pd: R ≈ 1.2 * (2^0.33 + 106^0.33) ≈ 7.2 fm
        // V_c ≈ 1 * 46 * 1.44 / 7.2 ≈ 9.2 MeV

        // O-P reduces this because neutron can tunnel ahead
        // Effective barrier reduction: ~50% for favorable cases
        let full_barrier_kev = 9200.0;
        let barrier_reduction_kev = full_barrier_kev * 0.5;

        // Enhancement is enormous but reaction produces different products
        // This is NOT D-D fusion, so doesn't produce 2.45 MeV neutrons
        let enhancement_factor = 1e10; // Very rough estimate

        // However, O-P on Pd is endothermic (Q < 0), so rate is suppressed
        // D + ¹⁰⁶Pd → p + ¹⁰⁷Pd* requires energy input
        let predicted_rate_per_cm3 = 1e-30; // Negligible for energy-producing reaction

        Self {
            target_z,
            target_a,
            barrier_reduction_kev,
            enhancement_factor,
            predicted_rate_per_cm3,
            products: "p + ¹⁰⁷Pd (not D-D fusion products)".to_string(),
        }
    }

    /// Check if O-P could explain neutron observations.
    ///
    /// Key issue: O-P on most targets doesn't produce 2.45 MeV neutrons.
    /// If observations show 2.45 MeV neutrons, O-P is not the explanation.
    pub fn explains_dd_neutrons(&self) -> bool {
        // O-P produces different products than D-D, so cannot explain
        // observations of 2.45 MeV neutrons which are D-D signature
        false
    }
}

// ============================================================================
// Hypothesis 5: Electron Capture / Virtual Photon
// ============================================================================

/// Model for electron-assisted fusion.
///
/// A lattice electron could briefly form a virtual "dineutron" state,
/// reducing the Coulomb barrier. This is highly speculative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronAssistedModel {
    /// Electron screening contribution (eV)
    pub screening_ev: f64,
    /// Virtual state lifetime (s)
    pub virtual_lifetime_s: f64,
    /// Enhancement factor (highly uncertain)
    pub enhancement_factor: f64,
    /// Theoretical basis
    pub theory_status: String,
}

impl ElectronAssistedModel {
    /// Compute electron-assisted model.
    pub fn compute() -> Self {
        // This is highly speculative physics
        // The idea is that an electron could briefly screen the D-D
        // interaction more effectively than static screening

        // Virtual state lifetime from uncertainty principle
        // Δt ~ ℏ / ΔE, for ΔE ~ 1 keV, Δt ~ 10⁻¹⁸ s
        let virtual_lifetime_s = 1e-18;

        // Extra screening from dynamic electron
        let screening_ev = 500.0; // Speculative

        // Enhancement is modest because virtual state is brief
        let enhancement_factor = 10.0;

        Self {
            screening_ev,
            virtual_lifetime_s,
            enhancement_factor,
            theory_status: "Highly speculative - no accepted theoretical framework".to_string(),
        }
    }
}

// ============================================================================
// Combined Model Comparison
// ============================================================================

/// Comparison of all hypothesis models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisComparison {
    /// Hot spot model prediction
    pub hot_spot: HotSpotResult,
    /// Phonon cascade prediction
    pub phonon_cascade: PhononResult,
    /// Super-screening prediction
    pub super_screening: ScreeningResult,
    /// Which hypothesis best explains observations
    pub best_fit: String,
    /// Required conditions for each hypothesis
    pub required_conditions: Vec<RequiredCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSpotResult {
    pub predicted_rate: f64,
    pub gap_factor: f64,
    pub required_temperature_k: f64,
    pub plausible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhononResult {
    pub predicted_rate_n3: f64, // With 3 phonons
    pub gap_factor: f64,
    pub phonons_needed: u32,
    pub plausible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningResult {
    pub predicted_rate: f64,
    pub gap_factor: f64,
    pub screening_needed_ev: f64,
    pub plausible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredCondition {
    pub hypothesis: String,
    pub parameter: String,
    pub required_value: f64,
    pub unit: String,
    pub physically_reasonable: bool,
}

impl HypothesisComparison {
    /// Compare all hypotheses against NASA observation.
    pub fn compare_all() -> Self {
        let nasa_rate = NASA_OBSERVED_RATE;

        // Hot spot model (NASA conditions: 12 keV, ~10¹² photons/s/cm²)
        let hot_spot_model = HotSpotModel::compute(12.0, 1e12);
        let hot_spot = HotSpotResult {
            predicted_rate: hot_spot_model.predicted_rate,
            gap_factor: nasa_rate / hot_spot_model.predicted_rate.max(1e-100),
            required_temperature_k: hot_spot_model.required_temperature_for_nasa(),
            plausible: hot_spot_model.required_temperature_for_nasa() < 1e6, // < 1 million K
        };

        // Phonon cascade
        let phonon_model_3 = PhononCascadeModel::compute(3, 1.0);
        let phonons_needed = PhononCascadeModel::phonons_needed_for_nasa();
        let phonon_cascade = PhononResult {
            predicted_rate_n3: phonon_model_3.predicted_rate_per_cm3 * 0.01, // 0.01 cm³
            gap_factor: nasa_rate / (phonon_model_3.predicted_rate_per_cm3 * 0.01).max(1e-100),
            phonons_needed,
            plausible: phonons_needed < 20, // < 20 coherent phonons
        };

        // Super-screening
        let screening_model = SuperScreeningModel::compute(0.7, 0.0);
        let screening_needed = SuperScreeningModel::screening_needed_for_nasa();
        let super_screening = ScreeningResult {
            predicted_rate: screening_model.predicted_rate_per_cm3 * 0.01,
            gap_factor: nasa_rate / (screening_model.predicted_rate_per_cm3 * 0.01).max(1e-100),
            screening_needed_ev: screening_needed,
            plausible: screening_needed < 10000.0, // < 10 keV screening
        };

        // Determine best fit
        let best_fit = if hot_spot.plausible {
            "Hot Spots (transient high T)".to_string()
        } else if phonon_cascade.plausible {
            "Phonon Cascade (coherent energy boost)".to_string()
        } else if super_screening.plausible {
            "Super-Screening (enhanced Ue)".to_string()
        } else {
            "None - all require implausible conditions".to_string()
        };

        let required_conditions = vec![
            RequiredCondition {
                hypothesis: "Hot Spots".to_string(),
                parameter: "Peak temperature".to_string(),
                required_value: hot_spot.required_temperature_k,
                unit: "K".to_string(),
                physically_reasonable: hot_spot.required_temperature_k < 1e6,
            },
            RequiredCondition {
                hypothesis: "Phonon Cascade".to_string(),
                parameter: "Coherent phonons".to_string(),
                required_value: phonons_needed as f64,
                unit: "modes".to_string(),
                physically_reasonable: phonons_needed < 20,
            },
            RequiredCondition {
                hypothesis: "Super-Screening".to_string(),
                parameter: "Screening energy".to_string(),
                required_value: screening_needed,
                unit: "eV".to_string(),
                physically_reasonable: screening_needed < 10000.0,
            },
        ];

        Self {
            hot_spot,
            phonon_cascade,
            super_screening,
            best_fit,
            required_conditions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_spot_model() {
        let model = HotSpotModel::compute(12.0, 1e12);
        assert!(model.peak_temperature_k > 300.0);
        assert!(model.cooling_time_ps > 0.0);
    }

    #[test]
    fn test_phonon_cascade_model() {
        let model = PhononCascadeModel::compute(3, 1.0);
        assert!(model.enhancement_factor >= 1.0);
        assert!(model.energy_boost_kev > 0.0);
    }

    #[test]
    fn test_super_screening_model() {
        let model = SuperScreeningModel::compute(0.8, 1.0);
        assert!(model.effective_screening_ev >= model.base_screening_ev);
    }

    #[test]
    fn test_hypothesis_comparison() {
        let comparison = HypothesisComparison::compare_all();
        assert!(!comparison.best_fit.is_empty());
        assert!(!comparison.required_conditions.is_empty());
    }
}
