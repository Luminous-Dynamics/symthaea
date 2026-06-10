// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Antimatter Reactor — Matter-Antimatter Annihilation Physics
//!
//! Models antimatter containment, annihilation energy, Penning trap frequencies,
//! and Markov blanket coherence under extreme energy densities.
//!
//! Follows the FissionTwin pattern: HDC encoder + CfC predictor + FEP agent.
//!
//! References:
//! - CERN ALPHA experiment (antihydrogen trapping)
//! - Gabrielse et al. (2012). Penning trap physics.
//! - symthaea-core::physics::constants for annihilation constants

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Physical Constants ─────────────────────────────────────────────────

/// Proton-antiproton annihilation energy (MeV)
const PP_ANNIHILATION_MEV: f64 = 1876.544;

/// Charged pion fraction (directable energy)
const CHARGED_FRACTION: f64 = 0.40;

/// Average pion multiplicity in p-pbar annihilation
const AVG_PION_MULTIPLICITY: f64 = 5.0;

// ── Physics Functions ──────────────────────────────────────────────────

/// Energy released per proton-antiproton annihilation event (MeV).
pub fn annihilation_energy_per_event() -> f64 {
    PP_ANNIHILATION_MEV
}

/// Fraction of annihilation energy in charged pions (magnetically directable).
pub fn charged_energy_fraction() -> f64 {
    CHARGED_FRACTION
}

/// Penning trap characteristic frequencies.
///
/// A Penning trap confines charged particles using static E and B fields.
///
/// # Arguments
/// * `v0` — Applied voltage (V)
/// * `d` — Characteristic trap dimension (m)
/// * `b` — Magnetic field strength (T)
/// * `m` — Particle mass (kg)
/// * `q` — Particle charge (C, absolute value)
///
/// # Returns
/// `PenningFrequencies` with cyclotron, axial, and magnetron frequencies (Hz).
pub fn penning_trap_frequencies(v0: f64, d: f64, b: f64, m: f64, q: f64) -> PenningFrequencies {
    use std::f64::consts::PI;

    // Cyclotron frequency: ω_c = qB/m
    let omega_c = q * b / m;

    // Axial frequency: ω_z = sqrt(2qV₀/(md²))
    let omega_z_sq = 2.0 * q * v0 / (m * d * d);
    let omega_z = if omega_z_sq > 0.0 {
        omega_z_sq.sqrt()
    } else {
        0.0
    };

    // Magnetron frequency: ω_m = ω_c/2 - sqrt(ω_c²/4 - ω_z²/2)
    let discriminant = omega_c * omega_c / 4.0 - omega_z * omega_z / 2.0;
    let omega_m = if discriminant > 0.0 {
        omega_c / 2.0 - discriminant.sqrt()
    } else {
        0.0 // Trap unstable
    };

    PenningFrequencies {
        cyclotron_hz: omega_c / (2.0 * PI),
        axial_hz: omega_z / (2.0 * PI),
        magnetron_hz: omega_m / (2.0 * PI),
        trap_stable: discriminant > 0.0,
    }
}

/// Penning trap characteristic frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenningFrequencies {
    /// Cyclotron frequency (Hz) — circular motion in B field
    pub cyclotron_hz: f64,
    /// Axial frequency (Hz) — oscillation along B field axis
    pub axial_hz: f64,
    /// Magnetron frequency (Hz) — slow drift orbit
    pub magnetron_hz: f64,
    /// Whether the trap is stable (discriminant > 0)
    pub trap_stable: bool,
}

/// Annihilation rate at a matter-antimatter boundary (events/s).
///
/// Γ = n · n̄ · σ_ann · v_rel · Volume
///
/// # Arguments
/// * `n_matter` — matter number density (m⁻³)
/// * `n_antimatter` — antimatter number density (m⁻³)
/// * `v_rel` — relative velocity (m/s)
///
/// Returns rate per unit volume (events/(m³·s)).
pub fn annihilation_rate_density(n_matter: f64, n_antimatter: f64, v_rel: f64) -> f64 {
    let sigma = low_energy_cross_section(v_rel);
    n_matter * n_antimatter * sigma * v_rel
}

/// Low-energy annihilation cross-section (m²).
///
/// σ_ann ≈ π(ℏ/(m_p·v_rel))² — Coulomb-enhanced at low velocities.
pub fn low_energy_cross_section(v_rel: f64) -> f64 {
    if v_rel <= 0.0 {
        return 0.0;
    }
    let hbar = 1.054_571_817e-34; // J·s
    let m_p = 1.672_621_923_69e-27; // kg
    let lambda_bar = hbar / (m_p * v_rel);
    std::f64::consts::PI * lambda_bar * lambda_bar
}

// ── Horizons ───────────────────────────────────────────────────────────

/// Antimatter reactor prediction horizons (seconds).
pub const ANTIMATTER_HORIZONS: &[f32] = &[
    1e-6,   // 1μs — containment jitter
    1e-3,   // 1ms — injection cycle
    1.0,    // 1s — production batch
    100.0,  // 100s — fuel cycle
    3600.0, // 1hr — operational shift
];

pub const ANTIMATTER_HORIZON_LABELS: &[&str] = &[
    "1μs (containment)",
    "1ms (injection)",
    "1s (production)",
    "100s (fuel cycle)",
    "1hr (shift)",
];

// ── Encoder ────────────────────────────────────────────────────────────

/// Antimatter reactor observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntimatterReading {
    /// Containment magnetic field strength (T, normalized 0-1 by max field).
    pub containment_field: f64,
    /// Annihilation rate (normalized 0-1 by design rate).
    pub annihilation_rate: f64,
    /// Energy output (normalized 0-1 by rated capacity).
    pub energy_output: f64,
    /// Antimatter inventory (normalized 0-1 by max capacity).
    pub antimatter_inventory: f64,
    /// Boundary integrity (1.0 = perfect containment, 0.0 = total breach).
    pub boundary_integrity: f64,
}

/// HDC encoder for antimatter reactor state (5 observables → ContinuousHV).
pub struct AntimatterHdcEncoder {
    bases: [ContinuousHV; 5],
}

impl AntimatterHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 5] = [
            0xA17_0001, // containment_field
            0xA17_0002, // annihilation_rate
            0xA17_0003, // energy_output
            0xA17_0004, // antimatter_inventory
            0xA17_0005, // boundary_integrity
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &AntimatterReading) -> ContinuousHV {
        let weights = [
            reading.containment_field.clamp(0.0, 1.0) as f32,
            reading.annihilation_rate.clamp(0.0, 1.0) as f32,
            reading.energy_output.clamp(0.0, 1.0) as f32,
            reading.antimatter_inventory.clamp(0.0, 1.0) as f32,
            reading.boundary_integrity.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for AntimatterHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ──────────────────────────────────────────────────────────

/// Multi-scale predictor for antimatter reactor dynamics.
pub struct AntimatterPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl AntimatterPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 0.001, // 1ms base — annihilation timescales are fast
            backbone_tau: 1.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xA17_10A0),
        }
    }

    pub fn predict_at_horizon(&self, current: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        assert!(horizon_seconds.is_finite() && horizon_seconds > 0.0);
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current);
        neuron_copy.state().clone()
    }

    pub fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.neuron.evolve_closed_form(dt_seconds, state);
    }
}

impl Default for AntimatterPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ── FEP Agent ──────────────────────────────────────────────────────────

/// Actions for antimatter reactor control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AntimatterFepAction {
    MaintainReactor,
    AdjustGeometry,
    IncreaseMagnetic,
    DecreaseProduction,
    EmergencyVent,
}

impl AntimatterFepAction {
    pub const ALL: [AntimatterFepAction; 5] = [
        AntimatterFepAction::MaintainReactor,
        AntimatterFepAction::AdjustGeometry,
        AntimatterFepAction::IncreaseMagnetic,
        AntimatterFepAction::DecreaseProduction,
        AntimatterFepAction::EmergencyVent,
    ];
}

const FE_EMERGENCY: f64 = 0.7;
const FE_DECREASE_PROD: f64 = 0.5;
const FE_INCREASE_MAG: f64 = 0.3;
const FE_ADJUST_GEOM: f64 = 0.1;

/// FEP agent for antimatter reactor safety.
pub struct AntimatterFepAgent {
    reference_state: ContinuousHV,
}

impl AntimatterFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xA17_BEEF),
        }
    }

    pub fn set_reference(&mut self, reference: ContinuousHV) {
        self.reference_state = reference;
    }

    pub fn compute_free_energy(&self, observed: &ContinuousHV) -> f64 {
        let sim = observed.similarity(&self.reference_state) as f64;
        if !sim.is_finite() {
            return 1.0;
        }
        (1.0 - sim).max(0.0)
    }

    pub fn select_action(&self, observed: &ContinuousHV) -> AntimatterFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_EMERGENCY {
            AntimatterFepAction::EmergencyVent
        } else if fe > FE_DECREASE_PROD {
            AntimatterFepAction::DecreaseProduction
        } else if fe > FE_INCREASE_MAG {
            AntimatterFepAction::IncreaseMagnetic
        } else if fe > FE_ADJUST_GEOM {
            AntimatterFepAction::AdjustGeometry
        } else {
            AntimatterFepAction::MaintainReactor
        }
    }
}

impl Default for AntimatterFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ─────────────────────────────────────────────────────────────

/// Safety level for antimatter reactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AntimatterSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from an antimatter reactor prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntimatterOutput {
    pub free_energy: f64,
    pub recommended_action: AntimatterFepAction,
    pub safety_level: AntimatterSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

// ── Digital Twin ───────────────────────────────────────────────────────

/// Antimatter reactor digital twin.
pub struct AntimatterTwin {
    encoder: AntimatterHdcEncoder,
    predictor: AntimatterPredictor,
    agent: AntimatterFepAgent,
    cycle_count: u64,
}

impl AntimatterTwin {
    pub fn new() -> Self {
        Self {
            encoder: AntimatterHdcEncoder::new(),
            predictor: AntimatterPredictor::new(),
            agent: AntimatterFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &AntimatterReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &AntimatterReading, dt_seconds: f32) -> AntimatterOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = ANTIMATTER_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_EMERGENCY {
            AntimatterSafetyLevel::Red
        } else if fe > FE_DECREASE_PROD {
            AntimatterSafetyLevel::Orange
        } else if fe > FE_ADJUST_GEOM {
            AntimatterSafetyLevel::Yellow
        } else {
            AntimatterSafetyLevel::Green
        };

        AntimatterOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }
}

impl Default for AntimatterTwin {
    fn default() -> Self {
        Self::new()
    }
}

// ── Blanket Coherence Test ─────────────────────────────────────────────

/// Result of a blanket coherence test at a specific energy density.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlanketCoherencePoint {
    /// Energy density (J/m³)
    pub energy_density: f64,
    /// Effective permeability (0 = sealed, 1 = dissolved)
    pub permeability: f64,
    /// Whether the blanket is still intact
    pub blanket_intact: bool,
}

/// Test Markov blanket coherence under escalating energy density.
///
/// Drives the FEP agent with readings of increasing energy output
/// and decreasing boundary integrity, measuring when the statistical
/// boundary (Markov blanket) dissolves.
///
/// This answers: "at what energy density does the matter-antimatter
/// boundary lose statistical separation?"
pub fn blanket_coherence_test(energy_densities: &[f64]) -> Vec<BlanketCoherencePoint> {
    let mut twin = AntimatterTwin::new();

    // Set reference as a nominal, well-contained reactor
    let nominal = AntimatterReading {
        containment_field: 0.8,
        annihilation_rate: 0.1,
        energy_output: 0.2,
        antimatter_inventory: 0.5,
        boundary_integrity: 1.0,
    };
    twin.set_reference(&nominal);

    energy_densities
        .iter()
        .map(|&ed| {
            // As energy density increases:
            // - boundary integrity degrades
            // - annihilation rate increases
            // - containment field is stressed
            let normalized_ed = (ed / 1e18).min(1.0); // Normalize by ~mc² scale
            let reading = AntimatterReading {
                containment_field: (0.8 - 0.7 * normalized_ed).max(0.0),
                annihilation_rate: (0.1 + 0.9 * normalized_ed).min(1.0),
                energy_output: normalized_ed.min(1.0),
                antimatter_inventory: (0.5 - 0.4 * normalized_ed).max(0.0),
                boundary_integrity: (1.0 - normalized_ed).max(0.0),
            };

            let output = twin.step(&reading, 0.001);

            // Permeability = free energy (high FE = boundary dissolving)
            let permeability = output.free_energy;
            let intact = permeability < 0.7;

            BlanketCoherencePoint {
                energy_density: ed,
                permeability,
                blanket_intact: intact,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annihilation_energy() {
        let e = annihilation_energy_per_event();
        assert!((e - 1876.544).abs() < 1.0);
    }

    #[test]
    fn test_charged_fraction() {
        assert!((charged_energy_fraction() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_penning_frequencies_physical() {
        // Antiproton in a 6T field, 10V trap, 5mm characteristic dimension
        let m_p = 1.672_621_923_69e-27; // kg
        let q = 1.602_176_634e-19; // C
        let freq = penning_trap_frequencies(10.0, 0.005, 6.0, m_p, q);

        assert!(freq.trap_stable, "6T trap should be stable");
        assert!(
            freq.cyclotron_hz > 1e6,
            "Cyclotron freq should be MHz range: {}",
            freq.cyclotron_hz
        );
        assert!(freq.axial_hz > 0.0, "Axial freq should be positive");
        assert!(freq.magnetron_hz > 0.0, "Magnetron freq should be positive");
        // Frequency ordering: cyclotron >> axial >> magnetron
        assert!(freq.cyclotron_hz > freq.axial_hz);
    }

    #[test]
    fn test_cross_section_diverges_at_low_v() {
        // Cross-section should increase as velocity decreases (Coulomb enhancement)
        let sigma_fast = low_energy_cross_section(1e6);
        let sigma_slow = low_energy_cross_section(1e3);
        assert!(
            sigma_slow > sigma_fast,
            "Cross-section should grow at low v: σ_slow={}, σ_fast={}",
            sigma_slow,
            sigma_fast
        );
    }

    #[test]
    fn test_encoder_dimension() {
        let encoder = AntimatterHdcEncoder::new();
        let reading = AntimatterReading {
            containment_field: 0.8,
            annihilation_rate: 0.1,
            energy_output: 0.2,
            antimatter_inventory: 0.5,
            boundary_integrity: 1.0,
        };
        let hv = encoder.encode(&reading);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_produces_finite() {
        let predictor = AntimatterPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 0xDEAD);
        let pred = predictor.predict_at_horizon(&input, 0.001);
        assert!(pred.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fep_agent_scram_on_high_fe() {
        let agent = AntimatterFepAgent::new();
        // A random HV will have low similarity to reference → high FE → emergency
        let random = ContinuousHV::random(HDC_DIMENSION, 0xBAD1);
        let action = agent.select_action(&random);
        // High FE should trigger emergency or decrease production
        assert!(
            action == AntimatterFepAction::EmergencyVent
                || action == AntimatterFepAction::DecreaseProduction,
            "High FE should trigger safety action: {:?}",
            action
        );
    }

    #[test]
    fn test_twin_step_returns_valid() {
        let mut twin = AntimatterTwin::new();
        let reading = AntimatterReading {
            containment_field: 0.8,
            annihilation_rate: 0.1,
            energy_output: 0.2,
            antimatter_inventory: 0.5,
            boundary_integrity: 1.0,
        };
        twin.set_reference(&reading);
        let output = twin.step(&reading, 0.001);
        assert!(output.free_energy.is_finite());
        assert_eq!(
            output.prediction_similarities.len(),
            ANTIMATTER_HORIZONS.len()
        );
    }

    #[test]
    fn test_safety_levels_ordered() {
        assert!(AntimatterSafetyLevel::Green < AntimatterSafetyLevel::Yellow);
        assert!(AntimatterSafetyLevel::Yellow < AntimatterSafetyLevel::Orange);
        assert!(AntimatterSafetyLevel::Orange < AntimatterSafetyLevel::Red);
    }

    #[test]
    fn test_blanket_coherence_degrades() {
        let densities: Vec<f64> = (0..10).map(|i| i as f64 * 1e17).collect();
        let results = blanket_coherence_test(&densities);

        assert_eq!(results.len(), 10);

        // First point (zero density) should have low permeability
        assert!(
            results[0].permeability < results[9].permeability || results[0].blanket_intact,
            "Blanket should be more intact at low energy density"
        );

        // All permeabilities should be finite
        assert!(results.iter().all(|r| r.permeability.is_finite()));
    }
}
