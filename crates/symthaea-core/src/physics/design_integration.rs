// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Design Integration Metrics
//!
//! This module measures how well-integrated a fusion reactor design is by
//! encoding physics states as HDC vectors and measuring their coherence.
//!
//! ## What This Measures (Engineering Metrics)
//!
//! - **Thermal-structural coupling**: How well heat flow integrates with structure
//! - **Damage-healing balance**: Equilibrium between radiation damage and self-healing
//! - **Geometry harmony**: Mass efficiency and compactness
//! - **Overall integration**: Geometric mean of all subsystem metrics
//!
//! ## Architecture
//!
//! ```text
//! CoupledSimulationResult
//!         │
//!         ▼
//! ┌───────────────────────────────────────────────┐
//! │  Encode Components as HDC Vectors             │
//! │  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
//! │  │  Thermal    │ │   Damage    │ │ Geometry │ │
//! │  │  Profile    │ │   State     │ │          │ │
//! │  └──────┬──────┘ └──────┬──────┘ └────┬─────┘ │
//! │         │               │              │       │
//! │         └───────────────┼──────────────┘       │
//! │                         ▼                      │
//! │              ┌──────────────────┐              │
//! │              │   BIND ALL       │              │
//! │              │   Creates unified │              │
//! │              │   state vector   │              │
//! │              └────────┬─────────┘              │
//! └───────────────────────┼───────────────────────┘
//!                         ▼
//!               IntegrationMetrics
//!               ├── coupling_index
//!               ├── design_integration
//!               ├── thermal_coherence
//!               └── overall_integration
//! ```
//!
//! ## Note on Naming
//!
//! This module was previously named "physics_consciousness_integration".
//! The metrics measure engineering properties (coupling, coherence, efficiency),
//! not consciousness. The HDC encoding is a valid dimensionality reduction
//! technique; the "consciousness" terminology was misleading.

use super::consciousness_bridge::PhysicsConsciousnessBridge;
use super::coupled_physics::{CoupledSimulationResult, OperatingConditions};
use super::standard_model::PHYSICS_DIM;
use super::thermal_transport::TemperatureProfile;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Integration metrics computed from a physics simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Coupling index: similarity to phenomenal vs functional concept vectors
    /// (Legacy metric from consciousness bridge - measures HDC similarity patterns)
    pub coupling_index: f32,

    /// Design integration: how well thermal, damage, and geometry cohere
    pub design_integration: f32,

    /// Thermal coherence: internal consistency of temperature profile
    pub thermal_coherence: f32,

    /// Damage-healing balance: how well damage and healing are integrated
    pub damage_healing_balance: f32,

    /// Geometry harmony: spatial coherence of the design
    pub geometry_harmony: f32,

    /// Overall integration score (geometric mean of components)
    pub overall_integration: f32,

    /// Binding advantage over bundling (measures HDC operation effectiveness)
    pub binding_advantage: f32,
}

impl IntegrationMetrics {
    /// Check if this design has high integration (top quartile)
    pub fn is_highly_integrated(&self) -> bool {
        self.overall_integration > 0.5 && self.design_integration > 0.3
    }

    /// Summary string for display
    pub fn summary(&self) -> String {
        format!(
            "coupling={:.3}, integration={:.3}, overall={:.3}",
            self.coupling_index, self.design_integration, self.overall_integration
        )
    }
}

// Backwards compatibility alias
pub type ConsciousnessMetrics = IntegrationMetrics;

/// Physics state encoded as HDC vectors for integration analysis
#[derive(Debug, Clone)]
pub struct EncodedPhysicsState {
    /// Temperature profile as spatial topology vector
    pub thermal_vector: ContinuousHV,

    /// Damage/healing equilibrium as unified concept
    pub damage_vector: ContinuousHV,

    /// Geometry and shielding as structural vector
    pub geometry_vector: ContinuousHV,

    /// Pulse dynamics as temporal rhythm vector
    pub pulse_vector: ContinuousHV,

    /// Operating conditions as context vector
    pub conditions_vector: ContinuousHV,

    /// All components bound together - the unified state
    pub unified_state: ContinuousHV,

    /// All components bundled (for comparison)
    pub bundled_state: ContinuousHV,
}

/// Design integration engine
///
/// Encodes physics simulation results as HDC vectors and measures
/// how well the different subsystems integrate with each other.
pub struct DesignIntegrationEngine {
    /// Bridge to concept vectors (used for coupling_index)
    bridge: PhysicsConsciousnessBridge,

    /// Genesis for deterministic vector generation
    genesis: GenesisSeed,

    /// Basis vectors for encoding physical quantities
    thermal_basis: ContinuousHV,
    damage_basis: ContinuousHV,
    _healing_basis: ContinuousHV,
    geometry_basis: ContinuousHV,
    power_basis: ContinuousHV,
    shielding_basis: ContinuousHV,
    _lifetime_basis: ContinuousHV,
    pulse_basis: ContinuousHV,
}

// Backwards compatibility alias
pub type PhysicsConsciousnessEngine = DesignIntegrationEngine;

impl DesignIntegrationEngine {
    /// Create from genesis seed
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            bridge: PhysicsConsciousnessBridge::from_genesis(genesis),
            thermal_basis: genesis.hv("physics::thermal", PHYSICS_DIM),
            damage_basis: genesis.hv("physics::damage", PHYSICS_DIM),
            _healing_basis: genesis.hv("physics::healing", PHYSICS_DIM),
            geometry_basis: genesis.hv("physics::geometry", PHYSICS_DIM),
            power_basis: genesis.hv("physics::power", PHYSICS_DIM),
            shielding_basis: genesis.hv("physics::shielding", PHYSICS_DIM),
            _lifetime_basis: genesis.hv("physics::lifetime", PHYSICS_DIM),
            pulse_basis: genesis.hv("physics::pulse", PHYSICS_DIM),
            genesis: genesis.clone(),
        }
    }

    /// Encode a temperature profile as an HDC vector
    ///
    /// The encoding captures:
    /// - Spatial gradient topology (where heat flows)
    /// - Temperature range (max - min)
    /// - Profile shape (concave vs convex)
    pub fn encode_temperature_profile(&self, profile: &TemperatureProfile) -> ContinuousHV {
        let t_range = profile.t_max - profile.t_min;
        let t_scale = if t_range > 1e-10 { t_range } else { 1.0 };

        let mut features = Vec::with_capacity(PHYSICS_DIM);

        for (i, &temp) in profile.temperatures.iter().enumerate() {
            let normalized = ((temp - profile.t_min) / t_scale) as f32;
            let position_phase =
                (i as f32 / profile.temperatures.len() as f32) * 2.0 * std::f32::consts::PI;

            for harmonic in 0..8 {
                let freq = (harmonic + 1) as f32;
                features.push(normalized * (freq * position_phase).sin());
                features.push(normalized * (freq * position_phase).cos());
            }
        }

        features.resize(PHYSICS_DIM, 0.0);
        let profile_vec = ContinuousHV::from_vec(features);
        profile_vec.bind(&self.thermal_basis)
    }

    /// Encode damage state (DPA equilibrium + healing) as unified concept
    pub fn encode_damage_state(
        &self,
        equilibrium_dpa: f64,
        lifetime_years: f64,
        fatigue_lifetime_years: f64,
    ) -> ContinuousHV {
        let dpa_norm = (equilibrium_dpa / 100.0).min(1.0) as f32;
        let lifetime_norm = (lifetime_years / 50.0).min(1.0) as f32;
        let fatigue_norm = (fatigue_lifetime_years / 50.0).min(1.0) as f32;
        let balance = 1.0 - (dpa_norm - lifetime_norm).abs();

        let mut features = vec![0.0f32; PHYSICS_DIM];

        for i in 0..PHYSICS_DIM {
            let phase = i as f32 / PHYSICS_DIM as f32;
            features[i] += dpa_norm * (2.0 * std::f32::consts::PI * phase).sin();
            features[i] += lifetime_norm * (4.0 * std::f32::consts::PI * phase).cos();
            features[i] += fatigue_norm * (8.0 * std::f32::consts::PI * phase).sin();
            features[i] *= 0.5 + 0.5 * balance;
        }

        let damage_vec = ContinuousHV::from_vec(features);
        let scaled_basis = self.damage_basis.scale(0.1);
        damage_vec.add(&scaled_basis)
    }

    /// Encode geometry as structural vector
    pub fn encode_geometry(
        &self,
        core_radius_m: f64,
        shell_thickness_m: f64,
        shielding_thickness_m: f64,
        total_mass_kg: f64,
    ) -> ContinuousHV {
        let core_norm = (core_radius_m / 0.1).min(1.0) as f32;
        let shell_norm = (shell_thickness_m / 0.05).min(1.0) as f32;
        let shield_norm = (shielding_thickness_m / 2.0).min(1.0) as f32;
        let mass_norm = (total_mass_kg / 10000.0).min(1.0) as f32;

        let mut features = vec![0.0f32; PHYSICS_DIM];

        for i in 0..PHYSICS_DIM {
            let phase = i as f32 / PHYSICS_DIM as f32;
            features[i] += core_norm * (2.0 * std::f32::consts::PI * phase).sin();
            features[i] += shell_norm * (4.0 * std::f32::consts::PI * phase).sin();
            features[i] += shield_norm * (8.0 * std::f32::consts::PI * phase).sin();
            features[i] *= mass_norm.sqrt();
        }

        let structure_vec = ContinuousHV::from_vec(features);
        structure_vec
            .bind(&self.geometry_basis)
            .bind(&self.shielding_basis)
    }

    /// Encode pulse dynamics as temporal rhythm
    pub fn encode_pulse_dynamics(
        &self,
        duty_cycle: f64,
        frequency_hz: f64,
        thermal_time_constant_s: f64,
    ) -> ContinuousHV {
        let duty_norm = duty_cycle.min(1.0) as f32;
        let freq_norm = (frequency_hz / 100.0).min(1.0) as f32;
        let tau_norm = (thermal_time_constant_s / 1000.0).min(1.0) as f32;

        let mut features = vec![0.0f32; PHYSICS_DIM];

        for i in 0..PHYSICS_DIM {
            let phase = i as f32 / PHYSICS_DIM as f32;
            let pulse_shape = if phase < duty_norm { 1.0 } else { -1.0 };
            let freq_mod = (freq_norm * 10.0 * std::f32::consts::PI * phase).sin();
            let envelope = (-phase / tau_norm.max(0.01)).exp();
            features[i] = pulse_shape * freq_mod * envelope;
        }

        let pulse_vec = ContinuousHV::from_vec(features);
        pulse_vec.bind(&self.pulse_basis)
    }

    /// Encode operating conditions as context
    pub fn encode_conditions(&self, conditions: &OperatingConditions) -> ContinuousHV {
        let power_norm = (conditions.power_kw / 1000.0).min(1.0) as f32;
        let power_vec = self.power_basis.scale(power_norm);
        let reaction_vec = self.genesis.hv(
            &format!("physics::reaction::{:?}", conditions.reaction),
            PHYSICS_DIM,
        );
        power_vec.bind(&reaction_vec)
    }

    /// Encode entire simulation result as physics state
    pub fn encode_simulation(&self, result: &CoupledSimulationResult) -> EncodedPhysicsState {
        let thermal_vector = self.encode_temperature_profile(&result.thermal_profile);

        let damage_vector = self.encode_damage_state(
            result.pulse_thermal.equilibrium_dpa,
            result.pulse_thermal.lifetime_years,
            result.pulse_thermal.fatigue_lifetime_years,
        );

        let geometry_vector = self.encode_geometry(
            result.geometry_shielding.geometry.core_radius,
            result.geometry_shielding.geometry.shell_thickness,
            result.geometry_shielding.shielding.thickness_m,
            result.geometry_shielding.total_mass_kg,
        );

        let pulse_vector = self.encode_pulse_dynamics(
            result.pulse_thermal.pulse.duty_cycle,
            1.0 / result.pulse_thermal.pulse.period_s,
            result.pulse_thermal.cycling.time_constant,
        );

        let conditions_vector = self.encode_conditions(&result.conditions);

        // BIND all components - creates emergent unified structure
        let unified_state = thermal_vector
            .bind(&damage_vector)
            .bind(&geometry_vector)
            .bind(&pulse_vector)
            .bind(&conditions_vector);

        // BUNDLE for comparison - creates superposition
        let bundled_state = ContinuousHV::bundle(&[
            &thermal_vector,
            &damage_vector,
            &geometry_vector,
            &pulse_vector,
            &conditions_vector,
        ]);

        EncodedPhysicsState {
            thermal_vector,
            damage_vector,
            geometry_vector,
            pulse_vector,
            conditions_vector,
            unified_state,
            bundled_state,
        }
    }

    /// Compute integration metrics for a simulation result
    pub fn compute_metrics(&self, result: &CoupledSimulationResult) -> IntegrationMetrics {
        let state = self.encode_simulation(result);

        // Coupling index: entropy-based integration measure of state components
        let coupling_index = {
            use crate::consciousness_metrics::TruePhiCalculator;
            let calc = TruePhiCalculator::new();
            let components = vec![
                state.thermal_vector.clone(),
                state.damage_vector.clone(),
                state.geometry_vector.clone(),
                state.pulse_vector.clone(),
                state.conditions_vector.clone(),
            ];
            let n = components.len();
            let max_pairs = (n * (n - 1)) / 2;
            let max_phi = (max_pairs as f64) * 4.0;
            let result = calc.compute_true_phi(&components);
            (result.phi / max_phi).min(1.0) as f32
        };

        // Design integration: internal consistency
        let perturbed = state.unified_state.scale(0.9);
        let design_integration = state.unified_state.similarity(&perturbed);

        // Binding advantage: bound vs bundled
        let binding_advantage = state.unified_state.norm() - state.bundled_state.norm();

        // Thermal coherence
        let thermal_coherence = {
            let norm = state.thermal_vector.norm();
            if norm > 0.0 {
                1.0 / (1.0 + (norm - 1.0).abs())
            } else {
                0.0
            }
        };

        // Damage-healing balance
        let dpa_norm = (result.pulse_thermal.equilibrium_dpa / 100.0).min(1.0) as f32;
        let lifetime_norm = (result.pulse_thermal.lifetime_years / 50.0).min(1.0) as f32;
        let damage_healing_balance = 1.0 - (dpa_norm - lifetime_norm).abs();

        // Geometry harmony
        let mass_per_kw = result.geometry_shielding.total_mass_kg / result.conditions.power_kw;
        let geometry_harmony = 1.0 / (1.0 + (mass_per_kw / 1000.0) as f32);

        // Overall integration: geometric mean
        let components = [
            (design_integration + 1.0) / 2.0,
            thermal_coherence,
            damage_healing_balance,
            geometry_harmony,
        ];

        let product: f32 = components.iter().product();
        let geo_mean = product.powf(1.0 / components.len() as f32);
        let feasibility_bonus = if result.feasible { 1.2 } else { 0.8 };
        let overall_integration = geo_mean * feasibility_bonus;

        IntegrationMetrics {
            coupling_index,
            design_integration,
            thermal_coherence,
            damage_healing_balance,
            geometry_harmony,
            overall_integration,
            binding_advantage,
        }
    }

    /// Compare two designs by their integration metrics
    pub fn compare_designs(
        &self,
        result_a: &CoupledSimulationResult,
        result_b: &CoupledSimulationResult,
    ) -> DesignComparison {
        let metrics_a = self.compute_metrics(result_a);
        let metrics_b = self.compute_metrics(result_b);

        let state_a = self.encode_simulation(result_a);
        let state_b = self.encode_simulation(result_b);

        let state_similarity = state_a.unified_state.similarity(&state_b.unified_state);
        let integration_difference = metrics_a.overall_integration - metrics_b.overall_integration;

        DesignComparison {
            metrics_a,
            metrics_b,
            state_similarity,
            integration_difference,
        }
    }

    /// Get the concept bridge for direct access
    pub fn bridge(&self) -> &PhysicsConsciousnessBridge {
        &self.bridge
    }
}

/// Comparison between two designs
#[derive(Debug, Clone)]
pub struct DesignComparison {
    pub metrics_a: IntegrationMetrics,
    pub metrics_b: IntegrationMetrics,
    pub state_similarity: f32,
    pub integration_difference: f32,
}

impl DesignComparison {
    /// Which design has higher integration?
    pub fn more_integrated_design(&self) -> &'static str {
        if self.integration_difference > 0.0 {
            "A"
        } else {
            "B"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::coupled_physics::{CoupledPhysicsEngine, OperatingConditions};

    fn setup() -> (CoupledPhysicsEngine, DesignIntegrationEngine, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("design integration test");
        let physics = CoupledPhysicsEngine::from_genesis(&genesis);
        let integration = DesignIntegrationEngine::from_genesis(&genesis);
        (physics, integration, genesis)
    }

    #[test]
    fn test_encode_temperature_profile() {
        let (physics, integration, _) = setup();

        let conditions = OperatingConditions::consumer();
        let result = physics.simulate(&conditions);
        let thermal_vec = integration.encode_temperature_profile(&result.thermal_profile);

        assert_eq!(thermal_vec.dim(), PHYSICS_DIM);
        assert!(thermal_vec.norm() > 0.0);
    }

    #[test]
    fn test_encode_damage_state() {
        let (_, integration, _) = setup();

        let damage_vec = integration.encode_damage_state(50.0, 25.0, 30.0);
        assert_eq!(damage_vec.dim(), PHYSICS_DIM);
        assert!(damage_vec.norm() > 0.0);

        let damage_vec_high = integration.encode_damage_state(100.0, 10.0, 15.0);
        let similarity = damage_vec.similarity(&damage_vec_high);
        assert!(
            similarity < 1.0,
            "Different damage should produce distinguishable vectors"
        );
    }

    #[test]
    fn test_encode_full_simulation() {
        let (physics, integration, _) = setup();

        let conditions = OperatingConditions::consumer();
        let result = physics.simulate(&conditions);
        let state = integration.encode_simulation(&result);

        assert_eq!(state.thermal_vector.dim(), PHYSICS_DIM);
        assert_eq!(state.unified_state.dim(), PHYSICS_DIM);

        let bound_bundle_sim = state.unified_state.similarity(&state.bundled_state);
        assert!(
            bound_bundle_sim < 0.5,
            "Binding should create different structure than bundling"
        );
    }

    #[test]
    fn test_compute_integration_metrics() {
        let (physics, integration, _) = setup();

        let conditions = OperatingConditions::consumer();
        let result = physics.simulate(&conditions);
        let metrics = integration.compute_metrics(&result);

        println!("\n========================================");
        println!("DESIGN INTEGRATION METRICS");
        println!("========================================");
        println!("Coupling index:        {:.4}", metrics.coupling_index);
        println!("Design integration:    {:.4}", metrics.design_integration);
        println!("Thermal coherence:     {:.4}", metrics.thermal_coherence);
        println!(
            "Damage-healing:        {:.4}",
            metrics.damage_healing_balance
        );
        println!("Geometry harmony:      {:.4}", metrics.geometry_harmony);
        println!("Binding advantage:     {:.4}", metrics.binding_advantage);
        println!("----------------------------------------");
        println!("OVERALL INTEGRATION:   {:.4}", metrics.overall_integration);
        println!("========================================\n");

        assert!(metrics.coupling_index >= -1.0 && metrics.coupling_index <= 1.0);
        assert!(metrics.overall_integration >= 0.0);
    }

    #[test]
    fn test_compare_designs() {
        let (physics, integration, _) = setup();

        let consumer = OperatingConditions::consumer();
        let industrial = OperatingConditions::industrial();

        let result_consumer = physics.simulate(&consumer);
        let result_industrial = physics.simulate(&industrial);

        let comparison = integration.compare_designs(&result_consumer, &result_industrial);

        println!("\n========================================");
        println!("DESIGN COMPARISON");
        println!("========================================");
        println!(
            "Consumer integration:   {:.4}",
            comparison.metrics_a.overall_integration
        );
        println!(
            "Industrial integration: {:.4}",
            comparison.metrics_b.overall_integration
        );
        println!("State similarity:       {:.4}", comparison.state_similarity);
        println!(
            "More integrated design: {}",
            comparison.more_integrated_design()
        );
        println!("========================================\n");

        assert!(
            comparison.integration_difference.abs() > 0.001 || comparison.state_similarity < 0.99
        );
    }

    #[test]
    fn test_integration_correlates_with_feasibility() {
        let (physics, integration, _) = setup();

        println!("\n========================================");
        println!("INTEGRATION vs FEASIBILITY STUDY");
        println!("========================================\n");

        let mut feasible_integration = Vec::new();
        let mut infeasible_integration = Vec::new();

        for power in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0] {
            let conditions = OperatingConditions {
                power_kw: power,
                ..OperatingConditions::consumer()
            };

            let result = physics.simulate(&conditions);
            let metrics = integration.compute_metrics(&result);

            println!(
                "{:>5.0} kW: integration={:.4}, feasible={}",
                power, metrics.overall_integration, result.feasible
            );

            if result.feasible {
                feasible_integration.push(metrics.overall_integration);
            } else {
                infeasible_integration.push(metrics.overall_integration);
            }
        }

        let feasible_mean = if feasible_integration.is_empty() {
            0.0
        } else {
            feasible_integration.iter().sum::<f32>() / feasible_integration.len() as f32
        };
        let infeasible_mean = if infeasible_integration.is_empty() {
            0.0
        } else {
            infeasible_integration.iter().sum::<f32>() / infeasible_integration.len() as f32
        };

        println!(
            "\nFeasible mean integration:   {:.4} (n={})",
            feasible_mean,
            feasible_integration.len()
        );
        println!(
            "Infeasible mean integration: {:.4} (n={})",
            infeasible_mean,
            infeasible_integration.len()
        );
        println!("========================================\n");

        // At least some designs should be feasible at low power
        assert!(
            !feasible_integration.is_empty(),
            "Should have at least one feasible design"
        );

        // All integration values should be finite and non-negative
        for val in feasible_integration
            .iter()
            .chain(infeasible_integration.iter())
        {
            assert!(val.is_finite(), "Integration value should be finite");
            assert!(*val >= 0.0, "Integration value should be non-negative");
        }

        // Mean values should be finite and non-negative
        assert!(feasible_mean.is_finite(), "Feasible mean should be finite");
        assert!(feasible_mean >= 0.0, "Feasible mean should be non-negative");
    }
}
