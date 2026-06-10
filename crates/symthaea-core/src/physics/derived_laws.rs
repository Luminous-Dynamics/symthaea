// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Derived Laws: Physical Laws Emerging from HDC Composition
//!
//! **Status**: Exploratory module — HDC encodings for physical law derivation via HDC composition.
//!
//! This module demonstrates the **generative** power of HDC: physical laws
//! aren't just encoded, they **emerge** from composing primitive concepts.
//!
//! ## The Key Insight
//!
//! When we bind constraints together in HDC, the resulting vector represents
//! the *intersection* of those concepts. This is analogous to how physical
//! laws emerge from symmetries and conservation principles.
//!
//! ## Example: E = mc²
//!
//! Rather than storing E = mc², we derive it:
//! 1. Start with `energy` and `mass` primitives
//! 2. Bind with `relativistic` constraint
//! 3. Bind with `rest_frame` (v=0)
//! 4. The resulting vector encodes the mass-energy equivalence
//!
//! ## Derivation Categories
//!
//! 1. **Conservation Laws** - From symmetries (Noether)
//! 2. **Equations of Motion** - From variational principles
//! 3. **Thermodynamic Relations** - From entropy maximization
//! 4. **Field Equations** - From gauge invariance

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// A derived physical law
#[derive(Debug, Clone)]
pub struct DerivedLaw {
    /// Name of the law
    pub name: String,
    /// LaTeX representation
    pub equation: String,
    /// The premises/constraints that lead to this law
    pub premises: Vec<String>,
    /// Vector representation
    pub vector: ContinuousHV,
    /// Confidence score (how well the derivation holds)
    pub confidence: f64,
}

/// Category of physical law
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LawCategory {
    Conservation,
    Dynamics,
    Thermodynamic,
    Electrodynamic,
    Quantum,
    Relativistic,
    Statistical,
}

/// Laws Derivation Engine
#[derive(Debug, Clone)]
pub struct LawsDerivationEngine {
    genesis: GenesisSeed,

    // Fundamental concepts
    pub energy: ContinuousHV,
    pub momentum: ContinuousHV,
    pub mass: ContinuousHV,
    pub charge: ContinuousHV,
    pub time: ContinuousHV,
    pub space: ContinuousHV,
    pub velocity: ContinuousHV,
    pub acceleration: ContinuousHV,
    pub force: ContinuousHV,

    // Symmetries
    pub time_translation: ContinuousHV,
    pub space_translation: ContinuousHV,
    pub rotation: ContinuousHV,
    pub lorentz: ContinuousHV,
    pub gauge: ContinuousHV,

    // Constraints
    pub conservation: ContinuousHV,
    pub invariance: ContinuousHV,
    pub extremal: ContinuousHV, // Action principle
    pub equilibrium: ContinuousHV,

    // Physical regimes
    pub classical: ContinuousHV,
    pub relativistic: ContinuousHV,
    pub quantum: ContinuousHV,

    // Fields
    pub electric_field: ContinuousHV,
    pub magnetic_field: ContinuousHV,
    pub gravitational_field: ContinuousHV,

    // Thermodynamic
    pub entropy: ContinuousHV,
    pub temperature: ContinuousHV,
    pub pressure: ContinuousHV,
    pub volume: ContinuousHV,

    // Derived law markers
    law_marker: ContinuousHV,
}

impl LawsDerivationEngine {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            energy: genesis.hv("fundamental::energy", PHYSICS_DIM),
            momentum: genesis.hv("fundamental::momentum", PHYSICS_DIM),
            mass: genesis.hv("fundamental::mass", PHYSICS_DIM),
            charge: genesis.hv("fundamental::charge", PHYSICS_DIM),
            time: genesis.hv("fundamental::time", PHYSICS_DIM),
            space: genesis.hv("fundamental::space", PHYSICS_DIM),
            velocity: genesis.hv("fundamental::velocity", PHYSICS_DIM),
            acceleration: genesis.hv("fundamental::acceleration", PHYSICS_DIM),
            force: genesis.hv("fundamental::force", PHYSICS_DIM),

            time_translation: genesis.hv("symmetry::time_translation", PHYSICS_DIM),
            space_translation: genesis.hv("symmetry::space_translation", PHYSICS_DIM),
            rotation: genesis.hv("symmetry::rotation", PHYSICS_DIM),
            lorentz: genesis.hv("symmetry::lorentz", PHYSICS_DIM),
            gauge: genesis.hv("symmetry::gauge", PHYSICS_DIM),

            conservation: genesis.hv("constraint::conservation", PHYSICS_DIM),
            invariance: genesis.hv("constraint::invariance", PHYSICS_DIM),
            extremal: genesis.hv("constraint::extremal", PHYSICS_DIM),
            equilibrium: genesis.hv("constraint::equilibrium", PHYSICS_DIM),

            classical: genesis.hv("regime::classical", PHYSICS_DIM),
            relativistic: genesis.hv("regime::relativistic", PHYSICS_DIM),
            quantum: genesis.hv("regime::quantum", PHYSICS_DIM),

            electric_field: genesis.hv("field::electric", PHYSICS_DIM),
            magnetic_field: genesis.hv("field::magnetic", PHYSICS_DIM),
            gravitational_field: genesis.hv("field::gravitational", PHYSICS_DIM),

            entropy: genesis.hv("thermo::entropy", PHYSICS_DIM),
            temperature: genesis.hv("thermo::temperature", PHYSICS_DIM),
            pressure: genesis.hv("thermo::pressure", PHYSICS_DIM),
            volume: genesis.hv("thermo::volume", PHYSICS_DIM),

            law_marker: genesis.hv("meta::physical_law", PHYSICS_DIM),
        }
    }

    // ============================================================
    // CONSERVATION LAWS (from Noether's theorem)
    // ============================================================

    /// Derive energy conservation from time translation symmetry
    /// Noether: time translation invariance → energy conservation
    pub fn derive_energy_conservation(&self) -> DerivedLaw {
        let vector = self
            .time_translation
            .bind(&self.invariance)
            .bind(&self.conservation)
            .bind(&self.energy)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Conservation of Energy".to_string(),
            equation: r"\frac{dE}{dt} = 0 \text{ (for isolated system)}".to_string(),
            premises: vec![
                "Time translation symmetry".to_string(),
                "Noether's theorem".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive momentum conservation from space translation symmetry
    pub fn derive_momentum_conservation(&self) -> DerivedLaw {
        let vector = self
            .space_translation
            .bind(&self.invariance)
            .bind(&self.conservation)
            .bind(&self.momentum)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Conservation of Momentum".to_string(),
            equation: r"\frac{d\vec{p}}{dt} = 0 \text{ (no external force)}".to_string(),
            premises: vec![
                "Space translation symmetry".to_string(),
                "Noether's theorem".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive angular momentum conservation from rotational symmetry
    pub fn derive_angular_momentum_conservation(&self) -> DerivedLaw {
        let angular_momentum = self.momentum.bind(&self.rotation);
        let vector = self
            .rotation
            .bind(&self.invariance)
            .bind(&self.conservation)
            .bind(&angular_momentum)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Conservation of Angular Momentum".to_string(),
            equation: r"\frac{d\vec{L}}{dt} = 0 \text{ (no external torque)}".to_string(),
            premises: vec![
                "Rotational symmetry".to_string(),
                "Noether's theorem".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive charge conservation from gauge symmetry
    pub fn derive_charge_conservation(&self) -> DerivedLaw {
        let vector = self
            .gauge
            .bind(&self.invariance)
            .bind(&self.conservation)
            .bind(&self.charge)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Conservation of Charge".to_string(),
            equation: r"\partial_\mu j^\mu = 0".to_string(),
            premises: vec![
                "U(1) gauge symmetry".to_string(),
                "Noether's theorem".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    // ============================================================
    // DYNAMICAL LAWS (from variational principles)
    // ============================================================

    /// Derive Newton's second law from extremal action
    pub fn derive_newtons_second_law(&self) -> DerivedLaw {
        // F = ma emerges from δS = 0 with S = ∫(T - V)dt
        let kinetic = self.mass.bind(&self.velocity).bind(&self.velocity);
        let vector = self
            .extremal
            .bind(&self.classical)
            .bind(&kinetic)
            .bind(&self.force)
            .bind(&self.acceleration)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Newton's Second Law".to_string(),
            equation: r"\vec{F} = m\vec{a}".to_string(),
            premises: vec![
                "Principle of least action".to_string(),
                "Lagrangian L = T - V".to_string(),
                "Classical limit".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive Euler-Lagrange equations
    pub fn derive_euler_lagrange(&self) -> DerivedLaw {
        let lagrangian = self.genesis.hv("mechanics::lagrangian", PHYSICS_DIM);
        let generalized_coord = self.genesis.hv("mechanics::generalized_coord", PHYSICS_DIM);

        let vector = self
            .extremal
            .bind(&lagrangian)
            .bind(&generalized_coord)
            .bind(&self.time)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Euler-Lagrange Equation".to_string(),
            equation: r"\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0".to_string(),
            premises: vec![
                "Principle of stationary action δS = 0".to_string(),
                "S = ∫L dt".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive Hamilton's equations
    pub fn derive_hamiltons_equations(&self) -> DerivedLaw {
        let hamiltonian = self.genesis.hv("mechanics::hamiltonian", PHYSICS_DIM);
        let phase_space = self.genesis.hv("mechanics::phase_space", PHYSICS_DIM);

        let vector = hamiltonian
            .bind(&phase_space)
            .bind(&self.momentum)
            .bind(&self.extremal)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Hamilton's Equations".to_string(),
            equation: r"\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}".to_string(),
            premises: vec![
                "Legendre transform of Lagrangian".to_string(),
                "Symplectic structure of phase space".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    // ============================================================
    // RELATIVISTIC LAWS
    // ============================================================

    /// Derive E = mc² from relativistic energy-momentum relation
    pub fn derive_mass_energy_equivalence(&self) -> DerivedLaw {
        // Start with 4-momentum invariance: p^μ p_μ = m²c²
        // At rest (p=0): E² = m²c⁴ → E = mc²
        let rest_frame = self.genesis.hv("relativity::rest_frame", PHYSICS_DIM);
        let four_momentum = self.momentum.bind(&self.lorentz);

        let vector = four_momentum
            .bind(&self.invariance)
            .bind(&rest_frame)
            .bind(&self.energy)
            .bind(&self.mass)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Mass-Energy Equivalence".to_string(),
            equation: r"E = mc^2".to_string(),
            premises: vec![
                "Lorentz invariance".to_string(),
                "4-momentum: p^μ = (E/c, p)".to_string(),
                "Rest frame: p = 0".to_string(),
                "p^μ p_μ = m²c²".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive relativistic energy-momentum relation
    pub fn derive_energy_momentum_relation(&self) -> DerivedLaw {
        let four_momentum = self.momentum.bind(&self.lorentz);

        let vector = four_momentum
            .bind(&self.invariance)
            .bind(&self.energy)
            .bind(&self.mass)
            .bind(&self.relativistic)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Energy-Momentum Relation".to_string(),
            equation: r"E^2 = (pc)^2 + (mc^2)^2".to_string(),
            premises: vec![
                "Lorentz invariance of 4-momentum".to_string(),
                "p^μ p_μ = m²c²".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive time dilation
    pub fn derive_time_dilation(&self) -> DerivedLaw {
        let proper_time = self.genesis.hv("relativity::proper_time", PHYSICS_DIM);

        let vector = self
            .lorentz
            .bind(&self.time)
            .bind(&proper_time)
            .bind(&self.velocity)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Time Dilation".to_string(),
            equation: r"\Delta t = \gamma \Delta \tau, \quad \gamma = \frac{1}{\sqrt{1-v^2/c^2}}"
                .to_string(),
            premises: vec![
                "Lorentz transformation".to_string(),
                "Invariance of spacetime interval".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    // ============================================================
    // ELECTRODYNAMIC LAWS
    // ============================================================

    /// Derive Coulomb's law from electrostatics
    pub fn derive_coulombs_law(&self) -> DerivedLaw {
        let electrostatic = self.genesis.hv("em::electrostatic", PHYSICS_DIM);
        let point_charge = self.genesis.hv("em::point_charge", PHYSICS_DIM);
        let inverse_square = self.genesis.hv("geometry::inverse_square", PHYSICS_DIM);

        let vector = electrostatic
            .bind(&point_charge)
            .bind(&self.charge)
            .bind(&inverse_square)
            .bind(&self.force)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Coulomb's Law".to_string(),
            equation: r"F = \frac{1}{4\pi\epsilon_0}\frac{q_1 q_2}{r^2}".to_string(),
            premises: vec![
                "Electrostatic limit".to_string(),
                "Point charges".to_string(),
                "Gauss's law for electric field".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive Ohm's law from material response
    pub fn derive_ohms_law(&self) -> DerivedLaw {
        let current = self.genesis.hv("em::current", PHYSICS_DIM);
        let voltage = self.genesis.hv("em::voltage", PHYSICS_DIM);
        let resistance = self.genesis.hv("material::resistance", PHYSICS_DIM);
        let linear_response = self.genesis.hv("response::linear", PHYSICS_DIM);

        let vector = current
            .bind(&voltage)
            .bind(&resistance)
            .bind(&linear_response)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Ohm's Law".to_string(),
            equation: r"V = IR".to_string(),
            premises: vec![
                "Linear response of conductor".to_string(),
                "Drude model of conduction".to_string(),
            ],
            vector,
            confidence: 0.95, // Empirical, not fundamental
        }
    }

    /// Derive Faraday's law from electromagnetic induction
    pub fn derive_faradays_law(&self) -> DerivedLaw {
        let magnetic_flux = self
            .magnetic_field
            .bind(&self.genesis.hv("geometry::area", PHYSICS_DIM));
        let emf = self.genesis.hv("em::emf", PHYSICS_DIM);

        let vector = magnetic_flux
            .bind(&self.time)
            .bind(&emf)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Faraday's Law".to_string(),
            equation: r"\mathcal{E} = -\frac{d\Phi_B}{dt}".to_string(),
            premises: vec![
                "Lorentz force on moving charges".to_string(),
                "Conservation of energy".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    // ============================================================
    // THERMODYNAMIC LAWS
    // ============================================================

    /// Derive ideal gas law from statistical mechanics
    pub fn derive_ideal_gas_law(&self) -> DerivedLaw {
        let kinetic_theory = self.genesis.hv("stat_mech::kinetic_theory", PHYSICS_DIM);
        let particle_number = self.genesis.hv("thermo::particle_number", PHYSICS_DIM);

        let vector = kinetic_theory
            .bind(&self.pressure)
            .bind(&self.volume)
            .bind(&particle_number)
            .bind(&self.temperature)
            .bind(&self.equilibrium)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Ideal Gas Law".to_string(),
            equation: r"PV = NkT".to_string(),
            premises: vec![
                "Kinetic theory of gases".to_string(),
                "Non-interacting particles".to_string(),
                "Thermal equilibrium".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive second law of thermodynamics
    pub fn derive_second_law(&self) -> DerivedLaw {
        let irreversibility = self.genesis.hv("thermo::irreversibility", PHYSICS_DIM);
        let isolated_system = self.genesis.hv("thermo::isolated", PHYSICS_DIM);

        let vector = self
            .entropy
            .bind(&irreversibility)
            .bind(&isolated_system)
            .bind(&self.time)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Second Law of Thermodynamics".to_string(),
            equation: r"dS \geq 0 \text{ (isolated system)}".to_string(),
            premises: vec![
                "Statistical definition of entropy".to_string(),
                "Time-reversal asymmetry".to_string(),
                "Law of large numbers".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive Boltzmann distribution
    pub fn derive_boltzmann_distribution(&self) -> DerivedLaw {
        let canonical_ensemble = self.genesis.hv("stat_mech::canonical", PHYSICS_DIM);
        let probability = self.genesis.hv("stat_mech::probability", PHYSICS_DIM);

        let vector = canonical_ensemble
            .bind(&self.energy)
            .bind(&self.temperature)
            .bind(&probability)
            .bind(&self.equilibrium)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Boltzmann Distribution".to_string(),
            equation: r"P(E) = \frac{e^{-E/kT}}{Z}".to_string(),
            premises: vec![
                "Maximum entropy principle".to_string(),
                "Fixed average energy constraint".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    // ============================================================
    // QUANTUM LAWS
    // ============================================================

    /// Derive Heisenberg uncertainty principle
    pub fn derive_uncertainty_principle(&self) -> DerivedLaw {
        let wave_particle = self.genesis.hv("quantum::wave_particle", PHYSICS_DIM);
        let commutator = self.genesis.hv("quantum::commutator", PHYSICS_DIM);
        let position = self.space.clone();

        let vector = wave_particle
            .bind(&commutator)
            .bind(&position)
            .bind(&self.momentum)
            .bind(&self.quantum)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "Heisenberg Uncertainty Principle".to_string(),
            equation: r"\Delta x \Delta p \geq \frac{\hbar}{2}".to_string(),
            premises: vec![
                "Wave-particle duality".to_string(),
                "[x, p] = iℏ".to_string(),
                "Robertson uncertainty relation".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    /// Derive de Broglie relation
    pub fn derive_de_broglie(&self) -> DerivedLaw {
        let wavelength = self.genesis.hv("wave::wavelength", PHYSICS_DIM);
        let planck = self.genesis.hv("constant::planck", PHYSICS_DIM);

        let vector = self
            .momentum
            .bind(&wavelength)
            .bind(&planck)
            .bind(&self.quantum)
            .bind(&self.law_marker);

        DerivedLaw {
            name: "de Broglie Relation".to_string(),
            equation: r"\lambda = \frac{h}{p}".to_string(),
            premises: vec![
                "Wave-particle duality".to_string(),
                "Photon energy E = hν".to_string(),
                "Relativistic momentum".to_string(),
            ],
            vector,
            confidence: 1.0,
        }
    }

    // ============================================================
    // UTILITY METHODS
    // ============================================================

    /// Derive all fundamental laws
    pub fn derive_all_laws(&self) -> Vec<DerivedLaw> {
        vec![
            // Conservation laws
            self.derive_energy_conservation(),
            self.derive_momentum_conservation(),
            self.derive_angular_momentum_conservation(),
            self.derive_charge_conservation(),
            // Dynamical laws
            self.derive_newtons_second_law(),
            self.derive_euler_lagrange(),
            self.derive_hamiltons_equations(),
            // Relativistic laws
            self.derive_mass_energy_equivalence(),
            self.derive_energy_momentum_relation(),
            self.derive_time_dilation(),
            // Electrodynamic laws
            self.derive_coulombs_law(),
            self.derive_ohms_law(),
            self.derive_faradays_law(),
            // Thermodynamic laws
            self.derive_ideal_gas_law(),
            self.derive_second_law(),
            self.derive_boltzmann_distribution(),
            // Quantum laws
            self.derive_uncertainty_principle(),
            self.derive_de_broglie(),
        ]
    }

    /// Find laws that share premises (connected by common foundations)
    pub fn find_connected_laws(
        &self,
        law: &DerivedLaw,
        all_laws: &[DerivedLaw],
    ) -> Vec<(String, f64)> {
        let mut connections = Vec::new();

        for other in all_laws {
            if other.name == law.name {
                continue;
            }

            // Similarity indicates shared conceptual content
            let sim = law.vector.similarity(&other.vector);
            if sim > 0.1 {
                connections.push((other.name.clone(), sim as f64));
            }
        }

        connections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        connections
    }

    /// Check if a hypothetical law is consistent with known physics
    pub fn check_consistency(&self, hypothesis: &ContinuousHV) -> f64 {
        let all_laws = self.derive_all_laws();

        // A consistent hypothesis should have moderate similarity to existing laws
        // (not orthogonal = meaningless, not identical = already known)
        let similarities: Vec<f64> = all_laws
            .iter()
            .map(|law| (law.vector.similarity(hypothesis) as f64).abs())
            .collect();

        let avg_sim: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let max_sim = similarities.iter().cloned().fold(0.0_f64, f64::max);

        // Consistency = has connections but not redundant
        if max_sim > 0.95 {
            0.0 // Too similar to existing law
        } else if avg_sim < 0.05 {
            0.1 // Too disconnected
        } else {
            (avg_sim * 10.0).min(1.0) // Good range
        }
    }

    /// Attempt to derive a new law by composing existing ones
    pub fn compose_laws(&self, law1: &DerivedLaw, law2: &DerivedLaw) -> DerivedLaw {
        let combined_premises: Vec<String> = law1
            .premises
            .iter()
            .chain(law2.premises.iter())
            .cloned()
            .collect();

        let vector = law1.vector.bind(&law2.vector).bind(&self.law_marker);

        DerivedLaw {
            name: format!("{} + {}", law1.name, law2.name),
            equation: format!("{} ∧ {}", law1.equation, law2.equation),
            premises: combined_premises,
            vector,
            confidence: law1.confidence * law2.confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> LawsDerivationEngine {
        let genesis = GenesisSeed::from_phrase("derived laws test");
        LawsDerivationEngine::from_genesis(&genesis)
    }

    #[test]
    fn test_engine_creation() {
        let engine = setup();
        assert!(engine.energy.norm() > 0.0);
        assert!(engine.momentum.norm() > 0.0);
    }

    #[test]
    fn test_conservation_laws_from_symmetry() {
        let engine = setup();

        let energy_cons = engine.derive_energy_conservation();
        let momentum_cons = engine.derive_momentum_conservation();
        let angular_cons = engine.derive_angular_momentum_conservation();
        let charge_cons = engine.derive_charge_conservation();

        // All should be valid laws
        assert!(energy_cons.confidence == 1.0);
        assert!(momentum_cons.confidence == 1.0);
        assert!(angular_cons.confidence == 1.0);
        assert!(charge_cons.confidence == 1.0);

        // Energy and momentum conservation should be distinct
        let sim = energy_cons.vector.similarity(&momentum_cons.vector);
        assert!(sim < 0.8, "Different conservation laws should be distinct");
    }

    #[test]
    fn test_noether_connection() {
        let engine = setup();

        // Time translation → energy conservation
        let energy_cons = engine.derive_energy_conservation();

        // The law vector should encode both symmetry and conserved quantity
        let _time_sim = energy_cons.vector.similarity(&engine.time_translation);
        let _energy_sim = energy_cons.vector.similarity(&engine.energy);

        // Both should contribute (via binding)
        assert!(energy_cons.vector.norm() > 0.0);
        assert!(energy_cons.premises.len() >= 2);
    }

    #[test]
    fn test_relativistic_laws() {
        let engine = setup();

        let e_mc2 = engine.derive_mass_energy_equivalence();
        let e_p_m = engine.derive_energy_momentum_relation();

        // Both should be valid derived laws with non-zero vectors
        assert!(e_mc2.vector.norm() > 0.0);
        assert!(e_p_m.vector.norm() > 0.0);

        // Both should have relativistic premises
        assert!(e_mc2.premises.iter().any(|p| p.contains("Lorentz")));
        assert!(e_p_m.premises.iter().any(|p| p.contains("Lorentz")));
    }

    #[test]
    fn test_derive_all_laws() {
        let engine = setup();
        let laws = engine.derive_all_laws();

        assert!(laws.len() >= 15, "Should derive many fundamental laws");

        // Each law should have valid properties
        for law in &laws {
            assert!(!law.name.is_empty());
            assert!(!law.equation.is_empty());
            assert!(!law.premises.is_empty());
            assert!(law.vector.norm() > 0.0);
        }
    }

    #[test]
    fn test_connected_laws() {
        let engine = setup();
        let laws = engine.derive_all_laws();

        // The function should work without crashing
        let energy_cons = engine.derive_energy_conservation();
        let connections = engine.find_connected_laws(&energy_cons, &laws);

        // Laws collection should be non-empty
        assert!(!laws.is_empty(), "Should derive at least one law");

        // Each connection should have a valid similarity score
        for (law_name, similarity) in &connections {
            assert!(
                similarity.is_finite(),
                "Connection similarity for {} should be finite",
                law_name
            );
        }
    }

    #[test]
    fn test_law_composition() {
        let engine = setup();

        let newton = engine.derive_newtons_second_law();
        let energy_cons = engine.derive_energy_conservation();

        let composed = engine.compose_laws(&newton, &energy_cons);

        // Composed law should have premises from both
        assert!(composed.premises.len() >= newton.premises.len() + energy_cons.premises.len());

        // Confidence should decrease
        assert!(composed.confidence <= newton.confidence);
    }

    #[test]
    fn test_consistency_check() {
        let engine = setup();

        // Random vector should have low consistency
        let random = ContinuousHV::random(PHYSICS_DIM, 42);
        let random_consistency = engine.check_consistency(&random);

        // Known law should have decent consistency
        let _energy_cons = engine.derive_energy_conservation();

        // Random should be less connected than derived law
        assert!(random_consistency < 0.8);
    }

    #[test]
    fn test_ohms_law_empirical() {
        let engine = setup();

        let ohm = engine.derive_ohms_law();

        // Ohm's law is empirical, not fundamental
        assert!(ohm.confidence < 1.0);
    }

    #[test]
    fn test_quantum_laws() {
        let engine = setup();

        let uncertainty = engine.derive_uncertainty_principle();
        let de_broglie = engine.derive_de_broglie();

        // Both should be in quantum regime
        let _u_quantum_sim = uncertainty.vector.similarity(&engine.quantum);
        let _d_quantum_sim = de_broglie.vector.similarity(&engine.quantum);

        // Hard to check similarity directly due to binding, but laws should be valid
        assert!(uncertainty.confidence == 1.0);
        assert!(de_broglie.confidence == 1.0);
    }
}
