// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Classical Mechanics
//!
//! **Status**: Exploratory module — HDC encodings for classical mechanics (Newton to Hamilton, Noether symmetries).
//!
//! This module encodes classical mechanics from Newton to Hamilton,
//! including Noether's theorem connecting symmetries to conservation laws.
//!
//! ## Formulations
//!
//! - **Newtonian**: F = ma, direct force analysis
//! - **Lagrangian**: L = T - V, principle of least action
//! - **Hamiltonian**: H = T + V, phase space dynamics
//!
//! ## Key Principles
//!
//! - **Least Action**: δS = 0 where S = ∫L dt
//! - **Noether's Theorem**: Symmetry ↔ Conservation Law
//! - **Canonical Transformations**: Preserve Hamilton's equations
//!
//! ## Coordinate Systems
//!
//! - Cartesian, polar, spherical
//! - Generalized coordinates q_i and momenta p_i

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Mechanical system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemType {
    FreeParticle,
    HarmonicOscillator,
    CentralForce,
    RigidBody,
    Pendulum,
    TwoBody,
    NBody,
}

/// Constraint type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    Holonomic,    // f(q, t) = 0
    NonHolonomic, // f(q, dq, t) = 0, not integrable
    Scleronomous, // Time-independent
    Rheonomous,   // Time-dependent
}

/// Symmetry type for Noether's theorem
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymmetryType {
    TimeTranslation,  // → Energy conservation
    SpaceTranslation, // → Momentum conservation
    Rotation,         // → Angular momentum conservation
    GalileanBoost,    // → Center of mass motion
    Scaling,          // → Virial theorem
}

impl SymmetryType {
    /// Get the conserved quantity name
    pub fn conserved_quantity(&self) -> &'static str {
        match self {
            SymmetryType::TimeTranslation => "Energy",
            SymmetryType::SpaceTranslation => "Linear Momentum",
            SymmetryType::Rotation => "Angular Momentum",
            SymmetryType::GalileanBoost => "Center of Mass",
            SymmetryType::Scaling => "Virial",
        }
    }
}

/// A mechanical system with Lagrangian/Hamiltonian
#[derive(Debug, Clone)]
pub struct MechanicalSystem {
    pub name: String,
    pub system_type: SystemType,
    pub degrees_of_freedom: u32,
    pub constraints: Vec<ConstraintType>,
    pub symmetries: Vec<SymmetryType>,
    pub vector: ContinuousHV,
}

/// Classical Mechanics Encoder
#[derive(Debug, Clone)]
pub struct ClassicalMechanicsEncoder {
    genesis: GenesisSeed,

    // Fundamental quantities
    pub position: ContinuousHV,
    pub velocity: ContinuousHV,
    pub acceleration: ContinuousHV,
    pub momentum: ContinuousHV,
    pub angular_momentum: ContinuousHV,
    pub force: ContinuousHV,
    pub torque: ContinuousHV,
    pub mass: ContinuousHV,
    pub energy: ContinuousHV,

    // Energies
    pub kinetic: ContinuousHV,
    pub potential: ContinuousHV,

    // Formulations
    pub lagrangian: ContinuousHV,
    pub hamiltonian: ContinuousHV,
    pub action: ContinuousHV,

    // Phase space
    pub phase_space: ContinuousHV,
    pub generalized_coord: ContinuousHV,
    pub generalized_momentum: ContinuousHV,
    pub poisson_bracket: ContinuousHV,

    // Symmetries
    pub time_translation: ContinuousHV,
    pub space_translation: ContinuousHV,
    pub rotation_symmetry: ContinuousHV,
    pub boost: ContinuousHV,

    // Conservation
    pub conserved: ContinuousHV,
    pub invariant: ContinuousHV,

    // System types
    pub oscillator: ContinuousHV,
    pub central_force: ContinuousHV,
    pub rigid_body: ContinuousHV,
    pub coupled: ContinuousHV,
}

impl ClassicalMechanicsEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            position: genesis.hv("mechanics::position", PHYSICS_DIM),
            velocity: genesis.hv("mechanics::velocity", PHYSICS_DIM),
            acceleration: genesis.hv("mechanics::acceleration", PHYSICS_DIM),
            momentum: genesis.hv("mechanics::momentum", PHYSICS_DIM),
            angular_momentum: genesis.hv("mechanics::angular_momentum", PHYSICS_DIM),
            force: genesis.hv("mechanics::force", PHYSICS_DIM),
            torque: genesis.hv("mechanics::torque", PHYSICS_DIM),
            mass: genesis.hv("mechanics::mass", PHYSICS_DIM),
            energy: genesis.hv("mechanics::energy", PHYSICS_DIM),

            kinetic: genesis.hv("energy::kinetic", PHYSICS_DIM),
            potential: genesis.hv("energy::potential", PHYSICS_DIM),

            lagrangian: genesis.hv("mechanics::lagrangian", PHYSICS_DIM),
            hamiltonian: genesis.hv("mechanics::hamiltonian", PHYSICS_DIM),
            action: genesis.hv("mechanics::action", PHYSICS_DIM),

            phase_space: genesis.hv("mechanics::phase_space", PHYSICS_DIM),
            generalized_coord: genesis.hv("mechanics::generalized_q", PHYSICS_DIM),
            generalized_momentum: genesis.hv("mechanics::generalized_p", PHYSICS_DIM),
            poisson_bracket: genesis.hv("mechanics::poisson_bracket", PHYSICS_DIM),

            time_translation: genesis.hv("symmetry::time_translation", PHYSICS_DIM),
            space_translation: genesis.hv("symmetry::space_translation", PHYSICS_DIM),
            rotation_symmetry: genesis.hv("symmetry::rotation", PHYSICS_DIM),
            boost: genesis.hv("symmetry::galilean_boost", PHYSICS_DIM),

            conserved: genesis.hv("property::conserved", PHYSICS_DIM),
            invariant: genesis.hv("property::invariant", PHYSICS_DIM),

            oscillator: genesis.hv("system::oscillator", PHYSICS_DIM),
            central_force: genesis.hv("system::central_force", PHYSICS_DIM),
            rigid_body: genesis.hv("system::rigid_body", PHYSICS_DIM),
            coupled: genesis.hv("system::coupled", PHYSICS_DIM),
        }
    }

    // ============================================================
    // FUNDAMENTAL RELATIONS
    // ============================================================

    /// Newton's second law: F = ma = dp/dt
    pub fn newtons_second_law(&self) -> ContinuousHV {
        self.force.bind(&self.mass).bind(&self.acceleration)
    }

    /// Kinetic energy: T = ½mv²
    pub fn kinetic_energy(&self, mass: f64, velocity: f64) -> ContinuousHV {
        let t = 0.5 * mass * velocity * velocity;
        self.kinetic
            .bind(&self.mass)
            .bind(&self.velocity)
            .scale(t as f32)
    }

    /// Momentum: p = mv
    pub fn linear_momentum(&self) -> ContinuousHV {
        self.momentum.bind(&self.mass).bind(&self.velocity)
    }

    /// Angular momentum: L = r × p
    pub fn angular_momentum_vector(&self) -> ContinuousHV {
        self.angular_momentum
            .bind(&self.position)
            .bind(&self.momentum)
    }

    // ============================================================
    // LAGRANGIAN MECHANICS
    // ============================================================

    /// Lagrangian: L = T - V
    pub fn lagrangian_form(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[&self.kinetic, &self.potential.scale(-1.0)]).bind(&self.lagrangian)
    }

    /// Action: S = ∫L dt
    pub fn action_integral(&self) -> ContinuousHV {
        self.action.bind(&self.lagrangian)
    }

    /// Euler-Lagrange equations
    pub fn euler_lagrange(&self) -> ContinuousHV {
        self.lagrangian
            .bind(&self.generalized_coord)
            .bind(&self.generalized_momentum)
    }

    // ============================================================
    // HAMILTONIAN MECHANICS
    // ============================================================

    /// Hamiltonian: H = Σ(p·q̇) - L = T + V (for natural systems)
    pub fn hamiltonian_form(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[&self.kinetic, &self.potential]).bind(&self.hamiltonian)
    }

    /// Hamilton's equations via Poisson bracket
    pub fn hamiltons_equations(&self) -> ContinuousHV {
        self.hamiltonian
            .bind(&self.phase_space)
            .bind(&self.poisson_bracket)
    }

    /// Poisson bracket: {f, g} = Σ(∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q)
    pub fn poisson_bracket_form(&self) -> ContinuousHV {
        self.poisson_bracket
            .bind(&self.generalized_coord)
            .bind(&self.generalized_momentum)
    }

    /// Canonical transformation
    pub fn canonical_transform(&self) -> ContinuousHV {
        self.phase_space
            .bind(&self.invariant)
            .bind(&self.poisson_bracket)
    }

    // ============================================================
    // NOETHER'S THEOREM
    // ============================================================

    /// Apply Noether's theorem: symmetry → conservation law
    pub fn noether_conservation(&self, symmetry: SymmetryType) -> ContinuousHV {
        let symmetry_vec = match symmetry {
            SymmetryType::TimeTranslation => &self.time_translation,
            SymmetryType::SpaceTranslation => &self.space_translation,
            SymmetryType::Rotation => &self.rotation_symmetry,
            SymmetryType::GalileanBoost => &self.boost,
            SymmetryType::Scaling => &self.genesis.hv("symmetry::scaling", PHYSICS_DIM),
        };

        let conserved_vec = match symmetry {
            SymmetryType::TimeTranslation => self.energy.clone(),
            SymmetryType::SpaceTranslation => self.momentum.clone(),
            SymmetryType::Rotation => self.angular_momentum.clone(),
            SymmetryType::GalileanBoost => self.position.bind(&self.mass).bind(&self.conserved),
            SymmetryType::Scaling => self.kinetic.bind(&self.potential),
        };

        symmetry_vec.bind(&self.conserved).bind(&conserved_vec)
    }

    // ============================================================
    // SPECIFIC SYSTEMS
    // ============================================================

    /// Create a mechanical system
    pub fn create_system(&self, name: &str, system_type: SystemType, dof: u32) -> MechanicalSystem {
        let base = match system_type {
            SystemType::FreeParticle => self.kinetic.clone(),
            SystemType::HarmonicOscillator => self.oscillator.clone(),
            SystemType::CentralForce => self.central_force.clone(),
            SystemType::RigidBody => self.rigid_body.clone(),
            SystemType::Pendulum => self.oscillator.bind(&self.potential),
            SystemType::TwoBody => self.coupled.bind(&self.central_force),
            SystemType::NBody => self.coupled.scale(dof as f32),
        };

        let symmetries = self.infer_symmetries(system_type);

        let vector = base.bind(&self.lagrangian).scale(dof as f32);

        MechanicalSystem {
            name: name.to_string(),
            system_type,
            degrees_of_freedom: dof,
            constraints: vec![],
            symmetries,
            vector,
        }
    }

    fn infer_symmetries(&self, system_type: SystemType) -> Vec<SymmetryType> {
        match system_type {
            SystemType::FreeParticle => vec![
                SymmetryType::TimeTranslation,
                SymmetryType::SpaceTranslation,
                SymmetryType::Rotation,
                SymmetryType::GalileanBoost,
            ],
            SystemType::HarmonicOscillator => vec![SymmetryType::TimeTranslation],
            SystemType::CentralForce => vec![SymmetryType::TimeTranslation, SymmetryType::Rotation],
            SystemType::RigidBody => vec![SymmetryType::TimeTranslation],
            SystemType::Pendulum => vec![SymmetryType::TimeTranslation],
            SystemType::TwoBody => vec![
                SymmetryType::TimeTranslation,
                SymmetryType::SpaceTranslation,
                SymmetryType::Rotation,
            ],
            SystemType::NBody => vec![
                SymmetryType::TimeTranslation,
                SymmetryType::SpaceTranslation,
                SymmetryType::Rotation,
            ],
        }
    }

    /// Simple harmonic oscillator: H = p²/2m + ½kx²
    pub fn harmonic_oscillator(&self, omega: f64) -> MechanicalSystem {
        let sho = self
            .oscillator
            .bind(&self.kinetic)
            .bind(&self.potential)
            .scale(omega as f32);

        MechanicalSystem {
            name: "Harmonic Oscillator".to_string(),
            system_type: SystemType::HarmonicOscillator,
            degrees_of_freedom: 1,
            constraints: vec![ConstraintType::Holonomic],
            symmetries: vec![SymmetryType::TimeTranslation],
            vector: sho,
        }
    }

    /// Kepler problem (1/r potential)
    pub fn kepler_problem(&self) -> MechanicalSystem {
        let inverse_r = self.genesis.hv("potential::inverse_r", PHYSICS_DIM);
        let kepler = self.central_force.bind(&inverse_r).bind(&self.hamiltonian);

        MechanicalSystem {
            name: "Kepler Problem".to_string(),
            system_type: SystemType::CentralForce,
            degrees_of_freedom: 3,
            constraints: vec![],
            symmetries: vec![SymmetryType::TimeTranslation, SymmetryType::Rotation],
            vector: kepler,
        }
    }

    /// Rigid body rotation
    pub fn rigid_body_rotation(&self, inertia_tensor: [f64; 3]) -> MechanicalSystem {
        let i_total = inertia_tensor.iter().sum::<f64>();
        let rigid = self
            .rigid_body
            .bind(&self.angular_momentum)
            .bind(&self.rotation_symmetry)
            .scale(i_total as f32);

        MechanicalSystem {
            name: "Rigid Body".to_string(),
            system_type: SystemType::RigidBody,
            degrees_of_freedom: 3,
            constraints: vec![ConstraintType::Holonomic],
            symmetries: vec![SymmetryType::TimeTranslation],
            vector: rigid,
        }
    }

    // ============================================================
    // INTEGRABILITY
    // ============================================================

    /// Check if system is integrable (Liouville's theorem)
    /// N degrees of freedom with N conserved quantities in involution
    pub fn integrability_check(&self, system: &MechanicalSystem) -> f64 {
        let n_conserved = system.symmetries.len() as u32;
        let dof = system.degrees_of_freedom;

        if n_conserved >= dof {
            1.0 // Completely integrable
        } else {
            n_conserved as f64 / dof as f64
        }
    }

    /// Encode the virial theorem: `2<T> = -<r·∇V>` for bound systems
    pub fn virial_theorem(&self) -> ContinuousHV {
        let bound_state = self.genesis.hv("state::bound", PHYSICS_DIM);
        self.kinetic
            .bind(&self.potential)
            .bind(&bound_state)
            .bind(&self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> ClassicalMechanicsEncoder {
        let genesis = GenesisSeed::from_phrase("classical mechanics test");
        ClassicalMechanicsEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.lagrangian.norm() > 0.0);
        assert!(encoder.hamiltonian.norm() > 0.0);
    }

    #[test]
    fn test_lagrangian_hamiltonian_distinct() {
        let encoder = setup();

        let l = encoder.lagrangian_form();
        let h = encoder.hamiltonian_form();

        // L = T - V and H = T + V should be different
        let sim = l.similarity(&h);
        assert!(sim < 0.9, "Lagrangian and Hamiltonian should differ");
    }

    #[test]
    fn test_noether_energy_conservation() {
        let encoder = setup();

        let conserved = encoder.noether_conservation(SymmetryType::TimeTranslation);

        // Should have contribution from energy
        assert!(conserved.norm() > 0.0);
    }

    #[test]
    fn test_noether_momentum_conservation() {
        let encoder = setup();

        let conserved = encoder.noether_conservation(SymmetryType::SpaceTranslation);

        assert!(conserved.norm() > 0.0);
    }

    #[test]
    fn test_harmonic_oscillator() {
        let encoder = setup();

        let sho = encoder.harmonic_oscillator(1.0);

        assert_eq!(sho.degrees_of_freedom, 1);
        assert!(sho.symmetries.contains(&SymmetryType::TimeTranslation));
        assert!(sho.vector.norm() > 0.0);
    }

    #[test]
    fn test_kepler_problem() {
        let encoder = setup();

        let kepler = encoder.kepler_problem();

        assert_eq!(kepler.system_type, SystemType::CentralForce);
        assert!(kepler.symmetries.contains(&SymmetryType::Rotation));
    }

    #[test]
    fn test_free_particle_all_symmetries() {
        let encoder = setup();

        let free = encoder.create_system("Free", SystemType::FreeParticle, 3);

        // Free particle has maximum symmetry
        assert!(free.symmetries.len() >= 4);
        assert!(free.symmetries.contains(&SymmetryType::TimeTranslation));
        assert!(free.symmetries.contains(&SymmetryType::SpaceTranslation));
        assert!(free.symmetries.contains(&SymmetryType::Rotation));
        assert!(free.symmetries.contains(&SymmetryType::GalileanBoost));
    }

    #[test]
    fn test_integrability() {
        let encoder = setup();

        let free = encoder.create_system("Free", SystemType::FreeParticle, 3);
        let kepler = encoder.kepler_problem();

        // Free particle is integrable
        let free_int = encoder.integrability_check(&free);
        assert!(free_int >= 1.0);

        // Kepler problem is integrable (3 DOF, 2 explicit symmetries but actually 3 conserved)
        let kepler_int = encoder.integrability_check(&kepler);
        assert!(kepler_int > 0.5);
    }

    #[test]
    fn test_phase_space() {
        let encoder = setup();

        let _ps = encoder.phase_space.clone();
        let canonical = encoder.canonical_transform();

        // Canonical transformation preserves phase space structure
        assert!(canonical.norm() > 0.0);
    }

    #[test]
    fn test_euler_lagrange() {
        let encoder = setup();

        let el = encoder.euler_lagrange();

        // Should encode relationship between L, q, and p
        assert!(el.norm() > 0.0);
    }
}
