// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Physics Primitives: From Quarks to Chemistry
//!
//! This module implements a compositional physics hierarchy where higher-level
//! structures are **derived** from lower-level ones using HDC operations.

mod advanced_materials;
pub mod chemistry;
mod consciousness_bridge;
pub mod constants;
mod coupled_physics;
mod design_integration;
mod design_space;
mod economics;
mod electron_nuclear_coupling;
mod geometry;
mod hadrons;
mod high_entropy_alloys;
mod inverse_search;
mod literature_validation;
mod manufacturing;
mod neutron_shielding;
mod nuclear;
mod periodic_table;
mod phonon_dynamics;
mod pulse_dynamics;
mod radiation_damage;
mod spark_engine;
mod spark_prototype_spec;
mod standard_model;
mod thermal_transport;
mod trajectory_analysis;
#[allow(dead_code)]
// Most items unused; LcfPhysicsConstants, GamowIntegrationResult, DDChannelResult used by coupled_physics
mod trigger_systems;
mod uncertainty;

// HDC Scientific Knowledge Expansion modules
mod antimatter;
mod astrophysics;
mod emergence_chain;
mod hdc_emergence_metrics;
mod molecular_biology;
mod neuroscience;
mod phase_transitions;
#[allow(dead_code)] // Exploratory: QFT Feynman diagrams, propagators, vertices
mod qft;
mod quantum_gravity;
pub mod thermodynamics;
// true_phi relocated to crate::consciousness_metrics

// Phase 2: Additional physics domains (pub mod to avoid glob conflicts)
// Exploratory modules - HDC encodings for fundamental physics domains.
// Items are publicly exported; allow(dead_code) suppresses warnings for
// items not yet consumed within the crate itself.
#[allow(dead_code)]
pub mod condensed_matter;
mod cosmology;
#[allow(dead_code)]
pub mod electromagnetism;
mod fluid_dynamics;
#[allow(dead_code)]
pub mod general_relativity;
#[allow(dead_code)]
pub mod plasma_physics;
mod quantum_information;

// Phase 3: Laws derivation, classical mechanics, and more domains
// Exploratory modules - HDC encodings for classical and statistical physics.
// Public APIs are re-exported below; allow(dead_code) suppresses warnings
// for items not yet consumed within the crate.
#[allow(dead_code)]
mod acoustics;
#[allow(dead_code)]
mod chemical_kinetics;
#[allow(dead_code)]
mod classical_mechanics;
#[allow(dead_code)]
mod derived_laws;
#[allow(dead_code)]
mod statistical_mechanics;

// Phase 4: Specialized domains and analogy discovery
// Exploratory modules - HDC encodings for specialized physics and cross-domain
// analogy discovery. Public APIs are re-exported below.
#[allow(dead_code)]
mod analogy_engine;
#[allow(dead_code)]
mod biophysics;
#[allow(dead_code)]
mod geophysics;
#[allow(dead_code)]
mod optics;

// Phase 5: New physics implementations (reorganization plan)
mod chaos_dynamics;
mod decoherence;
mod nonequilibrium;
mod quantum_tunneling;
mod tensor_algebra;

// Phase 6: Simulation bridge (connects encoding modules to dynamical system framework)
pub mod simulation_bridge;

// Integration examples showing cross-module usage
#[cfg(test)]
mod physics_benchmark_gates;
#[cfg(test)]
mod physics_correctness_audit;
#[cfg(test)]
mod physics_cross_validation;
#[cfg(test)]
mod physics_integration_examples;
#[cfg(test)]
mod physics_numerical_validation;
#[cfg(test)]
mod physics_proptest_validation;
#[cfg(test)]
mod physics_test_helpers;
#[cfg(test)]
mod physics_validation_particle;
#[cfg(test)]
mod physics_validation_r7;

// Demonstration module
#[cfg(test)]
mod demo;

pub use hadrons::*;
pub use periodic_table::*;
pub use standard_model::*;
// Note: chemistry::ReactionType conflicts with periodic_table::ReactionType.
// Access chemistry types via physics::chemistry:: prefix.
// pub use chemistry::*;
pub use advanced_materials::*;
pub use consciousness_bridge::*;
pub use coupled_physics::*;
pub use design_integration::*;
pub use design_space::*;
pub use economics::*;
pub use electron_nuclear_coupling::*;
pub use geometry::*;
pub use high_entropy_alloys::*;
pub use inverse_search::*;
pub use literature_validation::*;
pub use manufacturing::*;
pub use neutron_shielding::*;
pub use nuclear::*;
pub use phonon_dynamics::*;
pub use pulse_dynamics::*;
pub use radiation_damage::*;
pub use spark_engine::*;
pub use spark_prototype_spec::*;
pub use thermal_transport::*;
pub use trajectory_analysis::*;
pub use uncertainty::*;

// HDC Scientific Knowledge Expansion exports
pub use crate::consciousness_metrics::*;
pub use antimatter::*;
pub use astrophysics::*;
pub use emergence_chain::*;
pub use hdc_emergence_metrics::*;
pub use molecular_biology::*;
pub use neuroscience::*;
pub use phase_transitions::*;
pub use qft::*;
pub use quantum_gravity::*;
pub use thermodynamics::*;

// Phase 2: exports (non-conflicting only; access others via physics::module_name)
pub use cosmology::*;
pub use fluid_dynamics::*;
pub use quantum_information::*;

// Phase 3: exports (statistical_mechanics and chemical_kinetics have
// conflicting names with thermodynamics and chemistry - access via physics::module_name)
pub use classical_mechanics::*;
pub use derived_laws::*;
// Note: statistical_mechanics conflicts with thermodynamics (K_BOLTZMANN, Ensemble)
// Note: chemical_kinetics conflicts with chemistry (ReactionType, Reaction)
pub use acoustics::*;

// Phase 4: exports
pub use analogy_engine::*;
pub use biophysics::*;
pub use geophysics::*;
pub use optics::*;

// Phase 5: exports (new physics implementations)
pub use chaos_dynamics::*;
pub use decoherence::*;
pub use nonequilibrium::*;
pub use quantum_tunneling::*;
pub use tensor_algebra::*;
