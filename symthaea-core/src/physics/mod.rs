//! # Physics Primitives: From Quarks to Chemistry
//!
//! This module implements a compositional physics hierarchy where higher-level
//! structures are **derived** from lower-level ones using HDC operations.

pub mod constants;
mod standard_model;
mod hadrons;
mod periodic_table;
pub mod chemistry;
mod nuclear;
mod consciousness_bridge;
mod electron_nuclear_coupling;
mod phonon_dynamics;
mod inverse_search;
mod radiation_damage;
mod high_entropy_alloys;
mod advanced_materials;
mod spark_engine;
mod thermal_transport;
mod neutron_shielding;
mod geometry;
mod pulse_dynamics;
#[allow(dead_code)]
mod trigger_systems;
mod coupled_physics;
mod spark_prototype_spec;
mod design_space;
mod literature_validation;
mod uncertainty;
mod manufacturing;
mod economics;
mod design_integration;
mod trajectory_analysis;

// HDC Scientific Knowledge Expansion modules
mod antimatter;
mod thermodynamics;
mod astrophysics;
mod phase_transitions;
#[allow(dead_code)]
mod qft;
mod molecular_biology;
#[allow(dead_code)]
mod neuroscience;
mod quantum_gravity;
mod emergence_chain;
mod true_phi;
mod hdc_emergence_metrics;

// Phase 2: Additional physics domains (pub mod to avoid glob conflicts)
#[allow(dead_code)]
pub mod general_relativity;
mod quantum_information;
#[allow(dead_code)]
pub mod electromagnetism;
#[allow(dead_code)]
pub mod condensed_matter;
mod fluid_dynamics;
#[allow(dead_code)]
pub mod plasma_physics;
mod cosmology;

// Phase 3: Laws derivation, classical mechanics, and more domains
#[allow(dead_code)]
mod derived_laws;
#[allow(dead_code)]
mod classical_mechanics;
#[allow(dead_code)]
mod statistical_mechanics;
#[allow(dead_code)]
mod chemical_kinetics;
#[allow(dead_code)]
mod acoustics;

// Phase 4: Specialized domains and analogy discovery
#[allow(dead_code)]
mod optics;
#[allow(dead_code)]
mod biophysics;
#[allow(dead_code)]
mod geophysics;
#[allow(dead_code)]
mod analogy_engine;

// Phase 5: New physics implementations (reorganization plan)
mod quantum_tunneling;
mod decoherence;
mod tensor_algebra;
mod chaos_dynamics;
mod nonequilibrium;

// Demonstration module
#[cfg(test)]
mod demo;

pub use standard_model::*;
pub use hadrons::*;
pub use periodic_table::*;
// Note: chemistry::ReactionType conflicts with periodic_table::ReactionType.
// Access chemistry types via physics::chemistry:: prefix.
// pub use chemistry::*;
pub use nuclear::*;
pub use consciousness_bridge::*;
pub use electron_nuclear_coupling::*;
pub use phonon_dynamics::*;
pub use inverse_search::*;
pub use radiation_damage::*;
pub use high_entropy_alloys::*;
pub use advanced_materials::*;
pub use spark_engine::*;
pub use thermal_transport::*;
pub use neutron_shielding::*;
pub use geometry::*;
pub use pulse_dynamics::*;
pub use coupled_physics::*;
pub use spark_prototype_spec::*;
pub use design_space::*;
pub use literature_validation::*;
pub use uncertainty::*;
pub use manufacturing::*;
pub use economics::*;
pub use design_integration::*;
pub use trajectory_analysis::*;

// HDC Scientific Knowledge Expansion exports
pub use antimatter::*;
pub use thermodynamics::*;
pub use astrophysics::*;
pub use phase_transitions::*;
pub use qft::*;
pub use molecular_biology::*;
pub use neuroscience::*;
pub use quantum_gravity::*;
pub use emergence_chain::*;
pub use true_phi::*;
pub use hdc_emergence_metrics::*;

// Phase 2: exports (non-conflicting only; access others via physics::module_name)
pub use quantum_information::*;
pub use fluid_dynamics::*;
pub use cosmology::*;

// Phase 3: exports (statistical_mechanics and chemical_kinetics have
// conflicting names with thermodynamics and chemistry - access via physics::module_name)
pub use derived_laws::*;
pub use classical_mechanics::*;
// Note: statistical_mechanics conflicts with thermodynamics (K_BOLTZMANN, Ensemble)
// Note: chemical_kinetics conflicts with chemistry (ReactionType, Reaction)
pub use acoustics::*;

// Phase 4: exports
pub use optics::*;
pub use biophysics::*;
pub use geophysics::*;
pub use analogy_engine::*;

// Phase 5: exports (new physics implementations)
pub use quantum_tunneling::*;
pub use decoherence::*;
pub use tensor_algebra::*;
pub use chaos_dynamics::*;
pub use nonequilibrium::*;
