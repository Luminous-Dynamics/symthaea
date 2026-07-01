// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spark Engine: Lattice Confinement Fusion Physics
//!
//! A physics-honest simulation framework for LCF research, focused on:
//! - Rigorous Gamow integration with screening and phonon effects
//! - Q factor calculation with honest assessment (Q << 1 at room temperature)
//! - Lattice lifetime modeling with DPA tracking
//! - Anomaly investigation (Path C): Understanding observed vs predicted rates
//! - Neutron source design (Path D): Compact neutron generator applications
//!
//! ## Key Finding
//!
//! Standard Gamow physics predicts `<σv>` ~ 10⁻⁵⁰ cm³/s at room temperature,
//! yet NASA observed ~10³ neutrons/s. This gap is the central mystery.
//!
//! ## Modules
//!
//! - **physics**: Core Gamow integration, Q factor, screening calculations
//! - **lattice**: DPA tracking, lattice lifetime, D/Pd ratio decay
//! - **anomaly**: Gap analysis, hypothesis framework
//! - **literature**: Database of LCF/LENR observations with credibility scoring
//! - **hypothesis_models**: Detailed physics models (hot spots, phonon cascade, screening)
//! - **rate_gap**: Quantitative rate gap calculator with sensitivity analysis
//! - **experimental_design**: Tools for designing discriminating experiments
//! - **neutron_source**: Compact neutron generator design
//! - **shielding**: Neutron shielding calculations
//!
//! ## Usage
//!
//! ```rust
//! use spark_engine::physics::{GamowIntegration, QFactor, QFactorParams};
//! use spark_engine::anomaly::AnomalyAnalysis;
//!
//! // Standard physics calculation
//! let gamow = GamowIntegration::dd_rate(300.0, 309.0, 2);
//! let (q, achievable) = QFactor::compute(&gamow, &QFactorParams::lcf_typical());
//! assert!(q < 1.0, "Q < 1 at room temperature - physics is honest");
//!
//! // Anomaly investigation
//! let observed = 1e3; // NASA neutron rate
//! let n_d = 4.8e22; // D atoms/cm³
//! let volume = 0.01; // cm³
//! let predicted = gamow.to_neutron_rate(n_d, volume);
//! let gap = AnomalyAnalysis::compute_gap(observed, predicted);
//! ```

// Core physics modules
pub mod anomaly;
pub mod lattice;
pub mod neutron_source;
pub mod physics;
pub mod shielding;
// pub mod reactor_design;
// pub mod economics;
pub mod error;

// Path C: Anomaly Investigation modules
pub mod experimental_design;
pub mod hypothesis_models;
pub mod literature;
pub mod rate_gap;
// pub mod experimental_proposal;
pub mod uncertainty;

// Integration modules
pub mod bridge;

// Extended fusion physics (D-T, p-B11, muon-catalyzed)
pub mod reactions;

// Multi-physics discovery engine
pub mod multi_physics;

// CfC temporal dynamics analysis
pub mod cfc_physics;

// Bayesian hypothesis discrimination
// pub mod bayesian;

// Emergent physics simulation
// pub mod emergent;

// Design optimization
// pub mod optimization;

// Visualization and export
// pub mod visualization;

// Extended materials database
// pub mod materials;

// Experimental data assimilation and digital twin
// pub mod data_assimilation;

// Temporal degradation modeling
// pub mod degradation;

// ML surrogate models and inverse design
// pub mod surrogate;

// Safety analysis (FMEA, fault trees, control systems, compliance)
// pub mod safety;

// Multi-scale physics coupling
// pub mod multiscale;

// Bayesian optimal experiment design
// pub mod optimal_experiment;

// Core re-exports
pub use anomaly::{AnomalyAnalysis, AnomalyHypothesis};
pub use lattice::{LatticeLifetime, LatticeState};
pub use neutron_source::{NeutronSourceDesign, SourceSpec};
pub use physics::{GamowIntegration, GamowResult, QFactor, QFactorParams};
pub use shielding::{NeutronShielding, ShieldingResult};
// pub use reactor_design::{ReactorDesign, ReactorDesigner, MassBreakdown, CostBreakdown, FuelCycle, ThermalConstraint, CoolingMethod};
pub use error::{Result, SparkError};

// Path C re-exports
pub use experimental_design::{ExperimentDesign, ExperimentDesigner, ExperimentalProgram};
pub use hypothesis_models::{
    HotSpotModel, HypothesisComparison, HypothesisType, PhononCascadeModel, SuperScreeningModel,
};
pub use literature::{
    Credibility, HostMaterial as LitHostMaterial, LiteratureDatabase, Observation, TriggerMethod,
};
pub use rate_gap::{ExperimentalConditions, RateGapAnalysis, RateGapCalculator};

/// Spark Engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Physics constants used throughout
pub mod constants {
    /// D-D astrophysical S-factor (keV·barn)
    pub const DD_S_FACTOR: f64 = 55.0;

    /// D-D Gamow constant (keV^0.5)
    pub const DD_GAMOW_CONSTANT: f64 = 31.38;

    /// D-D Q-value, neutron branch (MeV)
    pub const DD_Q_NEUTRON: f64 = 3.269;

    /// D-D Q-value, proton branch (MeV)
    pub const DD_Q_PROTON: f64 = 4.033;

    /// D-D average Q-value (MeV)
    pub const DD_Q_AVERAGE: f64 = 3.651;

    /// D-D neutron energy (MeV)
    pub const DD_NEUTRON_ENERGY: f64 = 2.45;

    /// D-T neutron energy (MeV)
    pub const DT_NEUTRON_ENERGY: f64 = 14.1;

    /// Electron screening in Pd (eV) - Raiola et al. 2004
    pub const SCREENING_PD_EV: f64 = 309.0;

    /// PdD optical phonon energy (keV)
    pub const PDD_PHONON_KEV: f64 = 0.056;

    /// Boltzmann constant (keV/K)
    pub const KB_KEV: f64 = 8.617e-8;

    /// NASA observed neutron rate (n/s per cm²)
    pub const NASA_OBSERVED_RATE: f64 = 1e3;

    /// Pd density (g/cm³)
    pub const PD_DENSITY: f64 = 12.02;

    /// Pd atomic mass (u)
    pub const PD_ATOMIC_MASS: f64 = 106.42;
}

/// Prelude for common imports
pub mod prelude {
    // Core physics
    pub use crate::anomaly::{AnomalyAnalysis, AnomalyGap, AnomalyHypothesis};
    pub use crate::lattice::{LatticeLifetime, LatticeState};
    pub use crate::neutron_source::{NeutronSourceDesign, NeutronSourceDesigner, SourceSpec};
    pub use crate::physics::{GamowIntegration, GamowResult, QFactor, QFactorParams};
    pub use crate::shielding::{NeutronShielding, ShieldingResult};

    // Path C: Anomaly investigation
    pub use crate::experimental_design::{
        ExperimentDesign, ExperimentDesigner, ExperimentalProgram,
    };
    pub use crate::hypothesis_models::{
        HotSpotModel, HypothesisComparison, HypothesisType, PhononCascadeModel, SuperScreeningModel,
    };
    pub use crate::literature::{LiteratureDatabase, Observation};
    pub use crate::rate_gap::{RateGapAnalysis, RateGapCalculator};

    pub use crate::constants;
    pub use crate::{Result, SparkError};
}
