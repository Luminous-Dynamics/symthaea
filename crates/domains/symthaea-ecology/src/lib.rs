// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-ecology
//!
//! Analytic population ecology: instantaneous and delayed density dependence,
//! monotone and oscillatory generation maps, harvested and Allee thresholds,
//! occupancy and source-sink networks, stage structure, finite-population
//! branching risk, epidemic thresholds, biodiversity and trophic accounting,
//! succession, resource-consumer and predator-prey systems, dynamic competition,
//! climate-moisture-nutrient replay, calibration,
//! continuation, recovery diagnostics, and oracle metrics.
//!
//! **Non-duplication:** this is the closed-form / differential-equation
//! counterpart to `symthaea-alife`'s *agent-based* predator-prey simulation
//! (which derives interaction rates from real per-agent choices). Here the rates
//! are model parameters and the results are analytic (equilibria, conserved
//! quantities, stability conditions). `symthaea-population` is population
//! *genetics*, and `symthaea-biota` is a sanctuary platform — neither models
//! ecological dynamics.
//!
//! Pure `std`, zero dependencies, and no `symthaea-core` link.
//! Models are checked against closed forms, invariants, Jacobians, bifurcation
//! thresholds, and independently computed numerical oracles.
//!
//! ## Example
//!
//! ```
//! use symthaea_ecology::LotkaVolterra;
//! let m = LotkaVolterra { alpha: 1.0, beta: 0.1, delta: 0.075, gamma: 1.5 };
//! // Coexistence equilibrium (γ/δ, α/β) = (20, 10).
//! let (prey, pred) = m.equilibrium();
//! assert!((prey - 20.0).abs() < 1e-12 && (pred - 10.0).abs() < 1e-12);
//! ```

pub mod allee;
pub mod beverton_holt;
pub mod branching;
pub mod calibration;
pub mod chemostat;
pub mod competition;
pub mod delayed_logistic;
pub mod diversity;
pub mod environment;
pub mod environment_timeline;
pub mod error;
pub mod harvested_logistic;
pub mod integration;
pub mod leslie;
pub mod logistic;
pub mod lotka_volterra;
pub mod metapopulation;
pub mod moisture_environment;
pub mod network_metapopulation;
pub mod nutrient_environment;
pub mod oracle;
pub mod periodic_environment;
pub mod recovery;
pub mod ricker;
pub mod rosenzweig_macarthur;
pub mod sir;
pub mod source_sink;
pub mod succession;
pub mod trophic_transfer;
pub mod two_patch;

pub use allee::{AlleeBasin, AlleeEquilibrium, AlleeEquilibriumStability, StrongAlleeModel};
pub use beverton_holt::{BevertonHoltModel, BevertonHoltSample};
pub use branching::{BranchingRegime, PoissonBranchingProcess};
pub use calibration::{LogisticCalibration, fit_logistic_known_capacity};
pub use chemostat::{ChemostatModel, ChemostatRegime, ChemostatSample, ChemostatState};
pub use competition::{Competition, CompetitionDynamics, CompetitionOutcome, CompetitionStability};
pub use delayed_logistic::{
    DelayEquilibriumStability, DelayedLogisticSample, HutchinsonDelayLogistic,
};
pub use diversity::{BiodiversitySummary, biodiversity_summary};
pub use environment::{
    EnvironmentalDrivers, GaussianThermalResponse, LogisticEnvironmentCoupling,
    LogisticEnvironmentEvaluation,
};
pub use environment_timeline::{
    EnvironmentalDriverSource, EnvironmentalTimeline, EnvironmentalWaypoint,
    LogisticEnvironmentSample, simulate_logistic_driver_source, simulate_logistic_environment,
};
pub use error::ModelError;
pub use harvested_logistic::{
    HarvestEquilibrium, HarvestEquilibriumStability, HarvestRegime, HarvestedLogistic,
};
pub use integration::{MAX_TRAJECTORY_STEPS, PopulationPairSample, PopulationSample};
pub use leslie::{
    AsymptoticGrowth, LeslieAnalysis, LeslieMatrix, MAX_STAGES, StagePopulationSample,
};
pub use logistic::LogisticModel;
pub use lotka_volterra::LotkaVolterra;
pub use metapopulation::{
    LevinsMetapopulation, MetapopulationEquilibrium, MetapopulationRegime, MetapopulationSample,
};
pub use moisture_environment::{
    HydroEnvironmentalDrivers, HydroLogisticEnvironmentCoupling, HydroLogisticEvaluation,
    SoilMoistureResponse,
};
pub use network_metapopulation::{
    MAX_NETWORK_PATCHES, MAX_NETWORK_TRAJECTORY_VALUES, NetworkOccupancySample,
    NetworkPersistenceDiagnostic, NetworkPersistenceRegime, PatchNetworkMetapopulation,
};
pub use nutrient_environment::{
    MineralNutrientResponse, NutrientEnvironmentalDrivers, NutrientLogisticEnvironmentCoupling,
    NutrientLogisticEvaluation,
};
pub use oracle::{
    ErrorSummary, InvariantDriftSummary, logistic_error_summary, lotka_volterra_invariant_drift,
};
pub use periodic_environment::{PeriodicEnvironment, PeriodicSignal};
pub use recovery::{LinearStability, RecoveryDiagnostic, scalar_recovery_diagnostic};
pub use ricker::{RickerFixedPointStability, RickerModel, RickerSample};
pub use rosenzweig_macarthur::{CoexistenceStability, EnrichmentSlice, RosenzweigMacArthur};
pub use sir::{EpidemicDiagnostic, EpidemicRegime, SirModel, SirSample, SirState};
pub use source_sink::{
    SourceSinkDiagnostic, SourceSinkRegime, SourceSinkSample, SourceSinkState, TwoPatchSourceSink,
};
pub use succession::{
    CommunitySuccession, MAX_SUCCESSION_GENERATIONS, MAX_SUCCESSION_STATES, SuccessionSample,
};
pub use trophic_transfer::{
    MAX_TROPHIC_TRANSFERS, TrophicLevelLedger, TrophicTransferLedger, TrophicTransferModel,
};
pub use two_patch::{TwoPatchMetapopulation, TwoPatchRegime, TwoPatchSample};
