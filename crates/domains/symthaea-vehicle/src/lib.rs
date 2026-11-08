// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-vehicle
//!
//! HDC-LTC + FEP Active Inference autonomous vehicle control via bicycle-model dynamics.
//!
//! Uses the full 16,384D `HdcLtcUnifiedNetwork` from `symthaea-core` as the temporal
//! dynamics engine, with `symthaea-fep` providing precision-weighted Active Inference
//! modulation. Multi-rate architecture: 200Hz physics + 50Hz cognitive tick.
//!
//! ## Architecture — TARGET DEPLOYMENT SHAPE, NOT IMPLEMENTED
//!
//! Everything in this diagram outside the "Cognitive Core" box (Coral/Hailo
//! retina coprocessor, Raspberry Pi host, CAN bus actuators, LoRa mesh
//! radio) is aspirational hardware — none of it exists in this crate, which
//! is a pure-Rust simulation. The in-crate mesh (`swarm.rs`) exchanges peer
//! state instantly with no latency/dropout model. (Labeled honestly per the
//! 2026-07 robotics deep review.)
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │  "Retina" Coprocessor (Coral TPU / Hailo-8)              │
//! │  Camera/LiDAR → 16,384D ContinuousHV                    │
//! └────────────────────┬─────────────────────────────────────┘
//!                      │
//! ┌────────────────────▼─────────────────────────────────────┐
//! │  Symthaea Cognitive Core (Raspberry Pi, 50Hz)            │
//! │                                                          │
//! │  Sensors (14D) → VehicleHdcEncoder → ContinuousHV(16384D)│
//! │                                            ↓             │
//! │                     HdcLtcUnifiedNetwork (3×8 neurons)   │
//! │                         evolve_closed_form(dt)           │
//! │                                            ↓             │
//! │                     VehicleController (16384→3 projection)│
//! │                                            ↓             │
//! │                     VehicleCommand [steering, throttle,   │
//! │                                     brake — all tanh]    │
//! └────────────────────┬─────────────────────────────────────┘
//!                      │
//! ┌────────────────────▼─────────────────────────────────────┐
//! │  CAN Bus → Vehicle Actuators                             │
//! │  Steering motor, throttle-by-wire, brake-by-wire         │
//! └────────────────────┬─────────────────────────────────────┘
//!                      │
//! ┌────────────────────▼─────────────────────────────────────┐
//! │  LoRa Mesh Radio → Swarm Horizon                         │
//! │  2KB Wisdom Vectors from ±5 vehicles                     │
//! │  ABS engagement propagates at speed of light              │
//! └──────────────────────────────────────────────────────────┘
//!
//! Every 4th physics step (50Hz):
//!   ActiveInferenceAgent modulates τ, learning rate, prior precision
//!   EFE decomposition:
//!     Risk  → minimize entropy (collision/traction loss = spike)
//!     Ambiguity → minimize uncertainty (rain/fog → slow down)
//! ```
//!
//! ## Tasks
//!
//! - **LaneKeep**: Maintain lane position at target velocity (~13.4 m/s / 30 mph)
//! - **Follow**: Adaptive cruise — match lead vehicle speed
//! - **LaneChange**: Execute a safe lane change maneuver
//! - **EmergencyStop**: Bring vehicle to zero from speed
//!
//! ## Perturbation Crucibles
//!
//! - **Black Ice**: Friction drops to 0.05 at step 400
//! - **Crosswind**: 3000N sustained lateral wind at step 300
//! - **Sensor Blindness**: Camera channels zeroed (heavy rain)
//! - **Tire Blowout**: Asymmetric grip loss + yaw moment
//! - **Mesh Spoof**: Byzantine false brake signals (must detect + ignore)
//! - **Gauntlet**: All perturbations at staggered intervals

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod classical_baselines;
pub mod counterfactual_decision;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod paired_rollout;
pub mod paired_statistics;
pub mod model_stability;
pub mod plugin;
pub mod reward;
pub mod risk_calibration;
pub mod rule_compliance;
pub mod road;
pub mod runtime_assurance;
pub mod rta_shadow_audit;
pub mod scenario_discovery;
pub mod shadow_evaluation;
pub mod scenarios;
pub mod simulator;
pub mod subterranean;
pub mod subterranean_navigation;
pub mod trajectory_coverage;
pub mod swarm;
pub mod training;
pub mod types;

pub use classical_baselines::{
    PURE_PURSUIT_ROAD_BASELINE_POLICY_ID, STANLEY_ROAD_BASELINE_POLICY_ID,
    PurePursuitRoadConfig, StanleyRoadConfig, VehicleClassicalBaseline,
};
pub use risk_calibration::{
    DEFAULT_RISK_CALIBRATION_BINS, MAX_RISK_CALIBRATION_BINS,
    MIN_RISK_CALIBRATION_BINS, RISK_LOG_LOSS_EPSILON,
    RiskCalibrationBin, RiskCalibrationError, RiskFamilyCalibration,
    RiskForecastObservation, RiskForecastOutcome, VEHICLE_RISK_CALIBRATION_SCHEMA,
    VehicleRiskCalibrationReport, summarize_vehicle_risk_forecasts,
};
pub use rule_compliance::{
    RuleComplianceFinding, RuleComplianceOutcome, RuleComplianceReason, RuleEvaluationError,
    RuleTrajectorySample, TRAFFIC_RULE_COMPLIANCE_SCHEMA, TrafficRuleComplianceReport,
    TrafficRuleEvidence, TrafficRuleEvidenceStatus, TrafficRuleKind, TrafficRuleRequirement,
    evaluate_traffic_rules,
};
pub use counterfactual_decision::{
    COUNTERFACTUAL_DECISION_SCHEMA, CounterfactualActionEvaluation,
    CounterfactualDecisionError, CounterfactualDecisionFinding, CounterfactualDecisionInput,
    CounterfactualDecisionOutcome, CounterfactualDecisionReason, CounterfactualDecisionReport,
    CounterfactualFeasibility, CounterfactualMetricDirection, evaluate_counterfactual_decision,
};
pub use controller::VehicleController;
pub use trajectory_coverage::{
    MIN_FAMILY_COVERAGE_SAMPLES, TRAJECTORY_COVERAGE_TARGET_P95,
    VEHICLE_TRAJECTORY_COVERAGE_SCHEMA, TrajectoryCalibrationObservation,
    TrajectoryCoverageError, TrajectoryCoverageResult, TrajectoryEvaluationObservation,
    TrajectoryFamilyCoverage, VehicleTrajectoryCoverageReport,
    summarize_vehicle_trajectory_coverage,
};
pub use encoder::VehicleHdcEncoder;
pub use fep_agent::{ActiveInferenceVehicleAgent, VehicleFepConfig, VehicleFepResult};
pub use perturbations::{PerturbationSchedule, VehiclePerturbation};
pub use paired_rollout::{
    MAX_PAIRED_ROLLOUT_STEPS, VEHICLE_PAIRED_ROLLOUT_SCHEMA, VehiclePairedRolloutConfig,
    VehiclePairedRolloutError, VehiclePairedRolloutFrame, VehiclePairedRolloutResult,
    VehicleTrajectoryMetrics, run_paired_lane_keep_rollout,
    run_paired_lane_keep_rollout_with_baseline,
};
pub use model_stability::{
    MIN_MODEL_STABILITY_PROFILES, MAX_MODEL_STABILITY_PROFILES,
    VEHICLE_MODEL_STABILITY_SCHEMA, ModelActionEvaluation, ModelActionFeasibility,
    ModelStabilityDirection, ModelStabilityError, ModelStabilityOutcome,
    PerModelStabilityFinding, VehicleModelStabilityFinding, VehicleModelStabilityInput,
    VehicleModelStabilityReport, evaluate_vehicle_model_stability,
};
pub use paired_statistics::{
    MAX_PAIRED_BOOTSTRAP_REPLICATES, MIN_PAIRED_BOOTSTRAP_REPLICATES,
    PAIRED_STATISTICS_SCHEMA, PairedFamilyEffect, PairedMetricDirection,
    PairedMetricObservation, PairedMetricSummary, PairedStatisticsError,
    summarize_paired_metric,
};
pub use reward::{episode_reward, follow_distance_reward, safety_reward, speed_reward};
pub use road::{Road, RoadSegment};
pub use scenario_discovery::{
    CounterexampleEvaluationReceipt, CounterexampleMinimizationError,
    CounterexampleMinimizationTermination, CrossingActor, IntersectionCounterexampleMinimization,
    IntersectionFactorChange, IntersectionGenerationError, IntersectionScenarioCandidate,
    ScenarioRangeSemantics, MAX_COUNTEREXAMPLE_MINIMIZATION_EVALUATIONS,
    generate_seeded_random_intersection_candidates,
    generate_stratified_intersection_candidates, minimize_intersection_counterexample,
};
pub use shadow_evaluation::{
    PD_ROAD_BASELINE_POLICY_ID, VEHICLE_SHADOW_SCHEMA, VehicleShadowError,
    VehicleShadowEvaluator, VehicleShadowInput, VehicleShadowProfile, VehicleShadowRecord,
    VehicleShadowReport, pd_road_baseline_digest,
};
pub use runtime_assurance::{
    VehicleRtaConfig, VehicleRtaDecision, VehicleRtaError, VehicleRtaMode, VehicleRtaObservation,
    VehicleRtaReason, VehicleRtaWitness, VehicleRuntimeAssurance,
};
pub use rta_shadow_audit::{
    VEHICLE_RTA_SHADOW_AUDIT_SCHEMA, VehicleRtaShadowAudit, VehicleRtaShadowFrame,
    VehicleRtaShadowReport,
};
pub use scenarios::{
    ScenarioConfig, ScenarioResult, default_scenarios, results_to_csv, run_all_scenarios,
    run_scenario,
};
pub use simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
pub use subterranean::{CaveRelayDecision, SubterraneanBridge, SurveyAnchor, TunnelRelayNode};
pub use subterranean_navigation::{
    SubterraneanEstimate, SubterraneanNavigator, SubterraneanTrainingContext,
};
pub use swarm::{SwarmConfig, SwarmSimulator};
pub use training::{EpisodeMetrics, VehicleTrainer};
pub use types::*;
