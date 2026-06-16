// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Trajectory Analysis
//!
//! Analyzes how a fusion reactor design evolves through operational states over time.
//! Uses temporal binding to create a unified trajectory representation.
//!
//! ## What This Measures (Engineering Metrics)
//!
//! - **Trajectory Coherence**: How smoothly does the design evolve?
//! - **State Transitions**: How predictable are operational changes?
//! - **Stability**: Variance in integration metrics over time
//! - **Trend**: Is the design improving or degrading?
//!
//! ## Architecture
//!
//! ```text
//!   Time →
//!   t₀     t₁     t₂     t₃     ...    tₙ
//!   │      │      │      │             │
//!   ▼      ▼      ▼      ▼             ▼
//! ┌───┐  ┌───┐  ┌───┐  ┌───┐       ┌───┐
//! │S₀ │──│S₁ │──│S₂ │──│S₃ │─ ... ─│Sₙ │
//! └───┘  └───┘  └───┘  └───┘       └───┘
//!   │      │      │      │             │
//!   └──────┴──────┴──────┴─────────────┘
//!                    │
//!                    ▼
//!            ┌──────────────┐
//!            │  TRAJECTORY  │
//!            │  BINDING     │
//!            └──────────────┘
//!                    │
//!                    ▼
//!        Unified Trajectory Vector
//! ```
//!
//! ## Note on Naming
//!
//! This module was previously named "physics_temporal_trajectory" with
//! "trajectory consciousness" metrics. The metrics measure engineering
//! properties (coherence, stability, trend), not consciousness.

use super::coupled_physics::{CoupledPhysicsEngine, CoupledSimulationResult, OperatingConditions};
use super::design_integration::{DesignIntegrationEngine, EncodedPhysicsState, IntegrationMetrics};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::temporal_binding::{StreamHealth, TemporalBindingConfig, TemporalBindingEngine};
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for trajectory analysis
#[derive(Clone, Debug)]
pub struct TrajectoryConfig {
    /// Window size (number of states to keep in memory)
    pub window_size: usize,
    /// Decay rate for past states
    pub decay_rate: f64,
    /// Weight for anticipatory influence
    pub anticipation_weight: f64,
    /// HDC dimension (must match physics encoding)
    pub dim: usize,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            decay_rate: 0.05,
            anticipation_weight: 0.2,
            dim: PHYSICS_DIM,
        }
    }
}

/// A physics state along the trajectory
#[derive(Clone, Debug)]
pub struct TrajectoryState {
    /// Step number in the trajectory
    pub step: usize,
    /// The physics simulation result
    pub result: CoupledSimulationResult,
    /// Encoded as HDC vector
    pub encoded: EncodedPhysicsState,
    /// Integration metrics for this state
    pub metrics: IntegrationMetrics,
    /// Temporal binding with previous states
    pub temporal_binding: f32,
    /// Anticipation match (how well predicted)
    pub anticipation_match: f32,
    /// Continuity with previous state
    pub continuity: f32,
}

/// Summary metrics for an entire trajectory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryMetrics {
    /// Overall trajectory quality score
    pub trajectory_quality: f32,
    /// How smoothly the design evolves
    pub coherence: f32,
    /// Past-present binding strength
    pub past_binding: f32,
    /// Present-future binding strength
    pub future_binding: f32,
    /// Narrative length (states integrated)
    pub narrative_length: usize,
    /// Mean state integration
    pub mean_integration: f32,
    /// Variance in integration
    pub integration_variance: f32,
    /// Maximum integration reached
    pub peak_integration: f32,
    /// Minimum integration reached
    pub valley_integration: f32,
}

impl TrajectoryMetrics {
    /// Check if trajectory is healthy
    pub fn is_healthy(&self) -> bool {
        self.coherence > 0.5 && self.trajectory_quality > 0.3
    }

    /// Summary string
    pub fn summary(&self) -> String {
        format!(
            "Trajectory[quality={:.3}, coherence={:.3}, states={}, range={:.3}-{:.3}]",
            self.trajectory_quality,
            self.coherence,
            self.narrative_length,
            self.valley_integration,
            self.peak_integration
        )
    }
}

/// Trajectory analysis engine
///
/// Binds a sequence of physics states into a unified trajectory representation
/// that captures the design's evolution through operational space.
pub struct TrajectoryAnalysisEngine {
    /// Temporal binding engine (uses ContinuousHV internally)
    temporal: TemporalBindingEngine,
    /// Design integration engine
    integration: DesignIntegrationEngine,
    /// History of trajectory states
    history: VecDeque<TrajectoryState>,
    /// Configuration
    config: TrajectoryConfig,
    /// Step counter
    step: usize,
    /// Running trajectory vector (accumulated)
    trajectory_vector: ContinuousHV,
    /// Running sum for mean calculation
    integration_sum: f32,
    /// Running max integration
    integration_max: f32,
    /// Running min integration
    integration_min: f32,
}

// Backwards compatibility alias
pub type PhysicsTrajectoryEngine = TrajectoryAnalysisEngine;

impl TrajectoryAnalysisEngine {
    /// Create from genesis seed
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self::with_config(genesis, TrajectoryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(genesis: &GenesisSeed, config: TrajectoryConfig) -> Self {
        let temporal_config = TemporalBindingConfig {
            window_size: config.window_size,
            decay_rate: config.decay_rate,
            anticipation_weight: config.anticipation_weight,
            dim: config.dim,
        };

        Self {
            temporal: TemporalBindingEngine::new(temporal_config),
            integration: DesignIntegrationEngine::from_genesis(genesis),
            history: VecDeque::new(),
            config,
            step: 0,
            trajectory_vector: ContinuousHV::zero(PHYSICS_DIM),
            integration_sum: 0.0,
            integration_max: f32::NEG_INFINITY,
            integration_min: f32::INFINITY,
        }
    }

    /// Process a new simulation result into the trajectory
    pub fn process(&mut self, result: &CoupledSimulationResult) -> TrajectoryState {
        self.step += 1;

        // Encode the physics state
        let encoded = self.integration.encode_simulation(result);
        let metrics = self.integration.compute_metrics(result);

        // Convert ContinuousHV to ContinuousHV for temporal binding
        let state_vector = continuous_to_real(&encoded.unified_state);

        // Bind temporally
        let moment = self.temporal.bind(&state_vector);

        // Update trajectory vector (slow accumulation)
        let lr = 0.1;
        self.trajectory_vector = self
            .trajectory_vector
            .scale((1.0 - lr) as f32)
            .add(&moment.bound_experience.scale(lr as f32));

        // Update statistics
        self.integration_sum += metrics.overall_integration;
        self.integration_max = self.integration_max.max(metrics.overall_integration);
        self.integration_min = self.integration_min.min(metrics.overall_integration);

        // Create trajectory state
        let state = TrajectoryState {
            step: self.step,
            result: result.clone(),
            encoded,
            metrics,
            temporal_binding: moment.past_integration as f32,
            anticipation_match: moment.anticipation_match as f32,
            continuity: moment.continuity as f32,
        };

        // Store in history
        self.history.push_back(state.clone());
        if self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        state
    }

    /// Process a sequence of operating conditions
    pub fn process_trajectory(
        &mut self,
        physics: &CoupledPhysicsEngine,
        conditions_sequence: &[OperatingConditions],
    ) -> Vec<TrajectoryState> {
        conditions_sequence
            .iter()
            .map(|cond| {
                let result = physics.simulate(cond);
                self.process(&result)
            })
            .collect()
    }

    /// Get trajectory metrics summary
    pub fn trajectory_metrics(&self) -> TrajectoryMetrics {
        let temporal_integration = self.temporal.integration_summary();

        let mean_integration = if self.step > 0 {
            self.integration_sum / self.step as f32
        } else {
            0.0
        };

        // Compute variance
        let variance = if self.history.len() > 1 {
            let sum_sq: f32 = self
                .history
                .iter()
                .map(|s| (s.metrics.overall_integration - mean_integration).powi(2))
                .sum();
            sum_sq / self.history.len() as f32
        } else {
            0.0
        };

        // Trajectory quality: combines temporal coherence with mean integration
        let trajectory_quality = 0.5 * temporal_integration.coherence as f32
            + 0.3 * mean_integration
            + 0.2 * (1.0 - variance.sqrt().min(1.0));

        TrajectoryMetrics {
            trajectory_quality,
            coherence: temporal_integration.coherence as f32,
            past_binding: temporal_integration.past_binding as f32,
            future_binding: temporal_integration.future_binding as f32,
            narrative_length: self.history.len(),
            mean_integration,
            integration_variance: variance,
            peak_integration: if self.integration_max.is_finite() {
                self.integration_max
            } else {
                0.0
            },
            valley_integration: if self.integration_min.is_finite() {
                self.integration_min
            } else {
                0.0
            },
        }
    }

    /// Get stream health (real-time metrics)
    pub fn stream_health(&self) -> StreamHealth {
        self.temporal.stream_health()
    }

    /// Get the unified trajectory vector
    pub fn trajectory_vector(&self) -> &ContinuousHV {
        &self.trajectory_vector
    }

    /// Get the temporal narrative vector
    pub fn narrative(&self) -> &ContinuousHV {
        self.temporal.narrative()
    }

    /// Compare this trajectory to another
    pub fn similarity_to(&self, other: &TrajectoryAnalysisEngine) -> f32 {
        self.trajectory_vector.similarity(&other.trajectory_vector)
    }

    /// Get recent history
    pub fn recent_history(&self, n: usize) -> Vec<&TrajectoryState> {
        self.history.iter().rev().take(n).collect()
    }

    /// Get integration trend (positive = improving)
    pub fn integration_trend(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.history.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, state) in self.history.iter().enumerate() {
            let x = i as f32;
            let y = state.metrics.overall_integration;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    }
}

/// Convert ContinuousHV to ContinuousHV (both are f32-based)
fn continuous_to_real(continuous: &crate::hdc::unified_hv::ContinuousHV) -> ContinuousHV {
    ContinuousHV::from_vec(continuous.values.clone())
}

/// Trajectory comparison result
#[derive(Debug, Clone)]
pub struct TrajectoryComparison {
    /// Similarity between trajectory vectors
    pub vector_similarity: f32,
    /// Difference in trajectory quality
    pub quality_delta: f32,
    /// Which trajectory is healthier
    pub healthier: &'static str,
    /// Metrics for trajectory A
    pub metrics_a: TrajectoryMetrics,
    /// Metrics for trajectory B
    pub metrics_b: TrajectoryMetrics,
}

/// Compare two trajectories
pub fn compare_trajectories(
    engine_a: &TrajectoryAnalysisEngine,
    engine_b: &TrajectoryAnalysisEngine,
) -> TrajectoryComparison {
    let metrics_a = engine_a.trajectory_metrics();
    let metrics_b = engine_b.trajectory_metrics();

    let vector_similarity = engine_a
        .trajectory_vector
        .similarity(&engine_b.trajectory_vector);
    let quality_delta = metrics_a.trajectory_quality - metrics_b.trajectory_quality;

    let healthier = if metrics_a.trajectory_quality > metrics_b.trajectory_quality {
        "A"
    } else {
        "B"
    };

    TrajectoryComparison {
        vector_similarity,
        quality_delta,
        healthier,
        metrics_a,
        metrics_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::FusionReaction;

    fn setup() -> (GenesisSeed, CoupledPhysicsEngine, TrajectoryAnalysisEngine) {
        let genesis = GenesisSeed::from_phrase("trajectory analysis test");
        let physics = CoupledPhysicsEngine::from_genesis(&genesis);
        let trajectory = TrajectoryAnalysisEngine::from_genesis(&genesis);
        (genesis, physics, trajectory)
    }

    #[test]
    fn test_single_state_processing() {
        let (_, physics, mut trajectory) = setup();

        let conditions = OperatingConditions::consumer();
        let result = physics.simulate(&conditions);
        let state = trajectory.process(&result);

        assert_eq!(state.step, 1);
        assert!(state.metrics.overall_integration > 0.0);

        println!("\nSingle state processed:");
        println!("  Step: {}", state.step);
        println!("  Integration: {:.4}", state.metrics.overall_integration);
        println!("  Temporal binding: {:.4}", state.temporal_binding);
        println!("  Continuity: {:.4}", state.continuity);
    }

    #[test]
    fn test_trajectory_evolution() {
        let (_, physics, mut trajectory) = setup();

        println!("\n========================================");
        println!("TRAJECTORY EVOLUTION TEST");
        println!("========================================\n");

        let powers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        for power in powers {
            let conditions = OperatingConditions {
                power_kw: power,
                ..OperatingConditions::consumer()
            };
            let result = physics.simulate(&conditions);
            let state = trajectory.process(&result);

            println!(
                "  {:>5.0} kW: I={:.4}, binding={:.4}, continuity={:.4}",
                power, state.metrics.overall_integration, state.temporal_binding, state.continuity
            );
        }

        let metrics = trajectory.trajectory_metrics();
        println!("\n{}", metrics.summary());
        println!("  Trend: {:.4}", trajectory.integration_trend());

        assert!(metrics.narrative_length == 10);
        assert!(metrics.coherence > 0.0);
    }

    #[test]
    fn test_trajectory_comparison() {
        let genesis = GenesisSeed::from_phrase("comparison test");
        let physics = CoupledPhysicsEngine::from_genesis(&genesis);

        // Trajectory A: Steady low power
        let mut traj_a = TrajectoryAnalysisEngine::from_genesis(&genesis);
        for _ in 0..10 {
            let result = physics.simulate(&OperatingConditions::consumer());
            traj_a.process(&result);
        }

        // Trajectory B: Ramping power
        let mut traj_b = TrajectoryAnalysisEngine::from_genesis(&genesis);
        for i in 0..10 {
            let conditions = OperatingConditions {
                power_kw: 5.0 + i as f64 * 5.0,
                ..OperatingConditions::consumer()
            };
            let result = physics.simulate(&conditions);
            traj_b.process(&result);
        }

        let comparison = compare_trajectories(&traj_a, &traj_b);

        println!("\n========================================");
        println!("TRAJECTORY COMPARISON");
        println!("========================================");
        println!(
            "Steady (A) quality: {:.4}",
            comparison.metrics_a.trajectory_quality
        );
        println!(
            "Ramping (B) quality: {:.4}",
            comparison.metrics_b.trajectory_quality
        );
        println!("Vector similarity: {:.4}", comparison.vector_similarity);
        println!("Healthier trajectory: {}", comparison.healthier);
        println!("========================================\n");

        assert!(
            (comparison.metrics_a.trajectory_quality - comparison.metrics_b.trajectory_quality)
                .abs()
                > 0.001,
            "Trajectories should have different quality values"
        );
    }

    #[test]
    fn test_reaction_type_trajectories() {
        let genesis = GenesisSeed::from_phrase("reaction trajectories");
        let physics = CoupledPhysicsEngine::from_genesis(&genesis);

        println!("\n========================================");
        println!("REACTION TYPE TRAJECTORIES");
        println!("========================================\n");

        for (reaction, name) in [
            (FusionReaction::DD, "D-D"),
            (FusionReaction::DT, "D-T"),
            (FusionReaction::DHe3, "D-He3"),
        ] {
            let mut trajectory = TrajectoryAnalysisEngine::from_genesis(&genesis);

            for _ in 0..15 {
                let conditions = OperatingConditions {
                    power_kw: 5.0,
                    reaction,
                    ..OperatingConditions::consumer()
                };
                let result = physics.simulate(&conditions);
                trajectory.process(&result);
            }

            let metrics = trajectory.trajectory_metrics();
            println!("{:6}: {}", name, metrics.summary());
        }
        println!("========================================\n");
    }

    #[test]
    fn test_stream_health_evolution() {
        let (_, physics, mut trajectory) = setup();

        println!("\n========================================");
        println!("STREAM HEALTH EVOLUTION");
        println!("========================================\n");

        for i in 0..20 {
            let conditions = OperatingConditions {
                power_kw: 5.0 + (i as f64 * 0.1).sin() * 2.0,
                ..OperatingConditions::consumer()
            };
            let result = physics.simulate(&conditions);
            trajectory.process(&result);

            if i % 5 == 4 {
                let health = trajectory.stream_health();
                println!("Step {:>2}: {}", i + 1, health);
            }
        }

        let final_health = trajectory.stream_health();
        println!("\nFinal: {}", final_health);

        assert!(
            final_health.is_flowing,
            "Stream should be flowing after 20 states"
        );
    }
}
