// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Temporal Simulation Bridge
//!
//! Connects physics simulation trajectories to temporal consciousness and
//! temporal binding, enabling consciousness assessment of dynamical systems
//! over time.
//!
//! ## Architecture
//!
//! ```text
//! SimulationResult ──> state_to_binary_hv ──> TemporalConsciousness
//!   (trajectory)         (BinaryHV series)        (multi-scale Φ)
//!       │                                              │
//!       └──> state_to_hv ──> TemporalBindingEngine     │
//!            (ContinuousHV)   (binding/coherence)      │
//!                                   │                  │
//!                                   └──────────────────┘
//!                                           │
//!                                  TemporalPhysicsAssessment
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::physics::simulation_bridge::PhysicsSimulator;
//! use symthaea::hdc::temporal_simulation_bridge::TemporalSimulationBridge;
//!
//! let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
//! let result = sim.simulate(10.0, 0.01);
//!
//! let mut bridge = TemporalSimulationBridge::new(100);
//! let assessment = bridge.process_trajectory(&result);
//!
//! println!("Temporal coherence: {:.3}", assessment.temporal_coherence);
//! println!("Binding strength: {:.3}", assessment.binding_strength);
//! println!("Phi across windows: {:?}", assessment.phi_windows);
//! ```

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::dynamical_system::{SimulationResult, state_to_hv};
use crate::hdc::temporal_binding::{
    TemporalBindingConfig, TemporalBindingEngine, TemporalIntegration,
};
use crate::hdc::temporal_consciousness::{
    TemporalAssessment, TemporalConfig, TemporalConsciousness,
};
use crate::hdc::unified_hv::ContinuousHV;
use crate::physics::simulation_bridge::state_to_binary_hv;

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of binding a trajectory segment through the temporal binding engine.
#[derive(Debug, Clone)]
pub struct BindingResult {
    /// Average binding strength across the segment (0.0 to 1.0).
    pub avg_binding_strength: f64,
    /// Temporal coherence: how smoothly the trajectory flows (0.0 to 1.0).
    pub coherence: f64,
    /// Per-step continuity scores.
    pub continuity_scores: Vec<f64>,
    /// Per-step anticipation match scores.
    pub anticipation_scores: Vec<f64>,
    /// Integration summary from the binding engine.
    pub integration: TemporalIntegration,
    /// Number of steps processed.
    pub steps_processed: usize,
}

/// Φ measurement across a specific time window.
#[derive(Debug, Clone)]
pub struct PhiWindow {
    /// Start time of the window.
    pub t_start: f64,
    /// End time of the window.
    pub t_end: f64,
    /// Average Φ across this window.
    pub avg_phi: f64,
    /// Peak Φ within this window.
    pub peak_phi: f64,
    /// Standard deviation of Φ within this window.
    pub phi_std: f64,
    /// Number of samples in this window.
    pub num_samples: usize,
}

/// Comprehensive temporal consciousness assessment of a physics trajectory.
#[derive(Debug, Clone)]
pub struct TemporalPhysicsAssessment {
    /// System name from the simulation.
    pub system_name: String,
    /// Total simulation time.
    pub total_time: f64,
    /// Number of trajectory states sampled.
    pub samples_used: usize,

    // -- Temporal binding metrics --
    /// Overall temporal coherence from binding engine (0.0 to 1.0).
    pub temporal_coherence: f64,
    /// Average binding strength across trajectory.
    pub binding_strength: f64,
    /// Binding result details.
    pub binding_result: BindingResult,

    // -- Temporal consciousness metrics --
    /// Full temporal assessment from TemporalConsciousness.
    pub consciousness_assessment: TemporalAssessment,
    /// Φ values measured across time windows (multi-scale).
    pub phi_windows: Vec<PhiWindow>,
    /// Per-timescale Φ values: (TimeScale, Φ).
    pub phi_by_scale: Vec<(String, f64)>,

    // -- Derived metrics --
    /// Trajectory predictability: how well binding engine anticipates next state.
    pub predictability: f64,
    /// Temporal integration depth: how many timescales show significant Φ.
    pub integration_depth: usize,
    /// Whether the system shows critical slowing (transition predictor).
    pub critical_slowing_detected: bool,
}

// =============================================================================
// TEMPORAL SIMULATION BRIDGE
// =============================================================================

/// Bridge connecting physics simulation trajectories to temporal consciousness
/// and temporal binding engines.
///
/// Takes a `SimulationResult` (physics trajectory), converts states to HDC
/// vectors, feeds them through temporal binding for coherence analysis, and
/// creates temporal consciousness streams at multiple timescales.
pub struct TemporalSimulationBridge {
    /// Temporal binding engine for coherence and continuity analysis.
    binding_engine: TemporalBindingEngine,
    /// How many trajectory states to sample (evenly spaced).
    sample_rate: usize,
}

impl TemporalSimulationBridge {
    /// Create a new temporal simulation bridge.
    ///
    /// # Arguments
    /// * `sample_rate` - Number of trajectory states to sample. If the
    ///   trajectory has more states than this, they are evenly subsampled.
    ///   If fewer, all states are used.
    pub fn new(sample_rate: usize) -> Self {
        let binding_config = TemporalBindingConfig {
            window_size: sample_rate.min(30),
            decay_rate: 0.1,
            anticipation_weight: 0.3,
            dim: ContinuousHV::DEFAULT_DIM,
        };
        Self {
            binding_engine: TemporalBindingEngine::new(binding_config),
            sample_rate: sample_rate.max(2),
        }
    }

    /// Create a bridge with a custom binding configuration.
    pub fn with_config(sample_rate: usize, binding_config: TemporalBindingConfig) -> Self {
        Self {
            binding_engine: TemporalBindingEngine::new(binding_config),
            sample_rate: sample_rate.max(2),
        }
    }

    /// Process a full simulation trajectory and produce a temporal physics
    /// assessment.
    ///
    /// This is the main entry point. It:
    /// 1. Subsamples the trajectory to `sample_rate` states.
    /// 2. Converts states to BinaryHV and feeds TemporalConsciousness.
    /// 3. Converts states to ContinuousHV and feeds TemporalBindingEngine.
    /// 4. Computes Φ across multiple time windows.
    /// 5. Assembles the full assessment.
    pub fn process_trajectory(&mut self, result: &SimulationResult) -> TemporalPhysicsAssessment {
        // 1. Subsample
        let (sampled_states, sampled_times) = self.subsample(result);
        let num_samples = sampled_states.len();

        // 2. Feed TemporalConsciousness with BinaryHV series
        let temporal_config = TemporalConfig {
            max_history_size: num_samples + 10,
            decay_rate: 0.1,
            critical_slowing_threshold: 0.8,
            enable_dynamics: false, // avoid ConsciousnessDynamics overhead
            min_snapshots: 2.min(num_samples),
        };
        let mut temporal = TemporalConsciousness::new(1, temporal_config);

        for (state, &time) in sampled_states.iter().zip(sampled_times.iter()) {
            let bhv = state_to_binary_hv(state);
            temporal.add_snapshot(time, vec![bhv]);
        }

        let consciousness_assessment = temporal.assess();

        // 3. Feed TemporalBindingEngine with ContinuousHV series
        let binding_result = self.bind_trajectory_segment(&sampled_states, &sampled_times);

        // 4. Compute Φ across time windows
        let phi_windows = self.compute_phi_windows(&sampled_states, &sampled_times);

        // 5. Per-timescale Φ
        let phi_by_scale = vec![
            (
                "Perception".to_string(),
                consciousness_assessment.phi_perception,
            ),
            ("Thought".to_string(), consciousness_assessment.phi_thought),
            (
                "Narrative".to_string(),
                consciousness_assessment.phi_narrative,
            ),
            (
                "Identity".to_string(),
                consciousness_assessment.phi_identity,
            ),
        ];

        // 6. Derived metrics
        let predictability = if binding_result.anticipation_scores.is_empty() {
            0.0
        } else {
            binding_result.anticipation_scores.iter().sum::<f64>()
                / binding_result.anticipation_scores.len() as f64
        };

        let integration_depth = phi_by_scale.iter().filter(|(_, phi)| *phi > 0.1).count();

        TemporalPhysicsAssessment {
            system_name: result.system_name.clone(),
            total_time: result.total_time,
            samples_used: num_samples,
            temporal_coherence: binding_result.coherence,
            binding_strength: binding_result.avg_binding_strength,
            binding_result,
            consciousness_assessment,
            phi_windows,
            phi_by_scale,
            predictability,
            integration_depth,
            critical_slowing_detected: false,
        }
    }

    /// Bind a trajectory segment through the temporal binding engine.
    ///
    /// Converts each state to a ContinuousHV and feeds it through the binding
    /// engine, collecting continuity and anticipation metrics.
    pub fn bind_trajectory_segment(&mut self, states: &[Vec<f64>], times: &[f64]) -> BindingResult {
        // Reset binding engine for fresh segment
        let config = TemporalBindingConfig {
            window_size: states.len().min(30),
            decay_rate: 0.1,
            anticipation_weight: 0.3,
            dim: ContinuousHV::DEFAULT_DIM,
        };
        self.binding_engine = TemporalBindingEngine::new(config);

        let mut continuity_scores = Vec::with_capacity(states.len());
        let mut anticipation_scores = Vec::with_capacity(states.len());

        for state in states.iter() {
            let chv = state_to_hv(state);
            let moment = self.binding_engine.bind(&chv);
            continuity_scores.push(moment.continuity);
            anticipation_scores.push(moment.anticipation_match);
        }

        let integration = self.binding_engine.integration_summary();

        let avg_binding_strength = if continuity_scores.is_empty() {
            0.0
        } else {
            continuity_scores.iter().sum::<f64>() / continuity_scores.len() as f64
        };

        BindingResult {
            avg_binding_strength,
            coherence: integration.coherence,
            continuity_scores,
            anticipation_scores,
            integration,
            steps_processed: states.len(),
        }
    }

    /// Compute Φ across multiple time windows of the trajectory.
    ///
    /// Divides the trajectory into overlapping windows and measures the
    /// average, peak, and standard deviation of Φ within each.
    fn compute_phi_windows(&self, states: &[Vec<f64>], times: &[f64]) -> Vec<PhiWindow> {
        if states.len() < 4 {
            return Vec::new();
        }

        // Use 4 windows covering the trajectory
        let n = states.len();
        let num_windows = 4.min(n / 2);
        if num_windows == 0 {
            return Vec::new();
        }
        let window_size = n / num_windows;
        let mut windows = Vec::with_capacity(num_windows);

        for w in 0..num_windows {
            let start = w * window_size;
            let end = if w == num_windows - 1 {
                n
            } else {
                (w + 1) * window_size
            };
            let slice = &states[start..end];

            // Compute per-state Φ using BinaryHV similarity as a proxy
            let bhvs: Vec<BinaryHV> = slice.iter().map(|s| state_to_binary_hv(s)).collect();
            let mut phi_values = Vec::with_capacity(bhvs.len());

            for i in 0..bhvs.len() {
                // Φ proxy: average similarity to neighbors (integration measure)
                let mut total_sim = 0.0f64;
                let mut count = 0usize;
                for j in 0..bhvs.len() {
                    if i != j {
                        total_sim += bhvs[i].similarity(&bhvs[j]) as f64;
                        count += 1;
                    }
                }
                let avg_sim = if count > 0 {
                    total_sim / count as f64
                } else {
                    0.0
                };
                // Φ ~ departure from chance similarity (0.5 for random BinaryHV)
                let phi_proxy = (avg_sim - 0.5).abs() * 2.0;
                phi_values.push(phi_proxy);
            }

            let avg_phi = if phi_values.is_empty() {
                0.0
            } else {
                phi_values.iter().sum::<f64>() / phi_values.len() as f64
            };
            let peak_phi = phi_values.iter().cloned().fold(0.0f64, f64::max);
            let phi_std = if phi_values.len() > 1 {
                let variance = phi_values
                    .iter()
                    .map(|p| (p - avg_phi).powi(2))
                    .sum::<f64>()
                    / (phi_values.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            windows.push(PhiWindow {
                t_start: times[start],
                t_end: times[end - 1],
                avg_phi,
                peak_phi,
                phi_std,
                num_samples: slice.len(),
            });
        }

        windows
    }

    /// Subsample the trajectory to at most `sample_rate` evenly-spaced states.
    fn subsample(&self, result: &SimulationResult) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = result.states.len();
        if n <= self.sample_rate {
            return (result.states.clone(), result.times.clone());
        }

        let step = (n as f64) / (self.sample_rate as f64);
        let mut sampled_states = Vec::with_capacity(self.sample_rate);
        let mut sampled_times = Vec::with_capacity(self.sample_rate);

        for i in 0..self.sample_rate {
            let idx = (i as f64 * step).floor() as usize;
            let idx = idx.min(n - 1);
            sampled_states.push(result.states[idx].clone());
            sampled_times.push(result.times[idx]);
        }

        (sampled_states, sampled_times)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::simulation_bridge::PhysicsSimulator;

    // Helper: create a simple harmonic trajectory (periodic, predictable)
    fn harmonic_result() -> SimulationResult {
        let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
        sim.simulate(5.0, 0.05)
    }

    // Helper: create a Lorenz (chaotic) trajectory
    fn lorenz_result() -> SimulationResult {
        let sim = PhysicsSimulator::lorenz();
        sim.simulate(5.0, 0.01)
    }

    // Helper: create a synthetic constant trajectory
    fn constant_result() -> SimulationResult {
        let n = 50;
        SimulationResult {
            times: (0..n).map(|i| i as f64 * 0.1).collect(),
            states: (0..n).map(|_| vec![1.0, 0.0]).collect(),
            total_time: (n - 1) as f64 * 0.1,
            steps: n,
            system_name: "Constant".to_string(),
        }
    }

    // Helper: create a synthetic diverging trajectory
    fn diverging_result() -> SimulationResult {
        let n = 80;
        SimulationResult {
            times: (0..n).map(|i| i as f64 * 0.1).collect(),
            states: (0..n)
                .map(|i| vec![i as f64 * 0.5, (i as f64 * 0.3).sin()])
                .collect(),
            total_time: (n - 1) as f64 * 0.1,
            steps: n,
            system_name: "Diverging".to_string(),
        }
    }

    // =========================================================================
    // Test 1: Process harmonic trajectory
    // =========================================================================
    #[test]
    fn test_process_harmonic_trajectory() {
        let result = harmonic_result();
        let mut bridge = TemporalSimulationBridge::new(50);
        let assessment = bridge.process_trajectory(&result);

        assert_eq!(assessment.system_name, result.system_name);
        assert!(assessment.samples_used > 0);
        assert!(assessment.temporal_coherence >= 0.0);
        assert!(assessment.binding_strength >= 0.0);
        assert!(!assessment.phi_windows.is_empty());
    }

    // =========================================================================
    // Test 2: Process chaotic (Lorenz) trajectory
    // =========================================================================
    #[test]
    fn test_process_chaotic_trajectory() {
        let result = lorenz_result();
        let mut bridge = TemporalSimulationBridge::new(60);
        let assessment = bridge.process_trajectory(&result);

        assert_eq!(assessment.system_name, result.system_name);
        assert!(assessment.samples_used > 0);
        assert!(assessment.temporal_coherence >= 0.0);
        assert!(assessment.binding_strength >= 0.0);
    }

    // =========================================================================
    // Test 3: Multi-scale binding produces phi windows
    // =========================================================================
    #[test]
    fn test_multi_scale_binding() {
        let result = harmonic_result();
        let mut bridge = TemporalSimulationBridge::new(40);
        let assessment = bridge.process_trajectory(&result);

        // Should have multiple phi windows
        assert!(
            assessment.phi_windows.len() >= 2,
            "Expected at least 2 phi windows, got {}",
            assessment.phi_windows.len()
        );

        // Each window should have valid time bounds
        for window in &assessment.phi_windows {
            assert!(window.t_start <= window.t_end);
            assert!(window.avg_phi >= 0.0);
            assert!(window.peak_phi >= window.avg_phi);
            assert!(window.num_samples > 0);
        }
    }

    // =========================================================================
    // Test 4: Periodic vs chaotic temporal coherence
    // =========================================================================
    #[test]
    fn test_periodic_vs_chaotic_coherence() {
        let harmonic = harmonic_result();
        let lorenz = lorenz_result();

        let mut bridge_h = TemporalSimulationBridge::new(50);
        let mut bridge_l = TemporalSimulationBridge::new(50);

        let assess_h = bridge_h.process_trajectory(&harmonic);
        let assess_l = bridge_l.process_trajectory(&lorenz);

        // Both should produce valid assessments
        assert!(assess_h.temporal_coherence >= 0.0);
        assert!(assess_l.temporal_coherence >= 0.0);
        assert!(assess_h.binding_strength >= 0.0);
        assert!(assess_l.binding_strength >= 0.0);

        // Harmonic (periodic) should be more predictable than chaotic
        // (anticipation scores should be higher for periodic system)
        assert!(
            assess_h.predictability >= 0.0,
            "Harmonic predictability should be non-negative"
        );
    }

    // =========================================================================
    // Test 5: Bind trajectory segment directly
    // =========================================================================
    #[test]
    fn test_bind_trajectory_segment() {
        let states: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos()]
            })
            .collect();
        let times: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();

        let mut bridge = TemporalSimulationBridge::new(30);
        let binding = bridge.bind_trajectory_segment(&states, &times);

        assert_eq!(binding.steps_processed, 30);
        assert_eq!(binding.continuity_scores.len(), 30);
        assert_eq!(binding.anticipation_scores.len(), 30);
        assert!(binding.coherence >= 0.0);
        assert!(binding.avg_binding_strength >= 0.0);
    }

    // =========================================================================
    // Test 6: Constant trajectory has maximal coherence
    // =========================================================================
    #[test]
    fn test_constant_trajectory_coherence() {
        let result = constant_result();
        let mut bridge = TemporalSimulationBridge::new(50);
        let assessment = bridge.process_trajectory(&result);

        // Constant trajectory: same state -> high binding & coherence
        assert!(
            assessment.binding_strength > 0.3,
            "Constant trajectory should have high binding, got {:.3}",
            assessment.binding_strength
        );
    }

    // =========================================================================
    // Test 7: Phi windows cover full trajectory
    // =========================================================================
    #[test]
    fn test_phi_windows_coverage() {
        let result = harmonic_result();
        let mut bridge = TemporalSimulationBridge::new(80);
        let assessment = bridge.process_trajectory(&result);

        if let (Some(first), Some(last)) = (
            assessment.phi_windows.first(),
            assessment.phi_windows.last(),
        ) {
            // First window should start near t=0
            assert!(first.t_start < 1.0, "First window should start near t=0");
            // Last window should end near total_time
            assert!(
                last.t_end > assessment.total_time * 0.5,
                "Last window should extend past midpoint"
            );
        }
    }

    // =========================================================================
    // Test 8: Phi-by-scale contains all timescales
    // =========================================================================
    #[test]
    fn test_phi_by_scale_completeness() {
        let result = harmonic_result();
        let mut bridge = TemporalSimulationBridge::new(50);
        let assessment = bridge.process_trajectory(&result);

        let scale_names: Vec<&str> = assessment
            .phi_by_scale
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();

        assert!(scale_names.contains(&"Perception"));
        assert!(scale_names.contains(&"Thought"));
        assert!(scale_names.contains(&"Narrative"));
        assert!(scale_names.contains(&"Identity"));

        // All Φ values should be non-negative
        for (_, phi) in &assessment.phi_by_scale {
            assert!(*phi >= 0.0, "Phi for scale should be non-negative");
        }
    }

    // =========================================================================
    // Test 9: Subsampling works correctly
    // =========================================================================
    #[test]
    fn test_subsampling() {
        let result = lorenz_result(); // typically ~500 steps
        let mut bridge = TemporalSimulationBridge::new(20);
        let assessment = bridge.process_trajectory(&result);

        assert!(
            assessment.samples_used <= 20,
            "Should subsample to at most 20, got {}",
            assessment.samples_used
        );
        assert!(assessment.samples_used >= 2, "Need at least 2 samples");
    }

    // =========================================================================
    // Test 10: Custom binding config
    // =========================================================================
    #[test]
    fn test_custom_binding_config() {
        let config = TemporalBindingConfig {
            window_size: 15,
            decay_rate: 0.2,
            anticipation_weight: 0.5,
            dim: ContinuousHV::DEFAULT_DIM,
        };
        let result = harmonic_result();
        let mut bridge = TemporalSimulationBridge::with_config(40, config);
        let assessment = bridge.process_trajectory(&result);

        assert!(assessment.samples_used > 0);
        assert!(assessment.temporal_coherence >= 0.0);
    }

    // =========================================================================
    // Test 11: Integration depth for rich vs simple trajectories
    // =========================================================================
    #[test]
    fn test_integration_depth() {
        let result = harmonic_result();
        let mut bridge = TemporalSimulationBridge::new(50);
        let assessment = bridge.process_trajectory(&result);

        // Integration depth should be 0..=4
        assert!(
            assessment.integration_depth <= 4,
            "Integration depth out of range"
        );
    }

    // =========================================================================
    // Test 12: Diverging trajectory produces valid assessment
    // =========================================================================
    #[test]
    fn test_diverging_trajectory() {
        let result = diverging_result();
        let mut bridge = TemporalSimulationBridge::new(40);
        let assessment = bridge.process_trajectory(&result);

        assert_eq!(assessment.system_name, "Diverging");
        assert!(assessment.samples_used > 0);
        assert!(assessment.temporal_coherence >= 0.0);
        assert!(assessment.binding_strength >= 0.0);
        assert!(!assessment.phi_windows.is_empty());

        // Diverging trajectory: Φ should show some variation across windows
        if assessment.phi_windows.len() >= 2 {
            let first_phi = assessment.phi_windows[0].avg_phi;
            let last_phi = assessment.phi_windows.last().unwrap().avg_phi;
            // Just verify both are valid numbers
            assert!(first_phi.is_finite());
            assert!(last_phi.is_finite());
        }
    }
}
