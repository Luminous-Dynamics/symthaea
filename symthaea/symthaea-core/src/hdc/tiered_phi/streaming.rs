// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Streaming Phi Gradient
//!
//! Real-time gradient computation for continuous consciousness optimization.
//!
//! ## Problem
//!
//! Full gradient computation via leave-one-out analysis requires O(n × Φ_complexity)
//! operations. For real-time optimization, we need faster streaming updates.
//!
//! ## Solution
//!
//! Use incremental state from `IncrementalPhiState` to approximate gradients:
//!
//! 1. **Sensitivity Gradient**: Approximate ∂Φ/∂component[i] using cached similarity matrix
//! 2. **Streaming Updates**: Emit gradient events through consciousness streaming
//! 3. **Adaptive Precision**: Trade accuracy for speed based on system requirements
//!
//! ## Complexity
//!
//! - Fast approximation: O(n) using degree centrality
//! - Incremental gradient: O(k × n) when k components change
//! - Full gradient: O(n²) using leave-one-out on cached matrix
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::tiered_phi::streaming::{StreamingPhiGradient, GradientConfig};
//!
//! let mut gradient = StreamingPhiGradient::new(GradientConfig::default());
//!
//! // Compute gradient for current state
//! let grad = gradient.compute_gradient(&components);
//!
//! // Identify optimization targets
//! for (idx, sensitivity) in grad.component_gradients.iter().enumerate() {
//!     if *sensitivity < 0.0 {
//!         println!("Component {} is lowering Φ, consider replacing", idx);
//!     }
//! }
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use super::core::{ApproximationTier, TieredPhi};
use crate::hdc::binary_hv::BinaryHV;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Gradient computation precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GradientPrecision {
    /// O(n): Fast centrality-based approximation
    Fast,
    /// O(n²): Medium precision using cached similarity matrix
    #[default]
    Medium,
    /// O(n³): High precision using leave-one-out on similarity
    High,
}

/// Configuration for streaming gradients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    /// Precision level for gradient computation
    pub precision: GradientPrecision,
    /// Minimum interval between gradient updates (ms)
    pub min_update_interval_ms: u64,
    /// Emit gradient events through streaming
    pub enable_streaming: bool,
    /// Threshold for significant gradient (below this is noise)
    pub significance_threshold: f64,
    /// Number of top components to track for optimization
    pub top_k_components: usize,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            precision: GradientPrecision::Medium,
            min_update_interval_ms: 100,
            enable_streaming: true,
            significance_threshold: 0.01,
            top_k_components: 5,
        }
    }
}

// =============================================================================
// GRADIENT RESULT
// =============================================================================

/// Result of gradient computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiGradient {
    /// Current Phi value
    pub phi: f64,
    /// Gradient for each component: ∂Φ/∂component[i]
    /// Positive = increasing this component helps Φ
    /// Negative = component is hurting integration
    pub component_gradients: Vec<f64>,
    /// Indices of components sorted by gradient magnitude (highest first)
    pub gradient_ranking: Vec<usize>,
    /// Components with significant positive gradient (helpers)
    pub integration_helpers: Vec<usize>,
    /// Components with significant negative gradient (hinderers)
    pub integration_hinderers: Vec<usize>,
    /// Overall gradient magnitude (L2 norm)
    pub gradient_norm: f64,
    /// Direction of steepest ascent (unit vector)
    pub ascent_direction: Vec<f64>,
    /// Computation time
    pub compute_time_us: u64,
    /// Precision level used
    pub precision: GradientPrecision,
}

impl PhiGradient {
    /// Get top k components that most help integration
    pub fn top_helpers(&self, k: usize) -> Vec<(usize, f64)> {
        self.integration_helpers
            .iter()
            .take(k)
            .map(|&idx| (idx, self.component_gradients[idx]))
            .collect()
    }

    /// Get top k components that most hurt integration
    pub fn top_hinderers(&self, k: usize) -> Vec<(usize, f64)> {
        self.integration_hinderers
            .iter()
            .take(k)
            .map(|&idx| (idx, self.component_gradients[idx]))
            .collect()
    }

    /// Suggest optimization action
    pub fn optimization_suggestion(&self) -> OptimizationAction {
        if self.integration_hinderers.is_empty() {
            OptimizationAction::Stable
        } else if !self.integration_helpers.is_empty() {
            OptimizationAction::ReplaceHindererWithHelper {
                hinderer: self.integration_hinderers[0],
                helper_template: self.integration_helpers[0],
            }
        } else {
            OptimizationAction::RemoveHinderer {
                hinderer: self.integration_hinderers[0],
            }
        }
    }
}

/// Suggested optimization action based on gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAction {
    /// System is stable, no action needed
    Stable,
    /// Replace a hinderer component with something like a helper
    ReplaceHindererWithHelper {
        hinderer: usize,
        helper_template: usize,
    },
    /// Remove a hinderer component
    RemoveHinderer { hinderer: usize },
}

// =============================================================================
// STREAMING GRADIENT EVENT
// =============================================================================

/// Event emitted when gradient is computed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientEvent {
    /// Timestamp (Unix millis)
    pub timestamp: u64,
    /// Current Phi
    pub phi: f64,
    /// Change in Phi since last event
    pub phi_delta: f64,
    /// Gradient norm (overall optimization pressure)
    pub gradient_norm: f64,
    /// Top helper indices
    pub top_helpers: Vec<usize>,
    /// Top hinderer indices
    pub top_hinderers: Vec<usize>,
    /// Suggested action
    pub suggestion: OptimizationAction,
}

// =============================================================================
// STREAMING PHI GRADIENT
// =============================================================================

/// Streaming gradient computation with real-time updates
pub struct StreamingPhiGradient {
    /// Configuration
    config: GradientConfig,
    /// Underlying Phi calculator with incremental state
    phi_calc: TieredPhi,
    /// Last gradient result
    last_gradient: Option<PhiGradient>,
    /// Last update time
    last_update: Instant,
    /// Event sender (if streaming enabled)
    event_sender: Option<Sender<GradientEvent>>,
    /// Statistics
    pub stats: GradientStats,
}

/// Statistics for gradient computation
#[derive(Debug, Clone, Default)]
pub struct GradientStats {
    /// Total gradient computations
    pub total_computations: u64,
    /// Fast precision computations
    pub fast_computations: u64,
    /// Medium precision computations
    pub medium_computations: u64,
    /// High precision computations
    pub high_computations: u64,
    /// Total computation time (microseconds)
    pub total_time_us: u64,
    /// Events emitted
    pub events_emitted: u64,
}

impl StreamingPhiGradient {
    /// Create a new streaming gradient calculator
    pub fn new(config: GradientConfig) -> Self {
        Self {
            config,
            phi_calc: TieredPhi::new(ApproximationTier::SampledPartition),
            last_gradient: None,
            last_update: Instant::now(),
            event_sender: None,
            stats: GradientStats::default(),
        }
    }

    /// Create with a custom Phi calculator
    pub fn with_phi_calc(config: GradientConfig, phi_calc: TieredPhi) -> Self {
        Self {
            config,
            phi_calc,
            last_gradient: None,
            last_update: Instant::now(),
            event_sender: None,
            stats: GradientStats::default(),
        }
    }

    /// Enable streaming and get event receiver
    pub fn enable_streaming(&mut self) -> Receiver<GradientEvent> {
        let (sender, receiver) = channel();
        self.event_sender = Some(sender);
        receiver
    }

    /// Compute gradient for current components
    pub fn compute_gradient(&mut self, components: &[BinaryHV]) -> PhiGradient {
        let start = Instant::now();
        let n = components.len();

        // Compute base Phi using incremental method
        let phi = self.phi_calc.compute_incremental(components);

        // Compute gradients based on precision level
        let component_gradients = match self.config.precision {
            GradientPrecision::Fast => self.compute_gradient_fast(components, phi),
            GradientPrecision::Medium => self.compute_gradient_medium(components, phi),
            GradientPrecision::High => self.compute_gradient_high(components, phi),
        };

        // Compute gradient statistics
        let gradient_norm = (component_gradients.iter().map(|g| g * g).sum::<f64>()).sqrt();

        let ascent_direction = if gradient_norm > 0.0 {
            component_gradients
                .iter()
                .map(|g| g / gradient_norm)
                .collect()
        } else {
            vec![0.0; n]
        };

        // Rank by magnitude
        let mut gradient_ranking: Vec<usize> = (0..n).collect();
        gradient_ranking.sort_by(|&a, &b| {
            component_gradients[b]
                .abs()
                .total_cmp(&component_gradients[a].abs())
        });

        // Identify helpers and hinderers
        let threshold = self.config.significance_threshold;
        let integration_helpers: Vec<usize> = (0..n)
            .filter(|&i| component_gradients[i] > threshold)
            .collect();
        let integration_hinderers: Vec<usize> = (0..n)
            .filter(|&i| component_gradients[i] < -threshold)
            .collect();

        let elapsed_us = start.elapsed().as_micros() as u64;

        // Build result
        let gradient = PhiGradient {
            phi,
            component_gradients,
            gradient_ranking,
            integration_helpers,
            integration_hinderers,
            gradient_norm,
            ascent_direction,
            compute_time_us: elapsed_us,
            precision: self.config.precision,
        };

        // Update statistics
        self.stats.total_computations += 1;
        self.stats.total_time_us += elapsed_us;
        match self.config.precision {
            GradientPrecision::Fast => self.stats.fast_computations += 1,
            GradientPrecision::Medium => self.stats.medium_computations += 1,
            GradientPrecision::High => self.stats.high_computations += 1,
        }

        // Emit event if streaming enabled and enough time has passed
        if self.config.enable_streaming {
            let min_interval = Duration::from_millis(self.config.min_update_interval_ms);
            if self.last_update.elapsed() >= min_interval {
                self.emit_event(&gradient);
                self.last_update = Instant::now();
            }
        }

        self.last_gradient = Some(gradient.clone());
        gradient
    }

    /// Fast gradient: O(n) using degree centrality from similarity matrix
    fn compute_gradient_fast(&self, components: &[BinaryHV], base_phi: f64) -> Vec<f64> {
        let n = components.len();
        if n < 2 {
            return vec![0.0; n];
        }

        // Use incremental state if available
        if let Some(ref state) = self.phi_calc.incremental_state {
            // Gradient approximation: components with high degree contribute positively
            // Components with low degree relative to average hurt integration
            let degrees = state.degrees();
            let avg_degree: f64 = degrees.iter().sum::<f64>() / n as f64;

            degrees
                .iter()
                .map(|&d| {
                    // Normalize: positive if above average, negative if below
                    let normalized = (d - avg_degree) / avg_degree.max(0.01);
                    // Scale by base_phi to make gradient meaningful
                    normalized * base_phi * 0.1
                })
                .collect()
        } else {
            // Fallback: compute pairwise similarities directly
            let avg_similarities: Vec<f64> = (0..n)
                .map(|i| {
                    let sum: f64 = (0..n)
                        .filter(|&j| i != j)
                        .map(|j| components[i].similarity(&components[j]) as f64)
                        .sum();
                    sum / (n - 1).max(1) as f64
                })
                .collect();

            let global_avg: f64 = avg_similarities.iter().sum::<f64>() / n as f64;

            avg_similarities
                .iter()
                .map(|&s| (s - global_avg) * base_phi * 0.1)
                .collect()
        }
    }

    /// Medium gradient: O(n²) using cached similarity matrix perturbation
    fn compute_gradient_medium(&self, components: &[BinaryHV], base_phi: f64) -> Vec<f64> {
        let n = components.len();
        if n < 2 {
            return vec![0.0; n];
        }

        if let Some(ref state) = self.phi_calc.incremental_state {
            // Approximate gradient using row/column removal effect on connectivity
            let sim_matrix = state.similarity_matrix();
            let total_sum: f64 = sim_matrix
                .iter()
                .map(|row: &Vec<f64>| row.iter().sum::<f64>())
                .sum();

            (0..n)
                .map(|i| {
                    // Contribution = how much this component's connections add to total
                    let row_sum: f64 = sim_matrix[i].iter().sum::<f64>();

                    // Normalized contribution
                    let contribution = row_sum / total_sum.max(0.01);

                    // Compare to uniform contribution (1/n)
                    let expected = 1.0 / n as f64;
                    let deviation = contribution - expected;

                    // Scale to meaningful gradient
                    deviation * base_phi * n as f64 * 0.5
                })
                .collect()
        } else {
            // Fall back to fast method
            self.compute_gradient_fast(components, base_phi)
        }
    }

    /// High gradient: O(n³) using numerical differentiation on Phi
    fn compute_gradient_high(&mut self, components: &[BinaryHV], base_phi: f64) -> Vec<f64> {
        let n = components.len();
        if n < 2 {
            return vec![0.0; n];
        }

        // For each component, estimate gradient by computing Φ without it
        let gradient_fn = |i: usize| {
            // Create component list without i
            let remaining: Vec<BinaryHV> = components
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, c)| *c)
                .collect();

            // Use fresh calculator to avoid state contamination
            let mut calc = TieredPhi::new(ApproximationTier::SampledPartition);
            let phi_without = calc.compute(&remaining);

            // Gradient = how much Φ drops when we remove this component
            // Positive gradient means component helps, negative means it hurts
            base_phi - phi_without
        };

        #[cfg(feature = "parallel")]
        let gradients: Vec<f64> = (0..n).into_par_iter().map(gradient_fn).collect();
        #[cfg(not(feature = "parallel"))]
        let gradients: Vec<f64> = (0..n).map(gradient_fn).collect();

        gradients
    }

    /// Emit gradient event
    fn emit_event(&mut self, gradient: &PhiGradient) {
        if let Some(ref sender) = self.event_sender {
            let phi_delta = self
                .last_gradient
                .as_ref()
                .map(|lg| gradient.phi - lg.phi)
                .unwrap_or(0.0);

            let event = GradientEvent {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                phi: gradient.phi,
                phi_delta,
                gradient_norm: gradient.gradient_norm,
                top_helpers: gradient
                    .integration_helpers
                    .iter()
                    .take(self.config.top_k_components)
                    .cloned()
                    .collect(),
                top_hinderers: gradient
                    .integration_hinderers
                    .iter()
                    .take(self.config.top_k_components)
                    .cloned()
                    .collect(),
                suggestion: gradient.optimization_suggestion(),
            };

            if sender.send(event).is_ok() {
                self.stats.events_emitted += 1;
            }
        }
    }

    /// Get last computed gradient
    pub fn last_gradient(&self) -> Option<&PhiGradient> {
        self.last_gradient.as_ref()
    }

    /// Get gradient statistics
    pub fn stats(&self) -> &GradientStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GradientStats::default();
    }

    /// Set precision level
    pub fn set_precision(&mut self, precision: GradientPrecision) {
        self.config.precision = precision;
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Compute gradient with default settings
pub fn compute_phi_gradient(components: &[BinaryHV]) -> PhiGradient {
    let mut streamer = StreamingPhiGradient::new(GradientConfig::default());
    streamer.compute_gradient(components)
}

/// Compute fast gradient for real-time use
pub fn compute_phi_gradient_fast(components: &[BinaryHV]) -> PhiGradient {
    let config = GradientConfig {
        precision: GradientPrecision::Fast,
        ..Default::default()
    };
    let mut streamer = StreamingPhiGradient::new(config);
    streamer.compute_gradient(components)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_components(n: usize) -> Vec<BinaryHV> {
        (0..n).map(|i| BinaryHV::random(i as u64)).collect()
    }

    #[test]
    fn test_gradient_basic() {
        let components = create_test_components(8);
        let gradient = compute_phi_gradient(&components);

        assert!(gradient.phi >= 0.0 && gradient.phi <= 1.0);
        assert_eq!(gradient.component_gradients.len(), 8);
        assert_eq!(gradient.gradient_ranking.len(), 8);
    }

    #[test]
    fn test_gradient_precision_levels() {
        let components = create_test_components(10);

        for precision in [
            GradientPrecision::Fast,
            GradientPrecision::Medium,
            GradientPrecision::High,
        ] {
            let config = GradientConfig {
                precision,
                ..Default::default()
            };
            let mut streamer = StreamingPhiGradient::new(config);
            let gradient = streamer.compute_gradient(&components);

            assert_eq!(gradient.precision, precision);
            assert!(gradient.gradient_norm >= 0.0);
        }
    }

    #[test]
    fn test_streaming_events() {
        let components = create_test_components(8);

        let config = GradientConfig {
            min_update_interval_ms: 0, // Immediate updates
            ..Default::default()
        };
        let mut streamer = StreamingPhiGradient::new(config);
        let receiver = streamer.enable_streaming();

        // First computation should emit event
        let _ = streamer.compute_gradient(&components);

        // Small delay to ensure event is sent
        std::thread::sleep(Duration::from_millis(10));

        // Should have received an event
        if let Ok(event) = receiver.try_recv() {
            assert!(event.phi >= 0.0);
        }
    }

    #[test]
    fn test_optimization_suggestion() {
        let components = create_test_components(8);
        let gradient = compute_phi_gradient(&components);

        // Should return some suggestion
        match gradient.optimization_suggestion() {
            OptimizationAction::Stable => {}
            OptimizationAction::ReplaceHindererWithHelper {
                hinderer,
                helper_template,
            } => {
                assert!(hinderer < 8);
                assert!(helper_template < 8);
            }
            OptimizationAction::RemoveHinderer { hinderer } => {
                assert!(hinderer < 8);
            }
        }
    }

    #[test]
    fn test_gradient_stats() {
        let components = create_test_components(5);
        let mut streamer = StreamingPhiGradient::new(GradientConfig::default());

        for _ in 0..3 {
            let _ = streamer.compute_gradient(&components);
        }

        assert_eq!(streamer.stats().total_computations, 3);
    }

    #[test]
    fn test_empty_components() {
        let gradient = compute_phi_gradient(&[]);
        assert_eq!(gradient.phi, 0.0);
        assert!(gradient.component_gradients.is_empty());
    }

    #[test]
    fn test_single_component() {
        let components = create_test_components(1);
        let gradient = compute_phi_gradient(&components);
        assert_eq!(gradient.phi, 0.0);
        assert_eq!(gradient.component_gradients.len(), 1);
    }
}
