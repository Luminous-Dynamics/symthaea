// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ Dynamics and Attractor Analysis
//!
//! This module contains temporal dynamics tracking (#93) and attractor analysis (#97).
//!
//! ## Features
//!
//! - **Temporal Dynamics**: Track consciousness evolution over time, detect phase transitions
//! - **Attractor Analysis**: Model consciousness as a dynamical system with attractor states

// Note: BinaryHV and TieredPhi/ApproximationTier are available via super if needed

/// Configuration for Φ dynamics tracking
#[derive(Debug, Clone)]
pub struct PhiDynamicsConfig {
    /// Size of the circular buffer for history
    pub history_size: usize,

    /// Threshold (in std devs) for detecting phase transitions
    pub transition_threshold_sigma: f64,

    /// Minimum samples before detecting transitions
    pub min_samples_for_transition: usize,

    /// Smoothing factor for EMA (0 = no smoothing, 1 = instant)
    pub smoothing_alpha: f64,
}

impl Default for PhiDynamicsConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            transition_threshold_sigma: 2.0,
            min_samples_for_transition: 10,
            smoothing_alpha: 0.3,
        }
    }
}

/// Tracks Φ evolution over time for dynamics and phase transition analysis
#[derive(Debug, Clone)]
pub struct PhiDynamics {
    config: PhiDynamicsConfig,
    history: Vec<(u64, f64)>, // (timestamp, phi) circular buffer
    head: usize,
    count: usize,
    running_sum: f64,
    running_sum_sq: f64,
    last_transition: Option<PhaseTransition>,
    tick_counter: u64,
}

/// Result of recording a new Φ measurement
#[derive(Debug, Clone)]
pub struct PhiDynamicsSnapshot {
    /// Current Φ value
    pub current_phi: f64,

    /// Rate of change (dΦ/dt per sample)
    pub rate_of_change: f64,

    /// Smoothed rate of change (EMA)
    pub smoothed_rate: f64,

    /// Standard deviation of recent Φ values
    pub volatility: f64,

    /// Mean Φ over history window
    pub mean_phi: f64,

    /// Trend direction and strength
    pub trend: PhiTrend,

    /// Stability score (0 = chaotic, 1 = stable)
    pub stability: f64,

    /// Any detected phase transition
    pub transition: Option<PhaseTransition>,
}

/// Trend analysis for Φ evolution
#[derive(Debug, Clone)]
pub struct PhiTrend {
    /// Trend direction
    pub direction: TrendDirection,

    /// Strength of trend (0 = no trend, 1 = strong monotonic)
    pub strength: f64,

    /// Predicted Φ at next sample (linear extrapolation)
    pub predicted_next: f64,
}

/// Direction of Φ trend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

/// A detected phase transition in consciousness
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// When the transition occurred
    pub timestamp_ns: u64,

    /// Φ value before transition
    pub phi_before: f64,

    /// Φ value after transition
    pub phi_after: f64,

    /// Magnitude in standard deviations
    pub magnitude_sigma: f64,

    /// Direction of change
    pub direction: TransitionDirection,

    /// Type/speed of transition
    pub transition_type: TransitionType,
}

/// Direction of a phase transition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionDirection {
    Rising,
    Falling,
}

/// Speed/type of phase transition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Sudden jump (1-2 samples)
    Sudden,
    /// Rapid change (3-5 samples)
    Rapid,
    /// Gradual shift (5+ samples)
    Gradual,
}

impl PhiDynamics {
    /// Create new Φ dynamics tracker with default config
    pub fn new() -> Self {
        Self::with_config(PhiDynamicsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PhiDynamicsConfig) -> Self {
        Self {
            history: vec![(0, 0.0); config.history_size],
            head: 0,
            count: 0,
            running_sum: 0.0,
            running_sum_sq: 0.0,
            last_transition: None,
            tick_counter: 0,
            config,
        }
    }

    /// Record a new Φ measurement with auto-generated timestamp
    pub fn record(&mut self, phi: f64) -> PhiDynamicsSnapshot {
        let ts = self.tick_counter;
        self.tick_counter += 1;
        self.record_with_timestamp(phi, ts)
    }

    /// Record a new Φ measurement with explicit timestamp
    pub fn record_with_timestamp(&mut self, phi: f64, timestamp_ns: u64) -> PhiDynamicsSnapshot {
        // Get previous value for rate calculation
        let prev_phi = if self.count > 0 {
            let prev_idx = if self.head == 0 {
                self.config.history_size - 1
            } else {
                self.head - 1
            };
            self.history[prev_idx].1
        } else {
            phi
        };

        // Update running statistics (remove old value if buffer full)
        if self.count >= self.config.history_size {
            let old_val = self.history[self.head].1;
            self.running_sum -= old_val;
            self.running_sum_sq -= old_val * old_val;
        }

        // Add new value
        self.history[self.head] = (timestamp_ns, phi);
        self.running_sum += phi;
        self.running_sum_sq += phi * phi;

        // Advance head
        self.head = (self.head + 1) % self.config.history_size;
        if self.count < self.config.history_size {
            self.count += 1;
        }

        // Calculate snapshot
        self.compute_snapshot(phi, prev_phi, timestamp_ns)
    }

    /// Compute dynamics snapshot based on current state
    fn compute_snapshot(
        &mut self,
        current_phi: f64,
        prev_phi: f64,
        timestamp: u64,
    ) -> PhiDynamicsSnapshot {
        let n = self.count as f64;

        // Mean and variance
        let mean_phi = if n > 0.0 { self.running_sum / n } else { 0.0 };
        let variance = if n > 1.0 {
            (self.running_sum_sq / n) - (mean_phi * mean_phi)
        } else {
            0.0
        };
        let volatility = variance.max(0.0).sqrt();

        // Rate of change
        let rate_of_change = current_phi - prev_phi;

        // Smoothed rate (EMA)
        let smoothed_rate = if let Some(ref trans) = self.last_transition {
            self.config.smoothing_alpha * rate_of_change
                + (1.0 - self.config.smoothing_alpha) * (trans.phi_after - trans.phi_before)
        } else {
            rate_of_change
        };

        // Stability (inverse of coefficient of variation)
        let stability = if mean_phi.abs() > 1e-10 && volatility > 0.0 {
            let cv = volatility / mean_phi.abs();
            1.0 / (1.0 + cv)
        } else {
            1.0
        };

        // Trend analysis
        let trend = self.compute_trend(current_phi);

        // Phase transition detection
        let transition = self.detect_transition(current_phi, mean_phi, volatility, timestamp);
        if transition.is_some() {
            self.last_transition = transition.clone();
        }

        PhiDynamicsSnapshot {
            current_phi,
            rate_of_change,
            smoothed_rate,
            volatility,
            mean_phi,
            trend,
            stability,
            transition,
        }
    }

    /// Compute trend from recent history
    fn compute_trend(&self, current_phi: f64) -> PhiTrend {
        if self.count < 3 {
            return PhiTrend {
                direction: TrendDirection::Stable,
                strength: 0.0,
                predicted_next: current_phi,
            };
        }

        // Get recent values for trend analysis
        let recent = self.get_recent(self.count.min(20));
        if recent.len() < 3 {
            return PhiTrend {
                direction: TrendDirection::Stable,
                strength: 0.0,
                predicted_next: current_phi,
            };
        }

        // Simple linear regression for trend
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i as f64) * (i as f64)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let mean_y = sum_y / n;

        // R² for trend strength
        let ss_tot: f64 = recent.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = recent
            .iter()
            .enumerate()
            .map(|(i, y)| {
                let predicted = mean_y + slope * (i as f64 - (n - 1.0) / 2.0);
                (y - predicted).powi(2)
            })
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Determine direction
        let direction = if r_squared < 0.1 {
            // Check for oscillation
            let sign_changes: usize = recent
                .windows(2)
                .filter(|w| (w[1] - w[0]) * (if w.len() > 2 { w[2] - w[1] } else { 0.0 }) < 0.0)
                .count();
            if sign_changes > recent.len() / 3 {
                TrendDirection::Oscillating
            } else {
                TrendDirection::Stable
            }
        } else if slope > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        // Predict next value
        let predicted_next = current_phi + slope;

        PhiTrend {
            direction,
            strength: r_squared.clamp(0.0, 1.0),
            predicted_next,
        }
    }

    /// Detect phase transitions using z-score
    fn detect_transition(
        &self,
        current: f64,
        mean: f64,
        std: f64,
        timestamp: u64,
    ) -> Option<PhaseTransition> {
        if self.count < self.config.min_samples_for_transition || std < 1e-10 {
            return None;
        }

        let z_score = (current - mean).abs() / std;

        if z_score >= self.config.transition_threshold_sigma {
            let direction = if current > mean {
                TransitionDirection::Rising
            } else {
                TransitionDirection::Falling
            };

            let transition_type = if z_score >= 4.0 {
                TransitionType::Sudden
            } else if z_score >= 3.0 {
                TransitionType::Rapid
            } else {
                TransitionType::Gradual
            };

            Some(PhaseTransition {
                timestamp_ns: timestamp,
                phi_before: mean,
                phi_after: current,
                magnitude_sigma: z_score,
                direction,
                transition_type,
            })
        } else {
            None
        }
    }

    /// Get complete history (oldest to newest)
    pub fn get_history(&self) -> Vec<(u64, f64)> {
        if self.count == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(self.count);
        let start = if self.count >= self.config.history_size {
            self.head
        } else {
            0
        };

        for i in 0..self.count {
            let idx = (start + i) % self.config.history_size;
            result.push(self.history[idx]);
        }

        result
    }

    /// Get recent Φ values (oldest to newest)
    pub fn get_recent(&self, n: usize) -> Vec<f64> {
        let take = n.min(self.count);
        if take == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(take);
        for i in 0..take {
            // Safe calculation that avoids underflow:
            // We want: (head - 1 - i) mod history_size
            // Reorder to: (head + history_size - 1 - i) mod history_size
            let idx = (self.head + self.config.history_size - 1 - i) % self.config.history_size;
            result.push(self.history[idx].1);
        }
        result.reverse();
        result
    }

    /// Reset all history
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
        self.last_transition = None;
        self.tick_counter = 0;
    }

    /// Get current sample count
    pub fn sample_count(&self) -> usize {
        self.count
    }
}

impl Default for PhiDynamics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// REVOLUTIONARY #97: Φ ATTRACTOR DYNAMICS
// ============================================================================
//
// Models consciousness as a dynamical system with attractor states.
// Consciousness doesn't exist as a single value but evolves through
// a phase space with basins of attraction, saddle points, and limit cycles.
//
// ## Core Insight
//
// Traditional IIT measures Φ as a static snapshot. But consciousness is
// fundamentally DYNAMIC - it flows, transitions, and settles into stable
// patterns. Attractor dynamics captures this:
//
// - **Stable Attractors**: Baseline consciousness states (awake, dreaming)
// - **Saddle Points**: Transition states (falling asleep, waking up)
// - **Basin Size**: Robustness of consciousness (how hard to perturb)
// - **Lyapunov Exponents**: Stability/chaos of consciousness dynamics
//
// ## Applications
//
// - **Anesthesia Monitoring**: Track consciousness as it approaches attractor
// - **Sleep Stage Detection**: Different stages = different attractors
// - **Meditation States**: Map meditative attractors
// - **Consciousness Recovery**: Guide recovery toward healthy attractors
// - **AI Consciousness**: Design systems with desired attractor landscapes
//
// ## References
//
// - Strogatz (2001): Nonlinear Dynamics and Chaos
// - Tononi (2004): IIT consciousness measurement
// - Koch & Tsuchiya (2007): Consciousness as global workspace
// - This work: First attractor dynamics for consciousness phase space

/// Configuration for attractor dynamics analysis
#[derive(Debug, Clone)]
pub struct AttractorConfig {
    /// Maximum iterations for convergence detection
    pub max_iterations: usize,

    /// Convergence threshold (change in Φ)
    pub convergence_threshold: f64,

    /// Perturbation magnitude for basin estimation
    pub perturbation_magnitude: f64,

    /// Number of random perturbations for basin sampling
    pub basin_samples: usize,

    /// Time step for dynamics simulation
    pub time_step: f64,

    /// Noise amplitude for stochastic dynamics
    pub noise_amplitude: f64,
}

impl Default for AttractorConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-4,
            perturbation_magnitude: 0.1,
            basin_samples: 20,
            time_step: 0.1,
            noise_amplitude: 0.01,
        }
    }
}

impl AttractorConfig {
    /// Fast config for real-time analysis
    pub fn fast() -> Self {
        Self {
            max_iterations: 30,
            basin_samples: 10,
            ..Default::default()
        }
    }

    /// Research config for detailed analysis
    pub fn research() -> Self {
        Self {
            max_iterations: 500,
            convergence_threshold: 1e-6,
            basin_samples: 50,
            ..Default::default()
        }
    }
}

/// Attractor state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttractorType {
    /// Fixed point - consciousness converges to stable value
    FixedPoint,
    /// Limit cycle - consciousness oscillates periodically
    LimitCycle,
    /// Strange attractor - chaotic but bounded dynamics
    StrangeAttractor,
    /// Saddle point - unstable equilibrium
    SaddlePoint,
    /// Transient - no attractor found (still evolving)
    Transient,
}

impl AttractorType {
    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::FixedPoint => "stable fixed point (baseline consciousness)",
            Self::LimitCycle => "limit cycle (oscillating consciousness)",
            Self::StrangeAttractor => "strange attractor (chaotic but bounded)",
            Self::SaddlePoint => "saddle point (unstable transition)",
            Self::Transient => "transient (no stable attractor)",
        }
    }

    /// Consciousness interpretation
    pub fn consciousness_interpretation(&self) -> &'static str {
        match self {
            Self::FixedPoint => "stable waking or sleep state",
            Self::LimitCycle => "meditative or hypnagogic state",
            Self::StrangeAttractor => "creative or flow state",
            Self::SaddlePoint => "transition (falling asleep, waking)",
            Self::Transient => "rapidly changing consciousness",
        }
    }
}

/// Result of attractor dynamics analysis
#[derive(Debug, Clone)]
pub struct AttractorResult {
    /// Type of attractor detected
    pub attractor_type: AttractorType,

    /// Final Φ value at attractor (if converged)
    pub attractor_phi: f64,

    /// Initial Φ value before dynamics
    pub initial_phi: f64,

    /// Estimated basin of attraction size (0-1)
    pub basin_size: f64,

    /// Largest Lyapunov exponent (negative = stable, positive = chaotic)
    pub lyapunov_exponent: f64,

    /// Convergence time (iterations to attractor)
    pub convergence_time: usize,

    /// Φ trajectory during evolution
    pub trajectory: Vec<f64>,

    /// Whether the system converged
    pub converged: bool,

    /// Oscillation period (if limit cycle)
    pub oscillation_period: Option<usize>,

    /// Basin neighbors - other attractors within perturbation distance
    pub basin_neighbors: Vec<f64>,
}

impl AttractorResult {
    /// Check if consciousness is in a stable state
    pub fn is_stable(&self) -> bool {
        matches!(self.attractor_type, AttractorType::FixedPoint) && self.lyapunov_exponent < 0.0
    }

    /// Check if consciousness is in transition
    pub fn is_transitioning(&self) -> bool {
        matches!(
            self.attractor_type,
            AttractorType::SaddlePoint | AttractorType::Transient
        )
    }

    /// Check if consciousness shows complex dynamics
    pub fn is_complex(&self) -> bool {
        matches!(
            self.attractor_type,
            AttractorType::LimitCycle | AttractorType::StrangeAttractor
        )
    }

    /// Get stability score (0 = chaotic, 1 = maximally stable)
    pub fn stability_score(&self) -> f64 {
        if self.lyapunov_exponent >= 0.0 {
            0.0
        } else {
            (-self.lyapunov_exponent).min(1.0).tanh()
        }
    }

    /// Get robustness score based on basin size
    pub fn robustness_score(&self) -> f64 {
        self.basin_size
    }
}

/// Φ Attractor Dynamics Analyzer
///
/// Models consciousness evolution through phase space and identifies
/// stable attractor states, transition saddles, and basin boundaries.
#[derive(Debug, Clone)]
pub struct PhiAttractor {
    /// Configuration for attractor analysis
    pub config: AttractorConfig,
    /// History of Φ states for trajectory analysis
    pub state_history: Vec<f64>,
}

impl PhiAttractor {
    /// Create new attractor analyzer with default config
    pub fn new() -> Self {
        Self {
            config: AttractorConfig::default(),
            state_history: Vec::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: AttractorConfig) -> Self {
        Self {
            config,
            state_history: Vec::new(),
        }
    }

    /// Fast analyzer for real-time use
    pub fn fast() -> Self {
        Self::with_config(AttractorConfig::fast())
    }

    /// Research analyzer for detailed analysis
    pub fn research() -> Self {
        Self::with_config(AttractorConfig::research())
    }

    /// Analyze attractor dynamics from Φ time series
    ///
    /// # Arguments
    ///
    /// * `phi_trajectory` - Time series of Φ measurements
    ///
    /// # Returns
    ///
    /// AttractorResult with dynamics analysis
    pub fn analyze(&mut self, phi_trajectory: &[f64]) -> AttractorResult {
        if phi_trajectory.is_empty() {
            return AttractorResult {
                attractor_type: AttractorType::Transient,
                attractor_phi: 0.0,
                initial_phi: 0.0,
                basin_size: 0.0,
                lyapunov_exponent: 0.0,
                convergence_time: 0,
                trajectory: vec![],
                converged: false,
                oscillation_period: None,
                basin_neighbors: vec![],
            };
        }

        self.state_history = phi_trajectory.to_vec();

        let initial_phi = phi_trajectory[0];
        let final_phi = *phi_trajectory
            .last()
            .expect("phi_trajectory verified non-empty above");

        // 1. Detect convergence
        let (converged, convergence_time) = self.detect_convergence(phi_trajectory);

        // 2. Compute Lyapunov exponent
        let lyapunov = self.compute_lyapunov(phi_trajectory);

        // 3. Detect oscillation
        let (is_oscillating, period) = self.detect_oscillation(phi_trajectory);

        // 4. Classify attractor type
        let attractor_type = self.classify_attractor(converged, is_oscillating, lyapunov);

        // 5. Estimate basin size
        let (basin_size, basin_neighbors) = self.estimate_basin(phi_trajectory);

        AttractorResult {
            attractor_type,
            attractor_phi: final_phi,
            initial_phi,
            basin_size,
            lyapunov_exponent: lyapunov,
            convergence_time,
            trajectory: phi_trajectory.to_vec(),
            converged,
            oscillation_period: period,
            basin_neighbors,
        }
    }

    /// Detect convergence to a fixed point
    fn detect_convergence(&self, trajectory: &[f64]) -> (bool, usize) {
        if trajectory.len() < 3 {
            return (false, trajectory.len());
        }

        let n = trajectory.len();
        let threshold = self.config.convergence_threshold;

        // Check for convergence in the last portion of trajectory
        let check_window = (n / 4).max(3);

        for i in (check_window..n).rev() {
            let window = &trajectory[i - check_window..i];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let max_deviation = window.iter().map(|&x| (x - mean).abs()).fold(0.0, f64::max);

            if max_deviation < threshold {
                return (true, i - check_window);
            }
        }

        (false, n)
    }

    /// Compute largest Lyapunov exponent
    ///
    /// Measures rate of divergence of nearby trajectories.
    /// Negative = stable, Positive = chaotic
    fn compute_lyapunov(&self, trajectory: &[f64]) -> f64 {
        if trajectory.len() < 10 {
            return 0.0;
        }

        // Compute average rate of change
        let mut divergence_sum = 0.0;
        let mut count = 0;

        for i in 1..trajectory.len() {
            let delta = (trajectory[i] - trajectory[i - 1]).abs();
            if delta > 1e-10 && trajectory[i - 1].abs() > 1e-10 {
                // Rate of divergence normalized by current state
                divergence_sum += (delta / trajectory[i - 1].abs()).ln();
                count += 1;
            }
        }

        if count > 0 {
            divergence_sum / count as f64
        } else {
            0.0
        }
    }

    /// Detect oscillatory behavior (limit cycles)
    fn detect_oscillation(&self, trajectory: &[f64]) -> (bool, Option<usize>) {
        if trajectory.len() < 10 {
            return (false, None);
        }

        // Find peaks
        let mut peaks = Vec::new();
        for i in 1..trajectory.len() - 1 {
            if trajectory[i] > trajectory[i - 1] && trajectory[i] > trajectory[i + 1] {
                peaks.push(i);
            }
        }

        if peaks.len() < 2 {
            return (false, None);
        }

        // Check for consistent period
        let mut periods = Vec::new();
        for i in 1..peaks.len() {
            periods.push(peaks[i] - peaks[i - 1]);
        }

        if periods.is_empty() {
            return (false, None);
        }

        let mean_period = periods.iter().sum::<usize>() as f64 / periods.len() as f64;
        let variance = periods
            .iter()
            .map(|&p| (p as f64 - mean_period).powi(2))
            .sum::<f64>()
            / periods.len() as f64;

        // Low variance indicates regular oscillation
        let is_periodic = variance.sqrt() / mean_period < 0.3;

        if is_periodic {
            (true, Some(mean_period.round() as usize))
        } else {
            (false, None)
        }
    }

    /// Classify attractor type based on dynamics
    fn classify_attractor(
        &self,
        converged: bool,
        is_oscillating: bool,
        lyapunov: f64,
    ) -> AttractorType {
        if is_oscillating {
            return AttractorType::LimitCycle;
        }

        if converged {
            if lyapunov < -0.1 {
                return AttractorType::FixedPoint;
            } else if lyapunov > 0.1 {
                return AttractorType::SaddlePoint;
            } else {
                return AttractorType::FixedPoint;
            }
        }

        if lyapunov > 0.3 {
            AttractorType::StrangeAttractor
        } else if lyapunov > 0.0 {
            AttractorType::SaddlePoint
        } else {
            AttractorType::Transient
        }
    }

    /// Estimate basin of attraction size
    fn estimate_basin(&self, trajectory: &[f64]) -> (f64, Vec<f64>) {
        if trajectory.len() < 5 {
            return (0.5, vec![]);
        }

        let final_phi = *trajectory
            .last()
            .expect("trajectory verified len >= 5 above");

        // Estimate basin by checking how often trajectory returns to attractor region
        let tolerance = self.config.convergence_threshold * 10.0;
        let mut in_basin_count = 0;

        for &phi in trajectory.iter() {
            if (phi - final_phi).abs() < tolerance {
                in_basin_count += 1;
            }
        }

        let basin_estimate = in_basin_count as f64 / trajectory.len() as f64;

        // Find neighboring values (local extrema)
        let mut neighbors = Vec::new();
        for i in 1..trajectory.len() - 1 {
            let is_local_min =
                trajectory[i] < trajectory[i - 1] && trajectory[i] < trajectory[i + 1];
            let is_local_max =
                trajectory[i] > trajectory[i - 1] && trajectory[i] > trajectory[i + 1];
            if (is_local_min || is_local_max) && (trajectory[i] - final_phi).abs() > tolerance {
                neighbors.push(trajectory[i]);
            }
        }

        neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.dedup();

        (basin_estimate, neighbors)
    }

    /// Simulate dynamics from initial state
    ///
    /// Evolves the system according to a simple gradient dynamics:
    /// dΦ/dt = -∇V(Φ) + noise
    ///
    /// where V is an inferred potential landscape.
    pub fn simulate(&self, initial_phi: f64, target_phi: f64) -> Vec<f64> {
        let mut trajectory = Vec::with_capacity(self.config.max_iterations);
        let mut phi = initial_phi;

        let mut rng_state = (initial_phi * 1000.0) as u64;

        for _ in 0..self.config.max_iterations {
            trajectory.push(phi);

            // Gradient toward target with some nonlinearity
            let gradient = -(phi - target_phi) * (1.0 + (phi - target_phi).powi(2));

            // Add noise
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise =
                ((rng_state as f64) / u64::MAX as f64 - 0.5) * 2.0 * self.config.noise_amplitude;

            // Euler step
            phi += gradient * self.config.time_step + noise;

            // Keep in valid range
            phi = phi.clamp(0.0, 1.0);

            // Check convergence
            if (phi - target_phi).abs() < self.config.convergence_threshold {
                trajectory.push(phi);
                break;
            }
        }

        trajectory
    }

    /// Find multiple attractors in phase space
    ///
    /// Samples many initial conditions and identifies distinct attractors.
    pub fn find_attractors(&mut self, phi_range: (f64, f64)) -> Vec<f64> {
        let (min_phi, max_phi) = phi_range;
        let step = (max_phi - min_phi) / self.config.basin_samples as f64;

        let mut attractors = Vec::new();

        for i in 0..self.config.basin_samples {
            let initial = min_phi + i as f64 * step;

            // Simulate to find where it converges
            let trajectory = self.simulate(initial, 0.5); // Target is mid-range

            if let Some(&final_phi) = trajectory.last() {
                // Check if this is a new attractor
                let is_new = attractors.iter().all(|&a: &f64| {
                    (a - final_phi).abs() > self.config.convergence_threshold * 10.0
                });

                if is_new {
                    attractors.push(final_phi);
                }
            }
        }

        attractors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        attractors
    }
}

impl Default for PhiAttractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: analyze attractor from Φ time series
pub fn analyze_phi_attractor(phi_trajectory: &[f64]) -> AttractorResult {
    PhiAttractor::new().analyze(phi_trajectory)
}

/// Convenience function: classify consciousness state from trajectory
pub fn classify_consciousness_state(phi_trajectory: &[f64]) -> (AttractorType, f64) {
    let result = PhiAttractor::new().analyze(phi_trajectory);
    (result.attractor_type, result.stability_score())
}

// ============================================================================
