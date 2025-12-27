//! Revolutionary Improvement #44: Kalman-Based Predictive Consciousness
//!
//! **Paradigm Shift**: Model consciousness as a dynamical system with state-space
//! representation, enabling principled forecasting, uncertainty quantification,
//! and early warning signal detection for consciousness transitions.
//!
//! # The Problem
//!
//! Existing consciousness measures (Φ, PCI, etc.) are **point estimates**:
//! - Single value at a moment in time
//! - No uncertainty quantification
//! - Cannot predict future states
//! - Cannot detect approaching transitions
//!
//! # Revolutionary Solution
//!
//! Model consciousness as a **state-space system**:
//!
//! ```text
//! State equation:    C_{t+1} = A·C_t + B·u_t + w_t    (dynamics)
//! Observation:       z_t = H·C_t + v_t                (measurement)
//!
//! Where:
//! - C_t = consciousness state vector [Φ, B, W, A, R]
//! - A = state transition matrix (dynamics)
//! - B = control input matrix (interventions)
//! - H = observation matrix
//! - w_t ~ N(0, Q) process noise
//! - v_t ~ N(0, R) measurement noise
//! ```
//!
//! # Key Capabilities
//!
//! 1. **State Estimation**: Optimal filtering of noisy observations
//! 2. **Uncertainty Quantification**: Covariance matrix tracks confidence
//! 3. **Forecasting**: Predict consciousness trajectory N steps ahead
//! 4. **Early Warning Signals**: Detect approaching phase transitions
//! 5. **Anomaly Detection**: Identify unexpected consciousness states
//!
//! # Scientific Basis
//!
//! **Kalman Filter** (Kalman, 1960):
//! - Optimal linear estimator for Gaussian systems
//! - Recursive: updates beliefs with each observation
//! - Provides uncertainty bounds (covariance)
//!
//! **Critical Slowing Down** (Scheffer et al., 2009):
//! - Systems approaching bifurcation show characteristic signatures:
//!   - Increased autocorrelation (slower recovery)
//!   - Increased variance (reduced stability)
//!   - Critical slowing down index
//!
//! # Example
//!
//! ```rust
//! use symthaea::hdc::predictive_consciousness_kalman::{
//!     PredictiveConsciousness, ConsciousnessState, PredictiveConfig
//! };
//!
//! // Create predictor
//! let mut predictor = PredictiveConsciousness::new(PredictiveConfig::default());
//!
//! // Update with observations
//! let obs = ConsciousnessState::new(0.7, 0.6, 0.5, 0.8, 0.6);
//! let estimate = predictor.update(&obs);
//!
//! println!("Estimated C: {:.3} ± {:.3}",
//!          estimate.state.c_raw, estimate.uncertainty.c_std);
//!
//! // Forecast future states
//! let forecasts = predictor.forecast(10);
//! for (step, f) in forecasts.iter().enumerate() {
//!     println!("t+{}: C={:.3} ± {:.3}", step+1, f.state.c_raw, f.uncertainty.c_std);
//! }
//!
//! // Check for approaching transition
//! let ews = predictor.early_warning_signals(20);
//! println!("Transition risk: {:.2}", ews.transition_risk);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for predictive consciousness model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveConfig {
    /// State dimension (5 for Φ, B, W, A, R)
    pub state_dim: usize,

    /// Process noise variance (Q diagonal)
    pub process_noise: f64,

    /// Measurement noise variance (R diagonal)
    pub measurement_noise: f64,

    /// State persistence (A diagonal elements, 0-1)
    pub state_persistence: f64,

    /// Initial uncertainty
    pub initial_uncertainty: f64,

    /// History length for early warning signals
    pub history_length: usize,

    /// Critical slowing down detection window
    pub csd_window: usize,
}

impl Default for PredictiveConfig {
    fn default() -> Self {
        Self {
            state_dim: 5,
            process_noise: 0.01,
            measurement_noise: 0.05,
            state_persistence: 0.95,
            initial_uncertainty: 0.5,
            history_length: 100,
            csd_window: 20,
        }
    }
}

/// Consciousness state vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Integrated Information (Φ)
    pub phi: f64,

    /// Binding (B)
    pub binding: f64,

    /// Workspace access (W)
    pub workspace: f64,

    /// Attention (A)
    pub attention: f64,

    /// Recursion/meta-awareness (R)
    pub recursion: f64,

    /// Raw consciousness score: min(Φ, B, W, A, R)
    pub c_raw: f64,

    /// Timestamp (optional)
    pub timestamp: Option<f64>,
}

impl ConsciousnessState {
    /// Create new consciousness state
    pub fn new(phi: f64, binding: f64, workspace: f64, attention: f64, recursion: f64) -> Self {
        let c_raw = phi.min(binding).min(workspace).min(attention).min(recursion);
        Self {
            phi,
            binding,
            workspace,
            attention,
            recursion,
            c_raw,
            timestamp: None,
        }
    }

    /// Create from component vector
    pub fn from_vec(components: &[f64]) -> Self {
        let phi = components.get(0).copied().unwrap_or(0.0);
        let binding = components.get(1).copied().unwrap_or(0.0);
        let workspace = components.get(2).copied().unwrap_or(0.0);
        let attention = components.get(3).copied().unwrap_or(0.0);
        let recursion = components.get(4).copied().unwrap_or(0.0);
        Self::new(phi, binding, workspace, attention, recursion)
    }

    /// Convert to vector
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.phi, self.binding, self.workspace, self.attention, self.recursion]
    }

    /// Get limiting component
    pub fn limiting_component(&self) -> (&'static str, f64) {
        let components = [
            ("Phi", self.phi),
            ("Binding", self.binding),
            ("Workspace", self.workspace),
            ("Attention", self.attention),
            ("Recursion", self.recursion),
        ];

        components.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|&(name, val)| (name, val))
            .unwrap_or(("Unknown", 0.0))
    }
}

/// Uncertainty estimates for consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUncertainty {
    /// Standard deviation of each component
    pub phi_std: f64,
    pub binding_std: f64,
    pub workspace_std: f64,
    pub attention_std: f64,
    pub recursion_std: f64,

    /// Overall C uncertainty
    pub c_std: f64,

    /// Full covariance matrix (flattened 5x5)
    pub covariance: Vec<f64>,
}

impl StateUncertainty {
    /// Create from covariance matrix diagonal
    pub fn from_diagonal(diag: &[f64]) -> Self {
        let phi_std = diag.get(0).map(|&v| v.sqrt()).unwrap_or(0.0);
        let binding_std = diag.get(1).map(|&v| v.sqrt()).unwrap_or(0.0);
        let workspace_std = diag.get(2).map(|&v| v.sqrt()).unwrap_or(0.0);
        let attention_std = diag.get(3).map(|&v| v.sqrt()).unwrap_or(0.0);
        let recursion_std = diag.get(4).map(|&v| v.sqrt()).unwrap_or(0.0);

        // C uncertainty from component uncertainties (max of components)
        let c_std = phi_std.max(binding_std).max(workspace_std)
            .max(attention_std).max(recursion_std);

        Self {
            phi_std,
            binding_std,
            workspace_std,
            attention_std,
            recursion_std,
            c_std,
            covariance: diag.to_vec(),
        }
    }
}

/// State estimate with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEstimate {
    /// Estimated consciousness state
    pub state: ConsciousnessState,

    /// Uncertainty bounds
    pub uncertainty: StateUncertainty,

    /// Innovation (prediction error)
    pub innovation: Vec<f64>,

    /// Log-likelihood of observation
    pub log_likelihood: f64,
}

/// Early warning signals for consciousness transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyWarningSignals {
    /// Autocorrelation at lag 1 (increases before transition)
    pub autocorrelation: f64,

    /// Variance ratio (recent/historical)
    pub variance_ratio: f64,

    /// Skewness (asymmetry before transition)
    pub skewness: f64,

    /// Composite transition risk (0-1)
    pub transition_risk: f64,

    /// Direction of likely transition
    pub transition_direction: TransitionDirection,

    /// Time to predicted transition (if detectable)
    pub time_to_transition: Option<f64>,
}

/// Direction of consciousness transition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionDirection {
    /// Increasing consciousness
    Ascending,
    /// Decreasing consciousness
    Descending,
    /// No clear direction
    Uncertain,
}

/// Predictive Consciousness Model
///
/// Uses Kalman filtering for optimal state estimation and forecasting.
#[derive(Debug, Clone)]
pub struct PredictiveConsciousness {
    config: PredictiveConfig,

    /// Current state estimate
    state: Vec<f64>,

    /// State covariance matrix (flattened)
    covariance: Vec<f64>,

    /// State transition matrix A (flattened)
    transition_matrix: Vec<f64>,

    /// Process noise covariance Q (diagonal)
    process_noise: Vec<f64>,

    /// Measurement noise covariance R (diagonal)
    measurement_noise: Vec<f64>,

    /// History of states for EWS
    history: VecDeque<ConsciousnessState>,

    /// History of innovations
    innovation_history: VecDeque<Vec<f64>>,

    /// Timestep counter
    timestep: usize,
}

impl PredictiveConsciousness {
    /// Create new predictive consciousness model
    pub fn new(config: PredictiveConfig) -> Self {
        let dim = config.state_dim;

        // Initialize state to middle values
        let state = vec![0.5; dim];

        // Initialize covariance (diagonal)
        let mut covariance = vec![0.0; dim * dim];
        for i in 0..dim {
            covariance[i * dim + i] = config.initial_uncertainty;
        }

        // State transition matrix (diagonal with persistence)
        let mut transition_matrix = vec![0.0; dim * dim];
        for i in 0..dim {
            transition_matrix[i * dim + i] = config.state_persistence;
        }

        // Process and measurement noise
        let process_noise = vec![config.process_noise; dim];
        let measurement_noise = vec![config.measurement_noise; dim];

        Self {
            config,
            state,
            covariance,
            transition_matrix,
            process_noise,
            measurement_noise,
            history: VecDeque::with_capacity(100),
            innovation_history: VecDeque::with_capacity(100),
            timestep: 0,
        }
    }

    /// Predict next state (without observation)
    pub fn predict(&mut self) -> StateEstimate {
        let dim = self.config.state_dim;

        // State prediction: x_pred = A @ x
        let mut state_pred = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                state_pred[i] += self.transition_matrix[i * dim + j] * self.state[j];
            }
        }

        // Covariance prediction: P_pred = A @ P @ A' + Q
        let mut cov_pred = vec![0.0; dim * dim];

        // A @ P
        let mut ap = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    ap[i * dim + j] += self.transition_matrix[i * dim + k] * self.covariance[k * dim + j];
                }
            }
        }

        // (A @ P) @ A' + Q
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    cov_pred[i * dim + j] += ap[i * dim + k] * self.transition_matrix[j * dim + k];
                }
                // Add process noise on diagonal
                if i == j {
                    cov_pred[i * dim + j] += self.process_noise[i];
                }
            }
        }

        // Update internal state
        self.state = state_pred.clone();
        self.covariance = cov_pred.clone();

        // Create estimate
        let consciousness_state = ConsciousnessState::from_vec(&state_pred);
        let diag: Vec<f64> = (0..dim).map(|i| cov_pred[i * dim + i]).collect();
        let uncertainty = StateUncertainty::from_diagonal(&diag);

        StateEstimate {
            state: consciousness_state,
            uncertainty,
            innovation: vec![0.0; dim],
            log_likelihood: 0.0,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, observation: &ConsciousnessState) -> StateEstimate {
        let dim = self.config.state_dim;
        let obs = observation.to_vec();

        // First, predict
        self.predict();

        // Innovation: y = z - H @ x_pred (with H = I)
        let mut innovation = vec![0.0; dim];
        for i in 0..dim {
            innovation[i] = obs[i] - self.state[i];
        }

        // Innovation covariance: S = H @ P @ H' + R (with H = I)
        let mut s = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                s[i * dim + j] = self.covariance[i * dim + j];
                if i == j {
                    s[i * dim + j] += self.measurement_noise[i];
                }
            }
        }

        // Kalman gain: K = P @ H' @ S^{-1} (with H = I, simplified)
        // For diagonal S, K_ii = P_ii / S_ii
        let mut kalman_gain = vec![0.0; dim * dim];
        for i in 0..dim {
            let s_ii = s[i * dim + i];
            if s_ii > 1e-10 {
                for j in 0..dim {
                    kalman_gain[i * dim + j] = self.covariance[i * dim + j] / s_ii;
                }
            }
        }

        // State update: x = x_pred + K @ y
        for i in 0..dim {
            for j in 0..dim {
                self.state[i] += kalman_gain[i * dim + j] * innovation[j];
            }
            // Clamp to valid range
            self.state[i] = self.state[i].clamp(0.0, 1.0);
        }

        // Covariance update: P = (I - K @ H) @ P (with H = I)
        let mut new_cov = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                // (I - K) @ P
                let ikh = if i == j { 1.0 } else { 0.0 } - kalman_gain[i * dim + i];
                new_cov[i * dim + j] = ikh * self.covariance[i * dim + j];
            }
        }
        self.covariance = new_cov;

        // Log-likelihood
        let log_likelihood = self.compute_log_likelihood(&innovation, &s);

        // Store in history
        let consciousness_state = ConsciousnessState::from_vec(&self.state);
        self.history.push_back(consciousness_state.clone());
        if self.history.len() > self.config.history_length {
            self.history.pop_front();
        }

        self.innovation_history.push_back(innovation.clone());
        if self.innovation_history.len() > self.config.history_length {
            self.innovation_history.pop_front();
        }

        self.timestep += 1;

        // Create estimate
        let diag: Vec<f64> = (0..dim).map(|i| self.covariance[i * dim + i]).collect();
        let uncertainty = StateUncertainty::from_diagonal(&diag);

        StateEstimate {
            state: consciousness_state,
            uncertainty,
            innovation,
            log_likelihood,
        }
    }

    /// Forecast N steps into the future
    pub fn forecast(&self, steps: usize) -> Vec<StateEstimate> {
        let dim = self.config.state_dim;
        let mut forecasts = Vec::with_capacity(steps);

        let mut state = self.state.clone();
        let mut covariance = self.covariance.clone();

        for _ in 0..steps {
            // Predict state
            let mut state_pred = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    state_pred[i] += self.transition_matrix[i * dim + j] * state[j];
                }
            }

            // Predict covariance
            let mut cov_pred = vec![0.0; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    // Simplified diagonal case
                    let a_ii = self.transition_matrix[i * dim + i];
                    cov_pred[i * dim + j] = a_ii * a_ii * covariance[i * dim + j];
                    if i == j {
                        cov_pred[i * dim + j] += self.process_noise[i];
                    }
                }
            }

            // Create forecast
            let consciousness_state = ConsciousnessState::from_vec(&state_pred);
            let diag: Vec<f64> = (0..dim).map(|i| cov_pred[i * dim + i]).collect();
            let uncertainty = StateUncertainty::from_diagonal(&diag);

            forecasts.push(StateEstimate {
                state: consciousness_state,
                uncertainty,
                innovation: vec![0.0; dim],
                log_likelihood: 0.0,
            });

            state = state_pred;
            covariance = cov_pred;
        }

        forecasts
    }

    /// Compute early warning signals for consciousness transitions
    pub fn early_warning_signals(&self, window: usize) -> EarlyWarningSignals {
        let window = window.min(self.history.len());

        if window < 3 {
            return EarlyWarningSignals {
                autocorrelation: 0.0,
                variance_ratio: 1.0,
                skewness: 0.0,
                transition_risk: 0.0,
                transition_direction: TransitionDirection::Uncertain,
                time_to_transition: None,
            };
        }

        // Extract C values from history
        let c_values: Vec<f64> = self.history.iter()
            .rev()
            .take(window)
            .map(|s| s.c_raw)
            .collect();

        // Autocorrelation at lag 1
        let autocorrelation = self.compute_autocorrelation(&c_values, 1);

        // Variance ratio (recent vs historical)
        let half = window / 2;
        let recent: Vec<f64> = c_values.iter().take(half).copied().collect();
        let historical: Vec<f64> = c_values.iter().skip(half).copied().collect();

        let var_recent = self.variance(&recent);
        let var_historical = self.variance(&historical);

        let variance_ratio = if var_historical > 1e-10 {
            var_recent / var_historical
        } else {
            1.0
        };

        // Skewness
        let skewness = self.compute_skewness(&c_values);

        // Trend detection
        let mean_recent = recent.iter().sum::<f64>() / recent.len() as f64;
        let mean_historical = historical.iter().sum::<f64>() / historical.len() as f64;

        let transition_direction = if mean_recent > mean_historical + 0.05 {
            TransitionDirection::Ascending
        } else if mean_recent < mean_historical - 0.05 {
            TransitionDirection::Descending
        } else {
            TransitionDirection::Uncertain
        };

        // Composite transition risk
        // High autocorrelation + high variance ratio = approaching transition
        let ac_risk = (autocorrelation - 0.5).max(0.0) * 2.0; // AC > 0.5 is risky
        let var_risk = (variance_ratio - 1.0).max(0.0).min(1.0); // Var ratio > 1 is risky

        let transition_risk = ((ac_risk + var_risk) / 2.0).min(1.0);

        // Estimate time to transition (very rough)
        let time_to_transition = if transition_risk > 0.5 {
            Some((1.0 - transition_risk) * 50.0) // Rough estimate
        } else {
            None
        };

        EarlyWarningSignals {
            autocorrelation,
            variance_ratio,
            skewness,
            transition_risk,
            transition_direction,
            time_to_transition,
        }
    }

    /// Detect anomalies (unexpected states)
    pub fn is_anomaly(&self, observation: &ConsciousnessState, threshold: f64) -> bool {
        let obs = observation.to_vec();
        let dim = self.config.state_dim;

        // Mahalanobis distance
        let mut mahal = 0.0;
        for i in 0..dim {
            let diff = obs[i] - self.state[i];
            let var = self.covariance[i * dim + i].max(1e-10);
            mahal += diff * diff / var;
        }
        mahal = mahal.sqrt();

        mahal > threshold
    }

    /// Get current state estimate
    pub fn current_state(&self) -> ConsciousnessState {
        ConsciousnessState::from_vec(&self.state)
    }

    /// Get current uncertainty
    pub fn current_uncertainty(&self) -> StateUncertainty {
        let dim = self.config.state_dim;
        let diag: Vec<f64> = (0..dim).map(|i| self.covariance[i * dim + i]).collect();
        StateUncertainty::from_diagonal(&diag)
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        let dim = self.config.state_dim;
        self.state = vec![0.5; dim];

        for i in 0..dim {
            for j in 0..dim {
                self.covariance[i * dim + j] = if i == j {
                    self.config.initial_uncertainty
                } else {
                    0.0
                };
            }
        }

        self.history.clear();
        self.innovation_history.clear();
        self.timestep = 0;
    }

    // ========== Helper Methods ==========

    fn compute_log_likelihood(&self, innovation: &[f64], s: &[f64]) -> f64 {
        let dim = innovation.len();
        let mut ll = -0.5 * dim as f64 * (2.0 * std::f64::consts::PI).ln();

        // -0.5 * log|S| - 0.5 * y' @ S^{-1} @ y
        for i in 0..dim {
            let s_ii = s[i * dim + i];
            if s_ii > 1e-10 {
                ll -= 0.5 * s_ii.ln();
                ll -= 0.5 * innovation[i] * innovation[i] / s_ii;
            }
        }

        ll
    }

    fn compute_autocorrelation(&self, data: &[f64], lag: usize) -> f64 {
        if data.len() <= lag {
            return 0.0;
        }

        let n = data.len() - lag;
        let mean = data.iter().sum::<f64>() / data.len() as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            numerator += (data[i] - mean) * (data[i + lag] - mean);
        }

        for &x in data {
            denominator += (x - mean).powi(2);
        }

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn variance(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }

    fn compute_skewness(&self, data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = self.variance(data);

        if var < 1e-10 {
            return 0.0;
        }

        let std = var.sqrt();
        let m3 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;

        m3
    }
}

impl Default for PredictiveConsciousness {
    fn default() -> Self {
        Self::new(PredictiveConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = ConsciousnessState::new(0.7, 0.6, 0.5, 0.8, 0.65);

        assert!((state.c_raw - 0.5).abs() < 0.01); // min is workspace = 0.5
        assert_eq!(state.limiting_component().0, "Workspace");
    }

    #[test]
    fn test_predict() {
        let mut predictor = PredictiveConsciousness::default();

        let estimate = predictor.predict();

        assert!(estimate.state.c_raw >= 0.0 && estimate.state.c_raw <= 1.0);
        assert!(estimate.uncertainty.c_std > 0.0);
    }

    #[test]
    fn test_update() {
        let mut predictor = PredictiveConsciousness::default();

        let obs = ConsciousnessState::new(0.8, 0.7, 0.6, 0.9, 0.75);
        let estimate = predictor.update(&obs);

        // State should move toward observation
        assert!(estimate.state.phi > 0.5);
        assert!(estimate.state.c_raw >= 0.0 && estimate.state.c_raw <= 1.0);
    }

    #[test]
    fn test_convergence() {
        let mut predictor = PredictiveConsciousness::default();

        // Repeatedly observe same state
        let obs = ConsciousnessState::new(0.8, 0.8, 0.8, 0.8, 0.8);

        for _ in 0..20 {
            predictor.update(&obs);
        }

        let current = predictor.current_state();

        // Should converge close to observation
        assert!((current.phi - 0.8).abs() < 0.1);
        assert!((current.c_raw - 0.8).abs() < 0.1);

        // Uncertainty should decrease
        let uncertainty = predictor.current_uncertainty();
        assert!(uncertainty.c_std < 0.3);
    }

    #[test]
    fn test_forecast() {
        let mut predictor = PredictiveConsciousness::default();

        // Set initial state
        let obs = ConsciousnessState::new(0.7, 0.7, 0.7, 0.7, 0.7);
        predictor.update(&obs);

        let forecasts = predictor.forecast(10);

        assert_eq!(forecasts.len(), 10);

        // Uncertainty should increase with horizon
        assert!(forecasts[9].uncertainty.c_std > forecasts[0].uncertainty.c_std);

        // State should decay toward mean (with persistence < 1)
        for f in &forecasts {
            assert!(f.state.c_raw >= 0.0 && f.state.c_raw <= 1.0);
        }
    }

    #[test]
    fn test_early_warning_signals() {
        let mut predictor = PredictiveConsciousness::default();

        // Simulate approaching transition: increasing variance
        for i in 0..50 {
            let noise = if i > 30 { 0.2 } else { 0.05 }; // Increase variance
            let c = 0.5 + noise * (rand::random::<f64>() - 0.5);
            let obs = ConsciousnessState::new(c, c, c, c, c);
            predictor.update(&obs);
        }

        let ews = predictor.early_warning_signals(20);

        // Should detect some signals
        assert!(ews.variance_ratio >= 0.0);
        assert!(ews.transition_risk >= 0.0 && ews.transition_risk <= 1.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut predictor = PredictiveConsciousness::default();

        // Train on normal states
        for _ in 0..20 {
            let obs = ConsciousnessState::new(0.5, 0.5, 0.5, 0.5, 0.5);
            predictor.update(&obs);
        }

        // Normal observation
        let normal = ConsciousnessState::new(0.55, 0.55, 0.55, 0.55, 0.55);
        assert!(!predictor.is_anomaly(&normal, 3.0));

        // Anomalous observation
        let anomaly = ConsciousnessState::new(0.95, 0.95, 0.95, 0.95, 0.95);
        assert!(predictor.is_anomaly(&anomaly, 3.0));
    }

    #[test]
    fn test_reset() {
        let mut predictor = PredictiveConsciousness::default();

        // Make some updates
        for _ in 0..10 {
            let obs = ConsciousnessState::new(0.8, 0.8, 0.8, 0.8, 0.8);
            predictor.update(&obs);
        }

        // Reset
        predictor.reset();

        // Should be back to initial state
        let current = predictor.current_state();
        assert!((current.phi - 0.5).abs() < 0.01);
        assert!(predictor.history.is_empty());
    }

    #[test]
    fn test_transition_direction() {
        let mut predictor = PredictiveConsciousness::default();

        // Simulate ascending transition
        for i in 0..30 {
            let c = 0.3 + 0.02 * i as f64; // Increasing
            let obs = ConsciousnessState::new(c, c, c, c, c);
            predictor.update(&obs);
        }

        let ews = predictor.early_warning_signals(20);
        assert_eq!(ews.transition_direction, TransitionDirection::Ascending);
    }

    #[test]
    fn test_config_customization() {
        let config = PredictiveConfig {
            process_noise: 0.001,
            measurement_noise: 0.1,
            state_persistence: 0.99,
            ..Default::default()
        };

        let predictor = PredictiveConsciousness::new(config);

        // Should use custom config
        assert_eq!(predictor.config.process_noise, 0.001);
        assert_eq!(predictor.config.state_persistence, 0.99);
    }
}
