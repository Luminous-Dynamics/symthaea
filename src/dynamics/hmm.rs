// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hidden Markov Model for Sequential State Classification
//!
//! Provides a complete Hidden Markov Model (HMM) implementation for temporal
//! sequence classification. Primary application: smoothing consciousness state
//! predictions (sleep staging, anesthesia depth, meditation states) by enforcing
//! biologically plausible transition constraints.
//!
//! ## Algorithms
//!
//! - **Forward algorithm**: P(observations | model) — likelihood computation
//! - **Backward algorithm**: Posterior state probabilities
//! - **Viterbi decoding**: Most likely state sequence (dynamic programming)
//! - **Baum-Welch**: Unsupervised parameter estimation (EM algorithm)
//!
//! ## Consciousness Connection
//!
//! Consciousness states (Wake, N1, N2, N3, REM) follow structured transitions.
//! A person doesn't jump from deep sleep (N3) directly to REM without passing
//! through lighter stages. The HMM encodes these constraints as transition
//! probabilities, dramatically improving per-epoch classification accuracy.
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::dynamics::hmm::{HiddenMarkovModel, HmmConfig};
//!
//! // 5-state sleep staging HMM
//! let hmm = HiddenMarkovModel::sleep_staging();
//!
//! // Raw per-epoch observation likelihoods from spectral classifier
//! let observations: Vec<Vec<f64>> = epoch_likelihoods();
//!
//! // Viterbi decoding: most likely state sequence
//! let states = hmm.viterbi(&observations);
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// HMM configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmConfig {
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of observation symbols (for discrete HMM) or observation
    /// dimensions (for continuous HMM with Gaussian emissions).
    pub n_observations: usize,
    /// Convergence threshold for Baum-Welch (log-likelihood change).
    pub convergence_threshold: f64,
    /// Maximum Baum-Welch iterations.
    pub max_iterations: usize,
    /// Minimum probability floor to prevent log(0).
    pub prob_floor: f64,
}

impl Default for HmmConfig {
    fn default() -> Self {
        Self {
            n_states: 5,
            n_observations: 5,
            convergence_threshold: 1e-6,
            max_iterations: 100,
            prob_floor: 1e-300,
        }
    }
}

// ---------------------------------------------------------------------------
// Hidden Markov Model
// ---------------------------------------------------------------------------

/// A Hidden Markov Model with continuous (Gaussian mixture) or direct
/// observation-probability emissions.
///
/// The model is parameterized by:
/// - **π** (initial): Prior probability of starting in each state
/// - **A** (transition): `A[i][j]` = P(state_j at t+1 | state_i at t)
/// - **B** (emission): Observation likelihood given state (provided externally
///   or computed from Gaussian parameters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenMarkovModel {
    /// Configuration.
    pub config: HmmConfig,
    /// Initial state distribution π[i] = P(state_0 = i).
    /// Length: n_states.
    pub initial: Vec<f64>,
    /// State transition matrix A[i][j] = P(state_{t+1} = j | state_t = i).
    /// Dimensions: n_states × n_states. Each row sums to 1.
    pub transition: Vec<Vec<f64>>,
    /// State names (optional, for display).
    pub state_names: Vec<String>,
}

impl HiddenMarkovModel {
    /// Create a new HMM with uniform initialization.
    pub fn new(config: HmmConfig) -> Self {
        let n = config.n_states;
        let uniform = 1.0 / n as f64;
        Self {
            initial: vec![uniform; n],
            transition: vec![vec![uniform; n]; n],
            state_names: (0..n).map(|i| format!("S{i}")).collect(),
            config,
        }
    }

    /// Create an HMM with the given parameters.
    pub fn with_params(
        initial: Vec<f64>,
        transition: Vec<Vec<f64>>,
        state_names: Vec<String>,
    ) -> Self {
        let n = initial.len();
        assert_eq!(transition.len(), n);
        for row in &transition {
            assert_eq!(row.len(), n);
        }
        Self {
            config: HmmConfig {
                n_states: n,
                ..Default::default()
            },
            initial,
            transition,
            state_names,
        }
    }

    /// Pre-configured HMM for 5-stage sleep staging.
    ///
    /// States: Wake(0), N1(1), N2(2), N3(3), REM(4)
    ///
    /// Transition probabilities based on sleep architecture literature
    /// (Carskadon & Dement 2011, AASM scoring manual):
    /// - Wake → N1 is the primary sleep onset pathway
    /// - N1 → N2 → N3 is the deepening pathway
    /// - N3 → N2 → REM is the lightening/REM pathway
    /// - REM → N2 or Wake for cycle transitions
    /// - Direct N3 → REM and Wake → N3 are rare/impossible
    pub fn sleep_staging() -> Self {
        // Transition probabilities for 30-second epochs
        // Rows: from-state, Columns: to-state
        //              Wake    N1      N2      N3      REM
        let transition = vec![
            vec![0.85, 0.10, 0.03, 0.00, 0.02], // Wake
            vec![0.10, 0.50, 0.35, 0.00, 0.05], // N1
            vec![0.02, 0.05, 0.75, 0.13, 0.05], // N2
            vec![0.00, 0.00, 0.10, 0.85, 0.05], // N3
            vec![0.05, 0.05, 0.15, 0.00, 0.75], // REM
        ];

        let initial = vec![0.60, 0.20, 0.10, 0.05, 0.05];

        let state_names = vec![
            "Wake".into(),
            "N1".into(),
            "N2".into(),
            "N3".into(),
            "REM".into(),
        ];

        Self::with_params(initial, transition, state_names)
    }

    /// Pre-configured HMM for anesthesia depth monitoring.
    ///
    /// States: Alert(0), Light(1), Moderate(2), Deep(3), Surgical(4), Burst-Suppression(5)
    pub fn anesthesia() -> Self {
        let transition = vec![
            vec![0.80, 0.15, 0.04, 0.01, 0.00, 0.00], // Alert
            vec![0.10, 0.65, 0.20, 0.05, 0.00, 0.00], // Light
            vec![0.02, 0.08, 0.70, 0.15, 0.05, 0.00], // Moderate
            vec![0.00, 0.02, 0.10, 0.70, 0.15, 0.03], // Deep
            vec![0.00, 0.00, 0.03, 0.12, 0.75, 0.10], // Surgical
            vec![0.00, 0.00, 0.00, 0.05, 0.15, 0.80], // Burst-Suppression
        ];

        let initial = vec![0.90, 0.05, 0.03, 0.01, 0.005, 0.005];

        let state_names = vec![
            "Alert".into(),
            "Light".into(),
            "Moderate".into(),
            "Deep".into(),
            "Surgical".into(),
            "BurstSuppression".into(),
        ];

        Self::with_params(initial, transition, state_names)
    }

    // -----------------------------------------------------------------------
    // Forward Algorithm
    // -----------------------------------------------------------------------

    /// Forward algorithm: compute P(O₁..Oₜ, state_t = i | model) for all t.
    ///
    /// `observations[t][i]` = P(observation_t | state_i), i.e., the emission
    /// likelihood for observation at time t given state i.
    ///
    /// Returns (alpha, log_likelihood):
    /// - `alpha[t][i]` = scaled forward variable
    /// - `scaling[t]` = normalization factor at time t
    /// - `log_likelihood` = log P(O | model)
    pub fn forward(&self, observations: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
        let n = self.config.n_states;
        let t_len = observations.len();

        if t_len == 0 {
            return (vec![], vec![], f64::NEG_INFINITY);
        }

        let mut alpha = vec![vec![0.0; n]; t_len];
        let mut scaling = vec![0.0; t_len];

        // Initialization: alpha_0(i) = pi(i) * b_i(O_0)
        for i in 0..n {
            alpha[0][i] = self.initial[i] * observations[0][i].max(self.config.prob_floor);
        }

        // Scale to prevent underflow
        let sum: f64 = alpha[0].iter().sum();
        scaling[0] = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for i in 0..n {
            alpha[0][i] *= scaling[0];
        }

        // Induction: alpha_t(j) = [sum_i alpha_{t-1}(i) * a_{ij}] * b_j(O_t)
        for t in 1..t_len {
            for j in 0..n {
                let mut sum_prev = 0.0;
                for i in 0..n {
                    sum_prev += alpha[t - 1][i] * self.transition[i][j];
                }
                alpha[t][j] = sum_prev * observations[t][j].max(self.config.prob_floor);
            }

            // Scale
            let sum: f64 = alpha[t].iter().sum();
            scaling[t] = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            for j in 0..n {
                alpha[t][j] *= scaling[t];
            }
        }

        // Log-likelihood = -sum(log(scaling[t]))
        let log_likelihood: f64 = scaling.iter().map(|&c| -(c.ln())).sum();

        (alpha, scaling, log_likelihood)
    }

    // -----------------------------------------------------------------------
    // Backward Algorithm
    // -----------------------------------------------------------------------

    /// Backward algorithm: compute P(O_{t+1}..O_T | state_t = i, model).
    ///
    /// Uses the same scaling factors from the forward pass to prevent underflow.
    pub fn backward(&self, observations: &[Vec<f64>], scaling: &[f64]) -> Vec<Vec<f64>> {
        let n = self.config.n_states;
        let t_len = observations.len();

        if t_len == 0 {
            return vec![];
        }

        let mut beta = vec![vec![0.0; n]; t_len];

        // Initialization: beta_T(i) = 1 (scaled)
        for i in 0..n {
            beta[t_len - 1][i] = scaling[t_len - 1];
        }

        // Induction: beta_t(i) = sum_j a_{ij} * b_j(O_{t+1}) * beta_{t+1}(j)
        for t in (0..t_len - 1).rev() {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += self.transition[i][j]
                        * observations[t + 1][j].max(self.config.prob_floor)
                        * beta[t + 1][j];
                }
                beta[t][i] = sum * scaling[t];
            }
        }

        beta
    }

    // -----------------------------------------------------------------------
    // Viterbi Decoding
    // -----------------------------------------------------------------------

    /// Viterbi algorithm: find the most likely state sequence.
    ///
    /// Uses dynamic programming to find:
    ///   argmax_{s₁..sₜ} P(s₁..sₜ | O₁..Oₜ, model)
    ///
    /// `observations[t][i]` = P(observation_t | state_i).
    ///
    /// Returns the optimal state sequence as indices.
    pub fn viterbi(&self, observations: &[Vec<f64>]) -> Vec<usize> {
        let n = self.config.n_states;
        let t_len = observations.len();

        if t_len == 0 {
            return vec![];
        }

        // Work in log-space to prevent underflow
        let log_initial: Vec<f64> = self
            .initial
            .iter()
            .map(|&p| (p.max(self.config.prob_floor)).ln())
            .collect();

        let log_transition: Vec<Vec<f64>> = self
            .transition
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&p| (p.max(self.config.prob_floor)).ln())
                    .collect()
            })
            .collect();

        // delta[t][j] = max log-probability of being in state j at time t
        let mut delta = vec![vec![f64::NEG_INFINITY; n]; t_len];
        // psi[t][j] = argmax predecessor state for backtracking
        let mut psi = vec![vec![0usize; n]; t_len];

        // Initialization
        for j in 0..n {
            let log_obs = (observations[0][j].max(self.config.prob_floor)).ln();
            delta[0][j] = log_initial[j] + log_obs;
        }

        // Recursion
        for t in 1..t_len {
            for j in 0..n {
                let log_obs = (observations[t][j].max(self.config.prob_floor)).ln();
                let mut best_val = f64::NEG_INFINITY;
                let mut best_i = 0;

                for i in 0..n {
                    let val = delta[t - 1][i] + log_transition[i][j];
                    if val > best_val {
                        best_val = val;
                        best_i = i;
                    }
                }

                delta[t][j] = best_val + log_obs;
                psi[t][j] = best_i;
            }
        }

        // Termination: find best final state
        let mut best_final = 0;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..n {
            if delta[t_len - 1][j] > best_val {
                best_val = delta[t_len - 1][j];
                best_final = j;
            }
        }

        // Backtrack
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_final;
        for t in (0..t_len - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }

        path
    }

    // -----------------------------------------------------------------------
    // Posterior State Probabilities (Forward-Backward)
    // -----------------------------------------------------------------------

    /// Compute posterior state probabilities γ(t, i) = P(state_t = i | O, model).
    ///
    /// Uses the forward-backward algorithm. Returns `gamma[t][i]`.
    pub fn posterior(&self, observations: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.config.n_states;
        let t_len = observations.len();

        if t_len == 0 {
            return vec![];
        }

        let (alpha, scaling, _log_ll) = self.forward(observations);
        let beta = self.backward(observations, &scaling);

        let mut gamma = vec![vec![0.0; n]; t_len];

        for t in 0..t_len {
            let mut sum = 0.0;
            for i in 0..n {
                gamma[t][i] = alpha[t][i] * beta[t][i];
                sum += gamma[t][i];
            }
            // Normalize
            if sum > 0.0 {
                for i in 0..n {
                    gamma[t][i] /= sum;
                }
            }
        }

        gamma
    }

    /// Smooth raw observation likelihoods into posterior state probabilities,
    /// then return the most probable state at each time step.
    ///
    /// This is the "soft" alternative to Viterbi — it finds the individually
    /// most probable state at each time step (which may not form a globally
    /// optimal sequence, but is often more robust for noisy observations).
    pub fn smooth_classify(&self, observations: &[Vec<f64>]) -> Vec<usize> {
        let gamma = self.posterior(observations);
        gamma
            .iter()
            .map(|probs| {
                probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Baum-Welch (EM) Parameter Estimation
    // -----------------------------------------------------------------------

    /// Baum-Welch algorithm: estimate HMM parameters from observations.
    ///
    /// This is the Expectation-Maximization (EM) algorithm for HMMs.
    /// It iteratively refines the transition matrix and initial distribution
    /// to maximize P(observations | model).
    ///
    /// `observations[t][i]` = emission likelihood for time t, state i.
    ///
    /// Returns the number of iterations and final log-likelihood.
    pub fn baum_welch(&mut self, observations: &[Vec<f64>]) -> (usize, f64) {
        let n = self.config.n_states;
        let t_len = observations.len();

        if t_len < 2 {
            return (0, f64::NEG_INFINITY);
        }

        let max_iter = self.config.max_iterations;
        let threshold = self.config.convergence_threshold;
        let floor = self.config.prob_floor;

        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..max_iter {
            // E-step: compute forward, backward, gamma, xi
            let (alpha, scaling, log_ll) = self.forward(observations);
            let beta = self.backward(observations, &scaling);

            // Check convergence
            if (log_ll - prev_ll).abs() < threshold && iter > 0 {
                return (iter, log_ll);
            }
            prev_ll = log_ll;

            // Gamma: P(state_t = i | O, model)
            let mut gamma = vec![vec![0.0; n]; t_len];
            for t in 0..t_len {
                let mut sum = 0.0;
                for i in 0..n {
                    gamma[t][i] = alpha[t][i] * beta[t][i];
                    sum += gamma[t][i];
                }
                if sum > 0.0 {
                    for i in 0..n {
                        gamma[t][i] /= sum;
                    }
                }
            }

            // Xi: P(state_t = i, state_{t+1} = j | O, model)
            let mut xi = vec![vec![vec![0.0; n]; n]; t_len - 1];
            for t in 0..t_len - 1 {
                let mut denom = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        xi[t][i][j] = alpha[t][i]
                            * self.transition[i][j]
                            * observations[t + 1][j].max(floor)
                            * beta[t + 1][j];
                        denom += xi[t][i][j];
                    }
                }
                if denom > 0.0 {
                    for i in 0..n {
                        for j in 0..n {
                            xi[t][i][j] /= denom;
                        }
                    }
                }
            }

            // M-step: re-estimate parameters

            // Initial distribution
            for i in 0..n {
                self.initial[i] = gamma[0][i].max(floor);
            }
            let init_sum: f64 = self.initial.iter().sum();
            if init_sum > 0.0 {
                for p in self.initial.iter_mut() {
                    *p /= init_sum;
                }
            }

            // Transition matrix
            for i in 0..n {
                let gamma_sum: f64 = (0..t_len - 1).map(|t| gamma[t][i]).sum();
                for j in 0..n {
                    let xi_sum: f64 = (0..t_len - 1).map(|t| xi[t][i][j]).sum();
                    self.transition[i][j] = if gamma_sum > 0.0 {
                        (xi_sum / gamma_sum).max(floor)
                    } else {
                        floor
                    };
                }
                // Normalize row
                let row_sum: f64 = self.transition[i].iter().sum();
                if row_sum > 0.0 {
                    for j in 0..n {
                        self.transition[i][j] /= row_sum;
                    }
                }
            }
        }

        let (_, _, log_ll) = self.forward(observations);
        (max_iter, log_ll)
    }

    // -----------------------------------------------------------------------
    // Likelihood
    // -----------------------------------------------------------------------

    /// Compute log P(observations | model) using the forward algorithm.
    pub fn log_likelihood(&self, observations: &[Vec<f64>]) -> f64 {
        let (_, _, ll) = self.forward(observations);
        ll
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// Convert raw classifier outputs (per-epoch probabilities) into
    /// observation likelihoods suitable for HMM input.
    ///
    /// Ensures each row sums to 1 and has no zeros.
    pub fn normalize_observations(raw: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let floor = 1e-10;
        raw.iter()
            .map(|row| {
                let sum: f64 = row.iter().map(|&v| v.max(floor)).sum();
                row.iter().map(|&v| v.max(floor) / sum).collect()
            })
            .collect()
    }

    /// Get the state name for a given index.
    pub fn state_name(&self, idx: usize) -> &str {
        self.state_names
            .get(idx)
            .map(|s| s.as_str())
            .unwrap_or("Unknown")
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 2-state HMM (Rainy/Sunny) for testing.
    fn two_state_hmm() -> HiddenMarkovModel {
        HiddenMarkovModel::with_params(
            vec![0.6, 0.4],
            vec![
                vec![0.7, 0.3], // Rainy → Rainy: 0.7, Rainy → Sunny: 0.3
                vec![0.4, 0.6], // Sunny → Rainy: 0.4, Sunny → Sunny: 0.6
            ],
            vec!["Rainy".into(), "Sunny".into()],
        )
    }

    #[test]
    fn test_forward_algorithm_basic() {
        let hmm = two_state_hmm();
        // Observation likelihoods: [P(obs|Rainy), P(obs|Sunny)]
        let obs = vec![
            vec![0.9, 0.2], // Likely Rainy
            vec![0.8, 0.3], // Likely Rainy
            vec![0.2, 0.9], // Likely Sunny
        ];

        let (alpha, _scaling, log_ll) = hmm.forward(&obs);
        assert_eq!(alpha.len(), 3);
        assert!(log_ll.is_finite());

        // Forward probabilities should sum to ~1 at each step (due to scaling)
        for t in 0..3 {
            let sum: f64 = alpha[t].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Forward probs at t={} sum to {} (expected 1.0)",
                t,
                sum
            );
        }
    }

    #[test]
    fn test_viterbi_follows_strong_evidence() {
        let hmm = two_state_hmm();
        // Strong evidence for: Rainy, Rainy, Sunny, Sunny
        let obs = vec![
            vec![0.95, 0.05],
            vec![0.90, 0.10],
            vec![0.10, 0.90],
            vec![0.05, 0.95],
        ];

        let path = hmm.viterbi(&obs);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], 0, "Should decode as Rainy");
        assert_eq!(path[1], 0, "Should decode as Rainy");
        assert_eq!(path[2], 1, "Should decode as Sunny");
        assert_eq!(path[3], 1, "Should decode as Sunny");
    }

    #[test]
    fn test_viterbi_transition_constraints() {
        // Test that transition constraints override weak observation evidence
        let hmm = HiddenMarkovModel::with_params(
            vec![0.99, 0.01],
            vec![
                vec![0.99, 0.01], // State 0 is very sticky
                vec![0.01, 0.99], // State 1 is very sticky
            ],
            vec!["A".into(), "B".into()],
        );

        // Weak evidence that alternates, but sticky transitions should
        // keep the model in state 0 (started there with high prior)
        let obs = vec![
            vec![0.55, 0.45],
            vec![0.45, 0.55],
            vec![0.55, 0.45],
            vec![0.45, 0.55],
        ];

        let path = hmm.viterbi(&obs);
        // With 99% self-transition and nearly ambiguous observations,
        // Viterbi should stay in starting state
        assert!(
            path.iter().all(|&s| s == 0),
            "Sticky transitions should keep in state 0, got {:?}",
            path
        );
    }

    #[test]
    fn test_posterior_sums_to_one() {
        let hmm = two_state_hmm();
        let obs = vec![vec![0.9, 0.2], vec![0.5, 0.5], vec![0.2, 0.8]];

        let gamma = hmm.posterior(&obs);
        assert_eq!(gamma.len(), 3);

        for (t, probs) in gamma.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Posterior at t={} sums to {} (expected 1.0)",
                t,
                sum
            );
        }
    }

    #[test]
    fn test_smooth_classify() {
        let hmm = two_state_hmm();
        let obs = vec![vec![0.9, 0.1], vec![0.8, 0.2], vec![0.1, 0.9]];

        let states = hmm.smooth_classify(&obs);
        assert_eq!(states.len(), 3);
        assert_eq!(states[0], 0, "First epoch clearly Rainy");
        assert_eq!(states[2], 1, "Last epoch clearly Sunny");
    }

    #[test]
    fn test_sleep_staging_hmm_valid() {
        let hmm = HiddenMarkovModel::sleep_staging();
        assert_eq!(hmm.config.n_states, 5);
        assert_eq!(hmm.state_names.len(), 5);

        // Verify rows sum to 1
        for (i, row) in hmm.transition.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Transition row {} sums to {}",
                i,
                sum
            );
        }

        // Verify initial sums to 1
        let init_sum: f64 = hmm.initial.iter().sum();
        assert!(
            (init_sum - 1.0).abs() < 1e-10,
            "Initial distribution sums to {}",
            init_sum
        );

        // Verify no direct Wake→N3 transition
        assert!(
            hmm.transition[0][3] < 0.01,
            "Wake→N3 should be near-zero, got {}",
            hmm.transition[0][3]
        );
    }

    #[test]
    fn test_sleep_viterbi_prevents_impossible_jumps() {
        let hmm = HiddenMarkovModel::sleep_staging();

        // Simulate: strong N3 signal, then suddenly strong Wake signal
        // HMM should smooth through N2 → N1 → Wake instead of jumping
        let mut obs = Vec::new();
        for _ in 0..10 {
            obs.push(vec![0.02, 0.03, 0.05, 0.85, 0.05]); // Strong N3
        }
        for _ in 0..10 {
            obs.push(vec![0.85, 0.05, 0.03, 0.02, 0.05]); // Strong Wake
        }

        let path = hmm.viterbi(&obs);

        // First epochs should be N3
        assert_eq!(path[0], 3, "Should start in N3");

        // Last epochs should be Wake
        assert_eq!(path[19], 0, "Should end in Wake");

        // The transition should go through intermediate stages
        // (not jump directly N3 → Wake since transition prob is 0.0)
        // Valid intermediates: N1 (1), N2 (2), or REM (4)
        let mut has_intermediate = false;
        for i in 5..15 {
            if path[i] == 1 || path[i] == 2 || path[i] == 4 {
                has_intermediate = true;
                break;
            }
        }
        assert!(
            has_intermediate,
            "Should pass through N1, N2, or REM between N3 and Wake, got {:?}",
            &path[5..15]
        );
    }

    #[test]
    fn test_baum_welch_improves_likelihood() {
        let mut hmm = HiddenMarkovModel::new(HmmConfig {
            n_states: 2,
            ..Default::default()
        });

        // Generate observations from a known model
        let obs = vec![
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.3, 0.7],
            vec![0.2, 0.8],
            vec![0.7, 0.3],
            vec![0.9, 0.1],
            vec![0.4, 0.6],
            vec![0.1, 0.9],
        ];

        let ll_before = hmm.log_likelihood(&obs);
        let (iters, ll_after) = hmm.baum_welch(&obs);

        assert!(
            ll_after >= ll_before - 1e-6,
            "Baum-Welch should not decrease likelihood: before={}, after={}, iters={}",
            ll_before,
            ll_after,
            iters
        );
    }

    #[test]
    fn test_normalize_observations() {
        let raw = vec![vec![0.0, 0.0, 1.0], vec![5.0, 3.0, 2.0]];

        let normalized = HiddenMarkovModel::normalize_observations(&raw);

        // First row: zeros become floor, then normalized
        let sum0: f64 = normalized[0].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-6);

        // Second row
        let sum1: f64 = normalized[1].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_anesthesia_hmm_valid() {
        let hmm = HiddenMarkovModel::anesthesia();
        assert_eq!(hmm.config.n_states, 6);

        for (i, row) in hmm.transition.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Anesthesia transition row {} sums to {}",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_empty_observations() {
        let hmm = two_state_hmm();
        let empty: Vec<Vec<f64>> = vec![];

        assert_eq!(hmm.viterbi(&empty), Vec::<usize>::new());
        assert_eq!(hmm.posterior(&empty), Vec::<Vec<f64>>::new());
        assert_eq!(hmm.smooth_classify(&empty), Vec::<usize>::new());
    }
}
