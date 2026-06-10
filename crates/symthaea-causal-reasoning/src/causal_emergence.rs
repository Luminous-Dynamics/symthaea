// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Emergence - Does Consciousness Have Real Causal Power?
//!
//! This module implements Erik Hoel's theory of Causal Emergence, which provides
//! a rigorous mathematical framework for determining whether macro-level
//! descriptions (like consciousness) have genuine causal power beyond their
//! micro-level substrates (neurons).
//!
//! # The Key Question
//!
//! Is consciousness merely an epiphenomenon (a passive side-effect of neural
//! activity with no causal role), or does it have genuine "downward causation"
//! where the conscious state actually influences neural dynamics?
//!
//! Causal emergence provides a quantitative answer.
//!
//! # Theory (Hoel et al.)
//!
//! **Effective Information (EI)**: Measures how much information a system's
//! causal mechanism generates about its effects.
//!
//! ```text
//! EI(X → Y) = MI(Xmax; Y)
//! ```
//!
//! Where Xmax is the maximum entropy distribution over inputs, and MI is
//! mutual information with outputs Y.
//!
//! **Causal Emergence (CE)**: Occurs when macro-level EI exceeds micro-level EI.
//!
//! ```text
//! CE = EI(macro) - EI(micro) > 0
//! ```
//!
//! When CE > 0, the macro description has MORE causal power than the micro.
//! Consciousness would then be "real" in the strongest possible sense.
//!
//! # Implementation
//!
//! We compute:
//! 1. **Micro-level EI**: Information in neuron → neuron transitions
//! 2. **Macro-level EI**: Information in conscious-state → conscious-state transitions
//! 3. **Emergence**: CE = macro_EI - micro_EI
//!
//! # References
//!
//! - Hoel, E.P. (2017). "When the Map Is Better Than the Territory"
//! - Hoel, E.P. et al. (2016). "Quantifying causal emergence shows that macro
//!   can beat micro"

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// EFFECTIVE INFORMATION
// ═══════════════════════════════════════════════════════════════════════════

/// Compute entropy of a probability distribution
fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Compute mutual information between two discrete variables
///
/// Given joint distribution P(X,Y), computes:
/// MI(X;Y) = H(X) + H(Y) - H(X,Y)
fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    if joint.is_empty() || joint[0].is_empty() {
        return 0.0;
    }

    let rows = joint.len();
    let cols = joint[0].len();

    // Compute marginals
    let mut p_x = vec![0.0; rows];
    let mut p_y = vec![0.0; cols];

    for (i, row) in joint.iter().enumerate() {
        for (j, &p) in row.iter().enumerate() {
            p_x[i] += p;
            p_y[j] += p;
        }
    }

    // H(X)
    let h_x = entropy(&p_x);

    // H(Y)
    let h_y = entropy(&p_y);

    // H(X,Y) - entropy of joint
    let joint_probs: Vec<f64> = joint.iter().flatten().copied().collect();
    let h_xy = entropy(&joint_probs);

    // MI = H(X) + H(Y) - H(X,Y)
    (h_x + h_y - h_xy).max(0.0)
}

/// Effective Information of a transition probability matrix
///
/// EI(TPM) = MI(Xmax; Y) where Xmax is maximum entropy over inputs
///
/// For a transition matrix T where T[i][j] = P(Y=j | X=i):
/// - We assume maximum entropy input (uniform distribution)
/// - Compute mutual information with output
pub fn effective_information(tpm: &[Vec<f64>]) -> f64 {
    if tpm.is_empty() || tpm[0].is_empty() {
        return 0.0;
    }

    let num_states = tpm.len();

    // Maximum entropy input = uniform distribution
    let p_input = 1.0 / num_states as f64;

    // Construct joint distribution P(X,Y) = P(X) * P(Y|X)
    let mut joint = vec![vec![0.0; num_states]; num_states];

    for (i, row) in tpm.iter().enumerate() {
        for (j, &p_yx) in row.iter().enumerate() {
            joint[i][j] = p_input * p_yx;
        }
    }

    mutual_information(&joint)
}

/// Compute determinism of a TPM (how deterministic are transitions?)
///
/// Determinism = 1 means perfectly deterministic (each state goes to exactly one next state)
/// Determinism = 0 means completely random (uniform transitions)
pub fn determinism(tpm: &[Vec<f64>]) -> f64 {
    if tpm.is_empty() {
        return 0.0;
    }

    let num_states = tpm.len();
    let max_entropy = (num_states as f64).log2();

    if max_entropy < 1e-10 {
        return 1.0;
    }

    // Average entropy of each row (conditional entropy of output given input)
    let avg_row_entropy: f64 = tpm.iter().map(|row| entropy(row)).sum::<f64>() / num_states as f64;

    // Determinism = 1 - (avg_row_entropy / max_entropy)
    (1.0 - avg_row_entropy / max_entropy).clamp(0.0, 1.0)
}

/// Compute degeneracy of a TPM (how many inputs lead to same output?)
///
/// Degeneracy = 0 means each input leads to unique output distribution
/// Degeneracy = 1 means all inputs lead to identical output
pub fn degeneracy(tpm: &[Vec<f64>]) -> f64 {
    if tpm.is_empty() || tpm.len() < 2 {
        return 0.0;
    }

    let num_states = tpm.len();

    // Compute variance of transition probabilities across inputs
    let mut total_variance = 0.0;

    for j in 0..num_states {
        // Get column j (probability of reaching state j from each input)
        let column: Vec<f64> = tpm.iter().map(|row| row[j]).collect();
        let mean = column.iter().sum::<f64>() / num_states as f64;
        let variance: f64 =
            column.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / num_states as f64;
        total_variance += variance;
    }

    // Low variance = high degeneracy (all inputs lead to similar outputs)
    let avg_variance = total_variance / num_states as f64;
    let max_variance = 0.25; // Maximum for [0,1] distributions

    (1.0 - (avg_variance / max_variance).sqrt()).clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// CAUSAL EMERGENCE ANALYZER
// ═══════════════════════════════════════════════════════════════════════════

/// State discretization for micro/macro analysis
#[derive(Debug, Clone)]
pub struct StateDiscretizer {
    /// Number of discrete states for micro-level
    pub micro_states: usize,

    /// Number of discrete states for macro-level
    pub macro_states: usize,

    /// Discretization thresholds for micro
    micro_thresholds: Vec<f64>,

    /// Discretization thresholds for macro
    macro_thresholds: Vec<f64>,
}

impl StateDiscretizer {
    /// Create discretizer with given granularities
    pub fn new(micro_states: usize, macro_states: usize) -> Self {
        // Equal-width thresholds
        let micro_thresholds: Vec<f64> = (1..micro_states)
            .map(|i| i as f64 / micro_states as f64)
            .collect();

        let macro_thresholds: Vec<f64> = (1..macro_states)
            .map(|i| i as f64 / macro_states as f64)
            .collect();

        Self {
            micro_states,
            macro_states,
            micro_thresholds,
            macro_thresholds,
        }
    }

    /// Discretize a continuous value to micro state
    pub fn to_micro(&self, value: f64) -> usize {
        let v = value.clamp(0.0, 1.0);
        for (i, &thresh) in self.micro_thresholds.iter().enumerate() {
            if v < thresh {
                return i;
            }
        }
        self.micro_states - 1
    }

    /// Discretize a continuous value to macro state
    pub fn to_macro(&self, value: f64) -> usize {
        let v = value.clamp(0.0, 1.0);
        for (i, &thresh) in self.macro_thresholds.iter().enumerate() {
            if v < thresh {
                return i;
            }
        }
        self.macro_states - 1
    }
}

/// Result of causal emergence analysis
#[derive(Debug, Clone)]
pub struct CausalEmergenceResult {
    /// Effective information at micro-level
    pub micro_ei: f64,

    /// Effective information at macro-level
    pub macro_ei: f64,

    /// Causal emergence (macro_ei - micro_ei)
    pub emergence: f64,

    /// Determinism at micro-level
    pub micro_determinism: f64,

    /// Determinism at macro-level
    pub macro_determinism: f64,

    /// Degeneracy at micro-level
    pub micro_degeneracy: f64,

    /// Degeneracy at macro-level
    pub macro_degeneracy: f64,

    /// Does consciousness emerge? (CE > 0)
    pub consciousness_emerges: bool,

    /// Interpretation
    pub interpretation: String,
}

/// Causal Emergence Analyzer
///
/// Analyzes whether consciousness exhibits causal emergence over neural substrates.
pub struct CausalEmergenceAnalyzer {
    /// State discretizer
    discretizer: StateDiscretizer,

    /// History of micro states (neuron activations)
    micro_history: Vec<Vec<f64>>,

    /// History of macro states (consciousness levels)
    macro_history: Vec<f64>,

    /// Transition counts for micro-level TPM
    micro_transitions: HashMap<(usize, usize), usize>,

    /// Transition counts for macro-level TPM
    macro_transitions: HashMap<(usize, usize), usize>,

    /// Total micro transitions observed
    micro_total: usize,

    /// Total macro transitions observed
    macro_total: usize,
}

impl CausalEmergenceAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self::with_granularity(16, 8) // 16 micro states, 8 macro states
    }

    /// Create with custom granularity
    pub fn with_granularity(micro_states: usize, macro_states: usize) -> Self {
        Self {
            discretizer: StateDiscretizer::new(micro_states, macro_states),
            micro_history: Vec::new(),
            macro_history: Vec::new(),
            micro_transitions: HashMap::new(),
            macro_transitions: HashMap::new(),
            micro_total: 0,
            macro_total: 0,
        }
    }

    /// Record an observation
    ///
    /// # Arguments
    /// * `micro_state` - Vector of neuron activations [0, 1]
    /// * `macro_state` - Consciousness level [0, 1]
    pub fn observe(&mut self, micro_state: &[f64], macro_state: f64) {
        // Discretize current states
        let micro_discrete: Vec<usize> = micro_state
            .iter()
            .map(|&v| self.discretizer.to_micro(v))
            .collect();
        let macro_discrete = self.discretizer.to_macro(macro_state);

        // Record transitions from previous state
        if let Some(prev_micro_state) = self.micro_history.last() {
            // For micro: use aggregate state (sum of discrete values)
            let prev_micro: usize = prev_micro_state
                .iter()
                .map(|&v| self.discretizer.to_micro(v))
                .sum::<usize>()
                % self.discretizer.micro_states;
            let curr_micro: usize =
                micro_discrete.iter().sum::<usize>() % self.discretizer.micro_states;

            *self
                .micro_transitions
                .entry((prev_micro, curr_micro))
                .or_insert(0) += 1;
            self.micro_total += 1;
        }

        if let Some(&last_macro) = self.macro_history.last() {
            let prev_macro = self.discretizer.to_macro(last_macro);
            *self
                .macro_transitions
                .entry((prev_macro, macro_discrete))
                .or_insert(0) += 1;
            self.macro_total += 1;
        }

        // Update history
        self.micro_history.push(micro_state.to_vec());
        self.macro_history.push(macro_state);

        // Keep history bounded
        if self.micro_history.len() > 10000 {
            self.micro_history.remove(0);
            self.macro_history.remove(0);
        }
    }

    /// Build transition probability matrix from counts
    fn build_tpm(
        &self,
        transitions: &HashMap<(usize, usize), usize>,
        num_states: usize,
    ) -> Vec<Vec<f64>> {
        let mut tpm = vec![vec![0.0; num_states]; num_states];
        let mut row_sums = vec![0usize; num_states];

        // Count transitions per row
        for (&(from, to), &count) in transitions {
            if from < num_states && to < num_states {
                tpm[from][to] += count as f64;
                row_sums[from] += count;
            }
        }

        // Normalize rows to probabilities
        for (i, row) in tpm.iter_mut().enumerate() {
            let sum = row_sums[i];
            if sum > 0 {
                for p in row.iter_mut() {
                    *p /= sum as f64;
                }
            } else {
                // No transitions observed: assume uniform
                for p in row.iter_mut() {
                    *p = 1.0 / num_states as f64;
                }
            }
        }

        tpm
    }

    /// Analyze causal emergence
    pub fn analyze(&self) -> CausalEmergenceResult {
        // Build TPMs
        let micro_tpm = self.build_tpm(&self.micro_transitions, self.discretizer.micro_states);
        let macro_tpm = self.build_tpm(&self.macro_transitions, self.discretizer.macro_states);

        // Compute effective information
        let micro_ei = effective_information(&micro_tpm);
        let macro_ei = effective_information(&macro_tpm);

        // Compute emergence
        let emergence = macro_ei - micro_ei;

        // Compute determinism and degeneracy
        let micro_determinism = determinism(&micro_tpm);
        let macro_determinism = determinism(&macro_tpm);
        let micro_degeneracy = degeneracy(&micro_tpm);
        let macro_degeneracy = degeneracy(&macro_tpm);

        // Interpret results
        let consciousness_emerges = emergence > 0.0;

        let interpretation = if emergence > 0.5 {
            "Strong causal emergence: Consciousness has significantly more causal power \
             than the underlying neurons. The macro-level description is genuinely \
             better than the micro-level - consciousness is REAL."
                .to_string()
        } else if emergence > 0.1 {
            "Moderate causal emergence: Consciousness shows some irreducible causal \
             power beyond neural activity. The macro-level has explanatory advantage."
                .to_string()
        } else if emergence > 0.0 {
            "Weak causal emergence: Slight advantage for macro-level description. \
             Consciousness may have marginal causal efficacy."
                .to_string()
        } else if emergence > -0.1 {
            "Causal equivalence: Macro and micro descriptions have similar causal power. \
             Consciousness is neither more nor less causally effective than neurons."
                .to_string()
        } else {
            "Causal reduction: Micro-level has MORE causal power than macro. \
             Consciousness may be epiphenomenal - a passive side-effect."
                .to_string()
        };

        CausalEmergenceResult {
            micro_ei,
            macro_ei,
            emergence,
            micro_determinism,
            macro_determinism,
            micro_degeneracy,
            macro_degeneracy,
            consciousness_emerges,
            interpretation,
        }
    }

    /// Get number of observations
    pub fn num_observations(&self) -> usize {
        self.micro_history.len()
    }

    /// Reset analyzer
    pub fn reset(&mut self) {
        self.micro_history.clear();
        self.macro_history.clear();
        self.micro_transitions.clear();
        self.macro_transitions.clear();
        self.micro_total = 0;
        self.macro_total = 0;
    }
}

impl Default for CausalEmergenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        // Uniform distribution
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy(&uniform);
        assert!(
            (h - 2.0).abs() < 0.01,
            "Uniform over 4 states should have entropy 2"
        );

        // Deterministic
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let h = entropy(&deterministic);
        assert!(h.abs() < 0.01, "Deterministic should have entropy 0");
    }

    #[test]
    fn test_effective_information() {
        // Deterministic TPM (high EI)
        let deterministic_tpm = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let ei_det = effective_information(&deterministic_tpm);

        // Random TPM (low EI)
        let random_tpm = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let ei_rand = effective_information(&random_tpm);

        println!("Deterministic EI: {:.3}", ei_det);
        println!("Random EI: {:.3}", ei_rand);

        assert!(
            ei_det > ei_rand,
            "Deterministic TPM should have higher EI than random"
        );
    }

    #[test]
    fn test_causal_emergence_analyzer() {
        let mut analyzer = CausalEmergenceAnalyzer::new();

        // Simulate a system where macro state has clearer dynamics than micro
        for i in 0..1000 {
            // Micro: noisy neuron activations
            let micro: Vec<f64> = (0..10)
                .map(|j| {
                    let base = (i as f64 * 0.1 + j as f64 * 0.3).sin() * 0.5 + 0.5;
                    let noise = ((i * 17 + j * 23) % 100) as f64 / 200.0 - 0.25;
                    (base + noise).clamp(0.0, 1.0)
                })
                .collect();

            // Macro: cleaner consciousness level (average with less noise)
            let macro_state = (i as f64 * 0.1).sin() * 0.3 + 0.5;

            analyzer.observe(&micro, macro_state);
        }

        let result = analyzer.analyze();

        println!("\nCausal Emergence Analysis:");
        println!("  Micro EI: {:.3} bits", result.micro_ei);
        println!("  Macro EI: {:.3} bits", result.macro_ei);
        println!("  Emergence: {:.3} bits", result.emergence);
        // EI values should be finite and non-negative
        assert!(
            result.micro_ei >= 0.0 && result.micro_ei.is_finite(),
            "Micro EI must be finite non-negative: {}",
            result.micro_ei
        );
        assert!(
            result.macro_ei >= 0.0 && result.macro_ei.is_finite(),
            "Macro EI must be finite non-negative: {}",
            result.macro_ei
        );
        // Determinism should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&result.micro_determinism),
            "Micro determinism must be in [0,1]: {}",
            result.micro_determinism
        );
        assert!(
            (0.0..=1.0).contains(&result.macro_determinism),
            "Macro determinism must be in [0,1]: {}",
            result.macro_determinism
        );
        // Interpretation should not be empty
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_deterministic_macro_emergence() {
        let mut analyzer = CausalEmergenceAnalyzer::with_granularity(8, 4);

        // Create a system where:
        // - Micro is noisy/degenerate
        // - Macro is deterministic
        for i in 0..500 {
            // Micro: random-looking
            let micro: Vec<f64> = (0..5)
                .map(|j| ((i * 13 + j * 37) % 100) as f64 / 100.0)
                .collect();

            // Macro: perfectly cycling through 4 states
            let macro_state = (i % 4) as f64 / 4.0 + 0.125;

            analyzer.observe(&micro, macro_state);
        }

        let result = analyzer.analyze();

        println!("\nDeterministic Macro Test:");
        println!("  Emergence: {:.3} bits", result.emergence);

        // With deterministic macro and noisy micro, we should see positive emergence
        assert!(
            result.macro_determinism > result.micro_determinism,
            "Macro should be more deterministic"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Entropy edge-case tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_entropy_empty_distribution() {
        assert_eq!(entropy(&[]), 0.0);
    }

    #[test]
    fn test_entropy_single_certain_event() {
        assert!(entropy(&[1.0]).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_binary_maximum() {
        // Maximum entropy for binary variable = 1 bit
        let h = entropy(&[0.5, 0.5]);
        assert!(
            (h - 1.0).abs() < 1e-10,
            "Binary uniform entropy should be 1.0 bit, got {}",
            h
        );
    }

    #[test]
    fn test_entropy_near_zero_probs_ignored() {
        // Very small probabilities shouldn't produce NaN
        let probs = vec![1e-15, 1.0 - 1e-15];
        let h = entropy(&probs);
        assert!(h.is_finite(), "Entropy should handle near-zero probs");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Mutual information tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mutual_information_independent() {
        // Independent joint: P(X,Y) = P(X)*P(Y)
        let joint = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let mi = mutual_information(&joint);
        assert!(
            mi.abs() < 1e-6,
            "MI of independent variables should be 0, got {}",
            mi
        );
    }

    #[test]
    fn test_mutual_information_perfectly_correlated() {
        // Perfectly correlated: P(X=0,Y=0) = P(X=1,Y=1) = 0.5
        let joint = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        let mi = mutual_information(&joint);
        assert!(
            (mi - 1.0).abs() < 1e-6,
            "MI of perfectly correlated binary vars should be 1 bit, got {}",
            mi
        );
    }

    #[test]
    fn test_mutual_information_empty_joint() {
        assert_eq!(mutual_information(&[]), 0.0);
        let empty_cols: Vec<Vec<f64>> = vec![vec![]];
        assert_eq!(mutual_information(&empty_cols), 0.0);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Effective information tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_effective_information_empty_tpm() {
        assert_eq!(effective_information(&[]), 0.0);
    }

    #[test]
    fn test_effective_information_identity_tpm() {
        // Identity TPM: each state maps to itself
        let identity = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ei = effective_information(&identity);
        // EI should equal log2(3) ≈ 1.585 for a perfectly deterministic 1-to-1 map
        let expected = 3.0_f64.log2();
        assert!(
            (ei - expected).abs() < 0.01,
            "Identity TPM should have EI = log2(3) ≈ {:.3}, got {:.3}",
            expected,
            ei
        );
    }

    #[test]
    fn test_effective_information_uniform_tpm_is_zero() {
        let uniform = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let ei = effective_information(&uniform);
        assert!(
            ei.abs() < 1e-6,
            "Uniform TPM should have EI = 0, got {}",
            ei
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Determinism tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_determinism_empty_tpm() {
        assert_eq!(determinism(&[]), 0.0);
    }

    #[test]
    fn test_determinism_perfectly_deterministic() {
        let tpm = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let d = determinism(&tpm);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "Perfectly deterministic TPM should have determinism 1.0, got {}",
            d
        );
    }

    #[test]
    fn test_determinism_uniform_tpm() {
        let tpm = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let d = determinism(&tpm);
        assert!(
            d.abs() < 1e-6,
            "Uniform TPM should have determinism 0, got {}",
            d
        );
    }

    #[test]
    fn test_determinism_bounded_zero_one() {
        // Partially random TPM
        let tpm = vec![vec![0.6, 0.4], vec![0.3, 0.7]];
        let d = determinism(&tpm);
        assert!(
            d >= 0.0 && d <= 1.0,
            "Determinism must be in [0,1], got {}",
            d
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Degeneracy tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_degeneracy_empty_tpm() {
        assert_eq!(degeneracy(&[]), 0.0);
    }

    #[test]
    fn test_degeneracy_single_row() {
        // Single row means fewer than 2 states — degeneracy should be 0
        let tpm = vec![vec![0.5, 0.5]];
        assert_eq!(degeneracy(&tpm), 0.0);
    }

    #[test]
    fn test_degeneracy_identical_rows() {
        // All inputs lead to same output distribution = maximum degeneracy
        // Note: degeneracy() assumes square TPM (num_states x num_states)
        let tpm = vec![
            vec![0.3, 0.3, 0.4],
            vec![0.3, 0.3, 0.4],
            vec![0.3, 0.3, 0.4],
        ];
        let d = degeneracy(&tpm);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "Identical rows should give degeneracy = 1.0, got {}",
            d
        );
    }

    #[test]
    fn test_degeneracy_bounded_zero_one() {
        let tpm = vec![vec![0.8, 0.2], vec![0.4, 0.6]];
        let d = degeneracy(&tpm);
        assert!(
            d >= 0.0 && d <= 1.0,
            "Degeneracy must be in [0,1], got {}",
            d
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // StateDiscretizer tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_state_discretizer_boundary_values() {
        let disc = StateDiscretizer::new(4, 2);
        // Micro: 4 states with thresholds at 0.25, 0.5, 0.75
        assert_eq!(disc.to_micro(0.0), 0);
        assert_eq!(disc.to_micro(0.24), 0);
        assert_eq!(disc.to_micro(0.26), 1);
        assert_eq!(disc.to_micro(0.99), 3);
        assert_eq!(disc.to_micro(1.0), 3);
        // Macro: 2 states with threshold at 0.5
        assert_eq!(disc.to_macro(0.0), 0);
        assert_eq!(disc.to_macro(0.49), 0);
        assert_eq!(disc.to_macro(0.51), 1);
        assert_eq!(disc.to_macro(1.0), 1);
    }

    #[test]
    fn test_state_discretizer_clamps_out_of_range() {
        let disc = StateDiscretizer::new(3, 3);
        // Values outside [0, 1] should be clamped
        assert_eq!(disc.to_micro(-1.0), 0);
        assert_eq!(disc.to_micro(5.0), 2);
        assert_eq!(disc.to_macro(-0.5), 0);
        assert_eq!(disc.to_macro(2.0), 2);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Analyzer lifecycle tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_analyzer_default() {
        let analyzer = CausalEmergenceAnalyzer::default();
        assert_eq!(analyzer.num_observations(), 0);
    }

    #[test]
    fn test_analyzer_reset() {
        let mut analyzer = CausalEmergenceAnalyzer::new();
        analyzer.observe(&[0.5, 0.3], 0.6);
        analyzer.observe(&[0.2, 0.8], 0.4);
        assert_eq!(analyzer.num_observations(), 2);
        analyzer.reset();
        assert_eq!(analyzer.num_observations(), 0);
    }

    #[test]
    fn test_analyzer_no_observations_still_produces_result() {
        let analyzer = CausalEmergenceAnalyzer::new();
        let result = analyzer.analyze();
        // With no observations, TPMs are built from empty transitions → uniform rows
        assert!(result.micro_ei.is_finite());
        assert!(result.macro_ei.is_finite());
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_analyzer_single_observation_no_panic() {
        let mut analyzer = CausalEmergenceAnalyzer::new();
        analyzer.observe(&[0.5], 0.5);
        let result = analyzer.analyze();
        assert!(result.micro_ei.is_finite());
    }

    #[test]
    fn test_causal_emergence_result_fields() {
        let mut analyzer = CausalEmergenceAnalyzer::with_granularity(4, 2);
        for i in 0..100 {
            let micro = vec![
                (i as f64 * 0.07).sin() * 0.5 + 0.5,
                (i as f64 * 0.13).cos() * 0.5 + 0.5,
            ];
            let macro_state = (i as f64 * 0.05).sin() * 0.4 + 0.5;
            analyzer.observe(&micro, macro_state);
        }
        let r = analyzer.analyze();
        // All metric fields should be finite and bounded
        assert!(r.micro_ei >= 0.0 && r.micro_ei.is_finite());
        assert!(r.macro_ei >= 0.0 && r.macro_ei.is_finite());
        assert!(r.emergence.is_finite());
        assert!((0.0..=1.0).contains(&r.micro_determinism));
        assert!((0.0..=1.0).contains(&r.macro_determinism));
        assert!((0.0..=1.0).contains(&r.micro_degeneracy));
        assert!((0.0..=1.0).contains(&r.macro_degeneracy));
        // consciousness_emerges should match sign of emergence
        assert_eq!(r.consciousness_emerges, r.emergence > 0.0);
    }
}
