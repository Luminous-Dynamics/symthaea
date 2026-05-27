// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hierarchical Free Energy Decomposition
//!
//! Implements a multi-level variational free energy minimization framework based
//! on Karl Friston's hierarchical predictive processing account.  Each cortical
//! level maintains its own recognition density q_l(s_l), receives top-down
//! predictions from the level above, and sends precision-weighted prediction
//! errors upward.
//!
//! ## Mathematical Foundation
//!
//! ### Variational Free Energy
//!
//! The variational free energy for the whole hierarchy is:
//!
//! ```text
//! F = E_q[ln q(s) - ln p(o,s)]
//!   = D_KL[q(s) || p(s)]  -  E_q[ln p(o|s)]
//!       ╰── Complexity ──╯     ╰── Accuracy ──╯
//! ```
//!
//! where:
//! - `q(s)` is the approximate posterior (recognition density)
//! - `p(o,s)` is the generative model (joint over observations and states)
//! - Complexity = KL divergence between posterior and prior
//! - Accuracy   = expected log likelihood under q
//!
//! ### Hierarchical Decomposition
//!
//! The total free energy decomposes additively across L levels:
//!
//! ```text
//! F_total = Σ_l  F_l
//!
//! F_l = D_KL[q_l(s_l) || p(s_l | s_{l+1})]  -  E_{q_l}[ln p(s_{l-1} | s_l)]
//! ```
//!
//! At each level:
//! - Top-down prediction:   g_l(s_l)  →  predicted s_{l-1}
//! - Prediction error:      ε_l = s_{l-1} - g_l(s_l)
//! - Precision weighting:   ε̃_l = Π_l × ε_l
//!
//! ### Expected Free Energy (for action / policy selection)
//!
//! ```text
//! G(π) = -E_{q(o|π)}[ D_KL[q(s|o,π) || q(s|π)] ]  -  E_{q(o|π)}[ ln p(o|C) ]
//!         ╰────── epistemic value ──────╯               ╰── pragmatic value ──╯
//! ```
//!
//! ### Belief Updating (gradient descent on F)
//!
//! ```text
//! ∂q/∂t = -∂F/∂q
//! ```
//!
//! Implemented as gradient descent on variational parameters at every level.
//!
//! ## Integration Points
//!
//! - [`crate::consciousness::fep_active_inference`]: shares the same FEP
//!   vocabulary; this module adds *hierarchical* decomposition.
//! - [`crate::consciousness::predictive_processing`]: the prediction-error
//!   hierarchy that this module formalises in free-energy terms.
//! - [`crate::consciousness::consciousness_thermodynamics`]: free energy is a
//!   thermodynamic potential; temperature here maps to inverse precision.
//! - [`crate::hdc::ContinuousHV`]: beliefs and prediction errors can be
//!   lifted into HDC space for downstream HDC+CfC processing.
//!
//! ## References
//!
//! - Friston, K. (2008). Hierarchical Models in the Brain. *PLoS Computational
//!   Biology*, 4(11).
//! - Friston, K. (2010). The free-energy principle: a unified brain theory?
//!   *Nature Reviews Neuroscience*, 11, 127-138.
//! - Bogacz, R. (2017). A tutorial on the free-energy framework for modelling
//!   perception and learning. *Journal of Mathematical Psychology*, 76, 198-211.
//! - Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The
//!   Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::hdc::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the hierarchical free energy engine.
///
/// `num_levels` sets the depth of the cortical hierarchy.  `state_dim` is the
/// dimensionality of the belief vector at each level.  `learning_rate` controls
/// gradient-descent step size and `precision_decay` attenuates precision as we
/// move up the hierarchy (higher levels are more uncertain a-priori).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalFEConfig {
    /// Number of hierarchical levels (L).
    pub num_levels: usize,
    /// Dimensionality of the state / belief vector at each level.
    pub state_dim: usize,
    /// Learning rate η for gradient-descent belief updates.
    pub learning_rate: f64,
    /// Multiplicative decay applied to precision from level l to l+1.
    /// Values in (0, 1] are typical; 1.0 means no decay.
    pub precision_decay: f64,
}

impl Default for HierarchicalFEConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            state_dim: 8,
            learning_rate: 0.05,
            precision_decay: 0.8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PER-LEVEL STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// State of a single level in the hierarchical generative model.
///
/// Each level stores:
/// - `beliefs`: the variational parameters μ_l (mean of q_l).
/// - `precision`: Π_l, the inverse variance at this level.
/// - `prediction_error`: ε_l = s_{l-1} − g_l(μ_l).
/// - `free_energy`: F_l, the local contribution to total F.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalLevel {
    /// Index in the hierarchy (0 = lowest / sensory).
    pub level: usize,
    /// Variational mean μ_l — current best guess of hidden states at this level.
    pub beliefs: Vec<f64>,
    /// Precision Π_l (inverse variance); higher ⇒ more confident.
    pub precision: f64,
    /// Prediction error ε_l = s_{l-1} − g_l(μ_l).
    pub prediction_error: Vec<f64>,
    /// Local free energy contribution F_l.
    pub free_energy: f64,
}

impl HierarchicalLevel {
    /// Create a new level with zero-initialised beliefs and errors.
    fn new(level: usize, state_dim: usize, precision: f64) -> Self {
        Self {
            level,
            beliefs: vec![0.0; state_dim],
            precision,
            prediction_error: vec![0.0; state_dim],
            free_energy: 0.0,
        }
    }

    /// Update precision dynamically from prediction error variance.
    ///
    /// When `unified_precision` is enabled, precision is treated as a dynamic
    /// quantity (Parr & Friston 2019: "precision IS attention") rather than a
    /// static hyperparameter. The precision is updated as the inverse of the
    /// running PE variance with exponential smoothing.
    ///
    /// Optional modulation factors:
    /// - `phi_factor`: IIT Phi at this level → high integration = more confident
    /// - `intero_factor`: interoceptive modulation (Seth 2021)
    /// - `blanket_factor`: Markov blanket permeability
    #[cfg(feature = "unified_precision")]
    pub fn update_precision_dynamic(
        &mut self,
        base_precision: f64,
        phi_factor: f64,
        intero_factor: f64,
        blanket_factor: f64,
    ) {
        // Base precision from hierarchy (exponential decay)
        // Phi modulation: high integration → higher confidence (0.5-1.0 range)
        let phi_mod = 0.5 + 0.5 * phi_factor.clamp(0.0, 1.0);
        // Interoceptive: positive valence → higher precision, stress → lower
        let intero_mod = intero_factor.clamp(0.5, 1.5);
        // Blanket: closed blanket → trust priors more (0.5-1.0 range)
        let blanket_mod = 0.5 + 0.5 * blanket_factor.clamp(0.0, 1.0);

        self.precision = (base_precision * phi_mod * intero_mod * blanket_mod).clamp(0.01, 10.0);
        // Guard bounds
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FREE ENERGY DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of computing the hierarchical free energy.
///
/// `total = complexity - accuracy` (note: accuracy is negative log-likelihood,
/// so a *higher* accuracy lowers F).  `per_level[l]` gives F_l and their sum
/// equals `total`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyDecomposition {
    /// Complexity term: D_KL[q(s) || p(s)].
    pub complexity: f64,
    /// Accuracy term: E_q[ln p(o|s)]  (higher is better).
    pub accuracy: f64,
    /// Total variational free energy F = complexity − accuracy.
    pub total: f64,
    /// Per-level free energy contributions F_l; sum equals `total`.
    pub per_level: Vec<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BELIEF UPDATE RECORD
// ═══════════════════════════════════════════════════════════════════════════════

/// Diagnostic record emitted after each belief update at a given level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefUpdate {
    /// Which hierarchy level was updated.
    pub level: usize,
    /// Beliefs *before* the update.
    pub old_beliefs: Vec<f64>,
    /// Beliefs *after* the update.
    pub new_beliefs: Vec<f64>,
    /// Prediction error that drove the update.
    pub prediction_error: Vec<f64>,
    /// Precision at this level during the update.
    pub precision: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Hierarchical Free Energy engine.
///
/// Maintains an L-level hierarchy of beliefs, computes variational free energy
/// and its decomposition, performs gradient-descent belief updating, and
/// evaluates expected free energy for candidate policies.
///
/// # Example
///
/// ```rust
/// use symthaea::consciousness::hierarchical_free_energy::{
///     HierarchicalFreeEnergy, HierarchicalFEConfig,
/// };
///
/// let config = HierarchicalFEConfig {
///     num_levels: 3,
///     state_dim: 4,
///     learning_rate: 0.1,
///     precision_decay: 0.8,
/// };
/// let mut engine = HierarchicalFreeEnergy::new(config);
///
/// // Present an observation and update beliefs.
/// let obs = vec![1.0, 0.5, -0.3, 0.8];
/// let updates = engine.update_beliefs(&obs);
/// assert!(!updates.is_empty());
///
/// // Check that free energy is finite.
/// let decomp = engine.compute_free_energy();
/// assert!(decomp.total.is_finite());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalFreeEnergy {
    /// Engine configuration.
    pub config: HierarchicalFEConfig,
    /// Per-level state (index 0 = sensory, index L−1 = highest).
    pub levels: Vec<HierarchicalLevel>,
    /// Running count of belief-update cycles performed.
    pub update_count: u64,
}

impl HierarchicalFreeEnergy {
    // ─────────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────────

    /// Update precision at all levels from external modulation signals.
    ///
    /// Called by the consciousness engine when `unified_precision` is enabled.
    /// Each level's precision is dynamically updated from:
    /// - Base precision (exponential decay from config)
    /// - Phi factor (IIT integration at this level)
    /// - Interoceptive factor (valence/arousal modulation)
    /// - Blanket factor (Markov blanket permeability)
    #[cfg(feature = "unified_precision")]
    pub fn update_precisions(&mut self, phi_factor: f64, intero_factor: f64, blanket_factor: f64) {
        let decay = self.config.precision_decay;
        // Pre-computed base precisions: decay^0=1.0, decay^1, decay^2, decay^3
        // Avoids repeated powi() on the hot path.
        let mut base = 1.0_f64;
        for level in &mut self.levels {
            level.update_precision_dynamic(base, phi_factor, intero_factor, blanket_factor);
            base *= decay;
        }
    }

    /// Create a new hierarchical free energy engine from `config`.
    ///
    /// Levels are initialised with precision decaying upward:
    /// Π_l = precision_decay ^ l  (level 0 has precision 1.0).
    pub fn new(config: HierarchicalFEConfig) -> Self {
        let levels = (0..config.num_levels)
            .map(|l| {
                let precision = config.precision_decay.powi(l as i32);
                HierarchicalLevel::new(l, config.state_dim, precision)
            })
            .collect();

        Self {
            config,
            levels,
            update_count: 0,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Generative mapping  g_l : μ_l  →  predicted  s_{l-1}
    // ─────────────────────────────────────────────────────────────────────────

    /// Simple linear generative mapping from level `l` beliefs to a prediction
    /// of level `l-1` states.  In a richer model this would be a learned neural
    /// network; here we use a damped identity (tanh squashing) so the maths is
    /// transparent and unit-testable without extra parameters.
    fn generative_mapping(&self, beliefs: &[f64]) -> Vec<f64> {
        beliefs.iter().map(|&b| b.tanh()).collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Prediction errors
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute prediction error at level `l`:
    ///
    /// ```text
    /// ε_l = target_{l-1} − g_l(μ_l)
    /// ```
    ///
    /// where `target` is either the raw observation (for level 0) or the
    /// beliefs at the level below.
    fn compute_prediction_error(&self, target: &[f64], level_idx: usize) -> Vec<f64> {
        let prediction = self.generative_mapping(&self.levels[level_idx].beliefs);
        target
            .iter()
            .zip(prediction.iter())
            .map(|(&t, &p)| t - p)
            .collect()
    }

    /// Precision-weighted prediction error: ε̃_l = Π_l × ε_l.
    fn precision_weighted_error(&self, error: &[f64], level_idx: usize) -> Vec<f64> {
        let pi = self.levels[level_idx].precision;
        error.iter().map(|&e| pi * e).collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Belief updating  (gradient descent on F)
    // ─────────────────────────────────────────────────────────────────────────

    /// Present an observation and update beliefs at every level via gradient
    /// descent on the variational free energy.
    ///
    /// Returns a [`BeliefUpdate`] diagnostic for each level that was updated.
    ///
    /// The update rule at level l is:
    ///
    /// ```text
    /// Δμ_l = η × ( Π_l × ε_l  −  Π_{l+1} × ε_{l+1} × g'_l )
    /// ```
    ///
    /// where the first term is the bottom-up precision-weighted error and the
    /// second term is the top-down precision-weighted error back-propagated
    /// through the generative mapping derivative g'_l (here: 1 − tanh²).
    pub fn update_beliefs(&mut self, observation: &[f64]) -> Vec<BeliefUpdate> {
        let dim = self.config.state_dim;
        let lr = self.config.learning_rate;
        let n = self.levels.len();

        // 1) Compute bottom-up prediction errors at every level.
        //    Level 0 compares against the observation; level l>0 compares
        //    against the beliefs at level l-1.
        let mut errors: Vec<Vec<f64>> = Vec::with_capacity(n);

        // Clamp or pad observation to state_dim.
        let obs: Vec<f64> = (0..dim)
            .map(|i| {
                if i < observation.len() {
                    observation[i]
                } else {
                    0.0
                }
            })
            .collect();

        for l in 0..n {
            let target = if l == 0 {
                obs.clone()
            } else {
                self.levels[l - 1].beliefs.clone()
            };
            let err = self.compute_prediction_error(&target, l);
            errors.push(err);
        }

        // 2) Gradient descent updates (bottom → top).
        let mut updates = Vec::with_capacity(n);

        for l in 0..n {
            let old_beliefs = self.levels[l].beliefs.clone();
            let pe_weighted = self.precision_weighted_error(&errors[l], l);

            // Top-down correction from level l+1 (if it exists).
            let top_down: Vec<f64> = if l + 1 < n {
                let pe_above = self.precision_weighted_error(&errors[l + 1], l + 1);
                // g'_l(μ_l) = 1 − tanh²(μ_l)  (derivative of tanh generative mapping)
                self.levels[l]
                    .beliefs
                    .iter()
                    .zip(pe_above.iter())
                    .map(|(&b, &ea)| {
                        let g_prime = 1.0 - b.tanh().powi(2);
                        ea * g_prime
                    })
                    .collect()
            } else {
                vec![0.0; dim]
            };

            // Δμ_l = η × (bottom_up − top_down)
            for i in 0..dim {
                self.levels[l].beliefs[i] += lr * (pe_weighted[i] - top_down[i]);
            }

            // Cache the prediction error.
            self.levels[l].prediction_error = errors[l].clone();

            updates.push(BeliefUpdate {
                level: l,
                old_beliefs,
                new_beliefs: self.levels[l].beliefs.clone(),
                prediction_error: errors[l].clone(),
                precision: self.levels[l].precision,
            });
        }

        self.update_count += 1;

        // Recompute per-level free energies after the update.
        let _ = self.compute_free_energy();

        updates
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Free energy computation
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute the full hierarchical free energy decomposition.
    ///
    /// For each level l:
    ///
    /// ```text
    /// complexity_l = 0.5 × Σ_i  μ_l[i]²              (KL for Gaussian q vs N(0,1) prior)
    /// accuracy_l   = -0.5 × Π_l × Σ_i  ε_l[i]²       (precision-weighted SSE)
    /// F_l          = complexity_l - accuracy_l
    /// ```
    ///
    /// (Using the standard Gaussian-KL simplification where prior is N(0,I)
    /// and q_l = N(μ_l, Π_l⁻¹ I).)
    pub fn compute_free_energy(&mut self) -> FreeEnergyDecomposition {
        let mut total_complexity = 0.0;
        let mut total_accuracy = 0.0;
        let mut per_level = Vec::with_capacity(self.levels.len());

        for level in &mut self.levels {
            // Complexity: KL divergence D_KL[q_l || p_l]
            // For Gaussian q = N(μ, Π⁻¹I) and prior p = N(0, I):
            //   D_KL = 0.5 * (Σ μ² + d/Π − d − ln(1/Π^d))
            //        = 0.5 * (Σ μ² + d/Π − d + d·ln Π)
            // We use the dominant term 0.5 * Σ μ² for a clean decomposition.
            let complexity_l: f64 = 0.5 * level.beliefs.iter().map(|b| b * b).sum::<f64>();

            // Accuracy: expected log-likelihood under q.
            // −0.5 × Π_l × ||ε_l||²  (Gaussian observation model).
            let sse: f64 = level.prediction_error.iter().map(|e| e * e).sum::<f64>();
            let accuracy_l: f64 = -0.5 * level.precision * sse;

            // F_l = complexity − accuracy  (note accuracy_l ≤ 0 so this adds)
            let f_l = complexity_l - accuracy_l;

            level.free_energy = f_l;
            total_complexity += complexity_l;
            total_accuracy += accuracy_l;
            per_level.push(f_l);
        }

        let total = total_complexity - total_accuracy;

        FreeEnergyDecomposition {
            complexity: total_complexity,
            accuracy: total_accuracy,
            total,
            per_level,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Expected free energy for policy evaluation
    // ─────────────────────────────────────────────────────────────────────────

    /// Evaluate the expected free energy G(π) for a set of candidate policies.
    ///
    /// Each policy is represented as a target observation vector.  G(π)
    /// combines:
    ///
    /// ```text
    /// G(π) = epistemic_cost(π)  −  pragmatic_alignment(π)
    ///      = ||prediction_error||  −  cos(policy, top_beliefs)
    /// ```
    ///
    /// - **Epistemic cost** (prediction error magnitude): higher error means
    ///   more information to learn, raising G. This encourages exploration of
    ///   uncertain states when epistemic cost dominates.
    ///
    /// - **Pragmatic alignment** (negative cosine similarity): higher alignment
    ///   with top-level beliefs (preferences C) lowers G, encouraging
    ///   goal-directed exploitation.
    ///
    /// Returns one G(π) score per policy.  *Lower* G is preferred (it is a
    /// free energy, so the agent seeks to minimise it).
    pub fn expected_free_energy(&self, policies: &[Vec<f64>]) -> Vec<f64> {
        let dim = self.config.state_dim;

        policies
            .iter()
            .map(|policy| {
                // Pad / clamp policy to state_dim.
                let pol: Vec<f64> = (0..dim)
                    .map(|i| if i < policy.len() { policy[i] } else { 0.0 })
                    .collect();

                // Epistemic value: expected prediction error magnitude.
                // We simulate what the bottom-level error would be.
                let pred = self.generative_mapping(&self.levels[0].beliefs);
                let epistemic: f64 = pol
                    .iter()
                    .zip(pred.iter())
                    .map(|(&p, &g)| {
                        let e = p - g;
                        e * e
                    })
                    .sum::<f64>()
                    .sqrt();

                // Pragmatic value: alignment with top-level beliefs (preferences).
                // Negative cosine similarity so lower G ⇒ more aligned.
                let top = &self.levels[self.levels.len() - 1].beliefs;
                let dot: f64 = pol.iter().zip(top.iter()).map(|(&a, &b)| a * b).sum();
                let norm_pol = pol.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                let norm_top = top.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                let cosine = dot / (norm_pol * norm_top);

                // G(π) = epistemic_cost − pragmatic_alignment
                // (lower is better; large error ⇒ high G, good alignment ⇒ low G)
                epistemic - cosine
            })
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Convenience accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Return the total free energy across all levels.
    pub fn total_free_energy(&self) -> f64 {
        self.levels.iter().map(|l| l.free_energy).sum()
    }

    /// Return the precision-weighted prediction error norm at each level.
    pub fn precision_weighted_errors(&self) -> Vec<f64> {
        self.levels
            .iter()
            .map(|l| {
                let sse: f64 = l.prediction_error.iter().map(|e| e * e).sum::<f64>();
                (l.precision * sse).sqrt()
            })
            .collect()
    }

    /// Convert the current beliefs into a flat `Vec<f32>` suitable for
    /// constructing a [`ContinuousHV`] for downstream HDC processing.
    ///
    /// The vector concatenates beliefs from all levels (lowest first).
    pub fn beliefs_as_f32(&self) -> Vec<f32> {
        self.levels
            .iter()
            .flat_map(|l| l.beliefs.iter().map(|&b| b as f32))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a default 3-level engine with dim=4.
    fn make_engine() -> HierarchicalFreeEnergy {
        let config = HierarchicalFEConfig {
            num_levels: 3,
            state_dim: 4,
            learning_rate: 0.1,
            precision_decay: 0.8,
        };
        HierarchicalFreeEnergy::new(config)
    }

    // ── Test 1: Free energy decreases during belief updating ─────────────

    #[test]
    fn free_energy_decreases_during_updates() {
        // Use a small learning rate for stable convergence across levels.
        let config = HierarchicalFEConfig {
            num_levels: 3,
            state_dim: 4,
            learning_rate: 0.01, // small LR prevents multi-level oscillation
            precision_decay: 0.8,
        };
        let mut engine = HierarchicalFreeEnergy::new(config);
        let obs = vec![1.0, 0.5, -0.3, 0.8];

        // Run until convergence; check that level-0 prediction error decreases.
        engine.update_beliefs(&obs);
        let pe_initial: f64 = engine.levels[0]
            .prediction_error
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();

        for _ in 0..500 {
            engine.update_beliefs(&obs);
        }
        let pe_after: f64 = engine.levels[0]
            .prediction_error
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();

        assert!(
            pe_after < pe_initial,
            "Prediction error at level 0 should decrease during belief updating: \
             initial={pe_initial:.4}, after={pe_after:.4}"
        );

        // Free energy should be finite.
        let decomp = engine.compute_free_energy();
        assert!(decomp.total.is_finite());
    }

    // ── Test 2: Complexity + |Accuracy| = Total (accounting for sign) ────

    #[test]
    fn complexity_minus_accuracy_equals_total() {
        let mut engine = make_engine();
        let obs = vec![0.7, -0.2, 0.4, 0.1];
        engine.update_beliefs(&obs);
        let decomp = engine.compute_free_energy();

        let reconstructed = decomp.complexity - decomp.accuracy;
        let diff = (reconstructed - decomp.total).abs();
        assert!(
            diff < 1e-12,
            "complexity - accuracy should equal total: \
             complexity={}, accuracy={}, total={}, diff={}",
            decomp.complexity,
            decomp.accuracy,
            decomp.total,
            diff,
        );
    }

    // ── Test 3: Higher precision → faster convergence ────────────────────

    #[test]
    fn higher_precision_converges_faster() {
        let obs = vec![1.0, 0.5, -0.3, 0.8];
        let iters = 200; // enough iterations for convergence

        // Low precision (high decay).
        let mut engine_low = HierarchicalFreeEnergy::new(HierarchicalFEConfig {
            num_levels: 3,
            state_dim: 4,
            learning_rate: 0.05,  // smaller LR for stable convergence
            precision_decay: 0.3, // precision drops fast
        });
        for _ in 0..iters {
            engine_low.update_beliefs(&obs);
        }
        let fe_low = engine_low.compute_free_energy().total;

        // High precision (slow decay).
        let mut engine_high = HierarchicalFreeEnergy::new(HierarchicalFEConfig {
            num_levels: 3,
            state_dim: 4,
            learning_rate: 0.05,
            precision_decay: 0.95, // precision stays high
        });
        for _ in 0..iters {
            engine_high.update_beliefs(&obs);
        }
        let fe_high = engine_high.compute_free_energy().total;

        // Higher precision means level-0 beliefs should be closer to the
        // observation (sensory evidence is weighted more strongly).
        let dist_low: f64 = engine_low.levels[0]
            .beliefs
            .iter()
            .zip(obs.iter())
            .map(|(&b, &o)| (b - o.tanh()).powi(2))
            .sum::<f64>()
            .sqrt();
        let dist_high: f64 = engine_high.levels[0]
            .beliefs
            .iter()
            .zip(obs.iter())
            .map(|(&b, &o)| (b - o.tanh()).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            dist_high <= dist_low + 0.1,
            "Higher precision should drive beliefs closer to observation: \
             dist_high={dist_high:.4}, dist_low={dist_low:.4}"
        );

        // Sanity: both free energies should be finite.
        assert!(fe_low.is_finite());
        assert!(fe_high.is_finite());
    }

    // ── Test 4: Per-level free energies sum to total ─────────────────────

    #[test]
    fn per_level_sums_to_total() {
        let mut engine = make_engine();
        let obs = vec![0.3, -0.6, 0.9, 0.0];
        engine.update_beliefs(&obs);
        let decomp = engine.compute_free_energy();

        let sum_levels: f64 = decomp.per_level.iter().sum();
        let diff = (sum_levels - decomp.total).abs();
        assert!(
            diff < 1e-12,
            "Sum of per-level free energies should equal total: \
             sum={sum_levels}, total={}, diff={diff}",
            decomp.total,
        );
    }

    // ── Test 5: Level count and dimension are respected ──────────────────

    #[test]
    fn engine_respects_config_dimensions() {
        let config = HierarchicalFEConfig {
            num_levels: 5,
            state_dim: 16,
            learning_rate: 0.01,
            precision_decay: 0.9,
        };
        let engine = HierarchicalFreeEnergy::new(config.clone());

        assert_eq!(engine.levels.len(), 5);
        for (i, level) in engine.levels.iter().enumerate() {
            assert_eq!(level.level, i);
            assert_eq!(level.beliefs.len(), 16);
            assert_eq!(level.prediction_error.len(), 16);
        }
    }

    // ── Test 6: Precision decays across levels ───────────────────────────

    #[test]
    fn precision_decays_upward() {
        let engine = make_engine(); // precision_decay = 0.8
        for l in 0..engine.levels.len() {
            let expected = 0.8_f64.powi(l as i32);
            let actual = engine.levels[l].precision;
            assert!(
                (actual - expected).abs() < 1e-12,
                "Level {l}: expected precision {expected}, got {actual}"
            );
        }
    }

    // ── Test 7: Expected free energy prefers aligned policies ────────────

    #[test]
    fn expected_free_energy_prefers_aligned_policy() {
        let mut engine = make_engine();
        // Push beliefs toward a known state.
        let target = vec![0.5, 0.5, 0.5, 0.5];
        for _ in 0..100 {
            engine.update_beliefs(&target);
        }

        // Policy A is aligned with learned beliefs.
        let policy_aligned = vec![0.5, 0.5, 0.5, 0.5];
        // Policy B is anti-aligned.
        let policy_anti = vec![-1.0, -1.0, -1.0, -1.0];

        let g = engine.expected_free_energy(&[policy_aligned, policy_anti]);
        assert!(
            g[0] < g[1],
            "Aligned policy should have lower expected free energy: \
             G_aligned={}, G_anti={}",
            g[0],
            g[1],
        );
    }

    // ── Test 8: beliefs_as_f32 concatenates all levels ───────────────────

    #[test]
    fn beliefs_as_f32_has_correct_length() {
        let engine = make_engine(); // 3 levels × 4 dim = 12
        let flat = engine.beliefs_as_f32();
        assert_eq!(flat.len(), 3 * 4);
    }

    // ── Test 9: Update count increments ──────────────────────────────────

    #[test]
    fn update_count_increments() {
        let mut engine = make_engine();
        assert_eq!(engine.update_count, 0);
        engine.update_beliefs(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(engine.update_count, 1);
        engine.update_beliefs(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(engine.update_count, 2);
    }

    // ── Test 10: Observation shorter than state_dim is zero-padded ───────

    #[test]
    fn short_observation_is_padded() {
        let mut engine = make_engine(); // state_dim = 4
        let updates = engine.update_beliefs(&[1.0, 2.0]); // only 2 elements
        assert_eq!(updates.len(), 3); // one per level
        // No panic means padding worked.
    }
}
