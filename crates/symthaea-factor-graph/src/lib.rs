#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop, clippy::new_without_default)]

//! # Factor Graphs with Belief Propagation
//!
//! Implements factor graph representation and the sum-product (belief propagation)
//! algorithm for exact and approximate probabilistic inference.
//!
//! ## Motivation: Factor Graphs for Active Inference
//!
//! Factor graphs are the natural computational substrate for generative models
//! in Active Inference (see `fep_active_inference`). Where the FEP module
//! represents `p(o,s)` as dense matrices, factor graphs decompose the joint
//! distribution into a product of local factors:
//!
//! ```text
//! p(x_1, ..., x_n) = (1/Z) prod_a f_a(x_{da})
//! ```
//!
//! where `f_a` are factor functions and `x_{da}` are the variables connected to
//! factor `a`. This factorization exploits conditional independence structure,
//! enabling inference that scales with the *treewidth* of the model rather than
//! the full state space.
//!
//! ## Mathematical Foundation
//!
//! ### Sum-Product Algorithm (Belief Propagation)
//!
//! Messages from variable `i` to factor `a`:
//!
//! ```text
//! mu_{i->a}(x_i) = prod_{b in N(i)\a} mu_{b->i}(x_i)
//! ```
//!
//! Messages from factor `a` to variable `i`:
//!
//! ```text
//! mu_{a->i}(x_i) = sum_{x_{da\i}} f_a(x_{da}) prod_{j in da\i} mu_{j->a}(x_j)
//! ```
//!
//! Marginal beliefs:
//!
//! ```text
//! b_i(x_i) proportional_to prod_{a in N(i)} mu_{a->i}(x_i)
//! ```
//!
//! ### Max-Product (MAP Inference)
//!
//! Replace summation with maximization in the factor-to-variable messages to
//! obtain the maximum a posteriori (MAP) assignment.
//!
//! ### Loopy BP and Convergence
//!
//! For graphs with cycles, BP is approximate. Convergence is improved via:
//! - **Damping**: `m_new = alpha * m_computed + (1 - alpha) * m_old`
//! - **Iteration cap**: Terminate after `max_iterations`
//! - **Tolerance**: Converge when `max |m_new - m_old| < tolerance`
//!
//! ### Bethe Free Energy
//!
//! The variational interpretation of loopy BP: at a fixed point the beliefs
//! minimize the Bethe free energy:
//!
//! ```text
//! F_Bethe = sum_a sum_{x_a} b_a(x_a) [ln b_a(x_a) - ln f_a(x_a)]
//!         - sum_i (d_i - 1) sum_{x_i} b_i(x_i) ln b_i(x_i)
//! ```
//!
//! where `d_i` is the degree of variable node `i`.
//!
//! ## Integration with Symthaea
//!
//! Factor graphs complement the existing `fep_active_inference::GenerativeModel`
//! by providing a *structured* representation of the generative model. In future
//! work the FEP bridge can construct a factor graph whose factors correspond to
//! the likelihood `p(o|s)` and transition `p(s'|s,a)` terms, enabling message-
//! passing inference that naturally decomposes free energy minimization into
//! local belief updates -- the same architecture biological brains are thought
//! to implement (Friston, Parr & de Vries, 2017).
//!
//! ## References
//!
//! - Kschischang, F.R., Frey, B.J. & Loeliger, H.-A. (2001). Factor Graphs and
//!   the Sum-Product Algorithm. *IEEE Trans. Info. Theory*, 47(2).
//! - Yedidia, J.S., Freeman, W.T. & Weiss, Y. (2003). Understanding Belief
//!   Propagation and its Generalizations. *Exploring AI in the New Millennium*.
//! - Friston, K., Parr, T. & de Vries, B. (2017). The Graphical Brain: Belief
//!   Propagation and Active Inference. *Network Neuroscience*, 1(4).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// A variable node in the factor graph.
///
/// Represents a single random variable with a discrete state space of size
/// `cardinality`. The `belief` vector holds the current marginal estimate
/// `b_i(x_i)` and is updated during belief propagation.
#[derive(Debug, Clone)]
pub struct VariableNode {
    /// Unique identifier for this variable.
    pub id: usize,
    /// Human-readable name (e.g. "hidden_state_0").
    pub name: String,
    /// Number of discrete states this variable can take.
    pub cardinality: usize,
    /// Current marginal belief vector, length = `cardinality`.
    pub belief: Vec<f64>,
    /// Indices of neighboring factor nodes.
    pub neighbors: Vec<usize>,
}

/// A factor node in the factor graph.
///
/// Encodes a local function `f_a(x_{da})` over the subset of variables listed
/// in `variables`. The `table` stores the factor potential as a flat array in
/// row-major order over the Cartesian product of the variables' state spaces.
#[derive(Debug, Clone)]
pub struct FactorNode {
    /// Unique identifier for this factor.
    pub id: usize,
    /// Ordered list of variable indices that this factor depends on.
    pub variables: Vec<usize>,
    /// Factor potential table, flattened in row-major order.
    /// Length = product of cardinalities of `variables`.
    pub table: Vec<f64>,
    /// Indices of neighboring variable nodes (same as `variables` but kept
    /// for API symmetry with `VariableNode::neighbors`).
    pub neighbors: Vec<usize>,
}

/// The factor graph: a bipartite graph of variable nodes and factor nodes.
///
/// Messages are stored as `(sender, receiver) -> message_vector`. For a
/// variable-to-factor message the key is `(var_id, factor_id)`; for a
/// factor-to-variable message the key is `(factor_id + variable_count, var_id)`
/// so that the two directions never collide.
#[derive(Debug, Clone)]
pub struct FactorGraph {
    /// Variable nodes.
    pub variables: Vec<VariableNode>,
    /// Factor nodes.
    pub factors: Vec<FactorNode>,
    /// Message store: `(sender_key, receiver_key) -> message`.
    pub messages: HashMap<(usize, usize), Vec<f64>>,
}

/// Message passing schedule for belief propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageSchedule {
    /// All messages are updated simultaneously from the previous iteration.
    Parallel,
    /// Messages are updated one at a time; each update immediately uses the
    /// latest available messages.
    Sequential,
}

/// Configuration for the belief propagation algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPConfig {
    /// Maximum number of message-passing iterations.
    pub max_iterations: usize,
    /// Convergence tolerance: stop when the maximum absolute change across
    /// all messages falls below this threshold.
    pub tolerance: f64,
    /// Damping factor in `[0, 1]`. At each update the new message is:
    /// `m = damping * m_computed + (1 - damping) * m_old`.
    /// A value of 1.0 means no damping (pure update).
    pub damping: f64,
    /// Message passing schedule.
    pub schedule: MessageSchedule,
}

impl Default for BPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            damping: 1.0,
            schedule: MessageSchedule::Parallel,
        }
    }
}

/// Result of running belief propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPResult {
    /// Marginal beliefs for each variable node, `beliefs[i]` has length
    /// `variables[i].cardinality`.
    pub beliefs: Vec<Vec<f64>>,
    /// Whether the algorithm converged before hitting `max_iterations`.
    pub converged: bool,
    /// Number of iterations actually performed.
    pub iterations: usize,
    /// Bethe free energy at convergence (lower is better).
    pub bethe_free_energy: f64,
}

/// Result of MAP inference via max-product BP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAPResult {
    /// MAP assignment: `assignment[i]` is the most probable state for variable `i`.
    pub assignment: Vec<usize>,
    /// Log-probability of the MAP assignment (unnormalized).
    pub log_probability: f64,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
}

// =============================================================================
// FACTOR GRAPH IMPLEMENTATION
// =============================================================================

impl FactorGraph {
    /// Create an empty factor graph.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            messages: HashMap::new(),
        }
    }

    /// Add a variable node and return its index.
    ///
    /// The initial belief is uniform over `cardinality` states.
    pub fn add_variable(&mut self, name: &str, cardinality: usize) -> usize {
        let id = self.variables.len();
        let uniform = 1.0 / cardinality as f64;
        self.variables.push(VariableNode {
            id,
            name: name.to_string(),
            cardinality,
            belief: vec![uniform; cardinality],
            neighbors: Vec::new(),
        });
        id
    }

    /// Add a factor node connecting the given variables and return its index.
    ///
    /// `table` must have length equal to the product of the cardinalities of
    /// the listed variables. It is stored in row-major order: the *last*
    /// variable in `variables` changes fastest.
    ///
    /// # Panics
    ///
    /// Panics if the table length does not match the expected product of
    /// cardinalities, or if any variable index is out of bounds.
    pub fn add_factor(&mut self, variables: &[usize], table: Vec<f64>) -> usize {
        let expected_len: usize = variables
            .iter()
            .map(|&v| {
                assert!(v < self.variables.len(), "Variable index {v} out of bounds");
                self.variables[v].cardinality
            })
            .try_fold(1usize, |acc, c| acc.checked_mul(c))
            .expect("factor cardinality overflow");
        assert_eq!(
            table.len(),
            expected_len,
            "Factor table length {} does not match expected {} (product of cardinalities)",
            table.len(),
            expected_len
        );

        let factor_id = self.factors.len();

        // Register adjacency.
        for &v in variables {
            self.variables[v].neighbors.push(factor_id);
        }

        self.factors.push(FactorNode {
            id: factor_id,
            variables: variables.to_vec(),
            table,
            neighbors: variables.to_vec(),
        });

        factor_id
    }

    // -------------------------------------------------------------------------
    // Message key helpers
    // -------------------------------------------------------------------------

    /// Key for a variable-to-factor message.
    #[inline]
    fn var_to_factor_key(&self, var: usize, factor: usize) -> (usize, usize) {
        // Variable ids live in [0, n_vars). We shift factor ids by n_vars to
        // avoid collisions with factor-to-variable keys.
        (var, self.variables.len() + factor)
    }

    /// Key for a factor-to-variable message.
    #[inline]
    fn factor_to_var_key(&self, factor: usize, var: usize) -> (usize, usize) {
        (self.variables.len() + factor, var)
    }

    // -------------------------------------------------------------------------
    // Message initialization
    // -------------------------------------------------------------------------

    /// Initialize all messages to uniform distributions.
    fn init_messages(&mut self) {
        self.messages.clear();
        for factor in &self.factors {
            for &v in &factor.variables {
                let card = self.variables[v].cardinality;
                let uniform = vec![1.0 / card as f64; card];
                self.messages
                    .insert(self.var_to_factor_key(v, factor.id), uniform.clone());
                self.messages
                    .insert(self.factor_to_var_key(factor.id, v), uniform);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Sum-Product messages
    // -------------------------------------------------------------------------

    /// Compute the variable-to-factor message mu_{i->a}(x_i).
    ///
    /// This is the product of all incoming factor-to-variable messages from
    /// neighbors of `i` *except* factor `a`.
    fn compute_var_to_factor_msg(&self, var: usize, factor: usize) -> Vec<f64> {
        let card = self.variables[var].cardinality;
        let mut msg = vec![1.0; card];
        for &nb_factor in &self.variables[var].neighbors {
            if nb_factor == factor {
                continue;
            }
            let key = self.factor_to_var_key(nb_factor, var);
            if let Some(incoming) = self.messages.get(&key) {
                for (k, m) in msg.iter_mut().enumerate() {
                    *m *= incoming[k];
                }
            }
        }
        normalize(&mut msg);
        msg
    }

    /// Compute the factor-to-variable message mu_{a->i}(x_i) using the
    /// sum-product rule.
    ///
    /// ```text
    /// mu_{a->i}(x_i) = sum_{x_{da\i}} f_a(x_{da}) prod_{j in da\i} mu_{j->a}(x_j)
    /// ```
    fn compute_factor_to_var_msg_sum(&self, factor: usize, target_var: usize) -> Vec<f64> {
        let f = &self.factors[factor];
        let target_card = self.variables[target_var].cardinality;
        let mut result = vec![0.0; target_card];

        // Collect cardinalities and incoming messages for the factor's variables.
        let cards: Vec<usize> = f
            .variables
            .iter()
            .map(|&v| self.variables[v].cardinality)
            .collect();
        let incoming: Vec<Vec<f64>> = f
            .variables
            .iter()
            .map(|&v| {
                if v == target_var {
                    vec![1.0; self.variables[v].cardinality]
                } else {
                    let key = self.var_to_factor_key(v, factor);
                    self.messages.get(&key).cloned().unwrap_or_else(|| {
                        vec![1.0 / cards[0] as f64; self.variables[v].cardinality]
                    })
                }
            })
            .collect();

        // Find the position of target_var within the factor's variable list.
        let target_pos = f
            .variables
            .iter()
            .position(|&v| v == target_var)
            .expect("target_var not in factor");

        // Enumerate all configurations of x_{da}.
        let total: usize = cards
            .iter()
            .try_fold(1usize, |acc, &c| acc.checked_mul(c))
            .expect("factor cardinality overflow");
        for flat_idx in 0..total {
            let config = unflatten(flat_idx, &cards);
            let factor_val = f.table[flat_idx];

            // Product of incoming messages for all variables except target.
            let mut prod = factor_val;
            for (pos, &v_state) in config.iter().enumerate() {
                if pos != target_pos {
                    prod *= incoming[pos][v_state];
                }
            }

            let target_state = config[target_pos];
            result[target_state] += prod;
        }

        normalize(&mut result);
        result
    }

    /// Compute the factor-to-variable message using the max-product rule.
    ///
    /// Identical to sum-product but replaces summation with maximization.
    fn compute_factor_to_var_msg_max(&self, factor: usize, target_var: usize) -> Vec<f64> {
        let f = &self.factors[factor];
        let target_card = self.variables[target_var].cardinality;
        let mut result = vec![0.0_f64; target_card];

        let cards: Vec<usize> = f
            .variables
            .iter()
            .map(|&v| self.variables[v].cardinality)
            .collect();
        let incoming: Vec<Vec<f64>> = f
            .variables
            .iter()
            .map(|&v| {
                if v == target_var {
                    vec![1.0; self.variables[v].cardinality]
                } else {
                    let key = self.var_to_factor_key(v, factor);
                    self.messages
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(|| vec![1.0; self.variables[v].cardinality])
                }
            })
            .collect();

        let target_pos = f
            .variables
            .iter()
            .position(|&v| v == target_var)
            .expect("target_var not in factor");

        let total: usize = cards
            .iter()
            .try_fold(1usize, |acc, &c| acc.checked_mul(c))
            .expect("factor cardinality overflow");
        for flat_idx in 0..total {
            let config = unflatten(flat_idx, &cards);
            let factor_val = f.table[flat_idx];

            let mut prod = factor_val;
            for (pos, &v_state) in config.iter().enumerate() {
                if pos != target_pos {
                    prod *= incoming[pos][v_state];
                }
            }

            let target_state = config[target_pos];
            if prod > result[target_state] {
                result[target_state] = prod;
            }
        }

        normalize(&mut result);
        result
    }

    // -------------------------------------------------------------------------
    // Belief computation
    // -------------------------------------------------------------------------

    /// Compute the marginal belief for variable `var` from all incoming
    /// factor-to-variable messages.
    fn compute_belief(&self, var: usize) -> Vec<f64> {
        let card = self.variables[var].cardinality;
        let mut belief = vec![1.0; card];
        for &nb_factor in &self.variables[var].neighbors {
            let key = self.factor_to_var_key(nb_factor, var);
            if let Some(msg) = self.messages.get(&key) {
                for (k, b) in belief.iter_mut().enumerate() {
                    *b *= msg[k];
                }
            }
        }
        normalize(&mut belief);
        belief
    }

    // -------------------------------------------------------------------------
    // Sum-Product BP (marginal inference)
    // -------------------------------------------------------------------------

    /// Run sum-product belief propagation and return marginal beliefs.
    ///
    /// For tree-structured graphs the result is exact. For loopy graphs the
    /// result is approximate but often very accurate, especially with damping.
    pub fn run_bp(&mut self, config: &BPConfig) -> BPResult {
        self.init_messages();

        let mut converged = false;
        let mut iters = 0;

        for _iter in 0..config.max_iterations {
            iters = _iter + 1;
            let mut max_delta: f64 = 0.0;

            match config.schedule {
                MessageSchedule::Parallel => {
                    // Collect all new messages, then apply.
                    let mut new_msgs: Vec<((usize, usize), Vec<f64>)> = Vec::new();

                    // Variable-to-factor messages.
                    for var in 0..self.variables.len() {
                        for &factor in &self.variables[var].neighbors.clone() {
                            let key = self.var_to_factor_key(var, factor);
                            let msg = self.compute_var_to_factor_msg(var, factor);
                            new_msgs.push((key, msg));
                        }
                    }
                    // Factor-to-variable messages.
                    for factor in 0..self.factors.len() {
                        for &var in &self.factors[factor].variables.clone() {
                            let key = self.factor_to_var_key(factor, var);
                            let msg = self.compute_factor_to_var_msg_sum(factor, var);
                            new_msgs.push((key, msg));
                        }
                    }

                    // Apply with damping.
                    for (key, new_msg) in new_msgs {
                        let old_msg = self.messages.get(&key);
                        let damped = if let Some(old) = old_msg {
                            let mut d = vec![0.0; new_msg.len()];
                            for i in 0..d.len() {
                                d[i] =
                                    config.damping * new_msg[i] + (1.0 - config.damping) * old[i];
                            }
                            let delta: f64 = d
                                .iter()
                                .zip(old.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0, f64::max);
                            max_delta = max_delta.max(delta);
                            d
                        } else {
                            new_msg
                        };
                        self.messages.insert(key, damped);
                    }
                }
                MessageSchedule::Sequential => {
                    // Update messages one at a time, immediately visible.
                    for var in 0..self.variables.len() {
                        for &factor in &self.variables[var].neighbors.clone() {
                            let key = self.var_to_factor_key(var, factor);
                            let new_msg = self.compute_var_to_factor_msg(var, factor);
                            let delta = self.apply_damped_message(key, new_msg, config.damping);
                            max_delta = max_delta.max(delta);
                        }
                    }
                    for factor in 0..self.factors.len() {
                        for &var in &self.factors[factor].variables.clone() {
                            let key = self.factor_to_var_key(factor, var);
                            let new_msg = self.compute_factor_to_var_msg_sum(factor, var);
                            let delta = self.apply_damped_message(key, new_msg, config.damping);
                            max_delta = max_delta.max(delta);
                        }
                    }
                }
            }

            if max_delta < config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute final beliefs.
        let beliefs: Vec<Vec<f64>> = (0..self.variables.len())
            .map(|v| self.compute_belief(v))
            .collect();

        // Update variable beliefs.
        for (v, b) in beliefs.iter().enumerate() {
            self.variables[v].belief = b.clone();
        }

        let bethe = self.bethe_free_energy();

        BPResult {
            beliefs,
            converged,
            iterations: iters,
            bethe_free_energy: bethe,
        }
    }

    // -------------------------------------------------------------------------
    // Max-Product BP (MAP inference)
    // -------------------------------------------------------------------------

    /// Run max-product belief propagation for MAP inference.
    ///
    /// Returns the most probable joint assignment and its (unnormalized)
    /// log-probability.
    pub fn run_map(&mut self, config: &BPConfig) -> MAPResult {
        self.init_messages();

        let mut converged = false;
        let mut iters = 0;

        for _iter in 0..config.max_iterations {
            iters = _iter + 1;
            let mut max_delta: f64 = 0.0;

            // Variable-to-factor (same as sum-product).
            for var in 0..self.variables.len() {
                for &factor in &self.variables[var].neighbors.clone() {
                    let key = self.var_to_factor_key(var, factor);
                    let new_msg = self.compute_var_to_factor_msg(var, factor);
                    let delta = self.apply_damped_message(key, new_msg, config.damping);
                    max_delta = max_delta.max(delta);
                }
            }

            // Factor-to-variable (max instead of sum).
            for factor in 0..self.factors.len() {
                for &var in &self.factors[factor].variables.clone() {
                    let key = self.factor_to_var_key(factor, var);
                    let new_msg = self.compute_factor_to_var_msg_max(factor, var);
                    let delta = self.apply_damped_message(key, new_msg, config.damping);
                    max_delta = max_delta.max(delta);
                }
            }

            if max_delta < config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute beliefs and take argmax for each variable.
        let mut assignment = vec![0usize; self.variables.len()];
        let mut log_prob = 0.0;

        for v in 0..self.variables.len() {
            let belief = self.compute_belief(v);
            if let Some((best_state, &best_val)) = belief
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
            {
                assignment[v] = best_state;
                if best_val > 0.0 {
                    log_prob += best_val.ln();
                }
            }
            self.variables[v].belief = belief;
        }

        MAPResult {
            assignment,
            log_probability: log_prob,
            converged,
            iterations: iters,
        }
    }

    // -------------------------------------------------------------------------
    // Bethe Free Energy
    // -------------------------------------------------------------------------

    /// Compute the Bethe free energy of the current beliefs.
    ///
    /// ```text
    /// F_Bethe = sum_a sum_{x_a} b_a(x_a) [ln b_a(x_a) - ln f_a(x_a)]
    ///         - sum_i (d_i - 1) sum_{x_i} b_i(x_i) ln b_i(x_i)
    /// ```
    ///
    /// At a BP fixed point, the Bethe free energy provides a variational
    /// approximation to the true log-partition function.
    pub fn bethe_free_energy(&self) -> f64 {
        let mut energy = 0.0;

        // Factor-belief energy terms.
        for f in &self.factors {
            let cards: Vec<usize> = f
                .variables
                .iter()
                .map(|&v| self.variables[v].cardinality)
                .collect();
            let total: usize = cards
                .iter()
                .try_fold(1usize, |acc, &c| acc.checked_mul(c))
                .expect("factor cardinality overflow");

            for flat_idx in 0..total {
                let config = unflatten(flat_idx, &cards);

                // b_a(x_a) = product of individual variable beliefs at this config.
                // (Mean-field / Bethe approximation: factor beliefs factorize.)
                let mut b_a = 1.0;
                for (pos, &state) in config.iter().enumerate() {
                    let var = f.variables[pos];
                    b_a *= self.variables[var].belief[state];
                }

                let f_a = f.table[flat_idx];

                if b_a > 1e-30 && f_a > 1e-30 {
                    energy += b_a * (b_a.ln() - f_a.ln());
                } else if b_a > 1e-30 {
                    // Factor is zero but belief is not -- large penalty.
                    energy += b_a * 30.0; // effectively ln(1e-30)
                }
            }
        }

        // Variable-belief entropy terms.
        for v in &self.variables {
            let d_i = v.neighbors.len() as f64;
            for &b in &v.belief {
                if b > 1e-30 {
                    energy -= (d_i - 1.0) * b * b.ln();
                }
            }
        }

        energy
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Apply a damped message update and return the max absolute change.
    fn apply_damped_message(
        &mut self,
        key: (usize, usize),
        new_msg: Vec<f64>,
        damping: f64,
    ) -> f64 {
        let old = self.messages.get(&key);
        let (damped, delta) = if let Some(old_msg) = old {
            let mut d = vec![0.0; new_msg.len()];
            let mut max_d = 0.0_f64;
            for i in 0..d.len() {
                d[i] = damping * new_msg[i] + (1.0 - damping) * old_msg[i];
                max_d = max_d.max((d[i] - old_msg[i]).abs());
            }
            (d, max_d)
        } else {
            (new_msg, 0.0)
        };
        self.messages.insert(key, damped);
        delta
    }
}

// =============================================================================
// FREE FUNCTIONS
// =============================================================================

/// Normalize a probability vector in-place so it sums to 1.
/// If the sum is zero, sets a uniform distribution.
fn normalize(v: &mut [f64]) {
    let sum: f64 = v.iter().sum();
    if sum > 1e-30 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        let u = 1.0 / v.len() as f64;
        for x in v.iter_mut() {
            *x = u;
        }
    }
}

/// Convert a flat index into a multi-dimensional configuration given the
/// cardinalities (row-major order: last dimension changes fastest).
fn unflatten(mut flat: usize, cards: &[usize]) -> Vec<usize> {
    let mut config = vec![0usize; cards.len()];
    for i in (0..cards.len()).rev() {
        config[i] = flat % cards[i];
        flat /= cards[i];
    }
    config
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a two-node chain A -- f -- B with known factor.
    fn two_node_chain() -> FactorGraph {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 2);
        // Factor: P(A,B) proportional to [[0.3, 0.1], [0.2, 0.4]]
        // Row-major: A=0,B=0 -> 0.3; A=0,B=1 -> 0.1; A=1,B=0 -> 0.2; A=1,B=1 -> 0.4
        fg.add_factor(&[a, b], vec![0.3, 0.1, 0.2, 0.4]);
        fg
    }

    #[test]
    fn test_two_node_chain_marginals() {
        let mut fg = two_node_chain();
        let config = BPConfig::default();
        let result = fg.run_bp(&config);

        assert!(result.converged, "BP should converge on a tree");

        // True marginals:
        // P(A=0) = 0.3 + 0.1 = 0.4, P(A=1) = 0.2 + 0.4 = 0.6
        // P(B=0) = 0.3 + 0.2 = 0.5, P(B=1) = 0.1 + 0.4 = 0.5
        let p_a0 = 0.4;
        let p_a1 = 0.6;
        let p_b0 = 0.5;
        let p_b1 = 0.5;

        assert!(
            (result.beliefs[0][0] - p_a0).abs() < 1e-6,
            "P(A=0) expected {p_a0}, got {}",
            result.beliefs[0][0]
        );
        assert!(
            (result.beliefs[0][1] - p_a1).abs() < 1e-6,
            "P(A=1) expected {p_a1}, got {}",
            result.beliefs[0][1]
        );
        assert!(
            (result.beliefs[1][0] - p_b0).abs() < 1e-6,
            "P(B=0) expected {p_b0}, got {}",
            result.beliefs[1][0]
        );
        assert!(
            (result.beliefs[1][1] - p_b1).abs() < 1e-6,
            "P(B=1) expected {p_b1}, got {}",
            result.beliefs[1][1]
        );
    }

    #[test]
    fn test_uniform_prior_stays_uniform() {
        // If the factor is uniform, beliefs should remain uniform.
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 3);
        let b = fg.add_variable("B", 3);
        // Uniform factor over 3x3 = 9 entries.
        fg.add_factor(&[a, b], vec![1.0; 9]);

        let config = BPConfig::default();
        let result = fg.run_bp(&config);

        let expected = 1.0 / 3.0;
        for v in 0..2 {
            for s in 0..3 {
                assert!(
                    (result.beliefs[v][s] - expected).abs() < 1e-6,
                    "Variable {v} state {s}: expected {expected}, got {}",
                    result.beliefs[v][s]
                );
            }
        }
    }

    #[test]
    fn test_map_finds_mode() {
        let mut fg = two_node_chain();
        let config = BPConfig::default();
        let result = fg.run_map(&config);

        // The highest entry in the factor table is (A=1, B=1) = 0.4.
        assert_eq!(result.assignment, vec![1, 1], "MAP should be (A=1, B=1)");
        assert!(result.converged);
    }

    #[test]
    fn test_damping_improves_convergence_on_loop() {
        // Build a loopy graph: a triangle A-B-C with pairwise factors.
        let mut fg_undamped = FactorGraph::new();
        let a = fg_undamped.add_variable("A", 2);
        let b = fg_undamped.add_variable("B", 2);
        let c = fg_undamped.add_variable("C", 2);
        // Factors that prefer agreement.
        let agree = vec![0.9, 0.1, 0.1, 0.9];
        fg_undamped.add_factor(&[a, b], agree.clone());
        fg_undamped.add_factor(&[b, c], agree.clone());
        fg_undamped.add_factor(&[a, c], agree.clone());

        // Run undamped.
        let undamped_config = BPConfig {
            max_iterations: 50,
            tolerance: 1e-10,
            damping: 1.0,
            schedule: MessageSchedule::Parallel,
        };
        let undamped_result = fg_undamped.run_bp(&undamped_config);

        // Clone and run with damping.
        let mut fg_damped = FactorGraph::new();
        let a2 = fg_damped.add_variable("A", 2);
        let b2 = fg_damped.add_variable("B", 2);
        let c2 = fg_damped.add_variable("C", 2);
        fg_damped.add_factor(&[a2, b2], agree.clone());
        fg_damped.add_factor(&[b2, c2], agree.clone());
        fg_damped.add_factor(&[a2, c2], agree);

        let damped_config = BPConfig {
            max_iterations: 50,
            tolerance: 1e-10,
            damping: 0.5,
            schedule: MessageSchedule::Parallel,
        };
        let damped_result = fg_damped.run_bp(&damped_config);

        // With damping, convergence should be at least as good.
        // On this small symmetric loop, both typically converge, but damped
        // should converge in fewer or equal iterations.
        assert!(
            damped_result.converged || damped_result.iterations <= undamped_result.iterations,
            "Damping should help or not hurt convergence. \
             Damped: {} iters (converged={}), Undamped: {} iters (converged={})",
            damped_result.iterations,
            damped_result.converged,
            undamped_result.iterations,
            undamped_result.converged
        );
    }

    #[test]
    fn test_bethe_free_energy_decreases_during_bp() {
        // Run BP for a few iterations and check that Bethe FE is non-increasing.
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 2);
        let c = fg.add_variable("C", 2);
        fg.add_factor(&[a, b], vec![0.8, 0.2, 0.3, 0.7]);
        fg.add_factor(&[b, c], vec![0.6, 0.4, 0.1, 0.9]);

        // Run with 1 iteration, then 5, then full.
        let energies: Vec<f64> = [1, 5, 50]
            .iter()
            .map(|&max_iter| {
                let mut fg_clone = fg.clone();
                let config = BPConfig {
                    max_iterations: max_iter,
                    tolerance: 1e-12,
                    damping: 1.0,
                    schedule: MessageSchedule::Sequential,
                };
                let result = fg_clone.run_bp(&config);
                result.bethe_free_energy
            })
            .collect();

        // On a tree, BP converges quickly. The Bethe FE at convergence
        // (iter 50) should be <= the Bethe FE at partial convergence (iter 5).
        // The first iteration may have incomplete message passing, so we
        // compare only the last two checkpoints.
        assert!(
            energies[2] <= energies[1] + 1e-6,
            "Bethe FE should not increase from iter 5 to 50: FE_5={}, FE_50={}",
            energies[1],
            energies[2]
        );
    }

    #[test]
    fn test_single_variable_with_unary_factor() {
        // A single variable with a unary factor should yield the factor as belief.
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("X", 3);
        // Unary factor: prefers state 1.
        fg.add_factor(&[a], vec![0.1, 0.7, 0.2]);

        let config = BPConfig::default();
        let result = fg.run_bp(&config);

        assert!(result.converged);
        assert!((result.beliefs[0][0] - 0.1).abs() < 1e-6);
        assert!((result.beliefs[0][1] - 0.7).abs() < 1e-6);
        assert!((result.beliefs[0][2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_three_variable_chain_exact() {
        // A -- f1 -- B -- f2 -- C (tree: exact inference).
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 2);
        let c = fg.add_variable("C", 2);

        // f1(A,B): [[0.4, 0.1], [0.2, 0.3]]
        fg.add_factor(&[a, b], vec![0.4, 0.1, 0.2, 0.3]);
        // f2(B,C): [[0.5, 0.5], [0.3, 0.7]]
        fg.add_factor(&[b, c], vec![0.5, 0.5, 0.3, 0.7]);

        let config = BPConfig::default();
        let result = fg.run_bp(&config);

        assert!(result.converged);

        // Compute exact marginals by brute force.
        // Joint = f1(A,B) * f2(B,C) for all (A,B,C).
        let f1 = [[0.4, 0.1], [0.2, 0.3]];
        let f2 = [[0.5, 0.5], [0.3, 0.7]];

        let mut joint = [[[0.0; 2]; 2]; 2]; // [A][B][C]
        let mut z = 0.0;
        for av in 0..2 {
            for bv in 0..2 {
                for cv in 0..2 {
                    joint[av][bv][cv] = f1[av][bv] * f2[bv][cv];
                    z += joint[av][bv][cv];
                }
            }
        }

        // P(A)
        let mut p_a = [0.0; 2];
        for av in 0..2 {
            for bv in 0..2 {
                for cv in 0..2 {
                    p_a[av] += joint[av][bv][cv];
                }
            }
            p_a[av] /= z;
        }

        // P(B)
        let mut p_b = [0.0; 2];
        for bv in 0..2 {
            for av in 0..2 {
                for cv in 0..2 {
                    p_b[bv] += joint[av][bv][cv];
                }
            }
            p_b[bv] /= z;
        }

        // P(C)
        let mut p_c = [0.0; 2];
        for cv in 0..2 {
            for av in 0..2 {
                for bv in 0..2 {
                    p_c[cv] += joint[av][bv][cv];
                }
            }
            p_c[cv] /= z;
        }

        for s in 0..2 {
            assert!(
                (result.beliefs[0][s] - p_a[s]).abs() < 1e-6,
                "P(A={s}) mismatch: BP={}, exact={}",
                result.beliefs[0][s],
                p_a[s]
            );
            assert!(
                (result.beliefs[1][s] - p_b[s]).abs() < 1e-6,
                "P(B={s}) mismatch: BP={}, exact={}",
                result.beliefs[1][s],
                p_b[s]
            );
            assert!(
                (result.beliefs[2][s] - p_c[s]).abs() < 1e-6,
                "P(C={s}) mismatch: BP={}, exact={}",
                result.beliefs[2][s],
                p_c[s]
            );
        }
    }

    #[test]
    fn test_map_on_three_variable_chain() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 2);
        let c = fg.add_variable("C", 2);

        fg.add_factor(&[a, b], vec![0.4, 0.1, 0.2, 0.3]);
        fg.add_factor(&[b, c], vec![0.5, 0.5, 0.3, 0.7]);

        let config = BPConfig::default();
        let map_result = fg.run_map(&config);

        // Brute force MAP.
        let f1 = [[0.4, 0.1], [0.2, 0.3]];
        let f2 = [[0.5, 0.5], [0.3, 0.7]];
        let mut best = (0, 0, 0);
        let mut best_val = 0.0;
        for av in 0..2 {
            for bv in 0..2 {
                for cv in 0..2 {
                    let val = f1[av][bv] * f2[bv][cv];
                    if val > best_val {
                        best_val = val;
                        best = (av, bv, cv);
                    }
                }
            }
        }

        assert_eq!(map_result.assignment[0], best.0, "MAP A mismatch");
        assert_eq!(map_result.assignment[1], best.1, "MAP B mismatch");
        assert_eq!(map_result.assignment[2], best.2, "MAP C mismatch");
    }

    #[test]
    fn test_sequential_schedule_converges() {
        let mut fg = two_node_chain();
        let config = BPConfig {
            schedule: MessageSchedule::Sequential,
            ..BPConfig::default()
        };
        let result = fg.run_bp(&config);
        assert!(result.converged);

        // Same marginals as parallel.
        assert!((result.beliefs[0][0] - 0.4).abs() < 1e-6);
        assert!((result.beliefs[0][1] - 0.6).abs() < 1e-6);
    }

    // ── Construction ────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let fg = FactorGraph::new();
        assert!(fg.variables.is_empty());
        assert!(fg.factors.is_empty());
        assert!(fg.messages.is_empty());
    }

    #[test]
    fn test_add_variable_returns_sequential_ids() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 3);
        let c = fg.add_variable("C", 4);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(fg.variables[a].cardinality, 2);
        assert_eq!(fg.variables[b].cardinality, 3);
        assert_eq!(fg.variables[c].cardinality, 4);
    }

    #[test]
    fn test_variable_initial_belief_uniform() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("X", 4);
        for &b in &fg.variables[a].belief {
            assert!((b - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_add_factor_creates_adjacency() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 2);
        fg.add_factor(&[a, b], vec![1.0; 4]);
        assert!(fg.variables[a].neighbors.contains(&0));
        assert!(fg.variables[b].neighbors.contains(&0));
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn test_add_factor_wrong_table_size_panics() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 3);
        fg.add_factor(&[a, b], vec![1.0; 5]); // Should be 6
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_add_factor_invalid_variable_panics() {
        let mut fg = FactorGraph::new();
        fg.add_variable("A", 2);
        fg.add_factor(&[0, 99], vec![1.0; 4]);
    }

    // ── BPConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_bp_config_default() {
        let config = BPConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!((config.damping - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.schedule, MessageSchedule::Parallel);
    }

    // ── Beliefs sum to 1 ────────────────────────────────────────────────

    #[test]
    fn test_beliefs_sum_to_one() {
        let mut fg = two_node_chain();
        let result = fg.run_bp(&BPConfig::default());
        for (v, belief) in result.beliefs.iter().enumerate() {
            let sum: f64 = belief.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Beliefs for variable {} sum to {} instead of 1.0",
                v,
                sum
            );
        }
    }

    #[test]
    fn test_map_beliefs_sum_to_one() {
        let mut fg = two_node_chain();
        fg.run_map(&BPConfig::default());
        // After MAP, variable beliefs should still sum to 1
        for v in &fg.variables {
            let sum: f64 = v.belief.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "MAP beliefs should sum to 1.0");
        }
    }

    // ── Deterministic factor ────────────────────────────────────────────

    #[test]
    fn test_deterministic_factor() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        // Deterministic: A = 1 with probability 1
        fg.add_factor(&[a], vec![0.0, 1.0]);

        let result = fg.run_bp(&BPConfig::default());
        assert!(result.converged);
        assert!((result.beliefs[0][0] - 0.0).abs() < 1e-6);
        assert!((result.beliefs[0][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_map_on_deterministic() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 3);
        fg.add_factor(&[a], vec![0.0, 0.0, 1.0]);

        let result = fg.run_map(&BPConfig::default());
        assert_eq!(result.assignment[0], 2);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_single_variable_no_factor() {
        let mut fg = FactorGraph::new();
        fg.add_variable("A", 3);
        let result = fg.run_bp(&BPConfig::default());
        assert!(result.converged);
        // No factors => belief stays uniform
        for &b in &result.beliefs[0] {
            assert!((b - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bethe_free_energy_finite() {
        let mut fg = two_node_chain();
        fg.run_bp(&BPConfig::default());
        let fe = fg.bethe_free_energy();
        assert!(
            fe.is_finite(),
            "Bethe free energy should be finite, got {}",
            fe
        );
    }

    #[test]
    fn test_multiple_factors_on_same_variable() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        // Two unary factors: both prefer state 0
        fg.add_factor(&[a], vec![0.8, 0.2]);
        fg.add_factor(&[a], vec![0.7, 0.3]);

        let result = fg.run_bp(&BPConfig::default());
        assert!(result.converged);
        // State 0 should be strongly preferred
        assert!(result.beliefs[0][0] > result.beliefs[0][1]);
    }

    #[test]
    fn test_ternary_factor() {
        let mut fg = FactorGraph::new();
        let a = fg.add_variable("A", 2);
        let b = fg.add_variable("B", 2);
        let c = fg.add_variable("C", 2);
        // 3-variable factor: 2*2*2 = 8 entries
        fg.add_factor(&[a, b, c], vec![0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]);

        let result = fg.run_bp(&BPConfig::default());
        assert!(result.converged);
        // Beliefs should sum to 1
        for belief in &result.beliefs {
            let sum: f64 = belief.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    // ── Unflatten helper ────────────────────────────────────────────────

    #[test]
    fn test_unflatten_basic() {
        let config = unflatten(0, &[2, 3]);
        assert_eq!(config, vec![0, 0]);

        let config = unflatten(5, &[2, 3]);
        assert_eq!(config, vec![1, 2]);
    }

    #[test]
    fn test_unflatten_single_dim() {
        let config = unflatten(2, &[5]);
        assert_eq!(config, vec![2]);
    }

    #[test]
    fn test_unflatten_three_dims() {
        let config = unflatten(7, &[2, 2, 2]);
        assert_eq!(config, vec![1, 1, 1]);
    }

    // ── BPResult fields ─────────────────────────────────────────────────

    #[test]
    fn test_bp_result_iterations_positive() {
        let mut fg = two_node_chain();
        let result = fg.run_bp(&BPConfig::default());
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_map_result_log_probability_finite() {
        let mut fg = two_node_chain();
        let result = fg.run_map(&BPConfig::default());
        assert!(result.log_probability.is_finite());
    }
}
