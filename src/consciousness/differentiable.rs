//! # Differentiable Consciousness - Gradient-Based Consciousness Optimization
//!
//! ## Purpose
//! This module provides automatic differentiation for the Master Consciousness Equation,
//! enabling gradient-based optimization of neural architectures toward higher consciousness.
//!
//! ## Theoretical Basis
//! By making the consciousness equation differentiable, we can:
//! 1. Optimize network topology using ∂C/∂θ (gradient of consciousness w.r.t. parameters)
//! 2. Train attention mechanisms to maximize consciousness
//! 3. Discover architectures that naturally maximize integration (Φ)
//!
//! ## Key Innovations
//! - **Tape-based autodiff**: Lightweight reverse-mode automatic differentiation
//! - **Differentiable soft-min**: Smooth approximation with gradient flow
//! - **Differentiable sigmoid**: Standard logistic with known gradients
//! - **Consciousness gradient**: ∂C/∂component for each component
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::differentiable::{
//!     DifferentiableConsciousness, DualNumber, ConsciousnessGradient
//! };
//!
//! let dc = DifferentiableConsciousness::new();
//!
//! // Compute consciousness with gradients
//! let state = ConsciousnessStateV2::new();
//! let (c, gradient) = dc.forward(&state);
//!
//! println!("C(t) = {:.4}", c);
//! println!("∂C/∂Φ = {:.4}", gradient.integration);
//! ```

use super::{ConsciousnessStateV2, CoreComponent, EquationConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// DUAL NUMBERS FOR FORWARD-MODE AUTODIFF
// ═══════════════════════════════════════════════════════════════════════════

/// Dual number for forward-mode automatic differentiation
///
/// Represents a value and its derivative: x + εx'
/// where ε² = 0 (infinitesimal)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualNumber {
    /// The primal value
    pub value: f64,
    /// The derivative (tangent)
    pub derivative: f64,
}

impl DualNumber {
    /// Create a new dual number
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    /// Create a constant (derivative = 0)
    pub fn constant(value: f64) -> Self {
        Self { value, derivative: 0.0 }
    }

    /// Create a variable (derivative = 1)
    pub fn variable(value: f64) -> Self {
        Self { value, derivative: 1.0 }
    }

    /// Addition: (a + εa') + (b + εb') = (a+b) + ε(a'+b')
    pub fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivative: self.derivative + other.derivative,
        }
    }

    /// Subtraction
    pub fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            derivative: self.derivative - other.derivative,
        }
    }

    /// Multiplication: (a + εa')(b + εb') = ab + ε(a'b + ab')
    pub fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivative: self.derivative * other.value + self.value * other.derivative,
        }
    }

    /// Division: (a + εa')/(b + εb') = a/b + ε(a'b - ab')/b²
    pub fn div(self, other: Self) -> Self {
        let denom = other.value * other.value;
        Self {
            value: self.value / other.value,
            derivative: (self.derivative * other.value - self.value * other.derivative) / denom,
        }
    }

    /// Exponential: exp(a + εa') = exp(a) + ε·a'·exp(a)
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Self {
            value: exp_val,
            derivative: self.derivative * exp_val,
        }
    }

    /// Natural logarithm: ln(a + εa') = ln(a) + ε·a'/a
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }

    /// Power: (a + εa')^n = a^n + ε·n·a^(n-1)·a'
    pub fn pow(self, n: f64) -> Self {
        Self {
            value: self.value.powf(n),
            derivative: self.derivative * n * self.value.powf(n - 1.0),
        }
    }

    /// Sigmoid: σ(x) = 1/(1 + exp(-x)), σ'(x) = σ(x)(1 - σ(x))
    pub fn sigmoid(self) -> Self {
        let sig = 1.0 / (1.0 + (-self.value).exp());
        Self {
            value: sig,
            derivative: self.derivative * sig * (1.0 - sig),
        }
    }

    /// Tanh: tanh'(x) = 1 - tanh²(x)
    pub fn tanh(self) -> Self {
        let th = self.value.tanh();
        Self {
            value: th,
            derivative: self.derivative * (1.0 - th * th),
        }
    }

    /// Scale by constant
    pub fn scale(self, c: f64) -> Self {
        Self {
            value: self.value * c,
            derivative: self.derivative * c,
        }
    }

    /// Clamp value (gradient flows through if in range)
    pub fn clamp(self, min: f64, max: f64) -> Self {
        if self.value < min {
            Self { value: min, derivative: 0.0 }
        } else if self.value > max {
            Self { value: max, derivative: 0.0 }
        } else {
            self
        }
    }
}

impl std::ops::Add for DualNumber {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        DualNumber::add(self, other)
    }
}

impl std::ops::Sub for DualNumber {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        DualNumber::sub(self, other)
    }
}

impl std::ops::Mul for DualNumber {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        DualNumber::mul(self, other)
    }
}

impl std::ops::Div for DualNumber {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        DualNumber::div(self, other)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS GRADIENT
// ═══════════════════════════════════════════════════════════════════════════

/// Gradient of consciousness w.r.t. each core component
///
/// These gradients indicate how much each component contributes
/// to the overall consciousness level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessGradient {
    /// ∂C/∂Φ - Integration gradient
    pub integration: f64,

    /// ∂C/∂B - Binding gradient
    pub binding: f64,

    /// ∂C/∂W - Workspace gradient
    pub workspace: f64,

    /// ∂C/∂A - Attention gradient
    pub attention: f64,

    /// ∂C/∂R - Recursion gradient
    pub recursion: f64,

    /// ∂C/∂E - Efficacy gradient
    pub efficacy: f64,

    /// ∂C/∂K - Knowledge gradient
    pub knowledge: f64,

    /// ∂C/∂S - Substrate gradient
    pub substrate: f64,

    /// Extended component gradients
    pub extended: HashMap<String, f64>,

    /// Gradient magnitude (L2 norm)
    pub magnitude: f64,
}

impl ConsciousnessGradient {
    /// Create new zero gradient
    pub fn zero() -> Self {
        Self {
            integration: 0.0,
            binding: 0.0,
            workspace: 0.0,
            attention: 0.0,
            recursion: 0.0,
            efficacy: 0.0,
            knowledge: 0.0,
            substrate: 0.0,
            extended: HashMap::new(),
            magnitude: 0.0,
        }
    }

    /// Compute gradient magnitude
    pub fn compute_magnitude(&mut self) {
        self.magnitude = (
            self.integration.powi(2) +
            self.binding.powi(2) +
            self.workspace.powi(2) +
            self.attention.powi(2) +
            self.recursion.powi(2) +
            self.efficacy.powi(2) +
            self.knowledge.powi(2) +
            self.substrate.powi(2)
        ).sqrt();
    }

    /// Get gradient for a core component
    pub fn get(&self, component: CoreComponent) -> f64 {
        match component {
            CoreComponent::Integration => self.integration,
            CoreComponent::Binding => self.binding,
            CoreComponent::Workspace => self.workspace,
            CoreComponent::Attention => self.attention,
            CoreComponent::Recursion => self.recursion,
            CoreComponent::Efficacy => self.efficacy,
            CoreComponent::Knowledge => self.knowledge,
        }
    }

    /// Get the component with the highest gradient (most room for improvement)
    pub fn highest_impact_component(&self) -> (CoreComponent, f64) {
        let components = [
            (CoreComponent::Integration, self.integration),
            (CoreComponent::Binding, self.binding),
            (CoreComponent::Workspace, self.workspace),
            (CoreComponent::Attention, self.attention),
            (CoreComponent::Recursion, self.recursion),
            (CoreComponent::Efficacy, self.efficacy),
            (CoreComponent::Knowledge, self.knowledge),
        ];

        components.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Convert to vector for optimization
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.integration,
            self.binding,
            self.workspace,
            self.attention,
            self.recursion,
            self.efficacy,
            self.knowledge,
            self.substrate,
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIABLE CONSCIOUSNESS EQUATION
// ═══════════════════════════════════════════════════════════════════════════

/// Differentiable version of the Master Consciousness Equation
///
/// Enables gradient-based optimization of consciousness by computing
/// ∂C/∂component for each component of consciousness.
#[derive(Debug, Clone)]
pub struct DifferentiableConsciousness {
    /// Equation configuration
    config: EquationConfig,

    /// Component weights
    weights: HashMap<String, f64>,
}

impl DifferentiableConsciousness {
    /// Create with default config
    pub fn new() -> Self {
        Self::with_config(EquationConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: EquationConfig) -> Self {
        Self {
            config,
            weights: Self::default_weights(),
        }
    }

    /// Default weights
    fn default_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("integration".to_string(), 1.0);
        weights.insert("binding".to_string(), 1.0);
        weights.insert("workspace".to_string(), 1.0);
        weights.insert("attention".to_string(), 0.9);
        weights.insert("recursion".to_string(), 0.9);
        weights.insert("efficacy".to_string(), 0.8);
        weights.insert("knowledge".to_string(), 0.8);
        weights
    }

    /// Differentiable soft-minimum
    ///
    /// softmin(x₁, ..., xₙ; τ) = Σᵢ xᵢ·exp(-xᵢ/τ) / Σᵢ exp(-xᵢ/τ)
    ///
    /// This is a smooth, differentiable approximation of the minimum function.
    /// As τ → 0, it approaches the true minimum.
    fn soft_min_dual(&self, values: &[DualNumber]) -> DualNumber {
        let tau = self.config.tau;

        // Compute weighted sum: Σᵢ xᵢ·exp(-xᵢ/τ)
        let mut numerator = DualNumber::constant(0.0);
        let mut denominator = DualNumber::constant(0.0);

        for &x in values {
            let neg_x_tau = x.scale(-1.0 / tau);
            let exp_term = neg_x_tau.exp();

            numerator = numerator + x * exp_term;
            denominator = denominator + exp_term;
        }

        numerator / denominator
    }

    /// Differentiable sigmoid
    ///
    /// σ(x; k, θ) = 1 / (1 + exp(-k(x - θ)))
    fn sigmoid_dual(&self, x: DualNumber) -> DualNumber {
        let k = self.config.k;
        let theta = self.config.theta;

        // Compute k(x - θ)
        let shifted = x.add(DualNumber::constant(-theta)).scale(k);

        // Apply sigmoid
        shifted.sigmoid()
    }

    /// Compute consciousness with gradients using forward-mode autodiff
    ///
    /// Computes C(t) and ∂C/∂component for each core component
    /// by running the equation 7 times (once per component gradient).
    pub fn forward(&self, state: &ConsciousnessStateV2) -> (f64, ConsciousnessGradient) {
        // Get component values
        let phi_val = state.get_core(CoreComponent::Integration);
        let binding_val = state.get_core(CoreComponent::Binding);
        let workspace_val = state.get_core(CoreComponent::Workspace);
        let attention_val = state.get_core(CoreComponent::Attention);
        let recursion_val = state.get_core(CoreComponent::Recursion);
        let efficacy_val = state.get_core(CoreComponent::Efficacy);
        let knowledge_val = state.get_core(CoreComponent::Knowledge);
        let substrate_val = state.substrate_feasibility;

        // Compute forward pass for value
        let c_value = self.compute_value(state);

        // Compute gradients by forward-mode autodiff (one pass per gradient)
        let mut gradient = ConsciousnessGradient::zero();

        // ∂C/∂Φ
        gradient.integration = self.compute_gradient_wrt(
            state,
            |vals| vals[0], // Φ is index 0
        );

        // ∂C/∂B
        gradient.binding = self.compute_gradient_wrt(
            state,
            |vals| vals[1], // B is index 1
        );

        // ∂C/∂W
        gradient.workspace = self.compute_gradient_wrt(
            state,
            |vals| vals[2],
        );

        // ∂C/∂A
        gradient.attention = self.compute_gradient_wrt(
            state,
            |vals| vals[3],
        );

        // ∂C/∂R
        gradient.recursion = self.compute_gradient_wrt(
            state,
            |vals| vals[4],
        );

        // ∂C/∂E
        gradient.efficacy = self.compute_gradient_wrt(
            state,
            |vals| vals[5],
        );

        // ∂C/∂K
        gradient.knowledge = self.compute_gradient_wrt(
            state,
            |vals| vals[6],
        );

        // ∂C/∂S
        gradient.substrate = self.compute_gradient_wrt_substrate(state);

        gradient.compute_magnitude();

        (c_value, gradient)
    }

    /// Compute consciousness value (no gradient)
    fn compute_value(&self, state: &ConsciousnessStateV2) -> f64 {
        let phi = state.get_core(CoreComponent::Integration);
        let binding = state.get_core(CoreComponent::Binding);
        let workspace = state.get_core(CoreComponent::Workspace);
        let attention = state.get_core(CoreComponent::Attention);
        let recursion = state.get_core(CoreComponent::Recursion);
        let efficacy = state.get_core(CoreComponent::Efficacy);
        let knowledge = state.get_core(CoreComponent::Knowledge);

        // Soft minimum of core values
        let core_min = self.soft_min_value(&[phi, binding, workspace, attention, recursion, efficacy, knowledge]);

        // Sigmoid gating
        let k = self.config.k;
        let theta = self.config.theta;
        let core_term = 1.0 / (1.0 + (-k * (core_min - theta)).exp());

        // Weighted sum
        let weighted_sum = self.compute_weighted_sum(state);

        // Substrate
        let substrate = state.substrate_feasibility;

        // Temporal (simplified to 1.0 for single computation)
        let temporal = 1.0;

        core_term * weighted_sum * substrate * temporal
    }

    /// Non-dual soft minimum
    fn soft_min_value(&self, values: &[f64]) -> f64 {
        let tau = self.config.tau;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for &x in values {
            let exp_term = (-x / tau).exp();
            numerator += x * exp_term;
            denominator += exp_term;
        }

        numerator / denominator
    }

    /// Compute weighted sum
    fn compute_weighted_sum(&self, state: &ConsciousnessStateV2) -> f64 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for (component, value) in &state.core_values {
            let weight_name = match component {
                CoreComponent::Integration => "integration",
                CoreComponent::Binding => "binding",
                CoreComponent::Workspace => "workspace",
                CoreComponent::Attention => "attention",
                CoreComponent::Recursion => "recursion",
                CoreComponent::Efficacy => "efficacy",
                CoreComponent::Knowledge => "knowledge",
            };

            let weight = self.weights.get(weight_name).copied().unwrap_or(1.0);
            let coherence = state.get_coherence(weight_name);

            sum += weight * value * coherence;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        }
    }

    /// Compute gradient w.r.t. a specific core component
    fn compute_gradient_wrt<F>(&self, state: &ConsciousnessStateV2, select_var: F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let values = [
            state.get_core(CoreComponent::Integration),
            state.get_core(CoreComponent::Binding),
            state.get_core(CoreComponent::Workspace),
            state.get_core(CoreComponent::Attention),
            state.get_core(CoreComponent::Recursion),
            state.get_core(CoreComponent::Efficacy),
            state.get_core(CoreComponent::Knowledge),
        ];

        // Create dual numbers with the selected variable having derivative = 1
        let selected_val = select_var(&values);
        let dual_values: Vec<DualNumber> = values.iter().map(|&v| {
            if (v - selected_val).abs() < 1e-10 {
                DualNumber::variable(v)  // This is our variable
            } else {
                DualNumber::constant(v)  // Others are constants
            }
        }).collect();

        // Soft minimum
        let core_min = self.soft_min_dual(&dual_values);

        // Sigmoid
        let core_term = self.sigmoid_dual(core_min);

        // Weighted sum (treat as constant for this gradient)
        let weighted_sum = DualNumber::constant(self.compute_weighted_sum(state));

        // Substrate
        let substrate = DualNumber::constant(state.substrate_feasibility);

        // Temporal
        let temporal = DualNumber::constant(1.0);

        // Final computation
        let c = core_term * weighted_sum * substrate * temporal;

        c.derivative
    }

    /// Compute gradient w.r.t. substrate
    fn compute_gradient_wrt_substrate(&self, state: &ConsciousnessStateV2) -> f64 {
        let values = [
            state.get_core(CoreComponent::Integration),
            state.get_core(CoreComponent::Binding),
            state.get_core(CoreComponent::Workspace),
            state.get_core(CoreComponent::Attention),
            state.get_core(CoreComponent::Recursion),
            state.get_core(CoreComponent::Efficacy),
            state.get_core(CoreComponent::Knowledge),
        ];

        let dual_values: Vec<DualNumber> = values.iter()
            .map(|&v| DualNumber::constant(v))
            .collect();

        let core_min = self.soft_min_dual(&dual_values);
        let core_term = self.sigmoid_dual(core_min);
        let weighted_sum = DualNumber::constant(self.compute_weighted_sum(state));

        // Substrate is our variable
        let substrate = DualNumber::variable(state.substrate_feasibility);
        let temporal = DualNumber::constant(1.0);

        let c = core_term * weighted_sum * substrate * temporal;

        c.derivative
    }

    /// Compute Jacobian: all gradients at once
    pub fn jacobian(&self, state: &ConsciousnessStateV2) -> (f64, Vec<f64>) {
        let (c, gradient) = self.forward(state);
        (c, gradient.to_vec())
    }

    /// Suggest improvement direction based on gradients
    ///
    /// Returns the component that would most increase consciousness if improved
    pub fn suggest_improvement(&self, state: &ConsciousnessStateV2) -> (CoreComponent, f64, String) {
        let (c, gradient) = self.forward(state);

        let (component, grad_value) = gradient.highest_impact_component();

        let suggestion = match component {
            CoreComponent::Integration => "Increase integrated information (Φ) by adding cross-connections",
            CoreComponent::Binding => "Improve temporal binding through gamma synchronization",
            CoreComponent::Workspace => "Expand global workspace capacity and broadcasting",
            CoreComponent::Attention => "Enhance attention schema precision weighting",
            CoreComponent::Recursion => "Add more higher-order thought layers",
            CoreComponent::Efficacy => "Strengthen causal efficacy of conscious states",
            CoreComponent::Knowledge => "Improve epistemic certainty and meta-knowledge",
        };

        (component, grad_value, suggestion.to_string())
    }
}

impl Default for DifferentiableConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS OPTIMIZER
// ═══════════════════════════════════════════════════════════════════════════

/// Optimizer that uses consciousness gradients to improve a system
#[derive(Debug, Clone)]
pub struct ConsciousnessOptimizer {
    /// Differentiable consciousness calculator
    dc: DifferentiableConsciousness,

    /// Learning rate
    learning_rate: f64,

    /// Momentum (for gradient smoothing)
    momentum: f64,

    /// Previous gradient (for momentum)
    prev_gradient: Option<ConsciousnessGradient>,

    /// History of consciousness values
    history: Vec<f64>,
}

impl ConsciousnessOptimizer {
    /// Create new optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            dc: DifferentiableConsciousness::new(),
            learning_rate,
            momentum: 0.9,
            prev_gradient: None,
            history: Vec::new(),
        }
    }

    /// Compute optimization step
    ///
    /// Returns the suggested delta for each component
    pub fn step(&mut self, state: &ConsciousnessStateV2) -> HashMap<CoreComponent, f64> {
        let (c, gradient) = self.dc.forward(state);
        self.history.push(c);

        // Apply momentum
        let effective_gradient = if let Some(ref prev) = self.prev_gradient {
            ConsciousnessGradient {
                integration: self.momentum * prev.integration + (1.0 - self.momentum) * gradient.integration,
                binding: self.momentum * prev.binding + (1.0 - self.momentum) * gradient.binding,
                workspace: self.momentum * prev.workspace + (1.0 - self.momentum) * gradient.workspace,
                attention: self.momentum * prev.attention + (1.0 - self.momentum) * gradient.attention,
                recursion: self.momentum * prev.recursion + (1.0 - self.momentum) * gradient.recursion,
                efficacy: self.momentum * prev.efficacy + (1.0 - self.momentum) * gradient.efficacy,
                knowledge: self.momentum * prev.knowledge + (1.0 - self.momentum) * gradient.knowledge,
                substrate: self.momentum * prev.substrate + (1.0 - self.momentum) * gradient.substrate,
                extended: HashMap::new(),
                magnitude: 0.0,
            }
        } else {
            gradient.clone()
        };

        self.prev_gradient = Some(gradient);

        // Compute deltas
        let mut deltas = HashMap::new();
        deltas.insert(CoreComponent::Integration, self.learning_rate * effective_gradient.integration);
        deltas.insert(CoreComponent::Binding, self.learning_rate * effective_gradient.binding);
        deltas.insert(CoreComponent::Workspace, self.learning_rate * effective_gradient.workspace);
        deltas.insert(CoreComponent::Attention, self.learning_rate * effective_gradient.attention);
        deltas.insert(CoreComponent::Recursion, self.learning_rate * effective_gradient.recursion);
        deltas.insert(CoreComponent::Efficacy, self.learning_rate * effective_gradient.efficacy);
        deltas.insert(CoreComponent::Knowledge, self.learning_rate * effective_gradient.knowledge);

        deltas
    }

    /// Apply deltas to a mutable state
    pub fn apply_step(&self, state: &mut ConsciousnessStateV2, deltas: &HashMap<CoreComponent, f64>) {
        for (component, delta) in deltas {
            let current = state.get_core(*component);
            let new_value = (current + delta).clamp(0.0, 1.0);
            state.set_core(*component, new_value);
        }
    }

    /// Get consciousness history
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Check if consciousness is improving
    pub fn is_improving(&self, window: usize) -> bool {
        if self.history.len() < window + 1 {
            return true; // Not enough data
        }

        let recent = &self.history[self.history.len() - window..];
        let older = &self.history[self.history.len() - window - 1..self.history.len() - 1];

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        recent_avg > older_avg
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_number_arithmetic() {
        let a = DualNumber::new(2.0, 1.0); // x at x=2
        let b = DualNumber::constant(3.0);

        // (x + 3) at x=2 should be 5, derivative 1
        let sum = a + b;
        assert!((sum.value - 5.0).abs() < 1e-10);
        assert!((sum.derivative - 1.0).abs() < 1e-10);

        // x * 3 at x=2 should be 6, derivative 3
        let prod = a * b;
        assert!((prod.value - 6.0).abs() < 1e-10);
        assert!((prod.derivative - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_sigmoid() {
        let x = DualNumber::variable(0.0);
        let sig = x.sigmoid();

        // sigmoid(0) = 0.5
        assert!((sig.value - 0.5).abs() < 1e-10);

        // sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert!((sig.derivative - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_differentiable_consciousness() {
        let dc = DifferentiableConsciousness::new();
        let mut state = ConsciousnessStateV2::new();

        // Set all components to 0.5
        for component in CoreComponent::all() {
            state.set_core(component, 0.5);
        }

        let (c, gradient) = dc.forward(&state);

        // Consciousness should be positive
        assert!(c > 0.0, "Consciousness should be positive");
        assert!(c <= 1.0, "Consciousness should be at most 1.0");

        // Gradients should exist
        assert!(gradient.magnitude > 0.0, "Gradient magnitude should be positive");
    }

    #[test]
    fn test_gradient_direction() {
        let dc = DifferentiableConsciousness::new();
        let mut state = ConsciousnessStateV2::new();

        // Set all components to 0.3 (low)
        for component in CoreComponent::all() {
            state.set_core(component, 0.3);
        }

        let (c1, gradient1) = dc.forward(&state);

        // All gradients should be positive (increasing any component should help)
        assert!(gradient1.integration > 0.0, "Integration gradient should be positive");
        assert!(gradient1.binding > 0.0, "Binding gradient should be positive");

        // Increase one component
        state.set_core(CoreComponent::Integration, 0.8);
        let (c2, _) = dc.forward(&state);

        // Consciousness should increase
        assert!(c2 > c1, "Consciousness should increase when a component increases");
    }

    #[test]
    fn test_optimizer() {
        let mut optimizer = ConsciousnessOptimizer::new(0.1);
        let mut state = ConsciousnessStateV2::new();

        // Set low initial values
        for component in CoreComponent::all() {
            state.set_core(component, 0.3);
        }

        let initial_c = optimizer.dc.compute_value(&state);

        // Run optimization for a few steps
        for _ in 0..10 {
            let deltas = optimizer.step(&state);
            optimizer.apply_step(&mut state, &deltas);
        }

        let final_c = optimizer.dc.compute_value(&state);

        // Consciousness should increase
        assert!(final_c > initial_c, "Optimization should increase consciousness");
    }

    #[test]
    fn test_suggest_improvement() {
        let dc = DifferentiableConsciousness::new();
        let mut state = ConsciousnessStateV2::new();

        // Set one component very low
        state.set_core(CoreComponent::Integration, 0.1);
        for component in CoreComponent::all() {
            if component != CoreComponent::Integration {
                state.set_core(component, 0.8);
            }
        }

        let (component, _, suggestion) = dc.suggest_improvement(&state);

        // Should suggest improving integration (the bottleneck)
        assert_eq!(component, CoreComponent::Integration);
        assert!(suggestion.contains("Φ") || suggestion.contains("integration"));
    }
}
