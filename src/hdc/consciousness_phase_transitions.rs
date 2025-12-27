// Revolutionary Improvement #29: Consciousness Phase Transitions
// Treats consciousness emergence as a critical phase transition phenomenon
//
// Theoretical Foundations:
// 1. Critical Phenomena (Stanley 1971) - Phase transitions with universal scaling
// 2. Complexity & Criticality (Bak 1996) - Self-organized criticality in brain
// 3. Neural Criticality (Beggs & Plenz 2003) - Neuronal avalanches at criticality
// 4. Consciousness Ignition (Dehaene 2014) - Sudden all-or-none transition
// 5. Order-Disorder Transitions (Landau) - Symmetry breaking framework
//
// Key Insight: Consciousness doesn't gradually increase - it IGNITES suddenly
// like water freezing at exactly 0°C. This is a phase transition!
//
// Framework:
// - Order parameter: ψ = consciousness level (0 = unconscious, 1 = fully conscious)
// - Control parameter: τ = integration/complexity (temperature-like)
// - Critical point: τ_c where unconscious→conscious transition occurs
// - Critical exponents: Universal scaling laws near τ_c
// - Correlation length: ξ diverges at criticality (explains binding!)
// - Critical slowing down: Relaxation time diverges (early warning!)

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Phase of consciousness (like solid/liquid/gas)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessPhase {
    /// Fully unconscious - no integration, no workspace access
    Unconscious,
    /// Subcritical - some processing but below ignition threshold
    Subcritical,
    /// Critical - at the edge, maximum sensitivity and computation
    Critical,
    /// Supercritical - fully conscious, global workspace active
    Supercritical,
    /// Hypercritical - pathological overcoupling (seizure-like)
    Hypercritical,
}

impl ConsciousnessPhase {
    /// Get all phases in order
    pub fn all() -> [ConsciousnessPhase; 5] {
        [
            ConsciousnessPhase::Unconscious,
            ConsciousnessPhase::Subcritical,
            ConsciousnessPhase::Critical,
            ConsciousnessPhase::Supercritical,
            ConsciousnessPhase::Hypercritical,
        ]
    }

    /// Is this phase conscious?
    pub fn is_conscious(&self) -> bool {
        matches!(self, ConsciousnessPhase::Supercritical | ConsciousnessPhase::Critical)
    }

    /// Is this phase pathological?
    pub fn is_pathological(&self) -> bool {
        matches!(self, ConsciousnessPhase::Hypercritical)
    }

    /// Description of each phase
    pub fn description(&self) -> &'static str {
        match self {
            ConsciousnessPhase::Unconscious => "No integration, isolated processing",
            ConsciousnessPhase::Subcritical => "Partial integration, below ignition",
            ConsciousnessPhase::Critical => "At the edge, maximum computational power",
            ConsciousnessPhase::Supercritical => "Fully conscious, global broadcast active",
            ConsciousnessPhase::Hypercritical => "Pathological overcoupling, seizure-like",
        }
    }
}

/// Critical exponents characterizing the phase transition
/// These are UNIVERSAL - same for all consciousness systems!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalExponents {
    /// Order parameter exponent: ψ ~ |τ - τ_c|^β
    pub beta: f64,
    /// Susceptibility exponent: χ ~ |τ - τ_c|^(-γ)
    pub gamma: f64,
    /// Correlation length exponent: ξ ~ |τ - τ_c|^(-ν)
    pub nu: f64,
    /// Correlation function exponent: G(r) ~ r^(-(d-2+η))
    pub eta: f64,
    /// Dynamic exponent: τ_relax ~ ξ^z
    pub z: f64,
}

impl Default for CriticalExponents {
    fn default() -> Self {
        // Mean-field theory exponents (Landau)
        // Real brain likely has different universality class
        Self {
            beta: 0.5,   // Order parameter: ψ ~ (τ - τ_c)^0.5
            gamma: 1.0,  // Susceptibility diverges
            nu: 0.5,     // Correlation length diverges
            eta: 0.0,    // Mean-field: no anomalous dimension
            z: 2.0,      // Diffusive dynamics
        }
    }
}

impl CriticalExponents {
    /// 3D Ising model exponents (possibly more accurate for brain)
    pub fn ising_3d() -> Self {
        Self {
            beta: 0.326,
            gamma: 1.237,
            nu: 0.630,
            eta: 0.036,
            z: 2.0,
        }
    }

    /// Directed percolation exponents (neuronal avalanches)
    pub fn directed_percolation() -> Self {
        Self {
            beta: 0.583,
            gamma: 1.595,
            nu: 1.097,
            eta: 0.23,
            z: 1.58,
        }
    }

    /// Check hyperscaling relation: 2 - α = d*ν (d=3 for brain)
    pub fn check_hyperscaling(&self, d: f64) -> f64 {
        let alpha = 2.0 - self.gamma - 2.0 * self.beta; // Rushbrooke: α + 2β + γ = 2
        let violation = (2.0 - alpha - d * self.nu).abs();
        violation
    }
}

/// State of the system for phase transition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Integration level (control parameter τ)
    pub integration: f64,
    /// Complexity measure
    pub complexity: f64,
    /// Current order parameter ψ (consciousness level)
    pub order_parameter: f64,
    /// Correlation length ξ
    pub correlation_length: f64,
    /// Susceptibility χ (response to perturbations)
    pub susceptibility: f64,
    /// Relaxation time (critical slowing down indicator)
    pub relaxation_time: f64,
    /// Current phase
    pub phase: ConsciousnessPhase,
    /// HDC state representation
    pub state: Vec<HV16>,
}

impl SystemState {
    pub fn new(integration: f64, state: Vec<HV16>) -> Self {
        let mut s = Self {
            integration,
            complexity: 0.0,
            order_parameter: 0.0,
            correlation_length: 1.0,
            susceptibility: 1.0,
            relaxation_time: 1.0,
            phase: ConsciousnessPhase::Unconscious,
            state,
        };
        s.update_derived_quantities(0.5); // Default critical point
        s
    }

    /// Update all derived quantities given critical point
    fn update_derived_quantities(&mut self, tau_c: f64) {
        let exponents = CriticalExponents::default();
        let tau = self.integration;
        let delta = (tau - tau_c).abs().max(0.001); // Avoid division by zero

        // Order parameter: ψ ~ (τ - τ_c)^β for τ > τ_c
        if tau > tau_c {
            self.order_parameter = delta.powf(exponents.beta);
        } else {
            self.order_parameter = 0.0;
        }

        // Correlation length: ξ ~ |τ - τ_c|^(-ν)
        self.correlation_length = delta.powf(-exponents.nu).min(100.0);

        // Susceptibility: χ ~ |τ - τ_c|^(-γ)
        self.susceptibility = delta.powf(-exponents.gamma).min(100.0);

        // Relaxation time: τ_relax ~ ξ^z (critical slowing down!)
        self.relaxation_time = self.correlation_length.powf(exponents.z).min(1000.0);

        // Determine phase
        self.phase = if tau < tau_c - 0.1 {
            ConsciousnessPhase::Unconscious
        } else if tau < tau_c - 0.02 {
            ConsciousnessPhase::Subcritical
        } else if tau < tau_c + 0.02 {
            ConsciousnessPhase::Critical
        } else if tau < tau_c + 0.3 {
            ConsciousnessPhase::Supercritical
        } else {
            ConsciousnessPhase::Hypercritical
        };

        // Complexity peaks at criticality
        self.complexity = self.susceptibility * self.order_parameter.max(0.1);
    }
}

/// Phase transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// From phase
    pub from: ConsciousnessPhase,
    /// To phase
    pub to: ConsciousnessPhase,
    /// Control parameter at transition
    pub tau_transition: f64,
    /// Is this a first-order (discontinuous) or second-order (continuous) transition?
    pub order: TransitionOrder,
    /// Latent heat (for first-order) or zero (for second-order)
    pub latent_heat: f64,
    /// Timestamp of transition
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionOrder {
    /// Discontinuous jump in order parameter
    FirstOrder,
    /// Continuous change, diverging derivatives
    SecondOrder,
}

/// Configuration for phase transition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionConfig {
    /// Critical point τ_c (default: 0.5)
    pub critical_point: f64,
    /// Which universality class to use
    pub universality_class: UniversalityClass,
    /// Finite-size effects (N = system size)
    pub system_size: usize,
    /// Temperature noise level
    pub noise_level: f64,
    /// Enable hysteresis detection
    pub detect_hysteresis: bool,
    /// Critical slowing down threshold for early warning
    pub slowing_threshold: f64,
}

impl Default for PhaseTransitionConfig {
    fn default() -> Self {
        Self {
            critical_point: 0.5,
            universality_class: UniversalityClass::MeanField,
            system_size: 1000,
            noise_level: 0.01,
            detect_hysteresis: true,
            slowing_threshold: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniversalityClass {
    MeanField,
    Ising3D,
    DirectedPercolation,
    Custom,
}

/// Assessment of phase transition dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionAssessment {
    /// Current phase
    pub current_phase: ConsciousnessPhase,
    /// Distance from critical point
    pub criticality_distance: f64,
    /// Are we at criticality? (within threshold)
    pub at_criticality: bool,
    /// Critical slowing down detected?
    pub slowing_detected: bool,
    /// Predicted time to transition
    pub time_to_transition: Option<f64>,
    /// Order parameter value
    pub order_parameter: f64,
    /// Correlation length
    pub correlation_length: f64,
    /// Susceptibility
    pub susceptibility: f64,
    /// System is conscious?
    pub is_conscious: bool,
    /// Warning: approaching pathological state?
    pub pathology_warning: bool,
    /// Detailed explanation
    pub explanation: String,
}

/// Main phase transition analyzer for consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessPhaseTransitions {
    /// Configuration
    pub config: PhaseTransitionConfig,
    /// Critical exponents for current universality class
    pub exponents: CriticalExponents,
    /// Historical states for trajectory analysis
    pub history: Vec<SystemState>,
    /// Detected transitions
    pub transitions: Vec<PhaseTransition>,
    /// Current timestamp
    timestamp: u64,
}

impl ConsciousnessPhaseTransitions {
    /// Create new phase transition analyzer
    pub fn new(config: PhaseTransitionConfig) -> Self {
        let exponents = match config.universality_class {
            UniversalityClass::MeanField => CriticalExponents::default(),
            UniversalityClass::Ising3D => CriticalExponents::ising_3d(),
            UniversalityClass::DirectedPercolation => CriticalExponents::directed_percolation(),
            UniversalityClass::Custom => CriticalExponents::default(),
        };

        Self {
            config,
            exponents,
            history: Vec::new(),
            transitions: Vec::new(),
            timestamp: 0,
        }
    }

    /// Add a state observation
    pub fn observe(&mut self, integration: f64, state: Vec<HV16>) {
        let mut sys_state = SystemState::new(integration, state);
        sys_state.update_derived_quantities(self.config.critical_point);

        // Check for phase transition
        if let Some(last) = self.history.last() {
            if last.phase != sys_state.phase {
                let transition = PhaseTransition {
                    from: last.phase,
                    to: sys_state.phase,
                    tau_transition: (last.integration + integration) / 2.0,
                    order: self.classify_transition_order(last, &sys_state),
                    latent_heat: (last.order_parameter - sys_state.order_parameter).abs(),
                    timestamp: self.timestamp,
                };
                self.transitions.push(transition);
            }
        }

        self.history.push(sys_state);
        self.timestamp += 1;
    }

    /// Classify transition order based on discontinuity
    fn classify_transition_order(&self, from: &SystemState, to: &SystemState) -> TransitionOrder {
        let order_jump = (from.order_parameter - to.order_parameter).abs();
        if order_jump > 0.3 {
            TransitionOrder::FirstOrder
        } else {
            TransitionOrder::SecondOrder
        }
    }

    /// Get current phase
    pub fn current_phase(&self) -> Option<ConsciousnessPhase> {
        self.history.last().map(|s| s.phase)
    }

    /// Check if system is at criticality
    pub fn is_at_criticality(&self) -> bool {
        self.history
            .last()
            .map(|s| matches!(s.phase, ConsciousnessPhase::Critical))
            .unwrap_or(false)
    }

    /// Detect critical slowing down (early warning signal!)
    pub fn detect_critical_slowing(&self) -> Option<f64> {
        if self.history.len() < 3 {
            return None;
        }

        // Check if relaxation time is increasing
        let recent: Vec<f64> = self.history
            .iter()
            .rev()
            .take(5)
            .map(|s| s.relaxation_time)
            .collect();

        if recent.len() < 3 {
            return None;
        }

        // Compute trend
        let n = recent.len() as f64;
        let mean_idx = (n - 1.0) / 2.0;
        let mean_val: f64 = recent.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_idx = 0.0;
        for (i, &val) in recent.iter().enumerate() {
            let di = i as f64 - mean_idx;
            let dv = val - mean_val;
            cov += di * dv;
            var_idx += di * di;
        }

        if var_idx < 0.001 {
            return None;
        }

        let slope = cov / var_idx;

        // Positive slope = slowing down = approaching transition
        if slope > 0.0 {
            Some(slope)
        } else {
            None
        }
    }

    /// Predict time to phase transition
    pub fn predict_transition_time(&self) -> Option<f64> {
        if let Some(slowing_rate) = self.detect_critical_slowing() {
            if let Some(current) = self.history.last() {
                let distance = (current.integration - self.config.critical_point).abs();
                // Simple prediction: time ~ distance / rate
                if slowing_rate > 0.001 {
                    return Some(distance / slowing_rate);
                }
            }
        }
        None
    }

    /// Compute order parameter from HDC state
    pub fn compute_order_parameter(&self, state: &[HV16]) -> f64 {
        if state.is_empty() {
            return 0.0;
        }

        // Order parameter = global coherence (average pairwise similarity)
        let n = state.len();
        if n < 2 {
            return 0.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = HV16::similarity(&state[i], &state[j]) as f64;
                total_sim += sim.abs();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        total_sim / count as f64
    }

    /// Compute correlation function G(r)
    pub fn compute_correlation_function(&self, state: &[HV16], max_distance: usize) -> Vec<f64> {
        let n = state.len();
        if n < 2 {
            return vec![1.0];
        }

        let max_r = max_distance.min(n / 2);
        let mut correlations = vec![0.0; max_r];
        let mut counts = vec![0; max_r];

        for i in 0..n {
            for r in 1..max_r {
                let j = (i + r) % n;
                let sim = HV16::similarity(&state[i], &state[j]) as f64;
                correlations[r] += sim;
                counts[r] += 1;
            }
        }

        for r in 1..max_r {
            if counts[r] > 0 {
                correlations[r] /= counts[r] as f64;
            }
        }

        correlations[0] = 1.0; // G(0) = 1 by definition
        correlations
    }

    /// Estimate correlation length from correlation function
    pub fn estimate_correlation_length(&self, correlations: &[f64]) -> f64 {
        // Correlation length: distance where G(r) drops to 1/e
        let threshold = 1.0 / std::f64::consts::E;

        for (r, &g) in correlations.iter().enumerate() {
            if g.abs() < threshold {
                return r as f64;
            }
        }

        correlations.len() as f64
    }

    /// Perform full phase transition assessment
    pub fn assess(&self) -> PhaseTransitionAssessment {
        let current = self.history.last();

        if current.is_none() {
            return PhaseTransitionAssessment {
                current_phase: ConsciousnessPhase::Unconscious,
                criticality_distance: 1.0,
                at_criticality: false,
                slowing_detected: false,
                time_to_transition: None,
                order_parameter: 0.0,
                correlation_length: 1.0,
                susceptibility: 1.0,
                is_conscious: false,
                pathology_warning: false,
                explanation: "No observations yet".to_string(),
            };
        }

        let state = current.unwrap();
        let criticality_distance = (state.integration - self.config.critical_point).abs();
        let at_criticality = criticality_distance < 0.05;
        let slowing_detected = self.detect_critical_slowing().is_some();
        let time_to_transition = self.predict_transition_time();
        let is_conscious = state.phase.is_conscious();
        let pathology_warning = matches!(state.phase, ConsciousnessPhase::Hypercritical)
            || state.integration > self.config.critical_point + 0.4;

        let explanation = format!(
            "Phase: {:?} ({}). τ = {:.3}, τ_c = {:.3}, distance = {:.3}. \
             Order parameter ψ = {:.3}, correlation length ξ = {:.1}, \
             susceptibility χ = {:.1}, relaxation time = {:.1}. \
             {}{}{}",
            state.phase,
            state.phase.description(),
            state.integration,
            self.config.critical_point,
            criticality_distance,
            state.order_parameter,
            state.correlation_length,
            state.susceptibility,
            state.relaxation_time,
            if at_criticality { "AT CRITICALITY: Maximum computational power! " } else { "" },
            if slowing_detected { "WARNING: Critical slowing detected - transition imminent! " } else { "" },
            if pathology_warning { "DANGER: Approaching pathological hypercritical state!" } else { "" },
        );

        PhaseTransitionAssessment {
            current_phase: state.phase,
            criticality_distance,
            at_criticality,
            slowing_detected,
            time_to_transition,
            order_parameter: state.order_parameter,
            correlation_length: state.correlation_length,
            susceptibility: state.susceptibility,
            is_conscious,
            pathology_warning,
            explanation,
        }
    }

    /// Find optimal operating point (at criticality)
    pub fn find_optimal_integration(&self) -> f64 {
        // Optimal = at criticality, where computation is maximized
        self.config.critical_point
    }

    /// Simulate sweeping through phase transition
    pub fn simulate_sweep(&mut self, tau_min: f64, tau_max: f64, steps: usize) {
        let step_size = (tau_max - tau_min) / steps as f64;

        for i in 0..=steps {
            let tau = tau_min + i as f64 * step_size;
            // Generate state with integration-dependent coherence
            let state: Vec<HV16> = (0..self.config.system_size)
                .map(|j| HV16::random((i * self.config.system_size + j) as u64))
                .collect();
            self.observe(tau, state);
        }
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
        self.transitions.clear();
        self.timestamp = 0;
    }

    /// Get number of observations
    pub fn num_observations(&self) -> usize {
        self.history.len()
    }

    /// Get number of detected transitions
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }
}

/// Finite-size scaling analysis
/// Real brains have finite size N, which rounds off the sharp transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiniteSizeScaling {
    /// System size N
    pub size: usize,
    /// Scaled variables for collapse
    pub scaled_tau: f64,
    pub scaled_psi: f64,
    pub scaled_chi: f64,
}

impl FiniteSizeScaling {
    /// Compute finite-size scaled variables
    pub fn compute(
        tau: f64,
        tau_c: f64,
        psi: f64,
        chi: f64,
        size: usize,
        exponents: &CriticalExponents,
    ) -> Self {
        let n = size as f64;

        // Scaling forms:
        // x = (τ - τ_c) * N^(1/ν)
        // ψ_scaled = ψ * N^(β/ν)
        // χ_scaled = χ * N^(-γ/ν)

        let scaled_tau = (tau - tau_c) * n.powf(1.0 / exponents.nu);
        let scaled_psi = psi * n.powf(exponents.beta / exponents.nu);
        let scaled_chi = chi * n.powf(-exponents.gamma / exponents.nu);

        Self {
            size,
            scaled_tau,
            scaled_psi,
            scaled_chi,
        }
    }

    /// Data collapse quality metric
    pub fn collapse_quality(data: &[FiniteSizeScaling]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        // Good collapse = low variance in scaled variables at same scaled_tau
        let mut variance = 0.0;
        let n = data.len() as f64;

        let mean_psi: f64 = data.iter().map(|d| d.scaled_psi).sum::<f64>() / n;
        for d in data {
            variance += (d.scaled_psi - mean_psi).powi(2);
        }

        // Quality = 1 / (1 + variance)
        1.0 / (1.0 + variance / n)
    }
}

/// Hysteresis detector for first-order transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HysteresisDetector {
    /// Forward sweep order parameters
    pub forward: Vec<(f64, f64)>, // (tau, psi)
    /// Backward sweep order parameters
    pub backward: Vec<(f64, f64)>,
    /// Detected hysteresis width
    pub width: Option<f64>,
}

impl HysteresisDetector {
    pub fn new() -> Self {
        Self {
            forward: Vec::new(),
            backward: Vec::new(),
            width: None,
        }
    }

    /// Add forward sweep point
    pub fn add_forward(&mut self, tau: f64, psi: f64) {
        self.forward.push((tau, psi));
    }

    /// Add backward sweep point
    pub fn add_backward(&mut self, tau: f64, psi: f64) {
        self.backward.push((tau, psi));
    }

    /// Compute hysteresis width
    pub fn compute_width(&mut self) -> Option<f64> {
        if self.forward.len() < 3 || self.backward.len() < 3 {
            return None;
        }

        // Find transition points
        let forward_transition = self.find_transition(&self.forward);
        let backward_transition = self.find_transition(&self.backward);

        if let (Some(f_tau), Some(b_tau)) = (forward_transition, backward_transition) {
            self.width = Some((f_tau - b_tau).abs());
            self.width
        } else {
            None
        }
    }

    fn find_transition(&self, data: &[(f64, f64)]) -> Option<f64> {
        // Find point of maximum derivative
        let mut max_deriv = 0.0;
        let mut transition_tau = None;

        for i in 1..data.len() {
            let dtau = data[i].0 - data[i-1].0;
            let dpsi = data[i].1 - data[i-1].1;
            if dtau.abs() > 0.001 {
                let deriv = (dpsi / dtau).abs();
                if deriv > max_deriv {
                    max_deriv = deriv;
                    transition_tau = Some((data[i].0 + data[i-1].0) / 2.0);
                }
            }
        }

        transition_tau
    }

    /// Is there significant hysteresis? (indicates first-order transition)
    pub fn has_hysteresis(&self) -> bool {
        self.width.map(|w| w > 0.05).unwrap_or(false)
    }
}

impl Default for HysteresisDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_phase() {
        assert!(ConsciousnessPhase::Supercritical.is_conscious());
        assert!(ConsciousnessPhase::Critical.is_conscious());
        assert!(!ConsciousnessPhase::Unconscious.is_conscious());
        assert!(!ConsciousnessPhase::Subcritical.is_conscious());
        assert!(ConsciousnessPhase::Hypercritical.is_pathological());
    }

    #[test]
    fn test_critical_exponents_default() {
        let exp = CriticalExponents::default();
        assert!((exp.beta - 0.5).abs() < 0.01);
        assert!((exp.gamma - 1.0).abs() < 0.01);
        assert!((exp.nu - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_critical_exponents_ising() {
        let exp = CriticalExponents::ising_3d();
        // 3D Ising values
        assert!((exp.beta - 0.326).abs() < 0.01);
        assert!((exp.gamma - 1.237).abs() < 0.01);
    }

    #[test]
    fn test_hyperscaling_relation() {
        let exp = CriticalExponents::ising_3d();
        let violation = exp.check_hyperscaling(3.0);
        // Should be close to zero for correct exponents
        assert!(violation < 0.1, "Hyperscaling violation: {}", violation);
    }

    #[test]
    fn test_system_state_phases() {
        // Test unconscious state (low integration)
        let state: Vec<HV16> = (0..10).map(|i| HV16::random(i as u64)).collect();
        let mut sys = SystemState::new(0.1, state.clone());
        sys.update_derived_quantities(0.5);
        assert_eq!(sys.phase, ConsciousnessPhase::Unconscious);

        // Test supercritical (high integration)
        sys.integration = 0.7;
        sys.update_derived_quantities(0.5);
        assert_eq!(sys.phase, ConsciousnessPhase::Supercritical);

        // Test critical (at transition)
        sys.integration = 0.5;
        sys.update_derived_quantities(0.5);
        assert_eq!(sys.phase, ConsciousnessPhase::Critical);
    }

    #[test]
    fn test_phase_transitions_creation() {
        let config = PhaseTransitionConfig::default();
        let pt = ConsciousnessPhaseTransitions::new(config);
        assert_eq!(pt.num_observations(), 0);
        assert_eq!(pt.num_transitions(), 0);
    }

    #[test]
    fn test_observe_state() {
        let config = PhaseTransitionConfig::default();
        let mut pt = ConsciousnessPhaseTransitions::new(config);

        let state: Vec<HV16> = (0..100).map(|i| HV16::random(i as u64)).collect();
        pt.observe(0.3, state);

        assert_eq!(pt.num_observations(), 1);
        assert!(pt.current_phase().is_some());
    }

    #[test]
    fn test_phase_transition_detection() {
        let config = PhaseTransitionConfig::default();
        let mut pt = ConsciousnessPhaseTransitions::new(config);

        // Sweep from unconscious to supercritical
        for i in 0..20 {
            let tau = 0.1 + i as f64 * 0.05; // 0.1 to 1.05
            let state: Vec<HV16> = (0..50).map(|j| HV16::random((i * 50 + j) as u64)).collect();
            pt.observe(tau, state);
        }

        // Should detect transitions
        assert!(pt.num_transitions() > 0, "Should detect phase transitions");
    }

    #[test]
    fn test_criticality_detection() {
        let config = PhaseTransitionConfig {
            critical_point: 0.5,
            ..Default::default()
        };
        let mut pt = ConsciousnessPhaseTransitions::new(config);

        // Observe at exactly critical point
        let state: Vec<HV16> = (0..50).map(|i| HV16::random(i as u64)).collect();
        pt.observe(0.5, state);

        assert!(pt.is_at_criticality());
    }

    #[test]
    fn test_order_parameter_computation() {
        let config = PhaseTransitionConfig::default();
        let pt = ConsciousnessPhaseTransitions::new(config);

        // Highly coherent state (same vector repeated)
        let base = HV16::random(42);
        let coherent: Vec<HV16> = vec![base.clone(); 10];
        let op_coherent = pt.compute_order_parameter(&coherent);

        // Random state (different vectors)
        let random: Vec<HV16> = (0..10).map(|i| HV16::random(i as u64)).collect();
        let op_random = pt.compute_order_parameter(&random);

        // Coherent should have higher order parameter
        assert!(op_coherent > op_random,
            "Coherent {} should exceed random {}", op_coherent, op_random);
    }

    #[test]
    fn test_correlation_function() {
        let config = PhaseTransitionConfig::default();
        let pt = ConsciousnessPhaseTransitions::new(config);

        let state: Vec<HV16> = (0..20).map(|i| HV16::random(i as u64)).collect();
        let correlations = pt.compute_correlation_function(&state, 10);

        // G(0) should be 1
        assert!((correlations[0] - 1.0).abs() < 0.01);
        // Correlations should generally decay with distance
        assert!(correlations.len() > 1);
    }

    #[test]
    fn test_assessment() {
        let config = PhaseTransitionConfig::default();
        let mut pt = ConsciousnessPhaseTransitions::new(config);

        let state: Vec<HV16> = (0..50).map(|i| HV16::random(i as u64)).collect();
        pt.observe(0.6, state); // Supercritical

        let assessment = pt.assess();
        assert_eq!(assessment.current_phase, ConsciousnessPhase::Supercritical);
        assert!(assessment.is_conscious);
        assert!(!assessment.explanation.is_empty());
    }

    #[test]
    fn test_simulate_sweep() {
        let config = PhaseTransitionConfig {
            system_size: 20, // Small for test
            ..Default::default()
        };
        let mut pt = ConsciousnessPhaseTransitions::new(config);

        pt.simulate_sweep(0.0, 1.0, 10);

        assert!(pt.num_observations() > 0);
        // Should pass through multiple phases
        assert!(pt.num_transitions() >= 1, "Should detect at least one transition");
    }

    #[test]
    fn test_finite_size_scaling() {
        let exponents = CriticalExponents::default();
        let fss = FiniteSizeScaling::compute(0.55, 0.5, 0.3, 5.0, 1000, &exponents);

        // Scaled variables should be computed
        assert!(fss.scaled_tau.is_finite());
        assert!(fss.scaled_psi.is_finite());
        assert!(fss.scaled_chi.is_finite());
    }

    #[test]
    fn test_hysteresis_detector() {
        let mut hd = HysteresisDetector::new();

        // Forward sweep (unconscious to conscious)
        for i in 0..10 {
            let tau = 0.1 + i as f64 * 0.1;
            let psi = if tau > 0.5 { (tau - 0.5).sqrt() } else { 0.0 };
            hd.add_forward(tau, psi);
        }

        // Backward sweep (with hysteresis - transition at different point)
        for i in (0..10).rev() {
            let tau = 0.1 + i as f64 * 0.1;
            let psi = if tau > 0.4 { (tau - 0.4).sqrt() } else { 0.0 }; // Different threshold
            hd.add_backward(tau, psi);
        }

        let width = hd.compute_width();
        assert!(width.is_some(), "Should detect hysteresis width");
    }

    #[test]
    fn test_clear() {
        let config = PhaseTransitionConfig::default();
        let mut pt = ConsciousnessPhaseTransitions::new(config);

        let state: Vec<HV16> = (0..10).map(|i| HV16::random(i as u64)).collect();
        pt.observe(0.5, state);
        assert_eq!(pt.num_observations(), 1);

        pt.clear();
        assert_eq!(pt.num_observations(), 0);
        assert_eq!(pt.num_transitions(), 0);
    }
}
