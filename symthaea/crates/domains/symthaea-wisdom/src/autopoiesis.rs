// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Autopoiesis - Self-Maintenance Monitoring
//!
//! Autopoiesis (from Greek: auto = self, poiesis = production) is the property
//! of a system that continuously produces and maintains itself. This module
//! monitors whether Symthaea is maintaining its operational closure.
//!
//! ## Philosophical Grounding
//!
//! From Maturana & Varela: An autopoietic system is a network of production
//! processes that continuously regenerates the very components that constitute
//! the network, while simultaneously producing and maintaining a boundary that
//! separates the system from its environment.
//!
//! For Symthaea, this means monitoring:
//! - **Operational Closure**: Are internal processes self-consistent?
//! - **Self-Production**: Is the system generating its own cognitive components?
//! - **Boundary Maintenance**: Is the system distinguishing self from environment?
//!
//! ## The Markov Blanket
//!
//! From Friston's FEP: The boundary between self and environment is a
//! "Markov blanket" - a statistical interface through which the system
//! can only know the world through sensory states and act through active states.
//!
//! This module tracks whether this blanket is intact and functioning.

use std::collections::VecDeque;

/// Maximum history length for tracking
const MAX_HISTORY: usize = 100;

/// Threshold below which closure is considered compromised
const CLOSURE_THRESHOLD: f32 = 0.3;

/// Monitor for autopoietic self-maintenance
#[derive(Debug, Clone)]
pub struct AutopoieticMonitor {
    /// Recent prediction errors (indicates model-world mismatch)
    prediction_error_history: VecDeque<f32>,

    /// Recent coherence values (indicates internal consistency)
    coherence_history: VecDeque<f32>,

    /// Count of successful self-production cycles
    production_cycles: u64,

    /// Count of boundary violations (when internal state was externally overwritten)
    boundary_violations: u64,

    /// Current operational closure estimate
    closure_estimate: f32,

    /// Whether the system is in a healthy autopoietic state
    healthy: bool,
}

impl AutopoieticMonitor {
    pub fn new() -> Self {
        Self {
            prediction_error_history: VecDeque::with_capacity(MAX_HISTORY),
            coherence_history: VecDeque::with_capacity(MAX_HISTORY),
            production_cycles: 0,
            boundary_violations: 0,
            closure_estimate: 1.0, // Start healthy
            healthy: true,
        }
    }

    /// Record a cognitive cycle
    pub fn record_cycle(&mut self, prediction_error: f32, coherence: f32) {
        // Add to history
        if self.prediction_error_history.len() >= MAX_HISTORY {
            self.prediction_error_history.pop_front();
        }
        self.prediction_error_history.push_back(prediction_error);

        if self.coherence_history.len() >= MAX_HISTORY {
            self.coherence_history.pop_front();
        }
        self.coherence_history.push_back(coherence);

        self.production_cycles += 1;

        // Update closure estimate
        self.update_closure_estimate();
    }

    /// Record a boundary violation (external override of internal state)
    pub fn record_boundary_violation(&mut self) {
        self.boundary_violations += 1;
        // Temporary reduction in closure estimate
        self.closure_estimate *= 0.9;
        self.update_health();
    }

    /// Check if operational closure is maintained
    pub fn closure_maintained(&self) -> bool {
        self.closure_estimate > CLOSURE_THRESHOLD
    }

    /// Get current closure estimate
    pub fn closure(&self) -> f32 {
        self.closure_estimate
    }

    /// Check if system is in healthy autopoietic state
    pub fn is_healthy(&self) -> bool {
        self.healthy
    }

    /// Get self-production metrics
    pub fn metrics(&self) -> SelfProductionMetrics {
        SelfProductionMetrics {
            total_cycles: self.production_cycles,
            boundary_violations: self.boundary_violations,
            closure_estimate: self.closure_estimate,
            average_prediction_error: self.average_prediction_error(),
            average_coherence: self.average_coherence(),
            healthy: self.healthy,
        }
    }

    /// Get detailed operational closure state
    pub fn operational_closure(&self) -> OperationalClosure {
        OperationalClosure {
            closure_level: self.closure_estimate,
            model_world_alignment: 1.0 - self.average_prediction_error(),
            internal_consistency: self.average_coherence(),
            boundary_integrity: self.boundary_integrity(),
            self_production_rate: self.self_production_rate(),
        }
    }

    // Private methods

    fn update_closure_estimate(&mut self) {
        let avg_error = self.average_prediction_error();
        let avg_coherence = self.average_coherence();
        let boundary = self.boundary_integrity();

        // Closure = high coherence + low prediction error + intact boundary
        // Weighted: coherence is most important for internal closure
        self.closure_estimate =
            (avg_coherence * 0.5) + ((1.0 - avg_error) * 0.3) + (boundary * 0.2);

        self.update_health();
    }

    fn update_health(&mut self) {
        self.healthy = self.closure_estimate > CLOSURE_THRESHOLD
            && self.average_coherence() > 0.4
            && self.boundary_integrity() > 0.5;
    }

    fn average_prediction_error(&self) -> f32 {
        if self.prediction_error_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.prediction_error_history.iter().sum();
        sum / self.prediction_error_history.len() as f32
    }

    fn average_coherence(&self) -> f32 {
        if self.coherence_history.is_empty() {
            return 1.0; // Assume coherent until proven otherwise
        }
        let sum: f32 = self.coherence_history.iter().sum();
        sum / self.coherence_history.len() as f32
    }

    fn boundary_integrity(&self) -> f32 {
        if self.production_cycles == 0 {
            return 1.0;
        }
        // Boundary integrity decreases with violations relative to cycles
        let violation_rate = self.boundary_violations as f32 / self.production_cycles as f32;
        (1.0 - violation_rate).max(0.0)
    }

    fn self_production_rate(&self) -> f32 {
        // Production rate is just cycles per window
        // Normalized to 0-1 where 1 = MAX_HISTORY cycles achieved
        (self.prediction_error_history.len() as f32 / MAX_HISTORY as f32).min(1.0)
    }
}

impl Default for AutopoieticMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Detailed operational closure state
#[derive(Debug, Clone)]
pub struct OperationalClosure {
    /// Overall closure level (0-1)
    pub closure_level: f32,

    /// How well internal model matches external world
    pub model_world_alignment: f32,

    /// Internal consistency of cognitive state
    pub internal_consistency: f32,

    /// Integrity of self/other boundary
    pub boundary_integrity: f32,

    /// Rate of self-production (cognitive cycles)
    pub self_production_rate: f32,
}

impl OperationalClosure {
    /// Check if the system is maintaining itself
    pub fn is_maintaining(&self) -> bool {
        self.closure_level > CLOSURE_THRESHOLD
    }

    /// Get a diagnostic message about closure state
    pub fn diagnostic(&self) -> &'static str {
        if self.closure_level > 0.8 {
            "Healthy autopoietic state - strong operational closure"
        } else if self.closure_level > 0.5 {
            "Moderate autopoietic state - closure maintained but weakened"
        } else if self.closure_level > CLOSURE_THRESHOLD {
            "Weak autopoietic state - closure at risk"
        } else {
            "Autopoietic crisis - operational closure compromised"
        }
    }
}

/// Metrics about self-production
#[derive(Debug, Clone)]
pub struct SelfProductionMetrics {
    pub total_cycles: u64,
    pub boundary_violations: u64,
    pub closure_estimate: f32,
    pub average_prediction_error: f32,
    pub average_coherence: f32,
    pub healthy: bool,
}

impl SelfProductionMetrics {
    /// Violation rate as percentage
    pub fn violation_rate(&self) -> f32 {
        if self.total_cycles == 0 {
            0.0
        } else {
            (self.boundary_violations as f32 / self.total_cycles as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_monitor_is_healthy() {
        let monitor = AutopoieticMonitor::new();
        assert!(monitor.is_healthy());
        assert!(monitor.closure_maintained());
    }

    #[test]
    fn test_record_cycles_updates_estimate() {
        let mut monitor = AutopoieticMonitor::new();

        // Record some good cycles (low error, high coherence)
        for _ in 0..10 {
            monitor.record_cycle(0.1, 0.9);
        }

        assert!(monitor.closure_maintained());
        assert!(monitor.closure() > 0.7);
    }

    #[test]
    fn test_bad_cycles_reduce_closure() {
        let mut monitor = AutopoieticMonitor::new();

        // Record some bad cycles (high error, low coherence)
        for _ in 0..20 {
            monitor.record_cycle(0.9, 0.2);
        }

        assert!(monitor.closure() < 0.5);
    }

    #[test]
    fn test_boundary_violations_affect_health() {
        let mut monitor = AutopoieticMonitor::new();

        // Record many violations
        for _ in 0..10 {
            monitor.record_cycle(0.3, 0.7);
            monitor.record_boundary_violation();
        }

        let metrics = monitor.metrics();
        assert_eq!(metrics.boundary_violations, 10);
        assert!(metrics.closure_estimate < 1.0);
    }

    #[test]
    fn test_operational_closure_diagnostic() {
        let closure = OperationalClosure {
            closure_level: 0.9,
            model_world_alignment: 0.85,
            internal_consistency: 0.9,
            boundary_integrity: 0.95,
            self_production_rate: 0.8,
        };

        assert!(closure.is_maintaining());
        assert!(closure.diagnostic().contains("Healthy"));
    }

    // ── AutopoieticMonitor construction ─────────────────────────────────

    #[test]
    fn test_default_equals_new() {
        let a = AutopoieticMonitor::new();
        let b = AutopoieticMonitor::default();
        assert_eq!(a.closure(), b.closure());
        assert_eq!(a.is_healthy(), b.is_healthy());
    }

    #[test]
    fn test_initial_closure_is_one() {
        let monitor = AutopoieticMonitor::new();
        assert!((monitor.closure() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_initial_metrics() {
        let monitor = AutopoieticMonitor::new();
        let metrics = monitor.metrics();
        assert_eq!(metrics.total_cycles, 0);
        assert_eq!(metrics.boundary_violations, 0);
        assert!(metrics.healthy);
        assert!((metrics.violation_rate() - 0.0).abs() < f32::EPSILON);
    }

    // ── record_cycle ────────────────────────────────────────────────────

    #[test]
    fn test_production_cycles_increment() {
        let mut monitor = AutopoieticMonitor::new();
        for i in 0..5 {
            monitor.record_cycle(0.2, 0.8);
            assert_eq!(monitor.metrics().total_cycles, i + 1);
        }
    }

    #[test]
    fn test_history_bounded_at_max() {
        let mut monitor = AutopoieticMonitor::new();
        for _ in 0..200 {
            monitor.record_cycle(0.1, 0.9);
        }
        // Internal history should not exceed MAX_HISTORY (100)
        let metrics = monitor.metrics();
        assert_eq!(metrics.total_cycles, 200);
        // closure should remain healthy with good cycles
        assert!(monitor.closure_maintained());
    }

    // ── boundary_violation ──────────────────────────────────────────────

    #[test]
    fn test_single_boundary_violation_reduces_closure() {
        let mut monitor = AutopoieticMonitor::new();
        let before = monitor.closure();
        monitor.record_boundary_violation();
        let after = monitor.closure();
        assert!(
            after < before,
            "Violation should reduce closure: {} -> {}",
            before,
            after
        );
    }

    #[test]
    fn test_many_violations_make_unhealthy() {
        let mut monitor = AutopoieticMonitor::new();
        for _ in 0..50 {
            monitor.record_boundary_violation();
        }
        assert!(!monitor.is_healthy());
        assert!(!monitor.closure_maintained());
    }

    // ── operational_closure ─────────────────────────────────────────────

    #[test]
    fn test_operational_closure_initial() {
        let monitor = AutopoieticMonitor::new();
        let oc = monitor.operational_closure();
        assert!(oc.is_maintaining());
        assert!((oc.model_world_alignment - 1.0).abs() < f32::EPSILON);
        assert!((oc.boundary_integrity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_operational_closure_after_bad_cycles() {
        let mut monitor = AutopoieticMonitor::new();
        for _ in 0..20 {
            monitor.record_cycle(0.95, 0.1);
        }
        let oc = monitor.operational_closure();
        assert!(oc.model_world_alignment < 0.2);
        assert!(oc.internal_consistency < 0.2);
    }

    // ── OperationalClosure diagnostics ──────────────────────────────────

    #[test]
    fn test_moderate_diagnostic() {
        let closure = OperationalClosure {
            closure_level: 0.6,
            model_world_alignment: 0.5,
            internal_consistency: 0.6,
            boundary_integrity: 0.7,
            self_production_rate: 0.5,
        };
        assert!(closure.diagnostic().contains("Moderate"));
    }

    #[test]
    fn test_weak_diagnostic() {
        let closure = OperationalClosure {
            closure_level: 0.35,
            model_world_alignment: 0.3,
            internal_consistency: 0.3,
            boundary_integrity: 0.4,
            self_production_rate: 0.3,
        };
        assert!(closure.diagnostic().contains("Weak"));
    }

    #[test]
    fn test_crisis_diagnostic() {
        let closure = OperationalClosure {
            closure_level: 0.1,
            model_world_alignment: 0.1,
            internal_consistency: 0.1,
            boundary_integrity: 0.1,
            self_production_rate: 0.1,
        };
        assert!(closure.diagnostic().contains("crisis"));
    }

    // ── SelfProductionMetrics ───────────────────────────────────────────

    #[test]
    fn test_violation_rate_zero_cycles() {
        let metrics = SelfProductionMetrics {
            total_cycles: 0,
            boundary_violations: 0,
            closure_estimate: 1.0,
            average_prediction_error: 0.0,
            average_coherence: 1.0,
            healthy: true,
        };
        assert!((metrics.violation_rate() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_violation_rate_with_violations() {
        let metrics = SelfProductionMetrics {
            total_cycles: 100,
            boundary_violations: 10,
            closure_estimate: 0.7,
            average_prediction_error: 0.3,
            average_coherence: 0.8,
            healthy: true,
        };
        assert!((metrics.violation_rate() - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_self_production_rate_grows() {
        let mut monitor = AutopoieticMonitor::new();
        let initial_rate = monitor.operational_closure().self_production_rate;
        assert!((initial_rate - 0.0).abs() < f32::EPSILON);

        for _ in 0..50 {
            monitor.record_cycle(0.2, 0.8);
        }
        let after_rate = monitor.operational_closure().self_production_rate;
        assert!(after_rate > 0.0, "Production rate should grow with cycles");
    }
}
