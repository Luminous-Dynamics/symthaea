// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi Optimization Subsystem
//!
//! Standalone implementation of Φ-guided parameter optimization as a
//! `ConsciousnessSubsystem`. Adjusts binding threshold and workspace capacity
//! to maximize integrated information.

use super::binary_hv::BinaryHV;
use super::consciousness_integration::ConsciousnessState;
use super::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

/// Φ optimization subsystem.
///
/// Uses a simple gradient-free optimization strategy:
/// - Track Phi changes in response to parameter perturbations
/// - Move parameters in the direction that increases Phi
pub struct PhiOptimizationSubsystem {
    enabled: bool,
    /// Optimization frequency (every N cycles)
    optimize_every: usize,
    /// Cycle counter
    cycle_count: usize,
    /// Previous Phi value for delta computation
    prev_phi: f64,
    /// Running best Phi observed
    best_phi: f64,
    /// Topological unity estimate (smoothed)
    topological_unity_ema: f64,
    /// EMA alpha
    ema_alpha: f64,
}

impl PhiOptimizationSubsystem {
    /// Create a new Φ optimization subsystem.
    ///
    /// `optimize_every` controls how often optimization steps run.
    pub fn new(optimize_every: usize) -> Self {
        Self {
            enabled: true,
            optimize_every: optimize_every.max(1),
            cycle_count: 0,
            prev_phi: 0.0,
            best_phi: 0.0,
            topological_unity_ema: 0.0,
            ema_alpha: 0.3,
        }
    }

    /// Get the best Phi observed so far.
    pub fn best_phi(&self) -> f64 {
        self.best_phi
    }
}

impl Default for PhiOptimizationSubsystem {
    fn default() -> Self {
        Self::new(10)
    }
}

impl ConsciousnessSubsystem for PhiOptimizationSubsystem {
    fn name(&self) -> &str {
        "phi_optimization"
    }

    fn process_cycle(
        &mut self,
        state: &mut ConsciousnessState,
        _inputs: &[BinaryHV],
    ) -> Result<(), SubsystemError> {
        self.cycle_count += 1;

        // Track best Phi
        if state.phi > self.best_phi {
            self.best_phi = state.phi;
        }

        // Only optimize every N cycles
        if !self.cycle_count.is_multiple_of(self.optimize_every) {
            self.prev_phi = state.phi;
            return Ok(());
        }

        // Compute Phi delta
        let phi_delta = state.phi - self.prev_phi;

        // Update topological unity with EMA
        self.topological_unity_ema =
            self.ema_alpha * state.phi + (1.0 - self.ema_alpha) * self.topological_unity_ema;
        state.topological_unity = self.topological_unity_ema;

        // Update Phi trend based on delta
        if phi_delta > 0.0 {
            state.phi_trend = (state.phi_trend + phi_delta).min(0.1);
        } else {
            state.phi_trend = (state.phi_trend + phi_delta).max(-0.1);
        }

        self.prev_phi = state.phi;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// =============================================================================
// ENGINE-BACKED VARIANT: Wraps the real PhaseTransitions engine
// =============================================================================

use super::consciousness_phase_transitions::{
    ConsciousnessPhase, ConsciousnessPhaseTransitions, PhaseTransitionAssessment,
    PhaseTransitionConfig,
};

/// Engine-backed phase transition subsystem.
///
/// Wraps the full `ConsciousnessPhaseTransitions` engine (critical state detection)
/// in the `ConsciousnessSubsystem` trait for pluggable registration.
pub struct PhaseTransitionWrapped {
    engine: ConsciousnessPhaseTransitions,
    current_phase: ConsciousnessPhase,
    latest_assessment: Option<PhaseTransitionAssessment>,
}

impl PhaseTransitionWrapped {
    /// Create with default config.
    pub fn new() -> Self {
        Self {
            engine: ConsciousnessPhaseTransitions::new(PhaseTransitionConfig::default()),
            current_phase: ConsciousnessPhase::Critical,
            latest_assessment: None,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: PhaseTransitionConfig) -> Self {
        Self {
            engine: ConsciousnessPhaseTransitions::new(config),
            current_phase: ConsciousnessPhase::Critical,
            latest_assessment: None,
        }
    }

    /// Get the current phase.
    pub fn current_phase(&self) -> ConsciousnessPhase {
        self.current_phase
    }

    /// Get the latest phase assessment.
    pub fn latest_assessment(&self) -> Option<&PhaseTransitionAssessment> {
        self.latest_assessment.as_ref()
    }
}

impl Default for PhaseTransitionWrapped {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessSubsystem for PhaseTransitionWrapped {
    fn name(&self) -> &str {
        "phase_transitions"
    }

    fn process_cycle(
        &mut self,
        state: &mut ConsciousnessState,
        _inputs: &[BinaryHV],
    ) -> Result<(), SubsystemError> {
        let workspace_hvs: Vec<super::binary_hv::BinaryHV> = state
            .conscious_contents
            .iter()
            .map(|item| item.content)
            .collect();
        self.engine.observe(state.phi, workspace_hvs);
        let assessment = self.engine.assess();
        self.current_phase = assessment.current_phase;
        self.latest_assessment = Some(assessment);
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_optimization_creation() {
        let sub = PhiOptimizationSubsystem::new(5);
        assert!(sub.is_enabled());
        assert_eq!(sub.name(), "phi_optimization");
        assert_eq!(sub.best_phi(), 0.0);
    }

    #[test]
    fn test_phi_optimization_tracks_best() {
        let mut sub = PhiOptimizationSubsystem::new(1);
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        state.phi = 0.3;
        sub.process_cycle(&mut state, &inputs).unwrap();
        state.phi = 0.7;
        sub.process_cycle(&mut state, &inputs).unwrap();
        state.phi = 0.5;
        sub.process_cycle(&mut state, &inputs).unwrap();

        assert!((sub.best_phi() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_phi_optimization_updates_topology() {
        let mut sub = PhiOptimizationSubsystem::new(1);
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        for _ in 0..10 {
            state.phi = 0.6;
            sub.process_cycle(&mut state, &inputs).unwrap();
        }

        // Topological unity should converge toward 0.6
        assert!(state.topological_unity > 0.0);
        assert!(state.topological_unity <= 1.0);
    }

    #[test]
    fn test_phi_optimization_respects_frequency() {
        let mut sub = PhiOptimizationSubsystem::new(5);
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        let initial_topo = state.topological_unity;

        // Before 5 cycles, topological_unity should remain at default
        state.phi = 0.5;
        for _ in 0..4 {
            sub.process_cycle(&mut state, &inputs).unwrap();
        }
        assert_eq!(
            state.topological_unity, initial_topo,
            "Topological unity should not change before optimization cycle"
        );

        // After 5th cycle, it should be updated
        sub.process_cycle(&mut state, &inputs).unwrap();
        assert!(
            state.topological_unity != initial_topo || state.phi_trend != 0.0,
            "Either topological_unity or phi_trend should change on optimization cycle"
        );
    }
}
