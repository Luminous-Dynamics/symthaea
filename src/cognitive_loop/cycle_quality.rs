// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quality-aware adaptive processing + end-of-cycle homeostasis.
//!
//! Extracted from cycle_phase_feedback.rs — all logic and behavior preserved exactly.
//!
//! Contains:
//! - Dissipative health + phenomenal binding → learning gate (Prigogine 1977)
//! - Coherence velocity → dynamic gating (Kelso 1995)
//! - End-of-cycle homeostasis: drift clamp, boredom, exploration (Turrigiano 2004)

use std::time::Instant;

use super::CognitiveLoopService;

// ═══════════════════════════════════════════════════════════════════════════════
// Result struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the quality + homeostasis phase (Phase 16 + end-of-cycle).
pub(crate) struct QualityPhaseResult {
    pub(crate) dissipative_lr_factor: f32,
    pub(crate) dissipative_health_gated: bool,
    pub(crate) coherence_velocity: f32,
    pub(crate) coherence_velocity_gated: bool,
}

impl CognitiveLoopService {
    /// Quality-aware adaptive processing + end-of-cycle homeostasis.
    ///
    /// Covers:
    /// - Dissipative health + phenomenal binding → learning gate (Prigogine 1977)
    /// - Coherence velocity → dynamic gating (Kelso 1995)
    /// - End-of-cycle homeostasis: drift clamp, boredom, exploration (Turrigiano 2004)
    ///
    /// `dissipative_health` and `phenomenal_binding_strength` are read from
    /// `self.carryover.quality` (written by earlier phases). The new values
    /// for `dissipative_health` and `phenomenal_binding_strength` are written
    /// back to carryover for the next cycle.
    pub(super) fn run_quality_and_homeostasis(
        &mut self,
        coherence: f32,
        temporal_discontinuity: bool,
        exploration_urge_start: f32,
        homeostasis_pull_strength: f32,
        dissipative_health: f64,
        phenomenal_binding_strength: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> QualityPhaseResult {
        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 16: Quality-Aware Adaptive Processing
        // ═══════════════════════════════════════════════════════════════════════

        // ── Task #55: Dissipative health + phenomenal binding → learning gate ──
        // Science: Prigogine (1977) — dissipative structures require stability to consolidate.
        // When the system's thermodynamic health is low OR phenomenal binding is fragmented,
        // dampen learning to prevent cementing unstable patterns.
        let dissipative_lr_factor;
        let dissipative_health_gated;
        {
            let dh = self.carryover.quality.last_dissipative_health as f32;
            let pb = self.carryover.quality.last_phenomenal_binding as f32;
            if dh < 0.5 && pb < 0.6 && self.stats.total_cycles > 50 {
                // Both unhealthy → strong dampening (up to -30%)
                let dampening = (1.0 - dh) * (1.0 - pb) * 0.3;
                dissipative_lr_factor = (1.0 - dampening).max(0.7);
                self.carryover.learning.subsystem_lr_factor *= dissipative_lr_factor;
                dissipative_health_gated = true;
                self.stats.dissipative_health_gated_count += 1;
            } else if dh < 0.5 || pb < 0.4 {
                // One unhealthy → mild dampening (up to -15%)
                let dampening = if dh < 0.5 {
                    (0.5 - dh) * 0.15
                } else {
                    (0.4 - pb) * 0.15
                };
                dissipative_lr_factor = (1.0 - dampening).max(0.85);
                self.carryover.learning.subsystem_lr_factor *= dissipative_lr_factor;
                dissipative_health_gated = true;
                self.stats.dissipative_health_gated_count += 1;
            } else {
                dissipative_lr_factor = 1.0;
                dissipative_health_gated = false;
            }
            // Cache for next cycle
            self.carryover.quality.last_dissipative_health = dissipative_health;
            self.carryover.quality.last_phenomenal_binding = phenomenal_binding_strength;
        }

        // ── Task #57: Coherence velocity → dynamic gating ─────────────────────
        // Science: Kelso (1995) — phase transitions in coordination dynamics.
        // Track rate of coherence change; rapid drops indicate instability.
        let coherence_velocity;
        let coherence_velocity_gated;
        {
            let prev_coh = self.carryover.quality.last_coherence;
            coherence_velocity = coherence - prev_coh;
            self.carryover.quality.coherence_velocity = coherence_velocity;
            self.carryover.quality.last_coherence = coherence;

            // If temporal discontinuity detected OR rapid coherence drop,
            // raise confidence requirements and tighten exploration
            if temporal_discontinuity || coherence_velocity < -0.15 {
                let severity = if temporal_discontinuity {
                    0.5 + (-coherence_velocity).max(0.0)
                } else {
                    (-coherence_velocity - 0.15).min(0.5)
                };
                // Dampen confidence proportional to severity
                self.scale_confidence("coherence_vel_drop", 1.0 - severity * 0.1);
                // Raise learning threshold (require higher error to trigger training)
                self.scale_threshold("coherence_vel_drop", 1.0 + severity * 0.2);
                coherence_velocity_gated = true;
                self.stats.coherence_velocity_gated_count += 1;
            } else if coherence_velocity > 0.1 {
                // Rapidly improving coherence → mild confidence boost (proactive).
                // Bar (2009): rising prediction consistency signals successful model update.
                self.adjust_confidence("coherence_vel_rise", 0.01);
                coherence_velocity_gated = false;
            } else {
                coherence_velocity_gated = false;
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // END-OF-CYCLE HOMEOSTASIS: Prevent asymmetric drift and runaway spirals
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();

        // Guard: clamp total per-cycle confidence drift to ±15%.
        // prediction_confidence is modified ~25 times per cycle by different subsystems,
        // each reading a different intermediate value. Without bounding, subsystems can
        // compound to produce wild swings. This ensures no single cycle changes confidence
        // by more than 15% regardless of subsystem ordering.
        // Science: Homeostatic plasticity (Turrigiano 2004) — bound rate of change.
        {
            let confidence_start = self.carryover.learning.prediction_confidence;
            let max_drift = confidence_start * 0.15 + 0.02; // ±15% + 2% floor
            let clamped = self
                .prediction_confidence
                .clamp(confidence_start - max_drift, confidence_start + max_drift)
                .clamp(0.0, 1.0);
            self.set_confidence("homeostasis_drift_clamp", clamped as f32);
        }

        // Clamp attention_sensitivity to [0.5, 2.0] after all modifications.
        // 10+ subsystems multiply this field per cycle; without bounding, it can drift
        // to extreme values. Science: Weber-Fechner law — perception has bounded dynamic range.
        self.behavior.adaptive_behavior.attention_sensitivity = self
            .behavior
            .adaptive_behavior
            .attention_sensitivity
            .clamp(0.5, 2.0);

        // Boredom↔confidence homeostasis (Turrigiano 2004 — homeostatic plasticity)
        // High boredom signals stagnation → dampen confidence (system shouldn't be confident
        // when stuck in repetitive states). Prevents confidence runaway from accumulated boosts.
        if self.behavior.curiosity_drive.boredom > 0.7 {
            let boredom_dampen = (self.behavior.curiosity_drive.boredom - 0.7) * 0.15;
            self.scale_confidence("boredom_dampen", (1.0 - boredom_dampen).max(0.85));
        }

        // Boredom homeostasis: slow drift toward neutral (0.5) prevents monotonic saturation.
        // Phase 18: urgency-adaptive pull (Cruise=1.5×, Critical=0.6×)
        self.behavior.curiosity_drive.boredom +=
            (0.5 - self.behavior.curiosity_drive.boredom) * 0.02 * homeostasis_pull_strength;

        // Exploration urge per-cycle budget: clamp total change to ±0.5.
        // 15+ subsystems write exploration_urge per cycle; without bounding, cumulative
        // nudges can pin it to 0.0 or 1.0. Science: Homeostatic control of exploration.
        let start64 = exploration_urge_start as f64;
        let clamped = self
            .curiosity_drive()
            .exploration_urge
            .clamp((start64 - 0.5).max(0.0), (start64 + 0.5).min(1.0));
        self.set_exploration("homeostasis_budget_clamp", clamped as f32);

        // Exploration urge homeostasis: slow drift toward neutral (0.3) prevents saturation.
        // Phase 18: urgency-adaptive pull (Cruise=1.5×, Critical=0.6×)
        let drift_delta = (0.3 - self.behavior.curiosity_drive.exploration_urge)
            * 0.03
            * homeostasis_pull_strength as f64;
        self.adjust_exploration("homeostasis_drift", drift_delta as f32);

        // Store urgency for next cycle's hysteresis
        // NOTE: urgency is stored by the caller (perception.urgency), not here,
        // because this method doesn't receive urgency. The caller handles:
        //   self.carryover.urgency.urgency = perception.urgency;
        module_timings.homeostasis = _t.elapsed().as_micros() as u64;

        QualityPhaseResult {
            dissipative_lr_factor,
            dissipative_health_gated,
            coherence_velocity,
            coherence_velocity_gated,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ModuleTimings};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn test_quality_healthy_system_no_gating() {
        let mut svc = make_service();
        svc.carryover.quality.last_dissipative_health = 0.9;
        svc.carryover.quality.last_phenomenal_binding = 0.8;
        svc.carryover.quality.last_coherence = 0.7;
        svc.stats.total_cycles = 100;
        let mut timings = ModuleTimings::default();
        let result = svc.run_quality_and_homeostasis(0.7, false, 0.5, 1.0, 0.9, 0.8, &mut timings);
        assert_eq!(result.dissipative_lr_factor, 1.0);
        assert!(!result.dissipative_health_gated);
    }

    #[test]
    fn test_quality_unhealthy_both_gated() {
        let mut svc = make_service();
        svc.carryover.quality.last_dissipative_health = 0.2;
        svc.carryover.quality.last_phenomenal_binding = 0.3;
        svc.carryover.quality.last_coherence = 0.7;
        svc.stats.total_cycles = 100;
        let mut timings = ModuleTimings::default();
        let result = svc.run_quality_and_homeostasis(0.7, false, 0.5, 1.0, 0.2, 0.3, &mut timings);
        assert!(result.dissipative_lr_factor < 1.0);
        assert!(result.dissipative_lr_factor >= 0.7);
        assert!(result.dissipative_health_gated);
    }

    #[test]
    fn test_quality_unhealthy_single_dimension() {
        let mut svc = make_service();
        svc.carryover.quality.last_dissipative_health = 0.3;
        svc.carryover.quality.last_phenomenal_binding = 0.8;
        svc.carryover.quality.last_coherence = 0.7;
        svc.stats.total_cycles = 100;
        let mut timings = ModuleTimings::default();
        let result = svc.run_quality_and_homeostasis(0.7, false, 0.5, 1.0, 0.3, 0.8, &mut timings);
        assert!(result.dissipative_lr_factor < 1.0);
        assert!(result.dissipative_lr_factor >= 0.85);
        assert!(result.dissipative_health_gated);
    }

    #[test]
    fn test_coherence_velocity_stable() {
        let mut svc = make_service();
        svc.carryover.quality.last_coherence = 0.7;
        svc.stats.total_cycles = 100;
        let mut timings = ModuleTimings::default();
        // Small coherence change → no gating
        let result = svc.run_quality_and_homeostasis(0.72, false, 0.5, 1.0, 0.9, 0.8, &mut timings);
        assert!(!result.coherence_velocity_gated);
        assert!((result.coherence_velocity - 0.02).abs() < 0.01);
    }

    #[test]
    fn test_coherence_velocity_rapid_drop() {
        let mut svc = make_service();
        svc.carryover.quality.last_coherence = 0.7;
        svc.stats.total_cycles = 100;
        let mut timings = ModuleTimings::default();
        // Large coherence drop → gating
        let result = svc.run_quality_and_homeostasis(0.4, false, 0.5, 1.0, 0.9, 0.8, &mut timings);
        assert!(result.coherence_velocity_gated);
        assert!(result.coherence_velocity < -0.15);
    }

    #[test]
    fn test_temporal_discontinuity_gates() {
        let mut svc = make_service();
        svc.carryover.quality.last_coherence = 0.7;
        svc.stats.total_cycles = 100;
        let mut timings = ModuleTimings::default();
        // Temporal discontinuity → gating even with stable coherence
        let result = svc.run_quality_and_homeostasis(0.7, true, 0.5, 1.0, 0.9, 0.8, &mut timings);
        assert!(result.coherence_velocity_gated);
    }

    #[test]
    fn test_homeostasis_exploration_clamp() {
        let mut svc = make_service();
        svc.carryover.quality.last_coherence = 0.5;
        svc.stats.total_cycles = 100;
        // Start with extreme exploration
        svc.behavior.curiosity_drive.exploration_urge = 0.1;
        let mut timings = ModuleTimings::default();
        let _result = svc.run_quality_and_homeostasis(0.5, false, 0.1, 1.0, 0.9, 0.8, &mut timings);
        // Exploration should be clamped within ±0.5 of start (0.1)
        assert!(svc.behavior.curiosity_drive.exploration_urge >= 0.0);
        assert!(svc.behavior.curiosity_drive.exploration_urge <= 0.6);
    }
}
