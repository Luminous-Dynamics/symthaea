// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint-space controller + **dual-channel cautery interlock.**
//!
//! Per post-shipment industry research (see memory
//! `symtropy_phase_c2_flight_demo.md`, Apr 18), regulators expect
//! diverse redundancy on safety-critical interlocks: two independent
//! channels that must BOTH agree before a hazardous action fires.
//! The surgical demo's cautery energy delivery is exactly this class
//! of action (ISO/IEC 80601-2-77 governs surgical robot safety; the
//! foot-pedal deadman on production systems is a hardware-independent
//! channel from the robot's own software).
//!
//! We therefore gate cautery on BOTH:
//!   1. **Epistemic channel** — `SurgicalSafetyLevel::cautery_allowed()`,
//!      which requires Φ > 0.6 (FullControl tier). This is the
//!      "model-derived confidence" side.
//!   2. **Hardware-limit channel** — a pure state-threshold check that
//!      does NOT read Φ: distance-to-critical-structure > 5 mm AND
//!      tip-force magnitude < 2 N. This is the "independent safety
//!      envelope" side.
//!
//! Either channel returning `false` blocks cautery. Both channels'
//! decisions are surfaced to the HUD so failure modes are legible.

use symthaea_surgical::types::{SurgicalCommand, SurgicalSafetyLevel, SurgicalState, NUM_JOINTS};

/// Hard-limit thresholds for the non-Φ channel. Deliberately simple:
/// these are the values a safety case would argue for independently of
/// whatever the consciousness engine is reporting.
pub const MIN_CRITICAL_STRUCTURE_MM: f64 = 5.0;
pub const MAX_TIP_FORCE_N: f64 = 2.0;

pub struct CauteryProcedureController {
    /// Joint-space target (close to tissue, slight pitch for cautery angle).
    pub target_angles: [f64; NUM_JOINTS],
    pub kp: f64,
    pub kd: f64,
}

impl Default for CauteryProcedureController {
    fn default() -> Self {
        // Joints 0..5 = shoulder, elbow, wrist pitch, wrist yaw, tool roll, jaw.
        // Target pose drops the tip into the tissue region for the procedure.
        Self {
            target_angles: [0.25, 0.55, -0.35, 0.20, 0.0, 0.0],
            kp: 2.5,
            kd: 1.0,
        }
    }
}

/// Independent hard-limit channel: returns `true` only when the pure
/// geometric + contact-force state is within safe bounds. Does NOT read
/// Φ — this is the diverse-redundancy second channel.
pub fn hardware_cautery_gate(state: &SurgicalState) -> bool {
    let dist_ok = state.critical_structure_distance > MIN_CRITICAL_STRUCTURE_MM;
    let force_ok = state.force_magnitude() < MAX_TIP_FORCE_N;
    dist_ok && force_ok
}

/// Result of one cautery interlock evaluation, exposed to the UI so both
/// channels can be displayed.
#[derive(Debug, Clone, Copy)]
pub struct InterlockDecision {
    pub phi_channel: bool,
    pub hardware_channel: bool,
    pub combined: bool,
    pub hw_dist_mm: f64,
    pub hw_force_n: f64,
}

impl CauteryProcedureController {
    /// Compute the motor command + interlock decision. Returning the
    /// decision (instead of just baking it into the cautery power) means
    /// the UI layer can show WHY cautery is armed or blocked.
    pub fn compute(
        &self,
        state: &SurgicalState,
        level: SurgicalSafetyLevel,
    ) -> (SurgicalCommand, InterlockDecision) {
        let mut torques = [0.0f32; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            let err = self.target_angles[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];
            let raw = self.kp * err - self.kd * vel;
            // Normalize into [-1, 1] — simulator multiplies by max_joint_torques.
            torques[i] = (raw / 3.0).clamp(-1.0, 1.0) as f32;
        }

        // Jaw closes slowly toward 0.6 once we're approximately in pose.
        let in_pose_err: f64 = self
            .target_angles
            .iter()
            .zip(state.joint_angles.iter())
            .map(|(t, q)| (t - q).abs())
            .sum();
        let jaw_target = if in_pose_err < 0.6 { 0.6 } else { 0.0 };

        // Request cautery when in pose; the TWO interlock channels below
        // decide whether it actually fires.
        let cautery_request = if in_pose_err < 0.3 { 1.0 } else { 0.0 };

        // Apply the platform's torque gain (single-channel — torque
        // scaling is not safety-critical in the same way energy delivery is).
        let gain = level.torque_gain();
        for t in &mut torques {
            *t *= gain;
        }

        // Channel 1: epistemic (Φ-derived).
        let phi_channel = level.cautery_allowed();
        // Channel 2: hardware limits (Φ-independent).
        let hardware_channel = hardware_cautery_gate(state);
        // Combined: AND — either channel alone can block.
        let combined = phi_channel && hardware_channel;

        let cautery = if combined { cautery_request } else { 0.0 };

        let decision = InterlockDecision {
            phi_channel,
            hardware_channel,
            combined,
            hw_dist_mm: state.critical_structure_distance,
            hw_force_n: state.force_magnitude(),
        };

        let cmd = SurgicalCommand {
            joint_torques: torques,
            jaw: jaw_target as f32,
            cautery,
        };

        (cmd, decision)
    }
}

// ─── Regression tests for the safety invariant ────────────────────────────
//
// These tests lock in the dual-channel interlock behavior so a future
// refactor that silently collapses the logic back to single-channel
// (e.g., by inlining `level.cautery_allowed()` without the HW gate, or
// the reverse) fails CI. The invariant is simple enough to state directly:
// **cautery fires only when BOTH channels approve**.

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a standing-pose `SurgicalState` with override hooks for the two
    /// hardware-channel inputs. Anything unspecified defaults to "safe".
    fn state_with(critical_structure_distance: f64, tip_force: [f64; 3]) -> SurgicalState {
        let mut s = SurgicalState::home();
        s.critical_structure_distance = critical_structure_distance;
        s.tip_force = tip_force;
        s
    }

    #[test]
    fn hw_gate_passes_when_dist_and_force_are_safe() {
        let s = state_with(20.0, [0.0, 0.0, 0.0]);
        assert!(hardware_cautery_gate(&s));
    }

    #[test]
    fn hw_gate_blocks_when_dist_at_boundary() {
        // At exactly MIN_CRITICAL_STRUCTURE_MM the gate must NOT pass —
        // strict inequality is load-bearing for the safety margin.
        let s = state_with(MIN_CRITICAL_STRUCTURE_MM, [0.0, 0.0, 0.0]);
        assert!(!hardware_cautery_gate(&s));
    }

    #[test]
    fn hw_gate_blocks_when_dist_below_threshold() {
        let s = state_with(MIN_CRITICAL_STRUCTURE_MM - 0.1, [0.0, 0.0, 0.0]);
        assert!(!hardware_cautery_gate(&s));
    }

    #[test]
    fn hw_gate_blocks_when_force_at_boundary() {
        // force_magnitude() == MAX_TIP_FORCE_N must block — again the
        // strict inequality is the safety margin.
        let s = state_with(20.0, [MAX_TIP_FORCE_N, 0.0, 0.0]);
        assert!(!hardware_cautery_gate(&s));
    }

    #[test]
    fn hw_gate_blocks_when_force_above_threshold() {
        let s = state_with(20.0, [MAX_TIP_FORCE_N + 0.5, 0.0, 0.0]);
        assert!(!hardware_cautery_gate(&s));
    }

    #[test]
    fn hw_gate_sums_force_components_as_magnitude() {
        // A (1.5, 1.5, 0) force has magnitude ~2.12 > 2.0 — should block,
        // even though each individual component is < MAX_TIP_FORCE_N.
        let s = state_with(20.0, [1.5, 1.5, 0.0]);
        assert!(!hardware_cautery_gate(&s));
        // Similarly, (1.0, 1.0, 1.0) is magnitude ~1.73 < 2.0 — should pass.
        let s2 = state_with(20.0, [1.0, 1.0, 1.0]);
        assert!(hardware_cautery_gate(&s2));
    }

    // ── Full interlock decision: both channels must agree ──────────────
    //
    // We run the controller's `compute()` and inspect the `InterlockDecision`
    // it returns. `compute()` itself takes a `SurgicalSafetyLevel` (the
    // Φ channel) plus a state (for the HW channel), so we can independently
    // drive each channel and check the combined outcome.

    #[test]
    fn both_channels_on_allows_cautery() {
        // Full control + safe HW state → combined=true.
        let ctrl = CauteryProcedureController::default();
        // State at the controller's target pose AND safe HW conditions.
        let mut s = SurgicalState::home();
        s.joint_angles = ctrl.target_angles;
        s.critical_structure_distance = 20.0;
        s.tip_force = [0.0, 0.0, 0.0];
        let (_, d) = ctrl.compute(&s, SurgicalSafetyLevel::FullControl);
        assert!(d.phi_channel);
        assert!(d.hardware_channel);
        assert!(d.combined);
    }

    #[test]
    fn phi_channel_off_blocks_even_with_safe_hw() {
        // Reduced tier → Φ channel false, regardless of HW state.
        let ctrl = CauteryProcedureController::default();
        let mut s = SurgicalState::home();
        s.joint_angles = ctrl.target_angles;
        s.critical_structure_distance = 20.0;
        s.tip_force = [0.0, 0.0, 0.0];
        let (_, d) = ctrl.compute(&s, SurgicalSafetyLevel::Reduced);
        assert!(!d.phi_channel);
        assert!(d.hardware_channel);
        assert!(!d.combined, "either channel's NO must block");
    }

    #[test]
    fn hw_channel_off_blocks_even_with_full_phi() {
        // Full control tier → Φ channel true, but HW blocks due to close
        // critical structure. Combined must still be blocked.
        let ctrl = CauteryProcedureController::default();
        let mut s = SurgicalState::home();
        s.joint_angles = ctrl.target_angles;
        s.critical_structure_distance = MIN_CRITICAL_STRUCTURE_MM - 0.1;
        s.tip_force = [0.0, 0.0, 0.0];
        let (_, d) = ctrl.compute(&s, SurgicalSafetyLevel::FullControl);
        assert!(d.phi_channel);
        assert!(!d.hardware_channel);
        assert!(!d.combined, "either channel's NO must block");
    }

    #[test]
    fn both_channels_off_blocks() {
        let ctrl = CauteryProcedureController::default();
        let mut s = SurgicalState::home();
        s.critical_structure_distance = 0.5; // too close
        s.tip_force = [5.0, 0.0, 0.0]; // too hard
        let (_, d) = ctrl.compute(&s, SurgicalSafetyLevel::Freeze);
        assert!(!d.phi_channel);
        assert!(!d.hardware_channel);
        assert!(!d.combined);
    }

    #[test]
    fn cautery_power_is_zero_when_combined_is_blocked() {
        // When combined=false, the returned SurgicalCommand.cautery MUST be
        // zero. This is the final wire-level assertion on the safety claim —
        // the interlock reaches all the way to the physics input.
        let ctrl = CauteryProcedureController::default();
        let mut s = SurgicalState::home();
        s.joint_angles = ctrl.target_angles;
        s.critical_structure_distance = MIN_CRITICAL_STRUCTURE_MM - 0.1;
        s.tip_force = [0.0, 0.0, 0.0];
        let (cmd, d) = ctrl.compute(&s, SurgicalSafetyLevel::FullControl);
        assert!(!d.combined);
        assert_eq!(cmd.cautery, 0.0);
    }
}
