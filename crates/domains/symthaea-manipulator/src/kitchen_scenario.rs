// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kitchen scenario — deformable/fragile-object grasping under hazard gating.
//!
//! Per `MANIPULATOR_KITCHEN_SCENARIO_PLAN_2026-07-09.md` Phase 0: a scenario, not a new platform
//! (a kitchen arm is still a 7-DOF arm), following `co_assembly.rs`'s pattern of a small
//! task-scenario module layered on the existing arm/simulator/safety stack.
//!
//! The core idea: an environmental hazard (a hot pan, a knife edge) should force a more
//! conservative [`MotorSafetyLevel`] *regardless of Φ*. Since `MotorSafetyLevel` already derives
//! `Ord` in severity order (`Green < Yellow < Orange < Red`), composing "the worse of the
//! Φ-derived tier and the hazard-derived tier" needs no new machinery — see [`hazard_tier`] and
//! its use via `.max()` at the call site.
//!
//! Workspace, force, speed, and human-zone limits are enforced by the live
//! `ManipulatorSafetySupervisor`. Hazard tiers compose into that same canonical
//! `MotorSafetyLevel`, so kitchen hazards and geometric safety cannot diverge.

use crate::embodiment::MotorSafetyLevel;
use symthaea_thermofluids::thermal::convection_heat_rate;

/// Ambient temperature, °C — what the gripper's thermal exposure relaxes
/// toward when nothing hot is held.
pub const AMBIENT_TEMP_C: f64 = 20.0;

/// Burn-risk threshold, °C. A conservative "hot to the touch" scald-risk line —
/// well below boiling, consistent with the kind of surface-temperature caution
/// literature cited for `symthaea-culinary`'s coagulation thresholds (McGee
/// 2004-adjacent range: skin contact burn risk begins well under 100 °C).
pub const SCALD_RISK_TEMP_C: f64 = 60.0;

/// An object the manipulator may be holding, with the properties that make it
/// hazardous independent of the arm's own Φ/confidence state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KitchenObject {
    pub temperature_c: f64,
    /// The most force (N) this object can take before crushing/deforming
    /// unacceptably (e.g. a tomato vs. a cast-iron pan).
    pub max_safe_grip_force_n: f64,
    pub is_sharp: bool,
}

impl KitchenObject {
    pub fn is_hot(&self) -> bool {
        self.temperature_c >= SCALD_RISK_TEMP_C
    }
}

/// The safety tier this object's own hazards demand, independent of Φ.
///
/// `None` (holding nothing) is `Green` — a hazard tier is a floor, not a
/// default penalty. A sharp OR hot object floors at `Orange`; both floor at
/// `Red`. Compose with the Φ-derived tier via `.max()` at the call site so the
/// arm is gated by whichever signal is currently more conservative.
pub fn hazard_tier(object: Option<&KitchenObject>) -> MotorSafetyLevel {
    let Some(obj) = object else {
        return MotorSafetyLevel::Green;
    };
    match (obj.is_sharp, obj.is_hot()) {
        (true, true) => MotorSafetyLevel::Red,
        (true, false) => MotorSafetyLevel::Orange,
        (false, true) => MotorSafetyLevel::Orange,
        (false, false) => MotorSafetyLevel::Green,
    }
}

/// Full-authority grip force (N) at `Green` with nothing else constraining it —
/// the same Panda-class full-force figure already used elsewhere in this crate
/// (`workspace_safety.rs`'s Green row, though that module itself is dead code —
/// see the module doc; only the sourced number is reused here, not its path).
pub const PLATFORM_MAX_GRIP_FORCE_N: f64 = 87.0;

/// Grip-force cap (N) this safety tier alone permits, independent of any held
/// object. `Red` is 0 — `SafeFallback::GravityHold` freezes the gripper at its
/// current opening before a new grip command is even considered, same as
/// `admittance.rs`'s `compliance_gain` documents for its own Red entry.
fn tier_grip_force_cap(tier: MotorSafetyLevel) -> f64 {
    match tier {
        MotorSafetyLevel::Green => PLATFORM_MAX_GRIP_FORCE_N,
        MotorSafetyLevel::Yellow => 20.0,
        MotorSafetyLevel::Orange => 5.0,
        MotorSafetyLevel::Red => 0.0,
    }
}

/// Clamp a commanded gripper value so the *squeeze* force it implies never
/// exceeds the lower of the safety tier's own cap and the held object's crush
/// threshold.
///
/// `ManipulatorCommand::gripper` is `0.0 = fully closed (max squeeze force),
/// 1.0 = fully open (zero force)` (see `types.rs`), so implied force is
/// `(1 - gripper) * PLATFORM_MAX_GRIP_FORCE_N`. Clamping never *closes* the
/// gripper more than commanded — only forces it more open when the commanded
/// value would squeeze harder than the effective cap allows.
pub fn clamp_grip_command(
    commanded_gripper: f32,
    tier: MotorSafetyLevel,
    held_object: Option<&KitchenObject>,
) -> f32 {
    let object_cap = held_object
        .map(|o| o.max_safe_grip_force_n)
        .unwrap_or(f64::INFINITY);
    let effective_cap_n = tier_grip_force_cap(tier).min(object_cap);
    let min_safe_gripper = 1.0 - (effective_cap_n / PLATFORM_MAX_GRIP_FORCE_N).clamp(0.0, 1.0);
    commanded_gripper.max(min_safe_gripper as f32)
}

/// Convective coefficient (W·m⁻²·K⁻¹) and contact-area/mass/specific-heat
/// figures for a small gripper fingertip mass — the same lumped-capacitance
/// parameterization style `symthaea-culinary::dynamics::NewtonCooling` uses,
/// scaled down for a much smaller thermal mass than a sauce pot.
const GRIPPER_H: f64 = 60.0;
const GRIPPER_AREA_M2: f64 = 0.01;
const GRIPPER_MASS_KG: f64 = 0.05;
const GRIPPER_SPECIFIC_HEAT: f64 = 900.0; // aluminum-ish fingertip

/// The gripper's own thermal exposure (°C) — a real, if simplified, lumped-
/// capacitance state evolved by Newton's law of cooling (reusing
/// `symthaea-thermofluids::thermal::convection_heat_rate` directly, not
/// re-deriving it) toward whatever it is currently in contact with: the held
/// object's temperature while holding, [`AMBIENT_TEMP_C`] otherwise.
///
/// This is a separate, additive signal from [`hazard_tier`] — Phase 1's gating
/// already reacts instantly to a *held object's own* temperature; this models
/// the gripper's *own* accumulated/decaying exposure over time, e.g. for
/// telemetry or future extensions, not (yet) fed back into gating itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GripperThermalState {
    pub exposure_c: f64,
}

impl GripperThermalState {
    pub fn ambient() -> Self {
        Self {
            exposure_c: AMBIENT_TEMP_C,
        }
    }

    /// Advance exposure by one manipulator tick (`dt_s` seconds) toward
    /// `target_c` (the held object's temperature, or ambient if nothing is
    /// held) via explicit-Euler integration of Newton's law of cooling.
    pub fn step(&mut self, target_c: f64, dt_s: f64) {
        let q = convection_heat_rate(GRIPPER_H, GRIPPER_AREA_M2, target_c - self.exposure_c);
        self.exposure_c += q * dt_s / (GRIPPER_MASS_KG * GRIPPER_SPECIFIC_HEAT);
    }
}

/// What the gripper's thermal exposure should be relaxing toward this tick.
pub fn thermal_target_c(held_object: Option<&KitchenObject>) -> f64 {
    held_object
        .map(|o| o.temperature_c)
        .unwrap_or(AMBIENT_TEMP_C)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn benign() -> KitchenObject {
        KitchenObject {
            temperature_c: 20.0,
            max_safe_grip_force_n: 50.0,
            is_sharp: false,
        }
    }

    #[test]
    fn holding_nothing_is_green() {
        assert_eq!(hazard_tier(None), MotorSafetyLevel::Green);
    }

    #[test]
    fn benign_object_is_green() {
        assert_eq!(hazard_tier(Some(&benign())), MotorSafetyLevel::Green);
    }

    #[test]
    fn sharp_alone_floors_at_orange() {
        let obj = KitchenObject {
            is_sharp: true,
            ..benign()
        };
        assert_eq!(hazard_tier(Some(&obj)), MotorSafetyLevel::Orange);
    }

    #[test]
    fn hot_alone_floors_at_orange() {
        let obj = KitchenObject {
            temperature_c: 90.0,
            ..benign()
        };
        assert_eq!(hazard_tier(Some(&obj)), MotorSafetyLevel::Orange);
    }

    #[test]
    fn sharp_and_hot_floors_at_red() {
        let obj = KitchenObject {
            temperature_c: 90.0,
            is_sharp: true,
            ..benign()
        };
        assert_eq!(hazard_tier(Some(&obj)), MotorSafetyLevel::Red);
    }

    #[test]
    fn scald_threshold_is_a_hard_boundary() {
        let just_under = KitchenObject {
            temperature_c: SCALD_RISK_TEMP_C - 0.01,
            ..benign()
        };
        let just_at = KitchenObject {
            temperature_c: SCALD_RISK_TEMP_C,
            ..benign()
        };
        assert_eq!(hazard_tier(Some(&just_under)), MotorSafetyLevel::Green);
        assert_eq!(hazard_tier(Some(&just_at)), MotorSafetyLevel::Orange);
    }

    /// The whole point of a hazard *floor*: composing with any Φ-derived tier
    /// via `.max()` never yields something less severe than the hazard alone.
    #[test]
    fn hazard_tier_is_a_floor_under_max_composition() {
        let obj = KitchenObject {
            is_sharp: true,
            ..benign()
        };
        let hazard = hazard_tier(Some(&obj));
        for phi_tier in [
            MotorSafetyLevel::Green,
            MotorSafetyLevel::Yellow,
            MotorSafetyLevel::Orange,
            MotorSafetyLevel::Red,
        ] {
            assert!(phi_tier.max(hazard) >= hazard);
        }
    }

    /// Force implied by a gripper command: (1 - gripper) * platform max.
    fn implied_force(gripper: f32) -> f64 {
        (1.0 - gripper as f64) * PLATFORM_MAX_GRIP_FORCE_N
    }

    #[test]
    fn full_squeeze_on_a_fragile_object_never_exceeds_its_crush_threshold() {
        let fragile = KitchenObject {
            max_safe_grip_force_n: 5.0,
            ..benign()
        };
        for tier in [
            MotorSafetyLevel::Green,
            MotorSafetyLevel::Yellow,
            MotorSafetyLevel::Orange,
            MotorSafetyLevel::Red,
        ] {
            // Command full squeeze (gripper = 0.0, fully closed).
            let clamped = clamp_grip_command(0.0, tier, Some(&fragile));
            let force = implied_force(clamped);
            assert!(
                force <= 5.0 + 1e-6,
                "tier={tier:?} clamped={clamped} implied_force={force}"
            );
        }
    }

    #[test]
    fn no_object_never_clamps_below_the_tiers_own_cap() {
        // With nothing held, the only constraint is the tier's own cap.
        let clamped = clamp_grip_command(0.0, MotorSafetyLevel::Yellow, None);
        assert!(implied_force(clamped) <= 20.0 + 1e-6);
        // Green with nothing held permits full authority (no clamping at all).
        let clamped_green = clamp_grip_command(0.0, MotorSafetyLevel::Green, None);
        assert_eq!(clamped_green, 0.0);
    }

    #[test]
    fn clamp_never_closes_more_than_commanded() {
        // A gentle, already-safe command must pass through unchanged, never
        // forced tighter than what was actually commanded.
        let fragile = KitchenObject {
            max_safe_grip_force_n: 5.0,
            ..benign()
        };
        let gentle = 0.99; // barely closed, ~0.87N implied
        let clamped = clamp_grip_command(gentle, MotorSafetyLevel::Green, Some(&fragile));
        assert_eq!(clamped, gentle);
    }

    #[test]
    fn exposure_accumulates_while_holding_a_hot_object() {
        let mut thermal = GripperThermalState::ambient();
        let hot = KitchenObject {
            temperature_c: 90.0,
            ..benign()
        };
        let start = thermal.exposure_c;
        for _ in 0..20 {
            thermal.step(thermal_target_c(Some(&hot)), 1.0);
        }
        assert!(
            thermal.exposure_c > start,
            "exposure should rise toward the held object's temperature"
        );
        assert!(
            thermal.exposure_c < hot.temperature_c,
            "should be approaching, not exceeding, the target"
        );
    }

    #[test]
    fn exposure_decays_back_to_ambient_after_release() {
        let mut thermal = GripperThermalState::ambient();
        let hot = KitchenObject {
            temperature_c: 90.0,
            ..benign()
        };
        for _ in 0..20 {
            thermal.step(thermal_target_c(Some(&hot)), 1.0);
        }
        let peak = thermal.exposure_c;
        assert!(peak > AMBIENT_TEMP_C);

        for _ in 0..40 {
            thermal.step(thermal_target_c(None), 1.0);
        }
        assert!(
            thermal.exposure_c < peak,
            "exposure should decay after release"
        );
        assert!(
            (thermal.exposure_c - AMBIENT_TEMP_C).abs() < (peak - AMBIENT_TEMP_C).abs(),
            "should be relaxing back toward ambient"
        );
    }

    #[test]
    fn thermal_target_is_ambient_when_holding_nothing() {
        assert_eq!(thermal_target_c(None), AMBIENT_TEMP_C);
    }

    #[test]
    fn thermal_target_is_the_held_objects_temperature() {
        let obj = KitchenObject {
            temperature_c: 75.0,
            ..benign()
        };
        assert_eq!(thermal_target_c(Some(&obj)), 75.0);
    }
}
