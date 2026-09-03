// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure localized structural plan for the `HardwareBud` presentation stage.
//!
//! Hardware change is a persistent topology fact, so its visual grammar should
//! be a bounded local addition rather than a global brightness change. This
//! module derives normalized geometry only. It performs no I/O, allocates no
//! per-frame storage, and has no authority over boot state or device discovery.

use std::f32::consts::{PI, TAU};

/// Normalized geometry for one localized hardware-growth event.
///
/// Radii and lengths are expressed as fractions of the renderer's minimum
/// dimension. The renderer remains responsible for choosing an actual topology
/// attachment point close to `anchor_angle` / `anchor_radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareBudPlan {
    /// Direction from the organism's visual center in radians, `[-PI, PI]`.
    pub anchor_angle: f32,
    /// Preferred attachment radius as a fraction of the minimum dimension.
    pub anchor_radius: f32,
    /// Stem length as a fraction of the minimum dimension.
    pub stem_length: f32,
    /// Child branch length relative to the stem.
    pub branch_length_scale: f32,
    /// Angular separation between child branches in radians.
    pub branch_spread: f32,
    /// Small bounded number of local child branches.
    pub branch_count: u8,
    /// Stage-local structural reveal, `[0, 1]`.
    pub growth: f32,
}

impl HardwareBudPlan {
    /// Derive stable localized geometry from an already-established visual seed.
    /// Only `growth` may vary with stage progress; structural identity remains
    /// fixed across frames of the same hardware event.
    pub fn derive(seed: &[u8; 32], stage_progress: f32) -> Self {
        let a = selector(seed, 3);
        let b = selector(seed, 11);
        let c = selector(seed, 17);
        let d = selector(seed, 23);
        let e = selector(seed, 29);

        Self {
            anchor_angle: a * TAU - PI,
            anchor_radius: 0.20 + b * 0.13,
            stem_length: 0.055 + c * 0.045,
            branch_length_scale: 0.42 + d * 0.28,
            branch_spread: 0.20 + e * 0.28,
            branch_count: 2 + seed[7] % 3,
            growth: smoothstep(finite_unit(stage_progress)),
        }
    }

    /// Direction of a child branch. Branches are arranged symmetrically around
    /// the outward stem, preserving one localized focal event.
    pub fn branch_angle(self, index: u8) -> f32 {
        let count = self.branch_count.max(1) as f32;
        let centered = index.min(self.branch_count.saturating_sub(1)) as f32
            - (count - 1.0) * 0.5;
        self.anchor_angle + centered * self.branch_spread
    }

    /// True once the structural bud is visually meaningful enough to draw.
    pub fn should_render(self) -> bool {
        self.growth >= 0.01
    }
}

fn selector(seed: &[u8; 32], offset: usize) -> f32 {
    // Mix several seed positions so a single byte cannot dominate placement.
    let a = seed[offset % 32] as u32;
    let b = seed[(offset + 9) % 32] as u32;
    let c = seed[(offset + 21) % 32] as u32;
    let mixed = (a * 73 + b * 151 + c * 199 + offset as u32 * 37) & 0xffff;
    mixed as f32 / 65_535.0
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn smoothstep(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structural_tuple(plan: HardwareBudPlan) -> (f32, f32, f32, f32, f32, u8) {
        (
            plan.anchor_angle,
            plan.anchor_radius,
            plan.stem_length,
            plan.branch_length_scale,
            plan.branch_spread,
            plan.branch_count,
        )
    }

    #[test]
    fn plan_is_deterministic_and_bounded() {
        for byte in [0x00, 0x19, 0x42, 0x7f, 0xff] {
            let seed = [byte; 32];
            let a = HardwareBudPlan::derive(&seed, 0.55);
            let b = HardwareBudPlan::derive(&seed, 0.55);
            assert_eq!(a, b);
            assert!((-PI..=PI).contains(&a.anchor_angle));
            assert!((0.20..=0.33).contains(&a.anchor_radius));
            assert!((0.055..=0.10).contains(&a.stem_length));
            assert!((0.42..=0.70).contains(&a.branch_length_scale));
            assert!((0.20..=0.48).contains(&a.branch_spread));
            assert!((2..=4).contains(&a.branch_count));
            assert!((0.0..=1.0).contains(&a.growth));
        }
    }

    #[test]
    fn structure_is_stable_while_growth_changes() {
        let seed = [0x53; 32];
        let early = HardwareBudPlan::derive(&seed, 0.20);
        let late = HardwareBudPlan::derive(&seed, 0.80);
        assert_eq!(structural_tuple(early), structural_tuple(late));
        assert!(early.growth < late.growth);
    }

    #[test]
    fn growth_is_monotonic_and_invalid_progress_fails_closed() {
        let seed = [0xa7; 32];
        let mut previous = 0.0;
        for progress in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let plan = HardwareBudPlan::derive(&seed, progress);
            assert!(plan.growth >= previous);
            previous = plan.growth;
        }
        assert_eq!(HardwareBudPlan::derive(&seed, f32::NAN).growth, 0.0);
        assert_eq!(HardwareBudPlan::derive(&seed, f32::INFINITY).growth, 0.0);
    }

    #[test]
    fn child_branches_remain_local_to_one_outward_event() {
        let plan = HardwareBudPlan::derive(&[0x31; 32], 1.0);
        let half_span = (plan.branch_count.saturating_sub(1) as f32 * plan.branch_spread) * 0.5;
        for index in 0..plan.branch_count {
            let delta = plan.branch_angle(index) - plan.anchor_angle;
            assert!(delta.abs() <= half_span + f32::EPSILON);
        }
    }

    #[test]
    fn near_zero_growth_can_skip_drawing() {
        let seed = [0x88; 32];
        assert!(!HardwareBudPlan::derive(&seed, 0.0).should_render());
        assert!(HardwareBudPlan::derive(&seed, 0.25).should_render());
    }

    #[test]
    fn different_seeds_can_produce_different_local_structure() {
        let a = HardwareBudPlan::derive(&[0x10; 32], 0.5);
        let b = HardwareBudPlan::derive(&[0x90; 32], 0.5);
        assert_ne!(structural_tuple(a), structural_tuple(b));
    }
}
