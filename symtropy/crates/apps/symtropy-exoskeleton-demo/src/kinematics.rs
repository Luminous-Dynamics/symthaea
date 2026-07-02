// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sagittal-plane forward kinematics for visualizing the 6-DoF leg chain.
//!
//! Joint indices (from symthaea-exoskeleton):
//!   0,1,2 = right hip, knee, ankle (pitch/flexion only)
//!   3,4,5 = left  hip, knee, ankle
//!
//! Segment lengths (from `SimpleExoskeletonSimulator::new`):
//!   thigh = 0.45 m, shin = 0.43 m, foot = 0.08 m

use bevy::prelude::Vec3;

pub const THIGH_LEN: f32 = 0.45;
pub const SHIN_LEN: f32 = 0.43;
pub const FOOT_LEN: f32 = 0.08;
pub const HIP_WIDTH: f32 = 0.30;

/// 3 segment-world-frame transforms (positions + rotations) for one leg,
/// given the hip origin, side sign (`+1` = right / +Y, `-1` = left / -Y),
/// and the three joint angles (hip_flex, knee_flex, ankle_flex) in rad.
pub struct LegTransforms {
    /// Thigh: rotation + midpoint position.
    pub thigh: (Vec3, f32),
    /// Shin: rotation + midpoint position.
    pub shin: (Vec3, f32),
    /// Foot: rotation + midpoint position.
    pub foot: (Vec3, f32),
}

pub fn leg_chain(
    hip_origin: Vec3,
    _side: f32,
    hip_flex: f32,
    knee_flex: f32,
    ankle_flex: f32,
) -> LegTransforms {
    // All motion is in the sagittal (x-z) plane. Positive flex rotates the
    // segment forward (+x) and up from straight-down.
    let thigh_angle = hip_flex;
    let knee_world_angle = hip_flex + knee_flex;
    let ankle_world_angle = hip_flex + knee_flex + ankle_flex;

    // Thigh goes from hip → knee. Its "down" vector in world is
    // (-sin(angle), 0, -cos(angle)) — rotating about Y axis.
    let thigh_dir = Vec3::new(-thigh_angle.sin(), 0.0, -thigh_angle.cos());
    let knee_pos = hip_origin + thigh_dir * THIGH_LEN;
    let thigh_mid = hip_origin + thigh_dir * (THIGH_LEN * 0.5);

    let shin_dir = Vec3::new(-knee_world_angle.sin(), 0.0, -knee_world_angle.cos());
    let ankle_pos = knee_pos + shin_dir * SHIN_LEN;
    let shin_mid = knee_pos + shin_dir * (SHIN_LEN * 0.5);

    // Foot points forward (+x) not down; rotated by ankle world angle.
    let foot_dir = Vec3::new(ankle_world_angle.cos(), 0.0, -ankle_world_angle.sin());
    let foot_mid = ankle_pos + foot_dir * (FOOT_LEN * 0.5);

    LegTransforms {
        thigh: (thigh_mid, thigh_angle),
        shin: (shin_mid, knee_world_angle),
        foot: (foot_mid, ankle_world_angle - std::f32::consts::FRAC_PI_2),
    }
}
