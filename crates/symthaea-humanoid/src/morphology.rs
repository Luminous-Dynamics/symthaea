// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parameterized humanoid morphology: from DMC-standard 21 DOF to full 67+ DOF.
//!
//! The morphology defines joint count, names, limits, body model, and encoder
//! channel layout. All existing code works unchanged with `Dmc21`.
//!
//! ## Joint Layout (Dexterous53)
//!
//! ```text
//! [0-2]   abdomen (3)          — torso stability
//! [3-8]   right_leg (6)        — hip×3, knee, ankle×2
//! [9-14]  left_leg (6)         — mirror
//! [15-17] right_arm (3)        — shoulder×2, elbow
//! [18-20] left_arm (3)         — mirror
//! [21-36] right_hand (16)      — thumb(4), index(4), middle(3), ring(2), pinky(3)
//! [37-52] left_hand (16)       — mirror
//! ```

use serde::{Deserialize, Serialize};

/// Humanoid morphology variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidMorphology {
    /// DMC standard: 21 actuators (current, backward-compatible).
    Dmc21,
    /// Dexterous: 21 base + 32 hand DOF = 53 actuators.
    Dexterous53,
    /// Full upper body: 53 + neck(3) + wrist(4) = 60 actuators.
    WithNeckWrist,
    /// Full spine: 60 + 4 extra spine segments = 64 actuators.
    FullSpine,
}

impl Default for HumanoidMorphology {
    fn default() -> Self {
        Self::Dmc21
    }
}

impl HumanoidMorphology {
    /// Total number of actuated joints.
    pub fn num_actuators(&self) -> usize {
        match self {
            Self::Dmc21 => 21,
            Self::Dexterous53 => 53,
            Self::WithNeckWrist => 60,
            Self::FullSpine => 64,
        }
    }

    /// Joint names for this morphology.
    pub fn joint_names(&self) -> Vec<&'static str> {
        let mut names = DMC21_JOINT_NAMES.to_vec();
        if self.num_actuators() > 21 {
            names.extend_from_slice(&HAND_JOINT_NAMES);
        }
        if self.num_actuators() > 53 {
            names.extend_from_slice(&NECK_WRIST_JOINT_NAMES);
        }
        if self.num_actuators() > 60 {
            names.extend_from_slice(&SPINE_JOINT_NAMES);
        }
        names
    }

    /// Joint limits [min, max] in radians for each actuator.
    pub fn joint_limits(&self) -> Vec<[f64; 2]> {
        let mut limits = DMC21_JOINT_LIMITS.to_vec();
        if self.num_actuators() > 21 {
            limits.extend_from_slice(&HAND_JOINT_LIMITS);
        }
        if self.num_actuators() > 53 {
            limits.extend_from_slice(&NECK_WRIST_JOINT_LIMITS);
        }
        if self.num_actuators() > 60 {
            limits.extend_from_slice(&SPINE_JOINT_LIMITS);
        }
        limits
    }

    /// Per-joint inertia (kg·m²).
    pub fn joint_inertias(&self) -> Vec<f64> {
        let mut inertias = DMC21_INERTIAS.to_vec();
        if self.num_actuators() > 21 {
            inertias.extend_from_slice(&HAND_INERTIAS);
        }
        if self.num_actuators() > 53 {
            inertias.extend_from_slice(&NECK_WRIST_INERTIAS);
        }
        if self.num_actuators() > 60 {
            inertias.extend_from_slice(&SPINE_INERTIAS);
        }
        inertias
    }

    /// Per-joint damping (Nm·s/rad).
    pub fn joint_damping(&self) -> Vec<f64> {
        let mut damping = DMC21_DAMPING.to_vec();
        if self.num_actuators() > 21 {
            damping.extend_from_slice(&HAND_DAMPING);
        }
        if self.num_actuators() > 53 {
            damping.extend_from_slice(&NECK_WRIST_DAMPING);
        }
        if self.num_actuators() > 60 {
            damping.extend_from_slice(&SPINE_DAMPING);
        }
        damping
    }

    /// Per-joint torque scale (Nm per normalized command unit).
    pub fn joint_torque_scales(&self) -> Vec<f64> {
        let mut scales = DMC21_TORQUE_SCALES.to_vec();
        if self.num_actuators() > 21 {
            scales.extend_from_slice(&HAND_TORQUE_SCALES);
        }
        if self.num_actuators() > 53 {
            scales.extend_from_slice(&NECK_WRIST_TORQUE_SCALES);
        }
        if self.num_actuators() > 60 {
            scales.extend_from_slice(&SPINE_TORQUE_SCALES);
        }
        scales
    }

    /// Per-joint PD gains (kp, kd).
    pub fn pd_gains(&self) -> (Vec<f64>, Vec<f64>) {
        let mut kp = DMC21_KP.to_vec();
        let mut kd = DMC21_KD.to_vec();
        if self.num_actuators() > 21 {
            kp.extend_from_slice(&HAND_KP);
            kd.extend_from_slice(&HAND_KD);
        }
        if self.num_actuators() > 53 {
            kp.extend_from_slice(&NECK_WRIST_KP);
            kd.extend_from_slice(&NECK_WRIST_KD);
        }
        if self.num_actuators() > 60 {
            kp.extend_from_slice(&SPINE_KP);
            kd.extend_from_slice(&SPINE_KD);
        }
        (kp, kd)
    }

    /// Number of HDC encoder channels for this morphology.
    pub fn num_channels(&self) -> usize {
        // Base: 72 (DMC21)
        // Each extra DOF adds 2 channels (angle + velocity)
        let extra_dof = self.num_actuators() - 21;
        let extra_features = match self {
            Self::Dmc21 => 0,
            Self::Dexterous53 => 6,      // 2 hand centroid positions
            Self::WithNeckWrist => 9,     // + 3 gaze direction
            Self::FullSpine => 9,         // same computed features
        };
        72 + extra_dof * 2 + extra_features
    }
}

// ─── DMC21 Constants (existing 21-DOF parameters) ───

pub const DMC21_JOINT_NAMES: [&str; 21] = [
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "right_ankle_x", "right_ankle_y",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "left_ankle_x", "left_ankle_y",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
];

pub const DMC21_JOINT_LIMITS: [[f64; 2]; 21] = [
    [-1.31, 0.52], [-0.79, 0.79], [-0.61, 0.61], // abdomen
    [-0.44, 0.09], [-1.05, 0.61], [-1.92, 0.35], [-2.79, 0.03], // right leg
    [-0.87, 0.87], [-0.87, 0.87],
    [-0.44, 0.09], [-1.05, 0.61], [-1.92, 0.35], [-2.79, 0.03], // left leg
    [-0.87, 0.87], [-0.87, 0.87],
    [-1.48, 1.05], [-1.48, 1.05], [-1.57, 0.87], // right arm
    [-1.05, 1.48], [-1.05, 1.48], [-1.57, 0.87], // left arm
];

const DMC21_INERTIAS: [f64; 21] = [
    0.50, 0.50, 0.50, // abdomen
    0.30, 0.30, 0.40, 0.25, 0.10, 0.10, // right leg
    0.30, 0.30, 0.40, 0.25, 0.10, 0.10, // left leg
    0.10, 0.10, 0.06, // right arm
    0.10, 0.10, 0.06, // left arm
];

const DMC21_DAMPING: [f64; 21] = [
    6.0, 6.0, 6.0, // abdomen
    5.0, 5.0, 8.0, 6.0, 3.0, 3.0, // right leg
    5.0, 5.0, 8.0, 6.0, 3.0, 3.0, // left leg
    2.0, 2.0, 1.5, // right arm
    2.0, 2.0, 1.5, // left arm
];

const DMC21_TORQUE_SCALES: [f64; 21] = [
    100.0, 100.0, 100.0, // abdomen
    100.0, 100.0, 300.0, 200.0, 100.0, 100.0, // right leg
    100.0, 100.0, 300.0, 200.0, 100.0, 100.0, // left leg
    25.0, 25.0, 25.0, // right arm
    25.0, 25.0, 25.0, // left arm
];

const DMC21_KP: [f64; 21] = [
    100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 120.0, 80.0, 80.0,
    100.0, 100.0, 100.0, 120.0, 80.0, 80.0,
    40.0, 40.0, 40.0,
    40.0, 40.0, 40.0,
];

const DMC21_KD: [f64; 21] = [
    10.0, 10.0, 10.0,
    10.0, 10.0, 10.0, 12.0, 8.0, 8.0,
    10.0, 10.0, 10.0, 12.0, 8.0, 8.0,
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0,
];

// ─── Hand Joint Constants (32 DOF: 16 per hand) ───

/// Right hand (16) + Left hand (16) = 32 joint names.
const HAND_JOINT_NAMES: [&str; 32] = [
    // Right hand (16 DOF)
    "r_thumb_cmc_abd", "r_thumb_cmc_flex", "r_thumb_mcp", "r_thumb_ip",
    "r_index_mcp_abd", "r_index_mcp_flex", "r_index_pip", "r_index_dip",
    "r_middle_mcp_abd", "r_middle_mcp_flex", "r_middle_pip",
    "r_ring_mcp_flex", "r_ring_pip",
    "r_pinky_mcp_flex", "r_pinky_pip", "r_pinky_abd",
    // Left hand (16 DOF, mirror)
    "l_thumb_cmc_abd", "l_thumb_cmc_flex", "l_thumb_mcp", "l_thumb_ip",
    "l_index_mcp_abd", "l_index_mcp_flex", "l_index_pip", "l_index_dip",
    "l_middle_mcp_abd", "l_middle_mcp_flex", "l_middle_pip",
    "l_ring_mcp_flex", "l_ring_pip",
    "l_pinky_mcp_flex", "l_pinky_pip", "l_pinky_abd",
];

/// Hand joint limits (radians). Anatomically based.
const HAND_JOINT_LIMITS: [[f64; 2]; 32] = [
    // Right thumb
    [-0.26, 0.87],   // CMC abd: -15° to 50°
    [-0.17, 1.22],   // CMC flex: -10° to 70°
    [0.0, 1.57],     // MCP: 0° to 90°
    [0.0, 1.22],     // IP: 0° to 70°
    // Right index
    [-0.35, 0.35],   // MCP abd: ±20°
    [0.0, 1.57],     // MCP flex: 0° to 90°
    [0.0, 1.75],     // PIP: 0° to 100°
    [0.0, 1.22],     // DIP: 0° to 70° (coupled: 0.67 × PIP)
    // Right middle
    [-0.26, 0.26],   // MCP abd: ±15°
    [0.0, 1.57],     // MCP flex: 0° to 90°
    [0.0, 1.75],     // PIP: 0° to 100°
    // Right ring
    [0.0, 1.57],     // MCP flex: 0° to 90°
    [0.0, 1.75],     // PIP: 0° to 100°
    // Right pinky
    [0.0, 1.57],     // MCP flex: 0° to 90°
    [0.0, 1.75],     // PIP: 0° to 100°
    [-0.35, 0.35],   // abd: ±20°
    // Left hand (mirror)
    [-0.26, 0.87], [-0.17, 1.22], [0.0, 1.57], [0.0, 1.22],
    [-0.35, 0.35], [0.0, 1.57], [0.0, 1.75], [0.0, 1.22],
    [-0.26, 0.26], [0.0, 1.57], [0.0, 1.75],
    [0.0, 1.57], [0.0, 1.75],
    [0.0, 1.57], [0.0, 1.75], [-0.35, 0.35],
];

/// Hand joint inertias — very low (finger segments are light).
const HAND_INERTIAS: [f64; 32] = [0.001; 32];

/// Hand joint damping — light viscous friction.
const HAND_DAMPING: [f64; 32] = [0.1; 32];

/// Hand torque scales — small actuators (~2 Nm).
const HAND_TORQUE_SCALES: [f64; 32] = [2.0; 32];

/// Hand PD gains — moderate for grasp control.
const HAND_KP: [f64; 32] = [20.0; 32];
const HAND_KD: [f64; 32] = [2.0; 32];

// ─── Neck + Wrist Constants (7 DOF) ───

const NECK_WRIST_JOINT_NAMES: [&str; 7] = [
    "neck_pan", "neck_tilt", "neck_roll",
    "r_wrist_flex", "r_wrist_dev",
    "l_wrist_flex", "l_wrist_dev",
];

const NECK_WRIST_JOINT_LIMITS: [[f64; 2]; 7] = [
    [-1.40, 1.40],   // neck pan: ±80°
    [-1.05, 1.05],   // neck tilt: ±60°
    [-0.70, 0.70],   // neck roll: ±40°
    [-1.40, 1.40],   // r_wrist flex: ±80°
    [-0.35, 0.35],   // r_wrist dev: ±20°
    [-1.40, 1.40],   // l_wrist flex
    [-0.35, 0.35],   // l_wrist dev
];

const NECK_WRIST_INERTIAS: [f64; 7] = [0.02, 0.02, 0.02, 0.005, 0.005, 0.005, 0.005];
const NECK_WRIST_DAMPING: [f64; 7] = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5];
const NECK_WRIST_TORQUE_SCALES: [f64; 7] = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0];
const NECK_WRIST_KP: [f64; 7] = [50.0, 50.0, 50.0, 30.0, 30.0, 30.0, 30.0];
const NECK_WRIST_KD: [f64; 7] = [5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0];

// ─── Spine Constants (4 DOF) ───

const SPINE_JOINT_NAMES: [&str; 4] = [
    "lumbar_flex", "lumbar_lateral",
    "thoracic_flex", "thoracic_lateral",
];

const SPINE_JOINT_LIMITS: [[f64; 2]; 4] = [
    [-0.79, 0.79],   // lumbar flex: ±45°
    [-0.52, 0.52],   // lumbar lateral: ±30°
    [-0.52, 0.52],   // thoracic flex: ±30°
    [-0.35, 0.35],   // thoracic lateral: ±20°
];

const SPINE_INERTIAS: [f64; 4] = [0.10, 0.10, 0.08, 0.08];
const SPINE_DAMPING: [f64; 4] = [2.0, 2.0, 1.5, 1.5];
const SPINE_TORQUE_SCALES: [f64; 4] = [50.0, 50.0, 40.0, 40.0];
const SPINE_KP: [f64; 4] = [80.0, 80.0, 60.0, 60.0];
const SPINE_KD: [f64; 4] = [8.0, 8.0, 6.0, 6.0];

// ─── DIP Coupling ───

/// DIP-PIP coupling ratio: DIP angle = ratio × PIP angle.
/// Based on anatomical coupling in human fingers.
pub const DIP_PIP_COUPLING_RATIO: f64 = 0.67;

/// Indices of DIP joints that are coupled to their preceding PIP.
/// In the Dexterous53 layout, DIP joints are at index 7 (r_index_dip)
/// and 23 (l_index_dip). Their PIP predecessors are at index 6 and 22.
pub const DIP_COUPLED_PAIRS: [(usize, usize); 2] = [
    (7, 6),   // r_index_dip coupled to r_index_pip
    (23, 22), // l_index_dip coupled to l_index_pip
];

// ─── Hand FK Geometry ───

/// Which hand to use for reaching/grasping tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandSide {
    Right,
    Left,
}

impl Default for HandSide {
    fn default() -> Self { Self::Right }
}

/// Finger segment lengths in meters: [proximal, middle, distal].
/// Based on anthropometric data for adult hands.
pub const FINGER_SEGMENTS: [[f64; 3]; 5] = [
    [0.040, 0.030, 0.025], // thumb: CMC→MCP, MCP→IP, IP→tip
    [0.045, 0.030, 0.020], // index: MCP→PIP, PIP→DIP, DIP→tip
    [0.048, 0.032, 0.022], // middle
    [0.043, 0.028, 0.020], // ring
    [0.035, 0.022, 0.018], // pinky
];

/// Number of DOF per finger in the Dexterous53 layout.
pub const FINGER_DOF: [usize; 5] = [4, 4, 3, 2, 3]; // thumb, index, middle, ring, pinky

/// Compute hand centroid from 16 hand joint angles.
///
/// Runs simplified planar FK for each finger, computes fingertip position,
/// returns the mean (centroid) of all 5 fingertips.
///
/// `hand_base`: 3D position of the hand base (from arm FK).
/// `hand_joints`: 16 joint angles for one hand (right or left).
/// `forward_dir`: unit vector pointing forward from the palm (typically [1, 0, 0] for right hand).
pub fn compute_hand_centroid(
    hand_base: [f64; 3],
    hand_joints: &[f64],
    forward_dir: [f64; 3],
) -> [f64; 3] {
    assert!(hand_joints.len() >= 16, "Need 16 hand joint angles, got {}", hand_joints.len());

    let mut centroid = [0.0f64; 3];
    let mut joint_offset = 0;

    for finger in 0..5 {
        let n_dof = FINGER_DOF[finger];
        let segs = FINGER_SEGMENTS[finger];
        let joints = &hand_joints[joint_offset..joint_offset + n_dof];

        // Compute fingertip using planar FK chain
        let tip = finger_fk(hand_base, joints, &segs, forward_dir, finger);

        centroid[0] += tip[0];
        centroid[1] += tip[1];
        centroid[2] += tip[2];

        joint_offset += n_dof;
    }

    // Mean of 5 fingertips
    centroid[0] /= 5.0;
    centroid[1] /= 5.0;
    centroid[2] /= 5.0;
    centroid
}

/// Simplified planar FK for one finger.
///
/// Each joint flexes in the sagittal plane of the finger. The finger extends
/// forward from hand_base, curling downward with flexion.
fn finger_fk(
    base: [f64; 3],
    joints: &[f64],
    segments: &[f64; 3],
    forward: [f64; 3],
    finger_idx: usize,
) -> [f64; 3] {
    // Lateral spread: fingers fan out from thumb (-30°) to pinky (+20°)
    let spread_angle = match finger_idx {
        0 => -0.5,  // thumb: offset laterally
        1 => -0.15, // index
        2 => 0.0,   // middle (center)
        3 => 0.12,  // ring
        _ => 0.22,  // pinky
    };

    // Compute cumulative flexion angle
    // For thumb (4 DOF): CMC_abd + CMC_flex + MCP + IP
    // For index (4 DOF): MCP_abd + MCP_flex + PIP + DIP
    // For middle (3 DOF): MCP_abd + MCP_flex + PIP (DIP coupled)
    // For ring (2 DOF): MCP_flex + PIP (DIP coupled)
    // For pinky (3 DOF): MCP_flex + PIP + abd (DIP coupled)

    // Extract abduction (first joint for thumb/index/middle, last for pinky)
    let abd = match finger_idx {
        0 => joints[0],            // thumb CMC_abd
        1 => joints[0],            // index MCP_abd
        2 => joints[0],            // middle MCP_abd
        4 => joints[2],            // pinky abd (last DOF)
        _ => 0.0,                  // ring: no abd
    };

    // Extract flexion angles for the 3 segments
    let (flex1, flex2, flex3) = match finger_idx {
        0 => (joints[1], joints[2], joints[3]),           // thumb: CMC_flex, MCP, IP
        1 => (joints[1], joints[2], joints[3]),           // index: MCP_flex, PIP, DIP
        2 => (joints[1], joints[2], joints[2] * DIP_PIP_COUPLING_RATIO), // middle: coupled DIP
        3 => (joints[0], joints[1], joints[1] * DIP_PIP_COUPLING_RATIO), // ring: coupled DIP
        _ => (joints[0], joints[1], joints[1] * DIP_PIP_COUPLING_RATIO), // pinky: coupled DIP
    };

    // FK: chain 3 segments, each rotated by cumulative flexion
    // Forward direction modified by spread + abduction
    let total_abd = spread_angle + abd * 0.3; // Scale abd contribution
    let up = [0.0, 0.0, 1.0]; // gravity axis

    // Segment 1: proximal phalanx
    let angle1 = flex1;
    let dx1 = segments[0] * (forward[0] * angle1.cos() - up[0] * angle1.sin());
    let dy1 = segments[0] * (forward[1] * angle1.cos() + total_abd.sin() * angle1.cos());
    let dz1 = segments[0] * (-angle1.sin());

    // Segment 2: middle phalanx
    let angle2 = angle1 + flex2;
    let dx2 = segments[1] * (forward[0] * angle2.cos());
    let dy2 = segments[1] * (forward[1] * angle2.cos() + total_abd.sin() * angle2.cos() * 0.5);
    let dz2 = segments[1] * (-angle2.sin());

    // Segment 3: distal phalanx
    let angle3 = angle2 + flex3;
    let dx3 = segments[2] * (forward[0] * angle3.cos());
    let dy3 = segments[2] * (forward[1] * angle3.cos());
    let dz3 = segments[2] * (-angle3.sin());

    [
        base[0] + dx1 + dx2 + dx3,
        base[1] + dy1 + dy2 + dy3,
        base[2] + dz1 + dz2 + dz3,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dmc21_has_21_actuators() {
        let m = HumanoidMorphology::Dmc21;
        assert_eq!(m.num_actuators(), 21);
        assert_eq!(m.joint_names().len(), 21);
        assert_eq!(m.joint_limits().len(), 21);
        assert_eq!(m.joint_inertias().len(), 21);
        assert_eq!(m.joint_damping().len(), 21);
        assert_eq!(m.joint_torque_scales().len(), 21);
    }

    #[test]
    fn dexterous53_has_53_actuators() {
        let m = HumanoidMorphology::Dexterous53;
        assert_eq!(m.num_actuators(), 53);
        assert_eq!(m.joint_names().len(), 53);
        assert_eq!(m.joint_limits().len(), 53);
        assert_eq!(m.joint_inertias().len(), 53);
    }

    #[test]
    fn with_neck_wrist_has_60_actuators() {
        let m = HumanoidMorphology::WithNeckWrist;
        assert_eq!(m.num_actuators(), 60);
        assert_eq!(m.joint_names().len(), 60);
        assert_eq!(m.joint_limits().len(), 60);
    }

    #[test]
    fn full_spine_has_64_actuators() {
        let m = HumanoidMorphology::FullSpine;
        assert_eq!(m.num_actuators(), 64);
        assert_eq!(m.joint_names().len(), 64);
        assert_eq!(m.joint_limits().len(), 64);
    }

    #[test]
    fn dmc21_channels_72() {
        assert_eq!(HumanoidMorphology::Dmc21.num_channels(), 72);
    }

    #[test]
    fn dexterous53_channels_increase() {
        let dmc = HumanoidMorphology::Dmc21.num_channels();
        let dex = HumanoidMorphology::Dexterous53.num_channels();
        assert!(dex > dmc, "Dexterous should have more channels: dmc={dmc}, dex={dex}");
        // 72 + 32*2 + 6 = 142
        assert_eq!(dex, 142);
    }

    #[test]
    fn pd_gains_match_actuator_count() {
        for morph in [
            HumanoidMorphology::Dmc21,
            HumanoidMorphology::Dexterous53,
            HumanoidMorphology::WithNeckWrist,
            HumanoidMorphology::FullSpine,
        ] {
            let (kp, kd) = morph.pd_gains();
            assert_eq!(kp.len(), morph.num_actuators(), "{morph:?} kp mismatch");
            assert_eq!(kd.len(), morph.num_actuators(), "{morph:?} kd mismatch");
        }
    }

    #[test]
    fn all_joint_limits_are_valid() {
        for morph in [
            HumanoidMorphology::Dmc21,
            HumanoidMorphology::Dexterous53,
            HumanoidMorphology::WithNeckWrist,
            HumanoidMorphology::FullSpine,
        ] {
            for (i, [min, max]) in morph.joint_limits().iter().enumerate() {
                assert!(
                    min < max,
                    "{morph:?} joint {i}: min ({min}) >= max ({max})"
                );
            }
        }
    }

    #[test]
    fn hand_joint_names_are_prefixed() {
        let names = HumanoidMorphology::Dexterous53.joint_names();
        assert!(names[21].starts_with("r_thumb"), "First hand joint: {}", names[21]);
        assert!(names[37].starts_with("l_thumb"), "First left hand joint: {}", names[37]);
    }

    #[test]
    fn dip_coupling_indices_valid() {
        let n = HumanoidMorphology::Dexterous53.num_actuators();
        for (dip, pip) in &DIP_COUPLED_PAIRS {
            assert!(*dip < n, "DIP index {dip} out of range");
            assert!(*pip < n, "PIP index {pip} out of range");
            assert!(dip > pip, "DIP should follow PIP");
        }
    }

    #[test]
    fn default_is_dmc21() {
        assert_eq!(HumanoidMorphology::default(), HumanoidMorphology::Dmc21);
    }

    #[test]
    fn hand_fk_open_fingers_extend_forward() {
        let base = [0.3, -0.17, 0.8];
        let joints = [0.0f64; 16]; // All joints at zero = fingers extended
        let forward = [1.0, 0.0, 0.0];
        let centroid = compute_hand_centroid(base, &joints, forward);
        // Centroid should be forward of base (extended fingers)
        assert!(centroid[0] > base[0], "Centroid x={:.3} should be ahead of base x={:.3}", centroid[0], base[0]);
    }

    #[test]
    fn hand_fk_closed_fist_curls_down() {
        let base = [0.3, -0.17, 0.8];
        // All flexion joints at ~1.2 rad (~70°)
        let joints = [0.0, 1.2, 1.2, 1.0,  // thumb
                      0.0, 1.2, 1.2, 0.8,  // index
                      0.0, 1.2, 1.2,        // middle
                      1.2, 1.2,              // ring
                      1.2, 1.2, 0.0];       // pinky
        let forward = [1.0, 0.0, 0.0];
        let centroid = compute_hand_centroid(base, &joints, forward);
        // Closed fist: fingertips should be below the base (curled down)
        assert!(centroid[2] < base[2], "Closed fist z={:.3} should be below base z={:.3}", centroid[2], base[2]);
    }

    #[test]
    fn hand_fk_centroid_is_finite() {
        let base = [0.3, -0.17, 0.8];
        // Various joint angles
        let joints = [0.1, 0.5, 0.3, 0.2,
                      -0.1, 0.8, 0.6, 0.4,
                      0.0, 0.7, 0.5,
                      0.6, 0.4,
                      0.5, 0.3, 0.1];
        let forward = [1.0, 0.0, 0.0];
        let centroid = compute_hand_centroid(base, &joints, forward);
        assert!(centroid[0].is_finite());
        assert!(centroid[1].is_finite());
        assert!(centroid[2].is_finite());
    }

    #[test]
    fn finger_dof_sum_is_16() {
        let total: usize = FINGER_DOF.iter().sum();
        assert_eq!(total, 16, "Total finger DOF should be 16 per hand");
    }

    #[test]
    fn hand_side_default_is_right() {
        assert_eq!(HandSide::default(), HandSide::Right);
    }
}
