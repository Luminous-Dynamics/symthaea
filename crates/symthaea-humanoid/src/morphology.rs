// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structural topologies and biomechanical morphology configurations.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandSide {
    Right,
    Left,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidMorphology {
    Dmc21,
    Dexterous53,
    FullSpine,
    WithNeckWrist,
}

impl HumanoidMorphology {
    pub fn num_actuators(&self) -> usize {
        match self {
            HumanoidMorphology::Dmc21 => 21,
            HumanoidMorphology::WithNeckWrist => 21,
            HumanoidMorphology::Dexterous53 => 53,
            HumanoidMorphology::FullSpine => 64,
        }
    }

    pub fn joint_limits(&self) -> Vec<[f64; 2]> {
        // CORE ANATOMICAL SIGN ALIGNMENT: Knee upper extensions (indices 6 and 12)
        // are flipped to positive boundaries [0.15, 2.40] to force human-like backward bending.
        let mut limits = vec![
            [-1.31, 0.52],
            [-0.79, 0.79],
            [-0.61, 0.61], // Abdomen Base [0..3]
            [-0.44, 0.09],
            [-1.05, 0.61],
            [-1.92, 0.35],
            [0.15, 2.40],
            [-0.87, 0.87],
            [-0.87, 0.87], // Right Leg [3..9]
            [-0.44, 0.09],
            [-1.05, 0.61],
            [-1.92, 0.35],
            [0.15, 2.40],
            [-0.87, 0.87],
            [-0.87, 0.87], // Left Leg [9..15]
            [-1.48, 1.05],
            [-1.48, 1.05],
            [-1.57, 0.87], // Right Arm [15..18]
            [-1.05, 1.48],
            [-1.05, 1.48],
            [-1.57, 0.87], // Left Arm [18..21]
        ];

        if *self == HumanoidMorphology::Dexterous53 || *self == HumanoidMorphology::FullSpine {
            for _ in 0..32 {
                limits.push([-0.78, 0.78]);
            } // Hands [21..53]
        }
        if *self == HumanoidMorphology::FullSpine {
            for _ in 0..11 {
                limits.push([-0.35, 0.35]);
            } // Spine [53..64]
        }
        limits
    }

    pub fn joint_inertias(&self) -> Vec<f64> {
        let mut inertias = vec![
            0.30, 0.30, 0.30, 0.50, 0.50, 0.50, 0.20, 0.05, 0.05, 0.50, 0.50, 0.50, 0.20, 0.05,
            0.05, 0.10, 0.10, 0.06, 0.10, 0.10, 0.06,
        ];
        if *self == HumanoidMorphology::Dexterous53 || *self == HumanoidMorphology::FullSpine {
            inertias.extend(vec![0.005; 32]);
        }
        if *self == HumanoidMorphology::FullSpine {
            inertias.extend(vec![0.07; 11]);
        }
        inertias
    }

    pub fn joint_damping(&self) -> Vec<f64> {
        let mut damping = vec![
            6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 5.0, 3.0, 3.0, 8.0, 8.0, 8.0, 5.0, 3.0, 3.0, 2.0, 2.0,
            1.5, 2.0, 2.0, 1.5,
        ];
        if *self == HumanoidMorphology::Dexterous53 || *self == HumanoidMorphology::FullSpine {
            damping.extend(vec![0.6; 32]);
        }
        if *self == HumanoidMorphology::FullSpine {
            damping.extend(vec![4.5; 11]);
        }
        damping
    }

    pub fn joint_torque_scales(&self) -> Vec<f64> {
        let mut scales = vec![
            100.0, 100.0, 100.0, 100.0, 100.0, 300.0, 200.0, 100.0, 100.0, 100.0, 100.0, 300.0,
            200.0, 100.0, 100.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
        ];
        if *self == HumanoidMorphology::Dexterous53 || *self == HumanoidMorphology::FullSpine {
            scales.extend(vec![6.0; 32]);
        }
        if *self == HumanoidMorphology::FullSpine {
            scales.extend(vec![45.0; 11]);
        }
        scales
    }

    pub fn joint_names(&self) -> Vec<String> {
        (0..self.num_actuators())
            .map(|i| format!("j_{}", i))
            .collect()
    }

    pub fn pd_gains(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_actuators();
        let mut kp = vec![200.0; n];
        let mut kd = vec![12.0; n];
        if n >= 53 {
            for i in 21..53 {
                kp[i] = 45.0;
                kd[i] = 1.5;
            }
        }
        if n >= 64 {
            for i in 53..64 {
                kp[i] = 160.0;
                kd[i] = 8.0;
            }
        }
        (kp, kd)
    }

    pub fn to_mjcf(&self) -> String {
        let mut out = String::new();
        let lim = self.joint_limits();
        let actuators_count = self.num_actuators();

        out.push_str("\n");
        out.push_str("<mujoco model=\"flagship_humanoid\">\n");
        out.push_str(
            "  <compiler angle=\"radian\" inertiafromgeom=\"true\" coordinate=\"local\"/>\n",
        );
        out.push_str("  <option timestep=\"0.0025\" gravity=\"0 0 -9.81\"/>\n");
        out.push_str("  <default>\n");
        out.push_str("    <joint limited=\"true\" damping=\"35.0\" armature=\"0.08\" stiffness=\"120.0\"/>\n");
        out.push_str("    <geom contype=\"1\" conaffinity=\"1\" condim=\"3\" friction=\"1.6 0.2 0.1\" rgba=\"0.85 0.65 0.45 1\"/>\n");
        out.push_str(
            "    <position ctrllimited=\"true\" forcelimited=\"true\" forcerange=\"-400 400\"/>\n",
        );
        out.push_str("  </default>\n");
        out.push_str("  <worldbody>\n");
        out.push_str("    <light pos=\"0 0 6\" dir=\"0 0 -1\" diffuse=\"0.8 0.8 0.8\"/>\n");
        out.push_str("    <geom name=\"floor\" type=\"plane\" pos=\"0 0 0\" size=\"30 30 0.1\" rgba=\"0.15 0.15 0.15 1\"/>\n");

        out.push_str("    \n");
        out.push_str("    <geom name=\"step_1\" type=\"box\" pos=\"1.4 0 0.02\" size=\"0.22 1.2 0.02\" rgba=\"0.35 0.35 0.4 1\"/>\n");
        out.push_str("    <geom name=\"step_2\" type=\"box\" pos=\"1.8 0 0.04\" size=\"0.22 1.2 0.04\" rgba=\"0.40 0.40 0.45 1\"/>\n");
        out.push_str("    <geom name=\"step_3\" type=\"box\" pos=\"2.2 0 0.06\" size=\"0.22 1.2 0.06\" rgba=\"0.45 0.45 0.5 1\"/>\n");
        out.push_str("    <geom name=\"block_left\" type=\"box\" pos=\"3.6 0.22 0.04\" size=\"0.35 0.35 0.04\" rgba=\"0.5 0.35 0.35 1\"/>\n");
        out.push_str("    <geom name=\"block_right\" type=\"box\" pos=\"4.5 -0.22 0.06\" size=\"0.35 0.35 0.06\" rgba=\"0.35 0.5 0.35 1\"/>\n");
        out.push_str("    <geom name=\"ridge_track\" type=\"box\" pos=\"7.5 0 0.03\" size=\"2.0 0.6 0.03\" rgba=\"0.28 0.28 0.32 1\"/>\n");

        out.push_str("    <body name=\"pelvis\" pos=\"0 0 0.74\">\n");
        out.push_str("      <freejoint name=\"root\"/>\n");
        out.push_str(
            "      <geom name=\"pelvis_g\" type=\"sphere\" size=\"0.10\" mass=\"10.0\"/>\n",
        );
        out.push_str(
            "      <site name=\"vestibular_ear\" pos=\"0 0 0\" size=\"0.01\" rgba=\"1 0 0 1\"/>\n",
        );

        // BRANCH 1: Right Leg Tree
        out.push_str("      <body name=\"r_thigh\" pos=\"0 -0.11 -0.05\">\n");
        out.push_str(&format!(
            "        <joint name=\"j_3\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[3][0], lim[3][1]
        ));
        out.push_str(&format!(
            "        <joint name=\"j_4\" type=\"hinge\" axis=\"0 0 1\" range=\"{} {}\"/>\n",
            lim[4][0], lim[4][1]
        ));
        out.push_str(&format!(
            "        <joint name=\"j_5\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[5][0], lim[5][1]
        ));
        out.push_str("        <geom name=\"r_thigh_g\" type=\"capsule\" size=\"0.045\" fromto=\"0 0 0 0 0 -0.34\" mass=\"5.5\"/>\n");
        out.push_str("        <body name=\"r_shin\" pos=\"0 0 -0.34\">\n");
        out.push_str(&format!(
            "          <joint name=\"j_6\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[6][0], lim[6][1]
        ));
        out.push_str("          <geom name=\"r_shin_g\" type=\"capsule\" size=\"0.038\" fromto=\"0 0 0 0 0 -0.32\" mass=\"4.0\"/>\n");
        out.push_str("          <body name=\"r_foot\" pos=\"0 0 -0.32\">\n");
        out.push_str(&format!(
            "            <joint name=\"j_7\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[7][0], lim[7][1]
        ));
        out.push_str(&format!(
            "            <joint name=\"j_8\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[8][0], lim[8][1]
        ));
        out.push_str("            <geom name=\"r_foot_g\" type=\"box\" size=\"0.14 0.07 0.025\" pos=\"0.05 0 -0.02\" mass=\"1.8\"/>\n");
        out.push_str("            <site name=\"r_foot_site\" type=\"box\" size=\"0.142 0.072 0.026\" pos=\"0.05 0 -0.02\" rgba=\"1 0 0 0\"/>\n");
        out.push_str("          </body></body>\n");
        out.push_str("      </body>\n");

        // BRANCH 2: Left Leg Tree
        out.push_str("      <body name=\"l_thigh\" pos=\"0 0.11 -0.05\">\n");
        out.push_str(&format!(
            "        <joint name=\"j_9\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[9][0], lim[9][1]
        ));
        out.push_str(&format!(
            "        <joint name=\"j_10\" type=\"hinge\" axis=\"0 0 1\" range=\"{} {}\"/>\n",
            lim[10][0], lim[10][1]
        ));
        out.push_str(&format!(
            "        <joint name=\"j_11\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[11][0], lim[11][1]
        ));
        out.push_str("        <geom name=\"l_thigh_g\" type=\"capsule\" size=\"0.045\" fromto=\"0 0 0 0 0 -0.34\" mass=\"5.5\"/>\n");
        out.push_str("        <body name=\"l_shin\" pos=\"0 0 -0.34\">\n");
        out.push_str(&format!(
            "          <joint name=\"j_12\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[12][0], lim[12][1]
        ));
        out.push_str("          <geom name=\"l_shin_g\" type=\"capsule\" size=\"0.038\" fromto=\"0 0 0 0 0 -0.32\" mass=\"4.0\"/>\n");
        out.push_str("          <body name=\"l_foot\" pos=\"0 0 -0.32\">\n");
        out.push_str(&format!(
            "            <joint name=\"j_13\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[13][0], lim[13][1]
        ));
        out.push_str(&format!(
            "            <joint name=\"j_14\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[14][0], lim[14][1]
        ));
        out.push_str("            <geom name=\"l_foot_g\" type=\"box\" size=\"0.14 0.07 0.025\" pos=\"0.05 0 -0.02\" mass=\"1.8\"/>\n");
        out.push_str("            <site name=\"l_foot_site\" type=\"box\" size=\"0.142 0.072 0.026\" pos=\"0.05 0 -0.02\" rgba=\"1 0 0 0\"/>\n");
        out.push_str("          </body></body>\n");
        out.push_str("      </body>\n");

        // BRANCH 3: Upper Torso Core
        out.push_str("      <body name=\"lower_torso\" pos=\"0 0 0.06\">\n");
        out.push_str(&format!(
            "        <joint name=\"j_0\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[0][0], lim[0][1]
        ));
        out.push_str(&format!(
            "        <joint name=\"j_1\" type=\"hinge\" axis=\"0 0 1\" range=\"{} {}\"/>\n",
            lim[1][0], lim[1][1]
        ));
        out.push_str(&format!(
            "        <joint name=\"j_2\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[2][0], lim[2][1]
        ));
        out.push_str("        <geom name=\"lower_torso_g\" type=\"capsule\" size=\"0.065\" fromto=\"0 0 0 0 0 0.12\" mass=\"4.0\"/>\n");

        out.push_str("        <body name=\"upper_chest\" pos=\"0 0 0.12\">\n");
        out.push_str("          <geom name=\"chest_g\" type=\"capsule\" size=\"0.08\" fromto=\"0 -0.12 0.08 0 0.12 0.08\" mass=\"8.0\"/>\n");

        // Sub-Branch 3A: Right Arm Tree
        out.push_str("          <body name=\"r_upper_arm\" pos=\"0 -0.16 0.08\">\n");
        out.push_str(&format!(
            "            <joint name=\"j_15\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[15][0], lim[15][1]
        ));
        out.push_str(&format!(
            "            <joint name=\"j_16\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[16][0], lim[16][1]
        ));
        out.push_str("            <geom name=\"r_uarm_g\" type=\"capsule\" size=\"0.035\" fromto=\"0 0 0 0 0 -0.22\" mass=\"1.5\"/>\n");
        out.push_str("            <body name=\"r_forearm\" pos=\"0 0 -0.22\">\n");
        out.push_str(&format!(
            "              <joint name=\"j_17\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[17][0], lim[17][1]
        ));
        out.push_str("              <geom name=\"r_farm_g\" type=\"capsule\" size=\"0.028\" fromto=\"0 0 0 0 0 -0.18\" mass=\"1.0\"/>\n");
        out.push_str("              <body name=\"r_hand\" pos=\"0 0 -0.18\">\n");
        for k in 0..16 {
            out.push_str(&format!(
                "                <body name=\"r_finger_{}\" pos=\"{} {} -0.01\">\n",
                k,
                (k % 4) as f64 * 0.012 - 0.02,
                (k / 4) as f64 * 0.012 - 0.02
            ));
            out.push_str(&format!("                  <joint name=\"j_{}\" type=\"hinge\" axis=\"0 1 0\" range=\"-0.78 0.78\" stiffness=\"10.0\" damping=\"1.0\"/>\n", 21 + k));
            out.push_str(&format!("                  <geom name=\"r_fg_{}\" type=\"capsule\" size=\"0.005\" fromto=\"0 0 0 0 0 -0.03\" mass=\"0.05\"/>\n", k));
            out.push_str("                </body>\n");
        }
        out.push_str("              </body>\n            </body>\n          </body>\n");

        // Sub-Branch 3B: Left Arm Tree
        out.push_str("          <body name=\"l_upper_arm\" pos=\"0 0.16 0.08\">\n");
        out.push_str(&format!(
            "            <joint name=\"j_18\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[18][0], lim[18][1]
        ));
        out.push_str(&format!(
            "            <joint name=\"j_19\" type=\"hinge\" axis=\"1 0 0\" range=\"{} {}\"/>\n",
            lim[19][0], lim[19][1]
        ));
        out.push_str("            <geom name=\"l_uarm_g\" type=\"capsule\" size=\"0.035\" fromto=\"0 0 0 0 0 -0.22\" mass=\"1.5\"/>\n");
        out.push_str("            <body name=\"l_forearm\" pos=\"0 0 -0.22\">\n");
        out.push_str(&format!(
            "              <joint name=\"j_20\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\"/>\n",
            lim[20][0], lim[20][1]
        ));
        out.push_str("              <geom name=\"l_farm_g\" type=\"capsule\" size=\"0.028\" fromto=\"0 0 0 0 0 -0.18\" mass=\"1.0\"/>\n");
        out.push_str("              <body name=\"l_hand\" pos=\"0 0 -0.18\">\n");
        for k in 0..16 {
            out.push_str(&format!(
                "                <body name=\"l_finger_{}\" pos=\"{} {} -0.01\">\n",
                k,
                (k % 4) as f64 * 0.012 - 0.02,
                (k / 4) as f64 * 0.012 - 0.02
            ));
            out.push_str(&format!("                  <joint name=\"j_{}\" type=\"hinge\" axis=\"0 1 0\" range=\"-0.78 0.78\" stiffness=\"10.0\" damping=\"1.0\"/>\n", 37 + k));
            out.push_str(&format!("                  <geom name=\"l_fg_{}\" type=\"capsule\" size=\"0.005\" fromto=\"0 0 0 0 0 -0.03\" mass=\"0.05\"/>\n", k));
            out.push_str("                </body>\n");
        }
        out.push_str("              </body>\n            </body>\n          </body>\n");

        // Sub-Branch 3C: Compact Proportional Cervical Spine Chain
        out.push_str("          <body name=\"spine_base\" pos=\"0 0 0.14\">\n");
        out.push_str("            <body name=\"v_seg_0\" pos=\"0 0 0.012\">\n");
        out.push_str(&format!("              <joint name=\"j_53\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\" stiffness=\"60.0\"/>\n", lim[53][0], lim[53][1]));
        out.push_str("              <geom name=\"vg_0\" type=\"capsule\" size=\"0.022\" fromto=\"0 0 0 0 0 0.012\" mass=\"0.2\"/>\n");
        for k in 1..11 {
            out.push_str(&format!(
                "              <body name=\"v_seg_{}\" pos=\"0 0 0.012\">\n",
                k
            ));
            out.push_str(&format!("                <joint name=\"j_{}\" type=\"hinge\" axis=\"0 1 0\" range=\"{} {}\" stiffness=\"60.0\"/>\n", 53 + k, lim[53+k][0], lim[53+k][1]));
            out.push_str(&format!("                <geom name=\"vg_{}\" type=\"capsule\" size=\"0.022\" fromto=\"0 0 0 0 0 0.012\" mass=\"0.2\"/>\n", k));
        }

        // Mount sensory head cluster with camera layout
        out.push_str("                <body name=\"head\" pos=\"0 0 0.012\">\n");
        out.push_str("                  <geom name=\"head_g\" type=\"sphere\" size=\"0.075\" mass=\"2.5\" pos=\"0 0 0.05\" rgba=\"0.85 0.65 0.45 1\"/>\n");
        out.push_str("                  <geom name=\"visor_g\" type=\"box\" size=\"0.035 0.045 0.02\" pos=\"0.05 0 0.05\" rgba=\"0.15 0.15 0.15 1\"/>\n");
        out.push_str("                  <camera name=\"visor_camera\" pos=\"0.06 0 0.05\" euler=\"0 1.5708 0\" fovy=\"85\"/>\n");
        out.push_str("                </body>\n");

        for _ in 0..11 {
            out.push_str("              </body>\n");
        }
        out.push_str("          </body>\n");

        out.push_str("        </body>\n");
        out.push_str("      </body>\n");
        out.push_str("    </body>\n");
        out.push_str("  </worldbody>\n");

        out.push_str("  <contact>\n");
        out.push_str("    <exclude body1=\"upper_chest\" body2=\"r_upper_arm\"/>\n");
        out.push_str("    <exclude body1=\"upper_chest\" body2=\"l_upper_arm\"/>\n");
        out.push_str("    <exclude body1=\"upper_chest\" body2=\"r_forearm\"/>\n");
        out.push_str("    <exclude body1=\"upper_chest\" body2=\"l_forearm\"/>\n");
        out.push_str("  </contact>\n");

        out.push_str("  <sensor>\n");
        out.push_str("    <accelerometer name=\"vestibular_accel\" site=\"vestibular_ear\"/>\n");
        out.push_str("    <gyro name=\"vestibular_gyro\" site=\"vestibular_ear\"/>\n");
        out.push_str("    \n");
        out.push_str("    <touch name=\"r_foot_pressure\" site=\"r_foot_site\"/>\n");
        out.push_str("    <touch name=\"l_foot_pressure\" site=\"l_foot_site\"/>\n");
        out.push_str("  </sensor>\n");

        out.push_str("  <actuator>\n");
        for i in 0..actuators_count {
            let gain = if i < 21 { "kp=\"650\"" } else { "kp=\"150\"" };
            out.push_str(&format!(
                "    <position name=\"actuator_{}\" joint=\"j_{}\" {} ctrlrange=\"{} {}\"/>\n",
                i, i, gain, lim[i][0], lim[i][1]
            ));
        }
        out.push_str("  </actuator>\n");
        out.push_str("</mujoco>\n");
        out
    }
}

pub fn compute_hand_centroid(
    hand_base: [f64; 3],
    _joint_angles: &[f64],
    dir: [f64; 3],
) -> [f64; 3] {
    [
        hand_base[0] + dir[0] * 0.08,
        hand_base[1] + dir[1] * 0.08,
        hand_base[2] + dir[2] * 0.08,
    ]
}
