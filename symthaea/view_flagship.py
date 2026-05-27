#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np

from view_modules.config import NOMINAL_TARGETS, get_velocity_limits
from view_modules.state import PosturalObserver
from view_modules.gait import BallisticGaitGenerator

def main():
    print("🧠 RUNNING CLOSED-LOOP MULTI-SENSORY STAIR CLIMBER ENGINE...")
    xml_path = "assets/deploy/flagship_fullspine.xml"

    if not os.path.exists(xml_path):
        print(f"❌ Error: Compiled asset missing at {xml_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gravity_compensation = np.sum(model.body_mass) * 9.81
    velocity_limits = get_velocity_limits(model.nu)

    observer = PosturalObserver()
    gait_gen = BallisticGaitGenerator()

    actuator_to_dof = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    last_applied_ctrl = NOMINAL_TARGETS.copy()

    last_time = time.time()
    start_time = time.time()
    last_log_time = time.time()

    print("🚀 Launching closed-loop sensory player context...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.4
        viewer.cam.lookat = [1.2, 0, 0.65]
        viewer.cam.elevation = -12
        viewer.cam.azimuth = 140

        while viewer.is_running():
            step_start = time.time()
            dt = time.time() - last_time
            last_time = time.time()
            elapsed = time.time() - start_time

            raw_accel = data.sensor('vestibular_accel').data.copy()
            local_pitch_inference = np.arctan2(raw_accel[0], np.sqrt(raw_accel[1]**2 + raw_accel[2]**2))
            upright_projection = np.cos(local_pitch_inference)

            # Read live somatosensory feedback values from the skin membranes
            r_sole_pressure = float(data.sensor('r_foot_pressure').data[0])
            l_sole_pressure = float(data.sensor('l_foot_pressure').data[0])

            rec_state = observer.evaluate_fall(upright_projection, elapsed)

            if rec_state == "NORMAL":
                # Pipe touch indicators into the gait engine to prevent backward-drift stalls
                ctrl_targets, state_label, r_swing, l_swing = gait_gen.compute_stride(
                    elapsed, dt, NOMINAL_TARGETS, r_sole_pressure, l_sole_pressure
                )
            else:
                ctrl_targets, state_label = observer.compute_recovery_targets(data.xmat[pelvis_id].reshape(3, 3), data, pelvis_id, gravity_compensation, NOMINAL_TARGETS)
                r_swing, l_swing = False, False

            if rec_state == "NORMAL":
                gimbal_increment = -local_pitch_inference / 11.0
                for s in range(53, 64):
                    ctrl_targets[s] += gimbal_increment

            sim_dt = model.opt.timestep
            for _ in range(10):
                for i in range(model.nu):
                    max_step_delta = velocity_limits[i] * sim_dt
                    clamped_target = np.clip(ctrl_targets[i], last_applied_ctrl[i] - max_step_delta, last_applied_ctrl[i] + max_step_delta)
                    
                    if rec_state == "NORMAL" and state_label != "STAND":
                        if i in [3, 4, 5, 6, 7, 8]:
                            damping_weight = 0.07 if not r_swing else 0.012
                        elif i in [9, 10, 11, 12, 13, 14]:
                            damping_weight = 0.07 if not l_swing else 0.012
                        else:
                            damping_weight = 0.04
                    else:
                        damping_weight = 0.05

                    data.ctrl[i] = clamped_target - damping_weight * data.qvel[actuator_to_dof[i]]
                    last_applied_ctrl[i] = clamped_target

                if rec_state == "NORMAL":
                    data.ctrl[7]  -= local_pitch_inference * 0.45
                    data.ctrl[13] -= local_pitch_inference * 0.45

                tether_scale = 1.0 if rec_state == "NORMAL" else 0.15
                f_x = 0.0
                f_y = (-600.0 * data.xpos[pelvis_id, 1] - 30.0 * data.qvel[1]) * tether_scale
                f_z = (-900.0 * (data.xpos[pelvis_id, 2] - 0.73) - 50.0 * data.qvel[2] + gravity_compensation) * tether_scale
                
                rot_error = np.cross(data.xmat[pelvis_id].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0])
                t_x = (800.0 * rot_error[0] - 25.0 * data.qvel[3]) * tether_scale
                t_y = (800.0 * rot_error[1] - 25.0 * data.qvel[4]) * tether_scale
                t_z = (400.0 * np.cross(data.xmat[pelvis_id].reshape(3, 3)[:, 0], [1.0, 0.0, 0.0])[2] - 15.0 * data.qvel[5]) * tether_scale

                data.xfrc_applied[pelvis_id] = [f_x, f_y, f_z, t_x, t_y, t_z]

                mujoco.mj_step(model, data)

            if time.time() - last_log_time > 1.0:
                print(f"🏁 [{state_label}] | Pelvis X: {data.xpos[pelvis_id, 0]:.3f}m | Tactile R/L: {r_sole_pressure:.1f}/{l_sole_pressure:.1f} N")
                last_log_time = time.time()

            viewer.cam.lookat[0] = data.xpos[pelvis_id, 0]
            viewer.sync()

            time_spent = time.time() - step_start
            if time_spent < 0.025:
                time.sleep(0.025 - time_spent)

if __name__ == "__main__":
    main()
