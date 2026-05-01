// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin: wires resources, systems, and schedules.

use bevy::prelude::*;
use symthaea_manipulator::simulator::ManipulatorPhysicsSimulator;
use symthaea_manipulator::types::NUM_JOINTS;

use crate::admittance_control::AdmittanceController;
use crate::camera;
use crate::consciousness_bridge;
use crate::grasp_object::GraspObject;
use crate::hud;
use crate::human_obstacle::HumanObstacle;
use crate::prediction_bridge;
use crate::resources::*;
use crate::visualization;

/// Main plugin for the manipulator demo.
pub struct ManipulatorDemoPlugin;

impl Plugin for ManipulatorDemoPlugin {
    fn build(&self, app: &mut App) {
        app
            // Resources
            .insert_resource(PhiArmState::new())
            .insert_resource(IsoArmState::new())
            .insert_resource(SharedHuman {
                obstacle: HumanObstacle::new(12345),
            })
            .insert_resource(AdaptiveGraspObject {
                object: GraspObject::at_pick(),
            })
            .insert_resource(IsoGraspObject {
                object: GraspObject::at_pick(),
            })
            .insert_resource(SimTime::default())
            .insert_resource(ThroughputMetrics::default())
            // Startup systems
            .add_systems(
                Startup,
                (
                    camera::setup_camera,
                    visualization::setup_scene,
                    hud::setup_hud,
                ),
            )
            // Per-frame update systems (ordered)
            .add_systems(
                Update,
                (
                    update_sim_time,
                    update_human,
                    step_phi_arm,
                    step_iso_arm,
                    update_throughput_metrics,
                    visualization::update_human_visual,
                    visualization::update_iso_arm_visual,
                    visualization::update_workspace_envelope,
                    hud::update_phi_hud,
                    hud::update_iso_hud,
                    hud::update_throughput_hud,
                    print_status,
                )
                    .chain(),
            );
    }
}

/// Advance simulation time.
fn update_sim_time(time: Res<Time>, mut sim_time: ResMut<SimTime>) {
    sim_time.elapsed += time.delta_secs_f64();
}

/// Step the human obstacle.
fn update_human(time: Res<Time>, mut human: ResMut<SharedHuman>) {
    human.obstacle.update(time.delta_secs_f64());
}

/// Step the Phi-gradient arm: consciousness-gated admittance control.
///
/// The emergent causal chain:
/// 1. Human proximity → external force on EE
/// 2. External force → state change → HDC prediction error spike
/// 3. PE + danger → RoboticAgent.tick() → MasterConsciousnessEquation → Phi
/// 4. Phi → admittance stiffness → IK/compliant torque blend
/// 5. Blended torques → physics step → cycle continues or pauses
fn step_phi_arm(
    time: Res<Time>,
    human: Res<SharedHuman>,
    sim_time: Res<SimTime>,
    mut phi_arm: ResMut<PhiArmState>,
    mut grasp_obj: ResMut<AdaptiveGraspObject>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    let state = phi_arm.simulator.state().clone();
    let ee_pos = state.end_effector_position;

    // Update grasp object (track gripper state)
    grasp_obj.object.update(&ee_pos, state.gripper_opening);

    // === Step 1: Compute external forces ===
    // Human proximity repulsive force
    let human_forces = prediction_bridge::compute_human_interaction_forces(&human.obstacle, &state);
    // Object gravity load (changes arm dynamics when holding object)
    let gravity_load = grasp_obj.object.gravity_load();

    // Combined external forces
    let external_forces = [
        human_forces[0] + gravity_load[0],
        human_forces[1] + gravity_load[1],
        human_forces[2] + gravity_load[2],
    ];

    // Apply external forces to simulator
    phi_arm.simulator.external_forces = external_forces;

    // === Step 2: Compute danger level for consciousness input ===
    // Combine human proximity danger with encoder confidence.
    // When the encoder is surprised (low confidence), danger increases even
    // without direct human proximity — e.g., holding an object changes dynamics.
    let human_danger = prediction_bridge::danger_level_from_human(&human.obstacle, &ee_pos);
    let encoder_surprise = phi_arm.last_prediction_error as f64; // PE from HDC encoder
    let danger = (human_danger + encoder_surprise * 0.5).min(1.0);

    // === Step 3: Run full consciousness pipeline ===
    let last_pe = phi_arm.last_prediction_error;
    let (phi, safety_tier, _motor_gain) =
        consciousness_bridge::consciousness_tick(&mut phi_arm.robot_agent, last_pe, danger);
    phi_arm.current_phi = phi;
    phi_arm.current_safety = safety_tier;

    // === Step 4: Update admittance stiffness from Phi ===
    phi_arm.stiffness = phi.clamp(0.1, 1.0);
    let mut admittance = AdmittanceController {
        stiffness: phi_arm.stiffness,
    };

    // === Step 5: Determine if task should pause ===
    // Orange/Red → pause task (arm retreats or holds)
    let should_pause = phi < 0.3;
    phi_arm.task.set_paused(should_pause);

    // === Step 6: Compute IK torques toward assembly target ===
    let target = phi_arm.task.current_target();
    let ik_torques = if let Some(q_target) =
        phi_arm
            .kinematics
            .ik_dls(&target, &state.joint_angles, 0.1, 50, 0.01)
    {
        let mut torques = [0.0f32; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            let error = q_target[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];
            let kp = 8.0;
            let kd = 2.0;
            torques[i] = ((kp * error - kd * vel) as f32).clamp(-1.0, 1.0);
        }
        torques
    } else {
        [0.0f32; NUM_JOINTS] // IK failed → zero torque (safe)
    };

    // === Step 7: Admittance blend — prevents IK/DLS chatter ===
    let blended_torques = admittance.blend_torques(
        &ik_torques,
        &external_forces,
        &state.joint_angles,
        &phi_arm.kinematics,
    );

    // === Step 8: Apply blended torques to physics ===
    let mut cmd = symthaea_manipulator::ManipulatorCommand::zero();
    cmd.joint_torques = blended_torques;
    cmd.gripper = phi_arm.task.desired_gripper() as f32;
    phi_arm.simulator.step(&cmd, dt);

    // === Step 9: Compute prediction error via HDC cosine dissimilarity ===
    // Encode current state into 16,384D hypervector
    let new_state = phi_arm.simulator.state().clone();
    let current_hv = phi_arm.encoder.encode(&new_state);

    // PE = 1 - cosine_similarity(current, previous)
    // This is the REAL prediction error from HDC space, not a proxy.
    // Force channels have weight 2.5 (highest), so human proximity forces
    // naturally dominate the dissimilarity.
    let pe = if let Some(ref prev_hv) = phi_arm.last_perception {
        let sim = current_hv.similarity(prev_hv);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0 // First tick: no previous perception
    };
    phi_arm.last_prediction_error = pe;
    phi_arm.last_perception = Some(current_hv);

    // === Step 10: Update task state machine ===
    let ee_pos = new_state.end_effector_position;
    let cycle_completed = phi_arm.task.update(&ee_pos, dt);

    if cycle_completed {
        let cycle_time = sim_time.elapsed - phi_arm.current_cycle_start;
        phi_arm.total_cycle_time += cycle_time;
        phi_arm.cycles_completed += 1;
        phi_arm.current_cycle_start = sim_time.elapsed;
        grasp_obj.object.reset_to_home();
    }
}

/// Step the ISO arm: IK toward task target, binary stop/go on human proximity.
fn step_iso_arm(
    time: Res<Time>,
    human: Res<SharedHuman>,
    sim_time: Res<SimTime>,
    mut iso: ResMut<IsoArmState>,
    mut grasp_obj: ResMut<IsoGraspObject>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    let state = iso.simulator.state().clone();
    let ee_pos = state.end_effector_position;

    // Update grasp object
    grasp_obj.object.update(&ee_pos, state.gripper_opening);

    // Compute human distance to EE
    let human_dist = human.obstacle.distance_to(&ee_pos);

    // Estimate robot TCP speed from joint velocities
    let robot_speed = (state.joint_velocities.iter().map(|v| v * v).sum::<f64>() * 0.01).sqrt();

    // ISO SSM decision: binary stop/go
    let gain = iso.ssm.motor_gain(human_dist, robot_speed);
    iso.is_stopped = gain == 0.0;
    iso.current_protective_distance = iso.ssm.protective_distance(robot_speed);

    let stopped = iso.is_stopped;
    iso.task.set_paused(stopped);

    if !iso.is_stopped {
        let target = iso.task.current_target();
        if let Some(q_target) = iso
            .kinematics
            .ik_dls(&target, &state.joint_angles, 0.1, 50, 0.01)
        {
            let mut cmd = symthaea_manipulator::ManipulatorCommand::zero();
            for i in 0..NUM_JOINTS {
                let error = q_target[i] - state.joint_angles[i];
                let vel = state.joint_velocities[i];
                let kp = 8.0;
                let kd = 2.0;
                cmd.joint_torques[i] = ((kp * error - kd * vel) as f32).clamp(-1.0, 1.0);
            }
            cmd.gripper = iso.task.desired_gripper() as f32;
            // Apply object gravity load to ISO arm too (fair comparison)
            iso.simulator.external_forces = grasp_obj.object.gravity_load();
            iso.simulator.step(&cmd, dt);
        }
    }

    let ee_pos = iso.simulator.state().end_effector_position;
    let cycle_completed = iso.task.update(&ee_pos, dt);

    if cycle_completed {
        let cycle_time = sim_time.elapsed - iso.current_cycle_start;
        iso.total_cycle_time += cycle_time;
        iso.cycles_completed += 1;
        iso.current_cycle_start = sim_time.elapsed;
        grasp_obj.object.reset_to_home();
    }
}

/// Update throughput comparison metrics.
fn update_throughput_metrics(
    iso: Res<IsoArmState>,
    phi: Res<PhiArmState>,
    mut metrics: ResMut<ThroughputMetrics>,
) {
    let iso_mean = iso.mean_cycle_time();
    let phi_mean = phi.mean_cycle_time();

    if iso_mean > 0.0 && phi_mean > 0.0 {
        metrics.advantage_percent = ((iso_mean - phi_mean) / iso_mean) * 100.0;
    }
}

/// Print status to terminal periodically (every ~2 seconds).
fn print_status(
    sim_time: Res<SimTime>,
    iso: Res<IsoArmState>,
    phi_arm: Res<PhiArmState>,
    human: Res<SharedHuman>,
    metrics: Res<ThroughputMetrics>,
) {
    let tick = (sim_time.elapsed * 0.5) as u64;
    let prev_tick = ((sim_time.elapsed - 0.016) * 0.5) as u64;
    if tick == prev_tick {
        return;
    }

    let iso_ee = iso.simulator.state().end_effector_position;
    let human_dist = human.obstacle.distance_to(&iso_ee);

    println!(
        "[{:.1}s] Adaptive: cert={:.3} {:?} stiff={:.0}% | {} cycles ({:.2}s) | \
         ISO: {} cycles ({:.2}s) | Advantage: {:.1}% | Human: {} ({:.2}m)",
        sim_time.elapsed,
        phi_arm.current_phi,
        phi_arm.current_safety,
        phi_arm.stiffness * 100.0,
        phi_arm.cycles_completed,
        phi_arm.mean_cycle_time(),
        iso.cycles_completed,
        iso.mean_cycle_time(),
        metrics.advantage_percent,
        human.obstacle.phase_name(),
        human_dist,
    );
}
