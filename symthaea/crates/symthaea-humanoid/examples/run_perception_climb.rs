// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{Instant, Duration};
use std::thread;
use std::sync::{Arc, Mutex};

use symthaea_vision_manifold::{VisionManifold, VisionConfig};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};
use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
use symthaea_humanoid::types::HumanoidConfig; 

use symthaea_humanoid::{GaitGenome, Rng, GaitControlProfile, MachineState, CfcCpg, HdcWorkspace, execute_modular_gait};

const POPULATION_SIZE: usize = 4; 
const SIMULATION_HORIZON_STEPS: u32 = 1200; 

struct LocalRng {
    state: u64,
}

impl LocalRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) as f32 / (1u32 << 31) as f32
    }

    fn random_mutation(&mut self, base: f32, variance: f32) -> f32 {
        base + (self.next_f32() * 2.0 - 1.0) * variance
    }
}

#[derive(Clone)]
struct RolloutResult {
    genome: GaitGenome,
    distance: f32,
    steps_survived: u32,
    accumulated_effort: f32,
    tracking_error: f32,
}

struct DiagnosticTracker {
    best_dist: f32,
    best_effort: f32,
    best_survival: u32,
    best_tracking: f32,
}

fn run_fundamental_stance_check(act_count: usize, dt: f64) -> Result<(), String> {
    println!("🧪 [Baby Test] Evaluating Fundamental Stance Stability...");
    let mut sim = MuJoCoHumanoidSimulator::from_bundled_asset()
        .map_err(|e| format!("Failed to build validation asset: {:?}", e))?;
        
    let mut cpg = CfcCpg::zero();
    let cmd = symthaea_humanoid::types::HumanoidCommand::zero_for(act_count);
    
    // FIXED: Upgraded sensory precision variables to support 64-DoF masses during the baseline pass
    let static_profile = GaitControlProfile {
        target_velocity: 0.0,
        stride_amplitude: 0.0,
        knee_clearance: 0.45,
        target_lean: 0.02,
        kp: 80.0, 
        kd: 8.0,
        scale: 40.0,
        activation_alpha: 0.25,
    };

    for tick in 0..100 {
        let state = sim.state().clone();
        if state.root_height < 0.60 || state.root_height > 1.50 {
            return Err(format!("Catastrophic postural drop detected at tick {} (Height: {:.2}m)", tick, state.root_height));
        }
        let step_cmd = execute_modular_gait(
            &static_profile,
            0.0,
            &mut cpg,
            &state,
            0.0,
            act_count,
            &cmd,
            dt as f32
        );
        sim.step(&step_cmd, dt);
    }
    println!("✅ [Baby Test] Fundamental stance validated. Joint coordinates stable.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   **SYMTHAEA: PERSISTENT COGNITIVE CONTINUUM ENGINE** ║");
    println!("╚═══════════════════════════════════════════════════════════════╗");
    
    let cam_width: u32 = 320;
    let cam_height: u32 = 240;
    let channels: usize = 4;

    let humanoid_config = HumanoidConfig::default(); 
    let physics_dt = humanoid_config.physics_dt(); 

    let mut physics = MuJoCoHumanoidSimulator::from_bundled_asset()
        .map_err(|e| format!("❌ Failed loading bundled MJCF schema: {:?}", e))?;
    let total_actuators = physics.data_mut().ctrl_mut().len();

    if let Err(failure_reason) = run_fundamental_stance_check(total_actuators, physics_dt) {
        println!("❌ [Pipeline Abort] Fundamentals Validation Failed!");
        println!("   Reason: {}", failure_reason);
        return Ok(());
    }

    let mut agent_config = ActiveInferenceAgentConfig::default();
    agent_config.state_dim = 8;
    agent_config.obs_dim = 5;
    agent_config.num_actions = 4;
    agent_config.enable_td_learning = true;
    agent_config.action_temperature = 0.15;

    let async_surprise = Arc::new(Mutex::new(0.0f32));
    let vision_surprise_bridge = async_surprise.clone();
    
    thread::spawn(move || {
        let mut vision_manifold = VisionManifold::new(VisionConfig::default(), cam_width, cam_height);
        let mock_rgba_buffer = vec![128u8; (cam_width * cam_height * 4) as usize]; 
        let mut last_vision_time = Instant::now();
        loop {
            let loop_dt = last_vision_time.elapsed().as_secs_f64() as f32;
            last_vision_time = Instant::now();
            vision_manifold.observe_frame(&mock_rgba_buffer, cam_width, cam_height, channels, loop_dt);
            if let Ok(mut guard) = vision_surprise_bridge.lock() {
                *guard = vision_manifold.prediction_error();
            }
            thread::sleep(Duration::from_millis(25));
        }
    });

    let mut diags = DiagnosticTracker {
        best_dist: 0.0,
        best_effort: 999.0,
        best_survival: 0,
        best_tracking: 999.0,
    };
    
    let mut champion_genome = GaitGenome {
        walk_v: 0.45,
        walk_s: 0.28,
        walk_k: 0.20,
        climb_v: 0.30,
        climb_s: 0.18,
        climb_k: 0.30,
        vel_gain: 0.60,
        stride_gain: 0.10,
    };
    
    let mut best_score = -1000.0f32; 
    let mut master_rng = Rng::new(8888); 
    let mut generation = 1;
    let mut consecutive_failures = 0;

    let hdc = HdcWorkspace::new();
    let mut visual_brain = ActiveInferenceAgent::new(agent_config.clone());

    println!("🖥️ Launching main passive visualization context...");
    let mut window = mujoco_rs::viewer::MjViewer::launch_passive(physics.model_arc().clone(), 0)
        .map_err(|e| format!("❌ Viewport failure: {:?}", e))?;
    
    while window.running() {
        let burst_modifier = if consecutive_failures > 3 { 2.5f32 } else { 1.0f32 };
        
        println!("         [Generation {}] Launching Parallel Headless Rollouts...", generation);
        let mut worker_handles = vec![];
        for worker_id in 0..POPULATION_SIZE {
            let base_champion = champion_genome;
            let mut worker_rng = LocalRng::new(master_rng.next_f32().to_bits() as u64 + worker_id as u64);
            let fail_count = consecutive_failures;
            let act_count = total_actuators;
            let dt = physics_dt as f32;
            let variance_scale = burst_modifier;
            let local_agent_config = agent_config.clone();
            let thread_hdc = HdcWorkspace::new(); 
            
            let handle = thread::spawn(move || {
                let mut sim = match MuJoCoHumanoidSimulator::from_bundled_asset() {
                    Ok(s) => s,
                    Err(_) => return None,
                };
                
                let mutation_scale = if fail_count > 3 { 0.5f32 } else { 1.0f32 };
                let candidate_genome = GaitGenome {
                    walk_v: worker_rng.random_mutation(base_champion.walk_v, 0.06 * variance_scale * mutation_scale),
                    walk_s: worker_rng.random_mutation(base_champion.walk_s, 0.03 * variance_scale * mutation_scale),
                    walk_k: worker_rng.random_mutation(base_champion.walk_k, 0.03 * variance_scale * mutation_scale),
                    climb_v: worker_rng.random_mutation(base_champion.climb_v, 0.06 * variance_scale * mutation_scale),
                    climb_s: worker_rng.random_mutation(base_champion.climb_s, 0.03 * variance_scale * mutation_scale),
                    climb_k: worker_rng.random_mutation(base_champion.climb_k, 0.03 * variance_scale * mutation_scale),
                    vel_gain: worker_rng.random_mutation(base_champion.vel_gain, 0.05 * variance_scale * mutation_scale).max(0.0).min(1.5),
                    stride_gain: worker_rng.random_mutation(base_champion.stride_gain, 0.02 * variance_scale * mutation_scale).max(0.0).min(0.5),
                };
                
                let mut local_max_x = 0.0f32;
                let mut steps_active = 0u32;
                let mut thread_cpg = CfcCpg::zero();
                let mut running_effort = 0.0f32;
                let mut running_tracking_error = 0.0f32;
                let mut thread_cmd = symthaea_humanoid::types::HumanoidCommand::zero_for(act_count);
                
                let mut thread_brain = ActiveInferenceAgent::new(local_agent_config);
                let mut prev_action: Option<usize> = None;
                let mut thread_previous_hypervector = [0.0f32; 32];
                
                sim.reset_with_perturbation(0.0, worker_id as u64);
                
                for step in 0..SIMULATION_HORIZON_STEPS {
                    let state = sim.state().clone();
                    let torso_z = state.root_height as f32;
                    let current_x = sim.data_mut().qpos()[0] as f32; 
                    
                    if current_x > local_max_x { local_max_x = current_x; }
                    steps_active += 1;
                    
                    if torso_z < 0.55 || torso_z > 1.60 { break; }
                    
                    let current_stress = thread_cpg.homeostatic_stress;
                    let mind_is_quiescent = prev_action == Some(0) && current_stress < 0.03f32;

                    if step % 24 == 0 && !mind_is_quiescent {
                        let obs_pitch = state.joint_angles[0].abs().min(1.0); 
                        let obs_vel = ((state.root_linear_velocity[0] as f32 + 2.0f32) / 4.0f32).clamp(0.0, 1.0);
                        let obs_z = (state.root_height / 1.5).clamp(0.0, 1.0);
                        let next_stair_x = if current_x < 1.25 { 1.25 } else { 1.25 + ((current_x - 1.25) / 0.50).floor() * 0.50 + 0.50 };
                        let obs_dist = (next_stair_x - current_x).clamp(0.0, 1.0);
                        let obs_stress = current_stress as f64;

                        let current_hypervector = thread_hdc.encode(&state);
                        let mut semantic_surprise_variance = 0.0f32;
                        for i in 0..32 {
                            semantic_surprise_variance += (current_hypervector[i] - thread_previous_hypervector[i]).powi(2);
                        }
                        thread_previous_hypervector = current_hypervector;

                        let modulated_temperature = 0.08f32 + (semantic_surprise_variance * 4.5f32).min(0.40f32);
                        let observation = Observation::new(vec![obs_stress, obs_pitch as f64, obs_vel as f64, obs_z, obs_dist as f64], modulated_temperature as f64, "holographic");
                        
                        if let Some(last_act) = prev_action {
                            thread_brain.learn_from_outcome(last_act, &observation);
                        } else {
                            thread_brain.perceive(&observation);
                        }

                        let sel_act = thread_brain.select_action().action;
                        prev_action = Some(sel_act);

                        match sel_act {
                            0 => thread_brain.set_goals(vec![0.0, 0.0, 0.00, 0.90, 0.50], 9.0),
                            1 => thread_brain.set_goals(vec![0.0, 0.18, 0.75, 0.90, 0.50], 5.0),
                            2 => thread_brain.set_goals(vec![0.0, 0.14, 0.60, 0.95, 0.20], 6.0),
                            _ => thread_brain.set_goals(vec![0.0, 0.24, 0.55, 0.95, 0.0], 8.0),
                        }
                    }

                    let current_action = prev_action.unwrap_or(0);
                    let profile = match current_action {
                        0 => GaitControlProfile { target_velocity: 0.0, stride_amplitude: 0.0, knee_clearance: 0.45, target_lean: 0.02, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 },
                        1 => GaitControlProfile { target_velocity: candidate_genome.walk_v, stride_amplitude: candidate_genome.walk_s, knee_clearance: candidate_genome.walk_k, target_lean: 0.18, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 },
                        2 => GaitControlProfile { target_velocity: candidate_genome.walk_v * 0.6, stride_amplitude: candidate_genome.walk_s * 1.2, knee_clearance: candidate_genome.walk_k + 0.15, target_lean: 0.14, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 },
                        _ => GaitControlProfile { target_velocity: candidate_genome.climb_v, stride_amplitude: candidate_genome.climb_s, knee_clearance: candidate_genome.climb_k, target_lean: 0.24, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 },
                    };

                    thread_cmd = execute_modular_gait(
                        &profile,
                        candidate_genome.stride_gain,
                        &mut thread_cpg,
                        &state,
                        0.0f32,
                        act_count,
                        &thread_cmd,
                        dt
                    );
                    
                    running_effort += thread_cmd.control_effort();
                    running_tracking_error += (profile.target_lean - state.joint_angles[0] as f32).powi(2);
                    sim.step(&thread_cmd, physics_dt);
                }
                
                Some(RolloutResult {
                    genome: candidate_genome,
                    distance: local_max_x,
                    steps_survived: steps_active,
                    accumulated_effort: running_effort,
                    tracking_error: (running_tracking_error / steps_active as f32).sqrt(),
                })
            });
            worker_handles.push(handle);
        }
        
        let mut batch_results = vec![];
        for handle in worker_handles {
            if let Ok(Some(result)) = handle.join() {
                batch_results.push(result);
            }
        }
        
        if let Some(winner) = batch_results.iter().max_by(|a, b| {
            let base_score_a = if a.steps_survived < SIMULATION_HORIZON_STEPS { a.distance - 5.0f32 } else { a.distance + 5.0f32 };
            let base_score_b = if b.steps_survived < SIMULATION_HORIZON_STEPS { b.distance - 5.0f32 } else { b.distance + 5.0f32 };
            let metabolic_fitness_a = base_score_a - (a.accumulated_effort * 0.040f32);
            let metabolic_fitness_b = base_score_b - (b.accumulated_effort * 0.040f32);
            metabolic_fitness_a.partial_cmp(&metabolic_fitness_b).unwrap()
        }) {
            let base_score = if winner.steps_survived < SIMULATION_HORIZON_STEPS { winner.distance - 5.0f32 } else { winner.distance + 5.0f32 };
            let candidate_score = base_score - (winner.accumulated_effort * 0.040f32);
            
            if candidate_score > best_score + 0.01f32 {
                best_score = candidate_score;
                champion_genome = winner.genome;
                consecutive_failures = 0;

                println!("\n📊 ────────── [GENERATION {} METRIC EVOLUTION] ──────────", generation);
                println!("| Dimension           | Prior Historical Best | Current Gen Champion | Delta Status |");
                println!("|---------------------|-----------------------|----------------------|--------------|");
                println!("| Stride Distance     | {:>19.2}m | {:>18.2}m | {:>12} |", diags.best_dist, winner.distance, if winner.distance > diags.best_dist { "🔥 IMPROVED" } else { "● STABLE" });
                println!("| Avg Control Effort  | {:>21.3} | {:>20.3} | {:>12} |", diags.best_effort, winner.accumulated_effort / winner.steps_survived as f32, if (winner.accumulated_effort / winner.steps_survived as f32) < diags.best_effort { "🍃 GREEN" } else { "● BOUNDED" });
                println!("| Posture RMSE Error  | {:>21.3} | {:>20.3} | {:>12} |", diags.best_tracking, winner.tracking_error, if winner.tracking_error < diags.best_tracking { "🎯 SHARPER" } else { "● LOCKED" });
                println!("| Survival Horizon    | {:>17} steps | {:>16} steps | {:>12} |", diags.best_survival, winner.steps_survived, if winner.steps_survived > diags.best_survival { "🚀 LONGER" } else { "● MATCHED" });
                println!("──────────────────────────────────────────────────────────────────────────\n");

                diags.best_dist = diags.best_dist.max(winner.distance);
                diags.best_effort = diags.best_effort.min(winner.accumulated_effort / winner.steps_survived as f32);
                diags.best_survival = diags.best_survival.max(winner.steps_survived);
                diags.best_tracking = diags.best_tracking.min(winner.tracking_error);
            } else {
                consecutive_failures += 1;
                println!("💀 [Generation {}] Landscape exploration active. Searching alternative topologies.", generation);
            }
        }

        println!("📺 Demonstrating Current Champion Trajectory in Viewport Context...");
        let mut visual_steps = 0;
        let mut visual_cpg = CfcCpg::zero();
        let mut previous_brain_action: Option<usize> = None;
        let mut visual_state = MachineState::ActiveInference;
        let mut visual_cmd = symthaea_humanoid::types::HumanoidCommand::zero_for(total_actuators);
        
        let mut previous_hypervector = [0.0f32; 32];
        
        physics.reset_with_perturbation(0.0, 1337);
        visual_brain.end_episode(); 

        while visual_steps < SIMULATION_HORIZON_STEPS && window.running() {
            let current_visual_surprise = if let Ok(guard) = async_surprise.lock() { *guard } else { 0.0f32 };
            let cognitive_crouch = if current_visual_surprise > 15.0 { current_visual_surprise * 0.008f32 } else { 0.0f32 };

            let state = physics.state().clone(); 
            let current_x_pos = physics.data_mut().qpos()[0] as f32; 
            let actual_vel_x = state.root_linear_velocity[0] as f32;

            let current_stress = visual_cpg.homeostatic_stress;
            let mind_is_quiescent = previous_brain_action == Some(0) && current_stress < 0.03f32;

            if visual_steps % 24 == 0 && !mind_is_quiescent {
                let obs_pitch = state.joint_angles[0].abs().min(1.0); 
                let obs_vel = ((actual_vel_x + 2.0f32) / 4.0f32).clamp(0.0, 1.0);
                let obs_z = (state.root_height / 1.5).clamp(0.0, 1.0); 
                let next_stair_x = if current_x_pos < 1.25 { 1.25 } else { 1.25 + ((current_x_pos - 1.25) / 0.50).floor() * 0.50 + 0.50 };
                let obs_dist = (next_stair_x - current_x_pos).clamp(0.0, 1.0);
                let obs_stress = current_stress as f64;

                let current_hypervector = hdc.encode(&state);
                let mut semantic_surprise_variance = 0.0f32;
                for i in 0..32 {
                    semantic_surprise_variance += (current_hypervector[i] - previous_hypervector[i]).powi(2);
                }
                previous_hypervector = current_hypervector;

                let modulated_temperature = 0.08f32 + (semantic_surprise_variance * 4.5f32).min(0.40f32);
                let observation = Observation::new(vec![obs_stress, obs_pitch as f64, obs_vel as f64, obs_z, obs_dist as f64], modulated_temperature as f64, "holographic"); 

                if let Some(last_act) = previous_brain_action {
                    visual_brain.learn_from_outcome(last_act, &observation); 
                } else {
                    visual_brain.perceive(&observation); 
                }

                let selected_action = visual_brain.select_action().action; 
                previous_brain_action = Some(selected_action);

                match selected_action {
                    0 => visual_brain.set_goals(vec![0.0, 0.0, 0.00, 0.90, 0.50], 9.0), 
                    1 => visual_brain.set_goals(vec![0.0, 0.18, 0.75, 0.90, 0.50], 5.0),
                    2 => visual_brain.set_goals(vec![0.0, 0.14, 0.60, 0.95, 0.20], 6.0),
                    _ => visual_brain.set_goals(vec![0.0, 0.24, 0.55, 0.95, 0.0], 8.0),
                }
            }

            let current_action = previous_brain_action.unwrap_or(0);
            let (profile, label) = match current_action {
                0 => (GaitControlProfile { target_velocity: 0.0, stride_amplitude: 0.0, knee_clearance: 0.45, target_lean: 0.02, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 }, "CognitiveStand"),
                1 => (GaitControlProfile { target_velocity: champion_genome.walk_v, stride_amplitude: champion_genome.walk_s, knee_clearance: champion_genome.walk_k, target_lean: 0.18, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 }, "CognitiveCharge"),
                2 => (GaitControlProfile { target_velocity: champion_genome.walk_v * 0.6, stride_amplitude: champion_genome.walk_s * 1.2, knee_clearance: champion_genome.walk_k + 0.15, target_lean: 0.14, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 }, "HighStepClearance"),
                _ => (GaitControlProfile { target_velocity: champion_genome.climb_v, stride_amplitude: champion_genome.climb_s, knee_clearance: champion_genome.climb_k, target_lean: 0.24, kp: 42.0, kd: 5.5, scale: 40.0, activation_alpha: 0.25 }, "CentroidalTransition"),
            };

            let current_state = physics.state().clone(); 
            let torso_z_height = current_state.root_height; 
            let visual_x = physics.data_mut().qpos()[0] as f32; 
            
            visual_steps += 1;

            if torso_z_height < 0.55 || torso_z_height > 1.60 { 
                visual_state = MachineState::Recovering;
            }

            if visual_state == MachineState::ActiveInference {
                visual_cmd = execute_modular_gait(
                    &profile,
                    champion_genome.stride_gain,
                    &mut visual_cpg,
                    &current_state,
                    cognitive_crouch,
                    total_actuators,
                    &visual_cmd,
                    physics_dt as f32
                );
                physics.step(&visual_cmd, physics_dt); 
            }

            if visual_state == MachineState::Recovering {
                println!("💥 [Viewer] Posture validation breached. Resetting execution frame instantly.");
                break; 
            }

            window.sync_data(physics.data_mut()); 
            window.render();
            
            if visual_steps % 20 == 0 {
                let sim_time = physics.state().timestamp;
                println!(
                    "   🤖 [HLB Output] Sim: {:.2}s | Dist: {:.2}m | Mode: {} | Effort: {:.2} | Stress: {:.2}",
                    sim_time, visual_x, label, visual_cmd.control_effort(), visual_cpg.homeostatic_stress
                );
            }
            thread::sleep(Duration::from_millis(25));
        }
        
        generation += 1;
    }

    Ok(())
}
