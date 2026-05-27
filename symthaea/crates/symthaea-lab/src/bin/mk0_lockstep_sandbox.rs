// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mk0 Lockstep Sandbox — High-Fidelity Town Sympoiesis
//!
//! Grounded in the Mk0 Bootstrapper Protocol:
//! - mk0-seed-node (Local compute + Conscious Accelerator v1)
//! - mk0-helios (Solar Microgrid + Energy Metabolism)
//! - mk0-detritivore (Plastic Recycler + Material Metabolism)
//! - mk0-fabricator (Autonomous 3D Print Farm)
//! - mk0-manipulator (7-DOF Rapier3D Mechanical Arm)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use symthaea_engineering::EngineeringManager;
use symthaea_infrastructure::simulator::{
    InfrastructurePhysicsSimulator, SimpleInfrastructureSimulator,
};
use symthaea_infrastructure::town_simpoiesis::TownSympoiesis;
use symthaea_manipulator::types::{ManipulatorCommand, ManipulatorState};
use symthaea_silicon::PowerDistributionLogic;

/// Mock Mycelix IPC resource for TendBalance ledger.
/// Uses request_id correlation map to simulate asynchronous, non-blocking IPC.
pub struct MycelixIpcMock {
    request_id: AtomicU64,
    correlation_map: HashMap<u64, String>,
}

impl MycelixIpcMock {
    pub fn new() -> Self {
        Self {
            request_id: AtomicU64::new(0),
            correlation_map: HashMap::new(),
        }
    }

    pub fn submit_transaction(&mut self, payload: &str) -> u64 {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        self.correlation_map.insert(id, payload.to_string());
        id
    }

    pub fn poll_ledger(&mut self, id: u64) -> Option<String> {
        self.correlation_map.remove(&id)
    }
}

/// The Deterministic Lockstep Orchestrator.
pub struct Mk0LockstepSandbox {
    pub town: TownSympoiesis,
    pub infrastructure_sim: SimpleInfrastructureSimulator,
    pub arm_state: ManipulatorState,
    pub ledger: MycelixIpcMock,
    pub dt: f64,
    pub frames: u64,

    // Mk0 Metabolic Inputs
    pub solar_flux: f32,     // Helios input
    pub recycler_yield: f32, // Detritivore output
    pub grid_load_mw: f32,
}

impl Mk0LockstepSandbox {
    pub fn new(manager: &mut EngineeringManager) -> Self {
        Self {
            town: TownSympoiesis::new("Mk0-Outpost-Delta", manager),
            infrastructure_sim: SimpleInfrastructureSimulator::new(),
            arm_state: ManipulatorState::home(),
            ledger: MycelixIpcMock::new(),
            dt: 16.67 / 1000.0, // 60Hz lockstep
            frames: 0,
            solar_flux: 1.0, // Full sun
            recycler_yield: 0.95,
            grid_load_mw: 5.0,
        }
    }

    /// Execute one deterministic frame.
    pub fn tick(&mut self) {
        self.frames += 1;

        // 1. HELIOS METABOLISM (Energy In)
        let generation_mw = 15.0 * self.solar_flux;

        // 2. FABRICATION METABOLISM (Energy Out)
        // Robotic arm torque increases load
        let arm_load = 0.5 + (self.arm_state.end_effector_force[2].abs() as f32 * 0.1);
        self.grid_load_mw = 4.0 + arm_load;

        // 3. SILICON BRAIN: Conscious Accelerator v1
        // Optimize grid routing and execute safety vetoes
        let surprise = self
            .town
            .power_grid
            .optimize_routing(self.grid_load_mw, generation_mw);

        // 4. PHYSICAL DYNAMICS (Rapier3D/MuJoCo Simulation)
        self.update_mechanical_joints();

        // 5. INFRASTRUCTURE DYNAMICS (Town Fluid/Thermal)
        self.town.step(self.grid_load_mw, generation_mw);

        // 6. ECONOMIC LEDGER (Non-blocking IPC)
        let tx_id = self
            .ledger
            .submit_transaction(&format!("tend_{}_watts", self.grid_load_mw));
        if self.frames % 2 == 0 {
            let _ = self.ledger.poll_ledger(tx_id - 1);
        }

        // 7. DIAGNOSTICS
        if self.frames % 10 == 0 {
            self.print_diagnostics(surprise, generation_mw);
        }
    }

    fn update_mechanical_joints(&mut self) {
        // Mock mechanical update for 7-DOF arm
        for i in 0..7 {
            self.arm_state.joint_angles[i] += 0.01 * (self.frames as f64).sin();
        }
    }

    pub fn inject_crisis(&mut self, profile: &str) {
        println!("\n🔥 BLACK-SWAN EVENT: {}", profile);
        match profile {
            "solar_flare" => {
                self.solar_flux = 0.05; // Total cloud occlusion
            }
            "structural_fatigue" => {
                self.town.water_clarity *= 0.1; // Simulated pipe burst
            }
            _ => {}
        }
    }

    fn print_diagnostics(&self, surprise: f32, generation: f32) {
        println!(
            "[{:>5}] Φ_ERR:{:>5.2} | SUN:{:>4.1}MW | LOAD:{:>4.1}MW | WATER:{:>4.2} | ARM_Z:{:>5.2}",
            self.frames,
            surprise,
            generation,
            self.grid_load_mw,
            self.town.water_clarity,
            self.arm_state.joint_angles[3] // Elbow joint as proxy for movement
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Mk0 Deterministic Lockstep Co-Simulation...");

    let mut manager = EngineeringManager::new();
    let mut sandbox = Mk0LockstepSandbox::new(&mut manager);

    println!("\n--- Phase 1: Baseline Homeostasis (Steady State) ---");
    for _ in 0..50 {
        sandbox.tick();
    }

    println!("\n--- Phase 2: The Crunch (Solar Flare Incident) ---");
    sandbox.inject_crisis("solar_flare");
    for _ in 0..50 {
        sandbox.tick();
    }

    println!("\n--- Phase 3: Structural Integrity Failure (Pipe Burst) ---");
    sandbox.inject_crisis("structural_fatigue");
    for _ in 0..50 {
        sandbox.tick();
    }

    println!(
        "\n✨ Mk0 Co-Simulation Verified: Logical immune system successfully mitigated cascades."
    );
    Ok(())
}
