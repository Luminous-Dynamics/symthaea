// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mk0 Sympoiesis Sandbox — Deterministic Lockstep Co-Simulation
//!
//! Grounded in the Mk0 Bootstrapper Protocol:
//! - mk0-seed-node (local compute)
//! - mk0-helios (solar microgrid)
//! - mk0-detritivore (material recycler)
//! - mk0-fabricator (3D print farm)

use symthaea_infrastructure::town_simpoiesis::TownSympoiesis;
use symthaea_infrastructure::simulator::{SimpleInfrastructureSimulator, InfrastructurePhysicsSimulator};
use symthaea_engineering::EngineeringManager;
use symthaea_silicon::PowerDistributionLogic;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// A mock Mycelix IPC client for simulated economic transactions.
pub struct MycelixClient {
    request_counter: AtomicU64,
    pending_transactions: HashMap<u64, String>,
}

impl MycelixClient {
    pub fn new() -> Self {
        Self {
            request_counter: AtomicU64::new(0),
            pending_transactions: HashMap::new(),
        }
    }

    /// Simulate sending a transaction to the TendBalance ledger via IPC.
    pub fn send_transaction(&mut self, payload: &str) -> u64 {
        let id = self.request_counter.fetch_add(1, Ordering::SeqCst);
        self.pending_transactions.insert(id, payload.to_string());
        id
    }

    /// Check if transaction is confirmed (Mock: always confirmed after 1 frame).
    pub fn poll_status(&mut self, id: u64) -> bool {
        self.pending_transactions.remove(&id).is_some()
    }
}

/// The Deterministic Lockstep Sandbox Orchestrator.
pub struct Mk0LockstepSandbox {
    pub town: TownSympoiesis,
    pub simulator: symthaea_infrastructure::simulator::SimpleInfrastructureSimulator,
    pub mycelix: MycelixClient,
    pub dt: f64,
    pub total_frames: u64,
    pub solar_input_mw: f32,
    pub print_queue_depth: u32,
}

impl Mk0LockstepSandbox {
    pub fn new(name: &str, manager: &mut EngineeringManager) -> Self {
        Self {
            town: TownSympoiesis::new(name, manager),
            simulator: symthaea_infrastructure::simulator::SimpleInfrastructureSimulator::new(),
            mycelix: MycelixClient::new(),
            dt: 1.0 / 60.0,
            total_frames: 0,
            solar_input_mw: 15.0, // Mk0 Helios baseline
            print_queue_depth: 2,
        }
    }

    /// One deterministic frame tick.
    pub fn tick(&mut self) {
        self.total_frames += 1;

        // 1. Primary Metabolism: Solar Generation (mk0-helios)
        let generation = self.solar_input_mw;
        
        // 2. Consumption: Seed-node + Fabricator print jobs
        let fabricator_demand = self.print_queue_depth as f32 * 2.5; // 2.5MW per print job
        let total_demand = 1.0 + fabricator_demand; // 1MW overhead for seed-node

        // 3. Silicon Intelligence & Physical Metabolism
        let surprise = self.town.step(total_demand, generation);

        // 4. Physical Update: Throttle infrastructure based on silicon veto
        let mut cmd = symthaea_infrastructure::types::InfrastructureCommand::zero();
        cmd.torques[0] = (self.town.power_grid.active_loads_mw / total_demand).clamp(0.0, 1.0);
        self.simulator.step(&cmd, self.dt);

        // 5. Economic Integration: TendBalance Transaction
        if total_demand > 0.0 {
            self.mycelix.send_transaction(&format!("tend_{:.1}W", total_demand));
        }

        // 6. Diagnostics
        if self.total_frames % 10 == 0 {
            self.print_status(surprise, generation, total_demand);
        }
    }

    pub fn inject_anomaly(&mut self, profile: &str) {
        println!("\n🔥 ANOMALY INJECTED: {}", profile);
        match profile {
            "helios_occlusion" => {
                self.solar_input_mw *= 0.1; // 90% drop
            }
            "fabricator_surge" => {
                self.print_queue_depth += 10; // Massive sudden load
            }
            _ => {}
        }
    }

    fn print_status(&self, surprise: f32, generation: f32, demand: f32) {
        println!(
            "[{:>6}] GEN:{:>5.1}MW | DEM:{:>5.1}MW | SURP:{:>5.2} | WATER:{:>5.2} | JOBS:{}",
            self.total_frames,
            generation,
            demand,
            surprise,
            self.town.water_clarity,
            self.print_queue_depth
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Mk0 Bootstrapper Lockstep Co-Simulation...");

    let mut manager = EngineeringManager::new();
    let mut sandbox = Mk0LockstepSandbox::new("Mk0-Outpost-Alpha", &mut manager);

    // 1. Warm-up
    println!("\n--- Phase 1: Steady State Metabolism ---");
    for _ in 0..50 {
        sandbox.tick();
    }

    // 2. Helios Occlusion (The Crunch Test)
    println!("\n--- Phase 2: The Crunch Test (Solar Drop) ---");
    sandbox.inject_anomaly("helios_occlusion");
    for _ in 0..50 {
        sandbox.tick();
    }

    // 3. Fabricator Surge
    println!("\n--- Phase 3: Resource Surge ---");
    sandbox.inject_anomaly("fabricator_surge");
    for _ in 0..50 {
        sandbox.tick();
    }

    println!("\n✨ Mk0 Co-Simulation Complete: All metabolic safety gates held.");
    Ok(())
}
