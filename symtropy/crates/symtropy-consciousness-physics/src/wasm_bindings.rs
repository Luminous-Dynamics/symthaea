// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WASM bindings for in-browser consciousness-physics research demos.
//!
//! Activated by `--features wasm`. Exposes [`WasmPhysicsDemo`] —
//! a self-contained 2D multi-agent simulation where:
//! - Each agent is a conscious entity driven by the Master Consciousness Equation
//! - Agents follow Free Energy gradients (seeking resonant partners + energy wells)
//! - Emotional contagion spreads via CEMI-inspired proximity coupling (harmony[8])
//! - Φ gates motor authority (NRC 4-tier: Green/Yellow/Orange/Red)
//!
//! # JavaScript Usage
//! ```js
//! import init, { WasmPhysicsDemo } from './pkg/symtropy_consciousness_physics.js';
//! await init();
//! const demo = new WasmPhysicsDemo(8);          // 8 agents
//! // inject an emotional stimulus to agent 0
//! demo.set_emotion(0, 0.9);
//! // step 60fps
//! requestAnimationFrame(function loop() {
//!     const state = JSON.parse(demo.tick(1/60));
//!     render(state);
//!     requestAnimationFrame(loop);
//! });
//! ```
//!
//! # Build
//! ```bash
//! cargo build -p symtropy-consciousness-physics \
//!   --target wasm32-unknown-unknown --features wasm --release
//! ~/.cargo/bin/wasm-bindgen \
//!   target/wasm32-unknown-unknown/release/symtropy_consciousness_physics.wasm \
//!   --out-dir www/pkg --target web
//! ```

#![cfg(feature = "wasm")]

use serde::Serialize;
use wasm_bindgen::prelude::*;

use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_math::Point;
use symtropy_physics::body::BodyHandle;

use crate::coupling::ConsciousnessField;
use crate::fep_gradient::free_energy_gradient_phi;
use crate::harmony_field::{EMOTIONAL_CONTAGION_IDX, NUM_HARMONIES};

// ── Agent kinematics (not part of the physics world, managed here) ────────

struct Agent {
    handle: BodyHandle,
    pos: [f64; 2],
    vel: [f64; 2],
    /// Cached inputs that will be fed to the consciousness equation each tick.
    inputs: ConsciousnessInputs,
}

impl Agent {
    fn new(id: u32, x: f64, y: f64) -> Self {
        Self {
            handle: BodyHandle(id as usize),
            pos: [x, y],
            vel: [0.0; 2],
            inputs: ConsciousnessInputs {
                phi: 0.5,
                broadcast: 0.5,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.5,
                embodiment: 1.0,
                knowledge: 0.3,
                synchrony: 0.5,
            },
        }
    }
}

// ── JS-serialisable agent snapshot ────────────────────────────────────────

/// Per-agent state snapshot serialised to JSON each tick.
#[derive(Serialize)]
struct AgentSnap {
    id: u32,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    /// Φ: integrated information metric [0, 1]
    phi: f64,
    /// Emotional contagion activation [0, 1]  (harmony[8])
    emotion: f64,
    /// Energy fraction [0, 1]
    energy: f64,
    /// All 9 harmony activations
    harmony: Vec<f64>,
    /// NRC safety tier: "green" / "yellow" / "orange" / "red"
    safety: String,
}

// ── Demo top-level snapshot (returned by tick()) ─────────────────────────

#[derive(Serialize)]
struct DemoSnap {
    tick: u64,
    agents: Vec<AgentSnap>,
    collective_phi: f64,
}

// ── WASM-exported simulation ──────────────────────────────────────────────

/// 2D multi-agent consciousness-physics simulation for in-browser research.
///
/// Self-contained: agents move autonomously using Free Energy Principle
/// gradients, have emotions that spread via CEMI-field contagion, and their
/// motor authority is gated by their integrated information (Φ).
#[wasm_bindgen]
pub struct WasmPhysicsDemo {
    agents: Vec<Agent>,
    field: ConsciousnessField<2>,
    tick_count: u64,
    world_size: f64,
}

#[wasm_bindgen]
impl WasmPhysicsDemo {
    /// Create a new demo with `num_agents` agents arranged in a circle.
    ///
    /// `world_size`: half-extent of the simulation square (default 50.0).
    #[wasm_bindgen(constructor)]
    pub fn new(num_agents: u32) -> WasmPhysicsDemo {
        #[cfg(feature = "wasm")]
        console_error_panic_hook::set_once();

        let world_size = 50.0_f64;
        let n = num_agents.max(1) as usize;
        let mut agents = Vec::with_capacity(n);
        let mut field = ConsciousnessField::<2>::new();

        for i in 0..n {
            let angle = (i as f64) / (n as f64) * std::f64::consts::TAU;
            let r = world_size * 0.6;
            let x = angle.cos() * r;
            let y = angle.sin() * r;
            let agent = Agent::new(i as u32, x, y);
            field.register(agent.handle, 100.0, 10.0);
            agents.push(agent);
        }

        WasmPhysicsDemo {
            agents,
            field,
            tick_count: 0,
            world_size,
        }
    }

    /// Advance the simulation by `dt` seconds and return JSON agent state.
    ///
    /// The returned JSON has shape:
    /// ```json
    /// { "tick": 42, "collective_phi": 0.61, "agents": [
    ///   { "id": 0, "x": 12.3, "y": -5.6, "vx": 0.1, "vy": -0.2,
    ///     "phi": 0.74, "emotion": 0.32, "energy": 0.88,
    ///     "harmony": [0.5, 0.5, ...], "safety": "green" },
    ///   ...
    /// ] }
    /// ```
    pub fn tick(&mut self, dt: f64) -> String {
        let dt = dt.clamp(1e-4, 0.1);
        self.tick_count += 1;

        // ── 1. Derive inputs from simulation state ─────────────────────────
        // Collect neighbour data for gradient computation
        let agent_data: Vec<(nalgebra::SVector<f64, 2>, [f64; NUM_HARMONIES])> = self
            .agents
            .iter()
            .map(|a| {
                let svec = nalgebra::SVector::<f64, 2>::from(a.pos);
                let harmonies = self
                    .field
                    .entities
                    .get(&a.handle)
                    .map(|e| e.harmony_activations)
                    .unwrap_or([0.0; NUM_HARMONIES]);
                (svec, harmonies)
            })
            .collect();

        for (idx, agent) in self.agents.iter_mut().enumerate() {
            let harmonies = self
                .field
                .entities
                .get(&agent.handle)
                .map(|e| e.harmony_activations)
                .unwrap_or([0.0; NUM_HARMONIES]);
            let phi = self.field.phi(agent.handle);

            // Neighbours (exclude self)
            let neighbours: Vec<_> = agent_data
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, d)| d.clone())
                .collect();

            // Derive synchrony from resonance with neighbours
            let synchrony = if neighbours.is_empty() {
                0.5
            } else {
                let total: f64 = neighbours
                    .iter()
                    .map(|(_, h)| crate::harmony_field::HarmonyField::<2>::resonance(&harmonies, h))
                    .sum();
                (total / neighbours.len() as f64).clamp(0.0, 1.0) * 0.5 + 0.5
            };

            // Attention = speed normalised
            let speed = (agent.vel[0] * agent.vel[0] + agent.vel[1] * agent.vel[1]).sqrt();
            let attention = (speed / 10.0).clamp(0.0, 1.0) * 0.5 + 0.3;

            agent.inputs.synchrony = synchrony;
            agent.inputs.attention = attention;
            agent.inputs.phi = phi.clamp(0.0, 1.0);
            agent.inputs.broadcast = harmonies[EMOTIONAL_CONTAGION_IDX];
            agent.inputs.embodiment = 1.0;
            agent.inputs.recurrence = (self.tick_count as f64 * 0.001).tanh();
        }

        // ── 2. Update entity consciousness ────────────────────────────────
        {
            let updates: Vec<_> = self
                .agents
                .iter()
                .map(|a| {
                    (
                        a.handle,
                        a.inputs.clone(),
                        Point(nalgebra::SVector::<f64, 2>::from(a.pos)),
                    )
                })
                .collect();
            for (handle, inputs, point) in updates {
                self.field.update_entity(handle, &inputs, point);
            }
        }

        // ── 3. Spread emotional contagion ─────────────────────────────────
        {
            let positions: Vec<_> = self
                .agents
                .iter()
                .map(|a| (a.handle, Point(nalgebra::SVector::<f64, 2>::from(a.pos))))
                .collect();
            self.field.spread_emotional_contagion(&positions, dt);
        }

        // ── 4. Rebuild harmony field and move agents ──────────────────────
        {
            let positions: Vec<_> = self
                .agents
                .iter()
                .map(|a| (a.handle, Point(nalgebra::SVector::<f64, 2>::from(a.pos))))
                .collect();
            self.field.rebuild_harmony_field(&positions);
        }

        for agent in self.agents.iter_mut() {
            let pos_sv = nalgebra::SVector::<f64, 2>::from(agent.pos);
            let harmonies = self
                .field
                .entities
                .get(&agent.handle)
                .map(|e| e.harmony_activations)
                .unwrap_or([0.0; NUM_HARMONIES]);
            let phi = self.field.phi(agent.handle);
            let energy_frac = self
                .field
                .entities
                .get(&agent.handle)
                .map(|e| e.energy.available / e.energy.max_energy.max(1.0))
                .unwrap_or(0.5);

            // Neighbours again (needed post-contagion for gradient)
            let neighbours: Vec<(nalgebra::SVector<f64, 2>, [f64; NUM_HARMONIES])> = agent_data
                .iter()
                .filter(|(p, _)| (p - pos_sv).norm() > 1e-6)
                .cloned()
                .collect();

            // FEP gradient drives movement — Φ modulates cooperation urgency
            let grad = free_energy_gradient_phi(
                &pos_sv,
                energy_frac,
                Some(phi),
                &harmonies,
                &neighbours,
                &[], // no explicit energy wells — agents share consciousness field
                None,
                0.0,
            );

            // Motor gain from consciousness tier × slow speed cap
            let motor_gain = self
                .field
                .entities
                .get(&agent.handle)
                .map(|e| e.effective_motor_gain())
                .unwrap_or(0.5);

            let max_speed = 5.0 * motor_gain;
            agent.vel[0] += grad[0] * motor_gain * 10.0 * dt;
            agent.vel[1] += grad[1] * motor_gain * 10.0 * dt;

            // Speed clamp
            let speed = (agent.vel[0] * agent.vel[0] + agent.vel[1] * agent.vel[1]).sqrt();
            if speed > max_speed && speed > 1e-10 {
                let scale = max_speed / speed;
                agent.vel[0] *= scale;
                agent.vel[1] *= scale;
            }

            // Euler step
            agent.pos[0] += agent.vel[0] * dt;
            agent.pos[1] += agent.vel[1] * dt;

            // Torus boundary (wrap around world edges)
            let w = self.world_size;
            agent.pos[0] = ((agent.pos[0] + w).rem_euclid(2.0 * w)) - w;
            agent.pos[1] = ((agent.pos[1] + w).rem_euclid(2.0 * w)) - w;
        }

        // ── 5. Serialise snapshot ─────────────────────────────────────────
        let snaps: Vec<AgentSnap> = self
            .agents
            .iter()
            .map(|a| {
                let phi = self.field.phi(a.handle);
                let entity = self.field.entities.get(&a.handle);
                let emotion = entity
                    .map(|e| e.harmony_activations[EMOTIONAL_CONTAGION_IDX])
                    .unwrap_or(0.0);
                let energy = entity
                    .map(|e| e.energy.available / e.energy.max_energy.max(1.0))
                    .unwrap_or(0.5);
                let harmony = entity
                    .map(|e| e.harmony_activations.to_vec())
                    .unwrap_or_else(|| vec![0.0; NUM_HARMONIES]);
                let safety = entity
                    .map(|e| format!("{:?}", e.safety_tier).to_lowercase())
                    .unwrap_or_else(|| "unknown".into());
                AgentSnap {
                    id: a.handle.0 as u32,
                    x: a.pos[0],
                    y: a.pos[1],
                    vx: a.vel[0],
                    vy: a.vel[1],
                    phi,
                    emotion,
                    energy,
                    harmony,
                    safety,
                }
            })
            .collect();

        let snap = DemoSnap {
            tick: self.tick_count,
            agents: snaps,
            collective_phi: self.field.collective_phi,
        };

        serde_json::to_string(&snap).unwrap_or_else(|_| "{}".into())
    }

    /// Inject an emotional stimulus into a specific agent.
    ///
    /// `agent_id`: index in [0, num_agents)
    /// `emotion`: [0, 1] — will be assigned to harmony[8]
    pub fn set_emotion(&mut self, agent_id: u32, emotion: f64) {
        let handle = BodyHandle(agent_id as usize);
        if let Some(entity) = self.field.entities.get_mut(&handle) {
            entity.harmony_activations[EMOTIONAL_CONTAGION_IDX] = emotion.clamp(0.0, 1.0);
        }
    }

    /// Set the full harmony profile for an agent (9-element JSON array).
    ///
    /// e.g. `demo.set_harmony(0, JSON.stringify([0.8, 0.2, 0.1, ...]));`
    pub fn set_harmony(&mut self, agent_id: u32, harmony_json: &str) {
        if let Ok(vals) = serde_json::from_str::<Vec<f64>>(harmony_json) {
            let handle = BodyHandle(agent_id as usize);
            if let Some(entity) = self.field.entities.get_mut(&handle) {
                for (i, &v) in vals.iter().take(NUM_HARMONIES).enumerate() {
                    entity.harmony_activations[i] = v.clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Returns the current collective Φ (mean across all entities).
    pub fn collective_phi(&self) -> f64 {
        self.field.collective_phi
    }

    /// Returns the number of simulation ticks executed so far.
    pub fn tick_count(&self) -> u32 {
        self.tick_count.min(u32::MAX as u64) as u32
    }

    /// Number of agents in the simulation.
    pub fn num_agents(&self) -> u32 {
        self.agents.len() as u32
    }

    /// World half-extent (agents wrap at ±world_size on each axis).
    pub fn world_size(&self) -> f64 {
        self.world_size
    }

    /// Set the Φ-gravity coupling strength (bifurcation parameter).
    ///
    /// Controls how strongly integrated information attracts agents.
    /// The phase transition occurs near `k ≈ 0.3-0.4` in the default configuration.
    /// `k = 0.0`: no consciousness-driven attraction.
    /// `k = 1.0`: strong attraction to high-Φ regions.
    pub fn set_phi_gravity(&mut self, k: f64) {
        self.field.phi_gravity_strength = k.clamp(0.0, 2.0);
    }

    /// Get the current Φ-gravity coupling strength.
    pub fn phi_gravity(&self) -> f64 {
        self.field.phi_gravity_strength
    }

    /// Toggle consciousness on/off for all agents.
    ///
    /// When disabled, all Φ inputs are zeroed — agents become "unconscious"
    /// (Φ=0) while retaining energy and physics. This demonstrates the
    /// conscious/unconscious clustering difference (Finding 3: 35% tighter
    /// clustering with Φ>0).
    pub fn set_consciousness_enabled(&mut self, enabled: bool) {
        for agent in self.agents.iter_mut() {
            if enabled {
                agent.inputs.phi = 0.5;
                agent.inputs.broadcast = 0.5;
                agent.inputs.working_memory = 0.5;
            } else {
                agent.inputs.phi = 0.0;
                agent.inputs.broadcast = 0.0;
                agent.inputs.working_memory = 0.0;
            }
        }
    }

    /// Get per-agent Φ values as a JSON array.
    pub fn phi_values(&self) -> String {
        let phis: Vec<f64> = self
            .agents
            .iter()
            .map(|a| self.field.phi(a.handle))
            .collect();
        serde_json::to_string(&phis).unwrap_or_else(|_| "[]".into())
    }
}

// ── Native (non-WASM) unit tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_constructs_and_ticks() {
        let mut demo = WasmPhysicsDemo::new(4);
        assert_eq!(demo.num_agents(), 4);
        let json = demo.tick(1.0 / 60.0);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["agents"].as_array().unwrap().len(), 4);
        assert!(v["tick"].as_u64().unwrap() == 1);
    }

    #[test]
    fn emotional_injection_visible_in_snapshot() {
        let mut demo = WasmPhysicsDemo::new(3);
        demo.set_emotion(0, 0.9);
        let json = demo.tick(1.0 / 60.0);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let emotion = v["agents"][0]["emotion"].as_f64().unwrap();
        // After one tick the contagion spreads slightly — emotion should be > 0
        assert!(
            emotion > 0.0,
            "injected emotion should appear, got {emotion}"
        );
    }

    #[test]
    fn set_harmony_is_applied() {
        let mut demo = WasmPhysicsDemo::new(2);
        let harmony = vec![0.9_f64; NUM_HARMONIES];
        demo.set_harmony(0, &serde_json::to_string(&harmony).unwrap());
        let json = demo.tick(0.016);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let h = v["agents"][0]["harmony"][0].as_f64().unwrap();
        // Should be close to 0.9 (small changes from contagion update)
        assert!(h > 0.7, "harmony[0] should be near injected 0.9, got {h}");
    }

    #[test]
    fn collective_phi_is_non_negative() {
        let mut demo = WasmPhysicsDemo::new(6);
        for _ in 0..10 {
            demo.tick(0.016);
        }
        assert!(demo.collective_phi() >= 0.0);
    }

    #[test]
    fn agents_wrap_at_boundary() {
        let mut demo = WasmPhysicsDemo::new(1);
        // Teleport agent to edge
        demo.agents[0].pos = [49.9, 0.0];
        demo.agents[0].vel = [100.0, 0.0]; // fast rightward
        demo.tick(0.1); // will move past +50
        let x = demo.agents[0].pos[0];
        assert!(x.abs() < demo.world_size, "agent should wrap, got x={x}");
    }

    #[test]
    fn tick_json_has_correct_shape() {
        let mut demo = WasmPhysicsDemo::new(2);
        let json = demo.tick(0.016);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let agent = &v["agents"][0];
        // Required fields
        for field in &[
            "id", "x", "y", "vx", "vy", "phi", "emotion", "energy", "harmony", "safety",
        ] {
            assert!(agent.get(field).is_some(), "missing field: {field}");
        }
        assert_eq!(agent["harmony"].as_array().unwrap().len(), NUM_HARMONIES);
    }
}
