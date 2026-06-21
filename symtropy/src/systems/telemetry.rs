// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Telemetry Hydration Loop system.
//!
//! Queries active entities (Player, Crew NPCs, rogue Null Drones, and bioregional infrastructure),
//! extracts their physical TGS impulses, Active Inference variational free energy, and Φ coherence scores,
//! and packs them into the std430-aligned GPU Storage Buffer Object (SSBO) resource mirror.

use crate::components::{ConsciousnessComp, CrewNpc, NullDrone, Player, PowerJunction, WaterPump};
use crate::resources::PhysicsWorldRes;
use bevy::prelude::*;
use std::collections::HashMap;
use symtropy_render_bridge::{
    LinkTelemetryGpu, NodeTelemetryGpu, PhysicsBody, TelemetryBufferResource,
};

/// Hydrate Bevy's telemetry SSBO mirror with live physics, active inference, and system state metrics.
pub fn hydrate_gpu_telemetry_system(
    physics: Res<PhysicsWorldRes>,
    mut telemetry: ResMut<TelemetryBufferResource>,
    time: Res<Time>,
    // Queries for nodes
    player_query: Query<
        (Entity, &Transform, &PhysicsBody, Option<&ConsciousnessComp>),
        With<Player>,
    >,
    npc_query: Query<
        (
            Entity,
            &Transform,
            &PhysicsBody,
            &CrewNpc,
            Option<&ConsciousnessComp>,
        ),
        Without<Player>,
    >,
    drone_query: Query<
        (Entity, &Transform, &PhysicsBody, &NullDrone),
        (Without<Player>, Without<CrewNpc>),
    >,
    pump_query: Query<
        (Entity, &Transform, &PhysicsBody, &WaterPump),
        (Without<Player>, Without<CrewNpc>, Without<NullDrone>),
    >,
    junction_query: Query<
        (Entity, &Transform, &PhysicsBody, &PowerJunction),
        (
            Without<Player>,
            Without<CrewNpc>,
            Without<NullDrone>,
            Without<WaterPump>,
        ),
    >,
) {
    telemetry.clear();
    let mut entity_to_index: HashMap<Entity, u32> = HashMap::new();

    // 1. Populate Player Node
    for (entity, tf, body, consciousness) in &player_query {
        let (prediction_error, phi) = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| (e.prediction_error as f32, e.phi() as f32))
            .unwrap_or((0.0, 0.9));

        let last_impulse = physics
            .world
            .collision_events
            .iter()
            .filter(|e| e.body_a == body.handle || e.body_b == body.handle)
            .map(|e| e.impulse)
            .fold(0.0, f64::max) as f32;

        let node = NodeTelemetryGpu {
            position: tf.translation,
            variational_free_energy: prediction_error,
            bandwidth_bps: 10.0e6, // Stable high bandwidth
            latency_ms: 5.0,
            tunnel_state: 1, // Sovereign ML-KEM Secure
            dht_holding_completeness: 1.0,
            gossip_frequency_hz: 12.0,
            validation_failure_count: 0,
            wasm_memory_fraction: 0.1,
            last_hot_reload_time: 0.0,
            holographic_coherence: phi,
            thermal_gradient: 293.15 + last_impulse * 2.0, // Impulses heat up the body slightly
            circuit_load: 0.05,
            _padding: 0.0,
        };

        let idx = telemetry.nodes.len() as u32;
        telemetry.nodes.push(node);
        entity_to_index.insert(entity, idx);
    }

    // 2. Populate Crew NPC Nodes
    for (entity, tf, body, npc, consciousness) in &npc_query {
        let (prediction_error, phi) = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| (e.prediction_error as f32, e.phi() as f32))
            .unwrap_or((0.0, 0.5));

        let last_impulse = physics
            .world
            .collision_events
            .iter()
            .filter(|e| e.body_a == body.handle || e.body_b == body.handle)
            .map(|e| e.impulse)
            .fold(0.0, f64::max) as f32;

        let sim_phi = consciousness.map(|c| c.sim_phi() as f32).unwrap_or(phi);

        let node = NodeTelemetryGpu {
            position: tf.translation,
            variational_free_energy: prediction_error,
            bandwidth_bps: 5.0e6,
            latency_ms: 12.0,
            tunnel_state: 1, // Sovereign ML-KEM Secure
            dht_holding_completeness: 0.95,
            gossip_frequency_hz: 10.0,
            validation_failure_count: 0,
            wasm_memory_fraction: 0.25,
            last_hot_reload_time: 0.0,
            holographic_coherence: sim_phi,
            thermal_gradient: 293.15 + last_impulse * 3.0,
            circuit_load: 0.08,
            _padding: 0.0,
        };

        let idx = telemetry.nodes.len() as u32;
        telemetry.nodes.push(node);
        entity_to_index.insert(entity, idx);
    }

    // 3. Populate Rogue Null Drone Nodes (Untrusted chaos cloud)
    for (entity, tf, body, drone) in &drone_query {
        let last_impulse = physics
            .world
            .collision_events
            .iter()
            .filter(|e| e.body_a == body.handle || e.body_b == body.handle)
            .map(|e| e.impulse)
            .fold(0.0, f64::max) as f32;

        let node = NodeTelemetryGpu {
            position: tf.translation,
            variational_free_energy: 0.85, // High cognitive noise / surprise
            bandwidth_bps: 100.0,          // Scraping telemetry
            latency_ms: 180.0,             // Unstable link
            tunnel_state: 3,               // Unvetted / Untrusted
            dht_holding_completeness: 0.0,
            gossip_frequency_hz: 1.5,
            validation_failure_count: 4, // High validation failures
            wasm_memory_fraction: 0.9,   // Leaking WASM memory
            last_hot_reload_time: time.elapsed_secs(),
            holographic_coherence: 0.1, // Fragmented consciousness
            thermal_gradient: 310.15 + last_impulse * 5.0, // Running hot
            circuit_load: 0.45,
            _padding: 0.0,
        };

        let idx = telemetry.nodes.len() as u32;
        telemetry.nodes.push(node);
        entity_to_index.insert(entity, idx);
    }

    // 4. Populate Water Pump Nodes
    for (entity, tf, body, pump) in &pump_query {
        let efficiency = pump.efficiency;
        let is_running = pump.is_running;
        let is_sabotaged = pump.is_sabotaged;

        let load = if is_running { 0.4 } else { 0.0 } + if is_sabotaged { 0.3 } else { 0.0 };

        let node = NodeTelemetryGpu {
            position: tf.translation,
            variational_free_energy: if is_sabotaged { 0.9 } else { 0.05 },
            bandwidth_bps: 2.0e6,
            latency_ms: 15.0,
            tunnel_state: if is_sabotaged { 2 } else { 1 }, // Fallback secure if compromised
            dht_holding_completeness: 0.9,
            gossip_frequency_hz: 5.0,
            validation_failure_count: if is_sabotaged { 2 } else { 0 },
            wasm_memory_fraction: 0.2,
            last_hot_reload_time: 0.0,
            holographic_coherence: efficiency,
            thermal_gradient: 288.15 + load * 25.0,
            circuit_load: load,
            _padding: 0.0,
        };

        let idx = telemetry.nodes.len() as u32;
        telemetry.nodes.push(node);
        entity_to_index.insert(entity, idx);
    }

    // 5. Populate Power Junction Nodes
    for (entity, tf, body, junction) in &junction_query {
        let is_damaged = junction.is_damaged;
        let output = junction.output;

        let node = NodeTelemetryGpu {
            position: tf.translation,
            variational_free_energy: if is_damaged { 0.8 } else { 0.1 },
            bandwidth_bps: 4.0e6,
            latency_ms: 10.0,
            tunnel_state: if is_damaged { 2 } else { 1 },
            dht_holding_completeness: 0.95,
            gossip_frequency_hz: 6.0,
            validation_failure_count: if is_damaged { 1 } else { 0 },
            wasm_memory_fraction: 0.15,
            last_hot_reload_time: 0.0,
            holographic_coherence: output,
            thermal_gradient: 290.15 + output * 30.0 + if is_damaged { 15.0 } else { 0.0 },
            circuit_load: output,
            _padding: 0.0,
        };

        let idx = telemetry.nodes.len() as u32;
        telemetry.nodes.push(node);
        entity_to_index.insert(entity, idx);
    }

    // 6. Generate Link Topologies
    // Link player to all crew NPCs (Secure, blue gossip links)
    let player_entities: Vec<Entity> = player_query.iter().map(|(e, _, _, _)| e).collect();
    let npc_entities: Vec<Entity> = npc_query.iter().map(|(e, _, _, _, _)| e).collect();

    for &pe in &player_entities {
        let Some(&p_idx) = entity_to_index.get(&pe) else {
            continue;
        };

        for &ne in &npc_entities {
            let Some(&n_idx) = entity_to_index.get(&ne) else {
                continue;
            };

            telemetry.links.push(LinkTelemetryGpu {
                source_node_idx: p_idx,
                target_node_idx: n_idx,
                link_thickness: 2.0,
                particle_velocity: 15.0,
                link_color: Vec4::new(0.0, 0.7, 1.0, 0.8), // Ice-blue Hybrid Secure link
            });
        }
    }

    // Link drones to their targets if actively sabotaging (Amber/crimson unvetted links)
    for (drone_entity, _, _, drone) in &drone_query {
        let Some(&d_idx) = entity_to_index.get(&drone_entity) else {
            continue;
        };

        if let Some(target_machine) = drone.target_machine {
            if let Some(&t_idx) = entity_to_index.get(&target_machine) {
                telemetry.links.push(LinkTelemetryGpu {
                    source_node_idx: d_idx,
                    target_node_idx: t_idx,
                    link_thickness: 3.5,
                    particle_velocity: 25.0,
                    link_color: Vec4::new(1.0, 0.2, 0.0, 0.9), // Glowing crimson warning link
                });
            }
        }
    }
}

use bevy::render::storage::ShaderStorageBuffer;

/// GPU Storage Buffer handle for telemetry data.
#[derive(Resource)]
pub struct TelemetryGpuBuffer {
    pub nodes_buffer: Handle<ShaderStorageBuffer>,
}

/// Initialize the GPU Storage Buffer asset and resource.
pub fn setup_telemetry_gpu_buffer(
    mut commands: Commands,
    storage_buffers: Option<ResMut<Assets<ShaderStorageBuffer>>>,
) {
    let Some(mut storage_buffers) = storage_buffers else {
        return;
    };
    let nodes_buffer = storage_buffers.add(ShaderStorageBuffer::new(
        &[], // Starts empty
        bevy::asset::RenderAssetUsages::default(),
    ));
    commands.insert_resource(TelemetryGpuBuffer { nodes_buffer });
}

/// Sync the CPU-side telemetry buffer to the GPU storage buffer asset.
pub fn sync_telemetry_to_gpu_system(
    telemetry: Res<TelemetryBufferResource>,
    gpu_buffer: Option<Res<TelemetryGpuBuffer>>,
    storage_buffers: Option<ResMut<Assets<ShaderStorageBuffer>>>,
) {
    let (Some(gpu_buffer), Some(mut storage_buffers)) = (gpu_buffer, storage_buffers) else {
        return;
    };
    if let Some(buffer) = storage_buffers.get_mut(&gpu_buffer.nodes_buffer) {
        buffer.set_data(&telemetry.nodes);
    }
}
