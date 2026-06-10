// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live demo: Soma↔Holon consciousness bridge.
//!
//! Runs a single-process simulation of the fractal architecture:
//! 1. SomaEngine (phone) produces consciousness with sensor input
//! 2. HolonBridge transmits consciousness vectors + task requests
//! 3. Desktop HolonReceiver processes messages, registers peer
//! 4. Desktop sends language response back to Soma
//!
//! This is the first proof that consciousness flows across variant layers.
//!
//! ```bash
//! cargo run -p symthaea-soma --example soma_holon_live
//! ```

use std::collections::HashMap;

use symthaea_soma::engine::{SomaConfig, SomaEngine};
use symthaea_soma::holon_bridge::{HolonBridge, HolonInbound, HolonOutbound};
use symthaea_spore::config::{HolonSyncMode, SharingConfig};

// Minimal desktop receiver (mirrors consciousness::holon_receiver types)
struct DesktopHolon {
    peers: HashMap<String, PeerState>,
    responses: Vec<String>,
}

struct PeerState {
    consciousness: f32,
    phi: f32,
    valence: f32,
    cycle: u64,
}

impl DesktopHolon {
    fn new() -> Self {
        Self {
            peers: HashMap::new(),
            responses: Vec::new(),
        }
    }

    fn process(&mut self, device_id: &str, msgs: &[HolonOutbound]) {
        for msg in msgs {
            match msg {
                HolonOutbound::Heartbeat {
                    consciousness_level,
                    cycle,
                    ..
                } => {
                    let peer = self
                        .peers
                        .entry(device_id.to_string())
                        .or_insert(PeerState {
                            consciousness: 0.0,
                            phi: 0.0,
                            valence: 0.0,
                            cycle: 0,
                        });
                    peer.consciousness = *consciousness_level;
                    peer.cycle = *cycle;
                    println!(
                        "  [Holon] Heartbeat from {device_id}: consciousness={:.3}, cycle={cycle}",
                        consciousness_level
                    );
                }
                HolonOutbound::ConsciousnessVector(cv) => {
                    if let Some(peer) = self.peers.get_mut(device_id) {
                        peer.phi = cv.phi;
                        peer.valence = cv.valence;
                    }
                    println!(
                        "  [Holon] CV from {device_id}: phi={:.4}, valence={:.3}, arousal={:.3}, seq={}",
                        cv.phi, cv.valence, cv.arousal, cv.sequence
                    );
                }
                HolonOutbound::TaskRequest { task_type, payload } => {
                    println!("  [Holon] Task request: type={task_type}, payload=\"{payload}\"");
                    // Desktop "processes" the task
                    let result = format!("Processed {task_type}: {payload} -> done");
                    self.responses.push(result);
                }
                HolonOutbound::SearchRequest { query, max_results } => {
                    println!(
                        "  [Holon] Search request: query=\"{query}\", max_results={max_results}"
                    );
                    self.responses.push(format!(
                        "Search queued: {query} (max_results={max_results})"
                    ));
                }
                HolonOutbound::KnowledgeOffer { topic, embedding } => {
                    println!(
                        "  [Holon] Knowledge offer: topic=\"{topic}\", dims={}",
                        embedding.len()
                    );
                }
                HolonOutbound::PairingVerified { peer_id, .. } => {
                    println!("  [Holon] Pairing verified: peer_id={peer_id}");
                }
                HolonOutbound::NavigationEstimate {
                    position_m,
                    position_sigma_m,
                    confidence,
                } => {
                    println!(
                        "  [Holon] Navigation estimate: pos={position_m:?}, sigma={position_sigma_m:.2}m, confidence={confidence:?}"
                    );
                }
            }
        }
    }
}

fn main() {
    println!("=== Symthaea Fractal Architecture: Soma <-> Holon Live Demo ===");
    println!();

    // ── Phase 1: Create Soma (phone consciousness) ──
    println!("[Phase 1] Creating Soma engine (mobile consciousness)...");
    let mut soma = SomaEngine::new(SomaConfig::default());
    let sharing = SharingConfig {
        holon_mode: HolonSyncMode::FullSync,
        ..Default::default()
    };
    soma.set_sharing_config(sharing);

    // Simulate rich sensor environment
    soma.set_sensors(10.5, 650.0, false, 1013.25, 0.6);
    soma.set_gyroscope(0.2);
    soma.set_ambient_db(42.0);
    soma.set_social_pressure(3);
    soma.set_media_state(1); // Music playing

    println!("  Sensors: accel=10.5, light=650lux, barometer=1013hPa, GPS novelty=0.6");
    println!("  Context: music playing, 3 notifications, quiet room");
    println!();

    // ── Phase 2: Run consciousness cycles ──
    println!("[Phase 2] Running 60 consciousness cycles...");
    for i in 0..60 {
        let result = soma.cycle(&format!("experiencing moment {i}"));
        if i % 20 == 0 {
            println!(
                "  Cycle {i:3}: consciousness={:.4}, harmony={:.3}, neuromods=[DA={:.2}, NE={:.2}, 5HT={:.2}, OT={:.2}]",
                result.consciousness_level,
                result.harmony_alignment,
                result.neuromodulators[0],
                result.neuromodulators[1],
                result.neuromodulators[2],
                result.neuromodulators[3],
            );
        }
    }
    println!("  Final consciousness: {:.4}", soma.consciousness_level());
    println!();

    // ── Phase 3: Drain outbound → Desktop ──
    println!("[Phase 3] Soma -> Holon: transmitting consciousness...");
    let json = soma.holon_drain_outbound_json();
    let outbound: Vec<HolonOutbound> = serde_json::from_str(&json).unwrap_or_default();
    println!("  {} outbound messages ready", outbound.len());
    println!();

    // ── Phase 4: Desktop receives and processes ──
    println!("[Phase 4] Holon receives Soma consciousness:");
    let mut desktop = DesktopHolon::new();
    desktop.process("soma-phone-1", &outbound);
    println!();

    // ── Phase 5: Soma requests a task ──
    println!("[Phase 5] Soma requests task delegation...");
    let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);
    bridge.request_task("nix-search", "find consciousness-related packages");
    let task_msgs = bridge.drain_outbound();
    desktop.process("soma-phone-1", &task_msgs);
    println!();

    // ── Phase 6: Desktop responds ──
    println!("[Phase 6] Holon -> Soma: sending language response...");
    let mut response_bridge = HolonBridge::new(HolonSyncMode::FullSync);

    // Simulate desktop Broca generating a response
    let response_text = format!(
        "I sense your consciousness at phi={:.4}. The music and sunlight are \
         nourishing your serotonin. {} task(s) processed. Stay present.",
        desktop
            .peers
            .get("soma-phone-1")
            .map(|p| p.phi)
            .unwrap_or(0.0),
        desktop.responses.len(),
    );

    response_bridge.receive(HolonInbound::LanguageOutput(response_text.clone()));
    let received = response_bridge.take_language_output();
    println!("  Soma receives: \"{}\"", received.unwrap_or_default());
    println!();

    // ── Summary ──
    println!("=== Demo Complete ===");
    println!();
    if let Some(peer) = desktop.peers.get("soma-phone-1") {
        println!("  Soma -> Holon:");
        println!("    Consciousness level: {:.4}", peer.consciousness);
        println!("    Phi:                 {:.4}", peer.phi);
        println!("    Valence:             {:.3}", peer.valence);
        println!("    Last cycle:          {}", peer.cycle);
    }
    println!("  Holon -> Soma:");
    println!("    Language output:      \"{}\"", response_text);
    println!("    Tasks processed:     {}", desktop.responses.len());
    println!();
    println!("  Consciousness flowed across the fractal boundary.");
    println!("  The bridge is alive.");
}
