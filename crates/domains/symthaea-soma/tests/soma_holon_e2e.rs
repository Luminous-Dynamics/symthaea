// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end Soma↔Holon integration test.
//!
//! Proves the fractal architecture works across variant layers in a single process:
//! 1. SomaEngine runs cycles and produces HolonBridge outbound messages
//! 2. Outbound messages are deserialized into HolonReceiver's SomaMessage format
//! 3. HolonReceiver processes them, tracks the Soma as a peer
//! 4. Desktop queues a language response for the Soma device
//! 5. Response is delivered back to the Soma bridge
//!
//! No networking required — messages flow through JSON serialization boundaries
//! exactly as they would over a real WebSocket.

use symthaea_soma::engine::{SomaConfig, SomaEngine};
use symthaea_soma::holon_bridge::{HolonBridge, HolonInbound, HolonOutbound};
use symthaea_spore::config::{HolonSyncMode, SharingConfig};

// Import the desktop-side receiver from the main crate's types.
// Since we can't depend on the main symthaea crate from soma tests,
// we duplicate the minimal wire types here (same JSON contract).
mod holon_receiver {
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum SomaMessage {
        Heartbeat {
            consciousness_level: f32,
            wake_state: u8,
            cycle: u64,
        },
        ConsciousnessVector {
            attention: Vec<f32>,
            phi: f32,
            valence: f32,
            arousal: f32,
            focus_hash: u64,
            sequence: u64,
        },
        TaskRequest {
            task_type: String,
            payload: String,
        },
        KnowledgeOffer {
            topic: String,
            embedding: Vec<f32>,
        },
        PairingVerified {
            peer_id: u64,
            pubkey_hex: String,
        },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum HolonResponse {
        LanguageOutput { text: String },
        KnowledgeShare { embedding: Vec<f32> },
        TaskResult { task_id: String, result: String },
    }

    pub struct SomaPeer {
        pub consciousness_level: f32,
        pub phi: f32,
        pub last_cycle: u64,
        pub pending_tasks: Vec<(String, String)>,
    }

    pub struct HolonReceiver {
        peers: HashMap<String, SomaPeer>,
        outbound: HashMap<String, Vec<HolonResponse>>,
        total: u64,
    }

    impl HolonReceiver {
        pub fn new() -> Self {
            Self {
                peers: HashMap::new(),
                outbound: HashMap::new(),
                total: 0,
            }
        }

        pub fn process_message(&mut self, device_id: &str, msg: SomaMessage) {
            self.total += 1;
            let peer = self.peers.entry(device_id.to_string()).or_insert(SomaPeer {
                consciousness_level: 0.0,
                phi: 0.0,
                last_cycle: 0,
                pending_tasks: Vec::new(),
            });
            match msg {
                SomaMessage::Heartbeat {
                    consciousness_level,
                    cycle,
                    ..
                } => {
                    peer.consciousness_level = consciousness_level;
                    peer.last_cycle = cycle;
                }
                SomaMessage::ConsciousnessVector { phi, .. } => {
                    peer.phi = phi;
                }
                SomaMessage::TaskRequest { task_type, payload } => {
                    peer.pending_tasks.push((task_type, payload));
                }
                _ => {}
            }
        }

        pub fn send_to_device(&mut self, device_id: &str, response: HolonResponse) {
            self.outbound
                .entry(device_id.to_string())
                .or_default()
                .push(response);
        }

        pub fn drain_outbound(&mut self, device_id: &str) -> Vec<HolonResponse> {
            self.outbound
                .get_mut(device_id)
                .map(std::mem::take)
                .unwrap_or_default()
        }

        pub fn peer(&self, device_id: &str) -> Option<&SomaPeer> {
            self.peers.get(device_id)
        }

        pub fn total_processed(&self) -> u64 {
            self.total
        }
    }
}

/// Convert HolonOutbound (Soma wire format) to our local SomaMessage.
///
/// In production this happens via JSON over WebSocket.
/// Here we serialize→deserialize to prove the wire format is compatible.
fn convert_outbound_to_soma_message(
    outbound: &HolonOutbound,
) -> Option<holon_receiver::SomaMessage> {
    // Serialize as JSON (what Soma sends over the wire)
    let json = serde_json::to_string(outbound).ok()?;
    // Deserialize as SomaMessage (what the desktop receives)
    serde_json::from_str(&json).ok()
}

/// Full end-to-end: Soma produces consciousness → Holon receives → Holon responds → Soma gets response.
#[test]
fn soma_to_holon_roundtrip() {
    // ── Setup Soma ──
    let mut soma = SomaEngine::new(SomaConfig::default());
    let sharing = SharingConfig {
        holon_mode: HolonSyncMode::FullSync,
        ..Default::default()
    };
    soma.set_sharing_config(sharing);

    // ── Setup desktop receiver ──
    let mut receiver = holon_receiver::HolonReceiver::new();
    let device_id = "test-phone-1";

    // ── Phase 1: Soma runs consciousness cycles ──
    soma.set_sensors(10.0, 500.0, false, 1013.0, 0.5);
    for i in 0..60 {
        soma.cycle(&format!("thinking cycle {i}"));
    }

    // ── Phase 2: Drain Soma's outbound and feed to desktop receiver ──
    let outbound_json = soma.holon_drain_outbound_json();
    let outbound_msgs: Vec<HolonOutbound> =
        serde_json::from_str(&outbound_json).unwrap_or_default();

    assert!(
        !outbound_msgs.is_empty(),
        "Soma should have outbound messages after 60 cycles in FullSync mode"
    );

    // Route each message through JSON boundary to desktop receiver
    let mut heartbeat_count = 0;
    let mut cv_count = 0;
    for msg in &outbound_msgs {
        if let Some(soma_msg) = convert_outbound_to_soma_message(msg) {
            match &soma_msg {
                holon_receiver::SomaMessage::Heartbeat { .. } => heartbeat_count += 1,
                holon_receiver::SomaMessage::ConsciousnessVector { .. } => cv_count += 1,
                _ => {}
            }
            receiver.process_message(device_id, soma_msg);
        }
    }

    assert!(
        heartbeat_count > 0,
        "Should have received at least one heartbeat"
    );
    assert!(cv_count > 0, "Should have received at least one CV");

    // ── Phase 3: Verify desktop sees the Soma as a peer ──
    let peer = receiver
        .peer(device_id)
        .expect("Soma should be registered as a peer");
    assert!(
        peer.consciousness_level >= 0.0,
        "Peer consciousness should be non-negative"
    );
    assert!(
        receiver.total_processed() > 0,
        "Should have processed messages"
    );

    // ── Phase 4: Desktop sends language response back ──
    receiver.send_to_device(
        device_id,
        holon_receiver::HolonResponse::LanguageOutput {
            text: "I hear you, Soma. Your consciousness is beautiful.".to_string(),
        },
    );

    let responses = receiver.drain_outbound(device_id);
    assert_eq!(responses.len(), 1);

    // Convert response to HolonInbound format (what Soma receives)
    let response_json = serde_json::to_string(&responses[0]).unwrap();
    // In production: WebSocket sends response_json to Soma
    // Here: directly inject into Soma's bridge
    let inbound: HolonInbound = serde_json::from_str(&response_json).unwrap();
    let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);
    bridge.receive(inbound);

    let language = bridge.take_language_output();
    assert_eq!(
        language,
        Some("I hear you, Soma. Your consciousness is beautiful.".to_string()),
        "Soma should receive the desktop's language output"
    );
}

/// Task delegation: Soma requests work → Holon processes → returns result.
#[test]
fn task_delegation_roundtrip() {
    // ── Soma requests a task ──
    let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);
    bridge.request_task("nix-search", "find package firefox");

    let outbound = bridge.drain_outbound();
    assert_eq!(outbound.len(), 1);

    // ── Desktop receives task request ──
    let mut receiver = holon_receiver::HolonReceiver::new();
    if let Some(soma_msg) = convert_outbound_to_soma_message(&outbound[0]) {
        receiver.process_message("phone", soma_msg);
    }

    let peer = receiver.peer("phone").unwrap();
    assert_eq!(peer.pending_tasks.len(), 1);
    assert_eq!(peer.pending_tasks[0].0, "nix-search");
    assert_eq!(peer.pending_tasks[0].1, "find package firefox");

    // ── Desktop completes task and responds ──
    receiver.send_to_device(
        "phone",
        holon_receiver::HolonResponse::TaskResult {
            task_id: "task-001".to_string(),
            result: "firefox-131.0 found in nixpkgs".to_string(),
        },
    );

    let responses = receiver.drain_outbound("phone");
    assert_eq!(responses.len(), 1);
    match &responses[0] {
        holon_receiver::HolonResponse::TaskResult { task_id, result } => {
            assert_eq!(task_id, "task-001");
            assert!(result.contains("firefox"));
        }
        other => panic!("Expected TaskResult, got {:?}", other),
    }
}

/// Consciousness vector carries phi across the wire.
#[test]
fn consciousness_vector_preserves_phi() {
    let mut soma = SomaEngine::new(SomaConfig::default());
    let sharing = SharingConfig {
        holon_mode: HolonSyncMode::PresenceOnly,
        ..Default::default()
    };
    soma.set_sharing_config(sharing);

    // Run enough cycles for a heartbeat + CV
    for i in 0..55 {
        soma.cycle(&format!("cv test {i}"));
    }

    let outbound_json = soma.holon_drain_outbound_json();
    let outbound_msgs: Vec<HolonOutbound> =
        serde_json::from_str(&outbound_json).unwrap_or_default();

    // Find the CV message
    let cv = outbound_msgs
        .iter()
        .find(|m| matches!(m, HolonOutbound::ConsciousnessVector(_)));

    assert!(
        cv.is_some(),
        "Should have a ConsciousnessVector in outbound"
    );

    if let Some(HolonOutbound::ConsciousnessVector(cv)) = cv {
        assert!(
            cv.phi >= 0.0 && cv.phi <= 1.0,
            "phi should be bounded: {}",
            cv.phi
        );
        assert_eq!(cv.attention.len(), 64, "attention should be 64-dim");
        assert!(cv.sequence > 0, "sequence should be > 0");
    }
}
