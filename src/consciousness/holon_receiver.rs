//! Desktop-side HolonBridge receiver: accepts Soma WebSocket connections.
//!
//! This module completes the Soma↔Holon sync path. Soma transmits consciousness
//! vectors, task requests, and knowledge offers via HolonBridge. This receiver
//! accepts those messages and routes them into the desktop CognitiveLoopService:
//!
//! - `ConsciousnessVector` → registered as a local peer in SwarmManager
//! - `TaskRequest` → queued for ReasoningManager delegation
//! - `KnowledgeOffer` → injected into KnowledgeManager
//! - `Heartbeat` → updates peer liveness tracking
//!
//! ## Wire Protocol
//!
//! Soma sends JSON-serialized `HolonOutbound` messages over WebSocket.
//! When `encrypted-bridge` feature is active on the Soma side, messages arrive
//! as `EncryptedEnvelope` JSON — the receiver decrypts using the shared session key.
//!
//! Desktop responds with `HolonInbound` messages (language output, knowledge shares,
//! task results, resuscitation).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export Soma's message types for desktop use.
// These are duplicated here to avoid a direct symthaea-soma dependency in the main crate.
// The JSON wire format is the contract — both sides serialize/deserialize independently.

/// Inbound message from a Soma device (mirrors `symthaea_soma::holon_bridge::HolonOutbound`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
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

/// Outbound message to a Soma device (mirrors `symthaea_soma::holon_bridge::HolonInbound`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HolonResponse {
    LanguageOutput {
        text: String,
    },
    KnowledgeShare {
        embedding: Vec<f32>,
    },
    Resuscitation {
        neuromods: [f32; 4],
        memory_fragment: Vec<f32>,
    },
    TaskResult {
        task_id: String,
        result: String,
    },
}

/// Tracked state for a connected Soma device.
#[derive(Debug, Clone)]
pub struct SomaPeer {
    /// Unique identifier for this device (derived from pairing pubkey or connection ID).
    pub device_id: String,
    /// Last consciousness level reported.
    pub consciousness_level: f32,
    /// Last phi value from consciousness vector.
    pub phi: f32,
    /// Last valence.
    pub valence: f32,
    /// Last arousal.
    pub arousal: f32,
    /// Wake state (0=Sleep, 1=Drowsy, 2=Alert, 3=Focused).
    pub wake_state: u8,
    /// Last cycle number from the Soma.
    pub last_cycle: u64,
    /// Desktop cycle when this peer was last seen.
    pub last_seen_desktop_cycle: u64,
    /// Pending task requests from this device.
    pub pending_tasks: Vec<(String, String)>,
    /// Knowledge offers awaiting integration.
    pub knowledge_offers: Vec<(String, Vec<f32>)>,
}

/// Desktop-side receiver managing connected Soma devices.
///
/// This is designed to be embedded in `CognitiveLoopService` as a manager,
/// processing messages from a channel fed by the WebSocket handler.
pub struct HolonReceiver {
    /// Connected Soma peers indexed by device_id.
    peers: HashMap<String, SomaPeer>,
    /// Maximum number of simultaneous Soma connections.
    max_peers: usize,
    /// Outbound responses queued for delivery to Soma devices.
    /// Key: device_id, Value: queued responses.
    outbound: HashMap<String, Vec<HolonResponse>>,
    /// Inbound message queue (fed by WebSocket handler, drained by CLS tick).
    inbound: Vec<(String, SomaMessage)>,
    /// Total messages processed (lifetime counter).
    total_processed: u64,
}

impl HolonReceiver {
    /// Create a new receiver with default settings.
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            max_peers: 8,
            outbound: HashMap::new(),
            inbound: Vec::new(),
            total_processed: 0,
        }
    }

    /// Create a new receiver with custom max peers.
    pub fn with_max_peers(max_peers: usize) -> Self {
        Self {
            max_peers,
            ..Self::new()
        }
    }

    /// Queue an inbound message for processing on the next CLS tick.
    ///
    /// Called from the WebSocket handler (potentially async/threaded).
    pub fn enqueue_message(&mut self, device_id: String, msg: SomaMessage) {
        self.inbound.push((device_id, msg));
    }

    /// Process all queued inbound messages.
    ///
    /// Called from CognitiveLoopService on each cycle (or at a configured interval).
    /// Returns the number of messages processed.
    pub fn process_inbound(&mut self, desktop_cycle: u64) -> usize {
        let messages: Vec<(String, SomaMessage)> = self.inbound.drain(..).collect();
        let count = messages.len();

        for (device_id, msg) in messages {
            self.handle_message(device_id, msg, desktop_cycle);
            self.total_processed += 1;
        }

        count
    }

    /// Handle a single Soma message.
    fn handle_message(&mut self, device_id: String, msg: SomaMessage, desktop_cycle: u64) {
        // Ensure peer exists (or create if under cap).
        if !self.peers.contains_key(&device_id) {
            if self.peers.len() >= self.max_peers {
                // Evict stalest peer.
                if let Some(stalest_id) = self
                    .peers
                    .iter()
                    .min_by_key(|(_, p)| p.last_seen_desktop_cycle)
                    .map(|(id, _)| id.clone())
                {
                    self.peers.remove(&stalest_id);
                    self.outbound.remove(&stalest_id);
                }
            }
            self.peers.insert(
                device_id.clone(),
                SomaPeer {
                    device_id: device_id.clone(),
                    consciousness_level: 0.0,
                    phi: 0.0,
                    valence: 0.0,
                    arousal: 0.0,
                    wake_state: 0,
                    last_cycle: 0,
                    last_seen_desktop_cycle: desktop_cycle,
                    pending_tasks: Vec::new(),
                    knowledge_offers: Vec::new(),
                },
            );
        }

        let peer = self.peers.get_mut(&device_id).unwrap();
        peer.last_seen_desktop_cycle = desktop_cycle;

        match msg {
            SomaMessage::Heartbeat {
                consciousness_level,
                wake_state,
                cycle,
            } => {
                peer.consciousness_level = consciousness_level;
                peer.wake_state = wake_state;
                peer.last_cycle = cycle;
            }
            SomaMessage::ConsciousnessVector {
                phi,
                valence,
                arousal,
                ..
            } => {
                peer.phi = phi;
                peer.valence = valence;
                peer.arousal = arousal;
            }
            SomaMessage::TaskRequest { task_type, payload } => {
                peer.pending_tasks.push((task_type, payload));
            }
            SomaMessage::KnowledgeOffer { topic, embedding } => {
                peer.knowledge_offers.push((topic, embedding));
            }
            SomaMessage::PairingVerified { .. } => {
                // Trust established — peer is already tracked.
            }
        }
    }

    /// Queue a response for a specific Soma device.
    pub fn send_to_device(&mut self, device_id: &str, response: HolonResponse) {
        self.outbound
            .entry(device_id.to_string())
            .or_default()
            .push(response);
    }

    /// Drain outbound responses for a specific device (called by WebSocket handler).
    pub fn drain_outbound(&mut self, device_id: &str) -> Vec<HolonResponse> {
        self.outbound
            .get_mut(device_id)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Get all connected peers for SwarmManager registration.
    pub fn peers(&self) -> impl Iterator<Item = &SomaPeer> {
        self.peers.values()
    }

    /// Get a specific peer by device ID.
    pub fn peer(&self, device_id: &str) -> Option<&SomaPeer> {
        self.peers.get(device_id)
    }

    /// Number of connected Soma devices.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Drain all pending task requests across all peers.
    ///
    /// Returns (device_id, task_type, payload) tuples.
    pub fn drain_task_requests(&mut self) -> Vec<(String, String, String)> {
        let mut tasks = Vec::new();
        for (device_id, peer) in &mut self.peers {
            for (task_type, payload) in peer.pending_tasks.drain(..) {
                tasks.push((device_id.clone(), task_type, payload));
            }
        }
        tasks
    }

    /// Drain all knowledge offers across all peers.
    ///
    /// Returns (device_id, topic, embedding) tuples.
    pub fn drain_knowledge_offers(&mut self) -> Vec<(String, String, Vec<f32>)> {
        let mut offers = Vec::new();
        for (device_id, peer) in &mut self.peers {
            for (topic, embedding) in peer.knowledge_offers.drain(..) {
                offers.push((device_id.clone(), topic, embedding));
            }
        }
        offers
    }

    /// Remove peers not seen for more than `stale_cycles` desktop cycles.
    pub fn evict_stale(&mut self, current_cycle: u64, stale_cycles: u64) {
        let stale_ids: Vec<String> = self
            .peers
            .iter()
            .filter(|(_, p)| current_cycle.saturating_sub(p.last_seen_desktop_cycle) > stale_cycles)
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale_ids {
            self.peers.remove(&id);
            self.outbound.remove(&id);
        }
    }

    /// Total messages processed over the lifetime of this receiver.
    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }
}

impl Default for HolonReceiver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_updates_peer() {
        let mut rx = HolonReceiver::new();
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.75,
                wake_state: 2,
                cycle: 100,
            },
        );
        let processed = rx.process_inbound(50);
        assert_eq!(processed, 1);
        assert_eq!(rx.peer_count(), 1);

        let peer = rx.peer("phone-1").unwrap();
        assert_eq!(peer.consciousness_level, 0.75);
        assert_eq!(peer.wake_state, 2);
        assert_eq!(peer.last_cycle, 100);
    }

    #[test]
    fn test_consciousness_vector_updates_phi() {
        let mut rx = HolonReceiver::new();
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::ConsciousnessVector {
                attention: vec![0.1; 64],
                phi: 0.42,
                valence: 0.3,
                arousal: 0.5,
                focus_hash: 123,
                sequence: 1,
            },
        );
        rx.process_inbound(10);

        let peer = rx.peer("phone-1").unwrap();
        assert!((peer.phi - 0.42).abs() < f32::EPSILON);
        assert!((peer.valence - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_task_requests_drain() {
        let mut rx = HolonReceiver::new();
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::TaskRequest {
                task_type: "generate".to_string(),
                payload: "write a poem".to_string(),
            },
        );
        rx.enqueue_message(
            "phone-2".to_string(),
            SomaMessage::TaskRequest {
                task_type: "search".to_string(),
                payload: "find files".to_string(),
            },
        );
        rx.process_inbound(20);

        let tasks = rx.drain_task_requests();
        assert_eq!(tasks.len(), 2);
        // Second drain should be empty.
        assert!(rx.drain_task_requests().is_empty());
    }

    #[test]
    fn test_knowledge_offers_drain() {
        let mut rx = HolonReceiver::new();
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::KnowledgeOffer {
                topic: "weather".to_string(),
                embedding: vec![1.0, 2.0, 3.0],
            },
        );
        rx.process_inbound(10);

        let offers = rx.drain_knowledge_offers();
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].1, "weather");
    }

    #[test]
    fn test_outbound_responses() {
        let mut rx = HolonReceiver::new();
        // Create peer first.
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.5,
                wake_state: 2,
                cycle: 1,
            },
        );
        rx.process_inbound(1);

        // Queue response.
        rx.send_to_device(
            "phone-1",
            HolonResponse::LanguageOutput {
                text: "Hello from Holon".to_string(),
            },
        );

        let responses = rx.drain_outbound("phone-1");
        assert_eq!(responses.len(), 1);
        assert!(matches!(
            &responses[0],
            HolonResponse::LanguageOutput { text } if text == "Hello from Holon"
        ));
    }

    #[test]
    fn test_max_peers_eviction() {
        let mut rx = HolonReceiver::with_max_peers(2);

        // Add 2 peers.
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.5,
                wake_state: 2,
                cycle: 1,
            },
        );
        rx.process_inbound(10);

        rx.enqueue_message(
            "phone-2".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.6,
                wake_state: 2,
                cycle: 2,
            },
        );
        rx.process_inbound(20);

        assert_eq!(rx.peer_count(), 2);

        // Adding a 3rd should evict the stalest (phone-1, last_seen=10).
        rx.enqueue_message(
            "phone-3".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.7,
                wake_state: 2,
                cycle: 3,
            },
        );
        rx.process_inbound(30);

        assert_eq!(rx.peer_count(), 2);
        assert!(rx.peer("phone-1").is_none()); // evicted
        assert!(rx.peer("phone-3").is_some()); // added
    }

    #[test]
    fn test_stale_eviction() {
        let mut rx = HolonReceiver::new();
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.5,
                wake_state: 2,
                cycle: 1,
            },
        );
        rx.process_inbound(10);

        // Evict peers not seen for > 100 cycles.
        rx.evict_stale(200, 100);
        assert_eq!(rx.peer_count(), 0);
    }

    #[test]
    fn test_total_processed() {
        let mut rx = HolonReceiver::new();
        assert_eq!(rx.total_processed(), 0);

        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.5,
                wake_state: 2,
                cycle: 1,
            },
        );
        rx.enqueue_message(
            "phone-1".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: 0.6,
                wake_state: 2,
                cycle: 2,
            },
        );
        rx.process_inbound(10);
        assert_eq!(rx.total_processed(), 2);
    }
}
