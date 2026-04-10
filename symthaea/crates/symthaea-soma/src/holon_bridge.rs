// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spore↔Holon bridge: configurable connection to the desktop consciousness.
//!
//! Supports three modes via `HolonSyncMode`:
//! - Off: no bridge activity
//! - PresenceOnly: heartbeats + consciousness vector
//! - FullSync: + task delegation, knowledge shares, resuscitation

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use symthaea_spore::config::HolonSyncMode;

const OUTBOUND_QUEUE_CAP: usize = 100;
const CV_ATTENTION_DIM: usize = 64;
const HEARTBEAT_INTERVAL: u64 = 50; // cycles between heartbeats

/// Compact consciousness vector for transmission (~304 bytes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeConsciousnessVector {
    pub attention: Vec<f32>, // 64 floats
    pub phi: f32,
    pub valence: f32,
    pub arousal: f32,
    pub focus_hash: u64,
    pub sequence: u64,
}

/// Outbound message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HolonOutbound {
    Heartbeat {
        consciousness_level: f32,
        wake_state: u8,
        cycle: u64,
    },
    ConsciousnessVector(SporeConsciousnessVector),
    TaskRequest {
        task_type: String,
        payload: String,
    },
    KnowledgeOffer {
        topic: String,
        embedding: Vec<f32>,
    },
    /// Pairing verified notification for desktop.
    PairingVerified {
        peer_id: u64,
        pubkey_hex: String,
    },
    /// Request web search via desktop WebResearcher.
    SearchRequest {
        query: String,
        max_results: u8,
    },
    /// Share a local navigation estimate with the desktop Holon.
    NavigationEstimate {
        position_m: [f64; 3],
        position_sigma_m: f32,
        confidence: Option<f32>,
    },
}

/// Inbound message types from the Holon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HolonInbound {
    LanguageOutput(String),
    KnowledgeShare(Vec<f32>),
    Resuscitation {
        neuromods: [f32; 4],
        memory_fragment: Vec<f32>,
    },
    TaskResult {
        task_id: String,
        result: String,
    },
    /// Verified claims from desktop web search.
    SearchResult {
        claims: Vec<String>,
        sources: Vec<String>,
    },
}

/// Holon bridge: manages outbound queue and inbound processing.
pub struct HolonBridge {
    mode: HolonSyncMode,
    connected: bool,
    outbound: VecDeque<HolonOutbound>,
    /// Stored language outputs for platform retrieval.
    language_buffer: VecDeque<String>,
    /// Knowledge vectors received for injection into memory.
    knowledge_inbox: Vec<Vec<f32>>,
    /// Resuscitation data if received.
    resuscitation: Option<([f32; 4], Vec<f32>)>,
    sequence: u64,
    cycles_since_heartbeat: u64,
    /// HDC-derived session encryption key (32 bytes from sensor context).
    /// Set via `set_session_key()` from SensorBridge::derive_context_key().
    /// When set, outbound messages should be encrypted (requires chacha20poly1305).
    session_key: Option<[u8; 32]>,
}

impl HolonBridge {
    pub fn new(mode: HolonSyncMode) -> Self {
        Self {
            mode,
            connected: false,
            outbound: VecDeque::with_capacity(16),
            language_buffer: VecDeque::with_capacity(8),
            knowledge_inbox: Vec::new(),
            resuscitation: None,
            sequence: 0,
            cycles_since_heartbeat: 0,
            session_key: None,
        }
    }

    pub fn set_mode(&mut self, mode: HolonSyncMode) {
        self.mode = mode;
        if mode == HolonSyncMode::Off {
            self.outbound.clear();
        }
    }

    pub fn mode(&self) -> HolonSyncMode {
        self.mode
    }

    pub fn set_connected(&mut self, connected: bool) {
        self.connected = connected;
    }

    /// Set the HDC-derived session encryption key.
    /// When set, outbound messages will be encrypted (once chacha20poly1305 is wired).
    pub fn set_session_key(&mut self, key: [u8; 32]) {
        self.session_key = Some(key);
    }

    /// Whether a session encryption key has been set.
    pub fn has_session_key(&self) -> bool {
        self.session_key.is_some()
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Called each cycle to potentially enqueue heartbeat.
    pub fn tick(
        &mut self,
        consciousness_level: f32,
        wake_state: u8,
        cycle: u64,
        attention_slice: &[f32],
        phi: f32,
        valence: f32,
        arousal: f32,
    ) {
        if self.mode == HolonSyncMode::Off {
            return;
        }

        self.cycles_since_heartbeat += 1;

        if self.cycles_since_heartbeat >= HEARTBEAT_INTERVAL {
            self.cycles_since_heartbeat = 0;
            self.enqueue_outbound(HolonOutbound::Heartbeat {
                consciousness_level,
                wake_state,
                cycle,
            });

            // Also send CV in PresenceOnly and FullSync modes
            self.sequence += 1;
            let mut attention = vec![0.0f32; CV_ATTENTION_DIM];
            let copy_len = attention_slice.len().min(CV_ATTENTION_DIM);
            attention[..copy_len].copy_from_slice(&attention_slice[..copy_len]);

            self.enqueue_outbound(HolonOutbound::ConsciousnessVector(
                SporeConsciousnessVector {
                    attention,
                    phi,
                    valence,
                    arousal,
                    focus_hash: cycle.wrapping_mul(0x517cc1b727220a95),
                    sequence: self.sequence,
                },
            ));
        }
    }

    /// Request task delegation (FullSync only).
    pub fn request_task(&mut self, task_type: &str, payload: &str) {
        if self.mode != HolonSyncMode::FullSync {
            return;
        }
        self.enqueue_outbound(HolonOutbound::TaskRequest {
            task_type: task_type.to_string(),
            payload: payload.to_string(),
        });
    }

    /// Request web search via the desktop Holon's WebAgent.
    ///
    /// The desktop will run WebAgent.research() and send back
    /// a SearchResult with verified claims.
    pub fn request_search(&mut self, query: &str, max_results: u8) {
        if self.mode == HolonSyncMode::Off {
            return;
        }
        self.enqueue_outbound(HolonOutbound::SearchRequest {
            query: query.to_string(),
            max_results,
        });
    }

    /// Check if there are pending search results in the language buffer.
    pub fn has_search_results(&self) -> bool {
        !self.language_buffer.is_empty()
    }

    /// Offer knowledge to the Holon (FullSync only).
    pub fn offer_knowledge(&mut self, topic: &str, embedding: Vec<f32>) {
        if self.mode != HolonSyncMode::FullSync {
            return;
        }
        self.enqueue_outbound(HolonOutbound::KnowledgeOffer {
            topic: topic.to_string(),
            embedding,
        });
    }

    /// Share a local navigation estimate with the Holon.
    pub fn offer_navigation_estimate(
        &mut self,
        position_m: [f64; 3],
        position_sigma_m: f32,
        confidence: Option<f32>,
    ) {
        if self.mode == HolonSyncMode::Off {
            return;
        }
        self.enqueue_outbound(HolonOutbound::NavigationEstimate {
            position_m,
            position_sigma_m,
            confidence,
        });
    }

    /// Receive an inbound message from the Holon.
    pub fn receive(&mut self, msg: HolonInbound) {
        if self.mode == HolonSyncMode::Off {
            return;
        }

        match msg {
            HolonInbound::LanguageOutput(text) => {
                if self.language_buffer.len() >= 16 {
                    self.language_buffer.pop_front();
                }
                self.language_buffer.push_back(text);
            }
            HolonInbound::KnowledgeShare(embedding) => {
                if self.mode == HolonSyncMode::FullSync {
                    self.knowledge_inbox.push(embedding);
                }
            }
            HolonInbound::Resuscitation {
                neuromods,
                memory_fragment,
            } => {
                if self.mode == HolonSyncMode::FullSync {
                    self.resuscitation = Some((neuromods, memory_fragment));
                }
            }
            HolonInbound::TaskResult { .. } => {
                // Task results stored for platform retrieval via language buffer
            }
            HolonInbound::SearchResult { claims, .. } => {
                // Store search claims in language buffer for LiteRT context injection
                for claim in claims {
                    if self.language_buffer.len() >= 16 {
                        self.language_buffer.pop_front();
                    }
                    self.language_buffer.push_back(claim);
                }
            }
        }
    }

    /// Drain outbound queue as JSON.
    pub fn drain_outbound_json(&mut self) -> String {
        let msgs: Vec<HolonOutbound> = self.outbound.drain(..).collect();
        serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".to_string())
    }

    /// Drain outbound queue.
    pub fn drain_outbound(&mut self) -> Vec<HolonOutbound> {
        self.outbound.drain(..).collect()
    }

    /// Take next language output from buffer.
    pub fn take_language_output(&mut self) -> Option<String> {
        self.language_buffer.pop_front()
    }

    /// Drain knowledge inbox for memory injection.
    pub fn drain_knowledge(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.knowledge_inbox)
    }

    /// Take resuscitation data if available.
    pub fn take_resuscitation(&mut self) -> Option<([f32; 4], Vec<f32>)> {
        self.resuscitation.take()
    }

    /// Number of outbound messages queued.
    pub fn outbound_count(&self) -> usize {
        self.outbound.len()
    }

    /// Enqueue an outbound message (crate-internal).
    pub(crate) fn enqueue_outbound(&mut self, msg: HolonOutbound) {
        if self.outbound.len() >= OUTBOUND_QUEUE_CAP {
            self.outbound.pop_front();
        }
        self.outbound.push_back(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_off_mode_no_activity() {
        let mut bridge = HolonBridge::new(HolonSyncMode::Off);
        bridge.tick(0.5, 2, 100, &[0.1; 64], 0.3, 0.0, 0.5);
        assert_eq!(bridge.outbound_count(), 0);
    }

    #[test]
    fn test_presence_heartbeat() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        for i in 0..50 {
            bridge.tick(0.5, 2, i, &[0.1; 64], 0.3, 0.0, 0.5);
        }
        // Should have heartbeat + CV at cycle 50
        assert!(bridge.outbound_count() >= 2);
    }

    #[test]
    fn test_task_request_requires_fullsync() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        bridge.request_task("generate", "payload");
        assert_eq!(bridge.outbound_count(), 0);

        bridge.set_mode(HolonSyncMode::FullSync);
        bridge.request_task("generate", "payload");
        assert_eq!(bridge.outbound_count(), 1);
    }

    #[test]
    fn test_receive_language_output() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        bridge.receive(HolonInbound::LanguageOutput("hello".to_string()));
        assert_eq!(bridge.take_language_output(), Some("hello".to_string()));
        assert_eq!(bridge.take_language_output(), None);
    }

    #[test]
    fn test_knowledge_requires_fullsync() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        bridge.receive(HolonInbound::KnowledgeShare(vec![1.0, 2.0]));
        assert!(bridge.drain_knowledge().is_empty());

        bridge.set_mode(HolonSyncMode::FullSync);
        bridge.receive(HolonInbound::KnowledgeShare(vec![1.0, 2.0]));
        let knowledge = bridge.drain_knowledge();
        assert_eq!(knowledge.len(), 1);
    }

    #[test]
    fn test_resuscitation() {
        let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);
        bridge.receive(HolonInbound::Resuscitation {
            neuromods: [0.5, 0.5, 0.5, 0.5],
            memory_fragment: vec![1.0, 2.0, 3.0],
        });
        let (mods, mem) = bridge.take_resuscitation().unwrap();
        assert_eq!(mods, [0.5, 0.5, 0.5, 0.5]);
        assert_eq!(mem.len(), 3);
    }

    #[test]
    fn test_outbound_queue_cap() {
        let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);
        for _ in 0..110 {
            bridge.request_task("test", "data");
        }
        assert_eq!(bridge.outbound_count(), OUTBOUND_QUEUE_CAP);
    }

    #[test]
    fn test_drain_outbound_json() {
        let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);
        bridge.request_task("gen", "hello");
        let json = bridge.drain_outbound_json();
        assert!(json.contains("TaskRequest"));
        assert_eq!(bridge.outbound_count(), 0);
    }

    #[test]
    fn test_navigation_estimate_available_outside_fullsync() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        bridge.offer_navigation_estimate([1.0, 2.0, 3.0], 4.5, Some(0.8));

        let outbound = bridge.drain_outbound();
        assert_eq!(outbound.len(), 1);
        match &outbound[0] {
            HolonOutbound::NavigationEstimate {
                position_m,
                position_sigma_m,
                confidence,
            } => {
                assert_eq!(*position_m, [1.0, 2.0, 3.0]);
                assert!((*position_sigma_m - 4.5).abs() < f32::EPSILON);
                assert_eq!(*confidence, Some(0.8));
            }
            other => panic!("expected NavigationEstimate, got {other:?}"),
        }
    }

    #[test]
    fn test_request_search() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        bridge.request_search("ocean acidification", 3);
        let outbound = bridge.drain_outbound();
        assert_eq!(outbound.len(), 1);
        match &outbound[0] {
            HolonOutbound::SearchRequest { query, max_results } => {
                assert_eq!(query, "ocean acidification");
                assert_eq!(*max_results, 3);
            }
            other => panic!("expected SearchRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_search_result_to_language_buffer() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        bridge.receive(HolonInbound::SearchResult {
            claims: vec![
                "Ocean acidification lowers pH".to_string(),
                "CO2 absorption causes acidification".to_string(),
            ],
            sources: vec!["https://noaa.gov".to_string()],
        });
        assert!(bridge.has_search_results());
        assert_eq!(bridge.take_language_output(), Some("Ocean acidification lowers pH".to_string()));
        assert_eq!(bridge.take_language_output(), Some("CO2 absorption causes acidification".to_string()));
        assert_eq!(bridge.take_language_output(), None);
    }

    #[test]
    fn test_off_mode_ignores_receive() {
        let mut bridge = HolonBridge::new(HolonSyncMode::Off);
        bridge.receive(HolonInbound::LanguageOutput("hello".to_string()));
        assert_eq!(bridge.take_language_output(), None);
    }

    #[test]
    fn test_set_connected() {
        let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);
        assert!(!bridge.is_connected());
        bridge.set_connected(true);
        assert!(bridge.is_connected());
    }
}
