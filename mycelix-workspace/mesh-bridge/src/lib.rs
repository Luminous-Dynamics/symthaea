// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Mesh Bridge — library crate.
//!
//! Exports modules for integration tests and the binary entry point.

pub mod poller;
pub mod relay;
pub mod serializer;
pub mod transport;

pub mod dedup_cache;
pub mod encryption;
pub mod meshtastic;
pub mod espnow;
pub mod reticulum;
pub mod ccsds;

#[cfg(feature = "mqtt")]
pub mod mqtt;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Shared operational metrics for the mesh bridge.
///
/// Exposed via the `/health` HTTP endpoint so community operators
/// can monitor mesh health without digging through logs.
#[derive(Clone)]
pub struct BridgeMetrics {
    /// Total messages relayed outbound (conductor → mesh).
    pub messages_sent: Arc<AtomicU64>,
    /// Total messages relayed inbound (mesh → conductor).
    pub messages_received: Arc<AtomicU64>,
    /// Total LoRa fragments dropped (reassembly timeout or malformed).
    pub fragments_dropped: Arc<AtomicU64>,
    /// Distinct peer origin IDs seen (via heartbeat or relay).
    pub peers_seen: Arc<AtomicU64>,
    /// Epoch millis of last successful outbound relay.
    pub last_send_at: Arc<AtomicU64>,
    /// Epoch millis of last successful inbound relay.
    pub last_recv_at: Arc<AtomicU64>,
    /// Total poll cycles completed.
    pub poll_cycles: Arc<AtomicU64>,
    /// Total conductor connection failures.
    pub connection_failures: Arc<AtomicU64>,
}

impl BridgeMetrics {
    pub fn new() -> Self {
        Self {
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            fragments_dropped: Arc::new(AtomicU64::new(0)),
            peers_seen: Arc::new(AtomicU64::new(0)),
            last_send_at: Arc::new(AtomicU64::new(0)),
            last_recv_at: Arc::new(AtomicU64::new(0)),
            poll_cycles: Arc::new(AtomicU64::new(0)),
            connection_failures: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Return metrics in Prometheus text exposition format (version 0.0.4).
    pub fn to_prometheus(&self) -> String {
        let sent = self.messages_sent.load(Ordering::Relaxed);
        let recv = self.messages_received.load(Ordering::Relaxed);
        let dropped = self.fragments_dropped.load(Ordering::Relaxed);
        let peers = self.peers_seen.load(Ordering::Relaxed);
        let cycles = self.poll_cycles.load(Ordering::Relaxed);
        let failures = self.connection_failures.load(Ordering::Relaxed);
        let last_send = self.last_send_at.load(Ordering::Relaxed);
        let last_recv = self.last_recv_at.load(Ordering::Relaxed);

        format!(
            "# HELP mesh_bridge_messages_sent Total messages relayed outbound\n\
             # TYPE mesh_bridge_messages_sent counter\n\
             mesh_bridge_messages_sent {sent}\n\
             # HELP mesh_bridge_messages_received Total messages relayed inbound\n\
             # TYPE mesh_bridge_messages_received counter\n\
             mesh_bridge_messages_received {recv}\n\
             # HELP mesh_bridge_fragments_dropped Total fragments dropped\n\
             # TYPE mesh_bridge_fragments_dropped counter\n\
             mesh_bridge_fragments_dropped {dropped}\n\
             # HELP mesh_bridge_peers_seen Distinct peers seen\n\
             # TYPE mesh_bridge_peers_seen gauge\n\
             mesh_bridge_peers_seen {peers}\n\
             # HELP mesh_bridge_poll_cycles Total poll cycles completed\n\
             # TYPE mesh_bridge_poll_cycles counter\n\
             mesh_bridge_poll_cycles {cycles}\n\
             # HELP mesh_bridge_connection_failures Total conductor connection failures\n\
             # TYPE mesh_bridge_connection_failures counter\n\
             mesh_bridge_connection_failures {failures}\n\
             # HELP mesh_bridge_last_send_at Epoch ms of last outbound relay\n\
             # TYPE mesh_bridge_last_send_at gauge\n\
             mesh_bridge_last_send_at {last_send}\n\
             # HELP mesh_bridge_last_recv_at Epoch ms of last inbound relay\n\
             # TYPE mesh_bridge_last_recv_at gauge\n\
             mesh_bridge_last_recv_at {last_recv}\n"
        )
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "messages_sent": self.messages_sent.load(Ordering::Relaxed),
            "messages_received": self.messages_received.load(Ordering::Relaxed),
            "fragments_dropped": self.fragments_dropped.load(Ordering::Relaxed),
            "peers_seen": self.peers_seen.load(Ordering::Relaxed),
            "last_send_at": self.last_send_at.load(Ordering::Relaxed),
            "last_recv_at": self.last_recv_at.load(Ordering::Relaxed),
            "poll_cycles": self.poll_cycles.load(Ordering::Relaxed),
            "connection_failures": self.connection_failures.load(Ordering::Relaxed),
        })
    }
}

impl Default for BridgeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_default() {
        let m = BridgeMetrics::new();
        assert_eq!(m.messages_sent.load(Ordering::Relaxed), 0);
        assert_eq!(m.peers_seen.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_to_json() {
        let m = BridgeMetrics::new();
        m.messages_sent.store(42, Ordering::Relaxed);
        m.peers_seen.store(3, Ordering::Relaxed);
        let json = m.to_json();
        assert_eq!(json["messages_sent"], 42);
        assert_eq!(json["peers_seen"], 3);
        assert_eq!(json["fragments_dropped"], 0);
    }

    #[test]
    fn test_metrics_to_prometheus() {
        let m = BridgeMetrics::new();
        m.messages_sent.store(10, Ordering::Relaxed);
        m.messages_received.store(5, Ordering::Relaxed);
        m.fragments_dropped.store(2, Ordering::Relaxed);
        m.peers_seen.store(3, Ordering::Relaxed);
        m.poll_cycles.store(100, Ordering::Relaxed);
        m.connection_failures.store(1, Ordering::Relaxed);
        m.last_send_at.store(1700000000000, Ordering::Relaxed);
        m.last_recv_at.store(1700000001000, Ordering::Relaxed);
        let prom = m.to_prometheus();
        assert!(prom.contains("mesh_bridge_messages_sent 10"));
        assert!(prom.contains("mesh_bridge_messages_received 5"));
        assert!(prom.contains("mesh_bridge_fragments_dropped 2"));
        assert!(prom.contains("mesh_bridge_peers_seen 3"));
        assert!(prom.contains("mesh_bridge_poll_cycles 100"));
        assert!(prom.contains("mesh_bridge_connection_failures 1"));
        assert!(prom.contains("mesh_bridge_last_send_at 1700000000000"));
        assert!(prom.contains("mesh_bridge_last_recv_at 1700000001000"));
        assert!(prom.contains("# TYPE mesh_bridge_messages_sent counter"));
        assert!(prom.contains("# TYPE mesh_bridge_peers_seen gauge"));
    }

    #[test]
    fn test_metrics_clone_shares_state() {
        let m = BridgeMetrics::new();
        let m2 = m.clone();
        m.messages_sent.fetch_add(1, Ordering::Relaxed);
        assert_eq!(m2.messages_sent.load(Ordering::Relaxed), 1);
    }
}
