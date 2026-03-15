//! Mycelix Mesh Bridge — library crate.
//!
//! Exports modules for integration tests and the binary entry point.

pub mod poller;
pub mod relay;
pub mod serializer;
pub mod transport;

pub mod dedup_cache;

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
    fn test_metrics_clone_shares_state() {
        let m = BridgeMetrics::new();
        let m2 = m.clone();
        m.messages_sent.fetch_add(1, Ordering::Relaxed);
        assert_eq!(m2.messages_sent.load(Ordering::Relaxed), 1);
    }
}
