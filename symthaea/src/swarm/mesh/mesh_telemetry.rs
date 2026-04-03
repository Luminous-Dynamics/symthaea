// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh telemetry types: stats, urgency, payload type, and error types.

use crate::cognitive_loop::types::CycleUrgency;
use serde::{Deserialize, Serialize};

// ============================================================================
// MESH TELEMETRY
// ============================================================================

/// Aggregate counters for mesh packet flow (observability).
#[derive(Debug, Clone, Default)]
pub struct MeshStats {
    /// Wisdom vectors emitted to mesh.
    pub wisdom_sent: u64,
    /// Heartbeat packets emitted to mesh.
    pub heartbeats_sent: u64,
    /// Affective state packets emitted to mesh.
    pub affective_sent: u64,
    /// Gradient packets emitted to mesh.
    pub gradients_sent: u64,
    /// Moral topology packets emitted to mesh.
    pub moral_topology_sent: u64,
    /// Wisdom vectors received from mesh peers.
    pub wisdom_received: u64,
    /// Heartbeat packets received from mesh peers.
    pub heartbeats_received: u64,
    /// Affective packets received from mesh peers.
    pub affective_received: u64,
    /// Gradient packets received from mesh peers.
    pub gradients_received: u64,
    /// Moral topology packets received from mesh peers.
    pub moral_topology_received: u64,
    /// Number of peers removed by expiry.
    pub peers_expired: u64,
    /// Total bytes sent over mesh (estimated from packet count × WISDOM_PACKET_SIZE).
    pub bytes_sent: u64,
    /// Total bytes received from mesh (estimated from packet count × WISDOM_PACKET_SIZE).
    pub bytes_received: u64,
    /// Packets dropped due to inbox/outbox backpressure caps.
    pub packets_dropped: u64,
    /// Duplicate packets detected and skipped.
    pub packets_deduplicated: u64,
    /// Packets rejected by per-peer rate limiting.
    pub packets_rate_limited: u64,
    /// Emissions throttled by bandwidth budget enforcement.
    pub bandwidth_throttled: u64,
    /// Packets that failed authentication (MAC verification).
    pub packets_auth_failed: u64,
    /// Packets that failed decryption (wrong key or corrupted ciphertext).
    pub packets_decrypt_failed: u64,
    /// Packets forwarded via gossip TTL.
    pub packets_forwarded: u64,
    /// Packets replayed to newly-discovered peers.
    pub packets_replayed: u64,
    /// Total bytes before compression (for compression ratio telemetry).
    pub bytes_before_compression: u64,
    /// Total bytes after compression (for compression ratio telemetry).
    pub bytes_after_compression: u64,
    /// Current dynamic bandwidth budget (AIMD-adjusted).
    pub bandwidth_budget_current: u64,
    /// Number of AIMD additive increases (budget went up).
    pub bandwidth_increases: u64,
    /// Number of AIMD multiplicative decreases (budget went down).
    pub bandwidth_decreases: u64,
    /// Packets sent with encryption enabled.
    pub encrypted_packets_sent: u64,
    /// Packets received and successfully decrypted.
    pub encrypted_packets_received: u64,
    /// Time beacon packets emitted (Sovereign Clock).
    pub time_beacons_sent: u64,
    /// Time beacon packets received (Sovereign Clock).
    pub time_beacons_received: u64,
    /// Name query packets emitted (Sovereign Name).
    pub name_queries_sent: u64,
    /// Name query packets received (Sovereign Name).
    pub name_queries_received: u64,
    /// Name response packets emitted (Sovereign Name).
    pub name_responses_sent: u64,
    /// Name response packets received (Sovereign Name).
    pub name_responses_received: u64,
    /// Content announcement packets emitted (Sovereign Social).
    pub content_announces_sent: u64,
    /// Content announcement packets received (Sovereign Social).
    pub content_announces_received: u64,
}

impl MeshStats {
    /// Total packets sent across all types.
    pub(super) fn total_sent(&self) -> u64 {
        self.wisdom_sent
            + self.heartbeats_sent
            + self.affective_sent
            + self.gradients_sent
            + self.moral_topology_sent
            + self.time_beacons_sent
            + self.name_queries_sent
            + self.name_responses_sent
            + self.content_announces_sent
    }

    /// Total packets received across all types.
    pub(super) fn total_received(&self) -> u64 {
        self.wisdom_received
            + self.heartbeats_received
            + self.affective_received
            + self.gradients_received
            + self.moral_topology_received
            + self.time_beacons_received
            + self.name_queries_received
            + self.name_responses_received
            + self.content_announces_received
    }

    /// Returns the compression ratio (0.0–1.0, lower is better).
    /// Returns 1.0 (no compression) if no data has been compressed.
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_before_compression == 0 {
            return 1.0;
        }
        self.bytes_after_compression as f64 / self.bytes_before_compression as f64
    }

    /// Compute a composite health score for the mesh network.
    ///
    /// Returns a value in `[0.0, 1.0]` combining:
    /// - **Connectivity** (40%): saturates at 5 peers
    /// - **Bidirectionality** (40%): balanced send/recv ratio → 1.0, one-sided → 0.0
    /// - **Stability** (20%): fewer expired peers relative to activity → higher score
    ///
    /// Returns 0.0 if no packets have been sent or received.
    pub fn health_score(&self, peer_count: usize) -> f32 {
        let sent = self.total_sent();
        let recv = self.total_received();
        if sent == 0 && recv == 0 {
            return 0.0;
        }

        // Connectivity: saturates at 5 peers
        let connectivity = (peer_count as f32 / 5.0).clamp(0.0, 1.0);

        // Bidirectionality: ratio of min/max → 1.0 when balanced, 0.0 when one-sided
        let bidirectionality = if sent == 0 || recv == 0 {
            0.0
        } else {
            let min = sent.min(recv) as f32;
            let max = sent.max(recv) as f32;
            min / max
        };

        // Stability: fewer expired peers relative to total activity → better
        let stability = 1.0 - (self.peers_expired as f32 / (self.peers_expired + recv + 1) as f32);

        (connectivity * 0.4 + bidirectionality * 0.4 + stability * 0.2).clamp(0.0, 1.0)
    }
}

/// Structured snapshot of mesh network telemetry at a point in time.
#[derive(Debug, Clone, Default)]
pub struct MeshTelemetry {
    /// Aggregate packet counters.
    pub stats: MeshStats,
    /// Number of currently tracked peers.
    pub peer_count: usize,
    /// Average Phi across tracked peers.
    pub avg_phi: f32,
    /// Composite health score (0.0–1.0).
    pub health_score: f32,
    /// Moral topology summary for cross-agent coherence.
    pub moral_topology: Option<crate::hdc::moral_topology::MoralTopologySummary>,
}

// ============================================================================
// MESH URGENCY
// ============================================================================

/// Cognitive urgency level mapped to physical transport selection.
///
/// The mind's internal prediction error directly controls which radio
/// frequency carries its thoughts. High surprise? Blast it over WiFi mesh
/// at the speed of light. Cruise-mode philosophy? Let it drift over LoRa
/// at the speed of wisdom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MeshUrgency {
    /// Stable state — route over LoRa (low power, 10-15km, ~0.3 Hz).
    Cruise = 0,
    /// Standard processing — route over Yggdrasil/Iroh (encrypted overlay).
    Normal = 1,
    /// High prediction error — route over B.A.T.M.A.N. (WiFi mesh, <10ms).
    Critical = 2,
}

impl MeshUrgency {
    /// Decode from a single byte (radio wire format).
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Cruise,
            1 => Self::Normal,
            _ => Self::Critical,
        }
    }
}

impl From<CycleUrgency> for MeshUrgency {
    fn from(urgency: CycleUrgency) -> Self {
        match urgency {
            CycleUrgency::Critical => Self::Critical,
            CycleUrgency::Normal => Self::Normal,
            CycleUrgency::Cruise => Self::Cruise,
        }
    }
}

// ============================================================================
// PAYLOAD TYPE
// ============================================================================

/// Discriminant for what a [`WisdomPacket`](super::WisdomPacket) carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PayloadType {
    /// Full 16,384-dimensional `BinaryHV` (wisdom distillation).
    WisdomVector = 0,
    /// Affective state synchronization (VAD model).
    Affective = 1,
    /// Heartbeat / presence announcement.
    Heartbeat = 2,
    /// Federated gradient fragment.
    Gradient = 3,
    /// Moral topology summary for cross-agent coherence.
    MoralTopology = 4,
    /// Time beacon for mesh-time consensus (Sovereign Clock).
    TimeBeacon = 5,
    /// Name query for mesh name resolution (Sovereign Name).
    NameQuery = 6,
    /// Name response for mesh name resolution (Sovereign Name).
    NameResponse = 7,
    /// Content announcement for resonance-based discovery (Sovereign Social).
    ContentAnnounce = 8,
}

impl PayloadType {
    /// Decode from a single byte (radio wire format).
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::WisdomVector,
            1 => Self::Affective,
            2 => Self::Heartbeat,
            3 => Self::Gradient,
            4 => Self::MoralTopology,
            5 => Self::TimeBeacon,
            6 => Self::NameQuery,
            7 => Self::NameResponse,
            8 => Self::ContentAnnounce,
            _ => Self::Heartbeat, // unknown types become heartbeats (safe no-op)
        }
    }

    /// Backpressure priority: higher values are retained first when inbox is full.
    /// Heartbeat(3) > Wisdom(2) > TimeBeacon(2) > MoralTopology/Affective/Name/Content(1) > Gradient(0).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Heartbeat => 3,
            Self::WisdomVector => 2,
            Self::TimeBeacon => 2,
            Self::Affective => 1,
            Self::MoralTopology => 1,
            Self::NameQuery => 1,
            Self::NameResponse => 1,
            Self::ContentAnnounce => 1,
            Self::Gradient => 0,
        }
    }
}

// ============================================================================
// MESH ERROR
// ============================================================================

/// Errors from mesh transport operations.
#[derive(Debug)]
pub enum MeshError {
    /// No transport available for the requested urgency level.
    NoTransport,
    /// Transport I/O error.
    Io(String),
    /// Fragment reassembly failed (too many losses).
    ReassemblyFailed,
    /// Payload too large for fragmentation.
    PayloadTooLarge,
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTransport => write!(f, "no mesh transport available"),
            Self::Io(msg) => write!(f, "mesh I/O error: {msg}"),
            Self::ReassemblyFailed => write!(f, "fragment reassembly failed"),
            Self::PayloadTooLarge => write!(f, "payload exceeds max fragment count"),
        }
    }
}

impl std::error::Error for MeshError {}
