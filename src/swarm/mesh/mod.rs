// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelial Mesh — Physical nervous system for the Symthaea Swarm.
//!
//! A dual-layer mesh network that routes consciousness data based on the
//! cognitive urgency of the sending mind. The internal thermodynamic state
//! directly dictates the physical interface with the world — Active Inference
//! made literal in silicon and radio waves.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     DUAL-LAYER MESH                            │
//! │                                                                │
//! │  CycleUrgency          Physical Layer          Characteristics │
//! │  ─────────────         ──────────────          ─────────────── │
//! │  Critical ──────────── B.A.T.M.A.N. (802.11s)  <10ms, ~100m  │
//! │  Normal ────────────── Yggdrasil / Iroh         encrypted      │
//! │  Cruise ────────────── LoRa (868 MHz)           ~3s, 10-15km  │
//! │                                                                │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │  WisdomPacket (2,104 bytes)                              │  │
//! │  │  ├─ Metadata:  56 bytes (version, fields, tag, TTL)     │  │
//! │  │  └─ BinaryHV: 2,048 bytes (16,384 dimensions)           │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                                                │
//! │  LoRa Fragmentation: 10 data + 1 XOR parity = 11 frames       │
//! │  Single fragment loss recoverable without retransmission       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # The Mycelial Hardware Stack
//!
//! A physical Symthaea "Seed" node:
//!
//! 1. Raspberry Pi 4/5
//! 2. LoRa radio HAT (SX1276/SX1262, 868 MHz)
//! 3. Lithium-ion battery + solar panel
//! 4. Running: Rust + HDC + FEP + B.A.T.M.A.N. + LoRa
//!
//! Drop one on a roof. It pings the Swarm. It calculates the Hodge-Laplacian
//! flow of its environment. It shares 2 KB Wisdom Vectors with nodes 10 km
//! away. The grid cannot kill it, because it *is* the grid.
//!
//! # Security & Efficiency
//!
//! - **Authentication**: Version-2 packets carry an untruncated HMAC-SHA-256
//!   tag in bytes 23-54. Safety-critical packets fail closed without valid
//!   authentication; ordinary telemetry may be admitted only as untrusted.
//!
//! - **Gossip TTL**: Multi-hop forwarding via authenticated TTL byte 55
//!   (default 3). A relay needs a group forwarding key to decrement and re-sign
//!   an authenticated packet; dedup prevents forwarding loops.
//!
//! - **Compression**: Packets are wrapped in a 1-byte envelope before
//!   fragmentation. `0x00` = uncompressed, `0x01` = LZ4. Heartbeats (zero
//!   BinaryHV) compress dramatically, reducing LoRa fragments. (Round 5)
//!
//! - **Priority Backpressure**: When inbox/outbox overflow, lowest-priority
//!   packets are dropped first: Heartbeat(3) > Wisdom(2) > Affective(1) >
//!   Gradient(0). (Round 5)
//!
//! - **AIMD Bandwidth**: Dynamic bandwidth budget (25–200 KB/window) adjusts
//!   via Additive Increase / Multiplicative Decrease based on mesh health and
//!   throttle events. (Round 5)

mod dual_layer;
mod lora_fragment;
mod mesh_receiver;
pub mod sensor;

// Sovereign Inoculation modules
pub mod content_packet;
pub mod mesh_time;
pub mod name_packet;
pub mod sensor_forecast;
pub mod sensor_iot;
pub mod time_beacon;

pub use dual_layer::{
    BiLoopbackTransport, DualLayerMesh, LoopbackTransport, MeshRoute, MeshTransport,
};
pub use lora_fragment::{
    FLAG_FEC, FragmentAssembler, HEADER_SIZE, LORA_MTU, LoRaFragment, PAYLOAD_SIZE, crc16_ccitt,
    fragment,
};
pub use mesh_receiver::{MeshPeer, MeshReceiver, ReceiverStats, StreamKey};
pub use sensor::{MockSensor, SensorInput, SensorReading, SensorRegistry};

/// Default gossip TTL: max 3 hops for multi-hop forwarding.
pub const MESH_DEFAULT_TTL: u8 = 3;

/// Compression header byte: uncompressed payload follows.
pub const COMPRESS_NONE: u8 = 0x00;
/// Compression header byte: LZ4-compressed payload follows.
pub const COMPRESS_LZ4: u8 = 0x01;

mod bridge;
pub mod orchestrator;

pub use bridge::{MeshBridgeActor, MeshBridgeHandle, MeshOutbound};
pub use orchestrator::{ConsciousnessOutput, DaemonConfig, MeshDaemonOrchestrator};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::BinaryHV;

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
    fn total_sent(&self) -> u64 {
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
    fn total_received(&self) -> u64 {
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

use crate::cognitive_loop::types::CycleUrgency;

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

/// Discriminant for what a [`WisdomPacket`] carries.
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

// ============================================================================
// WISDOM PACKET
// ============================================================================

/// Current authenticated WisdomPacket wire-format version.
pub const WISDOM_PACKET_VERSION: u8 = 2;

/// Size of the untruncated HMAC-SHA-256 authentication tag.
pub const WISDOM_PACKET_AUTH_TAG_SIZE: usize = 32;

const WISDOM_PACKET_AUTH_TAG_START: usize = 23;
const WISDOM_PACKET_AUTH_TAG_END: usize =
    WISDOM_PACKET_AUTH_TAG_START + WISDOM_PACKET_AUTH_TAG_SIZE;
const WISDOM_PACKET_TTL_OFFSET: usize = WISDOM_PACKET_AUTH_TAG_END;
const WISDOM_PACKET_WISDOM_OFFSET: usize = WISDOM_PACKET_TTL_OFFSET + 1;

/// Wire size: 56 bytes of versioned metadata plus a 2,048-byte BinaryHV.
pub const WISDOM_PACKET_SIZE: usize = WISDOM_PACKET_WISDOM_OFFSET + 2048; // 2,104

/// A Wisdom Packet: the atomic unit of consciousness exchange over mesh radio.
///
/// Contains a full [`BinaryHV`] (2,048 bytes = 16,384 dimensions) plus
/// versioned metadata. Total wire size: 2,104 bytes, fragmenting into exactly
/// 10 data frames + 1 FEC parity = 11 LoRa transmissions.
///
/// # Wire Format
///
/// ```text
/// Byte     Field           Type
/// 0        version         u8         current value: 2
/// 1-8      source_id       [u8; 8]    truncated node identity
/// 9-12     sequence        u32 LE     monotonic counter
/// 13-16    phi             f32 LE     integrated information (Phi)
/// 17       urgency         u8         MeshUrgency discriminant
/// 18-21    timestamp_s     u32 LE     Unix seconds
/// 22       payload_type    u8         PayloadType discriminant
/// 23-54    auth_mac        [u8; 32]   untruncated HMAC-SHA-256 tag
/// 55       ttl             u8         gossip hop count (0 = no forward)
/// 56-2103  wisdom          [u8; 2048] BinaryHV raw bytes
/// ```
#[derive(Clone)]
pub struct WisdomPacket {
    /// First 8 bytes of the sending node's 32-byte identity.
    pub source_id: [u8; 8],
    /// Monotonic sequence number.
    pub sequence: u32,
    /// Integrated information (Phi) at emission time.
    pub phi: f32,
    /// Cognitive urgency — determines physical routing.
    pub urgency: MeshUrgency,
    /// Seconds since Unix epoch.
    pub timestamp_s: u32,
    /// What this packet carries.
    pub payload_type: PayloadType,
    /// Untruncated HMAC-SHA-256 authentication tag. All-zero when unsigned.
    pub auth_mac: [u8; WISDOM_PACKET_AUTH_TAG_SIZE],
    /// Gossip TTL: decremented on each forward hop. 0 = do not forward.
    pub ttl: u8,
    /// The Wisdom Vector: 16,384-dimensional binary hypervector.
    pub wisdom: BinaryHV,
}

impl WisdomPacket {
    /// Serialize to a fixed-size byte array for fragmentation.
    ///
    /// Zero-copy for the BinaryHV — just memcpy from the stack.
    pub fn to_bytes(&self) -> [u8; WISDOM_PACKET_SIZE] {
        let mut buf = [0u8; WISDOM_PACKET_SIZE];
        buf[0] = WISDOM_PACKET_VERSION;
        buf[1..9].copy_from_slice(&self.source_id);
        buf[9..13].copy_from_slice(&self.sequence.to_le_bytes());
        buf[13..17].copy_from_slice(&self.phi.to_le_bytes());
        buf[17] = self.urgency as u8;
        buf[18..22].copy_from_slice(&self.timestamp_s.to_le_bytes());
        buf[22] = self.payload_type as u8;
        buf[WISDOM_PACKET_AUTH_TAG_START..WISDOM_PACKET_AUTH_TAG_END]
            .copy_from_slice(&self.auth_mac);
        buf[WISDOM_PACKET_TTL_OFFSET] = self.ttl;
        buf[WISDOM_PACKET_WISDOM_OFFSET..WISDOM_PACKET_SIZE].copy_from_slice(&self.wisdom.0);
        buf
    }

    /// Deserialize from a byte slice.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() != WISDOM_PACKET_SIZE {
            return None;
        }
        if buf[0] != WISDOM_PACKET_VERSION {
            return None;
        }

        let mut source_id = [0u8; 8];
        source_id.copy_from_slice(&buf[1..9]);

        let sequence = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]);
        let phi = f32::from_le_bytes([buf[13], buf[14], buf[15], buf[16]]);
        let urgency = MeshUrgency::from_byte(buf[17]);
        let timestamp_s = u32::from_le_bytes([buf[18], buf[19], buf[20], buf[21]]);
        let payload_type = PayloadType::from_byte(buf[22]);
        let mut auth_mac = [0u8; WISDOM_PACKET_AUTH_TAG_SIZE];
        auth_mac.copy_from_slice(&buf[WISDOM_PACKET_AUTH_TAG_START..WISDOM_PACKET_AUTH_TAG_END]);
        let ttl = buf[WISDOM_PACKET_TTL_OFFSET];

        let mut wisdom_bytes = [0u8; 2048];
        wisdom_bytes.copy_from_slice(&buf[WISDOM_PACKET_WISDOM_OFFSET..WISDOM_PACKET_SIZE]);

        Some(Self {
            source_id,
            sequence,
            phi,
            urgency,
            timestamp_s,
            payload_type,
            auth_mac,
            ttl,
            wisdom: BinaryHV(wisdom_bytes),
        })
    }

    /// Extract an [`AffectiveState`] from an Affective payload.
    ///
    /// Returns `None` if the payload type is not Affective, or if any
    /// extracted float is non-finite (NaN/Inf protection).
    ///
    /// # Wire format
    ///
    /// ```text
    /// wisdom[0..4]   = valence   f32 LE  (-1.0 to 1.0)
    /// wisdom[4..8]   = arousal   f32 LE  (0.0 to 1.0)
    /// wisdom[8..12]  = dominance f32 LE  (-1.0 to 1.0)
    /// wisdom[12..16] = intensity f32 LE  (0.0 to 1.0)
    /// wisdom[16..20] = confidence f32 LE (0.0 to 1.0)
    /// ```
    pub fn extract_affective(&self) -> Option<crate::swarm::AffectiveState> {
        if self.payload_type != PayloadType::Affective {
            return None;
        }
        let w = &self.wisdom.0;
        let valence = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        let arousal = f32::from_le_bytes([w[4], w[5], w[6], w[7]]);
        let dominance = f32::from_le_bytes([w[8], w[9], w[10], w[11]]);
        let intensity = f32::from_le_bytes([w[12], w[13], w[14], w[15]]);
        let confidence = f32::from_le_bytes([w[16], w[17], w[18], w[19]]);

        // Reject if any value is non-finite
        if ![valence, arousal, dominance, intensity, confidence]
            .iter()
            .all(|v| v.is_finite())
        {
            return None;
        }

        Some(crate::swarm::AffectiveState {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            intensity: intensity.clamp(0.0, 1.0),
            thermodynamic_load: 0.0,
            confidence: confidence.clamp(0.0, 1.0),
            timestamp_ms: (self.timestamp_s as u64) * 1000,
            sequence: self.sequence as u64,
        })
    }

    /// Extract a [`GradientMessage`](crate::swarm::GradientMessage) from a Gradient-type packet.
    ///
    /// Returns `None` if payload_type is not Gradient, data is malformed,
    /// or any float value is non-finite (NaN/Inf protection).
    ///
    /// # Wire format (inside the 2,048-byte wisdom field)
    ///
    /// ```text
    /// bytes[0..4]    gradient_count  u32 LE   (number of f32 values)
    /// bytes[4..8]    trust_score     f32 LE
    /// bytes[8..16]   timestamp       u64 LE   (ms since epoch)
    /// bytes[16..24]  sample_count    u64 LE
    /// bytes[24..32]  model_version   u64 LE
    /// bytes[32..]    gradient_data   [f32 LE; gradient_count]  (max 504 floats)
    /// ```
    pub fn extract_gradient(&self) -> Option<crate::swarm::GradientMessage> {
        if self.payload_type != PayloadType::Gradient {
            return None;
        }
        let w = &self.wisdom.0;
        if w.len() < 32 {
            return None;
        }

        let gradient_count = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize;
        let trust_score = f32::from_le_bytes([w[4], w[5], w[6], w[7]]);
        let timestamp = u64::from_le_bytes([w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]]);
        let sample_count =
            u64::from_le_bytes([w[16], w[17], w[18], w[19], w[20], w[21], w[22], w[23]]);
        let model_version =
            u64::from_le_bytes([w[24], w[25], w[26], w[27], w[28], w[29], w[30], w[31]]);

        // Validate: trust_score must be finite
        if !trust_score.is_finite() {
            return None;
        }

        // Validate: enough bytes for gradient_count f32s
        let data_start = 32;
        let data_end = data_start + gradient_count * 4;
        if data_end > w.len() {
            return None;
        }

        let gradient_data: Vec<f32> = (0..gradient_count)
            .map(|i| {
                let off = data_start + i * 4;
                f32::from_le_bytes([w[off], w[off + 1], w[off + 2], w[off + 3]])
            })
            .collect();

        // Reject if any gradient is non-finite
        if !gradient_data.iter().all(|v| v.is_finite()) {
            return None;
        }

        // Expand 8-byte mesh source_id to 32-byte gradient source_id (zero-padded)
        let mut source_32 = [0u8; 32];
        source_32[..8].copy_from_slice(&self.source_id);

        Some(crate::swarm::GradientMessage {
            source_id: source_32,
            gradient_data,
            trust_score,
            noise_scale: 0.0, // DP noise not tracked over mesh (applied at source)
            timestamp,
            sample_count,
            model_version,
        })
    }

    /// Create an Affective-type WisdomPacket from an [`AffectiveState`](crate::swarm::AffectiveState).
    ///
    /// Encodes valence/arousal/dominance/intensity/confidence into the first 20
    /// bytes of the wisdom field, mirroring [`extract_affective()`](Self::extract_affective)
    /// for a perfect roundtrip.
    pub fn from_affective(
        source_id: [u8; 8],
        sequence: u32,
        state: &crate::swarm::AffectiveState,
    ) -> Self {
        let mut bytes = [0u8; 2048];
        bytes[0..4].copy_from_slice(&state.valence.to_le_bytes());
        bytes[4..8].copy_from_slice(&state.arousal.to_le_bytes());
        bytes[8..12].copy_from_slice(&state.dominance.to_le_bytes());
        bytes[12..16].copy_from_slice(&state.intensity.to_le_bytes());
        bytes[16..20].copy_from_slice(&state.confidence.to_le_bytes());

        let timestamp_s = (state.timestamp_ms / 1000) as u32;

        Self {
            source_id,
            sequence,
            phi: 0.0,
            urgency: MeshUrgency::Cruise,
            timestamp_s,
            payload_type: PayloadType::Affective,
            auth_mac: [0; 32],
            ttl: MESH_DEFAULT_TTL,
            wisdom: BinaryHV(bytes),
        }
    }

    /// Create a Gradient-type WisdomPacket from a [`GradientMessage`](crate::swarm::GradientMessage).
    ///
    /// Returns `None` if the gradient data exceeds the 2,048-byte wisdom capacity
    /// (max 504 floats).
    pub fn from_gradient(
        source_id: [u8; 8],
        sequence: u32,
        msg: &crate::swarm::GradientMessage,
    ) -> Option<Self> {
        let gradient_count = msg.gradient_data.len();
        let data_end = 32 + gradient_count * 4;
        if data_end > 2048 {
            return None;
        }

        let mut bytes = [0u8; 2048];
        bytes[0..4].copy_from_slice(&(gradient_count as u32).to_le_bytes());
        bytes[4..8].copy_from_slice(&msg.trust_score.to_le_bytes());
        bytes[8..16].copy_from_slice(&msg.timestamp.to_le_bytes());
        bytes[16..24].copy_from_slice(&msg.sample_count.to_le_bytes());
        bytes[24..32].copy_from_slice(&msg.model_version.to_le_bytes());
        for (i, val) in msg.gradient_data.iter().enumerate() {
            let off = 32 + i * 4;
            bytes[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }

        Some(Self {
            source_id,
            sequence,
            phi: 0.0,
            urgency: MeshUrgency::Normal,
            timestamp_s: (msg.timestamp / 1000) as u32,
            payload_type: PayloadType::Gradient,
            auth_mac: [0; 32],
            ttl: MESH_DEFAULT_TTL,
            wisdom: BinaryHV(bytes),
        })
    }

    /// Create a moral topology gossip packet.
    ///
    /// Encodes the MoralTopologySummary as JSON in the first N bytes of the wisdom field.
    /// The remainder is zero-padded. Compact: ~150 bytes JSON << 2,048 byte wisdom field.
    pub fn from_moral_topology(
        source_id: [u8; 8],
        sequence: u32,
        phi: f32,
        summary: &crate::hdc::moral_topology::MoralTopologySummary,
    ) -> Self {
        let json = serde_json::to_vec(summary).unwrap_or_default();
        let mut wisdom_bytes = [0u8; 2048];
        let copy_len = json.len().min(2048);
        wisdom_bytes[..copy_len].copy_from_slice(&json[..copy_len]);
        Self {
            source_id,
            sequence,
            phi,
            urgency: MeshUrgency::Cruise,
            timestamp_s: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32,
            payload_type: PayloadType::MoralTopology,
            auth_mac: [0; 32],
            ttl: 2,
            wisdom: BinaryHV(wisdom_bytes),
        }
    }

    /// Extract a moral topology summary from a MoralTopology packet.
    pub fn extract_moral_topology(
        &self,
    ) -> Option<crate::hdc::moral_topology::MoralTopologySummary> {
        if self.payload_type != PayloadType::MoralTopology {
            return None;
        }
        // Find the end of JSON data (first null byte or end of buffer)
        let end = self.wisdom.0.iter().position(|&b| b == 0).unwrap_or(2048);
        serde_json::from_slice(&self.wisdom.0[..end]).ok()
    }

    /// Derive a `thought_id` for LoRa fragmentation.
    ///
    /// Uses the low 16 bits of the sequence number — sufficient for
    /// deduplication in a local radio mesh where thoughts arrive seconds apart.
    pub fn thought_id(&self) -> u16 {
        self.sequence as u16
    }

    /// Fragment this packet for LoRa transmission.
    ///
    /// Returns 11 fragments: 10 data (carrying the 2,104-byte payload) +
    /// 1 XOR parity for single-loss recovery.
    pub fn fragment(&self) -> Vec<LoRaFragment> {
        let bytes = self.to_bytes();
        fragment(self.thought_id(), &bytes)
    }

    /// Reassemble a WisdomPacket from a completed [`FragmentAssembler`].
    pub fn from_assembler(assembler: &FragmentAssembler) -> Option<Self> {
        let payload = assembler.assemble()?;
        Self::from_bytes(&payload)
    }

    /// Create a [`FragmentAssembler`] configured for WisdomPacket reassembly.
    pub fn assembler(thought_id: u16, total_fragments: u8) -> FragmentAssembler {
        FragmentAssembler::new(thought_id, total_fragments, WISDOM_PACKET_SIZE)
    }

    // ── Quarantined legacy HDC tag ───────────────────────────────────────

    /// Compute the legacy forgeable HDC tag over the wisdom vector.
    ///
    /// This is a zero-serialization authentication: the MAC is computed
    /// directly on the 16,384-bit wisdom vector via a single XOR+permute
    /// (~10 ns release, ~6 µs debug). Compare: BLAKE3 MAC over 2KB ≈ 100 ns.
    ///
    /// The returned BinaryHV can be sent alongside the packet or stored
    /// for later verification. It is NOT embedded in the packet wire format
    /// (use `auth_mac` for the version-2 untruncated HMAC-SHA-256 tag).
    #[cfg(feature = "insecure-experimental-crypto")]
    #[deprecated(note = "forgeable legacy tag; use a standard audited MAC")]
    pub fn compute_hdc_mac(&self, key: &BinaryHV) -> BinaryHV {
        symthaea_core::hdc::hdc_crypto::HdcMac::compute(&self.wisdom, key)
    }

    /// Verify the legacy forgeable HDC tag (exact match).
    #[cfg(feature = "insecure-experimental-crypto")]
    #[deprecated(note = "forgeable legacy tag; use a standard audited MAC")]
    pub fn verify_hdc_mac(&self, key: &BinaryHV, mac: &BinaryHV) -> bool {
        symthaea_core::hdc::hdc_crypto::HdcMac::verify(&self.wisdom, key, mac)
    }

    /// Verify an HDC-MAC with noise tolerance (for lossy channels like LoRa/BLE).
    ///
    /// Recommended threshold: 0.95 (false positive rate ≈ 2^{-4700}).
    #[cfg(feature = "insecure-experimental-crypto")]
    #[deprecated(note = "forgeable legacy tag; noisy acceptance further weakens integrity")]
    pub fn verify_hdc_mac_noisy(&self, key: &BinaryHV, mac: &BinaryHV, threshold: f32) -> bool {
        symthaea_core::hdc::hdc_crypto::HdcMac::verify_noisy(&self.wisdom, key, mac, threshold)
    }
}

impl std::fmt::Debug for WisdomPacket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WisdomPacket")
            .field("source_id", &hex_short(&self.source_id))
            .field("sequence", &self.sequence)
            .field("phi", &self.phi)
            .field("urgency", &self.urgency)
            .field("payload_type", &self.payload_type)
            .finish()
    }
}

/// Format a byte slice as short hex for debug output.
pub fn hex_short(bytes: &[u8]) -> String {
    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

// ============================================================================
// MESH PEER REGISTRY
// ============================================================================

/// Maximum packets allowed per peer within the rate limit window.
const MESH_RATE_LIMIT_MAX: u64 = 100;
/// Duration of the per-peer rate limiting window.
const MESH_RATE_LIMIT_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// Tracked state for a single mesh peer (distinct from [`MeshPeer`] which
/// tracks LoRa fragment reassembly statistics).
#[derive(Debug, Clone)]
pub struct MeshPeerEntry {
    /// First 8 bytes of the peer's node identity.
    pub source_id: [u8; 8],
    /// Last time we received a packet from this peer.
    pub last_seen: std::time::Instant,
    /// Total packets received from this peer.
    pub packets_received: u64,
    /// Most recent Phi value reported by this peer.
    pub last_phi: f32,
    /// Most recent urgency level from this peer.
    pub last_urgency: MeshUrgency,
    /// Most recent payload type from this peer.
    pub last_payload_type: PayloadType,
    /// Start of the current rate-limiting window.
    pub window_start: std::time::Instant,
    /// Packet count within the current rate-limiting window.
    pub window_count: u64,
}

/// Registry of active mesh peers, updated each tick from incoming packets.
///
/// Provides swarm awareness: "which peers are alive?", "what's the swarm's
/// average phi?", etc. Stale peers are expired after `stale_timeout`.
pub struct MeshPeerRegistry {
    peers: std::collections::HashMap<[u8; 8], MeshPeerEntry>,
    stale_timeout: std::time::Duration,
}

impl MeshPeerRegistry {
    /// Create a new registry with the default 60-second stale timeout.
    pub fn new() -> Self {
        Self {
            peers: std::collections::HashMap::new(),
            stale_timeout: std::time::Duration::from_secs(60),
        }
    }

    /// Create a new registry with a custom stale timeout (useful for tests).
    pub fn with_timeout(timeout: std::time::Duration) -> Self {
        Self {
            peers: std::collections::HashMap::new(),
            stale_timeout: timeout,
        }
    }

    /// Update the registry with a received packet.
    pub fn update(&mut self, packet: &WisdomPacket) {
        let entry = self
            .peers
            .entry(packet.source_id)
            .or_insert_with(|| MeshPeerEntry {
                source_id: packet.source_id,
                last_seen: std::time::Instant::now(),
                packets_received: 0,
                last_phi: packet.phi,
                last_urgency: packet.urgency,
                last_payload_type: packet.payload_type,
                window_start: std::time::Instant::now(),
                window_count: 0,
            });
        entry.last_seen = std::time::Instant::now();
        entry.packets_received += 1;
        entry.last_phi = packet.phi;
        entry.last_urgency = packet.urgency;
        entry.last_payload_type = packet.payload_type;
    }

    /// Remove peers not seen within the stale timeout.
    ///
    /// Returns the `source_id`s of expired peers (empty if none expired).
    pub fn expire_stale(&mut self) -> Vec<[u8; 8]> {
        let cutoff = std::time::Instant::now() - self.stale_timeout;
        let expired: Vec<[u8; 8]> = self
            .peers
            .iter()
            .filter(|(_, entry)| entry.last_seen <= cutoff)
            .map(|(id, _)| *id)
            .collect();
        for id in &expired {
            self.peers.remove(id);
        }
        expired
    }

    /// Get all active (non-expired) peers.
    pub fn active_peers(&self) -> Vec<&MeshPeerEntry> {
        self.peers.values().collect()
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Check if a peer has exceeded the rate limit.
    ///
    /// Returns `true` if the peer's packet count within the current window
    /// exceeds [`MESH_RATE_LIMIT_MAX`]. Resets the window if the previous
    /// one has elapsed.
    pub fn is_rate_limited(&mut self, source_id: &[u8; 8]) -> bool {
        let entry = match self.peers.get_mut(source_id) {
            Some(e) => e,
            None => return false, // Unknown peer — not rate limited
        };
        let now = std::time::Instant::now();
        if now.duration_since(entry.window_start) >= MESH_RATE_LIMIT_WINDOW {
            entry.window_start = now;
            entry.window_count = 0;
        }
        entry.window_count += 1;
        entry.window_count > MESH_RATE_LIMIT_MAX
    }

    /// Average Phi across all tracked peers. Returns 0.0 if no peers.
    pub fn average_phi(&self) -> f32 {
        if self.peers.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.peers.values().map(|p| p.last_phi).sum();
        sum / self.peers.len() as f32
    }

    /// Check if a peer with the given source_id is currently tracked.
    pub fn has_peer(&self, source_id: &[u8; 8]) -> bool {
        self.peers.contains_key(source_id)
    }

    /// Detect if the mesh appears partitioned.
    ///
    /// Returns `true` if we previously had peers but now have zero
    /// (all expired), suggesting a network partition rather than normal
    /// peer departure. Returns `false` if we never had peers or still have some.
    pub fn is_partitioned(&self, stats: &MeshStats) -> bool {
        self.peer_count() == 0 && stats.peers_expired > 0 && stats.wisdom_received > 0
    }

    /// Count peers that haven't been seen within the given duration.
    pub fn stale_peer_count(&self, timeout: std::time::Duration) -> usize {
        let now = std::time::Instant::now();
        self.peers
            .values()
            .filter(|p| now.duration_since(p.last_seen) > timeout)
            .count()
    }
}

impl Default for MeshPeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PACKET AUTHENTICATION (HMAC-SHA-256)
// ============================================================================

/// Compute an untruncated HMAC-SHA-256 tag over the packet bytes.
///
/// The tag is computed with its own 32-byte field zeroed,
/// so the MAC doesn't include itself in the hash input.
pub fn compute_packet_mac(
    packet_bytes: &[u8; WISDOM_PACKET_SIZE],
    key: &[u8; 32],
) -> [u8; WISDOM_PACKET_AUTH_TAG_SIZE] {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let mut input = *packet_bytes;
    input[WISDOM_PACKET_AUTH_TAG_START..WISDOM_PACKET_AUTH_TAG_END].fill(0);
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts keys of any length");
    mac.update(&input);
    mac.finalize().into_bytes().into()
}

/// Verify the MAC on a packet byte slice.
///
/// Returns `true` if the MAC matches, `false` otherwise.
/// Returns `false` unless the slice is exactly one version-2 packet.
pub fn verify_packet_mac(packet_bytes: &[u8], key: &[u8; 32]) -> bool {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    if packet_bytes.len() != WISDOM_PACKET_SIZE || packet_bytes[0] != WISDOM_PACKET_VERSION {
        return false;
    }
    let mut input = [0u8; WISDOM_PACKET_SIZE];
    input.copy_from_slice(&packet_bytes[..WISDOM_PACKET_SIZE]);
    let mut stored_mac = [0u8; WISDOM_PACKET_AUTH_TAG_SIZE];
    stored_mac.copy_from_slice(&input[WISDOM_PACKET_AUTH_TAG_START..WISDOM_PACKET_AUTH_TAG_END]);
    input[WISDOM_PACKET_AUTH_TAG_START..WISDOM_PACKET_AUTH_TAG_END].fill(0);
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts keys of any length");
    mac.update(&input);
    mac.verify_slice(&stored_mac).is_ok()
}

// ============================================================================
// PACKET COMPRESSION (LZ4)
// ============================================================================

/// Compress a raw WISDOM_PACKET_SIZE packet into an envelope.
///
/// Returns `[COMPRESS_NONE | raw]` if compression doesn't help (output >= input),
/// or `[COMPRESS_LZ4 | lz4_data]` if compression reduces size.
///
/// When `lz4_compression` feature is disabled, always returns uncompressed.
pub fn compress_packet(raw: &[u8; WISDOM_PACKET_SIZE]) -> Vec<u8> {
    #[cfg(feature = "lz4_compression")]
    {
        let compressed = lz4_flex::compress_prepend_size(raw);
        if compressed.len() + 1 < WISDOM_PACKET_SIZE + 1 {
            let mut envelope = Vec::with_capacity(1 + compressed.len());
            envelope.push(COMPRESS_LZ4);
            envelope.extend_from_slice(&compressed);
            return envelope;
        }
    }
    // Uncompressed fallback (or feature disabled)
    let mut envelope = Vec::with_capacity(1 + WISDOM_PACKET_SIZE);
    envelope.push(COMPRESS_NONE);
    envelope.extend_from_slice(raw);
    envelope
}

/// Decompress a packet envelope back to WISDOM_PACKET_SIZE bytes.
///
/// Returns `None` if the header byte is unknown or decompression fails.
/// Tolerates trailing bytes (FEC safety).
pub fn decompress_packet(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    match data[0] {
        COMPRESS_NONE => {
            if data.len() < 1 + WISDOM_PACKET_SIZE {
                return None;
            }
            Some(data[1..1 + WISDOM_PACKET_SIZE].to_vec())
        }
        COMPRESS_LZ4 => {
            #[cfg(feature = "lz4_compression")]
            {
                lz4_flex::decompress_size_prepended(&data[1..])
                    .ok()
                    .filter(|d| d.len() == WISDOM_PACKET_SIZE)
            }
            #[cfg(not(feature = "lz4_compression"))]
            {
                let _ = data;
                None
            }
        }
        _ => None,
    }
}

// ============================================================================
// ENCRYPTION (ChaCha20-Poly1305)
// ============================================================================

/// Nonce size for ChaCha20-Poly1305 (96 bits).
#[cfg(feature = "mesh-encryption")]
pub const AEAD_NONCE_SIZE: usize = 12;

/// Authentication tag size (128 bits).
#[cfg(feature = "mesh-encryption")]
pub const AEAD_TAG_SIZE: usize = 16;

/// Build a 12-byte ChaCha20-Poly1305 nonce with type separation and epoch.
///
/// Layout: `source_id[0..6] | payload_type | epoch | sequence[0..4]`
///
/// - **payload_type** prevents nonce collision across wisdom/heartbeat/affective/gradient
///   sequences (all start at 0 but use different type bytes).
/// - **epoch** is a random byte generated once at Mind construction, preventing
///   restart nonce reuse under the same key.
/// - **sequence** is per-type monotonic (wraps safely at 2^32 ≈ 2.7 years at 50Hz).
#[cfg(feature = "mesh-encryption")]
pub fn build_nonce(source_id: &[u8; 8], payload_type: u8, epoch: u8, sequence: u32) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[..6].copy_from_slice(&source_id[..6]);
    nonce[6] = payload_type;
    nonce[7] = epoch;
    nonce[8..12].copy_from_slice(&sequence.to_le_bytes());
    nonce
}

/// Encrypt a compressed packet envelope using ChaCha20-Poly1305.
///
/// Returns `[nonce (12 bytes) | ciphertext+tag]`.
/// The nonce includes payload_type and epoch to prevent cross-type
/// and restart nonce reuse. See [`build_nonce`].
///
/// **`epoch`** must be a random byte generated once at node startup
/// (via `rand::thread_rng().r#gen::<u8>()`). This prevents nonce reuse
/// across node restarts when using the same key material.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet(
    envelope: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    epoch: u8,
    sequence: u32,
) -> Vec<u8> {
    encrypt_packet_typed(envelope, key, source_id, 0, epoch, sequence)
}

/// Encrypt with explicit payload_type and epoch (nonce-safe variant).
///
/// Prefer this over [`encrypt_packet`] to prevent nonce reuse across
/// packet types and across node restarts.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet_typed(
    envelope: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    payload_type: u8,
    epoch: u8,
    sequence: u32,
) -> Vec<u8> {
    match try_encrypt_packet_typed(envelope, key, source_id, payload_type, epoch, sequence) {
        Ok(out) => out,
        Err(e) => {
            // This should never happen with valid inputs, but if it does,
            // log and return empty (safer than panic in production).
            tracing::error!("ChaCha20-Poly1305 encryption failed: {e}");
            Vec::new()
        }
    }
}

/// Fallible variant of [`encrypt_packet_typed`] that returns `Result`.
///
/// Prefer this when the caller can handle encryption failures gracefully
/// (e.g., skip the packet rather than crash).
#[cfg(feature = "mesh-encryption")]
pub fn try_encrypt_packet_typed(
    envelope: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    payload_type: u8,
    epoch: u8,
    sequence: u32,
) -> Result<Vec<u8>, crate::swarm::SwarmError> {
    use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce_bytes = build_nonce(source_id, payload_type, epoch, sequence);
    let nonce = Nonce::from(nonce_bytes);
    let ciphertext = cipher.encrypt(&nonce, envelope).map_err(|e| {
        crate::swarm::SwarmError::EncryptionFailed {
            reason: format!("ChaCha20-Poly1305: {e}"),
        }
    })?;
    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Encrypt with a 1-byte key version prefix for versioned decrypt.
///
/// Wire format: `[key_version (1) | nonce (12) | ciphertext+tag]`
/// The version byte lets the receiver select the correct key without
/// blind trial decryption during key rotation grace periods.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet_versioned(
    envelope: &[u8],
    key: &[u8; 32],
    key_version: u8,
    source_id: &[u8; 8],
    payload_type: u8,
    epoch: u8,
    sequence: u32,
) -> Vec<u8> {
    let inner = encrypt_packet_typed(envelope, key, source_id, payload_type, epoch, sequence);
    let mut out = Vec::with_capacity(1 + inner.len());
    out.push(key_version);
    out.extend_from_slice(&inner);
    out
}

/// Decrypt a packet encrypted with `encrypt_packet`.
///
/// Returns `None` if decryption/authentication fails.
#[cfg(feature = "mesh-encryption")]
pub fn decrypt_packet(data: &[u8], key: &[u8; 32]) -> Option<Vec<u8>> {
    use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};
    if data.len() < AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
        return None;
    }
    let (nonce_bytes, ciphertext) = data.split_at(AEAD_NONCE_SIZE);
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext).ok()
}

// ============================================================================
// XChaCha20-Poly1305 (NONCE-MISUSE RESISTANT)
// ============================================================================

/// Nonce size for XChaCha20-Poly1305 (192 bits).
#[cfg(feature = "mesh-encryption")]
pub const XCHACHA_NONCE_SIZE: usize = 24;

/// Encrypt using XChaCha20-Poly1305 with a random 24-byte nonce.
///
/// XChaCha20 uses a 192-bit nonce large enough for random generation
/// without collision risk (birthday bound ≈ 2^96). This eliminates
/// nonce-reuse concerns entirely at the cost of 12 extra nonce bytes per packet.
///
/// Returns `[nonce (24 bytes) | ciphertext+tag]`.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet_xchacha(envelope: &[u8], key: &[u8; 32]) -> Vec<u8> {
    use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce, aead::Aead};
    let cipher = XChaCha20Poly1305::new(key.into());
    let mut nonce_bytes = [0u8; 24];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_bytes);
    let nonce = XNonce::from(nonce_bytes);
    let ciphertext = cipher
        .encrypt(&nonce, envelope)
        .expect("encryption never fails for valid inputs");
    let mut out = Vec::with_capacity(24 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

/// Decrypt a packet encrypted with [`encrypt_packet_xchacha`].
///
/// Returns `None` if decryption/authentication fails.
#[cfg(feature = "mesh-encryption")]
pub fn decrypt_packet_xchacha(data: &[u8], key: &[u8; 32]) -> Option<Vec<u8>> {
    use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce, aead::Aead};
    if data.len() < XCHACHA_NONCE_SIZE + AEAD_TAG_SIZE {
        return None;
    }
    let (nonce_bytes, ciphertext) = data.split_at(XCHACHA_NONCE_SIZE);
    let cipher = XChaCha20Poly1305::new(key.into());
    let nonce = XNonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext).ok()
}

// ============================================================================
// KEY ROTATION
// ============================================================================

/// Manages encryption key rotation with a grace period.
///
/// During rotation, both the old and new keys are tried for decryption.
/// After the grace period expires, only the new key is accepted.
///
/// ```text
/// rotate_key(new_key)
///   ├── grace period (accepts old OR new) ──► grace expires
///   │                                          ├── old_key = None
///   │                                          └── only new_key accepted
/// ```
#[cfg(feature = "mesh-encryption")]
#[derive(Clone, zeroize::ZeroizeOnDrop)]
pub struct RotatingKeyPair {
    /// Current (primary) encryption key — used for all outbound packets.
    current: [u8; 32],
    /// Previous key — accepted for inbound during grace period, then discarded.
    /// NOT skipped by zeroize — key material must be wiped on drop.
    previous: Option<[u8; 32]>,
    /// Tick at which the previous key expires (absolute tick count).
    #[zeroize(skip)]
    grace_expires_at: u64,
    /// Key version — incremented on every rotation, embedded in ciphertext header.
    /// Wraps at u8::MAX. Prevents nonce reuse across key rotations.
    #[zeroize(skip)]
    key_version: u8,
    /// Version of the previous key (for versioned decrypt hints).
    #[zeroize(skip)]
    previous_version: u8,
    /// Random epoch byte generated once at construction. Prevents nonce reuse
    /// across node restarts under the same key material (P0 security fix).
    #[zeroize(skip)]
    epoch: u8,
    /// Optional sensor context key overlay. When set, the effective encryption
    /// key is `current XOR context_overlay` — binding mesh encryption to the
    /// physical environment (location, light, motion, altitude).
    /// Both peers must independently derive the same context key.
    context_overlay: Option<[u8; 32]>,
}

#[cfg(feature = "mesh-encryption")]
impl RotatingKeyPair {
    /// Create a new key pair with a single key (no rotation in progress).
    ///
    /// Generates a random epoch byte to prevent nonce reuse across restarts.
    pub fn new(key: [u8; 32]) -> Self {
        Self {
            current: key,
            previous: None,
            grace_expires_at: 0,
            key_version: 0,
            previous_version: 0,
            epoch: rand::Rng::r#gen(&mut rand::thread_rng()),
            context_overlay: None,
        }
    }

    /// Rotate to a new key. The old key remains valid for `grace_ticks` cycles.
    ///
    /// Outbound packets immediately use the new key. Inbound packets accept
    /// either key until the grace period expires. The key version is incremented
    /// to prevent nonce reuse across rotations.
    pub fn rotate(&mut self, new_key: [u8; 32], current_tick: u64, grace_ticks: u64) {
        self.previous = Some(self.current);
        self.previous_version = self.key_version;
        self.key_version = self.key_version.wrapping_add(1);
        self.current = new_key;
        self.grace_expires_at = current_tick.saturating_add(grace_ticks);
    }

    /// The current key version (embedded in ciphertext header).
    pub fn key_version(&self) -> u8 {
        self.key_version
    }

    /// Expire the old key if the grace period has elapsed.
    ///
    /// Call this once per tick to garbage-collect the old key.
    pub fn tick(&mut self, current_tick: u64) {
        if self.previous.is_some() && current_tick >= self.grace_expires_at {
            self.previous = None;
        }
    }

    /// The current key (used for encryption).
    pub fn current_key(&self) -> &[u8; 32] {
        &self.current
    }

    /// Whether a rotation is in progress (grace period active).
    pub fn is_rotating(&self) -> bool {
        self.previous.is_some()
    }

    /// Encrypt with the current key using the stored random epoch.
    ///
    /// If a sensor context overlay is set, the effective key is
    /// `current XOR context_overlay` — physically binding the encryption.
    pub fn encrypt(&self, envelope: &[u8], source_id: &[u8; 8], sequence: u32) -> Vec<u8> {
        let key = self.effective_key();
        encrypt_packet(envelope, &key, source_id, self.epoch, sequence)
    }

    /// Get the epoch byte (useful for logging/telemetry).
    pub fn epoch(&self) -> u8 {
        self.epoch
    }

    /// Set the sensor context key overlay.
    ///
    /// When set, all encryption/decryption uses `base_key XOR context_overlay`
    /// as the effective key. Both peers must independently derive the same
    /// context key from their sensor readings (same location, similar conditions).
    ///
    /// Call with `None` to disable context binding.
    pub fn set_context_overlay(&mut self, overlay: Option<[u8; 32]>) {
        self.context_overlay = overlay;
    }

    /// Whether a sensor context overlay is active.
    pub fn has_context_overlay(&self) -> bool {
        self.context_overlay.is_some()
    }

    /// Compute the effective encryption key (base XOR context overlay).
    fn effective_key(&self) -> [u8; 32] {
        match &self.context_overlay {
            Some(overlay) => {
                let mut key = self.current;
                for (k, o) in key.iter_mut().zip(overlay.iter()) {
                    *k ^= o;
                }
                key
            }
            None => self.current,
        }
    }

    /// Encrypt with the current key, using typed nonce for cross-type safety.
    ///
    /// Prepends a 1-byte key version header for versioned decrypt hints.
    pub fn encrypt_typed(
        &self,
        envelope: &[u8],
        source_id: &[u8; 8],
        payload_type: u8,
        epoch: u8,
        sequence: u32,
    ) -> Vec<u8> {
        encrypt_packet_versioned(
            envelope,
            &self.current,
            self.key_version,
            source_id,
            payload_type,
            epoch,
            sequence,
        )
    }

    /// Decrypt trying versioned format first, then unversioned for backward compat.
    ///
    /// Versioned format: `[key_version (1) | nonce (12) | ciphertext+tag]`
    /// - If version matches current → try current key only
    /// - If version matches previous → try previous key only
    /// - Otherwise → try both (backward compat with pre-versioned packets)
    pub fn decrypt(&self, data: &[u8]) -> Option<Vec<u8>> {
        // Need at least 1 (version) + 12 (nonce) + 16 (tag) = 29 bytes for versioned
        if data.len() >= 1 + AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
            let version_hint = data[0];
            let inner = &data[1..];

            if version_hint == self.key_version {
                // Matches current — try only current key
                if let Some(pt) = decrypt_packet(inner, &self.current) {
                    return Some(pt);
                }
            } else if version_hint == self.previous_version {
                // Matches previous — try only previous key
                if let Some(ref prev) = self.previous {
                    if let Some(pt) = decrypt_packet(inner, prev) {
                        return Some(pt);
                    }
                }
            }
        }

        // Fall back to unversioned (backward compat with pre-versioned packets)
        if let Some(plaintext) = decrypt_packet(data, &self.current) {
            return Some(plaintext);
        }
        if let Some(ref prev) = self.previous {
            return decrypt_packet(data, prev);
        }
        None
    }
}

// ============================================================================
// X25519 PER-PEER KEY AGREEMENT
// ============================================================================

/// Per-peer key store: X25519 Diffie-Hellman → BLAKE3 KDF → ChaCha20 key.
///
/// Each peer pair derives a unique symmetric key from their DH shared secret.
/// This provides forward secrecy (compromising one peer's key doesn't reveal
/// other pairs' traffic) and eliminates the single shared secret.
///
/// ```text
/// Node A (secret_a)                    Node B (secret_b)
///   │                                    │
///   ├─ public_a = X25519(secret_a) ─────►│
///   │◄──── public_b = X25519(secret_b) ──┤
///   │                                    │
///   ├─ shared = DH(secret_a, public_b)   ├─ shared = DH(secret_b, public_a)
///   ├─ key = BLAKE3(shared ‖ ctx)        ├─ key = BLAKE3(shared ‖ ctx)
///   └─ (same key on both sides)          └─ (same key on both sides)
/// ```
#[cfg(feature = "mesh-key-exchange")]
pub struct PeerKeyStore {
    /// Our long-term X25519 secret key.
    secret: x25519_dalek::StaticSecret,
    /// Our public key (derived from secret).
    public: x25519_dalek::PublicKey,
    /// Derived symmetric keys per peer, keyed by source_id.
    peer_keys: std::collections::HashMap<[u8; 8], [u8; 32]>,
    /// Derived authentication keys per peer (separate from encryption keys).
    peer_auth_keys: std::collections::HashMap<[u8; 8], [u8; 32]>,
}

#[cfg(feature = "mesh-key-exchange")]
impl PeerKeyStore {
    /// Create a new key store with a random X25519 keypair.
    pub fn new(secret_bytes: [u8; 32]) -> Self {
        let secret = x25519_dalek::StaticSecret::from(secret_bytes);
        let public = x25519_dalek::PublicKey::from(&secret);
        Self {
            secret,
            public,
            peer_keys: std::collections::HashMap::new(),
            peer_auth_keys: std::collections::HashMap::new(),
        }
    }

    /// Our X25519 public key (32 bytes, send to peers).
    pub fn public_key(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    /// Perform DH key agreement with a peer's public key.
    ///
    /// Derives two symmetric keys via HKDF-SHA256:
    /// - Encryption key (info: `symthaea-mesh-chacha20-v1`)
    /// - Authentication key (info: `symthaea-mesh-auth-v1`)
    ///
    /// Returns the encryption key.
    pub fn agree(&mut self, peer_source_id: [u8; 8], peer_public: &[u8; 32]) -> [u8; 32] {
        let peer_pk = x25519_dalek::PublicKey::from(*peer_public);
        let shared_secret = self.secret.diffie_hellman(&peer_pk);

        // KDF: HKDF-SHA256 with domain-separated info strings
        let hk = hkdf::Hkdf::<sha2::Sha256>::new(None, shared_secret.as_bytes());

        let mut chacha_key = [0u8; 32];
        hk.expand(b"symthaea-mesh-chacha20-v1", &mut chacha_key)
            .expect("32-byte output is valid for HKDF-SHA256");

        let mut auth_key = [0u8; 32];
        hk.expand(b"symthaea-mesh-auth-v1", &mut auth_key)
            .expect("32-byte output is valid for HKDF-SHA256");

        self.peer_keys.insert(peer_source_id, chacha_key);
        self.peer_auth_keys.insert(peer_source_id, auth_key);
        chacha_key
    }

    /// Get the derived symmetric key for a peer (if agreement has been done).
    pub fn peer_key(&self, source_id: &[u8; 8]) -> Option<&[u8; 32]> {
        self.peer_keys.get(source_id)
    }

    /// Get the derived authentication key for a peer.
    pub fn peer_auth_key(&self, source_id: &[u8; 8]) -> Option<&[u8; 32]> {
        self.peer_auth_keys.get(source_id)
    }

    /// Number of peers with established keys.
    pub fn peer_count(&self) -> usize {
        self.peer_keys.len()
    }

    /// Remove a peer's keys (both encryption and auth).
    pub fn remove_peer(&mut self, source_id: &[u8; 8]) {
        self.peer_keys.remove(source_id);
        self.peer_auth_keys.remove(source_id);
    }
}

#[cfg(feature = "mesh-key-exchange")]
impl Drop for PeerKeyStore {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        for key in self.peer_keys.values_mut() {
            key.zeroize();
        }
        for key in self.peer_auth_keys.values_mut() {
            key.zeroize();
        }
    }
}

// ============================================================================
// FRAGMENT-LEVEL AEAD ENCRYPTION
// ============================================================================

/// Encrypt a single LoRa fragment with ChaCha20-Poly1305.
///
/// Each fragment gets its own AEAD envelope so that:
/// 1. Individual fragments are authenticated (tamper-proof)
/// 2. A partial capture (< all fragments) cannot be decrypted
/// 3. Fragment reordering is detected
///
/// Nonce derivation: `source_id[0..8] ‖ thought_id[2] ‖ fragment_index[1] ‖ 0[1]`
///
/// Returns `[nonce (12) | ciphertext + tag]`. Overhead: 28 bytes per fragment.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_fragment(
    payload: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    thought_id: u16,
    fragment_index: u8,
) -> Vec<u8> {
    use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};
    let cipher = ChaCha20Poly1305::new(key.into());
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes[..8].copy_from_slice(source_id);
    nonce_bytes[8..10].copy_from_slice(&thought_id.to_le_bytes());
    nonce_bytes[10] = fragment_index;
    nonce_bytes[11] = 0; // reserved
    let nonce = Nonce::from(nonce_bytes);
    let ciphertext = match cipher.encrypt(&nonce, payload) {
        Ok(ct) => ct,
        Err(e) => {
            tracing::error!("Fragment encryption failed: {e}");
            return Vec::new();
        }
    };
    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

/// Decrypt a fragment encrypted with [`encrypt_fragment`].
///
/// Returns `None` if decryption/authentication fails.
#[cfg(feature = "mesh-encryption")]
pub fn decrypt_fragment(data: &[u8], key: &[u8; 32]) -> Option<Vec<u8>> {
    use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};
    if data.len() < AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
        return None;
    }
    let (nonce_bytes, ciphertext) = data.split_at(AEAD_NONCE_SIZE);
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext).ok()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;
