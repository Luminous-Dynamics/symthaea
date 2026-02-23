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
//! │  │  WisdomPacket (2,072 bytes)                              │  │
//! │  │  ├─ Metadata:  24 bytes (source, phi, urgency, time)    │  │
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

mod dual_layer;
mod lora_fragment;
mod mesh_receiver;
pub mod sensor;

pub use dual_layer::{
    BiLoopbackTransport, DualLayerMesh, LoopbackTransport, MeshRoute, MeshTransport,
};
pub use lora_fragment::{
    crc16_ccitt, fragment, FragmentAssembler, LoRaFragment, FLAG_FEC, HEADER_SIZE, LORA_MTU,
    PAYLOAD_SIZE,
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

pub use bridge::{MeshBridgeActor, MeshBridgeHandle, MeshOutbound};
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
    /// Wisdom vectors received from mesh peers.
    pub wisdom_received: u64,
    /// Heartbeat packets received from mesh peers.
    pub heartbeats_received: u64,
    /// Affective packets received from mesh peers.
    pub affective_received: u64,
    /// Gradient packets received from mesh peers.
    pub gradients_received: u64,
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
}

impl MeshStats {
    /// Total packets sent across all types.
    fn total_sent(&self) -> u64 {
        self.wisdom_sent + self.heartbeats_sent + self.affective_sent + self.gradients_sent
    }

    /// Total packets received across all types.
    fn total_received(&self) -> u64 {
        self.wisdom_received
            + self.heartbeats_received
            + self.affective_received
            + self.gradients_received
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
        let stability =
            1.0 - (self.peers_expired as f32 / (self.peers_expired + recv + 1) as f32);

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
}

impl PayloadType {
    /// Decode from a single byte (radio wire format).
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::WisdomVector,
            1 => Self::Affective,
            2 => Self::Heartbeat,
            _ => Self::Gradient,
        }
    }

    /// Backpressure priority: higher values are retained first when inbox is full.
    /// Heartbeat(3) > Wisdom(2) > Affective(1) > Gradient(0).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Heartbeat => 3,
            Self::WisdomVector => 2,
            Self::Affective => 1,
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

/// Wire size of a [`WisdomPacket`]: 24 bytes metadata + 2,048 bytes BinaryHV.
pub const WISDOM_PACKET_SIZE: usize = 24 + 2048; // 2,072

/// A Wisdom Packet: the atomic unit of consciousness exchange over mesh radio.
///
/// Contains a full [`BinaryHV`] (2,048 bytes = 16,384 dimensions) plus
/// minimal metadata. Total wire size: 2,072 bytes, fragmenting into exactly
/// 10 data frames + 1 FEC parity = 11 LoRa transmissions.
///
/// # Wire Format
///
/// ```text
/// Byte     Field           Type
/// 0-7      source_id       [u8; 8]    truncated node identity
/// 8-11     sequence        u32 LE     monotonic counter
/// 12-15    phi             f32 LE     integrated information (Phi)
/// 16       urgency         u8         MeshUrgency discriminant
/// 17-20    timestamp_s     u32 LE     Unix seconds
/// 21       payload_type    u8         PayloadType discriminant
/// 22       auth_mac        u8         BLAKE3 keyed MAC (truncated to 8 bits)
/// 23       ttl             u8         gossip hop count (0 = no forward)
/// 24-2071  wisdom          [u8; 2048] BinaryHV raw bytes
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
    /// BLAKE3 keyed MAC (truncated to 8 bits). 0 when no auth key is set.
    pub auth_mac: u8,
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
        buf[0..8].copy_from_slice(&self.source_id);
        buf[8..12].copy_from_slice(&self.sequence.to_le_bytes());
        buf[12..16].copy_from_slice(&self.phi.to_le_bytes());
        buf[16] = self.urgency as u8;
        buf[17..21].copy_from_slice(&self.timestamp_s.to_le_bytes());
        buf[21] = self.payload_type as u8;
        buf[22] = self.auth_mac;
        buf[23] = self.ttl;
        buf[24..WISDOM_PACKET_SIZE].copy_from_slice(&self.wisdom.0);
        buf
    }

    /// Deserialize from a byte slice.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < WISDOM_PACKET_SIZE {
            return None;
        }

        let mut source_id = [0u8; 8];
        source_id.copy_from_slice(&buf[0..8]);

        let sequence = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let phi = f32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let urgency = MeshUrgency::from_byte(buf[16]);
        let timestamp_s = u32::from_le_bytes([buf[17], buf[18], buf[19], buf[20]]);
        let payload_type = PayloadType::from_byte(buf[21]);
        let auth_mac = buf[22];
        let ttl = buf[23];

        let mut wisdom_bytes = [0u8; 2048];
        wisdom_bytes.copy_from_slice(&buf[24..WISDOM_PACKET_SIZE]);

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
            auth_mac: 0,
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
            auth_mac: 0,
            ttl: MESH_DEFAULT_TTL,
            wisdom: BinaryHV(bytes),
        })
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
    /// Returns 11 fragments: 10 data (carrying the 2,072 byte payload) +
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
}

impl Default for MeshPeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PACKET AUTHENTICATION (BLAKE3)
// ============================================================================

/// Compute a truncated 8-bit BLAKE3 keyed MAC over the packet bytes.
///
/// The MAC is computed with byte 22 (the auth_mac field itself) zeroed,
/// so the MAC doesn't include itself in the hash input.
pub fn compute_packet_mac(packet_bytes: &[u8; WISDOM_PACKET_SIZE], key: &[u8; 32]) -> u8 {
    let mut input = *packet_bytes;
    input[22] = 0; // Zero the auth_mac field before computing
    let hash = blake3::keyed_hash(key, &input);
    hash.as_bytes()[0]
}

/// Verify the MAC on a packet byte slice.
///
/// Returns `true` if the MAC matches, `false` otherwise.
/// Returns `false` if the slice is too short.
pub fn verify_packet_mac(packet_bytes: &[u8], key: &[u8; 32]) -> bool {
    if packet_bytes.len() < WISDOM_PACKET_SIZE {
        return false;
    }
    let mut input = [0u8; WISDOM_PACKET_SIZE];
    input.copy_from_slice(&packet_bytes[..WISDOM_PACKET_SIZE]);
    let stored_mac = input[22];
    input[22] = 0;
    let hash = blake3::keyed_hash(key, &input);
    hash.as_bytes()[0] == stored_mac
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
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hv(seed: u8) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = seed.wrapping_mul(i as u8).wrapping_add((i >> 3) as u8);
        }
        BinaryHV(bytes)
    }

    #[test]
    fn wisdom_packet_roundtrip() {
        let packet = WisdomPacket {
            source_id: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            sequence: 42,
            phi: 0.73,
            urgency: MeshUrgency::Cruise,
            timestamp_s: 1_700_000_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0xFF),
        };

        let bytes = packet.to_bytes();
        assert_eq!(bytes.len(), WISDOM_PACKET_SIZE);

        let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.source_id, packet.source_id);
        assert_eq!(decoded.sequence, 42);
        assert!((decoded.phi - 0.73).abs() < 1e-6);
        assert_eq!(decoded.urgency, MeshUrgency::Cruise);
        assert_eq!(decoded.timestamp_s, 1_700_000_000);
        assert_eq!(decoded.payload_type, PayloadType::WisdomVector);
        assert_eq!(decoded.wisdom.0, packet.wisdom.0);
    }

    #[test]
    fn wisdom_packet_fragment_count() {
        let packet = WisdomPacket {
            source_id: [0; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0xAA),
        };

        let frags = packet.fragment();
        // 2072 / 214 = 9.68 → 10 data + 1 FEC = 11
        assert_eq!(frags.len(), 11);
    }

    #[test]
    fn wisdom_packet_full_radio_roundtrip() {
        let original = WisdomPacket {
            source_id: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            sequence: 1337,
            phi: 0.91,
            urgency: MeshUrgency::Critical,
            timestamp_s: 1_708_000_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x42),
        };

        // Fragment
        let frags = original.fragment();

        // Simulate radio: serialize, drop fragment 6, deserialize
        let mut assembler = WisdomPacket::assembler(original.thought_id(), 11);
        let mut buf = [0u8; LORA_MTU];

        for (i, frag) in frags.iter().enumerate() {
            if i == 6 {
                continue; // lost in transit
            }
            let len = frag.to_bytes(&mut buf);
            let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
            assembler.feed(&decoded);
        }

        assert!(assembler.is_complete());
        let recovered = WisdomPacket::from_assembler(&assembler).unwrap();

        assert_eq!(recovered.source_id, original.source_id);
        assert_eq!(recovered.sequence, 1337);
        assert!((recovered.phi - 0.91).abs() < 1e-6);
        assert_eq!(recovered.urgency, MeshUrgency::Critical);
        assert_eq!(recovered.wisdom.0, original.wisdom.0);
    }

    #[test]
    fn mesh_urgency_byte_roundtrip() {
        assert_eq!(MeshUrgency::from_byte(0), MeshUrgency::Cruise);
        assert_eq!(MeshUrgency::from_byte(1), MeshUrgency::Normal);
        assert_eq!(MeshUrgency::from_byte(2), MeshUrgency::Critical);
        assert_eq!(MeshUrgency::from_byte(255), MeshUrgency::Critical);
    }

    #[test]
    fn payload_type_byte_roundtrip() {
        assert_eq!(PayloadType::from_byte(0), PayloadType::WisdomVector);
        assert_eq!(PayloadType::from_byte(1), PayloadType::Affective);
        assert_eq!(PayloadType::from_byte(2), PayloadType::Heartbeat);
        assert_eq!(PayloadType::from_byte(3), PayloadType::Gradient);
    }

    #[test]
    fn wisdom_packet_too_short_rejected() {
        assert!(WisdomPacket::from_bytes(&[0; 100]).is_none());
        assert!(WisdomPacket::from_bytes(&[0; WISDOM_PACKET_SIZE - 1]).is_none());
    }

    #[test]
    fn cycle_urgency_to_mesh_urgency() {
        use crate::cognitive_loop::types::CycleUrgency;

        assert_eq!(
            MeshUrgency::from(CycleUrgency::Critical),
            MeshUrgency::Critical
        );
        assert_eq!(MeshUrgency::from(CycleUrgency::Normal), MeshUrgency::Normal);
        assert_eq!(MeshUrgency::from(CycleUrgency::Cruise), MeshUrgency::Cruise);
    }

    // ====================================================================
    // extract_affective Tests
    // ====================================================================

    /// Build an Affective WisdomPacket with specific VAD floats in the wisdom field.
    fn affective_packet(
        valence: f32,
        arousal: f32,
        dominance: f32,
        intensity: f32,
        confidence: f32,
    ) -> WisdomPacket {
        let mut wisdom_bytes = [0u8; 2048];
        wisdom_bytes[0..4].copy_from_slice(&valence.to_le_bytes());
        wisdom_bytes[4..8].copy_from_slice(&arousal.to_le_bytes());
        wisdom_bytes[8..12].copy_from_slice(&dominance.to_le_bytes());
        wisdom_bytes[12..16].copy_from_slice(&intensity.to_le_bytes());
        wisdom_bytes[16..20].copy_from_slice(&confidence.to_le_bytes());
        WisdomPacket {
            source_id: [0xAF; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::Affective,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV(wisdom_bytes),
        }
    }

    #[test]
    fn test_extract_affective_valid() {
        let pkt = affective_packet(0.5, 0.7, -0.3, 0.8, 0.95);
        let affect = pkt.extract_affective().unwrap();
        assert!((affect.valence - 0.5).abs() < 1e-6);
        assert!((affect.arousal - 0.7).abs() < 1e-6);
        assert!((affect.dominance - (-0.3)).abs() < 1e-6);
        assert!((affect.intensity - 0.8).abs() < 1e-6);
        assert!((affect.confidence - 0.95).abs() < 1e-6);
        assert_eq!(affect.timestamp_ms, 1_700_000_000);
        assert_eq!(affect.sequence, 1);
    }

    #[test]
    fn test_extract_affective_wrong_type() {
        let mut pkt = affective_packet(0.5, 0.7, -0.3, 0.8, 0.95);
        pkt.payload_type = PayloadType::WisdomVector;
        assert!(pkt.extract_affective().is_none());
    }

    #[test]
    fn test_extract_affective_nan_rejected() {
        let pkt = affective_packet(f32::NAN, 0.7, -0.3, 0.8, 0.95);
        assert!(pkt.extract_affective().is_none());
    }

    #[test]
    fn test_extract_affective_clamped() {
        // Values outside range should be clamped
        let pkt = affective_packet(2.0, -1.0, 5.0, 3.0, -0.5);
        let affect = pkt.extract_affective().unwrap();
        assert!((affect.valence - 1.0).abs() < 1e-6);
        assert!((affect.arousal - 0.0).abs() < 1e-6);
        assert!((affect.dominance - 1.0).abs() < 1e-6);
        assert!((affect.intensity - 1.0).abs() < 1e-6);
        assert!((affect.confidence - 0.0).abs() < 1e-6);
    }

    // ====================================================================
    // extract_gradient / from_gradient Tests
    // ====================================================================

    #[test]
    fn test_extract_gradient_valid() {
        let msg = crate::swarm::GradientMessage {
            source_id: [0u8; 32],
            gradient_data: vec![0.1, -0.2, 0.3, 0.0],
            trust_score: 0.85,
            noise_scale: 0.0,
            timestamp: 1_700_000_000_000,
            sample_count: 100,
            model_version: 3,
        };
        let pkt = WisdomPacket::from_gradient([0xAB; 8], 42, &msg).unwrap();
        let extracted = pkt.extract_gradient().unwrap();

        assert_eq!(extracted.gradient_data.len(), 4);
        assert!((extracted.gradient_data[0] - 0.1).abs() < 1e-6);
        assert!((extracted.gradient_data[1] - (-0.2)).abs() < 1e-6);
        assert!((extracted.gradient_data[2] - 0.3).abs() < 1e-6);
        assert!((extracted.gradient_data[3] - 0.0).abs() < 1e-6);
        assert!((extracted.trust_score - 0.85).abs() < 1e-6);
        assert_eq!(extracted.timestamp, 1_700_000_000_000);
        assert_eq!(extracted.sample_count, 100);
        assert_eq!(extracted.model_version, 3);
        // source_id should be zero-padded from 8 → 32 bytes
        assert_eq!(&extracted.source_id[..8], &[0xAB; 8]);
        assert_eq!(&extracted.source_id[8..], &[0u8; 24]);
    }

    #[test]
    fn test_extract_gradient_wrong_type() {
        let pkt = WisdomPacket {
            source_id: [0; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        };
        assert!(pkt.extract_gradient().is_none());
    }

    #[test]
    fn test_extract_gradient_nan_rejected() {
        // NaN in gradient data — manually build the packet to test extraction rejects it
        let mut bytes = [0u8; 2048];
        bytes[0..4].copy_from_slice(&3u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0.5f32.to_le_bytes());
        bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&1u64.to_le_bytes());
        bytes[32..36].copy_from_slice(&0.1f32.to_le_bytes());
        bytes[36..40].copy_from_slice(&f32::NAN.to_le_bytes());
        bytes[40..44].copy_from_slice(&0.3f32.to_le_bytes());

        let pkt = WisdomPacket {
            source_id: [0; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::Gradient,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV(bytes),
        };
        assert!(pkt.extract_gradient().is_none());
    }

    #[test]
    fn test_extract_gradient_nan_trust_rejected() {
        let mut bytes = [0u8; 2048];
        bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&f32::NAN.to_le_bytes()); // NaN trust
        bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&1u64.to_le_bytes());
        bytes[32..36].copy_from_slice(&0.1f32.to_le_bytes());

        let pkt = WisdomPacket {
            source_id: [0; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::Gradient,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV(bytes),
        };
        assert!(pkt.extract_gradient().is_none());
    }

    #[test]
    fn test_extract_gradient_overflow_rejected() {
        // gradient_count claims 999 floats, but not enough bytes
        let mut bytes = [0u8; 2048];
        bytes[0..4].copy_from_slice(&999u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0.5f32.to_le_bytes());
        bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&1u64.to_le_bytes());

        let pkt = WisdomPacket {
            source_id: [0; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::Gradient,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV(bytes),
        };
        assert!(pkt.extract_gradient().is_none());
    }

    #[test]
    fn test_gradient_roundtrip() {
        let original = crate::swarm::GradientMessage {
            source_id: [0u8; 32],
            gradient_data: vec![1.0, -2.5, 0.001, 42.0, -0.0],
            trust_score: 0.73,
            noise_scale: 0.01, // not preserved (set to 0 on extraction)
            timestamp: 1_708_000_000_000,
            sample_count: 500,
            model_version: 7,
        };

        let pkt = WisdomPacket::from_gradient([0x42; 8], 99, &original).unwrap();
        assert_eq!(pkt.payload_type, PayloadType::Gradient);
        assert_eq!(pkt.sequence, 99);

        let extracted = pkt.extract_gradient().unwrap();
        assert_eq!(extracted.gradient_data.len(), original.gradient_data.len());
        for (a, b) in extracted.gradient_data.iter().zip(&original.gradient_data) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        assert!((extracted.trust_score - original.trust_score).abs() < 1e-6);
        assert_eq!(extracted.timestamp, original.timestamp);
        assert_eq!(extracted.sample_count, original.sample_count);
        assert_eq!(extracted.model_version, original.model_version);
        assert_eq!(extracted.noise_scale, 0.0); // not preserved
    }

    #[test]
    fn test_from_gradient_too_large() {
        let msg = crate::swarm::GradientMessage {
            source_id: [0u8; 32],
            gradient_data: vec![0.0; 505], // 32 + 505*4 = 2052 > 2048
            trust_score: 0.5,
            noise_scale: 0.0,
            timestamp: 0,
            sample_count: 1,
            model_version: 1,
        };
        assert!(WisdomPacket::from_gradient([0; 8], 1, &msg).is_none());
    }

    // ====================================================================
    // MeshPeerRegistry Tests
    // ====================================================================

    #[test]
    fn test_peer_registry_tracks_packets() {
        let mut registry = MeshPeerRegistry::new();

        let pkt_a1 = WisdomPacket {
            source_id: [0xAA; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x01),
        };
        let pkt_a2 = WisdomPacket {
            source_id: [0xAA; 8],
            sequence: 2,
            phi: 0.6,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x02),
        };
        let pkt_b = WisdomPacket {
            source_id: [0xBB; 8],
            sequence: 1,
            phi: 0.9,
            urgency: MeshUrgency::Critical,
            timestamp_s: 0,
            payload_type: PayloadType::Heartbeat,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x03),
        };

        registry.update(&pkt_a1);
        registry.update(&pkt_a2);
        registry.update(&pkt_b);

        assert_eq!(registry.peer_count(), 2);
        let peers = registry.active_peers();
        let peer_a = peers.iter().find(|p| p.source_id == [0xAA; 8]).unwrap();
        assert_eq!(peer_a.packets_received, 2);
        assert!((peer_a.last_phi - 0.6).abs() < 1e-6);

        let peer_b = peers.iter().find(|p| p.source_id == [0xBB; 8]).unwrap();
        assert_eq!(peer_b.packets_received, 1);
    }

    #[test]
    fn test_peer_registry_expire_stale() {
        let mut registry = MeshPeerRegistry::with_timeout(std::time::Duration::from_millis(10));

        let pkt = WisdomPacket {
            source_id: [0xCC; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Cruise,
            timestamp_s: 0,
            payload_type: PayloadType::Heartbeat,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x04),
        };
        registry.update(&pkt);
        assert_eq!(registry.peer_count(), 1);

        // Wait for entry to become stale
        std::thread::sleep(std::time::Duration::from_millis(20));
        let expired = registry.expire_stale();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], [0xCC; 8]);
        assert_eq!(registry.peer_count(), 0);
    }

    #[test]
    fn test_peer_registry_average_phi() {
        let mut registry = MeshPeerRegistry::new();

        // No peers → 0.0
        assert_eq!(registry.average_phi(), 0.0);

        let pkt1 = WisdomPacket {
            source_id: [0x01; 8],
            sequence: 1,
            phi: 0.4,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x10),
        };
        let pkt2 = WisdomPacket {
            source_id: [0x02; 8],
            sequence: 1,
            phi: 0.8,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x20),
        };

        registry.update(&pkt1);
        registry.update(&pkt2);

        let avg = registry.average_phi();
        assert!((avg - 0.6).abs() < 1e-6, "Expected ~0.6, got {avg}");
    }

    // ====================================================================
    // from_affective Roundtrip Test
    // ====================================================================

    // ====================================================================
    // Health Score Tests
    // ====================================================================

    #[test]
    fn test_health_score_no_activity() {
        let stats = MeshStats::default();
        assert_eq!(stats.health_score(0), 0.0);
        assert_eq!(stats.health_score(5), 0.0);
    }

    #[test]
    fn test_health_score_healthy_mesh() {
        let stats = MeshStats {
            wisdom_sent: 50,
            wisdom_received: 48,
            heartbeats_sent: 20,
            heartbeats_received: 18,
            peers_expired: 1,
            ..Default::default()
        };
        let score = stats.health_score(5);
        // Connectivity: 5/5 = 1.0 → 0.40
        // Bidirectionality: min(70,66)/max(70,66) = 66/70 ≈ 0.943 → 0.377
        // Stability: 1 - 1/(1+66+1) ≈ 0.985 → 0.197
        // Total ≈ 0.97+
        assert!(score > 0.9, "Healthy mesh score should be high: {score}");
    }

    #[test]
    fn test_health_score_send_only() {
        let stats = MeshStats {
            wisdom_sent: 100,
            heartbeats_sent: 50,
            ..Default::default()
        };
        let score = stats.health_score(3);
        // Connectivity: 3/5 = 0.6 → 0.24
        // Bidirectionality: 0 (no receives) → 0.0
        // Stability: 1 - 0/(0+0+1) = 1.0 → 0.20
        // Total = 0.44
        assert!(
            score < 0.5,
            "Send-only mesh should have low score: {score}"
        );
        // But not zero — we do have connectivity
        assert!(score > 0.0, "Score should be > 0 with peers: {score}");
    }

    // ====================================================================
    // from_affective Roundtrip Test
    // ====================================================================

    #[test]
    fn test_from_affective_roundtrip() {
        let affect = crate::swarm::AffectiveState {
            valence: 0.6,
            arousal: 0.8,
            dominance: -0.3,
            intensity: 0.7,
            confidence: 0.95,
            timestamp_ms: 1_700_000_000,
            sequence: 42,
        };

        let pkt = WisdomPacket::from_affective([0xAF; 8], 42, &affect);
        assert_eq!(pkt.payload_type, PayloadType::Affective);
        assert_eq!(pkt.sequence, 42);

        let extracted = pkt.extract_affective().unwrap();
        assert!((extracted.valence - 0.6).abs() < 1e-6);
        assert!((extracted.arousal - 0.8).abs() < 1e-6);
        assert!((extracted.dominance - (-0.3)).abs() < 1e-6);
        assert!((extracted.intensity - 0.7).abs() < 1e-6);
        assert!((extracted.confidence - 0.95).abs() < 1e-6);
        assert_eq!(extracted.timestamp_ms, 1_700_000_000);
        assert_eq!(extracted.sequence, 42);
    }

    // ====================================================================
    // Per-Peer Rate Limiting Tests
    // ====================================================================

    #[test]
    fn test_rate_limit_allows_under_limit() {
        let mut registry = MeshPeerRegistry::new();
        let peer_id = [0xAA; 8];

        // Register peer first
        let pkt = WisdomPacket {
            source_id: peer_id,
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x01),
        };
        registry.update(&pkt);

        // 99 calls should all be under limit (window_count 1..=99)
        for _ in 0..99 {
            assert!(!registry.is_rate_limited(&peer_id), "Should be under limit");
        }
    }

    #[test]
    fn test_rate_limit_blocks_over_limit() {
        let mut registry = MeshPeerRegistry::new();
        let peer_id = [0xBB; 8];

        let pkt = WisdomPacket {
            source_id: peer_id,
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x02),
        };
        registry.update(&pkt);

        // First 100 should pass (window_count 1..=100)
        for _ in 0..100 {
            registry.is_rate_limited(&peer_id);
        }
        // 101st should be blocked (window_count == 101 > 100)
        assert!(
            registry.is_rate_limited(&peer_id),
            "101st call should be rate limited"
        );
    }

    #[test]
    fn test_rate_limit_window_resets() {
        let mut registry = MeshPeerRegistry::new();
        let peer_id = [0xCC; 8];

        let pkt = WisdomPacket {
            source_id: peer_id,
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x03),
        };
        registry.update(&pkt);

        // Exhaust the limit
        for _ in 0..101 {
            registry.is_rate_limited(&peer_id);
        }
        assert!(registry.is_rate_limited(&peer_id), "Should be rate limited");

        // Manually reset window_start to simulate time passage
        if let Some(entry) = registry.peers.get_mut(&peer_id) {
            entry.window_start = std::time::Instant::now() - std::time::Duration::from_secs(11);
        }

        // Should be allowed again after window reset
        assert!(
            !registry.is_rate_limited(&peer_id),
            "Should be allowed after window reset"
        );
    }

    // ====================================================================
    // TTL Wire Format Tests (Item 4)
    // ====================================================================

    #[test]
    fn test_ttl_wire_roundtrip() {
        let packet = WisdomPacket {
            source_id: [0x01; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 5,
            wisdom: test_hv(0x01),
        };
        let bytes = packet.to_bytes();
        let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.ttl, 5);
    }

    #[test]
    fn test_ttl_zero_backward_compat() {
        // Legacy packets with zeroed reserved bytes should parse with ttl=0
        let packet = WisdomPacket {
            source_id: [0x01; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x01),
        };
        let bytes = packet.to_bytes();
        assert_eq!(bytes[23], 0);
        let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.ttl, 0);
    }

    // ====================================================================
    // Packet Authentication Tests (Item 1)
    // ====================================================================

    #[test]
    fn test_packet_mac_roundtrip() {
        let key = [0x42u8; 32];
        let mut packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: test_hv(0xAB),
        };
        let mut bytes = packet.to_bytes();
        let mac = compute_packet_mac(&bytes, &key);
        packet.auth_mac = mac;
        bytes = packet.to_bytes();
        assert!(verify_packet_mac(&bytes, &key));
    }

    #[test]
    fn test_packet_mac_rejects_tampered() {
        let key = [0x42u8; 32];
        let mut packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: test_hv(0xAB),
        };
        let mut bytes = packet.to_bytes();
        let mac = compute_packet_mac(&bytes, &key);
        packet.auth_mac = mac;
        bytes = packet.to_bytes();
        // Tamper with a wisdom byte
        bytes[100] ^= 0xFF;
        assert!(!verify_packet_mac(&bytes, &key));
    }

    #[test]
    fn test_packet_mac_rejects_wrong_key() {
        let key_a = [0x42u8; 32];
        let key_b = [0x99u8; 32];
        let mut packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: test_hv(0xAB),
        };
        let mut bytes = packet.to_bytes();
        let mac = compute_packet_mac(&bytes, &key_a);
        packet.auth_mac = mac;
        bytes = packet.to_bytes();
        assert!(!verify_packet_mac(&bytes, &key_b));
    }

    #[test]
    fn test_packet_mac_zero_without_key() {
        let packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0xAB),
        };
        assert_eq!(packet.auth_mac, 0);
    }

    #[test]
    fn test_packet_mac_preserved_through_serde() {
        let packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0xAB,
            ttl: 3,
            wisdom: test_hv(0xAB),
        };
        let bytes = packet.to_bytes();
        let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.auth_mac, 0xAB);
        assert_eq!(decoded.ttl, 3);
    }

    #[test]
    fn test_packet_mac_fragment_roundtrip() {
        let key = [0x77u8; 32];
        let mut packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: test_hv(0xAB),
        };
        // Sign the packet
        let bytes = packet.to_bytes();
        let mac = compute_packet_mac(&bytes, &key);
        packet.auth_mac = mac;

        // Fragment and reassemble (drop fragment 3)
        let frags = packet.fragment();
        let mut assembler = WisdomPacket::assembler(packet.thought_id(), 11);
        let mut buf = [0u8; LORA_MTU];
        for (i, frag) in frags.iter().enumerate() {
            if i == 3 { continue; }
            let len = frag.to_bytes(&mut buf);
            let decoded_frag = LoRaFragment::from_bytes(&buf[..len]).unwrap();
            assembler.feed(&decoded_frag);
        }
        assert!(assembler.is_complete());
        let recovered = WisdomPacket::from_assembler(&assembler).unwrap();
        let recovered_bytes = recovered.to_bytes();
        assert!(verify_packet_mac(&recovered_bytes, &key));
    }

    // ====================================================================
    // Priority Ordering Test (Item 2)
    // ====================================================================

    #[test]
    fn test_payload_type_priority_ordering() {
        assert!(PayloadType::Heartbeat.priority() > PayloadType::WisdomVector.priority());
        assert!(PayloadType::WisdomVector.priority() > PayloadType::Affective.priority());
        assert!(PayloadType::Affective.priority() > PayloadType::Gradient.priority());
        assert_eq!(PayloadType::Heartbeat.priority(), 3);
        assert_eq!(PayloadType::Gradient.priority(), 0);
    }

    // ====================================================================
    // MeshPeerRegistry has_peer Test (Item 5)
    // ====================================================================

    #[test]
    fn test_peer_registry_has_peer() {
        let mut registry = MeshPeerRegistry::new();
        let peer_id = [0xDD; 8];
        assert!(!registry.has_peer(&peer_id));

        let pkt = WisdomPacket {
            source_id: peer_id,
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x01),
        };
        registry.update(&pkt);
        assert!(registry.has_peer(&peer_id));
    }

    // ====================================================================
    // Compression Tests (Item 3)
    // ====================================================================

    #[test]
    fn test_compress_decompress_roundtrip() {
        let packet = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: test_hv(0xAB),
        };
        let raw = packet.to_bytes();
        let compressed = compress_packet(&raw);
        let decompressed = decompress_packet(&compressed).unwrap();
        assert_eq!(decompressed.len(), WISDOM_PACKET_SIZE);
        assert_eq!(&decompressed[..], &raw[..]);
    }

    #[test]
    fn test_compress_heartbeat_uses_envelope() {
        // Heartbeat is mostly zeros — should compress (with lz4 feature)
        // or at least produce a valid envelope
        let packet = WisdomPacket {
            source_id: [0; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Cruise,
            timestamp_s: 0,
            payload_type: PayloadType::Heartbeat,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0u8; 2048]),
        };
        let raw = packet.to_bytes();
        let compressed = compress_packet(&raw);
        // First byte must be a valid compression header
        assert!(compressed[0] == COMPRESS_NONE || compressed[0] == COMPRESS_LZ4);
        // Roundtrip must succeed
        let decompressed = decompress_packet(&compressed).unwrap();
        assert_eq!(&decompressed[..], &raw[..]);
    }

    #[test]
    fn test_decompress_invalid_header() {
        let data = vec![0xFF, 0x01, 0x02];
        assert!(decompress_packet(&data).is_none());
    }

    #[test]
    fn test_decompress_tolerates_trailing_bytes() {
        // Simulate FEC adding trailing garbage after COMPRESS_NONE envelope
        let packet = WisdomPacket {
            source_id: [0x01; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0x42),
        };
        let raw = packet.to_bytes();
        let mut envelope = compress_packet(&raw);
        // Append trailing garbage (simulating FEC padding)
        envelope.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let decompressed = decompress_packet(&envelope).unwrap();
        assert_eq!(&decompressed[..], &raw[..]);
    }
}
