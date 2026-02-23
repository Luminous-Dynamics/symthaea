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

pub use dual_layer::{DualLayerMesh, LoopbackTransport, MeshRoute, MeshTransport};
pub use lora_fragment::{
    crc16_ccitt, fragment, FragmentAssembler, LoRaFragment, FLAG_FEC, HEADER_SIZE, LORA_MTU,
    PAYLOAD_SIZE,
};
pub use mesh_receiver::{MeshPeer, MeshReceiver, ReceiverStats, StreamKey};

mod bridge;

pub use bridge::{MeshBridgeActor, MeshBridgeHandle, MeshOutbound};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::BinaryHV;

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
/// 22-23    reserved        [0, 0]     alignment / future use
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
        // buf[22..24] reserved (zeros)
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

        let mut wisdom_bytes = [0u8; 2048];
        wisdom_bytes.copy_from_slice(&buf[24..WISDOM_PACKET_SIZE]);

        Some(Self {
            source_id,
            sequence,
            phi,
            urgency,
            timestamp_s,
            payload_type,
            wisdom: BinaryHV(wisdom_bytes),
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
    bytes
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
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

        assert_eq!(MeshUrgency::from(CycleUrgency::Critical), MeshUrgency::Critical);
        assert_eq!(MeshUrgency::from(CycleUrgency::Normal), MeshUrgency::Normal);
        assert_eq!(MeshUrgency::from(CycleUrgency::Cruise), MeshUrgency::Cruise);
    }
}
