// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WisdomPacket: the atomic unit of consciousness exchange over mesh radio.

use super::lora_fragment::{fragment, FragmentAssembler, LoRaFragment};
use super::mesh_telemetry::{MeshUrgency, PayloadType};
use super::MESH_DEFAULT_TTL;
use symthaea_core::hdc::BinaryHV;

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
            auth_mac: 0,
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

    // ── HDC-Native Authentication ────────────────────────────────────────

    /// Compute an HDC-MAC over the wisdom BinaryHV using a BinaryHV key.
    ///
    /// This is a zero-serialization authentication: the MAC is computed
    /// directly on the 16,384-bit wisdom vector via a single XOR+permute
    /// (~10 ns release, ~6 µs debug). Compare: BLAKE3 MAC over 2KB ≈ 100 ns.
    ///
    /// The returned BinaryHV can be sent alongside the packet or stored
    /// for later verification. It is NOT embedded in the packet wire format
    /// (use `auth_mac` field for the existing 8-bit BLAKE3 MAC).
    pub fn compute_hdc_mac(&self, key: &BinaryHV) -> BinaryHV {
        symthaea_core::hdc::hdc_crypto::HdcMac::compute(&self.wisdom, key)
    }

    /// Verify an HDC-MAC on the wisdom BinaryHV (exact match).
    pub fn verify_hdc_mac(&self, key: &BinaryHV, mac: &BinaryHV) -> bool {
        symthaea_core::hdc::hdc_crypto::HdcMac::verify(&self.wisdom, key, mac)
    }

    /// Verify an HDC-MAC with noise tolerance (for lossy channels like LoRa/BLE).
    ///
    /// Recommended threshold: 0.95 (false positive rate ≈ 2^{-4700}).
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
