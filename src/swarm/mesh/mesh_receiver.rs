// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh Receiver — Multi-peer fragment reassembly with timeout expiry.
//!
//! Listens on any mesh transport (LoRa, B.A.T.M.A.N., Yggdrasil) and
//! reassembles incoming data into [`WisdomPacket`]s. Handles:
//!
//! - **Multi-peer isolation**: Fragments keyed by `(source, thought_id)` —
//!   two peers using the same `thought_id` never collide.
//! - **Timeout expiry**: Stale incomplete assemblies garbage collected
//!   after configurable timeout (default 30s).
//! - **Capacity limits**: Evicts oldest assembly when at capacity to prevent
//!   memory exhaustion on embedded targets (Raspberry Pi).
//! - **Dual-mode intake**: Fragmented (LoRa) and whole-packet
//!   (B.A.T.M.A.N./Yggdrasil) paths.
//! - **Peer discovery**: Learns about peers from received data.
//! - **Telemetry**: Fragment loss rates, FEC recovery stats, peer visibility.

use super::{FragmentAssembler, LoRaFragment, WISDOM_PACKET_SIZE, WisdomPacket};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// STREAM KEY
// ============================================================================

/// Composite key for disambiguating concurrent fragment streams.
///
/// Two peers can independently use the same `thought_id` (since it's just
/// `sequence as u16`). Keying on `(source, thought_id)` prevents fragment
/// collision — fragments from different peers never mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamKey {
    /// Peer identity from the transport layer (e.g., LoRa MAC, B.A.T.M.A.N.
    /// HW addr, Yggdrasil IPv6 prefix). Normalized to 8 bytes.
    pub source: [u8; 8],
    /// Fragment group identifier (from LoRa fragment header).
    pub thought_id: u16,
}

// ============================================================================
// PENDING ASSEMBLY
// ============================================================================

/// State for one in-progress fragment reassembly.
struct PendingAssembly {
    assembler: FragmentAssembler,
    created_at: Instant,
    last_fragment_at: Instant,
    /// Last fragment index seen — used to reject out-of-order fragments
    /// with gap exceeding the fragment range (likely injection).
    last_fragment_index: Option<u8>,
    /// Key fingerprint when the first fragment arrived — used to detect
    /// mid-stream key rotation. 16-bit hash of the key for ~1/65536 collision rate.
    #[cfg(feature = "mesh-encryption")]
    key_fingerprint: Option<u16>,
}

// ============================================================================
// RECEIVER STATS
// ============================================================================

/// Telemetry from the mesh receiver.
#[derive(Debug, Clone, Default)]
pub struct ReceiverStats {
    /// WisdomPackets successfully reassembled (fragmented + whole).
    pub packets_complete: u64,
    /// Packets that required FEC recovery (at least one fragment was lost).
    pub packets_recovered: u64,
    /// Assemblies that expired before completion (fragments lost to wind).
    pub packets_expired: u64,
    /// Total LoRa fragments fed into assemblers.
    pub fragments_received: u64,
    /// Fragments rejected by CRC-16 validation (corrupted in transit).
    pub fragments_corrupt: u64,
    /// Duplicate fragments (same position already received, ignored).
    pub fragments_duplicate: u64,
    /// Whole packets received (B.A.T.M.A.N./Yggdrasil, no fragmentation).
    pub whole_packets: u64,
    /// Packets that failed decryption (wrong key or corrupted ciphertext).
    pub packets_decrypt_failed: u64,
    /// Fragments rejected due to excessive reordering (gap > 3 from last index).
    pub fragments_reordered: u64,
    /// Fragments rejected due to key fingerprint mismatch (mid-stream key rotation).
    #[cfg(feature = "mesh-encryption")]
    pub fragments_key_mismatch: u64,
}

// ============================================================================
// MESH PEER
// ============================================================================

/// A mesh peer discovered through received data.
#[derive(Debug, Clone)]
pub struct MeshPeer {
    /// Peer source identifier (transport-layer address, 8 bytes).
    pub source: [u8; 8],
    /// When we last received data from this peer.
    pub last_seen: Instant,
    /// Total WisdomPackets received from this peer.
    pub packets_received: u64,
    /// Last reported Phi value (from WisdomPacket metadata).
    pub last_phi: f32,
}

// ============================================================================
// MESH RECEIVER
// ============================================================================

/// Receives and reassembles mesh data from multiple concurrent peers.
///
/// # Usage
///
/// ```rust,ignore
/// let mut receiver = MeshReceiver::new();
///
/// // LoRa: fragments arrive one at a time from the radio
/// let source = [0x01; 8]; // transport-layer peer address
/// if let Some(packet) = receiver.receive_fragment(source, &raw_bytes) {
///     println!("Wisdom from {:02x?}: phi={}", packet.source_id, packet.phi);
/// }
///
/// // B.A.T.M.A.N./Yggdrasil: whole packet in one frame
/// if let Some(packet) = receiver.receive_whole(&raw_bytes) {
///     println!("Urgent thought: {:?}", packet.urgency);
/// }
///
/// // Call periodically to garbage collect stale assemblies
/// receiver.expire_stale();
/// ```
pub struct MeshReceiver {
    /// In-progress fragment reassemblies, keyed by (source, thought_id).
    pending: HashMap<StreamKey, PendingAssembly>,
    /// Recently completed stream keys — used to silently drop late
    /// fragments that arrive after assembly completion (e.g., the FEC
    /// fragment arriving after all 10 data fragments completed the assembly).
    recently_completed: Vec<StreamKey>,
    /// Known peers, discovered through received data.
    peers: HashMap<[u8; 8], MeshPeer>,
    /// Timeout for incomplete assemblies before expiry.
    timeout: Duration,
    /// Expected payload size for WisdomPacket reassembly.
    expected_payload_size: usize,
    /// Maximum concurrent pending assemblies (prevents memory exhaustion).
    max_pending: usize,
    /// Maximum recently-completed keys to remember (ring buffer).
    max_recent: usize,
    /// Cumulative statistics.
    stats: ReceiverStats,
    /// Optional ChaCha20-Poly1305 decryption key for incoming packets.
    #[cfg(feature = "mesh-encryption")]
    encryption_key: Option<[u8; 32]>,
    /// Whether to decrypt individual LoRa fragments (fragment-level AEAD).
    #[cfg(feature = "mesh-encryption")]
    fragment_encryption: bool,
}

impl MeshReceiver {
    /// Create a new receiver with default settings.
    ///
    /// Defaults: 30s timeout, 64 max pending, WisdomPacket payload size.
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            recently_completed: Vec::new(),
            peers: HashMap::new(),
            timeout: Duration::from_secs(30),
            expected_payload_size: WISDOM_PACKET_SIZE + 64,
            max_pending: 64,
            max_recent: 32,
            stats: ReceiverStats::default(),
            #[cfg(feature = "mesh-encryption")]
            encryption_key: None,
            #[cfg(feature = "mesh-encryption")]
            fragment_encryption: false,
        }
    }

    /// Set the ChaCha20-Poly1305 decryption key for incoming packets.
    ///
    /// When set, incoming data is decrypted before decompression.
    /// If decryption fails, the packet is rejected (no fallback when key is set).
    #[cfg(feature = "mesh-encryption")]
    pub fn with_encryption_key(mut self, key: [u8; 32]) -> Self {
        self.encryption_key = Some(key);
        self
    }

    /// Update the encryption key at runtime (for bridge key propagation).
    #[cfg(feature = "mesh-encryption")]
    pub fn set_encryption_key(&mut self, key: Option<[u8; 32]>) {
        self.encryption_key = key;
    }

    /// Enable fragment-level AEAD decryption for LoRa fragments.
    #[cfg(feature = "mesh-encryption")]
    pub fn with_fragment_encryption(mut self, enabled: bool) -> Self {
        self.fragment_encryption = enabled;
        self
    }

    /// Set the timeout for incomplete assemblies.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum number of concurrent pending assemblies.
    pub fn with_max_pending(mut self, max: usize) -> Self {
        self.max_pending = max;
        self
    }

    /// Set the expected payload size (default: [`WISDOM_PACKET_SIZE`]).
    pub fn with_expected_payload_size(mut self, size: usize) -> Self {
        self.expected_payload_size = size;
        self
    }

    /// Process a raw LoRa fragment from a known source.
    ///
    /// - `source`: 8-byte peer identity from the transport layer
    ///   (LoRa MAC address, zero-padded if shorter)
    /// - `raw`: raw bytes received from radio (header + payload)
    ///
    /// Returns a completed [`WisdomPacket`] if this fragment completed the
    /// reassembly. Returns `None` if more fragments are needed, the fragment
    /// was corrupt, or it was a duplicate.
    pub fn receive_fragment(&mut self, source: [u8; 8], raw: &[u8]) -> Option<WisdomPacket> {
        // Decrypt fragment-level AEAD if enabled
        #[cfg(feature = "mesh-encryption")]
        let decrypted_buf: zeroize::Zeroizing<Vec<u8>>;
        #[cfg(feature = "mesh-encryption")]
        let mut nonce_bytes_saved: Option<[u8; 12]> = None;
        #[cfg(feature = "mesh-encryption")]
        let raw = if self.fragment_encryption {
            if let Some(ref key) = self.encryption_key {
                // Verify the nonce's source_id matches the transport-layer source
                // before attempting decryption — prevents cross-stream injection.
                if raw.len() >= super::AEAD_NONCE_SIZE {
                    let nonce_source = &raw[..8];
                    if nonce_source != &source {
                        self.stats.packets_decrypt_failed += 1;
                        return None;
                    }
                    // Save nonce for post-parse validation
                    let mut nb = [0u8; 12];
                    nb.copy_from_slice(&raw[..12]);
                    nonce_bytes_saved = Some(nb);
                }
                match super::decrypt_fragment(raw, key) {
                    Some(plain) => {
                        decrypted_buf = zeroize::Zeroizing::new(plain);
                        decrypted_buf.as_slice()
                    }
                    None => {
                        self.stats.packets_decrypt_failed += 1;
                        return None;
                    }
                }
            } else {
                raw
            }
        } else {
            raw
        };

        // Parse and CRC-validate
        let frag = match LoRaFragment::from_bytes(raw) {
            Some(f) => f,
            None => {
                self.stats.fragments_corrupt += 1;
                return None;
            }
        };

        // Verify nonce thought_id matches parsed fragment — detects cross-stream
        // injection where an attacker feeds a fragment encrypted for stream B
        // into stream A's reassembly (nonce carries the original thought_id).
        #[cfg(feature = "mesh-encryption")]
        if let Some(nb) = nonce_bytes_saved {
            let nonce_thought_id = u16::from_le_bytes([nb[8], nb[9]]);
            if nonce_thought_id != frag.thought_id {
                self.stats.packets_decrypt_failed += 1;
                return None;
            }
        }

        self.stats.fragments_received += 1;
        let now = Instant::now();

        let key = StreamKey {
            source,
            thought_id: frag.thought_id,
        };

        // Suppress late fragments for already-completed streams
        if self.recently_completed.contains(&key) {
            return None;
        }

        // Ensure capacity before inserting a new assembly
        if !self.pending.contains_key(&key) {
            self.ensure_capacity(now);
        }

        // Compute 16-bit key fingerprint for this fragment (Item 12).
        // FNV-1a-inspired 16-bit hash over 8 evenly-spaced key bytes.
        // Position-dependent mixing avoids collisions for uniform keys
        // (e.g., [0xAA; 32] vs [0xBB; 32] which pure XOR would collapse to 0).
        #[cfg(feature = "mesh-encryption")]
        let current_fingerprint: Option<u16> = self.encryption_key.map(|k| {
            let samples = [k[0], k[4], k[8], k[12], k[16], k[20], k[24], k[31]];
            let mut h: u16 = 0x811C; // FNV offset basis (truncated)
            for &b in &samples {
                h ^= b as u16;
                h = h.wrapping_mul(0x0101); // FNV-like prime
            }
            h
        });

        // Get or create the assembler for this stream
        let assembly = self.pending.entry(key).or_insert_with(|| PendingAssembly {
            assembler: FragmentAssembler::new(
                frag.thought_id,
                frag.total_fragments,
                self.expected_payload_size,
            ),
            created_at: now,
            last_fragment_at: now,
            last_fragment_index: None,
            #[cfg(feature = "mesh-encryption")]
            key_fingerprint: None,
        });

        assembly.last_fragment_at = now;

        // Track duplicates before feeding
        if assembly.assembler.has_fragment(frag.fragment_index) {
            self.stats.fragments_duplicate += 1;
            return None;
        }

        // Item 5: Reject fragments with excessive reordering.
        // Use total_fragments as the max legitimate gap — radio delivers in
        // arbitrary order, and FEC fragments (index 10) can arrive after any
        // data fragment (indices 0-9). Only reject gaps that exceed the valid
        // fragment index range, which indicates injection or severe corruption.
        if let Some(last_idx) = assembly.last_fragment_index {
            let gap = if frag.fragment_index > last_idx {
                frag.fragment_index - last_idx
            } else {
                last_idx - frag.fragment_index
            };
            let max_gap = frag.total_fragments.saturating_sub(1).max(3);
            if gap > max_gap {
                self.stats.fragments_reordered += 1;
                return None;
            }
        }
        assembly.last_fragment_index = Some(frag.fragment_index);

        // Item 12: Reject fragments if encryption key changed mid-stream
        #[cfg(feature = "mesh-encryption")]
        {
            if let Some(fp) = current_fingerprint {
                match assembly.key_fingerprint {
                    None => assembly.key_fingerprint = Some(fp),
                    Some(stored_fp) if stored_fp != fp => {
                        self.stats.fragments_key_mismatch += 1;
                        return None;
                    }
                    _ => {}
                }
            }
        }

        let is_complete = assembly.assembler.feed(&frag);

        if is_complete {
            // Extract the completed assembly
            let Some(assembly) = self.pending.remove(&key) else {
                return None; // Assembly entry missing despite is_complete — should not happen
            };
            let used_fec = assembly.assembler.used_fec_recovery();

            // When fragment_encryption is active, fragments were already
            // decrypted individually — skip packet-level decryption and
            // go straight to decompress → parse.
            #[cfg(feature = "mesh-encryption")]
            let already_decrypted = self.fragment_encryption && self.encryption_key.is_some();
            #[cfg(not(feature = "mesh-encryption"))]
            let already_decrypted = false;

            let packet = assembly.assembler.assemble().and_then(|assembled| {
                if already_decrypted {
                    super::decompress_packet(&assembled)
                        .and_then(|raw| WisdomPacket::from_bytes(&raw))
                        .or_else(|| WisdomPacket::from_bytes(&assembled))
                } else {
                    self.decode_envelope(&assembled)
                }
            });

            if let Some(packet) = packet {
                self.stats.packets_complete += 1;
                if used_fec {
                    self.stats.packets_recovered += 1;
                }
                // Remember this key to suppress late-arriving fragments
                self.recently_completed.push(key);
                if self.recently_completed.len() > self.max_recent {
                    self.recently_completed.remove(0);
                }
                self.touch_peer(source, &packet, now);
                return Some(packet);
            }
        }

        None
    }

    /// Process a whole WisdomPacket from B.A.T.M.A.N. or Yggdrasil.
    ///
    /// Handles both compressed envelopes (1-byte header + payload) and
    /// raw legacy packets for backward compatibility.
    pub fn receive_whole(&mut self, raw: &[u8]) -> Option<WisdomPacket> {
        // Decrypt (if key set) → decompress → parse, with fallback.
        let packet = self.decode_envelope(raw)?;

        self.stats.whole_packets += 1;
        self.stats.packets_complete += 1;

        let now = Instant::now();
        self.touch_peer(packet.source_id, &packet, now);

        Some(packet)
    }

    /// Decode an assembled/received envelope: decrypt → decompress → parse.
    ///
    /// When an encryption key is set, attempts decryption first. Tries versioned
    /// format (1-byte version prefix), then standard ChaCha20-Poly1305,
    /// then XChaCha20-Poly1305. If all fail, the packet is rejected.
    fn decode_envelope(&mut self, data: &[u8]) -> Option<WisdomPacket> {
        #[cfg(feature = "mesh-encryption")]
        if let Some(ref key) = self.encryption_key {
            // Try versioned format: [version (1) | nonce (12) | ciphertext+tag]
            if data.len() >= 1 + super::AEAD_NONCE_SIZE + super::AEAD_TAG_SIZE {
                if let Some(decrypted) = super::decrypt_packet(&data[1..], key) {
                    let decrypted = zeroize::Zeroizing::new(decrypted);
                    return super::decompress_packet(&decrypted)
                        .and_then(|raw| WisdomPacket::from_bytes(&raw))
                        .or_else(|| WisdomPacket::from_bytes(&decrypted));
                }
            }
            // Try standard ChaCha20-Poly1305 (12-byte nonce, no version prefix)
            if let Some(decrypted) = super::decrypt_packet(data, key) {
                let decrypted = zeroize::Zeroizing::new(decrypted);
                return super::decompress_packet(&decrypted)
                    .and_then(|raw| WisdomPacket::from_bytes(&raw))
                    .or_else(|| WisdomPacket::from_bytes(&decrypted));
            }
            // Try XChaCha20-Poly1305 (24-byte nonce)
            if let Some(decrypted) = super::decrypt_packet_xchacha(data, key) {
                let decrypted = zeroize::Zeroizing::new(decrypted);
                return super::decompress_packet(&decrypted)
                    .and_then(|raw| WisdomPacket::from_bytes(&raw))
                    .or_else(|| WisdomPacket::from_bytes(&decrypted));
            }
            // All failed — reject. When key is set, only authenticated
            // ciphertext is accepted.
            self.stats.packets_decrypt_failed += 1;
            return None;
        }

        // Unencrypted path (no encryption key configured)
        super::decompress_packet(data)
            .and_then(|raw| WisdomPacket::from_bytes(&raw))
            .or_else(|| WisdomPacket::from_bytes(data))
    }

    /// Expire stale incomplete assemblies.
    ///
    /// Call this periodically — e.g., once per cognitive cycle (50 Hz) or
    /// once per second. Returns the keys of expired assemblies for logging.
    pub fn expire_stale(&mut self) -> Vec<StreamKey> {
        let now = Instant::now();
        let timeout = self.timeout;

        let expired_keys: Vec<StreamKey> = self
            .pending
            .iter()
            .filter(|(_, a)| now.duration_since(a.last_fragment_at) > timeout)
            .map(|(k, _)| *k)
            .collect();

        for key in &expired_keys {
            self.pending.remove(key);
            self.stats.packets_expired += 1;
        }

        expired_keys
    }

    /// Number of in-progress assemblies.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get cumulative receiver statistics.
    pub fn stats(&self) -> &ReceiverStats {
        &self.stats
    }

    /// Get all known peers (discovered through received data).
    pub fn peers(&self) -> &HashMap<[u8; 8], MeshPeer> {
        &self.peers
    }

    /// Get a specific peer by transport-layer source address.
    pub fn peer(&self, source: &[u8; 8]) -> Option<&MeshPeer> {
        self.peers.get(source)
    }

    /// Number of discovered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Ensure we have room for a new pending assembly.
    ///
    /// First tries to expire stale entries. If still at capacity,
    /// evicts the oldest assembly (by creation time).
    fn ensure_capacity(&mut self, now: Instant) {
        if self.pending.len() < self.max_pending {
            return;
        }

        // Try expiring stale entries first
        let timeout = self.timeout;
        let stale: Vec<StreamKey> = self
            .pending
            .iter()
            .filter(|(_, a)| now.duration_since(a.last_fragment_at) > timeout)
            .map(|(k, _)| *k)
            .collect();

        for key in &stale {
            self.pending.remove(key);
            self.stats.packets_expired += 1;
        }

        // If still at capacity, evict the oldest
        if self.pending.len() >= self.max_pending {
            if let Some(oldest_key) = self
                .pending
                .iter()
                .min_by_key(|(_, a)| a.created_at)
                .map(|(k, _)| *k)
            {
                self.pending.remove(&oldest_key);
                self.stats.packets_expired += 1;
            }
        }
    }

    /// Update peer tracking after receiving a complete packet.
    fn touch_peer(&mut self, source: [u8; 8], packet: &WisdomPacket, now: Instant) {
        let peer = self.peers.entry(source).or_insert_with(|| MeshPeer {
            source,
            last_seen: now,
            packets_received: 0,
            last_phi: 0.0,
        });
        peer.last_seen = now;
        peer.packets_received += 1;
        peer.last_phi = packet.phi;
    }
}

impl Default for MeshReceiver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::mesh::{LORA_MTU, MeshUrgency, PayloadType};
    use symthaea_core::hdc::BinaryHV;

    fn test_hv(seed: u8) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = seed.wrapping_mul(i as u8).wrapping_add((i >> 3) as u8);
        }
        BinaryHV(bytes)
    }

    fn test_packet(seq: u32, source: [u8; 8], seed: u8) -> WisdomPacket {
        WisdomPacket {
            source_id: source,
            sequence: seq,
            phi: 0.73,
            urgency: MeshUrgency::Cruise,
            timestamp_s: 1_700_000_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: [0; 32],
            ttl: 0,
            wisdom: test_hv(seed),
        }
    }

    /// Serialize fragments to raw wire bytes (as they'd arrive from radio).
    fn to_wire(frags: &[LoRaFragment]) -> Vec<Vec<u8>> {
        frags
            .iter()
            .map(|frag| {
                let mut buf = [0u8; LORA_MTU];
                let len = frag.to_bytes(&mut buf);
                buf[..len].to_vec()
            })
            .collect()
    }

    const PEER_A: [u8; 8] = [0xAA; 8];
    const PEER_B: [u8; 8] = [0xBB; 8];
    const PEER_C: [u8; 8] = [0xCC; 8];

    // -- Basic reception --

    #[test]
    fn receive_all_fragments() {
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(1, PEER_A, 0x11);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        // Feed all 11 fragments — completes when last data fragment arrives
        // (10 data fragments at indices 0-9, FEC at index 10)
        let mut completed = None;
        for raw in &wire {
            if let Some(p) = receiver.receive_fragment(PEER_A, raw) {
                assert!(completed.is_none(), "should complete exactly once");
                completed = Some(p);
            }
        }

        let recovered = completed.expect("should complete");
        assert_eq!(recovered.sequence, 1);
        assert_eq!(recovered.wisdom.0, packet.wisdom.0);

        assert_eq!(receiver.stats().packets_complete, 1);
        assert_eq!(receiver.stats().packets_recovered, 0);
        assert_eq!(receiver.pending_count(), 0);
    }

    // -- FEC recovery --

    #[test]
    fn receive_with_fec_recovery() {
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(2, PEER_A, 0x22);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        // Drop fragment 4 (Karoo wind)
        for (i, raw) in wire.iter().enumerate() {
            if i == 4 {
                continue;
            }
            receiver.receive_fragment(PEER_A, raw);
        }

        // Should have completed via FEC
        assert_eq!(receiver.stats().packets_complete, 1);
        assert_eq!(receiver.stats().packets_recovered, 1);
        assert_eq!(receiver.stats().fragments_received, 10);
    }

    // -- Multi-peer isolation (the collision fix) --

    #[test]
    fn two_peers_same_thought_id_no_collision() {
        let mut receiver = MeshReceiver::new();

        // Both peers use sequence=1 → same thought_id
        let packet_a = test_packet(1, PEER_A, 0x33);
        let packet_b = test_packet(1, PEER_B, 0x44);
        assert_eq!(packet_a.thought_id(), packet_b.thought_id());

        let frags_a = packet_a.fragment();
        let frags_b = packet_b.fragment();
        let wire_a = to_wire(&frags_a);
        let wire_b = to_wire(&frags_b);

        // Interleave fragments from both peers
        let mut result_a = None;
        let mut result_b = None;
        for i in 0..11 {
            if let Some(p) = receiver.receive_fragment(PEER_A, &wire_a[i]) {
                result_a = Some(p);
            }
            if let Some(p) = receiver.receive_fragment(PEER_B, &wire_b[i]) {
                result_b = Some(p);
            }
        }

        // Both should complete with correct data (no cross-contamination)
        let recovered_a = result_a.expect("peer A should complete");
        let recovered_b = result_b.expect("peer B should complete");

        assert_eq!(recovered_a.wisdom.0, packet_a.wisdom.0);
        assert_eq!(recovered_b.wisdom.0, packet_b.wisdom.0);
        assert_ne!(
            recovered_a.wisdom.0, recovered_b.wisdom.0,
            "different seeds must produce different vectors"
        );

        assert_eq!(receiver.stats().packets_complete, 2);
        assert_eq!(receiver.peer_count(), 2);
    }

    // -- Whole packet reception --

    #[test]
    fn receive_whole_packet() {
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(5, PEER_A, 0x55);
        let bytes = packet.to_bytes();

        let result = receiver.receive_whole(&bytes).expect("should parse");
        assert_eq!(result.sequence, 5);
        assert_eq!(result.wisdom.0, packet.wisdom.0);

        assert_eq!(receiver.stats().packets_complete, 1);
        assert_eq!(receiver.stats().whole_packets, 1);
        assert_eq!(receiver.peer_count(), 1);
    }

    #[test]
    fn receive_whole_too_short_rejected() {
        let mut receiver = MeshReceiver::new();
        assert!(receiver.receive_whole(&[0; 100]).is_none());
        assert_eq!(receiver.stats().packets_complete, 0);
    }

    // -- Corrupt fragments --

    #[test]
    fn corrupt_fragment_counted() {
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(1, PEER_A, 0x66);
        let frags = packet.fragment();
        let mut wire = to_wire(&frags);

        // Corrupt fragment 2
        wire[2][12] ^= 0xFF;

        for raw in &wire {
            receiver.receive_fragment(PEER_A, raw);
        }

        assert_eq!(receiver.stats().fragments_corrupt, 1);
        // Should still complete via FEC (only 1 fragment lost)
        assert_eq!(receiver.stats().packets_complete, 1);
        assert_eq!(receiver.stats().packets_recovered, 1);
    }

    // -- Duplicate fragments --

    #[test]
    fn duplicate_fragments_counted() {
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(1, PEER_A, 0x77);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        // Feed each fragment twice
        for raw in &wire {
            receiver.receive_fragment(PEER_A, raw);
            receiver.receive_fragment(PEER_A, raw);
        }

        // First pass creates new assembly and completes it, second feeds
        // into nothing (assembly already removed). But duplicates within
        // the first pass are counted.
        assert_eq!(receiver.stats().packets_complete, 1);
        // The 11 duplicates from the second pass go to NEW assemblers
        // (old one was removed), but CRC pass so they're not "corrupt".
        // They create a new assembly that never completes.
    }

    // -- Timeout expiry --

    #[test]
    fn stale_assemblies_expired() {
        // Use zero timeout so everything expires immediately
        let mut receiver = MeshReceiver::new().with_timeout(Duration::ZERO);

        let packet = test_packet(1, PEER_A, 0x88);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        // Feed only first 5 fragments (incomplete)
        for raw in &wire[..5] {
            receiver.receive_fragment(PEER_A, raw);
        }
        assert_eq!(receiver.pending_count(), 1);

        // Expire — should clean up the incomplete assembly
        // (timeout is ZERO, so any elapsed time triggers expiry)
        std::thread::sleep(Duration::from_millis(1));
        let expired = receiver.expire_stale();

        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].source, PEER_A);
        assert_eq!(receiver.pending_count(), 0);
        assert_eq!(receiver.stats().packets_expired, 1);
    }

    // -- Capacity limits --

    #[test]
    fn capacity_evicts_oldest() {
        let mut receiver = MeshReceiver::new()
            .with_max_pending(2)
            .with_timeout(Duration::from_secs(3600)); // long timeout

        // Create 3 incomplete assemblies (exceeds max_pending of 2)
        let packet_a = test_packet(1, PEER_A, 0x99);
        let packet_b = test_packet(2, PEER_B, 0xAA);
        let packet_c = test_packet(3, PEER_C, 0xBB);

        let wire_a = to_wire(&packet_a.fragment());
        let wire_b = to_wire(&packet_b.fragment());
        let wire_c = to_wire(&packet_c.fragment());

        // Feed first fragment only (creates pending assembly)
        receiver.receive_fragment(PEER_A, &wire_a[0]);
        receiver.receive_fragment(PEER_B, &wire_b[0]);
        assert_eq!(receiver.pending_count(), 2);

        // Third peer should trigger eviction of oldest (PEER_A)
        receiver.receive_fragment(PEER_C, &wire_c[0]);
        assert_eq!(receiver.pending_count(), 2);
        assert_eq!(receiver.stats().packets_expired, 1); // oldest evicted
    }

    // -- Peer discovery --

    #[test]
    fn peers_discovered_from_packets() {
        let mut receiver = MeshReceiver::new();

        let packet_a = test_packet(1, PEER_A, 0xDD);
        let packet_b = test_packet(1, PEER_B, 0xEE);

        // Complete both
        for raw in &to_wire(&packet_a.fragment()) {
            receiver.receive_fragment(PEER_A, raw);
        }
        for raw in &to_wire(&packet_b.fragment()) {
            receiver.receive_fragment(PEER_B, raw);
        }

        assert_eq!(receiver.peer_count(), 2);

        let peer_a = receiver.peer(&PEER_A).unwrap();
        assert_eq!(peer_a.source, PEER_A);
        assert_eq!(peer_a.packets_received, 1);
        assert!((peer_a.last_phi - 0.73).abs() < 1e-6);

        let peer_b = receiver.peer(&PEER_B).unwrap();
        assert_eq!(peer_b.packets_received, 1);
    }

    #[test]
    fn peer_phi_updated_on_new_packet() {
        let mut receiver = MeshReceiver::new();

        let mut packet1 = test_packet(1, PEER_A, 0xF1);
        packet1.phi = 0.5;
        let mut packet2 = test_packet(2, PEER_A, 0xF2);
        packet2.phi = 0.9;

        for raw in &to_wire(&packet1.fragment()) {
            receiver.receive_fragment(PEER_A, raw);
        }
        assert!((receiver.peer(&PEER_A).unwrap().last_phi - 0.5).abs() < 1e-6);

        for raw in &to_wire(&packet2.fragment()) {
            receiver.receive_fragment(PEER_A, raw);
        }
        assert!((receiver.peer(&PEER_A).unwrap().last_phi - 0.9).abs() < 1e-6);
        assert_eq!(receiver.peer(&PEER_A).unwrap().packets_received, 2);
    }

    // -- Full integration: three peers, interleaved, with loss --

    #[test]
    fn three_peers_interleaved_with_loss() {
        let mut receiver = MeshReceiver::new();

        let packet_a = test_packet(10, PEER_A, 0xA0);
        let packet_b = test_packet(20, PEER_B, 0xB0);
        let packet_c = test_packet(30, PEER_C, 0xC0);

        let wire_a = to_wire(&packet_a.fragment());
        let wire_b = to_wire(&packet_b.fragment());
        let wire_c = to_wire(&packet_c.fragment());

        // Interleave all three, dropping one fragment each
        let mut results = Vec::new();
        for i in 0..11 {
            if i != 2 {
                if let Some(p) = receiver.receive_fragment(PEER_A, &wire_a[i]) {
                    results.push(('A', p));
                }
            }
            if i != 7 {
                if let Some(p) = receiver.receive_fragment(PEER_B, &wire_b[i]) {
                    results.push(('B', p));
                }
            }
            if i != 0 {
                if let Some(p) = receiver.receive_fragment(PEER_C, &wire_c[i]) {
                    results.push(('C', p));
                }
            }
        }

        // All three should complete via FEC
        assert_eq!(results.len(), 3);
        assert_eq!(receiver.stats().packets_complete, 3);
        assert_eq!(receiver.stats().packets_recovered, 3); // all used FEC
        assert_eq!(receiver.peer_count(), 3);

        // Verify data integrity
        for (label, recovered) in &results {
            match label {
                'A' => assert_eq!(recovered.wisdom.0, packet_a.wisdom.0),
                'B' => assert_eq!(recovered.wisdom.0, packet_b.wisdom.0),
                'C' => assert_eq!(recovered.wisdom.0, packet_c.wisdom.0),
                _ => unreachable!(),
            }
        }
    }

    // -- Stats --

    #[test]
    fn stats_accumulate() {
        let receiver = MeshReceiver::new();
        let stats = receiver.stats();
        assert_eq!(stats.packets_complete, 0);
        assert_eq!(stats.fragments_received, 0);
    }

    // -- Item 6: Edge case tests --

    #[test]
    fn test_receive_whole_compressed_envelope() {
        use crate::swarm::mesh::{COMPRESS_NONE, WISDOM_PACKET_SIZE, compress_packet};
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(10, PEER_A, 0xCC);
        let raw = packet.to_bytes();
        // Create a COMPRESS_NONE envelope manually
        let mut envelope = Vec::with_capacity(1 + WISDOM_PACKET_SIZE);
        envelope.push(COMPRESS_NONE);
        envelope.extend_from_slice(&raw);
        let result = receiver
            .receive_whole(&envelope)
            .expect("should parse compressed envelope");
        assert_eq!(result.sequence, 10);
        assert_eq!(result.wisdom.0, packet.wisdom.0);
        // Also verify that compress_packet produces a parseable envelope
        let auto_envelope = compress_packet(&raw);
        let mut receiver2 = MeshReceiver::new();
        let result2 = receiver2
            .receive_whole(&auto_envelope)
            .expect("should parse auto-compressed envelope");
        assert_eq!(result2.sequence, 10);
    }

    #[test]
    fn test_receive_whole_legacy_backward_compat() {
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(20, PEER_B, 0xDD);
        // Feed raw packet bytes (no envelope header) — legacy format
        let raw = packet.to_bytes();
        let result = receiver
            .receive_whole(&raw)
            .expect("should parse legacy raw packet");
        assert_eq!(result.sequence, 20);
        assert_eq!(result.wisdom.0, packet.wisdom.0);
    }

    // -- Encryption pipeline tests --

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_receiver_decrypts_whole_packet() {
        use crate::swarm::mesh::{compress_packet, encrypt_packet};

        let key = [0xCD; 32];
        let packet = test_packet(42, PEER_A, 0xEE);
        let raw = packet.to_bytes();

        // Manually: compress → encrypt (mimics DualLayerMesh send path)
        let compressed = compress_packet(&raw);
        let encrypted = encrypt_packet(&compressed, &key, &packet.source_id, 0xAB, packet.sequence);

        // Receiver with matching key should decrypt → decompress → parse
        let mut receiver = MeshReceiver::new().with_encryption_key(key);
        let result = receiver
            .receive_whole(&encrypted)
            .expect("should decrypt and parse");
        assert_eq!(result.sequence, 42);
        assert_eq!(result.wisdom.0, packet.wisdom.0);
        assert_eq!(receiver.stats().packets_decrypt_failed, 0);
    }

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_receiver_rejects_unencrypted_when_key_set() {
        use crate::swarm::mesh::compress_packet;

        let key = [0xEF; 32];
        let packet = test_packet(99, PEER_B, 0xFF);
        let raw = packet.to_bytes();

        // Send unencrypted compressed packet to a receiver WITH key set.
        // Decryption fails and the packet is rejected — when encryption is
        // enabled, only authenticated ciphertext is accepted.
        let compressed = compress_packet(&raw);

        let mut receiver = MeshReceiver::new().with_encryption_key(key);
        let result = receiver.receive_whole(&compressed);
        assert!(
            result.is_none(),
            "Unencrypted data should be rejected when encryption key is set"
        );
        assert_eq!(receiver.stats().packets_decrypt_failed, 1);
    }

    // -- Item 5: Fragment reorder validation --

    #[test]
    fn test_reorder_within_range_accepted() {
        // Fragments arriving out of order but within total_fragments range
        // should be accepted (normal radio behavior).
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(1, PEER_A, 0xAB);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        // Feed fragments in reverse order: 10, 9, 8, ... 0
        let mut completed = false;
        for raw in wire.iter().rev() {
            if receiver.receive_fragment(PEER_A, raw).is_some() {
                completed = true;
            }
        }

        assert!(completed, "Reverse-order delivery should still complete");
        assert_eq!(
            receiver.stats().fragments_reordered,
            0,
            "In-range reordering should not be rejected"
        );
        assert_eq!(receiver.stats().packets_complete, 1);
    }

    #[test]
    fn test_fec_after_early_data_fragment_accepted() {
        // FEC fragment (index 10) arriving right after data fragment 0
        // should NOT be rejected — gap=10 is within total_fragments=11.
        let mut receiver = MeshReceiver::new();
        let packet = test_packet(1, PEER_A, 0xCD);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        // Feed fragment 0, then FEC (index 10), then the rest
        receiver.receive_fragment(PEER_A, &wire[0]);
        receiver.receive_fragment(PEER_A, &wire[10]); // FEC, gap = 10

        assert_eq!(
            receiver.stats().fragments_reordered,
            0,
            "FEC fragment should not be rejected as reordered"
        );

        // Feed remaining fragments to complete assembly
        let mut completed = false;
        for raw in &wire[1..10] {
            if receiver.receive_fragment(PEER_A, raw).is_some() {
                completed = true;
            }
        }
        assert!(completed, "Assembly should complete with all fragments");
    }

    // -- Item 12: Key fingerprint tracking --

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_key_fingerprint_consistent_key_passes() {
        // All fragments received under the same encryption key — should complete.
        use crate::swarm::mesh::encrypt_fragment;

        let key = [0x55u8; 32];
        let source = PEER_A;
        let packet = test_packet(1, source, 0xEF);
        let frags = packet.fragment();

        let encrypted: Vec<Vec<u8>> = frags
            .iter()
            .map(|f| {
                let mut buf = [0u8; LORA_MTU];
                let len = f.to_bytes(&mut buf);
                encrypt_fragment(&buf[..len], &key, &source, f.thought_id, f.fragment_index)
            })
            .collect();

        let mut receiver = MeshReceiver::new()
            .with_encryption_key(key)
            .with_fragment_encryption(true);

        let mut completed = false;
        for raw in &encrypted {
            if receiver.receive_fragment(source, raw).is_some() {
                completed = true;
            }
        }

        assert!(completed, "Same-key fragments should reassemble");
        assert_eq!(
            receiver.stats().fragments_key_mismatch,
            0,
            "No key mismatch when key is consistent"
        );
    }

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_key_fingerprint_mid_stream_change_rejected() {
        // Start receiving with key A, then change to key B mid-stream.
        // Fragments under key B should be rejected due to fingerprint mismatch.
        let key_a = [0xAA; 32];
        let key_b = [0xBB; 32];
        let source = PEER_A;

        let packet = test_packet(1, source, 0xFE);
        let frags = packet.fragment();
        let wire = to_wire(&frags);

        let mut receiver = MeshReceiver::new().with_encryption_key(key_a);

        // Feed first 3 fragments under key A (unencrypted fragment data, key
        // only used for fingerprint tracking in non-fragment-encryption mode)
        for raw in &wire[..3] {
            receiver.receive_fragment(source, raw);
        }

        // Change key mid-stream
        receiver.set_encryption_key(Some(key_b));

        // Feed remaining fragments — fingerprint should mismatch
        for raw in &wire[3..] {
            receiver.receive_fragment(source, raw);
        }

        assert!(
            receiver.stats().fragments_key_mismatch > 0,
            "Mid-stream key change should trigger fingerprint mismatch"
        );
    }
}
