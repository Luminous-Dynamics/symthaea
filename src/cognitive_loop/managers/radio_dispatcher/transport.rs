// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transport layer: compression, FEC, routing, encryption, and peer discovery.

use super::tier::{CompressionStrategy, RadioTier};
use std::collections::{HashMap, VecDeque};

use super::super::super::thresholds::{
    RADIO_BEACON_SIZE, RADIO_CRYPTO_NONCE_SIZE, RADIO_FEC_MIN_PAYLOAD,
};

// ═══════════════════════════════════════════════════════════════════════════════
// DELTA COMPRESSOR — XOR + RLE for BinaryHV diffs
// ═══════════════════════════════════════════════════════════════════════════════

/// Compressed delta between two BinaryHV vectors.
///
/// Uses XOR to find changed bits, then run-length encodes the result.
/// For incremental cognitive state updates where few dimensions flip per cycle,
/// this typically compresses 2,048 bytes → 50-200 bytes.
#[derive(Debug, Clone)]
pub struct CompressedDelta {
    /// RLE-encoded XOR diff. Format: repeated (count: u16 LE, byte: u8) triples.
    /// A run of zeros means "unchanged", a run of non-zero means "these bits flipped".
    pub rle_data: Vec<u8>,
    /// Number of bytes that differ (for telemetry).
    pub changed_bytes: usize,
    /// Whether this is a full vector (not a delta) — used on reconnect.
    pub is_full: bool,
}

impl CompressedDelta {
    /// Compute XOR delta between two BinaryHV byte arrays, then RLE-compress.
    ///
    /// If the compressed result would be larger than the raw vector (high entropy),
    /// returns a full vector instead.
    pub fn from_diff(previous: &[u8; 2048], current: &[u8; 2048]) -> Self {
        // XOR to find changed bits
        let mut diff = [0u8; 2048];
        let mut changed_bytes = 0usize;
        for i in 0..2048 {
            diff[i] = previous[i] ^ current[i];
            if diff[i] != 0 {
                changed_bytes += 1;
            }
        }

        let rle_data = Self::rle_encode(&diff);

        // If RLE is larger than raw, just send full vector
        if rle_data.len() >= 2048 {
            return Self::full(current);
        }

        Self {
            rle_data,
            changed_bytes,
            is_full: false,
        }
    }

    /// Create a full (non-delta) compressed payload for initial sync or reconnect.
    pub fn full(data: &[u8; 2048]) -> Self {
        Self {
            rle_data: data.to_vec(),
            changed_bytes: 2048,
            is_full: true,
        }
    }

    /// Apply this delta to a previous BinaryHV to reconstruct the current one.
    ///
    /// For full vectors, ignores `previous` and returns the stored data directly.
    pub fn apply(&self, previous: &[u8; 2048]) -> Option<[u8; 2048]> {
        if self.is_full {
            if self.rle_data.len() != 2048 {
                return None;
            }
            let mut result = [0u8; 2048];
            result.copy_from_slice(&self.rle_data);
            return Some(result);
        }

        let diff = Self::rle_decode(&self.rle_data)?;
        if diff.len() != 2048 {
            return None;
        }

        let mut result = *previous;
        for i in 0..2048 {
            result[i] ^= diff[i];
        }
        Some(result)
    }

    /// Compressed size in bytes.
    pub fn wire_size(&self) -> usize {
        self.rle_data.len()
    }

    /// Compression ratio (1.0 = no compression, 0.0 = perfectly compressed).
    pub fn compression_ratio(&self) -> f64 {
        self.rle_data.len() as f64 / 2048.0
    }

    /// RLE encode: repeated (count_hi: u8, count_lo: u8, byte: u8) triples.
    /// Count is u16 LE to handle runs up to 65535.
    pub(super) fn rle_encode(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() / 4);
        if data.is_empty() {
            return result;
        }

        let mut run_byte = data[0];
        let mut run_len: u16 = 1;

        for &b in &data[1..] {
            if b == run_byte && run_len < u16::MAX {
                run_len += 1;
            } else {
                // Emit run
                result.extend_from_slice(&run_len.to_le_bytes());
                result.push(run_byte);
                run_byte = b;
                run_len = 1;
            }
        }
        // Emit final run
        result.extend_from_slice(&run_len.to_le_bytes());
        result.push(run_byte);

        result
    }

    /// RLE decode: inverse of `rle_encode`.
    pub(super) fn rle_decode(data: &[u8]) -> Option<Vec<u8>> {
        if data.len() % 3 != 0 {
            return None;
        }

        let mut result = Vec::with_capacity(2048);
        let mut i = 0;
        while i + 2 < data.len() {
            let count = u16::from_le_bytes([data[i], data[i + 1]]) as usize;
            let byte = data[i + 2];
            // Safety cap: prevent OOM from malicious data
            if result.len() + count > 65536 {
                return None;
            }
            result.extend(std::iter::repeat(byte).take(count));
            i += 3;
        }
        Some(result)
    }
}

/// Result of tier-adaptive compression.
#[derive(Debug, Clone)]
pub struct TierCompressedPayload {
    /// The compression strategy used.
    pub strategy: CompressionStrategy,
    /// Compressed payload bytes (wire format).
    pub data: Vec<u8>,
    /// Original uncompressed size.
    pub original_size: usize,
}

impl TierCompressedPayload {
    /// Wire size in bytes.
    pub fn wire_size(&self) -> usize {
        self.data.len()
    }

    /// Compression ratio (0.0 = perfect, 1.0 = no compression).
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.data.len() as f64 / self.original_size as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEC — Forward Error Correction (Reed-Solomon approximation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple XOR-based FEC for Metro/Regional tiers.
///
/// Not full Reed-Solomon (would require a GF(2^8) library), but provides
/// single-block recovery via XOR parity. For N data blocks, produces 1 parity
/// block that can recover any single lost block.
///
/// Basis: Lin & Costello (2004) — error control coding fundamentals.
///
/// NOTE: For LoRa fragmentation, use `swarm::mesh::lora_fragment::FragmentAssembler`
/// which provides the same XOR FEC integrated with CRC-16 and reassembly.
/// This encoder is for non-LoRa payloads (Metro/Regional custom frames).
pub struct FecEncoder;

impl FecEncoder {
    /// Encode data with XOR parity FEC.
    ///
    /// Splits data into `block_size`-byte blocks and appends one XOR parity block.
    /// Returns the data with parity appended.
    pub fn encode(data: &[u8], block_size: usize) -> Vec<u8> {
        if data.len() < RADIO_FEC_MIN_PAYLOAD || block_size == 0 {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        let mut parity = vec![0u8; block_size];

        for chunk in data.chunks(block_size) {
            for (i, &b) in chunk.iter().enumerate() {
                parity[i] ^= b;
            }
        }

        result.extend_from_slice(&parity);
        result
    }

    /// Decode FEC-encoded data, recovering from a single lost block.
    ///
    /// `lost_block_index`: which block (0-based) was lost. If `None`, just strips parity.
    pub fn decode(encoded: &[u8], block_size: usize, lost_block_index: Option<usize>) -> Vec<u8> {
        if block_size == 0 || encoded.len() <= block_size {
            return encoded.to_vec();
        }

        let data_len = encoded.len() - block_size;
        let parity = &encoded[data_len..];
        let mut data = encoded[..data_len].to_vec();

        if let Some(lost_idx) = lost_block_index {
            let start = lost_idx * block_size;
            let end = (start + block_size).min(data_len);

            // Recover: parity XOR all other blocks
            let mut recovered = parity.to_vec();
            for (block_idx, chunk) in data.chunks(block_size).enumerate() {
                if block_idx != lost_idx {
                    for (i, &b) in chunk.iter().enumerate() {
                        if i < recovered.len() {
                            recovered[i] ^= b;
                        }
                    }
                }
            }

            // Write recovered block
            for i in start..end {
                if i - start < recovered.len() {
                    data[i] = recovered[i - start];
                }
            }
        }

        data
    }

    /// Calculate FEC overhead for a given data size and block size.
    pub fn overhead(data_len: usize, block_size: usize) -> usize {
        if data_len < RADIO_FEC_MIN_PAYLOAD || block_size == 0 {
            0
        } else {
            block_size
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PEER DISCOVERY — Lightweight beacon protocol
// ═══════════════════════════════════════════════════════════════════════════════

/// Discovery beacon broadcast on Regional/Metro tiers.
///
/// Minimal payload (24 bytes) designed to fit within Regional MTU (50 bytes):
/// - 8 bytes: node ID
/// - 8 bytes: capabilities hash
/// - 4 bytes: cycle counter (for liveliness)
/// - 4 bytes: network health + tier mask + reserved
#[derive(Debug, Clone)]
pub struct DiscoveryBeacon {
    /// First 8 bytes of the node's identity.
    pub node_id: [u8; 8],
    /// Hash of the node's capabilities (features, firmware version, etc.).
    pub capabilities_hash: [u8; 8],
    /// Cycle counter at beacon time (monotonically increasing).
    pub cycle_counter: u32,
    /// Current network health level (0-3).
    pub network_health: u8,
    /// Bitmask of available tiers (bit 0 = Local, 1 = Metro, 2 = Regional).
    pub tier_mask: u8,
}

impl DiscoveryBeacon {
    /// Serialize to wire format (24 bytes).
    pub fn to_bytes(&self) -> [u8; RADIO_BEACON_SIZE] {
        let mut buf = [0u8; RADIO_BEACON_SIZE];
        buf[0..8].copy_from_slice(&self.node_id);
        buf[8..16].copy_from_slice(&self.capabilities_hash);
        buf[16..20].copy_from_slice(&self.cycle_counter.to_le_bytes());
        buf[20] = self.network_health;
        buf[21] = self.tier_mask;
        // 22-23: reserved
        buf
    }

    /// Deserialize from wire format.
    pub fn from_bytes(data: &[u8; RADIO_BEACON_SIZE]) -> Self {
        let mut node_id = [0u8; 8];
        node_id.copy_from_slice(&data[0..8]);
        let mut capabilities_hash = [0u8; 8];
        capabilities_hash.copy_from_slice(&data[8..16]);
        let cycle_counter = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        Self {
            node_id,
            capabilities_hash,
            cycle_counter,
            network_health: data[20],
            tier_mask: data[21],
        }
    }

    /// Generate tier mask from availability array.
    pub fn tier_mask_from(available: &[bool; 3]) -> u8 {
        let mut mask = 0u8;
        if available[0] {
            mask |= 0x01;
        }
        if available[1] {
            mask |= 0x02;
        }
        if available[2] {
            mask |= 0x04;
        }
        mask
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-HOP RELAY ROUTING — Mesh route table
// ═══════════════════════════════════════════════════════════════════════════════

/// A route to a peer via zero or more relay hops.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// Destination node ID (first 8 bytes).
    pub destination: [u8; 8],
    /// Next-hop node ID (direct neighbor to forward to).
    pub next_hop: [u8; 8],
    /// Number of hops to destination (0 = direct neighbor).
    pub hop_count: u8,
    /// Best tier to reach next hop.
    pub tier: RadioTier,
    /// Cycle when this route was last refreshed.
    pub last_seen_cycle: u64,
    /// Estimated link quality (0.0–1.0).
    pub link_quality: f32,
}

/// Mesh routing table with TTL-based expiry.
///
/// Routes are learned from beacon reception and forwarded route advertisements.
/// Stale routes are pruned each cycle to prevent routing to departed nodes.
///
/// Basis: Perkins & Royer (1999) — Ad hoc On-Demand Distance Vector (AODV).
pub struct RouteTable {
    /// Known routes, keyed by destination node ID.
    routes: HashMap<[u8; 8], RouteEntry>,
    /// Maximum entries.
    capacity: usize,
}

impl RouteTable {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            routes: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Add or update a route. Prefers shorter hop counts and higher link quality.
    pub fn update(&mut self, entry: RouteEntry) {
        if let Some(existing) = self.routes.get(&entry.destination) {
            // Only update if better (fewer hops, or same hops but better quality)
            if entry.hop_count < existing.hop_count
                || (entry.hop_count == existing.hop_count
                    && entry.link_quality > existing.link_quality)
            {
                self.routes.insert(entry.destination, entry);
            } else {
                // Just refresh the timestamp
                if let Some(e) = self.routes.get_mut(&entry.destination) {
                    e.last_seen_cycle = entry.last_seen_cycle;
                }
            }
        } else if self.routes.len() < self.capacity {
            self.routes.insert(entry.destination, entry);
        }
    }

    /// Look up the best route to a destination.
    pub fn lookup(&self, destination: &[u8; 8]) -> Option<&RouteEntry> {
        self.routes.get(destination)
    }

    /// Prune routes older than `max_age_cycles` from `current_cycle`.
    pub fn prune(&mut self, current_cycle: u64, max_age_cycles: u64) {
        self.routes.retain(|_, entry| {
            current_cycle.saturating_sub(entry.last_seen_cycle) < max_age_cycles
        });
    }

    /// Number of known routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// All known destinations.
    pub fn destinations(&self) -> Vec<[u8; 8]> {
        self.routes.keys().copied().collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESH ENCRYPTION — Per-peer session encryption
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-peer encryption session state.
///
/// Uses a pre-shared key model (PSK) with per-peer derived keys.
/// In production, this would use X25519 key exchange; here we model
/// the session state and nonce management.
///
/// Basis: Bernstein (2008) — ChaCha20-Poly1305 AEAD construction.
#[derive(Debug, Clone)]
pub struct PeerSession {
    /// Peer node ID.
    pub peer_id: [u8; 8],
    /// Shared session key (32 bytes, derived from PSK + peer IDs).
    pub session_key: [u8; 32],
    /// Outgoing nonce counter (monotonically increasing, prevents replay).
    pub tx_nonce_counter: u64,
    /// Last received nonce (for replay detection).
    pub rx_nonce_seen: u64,
    /// Cycle when this session was established.
    pub established_cycle: u64,
}

impl PeerSession {
    /// Create a new session with a derived key.
    ///
    /// In production, `session_key` would come from X25519 DH + HKDF.
    /// Here we accept it as a parameter.
    pub fn new(peer_id: [u8; 8], session_key: [u8; 32], cycle: u64) -> Self {
        Self {
            peer_id,
            session_key,
            tx_nonce_counter: 0,
            rx_nonce_seen: 0,
            established_cycle: cycle,
        }
    }

    /// Generate the next nonce (12 bytes: 4 zero + 8 counter LE).
    pub fn next_nonce(&mut self) -> [u8; RADIO_CRYPTO_NONCE_SIZE] {
        self.tx_nonce_counter += 1;
        let mut nonce = [0u8; RADIO_CRYPTO_NONCE_SIZE];
        nonce[4..12].copy_from_slice(&self.tx_nonce_counter.to_le_bytes());
        nonce
    }

    /// Check if a received nonce is valid (not replayed).
    pub fn check_nonce(&mut self, nonce_counter: u64) -> bool {
        if nonce_counter <= self.rx_nonce_seen {
            return false; // Replay detected
        }
        self.rx_nonce_seen = nonce_counter;
        true
    }
}

/// Mesh encryption manager — tracks per-peer session keys and nonces.
///
/// In a real deployment, this would integrate with Mycelix identity
/// for peer authentication and X25519 key exchange.
pub struct MeshEncryption {
    /// Active sessions keyed by peer node ID.
    sessions: HashMap<[u8; 8], PeerSession>,
    /// Maximum sessions.
    capacity: usize,
}

impl MeshEncryption {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            sessions: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Register a peer session.
    pub fn add_session(&mut self, session: PeerSession) {
        if self.sessions.len() < self.capacity {
            self.sessions.insert(session.peer_id, session);
        }
    }

    /// Get a session for a peer.
    pub fn get_session(&self, peer_id: &[u8; 8]) -> Option<&PeerSession> {
        self.sessions.get(peer_id)
    }

    /// Get a mutable session for a peer.
    pub fn get_session_mut(&mut self, peer_id: &[u8; 8]) -> Option<&mut PeerSession> {
        self.sessions.get_mut(peer_id)
    }

    /// Remove a peer session.
    pub fn remove_session(&mut self, peer_id: &[u8; 8]) -> Option<PeerSession> {
        self.sessions.remove(peer_id)
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Encrypt plaintext using XOR scrambling (test/simulation placeholder).
    ///
    /// **WARNING: NOT cryptographically secure.** For production mesh encryption,
    /// use `swarm::mesh::mod.rs` which implements real ChaCha20-Poly1305 via the
    /// `chacha20poly1305` crate (feature: `mesh-encryption`).
    ///
    /// This placeholder preserves the API shape for unit testing session
    /// management (nonce tracking, replay detection) without requiring
    /// the full crypto dependency chain.
    pub fn encrypt(
        key: &[u8; 32],
        nonce: &[u8; RADIO_CRYPTO_NONCE_SIZE],
        plaintext: &[u8],
    ) -> Vec<u8> {
        let mut ciphertext = plaintext.to_vec();
        for (i, byte) in ciphertext.iter_mut().enumerate() {
            *byte ^= key[i % 32] ^ nonce[i % RADIO_CRYPTO_NONCE_SIZE];
        }
        // 16-byte simulated auth tag
        let mut tag = [0u8; 16];
        for (i, &b) in ciphertext.iter().enumerate() {
            tag[i % 16] ^= b;
        }
        ciphertext.extend_from_slice(&tag);
        ciphertext
    }

    /// Decrypt ciphertext using XOR scrambling (test/simulation placeholder).
    ///
    /// **WARNING: NOT cryptographically secure.** See `encrypt()` doc.
    pub fn decrypt(
        key: &[u8; 32],
        nonce: &[u8; RADIO_CRYPTO_NONCE_SIZE],
        ciphertext: &[u8],
    ) -> Option<Vec<u8>> {
        if ciphertext.len() < 16 {
            return None;
        }
        let data_len = ciphertext.len() - 16;
        let data = &ciphertext[..data_len];
        let tag = &ciphertext[data_len..];

        let mut expected_tag = [0u8; 16];
        for (i, &b) in data.iter().enumerate() {
            expected_tag[i % 16] ^= b;
        }
        // Constant-time comparison to prevent timing attacks on the auth tag.
        // Uses the same constant_time_eq from the handshake module.
        if !crate::swarm::handshake::constant_time_eq(tag, &expected_tag) {
            return None;
        }

        let mut plaintext = data.to_vec();
        for (i, byte) in plaintext.iter_mut().enumerate() {
            *byte ^= key[i % 32] ^ nonce[i % RADIO_CRYPTO_NONCE_SIZE];
        }
        Some(plaintext)
    }
}
