// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LoRa Radio Fragmentation Protocol for HDC Payloads
//!
//! Deterministic fragmentation of binary hypervectors for transmission over
//! LoRa (Long Range) radio links. Designed for the Mycelial Mesh: off-grid,
//! solar-powered Symthaea nodes exchanging 2 KB "Wisdom Vectors" over 10-15 km.
//!
//! # Protocol
//!
//! A 2,048-byte `BinaryHV` fragments into exactly **10 data frames + 1 XOR
//! parity frame = 11 LoRa transmissions**. At SF7/BW125kHz, this takes ~3
//! seconds — fast enough for Cruise-mode philosophical updates.
//!
//! Single fragment loss is recoverable without retransmission via XOR FEC.
//!
//! # Frame Format
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  8-byte header                                   │
//! │  ┌─────────────────────────────────────────────┐ │
//! │  │ thought_id     u16   packet identifier      │ │
//! │  │ fragment_index u8    0..total_fragments      │ │
//! │  │ total_fragments u8   data + FEC count        │ │
//! │  │ payload_len    u8    valid bytes (≤ 214)     │ │
//! │  │ flags          u8    bit 0 = FEC parity      │ │
//! │  │ checksum       u16   CRC-16/CCITT            │ │
//! │  └─────────────────────────────────────────────┘ │
//! │  payload: up to 214 bytes                        │
//! └──────────────────────────────────────────────────┘
//! ```

/// Maximum LoRa frame size (SF7, BW 125kHz, CR 4/5, 868 MHz).
pub const LORA_MTU: usize = 222;

/// Fragment header size in bytes.
pub const HEADER_SIZE: usize = 8;

/// Usable payload bytes per fragment.
pub const PAYLOAD_SIZE: usize = LORA_MTU - HEADER_SIZE; // 214

/// Flag bit: this fragment carries XOR Forward Error Correction parity.
pub const FLAG_FEC: u8 = 0x01;

// ============================================================================
// CRC-16/CCITT
// ============================================================================

/// Incremental CRC-16/CCITT calculator (poly 0x1021, init 0xFFFF).
///
/// Standard checksum for radio protocols — no lookup table needed.
struct Crc16(u16);

impl Crc16 {
    fn new() -> Self {
        Self(0xFFFF)
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.0 ^= (byte as u16) << 8;
            for _ in 0..8 {
                if self.0 & 0x8000 != 0 {
                    self.0 = (self.0 << 1) ^ 0x1021;
                } else {
                    self.0 <<= 1;
                }
            }
        }
    }

    fn finalize(self) -> u16 {
        self.0
    }
}

/// Compute CRC-16/CCITT over a byte slice.
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc = Crc16::new();
    crc.update(data);
    crc.finalize()
}

// ============================================================================
// LORA FRAGMENT
// ============================================================================

/// A single LoRa radio fragment, ready for transmission.
///
/// 8-byte header + up to 214 bytes payload. Fits within LoRa's ~222-byte MTU.
/// Stack-allocated (no heap) for zero-cost creation on embedded targets.
#[derive(Clone)]
pub struct LoRaFragment {
    /// Rolling packet identifier for reassembly grouping.
    pub thought_id: u16,
    /// Index within the packet (0-based, FEC is last).
    pub fragment_index: u8,
    /// Total fragment count including FEC parity.
    pub total_fragments: u8,
    /// Valid payload bytes in this fragment (always ≤ PAYLOAD_SIZE).
    pub payload_len: u8,
    /// Bit 0: FEC parity fragment. Bits 1-7: reserved.
    pub flags: u8,
    /// CRC-16/CCITT over header fields (excl. checksum) + payload.
    pub checksum: u16,
    /// Fragment payload (only first `payload_len` bytes are meaningful).
    pub payload: [u8; PAYLOAD_SIZE],
}

impl std::fmt::Debug for LoRaFragment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoRaFragment")
            .field("thought_id", &self.thought_id)
            .field(
                "fragment",
                &format_args!("{}/{}", self.fragment_index, self.total_fragments),
            )
            .field("payload_len", &self.payload_len)
            .field("fec", &self.is_fec())
            .field("checksum", &format_args!("{:#06x}", self.checksum))
            .finish()
    }
}

impl LoRaFragment {
    /// Serialize to a byte buffer for radio transmission.
    ///
    /// Returns the number of bytes written: `HEADER_SIZE + payload_len`.
    /// The buffer must be at least `HEADER_SIZE + payload_len` bytes.
    pub fn to_bytes(&self, buf: &mut [u8]) -> usize {
        let total = HEADER_SIZE + self.payload_len as usize;
        assert!(
            buf.len() >= total,
            "buffer too small: need {total}, got {}",
            buf.len()
        );

        let id_bytes = self.thought_id.to_le_bytes();
        buf[0] = id_bytes[0];
        buf[1] = id_bytes[1];
        buf[2] = self.fragment_index;
        buf[3] = self.total_fragments;
        buf[4] = self.payload_len;
        buf[5] = self.flags;
        let ck_bytes = self.checksum.to_le_bytes();
        buf[6] = ck_bytes[0];
        buf[7] = ck_bytes[1];
        buf[HEADER_SIZE..total].copy_from_slice(&self.payload[..self.payload_len as usize]);

        total
    }

    /// Deserialize from a received byte buffer.
    ///
    /// Returns `None` if the buffer is too short, fields are invalid, or
    /// the CRC-16 checksum fails (corrupted in transit).
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < HEADER_SIZE {
            return None;
        }

        let thought_id = u16::from_le_bytes([buf[0], buf[1]]);
        let fragment_index = buf[2];
        let total_fragments = buf[3];
        let payload_len = buf[4];
        let flags = buf[5];
        let checksum = u16::from_le_bytes([buf[6], buf[7]]);

        // Sanity checks
        if total_fragments == 0 || fragment_index >= total_fragments {
            return None;
        }
        if payload_len as usize > PAYLOAD_SIZE {
            return None;
        }
        if buf.len() < HEADER_SIZE + payload_len as usize {
            return None;
        }

        let mut payload = [0u8; PAYLOAD_SIZE];
        payload[..payload_len as usize]
            .copy_from_slice(&buf[HEADER_SIZE..HEADER_SIZE + payload_len as usize]);

        let frag = Self {
            thought_id,
            fragment_index,
            total_fragments,
            payload_len,
            flags,
            checksum,
            payload,
        };

        // Verify integrity
        if frag.compute_checksum() != checksum {
            return None;
        }

        Some(frag)
    }

    /// Whether this is the XOR FEC parity fragment.
    #[inline]
    pub fn is_fec(&self) -> bool {
        self.flags & FLAG_FEC != 0
    }

    /// Compute CRC-16/CCITT over header fields (excluding checksum) + payload.
    fn compute_checksum(&self) -> u16 {
        let mut crc = Crc16::new();
        crc.update(&self.thought_id.to_le_bytes());
        crc.update(&[
            self.fragment_index,
            self.total_fragments,
            self.payload_len,
            self.flags,
        ]);
        crc.update(&self.payload[..self.payload_len as usize]);
        crc.finalize()
    }
}

// ============================================================================
// FRAGMENTATION
// ============================================================================

/// Fragment a payload into LoRa-sized frames with XOR FEC parity.
///
/// Returns `ceil(payload.len() / 214) + 1` fragments. The last fragment is
/// an XOR parity of all data fragments, enabling recovery of any single lost
/// data fragment without retransmission.
///
/// # Panics
///
/// Panics if `payload` is empty or exceeds `254 * 214` bytes (~53 KB).
///
/// # Example: BinaryHV (2,048 bytes)
///
/// ```text
/// Fragment  0: bytes    0..214  (214 bytes)
/// Fragment  1: bytes  214..428  (214 bytes)
/// ...
/// Fragment  8: bytes 1712..1926 (214 bytes)
/// Fragment  9: bytes 1926..2048 (122 bytes)  ← last data fragment
/// Fragment 10: XOR parity       (214 bytes)  ← FEC
/// ```
pub fn fragment(thought_id: u16, payload: &[u8]) -> Vec<LoRaFragment> {
    assert!(!payload.is_empty(), "cannot fragment empty payload");
    let data_count = (payload.len() + PAYLOAD_SIZE - 1) / PAYLOAD_SIZE;
    assert!(
        data_count <= 254,
        "payload too large: {data_count} fragments exceeds u8 index space"
    );

    let total_fragments = (data_count + 1) as u8; // +1 for FEC
    let mut fragments = Vec::with_capacity(total_fragments as usize);
    let mut fec_accum = [0u8; PAYLOAD_SIZE];

    for i in 0..data_count {
        let start = i * PAYLOAD_SIZE;
        let end = (start + PAYLOAD_SIZE).min(payload.len());
        let len = end - start;

        let mut frag_payload = [0u8; PAYLOAD_SIZE];
        frag_payload[..len].copy_from_slice(&payload[start..end]);
        // Zero-padded remainder participates in XOR (zeros are identity)

        // Accumulate XOR parity
        for (fec, &data) in fec_accum.iter_mut().zip(frag_payload.iter()) {
            *fec ^= data;
        }

        let mut frag = LoRaFragment {
            thought_id,
            fragment_index: i as u8,
            total_fragments,
            payload_len: len as u8,
            flags: 0,
            checksum: 0,
            payload: frag_payload,
        };
        frag.checksum = frag.compute_checksum();
        fragments.push(frag);
    }

    // FEC parity fragment — XOR of all data fragments (zero-padded)
    let mut fec_frag = LoRaFragment {
        thought_id,
        fragment_index: data_count as u8,
        total_fragments,
        payload_len: PAYLOAD_SIZE as u8, // FEC is always full-size
        flags: FLAG_FEC,
        checksum: 0,
        payload: fec_accum,
    };
    fec_frag.checksum = fec_frag.compute_checksum();
    fragments.push(fec_frag);

    fragments
}

// ============================================================================
// REASSEMBLY
// ============================================================================

/// Reassembles LoRa fragments into the original payload.
///
/// Handles out-of-order arrival, duplicate detection, and single-fragment
/// loss recovery via XOR FEC parity.
///
/// # Usage
///
/// ```rust,ignore
/// let mut assembler = FragmentAssembler::new(thought_id, total_fragments, 2048);
/// for fragment in received_fragments {
///     if assembler.feed(&fragment) {
///         let payload = assembler.assemble().unwrap();
///         // payload is the original BinaryHV bytes
///     }
/// }
/// ```
pub struct FragmentAssembler {
    thought_id: u16,
    total_fragments: u8,
    data_count: u8,
    expected_payload_size: usize,
    received: Vec<bool>,
    fragments: Vec<Option<[u8; PAYLOAD_SIZE]>>,
    payload_lens: Vec<u8>,
    fec: Option<[u8; PAYLOAD_SIZE]>,
}

impl FragmentAssembler {
    /// Create a new assembler.
    ///
    /// - `thought_id`: must match incoming fragments
    /// - `total_fragments`: data_count + 1 (from any received fragment header)
    /// - `expected_payload_size`: total original payload size (e.g., 2048 for BinaryHV)
    pub fn new(thought_id: u16, total_fragments: u8, expected_payload_size: usize) -> Self {
        let data_count = total_fragments.saturating_sub(1);
        Self {
            thought_id,
            total_fragments,
            data_count,
            expected_payload_size,
            received: vec![false; total_fragments as usize],
            fragments: vec![None; data_count as usize],
            payload_lens: vec![0; data_count as usize],
            fec: None,
        }
    }

    /// Feed a received fragment. Returns `true` when the payload is reconstructable.
    pub fn feed(&mut self, frag: &LoRaFragment) -> bool {
        if frag.thought_id != self.thought_id {
            return false;
        }
        if frag.fragment_index >= self.total_fragments {
            return false;
        }

        let idx = frag.fragment_index as usize;

        // Duplicate — ignore but report current status
        if self.received[idx] {
            return self.is_complete();
        }

        self.received[idx] = true;

        if frag.is_fec() {
            self.fec = Some(frag.payload);
        } else {
            self.fragments[idx] = Some(frag.payload);
            self.payload_lens[idx] = frag.payload_len;
        }

        self.is_complete()
    }

    /// Whether the payload can be fully reconstructed.
    pub fn is_complete(&self) -> bool {
        let data_received = self.received[..self.data_count as usize]
            .iter()
            .filter(|&&r| r)
            .count();
        let data_count = self.data_count as usize;
        let has_fec = self.fec.is_some();

        // All data fragments present
        if data_received == data_count {
            return true;
        }
        // Exactly one missing + FEC available → recoverable
        if data_received == data_count - 1 && has_fec {
            return true;
        }
        false
    }

    /// Number of fragments not yet received.
    pub fn missing_count(&self) -> usize {
        self.received.iter().filter(|&&r| !r).count()
    }

    /// Number of data fragments received (excluding FEC).
    pub fn received_data_count(&self) -> usize {
        self.received[..self.data_count as usize]
            .iter()
            .filter(|&&r| r)
            .count()
    }

    /// Whether a specific fragment index has already been received.
    pub fn has_fragment(&self, index: u8) -> bool {
        (index as usize) < self.received.len() && self.received[index as usize]
    }

    /// Whether FEC recovery was needed (a data fragment was missing).
    ///
    /// Only meaningful when [`is_complete()`] returns `true`.
    pub fn used_fec_recovery(&self) -> bool {
        self.is_complete()
            && self.received[..self.data_count as usize]
                .iter()
                .any(|&r| !r)
    }

    /// Reconstruct the original payload. Returns `None` if insufficient fragments.
    pub fn assemble(&self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }

        let data_count = self.data_count as usize;

        // Find the missing data fragment index (if any)
        let missing_idx = self.received[..data_count].iter().position(|&r| !r);

        // Recover the missing fragment via XOR
        let recovered: Option<[u8; PAYLOAD_SIZE]> = missing_idx.map(|missing| {
            let fec = self
                .fec
                .as_ref()
                .expect("FEC required when data fragment missing");
            let mut recovered = *fec;
            for i in 0..data_count {
                if i != missing {
                    let frag = self.fragments[i].as_ref().expect("data fragment present");
                    for (r, &f) in recovered.iter_mut().zip(frag.iter()) {
                        *r ^= f;
                    }
                }
            }
            recovered
        });

        // Assemble the full payload
        let mut payload = Vec::with_capacity(self.expected_payload_size);
        for i in 0..data_count {
            let frag_data = if Some(i) == missing_idx {
                recovered
                    .as_ref()
                    .expect("FEC recovery must produce data for missing fragment")
            } else {
                self.fragments[i]
                    .as_ref()
                    .expect("non-missing fragment must be present")
            };

            let len = if Some(i) == missing_idx {
                // Recovered fragment: compute its length from the expected total
                if i < data_count - 1 {
                    PAYLOAD_SIZE
                } else {
                    // Last data fragment — infer from expected payload size
                    self.expected_payload_size - (data_count - 1) * PAYLOAD_SIZE
                }
            } else {
                self.payload_lens[i] as usize
            };

            payload.extend_from_slice(&frag_data[..len]);
        }

        Some(payload)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::BinaryHV;

    /// Create a deterministic test vector.
    fn test_hv(seed: u8) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = seed.wrapping_mul(i as u8).wrapping_add((i >> 3) as u8);
        }
        BinaryHV(bytes)
    }

    // -- CRC-16 --

    #[test]
    fn crc16_known_vector() {
        // Standard CRC-16/CCITT test: "123456789" → 0x29B1
        let result = crc16_ccitt(b"123456789");
        assert_eq!(result, 0x29B1, "CRC-16/CCITT mismatch");
    }

    #[test]
    fn crc16_empty() {
        assert_eq!(crc16_ccitt(b""), 0xFFFF);
    }

    // -- Fragment counts --

    #[test]
    fn fragment_binary_hv_count() {
        let hv = test_hv(42);
        let frags = fragment(1, &hv.0);

        assert_eq!(frags.len(), 11, "BinaryHV should produce 10 data + 1 FEC");
        assert!(!frags[0].is_fec());
        assert!(!frags[9].is_fec());
        assert!(frags[10].is_fec(), "last fragment must be FEC parity");
    }

    #[test]
    fn fragment_payload_sizes() {
        let hv = test_hv(7);
        let frags = fragment(1, &hv.0);

        // First 9 fragments: full payload
        for frag in &frags[..9] {
            assert_eq!(frag.payload_len as usize, PAYLOAD_SIZE);
        }

        // Fragment 9: 2048 - 9*214 = 2048 - 1926 = 122 bytes
        assert_eq!(frags[9].payload_len, 122);

        // FEC: always full-size
        assert_eq!(frags[10].payload_len as usize, PAYLOAD_SIZE);
    }

    // -- Round-trip (no loss) --

    #[test]
    fn roundtrip_binary_hv() {
        let hv = test_hv(0xAB);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for frag in &frags {
            assembler.feed(frag);
        }

        let payload = assembler.assemble().unwrap();
        assert_eq!(payload.len(), 2048);
        assert_eq!(&payload[..], &hv.0[..]);
    }

    #[test]
    fn roundtrip_out_of_order() {
        let hv = test_hv(0xCD);
        let frags = fragment(1, &hv.0);

        // Feed in reverse order
        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for frag in frags.iter().rev() {
            assembler.feed(frag);
        }

        let payload = assembler.assemble().unwrap();
        assert_eq!(&payload[..], &hv.0[..]);
    }

    // -- Single-loss FEC recovery --

    #[test]
    fn recover_first_fragment() {
        let hv = test_hv(0x11);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 0 {
                assembler.feed(frag);
            }
        }

        assert!(assembler.is_complete());
        let payload = assembler.assemble().unwrap();
        assert_eq!(&payload[..], &hv.0[..]);
    }

    #[test]
    fn recover_middle_fragment() {
        let hv = test_hv(0x22);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 5 {
                assembler.feed(frag);
            }
        }

        assert!(assembler.is_complete());
        let payload = assembler.assemble().unwrap();
        assert_eq!(&payload[..], &hv.0[..]);
    }

    #[test]
    fn recover_last_data_fragment() {
        let hv = test_hv(0x33);
        let frags = fragment(1, &hv.0);

        // Drop fragment 9 (last data, 122 bytes — the tricky one)
        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 9 {
                assembler.feed(frag);
            }
        }

        assert!(assembler.is_complete());
        let payload = assembler.assemble().unwrap();
        assert_eq!(payload.len(), 2048);
        assert_eq!(&payload[..], &hv.0[..]);
    }

    #[test]
    fn lost_fec_no_data_loss_ok() {
        let hv = test_hv(0x44);
        let frags = fragment(1, &hv.0);

        // Drop FEC (fragment 10) but keep all data — should still work
        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 10 {
                assembler.feed(frag);
            }
        }

        assert!(assembler.is_complete());
        let payload = assembler.assemble().unwrap();
        assert_eq!(&payload[..], &hv.0[..]);
    }

    // -- Double loss → failure --

    #[test]
    fn double_loss_unrecoverable() {
        let hv = test_hv(0x55);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 3 && i != 7 {
                assembler.feed(frag);
            }
        }

        assert!(!assembler.is_complete());
        assert!(assembler.assemble().is_none());
    }

    #[test]
    fn data_plus_fec_loss_unrecoverable() {
        let hv = test_hv(0x66);
        let frags = fragment(1, &hv.0);

        // Lose one data fragment AND the FEC
        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 4 && i != 10 {
                assembler.feed(frag);
            }
        }

        assert!(!assembler.is_complete());
    }

    // -- Duplicates --

    #[test]
    fn duplicate_fragments_ignored() {
        let hv = test_hv(0x77);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        // Feed each fragment twice
        for frag in &frags {
            assembler.feed(frag);
            assembler.feed(frag);
        }

        let payload = assembler.assemble().unwrap();
        assert_eq!(&payload[..], &hv.0[..]);
    }

    // -- Wire format (serialize/deserialize) --

    #[test]
    fn wire_roundtrip() {
        let hv = test_hv(0x88);
        let frags = fragment(42, &hv.0);

        let mut buf = [0u8; LORA_MTU];
        for frag in &frags {
            let len = frag.to_bytes(&mut buf);
            assert!(len <= LORA_MTU);

            let decoded = LoRaFragment::from_bytes(&buf[..len]).expect("decode should succeed");
            assert_eq!(decoded.thought_id, 42);
            assert_eq!(decoded.fragment_index, frag.fragment_index);
            assert_eq!(decoded.payload_len, frag.payload_len);
            assert_eq!(
                &decoded.payload[..decoded.payload_len as usize],
                &frag.payload[..frag.payload_len as usize]
            );
        }
    }

    #[test]
    fn corrupt_fragment_rejected() {
        let hv = test_hv(0x99);
        let frags = fragment(1, &hv.0);

        let mut buf = [0u8; LORA_MTU];
        let len = frags[0].to_bytes(&mut buf);

        // Flip a bit in the payload
        buf[12] ^= 0x01;

        assert!(
            LoRaFragment::from_bytes(&buf[..len]).is_none(),
            "corrupted fragment must be rejected"
        );
    }

    #[test]
    fn truncated_buffer_rejected() {
        assert!(LoRaFragment::from_bytes(&[0; 4]).is_none());
        assert!(LoRaFragment::from_bytes(&[]).is_none());
    }

    // -- Wrong thought_id rejected --

    #[test]
    fn wrong_thought_id_rejected() {
        let hv = test_hv(0xAA);
        let frags = fragment(100, &hv.0);

        let mut assembler = FragmentAssembler::new(999, 11, 2048);
        for frag in &frags {
            assert!(!assembler.feed(frag));
        }
        assert!(!assembler.is_complete());
    }

    // -- Edge cases --

    #[test]
    fn fragment_single_byte() {
        let frags = fragment(1, &[0x42]);
        assert_eq!(frags.len(), 2); // 1 data + 1 FEC
        assert_eq!(frags[0].payload_len, 1);
        assert!(frags[1].is_fec());

        let mut assembler = FragmentAssembler::new(1, 2, 1);
        for frag in &frags {
            assembler.feed(frag);
        }
        assert_eq!(assembler.assemble().unwrap(), vec![0x42]);
    }

    #[test]
    fn fragment_exact_payload_size() {
        let data = vec![0xBB; PAYLOAD_SIZE]; // exactly 214 bytes
        let frags = fragment(1, &data);
        assert_eq!(frags.len(), 2); // 1 data + 1 FEC
        assert_eq!(frags[0].payload_len as usize, PAYLOAD_SIZE);

        let mut assembler = FragmentAssembler::new(1, 2, PAYLOAD_SIZE);
        for frag in &frags {
            assembler.feed(frag);
        }
        assert_eq!(assembler.assemble().unwrap(), data);
    }

    #[test]
    fn fragment_exact_double_payload() {
        let data = vec![0xCC; PAYLOAD_SIZE * 2]; // exactly 428 bytes
        let frags = fragment(1, &data);
        assert_eq!(frags.len(), 3); // 2 data + 1 FEC
        assert_eq!(frags[0].payload_len as usize, PAYLOAD_SIZE);
        assert_eq!(frags[1].payload_len as usize, PAYLOAD_SIZE);

        let mut assembler = FragmentAssembler::new(1, 3, PAYLOAD_SIZE * 2);
        for frag in &frags {
            assembler.feed(frag);
        }
        assert_eq!(assembler.assemble().unwrap(), data);
    }

    // -- Full integration: fragment → wire → reassemble with FEC recovery --

    #[test]
    fn full_integration_with_loss_and_wire() {
        let hv = test_hv(0xEF);
        let frags = fragment(7, &hv.0);

        // Simulate radio: serialize each fragment, drop fragment 3, deserialize rest
        let mut assembler = FragmentAssembler::new(7, 11, 2048);
        let mut buf = [0u8; LORA_MTU];

        for (i, frag) in frags.iter().enumerate() {
            if i == 3 {
                continue; // dropped by the Karoo wind
            }
            let len = frag.to_bytes(&mut buf);
            let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
            assembler.feed(&decoded);
        }

        assert!(assembler.is_complete());
        let payload = assembler.assemble().unwrap();
        assert_eq!(
            &payload[..],
            &hv.0[..],
            "recovered payload must match original"
        );
    }

    // -- FEC recovery detection --

    #[test]
    fn fec_recovery_not_needed_when_all_data_present() {
        let hv = test_hv(0xBB);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for frag in &frags {
            assembler.feed(frag);
        }
        assert!(assembler.is_complete());
        assert!(!assembler.used_fec_recovery());
    }

    #[test]
    fn fec_recovery_detected_when_data_missing() {
        let hv = test_hv(0xBC);
        let frags = fragment(1, &hv.0);

        // Drop fragment 5 → FEC recovery needed
        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        for (i, frag) in frags.iter().enumerate() {
            if i != 5 {
                assembler.feed(frag);
            }
        }
        assert!(assembler.is_complete());
        assert!(assembler.used_fec_recovery());
    }

    #[test]
    fn has_fragment_tracking() {
        let hv = test_hv(0xBD);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        assert!(!assembler.has_fragment(0));
        assert!(!assembler.has_fragment(5));

        assembler.feed(&frags[0]);
        assert!(assembler.has_fragment(0));
        assert!(!assembler.has_fragment(5));

        assembler.feed(&frags[5]);
        assert!(assembler.has_fragment(0));
        assert!(assembler.has_fragment(5));

        // Out of range
        assert!(!assembler.has_fragment(255));
    }

    // -- Metrics --

    #[test]
    fn assembler_metrics() {
        let hv = test_hv(0xDD);
        let frags = fragment(1, &hv.0);

        let mut assembler = FragmentAssembler::new(1, 11, 2048);
        assert_eq!(assembler.missing_count(), 11);
        assert_eq!(assembler.received_data_count(), 0);

        assembler.feed(&frags[0]);
        assert_eq!(assembler.missing_count(), 10);
        assert_eq!(assembler.received_data_count(), 1);

        for frag in &frags[1..] {
            assembler.feed(frag);
        }
        assert_eq!(assembler.missing_count(), 0);
        assert_eq!(assembler.received_data_count(), 10);
    }
}
