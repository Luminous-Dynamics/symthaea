// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Experimental V3 LoRa framing with explicit session-generation + message-sequence identity.
//!
//! V3 is a separate wire version; it does not change V1/V2 interpretation or production routing.
//! The 22-byte header binds a frame to (session_generation, message_sequence), carries exact
//! encoded length, and keeps a discriminator at byte 5 so a hardened V1 parser rejects it.
//!
//! Wire layout:
//! - 0..2: fixed V3 magic/version bytes
//! - 2: fragment_index
//! - 3: total_fragments including XOR parity
//! - 4: payload_len
//! - 5: flags (bit 7 V3 discriminator; bit 0 XOR parity; bit 2 AEAD-size profile)
//! - 6..8: CRC-16/CCITT
//! - 8..16: session_generation (u64 LE)
//! - 16..20: message_sequence (u32 LE)
//! - 20..22: exact encoded_len (u16 LE)
//! - 22..: payload
//!
//! This codec only frames bytes. The AEAD-size profile is accounting only and does not encrypt.
//! CRC is not authentication. The caller MUST allocate non-repeating session generations through
//! the reviewed clone-safe key/session lifecycle, must not reuse message_sequence in one generation,
//! and must authenticate the complete identity/header as associated data (or an equivalent reviewed
//! authenticated context) before trusting payload bytes. Reassembly is also scoped to a 32-byte
//! authenticated peer context supplied by the caller; never pass an unauthenticated source-ID field.
//! A generation field alone cannot make a rolled-back sender safe. This module is experimental and
//! is not wired into production paths.

use super::lora_fragment::crc16_ccitt;

pub const V3_MAGIC: [u8; 2] = [0xD3, 0x03];
pub const V3_HEADER_SIZE: usize = 22;
pub const V3_MTU: usize = 222;
pub const V3_AEAD_OVERHEAD: usize = 12 + 16;
pub const V3_PLAIN_PAYLOAD_SIZE: usize = V3_MTU - V3_HEADER_SIZE;
pub const V3_AEAD_PAYLOAD_SIZE: usize = V3_MTU - V3_HEADER_SIZE - V3_AEAD_OVERHEAD;
pub const V3_FLAG_FEC: u8 = 0x01;
pub const V3_FLAG_AEAD_BUDGET: u8 = 0x04;
pub const V3_FLAG_VERSION_MARKER: u8 = 0x80;
const V3_ALLOWED_FLAGS: u8 = V3_FLAG_FEC | V3_FLAG_AEAD_BUDGET | V3_FLAG_VERSION_MARKER;
pub const V3_MAX_DATA_FRAGMENTS: usize = 254;
pub const V3_MAX_PLAIN_ENCODED_LEN: usize = V3_MAX_DATA_FRAGMENTS * V3_PLAIN_PAYLOAD_SIZE;
pub const V3_MAX_AEAD_ENCODED_LEN: usize = V3_MAX_DATA_FRAGMENTS * V3_AEAD_PAYLOAD_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V3FragmentError {
    EmptyPayload,
    PayloadTooLarge,
    InvalidSessionGeneration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoRaFragmentV3 {
    session_generation: u64,
    message_sequence: u32,
    fragment_index: u8,
    total_fragments: u8,
    payload_len: u8,
    flags: u8,
    checksum: u16,
    encoded_len: u16,
    payload: [u8; V3_PLAIN_PAYLOAD_SIZE],
}

impl LoRaFragmentV3 {
    pub fn session_generation(&self) -> u64 {
        self.session_generation
    }

    pub fn message_sequence(&self) -> u32 {
        self.message_sequence
    }

    pub fn fragment_index(&self) -> u8 {
        self.fragment_index
    }

    pub fn total_fragments(&self) -> u8 {
        self.total_fragments
    }

    pub fn encoded_len(&self) -> usize {
        usize::from(self.encoded_len)
    }

    pub fn payload_len(&self) -> usize {
        usize::from(self.payload_len)
    }

    pub fn is_fec(&self) -> bool {
        self.flags & V3_FLAG_FEC != 0
    }

    pub fn uses_aead_budget(&self) -> bool {
        self.flags & V3_FLAG_AEAD_BUDGET != 0
    }

    fn payload_capacity(flags: u8) -> usize {
        if flags & V3_FLAG_AEAD_BUDGET != 0 {
            V3_AEAD_PAYLOAD_SIZE
        } else {
            V3_PLAIN_PAYLOAD_SIZE
        }
    }

    fn data_count_for_len(encoded_len: usize, capacity: usize) -> Option<usize> {
        if capacity == 0 || encoded_len == 0 {
            return None;
        }
        let max_len = V3_MAX_DATA_FRAGMENTS.checked_mul(capacity)?;
        if encoded_len > max_len || encoded_len > usize::from(u16::MAX) {
            return None;
        }
        Some(encoded_len.checked_add(capacity - 1)? / capacity)
    }

    fn expected_payload_len(
        encoded_len: usize,
        data_count: usize,
        index: usize,
        is_fec: bool,
        capacity: usize,
    ) -> Option<usize> {
        if is_fec {
            return (index == data_count).then_some(capacity);
        }
        if index >= data_count {
            return None;
        }
        if index + 1 < data_count {
            Some(capacity)
        } else {
            encoded_len.checked_sub((data_count - 1).checked_mul(capacity)?)
        }
    }

    fn checksum_for(
        session_generation: u64,
        message_sequence: u32,
        fragment_index: u8,
        total_fragments: u8,
        payload_len: u8,
        flags: u8,
        encoded_len: u16,
        payload: &[u8],
    ) -> u16 {
        // CRC input: 20 bytes of non-checksum header material plus the actual payload.
        debug_assert!(payload.len() <= V3_PLAIN_PAYLOAD_SIZE);
        let mut input = [0u8; 20 + V3_PLAIN_PAYLOAD_SIZE];
        input[0..2].copy_from_slice(&V3_MAGIC);
        input[2..6].copy_from_slice(&[
            fragment_index,
            total_fragments,
            payload_len,
            flags,
        ]);
        input[6..14].copy_from_slice(&session_generation.to_le_bytes());
        input[14..18].copy_from_slice(&message_sequence.to_le_bytes());
        input[18..20].copy_from_slice(&encoded_len.to_le_bytes());
        let end = 20 + payload.len();
        input[20..end].copy_from_slice(payload);
        crc16_ccitt(&input[..end])
    }

    fn is_well_formed(&self) -> bool {
        if self.session_generation == 0
            || self.flags & V3_FLAG_VERSION_MARKER == 0
            || self.flags & !V3_ALLOWED_FLAGS != 0
        {
            return false;
        }
        let capacity = Self::payload_capacity(self.flags);
        let encoded_len = self.encoded_len();
        let Some(data_count) = Self::data_count_for_len(encoded_len, capacity) else {
            return false;
        };
        if usize::from(self.total_fragments) != data_count + 1 {
            return false;
        }
        let index = usize::from(self.fragment_index);
        if index > data_count || self.is_fec() != (index == data_count) {
            return false;
        }
        let Some(expected_len) = Self::expected_payload_len(
            encoded_len,
            data_count,
            index,
            self.is_fec(),
            capacity,
        ) else {
            return false;
        };
        if usize::from(self.payload_len) != expected_len || expected_len > capacity {
            return false;
        }
        self.checksum
            == Self::checksum_for(
                self.session_generation,
                self.message_sequence,
                self.fragment_index,
                self.total_fragments,
                self.payload_len,
                self.flags,
                self.encoded_len,
                &self.payload[..expected_len],
            )
    }

    /// Serialize into caller-owned storage without allocation. Returns exact bytes written.
    /// Insufficient storage returns None before any destination bytes are modified.
    pub fn write_to(&self, out: &mut [u8]) -> Option<usize> {
        if !self.is_well_formed() {
            return None;
        }
        let payload_len = usize::from(self.payload_len);
        let total = V3_HEADER_SIZE.checked_add(payload_len)?;
        if out.len() < total {
            return None;
        }

        out[0..2].copy_from_slice(&V3_MAGIC);
        out[2] = self.fragment_index;
        out[3] = self.total_fragments;
        out[4] = self.payload_len;
        out[5] = self.flags;
        out[6..8].copy_from_slice(&self.checksum.to_le_bytes());
        out[8..16].copy_from_slice(&self.session_generation.to_le_bytes());
        out[16..20].copy_from_slice(&self.message_sequence.to_le_bytes());
        out[20..22].copy_from_slice(&self.encoded_len.to_le_bytes());
        out[V3_HEADER_SIZE..total].copy_from_slice(&self.payload[..payload_len]);
        Some(total)
    }

    /// Convenience serializer. Prefer write_to when the caller owns a reusable MTU buffer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let total = V3_HEADER_SIZE + usize::from(self.payload_len);
        let mut out = vec![0u8; total];
        let written = self.write_to(&mut out).expect("V3 instance must be valid");
        debug_assert_eq!(written, total);
        out
    }

    /// Parses a complete V3 frame, rejecting wrong magic/version, lengths, identity, flags and CRC.
    pub fn from_bytes(raw: &[u8]) -> Option<Self> {
        if raw.len() < V3_HEADER_SIZE || raw[0] != V3_MAGIC[0] || raw[1] != V3_MAGIC[1] {
            return None;
        }
        let fragment_index = raw[2];
        let total_fragments = raw[3];
        let payload_len = raw[4];
        let flags = raw[5];
        let checksum = u16::from_le_bytes([raw[6], raw[7]]);
        let session_generation = u64::from_le_bytes(raw[8..16].try_into().ok()?);
        let message_sequence = u32::from_le_bytes(raw[16..20].try_into().ok()?);
        let encoded_len = u16::from_le_bytes([raw[20], raw[21]]);

        if flags & V3_FLAG_VERSION_MARKER == 0
            || flags & !V3_ALLOWED_FLAGS != 0
            || usize::from(payload_len) > V3_PLAIN_PAYLOAD_SIZE
            || raw.len() != V3_HEADER_SIZE + usize::from(payload_len)
        {
            return None;
        }

        let mut payload = [0u8; V3_PLAIN_PAYLOAD_SIZE];
        payload[..usize::from(payload_len)].copy_from_slice(&raw[V3_HEADER_SIZE..]);
        let frame = Self {
            session_generation,
            message_sequence,
            fragment_index,
            total_fragments,
            payload_len,
            flags,
            checksum,
            encoded_len,
            payload,
        };
        frame.is_well_formed().then_some(frame)
    }
}

/// Fragment an already-encoded message using V3 framing and one XOR parity frame.
pub fn fragment_v3(
    session_generation: u64,
    message_sequence: u32,
    encoded: &[u8],
) -> Result<Vec<LoRaFragmentV3>, V3FragmentError> {
    fragment_v3_with_capacity(
        session_generation,
        message_sequence,
        encoded,
        V3_PLAIN_PAYLOAD_SIZE,
        false,
    )
}

/// Select the reduced payload capacity for an external 12-byte nonce + 16-byte tag wrapper.
/// This is size accounting only: it performs no encryption or authentication.
pub fn fragment_v3_with_aead_overhead(
    session_generation: u64,
    message_sequence: u32,
    encoded: &[u8],
) -> Result<Vec<LoRaFragmentV3>, V3FragmentError> {
    fragment_v3_with_capacity(
        session_generation,
        message_sequence,
        encoded,
        V3_AEAD_PAYLOAD_SIZE,
        true,
    )
}

fn fragment_v3_with_capacity(
    session_generation: u64,
    message_sequence: u32,
    encoded: &[u8],
    capacity: usize,
    aead_budget: bool,
) -> Result<Vec<LoRaFragmentV3>, V3FragmentError> {
    if session_generation == 0 {
        return Err(V3FragmentError::InvalidSessionGeneration);
    }
    if encoded.is_empty() {
        return Err(V3FragmentError::EmptyPayload);
    }
    let max_len = V3_MAX_DATA_FRAGMENTS * capacity;
    if encoded.len() > max_len || encoded.len() > usize::from(u16::MAX) {
        return Err(V3FragmentError::PayloadTooLarge);
    }
    let data_count = (encoded.len() + capacity - 1) / capacity;
    let total_fragments =
        u8::try_from(data_count + 1).map_err(|_| V3FragmentError::PayloadTooLarge)?;
    let encoded_len = u16::try_from(encoded.len()).map_err(|_| V3FragmentError::PayloadTooLarge)?;
    let profile_flag = if aead_budget { V3_FLAG_AEAD_BUDGET } else { 0 };
    let mut frames = Vec::with_capacity(data_count + 1);
    let mut parity = [0u8; V3_PLAIN_PAYLOAD_SIZE];

    for index in 0..data_count {
        let start = index * capacity;
        let end = (start + capacity).min(encoded.len());
        let chunk = &encoded[start..end];
        let mut payload = [0u8; V3_PLAIN_PAYLOAD_SIZE];
        payload[..chunk.len()].copy_from_slice(chunk);
        for byte_index in 0..capacity {
            parity[byte_index] ^= payload[byte_index];
        }
        let payload_len =
            u8::try_from(chunk.len()).map_err(|_| V3FragmentError::PayloadTooLarge)?;
        let flags = V3_FLAG_VERSION_MARKER | profile_flag;
        let checksum = LoRaFragmentV3::checksum_for(
            session_generation,
            message_sequence,
            index as u8,
            total_fragments,
            payload_len,
            flags,
            encoded_len,
            &payload[..chunk.len()],
        );
        frames.push(LoRaFragmentV3 {
            session_generation,
            message_sequence,
            fragment_index: index as u8,
            total_fragments,
            payload_len,
            flags,
            checksum,
            encoded_len,
            payload,
        });
    }

    let index = data_count as u8;
    let payload_len = capacity as u8;
    let flags = V3_FLAG_VERSION_MARKER | V3_FLAG_FEC | profile_flag;
    let checksum = LoRaFragmentV3::checksum_for(
        session_generation,
        message_sequence,
        index,
        total_fragments,
        payload_len,
        flags,
        encoded_len,
        &parity[..capacity],
    );
    frames.push(LoRaFragmentV3 {
        session_generation,
        message_sequence,
        fragment_index: index,
        total_fragments,
        payload_len,
        flags,
        checksum,
        encoded_len,
        payload: parity,
    });
    Ok(frames)
}

/// Reassembler scoped to an authenticated peer context and exact stream identity.
pub struct FragmentAssemblerV3 {
    authenticated_peer_context: [u8; 32],
    session_generation: u64,
    message_sequence: u32,
    total_fragments: u8,
    encoded_len: usize,
    data_count: usize,
    capacity: usize,
    received: Vec<bool>,
    fragments: Vec<Option<[u8; V3_PLAIN_PAYLOAD_SIZE]>>,
    payload_lens: Vec<u8>,
    fec: Option<[u8; V3_PLAIN_PAYLOAD_SIZE]>,
    poisoned: bool,
}

impl FragmentAssemblerV3 {
    /// Configure from a validated first frame and the peer context produced by successful
    /// authentication. The first frame is not yet received; feed it with that same context.
    pub fn new(
        authenticated_peer_context: [u8; 32],
        first: &LoRaFragmentV3,
    ) -> Option<Self> {
        if !first.is_well_formed() {
            return None;
        }
        let capacity = LoRaFragmentV3::payload_capacity(first.flags);
        let encoded_len = first.encoded_len();
        let data_count = LoRaFragmentV3::data_count_for_len(encoded_len, capacity)?;
        Some(Self {
            authenticated_peer_context,
            session_generation: first.session_generation,
            message_sequence: first.message_sequence,
            total_fragments: first.total_fragments,
            encoded_len,
            data_count,
            capacity,
            received: vec![false; data_count + 1],
            fragments: vec![None; data_count],
            payload_lens: vec![0; data_count],
            fec: None,
            poisoned: false,
        })
    }

    /// Accept only after the caller authenticates the frame and derives the peer context.
    /// All stream identity and profile fields must agree. Conflicting duplicates poison the stream.
    pub fn feed(
        &mut self,
        authenticated_peer_context: &[u8; 32],
        frame: &LoRaFragmentV3,
    ) -> bool {
        if self.poisoned
            || authenticated_peer_context != &self.authenticated_peer_context
            || !frame.is_well_formed()
            || frame.session_generation != self.session_generation
            || frame.message_sequence != self.message_sequence
            || frame.total_fragments != self.total_fragments
            || frame.encoded_len() != self.encoded_len
            || LoRaFragmentV3::payload_capacity(frame.flags) != self.capacity
        {
            return false;
        }
        let index = usize::from(frame.fragment_index);
        if index >= self.received.len() {
            return false;
        }
        if self.received[index] {
            let identical = if frame.is_fec() {
                self.fec.as_ref() == Some(&frame.payload)
            } else {
                self.fragments[index].as_ref() == Some(&frame.payload)
                    && self.payload_lens[index] == frame.payload_len
            };
            if !identical {
                self.poisoned = true;
                return false;
            }
            return self.is_complete();
        }

        self.received[index] = true;
        if frame.is_fec() {
            self.fec = Some(frame.payload);
        } else {
            self.fragments[index] = Some(frame.payload);
            self.payload_lens[index] = frame.payload_len;
        }
        self.is_complete()
    }

    pub fn is_complete(&self) -> bool {
        if self.poisoned {
            return false;
        }
        let present = self.received[..self.data_count]
            .iter()
            .filter(|&&received| received)
            .count();
        present == self.data_count || (present + 1 == self.data_count && self.fec.is_some())
    }

    pub fn assemble(&self) -> Option<Vec<u8>> {
        if !self.is_complete() || self.data_count == 0 {
            return None;
        }
        let missing = self.received[..self.data_count]
            .iter()
            .position(|&received| !received);
        let recovered = if let Some(missing_index) = missing {
            let mut recovered = self.fec?;
            for index in 0..self.data_count {
                if index == missing_index {
                    continue;
                }
                let data = self.fragments[index].as_ref()?;
                for (dst, src) in recovered.iter_mut().zip(data.iter()) {
                    *dst ^= *src;
                }
            }
            Some(recovered)
        } else {
            None
        };

        let mut output = Vec::with_capacity(self.encoded_len);
        for index in 0..self.data_count {
            if Some(index) == missing {
                let chunk = recovered.as_ref()?;
                let len = if index + 1 == self.data_count {
                    self.encoded_len.checked_sub((self.data_count - 1).checked_mul(self.capacity)?)?
                } else {
                    self.capacity
                };
                if len == 0 || len > self.capacity {
                    return None;
                }
                output.extend_from_slice(&chunk[..len]);
            } else {
                let chunk = self.fragments[index].as_ref()?;
                let len = usize::from(self.payload_lens[index]);
                if len == 0 || len > self.capacity {
                    return None;
                }
                output.extend_from_slice(&chunk[..len]);
            }
        }
        (output.len() == self.encoded_len).then_some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::lora_fragment::LoRaFragment;

    fn sample(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.max(1);
        let mut bytes = vec![0u8; len];
        for byte in &mut bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 24) as u8;
        }
        bytes
    }

    #[test]
    fn v3_checksum_has_pinned_wire_layout_vector() {
        let frames = fragment_v3(
            0x0102_0304_0506_0708,
            0x0a0b_0c0d,
            &[0xaa, 0xbb],
        )
        .unwrap();
        assert_eq!(frames[0].checksum, 0xd368);
        let wire = frames[0].to_bytes();
        assert_eq!(&wire[0..2], &V3_MAGIC);
        assert_eq!(&wire[6..8], &[0x68, 0xd3]);
        assert_eq!(&wire[8..16], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(&wire[16..20], &0x0a0b_0c0du32.to_le_bytes());
    }

    #[test]
    fn v3_exact_length_and_single_erasure_recovery_for_boundary_profiles() {
        for len in [1usize, V3_AEAD_PAYLOAD_SIZE, V3_AEAD_PAYLOAD_SIZE + 1, 1800, 2048, 2105, V3_MAX_AEAD_ENCODED_LEN] {
            let original = sample(len, len as u64 + 0x9e37);
            let frames = fragment_v3_with_aead_overhead(0xAABB_CCDD_EEFF_0011, 7, &original).unwrap();
            assert!(frames.iter().all(|frame| frame.to_bytes().len() + V3_AEAD_OVERHEAD <= V3_MTU));
            for lost in 0..frames.len() - 1 {
                let first = frames.iter().enumerate().find_map(|(i, f)| (i != lost).then_some(f)).unwrap();
                let peer_context = [0x11; 32];
                let mut assembler = FragmentAssemblerV3::new(peer_context, first).unwrap();
                for (index, frame) in frames.iter().enumerate() {
                    if index != lost {
                        let wire = frame.to_bytes();
                        assembler.feed(&peer_context, &LoRaFragmentV3::from_bytes(&wire).unwrap());
                    }
                }
                assert!(assembler.is_complete(), "len={len}, lost={lost}");
                assert_eq!(assembler.assemble().as_deref(), Some(original.as_slice()), "len={len}, lost={lost}");
            }
        }
    }

    #[test]
    fn same_shape_fragments_from_different_generations_are_rejected() {
        let a = fragment_v3(10, 4, &sample(2048, 1)).unwrap();
        let b = fragment_v3(11, 4, &sample(2048, 2)).unwrap();
        let peer_context = [0x11; 32];
        let mut assembler = FragmentAssemblerV3::new(peer_context, &a[0]).unwrap();
        for frame in &a[..6] {
            assembler.feed(&peer_context, frame);
        }
        for frame in &b[6..b.len() - 1] {
            assert!(!assembler.feed(&peer_context, frame));
        }
        assert!(!assembler.is_complete());
        assert!(assembler.assemble().is_none());
    }

    #[test]
    fn same_generation_fragments_from_different_message_sequences_are_rejected() {
        let a = fragment_v3(10, 4, &sample(2048, 1)).unwrap();
        let b = fragment_v3(10, 5, &sample(2048, 2)).unwrap();
        let mut assembler = FragmentAssemblerV3::new(&a[0]).unwrap();
        for frame in &a[..6] {
            assembler.feed(frame);
        }
        for frame in &b[6..b.len() - 1] {
            assert!(!assembler.feed(frame));
        }
        assert!(!assembler.is_complete());
    }

    #[test]
    fn same_numeric_stream_identity_from_different_authenticated_peers_is_rejected() {
        let a = fragment_v3(10, 4, &sample(2048, 1)).unwrap();
        let b = fragment_v3(10, 4, &sample(2048, 2)).unwrap();
        let peer_a = [0x11; 32];
        let peer_b = [0x22; 32];
        let mut assembler = FragmentAssemblerV3::new(peer_a, &a[0]).unwrap();
        for frame in &a[..6] {
            assembler.feed(&peer_a, frame);
        }
        assert!(!assembler.feed(&peer_b, &b[6]));
        assert!(!assembler.is_complete());
        assert!(assembler.assemble().is_none());
    }

    #[test]
    fn conflicting_duplicate_poisons_assembly() {
        let payload = sample(2048, 1);
        let a = fragment_v3(10, 4, &payload).unwrap();
        let b = fragment_v3(10, 4, &sample(2048, 2)).unwrap();
        let peer_context = [0x11; 32];
        let mut assembler = FragmentAssemblerV3::new(peer_context, &a[0]).unwrap();
        assembler.feed(&peer_context, &a[0]);
        assert!(!assembler.feed(&peer_context, &b[0]));
        assert!(!assembler.is_complete());
        assert!(assembler.assemble().is_none());
    }

    #[test]
    fn v3_is_rejected_by_hardened_legacy_v1_parser() {
        for frame in fragment_v3(99, 42, &sample(2048, 6)).unwrap() {
            assert!(LoRaFragment::from_bytes(&frame.to_bytes()).is_none());
        }
    }

    #[test]
    fn every_single_bit_flip_in_a_v3_frame_is_rejected() {
        let frame = fragment_v3(99, 42, &sample(2048, 6)).unwrap()[0].to_bytes();
        for byte_index in 0..frame.len() {
            for bit_index in 0..8 {
                let mut corrupted = frame.clone();
                corrupted[byte_index] ^= 1u8 << bit_index;
                assert!(LoRaFragmentV3::from_bytes(&corrupted).is_none(), "byte={byte_index}, bit={bit_index}");
            }
        }
    }

    #[test]
    fn v3_rejects_empty_zero_generation_and_oversized_messages() {
        assert_eq!(fragment_v3(1, 0, &[]), Err(V3FragmentError::InvalidSessionGeneration));
        assert_eq!(fragment_v3(1, 0, &[1]), Err(V3FragmentError::InvalidSessionGeneration));
        assert_eq!(fragment_v3(1, 0, &sample(V3_MAX_PLAIN_ENCODED_LEN + 1, 7)), Err(V3FragmentError::InvalidSessionGeneration));
        assert!(fragment_v3(1, 1, &[]).is_err());
        assert!(fragment_v3(1, 1, &sample(V3_MAX_PLAIN_ENCODED_LEN + 1, 7)).is_err());
    }
}
