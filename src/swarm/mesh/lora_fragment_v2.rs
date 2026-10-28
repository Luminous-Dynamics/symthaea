// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Experimental length-bound V2 framing for lossless XOR recovery.
//!
//! This module is deliberately not wired into DualLayerMesh. It prototypes a
//! versioned 10-byte header that carries the exact encoded message length in
//! every fragment. The receiver can therefore trim an XOR-recovered final
//! fragment exactly, independent of the raw WisdomPacket size.
//!
//! Wire layout (10-byte header):
//! - 0..2: thought_id (u16 little-endian)
//! - 2: fragment_index
//! - 3: total_fragments, including XOR parity
//! - 4: payload_len
//! - 5: flags (bit 0 = XOR parity, bit 1 = V2 length binding)
//! - 6..8: CRC-16/CCITT
//! - 8..10: exact encoded_len (u16 little-endian)
//! - 10..: payload
//!
//! The 10-byte header reduces per-frame data capacity from 214 to 212 bytes.
//! CRC detects accidental corruption; it does NOT authenticate metadata. An
//! authenticated deployment must bind this header under fragment AEAD or an
//! equivalent authenticated envelope. This prototype is research code only.

use super::lora_fragment::crc16_ccitt;

pub const V2_HEADER_SIZE: usize = 10;
pub const V2_MTU: usize = 222;
pub const V2_PAYLOAD_SIZE: usize = V2_MTU - V2_HEADER_SIZE;
pub const V2_FLAG_FEC: u8 = 0x01;
pub const V2_FLAG_LENGTH_BOUND: u8 = 0x02;
/// Selects the reduced payload budget needed when each whole frame is wrapped
/// in the current 12-byte-nonce + 16-byte-tag fragment AEAD envelope. This bit
/// is itself inside the frame and must be authenticated by the outer AEAD.
pub const V2_FLAG_AEAD_BUDGET: u8 = 0x04;
const V2_ALLOWED_FLAGS: u8 = V2_FLAG_FEC | V2_FLAG_LENGTH_BOUND | V2_FLAG_AEAD_BUDGET;
pub const V2_FRAGMENT_AEAD_OVERHEAD: usize = 12 + 16;
pub const V2_AEAD_PAYLOAD_SIZE: usize =
    V2_MTU - V2_HEADER_SIZE - V2_FRAGMENT_AEAD_OVERHEAD;
pub const V2_MAX_DATA_FRAGMENTS: usize = 254;
pub const V2_MAX_ENCODED_LEN: usize = V2_MAX_DATA_FRAGMENTS * V2_PAYLOAD_SIZE;
pub const V2_MAX_AEAD_ENCODED_LEN: usize =
    V2_MAX_DATA_FRAGMENTS * V2_AEAD_PAYLOAD_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2FragmentError {
    EmptyPayload,
    PayloadTooLarge,
}

/// A validated V2 fragment. Fields are private so instances can only be
/// constructed by the checked parser or fragment_v2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoRaFragmentV2 {
    thought_id: u16,
    fragment_index: u8,
    total_fragments: u8,
    payload_len: u8,
    flags: u8,
    checksum: u16,
    encoded_len: u16,
    payload: [u8; V2_PAYLOAD_SIZE],
}

impl LoRaFragmentV2 {
    pub fn thought_id(&self) -> u16 {
        self.thought_id
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
        self.flags & V2_FLAG_FEC != 0
    }

    pub fn aead_budget_profile(&self) -> bool {
        self.flags & V2_FLAG_AEAD_BUDGET != 0
    }

    fn data_count_for_len(encoded_len: usize, payload_capacity: usize) -> Option<usize> {
        let max_encoded_len = V2_MAX_DATA_FRAGMENTS.checked_mul(payload_capacity)?;
        if encoded_len == 0 || encoded_len > max_encoded_len || encoded_len > usize::from(u16::MAX) {
            return None;
        }
        Some((encoded_len + payload_capacity - 1) / payload_capacity)
    }

    fn expected_payload_len(
        encoded_len: usize,
        data_count: usize,
        index: usize,
        is_fec: bool,
        payload_capacity: usize,
    ) -> Option<usize> {
        if is_fec {
            return (index == data_count).then_some(payload_capacity);
        }
        if index >= data_count {
            return None;
        }
        if index + 1 < data_count {
            Some(payload_capacity)
        } else {
            encoded_len.checked_sub((data_count - 1).checked_mul(payload_capacity)?)
        }
    }

    fn payload_capacity_from_flags(flags: u8) -> usize {
        if flags & V2_FLAG_AEAD_BUDGET != 0 {
            V2_AEAD_PAYLOAD_SIZE
        } else {
            V2_PAYLOAD_SIZE
        }
    }

    fn checksum_for(
        thought_id: u16,
        fragment_index: u8,
        total_fragments: u8,
        payload_len: u8,
        flags: u8,
        encoded_len: u16,
        payload: &[u8],
    ) -> u16 {
        let mut bytes = Vec::with_capacity(2 + 4 + 2 + payload.len());
        bytes.extend_from_slice(&thought_id.to_le_bytes());
        bytes.extend_from_slice(&[fragment_index, total_fragments, payload_len, flags]);
        bytes.extend_from_slice(&encoded_len.to_le_bytes());
        bytes.extend_from_slice(payload);
        crc16_ccitt(&bytes)
    }

    fn is_well_formed(&self) -> bool {
        if self.flags & V2_FLAG_LENGTH_BOUND == 0 || self.flags & !V2_ALLOWED_FLAGS != 0 {
            return false;
        }
        let encoded_len = self.encoded_len();
        let payload_capacity = Self::payload_capacity_from_flags(self.flags);
        let Some(data_count) = Self::data_count_for_len(encoded_len, payload_capacity) else {
            return false;
        };
        if usize::from(self.total_fragments) != data_count + 1 {
            return false;
        }
        let index = usize::from(self.fragment_index);
        if index >= data_count + 1 {
            return false;
        }
        let is_fec = self.is_fec();
        if is_fec != (index == data_count) {
            return false;
        }
        let Some(expected_len) = Self::expected_payload_len(
            encoded_len,
            data_count,
            index,
            is_fec,
            payload_capacity,
        ) else {
            return false;
        };
        if usize::from(self.payload_len) != expected_len || expected_len > payload_capacity {
            return false;
        }
        self.checksum
            == Self::checksum_for(
                self.thought_id,
                self.fragment_index,
                self.total_fragments,
                self.payload_len,
                self.flags,
                self.encoded_len,
                &self.payload[..expected_len],
            )
    }

    /// Serialize a validated fragment to its exact wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        debug_assert!(self.is_well_formed());
        let payload_len = usize::from(self.payload_len);
        let mut out = vec![0u8; V2_HEADER_SIZE + payload_len];
        out[0..2].copy_from_slice(&self.thought_id.to_le_bytes());
        out[2] = self.fragment_index;
        out[3] = self.total_fragments;
        out[4] = self.payload_len;
        out[5] = self.flags;
        out[6..8].copy_from_slice(&self.checksum.to_le_bytes());
        out[8..10].copy_from_slice(&self.encoded_len.to_le_bytes());
        out[V2_HEADER_SIZE..].copy_from_slice(&self.payload[..payload_len]);
        out
    }

    /// Parse and validate exact frame length, bound metadata, index/type
    /// consistency, fragment lengths, and checksum before accepting the frame.
    pub fn from_bytes(raw: &[u8]) -> Option<Self> {
        if raw.len() < V2_HEADER_SIZE {
            return None;
        }
        let thought_id = u16::from_le_bytes([raw[0], raw[1]]);
        let fragment_index = raw[2];
        let total_fragments = raw[3];
        let payload_len = raw[4];
        let flags = raw[5];
        let checksum = u16::from_le_bytes([raw[6], raw[7]]);
        let encoded_len = u16::from_le_bytes([raw[8], raw[9]]);

        if flags & V2_FLAG_LENGTH_BOUND == 0 || flags & !V2_ALLOWED_FLAGS != 0 {
            return None;
        }
        if usize::from(payload_len) > V2_PAYLOAD_SIZE || raw.len() != V2_HEADER_SIZE + usize::from(payload_len) {
            return None;
        }

        let mut payload = [0u8; V2_PAYLOAD_SIZE];
        payload[..usize::from(payload_len)].copy_from_slice(&raw[V2_HEADER_SIZE..]);

        let frame = Self {
            thought_id,
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

/// Fragment an encoded message using V2 length-bound framing and one XOR parity
/// frame. Empty and over-bound messages are rejected instead of panicking.
pub fn fragment_v2(thought_id: u16, encoded: &[u8]) -> Result<Vec<LoRaFragmentV2>, V2FragmentError> {
    fragment_v2_with_capacity(thought_id, encoded, V2_PAYLOAD_SIZE, false)
}

/// Fragment for an outer 12-byte nonce + 16-byte tag wrapper while preserving
/// the 222-byte transport MTU. This only selects the correct byte budget; it
/// does NOT encrypt, authenticate, or make the existing nonce construction safe.
pub fn fragment_v2_with_aead_overhead(
    thought_id: u16,
    encoded: &[u8],
) -> Result<Vec<LoRaFragmentV2>, V2FragmentError> {
    fragment_v2_with_capacity(thought_id, encoded, V2_AEAD_PAYLOAD_SIZE, true)
}

fn fragment_v2_with_capacity(
    thought_id: u16,
    encoded: &[u8],
    payload_capacity: usize,
    aead_budget: bool,
) -> Result<Vec<LoRaFragmentV2>, V2FragmentError> {
    if encoded.is_empty() {
        return Err(V2FragmentError::EmptyPayload);
    }
    let max_encoded_len = V2_MAX_DATA_FRAGMENTS * payload_capacity;
    if encoded.len() > max_encoded_len || encoded.len() > usize::from(u16::MAX) {
        return Err(V2FragmentError::PayloadTooLarge);
    }
    let data_count = (encoded.len() + payload_capacity - 1) / payload_capacity;
    let total_fragments = u8::try_from(data_count + 1).map_err(|_| V2FragmentError::PayloadTooLarge)?;
    let encoded_len = u16::try_from(encoded.len()).map_err(|_| V2FragmentError::PayloadTooLarge)?;
    let mut frames = Vec::with_capacity(data_count + 1);
    let mut parity = [0u8; V2_PAYLOAD_SIZE];
    let profile_flag = if aead_budget { V2_FLAG_AEAD_BUDGET } else { 0 };

    for index in 0..data_count {
        let start = index * payload_capacity;
        let end = (start + payload_capacity).min(encoded.len());
        let chunk = &encoded[start..end];
        let mut payload = [0u8; V2_PAYLOAD_SIZE];
        payload[..chunk.len()].copy_from_slice(chunk);
        for byte_index in 0..payload_capacity {
            parity[byte_index] ^= payload[byte_index];
        }
        let payload_len = u8::try_from(chunk.len()).map_err(|_| V2FragmentError::PayloadTooLarge)?;
        let flags = V2_FLAG_LENGTH_BOUND | profile_flag;
        let checksum = LoRaFragmentV2::checksum_for(
            thought_id,
            index as u8,
            total_fragments,
            payload_len,
            flags,
            encoded_len,
            &payload[..chunk.len()],
        );
        frames.push(LoRaFragmentV2 {
            thought_id,
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
    let payload_len = payload_capacity as u8;
    let flags = V2_FLAG_LENGTH_BOUND | V2_FLAG_FEC | profile_flag;
    let checksum = LoRaFragmentV2::checksum_for(
        thought_id,
        index,
        total_fragments,
        payload_len,
        flags,
        encoded_len,
        &parity[..payload_capacity],
    );
    frames.push(LoRaFragmentV2 {
        thought_id,
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

/// Reassembler for V2 frames. All fragments must agree on the stream identity,
/// count, and exact encoded length before state can advance.
pub struct FragmentAssemblerV2 {
    thought_id: u16,
    total_fragments: u8,
    encoded_len: usize,
    data_count: usize,
    payload_capacity: usize,
    received: Vec<bool>,
    fragments: Vec<Option<[u8; V2_PAYLOAD_SIZE]>>,
    payload_lens: Vec<u8>,
    fec: Option<[u8; V2_PAYLOAD_SIZE]>,
}

impl FragmentAssemblerV2 {
    /// Configure empty reassembly state from validated stream metadata. The
    /// supplied frame is not counted as received; callers must pass every
    /// received frame to feed(), including this one.
    pub fn new(first: &LoRaFragmentV2) -> Option<Self> {
        if !first.is_well_formed() {
            return None;
        }
        let encoded_len = first.encoded_len();
        let payload_capacity = LoRaFragmentV2::payload_capacity_from_flags(first.flags);
        let data_count = LoRaFragmentV2::data_count_for_len(encoded_len, payload_capacity)?;
        Some(Self {
            thought_id: first.thought_id,
            total_fragments: first.total_fragments,
            encoded_len,
            data_count,
            payload_capacity,
            received: vec![false; data_count + 1],
            fragments: vec![None; data_count],
            payload_lens: vec![0; data_count],
            fec: None,
        })
    }

    pub fn feed(&mut self, frame: &LoRaFragmentV2) -> bool {
        if !frame.is_well_formed()
            || frame.thought_id != self.thought_id
            || frame.total_fragments != self.total_fragments
            || frame.encoded_len() != self.encoded_len
        {
            return false;
        }
        let index = usize::from(frame.fragment_index);
        if index >= self.received.len() {
            return false;
        }
        if self.received[index] {
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
        let present = self.received[..self.data_count].iter().filter(|&&received| received).count();
        present == self.data_count || (present + 1 == self.data_count && self.fec.is_some())
    }

    pub fn assemble(&self) -> Option<Vec<u8>> {
        if !self.is_complete() || self.data_count == 0 {
            return None;
        }
        let missing = self.received[..self.data_count].iter().position(|&received| !received);
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
                    self.encoded_len.checked_sub((self.data_count - 1).checked_mul(self.payload_capacity)?)?
                } else {
                    self.payload_capacity
                };
                if len == 0 || len > self.payload_capacity {
                    return None;
                }
                output.extend_from_slice(&chunk[..len]);
            } else {
                let chunk = self.fragments[index].as_ref()?;
                let len = usize::from(self.payload_lens[index]);
                if len == 0 || len > V2_PAYLOAD_SIZE {
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
    fn exact_length_and_final_fragment_loss_reconstruct_for_boundary_profiles() {
        for len in [1usize, 212, 213, 1800, 2104, 2105, 2133, V2_MAX_ENCODED_LEN] {
            let original = sample(len, len as u64 + 0x1234);
            let frames = fragment_v2(0xBEEF, &original).unwrap();
            assert!(frames.iter().all(|frame| frame.to_bytes().len() <= V2_MTU));
            assert!(frames.last().unwrap().is_fec());

            for lost_index in 0..(frames.len() - 1) {
                let first_received = frames
                    .iter()
                    .enumerate()
                    .find_map(|(index, frame)| (index != lost_index).then_some(frame))
                    .expect("at least one fragment remains");
                let mut assembler = FragmentAssemblerV2::new(first_received).unwrap();
                for (index, frame) in frames.iter().enumerate() {
                    if index == lost_index {
                        continue;
                    }
                    let wire = frame.to_bytes();
                    let decoded = LoRaFragmentV2::from_bytes(&wire).unwrap();
                    assembler.feed(&decoded);
                }
                assert!(assembler.is_complete(), "len={len}, lost={lost_index}");
                assert_eq!(assembler.assemble().as_deref(), Some(original.as_slice()), "len={len}, lost={lost_index}");
            }
        }
    }

    #[test]
    fn full_reassembly_survives_lost_parity() {
        let original = sample(2105, 7);
        let frames = fragment_v2(4, &original).unwrap();
        let mut assembler = FragmentAssemblerV2::new(&frames[0]).unwrap();
        for frame in &frames[..frames.len() - 1] {
            assembler.feed(frame);
        }
        assert_eq!(assembler.assemble().as_deref(), Some(original.as_slice()));
    }

    #[test]
    fn malformed_lengths_and_trailing_bytes_are_rejected() {
        let original = sample(2105, 8);
        let frames = fragment_v2(5, &original).unwrap();
        let mut wire = frames[0].to_bytes();
        wire.push(0);
        assert!(LoRaFragmentV2::from_bytes(&wire).is_none());

        // Change count to an inconsistent value and recompute CRC to ensure the
        // semantic check, not just checksum validation, rejects the frame.
        let mut malformed_wire = frames[0].to_bytes();
        let malformed_total = malformed_wire[3] - 1;
        malformed_wire[3] = malformed_total;
        let malformed_checksum = LoRaFragmentV2::checksum_for(
            frames[0].thought_id,
            frames[0].fragment_index,
            malformed_total,
            frames[0].payload_len,
            frames[0].flags,
            frames[0].encoded_len,
            &frames[0].payload[..usize::from(frames[0].payload_len)],
        );
        malformed_wire[6..8].copy_from_slice(&malformed_checksum.to_le_bytes());
        assert!(LoRaFragmentV2::from_bytes(&malformed_wire).is_none());

        assert!(fragment_v2(1, &[]).is_err());
        assert!(fragment_v2(1, &vec![0u8; V2_MAX_ENCODED_LEN + 1]).is_err());
    }

    #[test]
    fn v2_frames_are_not_accepted_by_the_legacy_parser() {
        use super::super::lora_fragment::LoRaFragment;

        let frames = fragment_v2(3, &sample(2105, 11)).unwrap();
        for frame in frames {
            assert!(
                LoRaFragment::from_bytes(&frame.to_bytes()).is_none(),
                "reserved V2 flag must make legacy parser fail closed"
            );
        }
    }

    #[test]
    fn assembler_rejects_mixed_encoded_lengths_without_advancing_state() {
        let frames_a = fragment_v2(6, &sample(2105, 9)).unwrap();
        let frames_b = fragment_v2(6, &sample(2104, 10)).unwrap();
        let mut assembler = FragmentAssemblerV2::new(&frames_a[0]).unwrap();
        assert!(!assembler.feed(&frames_b[0]));
        assert_eq!(assembler.received.iter().filter(|&&v| v).count(), 0);
    }
}
