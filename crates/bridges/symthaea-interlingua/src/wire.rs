// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protocol-stable HDC wire codecs for SCIP.
//!
//! `HdcPayload` remains the in-memory continuous projection. This module defines
//! simple byte encodings that independent implementations can reproduce without
//! depending on Symthaea's internal compression libraries.

use crate::HdcPayload;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MAX_HDC_WIRE_DIMENSION: usize = 65_536;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum HdcWireEncoding {
    /// Exact IEEE-754 f32 values, little-endian.
    F32LeV1,
    /// One unsigned byte per component. 0..=254 encode q=-127..=127.
    /// Code 255 is reserved and MUST be rejected.
    Q8SymmetricV1,
    /// Two 4-bit codes per byte. 0..=14 encode q=-7..=7.
    /// Code 15 is reserved and is the required pad nibble for odd dimensions.
    Q4SymmetricV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcWirePacket {
    pub encoding: HdcWireEncoding,
    pub dimension: u32,
    pub semantic_hash: String,
    pub profile_fingerprint: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireFidelity {
    pub cosine_similarity: f32,
    pub max_abs_error: f32,
    pub exact: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WireSelection {
    pub packet: HdcWirePacket,
    pub fidelity: WireFidelity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireSelectionPolicy {
    pub require_exact: bool,
    pub minimum_cosine: f32,
    pub prefer_smallest: bool,
}

impl Default for WireSelectionPolicy {
    fn default() -> Self {
        Self {
            require_exact: false,
            minimum_cosine: 0.995,
            prefer_smallest: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HdcWireError {
    InvalidDimension,
    InvalidPayload(&'static str),
    InvalidPacket(&'static str),
    NoCompatibleEncoding,
}

impl std::fmt::Display for HdcWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "invalid HDC wire dimension"),
            Self::InvalidPayload(message) => write!(f, "invalid HDC payload: {message}"),
            Self::InvalidPacket(message) => write!(f, "invalid HDC wire packet: {message}"),
            Self::NoCompatibleEncoding => write!(f, "no compatible HDC wire encoding"),
        }
    }
}

impl std::error::Error for HdcWireError {}

impl HdcWirePacket {
    pub fn encode(
        payload: &HdcPayload,
        encoding: HdcWireEncoding,
    ) -> Result<Self, HdcWireError> {
        if payload.values.is_empty()
            || payload.values.len() > MAX_HDC_WIRE_DIMENSION
            || payload.values.len() > u32::MAX as usize
        {
            return Err(HdcWireError::InvalidDimension);
        }
        if payload.semantic_hash.trim().is_empty()
            || payload.profile_fingerprint.trim().is_empty()
        {
            return Err(HdcWireError::InvalidPayload(
                "semantic hash and profile fingerprint are required",
            ));
        }
        if payload.values.iter().any(|value| !value.is_finite()) {
            return Err(HdcWireError::InvalidPayload("non-finite component"));
        }

        let bytes = match encoding {
            HdcWireEncoding::F32LeV1 => encode_f32_le(&payload.values),
            HdcWireEncoding::Q8SymmetricV1 => encode_q8(&payload.values)?,
            HdcWireEncoding::Q4SymmetricV1 => encode_q4(&payload.values)?,
        };

        Ok(Self {
            encoding,
            dimension: payload.values.len() as u32,
            semantic_hash: payload.semantic_hash.clone(),
            profile_fingerprint: payload.profile_fingerprint.clone(),
            bytes,
        })
    }

    pub fn decode(&self) -> Result<HdcPayload, HdcWireError> {
        let dimension = self.dimension as usize;
        if dimension == 0 || dimension > MAX_HDC_WIRE_DIMENSION {
            return Err(HdcWireError::InvalidDimension);
        }
        if self.semantic_hash.trim().is_empty() || self.profile_fingerprint.trim().is_empty() {
            return Err(HdcWireError::InvalidPacket(
                "semantic hash and profile fingerprint are required",
            ));
        }

        let values = match self.encoding {
            HdcWireEncoding::F32LeV1 => decode_f32_le(&self.bytes, dimension)?,
            HdcWireEncoding::Q8SymmetricV1 => decode_q8(&self.bytes, dimension)?,
            HdcWireEncoding::Q4SymmetricV1 => decode_q4(&self.bytes, dimension)?,
        };

        Ok(HdcPayload {
            values,
            semantic_hash: self.semantic_hash.clone(),
            profile_fingerprint: self.profile_fingerprint.clone(),
        })
    }

    pub fn fidelity_against(&self, source: &HdcPayload) -> Result<WireFidelity, HdcWireError> {
        if source.values.len() != self.dimension as usize
            || source.semantic_hash != self.semantic_hash
            || source.profile_fingerprint != self.profile_fingerprint
        {
            return Err(HdcWireError::InvalidPacket(
                "packet does not identify the source projection",
            ));
        }

        let decoded = self.decode()?;
        let cosine_similarity = cosine_similarity(&source.values, &decoded.values);
        let max_abs_error = source
            .values
            .iter()
            .zip(&decoded.values)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);

        Ok(WireFidelity {
            cosine_similarity,
            max_abs_error,
            exact: source.values == decoded.values,
        })
    }

    /// Verify a lossy projection exactly at the wire-representation level.
    ///
    /// A receiver that owns the grounded graph can recompute the expected HDC
    /// projection, re-encode it with the negotiated codec, and compare bytes.
    /// This is stronger than accepting an arbitrary cosine-similar vector.
    pub fn reencode_matches(&self, expected: &HdcPayload) -> Result<bool, HdcWireError> {
        Ok(Self::encode(expected, self.encoding)? == *self)
    }

    pub fn wire_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn compression_ratio_vs_f32(&self) -> f32 {
        let dense_bytes = self.dimension as usize * std::mem::size_of::<f32>();
        if self.bytes.is_empty() {
            f32::INFINITY
        } else {
            dense_bytes as f32 / self.bytes.len() as f32
        }
    }
}

pub fn select_wire_encoding(
    payload: &HdcPayload,
    local: &[HdcWireEncoding],
    remote: &[HdcWireEncoding],
    policy: WireSelectionPolicy,
) -> Result<WireSelection, HdcWireError> {
    if !policy.minimum_cosine.is_finite() || !(-1.0..=1.0).contains(&policy.minimum_cosine) {
        return Err(HdcWireError::InvalidPayload(
            "minimum cosine must be finite and in [-1, 1]",
        ));
    }

    let local: BTreeSet<_> = local.iter().copied().collect();
    let remote: BTreeSet<_> = remote.iter().copied().collect();
    let common = local.intersection(&remote).copied().collect::<Vec<_>>();

    if policy.require_exact {
        if !common.contains(&HdcWireEncoding::F32LeV1) {
            return Err(HdcWireError::NoCompatibleEncoding);
        }
        let packet = HdcWirePacket::encode(payload, HdcWireEncoding::F32LeV1)?;
        let fidelity = packet.fidelity_against(payload)?;
        return Ok(WireSelection { packet, fidelity });
    }

    let mut candidates = Vec::new();
    for encoding in common {
        let packet = HdcWirePacket::encode(payload, encoding)?;
        let fidelity = packet.fidelity_against(payload)?;
        if fidelity.cosine_similarity >= policy.minimum_cosine {
            candidates.push(WireSelection { packet, fidelity });
        }
    }

    if candidates.is_empty() {
        return Err(HdcWireError::NoCompatibleEncoding);
    }

    if policy.prefer_smallest {
        candidates.sort_by(|left, right| {
            left.packet
                .wire_bytes()
                .cmp(&right.packet.wire_bytes())
                .then_with(|| {
                    right
                        .fidelity
                        .cosine_similarity
                        .total_cmp(&left.fidelity.cosine_similarity)
                })
        });
    } else {
        candidates.sort_by(|left, right| {
            right
                .fidelity
                .cosine_similarity
                .total_cmp(&left.fidelity.cosine_similarity)
                .then_with(|| left.packet.wire_bytes().cmp(&right.packet.wire_bytes()))
        });
    }

    Ok(candidates.remove(0))
}

fn require_unit_interval(values: &[f32]) -> Result<(), HdcWireError> {
    if values.iter().any(|value| !(-1.0..=1.0).contains(value)) {
        return Err(HdcWireError::InvalidPayload(
            "Q8/Q4 require components in [-1, 1]",
        ));
    }
    Ok(())
}

/// Round half values away from zero, specified explicitly for interoperability.
fn round_away_from_zero(value: f32) -> i16 {
    if value >= 0.0 {
        (value + 0.5).floor() as i16
    } else {
        (value - 0.5).ceil() as i16
    }
}

fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32_le(bytes: &[u8], dimension: usize) -> Result<Vec<f32>, HdcWireError> {
    if bytes.len() != dimension * 4 {
        return Err(HdcWireError::InvalidPacket("incorrect f32 byte length"));
    }

    let mut values = Vec::with_capacity(dimension);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(HdcWireError::InvalidPacket("non-finite f32 component"));
        }
        values.push(value);
    }
    Ok(values)
}

fn encode_q8(values: &[f32]) -> Result<Vec<u8>, HdcWireError> {
    require_unit_interval(values)?;
    Ok(values
        .iter()
        .map(|value| {
            let quantized = round_away_from_zero(*value * 127.0).clamp(-127, 127);
            (quantized + 127) as u8
        })
        .collect())
}

fn decode_q8(bytes: &[u8], dimension: usize) -> Result<Vec<f32>, HdcWireError> {
    if bytes.len() != dimension {
        return Err(HdcWireError::InvalidPacket("incorrect Q8 byte length"));
    }

    bytes
        .iter()
        .map(|code| {
            if *code == 255 {
                return Err(HdcWireError::InvalidPacket("reserved Q8 code 255"));
            }
            let quantized = *code as i16 - 127;
            Ok(quantized as f32 / 127.0)
        })
        .collect()
}

fn encode_q4(values: &[f32]) -> Result<Vec<u8>, HdcWireError> {
    require_unit_interval(values)?;
    let mut codes = values
        .iter()
        .map(|value| {
            let quantized = round_away_from_zero(*value * 7.0).clamp(-7, 7);
            (quantized + 7) as u8
        })
        .collect::<Vec<_>>();

    if codes.len() % 2 == 1 {
        codes.push(15);
    }

    Ok(codes
        .chunks_exact(2)
        .map(|pair| (pair[0] << 4) | pair[1])
        .collect())
}

fn decode_q4(bytes: &[u8], dimension: usize) -> Result<Vec<f32>, HdcWireError> {
    if bytes.len() != dimension.div_ceil(2) {
        return Err(HdcWireError::InvalidPacket("incorrect Q4 byte length"));
    }

    let mut values = Vec::with_capacity(dimension);
    for (byte_index, byte) in bytes.iter().enumerate() {
        for nibble_index in 0..2 {
            let code = if nibble_index == 0 {
                byte >> 4
            } else {
                byte & 0x0f
            };
            let component_index = byte_index * 2 + nibble_index;
            if component_index >= dimension {
                if code != 15 {
                    return Err(HdcWireError::InvalidPacket(
                        "odd Q4 dimension requires reserved pad nibble 15",
                    ));
                }
                continue;
            }
            if code == 15 {
                return Err(HdcWireError::InvalidPacket(
                    "reserved Q4 code 15 inside vector",
                ));
            }
            let quantized = code as i16 - 7;
            values.push(quantized as f32 / 7.0);
        }
    }
    Ok(values)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let (mut dot, mut left_norm, mut right_norm) = (0.0f64, 0.0f64, 0.0f64);
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_value = left_value as f64;
        let right_value = right_value as f64;
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    let denominator = (left_norm * right_norm).sqrt();
    if denominator <= f64::EPSILON {
        if left == right { 1.0 } else { 0.0 }
    } else {
        (dot / denominator).clamp(-1.0, 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(values: Vec<f32>) -> HdcPayload {
        HdcPayload {
            values,
            semantic_hash: "semantic".into(),
            profile_fingerprint: "profile".into(),
        }
    }

    #[test]
    fn q8_golden_vector_is_stable() {
        let source = payload(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        let packet = HdcWirePacket::encode(&source, HdcWireEncoding::Q8SymmetricV1).unwrap();
        assert_eq!(packet.bytes, vec![0, 63, 127, 191, 254]);
        assert_eq!(packet.decode().unwrap().values.len(), 5);
    }

    #[test]
    fn q4_golden_vector_is_stable() {
        let source = payload(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        let packet = HdcWirePacket::encode(&source, HdcWireEncoding::Q4SymmetricV1).unwrap();
        assert_eq!(packet.bytes, vec![0x03, 0x7b, 0xef]);
        assert_eq!(packet.decode().unwrap().values.len(), 5);
    }

    #[test]
    fn f32_round_trip_is_exact() {
        let source = payload(vec![-0.25, 0.0, 0.75, 1.0]);
        let packet = HdcWirePacket::encode(&source, HdcWireEncoding::F32LeV1).unwrap();
        let fidelity = packet.fidelity_against(&source).unwrap();
        assert!(fidelity.exact);
        assert_eq!(fidelity.max_abs_error, 0.0);
        assert!(packet.reencode_matches(&source).unwrap());
    }

    #[test]
    fn reserved_q8_code_is_rejected() {
        let packet = HdcWirePacket {
            encoding: HdcWireEncoding::Q8SymmetricV1,
            dimension: 1,
            semantic_hash: "semantic".into(),
            profile_fingerprint: "profile".into(),
            bytes: vec![255],
        };
        assert!(packet.decode().is_err());
    }

    #[test]
    fn adaptive_selection_can_reject_coarse_q4() {
        let source = payload(vec![0.03, 0.07, 0.11, 0.17, 0.23, 0.31, 0.47, 0.73]);
        let encodings = [
            HdcWireEncoding::Q4SymmetricV1,
            HdcWireEncoding::Q8SymmetricV1,
            HdcWireEncoding::F32LeV1,
        ];
        let selection = select_wire_encoding(
            &source,
            &encodings,
            &encodings,
            WireSelectionPolicy {
                minimum_cosine: 0.999,
                ..Default::default()
            },
        )
        .unwrap();

        assert_ne!(selection.packet.encoding, HdcWireEncoding::Q4SymmetricV1);
        assert!(selection.fidelity.cosine_similarity >= 0.999);
    }
}
