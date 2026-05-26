// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! HDC encoders for ecological state.

pub mod aggregation;
pub mod biome;
pub mod prediction;
pub mod proof;

use crate::evidence::EvidencePacket;
use crate::hdc::proof::{HdcBindingProof, prove_interpretation_integrity};

/// Encodes ecological EvidencePackets into Hyperdimensional Vectors.
pub struct EcologicalEncoder {
    pub dimension: usize,
}

impl EcologicalEncoder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Encodes a packet and generates a Binius integrity proof.
    pub fn encode_with_proof(
        &self,
        packet: &EvidencePacket,
    ) -> anyhow::Result<(Vec<u8>, HdcBindingProof)> {
        // 1. Neural Encoding (HDC Projection)
        let mut hv = vec![0u8; self.dimension / 8];
        let index = (packet.value * (self.dimension - 1) as f64) as usize;
        if index < self.dimension {
            let byte_idx = index / 8;
            let bit_idx = index % 8;
            hv[byte_idx] |= 1 << bit_idx;
        }

        // 2. Binius Interpretation Proof (The Interpretation Lock)
        let integrity_proof = prove_interpretation_integrity(packet, &hv)?;

        Ok((hv, integrity_proof))
    }
}
