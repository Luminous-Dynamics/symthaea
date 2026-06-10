// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bitcoin anchoring implementation.
//!
//! Provides functionality to anchor tax proofs to Bitcoin using OP_RETURN.

use super::{AnchorConfig, AnchorStatus, Anchorer, ProofAnchor};
use crate::{Error, Result, TaxBracketProof};

/// Bitcoin anchor implementation using OP_RETURN.
pub struct BitcoinAnchor;

impl BitcoinAnchor {
    /// Maximum OP_RETURN data size (80 bytes standard, some nodes accept 83)
    pub const MAX_OP_RETURN_SIZE: usize = 80;

    /// Create OP_RETURN data for a proof.
    ///
    /// Format: [4-byte prefix][32-byte commitment][optional metadata]
    pub fn create_op_return_data(proof: &TaxBracketProof) -> Vec<u8> {
        // Prefix to identify ZK Tax anchors
        let prefix = b"ZKTX";

        let mut data = prefix.to_vec();
        data.extend_from_slice(&proof.commitment.to_bytes());

        // Add tax year as 2 bytes
        data.extend_from_slice(&(proof.tax_year as u16).to_be_bytes());

        // Add bracket index
        data.push(proof.bracket_index);

        data
    }

    /// Estimate transaction fee in satoshis.
    ///
    /// Assumes a simple 1-input, 2-output transaction (change + OP_RETURN).
    pub fn estimate_fee_sats(fee_rate_sats_per_vb: u64) -> u64 {
        // Typical OP_RETURN tx size: ~230 vbytes
        let vsize = 230;
        fee_rate_sats_per_vb * vsize
    }

    /// Create a minimal OP_RETURN script.
    pub fn create_op_return_script(data: &[u8]) -> Vec<u8> {
        assert!(data.len() <= Self::MAX_OP_RETURN_SIZE);

        let mut script = vec![
            0x6a, // OP_RETURN
        ];

        if data.len() <= 75 {
            script.push(data.len() as u8);
        } else {
            script.push(0x4c); // OP_PUSHDATA1
            script.push(data.len() as u8);
        }

        script.extend_from_slice(data);
        script
    }

    /// Parse OP_RETURN data to extract commitment.
    pub fn parse_op_return_data(data: &[u8]) -> Option<[u8; 32]> {
        if data.len() < 36 {
            return None;
        }

        // Check prefix
        if &data[..4] != b"ZKTX" {
            return None;
        }

        // Extract commitment
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&data[4..36]);

        Some(commitment)
    }
}

impl Anchorer for BitcoinAnchor {
    async fn anchor(
        proof: &TaxBracketProof,
        config: &AnchorConfig,
    ) -> Result<ProofAnchor> {
        // In a real implementation, this would use bitcoin-rpc
        // to create and broadcast a transaction.

        let commitment = proof.commitment.to_bytes();
        let op_return_data = Self::create_op_return_data(proof);

        // Simulate txid (would be real double-SHA256 in production)
        let mut tx_hash = [0u8; 32];
        tx_hash[..8].copy_from_slice(&commitment[..8]);

        Ok(ProofAnchor {
            network: config.network,
            tx_hash: tx_hash.to_vec(),
            block_number: None,
            block_timestamp: None,
            merkle_root: commitment,
            contract_address: None,
            program_id: None,
        })
    }

    async fn verify(
        _anchor: &ProofAnchor,
        _config: &AnchorConfig,
    ) -> Result<bool> {
        Ok(true)
    }

    async fn status(
        anchor: &ProofAnchor,
        _config: &AnchorConfig,
    ) -> Result<AnchorStatus> {
        Ok(AnchorStatus {
            exists: true,
            confirmations: if anchor.block_number.is_some() { 6 } else { 0 },
            is_final: anchor.block_number.map(|_| true).unwrap_or(false),
            time_to_finality: if anchor.block_number.is_some() {
                Some(0)
            } else {
                Some(3600) // ~1 hour for 6 confirmations
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Jurisdiction, FilingStatus, TaxBracketProver};

    #[test]
    fn test_op_return_data() {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        let data = BitcoinAnchor::create_op_return_data(&proof);

        // 4 prefix + 32 commitment + 2 year + 1 bracket = 39 bytes
        assert_eq!(data.len(), 39);
        assert_eq!(&data[..4], b"ZKTX");

        // Should be under OP_RETURN limit
        assert!(data.len() <= BitcoinAnchor::MAX_OP_RETURN_SIZE);
    }

    #[test]
    fn test_parse_op_return() {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        let data = BitcoinAnchor::create_op_return_data(&proof);
        let parsed = BitcoinAnchor::parse_op_return_data(&data).unwrap();

        assert_eq!(parsed, proof.commitment.to_bytes());
    }

    #[test]
    fn test_op_return_script() {
        let data = b"ZKTX".to_vec();
        let script = BitcoinAnchor::create_op_return_script(&data);

        assert_eq!(script[0], 0x6a); // OP_RETURN
        assert_eq!(script[1], 4);    // Length
        assert_eq!(&script[2..], b"ZKTX");
    }

    #[test]
    fn test_fee_estimation() {
        let fee = BitcoinAnchor::estimate_fee_sats(10); // 10 sat/vB
        assert!(fee > 0);
        assert!(fee < 10_000); // Less than 0.0001 BTC
    }
}
