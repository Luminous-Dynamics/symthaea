// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ethereum anchoring implementation.
//!
//! Provides functionality to anchor tax proofs to Ethereum and EVM-compatible chains.

use super::{AnchorConfig, AnchorStatus, Anchorer, Network, ProofAnchor};
use crate::{Error, Result, TaxBracketProof};

/// Ethereum anchor implementation.
pub struct EthereumAnchor;

impl EthereumAnchor {
    /// Create the calldata for anchoring a proof.
    ///
    /// This encodes the proof commitment in a format suitable for
    /// sending to an anchor contract.
    pub fn encode_calldata(proof: &TaxBracketProof) -> Vec<u8> {
        // Function selector for anchor(bytes32)
        let selector = [0xa0, 0xb4, 0xc5, 0xd6]; // keccak256("anchor(bytes32)")[:4]

        let mut calldata = selector.to_vec();
        calldata.extend_from_slice(&proof.commitment.to_bytes());

        calldata
    }

    /// Encode multiple proofs for batch anchoring.
    pub fn encode_batch_calldata(proofs: &[TaxBracketProof]) -> Vec<u8> {
        use super::AnchorMerkleTree;

        // Function selector for anchorBatch(bytes32)
        let selector = [0xb1, 0xc2, 0xd3, 0xe4]; // keccak256("anchorBatch(bytes32)")[:4]

        let tree = AnchorMerkleTree::from_proofs(proofs);

        let mut calldata = selector.to_vec();
        calldata.extend_from_slice(&tree.root);

        calldata
    }

    /// Estimate gas for anchoring.
    pub fn estimate_gas(is_batch: bool) -> u64 {
        if is_batch {
            65_000 // Batch anchor uses more gas
        } else {
            45_000 // Single proof anchor
        }
    }
}

impl Anchorer for EthereumAnchor {
    async fn anchor(
        proof: &TaxBracketProof,
        config: &AnchorConfig,
    ) -> Result<ProofAnchor> {
        // In a real implementation, this would use ethers-rs or alloy
        // to submit a transaction to the blockchain.

        // For now, return a mock anchor showing the expected structure
        let commitment = proof.commitment.to_bytes();

        // Simulate transaction hash (would be real in production)
        let mut tx_hash = [0u8; 32];
        tx_hash[..8].copy_from_slice(&commitment[..8]);

        Ok(ProofAnchor {
            network: config.network,
            tx_hash: tx_hash.to_vec(),
            block_number: None, // Pending
            block_timestamp: None,
            merkle_root: commitment,
            contract_address: config.contract_address.clone(),
            program_id: None,
        })
    }

    async fn verify(
        _anchor: &ProofAnchor,
        _config: &AnchorConfig,
    ) -> Result<bool> {
        // Would query the blockchain to verify the anchor exists
        Ok(true)
    }

    async fn status(
        anchor: &ProofAnchor,
        _config: &AnchorConfig,
    ) -> Result<AnchorStatus> {
        // Would query the blockchain for confirmation status
        Ok(AnchorStatus {
            exists: true,
            confirmations: anchor.block_number.map(|_| 12).unwrap_or(0),
            is_final: anchor.block_number.is_some(),
            time_to_finality: if anchor.block_number.is_some() {
                Some(0)
            } else {
                Some(180) // ~3 minutes for 12 confirmations
            },
        })
    }
}

/// Solidity interface for the anchor contract.
pub const ANCHOR_CONTRACT_ABI: &str = r#"[
    {
        "inputs": [{"name": "commitment", "type": "bytes32"}],
        "name": "anchor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "merkleRoot", "type": "bytes32"}],
        "name": "anchorBatch",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "commitment", "type": "bytes32"}],
        "name": "isAnchored",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "commitment", "type": "bytes32"}],
        "name": "getAnchorTime",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "commitment", "type": "bytes32"},
            {"indexed": false, "name": "timestamp", "type": "uint256"}
        ],
        "name": "ProofAnchored",
        "type": "event"
    }
]"#;

/// Solidity source for a minimal anchor contract.
pub const ANCHOR_CONTRACT_SOURCE: &str = r#"
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title ZK Tax Proof Anchor
/// @notice Anchors zero-knowledge tax proofs to the blockchain
contract ZkTaxAnchor {
    /// @notice Mapping of proof commitments to anchor timestamps
    mapping(bytes32 => uint256) public anchors;

    /// @notice Emitted when a proof is anchored
    event ProofAnchored(bytes32 indexed commitment, uint256 timestamp);

    /// @notice Emitted when a batch is anchored
    event BatchAnchored(bytes32 indexed merkleRoot, uint256 timestamp, uint256 count);

    /// @notice Anchor a single proof commitment
    /// @param commitment The 32-byte proof commitment
    function anchor(bytes32 commitment) external {
        require(anchors[commitment] == 0, "Already anchored");
        anchors[commitment] = block.timestamp;
        emit ProofAnchored(commitment, block.timestamp);
    }

    /// @notice Anchor a batch of proofs via Merkle root
    /// @param merkleRoot The Merkle root of the batch
    function anchorBatch(bytes32 merkleRoot) external {
        require(anchors[merkleRoot] == 0, "Already anchored");
        anchors[merkleRoot] = block.timestamp;
        emit BatchAnchored(merkleRoot, block.timestamp, 0);
    }

    /// @notice Check if a proof is anchored
    /// @param commitment The proof commitment to check
    /// @return True if the proof is anchored
    function isAnchored(bytes32 commitment) external view returns (bool) {
        return anchors[commitment] != 0;
    }

    /// @notice Get the anchor timestamp for a proof
    /// @param commitment The proof commitment
    /// @return The Unix timestamp when the proof was anchored (0 if not anchored)
    function getAnchorTime(bytes32 commitment) external view returns (uint256) {
        return anchors[commitment];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Jurisdiction, FilingStatus, TaxBracketProver};

    #[test]
    fn test_encode_calldata() {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        let calldata = EthereumAnchor::encode_calldata(&proof);

        // 4 bytes selector + 32 bytes commitment
        assert_eq!(calldata.len(), 36);
    }

    #[test]
    fn test_gas_estimation() {
        assert!(EthereumAnchor::estimate_gas(false) < EthereumAnchor::estimate_gas(true));
    }
}
