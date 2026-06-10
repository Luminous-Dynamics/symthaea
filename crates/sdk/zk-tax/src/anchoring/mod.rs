// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Blockchain anchoring for tax proofs.
//!
//! This module provides utilities for anchoring tax proofs to various
//! blockchains, creating immutable records of proof existence.
//!
//! # Supported Blockchains
//!
//! - Ethereum (and EVM-compatible chains)
//! - Solana
//! - Bitcoin (via OP_RETURN)
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_zk_tax::anchoring::{EthereumAnchor, AnchorConfig};
//!
//! // Create anchor configuration
//! let config = AnchorConfig::ethereum()
//!     .rpc_url("https://mainnet.infura.io/v3/YOUR_KEY")
//!     .contract_address("0x...");
//!
//! // Anchor a proof
//! let anchor = EthereumAnchor::anchor(&proof, &config).await?;
//! println!("Anchored at tx: 0x{}", hex::encode(anchor.tx_hash));
//!
//! // Verify anchor
//! let verified = anchor.verify(&config).await?;
//! assert!(verified);
//! ```

pub mod ethereum;
pub mod solana;
pub mod bitcoin;

use crate::{TaxBracketProof, Error, Result};
use serde::{Deserialize, Serialize};

// =============================================================================
// Common Types
// =============================================================================

/// Blockchain network identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Network {
    /// Ethereum mainnet
    EthereumMainnet,
    /// Ethereum Sepolia testnet
    EthereumSepolia,
    /// Ethereum Goerli testnet (deprecated)
    EthereumGoerli,
    /// Polygon mainnet
    Polygon,
    /// Polygon Mumbai testnet
    PolygonMumbai,
    /// Arbitrum One
    Arbitrum,
    /// Optimism mainnet
    Optimism,
    /// Base mainnet
    Base,
    /// Solana mainnet-beta
    SolanaMainnet,
    /// Solana devnet
    SolanaDevnet,
    /// Bitcoin mainnet
    BitcoinMainnet,
    /// Bitcoin testnet
    BitcoinTestnet,
}

impl Network {
    /// Get the chain ID for EVM networks.
    pub fn chain_id(&self) -> Option<u64> {
        match self {
            Network::EthereumMainnet => Some(1),
            Network::EthereumSepolia => Some(11155111),
            Network::EthereumGoerli => Some(5),
            Network::Polygon => Some(137),
            Network::PolygonMumbai => Some(80001),
            Network::Arbitrum => Some(42161),
            Network::Optimism => Some(10),
            Network::Base => Some(8453),
            _ => None,
        }
    }

    /// Check if this is a testnet.
    pub fn is_testnet(&self) -> bool {
        matches!(
            self,
            Network::EthereumSepolia
                | Network::EthereumGoerli
                | Network::PolygonMumbai
                | Network::SolanaDevnet
                | Network::BitcoinTestnet
        )
    }

    /// Get the network name.
    pub fn name(&self) -> &'static str {
        match self {
            Network::EthereumMainnet => "Ethereum Mainnet",
            Network::EthereumSepolia => "Ethereum Sepolia",
            Network::EthereumGoerli => "Ethereum Goerli",
            Network::Polygon => "Polygon",
            Network::PolygonMumbai => "Polygon Mumbai",
            Network::Arbitrum => "Arbitrum One",
            Network::Optimism => "Optimism",
            Network::Base => "Base",
            Network::SolanaMainnet => "Solana Mainnet",
            Network::SolanaDevnet => "Solana Devnet",
            Network::BitcoinMainnet => "Bitcoin Mainnet",
            Network::BitcoinTestnet => "Bitcoin Testnet",
        }
    }
}

/// A blockchain anchor for a tax proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofAnchor {
    /// The network where the proof was anchored
    pub network: Network,
    /// Transaction hash (32 bytes for EVM/Solana, variable for Bitcoin)
    pub tx_hash: Vec<u8>,
    /// Block number (None for unconfirmed)
    pub block_number: Option<u64>,
    /// Block timestamp (Unix epoch)
    pub block_timestamp: Option<u64>,
    /// Merkle root of the proof commitment
    pub merkle_root: [u8; 32],
    /// Contract address (for smart contract anchors)
    pub contract_address: Option<String>,
    /// Program ID (for Solana)
    pub program_id: Option<String>,
}

impl ProofAnchor {
    /// Get the transaction hash as hex string.
    pub fn tx_hash_hex(&self) -> String {
        hex::encode(&self.tx_hash)
    }

    /// Check if the anchor is confirmed.
    pub fn is_confirmed(&self) -> bool {
        self.block_number.is_some()
    }

    /// Get explorer URL for this transaction.
    pub fn explorer_url(&self) -> Option<String> {
        let tx_hex = self.tx_hash_hex();
        match self.network {
            Network::EthereumMainnet => Some(format!("https://etherscan.io/tx/0x{}", tx_hex)),
            Network::EthereumSepolia => Some(format!("https://sepolia.etherscan.io/tx/0x{}", tx_hex)),
            Network::Polygon => Some(format!("https://polygonscan.com/tx/0x{}", tx_hex)),
            Network::Arbitrum => Some(format!("https://arbiscan.io/tx/0x{}", tx_hex)),
            Network::Optimism => Some(format!("https://optimistic.etherscan.io/tx/0x{}", tx_hex)),
            Network::Base => Some(format!("https://basescan.org/tx/0x{}", tx_hex)),
            Network::SolanaMainnet => Some(format!("https://solscan.io/tx/{}", tx_hex)),
            Network::SolanaDevnet => Some(format!("https://solscan.io/tx/{}?cluster=devnet", tx_hex)),
            Network::BitcoinMainnet => Some(format!("https://blockstream.info/tx/{}", tx_hex)),
            Network::BitcoinTestnet => Some(format!("https://blockstream.info/testnet/tx/{}", tx_hex)),
            _ => None,
        }
    }
}

/// Configuration for anchoring operations.
#[derive(Clone, Debug)]
pub struct AnchorConfig {
    /// Target network
    pub network: Network,
    /// RPC endpoint URL
    pub rpc_url: String,
    /// Private key for signing (hex encoded, without 0x prefix)
    pub private_key: Option<String>,
    /// Contract address for smart contract anchors
    pub contract_address: Option<String>,
    /// Gas price in gwei (for EVM chains)
    pub gas_price_gwei: Option<u64>,
    /// Maximum gas limit
    pub gas_limit: Option<u64>,
}

impl AnchorConfig {
    /// Create config for Ethereum mainnet.
    pub fn ethereum_mainnet(rpc_url: &str) -> Self {
        Self {
            network: Network::EthereumMainnet,
            rpc_url: rpc_url.to_string(),
            private_key: None,
            contract_address: None,
            gas_price_gwei: None,
            gas_limit: Some(100_000),
        }
    }

    /// Create config for Ethereum Sepolia testnet.
    pub fn ethereum_sepolia(rpc_url: &str) -> Self {
        Self {
            network: Network::EthereumSepolia,
            rpc_url: rpc_url.to_string(),
            private_key: None,
            contract_address: None,
            gas_price_gwei: None,
            gas_limit: Some(100_000),
        }
    }

    /// Create config for Solana mainnet.
    pub fn solana_mainnet(rpc_url: &str) -> Self {
        Self {
            network: Network::SolanaMainnet,
            rpc_url: rpc_url.to_string(),
            private_key: None,
            contract_address: None,
            gas_price_gwei: None,
            gas_limit: None,
        }
    }

    /// Set the private key.
    pub fn with_private_key(mut self, key: &str) -> Self {
        self.private_key = Some(key.trim_start_matches("0x").to_string());
        self
    }

    /// Set the contract address.
    pub fn with_contract(mut self, address: &str) -> Self {
        self.contract_address = Some(address.to_string());
        self
    }

    /// Set gas price in gwei.
    pub fn with_gas_price(mut self, gwei: u64) -> Self {
        self.gas_price_gwei = Some(gwei);
        self
    }
}

/// Trait for blockchain anchoring implementations.
pub trait Anchorer {
    /// Anchor a proof to the blockchain.
    fn anchor(
        proof: &TaxBracketProof,
        config: &AnchorConfig,
    ) -> impl std::future::Future<Output = Result<ProofAnchor>> + Send;

    /// Verify an existing anchor.
    fn verify(
        anchor: &ProofAnchor,
        config: &AnchorConfig,
    ) -> impl std::future::Future<Output = Result<bool>> + Send;

    /// Get anchor status (confirmations, etc.).
    fn status(
        anchor: &ProofAnchor,
        config: &AnchorConfig,
    ) -> impl std::future::Future<Output = Result<AnchorStatus>> + Send;
}

/// Status of an anchor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorStatus {
    /// Whether the anchor exists on-chain
    pub exists: bool,
    /// Number of confirmations
    pub confirmations: u64,
    /// Whether it's considered final
    pub is_final: bool,
    /// Estimated time to finality (seconds)
    pub time_to_finality: Option<u64>,
}

// =============================================================================
// Merkle Tree for Batch Anchoring
// =============================================================================

/// Merkle tree for efficient batch anchoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorMerkleTree {
    /// All leaf hashes (proof commitments)
    pub leaves: Vec<[u8; 32]>,
    /// Root hash
    pub root: [u8; 32],
    /// Tree height
    pub height: usize,
}

impl AnchorMerkleTree {
    /// Create a new Merkle tree from proofs.
    pub fn from_proofs(proofs: &[TaxBracketProof]) -> Self {
        let leaves: Vec<[u8; 32]> = proofs
            .iter()
            .map(|p| p.commitment.to_bytes())
            .collect();

        let root = Self::compute_root(&leaves);
        let height = (leaves.len() as f64).log2().ceil() as usize;

        Self { leaves, root, height }
    }

    /// Compute the Merkle root.
    fn compute_root(leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return [0u8; 32];
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        let mut current_level = leaves.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let hash = if chunk.len() == 2 {
                    Self::hash_pair(&chunk[0], &chunk[1])
                } else {
                    // Odd number of nodes: duplicate the last one
                    Self::hash_pair(&chunk[0], &chunk[0])
                };
                next_level.push(hash);
            }

            current_level = next_level;
        }

        current_level[0]
    }

    /// Hash two nodes together.
    fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);

        let result = hasher.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result);
        output
    }

    /// Generate a Merkle proof for a specific leaf.
    pub fn proof_for(&self, index: usize) -> Option<Vec<[u8; 32]>> {
        if index >= self.leaves.len() {
            return None;
        }

        let mut proof = Vec::new();
        let mut current_level = self.leaves.clone();
        let mut current_index = index;

        while current_level.len() > 1 {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            if sibling_index < current_level.len() {
                proof.push(current_level[sibling_index]);
            } else {
                proof.push(current_level[current_index]);
            }

            // Move to next level
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let hash = if chunk.len() == 2 {
                    Self::hash_pair(&chunk[0], &chunk[1])
                } else {
                    Self::hash_pair(&chunk[0], &chunk[0])
                };
                next_level.push(hash);
            }

            current_level = next_level;
            current_index /= 2;
        }

        Some(proof)
    }

    /// Verify a Merkle proof.
    pub fn verify_proof(
        leaf: &[u8; 32],
        proof: &[[u8; 32]],
        index: usize,
        root: &[u8; 32],
    ) -> bool {
        let mut current = *leaf;
        let mut current_index = index;

        for sibling in proof {
            current = if current_index % 2 == 0 {
                Self::hash_pair(&current, sibling)
            } else {
                Self::hash_pair(sibling, &current)
            };
            current_index /= 2;
        }

        current == *root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_chain_ids() {
        assert_eq!(Network::EthereumMainnet.chain_id(), Some(1));
        assert_eq!(Network::Polygon.chain_id(), Some(137));
        assert_eq!(Network::SolanaMainnet.chain_id(), None);
    }

    #[test]
    fn test_merkle_tree_single() {
        let leaf = [1u8; 32];
        let tree = AnchorMerkleTree {
            leaves: vec![leaf],
            root: leaf,
            height: 0,
        };
        assert_eq!(tree.root, leaf);
    }

    #[test]
    fn test_merkle_tree_verification() {
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| {
                let mut arr = [0u8; 32];
                arr[0] = i;
                arr
            })
            .collect();

        let root = AnchorMerkleTree::compute_root(&leaves);
        let tree = AnchorMerkleTree {
            leaves: leaves.clone(),
            root,
            height: 2,
        };

        // Verify proof for each leaf
        for (i, leaf) in leaves.iter().enumerate() {
            let proof = tree.proof_for(i).unwrap();
            assert!(AnchorMerkleTree::verify_proof(leaf, &proof, i, &root));
        }
    }
}
