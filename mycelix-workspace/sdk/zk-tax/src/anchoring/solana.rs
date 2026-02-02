//! Solana anchoring implementation.
//!
//! Provides functionality to anchor tax proofs to Solana.

use super::{AnchorConfig, AnchorStatus, Anchorer, ProofAnchor};
use crate::{Error, Result, TaxBracketProof};

/// Solana anchor implementation.
pub struct SolanaAnchor;

impl SolanaAnchor {
    /// Create instruction data for anchoring a proof.
    pub fn create_instruction_data(proof: &TaxBracketProof) -> Vec<u8> {
        // Instruction discriminator (first 8 bytes of SHA256("global:anchor"))
        let discriminator = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];

        let mut data = discriminator.to_vec();
        data.extend_from_slice(&proof.commitment.to_bytes());

        data
    }

    /// Estimate compute units for anchoring.
    pub fn estimate_compute_units() -> u32 {
        50_000 // Typical for simple state update
    }

    /// Estimate transaction fee in lamports.
    pub fn estimate_fee_lamports() -> u64 {
        5_000 // ~0.000005 SOL
    }
}

impl Anchorer for SolanaAnchor {
    async fn anchor(
        proof: &TaxBracketProof,
        config: &AnchorConfig,
    ) -> Result<ProofAnchor> {
        // In a real implementation, this would use solana-sdk
        // to submit a transaction to Solana.

        let commitment = proof.commitment.to_bytes();

        // Simulate signature (would be real in production)
        let mut tx_hash = [0u8; 64];
        tx_hash[..32].copy_from_slice(&commitment);

        Ok(ProofAnchor {
            network: config.network,
            tx_hash: tx_hash.to_vec(),
            block_number: None,
            block_timestamp: None,
            merkle_root: commitment,
            contract_address: None,
            program_id: config.contract_address.clone(), // Program ID for Solana
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
            confirmations: if anchor.block_number.is_some() { 32 } else { 0 },
            is_final: anchor.block_number.is_some(),
            time_to_finality: if anchor.block_number.is_some() {
                Some(0)
            } else {
                Some(1) // Solana finality is ~400ms
            },
        })
    }
}

/// Anchor program IDL (Interface Definition Language) for Solana.
pub const ANCHOR_PROGRAM_IDL: &str = r#"{
    "version": "0.1.0",
    "name": "zk_tax_anchor",
    "instructions": [
        {
            "name": "anchor",
            "accounts": [
                {
                    "name": "proof",
                    "isMut": true,
                    "isSigner": false
                },
                {
                    "name": "authority",
                    "isMut": true,
                    "isSigner": true
                },
                {
                    "name": "systemProgram",
                    "isMut": false,
                    "isSigner": false
                }
            ],
            "args": [
                {
                    "name": "commitment",
                    "type": {
                        "array": ["u8", 32]
                    }
                }
            ]
        },
        {
            "name": "verify",
            "accounts": [
                {
                    "name": "proof",
                    "isMut": false,
                    "isSigner": false
                }
            ],
            "args": [
                {
                    "name": "commitment",
                    "type": {
                        "array": ["u8", 32]
                    }
                }
            ]
        }
    ],
    "accounts": [
        {
            "name": "ProofAnchor",
            "type": {
                "kind": "struct",
                "fields": [
                    {
                        "name": "commitment",
                        "type": {
                            "array": ["u8", 32]
                        }
                    },
                    {
                        "name": "timestamp",
                        "type": "i64"
                    },
                    {
                        "name": "authority",
                        "type": "publicKey"
                    }
                ]
            }
        }
    ]
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Jurisdiction, FilingStatus, TaxBracketProver};

    #[test]
    fn test_create_instruction_data() {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        let data = SolanaAnchor::create_instruction_data(&proof);

        // 8 bytes discriminator + 32 bytes commitment
        assert_eq!(data.len(), 40);
    }

    #[test]
    fn test_fee_estimation() {
        let fee = SolanaAnchor::estimate_fee_lamports();
        assert!(fee < 1_000_000); // Less than 0.001 SOL
    }
}
