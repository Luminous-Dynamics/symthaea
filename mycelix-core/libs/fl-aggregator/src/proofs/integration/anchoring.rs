//! Cross-Chain Proof Anchoring
//!
//! Anchor proof commitments to blockchain networks for timestamping and
//! immutable verification.
//!
//! ## Features
//!
//! - Bitcoin OP_RETURN anchoring
//! - Ethereum event log anchoring
//! - Merkle root timestamping
//! - Proof-of-inclusion verification
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::anchoring::{
//!     Anchor, AnchorConfig, BitcoinAnchor, EthereumAnchor
//! };
//!
//! // Create anchor configuration
//! let config = AnchorConfig::bitcoin_testnet();
//!
//! // Anchor a merkle root
//! let anchor = BitcoinAnchor::new(config);
//! let receipt = anchor.anchor_commitment(&merkle_root).await?;
//!
//! // Verify anchoring
//! let verified = anchor.verify_anchor(&receipt).await?;
//! ```

use crate::proofs::{ProofError, ProofResult};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Blockchain network for anchoring
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Chain {
    /// Bitcoin mainnet
    BitcoinMainnet,
    /// Bitcoin testnet
    BitcoinTestnet,
    /// Ethereum mainnet
    EthereumMainnet,
    /// Ethereum Sepolia testnet
    EthereumSepolia,
    /// Polygon mainnet
    Polygon,
    /// Polygon Mumbai testnet
    PolygonMumbai,
    /// Custom chain
    Custom { chain_id: u64, name: String },
}

impl Chain {
    /// Get chain ID
    pub fn chain_id(&self) -> Option<u64> {
        match self {
            Chain::BitcoinMainnet => None,
            Chain::BitcoinTestnet => None,
            Chain::EthereumMainnet => Some(1),
            Chain::EthereumSepolia => Some(11155111),
            Chain::Polygon => Some(137),
            Chain::PolygonMumbai => Some(80001),
            Chain::Custom { chain_id, .. } => Some(*chain_id),
        }
    }

    /// Get chain name
    pub fn name(&self) -> &str {
        match self {
            Chain::BitcoinMainnet => "Bitcoin Mainnet",
            Chain::BitcoinTestnet => "Bitcoin Testnet",
            Chain::EthereumMainnet => "Ethereum Mainnet",
            Chain::EthereumSepolia => "Ethereum Sepolia",
            Chain::Polygon => "Polygon",
            Chain::PolygonMumbai => "Polygon Mumbai",
            Chain::Custom { name, .. } => name,
        }
    }
}

/// Anchor configuration
#[derive(Clone, Debug)]
pub struct AnchorConfig {
    /// Target blockchain
    pub chain: Chain,
    /// RPC endpoint URL
    pub rpc_url: String,
    /// API key (if required)
    pub api_key: Option<String>,
    /// Contract address (for EVM chains)
    pub contract_address: Option<String>,
}

impl AnchorConfig {
    /// Bitcoin testnet configuration
    pub fn bitcoin_testnet(rpc_url: &str) -> Self {
        Self {
            chain: Chain::BitcoinTestnet,
            rpc_url: rpc_url.to_string(),
            api_key: None,
            contract_address: None,
        }
    }

    /// Ethereum mainnet configuration
    pub fn ethereum_mainnet(rpc_url: &str, contract: &str) -> Self {
        Self {
            chain: Chain::EthereumMainnet,
            rpc_url: rpc_url.to_string(),
            api_key: None,
            contract_address: Some(contract.to_string()),
        }
    }

    /// Ethereum Sepolia testnet
    pub fn ethereum_sepolia(rpc_url: &str, contract: &str) -> Self {
        Self {
            chain: Chain::EthereumSepolia,
            rpc_url: rpc_url.to_string(),
            api_key: None,
            contract_address: Some(contract.to_string()),
        }
    }
}

/// Anchor receipt - proof of anchoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorReceipt {
    /// Commitment that was anchored
    pub commitment: [u8; 32],

    /// Chain where anchored
    pub chain: Chain,

    /// Transaction hash
    pub tx_hash: String,

    /// Block number (if confirmed)
    pub block_number: Option<u64>,

    /// Block hash (if confirmed)
    pub block_hash: Option<String>,

    /// Timestamp when anchored
    pub timestamp: u64,

    /// Number of confirmations
    pub confirmations: u64,

    /// Additional metadata
    pub metadata: Option<AnchorMetadata>,
}

/// Additional anchor metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorMetadata {
    /// Gas used (EVM chains)
    pub gas_used: Option<u64>,
    /// Transaction fee
    pub fee: Option<String>,
    /// Contract event log index
    pub log_index: Option<u64>,
    /// Proof composition ID
    pub composition_id: Option<String>,
}

/// Anchoring status
#[derive(Clone, Debug, PartialEq)]
pub enum AnchorStatus {
    /// Transaction submitted, waiting for confirmation
    Pending,
    /// Confirmed with N confirmations
    Confirmed(u64),
    /// Transaction failed
    Failed(String),
}

/// Trait for blockchain anchoring implementations
#[async_trait::async_trait]
pub trait Anchor: Send + Sync {
    /// Anchor a 32-byte commitment
    async fn anchor_commitment(&self, commitment: &[u8; 32]) -> ProofResult<AnchorReceipt>;

    /// Verify an anchor receipt
    async fn verify_anchor(&self, receipt: &AnchorReceipt) -> ProofResult<bool>;

    /// Get anchor status
    async fn get_status(&self, tx_hash: &str) -> ProofResult<AnchorStatus>;

    /// Get minimum confirmations required
    fn min_confirmations(&self) -> u64;
}

/// Bitcoin anchoring via OP_RETURN
pub struct BitcoinAnchor {
    config: AnchorConfig,
}

impl BitcoinAnchor {
    pub fn new(config: AnchorConfig) -> Self {
        Self { config }
    }

    /// Create OP_RETURN script
    fn create_op_return_script(commitment: &[u8; 32]) -> Vec<u8> {
        let mut script = Vec::with_capacity(34);
        script.push(0x6a); // OP_RETURN
        script.push(0x20); // Push 32 bytes
        script.extend_from_slice(commitment);
        script
    }
}

#[async_trait::async_trait]
impl Anchor for BitcoinAnchor {
    async fn anchor_commitment(&self, commitment: &[u8; 32]) -> ProofResult<AnchorReceipt> {
        // In production, this would:
        // 1. Create transaction with OP_RETURN output
        // 2. Sign with wallet
        // 3. Broadcast to network

        #[cfg(feature = "proofs-anchoring")]
        {
            // Placeholder - would use bitcoin RPC
            let tx_hash = format!("btc_{}", hex::encode(&commitment[..8]));

            Ok(AnchorReceipt {
                commitment: *commitment,
                chain: self.config.chain.clone(),
                tx_hash,
                block_number: None,
                block_hash: None,
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                confirmations: 0,
                metadata: None,
            })
        }

        #[cfg(not(feature = "proofs-anchoring"))]
        Err(ProofError::GenerationFailed("proofs-anchoring feature not enabled".to_string()))
    }

    async fn verify_anchor(&self, receipt: &AnchorReceipt) -> ProofResult<bool> {
        // In production, this would:
        // 1. Fetch transaction from blockchain
        // 2. Parse OP_RETURN output
        // 3. Compare commitment
        // 4. Check confirmations

        #[cfg(feature = "proofs-anchoring")]
        {
            Ok(receipt.confirmations >= self.min_confirmations())
        }

        #[cfg(not(feature = "proofs-anchoring"))]
        Err(ProofError::VerificationFailed("proofs-anchoring feature not enabled".to_string()))
    }

    async fn get_status(&self, _tx_hash: &str) -> ProofResult<AnchorStatus> {
        // Would query blockchain for transaction status
        Ok(AnchorStatus::Pending)
    }

    fn min_confirmations(&self) -> u64 {
        match self.config.chain {
            Chain::BitcoinMainnet => 6,
            Chain::BitcoinTestnet => 1,
            _ => 6,
        }
    }
}

/// Ethereum anchoring via contract events
pub struct EthereumAnchor {
    config: AnchorConfig,
}

impl EthereumAnchor {
    pub fn new(config: AnchorConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Anchor for EthereumAnchor {
    async fn anchor_commitment(&self, commitment: &[u8; 32]) -> ProofResult<AnchorReceipt> {
        #[cfg(feature = "proofs-anchoring")]
        {
            let _contract = self.config.contract_address.as_ref()
                .ok_or_else(|| ProofError::InvalidPublicInputs("Contract address required".to_string()))?;

            // In production, this would:
            // 1. Encode function call (e.g., anchor(bytes32))
            // 2. Sign transaction with ethers-rs or similar
            // 3. Send via eth_sendRawTransaction

            let tx_hash = format!("0x{}", hex::encode(&commitment[..16]));

            Ok(AnchorReceipt {
                commitment: *commitment,
                chain: self.config.chain.clone(),
                tx_hash,
                block_number: None,
                block_hash: None,
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                confirmations: 0,
                metadata: Some(AnchorMetadata {
                    gas_used: None,
                    fee: None,
                    log_index: None,
                    composition_id: None,
                }),
            })
        }

        #[cfg(not(feature = "proofs-anchoring"))]
        Err(ProofError::GenerationFailed("proofs-anchoring feature not enabled".to_string()))
    }

    async fn verify_anchor(&self, receipt: &AnchorReceipt) -> ProofResult<bool> {
        #[cfg(feature = "proofs-anchoring")]
        {
            // Would:
            // 1. Fetch transaction receipt
            // 2. Parse event logs
            // 3. Verify commitment matches
            Ok(receipt.confirmations >= self.min_confirmations())
        }

        #[cfg(not(feature = "proofs-anchoring"))]
        Err(ProofError::VerificationFailed("proofs-anchoring feature not enabled".to_string()))
    }

    async fn get_status(&self, _tx_hash: &str) -> ProofResult<AnchorStatus> {
        Ok(AnchorStatus::Pending)
    }

    fn min_confirmations(&self) -> u64 {
        match self.config.chain {
            Chain::EthereumMainnet => 12,
            Chain::EthereumSepolia => 1,
            Chain::Polygon => 128,
            Chain::PolygonMumbai => 1,
            _ => 12,
        }
    }
}

/// Anchor a composed proof's merkle root
pub async fn anchor_composition(
    merkle_root: &[u8; 32],
    anchor: &dyn Anchor,
) -> ProofResult<AnchorReceipt> {
    anchor.anchor_commitment(merkle_root).await
}

/// Verify a composition was anchored
pub async fn verify_composition_anchor(
    receipt: &AnchorReceipt,
    anchor: &dyn Anchor,
) -> ProofResult<bool> {
    anchor.verify_anchor(receipt).await
}

/// Batch anchor multiple commitments (Merkle tree of commitments)
pub fn create_anchor_merkle_root(commitments: &[[u8; 32]]) -> [u8; 32] {
    use blake3::Hasher;

    if commitments.is_empty() {
        return [0u8; 32];
    }

    if commitments.len() == 1 {
        return commitments[0];
    }

    // Build Merkle tree
    let mut current_level: Vec<[u8; 32]> = commitments.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(2) {
            let mut hasher = Hasher::new();
            hasher.update(&chunk[0]);
            if chunk.len() > 1 {
                hasher.update(&chunk[1]);
            } else {
                hasher.update(&chunk[0]); // Duplicate last element
            }
            next_level.push(*hasher.finalize().as_bytes());
        }

        current_level = next_level;
    }

    current_level[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_info() {
        assert_eq!(Chain::EthereumMainnet.chain_id(), Some(1));
        assert_eq!(Chain::BitcoinMainnet.chain_id(), None);
        assert_eq!(Chain::EthereumMainnet.name(), "Ethereum Mainnet");
    }

    #[test]
    fn test_op_return_script() {
        let commitment = [0x42u8; 32];
        let script = BitcoinAnchor::create_op_return_script(&commitment);

        assert_eq!(script.len(), 34);
        assert_eq!(script[0], 0x6a); // OP_RETURN
        assert_eq!(script[1], 0x20); // 32 bytes
        assert_eq!(&script[2..], &commitment);
    }

    #[test]
    fn test_anchor_merkle_root() {
        let commitments = [
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            [4u8; 32],
        ];

        let root = create_anchor_merkle_root(&commitments);
        assert_ne!(root, [0u8; 32]);

        // Same commitments should give same root
        let root2 = create_anchor_merkle_root(&commitments);
        assert_eq!(root, root2);

        // Different commitments should give different root
        let different = [[5u8; 32]; 4];
        let root3 = create_anchor_merkle_root(&different);
        assert_ne!(root, root3);
    }

    #[test]
    fn test_single_commitment_anchor() {
        let commitment = [0x42u8; 32];
        let root = create_anchor_merkle_root(&[commitment]);
        assert_eq!(root, commitment);
    }

    #[test]
    fn test_empty_anchor() {
        let root = create_anchor_merkle_root(&[]);
        assert_eq!(root, [0u8; 32]);
    }
}

// ============================================================================
// Proof-Specific Anchoring
// ============================================================================

use crate::proofs::{
    RangeProof, GradientIntegrityProof, IdentityAssuranceProof,
    VoteEligibilityProof, MembershipProof, ProofType,
    recursive::{RecursiveProof, ProofCommitment},
};

/// Trait for anchoring proofs
#[async_trait::async_trait]
pub trait AnchorableProof {
    /// Get the proof commitment (hash) for anchoring
    fn anchor_commitment(&self) -> [u8; 32];

    /// Get the proof type
    fn proof_type(&self) -> ProofType;

    /// Anchor this proof to a blockchain
    async fn anchor(&self, anchor: &dyn Anchor) -> ProofResult<AnchoredProof>;
}

/// An anchored proof with blockchain receipt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchoredProof {
    /// Proof commitment (hash)
    pub commitment: [u8; 32],
    /// Proof type
    pub proof_type: ProofType,
    /// Anchor receipt
    pub receipt: AnchorReceipt,
    /// Original proof bytes (optional - for full verification)
    pub proof_bytes: Option<Vec<u8>>,
}

impl AnchoredProof {
    /// Verify the anchoring is valid
    pub async fn verify_anchoring(&self, anchor: &dyn Anchor) -> ProofResult<bool> {
        // Verify the anchor receipt
        if !anchor.verify_anchor(&self.receipt).await? {
            return Ok(false);
        }

        // Verify commitment matches
        Ok(self.commitment == self.receipt.commitment)
    }

    /// Get the number of confirmations
    pub fn confirmations(&self) -> u64 {
        self.receipt.confirmations
    }

    /// Check if sufficiently confirmed
    pub fn is_confirmed(&self, min_confirmations: u64) -> bool {
        self.receipt.confirmations >= min_confirmations
    }
}

// Implement AnchorableProof for each proof type
macro_rules! impl_anchorable_proof {
    ($proof_type:ty, $variant:ident) => {
        #[async_trait::async_trait]
        impl AnchorableProof for $proof_type {
            fn anchor_commitment(&self) -> [u8; 32] {
                use sha2::{Sha256, Digest};
                let bytes = self.to_bytes();
                Sha256::digest(&bytes).into()
            }

            fn proof_type(&self) -> ProofType {
                ProofType::$variant
            }

            async fn anchor(&self, anchor: &dyn Anchor) -> ProofResult<AnchoredProof> {
                let commitment = self.anchor_commitment();
                let receipt = anchor.anchor_commitment(&commitment).await?;

                Ok(AnchoredProof {
                    commitment,
                    proof_type: self.proof_type(),
                    receipt,
                    proof_bytes: Some(self.to_bytes()),
                })
            }
        }
    };
}

impl_anchorable_proof!(RangeProof, Range);
impl_anchorable_proof!(GradientIntegrityProof, GradientIntegrity);
impl_anchorable_proof!(IdentityAssuranceProof, IdentityAssurance);
impl_anchorable_proof!(VoteEligibilityProof, VoteEligibility);
impl_anchorable_proof!(MembershipProof, Membership);

/// Anchor a recursive/aggregated proof
#[async_trait::async_trait]
impl AnchorableProof for RecursiveProof {
    fn anchor_commitment(&self) -> [u8; 32] {
        self.root
    }

    fn proof_type(&self) -> ProofType {
        // Recursive proofs are a meta-type covering multiple proofs
        ProofType::Range // Default, actual types are in the batch
    }

    async fn anchor(&self, anchor: &dyn Anchor) -> ProofResult<AnchoredProof> {
        let commitment = self.anchor_commitment();
        let receipt = anchor.anchor_commitment(&commitment).await?;

        Ok(AnchoredProof {
            commitment,
            proof_type: ProofType::Range, // Meta-type
            receipt,
            proof_bytes: self.to_bytes().ok(),
        })
    }
}

/// Batch anchoring result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchAnchorResult {
    /// Individual proof commitments
    pub commitments: Vec<ProofCommitment>,
    /// Combined Merkle root
    pub merkle_root: [u8; 32],
    /// Anchor receipt for the root
    pub receipt: AnchorReceipt,
    /// Number of proofs anchored
    pub count: usize,
}

/// Anchor multiple proofs with a single transaction
pub async fn anchor_proof_batch(
    commitments: &[ProofCommitment],
    anchor: &dyn Anchor,
) -> ProofResult<BatchAnchorResult> {
    if commitments.is_empty() {
        return Err(ProofError::InvalidInput("Empty batch".to_string()));
    }

    // Create Merkle root of all commitments
    let hashes: Vec<[u8; 32]> = commitments.iter().map(|c| c.hash).collect();
    let merkle_root = create_anchor_merkle_root(&hashes);

    // Anchor the root
    let receipt = anchor.anchor_commitment(&merkle_root).await?;

    Ok(BatchAnchorResult {
        commitments: commitments.to_vec(),
        merkle_root,
        receipt,
        count: commitments.len(),
    })
}

/// Verify a batch anchor
pub async fn verify_batch_anchor(
    result: &BatchAnchorResult,
    anchor: &dyn Anchor,
) -> ProofResult<bool> {
    // Verify the anchor
    if !anchor.verify_anchor(&result.receipt).await? {
        return Ok(false);
    }

    // Verify Merkle root matches
    let hashes: Vec<[u8; 32]> = result.commitments.iter().map(|c| c.hash).collect();
    let computed_root = create_anchor_merkle_root(&hashes);

    Ok(computed_root == result.merkle_root && result.merkle_root == result.receipt.commitment)
}

/// Check if a commitment is included in a batch anchor
pub fn verify_commitment_inclusion(
    commitment: &ProofCommitment,
    result: &BatchAnchorResult,
) -> bool {
    result.commitments.iter().any(|c| c.hash == commitment.hash)
}

#[cfg(test)]
mod proof_anchoring_tests {
    use super::*;
    use crate::proofs::{ProofConfig, SecurityLevel};

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_proof_commitment() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let commitment = proof.anchor_commitment();

        assert_ne!(commitment, [0u8; 32]);

        // Same proof should give same commitment
        let commitment2 = proof.anchor_commitment();
        assert_eq!(commitment, commitment2);
    }

    #[test]
    fn test_anchored_proof_creation() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let commitment = proof.anchor_commitment();

        let receipt = AnchorReceipt {
            commitment,
            chain: Chain::EthereumSepolia,
            tx_hash: "0x123".to_string(),
            block_number: Some(12345),
            block_hash: Some("0xabc".to_string()),
            timestamp: 1000000,
            confirmations: 10,
            metadata: None,
        };

        let anchored = AnchoredProof {
            commitment,
            proof_type: ProofType::Range,
            receipt,
            proof_bytes: Some(proof.to_bytes()),
        };

        assert!(anchored.is_confirmed(5));
        assert!(!anchored.is_confirmed(15));
    }

    #[test]
    fn test_batch_commitment_inclusion() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();

        let commitment1 = ProofCommitment {
            hash: proof1.anchor_commitment(),
            proof_type: ProofType::Range,
            original_size: proof1.to_bytes().len(),
        };

        let commitment2 = ProofCommitment {
            hash: proof2.anchor_commitment(),
            proof_type: ProofType::Range,
            original_size: proof2.to_bytes().len(),
        };

        let hashes = vec![commitment1.hash, commitment2.hash];
        let merkle_root = create_anchor_merkle_root(&hashes);

        let result = BatchAnchorResult {
            commitments: vec![commitment1.clone(), commitment2.clone()],
            merkle_root,
            receipt: AnchorReceipt {
                commitment: merkle_root,
                chain: Chain::EthereumSepolia,
                tx_hash: "0x456".to_string(),
                block_number: Some(12346),
                block_hash: None,
                timestamp: 1000001,
                confirmations: 5,
                metadata: None,
            },
            count: 2,
        };

        assert!(verify_commitment_inclusion(&commitment1, &result));
        assert!(verify_commitment_inclusion(&commitment2, &result));

        // Non-existent commitment
        let fake = ProofCommitment {
            hash: [99u8; 32],
            proof_type: ProofType::Range,
            original_size: 0,
        };
        assert!(!verify_commitment_inclusion(&fake, &result));
    }
}
