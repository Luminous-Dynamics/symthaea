//! Ethereum integration types
//!
//! Data structures for payment distribution and contract interactions.

use serde::{Deserialize, Serialize};

/// Payment split for a contributor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentSplit {
    /// Ethereum address
    pub address: String,
    /// Share in basis points (0-10000)
    pub basis_points: u64,
    /// Node/agent identifier (for tracing)
    pub node_id: String,
}

/// Payment distribution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentDistributionRequest {
    /// Model ID for payment routing
    pub model_id: String,
    /// FL round number
    pub round: u64,
    /// Total amount to distribute (in wei as string)
    pub total_amount_wei: String,
    /// Payment splits
    pub splits: Vec<PaymentSplit>,
    /// Platform fee in basis points
    pub platform_fee_bps: u64,
}

/// Result of payment distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentDistributionResult {
    /// Transaction hash
    pub tx_hash: String,
    /// Block number where included
    pub block_number: Option<u64>,
    /// Gas used
    pub gas_used: u64,
    /// Effective gas price
    pub effective_gas_price: u64,
    /// Status (success/failure)
    pub status: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Contract addresses for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractAddresses {
    /// PaymentRouter contract
    pub payment_router: String,
    /// ReputationAnchor contract
    pub reputation_anchor: String,
    /// ContributionRegistry contract
    pub contribution_registry: String,
    /// ModelRegistry contract
    pub model_registry: String,
    /// MycelixRegistry contract
    pub mycelix_registry: String,
}

impl Default for ContractAddresses {
    fn default() -> Self {
        Self {
            payment_router: "0x0000000000000000000000000000000000000000".to_string(),
            reputation_anchor: "0x0000000000000000000000000000000000000000".to_string(),
            contribution_registry: "0x0000000000000000000000000000000000000000".to_string(),
            model_registry: "0x0000000000000000000000000000000000000000".to_string(),
            mycelix_registry: "0x0000000000000000000000000000000000000000".to_string(),
        }
    }
}

/// Ethereum client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthereumConfig {
    /// RPC endpoint URL
    pub rpc_url: String,
    /// Chain ID
    pub chain_id: u64,
    /// Contract addresses
    pub contracts: ContractAddresses,
    /// Private key for signing (hex, no 0x prefix)
    /// SECURITY: In production, use HSM or secure key management
    pub private_key: Option<String>,
    /// Gas price multiplier (default: 1.1)
    pub gas_price_multiplier: f64,
    /// Maximum gas limit
    pub max_gas_limit: u64,
    /// Confirmation blocks to wait
    pub confirmations: u32,
    /// Transaction timeout in seconds
    pub tx_timeout_secs: u64,
}

impl Default for EthereumConfig {
    fn default() -> Self {
        Self {
            rpc_url: "http://localhost:8545".to_string(),
            chain_id: 1, // Ethereum mainnet
            contracts: ContractAddresses::default(),
            private_key: None,
            gas_price_multiplier: 1.1,
            max_gas_limit: 3_000_000,
            confirmations: 2,
            tx_timeout_secs: 120,
        }
    }
}

/// Anchor commitment for proof anchoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorCommitment {
    /// Proof hash
    pub proof_hash: String,
    /// Round number
    pub round: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Model ID
    pub model_id: String,
    /// Additional metadata
    pub metadata: Option<String>,
}

/// Result of anchoring operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorResult {
    /// Transaction hash
    pub tx_hash: String,
    /// Block number
    pub block_number: u64,
    /// Block hash
    pub block_hash: String,
    /// Timestamp
    pub timestamp: u64,
    /// Commitment ID (contract-assigned)
    pub commitment_id: Option<String>,
}

/// Reputation update for on-chain anchoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationUpdate {
    /// Agent/node identifier
    pub agent_id: String,
    /// Ethereum address
    pub eth_address: String,
    /// New reputation score (0-10000 basis points)
    pub score_bps: u64,
    /// Round number
    pub round: u64,
    /// Evidence hash
    pub evidence_hash: Option<String>,
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxStatus {
    /// Transaction pending in mempool
    Pending,
    /// Transaction included in block
    Included,
    /// Transaction confirmed (enough blocks)
    Confirmed,
    /// Transaction failed
    Failed,
    /// Transaction reverted
    Reverted,
    /// Transaction dropped
    Dropped,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EthereumConfig::default();
        assert_eq!(config.chain_id, 1);
        assert!(config.private_key.is_none());
    }

    #[test]
    fn test_payment_split_serialization() {
        let split = PaymentSplit {
            address: "0x742d35Cc6634C0532925a3b844Bc9e7595f8fF2B".to_string(),
            basis_points: 2500,
            node_id: "node-1".to_string(),
        };

        let json = serde_json::to_string(&split).unwrap();
        let deserialized: PaymentSplit = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.basis_points, 2500);
    }
}
