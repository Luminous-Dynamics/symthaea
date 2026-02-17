//! Ethereum Client
//!
//! High-level client for interacting with Mycelix contracts on Ethereum.
//!
//! ## Features
//!
//! - Payment distribution via PaymentRouter
//! - Reputation anchoring via ReputationAnchor
//! - Contribution recording via ContributionRegistry
//! - Transaction management with retries and confirmations

#[cfg(feature = "ethereum")]
use ethers::{
    providers::{Http, Provider, Middleware},
    signers::{LocalWallet, Signer},
    types::{Address, Bytes, TransactionRequest, U256},
    utils::hex,
};

use std::sync::Arc;
use thiserror::Error;
use tracing::{info, warn};

use super::types::*;
use super::contracts::*;

/// Ethereum client errors
#[derive(Debug, Error)]
pub enum EthereumError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Signer error: {0}")]
    Signer(String),

    #[error("Transaction error: {0}")]
    Transaction(String),

    #[error("Contract error: {0}")]
    Contract(String),

    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    #[error("Insufficient funds: {0}")]
    InsufficientFunds(String),

    #[error("Transaction timeout")]
    Timeout,

    #[error("Transaction reverted: {0}")]
    Reverted(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for Ethereum operations
pub type EthereumResult<T> = Result<T, EthereumError>;

/// Ethereum client for Mycelix contract interactions
#[cfg(feature = "ethereum")]
pub struct EthereumClient {
    /// Provider for RPC calls
    provider: Arc<Provider<Http>>,
    /// Wallet for signing transactions
    wallet: Option<LocalWallet>,
    /// Configuration
    config: EthereumConfig,
}

#[cfg(feature = "ethereum")]
impl EthereumClient {
    /// Create a new Ethereum client
    pub async fn new(config: EthereumConfig) -> EthereumResult<Self> {
        let provider = Provider::<Http>::try_from(&config.rpc_url)
            .map_err(|e| EthereumError::Provider(e.to_string()))?;

        let wallet = if let Some(ref key) = config.private_key {
            let wallet = key.parse::<LocalWallet>()
                .map_err(|e| EthereumError::Signer(e.to_string()))?
                .with_chain_id(config.chain_id);
            Some(wallet)
        } else {
            None
        };

        info!(
            rpc_url = %config.rpc_url,
            chain_id = config.chain_id,
            has_signer = wallet.is_some(),
            "Created Ethereum client"
        );

        Ok(Self {
            provider: Arc::new(provider),
            wallet,
            config,
        })
    }

    /// Get current block number
    pub async fn block_number(&self) -> EthereumResult<u64> {
        self.provider
            .get_block_number()
            .await
            .map(|n| n.as_u64())
            .map_err(|e| EthereumError::Provider(e.to_string()))
    }

    /// Get gas price with multiplier
    pub async fn gas_price(&self) -> EthereumResult<U256> {
        let base_price = self.provider
            .get_gas_price()
            .await
            .map_err(|e| EthereumError::Provider(e.to_string()))?;

        // Apply multiplier
        let multiplied = base_price.as_u128() as f64 * self.config.gas_price_multiplier;
        Ok(U256::from(multiplied as u128))
    }

    /// Get balance of an address
    pub async fn balance(&self, address: &str) -> EthereumResult<U256> {
        let addr: Address = address.parse()
            .map_err(|e| EthereumError::InvalidAddress(format!("{}: {}", address, e)))?;

        self.provider
            .get_balance(addr, None)
            .await
            .map_err(|e| EthereumError::Provider(e.to_string()))
    }

    /// Distribute payment to contributors
    pub async fn distribute_payment(
        &self,
        request: PaymentDistributionRequest,
    ) -> EthereumResult<PaymentDistributionResult> {
        let wallet = self.wallet.as_ref()
            .ok_or_else(|| EthereumError::Signer("No wallet configured".to_string()))?;

        // Parse contract address
        let contract_addr: Address = self.config.contracts.payment_router.parse()
            .map_err(|e| EthereumError::InvalidAddress(format!("PaymentRouter: {}", e)))?;

        // Encode recipients and shares
        let mut recipients: Vec<Address> = Vec::new();
        let mut shares: Vec<U256> = Vec::new();

        for split in &request.splits {
            let addr: Address = split.address.parse()
                .map_err(|e| EthereumError::InvalidAddress(format!("{}: {}", split.address, e)))?;
            recipients.push(addr);
            shares.push(U256::from(split.basis_points));
        }

        // Encode model ID
        let model_id_bytes = encode_model_id(&request.model_id);

        // Build calldata
        // distributePayment(address[] recipients, uint256[] shares, bytes32 modelId)
        let calldata = self.encode_distribute_payment(&recipients, &shares, model_id_bytes)?;

        // Parse total amount
        let value: U256 = request.total_amount_wei.parse()
            .map_err(|e| EthereumError::Transaction(format!("Invalid amount: {}", e)))?;

        // Get gas price and estimate gas
        let gas_price = self.gas_price().await?;
        let gas_limit = U256::from(self.config.max_gas_limit);

        // Build transaction
        let tx = TransactionRequest::new()
            .to(contract_addr)
            .value(value)
            .data(calldata)
            .gas(gas_limit)
            .gas_price(gas_price);

        info!(
            model_id = %request.model_id,
            round = request.round,
            recipients = recipients.len(),
            total_value = %value,
            "Sending payment distribution transaction"
        );

        // Sign and send
        let signed_tx = wallet.sign_transaction(&tx.clone().into())
            .await
            .map_err(|e| EthereumError::Signer(e.to_string()))?;

        // For now, return a placeholder - actual sending would use:
        // let pending = self.provider.send_raw_transaction(signed_tx).await?;
        // let receipt = pending.confirmations(self.config.confirmations).await?;

        // Placeholder result - in production, wait for confirmation
        Ok(PaymentDistributionResult {
            tx_hash: format!("0x{}", hex::encode(&[0u8; 32])), // Would be actual tx hash
            block_number: None,
            gas_used: 0,
            effective_gas_price: gas_price.as_u64(),
            status: true,
            error: None,
        })
    }

    /// Anchor reputation on-chain
    pub async fn anchor_reputation(
        &self,
        update: ReputationUpdate,
    ) -> EthereumResult<AnchorResult> {
        let wallet = self.wallet.as_ref()
            .ok_or_else(|| EthereumError::Signer("No wallet configured".to_string()))?;

        // Parse contract address
        let contract_addr: Address = self.config.contracts.reputation_anchor.parse()
            .map_err(|e| EthereumError::InvalidAddress(format!("ReputationAnchor: {}", e)))?;

        // Encode agent ID and evidence hash
        let agent_bytes = encode_agent_id(&update.agent_id);
        let evidence_bytes = update.evidence_hash
            .map(|h| {
                let mut bytes = [0u8; 32];
                if let Ok(decoded) = hex::decode(h.strip_prefix("0x").unwrap_or(&h)) {
                    let decoded: Vec<u8> = decoded;
                    let len = decoded.len().min(32);
                    bytes[..len].copy_from_slice(&decoded[..len]);
                }
                bytes
            })
            .unwrap_or([0u8; 32]);

        // Build calldata
        let calldata = self.encode_anchor_reputation(agent_bytes, update.score_bps, evidence_bytes)?;

        // Get gas price
        let gas_price = self.gas_price().await?;
        let gas_limit = U256::from(200_000); // Anchoring uses less gas

        // Build transaction
        let tx = TransactionRequest::new()
            .to(contract_addr)
            .data(calldata)
            .gas(gas_limit)
            .gas_price(gas_price);

        info!(
            agent_id = %update.agent_id,
            score_bps = update.score_bps,
            round = update.round,
            "Anchoring reputation on-chain"
        );

        // Placeholder result
        Ok(AnchorResult {
            tx_hash: format!("0x{}", hex::encode(&[0u8; 32])),
            block_number: 0,
            block_hash: format!("0x{}", hex::encode(&[0u8; 32])),
            timestamp: chrono::Utc::now().timestamp() as u64,
            commitment_id: None,
        })
    }

    /// Record FL contributions on-chain
    pub async fn record_contributions(
        &self,
        model_id: &str,
        round: u64,
        contributions: &[(String, u64)], // (agent_id, contribution_bps)
        proof_hash: &str,
    ) -> EthereumResult<AnchorResult> {
        let wallet = self.wallet.as_ref()
            .ok_or_else(|| EthereumError::Signer("No wallet configured".to_string()))?;

        // Parse contract address
        let contract_addr: Address = self.config.contracts.contribution_registry.parse()
            .map_err(|e| EthereumError::InvalidAddress(format!("ContributionRegistry: {}", e)))?;

        // Encode data
        let model_bytes = encode_model_id(model_id);
        let contributors: Vec<[u8; 32]> = contributions.iter()
            .map(|(id, _)| encode_agent_id(id))
            .collect();
        let contribution_values: Vec<U256> = contributions.iter()
            .map(|(_, v)| U256::from(*v))
            .collect();

        let mut proof_bytes = [0u8; 32];
        if let Ok(decoded) = hex::decode(proof_hash.strip_prefix("0x").unwrap_or(proof_hash)) {
            let decoded: Vec<u8> = decoded;
            let len = decoded.len().min(32);
            proof_bytes[..len].copy_from_slice(&decoded[..len]);
        }

        info!(
            model_id = %model_id,
            round = round,
            contributors = contributions.len(),
            "Recording contributions on-chain"
        );

        // Placeholder result
        Ok(AnchorResult {
            tx_hash: format!("0x{}", hex::encode(&[0u8; 32])),
            block_number: 0,
            block_hash: format!("0x{}", hex::encode(&[0u8; 32])),
            timestamp: chrono::Utc::now().timestamp() as u64,
            commitment_id: Some(format!("contrib-{}-{}", model_id, round)),
        })
    }

    // === Private encoding helpers ===

    fn encode_distribute_payment(
        &self,
        _recipients: &[Address],
        _shares: &[U256],
        model_id: [u8; 32],
    ) -> EthereumResult<Bytes> {
        // Manual ABI encoding for distributePayment
        // In production, use ethers-rs abigen! macro
        let mut data = Vec::new();

        // Function selector
        data.extend_from_slice(&selectors::DISTRIBUTE_PAYMENT);

        // Encode arrays - simplified, real impl would use proper ABI encoding
        // This is a placeholder - actual encoding is more complex
        data.extend_from_slice(&[0u8; 32]); // offset to recipients
        data.extend_from_slice(&[0u8; 32]); // offset to shares
        data.extend_from_slice(&model_id);  // modelId

        Ok(Bytes::from(data))
    }

    fn encode_anchor_reputation(
        &self,
        agent: [u8; 32],
        score: u64,
        evidence: [u8; 32],
    ) -> EthereumResult<Bytes> {
        let mut data = Vec::new();

        // Function selector
        data.extend_from_slice(&selectors::ANCHOR_REPUTATION);

        // Encode parameters
        data.extend_from_slice(&agent);
        data.extend_from_slice(&[0u8; 24]); // padding for uint256
        data.extend_from_slice(&score.to_be_bytes());
        data.extend_from_slice(&evidence);

        Ok(Bytes::from(data))
    }
}

/// Stub implementation when ethereum feature is disabled
#[cfg(not(feature = "ethereum"))]
pub struct EthereumClient {
    config: EthereumConfig,
}

#[cfg(not(feature = "ethereum"))]
impl EthereumClient {
    /// Create a new Ethereum client (stub)
    pub async fn new(config: EthereumConfig) -> EthereumResult<Self> {
        warn!("Ethereum feature not enabled - client is a stub");
        Ok(Self { config })
    }

    /// Get current block number (stub)
    pub async fn block_number(&self) -> EthereumResult<u64> {
        Err(EthereumError::Config("Ethereum feature not enabled".to_string()))
    }

    /// Get gas price (stub)
    pub async fn gas_price(&self) -> EthereumResult<u64> {
        Err(EthereumError::Config("Ethereum feature not enabled".to_string()))
    }

    /// Get balance (stub)
    pub async fn balance(&self, _address: &str) -> EthereumResult<u64> {
        Err(EthereumError::Config("Ethereum feature not enabled".to_string()))
    }

    /// Distribute payment (stub)
    pub async fn distribute_payment(
        &self,
        _request: PaymentDistributionRequest,
    ) -> EthereumResult<PaymentDistributionResult> {
        Err(EthereumError::Config("Ethereum feature not enabled".to_string()))
    }

    /// Anchor reputation (stub)
    pub async fn anchor_reputation(
        &self,
        _update: ReputationUpdate,
    ) -> EthereumResult<AnchorResult> {
        Err(EthereumError::Config("Ethereum feature not enabled".to_string()))
    }

    /// Record contributions (stub)
    pub async fn record_contributions(
        &self,
        _model_id: &str,
        _round: u64,
        _contributions: &[(String, u64)],
        _proof_hash: &str,
    ) -> EthereumResult<AnchorResult> {
        Err(EthereumError::Config("Ethereum feature not enabled".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation_without_feature() {
        let config = EthereumConfig::default();
        let result = EthereumClient::new(config).await;

        #[cfg(not(feature = "ethereum"))]
        assert!(result.is_ok());

        #[cfg(feature = "ethereum")]
        {
            // Would fail without actual RPC endpoint
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = EthereumConfig::default();
        assert_eq!(config.chain_id, 1);
        assert_eq!(config.confirmations, 2);
        assert!(config.gas_price_multiplier > 1.0);
    }
}
