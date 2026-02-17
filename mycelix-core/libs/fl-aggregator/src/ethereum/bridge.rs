//! Ethereum Bridge Handler
//!
//! Processes Holochain signals for Ethereum transactions and calls
//! back into Holochain to update intent status.
//!
//! ## Architecture
//!
//! ```text
//! Holochain Zome                    Native Host
//! ┌──────────────┐                 ┌──────────────────────────────┐
//! │ Bridge Zome  │ ──signal──────▶ │  EthereumBridgeHandler       │
//! │              │                 │        │                      │
//! │  PaymentIntent ◀──zome_call── │   EthereumClient              │
//! │  update_status │              │        │                      │
//! └──────────────┘                 │   Ethereum RPC               │
//!                                  └──────────────────────────────┘
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[cfg(feature = "holochain")]
use crate::holochain::{HolochainClient, HolochainSignal, PaymentSplitSignal};

use super::client::EthereumClient;
use super::types::{PaymentDistributionRequest, PaymentSplit, ReputationUpdate};

/// Result type for bridge operations.
pub type BridgeResult<T> = Result<T, BridgeError>;

/// Bridge handler errors.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Ethereum client error: {0}")]
    Ethereum(String),

    #[error("Holochain client error: {0}")]
    Holochain(String),

    #[error("Invalid signal data: {0}")]
    InvalidSignal(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Configuration for the Ethereum bridge handler.
#[derive(Clone, Debug)]
pub struct BridgeHandlerConfig {
    /// Enable automatic status updates back to Holochain.
    pub enable_status_callback: bool,

    /// Retry failed transactions.
    pub retry_failed: bool,

    /// Maximum retry attempts.
    pub max_retries: u32,

    /// Retry delay in milliseconds.
    pub retry_delay_ms: u64,
}

impl Default for BridgeHandlerConfig {
    fn default() -> Self {
        Self {
            enable_status_callback: true,
            retry_failed: true,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Handler for processing Ethereum bridge signals from Holochain.
///
/// This handler receives signals from the Holochain conductor,
/// processes them using the EthereumClient, and calls back into
/// Holochain to update intent status.
#[cfg(all(feature = "ethereum", feature = "holochain"))]
pub struct EthereumBridgeHandler {
    /// Ethereum client for transactions.
    ethereum_client: Arc<EthereumClient>,

    /// Holochain client for status callbacks.
    holochain_client: Arc<RwLock<HolochainClient>>,

    /// Configuration.
    config: BridgeHandlerConfig,
}

#[cfg(all(feature = "ethereum", feature = "holochain"))]
impl EthereumBridgeHandler {
    /// Create a new bridge handler.
    pub fn new(
        ethereum_client: EthereumClient,
        holochain_client: HolochainClient,
        config: BridgeHandlerConfig,
    ) -> Self {
        Self {
            ethereum_client: Arc::new(ethereum_client),
            holochain_client: Arc::new(RwLock::new(holochain_client)),
            config,
        }
    }

    /// Process a signal from Holochain.
    pub async fn handle_signal(&self, signal: HolochainSignal) -> BridgeResult<()> {
        match signal {
            HolochainSignal::EthereumPaymentRequest {
                intent_id,
                model_id,
                round,
                total_amount_wei,
                splits,
            } => {
                self.handle_payment_request(intent_id, model_id, round, total_amount_wei, splits)
                    .await
            }
            HolochainSignal::EthereumAnchorRequest {
                intent_id,
                anchor_type,
                agent_id,
                score_bps,
                round,
                evidence_hash,
            } => {
                self.handle_anchor_request(
                    intent_id,
                    anchor_type,
                    agent_id,
                    score_bps,
                    round,
                    evidence_hash,
                )
                .await
            }
            _ => {
                // Not an Ethereum signal, ignore
                Ok(())
            }
        }
    }

    /// Handle a payment distribution request.
    async fn handle_payment_request(
        &self,
        intent_id: String,
        model_id: String,
        round: u64,
        total_amount_wei: String,
        splits: Vec<PaymentSplitSignal>,
    ) -> BridgeResult<()> {
        info!(
            intent_id = %intent_id,
            model_id = %model_id,
            round = round,
            splits = splits.len(),
            "Processing Ethereum payment request"
        );

        // Convert signal splits to client request format
        let payment_splits: Vec<PaymentSplit> = splits
            .into_iter()
            .map(|s| PaymentSplit {
                address: s.address,
                basis_points: s.basis_points,
                node_id: s.agent_id,
            })
            .collect();

        let request = PaymentDistributionRequest {
            model_id: model_id.clone(),
            round,
            total_amount_wei: total_amount_wei.clone(),
            splits: payment_splits,
            platform_fee_bps: 0, // No platform fee in default configuration
        };

        // Attempt to distribute payment
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= self.config.max_retries {
            match self.ethereum_client.distribute_payment(request.clone()).await {
                Ok(result) => {
                    info!(
                        intent_id = %intent_id,
                        tx_hash = %result.tx_hash,
                        gas_used = result.gas_used,
                        "Payment distribution successful"
                    );

                    // Update intent status in Holochain
                    if self.config.enable_status_callback {
                        self.update_payment_status(
                            &intent_id,
                            "confirmed",
                            Some(result.tx_hash.clone()),
                            result.block_number,
                            None,
                        )
                        .await?;
                    }

                    return Ok(());
                }
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e.to_string());

                    if attempts <= self.config.max_retries && self.config.retry_failed {
                        warn!(
                            intent_id = %intent_id,
                            attempt = attempts,
                            error = %last_error.as_ref().unwrap(),
                            "Payment distribution failed, retrying"
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            self.config.retry_delay_ms * attempts as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        // All retries exhausted
        let error_msg = last_error.unwrap_or_else(|| "Unknown error".to_string());
        error!(
            intent_id = %intent_id,
            error = %error_msg,
            "Payment distribution failed after retries"
        );

        // Update status to failed
        if self.config.enable_status_callback {
            self.update_payment_status(&intent_id, "failed", None, None, Some(error_msg.clone()))
                .await?;
        }

        Err(BridgeError::Ethereum(error_msg))
    }

    /// Handle an anchor request.
    async fn handle_anchor_request(
        &self,
        intent_id: String,
        anchor_type: String,
        agent_id: String,
        score_bps: u64,
        round: u64,
        evidence_hash: Option<String>,
    ) -> BridgeResult<()> {
        info!(
            intent_id = %intent_id,
            anchor_type = %anchor_type,
            agent_id = %agent_id,
            score_bps = score_bps,
            "Processing Ethereum anchor request"
        );

        // Derive a deterministic Ethereum address from agent_id for demo
        // In production, this would look up actual Ethereum address from agent profile
        let eth_address = format!(
            "0x{:0>40}",
            &format!("{:x}", md5::compute(agent_id.as_bytes()))[..40]
        );

        let update = ReputationUpdate {
            agent_id: agent_id.clone(),
            eth_address,
            score_bps,
            round,
            evidence_hash: evidence_hash.clone(),
        };

        // Attempt to anchor reputation
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= self.config.max_retries {
            match self.ethereum_client.anchor_reputation(update.clone()).await {
                Ok(result) => {
                    info!(
                        intent_id = %intent_id,
                        tx_hash = %result.tx_hash,
                        block_number = result.block_number,
                        "Reputation anchor successful"
                    );

                    // Update intent status in Holochain
                    if self.config.enable_status_callback {
                        self.update_anchor_status(
                            &intent_id,
                            "confirmed",
                            Some(result.tx_hash.clone()),
                            Some(result.block_number),
                            result.commitment_id,
                            None,
                        )
                        .await?;
                    }

                    return Ok(());
                }
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e.to_string());

                    if attempts <= self.config.max_retries && self.config.retry_failed {
                        warn!(
                            intent_id = %intent_id,
                            attempt = attempts,
                            error = %last_error.as_ref().unwrap(),
                            "Anchor request failed, retrying"
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            self.config.retry_delay_ms * attempts as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        // All retries exhausted
        let error_msg = last_error.unwrap_or_else(|| "Unknown error".to_string());
        error!(
            intent_id = %intent_id,
            error = %error_msg,
            "Anchor request failed after retries"
        );

        // Update status to failed
        if self.config.enable_status_callback {
            self.update_anchor_status(&intent_id, "failed", None, None, None, Some(error_msg.clone()))
                .await?;
        }

        Err(BridgeError::Ethereum(error_msg))
    }

    /// Update payment intent status in Holochain.
    async fn update_payment_status(
        &self,
        intent_id: &str,
        status: &str,
        tx_hash: Option<String>,
        block_number: Option<u64>,
        error: Option<String>,
    ) -> BridgeResult<()> {
        debug!(
            intent_id = %intent_id,
            status = %status,
            "Updating payment intent status"
        );

        let client = self.holochain_client.read().await;

        // Call the Bridge zome to update status
        let payload = serde_json::json!({
            "intent_id": intent_id,
            "status": status,
            "tx_hash": tx_hash,
            "block_number": block_number,
            "error": error,
        });

        client
            .call_zome("bridge", "update_payment_intent_status", payload)
            .await
            .map_err(|e| BridgeError::Holochain(e.to_string()))?;

        Ok(())
    }

    /// Update anchor intent status in Holochain.
    async fn update_anchor_status(
        &self,
        intent_id: &str,
        status: &str,
        tx_hash: Option<String>,
        block_number: Option<u64>,
        commitment_id: Option<String>,
        error: Option<String>,
    ) -> BridgeResult<()> {
        debug!(
            intent_id = %intent_id,
            status = %status,
            "Updating anchor intent status"
        );

        let client = self.holochain_client.read().await;

        // Call the Bridge zome to update status
        let payload = serde_json::json!({
            "intent_id": intent_id,
            "status": status,
            "tx_hash": tx_hash,
            "block_number": block_number,
            "commitment_id": commitment_id,
            "error": error,
        });

        client
            .call_zome("bridge", "update_anchor_intent_status", payload)
            .await
            .map_err(|e| BridgeError::Holochain(e.to_string()))?;

        Ok(())
    }

    /// Create a signal handler closure for the listener.
    ///
    /// Returns a closure that can be registered with the HolochainListener
    /// to process Ethereum bridge signals asynchronously.
    pub fn create_signal_handler(
        self: Arc<Self>,
    ) -> impl Fn(HolochainSignal) + Send + Sync + 'static {
        move |signal: HolochainSignal| {
            let handler = self.clone();
            tokio::spawn(async move {
                if let Err(e) = handler.handle_signal(signal).await {
                    error!("Ethereum bridge handler error: {}", e);
                }
            });
        }
    }
}

/// Stub implementation when features are not enabled.
#[cfg(not(all(feature = "ethereum", feature = "holochain")))]
pub struct EthereumBridgeHandler;

#[cfg(not(all(feature = "ethereum", feature = "holochain")))]
impl EthereumBridgeHandler {
    /// Stub - features not enabled.
    pub fn new() -> Self {
        Self
    }
}

/// Integration helper for wiring up the bridge handler with the coordinator.
#[cfg(all(feature = "ethereum", feature = "holochain"))]
pub struct BridgeIntegration;

#[cfg(all(feature = "ethereum", feature = "holochain"))]
impl BridgeIntegration {
    /// Wire up the Ethereum bridge handler with the Holochain listener.
    ///
    /// This function creates the handler and registers it with the listener
    /// to process Ethereum bridge signals from Holochain zomes.
    pub async fn setup(
        listener: &crate::holochain::HolochainListener,
        ethereum_client: EthereumClient,
        holochain_client: HolochainClient,
        config: BridgeHandlerConfig,
    ) -> Arc<EthereumBridgeHandler> {
        let handler = Arc::new(EthereumBridgeHandler::new(
            ethereum_client,
            holochain_client,
            config,
        ));

        // Register payment handler
        let payment_handler = handler.clone();
        listener
            .on_ethereum_payment_request(move |signal| {
                let h = payment_handler.clone();
                tokio::spawn(async move {
                    if let Err(e) = h.handle_signal(signal).await {
                        error!("Payment handler error: {}", e);
                    }
                });
            })
            .await;

        // Register anchor handler
        let anchor_handler = handler.clone();
        listener
            .on_ethereum_anchor_request(move |signal| {
                let h = anchor_handler.clone();
                tokio::spawn(async move {
                    if let Err(e) = h.handle_signal(signal).await {
                        error!("Anchor handler error: {}", e);
                    }
                });
            })
            .await;

        info!("Ethereum bridge handlers registered with listener");

        handler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = BridgeHandlerConfig::default();
        assert!(config.enable_status_callback);
        assert!(config.retry_failed);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
    }
}
