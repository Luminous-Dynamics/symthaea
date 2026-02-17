//! Round Coordinator for Multi-Round Federated Learning
//!
//! Manages the lifecycle of training rounds, including:
//! - Round state machine (Waiting → Collecting → Aggregating → Complete)
//! - Automatic round transitions based on submission count or timeout
//! - Byzantine detection integration and event broadcasting
//! - Holochain persistence of round results
//!
//! # Example
//!
//! ```rust,ignore
//! use fl_aggregator::coordinator::{RoundCoordinator, CoordinatorConfig};
//!
//! let config = CoordinatorConfig::default()
//!     .with_round_timeout(Duration::from_secs(60))
//!     .with_min_participants(5);
//!
//! let coordinator = RoundCoordinator::new(config, aggregator, holochain_client);
//! coordinator.start().await;
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::aggregator::AsyncAggregator;
use crate::error::{AggregatorError, Result};
use crate::{Gradient, NodeId};

#[cfg(feature = "holochain")]
use crate::holochain::{
    ByzantineEvent, ByzantineSeverity, EarnReason, GradientRecord,
    HolochainClient, HolochainResult,
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the round coordinator.
#[derive(Clone, Debug)]
pub struct CoordinatorConfig {
    /// Timeout for each round (default: 60 seconds).
    pub round_timeout: Duration,

    /// Minimum participants required to start aggregation.
    pub min_participants: usize,

    /// Maximum rounds to run (0 = unlimited).
    pub max_rounds: u64,

    /// Grace period after min_participants reached before forcing aggregation.
    pub grace_period: Duration,

    /// Enable automatic Byzantine detection integration.
    pub enable_byzantine_detection: bool,

    /// Reputation penalty for Byzantine behavior.
    pub byzantine_penalty: f32,

    /// Reputation reward for quality gradients.
    pub quality_reward: f32,

    /// Credit amount for quality submissions.
    pub quality_credit: u64,

    /// Credit amount for Byzantine detection.
    pub detection_credit: u64,

    /// Credit amount for honest round participation.
    pub participation_reward: u64,

    /// Reputation bonus for honest participation.
    pub honest_bonus: f32,

    /// Enable Holochain persistence.
    pub enable_holochain: bool,

    /// Enable Ethereum bridge for payment distribution.
    pub enable_ethereum_bridge: bool,

    /// Model ID for this FL session (used in payment distribution).
    pub model_id: Option<String>,

    /// Default payment amount in Wei per round (optional).
    pub payment_amount_wei: Option<String>,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            round_timeout: Duration::from_secs(60),
            min_participants: 3,
            max_rounds: 0,
            grace_period: Duration::from_secs(10),
            enable_byzantine_detection: true,
            byzantine_penalty: -0.1,
            quality_reward: 0.01,
            quality_credit: 10,
            detection_credit: 50,
            participation_reward: 5,
            honest_bonus: 0.02,
            enable_holochain: true,
            enable_ethereum_bridge: false,
            model_id: None,
            payment_amount_wei: None,
        }
    }
}

impl CoordinatorConfig {
    /// Set round timeout.
    pub fn with_round_timeout(mut self, timeout: Duration) -> Self {
        self.round_timeout = timeout;
        self
    }

    /// Set minimum participants.
    pub fn with_min_participants(mut self, min: usize) -> Self {
        self.min_participants = min;
        self
    }

    /// Set maximum rounds.
    pub fn with_max_rounds(mut self, max: u64) -> Self {
        self.max_rounds = max;
        self
    }

    /// Set grace period.
    pub fn with_grace_period(mut self, grace: Duration) -> Self {
        self.grace_period = grace;
        self
    }

    /// Disable Holochain persistence.
    pub fn without_holochain(mut self) -> Self {
        self.enable_holochain = false;
        self
    }

    /// Enable Ethereum bridge for payment distribution.
    pub fn with_ethereum_bridge(mut self, model_id: impl Into<String>) -> Self {
        self.enable_ethereum_bridge = true;
        self.model_id = Some(model_id.into());
        self
    }

    /// Set payment amount for rounds.
    pub fn with_payment_amount(mut self, amount_wei: impl Into<String>) -> Self {
        self.payment_amount_wei = Some(amount_wei.into());
        self
    }
}

// =============================================================================
// Round State
// =============================================================================

/// Round state in the coordinator state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Waiting for round to start.
    Waiting,
    /// Collecting gradients from participants.
    Collecting,
    /// Aggregating collected gradients.
    Aggregating,
    /// Round complete, results available.
    Complete,
    /// Round failed (timeout or error).
    Failed,
}

/// Information about a single round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundInfo {
    /// Round number.
    pub round: u64,
    /// Current state.
    pub state: RoundState,
    /// Participants who submitted gradients.
    pub participants: Vec<String>,
    /// Nodes flagged as Byzantine.
    pub byzantine_nodes: Vec<String>,
    /// Start time (unix timestamp).
    pub started_at: u64,
    /// End time (unix timestamp, 0 if not ended).
    pub ended_at: u64,
    /// Aggregation result hash (if complete).
    pub result_hash: Option<String>,
    /// Error message (if failed).
    pub error: Option<String>,
}

impl RoundInfo {
    fn new(round: u64) -> Self {
        Self {
            round,
            state: RoundState::Waiting,
            participants: Vec::new(),
            byzantine_nodes: Vec::new(),
            started_at: 0,
            ended_at: 0,
            result_hash: None,
            error: None,
        }
    }
}

// =============================================================================
// Events
// =============================================================================

/// Events emitted by the coordinator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CoordinatorEvent {
    /// Round started.
    RoundStarted {
        round: u64,
        expected_participants: usize,
    },
    /// Gradient received from a node.
    GradientReceived {
        round: u64,
        node_id: String,
        total_received: usize,
    },
    /// Aggregation started.
    AggregationStarted {
        round: u64,
        participants: Vec<String>,
    },
    /// Round completed successfully.
    RoundCompleted {
        round: u64,
        participants: Vec<String>,
        byzantine_detected: Vec<String>,
        aggregation_time_ms: u64,
    },
    /// Round failed.
    RoundFailed {
        round: u64,
        reason: String,
    },
    /// Byzantine behavior detected.
    ByzantineDetected {
        round: u64,
        node_ids: Vec<String>,
        detection_method: String,
    },
    /// All rounds completed.
    TrainingComplete {
        total_rounds: u64,
    },
}

// =============================================================================
// Coordinator
// =============================================================================

/// Command sent to the coordinator.
#[derive(Debug)]
pub enum CoordinatorCommand {
    /// Start training.
    Start,
    /// Stop training.
    Stop,
    /// Force finalize current round.
    ForceFinalize,
    /// Submit gradient.
    SubmitGradient {
        node_id: String,
        gradient: Gradient,
        response: mpsc::Sender<Result<()>>,
    },
    /// Get current round info.
    GetRoundInfo {
        response: mpsc::Sender<RoundInfo>,
    },
}

/// Internal state for the coordinator.
struct CoordinatorState {
    current_round: u64,
    round_info: RoundInfo,
    round_started_at: Option<Instant>,
    grace_started_at: Option<Instant>,
    is_running: bool,
    submitted_this_round: HashSet<String>,
}

/// Round coordinator for multi-round federated learning.
pub struct RoundCoordinator {
    config: CoordinatorConfig,
    state: Arc<RwLock<CoordinatorState>>,
    aggregator: Arc<AsyncAggregator>,
    #[cfg(feature = "holochain")]
    holochain: Option<Arc<RwLock<crate::holochain::HolochainClient>>>,
    event_tx: broadcast::Sender<CoordinatorEvent>,
    cmd_tx: mpsc::Sender<CoordinatorCommand>,
    cmd_rx: Arc<RwLock<mpsc::Receiver<CoordinatorCommand>>>,
}

impl RoundCoordinator {
    /// Create a new round coordinator.
    pub fn new(
        config: CoordinatorConfig,
        aggregator: AsyncAggregator,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (cmd_tx, cmd_rx) = mpsc::channel(64);

        let state = CoordinatorState {
            current_round: 0,
            round_info: RoundInfo::new(0),
            round_started_at: None,
            grace_started_at: None,
            is_running: false,
            submitted_this_round: HashSet::new(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            aggregator: Arc::new(aggregator),
            #[cfg(feature = "holochain")]
            holochain: None,
            event_tx,
            cmd_tx,
            cmd_rx: Arc::new(RwLock::new(cmd_rx)),
        }
    }

    /// Set Holochain client for persistence.
    #[cfg(feature = "holochain")]
    pub fn with_holochain(mut self, client: HolochainClient) -> Self {
        self.holochain = Some(Arc::new(RwLock::new(client)));
        self
    }

    /// Get event receiver for subscribing to coordinator events.
    pub fn subscribe(&self) -> broadcast::Receiver<CoordinatorEvent> {
        self.event_tx.subscribe()
    }

    /// Get command sender for controlling the coordinator.
    pub fn command_sender(&self) -> mpsc::Sender<CoordinatorCommand> {
        self.cmd_tx.clone()
    }

    /// Start the coordinator (blocking).
    pub async fn run(&self) -> Result<()> {
        info!(
            min_participants = self.config.min_participants,
            max_rounds = self.config.max_rounds,
            "Starting round coordinator"
        );

        let mut cmd_rx = self.cmd_rx.write().await;

        // Start first round
        self.start_round().await?;

        // Main event loop
        loop {
            let timeout = self.calculate_timeout().await;

            tokio::select! {
                // Handle commands
                Some(cmd) = cmd_rx.recv() => {
                    match self.handle_command(cmd).await {
                        Ok(should_stop) if should_stop => {
                            info!("Coordinator stopped by command");
                            break;
                        }
                        Err(e) => {
                            error!("Command error: {}", e);
                        }
                        _ => {}
                    }
                }

                // Handle timeout
                _ = tokio::time::sleep(timeout) => {
                    self.handle_timeout().await?;
                }
            }

            // Check if we've hit max rounds
            let state = self.state.read().await;
            if self.config.max_rounds > 0 && state.current_round >= self.config.max_rounds {
                let _ = self.event_tx.send(CoordinatorEvent::TrainingComplete {
                    total_rounds: state.current_round,
                });
                info!(total_rounds = state.current_round, "Training complete");
                break;
            }
        }

        Ok(())
    }

    /// Calculate timeout for current state.
    async fn calculate_timeout(&self) -> Duration {
        let state = self.state.read().await;

        match state.round_info.state {
            RoundState::Waiting => Duration::from_secs(1),
            RoundState::Collecting => {
                if let Some(started) = state.round_started_at {
                    let elapsed = started.elapsed();
                    if elapsed >= self.config.round_timeout {
                        Duration::from_millis(10)
                    } else {
                        self.config.round_timeout - elapsed
                    }
                } else {
                    self.config.round_timeout
                }
            }
            RoundState::Aggregating => Duration::from_secs(30),
            _ => Duration::from_secs(1),
        }
    }

    /// Handle incoming command.
    async fn handle_command(&self, cmd: CoordinatorCommand) -> Result<bool> {
        match cmd {
            CoordinatorCommand::Start => {
                let mut state = self.state.write().await;
                state.is_running = true;
                Ok(false)
            }
            CoordinatorCommand::Stop => {
                let mut state = self.state.write().await;
                state.is_running = false;
                Ok(true)
            }
            CoordinatorCommand::ForceFinalize => {
                self.finalize_round().await?;
                Ok(false)
            }
            CoordinatorCommand::SubmitGradient {
                node_id,
                gradient,
                response,
            } => {
                let result = self.submit_gradient(&node_id, gradient).await;
                let _ = response.send(result).await;
                Ok(false)
            }
            CoordinatorCommand::GetRoundInfo { response } => {
                let state = self.state.read().await;
                let _ = response.send(state.round_info.clone()).await;
                Ok(false)
            }
        }
    }

    /// Handle timeout event.
    async fn handle_timeout(&self) -> Result<()> {
        let state = self.state.read().await;
        let round_state = state.round_info.state;
        let participants = state.submitted_this_round.len();
        drop(state);

        match round_state {
            RoundState::Collecting => {
                if participants >= self.config.min_participants {
                    // Check grace period
                    let state = self.state.read().await;
                    if let Some(grace_start) = state.grace_started_at {
                        if grace_start.elapsed() >= self.config.grace_period {
                            drop(state);
                            self.finalize_round().await?;
                        }
                    }
                } else {
                    // Check round timeout
                    let state = self.state.read().await;
                    if let Some(started) = state.round_started_at {
                        if started.elapsed() >= self.config.round_timeout {
                            drop(state);
                            self.handle_round_timeout().await?;
                        }
                    }
                }
            }
            RoundState::Complete => {
                self.start_round().await?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Register a node with the aggregator.
    pub async fn register_node(&self, node_id: impl Into<NodeId>) -> Result<()> {
        self.aggregator.register_node(node_id).await
    }

    /// Start a new round.
    pub async fn start_round(&self) -> Result<()> {
        let mut state = self.state.write().await;

        state.current_round += 1;
        state.round_info = RoundInfo::new(state.current_round);
        state.round_info.state = RoundState::Collecting;
        state.round_info.started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        state.round_started_at = Some(Instant::now());
        state.grace_started_at = None;
        state.submitted_this_round.clear();

        let round = state.current_round;
        let expected = self.config.min_participants;

        drop(state);

        // Note: AsyncAggregator auto-resets after finalize, so no explicit reset needed

        info!(round = round, expected = expected, "Round started");

        let _ = self.event_tx.send(CoordinatorEvent::RoundStarted {
            round,
            expected_participants: expected,
        });

        Ok(())
    }

    /// Submit a gradient for the current round.
    pub async fn submit_gradient(&self, node_id: &str, gradient: Gradient) -> Result<()> {
        let state = self.state.write().await;

        if state.round_info.state != RoundState::Collecting {
            return Err(AggregatorError::Internal(
                format!("Round {} not in collecting state", state.current_round)
            ));
        }

        if state.submitted_this_round.contains(node_id) {
            return Err(AggregatorError::DuplicateSubmission {
                node: node_id.to_string(),
                round: state.current_round,
            });
        }

        // Submit to aggregator
        drop(state);
        self.aggregator.submit(node_id, gradient.clone()).await?;

        let mut state = self.state.write().await;
        state.submitted_this_round.insert(node_id.to_string());
        state.round_info.participants.push(node_id.to_string());

        let round = state.current_round;
        let total = state.submitted_this_round.len();

        // Start grace period if we've hit minimum
        if total == self.config.min_participants && state.grace_started_at.is_none() {
            state.grace_started_at = Some(Instant::now());
            debug!(round = round, "Grace period started");
        }

        drop(state);

        let _ = self.event_tx.send(CoordinatorEvent::GradientReceived {
            round,
            node_id: node_id.to_string(),
            total_received: total,
        });

        // Store to Holochain
        #[cfg(feature = "holochain")]
        if self.config.enable_holochain {
            if let Some(ref hc) = self.holochain {
                let record = GradientRecord::new(
                    node_id,
                    round,
                    gradient.as_slice().unwrap_or(&[]),
                    1.0, // Default reputation
                );

                let client = hc.read().await;
                if let Err(e) = client.store_gradient(&record).await {
                    warn!("Failed to store gradient to Holochain: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Finalize the current round.
    pub async fn finalize_round(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.round_info.state != RoundState::Collecting {
            return Ok(());
        }

        state.round_info.state = RoundState::Aggregating;
        let round = state.current_round;
        let participants: Vec<String> = state.submitted_this_round.iter().cloned().collect();

        drop(state);

        let _ = self.event_tx.send(CoordinatorEvent::AggregationStarted {
            round,
            participants: participants.clone(),
        });

        let start_time = Instant::now();

        // Perform aggregation
        let result = self.aggregator.force_finalize().await;

        let aggregation_time_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(aggregated) => {
                // Note: Byzantine detection results would come from the detection module
                // For now, we'll use an empty list and integrate detection later
                let byzantine_detected: Vec<String> = Vec::new();

                // Update state
                let mut state = self.state.write().await;
                state.round_info.state = RoundState::Complete;
                state.round_info.ended_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                state.round_info.byzantine_nodes = byzantine_detected.clone();

                // Compute result hash from gradient bytes
                let gradient_bytes: Vec<u8> = aggregated
                    .as_slice()
                    .unwrap_or(&[])
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let hash = format!("{:x}", md5::compute(&gradient_bytes));
                state.round_info.result_hash = Some(hash);

                #[allow(unused_variables)]
                let result_hash = state.round_info.result_hash.clone().unwrap_or_default();
                drop(state);

                // Log Byzantine events and update reputation
                #[cfg(feature = "holochain")]
                if self.config.enable_holochain && !byzantine_detected.is_empty() {
                    self.log_byzantine_events(round, &byzantine_detected).await;
                }

                // Award credits to honest participants
                #[cfg(feature = "holochain")]
                if self.config.enable_holochain {
                    self.award_honest_participants(
                        round,
                        &participants,
                        &byzantine_detected,
                        &result_hash,
                    ).await;
                }

                // Trigger Ethereum bridge payment distribution
                #[cfg(feature = "holochain")]
                if self.config.enable_holochain && self.config.enable_ethereum_bridge {
                    self.trigger_ethereum_payment(
                        round,
                        &participants,
                        &byzantine_detected,
                    ).await;
                }

                info!(
                    round = round,
                    participants = participants.len(),
                    byzantine = byzantine_detected.len(),
                    time_ms = aggregation_time_ms,
                    "Round completed"
                );

                let _ = self.event_tx.send(CoordinatorEvent::RoundCompleted {
                    round,
                    participants,
                    byzantine_detected: byzantine_detected.clone(),
                    aggregation_time_ms,
                });

                if !byzantine_detected.is_empty() {
                    let _ = self.event_tx.send(CoordinatorEvent::ByzantineDetected {
                        round,
                        node_ids: byzantine_detected,
                        detection_method: "coordinator".to_string(),
                    });
                }

                Ok(())
            }
            Err(e) => {
                let mut state = self.state.write().await;
                state.round_info.state = RoundState::Failed;
                state.round_info.error = Some(e.to_string());
                drop(state);

                error!(round = round, error = %e, "Round failed");

                let _ = self.event_tx.send(CoordinatorEvent::RoundFailed {
                    round,
                    reason: e.to_string(),
                });

                Err(e)
            }
        }
    }

    /// Handle round timeout.
    async fn handle_round_timeout(&self) -> Result<()> {
        let state = self.state.read().await;
        let participants = state.submitted_this_round.len();
        drop(state);

        if participants >= self.config.min_participants {
            // We have enough, finalize
            self.finalize_round().await
        } else {
            // Not enough participants, fail round
            let mut state = self.state.write().await;
            let round = state.current_round;

            state.round_info.state = RoundState::Failed;
            state.round_info.error = Some(format!(
                "Timeout: only {} of {} required participants",
                participants, self.config.min_participants
            ));

            drop(state);

            warn!(
                round = round,
                got = participants,
                required = self.config.min_participants,
                "Round timed out with insufficient participants"
            );

            let _ = self.event_tx.send(CoordinatorEvent::RoundFailed {
                round,
                reason: "Insufficient participants".to_string(),
            });

            // Start next round
            self.start_round().await
        }
    }

    /// Log Byzantine events to Holochain.
    #[cfg(feature = "holochain")]
    async fn log_byzantine_events(&self, round: u64, node_ids: &[String]) {
        if let Some(ref hc) = self.holochain {
            let client = hc.read().await;

            for node_id in node_ids {
                let event = ByzantineEvent {
                    node_id: node_id.clone(),
                    round_num: round,
                    detection_method: "aggregator".to_string(),
                    severity: ByzantineSeverity::Medium,
                    details: serde_json::json!({
                        "detected_at": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    }),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                };

                if let Err(e) = client.log_byzantine_event(&event).await {
                    warn!("Failed to log Byzantine event: {}", e);
                }

                // Update reputation
                if let Err(e) = client
                    .update_reputation(node_id, self.config.byzantine_penalty, "byzantine_detected")
                    .await
                {
                    warn!("Failed to update reputation: {}", e);
                }
            }
        }
    }

    /// Award credits to honest participants after successful round.
    #[cfg(feature = "holochain")]
    async fn award_honest_participants(
        &self,
        round: u64,
        participants: &[String],
        byzantine_detected: &[String],
        result_hash: &str,
    ) {
        if let Some(ref hc) = self.holochain {
            let client = hc.read().await;
            let honest: Vec<_> = participants
                .iter()
                .filter(|p| !byzantine_detected.contains(p))
                .collect();

            for node_id in honest {
                // Award participation credits
                let reason = EarnReason::QualityGradient {
                    pogq_score: 1.0, // Full credit for honest participation
                    gradient_hash: format!("round_{}_participant_{}", round, node_id),
                };

                if let Err(e) = client
                    .issue_credit(node_id, self.config.participation_reward, reason)
                    .await
                {
                    warn!("Failed to issue credit to {}: {}", node_id, e);
                }

                // Update reputation positively
                if let Err(e) = client
                    .update_reputation(node_id, self.config.honest_bonus, "honest_participation")
                    .await
                {
                    warn!("Failed to update reputation for {}: {}", node_id, e);
                }
            }

            debug!(
                round = round,
                honest_count = participants.len() - byzantine_detected.len(),
                result_hash = result_hash,
                "Awarded credits to honest participants"
            );
        }
    }

    /// Trigger Ethereum payment distribution via Bridge zome.
    ///
    /// This calls the Bridge zome's `distribute_payment_on_chain` extern,
    /// which creates a PaymentIntent and emits a signal for the native
    /// host to process via EthereumClient.
    #[cfg(feature = "holochain")]
    async fn trigger_ethereum_payment(
        &self,
        round: u64,
        participants: &[String],
        byzantine_detected: &[String],
    ) {
        let model_id = match &self.config.model_id {
            Some(id) => id.clone(),
            None => {
                warn!("Ethereum bridge enabled but no model_id configured");
                return;
            }
        };

        // Calculate fair share for honest participants
        let honest_participants: Vec<&String> = participants
            .iter()
            .filter(|p| !byzantine_detected.contains(p))
            .collect();

        if honest_participants.is_empty() {
            warn!(round = round, "No honest participants to pay");
            return;
        }

        // Equal split among honest participants (basis points)
        let share_bps = 10000 / honest_participants.len() as u64;

        // Build splits array
        // In production, this would use Shapley values from the aggregation
        // and actual Ethereum addresses from agent profiles
        let splits: Vec<serde_json::Value> = honest_participants
            .iter()
            .enumerate()
            .map(|(i, agent_id)| {
                // Derive a deterministic "address" from agent_id for demo
                // In production, look up actual Ethereum address from agent profile
                let address = format!(
                    "0x{:0>40}",
                    &format!("{:x}", md5::compute(agent_id.as_bytes()))[..40]
                );
                serde_json::json!({
                    "address": address,
                    "basis_points": if i == 0 {
                        // First participant gets remainder to ensure sum = 10000
                        share_bps + (10000 % honest_participants.len() as u64)
                    } else {
                        share_bps
                    },
                    "agent_id": agent_id,
                })
            })
            .collect();

        // Get payment amount (use configured or default)
        let total_amount_wei = self.config.payment_amount_wei
            .clone()
            .unwrap_or_else(|| "1000000000000000000".to_string()); // 1 ETH default

        if let Some(ref hc) = self.holochain {
            let client = hc.read().await;

            let payload = serde_json::json!({
                "model_id": model_id,
                "round": round,
                "total_amount_wei": total_amount_wei,
                "splits": splits,
            });

            info!(
                round = round,
                model_id = %model_id,
                participants = honest_participants.len(),
                "Triggering Ethereum payment distribution"
            );

            match client.call_zome("bridge", "distribute_payment_on_chain", payload).await {
                Ok(_) => {
                    debug!(round = round, "Payment distribution triggered successfully");
                }
                Err(e) => {
                    error!(round = round, error = %e, "Failed to trigger payment distribution");
                }
            }
        }
    }

    /// Anchor reputation to Ethereum via Bridge zome.
    ///
    /// This is called after significant reputation changes to create
    /// an immutable on-chain record of agent reputation.
    #[cfg(feature = "holochain")]
    #[allow(dead_code)]
    async fn anchor_reputation_to_ethereum(
        &self,
        agent_id: &str,
        score_bps: u64,
        round: u64,
        evidence_hash: Option<String>,
    ) {
        if !self.config.enable_ethereum_bridge {
            return;
        }

        if let Some(ref hc) = self.holochain {
            let client = hc.read().await;

            let payload = serde_json::json!({
                "anchor_type": "reputation",
                "agent_id": agent_id,
                "score_bps": score_bps,
                "round": round,
                "evidence_hash": evidence_hash,
            });

            info!(
                agent_id = %agent_id,
                score_bps = score_bps,
                round = round,
                "Anchoring reputation to Ethereum"
            );

            match client.call_zome("bridge", "anchor_to_ethereum", payload).await {
                Ok(_) => {
                    debug!(agent_id = %agent_id, "Reputation anchor triggered successfully");
                }
                Err(e) => {
                    error!(agent_id = %agent_id, error = %e, "Failed to anchor reputation");
                }
            }
        }
    }

    /// Get current round info.
    pub async fn round_info(&self) -> RoundInfo {
        self.state.read().await.round_info.clone()
    }

    /// Get current round number.
    pub async fn current_round(&self) -> u64 {
        self.state.read().await.current_round
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregator::AggregatorConfig;
    use crate::byzantine::Defense;
    use ndarray::Array1;

    fn create_test_coordinator() -> RoundCoordinator {
        let config = CoordinatorConfig::default()
            .with_min_participants(2)
            .with_round_timeout(Duration::from_secs(5))
            .without_holochain();

        let agg_config = AggregatorConfig::default()
            .with_expected_nodes(5)
            .with_defense(Defense::FedAvg);

        let aggregator = AsyncAggregator::new(agg_config);
        RoundCoordinator::new(config, aggregator)
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = create_test_coordinator();
        let info = coordinator.round_info().await;
        assert_eq!(info.round, 0);
        assert_eq!(info.state, RoundState::Waiting);
    }

    #[tokio::test]
    async fn test_round_start() {
        let coordinator = create_test_coordinator();
        coordinator.start_round().await.unwrap();

        let info = coordinator.round_info().await;
        assert_eq!(info.round, 1);
        assert_eq!(info.state, RoundState::Collecting);
    }

    #[tokio::test]
    async fn test_gradient_submission() {
        let coordinator = create_test_coordinator();

        // Register nodes first
        coordinator.register_node("node_1").await.unwrap();

        coordinator.start_round().await.unwrap();

        let gradient = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        coordinator.submit_gradient("node_1", gradient).await.unwrap();

        let info = coordinator.round_info().await;
        assert_eq!(info.participants.len(), 1);
        assert!(info.participants.contains(&"node_1".to_string()));
    }

    #[tokio::test]
    async fn test_round_finalization() {
        let coordinator = create_test_coordinator();

        // Register nodes first
        coordinator.register_node("node_1").await.unwrap();
        coordinator.register_node("node_2").await.unwrap();

        coordinator.start_round().await.unwrap();

        // Submit minimum required gradients
        let g1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let g2 = Array1::from_vec(vec![1.5, 2.5, 3.5]);

        coordinator.submit_gradient("node_1", g1).await.unwrap();
        coordinator.submit_gradient("node_2", g2).await.unwrap();

        coordinator.finalize_round().await.unwrap();

        let info = coordinator.round_info().await;
        assert_eq!(info.state, RoundState::Complete);
        assert!(info.result_hash.is_some());
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let coordinator = create_test_coordinator();
        let mut rx = coordinator.subscribe();

        coordinator.start_round().await.unwrap();

        // Should receive RoundStarted event
        let event = rx.recv().await.unwrap();
        match event {
            CoordinatorEvent::RoundStarted { round, .. } => {
                assert_eq!(round, 1);
            }
            _ => panic!("Expected RoundStarted event"),
        }
    }
}
