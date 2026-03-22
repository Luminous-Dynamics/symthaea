//! Symthaea ↔ Mycelix Conductor Adapter
//!
//! Bridges Symthaea's governance dispatch commands to a real Holochain conductor.
//! Uses a trait-based transport (`ConductorTransport`) so that the actual
//! `holochain_client::AppWebsocket` connection can be provided by a separate
//! binary (avoiding serde version conflicts with the Symthaea workspace).
//!
//! # Architecture
//!
//! ```text
//! CognitiveLoopService
//!   └─ GovernanceManager (interval 37)
//!       └─ MycelixBridge::dispatch_governance_command()
//!           └─ mpsc::SyncSender<DispatchCommand>
//!               └─ [this crate] GovernanceDispatcher
//!                   └─ ConductorTransport::call_zome()
//!                       └─ Holochain Conductor (mycelix-governance DNA)
//! ```

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn};

// ============================================================================
// Types mirrored from mycelix_bridge.rs (avoid circular dependency)
// ============================================================================

/// Commands dispatched from the cognitive loop to the conductor.
/// Mirrors `GovernanceDispatchCommand` from `mycelix_bridge.rs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DispatchCommand {
    SubmitProposal {
        correlation_id: u64,
        description: String,
        proposer_did: String,
        consciousness_phi: f64,
        alignment_score: f64,
    },
    CastVote {
        correlation_id: u64,
        proposal_id: String,
        voter_did: String,
        approve: bool,
        rationale: String,
    },
    QueryActiveProposals,
    /// Evaluate an asset and record consciousness assessment on-chain.
    EvaluateAsset {
        correlation_id: u64,
        project_id: String,
        phi_score: f64,
        harmony_alignment: f64,
        per_harmony_scores: String,
        care_activation: f64,
        meta_awareness: f64,
    },
}

/// Outcome received back from the conductor.
/// Mirrors `GovernanceDispatchOutcome` from `mycelix_bridge.rs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DispatchOutcome {
    ProposalAccepted {
        correlation_id: u64,
        action_hash: Option<String>,
    },
    ProposalRejected {
        correlation_id: u64,
        reason: String,
    },
    VoteAccepted {
        correlation_id: u64,
        action_hash: Option<String>,
    },
    VoteRejected {
        correlation_id: u64,
        reason: String,
    },
    Timeout {
        correlation_id: u64,
    },
}

// ============================================================================
// Transport Trait
// ============================================================================

/// Abstract transport for calling Holochain zome functions.
///
/// Implement this trait with `holochain_client::AppWebsocket` in a separate
/// binary to avoid serde version conflicts. A mock implementation is provided
/// for testing.
#[async_trait::async_trait]
pub trait ConductorTransport: Send {
    /// Call a zome function and return the raw response bytes.
    async fn call_zome(
        &mut self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, String>;

    /// Whether the transport is currently connected.
    fn is_connected(&self) -> bool;
}

/// Mock transport that always returns success. For testing.
pub struct MockTransport;

#[async_trait::async_trait]
impl ConductorTransport for MockTransport {
    async fn call_zome(
        &mut self,
        _role_name: &str,
        _zome_name: &str,
        _fn_name: &str,
        _payload: Vec<u8>,
    ) -> Result<Vec<u8>, String> {
        Ok(vec![])
    }

    fn is_connected(&self) -> bool {
        true
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for connecting to a Holochain conductor.
#[derive(Debug, Clone)]
pub struct ConductorConfig {
    /// WebSocket URL (e.g., "ws://localhost:8888")
    pub url: String,
    /// App authentication token
    pub token: String,
    /// Installed app ID (e.g., "mycelix-unified")
    pub app_id: String,
    /// Timeout for individual zome calls
    pub call_timeout: Duration,
    /// Maximum reconnection attempts before giving up
    pub max_reconnect_attempts: u32,
}

impl ConductorConfig {
    /// Create from environment variables.
    ///
    /// Reads `MYCELIX_CONDUCTOR_URL`, `MYCELIX_APP_TOKEN`, `MYCELIX_APP_ID`.
    /// Returns `None` if any variable is missing.
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("MYCELIX_CONDUCTOR_URL").ok()?;
        let token = std::env::var("MYCELIX_APP_TOKEN").ok()?;
        let app_id = std::env::var("MYCELIX_APP_ID").ok()?;
        Some(Self {
            url,
            token,
            app_id,
            call_timeout: Duration::from_secs(30),
            max_reconnect_attempts: 5,
        })
    }
}

// ============================================================================
// Governance Dispatcher
// ============================================================================

/// Translates `DispatchCommand`s into conductor zome calls.
pub struct GovernanceDispatcher<T: ConductorTransport> {
    transport: T,
    /// Governance DNA role name in the unified hApp.
    governance_role: String,
}

impl<T: ConductorTransport> GovernanceDispatcher<T> {
    /// Create a new dispatcher targeting the governance role.
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            governance_role: "mycelix-governance".to_string(),
        }
    }

    /// Create with a custom governance role name.
    pub fn with_role(transport: T, role: impl Into<String>) -> Self {
        Self {
            transport,
            governance_role: role.into(),
        }
    }

    /// Dispatch a single command, returning the outcome.
    pub async fn dispatch(&mut self, cmd: DispatchCommand) -> DispatchOutcome {
        match cmd {
            DispatchCommand::SubmitProposal {
                correlation_id,
                description,
                proposer_did,
                consciousness_phi,
                alignment_score,
            } => {
                let payload = serde_json::json!({
                    "description": description,
                    "proposer_did": proposer_did,
                    "consciousness_phi": consciousness_phi,
                    "alignment_score": alignment_score,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(
                        &self.governance_role,
                        "agora",
                        "create_proposal",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(result) => {
                        let action_hash = String::from_utf8(result).ok();
                        info!(correlation_id, "Proposal accepted by conductor");
                        DispatchOutcome::ProposalAccepted {
                            correlation_id,
                            action_hash,
                        }
                    }
                    Err(reason) => {
                        warn!(correlation_id, %reason, "Proposal rejected by conductor");
                        DispatchOutcome::ProposalRejected {
                            correlation_id,
                            reason,
                        }
                    }
                }
            }

            DispatchCommand::CastVote {
                correlation_id,
                proposal_id,
                voter_did,
                approve,
                rationale,
            } => {
                let payload = serde_json::json!({
                    "proposal_id": proposal_id,
                    "voter_did": voter_did,
                    "approve": approve,
                    "rationale": rationale,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(&self.governance_role, "agora", "cast_vote", payload_bytes)
                    .await
                {
                    Ok(result) => {
                        let action_hash = String::from_utf8(result).ok();
                        info!(correlation_id, "Vote accepted by conductor");
                        DispatchOutcome::VoteAccepted {
                            correlation_id,
                            action_hash,
                        }
                    }
                    Err(reason) => {
                        warn!(correlation_id, %reason, "Vote rejected by conductor");
                        DispatchOutcome::VoteRejected {
                            correlation_id,
                            reason,
                        }
                    }
                }
            }

            DispatchCommand::QueryActiveProposals => {
                let _ = self
                    .transport
                    .call_zome(
                        &self.governance_role,
                        "agora",
                        "get_active_proposals",
                        vec![],
                    )
                    .await;
                DispatchOutcome::ProposalAccepted {
                    correlation_id: 0,
                    action_hash: None,
                }
            }

            DispatchCommand::EvaluateAsset {
                correlation_id,
                project_id,
                phi_score,
                harmony_alignment,
                per_harmony_scores,
                care_activation,
                meta_awareness,
            } => {
                let payload = serde_json::json!({
                    "project_id": project_id,
                    "scorer_did": "did:mycelix:symthaea",
                    "phi_score": phi_score,
                    "harmony_alignment": harmony_alignment,
                    "per_harmony_scores": per_harmony_scores,
                    "care_activation": care_activation,
                    "meta_awareness": meta_awareness,
                    "assessment_cycle": 0,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(
                        "energy",
                        "energy_bridge",
                        "record_consciousness_assessment",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(result) => {
                        let action_hash = String::from_utf8(result).ok();
                        info!(correlation_id, %project_id, phi_score, harmony_alignment,
                            "Asset consciousness assessment recorded on-chain");
                        DispatchOutcome::ProposalAccepted {
                            correlation_id,
                            action_hash,
                        }
                    }
                    Err(reason) => {
                        warn!(correlation_id, %project_id, %reason,
                            "Asset assessment rejected by conductor");
                        DispatchOutcome::ProposalRejected {
                            correlation_id,
                            reason,
                        }
                    }
                }
            }
        }
    }

    /// Run the dispatch loop, draining commands from the receiver.
    ///
    /// Sends outcomes back through the `outcome_tx` channel.
    /// Tracks pending commands and injects timeout events after 30s.
    pub async fn run_dispatch_loop(
        mut self,
        rx: std::sync::mpsc::Receiver<DispatchCommand>,
        outcome_tx: tokio::sync::mpsc::Sender<DispatchOutcome>,
    ) {
        info!("Governance dispatch loop started");
        let timeout_duration = Duration::from_secs(30);
        let mut pending: Vec<(u64, std::time::Instant)> = Vec::new();

        loop {
            while let Ok(cmd) = rx.try_recv() {
                let corr_id = match &cmd {
                    DispatchCommand::SubmitProposal { correlation_id, .. } => *correlation_id,
                    DispatchCommand::CastVote { correlation_id, .. } => *correlation_id,
                    DispatchCommand::QueryActiveProposals => 0,
                    DispatchCommand::EvaluateAsset { correlation_id, .. } => *correlation_id,
                };
                if corr_id > 0 {
                    pending.push((corr_id, std::time::Instant::now()));
                }

                let outcome = self.dispatch(cmd).await;
                let responded_id = match &outcome {
                    DispatchOutcome::ProposalAccepted { correlation_id, .. }
                    | DispatchOutcome::ProposalRejected { correlation_id, .. }
                    | DispatchOutcome::VoteAccepted { correlation_id, .. }
                    | DispatchOutcome::VoteRejected { correlation_id, .. }
                    | DispatchOutcome::Timeout { correlation_id } => *correlation_id,
                };
                pending.retain(|(id, _)| *id != responded_id);

                if outcome_tx.send(outcome).await.is_err() {
                    warn!("Outcome channel closed, stopping dispatch loop");
                    return;
                }
            }

            // Check for timeouts
            let now = std::time::Instant::now();
            let timed_out: Vec<u64> = pending
                .iter()
                .filter(|(_, sent_at)| now.duration_since(*sent_at) > timeout_duration)
                .map(|(id, _)| *id)
                .collect();

            for corr_id in &timed_out {
                warn!(correlation_id = corr_id, "Dispatch command timed out (30s)");
                let _ = outcome_tx
                    .send(DispatchOutcome::Timeout {
                        correlation_id: *corr_id,
                    })
                    .await;
            }
            pending.retain(|(id, _)| !timed_out.contains(id));

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_env_missing_vars() {
        std::env::remove_var("MYCELIX_CONDUCTOR_URL");
        assert!(ConductorConfig::from_env().is_none());
    }

    #[test]
    fn dispatch_command_serde_roundtrip() {
        let cmd = DispatchCommand::SubmitProposal {
            correlation_id: 42,
            description: "Test proposal".into(),
            proposer_did: "did:mycelix:test".into(),
            consciousness_phi: 0.8,
            alignment_score: 0.9,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let decoded: DispatchCommand = serde_json::from_str(&json).unwrap();
        match decoded {
            DispatchCommand::SubmitProposal { correlation_id, .. } => {
                assert_eq!(correlation_id, 42);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn dispatch_outcome_serde_roundtrip() {
        let outcome = DispatchOutcome::ProposalAccepted {
            correlation_id: 1,
            action_hash: Some("uhCkk...".to_string()),
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let decoded: DispatchOutcome = serde_json::from_str(&json).unwrap();
        match decoded {
            DispatchOutcome::ProposalAccepted {
                correlation_id,
                action_hash,
            } => {
                assert_eq!(correlation_id, 1);
                assert!(action_hash.is_some());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn mock_transport_connect_and_call() {
        let mut transport = MockTransport;
        assert!(transport.is_connected());
        let result = transport
            .call_zome("governance", "agora", "create_proposal", vec![])
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn mock_dispatcher_submit_proposal() {
        let mut dispatcher = GovernanceDispatcher::new(MockTransport);

        let cmd = DispatchCommand::SubmitProposal {
            correlation_id: 100,
            description: "Test proposal".into(),
            proposer_did: "did:mycelix:test".into(),
            consciousness_phi: 0.85,
            alignment_score: 0.9,
        };

        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::ProposalAccepted { correlation_id, .. } => {
                assert_eq!(correlation_id, 100);
            }
            _ => panic!("Expected ProposalAccepted, got {:?}", outcome),
        }
    }

    #[tokio::test]
    async fn mock_dispatcher_cast_vote() {
        let mut dispatcher = GovernanceDispatcher::new(MockTransport);

        let cmd = DispatchCommand::CastVote {
            correlation_id: 200,
            proposal_id: "uhCkk_test".into(),
            voter_did: "did:mycelix:voter".into(),
            approve: true,
            rationale: "Good proposal".into(),
        };

        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::VoteAccepted { correlation_id, .. } => {
                assert_eq!(correlation_id, 200);
            }
            _ => panic!("Expected VoteAccepted, got {:?}", outcome),
        }
    }

    #[tokio::test]
    async fn dispatch_loop_timeout_detection() {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel(10);
        let (outcome_tx, mut outcome_rx) = tokio::sync::mpsc::channel(10);

        // Create a transport that always fails (simulates disconnected conductor)
        struct FailTransport;
        #[async_trait::async_trait]
        impl ConductorTransport for FailTransport {
            async fn call_zome(
                &mut self,
                _: &str,
                _: &str,
                _: &str,
                _: Vec<u8>,
            ) -> Result<Vec<u8>, String> {
                Err("conductor unavailable".into())
            }
            fn is_connected(&self) -> bool {
                false
            }
        }

        let dispatcher = GovernanceDispatcher::new(FailTransport);

        // Send a command
        cmd_tx
            .send(DispatchCommand::SubmitProposal {
                correlation_id: 999,
                description: "will fail".into(),
                proposer_did: "did:test".into(),
                consciousness_phi: 0.5,
                alignment_score: 0.5,
            })
            .unwrap();

        // Run dispatch loop in background
        let handle = tokio::spawn(async move {
            dispatcher.run_dispatch_loop(cmd_rx, outcome_tx).await;
        });

        // Should get a rejection (not timeout, since the call returns immediately)
        let outcome = tokio::time::timeout(Duration::from_secs(5), outcome_rx.recv())
            .await
            .expect("should receive outcome")
            .expect("channel should not be closed");

        match outcome {
            DispatchOutcome::ProposalRejected {
                correlation_id,
                reason,
            } => {
                assert_eq!(correlation_id, 999);
                assert!(reason.contains("unavailable"));
            }
            _ => panic!("Expected ProposalRejected, got {:?}", outcome),
        }

        handle.abort();
    }
}
