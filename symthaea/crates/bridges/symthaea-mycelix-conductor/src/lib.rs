// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

fn now_micros_i64() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let micros = dur.as_micros();
    if micros > i64::MAX as u128 {
        i64::MAX
    } else {
        micros as i64
    }
}

fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

fn severity_level_str(severity: u8) -> &'static str {
    match severity.clamp(1, 5) {
        1 => "Level1",
        2 => "Level2",
        3 => "Level3",
        4 => "Level4",
        5 => "Level5",
        _ => "Level3",
    }
}

fn disaster_type_value(crisis_type: &str) -> serde_json::Value {
    // Mycelix civic emergency-incidents `DisasterType` uses Rust enum variant names
    // (e.g., "Infrastructure", "CyberAttack"). Unknown values map to `Other(String)`.
    match crisis_type {
        "Hurricane" | "Earthquake" | "Wildfire" | "Flood" | "Tornado" | "Pandemic"
        | "Industrial" | "MassCasualty" | "CyberAttack" | "Infrastructure" => {
            serde_json::Value::String(crisis_type.to_string())
        }
        other => serde_json::json!({ "Other": other }),
    }
}

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
        meta_awareness: f64,
        coherence: f64,
        care_activation: f64,
        alignment_score: f64,
    },
    CastVote {
        correlation_id: u64,
        proposal_id: String,
        voter_did: String,
        approve: bool,
        rationale: String,
        consciousness_phi: f64,
        meta_awareness: f64,
        coherence: f64,
        care_activation: f64,
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
    /// Declare a civic crisis to the emergency-incidents zome.
    DeclareCrisis {
        correlation_id: u64,
        severity: u8,
        crisis_type: String,
        description: String,
        confidence: f64,
        detected_at_cycle: u64,
    },
    /// Submit a robotics telemetry report to the robotics-dispatch zome.
    ///
    /// Targets `mycelix-civic` role, `robotics_dispatch` zome,
    /// `submit_telemetry` extern. Requires both asset_hash and order_hash —
    /// autonomous platforms with no active dispatch order have no telemetry
    /// target and should not emit this command.
    SubmitRoboticsTelemetry {
        correlation_id: u64,
        /// ActionHash of the registered RoboticAsset (raw 39-byte hash).
        asset_hash: Vec<u8>,
        /// ActionHash of the active DispatchOrder (raw 39-byte hash).
        order_hash: Vec<u8>,
        /// Current position (WGS84 lat/lon, meters altitude).
        lat: f64,
        lon: f64,
        alt: f64,
        /// Current Phi / consciousness level.
        consciousness_level: f64,
        /// Safety tier string — "Green"/"Yellow"/"Orange"/"Red".
        safety_level: String,
        /// Mission progress 0.0–1.0.
        mission_progress: f64,
        /// Fuel/battery level 0.0–1.0.
        fuel_level: f64,
        /// Platform name (e.g., "helicopter") — informational, bundled into
        /// `platform_specific` alongside platform-specific bytes.
        platform: String,
        /// Platform-specific serialized telemetry bytes (opaque to the zome).
        platform_specific: Vec<u8>,
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
    TelemetryAccepted {
        correlation_id: u64,
        action_hash: Option<String>,
    },
    TelemetryRejected {
        correlation_id: u64,
        reason: String,
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
            // Matches `mycelix-workspace/happs/happ.yaml` role name.
            governance_role: "governance".to_string(),
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
                meta_awareness,
                coherence,
                care_activation,
                alignment_score,
            } => {
                let now_micros = now_micros_i64();
                let voting_ends_micros = now_micros.saturating_add(7 * 24 * 60 * 60 * 1_000_000);

                // Governance proposals zome expects a full `Proposal` entry (see
                // `mycelix-governance/zomes/proposals/integrity/src/lib.rs`).
                // Timestamp fields are microseconds since UNIX epoch (i64).
                let proposal_id = format!("SYM-{}", correlation_id);
                let title = format!("Symthaea Proposal {}", correlation_id);

                // Preserve Symthaea's richer proposal context in `actions` (stringified JSON).
                let actions = serde_json::json!({
                    "source": "symthaea",
                    "correlation_id": correlation_id,
                    "proposer_did": proposer_did.clone(),
                    "consciousness_phi": consciousness_phi,
                    "meta_awareness": meta_awareness,
                    "coherence": coherence,
                    "care_activation": care_activation,
                    "alignment_score": alignment_score,
                })
                .to_string();

                let payload = serde_json::json!({
                    "id": proposal_id,
                    "title": title,
                    "description": description,
                    "proposal_type": "Standard",
                    "author": proposer_did,
                    "status": "Active",
                    "actions": actions,
                    "discussion_url": null,
                    "voting_starts": now_micros,
                    "voting_ends": voting_ends_micros,
                    "created": now_micros,
                    "updated": now_micros,
                    "version": 1,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(
                        &self.governance_role,
                        "proposals",
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
                consciousness_phi,
                meta_awareness,
                coherence,
                care_activation,
            } => {
                let _ = (
                    consciousness_phi,
                    meta_awareness,
                    coherence,
                    care_activation,
                );
                let reason = if rationale.trim().is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(rationale)
                };

                // Voting zome expects `CastVoteInput`:
                // `{ proposal_id, voter_did, choice: For|Against|Abstain, reason? }`
                let payload = serde_json::json!({
                    "proposal_id": proposal_id,
                    "voter_did": voter_did,
                    "choice": if approve { "For" } else { "Against" },
                    "reason": reason,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(&self.governance_role, "voting", "cast_vote", payload_bytes)
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
                let payload_bytes = rmp_serde::to_vec(&()).unwrap_or_default();
                match self
                    .transport
                    .call_zome(
                        &self.governance_role,
                        "proposals",
                        "get_active_proposals",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(_) => DispatchOutcome::ProposalAccepted {
                        correlation_id: 0,
                        action_hash: None,
                    },
                    Err(reason) => DispatchOutcome::ProposalRejected {
                        correlation_id: 0,
                        reason,
                    },
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

            DispatchCommand::DeclareCrisis {
                correlation_id,
                severity,
                crisis_type,
                description,
                confidence,
                detected_at_cycle,
            } => {
                // Mycelix civic emergency-incidents currently expects `DeclareDisasterInput`,
                // which includes geospatial fields. Symthaea's crisis detector doesn't yet
                // produce reliable geo coordinates, so we publish a transparent placeholder
                // "global/unknown" affected area (0,0 + large radius).
                let id = format!("symthaea-crisis-{}", correlation_id);
                let title = truncate_utf8(&format!("Symthaea Crisis: {}", crisis_type), 256);
                let desc = format!(
                    "{}\n\n[symthaea]\nconfidence={:.3}\ndetected_at_cycle={}\nNOTE: affected_area is a placeholder (global/unknown).",
                    description.trim(),
                    confidence,
                    detected_at_cycle
                );

                let payload = serde_json::json!({
                    "id": id,
                    "disaster_type": disaster_type_value(&crisis_type),
                    "title": title,
                    // Integrity validation caps description at 4096 bytes.
                    "description": truncate_utf8(&desc, 4096),
                    "severity": severity_level_str(severity),
                    "affected_area": {
                        "center_lat": 0.0,
                        "center_lon": 0.0,
                        "radius_km": 20000.0,
                        "boundary": null,
                        "zones": [],
                    },
                    "estimated_affected": 0,
                    "coordination_lead": null,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(
                        "civic",
                        "emergency_incidents",
                        "declare_disaster",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(result) => {
                        let action_hash = String::from_utf8(result).ok();
                        info!(correlation_id, severity, %crisis_type,
                            "Crisis incident declared on Mycelix civic DHT");
                        DispatchOutcome::ProposalAccepted {
                            correlation_id,
                            action_hash,
                        }
                    }
                    Err(reason) => {
                        warn!(correlation_id, severity, %crisis_type, %reason,
                            "Crisis declaration rejected by conductor");
                        DispatchOutcome::ProposalRejected {
                            correlation_id,
                            reason,
                        }
                    }
                }
            }
            DispatchCommand::SubmitRoboticsTelemetry {
                correlation_id,
                asset_hash,
                order_hash,
                lat,
                lon,
                alt,
                consciousness_level,
                safety_level,
                mission_progress,
                fuel_level,
                platform,
                platform_specific,
            } => {
                // Prepend platform name to platform_specific so the zome's
                // opaque-bytes field retains a minimal, self-describing header.
                // Format: [len(u8) | platform_utf8 | caller_bytes]
                let mut tagged = Vec::with_capacity(1 + platform.len() + platform_specific.len());
                let plen = platform.len().min(255) as u8;
                tagged.push(plen);
                tagged.extend_from_slice(&platform.as_bytes()[..plen as usize]);
                tagged.extend_from_slice(&platform_specific);

                let payload = serde_json::json!({
                    "asset_hash": asset_hash,
                    "order_hash": order_hash,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "consciousness_level": consciousness_level,
                    "safety_level": safety_level,
                    "mission_progress": mission_progress,
                    "fuel_level": fuel_level,
                    "platform_specific": tagged,
                });
                let payload_bytes = rmp_serde::to_vec(&payload).unwrap_or_default();

                match self
                    .transport
                    .call_zome(
                        "civic",
                        "robotics_dispatch",
                        "submit_telemetry",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(result) => {
                        let action_hash = String::from_utf8(result).ok();
                        info!(
                            correlation_id,
                            %platform,
                            %safety_level,
                            "Robotics telemetry submitted to Mycelix civic DHT"
                        );
                        DispatchOutcome::TelemetryAccepted {
                            correlation_id,
                            action_hash,
                        }
                    }
                    Err(reason) => {
                        warn!(
                            correlation_id,
                            %platform,
                            %reason,
                            "Robotics telemetry rejected by conductor"
                        );
                        DispatchOutcome::TelemetryRejected {
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
                    DispatchCommand::SubmitProposal { correlation_id, .. }
                    | DispatchCommand::CastVote { correlation_id, .. }
                    | DispatchCommand::EvaluateAsset { correlation_id, .. }
                    | DispatchCommand::DeclareCrisis { correlation_id, .. }
                    | DispatchCommand::SubmitRoboticsTelemetry { correlation_id, .. } => {
                        *correlation_id
                    }
                    DispatchCommand::QueryActiveProposals => 0,
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
                    | DispatchOutcome::TelemetryAccepted { correlation_id, .. }
                    | DispatchOutcome::TelemetryRejected { correlation_id, .. }
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
        unsafe {
            std::env::remove_var("MYCELIX_CONDUCTOR_URL");
        }
        assert!(ConductorConfig::from_env().is_none());
    }

    #[test]
    fn dispatch_command_serde_roundtrip() {
        let cmd = DispatchCommand::SubmitProposal {
            correlation_id: 42,
            description: "Test proposal".into(),
            proposer_did: "did:mycelix:test".into(),
            consciousness_phi: 0.8,
            meta_awareness: 0.6,
            coherence: 0.7,
            care_activation: 0.5,
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
            .call_zome("governance", "proposals", "create_proposal", vec![])
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
            meta_awareness: 0.6,
            coherence: 0.7,
            care_activation: 0.5,
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
            consciousness_phi: 0.75,
            meta_awareness: 0.6,
            coherence: 0.7,
            care_activation: 0.5,
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
                meta_awareness: 0.4,
                coherence: 0.3,
                care_activation: 0.4,
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

    #[tokio::test]
    async fn mock_dispatcher_submit_robotics_telemetry() {
        let mut dispatcher = GovernanceDispatcher::new(MockTransport);

        let cmd = DispatchCommand::SubmitRoboticsTelemetry {
            correlation_id: 4242,
            asset_hash: vec![0x84, 0x21, 0x24, 0x00],
            order_hash: vec![0x84, 0x21, 0x24, 0x01],
            lat: 40.7128,
            lon: -74.0060,
            alt: 1200.0,
            consciousness_level: 0.78,
            safety_level: "Green".into(),
            mission_progress: 0.25,
            fuel_level: 0.88,
            platform: "helicopter".into(),
            platform_specific: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };

        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::TelemetryAccepted { correlation_id, .. } => {
                assert_eq!(correlation_id, 4242);
            }
            _ => panic!("Expected TelemetryAccepted, got {:?}", outcome),
        }
    }

    #[test]
    fn telemetry_command_serde_roundtrip() {
        let cmd = DispatchCommand::SubmitRoboticsTelemetry {
            correlation_id: 7,
            asset_hash: vec![1, 2, 3],
            order_hash: vec![4, 5, 6],
            lat: 1.5,
            lon: -2.5,
            alt: 100.0,
            consciousness_level: 0.65,
            safety_level: "Yellow".into(),
            mission_progress: 0.5,
            fuel_level: 0.4,
            platform: "helicopter".into(),
            platform_specific: vec![9, 9, 9],
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let decoded: DispatchCommand = serde_json::from_str(&json).unwrap();
        match decoded {
            DispatchCommand::SubmitRoboticsTelemetry {
                correlation_id,
                platform,
                ..
            } => {
                assert_eq!(correlation_id, 7);
                assert_eq!(platform, "helicopter");
            }
            _ => panic!("wrong variant"),
        }
    }
}
