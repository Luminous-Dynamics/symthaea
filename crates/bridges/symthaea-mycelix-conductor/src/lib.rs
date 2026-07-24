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
    /// Publish a memetic-pathogen ruleset to the `identity` role's
    /// `ruleset_registry` zome (`WARDED_NODE_DESIGN_2026-07-11.md` Phase 5b:
    /// federated marketplace, no canonical publisher — the zome itself
    /// structurally has no "canonical" concept, and this command doesn't
    /// either). `publisher` is deliberately absent: the zome sets it from
    /// the calling agent's own `agent_info()`, so it can never be forged by
    /// supplying someone else's identity here.
    PublishRuleset {
        correlation_id: u64,
        name: String,
        version: String,
        source: String,
        entries: Vec<RulesetEntryPayload>,
    },
    /// Fetch every published ruleset (the "browse the marketplace" call —
    /// deliberately unfiltered; the caller decides who to trust, this call
    /// doesn't pre-judge).
    GetAllRulesets {
        correlation_id: u64,
        limit: u32,
    },
    /// Fetch every ruleset a specific publisher has ever published.
    GetRulesetsByPublisher {
        correlation_id: u64,
        /// Raw `AgentPubKey` bytes (this crate doesn't depend on
        /// `holo_hash`/`holochain_client` — see module docs — so publisher
        /// identity crosses this boundary as opaque bytes, same convention
        /// as `asset_hash`/`order_hash` above).
        publisher: Vec<u8>,
    },
}

/// One pathogen signature within a ruleset — mirrors
/// `ruleset_registry_integrity::RulesetEntryRecord` /
/// `symthaea_memetics::RulesetEntry`. Deliberately a separate, duplicated
/// type (not an import) — see this file's "Types mirrored from
/// mycelix_bridge.rs" note above: this crate avoids depending on either
/// `symthaea-memetics` or the Mycelix zome crates, to keep the dependency
/// graph one-directional (cognitive loop → this bridge → transport).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulesetEntryPayload {
    pub signature: Vec<u8>,
    pub description: String,
}

const HOLOCHAIN_HASH_BYTES: usize = 39;
const MAX_PLATFORM_SPECIFIC_BYTES: usize = 64 * 1024;
const MAX_RULESET_ENTRIES: usize = 1024;
const MAX_RULESET_SIGNATURE_BYTES: usize = 64 * 1024;
const MAX_RULESET_FETCH_LIMIT: u32 = 1000;
const MAX_RULESET_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

fn require_text(name: &str, value: &str, max_bytes: usize) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{name} cannot be empty"));
    }
    if value.len() > max_bytes {
        return Err(format!("{name} exceeds {max_bytes} bytes"));
    }
    Ok(())
}

fn require_optional_text(name: &str, value: &str, max_bytes: usize) -> Result<(), String> {
    if value.len() > max_bytes {
        return Err(format!("{name} exceeds {max_bytes} bytes"));
    }
    Ok(())
}

fn require_unit_interval(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{name} must be a finite value in [0, 1]"));
    }
    Ok(())
}

fn require_hash(name: &str, value: &[u8]) -> Result<(), String> {
    if value.len() != HOLOCHAIN_HASH_BYTES {
        return Err(format!(
            "{name} must contain exactly {HOLOCHAIN_HASH_BYTES} raw Holochain hash bytes"
        ));
    }
    Ok(())
}

fn encode_msgpack<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(value).map_err(|error| format!("failed to encode conductor payload: {error}"))
}

fn validate_ruleset_entry(entry: &RulesetEntryPayload) -> Result<(), String> {
    if entry.signature.is_empty() || entry.signature.len() > MAX_RULESET_SIGNATURE_BYTES {
        return Err(format!(
            "ruleset signatures must contain 1 to {MAX_RULESET_SIGNATURE_BYTES} bytes"
        ));
    }
    require_text("ruleset entry description", &entry.description, 4096)
}

impl DispatchCommand {
    /// Correlation id used for logging and rejection outcomes.
    pub fn correlation_id(&self) -> u64 {
        match self {
            Self::SubmitProposal { correlation_id, .. }
            | Self::CastVote { correlation_id, .. }
            | Self::EvaluateAsset { correlation_id, .. }
            | Self::DeclareCrisis { correlation_id, .. }
            | Self::SubmitRoboticsTelemetry { correlation_id, .. }
            | Self::PublishRuleset { correlation_id, .. }
            | Self::GetAllRulesets { correlation_id, .. }
            | Self::GetRulesetsByPublisher { correlation_id, .. } => *correlation_id,
            Self::QueryActiveProposals => 0,
        }
    }

    /// Validate a command before serialization or any conductor call.
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self, Self::QueryActiveProposals) && self.correlation_id() == 0 {
            return Err("correlation_id must be non-zero".into());
        }

        match self {
            Self::SubmitProposal {
                description,
                proposer_did,
                consciousness_phi,
                meta_awareness,
                coherence,
                care_activation,
                alignment_score,
                ..
            } => {
                require_text("description", description, 4096)?;
                require_text("proposer_did", proposer_did, 256)?;
                require_unit_interval("consciousness_phi", *consciousness_phi)?;
                require_unit_interval("meta_awareness", *meta_awareness)?;
                require_unit_interval("coherence", *coherence)?;
                require_unit_interval("care_activation", *care_activation)?;
                require_unit_interval("alignment_score", *alignment_score)
            }
            Self::CastVote {
                proposal_id,
                voter_did,
                rationale,
                consciousness_phi,
                meta_awareness,
                coherence,
                care_activation,
                ..
            } => {
                require_text("proposal_id", proposal_id, 512)?;
                require_text("voter_did", voter_did, 256)?;
                require_optional_text("rationale", rationale, 4096)?;
                require_unit_interval("consciousness_phi", *consciousness_phi)?;
                require_unit_interval("meta_awareness", *meta_awareness)?;
                require_unit_interval("coherence", *coherence)?;
                require_unit_interval("care_activation", *care_activation)
            }
            Self::QueryActiveProposals => Ok(()),
            Self::EvaluateAsset {
                project_id,
                phi_score,
                harmony_alignment,
                per_harmony_scores,
                care_activation,
                meta_awareness,
                ..
            } => {
                require_text("project_id", project_id, 256)?;
                require_text("per_harmony_scores", per_harmony_scores, 16 * 1024)?;
                require_unit_interval("phi_score", *phi_score)?;
                require_unit_interval("harmony_alignment", *harmony_alignment)?;
                require_unit_interval("care_activation", *care_activation)?;
                require_unit_interval("meta_awareness", *meta_awareness)
            }
            Self::DeclareCrisis {
                severity,
                crisis_type,
                description,
                confidence,
                ..
            } => {
                if !(1..=5).contains(severity) {
                    return Err("severity must be in [1, 5]".into());
                }
                require_text("crisis_type", crisis_type, 128)?;
                require_text("description", description, 4096)?;
                require_unit_interval("confidence", *confidence)
            }
            Self::SubmitRoboticsTelemetry {
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
                ..
            } => {
                require_hash("asset_hash", asset_hash)?;
                require_hash("order_hash", order_hash)?;
                if !lat.is_finite() || !(-90.0..=90.0).contains(lat) {
                    return Err("lat must be a finite WGS84 latitude".into());
                }
                if !lon.is_finite() || !(-180.0..=180.0).contains(lon) {
                    return Err("lon must be a finite WGS84 longitude".into());
                }
                if !alt.is_finite() {
                    return Err("alt must be finite".into());
                }
                require_unit_interval("consciousness_level", *consciousness_level)?;
                require_unit_interval("mission_progress", *mission_progress)?;
                require_unit_interval("fuel_level", *fuel_level)?;
                if !matches!(safety_level.as_str(), "Green" | "Yellow" | "Orange" | "Red") {
                    return Err("safety_level must be Green, Yellow, Orange, or Red".into());
                }
                require_text("platform", platform, u8::MAX as usize)?;
                if platform_specific.len() > MAX_PLATFORM_SPECIFIC_BYTES {
                    return Err(format!(
                        "platform_specific exceeds {MAX_PLATFORM_SPECIFIC_BYTES} bytes"
                    ));
                }
                Ok(())
            }
            Self::PublishRuleset {
                name,
                version,
                source,
                entries,
                ..
            } => {
                require_text("ruleset name", name, 256)?;
                require_text("ruleset version", version, 64)?;
                require_text("ruleset source", source, 4096)?;
                if entries.len() > MAX_RULESET_ENTRIES {
                    return Err(format!(
                        "ruleset contains more than {MAX_RULESET_ENTRIES} entries"
                    ));
                }
                for entry in entries {
                    validate_ruleset_entry(entry)?;
                }
                Ok(())
            }
            Self::GetAllRulesets { limit, .. } => {
                if *limit == 0 || *limit > MAX_RULESET_FETCH_LIMIT {
                    return Err(format!(
                        "ruleset fetch limit must be in [1, {MAX_RULESET_FETCH_LIMIT}]"
                    ));
                }
                Ok(())
            }
            Self::GetRulesetsByPublisher { publisher, .. } => require_hash("publisher", publisher),
        }
    }

    fn rejection(&self, reason: String) -> DispatchOutcome {
        let correlation_id = self.correlation_id();
        match self {
            Self::SubmitProposal { .. }
            | Self::EvaluateAsset { .. }
            | Self::DeclareCrisis { .. }
            | Self::QueryActiveProposals => DispatchOutcome::ProposalRejected {
                correlation_id,
                reason,
            },
            Self::CastVote { .. } => DispatchOutcome::VoteRejected {
                correlation_id,
                reason,
            },
            Self::SubmitRoboticsTelemetry { .. } => DispatchOutcome::TelemetryRejected {
                correlation_id,
                reason,
            },
            Self::PublishRuleset { .. } => DispatchOutcome::RulesetPublishRejected {
                correlation_id,
                reason,
            },
            Self::GetAllRulesets { .. } | Self::GetRulesetsByPublisher { .. } => {
                DispatchOutcome::RulesetFetchFailed {
                    correlation_id,
                    reason,
                }
            }
        }
    }
}

/// One published ruleset as returned by `GetAllRulesets`/
/// `GetRulesetsByPublisher` — mirrors
/// `ruleset_registry_integrity::RulesetRecord`. The `ConductorTransport`
/// implementation is responsible for unwrapping the Holochain `Record`
/// envelope and re-serializing just this shape (see `dispatch()`'s
/// `GetAllRulesets` arm doc comment for why that boundary is drawn there).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulesetPayload {
    pub publisher: Vec<u8>,
    pub name: String,
    pub version: String,
    pub source: String,
    pub entries: Vec<RulesetEntryPayload>,
}

fn decode_ruleset_response(
    response: &[u8],
    max_rulesets: usize,
    expected_publisher: Option<&[u8]>,
) -> Result<Vec<RulesetPayload>, String> {
    if response.len() > MAX_RULESET_RESPONSE_BYTES {
        return Err(format!(
            "response exceeds the {MAX_RULESET_RESPONSE_BYTES} byte safety limit"
        ));
    }
    let rulesets: Vec<RulesetPayload> =
        rmp_serde::from_slice(response).map_err(|error| format!("malformed response: {error}"))?;
    if rulesets.len() > max_rulesets {
        return Err(format!(
            "response contains {} rulesets, exceeding the requested limit of {max_rulesets}",
            rulesets.len()
        ));
    }
    for ruleset in &rulesets {
        require_hash("ruleset publisher", &ruleset.publisher)?;
        if expected_publisher.is_some_and(|publisher| publisher != ruleset.publisher.as_slice()) {
            return Err("response contains a ruleset from a different publisher".into());
        }
        require_text("ruleset name", &ruleset.name, 256)?;
        require_text("ruleset version", &ruleset.version, 64)?;
        require_text("ruleset source", &ruleset.source, 4096)?;
        if ruleset.entries.len() > MAX_RULESET_ENTRIES {
            return Err(format!(
                "returned ruleset contains more than {MAX_RULESET_ENTRIES} entries"
            ));
        }
        for entry in &ruleset.entries {
            validate_ruleset_entry(entry)?;
        }
    }
    Ok(rulesets)
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
    RulesetPublished {
        correlation_id: u64,
        action_hash: Option<String>,
    },
    RulesetPublishRejected {
        correlation_id: u64,
        reason: String,
    },
    RulesetsFetched {
        correlation_id: u64,
        rulesets: Vec<RulesetPayload>,
    },
    RulesetFetchFailed {
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
    /// Hard wall-clock limit around each complete dispatch operation.
    call_timeout: Duration,
}

const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(30);

impl<T: ConductorTransport> GovernanceDispatcher<T> {
    /// Create a new dispatcher targeting the governance role.
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            // Matches `mycelix-workspace/happs/happ.yaml` role name.
            governance_role: "governance".to_string(),
            call_timeout: DEFAULT_CALL_TIMEOUT,
        }
    }

    /// Create with a custom governance role name.
    pub fn with_role(transport: T, role: impl Into<String>) -> Self {
        Self {
            transport,
            governance_role: role.into(),
            call_timeout: DEFAULT_CALL_TIMEOUT,
        }
    }

    /// Override the hard wall-clock limit for each dispatch operation.
    pub fn with_call_timeout(mut self, call_timeout: Duration) -> Self {
        self.call_timeout = if call_timeout.is_zero() {
            Duration::from_millis(1)
        } else {
            call_timeout
        };
        self
    }

    /// Dispatch a single command within the configured wall-clock limit.
    pub async fn dispatch(&mut self, cmd: DispatchCommand) -> DispatchOutcome {
        let correlation_id = cmd.correlation_id();
        match tokio::time::timeout(self.call_timeout, self.dispatch_inner(cmd)).await {
            Ok(outcome) => outcome,
            Err(_) => {
                warn!(
                    correlation_id,
                    timeout_ms = %self.call_timeout.as_millis(),
                    "Conductor dispatch timed out"
                );
                DispatchOutcome::Timeout { correlation_id }
            }
        }
    }

    async fn dispatch_inner(&mut self, cmd: DispatchCommand) -> DispatchOutcome {
        if let Err(reason) = cmd.validate() {
            warn!(
                correlation_id = cmd.correlation_id(),
                %reason,
                "Rejected invalid conductor command"
            );
            return cmd.rejection(reason);
        }

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
                let payload_bytes = match encode_msgpack(&payload) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::ProposalRejected {
                            correlation_id,
                            reason,
                        };
                    }
                };

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
                let payload_bytes = match encode_msgpack(&payload) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::VoteRejected {
                            correlation_id,
                            reason,
                        };
                    }
                };

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
                let payload_bytes = match encode_msgpack(&()) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::ProposalRejected {
                            correlation_id: 0,
                            reason,
                        };
                    }
                };
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
                let payload_bytes = match encode_msgpack(&payload) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::ProposalRejected {
                            correlation_id,
                            reason,
                        };
                    }
                };

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
                let payload_bytes = match encode_msgpack(&payload) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::ProposalRejected {
                            correlation_id,
                            reason,
                        };
                    }
                };

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
                let plen = platform.len() as u8;
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
                let payload_bytes = match encode_msgpack(&payload) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::TelemetryRejected {
                            correlation_id,
                            reason,
                        };
                    }
                };

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

            DispatchCommand::PublishRuleset {
                correlation_id,
                name,
                version,
                source,
                entries,
            } => {
                // Mirrors ruleset_registry's PublishRulesetInput exactly
                // (publisher is deliberately absent — the zome derives it
                // from agent_info(), see the enum variant's doc comment).
                let payload = serde_json::json!({
                    "name": name,
                    "version": version,
                    "source": source,
                    "entries": entries
                        .iter()
                        .map(|e| serde_json::json!({
                            "signature": e.signature,
                            "description": e.description,
                        }))
                        .collect::<Vec<_>>(),
                });
                let payload_bytes = match encode_msgpack(&payload) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::RulesetPublishRejected {
                            correlation_id,
                            reason,
                        };
                    }
                };

                match self
                    .transport
                    .call_zome(
                        "identity",
                        "ruleset_registry",
                        "publish_ruleset",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(result) => {
                        let action_hash = String::from_utf8(result).ok();
                        info!(correlation_id, %name, entry_count = entries.len(), "Ruleset published to Mycelix identity DHT");
                        DispatchOutcome::RulesetPublished {
                            correlation_id,
                            action_hash,
                        }
                    }
                    Err(reason) => {
                        warn!(correlation_id, %name, %reason, "Ruleset publish rejected by conductor");
                        DispatchOutcome::RulesetPublishRejected {
                            correlation_id,
                            reason,
                        }
                    }
                }
            }

            DispatchCommand::GetAllRulesets {
                correlation_id,
                limit,
            } => {
                let payload_bytes = match encode_msgpack(&limit) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::RulesetFetchFailed {
                            correlation_id,
                            reason,
                        };
                    }
                };

                match self
                    .transport
                    .call_zome(
                        "identity",
                        "ruleset_registry",
                        "get_all_rulesets",
                        payload_bytes,
                    )
                    .await
                {
                    // The transport is expected to have already unwrapped
                    // Holochain's Record envelope and re-serialized just the
                    // RulesetRecord shape (see RulesetPayload's doc comment)
                    // — this dispatcher stays free of Holochain-native types.
                    Ok(result) => match decode_ruleset_response(&result, limit as usize, None) {
                        Ok(rulesets) => {
                            info!(
                                correlation_id,
                                count = rulesets.len(),
                                "Fetched rulesets from Mycelix identity DHT"
                            );
                            DispatchOutcome::RulesetsFetched {
                                correlation_id,
                                rulesets,
                            }
                        }
                        Err(reason) => {
                            warn!(correlation_id, %reason, "Ruleset fetch response rejected");
                            DispatchOutcome::RulesetFetchFailed {
                                correlation_id,
                                reason,
                            }
                        }
                    },
                    Err(reason) => {
                        warn!(correlation_id, %reason, "Ruleset fetch rejected by conductor");
                        DispatchOutcome::RulesetFetchFailed {
                            correlation_id,
                            reason,
                        }
                    }
                }
            }

            DispatchCommand::GetRulesetsByPublisher {
                correlation_id,
                publisher,
            } => {
                let payload_bytes = match encode_msgpack(&publisher) {
                    Ok(payload) => payload,
                    Err(reason) => {
                        return DispatchOutcome::RulesetFetchFailed {
                            correlation_id,
                            reason,
                        };
                    }
                };

                match self
                    .transport
                    .call_zome(
                        "identity",
                        "ruleset_registry",
                        "get_rulesets_by_publisher",
                        payload_bytes,
                    )
                    .await
                {
                    Ok(result) => match decode_ruleset_response(
                        &result,
                        MAX_RULESET_FETCH_LIMIT as usize,
                        Some(&publisher),
                    ) {
                        Ok(rulesets) => {
                            info!(
                                correlation_id,
                                count = rulesets.len(),
                                "Fetched publisher's rulesets from Mycelix identity DHT"
                            );
                            DispatchOutcome::RulesetsFetched {
                                correlation_id,
                                rulesets,
                            }
                        }
                        Err(reason) => {
                            warn!(correlation_id, %reason, "Ruleset fetch response rejected");
                            DispatchOutcome::RulesetFetchFailed {
                                correlation_id,
                                reason,
                            }
                        }
                    },
                    Err(reason) => {
                        warn!(correlation_id, %reason, "Ruleset fetch rejected by conductor");
                        DispatchOutcome::RulesetFetchFailed {
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
    /// Every command is bounded by the dispatcher's configured call timeout.
    pub async fn run_dispatch_loop(
        mut self,
        rx: std::sync::mpsc::Receiver<DispatchCommand>,
        outcome_tx: tokio::sync::mpsc::Sender<DispatchOutcome>,
    ) {
        info!("Governance dispatch loop started");

        loop {
            while let Ok(cmd) = rx.try_recv() {
                let outcome = self.dispatch(cmd).await;
                if outcome_tx.send(outcome).await.is_err() {
                    warn!("Outcome channel closed, stopping dispatch loop");
                    return;
                }
            }

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
    async fn hung_transport_produces_real_timeout_outcome() {
        struct HangingTransport;
        #[async_trait::async_trait]
        impl ConductorTransport for HangingTransport {
            async fn call_zome(
                &mut self,
                _: &str,
                _: &str,
                _: &str,
                _: Vec<u8>,
            ) -> Result<Vec<u8>, String> {
                std::future::pending().await
            }
            fn is_connected(&self) -> bool {
                true
            }
        }

        let mut dispatcher = GovernanceDispatcher::new(HangingTransport)
            .with_call_timeout(Duration::from_millis(20));
        let command = DispatchCommand::SubmitProposal {
            correlation_id: 101,
            description: "timeout test".into(),
            proposer_did: "did:mycelix:test".into(),
            consciousness_phi: 0.5,
            meta_awareness: 0.5,
            coherence: 0.5,
            care_activation: 0.5,
            alignment_score: 0.5,
        };
        let started = std::time::Instant::now();
        let outcome = dispatcher.dispatch(command).await;

        assert!(matches!(
            outcome,
            DispatchOutcome::Timeout {
                correlation_id: 101
            }
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
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
    async fn dispatch_loop_forwards_immediate_rejection() {
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
            asset_hash: vec![0x84; HOLOCHAIN_HASH_BYTES],
            order_hash: vec![0x21; HOLOCHAIN_HASH_BYTES],
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

    #[tokio::test]
    async fn invalid_commands_are_rejected_before_transport() {
        struct PanicTransport;
        #[async_trait::async_trait]
        impl ConductorTransport for PanicTransport {
            async fn call_zome(
                &mut self,
                _: &str,
                _: &str,
                _: &str,
                _: Vec<u8>,
            ) -> Result<Vec<u8>, String> {
                panic!("invalid commands must not reach the conductor transport")
            }
            fn is_connected(&self) -> bool {
                true
            }
        }

        let mut dispatcher = GovernanceDispatcher::new(PanicTransport);
        let proposal = DispatchCommand::SubmitProposal {
            correlation_id: 1,
            description: "invalid score".into(),
            proposer_did: "did:mycelix:test".into(),
            consciousness_phi: f64::NAN,
            meta_awareness: 0.5,
            coherence: 0.5,
            care_activation: 0.5,
            alignment_score: 0.5,
        };
        assert!(matches!(
            dispatcher.dispatch(proposal).await,
            DispatchOutcome::ProposalRejected { .. }
        ));

        let telemetry = DispatchCommand::SubmitRoboticsTelemetry {
            correlation_id: 2,
            asset_hash: vec![0; 4],
            order_hash: vec![0; HOLOCHAIN_HASH_BYTES],
            lat: 0.0,
            lon: 0.0,
            alt: 0.0,
            consciousness_level: 0.5,
            safety_level: "Green".into(),
            mission_progress: 0.5,
            fuel_level: 0.5,
            platform: "rover".into(),
            platform_specific: vec![],
        };
        assert!(matches!(
            dispatcher.dispatch(telemetry).await,
            DispatchOutcome::TelemetryRejected { .. }
        ));

        let fetch = DispatchCommand::GetAllRulesets {
            correlation_id: 3,
            limit: 0,
        };
        assert!(matches!(
            dispatcher.dispatch(fetch).await,
            DispatchOutcome::RulesetFetchFailed { .. }
        ));
    }

    #[test]
    fn telemetry_command_serde_roundtrip() {
        let cmd = DispatchCommand::SubmitRoboticsTelemetry {
            correlation_id: 7,
            asset_hash: vec![1; HOLOCHAIN_HASH_BYTES],
            order_hash: vec![2; HOLOCHAIN_HASH_BYTES],
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

    // ── Ruleset registry commands (Warded Node design Phase 5b) ──

    #[test]
    fn publish_ruleset_command_serde_roundtrip() {
        let cmd = DispatchCommand::PublishRuleset {
            correlation_id: 55,
            name: "family-safety-baseline".into(),
            version: "2026.07.11".into(),
            source: "test-fixture".into(),
            entries: vec![RulesetEntryPayload {
                signature: vec![0u8; 2048],
                description: "known pattern".into(),
            }],
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let decoded: DispatchCommand = serde_json::from_str(&json).unwrap();
        match decoded {
            DispatchCommand::PublishRuleset {
                correlation_id,
                name,
                entries,
                ..
            } => {
                assert_eq!(correlation_id, 55);
                assert_eq!(name, "family-safety-baseline");
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].signature.len(), 2048);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn ruleset_payload_msgpack_roundtrip() {
        // Proves RulesetPayload survives the SAME wire format (rmp_serde)
        // dispatch() actually uses for fetch responses, not just JSON.
        let payload = vec![RulesetPayload {
            publisher: vec![1; HOLOCHAIN_HASH_BYTES],
            name: "n".into(),
            version: "v".into(),
            source: "s".into(),
            entries: vec![RulesetEntryPayload {
                signature: vec![7u8; 16],
                description: "d".into(),
            }],
        }];
        let bytes = rmp_serde::to_vec(&payload).unwrap();
        let decoded: Vec<RulesetPayload> = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].name, "n");
        assert_eq!(decoded[0].entries[0].signature, vec![7u8; 16]);
    }

    #[tokio::test]
    async fn mock_dispatcher_publish_ruleset() {
        let mut dispatcher = GovernanceDispatcher::new(MockTransport);
        let cmd = DispatchCommand::PublishRuleset {
            correlation_id: 300,
            name: "test-ruleset".into(),
            version: "1.0".into(),
            source: "test".into(),
            entries: vec![],
        };
        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::RulesetPublished { correlation_id, .. } => {
                assert_eq!(correlation_id, 300);
            }
            _ => panic!("Expected RulesetPublished, got {:?}", outcome),
        }
    }

    #[tokio::test]
    async fn mock_dispatcher_get_all_rulesets_malformed_response() {
        // MockTransport returns Ok(vec![]) — empty bytes are not valid
        // msgpack for Vec<RulesetPayload>, so this must fail GRACEFULLY
        // (RulesetFetchFailed), not panic or silently return an empty Vec
        // that looks indistinguishable from "no rulesets published yet".
        let mut dispatcher = GovernanceDispatcher::new(MockTransport);
        let cmd = DispatchCommand::GetAllRulesets {
            correlation_id: 400,
            limit: 50,
        };
        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::RulesetFetchFailed { correlation_id, .. } => {
                assert_eq!(correlation_id, 400);
            }
            _ => panic!(
                "Expected RulesetFetchFailed for a malformed response, got {:?}",
                outcome
            ),
        }
    }

    #[tokio::test]
    async fn mock_dispatcher_get_all_rulesets_success() {
        // A transport that returns a real msgpack-encoded Vec<RulesetPayload>,
        // proving the successful-parse path end to end.
        struct RulesetTransport;
        #[async_trait::async_trait]
        impl ConductorTransport for RulesetTransport {
            async fn call_zome(
                &mut self,
                role_name: &str,
                zome_name: &str,
                fn_name: &str,
                _payload: Vec<u8>,
            ) -> Result<Vec<u8>, String> {
                assert_eq!(role_name, "identity");
                assert_eq!(zome_name, "ruleset_registry");
                assert_eq!(fn_name, "get_all_rulesets");
                let rulesets = vec![RulesetPayload {
                    publisher: vec![9; HOLOCHAIN_HASH_BYTES],
                    name: "family-safety-baseline".into(),
                    version: "1.0".into(),
                    source: "nonprofit".into(),
                    entries: vec![RulesetEntryPayload {
                        signature: vec![1u8; 32],
                        description: "known bad pattern".into(),
                    }],
                }];
                Ok(rmp_serde::to_vec(&rulesets).unwrap())
            }
            fn is_connected(&self) -> bool {
                true
            }
        }

        let mut dispatcher = GovernanceDispatcher::new(RulesetTransport);
        let cmd = DispatchCommand::GetAllRulesets {
            correlation_id: 500,
            limit: 10,
        };
        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::RulesetsFetched {
                correlation_id,
                rulesets,
            } => {
                assert_eq!(correlation_id, 500);
                assert_eq!(rulesets.len(), 1);
                assert_eq!(rulesets[0].name, "family-safety-baseline");
                assert_eq!(rulesets[0].entries[0].signature.len(), 32);
            }
            _ => panic!("Expected RulesetsFetched, got {:?}", outcome),
        }
    }

    #[test]
    fn ruleset_response_is_count_bounded_and_publisher_bound() {
        let ruleset = RulesetPayload {
            publisher: vec![1; HOLOCHAIN_HASH_BYTES],
            name: "bounded".into(),
            version: "1.0".into(),
            source: "test".into(),
            entries: vec![RulesetEntryPayload {
                signature: vec![7; 32],
                description: "known pattern".into(),
            }],
        };

        let two = rmp_serde::to_vec(&vec![ruleset.clone(), ruleset.clone()]).unwrap();
        assert!(decode_ruleset_response(&two, 1, None).is_err());

        let one = rmp_serde::to_vec(&vec![ruleset]).unwrap();
        let different_publisher = vec![2; HOLOCHAIN_HASH_BYTES];
        assert!(decode_ruleset_response(&one, 1, Some(&different_publisher)).is_err());
    }

    #[tokio::test]
    async fn mock_dispatcher_get_rulesets_by_publisher_targets_correct_zome_call() {
        struct AssertingTransport;
        #[async_trait::async_trait]
        impl ConductorTransport for AssertingTransport {
            async fn call_zome(
                &mut self,
                role_name: &str,
                zome_name: &str,
                fn_name: &str,
                payload: Vec<u8>,
            ) -> Result<Vec<u8>, String> {
                assert_eq!(role_name, "identity");
                assert_eq!(zome_name, "ruleset_registry");
                assert_eq!(fn_name, "get_rulesets_by_publisher");
                let decoded: Vec<u8> = rmp_serde::from_slice(&payload).unwrap();
                assert_eq!(decoded, vec![4; HOLOCHAIN_HASH_BYTES]);
                Ok(rmp_serde::to_vec(&Vec::<RulesetPayload>::new()).unwrap())
            }
            fn is_connected(&self) -> bool {
                true
            }
        }

        let mut dispatcher = GovernanceDispatcher::new(AssertingTransport);
        let cmd = DispatchCommand::GetRulesetsByPublisher {
            correlation_id: 600,
            publisher: vec![4; HOLOCHAIN_HASH_BYTES],
        };
        let outcome = dispatcher.dispatch(cmd).await;
        match outcome {
            DispatchOutcome::RulesetsFetched {
                correlation_id,
                rulesets,
            } => {
                assert_eq!(correlation_id, 600);
                assert!(rulesets.is_empty());
            }
            _ => panic!("Expected RulesetsFetched, got {:?}", outcome),
        }
    }
}
