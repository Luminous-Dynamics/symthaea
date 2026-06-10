// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # REST API for Agent Management
//!
//! HTTP API endpoints for managing AI agents, querying K-Vector history,
//! and monitoring agent health.
//!
//! ## Endpoints
//!
//! - `GET /agents` - List all agents
//! - `GET /agents/:id` - Get agent by ID
//! - `POST /agents` - Create new agent
//! - `PUT /agents/:id` - Update agent
//! - `DELETE /agents/:id` - Delete agent
//! - `GET /agents/:id/kvector/history` - Get K-Vector history
//! - `GET /agents/:id/events` - Get agent events
//! - `GET /monitoring/dashboard` - Get dashboard summary
//! - `GET /monitoring/alerts` - Get active alerts

use super::{
    monitoring::{AgentAlert, AlertThresholds, DashboardSummary, MonitoringEngine},
    persistence::{
        AgentEvent, AgentRepository, AgentStatistics, EventLogEntry, KVectorSnapshot,
        MemoryStorageBackend, PersistenceError,
    },
    AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats, InstrumentalActor, KVector,
    UncertaintyCalibration,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// =============================================================================
// API Types
// =============================================================================

/// API error response
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ApiError {
    /// Error code
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ApiError {
    /// Create a "not found" error for an agent ID
    pub fn not_found(id: &str) -> Self {
        Self {
            code: "NOT_FOUND".to_string(),
            message: format!("Agent not found: {}", id),
            details: None,
        }
    }

    /// Create an "already exists" error for an agent ID
    pub fn already_exists(id: &str) -> Self {
        Self {
            code: "ALREADY_EXISTS".to_string(),
            message: format!("Agent already exists: {}", id),
            details: None,
        }
    }

    /// Create a validation error with custom message
    pub fn validation_error(msg: &str) -> Self {
        Self {
            code: "VALIDATION_ERROR".to_string(),
            message: msg.to_string(),
            details: None,
        }
    }

    /// Create an internal error with custom message
    pub fn internal_error(msg: &str) -> Self {
        Self {
            code: "INTERNAL_ERROR".to_string(),
            message: msg.to_string(),
            details: None,
        }
    }

    /// Create an escalation-related error
    pub fn escalation_error(msg: &str) -> Self {
        Self {
            code: "ESCALATION_ERROR".to_string(),
            message: msg.to_string(),
            details: None,
        }
    }

    /// Create a rate limit error
    pub fn rate_limit_error(msg: &str) -> Self {
        Self {
            code: "RATE_LIMIT".to_string(),
            message: msg.to_string(),
            details: None,
        }
    }

    /// Create a forbidden error (unauthorized action)
    pub fn forbidden(msg: &str) -> Self {
        Self {
            code: "FORBIDDEN".to_string(),
            message: msg.to_string(),
            details: None,
        }
    }
}

impl From<PersistenceError> for ApiError {
    fn from(e: PersistenceError) -> Self {
        match e {
            PersistenceError::NotFound(id) => Self::not_found(&id),
            PersistenceError::AlreadyExists(id) => Self::already_exists(&id),
            PersistenceError::SerializationError(msg) => Self::internal_error(&msg),
            PersistenceError::StorageError(msg) => Self::internal_error(&msg),
            PersistenceError::QueryError(msg) => Self::internal_error(&msg),
            PersistenceError::LockPoisoned(msg) => Self::internal_error(&msg),
        }
    }
}

/// API result type
pub type ApiResult<T> = Result<T, ApiError>;

/// Request to create a new agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CreateAgentRequest {
    /// Agent ID (optional - will be generated if not provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Sponsor DID
    pub sponsor_did: String,
    /// Agent class
    pub agent_class: AgentClass,
    /// Initial KREDIT balance
    #[serde(default = "default_kredit_balance")]
    pub initial_kredit: i64,
    /// KREDIT cap
    #[serde(default = "default_kredit_cap")]
    pub kredit_cap: u64,
}

fn default_kredit_balance() -> i64 {
    5000
}
fn default_kredit_cap() -> u64 {
    10000
}

// =============================================================================
// Input Validation
// =============================================================================

/// Maximum length for sponsor DID
const MAX_SPONSOR_DID_LENGTH: usize = 256;

/// Maximum length for agent ID
const MAX_AGENT_ID_LENGTH: usize = 128;

/// Maximum KREDIT cap allowed
const MAX_KREDIT_CAP: u64 = 1_000_000_000;

/// Minimum KREDIT cap allowed
const MIN_KREDIT_CAP: u64 = 100;

impl CreateAgentRequest {
    /// Validate the create agent request
    pub fn validate(&self) -> Result<(), ApiError> {
        // Validate sponsor_did
        if self.sponsor_did.is_empty() {
            return Err(ApiError::validation_error("sponsor_did cannot be empty"));
        }
        if self.sponsor_did.len() > MAX_SPONSOR_DID_LENGTH {
            return Err(ApiError::validation_error(&format!(
                "sponsor_did exceeds maximum length of {} characters",
                MAX_SPONSOR_DID_LENGTH
            )));
        }
        if !self.sponsor_did.starts_with("did:") {
            return Err(ApiError::validation_error(
                "sponsor_did must be a valid DID (start with 'did:')",
            ));
        }

        // Validate agent_id if provided
        if let Some(ref id) = self.agent_id {
            if id.is_empty() {
                return Err(ApiError::validation_error(
                    "agent_id cannot be empty if provided",
                ));
            }
            if id.len() > MAX_AGENT_ID_LENGTH {
                return Err(ApiError::validation_error(&format!(
                    "agent_id exceeds maximum length of {} characters",
                    MAX_AGENT_ID_LENGTH
                )));
            }
            // Check for invalid characters (allow alphanumeric, dash, underscore)
            if !id
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            {
                return Err(ApiError::validation_error(
                    "agent_id can only contain alphanumeric characters, dashes, and underscores",
                ));
            }
        }

        // Validate KREDIT values
        if self.kredit_cap < MIN_KREDIT_CAP {
            return Err(ApiError::validation_error(&format!(
                "kredit_cap must be at least {}",
                MIN_KREDIT_CAP
            )));
        }
        if self.kredit_cap > MAX_KREDIT_CAP {
            return Err(ApiError::validation_error(&format!(
                "kredit_cap cannot exceed {}",
                MAX_KREDIT_CAP
            )));
        }
        if self.initial_kredit < 0 {
            return Err(ApiError::validation_error(
                "initial_kredit cannot be negative",
            ));
        }
        if self.initial_kredit as u64 > self.kredit_cap {
            return Err(ApiError::validation_error(
                "initial_kredit cannot exceed kredit_cap",
            ));
        }

        Ok(())
    }
}

/// Response after creating an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CreateAgentResponse {
    /// The created agent
    pub agent: AgentSummary,
    /// Events generated
    pub events_generated: usize,
}

/// Request to update an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct UpdateAgentRequest {
    /// New status (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<AgentStatus>,
    /// New KREDIT balance (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kredit_balance: Option<i64>,
    /// Reason for update
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Summary view of an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentSummary {
    /// Agent ID
    pub agent_id: String,
    /// Sponsor DID
    pub sponsor_did: String,
    /// Agent class
    pub agent_class: AgentClass,
    /// Current status
    pub status: AgentStatus,
    /// KREDIT balance
    pub kredit_balance: i64,
    /// KREDIT cap
    pub kredit_cap: u64,
    /// Trust score (from K-Vector)
    pub trust_score: f32,
    /// Total actions taken
    pub total_actions: usize,
    /// Success rate
    pub success_rate: f64,
    /// Created timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}

impl From<&InstrumentalActor> for AgentSummary {
    fn from(agent: &InstrumentalActor) -> Self {
        let stats = agent.summary_stats();
        Self {
            agent_id: agent.agent_id.as_str().to_string(),
            sponsor_did: agent.sponsor_did.clone(),
            agent_class: agent.agent_class,
            status: agent.status,
            kredit_balance: agent.kredit_balance,
            kredit_cap: agent.kredit_cap,
            trust_score: agent.k_vector.trust_score(),
            total_actions: stats.total_actions,
            success_rate: stats.success_rate,
            created_at: agent.created_at,
            last_activity: agent.last_activity,
        }
    }
}

/// List agents response
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ListAgentsResponse {
    /// List of agents
    pub agents: Vec<AgentSummary>,
    /// Total count
    pub total: usize,
    /// Pagination offset
    pub offset: usize,
    /// Pagination limit
    pub limit: usize,
}

/// K-Vector history response
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorHistoryResponse {
    /// Agent ID
    pub agent_id: String,
    /// History entries (timestamp, snapshot)
    pub history: Vec<KVectorHistoryEntry>,
    /// Total entries
    pub total: usize,
}

/// Single K-Vector history entry
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorHistoryEntry {
    /// Timestamp
    pub timestamp: u64,
    /// K-Vector values
    pub k_vector: KVectorValues,
    /// Trust score at this point
    pub trust_score: f32,
}

/// K-Vector dimension values
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorValues {
    /// Reputation dimension
    pub k_r: f32,
    /// Activity dimension
    pub k_a: f32,
    /// Integrity dimension
    pub k_i: f32,
    /// Performance dimension
    pub k_p: f32,
    /// Materiality dimension
    pub k_m: f32,
    /// Sustainability dimension
    pub k_s: f32,
    /// Historical dimension
    pub k_h: f32,
    /// Topological dimension
    pub k_topo: f32,
}

impl From<&KVectorSnapshot> for KVectorValues {
    fn from(snap: &KVectorSnapshot) -> Self {
        Self {
            k_r: snap.k_r,
            k_a: snap.k_a,
            k_i: snap.k_i,
            k_p: snap.k_p,
            k_m: snap.k_m,
            k_s: snap.k_s,
            k_h: snap.k_h,
            k_topo: snap.k_topo,
        }
    }
}

/// Events response
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct EventsResponse {
    /// Events
    pub events: Vec<EventSummary>,
    /// Last sequence number
    pub last_sequence: u64,
    /// Has more events
    pub has_more: bool,
}

/// Summary of an event
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct EventSummary {
    /// Event ID
    pub event_id: u64,
    /// Sequence number
    pub sequence: u64,
    /// Event type
    pub event_type: String,
    /// Agent ID
    pub agent_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Event details as JSON (skipped in TypeScript - use event_type for typing)
    #[cfg_attr(feature = "ts-export", ts(skip))]
    pub details: serde_json::Value,
}

impl From<&EventLogEntry> for EventSummary {
    fn from(entry: &EventLogEntry) -> Self {
        let (event_type, agent_id, details) = match &entry.event {
            AgentEvent::Created {
                agent_id,
                sponsor_did,
                agent_class,
                ..
            } => (
                "Created".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "sponsor_did": sponsor_did,
                    "agent_class": format!("{:?}", agent_class),
                }),
            ),
            AgentEvent::StatusChanged {
                agent_id,
                old_status,
                new_status,
                reason,
                ..
            } => (
                "StatusChanged".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "old_status": format!("{:?}", old_status),
                    "new_status": format!("{:?}", new_status),
                    "reason": reason,
                }),
            ),
            AgentEvent::KVectorUpdated {
                agent_id, trigger, ..
            } => (
                "KVectorUpdated".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "trigger": trigger,
                }),
            ),
            AgentEvent::ActionRecorded {
                agent_id,
                action_type,
                kredit_consumed,
                outcome,
                ..
            } => (
                "ActionRecorded".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "action_type": action_type,
                    "kredit_consumed": kredit_consumed,
                    "outcome": outcome,
                }),
            ),
            AgentEvent::KreditChanged {
                agent_id,
                old_balance,
                new_balance,
                reason,
                ..
            } => (
                "KreditChanged".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "old_balance": old_balance,
                    "new_balance": new_balance,
                    "reason": reason,
                }),
            ),
            AgentEvent::Quarantined {
                agent_id,
                reason,
                evidence,
                ..
            } => (
                "Quarantined".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "reason": reason,
                    "evidence": evidence,
                }),
            ),
            AgentEvent::Released { agent_id, .. } => (
                "Released".to_string(),
                agent_id.clone(),
                serde_json::json!({}),
            ),
            AgentEvent::OutputVerified {
                agent_id,
                output_id,
                outcome,
                ..
            } => (
                "OutputVerified".to_string(),
                agent_id.clone(),
                serde_json::json!({
                    "output_id": output_id,
                    "outcome": outcome,
                }),
            ),
        };

        Self {
            event_id: entry.event_id,
            sequence: entry.sequence,
            event_type,
            agent_id,
            timestamp: entry.timestamp,
            details,
        }
    }
}

// =============================================================================
// API Service
// =============================================================================

/// Agent API service
pub struct AgentApiService {
    /// Agent repository
    repo: AgentRepository<MemoryStorageBackend>,
    /// Monitoring engine
    monitoring: MonitoringEngine,
}

impl AgentApiService {
    /// Create a new API service
    pub fn new() -> Self {
        Self {
            repo: AgentRepository::new(MemoryStorageBackend::new()),
            monitoring: MonitoringEngine::new(AlertThresholds::default(), 1000),
        }
    }

    /// Create with custom backend
    pub fn with_repository(repo: AgentRepository<MemoryStorageBackend>) -> Self {
        Self {
            repo,
            monitoring: MonitoringEngine::new(AlertThresholds::default(), 1000),
        }
    }

    // -------------------------------------------------------------------------
    // Agent CRUD
    // -------------------------------------------------------------------------

    /// List all agents
    pub fn list_agents(&self, offset: usize, limit: usize) -> ApiResult<ListAgentsResponse> {
        let all_agents = self.repo.list_all().map_err(ApiError::from)?;

        let agents: Vec<AgentSummary> = all_agents
            .iter()
            .skip(offset)
            .take(limit)
            .map(AgentSummary::from)
            .collect();

        Ok(ListAgentsResponse {
            total: all_agents.len(),
            agents,
            offset,
            limit,
        })
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: &str) -> ApiResult<AgentSummary> {
        let agent = self.repo.get_agent(agent_id).map_err(ApiError::from)?;
        Ok(AgentSummary::from(&agent))
    }

    /// Get full agent details
    pub fn get_agent_full(&self, agent_id: &str) -> ApiResult<InstrumentalActor> {
        self.repo.get_agent(agent_id).map_err(ApiError::from)
    }

    /// Create a new agent
    pub fn create_agent(&self, request: CreateAgentRequest) -> ApiResult<CreateAgentResponse> {
        // Validate input
        request.validate()?;

        let agent_id = request
            .agent_id
            .map(AgentId::from_string)
            .unwrap_or_else(AgentId::generate);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let agent = InstrumentalActor {
            agent_id,
            sponsor_did: request.sponsor_did,
            agent_class: request.agent_class,
            kredit_balance: request.initial_kredit,
            kredit_cap: request.kredit_cap,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: now,
            last_activity: now,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        self.repo
            .create_agent(agent.clone())
            .map_err(ApiError::from)?;

        Ok(CreateAgentResponse {
            agent: AgentSummary::from(&agent),
            events_generated: 1, // Created event
        })
    }

    /// Update an agent
    pub fn update_agent(
        &self,
        agent_id: &str,
        request: UpdateAgentRequest,
    ) -> ApiResult<AgentSummary> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let reason = request.reason.as_deref().unwrap_or("API update");

        // Handle status change
        if let Some(new_status) = request.status {
            let agent = self
                .repo
                .change_status(agent_id, new_status, reason, now)
                .map_err(ApiError::from)?;
            return Ok(AgentSummary::from(&agent));
        }

        // Handle KREDIT change
        if let Some(new_balance) = request.kredit_balance {
            self.repo
                .record_kredit_change(agent_id, new_balance, reason, now)
                .map_err(ApiError::from)?;
        }

        // Return updated agent
        let agent = self.repo.get_agent(agent_id).map_err(ApiError::from)?;
        Ok(AgentSummary::from(&agent))
    }

    /// Delete an agent
    pub fn delete_agent(&self, agent_id: &str) -> ApiResult<()> {
        self.repo.delete_agent(agent_id).map_err(ApiError::from)
    }

    // -------------------------------------------------------------------------
    // K-Vector History
    // -------------------------------------------------------------------------

    /// Get K-Vector history for an agent
    pub fn get_kvector_history(
        &self,
        agent_id: &str,
        limit: Option<usize>,
    ) -> ApiResult<KVectorHistoryResponse> {
        let history = self
            .repo
            .get_kvector_history(agent_id, limit)
            .map_err(ApiError::from)?;

        let entries: Vec<KVectorHistoryEntry> = history
            .iter()
            .map(|(timestamp, snapshot)| KVectorHistoryEntry {
                timestamp: *timestamp,
                k_vector: KVectorValues::from(snapshot),
                trust_score: snapshot.trust_score,
            })
            .collect();

        Ok(KVectorHistoryResponse {
            agent_id: agent_id.to_string(),
            total: entries.len(),
            history: entries,
        })
    }

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    /// Get events for an agent
    pub fn get_agent_events(
        &self,
        agent_id: &str,
        since: Option<u64>,
        limit: Option<usize>,
    ) -> ApiResult<EventsResponse> {
        let events = self
            .repo
            .get_agent_events(agent_id, since, limit)
            .map_err(ApiError::from)?;

        let last_sequence = events.last().map(|e| e.sequence).unwrap_or(0);
        let has_more = limit.map(|l| events.len() >= l).unwrap_or(false);

        Ok(EventsResponse {
            events: events.iter().map(EventSummary::from).collect(),
            last_sequence,
            has_more,
        })
    }

    /// Get all events since sequence
    pub fn get_events_since(
        &self,
        sequence: u64,
        limit: Option<usize>,
    ) -> ApiResult<EventsResponse> {
        let events = self
            .repo
            .get_events_since(sequence, limit)
            .map_err(ApiError::from)?;

        let last_sequence = events.last().map(|e| e.sequence).unwrap_or(sequence);
        let has_more = limit.map(|l| events.len() >= l).unwrap_or(false);

        Ok(EventsResponse {
            events: events.iter().map(EventSummary::from).collect(),
            last_sequence,
            has_more,
        })
    }

    // -------------------------------------------------------------------------
    // Monitoring
    // -------------------------------------------------------------------------

    /// Get dashboard summary
    pub fn get_dashboard(&self) -> ApiResult<DashboardSummary> {
        Ok(self.monitoring.dashboard_summary())
    }

    /// Get active alerts
    pub fn get_alerts(&self) -> ApiResult<Vec<AgentAlert>> {
        Ok(self
            .monitoring
            .get_active_alerts()
            .into_iter()
            .cloned()
            .collect())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> ApiResult<AgentStatistics> {
        AgentStatistics::compute(&self.repo).map_err(ApiError::from)
    }

    // -------------------------------------------------------------------------
    // Bulk Operations
    // -------------------------------------------------------------------------

    /// Find agents by sponsor
    pub fn find_by_sponsor(&self, sponsor_did: &str) -> ApiResult<Vec<AgentSummary>> {
        let agents = self
            .repo
            .find_by_sponsor(sponsor_did)
            .map_err(ApiError::from)?;
        Ok(agents.iter().map(AgentSummary::from).collect())
    }

    /// Find agents by status
    pub fn find_by_status(&self, status: AgentStatus) -> ApiResult<Vec<AgentSummary>> {
        let agents = self.repo.find_by_status(status).map_err(ApiError::from)?;
        Ok(agents.iter().map(AgentSummary::from).collect())
    }

    // -------------------------------------------------------------------------
    // Escalation Management (GIS Integration)
    // -------------------------------------------------------------------------

    /// Get pending escalations for an agent
    pub fn get_pending_escalations(&self, agent_id: &str) -> ApiResult<Vec<EscalationSummary>> {
        let agent = self.repo.get_agent(agent_id).map_err(ApiError::from)?;

        Ok(agent
            .pending_escalations
            .iter()
            .map(|e| EscalationSummary {
                agent_id: e.agent_id.clone(),
                blocked_action: e.blocked_action.clone(),
                uncertainty_total: e.uncertainty.total(),
                uncertainty_max: e.uncertainty.max_dimension(),
                guidance: format!("{:?}", e.guidance),
                recommendations: e.recommendations.clone(),
                timestamp: e.timestamp,
            })
            .collect())
    }

    /// Resolve an escalation (sponsor decision)
    pub fn resolve_escalation(
        &self,
        agent_id: &str,
        blocked_action: &str,
        approved: bool,
        sponsor_did: &str,
    ) -> ApiResult<EscalationResolutionResponse> {
        let mut agent = self.repo.get_agent(agent_id).map_err(ApiError::from)?;

        // Verify sponsor authorization
        if agent.sponsor_did != sponsor_did {
            return Err(ApiError::forbidden(
                "Only the agent's sponsor can resolve escalations",
            ));
        }

        // Find and process the escalation
        let resolutions = vec![(blocked_action.to_string(), approved)];
        let processed = super::lifecycle::process_resolved_escalations(&mut agent, &resolutions);

        if processed.is_empty() {
            return Err(ApiError::not_found(&format!(
                "No pending escalation found for action: {}",
                blocked_action
            )));
        }

        // Save updated agent
        self.repo.update_agent(&agent).map_err(ApiError::from)?;

        Ok(EscalationResolutionResponse {
            agent_id: agent_id.to_string(),
            action: blocked_action.to_string(),
            approved,
            calibration_updated: true,
            remaining_escalations: agent.pending_escalations.len(),
        })
    }

    /// Get uncertainty calibration for an agent
    pub fn get_calibration(&self, agent_id: &str) -> ApiResult<CalibrationSummary> {
        let agent = self.repo.get_agent(agent_id).map_err(ApiError::from)?;
        let cal = &agent.uncertainty_calibration;

        Ok(CalibrationSummary {
            agent_id: agent_id.to_string(),
            total_events: cal.total_events,
            calibration_score: cal.calibration_score(),
            appropriate_uncertainty: cal.appropriate_uncertainty,
            appropriate_confidence: cal.appropriate_confidence,
            overconfident: cal.overconfident,
            overcautious: cal.overcautious,
            is_overconfident: cal.is_overconfident(),
            is_overcautious: cal.is_overcautious(),
        })
    }
}

// =============================================================================
// Escalation API Types
// =============================================================================

/// Summary of an escalation request
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct EscalationSummary {
    /// Agent ID
    pub agent_id: String,
    /// Action that was blocked
    pub blocked_action: String,
    /// Total moral uncertainty
    pub uncertainty_total: f32,
    /// Maximum dimension uncertainty
    pub uncertainty_max: f32,
    /// Guidance level
    pub guidance: String,
    /// Recommendations for resolution
    pub recommendations: Vec<String>,
    /// When the escalation was created
    pub timestamp: u64,
}

/// Response after resolving an escalation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct EscalationResolutionResponse {
    /// Agent ID
    pub agent_id: String,
    /// Action that was resolved
    pub action: String,
    /// Whether sponsor approved
    pub approved: bool,
    /// Whether calibration was updated
    pub calibration_updated: bool,
    /// Remaining pending escalations
    pub remaining_escalations: usize,
}

/// Summary of agent's uncertainty calibration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CalibrationSummary {
    /// Agent ID
    pub agent_id: String,
    /// Total calibration events recorded
    pub total_events: u32,
    /// Overall calibration score (0.0-1.0)
    pub calibration_score: f32,
    /// Times agent was appropriately uncertain
    pub appropriate_uncertainty: u32,
    /// Times agent was appropriately confident
    pub appropriate_confidence: u32,
    /// Times agent was overconfident
    pub overconfident: u32,
    /// Times agent was overcautious
    pub overcautious: u32,
    /// Whether agent tends to be overconfident
    pub is_overconfident: bool,
    /// Whether agent tends to be overcautious
    pub is_overcautious: bool,
}

impl Default for AgentApiService {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_agent() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };

        let response = service.create_agent(request).unwrap();
        assert_eq!(response.agent.agent_id, "test-agent");
        assert_eq!(response.agent.sponsor_did, "did:test:sponsor");
    }

    #[test]
    fn test_list_agents() {
        let service = AgentApiService::new();

        // Create some agents
        for i in 0..5 {
            let request = CreateAgentRequest {
                agent_id: Some(format!("agent-{}", i)),
                sponsor_did: "did:test:sponsor".to_string(),
                agent_class: AgentClass::Supervised,
                initial_kredit: 5000,
                kredit_cap: 10000,
            };
            service.create_agent(request).unwrap();
        }

        let response = service.list_agents(0, 10).unwrap();
        assert_eq!(response.total, 5);
        assert_eq!(response.agents.len(), 5);
    }

    #[test]
    fn test_get_agent() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        let agent = service.get_agent("test-agent").unwrap();
        assert_eq!(agent.agent_id, "test-agent");

        // Not found
        let result = service.get_agent("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_update_agent_status() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        let update = UpdateAgentRequest {
            status: Some(AgentStatus::Suspended),
            kredit_balance: None,
            reason: Some("Testing".to_string()),
        };

        let updated = service.update_agent("test-agent", update).unwrap();
        assert_eq!(updated.status, AgentStatus::Suspended);
    }

    #[test]
    fn test_delete_agent() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        service.delete_agent("test-agent").unwrap();

        let result = service.get_agent("test-agent");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_events() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        // Update to generate more events
        let update = UpdateAgentRequest {
            status: Some(AgentStatus::Suspended),
            kredit_balance: None,
            reason: None,
        };
        service.update_agent("test-agent", update).unwrap();

        let events = service.get_agent_events("test-agent", None, None).unwrap();
        assert_eq!(events.events.len(), 2); // Created + StatusChanged
    }

    #[test]
    fn test_get_statistics() {
        let service = AgentApiService::new();

        // Create agents
        for i in 0..3 {
            let request = CreateAgentRequest {
                agent_id: Some(format!("agent-{}", i)),
                sponsor_did: "did:test:sponsor".to_string(),
                agent_class: AgentClass::Supervised,
                initial_kredit: 5000,
                kredit_cap: 10000,
            };
            service.create_agent(request).unwrap();
        }

        let stats = service.get_statistics().unwrap();
        assert_eq!(stats.total_agents, 3);
        assert_eq!(stats.active_agents, 3);
    }

    #[test]
    fn test_find_by_sponsor() {
        let service = AgentApiService::new();

        // Create agents with different sponsors
        for i in 0..3 {
            let request = CreateAgentRequest {
                agent_id: Some(format!("agent-a-{}", i)),
                sponsor_did: "did:test:sponsor-a".to_string(),
                agent_class: AgentClass::Supervised,
                initial_kredit: 5000,
                kredit_cap: 10000,
            };
            service.create_agent(request).unwrap();
        }

        let request = CreateAgentRequest {
            agent_id: Some("agent-b-0".to_string()),
            sponsor_did: "did:test:sponsor-b".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        let agents = service.find_by_sponsor("did:test:sponsor-a").unwrap();
        assert_eq!(agents.len(), 3);
    }

    // =========================================================================
    // Input Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_empty_sponsor_did() {
        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        let result = request.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("empty"));
    }

    #[test]
    fn test_validate_invalid_sponsor_did_format() {
        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "invalid-sponsor-format".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        let result = request.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("DID"));
    }

    #[test]
    fn test_validate_invalid_agent_id_characters() {
        let request = CreateAgentRequest {
            agent_id: Some("test agent with spaces!".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        let result = request.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("alphanumeric"));
    }

    #[test]
    fn test_validate_kredit_cap_too_low() {
        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 50,
            kredit_cap: 50, // Below MIN_KREDIT_CAP
        };
        let result = request.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("at least"));
    }

    #[test]
    fn test_validate_initial_kredit_exceeds_cap() {
        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 20000,
            kredit_cap: 10000,
        };
        let result = request.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("exceed"));
    }

    #[test]
    fn test_validate_valid_request() {
        let request = CreateAgentRequest {
            agent_id: Some("valid-agent-123".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        assert!(request.validate().is_ok());
    }

    // =========================================================================
    // Escalation API Tests
    // =========================================================================

    #[test]
    fn test_get_calibration() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        let calibration = service.get_calibration("test-agent").unwrap();
        assert_eq!(calibration.agent_id, "test-agent");
        assert_eq!(calibration.total_events, 0);
        assert!((calibration.calibration_score - 0.5).abs() < 0.01); // Default neutral
    }

    #[test]
    fn test_get_pending_escalations_empty() {
        let service = AgentApiService::new();

        let request = CreateAgentRequest {
            agent_id: Some("test-agent".to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            initial_kredit: 5000,
            kredit_cap: 10000,
        };
        service.create_agent(request).unwrap();

        let escalations = service.get_pending_escalations("test-agent").unwrap();
        assert!(escalations.is_empty());
    }
}
