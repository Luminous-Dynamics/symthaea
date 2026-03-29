// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Agent Persistence Layer
//!
//! Database persistence for agents with history tracking and event sourcing.
//!
//! ## Features
//!
//! - **Agent Storage**: Persist agent state with efficient queries
//! - **K-Vector History**: Track trust profile evolution over time
//! - **Event Sourcing**: Full audit trail of agent lifecycle events
//! - **Backend Abstraction**: Support for multiple storage backends

use super::{AgentClass, AgentStatus, InstrumentalActor};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Event Sourcing
// ============================================================================

/// Agent lifecycle events for audit trail
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AgentEvent {
    /// Agent was created
    Created {
        /// Agent identifier.
        agent_id: String,
        /// Sponsor DID.
        sponsor_did: String,
        /// Classification of the agent.
        agent_class: AgentClass,
        /// Event timestamp.
        timestamp: u64,
    },
    /// Agent status changed
    StatusChanged {
        /// Agent identifier.
        agent_id: String,
        /// Previous agent status.
        old_status: AgentStatus,
        /// New agent status.
        new_status: AgentStatus,
        /// Reason for the event.
        reason: String,
        /// Event timestamp.
        timestamp: u64,
    },
    /// K-Vector updated
    KVectorUpdated {
        /// Agent identifier.
        agent_id: String,
        /// Previous K-Vector state.
        old_kvector: KVectorSnapshot,
        /// Updated K-Vector state.
        new_kvector: KVectorSnapshot,
        /// Event that triggered the update.
        trigger: String,
        /// Event timestamp.
        timestamp: u64,
    },
    /// Action recorded
    ActionRecorded {
        /// Agent identifier.
        agent_id: String,
        /// Type of action performed.
        action_type: String,
        /// Amount of KREDIT consumed.
        kredit_consumed: u64,
        /// Outcome of the action.
        outcome: String,
        /// Event timestamp.
        timestamp: u64,
    },
    /// KREDIT balance changed
    KreditChanged {
        /// Agent identifier.
        agent_id: String,
        /// Previous KREDIT balance.
        old_balance: i64,
        /// New KREDIT balance.
        new_balance: i64,
        /// Reason for the event.
        reason: String,
        /// Event timestamp.
        timestamp: u64,
    },
    /// Agent quarantined
    Quarantined {
        /// Agent identifier.
        agent_id: String,
        /// Reason for the event.
        reason: String,
        /// Evidence for quarantine.
        evidence: Vec<String>,
        /// Event timestamp.
        timestamp: u64,
    },
    /// Agent released from quarantine
    Released {
        /// Agent identifier.
        agent_id: String,
        /// Event timestamp.
        timestamp: u64,
    },
    /// Output verified
    OutputVerified {
        /// Agent identifier.
        agent_id: String,
        /// Identifier of the verified output.
        output_id: String,
        /// Outcome of the action.
        outcome: String,
        /// Event timestamp.
        timestamp: u64,
    },
}

/// Snapshot of K-Vector for storage
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorSnapshot {
    /// Reliability dimension.
    pub k_r: f32,
    /// Accuracy dimension.
    pub k_a: f32,
    /// Integrity dimension.
    pub k_i: f32,
    /// Participation dimension.
    pub k_p: f32,
    /// Maintenance dimension.
    pub k_m: f32,
    /// Social dimension.
    pub k_s: f32,
    /// Harmony dimension.
    pub k_h: f32,
    /// Topological dimension.
    pub k_topo: f32,
    /// Composite trust score.
    pub trust_score: f32,
}

impl From<&KVector> for KVectorSnapshot {
    fn from(kv: &KVector) -> Self {
        Self {
            k_r: kv.k_r,
            k_a: kv.k_a,
            k_i: kv.k_i,
            k_p: kv.k_p,
            k_m: kv.k_m,
            k_s: kv.k_s,
            k_h: kv.k_h,
            k_topo: kv.k_topo,
            trust_score: kv.trust_score(),
        }
    }
}

/// Event log entry with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventLogEntry {
    /// Unique event ID
    pub event_id: u64,
    /// Sequence number for ordering
    pub sequence: u64,
    /// The event
    pub event: AgentEvent,
    /// Event timestamp (Unix seconds)
    pub timestamp: u64,
}

// ============================================================================
// Storage Backend Trait
// ============================================================================

/// Error type for persistence operations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum PersistenceError {
    /// Agent not found
    NotFound(String),
    /// Duplicate agent ID
    AlreadyExists(String),
    /// Serialization error
    SerializationError(String),
    /// Storage backend error
    StorageError(String),
    /// Query error
    QueryError(String),
    /// Lock poisoned
    LockPoisoned(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Agent not found: {}", id),
            Self::AlreadyExists(id) => write!(f, "Agent already exists: {}", id),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::StorageError(msg) => write!(f, "Storage error: {}", msg),
            Self::QueryError(msg) => write!(f, "Query error: {}", msg),
            Self::LockPoisoned(msg) => write!(f, "Lock poisoned: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {}

/// Result type for persistence operations
pub type PersistenceResult<T> = Result<T, PersistenceError>;

/// Agent persistence backend trait
pub trait AgentStorageBackend: Send + Sync {
    /// Save an agent
    fn save_agent(&mut self, agent: &InstrumentalActor) -> PersistenceResult<()>;

    /// Load an agent by ID
    fn load_agent(&self, agent_id: &str) -> PersistenceResult<InstrumentalActor>;

    /// Delete an agent
    fn delete_agent(&mut self, agent_id: &str) -> PersistenceResult<()>;

    /// Check if agent exists
    fn agent_exists(&self, agent_id: &str) -> bool;

    /// List all agent IDs
    fn list_agents(&self) -> PersistenceResult<Vec<String>>;

    /// Find agents by sponsor
    fn find_by_sponsor(&self, sponsor_did: &str) -> PersistenceResult<Vec<InstrumentalActor>>;

    /// Find agents by status
    fn find_by_status(&self, status: AgentStatus) -> PersistenceResult<Vec<InstrumentalActor>>;

    /// Save K-Vector history entry
    fn save_kvector_history(
        &mut self,
        agent_id: &str,
        snapshot: KVectorSnapshot,
        timestamp: u64,
    ) -> PersistenceResult<()>;

    /// Get K-Vector history for agent
    fn get_kvector_history(
        &self,
        agent_id: &str,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<(u64, KVectorSnapshot)>>;

    /// Append event to log
    fn append_event(&mut self, event: AgentEvent) -> PersistenceResult<u64>;

    /// Get events for agent
    fn get_agent_events(
        &self,
        agent_id: &str,
        since: Option<u64>,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<EventLogEntry>>;

    /// Get all events since sequence number
    fn get_events_since(
        &self,
        sequence: u64,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<EventLogEntry>>;
}

// ============================================================================
// In-Memory Backend (for testing and development)
// ============================================================================

/// In-memory storage backend
pub struct MemoryStorageBackend {
    agents: HashMap<String, InstrumentalActor>,
    kvector_history: HashMap<String, Vec<(u64, KVectorSnapshot)>>,
    events: Vec<EventLogEntry>,
    next_event_id: u64,
    next_sequence: u64,
}

impl MemoryStorageBackend {
    /// Create new empty backend
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            kvector_history: HashMap::new(),
            events: Vec::new(),
            next_event_id: 1,
            next_sequence: 1,
        }
    }

    /// Get number of stored agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get total events count
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

impl Default for MemoryStorageBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentStorageBackend for MemoryStorageBackend {
    fn save_agent(&mut self, agent: &InstrumentalActor) -> PersistenceResult<()> {
        self.agents
            .insert(agent.agent_id.as_str().to_string(), agent.clone());
        Ok(())
    }

    fn load_agent(&self, agent_id: &str) -> PersistenceResult<InstrumentalActor> {
        self.agents
            .get(agent_id)
            .cloned()
            .ok_or_else(|| PersistenceError::NotFound(agent_id.to_string()))
    }

    fn delete_agent(&mut self, agent_id: &str) -> PersistenceResult<()> {
        self.agents
            .remove(agent_id)
            .map(|_| ())
            .ok_or_else(|| PersistenceError::NotFound(agent_id.to_string()))
    }

    fn agent_exists(&self, agent_id: &str) -> bool {
        self.agents.contains_key(agent_id)
    }

    fn list_agents(&self) -> PersistenceResult<Vec<String>> {
        Ok(self.agents.keys().cloned().collect())
    }

    fn find_by_sponsor(&self, sponsor_did: &str) -> PersistenceResult<Vec<InstrumentalActor>> {
        Ok(self
            .agents
            .values()
            .filter(|a| a.sponsor_did == sponsor_did)
            .cloned()
            .collect())
    }

    fn find_by_status(&self, status: AgentStatus) -> PersistenceResult<Vec<InstrumentalActor>> {
        Ok(self
            .agents
            .values()
            .filter(|a| a.status == status)
            .cloned()
            .collect())
    }

    fn save_kvector_history(
        &mut self,
        agent_id: &str,
        snapshot: KVectorSnapshot,
        timestamp: u64,
    ) -> PersistenceResult<()> {
        self.kvector_history
            .entry(agent_id.to_string())
            .or_default()
            .push((timestamp, snapshot));
        Ok(())
    }

    fn get_kvector_history(
        &self,
        agent_id: &str,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<(u64, KVectorSnapshot)>> {
        let history = self
            .kvector_history
            .get(agent_id)
            .cloned()
            .unwrap_or_default();

        Ok(if let Some(n) = limit {
            history.into_iter().rev().take(n).rev().collect()
        } else {
            history
        })
    }

    fn append_event(&mut self, event: AgentEvent) -> PersistenceResult<u64> {
        let timestamp = match &event {
            AgentEvent::Created { timestamp, .. } => *timestamp,
            AgentEvent::StatusChanged { timestamp, .. } => *timestamp,
            AgentEvent::KVectorUpdated { timestamp, .. } => *timestamp,
            AgentEvent::ActionRecorded { timestamp, .. } => *timestamp,
            AgentEvent::KreditChanged { timestamp, .. } => *timestamp,
            AgentEvent::Quarantined { timestamp, .. } => *timestamp,
            AgentEvent::Released { timestamp, .. } => *timestamp,
            AgentEvent::OutputVerified { timestamp, .. } => *timestamp,
        };

        let entry = EventLogEntry {
            event_id: self.next_event_id,
            sequence: self.next_sequence,
            event,
            timestamp,
        };

        self.next_event_id += 1;
        self.next_sequence += 1;
        self.events.push(entry);

        Ok(self.next_sequence - 1)
    }

    fn get_agent_events(
        &self,
        agent_id: &str,
        since: Option<u64>,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<EventLogEntry>> {
        let filtered: Vec<_> = self
            .events
            .iter()
            .filter(|e| {
                let event_agent_id = match &e.event {
                    AgentEvent::Created { agent_id, .. } => agent_id,
                    AgentEvent::StatusChanged { agent_id, .. } => agent_id,
                    AgentEvent::KVectorUpdated { agent_id, .. } => agent_id,
                    AgentEvent::ActionRecorded { agent_id, .. } => agent_id,
                    AgentEvent::KreditChanged { agent_id, .. } => agent_id,
                    AgentEvent::Quarantined { agent_id, .. } => agent_id,
                    AgentEvent::Released { agent_id, .. } => agent_id,
                    AgentEvent::OutputVerified { agent_id, .. } => agent_id,
                };
                event_agent_id == agent_id && since.is_none_or(|s| e.sequence > s)
            })
            .cloned()
            .collect();

        Ok(if let Some(n) = limit {
            filtered.into_iter().take(n).collect()
        } else {
            filtered
        })
    }

    fn get_events_since(
        &self,
        sequence: u64,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<EventLogEntry>> {
        let filtered: Vec<_> = self
            .events
            .iter()
            .filter(|e| e.sequence > sequence)
            .cloned()
            .collect();

        Ok(if let Some(n) = limit {
            filtered.into_iter().take(n).collect()
        } else {
            filtered
        })
    }
}

// ============================================================================
// Agent Repository (High-level API)
// ============================================================================

/// High-level repository for agent persistence
pub struct AgentRepository<B: AgentStorageBackend> {
    backend: Arc<RwLock<B>>,
}

impl<B: AgentStorageBackend> AgentRepository<B> {
    /// Create a new repository with the given backend
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(RwLock::new(backend)),
        }
    }

    /// Create a new agent and persist it
    pub fn create_agent(&self, agent: InstrumentalActor) -> PersistenceResult<()> {
        let mut backend = self
            .backend
            .write()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;

        if backend.agent_exists(agent.agent_id.as_str()) {
            return Err(PersistenceError::AlreadyExists(
                agent.agent_id.as_str().to_string(),
            ));
        }

        // Record creation event
        let event = AgentEvent::Created {
            agent_id: agent.agent_id.as_str().to_string(),
            sponsor_did: agent.sponsor_did.clone(),
            agent_class: agent.agent_class,
            timestamp: agent.created_at,
        };
        backend.append_event(event)?;

        // Save initial K-Vector snapshot
        backend.save_kvector_history(
            agent.agent_id.as_str(),
            KVectorSnapshot::from(&agent.k_vector),
            agent.created_at,
        )?;

        backend.save_agent(&agent)
    }

    /// Get an agent by ID
    pub fn get_agent(&self, agent_id: &str) -> PersistenceResult<InstrumentalActor> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.load_agent(agent_id)
    }

    /// Update an existing agent
    pub fn update_agent(&self, agent: &InstrumentalActor) -> PersistenceResult<()> {
        let mut backend = self
            .backend
            .write()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;

        if !backend.agent_exists(agent.agent_id.as_str()) {
            return Err(PersistenceError::NotFound(
                agent.agent_id.as_str().to_string(),
            ));
        }

        backend.save_agent(agent)
    }

    /// Update agent with K-Vector change tracking
    pub fn update_agent_with_kvector_history(
        &self,
        agent: &InstrumentalActor,
        old_kvector: &KVector,
        trigger: &str,
        timestamp: u64,
    ) -> PersistenceResult<()> {
        let mut backend = self
            .backend
            .write()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;

        // Record K-Vector update event
        let event = AgentEvent::KVectorUpdated {
            agent_id: agent.agent_id.as_str().to_string(),
            old_kvector: KVectorSnapshot::from(old_kvector),
            new_kvector: KVectorSnapshot::from(&agent.k_vector),
            trigger: trigger.to_string(),
            timestamp,
        };
        backend.append_event(event)?;

        // Save K-Vector snapshot
        backend.save_kvector_history(
            agent.agent_id.as_str(),
            KVectorSnapshot::from(&agent.k_vector),
            timestamp,
        )?;

        backend.save_agent(agent)
    }

    /// Change agent status with audit trail
    pub fn change_status(
        &self,
        agent_id: &str,
        new_status: AgentStatus,
        reason: &str,
        timestamp: u64,
    ) -> PersistenceResult<InstrumentalActor> {
        let mut backend = self
            .backend
            .write()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;

        let mut agent = backend.load_agent(agent_id)?;
        let old_status = agent.status;
        agent.status = new_status;

        // Record status change event
        let event = AgentEvent::StatusChanged {
            agent_id: agent_id.to_string(),
            old_status,
            new_status,
            reason: reason.to_string(),
            timestamp,
        };
        backend.append_event(event)?;

        backend.save_agent(&agent)?;
        Ok(agent)
    }

    /// Record KREDIT change with audit trail
    pub fn record_kredit_change(
        &self,
        agent_id: &str,
        new_balance: i64,
        reason: &str,
        timestamp: u64,
    ) -> PersistenceResult<()> {
        let mut backend = self
            .backend
            .write()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;

        let mut agent = backend.load_agent(agent_id)?;
        let old_balance = agent.kredit_balance;
        agent.kredit_balance = new_balance;

        let event = AgentEvent::KreditChanged {
            agent_id: agent_id.to_string(),
            old_balance,
            new_balance,
            reason: reason.to_string(),
            timestamp,
        };
        backend.append_event(event)?;

        backend.save_agent(&agent)
    }

    /// Delete an agent
    pub fn delete_agent(&self, agent_id: &str) -> PersistenceResult<()> {
        let mut backend = self
            .backend
            .write()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.delete_agent(agent_id)
    }

    /// List all agents
    pub fn list_all(&self) -> PersistenceResult<Vec<InstrumentalActor>> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        let ids = backend.list_agents()?;
        ids.iter().map(|id| backend.load_agent(id)).collect()
    }

    /// Find agents by sponsor
    pub fn find_by_sponsor(&self, sponsor_did: &str) -> PersistenceResult<Vec<InstrumentalActor>> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.find_by_sponsor(sponsor_did)
    }

    /// Find agents by status
    pub fn find_by_status(&self, status: AgentStatus) -> PersistenceResult<Vec<InstrumentalActor>> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.find_by_status(status)
    }

    /// Get K-Vector history for an agent
    pub fn get_kvector_history(
        &self,
        agent_id: &str,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<(u64, KVectorSnapshot)>> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.get_kvector_history(agent_id, limit)
    }

    /// Get events for an agent
    pub fn get_agent_events(
        &self,
        agent_id: &str,
        since: Option<u64>,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<EventLogEntry>> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.get_agent_events(agent_id, since, limit)
    }

    /// Subscribe to events since sequence (for event streaming)
    pub fn get_events_since(
        &self,
        sequence: u64,
        limit: Option<usize>,
    ) -> PersistenceResult<Vec<EventLogEntry>> {
        let backend = self
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;
        backend.get_events_since(sequence, limit)
    }
}

impl<B: AgentStorageBackend> Clone for AgentRepository<B> {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
        }
    }
}

// ============================================================================
// Query Builders
// ============================================================================

/// Builder for agent queries
pub struct AgentQueryBuilder {
    sponsor_filter: Option<String>,
    status_filter: Option<AgentStatus>,
    trust_min: Option<f32>,
    trust_max: Option<f32>,
    class_filter: Option<AgentClass>,
    limit: Option<usize>,
    offset: usize,
}

impl AgentQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            sponsor_filter: None,
            status_filter: None,
            trust_min: None,
            trust_max: None,
            class_filter: None,
            limit: None,
            offset: 0,
        }
    }

    /// Filter by sponsor DID
    pub fn sponsor(mut self, sponsor_did: &str) -> Self {
        self.sponsor_filter = Some(sponsor_did.to_string());
        self
    }

    /// Filter by status
    pub fn status(mut self, status: AgentStatus) -> Self {
        self.status_filter = Some(status);
        self
    }

    /// Filter by minimum trust score
    pub fn trust_min(mut self, min: f32) -> Self {
        self.trust_min = Some(min);
        self
    }

    /// Filter by maximum trust score
    pub fn trust_max(mut self, max: f32) -> Self {
        self.trust_max = Some(max);
        self
    }

    /// Filter by agent class
    pub fn class(mut self, class: AgentClass) -> Self {
        self.class_filter = Some(class);
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Skip first n results
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    /// Execute the query
    pub fn execute<B: AgentStorageBackend>(
        &self,
        repo: &AgentRepository<B>,
    ) -> PersistenceResult<Vec<InstrumentalActor>> {
        let all_agents = if let Some(ref sponsor) = self.sponsor_filter {
            repo.find_by_sponsor(sponsor)?
        } else if let Some(status) = self.status_filter {
            repo.find_by_status(status)?
        } else {
            repo.list_all()?
        };

        let filtered: Vec<_> = all_agents
            .into_iter()
            .filter(|a| {
                // Apply additional filters
                if let Some(status) = self.status_filter {
                    if a.status != status {
                        return false;
                    }
                }
                if let Some(min) = self.trust_min {
                    if a.k_vector.trust_score() < min {
                        return false;
                    }
                }
                if let Some(max) = self.trust_max {
                    if a.k_vector.trust_score() > max {
                        return false;
                    }
                }
                if let Some(class) = self.class_filter {
                    if a.agent_class != class {
                        return false;
                    }
                }
                true
            })
            .skip(self.offset)
            .take(self.limit.unwrap_or(usize::MAX))
            .collect();

        Ok(filtered)
    }
}

impl Default for AgentQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Statistics and Analytics
// ============================================================================

/// Aggregate statistics about stored agents
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentStatistics {
    /// Total number of agents
    pub total_agents: usize,
    /// Agents by status
    pub by_status: HashMap<String, usize>,
    /// Agents by class
    pub by_class: HashMap<String, usize>,
    /// Average trust score
    pub avg_trust_score: f64,
    /// Median trust score
    pub median_trust_score: f64,
    /// Trust score distribution (deciles)
    pub trust_distribution: [usize; 10],
    /// Total events recorded
    pub total_events: usize,
    /// Active agents (status = Active)
    pub active_agents: usize,
}

impl AgentStatistics {
    /// Compute statistics from repository
    pub fn compute<B: AgentStorageBackend>(repo: &AgentRepository<B>) -> PersistenceResult<Self> {
        let agents = repo.list_all()?;
        let backend = repo
            .backend
            .read()
            .map_err(|e| PersistenceError::LockPoisoned(e.to_string()))?;

        let mut by_status: HashMap<String, usize> = HashMap::new();
        let mut by_class: HashMap<String, usize> = HashMap::new();
        let mut trust_scores: Vec<f64> = Vec::new();
        let mut trust_distribution = [0usize; 10];
        let mut active_agents = 0;

        for agent in &agents {
            *by_status.entry(format!("{:?}", agent.status)).or_insert(0) += 1;
            *by_class
                .entry(format!("{:?}", agent.agent_class))
                .or_insert(0) += 1;

            let trust = agent.k_vector.trust_score() as f64;
            trust_scores.push(trust);

            // Assign to decile (0-9)
            let decile = ((trust * 10.0).floor() as usize).min(9);
            trust_distribution[decile] += 1;

            if agent.status == AgentStatus::Active {
                active_agents += 1;
            }
        }

        let avg_trust_score = if trust_scores.is_empty() {
            0.0
        } else {
            trust_scores.iter().sum::<f64>() / trust_scores.len() as f64
        };

        trust_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_trust_score = if trust_scores.is_empty() {
            0.0
        } else {
            trust_scores[trust_scores.len() / 2]
        };

        let total_events = backend.get_events_since(0, None)?.len();

        Ok(Self {
            total_agents: agents.len(),
            by_status,
            by_class,
            avg_trust_score,
            median_trust_score,
            trust_distribution,
            total_events,
            active_agents,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{AgentConstraints, AgentId, EpistemicStats};

    fn create_test_agent(id: &str, sponsor: &str) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: sponsor.to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 1000,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_memory_backend_basic_operations() {
        let mut backend = MemoryStorageBackend::new();

        let agent = create_test_agent("agent-1", "sponsor-1");
        backend.save_agent(&agent).unwrap();

        assert!(backend.agent_exists("agent-1"));
        assert!(!backend.agent_exists("agent-2"));

        let loaded = backend.load_agent("agent-1").unwrap();
        assert_eq!(loaded.agent_id.as_str(), "agent-1");
        assert_eq!(loaded.sponsor_did, "sponsor-1");

        backend.delete_agent("agent-1").unwrap();
        assert!(!backend.agent_exists("agent-1"));
    }

    #[test]
    fn test_repository_create_and_get() {
        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        let agent = create_test_agent("agent-1", "sponsor-1");
        repo.create_agent(agent).unwrap();

        let loaded = repo.get_agent("agent-1").unwrap();
        assert_eq!(loaded.agent_id.as_str(), "agent-1");

        // Duplicate should fail
        let agent2 = create_test_agent("agent-1", "sponsor-2");
        let result = repo.create_agent(agent2);
        assert!(matches!(result, Err(PersistenceError::AlreadyExists(_))));
    }

    #[test]
    fn test_event_sourcing() {
        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        let agent = create_test_agent("agent-1", "sponsor-1");
        repo.create_agent(agent).unwrap();

        // Change status
        repo.change_status("agent-1", AgentStatus::Suspended, "test", 2000)
            .unwrap();

        // Check events
        let events = repo.get_agent_events("agent-1", None, None).unwrap();
        assert_eq!(events.len(), 2); // Created + StatusChanged

        match &events[0].event {
            AgentEvent::Created { agent_id, .. } => assert_eq!(agent_id, "agent-1"),
            _ => panic!("Expected Created event"),
        }

        match &events[1].event {
            AgentEvent::StatusChanged {
                old_status,
                new_status,
                ..
            } => {
                assert_eq!(*old_status, AgentStatus::Active);
                assert_eq!(*new_status, AgentStatus::Suspended);
            }
            _ => panic!("Expected StatusChanged event"),
        }
    }

    #[test]
    fn test_kvector_history() {
        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        let mut agent = create_test_agent("agent-1", "sponsor-1");
        repo.create_agent(agent.clone()).unwrap();

        // Update K-Vector
        let old_kv = agent.k_vector.clone();
        agent.k_vector = KVector::new(0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.5, 0.3, 0.7, 0.65);

        repo.update_agent_with_kvector_history(&agent, &old_kv, "test_update", 2000)
            .unwrap();

        // Check history
        let history = repo.get_kvector_history("agent-1", None).unwrap();
        assert_eq!(history.len(), 2); // Initial + update
        assert!(history[1].1.k_r > history[0].1.k_r);
    }

    #[test]
    fn test_query_builder() {
        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        // Create agents
        let mut agent1 = create_test_agent("agent-1", "sponsor-1");
        agent1.k_vector = KVector::new(0.8, 0.7, 0.9, 0.8, 0.3, 0.4, 0.6, 0.3, 0.8, 0.75);
        repo.create_agent(agent1).unwrap();

        let mut agent2 = create_test_agent("agent-2", "sponsor-1");
        agent2.k_vector = KVector::new(0.3, 0.2, 0.4, 0.3, 0.1, 0.1, 0.2, 0.1, 0.3, 0.25);
        repo.create_agent(agent2).unwrap();

        let agent3 = create_test_agent("agent-3", "sponsor-2");
        repo.create_agent(agent3).unwrap();

        // Query by sponsor
        let results = AgentQueryBuilder::new()
            .sponsor("sponsor-1")
            .execute(&repo)
            .unwrap();
        assert_eq!(results.len(), 2);

        // Query by trust minimum (agent-1 has high trust, agent-2 has low trust)
        let results = AgentQueryBuilder::new()
            .trust_min(0.6)
            .execute(&repo)
            .unwrap();
        assert_eq!(results.len(), 1); // Only agent-1 has trust >= 0.6

        // Query with limit
        let results = AgentQueryBuilder::new().limit(2).execute(&repo).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_statistics() {
        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        // Create agents
        for i in 0..5 {
            let agent = create_test_agent(&format!("agent-{}", i), "sponsor-1");
            repo.create_agent(agent).unwrap();
        }

        let stats = AgentStatistics::compute(&repo).unwrap();
        assert_eq!(stats.total_agents, 5);
        assert_eq!(stats.active_agents, 5);
        assert_eq!(stats.total_events, 5); // 5 Created events
    }

    #[test]
    fn test_poisoned_lock_returns_error() {
        // Test that poisoned locks in AgentRepository return PersistenceError::LockPoisoned
        use std::panic;
        use std::sync::{Arc, RwLock};

        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        // Create an agent first
        let agent = create_test_agent("agent-1", "sponsor-1");
        repo.create_agent(agent.clone()).unwrap();

        // Get reference to the internal backend lock
        let backend_ref = Arc::clone(&repo.backend);

        // Poison the lock by panicking while holding a write guard
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = backend_ref.write().unwrap();
            panic!("intentional panic to poison lock");
        }));
        assert!(result.is_err(), "Panic should have occurred");

        // Verify the lock is poisoned
        assert!(backend_ref.read().is_err(), "Lock should be poisoned");
        assert!(backend_ref.write().is_err(), "Lock should be poisoned");

        // Now test that repository operations return LockPoisoned error:

        // create_agent (requires write)
        let new_agent = create_test_agent("agent-2", "sponsor-1");
        let result = repo.create_agent(new_agent);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "create_agent should return LockPoisoned error, got: {:?}",
            result
        );

        // get_agent (requires read)
        let result = repo.get_agent("agent-1");
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "get_agent should return LockPoisoned error, got: {:?}",
            result
        );

        // update_agent (requires write)
        let result = repo.update_agent(&agent);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "update_agent should return LockPoisoned error, got: {:?}",
            result
        );

        // update_agent_with_kvector_history (requires write)
        let old_kv = agent.k_vector.clone();
        let result = repo.update_agent_with_kvector_history(&agent, &old_kv, "test", 1000);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "update_agent_with_kvector_history should return LockPoisoned error, got: {:?}",
            result
        );

        // change_status (requires write)
        let result = repo.change_status("agent-1", AgentStatus::Suspended, "test", 2000);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "change_status should return LockPoisoned error, got: {:?}",
            result
        );

        // record_kredit_change (requires write)
        let result = repo.record_kredit_change("agent-1", 1000, "test", 2000);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "record_kredit_change should return LockPoisoned error, got: {:?}",
            result
        );

        // delete_agent (requires write)
        let result = repo.delete_agent("agent-1");
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "delete_agent should return LockPoisoned error, got: {:?}",
            result
        );

        // list_all (requires read)
        let result = repo.list_all();
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "list_all should return LockPoisoned error, got: {:?}",
            result
        );

        // find_by_sponsor (requires read)
        let result = repo.find_by_sponsor("sponsor-1");
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "find_by_sponsor should return LockPoisoned error, got: {:?}",
            result
        );

        // find_by_status (requires read)
        let result = repo.find_by_status(AgentStatus::Active);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "find_by_status should return LockPoisoned error, got: {:?}",
            result
        );

        // get_kvector_history (requires read)
        let result = repo.get_kvector_history("agent-1", None);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "get_kvector_history should return LockPoisoned error, got: {:?}",
            result
        );

        // get_agent_events (requires read)
        let result = repo.get_agent_events("agent-1", None, None);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "get_agent_events should return LockPoisoned error, got: {:?}",
            result
        );

        // get_events_since (requires read)
        let result = repo.get_events_since(0, None);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "get_events_since should return LockPoisoned error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_statistics_with_poisoned_lock() {
        // Test that AgentStatistics::compute handles poisoned locks
        use std::panic;

        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        // Create some agents first
        for i in 0..3 {
            let agent = create_test_agent(&format!("agent-{}", i), "sponsor-1");
            repo.create_agent(agent).unwrap();
        }

        // Poison the backend lock
        let backend_ref = Arc::clone(&repo.backend);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = backend_ref.write().unwrap();
            panic!("poison lock");
        }));

        // AgentStatistics::compute should fail gracefully
        let result = AgentStatistics::compute(&repo);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "AgentStatistics::compute should return LockPoisoned error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_query_builder_with_poisoned_lock() {
        // Test that AgentQueryBuilder handles poisoned locks
        use std::panic;

        let backend = MemoryStorageBackend::new();
        let repo = AgentRepository::new(backend);

        // Create some agents
        let agent = create_test_agent("agent-1", "sponsor-1");
        repo.create_agent(agent).unwrap();

        // Poison the lock
        let backend_ref = Arc::clone(&repo.backend);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = backend_ref.write().unwrap();
            panic!("poison lock");
        }));

        // Query execution should fail gracefully
        let result = AgentQueryBuilder::new().sponsor("sponsor-1").execute(&repo);
        assert!(
            matches!(result, Err(PersistenceError::LockPoisoned(_))),
            "Query builder should return LockPoisoned error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_persistence_error_display() {
        // Test Display impl for all PersistenceError variants

        let err = PersistenceError::NotFound("agent-123".to_string());
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("agent-123"));

        let err = PersistenceError::AlreadyExists("agent-456".to_string());
        assert!(err.to_string().contains("already exists"));
        assert!(err.to_string().contains("agent-456"));

        let err = PersistenceError::SerializationError("json parse failed".to_string());
        assert!(err.to_string().contains("Serialization error"));
        assert!(err.to_string().contains("json parse failed"));

        let err = PersistenceError::StorageError("disk full".to_string());
        assert!(err.to_string().contains("Storage error"));
        assert!(err.to_string().contains("disk full"));

        let err = PersistenceError::QueryError("invalid filter".to_string());
        assert!(err.to_string().contains("Query error"));
        assert!(err.to_string().contains("invalid filter"));

        let err = PersistenceError::LockPoisoned("RwLock poisoned by panic".to_string());
        assert!(err.to_string().contains("Lock poisoned"));
        assert!(err.to_string().contains("RwLock poisoned by panic"));
    }
}
