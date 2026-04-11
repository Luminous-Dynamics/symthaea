// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sync Integrity Zome
//!
//! Offline-first synchronization for Mycelix Mail using CRDTs and vector clocks.
//! Enables seamless offline operation with automatic conflict resolution.

use hdi::prelude::*;

/// Vector clock for causality tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct VectorClock {
    /// Map of agent to logical timestamp
    pub clocks: Vec<(AgentPubKey, u64)>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self { clocks: Vec::new() }
    }

    pub fn increment(&mut self, agent: &AgentPubKey) {
        for (a, t) in &mut self.clocks {
            if a == agent {
                *t += 1;
                return;
            }
        }
        self.clocks.push((agent.clone(), 1));
    }

    pub fn get(&self, agent: &AgentPubKey) -> u64 {
        for (a, t) in &self.clocks {
            if a == agent {
                return *t;
            }
        }
        0
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (agent, other_time) in &other.clocks {
            let mut found = false;
            for (a, t) in &mut self.clocks {
                if a == agent {
                    *t = (*t).max(*other_time);
                    found = true;
                    break;
                }
            }
            if !found {
                self.clocks.push((agent.clone(), *other_time));
            }
        }
    }

    /// Returns ordering: -1 if self < other, 1 if self > other, 0 if concurrent
    pub fn compare(&self, other: &VectorClock) -> i8 {
        let mut self_greater = false;
        let mut other_greater = false;

        // Check all agents in both clocks
        let mut all_agents: Vec<AgentPubKey> = self.clocks.iter().map(|(a, _)| a.clone()).collect();
        for (a, _) in &other.clocks {
            if !all_agents.contains(a) {
                all_agents.push(a.clone());
            }
        }

        for agent in &all_agents {
            let self_time = self.get(agent);
            let other_time = other.get(agent);
            if self_time > other_time {
                self_greater = true;
            }
            if other_time > self_time {
                other_greater = true;
            }
        }

        if self_greater && !other_greater {
            1 // self happened after
        } else if other_greater && !self_greater {
            -1 // other happened after
        } else {
            0 // concurrent (or equal)
        }
    }
}

/// Sync state for an agent
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SyncState {
    /// Agent this sync state belongs to
    pub agent: AgentPubKey,
    /// Current vector clock
    pub vector_clock: VectorClock,
    /// Last sync timestamp
    pub last_sync: Timestamp,
    /// Sync version (increments on each change)
    pub sync_version: u64,
    /// Hash of last known state
    pub state_hash: Vec<u8>,
    /// Pending operations count
    pub pending_ops: u32,
    /// Whether currently online
    pub is_online: bool,
}

/// A sync operation (CRDT operation)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SyncOperation {
    /// Unique operation ID
    pub op_id: String,
    /// Agent who performed the operation
    pub agent: AgentPubKey,
    /// Vector clock at time of operation
    pub vector_clock: VectorClock,
    /// Type of operation
    pub op_type: SyncOpType,
    /// Target entry (if applicable)
    pub target: Option<ActionHash>,
    /// Operation payload
    pub payload: SyncPayload,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Whether operation has been applied locally
    pub applied: bool,
    /// Dependencies (operations that must be applied first)
    pub dependencies: Vec<String>,
}

/// Types of sync operations
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum SyncOpType {
    /// Email operations
    EmailReceived,
    EmailRead,
    EmailDeleted,
    EmailMoved,
    EmailFlagged,
    EmailLabeled,

    /// Folder operations
    FolderCreated,
    FolderRenamed,
    FolderDeleted,
    FolderMoved,

    /// Draft operations
    DraftCreated,
    DraftUpdated,
    DraftDeleted,
    DraftSent,

    /// Settings operations
    SettingChanged,
    FilterCreated,
    FilterUpdated,
    FilterDeleted,

    /// Trust operations
    TrustAttestationCreated,
    TrustAttestationRevoked,

    /// Contact operations
    ContactCreated,
    ContactUpdated,
    ContactDeleted,

    /// Custom operation
    Custom(String),
}

/// Payload for sync operations (CRDT-friendly)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum SyncPayload {
    /// No payload
    None,

    /// Email state change
    EmailState {
        email_hash: ActionHash,
        new_state: EmailStateCrdt,
    },

    /// Folder change
    FolderChange {
        folder_hash: Option<ActionHash>,
        name: Option<String>,
        parent: Option<ActionHash>,
    },

    /// Label set (add-wins set CRDT)
    LabelSet {
        email_hash: ActionHash,
        labels: Vec<LabelOp>,
    },

    /// Draft content (Last-Writer-Wins)
    DraftContent {
        draft_hash: ActionHash,
        encrypted_content: Vec<u8>,
        lww_timestamp: Timestamp,
    },

    /// Setting change (LWW Register)
    Setting {
        key: String,
        value: Vec<u8>,
        lww_timestamp: Timestamp,
    },

    /// Generic JSON payload
    Json(String),
}

/// Email state as CRDT (OR-Set for flags, LWW for state)
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct EmailStateCrdt {
    /// Read state (LWW)
    pub is_read: Option<LwwBool>,
    /// Starred state (LWW)
    pub is_starred: Option<LwwBool>,
    /// Archived state (LWW)
    pub is_archived: Option<LwwBool>,
    /// Deleted state (LWW with tombstone)
    pub is_deleted: Option<LwwBool>,
    /// Current folder (LWW)
    pub folder: Option<LwwFolder>,
    /// Labels (Add-Wins Set)
    pub labels: Vec<AddWinsLabel>,
    /// Snooze until (LWW)
    pub snoozed_until: Option<LwwTimestamp>,
}

/// LWW (Last-Writer-Wins) Boolean
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LwwBool {
    pub value: bool,
    pub timestamp: Timestamp,
    pub agent: AgentPubKey,
}

/// LWW Folder reference
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LwwFolder {
    pub folder_hash: ActionHash,
    pub timestamp: Timestamp,
    pub agent: AgentPubKey,
}

/// LWW Timestamp value
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LwwTimestamp {
    pub value: Option<Timestamp>,
    pub timestamp: Timestamp,
    pub agent: AgentPubKey,
}

/// Add-Wins Set label element
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AddWinsLabel {
    pub label: String,
    pub added_at: Timestamp,
    pub added_by: AgentPubKey,
    /// If removed, when (for tombstone)
    pub removed_at: Option<Timestamp>,
    pub removed_by: Option<AgentPubKey>,
}

/// Label operation for Add-Wins Set
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum LabelOp {
    Add { label: String },
    Remove { label: String },
}

/// Sync conflict record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SyncConflict {
    /// Unique conflict ID
    pub conflict_id: String,
    /// Operations involved in conflict
    pub operations: Vec<String>,
    /// Type of conflict
    pub conflict_type: ConflictType,
    /// Timestamp detected
    pub detected_at: Timestamp,
    /// Resolution status
    pub resolution: ConflictResolution,
    /// Resolved value (if applicable)
    pub resolved_value: Option<Vec<u8>>,
}

/// Types of conflicts
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum ConflictType {
    /// Concurrent edits to same resource
    ConcurrentEdit,
    /// Delete vs update
    DeleteUpdate,
    /// Move to different folders
    ConcurrentMove,
    /// Ordering conflict
    CausalityViolation,
    /// Schema version mismatch
    VersionMismatch,
}

/// Conflict resolution strategy
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum ConflictResolution {
    /// Not yet resolved
    Pending,
    /// Automatically resolved by CRDT rules
    AutoResolved { strategy: String },
    /// Last writer wins
    LastWriterWins { winner: AgentPubKey },
    /// User manually resolved
    ManualResolution { resolver: AgentPubKey },
    /// Merged both changes
    Merged,
    /// Kept both versions
    BothKept,
}

/// Sync checkpoint for efficient incremental sync
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SyncCheckpoint {
    /// Agent this checkpoint is for
    pub agent: AgentPubKey,
    /// Checkpoint ID
    pub checkpoint_id: String,
    /// Vector clock at checkpoint
    pub vector_clock: VectorClock,
    /// State hash at checkpoint
    pub state_hash: Vec<u8>,
    /// Operations since last checkpoint
    pub ops_since_last: u32,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Compressed state delta (for efficient sync)
    pub delta: Option<Vec<u8>>,
}

/// Offline queue entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OfflineQueueEntry {
    /// Queue entry ID
    pub entry_id: String,
    /// Operation to execute when online
    pub operation: SyncOperation,
    /// Retry count
    pub retry_count: u8,
    /// Max retries
    pub max_retries: u8,
    /// Created at
    pub created_at: Timestamp,
    /// Last attempt
    pub last_attempt: Option<Timestamp>,
    /// Status
    pub status: QueueStatus,
    /// Error if failed
    pub error: Option<String>,
}

/// Offline queue status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum QueueStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Sync peer information
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SyncPeer {
    /// Peer agent
    pub agent: AgentPubKey,
    /// Last known vector clock
    pub last_known_clock: VectorClock,
    /// Last successful sync
    pub last_sync: Option<Timestamp>,
    /// Sync latency (ms)
    pub avg_latency_ms: u32,
    /// Reliability score (0-1)
    pub reliability: f32,
    /// Whether currently reachable
    pub is_reachable: bool,
}

/// Link types
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> their sync state
    AgentToSyncState,
    /// Agent -> their operations
    AgentToOperations,
    /// Agent -> their checkpoints
    AgentToCheckpoints,
    /// Agent -> offline queue
    AgentToOfflineQueue,
    /// Agent -> sync peers
    AgentToSyncPeers,
    /// Operation -> conflicts
    OperationToConflicts,
    /// Checkpoint -> operations since
    CheckpointToOperations,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 1)]
    SyncState(SyncState),
    #[entry_type(required_validations = 1)]
    SyncOperation(SyncOperation),
    #[entry_type(required_validations = 1)]
    SyncConflict(SyncConflict),
    #[entry_type(required_validations = 1)]
    SyncCheckpoint(SyncCheckpoint),
    #[entry_type(required_validations = 1)]
    OfflineQueueEntry(OfflineQueueEntry),
    #[entry_type(required_validations = 1)]
    SyncPeer(SyncPeer),
}

// ==================== VALIDATION ====================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::SyncState(state) => validate_sync_state(&state, &action),
        EntryTypes::SyncOperation(op) => validate_sync_operation(&op, &action),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_sync_state(
    state: &SyncState,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Agent must be author
    if state.agent != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Sync state agent must match author".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_sync_operation(
    op: &SyncOperation,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Agent must be author
    if op.agent != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Operation agent must match author".to_string(),
        ));
    }

    // Operation ID must not be empty
    if op.op_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Operation ID cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
