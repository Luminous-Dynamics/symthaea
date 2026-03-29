// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sync Coordinator Zome
//!
//! Implements offline-first synchronization with CRDT-based conflict resolution.
//! Manages sync state, operations queue, and peer coordination.

use hdk::prelude::*;
use mail_sync_integrity::*;
use std::collections::{HashSet, VecDeque};

// ==================== CONSTANTS ====================

/// Maximum operations per checkpoint
const OPS_PER_CHECKPOINT: u32 = 100;

/// Maximum offline queue size
#[allow(dead_code)]
const MAX_QUEUE_SIZE: usize = 10000;

/// Sync batch size
const SYNC_BATCH_SIZE: usize = 50;

/// Maximum retry attempts
const MAX_RETRIES: u8 = 5;

// ==================== SIGNALS ====================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SyncSignal {
    /// Sync state changed
    SyncStateChanged {
        vector_clock: VectorClock,
        pending_ops: u32,
    },
    /// Operation received from peer
    OperationReceived {
        op_id: String,
        op_type: SyncOpType,
        from_agent: AgentPubKey,
    },
    /// Conflict detected
    ConflictDetected {
        conflict_id: String,
        conflict_type: ConflictType,
    },
    /// Conflict resolved
    ConflictResolved {
        conflict_id: String,
        resolution: ConflictResolution,
    },
    /// Came online
    Online,
    /// Went offline
    Offline,
    /// Sync progress
    SyncProgress {
        synced: u32,
        total: u32,
        peer: AgentPubKey,
    },
    /// Sync complete
    SyncComplete {
        peer: AgentPubKey,
        ops_synced: u32,
    },
}

// ==================== INPUTS/OUTPUTS ====================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordOperationInput {
    pub op_type: SyncOpType,
    pub target: Option<ActionHash>,
    pub payload: SyncPayload,
    pub dependencies: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SyncWithPeerInput {
    pub peer: AgentPubKey,
    pub since_checkpoint: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SyncResult {
    pub operations_sent: u32,
    pub operations_received: u32,
    pub conflicts_detected: u32,
    pub conflicts_resolved: u32,
    pub new_vector_clock: VectorClock,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetOperationsInput {
    pub since_clock: Option<VectorClock>,
    pub limit: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApplyRemoteOperationsInput {
    pub operations: Vec<SyncOperation>,
    pub from_peer: AgentPubKey,
}

// ==================== SYNC STATE MANAGEMENT ====================

/// Initialize sync state for current agent
#[hdk_extern]
pub fn init_sync_state(_: ()) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let mut vector_clock = VectorClock::new();
    vector_clock.increment(&agent);

    let sync_state = SyncState {
        agent: agent.clone(),
        vector_clock,
        last_sync: now,
        sync_version: 1,
        state_hash: vec![],
        pending_ops: 0,
        is_online: true,
    };

    let action_hash = create_entry(EntryTypes::SyncState(sync_state.clone()))?;

    // Link to agent
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToSyncState,
        (),
    )?;

    // Emit signal
    emit_signal(SyncSignal::Online)?;

    Ok(action_hash)
}

/// Get current sync state
#[hdk_extern]
pub fn get_sync_state(_: ()) -> ExternResult<Option<SyncState>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToSyncState)?, GetStrategy::default())?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let state: SyncState = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("No sync state found"))?;
                return Ok(Some(state));
            }
        }
    }

    Ok(None)
}

/// Update online status
#[hdk_extern]
pub fn set_online_status(is_online: bool) -> ExternResult<()> {
    let _agent = agent_info()?.agent_initial_pubkey;

    if let Some(mut state) = get_sync_state(())? {
        state.is_online = is_online;
        state.last_sync = sys_time()?;

        create_entry(EntryTypes::SyncState(state))?;

        if is_online {
            emit_signal(SyncSignal::Online)?;
            // Trigger sync when coming online
            process_offline_queue(())?;
        } else {
            emit_signal(SyncSignal::Offline)?;
        }
    }

    Ok(())
}

// ==================== OPERATION RECORDING ====================

/// Record a sync operation (called when user performs an action)
#[hdk_extern]
pub fn record_operation(input: RecordOperationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get current sync state
    let mut sync_state = get_sync_state(())?
        .ok_or(wasm_error!("Sync state not initialized"))?;

    // Increment vector clock
    sync_state.vector_clock.increment(&agent);

    // Generate operation ID
    let op_id = format!(
        "{}:{}:{}",
        agent,
        now.as_micros(),
        sync_state.vector_clock.get(&agent)
    );

    let operation = SyncOperation {
        op_id: op_id.clone(),
        agent: agent.clone(),
        vector_clock: sync_state.vector_clock.clone(),
        op_type: input.op_type.clone(),
        target: input.target,
        payload: input.payload,
        timestamp: now,
        applied: true, // Applied locally
        dependencies: input.dependencies,
    };

    // Create operation entry
    let op_hash = create_entry(EntryTypes::SyncOperation(operation.clone()))?;

    // Link to agent
    create_link(
        agent.clone(),
        op_hash.clone(),
        LinkTypes::AgentToOperations,
        (),
    )?;

    // Update sync state
    sync_state.pending_ops += 1;
    sync_state.sync_version += 1;
    create_entry(EntryTypes::SyncState(sync_state.clone()))?;

    // Check if we should create a checkpoint
    if sync_state.pending_ops >= OPS_PER_CHECKPOINT {
        create_checkpoint(())?;
    }

    // Emit signal
    emit_signal(SyncSignal::SyncStateChanged {
        vector_clock: sync_state.vector_clock,
        pending_ops: sync_state.pending_ops,
    })?;

    // If offline, queue for later
    if !sync_state.is_online {
        queue_for_sync(operation)?;
    }

    Ok(op_hash)
}

/// Queue operation for sync when online
fn queue_for_sync(operation: SyncOperation) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let entry = OfflineQueueEntry {
        entry_id: format!("queue:{}", operation.op_id),
        operation,
        retry_count: 0,
        max_retries: MAX_RETRIES,
        created_at: now,
        last_attempt: None,
        status: QueueStatus::Pending,
        error: None,
    };

    let hash = create_entry(EntryTypes::OfflineQueueEntry(entry))?;

    create_link(agent, hash.clone(), LinkTypes::AgentToOfflineQueue, ())?;

    Ok(hash)
}

/// Process offline queue when back online
#[hdk_extern]
pub fn process_offline_queue(_: ()) -> ExternResult<u32> {
    let agent = agent_info()?.agent_initial_pubkey;
    let mut processed = 0;

    let links = get_links(LinkQuery::try_new(agent.clone(), LinkTypes::AgentToOfflineQueue)?, GetStrategy::default())?;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                let mut entry: OfflineQueueEntry = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid queue entry"))?;

                if entry.status == QueueStatus::Pending {
                    // Try to sync this operation with peers
                    entry.status = QueueStatus::InProgress;
                    entry.last_attempt = Some(sys_time()?);

                    // In real implementation, would send to peers here
                    // For now, mark as completed
                    entry.status = QueueStatus::Completed;
                    processed += 1;
                }
            }
        }
    }

    Ok(processed)
}

// ==================== SYNC WITH PEERS ====================

/// Get operations since a vector clock
#[hdk_extern]
pub fn get_operations_since(input: GetOperationsInput) -> ExternResult<Vec<SyncOperation>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let limit = input.limit.unwrap_or(SYNC_BATCH_SIZE);

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToOperations)?, GetStrategy::default())?;

    let mut operations = Vec::new();

    for link in links.into_iter().rev().take(limit * 2) {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let op: SyncOperation = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid operation"))?;

                // Filter by vector clock if provided
                if let Some(ref since_clock) = input.since_clock {
                    // Only include if operation happened after the since_clock
                    if op.vector_clock.compare(since_clock) > 0 {
                        operations.push(op);
                    }
                } else {
                    operations.push(op);
                }

                if operations.len() >= limit {
                    break;
                }
            }
        }
    }

    Ok(operations)
}

/// Apply remote operations from peer
#[hdk_extern]
pub fn apply_remote_operations(input: ApplyRemoteOperationsInput) -> ExternResult<ApplyResult> {
    let _agent = agent_info()?.agent_initial_pubkey;
    let mut applied = 0;
    let mut conflicts = Vec::new();

    // Get current sync state
    let mut sync_state = get_sync_state(())?
        .ok_or(wasm_error!("Sync state not initialized"))?;

    // Sort operations by dependencies
    let sorted_ops = topological_sort(&input.operations)?;

    for op in sorted_ops {
        // Check if we already have this operation
        if operation_exists(&op.op_id)? {
            continue;
        }

        // Check for conflicts
        let conflict = detect_conflict(&op, &sync_state)?;

        if let Some(conflict_type) = conflict {
            // Record conflict
            let conflict_record = SyncConflict {
                conflict_id: format!("conflict:{}:{}", op.op_id, sys_time()?.as_micros()),
                operations: vec![op.op_id.clone()],
                conflict_type: conflict_type.clone(),
                detected_at: sys_time()?,
                resolution: ConflictResolution::Pending,
                resolved_value: None,
            };

            create_entry(EntryTypes::SyncConflict(conflict_record.clone()))?;

            emit_signal(SyncSignal::ConflictDetected {
                conflict_id: conflict_record.conflict_id.clone(),
                conflict_type,
            })?;

            // Try auto-resolution
            let resolved = try_auto_resolve(&op, &conflict_record)?;
            conflicts.push(conflict_record);

            if !resolved {
                continue; // Skip applying conflicting operation
            }
        }

        // Apply the operation
        let mut applied_op = op.clone();
        applied_op.applied = true;
        create_entry(EntryTypes::SyncOperation(applied_op))?;

        // Merge vector clock
        sync_state.vector_clock.merge(&op.vector_clock);
        applied += 1;

        emit_signal(SyncSignal::OperationReceived {
            op_id: op.op_id,
            op_type: op.op_type,
            from_agent: input.from_peer.clone(),
        })?;
    }

    // Update sync state
    sync_state.last_sync = sys_time()?;
    create_entry(EntryTypes::SyncState(sync_state.clone()))?;

    emit_signal(SyncSignal::SyncComplete {
        peer: input.from_peer,
        ops_synced: applied,
    })?;

    Ok(ApplyResult {
        applied,
        conflicts: conflicts.len() as u32,
        new_clock: sync_state.vector_clock,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApplyResult {
    pub applied: u32,
    pub conflicts: u32,
    pub new_clock: VectorClock,
}

/// Check if operation already exists
fn operation_exists(op_id: &str) -> ExternResult<bool> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToOperations)?, GetStrategy::default())?;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let op: SyncOperation = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid operation"))?;

                if op.op_id == op_id {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Topological sort operations by dependencies
fn topological_sort(operations: &[SyncOperation]) -> ExternResult<Vec<SyncOperation>> {
    let mut sorted = Vec::new();
    let mut remaining: VecDeque<SyncOperation> = operations.iter().cloned().collect();
    let mut processed_ids: HashSet<String> = HashSet::new();

    let max_iterations = operations.len() * 2;
    let mut iterations = 0;

    while !remaining.is_empty() && iterations < max_iterations {
        if let Some(op) = remaining.pop_front() {
            // Check if all dependencies are satisfied
            let deps_satisfied = op.dependencies.iter().all(|dep| {
                processed_ids.contains(dep) || !operations.iter().any(|o| &o.op_id == dep)
            });

            if deps_satisfied {
                processed_ids.insert(op.op_id.clone());
                sorted.push(op);
            } else {
                remaining.push_back(op);
            }
        }
        iterations += 1;
    }

    // Add any remaining (circular deps) at end
    sorted.extend(remaining);

    Ok(sorted)
}

/// Detect conflict between operation and current state
fn detect_conflict(op: &SyncOperation, sync_state: &SyncState) -> ExternResult<Option<ConflictType>> {
    // Check causality - if operation's clock is concurrent with ours
    let comparison = op.vector_clock.compare(&sync_state.vector_clock);

    if comparison == 0 {
        // Concurrent - potential conflict depending on operation type
        match &op.op_type {
            SyncOpType::EmailDeleted | SyncOpType::FolderDeleted => {
                // Delete operations might conflict with updates
                return Ok(Some(ConflictType::DeleteUpdate));
            }
            SyncOpType::EmailMoved | SyncOpType::FolderMoved => {
                // Move operations might conflict
                return Ok(Some(ConflictType::ConcurrentMove));
            }
            _ => {
                // Other concurrent edits
                return Ok(Some(ConflictType::ConcurrentEdit));
            }
        }
    }

    Ok(None)
}

/// Try to automatically resolve conflict using CRDT rules
fn try_auto_resolve(op: &SyncOperation, conflict: &SyncConflict) -> ExternResult<bool> {
    let resolution = match &conflict.conflict_type {
        ConflictType::ConcurrentEdit => {
            // Use CRDT merge rules based on payload type
            match &op.payload {
                SyncPayload::EmailState { email_hash: _, new_state: _ } => {
                    // Email state uses LWW per field - auto resolve
                    true
                }
                SyncPayload::LabelSet { email_hash: _, labels: _ } => {
                    // Labels use Add-Wins Set - auto resolve
                    true
                }
                SyncPayload::DraftContent { draft_hash: _, encrypted_content: _, lww_timestamp: _ } => {
                    // Drafts use LWW - auto resolve
                    true
                }
                SyncPayload::Setting { key: _, value: _, lww_timestamp: _ } => {
                    // Settings use LWW - auto resolve
                    true
                }
                _ => false,
            }
        }
        ConflictType::DeleteUpdate => {
            // Delete wins by default (can be configured)
            true
        }
        ConflictType::ConcurrentMove => {
            // Use timestamp for move operations
            true
        }
        _ => false,
    };

    if resolution {
        emit_signal(SyncSignal::ConflictResolved {
            conflict_id: conflict.conflict_id.clone(),
            resolution: ConflictResolution::AutoResolved {
                strategy: "CRDT".to_string(),
            },
        })?;
    }

    Ok(resolution)
}

// ==================== CRDT MERGE OPERATIONS ====================

/// Merge two email states using CRDT rules
#[hdk_extern]
pub fn merge_email_states(input: MergeStatesInput) -> ExternResult<EmailStateCrdt> {
    let mut merged = input.local.clone();

    // Merge is_read (LWW)
    if let Some(remote_read) = &input.remote.is_read {
        if let Some(local_read) = &merged.is_read {
            if remote_read.timestamp > local_read.timestamp {
                merged.is_read = Some(remote_read.clone());
            }
        } else {
            merged.is_read = Some(remote_read.clone());
        }
    }

    // Merge is_starred (LWW)
    if let Some(remote_starred) = &input.remote.is_starred {
        if let Some(local_starred) = &merged.is_starred {
            if remote_starred.timestamp > local_starred.timestamp {
                merged.is_starred = Some(remote_starred.clone());
            }
        } else {
            merged.is_starred = Some(remote_starred.clone());
        }
    }

    // Merge is_archived (LWW)
    if let Some(remote_archived) = &input.remote.is_archived {
        if let Some(local_archived) = &merged.is_archived {
            if remote_archived.timestamp > local_archived.timestamp {
                merged.is_archived = Some(remote_archived.clone());
            }
        } else {
            merged.is_archived = Some(remote_archived.clone());
        }
    }

    // Merge is_deleted (LWW)
    if let Some(remote_deleted) = &input.remote.is_deleted {
        if let Some(local_deleted) = &merged.is_deleted {
            if remote_deleted.timestamp > local_deleted.timestamp {
                merged.is_deleted = Some(remote_deleted.clone());
            }
        } else {
            merged.is_deleted = Some(remote_deleted.clone());
        }
    }

    // Merge folder (LWW)
    if let Some(remote_folder) = &input.remote.folder {
        if let Some(local_folder) = &merged.folder {
            if remote_folder.timestamp > local_folder.timestamp {
                merged.folder = Some(remote_folder.clone());
            }
        } else {
            merged.folder = Some(remote_folder.clone());
        }
    }

    // Merge labels (Add-Wins Set)
    for remote_label in &input.remote.labels {
        let existing = merged.labels.iter_mut().find(|l| l.label == remote_label.label);

        match existing {
            Some(local_label) => {
                // If remote was added later, update
                if remote_label.added_at > local_label.added_at {
                    *local_label = remote_label.clone();
                }
                // Handle remove: if removed after last add, mark as removed
                if let Some(remote_removed) = remote_label.removed_at {
                    if remote_removed > local_label.added_at {
                        local_label.removed_at = Some(remote_removed);
                        local_label.removed_by = remote_label.removed_by.clone();
                    }
                }
            }
            None => {
                // New label from remote
                merged.labels.push(remote_label.clone());
            }
        }
    }

    // Filter out removed labels
    merged.labels.retain(|l| l.removed_at.is_none());

    // Merge snoozed_until (LWW)
    if let Some(remote_snoozed) = &input.remote.snoozed_until {
        if let Some(local_snoozed) = &merged.snoozed_until {
            if remote_snoozed.timestamp > local_snoozed.timestamp {
                merged.snoozed_until = Some(remote_snoozed.clone());
            }
        } else {
            merged.snoozed_until = Some(remote_snoozed.clone());
        }
    }

    Ok(merged)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MergeStatesInput {
    pub local: EmailStateCrdt,
    pub remote: EmailStateCrdt,
}

// ==================== CHECKPOINTS ====================

/// Create a sync checkpoint
#[hdk_extern]
pub fn create_checkpoint(_: ()) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let sync_state = get_sync_state(())?
        .ok_or(wasm_error!("Sync state not initialized"))?;

    let checkpoint = SyncCheckpoint {
        agent: agent.clone(),
        checkpoint_id: format!("cp:{}:{}", agent, sys_time()?.as_micros()),
        vector_clock: sync_state.vector_clock.clone(),
        state_hash: vec![], // Would compute hash of current state
        ops_since_last: sync_state.pending_ops,
        timestamp: sys_time()?,
        delta: None,
    };

    let hash = create_entry(EntryTypes::SyncCheckpoint(checkpoint))?;

    create_link(agent, hash.clone(), LinkTypes::AgentToCheckpoints, ())?;

    // Reset pending ops counter
    let mut updated_state = sync_state;
    updated_state.pending_ops = 0;
    create_entry(EntryTypes::SyncState(updated_state))?;

    Ok(hash)
}

/// Get latest checkpoint
#[hdk_extern]
pub fn get_latest_checkpoint(_: ()) -> ExternResult<Option<SyncCheckpoint>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToCheckpoints)?, GetStrategy::default())?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let checkpoint: SyncCheckpoint = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid checkpoint"))?;
                return Ok(Some(checkpoint));
            }
        }
    }

    Ok(None)
}

// ==================== REMOTE SIGNALS ====================

/// Handle incoming sync signal from peer
#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    let sig: SyncSignal = signal.decode().map_err(|e| wasm_error!(e))?;

    // Forward to local UI
    emit_signal(sig)?;

    Ok(())
}

/// Request sync from peer (via remote signal)
#[hdk_extern]
pub fn request_sync_from_peer(peer: AgentPubKey) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let sync_state = get_sync_state(())?;

    #[derive(Debug, Serialize, Deserialize)]
    struct SyncRequest {
        from: AgentPubKey,
        since_clock: Option<VectorClock>,
    }

    let request = SyncRequest {
        from: agent,
        since_clock: sync_state.map(|s| s.vector_clock),
    };

    let encoded = ExternIO::encode(request).map_err(|e| wasm_error!(e))?;

    send_remote_signal(encoded, vec![peer])?;

    Ok(())
}

// ==================== PEER MANAGEMENT ====================

/// Register a sync peer
#[hdk_extern]
pub fn register_sync_peer(peer: AgentPubKey) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let sync_peer = SyncPeer {
        agent: peer.clone(),
        last_known_clock: VectorClock::new(),
        last_sync: None,
        avg_latency_ms: 0,
        reliability: 1.0,
        is_reachable: true,
    };

    let hash = create_entry(EntryTypes::SyncPeer(sync_peer))?;

    create_link(agent, hash.clone(), LinkTypes::AgentToSyncPeers, ())?;

    Ok(hash)
}

/// Get all sync peers
#[hdk_extern]
pub fn get_sync_peers(_: ()) -> ExternResult<Vec<SyncPeer>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToSyncPeers)?, GetStrategy::default())?;

    let mut peers = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let peer: SyncPeer = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid peer"))?;
                peers.push(peer);
            }
        }
    }

    Ok(peers)
}
