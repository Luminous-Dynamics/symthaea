// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sync→async dispatch loop for bridging Symthaea's sync cognitive loop
//! to the async Holochain conductor.
//!
//! Symthaea's `MycelixBridge` sends `GovernanceDispatchCommand` variants
//! through a `std::sync::mpsc::Sender`. This module provides `run_dispatch_loop()`
//! which drains that channel and dispatches real zome calls.

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use crate::conductor::HolochainConductor;
use crate::dispatcher::GovernanceDispatcher;

/// Commands dispatched from Symthaea's sync `MycelixBridge`.
///
/// This enum mirrors `mycelix_bridge::GovernanceDispatchCommand` but is
/// defined here to avoid a circular dependency. The bridge constructs
/// these and sends them via `std::sync::mpsc::Sender`.
#[derive(Debug, Clone)]
pub enum GovernanceDispatchCommand {
    /// Submit a proposal to the governance cluster.
    SubmitProposal {
        description: String,
        proposer_did: String,
        consciousness_phi: f64,
        alignment_score: f64,
    },
    /// Cast a vote on an existing proposal.
    CastVote {
        proposal_id: String,
        voter_did: String,
        approve: bool,
        rationale: String,
    },
    /// Query active proposals (results logged, not returned).
    QueryActiveProposals,
}

/// Run the governance dispatch loop.
///
/// This async function drains commands from the sync channel and dispatches
/// them as real zome calls via the `HolochainConductor`. It runs indefinitely
/// until the sender is dropped (channel closed).
///
/// # Example
///
/// ```ignore
/// let (tx, rx) = std::sync::mpsc::channel();
/// // Give tx to MycelixBridge via set_governance_dispatch_tx()
/// // Spawn this in a tokio task:
/// tokio::spawn(run_dispatch_loop(rx, conductor));
/// ```
pub async fn run_dispatch_loop(
    rx: Receiver<GovernanceDispatchCommand>,
    conductor: Arc<HolochainConductor>,
) {
    let gov = GovernanceDispatcher::new(conductor.clone());

    // Ensure connected before starting
    if let Err(e) = conductor.connect().await {
        tracing::error!(error = %e, "Failed initial conductor connection — dispatch loop will retry on each command");
    }

    loop {
        // Block on the sync receiver — this is fine because we're in a
        // dedicated tokio task (spawned with spawn_blocking or on a
        // dedicated thread).
        let cmd = match rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => {
                tracing::info!("Governance dispatch channel closed — stopping loop");
                break;
            }
        };

        // Ensure connected before each dispatch
        if !conductor.is_connected() {
            if let Err(e) = conductor.connect().await {
                tracing::warn!(error = %e, "Conductor reconnect failed — dropping command");
                continue;
            }
        }

        match cmd {
            GovernanceDispatchCommand::SubmitProposal {
                description,
                proposer_did,
                consciousness_phi,
                alignment_score,
            } => {
                tracing::debug!(
                    phi = consciousness_phi,
                    alignment = alignment_score,
                    "Dispatching proposal to Holochain"
                );
                match gov.submit_proposal(&description, &proposer_did).await {
                    Ok(hash) => tracing::info!(
                        action_hash = %hash,
                        phi = consciousness_phi,
                        "Proposal submitted to Holochain governance"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "Failed to submit proposal to Holochain"
                    ),
                }
            }
            GovernanceDispatchCommand::CastVote {
                proposal_id,
                voter_did,
                approve,
                rationale,
            } => {
                tracing::debug!(
                    proposal = %proposal_id,
                    approve,
                    "Dispatching vote to Holochain"
                );
                match gov.cast_vote(&proposal_id, &voter_did, approve, &rationale).await {
                    Ok(hash) => tracing::info!(
                        action_hash = %hash,
                        proposal = %proposal_id,
                        "Vote dispatched to Holochain governance"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        proposal = %proposal_id,
                        "Failed to dispatch vote to Holochain"
                    ),
                }
            }
            GovernanceDispatchCommand::QueryActiveProposals => {
                match gov.get_active_proposals().await {
                    Ok(proposals) => tracing::info!(
                        count = proposals.len(),
                        "Fetched active proposals from Holochain"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "Failed to fetch active proposals from Holochain"
                    ),
                }
            }
        }
    }
}
