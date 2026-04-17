// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Request and response types flowing between Bevy systems and the tokio
//! background task.
//!
//! Milestone 1 covers `GetActiveProposals`. Milestone 2 adds proposal
//! submission, voting, and finance queries.

use bevy::prelude::*;

/// A zome call requested by a Bevy system.
///
/// Each variant carries the requesting [`Entity`] so the matching
/// [`MycelixResponse`] can be routed back. Use [`Entity::PLACEHOLDER`] for
/// requests that aren't associated with a specific entity (e.g. plugin startup
/// smoke tests).
#[derive(Debug, Clone)]
pub enum MycelixRequest {
    /// Fetch all currently active governance proposals from the `agora`
    /// coordinator zome.
    GetActiveProposals { requester: Entity },
}

/// A zome call response, delivered as a Bevy [`Message`] (formerly `Event`
/// in pre-0.18 Bevy).
///
/// Every outstanding [`MycelixRequest`] produces exactly one `MycelixResponse`
/// — either a success variant matching the request shape, or
/// [`MycelixResponse::Error`].
#[derive(Debug, Clone, Message)]
pub enum MycelixResponse {
    /// Successful response to [`MycelixRequest::GetActiveProposals`]. Proposals
    /// are returned as raw JSON values; typed wrappers land in Milestone 2.
    ActiveProposals {
        requester: Entity,
        proposals: Vec<serde_json::Value>,
    },
    /// Any error from transport, authentication, or zome execution.
    Error {
        requester: Entity,
        reason: String,
    },
}

impl MycelixResponse {
    /// The entity that originated the request this response answers.
    pub fn requester(&self) -> Entity {
        match self {
            MycelixResponse::ActiveProposals { requester, .. } => *requester,
            MycelixResponse::Error { requester, .. } => *requester,
        }
    }
}
