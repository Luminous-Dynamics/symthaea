// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Request and response types flowing between Bevy systems and the tokio
//! background task.
//!
//! Milestone 1 covered `GetActiveProposals`. Milestone 2 adds proposal
//! submission, voting, and TEND balance queries, plus correlation IDs
//! (a `u64` minted per request and echoed in the matching response).

use bevy::prelude::*;

/// A zome call requested by a Bevy system.
///
/// Each variant carries the requesting [`Entity`] so the matching
/// [`MycelixResponse`] can be routed back. Use [`Entity::PLACEHOLDER`] for
/// requests that aren't associated with a specific entity.
#[derive(Debug, Clone)]
pub enum MycelixRequest {
    /// Fetch all currently active governance proposals.
    GetActiveProposals { requester: Entity },
    /// Submit a new proposal to the `proposals` coordinator zome.
    SubmitProposal {
        requester: Entity,
        proposal_id: String,
        title: String,
        description: String,
        author_did: String,
    },
    /// Cast a vote on a proposal.
    CastVote {
        requester: Entity,
        proposal_id: String,
        voter_did: String,
        approve: bool,
        rationale: String,
    },
    /// Query a member's TEND (time-exchange) balance.
    QueryTendBalance {
        requester: Entity,
        member_did: String,
    },
    /// Fetch a single proposal by its ID. Returns `ProposalFound`
    /// (with the full Record as JSON) or `ProposalNotFound` via the
    /// `found` field.
    GetProposal {
        requester: Entity,
        proposal_id: String,
    },
}

impl MycelixRequest {
    /// The entity that originated this request.
    pub fn requester(&self) -> Entity {
        match self {
            MycelixRequest::GetActiveProposals { requester }
            | MycelixRequest::SubmitProposal { requester, .. }
            | MycelixRequest::CastVote { requester, .. }
            | MycelixRequest::QueryTendBalance { requester, .. }
            | MycelixRequest::GetProposal { requester, .. } => *requester,
        }
    }
}

/// A zome call response, delivered as a Bevy [`Message`] (formerly `Event`
/// in pre-0.18 Bevy).
///
/// Every outstanding [`MycelixRequest`] produces exactly one `MycelixResponse`
/// — either a success variant matching the request shape, or
/// [`MycelixResponse::Error`].
#[derive(Debug, Clone, Message)]
pub enum MycelixResponse {
    /// Success response to [`MycelixRequest::GetActiveProposals`]. Proposals
    /// are returned as raw JSON values (typed wrappers land in M3).
    ActiveProposals {
        requester: Entity,
        proposals: Vec<serde_json::Value>,
    },
    /// Success response to [`MycelixRequest::SubmitProposal`]. Carries the
    /// action hash that Holochain produced for the new proposal.
    ProposalSubmitted {
        requester: Entity,
        action_hash: String,
    },
    /// Success response to [`MycelixRequest::CastVote`].
    VoteCast {
        requester: Entity,
        proposal_id: String,
    },
    /// Success response to [`MycelixRequest::QueryTendBalance`]. Raw JSON so
    /// downstream code can decide whether to parse into a typed struct.
    TendBalance {
        requester: Entity,
        balance: serde_json::Value,
    },
    /// Success response to [`MycelixRequest::GetProposal`]. `record` is
    /// `Some(raw Record JSON)` when the proposal exists on chain, `None`
    /// when `get_proposal` returned `None`.
    Proposal {
        requester: Entity,
        proposal_id: String,
        record: Option<serde_json::Value>,
    },
    /// Any error from transport, authentication, or zome execution.
    Error { requester: Entity, reason: String },
}

impl MycelixResponse {
    /// The entity that originated the request this response answers.
    pub fn requester(&self) -> Entity {
        match self {
            MycelixResponse::ActiveProposals { requester, .. }
            | MycelixResponse::ProposalSubmitted { requester, .. }
            | MycelixResponse::VoteCast { requester, .. }
            | MycelixResponse::TendBalance { requester, .. }
            | MycelixResponse::Proposal { requester, .. }
            | MycelixResponse::Error { requester, .. } => *requester,
        }
    }
}
