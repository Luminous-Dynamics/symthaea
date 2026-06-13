// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mock conductor binary.
//!
//! Speaks the same JSON wire protocol as `mycelix-conductor-bridge` but
//! serves every request from an in-memory state machine — no Holochain, no
//! network. Used by the scenario harness so we can exercise the bridge
//! end-to-end without a live conductor.
//!
//! The mock's behavior is deliberately minimal:
//! - `SubmitProposal` stores the proposal, returns a synthetic action hash.
//! - `CastVote` records the vote (by voter_did + proposal_id), echoes
//!   proposal_id back.
//! - `QueryActiveProposals` returns every stored proposal as a JSON array.
//! - `QueryTendBalance` returns a member-specific balance (default 1000.0)
//!   that can be adjusted via `MOCK_TEND_INITIAL_BALANCE` env var.
//!
//! It accepts the same CLI args as the real bridge (`--conductor-url`,
//! `--app-id`, `--role`) and silently ignores them — so the mock is a
//! drop-in replacement for the real binary in `MycelixConfig::bridge_binary`.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Wire types (mirror `mycelix-conductor-bridge`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Request {
    #[serde(default)]
    request_id: Option<u64>,
    #[serde(flatten)]
    command: Command,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Command {
    SubmitProposal(ProposalInput),
    CastVote(VoteInput),
    QueryActiveProposals,
    QueryTendBalance { member_did: String },
    GetProposal { proposal_id: String },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ProposalInput {
    id: String,
    title: String,
    description: String,
    #[serde(default)]
    proposal_type: String,
    author: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    actions: String,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    voting_ends_at: Option<i64>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)] // voter_did / approve / rationale are stored but not yet read by the state machine
struct VoteInput {
    proposal_id: String,
    voter_did: String,
    approve: bool,
    #[serde(default)]
    rationale: String,
}

#[derive(Debug, Serialize)]
struct BridgeResponse {
    request_id: Option<u64>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl BridgeResponse {
    fn ok(request_id: Option<u64>, data: serde_json::Value) -> Self {
        Self {
            request_id,
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    fn err(request_id: Option<u64>, reason: impl Into<String>) -> Self {
        Self {
            request_id,
            ok: false,
            data: None,
            error: Some(reason.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// In-memory state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct State {
    proposals: Vec<ProposalInput>,
    votes: Vec<VoteInput>,
    next_hash: u64,
    tend_balances: HashMap<String, f64>,
    default_tend_balance: f64,
}

impl State {
    fn new() -> Self {
        let default_tend_balance = std::env::var("MOCK_TEND_INITIAL_BALANCE")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1000.0);
        Self {
            default_tend_balance,
            ..Default::default()
        }
    }

    fn handle(&mut self, req: Request) -> BridgeResponse {
        let request_id = req.request_id;
        match req.command {
            Command::SubmitProposal(input) => {
                self.next_hash += 1;
                let action_hash = format!("uhCkk_mock_{:016x}", self.next_hash);
                self.proposals.push(input);
                BridgeResponse::ok(request_id, serde_json::json!(action_hash))
            }
            Command::CastVote(vote) => {
                let proposal_exists = self.proposals.iter().any(|p| p.id == vote.proposal_id);
                if !proposal_exists {
                    return BridgeResponse::err(
                        request_id,
                        format!("no such proposal: {}", vote.proposal_id),
                    );
                }
                let proposal_id = vote.proposal_id.clone();
                self.votes.push(vote);
                BridgeResponse::ok(request_id, serde_json::json!(proposal_id))
            }
            Command::QueryActiveProposals => {
                let data = serde_json::to_value(&self.proposals)
                    .unwrap_or(serde_json::Value::Array(vec![]));
                BridgeResponse::ok(request_id, data)
            }
            Command::GetProposal { proposal_id } => {
                match self.proposals.iter().find(|p| p.id == proposal_id) {
                    Some(p) => BridgeResponse::ok(
                        request_id,
                        serde_json::to_value(p).unwrap_or(serde_json::Value::Null),
                    ),
                    None => BridgeResponse::ok(request_id, serde_json::Value::Null),
                }
            }
            Command::QueryTendBalance { member_did } => {
                let balance = *self
                    .tend_balances
                    .entry(member_did.clone())
                    .or_insert(self.default_tend_balance);
                BridgeResponse::ok(
                    request_id,
                    serde_json::json!({
                        "member_did": member_did,
                        "balance": balance,
                    }),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    // Silently drop CLI args — we're a drop-in replacement for the real
    // bridge binary, which takes --conductor-url / --app-id / --role.
    let _args: Vec<String> = std::env::args().collect();

    eprintln!("mycelix-mock-conductor: ready");

    let mut state = State::new();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(e) => {
                eprintln!("mycelix-mock-conductor: stdin read error: {e}");
                break;
            }
        };

        let resp = match serde_json::from_str::<Request>(&line) {
            Ok(req) => state.handle(req),
            Err(e) => BridgeResponse::err(None, format!("parse error: {e}")),
        };

        let encoded = serde_json::to_string(&resp)
            .unwrap_or_else(|e| format!(r#"{{"ok":false,"error":"encode: {e}"}}"#));
        if writeln!(out, "{encoded}").is_err() {
            break;
        }
        let _ = out.flush();
    }

    eprintln!("mycelix-mock-conductor: stdin closed, exiting");
}

// ---------------------------------------------------------------------------
// Tests (in-process — exercise the state machine directly)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_proposal(id: &str) -> ProposalInput {
        ProposalInput {
            id: id.to_string(),
            title: format!("title for {id}"),
            description: String::new(),
            proposal_type: "Standard".to_string(),
            author: format!("did:key:z6Mk{id}"),
            status: "active".to_string(),
            actions: String::new(),
            created_at: None,
            voting_ends_at: None,
        }
    }

    fn req(command: Command) -> Request {
        Request {
            request_id: Some(1),
            command,
        }
    }

    #[test]
    fn submit_proposal_stores_and_returns_hash() {
        let mut s = State::new();
        let resp = s.handle(req(Command::SubmitProposal(make_proposal("P1"))));
        assert!(resp.ok);
        assert_eq!(s.proposals.len(), 1);
        let hash = resp.data.unwrap();
        assert!(hash.as_str().unwrap().starts_with("uhCkk_mock_"));
    }

    #[test]
    fn query_returns_all_submitted_proposals() {
        let mut s = State::new();
        for i in 0..5 {
            let _ = s.handle(req(Command::SubmitProposal(make_proposal(&format!(
                "P{i}"
            )))));
        }
        let resp = s.handle(req(Command::QueryActiveProposals));
        assert!(resp.ok);
        let arr = resp.data.unwrap();
        assert_eq!(arr.as_array().unwrap().len(), 5);
    }

    #[test]
    fn vote_on_unknown_proposal_errors() {
        let mut s = State::new();
        let resp = s.handle(req(Command::CastVote(VoteInput {
            proposal_id: "ghost".to_string(),
            voter_did: "did:key:z6MkVoter".to_string(),
            approve: true,
            rationale: String::new(),
        })));
        assert!(!resp.ok);
        assert!(resp.error.unwrap().contains("no such proposal"));
    }

    #[test]
    fn vote_on_existing_proposal_succeeds() {
        let mut s = State::new();
        let _ = s.handle(req(Command::SubmitProposal(make_proposal("P1"))));
        let resp = s.handle(req(Command::CastVote(VoteInput {
            proposal_id: "P1".to_string(),
            voter_did: "did:key:z6MkVoter".to_string(),
            approve: true,
            rationale: String::new(),
        })));
        assert!(resp.ok);
        assert_eq!(s.votes.len(), 1);
    }

    #[test]
    fn tend_balance_defaults_to_configured_initial() {
        let mut s = State::new();
        let resp = s.handle(req(Command::QueryTendBalance {
            member_did: "did:key:z6MkAlice".to_string(),
        }));
        assert!(resp.ok);
        let balance = resp.data.unwrap()["balance"].as_f64().unwrap();
        assert_eq!(balance, s.default_tend_balance);
    }

    #[test]
    fn request_id_echoes() {
        let mut s = State::new();
        let resp = s.handle(Request {
            request_id: Some(42),
            command: Command::QueryActiveProposals,
        });
        assert_eq!(resp.request_id, Some(42));
    }
}
