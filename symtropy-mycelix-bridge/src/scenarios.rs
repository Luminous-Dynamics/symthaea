// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario harness — run declarative multi-agent scenarios against the
//! Mycelix bridge.
//!
//! The harness is transport-agnostic: it builds a Bevy `App` with
//! [`BevyMycelixPlugin`] and any bridge binary the caller points at. For
//! offline testing, point at the `mycelix-mock-conductor` binary shipped by
//! this crate (build with `--features mock-conductor`). For live testing,
//! point at `mycelix-conductor-bridge` against a running conductor.
//!
//! # Scenarios
//!
//! Each scenario is a function of shape:
//! ```ignore
//! fn run_scenario(config: ScenarioConfig) -> ScenarioReport
//! ```
//!
//! Currently shipped:
//! - [`proposal_vote_invariant`] — N agents each submit one proposal, all
//!   proposals are retrievable in the next `GetActiveProposals` response.
//!   Matches the "5-entity gate" from the M2 plan.
//!
//! Planned (same-pattern extensions, tracked in `plans/mycelix-bridge-plan.md`):
//! - `orange_tier_byzantine` — Orange-tier quorum cannot pass a constitutional
//!   proposal (requires consciousness-tier zome integration).
//! - `tend_velocity_under_demurrage` — TEND balance decays under inactivity.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::MinimalPlugins;

use crate::config::MycelixConfig;
use crate::events::{MycelixRequest, MycelixResponse};
use crate::plugin::BevyMycelixPlugin;
use crate::resource::MycelixClient;

/// Inputs for a scenario run.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Number of agents to spawn.
    pub n_agents: usize,
    /// Maximum number of Bevy ticks before the scenario is declared a
    /// timeout. Every scenario should converge in well under this budget.
    pub tick_budget: u32,
    /// Path to the bridge binary — `mycelix-mock-conductor` for offline
    /// scenarios, `mycelix-conductor-bridge` for live.
    pub bridge_binary: PathBuf,
    /// Seed for per-agent DIDs. Scenarios derive DIDs as
    /// `did:key:z6Mk{did_seed}-agent-{i}` to keep them deterministic and
    /// easy to grep in logs.
    pub did_seed: String,
}

impl ScenarioConfig {
    /// Standard config for mock runs: 5 agents, 60-tick budget, local mock.
    ///
    /// The 60-tick budget is ~1s of simulated game time at 60 Hz — plenty
    /// for mock round-trips (sub-millisecond) and safe for live conductor
    /// latency (~4 ms per read).
    pub fn mock_default() -> Self {
        Self {
            n_agents: 5,
            tick_budget: 60,
            bridge_binary: PathBuf::from("mycelix-mock-conductor"),
            did_seed: "scenario".to_string(),
        }
    }

    /// Override the number of agents.
    pub fn with_agents(mut self, n: usize) -> Self {
        self.n_agents = n;
        self
    }

    /// Override the tick budget.
    pub fn with_tick_budget(mut self, ticks: u32) -> Self {
        self.tick_budget = ticks;
        self
    }

    /// Override the bridge binary path.
    pub fn with_bridge_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.bridge_binary = path.into();
        self
    }
}

/// Outcome of a scenario run.
#[derive(Debug, Clone)]
pub struct ScenarioReport {
    /// Whether every asserted invariant held.
    pub passed: bool,
    /// Free-form description of the outcome (failure reason on `!passed`).
    pub summary: String,
    /// Wall-clock time the scenario took to converge (or time out).
    pub elapsed: Duration,
    /// Number of Bevy ticks the scenario consumed.
    pub ticks: u32,
    /// Number of requests sent.
    pub requests_sent: u32,
    /// Number of responses received.
    pub responses_received: u32,
    /// Number of error responses received.
    pub errors: u32,
}

impl ScenarioReport {
    fn timeout(elapsed: Duration, ticks: u32, reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            summary: format!("timeout after {ticks} ticks: {}", reason.into()),
            elapsed,
            ticks,
            requests_sent: 0,
            responses_received: 0,
            errors: 0,
        }
    }

    fn fail(&self) -> &Self {
        self
    }
}

// ---------------------------------------------------------------------------
// Scenario state (inserted as a Resource, observed by systems)
// ---------------------------------------------------------------------------

/// Per-agent bookkeeping.
#[derive(Clone, Debug)]
struct AgentState {
    did: String,
    proposal_id: String,
    submitted: bool,
    submission_confirmed: bool,
}

/// The scenario state machine. Kept as a Bevy `Resource` so systems can
/// advance it tick-by-tick.
#[derive(Resource)]
struct ProposalVoteState {
    agents: Vec<AgentState>,
    query_sent: bool,
    all_proposals_seen: bool,
    errors: Vec<String>,
    sent_count: u32,
    recv_count: u32,
}

// ---------------------------------------------------------------------------
// proposal_vote_invariant
// ---------------------------------------------------------------------------

/// Scenario: N agents each submit one proposal. After all submissions
/// confirm, issue a `GetActiveProposals` and assert every proposal appears
/// in the result.
///
/// This is the M2-gate 5-entity test and the smoke test for the harness.
pub fn proposal_vote_invariant(config: ScenarioConfig) -> ScenarioReport {
    let start = Instant::now();

    let agents: Vec<AgentState> = (0..config.n_agents)
        .map(|i| AgentState {
            did: format!("did:key:z6Mk{}-agent-{i}", config.did_seed),
            proposal_id: format!("MIP-{}-{i:04}", config.did_seed),
            submitted: false,
            submission_confirmed: false,
        })
        .collect();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(BevyMycelixPlugin::new(
            MycelixConfig::default()
                .with_bridge_binary(config.bridge_binary.clone()),
        ))
        .insert_resource(ProposalVoteState {
            agents,
            query_sent: false,
            all_proposals_seen: false,
            errors: Vec::new(),
            sent_count: 0,
            recv_count: 0,
        })
        .add_systems(Update, (proposal_vote_driver, proposal_vote_collector));

    // Give the subprocess-spawn task a moment to be scheduled before the
    // first tick. Without this, the dispatcher may not have launched the
    // child process by the time our driver tries to push requests.
    std::thread::sleep(Duration::from_millis(50));

    let mut ticks = 0u32;
    while ticks < config.tick_budget {
        app.update();
        // Subprocess IPC happens on a separate thread in real wall time;
        // without a small sleep, 60 ticks complete in microseconds and the
        // mock has no chance to round-trip. 10 ms per tick ≈ 100 Hz, still
        // well inside a reasonable scenario budget.
        std::thread::sleep(Duration::from_millis(10));
        ticks += 1;
        let state = app.world().resource::<ProposalVoteState>();
        if state.all_proposals_seen {
            return ScenarioReport {
                passed: true,
                summary: format!(
                    "{} agents, all proposals retrievable, 0 errors",
                    state.agents.len()
                ),
                elapsed: start.elapsed(),
                ticks,
                requests_sent: state.sent_count,
                responses_received: state.recv_count,
                errors: state.errors.len() as u32,
            };
        }
        if !state.errors.is_empty() {
            return ScenarioReport {
                passed: false,
                summary: format!(
                    "{} errors: {}",
                    state.errors.len(),
                    state.errors.join("; ")
                ),
                elapsed: start.elapsed(),
                ticks,
                requests_sent: state.sent_count,
                responses_received: state.recv_count,
                errors: state.errors.len() as u32,
            };
        }
    }

    ScenarioReport::timeout(
        start.elapsed(),
        ticks,
        "all_proposals_seen never reached",
    )
    .fail()
    .clone()
}

/// Per-tick driver: submit agent proposals, then issue the query once all
/// submissions are confirmed.
fn proposal_vote_driver(
    client: Res<MycelixClient>,
    mut state: ResMut<ProposalVoteState>,
    time: Res<Time>,
) {
    // Wait a beat for the subprocess to come up — 5 ticks ≈ 83 ms at 60 Hz
    // and is more than enough for the mock to print "ready".
    if time.elapsed_secs() < 0.08 {
        return;
    }

    // Phase 1: submit each agent's proposal. Count sends separately to
    // avoid holding a mutable borrow on `state.agents` and `state.sent_count`
    // at the same time.
    let mut sent_this_tick = 0u32;
    for agent in &mut state.agents {
        if agent.submitted {
            continue;
        }
        let request = MycelixRequest::SubmitProposal {
            requester: Entity::PLACEHOLDER,
            proposal_id: agent.proposal_id.clone(),
            title: format!("Proposal from {}", agent.did),
            description: "scenario_harness".to_string(),
            author_did: agent.did.clone(),
        };
        match client.send(request) {
            Ok(()) => {
                agent.submitted = true;
                sent_this_tick += 1;
            }
            Err(err) => {
                // Channel full — retry next tick.
                tracing::debug!(?err, "submit backpressured, will retry");
                state.sent_count += sent_this_tick;
                return;
            }
        }
    }
    state.sent_count += sent_this_tick;

    // Phase 2: all submitted, all confirmed — issue the query once.
    let all_confirmed = state.agents.iter().all(|a| a.submission_confirmed);
    if all_confirmed && !state.query_sent {
        match client.send(MycelixRequest::GetActiveProposals {
            requester: Entity::PLACEHOLDER,
        }) {
            Ok(()) => {
                state.query_sent = true;
                state.sent_count += 1;
            }
            Err(err) => {
                tracing::debug!(?err, "query backpressured, will retry");
            }
        }
    }
}

/// Per-tick collector: match responses to bookkeeping, fire the invariant
/// check when the query response arrives.
fn proposal_vote_collector(
    mut reader: MessageReader<MycelixResponse>,
    mut state: ResMut<ProposalVoteState>,
) {
    for response in reader.read() {
        state.recv_count += 1;
        match response {
            MycelixResponse::ProposalSubmitted { .. } => {
                // Order isn't guaranteed; mark the first not-yet-confirmed.
                if let Some(agent) = state
                    .agents
                    .iter_mut()
                    .find(|a| a.submitted && !a.submission_confirmed)
                {
                    agent.submission_confirmed = true;
                }
            }
            MycelixResponse::ActiveProposals { proposals, .. } => {
                if !state.query_sent {
                    // Stray — not ours.
                    continue;
                }
                let expected: std::collections::HashSet<&str> = state
                    .agents
                    .iter()
                    .map(|a| a.proposal_id.as_str())
                    .collect();
                let got: std::collections::HashSet<String> = proposals
                    .iter()
                    .filter_map(|p| p.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect();
                let missing: Vec<String> = expected
                    .iter()
                    .filter(|id| !got.contains(**id))
                    .map(|s| s.to_string())
                    .collect();
                if missing.is_empty() {
                    state.all_proposals_seen = true;
                } else {
                    state.errors.push(format!(
                        "missing proposals: {}",
                        missing.join(", ")
                    ));
                }
            }
            MycelixResponse::Error { reason, .. } => {
                state.errors.push(reason.clone());
            }
            MycelixResponse::VoteCast { .. } | MycelixResponse::TendBalance { .. } => {
                // Not used by this scenario.
            }
        }
    }
}
