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
    /// Optional full `MycelixConfig` override. When `None`, scenarios use
    /// `MycelixConfig::default().with_bridge_binary(self.bridge_binary)`.
    /// When `Some`, the provided config is used verbatim (with
    /// `bridge_binary` still overridden for consistency). Lets live runs
    /// point `app_id`/`role` at actually-installed hApps.
    pub mycelix_config: Option<MycelixConfig>,
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
            mycelix_config: None,
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

    /// Supply a full [`MycelixConfig`] override. The scenario will still
    /// apply `self.bridge_binary` to it, so the binary choice stays
    /// consistent with `with_bridge_binary`.
    pub fn with_mycelix_config(mut self, cfg: MycelixConfig) -> Self {
        self.mycelix_config = Some(cfg);
        self
    }

    /// Resolve the [`MycelixConfig`] this scenario will pass to
    /// [`BevyMycelixPlugin`]. Internal helper — `bridge_binary` wins.
    fn resolved_mycelix_config(&self) -> MycelixConfig {
        self.mycelix_config
            .clone()
            .unwrap_or_default()
            .with_bridge_binary(self.bridge_binary.clone())
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
    /// Tracks whether a direct `GetProposal(id)` lookup has returned
    /// a non-`None` Record for this agent's proposal. Distinct from
    /// `submission_confirmed` (which only means the conductor accepted
    /// the create_proposal zome call and returned a Record).
    retrieval_confirmed: bool,
    /// Have we dispatched the GetProposal query for this agent?
    get_query_sent: bool,
}

/// The scenario state machine. Kept as a Bevy `Resource` so systems can
/// advance it tick-by-tick.
#[derive(Resource)]
struct ProposalVoteState {
    agents: Vec<AgentState>,
    all_proposals_retrieved: bool,
    errors: Vec<String>,
    sent_count: u32,
    recv_count: u32,
    /// Monotonic tick counter. Used as a startup-delay gate instead of
    /// `Time::elapsed_secs` — Time under MinimalPlugins doesn't always
    /// advance reliably in a tight `app.update()` loop with external
    /// wall-time sleeps, and we'd rather count deterministic ticks.
    frame: u32,
}

// ---------------------------------------------------------------------------
// proposal_vote_invariant
// ---------------------------------------------------------------------------

/// Scenario: N agents each submit one proposal, then each proposal is
/// retrieved by ID via `GetProposal`. Asserts every submitted proposal is
/// retrievable on the next tick.
///
/// Uses per-proposal `GetProposal(id)` rather than `GetActiveProposals`
/// because the governance state machine requires Draft → Active transitions
/// (which need author-is-caller authorization) before proposals appear in
/// the active-proposals index. Direct lookup bypasses the state machine
/// and proves the full write/read round-trip end-to-end.
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
            retrieval_confirmed: false,
            get_query_sent: false,
        })
        .collect();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(BevyMycelixPlugin::new(config.resolved_mycelix_config()))
        .insert_resource(ProposalVoteState {
            agents,
            all_proposals_retrieved: false,
            errors: Vec::new(),
            sent_count: 0,
            recv_count: 0,
            frame: 0,
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
        if state.all_proposals_retrieved {
            return ScenarioReport {
                passed: true,
                summary: format!(
                    "{} agents, all proposals submitted + retrievable by id, 0 errors",
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
                summary: format!("{} errors: {}", state.errors.len(), state.errors.join("; ")),
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
        "all_proposals_retrieved never reached",
    )
    .fail()
    .clone()
}

/// Per-tick driver: submit agent proposals, then issue a `GetProposal`
/// per agent once submission is confirmed.
fn proposal_vote_driver(client: Res<MycelixClient>, mut state: ResMut<ProposalVoteState>) {
    // Tick-based startup delay: give the dispatcher task a few frames to
    // spawn the subprocess before we start pushing requests. With the
    // 10 ms/tick wall sleep in the outer loop, 5 ticks ≈ 50 ms which is
    // enough for the mock. Live bridge needs seconds to finish admin
    // connect + auth; those sends buffer cleanly on the flume channel
    // (budget 128), so early writes are safe — the subprocess will read
    // them after its startup completes.
    state.frame = state.frame.saturating_add(1);
    if state.frame < 5 {
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

    // Phase 2: submission confirmed → dispatch a GetProposal(id) per
    // agent. We check each agent independently so a backpressured send
    // for one doesn't block the others.
    let mut sent_this_tick = 0u32;
    for agent in &mut state.agents {
        if !agent.submission_confirmed || agent.get_query_sent {
            continue;
        }
        let req = MycelixRequest::GetProposal {
            requester: Entity::PLACEHOLDER,
            proposal_id: agent.proposal_id.clone(),
        };
        match client.send(req) {
            Ok(()) => {
                agent.get_query_sent = true;
                sent_this_tick += 1;
            }
            Err(err) => {
                tracing::debug!(?err, "get_proposal backpressured, will retry");
                break;
            }
        }
    }
    state.sent_count += sent_this_tick;
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
            MycelixResponse::Proposal {
                proposal_id,
                record,
                ..
            } => {
                // Find the matching agent by proposal_id. If the record
                // came back as Some, retrieval is confirmed.
                if let Some(agent) = state
                    .agents
                    .iter_mut()
                    .find(|a| a.proposal_id.as_str() == proposal_id.as_str())
                {
                    if record.is_some() {
                        agent.retrieval_confirmed = true;
                    } else {
                        state
                            .errors
                            .push(format!("get_proposal returned None for {proposal_id}"));
                    }
                }
                // If every agent's proposal is now retrievable, mark done.
                if state.agents.iter().all(|a| a.retrieval_confirmed) {
                    state.all_proposals_retrieved = true;
                }
            }
            MycelixResponse::Error { reason, .. } => {
                state.errors.push(reason.clone());
            }
            MycelixResponse::ActiveProposals { .. }
            | MycelixResponse::VoteCast { .. }
            | MycelixResponse::TendBalance { .. } => {
                // Not used by this scenario.
            }
        }
    }
}
