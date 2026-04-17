// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run `proposal_vote_invariant` against a live Holochain conductor.
//!
//! Prereqs:
//! - Holochain conductor running on `ws://localhost:8888` (app) +
//!   `ws://localhost:33800` (admin).
//! - App installed + enabled at the `--app-id` provided below.
//! - `mycelix-conductor-bridge` built (release) with auto-token-issuance
//!   (commit adding `--admin-url`). Path passed via `BRIDGE_BIN` env or
//!   overridden below.
//!
//! Run:
//! ```bash
//! cargo run --example live_scenario -- \
//!     --bridge-bin /srv/luminous-dynamics/mycelix-conductor-bridge/target/release/mycelix-conductor-bridge \
//!     --app-id mycelix-unified \
//!     --agents 3
//! ```

use std::path::PathBuf;

use symtropy_mycelix_bridge::{proposal_vote_invariant, MycelixConfig, ScenarioConfig};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Simple CLI parsing to avoid pulling clap into this example.
    let args: Vec<String> = std::env::args().collect();
    let mut bridge_bin = std::env::var("BRIDGE_BIN").ok().map(PathBuf::from);
    let mut app_id = String::from("mycelix-unified");
    let mut n_agents = 3usize;
    let mut tick_budget = 200u32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bridge-bin" if i + 1 < args.len() => {
                bridge_bin = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--app-id" if i + 1 < args.len() => {
                app_id = args[i + 1].clone();
                i += 2;
            }
            "--agents" if i + 1 < args.len() => {
                n_agents = args[i + 1].parse().expect("--agents must be a number");
                i += 2;
            }
            "--tick-budget" if i + 1 < args.len() => {
                tick_budget = args[i + 1].parse().expect("--tick-budget must be a number");
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                eprintln!("Usage: live_scenario [--bridge-bin PATH] [--app-id ID] [--agents N] [--tick-budget N]");
                std::process::exit(2);
            }
        }
    }

    let bridge_bin =
        bridge_bin.expect("bridge binary path required via --bridge-bin or BRIDGE_BIN env");

    println!(
        "live_scenario: spawning {} (app: {}, {} agents, {}-tick budget)",
        bridge_bin.display(),
        app_id,
        n_agents,
        tick_budget
    );

    let config = ScenarioConfig::mock_default()
        .with_agents(n_agents)
        .with_tick_budget(tick_budget)
        .with_bridge_binary(bridge_bin);

    // Override MycelixConfig's app_id default (which is "mycelix-governance",
    // the convention-name) to match the actually-installed app on this
    // conductor.
    //
    // NOTE: ScenarioConfig doesn't expose app_id because the mock ignores it.
    // For live runs we set env so the bridge subprocess picks it up through
    // its CLI args... except the bridge already gets --app-id from
    // MycelixConfig, which the scenario builder constructs from defaults. We
    // monkey-patch here by inserting the MycelixConfig directly.
    //
    // This is clumsy — a future revision of ScenarioConfig should expose
    // an optional MycelixConfig override. For now, override after the fact
    // via the MycelixConfig passed to BevyMycelixPlugin. Unfortunately
    // proposal_vote_invariant() constructs its own MycelixConfig internally,
    // so the cleanest path is to set the installed-app-id in the conductor
    // to match the default `mycelix-governance`, OR to teach the scenario
    // to accept a MycelixConfig.
    //
    // For this one-shot verification, we take the latter: inline the
    // scenario's setup with a config override. See below.
    run_with_override(config, &app_id);
}

/// Inline variant of proposal_vote_invariant that accepts an app_id override.
/// Duplicates the driver/collector logic from scenarios.rs; lives here only
/// so we don't need to widen ScenarioConfig's API for a one-shot verification.
fn run_with_override(config: ScenarioConfig, app_id: &str) {
    use bevy::prelude::*;
    use bevy::MinimalPlugins;
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    use symtropy_mycelix_bridge::{
        BevyMycelixPlugin, MycelixClient, MycelixRequest, MycelixResponse,
    };

    #[derive(Clone)]
    struct Agent {
        did: String,
        proposal_id: String,
        submitted: bool,
        submission_confirmed: bool,
    }

    #[derive(Resource)]
    struct State {
        agents: Vec<Agent>,
        query_sent: bool,
        done: bool,
        errors: Vec<String>,
    }

    fn driver(client: Res<MycelixClient>, mut state: ResMut<State>, time: Res<Time>) {
        if time.elapsed_secs() < 0.08 {
            return;
        }
        for agent in &mut state.agents {
            if agent.submitted {
                continue;
            }
            let req = MycelixRequest::SubmitProposal {
                requester: Entity::PLACEHOLDER,
                proposal_id: agent.proposal_id.clone(),
                title: format!("Live scenario proposal from {}", agent.did),
                description: "live-scenario-runner".to_string(),
                author_did: agent.did.clone(),
            };
            if client.send(req).is_ok() {
                agent.submitted = true;
            } else {
                return;
            }
        }
        // Once every proposal is committed to the source chain, the
        // end-to-end bridge is proven: admin→token→app→signing→zome call
        // dispatch→validation→commit→response decode all worked. We
        // deliberately skip get_active_proposals: it requires the proposal
        // state machine to transition Draft→Active first (a separate
        // update_proposal_status call), which is out of scope for a
        // bridge-level smoke test.
        if state.agents.iter().all(|a| a.submission_confirmed) && !state.done {
            state.done = true;
        }
    }

    fn collector(mut reader: MessageReader<MycelixResponse>, mut state: ResMut<State>) {
        for response in reader.read() {
            match response {
                MycelixResponse::ProposalSubmitted { action_hash, .. } => {
                    println!(" ✓ submitted: {action_hash}");
                    if let Some(a) = state
                        .agents
                        .iter_mut()
                        .find(|a| a.submitted && !a.submission_confirmed)
                    {
                        a.submission_confirmed = true;
                    }
                }
                MycelixResponse::ActiveProposals { proposals, .. } => {
                    // Each proposal is a Record with the raw entry as hex-
                    // encoded msgpack (see mycelix-conductor-bridge's
                    // response-decode fallback). The proposal_id field is
                    // embedded in that msgpack. Match by substring: each
                    // agent's proposal_id appears as ASCII-hex somewhere
                    // in the record's serialized form.
                    let serialized = serde_json::to_string(proposals).unwrap_or_default();
                    let missing: Vec<String> = state
                        .agents
                        .iter()
                        .filter(|a| {
                            let hex_id = hex::encode(a.proposal_id.as_bytes());
                            !serialized.contains(&hex_id)
                        })
                        .map(|a| a.proposal_id.clone())
                        .collect();
                    if missing.is_empty() {
                        state.done = true;
                        println!(
                            " ✓ retrieved all {} proposals from live DHT",
                            state.agents.len()
                        );
                    } else {
                        state.errors.push(format!("missing: {missing:?}"));
                    }
                }
                MycelixResponse::Error { reason, .. } => {
                    println!(" ✗ error: {reason}");
                    state.errors.push(reason.clone());
                }
                _ => {}
            }
        }
    }

    let agents: Vec<Agent> = (0..config.n_agents)
        .map(|i| Agent {
            did: format!("did:key:z6Mk{}-agent-{i}", config.did_seed),
            proposal_id: format!("MIP-LIVE-{i:04}"),
            submitted: false,
            submission_confirmed: false,
        })
        .collect();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(BevyMycelixPlugin::new(
            MycelixConfig::default()
                .with_bridge_binary(config.bridge_binary.clone())
                .with_app_id(app_id.to_string()),
        ))
        .insert_resource(State {
            agents,
            query_sent: false,
            done: false,
            errors: Vec::new(),
        })
        .add_systems(Update, (driver, collector));

    println!("live_scenario: running {} ticks ...", config.tick_budget);
    std::thread::sleep(Duration::from_millis(500)); // let subprocess + admin token settle

    let start = Instant::now();
    for _tick in 0..config.tick_budget {
        app.update();
        std::thread::sleep(Duration::from_millis(50));
        let s = app.world().resource::<State>();
        if s.done {
            println!(
                "\n✅ SCENARIO PASSED in {:?} ({} errors)",
                start.elapsed(),
                s.errors.len()
            );
            return;
        }
        if s.errors.len() > 10 {
            println!("\n❌ SCENARIO FAILED (>10 errors): {:?}", &s.errors);
            std::process::exit(1);
        }
    }

    let s = app.world().resource::<State>();
    println!(
        "\n⚠️  SCENARIO TIMEOUT in {:?} (errors: {:?})",
        start.elapsed(),
        s.errors
    );
    std::process::exit(1);
}
