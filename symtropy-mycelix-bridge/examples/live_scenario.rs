// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run `proposal_vote_invariant` against a live Holochain conductor.
//!
//! Prereqs:
//! - Holochain conductor running on `ws://localhost:8888` (app) +
//!   `ws://localhost:33800` (admin).
//! - App installed + enabled at the `--app-id` provided below.
//! - `mycelix-conductor-bridge` built (release) with auto-token-issuance.
//!   Path passed via `BRIDGE_BIN` env or `--bridge-bin`.
//!
//! Run:
//! ```bash
//! cargo run --example live_scenario -- \
//!     --bridge-bin /srv/luminous-dynamics/mycelix-conductor-bridge/target/release/mycelix-conductor-bridge \
//!     --app-id mycelix-unified \
//!     --agents 3
//! ```
//!
//! The scenario does **submit → get_proposal(id) per agent** and declares
//! success when every submitted proposal is retrievable by its ID on the
//! next tick. No inline driver/collector — uses the shared
//! `proposal_vote_invariant` + `ScenarioConfig::with_mycelix_config`.

use std::path::PathBuf;

use symtropy_mycelix_bridge::{proposal_vote_invariant, MycelixConfig, ScenarioConfig};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Tiny hand-rolled CLI parsing to avoid pulling clap into this example.
    let args: Vec<String> = std::env::args().collect();
    let mut bridge_bin = std::env::var("BRIDGE_BIN").ok().map(PathBuf::from);
    let mut app_id = String::from("mycelix-unified");
    let mut n_agents = 3usize;
    let mut tick_budget = 300u32;

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
                eprintln!(
                    "Usage: live_scenario [--bridge-bin PATH] [--app-id ID] \
                     [--agents N] [--tick-budget N]"
                );
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

    let mycelix_cfg = MycelixConfig::default()
        .with_bridge_binary(bridge_bin.clone())
        .with_app_id(app_id);

    let config = ScenarioConfig::mock_default()
        .with_agents(n_agents)
        .with_tick_budget(tick_budget)
        .with_bridge_binary(bridge_bin)
        .with_mycelix_config(mycelix_cfg);

    let report = proposal_vote_invariant(config);

    if report.passed {
        println!(
            "\n✅ SCENARIO PASSED in {:?} ({} ticks, {} req/{} resp, {} errors)\n   {}",
            report.elapsed,
            report.ticks,
            report.requests_sent,
            report.responses_received,
            report.errors,
            report.summary
        );
    } else {
        println!(
            "\n❌ SCENARIO FAILED in {:?} ({} ticks, {} req/{} resp, {} errors)\n   {}",
            report.elapsed,
            report.ticks,
            report.requests_sent,
            report.responses_received,
            report.errors,
            report.summary
        );
        std::process::exit(1);
    }
}
