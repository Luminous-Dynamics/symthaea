// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measures one-shot authorization issue/consume latency as retained ledger state grows.
//!
//! Run with:
//!   cargo run -p nixward --example one_shot_authorization_occupancy_bench --release
//!
//! This is a separate scaling lane from one_shot_authorization_bench. It keeps
//! issued permits alive, then measures a fixed probe set at each ledger size.
//! It does not measure RSS/allocator bytes, cleanup, concurrent races, or
//! cross-restart durability. Those require separate instrumented tests.

use std::time::{Duration, Instant};

use nixward::action::authorization::{
    ActionScope, AuthorizationAttempt, AuthorizationAuthority, AuthorizationRequest,
    CommandBinding, ExecutionAuthorizationLedger, ExecutionPermit,
};

const PROBE_COUNT: usize = 100;
const POPULATIONS: [usize; 4] = [1, 1_000, 10_000, 100_000];

struct Probe {
    permit: ExecutionPermit,
    attempt: AuthorizationAttempt,
    issue_ns: u64,
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn percentile(sorted_ns: &[u64], percentile: usize) -> u64 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let rank = (sorted_ns.len() * percentile).div_ceil(100).max(1);
    sorted_ns[rank.saturating_sub(1).min(sorted_ns.len() - 1)]
}

fn issue(
    ledger: &mut ExecutionAuthorizationLedger,
    decision_id: String,
) -> ExecutionPermit {
    let request = AuthorizationRequest::new(
        CommandBinding::new("nixos-rebuild", ["switch", "--flake", ".#bench-host"])
            .expect("static benchmark command is valid"),
        "operator:occupancy-bench",
        ActionScope::SystemCritical,
        AuthorizationAuthority::HumanOperator,
        decision_id,
        Duration::from_secs(1_800),
    )
    .expect("static benchmark authorization request is valid");

    ledger.issue(request).expect("benchmark decision ID is unique")
}

fn summarize(label: &str, samples: &[u64]) {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    println!("{label}_p50_ns={}", percentile(&sorted, 50));
    println!("{label}_p95_ns={}", percentile(&sorted, 95));
    println!("{label}_p99_ns={}", percentile(&sorted, 99));
}

fn main() {
    println!("benchmark=one_shot_authorization_occupancy");
    println!("probe_count_per_population={PROBE_COUNT}");
    println!("population_profiles={:?}", POPULATIONS);
    println!("measurement_scope=issue_and_consume_with_retained_ledger_state");
    println!("memory_bytes_measured=false");

    let mut contract_mismatches = 0usize;

    for population in POPULATIONS {
        let mut ledger = ExecutionAuthorizationLedger::new(Duration::from_secs(3_600))
            .expect("benchmark ledger lifetime is valid");
        let _base_permits: Vec<ExecutionPermit> = Vec::with_capacity(population);

        let setup_start = Instant::now();
        for index in 0..population {
            let permit = issue(&mut ledger, format!("occupancy-{population}-base-{index}"));
            _base_permits.push(permit);
        }
        let base_setup_ns = elapsed_ns(setup_start);

        // Issue all probes first. This keeps probe issue timings in the
        // occupancy band [population, population + PROBE_COUNT - 1].
        let mut probes = Vec::with_capacity(PROBE_COUNT);
        for index in 0..PROBE_COUNT {
            let decision_id = format!("occupancy-{population}-probe-{index}");
            let request = AuthorizationRequest::new(
                CommandBinding::new("nixos-rebuild", ["switch", "--flake", ".#bench-host"])
                    .expect("static benchmark command is valid"),
                "operator:occupancy-bench",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
                decision_id,
                Duration::from_secs(1_800),
            )
            .expect("static benchmark authorization request is valid");

            let issue_start = Instant::now();
            let permit = ledger.issue(request).expect("probe decision ID is unique");
            let issue_ns = elapsed_ns(issue_start);
            let attempt = AuthorizationAttempt::new(
                CommandBinding::new("nixos-rebuild", ["switch", "--flake", ".#bench-host"])
                    .expect("static benchmark command is valid"),
                "operator:occupancy-bench",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            )
            .expect("static benchmark attempt is valid");
            probes.push(Probe { permit, attempt, issue_ns });
        }

        let issue_latencies = probes.iter().map(|probe| probe.issue_ns).collect::<Vec<_>>();
        let mut consume_latencies = Vec::with_capacity(PROBE_COUNT);
        let issued_records_before_consumes = population + PROBE_COUNT;

        println!("profile_start");
        println!("base_population={population}");
        println!("probe_count={PROBE_COUNT}");
        println!("base_setup_ns={base_setup_ns}");
        println!("issued_records_before_consumes={issued_records_before_consumes}");

        let mut profile_mismatches = 0usize;
        for (index, probe) in probes.iter().enumerate() {
            let consume_start = Instant::now();
            let outcome = ledger.consume(&probe.permit, &probe.attempt);
            let consume_ns = elapsed_ns(consume_start);
            consume_latencies.push(consume_ns);

            if outcome.is_err() {
                contract_mismatches += 1;
                profile_mismatches += 1;
            }
            println!(
                "sample={},issue_occupancy_before={},consume_issued_record_count={},consumed_count_before={},issue_ns={},consume_ns={},allowed={}",
                index,
                population + index,
                issued_records_before_consumes,
                index,
                probe.issue_ns,
                consume_ns,
                outcome.is_ok()
            );
        }

        summarize("issue", &issue_latencies);
        summarize("consume", &consume_latencies);
        println!("profile_contract_mismatches={profile_mismatches}");
        println!("cumulative_contract_mismatches={contract_mismatches}");
        println!("profile_end");

        // `_base_permits` keeps the originally issued permits alive for the
        // entire profile; the ledger retains its own per-decision state too.
    }

    println!(
        "status={}",
        if contract_mismatches == 0 { "CONTRACT_MATCHED" } else { "CONTRACT_MISMATCH" }
    );
    println!("not_measured=allocator_bytes,rss,cleanup,concurrency,restart_durability,real_side_effects");

    if contract_mismatches != 0 {
        std::process::exit(1);
    }
}
