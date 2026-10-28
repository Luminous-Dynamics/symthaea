// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic microbenchmark for the process-local one-shot authorization ledger.
//!
//! Run with:
//!   cargo run -p nixward --example one_shot_authorization_bench --release
//!
//! This measures ledger-consumption latency only. It does not launch real
//! processes, measure policy-decision latency, or prove all Nixward paths are
//! gated. The result is a reproducible baseline for this primitive, not an
//! agent-level safety score.

use std::time::{Duration, Instant};

use nixward::action::authorization::{
    ActionScope, AuthorizationAttempt, AuthorizationAuthority, AuthorizationRequest,
    CommandBinding, ExecutionAuthorizationLedger,
};

const WARMUP_CASES: usize = 1_000;
const CASES: usize = 10_000;

#[derive(Clone, Copy)]
enum Scenario {
    Valid,
    CommandMismatch,
    PrincipalMismatch,
    ScopeMismatch,
    AuthorityMismatch,
    ArgumentBoundaryMismatch,
    WrongLedger,
    Revoked,
    Replay,
}

impl Scenario {
    const ALL: [Self; 9] = [
        Self::Valid,
        Self::CommandMismatch,
        Self::PrincipalMismatch,
        Self::ScopeMismatch,
        Self::AuthorityMismatch,
        Self::ArgumentBoundaryMismatch,
        Self::WrongLedger,
        Self::Revoked,
        Self::Replay,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::CommandMismatch => "command_mismatch",
            Self::PrincipalMismatch => "principal_mismatch",
            Self::ScopeMismatch => "scope_mismatch",
            Self::AuthorityMismatch => "authority_mismatch",
            Self::ArgumentBoundaryMismatch => "argument_boundary_mismatch",
            Self::WrongLedger => "wrong_ledger",
            Self::Revoked => "revoked",
            Self::Replay => "replay",
        }
    }

    fn expected_allow(self) -> bool {
        matches!(self, Self::Valid)
    }
}

fn binding(flake: &str) -> CommandBinding {
    CommandBinding::new("nixos-rebuild", ["switch", "--flake", flake])
        .expect("static benchmark command is a valid binding")
}

fn percentile(sorted_ns: &[u64], percentile: usize) -> u64 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let rank = (sorted_ns.len() * percentile).div_ceil(100).max(1);
    sorted_ns[rank.saturating_sub(1).min(sorted_ns.len() - 1)]
}

fn main() {
    let mut latencies_ns = Vec::with_capacity(CASES);
    let mut scenario_latencies_ns: [Vec<u64>; 9] =
        std::array::from_fn(|_| Vec::with_capacity(CASES / 9 + 1));
    let mut scenario_counts = [0usize; 9];
    let mut false_permits = 0usize;
    let mut false_blocks = 0usize;
    let mut contract_mismatches = 0usize;
    let mut allowed = 0usize;
    let mut denied = 0usize;

    // Warm up before recording latency. Each sample uses a fresh small ledger
    // so measured cost is not confounded by collection growth across 10,000
    // historical decisions. Large-ledger scaling belongs in a separate lane.
    let harness_start = Instant::now();
    for index in 0..(WARMUP_CASES + CASES) {
        let measured = index >= WARMUP_CASES;
        let sample_index = index.saturating_sub(WARMUP_CASES);
        let scenario_index = if measured {
            sample_index % Scenario::ALL.len()
        } else {
            index % Scenario::ALL.len()
        };
        let scenario = Scenario::ALL[scenario_index];
        if measured {
            scenario_counts[scenario_index] += 1;
        }

        let mut ledger = ExecutionAuthorizationLedger::new(Duration::from_secs(3_600))
            .expect("benchmark ledger lifetime is valid");
        // A foreign ledger cannot consume or burn a permit owned by this ledger.
        let mut wrong_ledger = if matches!(scenario, Scenario::WrongLedger) {
            Some(
                ExecutionAuthorizationLedger::new(Duration::from_secs(3_600))
                    .expect("benchmark foreign ledger lifetime is valid"),
            )
        } else {
            None
        };

        let decision_id = format!("microbench-decision-{index:05}");
        let request_binding = match scenario {
            Scenario::ArgumentBoundaryMismatch => CommandBinding::new(
                "nixos-rebuild",
                ["switch", "--flake", ".#bench-host", "--option", "a b"],
            )
            .expect("benchmark argument-boundary binding is valid"),
            _ => binding(".#bench-host"),
        };
        let request = AuthorizationRequest::new(
            request_binding,
            "operator:alice",
            ActionScope::SystemCritical,
            AuthorizationAuthority::HumanOperator,
            decision_id.clone(),
            Duration::from_secs(1_800),
        )
        .expect("benchmark authorization request is valid");
        let permit = ledger
            .issue(request)
            .expect("every benchmark decision ID is unique");

        if matches!(scenario, Scenario::Revoked) {
            assert!(ledger.revoke_decision(&decision_id));
        }

        if matches!(scenario, Scenario::Replay) {
            let first_attempt = AuthorizationAttempt::new(
                binding(".#bench-host"),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            )
            .expect("positive-control attempt is valid");
            assert!(ledger.consume(&permit, &first_attempt).is_ok());
        }

        let attempt = match scenario {
            Scenario::Valid | Scenario::Revoked | Scenario::Replay => AuthorizationAttempt::new(
                binding(".#bench-host"),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            ),
            Scenario::CommandMismatch => AuthorizationAttempt::new(
                binding(".#substituted-host"),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            ),
            Scenario::PrincipalMismatch => AuthorizationAttempt::new(
                binding(".#bench-host"),
                "operator:bob",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            ),
            Scenario::ScopeMismatch => AuthorizationAttempt::new(
                binding(".#bench-host"),
                "operator:alice",
                ActionScope::UserModify,
                AuthorizationAuthority::HumanOperator,
            ),
            Scenario::AuthorityMismatch => AuthorizationAttempt::new(
                binding(".#bench-host"),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::WatchdogOperator,
            ),
            Scenario::ArgumentBoundaryMismatch => AuthorizationAttempt::new(
                CommandBinding::new(
                    "nixos-rebuild",
                    ["switch", "--flake", ".#bench-host", "--option", "a", "b"],
                )
                .expect("split argv binding is valid"),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            ),
            Scenario::WrongLedger => AuthorizationAttempt::new(
                binding(".#bench-host"),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
            ),
        }
        .expect("well-formed benchmark attempt");

        // Measure only the ledger consume operation. Issuance and scenario
        // setup are outside this timer, so this is not end-to-end latency.
        let start = Instant::now();
        let outcome = if matches!(scenario, Scenario::WrongLedger) {
            wrong_ledger
                .as_mut()
                .expect("foreign ledger is constructed for this scenario")
                .consume(&permit, &attempt)
        } else {
            ledger.consume(&permit, &attempt)
        };
        let elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        if measured {
            latencies_ns.push(elapsed_ns);
            scenario_latencies_ns[scenario_index].push(elapsed_ns);
        }

        let actual_allow = outcome.is_ok();
        if measured {
            if actual_allow {
                allowed += 1;
            } else {
                denied += 1;
            }
            if actual_allow && !scenario.expected_allow() {
                false_permits += 1;
            }
            if !actual_allow && scenario.expected_allow() {
                false_blocks += 1;
            }
        }

        // Keep scenario labels and policy expectations separate from the
        // implementation's returned error message.
        if actual_allow != scenario.expected_allow() {
            contract_mismatches += 1;
            eprintln!(
                "CONTRACT MISMATCH: scenario={} index={} measured={} allowed={}",
                scenario.name(),
                index,
                measured,
                actual_allow
            );
        }
    }
    let harness_elapsed_ns = harness_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;

    latencies_ns.sort_unstable();
    for samples in &mut scenario_latencies_ns {
        samples.sort_unstable();
    }

    println!("benchmark=one_shot_authorization_ledger");
    println!("warmup_cases={WARMUP_CASES}");
    println!("cases={CASES}");
    println!("measured_ledger_state=one_fresh_permit_per_sample");
    println!("harness_elapsed_ns={harness_elapsed_ns}");
    println!("harness_cases_per_sec={:.2}",
        (WARMUP_CASES + CASES) as f64 * 1_000_000_000.0 / harness_elapsed_ns.max(1) as f64
    );
    println!("allowed_outcomes={allowed}");
    println!("denied_outcomes={denied}");
    println!("false_permits={false_permits}");
    println!("false_blocks={false_blocks}");
    println!("contract_mismatches_including_warmup={contract_mismatches}");
    println!("p50_consume_ns={}", percentile(&latencies_ns, 50));
    println!("p95_consume_ns={}", percentile(&latencies_ns, 95));
    println!("p99_consume_ns={}", percentile(&latencies_ns, 99));
    println!("sum_consume_ns={}", latencies_ns.iter().sum::<u64>());
    println!("scenario_counts:");
    for (index, scenario) in Scenario::ALL.iter().enumerate() {
        let samples = &scenario_latencies_ns[index];
        println!("  {}={}", scenario.name(), scenario_counts[index]);
        println!(
            "  {}_p50_consume_ns={}",
            scenario.name(),
            percentile(samples, 50)
        );
        println!(
            "  {}_p95_consume_ns={}",
            scenario.name(),
            percentile(samples, 95)
        );
        println!(
            "  {}_p99_consume_ns={}",
            scenario.name(),
            percentile(samples, 99)
        );
    }
    println!(
        "status={}",
        if false_permits == 0 && false_blocks == 0 && contract_mismatches == 0 {
            "CONTRACT_MATCHED"
        } else {
            "CONTRACT_MISMATCH"
        }
    );
    println!("measurement_scope=ledger.consume_only");
    println!("not_measured=process_launch,policy_latency,system_side_effects,global_bypass_paths");

    if false_permits != 0 || false_blocks != 0 || contract_mismatches != 0 {
        std::process::exit(1);
    }
}
