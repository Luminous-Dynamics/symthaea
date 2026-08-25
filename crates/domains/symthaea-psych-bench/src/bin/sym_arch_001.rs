// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::env;
use std::fs;
use std::path::PathBuf;
use symthaea_psych_bench::benchmarks::architecture::{SymArch001Config, run_sym_arch_001};

fn main() {
    let output = parse_output_path();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("failed to create SYM-ARCH-001 output directory");
    }

    let source_revision = env::var("GITHUB_SHA")
        .ok()
        .or_else(|| env::var("SYMTHAEA_SOURCE_REVISION").ok());
    let report = run_sym_arch_001(SymArch001Config::default(), source_revision);
    let json = serde_json::to_string_pretty(&report).expect("failed to serialize SYM-ARCH-001 report");
    fs::write(&output, &json).expect("failed to write SYM-ARCH-001 report");

    println!("SYM-ARCH-001 verdict: {}", report.decision.verdict);
    println!("evidence: {}", output.display());
    for agent in &report.agents {
        println!(
            "{:<20} retention={:.3} composition={:.3} forgetting={:.3} reversal={:.3} latency={:.1}",
            agent.agent,
            agent.retention.mean,
            agent.compositional.mean,
            agent.forgetting.mean,
            agent.reversal_final.mean,
            agent.reversal_latency.mean,
        );
    }
    println!(
        "candidate deltas: retention={:+.3}, composition={:+.3}, reversal={:+.3}, forgetting={:+.3}",
        report.decision.retention_delta_vs_best_control,
        report.decision.compositional_delta_vs_best_control,
        report.decision.reversal_delta_vs_best_control,
        report.decision.forgetting_delta_vs_best_control,
    );
}

fn parse_output_path() -> PathBuf {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--output" {
            return PathBuf::from(args.next().expect("--output requires a path"));
        }
    }
    PathBuf::from("artifacts/sym-arch-001/report.json")
}
