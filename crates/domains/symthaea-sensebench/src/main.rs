// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run reproducible SenseBench baselines.
//!
//! Examples:
//!   SENSEBENCH_CODE_REVISION=$(git rev-parse HEAD) \
//!     cargo run -p symthaea-sensebench -- --suite anomaly --scenario all --seed 42 --samples 512
//!   SENSEBENCH_CODE_REVISION=$(git rev-parse HEAD) \
//!     cargo run -p symthaea-sensebench -- --suite causal --trials 512 --budget 8 --seed 42
//!   cargo test -p symthaea-sensebench

use serde::Serialize;
use std::env;

use symthaea_sensebench::{
    EvaluationSplit, ScenarioKind, causal_benchmark::run_causal_selection_benchmark, evaluate,
    generate_scenario, z_score_baseline, ScenarioManifest, GENERATOR_VERSION,
    robustness_benchmark::run_model_misspecification_benchmark,
};

#[derive(Serialize)]
struct RunReport {
    report_version: u16,
    suite: &'static str,
    code_revision: Option<String>,
    evaluator: &'static str,
    threshold_sigma: f64,
    baseline_samples: usize,
    manifest: ScenarioManifest,
    metrics: symthaea_sensebench::DetectionMetrics,
}

#[derive(Serialize)]
struct CausalRunReport {
    report_version: u16,
    suite: &'static str,
    generator_version: &'static str,
    code_revision: Option<String>,
    seed: u64,
    trials: usize,
    experiment_budget: usize,
    posterior_threshold: f64,
    summaries: Vec<symthaea_sensebench::causal_benchmark::SelectionPolicySummary>,
}

#[derive(Serialize)]
struct RobustnessRunReport {
    report_version: u16,
    suite: &'static str,
    generator_version: &'static str,
    code_revision: Option<String>,
    benchmark: symthaea_sensebench::robustness_benchmark::ModelMisspecificationBenchmarkReport,
}

#[derive(Serialize)]
struct BlindWorldRunReport {
    report_version: u16,
    suite: &'static str,
    code_revision: Option<String>,
    benchmark: symthaea_sensebench::blind_world_benchmark::BlindWorldBenchmarkReport,
}

#[derive(Serialize)]
struct ScmGraphRunReport {
    report_version: u16,
    suite: &'static str,
    code_revision: Option<String>,
    benchmark: symthaea_sensebench::scm_graph_benchmark::ScmGraphBenchmarkReport,
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn code_revision() -> Option<String> {
    env::var("SENSEBENCH_CODE_REVISION")
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn run() -> Result<(), String> {
    let mut suite = "anomaly".to_owned();
    let mut scenario_arg = "all".to_owned();
    let mut seed = 42_u64;
    let mut samples = 512_usize;
    let mut trials = 512_usize;
    let mut trials_explicit = false;
    let mut budget = 8_usize;
    let mut allow_unversioned = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--suite" => suite = args.next().ok_or("--suite requires a value")?,
            "--allow-unversioned" => allow_unversioned = true,
            "--scenario" => scenario_arg = args.next().ok_or("--scenario requires a value")?,
            "--seed" => seed = args.next().ok_or("--seed requires a value")?
                .parse().map_err(|_| "invalid --seed; expected unsigned integer")?,
            "--samples" => samples = args.next().ok_or("--samples requires a value")?
                .parse().map_err(|_| "invalid --samples; expected positive integer")?,
            "--trials" => {
                trials = args.next().ok_or("--trials requires a value")?
                    .parse().map_err(|_| "invalid --trials; expected positive integer")?;
                trials_explicit = true;
            }
            "--budget" => budget = args.next().ok_or("--budget requires a value")?
                .parse().map_err(|_| "invalid --budget; expected positive integer")?,
            "--help" | "-h" => {
                println!(
                    "sensebench --suite anomaly --scenario all|null|step|spike|drift|dropout \
                     --seed N --samples N"
                );
                println!("sensebench --suite causal --trials N --budget N --seed N");
                println!("sensebench --suite robustness");
                println!("sensebench --suite blind-worlds --trials 512 --budget 8 --seed 42");
                println!("sensebench --suite scm-graph --trials 32 --budget 8 --seed 42");
                println!("set SENSEBENCH_CODE_REVISION to the exact Git commit before running");
                println!("--allow-unversioned is for exploratory runs only");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    // The graph suite has 112 SCM worlds × eleven policies. Default it to a
    // reproducible exploratory run size while preserving other suites' defaults.
    if suite == "scm-graph" && !trials_explicit {
        trials = 32;
    }

    let revision = code_revision();
    if revision.is_none() && !allow_unversioned {
        return Err(
            concat!(
                "SENSEBENCH_CODE_REVISION is required for reproducible reports; ",
                "use --allow-unversioned only for exploratory runs"
            )
            .to_owned(),
        );
    }

    if suite == "causal" {
        let summaries = run_causal_selection_benchmark(trials, budget, seed, 0.95)
            .map_err(|error| error.to_string())?;
        let report = CausalRunReport {
            report_version: 1,
            suite: "causal_selection",
            generator_version: GENERATOR_VERSION,
            code_revision: revision,
            seed,
            trials,
            experiment_budget: budget,
            posterior_threshold: 0.95,
            summaries,
        };
        println!("{}", serde_json::to_string_pretty(&report)
            .map_err(|error| format!("could not encode causal report: {error}"))?);
        return Ok(());
    }
    if suite == "robustness" {
        let benchmark = run_model_misspecification_benchmark()
            .map_err(|error| error.to_string())?;
        let report = RobustnessRunReport {
            report_version: 1,
            suite: "model_misspecification_regret",
            generator_version: GENERATOR_VERSION,
            code_revision: revision,
            benchmark,
        };
        println!("{}", serde_json::to_string_pretty(&report)
            .map_err(|error| format!("could not encode robustness report: {error}"))?);
        return Ok(());
    }
    if suite == "blind-worlds" {
        let benchmark = symthaea_sensebench::blind_world_benchmark::run_blind_world_benchmark(
            trials, budget, seed, 0.95,
        ).map_err(|error| error.to_string())?;
        let report = BlindWorldRunReport {
            report_version: 1,
            suite: "blind_world_active_selection",
            code_revision: revision,
            benchmark,
        };
        println!("{}", serde_json::to_string_pretty(&report)
            .map_err(|error| format!("could not encode blind-world report: {error}"))?);
        return Ok(());
    }
    if suite == "scm-graph" {
        let benchmark = symthaea_sensebench::scm_graph_benchmark::run_scm_graph_benchmark(
            trials, budget, seed, 0.95,
        ).map_err(|error| error.to_string())?;
        let report = ScmGraphRunReport {
            report_version: 6,
            suite: "scm_graph_recovery",
            code_revision: revision,
            benchmark,
        };
        println!("{}", serde_json::to_string_pretty(&report)
            .map_err(|error| format!("could not encode SCM graph report: {error}"))?);
        return Ok(());
    }
    if suite != "anomaly" {
        return Err(format!(
            "unknown suite: {suite}; expected anomaly, causal, robustness, blind-worlds, or scm-graph"
        ));
    }

    let scenarios: Vec<ScenarioKind> = if scenario_arg == "all" {
        vec![
            ScenarioKind::Null,
            ScenarioKind::StepChange,
            ScenarioKind::TransientSpike,
            ScenarioKind::BiasDrift,
            ScenarioKind::Dropout,
        ]
    } else {
        vec![ScenarioKind::parse(&scenario_arg)
            .ok_or_else(|| format!("unknown scenario: {scenario_arg}"))?]
    };

    let mut reports = Vec::new();
    for kind in scenarios {
        let scenario = generate_scenario(
            kind,
            seed,
            samples,
            1_000,
            EvaluationSplit::BlindEvaluation,
        ).map_err(str::to_owned)?;

        // Only observations are passed to the detector; the report attaches
        // evaluator truth and manifest after the score computation.
        let scores = z_score_baseline(&scenario.observations, 64);
        let metrics = evaluate(&scenario, &scores, 4.0);
        reports.push(RunReport {
            report_version: 1,
            suite: "anomaly_detection",
            code_revision: revision.clone(),
            evaluator: "fixed-initial-window-z-score",
            threshold_sigma: 4.0,
            baseline_samples: 64,
            manifest: scenario.manifest,
            metrics,
        });
    }

    println!("{}", serde_json::to_string_pretty(&reports)
        .map_err(|error| format!("could not encode report: {error}"))?);
    Ok(())
}
