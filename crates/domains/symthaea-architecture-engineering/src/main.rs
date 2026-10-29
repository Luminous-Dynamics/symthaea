use std::env;
use std::fs;
use std::process::ExitCode;

use serde::Deserialize;
use symthaea_architecture_engineering::multi_fidelity_benchmark::{
    run_multi_fidelity_sweep, MultiFidelityLandscape,
};
use symthaea_architecture_engineering::paired_evaluation::{
    evaluate_paired_cohort, EvaluationPlan, TrialObservation,
};
use symthaea_architecture_engineering::synthetic_search::{
    aggregate_runs, enumerate_candidates, exhaustive_oracle, run_synthetic_search,
    SyntheticLandscape, SyntheticProxyMode, SyntheticSearchPolicy,
};
use symthaea_architecture_engineering::validate_frozen_candidate_matrix;

#[derive(Debug, Deserialize)]
struct PairedEvaluationInput {
    plan: EvaluationPlan,
    observations: Vec<TrialObservation>,
}

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  symthaea-architecture-engineering validate-matrix <matrix.json>\n  symthaea-architecture-engineering evaluate-paired <evaluation.json>\n  symthaea-architecture-engineering synthetic-sweep <smooth_tradeoff|deceptive> <seed-count>\n  symthaea-architecture-engineering multi-fidelity-sweep <rank_preserving|rank_reversal|late_emerging_quality|miscalibrated_uncertainty> <seed-count>"
    );
    ExitCode::from(2)
}

fn read_input(path: &str) -> Result<String, ExitCode> {
    fs::read_to_string(path).map_err(|error| {
        eprintln!("error: cannot read {path:?}: {error}");
        ExitCode::FAILURE
    })
}

fn parse_landscape(value: &str) -> Option<SyntheticLandscape> {
    match value {
        "smooth_tradeoff" => Some(SyntheticLandscape::SmoothTradeoff),
        "deceptive" => Some(SyntheticLandscape::Deceptive),
        _ => None,
    }
}

fn parse_multi_fidelity_landscape(value: &str) -> Option<MultiFidelityLandscape> {
    match value {
        "rank_preserving" => Some(MultiFidelityLandscape::RankPreserving),
        "rank_reversal" => Some(MultiFidelityLandscape::RankReversal),
        "late_emerging_quality" => Some(MultiFidelityLandscape::LateEmergingQuality),
        "miscalibrated_uncertainty" => Some(MultiFidelityLandscape::MiscalibratedUncertainty),
        _ => None,
    }
}

fn run_multi_fidelity_sweep_command(args: Vec<String>) -> ExitCode {
    if args.len() != 2 {
        return usage();
    }
    let Some(landscape) = parse_multi_fidelity_landscape(&args[0]) else {
        eprintln!(
            "INVALID: landscape must be rank_preserving, rank_reversal, late_emerging_quality, or miscalibrated_uncertainty"
        );
        return ExitCode::from(2);
    };
    let seed_count = match args[1].parse::<usize>() {
        Ok(value) if value >= 2 && value <= 10_000 => value,
        _ => {
            eprintln!("INVALID: seed-count must be in 2..=10000");
            return ExitCode::from(2);
        }
    };

    let (runs, aggregate) = match run_multi_fidelity_sweep(landscape, seed_count) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("INVALID: multi-fidelity benchmark rejected input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let output = serde_json::json!({
        "schema": "symthaea.multi-fidelity-search-benchmark",
        "version": "0.1.0",
        "qualification_state": "SYNTHETIC_ONLY_NOT_MODEL_EVIDENCE",
        "landscape": landscape,
        "seed_count": seed_count,
        "runs": runs,
        "aggregate": aggregate,
        "claim_ceiling": "Synthetic multi-fidelity scheduling validation only; no neural-model or SOTA result."
    });
    match serde_json::to_string_pretty(&output) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: failed to serialize multi-fidelity report: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_synthetic_sweep(args: Vec<String>) -> ExitCode {
    if args.len() != 2 {
        return usage();
    }
    let Some(landscape) = parse_landscape(&args[0]) else {
        eprintln!("INVALID: landscape must be smooth_tradeoff or deceptive");
        return ExitCode::from(2);
    };
    let seed_count = match args[1].parse::<usize>() {
        Ok(value) if value >= 2 && value <= 10_000 => value,
        _ => {
            eprintln!("INVALID: seed-count must be in 2..=10000");
            return ExitCode::from(2);
        }
    };

    let candidates = enumerate_candidates();
    let oracle = exhaustive_oracle(landscape, 10);
    let budgets = [16usize, 32, 64, 128];
    let policy_modes = [
        (SyntheticSearchPolicy::Random, SyntheticProxyMode::Correlated),
        (SyntheticSearchPolicy::LocalSearch, SyntheticProxyMode::Correlated),
        (SyntheticSearchPolicy::ProxyRanked, SyntheticProxyMode::Correlated),
        (SyntheticSearchPolicy::ProxyRanked, SyntheticProxyMode::Shuffled),
        (SyntheticSearchPolicy::ProxyRanked, SyntheticProxyMode::AntiCorrelated),
    ];
    let mut aggregates = Vec::new();

    for budget in budgets {
        for (policy, proxy_mode) in policy_modes {
            let mut runs = Vec::with_capacity(seed_count);
            for seed in 1..=seed_count as u64 {
                let run = match run_synthetic_search(
                    &candidates,
                    landscape,
                    policy,
                    proxy_mode,
                    budget,
                    seed.wrapping_add(0x5359_4e54_4800_0000),
                ) {
                    Ok(result) => result,
                    Err(error) => {
                        eprintln!("INVALID: synthetic benchmark rejected input: {error}");
                        return ExitCode::FAILURE;
                    }
                };
                runs.push(run);
            }
            match aggregate_runs(&runs) {
                Ok(summary) => aggregates.push(summary),
                Err(error) => {
                    eprintln!("INVALID: cannot aggregate synthetic runs: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    let output = serde_json::json!({
        "schema": "symthaea.synthetic-search-benchmark",
        "version": "0.1.0",
        "qualification_state": "SYNTHETIC_ONLY_NOT_MODEL_EVIDENCE",
        "landscape": landscape,
        "candidate_count": candidates.len(),
        "seed_count": seed_count,
        "oracle": oracle,
        "aggregates": aggregates,
        "claim_ceiling": "Synthetic ground-truth search validation only; no neural-model or SOTA result."
    });
    match serde_json::to_string_pretty(&output) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: failed to serialize benchmark report: {error}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return usage();
    };

    if command == "synthetic-sweep" {
        return run_synthetic_sweep(args.collect());
    }
    if command == "multi-fidelity-sweep" {
        return run_multi_fidelity_sweep_command(args.collect());
    }

    let Some(path) = args.next() else {
        return usage();
    };
    if args.next().is_some() {
        return usage();
    }
    let content = match read_input(&path) {
        Ok(content) => content,
        Err(code) => return code,
    };

    match command.as_str() {
        "validate-matrix" => match validate_frozen_candidate_matrix(&content) {
            Ok(summary) => {
                println!(
                    "VALID: schema={} version={} architecture_candidates={} search_policies={} negative_controls={}",
                    summary.schema,
                    summary.version,
                    summary.architecture_candidate_ids.len(),
                    summary.search_policy_ids.len(),
                    summary.negative_control_ids.len()
                );
                println!("state: frozen design only; no experiments qualified");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("INVALID: {error}");
                ExitCode::FAILURE
            }
        },
        "evaluate-paired" => {
            let input: PairedEvaluationInput = match serde_json::from_str(&content) {
                Ok(input) => input,
                Err(error) => {
                    eprintln!("INVALID: malformed evaluation JSON: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let summary = match evaluate_paired_cohort(&input.plan, &input.observations) {
                Ok(summary) => summary,
                Err(error) => {
                    eprintln!("INVALID: {error}");
                    return ExitCode::FAILURE;
                }
            };
            match serde_json::to_string_pretty(&summary) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: failed to serialize evaluation receipt: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => usage(),
    }
}
