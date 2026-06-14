// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use symthaea_fractal_time_lab::BenchmarkConfig;
use symthaea_fractal_time_lab::metrics::{scorecards_to_csv, scorecards_to_json_array};
use symthaea_fractal_time_lab::report::scorecards_to_markdown_report;
use symthaea_fractal_time_lab::runner::run_benchmark_run;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run Symthaea Fractal Time Lab exploratory benchmarks"
)]
struct Args {
    /// Seed for reproducible null models.
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Number of null-model trials.
    #[arg(short, long, default_value_t = 32)]
    trials: usize,

    /// Output only JSON scorecards, without run metadata.
    #[arg(long)]
    json: bool,

    /// Output full JSON run metadata.
    #[arg(long)]
    json_run: bool,

    /// Output only CSV scorecards.
    #[arg(long)]
    csv: bool,

    /// Output Markdown benchmark report.
    #[arg(long)]
    markdown: bool,

    /// Write JSON scorecards to this path.
    #[arg(long)]
    json_out: Option<PathBuf>,

    /// Write full JSON benchmark run to this path.
    #[arg(long)]
    json_run_out: Option<PathBuf>,

    /// Write CSV scorecard to this path.
    #[arg(long)]
    csv_out: Option<PathBuf>,

    /// Write Markdown report to this path.
    #[arg(long)]
    markdown_out: Option<PathBuf>,

    /// Exit non-zero if any exploratory benchmark threshold fails.
    #[arg(long)]
    fail_on_benchmark_fail: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = BenchmarkConfig {
        seed: args.seed,
        trials: args.trials,
    };

    let run = run_benchmark_run(config);
    let json_scorecards = scorecards_to_json_array(&run.scorecards);
    let json_run = run.to_json();
    let csv = scorecards_to_csv(&run.scorecards);
    let markdown = scorecards_to_markdown_report(&run);

    write_optional(args.json_out.as_ref(), &json_scorecards)?;
    write_optional(args.json_run_out.as_ref(), &json_run)?;
    write_optional(args.csv_out.as_ref(), &csv)?;
    write_optional(args.markdown_out.as_ref(), &markdown)?;

    if args.json {
        println!("{json_scorecards}");
    } else if args.json_run {
        println!("{json_run}");
    } else if args.csv {
        println!("{csv}");
    } else if args.markdown {
        println!("{markdown}");
    } else {
        println!("--- Symthaea Fractal Time Lab: Benchmark Suite v0.5 ---");
        println!(
            "seed={}, trials={}",
            run.config.seed,
            run.config.trials.max(1)
        );
        println!("epistemic_status={}\n", run.epistemic_status);

        for card in &run.scorecards {
            println!("{}", card.compact_line());
            println!("  hypothesis: {}", card.hypothesis);
            println!("  caveat: {}\n", card.caveat);
        }

        println!("--- Claim Scope ---");
        for claim in &run.claims {
            println!("claim: {claim}");
        }
        for non_claim in &run.non_claims {
            println!("non_claim: {non_claim}");
        }
    }

    if args.fail_on_benchmark_fail && !run.all_passed() {
        std::process::exit(2);
    }

    Ok(())
}

fn write_optional(
    path: Option<&PathBuf>,
    contents: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, contents)?;
    }

    Ok(())
}
