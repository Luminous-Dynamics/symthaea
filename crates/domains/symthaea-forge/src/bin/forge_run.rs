// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLI entry point for a real `symthaea-forge` search run.
//!
//! Example:
//! ```sh
//! cargo run -p symthaea-forge --bin forge-run -- \
//!     --target-file crates/core/symthaea-core/src/consciousness_metrics/entropy.rs \
//!     --target-fn entropy_histogram \
//!     --package symthaea-core \
//!     --bench-example forge_bench_entropy_histogram \
//!     --population 6 --generations 3 \
//!     --out /tmp/forge-out/entropy-histogram
//! ```
//!
//! Never writes to the real source tree -- only to `--out`.

use std::path::PathBuf;
use symthaea_forge::{ForgeConfig, run_search};

struct Args {
    target_file: PathBuf,
    target_fn: String,
    package: String,
    workspace_root: PathBuf,
    test_filter: Option<String>,
    features: Vec<String>,
    bench_example: Option<String>,
    population: usize,
    generations: usize,
    seed: u64,
    out_dir: PathBuf,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut target_file = None;
    let mut target_fn = None;
    let mut package = None;
    let mut workspace_root = std::env::current_dir()?;
    let mut test_filter = None;
    let mut features = Vec::new();
    let mut bench_example = None;
    let mut population = 6usize;
    let mut generations = 3usize;
    let mut seed = 0xF0_5E_5EEDu64;
    let mut out_dir = None;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut next = |name: &str| {
            iter.next()
                .ok_or_else(|| anyhow::anyhow!("missing value for {name}"))
        };
        match arg.as_str() {
            "--target-file" => target_file = Some(PathBuf::from(next("--target-file")?)),
            "--target-fn" => target_fn = Some(next("--target-fn")?),
            "--package" => package = Some(next("--package")?),
            "--workspace-root" => workspace_root = PathBuf::from(next("--workspace-root")?),
            "--test-filter" => test_filter = Some(next("--test-filter")?),
            "--feature" => features.push(next("--feature")?),
            "--bench-example" => bench_example = Some(next("--bench-example")?),
            "--population" => population = next("--population")?.parse()?,
            "--generations" => generations = next("--generations")?.parse()?,
            "--seed" => seed = next("--seed")?.parse()?,
            "--out" => out_dir = Some(PathBuf::from(next("--out")?)),
            other => anyhow::bail!("unrecognized argument: {other}"),
        }
    }

    Ok(Args {
        target_file: target_file.ok_or_else(|| anyhow::anyhow!("--target-file is required"))?,
        target_fn: target_fn.ok_or_else(|| anyhow::anyhow!("--target-fn is required"))?,
        package: package.ok_or_else(|| anyhow::anyhow!("--package is required"))?,
        workspace_root,
        test_filter,
        features,
        bench_example,
        population,
        generations,
        seed,
        out_dir: out_dir.ok_or_else(|| anyhow::anyhow!("--out is required"))?,
    })
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let target_file_abs = args.workspace_root.join(&args.target_file);

    println!(
        "symthaea-forge: searching {}::{} (population={}, generations={})",
        args.target_file.display(),
        args.target_fn,
        args.population,
        args.generations
    );

    let config = ForgeConfig {
        target_file: target_file_abs,
        target_function: args.target_fn.clone(),
        package: args.package,
        workspace_root: args.workspace_root,
        test_filter: args.test_filter,
        features: args.features,
        bench_example: args.bench_example,
        population: args.population,
        generations: args.generations,
        seed: args.seed,
    };

    let outcome = run_search(&config)?;

    println!(
        "candidates: {} attempted, {} no-eligible-mutation, {} failed compile, {} failed test, \
         {} passed correctness, {} improved benchmark",
        outcome.stats.candidates_attempted,
        outcome.stats.candidates_no_eligible_mutation,
        outcome.stats.candidates_failed_compile,
        outcome.stats.candidates_failed_test,
        outcome.stats.candidates_passed_correctness,
        outcome.stats.candidates_improved_benchmark,
    );
    if let Some(baseline) = outcome.baseline_benchmark_score {
        println!("baseline benchmark score: {baseline:.2}");
    }

    match outcome.best {
        Some(cert) => {
            std::fs::create_dir_all(&args.out_dir)?;
            let cert_path = args.out_dir.join("certificate.json");
            std::fs::write(&cert_path, cert.to_json_pretty()?)?;
            let report_path = args.out_dir.join("report.md");
            std::fs::write(&report_path, render_report(&cert))?;
            println!("\n{}", cert.summary());
            println!(
                "\nHONEST FRAMING: this is a *proposed* candidate, not applied. Review \
                 {} and {} before deciding whether to copy the mutated function into \
                 {}.",
                cert_path.display(),
                report_path.display(),
                args.target_fn,
            );
        }
        None => {
            println!(
                "\nNo mutation in this run both passed every correctness gate and beat the \
                 baseline benchmark. This is a legitimate outcome, not a failure of the search \
                 infrastructure -- {} is a small, well-scoped function and this run's mutation \
                 operator set / population / generation budget did not happen to find an \
                 improving structural change. Widening the operator set, increasing \
                 population/generations, or targeting a less already-optimized function are the \
                 next things to try.",
                args.target_fn
            );
        }
    }

    Ok(())
}

fn render_report(cert: &symthaea_forge::ForgeCertificate) -> String {
    let bench_section = match &cert.benchmark {
        Some(b) => format!(
            "- Metric: `{}`\n- Baseline: `{:.2}`\n- Candidate: `{:.2}`\n- Improvement: `{:+.2}%`\n",
            b.metric_name,
            b.baseline_score,
            b.candidate_score,
            b.improvement_fraction * 100.0
        ),
        None => "- No benchmark configured for this run.\n".to_string(),
    };
    let gates_section: String = cert
        .gates
        .iter()
        .map(|g| {
            format!(
                "- `{}`: {}\n",
                g.gate,
                if g.passed { "PASS" } else { "FAIL" }
            )
        })
        .collect();
    let lineage_section: String = if cert.mutation_history.len() > 1 {
        let entries: String = cert
            .mutation_history
            .iter()
            .map(|m| {
                format!(
                    "{}. gen {}: **{}** — {}\n",
                    m.generation + 1,
                    m.generation,
                    m.operator,
                    m.detail
                )
            })
            .collect();
        format!(
            "\n**⚠ This diff compounds {n} mutations, not just the one above** -- elitism \
             means each generation mutates the previous generation's winner, so the \
             Before/After diff below reflects all {n} applied in order:\n\n{entries}\n",
            n = cert.mutation_history.len(),
        )
    } else {
        String::new()
    };
    format!(
        "# symthaea-forge candidate report\n\n\
         Generated: {generated} ms since epoch\n\
         Target: `{file}::{func}` (package `{package}`)\n\
         Git SHA at search time: `{sha}`\n\
         Generation found: {generation}\n\
         Mutation: **{op}** — {detail}\n\
         {lineage}\n\
         ## Gates\n{gates}\n\
         ## Benchmark\n{bench}\n\
         ## Before\n```rust\n{before}\n```\n\n\
         ## After\n```rust\n{after}\n```\n\n\
         ## How to apply (never automatic)\n\
         Manually replace the `{func}` function body in `{file}` with the \"After\" \
         block above, after reading it carefully — this certificate proves the gates \
         that were run and their results, not that the change is a good idea in every \
         context the function is called from. If the lineage above shows more than one \
         mutation, review EACH one individually before applying -- do not assume the \
         label on this report is the only thing that changed.\n",
        generated = cert.generated_at_unix_ms,
        file = cert.target_file.display(),
        func = cert.target_function,
        package = cert.package,
        sha = cert.git_sha.as_deref().unwrap_or("unknown"),
        generation = cert.generation,
        op = cert.mutation_operator,
        detail = cert.mutation_detail,
        lineage = lineage_section,
        gates = gates_section,
        bench = bench_section,
        before = cert.before_source,
        after = cert.after_source,
    )
}
