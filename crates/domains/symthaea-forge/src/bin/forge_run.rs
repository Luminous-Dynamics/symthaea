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
//! Candidate staging is temporary and restored by the library. Persistent proposal output is
//! required to resolve outside the canonical workspace and existing evidence files are never
//! overwritten.

use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
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
    let workspace_root = args.workspace_root.canonicalize()?;
    let target_file_abs = if args.target_file.is_absolute() {
        args.target_file.clone()
    } else {
        workspace_root.join(&args.target_file)
    };
    let out_dir = prepare_output_dir(&workspace_root, &args.out_dir)?;

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
        workspace_root,
        test_filter: args.test_filter,
        features: args.features,
        bench_example: args.bench_example,
        population: args.population,
        generations: args.generations,
        seed: args.seed,
    };

    let outcome = run_search(&config)?;

    println!(
        "candidates: {} attempted, {} no-eligible-mutation, {} failed compile, {} failed test, {} failed benchmark, {} passed correctness, {} improved search score",
        outcome.stats.candidates_attempted,
        outcome.stats.candidates_no_eligible_mutation,
        outcome.stats.candidates_failed_compile,
        outcome.stats.candidates_failed_test,
        outcome.stats.candidates_failed_benchmark,
        outcome.stats.candidates_passed_correctness,
        outcome.stats.candidates_improved_benchmark,
    );
    if let Some(baseline) = outcome.baseline_benchmark_score {
        println!("baseline benchmark score: {baseline:.2}");
    }

    match outcome.best {
        Some(cert) => {
            let cert_path = out_dir.join("certificate.json");
            write_new(&cert_path, cert.to_json_pretty()?.as_bytes())?;
            let report_path = out_dir.join("report.md");
            write_new(&report_path, render_report(&cert).as_bytes())?;
            println!("\n{}", cert.summary());
            println!(
                "\nHONEST FRAMING: this is a proposed search candidate, not an applied change or replicated performance result. Review {} and {} before any separate promotion decision for {}.",
                cert_path.display(),
                report_path.display(),
                args.target_fn,
            );
        }
        None => {
            println!(
                "\nNo mutation in this bounded run both passed every configured correctness gate and satisfied the search-selection rule. This is a legitimate negative result; do not weaken gates to manufacture a winner."
            );
        }
    }

    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Prepare a persistent proposal directory that is provably outside the canonical workspace at
/// the moment it is created. This prevents `--out` from becoming a hidden repository-write path.
fn prepare_output_dir(workspace_root: &Path, requested: &Path) -> anyhow::Result<PathBuf> {
    let requested = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        std::env::current_dir()?.join(requested)
    };
    let normalized = lexical_normalize(&requested)?;

    // Resolve the deepest existing ancestor to catch symlink aliases into the workspace before
    // creating missing output components.
    let mut ancestor = normalized.as_path();
    let mut missing = Vec::<OsString>::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("output path has no existing ancestor"))?;
        missing.push(name.to_os_string());
        ancestor = ancestor
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output path has no existing ancestor"))?;
    }
    let mut predicted = ancestor.canonicalize()?;
    for component in missing.iter().rev() {
        predicted.push(component);
    }
    if predicted.starts_with(workspace_root) {
        anyhow::bail!(
            "--out must resolve outside the workspace; refusing persistent Forge output under {}",
            workspace_root.display()
        );
    }

    std::fs::create_dir_all(&normalized)?;
    let canonical = normalized.canonicalize()?;
    if canonical.starts_with(workspace_root) {
        anyhow::bail!(
            "--out resolved inside the workspace after creation; refusing {}",
            canonical.display()
        );
    }
    Ok(canonical)
}

fn lexical_normalize(path: &Path) -> anyhow::Result<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    anyhow::bail!("output path escapes its filesystem root");
                }
            }
            Component::Normal(part) => out.push(part),
        }
    }
    Ok(out)
}

fn render_report(cert: &symthaea_forge::ForgeCertificate) -> String {
    let bench_section = match &cert.benchmark {
        Some(benchmark) => {
            let relative = benchmark
                .improvement_fraction
                .map(|fraction| format!("{:+.2}%", fraction * 100.0))
                .unwrap_or_else(|| "undefined (zero baseline)".into());
            format!(
                "- Metric: `{}`\n- Parent score: `{:.2}`\n- Candidate score: `{:.2}`\n- Absolute improvement: `{:+.2}`\n- Relative improvement: `{relative}`\n- Evidence class: single-run search heuristic; replication not established\n",
                benchmark.metric_name,
                benchmark.baseline_score,
                benchmark.candidate_score,
                benchmark.absolute_improvement,
            )
        }
        None => "- No benchmark configured; correctness-only search mode.\n".to_string(),
    };
    let gates_section: String = cert
        .gates
        .iter()
        .map(|gate| {
            format!(
                "- `{}`: {}\n",
                gate.gate,
                if gate.passed { "PASS" } else { "FAIL" }
            )
        })
        .collect();
    let lineage_section: String = if cert.mutation_history.len() > 1 {
        let entries: String = cert
            .mutation_history
            .iter()
            .enumerate()
            .map(|(index, mutation)| {
                format!(
                    "{}. gen {}: **{}** — {}\n",
                    index + 1,
                    mutation.generation,
                    mutation.operator,
                    mutation.detail
                )
            })
            .collect();
        format!(
            "\n**⚠ This diff compounds {n} mutations**. Review each ordered transformation, not only the final label:\n\n{entries}\n",
            n = cert.mutation_history.len(),
        )
    } else {
        String::new()
    };
    format!(
        "# symthaea-forge candidate report\n\nGenerated: {generated} ms since epoch\nTarget: `{file}::{func}` (package `{package}`)\nGit SHA at search time: `{sha}`\nGeneration found: {generation}\nMutation: **{op}** — {detail}\n{lineage}\n## Gates\n{gates}\n## Benchmark/search heuristic\n{bench}\n## Before\n```rust\n{before}\n```\n\n## After\n```rust\n{after}\n```\n\n## Authority boundary\nThis report is a human-review proposal. It does not establish replicated performance, production eligibility, or permission to modify runtime code. Promotion must occur through a separate reviewed/evidence-bearing path.\n",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_normalize_resolves_parent_components() {
        let path = Path::new("/tmp/a/../b/./c");
        assert_eq!(lexical_normalize(path).unwrap(), PathBuf::from("/tmp/b/c"));
    }

    #[test]
    fn write_new_refuses_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "forge-output-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("evidence.txt");
        write_new(&file, b"first").unwrap();
        assert!(write_new(&file, b"second").is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"first");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn output_inside_workspace_is_rejected() {
        let root = std::env::temp_dir().join(format!(
            "forge-workspace-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let canonical = root.canonicalize().unwrap();
        assert!(prepare_output_dir(&canonical, &root.join("forge-out")).is_err());
        let _ = std::fs::remove_dir_all(root);
    }
}