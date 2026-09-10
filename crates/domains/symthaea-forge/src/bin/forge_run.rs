// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLI entry point for a real `symthaea-forge` search run.
//!
//! Candidate staging is temporary and restored by the library. Persistent output resolves outside
//! the canonical workspace and is create-new only. Trace + observation payloads are read back and
//! cross-validated before survivor artifacts or the terminal manifest are written. An aborted
//! search seals its evidence bundle first and only then returns a non-zero process result.

use serde::Serialize;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use symthaea_algorithms::observation::{ObservationObject, ObservationStore};
use symthaea_forge::certificate::full_source_artifact_id;
use symthaea_forge::{
    read_completed_manifest, run_search_recorded, validate_forge_trace_observations, ForgeBundleManifest,
    ForgeBundleOutcome, ForgeCandidate, ForgeConfig, ForgeTraceEvent, SearchFailure, SearchOutcome,
    SearchRecord, ABORT_FILE, CANDIDATE_FILE, CERTIFICATE_FILE, MANIFEST_FILE, OBSERVATIONS_FILE,
    REPORT_FILE, TRACE_FILE,
};

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

#[derive(Serialize)]
struct AbortFile<'a> {
    terminal: &'static str,
    phase: &'a str,
    detail: &'a str,
    candidates_attempted: usize,
    candidates_no_eligible_mutation: usize,
    candidates_failed_compile: usize,
    candidates_failed_test: usize,
    candidates_failed_benchmark: usize,
    candidates_passed_correctness: usize,
    candidates_selected_by_search: usize,
    baseline_score_bits: Option<u64>,
    best_artifact_id: Option<&'a str>,
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
        target_function: args.target_fn,
        package: args.package,
        workspace_root,
        test_filter: args.test_filter,
        features: args.features,
        bench_example: args.bench_example,
        population: args.population,
        generations: args.generations,
        seed: args.seed,
    };

    match run_search_recorded(&config)? {
        SearchRecord::Completed(outcome) => persist_completed(&out_dir, outcome),
        SearchRecord::Aborted(failure) => {
            let summary = failure.summary();
            persist_aborted(&out_dir, failure)?;
            anyhow::bail!("{summary}")
        }
    }
}

fn persist_completed(out_dir: &Path, outcome: SearchOutcome) -> anyhow::Result<()> {
    validate_forge_trace_observations(&outcome.trace, &outcome.observations)?;
    preflight_bundle_names(out_dir)?;
    persist_trace_and_observations(out_dir, &outcome.trace, &outcome.observations)?;

    print_stats(
        &outcome.stats,
        outcome.baseline_benchmark_score,
        "completed",
    );
    let bundle_outcome = match outcome.best.as_ref() {
        Some(candidate) => {
            persist_survivor(
                out_dir,
                candidate,
                "This survivor came from a bounded search that reached SearchCompleted.",
            )?;
            ForgeBundleOutcome::Winner
        }
        None => {
            println!(
                "No mutation in this bounded run both passed every configured correctness gate and satisfied the search-selection rule. This is a legitimate completed negative result."
            );
            ForgeBundleOutcome::NoWinner
        }
    };

    seal_bundle(out_dir, bundle_outcome)?;
    Ok(())
}

fn persist_aborted(out_dir: &Path, failure: SearchFailure) -> anyhow::Result<()> {
    validate_forge_trace_observations(&failure.trace, &failure.observations)?;
    preflight_bundle_names(out_dir)?;
    persist_trace_and_observations(out_dir, &failure.trace, &failure.observations)?;

    let abort = AbortFile {
        terminal: "search-aborted",
        phase: &failure.phase,
        detail: &failure.detail,
        candidates_attempted: failure.stats.candidates_attempted,
        candidates_no_eligible_mutation: failure.stats.candidates_no_eligible_mutation,
        candidates_failed_compile: failure.stats.candidates_failed_compile,
        candidates_failed_test: failure.stats.candidates_failed_test,
        candidates_failed_benchmark: failure.stats.candidates_failed_benchmark,
        candidates_passed_correctness: failure.stats.candidates_passed_correctness,
        candidates_selected_by_search: failure.stats.candidates_selected_by_search,
        baseline_score_bits: failure.baseline_benchmark_score.map(f64::to_bits),
        best_artifact_id: failure.best.as_ref().map(|candidate| candidate.artifact_id().as_str()),
    };
    write_new(
        &out_dir.join(ABORT_FILE),
        &serde_json::to_vec_pretty(&abort)?,
    )?;

    print_stats(
        &failure.stats,
        failure.baseline_benchmark_score,
        "aborted",
    );
    if let Some(candidate) = failure.best.as_ref() {
        persist_survivor(
            out_dir,
            candidate,
            "The search later aborted. This is only the last previously validated continuation survivor, not a completed-run winner.",
        )?;
    }

    seal_bundle(out_dir, ForgeBundleOutcome::Aborted)?;
    eprintln!(
        "Forge search aborted during `{}`; retained reconstructable evidence in {}",
        failure.phase,
        out_dir.display()
    );
    Ok(())
}

fn persist_trace_and_observations(
    out_dir: &Path,
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
) -> anyhow::Result<()> {
    validate_forge_trace_observations(trace, observations)?;
    let expected_observation_snapshot = observations.snapshot_id()?;
    let trace_path = out_dir.join(TRACE_FILE);
    let observations_path = out_dir.join(OBSERVATIONS_FILE);

    write_new(&trace_path, &serde_json::to_vec_pretty(trace)?)?;
    let observation_objects: Vec<ObservationObject> = observations.objects().cloned().collect();
    write_new(
        &observations_path,
        &serde_json::to_vec_pretty(&observation_objects)?,
    )?;

    let persisted_trace: Vec<ForgeTraceEvent> =
        serde_json::from_slice(&std::fs::read(&trace_path)?)?;
    if persisted_trace != trace {
        anyhow::bail!("persisted Forge search trace changed during immediate read-back");
    }
    let persisted_objects: Vec<ObservationObject> =
        serde_json::from_slice(&std::fs::read(&observations_path)?)?;
    let persisted_store = ObservationStore::from_objects(persisted_objects)?;
    validate_forge_trace_observations(&persisted_trace, &persisted_store)?;
    if persisted_store.snapshot_id()? != expected_observation_snapshot {
        anyhow::bail!("persisted Forge observation store changed during immediate read-back");
    }
    Ok(())
}

fn persist_survivor(
    out_dir: &Path,
    candidate: &ForgeCandidate,
    run_note: &str,
) -> anyhow::Result<()> {
    candidate.validate()?;
    let candidate_path = out_dir.join(CANDIDATE_FILE);
    let cert_path = out_dir.join(CERTIFICATE_FILE);
    let report_path = out_dir.join(REPORT_FILE);

    write_new(&candidate_path, candidate.full_source().as_bytes())?;
    let persisted_source = std::fs::read_to_string(&candidate_path)?;
    let persisted_id = full_source_artifact_id(&persisted_source);
    if &persisted_id != candidate.artifact_id() {
        anyhow::bail!(
            "persisted candidate source does not match Forge artifact identity: expected {}, observed {}",
            candidate.artifact_id(),
            persisted_id
        );
    }

    let cert = candidate.certificate();
    write_new(&cert_path, cert.to_json_pretty()?.as_bytes())?;
    write_new(&report_path, render_report(cert, run_note).as_bytes())?;
    println!("\n{}", cert.summary());
    Ok(())
}

fn print_stats(stats: &symthaea_forge::SearchStats, baseline: Option<f64>, terminal: &str) {
    println!(
        "search {terminal}: {} attempted, {} no-eligible-mutation, {} failed compile, {} failed test, {} failed benchmark, {} passed correctness, {} selected for continuation",
        stats.candidates_attempted,
        stats.candidates_no_eligible_mutation,
        stats.candidates_failed_compile,
        stats.candidates_failed_test,
        stats.candidates_failed_benchmark,
        stats.candidates_passed_correctness,
        stats.candidates_selected_by_search,
    );
    if let Some(baseline) = baseline {
        println!("baseline benchmark score: {baseline:.2}");
    }
}

fn seal_bundle(out_dir: &Path, outcome: ForgeBundleOutcome) -> anyhow::Result<()> {
    let manifest_path = out_dir.join(MANIFEST_FILE);
    let manifest = ForgeBundleManifest::observe(out_dir, outcome)?;
    write_new(&manifest_path, manifest.to_json_pretty()?.as_bytes())?;
    let verified = read_completed_manifest(out_dir)?;
    if verified.id != manifest.id || verified.outcome != outcome {
        anyhow::bail!("Forge bundle manifest changed during immediate read-back validation");
    }
    println!(
        "sealed Forge evidence bundle: {} (outcome {:?}, manifest {})",
        manifest_path.display(),
        outcome,
        manifest.id
    );
    Ok(())
}

fn preflight_bundle_names(out_dir: &Path) -> anyhow::Result<()> {
    let paths = [
        out_dir.join(TRACE_FILE),
        out_dir.join(OBSERVATIONS_FILE),
        out_dir.join(ABORT_FILE),
        out_dir.join(CANDIDATE_FILE),
        out_dir.join(CERTIFICATE_FILE),
        out_dir.join(REPORT_FILE),
        out_dir.join(MANIFEST_FILE),
    ];
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    ensure_absent(&refs)
}

fn ensure_absent(paths: &[&Path]) -> anyhow::Result<()> {
    if let Some(existing) = paths.iter().find(|path| path.exists()) {
        anyhow::bail!(
            "refusing to overwrite existing Forge output evidence: {}",
            existing.display()
        );
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn prepare_output_dir(workspace_root: &Path, requested: &Path) -> anyhow::Result<PathBuf> {
    let requested = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        std::env::current_dir()?.join(requested)
    };
    let normalized = lexical_normalize(&requested)?;

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

fn render_report(cert: &symthaea_forge::ForgeCertificate, run_note: &str) -> String {
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
    let lineage_entries: String = cert
        .mutation_history
        .iter()
        .enumerate()
        .map(|(index, mutation)| {
            format!(
                "{}. gen {}: **{}** — {}\n   - parent: `{}`\n   - child: `{}`\n   - transformation: `{}`\n",
                index + 1,
                mutation.generation,
                mutation.operator,
                mutation.detail,
                mutation.parent_artifact_id,
                mutation.candidate_artifact_id,
                mutation.transformation_id,
            )
        })
        .collect();
    format!(
        "# symthaea-forge candidate report\n\n{run_note}\n\nGenerated: {generated} ms since epoch\nTarget: `{file}::{func}` (package `{package}`)\nGit SHA at search time: `{sha}`\nGeneration found: {generation}\nBaseline full-file artifact: `{baseline_artifact}`\nCandidate full-file artifact: `{candidate_artifact}`\nMutation: **{op}** — {detail}\n\n## Ordered artifact lineage\n{lineage}\n## Gates\n{gates}\n## Benchmark/search heuristic\n{bench}\n## Before\n```rust\n{before}\n```\n\n## After\n```rust\n{after}\n```\n\n## Authority boundary\n`candidate.rs` is the exact full-file survivor identified above. `search-trace.json` and `observations.json` retain reconstructable search memory. `bundle-manifest.json` is the terminal persistence marker. This is a human-review proposal only; it does not establish replicated performance, production eligibility, or permission to modify runtime code.\n",
        generated = cert.generated_at_unix_ms,
        file = cert.target_file.display(),
        func = cert.target_function,
        package = cert.package,
        sha = cert.git_sha.as_deref().unwrap_or("unknown"),
        generation = cert.generation,
        baseline_artifact = cert.baseline_artifact_id,
        candidate_artifact = cert.candidate_artifact_id,
        op = cert.mutation_operator,
        detail = cert.mutation_detail,
        lineage = lineage_entries,
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
        let root = std::env::temp_dir().join(format!("forge-output-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("evidence.txt");
        write_new(&file, b"first").unwrap();
        assert!(write_new(&file, b"second").is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"first");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn ensure_absent_detects_bundle_collision_before_writes() {
        let root = std::env::temp_dir().join(format!(
            "forge-output-preflight-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let manifest = root.join(MANIFEST_FILE);
        std::fs::write(&manifest, "existing").unwrap();
        assert!(preflight_bundle_names(&root).is_err());
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
