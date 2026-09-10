// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mutate -> verify -> measure -> select search loop.
//!
//! Forge uses a deliberately narrow in-place staging window because Cargo must see candidate
//! source on disk. Every candidate is explicitly restored before search decisions continue; the
//! RAII drop path is only a panic/early-return fallback. A configured benchmark that fails is a
//! rejected evaluation, never an alias for correctness-only mode.
//!
//! Search survivors retain the exact full-file source bytes that were staged and checked. Their
//! certificates content-address those bytes and every accepted parent->child mutation. This makes
//! Forge a useful candidate generator for `symthaea-algorithms` without upgrading its local search
//! heuristic into evidence-grade performance or promotion authority.

use crate::certificate::{
    BenchmarkEvidence, ForgeCandidate, ForgeCertificate, MutationRecord, full_source_artifact_id,
    gate_result_to_evidence,
};
use crate::fitness::{EvaluationTarget, run_benchmark, run_correctness_gates};
use crate::mutations::{Mutator, find_function_body_mut};
use crate::sandbox::Sandbox;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ForgeConfig {
    pub target_file: PathBuf,
    pub target_function: String,
    pub package: String,
    pub workspace_root: PathBuf,
    pub test_filter: Option<String>,
    pub features: Vec<String>,
    pub bench_example: Option<String>,
    pub population: usize,
    pub generations: usize,
    pub seed: u64,
}

#[derive(Debug, Default)]
pub struct SearchStats {
    pub candidates_attempted: usize,
    pub candidates_no_eligible_mutation: usize,
    pub candidates_failed_compile: usize,
    pub candidates_failed_test: usize,
    /// Correctness-passing candidates whose configured benchmark failed to produce one valid
    /// finite result. These are rejected, not retained as structure-only candidates.
    pub candidates_failed_benchmark: usize,
    pub candidates_passed_correctness: usize,
    pub candidates_improved_benchmark: usize,
}

impl SearchStats {
    pub fn candidates_failed_correctness(&self) -> usize {
        self.candidates_failed_compile + self.candidates_failed_test
    }
}

pub struct SearchOutcome {
    pub stats: SearchStats,
    pub baseline_benchmark_score: Option<f64>,
    /// Best search candidate under the configured heuristic. The object contains the exact full
    /// source bytes plus a self-validating certificate; it is still only a proposal.
    pub best: Option<ForgeCandidate>,
}

fn build_eval_target<'a>(config: &'a ForgeConfig, features: &'a [&'a str]) -> EvaluationTarget<'a> {
    EvaluationTarget {
        package: &config.package,
        workspace_root: &config.workspace_root,
        test_filter: config.test_filter.as_deref(),
        features,
        bench_example: config.bench_example.as_deref(),
    }
}

pub fn run_search(config: &ForgeConfig) -> anyhow::Result<SearchOutcome> {
    if config.population == 0 {
        anyhow::bail!("Forge population must be positive");
    }
    if config.generations == 0 {
        anyhow::bail!("Forge generations must be positive");
    }

    let target_dir = config
        .target_file
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| config.workspace_root.clone());
    let sandbox = Sandbox::new(&config.workspace_root, &[target_dir])?;

    let original_source = std::fs::read_to_string(&config.target_file)?;
    let baseline_artifact_id = full_source_artifact_id(&original_source);
    let mut current_best_source = original_source.clone();
    let mut current_best_artifact_id = baseline_artifact_id.clone();
    let feature_refs: Vec<&str> = config.features.iter().map(String::as_str).collect();
    let target = || build_eval_target(config, &feature_refs);

    // A broken baseline cannot meaningfully define candidate correctness or performance.
    let baseline_gates = run_correctness_gates(&target());
    if !baseline_gates.iter().all(|gate| gate.passed) {
        anyhow::bail!(
            "baseline ({file}) failed its own correctness gates -- refusing to search on top of a broken starting point. Gate results: {gates:?}",
            file = config.target_file.display(),
            gates = baseline_gates
                .iter()
                .map(|gate| (gate.gate.label(), gate.passed))
                .collect::<Vec<_>>(),
        );
    }
    let baseline_benchmark = run_benchmark(&target())
        .map_err(|error| anyhow::anyhow!("configured baseline benchmark failed: {error}"))?;
    let baseline_score = baseline_benchmark.as_ref().map(|benchmark| benchmark.score);

    let mutator = Mutator::default();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut stats = SearchStats::default();
    let mut best: Option<ForgeCandidate> = None;
    let mut current_best_score = baseline_score;
    let mut mutation_history: Vec<MutationRecord> = Vec::new();

    for generation in 0..config.generations {
        let mut generation_winner: Option<ForgeCandidate> = None;

        for _ in 0..config.population {
            stats.candidates_attempted += 1;

            let mut file: syn::File = match syn::parse_str(&current_best_source) {
                Ok(file) => file,
                Err(_) => {
                    stats.candidates_no_eligible_mutation += 1;
                    continue;
                }
            };
            let Some(body) = find_function_body_mut(&mut file, &config.target_function) else {
                anyhow::bail!(
                    "function `{}` not found in {}",
                    config.target_function,
                    config.target_file.display()
                );
            };
            let Some(mutation) = mutator.mutate_one(body, &mut rng) else {
                stats.candidates_no_eligible_mutation += 1;
                continue;
            };
            let candidate_source = render_file(&file);
            let candidate_artifact_id = full_source_artifact_id(&candidate_source);

            let mut staged = sandbox.stage(&config.target_file)?;
            staged.write(&candidate_source)?;
            let gates = run_correctness_gates(&target());
            let all_passed = gates.iter().all(|gate| gate.passed);

            if !all_passed {
                let failed_compile = gates
                    .iter()
                    .any(|gate| gate.gate == crate::fitness::Gate::Compile && !gate.passed);
                if failed_compile {
                    stats.candidates_failed_compile += 1;
                } else {
                    stats.candidates_failed_test += 1;
                }
                staged.restore()?;
                continue;
            }
            stats.candidates_passed_correctness += 1;

            let bench = match run_benchmark(&target()) {
                Ok(benchmark) => benchmark,
                Err(error) => {
                    stats.candidates_failed_benchmark += 1;
                    staged.restore()?;
                    eprintln!(
                        "forge: rejecting candidate after benchmark evaluation failure: {error}"
                    );
                    continue;
                }
            };

            // Source restoration is part of candidate validity. Nothing is retained until this
            // succeeds, and the certificate still names the exact bytes that were just evaluated.
            staged.restore()?;

            let benchmark_evidence = match (&bench, current_best_score) {
                (Some(candidate), Some(parent_score)) => {
                    let absolute_improvement = parent_score - candidate.score;
                    let improvement_fraction = if parent_score == 0.0 {
                        None
                    } else {
                        Some(absolute_improvement / parent_score.abs())
                    };
                    Some(BenchmarkEvidence {
                        metric_name: candidate.metric_name.clone(),
                        baseline_score: parent_score,
                        candidate_score: candidate.score,
                        absolute_improvement,
                        improvement_fraction,
                    })
                }
                (None, None) => None,
                _ => anyhow::bail!("Forge benchmark configuration changed during one search run"),
            };

            let improved = match (&bench, current_best_score) {
                (Some(candidate), Some(parent_score)) => candidate.score < parent_score,
                (None, None) => true,
                _ => false,
            };
            if !improved {
                continue;
            }

            stats.candidates_improved_benchmark += 1;
            let before_source = extract_function_source(&original_source, &config.target_function)
                .unwrap_or_default();
            let after_source = extract_function_source(&candidate_source, &config.target_function)
                .unwrap_or_default();
            let accepted_mutation = MutationRecord::new(
                generation,
                mutation.operator,
                mutation.detail.clone(),
                current_best_artifact_id.clone(),
                candidate_artifact_id.clone(),
            );
            let mut candidate_history = mutation_history.clone();
            candidate_history.push(accepted_mutation.clone());
            let certificate = ForgeCertificate {
                generated_at_unix_ms: now_millis(),
                target_file: config.target_file.clone(),
                target_function: config.target_function.clone(),
                package: config.package.clone(),
                git_sha: current_git_sha(&config.workspace_root),
                generation,
                baseline_artifact_id: baseline_artifact_id.clone(),
                candidate_artifact_id,
                mutation_operator: accepted_mutation.operator.clone(),
                mutation_detail: accepted_mutation.detail.clone(),
                mutation_history: candidate_history,
                gates: gates.iter().map(gate_result_to_evidence).collect(),
                benchmark: benchmark_evidence,
                before_source,
                after_source,
            };
            let candidate = ForgeCandidate::new(certificate, candidate_source)?;

            let should_replace_generation_winner = match (&generation_winner, candidate.certificate().benchmark.as_ref()) {
                (None, _) => true,
                (Some(previous), Some(candidate_benchmark)) => previous
                    .certificate()
                    .benchmark
                    .as_ref()
                    .is_some_and(|previous_benchmark| {
                        candidate_benchmark.candidate_score < previous_benchmark.candidate_score
                    }),
                // In correctness-only mode there is no evidence-grade ordering between two
                // passing mutations, so keep the first deterministic survivor of the generation.
                (Some(_), None) => false,
            };
            if should_replace_generation_winner {
                generation_winner = Some(candidate);
            }
        }

        if let Some(candidate) = generation_winner {
            candidate.validate()?;
            if let Some(benchmark) = &candidate.certificate().benchmark {
                current_best_score = Some(benchmark.candidate_score);
            }
            current_best_source = candidate.full_source().to_string();
            current_best_artifact_id = candidate.artifact_id().clone();
            mutation_history = candidate.certificate().mutation_history.clone();
            best = Some(candidate);
        }
    }

    Ok(SearchOutcome {
        stats,
        baseline_benchmark_score: baseline_score,
        best,
    })
}

fn render_file(file: &syn::File) -> String {
    prettyplease::unparse(file)
}

fn extract_function_source(full_source: &str, fn_name: &str) -> Option<String> {
    let file: syn::File = syn::parse_str(full_source).ok()?;
    for item in &file.items {
        match item {
            syn::Item::Fn(function) if function.sig.ident == fn_name => {
                return Some(prettyplease::unparse(&syn::File {
                    shebang: None,
                    attrs: vec![],
                    items: vec![syn::Item::Fn(function.clone())],
                }));
            }
            syn::Item::Impl(implementation) => {
                for impl_item in &implementation.items {
                    if let syn::ImplItem::Fn(function) = impl_item {
                        if function.sig.ident == fn_name {
                            return Some(quote::quote!(#function).to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn current_git_sha(workspace_root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(workspace_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_function_source_finds_a_free_function() {
        let src = "fn other() {}\nfn target(x: i32) -> i32 { x + 1 }\n";
        let extracted = extract_function_source(src, "target").unwrap();
        assert!(extracted.contains("fn target"));
        assert!(!extracted.contains("fn other"));
    }

    #[test]
    fn extract_function_source_finds_an_impl_method() {
        let src = "struct S; impl S { fn target(&self) -> i32 { 1 } }";
        let extracted = extract_function_source(src, "target").unwrap();
        assert!(extracted.contains("fn target"));
    }

    #[test]
    fn extract_function_source_returns_none_for_missing_name() {
        let src = "fn present() -> i32 { 1 }";
        assert!(extract_function_source(src, "absent").is_none());
    }

    #[test]
    fn render_file_produces_parseable_output() {
        let file: syn::File = syn::parse_str("fn f(x: i32) -> i32 { x + 1 }").unwrap();
        let rendered = render_file(&file);
        assert!(syn::parse_str::<syn::File>(&rendered).is_ok());
    }
}