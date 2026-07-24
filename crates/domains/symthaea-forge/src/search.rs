// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The mutate -> verify -> benchmark -> select loop (Tier 2.1 of
//! `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`).
//!
//! Design choice worth stating plainly: every candidate is staged on disk
//! just long enough for `cargo check`/`test`/the benchmark example to see
//! it, then **always restored** immediately afterward -- win or lose. The
//! search's notion of "current best" lives entirely in memory
//! ([`SearchOutcome::best`]), never on disk. The real source file is never
//! left mutated between candidates, and is never mutated at all beyond the
//! lifetime of a single gate-running window. The winning candidate (if
//! any) is written only to a separate output directory as a proposal --
//! never applied in place. A human decides whether to actually copy it
//! over the real file.

use crate::certificate::{BenchmarkEvidence, ForgeCertificate, gate_result_to_evidence};
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
    /// Failed at the `Compile` gate (didn't even build) -- distinguished
    /// from `candidates_failed_test` because a run with a high compile
    /// failure rate points at a mutation-operator bug (generating invalid
    /// syntax/types), while a high test failure rate is the expected,
    /// healthy outcome of trying semantically-wrong-but-well-typed
    /// mutations like an arithmetic operator swap.
    pub candidates_failed_compile: usize,
    pub candidates_failed_test: usize,
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
    /// The best-performing candidate found, if any candidate both passed
    /// every correctness gate *and* beat the baseline benchmark.
    pub best: Option<ForgeCertificate>,
}

/// A plain closure can't express "the returned `EvaluationTarget` borrows
/// from both `config` and a separately-lived `features` slice" -- closures
/// don't support the kind of per-call generic lifetime inference that a
/// named `fn` with explicit lifetime parameters does, so every call site
/// gets pinned to one fixed lifetime chosen at closure-definition time.
/// Named function instead.
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
    let target_dir = config
        .target_file
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| config.workspace_root.clone());
    let sandbox = Sandbox::new(&config.workspace_root, &[target_dir])?;

    let original_source = std::fs::read_to_string(&config.target_file)?;
    let mut current_best_source = original_source.clone();

    let feature_refs: Vec<&str> = config.features.iter().map(|s| s.as_str()).collect();

    // Sanity-check the baseline before searching: if the unmodified file
    // doesn't compile/test-pass, there is nothing meaningful to "improve".
    let baseline_gates = run_correctness_gates(&build_eval_target(config, &feature_refs));
    if !baseline_gates.iter().all(|g| g.passed) {
        anyhow::bail!(
            "baseline ({file}) failed its own correctness gates -- refusing to search on top \
             of a broken starting point. Gate results: {gates:?}",
            file = config.target_file.display(),
            gates = baseline_gates
                .iter()
                .map(|g| (g.gate.label(), g.passed))
                .collect::<Vec<_>>(),
        );
    }
    let baseline_benchmark = run_benchmark(&build_eval_target(config, &feature_refs));
    let baseline_score = baseline_benchmark.as_ref().map(|b| b.score);

    let mutator = Mutator::default();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut stats = SearchStats::default();
    let mut best: Option<ForgeCertificate> = None;
    let mut current_best_score = baseline_score;
    // Every generation-winning mutation accepted so far, in order. Elitism
    // means each generation mutates the PREVIOUS generation's winner, so a
    // generation-2 candidate's after_source reflects generation 1's
    // accepted mutation too, compounded with its own -- this history is
    // what lets the certificate say so honestly instead of labeling a
    // 2-mutation diff with only the latest mutation's description (a real
    // gap found the hard way on the first live chi_squared_test campaign).
    let mut mutation_history: Vec<crate::certificate::MutationRecord> = Vec::new();

    for generation in 0..config.generations {
        let mut generation_winner: Option<(String, ForgeCertificate)> = None;

        for _ in 0..config.population {
            stats.candidates_attempted += 1;

            let mut file: syn::File = match syn::parse_str(&current_best_source) {
                Ok(f) => f,
                Err(_) => {
                    // Should be unreachable (we only ever install
                    // syn-regenerated, previously-parseable text as
                    // current_best_source), but fail closed rather than
                    // panic on any future refactor mistake.
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

            let staged = sandbox.stage(&config.target_file)?;
            staged.write(&candidate_source)?;
            let gates = run_correctness_gates(&build_eval_target(config, &feature_refs));
            let all_passed = gates.iter().all(|g| g.passed);

            if !all_passed {
                let failed_compile = gates
                    .iter()
                    .any(|g| g.gate == crate::fitness::Gate::Compile && !g.passed);
                if failed_compile {
                    stats.candidates_failed_compile += 1;
                } else {
                    stats.candidates_failed_test += 1;
                }
                // guard drops at end of loop iteration, restoring the file
                continue;
            }
            stats.candidates_passed_correctness += 1;

            let bench = run_benchmark(&build_eval_target(config, &feature_refs));
            let benchmark_evidence = match (&bench, current_best_score) {
                (Some(b), Some(baseline)) => Some(BenchmarkEvidence {
                    metric_name: b.metric_name.clone(),
                    baseline_score: baseline,
                    candidate_score: b.score,
                    improvement_fraction: (baseline - b.score) / baseline,
                }),
                _ => None,
            };

            let improved = match (&bench, current_best_score) {
                (Some(b), Some(baseline)) => b.score < baseline,
                // No benchmark configured: any correctness-passing mutation
                // counts as a kept candidate (structure-only search mode).
                (None, _) => true,
                _ => false,
            };

            if improved {
                stats.candidates_improved_benchmark += 1;
                let before_source =
                    extract_function_source(&original_source, &config.target_function)
                        .unwrap_or_default();
                let after_source =
                    extract_function_source(&candidate_source, &config.target_function)
                        .unwrap_or_default();
                let mut candidate_history = mutation_history.clone();
                candidate_history.push(crate::certificate::MutationRecord {
                    generation,
                    operator: mutation.operator.to_string(),
                    detail: mutation.detail.clone(),
                });
                let cert = ForgeCertificate {
                    generated_at_unix_ms: now_millis(),
                    target_file: config.target_file.clone(),
                    target_function: config.target_function.clone(),
                    package: config.package.clone(),
                    git_sha: current_git_sha(&config.workspace_root),
                    generation,
                    mutation_operator: mutation.operator.to_string(),
                    mutation_detail: mutation.detail.clone(),
                    mutation_history: candidate_history,
                    gates: gates.iter().map(gate_result_to_evidence).collect(),
                    benchmark: benchmark_evidence,
                    before_source,
                    after_source,
                };
                let is_better_than_generation_winner = generation_winner
                    .as_ref()
                    .and_then(|(_, c)| c.benchmark.as_ref())
                    .zip(cert.benchmark.as_ref())
                    .map(|(prev, this)| this.candidate_score < prev.candidate_score)
                    .unwrap_or(true);
                if is_better_than_generation_winner {
                    generation_winner = Some((candidate_source, cert));
                }
            }
            // staged drops here -> file restored to current_best_source automatically
        }

        if let Some((source, cert)) = generation_winner {
            if let Some(b) = &cert.benchmark {
                current_best_score = Some(b.candidate_score);
            }
            current_best_source = source;
            mutation_history = cert.mutation_history.clone();
            best = Some(cert);
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

/// Best-effort extraction of a single function's source text (for the
/// certificate's before/after fields) by re-parsing and re-rendering just
/// that item, rather than diffing the whole file.
fn extract_function_source(full_source: &str, fn_name: &str) -> Option<String> {
    let file: syn::File = syn::parse_str(full_source).ok()?;
    for item in &file.items {
        match item {
            syn::Item::Fn(f) if f.sig.ident == fn_name => {
                return Some(prettyplease::unparse(&syn::File {
                    shebang: None,
                    attrs: vec![],
                    items: vec![syn::Item::Fn(f.clone())],
                }));
            }
            syn::Item::Impl(imp) => {
                for impl_item in &imp.items {
                    if let syn::ImplItem::Fn(f) = impl_item {
                        if f.sig.ident == fn_name {
                            return Some(quote::quote!(#f).to_string());
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
        .map(|d| d.as_millis())
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
        let reparsed: Result<syn::File, _> = syn::parse_str(&rendered);
        assert!(reparsed.is_ok());
    }
}
