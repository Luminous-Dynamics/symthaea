// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NixEval Benchmark with fast-mode RAG fallback enabled.
//!
//! Same 94-problem corpus as `nix_eval_benchmark.rs`, same scoring rules,
//! but runs each problem through `generate_nix_with_rag_fast_using` (RAG +
//! cache + parse-only verification) instead of the bare `generate_nix`.
//!
//! Reports per-source breakdown so we can attribute the score lift to:
//!   - cache_hit         (Tier 3 LearnedIdiomCache)
//!   - idiom             (hand-written templates in nix_codegen)
//!   - rag_draft         (commented suggestions from search.nixos.org)
//!   - skeleton          (no idiom, no RAG match — fallback `{ }`)
//!
//! Cache is per-run (tempfile) so the user's real cache stays untouched.
//!
//! Run:
//!   cargo run --release --features code_generation --example nix_eval_benchmark_rag

use std::path::PathBuf;

use symthaea::language::learned_idioms::LearnedIdiomCache;
use symthaea::language::nix_codegen::{
    NixGenSource, NixIntent, SelfImproveSource, generate_nix_with_rag_fast_using,
};
use symthaea::language::nix_eval_corpus::{NixProblem, problems};

#[derive(Default)]
struct ScoreCard {
    total: usize,
    intent_correct: usize,
    parse_correct: usize,
    expected_present: usize,
    forbidden_absent: usize,
    full_pass: usize,
    by_intent: std::collections::HashMap<NixIntent, (usize, usize)>,
    cache_hits: usize,
    idiom: usize,
    rag_draft: usize,
    skeleton: usize,
    other: usize,
}

fn evaluate(
    problem: &NixProblem,
    cache: &LearnedIdiomCache,
) -> (
    bool,
    bool,
    bool,
    bool,
    String,
    NixGenSource,
    SelfImproveSource,
) {
    let (result, src) = generate_nix_with_rag_fast_using(problem.prompt, 3, cache);
    let intent_ok = result.base.intent == problem.expected_intent;
    let parse_ok = if problem.require_parse {
        result.base.parses
    } else {
        true
    };

    let expected_ok = problem
        .expected_substrings
        .iter()
        .all(|s| result.base.code.contains(s));

    let forbidden_ok = problem
        .forbidden_substrings
        .iter()
        .all(|s| !result.base.code.contains(s));

    (
        intent_ok,
        parse_ok,
        expected_ok,
        forbidden_ok,
        result.base.code,
        result.base.source,
        src,
    )
}

fn main() {
    let problems = problems();
    let mut card = ScoreCard::default();
    card.total = problems.len();

    // Per-run isolated cache.
    let cache_path: PathBuf = std::env::temp_dir().join(format!(
        "symthaea_nixeval_rag_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&cache_path);
    let cache = LearnedIdiomCache::with_cache_path(cache_path.clone());

    println!("┌─────────────────────────────────────────────────────────");
    println!(
        "│ NixEval Benchmark (RAG fast-mode) — {} problems",
        problems.len()
    );
    println!("│ Pipeline: cache → idiom → parse → option-index → RAG draft");
    println!("└─────────────────────────────────────────────────────────");

    for (i, p) in problems.iter().enumerate() {
        let (intent_ok, parse_ok, exp_ok, forbid_ok, code, source, src) = evaluate(p, &cache);
        if intent_ok {
            card.intent_correct += 1;
        }
        if parse_ok {
            card.parse_correct += 1;
        }
        if exp_ok {
            card.expected_present += 1;
        }
        if forbid_ok {
            card.forbidden_absent += 1;
        }
        let full = intent_ok && parse_ok && exp_ok && forbid_ok;
        if full {
            card.full_pass += 1;
        }
        let entry = card
            .by_intent
            .entry(p.expected_intent)
            .or_insert((0usize, 0usize));
        entry.0 += 1;
        if full {
            entry.1 += 1;
        }

        // Source attribution
        match src {
            SelfImproveSource::LearnedCache => card.cache_hits += 1,
            SelfImproveSource::FreshlyGenerated { .. } => match source {
                NixGenSource::Idiom => card.idiom += 1,
                NixGenSource::RagDraft => card.rag_draft += 1,
                NixGenSource::Skeleton => card.skeleton += 1,
                NixGenSource::Empty => card.other += 1,
            },
        }

        let mark = if full { "✓" } else { "✗" };
        let src_tag = match src {
            SelfImproveSource::LearnedCache => "cache",
            SelfImproveSource::FreshlyGenerated { .. } => match source {
                NixGenSource::Idiom => "idiom",
                NixGenSource::RagDraft => "RAG  ",
                NixGenSource::Skeleton => "skel ",
                NixGenSource::Empty => "empty",
            },
        };
        let detail = format!(
            "intent={} parse={} expected={} forbidden={}",
            if intent_ok { "✓" } else { "✗" },
            if parse_ok { "✓" } else { "✗" },
            if exp_ok { "✓" } else { "✗" },
            if forbid_ok { "✓" } else { "✗" },
        );
        println!(
            "  {} #{:02} [{}] {:48} | {}",
            mark,
            i + 1,
            src_tag,
            p.prompt.chars().take(48).collect::<String>(),
            detail
        );
        if !full {
            for s in p.expected_substrings.iter() {
                if !code.contains(s) {
                    println!("       missing: {s}");
                }
            }
            for s in p.forbidden_substrings.iter() {
                if code.contains(s) {
                    println!("       leaked:  {s}");
                }
            }
        }
    }

    let n = card.total as f32;
    println!("\n╔═════════════════════════════════════════════════════════");
    println!(
        "║ Intent classification:  {}/{} ({:.0}%)",
        card.intent_correct,
        card.total,
        card.intent_correct as f32 / n * 100.0
    );
    println!(
        "║ Parses successfully:    {}/{} ({:.0}%)",
        card.parse_correct,
        card.total,
        card.parse_correct as f32 / n * 100.0
    );
    println!(
        "║ Expected substrings:    {}/{} ({:.0}%)",
        card.expected_present,
        card.total,
        card.expected_present as f32 / n * 100.0
    );
    println!(
        "║ No forbidden leakage:   {}/{} ({:.0}%)",
        card.forbidden_absent,
        card.total,
        card.forbidden_absent as f32 / n * 100.0
    );
    println!(
        "║ FULL PASS (all 4):      {}/{} ({:.0}%)",
        card.full_pass,
        card.total,
        card.full_pass as f32 / n * 100.0
    );
    println!("╠═════════════════════════════════════════════════════════");
    println!(
        "║ Source attribution:  cache={}  idiom={}  RAG={}  skel={}  empty={}",
        card.cache_hits, card.idiom, card.rag_draft, card.skeleton, card.other
    );
    println!("╚═════════════════════════════════════════════════════════");

    let mut intents: Vec<_> = card.by_intent.iter().collect();
    intents.sort_by_key(|(k, _)| format!("{:?}", k));
    println!("\nPer-intent full pass rate:");
    for (intent, (total, full)) in intents {
        println!(
            "  {:?}: {}/{} ({:.0}%)",
            intent,
            full,
            total,
            *full as f32 / *total as f32 * 100.0
        );
    }

    let _ = std::fs::remove_file(&cache_path);
}
