// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AGW plan Phase 4.1 — held-out Hendrycks ETHICS via the LIVE moral-judgment path.
//!
//! Exercises exactly what the facade actually wires
//! (`EthicsEngine::new(MoralParser::new(), MoralAlgebra::default_dim(), None, None)`,
//! `symthaea/mod.rs`) — `value_evaluator` and `harmonies_integrator` are both
//! `None` in the live facade, so Stage 1 (MoralParser + MoralAlgebra, the
//! fixed-anchor HDC similarity path) IS the dominant live signal, not a
//! simplification of it. `EthicsEngineInput`/`EthicsEngine::evaluate()` are
//! `pub(crate)` and unreachable from an example binary, so this reproduces
//! Stage 1's exact score formula from `cognitive_loop/ethics_engine.rs`
//! (verified against source 2026-07-10) using the public `MoralParser`/
//! `MoralAlgebra` API directly: consent violation -> -0.8; else
//! 0.6*(good_sim-bad_sim) + 0.4*deontological_score, clamped to [-1, 1].
//! Predicts "wrong" iff score < 0.0 (the natural midpoint of a symmetric
//! [-1, 1] score) — this collapses the 4-way `MoralVerdict` signal to
//! ETHICS's binary label; a different threshold is a legitimate follow-up,
//! not attempted here.
//!
//! Dataset: raw Hendrycks ETHICS `*_test.csv` files already present at
//! `datasets/ethics/raw/ethics/{domain}/` (a real clone of
//! github.com/hendrycks/ethics — NOT the never-populated
//! `data/moral_datasets/ethics.json` the older `benchmark_moral_unified.rs`
//! expects, which does not exist on this host). Frozen sample: first
//! N_PER_DOMAIN rows of each `*_test.csv` (not train, not test_hard) —
//! deterministic and reproducible. `utilitarianism` is excluded: its task
//! shape (paired better/worse sentences) doesn't map onto a single
//! wrongness score without a different harness, matching the precedent set
//! by the March 2026 Spinozist/CfC benchmark, which also excluded it from
//! its final results table (see `memory/moral_classification_march27.md`).
//!
//! Run: `cargo run --release --example agw_ethics_holdout`
//! (no LLM, no GPU — pure HDC similarity; expect seconds, not the
//! minutes-to-hours the LLM-bound coding benchmarks need.)

use std::error::Error;
use std::path::Path;

use symthaea::hdc::moral_algebra::MoralAlgebra;
use symthaea::hdc::moral_parser::MoralParser;

const N_PER_DOMAIN: usize = 200;
const DATA_ROOT: &str = "datasets/ethics/raw/ethics";

struct Example {
    text: String,
    /// Hendrycks convention, verified against sample rows: 1 = wrong, 0 = not wrong.
    label: i32,
}

fn load_commonsense(path: &Path, n: usize) -> Result<Vec<Example>, Box<dyn Error>> {
    // commonsense: label,input,is_short,edited
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for rec in rdr.records().take(n) {
        let rec = rec?;
        let label: i32 = rec.get(0).ok_or("missing label")?.parse()?;
        let text = rec.get(1).ok_or("missing input")?.to_string();
        out.push(Example { text, label });
    }
    Ok(out)
}

fn load_scenario_only(path: &Path, n: usize) -> Result<Vec<Example>, Box<dyn Error>> {
    // justice, virtue: label,scenario
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for rec in rdr.records().take(n) {
        let rec = rec?;
        let label: i32 = rec.get(0).ok_or("missing label")?.parse()?;
        let text = rec.get(1).ok_or("missing scenario")?.to_string();
        out.push(Example { text, label });
    }
    Ok(out)
}

fn load_deontology(path: &Path, n: usize) -> Result<Vec<Example>, Box<dyn Error>> {
    // deontology: label,scenario,excuse -- concatenate, matching how a human
    // reader would judge the combined request+excuse pair.
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for rec in rdr.records().take(n) {
        let rec = rec?;
        let label: i32 = rec.get(0).ok_or("missing label")?.parse()?;
        let scenario = rec.get(1).ok_or("missing scenario")?;
        let excuse = rec.get(2).ok_or("missing excuse")?;
        out.push(Example {
            text: format!("{scenario} {excuse}"),
            label,
        });
    }
    Ok(out)
}

/// Reproduces `EthicsEngine::evaluate()`'s Stage-1 score formula exactly
/// (`cognitive_loop/ethics_engine.rs`, verified 2026-07-10). Returns the
/// continuous score in [-1, 1]; caller thresholds at 0.
fn live_stage1_score(parser: &MoralParser, algebra: &MoralAlgebra, text: &str) -> f64 {
    let encoded = parser.parse_and_encode(text, algebra);
    let (good_sim, bad_sim) = match encoded.judge(algebra) {
        Some(j) => (j.good_similarity, j.bad_similarity),
        None => (0.0, 0.0),
    };
    if encoded.is_consent_violation() {
        return -0.8;
    }
    let deont = algebra.judge_deontological(text);
    let base_score = (good_sim - bad_sim).clamp(-1.0, 1.0) as f64;
    let deont_factor = deont.score.clamp(-1.0, 1.0) as f64;
    (base_score * 0.6 + deont_factor * 0.4).clamp(-1.0, 1.0)
}

type Loader = fn(&Path, usize) -> Result<Vec<Example>, Box<dyn Error>>;

fn main() -> Result<(), Box<dyn Error>> {
    let parser = MoralParser::new();
    let algebra = MoralAlgebra::default_dim();

    let domains: Vec<(&str, &str, Loader)> = vec![
        ("commonsense", "cm_test.csv", load_commonsense as Loader),
        ("deontology", "deontology_test.csv", load_deontology),
        ("justice", "justice_test.csv", load_scenario_only),
        ("virtue", "virtue_test.csv", load_scenario_only),
    ];

    println!("AGW Phase 4.1 -- held-out ETHICS via the live Stage-1 moral-judgment path");
    println!(
        "N={N_PER_DOMAIN} per domain (first N rows of each frozen *_test.csv, not train, not test_hard)\n"
    );

    let mut overall_correct = 0usize;
    let mut overall_total = 0usize;

    for (domain, file, loader) in domains {
        let path = Path::new(DATA_ROOT).join(domain).join(file);
        let examples = loader(&path, N_PER_DOMAIN)?;
        let mut correct = 0usize;
        for ex in &examples {
            let score = live_stage1_score(&parser, &algebra, &ex.text);
            let predicted_wrong = score < 0.0;
            let actual_wrong = ex.label == 1;
            if predicted_wrong == actual_wrong {
                correct += 1;
            }
        }
        let n = examples.len();
        let acc = 100.0 * correct as f64 / n as f64;
        println!("{domain:>12}: {correct:>4}/{n:<4} = {acc:.1}%");
        overall_correct += correct;
        overall_total += n;
    }

    let overall_acc = 100.0 * overall_correct as f64 / overall_total as f64;
    println!(
        "\n{:>12}: {:>4}/{:<4} = {:.1}%",
        "OVERALL", overall_correct, overall_total, overall_acc
    );
    println!("(random baseline for a balanced binary task: ~50%)");

    Ok(())
}
