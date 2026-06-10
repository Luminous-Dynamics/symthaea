// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase 6 Session 1 — HDC fingerprint + cluster separation check.
//!
//! For each Lake-verified miniF2F goal (accepted or rejected), encode
//! the raw Lean source as a 16,384-dimensional `BinaryHV` fingerprint:
//! token-bundle with positional binding. Then measure whether the
//! Lake-accepted goals cluster *more tightly* with each other (in HDC
//! cosine similarity) than they do with Lake-rejected goals.
//!
//! Per `docs/phase6-scoping/cognition-to-lean-bridge.md`, this is the
//! **go / no-go** for the learned-cascade direction. If cluster
//! separation exists, Session 2 (cascade variants) is worth running.
//! If not, the null result is informative — it tells us HDC-over-
//! source-tokens doesn't carry useful structural information about
//! math goals, and Phase 7 should pivot.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p symthaea-lean-bridge --example phase6_fingerprint
//! ```
//!
//! Reads `docs/phase3-results/ingest_baseline_seed42_n50.csv` to get
//! Lake outcomes. Reads raw `.lean` sources from the corpus.
//! Outputs CSV summary + separation metrics.
//!
//! # Input
//!
//! CSV with columns `name, stage, ..., lake_check`. Rows where
//! `stage = translated` and `lake_check ∈ {accepted, rejected}` are
//! used. (Lake errors and parse failures are excluded — they don't
//! test the signature hypothesis.)
//!
//! # Output
//!
//! Three numbers for each signature comparison:
//! - `within_accept_sim`: mean pairwise cosine similarity among
//!   Lake-accepted goals.
//! - `within_reject_sim`: same for Lake-rejected.
//! - `between_sim`: mean pairwise similarity across accept×reject.
//!
//! If accept goals cluster, `within_accept_sim > between_sim`.
//! Effect size = `within_accept_sim − between_sim`.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use symthaea_core::hdc::binary_hv::BinaryHV;

fn main() -> ExitCode {
    let root = match locate_corpus() {
        Some(r) => r,
        None => {
            eprintln!("error: miniF2F corpus not found");
            return ExitCode::SUCCESS;
        }
    };
    let csv_path = match locate_csv() {
        Some(p) => p,
        None => {
            eprintln!("error: ingest_baseline_seed42_n50.csv not found");
            return ExitCode::SUCCESS;
        }
    };

    let goals = load_goals(&csv_path, &root);
    eprintln!("Phase 6 Session 1 — HDC fingerprint cluster separation");
    eprintln!("  inputs: {}", csv_path.display());
    eprintln!(
        "  translated + Lake-verified: {} ({} accepted, {} rejected)",
        goals.len(),
        goals
            .iter()
            .filter(|g| g.outcome == Outcome::Accepted)
            .count(),
        goals
            .iter()
            .filter(|g| g.outcome == Outcome::Rejected)
            .count()
    );

    if goals.is_empty() {
        eprintln!("  no goals to fingerprint — stopping");
        return ExitCode::SUCCESS;
    }

    let signatures: Vec<(String, Outcome, BinaryHV)> = goals
        .iter()
        .map(|g| (g.name.clone(), g.outcome, fingerprint(&g.source)))
        .collect();

    // CSV: pairwise similarities with flags.
    println!("name_a,name_b,outcome_a,outcome_b,pair_kind,cosine_similarity");
    let mut within_accept = 0.0f64;
    let mut within_accept_n = 0u64;
    let mut within_reject = 0.0f64;
    let mut within_reject_n = 0u64;
    let mut between = 0.0f64;
    let mut between_n = 0u64;

    for i in 0..signatures.len() {
        for j in (i + 1)..signatures.len() {
            let (ref a_name, a_outcome, ref a_sig) = signatures[i];
            let (ref b_name, b_outcome, ref b_sig) = signatures[j];
            let sim = a_sig.cosine_similarity(b_sig) as f64;
            let kind = match (a_outcome, b_outcome) {
                (Outcome::Accepted, Outcome::Accepted) => {
                    within_accept += sim;
                    within_accept_n += 1;
                    "within_accept"
                }
                (Outcome::Rejected, Outcome::Rejected) => {
                    within_reject += sim;
                    within_reject_n += 1;
                    "within_reject"
                }
                _ => {
                    between += sim;
                    between_n += 1;
                    "between"
                }
            };
            println!(
                "{},{},{:?},{:?},{},{:.6}",
                a_name, b_name, a_outcome, b_outcome, kind, sim
            );
        }
    }

    let mean_within_accept = within_accept / within_accept_n.max(1) as f64;
    let mean_within_reject = within_reject / within_reject_n.max(1) as f64;
    let mean_between = between / between_n.max(1) as f64;

    eprintln!();
    eprintln!("━━━ Cluster separation scorecard ━━━");
    eprintln!(
        "  within-accept mean cosine:  {:+.6}  ({} pairs)",
        mean_within_accept, within_accept_n
    );
    eprintln!(
        "  within-reject mean cosine:  {:+.6}  ({} pairs)",
        mean_within_reject, within_reject_n
    );
    eprintln!(
        "  between     mean cosine:  {:+.6}  ({} pairs)",
        mean_between, between_n
    );
    let effect = mean_within_accept - mean_between;
    eprintln!("  effect size (accept − between): {:+.6}", effect);

    let verdict = if effect.abs() < 0.005 {
        "NULL: signatures do not separate Lake outcome (|effect| < 0.005)"
    } else if effect > 0.0 {
        "SIGNAL: Lake-accepted goals cluster tighter than baseline — Session 2 worth running"
    } else {
        "ANTI-SIGNAL: accepted goals cluster LOOSER than baseline — unexpected, investigate"
    };
    eprintln!("  {verdict}");

    ExitCode::SUCCESS
}

// ─── Goal + outcome ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Accepted,
    Rejected,
}

struct Goal {
    name: String,
    source: String,
    outcome: Outcome,
}

fn load_goals(csv_path: &Path, corpus_root: &Path) -> Vec<Goal> {
    let csv = fs::read_to_string(csv_path).unwrap_or_default();
    let mut goals = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        // Simple CSV: name,stage,error_category,error_message,lake_check
        // We only care about stage=translated + lake_check ∈ {accepted,
        // rejected}. Stage + outcome fields have no quoted commas in
        // those specific cases, so basic split suffices for this path.
        let fields: Vec<&str> = line.splitn(5, ',').collect();
        if fields.len() < 5 {
            continue;
        }
        let name = fields[0].trim();
        let stage = fields[1].trim();
        let lake = fields[4].trim();
        if stage != "translated" {
            continue;
        }
        let outcome = match lake {
            "accepted" => Outcome::Accepted,
            "rejected" => Outcome::Rejected,
            _ => continue, // skip lake_error / not_run
        };
        // Locate the raw source file under Valid/ or Test/
        let path_candidates = [
            corpus_root.join("Valid").join(format!("{}.lean", name)),
            corpus_root.join("Test").join(format!("{}.lean", name)),
        ];
        let source = path_candidates
            .iter()
            .find_map(|p| fs::read_to_string(p).ok())
            .unwrap_or_default();
        if source.is_empty() {
            continue;
        }
        goals.push(Goal {
            name: name.to_string(),
            source,
            outcome,
        });
    }
    goals
}

// ─── HDC fingerprint ────────────────────────────────────────────────

/// Token-bag fingerprint with positional binding. Each token maps to
/// a deterministic BinaryHV via hash→seed; positions permute so
/// ordered structure is preserved. Final signature is the bundle of
/// all position-bound tokens.
fn fingerprint(source: &str) -> BinaryHV {
    let tokens = tokenize_lean(source);
    if tokens.is_empty() {
        return BinaryHV::basis(0);
    }
    let token_hvs: Vec<BinaryHV> = tokens
        .iter()
        .enumerate()
        .map(|(pos, tok)| hv_for_token(tok).permute(pos))
        .collect();
    BinaryHV::bundle(&token_hvs)
}

/// Very coarse Lean tokenization — splits on whitespace + a few
/// high-signal punctuation characters. Good enough to distinguish
/// `∀`-rich goals from equation goals from inequality goals; not
/// meant to be a real parser.
fn tokenize_lean(source: &str) -> Vec<String> {
    const SEPS: &[char] = &[
        ' ', '\t', '\n', '\r', '(', ')', '{', '}', '[', ']', ',', ':', ';', '.',
    ];
    source
        .split(|c: char| SEPS.contains(&c))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn hv_for_token(tok: &str) -> BinaryHV {
    let mut h = DefaultHasher::new();
    tok.hash(&mut h);
    BinaryHV::random(h.finish())
}

// ─── I/O ────────────────────────────────────────────────────────────

fn locate_corpus() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()?
        .parent()?
        .join("data/benchmarks/minif2f/MiniF2F");
    root.exists().then_some(root)
}

fn locate_csv() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .parent()?
        .join("docs/phase3-results/ingest_baseline_seed42_n50.csv");
    p.exists().then_some(p)
}
