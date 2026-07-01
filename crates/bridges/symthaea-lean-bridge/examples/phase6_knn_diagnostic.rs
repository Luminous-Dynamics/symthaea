// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase 6 Session 1b — kNN diagnostic on HDC signatures.
//!
//! Session 1 showed a small positive cluster-separation signal
//! (+1.4%). That number tells us signatures carry *some* structure,
//! but it doesn't tell us whether the structure is useful for
//! *predicting* Lake outcome. This harness runs a leave-one-out kNN
//! classifier over the signatures:
//!
//! - for each goal, remove it from the pool, find its k=3 nearest
//!   neighbors by HDC cosine, and predict its Lake outcome by
//!   majority vote of those neighbors
//! - compute accuracy; compare against the majority-class baseline
//!   (always predict "accepted")
//!
//! The majority-class baseline on the seed-42 slice is 22/31 = 71.0%.
//! **kNN must beat 71% for signatures to be worth building a learned
//! cascade selector on top of.** If kNN ≤ 71%, Session 2 is not
//! justified — we'd be building infrastructure around an uninformative
//! signal. The null result is the honest conclusion.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p symthaea-lean-bridge --example phase6_knn_diagnostic
//! ```

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use symthaea_core::hdc::binary_hv::BinaryHV;

const K: usize = 3;

fn main() -> ExitCode {
    let (Some(root), Some(csv_path)) = (locate_corpus(), locate_csv()) else {
        eprintln!("error: corpus or CSV not found");
        return ExitCode::SUCCESS;
    };

    let goals = load_goals(&csv_path, &root);
    let n_accepted = goals
        .iter()
        .filter(|g| g.outcome == Outcome::Accepted)
        .count();
    let n_rejected = goals
        .iter()
        .filter(|g| g.outcome == Outcome::Rejected)
        .count();
    let n = goals.len();
    eprintln!("Phase 6 Session 1b — kNN leave-one-out diagnostic (k={K})");
    eprintln!("  goals: {n} ({n_accepted} accepted, {n_rejected} rejected)");
    let baseline_pct = 100.0 * n_accepted as f64 / n.max(1) as f64;
    eprintln!("  majority-class baseline (always predict 'accepted'): {baseline_pct:.1}%");

    if n < K + 2 {
        eprintln!("  too few goals for k={K} leave-one-out");
        return ExitCode::SUCCESS;
    }

    let signatures: Vec<(String, Outcome, BinaryHV)> = goals
        .iter()
        .map(|g| (g.name.clone(), g.outcome, fingerprint(&g.source)))
        .collect();

    println!(
        "name,true_outcome,predicted_outcome,nearest_k,agreement_with_true,vote_for_accept,vote_for_reject"
    );

    let mut correct = 0usize;
    let mut per_class_correct = (0usize, 0usize); // (accepted, rejected)

    for (i, (name, true_outcome, sig)) in signatures.iter().enumerate() {
        // Gather (similarity, neighbor-outcome) excluding self.
        let mut neighbors: Vec<(f32, Outcome)> = signatures
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, (_, out, other_sig))| (sig.cosine_similarity(other_sig), *out))
            .collect();
        // Sort descending by similarity.
        neighbors.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = &neighbors[..K];

        let vote_accept = top_k
            .iter()
            .filter(|(_, o)| *o == Outcome::Accepted)
            .count();
        let vote_reject = K - vote_accept;
        let predicted = if vote_accept >= vote_reject {
            Outcome::Accepted
        } else {
            Outcome::Rejected
        };
        let agreement = predicted == *true_outcome;
        if agreement {
            correct += 1;
            match true_outcome {
                Outcome::Accepted => per_class_correct.0 += 1,
                Outcome::Rejected => per_class_correct.1 += 1,
            }
        }

        let top_k_str = top_k
            .iter()
            .map(|(sim, o)| format!("{:?}:{:.3}", o, sim))
            .collect::<Vec<_>>()
            .join("|");
        println!(
            "{},{:?},{:?},{},{},{},{}",
            name, true_outcome, predicted, top_k_str, agreement, vote_accept, vote_reject
        );
    }

    let accuracy = 100.0 * correct as f64 / n as f64;
    let accept_recall = 100.0 * per_class_correct.0 as f64 / n_accepted.max(1) as f64;
    let reject_recall = 100.0 * per_class_correct.1 as f64 / n_rejected.max(1) as f64;

    eprintln!();
    eprintln!("━━━ kNN leave-one-out scorecard ━━━");
    eprintln!("  overall accuracy:         {correct}/{n} = {accuracy:.1}%");
    eprintln!("  majority-class baseline:  {n_accepted}/{n} = {baseline_pct:.1}%");
    eprintln!(
        "  lift over baseline:       {:+.1} pp",
        accuracy - baseline_pct
    );
    eprintln!(
        "  accepted recall:          {}/{} = {accept_recall:.1}%",
        per_class_correct.0, n_accepted
    );
    eprintln!(
        "  rejected recall:          {}/{} = {reject_recall:.1}%",
        per_class_correct.1, n_rejected
    );

    eprintln!();
    let verdict = if accuracy <= baseline_pct + 2.0 {
        // within 2pp of baseline → not informative
        "NULL: kNN does not beat majority-class baseline. Signatures are \
         not useful for prediction. Session 2 (cascade tournament) is NOT \
         justified on this encoder. Either try a richer encoder (cognitive \
         loop's wisdom_hv) or pivot away from the learned-cascade direction."
    } else if reject_recall < 20.0 {
        "WEAK: kNN beats baseline but almost never predicts 'rejected' \
         correctly. Signatures encode 'looks like a normal goal' but not \
         'looks like a hard goal'. Insufficient for cascade selection."
    } else {
        "SIGNAL: kNN beats baseline AND recovers rejected-class with >20% \
         recall. Signatures carry usable predictive information. Session 2 \
         (cascade tournament) is justified."
    };
    eprintln!("  {verdict}");

    ExitCode::SUCCESS
}

// ─── Goal loading (same as phase6_fingerprint.rs) ───────────────────

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
            continue;
        }
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
            _ => continue,
        };
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

// ─── HDC fingerprint (same as phase6_fingerprint.rs) ────────────────

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
