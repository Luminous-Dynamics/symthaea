// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Does the trained recurrent core actually USE temporal state?
//!
//! Suggested by external review of the 2026-07 Broca audit, and it is the cheapest decisive
//! input to the "should we implement real BPTT" question (`SYMTHAEA_BROCA_IMPROVEMENT_PLAN`
//! §B1: `backward_step` is a single-step local rule, not backpropagation through time).
//!
//! # Method
//!
//! Teacher-forced over the canonical suite, two arms differing in exactly one thing:
//!
//! - **NORMAL**  — `reset() + seed_from_thought()` once per sequence, then evolve across
//!   positions. The recurrent state accumulates history.
//! - **RESET**   — `reset() + seed_from_thought()` before *every* position. No accumulated
//!   history survives from one token to the next.
//!
//! Crucially this does **not** blind the model to all context. `forward_step` composes its
//! input as `thought ⊗ token_emb[prev_token] ⊗ permute(pos_base, pos)`, so the RESET arm still
//! sees the immediately-preceding token and the position. The ablation therefore isolates
//! precisely one thing: **does accumulated recurrent evolution add anything beyond the direct
//! (thought, prev_token, position) input?**
//!
//! # Decision rule, fixed before running
//!
//! - `ΔCE = CE_reset − CE_normal` **> 1.0 nat**  → recurrence carries real information;
//!   improving temporal credit assignment (BPTT) is plausibly worthwhile.
//! - `|ΔCE| < 0.2 nat`                            → recurrence contributes ~nothing as trained;
//!   the core is functioning as a feed-forward map of (thought, prev_token, pos), and BPTT
//!   would be improving a pathway the model does not currently use.
//! - between                                      → partial; report and judge with the number.
//!
//! A near-zero delta is the *informative* outcome: combined with the encoder finding
//! (degenerate input), it would mean Broca has two independent structural problems, which
//! materially changes whether an encoder redesign alone is worth funding.
//!
//! ```bash
//! cargo run --release -p symthaea-broca --example broca_state_reset_ablation -- [ckpt] [jsonl]
//! ```

use symthaea_broca::evaluation::CanonicalEvalDataset;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;

/// Numerically stable teacher-forced cross-entropy for one target.
fn ce(logits: &[f32], target: usize) -> Option<f32> {
    if target >= logits.len() {
        return None;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }
    Some(-((logits[target] - max) - sum.ln()))
}

fn rank_of(logits: &[f32], target: usize) -> usize {
    if target >= logits.len() {
        return logits.len();
    }
    let t = logits[target];
    logits.iter().filter(|&&l| l > t).count()
}

#[derive(Default)]
struct Acc {
    ce_sum: f64,
    rank_sum: f64,
    n: usize,
    top10: usize,
    top100: usize,
}

impl Acc {
    fn push(&mut self, c: f32, r: usize) {
        self.ce_sum += c as f64;
        self.rank_sum += r as f64;
        self.n += 1;
        if r < 10 {
            self.top10 += 1;
        }
        if r < 100 {
            self.top100 += 1;
        }
    }
    fn ce(&self) -> f64 {
        self.ce_sum / self.n.max(1) as f64
    }
    fn rank(&self) -> f64 {
        self.rank_sum / self.n.max(1) as f64
    }
    fn report(&self, label: &str) {
        println!(
            "{label:<10} tokens={:<6} meanCE={:<8.4} ppl={:<12.4e} meanRank={:<9.1} top10={:.4} top100={:.4}",
            self.n,
            self.ce(),
            self.ce().exp(),
            self.rank(),
            self.top10 as f64 / self.n.max(1) as f64,
            self.top100 as f64 / self.n.max(1) as f64,
        );
    }
}

fn main() -> anyhow::Result<()> {
    let mut a = std::env::args().skip(1);
    let ckpt = a.next().unwrap_or_else(|| {
        "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin".to_string()
    });
    let canon = a.next().unwrap_or_else(|| {
        "crates/domains/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl".to_string()
    });
    let genesis = GenesisSeed::from_phrase("broca-training-default");
    let (mut r#gen, ..) = BrocaGenerator::from_checkpoint(&ckpt, &genesis)
        .map_err(|e| anyhow::anyhow!("loading {ckpt}: {e}"))?;

    let tok = r#gen.tokenizer().clone();
    let ds = CanonicalEvalDataset::from_jsonl(&canon)?.to_training_dataset(&tok);
    println!(
        "cases={} vocab={} MAX_POS=24\n",
        ds.pairs.len(),
        r#gen.controller().vocab_size()
    );

    let mut normal = Acc::default();
    let mut reset = Acc::default();
    let mut skipped = 0usize;

    for reset_every_token in [false, true] {
        let acc = if reset_every_token {
            &mut reset
        } else {
            &mut normal
        };
        for pair in &ds.pairs {
            if pair.target_ids.is_empty() {
                continue;
            }
            let channels = pair.to_thought_channels();
            let thought = r#gen.encoder().encode(&channels);

            r#gen.controller_mut().reset();
            r#gen.controller_mut().seed_from_thought(&thought);

            let mut prev = tok.thought_id;
            for (pos, &tid) in pair.target_ids.iter().take(24).enumerate() {
                if reset_every_token {
                    // Wipe accumulated recurrent evolution. The model still receives
                    // prev_token and pos through the input binding below.
                    r#gen.controller_mut().reset();
                    r#gen.controller_mut().seed_from_thought(&thought);
                }
                let logits = r#gen.controller_mut().forward_step(&thought, prev, pos);
                match ce(&logits, tid as usize) {
                    Some(c) => acc.push(c, rank_of(&logits, tid as usize)),
                    None => skipped += 1,
                }
                prev = tid;
            }
        }
    }

    println!("=== teacher-forced, canonical suite ===");
    normal.report("NORMAL");
    reset.report("RESET");
    if skipped > 0 {
        println!("(skipped {skipped} non-finite logit steps)");
    }

    let d_ce = reset.ce() - normal.ce();
    let d_rank = reset.rank() - normal.rank();
    println!("\nΔCE   (reset − normal) = {d_ce:+.4} nats");
    println!("Δrank (reset − normal) = {d_rank:+.1}");

    println!("\n=== VERDICT (rule fixed before running) ===");
    // Added after the first run: the original rule only branched on d_ce > 1.0, |d_ce| < 0.2,
    // and "between" — it assumed removing recurrence could at best be NEUTRAL. A large
    // *negative* delta (reset markedly BETTER) was outside the hypothesis space, so the run
    // printed the "between 0.2 and 1.0" text for d_ce = -10.04, which is simply wrong. The
    // numbers were right; the interpretation branch was missing. Recorded rather than quietly
    // patched, because failing to anticipate a sign is a real methodological miss.
    if d_ce < -1.0 {
        println!(
            "  ΔCE {d_ce:+.4} < -1.0 -> ACCUMULATED RECURRENT STATE IS ACTIVELY HARMFUL.\n\
             Wiping it every token makes prediction dramatically BETTER. The recurrent core is\n\
             not merely unused — as trained it degrades the output. Implementing BPTT would be\n\
             improving a pathway that currently subtracts value; the dynamics themselves\n\
             (drift/saturation over positions, with no temporal objective to constrain them)\n\
             have to be understood first."
        );
    } else if d_ce > 1.0 {
        println!(
            "  ΔCE {d_ce:+.4} > 1.0 -> RECURRENCE CARRIES REAL INFORMATION. Removing accumulated\n\
             history measurably hurts, so better temporal credit assignment (BPTT) is plausibly\n\
             worthwhile."
        );
    } else if d_ce.abs() < 0.2 {
        println!(
            "  |ΔCE| {:.4} < 0.2 -> RECURRENCE CONTRIBUTES ~NOTHING as trained. The core is acting\n\
             as a feed-forward map of (thought, prev_token, pos); BPTT would improve a pathway the\n\
             model does not currently use. Combined with the degenerate input encoding, that is a\n\
             SECOND independent structural problem.",
            d_ce.abs()
        );
    } else {
        println!(
            "  ΔCE {d_ce:+.4} is between 0.2 and 1.0 -> PARTIAL. Recurrence contributes something\n\
             but little; judge against the encoder ceiling before funding BPTT work."
        );
    }
    Ok(())
}
