// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic: WHY does Broca land in the wrong region of embedding space?
//!
//! Follow-up to `SYMTHAEA_BROCA_BASELINE_2026-07-29.md`, which established that post-vocab-fix
//! generation is non-functional (mean CE 21.9-26.0 nats vs uniform-random 8.32) while the
//! thought signal is *present but weak* (within-category Jaccard 0.235 vs between 0.164) and
//! the first token is near-constant (7 distinct across 70 cases, one token 48/70).
//!
//! That bounds the defect but does not locate it. This probe answers the two questions the
//! baseline doc names as decisive, both pure read-outs of the existing forward pass — no
//! training, no generation sampling:
//!
//! **Q1. Where does the target token actually rank in teacher-forced logits?**
//!   - target near the middle of the pack  => weak, diffuse, plausibly trainable model
//!   - target reliably at the very bottom  => inverted / mis-projected mapping (a mechanical
//!     defect, and a very different fix)
//!   Mean CE ~23 nats only bounds p(target) from above; rank distinguishes these.
//!
//! **Q2. Is the near-constant first token driven by the thought, or by the seeded state?**
//!   If `output_hv` at position 0 is nearly identical across *different* thoughts, then
//!   `seed_from_thought` is not transmitting thought content and the collapse is located
//!   there. Measured directly as pairwise cosine between position-0 outputs.
//!
//! Run:
//! ```bash
//! cargo run --release -p symthaea-broca --example broca_rank_probe -- \
//!   [checkpoint] [canonical.jsonl] [genesis-phrase]
//! ```

use symthaea_broca::evaluation::CanonicalEvalDataset;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Rank of `target` among logits, 0 = highest-scoring. Ties count as "better than target",
/// which is the conservative direction for a "is the target near the bottom?" question.
fn rank_of(logits: &[f32], target: usize) -> usize {
    if target >= logits.len() {
        return logits.len();
    }
    let t = logits[target];
    logits.iter().filter(|&&l| l > t).count()
}

fn cosine(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    let (x, y) = (a.as_slice(), b.as_slice());
    let n = x.len().min(y.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += x[i] * y[i];
        na += x[i] * x[i];
        nb += y[i] * y[i];
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-10);
    dot / d
}

fn pct(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let ckpt = args.next().unwrap_or_else(|| {
        "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin".to_string()
    });
    let canon = args.next().unwrap_or_else(|| {
        "crates/domains/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl".to_string()
    });
    let phrase = args
        .next()
        .unwrap_or_else(|| "broca-training-default".to_string());

    eprintln!("checkpoint: {ckpt}\ncanonical:  {canon}\ngenesis:    {phrase:?}");
    let genesis = GenesisSeed::from_phrase(&phrase);
    let (mut r#gen, ..) = BrocaGenerator::from_checkpoint(&ckpt, &genesis)
        .map_err(|e| anyhow::anyhow!("loading {ckpt}: {e}"))?;

    let vocab = r#gen.controller().vocab_size();
    let tokenizer = r#gen.tokenizer().clone();
    let dataset = CanonicalEvalDataset::from_jsonl(&canon)?.to_training_dataset(&tokenizer);
    println!("vocab_size={vocab}  cases={}", dataset.pairs.len());

    // ---- Q1: teacher-forced target rank -------------------------------------------------
    let mut ranks: Vec<usize> = Vec::new();
    let mut pos0_outputs: Vec<ContinuousHV> = Vec::new();
    let mut pos0_argmax: Vec<usize> = Vec::new();
    let mut thoughts: Vec<ContinuousHV> = Vec::new();

    for pair in &dataset.pairs {
        if pair.target_ids.is_empty() {
            continue;
        }
        let channels = pair.to_thought_channels();
        let thought_hv = r#gen.encoder().encode(&channels);

        r#gen.controller_mut().reset();
        r#gen.controller_mut().seed_from_thought(&thought_hv);

        let mut prev = tokenizer.thought_id;
        for (pos, &tid) in pair.target_ids.iter().take(24).enumerate() {
            let logits = r#gen.controller_mut().forward_step(&thought_hv, prev, pos);
            ranks.push(rank_of(&logits, tid as usize));
            if pos == 0 {
                pos0_outputs.push(r#gen.controller().output_hv());
                let am = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                pos0_argmax.push(am);
                thoughts.push(thought_hv.clone());
            }
            prev = tid;
        }
    }

    let mut sorted = ranks.clone();
    sorted.sort_unstable();
    let n = ranks.len() as f64;
    let mean = ranks.iter().sum::<usize>() as f64 / n;
    let top10 = ranks.iter().filter(|&&r| r < 10).count() as f64 / n;
    let top100 = ranks.iter().filter(|&&r| r < 100).count() as f64 / n;
    let bottom10 = ranks.iter().filter(|&&r| r >= vocab * 9 / 10).count() as f64 / n;

    println!("\n=== Q1: target rank in teacher-forced logits (0 = best of {vocab}) ===");
    println!("  tokens scored     : {}", ranks.len());
    println!(
        "  mean rank         : {mean:.1}   (random expectation ~{:.1})",
        vocab as f64 / 2.0
    );
    println!(
        "  median / p25 / p75: {} / {} / {}",
        pct(&sorted, 0.50),
        pct(&sorted, 0.25),
        pct(&sorted, 0.75)
    );
    println!(
        "  best / worst      : {} / {}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!(
        "  target in top-10  : {:.3}   (chance {:.3})",
        top10,
        10.0 / vocab as f64
    );
    println!(
        "  target in top-100 : {:.3}   (chance {:.3})",
        top100,
        100.0 / vocab as f64
    );
    println!(
        "  target in bottom  : {:.3}   (chance 0.100)  <- decile",
        bottom10
    );
    println!(
        "  VERDICT: {}",
        if bottom10 > 0.5 {
            "target reliably at the BOTTOM => inverted/mis-projected mapping"
        } else if mean > vocab as f64 * 0.35 && mean < vocab as f64 * 0.65 {
            "target near CHANCE => weak/diffuse model, not inverted"
        } else {
            "target better than chance but far from top => weak signal, right direction"
        }
    );

    // ---- Q2: is position-0 output thought-dependent? ------------------------------------
    println!("\n=== Q2: position-0 output vs thought ===");
    let m = pos0_outputs.len();
    let mut out_sims = Vec::new();
    let mut thought_sims = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            out_sims.push(cosine(&pos0_outputs[i], &pos0_outputs[j]));
            thought_sims.push(cosine(&thoughts[i], &thoughts[j]));
        }
    }
    let mo = out_sims.iter().sum::<f32>() / out_sims.len().max(1) as f32;
    let mt = thought_sims.iter().sum::<f32>() / thought_sims.len().max(1) as f32;
    let distinct_am = {
        let mut v = pos0_argmax.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    println!("  cases                              : {m}");
    println!("  mean pairwise cos(output@0, output@0): {mo:.4}");
    println!("  mean pairwise cos(thought,   thought): {mt:.4}");
    println!("  distinct argmax @ pos 0            : {distinct_am}/{m}");
    println!(
        "  VERDICT: {}",
        if mo > 0.95 && mt < 0.9 {
            "position-0 output is ~CONSTANT while thoughts differ => seed_from_thought is not transmitting thought"
        } else if mo > mt + 0.15 {
            "position-0 outputs are markedly more alike than their thoughts => strong prior swamping the seed"
        } else {
            "position-0 output tracks thought variation => collapse is NOT located at seeding"
        }
    );

    Ok(())
}
