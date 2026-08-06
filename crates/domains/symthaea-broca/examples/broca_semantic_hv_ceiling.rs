// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Step 4.1 of `SYMTHAEA_BROCA_ENCODER_PLAN_2026-07-30.md`.
//!
//! Is `semantic_hv` actually discriminative? The plan's cheapest option (turn on the existing
//! NSM semantic blending) rests entirely on this untested assumption. If `semantic_hv` inherits
//! the same bundling dilution that flattens `ThoughtChannels` to 0.8224 mean pairwise cosine,
//! then blending it in buys nothing and the plan should skip straight to weighting the bundle.
//!
//! `broca_curriculum_sync.rs::encode_semantic_hv` is:
//!     tokenize(text) -> ids -> bundle(token_embeddings[ids])
//! i.e. an unweighted bag-of-words superposition of *trained* token embeddings. Same bundle
//! family as the channel encoder, so the question is real — but the mixing set is 4,096-way
//! vocabulary rather than 43 fixed roles, and distinct texts differ in many tokens at once.
//!
//! **Decision rule, fixed before running** (from the plan, do not renegotiate after seeing it):
//!   - mean pairwise cos <= 0.30  -> genuinely discriminative; Option A viable, proceed to 4.2
//!   - mean pairwise cos ~= 0.80  -> same dilution; Option A is theatre, go to Option B
//!   - in between                 -> report the number and decide with it in hand
//!
//! Reference points measured earlier: unrelated HDC pair = 0.0064; canonical thought HVs =
//! 0.8224; `with_intent(0..8)` = 0.9675.
//!
//! The corpus here is the canonical suite's `target_text` values — 70 distinct real strings
//! across 8 categories. They are used only as *representative text* to characterise the
//! encoding method. This is NOT a proposal to derive `semantic_hv` from target text at eval
//! time (that would leak the answer); the real source would be a description of the thought,
//! as `broca_curriculum_sync` does with objective identity.
//!
//! ```bash
//! cargo run --release -p symthaea-broca --example broca_semantic_hv_ceiling -- [ckpt] [jsonl]
//! ```

use std::collections::HashMap;

use symthaea_broca::evaluation::CanonicalEvalDataset;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

fn cosine(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    let (x, y) = (a.as_slice(), b.as_slice());
    let n = x.len().min(y.len());
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        d += x[i] * y[i];
        na += x[i] * x[i];
        nb += y[i] * y[i];
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-10)
}

/// Byte-for-byte the same construction as `broca_curriculum_sync.rs::encode_semantic_hv`.
fn encode_semantic_hv(r#gen: &BrocaGenerator, text: &str) -> Option<ContinuousHV> {
    let ids = r#gen.tokenizer().encode(text);
    let all = r#gen.controller().token_embeddings();
    let embs: Vec<&ContinuousHV> = ids.iter().filter_map(|&id| all.get(id as usize)).collect();
    if embs.is_empty() {
        None
    } else {
        Some(ContinuousHV::bundle(&embs))
    }
}

fn summarize(label: &str, v: &[f32]) {
    if v.is_empty() {
        println!("{label:<38} (no pairs)");
        return;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    println!(
        "{label:<38} n={:<6} mean={mean:.4}  min={lo:.4}  max={hi:.4}",
        v.len()
    );
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
    let (r#gen, ..) = BrocaGenerator::from_checkpoint(&ckpt, &genesis)
        .map_err(|e| anyhow::anyhow!("loading {ckpt}: {e}"))?;

    let ds = CanonicalEvalDataset::from_jsonl(&canon)?;
    println!(
        "cases={} vocab={}",
        ds.cases.len(),
        r#gen.controller().vocab_size()
    );

    let mut hvs = Vec::new();
    let mut cats = Vec::new();
    let mut skipped = 0usize;
    for c in &ds.cases {
        match encode_semantic_hv(&r#gen, &c.target_text) {
            Some(h) => {
                hvs.push(h);
                cats.push(c.category.clone());
            }
            None => skipped += 1,
        }
    }
    println!("encoded={} skipped(empty)={}", hvs.len(), skipped);

    // Overall pairwise, plus within- vs between-category (does it track semantics at all?).
    let mut all = Vec::new();
    let mut within = Vec::new();
    let mut between = Vec::new();
    for i in 0..hvs.len() {
        for j in (i + 1)..hvs.len() {
            let c = cosine(&hvs[i], &hvs[j]);
            all.push(c);
            if cats[i] == cats[j] {
                within.push(c)
            } else {
                between.push(c)
            }
        }
    }

    println!("\n=== semantic_hv pairwise similarity ===");
    summarize("ALL pairs", &all);
    summarize("within-category", &within);
    summarize("between-category", &between);

    // Sanity control: identical text must encode identically (cos = 1).
    if let (Some(x), Some(y)) = (
        encode_semantic_hv(&r#gen, "the quick brown fox"),
        encode_semantic_hv(&r#gen, "the quick brown fox"),
    ) {
        println!(
            "\ncontrol: identical text cos = {:.6} (must be 1.000000)",
            cosine(&x, &y)
        );
    }

    let mean_all = all.iter().sum::<f32>() / all.len().max(1) as f32;
    let mw = within.iter().sum::<f32>() / within.len().max(1) as f32;
    let mb = between.iter().sum::<f32>() / between.len().max(1) as f32;

    println!("\n=== reference points ===");
    println!("  unrelated HDC pair            0.0064   <- what 'dissimilar' looks like");
    println!("  canonical THOUGHT HVs         0.8224   <- the ceiling we are trying to beat");
    println!("  with_intent(0..8)             0.9675");
    println!("  >> semantic_hv (this run)     {mean_all:.4}");
    println!("  semantic separation (w - b)   {:+.4}", mw - mb);

    println!("\n=== VERDICT (rule fixed before running) ===");
    if mean_all <= 0.30 {
        println!(
            "  {mean_all:.4} <= 0.30 -> DISCRIMINATIVE. Option A is viable; proceed to plan 4.2."
        );
    } else if mean_all >= 0.70 {
        println!(
            "  {mean_all:.4} >= 0.70 -> SAME DILUTION. Option A is theatre; go to Option B \
             (weighted bundle)."
        );
    } else {
        println!(
            "  {mean_all:.4} is between 0.30 and 0.70 -> partial. Report and decide with the \
             number in hand; compare against the 0.8224 thought-HV ceiling it must beat."
        );
    }

    // Category makeup, so a skewed corpus can't be mistaken for a property of the encoder.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for c in &cats {
        *counts.entry(c.as_str()).or_default() += 1;
    }
    let mut ks: Vec<_> = counts.into_iter().collect();
    ks.sort();
    println!("\ncategories: {ks:?}");
    Ok(())
}
