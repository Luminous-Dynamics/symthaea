// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gate 1 of `SYMTHAEA_BROCA_ENCODER_PLAN_2026-07-30.md` Option B: does weighting the bundle
//! actually buy discriminability, and how much is reachable *at all*?
//!
//! Encoder only — no checkpoint, no training.
//!
//! # The prediction being tested
//!
//! Channels 0..7 are an **8-wide one-hot intent block**, so two different intents differ in
//! exactly **2 channels of 43** (one 0→1, one 1→0). With the intent block at weight `W` and
//! the other 35 channels at 1.0, the differing energy is `2W²` out of `8W² + 35`, so:
//!
//! ```text
//!     similarity ≈ 1 − 2W² / (8W² + 35)
//! ```
//!
//! which **asymptotes at 1 − 2/8 = 0.75** as `W → ∞`, because 6 of the 8 block channels stay
//! identical (both zero) no matter how heavily the block is weighted.
//!
//! If that holds, it is the headline result: **weighting alone cannot make intents
//! well-separated.** The one-hot layout caps it, so the real fix is representational (one
//! intent-identity vector bound to an intent role, rather than 8 mostly-off scalars) — a
//! different and larger change than re-weighting.
//!
//! Predicted, printed alongside measured so the model can be falsified rather than confirmed.
//!
//! ```bash
//! cargo run --release -p symthaea-broca --example broca_weight_sweep
//! ```

use symthaea_broca::encoder::{NUM_CHANNELS, ThoughtChannels, ThoughtLanguageEncoder};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const INTENT_BLOCK: std::ops::Range<usize> = 0..8;

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

fn mean_pairwise(hvs: &[ContinuousHV]) -> f32 {
    let mut v = Vec::new();
    for i in 0..hvs.len() {
        for j in (i + 1)..hvs.len() {
            v.push(cosine(&hvs[i], &hvs[j]));
        }
    }
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

fn main() {
    let genesis = GenesisSeed::from_phrase("broca-training-default");

    println!("NUM_CHANNELS={NUM_CHANNELS}  intent block = channels {INTENT_BLOCK:?} (one-hot)");
    println!(
        "\nPredicted intent similarity with intent-block weight W:  1 - 2W^2/(8W^2 + 35)\n\
         Asymptote as W->inf: {:.4}  <- the ceiling the one-hot layout imposes\n",
        1.0 - 2.0 / 8.0
    );

    println!(
        "{:>8}  {:>12}  {:>12}  {:>10}",
        "W", "intent(meas)", "intent(pred)", "delta"
    );

    for &w in &[1.0f32, 2.0, 3.0, 5.0, 10.0, 50.0, 500.0] {
        let mut enc = ThoughtLanguageEncoder::new(&genesis);
        if w != 1.0 {
            let mut weights = vec![1.0f32; NUM_CHANNELS];
            for c in INTENT_BLOCK {
                weights[c] = w;
            }
            assert!(enc.set_channel_weights(&weights), "weights rejected");
        }
        let intents: Vec<ContinuousHV> = (0..8)
            .map(|i| enc.encode(&ThoughtChannels::with_intent(i)))
            .collect();
        let meas = mean_pairwise(&intents);
        let pred = 1.0 - (2.0 * w * w) / (8.0 * w * w + 35.0);
        println!(
            "{w:>8.1}  {meas:>12.4}  {pred:>12.4}  {:>10.4}",
            meas - pred
        );
    }

    // Does weighting the intent block hurt everything else? Sweep a broader input set so a
    // gain on intent that costs general discriminability is visible rather than hidden.
    println!("\n--- effect on general (random-channel) discriminability ---");
    println!("{:>8}  {:>18}", "W", "random-chan mean");
    let mut seed: u64 = 0xC0FFEE_1234_5678;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let samples: Vec<[f32; NUM_CHANNELS]> = (0..48)
        .map(|_| {
            let mut c = [0.0f32; NUM_CHANNELS];
            for v in c.iter_mut() {
                *v = next();
            }
            c
        })
        .collect();
    for &w in &[1.0f32, 3.0, 10.0] {
        let mut enc = ThoughtLanguageEncoder::new(&genesis);
        if w != 1.0 {
            let mut weights = vec![1.0f32; NUM_CHANNELS];
            for c in INTENT_BLOCK {
                weights[c] = w;
            }
            enc.set_channel_weights(&weights);
        }
        let hvs: Vec<ContinuousHV> = samples
            .iter()
            .map(|s| {
                let mut ch = ThoughtChannels::default();
                ch.channels.copy_from_slice(s);
                enc.encode(&ch)
            })
            .collect();
        println!("{w:>8.1}  {:>18.4}", mean_pairwise(&hvs));
    }

    println!(
        "\nReference: unrelated HDC pair 0.0064 | canonical thought HVs 0.8224 | \
         with_intent uniform 0.9675"
    );
    println!(
        "\nRead: if measured tracks predicted and flattens near 0.75, weighting CANNOT fix\n\
         intent separation -- the 8-wide one-hot layout caps it, and the fix is representational\n\
         (bind one intent-identity vector to an intent role) rather than a re-weighting."
    );
}
