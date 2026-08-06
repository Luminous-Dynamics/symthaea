// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic: is `ThoughtLanguageEncoder`'s ~0.82 similarity floor a property of the ENCODER,
//! or just of the canonical fixture set?
//!
//! `examples/broca_rank_probe.rs` found that 70 semantically distinct canonical cases encode to
//! hypervectors that are 82% mutually similar, and that the network faithfully propagates that
//! (output similarity 0.8264 vs input 0.8224). That bounds achievable discrimination — but it
//! was measured on one fixture set, which might simply be clustered.
//!
//! This separates the two. It needs no checkpoint and no training: it only encodes.
//!
//! - If deliberately extreme / antipodal channel vectors ALSO land at ~0.8, the floor is
//!   structural to the encoder, and no amount of training or better data fixes it — the
//!   representation itself has to change.
//! - If they separate cleanly (similarity near 0), the encoder is fine and the canonical
//!   fixtures are merely clustered in channel space — a data problem, much cheaper to fix.
//!
//! Run:
//! ```bash
//! cargo run --release -p symthaea-broca --example broca_encoder_ceiling
//! ```

use symthaea_broca::encoder::{NUM_CHANNELS, ThoughtChannels, ThoughtLanguageEncoder};
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

/// Deterministic LCG. Uses HIGH bits only — the low-order bits of an LCG are notoriously
/// weak (period-2^k in bit k), so `% n` on the raw state would produce structured, not
/// pseudo-random, channel values and could manufacture the very clustering we're testing for.
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn stats(hvs: &[ContinuousHV]) -> (f32, f32, f32) {
    let mut v = Vec::new();
    for i in 0..hvs.len() {
        for j in (i + 1)..hvs.len() {
            v.push(cosine(&hvs[i], &hvs[j]));
        }
    }
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in &v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (mean, lo, hi)
}

fn main() {
    let genesis = GenesisSeed::from_phrase("broca-training-default");
    let enc = ThoughtLanguageEncoder::new(&genesis);
    let enc1 = |c: &ThoughtChannels| enc.encode(c);

    println!("NUM_CHANNELS = {NUM_CHANNELS}");
    println!(
        "\n{:<34} {:>8} {:>9} {:>9} {:>9}",
        "condition", "n", "mean cos", "min cos", "max cos"
    );

    // 1. The 8 built-in intents.
    let intents: Vec<ContinuousHV> = (0..8)
        .map(|i| enc1(&ThoughtChannels::with_intent(i)))
        .collect();
    let (m, lo, hi) = stats(&intents);
    println!(
        "{:<34} {:>8} {:>9.4} {:>9.4} {:>9.4}",
        "with_intent(0..8)",
        intents.len(),
        m,
        lo,
        hi
    );

    // 2. Uniform-random channels in [0,1].
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let rand01: Vec<ContinuousHV> = (0..64)
        .map(|_| {
            let mut c = ThoughtChannels::default();
            for k in 0..NUM_CHANNELS {
                c.channels[k] = rng.next_unit();
            }
            enc1(&c)
        })
        .collect();
    let (m, lo, hi) = stats(&rand01);
    println!(
        "{:<34} {:>8} {:>9.4} {:>9.4} {:>9.4}",
        "random channels U[0,1]",
        rand01.len(),
        m,
        lo,
        hi
    );

    // 3. Wide-range random, well outside nominal channel ranges (tests clamping behaviour).
    let mut rng = Lcg(0x5EED_1234_ABCD_0002);
    let rand_wide: Vec<ContinuousHV> = (0..64)
        .map(|_| {
            let mut c = ThoughtChannels::default();
            for k in 0..NUM_CHANNELS {
                c.channels[k] = rng.next_unit() * 20.0 - 10.0;
            }
            enc1(&c)
        })
        .collect();
    let (m, lo, hi) = stats(&rand_wide);
    println!(
        "{:<34} {:>8} {:>9.4} {:>9.4} {:>9.4}",
        "random channels U[-10,10]",
        rand_wide.len(),
        m,
        lo,
        hi
    );

    // 4. The two most extreme inputs the type admits: all-min vs all-max.
    let mut lo_c = ThoughtChannels::default();
    let mut hi_c = ThoughtChannels::default();
    for k in 0..NUM_CHANNELS {
        lo_c.channels[k] = -1000.0;
        hi_c.channels[k] = 1000.0;
    }
    let extremes = vec![enc1(&lo_c), enc1(&hi_c)];
    let (m, _, _) = stats(&extremes);
    println!(
        "{:<34} {:>8} {:>9.4} {:>9} {:>9}",
        "all-min vs all-max (antipodal)", 2, m, "-", "-"
    );

    // 5. One-hot: each channel maxed alone. If the encoder is discriminative at all, these
    //    should be the most separable inputs it can receive.
    let onehot: Vec<ContinuousHV> = (0..NUM_CHANNELS)
        .map(|k| {
            let mut c = ThoughtChannels::default();
            for j in 0..NUM_CHANNELS {
                c.channels[j] = 0.0;
            }
            c.channels[k] = 1000.0;
            enc1(&c)
        })
        .collect();
    let (m, lo, hi) = stats(&onehot);
    println!(
        "{:<34} {:>8} {:>9.4} {:>9.4} {:>9.4}",
        "one-hot per channel",
        onehot.len(),
        m,
        lo,
        hi
    );

    // 6. Reference: two genuinely unrelated HDC vectors. This is what "dissimilar" looks like
    //    in this space, and calibrates every number above.
    let r1 = ContinuousHV::from_genesis(&genesis, "unrelated-reference-a", 16384);
    let r2 = ContinuousHV::from_genesis(&genesis, "unrelated-reference-b", 16384);
    println!(
        "{:<34} {:>8} {:>9.4} {:>9} {:>9}",
        "REFERENCE: unrelated HDC pair",
        2,
        cosine(&r1, &r2),
        "-",
        "-"
    );

    println!(
        "\nInterpretation: the canonical suite measured 0.8224 mean pairwise similarity.\n\
         If the extreme conditions above are also ~0.8, the floor is STRUCTURAL to the encoder\n\
         (a representation problem). If they approach the unrelated-pair reference, the encoder\n\
         discriminates fine and the canonical fixtures are simply clustered (a data problem)."
    );
}
