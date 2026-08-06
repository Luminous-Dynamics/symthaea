// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Distributional characterization of `ContinuousHV`/`BinaryHV` binding
//! algebra. Phase 1b/1c ("Commit B") of the HDC Binding Algebra
//! Qualification and Migration Plan (2026-07-27).
//!
//! Deliberately an **example**, not a `#[test]` suite: this measures and
//! reports distributions across seeds/dimensions rather than asserting
//! pass/fail on an arbitrary sampled threshold (that pattern is exactly what
//! created the fragile/misleading claims this whole audit exists to correct
//! — see `binding_algebra_audit.rs`'s module doc). The hard, deterministic
//! contract tests live there instead.
//!
//! Run with: `cargo run --release --example binding_algebra_characterization -p symthaea-core`
//!
//! Output is printed to stdout AND written to
//! `docs/BINDING_ALGEBRA_CHARACTERIZATION_REPORT.md` for permanent record.

use std::fmt::Write as _;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::unified_hv::ContinuousHV;

const SEEDS: u64 = 40;
const DIMS: &[usize] = &[64, 256, 1024, 16_384];

#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    n: usize,
}

fn stats(samples: &[f64]) -> Stats {
    let n = samples.len();
    if n == 0 {
        return Stats::default();
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Stats {
        mean,
        std_dev,
        min,
        max,
        n,
    }
}

fn fmt_stats(s: &Stats) -> String {
    format!(
        "mean={:.4} std={:.4} min={:.4} max={:.4} (n={})",
        s.mean, s.std_dev, s.min, s.max, s.n
    )
}

/// Section 1: recovery similarity, inverse-based vs. double-bind, for
/// ContinuousHV; exact recovery for BinaryHV as the known-good comparison.
fn section_recovery_similarity(out: &mut String) {
    writeln!(
        out,
        "\n## 1. Recovery similarity: inverse-based vs. double-bind\n"
    )
    .unwrap();
    writeln!(
        out,
        "For each dimension, `n={SEEDS}` random (A,B) pairs. \
         `inverse-based` = sim((A⊗B)⊗B⁻¹, A). `double-bind` = sim((A⊗B)⊗B, A) \
         (the pattern `symthaea-psych-bench`'s P2/P4a actually use)."
    )
    .unwrap();
    writeln!(out, "\n| dim | inverse-based | double-bind |").unwrap();
    writeln!(out, "|---|---|---|").unwrap();
    for &dim in DIMS {
        let mut inverse_sims = Vec::with_capacity(SEEDS as usize);
        let mut double_sims = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            let a = ContinuousHV::random(dim, 100_000 + seed);
            let b = ContinuousHV::random(dim, 200_000 + seed);
            let bound = a.bind(&b);
            inverse_sims.push(bound.bind(&b.inverse()).similarity(&a) as f64);
            double_sims.push(bound.bind(&b).similarity(&a) as f64);
        }
        writeln!(
            out,
            "| {dim} | {} | {} |",
            fmt_stats(&stats(&inverse_sims)),
            fmt_stats(&stats(&double_sims))
        )
        .unwrap();
    }

    // BinaryHV comparison (fixed dim = 16,384).
    let mut binary_recovery = Vec::with_capacity(SEEDS as usize);
    for seed in 0..SEEDS {
        let a = BinaryHV::random(300_000 + seed);
        let b = BinaryHV::random(400_000 + seed);
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        binary_recovery.push(recovered.cosine_similarity(&a) as f64);
    }
    writeln!(
        out,
        "\nBinaryHV (dim=16384, XOR double-bind, exact by construction): {}",
        fmt_stats(&stats(&binary_recovery))
    )
    .unwrap();
}

/// Section 2: self-bind distance from a true identity (all-ones) vector.
fn section_self_bind_vs_identity(out: &mut String) {
    writeln!(out, "\n## 2. Self-bind (A⊗A) similarity to true identity\n").unwrap();
    writeln!(
        out,
        "| dim | ContinuousHV sim(A⊗A, ones) | BinaryHV cosine_sim(A⊗A, all-true-bits) |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|").unwrap();
    for &dim in DIMS {
        let identity = ContinuousHV::from_values(vec![1.0_f32; dim]);
        let mut sims = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            let a = ContinuousHV::random(dim, 500_000 + seed);
            sims.push(a.bind(&a).similarity(&identity) as f64);
        }
        writeln!(
            out,
            "| {dim} | {} | (see BinaryHV row below) |",
            fmt_stats(&stats(&sims))
        )
        .unwrap();
    }
    // BinaryHV: A XOR A = all-zero-bits, which under the bit-to-sign mapping
    // used by cosine_similarity is the all-"agree" identity for XOR's group
    // structure -- reported separately since BinaryHV has no meaningful
    // "all-ones-in-continuous-sense" analog.
    let mut binary_self_bind = Vec::with_capacity(SEEDS as usize);
    // BinaryHV's tuple field is public: `pub struct BinaryHV(pub [u8; 2048])`.
    let all_zero_bits = BinaryHV([0u8; 2048]);
    for seed in 0..SEEDS {
        let a = BinaryHV::random(600_000 + seed);
        let self_bound = a.bind(&a);
        // A⊕A should be the all-zero-bits vector, exactly, every time.
        binary_self_bind.push(self_bound.hamming_distance(&all_zero_bits) as f64);
    }
    writeln!(
        out,
        "\nBinaryHV: hamming_distance(A⊕A, all-zero-bits) across {SEEDS} seeds: {}",
        fmt_stats(&stats(&binary_self_bind))
    )
    .unwrap();
    writeln!(out, "(Expected: exactly 0 for every seed -- A⊕A is always the all-zero-bits vector by construction.)").unwrap();
}

/// Section 3: shared-carrier similarity distortion.
fn section_shared_carrier_distortion(out: &mut String) {
    writeln!(
        out,
        "\n## 3. Shared-carrier similarity distortion: sim(A,B) vs sim(A⊗C, B⊗C)\n"
    )
    .unwrap();
    writeln!(out, "Trivial case: A,B independent (expect both near 0). Non-trivial case: B = A + small correlated perturbation (expect sim(A,B) substantially > 0; measures whether binding with C preserves that correlation or destroys it).").unwrap();
    writeln!(out, "\n| dim | raw sim(A,B) [independent] | bound sim(A⊗C,B⊗C) [independent] | raw sim(A,B) [correlated] | bound sim(A⊗C,B⊗C) [correlated] |").unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();
    for &dim in DIMS {
        let mut raw_indep = Vec::with_capacity(SEEDS as usize);
        let mut bound_indep = Vec::with_capacity(SEEDS as usize);
        let mut raw_corr = Vec::with_capacity(SEEDS as usize);
        let mut bound_corr = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            let a = ContinuousHV::random(dim, 700_000 + seed);
            let b_indep = ContinuousHV::random(dim, 800_000 + seed);
            let c = ContinuousHV::random(dim, 900_000 + seed);
            raw_indep.push(a.similarity(&b_indep) as f64);
            bound_indep.push(a.bind(&c).similarity(&b_indep.bind(&c)) as f64);

            // Correlated case: b_corr = normalize(0.8*a + 0.2*noise).
            let noise = ContinuousHV::random(dim, 1_000_000 + seed);
            let b_corr = ContinuousHV::weighted_bundle(&[&a, &noise], &[0.8, 0.2]);
            raw_corr.push(a.similarity(&b_corr) as f64);
            bound_corr.push(a.bind(&c).similarity(&b_corr.bind(&c)) as f64);
        }
        writeln!(
            out,
            "| {dim} | {} | {} | {} | {} |",
            fmt_stats(&stats(&raw_indep)),
            fmt_stats(&stats(&bound_indep)),
            fmt_stats(&stats(&raw_corr)),
            fmt_stats(&stats(&bound_corr)),
        )
        .unwrap();
    }
}

/// Section 4: repeated-composition parity pattern (extends
/// `hdc_binding_properties.rs`'s Finding 4 into the core crate).
fn section_repeated_composition_parity(out: &mut String) {
    writeln!(
        out,
        "\n## 4. Repeated self-composition (A^k) similarity to A, by parity of k+1\n"
    )
    .unwrap();
    writeln!(out, "sim(A^k, A)'s numerator is Σ A_i^(k+1). Even k+1 -> every term ≥0 (positive bias expected). Odd k+1 -> symmetric zero-mean (no bias expected). See `hdc_binding_properties.rs` for the original single-dim derivation this generalizes.").unwrap();
    writeln!(
        out,
        "\n| dim | k=2 (k+1=3,odd) | k=3 (k+1=4,even) | k=4 (k+1=5,odd) | k=5 (k+1=6,even) |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();
    for &dim in DIMS {
        let mut by_k: Vec<Vec<f64>> = vec![Vec::with_capacity(SEEDS as usize); 4];
        for seed in 0..SEEDS {
            let a = ContinuousHV::random(dim, 1_100_000 + seed);
            let mut current = a.clone();
            for slot in by_k.iter_mut() {
                current = current.bind(&a);
                slot.push(current.similarity(&a) as f64);
            }
        }
        writeln!(
            out,
            "| {dim} | {} | {} | {} | {} |",
            fmt_stats(&stats(&by_k[0])),
            fmt_stats(&stats(&by_k[1])),
            fmt_stats(&stats(&by_k[2])),
            fmt_stats(&stats(&by_k[3])),
        )
        .unwrap();
    }
}

/// Section 5: norm distributions under weighted-bundle accumulation
/// (relevant to P2/P4a's EMA-style memory accumulators).
fn section_norm_accumulation(out: &mut String) {
    writeln!(
        out,
        "\n## 5. Norm growth under repeated weighted_bundle accumulation\n"
    )
    .unwrap();
    writeln!(out, "Simulates P2/P4a's accumulator pattern: `memory = weighted_bundle([memory, bind(X,Y)], [1-w, w])` repeated 40 times with a FIXED target (same X,Y every step) at w=0.1.").unwrap();
    writeln!(
        out,
        "\n| dim | norm after 1 step | norm after 10 steps | norm after 40 steps |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|").unwrap();
    for &dim in DIMS {
        let mut after_1 = Vec::with_capacity(SEEDS as usize);
        let mut after_10 = Vec::with_capacity(SEEDS as usize);
        let mut after_40 = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            let x = ContinuousHV::random(dim, 1_200_000 + seed);
            let y = ContinuousHV::random(dim, 1_300_000 + seed);
            let target = x.bind(&y);
            let mut memory = ContinuousHV::zero(dim);
            for step in 1..=40 {
                memory = ContinuousHV::weighted_bundle(&[&memory, &target], &[0.9, 0.1]);
                if step == 1 {
                    after_1.push(memory.norm() as f64);
                } else if step == 10 {
                    after_10.push(memory.norm() as f64);
                } else if step == 40 {
                    after_40.push(memory.norm() as f64);
                }
            }
        }
        writeln!(
            out,
            "| {dim} | {} | {} | {} |",
            fmt_stats(&stats(&after_1)),
            fmt_stats(&stats(&after_10)),
            fmt_stats(&stats(&after_40)),
        )
        .unwrap();
    }
}

/// Section 6 (Phase 1c): inverse() numerical stability.
fn section_inverse_numerical_stability(out: &mut String) {
    writeln!(out, "\n## 6. `inverse()` numerical stability\n").unwrap();
    writeln!(
        out,
        "Measured, not fixed, per the plan -- no clipping/regularization introduced here."
    )
    .unwrap();
    writeln!(
        out,
        "\n| dim | min \\|component\\| | max \\|inverse\\| | fraction non-finite | recon error (pre-normalize) | recon error (post-normalize) |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|---|---|").unwrap();
    for &dim in DIMS {
        let mut min_abs_components = Vec::with_capacity(SEEDS as usize);
        let mut max_inverse_mags = Vec::with_capacity(SEEDS as usize);
        let mut non_finite_fractions = Vec::with_capacity(SEEDS as usize);
        let mut recon_err_pre = Vec::with_capacity(SEEDS as usize);
        let mut recon_err_post = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            let a = ContinuousHV::random(dim, 1_400_000 + seed);
            let min_abs = a
                .values
                .iter()
                .map(|v| v.abs())
                .fold(f32::INFINITY, f32::min);
            min_abs_components.push(min_abs as f64);

            let inv = a.inverse();
            let max_inv_mag = inv.values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            max_inverse_mags.push(max_inv_mag as f64);

            let non_finite = inv.values.iter().filter(|v| !v.is_finite()).count();
            non_finite_fractions.push(non_finite as f64 / dim as f64);

            // Reconstruction: A ⊗ A⁻¹ should approximate the identity
            // (all-ones) vector where A's components are above epsilon.
            let identity = ContinuousHV::from_values(vec![1.0_f32; dim]);
            let recon = a.bind(&inv);
            let err_pre: f64 = recon
                .values
                .iter()
                .zip(identity.values.iter())
                .map(|(r, i)| ((r - i) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            recon_err_pre.push(err_pre);

            let recon_normalized = recon.normalize();
            let identity_normalized = identity.normalize();
            let err_post: f64 = recon_normalized
                .values
                .iter()
                .zip(identity_normalized.values.iter())
                .map(|(r, i)| ((r - i) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            recon_err_post.push(err_post);
        }
        writeln!(
            out,
            "| {dim} | {} | {} | {} | {} | {} |",
            fmt_stats(&stats(&min_abs_components)),
            fmt_stats(&stats(&max_inverse_mags)),
            fmt_stats(&stats(&non_finite_fractions)),
            fmt_stats(&stats(&recon_err_pre)),
            fmt_stats(&stats(&recon_err_post)),
        )
        .unwrap();
    }
}

/// Qualitative synthesis of the sections below, written against the actual
/// first real run's numbers (2026-07-27). Re-verify this prose against a
/// fresh run's tables if `ContinuousHV`'s implementation ever changes --
/// this section summarizes, it does not re-derive.
fn section_summary(out: &mut String) {
    writeln!(out, "\n## Summary — claimed vs. measured\n").unwrap();
    writeln!(
        out,
        "| Property | `ContinuousHV` doc claims | Actually measured | Required conditions |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|").unwrap();
    writeln!(out, "| Self-inverse (A⊗A ≈ identity) | \"A⊗A ≈ 1\" | **False.** sim(A⊗A, ones) ≈ 0.745, stable across all tested dims (64–16384) — a real, dimension-*invariant* distributional constant (E[A_i²] for uniform[-1,1]), not approaching 1.0. True self-inverse (BinaryHV: hamming(A⊕A, zero)=0 exactly) requires bipolar/binary components, confirmed via §2. | Bipolar (±1) components, not uniform-continuous. |").unwrap();
    writeln!(out, "| Preserves similarity (sim(A⊗C,B⊗C)=sim(A,B)) | \"sim(A⊗C, B⊗C) = sim(A, B)\" | **Approximately holds** in the regimes tested — independent-vector case both ≈0 (§3 cols 1–2), and the high-correlation case (sim(A,B)≈0.97) is preserved almost exactly after binding (§3 cols 3–4, Δ<0.01 across all dims). Not falsified here; the failure is specifically the self-inverse claim above, not this one. | Tested only for one correlation level (~0.97) and one carrier per pair — not swept across the full correlation range. |").unwrap();
    writeln!(out, "| Inverse-based recovery works | (implied by `inverse()`'s own doc: \"should yield a vector near the identity\") | **True and strong** — §1's inverse-based recovery (mean sim ≈0.92, tight across dims) is far better than the double-bind pattern P2/P4a actually use (mean sim ≈0.82, and it's a `A⊗B²`-style artifact, not textbook unbinding). | None — holds robustly across all tested dims. |").unwrap();
    writeln!(out, "| `inverse()` numerically stable | (undocumented — no stated bound) | **False as dimension grows.** Mean max\\|inverse\\| grows from ~356 (dim=64) to ~50,246 (dim=16384), with an observed extreme of 419,430 — WORSE at higher dimensions (more samples → higher chance of hitting a near-zero component), the opposite of the naive \"more dims = more stable\" intuition. Never produces non-finite output (epsilon floor works, confirmed by the hard contract test) and aggregate reconstruction error stays negligible at the whole-vector RMS level (§6) — but any code reading `inverse()`'s output component-wise, not just through a full re-bind+compare, should treat individual huge-magnitude components as expected, not a bug. | None identified yet that bounds the magnitude — this is the single most actionable finding for Commit D's Tier A review (which consumers use `inverse()` in a way sensitive to this?). |").unwrap();
    writeln!(out, "| Repeated composition parity | (undocumented) | **Confirmed, precisely, across all dims** (§4): odd total-degree (k+1) → near-zero mean (symmetric, unbiased); even total-degree → strong positive bias (~0.82–0.92). Matches `hdc_binding_properties.rs`'s single-dim derivation exactly, now generalized. | None — structural property of any odd-power-vs-even-power moment of a symmetric distribution. |").unwrap();
    writeln!(out, "\nSee sections 1–6 below for the full per-dimension tables these conclusions are drawn from.\n").unwrap();
}

fn main() {
    let mut report = String::new();
    writeln!(
        report,
        "# HDC Binding Algebra Characterization Report\n\n\
         Generated by `cargo run --release --example binding_algebra_characterization -p symthaea-core`.\n\n\
         Phase 1b/1c (\"Commit B\") of the HDC Binding Algebra Qualification and Migration Plan \
         (2026-07-27). This is a **distributional characterization artifact**, not a test suite \
         with asserted thresholds -- see `src/hdc/binding_algebra_audit.rs` for the hard, \
         deterministic contract tests, and the module doc there for why this split exists.\n\n\
         `n={SEEDS}` seeds per dimension unless noted; dimensions tested: {DIMS:?}."
    )
    .unwrap();

    section_summary(&mut report);
    section_recovery_similarity(&mut report);
    section_self_bind_vs_identity(&mut report);
    section_shared_carrier_distortion(&mut report);
    section_repeated_composition_parity(&mut report);
    section_norm_accumulation(&mut report);
    section_inverse_numerical_stability(&mut report);

    println!("{report}");

    let out_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/BINDING_ALGEBRA_CHARACTERIZATION_REPORT.md"
    );
    std::fs::write(out_path, &report).expect("failed to write characterization report");
    eprintln!("\nReport written to {out_path}");
}
