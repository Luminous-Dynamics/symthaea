// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta RNG bias check — direct follow-up to §15's disclosed, untested candidate cause
//! for the still-unexplained residual in `shuffled_collapses`: does the shuffled-context control's
//! xorshift64(13,7,17) sequence, read via `x % 2` (the LOWEST bit), have any real correlation with
//! the deterministic outcome schedule (`t % 2`) for the exact seed this research arc has used
//! throughout (`0x9E3779B97F4A7C15 ^ 1`)? This project has been burned by exactly this class of
//! bug before (see `feedback_lcg_low_order_bits_pitfall.md`) -- low-order bits of simple PRNGs are
//! a documented weak spot, and this checks it directly rather than assuming either way.
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_rng_bias_check --release`

const TICKS: u64 = 2000;
const SEED: u64 = 0x9E3779B97F4A7C15u64 ^ 1;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn main() {
    println!("MA-001R-delta RNG bias check -- seed={SEED:#x} ticks={TICKS}\n");

    let mut rng_state = SEED;
    // Four combinations: (random_context, true_outcome_schedule) where random_context is A iff
    // x%2==0 (matching run_shuffled_with_delta_rule_from_raw_observation's own exact logic) and
    // true_outcome_schedule is "would-be-A" iff t%2==0 (matching outcome_for_tick's own logic).
    let mut a_a = 0u64; // random says A, true schedule says A-tick
    let mut a_b = 0u64; // random says A, true schedule says B-tick
    let mut b_a = 0u64;
    let mut b_b = 0u64;

    let mut low_bit_sequence: Vec<u8> = Vec::with_capacity(TICKS as usize);

    for t in 0..TICKS {
        let x = xorshift(&mut rng_state);
        let random_is_a = x.is_multiple_of(2);
        let true_is_a_tick = t % 2 == 0;
        low_bit_sequence.push(if random_is_a { 1 } else { 0 });
        match (random_is_a, true_is_a_tick) {
            (true, true) => a_a += 1,
            (true, false) => a_b += 1,
            (false, true) => b_a += 1,
            (false, false) => b_b += 1,
        }
    }

    println!("=== Joint distribution (random_context x true_tick_parity) over {TICKS} ticks ===");
    println!(
        "  random=A, true=A-tick: {a_a} ({:.2}%)",
        100.0 * a_a as f64 / TICKS as f64
    );
    println!(
        "  random=A, true=B-tick: {a_b} ({:.2}%)",
        100.0 * a_b as f64 / TICKS as f64
    );
    println!(
        "  random=B, true=A-tick: {b_a} ({:.2}%)",
        100.0 * b_a as f64 / TICKS as f64
    );
    println!(
        "  random=B, true=B-tick: {b_b} ({:.2}%)",
        100.0 * b_b as f64 / TICKS as f64
    );
    println!("  (expect ~25% each under a fair, uncorrelated coin)\n");

    let random_a_total = a_a + a_b;
    let random_b_total = b_a + b_b;
    println!(
        "Marginal: random=A total = {random_a_total} ({:.2}%), random=B total = {random_b_total} ({:.2}%)",
        100.0 * random_a_total as f64 / TICKS as f64,
        100.0 * random_b_total as f64 / TICKS as f64
    );

    // Pearson correlation between the two binary sequences (random_is_a as 0/1, true_is_a_tick as 0/1).
    let n = TICKS as f64;
    let mean_random = random_a_total as f64 / n;
    let mean_true = (TICKS / 2) as f64 / n; // exactly 0.5, t%2==0 for exactly half of 0..2000
    let mut cov = 0.0;
    let mut var_random = 0.0;
    let mut var_true = 0.0;
    let mut rng_state2 = SEED;
    for t in 0..TICKS {
        let x = xorshift(&mut rng_state2);
        let r = if x.is_multiple_of(2) { 1.0 } else { 0.0 };
        let tr = if t % 2 == 0 { 1.0 } else { 0.0 };
        cov += (r - mean_random) * (tr - mean_true);
        var_random += (r - mean_random).powi(2);
        var_true += (tr - mean_true).powi(2);
    }
    let correlation = cov / (var_random.sqrt() * var_true.sqrt());
    println!("\nPearson correlation(random_context_is_A, true_tick_is_A): {correlation:.6}");

    // Longest run of identical low-bit values -- a crude "is this sequence suspiciously patterned"
    // check. A fair coin's expected longest run over 2000 flips is roughly log2(2000) ~ 11.
    let mut longest_run = 1u32;
    let mut current_run = 1u32;
    for i in 1..low_bit_sequence.len() {
        if low_bit_sequence[i] == low_bit_sequence[i - 1] {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 1;
        }
    }
    println!(
        "Longest run of identical low-bit values: {longest_run} (expected ~11 for a fair coin over {TICKS} flips)"
    );

    // Check period-2 self-correlation directly: does low_bit at tick t correlate with tick t-2,
    // t-4 etc (would indicate the xorshift state itself falls into a short cycle under %2)?
    let mut same_as_2_back = 0u64;
    for i in 2..low_bit_sequence.len() {
        if low_bit_sequence[i] == low_bit_sequence[i - 2] {
            same_as_2_back += 1;
        }
    }
    println!(
        "Fraction where low-bit(t) == low-bit(t-2): {:.4} (expect ~0.50 for no period-2 structure)",
        same_as_2_back as f64 / (low_bit_sequence.len() - 2) as f64
    );

    println!(
        "\nVERDICT: {}",
        if correlation.abs() > 0.05 {
            "REAL BIAS DETECTED -- the low-order bit of this xorshift sequence shows a \
            non-trivial correlation with tick parity for this specific seed. This would mean \
            part of shuffled_collapses's unexplained residual is a genuine RNG-quality artifact \
            (matching this project's own documented low-order-bit pitfall), not a deep property \
            of randomized-vs-fixed-schedule context assignment. Fix: draw from a higher bit (e.g. \
            (x >> 32) % 2 or (x >> 63)) rather than x % 2, matching this project's own standing \
            fix pattern for exactly this class of bug."
        } else {
            "NO MEANINGFUL BIAS -- correlation is near zero and the run-length/period-2 checks \
            look like a fair coin. This specific seed's random context assignment is NOT \
            correlated with the true outcome schedule, so it does not explain shuffled's residual \
            movement beyond the balanced-decorrelated control. The residual must come from \
            something else -- most plausibly, a single seed's necessarily-imperfect empirical \
            balance is itself enough (with only 2000 samples, even a genuinely unbiased random \
            assignment will show SOME nonzero sample correlation just by chance, and the delta \
            rule's own accumulation over many steps could amplify a small per-tick imbalance into \
            a visible aggregate effect) -- this would need averaging over multiple independent \
            seeds to distinguish from a real, seed-independent mechanism, not tested here."
        }
    );
}
