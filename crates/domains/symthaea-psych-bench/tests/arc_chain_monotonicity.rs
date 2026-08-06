//! Guards for `ArcChain`'s 2-AFC scoring.
//!
//! ## The defect these were written for (fixed 2026-07-31)
//!
//! On CI run 30496274683 the recorded metrics were:
//!
//! ```text
//! chain_2_accuracy: 0.9333
//! chain_3_accuracy: 0.0167   <-- 1/60, far BELOW 2-AFC chance of 0.50
//! chain_4_accuracy: 0.1500
//! ```
//!
//! Scoring is 2-AFC, so 0.50 is chance and 0.0167 is not "weak" — it is **systematic
//! anti-correlation**: the benchmark picked the distractor 59 times out of 60.
//!
//! Root cause was a **distance confound in the distractor**, not the prefix collision first
//! suspected. `fair_distractor_grid` returns a single-transform variation of the input, so its
//! distance from the input is constant, while the true output moves further from the input with
//! every added chain step. Measured: `apply_rule` sits ~0.78 similar to the input but only ~0.55
//! to a 3–4 step target. A near-identity prediction therefore matched the input-adjacent
//! distractor almost every time, and did so *worse the longer the chain*.
//!
//! The fix makes the distractor the output of a **different chain of the same length**, so both
//! options are equidistant from the input by construction. Effect, per chain:
//!
//! ```text
//! c0 len2  1.000 -> 1.000     c3 len3  0.000 -> 0.533
//! c1 len2  0.867 -> 1.000     c4 len4  0.267 -> 0.867
//! c2 len3  0.033 -> 0.700     c5 len4  0.033 -> 0.567
//! ```
//!
//! ## Why there is no length-monotonicity test here
//!
//! An earlier draft of this file asserted `chain_2 >= chain_3 >= chain_4`. That assertion was
//! **wrong to make**, and deleting it is deliberate rather than a concession to a red test.
//!
//! The chains are not matched on anything but length: the two weakest are exactly the two
//! containing `Rotate90` (c3 and c5), regardless of length, and the 4-step c4 (0.867) outscores
//! both 3-step chains. `get_chains` does not nest them either. So a non-monotone per-length
//! ordering is the *expected* result of this chain set, not a defect — and a test asserting
//! monotonicity would be permanently red for a reason no scoring fix could address.
//!
//! Measuring degradation-with-length properly needs nested chains (chain_3 = chain_2 + 1 step),
//! which would alter every recorded value. See the module docs on `arc_chain.rs`.
//!
//! ## Why the pre-existing test never caught any of this
//!
//! `test_degradation_with_length` in `arc_chain.rs` reads `chain_2_similarity` and
//! `chain_4_similarity`, then asserts only that both `.is_finite()`. It never compares them, so
//! it passes for any values including a total inversion — a probe that cannot fail for the
//! reason it exists. See `docs/VACUOUS_TEST_AUDIT_2026-07-31.md`; it was not an isolated case.

use symthaea_psych_bench::benchmarks::reasoning::arc_chain::ArcChainBenchmark;
use symthaea_psych_bench::harness::PsychBenchmark;
use symthaea_psych_bench::harness::config::BenchmarkConfig;

/// Matches the CI run's shape: 10 trials × 2 chains × 3 tasks = 60 observations per length,
/// which is what made 0.0167 read as exactly 1/60.
fn ci_shaped_config() -> BenchmarkConfig {
    BenchmarkConfig {
        trials_per_condition: 10,
        ..Default::default()
    }
}

/// The real invariant: 2-AFC scoring must never land a condition *below* chance.
///
/// Weak composition should score AT chance (~0.50), because a prediction carrying no information
/// about the specific rule is equally close to either option. Scoring reliably below chance means
/// the comparison is structurally biased toward the distractor — a defect in the benchmark, not a
/// capability result.
#[test]
fn no_condition_is_anti_correlated() {
    let result = ArcChainBenchmark.run(&ci_shaped_config());

    // A fair coin over 60 trials falls below 0.30 with probability ~1e-3, so this floor is not
    // flaky. It also sits well below the ~0.53 worst observed chain, leaving real headroom.
    const ANTI_CORRELATED_FLOOR: f64 = 0.30;

    for label in ["chain_2_accuracy", "chain_3_accuracy", "chain_4_accuracy"] {
        let acc = result.metrics[label].mean;
        assert!(
            acc >= ANTI_CORRELATED_FLOOR,
            "{label} = {acc:.4} is far below 2-AFC chance (0.50). The prediction is matching the \
             distractor more often than the true output, which means the two options are not \
             equidistant from the input — the exact defect fixed on 2026-07-31. Weak composition \
             should score AT chance, never below it."
        );
    }
}

/// Same invariant at per-chain resolution, which is where attribution actually happens.
///
/// The length aggregates hid *which* chain was failing; that is why the original investigation
/// could not be completed, and why these per-chain metrics now exist.
#[test]
fn no_individual_chain_is_anti_correlated() {
    let result = ArcChainBenchmark.run(&ci_shaped_config());

    for i in 0..6 {
        let key = format!("chain_c{i}_accuracy");
        let acc = result.metrics[key.as_str()].mean;
        assert!(
            acc >= 0.30,
            "chain c{i} = {acc:.4} is far below 2-AFC chance. Before the distractor fix c3 sat at \
             exactly 0.000 — a chain that never once beat its distractor across 60 trials."
        );
    }
}

/// Pins the mechanism, so a change that reintroduces an input-adjacent distractor fails loudly
/// here rather than silently producing below-chance scores again.
///
/// `fair_distractor_grid` is still used by six other ARC benchmarks (arc_strict, arc_noise,
/// arc_staircase, arc_scaling, arc_fewshot, arc_dataset) and was deliberately left unchanged
/// there. This documents what it does — and by extension that those benchmarks carry the same
/// confound wherever their target sits further from the input than one transform.
#[test]
fn generic_distractor_is_one_transform_from_the_input() {
    use symthaea_core::hdc::grid_encoder::GridEncoder;
    use symthaea_psych_bench::benchmarks::reasoning::arc_dataset::fair_distractor_grid;

    let input: Vec<Vec<u8>> = (0..5u8)
        .map(|r| (0..5u8).map(|c| (r * 5 + c) % 6).collect())
        .collect();
    let true_output = GridEncoder::rotate_90(&GridEncoder::reflect_y(&input));

    let distractor = fair_distractor_grid(&input, &true_output).expect("distractor exists");

    assert_eq!(
        distractor,
        GridEncoder::reflect_x(&input),
        "the generic distractor is reflect_x(input) — always exactly one transform from the \
         input, whatever the true output's distance from it. Any benchmark whose target is \
         several transforms away is therefore scored against a systematically input-adjacent \
         alternative."
    );
}
