// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end verification of the V2 benchmark pipeline.
//!
//! Every piece of this benchmark was unit-tested in isolation — generators,
//! validator, scorer — and **nothing had ever been run through all three
//! together**. That gap is how the missing timed-scoring path stayed invisible:
//! `score` could not accept a [`TimedItem`] corpus, so the entire §5.2 family
//! was unscoreable, and no component test could have noticed because no
//! component was wrong.
//!
//! These tests run generator → validator → scorer on real generated corpora and
//! assert the properties the pre-registration
//! (`docs/TEMPORAL_BENCHMARK_V2_PREREGISTRATION_2026-07-31.md`) declares as
//! hard gates. In particular **gate 3, "negative control is mechanical"**: a
//! zero-history arm must sit at chance on ambiguous points *by construction*.
//! Asserting that on a toy corpus proves nothing; asserting it on the corpora
//! the benchmark will actually use is the point.
//!
//! If a future change makes `Static` score above chance, the corpus leaks and
//! every downstream result is void. These tests are the tripwire for that.

use symthaea_evidence_plane::task_validator::{self, TimedItem, TokenId};

use super::context_aliasing::{self, ContextAliasingConfig};
use super::irregular_time::{self, IrregularTimeConfig};

/// Chance level for a two-branch task.
const CHANCE: f64 = 0.5;
/// How far from chance a zero-history arm may land before the corpus is
/// considered leaky. Tight on purpose: `Static` cannot do better than chance
/// unless information reached it that should not have.
const CHANCE_TOLERANCE: f64 = 1e-9;

// ═══════════════════════════════════════════════════════════════════════════
// §5.1 context aliasing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn context_aliasing_pipeline_generator_to_validator_to_scorer() {
    let task = context_aliasing::generate(ContextAliasingConfig {
        branches: 2,
        distractors: 2,
        repeats: 32,
    })
    .expect("generator must pass its own gate");

    // The validator agrees the corpus is what the generator claims.
    let metrics = task_validator::validate(&task.sequences, task.config.required_depth())
        .expect("independently re-validated");
    assert!(metrics.ambiguous_transitions > 0, "nothing to score");

    // An oracle arm reading the true next token scores perfectly.
    let oracle = task_validator::score(&task.sequences, |s, t| task.sequences[s][t + 1]);
    assert_eq!(oracle.ambiguous_accuracy, 1.0);

    // GATE 3: a zero-history arm must sit exactly at chance on ambiguous points.
    // It sees only the current token, which is identical across branches, so it
    // must commit to one answer and be right half the time.
    let static_arm = task_validator::score(&task.sequences, |_, _| {
        context_aliasing::ALIAS_TOKEN.wrapping_add(1)
    });
    assert!(
        static_arm.ambiguous_accuracy.abs() < CHANCE_TOLERANCE
            || (static_arm.ambiguous_accuracy - CHANCE).abs() < CHANCE_TOLERANCE,
        "zero-history arm must be at chance or zero, got {} — corpus may leak",
        static_arm.ambiguous_accuracy
    );

    // §6's whole reason for existing: the global number must not be trusted.
    assert!(
        oracle.global_accuracy >= oracle.ambiguous_accuracy - 1e-9,
        "global accuracy should be at least as forgiving as ambiguous-point accuracy"
    );
}

/// A committed-guess zero-history arm lands at exactly chance, and a global-only
/// report would make it look far better than it is.
#[test]
fn static_arm_looks_good_globally_and_is_at_chance_where_it_matters() {
    let task = context_aliasing::generate(ContextAliasingConfig {
        branches: 2,
        distractors: 3,
        repeats: 32,
    })
    .expect("must validate");

    // Predict the true next token everywhere EXCEPT the aliased point, where a
    // zero-history arm must guess. This is the best a memoryless arm can do.
    let seqs = &task.sequences;
    let alias_pos = seqs[0]
        .iter()
        .position(|&t| t == context_aliasing::ALIAS_TOKEN)
        .expect("alias present");
    let guess = seqs[0][alias_pos + 1];
    let s = task_validator::score(seqs, |sq, t| {
        if t == alias_pos {
            guess
        } else {
            seqs[sq][t + 1]
        }
    });

    assert!(
        (s.ambiguous_accuracy - CHANCE).abs() < CHANCE_TOLERANCE,
        "memoryless arm must be at chance on ambiguous points, got {}",
        s.ambiguous_accuracy
    );
    assert!(
        s.global_accuracy > 0.85,
        "and yet look strong globally ({}) — this is exactly the concealment §6 \
         exists to prevent",
        s.global_accuracy
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// §5.2 irregular time
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn irregular_time_pipeline_is_scoreable_end_to_end() {
    let task = irregular_time::generate(IrregularTimeConfig::default())
        .expect("generator must pass its own gate");

    // A timing-aware oracle: read the interval on the decision point and apply
    // the generating rule. This is the capability the family is meant to test.
    let train = &task.train;
    let timed_oracle = task_validator::score_timed(train, |s, t| {
        let interval = train[s][t].dt_since_prev;
        task.config.target_for(interval)
    });
    assert_eq!(
        timed_oracle.ambiguous_accuracy, 1.0,
        "a timing-aware arm must solve every ambiguous point"
    );

    // GATE 3 for this family: an arm that ignores timing must be at chance.
    // The token sequence is identical across branches, so it has nothing else.
    let timing_blind = task_validator::score_timed(train, |_, _| irregular_time::FAST_TARGET);
    assert!(
        (timing_blind.ambiguous_accuracy - CHANCE).abs() < 0.01,
        "timing-blind arm must be at chance, got {} — the corpus may leak the \
         answer through tokens",
        timing_blind.ambiguous_accuracy
    );
}

/// Interpolation and extrapolation must both be solvable by the generating rule
/// — otherwise a mechanism could fail them for reasons unrelated to temporal
/// capability and the split would measure nothing.
#[test]
fn held_out_splits_are_solvable_by_the_rule() {
    let task = irregular_time::generate(IrregularTimeConfig::default()).expect("must validate");

    for (name, corpus) in [
        ("interpolation", &task.interpolation),
        ("extrapolation", &task.extrapolation),
    ] {
        let s = task_validator::score_timed(corpus, |sq, t| {
            task.config.target_for(corpus[sq][t].dt_since_prev)
        });
        assert_eq!(
            s.ambiguous_accuracy, 1.0,
            "{name} split must be perfectly solvable by the generating rule"
        );
    }
}

/// The two families must not be accidentally interchangeable: §5.1 is solvable
/// from token history alone, §5.2 is not solvable from tokens at all. If both
/// were solvable the same way, the ladder would be one task wearing two names.
#[test]
fn the_two_families_require_genuinely_different_capabilities() {
    let aliasing = context_aliasing::generate(ContextAliasingConfig::default()).expect("ok");
    let timing = irregular_time::generate(IrregularTimeConfig::default()).expect("ok");

    // §5.1 validates from tokens alone.
    assert!(
        task_validator::validate(&aliasing.sequences, aliasing.config.required_depth()).is_ok()
    );

    // §5.2 does not — its token view is unsolvable.
    let timing_tokens: Vec<Vec<TokenId>> = timing
        .train
        .iter()
        .map(|s| s.iter().map(|i: &TimedItem| i.token).collect())
        .collect();
    assert!(
        task_validator::validate(&timing_tokens, 2).is_err(),
        "§5.2 must NOT be solvable from tokens, or it duplicates §5.1"
    );
}
