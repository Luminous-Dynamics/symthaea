// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Information-theoretic task validator for sequence benchmarks.
//!
//! Implements §4 of `SYMTHAEA_TEMPORAL_BENCHMARK_V2_PLAN.md`: **before any
//! training begins, a task generator must prove its target requires memory
//! rather than merely asserting it by construction.**
//!
//! This exists because that assertion silently failed once already. The
//! predecessor temporal benchmark used a corpus that was exactly periodic, so
//! next-item prediction was solvable from current-item identity alone with zero
//! history — and an entire measurement arc was run, reported, and later
//! retracted before anyone noticed. `validate()` would have caught it before a
//! single model trained. The test suite in this module includes that exact
//! corpus as a regression.
//!
//! # The two requirements
//!
//! For a next-item task with current item `X_t`, target `Y_{t+1}`, history `H_t`:
//!
//! - **Ambiguity**: `H(Y_{t+1} | X_t) > 0` — knowing only the current item must
//!   not uniquely determine the next one.
//! - **History resolves ambiguity**: `I(Y_{t+1}; H_t | X_t) > 0` — prior context
//!   must supply predictive information beyond the current item alone.
//!
//! Both are necessary. A task can be ambiguous but *irreducibly* so (noise),
//! in which case history does not help and the task measures nothing about
//! memory.
//!
//! # Hard-fail, not warn
//!
//! [`validate`] returns `Err` listing every violated check. Matching this
//! crate's existing philosophy, a benchmark should treat that as fatal — the
//! point is to make an unusable task impossible to run past, not to annotate it.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Token identifier. Deliberately opaque — the validator is content-agnostic.
pub type TokenId = u32;

/// How much prior context the history-aware oracle may use.
///
/// This should match the depth the benchmark intends to require. Validating at
/// depth 1 and then training a task that needs depth 4 proves nothing.
pub type HistoryDepth = usize;

/// A single check that can fail, with the measured value that failed it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskViolation {
    /// No current item has more than one distinct successor: the task is
    /// deterministic given the current item, so memory cannot help.
    NoAmbiguity { distinct_successors_max: usize },
    /// `H(Y|X) == 0`. Equivalent in spirit to `NoAmbiguity`, reported
    /// separately because a task can have multiple successors overall while
    /// still being deterministic in the weighted-average sense.
    ZeroConditionalEntropy { h_y_given_x: f64 },
    /// `I(Y; H | X) <= 0`: history supplies no predictive information beyond
    /// the current item. The ambiguity is irreducible (noise), not resolvable.
    HistoryUninformative { conditional_mutual_information: f64 },
    /// A current-item-only oracle already solves the task, so no history-bearing
    /// arm can demonstrate an advantage.
    CurrentItemOracleSolvesTask { accuracy: f64 },
    /// The history-aware oracle does not substantially outperform the
    /// current-item-only oracle, so the task cannot separate the arms.
    HistoryOracleNoBetter {
        current_only: f64,
        history_aware: f64,
        margin_required: f64,
    },
    /// Sequences are short-period repetitions. This is the predecessor's exact
    /// failure mode and is checked explicitly even though it usually also trips
    /// the ambiguity checks, because it is the one we know actually happened.
    PeriodicCorpus {
        sequences_periodic: usize,
        sequences_total: usize,
    },
    /// Not enough usable transitions to compute anything trustworthy.
    InsufficientData { transitions: usize, required: usize },
}

/// Measured quantities, reported whether or not validation passes.
///
/// Returned on success and embedded in the error, because the numbers are
/// diagnostic in both cases — a task that fails by a hair is a different
/// situation from one that fails by an order of magnitude.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskMetrics {
    /// `H(Y_{t+1} | X_t)` in bits.
    pub h_y_given_x: f64,
    /// `H(Y_{t+1} | X_t, H_t)` in bits.
    pub h_y_given_x_and_history: f64,
    /// `I(Y_{t+1}; H_t | X_t) = H(Y|X) - H(Y|X,H)`, in bits.
    pub conditional_mutual_information: f64,
    /// Accuracy of a best-case current-item-only oracle over ALL transitions.
    /// Global on purpose: "is this task already solved without history?" is a
    /// property of the whole corpus.
    pub current_only_oracle_accuracy: f64,
    /// Accuracy of a best-case history-aware oracle over ALL transitions.
    /// Diagnostic only — never used for separation, see the ambiguous variants.
    pub history_oracle_accuracy: f64,
    /// Current-item-only oracle restricted to ambiguous decision points.
    pub ambiguous_current_oracle_accuracy: f64,
    /// History-aware oracle restricted to ambiguous decision points. Separation
    /// is judged on THIS pair, per plan §6, so that adding deterministic filler
    /// cannot shrink the measured gap and reject a task for being harder.
    pub ambiguous_history_oracle_accuracy: f64,
    /// Largest number of distinct successors any single current item has.
    pub max_distinct_successors: usize,
    /// Transitions used.
    pub transitions: usize,
    /// Transitions at decision points the current item alone cannot resolve.
    /// Plan §6 requires scoring to be restricted to exactly these points; this
    /// is the validator identifying them, as §4 says it should.
    pub ambiguous_transitions: usize,
    /// History depth the metrics were computed at.
    pub history_depth: HistoryDepth,
}

/// Minimum oracle-accuracy gap for a task to be considered separating.
///
/// Not tuned against any result: it is set to the smallest gap that could not
/// plausibly be an artifact of finite sampling at the corpus sizes this
/// validator is meant for, and is exposed so a caller can demand more. It is
/// deliberately *not* adjustable downward at the call site by default.
pub const DEFAULT_ORACLE_MARGIN: f64 = 0.10;

/// Minimum transitions before the metrics are considered meaningful.
pub const MIN_TRANSITIONS: usize = 32;

/// Validate that a sequence corpus requires memory.
///
/// Returns `Ok(metrics)` if every check passes, or `Err((metrics, violations))`
/// listing *all* violated checks — not just the first — so a task generator can
/// be fixed in one pass rather than one failure at a time.
pub fn validate(
    sequences: &[Vec<TokenId>],
    history_depth: HistoryDepth,
) -> Result<TaskMetrics, (TaskMetrics, Vec<TaskViolation>)> {
    validate_with_margin(sequences, history_depth, DEFAULT_ORACLE_MARGIN)
}

/// [`validate`] with an explicit oracle-separation margin.
pub fn validate_with_margin(
    sequences: &[Vec<TokenId>],
    history_depth: HistoryDepth,
    oracle_margin: f64,
) -> Result<TaskMetrics, (TaskMetrics, Vec<TaskViolation>)> {
    // (current) -> successor counts
    let mut by_current: HashMap<TokenId, HashMap<TokenId, usize>> = HashMap::new();
    // (history, current) -> successor counts
    let mut by_context: HashMap<(Vec<TokenId>, TokenId), HashMap<TokenId, usize>> = HashMap::new();

    for seq in sequences {
        if seq.len() < 2 {
            continue;
        }
        for t in 0..seq.len() - 1 {
            let x = seq[t];
            let y = seq[t + 1];
            *by_current.entry(x).or_default().entry(y).or_insert(0) += 1;

            // History is the `history_depth` tokens strictly before X_t. Short
            // prefixes are kept (padded by truncation) rather than skipped, so
            // early positions still contribute rather than silently biasing the
            // estimate toward long-context regions.
            let start = t.saturating_sub(history_depth);
            let hist: Vec<TokenId> = seq[start..t].to_vec();
            *by_context
                .entry((hist, x))
                .or_default()
                .entry(y)
                .or_insert(0) += 1;
        }
    }

    let transitions: usize = by_current
        .values()
        .flat_map(|m| m.values())
        .copied()
        .sum::<usize>();

    let h_y_given_x = conditional_entropy(by_current.values());
    let h_y_given_x_and_history = conditional_entropy(by_context.values());
    let cmi = h_y_given_x - h_y_given_x_and_history;

    // Oracle separation is measured AT AMBIGUOUS DECISION POINTS ONLY, never
    // globally. Plan §6: "Global average error can hide the real comparison,
    // since most tokens in any real corpus may be solvable without history."
    // Averaging over deterministic filler transitions shrinks the gap toward
    // zero as distractors are added, so a global gap would reject harder tasks
    // for being harder. Found 2026-07-31 when the context-aliasing generator's
    // distractor knob made a valid depth-5 task fail its own gate.
    let ambiguous_x: std::collections::HashSet<TokenId> = by_current
        .iter()
        .filter(|(_, succ)| succ.len() > 1)
        .map(|(x, _)| *x)
        .collect();
    let ambiguous_transitions: usize = by_current
        .iter()
        .filter(|(x, _)| ambiguous_x.contains(x))
        .flat_map(|(_, succ)| succ.values())
        .sum();
    let global_current_oracle = oracle_accuracy(by_current.values(), transitions);
    let global_history_oracle = oracle_accuracy(by_context.values(), transitions);
    let ambiguous_current_oracle_accuracy = oracle_accuracy(
        by_current
            .iter()
            .filter(|(x, _)| ambiguous_x.contains(x))
            .map(|(_, succ)| succ),
        ambiguous_transitions,
    );
    let ambiguous_history_oracle_accuracy = oracle_accuracy(
        by_context
            .iter()
            .filter(|((_, x), _)| ambiguous_x.contains(x))
            .map(|(_, succ)| succ),
        ambiguous_transitions,
    );
    let max_distinct_successors = by_current.values().map(|m| m.len()).max().unwrap_or(0);

    let (sequences_periodic, sequences_total) = count_periodic(sequences);

    let metrics = TaskMetrics {
        h_y_given_x,
        h_y_given_x_and_history,
        conditional_mutual_information: cmi,
        current_only_oracle_accuracy: global_current_oracle,
        history_oracle_accuracy: global_history_oracle,
        ambiguous_current_oracle_accuracy,
        ambiguous_history_oracle_accuracy,
        max_distinct_successors,
        transitions,
        ambiguous_transitions,
        history_depth,
    };

    let mut violations = Vec::new();

    if transitions < MIN_TRANSITIONS {
        violations.push(TaskViolation::InsufficientData {
            transitions,
            required: MIN_TRANSITIONS,
        });
    }
    if max_distinct_successors <= 1 {
        violations.push(TaskViolation::NoAmbiguity {
            distinct_successors_max: max_distinct_successors,
        });
    }
    if h_y_given_x <= 0.0 {
        violations.push(TaskViolation::ZeroConditionalEntropy { h_y_given_x });
    }
    if cmi <= 0.0 {
        violations.push(TaskViolation::HistoryUninformative {
            conditional_mutual_information: cmi,
        });
    }
    if global_current_oracle >= 1.0 {
        violations.push(TaskViolation::CurrentItemOracleSolvesTask {
            accuracy: global_current_oracle,
        });
    }
    // Only meaningful where ambiguity exists; when it does not, `NoAmbiguity`
    // already reports the real problem and this would be a confusing duplicate.
    if ambiguous_transitions > 0
        && ambiguous_history_oracle_accuracy - ambiguous_current_oracle_accuracy < oracle_margin
    {
        violations.push(TaskViolation::HistoryOracleNoBetter {
            current_only: ambiguous_current_oracle_accuracy,
            history_aware: ambiguous_history_oracle_accuracy,
            margin_required: oracle_margin,
        });
    }
    if sequences_total > 0 && sequences_periodic == sequences_total {
        violations.push(TaskViolation::PeriodicCorpus {
            sequences_periodic,
            sequences_total,
        });
    }

    if violations.is_empty() {
        Ok(metrics)
    } else {
        Err((metrics, violations))
    }
}

/// Weighted conditional entropy `H(Y | C)` in bits over per-context successor
/// count maps, weighted by how often each context occurs.
fn conditional_entropy<'a, I>(contexts: I) -> f64
where
    I: Iterator<Item = &'a HashMap<TokenId, usize>>,
{
    let counts: Vec<&HashMap<TokenId, usize>> = contexts.collect();
    let total: usize = counts.iter().flat_map(|m| m.values()).copied().sum();
    if total == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for ctx in counts {
        let ctx_total: usize = ctx.values().copied().sum();
        if ctx_total == 0 {
            continue;
        }
        let mut h = 0.0;
        for &c in ctx.values() {
            if c == 0 {
                continue;
            }
            let p = c as f64 / ctx_total as f64;
            h -= p * p.log2();
        }
        acc += (ctx_total as f64 / total as f64) * h;
    }
    acc
}

/// Best-case oracle accuracy: always predict each context's most frequent
/// successor. This is the theoretical ceiling for a predictor with that
/// context, so it bounds what any model could achieve.
fn oracle_accuracy<'a, I>(contexts: I, total: usize) -> f64
where
    I: Iterator<Item = &'a HashMap<TokenId, usize>>,
{
    if total == 0 {
        return 0.0;
    }
    let hits: usize = contexts
        .map(|m| m.values().copied().max().unwrap_or(0))
        .sum();
    hits as f64 / total as f64
}

/// Count sequences that are short-period repetitions.
///
/// A sequence is periodic here if it has a period `p` with `p * 2 <= len`, i.e.
/// the whole sequence is at least two full repetitions of a block.
fn count_periodic(sequences: &[Vec<TokenId>]) -> (usize, usize) {
    let mut periodic = 0;
    let mut total = 0;
    for seq in sequences {
        if seq.len() < 4 {
            continue;
        }
        total += 1;
        if smallest_period(seq).is_some_and(|p| p * 2 <= seq.len()) {
            periodic += 1;
        }
    }
    (periodic, total)
}

fn smallest_period(seq: &[TokenId]) -> Option<usize> {
    for p in 1..=seq.len() / 2 {
        if seq.iter().skip(p).zip(seq.iter()).all(|(a, b)| a == b) {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat(pattern: &[TokenId], times: usize) -> Vec<TokenId> {
        pattern
            .iter()
            .copied()
            .cycle()
            .take(pattern.len() * times)
            .collect()
    }

    /// THE REGRESSION THAT MATTERS. The predecessor benchmark's corpus was
    /// exactly periodic, making next-item prediction solvable from current-item
    /// identity alone. An entire measurement arc ran on it before the flaw was
    /// found. This validator must reject it.
    #[test]
    fn rejects_the_periodic_corpus_that_broke_the_predecessor() {
        let corpus: Vec<Vec<TokenId>> = (0..8).map(|_| repeat(&[1, 2, 3, 4], 12)).collect();
        let err = validate(&corpus, 4).expect_err("periodic corpus must be rejected");
        let (metrics, violations) = err;

        // A periodic corpus is fully determined by the current item.
        assert_eq!(metrics.max_distinct_successors, 1);
        assert_eq!(metrics.current_only_oracle_accuracy, 1.0);
        assert!(violations.contains(&TaskViolation::NoAmbiguity {
            distinct_successors_max: 1
        }));
        assert!(violations.contains(&TaskViolation::CurrentItemOracleSolvesTask { accuracy: 1.0 }));
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, TaskViolation::PeriodicCorpus { .. }))
        );
    }

    /// The context-aliasing task the plan specifies: A->B->X->C and D->E->X->F.
    /// At X the current item is identical in both branches, so only history can
    /// resolve the target. This must PASS.
    #[test]
    fn accepts_context_aliasing_task() {
        let mut corpus = Vec::new();
        for _ in 0..24 {
            corpus.push(vec![1, 2, 99, 3]); // A B X C
            corpus.push(vec![4, 5, 99, 6]); // D E X F
        }
        let metrics = validate(&corpus, 2).expect("context-aliasing task must validate");

        assert!(metrics.max_distinct_successors >= 2);
        assert!(metrics.h_y_given_x > 0.0, "task must be ambiguous");
        assert!(
            metrics.conditional_mutual_information > 0.0,
            "history must resolve the ambiguity"
        );
        assert!(metrics.current_only_oracle_accuracy < 1.0);
        assert!(metrics.history_oracle_accuracy > metrics.current_only_oracle_accuracy);
    }

    /// Ambiguity alone is not enough. A corpus where the successor at the
    /// aliased point is genuinely random cannot be resolved by history, so it
    /// measures noise tolerance rather than memory, and must be rejected.
    #[test]
    fn rejects_irreducible_ambiguity() {
        // At token 99 the successor alternates independently of any prior
        // context: both branches lead to both outcomes equally.
        let mut corpus = Vec::new();
        for i in 0..48 {
            let tail = if i % 2 == 0 { 3 } else { 6 };
            corpus.push(vec![1, 2, 99, tail]);
        }
        let (metrics, violations) =
            validate(&corpus, 2).expect_err("irreducible ambiguity must be rejected");
        assert!(metrics.h_y_given_x > 0.0, "it IS ambiguous");
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, TaskViolation::HistoryUninformative { .. })
                    || matches!(v, TaskViolation::HistoryOracleNoBetter { .. })),
            "history cannot help, so it must fail one of the history checks; got {violations:?}"
        );
    }

    /// A validator that accepts everything is worthless. Guard the guard: an
    /// empty corpus must fail rather than vacuously pass.
    #[test]
    fn rejects_empty_and_trivial_corpora() {
        let (_, violations) = validate(&[], 2).expect_err("empty corpus must fail");
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, TaskViolation::InsufficientData { .. }))
        );

        let (_, violations) = validate(&[vec![1, 2]], 2).expect_err("two-token corpus must fail");
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, TaskViolation::InsufficientData { .. }))
        );
    }

    /// Validating at the wrong depth proves nothing. A task requiring 3 tokens
    /// of context must not validate at depth 1.
    #[test]
    fn depth_matters() {
        // Distinguishing cue is 3 back from the decision point.
        let mut corpus = Vec::new();
        for _ in 0..32 {
            corpus.push(vec![1, 7, 8, 99, 3]);
            corpus.push(vec![4, 7, 8, 99, 6]);
        }
        // At depth 1, history is just token 8 in both branches — no help.
        let shallow = validate(&corpus, 1);
        assert!(
            shallow.is_err(),
            "depth 1 cannot see the cue and must not validate"
        );
        // At depth 3 the cue (1 vs 4) is visible.
        let deep = validate(&corpus, 3).expect("depth 3 sees the cue");
        assert!(deep.conditional_mutual_information > 0.0);
    }

    #[test]
    fn entropy_of_deterministic_context_is_zero() {
        let mut m: HashMap<TokenId, usize> = HashMap::new();
        m.insert(5, 10);
        let maps = vec![m];
        assert_eq!(conditional_entropy(maps.iter()), 0.0);
    }

    #[test]
    fn entropy_of_uniform_binary_context_is_one_bit() {
        let mut m: HashMap<TokenId, usize> = HashMap::new();
        m.insert(5, 10);
        m.insert(6, 10);
        let maps = vec![m];
        assert!((conditional_entropy(maps.iter()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn smallest_period_detects_repetition() {
        assert_eq!(smallest_period(&[1, 2, 1, 2, 1, 2]), Some(2));
        assert_eq!(smallest_period(&[1, 2, 3, 4, 5, 6]), None);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Timed sequences — §5.2 irregular-time tasks
// ═══════════════════════════════════════════════════════════════════════════

/// One observation with the elapsed time since its predecessor.
///
/// Exists because the §5.2 irregular-time task family is *unvalidatable* by the
/// token-only [`validate`]: in `A --100ms--> B -> C` versus `A --20s--> B -> D`
/// the token history is identical and only the interval differs, so a token-only
/// analysis computes `I(Y; H | X) = 0` and correctly rejects the corpus as
/// unsolvable. It genuinely is unsolvable *from tokens*. The discriminating
/// information is the elapsed time, so the validator has to be able to see it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimedItem {
    pub token: TokenId,
    /// Elapsed time since the previous item. Ignored for the first item.
    pub dt_since_prev: f64,
}

/// Additional violations specific to timed validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimingViolation {
    /// Elapsed time supplies no predictive information beyond the tokens, so the
    /// task does not test the claim that time itself changes the prediction.
    TimingUninformative { timing_cmi: f64 },
    /// All intervals fall in one bin: there is no timing variation to exploit.
    NoIntervalVariation { occupied_bins: usize },
    /// Bin edges were not sorted/finite, so binning would be meaningless.
    MalformedBins,
}

/// Metrics for a timed corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedTaskMetrics {
    /// `H(Y | tokens only)` — history and current item, timing discarded.
    pub h_y_given_tokens: f64,
    /// `H(Y | tokens and binned intervals)`.
    pub h_y_given_tokens_and_timing: f64,
    /// `I(Y; timing | tokens)`. The quantity that matters: how much predictive
    /// information elapsed time carries *beyond* what the tokens already say.
    pub timing_cmi: f64,
    /// Oracle accuracy from tokens alone.
    pub token_oracle_accuracy: f64,
    /// Oracle accuracy from tokens plus binned timing.
    pub timed_oracle_accuracy: f64,
    /// Interval bins that actually occur.
    pub occupied_bins: usize,
    pub transitions: usize,
}

/// Validate that a timed corpus requires *elapsed time*, not just order.
///
/// `bin_edges` must be sorted and finite; an interval is assigned to the first
/// bin whose edge it does not exceed. Edges are supplied explicitly rather than
/// inferred, so the discretization is a stated choice of the experiment rather
/// than a hidden one chosen by this function.
pub fn validate_timed(
    sequences: &[Vec<TimedItem>],
    history_depth: HistoryDepth,
    bin_edges: &[f64],
) -> Result<TimedTaskMetrics, (TimedTaskMetrics, Vec<TimingViolation>)> {
    let bins_ok = !bin_edges.is_empty()
        && bin_edges.iter().all(|e| e.is_finite())
        && bin_edges.windows(2).all(|w| w[0] <= w[1]);

    let bin_of = |dt: f64| -> usize {
        match bin_edges.iter().position(|&e| dt <= e) {
            Some(i) => i,
            None => bin_edges.len(),
        }
    };

    // Context keyed on tokens only, and on tokens + binned intervals.
    let mut by_tokens: HashMap<(Vec<TokenId>, TokenId), HashMap<TokenId, usize>> = HashMap::new();
    let mut by_timed: HashMap<(Vec<TokenId>, Vec<usize>, TokenId, usize), HashMap<TokenId, usize>> =
        HashMap::new();
    let mut occupied: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for seq in sequences {
        if seq.len() < 2 {
            continue;
        }
        for t in 0..seq.len() - 1 {
            let x = seq[t].token;
            let y = seq[t + 1].token;
            let start = t.saturating_sub(history_depth);

            let hist_tokens: Vec<TokenId> = seq[start..t].iter().map(|i| i.token).collect();
            let hist_bins: Vec<usize> = seq[start..t]
                .iter()
                .map(|i| bin_of(i.dt_since_prev))
                .collect();
            // The current item's OWN incoming interval -- how long since the
            // previous event. Deliberately NOT `seq[t + 1].dt_since_prev`:
            // that is the interval leading into the target, which at prediction
            // time has not happened yet. Using it would be future information,
            // and would certify tasks that are causally unsolvable. Corrected
            // 2026-07-31 while building the §5.2 generator against this gate.
            let x_bin = bin_of(seq[t].dt_since_prev);
            occupied.insert(x_bin);

            *by_tokens
                .entry((hist_tokens.clone(), x))
                .or_default()
                .entry(y)
                .or_insert(0) += 1;
            *by_timed
                .entry((hist_tokens, hist_bins, x, x_bin))
                .or_default()
                .entry(y)
                .or_insert(0) += 1;
        }
    }

    let transitions: usize = by_tokens.values().flat_map(|m| m.values()).sum();
    let h_tokens = conditional_entropy(by_tokens.values());
    let h_timed = conditional_entropy(by_timed.values());
    let metrics = TimedTaskMetrics {
        h_y_given_tokens: h_tokens,
        h_y_given_tokens_and_timing: h_timed,
        timing_cmi: h_tokens - h_timed,
        token_oracle_accuracy: oracle_accuracy(by_tokens.values(), transitions),
        timed_oracle_accuracy: oracle_accuracy(by_timed.values(), transitions),
        occupied_bins: occupied.len(),
        transitions,
    };

    let mut violations = Vec::new();
    if !bins_ok {
        violations.push(TimingViolation::MalformedBins);
    }
    if metrics.occupied_bins <= 1 {
        violations.push(TimingViolation::NoIntervalVariation {
            occupied_bins: metrics.occupied_bins,
        });
    }
    if metrics.timing_cmi <= 0.0 {
        violations.push(TimingViolation::TimingUninformative {
            timing_cmi: metrics.timing_cmi,
        });
    }

    if violations.is_empty() {
        Ok(metrics)
    } else {
        Err((metrics, violations))
    }
}

#[cfg(test)]
mod timed_tests {
    use super::*;

    /// The §5.2 corpus: identical tokens and order, different elapsed time,
    /// different correct target.
    fn irregular_time_corpus() -> Vec<Vec<TimedItem>> {
        let mut out = Vec::new();
        for _ in 0..32 {
            // A --fast--> B -> C
            out.push(vec![
                TimedItem {
                    token: 1,
                    dt_since_prev: 0.0,
                },
                TimedItem {
                    token: 2,
                    dt_since_prev: 0.1,
                },
                TimedItem {
                    token: 3,
                    dt_since_prev: 0.1,
                },
            ]);
            // A --slow--> B -> D
            out.push(vec![
                TimedItem {
                    token: 1,
                    dt_since_prev: 0.0,
                },
                TimedItem {
                    token: 2,
                    dt_since_prev: 20.0,
                },
                TimedItem {
                    token: 4,
                    dt_since_prev: 20.0,
                },
            ]);
        }
        out
    }

    /// The point of the whole extension: this corpus MUST fail token-only
    /// validation and MUST pass timed validation. If it passed token-only, the
    /// timing would not be load-bearing and §5.2 would be testing nothing.
    #[test]
    fn irregular_time_task_needs_timing_and_only_timing() {
        let timed = irregular_time_corpus();
        let tokens_only: Vec<Vec<TokenId>> = timed
            .iter()
            .map(|s| s.iter().map(|i| i.token).collect())
            .collect();

        assert!(
            validate(&tokens_only, 2).is_err(),
            "token-only view must be unsolvable — the tokens are identical across branches"
        );

        let m = validate_timed(&timed, 2, &[1.0]).expect("timed view must validate");
        assert!(
            m.timing_cmi > 0.0,
            "elapsed time must carry predictive information beyond tokens"
        );
        assert!(m.timed_oracle_accuracy > m.token_oracle_accuracy);
        assert_eq!(m.occupied_bins, 2);
    }

    /// A corpus where timing varies but is irrelevant to the target must be
    /// rejected — otherwise the task would credit a mechanism for using timing
    /// that carries no signal.
    #[test]
    fn rejects_irrelevant_timing() {
        let mut out = Vec::new();
        for i in 0..64 {
            let dt = if i % 2 == 0 { 0.1 } else { 20.0 };
            // Target depends on the CUE, not on the interval.
            let (cue, target) = if i % 4 < 2 { (1, 3) } else { (5, 6) };
            out.push(vec![
                TimedItem {
                    token: cue,
                    dt_since_prev: 0.0,
                },
                TimedItem {
                    token: 2,
                    dt_since_prev: dt,
                },
                TimedItem {
                    token: target,
                    dt_since_prev: dt,
                },
            ]);
        }
        let (_, violations) =
            validate_timed(&out, 2, &[1.0]).expect_err("irrelevant timing must be rejected");
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, TimingViolation::TimingUninformative { .. }))
        );
    }

    #[test]
    fn rejects_single_bin_and_malformed_bins() {
        let corpus = irregular_time_corpus();
        // All intervals land in one bin.
        let (_, v) = validate_timed(&corpus, 2, &[1e9]).expect_err("single bin must fail");
        assert!(
            v.iter()
                .any(|x| matches!(x, TimingViolation::NoIntervalVariation { .. }))
        );

        let (_, v) = validate_timed(&corpus, 2, &[f64::NAN]).expect_err("malformed bins must fail");
        assert!(
            v.iter()
                .any(|x| matches!(x, TimingViolation::MalformedBins))
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §6 scoring — restricted to ambiguous decision points
// ═══════════════════════════════════════════════════════════════════════════

/// A transition the current item alone cannot resolve.
///
/// §4 requires the validator to identify "exactly which points those are", and
/// §6 requires scoring to be restricted to them. Counting them was not enough:
/// a scorer needs their locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmbiguousPoint {
    /// Index into the corpus.
    pub sequence: usize,
    /// Position `t` of the current item; the prediction target is `t + 1`.
    pub position: usize,
}

/// Locate every ambiguous decision point in a corpus.
///
/// A point is ambiguous when its current item has more than one distinct
/// successor anywhere in the corpus — i.e. exactly the points where history is
/// the only thing that can decide the answer.
pub fn ambiguous_points(sequences: &[Vec<TokenId>]) -> Vec<AmbiguousPoint> {
    let mut successors: HashMap<TokenId, std::collections::HashSet<TokenId>> = HashMap::new();
    for seq in sequences {
        for t in 0..seq.len().saturating_sub(1) {
            successors.entry(seq[t]).or_default().insert(seq[t + 1]);
        }
    }
    let mut out = Vec::new();
    for (s, seq) in sequences.iter().enumerate() {
        for t in 0..seq.len().saturating_sub(1) {
            if successors.get(&seq[t]).is_some_and(|set| set.len() > 1) {
                out.push(AmbiguousPoint {
                    sequence: s,
                    position: t,
                });
            }
        }
    }
    out
}

/// Result of scoring an arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguousScore {
    /// **The primary metric.** Accuracy restricted to ambiguous points.
    pub ambiguous_accuracy: f64,
    /// Accuracy over every point. Reported for contrast, never as the headline —
    /// §6: global average error hides the real comparison.
    pub global_accuracy: f64,
    pub ambiguous_points: usize,
    pub total_points: usize,
}

/// Score predictions per §6.
///
/// `predict(sequence_index, position) -> TokenId` is the arm under test,
/// predicting the item at `position + 1`.
///
/// Both accuracies are returned deliberately. A large gap between them is the
/// signal §6 exists to surface: an arm can look strong globally purely by
/// solving the deterministic majority while failing every point that needs
/// memory. Reporting only the global number is how the predecessor benchmark
/// concealed a corpus that required no memory at all.
pub fn score<F>(sequences: &[Vec<TokenId>], mut predict: F) -> AmbiguousScore
where
    F: FnMut(usize, usize) -> TokenId,
{
    let ambiguous: std::collections::HashSet<(usize, usize)> = ambiguous_points(sequences)
        .into_iter()
        .map(|p| (p.sequence, p.position))
        .collect();

    let (mut amb_hits, mut amb_n, mut all_hits, mut all_n) = (0usize, 0usize, 0usize, 0usize);
    for (s, seq) in sequences.iter().enumerate() {
        for t in 0..seq.len().saturating_sub(1) {
            let correct = predict(s, t) == seq[t + 1];
            all_n += 1;
            all_hits += correct as usize;
            if ambiguous.contains(&(s, t)) {
                amb_n += 1;
                amb_hits += correct as usize;
            }
        }
    }
    AmbiguousScore {
        ambiguous_accuracy: if amb_n == 0 {
            f64::NAN
        } else {
            amb_hits as f64 / amb_n as f64
        },
        global_accuracy: if all_n == 0 {
            f64::NAN
        } else {
            all_hits as f64 / all_n as f64
        },
        ambiguous_points: amb_n,
        total_points: all_n,
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::*;

    /// Context-aliasing shape: only the transition out of the aliased token is
    /// ambiguous; the cue transitions are deterministic.
    fn corpus() -> Vec<Vec<TokenId>> {
        let mut c = Vec::new();
        for _ in 0..16 {
            c.push(vec![1, 99, 3]);
            c.push(vec![4, 99, 6]);
        }
        c
    }

    #[test]
    fn identifies_only_the_aliased_points() {
        let pts = ambiguous_points(&corpus());
        // One per sequence, always at the aliased token (position 1).
        assert_eq!(pts.len(), 32);
        assert!(pts.iter().all(|p| p.position == 1));
    }

    /// THE §6 PROPERTY. An arm that solves every deterministic point and fails
    /// every ambiguous one looks strong globally and is worthless. The primary
    /// metric must expose that; the global metric must not be trusted.
    #[test]
    fn global_accuracy_hides_total_failure_on_the_points_that_matter() {
        let c = corpus();
        // Always predict 99 (right for the cue transitions, wrong at the alias).
        let s = score(&c, |_, _| 99);
        assert_eq!(
            s.ambiguous_accuracy, 0.0,
            "fails every point requiring memory"
        );
        assert!(
            s.global_accuracy >= 0.5,
            "yet looks respectable globally: {}",
            s.global_accuracy
        );
    }

    /// A perfect history-using arm scores 1.0 on both.
    #[test]
    fn oracle_arm_scores_perfectly() {
        let c = corpus();
        let s = score(&c, |seq, t| c[seq][t + 1]);
        assert_eq!(s.ambiguous_accuracy, 1.0);
        assert_eq!(s.global_accuracy, 1.0);
    }

    /// A zero-history arm is at chance on ambiguous points by construction —
    /// the mechanically guaranteed negative control the pre-registration
    /// requires, not a hopefully-weak baseline.
    #[test]
    fn static_arm_is_at_chance_on_ambiguous_points() {
        let c = corpus();
        // Cannot see history, so it must commit to one answer at the alias.
        let s = score(&c, |_, t| if t == 1 { 3 } else { 99 });
        assert!(
            (s.ambiguous_accuracy - 0.5).abs() < 1e-9,
            "expected chance, got {}",
            s.ambiguous_accuracy
        );
    }

    #[test]
    fn corpus_with_no_ambiguity_reports_nan_not_a_misleading_number() {
        let deterministic = vec![vec![1u32, 2, 3], vec![1, 2, 3]];
        let s = score(&deterministic, |seq, t| deterministic[seq][t + 1]);
        assert!(
            s.ambiguous_accuracy.is_nan(),
            "no ambiguous points to score"
        );
        assert_eq!(s.global_accuracy, 1.0);
    }
}

/// Score a **timed** corpus per §6.
///
/// Ambiguity is still determined from the token view — a point is ambiguous when
/// its current token has multiple successors — because that is what "history is
/// the only discriminator" means regardless of whether the discriminator is
/// order or elapsed time. What differs is that the arm may consult the timing:
/// `predict` receives `(sequence_index, position)` and can read intervals from
/// the corpus it was handed.
///
/// Exists because `score` could not reach [`TimedItem`] corpora at all, which
/// made the entire §5.2 family unscoreable — a gap invisible while each
/// component was only unit-tested in isolation.
pub fn score_timed<F>(sequences: &[Vec<TimedItem>], predict: F) -> AmbiguousScore
where
    F: FnMut(usize, usize) -> TokenId,
{
    let token_view: Vec<Vec<TokenId>> = sequences
        .iter()
        .map(|s| s.iter().map(|i| i.token).collect())
        .collect();
    score(&token_view, predict)
}
