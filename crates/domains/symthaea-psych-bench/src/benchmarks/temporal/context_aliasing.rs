// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Context-aliasing sequence task — §5.1 of `SYMTHAEA_TEMPORAL_BENCHMARK_V2_PLAN.md`.
//!
//! ```text
//! A → B → X → C
//! D → E → X → F
//! ```
//!
//! At `X` the current input is **identical** across branches, so a zero-history
//! arm cannot resolve which target is correct — it is at chance by construction,
//! which is what makes it a mechanically guaranteed negative control rather than
//! a hopefully-weak baseline. A history-bearing arm must retain the cue.
//!
//! # Every corpus is validated before it is returned
//!
//! [`generate`] runs the produced corpus through
//! [`symthaea_evidence_plane::task_validator`] and returns `Err` if it does not
//! provably require memory. That is not belt-and-braces: the predecessor
//! benchmark's corpus was *asserted* to require history and silently did not,
//! costing an entire measurement arc that was later retracted. A generator that
//! cannot pass its own gate must not be usable.
//!
//! # Difficulty is separated from length
//!
//! `distractors` inserts filler between the cue and the decision point, raising
//! the required context depth without changing the task's logical structure.
//! The filler is deliberately **shared across branches** — branch-specific
//! filler would leak the cue's identity into every intervening position and
//! silently reduce the depth actually required, which is exactly the class of
//! self-deception this benchmark exists to avoid. [`filler_does_not_leak_the_cue`]
//! pins that.

use symthaea_evidence_plane::task_validator::{self, TaskMetrics, TaskViolation, TokenId};

/// Token reserved for the aliased decision point, identical in every branch.
pub const ALIAS_TOKEN: TokenId = 9_000;
/// Filler tokens occupy this range, shared across all branches.
const FILLER_BASE: TokenId = 8_000;
/// Cue tokens occupy this range, one per branch.
const CUE_BASE: TokenId = 1_000;
/// Target tokens occupy this range, one per branch.
const TARGET_BASE: TokenId = 2_000;

/// Configuration for a context-aliasing corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextAliasingConfig {
    /// Distinct cue→target branches. Must be ≥ 2 or there is no ambiguity.
    pub branches: usize,
    /// Filler tokens between the cue and the aliased decision point. The
    /// required context depth is `distractors + 1`.
    pub distractors: usize,
    /// Sequences generated per branch.
    pub repeats: usize,
}

impl Default for ContextAliasingConfig {
    fn default() -> Self {
        Self {
            branches: 2,
            distractors: 0,
            repeats: 32,
        }
    }
}

impl ContextAliasingConfig {
    /// Context depth a model must retain to solve this task: the cue sits
    /// `distractors + 1` positions before the aliased decision point.
    pub fn required_depth(&self) -> usize {
        self.distractors + 1
    }
}

/// A validated context-aliasing corpus.
#[derive(Debug, Clone)]
pub struct ContextAliasingTask {
    pub sequences: Vec<Vec<TokenId>>,
    pub config: ContextAliasingConfig,
    /// Metrics from the validator, at [`ContextAliasingConfig::required_depth`].
    pub metrics: TaskMetrics,
}

/// Why a corpus could not be produced.
#[derive(Debug, Clone)]
pub enum GenerateError {
    /// Fewer than two branches: nothing to disambiguate.
    DegenerateConfig { branches: usize },
    /// The generated corpus failed its own validation gate.
    ValidationFailed {
        metrics: Box<TaskMetrics>,
        violations: Vec<TaskViolation>,
    },
}

/// Build the raw sequences without validating them.
///
/// Exposed so tests can construct deliberately-broken corpora and confirm the
/// gate rejects them. Production callers should use [`generate`].
pub fn generate_unvalidated(config: ContextAliasingConfig) -> Vec<Vec<TokenId>> {
    let mut sequences = Vec::with_capacity(config.branches * config.repeats);
    for rep in 0..config.repeats {
        for branch in 0..config.branches {
            let mut seq = Vec::with_capacity(config.distractors + 3);
            seq.push(CUE_BASE + branch as TokenId);
            // Filler is shared across branches: index by position, never by
            // branch, so it carries no information about which cue preceded it.
            for d in 0..config.distractors {
                seq.push(FILLER_BASE + d as TokenId);
            }
            seq.push(ALIAS_TOKEN);
            seq.push(TARGET_BASE + branch as TokenId);
            sequences.push(seq);
        }
        let _ = rep;
    }
    sequences
}

/// Build a context-aliasing corpus and prove it requires memory before
/// returning it.
pub fn generate(config: ContextAliasingConfig) -> Result<ContextAliasingTask, GenerateError> {
    if config.branches < 2 {
        return Err(GenerateError::DegenerateConfig {
            branches: config.branches,
        });
    }
    let sequences = generate_unvalidated(config);
    match task_validator::validate(&sequences, config.required_depth()) {
        Ok(metrics) => Ok(ContextAliasingTask {
            sequences,
            config,
            metrics,
        }),
        Err((metrics, violations)) => Err(GenerateError::ValidationFailed {
            metrics: Box::new(metrics),
            violations,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_task_passes_its_own_gate() {
        let task = generate(ContextAliasingConfig::default()).expect("must validate");
        assert!(task.metrics.conditional_mutual_information > 0.0);
        assert!(task.metrics.current_only_oracle_accuracy < 1.0);
        assert!(task.metrics.history_oracle_accuracy > task.metrics.current_only_oracle_accuracy);
    }

    /// The load-bearing property: the corpus must be UNSOLVABLE at less than the
    /// required depth and solvable at or above it. A generator that validates at
    /// every depth is not controlling difficulty, it is only claiming to.
    #[test]
    fn required_depth_is_actually_required() {
        for distractors in [1_usize, 2, 4] {
            let config = ContextAliasingConfig {
                branches: 2,
                distractors,
                repeats: 48,
            };
            let required = config.required_depth();
            let sequences = generate_unvalidated(config);

            for insufficient in 0..required {
                assert!(
                    task_validator::validate(&sequences, insufficient).is_err(),
                    "distractors={distractors}: corpus must NOT validate at depth \
                     {insufficient} < required {required}"
                );
            }
            assert!(
                task_validator::validate(&sequences, required).is_ok(),
                "distractors={distractors}: corpus must validate at its required depth \
                 {required}"
            );
        }
    }

    /// Difficulty must be separable from raw sequence length. Two corpora with
    /// the same number of branches but different distractor counts must differ
    /// in required depth while remaining valid tasks.
    #[test]
    fn distractors_raise_depth_without_breaking_the_task() {
        let shallow = generate(ContextAliasingConfig {
            distractors: 0,
            ..Default::default()
        })
        .expect("shallow validates");
        let deep = generate(ContextAliasingConfig {
            distractors: 4,
            ..Default::default()
        })
        .expect("deep validates");

        assert_eq!(shallow.config.required_depth(), 1);
        assert_eq!(deep.config.required_depth(), 5);
        assert!(deep.sequences[0].len() > shallow.sequences[0].len());
    }

    /// Filler must not leak the cue. If it did, the intervening positions would
    /// each identify the branch and the task would be solvable at depth 1
    /// regardless of `distractors` — the required depth would be a fiction.
    #[test]
    fn filler_does_not_leak_the_cue() {
        let config = ContextAliasingConfig {
            branches: 2,
            distractors: 3,
            repeats: 48,
        };
        let sequences = generate_unvalidated(config);

        // Every branch must use an identical filler run.
        let filler_of = |seq: &Vec<TokenId>| seq[1..1 + config.distractors].to_vec();
        let first = filler_of(&sequences[0]);
        for seq in &sequences {
            assert_eq!(
                filler_of(seq),
                first,
                "filler must be shared across branches, not branch-specific"
            );
        }
        // And the corpus must genuinely fail below the required depth, which is
        // the observable consequence of the filler carrying no cue information.
        assert!(task_validator::validate(&sequences, 1).is_err());
    }

    /// A branch-specific-filler corpus is the failure mode above, made concrete.
    /// It must be rejected at the shallow depth it would otherwise fake.
    #[test]
    fn leaky_filler_corpus_is_detectably_easier() {
        let mut leaky = Vec::new();
        for _ in 0..48 {
            for branch in 0..2u32 {
                // Filler encodes the branch — the leak.
                leaky.push(vec![
                    CUE_BASE + branch,
                    FILLER_BASE + 100 + branch,
                    ALIAS_TOKEN,
                    TARGET_BASE + branch,
                ]);
            }
        }
        // With a leak, depth 1 already resolves the alias, so a depth-2 claim
        // would be false. Confirm the leak is visible: the corpus validates at
        // depth 1, which the honest generator's corpus does not.
        assert!(
            task_validator::validate(&leaky, 1).is_ok(),
            "leaky corpus is solvable at depth 1 — this is the trap being guarded"
        );
        let honest = generate_unvalidated(ContextAliasingConfig {
            branches: 2,
            distractors: 1,
            repeats: 48,
        });
        assert!(
            task_validator::validate(&honest, 1).is_err(),
            "honest corpus must NOT be solvable at depth 1"
        );
    }

    #[test]
    fn single_branch_is_rejected_as_degenerate() {
        let err = generate(ContextAliasingConfig {
            branches: 1,
            ..Default::default()
        })
        .expect_err("one branch has nothing to disambiguate");
        assert!(matches!(
            err,
            GenerateError::DegenerateConfig { branches: 1 }
        ));
    }
}
