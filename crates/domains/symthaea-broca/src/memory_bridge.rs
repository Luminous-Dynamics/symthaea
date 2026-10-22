// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Memory Bridge — HDC Store retrieval and blending.
//!
//! Allows Broca to query the long-term narrative self and blend past experiences
//! into the current thought stream.

use anyhow::Result;
use symthaea_core::hdc::ContinuousHV;
use symthaea_hdc_store::store::HdcStore;

/// Experimental policy for deciding whether retrieved memory should alter the
/// current thought. The default bridge method continues to use `Always`.
///
/// Similarity thresholds are retrieval-confidence proxies, not estimates of
/// downstream task utility. They require calibration on development data and
/// evaluation on a held-out task split before they can be recommended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryInterventionPolicy {
    /// Preserve the legacy behavior: blend whenever at least one payload loads.
    Always,
    /// Do not query or blend persistent memory.
    Never,
    /// Blend only if the best loaded neighbor's similarity clears the threshold.
    MinTopSimilarity {
        /// Inclusive similarity threshold in `[0, 1]`.
        min_similarity: f32,
    },
    /// Require both an absolute score and a separation from the runner-up.
    ///
    /// Fails closed if fewer than two payloads load: the margin is unknown.
    MinTopSimilarityAndMargin {
        /// Inclusive top-neighbor similarity threshold in `[0, 1]`.
        min_similarity: f32,
        /// Inclusive top-minus-runner-up threshold in `[0, 1]`.
        min_margin: f32,
    },
}

impl MemoryInterventionPolicy {
    fn validate(self) -> Result<()> {
        let (min_similarity, min_margin) = match self {
            Self::MinTopSimilarity { min_similarity } => (Some(min_similarity), None),
            Self::MinTopSimilarityAndMargin {
                min_similarity,
                min_margin,
            } => (Some(min_similarity), Some(min_margin)),
            Self::Always | Self::Never => (None, None),
        };

        for (name, value) in [
            ("min_similarity", min_similarity),
            ("min_margin", min_margin),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    bail!("{name} must be finite and in [0, 1]");
                }
            }
        }
        Ok(())
    }

    fn should_intervene(self, scores: &[f32]) -> Result<bool> {
        match self {
            Self::Always => Ok(!scores.is_empty()),
            Self::Never => Ok(false),
            Self::MinTopSimilarity { min_similarity } => {
                if scores.is_empty() {
                    return Ok(false);
                }
                let (top, _) = validated_top_two(scores)?;
                Ok(top >= min_similarity)
            }
            Self::MinTopSimilarityAndMargin {
                min_similarity,
                min_margin,
            } => {
                if scores.is_empty() {
                    return Ok(false);
                }
                let (top, second) = validated_top_two(scores)?;
                if scores.len() < 2 {
                    return Ok(false);
                }
                Ok(top >= min_similarity && top - second >= min_margin)
            }
        }
    }
}

/// Returns the two largest scores, independent of incoming order. This helps
/// make the gate robust to callers that do not preserve rank ordering.
fn validated_top_two(scores: &[f32]) -> Result<(f32, f32)> {
    let mut top = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &score in scores {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("retrieval similarity must be finite and in [0, 1]");
        }
        if score > top {
            second = top;
            top = score;
        } else if score > second {
            second = score;
        }
    }
    if !top.is_finite() {
        bail!("no finite retrieval score");
    }
    Ok((top, second))
}

/// Bridge between Broca and the long-term HDC store.
///
/// Not `Clone`: `HdcStore` is a zero-copy mmap with an exclusive advisory
/// file lock ("one exclusive mutable opener") — cloning it would violate the
/// store's single-writer invariant. The old `#[derive(Clone)]` here only ever
/// compiled against a since-changed on-disk `symthaea-hdc-store` (that crate
/// is referenced by committed Cargo.tomls but has NO git history — a
/// never-committed-crate repo-integrity gap, found 2026-07-16 when a
/// workspace build first exercised this path).
pub struct MemoryBridge {
    pub store: HdcStore,
    /// Number of past experiences to retrieve and blend.
    pub top_k: usize,
    /// Blending factor (0.0 = only current, 1.0 = only past).
    pub blend_alpha: f32,
}

impl MemoryBridge {
    pub fn new(store: HdcStore, top_k: usize, blend_alpha: f32) -> Self {
        Self {
            store,
            top_k,
            blend_alpha,
        }
    }

    /// Retrieve similar past thoughts and blend them into the current thought.
    ///
    /// This preserves the legacy always-blend behavior. Use
    /// `blend_past_experiences_with_policy` to compare opt-in gates.
    pub fn blend_past_experiences(&self, current_thought: &mut ContinuousHV) -> Result<usize> {
        self.blend_past_experiences_with_policy(current_thought, MemoryInterventionPolicy::Always)
    }

    /// Retrieve and optionally blend memories under an explicit intervention policy.
    ///
    /// The score gates are experimental proxies only. They must be calibrated on
    /// a development split and evaluated on held-out task outcomes; retrieval
    /// similarity alone is not proof that a memory improves the task.
    ///
    /// Returns the number of memories blended. Returns zero when the policy
    /// abstains, no neighbors are available, or no retrieved payload can be loaded.
    pub fn blend_past_experiences_with_policy(
        &self,
        current_thought: &mut ContinuousHV,
        policy: MemoryInterventionPolicy,
    ) -> Result<usize> {
        policy.validate()?;
        if matches!(policy, MemoryInterventionPolicy::Never) {
            return Ok(0);
        }

        // 1. Convert to binary for LSH search.
        let binary_query = current_thought.to_binary(0.0);

        // 2. Query store for top-K similar vectors. The exact search returns
        // similarity-ranked entries; retain those scores for the optional gate.
        let similar = self.store.scan_similar(&binary_query, self.top_k);
        if similar.is_empty() {
            return Ok(0);
        }

        // 3. Fetch payloads and keep the score associated with each successful
        // fetch. A missing payload must not donate its score to a different item.
        let mut retrieved_hvs = Vec::with_capacity(similar.len());
        let mut retrieved_scores = Vec::with_capacity(similar.len());
        for (id, score) in &similar {
            if let Some(bhv) = self.store.get(*id) {
                retrieved_hvs.push(ContinuousHV::from_vec(bhv.to_bipolar()));
                retrieved_scores.push(*score);
            }
        }

        if retrieved_hvs.is_empty() {
            return Ok(0);
        }
        if !policy.should_intervene(&retrieved_scores)? {
            return Ok(0);
        }

        // 4. Bundle past experiences into a single "memory vector".
        // Permutation index reflects retrieval rank only, not chronology; the
        // store exposes caller-provided IDs but no timestamp contract.
        let mut bundled_values = vec![0.0f32; current_thought.dim()];
        let n = retrieved_hvs.len() as f32;
        for (rank, hv) in retrieved_hvs.iter().enumerate() {
            let shifted = hv.permute(rank);
            for (b, &s) in bundled_values.iter_mut().zip(shifted.as_slice().iter()) {
                *b += s / n;
            }
        }
        let memory_hv = ContinuousHV::from_vec(bundled_values);

        // 5. Legacy holographic-resonance blending heuristic. This remains
        // unchanged for controlled A/B comparison; resonance is not a task-utility
        // estimate and does not replace held-out evaluation of the intervention.
        let resonance = current_thought.similarity(&memory_hv).max(0.1);
        let dynamic_alpha = (self.blend_alpha * (1.0 + resonance)).clamp(0.05, 0.95);
        current_thought.lerp_in_place(&memory_hv, 1.0 - dynamic_alpha, dynamic_alpha);

        Ok(retrieved_hvs.len())
    }

    /// Append a new experience to long-term memory.
    pub fn remember(&mut self, id: u64, thought: &ContinuousHV) -> Result<()> {
        let binary = thought.to_binary(0.0);
        self.store
            .append(id, &binary)
            .map_err(|e| anyhow::anyhow!("Store append failed: {}", e))
    }
}


#[cfg(test)]
mod intervention_policy_tests {
    use super::*;

    #[test]
    fn always_policy_preserves_nonempty_legacy_gate_semantics() {
        assert!(MemoryInterventionPolicy::Always
            .should_intervene(&[0.05])
            .unwrap());
        assert!(!MemoryInterventionPolicy::Always
            .should_intervene(&[])
            .unwrap());
    }

    #[test]
    fn never_policy_never_intervenes_even_on_perfect_similarity() {
        assert!(!MemoryInterventionPolicy::Never
            .should_intervene(&[1.0, 0.99])
            .unwrap());
        assert!(!MemoryInterventionPolicy::Never
            .should_intervene(&[])
            .unwrap());
    }

    #[test]
    fn top_similarity_gate_obeys_inclusive_threshold() {
        let policy = MemoryInterventionPolicy::MinTopSimilarity {
            min_similarity: 0.8,
        };
        assert!(policy.should_intervene(&[0.8, 0.7]).unwrap());
        assert!(!policy.should_intervene(&[0.79, 0.7]).unwrap());
        assert!(!policy.should_intervene(&[]).unwrap());
    }

    #[test]
    fn score_margin_gate_rejects_ambiguous_neighbors() {
        let policy = MemoryInterventionPolicy::MinTopSimilarityAndMargin {
            min_similarity: 0.8,
            min_margin: 0.1,
        };
        assert!(!policy.should_intervene(&[0.91, 0.89]).unwrap());
        assert!(policy.should_intervene(&[0.91, 0.70]).unwrap());
    }

    #[test]
    fn score_margin_gate_fails_closed_with_one_neighbor() {
        let policy = MemoryInterventionPolicy::MinTopSimilarityAndMargin {
            min_similarity: 0.5,
            min_margin: 0.1,
        };
        assert!(!policy.should_intervene(&[0.99]).unwrap());
    }

    #[test]
    fn gate_computes_top_two_without_assuming_input_order() {
        let policy = MemoryInterventionPolicy::MinTopSimilarityAndMargin {
            min_similarity: 0.8,
            min_margin: 0.2,
        };
        assert!(policy.should_intervene(&[0.55, 0.95, 0.6]).unwrap());
    }

    #[test]
    fn invalid_thresholds_and_scores_are_rejected() {
        assert!(MemoryInterventionPolicy::MinTopSimilarity {
            min_similarity: f32::NAN,
        }
        .validate()
        .is_err());
        assert!(MemoryInterventionPolicy::MinTopSimilarityAndMargin {
            min_similarity: 0.5,
            min_margin: 1.1,
        }
        .validate()
        .is_err());
        assert!(MemoryInterventionPolicy::MinTopSimilarity {
            min_similarity: 0.5,
        }
        .should_intervene(&[f32::INFINITY])
        .is_err());
        assert!(MemoryInterventionPolicy::MinTopSimilarityAndMargin {
            min_similarity: 0.5,
            min_margin: 0.1,
        }
        .should_intervene(&[f32::NAN])
        .is_err());
    }
}
