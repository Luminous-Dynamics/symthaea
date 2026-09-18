// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Atomic coordination between exact dream feedback and semantic prior memory.
//!
//! The coordinator stages both updates on clones and commits them together only
//! after the semantic sidecar accepts the resulting exact action prior. This keeps
//! exact-prior state and semantic-prior state synchronized without changing the
//! serialized `DreamFeedbackBridge` layout.
//!
//! This layer still grants no confidence or evidence authority. Public confidence
//! behavior remains governed by `DreamFeedbackBridge`/`DreamConfidenceGate`.

use super::dream_feedback::{DreamFeedbackBridge, DreamInsight};
use super::semantic_context::{ContextIdentity, SemanticContextError};
use super::semantic_prior::{SemanticPriorError, SemanticPriorMemory};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamSemanticCoordinatorError {
    Context(SemanticContextError),
    ContextHashMismatch { expected: u64, observed: u64 },
    MissingPriorAfterAcceptance,
    SemanticPrior(SemanticPriorError),
}

impl From<SemanticContextError> for DreamSemanticCoordinatorError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

impl From<SemanticPriorError> for DreamSemanticCoordinatorError {
    fn from(value: SemanticPriorError) -> Self {
        Self::SemanticPrior(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamSemanticOutcome {
    /// The dream bridge rejected the generated insight under its existing policy.
    RejectedByDreamPolicy,
    /// Both exact and semantic prior state committed atomically.
    Indexed { identity: ContextIdentity },
}

/// Process one dream insight and, if accepted, index the resulting action prior
/// under the collision-resistant semantic context identity.
///
/// Accepted prior state is committed only after semantic indexing succeeds. A
/// bridge-policy rejection preserves the bridge's existing rejection telemetry but
/// creates no exact or semantic prior. Any semantic indexing error after acceptance
/// rolls back the staged bridge update as well.
pub fn process_semantic_insight_atomically(
    bridge: &mut DreamFeedbackBridge,
    semantic_memory: &mut SemanticPriorMemory,
    context: &[f32],
    insight: DreamInsight,
) -> Result<DreamSemanticOutcome, DreamSemanticCoordinatorError> {
    let identity = ContextIdentity::from_context(context)?;
    if identity.fast_hash != insight.context_hash {
        return Err(DreamSemanticCoordinatorError::ContextHashMismatch {
            expected: insight.context_hash,
            observed: identity.fast_hash,
        });
    }

    let mut staged_bridge = bridge.clone();
    let mut staged_memory = semantic_memory.clone();

    if !staged_bridge.process_insight(insight) {
        // Preserve the legacy bridge's accounting of rejected generated insights,
        // while still guaranteeing that no prior/semantic state was created.
        *bridge = staged_bridge;
        return Ok(DreamSemanticOutcome::RejectedByDreamPolicy);
    }

    let prior = staged_bridge
        .get_prior(identity.fast_hash)
        .ok_or(DreamSemanticCoordinatorError::MissingPriorAfterAcceptance)?;
    let indexed_identity = staged_memory.index_prior(context, prior)?;
    debug_assert_eq!(identity, indexed_identity);

    *bridge = staged_bridge;
    *semantic_memory = staged_memory;

    Ok(DreamSemanticOutcome::Indexed { identity })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insight(context: &[f32], alternative: Vec<f32>, phi_improvement: f64) -> DreamInsight {
        DreamInsight::new(
            super::super::dream_feedback::hash_context(context),
            vec![0.1, 0.2, 0.3],
            alternative,
            phi_improvement,
        )
    }

    #[test]
    fn accepted_insight_commits_exact_and_semantic_state_together() {
        let context = [0.2, -0.1, 0.7];
        let mut bridge = DreamFeedbackBridge::new();
        let mut semantic_memory = SemanticPriorMemory::new();

        let outcome = process_semantic_insight_atomically(
            &mut bridge,
            &mut semantic_memory,
            &context,
            insight(&context, vec![0.4, 0.5, 0.6], 0.4),
        )
        .unwrap();

        let DreamSemanticOutcome::Indexed { identity } = outcome else {
            panic!("accepted insight was not indexed");
        };
        assert!(bridge.get_prior(identity.fast_hash).is_some());
        assert!(semantic_memory.contains_exact(identity));
        assert_eq!(semantic_memory.len(), 1);
    }

    #[test]
    fn dream_policy_rejection_preserves_telemetry_but_creates_no_prior() {
        let context = [0.2, -0.1, 0.7];
        let mut bridge = DreamFeedbackBridge::new();
        let mut semantic_memory = SemanticPriorMemory::new();

        let outcome = process_semantic_insight_atomically(
            &mut bridge,
            &mut semantic_memory,
            &context,
            insight(&context, vec![0.4, 0.5, 0.6], 0.001),
        )
        .unwrap();

        assert_eq!(outcome, DreamSemanticOutcome::RejectedByDreamPolicy);
        assert_eq!(bridge.num_priors(), 0);
        assert_eq!(bridge.stats().total_insights, 1);
        assert!(semantic_memory.is_empty());
    }

    #[test]
    fn semantic_index_failure_rolls_back_exact_prior_mutation() {
        let context = [0.2, -0.1, 0.7];
        let mut bridge = DreamFeedbackBridge::new();
        let mut semantic_memory = SemanticPriorMemory::new();

        let result = process_semantic_insight_atomically(
            &mut bridge,
            &mut semantic_memory,
            &context,
            insight(&context, Vec::new(), 0.4),
        );

        assert_eq!(
            result,
            Err(DreamSemanticCoordinatorError::SemanticPrior(
                SemanticPriorError::EmptyPreferredDirection
            ))
        );
        assert_eq!(bridge.num_priors(), 0);
        assert_eq!(bridge.stats().total_insights, 0);
        assert!(semantic_memory.is_empty());
    }

    #[test]
    fn mismatched_context_identity_fails_before_any_mutation() {
        let context = [0.2, -0.1, 0.7];
        let wrong_context = [0.21, -0.1, 0.7];
        let mut bridge = DreamFeedbackBridge::new();
        let mut semantic_memory = SemanticPriorMemory::new();

        let result = process_semantic_insight_atomically(
            &mut bridge,
            &mut semantic_memory,
            &wrong_context,
            insight(&context, vec![0.4, 0.5, 0.6], 0.4),
        );

        assert!(matches!(
            result,
            Err(DreamSemanticCoordinatorError::ContextHashMismatch { .. })
        ));
        assert_eq!(bridge.num_priors(), 0);
        assert_eq!(bridge.stats().total_insights, 0);
        assert!(semantic_memory.is_empty());
    }

    #[test]
    fn semantic_indexing_does_not_bypass_public_confidence_boundary() {
        let context = [0.2, -0.1, 0.7];
        let context_hash = super::super::dream_feedback::hash_context(&context);
        let mut bridge = DreamFeedbackBridge::new();
        let mut semantic_memory = SemanticPriorMemory::new();

        process_semantic_insight_atomically(
            &mut bridge,
            &mut semantic_memory,
            &context,
            insight(&context, vec![0.4, 0.5, 0.6], 0.8),
        )
        .unwrap();

        let (adjusted, informed) = bridge.adjust_confidence(0.7, context_hash);
        assert!(informed);
        assert!(adjusted <= 0.7);

        let candidates = semantic_memory
            .retrieve_unfiltered(&[0.25, -0.1, 0.7], 1)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].effective_search_strength <= candidates[0].prior_strength);
    }
}
