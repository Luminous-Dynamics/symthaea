// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hardened operational façade for integrity-bound semantic prior memory.
//!
//! This façade deliberately owns the v2 memory privately. It adds three operational
//! guarantees without changing the underlying diagnostic/raw APIs:
//!
//! - every direct indexing mutation is staged on a clone and committed only after
//!   the v2 index operation succeeds;
//! - clamp-saturated queries fail explicitly with their bound support receipt rather
//!   than being indistinguishable from a legitimate zero-match result;
//! - generated dream feedback and operational semantic state can commit atomically,
//!   so exact priors cannot diverge from their support-bound semantic provenance.
//!
//! These guarantees remain retrieval/integrity properties only. They do not grant
//! empirical evidence, confidence, or validation authority.

use super::dream_feedback::{ActionPrior, DreamFeedbackBridge, DreamInsight};
use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};
use super::semantic_null_calibration::SemanticNullCalibration;
use super::semantic_support::{
    IntegrityBoundCalibratedSemanticPriorCandidate, IntegrityBoundSemanticPriorCandidate,
    IntegrityBoundSemanticPriorError, IntegrityBoundSemanticPriorMemory, SemanticSupportBinding,
    SemanticSupportBindingError,
};

#[derive(Debug, Clone, PartialEq)]
pub enum OperationalSemanticPriorError {
    Binding(SemanticSupportBindingError),
    Inner(IntegrityBoundSemanticPriorError),
    QueryOutsideNominalSupport { binding: SemanticSupportBinding },
}

impl From<SemanticSupportBindingError> for OperationalSemanticPriorError {
    fn from(value: SemanticSupportBindingError) -> Self {
        Self::Binding(value)
    }
}

impl From<IntegrityBoundSemanticPriorError> for OperationalSemanticPriorError {
    fn from(value: IntegrityBoundSemanticPriorError) -> Self {
        Self::Inner(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperationalDreamSemanticError {
    Context(SemanticContextError),
    ContextHashMismatch { expected: u64, observed: u64 },
    MissingPriorAfterAcceptance,
    Memory(OperationalSemanticPriorError),
}

impl From<SemanticContextError> for OperationalDreamSemanticError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

impl From<OperationalSemanticPriorError> for OperationalDreamSemanticError {
    fn from(value: OperationalSemanticPriorError) -> Self {
        Self::Memory(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalDreamSemanticOutcome {
    RejectedByDreamPolicy,
    Indexed { identity: ContextIdentity },
}

/// Runtime-only fail-closed semantic memory for live use.
///
/// The wrapped v2 memory has no mutable accessor. Operational callers therefore
/// cannot bypass clone-before-index transactionality or explicit query-support
/// admission through this capability surface.
#[derive(Clone, Default)]
pub struct OperationalSemanticPriorMemory {
    inner: IntegrityBoundSemanticPriorMemory,
}

impl OperationalSemanticPriorMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn context_encoder(&self) -> SemanticContextEncoder {
        self.inner.context_encoder()
    }

    pub fn source_binding(&self, identity: ContextIdentity) -> Option<SemanticSupportBinding> {
        self.inner.source_binding(identity)
    }

    /// Transactionally index one prior plus its integrity-bound source-support receipt.
    ///
    /// Even if the wrapped implementation ever develops an internal failure after
    /// partial mutation, this façade's original state remains untouched because all
    /// work is staged on a clone and committed only after success.
    pub fn index_prior(
        &mut self,
        context: &[f32],
        prior: &ActionPrior,
    ) -> Result<ContextIdentity, OperationalSemanticPriorError> {
        let mut staged = self.inner.clone();
        let identity = staged.index_prior(context, prior)?;
        self.inner = staged;
        Ok(identity)
    }

    pub fn retrieve(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<IntegrityBoundSemanticPriorCandidate>, OperationalSemanticPriorError> {
        let binding = self.require_supported_query(query)?;
        let candidates = self
            .inner
            .retrieve_integrity_bound(query, min_similarity, limit)?;
        self.verify_query_bindings(
            &binding,
            candidates.iter().map(|candidate| candidate.query_binding),
        )?;
        Ok(candidates)
    }

    pub fn retrieve_calibrated(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<Vec<IntegrityBoundCalibratedSemanticPriorCandidate>, OperationalSemanticPriorError>
    {
        let binding = self.require_supported_query(query)?;
        let candidates = self.inner.retrieve_calibrated_integrity_bound(
            query,
            min_similarity,
            limit,
            calibration,
        )?;
        self.verify_query_bindings(
            &binding,
            candidates.iter().map(|candidate| candidate.query_binding),
        )?;
        Ok(candidates)
    }

    fn require_supported_query(
        &self,
        query: &[f32],
    ) -> Result<SemanticSupportBinding, OperationalSemanticPriorError> {
        let binding = SemanticSupportBinding::create(self.context_encoder(), query)?;
        if !binding.within_nominal_support() {
            return Err(OperationalSemanticPriorError::QueryOutsideNominalSupport {
                binding,
            });
        }
        Ok(binding)
    }

    fn verify_query_bindings(
        &self,
        expected: &SemanticSupportBinding,
        observed: impl Iterator<Item = SemanticSupportBinding>,
    ) -> Result<(), OperationalSemanticPriorError> {
        for binding in observed {
            if binding != *expected {
                return Err(OperationalSemanticPriorError::Inner(
                    IntegrityBoundSemanticPriorError::BindingDrift {
                        identity: binding.context_identity,
                    },
                ));
            }
        }
        Ok(())
    }
}

/// Atomically process one generated dream through exact feedback and the hardened
/// operational semantic memory.
///
/// A policy rejection preserves only the bridge's legacy rejection telemetry. An
/// accepted insight commits the exact action prior and its collision-resistant,
/// support-bound semantic provenance together. Any semantic-memory failure rolls
/// back the staged bridge mutation as well.
pub fn process_operational_dream_insight_atomically(
    bridge: &mut DreamFeedbackBridge,
    memory: &mut OperationalSemanticPriorMemory,
    context: &[f32],
    insight: DreamInsight,
) -> Result<OperationalDreamSemanticOutcome, OperationalDreamSemanticError> {
    let identity = ContextIdentity::from_context(context)?;
    if identity.fast_hash != insight.context_hash {
        return Err(OperationalDreamSemanticError::ContextHashMismatch {
            expected: insight.context_hash,
            observed: identity.fast_hash,
        });
    }

    let mut staged_bridge = bridge.clone();
    let mut staged_memory = memory.clone();

    if !staged_bridge.process_insight(insight) {
        *bridge = staged_bridge;
        return Ok(OperationalDreamSemanticOutcome::RejectedByDreamPolicy);
    }

    let prior = staged_bridge
        .get_prior(identity.fast_hash)
        .ok_or(OperationalDreamSemanticError::MissingPriorAfterAcceptance)?;
    let indexed_identity = staged_memory.index_prior(context, prior)?;
    if indexed_identity != identity {
        return Err(OperationalDreamSemanticError::Memory(
            OperationalSemanticPriorError::Inner(
                IntegrityBoundSemanticPriorError::BindingDrift {
                    identity: indexed_identity,
                },
            ),
        ));
    }

    *bridge = staged_bridge;
    *memory = staged_memory;

    Ok(OperationalDreamSemanticOutcome::Indexed { identity })
}

#[cfg(test)]
mod tests {
    use super::super::dream_feedback::hash_context;
    use super::super::semantic_null_calibration::SemanticNullConfig;
    use super::super::semantic_prior::SemanticPriorError;
    use super::super::semantic_supported_retrieval::SemanticQuerySupportError;
    use super::*;

    fn prior(context: &[f32], direction: Vec<f32>) -> ActionPrior {
        ActionPrior {
            context_hash: hash_context(context),
            preferred_direction: direction,
            strength: 0.8,
            evidence_count: 1,
        }
    }

    fn insight(context: &[f32], direction: Vec<f32>, phi_improvement: f64) -> DreamInsight {
        DreamInsight::new(
            hash_context(context),
            context.to_vec(),
            direction,
            phi_improvement,
        )
    }

    fn calibration(memory: &OperationalSemanticPriorMemory) -> SemanticNullCalibration {
        let references = (0..8)
            .map(|index| vec![-0.9 + index as f32 * 0.25])
            .collect::<Vec<_>>();
        SemanticNullCalibration::build(
            &references,
            memory.context_encoder(),
            SemanticNullConfig {
                max_reference_contexts_per_dimension: 8,
                min_null_pairs_per_dimension: 3,
            },
        )
        .unwrap()
    }

    #[test]
    fn successful_index_commits_bound_source_state() {
        let context = [0.2];
        let mut memory = OperationalSemanticPriorMemory::new();
        let identity = memory
            .index_prior(&context, &prior(&context, vec![0.3]))
            .unwrap();

        assert_eq!(memory.len(), 1);
        let binding = memory.source_binding(identity).unwrap();
        binding
            .validate_against_context(memory.context_encoder(), &context)
            .unwrap();
    }

    #[test]
    fn failed_index_keeps_original_memory_unchanged() {
        let context = [0.2];
        let mut memory = OperationalSemanticPriorMemory::new();
        memory
            .index_prior(&context, &prior(&context, vec![0.3]))
            .unwrap();
        let len_before = memory.len();

        let bad_context = [0.4];
        let error = memory
            .index_prior(&bad_context, &prior(&bad_context, Vec::new()))
            .unwrap_err();

        assert_eq!(memory.len(), len_before);
        assert!(matches!(
            error,
            OperationalSemanticPriorError::Inner(
                IntegrityBoundSemanticPriorError::Retrieval(
                    SemanticQuerySupportError::Prior(SemanticPriorError::EmptyPreferredDirection)
                )
            )
        ));
    }

    #[test]
    fn unsupported_query_is_explicit_not_empty_match() {
        let context = [0.2];
        let mut memory = OperationalSemanticPriorMemory::new();
        memory
            .index_prior(&context, &prior(&context, vec![0.3]))
            .unwrap();

        let error = memory.retrieve(&[2.5], 0.0, 4).unwrap_err();
        match error {
            OperationalSemanticPriorError::QueryOutsideNominalSupport { binding } => {
                assert!(!binding.within_nominal_support());
                assert_eq!(binding.support.clipped_value_count, 1);
                binding.validate_self().unwrap();
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn supported_query_preserves_exact_bound_query_receipt() {
        let context = [0.2];
        let mut memory = OperationalSemanticPriorMemory::new();
        memory
            .index_prior(&context, &prior(&context, vec![0.3]))
            .unwrap();

        let candidates = memory.retrieve(&context, 0.0, 4).unwrap();
        assert_eq!(candidates.len(), 1);
        candidates[0]
            .query_binding
            .validate_against_context(memory.context_encoder(), &context)
            .unwrap();
    }

    #[test]
    fn calibrated_path_uses_same_explicit_query_admission() {
        let context = [0.2];
        let mut memory = OperationalSemanticPriorMemory::new();
        memory
            .index_prior(&context, &prior(&context, vec![0.3]))
            .unwrap();
        let calibration = calibration(&memory);

        let supported = memory
            .retrieve_calibrated(&context, 0.0, 4, &calibration)
            .unwrap();
        assert_eq!(supported.len(), 1);

        assert!(matches!(
            memory.retrieve_calibrated(&[2.5], 0.0, 4, &calibration),
            Err(OperationalSemanticPriorError::QueryOutsideNominalSupport { .. })
        ));
    }

    #[test]
    fn operational_dream_commit_binds_exact_prior_and_source_support() {
        let context = [0.2];
        let mut bridge = DreamFeedbackBridge::new();
        let mut memory = OperationalSemanticPriorMemory::new();

        let outcome = process_operational_dream_insight_atomically(
            &mut bridge,
            &mut memory,
            &context,
            insight(&context, vec![0.3], 0.4),
        )
        .unwrap();

        let OperationalDreamSemanticOutcome::Indexed { identity } = outcome else {
            panic!("accepted dream was not indexed");
        };
        assert!(bridge.get_prior(identity.fast_hash).is_some());
        memory
            .source_binding(identity)
            .unwrap()
            .validate_against_context(memory.context_encoder(), &context)
            .unwrap();
    }

    #[test]
    fn operational_dream_failure_rolls_back_exact_and_semantic_state() {
        let context = [0.2];
        let mut bridge = DreamFeedbackBridge::new();
        let mut memory = OperationalSemanticPriorMemory::new();

        let error = process_operational_dream_insight_atomically(
            &mut bridge,
            &mut memory,
            &context,
            insight(&context, Vec::new(), 0.4),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            OperationalDreamSemanticError::Memory(
                OperationalSemanticPriorError::Inner(
                    IntegrityBoundSemanticPriorError::Retrieval(
                        SemanticQuerySupportError::Prior(
                            SemanticPriorError::EmptyPreferredDirection
                        )
                    )
                )
            )
        ));
        assert_eq!(bridge.num_priors(), 0);
        assert_eq!(bridge.stats().total_insights, 0);
        assert!(memory.is_empty());
    }

    #[test]
    fn clamp_saturated_dream_is_exactly_recorded_but_operationally_unsupported() {
        let context = [2.0];
        let mut bridge = DreamFeedbackBridge::new();
        let mut memory = OperationalSemanticPriorMemory::new();

        let outcome = process_operational_dream_insight_atomically(
            &mut bridge,
            &mut memory,
            &context,
            insight(&context, vec![0.3], 0.4),
        )
        .unwrap();

        let OperationalDreamSemanticOutcome::Indexed { identity } = outcome else {
            panic!("accepted dream was not indexed");
        };
        assert!(bridge.get_prior(identity.fast_hash).is_some());
        let source_binding = memory.source_binding(identity).unwrap();
        assert!(!source_binding.within_nominal_support());
        assert!(matches!(
            memory.retrieve(&context, 0.0, 1),
            Err(OperationalSemanticPriorError::QueryOutsideNominalSupport { .. })
        ));
    }
}
