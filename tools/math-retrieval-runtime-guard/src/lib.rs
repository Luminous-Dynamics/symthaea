// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-001C — pre-materialization candidate-universe guard.
//!
//! This crate composes with the owner-independent retrieval seam without
//! changing it. A representation-specific backend is allowed to choose ranked
//! source identities, but those identities must all belong to the exact frozen
//! eligible universe before the downstream executor is permitted to materialize
//! payloads, pack context, or emit evidence.

use std::collections::BTreeSet;

use symthaea_math_retrieval_runtime_seam::{
    BackendExecution, BackendRetrieval, QualifiedRetrievalBackend, QualifiedRetrievalRequest,
    RetrievalError, Sha256Digest,
};

/// Exact eligible retrieval universe bound to one content-addressed artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenCandidateUniverse {
    candidate_set_sha256: Sha256Digest,
    candidates: BTreeSet<Sha256Digest>,
}

impl FrozenCandidateUniverse {
    /// Build a frozen universe from the exact candidate-set file digest and the
    /// source identities parsed from that artifact.
    ///
    /// Duplicate identities are rejected rather than silently deduplicated,
    /// because candidate_count is part of the qualified graph identity.
    pub fn new(
        candidate_set_sha256: Sha256Digest,
        candidates: Vec<Sha256Digest>,
    ) -> Result<Self, RetrievalError> {
        if candidates.is_empty() {
            return Err(RetrievalError::Candidate(
                "candidate universe must be non-empty".into(),
            ));
        }
        let declared_len = candidates.len();
        let candidates: BTreeSet<_> = candidates.into_iter().collect();
        if candidates.len() != declared_len {
            return Err(RetrievalError::Candidate(
                "candidate universe contains duplicate source identities".into(),
            ));
        }
        Ok(Self {
            candidate_set_sha256,
            candidates,
        })
    }

    pub fn candidate_set_sha256(&self) -> &Sha256Digest {
        &self.candidate_set_sha256
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub fn contains(&self, source: &Sha256Digest) -> bool {
        self.candidates.contains(source)
    }

    /// Prove that the runtime request is bound to this exact universe.
    pub fn validate_request_binding(
        &self,
        request: &QualifiedRetrievalRequest,
    ) -> Result<(), RetrievalError> {
        if request.graph.candidate_set_sha256 != self.candidate_set_sha256 {
            return Err(RetrievalError::Candidate(
                "request candidate_set_sha256 does not match guarded universe".into(),
            ));
        }
        if request.graph.candidate_count != self.candidate_count() {
            return Err(RetrievalError::Candidate(format!(
                "request candidate_count={} but guarded universe contains {} identities",
                request.graph.candidate_count,
                self.candidate_count()
            )));
        }
        Ok(())
    }

    fn validate_ranked(
        &self,
        label: &str,
        ranked: &[Sha256Digest],
    ) -> Result<(), RetrievalError> {
        for (offset, source) in ranked.iter().enumerate() {
            if !self.contains(source) {
                return Err(RetrievalError::Candidate(format!(
                    "{label} rank {} returned source outside frozen candidate universe: {}",
                    offset + 1,
                    source
                )));
            }
        }
        Ok(())
    }

    /// Validate every raw source identity returned by the backend.
    ///
    /// Fusion channel inputs are checked independently. This is intentionally
    /// earlier than fusion/deduplication so an illegal source cannot disappear
    /// during deterministic fusion and thereby evade the membership proof.
    pub fn validate_execution(&self, execution: &BackendExecution) -> Result<(), RetrievalError> {
        match &execution.retrieval {
            BackendRetrieval::SingleIndex(single) => {
                self.validate_ranked("single-index", &single.ranked_source_object_digests)
            }
            BackendRetrieval::Fusion(fusion) => {
                self.validate_ranked(
                    "fusion Syntax channel",
                    &fusion.syntax.ranked_source_object_digests,
                )?;
                self.validate_ranked(
                    "fusion ExactNormalForm channel",
                    &fusion.normal_form.ranked_source_object_digests,
                )
            }
        }
    }
}

/// Composable guard around any qualified retrieval backend.
///
/// The inner backend is called exactly once. Its raw output is rejected unless
/// every returned source identity belongs to the exact universe bound by the
/// request. Only a validated `BackendExecution` can reach `RetrievalExecutor`.
pub struct MembershipGuardBackend<B> {
    inner: B,
    universe: FrozenCandidateUniverse,
}

impl<B> MembershipGuardBackend<B> {
    pub fn new(inner: B, universe: FrozenCandidateUniverse) -> Self {
        Self { inner, universe }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.inner
    }

    pub fn universe(&self) -> &FrozenCandidateUniverse {
        &self.universe
    }

    pub fn into_inner(self) -> B {
        self.inner
    }
}

impl<B: QualifiedRetrievalBackend> QualifiedRetrievalBackend for MembershipGuardBackend<B> {
    fn execute(
        &mut self,
        request: &QualifiedRetrievalRequest,
    ) -> Result<BackendExecution, RetrievalError> {
        // Validate the universe identity before invoking representation-specific
        // retrieval. A mismatched qualification package must not run at all.
        self.universe.validate_request_binding(request)?;

        let execution = self.inner.execute(request)?;
        self.universe.validate_execution(&execution)?;
        Ok(execution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_math_retrieval_runtime_seam::{
        BackendExecution, BackendRetrieval, CanonicalSourceMaterializer, ChannelExecution,
        ControlBinding, ControlTransform, FusionExecution, FusionMethod, GraphIdentity,
        InMemoryEvidenceSink, QualifiedRetrievalRequest, RetrievalBudget, RetrievalChannel,
        RetrievalExecutor, SingleIndexExecution,
    };

    fn digest(n: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("sha256:{:064x}", n)).unwrap()
    }

    fn request() -> QualifiedRetrievalRequest {
        QualifiedRetrievalRequest {
            trace_id: "membership-trace".into(),
            audit_id: "membership-audit".into(),
            experiment_id: "membership-experiment".into(),
            arm_id: "S".into(),
            experiment_seed: 7,
            query_id: "q-1".into(),
            query_source_object_sha256: digest(250),
            graph: GraphIdentity {
                bundle_sha256: digest(1),
                graph_report_sha256: digest(2),
                experiment_sha256: digest(3),
                retrieval_binding_sha256: digest(4),
                candidate_set_sha256: digest(5),
                candidate_count: 3,
                context_packer_sha256: digest(6),
                source_object_contract_sha256: digest(7),
                source_fetch_policy_sha256: digest(8),
                payload_serialization_sha256: digest(9),
            },
            budget: RetrievalBudget {
                max_output_items: 3,
                max_output_bytes: 64,
                max_output_item_bytes: 32,
                max_retrieval_queries: 2,
                max_normalized_compute_microunits: 1_000_000,
                max_wall_time_ms: 1_000,
            },
        }
    }

    fn universe() -> FrozenCandidateUniverse {
        FrozenCandidateUniverse::new(digest(5), vec![digest(20), digest(21), digest(22)]).unwrap()
    }

    fn control(index: Sha256Digest) -> ControlBinding {
        ControlBinding {
            index_manifest_sha256: index,
            control_transform: ControlTransform::None,
            control_seed: None,
            control_artifact_sha256: None,
        }
    }

    #[derive(Clone)]
    struct FakeBackend {
        execution: BackendExecution,
        calls: usize,
    }

    impl QualifiedRetrievalBackend for FakeBackend {
        fn execute(
            &mut self,
            _request: &QualifiedRetrievalRequest,
        ) -> Result<BackendExecution, RetrievalError> {
            self.calls += 1;
            Ok(self.execution.clone())
        }
    }

    fn single(ids: Vec<Sha256Digest>) -> FakeBackend {
        let index = digest(10);
        FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(11),
                    requested_k: 3,
                    ranked_source_object_digests: ids,
                    control: control(index),
                }),
                retrieval_queries_used: 1,
                wall_time_ms_used: 5,
                normalized_compute_microunits_used: 10,
            },
            calls: 0,
        }
    }

    fn channel(
        channel: RetrievalChannel,
        index: Sha256Digest,
        ids: Vec<Sha256Digest>,
    ) -> ChannelExecution {
        ChannelExecution {
            channel,
            index_manifest_sha256: index.clone(),
            index_artifact_sha256: digest(if channel == RetrievalChannel::Syntax { 31 } else { 32 }),
            requested_k: 1,
            input_bytes_used: 4,
            ranked_source_object_digests: ids,
            control: control(index),
        }
    }

    #[test]
    fn exact_universe_binding_is_required_before_backend_execution() {
        let mut req = request();
        req.graph.candidate_set_sha256 = digest(99);
        let inner = single(vec![digest(20)]);
        let mut guarded = MembershipGuardBackend::new(inner, universe());
        let err = guarded.execute(&req).unwrap_err();
        assert!(matches!(err, RetrievalError::Candidate(_)));
        assert_eq!(guarded.inner().calls, 0);

        let mut req = request();
        req.graph.candidate_count = 2;
        let inner = single(vec![digest(20)]);
        let mut guarded = MembershipGuardBackend::new(inner, universe());
        let err = guarded.execute(&req).unwrap_err();
        assert!(matches!(err, RetrievalError::Candidate(_)));
        assert_eq!(guarded.inner().calls, 0);
    }

    #[test]
    fn legal_single_index_execution_passes_membership_gate_once() {
        let inner = single(vec![digest(20), digest(21)]);
        let mut guarded = MembershipGuardBackend::new(inner, universe());
        let execution = guarded.execute(&request()).unwrap();
        assert_eq!(guarded.inner().calls, 1);
        match execution.retrieval {
            BackendRetrieval::SingleIndex(single) => {
                assert_eq!(single.ranked_source_object_digests, vec![digest(20), digest(21)]);
            }
            BackendRetrieval::Fusion(_) => panic!("expected single-index execution"),
        }
    }

    #[test]
    fn illegal_single_index_source_is_rejected() {
        let inner = single(vec![digest(20), digest(99)]);
        let mut guarded = MembershipGuardBackend::new(inner, universe());
        let err = guarded.execute(&request()).unwrap_err();
        assert!(matches!(err, RetrievalError::Candidate(_)));
        assert_eq!(guarded.inner().calls, 1);
    }

    #[test]
    fn illegal_source_in_either_fusion_channel_is_rejected_before_fusion() {
        let syntax_index = digest(40);
        let normal_index = digest(41);
        let inner = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::Fusion(FusionExecution {
                    fusion_policy_sha256: digest(42),
                    method: FusionMethod::DeterministicInterleave {
                        start_channel: RetrievalChannel::Syntax,
                    },
                    syntax: channel(
                        RetrievalChannel::Syntax,
                        syntax_index,
                        vec![digest(20)],
                    ),
                    normal_form: channel(
                        RetrievalChannel::ExactNormalForm,
                        normal_index,
                        vec![digest(99)],
                    ),
                }),
                retrieval_queries_used: 2,
                wall_time_ms_used: 5,
                normalized_compute_microunits_used: 10,
            },
            calls: 0,
        };
        let mut guarded = MembershipGuardBackend::new(inner, universe());
        let err = guarded.execute(&request()).unwrap_err();
        assert!(matches!(err, RetrievalError::Candidate(_)));
        assert_eq!(guarded.inner().calls, 1);
    }

    #[derive(Default)]
    struct CountingMaterializer {
        calls: usize,
    }

    impl CanonicalSourceMaterializer for CountingMaterializer {
        fn materialize(
            &mut self,
            _source_object_sha256: &Sha256Digest,
        ) -> Result<Vec<u8>, RetrievalError> {
            self.calls += 1;
            Ok(b"canonical payload".to_vec())
        }
    }

    #[test]
    fn illegal_candidate_cannot_reach_materializer_or_evidence_sinks() {
        let inner = single(vec![digest(99)]);
        let mut guarded = MembershipGuardBackend::new(inner, universe());
        let mut materializer = CountingMaterializer::default();
        let mut sink = InMemoryEvidenceSink::default();

        let err = RetrievalExecutor::execute(
            &mut guarded,
            &mut materializer,
            &mut sink,
            request(),
        )
        .unwrap_err();

        assert!(matches!(err, RetrievalError::Candidate(_)));
        assert_eq!(guarded.inner().calls, 1);
        assert_eq!(materializer.calls, 0);
        assert!(sink.committed().is_empty());
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn duplicate_candidate_universe_is_rejected_instead_of_silently_deduped() {
        let err = FrozenCandidateUniverse::new(digest(5), vec![digest(20), digest(20)])
            .unwrap_err();
        assert!(matches!(err, RetrievalError::Candidate(_)));
    }
}
