// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-002C — production retrieval composition boundary.
//!
//! This crate composes the previously separated production gates around the
//! frozen retrieval executor. A production caller supplies exact candidate
//! artifact bytes, a qualified artifact binding, a retrieval backend, an
//! identity-providing canonical materializer, and an atomic evidence sink.
//! The runtime owns the gate ordering so callers cannot accidentally skip the
//! candidate loader, membership guard, or materializer identity bind.

use std::fmt;
use std::path::Path;
use symthaea_math_retrieval_candidate_loader::{
    CandidateArtifactBinding, CandidateArtifactLoader, CandidateLoaderError,
    LoadedCandidateArtifact,
};
use symthaea_math_retrieval_materializer_identity::{
    MaterializerBinding, MaterializerIdentityProvider, QualifiedMaterializer,
};
use symthaea_math_retrieval_membership_guard::MembershipGuardBackend;
use symthaea_math_retrieval_runtime_seam::{
    AtomicEvidenceSink, CanonicalSourceMaterializer, QualifiedRetrievalBackend,
    QualifiedRetrievalOutcome, QualifiedRetrievalRequest, RetrievalError, RetrievalExecutor,
};

#[derive(Debug)]
pub enum ProductionRuntimeError {
    CandidateArtifact(CandidateLoaderError),
    Retrieval(RetrievalError),
}

impl fmt::Display for ProductionRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateArtifact(error) => write!(f, "candidate artifact admission failed: {error}"),
            Self::Retrieval(error) => write!(f, "qualified retrieval failed: {error}"),
        }
    }
}

impl std::error::Error for ProductionRuntimeError {}

impl From<CandidateLoaderError> for ProductionRuntimeError {
    fn from(error: CandidateLoaderError) -> Self {
        Self::CandidateArtifact(error)
    }
}

impl From<RetrievalError> for ProductionRuntimeError {
    fn from(error: RetrievalError) -> Self {
        Self::Retrieval(error)
    }
}

/// One production retrieval runtime with all qualification gates installed.
pub struct ProductionRetrievalRuntime<B, M> {
    candidate_artifact: LoadedCandidateArtifact,
    backend: MembershipGuardBackend<B>,
    materializer: QualifiedMaterializer<M>,
    materializer_binding: MaterializerBinding,
}

impl<B, M> ProductionRetrievalRuntime<B, M> {
    pub fn from_candidate_bytes(
        candidate_artifact_bytes: &[u8],
        candidate_binding: &CandidateArtifactBinding,
        backend: B,
        materializer: M,
        materializer_binding: MaterializerBinding,
    ) -> Result<Self, ProductionRuntimeError> {
        let loaded = CandidateArtifactLoader::load_bytes(
            candidate_artifact_bytes,
            candidate_binding,
        )?;
        Ok(Self::from_loaded(
            loaded,
            backend,
            materializer,
            materializer_binding,
        ))
    }

    pub fn from_candidate_file(
        path: impl AsRef<Path>,
        candidate_binding: &CandidateArtifactBinding,
        backend: B,
        materializer: M,
        materializer_binding: MaterializerBinding,
    ) -> Result<Self, ProductionRuntimeError> {
        let loaded = CandidateArtifactLoader::load_file(path, candidate_binding)?;
        Ok(Self::from_loaded(
            loaded,
            backend,
            materializer,
            materializer_binding,
        ))
    }

    fn from_loaded(
        candidate_artifact: LoadedCandidateArtifact,
        backend: B,
        materializer: M,
        materializer_binding: MaterializerBinding,
    ) -> Self {
        let guarded_backend =
            MembershipGuardBackend::new(backend, candidate_artifact.universe().clone());
        Self {
            candidate_artifact,
            backend: guarded_backend,
            materializer: QualifiedMaterializer::new(materializer),
            materializer_binding,
        }
    }

    pub fn candidate_artifact(&self) -> &LoadedCandidateArtifact {
        &self.candidate_artifact
    }

    pub fn backend(&self) -> &MembershipGuardBackend<B> {
        &self.backend
    }

    pub fn materializer(&self) -> &QualifiedMaterializer<M> {
        &self.materializer
    }

    pub fn materializer_binding(&self) -> &MaterializerBinding {
        &self.materializer_binding
    }
}

impl<B, M> ProductionRetrievalRuntime<B, M>
where
    B: QualifiedRetrievalBackend,
    M: CanonicalSourceMaterializer + MaterializerIdentityProvider,
{
    /// Execute one qualified retrieval through every production gate.
    ///
    /// Ordering is intentional:
    /// 1. request must bind the already-loaded candidate universe;
    /// 2. materializer must bind the request + implementation qualification;
    /// 3. only then may the retrieval backend execute;
    /// 4. the membership guard checks raw backend source IDs;
    /// 5. the frozen executor materializes, packs, audits, and commits evidence.
    pub fn execute<S>(
        &mut self,
        sink: &mut S,
        request: QualifiedRetrievalRequest,
    ) -> Result<QualifiedRetrievalOutcome, ProductionRuntimeError>
    where
        S: AtomicEvidenceSink,
    {
        let Self {
            backend,
            materializer,
            materializer_binding,
            ..
        } = self;

        // Preflight the candidate binding before even asking the materializer
        // to bind. The membership guard repeats this check when the backend is
        // invoked; the duplication is deliberate defense in depth.
        backend.universe().validate_request_binding(&request)?;

        let mut bound_materializer = materializer.bind(&request, materializer_binding)?;

        RetrievalExecutor::execute(
            backend,
            &mut bound_materializer,
            sink,
            request,
        )
        .map_err(ProductionRuntimeError::Retrieval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_math_retrieval_materializer_identity::{
        MaterializerIdentity, MaterializerIdentityProvider,
    };
    use symthaea_math_retrieval_runtime_seam::{
        BackendExecution, BackendRetrieval, CanonicalSourceMaterializer, ControlBinding,
        ControlTransform, GraphIdentity, InMemoryEvidenceSink, RetrievalBudget,
        Sha256Digest, SingleIndexExecution,
    };

    const CANDIDATE_ARTIFACT: &[u8] =
        include_bytes!("../tests/production_candidate_fixture.json");
    const CANDIDATE_ARTIFACT_SHA256: &str =
        "sha256:e92003a3eeb556515927dae44e2c95847fa64a65d9ac7c43535f1f410576b77a";

    fn digest(n: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("sha256:{n:064x}")).unwrap()
    }

    fn candidate_binding() -> CandidateArtifactBinding {
        CandidateArtifactBinding {
            candidate_set_sha256: Sha256Digest::parse(CANDIDATE_ARTIFACT_SHA256).unwrap(),
            candidate_count: 3,
            corpus_snapshot_sha256: digest(100),
            knowledge_boundary_sha256: digest(101),
            candidate_eligibility_policy_sha256: digest(102),
        }
    }

    fn materializer_identity() -> MaterializerIdentity {
        MaterializerIdentity {
            source_object_contract_sha256: digest(7),
            source_fetch_policy_sha256: digest(8),
            payload_serialization_sha256: digest(9),
            implementation_sha256: digest(30),
        }
    }

    fn materializer_binding() -> MaterializerBinding {
        MaterializerBinding {
            identity: materializer_identity(),
        }
    }

    fn request() -> QualifiedRetrievalRequest {
        QualifiedRetrievalRequest {
            trace_id: "production-trace".into(),
            audit_id: "production-audit".into(),
            experiment_id: "production-experiment".into(),
            arm_id: "S".into(),
            experiment_seed: 42,
            query_id: "q-1".into(),
            query_source_object_sha256: digest(250),
            graph: GraphIdentity {
                bundle_sha256: digest(1),
                graph_report_sha256: digest(2),
                experiment_sha256: digest(3),
                retrieval_binding_sha256: digest(4),
                candidate_set_sha256: Sha256Digest::parse(CANDIDATE_ARTIFACT_SHA256).unwrap(),
                candidate_count: 3,
                context_packer_sha256: digest(6),
                source_object_contract_sha256: digest(7),
                source_fetch_policy_sha256: digest(8),
                payload_serialization_sha256: digest(9),
            },
            budget: RetrievalBudget {
                max_output_items: 3,
                max_output_bytes: 256,
                max_output_item_bytes: 128,
                max_retrieval_queries: 1,
                max_normalized_compute_microunits: 1000,
                max_wall_time_ms: 1000,
            },
        }
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
        ranked: Vec<Sha256Digest>,
        calls: usize,
    }

    impl QualifiedRetrievalBackend for FakeBackend {
        fn execute(
            &mut self,
            _request: &QualifiedRetrievalRequest,
        ) -> Result<BackendExecution, RetrievalError> {
            self.calls += 1;
            let index = digest(10);
            Ok(BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(11),
                    requested_k: self.ranked.len() as u32,
                    ranked_source_object_digests: self.ranked.clone(),
                    control: control(index),
                }),
                retrieval_queries_used: 1,
                wall_time_ms_used: 1,
                normalized_compute_microunits_used: 1,
            })
        }
    }

    struct FakeMaterializer {
        identity: MaterializerIdentity,
        calls: usize,
    }

    impl MaterializerIdentityProvider for FakeMaterializer {
        fn materializer_identity(&self) -> MaterializerIdentity {
            self.identity.clone()
        }
    }

    impl CanonicalSourceMaterializer for FakeMaterializer {
        fn materialize(
            &mut self,
            source_object_sha256: &Sha256Digest,
        ) -> Result<Vec<u8>, RetrievalError> {
            self.calls += 1;
            Ok(format!("payload:{}", source_object_sha256.as_str()).into_bytes())
        }
    }

    fn runtime(
        ranked: Vec<Sha256Digest>,
        identity: MaterializerIdentity,
    ) -> ProductionRetrievalRuntime<FakeBackend, FakeMaterializer> {
        ProductionRetrievalRuntime::from_candidate_bytes(
            CANDIDATE_ARTIFACT,
            &candidate_binding(),
            FakeBackend { ranked, calls: 0 },
            FakeMaterializer { identity, calls: 0 },
            materializer_binding(),
        )
        .unwrap()
    }

    #[test]
    fn exact_candidate_artifact_is_retained_by_production_runtime() {
        let runtime = runtime(vec![digest(20)], materializer_identity());
        assert_eq!(runtime.candidate_artifact().artifact_bytes(), CANDIDATE_ARTIFACT);
        assert_eq!(runtime.candidate_artifact().candidate_count(), 3);
    }

    #[test]
    fn mutated_candidate_bytes_fail_before_runtime_construction() {
        let mut mutated = CANDIDATE_ARTIFACT.to_vec();
        mutated.push(b'\n');
        let result = ProductionRetrievalRuntime::from_candidate_bytes(
            &mutated,
            &candidate_binding(),
            FakeBackend {
                ranked: vec![digest(20)],
                calls: 0,
            },
            FakeMaterializer {
                identity: materializer_identity(),
                calls: 0,
            },
            materializer_binding(),
        );
        assert!(matches!(result, Err(ProductionRuntimeError::CandidateArtifact(_))));
    }

    #[test]
    fn wrong_request_candidate_binding_rejects_before_backend_or_materializer() {
        let mut runtime = runtime(vec![digest(20)], materializer_identity());
        let mut req = request();
        req.graph.candidate_set_sha256 = digest(99);
        let mut sink = InMemoryEvidenceSink::default();

        let error = runtime.execute(&mut sink, req).unwrap_err();
        assert!(matches!(error, ProductionRuntimeError::Retrieval(RetrievalError::Candidate(_))));
        assert_eq!(runtime.backend().inner().calls, 0);
        assert_eq!(runtime.materializer().inner().calls, 0);
        assert!(sink.committed().is_empty());
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn materializer_identity_mismatch_rejects_before_backend_or_fetch() {
        let mut wrong = materializer_identity();
        wrong.implementation_sha256 = digest(99);
        let mut runtime = runtime(vec![digest(20)], wrong);
        let mut sink = InMemoryEvidenceSink::default();

        let error = runtime.execute(&mut sink, request()).unwrap_err();
        assert!(matches!(
            error,
            ProductionRuntimeError::Retrieval(RetrievalError::Materialization(_))
        ));
        assert_eq!(runtime.backend().inner().calls, 0);
        assert_eq!(runtime.materializer().inner().calls, 0);
        assert!(sink.committed().is_empty());
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn illegal_backend_candidate_is_rejected_before_materialization_or_evidence() {
        let mut runtime = runtime(vec![digest(99)], materializer_identity());
        let mut sink = InMemoryEvidenceSink::default();

        let error = runtime.execute(&mut sink, request()).unwrap_err();
        assert!(matches!(error, ProductionRuntimeError::Retrieval(RetrievalError::Candidate(_))));
        assert_eq!(runtime.backend().inner().calls, 1);
        assert_eq!(runtime.materializer().inner().calls, 0);
        assert!(sink.committed().is_empty());
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn legal_production_execution_commits_trace_and_actual_payload_audit() {
        let mut runtime = runtime(vec![digest(20)], materializer_identity());
        let mut sink = InMemoryEvidenceSink::default();

        let outcome = runtime.execute(&mut sink, request()).unwrap();
        assert!(outcome.evidence_committed);
        assert_eq!(runtime.backend().inner().calls, 1);
        assert_eq!(runtime.materializer().inner().calls, 1);
        assert_eq!(sink.committed().len(), 1);
        assert!(!sink.has_staged_state());
        assert_eq!(outcome.trace.graph.candidate_set_sha256.as_str(), CANDIDATE_ARTIFACT_SHA256);
        assert_eq!(outcome.payload_audit.entries.len(), 1);
        assert_eq!(outcome.payload_audit.entries[0].source_object_sha256, digest(20));
        assert_eq!(
            outcome.payload_audit.entries[0].canonical_payload_utf8,
            format!("payload:{}", digest(20).as_str()).into_bytes()
        );
    }
}
