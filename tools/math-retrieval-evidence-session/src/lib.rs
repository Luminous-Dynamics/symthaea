// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-002E — transaction-scoped production evidence session.
//!
//! The frozen executor already aborts staged evidence when its own evidence
//! stage/commit operations fail. Production preflight, however, can fail before
//! the executor takes responsibility for the sink. This layer owns the sink and
//! makes clean staged state a transaction invariant across success, ordinary
//! errors, and Rust panic unwinding.

use symthaea_math_retrieval_materializer_identity::MaterializerIdentityProvider;
use symthaea_math_retrieval_production_runtime::{
    ProductionRetrievalRuntime, ProductionRuntimeError,
};
use symthaea_math_retrieval_runtime_seam::{
    AtomicEvidenceSink, CanonicalSourceMaterializer, QualifiedRetrievalBackend,
    QualifiedRetrievalOutcome, QualifiedRetrievalRequest,
};

/// Owns one evidence sink and exposes it mutably only inside a guarded
/// transaction.
///
/// Staged state is cleared:
/// - when the session is constructed;
/// - before every transaction;
/// - whenever a transaction guard is dropped (success, error, or panic);
/// - when the session itself is dropped or finished.
///
/// `AtomicEvidenceSink::abort` is contractually limited to staged state, so
/// already committed evidence remains visible.
pub struct ProductionEvidenceSession<S: AtomicEvidenceSink> {
    sink: Option<S>,
}

impl<S: AtomicEvidenceSink> ProductionEvidenceSession<S> {
    pub fn new(mut sink: S) -> Self {
        sink.abort();
        Self { sink: Some(sink) }
    }

    /// Read-only access for inspection/telemetry.
    ///
    /// There is intentionally no mutable accessor: staged evidence may only be
    /// created while a transaction guard owns the sink.
    pub fn inner(&self) -> &S {
        self.sink
            .as_ref()
            .expect("evidence session sink is unavailable only during finish")
    }

    /// Run one operation with transaction-scoped mutable sink access.
    ///
    /// The guard clears stale staged state before the callback and clears any
    /// remaining staged state again when it drops. This includes panic
    /// unwinding when `panic=unwind`.
    pub fn transact<T, E, F>(&mut self, operation: F) -> Result<T, E>
    where
        F: FnOnce(&mut S) -> Result<T, E>,
    {
        let sink = self
            .sink
            .as_mut()
            .expect("evidence session sink is unavailable only during finish");
        let mut guard = EvidenceTransactionGuard::new(sink);
        operation(guard.sink_mut())
    }

    /// Execute one production retrieval with transaction-scoped evidence.
    pub fn execute<B, M>(
        &mut self,
        runtime: &mut ProductionRetrievalRuntime<B, M>,
        request: QualifiedRetrievalRequest,
    ) -> Result<QualifiedRetrievalOutcome, ProductionRuntimeError>
    where
        B: QualifiedRetrievalBackend,
        M: CanonicalSourceMaterializer + MaterializerIdentityProvider,
    {
        self.transact(|sink| runtime.execute(sink, request))
    }

    /// Consume the session and return the owned sink with staged state cleared.
    pub fn finish(mut self) -> S {
        let mut sink = self
            .sink
            .take()
            .expect("evidence session sink must exist until finish");
        sink.abort();
        sink
    }
}

impl<S: AtomicEvidenceSink> Drop for ProductionEvidenceSession<S> {
    fn drop(&mut self) {
        if let Some(sink) = self.sink.as_mut() {
            sink.abort();
        }
    }
}

struct EvidenceTransactionGuard<'a, S: AtomicEvidenceSink> {
    sink: &'a mut S,
}

impl<'a, S: AtomicEvidenceSink> EvidenceTransactionGuard<'a, S> {
    fn new(sink: &'a mut S) -> Self {
        sink.abort();
        Self { sink }
    }

    fn sink_mut(&mut self) -> &mut S {
        self.sink
    }
}

impl<S: AtomicEvidenceSink> Drop for EvidenceTransactionGuard<'_, S> {
    fn drop(&mut self) {
        self.sink.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use symthaea_math_retrieval_candidate_loader::CandidateArtifactBinding;
    use symthaea_math_retrieval_materializer_identity::{
        MaterializerBinding, MaterializerIdentity,
    };
    use symthaea_math_retrieval_runtime_seam::{
        AtomicEvidenceSink, BackendExecution, BackendRetrieval, CanonicalSourceMaterializer,
        ControlBinding, ControlTransform, GraphIdentity, InMemoryEvidenceSink,
        PayloadAuditBatch, PayloadAuditSink, QualifiedRetrievalBackend, RetrievalBudget,
        RetrievalError, RetrievalTraceReceipt, RetrievalTraceSink, Sha256Digest,
        SingleIndexExecution, SinkError,
    };

    #[derive(Debug, Default)]
    struct TrackingSink {
        trace_staged: bool,
        payload_staged: bool,
        commits: usize,
        aborts: usize,
    }

    impl RetrievalTraceSink for TrackingSink {
        fn stage_trace(&mut self, _trace: &RetrievalTraceReceipt) -> Result<(), SinkError> {
            if self.trace_staged {
                return Err(SinkError("trace already staged".into()));
            }
            self.trace_staged = true;
            Ok(())
        }
    }

    impl PayloadAuditSink for TrackingSink {
        fn stage_payload_audit(&mut self, _audit: &PayloadAuditBatch) -> Result<(), SinkError> {
            if self.payload_staged {
                return Err(SinkError("payload already staged".into()));
            }
            self.payload_staged = true;
            Ok(())
        }
    }

    impl AtomicEvidenceSink for TrackingSink {
        fn commit(&mut self) -> Result<(), SinkError> {
            if !self.trace_staged || !self.payload_staged {
                return Err(SinkError("commit requires both staged artifacts".into()));
            }
            self.trace_staged = false;
            self.payload_staged = false;
            self.commits += 1;
            Ok(())
        }

        fn abort(&mut self) {
            self.trace_staged = false;
            self.payload_staged = false;
            self.aborts += 1;
        }
    }

    #[test]
    fn constructor_clears_preexisting_staged_state() {
        let sink = TrackingSink {
            trace_staged: true,
            payload_staged: true,
            ..TrackingSink::default()
        };
        let session = ProductionEvidenceSession::new(sink);
        assert!(!session.inner().trace_staged);
        assert!(!session.inner().payload_staged);
        assert_eq!(session.inner().aborts, 1);
    }

    #[test]
    fn ordinary_error_clears_staged_state() {
        let mut session = ProductionEvidenceSession::new(TrackingSink::default());
        let result: Result<(), &'static str> = session.transact(|sink| {
            sink.trace_staged = true;
            sink.payload_staged = true;
            Err("injected failure")
        });
        assert_eq!(result, Err("injected failure"));
        assert!(!session.inner().trace_staged);
        assert!(!session.inner().payload_staged);
        assert_eq!(session.inner().commits, 0);
    }

    #[test]
    fn successful_commit_is_preserved_but_staged_state_is_cleared() {
        let mut session = ProductionEvidenceSession::new(TrackingSink::default());
        let result: Result<(), ()> = session.transact(|sink| {
            sink.trace_staged = true;
            sink.payload_staged = true;
            sink.commit().unwrap();
            Ok(())
        });
        assert_eq!(result, Ok(()));
        assert_eq!(session.inner().commits, 1);
        assert!(!session.inner().trace_staged);
        assert!(!session.inner().payload_staged);
    }

    #[test]
    fn successful_callback_without_commit_cannot_leak_staged_state() {
        let mut session = ProductionEvidenceSession::new(TrackingSink::default());
        let result: Result<(), ()> = session.transact(|sink| {
            sink.trace_staged = true;
            sink.payload_staged = true;
            Ok(())
        });
        assert_eq!(result, Ok(()));
        assert_eq!(session.inner().commits, 0);
        assert!(!session.inner().trace_staged);
        assert!(!session.inner().payload_staged);
    }

    #[test]
    fn panic_unwinding_clears_staged_state() {
        let mut session = ProductionEvidenceSession::new(TrackingSink::default());
        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _: Result<(), ()> = session.transact(|sink| {
                sink.trace_staged = true;
                sink.payload_staged = true;
                panic!("injected panic");
            });
        }));
        assert!(panic_result.is_err());
        assert!(!session.inner().trace_staged);
        assert!(!session.inner().payload_staged);
        assert_eq!(session.inner().commits, 0);
    }

    #[test]
    fn finish_returns_clean_sink_without_erasing_commits() {
        let mut session = ProductionEvidenceSession::new(TrackingSink::default());
        let _: Result<(), ()> = session.transact(|sink| {
            sink.trace_staged = true;
            sink.payload_staged = true;
            sink.commit().unwrap();
            sink.trace_staged = true;
            Ok(())
        });
        let sink = session.finish();
        assert_eq!(sink.commits, 1);
        assert!(!sink.trace_staged);
        assert!(!sink.payload_staged);
    }

    const CANDIDATE_ARTIFACT: &[u8] = include_bytes!(
        "../../math-retrieval-production-runtime/tests/production_candidate_fixture.json"
    );
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
            trace_id: "session-trace".into(),
            audit_id: "session-audit".into(),
            experiment_id: "session-experiment".into(),
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

    fn runtime() -> ProductionRetrievalRuntime<FakeBackend, FakeMaterializer> {
        ProductionRetrievalRuntime::from_candidate_bytes(
            CANDIDATE_ARTIFACT,
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
        )
        .unwrap()
    }

    #[test]
    fn production_runtime_execution_through_session_commits_and_finishes_clean() {
        let mut runtime = runtime();
        let mut session = ProductionEvidenceSession::new(InMemoryEvidenceSink::default());

        let outcome = session.execute(&mut runtime, request()).unwrap();
        assert!(outcome.evidence_committed);
        assert_eq!(session.inner().committed().len(), 1);
        assert!(!session.inner().has_staged_state());

        let sink = session.finish();
        assert_eq!(sink.committed().len(), 1);
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn failed_production_preflight_through_session_leaves_no_staged_evidence() {
        let mut runtime = runtime();
        let mut session = ProductionEvidenceSession::new(InMemoryEvidenceSink::default());
        let mut bad_request = request();
        bad_request.graph.candidate_set_sha256 = digest(99);

        let error = session.execute(&mut runtime, bad_request).unwrap_err();
        assert!(matches!(
            error,
            ProductionRuntimeError::Retrieval(RetrievalError::Candidate(_))
        ));
        assert!(session.inner().committed().is_empty());
        assert!(!session.inner().has_staged_state());
    }
}
