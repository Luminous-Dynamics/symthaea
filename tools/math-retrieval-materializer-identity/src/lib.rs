// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-002B — pre-fetch materializer identity binding.
//!
//! The frozen runtime seam intentionally keeps `CanonicalSourceMaterializer`
//! minimal: materialization receives only a source-object digest. This crate
//! adds a production qualification layer without modifying that predecessor.
//! A materializer implementation must expose its frozen identity, and only a
//! successfully request-bound wrapper implements the seam's materializer trait.

use symthaea_math_retrieval_runtime_seam::{
    CanonicalSourceMaterializer, QualifiedRetrievalRequest, RetrievalError, Sha256Digest,
};

/// Exact identity realized by one canonical source materializer implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializerIdentity {
    pub source_object_contract_sha256: Sha256Digest,
    pub source_fetch_policy_sha256: Sha256Digest,
    pub payload_serialization_sha256: Sha256Digest,
    pub implementation_sha256: Sha256Digest,
}

/// Qualification binding expected by the production caller.
///
/// The first three identities must also agree with the retrieval request's
/// qualified graph. `implementation_sha256` is deliberately separate because
/// the predecessor graph contract does not contain an implementation identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializerBinding {
    pub identity: MaterializerIdentity,
}

/// Implemented by the concrete production materializer itself.
///
/// This avoids a wrapper constructor that accepts an arbitrary caller-supplied
/// identity next to an unrelated inner materializer.
pub trait MaterializerIdentityProvider {
    fn materializer_identity(&self) -> MaterializerIdentity;
}

/// Unbound materializer. This type intentionally does NOT implement
/// `CanonicalSourceMaterializer`.
pub struct QualifiedMaterializer<M> {
    inner: M,
}

impl<M> QualifiedMaterializer<M> {
    pub fn new(inner: M) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M> QualifiedMaterializer<M>
where
    M: CanonicalSourceMaterializer + MaterializerIdentityProvider,
{
    /// Validate the complete materializer identity before returning the only
    /// wrapper that can be passed to `RetrievalExecutor`.
    pub fn bind<'a>(
        &'a mut self,
        request: &QualifiedRetrievalRequest,
        binding: &MaterializerBinding,
    ) -> Result<BoundMaterializer<'a, M>, RetrievalError> {
        let actual = self.inner.materializer_identity();

        compare(
            "source_object_contract_sha256",
            &actual.source_object_contract_sha256,
            &binding.identity.source_object_contract_sha256,
        )?;
        compare(
            "source_fetch_policy_sha256",
            &actual.source_fetch_policy_sha256,
            &binding.identity.source_fetch_policy_sha256,
        )?;
        compare(
            "payload_serialization_sha256",
            &actual.payload_serialization_sha256,
            &binding.identity.payload_serialization_sha256,
        )?;
        compare(
            "implementation_sha256",
            &actual.implementation_sha256,
            &binding.identity.implementation_sha256,
        )?;

        compare(
            "request.graph.source_object_contract_sha256",
            &request.graph.source_object_contract_sha256,
            &binding.identity.source_object_contract_sha256,
        )?;
        compare(
            "request.graph.source_fetch_policy_sha256",
            &request.graph.source_fetch_policy_sha256,
            &binding.identity.source_fetch_policy_sha256,
        )?;
        compare(
            "request.graph.payload_serialization_sha256",
            &request.graph.payload_serialization_sha256,
            &binding.identity.payload_serialization_sha256,
        )?;

        Ok(BoundMaterializer {
            inner: &mut self.inner,
            identity: actual,
        })
    }
}

fn compare(
    field: &str,
    actual: &Sha256Digest,
    expected: &Sha256Digest,
) -> Result<(), RetrievalError> {
    if actual != expected {
        return Err(RetrievalError::Materialization(format!(
            "materializer identity mismatch for {field}: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

/// Request-bound materializer. This is the only type in this crate that
/// implements the predecessor seam's canonical materializer trait.
pub struct BoundMaterializer<'a, M> {
    inner: &'a mut M,
    identity: MaterializerIdentity,
}

impl<'a, M> BoundMaterializer<'a, M> {
    pub fn identity(&self) -> &MaterializerIdentity {
        &self.identity
    }

    pub fn inner(&self) -> &M {
        self.inner
    }

    pub fn inner_mut(&mut self) -> &mut M {
        self.inner
    }
}

impl<M: CanonicalSourceMaterializer> CanonicalSourceMaterializer for BoundMaterializer<'_, M> {
    fn materialize(
        &mut self,
        source_object_sha256: &Sha256Digest,
    ) -> Result<Vec<u8>, RetrievalError> {
        self.inner.materialize(source_object_sha256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_math_retrieval_runtime_seam::{GraphIdentity, RetrievalBudget};

    fn digest(n: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("sha256:{n:064x}")).unwrap()
    }

    fn identity() -> MaterializerIdentity {
        MaterializerIdentity {
            source_object_contract_sha256: digest(7),
            source_fetch_policy_sha256: digest(8),
            payload_serialization_sha256: digest(9),
            implementation_sha256: digest(10),
        }
    }

    fn binding() -> MaterializerBinding {
        MaterializerBinding {
            identity: identity(),
        }
    }

    fn request() -> QualifiedRetrievalRequest {
        QualifiedRetrievalRequest {
            trace_id: "materializer-trace".into(),
            audit_id: "materializer-audit".into(),
            experiment_id: "materializer-experiment".into(),
            arm_id: "S".into(),
            experiment_seed: 1,
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
                max_output_bytes: 1024,
                max_output_item_bytes: 512,
                max_retrieval_queries: 1,
                max_normalized_compute_microunits: 1000,
                max_wall_time_ms: 1000,
            },
        }
    }

    struct CountingMaterializer {
        identity: MaterializerIdentity,
        calls: usize,
    }

    impl CountingMaterializer {
        fn new(identity: MaterializerIdentity) -> Self {
            Self { identity, calls: 0 }
        }
    }

    impl MaterializerIdentityProvider for CountingMaterializer {
        fn materializer_identity(&self) -> MaterializerIdentity {
            self.identity.clone()
        }
    }

    impl CanonicalSourceMaterializer for CountingMaterializer {
        fn materialize(
            &mut self,
            source_object_sha256: &Sha256Digest,
        ) -> Result<Vec<u8>, RetrievalError> {
            self.calls += 1;
            Ok(format!("canonical:{}", source_object_sha256.as_str()).into_bytes())
        }
    }

    fn assert_bind_rejects_without_fetch(
        request: QualifiedRetrievalRequest,
        materializer_identity: MaterializerIdentity,
        binding: MaterializerBinding,
    ) {
        let mut materializer = QualifiedMaterializer::new(CountingMaterializer::new(
            materializer_identity,
        ));
        let error = match materializer.bind(&request, &binding) {
            Ok(_) => panic!("identity mismatch unexpectedly bound materializer"),
            Err(error) => error,
        };
        assert!(matches!(error, RetrievalError::Materialization(_)));
        assert_eq!(materializer.inner().calls, 0);
    }

    #[test]
    fn wrong_request_source_object_contract_rejects_before_first_fetch() {
        let mut req = request();
        req.graph.source_object_contract_sha256 = digest(99);
        assert_bind_rejects_without_fetch(req, identity(), binding());
    }

    #[test]
    fn wrong_request_fetch_policy_rejects_before_first_fetch() {
        let mut req = request();
        req.graph.source_fetch_policy_sha256 = digest(99);
        assert_bind_rejects_without_fetch(req, identity(), binding());
    }

    #[test]
    fn wrong_request_payload_serialization_rejects_before_first_fetch() {
        let mut req = request();
        req.graph.payload_serialization_sha256 = digest(99);
        assert_bind_rejects_without_fetch(req, identity(), binding());
    }

    #[test]
    fn wrong_implementation_identity_rejects_before_first_fetch() {
        let mut actual = identity();
        actual.implementation_sha256 = digest(99);
        assert_bind_rejects_without_fetch(request(), actual, binding());
    }

    #[test]
    fn materializer_cannot_substitute_any_semantic_policy_identity() {
        let mutations: [fn(&mut MaterializerIdentity); 3] = [
            |id| id.source_object_contract_sha256 = digest(91),
            |id| id.source_fetch_policy_sha256 = digest(92),
            |id| id.payload_serialization_sha256 = digest(93),
        ];
        for mutate in mutations {
            let mut actual = identity();
            mutate(&mut actual);
            assert_bind_rejects_without_fetch(request(), actual, binding());
        }
    }

    #[test]
    fn successful_bind_exposes_exact_identity_and_forwards_bytes() {
        let source = digest(20);
        let req = request();
        let expected = binding();
        let mut qualified = QualifiedMaterializer::new(CountingMaterializer::new(identity()));
        let mut bound = qualified.bind(&req, &expected).unwrap();

        assert_eq!(bound.identity(), &identity());
        let bytes = bound.materialize(&source).unwrap();
        assert_eq!(bytes, format!("canonical:{}", source.as_str()).into_bytes());
        assert_eq!(bound.inner().calls, 1);
    }

    #[test]
    fn same_source_under_one_bound_identity_yields_same_canonical_bytes() {
        let source = digest(21);
        let req = request();
        let expected = binding();
        let mut qualified = QualifiedMaterializer::new(CountingMaterializer::new(identity()));
        let mut bound = qualified.bind(&req, &expected).unwrap();

        let first = bound.materialize(&source).unwrap();
        let second = bound.materialize(&source).unwrap();
        assert_eq!(first, second);
        assert_eq!(bound.inner().calls, 2);
    }

    #[test]
    fn binding_cannot_relabel_actual_implementation() {
        let mut expected = binding();
        expected.identity.implementation_sha256 = digest(77);
        assert_bind_rejects_without_fetch(request(), identity(), expected);
    }
}
