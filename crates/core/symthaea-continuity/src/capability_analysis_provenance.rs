// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Algorithm-bound provenance for deterministic capability analyses.
//!
//! Exact result receipts bind inputs and derived outputs, but a result identity
//! alone should not be interpreted as naming the semantics of the algorithm
//! that produced it. This layer binds each exact result receipt to an explicit,
//! versioned algorithm-semantics identity.
//!
//! Core theorem:
//!
//! `AlgorithmProvenance != Observation != AvailabilityEvidence != Authority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CapabilityActivationAssumptionsId, CapabilityActivationClosureReceiptId,
    CapabilityActivationClosureReceiptV1, CapabilityActivationClosureV1,
    CapabilityCounterfactualConfigId, CapabilityCounterfactualFrontierReceiptId,
    CapabilityCounterfactualFrontierReceiptV1, CapabilityCounterfactualFrontierV1,
    CapabilityCounterfactualQueryId, CapabilityGraphSnapshotId,
};

/// Transport schema for algorithm-bound activation-closure provenance.
pub const CAPABILITY_ACTIVATION_PROVENANCE_SCHEMA_V1: &str =
    "symthaea-continuity-capability-activation-provenance-v1";
/// Transport schema for algorithm-bound counterfactual-frontier provenance.
pub const CAPABILITY_COUNTERFACTUAL_PROVENANCE_SCHEMA_V1: &str =
    "symthaea-continuity-capability-counterfactual-provenance-v1";

/// Stable semantic name of the CC-03B monotone fixed-point activation algorithm.
///
/// Any intentional change to activation semantics must use a new semantic name
/// and therefore a new algorithm identity, even if a fixture happens to produce
/// the same closure bytes.
pub const CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1: &str =
    "symthaea-continuity-capability-activation-monotone-fixed-point-v1";

/// Stable semantic name of the CC-03C bounded support-frontier algorithm.
///
/// Any intentional change to frontier search/completeness semantics must use a
/// new semantic name and therefore a new algorithm identity.
pub const CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1: &str =
    "symthaea-continuity-capability-counterfactual-bounded-support-frontier-v1";

const ACTIVATION_ALGORITHM_DOMAIN: &[u8] =
    b"symthaea.continuity.capability-activation.algorithm-semantics.v1\0";
const COUNTERFACTUAL_ALGORITHM_DOMAIN: &[u8] =
    b"symthaea.continuity.capability-counterfactual.algorithm-semantics.v1\0";
const ACTIVATION_PROVENANCE_DOMAIN: &[u8] =
    b"symthaea.continuity.capability-activation.provenance.v1\0";
const COUNTERFACTUAL_PROVENANCE_DOMAIN: &[u8] =
    b"symthaea.continuity.capability-counterfactual.provenance.v1\0";

/// Exact identity of one activation algorithm semantics revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityActivationAlgorithmId([u8; 32]);

impl CapabilityActivationAlgorithmId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact identity of one counterfactual algorithm semantics revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityCounterfactualAlgorithmId([u8; 32]);

impl CapabilityCounterfactualAlgorithmId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact identity of one activation result plus algorithm-semantics binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityActivationProvenanceId([u8; 32]);

impl CapabilityActivationProvenanceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact identity of one counterfactual result plus algorithm-semantics binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityCounterfactualProvenanceId([u8; 32]);

impl CapabilityCounterfactualProvenanceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Current V1 activation algorithm identity.
pub fn capability_activation_algorithm_id_v1() -> CapabilityActivationAlgorithmId {
    CapabilityActivationAlgorithmId(domain_hash(
        ACTIVATION_ALGORITHM_DOMAIN,
        CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1.as_bytes(),
    ))
}

/// Current V1 counterfactual algorithm identity.
pub fn capability_counterfactual_algorithm_id_v1() -> CapabilityCounterfactualAlgorithmId {
    CapabilityCounterfactualAlgorithmId(domain_hash(
        COUNTERFACTUAL_ALGORITHM_DOMAIN,
        CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1.as_bytes(),
    ))
}

/// Serializable provenance binding for one exact CC-03B activation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityActivationProvenanceV1 {
    schema_version: String,
    algorithm_semantics: String,
    algorithm_id: CapabilityActivationAlgorithmId,
    source_snapshot_id: CapabilityGraphSnapshotId,
    assumptions_id: CapabilityActivationAssumptionsId,
    result_receipt_id: CapabilityActivationClosureReceiptId,
    provenance_id: CapabilityActivationProvenanceId,
}

impl CapabilityActivationProvenanceV1 {
    pub fn from_closure(closure: &CapabilityActivationClosureV1) -> Self {
        let algorithm_semantics = CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1.to_owned();
        let algorithm_id = capability_activation_algorithm_id_v1();
        let source_snapshot_id = closure.source_snapshot_id();
        let assumptions_id = closure.assumptions_id();
        let result_receipt_id = CapabilityActivationClosureReceiptV1::from_closure(closure).id();
        let provenance_id = CapabilityActivationProvenanceId(hash_activation_provenance(
            &algorithm_semantics,
            algorithm_id,
            source_snapshot_id,
            assumptions_id,
            result_receipt_id,
        ));
        Self {
            schema_version: CAPABILITY_ACTIVATION_PROVENANCE_SCHEMA_V1.to_owned(),
            algorithm_semantics,
            algorithm_id,
            source_snapshot_id,
            assumptions_id,
            result_receipt_id,
            provenance_id,
        }
    }

    pub fn id(&self) -> CapabilityActivationProvenanceId {
        self.provenance_id
    }

    pub fn algorithm_semantics(&self) -> &str {
        &self.algorithm_semantics
    }

    pub fn algorithm_id(&self) -> CapabilityActivationAlgorithmId {
        self.algorithm_id
    }

    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    pub fn assumptions_id(&self) -> CapabilityActivationAssumptionsId {
        self.assumptions_id
    }

    pub fn result_receipt_id(&self) -> CapabilityActivationClosureReceiptId {
        self.result_receipt_id
    }

    /// Rebind transported provenance to the exact runtime closure and the V1
    /// algorithm-semantics contract. No operational authority is inferred.
    pub fn validate_against(
        &self,
        closure: &CapabilityActivationClosureV1,
    ) -> Result<(), CapabilityAnalysisProvenanceError> {
        if self.schema_version != CAPABILITY_ACTIVATION_PROVENANCE_SCHEMA_V1 {
            return Err(CapabilityAnalysisProvenanceError::UnsupportedActivationProvenanceSchema(
                self.schema_version.clone(),
            ));
        }
        if self.algorithm_semantics != CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1 {
            return Err(CapabilityAnalysisProvenanceError::UnsupportedActivationAlgorithmSemantics(
                self.algorithm_semantics.clone(),
            ));
        }
        if self.algorithm_id != capability_activation_algorithm_id_v1() {
            return Err(CapabilityAnalysisProvenanceError::ActivationAlgorithmIdentityMismatch);
        }
        if self.source_snapshot_id != closure.source_snapshot_id() {
            return Err(CapabilityAnalysisProvenanceError::ActivationSourceMismatch);
        }
        if self.assumptions_id != closure.assumptions_id() {
            return Err(CapabilityAnalysisProvenanceError::ActivationAssumptionsMismatch);
        }
        let expected_receipt = CapabilityActivationClosureReceiptV1::from_closure(closure).id();
        if self.result_receipt_id != expected_receipt {
            return Err(CapabilityAnalysisProvenanceError::ActivationResultReceiptMismatch);
        }
        let expected_id = CapabilityActivationProvenanceId(hash_activation_provenance(
            &self.algorithm_semantics,
            self.algorithm_id,
            self.source_snapshot_id,
            self.assumptions_id,
            self.result_receipt_id,
        ));
        if self.provenance_id != expected_id {
            return Err(CapabilityAnalysisProvenanceError::ActivationProvenanceIdentityMismatch);
        }
        Ok(())
    }
}

/// Serializable provenance binding for one exact CC-03C counterfactual result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityCounterfactualProvenanceV1 {
    schema_version: String,
    algorithm_semantics: String,
    algorithm_id: CapabilityCounterfactualAlgorithmId,
    source_snapshot_id: CapabilityGraphSnapshotId,
    base_assumptions_id: CapabilityActivationAssumptionsId,
    query_id: CapabilityCounterfactualQueryId,
    config_id: CapabilityCounterfactualConfigId,
    result_receipt_id: CapabilityCounterfactualFrontierReceiptId,
    provenance_id: CapabilityCounterfactualProvenanceId,
}

impl CapabilityCounterfactualProvenanceV1 {
    pub fn from_frontier(frontier: &CapabilityCounterfactualFrontierV1) -> Self {
        let algorithm_semantics = CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1.to_owned();
        let algorithm_id = capability_counterfactual_algorithm_id_v1();
        let source_snapshot_id = frontier.source_snapshot_id();
        let base_assumptions_id = frontier.base_assumptions_id();
        let query_id = frontier.query_id();
        let config_id = frontier.config_id();
        let result_receipt_id = CapabilityCounterfactualFrontierReceiptV1::from_frontier(frontier).id();
        let provenance_id = CapabilityCounterfactualProvenanceId(hash_counterfactual_provenance(
            &algorithm_semantics,
            algorithm_id,
            source_snapshot_id,
            base_assumptions_id,
            query_id,
            config_id,
            result_receipt_id,
        ));
        Self {
            schema_version: CAPABILITY_COUNTERFACTUAL_PROVENANCE_SCHEMA_V1.to_owned(),
            algorithm_semantics,
            algorithm_id,
            source_snapshot_id,
            base_assumptions_id,
            query_id,
            config_id,
            result_receipt_id,
            provenance_id,
        }
    }

    pub fn id(&self) -> CapabilityCounterfactualProvenanceId {
        self.provenance_id
    }

    pub fn algorithm_semantics(&self) -> &str {
        &self.algorithm_semantics
    }

    pub fn algorithm_id(&self) -> CapabilityCounterfactualAlgorithmId {
        self.algorithm_id
    }

    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    pub fn base_assumptions_id(&self) -> CapabilityActivationAssumptionsId {
        self.base_assumptions_id
    }

    pub fn query_id(&self) -> CapabilityCounterfactualQueryId {
        self.query_id
    }

    pub fn config_id(&self) -> CapabilityCounterfactualConfigId {
        self.config_id
    }

    pub fn result_receipt_id(&self) -> CapabilityCounterfactualFrontierReceiptId {
        self.result_receipt_id
    }

    pub fn validate_against(
        &self,
        frontier: &CapabilityCounterfactualFrontierV1,
    ) -> Result<(), CapabilityAnalysisProvenanceError> {
        if self.schema_version != CAPABILITY_COUNTERFACTUAL_PROVENANCE_SCHEMA_V1 {
            return Err(CapabilityAnalysisProvenanceError::UnsupportedCounterfactualProvenanceSchema(
                self.schema_version.clone(),
            ));
        }
        if self.algorithm_semantics != CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1 {
            return Err(
                CapabilityAnalysisProvenanceError::UnsupportedCounterfactualAlgorithmSemantics(
                    self.algorithm_semantics.clone(),
                ),
            );
        }
        if self.algorithm_id != capability_counterfactual_algorithm_id_v1() {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualAlgorithmIdentityMismatch);
        }
        if self.source_snapshot_id != frontier.source_snapshot_id() {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualSourceMismatch);
        }
        if self.base_assumptions_id != frontier.base_assumptions_id() {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualAssumptionsMismatch);
        }
        if self.query_id != frontier.query_id() {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualQueryMismatch);
        }
        if self.config_id != frontier.config_id() {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualConfigMismatch);
        }
        let expected_receipt = CapabilityCounterfactualFrontierReceiptV1::from_frontier(frontier).id();
        if self.result_receipt_id != expected_receipt {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualResultReceiptMismatch);
        }
        let expected_id = CapabilityCounterfactualProvenanceId(hash_counterfactual_provenance(
            &self.algorithm_semantics,
            self.algorithm_id,
            self.source_snapshot_id,
            self.base_assumptions_id,
            self.query_id,
            self.config_id,
            self.result_receipt_id,
        ));
        if self.provenance_id != expected_id {
            return Err(CapabilityAnalysisProvenanceError::CounterfactualProvenanceIdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityAnalysisProvenanceError {
    #[error("unsupported activation provenance schema: {0}")]
    UnsupportedActivationProvenanceSchema(String),
    #[error("unsupported activation algorithm semantics: {0}")]
    UnsupportedActivationAlgorithmSemantics(String),
    #[error("activation algorithm identity does not match the supported semantics")]
    ActivationAlgorithmIdentityMismatch,
    #[error("activation provenance source snapshot does not match runtime closure")]
    ActivationSourceMismatch,
    #[error("activation provenance assumptions do not match runtime closure")]
    ActivationAssumptionsMismatch,
    #[error("activation provenance result receipt does not match runtime closure")]
    ActivationResultReceiptMismatch,
    #[error("activation provenance identity does not match canonical fields")]
    ActivationProvenanceIdentityMismatch,
    #[error("unsupported counterfactual provenance schema: {0}")]
    UnsupportedCounterfactualProvenanceSchema(String),
    #[error("unsupported counterfactual algorithm semantics: {0}")]
    UnsupportedCounterfactualAlgorithmSemantics(String),
    #[error("counterfactual algorithm identity does not match the supported semantics")]
    CounterfactualAlgorithmIdentityMismatch,
    #[error("counterfactual provenance source snapshot does not match runtime frontier")]
    CounterfactualSourceMismatch,
    #[error("counterfactual provenance assumptions do not match runtime frontier")]
    CounterfactualAssumptionsMismatch,
    #[error("counterfactual provenance query does not match runtime frontier")]
    CounterfactualQueryMismatch,
    #[error("counterfactual provenance config does not match runtime frontier")]
    CounterfactualConfigMismatch,
    #[error("counterfactual provenance result receipt does not match runtime frontier")]
    CounterfactualResultReceiptMismatch,
    #[error("counterfactual provenance identity does not match canonical fields")]
    CounterfactualProvenanceIdentityMismatch,
}

fn hash_activation_provenance(
    algorithm_semantics: &str,
    algorithm_id: CapabilityActivationAlgorithmId,
    source_snapshot_id: CapabilityGraphSnapshotId,
    assumptions_id: CapabilityActivationAssumptionsId,
    result_receipt_id: CapabilityActivationClosureReceiptId,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, CAPABILITY_ACTIVATION_PROVENANCE_SCHEMA_V1);
    put_str(&mut bytes, algorithm_semantics);
    bytes.extend_from_slice(algorithm_id.as_bytes());
    bytes.extend_from_slice(source_snapshot_id.as_bytes());
    bytes.extend_from_slice(assumptions_id.as_bytes());
    bytes.extend_from_slice(result_receipt_id.as_bytes());
    domain_hash(ACTIVATION_PROVENANCE_DOMAIN, &bytes)
}

fn hash_counterfactual_provenance(
    algorithm_semantics: &str,
    algorithm_id: CapabilityCounterfactualAlgorithmId,
    source_snapshot_id: CapabilityGraphSnapshotId,
    base_assumptions_id: CapabilityActivationAssumptionsId,
    query_id: CapabilityCounterfactualQueryId,
    config_id: CapabilityCounterfactualConfigId,
    result_receipt_id: CapabilityCounterfactualFrontierReceiptId,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, CAPABILITY_COUNTERFACTUAL_PROVENANCE_SCHEMA_V1);
    put_str(&mut bytes, algorithm_semantics);
    bytes.extend_from_slice(algorithm_id.as_bytes());
    bytes.extend_from_slice(source_snapshot_id.as_bytes());
    bytes.extend_from_slice(base_assumptions_id.as_bytes());
    bytes.extend_from_slice(query_id.as_bytes());
    bytes.extend_from_slice(config_id.as_bytes());
    bytes.extend_from_slice(result_receipt_id.as_bytes());
    domain_hash(COUNTERFACTUAL_PROVENANCE_DOMAIN, &bytes)
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilityActivationAssumptionsV1, CapabilityCounterfactualConfigV1,
        CapabilityCounterfactualQueryV1, CapabilityDefinitionV1, CapabilityGraphSnapshotV1,
        CapabilityRequirementV1, derive_capability_activation_closure,
        derive_capability_counterfactual_frontier,
    };

    fn leaf(name: &str) -> CapabilityDefinitionV1 {
        CapabilityDefinitionV1::new("org.example", name, None).unwrap()
    }

    fn activation_fixture() -> CapabilityActivationClosureV1 {
        let dependency = leaf("dependency");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::leaf(dependency.id())),
        )
        .unwrap();
        let target_id = target.id();
        let graph = CapabilityGraphSnapshotV1::new(vec![dependency.clone(), target])
            .unwrap()
            .validate()
            .unwrap();
        let assumptions = CapabilityActivationAssumptionsV1::new(
            &graph,
            vec![dependency.id()],
            vec![target_id],
        )
        .unwrap()
        .validate(&graph)
        .unwrap();
        derive_capability_activation_closure(&graph, &assumptions).unwrap()
    }

    #[test]
    fn activation_provenance_is_stable_and_rebinds_exact_output() {
        let closure = activation_fixture();
        let left = CapabilityActivationProvenanceV1::from_closure(&closure);
        let right = CapabilityActivationProvenanceV1::from_closure(&closure);
        assert_eq!(left, right);
        assert_eq!(left.algorithm_semantics(), CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1);
        assert_eq!(left.algorithm_id(), capability_activation_algorithm_id_v1());
        left.validate_against(&closure).unwrap();
    }

    #[test]
    fn activation_algorithm_semantics_substitution_fails_closed() {
        let closure = activation_fixture();
        let mut provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
        provenance.algorithm_semantics = "future-activation-semantics-v2".to_owned();
        assert_eq!(
            provenance.validate_against(&closure),
            Err(CapabilityAnalysisProvenanceError::UnsupportedActivationAlgorithmSemantics(
                "future-activation-semantics-v2".to_owned(),
            ))
        );
    }

    #[test]
    fn activation_result_receipt_substitution_fails_closed() {
        let closure = activation_fixture();
        let mut provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
        provenance.result_receipt_id = CapabilityActivationClosureReceiptId([9; 32]);
        assert_eq!(
            provenance.validate_against(&closure),
            Err(CapabilityAnalysisProvenanceError::ActivationResultReceiptMismatch)
        );
    }

    #[test]
    fn counterfactual_provenance_binds_algorithm_query_config_and_result() {
        let dependency = leaf("dependency");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::leaf(dependency.id())),
        )
        .unwrap();
        let target_id = target.id();
        let graph = CapabilityGraphSnapshotV1::new(vec![dependency, target])
            .unwrap()
            .validate()
            .unwrap();
        let assumptions = CapabilityActivationAssumptionsV1::new(&graph, vec![], vec![target_id])
            .unwrap()
            .validate(&graph)
            .unwrap();
        let query = CapabilityCounterfactualQueryV1::new(&graph, &assumptions, vec![target_id])
            .unwrap()
            .validate(&graph, &assumptions)
            .unwrap();
        let config = CapabilityCounterfactualConfigV1::new(8, 2, 16, 100)
            .unwrap()
            .validate()
            .unwrap();
        let frontier = derive_capability_counterfactual_frontier(
            &graph,
            &assumptions,
            &query,
            &config,
        )
        .unwrap();

        let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
        assert_eq!(
            provenance.algorithm_semantics(),
            CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1
        );
        assert_eq!(provenance.algorithm_id(), capability_counterfactual_algorithm_id_v1());
        assert_eq!(provenance.query_id(), query.id());
        assert_eq!(provenance.config_id(), config.id());
        provenance.validate_against(&frontier).unwrap();
    }
}
