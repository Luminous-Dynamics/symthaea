// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact receipts for deterministic capability analyses.
//!
//! Graph, assumption, query, and configuration identities name the inputs to an
//! analysis. They do not by themselves name the exact derived output. This
//! module gives CC-03B activation closures and CC-03C counterfactual frontiers
//! compact, versioned, domain-separated result identities without making the
//! underlying derived structures transport-authoritative.
//!
//! Core theorem:
//!
//! `AnalysisReceipt != Observation != AvailabilityEvidence != Authority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CapabilityActivationAssumptionsId, CapabilityActivationClosureV1,
    CapabilityCounterfactualConfigId, CapabilityCounterfactualFrontierV1,
    CapabilityCounterfactualQueryId, CapabilityGraphSnapshotId, CapabilityId,
    UnsatisfiedCapabilityRequirementV1,
};

/// Stable semantics/version marker for activation-closure result receipts.
pub const CAPABILITY_ACTIVATION_CLOSURE_RECEIPT_SCHEMA_V1: &str =
    "symthaea-continuity-capability-activation-closure-receipt-v1";
/// Stable semantics/version marker for counterfactual-frontier result receipts.
pub const CAPABILITY_COUNTERFACTUAL_FRONTIER_RECEIPT_SCHEMA_V1: &str =
    "symthaea-continuity-capability-counterfactual-frontier-receipt-v1";

const ACTIVATION_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.continuity.capability-activation-closure-receipt.v1\0";
const FRONTIER_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.continuity.capability-counterfactual-frontier-receipt.v1\0";

/// Exact content identity of one deterministic CC-03B activation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityActivationClosureReceiptId([u8; 32]);

impl CapabilityActivationClosureReceiptId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact content identity of one deterministic CC-03C frontier result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityCounterfactualFrontierReceiptId([u8; 32]);

impl CapabilityCounterfactualFrontierReceiptId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Small serializable receipt naming one exact activation-closure output.
///
/// The receipt is reproducibility metadata only. It does not promote the
/// closure's assumptions into observations or availability facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityActivationClosureReceiptV1 {
    schema_version: String,
    source_snapshot_id: CapabilityGraphSnapshotId,
    assumptions_id: CapabilityActivationAssumptionsId,
    receipt_id: CapabilityActivationClosureReceiptId,
}

impl CapabilityActivationClosureReceiptV1 {
    pub fn from_closure(closure: &CapabilityActivationClosureV1) -> Self {
        let source_snapshot_id = closure.source_snapshot_id();
        let assumptions_id = closure.assumptions_id();
        let receipt_id = CapabilityActivationClosureReceiptId(hash_activation_closure(closure));
        Self {
            schema_version: CAPABILITY_ACTIVATION_CLOSURE_RECEIPT_SCHEMA_V1.to_owned(),
            source_snapshot_id,
            assumptions_id,
            receipt_id,
        }
    }

    pub fn id(&self) -> CapabilityActivationClosureReceiptId {
        self.receipt_id
    }

    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    pub fn assumptions_id(&self) -> CapabilityActivationAssumptionsId {
        self.assumptions_id
    }

    /// Rebind a transported receipt to the exact runtime closure it claims to
    /// name. Validation proves receipt/output equality only.
    pub fn validate_against(
        &self,
        closure: &CapabilityActivationClosureV1,
    ) -> Result<(), CapabilityAnalysisReceiptError> {
        if self.schema_version != CAPABILITY_ACTIVATION_CLOSURE_RECEIPT_SCHEMA_V1 {
            return Err(
                CapabilityAnalysisReceiptError::UnsupportedActivationReceiptSchema(
                    self.schema_version.clone(),
                ),
            );
        }
        if self.source_snapshot_id != closure.source_snapshot_id() {
            return Err(CapabilityAnalysisReceiptError::ActivationSourceMismatch);
        }
        if self.assumptions_id != closure.assumptions_id() {
            return Err(CapabilityAnalysisReceiptError::ActivationAssumptionsMismatch);
        }
        let expected = CapabilityActivationClosureReceiptId(hash_activation_closure(closure));
        if expected != self.receipt_id {
            return Err(CapabilityAnalysisReceiptError::ActivationReceiptIdentityMismatch);
        }
        Ok(())
    }
}

/// Small serializable receipt naming one exact counterfactual-frontier output.
///
/// Query and configuration identities are repeated here intentionally so a
/// stored receipt cannot be detached from its exact scoped question and search
/// completeness boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityCounterfactualFrontierReceiptV1 {
    schema_version: String,
    source_snapshot_id: CapabilityGraphSnapshotId,
    base_assumptions_id: CapabilityActivationAssumptionsId,
    query_id: CapabilityCounterfactualQueryId,
    config_id: CapabilityCounterfactualConfigId,
    receipt_id: CapabilityCounterfactualFrontierReceiptId,
}

impl CapabilityCounterfactualFrontierReceiptV1 {
    pub fn from_frontier(frontier: &CapabilityCounterfactualFrontierV1) -> Self {
        let source_snapshot_id = frontier.source_snapshot_id();
        let base_assumptions_id = frontier.base_assumptions_id();
        let query_id = frontier.query_id();
        let config_id = frontier.config_id();
        let receipt_id =
            CapabilityCounterfactualFrontierReceiptId(hash_counterfactual_frontier(frontier));
        Self {
            schema_version: CAPABILITY_COUNTERFACTUAL_FRONTIER_RECEIPT_SCHEMA_V1.to_owned(),
            source_snapshot_id,
            base_assumptions_id,
            query_id,
            config_id,
            receipt_id,
        }
    }

    pub fn id(&self) -> CapabilityCounterfactualFrontierReceiptId {
        self.receipt_id
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

    pub fn validate_against(
        &self,
        frontier: &CapabilityCounterfactualFrontierV1,
    ) -> Result<(), CapabilityAnalysisReceiptError> {
        if self.schema_version != CAPABILITY_COUNTERFACTUAL_FRONTIER_RECEIPT_SCHEMA_V1 {
            return Err(
                CapabilityAnalysisReceiptError::UnsupportedFrontierReceiptSchema(
                    self.schema_version.clone(),
                ),
            );
        }
        if self.source_snapshot_id != frontier.source_snapshot_id() {
            return Err(CapabilityAnalysisReceiptError::FrontierSourceMismatch);
        }
        if self.base_assumptions_id != frontier.base_assumptions_id() {
            return Err(CapabilityAnalysisReceiptError::FrontierAssumptionsMismatch);
        }
        if self.query_id != frontier.query_id() {
            return Err(CapabilityAnalysisReceiptError::FrontierQueryMismatch);
        }
        if self.config_id != frontier.config_id() {
            return Err(CapabilityAnalysisReceiptError::FrontierConfigMismatch);
        }
        let expected =
            CapabilityCounterfactualFrontierReceiptId(hash_counterfactual_frontier(frontier));
        if expected != self.receipt_id {
            return Err(CapabilityAnalysisReceiptError::FrontierReceiptIdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityAnalysisReceiptError {
    #[error("unsupported activation-closure receipt schema: {0}")]
    UnsupportedActivationReceiptSchema(String),
    #[error("activation receipt source snapshot does not match runtime closure")]
    ActivationSourceMismatch,
    #[error("activation receipt assumptions do not match runtime closure")]
    ActivationAssumptionsMismatch,
    #[error("activation receipt identity does not match exact runtime closure output")]
    ActivationReceiptIdentityMismatch,
    #[error("unsupported counterfactual-frontier receipt schema: {0}")]
    UnsupportedFrontierReceiptSchema(String),
    #[error("frontier receipt source snapshot does not match runtime frontier")]
    FrontierSourceMismatch,
    #[error("frontier receipt base assumptions do not match runtime frontier")]
    FrontierAssumptionsMismatch,
    #[error("frontier receipt query does not match runtime frontier")]
    FrontierQueryMismatch,
    #[error("frontier receipt config does not match runtime frontier")]
    FrontierConfigMismatch,
    #[error("frontier receipt identity does not match exact runtime frontier output")]
    FrontierReceiptIdentityMismatch,
}

fn hash_activation_closure(closure: &CapabilityActivationClosureV1) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, CAPABILITY_ACTIVATION_CLOSURE_RECEIPT_SCHEMA_V1);
    bytes.extend_from_slice(closure.source_snapshot_id().as_bytes());
    bytes.extend_from_slice(closure.assumptions_id().as_bytes());
    put_ids(&mut bytes, closure.initially_available());

    put_len(&mut bytes, closure.activation_rounds().len());
    for round in closure.activation_rounds() {
        put_u64(&mut bytes, round.ordinal());
        put_ids(&mut bytes, round.activated());
    }

    put_ids(&mut bytes, closure.available_after_closure());
    put_len(&mut bytes, closure.blocked_activatable().len());
    for blocked in closure.blocked_activatable() {
        bytes.extend_from_slice(blocked.capability_id().as_bytes());
        encode_unsatisfied(&mut bytes, blocked.unsatisfied());
    }

    domain_hash(ACTIVATION_RECEIPT_DOMAIN, &bytes)
}

fn hash_counterfactual_frontier(frontier: &CapabilityCounterfactualFrontierV1) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(
        &mut bytes,
        CAPABILITY_COUNTERFACTUAL_FRONTIER_RECEIPT_SCHEMA_V1,
    );
    bytes.extend_from_slice(frontier.source_snapshot_id().as_bytes());
    bytes.extend_from_slice(frontier.base_assumptions_id().as_bytes());
    bytes.extend_from_slice(frontier.query_id().as_bytes());
    bytes.extend_from_slice(frontier.config_id().as_bytes());
    put_ids(&mut bytes, frontier.base_available_after_closure());
    put_u64_raw(&mut bytes, frontier.simulations_evaluated());

    put_len(&mut bytes, frontier.targets().len());
    for target in frontier.targets() {
        bytes.extend_from_slice(target.target_capability_id().as_bytes());
        put_ids(&mut bytes, target.dependency_universe());
        put_u64(&mut bytes, target.complete_through_support_width());
        put_len(&mut bytes, target.options().len());
        for option in target.options() {
            bytes.extend_from_slice(option.target_capability_id().as_bytes());
            put_ids(&mut bytes, option.assumed_support());
            bytes.extend_from_slice(option.counterfactual_assumptions_id().as_bytes());
            put_ids(&mut bytes, option.marginally_activated());
        }
    }

    domain_hash(FRONTIER_RECEIPT_DOMAIN, &bytes)
}

fn encode_unsatisfied(out: &mut Vec<u8>, requirement: &UnsatisfiedCapabilityRequirementV1) {
    match requirement {
        UnsatisfiedCapabilityRequirementV1::Leaf { capability_id } => {
            out.push(1);
            out.extend_from_slice(capability_id.as_bytes());
        }
        UnsatisfiedCapabilityRequirementV1::AllOf { requirements } => {
            out.push(2);
            put_len(out, requirements.len());
            for requirement in requirements {
                encode_unsatisfied(out, requirement);
            }
        }
        UnsatisfiedCapabilityRequirementV1::AnyOf { requirements } => {
            out.push(3);
            put_len(out, requirements.len());
            for requirement in requirements {
                encode_unsatisfied(out, requirement);
            }
        }
    }
}

fn put_ids(out: &mut Vec<u8>, ids: &[CapabilityId]) {
    put_len(out, ids.len());
    for id in ids {
        out.extend_from_slice(id.as_bytes());
    }
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    put_u64(out, len);
}

fn put_u64(out: &mut Vec<u8>, value: usize) {
    let value = u64::try_from(value).expect("usize always fits u64 on supported Rust targets");
    put_u64_raw(out, value);
}

fn put_u64_raw(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
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

    #[test]
    fn activation_receipt_is_stable_and_rebinds_exact_output() {
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
        let assumptions =
            CapabilityActivationAssumptionsV1::new(&graph, vec![dependency.id()], vec![target_id])
                .unwrap()
                .validate(&graph)
                .unwrap();
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        let left = CapabilityActivationClosureReceiptV1::from_closure(&closure);
        let right = CapabilityActivationClosureReceiptV1::from_closure(&closure);
        assert_eq!(left, right);
        left.validate_against(&closure).unwrap();
    }

    #[test]
    fn tampered_activation_receipt_fails_closed() {
        let target = leaf("target");
        let target_id = target.id();
        let graph = CapabilityGraphSnapshotV1::new(vec![target])
            .unwrap()
            .validate()
            .unwrap();
        let assumptions = CapabilityActivationAssumptionsV1::new(&graph, vec![], vec![target_id])
            .unwrap()
            .validate(&graph)
            .unwrap();
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();
        let mut receipt = CapabilityActivationClosureReceiptV1::from_closure(&closure);
        receipt.receipt_id = CapabilityActivationClosureReceiptId([7; 32]);
        assert_eq!(
            receipt.validate_against(&closure),
            Err(CapabilityAnalysisReceiptError::ActivationReceiptIdentityMismatch)
        );
    }

    #[test]
    fn counterfactual_receipt_binds_exact_frontier() {
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
        let frontier =
            derive_capability_counterfactual_frontier(&graph, &assumptions, &query, &config)
                .unwrap();

        let receipt = CapabilityCounterfactualFrontierReceiptV1::from_frontier(&frontier);
        receipt.validate_against(&frontier).unwrap();
        assert_eq!(receipt.query_id(), query.id());
        assert_eq!(receipt.config_id(), config.id());
    }
}
