// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consumer-side quorum verification for independent source-origin observers.
//!
//! This module does not equate multiple signatures with independence. A
//! consumer-owned policy pins each operator to a distinct trust profile and
//! declared acquisition failure domain, requires both operator and failure-
//! domain diversity, and rejects any signed ancestry disagreement rather than
//! resolving it by majority vote.
//!
//! Witnesses corroborate the graph and the exact fourth-authority origin
//! receipt. They do not prove universal source truth or completeness beyond
//! the independently acquired metadata and trust assumptions represented by
//! their signed observations.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::qualification_dsse::{
    parse_dsse_envelope_v1, parse_strict_json_bounded, DSSE_PAYLOAD_TYPE_V1,
    IN_TOTO_STATEMENT_TYPE_V1,
};
use crate::qualification_trust::PinnedEd25519QualificationVerifierV1;
use crate::source_lineage_graph::{
    VerifiedSourceLineageGraphV1, VerifiedSourceLineageNodeBindingV1,
};
use crate::source_origin_receipt::{
    SourceOriginArtifactInputV1, SourceOriginRejectV1, VerifiedSourceOriginSetV1,
};

pub const SOURCE_ORIGIN_WITNESS_VERIFIER_ID_V1: &str =
    "mace-independent-source-origin-witness";
pub const MACE_SOURCE_ORIGIN_WITNESS_PREDICATE_TYPE_V1: &str =
    "https://luminousdynamics.io/attestations/mace/source-origin-witness/v1";
pub const SOURCE_ORIGIN_WITNESS_POLICY_SCHEMA_V1: &str =
    "mace_source_origin_witness_policy_v1";
pub const SOURCE_ORIGIN_WITNESS_RECEIPT_SCHEMA_V1: &str =
    "mace_source_origin_witness_receipt_v1";

const MAX_WITNESSES_V1: usize = 16;
const MAX_WITNESS_ENVELOPES_V1: usize = 100_000;
const MAX_WITNESS_ENVELOPE_BYTES_V1: usize = 2 * 1024 * 1024;
const MAX_WITNESS_PAYLOAD_BYTES_V1: usize = 1024 * 1024;
const MAX_OBSERVATION_AGE_SECS_V1: u64 = 365 * 24 * 60 * 60;
const MAX_RECEIPT_LIFETIME_SECS_V1: u64 = 31 * 24 * 60 * 60;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceOriginWitnessRejectV1 {
    InvalidPolicy,
    InvalidVerifierSet,
    InvalidEnvelope,
    InvalidJson,
    InvalidReceipt,
    InvalidDigest,
    WrongPredicateType,
    SubjectMismatch,
    BindingMismatch,
    ProfileMismatch,
    WrongVerifierRole,
    SharedAuthority,
    InvalidSignature,
    UnknownOrRevokedSigner,
    NotYetValid,
    Expired,
    ObservationFromFuture,
    ObservationStale,
    MissingWitness,
    UnexpectedWitness,
    DuplicateWitness,
    TooFewWitnesses,
    TooFewFailureDomains,
    SharedObservationRecord,
    ObservedSourceMismatch,
    AncestryDisagreement,
    OriginReceiptMismatch,
    Origin(SourceOriginRejectV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceOriginWitnessOperatorV1 {
    /// Stable operator identity provisioned by the consumer, not self-selected
    /// by the candidate memory system.
    pub operator_id: String,
    /// Consumer-defined acquisition/control-plane failure-domain commitment.
    /// Distinct names do not prove independence; the consumer must govern this
    /// registry and avoid aliases for one organization or shared control plane.
    pub failure_domain_sha256: String,
    /// Exact out-of-band pinned trust-profile digest for this operator.
    pub verifier_profile_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceOriginWitnessPolicyV1 {
    pub schema_version: String,
    pub graph_sha256: String,
    pub minimum_operators: u8,
    pub minimum_failure_domains: u8,
    pub maximum_observation_age_secs: u64,
    /// Strictly sorted by operator_id.
    pub operators: Vec<SourceOriginWitnessOperatorV1>,
}

impl SourceOriginWitnessPolicyV1 {
    pub fn validate(&self) -> Result<(), SourceOriginWitnessRejectV1> {
        if self.schema_version != SOURCE_ORIGIN_WITNESS_POLICY_SCHEMA_V1
            || !valid_sha256(&self.graph_sha256)
            || self.operators.len() < 2
            || self.operators.len() > MAX_WITNESSES_V1
            || self.minimum_operators < 2
            || self.minimum_operators as usize > self.operators.len()
            || self.minimum_failure_domains < 2
            || self.minimum_failure_domains > self.minimum_operators
            || self.maximum_observation_age_secs == 0
            || self.maximum_observation_age_secs > MAX_OBSERVATION_AGE_SECS_V1
        {
            return Err(SourceOriginWitnessRejectV1::InvalidPolicy);
        }
        let mut profiles = BTreeSet::new();
        let mut failure_domains = BTreeSet::new();
        for (index, operator) in self.operators.iter().enumerate() {
            if !valid_atom(&operator.operator_id, 256)
                || !valid_sha256(&operator.failure_domain_sha256)
                || !valid_sha256(&operator.verifier_profile_sha256)
                || !profiles.insert(operator.verifier_profile_sha256.as_str())
            {
                return Err(SourceOriginWitnessRejectV1::InvalidPolicy);
            }
            failure_domains.insert(operator.failure_domain_sha256.as_str());
            if index > 0
                && self.operators[index - 1].operator_id >= operator.operator_id
            {
                return Err(SourceOriginWitnessRejectV1::InvalidPolicy);
            }
        }
        if failure_domains.len() < self.minimum_failure_domains as usize {
            return Err(SourceOriginWitnessRejectV1::InvalidPolicy);
        }
        Ok(())
    }

    /// Domain-separated digest of the complete consumer-owned quorum policy.
    pub fn digest(&self) -> Result<String, SourceOriginWitnessRejectV1> {
        self.validate()?;
        let mut h = Sha256::new();
        h.update(b"luminous:mace:source-origin-witness-policy:v1\0");
        hash_field(&mut h, &self.schema_version);
        hash_field(&mut h, &self.graph_sha256);
        h.update([self.minimum_operators]);
        h.update([self.minimum_failure_domains]);
        h.update(self.maximum_observation_age_secs.to_be_bytes());
        h.update((self.operators.len() as u64).to_be_bytes());
        for operator in &self.operators {
            hash_field(&mut h, &operator.operator_id);
            hash_field(&mut h, &operator.failure_domain_sha256);
            hash_field(&mut h, &operator.verifier_profile_sha256);
        }
        Ok(hex_digest(&h.finalize()))
    }
}

pub struct SourceOriginWitnessVerifierV1<'a> {
    pub operator_id: &'a str,
    pub verifier: &'a PinnedEd25519QualificationVerifierV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceOriginWitnessEnvelopeInputV1<'a> {
    pub operator_id: &'a str,
    pub evidence_id: &'a str,
    /// Exact UTF-8 DSSE envelope. The hash of these exact bytes is retained.
    pub envelope_json: &'a str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceOriginWitnessReceiptV1 {
    run_id: String,
    query_invocation_id: String,
    graph_sha256: String,
    source_origin_receipt_sha256: String,
    operator_id: String,
    failure_domain_sha256: String,
    evidence_id: String,
    source_id: String,
    source_sha256: String,
    payload_sha256: String,
    evidence_class_sha256: String,
    observed_parent_evidence_ids: Vec<String>,
    derivation_receipt_sha256: Option<String>,
    common_cause_group_sha256: Vec<String>,
    publisher_identity_sha256: String,
    acquisition_record_sha256: String,
    source_revision_sha256: String,
    observed_at_unix: u64,
    issued_at_unix: u64,
    expires_at_unix: u64,
    status: String,
    issuer_profile_sha256: String,
    key_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VerifiedSourceOriginWitnessV1 {
    receipt: SourceOriginWitnessReceiptV1,
    envelope_sha256: String,
    pae: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSourceOriginWitnessSetV1 {
    graph_sha256: String,
    source_origin_receipt_bundle_sha256: String,
    policy_sha256: String,
    witnesses: Vec<VerifiedSourceOriginWitnessV1>,
    witness_operator_ids: Vec<String>,
    failure_domain_sha256: Vec<String>,
}

impl VerifiedSourceOriginWitnessSetV1 {
    pub fn len(&self) -> usize { self.witnesses.len() }
    pub fn is_empty(&self) -> bool { self.witnesses.is_empty() }
    pub fn graph_sha256(&self) -> &str { &self.graph_sha256 }
    pub fn policy_sha256(&self) -> &str { &self.policy_sha256 }
    pub fn witness_operator_ids(&self) -> &[String] { &self.witness_operator_ids }
    pub fn failure_domain_sha256(&self) -> &[String] { &self.failure_domain_sha256 }

    /// Revalidate the four-authority source-origin result, exact artifacts,
    /// current witness profiles, per-node quorum and every witness signature
    /// at one trusted use-time sample. A stale quorum token cannot outlive the
    /// original source-origin receipts or current trust profiles.
    pub fn validate_at_use(
        &self,
        graph: &VerifiedSourceLineageGraphV1,
        origin: &VerifiedSourceOriginSetV1,
        artifacts: &[SourceOriginArtifactInputV1<'_>],
        current_origin_profile: &PinnedEd25519QualificationVerifierV1,
        policy: &SourceOriginWitnessPolicyV1,
        verifiers: &[SourceOriginWitnessVerifierV1<'_>],
        now_unix: u64,
    ) -> Result<(), SourceOriginWitnessRejectV1> {
        let policy_sha256 = policy.digest()?;
        if self.graph_sha256 != graph.graph_sha256()
            || policy.graph_sha256 != graph.graph_sha256()
            || self.policy_sha256 != policy_sha256
        {
            return Err(SourceOriginWitnessRejectV1::BindingMismatch);
        }
        origin.validate_at_use(graph, artifacts, current_origin_profile, now_unix)
            .map_err(SourceOriginWitnessRejectV1::Origin)?;
        validate_verifier_set(graph, origin, current_origin_profile, policy, verifiers)?;
        let origin_bundle_sha256 = source_origin_bundle_sha256(origin);
        if origin_bundle_sha256 != self.source_origin_receipt_bundle_sha256 {
            return Err(SourceOriginWitnessRejectV1::OriginReceiptMismatch);
        }
        let verifier_map = verifier_map(verifiers)?;
        let mut cross_operator_keys = BTreeMap::<String, String>::new();
        for item in &self.witnesses {
            let receipt = &item.receipt;
            let operator = policy.operators.iter()
                .find(|operator| operator.operator_id == receipt.operator_id)
                .ok_or(SourceOriginWitnessRejectV1::UnexpectedWitness)?;
            let node = graph.node_bindings().iter()
                .find(|node| node.evidence_id == receipt.evidence_id)
                .ok_or(SourceOriginWitnessRejectV1::UnexpectedWitness)?;
            validate_witness_binding(
                receipt, graph, origin, node, operator, policy, now_unix,
            )?;
            let verifier = verifier_map.get(receipt.operator_id.as_str())
                .ok_or(SourceOriginWitnessRejectV1::InvalidVerifierSet)?;
            if graph.authority_key_ids().iter().any(|key| key == &receipt.key_id)
                || origin.signer_key_ids().iter().any(|key| key == &receipt.key_id)
            {
                return Err(SourceOriginWitnessRejectV1::SharedAuthority);
            }
            if let Some(owner) = cross_operator_keys.get(&receipt.key_id) {
                if owner != &receipt.operator_id {
                    return Err(SourceOriginWitnessRejectV1::SharedAuthority);
                }
            } else {
                cross_operator_keys.insert(receipt.key_id.clone(), receipt.operator_id.clone());
            }
            validate_witness_time_and_signature(
                receipt, &item.pae, &item.signature, verifier, now_unix,
            )?;
        }
        validate_per_node_quorum(graph, policy, &self.witnesses)?;
        Ok(())
    }
}

struct ExpectedWitnessBindingV1<'a> {
    graph: &'a VerifiedSourceLineageGraphV1,
    origin: &'a VerifiedSourceOriginSetV1,
    node: &'a VerifiedSourceLineageNodeBindingV1,
    operator: &'a SourceOriginWitnessOperatorV1,
    policy: &'a SourceOriginWitnessPolicyV1,
}

pub fn verify_source_origin_witnesses_v1(
    graph: &VerifiedSourceLineageGraphV1,
    origin: &VerifiedSourceOriginSetV1,
    artifacts: &[SourceOriginArtifactInputV1<'_>],
    current_origin_profile: &PinnedEd25519QualificationVerifierV1,
    policy: &SourceOriginWitnessPolicyV1,
    verifiers: &[SourceOriginWitnessVerifierV1<'_>],
    envelopes: &[SourceOriginWitnessEnvelopeInputV1<'_>],
    now_unix: u64,
) -> Result<VerifiedSourceOriginWitnessSetV1, SourceOriginWitnessRejectV1> {
    let policy_sha256 = policy.digest()?;
    if policy.graph_sha256 != graph.graph_sha256() {
        return Err(SourceOriginWitnessRejectV1::BindingMismatch);
    }
    origin.validate_at_use(graph, artifacts, current_origin_profile, now_unix)
        .map_err(SourceOriginWitnessRejectV1::Origin)?;
    validate_verifier_set(graph, origin, current_origin_profile, policy, verifiers)?;

    let max_envelopes = graph.node_bindings().len()
        .checked_mul(policy.operators.len())
        .ok_or(SourceOriginWitnessRejectV1::InvalidPolicy)?;
    if max_envelopes > MAX_WITNESS_ENVELOPES_V1
        || envelopes.len() > max_envelopes
    {
        return Err(SourceOriginWitnessRejectV1::UnexpectedWitness);
    }

    let verifier_map = verifier_map(verifiers)?;
    let operator_map = policy.operators.iter()
        .map(|operator| (operator.operator_id.as_str(), operator))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut verified = Vec::with_capacity(envelopes.len());
    let mut cross_operator_keys = BTreeMap::<String, String>::new();

    for input in envelopes {
        if !seen.insert((input.evidence_id, input.operator_id)) {
            return Err(SourceOriginWitnessRejectV1::DuplicateWitness);
        }
        let node = graph.node_bindings().iter()
            .find(|node| node.evidence_id == input.evidence_id)
            .ok_or(SourceOriginWitnessRejectV1::UnexpectedWitness)?;
        let operator = operator_map.get(input.operator_id)
            .ok_or(SourceOriginWitnessRejectV1::UnexpectedWitness)?;
        let verifier = verifier_map.get(input.operator_id)
            .ok_or(SourceOriginWitnessRejectV1::InvalidVerifierSet)?;
        let binding = ExpectedWitnessBindingV1 {
            graph, origin, node, operator, policy,
        };
        let item = verify_one_witness_receipt_v1(
            input.envelope_json, &binding, verifier, now_unix,
        )?;
        if graph.authority_key_ids().iter().any(|key| key == &item.receipt.key_id)
            || origin.signer_key_ids().iter().any(|key| key == &item.receipt.key_id)
        {
            return Err(SourceOriginWitnessRejectV1::SharedAuthority);
        }
        if let Some(owner) = cross_operator_keys.get(&item.receipt.key_id) {
            if owner != &item.receipt.operator_id {
                return Err(SourceOriginWitnessRejectV1::SharedAuthority);
            }
        } else {
            cross_operator_keys.insert(
                item.receipt.key_id.clone(), item.receipt.operator_id.clone(),
            );
        }
        verified.push(item);
    }

    verified.sort_by(|a, b| {
        (&a.receipt.evidence_id, &a.receipt.operator_id)
            .cmp(&(&b.receipt.evidence_id, &b.receipt.operator_id))
    });
    validate_per_node_quorum(graph, policy, &verified)?;
    let mut witness_operator_ids = verified.iter()
        .map(|item| item.receipt.operator_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter().collect::<Vec<_>>();
    witness_operator_ids.sort();
    let mut failure_domain_sha256 = verified.iter()
        .map(|item| item.receipt.failure_domain_sha256.clone())
        .collect::<BTreeSet<_>>()
        .into_iter().collect::<Vec<_>>();
    failure_domain_sha256.sort();

    Ok(VerifiedSourceOriginWitnessSetV1 {
        graph_sha256: graph.graph_sha256().to_owned(),
        source_origin_receipt_bundle_sha256: source_origin_bundle_sha256(origin),
        policy_sha256,
        witnesses: verified,
        witness_operator_ids,
        failure_domain_sha256,
    })
}

fn validate_verifier_set(
    graph: &VerifiedSourceLineageGraphV1,
    origin: &VerifiedSourceOriginSetV1,
    current_origin_profile: &PinnedEd25519QualificationVerifierV1,
    policy: &SourceOriginWitnessPolicyV1,
    verifiers: &[SourceOriginWitnessVerifierV1<'_>],
) -> Result<(), SourceOriginWitnessRejectV1> {
    if verifiers.len() != policy.operators.len() {
        return Err(SourceOriginWitnessRejectV1::InvalidVerifierSet);
    }
    let mut profiles = BTreeSet::new();
    if current_origin_profile.profile_sha256() != origin.origin_profile_sha256() {
        return Err(SourceOriginWitnessRejectV1::ProfileMismatch);
    }
    let mut keys = graph.authority_key_ids().iter().cloned()
        .collect::<BTreeSet<_>>();
    keys.extend(current_origin_profile.key_ids());
    for operator in &policy.operators {
        if graph.authority_profile_sha256().contains(&operator.verifier_profile_sha256)
            || origin.origin_profile_sha256() == operator.verifier_profile_sha256
            || !profiles.insert(operator.verifier_profile_sha256.as_str())
        {
            return Err(SourceOriginWitnessRejectV1::SharedAuthority);
        }
        let verifier = verifiers.iter()
            .find(|item| item.operator_id == operator.operator_id)
            .ok_or(SourceOriginWitnessRejectV1::InvalidVerifierSet)?;
        if verifier.verifier.verifier_id() != SOURCE_ORIGIN_WITNESS_VERIFIER_ID_V1 {
            return Err(SourceOriginWitnessRejectV1::WrongVerifierRole);
        }
        if verifier.verifier.profile_sha256() != operator.verifier_profile_sha256 {
            return Err(SourceOriginWitnessRejectV1::ProfileMismatch);
        }
        // Reject reused keys across *all* provisioned profiles, not only keys
        // that happened to sign this run's observations.
        for key_id in verifier.verifier.key_ids() {
            if !keys.insert(key_id) {
                return Err(SourceOriginWitnessRejectV1::SharedAuthority);
            }
        }
    }
    let prior_profiles = graph.authority_profile_sha256().iter()
        .cloned().chain(std::iter::once(origin.origin_profile_sha256().to_owned()))
        .collect::<BTreeSet<_>>();
    for profile in profiles {
        if prior_profiles.contains(profile) {
            return Err(SourceOriginWitnessRejectV1::SharedAuthority);
        }
    }
    Ok(())
}

fn verifier_map<'a>(
    verifiers: &'a [SourceOriginWitnessVerifierV1<'a>],
) -> Result<BTreeMap<&'a str, &'a PinnedEd25519QualificationVerifierV1>, SourceOriginWitnessRejectV1> {
    let mut map = BTreeMap::new();
    for item in verifiers {
        if item.operator_id.trim().is_empty()
            || map.insert(item.operator_id, item.verifier).is_some()
        {
            return Err(SourceOriginWitnessRejectV1::InvalidVerifierSet);
        }
    }
    Ok(map)
}

fn validate_witness_metadata_agreement(
    witnesses: &[&VerifiedSourceOriginWitnessV1],
) -> Result<(), SourceOriginWitnessRejectV1> {
    let mut acquisition_records = BTreeSet::new();
    let mut publisher_identity: Option<&str> = None;
    let mut source_revision: Option<&str> = None;
    for item in witnesses {
        let receipt = &item.receipt;
        if let Some(expected) = publisher_identity {
            if expected != receipt.publisher_identity_sha256.as_str() {
                return Err(SourceOriginWitnessRejectV1::ObservedSourceMismatch);
            }
        } else {
            publisher_identity = Some(&receipt.publisher_identity_sha256);
        }
        if let Some(expected) = source_revision {
            if expected != receipt.source_revision_sha256.as_str() {
                return Err(SourceOriginWitnessRejectV1::ObservedSourceMismatch);
            }
        } else {
            source_revision = Some(&receipt.source_revision_sha256);
        }
        if !acquisition_records.insert(receipt.acquisition_record_sha256.as_str()) {
            return Err(SourceOriginWitnessRejectV1::SharedObservationRecord);
        }
    }
    Ok(())
}

fn validate_per_node_quorum(
    graph: &VerifiedSourceLineageGraphV1,
    policy: &SourceOriginWitnessPolicyV1,
    witnesses: &[VerifiedSourceOriginWitnessV1],
) -> Result<(), SourceOriginWitnessRejectV1> {
    for node in graph.node_bindings() {
        let mut operators = BTreeSet::new();
        let mut domains = BTreeSet::new();
        let node_witnesses = witnesses.iter()
            .filter(|item| item.receipt.evidence_id == node.evidence_id)
            .collect::<Vec<_>>();
        validate_witness_metadata_agreement(&node_witnesses)?;
        for item in &node_witnesses {
            operators.insert(item.receipt.operator_id.as_str());
            domains.insert(item.receipt.failure_domain_sha256.as_str());
        }
        if operators.len() < policy.minimum_operators as usize {
            return Err(SourceOriginWitnessRejectV1::TooFewWitnesses);
        }
        if domains.len() < policy.minimum_failure_domains as usize {
            return Err(SourceOriginWitnessRejectV1::TooFewFailureDomains);
        }
    }
    Ok(())
}

fn verify_one_witness_receipt_v1(
    envelope_json: &str,
    binding: &ExpectedWitnessBindingV1<'_>,
    verifier: &PinnedEd25519QualificationVerifierV1,
    now_unix: u64,
) -> Result<VerifiedSourceOriginWitnessV1, SourceOriginWitnessRejectV1> {
    if envelope_json.len() > MAX_WITNESS_ENVELOPE_BYTES_V1 {
        return Err(SourceOriginWitnessRejectV1::InvalidEnvelope);
    }
    let envelope = parse_dsse_envelope_v1(envelope_json, DSSE_PAYLOAD_TYPE_V1)
        .map_err(|_| SourceOriginWitnessRejectV1::InvalidEnvelope)?;
    if envelope.payload.len() > MAX_WITNESS_PAYLOAD_BYTES_V1 {
        return Err(SourceOriginWitnessRejectV1::InvalidEnvelope);
    }
    let statement_value = parse_strict_json_bounded(
        &envelope.payload, MAX_WITNESS_PAYLOAD_BYTES_V1,
    ).map_err(|_| SourceOriginWitnessRejectV1::InvalidJson)?;
    let statement = statement_value.as_object()
        .ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
    const STATEMENT_FIELDS: &[&str] = &["_type", "subject", "predicateType", "predicate"];
    if statement.len() != STATEMENT_FIELDS.len()
        || statement.keys().any(|field| !STATEMENT_FIELDS.contains(&field.as_str()))
    {
        return Err(SourceOriginWitnessRejectV1::InvalidJson);
    }
    if statement.get("_type").and_then(Value::as_str) != Some(IN_TOTO_STATEMENT_TYPE_V1)
        || statement.get("predicateType").and_then(Value::as_str)
            != Some(MACE_SOURCE_ORIGIN_WITNESS_PREDICATE_TYPE_V1)
    {
        return Err(SourceOriginWitnessRejectV1::WrongPredicateType);
    }
    let subjects = statement.get("subject").and_then(Value::as_array)
        .ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
    if subjects.len() != 1 {
        return Err(SourceOriginWitnessRejectV1::SubjectMismatch);
    }
    let subject = subjects[0].as_object().ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
    let digest = subject.get("digest").and_then(Value::as_object)
        .ok_or(SourceOriginWitnessRejectV1::SubjectMismatch)?;
    let expected_subject_name = format!(
        "urn:luminous:mace:source-origin-witness-node:v1:{}:{}",
        binding.operator.operator_id, binding.node.evidence_id,
    );
    if subject.len() != 2 || digest.len() != 1
        || subject.get("name").and_then(Value::as_str) != Some(expected_subject_name.as_str())
        || digest.get("sha256").and_then(Value::as_str)
            != Some(binding.node.source_sha256.as_str())
    {
        return Err(SourceOriginWitnessRejectV1::SubjectMismatch);
    }

    let predicate = statement.get("predicate").ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
    let receipt = parse_witness_predicate(predicate)?;
    validate_witness_binding(
        &receipt, binding.graph, binding.origin, binding.node,
        binding.operator, binding.policy, now_unix,
    )?;
    if envelope.key_id_hint.as_deref() != Some(receipt.key_id.as_str()) {
        return Err(SourceOriginWitnessRejectV1::BindingMismatch);
    }
    validate_witness_time(&receipt, binding.policy, now_unix)?;
    verifier.verify_message(
        &receipt.issuer_profile_sha256,
        &receipt.key_id,
        &envelope.pae,
        &envelope.signature,
        receipt.issued_at_unix,
        now_unix,
    ).map_err(map_signature_error)?;

    Ok(VerifiedSourceOriginWitnessV1 {
        envelope_sha256: sha256_hex(envelope_json.as_bytes()),
        receipt,
        pae: envelope.pae,
        signature: envelope.signature,
    })
}

fn validate_witness_binding(
    receipt: &SourceOriginWitnessReceiptV1,
    graph: &VerifiedSourceLineageGraphV1,
    origin: &VerifiedSourceOriginSetV1,
    node: &VerifiedSourceLineageNodeBindingV1,
    operator: &SourceOriginWitnessOperatorV1,
    policy: &SourceOriginWitnessPolicyV1,
    now_unix: u64,
) -> Result<(), SourceOriginWitnessRejectV1> {
    if receipt.run_id != graph.run_id()
        || receipt.query_invocation_id != graph.query_invocation_id()
        || receipt.graph_sha256 != graph.graph_sha256()
        || receipt.operator_id != operator.operator_id
        || receipt.failure_domain_sha256 != operator.failure_domain_sha256
        || receipt.evidence_id != node.evidence_id
        || receipt.source_origin_receipt_sha256
            != origin.envelope_sha256_for(&node.evidence_id)
                .ok_or(SourceOriginWitnessRejectV1::OriginReceiptMismatch)?
        || receipt.issuer_profile_sha256 != operator.verifier_profile_sha256
        || receipt.status != "observed_consistent"
    {
        return Err(SourceOriginWitnessRejectV1::BindingMismatch);
    }
    if receipt.source_id != node.source_id
        || receipt.source_sha256 != node.source_sha256
        || receipt.payload_sha256 != node.payload_sha256
        || receipt.evidence_class_sha256 != node.evidence_class_sha256
    {
        return Err(SourceOriginWitnessRejectV1::ObservedSourceMismatch);
    }
    validate_observed_ancestry(receipt, node)?;
    validate_witness_time(receipt, policy, now_unix)
}

fn validate_observed_ancestry(
    receipt: &SourceOriginWitnessReceiptV1,
    node: &VerifiedSourceLineageNodeBindingV1,
) -> Result<(), SourceOriginWitnessRejectV1> {
    if receipt.observed_parent_evidence_ids != node.parent_evidence_ids
        || receipt.derivation_receipt_sha256 != node.derivation_receipt_sha256
        || receipt.common_cause_group_sha256 != node.common_cause_group_sha256
    {
        return Err(SourceOriginWitnessRejectV1::AncestryDisagreement);
    }
    Ok(())
}

fn validate_witness_time(
    receipt: &SourceOriginWitnessReceiptV1,
    policy: &SourceOriginWitnessPolicyV1,
    now_unix: u64,
) -> Result<(), SourceOriginWitnessRejectV1> {
    if receipt.observed_at_unix > now_unix {
        return Err(SourceOriginWitnessRejectV1::ObservationFromFuture);
    }
    if now_unix.saturating_sub(receipt.observed_at_unix)
        > policy.maximum_observation_age_secs
    {
        return Err(SourceOriginWitnessRejectV1::ObservationStale);
    }
    if receipt.issued_at_unix > now_unix {
        return Err(SourceOriginWitnessRejectV1::NotYetValid);
    }
    if receipt.issued_at_unix < receipt.observed_at_unix
        || receipt.issued_at_unix >= receipt.expires_at_unix
        || now_unix >= receipt.expires_at_unix
        || receipt.expires_at_unix.saturating_sub(receipt.issued_at_unix)
            > MAX_RECEIPT_LIFETIME_SECS_V1
    {
        return Err(SourceOriginWitnessRejectV1::Expired);
    }
    Ok(())
}

fn parse_witness_predicate(value: &Value) -> Result<SourceOriginWitnessReceiptV1, SourceOriginWitnessRejectV1> {
    let object = value.as_object().ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
    const FIELDS: &[&str] = &[
        "schema_version", "run_id", "query_invocation_id", "graph_sha256",
        "source_origin_receipt_sha256", "operator_id", "failure_domain_sha256",
        "evidence_id", "source_id", "source_sha256", "payload_sha256",
        "evidence_class_sha256", "observed_parent_evidence_ids",
        "derivation_receipt_sha256", "common_cause_group_sha256",
        "publisher_identity_sha256", "acquisition_record_sha256",
        "source_revision_sha256", "observed_at_unix", "issued_at_unix",
        "expires_at_unix", "status", "issuer_profile_sha256", "key_id",
    ];
    if object.len() != FIELDS.len()
        || object.keys().any(|field| !FIELDS.contains(&field.as_str()))
        || object.get("schema_version").and_then(Value::as_str)
            != Some(SOURCE_ORIGIN_WITNESS_RECEIPT_SCHEMA_V1)
    {
        return Err(SourceOriginWitnessRejectV1::InvalidJson);
    }
    let string = |field: &str| -> Result<String, SourceOriginWitnessRejectV1> {
        object.get(field).and_then(Value::as_str).map(str::to_owned)
            .ok_or(SourceOriginWitnessRejectV1::InvalidJson)
    };
    let integer = |field: &str| -> Result<u64, SourceOriginWitnessRejectV1> {
        object.get(field).and_then(Value::as_u64)
            .ok_or(SourceOriginWitnessRejectV1::InvalidJson)
    };
    let derivation_receipt_sha256 = match object.get("derivation_receipt_sha256") {
        Some(Value::Null) => None,
        Some(Value::String(value)) if valid_sha256(value) => Some(value.clone()),
        _ => return Err(SourceOriginWitnessRejectV1::InvalidDigest),
    };
    let receipt = SourceOriginWitnessReceiptV1 {
        run_id: string("run_id")?,
        query_invocation_id: string("query_invocation_id")?,
        graph_sha256: string("graph_sha256")?,
        source_origin_receipt_sha256: string("source_origin_receipt_sha256")?,
        operator_id: string("operator_id")?,
        failure_domain_sha256: string("failure_domain_sha256")?,
        evidence_id: string("evidence_id")?,
        source_id: string("source_id")?,
        source_sha256: string("source_sha256")?,
        payload_sha256: string("payload_sha256")?,
        evidence_class_sha256: string("evidence_class_sha256")?,
        observed_parent_evidence_ids: string_array(object, "observed_parent_evidence_ids", 1_024, false)?,
        derivation_receipt_sha256,
        common_cause_group_sha256: string_array(object, "common_cause_group_sha256", 1_024, true)?,
        publisher_identity_sha256: string("publisher_identity_sha256")?,
        acquisition_record_sha256: string("acquisition_record_sha256")?,
        source_revision_sha256: string("source_revision_sha256")?,
        observed_at_unix: integer("observed_at_unix")?,
        issued_at_unix: integer("issued_at_unix")?,
        expires_at_unix: integer("expires_at_unix")?,
        status: string("status")?,
        issuer_profile_sha256: string("issuer_profile_sha256")?,
        key_id: string("key_id")?,
    };
    for digest in [
        receipt.graph_sha256.as_str(),
        receipt.source_origin_receipt_sha256.as_str(),
        receipt.failure_domain_sha256.as_str(),
        receipt.source_sha256.as_str(),
        receipt.payload_sha256.as_str(),
        receipt.evidence_class_sha256.as_str(),
        receipt.publisher_identity_sha256.as_str(),
        receipt.acquisition_record_sha256.as_str(),
        receipt.source_revision_sha256.as_str(),
        receipt.issuer_profile_sha256.as_str(),
        receipt.key_id.as_str(),
    ] {
        if !valid_sha256(digest) {
            return Err(SourceOriginWitnessRejectV1::InvalidDigest);
        }
    }
    for atom in [
        receipt.run_id.as_str(), receipt.query_invocation_id.as_str(),
        receipt.operator_id.as_str(), receipt.evidence_id.as_str(),
        receipt.source_id.as_str(), receipt.status.as_str(),
    ] {
        if !valid_atom(atom, 1_024) {
            return Err(SourceOriginWitnessRejectV1::InvalidReceipt);
        }
    }
    if receipt.issued_at_unix >= receipt.expires_at_unix
        || receipt.expires_at_unix.saturating_sub(receipt.issued_at_unix)
            > MAX_RECEIPT_LIFETIME_SECS_V1
        || receipt.status != "observed_consistent"
    {
        return Err(SourceOriginWitnessRejectV1::InvalidReceipt);
    }
    Ok(receipt)
}

fn validate_witness_time_and_signature(
    receipt: &SourceOriginWitnessReceiptV1,
    pae: &[u8],
    signature: &[u8],
    verifier: &PinnedEd25519QualificationVerifierV1,
    now_unix: u64,
) -> Result<(), SourceOriginWitnessRejectV1> {
    if now_unix < receipt.issued_at_unix {
        return Err(SourceOriginWitnessRejectV1::NotYetValid);
    }
    if now_unix >= receipt.expires_at_unix {
        return Err(SourceOriginWitnessRejectV1::Expired);
    }
    verifier.verify_message(
        &receipt.issuer_profile_sha256,
        &receipt.key_id,
        pae,
        signature,
        receipt.issued_at_unix,
        now_unix,
    ).map_err(map_signature_error)
}

fn map_signature_error(
    error: crate::evidence_qualification_gate::EvidenceQualificationRejectV1,
) -> SourceOriginWitnessRejectV1 {
    use crate::evidence_qualification_gate::EvidenceQualificationRejectV1 as E;
    match error {
        E::UnknownKey | E::RevokedKey => SourceOriginWitnessRejectV1::UnknownOrRevokedSigner,
        E::KeyNotYetValid => SourceOriginWitnessRejectV1::NotYetValid,
        E::KeyExpired => SourceOriginWitnessRejectV1::Expired,
        E::TrustProfileDigestMismatch => SourceOriginWitnessRejectV1::ProfileMismatch,
        _ => SourceOriginWitnessRejectV1::InvalidSignature,
    }
}

fn source_origin_bundle_sha256(origin: &VerifiedSourceOriginSetV1) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:verified-source-origin-bundle:v1\0");
    hash_field(&mut h, origin.graph_sha256());
    hash_field(&mut h, origin.origin_profile_sha256());
    let mut signers = origin.signer_key_ids();
    signers.sort();
    h.update((signers.len() as u64).to_be_bytes());
    for signer in signers { hash_field(&mut h, &signer); }
    let receipts = origin.envelope_sha256s();
    h.update((receipts.len() as u64).to_be_bytes());
    for receipt in receipts { hash_field(&mut h, &receipt); }
    hex_digest(&h.finalize())
}

fn string_array(
    object: &serde_json::Map<String, Value>,
    field: &str,
    maximum: usize,
    digest: bool,
) -> Result<Vec<String>, SourceOriginWitnessRejectV1> {
    let values = object.get(field).and_then(Value::as_array)
        .ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
    if values.len() > maximum { return Err(SourceOriginWitnessRejectV1::InvalidReceipt); }
    let parsed = values.iter().map(|value| {
        let value = value.as_str().ok_or(SourceOriginWitnessRejectV1::InvalidJson)?;
        if value.trim().is_empty() || value != value.trim()
            || value.chars().any(char::is_control)
            || digest && !valid_sha256(value)
        {
            return Err(if digest { SourceOriginWitnessRejectV1::InvalidDigest }
                else { SourceOriginWitnessRejectV1::InvalidReceipt });
        }
        Ok(value.to_owned())
    }).collect::<Result<Vec<_>, _>>()?;
    if parsed.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SourceOriginWitnessRejectV1::InvalidReceipt);
    }
    Ok(parsed)
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty() && value == value.trim()
        && value.len() <= maximum && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_digest(&digest)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_of_three_policy_requires_distinct_profiles_and_failure_domains() {
        let policy = SourceOriginWitnessPolicyV1 {
            schema_version: SOURCE_ORIGIN_WITNESS_POLICY_SCHEMA_V1.to_owned(),
            graph_sha256: "a".repeat(64),
            minimum_operators: 2,
            minimum_failure_domains: 2,
            maximum_observation_age_secs: 24 * 60 * 60,
            operators: vec![
                SourceOriginWitnessOperatorV1 {
                    operator_id: "archive-a".into(),
                    failure_domain_sha256: "b".repeat(64),
                    verifier_profile_sha256: "c".repeat(64),
                },
                SourceOriginWitnessOperatorV1 {
                    operator_id: "archive-b".into(),
                    failure_domain_sha256: "d".repeat(64),
                    verifier_profile_sha256: "e".repeat(64),
                },
                SourceOriginWitnessOperatorV1 {
                    operator_id: "publisher-c".into(),
                    failure_domain_sha256: "f".repeat(64),
                    verifier_profile_sha256: "1".repeat(64),
                },
            ],
        };
        assert!(policy.digest().is_ok());
    }

    #[test]
    fn a_shared_failure_domain_cannot_satisfy_a_two_domain_threshold() {
        let policy = SourceOriginWitnessPolicyV1 {
            schema_version: SOURCE_ORIGIN_WITNESS_POLICY_SCHEMA_V1.to_owned(),
            graph_sha256: "a".repeat(64),
            minimum_operators: 2,
            minimum_failure_domains: 2,
            maximum_observation_age_secs: 60,
            operators: vec![
                SourceOriginWitnessOperatorV1 {
                    operator_id: "archive-a".into(),
                    failure_domain_sha256: "b".repeat(64),
                    verifier_profile_sha256: "c".repeat(64),
                },
                SourceOriginWitnessOperatorV1 {
                    operator_id: "archive-b".into(),
                    failure_domain_sha256: "b".repeat(64),
                    verifier_profile_sha256: "d".repeat(64),
                },
            ],
        };
        assert_eq!(policy.validate(), Err(SourceOriginWitnessRejectV1::InvalidPolicy));
    }

    #[test]
    fn hidden_parent_or_common_cause_disagreement_is_never_majority_voted_away() {
        let node = VerifiedSourceLineageNodeBindingV1 {
            evidence_id: "evidence-1".into(),
            source_id: "source-1".into(),
            source_sha256: "a".repeat(64),
            payload_sha256: "b".repeat(64),
            evidence_class_sha256: "c".repeat(64),
            parent_evidence_ids: vec!["parent-known".into()],
            derivation_receipt_sha256: Some("d".repeat(64)),
            common_cause_group_sha256: vec!["e".repeat(64)],
        };
        let mut receipt = SourceOriginWitnessReceiptV1 {
            run_id: "run-1".into(),
            query_invocation_id: "invocation-1".into(),
            graph_sha256: "f".repeat(64),
            source_origin_receipt_sha256: "1".repeat(64),
            operator_id: "archive-a".into(),
            failure_domain_sha256: "2".repeat(64),
            evidence_id: "evidence-1".into(),
            source_id: "source-1".into(),
            source_sha256: "a".repeat(64),
            payload_sha256: "b".repeat(64),
            evidence_class_sha256: "c".repeat(64),
            observed_parent_evidence_ids: vec!["parent-known".into()],
            derivation_receipt_sha256: Some("d".repeat(64)),
            common_cause_group_sha256: vec!["e".repeat(64)],
            publisher_identity_sha256: "3".repeat(64),
            acquisition_record_sha256: "4".repeat(64),
            source_revision_sha256: "5".repeat(64),
            observed_at_unix: 1,
            issued_at_unix: 2,
            expires_at_unix: 3,
            status: "observed_consistent".into(),
            issuer_profile_sha256: "6".repeat(64),
            key_id: "7".repeat(64),
        };
        assert_eq!(validate_observed_ancestry(&receipt, &node), Ok(()));
        receipt.observed_parent_evidence_ids.push("hidden-parent".into());
        assert_eq!(
            validate_observed_ancestry(&receipt, &node),
            Err(SourceOriginWitnessRejectV1::AncestryDisagreement)
        );
        receipt.observed_parent_evidence_ids = node.parent_evidence_ids.clone();
        receipt.common_cause_group_sha256.clear();
        assert_eq!(
            validate_observed_ancestry(&receipt, &node),
            Err(SourceOriginWitnessRejectV1::AncestryDisagreement)
        );
    }

    #[test]
    fn shared_acquisition_record_cannot_masquerade_as_two_independent_observations() {
        let make = |operator_id: &str, acquisition_record: &str| {
            VerifiedSourceOriginWitnessV1 {
                receipt: SourceOriginWitnessReceiptV1 {
                    run_id: "run-1".into(),
                    query_invocation_id: "invocation-1".into(),
                    graph_sha256: "a".repeat(64),
                    source_origin_receipt_sha256: "b".repeat(64),
                    operator_id: operator_id.into(),
                    failure_domain_sha256: if operator_id == "archive-a" {
                        "c".repeat(64)
                    } else {
                        "d".repeat(64)
                    },
                    evidence_id: "node-1".into(),
                    source_id: "source-1".into(),
                    source_sha256: "e".repeat(64),
                    payload_sha256: "f".repeat(64),
                    evidence_class_sha256: "1".repeat(64),
                    observed_parent_evidence_ids: Vec::new(),
                    derivation_receipt_sha256: None,
                    common_cause_group_sha256: Vec::new(),
                    publisher_identity_sha256: "2".repeat(64),
                    acquisition_record_sha256: acquisition_record.into(),
                    source_revision_sha256: "3".repeat(64),
                    observed_at_unix: 10,
                    issued_at_unix: 11,
                    expires_at_unix: 20,
                    status: "observed_consistent".into(),
                    issuer_profile_sha256: "4".repeat(64),
                    key_id: if operator_id == "archive-a" {
                        "5".repeat(64)
                    } else {
                        "6".repeat(64)
                    },
                },
                envelope_sha256: "7".repeat(64),
                pae: Vec::new(),
                signature: Vec::new(),
            }
        };
        let left = make("archive-a", &"8".repeat(64));
        let right = make("archive-b", &"8".repeat(64));
        assert_eq!(
            validate_witness_metadata_agreement(&[&left, &right]),
            Err(SourceOriginWitnessRejectV1::SharedObservationRecord)
        );
    }

    #[test]
    fn publisher_and_revision_disagreement_are_not_averaged() {
        let mut first = SourceOriginWitnessReceiptV1 {
            run_id: "run-1".into(),
            query_invocation_id: "invocation-1".into(),
            graph_sha256: "a".repeat(64),
            source_origin_receipt_sha256: "b".repeat(64),
            operator_id: "archive-a".into(),
            failure_domain_sha256: "c".repeat(64),
            evidence_id: "node-1".into(),
            source_id: "source-1".into(),
            source_sha256: "d".repeat(64),
            payload_sha256: "e".repeat(64),
            evidence_class_sha256: "f".repeat(64),
            observed_parent_evidence_ids: Vec::new(),
            derivation_receipt_sha256: None,
            common_cause_group_sha256: Vec::new(),
            publisher_identity_sha256: "1".repeat(64),
            acquisition_record_sha256: "2".repeat(64),
            source_revision_sha256: "3".repeat(64),
            observed_at_unix: 10,
            issued_at_unix: 11,
            expires_at_unix: 20,
            status: "observed_consistent".into(),
            issuer_profile_sha256: "4".repeat(64),
            key_id: "5".repeat(64),
        };
        let mut second = first.clone();
        second.operator_id = "archive-b".into();
        second.acquisition_record_sha256 = "6".repeat(64);
        second.key_id = "7".repeat(64);
        second.publisher_identity_sha256 = "8".repeat(64);
        assert_eq!(
            validate_witness_metadata_agreement(&[
                &VerifiedSourceOriginWitnessV1 {
                    receipt: first.clone(), envelope_sha256: "9".repeat(64),
                    pae: Vec::new(), signature: Vec::new(),
                },
                &VerifiedSourceOriginWitnessV1 {
                    receipt: second.clone(), envelope_sha256: "a".repeat(64),
                    pae: Vec::new(), signature: Vec::new(),
                },
            ]),
            Err(SourceOriginWitnessRejectV1::ObservedSourceMismatch)
        );
        second.publisher_identity_sha256 = first.publisher_identity_sha256.clone();
        second.source_revision_sha256 = "b".repeat(64);
        assert_eq!(
            validate_witness_metadata_agreement(&[
                &VerifiedSourceOriginWitnessV1 {
                    receipt: first, envelope_sha256: "9".repeat(64),
                    pae: Vec::new(), signature: Vec::new(),
                },
                &VerifiedSourceOriginWitnessV1 {
                    receipt: second, envelope_sha256: "a".repeat(64),
                    pae: Vec::new(), signature: Vec::new(),
                },
            ]),
            Err(SourceOriginWitnessRejectV1::ObservedSourceMismatch)
        );
    }

    #[test]
    fn same_operator_cannot_supply_duplicate_node_observations() {
        let mut seen = BTreeSet::new();
        assert!(seen.insert(("evidence-1", "archive-a")));
        assert!(!seen.insert(("evidence-1", "archive-a")));
    }

    #[test]
    fn policy_digest_binds_quorum_thresholds_and_failure_domains() {
        let make = |minimum_operators| SourceOriginWitnessPolicyV1 {
            schema_version: SOURCE_ORIGIN_WITNESS_POLICY_SCHEMA_V1.to_owned(),
            graph_sha256: "a".repeat(64),
            minimum_operators,
            minimum_failure_domains: 2,
            maximum_observation_age_secs: 600,
            operators: vec![
                SourceOriginWitnessOperatorV1 {
                    operator_id: "archive-a".into(),
                    failure_domain_sha256: "b".repeat(64),
                    verifier_profile_sha256: "c".repeat(64),
                },
                SourceOriginWitnessOperatorV1 {
                    operator_id: "archive-b".into(),
                    failure_domain_sha256: "d".repeat(64),
                    verifier_profile_sha256: "e".repeat(64),
                },
                SourceOriginWitnessOperatorV1 {
                    operator_id: "publisher-c".into(),
                    failure_domain_sha256: "f".repeat(64),
                    verifier_profile_sha256: "1".repeat(64),
                },
            ],
        };
        assert_ne!(make(2).digest().unwrap(), make(3).digest().unwrap());
    }
}
