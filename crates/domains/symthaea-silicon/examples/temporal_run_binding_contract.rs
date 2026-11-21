// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run-level binding for temporal workload, work-ledger and freshness receipts.
//!
//! This example does not reimplement either receipt schema. It validates the
//! identity seam between the existing temporal-work-ledger 0.2 receipt and
//! decision-freshness 0.1 receipt, and binds both to exact candidate-visible
//! input bytes. The evaluator's hidden oracle identity remains separate.
//!
//! Current workload_id encoding:
//!   <family>@<version>:sha256:<SHA256 of exact candidate-visible input bytes>
//!
//! This lets current receipt versions bind the input trace without adding
//! duplicate schema fields. A future versioned receipt can promote the digest
//! to a first-class field without changing the join semantics.
//!
//! From the repository root:
//!   cargo fmt --check
//!   cargo test --locked -p symthaea-silicon --example temporal_run_binding_contract
//!   cargo run --locked -p symthaea-silicon --example temporal_run_binding_contract
//!
//! Hashes are over exact byte slices. Re-serialization or whitespace changes
//! produce a different digest and are intentionally not treated as equivalent.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::error::Error;

const SCHEMA: &str = "symthaea.temporal-run-binding";
const SCHEMA_VERSION: &str = "0.1.0";
const WORK_SCHEMA: &str = "symthaea.temporal-work-ledger";
const WORK_VERSION: &str = "0.2";
const FRESHNESS_SCHEMA: &str = "symthaea.temporal-decision-freshness";
const FRESHNESS_VERSION: &str = "0.1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateTraceIdentity {
    workload_family: String,
    workload_version: String,
    profile_id: String,
    source_revision: String,
    configuration_id: String,
    trace_sha256: String,
}

impl CandidateTraceIdentity {
    fn validate(&self) -> Result<(), &'static str> {
        for value in [
            self.workload_family.as_str(),
            self.workload_version.as_str(),
            self.profile_id.as_str(),
            self.source_revision.as_str(),
            self.configuration_id.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err("candidate trace identity fields cannot be empty");
            }
        }
        if !is_sha256(&self.trace_sha256) {
            return Err("candidate-visible trace identity must be a SHA-256 digest");
        }
        Ok(())
    }

    fn workload_id(&self) -> String {
        format!(
            "{}@{}:sha256:{}",
            self.workload_family, self.workload_version, self.trace_sha256
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinedRunBinding {
    schema: String,
    schema_version: String,
    run_id: String,
    workload_family: String,
    workload_version: String,
    workload_id: String,
    profile_id: String,
    trace_sha256: String,
    source_revision: String,
    configuration_id: String,
    temporal_work_receipt_sha256: String,
    freshness_receipt_sha256: String,
    decision_contract_id: String,
    independent_verifier_id: String,
    decision_opportunities: u64,
    logical_duration_ns: u64,
    logical_horizon_end_ns: u64,
    evaluator_oracle_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BindingError {
    InvalidCandidateIdentity(&'static str),
    EmptyRunId,
    InvalidJson(&'static str),
    MissingField(&'static str),
    WrongSchema(&'static str),
    WrongVersion(&'static str),
    NotBenchmarkObservation,
    IdentityMismatch(&'static str),
    InvalidDigest(&'static str),
    WorkReceiptDigestMismatch,
    DecisionOpportunityCountMismatch,
    LogicalHorizonOutOfRange,
    InvalidOracleDigest,
    ArithmeticOverflow,
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for BindingError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn object_value<'a>(object: &'a Value, key: &'static str) -> Result<&'a Value, BindingError> {
    object
        .as_object()
        .and_then(|map| map.get(key))
        .ok_or(BindingError::MissingField(key))
}

fn string_field(object: &Value, key: &'static str) -> Result<String, BindingError> {
    object_value(object, key)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or(BindingError::MissingField(key))
}

fn u64_field(object: &Value, key: &'static str) -> Result<u64, BindingError> {
    object_value(object, key)?
        .as_u64()
        .ok_or(BindingError::MissingField(key))
}

fn parse_json(bytes: &[u8], name: &'static str) -> Result<Value, BindingError> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|_| BindingError::InvalidJson(name))?;
    if !value.is_object() {
        return Err(BindingError::InvalidJson(name));
    }
    Ok(value)
}

/// Join exact serialized bytes of a work receipt and freshness receipt.
///
/// evaluator_oracle_sha256 is evaluator-side metadata only. It must never be
/// inserted into candidate-visible observations or scheduling inputs.
fn join_receipts(
    run_id: &str,
    identity: &CandidateTraceIdentity,
    work_receipt_bytes: &[u8],
    freshness_receipt_bytes: &[u8],
    evaluator_oracle_sha256: Option<&str>,
) -> Result<JoinedRunBinding, BindingError> {
    identity
        .validate()
        .map_err(BindingError::InvalidCandidateIdentity)?;
    if run_id.trim().is_empty() {
        return Err(BindingError::EmptyRunId);
    }

    let work = parse_json(work_receipt_bytes, "temporal-work receipt")?;
    let freshness = parse_json(freshness_receipt_bytes, "decision-freshness receipt")?;

    if string_field(&work, "schema")? != WORK_SCHEMA {
        return Err(BindingError::WrongSchema("temporal-work"));
    }
    if string_field(&work, "schema_version")? != WORK_VERSION {
        return Err(BindingError::WrongVersion("temporal-work"));
    }
    if string_field(&work, "kind")? != "benchmark_observation" {
        return Err(BindingError::NotBenchmarkObservation);
    }
    if string_field(&freshness, "schema")? != FRESHNESS_SCHEMA {
        return Err(BindingError::WrongSchema("decision-freshness"));
    }
    if string_field(&freshness, "schema_version")? != FRESHNESS_VERSION {
        return Err(BindingError::WrongVersion("decision-freshness"));
    }

    let expected_workload_id = identity.workload_id();
    for (field, expected) in [
        ("workload_id", expected_workload_id.as_str()),
        ("profile_id", identity.profile_id.as_str()),
        ("source_revision", identity.source_revision.as_str()),
        ("configuration_id", identity.configuration_id.as_str()),
    ] {
        if string_field(&work, field)? != expected {
            return Err(BindingError::IdentityMismatch(field));
        }
        if string_field(&freshness, field)? != expected {
            return Err(BindingError::IdentityMismatch(field));
        }
    }

    let work_digest = sha256_hex(work_receipt_bytes);
    let bound_work_digest = string_field(&freshness, "temporal_work_receipt_sha256")?;
    if !is_sha256(&bound_work_digest) {
        return Err(BindingError::InvalidDigest("freshness work-receipt digest"));
    }
    if bound_work_digest != work_digest {
        return Err(BindingError::WorkReceiptDigestMismatch);
    }

    let work_section = object_value(&work, "work")?;
    let work_opportunities = u64_field(work_section, "decision_opportunities")?;
    let freshness_opportunities = object_value(&freshness, "opportunities")?
        .as_array()
        .ok_or(BindingError::MissingField("opportunities"))?;
    let freshness_count = u64::try_from(freshness_opportunities.len())
        .map_err(|_| BindingError::ArithmeticOverflow)?;
    if work_opportunities != freshness_count {
        return Err(BindingError::DecisionOpportunityCountMismatch);
    }

    let logical_duration_ns = u64_field(&work, "logical_duration_ns")?;
    let logical_horizon_end_ns = u64_field(&freshness, "logical_horizon_end_ns")?;
    if logical_duration_ns == 0
        || logical_horizon_end_ns == 0
        || logical_horizon_end_ns > logical_duration_ns
    {
        return Err(BindingError::LogicalHorizonOutOfRange);
    }

    let decision_contract_id = string_field(&freshness, "decision_contract_id")?;
    let independent_verifier_id = string_field(&freshness, "independent_verifier_id")?;
    if decision_contract_id.trim().is_empty() || independent_verifier_id.trim().is_empty() {
        return Err(BindingError::MissingField(
            "decision contract / verifier identity",
        ));
    }

    let evaluator_oracle_sha256 = evaluator_oracle_sha256.map(ToOwned::to_owned);
    if evaluator_oracle_sha256
        .as_deref()
        .is_some_and(|digest| !is_sha256(digest))
    {
        return Err(BindingError::InvalidOracleDigest);
    }

    let freshness_digest = sha256_hex(freshness_receipt_bytes);
    Ok(JoinedRunBinding {
        schema: SCHEMA.to_owned(),
        schema_version: SCHEMA_VERSION.to_owned(),
        run_id: run_id.to_owned(),
        workload_family: identity.workload_family.clone(),
        workload_version: identity.workload_version.clone(),
        workload_id: expected_workload_id,
        profile_id: identity.profile_id.clone(),
        trace_sha256: identity.trace_sha256.clone(),
        source_revision: identity.source_revision.clone(),
        configuration_id: identity.configuration_id.clone(),
        temporal_work_receipt_sha256: work_digest,
        freshness_receipt_sha256: freshness_digest,
        decision_contract_id,
        independent_verifier_id,
        decision_opportunities: work_opportunities,
        logical_duration_ns,
        logical_horizon_end_ns,
        evaluator_oracle_sha256,
    })
}

fn example_receipts(
    identity: &CandidateTraceIdentity,
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let workload_id = identity.workload_id();
    let work = serde_json::json!({
        "schema": WORK_SCHEMA,
        "schema_version": WORK_VERSION,
        "kind": "benchmark_observation",
        "profile_id": identity.profile_id.clone(),
        "workload_id": workload_id,
        "source_revision": identity.source_revision.clone(),
        "configuration_id": identity.configuration_id.clone(),
        "seed": 17_u64,
        "logical_duration_ns": 1_000_u64,
        "work": {
            "decision_opportunities": 2_u64,
            "output_decisions": 2_u64,
            "late_decisions": 0_u64,
            "abstained_decisions": 0_u64,
            "unserved_decisions": 0_u64
        }
    });
    let work_bytes = serde_json::to_vec(&work)?;
    let work_digest = sha256_hex(&work_bytes);
    let freshness = serde_json::json!({
        "schema": FRESHNESS_SCHEMA,
        "schema_version": FRESHNESS_VERSION,
        "workload_id": identity.workload_id(),
        "profile_id": identity.profile_id.clone(),
        "source_revision": identity.source_revision.clone(),
        "configuration_id": identity.configuration_id.clone(),
        "decision_contract_id": "fixture-decision-contract-v1",
        "independent_verifier_id": "fixture-independent-verifier-v1",
        "temporal_work_receipt_sha256": work_digest,
        "logical_horizon_end_ns": 1_000_u64,
        "opportunities": [{ "opportunity_id": 0 }, { "opportunity_id": 1 }]
    });
    Ok((work_bytes, serde_json::to_vec(&freshness)?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let candidate_trace_bytes =
        br#"{"profile":"jittered","events":[[0,0.0],[9,0.3],[19,0.1]]}"#;
    let identity = CandidateTraceIdentity {
        workload_family: "leaky-irregular-reference".to_owned(),
        workload_version: "v0.1".to_owned(),
        profile_id: "jittered".to_owned(),
        source_revision: "40c3b0e75845a20b3ef2dbbe0ea3a0af0068f4b7".to_owned(),
        configuration_id: "scalar-leaky-tau-140-seed-17".to_owned(),
        trace_sha256: sha256_hex(candidate_trace_bytes),
    };
    let (work, freshness) = example_receipts(&identity)?;
    let oracle_digest = "a".repeat(64);
    let joined = join_receipts(
        "example-run-0001",
        &identity,
        &work,
        &freshness,
        Some(&oracle_digest),
    )?;
    println!("{}", serde_json::to_string_pretty(&joined)?);
    println!("scope: receipt-identity join fixture; no model, timing, energy or correctness result");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CandidateTraceIdentity {
        let trace = br#"{"events":[[0,0.0],[10,0.5],[20,-0.2]]}"#;
        CandidateTraceIdentity {
            workload_family: "scalar-reference".to_owned(),
            workload_version: "v0.1".to_owned(),
            profile_id: "jittered".to_owned(),
            source_revision: "abc123".to_owned(),
            configuration_id: "cfg-17".to_owned(),
            trace_sha256: sha256_hex(trace),
        }
    }

    #[test]
    fn joins_work_and_freshness_receipts_when_all_identity_roots_match() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        let oracle_digest = "b".repeat(64);
        let joined = join_receipts(
            "run-1",
            &id,
            &work,
            &freshness,
            Some(&oracle_digest),
        )
        .unwrap();
        assert_eq!(joined.decision_opportunities, 2);
        assert_eq!(joined.workload_id, id.workload_id());
        assert_eq!(joined.temporal_work_receipt_sha256, sha256_hex(&work));
        assert_eq!(joined.freshness_receipt_sha256, sha256_hex(&freshness));
    }

    #[test]
    fn rejects_contract_smoke_receipt_as_benchmark_observation() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        let mut value: Value = serde_json::from_slice(&work).unwrap();
        value["kind"] = Value::String("contract_smoke_fixture".to_owned());
        let modified = serde_json::to_vec(&value).unwrap();
        let error = join_receipts("run-2", &id, &modified, &freshness, None).unwrap_err();
        assert_eq!(error, BindingError::NotBenchmarkObservation);
    }

    #[test]
    fn rejects_work_receipt_digest_mismatch() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        let mut value: Value = serde_json::from_slice(&freshness).unwrap();
        value["temporal_work_receipt_sha256"] = Value::String("0".repeat(64));
        let modified = serde_json::to_vec(&value).unwrap();
        let error = join_receipts("run-3", &id, &work, &modified, None).unwrap_err();
        assert_eq!(error, BindingError::WorkReceiptDigestMismatch);
    }

    #[test]
    fn rejects_trace_digest_or_workload_identity_mismatch() {
        let mut id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        id.trace_sha256 = "c".repeat(64);
        let error = join_receipts("run-4", &id, &work, &freshness, None).unwrap_err();
        assert_eq!(error, BindingError::IdentityMismatch("workload_id"));
    }

    #[test]
    fn rejects_cross_receipt_source_or_configuration_mismatch() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        let mut value: Value = serde_json::from_slice(&freshness).unwrap();
        value["source_revision"] = Value::String("different-source".to_owned());
        let modified = serde_json::to_vec(&value).unwrap();
        let error = join_receipts("run-5", &id, &work, &modified, None).unwrap_err();
        assert_eq!(error, BindingError::IdentityMismatch("source_revision"));
    }

    #[test]
    fn rejects_decision_opportunity_denominator_mismatch() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        let mut value: Value = serde_json::from_slice(&freshness).unwrap();
        value["opportunities"] = serde_json::json!([{ "opportunity_id": 0 }]);
        let modified = serde_json::to_vec(&value).unwrap();
        let error = join_receipts("run-6", &id, &work, &modified, None).unwrap_err();
        assert_eq!(error, BindingError::DecisionOpportunityCountMismatch);
    }

    #[test]
    fn rejects_logical_horizon_beyond_workload_duration() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        let mut value: Value = serde_json::from_slice(&freshness).unwrap();
        value["logical_horizon_end_ns"] = Value::from(1_001_u64);
        let modified = serde_json::to_vec(&value).unwrap();
        let error = join_receipts("run-7", &id, &work, &modified, None).unwrap_err();
        assert_eq!(error, BindingError::LogicalHorizonOutOfRange);
    }

    #[test]
    fn hidden_oracle_digest_is_separately_validated_evaluator_metadata() {
        let id = identity();
        let (work, freshness) = example_receipts(&id).unwrap();
        assert!(join_receipts("run-8", &id, &work, &freshness, Some("bad")).is_err());
        let oracle_digest = "d".repeat(64);
        let joined = join_receipts(
            "run-9",
            &id,
            &work,
            &freshness,
            Some(&oracle_digest),
        )
        .unwrap();
        assert_eq!(joined.evaluator_oracle_sha256, Some(oracle_digest));
    }
}