use std::io::{self, Write};

use serde_json::{Value, json};
use symthaea_rea_compat::{
    canonical_jcs_bytes, sha256_hex, verify_evidence_json,
};

const CLAIM_SCHEMA: &str = "luminous.rea.claim.v0";
const RESULT_SCHEMA: &str = "luminous.rea.claim-evaluation.v0";
const EVALUATOR_VERSION: &str = "rea-claim-evaluator/v0";
const FALSIFICATION: &str =
    "A same-scope verified Evidence record has a different normalized_result";

const CLAIM_CEILING: [&str; 4] = [
    "A supported result is scoped to the exact subject, operation, environment fields and Evidence IDs in this derivation.",
    "This evaluator verifies what a REA Evidence record reports; it does not prove the underlying real-world proposition is true.",
    "Evidence ID verification is a semantic commitment, not producer authentication or trusted-time evidence.",
    "Exact normalized_result equality establishes a recorded provider observation only; it does not establish artifact semantics, completeness, authenticity or safety.",
];

fn digest(value: &Value) -> String {
    sha256_hex(&canonical_jcs_bytes(value).expect("seed JSON canonicalizes"))
}

fn result_from_body(mut body: Value) -> Value {
    let derivation_sha256 = digest(&body);
    body.as_object_mut()
        .expect("derivation body is an object")
        .insert("derivation_sha256".to_owned(), json!(derivation_sha256));
    body
}

fn frame(claim: &Value, result: &Value, evidence_records: &[&[u8]]) -> Vec<u8> {
    let claim_bytes = serde_json::to_vec(claim).expect("claim serializes");
    let result_bytes = serde_json::to_vec(result).expect("result serializes");
    let mut output = Vec::new();

    output.extend_from_slice(&(claim_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&(result_bytes.len() as u32).to_le_bytes());
    output.push(evidence_records.len() as u8);
    for record in evidence_records {
        output.extend_from_slice(&(record.len() as u32).to_le_bytes());
    }
    output.extend_from_slice(&claim_bytes);
    output.extend_from_slice(&result_bytes);
    for record in evidence_records {
        output.extend_from_slice(record);
    }
    output
}

fn claim_value(subject_sha256: &str, operation: &str, asserted_value: Value, evidence_ids: Vec<String>) -> Value {
    json!({
        "schema": CLAIM_SCHEMA,
        "claim_id": "fuzz-seed-record-level-claim",
        "predicate": "provider_reported_observation",
        "scope": {
            "subject_sha256": subject_sha256,
            "operation": operation,
            "platform": null,
            "platform_version": null,
            "environment_id": null
        },
        "asserted_value": asserted_value,
        "source_predicate_type": "rea.analysis",
        "evidence_ids": evidence_ids,
        "assumptions": [],
        "falsification_condition": FALSIFICATION
    })
}

fn underdetermined_seed() -> Vec<u8> {
    let subject_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let claim = claim_value("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "inspect_artifact", json!({"status":"complete"}), Vec::new());
    let claim_request_sha256 = digest(&claim);
    let scope = claim["scope"].clone();
    let body = json!({
        "schema": RESULT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "rule_id": "provider-reported-observation/exact-normalized-result/v0",
        "claim_id": "fuzz-seed-record-level-claim",
        "claim_request_sha256": claim_request_sha256,
        "predicate": "provider_reported_observation",
        "disposition": "underdetermined",
        "scope": scope,
        "asserted_value": {"status":"complete"},
        "source_predicate_type": "rea.analysis",
        "requested_evidence_ids": [],
        "assumptions": [],
        "evidence_assessments": [],
        "missing_evidence_ids": [],
        "unmet_obligations": [
            "no Evidence IDs were supplied",
            "no applicable premise established or contradicted the asserted value"
        ],
        "falsification_condition": FALSIFICATION,
        "claim_ceiling": CLAIM_CEILING
    });
    let result = result_from_body(body);
    frame(&claim, &result, &[])
}

fn supporting_seed(evidence_bytes: &[u8]) -> Vec<u8> {
    let evidence = verify_evidence_json(evidence_bytes).expect("pinned fixture verifies");
    let subject_sha256 = evidence.value["subject"]["digest"]["sha256"]
        .as_str()
        .expect("fixture has subject digest");
    let operation = evidence.value["operation"]
        .as_str()
        .expect("fixture has operation");
    let predicate_type = evidence.value["predicate_type"]
        .as_str()
        .expect("fixture has predicate type");
    let asserted_value = evidence.value["normalized_result"].clone();

    let mut claim = claim_value(
        subject_sha256,
        operation,
        asserted_value.clone(),
        vec![evidence.evidence_id.clone()],
    );
    claim["source_predicate_type"] = json!(predicate_type);
    let claim_request_sha256 = digest(&claim);

    let assessment = json!({
        "evidence_id": evidence.evidence_id,
        "kind": "supports",
        "provider_id": evidence.value["provider"]["id"],
        "provider_name": evidence.value["provider"]["name"],
        "provider_version": evidence.value["provider"]["version"],
        "confidence": evidence.value["confidence"],
        "authority": evidence.value["authority"],
        "reason": "normalized_result matches asserted_value under RFC 8785 canonical JSON semantics"
    });
    let body = json!({
        "schema": RESULT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "rule_id": "provider-reported-observation/exact-normalized-result/v0",
        "claim_id": "fuzz-seed-record-level-claim",
        "claim_request_sha256": claim_request_sha256,
        "predicate": "provider_reported_observation",
        "disposition": "supported_within_scope",
        "scope": claim["scope"],
        "asserted_value": asserted_value,
        "source_predicate_type": predicate_type,
        "requested_evidence_ids": claim["evidence_ids"],
        "assumptions": [],
        "evidence_assessments": [assessment],
        "missing_evidence_ids": [],
        "unmet_obligations": [],
        "falsification_condition": FALSIFICATION,
        "claim_ceiling": CLAIM_CEILING
    });
    let result = result_from_body(body);
    frame(&claim, &result, &[evidence_bytes])
}

fn main() -> io::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "support".to_owned());
    let seed = match mode.as_str() {
        "support" => {
            let evidence = include_bytes!(
                "../../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
            );
            supporting_seed(evidence)
        }
        "underdetermined" => underdetermined_seed(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected seed mode: support or underdetermined",
            ));
        }
    };
    io::stdout().write_all(&seed)
}
