use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use symthaea_rea_compat::{
    analysis_profile_digest, verify_evidence_json, EvidenceErrorKind,
};

/// When the dedicated upstream differential workflow supplies a fixture,
/// verify the complete record emitted by the pinned TypeScript REA runtime.
/// Without that workflow variable this test is intentionally inert; the
/// source-derived unit vector remains a separate, clearly labelled baseline.
#[test]
fn verifies_evidence_emitted_by_pinned_upstream_rea_runtime() {
    let path = match std::env::var_os("REA_UPSTREAM_EVIDENCE_FIXTURE") {
        Some(path) => PathBuf::from(path),
        None => return,
    };

    let bytes = fs::read(&path).expect("read upstream-generated Evidence fixture");
    let upstream: Value =
        serde_json::from_slice(&bytes).expect("upstream REA emits valid JSON");
    let upstream_id = upstream["evidence_id"]
        .as_str()
        .expect("upstream fixture includes evidence_id");
    let upstream_profile_digest = upstream["analysis_profile"]["digest"]
        .as_str()
        .expect("upstream fixture includes analysis-profile digest");

    let verified = verify_evidence_json(&bytes)
        .unwrap_or_else(|error| panic!("Rust rejected upstream Evidence: {error}"));

    assert_eq!(verified.evidence_id, upstream_id);
    assert_eq!(
        analysis_profile_digest(&upstream["analysis_profile"]).unwrap(),
        upstream_profile_digest
    );
    assert_eq!(verified.raw_file_sha256.as_deref(), Some(
        symthaea_rea_compat::sha256_hex(&bytes).as_str()
    ));
    assert_ne!(verified.raw_file_sha256.as_deref(), Some(verified.semantic_sha256.as_str()));
    assert_ne!(verified.evidence_id, "");
    assert_ne!(verified.raw_file_sha256.as_deref(), Some(""));
    assert_eq!(
        verified.value["relationship_coverage"],
        Value::Null,
        "upstream unknown coverage omission/default must remain explicit null",
    );
    assert_ne!(
        EvidenceErrorKind::IdentityMismatch,
        EvidenceErrorKind::SchemaViolation,
        "error categories remain distinct",
    );
}
