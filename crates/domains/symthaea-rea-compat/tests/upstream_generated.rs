use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use symthaea_rea_compat::bundle::{
    BundleErrorKind, audit_evidence_link_closure, evidence_bundle_for_target,
    verify_evidence_bundle_json,
};
use symthaea_rea_compat::{
    EvidenceErrorKind, analysis_profile_digest, sha256_hex, verify_evidence_json,
};

fn fixture_file(directory: &Path, name: &str) -> Vec<u8> {
    assert!(
        !name.contains('/') && !name.contains('\\') && !name.contains(".."),
        "fixture manifest may only contain a simple filename"
    );
    fs::read(directory.join(name)).expect("read upstream-generated fixture")
}

fn verify_display_only_mutation(
    directory: &Path,
    file: &str,
    expected_id: &str,
    expected_raw_sha256: &str,
) {
    let bytes = fixture_file(directory, file);
    assert_eq!(sha256_hex(&bytes), expected_raw_sha256);

    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
    if let Some(subject) = value.get_mut("subject").and_then(Value::as_object_mut) {
        subject.insert(
            "name".to_owned(),
            Value::String("renamed-display-only.apk".to_owned()),
        );
        subject.insert(
            "local_path".to_owned(),
            Value::String("/relocated/display-only.apk".to_owned()),
        );
    }
    value["caller_input_path"] = Value::String("/relocated/caller-display-only.json".to_owned());

    let relocated = serde_json::to_vec(&value).unwrap();
    let verified = verify_evidence_json(&relocated)
        .unwrap_or_else(|error| panic!("display-only mutation changed Evidence identity: {error}"));
    assert_eq!(verified.evidence_id, expected_id);
    assert_ne!(
        verified.raw_file_sha256.as_deref(),
        Some(expected_raw_sha256),
        "raw-file digest must reflect display-only byte changes"
    );
}

fn verify_semantic_preserving_mutation(
    directory: &Path,
    file: &str,
    expected_id: &str,
    expected_raw_sha256: &str,
) {
    let bytes = fixture_file(directory, file);
    assert_eq!(sha256_hex(&bytes), expected_raw_sha256);
    let verified = verify_evidence_json(&bytes)
        .unwrap_or_else(|error| panic!("semantics-preserving mutation was rejected: {error}"));
    assert_eq!(verified.evidence_id, expected_id);
}

fn verify_stricter_importer_case(directory: &Path, fixture: &Value) {
    assert_eq!(fixture["upstream_object_parse_accepted"], Value::Bool(true));
    let bytes = fixture_file(directory, fixture["file"].as_str().unwrap());
    assert_eq!(
        sha256_hex(&bytes),
        fixture["raw_file_sha256"].as_str().unwrap()
    );
    let error = verify_evidence_json(&bytes)
        .expect_err("isolated UTF-16 surrogate must be an explicit raw-import rejection");
    assert_eq!(error.kind, EvidenceErrorKind::InvalidJson);
}

fn verify_negative_control(directory: &Path, fixture: &Value) {
    let filename = fixture["file"].as_str().expect("negative fixture file");
    let expected_hash = fixture["raw_file_sha256"]
        .as_str()
        .expect("negative fixture raw hash");
    let bytes = fixture_file(directory, filename);
    assert_eq!(sha256_hex(&bytes), expected_hash);

    let error = verify_evidence_json(&bytes)
        .expect_err("upstream-rejected negative control must not verify");
    let expected_kind = match fixture["expected_error"].as_str() {
        Some("identity-mismatch") => EvidenceErrorKind::IdentityMismatch,
        Some("schema-violation") => EvidenceErrorKind::SchemaViolation,
        other => panic!("unsupported expected error category: {other:?}"),
    };
    assert_eq!(
        error.kind, expected_kind,
        "negative control {} had an unexpected Rust disposition",
        fixture["name"]
    );
}

/// This test is activated only when the pinned upstream REA workflow supplies
/// an independently generated fixture corpus. The source-derived unit vector
/// remains a separate baseline and is not relabelled as upstream output.
#[test]
fn matches_pinned_upstream_rea_evidence_corpus() {
    let directory = match std::env::var_os("REA_UPSTREAM_EVIDENCE_FIXTURE_DIR") {
        Some(path) => PathBuf::from(path),
        None => return,
    };

    let manifest_bytes =
        fs::read(directory.join("manifest.json")).expect("read upstream corpus manifest");
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let receipt_bytes = fs::read(directory.join("manifest.provenance.json"))
        .expect("read upstream corpus provenance receipt");
    let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();

    assert_eq!(
        sha256_hex(&manifest_bytes),
        receipt["manifest_sha256"].as_str().unwrap(),
        "corpus manifest must match its provenance receipt"
    );
    assert_eq!(
        manifest["producer"]["commit"],
        "8250bf2320ea57aab4e664db4b4d44b8abd2004e"
    );
    assert_eq!(
        manifest["producer"]["node_version"], "v22.23.4",
        "the upstream producer runtime must match the pinned CI runtime"
    );
    assert_eq!(manifest["producer"]["npm_version"], "11.16.0");
    assert_eq!(manifest["producer"]["package_version"], "6.8.0");
    assert_eq!(receipt["producer"], manifest["producer"]);
    assert_eq!(receipt["generator"], manifest["generator"]);

    let fixtures = manifest["fixtures"]
        .as_array()
        .expect("manifest fixtures array");
    assert_eq!(
        manifest["fixture_count"].as_u64(),
        Some(fixtures.len() as u64)
    );
    assert!(
        fixtures.len() >= 8,
        "the upstream corpus must cover multiple schema families"
    );

    for fixture in fixtures {
        let filename = fixture["file"].as_str().expect("fixture filename");
        let expected_sha256 = fixture["raw_file_sha256"]
            .as_str()
            .expect("fixture raw digest");
        let expected_id = fixture["evidence_id"]
            .as_str()
            .expect("upstream Evidence ID");
        let bytes = fixture_file(&directory, filename);
        assert_eq!(bytes.len() as u64, fixture["bytes"].as_u64().unwrap());
        assert_eq!(sha256_hex(&bytes), expected_sha256);

        let upstream: Value = serde_json::from_slice(&bytes).unwrap();
        let upstream_id = upstream["evidence_id"].as_str().unwrap();
        assert_eq!(upstream_id, expected_id);

        let verified = verify_evidence_json(&bytes)
            .unwrap_or_else(|error| panic!("Rust rejected upstream case {filename}: {error}"));
        assert_eq!(verified.evidence_id, upstream_id);
        assert_eq!(verified.raw_file_sha256.as_deref(), Some(expected_sha256));
        assert_ne!(
            verified.raw_file_sha256.as_deref(),
            Some(verified.semantic_sha256.as_str())
        );

        if let Some(profile) = upstream
            .get("analysis_profile")
            .filter(|value| !value.is_null())
        {
            let expected_profile_digest = fixture["analysis_profile_digest"]
                .as_str()
                .expect("profile digest in manifest");
            assert_eq!(profile["digest"], expected_profile_digest);
            assert_eq!(
                analysis_profile_digest(profile).unwrap(),
                expected_profile_digest
            );
        } else {
            assert!(fixture["analysis_profile_digest"].is_null());
        }

        assert_eq!(
            upstream["relationship_coverage"], fixture["relationship_coverage"],
            "coverage state must be retained for {filename}"
        );

        let mut tampered = upstream.clone();
        tampered["predicate_type"] = Value::String(format!(
            "{}.tampered",
            tampered["predicate_type"].as_str().unwrap()
        ));
        let tampered_bytes = serde_json::to_vec(&tampered).unwrap();
        assert_eq!(
            verify_evidence_json(&tampered_bytes).unwrap_err().kind,
            EvidenceErrorKind::IdentityMismatch,
            "semantic mutation must invalidate Evidence ID for {filename}"
        );
    }

    let display_mutations = manifest["display_only_mutations"]
        .as_array()
        .expect("display-only mutations array");
    assert_eq!(display_mutations.len(), fixtures.len());
    for mutation in display_mutations {
        verify_display_only_mutation(
            &directory,
            mutation["file"].as_str().unwrap(),
            mutation["expected_evidence_id"].as_str().unwrap(),
            mutation["raw_file_sha256"].as_str().unwrap(),
        );
    }

    let key_order_mutations = manifest["key_order_mutations"]
        .as_array()
        .expect("key-order mutation array");
    assert_eq!(key_order_mutations.len(), fixtures.len());
    for mutation in key_order_mutations {
        verify_semantic_preserving_mutation(
            &directory,
            mutation["file"].as_str().unwrap(),
            mutation["expected_evidence_id"].as_str().unwrap(),
            mutation["raw_file_sha256"].as_str().unwrap(),
        );
    }

    let negative_controls = manifest["negative_controls"]
        .as_array()
        .expect("negative controls array");
    assert!(
        negative_controls.len() >= 6,
        "upstream rejected negative-control corpus is required"
    );
    for negative in negative_controls {
        verify_negative_control(&directory, negative);
    }

    let stricter_importer_cases = manifest["stricter_importer_cases"]
        .as_array()
        .expect("stricter importer case array");
    assert_eq!(stricter_importer_cases.len(), 1);
    for stricter in stricter_importer_cases {
        verify_stricter_importer_case(&directory, stricter);
    }
}

/// Verify a full EvidenceBundle and residual-unknown graph generated by the
/// pinned upstream REA TypeScript implementation.
#[test]
fn matches_pinned_upstream_rea_evidence_bundle_and_unknown_graph() {
    let directory = match std::env::var_os("REA_UPSTREAM_EVIDENCE_FIXTURE_DIR") {
        Some(path) => PathBuf::from(path),
        None => return,
    };
    let manifest_bytes =
        fs::read(directory.join("manifest.json")).expect("read upstream corpus manifest");
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let receipt_bytes = fs::read(directory.join("manifest.provenance.json"))
        .expect("read upstream corpus provenance receipt");
    let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();

    assert_eq!(
        sha256_hex(&manifest_bytes),
        receipt["manifest_sha256"].as_str().unwrap()
    );
    assert_eq!(
        manifest["producer"]["commit"],
        "8250bf2320ea57aab4e664db4b4d44b8abd2004e"
    );
    assert_eq!(receipt["producer"], manifest["producer"]);

    let bundle_meta = &manifest["evidence_bundle"];
    assert!(
        bundle_meta.is_object(),
        "upstream generator must emit an EvidenceBundle fixture"
    );
    assert_eq!(
        receipt["evidence_bundle_raw_sha256"],
        bundle_meta["raw_file_sha256"]
    );
    assert_eq!(
        receipt["evidence_bundle_canonical_sha256"],
        bundle_meta["canonical_sha256"]
    );

    let filename = bundle_meta["file"].as_str().unwrap();
    let bytes = fixture_file(&directory, filename);
    assert_eq!(bytes.len() as u64, bundle_meta["bytes"].as_u64().unwrap());
    assert_eq!(
        sha256_hex(&bytes),
        bundle_meta["raw_file_sha256"].as_str().unwrap()
    );

    let upstream: Value = serde_json::from_slice(&bytes).unwrap();
    let verified = verify_evidence_bundle_json(&bytes)
        .unwrap_or_else(|error| panic!("Rust rejected upstream EvidenceBundle: {error}"));
    assert_eq!(
        verified.raw_file_sha256.as_deref(),
        bundle_meta["raw_file_sha256"].as_str()
    );
    assert_eq!(
        verified.canonical_sha256,
        bundle_meta["canonical_sha256"].as_str().unwrap()
    );
    assert_eq!(
        verified.record_count as u64,
        bundle_meta["record_count"].as_u64().unwrap()
    );
    assert_eq!(
        verified.unknown_revision_count as u64,
        bundle_meta["unknown_revision_count"].as_u64().unwrap()
    );
    assert_eq!(verified.bundle, upstream);
    assert!(
        audit_evidence_link_closure(&verified.bundle)
            .expect("strict link-closure audit")
            .is_empty(),
        "upstream fixture should have internally closed Evidence links"
    );

    let target_meta = &manifest["evidence_bundle_target"];
    let target_bytes = fixture_file(&directory, target_meta["file"].as_str().unwrap());
    assert_eq!(
        sha256_hex(&target_bytes),
        target_meta["raw_file_sha256"].as_str().unwrap()
    );
    let target_upstream: Value = serde_json::from_slice(&target_bytes).unwrap();
    let projected =
        evidence_bundle_for_target(&upstream, target_meta["target_sha256"].as_str().unwrap())
            .unwrap();
    assert_eq!(projected, target_upstream);
    let target_verified = verify_evidence_bundle_json(&target_bytes).unwrap();
    assert_eq!(target_verified.bundle, projected);
    assert_eq!(
        target_verified.canonical_sha256,
        target_meta["canonical_sha256"].as_str().unwrap()
    );
    assert_eq!(
        target_verified.record_count as u64,
        target_meta["record_count"].as_u64().unwrap()
    );
    assert_eq!(
        target_verified.unknown_revision_count as u64,
        target_meta["unknown_revision_count"].as_u64().unwrap()
    );

    let other_target_meta = &manifest["evidence_bundle_other_target"];
    let other_target_bytes = fixture_file(&directory, other_target_meta["file"].as_str().unwrap());
    assert_eq!(
        sha256_hex(&other_target_bytes),
        other_target_meta["raw_file_sha256"].as_str().unwrap()
    );
    let other_target_upstream: Value = serde_json::from_slice(&other_target_bytes).unwrap();
    let other_projected = evidence_bundle_for_target(
        &upstream,
        other_target_meta["target_sha256"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(other_projected, other_target_upstream);
    assert!(
        other_projected["records"].as_array().unwrap().is_empty(),
        "other artifact projection must not leak records from the original target"
    );
    assert!(
        other_projected["unknowns"].as_array().unwrap().is_empty(),
        "other artifact projection must not leak residual-unknown histories"
    );
    let other_verified = verify_evidence_bundle_json(&other_target_bytes).unwrap();
    assert_eq!(other_verified.bundle, other_projected);
    assert_eq!(
        other_verified.canonical_sha256,
        other_target_meta["canonical_sha256"].as_str().unwrap()
    );

    let negative_controls = manifest["bundle_negative_controls"]
        .as_array()
        .expect("bundle negative controls array");
    assert!(
        negative_controls.len() >= 4,
        "upstream bundle generator must emit graph and canonical-order controls"
    );
    for fixture in negative_controls {
        let negative_bytes = fixture_file(&directory, fixture["file"].as_str().unwrap());
        assert_eq!(
            sha256_hex(&negative_bytes),
            fixture["raw_file_sha256"].as_str().unwrap()
        );
        let error = verify_evidence_bundle_json(&negative_bytes)
            .expect_err("upstream-rejected EvidenceBundle must not verify");
        let expected_kind = match fixture["expected_error"].as_str() {
            Some("DuplicateEvidenceRecordError") => BundleErrorKind::DuplicateEvidenceId,
            Some("NonCanonicalEvidenceBundleError") => BundleErrorKind::NonCanonicalBundle,
            Some("InvalidResidualUnknownGraphError") => BundleErrorKind::UnknownGraphInvalid,
            other => panic!("unsupported upstream bundle error class: {other:?}"),
        };
        assert_eq!(
            error.kind, expected_kind,
            "bundle negative control {} had an unexpected Rust disposition: {}",
            fixture["name"], error
        );
    }
}
