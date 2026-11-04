use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use symthaea_rea_compat::mobile_tool_effects::{
    DerivedToolAnnotations, MOBILE_TOOL_EFFECTS, ToolEffects, annotations_from_effects,
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

    // Compare the complete bounded mobile catalog to the actual pinned
    // upstream TOOL_EFFECTS table and its derived MCP annotations.
    let effect_manifest = &manifest["mobile_tool_effects"];
    assert_eq!(
        effect_manifest["tool_count"].as_u64(),
        Some(MOBILE_TOOL_EFFECTS.len() as u64)
    );
    assert_eq!(
        receipt["mobile_tool_effects"], *effect_manifest,
        "corpus provenance receipt must bind the effect fixture"
    );
    let effect_filename = effect_manifest["file"].as_str().unwrap();
    let effect_bytes = fixture_file(&directory, effect_filename);
    assert_eq!(
        sha256_hex(&effect_bytes),
        effect_manifest["raw_file_sha256"].as_str().unwrap()
    );
    let upstream_effects: Value = serde_json::from_slice(&effect_bytes).unwrap();
    assert_eq!(
        upstream_effects.as_object().map(|map| map.len()),
        Some(MOBILE_TOOL_EFFECTS.len())
    );
    for entry in &MOBILE_TOOL_EFFECTS {
        let upstream = &upstream_effects[entry.tool.as_str()];
        assert!(
            !upstream.is_null(),
            "pinned upstream effect row missing for {}",
            entry.tool.as_str()
        );
        let upstream_tool_effects: ToolEffects =
            serde_json::from_value(upstream["effects"].clone()).unwrap();
        assert_eq!(
            upstream_tool_effects,
            entry.effects,
            "effect row differs from pinned upstream for {}",
            entry.tool.as_str()
        );
        let upstream_annotations: DerivedToolAnnotations =
            serde_json::from_value(upstream["annotations"].clone()).unwrap();
        assert_eq!(
            annotations_from_effects(entry.effects),
            upstream_annotations,
            "derived MCP annotation differs from pinned upstream for {}",
            entry.tool.as_str()
        );
        assert_eq!(
            upstream["returns_target_content"],
            Value::Bool(entry.returns_target_content),
            "target-content trust marker differs from pinned upstream for {}",
            entry.tool.as_str()
        );
    }
}
