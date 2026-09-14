use symthaea_assurance_reference_integrity::{
    evaluate_reference_integrity, MeasurementExtractionVerificationReceipt, MeasurementSelector,
    NormalizedMeasurementSet, ObservedIntegrityMeasurement, ReferenceIntegrityError,
    ReferenceIntegrityIssue, ReferenceIntegrityManifest, ReferenceIntegrityPolicy,
    ReferenceIntegrityStatus, ReferenceMeasurementRule, RimSignatureVerificationReceipt,
};
use symthaea_assurance_tpm2_measured_boot_replay::{MeasuredBootReplayRecord, REPLAY_SCOPE};

fn d(ch: char) -> String {
    format!("blake3:{}", ch.to_string().repeat(64))
}

fn replay() -> MeasuredBootReplayRecord {
    MeasuredBootReplayRecord {
        schema_version: "1".into(),
        policy_id: "replay-policy".into(),
        possession_digest: d('a'),
        quote_artifact_digest: d('b'),
        log_bundle_digest: d('c'),
        replay_receipt_digest: d('d'),
        platform_qualification_digest: d('e'),
        pcr_selection_digest: d('1'),
        quoted_pcr_values_digest: d('2'),
        qualified_at_ms: 1_000,
        scope: REPLAY_SCOPE.into(),
        evidence_refs: vec!["evidence:replay".into()],
    }
}

fn selector(component: &str, event_type: &str, pcr: u8) -> MeasurementSelector {
    MeasurementSelector {
        component_ref: component.into(),
        event_type: event_type.into(),
        pcr_index: pcr,
        hash_alg: "sha256".into(),
    }
}

fn manifest() -> ReferenceIntegrityManifest {
    ReferenceIntegrityManifest {
        schema_version: "1".into(),
        manifest_id: "rim:platform-a".into(),
        revision: 3,
        predecessor_manifest_digest: Some(d('3')),
        issuer_ref: "issuer:oem-a".into(),
        subject_class_ref: "platform-class:a".into(),
        release_ref: "firmware-release:2026.09".into(),
        source_rim_ref: "artifact:rim-a".into(),
        source_rim_digest: d('4'),
        normalized_by_tool_digest: d('5'),
        valid_from_ms: 500,
        valid_until_ms: Some(10_000),
        rules: vec![
            ReferenceMeasurementRule {
                rule_id: "rule:firmware".into(),
                selector: selector("component:firmware", "EV_POST_CODE", 0),
                critical: true,
                minimum_occurrences: 1,
                maximum_occurrences: Some(1),
                allowed_digests: vec![d('6')],
                denied_digests: vec![d('f')],
                evidence_refs: vec!["rim:firmware".into()],
            },
            ReferenceMeasurementRule {
                rule_id: "rule:bootloader".into(),
                selector: selector("component:bootloader", "EV_EFI_BOOT_SERVICES_APPLICATION", 4),
                critical: true,
                minimum_occurrences: 1,
                maximum_occurrences: Some(1),
                allowed_digests: vec![d('7')],
                denied_digests: vec![],
                evidence_refs: vec!["rim:bootloader".into()],
            },
        ],
        critical_selectors: vec![selector(
            "component:secure-boot-policy",
            "EV_EFI_VARIABLE_DRIVER_CONFIG",
            7,
        )],
        evidence_refs: vec!["review:rim-normalization".into()],
    }
}

fn signature(manifest: &ReferenceIntegrityManifest) -> RimSignatureVerificationReceipt {
    RimSignatureVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: "receipt:rim-signature".into(),
        manifest_digest: manifest.manifest_digest(),
        source_rim_digest: manifest.source_rim_digest.clone(),
        issuer_ref: manifest.issuer_ref.clone(),
        signer_ref: "signer:oem-a".into(),
        signer_key_digest: d('9'),
        signature_artifact_digest: d('a'),
        verifier_ref: "verifier:rim".into(),
        verification_tool_digest: d('b'),
        verified_at_ms: 800,
        signature_valid: true,
        normalized_content_matches_signed_artifact: true,
        evidence_refs: vec!["audit:rim-signature".into()],
    }
}

fn measurements(replay: &MeasuredBootReplayRecord) -> NormalizedMeasurementSet {
    NormalizedMeasurementSet {
        schema_version: "1".into(),
        measurement_set_id: "measurements:boot-1".into(),
        replay_record_digest: replay.replay_record_digest(),
        log_bundle_digest: replay.log_bundle_digest.clone(),
        source_event_log_digest: d('c'),
        source_final_events_digest: Some(d('d')),
        measurements: vec![
            ObservedIntegrityMeasurement {
                measurement_id: "measurement:firmware".into(),
                sequence: 1,
                selector: selector("component:firmware", "EV_POST_CODE", 0),
                digest: d('6'),
                source_event_ref: "event:1".into(),
            },
            ObservedIntegrityMeasurement {
                measurement_id: "measurement:bootloader".into(),
                sequence: 2,
                selector: selector(
                    "component:bootloader",
                    "EV_EFI_BOOT_SERVICES_APPLICATION",
                    4,
                ),
                digest: d('7'),
                source_event_ref: "event:2".into(),
            },
        ],
        extracted_at_ms: 1_100,
        evidence_refs: vec!["artifact:event-normalization".into()],
    }
}

fn extraction(set: &NormalizedMeasurementSet) -> MeasurementExtractionVerificationReceipt {
    MeasurementExtractionVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: "receipt:extraction".into(),
        verifier_ref: "verifier:extraction".into(),
        extraction_tool_ref: "tool:event-normalizer".into(),
        extraction_tool_digest: d('e'),
        replay_record_digest: set.replay_record_digest.clone(),
        measurement_set_digest: set.measurement_set_digest(),
        log_bundle_digest: set.log_bundle_digest.clone(),
        source_event_log_digest: set.source_event_log_digest.clone(),
        source_final_events_digest: set.source_final_events_digest.clone(),
        extracted_measurement_count: set.measurements.len() as u64,
        all_events_extracted: true,
        selected_pcr_events_covered: true,
        verified_at_ms: 1_200,
        evidence_refs: vec!["audit:extraction".into()],
    }
}

fn policy(
    replay: &MeasuredBootReplayRecord,
    manifest: &ReferenceIntegrityManifest,
) -> ReferenceIntegrityPolicy {
    ReferenceIntegrityPolicy {
        schema_version: "1".into(),
        policy_id: "reference-policy:1".into(),
        expected_replay_record_digest: replay.replay_record_digest(),
        expected_manifest_digest: manifest.manifest_digest(),
        expected_subject_class_ref: manifest.subject_class_ref.clone(),
        expected_release_ref: manifest.release_ref.clone(),
        expected_manifest_verifier_ref: "verifier:rim".into(),
        expected_manifest_verification_tool_digest: d('b'),
        expected_extraction_verifier_ref: "verifier:extraction".into(),
        expected_extraction_tool_digest: d('e'),
        allow_unknown_noncritical: false,
        max_replay_to_evaluation_ms: 5_000,
        max_manifest_verification_age_ms: 5_000,
        max_extraction_verification_age_ms: 5_000,
        evidence_refs: vec!["review:reference-policy".into()],
    }
}

fn evaluate(
    replay: &MeasuredBootReplayRecord,
    manifest: &ReferenceIntegrityManifest,
    set: &NormalizedMeasurementSet,
    extraction: &MeasurementExtractionVerificationReceipt,
    policy: &ReferenceIntegrityPolicy,
) -> symthaea_assurance_reference_integrity::ReferenceIntegrityEvaluationReport {
    evaluate_reference_integrity(
        policy,
        replay,
        manifest,
        &signature(manifest),
        set,
        extraction,
        1_500,
    )
    .unwrap()
}

#[test]
fn exact_signed_reference_state_is_approved() {
    let replay = replay();
    let manifest = manifest();
    let set = measurements(&replay);
    let extraction = extraction(&set);
    let report = evaluate(
        &replay,
        &manifest,
        &set,
        &extraction,
        &policy(&replay, &manifest),
    );
    assert_eq!(report.status, ReferenceIntegrityStatus::Approved);
    assert!(report.issues.is_empty());
    assert!(!report.grants_physical_authority());
}

#[test]
fn known_bad_and_unknown_critical_measurements_are_rejected() {
    let replay = replay();
    let manifest = manifest();

    let mut known_bad = measurements(&replay);
    known_bad.measurements[0].digest = d('f');
    let known_bad_extraction = extraction(&known_bad);
    let report = evaluate(
        &replay,
        &manifest,
        &known_bad,
        &known_bad_extraction,
        &policy(&replay, &manifest),
    );
    assert_eq!(report.status, ReferenceIntegrityStatus::Rejected);
    assert!(report
        .issues
        .iter()
        .any(|issue| matches!(issue, ReferenceIntegrityIssue::ExplicitlyDenied { .. })));

    let mut unknown = measurements(&replay);
    unknown.measurements[0].digest = d('0');
    let unknown_extraction = extraction(&unknown);
    let report = evaluate(
        &replay,
        &manifest,
        &unknown,
        &unknown_extraction,
        &policy(&replay, &manifest),
    );
    assert_eq!(report.status, ReferenceIntegrityStatus::Rejected);
    assert!(report
        .issues
        .iter()
        .any(|issue| matches!(issue, ReferenceIntegrityIssue::UnknownCritical { .. })));
}

#[test]
fn missing_required_known_good_is_incomplete() {
    let replay = replay();
    let manifest = manifest();
    let mut set = measurements(&replay);
    set.measurements.pop();
    let extraction = extraction(&set);
    let report = evaluate(
        &replay,
        &manifest,
        &set,
        &extraction,
        &policy(&replay, &manifest),
    );
    assert_eq!(report.status, ReferenceIntegrityStatus::Incomplete);
    assert_eq!(report.missing_rule_ids, vec!["rule:bootloader".to_string()]);
}

#[test]
fn unknown_noncritical_is_explicitly_policy_gated() {
    let replay = replay();
    let manifest = manifest();
    let mut set = measurements(&replay);
    set.measurements.push(ObservedIntegrityMeasurement {
        measurement_id: "measurement:unknown-device".into(),
        sequence: 3,
        selector: selector("component:other-device", "EV_EFI_DRIVER_CONFIG", 2),
        digest: d('3'),
        source_event_ref: "event:3".into(),
    });
    let extraction = extraction(&set);

    let blocked = evaluate(
        &replay,
        &manifest,
        &set,
        &extraction,
        &policy(&replay, &manifest),
    );
    assert_eq!(blocked.status, ReferenceIntegrityStatus::Incomplete);

    let mut permissive = policy(&replay, &manifest);
    permissive.allow_unknown_noncritical = true;
    let approved = evaluate(&replay, &manifest, &set, &extraction, &permissive);
    assert_eq!(approved.status, ReferenceIntegrityStatus::Approved);
    assert_eq!(
        approved.unknown_measurement_ids,
        vec!["measurement:unknown-device".to_string()]
    );
}

#[test]
fn expired_manifest_and_incomplete_extraction_cannot_approve() {
    let replay = replay();
    let mut manifest = manifest();
    manifest.valid_until_ms = Some(1_400);
    let set = measurements(&replay);
    let extraction = extraction(&set);
    let report = evaluate(
        &replay,
        &manifest,
        &set,
        &extraction,
        &policy(&replay, &manifest),
    );
    assert_eq!(report.status, ReferenceIntegrityStatus::Incomplete);
    assert!(report
        .issues
        .contains(&ReferenceIntegrityIssue::ManifestExpired));

    let manifest = manifest();
    let set = measurements(&replay);
    let mut incomplete = extraction(&set);
    incomplete.all_events_extracted = false;
    let result = evaluate_reference_integrity(
        &policy(&replay, &manifest),
        &replay,
        &manifest,
        &signature(&manifest),
        &set,
        &incomplete,
        1_500,
    );
    assert_eq!(result, Err(ReferenceIntegrityError::InvalidExtractionReceipt));
}

#[test]
fn signed_source_must_cover_normalized_reference_content() {
    let replay = replay();
    let manifest = manifest();
    let set = measurements(&replay);
    let extraction = extraction(&set);
    let mut signature = signature(&manifest);
    signature.normalized_content_matches_signed_artifact = false;

    let result = evaluate_reference_integrity(
        &policy(&replay, &manifest),
        &replay,
        &manifest,
        &signature,
        &set,
        &extraction,
        1_500,
    );
    assert_eq!(
        result,
        Err(ReferenceIntegrityError::InvalidManifestSignatureReceipt)
    );
}
