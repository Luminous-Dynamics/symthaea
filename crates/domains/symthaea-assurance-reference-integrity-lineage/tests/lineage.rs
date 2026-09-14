use symthaea_assurance_reference_integrity::{
    MeasurementSelector, ReferenceIntegrityManifest, ReferenceMeasurementRule,
    RimSignatureVerificationReceipt,
};
use symthaea_assurance_reference_integrity_lineage::{
    qualify_reference_integrity_lineage, ReferenceIntegrityLineageEntry,
    ReferenceIntegrityLineageError, ReferenceIntegrityLineagePolicy,
    ReferenceManifestSignerRule,
};

fn d(ch: char) -> String {
    format!("blake3:{}", ch.to_string().repeat(64))
}

fn selector() -> MeasurementSelector {
    MeasurementSelector {
        component_ref: "component:firmware".into(),
        event_type: "EV_POST_CODE".into(),
        pcr_index: 0,
        hash_alg: "sha256".into(),
    }
}

fn manifest(
    revision: u64,
    predecessor: Option<String>,
    source_digest: String,
    release: &str,
    valid_from_ms: u64,
) -> ReferenceIntegrityManifest {
    ReferenceIntegrityManifest {
        schema_version: "1".into(),
        manifest_id: "rim:platform-a".into(),
        revision,
        predecessor_manifest_digest: predecessor,
        issuer_ref: "issuer:oem-a".into(),
        subject_class_ref: "platform-class:a".into(),
        release_ref: release.into(),
        source_rim_ref: format!("artifact:rim:{revision}"),
        source_rim_digest: source_digest,
        normalized_by_tool_digest: d('9'),
        valid_from_ms,
        valid_until_ms: None,
        rules: vec![ReferenceMeasurementRule {
            rule_id: "rule:firmware".into(),
            selector: selector(),
            critical: true,
            minimum_occurrences: 1,
            maximum_occurrences: Some(1),
            allowed_digests: vec![d(char::from_digit((revision % 10) as u32, 10).unwrap_or('1'))],
            denied_digests: vec![],
            evidence_refs: vec![format!("rim-rule:{revision}")],
        }],
        critical_selectors: vec![],
        evidence_refs: vec![format!("review:rim:{revision}")],
    }
}

fn signature(
    manifest: &ReferenceIntegrityManifest,
    signer_ref: &str,
    key_digest: String,
    receipt_suffix: u64,
) -> RimSignatureVerificationReceipt {
    RimSignatureVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: format!("receipt:signature:{receipt_suffix}"),
        manifest_digest: manifest.manifest_digest(),
        source_rim_digest: manifest.source_rim_digest.clone(),
        issuer_ref: manifest.issuer_ref.clone(),
        signer_ref: signer_ref.into(),
        signer_key_digest: key_digest,
        signature_artifact_digest: d('a'),
        verifier_ref: "verifier:rim".into(),
        verification_tool_digest: d('b'),
        verified_at_ms: 100 + receipt_suffix,
        signature_valid: true,
        normalized_content_matches_signed_artifact: true,
        evidence_refs: vec![format!("audit:signature:{receipt_suffix}")],
    }
}

fn lineage() -> Vec<ReferenceIntegrityLineageEntry> {
    let m1 = manifest(1, None, d('1'), "release:1", 100);
    let m2 = manifest(2, Some(m1.manifest_digest()), d('2'), "release:2", 200);
    let m3 = manifest(3, Some(m2.manifest_digest()), d('3'), "release:3", 300);
    vec![
        ReferenceIntegrityLineageEntry {
            signature: signature(&m1, "signer:oem-old", d('c'), 1),
            manifest: m1,
        },
        ReferenceIntegrityLineageEntry {
            signature: signature(&m2, "signer:oem-old", d('c'), 2),
            manifest: m2,
        },
        ReferenceIntegrityLineageEntry {
            signature: signature(&m3, "signer:oem-new", d('d'), 3),
            manifest: m3,
        },
    ]
}

fn policy(entries: &[ReferenceIntegrityLineageEntry]) -> ReferenceIntegrityLineagePolicy {
    ReferenceIntegrityLineagePolicy {
        schema_version: "1".into(),
        policy_id: "lineage-policy:1".into(),
        expected_manifest_id: "rim:platform-a".into(),
        expected_issuer_ref: "issuer:oem-a".into(),
        expected_subject_class_ref: "platform-class:a".into(),
        expected_signature_verifier_ref: "verifier:rim".into(),
        expected_signature_verification_tool_digest: d('b'),
        expected_tip_revision: entries.last().unwrap().manifest.revision,
        expected_tip_manifest_digest: entries.last().unwrap().manifest.manifest_digest(),
        signer_rules: vec![
            ReferenceManifestSignerRule {
                rule_id: "signer-rule:old".into(),
                signer_ref: "signer:oem-old".into(),
                signer_key_digest: d('c'),
                first_revision: 1,
                last_revision: Some(2),
                evidence_refs: vec!["review:old-signer".into()],
            },
            ReferenceManifestSignerRule {
                rule_id: "signer-rule:new".into(),
                signer_ref: "signer:oem-new".into(),
                signer_key_digest: d('d'),
                first_revision: 3,
                last_revision: None,
                evidence_refs: vec!["review:new-signer".into()],
            },
        ],
        evidence_refs: vec!["review:lineage-policy".into()],
    }
}

#[test]
fn contiguous_signed_lineage_with_reviewed_rotation_passes() {
    let entries = lineage();
    let report = qualify_reference_integrity_lineage(&policy(&entries), &entries).unwrap();
    assert_eq!(report.entry_count, 3);
    assert_eq!(report.tip_revision, 3);
    assert_eq!(report.tip_manifest_digest, entries[2].manifest.manifest_digest());
    assert!(!report.grants_physical_authority());
}

#[test]
fn forged_predecessor_fails() {
    let mut entries = lineage();
    entries[2].manifest.predecessor_manifest_digest = Some(d('e'));
    entries[2].signature = signature(&entries[2].manifest, "signer:oem-new", d('d'), 30);
    let result = qualify_reference_integrity_lineage(&policy(&entries), &entries);
    assert_eq!(result, Err(ReferenceIntegrityLineageError::PredecessorMismatch));
}

#[test]
fn truncated_history_cannot_satisfy_current_tip_policy() {
    let all = lineage();
    let current_policy = policy(&all);
    let truncated = all[..2].to_vec();
    let result = qualify_reference_integrity_lineage(&current_policy, &truncated);
    assert_eq!(result, Err(ReferenceIntegrityLineageError::TipRevisionMismatch));
}

#[test]
fn unreviewed_signer_replacement_fails() {
    let mut entries = lineage();
    entries[2].signature = signature(&entries[2].manifest, "signer:attacker", d('e'), 33);
    let result = qualify_reference_integrity_lineage(&policy(&entries), &entries);
    assert_eq!(result, Err(ReferenceIntegrityLineageError::UnauthorizedSigner));
}

#[test]
fn revision_gap_fails_even_with_valid_individual_signatures() {
    let mut entries = lineage();
    entries.remove(1);
    let result = qualify_reference_integrity_lineage(&policy(&entries), &entries);
    assert_eq!(result, Err(ReferenceIntegrityLineageError::RevisionGapOrReorder));
}

#[test]
fn source_rim_bytes_cannot_be_reused_as_new_revision() {
    let mut entries = lineage();
    entries[1].manifest.source_rim_digest = entries[0].manifest.source_rim_digest.clone();
    entries[1].signature = signature(&entries[1].manifest, "signer:oem-old", d('c'), 22);
    let result = qualify_reference_integrity_lineage(&policy(&entries), &entries);
    assert_eq!(result, Err(ReferenceIntegrityLineageError::SourceRimDigestReused));
}
