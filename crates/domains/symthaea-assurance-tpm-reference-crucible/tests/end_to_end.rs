use symthaea_assurance_reference_integrity::{
    MeasurementExtractionVerificationReceipt, MeasurementSelector, NormalizedMeasurementSet,
    ObservedIntegrityMeasurement, ReferenceIntegrityManifest, ReferenceIntegrityPolicy,
    ReferenceIntegrityStatus, ReferenceMeasurementRule, RimSignatureVerificationReceipt,
};
use symthaea_assurance_reference_integrity_lineage::{
    qualify_reference_integrity_lineage, ReferenceIntegrityLineageEntry,
    ReferenceIntegrityLineagePolicy, ReferenceManifestSignerRule,
};
use symthaea_assurance_tpm2_attestation_possession::{
    pcr_selection_digest, qualify_attestation_possession, AttestationChallenge,
    AttestationKeyBinding, AttestationPossessionError, AttestationPossessionPolicy,
    PcrBankSelection, QuoteVerificationReceipt, Tpm2QuoteArtifacts,
};
use symthaea_assurance_tpm2_measured_boot_replay::{
    qualify_measured_boot_replay, MeasuredBootLogBundle, MeasuredBootReplayError,
    MeasuredBootReplayPolicy, MeasuredBootReplayVerificationReceipt,
};
use symthaea_assurance_tpm2_platform_qualification::{
    qualify_tpm2_platform, Tpm2PlatformQualificationPolicy, Tpm2RuntimeSubjectEvidence,
};
use symthaea_assurance_tpm2_tools_adapter::{
    ToolExecution, Tpm2AdapterError, Tpm2NvCounterObservation, Tpm2NvPublicEvidence,
    Tpm2ReadHierarchy, Tpm2ToolsAdapterPolicy, Tpm2ToolsExecutor,
};
use symthaea_assurance_tpm_reference_crucible::{
    lower_policy_bundle_digest, qualify_tpm_reference_chain, TpmReferenceChainError,
    TpmReferenceChainInputs, TpmReferenceChainPolicy,
};

const GETCAP_PATH: &str = "/nix/store/getcap/bin/tpm2_getcap";
const NV_NAME: &str = "000bdeadbeef";

fn d(label: &str) -> String {
    format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
}

#[derive(Debug, Clone)]
struct FakeExecutor {
    getcap_stdout: Vec<u8>,
    getcap_digest: String,
}

impl Tpm2ToolsExecutor for FakeExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, Tpm2AdapterError> {
        if executable == GETCAP_PATH {
            Ok(self.getcap_digest.clone())
        } else {
            Err(Tpm2AdapterError::Io(executable.into()))
        }
    }

    fn execute(
        &self,
        executable: &str,
        args: &[String],
    ) -> Result<ToolExecution, Tpm2AdapterError> {
        assert_eq!(executable, GETCAP_PATH);
        assert_eq!(args, &["-T", "device:/dev/tpmrm0", "properties-fixed"]);
        Ok(ToolExecution {
            exit_code: Some(0),
            stdout: self.getcap_stdout.clone(),
            stderr: vec![],
        })
    }
}

fn fixed_properties_output() -> Vec<u8> {
    b"TPM2_PT_FAMILY_INDICATOR:\n  raw: 0x322E3000\n\
TPM2_PT_REVISION:\n  raw: 0xB9\n\
TPM2_PT_MANUFACTURER:\n  raw: 0x49465800\n\
TPM2_PT_FIRMWARE_VERSION_1:\n  raw: 0x47000C\n\
TPM2_PT_FIRMWARE_VERSION_2:\n  raw: 0x1020304\n"
        .to_vec()
}

fn executor() -> FakeExecutor {
    FakeExecutor {
        getcap_stdout: fixed_properties_output(),
        getcap_digest: d("getcap-tool"),
    }
}

fn adapter_policy() -> Tpm2ToolsAdapterPolicy {
    Tpm2ToolsAdapterPolicy {
        schema_version: "1".into(),
        adapter_id: "adapter:tpm2:node-1".into(),
        logical_store_id: "policy-root".into(),
        trust_store_ref: "trust-store:tpm2:node-1".into(),
        counter_epoch: "epoch:1".into(),
        nv_index: 0x0150_0016,
        expected_nv_name: NV_NAME.into(),
        tcti: "device:/dev/tpmrm0".into(),
        read_hierarchy: Tpm2ReadHierarchy::Owner,
        nvreadpublic_path: "/nix/store/public/bin/tpm2_nvreadpublic".into(),
        nvread_path: "/nix/store/read/bin/tpm2_nvread".into(),
        expected_nvreadpublic_blake3: d("nvreadpublic-tool"),
        expected_nvread_blake3: d("nvread-tool"),
        minimum_counter_value: 40,
        evidence_refs: vec!["review:adapter-policy".into()],
    }
}

fn observation(counter: u64, observed_at_ms: u64) -> Tpm2NvCounterObservation {
    let policy = adapter_policy();
    Tpm2NvCounterObservation {
        adapter_id: policy.adapter_id.clone(),
        logical_store_id: policy.logical_store_id.clone(),
        trust_store_ref: policy.trust_store_ref.clone(),
        counter_epoch: policy.counter_epoch.clone(),
        nv_index: policy.nv_index,
        tcti: policy.tcti.clone(),
        counter_value: counter,
        observed_at_ms,
        nvreadpublic_blake3: policy.expected_nvreadpublic_blake3,
        nvread_blake3: policy.expected_nvread_blake3,
        public_evidence: Tpm2NvPublicEvidence {
            nv_index: policy.nv_index,
            nv_name: NV_NAME.into(),
            attributes_friendly: "ownerread|nt=counter".into(),
            data_size: 8,
            raw_public_output_blake3: d("nv-public-output"),
        },
        raw_counter_blake3: d(&format!("counter:{counter}:{observed_at_ms}")),
        evidence_refs: vec!["capture:tpm-counter".into()],
    }
}

fn runtime_subject() -> Tpm2RuntimeSubjectEvidence {
    Tpm2RuntimeSubjectEvidence {
        adapter_artifact_ref: "git:tpm-adapter-head".into(),
        adapter_artifact_digest: d("adapter-artifact"),
        runtime_closure_ref: "nix:closure:tpm2-tools".into(),
        runtime_closure_digest: d("runtime-closure"),
        verified_by_ref: "verifier:closure".into(),
        verification_ref: "verification:closure:1".into(),
        verified_at_ms: 8_300,
        evidence_refs: vec!["audit:runtime-closure".into()],
    }
}

fn platform_policy() -> Tpm2PlatformQualificationPolicy {
    Tpm2PlatformQualificationPolicy {
        schema_version: "1".into(),
        qualification_id: "platform:node-1".into(),
        getcap_path: GETCAP_PATH.into(),
        expected_getcap_blake3: d("getcap-tool"),
        expected_family_indicator: 0x322E_3000,
        expected_specification_revision: 0xB9,
        expected_manufacturer: 0x4946_5800,
        expected_firmware_version_1: 0x0047_000C,
        expected_firmware_version_2: 0x0102_0304,
        evidence_refs: vec!["review:platform-policy".into()],
    }
}

fn selection() -> Vec<PcrBankSelection> {
    vec![PcrBankSelection {
        hash_alg: "sha256".into(),
        pcrs: vec![0, 2, 7],
    }]
}

fn selector(component: &str, event_type: &str, pcr_index: u8) -> MeasurementSelector {
    MeasurementSelector {
        component_ref: component.into(),
        event_type: event_type.into(),
        pcr_index,
        hash_alg: "sha256".into(),
    }
}

fn signature(
    manifest: &ReferenceIntegrityManifest,
    receipt_id: &str,
    signature_label: &str,
    verified_at_ms: u64,
) -> RimSignatureVerificationReceipt {
    RimSignatureVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: receipt_id.into(),
        manifest_digest: manifest.manifest_digest(),
        source_rim_digest: manifest.source_rim_digest.clone(),
        issuer_ref: manifest.issuer_ref.clone(),
        signer_ref: "signer:oem-a".into(),
        signer_key_digest: d("oem-key-a"),
        signature_artifact_digest: d(signature_label),
        verifier_ref: "verifier:rim".into(),
        verification_tool_digest: d("rim-verifier-tool"),
        verified_at_ms,
        signature_valid: true,
        normalized_content_matches_signed_artifact: true,
        evidence_refs: vec!["audit:rim-signature".into()],
    }
}

fn manifest_v1() -> ReferenceIntegrityManifest {
    ReferenceIntegrityManifest {
        schema_version: "1".into(),
        manifest_id: "rim:platform-a".into(),
        revision: 1,
        predecessor_manifest_digest: None,
        issuer_ref: "issuer:oem-a".into(),
        subject_class_ref: "platform-class:a".into(),
        release_ref: "firmware-release:2026.08".into(),
        source_rim_ref: "artifact:rim-v1".into(),
        source_rim_digest: d("source-rim-v1"),
        normalized_by_tool_digest: d("rim-normalizer"),
        valid_from_ms: 7_000,
        valid_until_ms: Some(20_000),
        rules: vec![ReferenceMeasurementRule {
            rule_id: "rule:firmware".into(),
            selector: selector("component:firmware", "EV_POST_CODE", 0),
            critical: true,
            minimum_occurrences: 1,
            maximum_occurrences: Some(1),
            allowed_digests: vec![d("firmware-good")],
            denied_digests: vec![d("firmware-bad")],
            evidence_refs: vec!["rim:firmware:v1".into()],
        }],
        critical_selectors: vec![selector(
            "component:secure-boot-policy",
            "EV_EFI_VARIABLE_DRIVER_CONFIG",
            7,
        )],
        evidence_refs: vec!["review:rim-v1".into()],
    }
}

fn manifest_v2() -> ReferenceIntegrityManifest {
    let previous = manifest_v1();
    ReferenceIntegrityManifest {
        schema_version: "1".into(),
        manifest_id: previous.manifest_id.clone(),
        revision: 2,
        predecessor_manifest_digest: Some(previous.manifest_digest()),
        issuer_ref: previous.issuer_ref.clone(),
        subject_class_ref: previous.subject_class_ref.clone(),
        release_ref: "firmware-release:2026.09".into(),
        source_rim_ref: "artifact:rim-v2".into(),
        source_rim_digest: d("source-rim-v2"),
        normalized_by_tool_digest: d("rim-normalizer"),
        valid_from_ms: 8_500,
        valid_until_ms: Some(20_000),
        rules: vec![ReferenceMeasurementRule {
            rule_id: "rule:firmware".into(),
            selector: selector("component:firmware", "EV_POST_CODE", 0),
            critical: true,
            minimum_occurrences: 1,
            maximum_occurrences: Some(1),
            allowed_digests: vec![d("firmware-good")],
            denied_digests: vec![d("firmware-bad")],
            evidence_refs: vec!["rim:firmware:v2".into()],
        }],
        critical_selectors: vec![selector(
            "component:secure-boot-policy",
            "EV_EFI_VARIABLE_DRIVER_CONFIG",
            7,
        )],
        evidence_refs: vec!["review:rim-v2".into()],
    }
}

fn fixture() -> (TpmReferenceChainPolicy, TpmReferenceChainInputs, FakeExecutor) {
    let adapter_policy = adapter_policy();
    let runtime_subject = runtime_subject();
    let before_counter = observation(42, 8_000);
    let after_counter = observation(42, 9_000);
    let platform_policy = platform_policy();
    let fake = executor();
    let platform_qualified_at_ms = 8_800;
    let platform = qualify_tpm2_platform(
        &platform_policy,
        &adapter_policy,
        &runtime_subject,
        &before_counter,
        &after_counter,
        platform_qualified_at_ms,
        &fake,
    )
    .unwrap();

    let pcr_selection = selection();
    let challenge = AttestationChallenge {
        schema_version: "1".into(),
        challenge_id: "challenge:1".into(),
        nonce_hex: "ab".repeat(32),
        issued_at_ms: 9_500,
        expires_at_ms: 10_500,
        verifier_ref: "verifier:quote".into(),
        pcr_selection: pcr_selection.clone(),
        evidence_refs: vec!["request:attestation".into()],
    };
    let attestation_key = AttestationKeyBinding {
        schema_version: "1".into(),
        ak_id: "ak:node-1".into(),
        ak_name_hex: "000b01020304".into(),
        ak_qualified_name_hex: "000b05060708".into(),
        public_key_digest: d("ak-public"),
        name_alg: "sha256".into(),
        signing_scheme: "ecdsa-sha256".into(),
        fixed_tpm: true,
        restricted_signing: true,
        reviewed_at_ms: 9_200,
        evidence_refs: vec!["review:ak".into()],
    };
    let selection_digest = pcr_selection_digest(&pcr_selection).unwrap();
    let quote = Tpm2QuoteArtifacts {
        schema_version: "1".into(),
        quote_id: "quote:1".into(),
        challenge_id: challenge.challenge_id.clone(),
        platform_qualification_digest: platform.qualification_digest(),
        qualifying_data_hex: challenge.nonce_hex.clone(),
        quote_message_digest: d("quote-message"),
        signature_digest: d("quote-signature"),
        pcr_values_digest: d("quoted-pcr-values"),
        ak_public_key_digest: attestation_key.public_key_digest.clone(),
        pcr_selection_digest: selection_digest.clone(),
        collected_at_ms: 9_800,
        evidence_refs: vec!["artifact:quote".into()],
    };
    let quote_verification = QuoteVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: "verification:quote:1".into(),
        verifier_ref: challenge.verifier_ref.clone(),
        verification_tool_ref: "tool:tpm2-checkquote".into(),
        verification_tool_digest: d("checkquote-tool"),
        verified_at_ms: 9_900,
        challenge_digest: challenge.challenge_digest(),
        ak_binding_digest: attestation_key.binding_digest(),
        quote_artifact_digest: quote.artifact_digest(),
        signature_valid: true,
        qualifying_data_matches: true,
        pcr_digest_matches: true,
        pcr_selection_matches: true,
        attestation_magic_valid: true,
        attestation_type_quote: true,
        evidence_refs: vec!["audit:quote-verification".into()],
    };
    let possession_policy = AttestationPossessionPolicy {
        schema_version: "1".into(),
        policy_id: "policy:possession".into(),
        expected_platform_qualification_digest: platform.qualification_digest(),
        expected_ak_binding_digest: attestation_key.binding_digest(),
        expected_pcr_selection_digest: selection_digest.clone(),
        expected_verifier_ref: challenge.verifier_ref.clone(),
        expected_verification_tool_digest: d("checkquote-tool"),
        max_challenge_lifetime_ms: 2_000,
        max_quote_to_verification_ms: 500,
        require_fixed_tpm: true,
        require_restricted_signing: true,
        evidence_refs: vec!["review:possession-policy".into()],
    };
    let possession_qualified_at_ms = 10_000;
    let possession = qualify_attestation_possession(
        &possession_policy,
        &platform,
        &challenge,
        &attestation_key,
        &quote,
        &quote_verification,
        possession_qualified_at_ms,
    )
    .unwrap();

    let log_bundle = MeasuredBootLogBundle {
        schema_version: "1".into(),
        boot_session_id: "boot:1".into(),
        platform_qualification_digest: platform.qualification_digest(),
        event_log_ref: "file:binary_bios_measurements".into(),
        event_log_digest: d("event-log"),
        final_events_ref: Some("file:final-events".into()),
        final_events_digest: Some(d("final-events")),
        event_count: 120,
        selected_pcr_event_count: 50,
        collected_at_ms: 10_100,
        evidence_refs: vec!["capture:measured-boot".into()],
    };
    let replay_verification = MeasuredBootReplayVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: "verification:replay:1".into(),
        verifier_ref: "verifier:measured-boot".into(),
        replay_tool_ref: "tool:eventlog-replay".into(),
        replay_tool_digest: d("eventlog-replay-tool"),
        possession_digest: possession.possession_digest(),
        quote_artifact_digest: quote.artifact_digest(),
        log_bundle_digest: log_bundle.bundle_digest(),
        event_log_digest: log_bundle.event_log_digest.clone(),
        final_events_digest: log_bundle.final_events_digest.clone(),
        pcr_selection_digest: selection_digest.clone(),
        quoted_pcr_values_digest: quote.pcr_values_digest.clone(),
        replayed_pcr_values_digest: quote.pcr_values_digest.clone(),
        event_count: log_bundle.event_count,
        selected_pcr_event_count: log_bundle.selected_pcr_event_count,
        order_preserved: true,
        all_selected_pcr_events_replayed: true,
        final_events_included: true,
        replay_matches_quoted_pcrs: true,
        verified_at_ms: 10_200,
        evidence_refs: vec!["audit:replay".into()],
    };
    let replay_policy = MeasuredBootReplayPolicy {
        schema_version: "1".into(),
        policy_id: "policy:replay".into(),
        expected_possession_digest: possession.possession_digest(),
        expected_platform_qualification_digest: platform.qualification_digest(),
        expected_pcr_selection_digest: selection_digest,
        expected_boot_session_id: log_bundle.boot_session_id.clone(),
        expected_verifier_ref: replay_verification.verifier_ref.clone(),
        expected_replay_tool_digest: replay_verification.replay_tool_digest.clone(),
        require_final_events: true,
        max_possession_to_replay_ms: 1_000,
        max_log_to_verification_ms: 500,
        evidence_refs: vec!["review:replay-policy".into()],
    };
    let replay_qualified_at_ms = 10_300;
    let replay = qualify_measured_boot_replay(
        &replay_policy,
        &possession,
        &quote,
        &log_bundle,
        &replay_verification,
        replay_qualified_at_ms,
    )
    .unwrap();

    let current_manifest = manifest_v2();
    let current_signature =
        signature(&current_manifest, "signature:rim:v2", "signature-artifact-v2", 9_100);
    let measurements = NormalizedMeasurementSet {
        schema_version: "1".into(),
        measurement_set_id: "measurements:boot-1".into(),
        replay_record_digest: replay.replay_record_digest(),
        log_bundle_digest: replay.log_bundle_digest.clone(),
        source_event_log_digest: log_bundle.event_log_digest.clone(),
        source_final_events_digest: log_bundle.final_events_digest.clone(),
        measurements: vec![ObservedIntegrityMeasurement {
            measurement_id: "measurement:firmware".into(),
            sequence: 1,
            selector: selector("component:firmware", "EV_POST_CODE", 0),
            digest: d("firmware-good"),
            source_event_ref: "event:1".into(),
        }],
        extracted_at_ms: 10_400,
        evidence_refs: vec!["artifact:event-normalization".into()],
    };
    let extraction = MeasurementExtractionVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: "verification:extraction:1".into(),
        verifier_ref: "verifier:extraction".into(),
        extraction_tool_ref: "tool:event-normalizer".into(),
        extraction_tool_digest: d("event-normalizer-tool"),
        replay_record_digest: measurements.replay_record_digest.clone(),
        measurement_set_digest: measurements.measurement_set_digest(),
        log_bundle_digest: measurements.log_bundle_digest.clone(),
        source_event_log_digest: measurements.source_event_log_digest.clone(),
        source_final_events_digest: measurements.source_final_events_digest.clone(),
        extracted_measurement_count: measurements.measurements.len() as u64,
        all_events_extracted: true,
        selected_pcr_events_covered: true,
        verified_at_ms: 10_600,
        evidence_refs: vec!["audit:extraction".into()],
    };
    let reference_policy = ReferenceIntegrityPolicy {
        schema_version: "1".into(),
        policy_id: "policy:reference".into(),
        expected_replay_record_digest: replay.replay_record_digest(),
        expected_manifest_digest: current_manifest.manifest_digest(),
        expected_subject_class_ref: current_manifest.subject_class_ref.clone(),
        expected_release_ref: current_manifest.release_ref.clone(),
        expected_manifest_verifier_ref: current_signature.verifier_ref.clone(),
        expected_manifest_verification_tool_digest: current_signature.verification_tool_digest.clone(),
        expected_extraction_verifier_ref: extraction.verifier_ref.clone(),
        expected_extraction_tool_digest: extraction.extraction_tool_digest.clone(),
        allow_unknown_noncritical: false,
        max_replay_to_evaluation_ms: 2_000,
        max_manifest_verification_age_ms: 5_000,
        max_extraction_verification_age_ms: 2_000,
        evidence_refs: vec!["review:reference-policy".into()],
    };
    let reference_evaluated_at_ms = 10_800;

    let first_manifest = manifest_v1();
    let first_signature =
        signature(&first_manifest, "signature:rim:v1", "signature-artifact-v1", 8_100);
    let lineage_entries = vec![
        ReferenceIntegrityLineageEntry {
            manifest: first_manifest,
            signature: first_signature,
        },
        ReferenceIntegrityLineageEntry {
            manifest: current_manifest.clone(),
            signature: current_signature.clone(),
        },
    ];
    let lineage_policy = ReferenceIntegrityLineagePolicy {
        schema_version: "1".into(),
        policy_id: "policy:rim-lineage".into(),
        expected_manifest_id: current_manifest.manifest_id.clone(),
        expected_issuer_ref: current_manifest.issuer_ref.clone(),
        expected_subject_class_ref: current_manifest.subject_class_ref.clone(),
        expected_signature_verifier_ref: current_signature.verifier_ref.clone(),
        expected_signature_verification_tool_digest: current_signature
            .verification_tool_digest
            .clone(),
        expected_tip_revision: current_manifest.revision,
        expected_tip_manifest_digest: current_manifest.manifest_digest(),
        signer_rules: vec![ReferenceManifestSignerRule {
            rule_id: "signer:oem-a:revisions-1-2".into(),
            signer_ref: current_signature.signer_ref.clone(),
            signer_key_digest: current_signature.signer_key_digest.clone(),
            first_revision: 1,
            last_revision: Some(2),
            evidence_refs: vec!["review:rim-signer".into()],
        }],
        evidence_refs: vec!["review:rim-lineage".into()],
    };
    qualify_reference_integrity_lineage(&lineage_policy, &lineage_entries).unwrap();

    let inputs = TpmReferenceChainInputs {
        adapter_policy,
        runtime_subject,
        before_counter,
        after_counter,
        platform_policy,
        platform_qualified_at_ms,
        possession_policy,
        challenge,
        attestation_key,
        quote,
        quote_verification,
        possession_qualified_at_ms,
        replay_policy,
        log_bundle,
        replay_verification,
        replay_qualified_at_ms,
        reference_policy,
        reference_manifest: current_manifest,
        reference_signature: current_signature,
        measurements,
        extraction,
        reference_evaluated_at_ms,
        lineage_policy,
        lineage_entries,
    };

    let policy = TpmReferenceChainPolicy {
        schema_version: "1".into(),
        campaign_id: "campaign:tpm-reference:1".into(),
        expected_adapter_id: inputs.adapter_policy.adapter_id.clone(),
        expected_trust_store_ref: inputs.adapter_policy.trust_store_ref.clone(),
        expected_runtime_subject_digest: inputs.runtime_subject.subject_digest(),
        expected_lower_policy_bundle_digest: lower_policy_bundle_digest(&inputs),
        expected_reference_policy_id: inputs.reference_policy.policy_id.clone(),
        expected_lineage_policy_id: inputs.lineage_policy.policy_id.clone(),
        max_platform_age_ms: 5_000,
        max_possession_age_ms: 5_000,
        max_replay_age_ms: 5_000,
        max_reference_age_ms: 5_000,
        evidence_refs: vec!["review:end-to-end-policy".into()],
    };

    (policy, inputs, fake)
}

#[test]
fn exact_current_chain_passes_without_granting_authority() {
    let (policy, inputs, fake) = fixture();
    let report = qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake).unwrap();
    assert!(report.report_digest().starts_with("blake3:"));
    assert_eq!(
        report.lower_policy_bundle_digest,
        policy.expected_lower_policy_bundle_digest
    );
    assert_eq!(report.current_manifest_digest, inputs.reference_manifest.manifest_digest());
    assert!(!report.grants_physical_authority());
}

#[test]
fn same_id_lower_policy_weakening_is_rejected_before_qualification() {
    let (policy, mut inputs, fake) = fixture();
    assert_eq!(inputs.reference_policy.policy_id, policy.expected_reference_policy_id);
    inputs.reference_policy.allow_unknown_noncritical = true;
    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::LowerPolicyBundleMismatch)
    );
}

#[test]
fn runtime_subject_splice_is_rejected_cross_layer() {
    let (policy, mut inputs, fake) = fixture();
    inputs.runtime_subject.runtime_closure_digest = d("substituted-runtime");
    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::RuntimeSubjectMismatch)
    );
}

#[test]
fn nonce_substitution_fails_in_fresh_possession_stage() {
    let (policy, mut inputs, fake) = fixture();
    inputs.quote.qualifying_data_hex = "cd".repeat(32);
    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::Possession(
            AttestationPossessionError::QuoteNonceMismatch
        ))
    );
}

#[test]
fn pcr_replay_mismatch_fails_before_reference_approval() {
    let (policy, mut inputs, fake) = fixture();
    inputs.replay_verification.replayed_pcr_values_digest = d("different-pcr-state");
    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::Replay(
            MeasuredBootReplayError::ReplayedPcrValuesMismatch
        ))
    );
}

#[test]
fn known_bad_reference_measurement_is_rejected_not_incomplete() {
    let (policy, mut inputs, fake) = fixture();
    inputs.measurements.measurements[0].digest = d("firmware-bad");
    inputs.extraction.measurement_set_digest = inputs.measurements.measurement_set_digest();
    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::ReferenceNotApproved(
            ReferenceIntegrityStatus::Rejected
        ))
    );
}

#[test]
fn valid_but_different_current_signature_receipt_cannot_be_spliced() {
    let (policy, mut inputs, fake) = fixture();
    inputs.reference_signature.receipt_id = "signature:rim:v2:alternate".into();
    inputs.reference_signature.signature_artifact_digest = d("alternate-signature-artifact");
    assert!(inputs
        .reference_signature
        .validate_for(&inputs.reference_manifest));

    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::ReferenceLineageSignatureSplice)
    );
}

#[test]
fn truncated_reference_history_cannot_become_current() {
    let (policy, mut inputs, fake) = fixture();
    inputs.lineage_entries.pop();
    assert!(matches!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::Lineage(_))
    ));
}

#[test]
fn stale_current_reference_blocks_the_whole_chain() {
    let (mut policy, inputs, fake) = fixture();
    policy.max_platform_age_ms = 20_000;
    policy.max_possession_age_ms = 20_000;
    policy.max_replay_age_ms = 20_000;
    policy.max_reference_age_ms = 50;
    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::StaleEvidence("reference"))
    );
}

#[test]
fn normalized_measurements_cannot_splice_a_different_event_log() {
    let (policy, mut inputs, fake) = fixture();
    inputs.measurements.source_event_log_digest = d("different-event-log");
    inputs.extraction.source_event_log_digest = inputs.measurements.source_event_log_digest.clone();
    inputs.extraction.measurement_set_digest = inputs.measurements.measurement_set_digest();

    assert_eq!(
        qualify_tpm_reference_chain(&policy, &inputs, 11_000, &fake),
        Err(TpmReferenceChainError::MeasurementSourceArtifactMismatch)
    );
}
