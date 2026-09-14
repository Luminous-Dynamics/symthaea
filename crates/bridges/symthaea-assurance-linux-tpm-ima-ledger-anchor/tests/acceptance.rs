use sha2::{Digest, Sha256};
use symthaea_assurance_linux_tpm_ima_ledger_anchor::*;
use symthaea_assurance_tpm_ima_pcr_binding::{
    bind_verified_tpm_ima_pcr, TpmImaPcrBindingPolicy, TpmImaPcrBindingQualification,
    TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1,
};
use symthaea_assurance_tpm2_attestation_possession::{
    pcr_selection_digest, AttestationAcceptanceLedger, AttestationAcceptanceRecord,
    AttestationChallenge, AttestationKeyBinding, AttestationPossessionPolicy, PcrBankSelection,
};
use symthaea_assurance_tpm2_checkquote_adapter::{
    verify_tpm2_quote, CheckquoteExecutionRequest, CheckquoteExecutor, RawTpm2QuoteBundle,
    Sha256PcrValue, ToolExecution, Tpm2CheckquotePolicy, Tpm2QuoteQualification,
    CHECKQUOTE_POLICY_SCHEMA_V1,
};
use symthaea_assurance_tpm2_platform_qualification::{
    Tpm2FixedProperties, Tpm2PlatformQualificationRecord,
};
use symthaea_assurance_tpm2_possession_policy_binding::canonical_possession_policy_digest;
use symthaea_linux_ima_replay::{
    verify_canonical_ima_sha256, ImaReplayPolicy, RequiredImaMeasurement, SupportedImaTemplate,
    IMA_REPLAY_POLICY_SCHEMA_V1,
};

const NONCE: [u8; 32] = [0xab; 32];
const QUALIFIED_SIGNER: [u8; 34] = [0x5a; 34];
const TPM2_GENERATED_VALUE: u32 = 0xff54_4347;
const TPM2_ST_ATTEST_QUOTE: u16 = 0x8018;
const TPM2_ALG_SHA256: u16 = 0x000b;

#[derive(Clone)]
struct FakeExecutor {
    digest: String,
}

impl CheckquoteExecutor for FakeExecutor {
    fn executable_blake3(&self, _executable: &str) -> Result<String, String> {
        Ok(self.digest.clone())
    }

    fn execute_checkquote(
        &self,
        _executable: &str,
        _request: &CheckquoteExecutionRequest,
    ) -> Result<ToolExecution, String> {
        Ok(ToolExecution {
            exit_code: Some(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
}

struct Fixture {
    platform: Tpm2PlatformQualificationRecord,
    challenge: AttestationChallenge,
    ak: AttestationKeyBinding,
    quote: Tpm2QuoteQualification,
    pcr_binding: TpmImaPcrBindingQualification,
    possession_policy: AttestationPossessionPolicy,
    anchor_policy: LinuxTpmImaLedgerAnchorPolicy,
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn platform() -> Tpm2PlatformQualificationRecord {
    Tpm2PlatformQualificationRecord {
        schema_version: "1".into(),
        qualification_id: "platform:q1".into(),
        adapter_id: "adapter:tpm2:1".into(),
        trust_store_ref: "trust-store:tpm2:1".into(),
        tcti: "device:/dev/tpmrm0".into(),
        nv_index: 0x0150_0016,
        nv_name: "000bdeadbeef".into(),
        before_observation_digest: digest(b"before"),
        after_observation_digest: digest(b"after"),
        before_counter_value: 42,
        after_counter_value: 42,
        fixed_properties: Tpm2FixedProperties {
            family_indicator: 0x322e3000,
            specification_revision: 185,
            manufacturer: 0x49465800,
            firmware_version_1: 1,
            firmware_version_2: 2,
            raw_output_digest: digest(b"getcap"),
        },
        getcap_blake3: digest(b"getcap-bin"),
        runtime_subject_digest: digest(b"runtime"),
        qualified_at_ms: 9_000,
        evidence_refs: vec!["audit:platform".into()],
    }
}

fn challenge() -> AttestationChallenge {
    AttestationChallenge {
        schema_version: "1".into(),
        challenge_id: "challenge:ledger-anchor:1".into(),
        nonce_hex: hex_lower(&NONCE),
        issued_at_ms: 10_000,
        expires_at_ms: 11_000,
        verifier_ref: "verifier:remote-1".into(),
        pcr_selection: vec![PcrBankSelection {
            hash_alg: "sha256".into(),
            pcrs: vec![10],
        }],
        evidence_refs: vec!["request:ledger-anchor:1".into()],
    }
}

fn ak(public: &[u8]) -> AttestationKeyBinding {
    AttestationKeyBinding {
        schema_version: "1".into(),
        ak_id: "ak:node-1".into(),
        ak_name_hex: "000b01020304".into(),
        ak_qualified_name_hex: hex_lower(&QUALIFIED_SIGNER),
        public_key_digest: digest(public),
        name_alg: "sha256".into(),
        signing_scheme: "ecdsa-sha256".into(),
        fixed_tpm: true,
        restricted_signing: true,
        reviewed_at_ms: 9_500,
        evidence_refs: vec!["review:ak".into()],
    }
}

fn quote_policy(tool_digest: String) -> Tpm2CheckquotePolicy {
    Tpm2CheckquotePolicy {
        schema_version: CHECKQUOTE_POLICY_SCHEMA_V1.into(),
        adapter_id: "adapter:tpm2-checkquote:ledger-anchor".into(),
        verifier_ref: "verifier:remote-1".into(),
        checkquote_path: "/nix/store/example/bin/tpm2_checkquote".into(),
        expected_checkquote_blake3: tool_digest,
        hash_algorithm: "sha256".into(),
        max_public_bytes: 1024 * 1024,
        max_message_bytes: 1024 * 1024,
        max_signature_bytes: 1024 * 1024,
        max_tool_output_bytes: 1024 * 1024,
        evidence_refs: vec!["review:checkquote".into()],
    }
}

fn put_tpm2b(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u16).to_be_bytes());
    target.extend_from_slice(value);
}

fn quote_message(pcr_value: [u8; 32]) -> Vec<u8> {
    let pcr_digest: [u8; 32] = Sha256::digest(pcr_value).into();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&TPM2_GENERATED_VALUE.to_be_bytes());
    bytes.extend_from_slice(&TPM2_ST_ATTEST_QUOTE.to_be_bytes());
    put_tpm2b(&mut bytes, &QUALIFIED_SIGNER);
    put_tpm2b(&mut bytes, &NONCE);
    bytes.extend_from_slice(&123u64.to_be_bytes());
    bytes.extend_from_slice(&1u32.to_be_bytes());
    bytes.extend_from_slice(&2u32.to_be_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&7u64.to_be_bytes());
    bytes.extend_from_slice(&1u32.to_be_bytes());
    bytes.extend_from_slice(&TPM2_ALG_SHA256.to_be_bytes());
    bytes.push(3);
    let mut bitmap = [0u8; 3];
    bitmap[1] = 1u8 << 2;
    bytes.extend_from_slice(&bitmap);
    put_tpm2b(&mut bytes, &pcr_digest);
    bytes
}

fn ima_fixture() -> (ImaReplayPolicy, Vec<u8>, [u8; 32]) {
    let event: [u8; 32] = Sha256::digest(b"verifier-executable").into();
    let mut d_ng = b"sha256:\0".to_vec();
    d_ng.extend_from_slice(&event);
    let n_ng = b"/nix/store/example/bin/verifier\0".to_vec();

    let mut data = Vec::new();
    data.extend_from_slice(&(d_ng.len() as u32).to_le_bytes());
    data.extend_from_slice(&d_ng);
    data.extend_from_slice(&(n_ng.len() as u32).to_le_bytes());
    data.extend_from_slice(&n_ng);
    let template_digest: [u8; 32] = Sha256::digest(&data).into();

    let mut record = Vec::new();
    record.extend_from_slice(&10u32.to_le_bytes());
    record.extend_from_slice(&template_digest);
    record.extend_from_slice(&6u32.to_le_bytes());
    record.extend_from_slice(b"ima-ng");
    record.extend_from_slice(&(data.len() as u32).to_le_bytes());
    record.extend_from_slice(&data);

    let mut extend = Sha256::new();
    extend.update([0u8; 32]);
    extend.update(template_digest);
    let final_pcr: [u8; 32] = extend.finalize().into();

    let policy = ImaReplayPolicy {
        schema_version: IMA_REPLAY_POLICY_SCHEMA_V1.into(),
        policy_id: "policy:ima:ledger-anchor".into(),
        expected_pcr: 10,
        expected_initial_pcr: [0u8; 32],
        allowed_templates: vec![SupportedImaTemplate::ImaNg],
        required_measurements: vec![RequiredImaMeasurement {
            measurement_id: "verifier-executable".into(),
            event_sha256: event,
            event_name_sha256: None,
        }],
        max_measurement_list_bytes: 1024 * 1024,
        max_records: 16,
        max_template_name_bytes: 64,
        max_template_data_bytes: 4096,
        max_field_bytes: 2048,
        evidence_refs: vec!["review:ima".into()],
    };
    (policy, record, final_pcr)
}

fn fixture() -> Fixture {
    let platform = platform();
    let challenge = challenge();
    let public = b"ak-public";
    let ak = ak(public);
    let tool_digest = digest(b"tpm2-checkquote");
    let quote_policy = quote_policy(tool_digest.clone());
    let (ima_policy, measurement_list, final_pcr) = ima_fixture();
    let ima = verify_canonical_ima_sha256(&measurement_list, final_pcr, &ima_policy).unwrap();

    let bundle = RawTpm2QuoteBundle {
        quote_id: "quote:ledger-anchor:1".into(),
        platform_qualification_digest: platform.qualification_digest(),
        ak_public: public.to_vec(),
        quote_message: quote_message(final_pcr),
        signature: b"fake-signature".to_vec(),
        pcr_values: vec![Sha256PcrValue {
            pcr: 10,
            value: final_pcr,
        }],
        collected_at_ms: 10_200,
        evidence_refs: vec!["artifact:quote".into()],
    };
    let quote = verify_tpm2_quote(
        &quote_policy,
        &challenge,
        &ak,
        &bundle,
        10_300,
        &FakeExecutor {
            digest: tool_digest.clone(),
        },
    )
    .unwrap();

    let pcr_binding_policy = TpmImaPcrBindingPolicy {
        schema_version: TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1.into(),
        binding_id: "binding:ledger-anchor:1".into(),
        required_pcr: 10,
        expected_quote_policy_digest: quote_policy.canonical_digest().unwrap(),
        expected_ima_policy_digest: ima_policy.canonical_digest().unwrap(),
        evidence_refs: vec!["review:pcr-binding".into()],
    };
    let pcr_binding = bind_verified_tpm_ima_pcr(
        &pcr_binding_policy,
        quote.verified(),
        ima.verified(),
    )
    .unwrap();

    let possession_policy = AttestationPossessionPolicy {
        schema_version: "1".into(),
        policy_id: "policy:possession:ledger-anchor".into(),
        expected_platform_qualification_digest: platform.qualification_digest(),
        expected_ak_binding_digest: ak.binding_digest(),
        expected_pcr_selection_digest: pcr_selection_digest(&challenge.pcr_selection).unwrap(),
        expected_verifier_ref: challenge.verifier_ref.clone(),
        expected_verification_tool_digest: tool_digest,
        max_challenge_lifetime_ms: 2_000,
        max_quote_to_verification_ms: 500,
        require_fixed_tpm: true,
        require_restricted_signing: true,
        evidence_refs: vec!["review:possession".into()],
    };

    let anchor_policy = LinuxTpmImaLedgerAnchorPolicy {
        schema_version: LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1.into(),
        anchor_id: "anchor:linux-tpm-ima:ledger-relative:1".into(),
        expected_pcr_binding_policy_digest: pcr_binding_policy.canonical_digest().unwrap(),
        expected_possession_policy_digest: canonical_possession_policy_digest(&possession_policy)
            .unwrap(),
        max_ledger_records: 128,
        evidence_refs: vec!["review:ledger-anchor".into()],
    };

    Fixture {
        platform,
        challenge,
        ak,
        quote,
        pcr_binding,
        possession_policy,
        anchor_policy,
    }
}

fn accept_fixture(
    fixture: &Fixture,
    ledger: &mut AttestationAcceptanceLedger,
    accepted_at_ms: u64,
) -> Result<LinuxTpmImaLedgerAnchorQualification, LinuxTpmImaLedgerAnchorReport> {
    accept_linux_tpm_ima_anchor(
        &fixture.anchor_policy,
        &fixture.possession_policy,
        &fixture.platform,
        &fixture.challenge,
        &fixture.ak,
        &fixture.quote,
        fixture.pcr_binding.verified(),
        10_400,
        accepted_at_ms,
        ledger,
    )
}

#[test]
fn valid_evidence_mints_ledger_relative_anchor() {
    let fixture = fixture();
    let mut ledger = AttestationAcceptanceLedger::default();

    let qualified = accept_fixture(&fixture, &mut ledger, 10_500).unwrap();

    assert_eq!(
        qualified.report.disposition,
        LinuxTpmImaLedgerAnchorDisposition::Qualified
    );
    assert_eq!(ledger.records.len(), 1);
    assert_eq!(qualified.verified().acceptance_revision(), 1);
    assert_eq!(qualified.verified().ledger_records_before(), 0);
    assert_eq!(qualified.verified().ledger_head_before(), None);
    assert_eq!(
        qualified.verified().acceptance_digest(),
        qualified.acceptance_record.acceptance_digest()
    );
    assert!(!qualified.verified().establishes_persistent_ledger_currentness());
    assert!(!qualified.verified().grants_physical_authority());
}

#[test]
fn challenge_rebinding_blocks_before_ledger_mutation() {
    let mut fixture = fixture();
    fixture.challenge.challenge_id = "challenge:substituted".into();
    let mut ledger = AttestationAcceptanceLedger::default();

    let report = accept_fixture(&fixture, &mut ledger, 10_500).unwrap_err();

    assert!(
        report
            .issues
            .contains(&LinuxTpmImaLedgerAnchorIssue::ChallengeBindingMismatch)
    );
    assert!(ledger.records.is_empty());
}

#[test]
fn same_id_possession_policy_drift_blocks_before_ledger_mutation() {
    let mut fixture = fixture();
    fixture.possession_policy.max_quote_to_verification_ms += 1;
    let mut ledger = AttestationAcceptanceLedger::default();

    let report = accept_fixture(&fixture, &mut ledger, 10_500).unwrap_err();

    assert!(
        report
            .issues
            .contains(&LinuxTpmImaLedgerAnchorIssue::PossessionPolicyMismatch)
    );
    assert!(ledger.records.is_empty());
}

#[test]
fn malformed_existing_ledger_is_invalid_before_mutation() {
    let fixture = fixture();
    let malformed = AttestationAcceptanceRecord {
        revision: 2,
        challenge_id: "challenge:old".into(),
        nonce_digest: digest(b"old-nonce"),
        possession_record_digest: digest(b"old-possession"),
        quote_artifact_digest: digest(b"old-quote"),
        verification_receipt_id: "receipt:old".into(),
        accepted_at_ms: 9_000,
        predecessor_acceptance_digest: None,
    };
    let mut ledger = AttestationAcceptanceLedger {
        records: vec![malformed],
    };

    let report = accept_fixture(&fixture, &mut ledger, 10_500).unwrap_err();

    assert_eq!(report.disposition, LinuxTpmImaLedgerAnchorDisposition::Invalid);
    assert!(report.issues.iter().any(|issue| matches!(
        issue,
        LinuxTpmImaLedgerAnchorIssue::InvalidLedgerState { .. }
    )));
    assert_eq!(ledger.records.len(), 1);
    assert_eq!(ledger.records[0].revision, 2);
}

#[test]
fn same_ledger_replay_is_blocked_without_second_append() {
    let fixture = fixture();
    let mut ledger = AttestationAcceptanceLedger::default();

    accept_fixture(&fixture, &mut ledger, 10_500).unwrap();
    let report = accept_fixture(&fixture, &mut ledger, 10_600).unwrap_err();

    assert_eq!(report.disposition, LinuxTpmImaLedgerAnchorDisposition::Blocked);
    assert!(report.issues.iter().any(|issue| matches!(
        issue,
        LinuxTpmImaLedgerAnchorIssue::AttestationAcceptanceRejected { .. }
    )));
    assert_eq!(ledger.records.len(), 1);
}

#[test]
fn fresh_ledger_can_accept_same_evidence_and_proves_claim_ceiling() {
    let fixture = fixture();
    let mut first = AttestationAcceptanceLedger::default();
    let mut second = AttestationAcceptanceLedger::default();

    let left = accept_fixture(&fixture, &mut first, 10_500).unwrap();
    let right = accept_fixture(&fixture, &mut second, 10_500).unwrap();

    assert_eq!(first.records.len(), 1);
    assert_eq!(second.records.len(), 1);
    assert_eq!(
        left.verified().acceptance_digest(),
        right.verified().acceptance_digest()
    );
    assert_eq!(
        left.verified().anti_replay_scope(),
        LEDGER_RELATIVE_ANTI_REPLAY_SCOPE_V1
    );
    assert!(!left.verified().establishes_persistent_ledger_currentness());
}
