use sha2::{Digest, Sha256};
use symthaea_assurance_linux_tpm_ima_pcr_anchor::{
    qualify_linux_tpm_ima_pcr_anchor, LinuxTpmImaPcrAnchorDisposition,
    LinuxTpmImaPcrAnchorIssue, LinuxTpmImaPcrAnchorPolicy,
    LINUX_TPM_IMA_PCR_ANCHOR_POLICY_SCHEMA_V1,
};
use symthaea_assurance_tpm2_attestation_possession::{
    pcr_selection_digest, AttestationChallenge, AttestationKeyBinding,
    AttestationPossessionPolicy, PcrBankSelection,
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
    verify_canonical_ima_sha256, ImaReplayPolicy, ImaReplayQualification, RequiredImaMeasurement,
    SupportedImaTemplate, IMA_REPLAY_POLICY_SCHEMA_V1,
};

const NONCE: [u8; 32] = [0xab; 32];
const QUALIFIED_SIGNER: [u8; 34] = [0x5a; 34];
const TPM2_GENERATED_VALUE: u32 = 0xff54_4347;
const TPM2_ST_ATTEST_QUOTE: u16 = 0x8018;
const TPM2_ALG_SHA256: u16 = 0x000b;

#[derive(Clone)]
struct FakeExecutor {
    digest: String,
    expected_pcr: [u8; 32],
}

impl CheckquoteExecutor for FakeExecutor {
    fn executable_blake3(&self, _executable: &str) -> Result<String, String> {
        Ok(self.digest.clone())
    }

    fn execute_checkquote(
        &self,
        _executable: &str,
        request: &CheckquoteExecutionRequest,
    ) -> Result<ToolExecution, String> {
        assert_eq!(request.pcr_list, "sha256:10");
        assert_eq!(request.qualification_hex, hex_lower(&NONCE));
        assert_eq!(request.pcr_values_raw, self.expected_pcr);
        Ok(ToolExecution {
            exit_code: Some(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
}

struct Fixture {
    anchor_policy: LinuxTpmImaPcrAnchorPolicy,
    checkquote_policy: Tpm2CheckquotePolicy,
    possession_policy: AttestationPossessionPolicy,
    platform: Tpm2PlatformQualificationRecord,
    challenge: AttestationChallenge,
    ak: AttestationKeyBinding,
    quote: Tpm2QuoteQualification,
    ima_policy: ImaReplayPolicy,
    ima: ImaReplayQualification,
}

fn fixture(quote_matches_ima: bool) -> Fixture {
    let (ima_bytes, ima_final, required) = ima_fixture();
    let ima_policy = ImaReplayPolicy {
        schema_version: IMA_REPLAY_POLICY_SCHEMA_V1.into(),
        policy_id: "policy:ima-replay:fixture".into(),
        expected_pcr: 10,
        expected_initial_pcr: [0; 32],
        allowed_templates: vec![SupportedImaTemplate::ImaNg],
        required_measurements: vec![required],
        max_measurement_list_bytes: 1024 * 1024,
        max_records: 100,
        max_template_name_bytes: 128,
        max_template_data_bytes: 4096,
        max_field_bytes: 1024,
        evidence_refs: vec!["fixture:ima-policy".into()],
    };
    let ima = verify_canonical_ima_sha256(&ima_bytes, ima_final, &ima_policy).unwrap();

    let platform = platform();
    let challenge = challenge();
    let public = b"-----BEGIN PUBLIC KEY-----fixture-----END PUBLIC KEY-----";
    let ak = ak(public);
    let tool_digest = digest(b"tpm2-checkquote-fixture");
    let checkquote_policy = checkquote_policy(tool_digest.clone());
    let quote_pcr = if quote_matches_ima { ima_final } else { [0x77; 32] };
    let quote_bundle = quote_bundle(public, platform.qualification_digest(), quote_pcr);
    let quote = verify_tpm2_quote(
        &checkquote_policy,
        &challenge,
        &ak,
        &quote_bundle,
        10_300,
        &FakeExecutor {
            digest: tool_digest.clone(),
            expected_pcr: quote_pcr,
        },
    )
    .unwrap();

    let possession_policy = AttestationPossessionPolicy {
        schema_version: "1".into(),
        policy_id: "policy:possession:fixture".into(),
        expected_platform_qualification_digest: platform.qualification_digest(),
        expected_ak_binding_digest: ak.binding_digest(),
        expected_pcr_selection_digest: pcr_selection_digest(&challenge.pcr_selection).unwrap(),
        expected_verifier_ref: challenge.verifier_ref.clone(),
        expected_verification_tool_digest: tool_digest,
        max_challenge_lifetime_ms: 2_000,
        max_quote_to_verification_ms: 500,
        require_fixed_tpm: true,
        require_restricted_signing: true,
        evidence_refs: vec!["fixture:possession-policy".into()],
    };

    let anchor_policy = LinuxTpmImaPcrAnchorPolicy {
        schema_version: LINUX_TPM_IMA_PCR_ANCHOR_POLICY_SCHEMA_V1.into(),
        policy_id: "policy:linux-tpm-ima-anchor:fixture".into(),
        expected_checkquote_policy_digest: checkquote_policy.canonical_digest().unwrap(),
        expected_possession_policy_digest: canonical_possession_policy_digest(&possession_policy)
            .unwrap(),
        expected_ima_replay_policy_digest: ima_policy.canonical_digest().unwrap(),
        expected_ima_pcr: 10,
        evidence_refs: vec!["fixture:anchor-policy".into()],
    };

    Fixture {
        anchor_policy,
        checkquote_policy,
        possession_policy,
        platform,
        challenge,
        ak,
        quote,
        ima_policy,
        ima,
    }
}

#[test]
fn exact_quote_possession_and_ima_replay_mint_opaque_pcr_anchor() {
    let fixture = fixture(true);
    let result = qualify_linux_tpm_ima_pcr_anchor(
        &fixture.anchor_policy,
        &fixture.checkquote_policy,
        &fixture.possession_policy,
        &fixture.platform,
        &fixture.challenge,
        &fixture.ak,
        &fixture.quote,
        &fixture.ima_policy,
        &fixture.ima,
        10_400,
    )
    .unwrap();

    assert_eq!(
        result.report.disposition,
        LinuxTpmImaPcrAnchorDisposition::Qualified
    );
    assert_eq!(result.verified().ima_pcr(), 10);
    assert_eq!(result.verified().pcr_value(), fixture.ima.verified().final_pcr());
    assert_eq!(
        result.verified().ima_measurement_list_digest(),
        fixture.ima.verified().measurement_list_digest()
    );
    assert!(!result.grants_physical_authority());
    assert!(!result.verified().grants_physical_authority());
}

#[test]
fn cryptographically_valid_quote_with_different_pcr_cannot_anchor_ima_replay() {
    let fixture = fixture(false);
    let report = qualify_linux_tpm_ima_pcr_anchor(
        &fixture.anchor_policy,
        &fixture.checkquote_policy,
        &fixture.possession_policy,
        &fixture.platform,
        &fixture.challenge,
        &fixture.ak,
        &fixture.quote,
        &fixture.ima_policy,
        &fixture.ima,
        10_400,
    )
    .unwrap_err();

    assert_eq!(report.disposition, LinuxTpmImaPcrAnchorDisposition::Invalid);
    assert!(report
        .issues
        .contains(&LinuxTpmImaPcrAnchorIssue::QuoteImaPcrMismatch));
}

#[test]
fn mutating_public_quote_receipt_after_raw_verification_is_rejected() {
    let mut fixture = fixture(true);
    fixture.quote.verification_receipt.receipt_id.push_str(":tampered");

    let report = qualify_linux_tpm_ima_pcr_anchor(
        &fixture.anchor_policy,
        &fixture.checkquote_policy,
        &fixture.possession_policy,
        &fixture.platform,
        &fixture.challenge,
        &fixture.ak,
        &fixture.quote,
        &fixture.ima_policy,
        &fixture.ima,
        10_400,
    )
    .unwrap_err();

    assert!(report.issues.iter().any(|issue| matches!(
        issue,
        LinuxTpmImaPcrAnchorIssue::QuoteCapabilityBindingMismatch(field)
            if field == "verification-receipt-digest" || field == "public-verification-receipt"
    )));
}

fn platform() -> Tpm2PlatformQualificationRecord {
    Tpm2PlatformQualificationRecord {
        schema_version: "1".into(),
        qualification_id: "qualification:platform:fixture".into(),
        adapter_id: "adapter:tpm2-platform:fixture".into(),
        trust_store_ref: "trust-store:tpm:fixture".into(),
        tcti: "device:/dev/tpmrm0".into(),
        nv_index: 0x0180_0001,
        nv_name: "nv:fixture".into(),
        before_observation_digest: digest(b"before-counter"),
        after_observation_digest: digest(b"after-counter"),
        before_counter_value: 7,
        after_counter_value: 8,
        fixed_properties: Tpm2FixedProperties {
            family_indicator: 0x322e_3000,
            specification_revision: 159,
            manufacturer: 0x4946_5800,
            firmware_version_1: 1,
            firmware_version_2: 2,
            raw_output_digest: digest(b"fixed-properties"),
        },
        getcap_blake3: digest(b"tpm2-getcap"),
        runtime_subject_digest: digest(b"runtime-subject"),
        qualified_at_ms: 9_000,
        evidence_refs: vec!["fixture:platform".into()],
    }
}

fn challenge() -> AttestationChallenge {
    AttestationChallenge {
        schema_version: "1".into(),
        challenge_id: "challenge:fixture".into(),
        nonce_hex: hex_lower(&NONCE),
        issued_at_ms: 10_000,
        expires_at_ms: 11_000,
        verifier_ref: "verifier:remote-fixture".into(),
        pcr_selection: vec![PcrBankSelection {
            hash_alg: "sha256".into(),
            pcrs: vec![10],
        }],
        evidence_refs: vec!["fixture:challenge".into()],
    }
}

fn ak(public: &[u8]) -> AttestationKeyBinding {
    AttestationKeyBinding {
        schema_version: "1".into(),
        ak_id: "ak:fixture".into(),
        ak_name_hex: "000b01020304".into(),
        ak_qualified_name_hex: hex_lower(&QUALIFIED_SIGNER),
        public_key_digest: digest(public),
        name_alg: "sha256".into(),
        signing_scheme: "ecdsa-sha256".into(),
        fixed_tpm: true,
        restricted_signing: true,
        reviewed_at_ms: 9_500,
        evidence_refs: vec!["fixture:ak".into()],
    }
}

fn checkquote_policy(tool_digest: String) -> Tpm2CheckquotePolicy {
    Tpm2CheckquotePolicy {
        schema_version: CHECKQUOTE_POLICY_SCHEMA_V1.into(),
        adapter_id: "adapter:tpm2-checkquote:fixture".into(),
        verifier_ref: "verifier:remote-fixture".into(),
        checkquote_path: "/nix/store/fixture-tpm2-tools/bin/tpm2_checkquote".into(),
        expected_checkquote_blake3: tool_digest,
        hash_algorithm: "sha256".into(),
        max_public_bytes: 1024 * 1024,
        max_message_bytes: 1024 * 1024,
        max_signature_bytes: 1024 * 1024,
        max_tool_output_bytes: 1024 * 1024,
        evidence_refs: vec!["fixture:checkquote-policy".into()],
    }
}

fn quote_bundle(
    public: &[u8],
    platform_qualification_digest: String,
    pcr_value: [u8; 32],
) -> RawTpm2QuoteBundle {
    RawTpm2QuoteBundle {
        quote_id: "quote:fixture".into(),
        platform_qualification_digest,
        ak_public: public.to_vec(),
        quote_message: quote_message(pcr_value),
        signature: b"fixture-tss-signature".to_vec(),
        pcr_values: vec![Sha256PcrValue { pcr: 10, value: pcr_value }],
        collected_at_ms: 10_200,
        evidence_refs: vec!["fixture:raw-quote".into()],
    }
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
    bitmap[10 / 8] |= 1u8 << (10 % 8);
    bytes.extend_from_slice(&bitmap);
    put_tpm2b(&mut bytes, &pcr_digest);
    bytes
}

fn put_tpm2b(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u16).to_be_bytes());
    target.extend_from_slice(value);
}

fn ima_fixture() -> (Vec<u8>, [u8; 32], RequiredImaMeasurement) {
    let event_digest = [0x11; 32];
    let event_name = b"/nix/store/fixture-verifier";
    let event_name_digest: [u8; 32] = Sha256::digest(event_name).into();

    let mut digest_field = b"sha256:\0".to_vec();
    digest_field.extend_from_slice(&event_digest);
    let mut name_field = event_name.to_vec();
    name_field.push(0);

    let mut template_data = Vec::new();
    push_field(&mut template_data, &digest_field);
    push_field(&mut template_data, &name_field);
    let template_digest: [u8; 32] = Sha256::digest(&template_data).into();

    let mut record = Vec::new();
    record.extend_from_slice(&10u32.to_le_bytes());
    record.extend_from_slice(&template_digest);
    record.extend_from_slice(&("ima-ng".len() as u32).to_le_bytes());
    record.extend_from_slice(b"ima-ng");
    record.extend_from_slice(&(template_data.len() as u32).to_le_bytes());
    record.extend_from_slice(&template_data);

    let mut extend_input = [0u8; 64];
    extend_input[32..].copy_from_slice(&template_digest);
    let final_pcr: [u8; 32] = Sha256::digest(extend_input).into();

    let required = RequiredImaMeasurement {
        measurement_id: "fixture:verifier-executable".into(),
        event_sha256: event_digest,
        event_name_sha256: Some(event_name_digest),
    };
    (record, final_pcr, required)
}

fn push_field(target: &mut Vec<u8>, field: &[u8]) {
    target.extend_from_slice(&(field.len() as u32).to_le_bytes());
    target.extend_from_slice(field);
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
