use symthaea_assurance_tpm_ima_pcr_binding::*;
use sha2::{Digest, Sha256};
use symthaea_assurance_tpm2_attestation_possession::{
    AttestationChallenge, AttestationKeyBinding, PcrBankSelection,
};
use symthaea_assurance_tpm2_checkquote_adapter::{
    verify_tpm2_quote, CheckquoteExecutionRequest, CheckquoteExecutor, RawTpm2QuoteBundle,
    Sha256PcrValue, ToolExecution, Tpm2CheckquotePolicy, CHECKQUOTE_POLICY_SCHEMA_V1,
};
use symthaea_linux_ima_replay::{
    verify_canonical_ima_sha256, ImaReplayPolicy, RequiredImaMeasurement,
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

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn challenge() -> AttestationChallenge {
    AttestationChallenge {
        schema_version: "1".into(),
        challenge_id: "challenge:pcr-binding:1".into(),
        nonce_hex: hex_lower(&NONCE),
        issued_at_ms: 10_000,
        expires_at_ms: 11_000,
        verifier_ref: "verifier:remote-1".into(),
        pcr_selection: vec![PcrBankSelection {
            hash_alg: "sha256".into(),
            pcrs: vec![10],
        }],
        evidence_refs: vec!["request:pcr-binding:1".into()],
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
        adapter_id: "adapter:tpm2-checkquote:pcr-binding".into(),
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

fn put_tpm2b(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u16).to_be_bytes());
    target.extend_from_slice(value);
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
        policy_id: "policy:ima:pcr-binding".into(),
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

fn verified_quote(
    pcr_value: [u8; 32],
) -> (
    Tpm2CheckquotePolicy,
    symthaea_assurance_tpm2_checkquote_adapter::Tpm2QuoteQualification,
) {
    let challenge = challenge();
    let public = b"ak-public";
    let ak = ak(public);
    let tool_digest = digest(b"tpm2-checkquote");
    let policy = quote_policy(tool_digest.clone());
    let bundle = RawTpm2QuoteBundle {
        quote_id: "quote:pcr-binding:1".into(),
        platform_qualification_digest: digest(b"platform"),
        ak_public: public.to_vec(),
        quote_message: quote_message(pcr_value),
        signature: b"fake-signature".to_vec(),
        pcr_values: vec![Sha256PcrValue {
            pcr: 10,
            value: pcr_value,
        }],
        collected_at_ms: 10_200,
        evidence_refs: vec!["artifact:quote".into()],
    };
    let quote = verify_tpm2_quote(
        &policy,
        &challenge,
        &ak,
        &bundle,
        10_300,
        &FakeExecutor {
            digest: tool_digest,
        },
    )
    .unwrap();
    (policy, quote)
}

fn binding_policy(
    quote_policy: &Tpm2CheckquotePolicy,
    ima_policy: &ImaReplayPolicy,
) -> TpmImaPcrBindingPolicy {
    TpmImaPcrBindingPolicy {
        schema_version: TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1.into(),
        binding_id: "binding:tpm-ima-pcr:1".into(),
        required_pcr: 10,
        expected_quote_policy_digest: quote_policy.canonical_digest().unwrap(),
        expected_ima_policy_digest: ima_policy.canonical_digest().unwrap(),
        evidence_refs: vec!["review:pcr-binding".into()],
    }
}

#[test]
fn equal_verified_pcrs_mint_opaque_binding() {
    let (ima_policy, bytes, final_pcr) = ima_fixture();
    let ima = verify_canonical_ima_sha256(&bytes, final_pcr, &ima_policy).unwrap();
    let (quote_policy, quote) = verified_quote(final_pcr);
    let policy = binding_policy(&quote_policy, &ima_policy);

    let qualified = bind_verified_tpm_ima_pcr(&policy, quote.verified(), ima.verified()).unwrap();

    assert_eq!(
        qualified.report.disposition,
        TpmImaPcrBindingDisposition::Qualified
    );
    assert_eq!(qualified.verified().pcr(), 10);
    assert_eq!(qualified.verified().pcr_value(), final_pcr);
    assert_eq!(
        qualified.verified().matched_required_measurements(),
        &["verifier-executable".to_string()]
    );
    assert!(!qualified.grants_physical_authority());
    assert!(!qualified.verified().grants_physical_authority());
}

#[test]
fn pcr_value_mismatch_is_blocked() {
    let (ima_policy, bytes, final_pcr) = ima_fixture();
    let ima = verify_canonical_ima_sha256(&bytes, final_pcr, &ima_policy).unwrap();
    let (quote_policy, quote) = verified_quote([0x42; 32]);
    let policy = binding_policy(&quote_policy, &ima_policy);

    let report = bind_verified_tpm_ima_pcr(&policy, quote.verified(), ima.verified()).unwrap_err();

    assert_eq!(report.disposition, TpmImaPcrBindingDisposition::Blocked);
    assert!(
        report
            .issues
            .contains(&TpmImaPcrBindingIssue::PcrValueMismatch)
    );
}

#[test]
fn exact_quote_policy_digest_is_required() {
    let (ima_policy, bytes, final_pcr) = ima_fixture();
    let ima = verify_canonical_ima_sha256(&bytes, final_pcr, &ima_policy).unwrap();
    let (quote_policy, quote) = verified_quote(final_pcr);
    let mut policy = binding_policy(&quote_policy, &ima_policy);
    policy.expected_quote_policy_digest = digest(b"other-quote-policy");

    let report = bind_verified_tpm_ima_pcr(&policy, quote.verified(), ima.verified()).unwrap_err();

    assert!(
        report
            .issues
            .contains(&TpmImaPcrBindingIssue::QuotePolicyMismatch)
    );
}

#[test]
fn exact_ima_policy_digest_is_required() {
    let (ima_policy, bytes, final_pcr) = ima_fixture();
    let ima = verify_canonical_ima_sha256(&bytes, final_pcr, &ima_policy).unwrap();
    let (quote_policy, quote) = verified_quote(final_pcr);
    let mut policy = binding_policy(&quote_policy, &ima_policy);
    policy.expected_ima_policy_digest = digest(b"other-ima-policy");

    let report = bind_verified_tpm_ima_pcr(&policy, quote.verified(), ima.verified()).unwrap_err();

    assert!(
        report
            .issues
            .contains(&TpmImaPcrBindingIssue::ImaPolicyMismatch)
    );
}

#[test]
fn policy_reference_order_is_nonsemantic() {
    let (ima_policy, _, _) = ima_fixture();
    let quote_policy = quote_policy(digest(b"tool"));
    let mut left = binding_policy(&quote_policy, &ima_policy);
    left.evidence_refs = vec!["review:a".into(), "review:b".into()];
    let mut right = left.clone();
    right.evidence_refs.reverse();
    assert_eq!(left.canonical_digest(), right.canonical_digest());
}
