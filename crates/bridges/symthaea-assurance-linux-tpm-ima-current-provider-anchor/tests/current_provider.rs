use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};
use symthaea_assurance_attestation_ledger_current_head::{
    verify_acceptance_ledger_head_statement, verify_current_acceptance_ledger_head,
    AcceptanceLedgerHeadCurrentnessAttestation, AcceptanceLedgerHeadStatement,
    AcceptanceLedgerHeadTracker, LedgerHeadAuthorityKey, LedgerHeadAuthorityPolicy,
    LedgerHeadClaimScope, LEDGER_HEAD_AUTHORITY_POLICY_SCHEMA_V1,
    LEDGER_HEAD_CURRENTNESS_SCHEMA_V1, LEDGER_HEAD_STATEMENT_SCHEMA_V1,
};
use symthaea_assurance_linux_tpm_ima_current_provider_anchor::*;
use symthaea_assurance_linux_tpm_ima_ledger_anchor::{
    accept_linux_tpm_ima_anchor, LinuxTpmImaLedgerAnchorPolicy,
    LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1,
};
use symthaea_assurance_tpm_ima_pcr_binding::{
    bind_verified_tpm_ima_pcr, TpmImaPcrBindingPolicy, TpmImaPcrBindingQualification,
    TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1,
};
use symthaea_assurance_tpm2_attestation_possession::{
    pcr_selection_digest, AttestationAcceptanceLedger, AttestationChallenge,
    AttestationKeyBinding, AttestationPossessionPolicy, PcrBankSelection,
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
    verify_canonical_ima_sha256, ImaReplayPolicy, RequiredImaMeasurement,
    SupportedImaTemplate, IMA_REPLAY_POLICY_SCHEMA_V1,
};

const NONCE: [u8; 32] = [0xab; 32];
const QUALIFIED_SIGNER: [u8; 34] = [0x5a; 34];
const HEAD_SEED: [u8; 32] = [3; 32];
const CURRENT_SEED: [u8; 32] = [4; 32];
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

struct ProviderFixture {
    ledger_anchor: symthaea_assurance_linux_tpm_ima_ledger_anchor::LinuxTpmImaLedgerAnchorQualification,
    pcr_binding: TpmImaPcrBindingQualification,
    head_policy: LedgerHeadAuthorityPolicy,
    provider_policy: CurrentLinuxTpmImaProviderPolicy,
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn nonce_digest(label: &str) -> String {
    blake3::hash(label.as_bytes()).to_hex().to_string()
}

fn public_hex(seed: [u8; 32]) -> String {
    hex::encode(SigningKey::from_bytes(&seed).verifying_key().to_bytes())
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
        challenge_id: "challenge:current-provider:1".into(),
        nonce_hex: hex_lower(&NONCE),
        issued_at_ms: 10_000,
        expires_at_ms: 11_000,
        verifier_ref: "verifier:remote-1".into(),
        pcr_selection: vec![PcrBankSelection {
            hash_alg: "sha256".into(),
            pcrs: vec![10],
        }],
        evidence_refs: vec!["request:current-provider:1".into()],
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
        adapter_id: "adapter:tpm2-checkquote:current-provider".into(),
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
        policy_id: "policy:ima:current-provider".into(),
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

fn head_policy() -> LedgerHeadAuthorityPolicy {
    LedgerHeadAuthorityPolicy {
        schema_version: LEDGER_HEAD_AUTHORITY_POLICY_SCHEMA_V1.into(),
        policy_id: "policy:ledger-head:provider".into(),
        sequence: 1,
        issued_at_ms: 9_000,
        expires_at_ms: 100_000,
        trusted_keys: vec![
            LedgerHeadAuthorityKey {
                key_id: "key:head".into(),
                public_key_ed25519_hex: public_hex(HEAD_SEED),
                valid_from_ms: 9_000,
                valid_until_ms: None,
                revoked_at_ms: None,
                allowed_scopes: vec![LedgerHeadClaimScope::HeadStatement],
                evidence_refs: vec!["review:key:head".into()],
            },
            LedgerHeadAuthorityKey {
                key_id: "key:current".into(),
                public_key_ed25519_hex: public_hex(CURRENT_SEED),
                valid_from_ms: 9_000,
                valid_until_ms: None,
                revoked_at_ms: None,
                allowed_scopes: vec![LedgerHeadClaimScope::CurrentnessAttestation],
                evidence_refs: vec!["review:key:current".into()],
            },
        ],
        evidence_refs: vec!["review:head-policy".into()],
    }
}

fn provider_fixture() -> ProviderFixture {
    let platform = platform();
    let challenge = challenge();
    let public = b"ak-public";
    let ak = ak(public);
    let tool_digest = digest(b"tpm2-checkquote");
    let quote_policy = quote_policy(tool_digest.clone());
    let (ima_policy, measurement_list, final_pcr) = ima_fixture();
    let ima = verify_canonical_ima_sha256(&measurement_list, final_pcr, &ima_policy).unwrap();

    let quote_bundle = RawTpm2QuoteBundle {
        quote_id: "quote:current-provider:1".into(),
        platform_qualification_digest: platform.qualification_digest(),
        ak_public: public.to_vec(),
        quote_message: quote_message(final_pcr),
        signature: b"fake-signature".to_vec(),
        pcr_values: vec![Sha256PcrValue { pcr: 10, value: final_pcr }],
        collected_at_ms: 10_200,
        evidence_refs: vec!["artifact:quote".into()],
    };
    let quote: Tpm2QuoteQualification = verify_tpm2_quote(
        &quote_policy,
        &challenge,
        &ak,
        &quote_bundle,
        10_300,
        &FakeExecutor { digest: tool_digest.clone() },
    )
    .unwrap();

    let pcr_policy = TpmImaPcrBindingPolicy {
        schema_version: TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1.into(),
        binding_id: "binding:current-provider:1".into(),
        required_pcr: 10,
        expected_quote_policy_digest: quote_policy.canonical_digest().unwrap(),
        expected_ima_policy_digest: ima_policy.canonical_digest().unwrap(),
        evidence_refs: vec!["review:pcr-binding".into()],
    };
    let pcr_binding = bind_verified_tpm_ima_pcr(&pcr_policy, quote.verified(), ima.verified()).unwrap();

    let possession = AttestationPossessionPolicy {
        schema_version: "1".into(),
        policy_id: "policy:possession:current-provider".into(),
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
    let ledger_anchor_policy = LinuxTpmImaLedgerAnchorPolicy {
        schema_version: LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1.into(),
        anchor_id: "anchor:ledger-relative:provider".into(),
        expected_pcr_binding_policy_digest: pcr_policy.canonical_digest().unwrap(),
        expected_possession_policy_digest: canonical_possession_policy_digest(&possession).unwrap(),
        max_ledger_records: 128,
        evidence_refs: vec!["review:ledger-anchor".into()],
    };
    let mut ledger = AttestationAcceptanceLedger::default();
    let ledger_anchor = accept_linux_tpm_ima_anchor(
        &ledger_anchor_policy,
        &possession,
        &platform,
        &challenge,
        &ak,
        &quote,
        pcr_binding.verified(),
        10_400,
        10_500,
        &mut ledger,
    )
    .unwrap();

    let head_policy = head_policy();
    let provider_policy = CurrentLinuxTpmImaProviderPolicy {
        schema_version: CURRENT_PROVIDER_POLICY_SCHEMA_V1.into(),
        provider_id: "provider:linux-tpm-ima:1".into(),
        expected_ledger_id: "ledger:attestation:prod-1".into(),
        expected_ledger_anchor_policy_digest: ledger_anchor_policy.canonical_digest().unwrap(),
        expected_pcr_binding_policy_digest: pcr_policy.canonical_digest().unwrap(),
        expected_ledger_head_authority_policy_digest: head_policy.canonical_digest().unwrap(),
        max_quote_age_at_use_ms: 1_000,
        max_acceptance_age_at_use_ms: 1_000,
        evidence_refs: vec!["review:current-provider".into()],
    };

    ProviderFixture { ledger_anchor, pcr_binding, head_policy, provider_policy }
}

fn current_head(
    fixture: &ProviderFixture,
    acceptance_digest: String,
    use_at_ms: u64,
) -> symthaea_assurance_attestation_ledger_current_head::CurrentAcceptanceLedgerHead {
    let mut statement = AcceptanceLedgerHeadStatement {
        schema_version: LEDGER_HEAD_STATEMENT_SCHEMA_V1.into(),
        ledger_id: "ledger:attestation:prod-1".into(),
        acceptance_revision: fixture.ledger_anchor.verified().acceptance_revision(),
        acceptance_digest,
        previous_head_statement_digest: None,
        issued_at_ms: 10_520,
        authority_policy_digest: fixture.head_policy.canonical_digest().unwrap(),
        signer_key_id: "key:head".into(),
        signer_public_key_ed25519_hex: public_hex(HEAD_SEED),
        evidence_refs: vec!["ledger:head:1".into()],
        signature_ed25519_hex: "00".repeat(64),
    };
    let bytes = statement.canonical_unsigned_bytes().unwrap();
    statement.signature_ed25519_hex =
        hex::encode(SigningKey::from_bytes(&HEAD_SEED).sign(&bytes).to_bytes());
    let verified = verify_acceptance_ledger_head_statement(&fixture.head_policy, &statement)
        .into_verified()
        .unwrap();
    let mut tracker = AcceptanceLedgerHeadTracker::default();
    let tracked = tracker.accept(&verified).unwrap();
    let nonce = nonce_digest("provider-currentness");

    let mut currentness = AcceptanceLedgerHeadCurrentnessAttestation {
        schema_version: LEDGER_HEAD_CURRENTNESS_SCHEMA_V1.into(),
        head_statement_digest: tracked.statement_digest().into(),
        ledger_id: tracked.ledger_id().into(),
        acceptance_revision: tracked.acceptance_revision(),
        acceptance_digest: tracked.acceptance_digest().into(),
        authority_policy_digest: fixture.head_policy.canonical_digest().unwrap(),
        asserted_current_at_ms: use_at_ms,
        challenge_nonce_blake3_hex: nonce.clone(),
        signer_key_id: "key:current".into(),
        signer_public_key_ed25519_hex: public_hex(CURRENT_SEED),
        evidence_refs: vec!["response:provider-currentness".into()],
        signature_ed25519_hex: "00".repeat(64),
    };
    let bytes = currentness.canonical_unsigned_bytes().unwrap();
    currentness.signature_ed25519_hex =
        hex::encode(SigningKey::from_bytes(&CURRENT_SEED).sign(&bytes).to_bytes());

    verify_current_acceptance_ledger_head(
        &tracker,
        &tracked,
        &fixture.head_policy,
        &currentness,
        &nonce,
        use_at_ms,
    )
    .into_current()
    .unwrap()
}

#[test]
fn exact_current_head_and_fresh_measurement_mint_provider_anchor() {
    let fixture = provider_fixture();
    let head = current_head(
        &fixture,
        fixture.ledger_anchor.verified().acceptance_digest().into(),
        10_600,
    );

    let qualified = promote_current_linux_tpm_ima_provider(
        &fixture.provider_policy,
        fixture.ledger_anchor.verified(),
        fixture.pcr_binding.verified(),
        &head,
    )
    .unwrap();

    assert_eq!(qualified.report.disposition, CurrentProviderDisposition::Current);
    assert_eq!(qualified.verified().acceptance_revision(), 1);
    assert_eq!(qualified.verified().use_at_ms(), 10_600);
    assert!(!qualified.verified().establishes_continuous_runtime_integrity());
    assert!(!qualified.verified().establishes_trusted_time());
    assert!(!qualified.verified().grants_physical_authority());
}

#[test]
fn authoritative_but_different_acceptance_history_is_blocked() {
    let fixture = provider_fixture();
    let head = current_head(&fixture, digest(b"different-acceptance-history"), 10_600);

    let report = promote_current_linux_tpm_ima_provider(
        &fixture.provider_policy,
        fixture.ledger_anchor.verified(),
        fixture.pcr_binding.verified(),
        &head,
    )
    .unwrap_err();

    assert_eq!(report.disposition, CurrentProviderDisposition::Blocked);
    assert!(report.issues.contains(&CurrentProviderIssue::AcceptanceDigestMismatch));
}

#[test]
fn stale_quote_is_blocked_even_when_ledger_head_is_current() {
    let fixture = provider_fixture();
    let head = current_head(
        &fixture,
        fixture.ledger_anchor.verified().acceptance_digest().into(),
        20_000,
    );

    let report = promote_current_linux_tpm_ima_provider(
        &fixture.provider_policy,
        fixture.ledger_anchor.verified(),
        fixture.pcr_binding.verified(),
        &head,
    )
    .unwrap_err();

    assert_eq!(report.disposition, CurrentProviderDisposition::Blocked);
    assert!(report.issues.iter().any(|issue| matches!(
        issue,
        CurrentProviderIssue::QuoteTooOld { .. }
    )));
    assert!(report.issues.iter().any(|issue| matches!(
        issue,
        CurrentProviderIssue::AcceptanceTooOld { .. }
    )));
}

#[test]
fn unreviewed_ledger_id_is_blocked() {
    let fixture = provider_fixture();
    let head = current_head(
        &fixture,
        fixture.ledger_anchor.verified().acceptance_digest().into(),
        10_600,
    );
    let mut policy = fixture.provider_policy.clone();
    policy.expected_ledger_id = "ledger:attestation:other".into();

    let report = promote_current_linux_tpm_ima_provider(
        &policy,
        fixture.ledger_anchor.verified(),
        fixture.pcr_binding.verified(),
        &head,
    )
    .unwrap_err();

    assert!(report.issues.contains(&CurrentProviderIssue::LedgerIdMismatch));
}
