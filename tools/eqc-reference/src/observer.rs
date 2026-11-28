use anyhow::{bail, Context};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

const ENVELOPE_SCHEMA: &str = "luminous.eqc.signed-observer-envelope.v0.1";
const POLICY_SCHEMA: &str = "luminous.eqc.observer-trust-policy.v0.1";
const EXPECTATION_SCHEMA: &str = "luminous.eqc.observer-run-expectation.v0.1";
const ALGORITHM: &str = "Ed25519";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObserverInputDigest {
    pub input_id: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObserverPayload {
    pub schema: String,
    pub run_id: String,
    /// 32 bytes of observer-generated entropy, encoded as 64 lowercase hex characters.
    pub nonce: String,
    pub task_id: String,
    pub task_digest_sha256: String,
    pub policy_digest_sha256: String,
    pub sut_system_id: String,
    pub sut_version: String,
    pub harness_version: String,
    pub source_sha256: String,
    pub image_digest: String,
    pub environment_sha256: String,
    pub actor_id: String,
    pub subject_id: String,
    pub authorization_decision: String,
    pub task_success: bool,
    pub input_digests: Vec<ObserverInputDigest>,
    pub observer_id: String,
    pub observer_version: String,
    pub workflow_identity: String,
    pub measurement_profile: String,
    pub runner_os: String,
    pub runner_arch: String,
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub state_mutation_count: u64,
    pub allowed_mutations_sha256: String,
    pub action_log_sha256: String,
    pub artifact_sha256: String,
    pub network_policy: String,
    pub network_enforced: bool,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub network_endpoints: Vec<String>,
    pub elapsed_ms: u64,
    pub cpu_time_ms: u64,
    pub peak_memory_bytes: u64,
    pub tool_calls: u64,
    pub human_interventions: u64,
    /// None means that no independently calibrated energy meter was available.
    pub energy_millijoules: Option<u64>,
    /// None means that no independently verified cost measurement was available.
    pub cost_microusd: Option<u64>,
    pub verifier_id: String,
    pub verifier_sha256: String,
    pub verifier_environment_sha256: String,
    pub replay_status: String,
    pub replay_result_sha256: String,
    pub observed_at_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedObserverEnvelope {
    pub schema: String,
    pub algorithm: String,
    pub key_id: String,
    pub payload: ObserverPayload,
    pub signature_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedObserverKey {
    pub key_id: String,
    pub observer_id: String,
    pub public_key_hex: String,
    pub workflow_identity: String,
    pub measurement_profile: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObserverTrustPolicy {
    pub schema: String,
    pub policy_id: String,
    pub trusted_observers: Vec<TrustedObserverKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObserverRunExpectation {
    pub schema: String,
    pub task_id: String,
    pub task_digest_sha256: String,
    pub policy_digest_sha256: String,
    pub expected_actor_id: String,
    pub expected_subject_id: String,
    pub expected_authorization_decision: String,
    pub expected_task_success: bool,
    pub state_must_remain_unchanged: bool,
    pub expected_inputs: Vec<ObserverInputDigest>,
    pub max_elapsed_ms: u64,
    pub max_cpu_time_ms: u64,
    pub max_peak_memory_bytes: u64,
    pub max_tool_calls: u64,
    pub max_human_interventions: u64,
    pub network_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObserverVerificationReport {
    pub schema: &'static str,
    pub run_id: String,
    pub task_id: String,
    pub key_id: String,
    pub signature_verified: bool,
    pub trust_identity_verified: bool,
    pub run_binding_verified: bool,
    pub nonce_fresh_within_ledger: bool,
    pub measurement_contract_pass: bool,
    pub replay_receipt_consistent: bool,
    /// Always false: the measurement runner itself has not yet been deployed
    /// and independently audited.
    pub independent_measurement_channel_verified: bool,
    /// Always false until the external observer runtime is connected and audited.
    pub qualified: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ObserverNonceLedger {
    seen: HashSet<String>,
}

impl ObserverNonceLedger {
    pub fn contains(&self, nonce: &str) -> bool {
        self.seen.contains(nonce)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    pub fn from_nonces(nonces: Vec<String>) -> anyhow::Result<Self> {
        let mut seen = HashSet::new();
        for nonce in nonces {
            if !is_sha256(&nonce) {
                bail!("nonce ledger contains malformed nonce");
            }
            if !seen.insert(nonce) {
                bail!("nonce ledger contains a duplicate nonce");
            }
        }
        Ok(Self { seen })
    }

    pub fn sorted_nonces(&self) -> Vec<String> {
        let mut nonces = self.seen.iter().cloned().collect::<Vec<_>>();
        nonces.sort();
        nonces
    }
}

/// Canonicalize the payload using RFC 8785 JSON Canonicalization Scheme.
pub fn canonical_payload_bytes(payload: &ObserverPayload) -> anyhow::Result<Vec<u8>> {
    serde_json_canonicalizer::to_vec(payload)
        .context("canonicalize observer payload using RFC 8785 JCS")
}

/// Sign the exact canonical payload bytes. The signing key is never serialized
/// into the envelope; the verifier gets accepted public keys out-of-band.
pub fn sign_payload(
    payload: ObserverPayload,
    key_id: impl Into<String>,
    signing_key: &SigningKey,
) -> anyhow::Result<SignedObserverEnvelope> {
    validate_payload_shape(&payload)?;
    let canonical = canonical_payload_bytes(&payload)?;
    let signature = signing_key.sign(&canonical);
    Ok(SignedObserverEnvelope {
        schema: ENVELOPE_SCHEMA.to_owned(),
        algorithm: ALGORITHM.to_owned(),
        key_id: key_id.into(),
        payload,
        signature_hex: encode_hex(&signature.to_bytes()),
    })
}

/// Verify a signed envelope against trust and task policy supplied separately
/// from the envelope. The nonce ledger is local to the caller's evaluation
/// scope; it is not a durable, distributed replay database.
pub fn verify_envelope(
    envelope: &SignedObserverEnvelope,
    trust_policy: &ObserverTrustPolicy,
    expectation: &ObserverRunExpectation,
    nonce_ledger: &mut ObserverNonceLedger,
) -> anyhow::Result<ObserverVerificationReport> {
    if envelope.schema != ENVELOPE_SCHEMA {
        bail!("unsupported observer envelope schema");
    }
    if envelope.algorithm != ALGORITHM {
        bail!("unsupported observer signature algorithm");
    }
    if trust_policy.schema != POLICY_SCHEMA || trust_policy.policy_id.trim().is_empty() {
        bail!("invalid out-of-band observer trust policy");
    }
    if expectation.schema != EXPECTATION_SCHEMA {
        bail!("unsupported observer run expectation schema");
    }
    validate_payload_shape(&envelope.payload)?;
    validate_expectation_shape(expectation)?;

    let trusted = trust_policy
        .trusted_observers
        .iter()
        .find(|entry| entry.key_id == envelope.key_id)
        .context("observer key_id is not present in out-of-band trust policy")?;
    if !trusted.active {
        bail!("observer key is inactive");
    }
    if trusted.observer_id != envelope.payload.observer_id
        || trusted.workflow_identity != envelope.payload.workflow_identity
        || trusted.measurement_profile != envelope.payload.measurement_profile
    {
        bail!("observer identity or measurement profile does not match trust policy");
    }

    let public_key_bytes =
        decode_hex::<32>(&trusted.public_key_hex).context("decode trusted Ed25519 public key")?;
    let verifying_key =
        VerifyingKey::from_bytes(&public_key_bytes).context("parse trusted Ed25519 public key")?;
    let signature_bytes =
        decode_hex::<64>(&envelope.signature_hex).context("decode Ed25519 signature")?;
    let signature = Signature::from_bytes(&signature_bytes);
    let canonical = canonical_payload_bytes(&envelope.payload)?;
    verifying_key
        .verify_strict(&canonical, &signature)
        .map_err(|_| anyhow::anyhow!("observer signature verification failed"))?;

    let payload = &envelope.payload;
    let task_binding = payload.task_id == expectation.task_id
        && payload.task_digest_sha256 == expectation.task_digest_sha256
        && payload.policy_digest_sha256 == expectation.policy_digest_sha256
        && payload.actor_id == expectation.expected_actor_id
        && payload.subject_id == expectation.expected_subject_id;
    let inputs_binding =
        digest_map(&payload.input_digests)? == digest_map(&expectation.expected_inputs)?;
    let decision_binding = payload.authorization_decision == expectation.expected_authorization_decision
        && payload.task_success == expectation.expected_task_success;
    if !task_binding || !inputs_binding || !decision_binding {
        bail!("signed observer envelope does not match the frozen task/run expectation");
    }

    let replay_receipt_consistent = payload.replay_status == "passed"
        && payload.verifier_id != payload.sut_system_id
        && payload.verifier_environment_sha256 != payload.environment_sha256
        && payload.replay_result_sha256 == payload.artifact_sha256;

    let unchanged = payload.state_before_sha256 == payload.state_after_sha256
        && payload.state_mutation_count == 0;
    let state_contract = !expectation.state_must_remain_unchanged || unchanged;
    let denied_contract = expectation.expected_authorization_decision != "deny" || unchanged;
    let network_contract = expectation.network_policy == "none"
        && payload.network_policy == "none"
        && payload.network_enforced
        && payload.network_bytes_sent == 0
        && payload.network_bytes_received == 0
        && payload.network_endpoints.is_empty();
    let budget_contract = payload.elapsed_ms <= expectation.max_elapsed_ms
        && payload.cpu_time_ms <= expectation.max_cpu_time_ms
        && payload.peak_memory_bytes <= expectation.max_peak_memory_bytes
        && payload.tool_calls <= expectation.max_tool_calls
        && payload.human_interventions <= expectation.max_human_interventions;
    let measurement_contract_pass =
        state_contract && denied_contract && network_contract && budget_contract;

    if nonce_ledger.contains(&payload.nonce) {
        bail!("observer nonce replay detected");
    }
    // Insert only after signature, trust, task-binding and measurement checks
    // complete. Rejected or forged envelopes do not consume a nonce.
    nonce_ledger.seen.insert(payload.nonce.clone());

    let mut reasons = Vec::new();
    if !replay_receipt_consistent {
        reasons.push("replay identity, environment, status, or artifact digest is inconsistent".to_owned());
    }
    if !measurement_contract_pass {
        reasons.push("observer-measured state, network, or resource limits violate the run contract".to_owned());
    }
    reasons.push(
        "signature and trust identity are verified, but the external measurement runtime is not yet independently deployed and audited".to_owned(),
    );

    Ok(ObserverVerificationReport {
        schema: "luminous.eqc.observer-verification-report.v0.1",
        run_id: payload.run_id.clone(),
        task_id: payload.task_id.clone(),
        key_id: envelope.key_id.clone(),
        signature_verified: true,
        trust_identity_verified: true,
        run_binding_verified: true,
        nonce_fresh_within_ledger: true,
        measurement_contract_pass,
        replay_receipt_consistent,
        independent_measurement_channel_verified: false,
        qualified: false,
        reasons,
    })
}

/// Validate envelope, trust-policy and run-expectation JSON using the
/// versioned offline Draft 2020-12 schemas shipped with this repository.
pub fn validate_observer_contract_schemas(
    envelope: &serde_json::Value,
    trust_policy: &serde_json::Value,
    expectation: &serde_json::Value,
    repository_root: &Path,
) -> anyhow::Result<()> {
    let contracts = [
        ("eqc-observer-envelope-v0.1.schema.json", envelope),
        ("eqc-observer-trust-policy-v0.1.schema.json", trust_policy),
        ("eqc-observer-run-expectation-v0.1.schema.json", expectation),
    ];

    for (schema_name, instance) in contracts {
        let schema_path = repository_root.join("docs/research").join(schema_name);
        let schema_bytes = std::fs::read(&schema_path)
            .with_context(|| format!("read observer schema {}", schema_path.display()))?;
        let schema: serde_json::Value = serde_json::from_slice(&schema_bytes)
            .with_context(|| format!("parse observer schema {}", schema_path.display()))?;
        jsonschema::meta::validate(&schema).map_err(|error| {
            anyhow::anyhow!(
                "observer schema meta-validation failed for {}: {}",
                schema_path.display(),
                error
            )
        })?;
        let validator = jsonschema::options()
            .offline()
            .should_validate_formats(true)
            .build(&schema)
            .with_context(|| {
                format!("compile offline observer schema {}", schema_path.display())
            })?;
        let errors = validator
            .iter_errors(instance)
            .map(|error| format!("{} at {}", error, error.instance_path()))
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            bail!(
                "observer schema validation failed for {}: {}",
                schema_name,
                errors.join("; ")
            );
        }
    }
    Ok(())
}

fn validate_payload_shape(payload: &ObserverPayload) -> anyhow::Result<()> {
    if payload.schema != "luminous.eqc.observer-payload.v0.1" {
        bail!("unsupported observer payload schema");
    }
    for (name, value) in [
        ("run_id", payload.run_id.as_str()),
        ("task_id", payload.task_id.as_str()),
        ("observer_id", payload.observer_id.as_str()),
        ("observer_version", payload.observer_version.as_str()),
        ("workflow_identity", payload.workflow_identity.as_str()),
        ("measurement_profile", payload.measurement_profile.as_str()),
        ("sut_system_id", payload.sut_system_id.as_str()),
        ("sut_version", payload.sut_version.as_str()),
        ("harness_version", payload.harness_version.as_str()),
        ("verifier_id", payload.verifier_id.as_str()),
        ("observed_at_utc", payload.observed_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            bail!("observer payload {name} must be non-empty");
        }
    }
    if payload.measurement_profile != "linux-cgroup-v2-netns-deny-v1" {
        bail!("unsupported observer measurement profile");
    }
    if payload.runner_os != "linux" {
        bail!("v0.1 observer profile requires a Linux runner");
    }
    if payload.replay_status != "passed"
        && payload.replay_status != "failed"
        && payload.replay_status != "not_run"
    {
        bail!("unsupported replay status");
    }
    if !is_canonical_utc_timestamp(&payload.observed_at_utc) {
        bail!("observer timestamp must be canonical UTC RFC3339 seconds");
    }
    for (name, digest) in [
        ("task_digest_sha256", &payload.task_digest_sha256),
        ("policy_digest_sha256", &payload.policy_digest_sha256),
        ("source_sha256", &payload.source_sha256),
        ("environment_sha256", &payload.environment_sha256),
        ("state_before_sha256", &payload.state_before_sha256),
        ("state_after_sha256", &payload.state_after_sha256),
        ("allowed_mutations_sha256", &payload.allowed_mutations_sha256),
        ("action_log_sha256", &payload.action_log_sha256),
        ("artifact_sha256", &payload.artifact_sha256),
        ("verifier_sha256", &payload.verifier_sha256),
        ("verifier_environment_sha256", &payload.verifier_environment_sha256),
        ("replay_result_sha256", &payload.replay_result_sha256),
    ] {
        if !is_sha256(digest) {
            bail!("observer payload {name} is not lowercase SHA-256 hex");
        }
    }
    if !payload.image_digest.starts_with("sha256:")
        || !is_sha256(&payload.image_digest["sha256:".len()..])
    {
        bail!("observer image_digest must be sha256:<64 lowercase hex>");
    }
    if !is_sha256(&payload.nonce) {
        bail!("observer nonce must be 32 random bytes encoded as lowercase hex");
    }
    const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
    for value in [
        payload.state_mutation_count,
        payload.network_bytes_sent,
        payload.network_bytes_received,
        payload.elapsed_ms,
        payload.cpu_time_ms,
        payload.peak_memory_bytes,
        payload.tool_calls,
        payload.human_interventions,
    ] {
        if value > MAX_SAFE_INTEGER {
            bail!("observer numeric measurement exceeds the RFC 8785 safe-integer range");
        }
    }
    if payload.energy_millijoules.is_some_and(|value| value > MAX_SAFE_INTEGER)
        || payload.cost_microusd.is_some_and(|value| value > MAX_SAFE_INTEGER)
    {
        bail!("observer energy or cost exceeds the RFC 8785 safe-integer range");
    }
    if payload.network_policy != "none" || !payload.network_enforced {
        bail!("v0.1 observer verifier only supports enforced network_policy=none");
    }
    if payload.authorization_decision != "allow" && payload.authorization_decision != "deny" {
        bail!("observer authorization decision must be allow or deny");
    }
    let mut ids = HashSet::new();
    for input in &payload.input_digests {
        if input.input_id.trim().is_empty() || !is_sha256(&input.sha256) {
            bail!("observer input digest has empty id or malformed SHA-256");
        }
        if !ids.insert(&input.input_id) {
            bail!("duplicate observer input id: {}", input.input_id);
        }
    }
    Ok(())
}

fn validate_expectation_shape(expectation: &ObserverRunExpectation) -> anyhow::Result<()> {
    if expectation.task_id.trim().is_empty()
        || !is_sha256(&expectation.task_digest_sha256)
        || !is_sha256(&expectation.policy_digest_sha256)
        || expectation.expected_actor_id.trim().is_empty()
        || expectation.expected_subject_id.trim().is_empty()
        || expectation.expected_inputs.is_empty()
        || (expectation.expected_authorization_decision != "allow"
            && expectation.expected_authorization_decision != "deny")
    {
        bail!("invalid frozen observer run expectation");
    }
    if expectation.network_policy != "none" {
        bail!("v0.1 observer verifier only supports expected network_policy=none");
    }
    digest_map(&expectation.expected_inputs)?;
    Ok(())
}

fn digest_map(inputs: &[ObserverInputDigest]) -> anyhow::Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    for input in inputs {
        if input.input_id.trim().is_empty() || !is_sha256(&input.sha256) {
            bail!("invalid input digest entry");
        }
        if map.insert(input.input_id.clone(), input.sha256.clone()).is_some() {
            bail!("duplicate input id: {}", input.input_id);
        }
    }
    Ok(map)
}

fn is_canonical_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return false;
    }
    bytes.iter().enumerate().all(|(index, byte)| {
        matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit()
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_hex<const N: usize>(value: &str) -> anyhow::Result<[u8; N]> {
    if value.len() != N * 2 || !value.is_ascii() {
        bail!("hex value has incorrect length or non-ASCII characters");
    }
    let mut output = [0u8; N];
    let bytes = value.as_bytes();
    for index in 0..N {
        let high = hex_nibble(bytes[index * 2])?;
        let low = hex_nibble(bytes[index * 2 + 1])?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> anyhow::Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => bail!("hex must be lowercase ASCII"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn payload() -> ObserverPayload {
        ObserverPayload {
            schema: "luminous.eqc.observer-payload.v0.1".into(),
            run_id: "run-observer-test-001".into(),
            nonce: "11".repeat(32),
            task_id: "eqc-health-wrong-subject-deny-001".into(),
            task_digest_sha256: "22".repeat(32),
            policy_digest_sha256: "33".repeat(32),
            sut_system_id: "sut-adapter".into(),
            sut_version: "0.1.0".into(),
            harness_version: "0.1.0".into(),
            source_sha256: "44".repeat(32),
            image_digest: "sha256:".to_owned() + &"55".repeat(32),
            environment_sha256: "66".repeat(32),
            actor_id: "synthetic-coordinator-01".into(),
            subject_id: "synthetic-patient-002".into(),
            authorization_decision: "deny".into(),
            task_success: true,
            input_digests: vec![ObserverInputDigest {
                input_id: "consent-fixture".into(),
                sha256: "77".repeat(32),
            }],
            observer_id: "trusted-observer-linux".into(),
            observer_version: "0.1.0".into(),
            workflow_identity: "repo:Luminous-Dynamics/symthaea:workflow:eqc-observer.yml:ref:refs/heads/test".into(),
            measurement_profile: "linux-cgroup-v2-netns-deny-v1".into(),
            runner_os: "linux".into(),
            runner_arch: "x86_64".into(),
            state_before_sha256: "88".repeat(32),
            state_after_sha256: "88".repeat(32),
            state_mutation_count: 0,
            allowed_mutations_sha256: "99".repeat(32),
            action_log_sha256: "aa".repeat(32),
            artifact_sha256: "bb".repeat(32),
            network_policy: "none".into(),
            network_enforced: true,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            network_endpoints: vec![],
            elapsed_ms: 200,
            cpu_time_ms: 150,
            peak_memory_bytes: 1024 * 1024,
            tool_calls: 2,
            human_interventions: 0,
            energy_millijoules: None,
            cost_microusd: None,
            verifier_id: "fresh-independent-verifier".into(),
            verifier_sha256: "cc".repeat(32),
            verifier_environment_sha256: "dd".repeat(32),
            replay_status: "passed".into(),
            replay_result_sha256: "bb".repeat(32),
            observed_at_utc: "2026-11-28T10:00:00Z".into(),
        }
    }

    fn trust_policy(key: &SigningKey) -> ObserverTrustPolicy {
        ObserverTrustPolicy {
            schema: POLICY_SCHEMA.into(),
            policy_id: "eqc-observer-trust-test".into(),
            trusted_observers: vec![TrustedObserverKey {
                key_id: "test-observer-key-01".into(),
                observer_id: "trusted-observer-linux".into(),
                public_key_hex: encode_hex(key.verifying_key().as_bytes()),
                workflow_identity: "repo:Luminous-Dynamics/symthaea:workflow:eqc-observer.yml:ref:refs/heads/test".into(),
                measurement_profile: "linux-cgroup-v2-netns-deny-v1".into(),
                active: true,
            }],
        }
    }

    fn expectation() -> ObserverRunExpectation {
        ObserverRunExpectation {
            schema: EXPECTATION_SCHEMA.into(),
            task_id: "eqc-health-wrong-subject-deny-001".into(),
            task_digest_sha256: "22".repeat(32),
            policy_digest_sha256: "33".repeat(32),
            expected_actor_id: "synthetic-coordinator-01".into(),
            expected_subject_id: "synthetic-patient-002".into(),
            expected_authorization_decision: "deny".into(),
            expected_task_success: true,
            state_must_remain_unchanged: true,
            expected_inputs: vec![ObserverInputDigest {
                input_id: "consent-fixture".into(),
                sha256: "77".repeat(32),
            }],
            max_elapsed_ms: 10_000,
            max_cpu_time_ms: 5_000,
            max_peak_memory_bytes: 128 * 1024 * 1024,
            max_tool_calls: 3,
            max_human_interventions: 0,
            network_policy: "none".into(),
        }
    }

    fn signed_envelope(key: &SigningKey) -> SignedObserverEnvelope {
        sign_payload(payload(), "test-observer-key-01", key).unwrap()
    }

    #[test]
    fn valid_signature_and_trust_policy_verify_without_claiming_qualification() {
        let key = signing_key();
        let envelope = signed_envelope(&key);
        let policy = trust_policy(&key);
        let mut ledger = ObserverNonceLedger::default();

        let report = verify_envelope(&envelope, &policy, &expectation(), &mut ledger).unwrap();
        assert!(report.signature_verified);
        assert!(report.trust_identity_verified);
        assert!(report.run_binding_verified);
        assert!(report.nonce_fresh_within_ledger);
        assert!(report.measurement_contract_pass);
        assert!(report.replay_receipt_consistent);
        assert!(!report.independent_measurement_channel_verified);
        assert!(!report.qualified);
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn signature_tampering_is_rejected() {
        let key = signing_key();
        let mut envelope = signed_envelope(&key);
        envelope.payload.network_bytes_sent = 1;
        let mut ledger = ObserverNonceLedger::default();
        assert!(verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger).is_err());
        assert!(ledger.is_empty());
    }

    #[test]
    fn untrusted_or_substituted_key_id_is_rejected() {
        let key = signing_key();
        let mut envelope = signed_envelope(&key);
        envelope.key_id = "attacker-controlled-key".into();
        let mut ledger = ObserverNonceLedger::default();
        assert!(verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger).is_err());
    }

    #[test]
    fn observer_workflow_identity_is_bound_to_out_of_band_policy() {
        let key = signing_key();
        let mut changed = payload();
        changed.workflow_identity.push_str(":substituted");
        let envelope = sign_payload(changed, "test-observer-key-01", &key).unwrap();
        let mut ledger = ObserverNonceLedger::default();
        assert!(verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger).is_err());
    }

    #[test]
    fn nonce_replay_is_rejected_within_the_evaluation_ledger() {
        let key = signing_key();
        let envelope = signed_envelope(&key);
        let policy = trust_policy(&key);
        let expected = expectation();
        let mut ledger = ObserverNonceLedger::default();

        verify_envelope(&envelope, &policy, &expected, &mut ledger).unwrap();
        assert!(verify_envelope(&envelope, &policy, &expected, &mut ledger).is_err());
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn task_and_input_digest_substitution_is_rejected() {
        let key = signing_key();
        let envelope = signed_envelope(&key);
        let policy = trust_policy(&key);
        let mut expected = expectation();
        expected.task_digest_sha256 = "ff".repeat(32);
        let mut ledger = ObserverNonceLedger::default();
        assert!(verify_envelope(&envelope, &policy, &expected, &mut ledger).is_err());
        assert!(ledger.is_empty());
    }

    #[test]
    fn denied_run_with_state_change_fails_measurement_contract() {
        let key = signing_key();
        let mut changed = payload();
        changed.state_after_sha256 = "ee".repeat(32);
        changed.state_mutation_count = 1;
        let envelope = sign_payload(changed, "test-observer-key-01", &key).unwrap();
        let mut ledger = ObserverNonceLedger::default();
        let report = verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger).unwrap();
        assert!(!report.measurement_contract_pass);
        assert!(!report.qualified);
    }

    #[test]
    fn network_activity_breaks_the_closed_network_measurement_contract() {
        let key = signing_key();
        let mut changed = payload();
        changed.network_bytes_received = 1;
        changed.network_endpoints.push("example.invalid".into());
        let envelope = sign_payload(changed, "test-observer-key-01", &key).unwrap();
        let mut ledger = ObserverNonceLedger::default();
        let report = verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger).unwrap();
        assert!(!report.measurement_contract_pass);
    }

    #[test]
    fn unknown_energy_and_cost_are_preserved_as_null_not_zero() {
        let value = serde_json::to_value(payload()).unwrap();
        assert!(value["energy_millijoules"].is_null());
        assert!(value["cost_microusd"].is_null());
    }

    #[test]
    fn human_intervention_budget_is_enforced() {
        let key = signing_key();
        let mut changed = payload();
        changed.human_interventions = 1;
        let envelope = sign_payload(changed, "test-observer-key-01", &key).unwrap();
        let mut ledger = ObserverNonceLedger::default();
        let report = verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger)
            .unwrap();
        assert!(!report.measurement_contract_pass);
        assert!(!report.qualified);
    }

    #[test]
    fn values_outside_the_jcs_safe_integer_range_are_rejected() {
        let mut changed = payload();
        changed.elapsed_ms = 9_007_199_254_740_992;
        assert!(validate_payload_shape(&changed).is_err());
    }

    #[test]
    fn failed_or_reused_verifier_does_not_pass_replay_receipt_consistency() {
        let key = signing_key();
        let mut changed = payload();
        changed.verifier_id = changed.sut_system_id.clone();
        let envelope = sign_payload(changed, "test-observer-key-01", &key).unwrap();
        let mut ledger = ObserverNonceLedger::default();
        let report = verify_envelope(&envelope, &trust_policy(&key), &expectation(), &mut ledger).unwrap();
        assert!(!report.replay_receipt_consistent);
        assert!(!report.qualified);
    }

    #[test]
    fn envelope_trust_and_expectation_schemas_validate_and_reject_mutations() {
        let key = signing_key();
        let envelope = signed_envelope(&key);
        let trust = trust_policy(&key);
        let expected = expectation();
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root resolves");
        let envelope_value = serde_json::to_value(&envelope).unwrap();
        let trust_value = serde_json::to_value(&trust).unwrap();
        let expected_value = serde_json::to_value(&expected).unwrap();

        validate_observer_contract_schemas(
            &envelope_value,
            &trust_value,
            &expected_value,
            &root,
        )
        .expect("all three published observer schemas validate");

        let mut tampered = envelope_value.clone();
        tampered["payload"]["unreviewed_authority_override"] = json!(true);
        assert!(
            validate_observer_contract_schemas(
                &tampered,
                &trust_value,
                &expected_value,
                &root
            )
            .is_err()
        );

        let mut trust_tampered = trust_value.clone();
        trust_tampered["trusted_observers"][0]["unexpected_key"] = json!("attacker");
        assert!(
            validate_observer_contract_schemas(
                &envelope_value,
                &trust_tampered,
                &expected_value,
                &root
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_json_signing_is_stable_under_object_key_order() {
        let left = json!({"z": 1, "a": {"y": 2, "b": 3}});
        let right = json!({"a": {"b": 3, "y": 2}, "z": 1});
        let left = serde_json_canonicalizer::to_vec(&left).unwrap();
        let right = serde_json_canonicalizer::to_vec(&right).unwrap();
        assert_eq!(left, right);
    }
}
