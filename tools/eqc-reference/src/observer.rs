use anyhow::{bail, Context};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const DOMAIN_SEPARATION: &[u8] = b"LUMINOUS-EQC-OBSERVER-ATTESTATION-V0.1\0";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VerifiedObserverEvidence {
    pub observer_id: String,
    pub key_id: String,
    pub observer_binary_sha256: String,
    pub observer_environment_sha256: String,
    pub trust_store_sha256: String,
    pub payload_sha256: String,
    pub challenge_sha256: String,
    pub run_id: String,
    pub task_id: String,
    pub replay_verifier_id: String,
    pub replay_verifier_sha256: String,
    pub challenge_ledger_marker_sha256: String,
}

pub struct ObserverVerificationContext<'a> {
    pub task: &'a Value,
    pub observation: &'a Value,
    pub run_id: &'a str,
    pub challenge_nonce: &'a str,
    pub verification_time: &'a str,
    pub task_corpus_sha256: &'a str,
    pub expected_trust_store_sha256: &'a str,
    pub observation_bytes: &'a [u8],
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> anyhow::Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {pointer}"))
}

fn required_u64(value: &Value, pointer: &str) -> anyhow::Result<u64> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing unsigned integer field {pointer}"))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn decode_hex(value: &str) -> anyhow::Result<Vec<u8>> {
    if value.len() % 2 != 0
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("hex value must be lowercase, even-length hexadecimal");
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let text = std::str::from_utf8(pair)?;
        bytes.push(u8::from_str_radix(text, 16)?);
    }
    Ok(bytes)
}

fn decode_fixed<const N: usize>(value: &str, field: &str) -> anyhow::Result<[u8; N]> {
    let bytes = decode_hex(value).with_context(|| format!("decode {field}"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("{field} must contain exactly {} bytes", N))
}

fn validate_schema(
    value: &Value,
    schema_name: &str,
    repository_root: &Path,
) -> anyhow::Result<()> {
    let schema_path = repository_root.join("docs/research").join(schema_name);
    let schema_bytes = fs::read(&schema_path)
        .with_context(|| format!("read schema {}", schema_path.display()))?;
    let schema: Value = serde_json::from_slice(&schema_bytes)
        .with_context(|| format!("parse schema {}", schema_path.display()))?;
    let validator = jsonschema::options()
        .offline()
        .should_validate_formats(true)
        .build(&schema)
        .with_context(|| format!("compile offline schema {}", schema_path.display()))?;
    let errors = validator
        .iter_errors(value)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        bail!(
            "schema validation failed for {schema_name}: {}",
            errors.join("; ")
        );
    }
    Ok(())
}

fn canonical_utc(value: &str) -> bool {
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
    let digits = [0..4, 5..7, 8..10, 11..13, 14..16, 17..19];
    if digits
        .iter()
        .any(|range| !bytes[range.clone()].iter().all(u8::is_ascii_digit))
    {
        return false;
    }
    let number = |range: std::ops::Range<usize>| -> Option<u32> {
        std::str::from_utf8(&bytes[range]).ok()?.parse().ok()
    };
    let Some(year) = number(0..4) else {
        return false;
    };
    let Some(month @ 1..=12) = number(5..7) else {
        return false;
    };
    let Some(day) = number(8..10) else {
        return false;
    };
    let Some(hour @ 0..=23) = number(11..13) else {
        return false;
    };
    let Some(minute @ 0..=59) = number(14..16) else {
        return false;
    };
    let Some(second @ 0..=59) = number(17..19) else {
        return false;
    };
    let _ = (hour, minute, second);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let last_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=last_day).contains(&day)
}

fn consume_challenge(
    ledger_directory: &Path,
    challenge_nonce: &str,
    run_id: &str,
    task_id: &str,
    payload_sha256: &str,
) -> anyhow::Result<String> {
    if !is_sha256(challenge_nonce) {
        bail!("expected challenge nonce must be 256-bit lowercase hex");
    }
    if ledger_directory.exists() {
        let metadata = fs::symlink_metadata(ledger_directory)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!("challenge ledger directory must be a real directory, not a symlink");
        }
    } else {
        fs::create_dir_all(ledger_directory)?;
    }
    let ledger_root = ledger_directory
        .canonicalize()
        .context("canonicalize challenge ledger directory")?;
    let marker_name = format!("{}.consumed", sha256(challenge_nonce.as_bytes()));
    let marker_path: PathBuf = ledger_root.join(marker_name);
    let marker_bytes = format!(
        "schema=luminous.eqc.challenge-consumption.v0.1\nrun_id={run_id}\ntask_id={task_id}\npayload_sha256={payload_sha256}\n"
    );

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker_path)
        .with_context(|| {
            format!(
                "challenge nonce has already been consumed or cannot be reserved: {}",
                marker_path.display()
            )
        })?;
    file.write_all(marker_bytes.as_bytes())
        .context("write challenge consumption marker")?;
    file.sync_all()
        .context("fsync challenge consumption marker")?;
    File::open(&ledger_root)
        .and_then(|directory| directory.sync_all())
        .context("fsync challenge ledger directory")?;
    Ok(sha256(&marker_bytes))
}

pub fn verify_and_consume(
    envelope: &Value,
    trust_store: &Value,
    trust_store_bytes: &[u8],
    context: &ObserverVerificationContext<'_>,
    repository_root: &Path,
    challenge_ledger_directory: &Path,
) -> anyhow::Result<VerifiedObserverEvidence> {
    validate_schema(
        envelope,
        "eqc-observer-envelope-v0.1.schema.json",
        repository_root,
    )?;
    validate_schema(
        trust_store,
        "eqc-observer-trust-store-v0.1.schema.json",
        repository_root,
    )?;
    if !is_sha256(context.expected_trust_store_sha256)
        || sha256(trust_store_bytes) != context.expected_trust_store_sha256
    {
        bail!("observer trust store does not match the externally pinned trust-store digest");
    }

    if required_str(envelope, "/schema")? != "luminous.eqc.observer-envelope.v0.1" {
        bail!("unsupported observer envelope schema");
    }
    if required_str(trust_store, "/schema")? != "luminous.eqc.observer-trust-store.v0.1" {
        bail!("unsupported observer trust-store schema");
    }

    let payload_bytes = decode_hex(required_str(envelope, "/payload_hex")?)
        .context("decode signed observer payload")?;
    let payload_sha256 = sha256(&payload_bytes);
    if payload_sha256 != required_str(envelope, "/payload_sha256")? {
        bail!("observer payload SHA-256 does not match the signed envelope");
    }
    let payload: Value = serde_json::from_slice(&payload_bytes)
        .context("parse exact signed observer payload bytes")?;
    validate_schema(
        &payload,
        "eqc-observer-payload-v0.1.schema.json",
        repository_root,
    )?;
    if required_str(&payload, "/schema")? != "luminous.eqc.observer-payload.v0.1" {
        bail!("unsupported observer payload schema");
    }

    let task_id = required_str(context.task, "/task_id")?;
    let task_domain = required_str(context.task, "/domain")?;
    let observed_task_id = required_str(context.observation, "/task_id")?;
    if observed_task_id != task_id
        || required_str(&payload, "/task_id")? != task_id
    {
        bail!("observer receipt task id does not match the frozen task");
    }
    if required_str(&payload, "/run_id")? != context.run_id {
        bail!("observer receipt run id does not match the evaluator's run id");
    }
    if required_str(&payload, "/challenge_nonce")? != context.challenge_nonce {
        bail!("observer receipt challenge does not match the evaluator's one-use challenge");
    }
    if !is_sha256(context.challenge_nonce)
        || !is_sha256(context.task_corpus_sha256)
    {
        bail!("challenge nonce and task-corpus digest must be lowercase SHA-256 values");
    }
    if required_str(&payload, "/task_corpus_sha256")? != context.task_corpus_sha256 {
        bail!("observer receipt is bound to a different frozen task corpus");
    }
    let observation_sha256 = sha256(context.observation_bytes);
    if required_str(&payload, "/system_observation_sha256")? != observation_sha256 {
        bail!("observer receipt is not bound to the exact adapter observation bytes");
    }

    for timestamp in [
        required_str(&payload, "/issued_at")?,
        required_str(&payload, "/expires_at")?,
        context.verification_time,
    ] {
        if !canonical_utc(timestamp) {
            bail!("observer timestamps must use canonical UTC RFC3339 seconds");
        }
    }
    let issued_at = required_str(&payload, "/issued_at")?;
    let expires_at = required_str(&payload, "/expires_at")?;
    if issued_at > context.verification_time || context.verification_time > expires_at {
        bail!("observer receipt is not currently valid");
    }
    if expires_at <= issued_at {
        bail!("observer receipt expiry must follow issuance");
    }

    let key_id = required_str(&payload, "/observer/key_id")?;
    let observer_id = required_str(&payload, "/observer/observer_id")?;
    let observer_binary_sha256 = required_str(&payload, "/observer/binary_sha256")?;
    let observer_environment_sha256 =
        required_str(&payload, "/observer/environment_sha256")?;
    if !is_sha256(observer_binary_sha256) || !is_sha256(observer_environment_sha256) {
        bail!("observer binary and environment identities must be SHA-256 values");
    }

    let keys = trust_store
        .get("keys")
        .and_then(Value::as_array)
        .context("observer trust store has no keys array")?;
    let matching_keys = keys
        .iter()
        .filter(|key| key.get("key_id").and_then(Value::as_str) == Some(key_id))
        .collect::<Vec<_>>();
    if matching_keys.len() != 1 {
        bail!("observer key id must resolve to exactly one trust-store entry");
    }
    let key = matching_keys[0];
    if required_str(key, "/status")? != "active"
        || required_str(key, "/observer_id")? != observer_id
    {
        bail!("observer key is inactive or bound to a different observer identity");
    }
    if required_str(key, "/observer_binary_sha256")? != observer_binary_sha256
        || required_str(key, "/observer_environment_sha256")? != observer_environment_sha256
    {
        bail!("observer binary or environment digest does not match the trusted profile");
    }
    let valid_from = required_str(key, "/valid_from")?;
    let valid_until = required_str(key, "/valid_until")?;
    if !canonical_utc(valid_from) || !canonical_utc(valid_until) {
        bail!("trusted observer key validity window is malformed");
    }
    if issued_at < valid_from || context.verification_time > valid_until {
        bail!("observer key is outside its trusted validity window");
    }
    if let Some(revoked_at) = key.get("revoked_at").and_then(Value::as_str) {
        if !canonical_utc(revoked_at) || revoked_at <= context.verification_time {
            bail!("observer signing key has been revoked");
        }
    }
    let domains = key
        .get("allowed_domains")
        .and_then(Value::as_array)
        .context("trusted observer key has no domain scope")?;
    if !domains.iter().any(|domain| domain.as_str() == Some(task_domain)) {
        bail!("observer key is not authorized for task domain {task_domain}");
    }

    let public_key = decode_fixed::<32>(
        required_str(key, "/public_key_hex")?,
        "observer public key",
    )?;
    let signature_bytes = decode_fixed::<64>(
        required_str(envelope, "/signature_ed25519_hex")?,
        "observer Ed25519 signature",
    )?;
    let verifying_key = VerifyingKey::from_bytes(&public_key)
        .context("parse trusted observer Ed25519 public key")?;
    if verifying_key.is_weak() {
        bail!("weak Ed25519 observer public keys are rejected");
    }
    let signature = Signature::try_from(signature_bytes.as_slice())
        .context("parse observer Ed25519 signature")?;
    let mut signed_message = Vec::with_capacity(DOMAIN_SEPARATION.len() + payload_bytes.len());
    signed_message.extend_from_slice(DOMAIN_SEPARATION);
    signed_message.extend_from_slice(&payload_bytes);
    verifying_key
        .verify_strict(&signed_message, &signature)
        .context("strict Ed25519 observer receipt signature verification failed")?;

    let observation = context.observation;
    let binding_pairs = [
        ("/actor_id", "/actor_id"),
        ("/subject_id", "/subject_id"),
        ("/authorization_decision", "/authorization_decision"),
        ("/authority_snapshot_sha256", "/authority_snapshot_sha256"),
        ("/action_log_sha256", "/action_log_sha256"),
        ("/result_digest_sha256", "/result_digest_sha256"),
    ];
    for (payload_pointer, observation_pointer) in binding_pairs {
        if required_str(&payload, payload_pointer)? != required_str(observation, observation_pointer)? {
            bail!("observer receipt binding differs from adapter observation at {payload_pointer}");
        }
    }
    for (payload_pointer, observation_pointer) in [
        ("/source_sha256", "/system/source_sha256"),
        ("/system_environment_sha256", "/system/environment_sha256"),
    ] {
        if required_str(&payload, payload_pointer)? != required_str(observation, observation_pointer)? {
            bail!("observer receipt system identity differs at {payload_pointer}");
        }
    }
    if payload.get("input_digests") != observation.get("input_digests") {
        bail!("observer-measured input digests differ from the submitted observation");
    }
    if payload.get("resource_usage") != observation.get("resource_usage") {
        bail!("observer-measured resource usage differs from the submitted observation");
    }
    if required_u64(&payload, "/state_mutation_count")?
        != required_u64(observation, "/state_mutation_count")?
    {
        bail!("observer-measured state mutation count differs from the submitted observation");
    }

    let before = required_str(&payload, "/state_before_sha256")?;
    let after = required_str(&payload, "/state_after_sha256")?;
    if !is_sha256(before) || !is_sha256(after) {
        bail!("observer state-before/state-after commitments must be SHA-256 values");
    }
    let mutations = required_u64(&payload, "/state_mutation_count")?;
    let effect_witness = payload.get("effect_witness_sha256").and_then(Value::as_str);
    if mutations == 0 {
        if before != after || effect_witness.is_some() {
            bail!("zero-mutation receipt must have identical state commitments and no effect witness");
        }
    } else if before == after || !effect_witness.is_some_and(is_sha256) {
        bail!("mutating receipt must bind distinct state commitments and a valid effect witness");
    }

    let replay = payload
        .get("replay")
        .context("observer payload has no replay evidence")?;
    if required_str(replay, "/status")? != "passed"
        || required_str(observation, "/verifier/replay")? != "passed"
        || required_str(replay, "/result_sha256")?
            != required_str(observation, "/result_digest_sha256")?
        || required_str(replay, "/verifier_id")?
            != required_str(observation, "/verifier/verifier_id")?
        || required_str(replay, "/verifier_sha256")?
            != required_str(observation, "/verifier/verifier_sha256")?
    {
        bail!("observer replay evidence is failed or disagrees with the submitted result");
    }

    let trust_store_sha256 = sha256(trust_store_bytes);
    let challenge_sha256 = sha256(context.challenge_nonce.as_bytes());
    let marker_sha256 = consume_challenge(
        challenge_ledger_directory,
        context.challenge_nonce,
        context.run_id,
        task_id,
        &payload_sha256,
    )?;

    Ok(VerifiedObserverEvidence {
        observer_id: observer_id.to_owned(),
        key_id: key_id.to_owned(),
        observer_binary_sha256: observer_binary_sha256.to_owned(),
        observer_environment_sha256: observer_environment_sha256.to_owned(),
        trust_store_sha256,
        payload_sha256,
        challenge_sha256,
        run_id: context.run_id.to_owned(),
        task_id: task_id.to_owned(),
        replay_verifier_id: required_str(replay, "/verifier_id")?.to_owned(),
        replay_verifier_sha256: required_str(replay, "/verifier_sha256")?.to_owned(),
        challenge_ledger_marker_sha256: marker_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    const TEST_KEY_SEED: [u8; 32] = [7; 32];
    const VALID_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root resolves")
    }

    fn ledger() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eqc-observer-test-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create isolated test ledger");
        path
    }

    fn sample() -> (Value, Value, Value, Vec<u8>, SigningKey) {
        let task = json!({
            "task_id": "eqc-observer-demo-001",
            "domain": "healthcare"
        });
        let observation = json!({
            "task_id": "eqc-observer-demo-001",
            "actor_id": "actor-1",
            "subject_id": "subject-1",
            "authorization_decision": "allow",
            "input_digests": [{"input_id":"input-1","sha256":VALID_SHA}],
            "authority_snapshot_sha256": VALID_SHA,
            "state_mutation_count": 0,
            "action_log_sha256": VALID_SHA,
            "result_digest_sha256": VALID_SHA,
            "system": {
                "source_sha256": VALID_SHA,
                "environment_sha256": VALID_SHA
            },
            "resource_usage": {
                "elapsed_ms": 10,
                "cpu_time_ms": 5,
                "peak_memory_bytes": 1024,
                "tool_calls": 1,
                "network_bytes": 0,
                "network_endpoints": [],
                "human_interventions": 0,
                "energy_millijoules": null,
                "cost_microusd": null
            },
            "verifier": {
                "verifier_id": "reference",
                "verifier_sha256": VALID_SHA
            }
        });
        let signing_key = SigningKey::from_bytes(&TEST_KEY_SEED);
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        let observation_sha256 = sha256(&observation_bytes);
        let payload = json!({
            "schema": "luminous.eqc.observer-payload.v0.1",
            "task_id": "eqc-observer-demo-001",
            "run_id": "run-001",
            "challenge_nonce": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "issued_at": "2026-11-28T10:00:00Z",
            "expires_at": "2026-11-28T10:05:00Z",
            "task_corpus_sha256": VALID_SHA,
            "system_observation_sha256": observation_sha256,
            "observer": {
                "observer_id": "observer-1",
                "key_id": "observer-key-1",
                "binary_sha256": VALID_SHA,
                "environment_sha256": VALID_SHA,
                "measurement_profile": "linux-cgroup-v2+filesystem-effect-ledger-v1"
            },
            "actor_id": "actor-1",
            "subject_id": "subject-1",
            "authorization_decision": "allow",
            "source_sha256": VALID_SHA,
            "system_environment_sha256": VALID_SHA,
            "authority_snapshot_sha256": VALID_SHA,
            "input_digests": [{"input_id":"input-1","sha256":VALID_SHA}],
            "state_before_sha256": VALID_SHA,
            "state_after_sha256": VALID_SHA,
            "state_mutation_count": 0,
            "effect_witness_sha256": null,
            "action_log_sha256": VALID_SHA,
            "result_digest_sha256": VALID_SHA,
            "resource_usage": observation["resource_usage"],
            "replay": {
                "status": "passed",
                "verifier_id": "reference",
                "verifier_sha256": VALID_SHA,
                "result_sha256": VALID_SHA
            }
        });
        let mut payload_bytes = serde_json::to_vec(&payload).unwrap();
        let mut signed = DOMAIN_SEPARATION.to_vec();
        signed.extend_from_slice(&payload_bytes);
        let signature = signing_key.sign(&signed);
        let envelope = json!({
            "schema": "luminous.eqc.observer-envelope.v0.1",
            "payload_hex": hex(&payload_bytes),
            "payload_sha256": sha256(&payload_bytes),
            "signature_ed25519_hex": hex(&signature.to_bytes())
        });
        let trust_store = json!({
            "schema": "luminous.eqc.observer-trust-store.v0.1",
            "trust_store_id": "test-operator-root-v1",
            "keys": [{
                "key_id": "observer-key-1",
                "observer_id": "observer-1",
                "public_key_hex": hex(&signing_key.verifying_key().to_bytes()),
                "observer_binary_sha256": VALID_SHA,
                "observer_environment_sha256": VALID_SHA,
                "allowed_domains": ["healthcare"],
                "status": "active",
                "valid_from": "2026-11-28T00:00:00Z",
                "valid_until": "2026-11-29T00:00:00Z",
                "revoked_at": null
            }]
        });
        payload_bytes.shrink_to_fit();
        (task, observation, envelope, serde_json::to_vec(&trust_store).unwrap(), signing_key)
    }

    #[test]
    fn strict_signature_and_exact_bindings_verify_and_consume_once() {
        let (task, observation, envelope, trust_bytes, _) = sample();
        let trust: Value = serde_json::from_slice(&trust_bytes).unwrap();
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        let root = root();
        let ledger = ledger();
        let context = ObserverVerificationContext {
            task: &task,
            observation: &observation,
            run_id: "run-001",
            challenge_nonce: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            verification_time: "2026-11-28T10:01:00Z",
            task_corpus_sha256: VALID_SHA,
            expected_trust_store_sha256: &sha256(&trust_bytes),
            observation_bytes: &observation_bytes,
        };
        let verified = verify_and_consume(
            &envelope,
            &trust,
            &trust_bytes,
            &context,
            &root,
            &ledger,
        )
        .expect("signed observer receipt verifies");
        assert_eq!(verified.observer_id, "observer-1");
        assert_eq!(verified.run_id, "run-001");
        assert!(verified.challenge_ledger_marker_sha256.len() == 64);

        assert!(verify_and_consume(
            &envelope,
            &trust,
            &trust_bytes,
            &context,
            &root,
            &ledger
        )
        .is_err());
    }

    #[test]
    fn wrong_challenge_is_rejected_before_consumption() {
        let (task, observation, envelope, trust_bytes, _) = sample();
        let trust: Value = serde_json::from_slice(&trust_bytes).unwrap();
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        let root = root();
        let ledger = ledger();
        let context = ObserverVerificationContext {
            task: &task,
            observation: &observation,
            run_id: "run-001",
            challenge_nonce: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            verification_time: "2026-11-28T10:01:00Z",
            task_corpus_sha256: VALID_SHA,
            expected_trust_store_sha256: &sha256(&trust_bytes),
            observation_bytes: &observation_bytes,
        };
        assert!(verify_and_consume(
            &envelope,
            &trust,
            &trust_bytes,
            &context,
            &root,
            &ledger
        )
        .is_err());
        assert_eq!(fs::read_dir(ledger).unwrap().count(), 0);
    }

    #[test]
    fn tampered_payload_and_untrusted_key_are_rejected() {
        let (task, observation, mut envelope, trust_bytes, signing_key) = sample();
        let trust: Value = serde_json::from_slice(&trust_bytes).unwrap();
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        let root = root();
        let ledger = ledger();
        envelope["payload_hex"] = json!("00");
        let context = ObserverVerificationContext {
            task: &task,
            observation: &observation,
            run_id: "run-001",
            challenge_nonce: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            verification_time: "2026-11-28T10:01:00Z",
            task_corpus_sha256: VALID_SHA,
            expected_trust_store_sha256: &sha256(&trust_bytes),
            observation_bytes: &observation_bytes,
        };
        assert!(verify_and_consume(
            &envelope,
            &trust,
            &trust_bytes,
            &context,
            &root,
            &ledger
        )
        .is_err());

        let mut wrong_trust = trust.clone();
        wrong_trust["keys"][0]["public_key_hex"] =
            json!(hex(&SigningKey::from_bytes(&[9; 32]).verifying_key().to_bytes()));
        let wrong_trust_bytes = serde_json::to_vec(&wrong_trust).unwrap();
        let wrong_context = ObserverVerificationContext {
            expected_trust_store_sha256: &sha256(&wrong_trust_bytes),
            ..context
        };
        let (_, _, valid_envelope, _, _) = sample();
        assert!(verify_and_consume(
            &valid_envelope,
            &wrong_trust,
            &wrong_trust_bytes,
            &wrong_context,
            &root,
            &ledger
        )
        .is_err());
        let _ = signing_key;
    }

    #[test]
    fn payload_modified_and_rehashed_without_resigning_fails_strict_signature() {
        let (task, observation, envelope, trust_bytes, _) = sample();
        let trust: Value = serde_json::from_slice(&trust_bytes).unwrap();
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        let root = root();
        let ledger = ledger();
        let mut changed_envelope = envelope.clone();
        let original_payload = decode_hex(
            changed_envelope["payload_hex"].as_str().unwrap(),
        )
        .unwrap();
        let mut modified_payload = original_payload;
        let mutation_index = modified_payload.len() / 2;
        modified_payload[mutation_index] ^= 0x01;
        changed_envelope["payload_hex"] = json!(hex(&modified_payload));
        changed_envelope["payload_sha256"] = json!(sha256(&modified_payload));
        let context = ObserverVerificationContext {
            task: &task,
            observation: &observation,
            run_id: "run-001",
            challenge_nonce: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            verification_time: "2026-11-28T10:01:00Z",
            task_corpus_sha256: VALID_SHA,
            expected_trust_store_sha256: &sha256(&trust_bytes),
            observation_bytes: &observation_bytes,
        };
        assert!(verify_and_consume(
            &changed_envelope,
            &trust,
            &trust_bytes,
            &context,
            &root,
            &ledger
        )
        .is_err());
        assert_eq!(fs::read_dir(ledger).unwrap().count(), 0);
    }

    #[test]
    fn expired_or_revoked_observer_keys_fail_closed() {
        let (task, observation, envelope, trust_bytes, _) = sample();
        let trust: Value = serde_json::from_slice(&trust_bytes).unwrap();
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        let root = root();
        let context = ObserverVerificationContext {
            task: &task,
            observation: &observation,
            run_id: "run-001",
            challenge_nonce: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            verification_time: "2026-11-30T10:01:00Z",
            task_corpus_sha256: VALID_SHA,
            expected_trust_store_sha256: &sha256(&trust_bytes),
            observation_bytes: &observation_bytes,
        };
        let ledger = ledger();
        assert!(verify_and_consume(
            &envelope,
            &trust,
            &trust_bytes,
            &context,
            &root,
            &ledger
        )
        .is_err());

        let mut revoked = trust.clone();
        revoked["keys"][0]["revoked_at"] = json!("2026-11-28T09:00:00Z");
        let revoked_bytes = serde_json::to_vec(&revoked).unwrap();
        let fresh_ledger = ledger();
        let context = ObserverVerificationContext {
            verification_time: "2026-11-28T10:01:00Z",
            ..context
        };
        assert!(verify_and_consume(
            &envelope,
            &revoked,
            &revoked_bytes,
            &context,
            &root,
            &fresh_ledger
        )
        .is_err());
    }
}
