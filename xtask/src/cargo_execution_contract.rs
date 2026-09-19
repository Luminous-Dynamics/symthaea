use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::repository_effect_policy::{
    EffectEvaluation, EffectPolicySpec, evaluate, validate_and_identify_policy,
};
use crate::repository_snapshot_diff::{RepositorySourceDiff, compare};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const INTENT_INPUT_SCHEMA: &str = "symthaea.cargo-execution-intent-input.v1";
const INTENT_SCHEMA: &str = "symthaea.cargo-execution-intent.v1";
const INTENT_HASH_DOMAIN: &[u8] = b"symthaea.cargo-execution-intent.v1\0";
const RESULT_SCHEMA: &str = "symthaea.cargo-execution-result.v1";
const RESULT_HASH_DOMAIN: &[u8] = b"symthaea.cargo-execution-result.v1\0";
const CARGO_OBSERVATION_SCHEMA: &str = "symthaea.cargo-build-observation.v1";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CargoExecutionIntentSpec {
    pub schema: String,
    #[serde(default)]
    pub git_worktree_state_before: Option<String>,
    pub build_context_id: String,
    pub invocation_id: String,
    #[serde(default)]
    pub plan_id: Option<String>,
    #[serde(default)]
    pub transaction_id: Option<String>,
    pub adapter_semantics_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CargoExecutionIntent {
    pub intent_id: String,
    pub schema: String,
    pub repository_source_before: String,
    pub git_worktree_state_before: Option<String>,
    pub build_context_id: String,
    pub invocation_id: String,
    pub effect_policy_id: String,
    pub plan_id: Option<String>,
    pub transaction_id: Option<String>,
    pub adapter_semantics_digest: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CargoExecutionResult {
    pub result_id: String,
    pub schema: &'static str,
    pub intent_id: String,
    pub repository_source_before: String,
    pub repository_source_after: String,
    pub git_worktree_state_before: Option<String>,
    pub git_worktree_state_after: Option<String>,
    pub build_context_id: String,
    pub invocation_id: String,
    pub effect_policy_id: String,
    pub plan_id: Option<String>,
    pub transaction_id: Option<String>,
    pub adapter_semantics_digest: String,
    pub raw_observation_id: String,
    pub raw_observation_sha256: String,
    pub observed_diff_id: String,
    pub effect_evaluation_sha256: String,
    pub effect_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservationPointer {
    observation_id: String,
    context_id: String,
    invocation_id: String,
}

#[derive(Serialize)]
struct IntentIdentity<'a> {
    schema: &'static str,
    repository_source_before: &'a str,
    git_worktree_state_before: &'a Option<String>,
    build_context_id: &'a str,
    invocation_id: &'a str,
    effect_policy_id: &'a str,
    plan_id: &'a Option<String>,
    transaction_id: &'a Option<String>,
    adapter_semantics_digest: &'a str,
}

#[derive(Serialize)]
struct ResultIdentity<'a> {
    schema: &'static str,
    intent_id: &'a str,
    repository_source_before: &'a str,
    repository_source_after: &'a str,
    git_worktree_state_before: &'a Option<String>,
    git_worktree_state_after: &'a Option<String>,
    build_context_id: &'a str,
    invocation_id: &'a str,
    effect_policy_id: &'a str,
    plan_id: &'a Option<String>,
    transaction_id: &'a Option<String>,
    adapter_semantics_digest: &'a str,
    raw_observation_id: &'a str,
    raw_observation_sha256: &'a str,
    observed_diff_id: &'a str,
    effect_evaluation_sha256: &'a str,
    effect_allowed: bool,
}

pub fn run_intent(
    spec_path: &Path,
    pre_snapshot_path: &Path,
    effect_policy_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo execution intent spec {}", spec_path.display()))?;
    let spec: CargoExecutionIntentSpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo execution intent spec {}", spec_path.display()))?;

    let pre = ValidatedSnapshotReceipt::load(pre_snapshot_path)?;
    let (_, policy_id) = load_effect_policy(effect_policy_path)?;
    let intent = build_intent(spec, &pre.snapshot_id, &policy_id)?;
    write_json("Cargo execution intent", &intent, output)
}

#[allow(clippy::too_many_arguments)]
pub fn run_result(
    intent_path: &Path,
    pre_snapshot_path: &Path,
    post_snapshot_path: &Path,
    observation_path: &Path,
    effect_policy_path: &Path,
    mut git_worktree_state_after: Option<String>,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let intent_bytes = fs::read(intent_path)
        .with_context(|| format!("read Cargo execution intent {}", intent_path.display()))?;
    let mut intent: CargoExecutionIntent = serde_json::from_slice(&intent_bytes)
        .with_context(|| format!("parse Cargo execution intent {}", intent_path.display()))?;
    validate_intent(&mut intent)?;

    let pre = ValidatedSnapshotReceipt::load(pre_snapshot_path)?;
    if pre.snapshot_id != intent.repository_source_before {
        bail!(
            "pre-execution source receipt {} does not match intent subject {}",
            pre.snapshot_id,
            intent.repository_source_before
        );
    }
    let post = ValidatedSnapshotReceipt::load(post_snapshot_path)?;
    let diff = compare(&pre, &post)?;

    let observation_bytes = fs::read(observation_path)
        .with_context(|| format!("read Cargo build observation {}", observation_path.display()))?;
    let observation = parse_observation_pointer(&observation_bytes)?;
    if observation.context_id != intent.build_context_id {
        bail!(
            "Cargo observation context {} does not match intent context {}",
            observation.context_id,
            intent.build_context_id
        );
    }
    if observation.invocation_id != intent.invocation_id {
        bail!(
            "Cargo observation invocation {} does not match intent invocation {}",
            observation.invocation_id,
            intent.invocation_id
        );
    }

    let (policy, policy_id) = load_effect_policy(effect_policy_path)?;
    if policy_id != intent.effect_policy_id {
        bail!(
            "effect policy {} does not match intent policy {}",
            policy_id,
            intent.effect_policy_id
        );
    }
    let evaluation = evaluate(policy_id, &policy, &diff);
    let evaluation_bytes =
        serde_json::to_vec(&evaluation).context("serialize canonical repository effect evaluation")?;

    normalize_optional_digest(
        "git_worktree_state_after",
        &mut git_worktree_state_after,
    )?;

    let result = build_result(
        &intent,
        &post,
        &diff,
        observation,
        sha256(&observation_bytes),
        &evaluation,
        sha256(&evaluation_bytes),
        git_worktree_state_after,
    )?;
    write_json("Cargo execution result", &result, output)
}

fn load_effect_policy(path: &Path) -> anyhow::Result<(EffectPolicySpec, String)> {
    let bytes = fs::read(path)
        .with_context(|| format!("read repository effect policy {}", path.display()))?;
    let mut policy: EffectPolicySpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse repository effect policy {}", path.display()))?;
    let policy_id = validate_and_identify_policy(&mut policy)?;
    Ok((policy, policy_id))
}

pub fn build_intent(
    mut spec: CargoExecutionIntentSpec,
    repository_source_before: &str,
    effect_policy_id: &str,
) -> anyhow::Result<CargoExecutionIntent> {
    if spec.schema != INTENT_INPUT_SCHEMA {
        bail!("unsupported Cargo execution intent input schema: {}", spec.schema);
    }

    let mut repository_source_before = repository_source_before.to_string();
    let mut effect_policy_id = effect_policy_id.to_string();
    normalize_digest("repository_source_before", &mut repository_source_before)?;
    normalize_digest("effect_policy_id", &mut effect_policy_id)?;
    normalize_optional_digest("git_worktree_state_before", &mut spec.git_worktree_state_before)?;
    normalize_digest("build_context_id", &mut spec.build_context_id)?;
    normalize_digest("invocation_id", &mut spec.invocation_id)?;
    normalize_optional_digest("plan_id", &mut spec.plan_id)?;
    normalize_optional_digest("transaction_id", &mut spec.transaction_id)?;
    normalize_digest("adapter_semantics_digest", &mut spec.adapter_semantics_digest)?;

    let identity = IntentIdentity {
        schema: INTENT_SCHEMA,
        repository_source_before: &repository_source_before,
        git_worktree_state_before: &spec.git_worktree_state_before,
        build_context_id: &spec.build_context_id,
        invocation_id: &spec.invocation_id,
        effect_policy_id: &effect_policy_id,
        plan_id: &spec.plan_id,
        transaction_id: &spec.transaction_id,
        adapter_semantics_digest: &spec.adapter_semantics_digest,
    };
    let bytes = serde_json::to_vec(&identity).context("serialize Cargo execution intent identity")?;
    let intent_id = domain_sha256(INTENT_HASH_DOMAIN, &bytes);

    Ok(CargoExecutionIntent {
        intent_id,
        schema: INTENT_SCHEMA.into(),
        repository_source_before,
        git_worktree_state_before: spec.git_worktree_state_before,
        build_context_id: spec.build_context_id,
        invocation_id: spec.invocation_id,
        effect_policy_id,
        plan_id: spec.plan_id,
        transaction_id: spec.transaction_id,
        adapter_semantics_digest: spec.adapter_semantics_digest,
    })
}

pub fn validate_intent(intent: &mut CargoExecutionIntent) -> anyhow::Result<()> {
    if intent.schema != INTENT_SCHEMA {
        bail!("unsupported Cargo execution intent schema: {}", intent.schema);
    }
    normalize_digest("intent_id", &mut intent.intent_id)?;
    normalize_digest(
        "repository_source_before",
        &mut intent.repository_source_before,
    )?;
    normalize_optional_digest(
        "git_worktree_state_before",
        &mut intent.git_worktree_state_before,
    )?;
    normalize_digest("build_context_id", &mut intent.build_context_id)?;
    normalize_digest("invocation_id", &mut intent.invocation_id)?;
    normalize_digest("effect_policy_id", &mut intent.effect_policy_id)?;
    normalize_optional_digest("plan_id", &mut intent.plan_id)?;
    normalize_optional_digest("transaction_id", &mut intent.transaction_id)?;
    normalize_digest(
        "adapter_semantics_digest",
        &mut intent.adapter_semantics_digest,
    )?;

    let identity = IntentIdentity {
        schema: INTENT_SCHEMA,
        repository_source_before: &intent.repository_source_before,
        git_worktree_state_before: &intent.git_worktree_state_before,
        build_context_id: &intent.build_context_id,
        invocation_id: &intent.invocation_id,
        effect_policy_id: &intent.effect_policy_id,
        plan_id: &intent.plan_id,
        transaction_id: &intent.transaction_id,
        adapter_semantics_digest: &intent.adapter_semantics_digest,
    };
    let bytes = serde_json::to_vec(&identity).context("serialize stored Cargo execution intent")?;
    let computed = domain_sha256(INTENT_HASH_DOMAIN, &bytes);
    if computed != intent.intent_id {
        bail!(
            "Cargo execution intent identity mismatch: declared {}, computed {}",
            intent.intent_id,
            computed
        );
    }
    Ok(())
}

fn parse_observation_pointer(bytes: &[u8]) -> anyhow::Result<ObservationPointer> {
    let value: Value = serde_json::from_slice(bytes).context("parse Cargo build observation JSON")?;
    require_schema(&value, CARGO_OBSERVATION_SCHEMA, "Cargo build observation")?;
    let mut observation_id = required_string(&value, "observation_id", "Cargo build observation")?;
    let mut context_id = required_string(&value, "context_id", "Cargo build observation")?;
    let mut invocation_id = required_string(&value, "invocation_id", "Cargo build observation")?;
    normalize_digest("observation_id", &mut observation_id)?;
    normalize_digest("context_id", &mut context_id)?;
    normalize_digest("invocation_id", &mut invocation_id)?;
    Ok(ObservationPointer {
        observation_id,
        context_id,
        invocation_id,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    intent: &CargoExecutionIntent,
    post: &ValidatedSnapshotReceipt,
    diff: &RepositorySourceDiff,
    observation: ObservationPointer,
    raw_observation_sha256: String,
    evaluation: &EffectEvaluation,
    effect_evaluation_sha256: String,
    git_worktree_state_after: Option<String>,
) -> anyhow::Result<CargoExecutionResult> {
    let identity = ResultIdentity {
        schema: RESULT_SCHEMA,
        intent_id: &intent.intent_id,
        repository_source_before: &intent.repository_source_before,
        repository_source_after: &post.snapshot_id,
        git_worktree_state_before: &intent.git_worktree_state_before,
        git_worktree_state_after: &git_worktree_state_after,
        build_context_id: &intent.build_context_id,
        invocation_id: &intent.invocation_id,
        effect_policy_id: &intent.effect_policy_id,
        plan_id: &intent.plan_id,
        transaction_id: &intent.transaction_id,
        adapter_semantics_digest: &intent.adapter_semantics_digest,
        raw_observation_id: &observation.observation_id,
        raw_observation_sha256: &raw_observation_sha256,
        observed_diff_id: &diff.diff_id,
        effect_evaluation_sha256: &effect_evaluation_sha256,
        effect_allowed: evaluation.allowed,
    };
    let bytes = serde_json::to_vec(&identity).context("serialize Cargo execution result identity")?;
    let result_id = domain_sha256(RESULT_HASH_DOMAIN, &bytes);

    Ok(CargoExecutionResult {
        result_id,
        schema: RESULT_SCHEMA,
        intent_id: intent.intent_id.clone(),
        repository_source_before: intent.repository_source_before.clone(),
        repository_source_after: post.snapshot_id.clone(),
        git_worktree_state_before: intent.git_worktree_state_before.clone(),
        git_worktree_state_after,
        build_context_id: intent.build_context_id.clone(),
        invocation_id: intent.invocation_id.clone(),
        effect_policy_id: intent.effect_policy_id.clone(),
        plan_id: intent.plan_id.clone(),
        transaction_id: intent.transaction_id.clone(),
        adapter_semantics_digest: intent.adapter_semantics_digest.clone(),
        raw_observation_id: observation.observation_id,
        raw_observation_sha256,
        observed_diff_id: diff.diff_id.clone(),
        effect_evaluation_sha256,
        effect_allowed: evaluation.allowed,
    })
}

fn require_schema(value: &Value, expected: &str, label: &str) -> anyhow::Result<()> {
    let actual = value
        .get("schema")
        .and_then(Value::as_str)
        .with_context(|| format!("{label} missing/non-string schema"))?;
    if actual != expected {
        bail!("unsupported {label} schema: {actual}");
    }
    Ok(())
}

fn required_string(value: &Value, key: &str, label: &str) -> anyhow::Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("{label} missing/non-string {key}"))
}

fn write_json<T: Serialize>(label: &str, value: &T, output: Option<PathBuf>) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(value)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {label} output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write {label} {}", path.display()))?;
        println!("{label} written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        normalize_digest(name, value)?;
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn intent_spec() -> CargoExecutionIntentSpec {
        CargoExecutionIntentSpec {
            schema: INTENT_INPUT_SCHEMA.into(),
            git_worktree_state_before: None,
            build_context_id: digest('b'),
            invocation_id: digest('c'),
            plan_id: Some(digest('e')),
            transaction_id: Some(digest('f')),
            adapter_semantics_digest: digest('1'),
        }
    }

    #[test]
    fn digest_spelling_normalizes_before_intent_identity() {
        let mut upper = intent_spec();
        upper.build_context_id.make_ascii_uppercase();
        upper.adapter_semantics_digest.make_ascii_uppercase();
        let a = build_intent(upper, &digest('a'), &digest('d')).unwrap();
        let b = build_intent(intent_spec(), &digest('a'), &digest('d')).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn source_subject_substitution_changes_intent_identity() {
        let a = build_intent(intent_spec(), &digest('a'), &digest('d')).unwrap();
        let b = build_intent(intent_spec(), &digest('9'), &digest('d')).unwrap();
        assert_ne!(a.intent_id, b.intent_id);
    }

    #[test]
    fn policy_substitution_changes_intent_identity() {
        let a = build_intent(intent_spec(), &digest('a'), &digest('d')).unwrap();
        let b = build_intent(intent_spec(), &digest('a'), &digest('9')).unwrap();
        assert_ne!(a.intent_id, b.intent_id);
    }

    #[test]
    fn adapter_semantics_are_authorization_significant() {
        let a = build_intent(intent_spec(), &digest('a'), &digest('d')).unwrap();
        let mut other = intent_spec();
        other.adapter_semantics_digest = digest('8');
        let b = build_intent(other, &digest('a'), &digest('d')).unwrap();
        assert_ne!(a.intent_id, b.intent_id);
    }

    #[test]
    fn stored_intent_must_recompute_to_declared_identity() {
        let mut intent = build_intent(intent_spec(), &digest('a'), &digest('d')).unwrap();
        intent.invocation_id = digest('7');
        assert!(validate_intent(&mut intent).is_err());
    }

    #[test]
    fn observation_pointer_requires_expected_schema_and_ids() {
        let bytes = format!(
            "{{\"schema\":\"{}\",\"observation_id\":\"{}\",\"context_id\":\"{}\",\"invocation_id\":\"{}\"}}",
            CARGO_OBSERVATION_SCHEMA,
            digest('2'),
            digest('b'),
            digest('c')
        );
        let pointer = parse_observation_pointer(bytes.as_bytes()).unwrap();
        assert_eq!(pointer.observation_id, digest('2'));
        assert_eq!(pointer.context_id, digest('b'));
        assert_eq!(pointer.invocation_id, digest('c'));
    }

    #[test]
    fn invalid_digest_fails_closed() {
        let mut spec = intent_spec();
        spec.invocation_id = "not-a-digest".into();
        assert!(build_intent(spec, &digest('a'), &digest('d')).is_err());
    }
}
