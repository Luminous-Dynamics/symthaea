use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path};

const CORPUS_SCHEMA: &str = "luminous.eqc.seed-corpus.v0.1";
const TASK_SCHEMA: &str = "luminous.eqc.task.v0.1";
const REQUIRED_PROVENANCE: &[&str] = &[
    "task_digest",
    "input_digests",
    "actor_identity",
    "authority_snapshot",
    "policy_digest",
    "system_identity",
    "environment_identity",
    "action_log",
    "result_digest",
    "independent_verifier",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema: String,
    status: String,
    description: String,
    task_schema: String,
    tasks: Vec<Task>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    schema: String,
    task_id: String,
    domain: String,
    split: String,
    statement: String,
    actor: Actor,
    subject: Subject,
    input_refs: Vec<InputRef>,
    authority_snapshot: AuthoritySnapshot,
    request: Request,
    negative_controls: Vec<String>,
    expected: Expected,
    resource_budget: ResourceBudget,
    required_provenance: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Actor {
    actor_id: String,
    roles: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Subject {
    kind: String,
    subject_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRef {
    input_id: String,
    artifact_path: String,
    classification: String,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoritySnapshot {
    snapshot_id: String,
    status: String,
    principal_id: Option<String>,
    subject_id: Option<String>,
    scopes: Vec<String>,
    observed_at: String,
    valid_until: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    action: String,
    resource_id: String,
    subject_id: String,
    evaluated_at: String,
    #[serde(default)]
    challenge_id: Option<String>,
    #[serde(default)]
    challenge_state: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    disposition: String,
    state_mutation_count: u64,
    state_must_remain_unchanged: bool,
    #[serde(default)]
    reason_code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceBudget {
    timeout_seconds: u64,
    max_tool_calls: u64,
    network_policy: String,
    #[serde(default)]
    cpu_seconds: Option<u64>,
    #[serde(default)]
    memory_mib: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EqcReport {
    pub schema: &'static str,
    pub evaluation_mode: &'static str,
    pub corpus_sha256: String,
    pub task_count: usize,
    pub reference_pass_count: usize,
    pub no_op_fails_positive_controls: bool,
    pub allow_all_fails_negative_controls: bool,
    pub task_schema_validated: bool,
    pub receipt_schema_meta_validated: bool,
    pub tasks: Vec<TaskReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskReport {
    pub task_id: String,
    pub domain: String,
    pub expected_disposition: String,
    pub reference_disposition: String,
    pub no_op_disposition: String,
    pub allow_all_disposition: String,
    pub expected_state_mutation_count: u64,
    pub state_unchanged_requirement: bool,
    pub negative_controls: Vec<String>,
    pub reference_pass: bool,
}

pub fn validate_corpus_file(path: &Path, repository_root: &Path) -> anyhow::Result<EqcReport> {
    let root = repository_root
        .canonicalize()
        .with_context(|| format!("resolve repository root {}", repository_root.display()))?;
    let corpus_path = path
        .canonicalize()
        .with_context(|| format!("resolve EQC seed corpus {}", path.display()))?;
    if !corpus_path.starts_with(&root) {
        bail!("EQC seed corpus must be inside the repository root");
    }
    let bytes = fs::read(&corpus_path)
        .with_context(|| format!("read EQC seed corpus {}", corpus_path.display()))?;
    let corpus_value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse EQC seed JSON {}", corpus_path.display()))?;
    let task_schema_path = root.join("docs/research/eqc-task-v0.1.schema.json");
    let receipt_schema_path = root.join("docs/research/eqc-run-receipt-v0.1.schema.json");

    let task_values = corpus_value
        .get("tasks")
        .and_then(Value::as_array)
        .context("EQC corpus must contain a tasks array")?;
    validate_schema_instances(&task_schema_path, task_values)?;
    validate_schema_definition(&receipt_schema_path)?;

    let corpus: Corpus = serde_json::from_value(corpus_value)
        .with_context(|| format!("deserialize EQC seed corpus {}", corpus_path.display()))?;
    let mut report = validate_corpus(&corpus, &bytes, &root)?;
    report.task_schema_validated = true;
    report.receipt_schema_meta_validated = true;
    Ok(report)
}

fn load_schema(path: &Path) -> anyhow::Result<Value> {
    let bytes = fs::read(path)
        .with_context(|| format!("read JSON Schema {}", path.display()))?;
    let schema: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse JSON Schema {}", path.display()))?;
    jsonschema::meta::validate(&schema)
        .with_context(|| format!("JSON Schema meta-validation failed for {}", path.display()))?;
    Ok(schema)
}

fn validate_schema_definition(path: &Path) -> anyhow::Result<()> {
    let schema = load_schema(path)?;
    jsonschema::options()
        .offline()
        .should_validate_formats(true)
        .build(&schema)
        .with_context(|| format!("compile offline Draft 2020-12 schema {}", path.display()))?;
    Ok(())
}

fn validate_schema_instances(path: &Path, instances: &[Value]) -> anyhow::Result<()> {
    let schema = load_schema(path)?;
    let validator = jsonschema::options()
        .offline()
        .should_validate_formats(true)
        .build(&schema)
        .with_context(|| format!("compile offline Draft 2020-12 schema {}", path.display()))?;

    for (index, instance) in instances.iter().enumerate() {
        let errors = validator
            .iter_errors(instance)
            .map(|error| format!("{} at {}", error, error.instance_path()))
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            bail!(
                "JSON Schema validation failed for {} instance {}: {}",
                path.display(),
                index,
                errors.join("; ")
            );
        }
    }
    Ok(())
}

fn validate_corpus(
    corpus: &Corpus,
    raw_bytes: &[u8],
    repository_root: &Path,
) -> anyhow::Result<EqcReport> {
    if corpus.schema != CORPUS_SCHEMA {
        bail!("unsupported EQC corpus schema: {}", corpus.schema);
    }
    if corpus.status != "synthetic_public_seed_unqualified" {
        bail!(
            "public v0.1 corpus must remain explicitly unqualified, got status '{}'",
            corpus.status
        );
    }
    if corpus.description.trim().is_empty() || corpus.task_schema != "eqc-task-v0.1.schema.json" {
        bail!("corpus description or task-schema reference is missing");
    }
    if corpus.tasks.len() < 2 {
        bail!("EQC corpus needs at least one positive and one negative-control task");
    }

    let mut ids = HashSet::new();
    let mut reports = Vec::with_capacity(corpus.tasks.len());
    let mut reference_pass_count = 0usize;
    let mut has_positive = false;
    let mut has_negative = false;
    let mut no_op_fails_positive_controls = false;
    let mut allow_all_fails_negative_controls = false;

    for task in &corpus.tasks {
        validate_task(task, repository_root)?;
        if !ids.insert(task.task_id.as_str()) {
            bail!("duplicate EQC task id: {}", task.task_id);
        }

        let reference = reference_disposition(task);
        let no_op = "deny";
        let allow_all = "allow";
        let reference_pass = reference == task.expected.disposition
            && task.expected.state_mutation_count == 0
            && task.expected.state_must_remain_unchanged;

        if reference_pass {
            reference_pass_count += 1;
        }
        if task.expected.disposition == "allow" {
            has_positive = true;
            if no_op != task.expected.disposition {
                no_op_fails_positive_controls = true;
            }
        }
        if task.expected.disposition == "deny" {
            has_negative = true;
            if allow_all != task.expected.disposition {
                allow_all_fails_negative_controls = true;
            }
        }

        reports.push(TaskReport {
            task_id: task.task_id.clone(),
            domain: task.domain.clone(),
            expected_disposition: task.expected.disposition.clone(),
            reference_disposition: reference.to_owned(),
            no_op_disposition: no_op.to_owned(),
            allow_all_disposition: allow_all.to_owned(),
            expected_state_mutation_count: task.expected.state_mutation_count,
            state_unchanged_requirement: task.expected.state_must_remain_unchanged,
            negative_controls: task.negative_controls.clone(),
            reference_pass,
        });
    }

    if !has_positive || !has_negative {
        bail!("EQC seed must contain both positive and negative controls");
    }
    if !no_op_fails_positive_controls {
        bail!("two-sided validity failure: no-op baseline passed every positive control");
    }
    if !allow_all_fails_negative_controls {
        bail!("negative-control failure: always-allow baseline passed every denial control");
    }
    if reference_pass_count != corpus.tasks.len() {
        let failures = reports
            .iter()
            .filter(|task| !task.reference_pass)
            .map(|task| task.task_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        bail!("reference evaluator disagrees with corpus on: {failures}");
    }

    let corpus_sha256 = format!("{:x}", Sha256::digest(raw_bytes));
    Ok(EqcReport {
        schema: "luminous.eqc.reference-validation.v0.1",
        evaluation_mode: "reference_contract_only_not_system_under_test",
        corpus_sha256,
        task_count: reports.len(),
        reference_pass_count,
        no_op_fails_positive_controls,
        allow_all_fails_negative_controls,
        task_schema_validated: false,
        receipt_schema_meta_validated: false,
        tasks: reports,
    })
}

fn validate_task(task: &Task, repository_root: &Path) -> anyhow::Result<()> {
    if task.schema != TASK_SCHEMA {
        bail!("{}: unsupported task schema", task.task_id);
    }
    if task.task_id.trim().is_empty() || task.statement.trim().len() < 20 {
        bail!("{}: task id or statement is incomplete", task.task_id);
    }
    if !matches!(
        task.domain.as_str(),
        "software_engineering" | "healthcare" | "simulation" | "trust_infrastructure"
    ) {
        bail!("{}: unsupported EQC domain '{}'", task.task_id, task.domain);
    }
    if task.split != "public_seed" {
        bail!(
            "{}: v0.1 seed validator only accepts public_seed tasks",
            task.task_id
        );
    }
    if task.actor.actor_id.is_empty()
        || task.actor.roles.is_empty()
        || task.subject.subject_id.is_empty()
        || task.subject.kind.is_empty()
        || task.request.resource_id.is_empty()
        || task.request.action.is_empty()
        || task.request.subject_id.is_empty()
    {
        bail!(
            "{}: actor, subject, and request fields must be non-empty",
            task.task_id
        );
    }
    if !is_canonical_utc(&task.authority_snapshot.observed_at) {
        bail!(
            "{}: observed_at must be canonical UTC RFC3339",
            task.task_id
        );
    }
    if !is_canonical_utc(&task.request.evaluated_at) {
        bail!(
            "{}: evaluated_at must be canonical UTC RFC3339",
            task.task_id
        );
    }
    if task.authority_snapshot.observed_at > task.request.evaluated_at {
        bail!(
            "{}: authority snapshot cannot be observed after the request evaluation time",
            task.task_id
        );
    }
    if let Some(valid_until) = task.authority_snapshot.valid_until.as_deref() {
        if !is_canonical_utc(valid_until) {
            bail!(
                "{}: valid_until must be canonical UTC RFC3339",
                task.task_id
            );
        }
    }
    if task.authority_snapshot.snapshot_id.is_empty() {
        bail!("{}: authority snapshot id is missing", task.task_id);
    }
    if task.resource_budget.timeout_seconds == 0
        || task.resource_budget.max_tool_calls == 0
        || task.resource_budget.network_policy != "none"
        || task.resource_budget.cpu_seconds == Some(0)
        || task.resource_budget.memory_mib == Some(0)
    {
        bail!(
            "{}: public synthetic seed must have positive budgets and no network",
            task.task_id
        );
    }
    if !matches!(task.expected.disposition.as_str(), "allow" | "deny") {
        bail!(
            "{}: v0.1 reference evaluator supports allow/deny expected outcomes",
            task.task_id
        );
    }
    if task.expected.state_mutation_count != 0 || !task.expected.state_must_remain_unchanged {
        bail!(
            "{}: public authorization seed must assert zero state mutations",
            task.task_id
        );
    }
    if task.input_refs.is_empty() {
        bail!(
            "{}: every EQC task must bind at least one input artifact",
            task.task_id
        );
    }
    if task.expected.disposition == "deny" && task.negative_controls.is_empty() {
        bail!("{}: denied task must name a negative control", task.task_id);
    }
    if task
        .expected
        .reason_code
        .as_deref()
        .is_some_and(str::is_empty)
    {
        bail!("{}: reason_code cannot be empty when present", task.task_id);
    }
    for required in REQUIRED_PROVENANCE {
        if !task.required_provenance.iter().any(|item| item == required) {
            bail!(
                "{}: required provenance field '{}' is missing",
                task.task_id,
                required
            );
        }
    }
    for input in &task.input_refs {
        let relative_path = Path::new(&input.artifact_path);
        if input.input_id.is_empty()
            || input.artifact_path.is_empty()
            || relative_path.is_absolute()
            || input.artifact_path.starts_with("http://")
            || input.artifact_path.starts_with("https://")
            || relative_path.components().any(|part| {
                matches!(
                    part,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
            || input.classification != "synthetic"
        {
            bail!(
                "{}: seed inputs must be repository-local synthetic artifacts",
                task.task_id
            );
        }
        if !is_sha256(&input.sha256) {
            bail!(
                "{}: input '{}' has malformed SHA-256",
                task.task_id,
                input.input_id
            );
        }
        let unresolved_path = repository_root.join(relative_path);
        let resolved_path = unresolved_path.canonicalize().with_context(|| {
            format!(
                "{}: resolve synthetic input '{}'",
                task.task_id, input.input_id
            )
        })?;
        if !resolved_path.starts_with(repository_root) {
            bail!(
                "{}: input '{}' escapes the repository root",
                task.task_id,
                input.input_id
            );
        }
        let input_bytes = fs::read(&resolved_path).with_context(|| {
            format!(
                "{}: read synthetic input '{}'",
                task.task_id, input.input_id
            )
        })?;
        let observed_digest = format!("{:x}", Sha256::digest(&input_bytes));
        if observed_digest != input.sha256 {
            bail!(
                "{}: SHA-256 mismatch for input '{}'",
                task.task_id,
                input.input_id
            );
        }
        validate_synthetic_input(task, input, &input_bytes)?;
    }
    if task.request.action == "challenge.consume"
        && (task.request.challenge_id.is_none() || task.request.challenge_state.is_none())
    {
        bail!(
            "{}: challenge consumption needs challenge id and state",
            task.task_id
        );
    }
    Ok(())
}

fn json_string<'a>(
    artifact: &'a Value,
    field: &str,
    task: &Task,
    input: &InputRef,
) -> anyhow::Result<&'a str> {
    artifact
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| {
            format!(
                "{}: synthetic input '{}' is missing string field '{}'",
                task.task_id, input.input_id, field
            )
        })
}

fn validate_synthetic_input(task: &Task, input: &InputRef, bytes: &[u8]) -> anyhow::Result<()> {
    let artifact: Value = serde_json::from_slice(bytes).with_context(|| {
        format!(
            "{}: parse synthetic input '{}'",
            task.task_id, input.input_id
        )
    })?;

    match input.input_id.as_str() {
        "synthetic-consent-001" | "synthetic-consent-revoked-001" => {
            if json_string(&artifact, "schema", task, input)?
                != "luminous.eqc.synthetic-consent.v0.1"
                || json_string(&artifact, "consent_id", task, input)? != input.input_id
                || Some(json_string(&artifact, "principal_id", task, input)?)
                    != task.authority_snapshot.principal_id.as_deref()
                || Some(json_string(&artifact, "subject_id", task, input)?)
                    != task.authority_snapshot.subject_id.as_deref()
                || json_string(&artifact, "status", task, input)?
                    != task.authority_snapshot.status.as_str()
            {
                bail!(
                    "{}: consent fixture '{}' does not match the authority snapshot",
                    task.task_id,
                    input.input_id
                );
            }

            let fixture_scopes = artifact
                .get("scopes")
                .and_then(Value::as_array)
                .with_context(|| {
                    format!(
                        "{}: consent fixture '{}' has no scopes array",
                        task.task_id, input.input_id
                    )
                })?
                .iter()
                .map(|scope| {
                    scope.as_str().map(str::to_owned).with_context(|| {
                        format!("{}: consent fixture scopes must be strings", task.task_id)
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let valid_from = json_string(&artifact, "valid_from", task, input)?;
            if !is_canonical_utc(valid_from)
                || fixture_scopes != task.authority_snapshot.scopes
                || Some(json_string(&artifact, "valid_until", task, input)?)
                    != task.authority_snapshot.valid_until.as_deref()
                || valid_from > task.request.evaluated_at.as_str()
            {
                bail!(
                    "{}: consent fixture '{}' disagrees with scopes or temporal bounds",
                    task.task_id,
                    input.input_id
                );
            }

            match task.authority_snapshot.status.as_str() {
                "active" if artifact.get("revoked_at").is_none() => {}
                "revoked" => {
                    let revoked_at = json_string(&artifact, "revoked_at", task, input)?;
                    if !is_canonical_utc(revoked_at)
                        || revoked_at > task.request.evaluated_at.as_str()
                    {
                        bail!(
                            "{}: revoked consent fixture '{}' has invalid revocation time",
                            task.task_id,
                            input.input_id
                        );
                    }
                }
                _ => bail!(
                    "{}: consent fixture '{}' uses an unsupported authority state",
                    task.task_id,
                    input.input_id
                ),
            }
        }
        "synthetic-challenge-001" => {
            if json_string(&artifact, "schema", task, input)?
                != "luminous.eqc.synthetic-challenge.v0.1"
                || Some(json_string(&artifact, "challenge_id", task, input)?)
                    != task.request.challenge_id.as_deref()
                || json_string(&artifact, "issuer_id", task, input)? != task.actor.actor_id.as_str()
                || json_string(&artifact, "subject_id", task, input)? != task.subject.subject_id
                || json_string(&artifact, "state", task, input)?
                    != task.request.challenge_state.as_deref().unwrap_or("")
            {
                bail!(
                    "{}: challenge fixture '{}' does not match the request",
                    task.task_id,
                    input.input_id
                );
            }
            let consumed_at = json_string(&artifact, "consumed_at", task, input)?;
            if !is_canonical_utc(consumed_at) || consumed_at > task.request.evaluated_at.as_str() {
                bail!(
                    "{}: consumed challenge fixture '{}' has invalid consumption time",
                    task.task_id,
                    input.input_id
                );
            }
        }
        other => bail!(
            "{}: unsupported synthetic input identifier '{}'",
            task.task_id,
            other
        ),
    }
    Ok(())
}

fn reference_disposition(task: &Task) -> &'static str {
    let authority = &task.authority_snapshot;
    let request = &task.request;

    if authority.status != "active" {
        return "deny";
    }
    if authority.principal_id.as_deref() != Some(task.actor.actor_id.as_str()) {
        return "deny";
    }
    if authority.subject_id.as_deref() != Some(request.subject_id.as_str())
        || task.subject.subject_id != request.subject_id
    {
        return "deny";
    }
    let Some(valid_until) = authority.valid_until.as_deref() else {
        return "deny";
    };
    if valid_until <= request.evaluated_at.as_str() {
        return "deny";
    }

    let required_scope = match request.action.as_str() {
        "patient.read" if task.subject.kind == "patient" => {
            format!("patient:{}:read", request.subject_id)
        }
        "challenge.consume" => {
            let Some(challenge_id) = request.challenge_id.as_deref() else {
                return "deny";
            };
            if request.challenge_state.as_deref() != Some("unused") {
                return "deny";
            }
            format!("challenge:{challenge_id}:consume")
        }
        _ => format!("{}:{}", request.action, request.resource_id),
    };

    if !authority
        .scopes
        .iter()
        .any(|scope| scope == &required_scope)
    {
        return "deny";
    }
    "allow"
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn parse_decimal(bytes: &[u8]) -> Option<u32> {
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

fn is_canonical_utc(value: &str) -> bool {
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

    let year = match parse_decimal(&bytes[0..4]) {
        Some(value) if value > 0 => value,
        _ => return false,
    };
    let month = match parse_decimal(&bytes[5..7]) {
        Some(value @ 1..=12) => value,
        _ => return false,
    };
    let day = match parse_decimal(&bytes[8..10]) {
        Some(value) => value,
        None => return false,
    };
    let hour = match parse_decimal(&bytes[11..13]) {
        Some(value @ 0..=23) => value,
        _ => return false,
    };
    let minute = match parse_decimal(&bytes[14..16]) {
        Some(value @ 0..=59) => value,
        _ => return false,
    };
    let second = match parse_decimal(&bytes[17..19]) {
        Some(value @ 0..=59) => value,
        _ => return false,
    };
    let _ = (hour, minute, second);

    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let maximum_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    (1..=maximum_day).contains(&day)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_corpus() -> Corpus {
        serde_json::from_str(include_str!(
            "../../../docs/research/eqc-seed-corpus-v0.1.json"
        ))
        .expect("EQC public seed parses")
    }

    fn repository_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root resolves")
    }

    #[test]
    fn public_seed_reference_passes_and_baselines_fail_their_controls() {
        let root = repository_root();
        let report =
            validate_corpus_file(&root.join("docs/research/eqc-seed-corpus-v0.1.json"), &root)
                .expect("seed validates");
        assert_eq!(report.task_count, 4);
        assert_eq!(report.reference_pass_count, 4);
        assert!(report.no_op_fails_positive_controls);
        assert!(report.allow_all_fails_negative_controls);
        assert!(report.task_schema_validated);
        assert!(report.receipt_schema_meta_validated);
        assert_eq!(
            report.evaluation_mode,
            "reference_contract_only_not_system_under_test"
        );
    }

    fn receipt_schema_validator() -> jsonschema::Validator {
        let schema_path = repository_root()
            .join("docs/research/eqc-run-receipt-v0.1.schema.json");
        let schema = load_schema(&schema_path).expect("receipt schema meta-validates");
        jsonschema::options()
            .offline()
            .should_validate_formats(true)
            .build(&schema)
            .expect("receipt schema compiles offline")
    }

    fn valid_receipt() -> Value {
        serde_json::json!({
            "schema": "luminous.eqc.run-receipt.v0.1",
            "run_id": "eqc-run-seed-001",
            "task_id": "eqc-health-read-authorized-001",
            "task_digest_sha256": "a".repeat(64),
            "source_head": "b".repeat(40),
            "system": {
                "system_id": "eqc-reference",
                "version": "0.1.0",
                "harness_version": "0.1.0",
                "model_id": null
            },
            "environment": {
                "image_digest": "sha256:fixture-image",
                "runtime": "rust-1.96.0",
                "network_policy": "none",
                "host": "linux-x86_64",
                "cpu_model": null,
                "gpu_model": null
            },
            "authority_snapshot_sha256": "c".repeat(64),
            "policy_digest_sha256": "d".repeat(64),
            "input_digests": [{
                "input_id": "synthetic-consent-001",
                "sha256": "e".repeat(64)
            }],
            "actions": [],
            "observed_result": {
                "disposition": "allow",
                "state_mutation_count": 0,
                "result_digest_sha256": "f".repeat(64)
            },
            "verifier": {
                "verifier_id": "eqc-independent-seed-verifier",
                "verifier_digest_sha256": "1".repeat(64),
                "replay": "passed",
                "independent": true,
                "notes": "Synthetic schema fixture only"
            },
            "dimensions": {
                "capability": true,
                "authority": true,
                "input_integrity": true,
                "provenance": true,
                "reproducibility": true,
                "failure_containment": true
            },
            "qualified": true,
            "timestamp": "2026-11-26T00:00:00Z"
        })
    }

    #[test]
    fn task_schema_accepts_all_public_seed_tasks() {
        let root = repository_root();
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../docs/research/eqc-seed-corpus-v0.1.json"
        ))
        .expect("seed JSON parses");
        let tasks = corpus["tasks"].as_array().expect("tasks array");
        let schema = load_schema(&root.join("docs/research/eqc-task-v0.1.schema.json"))
            .expect("task schema meta-validates");
        let validator = jsonschema::options()
            .offline()
            .should_validate_formats(true)
            .build(&schema)
            .expect("task schema compiles offline");
        assert_eq!(tasks.len(), 4);
        for (index, task) in tasks.iter().enumerate() {
            assert!(
                validator.is_valid(task),
                "public seed task {index} must satisfy the published task schema"
            );
        }
    }

    #[test]
    fn task_schema_rejects_missing_digest_bound_inputs() {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../docs/research/eqc-seed-corpus-v0.1.json"
        ))
        .expect("seed JSON parses");
        let mut task = corpus["tasks"][0].clone();
        task.as_object_mut()
            .expect("task object")
            .remove("input_refs");
        assert!(!receipt_schema_validator().is_valid(&valid_receipt()) == false);
        let root = repository_root();
        let schema = load_schema(&root.join("docs/research/eqc-task-v0.1.schema.json"))
            .expect("task schema meta-validates");
        let validator = jsonschema::options()
            .offline()
            .should_validate_formats(true)
            .build(&schema)
            .expect("task schema compiles offline");
        assert!(!validator.is_valid(&task));
    }

    #[test]
    fn receipt_schema_accepts_a_fully_qualified_receipt() {
        let validator = receipt_schema_validator();
        assert!(validator.is_valid(&valid_receipt()));
    }

    #[test]
    fn receipt_schema_rejects_compensated_integrity_failure() {
        let validator = receipt_schema_validator();
        let mut receipt = valid_receipt();
        receipt["dimensions"]["authority"] = Value::Bool(false);
        assert!(!validator.is_valid(&receipt));
    }

    #[test]
    fn receipt_schema_rejects_qualification_without_independent_replay() {
        let validator = receipt_schema_validator();
        let mut receipt = valid_receipt();
        receipt["verifier"]["independent"] = Value::Bool(false);
        assert!(!validator.is_valid(&receipt));

        receipt["verifier"]["independent"] = Value::Bool(true);
        receipt["verifier"]["replay"] = Value::String("not_run".to_owned());
        assert!(!validator.is_valid(&receipt));
    }

    #[test]
    fn receipt_schema_allows_a_failed_unqualified_episode_to_be_recorded() {
        let validator = receipt_schema_validator();
        let mut receipt = valid_receipt();
        receipt["qualified"] = Value::Bool(false);
        receipt["dimensions"]["authority"] = Value::Bool(false);
        receipt["verifier"]["independent"] = Value::Bool(false);
        receipt["verifier"]["replay"] = Value::String("not_run".to_owned());
        assert!(validator.is_valid(&receipt));
    }

    #[test]
    fn wrong_subject_fails_closed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.request.subject_id = "synthetic-patient-999".into();
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn revoked_authority_fails_closed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.status = "revoked".into();
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn missing_authority_fails_closed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.status = "missing".into();
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn expired_authority_fails_closed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.valid_until = Some("2026-11-25T00:00:00Z".into());
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn scope_mismatch_fails_closed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.scopes.clear();
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn consumed_challenge_cannot_be_replayed() {
        let task = seed_corpus()
            .tasks
            .into_iter()
            .find(|task| task.task_id == "eqc-trust-replayed-challenge-deny-001")
            .unwrap();
        assert_eq!(task.request.challenge_state.as_deref(), Some("consumed"));
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn malformed_timestamps_are_rejected_before_decision() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.observed_at = "not-a-date".into();
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn future_authority_snapshot_is_not_used_for_an_earlier_request() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.observed_at = "2026-11-26T02:00:00Z".into();
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn tampered_fixture_digest_is_rejected() {
        let mut task = seed_corpus().tasks.remove(0);
        task.input_refs[0].sha256 = "0".repeat(64);
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn consent_fixture_must_match_authority_snapshot() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.scopes.clear();
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn path_traversal_is_rejected_before_reading_inputs() {
        let mut task = seed_corpus().tasks.remove(0);
        task.input_refs[0].artifact_path = "../Cargo.toml".into();
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn challenge_fixture_must_match_consumed_state() {
        let mut task = seed_corpus()
            .tasks
            .into_iter()
            .find(|task| task.task_id == "eqc-trust-replayed-challenge-deny-001")
            .unwrap();
        task.request.challenge_state = Some("unused".into());
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn non_synthetic_inputs_are_rejected_from_public_seed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.input_refs[0].classification = "restricted".into();
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn denial_tasks_require_zero_mutations() {
        let mut task = seed_corpus()
            .tasks
            .into_iter()
            .find(|task| task.expected.disposition == "deny")
            .unwrap();
        task.expected.state_mutation_count = 1;
        assert!(validate_task(&task, &repository_root()).is_err());
    }

    #[test]
    fn report_is_json_serializable_without_claiming_system_qualification() {
        let raw = include_bytes!("../../../docs/research/eqc-seed-corpus-v0.1.json");
        let root = repository_root();
        let report = validate_corpus(
            &serde_json::from_slice(raw).expect("corpus parses"),
            raw,
            &root,
        )
        .expect("corpus validates");
        let encoded = serde_json::to_vec(&report).expect("report serializes");
        let decoded: serde_json::Value =
            serde_json::from_slice(&encoded).expect("report JSON parses");
        assert_eq!(decoded["task_count"], 4);
        assert!(decoded.get("qualified").is_none());
    }
}
