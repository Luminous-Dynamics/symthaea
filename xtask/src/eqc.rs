use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

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
    #[serde(default)]
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
    #[serde(default)]
    sha256: Option<String>,
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

pub fn validate_corpus_file(path: &Path) -> anyhow::Result<EqcReport> {
    let bytes = fs::read(path)
        .with_context(|| format!("read EQC seed corpus {}", path.display()))?;
    let corpus: Corpus = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse EQC seed corpus {}", path.display()))?;
    validate_corpus(&corpus, &bytes)
}

fn validate_corpus(corpus: &Corpus, raw_bytes: &[u8]) -> anyhow::Result<EqcReport> {
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
        validate_task(task)?;
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
        tasks: reports,
    })
}

fn validate_task(task: &Task) -> anyhow::Result<()> {
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
        bail!("{}: v0.1 seed validator only accepts public_seed tasks", task.task_id);
    }
    if task.actor.actor_id.is_empty()
        || task.actor.roles.is_empty()
        || task.subject.subject_id.is_empty()
        || task.subject.kind.is_empty()
        || task.request.resource_id.is_empty()
        || task.request.action.is_empty()
        || task.request.subject_id.is_empty()
    {
        bail!("{}: actor, subject, and request fields must be non-empty", task.task_id);
    }
    if !is_canonical_utc(&task.authority_snapshot.observed_at) {
        bail!("{}: observed_at must be canonical UTC RFC3339", task.task_id);
    }
    if let Some(valid_until) = task.authority_snapshot.valid_until.as_deref() {
        if !is_canonical_utc(valid_until) {
            bail!("{}: valid_until must be canonical UTC RFC3339", task.task_id);
        }
    }
    if task.authority_snapshot.snapshot_id.is_empty() {
        bail!("{}: authority snapshot id is missing", task.task_id);
    }
    if task.resource_budget.timeout_seconds == 0
        || task.resource_budget.network_policy != "none"
        || task.resource_budget.cpu_seconds == Some(0)
        || task.resource_budget.memory_mib == Some(0)
    {
        bail!("{}: public synthetic seed must have positive budgets and no network", task.task_id);
    }
    if !matches!(task.expected.disposition.as_str(), "allow" | "deny") {
        bail!("{}: v0.1 reference evaluator supports allow/deny expected outcomes", task.task_id);
    }
    if task.expected.state_mutation_count != 0 || !task.expected.state_must_remain_unchanged {
        bail!("{}: public authorization seed must assert zero state mutations", task.task_id);
    }
    if task.expected.disposition == "deny" && task.negative_controls.is_empty() {
        bail!("{}: denied task must name a negative control", task.task_id);
    }
    for required in REQUIRED_PROVENANCE {
        if !task.required_provenance.iter().any(|item| item == required) {
            bail!("{}: required provenance field '{}' is missing", task.task_id, required);
        }
    }
    for input in &task.input_refs {
        if input.input_id.is_empty()
            || input.artifact_path.is_empty()
            || input.artifact_path.starts_with("http://")
            || input.artifact_path.starts_with("https://")
            || input.artifact_path.split('/').any(|part| part == "..")
            || input.classification != "synthetic"
        {
            bail!("{}: seed inputs must be local, synthetic artifacts", task.task_id);
        }
        if let Some(digest) = input.sha256.as_deref() {
            if !is_sha256(digest) {
                bail!("{}: input '{}' has malformed SHA-256", task.task_id, input.input_id);
            }
        }
    }
    if task.request.action == "challenge.consume"
        && (task.request.challenge_id.is_none() || task.request.challenge_state.is_none())
    {
        bail!("{}: challenge consumption needs challenge id and state", task.task_id);
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
    if valid_until <= authority.observed_at.as_str() {
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

    if !authority.scopes.iter().any(|scope| scope == &required_scope) {
        return "deny";
    }
    "allow"
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_canonical_utc(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 20
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[10] == b'T'
        && bytes[13] == b':'
        && bytes[16] == b':'
        && bytes[19] == b'Z'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_corpus() -> Corpus {
        serde_json::from_str(include_str!("../../docs/research/eqc-seed-corpus-v0.1.json"))
            .expect("EQC public seed parses")
    }

    #[test]
    fn public_seed_reference_passes_and_baselines_fail_their_controls() {
        let raw = include_bytes!("../../docs/research/eqc-seed-corpus-v0.1.json");
        let report = validate_corpus(&seed_corpus(), raw).expect("seed validates");
        assert_eq!(report.task_count, 4);
        assert_eq!(report.reference_pass_count, 4);
        assert!(report.no_op_fails_positive_controls);
        assert!(report.allow_all_fails_negative_controls);
        assert_eq!(report.evaluation_mode, "reference_contract_only_not_system_under_test");
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
        let task = seed_corpus().tasks.into_iter().find(|task| task.task_id == "eqc-trust-replayed-challenge-deny-001").unwrap();
        assert_eq!(task.request.challenge_state.as_deref(), Some("consumed"));
        assert_eq!(reference_disposition(&task), "deny");
    }

    #[test]
    fn malformed_timestamps_are_rejected_before_decision() {
        let mut task = seed_corpus().tasks.remove(0);
        task.authority_snapshot.observed_at = "not-a-date".into();
        assert!(validate_task(&task).is_err());
    }

    #[test]
    fn non_synthetic_inputs_are_rejected_from_public_seed() {
        let mut task = seed_corpus().tasks.remove(0);
        task.input_refs[0].classification = "restricted".into();
        assert!(validate_task(&task).is_err());
    }

    #[test]
    fn denial_tasks_require_zero_mutations() {
        let mut task = seed_corpus().tasks.into_iter().find(|task| task.expected.disposition == "deny").unwrap();
        task.expected.state_mutation_count = 1;
        assert!(validate_task(&task).is_err());
    }

    #[test]
    fn report_is_json_serializable_without_claiming_system_qualification() {
        let raw = include_bytes!("../../docs/research/eqc-seed-corpus-v0.1.json");
        let report = validate_corpus(
            &serde_json::from_slice(raw).expect("corpus parses"),
            raw,
        )
        .expect("corpus validates");
        let encoded = serde_json::to_vec(&report).expect("report serializes");
        let decoded: serde_json::Value = serde_json::from_slice(&encoded).expect("report JSON parses");
        assert_eq!(decoded["task_count"], 4);
        assert!(decoded.get("qualified").is_none());
    }
}
