// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed semantic verifier for bounded speculative proof ledgers.
//! A branch candidate is untrusted; promotion binds its exact bytes to a
//! separate checker, an independent replay, and an unchanged target type.

#[path = "../snapshot_branch_receipt.rs"]
mod snapshot_branch_receipt;

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ledger {
    schema_version: String,
    evidence_class: String,
    task: TaskIdentity,
    arm_id: String,
    snapshot_receipt_sha256: Option<String>,
    budget: Budget,
    actual: Actual,
    branches: Vec<Branch>,
    promotion: Promotion,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskIdentity {
    task_id: String,
    repository_commit: String,
    target_type_sha256: String,
    proof_state_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Budget {
    max_branches: u64,
    max_active_branches: u64,
    max_lean_queries: u64,
    max_tokens: u64,
    max_model_calls: u64,
    max_lean_cpu_ms: u64,
    max_wall_ms: u64,
    max_branch_wall_work_ms: u64,
    max_cost_microusd: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Actual {
    branches_started: u64,
    peak_active_branches: u64,
    peak_lean_queries: u64,
    tokens: u64,
    model_calls: u64,
    lean_cpu_ms: u64,
    wall_ms: u64,
    branch_wall_work_ms: u64,
    cost_microusd: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Branch {
    branch_id: String,
    parent_branch_id: Option<String>,
    kind: String,
    workspace_id: String,
    workspace_isolation: String,
    base_commit: String,
    proof_state_sha256: String,
    status: String,
    result_type: String,
    candidate_sha256: Option<String>,
    resources: BranchResources,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BranchResources {
    tokens: u64,
    model_calls: u64,
    lean_cpu_ms: u64,
    wall_ms: u64,
    cost_microusd: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Promotion {
    status: String,
    branch_id: Option<String>,
    candidate_sha256: Option<String>,
    source_patch_sha256: Option<String>,
    checker_id: Option<String>,
    replay_verifier_id: Option<String>,
    checker_candidate_sha256: Option<String>,
    replay_candidate_sha256: Option<String>,
    target_type_before_sha256: Option<String>,
    target_type_after_sha256: Option<String>,
    patch_scope: Option<String>,
    checker_receipt_sha256: Option<String>,
    replay_receipt_sha256: Option<String>,
    kernel_check: String,
    contains_sorry_ax: bool,
    new_axiom_names: Vec<String>,
}

enum Mode {
    Help,
    Validate {
        ledger_path: PathBuf,
        snapshot_receipt_path: Option<PathBuf>,
    },
    SelfTest(PathBuf),
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_branch_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 64
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
}

fn checked_sum<I>(values: I, label: &str) -> Result<u64>
where
    I: IntoIterator<Item = u64>,
{
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .with_context(|| format!("{label} resource sum overflow"))
    })
}

fn valid_optional_digest(value: &Option<String>) -> bool {
    value
        .as_deref()
        .is_some_and(|value| is_lower_hex(value, 64))
}

fn validate(ledger: &Ledger) -> Result<()> {
    if ledger.schema_version != "pitb.sorrydb.speculative-branch-ledger.v0" {
        bail!("unsupported branch-ledger schema version");
    }
    if !matches!(
        ledger.evidence_class.as_str(),
        "synthetic_contract_fixture" | "measured_run"
    ) {
        bail!("unknown branch-ledger evidence class");
    }
    if !matches!(ledger.arm_id.as_str(), "R0" | "R1" | "R2" | "R3" | "R4" | "R5") {
        bail!("unknown branch runtime arm");
    }
    if ledger.arm_id == "R5" {
        if !valid_optional_digest(&ledger.snapshot_receipt_sha256) {
            bail!("R5 live-snapshot execution requires a valid snapshot receipt digest");
        }
    } else if ledger.snapshot_receipt_sha256.is_some() {
        bail!("snapshot receipt digest is permitted only for R5");
    }
    if !is_lower_hex(&ledger.task.task_id, 64)
        || !is_lower_hex(&ledger.task.repository_commit, 40)
        || !is_lower_hex(&ledger.task.target_type_sha256, 64)
        || !is_lower_hex(&ledger.task.proof_state_sha256, 64)
    {
        bail!("task identity has a malformed digest");
    }
    if ledger.branches.is_empty() || ledger.branches.len() as u64 > ledger.budget.max_branches {
        bail!("branch list is empty or exceeds max_branches");
    }

    let mut by_id = BTreeMap::new();
    let mut workspaces = BTreeSet::new();
    let mut roots = Vec::new();
    let mut started = 0_u64;

    for branch in &ledger.branches {
        if !valid_branch_id(&branch.branch_id) {
            bail!("malformed branch ID: {}", branch.branch_id);
        }
        if by_id.insert(branch.branch_id.as_str(), branch).is_some() {
            bail!("duplicate branch ID: {}", branch.branch_id);
        }
        if branch.workspace_id.trim().is_empty()
            || !workspaces.insert(branch.workspace_id.as_str())
        {
            bail!("branch workspaces must be nonempty and unique");
        }
        if branch.workspace_isolation != "private_worktree" {
            bail!("branch {} is not isolated in a private worktree", branch.branch_id);
        }
        if branch.base_commit != ledger.task.repository_commit {
            bail!("branch {} drifted from the task commit", branch.branch_id);
        }
        if branch.proof_state_sha256 != ledger.task.proof_state_sha256 {
            bail!("branch {} drifted from the source proof state", branch.branch_id);
        }
        if branch.status != "planned" {
            started = started.checked_add(1).context("branch counter overflow")?;
        }

        match (&branch.parent_branch_id, branch.kind.as_str()) {
            (None, "root") => roots.push(branch),
            (None, _) => bail!("only a root branch may omit its parent"),
            (Some(_), "root") => bail!("root branch cannot have a parent"),
            (Some(_), _) => {}
        }

        match (branch.status.as_str(), branch.result_type.as_str()) {
            ("candidate_available", "proof_candidate") => {
                let candidate = branch
                    .candidate_sha256
                    .as_deref()
                    .context("available proof candidate has no digest")?;
                if !is_lower_hex(candidate, 64) {
                    bail!("available proof candidate has a malformed digest");
                }
            }
            (_, "proof_candidate") => {
                bail!("proof_candidate result must have candidate_available status");
            }
            ("candidate_available", _) => {
                bail!("candidate_available status must return proof_candidate");
            }
            _ => {
                if branch.candidate_sha256.is_some() {
                    bail!("candidate digest appears without an available proof candidate");
                }
            }
        }
    }

    if roots.len() != 1 {
        bail!("branch graph must contain exactly one root");
    }

    for branch in &ledger.branches {
        if let Some(parent_id) = branch.parent_branch_id.as_deref() {
            if parent_id == branch.branch_id || !by_id.contains_key(parent_id) {
                bail!("branch {} has a missing or self-referential parent", branch.branch_id);
            }
        }
        let mut visited = BTreeSet::new();
        let mut current = Some(branch.branch_id.as_str());
        while let Some(id) = current {
            if !visited.insert(id) {
                bail!("branch graph contains a cycle at {id}");
            }
            current = by_id
                .get(id)
                .and_then(|node| node.parent_branch_id.as_deref());
        }
    }

    let actual = &ledger.actual;
    let budget = &ledger.budget;
    if actual.branches_started != started || started > budget.max_branches {
        bail!("started branch count differs from branch rows or exceeds budget");
    }
    if actual.peak_active_branches > budget.max_active_branches
        || actual.peak_active_branches > actual.branches_started
    {
        bail!("active branch peak exceeds its budget or started-branch count");
    }
    if actual.peak_lean_queries > budget.max_lean_queries {
        bail!("peak concurrent Lean queries exceed budget");
    }
    if actual.peak_lean_queries > 0 && actual.branches_started == 0 {
        bail!("Lean queries are recorded without a started branch");
    }

    let additive = [
        ("tokens", actual.tokens, budget.max_tokens),
        ("model_calls", actual.model_calls, budget.max_model_calls),
        ("lean_cpu_ms", actual.lean_cpu_ms, budget.max_lean_cpu_ms),
        ("cost_microusd", actual.cost_microusd, budget.max_cost_microusd),
    ];
    for (field, actual_total, cap) in additive {
        if actual_total > cap {
            bail!("{field} exceeds its budget");
        }
        let branch_total = match field {
            "tokens" => checked_sum(
                ledger.branches.iter().map(|branch| branch.resources.tokens),
                field,
            )?,
            "model_calls" => checked_sum(
                ledger.branches.iter().map(|branch| branch.resources.model_calls),
                field,
            )?,
            "lean_cpu_ms" => checked_sum(
                ledger.branches.iter().map(|branch| branch.resources.lean_cpu_ms),
                field,
            )?,
            "cost_microusd" => checked_sum(
                ledger.branches.iter().map(|branch| branch.resources.cost_microusd),
                field,
            )?,
            _ => unreachable!("fixed additive resource list"),
        };
        if branch_total != actual_total {
            bail!("{field} run total does not equal per-branch sum");
        }
    }

    // Parallel branch durations overlap. Their sum is branch wall-work, not
    // end-to-end latency; elapsed wall time is measured independently.
    let branch_wall_work = checked_sum(
        ledger.branches.iter().map(|branch| branch.resources.wall_ms),
        "branch wall-work",
    )?;
    if branch_wall_work != actual.branch_wall_work_ms {
        bail!("branch wall-work does not equal the per-branch sum");
    }
    if branch_wall_work > budget.max_branch_wall_work_ms {
        bail!("branch wall-work exceeds budget");
    }
    if actual.wall_ms > budget.max_wall_ms {
        bail!("elapsed wall-clock exceeds max_wall_ms");
    }
    if actual.branches_started == 0 && actual.wall_ms > 0 {
        bail!("elapsed work is recorded with no branch started");
    }

    let promotion = &ledger.promotion;
    match promotion.status.as_str() {
        "not_attempted" => {
            let absent = [
                &promotion.branch_id,
                &promotion.candidate_sha256,
                &promotion.source_patch_sha256,
                &promotion.checker_id,
                &promotion.replay_verifier_id,
                &promotion.checker_candidate_sha256,
                &promotion.replay_candidate_sha256,
                &promotion.target_type_before_sha256,
                &promotion.target_type_after_sha256,
                &promotion.patch_scope,
                &promotion.checker_receipt_sha256,
                &promotion.replay_receipt_sha256,
            ];
            if absent.iter().any(|value| value.is_some())
                || promotion.kernel_check != "not_run"
                || promotion.contains_sorry_ax
                || !promotion.new_axiom_names.is_empty()
            {
                bail!("not-attempted promotion contains execution evidence");
            }
        }
        "verified_success" => {
            let branch_id = promotion
                .branch_id
                .as_deref()
                .context("successful promotion has no source branch")?;
            let branch = by_id
                .get(branch_id)
                .context("successful promotion references an unknown branch")?;
            let candidate_sha = branch
                .candidate_sha256
                .as_deref()
                .context("promoted branch has no candidate digest")?;
            for (label, candidate_digest) in [
                ("promotion candidate", &promotion.candidate_sha256),
                ("checker candidate", &promotion.checker_candidate_sha256),
                ("replay candidate", &promotion.replay_candidate_sha256),
                ("source patch", &promotion.source_patch_sha256),
                ("checker receipt", &promotion.checker_receipt_sha256),
                ("replay receipt", &promotion.replay_receipt_sha256),
                ("target type before", &promotion.target_type_before_sha256),
                ("target type after", &promotion.target_type_after_sha256),
            ] {
                if !valid_optional_digest(candidate_digest) {
                    bail!("{label} digest is missing or malformed");
                }
            }
            if branch.status != "candidate_available" || branch.result_type != "proof_candidate" {
                bail!("promoted branch did not return a proof candidate");
            }
            if promotion.candidate_sha256.as_deref() != Some(candidate_sha)
                || promotion.checker_candidate_sha256.as_deref() != Some(candidate_sha)
                || promotion.replay_candidate_sha256.as_deref() != Some(candidate_sha)
            {
                bail!("promotion, checker and replay do not bind the exact candidate bytes");
            }
            if promotion.target_type_before_sha256.as_deref()
                != Some(ledger.task.target_type_sha256.as_str())
                || promotion.target_type_after_sha256.as_deref()
                    != Some(ledger.task.target_type_sha256.as_str())
            {
                bail!("promoted theorem target differs from its original type digest");
            }
            if promotion.patch_scope.as_deref() != Some("sorry-span-only") {
                bail!("promoted patch changed code outside the selected sorry span");
            }
            if promotion.kernel_check != "passed" || promotion.contains_sorry_ax {
                bail!("promoted proof failed kernel or sorryAx acceptance");
            }
            if !promotion.new_axiom_names.is_empty() {
                bail!("promoted proof introduced an unapproved axiom");
            }
            let checker_id = promotion
                .checker_id
                .as_deref()
                .context("promotion has no checker identity")?;
            let replay_id = promotion
                .replay_verifier_id
                .as_deref()
                .context("promotion has no replay verifier identity")?;
            if checker_id.trim().is_empty()
                || replay_id.trim().is_empty()
                || checker_id == replay_id
            {
                bail!("checker and replay must be distinct verifier identities");
            }
        }
        "rejected" | "verified_failure" => {}
        _ => bail!("unknown promotion status"),
    }

    Ok(())
}

fn validate_snapshot_receipt_link(
    ledger: &Ledger,
    snapshot_receipt_path: Option<&std::path::Path>,
) -> Result<()> {
    match (ledger.arm_id.as_str(), ledger.evidence_class.as_str(), snapshot_receipt_path) {
        ("R5", "measured_run", Some(path)) => {
            let expected = ledger.snapshot_receipt_sha256
                .as_deref()
                .context("R5 measured ledger has no snapshot receipt digest")?;
            let receipt = snapshot_branch_receipt::verify_file(path, Some(expected))?;
            if receipt.run_status != "measured" {
                bail!("R5 branch ledger links to a snapshot receipt that is not measured");
            }
            if receipt.task_id != ledger.task.task_id
                || receipt.repository_commit != ledger.task.repository_commit
                || receipt.proof_state_sha256 != ledger.task.proof_state_sha256
            {
                bail!("R5 snapshot receipt task, commit or proof-state identity differs from the branch ledger");
            }
            let started: BTreeSet<&str> = ledger.branches.iter()
                .filter(|branch| branch.status != "planned")
                .map(|branch| branch.branch_id.as_str())
                .collect();
            let receipt_ids: BTreeSet<&str> =
                receipt.branch_ids.iter().map(String::as_str).collect();
            if started != receipt_ids || receipt.branch_count as usize != started.len() {
                bail!("R5 snapshot receipt branch IDs/count do not match the branch ledger's started branches");
            }
        }
        ("R5", "measured_run", None) => {
            bail!("R5 measured ledger must supply --snapshot-receipt for exact-byte verification");
        }
        ("R5", "synthetic_contract_fixture", None) => {}
        ("R5", "synthetic_contract_fixture", Some(path)) => {
            let expected = ledger.snapshot_receipt_sha256.as_deref()
                .context("R5 synthetic ledger has no snapshot receipt digest")?;
            let receipt = snapshot_branch_receipt::verify_file(path, Some(expected))?;
            if receipt.run_status != "not_attempted" {
                bail!("synthetic R5 ledger must link to a no-run snapshot receipt");
            }
        }
        (_, _, Some(_)) => {
            bail!("--snapshot-receipt is allowed only with arm R5");
        }
        (_, _, None) => {}
    }
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --validate FILE [--snapshot-receipt FILE], --self-test FILE, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    let path = PathBuf::from(args.next().context("expected a ledger file path")?);
    match action.as_str() {
        "--validate" => {
            let mut snapshot_receipt_path = None;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--snapshot-receipt" => {
                        if snapshot_receipt_path.is_some() {
                            bail!("--snapshot-receipt supplied more than once");
                        }
                        snapshot_receipt_path = Some(PathBuf::from(
                            args.next().context("missing snapshot receipt path")?
                        ));
                    }
                    _ => bail!("unknown argument: {flag}"),
                }
            }
            Ok(Mode::Validate { ledger_path: path, snapshot_receipt_path })
        }
        "--self-test" => {
            if args.next().is_some() {
                bail!("--self-test accepts exactly one ledger path");
            }
            Ok(Mode::SelfTest(path))
        }
        _ => bail!("unknown action {action}"),
    }
}

fn load(path: &PathBuf) -> Result<(Value, Ledger)> {
    let raw = fs::read(path).with_context(|| format!("read ledger {}", path.display()))?;
    let value: Value = serde_json::from_slice(&raw).context("parse branch-ledger JSON")?;
    let ledger: Ledger =
        serde_json::from_value(value.clone()).context("decode strict branch ledger")?;
    Ok((value, ledger))
}

fn synthetic_success(mut value: Value) -> Value {
    let branches = value["branches"].as_array_mut().expect("synthetic fixture has branches");
    let root = &mut branches[0];
    root["status"] = Value::from("partial_result");
    root["result_type"] = Value::from("retrieval_route");
    root["resources"]["tokens"] = Value::from(40);
    root["resources"]["model_calls"] = Value::from(1);
    root["resources"]["lean_cpu_ms"] = Value::from(20);
    root["resources"]["wall_ms"] = Value::from(100);
    root["resources"]["cost_microusd"] = Value::from(10);

    let child = &mut branches[1];
    child["status"] = Value::from("candidate_available");
    child["result_type"] = Value::from("proof_candidate");
    child["candidate_sha256"] = Value::from("a".repeat(64));
    child["resources"]["tokens"] = Value::from(60);
    child["resources"]["model_calls"] = Value::from(1);
    child["resources"]["lean_cpu_ms"] = Value::from(30);
    child["resources"]["wall_ms"] = Value::from(80);
    child["resources"]["cost_microusd"] = Value::from(20);

    value["actual"]["branches_started"] = Value::from(2);
    value["actual"]["peak_active_branches"] = Value::from(2);
    value["actual"]["peak_lean_queries"] = Value::from(1);
    value["actual"]["tokens"] = Value::from(100);
    value["actual"]["model_calls"] = Value::from(2);
    value["actual"]["lean_cpu_ms"] = Value::from(50);
    value["actual"]["wall_ms"] = Value::from(120);
    value["actual"]["branch_wall_work_ms"] = Value::from(180);
    value["actual"]["cost_microusd"] = Value::from(30);

    let target_type = value["task"]["target_type_sha256"].clone();
    let promotion = &mut value["promotion"];
    promotion["status"] = Value::from("verified_success");
    promotion["branch_id"] = Value::from("delegated-1");
    promotion["candidate_sha256"] = Value::from("a".repeat(64));
    promotion["source_patch_sha256"] = Value::from("b".repeat(64));
    promotion["checker_id"] = Value::from("lean-kernel-checker");
    promotion["replay_verifier_id"] = Value::from("independent-clean-replay");
    promotion["checker_candidate_sha256"] = Value::from("a".repeat(64));
    promotion["replay_candidate_sha256"] = Value::from("a".repeat(64));
    promotion["target_type_before_sha256"] = target_type.clone();
    promotion["target_type_after_sha256"] = target_type;
    promotion["patch_scope"] = Value::from("sorry-span-only");
    promotion["checker_receipt_sha256"] = Value::from("c".repeat(64));
    promotion["replay_receipt_sha256"] = Value::from("d".repeat(64));
    promotion["kernel_check"] = Value::from("passed");
    promotion["contains_sorry_ax"] = Value::from(false);
    promotion["new_axiom_names"] = Value::Array(Vec::new());
    value
}

fn must_reject(label: &str, value: Value) -> Result<()> {
    let ledger: Ledger = serde_json::from_value(value).context("decode negative fixture")?;
    if validate(&ledger).is_ok() {
        bail!("negative control unexpectedly accepted: {label}");
    }
    println!("PITB_SPECULATIVE_BRANCH_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let (fixture_value, fixture) = load(path)?;
    if fixture.evidence_class != "synthetic_contract_fixture" {
        bail!("self-test fixture must be explicitly synthetic");
    }
    if fixture.promotion.status != "not_attempted" {
        bail!("checked-in fixture must not claim a promotion");
    }
    validate(&fixture)?;
    println!("PITB_SPECULATIVE_BRANCH_FIXTURE_PASS: synthetic no-run ledger");

    let mut missing_r5_receipt = fixture_value.clone();
    missing_r5_receipt["arm_id"] = Value::from("R5");
    must_reject("R5 without a live-snapshot receipt", missing_r5_receipt)?;

    let mut linked_r5 = fixture_value.clone();
    linked_r5["arm_id"] = Value::from("R5");
    linked_r5["snapshot_receipt_sha256"] = Value::from("a".repeat(64));
    let linked_r5_ledger: Ledger =
        serde_json::from_value(linked_r5).context("decode synthetic R5 receipt link")?;
    validate(&linked_r5_ledger)?;
    println!("PITB_SPECULATIVE_BRANCH_R5_LINK_PASS: synthetic digest-shape only; not a measured snapshot");

    let success = synthetic_success(fixture_value);
    let success_ledger: Ledger =
        serde_json::from_value(success.clone()).context("decode synthetic promotion")?;
    validate(&success_ledger)?;
    if success_ledger.actual.branch_wall_work_ms <= success_ledger.actual.wall_ms {
        bail!("parallel wall-work test does not exceed elapsed wall time");
    }
    println!(
        concat!(
            "PITB_SPECULATIVE_BRANCH_PROMOTION_PASS: exact candidate joins; ",
            "180 branch-ms / 120 elapsed-ms"
        )
    );

    let mut bad = success.clone();
    bad["promotion"]["candidate_sha256"] = Value::from("e".repeat(64));
    must_reject("promoted candidate differs from branch artifact", bad)?;

    let mut bad = success.clone();
    bad["promotion"]["replay_candidate_sha256"] = Value::from("e".repeat(64));
    must_reject("replay checked different candidate bytes", bad)?;

    let mut bad = success.clone();
    let replay_identity = bad["promotion"]["replay_verifier_id"].clone();
    bad["promotion"]["checker_id"] = replay_identity;
    must_reject("checker and replay identities are not independent", bad)?;

    let mut bad = success.clone();
    bad["promotion"]["target_type_after_sha256"] = Value::from("e".repeat(64));
    must_reject("promoted theorem type changed", bad)?;

    let mut bad = success.clone();
    bad["promotion"]["patch_scope"] = Value::from("outside-target-span");
    must_reject("patch escaped the sorry span", bad)?;

    let mut bad = success.clone();
    bad["promotion"]["new_axiom_names"] = serde_json::json!(["Unsafe.assumption"]);
    must_reject("unapproved axiom in promoted proof", bad)?;

    let mut bad = success.clone();
    bad["actual"]["wall_ms"] = Value::from(240_001);
    must_reject("elapsed wall-clock budget", bad)?;

    let mut bad = success;
    bad["actual"]["branch_wall_work_ms"] = Value::from(179);
    must_reject("per-branch wall-work accounting", bad)?;
    Ok(())
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: analyze_sorrydb_branch_ledger --validate FILE [--snapshot-receipt FILE] | --self-test FILE");
            Ok(())
        }
        Mode::Validate { ledger_path, snapshot_receipt_path } => {
            let (_, ledger) = load(&ledger_path)?;
            validate(&ledger)?;
            validate_snapshot_receipt_link(&ledger, snapshot_receipt_path.as_deref())?;
            println!(
                concat!(
                    "PITB_SPECULATIVE_BRANCH_LEDGER_PASS: branches={} started={} ",
                    "tokens={} model_calls={} lean_cpu_ms={} elapsed_wall_ms={} ",
                    "branch_wall_work_ms={} promotion={}"
                ),
                ledger.branches.len(),
                ledger.actual.branches_started,
                ledger.actual.tokens,
                ledger.actual.model_calls,
                ledger.actual.lean_cpu_ms,
                ledger.actual.wall_ms,
                ledger.actual.branch_wall_work_ms,
                ledger.promotion.status
            );
            Ok(())
        }
        Mode::SelfTest(path) => self_test(&path),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_SPECULATIVE_BRANCH_LEDGER_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_ids_are_lowercase_kebab_case() {
        assert!(valid_branch_id("branch-1"));
        assert!(!valid_branch_id("../branch"));
        assert!(!valid_branch_id("UPPER"));
    }

    #[test]
    fn checked_resource_sum_detects_overflow() {
        assert!(checked_sum([u64::MAX, 1], "test").is_err());
    }

    #[test]
    fn digest_validation_is_lowercase_and_fixed_width() {
        assert!(is_lower_hex(&"a".repeat(64), 64));
        assert!(!is_lower_hex(&"A".repeat(64), 64));
        assert!(!is_lower_hex(&"g".repeat(64), 64));
    }
}
