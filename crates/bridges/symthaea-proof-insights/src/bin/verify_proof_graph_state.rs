#[path = "../proof_graph.rs"]
mod proof_graph;

use std::{
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

enum Mode {
    Help,
    Verify(PathBuf),
    SelfTest(PathBuf),
}

fn must_reject(label: &str, value: Value) -> Result<()> {
    if proof_graph::validate_value(&value).is_ok() {
        bail!("proof graph verifier accepted negative control: {label}");
    }
    println!("PITB_PROOF_GRAPH_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read proof graph fixture {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes).context("parse proof graph fixture")?;
    let summary = proof_graph::validate_value(&value)?;
    if summary.evidence_class != "synthetic_contract_fixture"
        || summary.declaration_count != 3
        || summary.goal_count != 3
        || summary.proof_step_count != 1
        || summary.all_roots_stable
        || summary.ready_frontier_goal_ids != vec![
            "goal-target-root".to_owned(),
            "goal-target-secondary".to_owned()
        ]
        || summary.selected_goal_ids != summary.ready_frontier_goal_ids
    {
        bail!("proof graph fixture does not demonstrate two roots sharing one stable helper");
    }
    println!("PITB_PROOF_GRAPH_FIXTURE_PASS: two target roots share one stable helper; two ready goals batch into one specialist action");

    let mut bad = value.clone();
    bad["scheduler_decision"]["selected_goal_ids"] = json!(["goal-target-root"]);
    must_reject("scheduler omits a goal from the maximal shared-helper batch", bad)?;

    let mut capped = value.clone();
    capped["scheduler_decision"]["max_selected_goals"] = json!(1);
    capped["scheduler_decision"]["selected_goal_ids"] = json!(["goal-target-root"]);
    let capped_summary = proof_graph::validate_value(&capped)?;
    if capped_summary.selected_goal_ids != vec!["goal-target-root".to_owned()] {
        bail!("scheduler did not apply the registered deterministic batch cap");
    }
    println!("PITB_PROOF_GRAPH_BATCH_CAP_PASS: lexical deterministic truncation under a one-goal cap");

    let mut provisional = value.clone();
    for declaration in provisional["declarations"].as_array_mut().context("declarations array")? {
        if declaration["declaration_id"] == "helper-normalize" {
            declaration["lifecycle_state"] = json!("provisional");
            declaration["proof_receipt_sha256"] = Value::Null;
            declaration["kernel_verified"] = Value::Bool(false);
        }
    }
    for goal in provisional["goal_nodes"].as_array_mut().context("goal nodes array")? {
        if goal["goal_id"] == "goal-helper-root" {
            goal["status"] = json!("open");
            goal["closure_proof_sha256"] = Value::Null;
            goal["closure_receipt_sha256"] = Value::Null;
            goal["closure_kernel_verified"] = Value::Bool(false);
            goal["closure_contains_sorry_ax"] = Value::Bool(false);
        }
    }
    provisional["proof_steps"][0]["status"] = json!("candidate");
    provisional["proof_steps"][0]["transition_checker_id"] = Value::Null;
    provisional["proof_steps"][0]["transition_receipt_sha256"] = Value::Null;
    provisional["proof_steps"][0]["terminal_proof_sha256"] = Value::Null;
    provisional["proof_steps"][0]["terminal_receipt_sha256"] = Value::Null;
    provisional["proof_steps"][0]["terminal_kernel_verified"] = Value::Bool(false);
    provisional["proof_steps"][0]["terminal_contains_sorry_ax"] = Value::Bool(false);
    provisional["scheduler_decision"]["ready_frontier_goal_ids"] = json!(["goal-helper-root"]);
    provisional["scheduler_decision"]["selected_goal_ids"] = json!(["goal-helper-root"]);
    let provisional_summary = proof_graph::validate_value(&provisional)?;
    if provisional_summary.ready_frontier_goal_ids != vec!["goal-helper-root".to_owned()]
        || provisional_summary.selected_goal_ids != vec!["goal-helper-root".to_owned()]
    {
        bail!("goals depending on a provisional helper entered the ready frontier");
    }
    println!("PITB_PROOF_GRAPH_PREREQUISITE_PASS: provisional helper is scheduled before its dependents");

    let mut stopped = value.clone();
    for goal in stopped["goal_nodes"].as_array_mut().context("goal nodes array")? {
        if goal["status"] == "open" {
            goal["status"] = json!("discarded");
            goal["closure_proof_sha256"] = Value::Null;
            goal["closure_receipt_sha256"] = Value::Null;
            goal["closure_kernel_verified"] = Value::Bool(false);
            goal["closure_contains_sorry_ax"] = Value::Bool(false);
        }
    }
    stopped["scheduler_decision"]["ready_frontier_goal_ids"] = json!([]);
    stopped["scheduler_decision"]["selected_goal_ids"] = json!([]);
    stopped["scheduler_decision"]["specialist"] = json!("stop");
    stopped["scheduler_decision"]["reasoning_effort"] = json!("none");
    stopped["scheduler_decision"]["max_specialist_calls"] = json!(0);
    stopped["resource_ledger"]["actual"]["router_model_calls"] = json!(0);
    stopped["resource_ledger"]["actual"]["specialist_model_calls"] = json!(0);
    stopped["resource_ledger"]["actual"]["model_calls"] = json!(0);
    stopped["resource_ledger"]["actual"]["router_cost_microusd"] = json!(0);
    stopped["resource_ledger"]["actual"]["specialist_cost_microusd"] = json!(0);
    stopped["resource_ledger"]["actual"]["cost_microusd"] =
        stopped["resource_ledger"]["actual"]["lean_cost_microusd"].clone();
    let stop_summary = proof_graph::validate_value(&stopped)?;
    if !stop_summary.ready_frontier_goal_ids.is_empty() || !stop_summary.selected_goal_ids.is_empty() {
        bail!("scheduler did not stop with an empty dependency-ready frontier");
    }
    println!("PITB_PROOF_GRAPH_STOP_PASS: no work invented when the ready frontier is empty");

    let mut bad = stopped.clone();
    bad["scheduler_decision"]["specialist"] = json!("proof_writer");
    bad["scheduler_decision"]["reasoning_effort"] = json!("medium");
    bad["scheduler_decision"]["max_specialist_calls"] = json!(1);
    bad["resource_ledger"]["actual"]["router_model_calls"] = json!(1);
    bad["resource_ledger"]["actual"]["specialist_model_calls"] = json!(1);
    bad["resource_ledger"]["actual"]["model_calls"] = json!(2);
    must_reject("model specialist invoked with no ready goals", bad)?;

    let mut bad = value.clone();
    bad["declarations"][2]["lifecycle_state"] = json!("provisional");
    bad["declarations"][2]["proof_receipt_sha256"] = Value::Null;
    bad["declarations"][2]["kernel_verified"] = Value::Bool(false);
    bad["declarations"][2]["dependency_ids"] = json!(["target-main"]);
    must_reject("cross-declaration helper dependency cycle", bad)?;

    let mut bad = value.clone();
    bad["proof_steps"][0]["goal_state_sha256"] = json!("0".repeat(64));
    must_reject("stale proof-step goal-state identity", bad)?;

    let mut bad = value.clone();
    bad["proof_steps"][0]["subgoal_ids"] = json!(["missing-goal"]);
    bad["proof_steps"][0]["terminal_proof_sha256"] = Value::Null;
    bad["proof_steps"][0]["terminal_receipt_sha256"] = Value::Null;
    bad["proof_steps"][0]["terminal_kernel_verified"] = Value::Bool(false);
    must_reject("dangling AND-hyperedge subgoal", bad)?;

    let mut bad = value.clone();
    bad["goal_nodes"][0]["status"] = json!("closed");
    bad["goal_nodes"][0]["closure_proof_sha256"] = Value::Null;
    bad["goal_nodes"][0]["closure_receipt_sha256"] = Value::Null;
    bad["goal_nodes"][0]["closure_kernel_verified"] = Value::Bool(false);
    must_reject("closed goal without kernel proof and receipt", bad)?;

    let mut bad = value.clone();
    bad["scheduler_decision"]["ready_frontier_goal_ids"] = json!(["goal-target-root"]);
    must_reject("scheduler omits a dependency-ready goal", bad)?;

    let mut bad = value.clone();
    bad["scheduler_decision"]["selected_goal_ids"] = json!(["goal-helper-root"]);
    must_reject("scheduler selects a closed goal outside the ready frontier", bad)?;

    let mut bad = value.clone();
    bad["declarations"][2]["lifecycle_state"] = json!("discarded");
    bad["declarations"][2]["proof_receipt_sha256"] = Value::Null;
    bad["declarations"][2]["kernel_verified"] = Value::Bool(false);
    must_reject("active target still depends on a discarded helper", bad)?;

    let mut bad = value.clone();
    bad["resource_ledger"]["budget"]["model_calls"] = json!(1);
    must_reject("router plus specialist model-call budget overrun", bad)?;

    let mut bad = value.clone();
    bad["resource_ledger"]["actual"]["cost_microusd"] = json!(1);
    must_reject("router/specialist/Lean cost ledger does not reconcile", bad)?;

    let mut bad = value.clone();
    bad["scheduler_decision"]["specialist"] = json!("deterministic_automation");
    bad["scheduler_decision"]["reasoning_effort"] = json!("none");
    bad["scheduler_decision"]["max_specialist_calls"] = json!(0);
    must_reject("deterministic automation claims specialist model calls", bad)?;

    let mut bad = value.clone();
    bad["declarations"][0]["lifecycle_state"] = json!("stable");
    bad["declarations"][0]["proof_receipt_sha256"] = json!("1".repeat(64));
    bad["declarations"][0]["kernel_verified"] = Value::Bool(true);
    must_reject("stable target depends on unfinished goal", bad)?;

    println!("PITB_PROOF_GRAPH_SELF_TEST_PASS");
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --verify FILE, --self-test FILE, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    let path = PathBuf::from(args.next().context("missing proof graph state path")?);
    if args.next().is_some() {
        bail!("too many command arguments");
    }
    match action.as_str() {
        "--verify" => Ok(Mode::Verify(path)),
        "--self-test" => Ok(Mode::SelfTest(path)),
        _ => bail!("unknown action {action}"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: verify_proof_graph_state --verify FILE | --self-test FILE");
            Ok(())
        }
        Mode::SelfTest(path) => self_test(&path),
        Mode::Verify(path) => {
            let bytes = fs::read(&path).with_context(|| format!("read proof graph state {}", path.display()))?;
            let value: Value = serde_json::from_slice(&bytes).context("parse proof graph state")?;
            let summary = proof_graph::validate_value(&value)?;
            println!(
                "PITB_PROOF_GRAPH_PASS: task={} declarations={} goals={} steps={} ready={} selected={} all_roots_stable={}",
                summary.task_id,
                summary.declaration_count,
                summary.goal_count,
                summary.proof_step_count,
                summary.ready_frontier_goal_ids.len(),
                summary.selected_goal_ids.len(),
                summary.all_roots_stable
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_PROOF_GRAPH_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
