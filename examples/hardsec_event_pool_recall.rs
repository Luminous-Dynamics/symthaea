//! Compare sampled trace-pool events with an independently enumerated finite FSM.
//!
//! The reference set is complete only for the declared finite transition system
//! and its declared initial states. It is not a claim about an external circuit's
//! complete state space.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct Transition {
    id: String,
    event_id: String,
    from: String,
    input: String,
    to: String,
}

#[derive(Debug, Clone)]
struct Trace {
    id: String,
    initial_state: String,
    steps: Vec<String>,
}

#[derive(Debug, Clone)]
struct Args {
    spec: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone)]
struct CoverageSummary {
    spec_sha256: String,
    fixture_id: String,
    reference_state_count: usize,
    reference_transition_ids: BTreeSet<String>,
    reference_event_ids: BTreeSet<String>,
    observed_transition_ids: BTreeSet<String>,
    observed_event_ids: BTreeSet<String>,
    unreachable_transition_ids: BTreeSet<String>,
    unreachable_event_ids: BTreeSet<String>,
    trace_count: usize,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut spec = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--spec" => spec = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        spec: spec.ok_or("--spec is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let result = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if result.trim().is_empty() || result.trim() != result {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace"));
    }
    Ok(result)
}

fn required_u64(value: &Value, field: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be a non-negative integer"))
}

fn string_array(value: &Value, field: &str) -> Result<Vec<String>, String> {
    let entries = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?;
    let mut result = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let item = entry
            .as_str()
            .ok_or_else(|| format!("{field}[{index}] must be a string"))?;
        if item.trim().is_empty() || item.trim() != item {
            return Err(format!("{field}[{index}] must be a non-empty string"));
        }
        result.push(item.to_owned());
    }
    Ok(result)
}

fn unique_string_array(value: &Value, field: &str) -> Result<Vec<String>, String> {
    let entries = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?;
    let mut seen = BTreeSet::new();
    let mut result = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let item = entry
            .as_str()
            .ok_or_else(|| format!("{field}[{index}] must be a string"))?;
        if item.trim().is_empty() || item.trim() != item {
            return Err(format!("{field}[{index}] must be a non-empty string"));
        }
        if !seen.insert(item.to_owned()) {
            return Err(format!("{field} contains duplicate item {item}"));
        }
        result.push(item.to_owned());
    }
    Ok(result)
}

fn parse_transitions(
    value: &Value,
    states: &BTreeSet<String>,
) -> Result<(Vec<Transition>, BTreeMap<String, Transition>, BTreeMap<String, Vec<Transition>>), String> {
    let raw = value
        .get("transitions")
        .and_then(Value::as_array)
        .ok_or("transitions must be an array")?;
    if raw.is_empty() {
        return Err("transitions must not be empty".to_owned());
    }

    let mut transitions = Vec::with_capacity(raw.len());
    let mut by_id = BTreeMap::new();
    let mut by_state = BTreeMap::<String, Vec<Transition>>::new();
    let mut state_input_pairs = BTreeSet::new();

    for (index, item) in raw.iter().enumerate() {
        let id = required_string(item, "transition_id")?.to_owned();
        let event_id = required_string(item, "event_id")?.to_owned();
        let from = required_string(item, "from")?.to_owned();
        let input = required_string(item, "input")?.to_owned();
        let to = required_string(item, "to")?.to_owned();

        if !states.contains(&from) {
            return Err(format!("transition {id}: unknown source state {from}"));
        }
        if !states.contains(&to) {
            return Err(format!("transition {id}: unknown destination state {to}"));
        }
        if !state_input_pairs.insert((from.clone(), input.clone())) {
            return Err(format!(
                "transition {id}: nondeterministic duplicate (state,input)=({from},{input})"
            ));
        }
        if by_id.contains_key(&id) {
            return Err(format!("duplicate transition_id {id}"));
        }

        let transition = Transition {
            id: id.clone(),
            event_id,
            from: from.clone(),
            input,
            to,
        };
        by_state
            .entry(from)
            .or_default()
            .push(transition.clone());
        by_id.insert(id, transition.clone());
        transitions.push(transition);

        if index > 1_000_000 {
            return Err("transition count exceeds the one-million fixture safety limit".to_owned());
        }
    }

    for outgoing in by_state.values_mut() {
        outgoing.sort_by(|left, right| left.id.cmp(&right.id));
    }
    transitions.sort_by(|left, right| left.id.cmp(&right.id));
    Ok((transitions, by_id, by_state))
}

fn parse_traces(
    value: &Value,
    initial_states: &BTreeSet<String>,
) -> Result<Vec<Trace>, String> {
    let raw = value
        .get("trace_pool")
        .and_then(Value::as_array)
        .ok_or("trace_pool must be an array")?;
    let mut traces = Vec::with_capacity(raw.len());
    let mut trace_ids = BTreeSet::new();

    for item in raw {
        let id = required_string(item, "trace_id")?.to_owned();
        if !trace_ids.insert(id.clone()) {
            return Err(format!("duplicate trace_id {id}"));
        }
        let initial_state = required_string(item, "initial_state")?.to_owned();
        if !initial_states.contains(&initial_state) {
            return Err(format!(
                "trace {id}: initial_state {initial_state} is not declared as an allowed initial state"
            ));
        }
        let steps = string_array(item, "steps")?;
        traces.push(Trace {
            id,
            initial_state,
            steps,
        });
    }
    traces.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(traces)
}

fn enumerate_reference_events(
    initial_states: &BTreeSet<String>,
    by_state: &BTreeMap<String, Vec<Transition>>,
    state_limit: usize,
) -> Result<(BTreeSet<String>, BTreeSet<String>, usize), String> {
    if state_limit == 0 {
        return Err("max_states must be greater than zero".to_owned());
    }

    let mut queue = VecDeque::new();
    let mut visited_states = BTreeSet::new();
    let mut reachable_transition_ids = BTreeSet::new();
    let mut reachable_event_ids = BTreeSet::new();

    for state in initial_states {
        queue.push_back(state.clone());
    }

    while let Some(state) = queue.pop_front() {
        if !visited_states.insert(state.clone()) {
            continue;
        }
        if visited_states.len() > state_limit {
            return Err(format!(
                "reference enumeration exceeded max_states={state_limit}; completeness not established"
            ));
        }

        if let Some(outgoing) = by_state.get(&state) {
            for transition in outgoing {
                reachable_transition_ids.insert(transition.id.clone());
                reachable_event_ids.insert(transition.event_id.clone());
                if !visited_states.contains(&transition.to) {
                    queue.push_back(transition.to.clone());
                }
            }
        }
    }

    if reachable_transition_ids.is_empty() || reachable_event_ids.is_empty() {
        return Err("declared FSM has no reachable transitions/events".to_owned());
    }

    Ok((
        reachable_transition_ids,
        reachable_event_ids,
        visited_states.len(),
    ))
}

fn replay_trace_pool(
    traces: &[Trace],
    transitions_by_id: &BTreeMap<String, Transition>,
) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    let mut observed_transition_ids = BTreeSet::new();
    let mut observed_event_ids = BTreeSet::new();

    for trace in traces {
        let mut current_state = trace.initial_state.clone();
        for (index, transition_id) in trace.steps.iter().enumerate() {
            let transition = transitions_by_id.get(transition_id).ok_or_else(|| {
                format!(
                    "trace {} step {index}: unknown transition_id {transition_id}",
                    trace.id
                )
            })?;
            if transition.from != current_state {
                return Err(format!(
                    "trace {} step {index}: transition {} starts at {}, expected current state {}",
                    trace.id, transition.id, transition.from, current_state
                ));
            }

            observed_transition_ids.insert(transition.id.clone());
            observed_event_ids.insert(transition.event_id.clone());
            current_state = transition.to.clone();
        }
    }

    Ok((observed_transition_ids, observed_event_ids))
}

fn validate_spec(value: &Value, spec_sha256: String) -> Result<CoverageSummary, String> {
    if required_string(value, "schema_version")? != "hardsec-fsm-event-pool-v1" {
        return Err("unsupported schema_version".to_owned());
    }

    let fixture_id = required_string(value, "fixture_id")?.to_owned();
    let state_vec = unique_string_array(value, "states")?;
    if state_vec.is_empty() {
        return Err("states must not be empty".to_owned());
    }
    let states = state_vec.into_iter().collect::<BTreeSet<_>>();
    let initial_vec = unique_string_array(value, "initial_states")?;
    if initial_vec.is_empty() {
        return Err("initial_states must not be empty".to_owned());
    }
    let initial_states = initial_vec.into_iter().collect::<BTreeSet<_>>();
    for state in &initial_states {
        if !states.contains(state) {
            return Err(format!("unknown initial state {state}"));
        }
    }

    let state_limit = required_u64(value, "max_states")?;
    let state_limit = usize::try_from(state_limit)
        .map_err(|_| "max_states exceeds platform usize")?;

    let (_transitions, transitions_by_id, transitions_by_state) =
        parse_transitions(value, &states)?;
    let traces = parse_traces(value, &initial_states)?;

    let (reference_transition_ids, reference_event_ids, reference_state_count) =
        enumerate_reference_events(&initial_states, &transitions_by_state, state_limit)?;
    let (observed_transition_ids, observed_event_ids) =
        replay_trace_pool(&traces, &transitions_by_id)?;

    if !observed_transition_ids.is_subset(&reference_transition_ids) {
        return Err(
            "trace pool contains a transition not reachable from the declared initial states"
                .to_owned(),
        );
    }
    if !observed_event_ids.is_subset(&reference_event_ids) {
        return Err(
            "trace pool contains an event outside the enumerated reachable event set".to_owned(),
        );
    }

    let all_transition_ids = transitions_by_id.keys().cloned().collect::<BTreeSet<_>>();
    let all_event_ids = transitions_by_id
        .values()
        .map(|transition| transition.event_id.clone())
        .collect::<BTreeSet<_>>();
    let unreachable_transition_ids = all_transition_ids
        .difference(&reference_transition_ids)
        .cloned()
        .collect::<BTreeSet<_>>();
    let unreachable_event_ids = all_event_ids
        .difference(&reference_event_ids)
        .cloned()
        .collect::<BTreeSet<_>>();

    Ok(CoverageSummary {
        spec_sha256,
        fixture_id,
        reference_state_count,
        reference_transition_ids,
        reference_event_ids,
        observed_transition_ids,
        observed_event_ids,
        unreachable_transition_ids,
        unreachable_event_ids,
        trace_count: traces.len(),
    })
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

fn receipt(summary: &CoverageSummary) -> Value {
    let missed_event_ids = summary
        .reference_event_ids
        .difference(&summary.observed_event_ids)
        .cloned()
        .collect::<Vec<_>>();
    let missed_transition_ids = summary
        .reference_transition_ids
        .difference(&summary.observed_transition_ids)
        .cloned()
        .collect::<Vec<_>>();

    json!({
        "schema_version": "hardsec-event-pool-recall-receipt-v1",
        "result": "pass",
        "claim_scope": "complete_reachable_event_enumeration_for_declared_finite_fsm_only",
        "fixture_id": summary.fixture_id,
        "spec_sha256": summary.spec_sha256,
        "reference_state_count": summary.reference_state_count,
        "trace_count": summary.trace_count,
        "reference_event_count": summary.reference_event_ids.len(),
        "observed_event_count": summary.observed_event_ids.len(),
        "event_pool_recall": ratio(
            summary.observed_event_ids.len(),
            summary.reference_event_ids.len()
        ),
        "reference_event_ids": summary.reference_event_ids,
        "observed_event_ids": summary.observed_event_ids,
        "missed_event_ids": missed_event_ids,
        "reference_transition_count": summary.reference_transition_ids.len(),
        "observed_transition_count": summary.observed_transition_ids.len(),
        "transition_recall": ratio(
            summary.observed_transition_ids.len(),
            summary.reference_transition_ids.len()
        ),
        "missed_transition_ids": missed_transition_ids,
        "unreachable_transition_ids_excluded_from_denominator": summary.unreachable_transition_ids,
        "unreachable_event_ids_excluded_from_denominator": summary.unreachable_event_ids,
        "coverage_interpretation": "exact unique-event recall against the complete reachable transition graph of this declared finite FSM",
        "security_verdict": null,
        "limitations": [
            "Completeness applies only to the declared finite FSM and its declared initial states.",
            "Trace-pool recall is not physical hardware trigger coverage or reachable-state coverage of a real circuit.",
            "This prototype counts unique transition event identifiers; it does not yet measure temporal path-sequence recall or state-cube extraction loss."
        ]
    })
}

fn receipt_target(spec_path: &Path, receipt_path: &Path) -> Result<PathBuf, String> {
    let spec_parent = spec_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let spec_parent = fs::canonicalize(spec_parent)
        .map_err(|error| format!("canonicalize spec directory: {error}"))?;
    let receipt_parent = receipt_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let receipt_parent = fs::canonicalize(receipt_parent)
        .map_err(|error| format!("canonicalize receipt directory: {error}"))?;
    let filename = receipt_path
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let target = receipt_parent.join(filename);
    if target.starts_with(&spec_parent) {
        return Err("receipt must be outside the spec directory".to_owned());
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.spec)
        .map_err(|error| format!("stat spec {}: {error}", args.spec.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("spec must be a regular non-symlink file".to_owned());
    }
    let bytes = fs::read(&args.spec)
        .map_err(|error| format!("read spec {}: {error}", args.spec.display()))?;
    let digest = sha256_bytes(&bytes);
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse spec JSON: {error}"))?;
    let summary = validate_spec(&value, digest)?;
    let receipt_value = receipt(&summary);
    let output_path = receipt_target(&args.spec, &args.receipt)?;
    let serialized = serde_json::to_vec_pretty(&receipt_value)
        .map_err(|error| format!("serialize receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .map_err(|error| format!("create receipt {}: {error}", output_path.display()))?;
    let result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = result {
        let _ = fs::remove_file(&output_path);
        return Err(format!("write receipt {}: {error}", output_path.display()));
    }

    println!("HARDSEC event-pool recall audit: PASS");
    println!("fixture: {}", summary.fixture_id);
    println!(
        "events observed/reference: {}/{}",
        summary.observed_event_ids.len(),
        summary.reference_event_ids.len()
    );
    println!(
        "event-pool recall: {:.6}",
        ratio(
            summary.observed_event_ids.len(),
            summary.reference_event_ids.len()
        )
    );
    println!("receipt: {}", output_path.display());
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC event-pool recall audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!("usage: hardsec_event_pool_recall --spec FILE --receipt NEW_FILE");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("HARDSEC event-pool recall audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/hardsec/finite-event-pool-fsm.json"))
            .expect("valid FSM fixture JSON")
    }

    #[test]
    fn complete_reference_enumeration_includes_all_reachable_events() {
        let summary = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        assert_eq!(summary.reference_state_count, 5);
        assert_eq!(summary.reference_event_ids.len(), 6);
        assert!(!summary.reference_event_ids.contains("event_unreachable"));
        assert_eq!(summary.unreachable_transition_ids.len(), 1);
    }

    #[test]
    fn sampled_trace_pool_recall_exposes_missing_rare_events() {
        let summary = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        assert_eq!(summary.observed_event_ids.len(), 4);
        assert_eq!(summary.reference_event_ids.len(), 6);
        let receipt = receipt(&summary);
        assert_eq!(receipt["event_pool_recall"], json!(4.0 / 6.0));
        assert_eq!(
            receipt["missed_event_ids"],
            json!(["event_rare_completion", "event_rare_prefix"])
        );
    }

    #[test]
    fn duplicate_trace_observations_do_not_inflate_unique_event_recall() {
        let mut value = fixture();
        value["trace_pool"][0]["steps"] = json!([
            "t_reset_to_a",
            "t_a_idle",
            "t_a_idle",
            "t_a_idle",
            "t_a_to_done",
            "t_done_to_sink"
        ]);
        let summary = validate_spec(&value, "0".repeat(64)).unwrap();
        assert_eq!(summary.observed_event_ids.len(), 4);
    }

    #[test]
    fn unreachable_transitions_are_excluded_from_the_reference_denominator() {
        let summary = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        assert!(
            !summary
                .reference_transition_ids
                .contains("t_unreachable_to_sink")
        );
        assert!(
            summary
                .unreachable_transition_ids
                .contains("t_unreachable_to_sink")
        );
    }

    #[test]
    fn empty_trace_pool_has_zero_event_recall() {
        let mut value = fixture();
        value["trace_pool"] = json!([]);
        let summary = validate_spec(&value, "0".repeat(64)).unwrap();
        assert!(summary.observed_event_ids.is_empty());
        assert_eq!(ratio(0, summary.reference_event_ids.len()), 0.0);
    }

    #[test]
    fn rejects_trace_transition_out_of_order() {
        let mut value = fixture();
        value["trace_pool"][0]["steps"] = json!(["t_a_to_done"]);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("starts at")
        );
    }

    #[test]
    fn rejects_unknown_transition_in_trace() {
        let mut value = fixture();
        value["trace_pool"][0]["steps"] = json!(["missing-transition"]);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("unknown transition_id")
        );
    }

    #[test]
    fn rejects_duplicate_transition_ids() {
        let mut value = fixture();
        let duplicate = value["transitions"][0].clone();
        value["transitions"].as_array_mut().unwrap().push(duplicate);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("duplicate transition_id")
        );
    }

    #[test]
    fn rejects_nondeterministic_duplicate_state_input_pairs() {
        let mut value = fixture();
        value["transitions"].as_array_mut().unwrap().push(json!({
            "transition_id":"t_conflict",
            "event_id":"event_conflict",
            "from":"reset",
            "input":"a",
            "to":"done"
        }));
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("nondeterministic duplicate")
        );
    }

    #[test]
    fn rejects_initial_state_not_in_state_set() {
        let mut value = fixture();
        value["initial_states"] = json!(["not-a-state"]);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("unknown initial state")
        );
    }

    #[test]
    fn rejects_trace_initial_state_not_declared_as_allowed() {
        let mut value = fixture();
        value["trace_pool"][0]["initial_state"] = json!("state_a");
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("not declared as an allowed initial state")
        );
    }

    #[test]
    fn rejects_exceeded_reference_state_limit_without_claiming_completeness() {
        let mut value = fixture();
        value["max_states"] = json!(2);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("completeness not established")
        );
    }

    #[test]
    fn receipt_is_deterministic_for_the_same_spec() {
        let summary_a = validate_spec(&fixture(), "a".repeat(64)).unwrap();
        let summary_b = validate_spec(&fixture(), "a".repeat(64)).unwrap();
        assert_eq!(receipt(&summary_a), receipt(&summary_b));
    }
}
