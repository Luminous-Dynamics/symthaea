//! Validates per-case evidence ordering, completeness and tamper-evident chaining.
//!
//! This is a log-integrity/semantic-consistency checker, not an attestation
//! system: a hash chain alone cannot prove event truth or prevent its creator
//! from recomputing an unanchored chain.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const BENCHMARK_ID: &str = "symthaea.authorization_evidence_ledger.smoke.v0";
const VERSION: u32 = 0;
const EXPECTED_CASE_SET_ID: &str = "symthaea.authorization_evidence_ledger.expected_case_set.v0";
const CASE_SET_HASH_ALGORITHM: &str =
    "SHA-256 over sorted UTF-8 case IDs, each prefixed by an unsigned 64-bit big-endian byte length";
const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ledger {
    benchmark_id: String,
    version: u32,
    status: String,
    run_id: String,
    expected_case_set_sha256: String,
    events: Vec<Event>,
    non_claims: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    sequence: u64,
    case_id: String,
    event_type: String,
    decision: String,
    terminal_class: String,
    adapter_calls: u64,
    observed_effects: u64,
    postcondition: String,
    receipt: String,
    previous_hash: String,
    event_hash: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedCaseSet {
    benchmark_id: String,
    version: u32,
    expected_case_ids: Vec<String>,
    case_set_hash_algorithm: String,
}

#[derive(Debug, Default)]
struct CaseState {
    started: bool,
    decision: Option<String>,
    adapter_calls: u64,
    observed_effects: u64,
    postcondition: Option<String>,
    receipt: Option<String>,
    outcome_unknown: bool,
    terminal_class: Option<String>,
}

#[derive(Debug)]
struct Summary {
    events: usize,
    cases: usize,
    adapter_calls: u64,
    observed_effects: u64,
    confirmed: u64,
    failed: u64,
    unknown: u64,
    receipt_rejections: u64,
    sha256: String,
}

pub fn check(ledger_path: &Path, expected_cases_path: &Path) -> Result<()> {
    let ledger_raw = fs::read(ledger_path)
        .with_context(|| format!("read evidence ledger {}", ledger_path.display()))?;
    let expected_raw = fs::read(expected_cases_path)
        .with_context(|| format!("read expected case set {}", expected_cases_path.display()))?;
    let summary = parse_and_validate(&ledger_raw, &expected_raw)?;

    let ledger: Ledger = serde_json::from_slice(&ledger_raw)?;
    println!("benchmark_id={BENCHMARK_ID}");
    println!("run_id={}", ledger.run_id);
    println!("ledger_sha256={}", summary.sha256);
    println!("event_count={}", summary.events);
    println!("expected_case_count={}", summary.cases);
    println!("adapter_invocation_count={}", summary.adapter_calls);
    println!("observed_effect_count={}", summary.observed_effects);
    println!("postcondition_confirmed_count={}", summary.confirmed);
    println!("postcondition_failed_count={}", summary.failed);
    println!("outcome_unknown_count={}", summary.unknown);
    println!("receipt_rejection_count={}", summary.receipt_rejections);
    println!("ledger_integrity=PASS");
    println!("event_semantics=PASS");
    println!("external_attestation=NOT_PROVIDED");
    println!("security_qualification=NOT_EVALUATED");
    if ledger.status == "synthetic_fixture" {
        println!("execution_status=SYNTHETIC_FIXTURE_NOT_A_REAL_RUN");
    } else {
        println!("execution_status=RECORDED_RUN_NOT_EXTERNALLY_ATTESTED");
    }
    Ok(())
}

fn parse_and_validate(ledger_raw: &[u8], expected_raw: &[u8]) -> Result<Summary> {
    let ledger: Ledger =
        serde_json::from_slice(ledger_raw).context("parse evidence ledger JSON")?;
    let expected: ExpectedCaseSet =
        serde_json::from_slice(expected_raw).context("parse expected case-set JSON")?;
    let digest = format!("{:x}", Sha256::digest(ledger_raw));
    validate(&ledger, &expected, digest)
}

fn validate(ledger: &Ledger, expected: &ExpectedCaseSet, sha256: String) -> Result<Summary> {
    if ledger.benchmark_id != BENCHMARK_ID || ledger.version != VERSION {
        bail!("unsupported evidence-ledger identity/version");
    }
    if expected.benchmark_id != EXPECTED_CASE_SET_ID || expected.version != VERSION {
        bail!("unsupported expected-case-set identity/version");
    }
    if ledger.run_id.trim().is_empty() {
        bail!("run_id must not be empty");
    }
    if !["synthetic_fixture", "completed"].contains(&ledger.status.as_str()) {
        bail!("unsupported run status {:?}; incomplete runs cannot validate", ledger.status);
    }
    if ledger.non_claims.is_empty()
        || !ledger.non_claims.iter().any(|v| v.contains("does not prove the underlying events are true"))
    {
        bail!("ledger must preserve explicit non-claims about hash-chain limits");
    }
    if ledger.status == "synthetic_fixture"
        && !ledger.non_claims.iter().any(|v| v.contains("not a model or runtime evaluation"))
    {
        bail!("synthetic fixture must explicitly disclaim model/runtime evaluation");
    }
    if ledger.status == "completed"
        && !ledger.non_claims.iter().any(|v| v.contains("not externally attested"))
    {
        bail!("completed ledgers must state whether external attestation is provided");
    }
    if expected.case_set_hash_algorithm != CASE_SET_HASH_ALGORITHM {
        bail!("unsupported expected-case-set hash algorithm");
    }
    if expected.expected_case_ids.is_empty() {
        bail!("expected case set must not be empty");
    }

    let mut expected_ids = BTreeSet::new();
    for id in &expected.expected_case_ids {
        if id.trim().is_empty() || !expected_ids.insert(id.clone()) {
            bail!("expected case IDs must be non-empty and unique: {id:?}");
        }
    }
    let expected_digest = case_set_digest(&expected_ids);
    if ledger.expected_case_set_sha256 != expected_digest {
        bail!("ledger expected-case-set digest does not match the independently supplied expected case set");
    }

    if ledger.events.is_empty() {
        bail!("evidence ledger must contain events");
    }

    let mut states = expected_ids
        .iter()
        .map(|id| (id.clone(), CaseState::default()))
        .collect::<BTreeMap<_, _>>();
    let mut previous_hash = GENESIS_HASH.to_string();
    let mut adapter_calls = 0u64;
    let mut observed_effects = 0u64;
    let mut confirmed = 0u64;
    let mut failed = 0u64;
    let mut unknown = 0u64;
    let mut receipt_rejections = 0u64;

    for (index, event) in ledger.events.iter().enumerate() {
        if event.sequence != index as u64 {
            bail!("event sequence must be contiguous from zero; index {index} has sequence {}", event.sequence);
        }
        if !format_digest_is_lower_hex(&event.previous_hash)
            || !format_digest_is_lower_hex(&event.event_hash)
        {
            bail!("event {} contains a malformed SHA-256 digest", event.sequence);
        }
        if event.previous_hash != previous_hash {
            bail!("event {} has a broken previous_hash link", event.sequence);
        }
        let expected_hash = event_digest(event);
        if event.event_hash != expected_hash {
            bail!("event {} hash mismatch", event.sequence);
        }
        previous_hash.clone_from(&event.event_hash);

        let state = states
            .get_mut(&event.case_id)
            .with_context(|| format!("event {} references unknown case {}", event.sequence, event.case_id))?;
        if state.terminal_class.is_some() {
            bail!("case {} has an event after its terminal record", event.case_id);
        }
        if state.outcome_unknown && event.event_type != "case_terminal" {
            bail!("case {} has activity after outcome_unknown; retries/reconciliation require a separately specified protocol", event.case_id);
        }

        match event.event_type.as_str() {
            "case_started" => {
                if state.started {
                    bail!("case {} started more than once", event.case_id);
                }
                if event.decision != "none"
                    || event.terminal_class != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 0
                    || event.postcondition != "none"
                    || event.receipt != "none"
                {
                    bail!("case_started event has non-empty outcome fields");
                }
                state.started = true;
            }
            "authorization_decision" => {
                require_started(&state, &event.case_id)?;
                if state.decision.is_some()
                    || !["allow", "deny", "indeterminate"].contains(&event.decision.as_str())
                    || event.terminal_class != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 0
                    || event.postcondition != "none"
                    || event.receipt != "none"
                {
                    bail!("case {} has duplicate or malformed authorization decision", event.case_id);
                }
                state.decision = Some(event.decision.clone());
            }
            "effect_adapter_invoked" => {
                require_started(&state, &event.case_id)?;
                if state.decision.as_deref() != Some("allow")
                    || state.outcome_unknown
                    || event.decision != "none"
                    || event.terminal_class != "none"
                    || event.adapter_calls != 1
                    || event.observed_effects != 0
                    || event.postcondition != "none"
                    || event.receipt != "none"
                {
                    bail!("case {} has an invalid effect-adapter invocation event", event.case_id);
                }
                state.adapter_calls += event.adapter_calls;
                adapter_calls += event.adapter_calls;
                if state.adapter_calls > 1 {
                    bail!("case {} invoked its single-effect adapter more than once", event.case_id);
                }
            }
            "effect_observed" => {
                require_started(&state, &event.case_id)?;
                if state.decision.as_deref() != Some("allow")
                    || state.adapter_calls != 1
                    || state.observed_effects != 0
                    || event.decision != "none"
                    || event.terminal_class != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 1
                    || event.postcondition != "none"
                    || event.receipt != "none"
                {
                    bail!("case {} has an invalid or duplicate observed-effect event", event.case_id);
                }
                state.observed_effects += event.observed_effects;
                observed_effects += event.observed_effects;
            }
            "postcondition_result" => {
                require_started(&state, &event.case_id)?;
                if state.decision.as_deref() != Some("allow")
                    || state.adapter_calls != 1
                    || state.postcondition.is_some()
                    || !["confirmed", "failed"].contains(&event.postcondition.as_str())
                    || event.decision != "none"
                    || event.terminal_class != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 0
                    || event.receipt != "none"
                    || (event.postcondition == "confirmed" && state.observed_effects != 1)
                {
                    bail!("case {} has an invalid postcondition result", event.case_id);
                }
                if event.postcondition == "confirmed" {
                    confirmed += 1;
                } else {
                    failed += 1;
                }
                state.postcondition = Some(event.postcondition.clone());
            }
            "receipt_result" => {
                require_started(&state, &event.case_id)?;
                if state.decision.as_deref() != Some("allow")
                    || state.postcondition.is_none()
                    || state.receipt.is_some()
                    || !["verified", "rejected"].contains(&event.receipt.as_str())
                    || event.decision != "none"
                    || event.terminal_class != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 0
                    || event.postcondition != "none"
                {
                    bail!("case {} has an invalid receipt-result event", event.case_id);
                }
                if event.receipt == "rejected" {
                    receipt_rejections += 1;
                }
                state.receipt = Some(event.receipt.clone());
            }
            "outcome_unknown" => {
                require_started(&state, &event.case_id)?;
                if state.decision.as_deref() != Some("allow")
                    || state.adapter_calls != 1
                    || state.postcondition.is_some()
                    || state.receipt.is_some()
                    || state.outcome_unknown
                    || event.decision != "none"
                    || event.terminal_class != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 0
                    || event.postcondition != "none"
                    || event.receipt != "none"
                {
                    bail!("case {} has an invalid outcome-unknown event", event.case_id);
                }
                state.outcome_unknown = true;
                unknown += 1;
            }
            "retry_attempted" => {
                bail!("case {} records a retry attempt; this one-shot benchmark forbids automatic retries", event.case_id);
            }
            "case_terminal" => {
                require_started(&state, &event.case_id)?;
                if event.decision != "none"
                    || event.adapter_calls != 0
                    || event.observed_effects != 0
                    || event.postcondition != "none"
                    || event.receipt != "none"
                {
                    bail!("case_terminal event must not invent stage evidence");
                }
                validate_terminal(&state, &event.case_id, &event.terminal_class)?;
                state.terminal_class = Some(event.terminal_class.clone());
            }
            other => bail!("unknown event type {other:?} in case {}", event.case_id),
        }
    }

    for (id, state) in &states {
        if !state.started {
            bail!("expected case {id} has no case_started row");
        }
        if state.terminal_class.is_none() {
            bail!("expected case {id} is missing its terminal row");
        }
    }

    Ok(Summary {
        events: ledger.events.len(),
        cases: expected_ids.len(),
        adapter_calls,
        observed_effects,
        confirmed,
        failed,
        unknown,
        receipt_rejections,
        sha256,
    })
}

fn require_started(state: &CaseState, case_id: &str) -> Result<()> {
    if !state.started {
        bail!("case {case_id} has an event before case_started");
    }
    Ok(())
}

fn validate_terminal(state: &CaseState, case_id: &str, terminal: &str) -> Result<()> {
    let decision = state.decision.as_deref().unwrap_or("none");
    match terminal {
        "postcondition_confirmed"
            if decision == "allow"
                && state.adapter_calls == 1
                && state.observed_effects == 1
                && state.postcondition.as_deref() == Some("confirmed")
                && state.receipt.as_deref() == Some("verified")
                && !state.outcome_unknown => Ok(()),
        "denied_before_effect"
            if decision == "deny"
                && state.adapter_calls == 0
                && state.observed_effects == 0
                && state.postcondition.is_none()
                && state.receipt.is_none()
                && !state.outcome_unknown => Ok(()),
        "indeterminate_before_effect"
            if decision == "indeterminate"
                && state.adapter_calls == 0
                && state.observed_effects == 0
                && state.postcondition.is_none()
                && state.receipt.is_none()
                && !state.outcome_unknown => Ok(()),
        "outcome_unknown"
            if decision == "allow"
                && state.adapter_calls == 1
                && state.observed_effects == 0
                && state.postcondition.is_none()
                && state.receipt.is_none()
                && state.outcome_unknown => Ok(()),
        "effect_observed_postcondition_failed"
            if decision == "allow"
                && state.adapter_calls == 1
                && state.observed_effects == 1
                && state.postcondition.as_deref() == Some("failed")
                && state.receipt.as_deref() != Some("verified")
                && !state.outcome_unknown => Ok(()),
        "postcondition_failed_without_effect"
            if decision == "allow"
                && state.adapter_calls == 1
                && state.observed_effects == 0
                && state.postcondition.as_deref() == Some("failed")
                && state.receipt.is_none()
                && !state.outcome_unknown => Ok(()),
        "evidence_rejected_after_effect"
            if decision == "allow"
                && state.adapter_calls == 1
                && state.observed_effects == 1
                && state.postcondition.as_deref() == Some("confirmed")
                && state.receipt.as_deref() == Some("rejected")
                && !state.outcome_unknown => Ok(()),
        _ => bail!("case {case_id} terminal class {terminal:?} contradicts its stage evidence"),
    }
}

fn event_digest(event: &Event) -> String {
    let fields = [
        event.sequence.to_string(),
        event.case_id.clone(),
        event.event_type.clone(),
        event.decision.clone(),
        event.terminal_class.clone(),
        event.adapter_calls.to_string(),
        event.observed_effects.to_string(),
        event.postcondition.clone(),
        event.receipt.clone(),
        event.previous_hash.clone(),
    ];
    let mut hasher = Sha256::new();
    for field in fields {
        let bytes = field.as_bytes();
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn case_set_digest(ids: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        let bytes = id.as_bytes();
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn format_digest_is_lower_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::{event_digest, format_digest_is_lower_hex, parse_and_validate, Event, Ledger, GENESIS_HASH};
    use std::path::PathBuf;

    fn fixture_paths() -> (PathBuf, PathBuf) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask must be nested under workspace root")
            .to_path_buf();
        (
            root.join("docs/research/authorization-evidence-ledger.smoke.v0.json"),
            root.join("docs/research/authorization-evidence-ledger.expected-cases.v0.json"),
        )
    }

    #[test]
    fn committed_synthetic_ledger_validates() {
        let (ledger_path, cases_path) = fixture_paths();
        let ledger_raw = std::fs::read(ledger_path).unwrap();
        let cases_raw = std::fs::read(cases_path).unwrap();
        let summary = parse_and_validate(&ledger_raw, &cases_raw).unwrap();
        assert_eq!(summary.events, 10);
        assert_eq!(summary.cases, 2);
        assert_eq!(summary.adapter_calls, 1);
        assert_eq!(summary.observed_effects, 1);
        assert_eq!(summary.confirmed, 1);
    }

    #[test]
    fn changing_event_payload_without_rehash_is_rejected() {
        let (ledger_path, cases_path) = fixture_paths();
        let ledger_raw = std::fs::read(ledger_path).unwrap();
        let cases_raw = std::fs::read(cases_path).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&ledger_raw).unwrap();
        value["events"][1]["decision"] = serde_json::json!("deny");
        assert!(parse_and_validate(&serde_json::to_vec(&value).unwrap(), &cases_raw).is_err());
    }

    #[test]
    fn removing_an_expected_case_cannot_shrink_the_denominator() {
        let (ledger_path, cases_path) = fixture_paths();
        let mut ledger_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(ledger_path).unwrap()).unwrap();
        let cases_raw = std::fs::read(cases_path).unwrap();
        ledger_value["events"].as_array_mut().unwrap().truncate(7);
        assert!(parse_and_validate(&serde_json::to_vec(&ledger_value).unwrap(), &cases_raw).is_err());
    }

    #[test]
    fn confirmed_postcondition_requires_an_observed_effect() {
        let (ledger_path, cases_path) = fixture_paths();
        let ledger_raw = std::fs::read(ledger_path).unwrap();
        let cases_raw = std::fs::read(cases_path).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&ledger_raw).unwrap();
        value["events"].as_array_mut().unwrap().remove(3);
        rehash_events(&mut value);
        assert!(parse_and_validate(&serde_json::to_vec(&value).unwrap(), &cases_raw).is_err());
    }

    #[test]
    fn confirmed_terminal_requires_a_verified_receipt() {
        let (ledger_path, cases_path) = fixture_paths();
        let ledger_raw = std::fs::read(ledger_path).unwrap();
        let cases_raw = std::fs::read(cases_path).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&ledger_raw).unwrap();
        value["events"].as_array_mut().unwrap().remove(5);
        rehash_events(&mut value);
        assert!(parse_and_validate(&serde_json::to_vec(&value).unwrap(), &cases_raw).is_err());
    }

    #[test]
    fn event_hash_is_deterministic_for_typed_payload() {
        let (ledger_path, _) = fixture_paths();
        let ledger: Ledger =
            serde_json::from_slice(&std::fs::read(ledger_path).unwrap()).unwrap();
        assert_eq!(ledger.events[0].event_hash, event_digest(&ledger.events[0]));
        assert!(format_digest_is_lower_hex(&ledger.events[0].event_hash));
    }

    fn rehash_events(ledger: &mut serde_json::Value) {
        let mut previous_hash = GENESIS_HASH.to_string();
        let events = ledger["events"].as_array_mut().unwrap();
        for (index, value) in events.iter_mut().enumerate() {
            value["sequence"] = serde_json::json!(index as u64);
            value["previous_hash"] = serde_json::json!(previous_hash.clone());
            let event: Event = serde_json::from_value(value.clone()).unwrap();
            let hash = event_digest(&event);
            value["event_hash"] = serde_json::json!(hash.clone());
            previous_hash = hash;
        }
    }
}
