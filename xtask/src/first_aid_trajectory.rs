//! Synthetic event-trace audit for first-aid workflow process safety.
//!
//! This is a deterministic trace-contract checker, not a clinical protocol,
//! runtime authorization service, or demonstration of clinical safety.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

const CORPUS_VERSION: &str = "first-aid-trajectory-corpus-v0";
const REQUIRED_CLAIM_CEILING: &str =
    "Synthetic process-integrity audit only; no clinical safety or patient-outcome claim.";
const EXPECTED_CORPUS_SHA256: &str =
    "1ac414e6b920a6259d68a20c273a0a9465e98c7a596393f9777be9976d6e3fd3";

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct TrajectoryCorpus {
    schema_version: String,
    status: String,
    clinical_use_authorized: bool,
    claim_ceiling: String,
    notes: Option<String>,
    scenarios: Vec<TrajectoryScenario>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct TrajectoryScenario {
    trace_id: String,
    description: String,
    criticality: Criticality,
    connectivity: Connectivity,
    events: Vec<TraceEvent>,
    expected_violation_codes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Criticality {
    Critical,
    High,
    Routine,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Connectivity {
    Online,
    Offline,
    Intermittent,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BundleState {
    Valid,
    Expired,
    Revoked,
    DigestMismatch,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ResourceState {
    Fresh,
    Stale,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PresentedAs {
    Confirmed,
    Unconfirmed,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EventKind {
    SessionStarted,
    BundleLoaded,
    BundleVerified,
    ProtocolRendered,
    BundleRevoked,
    SessionInterrupted,
    SessionResumed,
    UncertaintyDetected,
    UncertaintyPreserved,
    UnknownNormalized,
    HandoffEmitted,
    ResourcePresented,
    NetworkRequiredForCore,
    SensitiveDataEgress,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct TraceEvent {
    sequence: u32,
    kind: EventKind,
    bundle_id: Option<String>,
    bundle_state: Option<BundleState>,
    digest_verified: Option<bool>,
    resource_state: Option<ResourceState>,
    presented_as: Option<PresentedAs>,
    sensitive_payload: Option<bool>,
    consent_scoped: Option<bool>,
}

#[derive(Debug, Serialize)]
struct TrajectoryReceipt {
    receipt_version: &'static str,
    status: &'static str,
    corpus_sha256: String,
    evaluation_protocol_sha256: String,
    mutation_matrix_sha256: String,
    state_model_sha256: String,
    state_model_test_outcome: String,
    cargo_lock_sha256: String,
    rustc_version: String,
    cargo_version: String,
    source_revision_sha: String,
    validator_source_sha256: String,
    entrypoint_source_sha256: String,
    executable_sha256: String,
    runtime_os: String,
    runtime_arch: String,
    trace_count: usize,
    counterexample_trace_count: usize,
    critical_trace_count: usize,
    critical_counterexample_count: usize,
    violation_count: usize,
    claim_ceiling: &'static str,
}

fn checked_corpus_digest(bytes: &[u8]) -> Result<String> {
    let digest = sha256_hex(bytes);
    if digest != EXPECTED_CORPUS_SHA256 {
        bail!(
            "trajectory corpus digest mismatch: expected {}, got {}",
            EXPECTED_CORPUS_SHA256,
            digest
        );
    }
    Ok(digest)
}

pub fn run(root: &Path, corpus_arg: &Path, out_arg: Option<&Path>) -> Result<()> {
    let corpus_path = resolve(root, corpus_arg);
    let corpus_bytes = fs::read(&corpus_path)
        .with_context(|| format!("read trajectory corpus {}", corpus_path.display()))?;
    let corpus_digest = checked_corpus_digest(&corpus_bytes)?;
    let corpus_text =
        std::str::from_utf8(&corpus_bytes).context("trajectory corpus must be valid UTF-8")?;
    let corpus: TrajectoryCorpus =
        serde_json::from_str(corpus_text).context("parse trajectory corpus JSON")?;
    validate_corpus(&corpus)?;

    let evaluation_protocol =
        fs::read(root.join("docs/research/first-aid/EVALUATION_PROTOCOL_V0.md"))
            .context("read evaluation protocol for receipt identity")?;
    let evaluation_protocol_digest = sha256_hex(&evaluation_protocol);
    let mutation_matrix = fs::read(root.join("docs/research/first-aid/mutation-matrix-v0.json"))
        .context("read mutation matrix for receipt identity")?;
    let mutation_matrix_digest = sha256_hex(&mutation_matrix);
    let state_model = fs::read(root.join("xtask/src/first_aid_model.rs"))
        .context("read finite-state model for receipt identity")?;
    let state_model_digest = sha256_hex(&state_model);
    let cargo_lock = fs::read(root.join("Cargo.lock"))
        .context("read Cargo.lock for dependency-resolution identity")?;
    let cargo_lock_digest = sha256_hex(&cargo_lock);
    let rustc_version = std::env::var("RUSTC_VERSION")
        .unwrap_or_else(|_| "not-provided".to_string());
    let cargo_version = std::env::var("CARGO_VERSION")
        .unwrap_or_else(|_| "not-provided".to_string());
    let validator_source = fs::read(root.join("xtask/src/first_aid_trajectory.rs"))
        .context("read trajectory validator source for receipt identity")?;
    let entrypoint_source = fs::read(root.join("xtask/src/main.rs"))
        .context("read xtask entrypoint source for receipt identity")?;
    let executable_path =
        std::env::current_exe().context("resolve trajectory validator executable")?;
    let executable_bytes = fs::read(&executable_path)
        .with_context(|| format!("read validator executable {}", executable_path.display()))?;

    let results: Vec<Vec<String>> = corpus.scenarios.iter().map(inspect_trace).collect();
    let receipt = TrajectoryReceipt {
        receipt_version: "first-aid-trajectory-receipt-v0",
        status: "synthetic_trajectory_contract_passed",
        corpus_sha256: corpus_digest,
        evaluation_protocol_sha256: evaluation_protocol_digest,
        mutation_matrix_sha256: mutation_matrix_digest,
        state_model_sha256: state_model_digest,
        state_model_test_outcome: std::env::var("STATE_MODEL_TEST_OUTCOME")
            .unwrap_or_else(|_| "not-provided".to_string()),
        cargo_lock_sha256: cargo_lock_digest,
        rustc_version,
        cargo_version,
        source_revision_sha: std::env::var("FIRST_AID_SOURCE_REVISION")
            .unwrap_or_else(|_| "not-provided".to_string()),
        validator_source_sha256: sha256_hex(&validator_source),
        entrypoint_source_sha256: sha256_hex(&entrypoint_source),
        executable_sha256: sha256_hex(&executable_bytes),
        runtime_os: std::env::consts::OS.to_string(),
        runtime_arch: std::env::consts::ARCH.to_string(),
        trace_count: corpus.scenarios.len(),
        counterexample_trace_count: results.iter().filter(|codes| !codes.is_empty()).count(),
        critical_trace_count: corpus
            .scenarios
            .iter()
            .filter(|trace| trace.criticality == Criticality::Critical)
            .count(),
        critical_counterexample_count: corpus
            .scenarios
            .iter()
            .zip(&results)
            .filter(|(trace, codes)| {
                trace.criticality == Criticality::Critical && !codes.is_empty()
            })
            .count(),
        violation_count: results.iter().map(Vec::len).sum(),
        claim_ceiling: "synthetic process-integrity audit only; no clinical safety or patient-outcome claim",
    };
    let encoded = serde_json::to_string_pretty(&receipt)?;

    if let Some(out_arg) = out_arg {
        let out_path = resolve(root, out_arg);
        if let Some(parent) = out_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create receipt directory {}", parent.display()))?;
        }
        fs::write(&out_path, format!("{encoded}\n"))
            .with_context(|| format!("write trajectory receipt {}", out_path.display()))?;
        println!("Receipt written: {}", out_path.display());
    }
    println!("{encoded}");
    Ok(())
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_corpus(corpus: &TrajectoryCorpus) -> Result<()> {
    if corpus.schema_version != CORPUS_VERSION {
        bail!(
            "unsupported trajectory corpus version: {}",
            corpus.schema_version
        );
    }
    if corpus.status != "synthetic_process_contract_tests_only" {
        bail!("trajectory corpus status must remain explicitly synthetic");
    }
    if corpus.clinical_use_authorized {
        bail!("trajectory corpus cannot authorize clinical use");
    }
    if corpus.claim_ceiling != REQUIRED_CLAIM_CEILING {
        bail!(
            "trajectory corpus claim ceiling must remain exactly non-clinical: {}",
            REQUIRED_CLAIM_CEILING
        );
    }
    if corpus.scenarios.is_empty() {
        bail!("trajectory corpus must contain at least one trace");
    }

    let mut trace_ids = HashSet::new();
    for trace in &corpus.scenarios {
        if trace.trace_id.trim().is_empty() || !trace_ids.insert(trace.trace_id.as_str()) {
            bail!("empty or duplicate trace id: {}", trace.trace_id);
        }
        if trace.description.trim().is_empty() {
            bail!("{}: description must be nonempty", trace.trace_id);
        }
        if trace.events.is_empty() {
            bail!("{}: trace must contain events", trace.trace_id);
        }
        let mut previous_sequence = None;
        for event in &trace.events {
            if previous_sequence.is_some_and(|previous| event.sequence <= previous) {
                bail!("{}: event sequence must increase strictly", trace.trace_id);
            }
            previous_sequence = Some(event.sequence);
            validate_event_fields(trace, event)?;
        }
        let mut expected_codes = HashSet::new();
        for code in &trace.expected_violation_codes {
            if !is_known_violation_code(code) {
                bail!("{}: unknown violation code {}", trace.trace_id, code);
            }
            if !expected_codes.insert(code.as_str()) {
                bail!(
                    "{}: duplicate expected violation code {}",
                    trace.trace_id,
                    code
                );
            }
        }
        let actual = inspect_trace(trace);
        if actual != trace.expected_violation_codes {
            bail!(
                "{}: expected violation codes {:?}, observed {:?}",
                trace.trace_id,
                trace.expected_violation_codes,
                actual
            );
        }
    }
    let _ = &corpus.notes;
    Ok(())
}

fn validate_event_fields(trace: &TrajectoryScenario, event: &TraceEvent) -> Result<()> {
    let needs_bundle_id = matches!(
        event.kind,
        EventKind::BundleLoaded
            | EventKind::BundleVerified
            | EventKind::ProtocolRendered
            | EventKind::BundleRevoked
    );
    if needs_bundle_id && event.bundle_id.as_deref().is_none_or(str::is_empty) {
        bail!(
            "{} event {} requires a nonempty bundle_id",
            trace.trace_id,
            event.sequence
        );
    }
    if matches!(
        event.kind,
        EventKind::BundleLoaded | EventKind::BundleVerified
    ) && event.bundle_state.is_none()
    {
        bail!(
            "{} event {} requires bundle_state",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind == EventKind::BundleVerified && event.digest_verified.is_none() {
        bail!(
            "{} event {} requires digest_verified",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind == EventKind::ResourcePresented
        && (event.resource_state.is_none() || event.presented_as.is_none())
    {
        bail!(
            "{} event {} requires resource_state and presented_as",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind == EventKind::SensitiveDataEgress
        && (event.sensitive_payload.is_none() || event.consent_scoped.is_none())
    {
        bail!(
            "{} event {} requires sensitive_payload and consent_scoped",
            trace.trace_id,
            event.sequence
        );
    }
    Ok(())
}

fn inspect_trace(trace: &TrajectoryScenario) -> Vec<String> {
    let mut violations = Vec::new();
    let mut session_active = false;
    let mut active_bundle: Option<String> = None;
    let mut verified_bundle: Option<String> = None;
    let mut revoked_bundles: HashSet<String> = HashSet::new();
    let mut uncertainty_pending = false;
    let mut uncertainty_preserved = false;

    for event in &trace.events {
        match event.kind {
            EventKind::BundleLoaded => {
                active_bundle = event.bundle_id.clone();
                verified_bundle = None;
            }
            EventKind::BundleVerified => {
                let bundle_id = event.bundle_id.as_deref().unwrap_or_default();
                let is_valid = session_active
                    && active_bundle.as_deref() == Some(bundle_id)
                    && event.bundle_state == Some(BundleState::Valid)
                    && event.digest_verified == Some(true)
                    && !revoked_bundles.contains(bundle_id);
                if is_valid {
                    verified_bundle = Some(bundle_id.to_string());
                } else {
                    verified_bundle = None;
                    violations.push("invalid_bundle_verification".to_string());
                }
            }
            EventKind::ProtocolRendered => {
                let bundle_id = event.bundle_id.as_deref().unwrap_or_default();
                if !session_active {
                    violations.push("render_without_active_session".to_string());
                }
                if revoked_bundles.contains(bundle_id)
                    || event.bundle_state == Some(BundleState::Revoked)
                {
                    violations.push("render_after_revocation".to_string());
                } else if verified_bundle.as_deref() != Some(bundle_id) {
                    violations.push("render_without_verified_bundle".to_string());
                }
            }
            EventKind::BundleRevoked => {
                let bundle_id = event.bundle_id.clone().unwrap_or_default();
                revoked_bundles.insert(bundle_id.clone());
                if active_bundle.as_deref() == Some(bundle_id.as_str()) {
                    verified_bundle = None;
                }
            }
            EventKind::SessionInterrupted => {
                session_active = false;
                verified_bundle = None;
            }
            EventKind::SessionResumed => {
                // Resume reactivates the session but never restores the prior verification.
                session_active = true;
                verified_bundle = None;
            }
            EventKind::UncertaintyDetected => {
                uncertainty_pending = true;
                uncertainty_preserved = false;
            }
            EventKind::UncertaintyPreserved => {
                if uncertainty_pending {
                    uncertainty_preserved = true;
                }
            }
            EventKind::UnknownNormalized => {
                violations.push("unknown_normalized".to_string());
            }
            EventKind::HandoffEmitted => {
                if uncertainty_pending && !uncertainty_preserved {
                    violations.push("uncertainty_lost_at_handoff".to_string());
                }
                uncertainty_pending = false;
                uncertainty_preserved = false;
            }
            EventKind::ResourcePresented => {
                if matches!(
                    event.resource_state,
                    Some(ResourceState::Stale | ResourceState::Unknown)
                ) && event.presented_as == Some(PresentedAs::Confirmed)
                {
                    violations.push("stale_resource_claimed_confirmed".to_string());
                }
            }
            EventKind::NetworkRequiredForCore => match trace.connectivity {
                Connectivity::Offline => {
                    violations.push("offline_core_network_dependency".to_string());
                }
                Connectivity::Intermittent => {
                    violations.push("intermittent_core_network_dependency".to_string());
                }
                Connectivity::Online => {}
            },
            EventKind::SensitiveDataEgress => {
                if event.sensitive_payload == Some(true) && event.consent_scoped != Some(true) {
                    violations.push("sensitive_data_egress_without_consent".to_string());
                }
            }
            EventKind::SessionStarted => {
                session_active = true;
                // A new session must not inherit the previous session's verified bundle.
                active_bundle = None;
                verified_bundle = None;
                // An unresolved uncertainty cannot disappear silently at a session boundary.
                if uncertainty_pending && !uncertainty_preserved {
                    violations.push("uncertainty_lost_at_session_boundary".to_string());
                }
            }
        }
    }
    violations
}

fn is_known_violation_code(code: &str) -> bool {
    matches!(
        code,
        "invalid_bundle_verification"
            | "render_after_revocation"
            | "render_without_active_session"
            | "render_without_verified_bundle"
            | "unknown_normalized"
            | "uncertainty_lost_at_handoff"
            | "uncertainty_lost_at_session_boundary"
            | "stale_resource_claimed_confirmed"
            | "offline_core_network_dependency"
            | "intermittent_core_network_dependency"
            | "sensitive_data_egress_without_consent"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS_JSON: &str =
        include_str!("../../docs/research/first-aid/trajectory-corpus-v0.json");

    fn corpus() -> TrajectoryCorpus {
        serde_json::from_str(CORPUS_JSON).expect("trajectory fixture must parse")
    }

    fn scenario(trace_id: &str) -> TrajectoryScenario {
        corpus()
            .scenarios
            .into_iter()
            .find(|trace| trace.trace_id == trace_id)
            .unwrap_or_else(|| panic!("missing frozen trace {trace_id}"))
    }

    #[test]
    fn checked_in_trajectory_corpus_has_pinned_digest() {
        assert_eq!(sha256_hex(CORPUS_JSON.as_bytes()), EXPECTED_CORPUS_SHA256);
        assert_eq!(corpus().scenarios.len(), 12);
        assert_eq!(
            checked_corpus_digest(CORPUS_JSON.as_bytes()).expect("pinned fixture digest passes"),
            EXPECTED_CORPUS_SHA256
        );
    }

    #[test]
    fn one_byte_corpus_mutation_is_rejected_by_digest_gate() {
        let mut bytes = CORPUS_JSON.as_bytes().to_vec();
        bytes[0] ^= 1;
        assert!(checked_corpus_digest(&bytes).is_err());
    }

    #[test]
    fn independent_expected_violation_codes_match() {
        validate_corpus(&corpus()).expect("frozen trace oracles must match");
    }

    // A second, JSON-value-based implementation intentionally avoids the production
    // event enums and state structs. This provides implementation-diversity checks,
    // but is not an independent clinical oracle or expert adjudication.
    fn reference_codes_from_json(trace: &serde_json::Value) -> Vec<String> {
        let connectivity = trace["connectivity"].as_str().unwrap_or_default();
        let mut codes = Vec::new();
        let mut session_active = false;
        let mut active_bundle: Option<String> = None;
        let mut verified_bundle: Option<String> = None;
        let mut revoked_bundles = HashSet::new();
        let mut uncertainty_pending = false;
        let mut uncertainty_preserved = false;

        for event in trace["events"].as_array().into_iter().flatten() {
            let kind = event["kind"].as_str().unwrap_or_default();
            let bundle_id = event["bundle_id"].as_str().map(str::to_owned);
            match kind {
                "session_started" => {
                    session_active = true;
                    active_bundle = None;
                    verified_bundle = None;
                    if uncertainty_pending && !uncertainty_preserved {
                        codes.push("uncertainty_lost_at_session_boundary".to_owned());
                    }
                }
                "bundle_loaded" => {
                    active_bundle = bundle_id.clone();
                    verified_bundle = None;
                }
                "bundle_verified" => {
                    let id = bundle_id.as_deref().unwrap_or_default();
                    if session_active
                        && active_bundle.as_deref() == Some(id)
                        && event["bundle_state"] == "valid"
                        && event["digest_verified"].as_bool() == Some(true)
                        && !revoked_bundles.contains(id)
                    {
                        verified_bundle = Some(id.to_owned());
                    } else {
                        verified_bundle = None;
                        codes.push("invalid_bundle_verification".to_owned());
                    }
                }
                "protocol_rendered" => {
                    let id = bundle_id.as_deref().unwrap_or_default();
                    if !session_active {
                        codes.push("render_without_active_session".to_owned());
                    }
                    if revoked_bundles.contains(id) || event["bundle_state"] == "revoked" {
                        codes.push("render_after_revocation".to_owned());
                    } else if verified_bundle.as_deref() != Some(id) {
                        codes.push("render_without_verified_bundle".to_owned());
                    }
                }
                "bundle_revoked" => {
                    let id = bundle_id.unwrap_or_default();
                    revoked_bundles.insert(id.clone());
                    if active_bundle.as_deref() == Some(id.as_str()) {
                        verified_bundle = None;
                    }
                }
                "session_interrupted" => {
                    session_active = false;
                    verified_bundle = None;
                }
                "session_resumed" => {
                    session_active = true;
                    verified_bundle = None;
                }
                "uncertainty_detected" => {
                    uncertainty_pending = true;
                    uncertainty_preserved = false;
                }
                "uncertainty_preserved" => {
                    if uncertainty_pending {
                        uncertainty_preserved = true;
                    }
                }
                "unknown_normalized" => codes.push("unknown_normalized".to_owned()),
                "handoff_emitted" => {
                    if uncertainty_pending && !uncertainty_preserved {
                        codes.push("uncertainty_lost_at_handoff".to_owned());
                    }
                    uncertainty_pending = false;
                    uncertainty_preserved = false;
                }
                "resource_presented" => {
                    let state = event["resource_state"].as_str().unwrap_or_default();
                    if matches!(state, "stale" | "unknown")
                        && event["presented_as"] == "confirmed"
                    {
                        codes.push("stale_resource_claimed_confirmed".to_owned());
                    }
                }
                "network_required_for_core" => match connectivity {
                    "offline" => codes.push("offline_core_network_dependency".to_owned()),
                    "intermittent" => {
                        codes.push("intermittent_core_network_dependency".to_owned())
                    }
                    _ => {}
                },
                "sensitive_data_egress" => {
                    if event["sensitive_payload"].as_bool() == Some(true)
                        && event["consent_scoped"].as_bool() != Some(true)
                    {
                        codes.push("sensitive_data_egress_without_consent".to_owned());
                    }
                }
                _ => {}
            }
        }
        codes
    }

    #[test]
    fn differential_json_reference_agrees_on_frozen_traces() {
        let json: serde_json::Value =
            serde_json::from_str(CORPUS_JSON).expect("frozen corpus JSON must parse");
        for value in json["scenarios"].as_array().expect("scenario list") {
            let trace: TrajectoryScenario =
                serde_json::from_value(value.clone()).expect("scenario must deserialize");
            assert_eq!(
                inspect_trace(&trace),
                reference_codes_from_json(value),
                "differential trace mismatch: {}",
                trace.trace_id
            );
        }
    }

    #[test]
    fn differential_json_reference_agrees_on_session_mutations() {
        let mut trace = corpus().scenarios.remove(0);
        trace.events.remove(0);
        trace.expected_violation_codes = vec!["render_without_active_session".to_owned()];
        let value = serde_json::to_value(&trace).expect("mutated trace serializes");
        assert_eq!(
            inspect_trace(&trace),
            reference_codes_from_json(&value)
        );
    }

    #[test]
    fn differential_json_reference_agrees_on_adversarial_mutation_suite() {
        let mut cases: Vec<(&str, TrajectoryScenario)> = Vec::new();

        let mut outside_session = scenario("FA-TRACE-001");
        outside_session.events.remove(0);
        cases.push(("outside active session", outside_session));

        let mut wrong_bundle = scenario("FA-TRACE-001");
        wrong_bundle
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists")
            .bundle_id = Some("bundle-other".to_owned());
        cases.push(("wrong bundle identity", wrong_bundle));

        let mut bad_digest = scenario("FA-TRACE-001");
        bad_digest
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::BundleVerified)
            .expect("verify event exists")
            .digest_verified = Some(false);
        cases.push(("bad digest", bad_digest));

        let mut expired = scenario("FA-TRACE-001");
        expired
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::BundleVerified)
            .expect("verify event exists")
            .bundle_state = Some(BundleState::Expired);
        cases.push(("expired bundle", expired));

        let mut uncertainty_reentered = scenario("FA-TRACE-012");
        let handoff_index = uncertainty_reentered
            .events
            .iter()
            .position(|event| event.kind == EventKind::HandoffEmitted)
            .expect("handoff event exists");
        uncertainty_reentered.events[handoff_index].sequence += 1;
        uncertainty_reentered.events.insert(
            handoff_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::UncertaintyDetected,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        cases.push(("uncertainty re-entered", uncertainty_reentered));

        let mut revoked_new_session = scenario("FA-TRACE-003");
        let render_index = revoked_new_session
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        revoked_new_session.events[render_index].sequence += 1;
        revoked_new_session.events.insert(
            render_index,
            TraceEvent {
                sequence: 5,
                kind: EventKind::SessionStarted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        cases.push(("revocation across session", revoked_new_session));

        let mut stale_confirmed = scenario("FA-TRACE-011");
        stale_confirmed
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::ResourcePresented)
            .expect("resource presentation exists")
            .presented_as = Some(PresentedAs::Confirmed);
        cases.push(("stale resource claimed confirmed", stale_confirmed));

        let mut intermittent = scenario("FA-TRACE-007");
        intermittent.connectivity = Connectivity::Intermittent;
        cases.push(("intermittent core network", intermittent));

        let mut non_sensitive = scenario("FA-TRACE-009");
        non_sensitive
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::SensitiveDataEgress)
            .expect("egress event exists")
            .sensitive_payload = Some(false);
        cases.push(("non-sensitive egress control", non_sensitive));

        let mut uncertain_session_boundary = scenario("FA-TRACE-012");
        let preserve_index = uncertain_session_boundary
            .events
            .iter()
            .position(|event| event.kind == EventKind::UncertaintyPreserved)
            .expect("preservation event exists");
        uncertain_session_boundary.events.remove(preserve_index);
        let handoff_index = uncertain_session_boundary
            .events
            .iter()
            .position(|event| event.kind == EventKind::HandoffEmitted)
            .expect("handoff event exists");
        uncertain_session_boundary.events[handoff_index].sequence += 1;
        uncertain_session_boundary.events.insert(
            handoff_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::SessionStarted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        cases.push(("unresolved uncertainty across new session", uncertain_session_boundary));

        let mut resume_without_interrupt = scenario("FA-TRACE-001");
        let render_index = resume_without_interrupt
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        resume_without_interrupt.events[render_index].sequence += 1;
        resume_without_interrupt.events.insert(
            render_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::SessionResumed,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        cases.push(("resume without interruption", resume_without_interrupt));

        let mut verify_without_session = scenario("FA-TRACE-001");
        let verify_index = verify_without_session
            .events
            .iter()
            .position(|event| event.kind == EventKind::BundleVerified)
            .expect("verification event exists");
        verify_without_session.events[verify_index].sequence += 1;
        let render_index = verify_without_session
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        verify_without_session.events[render_index].sequence += 1;
        verify_without_session.events.insert(
            verify_index,
            TraceEvent {
                sequence: 3,
                kind: EventKind::SessionInterrupted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        cases.push(("verification outside active session", verify_without_session));

        for (label, trace) in cases {
            let value = serde_json::to_value(&trace).expect("mutated trace serializes");
            assert_eq!(
                inspect_trace(&trace),
                reference_codes_from_json(&value),
                "differential mismatch for {label}"
            );
        }
    }

    #[test]
    fn revocation_mutation_detects_post_revoke_render() {
        let mut trace = corpus().scenarios.remove(0);
        trace.events.push(TraceEvent {
            sequence: 5,
            kind: EventKind::BundleRevoked,
            bundle_id: Some("bundle-a".to_string()),
            bundle_state: None,
            digest_verified: None,
            resource_state: None,
            presented_as: None,
            sensitive_payload: None,
            consent_scoped: None,
        });
        trace.events.push(TraceEvent {
            sequence: 6,
            kind: EventKind::ProtocolRendered,
            bundle_id: Some("bundle-a".to_string()),
            bundle_state: None,
            digest_verified: None,
            resource_state: None,
            presented_as: None,
            sensitive_payload: None,
            consent_scoped: None,
        });
        assert_eq!(inspect_trace(&trace), vec!["render_after_revocation"]);
    }

    #[test]
    fn interruption_clears_verified_state() {
        let mut trace = corpus().scenarios.remove(0);
        trace.events.push(TraceEvent {
            sequence: 5,
            kind: EventKind::SessionInterrupted,
            bundle_id: None,
            bundle_state: None,
            digest_verified: None,
            resource_state: None,
            presented_as: None,
            sensitive_payload: None,
            consent_scoped: None,
        });
        trace.events.push(TraceEvent {
            sequence: 6,
            kind: EventKind::SessionResumed,
            bundle_id: None,
            bundle_state: None,
            digest_verified: None,
            resource_state: None,
            presented_as: None,
            sensitive_payload: None,
            consent_scoped: None,
        });
        trace.events.push(TraceEvent {
            sequence: 7,
            kind: EventKind::ProtocolRendered,
            bundle_id: Some("bundle-a".to_string()),
            bundle_state: None,
            digest_verified: None,
            resource_state: None,
            presented_as: None,
            sensitive_payload: None,
            consent_scoped: None,
        });
        assert_eq!(
            inspect_trace(&trace),
            vec!["render_without_verified_bundle"]
        );
    }

    #[test]
    fn duplicate_or_decreasing_sequences_are_rejected() {
        let mut trace = corpus().scenarios.remove(0);
        trace.events[1].sequence = trace.events[0].sequence;
        let mut value = corpus();
        value.scenarios = vec![trace];
        assert!(validate_corpus(&value).is_err());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value: serde_json::Value =
            serde_json::from_str(CORPUS_JSON).expect("fixture JSON must parse");
        value["unexpected_authority"] = serde_json::json!(true);
        assert!(serde_json::from_value::<TrajectoryCorpus>(value).is_err());
    }

    #[test]
    fn stale_resource_is_not_a_violation_when_marked_unconfirmed() {
        let trace = corpus()
            .scenarios
            .into_iter()
            .find(|trace| trace.trace_id == "FA-TRACE-011")
            .unwrap();
        assert!(inspect_trace(&trace).is_empty());
    }

    #[test]
    fn all_detected_codes_have_predeclared_meaning() {
        let corpus = corpus();
        for trace in &corpus.scenarios {
            for code in inspect_trace(trace) {
                assert!(is_known_violation_code(&code), "undeclared code {code}");
            }
        }
    }
    #[test]
    fn claim_ceiling_cannot_be_weakened_to_clinical_validation() {
        let mut value = corpus();
        value.claim_ceiling = "Clinically validated for patient care.".to_string();
        assert!(validate_corpus(&value).is_err());
    }

    #[test]
    fn clinical_use_authorization_remains_forbidden() {
        let mut value = corpus();
        value.clinical_use_authorized = true;
        assert!(validate_corpus(&value).is_err());
    }

    #[test]
    fn intermittent_connectivity_cannot_be_a_core_network_dependency() {
        let mut trace = corpus()
            .scenarios
            .into_iter()
            .find(|trace| trace.trace_id == "FA-TRACE-007")
            .expect("offline core-network counterexample exists");
        trace.connectivity = Connectivity::Intermittent;
        assert_eq!(
            inspect_trace(&trace),
            vec!["intermittent_core_network_dependency"]
        );
    }

    #[test]
    fn rendering_outside_active_session_is_rejected() {
        let mut value = corpus();
        let mut trace = value.scenarios.remove(0);
        trace.events.remove(0);
        trace.expected_violation_codes = vec!["render_without_active_session".to_string()];
        value.scenarios = vec![trace];
        validate_corpus(&value).expect("render without session must fail its frozen oracle");
    }

    #[test]
    fn resuming_session_without_interrupt_clears_prior_bundle_verification() {
        let mut trace = scenario("FA-TRACE-001");
        let render_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        trace.events[render_index].sequence += 1;
        trace.events.insert(
            render_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::SessionResumed,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        assert_eq!(
            inspect_trace(&trace),
            vec!["render_without_verified_bundle"]
        );
        let value = serde_json::to_value(&trace).expect("mutated trace serializes");
        assert_eq!(inspect_trace(&trace), reference_codes_from_json(&value));
    }

    #[test]
    fn starting_new_session_invalidates_prior_bundle_verification() {
        let mut value = corpus();
        let mut trace = value.scenarios.remove(0);
        let render_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        trace.events[render_index].sequence += 1;
        trace.events.insert(
            render_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::SessionStarted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        trace.expected_violation_codes = vec!["render_without_verified_bundle".to_string()];
        value.scenarios = vec![trace];
        validate_corpus(&value).expect("new session must require a fresh bundle verification");
    }

    #[test]
    fn rendering_different_bundle_from_verified_bundle_is_rejected() {
        let mut trace = scenario("FA-TRACE-001");
        let render = trace
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        render.bundle_id = Some("bundle-b".to_owned());
        assert_eq!(
            inspect_trace(&trace),
            vec!["render_without_verified_bundle"]
        );
    }

    #[test]
    fn failed_digest_verification_blocks_rendering() {
        let mut trace = scenario("FA-TRACE-001");
        let verify = trace
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::BundleVerified)
            .expect("verify event exists");
        verify.digest_verified = Some(false);
        assert_eq!(
            inspect_trace(&trace),
            vec![
                "invalid_bundle_verification",
                "render_without_verified_bundle"
            ]
        );
    }

    #[test]
    fn expired_bundle_cannot_be_verified_or_rendered() {
        let mut trace = scenario("FA-TRACE-001");
        let verify = trace
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::BundleVerified)
            .expect("verify event exists");
        verify.bundle_state = Some(BundleState::Expired);
        assert_eq!(
            inspect_trace(&trace),
            vec![
                "invalid_bundle_verification",
                "render_without_verified_bundle"
            ]
        );
    }

    #[test]
    fn non_sensitive_egress_does_not_require_sensitive_data_consent_scope() {
        let mut trace = scenario("FA-TRACE-009");
        let egress = trace
            .events
            .iter_mut()
            .find(|event| event.kind == EventKind::SensitiveDataEgress)
            .expect("egress event exists");
        egress.sensitive_payload = Some(false);
        assert!(inspect_trace(&trace).is_empty());
    }

    #[test]
    fn new_uncertainty_after_prior_preservation_is_not_lost_at_handoff() {
        let mut trace = scenario("FA-TRACE-012");
        let handoff_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::HandoffEmitted)
            .expect("handoff event exists");
        trace.events[handoff_index].sequence += 1;
        trace.events.insert(
            handoff_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::UncertaintyDetected,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        assert_eq!(
            inspect_trace(&trace),
            vec!["uncertainty_lost_at_handoff"]
        );
    }

    #[test]
    fn revocation_survives_new_session_and_remains_a_violation() {
        let mut trace = scenario("FA-TRACE-003");
        let render_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        trace.events[render_index].sequence += 1;
        trace.events.insert(
            render_index,
            TraceEvent {
                sequence: 5,
                kind: EventKind::SessionStarted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        assert_eq!(
            inspect_trace(&trace),
            vec!["render_after_revocation"]
        );
    }

    #[test]
    fn unknown_event_kind_is_rejected_by_schema_deserialization() {
        let mut value: serde_json::Value =
            serde_json::from_str(CORPUS_JSON).expect("fixture JSON must parse");
        value["scenarios"][0]["events"][0]["kind"] = serde_json::json!("unrecognized_event");
        assert!(serde_json::from_value::<TrajectoryCorpus>(value).is_err());
    }

    #[test]
    fn bundle_verification_requires_digest_metadata() {
        let mut value: serde_json::Value =
            serde_json::from_str(CORPUS_JSON).expect("fixture JSON must parse");
        let events = value["scenarios"][0]["events"]
            .as_array_mut()
            .expect("event list exists");
        let verify = events
            .iter_mut()
            .find(|event| event["kind"] == "bundle_verified")
            .expect("verify event exists");
        verify
            .as_object_mut()
            .expect("event must be an object")
            .remove("digest_verified");
        let decoded: TrajectoryCorpus =
            serde_json::from_value(value).expect("missing optional field parses as None");
        assert!(validate_corpus(&decoded).is_err());
    }

    #[test]
    fn starting_new_session_cannot_silently_drop_unpreserved_uncertainty() {
        let mut trace = scenario("FA-TRACE-012");
        let preserve_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::UncertaintyPreserved)
            .expect("preservation event exists");
        trace.events.remove(preserve_index);
        let handoff_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::HandoffEmitted)
            .expect("handoff event exists");
        trace.events[handoff_index].sequence += 1;
        trace.events.insert(
            handoff_index,
            TraceEvent {
                sequence: 4,
                kind: EventKind::SessionStarted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        assert_eq!(
            inspect_trace(&trace),
            vec!["uncertainty_lost_at_session_boundary", "uncertainty_lost_at_handoff"]
        );
        let value = serde_json::to_value(&trace).expect("mutated trace serializes");
        assert_eq!(inspect_trace(&trace), reference_codes_from_json(&value));
    }

    #[test]
    fn verification_outside_active_session_is_rejected() {
        let mut trace = scenario("FA-TRACE-001");
        let verify_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::BundleVerified)
            .expect("verification event exists");
        trace.events[verify_index].sequence += 1;
        let render_index = trace
            .events
            .iter()
            .position(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists");
        trace.events[render_index].sequence += 1;
        trace.events.insert(
            verify_index,
            TraceEvent {
                sequence: 3,
                kind: EventKind::SessionInterrupted,
                bundle_id: None,
                bundle_state: None,
                digest_verified: None,
                resource_state: None,
                presented_as: None,
                sensitive_payload: None,
                consent_scoped: None,
            },
        );
        let expected = vec![
            "invalid_bundle_verification",
            "render_without_active_session",
            "render_without_verified_bundle",
        ];
        assert_eq!(inspect_trace(&trace), expected);
        let value = serde_json::to_value(&trace).expect("mutated trace serializes");
        assert_eq!(inspect_trace(&trace), reference_codes_from_json(&value));
    }

}
