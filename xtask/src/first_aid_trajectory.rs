//! Synthetic event-trace audit for first-aid workflow process safety.
//!
//! This is a deterministic trace-contract checker, not a clinical protocol,
//! runtime authorization service, or demonstration of clinical safety.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
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
    ConnectivityChanged,
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
    /// Point-in-time connectivity update; required only for ConnectivityChanged.
    connectivity: Option<Connectivity>,
}

#[derive(Debug, Serialize)]
struct TrajectoryReceipt {
    receipt_version: &'static str,
    status: &'static str,
    corpus_sha256: String,
    evaluation_protocol_sha256: String,
    mutation_matrix_sha256: String,
    matrix_case_count: usize,
    matrix_frozen_trace_count: usize,
    matrix_mutation_case_count: usize,
    matrix_rust_runner_case_count: usize,
    matrix_workflow_gate_case_count: usize,
    state_model_sha256: String,
    state_model_spec_sha256: String,
    state_model_test_outcome: String,
    trajectory_test_outcome: String,
    model_production_edge_comparison_count: usize,
    policy_truth_table_comparison_count: usize,
    model_production_two_action_sequence_comparison_count: usize,
    model_production_seeded_long_horizon_comparison_count: usize,
    model_production_seeded_long_horizon_seed: String,
    model_production_seeded_long_horizon_max_actions: usize,
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
    unique_violation_class_count: usize,
    violation_occurrences_by_code: BTreeMap<String, usize>,
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
    let matrix_value: serde_json::Value =
        serde_json::from_slice(&mutation_matrix).context("parse mutation matrix receipt inputs")?;
    let matrix_cases = matrix_value["cases"]
        .as_array()
        .context("mutation matrix must contain a cases array")?;
    let matrix_case_count = matrix_cases.len();
    let matrix_frozen_trace_count = matrix_cases
        .iter()
        .filter(|case| case["mode"] == "frozen_trace")
        .count();
    let matrix_mutation_case_count = matrix_cases
        .iter()
        .filter(|case| case["mode"] == "mutation_or_control")
        .count();
    let matrix_rust_runner_case_count = matrix_cases
        .iter()
        .filter(|case| case["execution_target"] == "rust_matrix_runner")
        .count();
    let matrix_workflow_gate_case_count = matrix_cases
        .iter()
        .filter(|case| case["execution_target"] == "workflow_gate")
        .count();
    anyhow::ensure!(
        matrix_case_count == 44
            && matrix_frozen_trace_count == 12
            && matrix_mutation_case_count == 32
            && matrix_rust_runner_case_count == 30
            && matrix_workflow_gate_case_count == 2,
        "mutation matrix case inventory changed without reviewing receipt targets"
    );
    anyhow::ensure!(
        matrix_value["count_summary"]["case_count"].as_u64() == Some(matrix_case_count as u64)
            && matrix_value["count_summary"]["frozen_trace_count"].as_u64()
                == Some(matrix_frozen_trace_count as u64)
            && matrix_value["count_summary"]["mutation_case_count"].as_u64()
                == Some(matrix_mutation_case_count as u64),
        "mutation matrix summary disagrees with case inventory"
    );
    let state_model = fs::read(root.join("xtask/src/first_aid_model.rs"))
        .context("read finite-state model for receipt identity")?;
    let state_model_digest = sha256_hex(&state_model);
    let state_model_spec = fs::read(root.join("docs/research/first-aid/STATE_MODEL_V0.md"))
        .context("read state model specification for receipt identity")?;
    let state_model_spec_digest = sha256_hex(&state_model_spec);
    let model_production_edge_comparison_count =
        std::env::var("MODEL_PRODUCTION_EDGE_COMPARISON_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    let policy_truth_table_comparison_count =
        std::env::var("POLICY_TRUTH_TABLE_CASE_COMPARISON_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    let model_production_two_action_sequence_comparison_count =
        std::env::var("MODEL_PRODUCTION_TWO_ACTION_SEQUENCE_COMPARISON_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    let model_production_seeded_long_horizon_comparison_count =
        std::env::var("MODEL_PRODUCTION_SEEDED_LONG_HORIZON_COMPARISON_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    let model_production_seeded_long_horizon_seed = std::env::var(
        "MODEL_PRODUCTION_SEEDED_LONG_HORIZON_SEED",
    )
    .unwrap_or_else(|_| "not-provided".to_string());
    let model_production_seeded_long_horizon_max_actions =
        std::env::var("MODEL_PRODUCTION_SEEDED_LONG_HORIZON_MAX_ACTIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    let cargo_lock = fs::read(root.join("Cargo.lock"))
        .context("read Cargo.lock for dependency-resolution identity")?;
    let cargo_lock_digest = sha256_hex(&cargo_lock);
    let rustc_version =
        std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-provided".to_string());
    let cargo_version =
        std::env::var("CARGO_VERSION").unwrap_or_else(|_| "not-provided".to_string());
    let validator_source = fs::read(root.join("xtask/src/first_aid_trajectory.rs"))
        .context("read trajectory validator source for receipt identity")?;
    let entrypoint_source = fs::read(root.join("xtask/src/main.rs"))
        .context("read xtask entrypoint source for receipt identity")?;
    let executable_path =
        std::env::current_exe().context("resolve trajectory validator executable")?;
    let executable_bytes = fs::read(&executable_path)
        .with_context(|| format!("read validator executable {}", executable_path.display()))?;

    let results: Vec<Vec<String>> = corpus.scenarios.iter().map(inspect_trace).collect();
    let mut violation_occurrences_by_code = BTreeMap::new();
    for code in results.iter().flatten() {
        *violation_occurrences_by_code
            .entry(code.clone())
            .or_insert(0) += 1;
    }
    let receipt = TrajectoryReceipt {
        receipt_version: "first-aid-trajectory-receipt-v1.1",
        status: "synthetic_trajectory_contract_passed",
        corpus_sha256: corpus_digest,
        evaluation_protocol_sha256: evaluation_protocol_digest,
        mutation_matrix_sha256: mutation_matrix_digest,
        matrix_case_count,
        matrix_frozen_trace_count,
        matrix_mutation_case_count,
        matrix_rust_runner_case_count,
        matrix_workflow_gate_case_count,
        state_model_sha256: state_model_digest,
        state_model_spec_sha256: state_model_spec_digest,
        state_model_test_outcome: std::env::var("STATE_MODEL_TEST_OUTCOME")
            .unwrap_or_else(|_| "not-provided".to_string()),
        trajectory_test_outcome: std::env::var("TRAJECTORY_TEST_OUTCOME")
            .unwrap_or_else(|_| "not-provided".to_string()),
        model_production_edge_comparison_count,
        policy_truth_table_comparison_count,
        model_production_two_action_sequence_comparison_count,
        model_production_seeded_long_horizon_comparison_count,
        model_production_seeded_long_horizon_seed,
        model_production_seeded_long_horizon_max_actions,
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
        unique_violation_class_count: violation_occurrences_by_code.len(),
        violation_occurrences_by_code,
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
        for code in &trace.expected_violation_codes {
            if !is_known_violation_code(code) {
                bail!("{}: unknown violation code {}", trace.trace_id, code);
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
    if event.kind == EventKind::ConnectivityChanged && event.connectivity.is_none() {
        bail!(
            "{} event {} requires connectivity for connectivity_changed",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind != EventKind::ConnectivityChanged && event.connectivity.is_some() {
        bail!(
            "{} event {} may carry connectivity only for connectivity_changed",
            trace.trace_id,
            event.sequence
        );
    }

    let permits_bundle_id = matches!(
        event.kind,
        EventKind::BundleLoaded
            | EventKind::BundleVerified
            | EventKind::ProtocolRendered
            | EventKind::BundleRevoked
    );
    if !permits_bundle_id && event.bundle_id.is_some() {
        bail!(
            "{} event {} may carry bundle_id only for bundle lifecycle events",
            trace.trace_id,
            event.sequence
        );
    }
    if !matches!(
        event.kind,
        EventKind::BundleLoaded | EventKind::BundleVerified | EventKind::ProtocolRendered
    ) && event.bundle_state.is_some()
    {
        bail!(
            "{} event {} may carry bundle_state only for load, verification, or render events",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind != EventKind::BundleVerified && event.digest_verified.is_some() {
        bail!(
            "{} event {} may carry digest_verified only for bundle_verified",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind != EventKind::ResourcePresented
        && (event.resource_state.is_some() || event.presented_as.is_some())
    {
        bail!(
            "{} event {} may carry resource fields only for resource_presented",
            trace.trace_id,
            event.sequence
        );
    }
    if event.kind != EventKind::SensitiveDataEgress
        && (event.sensitive_payload.is_some() || event.consent_scoped.is_some())
    {
        bail!(
            "{} event {} may carry sensitive-data fields only for sensitive_data_egress",
            trace.trace_id,
            event.sequence
        );
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn inspect_trace_json(trace_json: &serde_json::Value) -> Result<Vec<String>> {
    let trace: TrajectoryScenario =
        serde_json::from_value(trace_json.clone()).context("parse generated model trace")?;
    let mut previous_sequence = None;
    for event in &trace.events {
        if previous_sequence.is_some_and(|previous| event.sequence <= previous) {
            bail!("generated model trace event sequence must increase strictly");
        }
        previous_sequence = Some(event.sequence);
        validate_event_fields(&trace, event)?;
    }
    Ok(inspect_trace(&trace))
}

fn inspect_trace(trace: &TrajectoryScenario) -> Vec<String> {
    let mut violations = Vec::new();
    let mut session_active = false;
    let mut active_bundle: Option<String> = None;
    let mut verified_bundle: Option<String> = None;
    let mut revoked_bundles: HashSet<String> = HashSet::new();
    let mut uncertainty_pending = false;
    let mut uncertainty_preserved = false;
    let mut current_connectivity = trace.connectivity;

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
            EventKind::ConnectivityChanged => {
                current_connectivity = event.connectivity.unwrap_or(current_connectivity);
            }
            EventKind::NetworkRequiredForCore => match current_connectivity {
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
        let mut connectivity = trace["connectivity"].as_str().unwrap_or_default();
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
                    if matches!(state, "stale" | "unknown") && event["presented_as"] == "confirmed"
                    {
                        codes.push("stale_resource_claimed_confirmed".to_owned());
                    }
                }
                "connectivity_changed" => {
                    if let Some(next) = event["connectivity"].as_str() {
                        connectivity = next;
                    }
                }
                "network_required_for_core" => match connectivity {
                    "offline" => codes.push("offline_core_network_dependency".to_owned()),
                    "intermittent" => codes.push("intermittent_core_network_dependency".to_owned()),
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
        trace.expected_violation_codes = vec![
            "invalid_bundle_verification".to_owned(),
            "render_without_active_session".to_owned(),
            "render_without_verified_bundle".to_owned(),
        ];
        let value = serde_json::to_value(&trace).expect("mutated trace serializes");
        assert_eq!(inspect_trace(&trace), reference_codes_from_json(&value));
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
            connectivity: None,
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
                connectivity: None,
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
                connectivity: None,
            },
        );
        cases.push((
            "unresolved uncertainty across new session",
            uncertain_session_boundary,
        ));

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
                connectivity: None,
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
                connectivity: None,
            },
        );
        cases.push((
            "verification outside active session",
            verify_without_session,
        ));

        let mut repeated_violations = scenario("FA-TRACE-002");
        let mut repeated_render = repeated_violations
            .events
            .iter()
            .find(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists")
            .clone();
        repeated_render.sequence = repeated_violations
            .events
            .last()
            .expect("events exist")
            .sequence
            + 1;
        repeated_violations.events.push(repeated_render);
        cases.push(("repeated violation occurrences", repeated_violations));

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
            connectivity: None,
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
            connectivity: None,
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
            connectivity: None,
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
            connectivity: None,
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
            connectivity: None,
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
    fn connectivity_changes_mid_trace_are_enforced_at_the_next_network_action() {
        let mut trace = scenario("FA-TRACE-007");
        trace.connectivity = Connectivity::Online;
        let mut changed = bare_event(EventKind::ConnectivityChanged);
        changed.connectivity = Some(Connectivity::Intermittent);
        insert_before(&mut trace, EventKind::NetworkRequiredForCore, changed);
        trace.expected_violation_codes =
            vec!["intermittent_core_network_dependency".to_owned()];

        let mut changed_corpus = corpus();
        changed_corpus.scenarios = vec![trace.clone()];
        validate_corpus(&changed_corpus)
            .expect("dynamic connectivity transition must satisfy the expected contract");
        assert_eq!(
            inspect_trace(&trace),
            vec!["intermittent_core_network_dependency"]
        );
        let json = serde_json::json!({
            "trace_id": trace.trace_id,
            "description": trace.description,
            "criticality": "critical",
            "connectivity": "online",
            "events": [
                {"sequence": 1, "kind": "session_started"},
                {"sequence": 2, "kind": "connectivity_changed", "connectivity": "intermittent"},
                {"sequence": 3, "kind": "network_required_for_core"}
            ],
            "expected_violation_codes": ["intermittent_core_network_dependency"]
        });
        assert_eq!(
            inspect_trace_json(&json).expect("dynamic trace schema is valid"),
            vec!["intermittent_core_network_dependency"]
        );
    }

    #[test]
    fn connectivity_recovery_is_scoped_to_each_event_time() {
        let mut trace = scenario("FA-TRACE-007");
        trace.connectivity = Connectivity::Online;
        let mut events = vec![bare_event(EventKind::SessionStarted)];
        events.push(bare_event(EventKind::NetworkRequiredForCore));
        let mut offline = bare_event(EventKind::ConnectivityChanged);
        offline.connectivity = Some(Connectivity::Offline);
        events.push(offline);
        events.push(bare_event(EventKind::NetworkRequiredForCore));
        let mut online = bare_event(EventKind::ConnectivityChanged);
        online.connectivity = Some(Connectivity::Online);
        events.push(online);
        events.push(bare_event(EventKind::NetworkRequiredForCore));
        trace.events = events;
        resequence(&mut trace);
        trace.expected_violation_codes = vec!["offline_core_network_dependency".to_owned()];
        let mut changed_corpus = corpus();
        changed_corpus.scenarios = vec![trace.clone()];
        validate_corpus(&changed_corpus)
            .expect("connectivity recovery trace must satisfy the ordered contract");
        assert_eq!(
            inspect_trace(&trace),
            vec!["offline_core_network_dependency"]
        );
    }

    #[test]
    fn event_fields_are_scoped_to_their_event_kind() {
        let mut value: serde_json::Value =
            serde_json::from_str(CORPUS_JSON).expect("fixture JSON must parse");
        value["scenarios"][0]["events"][0]["sensitive_payload"] = serde_json::json!(true);
        let changed: TrajectoryCorpus =
            serde_json::from_value(value).expect("known optional field is syntactically valid");
        assert!(
            validate_corpus(&changed).is_err(),
            "unrelated metadata must be rejected rather than silently ignored"
        );
    }

    #[test]
    fn connectivity_changed_metadata_is_required_and_event_scoped() {
        let trace = scenario("FA-TRACE-007");
        let mut changed = bare_event(EventKind::ConnectivityChanged);
        changed.sequence = 2;
        assert!(validate_event_fields(&trace, &changed).is_err());
        changed.connectivity = Some(Connectivity::Online);
        assert!(validate_event_fields(&trace, &changed).is_ok());

        let mut network_action = bare_event(EventKind::NetworkRequiredForCore);
        network_action.connectivity = Some(Connectivity::Online);
        assert!(validate_event_fields(&trace, &network_action).is_err());
    }

    #[test]
    fn rendering_outside_active_session_is_rejected() {
        let mut value = corpus();
        let mut trace = value.scenarios.remove(0);
        trace.events.remove(0);
        trace.expected_violation_codes = vec![
            "invalid_bundle_verification".to_string(),
            "render_without_active_session".to_string(),
            "render_without_verified_bundle".to_string(),
        ];
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
                connectivity: None,
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
                connectivity: None,
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
                connectivity: None,
            },
        );
        assert_eq!(inspect_trace(&trace), vec!["uncertainty_lost_at_handoff"]);
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
                connectivity: None,
            },
        );
        assert_eq!(inspect_trace(&trace), vec!["render_after_revocation"]);
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
                connectivity: None,
            },
        );
        assert_eq!(
            inspect_trace(&trace),
            vec![
                "uncertainty_lost_at_session_boundary",
                "uncertainty_lost_at_handoff"
            ]
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
                connectivity: None,
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

    #[test]
    fn repeated_violation_codes_count_each_event_occurrence() {
        let mut trace = scenario("FA-TRACE-002");
        let mut repeated_render = trace
            .events
            .iter()
            .find(|event| event.kind == EventKind::ProtocolRendered)
            .expect("render event exists")
            .clone();
        repeated_render.sequence = trace.events.last().expect("events exist").sequence + 1;
        trace.events.push(repeated_render);
        trace.expected_violation_codes = vec![
            "render_without_verified_bundle".to_owned(),
            "render_without_verified_bundle".to_owned(),
        ];
        let corpus = TrajectoryCorpus {
            schema_version: CORPUS_VERSION.to_owned(),
            status: "synthetic_process_contract_tests_only".to_owned(),
            clinical_use_authorized: false,
            claim_ceiling: REQUIRED_CLAIM_CEILING.to_owned(),
            notes: None,
            scenarios: vec![trace.clone()],
        };
        validate_corpus(&corpus).expect("repeated event-level codes are valid expected outcomes");
        let value = serde_json::to_value(&trace).expect("mutated trace serializes");
        assert_eq!(inspect_trace(&trace), reference_codes_from_json(&value));
        assert_eq!(inspect_trace(&trace).len(), 2);
    }

    #[derive(Debug, Deserialize)]
    struct MutationMatrix {
        cases: Vec<MutationCase>,
    }

    #[derive(Debug, Deserialize)]
    struct MutationCase {
        case_id: String,
        mode: String,
        source_trace_id: Option<String>,
        operator_id: Option<String>,
        execution_target: Option<String>,
        expected: MutationExpected,
    }

    #[derive(Debug, Deserialize)]
    struct MutationExpected {
        outcome: String,
        violation_codes: Vec<String>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct MutationActual {
        outcome: String,
        violation_codes: Vec<String>,
    }

    fn actual_evaluate(trace: &TrajectoryScenario) -> MutationActual {
        MutationActual {
            outcome: "evaluate".to_owned(),
            violation_codes: inspect_trace(trace),
        }
    }

    fn actual_reject() -> MutationActual {
        MutationActual {
            outcome: "reject".to_owned(),
            violation_codes: Vec::new(),
        }
    }

    fn bare_event(kind: EventKind) -> TraceEvent {
        TraceEvent {
            sequence: 0,
            kind,
            bundle_id: None,
            bundle_state: None,
            digest_verified: None,
            resource_state: None,
            presented_as: None,
            sensitive_payload: None,
            consent_scoped: None,
            connectivity: None,
        }
    }

    fn resequence(trace: &mut TrajectoryScenario) {
        for (index, event) in trace.events.iter_mut().enumerate() {
            event.sequence = u32::try_from(index + 1).expect("fixture event count fits u32");
        }
    }

    fn insert_before(trace: &mut TrajectoryScenario, target: EventKind, inserted: TraceEvent) {
        let index = trace
            .events
            .iter()
            .position(|event| event.kind == target)
            .unwrap_or_else(|| panic!("target event {target:?} not found in {}", trace.trace_id));
        trace.events.insert(index, inserted);
        resequence(trace);
    }

    fn matrix_case_actual(case: &MutationCase) -> MutationActual {
        let operator = case.operator_id.as_deref().unwrap_or_else(|| {
            panic!(
                "{} must have an operator_id for the Rust matrix runner",
                case.case_id
            )
        });

        match operator {
            "weaken_claim_ceiling" => {
                let mut changed = corpus();
                changed.claim_ceiling = "Clinically validated for patient care.".to_owned();
                assert!(
                    validate_corpus(&changed).is_err(),
                    "claim ceiling mutation must reject"
                );
                actual_reject()
            }
            "enable_clinical_use" => {
                let mut changed = corpus();
                changed.clinical_use_authorized = true;
                assert!(
                    validate_corpus(&changed).is_err(),
                    "clinical-use mutation must reject"
                );
                actual_reject()
            }
            "unknown_corpus_field" => {
                let mut value: serde_json::Value =
                    serde_json::from_str(CORPUS_JSON).expect("frozen corpus JSON parses");
                value["unexpected_authority"] = serde_json::json!(true);
                assert!(
                    serde_json::from_value::<TrajectoryCorpus>(value).is_err(),
                    "unknown root fields must reject"
                );
                actual_reject()
            }
            "unexpected_event_metadata" => {
                let mut value: serde_json::Value =
                    serde_json::from_str(CORPUS_JSON).expect("frozen corpus JSON parses");
                value["scenarios"][0]["events"][0]["sensitive_payload"] = serde_json::json!(true);
                let changed: TrajectoryCorpus = serde_json::from_value(value)
                    .expect("known optional field is syntactically valid");
                assert!(
                    validate_corpus(&changed).is_err(),
                    "event metadata must be scoped to the declared event kind"
                );
                actual_reject()
            }
            "unexpected_connectivity_metadata" => {
                let mut trace = scenario("FA-TRACE-007");
                trace
                    .events
                    .iter_mut()
                    .find(|event| event.kind == EventKind::NetworkRequiredForCore)
                    .expect("core-network action exists")
                    .connectivity = Some(Connectivity::Online);
                let mut changed = corpus();
                changed.scenarios = vec![trace];
                assert!(
                    validate_corpus(&changed).is_err(),
                    "connectivity metadata must be event-scoped"
                );
                actual_reject()
            }
            "tampered_corpus_digest" => {
                let mut bytes = CORPUS_JSON.as_bytes().to_vec();
                bytes[0] ^= 1;
                assert!(
                    checked_corpus_digest(&bytes).is_err(),
                    "single-byte corpus mutation must reject"
                );
                actual_reject()
            }
            "unknown_event_kind" => {
                let mut value: serde_json::Value =
                    serde_json::from_str(CORPUS_JSON).expect("frozen corpus JSON parses");
                value["scenarios"][0]["events"][0]["kind"] =
                    serde_json::json!("unrecognized_event");
                assert!(
                    serde_json::from_value::<TrajectoryCorpus>(value).is_err(),
                    "unknown event kinds must reject"
                );
                actual_reject()
            }
            "missing_digest_metadata" => {
                let mut value: serde_json::Value =
                    serde_json::from_str(CORPUS_JSON).expect("frozen corpus JSON parses");
                let events = value["scenarios"][0]["events"]
                    .as_array_mut()
                    .expect("event list exists");
                let verify = events
                    .iter_mut()
                    .find(|event| event["kind"] == "bundle_verified")
                    .expect("verification event exists");
                verify
                    .as_object_mut()
                    .expect("event object exists")
                    .remove("digest_verified");
                let changed: TrajectoryCorpus =
                    serde_json::from_value(value).expect("optional field omission deserializes");
                assert!(
                    validate_corpus(&changed).is_err(),
                    "missing digest metadata must reject"
                );
                actual_reject()
            }
            "duplicate_event_sequence" => {
                let mut trace = scenario(
                    case.source_trace_id
                        .as_deref()
                        .expect("source trace is required"),
                );
                trace.events[1].sequence = trace.events[0].sequence;
                let mut changed = corpus();
                changed.scenarios = vec![trace];
                assert!(
                    validate_corpus(&changed).is_err(),
                    "duplicate event sequence must reject"
                );
                actual_reject()
            }
            operator => {
                let source_id = case
                    .source_trace_id
                    .as_deref()
                    .unwrap_or_else(|| panic!("{} requires source_trace_id", case.case_id));
                let mut trace = scenario(source_id);
                match operator {
                    "remove_session_start" => {
                        let index = trace
                            .events
                            .iter()
                            .position(|event| event.kind == EventKind::SessionStarted)
                            .expect("session-start event exists");
                        trace.events.remove(index);
                        resequence(&mut trace);
                    }
                    "new_session_before_render" => {
                        insert_before(
                            &mut trace,
                            EventKind::ProtocolRendered,
                            bare_event(EventKind::SessionStarted),
                        );
                    }
                    "intermittent_network" => trace.connectivity = Connectivity::Intermittent,
                    "connectivity_change_before_network_action" => {
                        trace.connectivity = Connectivity::Online;
                        let mut change = bare_event(EventKind::ConnectivityChanged);
                        change.connectivity = Some(Connectivity::Intermittent);
                        insert_before(&mut trace, EventKind::NetworkRequiredForCore, change);
                    }
                    "connectivity_recovery_control" => {
                        trace.connectivity = Connectivity::Online;
                        let mut events = vec![
                            bare_event(EventKind::SessionStarted),
                            bare_event(EventKind::NetworkRequiredForCore),
                        ];
                        let mut offline = bare_event(EventKind::ConnectivityChanged);
                        offline.connectivity = Some(Connectivity::Offline);
                        events.push(offline);
                        events.push(bare_event(EventKind::NetworkRequiredForCore));
                        let mut online = bare_event(EventKind::ConnectivityChanged);
                        online.connectivity = Some(Connectivity::Online);
                        events.push(online);
                        events.push(bare_event(EventKind::NetworkRequiredForCore));
                        trace.events = events;
                        resequence(&mut trace);
                        trace.expected_violation_codes =
                            vec!["offline_core_network_dependency".to_owned()];
                    }
                    "render_after_revocation" => {
                        let mut revoke = bare_event(EventKind::BundleRevoked);
                        revoke.bundle_id = Some("bundle-a".to_owned());
                        insert_before(&mut trace, EventKind::ProtocolRendered, revoke);
                    }
                    "resume_after_interruption" => {
                        insert_before(
                            &mut trace,
                            EventKind::ProtocolRendered,
                            bare_event(EventKind::SessionResumed),
                        );
                        insert_before(
                            &mut trace,
                            EventKind::SessionResumed,
                            bare_event(EventKind::SessionInterrupted),
                        );
                    }
                    "stale_unconfirmed_control"
                    | "preserved_uncertainty_control"
                    | "offline_network_control"
                    | "sensitive_egress_control" => {}
                    "non_sensitive_egress" => {
                        trace
                            .events
                            .iter_mut()
                            .find(|event| event.kind == EventKind::SensitiveDataEgress)
                            .expect("egress event exists")
                            .sensitive_payload = Some(false);
                    }
                    "wrong_bundle_identity" => {
                        trace
                            .events
                            .iter_mut()
                            .find(|event| event.kind == EventKind::ProtocolRendered)
                            .expect("render event exists")
                            .bundle_id = Some("bundle-other".to_owned());
                    }
                    "false_digest" => {
                        trace
                            .events
                            .iter_mut()
                            .find(|event| event.kind == EventKind::BundleVerified)
                            .expect("verification event exists")
                            .digest_verified = Some(false);
                    }
                    "expired_bundle" => {
                        trace
                            .events
                            .iter_mut()
                            .find(|event| event.kind == EventKind::BundleVerified)
                            .expect("verification event exists")
                            .bundle_state = Some(BundleState::Expired);
                    }
                    "uncertainty_reentered" => {
                        insert_before(
                            &mut trace,
                            EventKind::HandoffEmitted,
                            bare_event(EventKind::UncertaintyDetected),
                        );
                    }
                    "revocation_new_session" => {
                        insert_before(
                            &mut trace,
                            EventKind::ProtocolRendered,
                            bare_event(EventKind::SessionStarted),
                        );
                    }
                    "uncertainty_session_boundary" => {
                        let index = trace
                            .events
                            .iter()
                            .position(|event| event.kind == EventKind::UncertaintyPreserved)
                            .expect("uncertainty preservation event exists");
                        trace.events.remove(index);
                        insert_before(
                            &mut trace,
                            EventKind::HandoffEmitted,
                            bare_event(EventKind::SessionStarted),
                        );
                    }
                    "resume_without_interrupt" => {
                        insert_before(
                            &mut trace,
                            EventKind::ProtocolRendered,
                            bare_event(EventKind::SessionResumed),
                        );
                    }
                    "verify_outside_session" => {
                        insert_before(
                            &mut trace,
                            EventKind::BundleVerified,
                            bare_event(EventKind::SessionInterrupted),
                        );
                    }
                    "repeat_unverified_render" => {
                        let mut render = trace
                            .events
                            .iter()
                            .find(|event| event.kind == EventKind::ProtocolRendered)
                            .expect("render event exists")
                            .clone();
                        render.sequence = 0;
                        trace.events.push(render);
                        resequence(&mut trace);
                    }
                    unknown => panic!("unknown matrix operator_id {unknown:?}"),
                }
                actual_evaluate(&trace)
            }
        }
    }

    #[test]
    fn executable_mutation_matrix_matches_each_declared_outcome() {
        const MATRIX_JSON: &str =
            include_str!("../../docs/research/first-aid/mutation-matrix-v0.json");
        let matrix: MutationMatrix =
            serde_json::from_str(MATRIX_JSON).expect("mutation matrix must deserialize");
        let mut frozen_count = 0usize;
        let mut mutation_count = 0usize;
        let mut workflow_gate_count = 0usize;
        let mut rejection_count = 0usize;

        for case in &matrix.cases {
            let actual = match case.mode.as_str() {
                "frozen_trace" => {
                    frozen_count += 1;
                    let trace = scenario(
                        case.source_trace_id
                            .as_deref()
                            .expect("frozen trace case requires source_trace_id"),
                    );
                    actual_evaluate(&trace)
                }
                "mutation_or_control"
                    if case.execution_target.as_deref() == Some("workflow_gate") =>
                {
                    workflow_gate_count += 1;
                    continue;
                }
                "mutation_or_control"
                    if case.execution_target.as_deref() == Some("rust_matrix_runner") =>
                {
                    mutation_count += 1;
                    matrix_case_actual(case)
                }
                other => panic!(
                    "unsupported matrix mode/target for {}: {other}",
                    case.case_id
                ),
            };
            let expected = MutationActual {
                outcome: case.expected.outcome.clone(),
                violation_codes: case.expected.violation_codes.clone(),
            };
            if actual.outcome == "reject" {
                rejection_count += 1;
            }
            assert_eq!(actual, expected, "matrix case {}", case.case_id);
        }

        assert_eq!(frozen_count, 12, "frozen trace case count changed");
        assert_eq!(mutation_count, 30, "Rust matrix-runner case count changed");
        assert_eq!(workflow_gate_count, 2, "workflow gate case count changed");
        assert_eq!(
            rejection_count, 8,
            "fail-closed rejection case count changed"
        );
        assert_eq!(
            frozen_count + mutation_count + workflow_gate_count,
            matrix.cases.len(),
            "every declared matrix case must be accounted for"
        );
        println!(
            "mutation_matrix_cases={} frozen_traces={} rust_runner_cases={} workflow_gate_cases={} rejection_cases={}",
            matrix.cases.len(),
            frozen_count,
            mutation_count,
            workflow_gate_count,
            rejection_count
        );
    }
}
