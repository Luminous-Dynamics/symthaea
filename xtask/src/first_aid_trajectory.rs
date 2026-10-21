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

pub fn run(root: &Path, corpus_arg: &Path, out_arg: Option<&Path>) -> Result<()> {
    let corpus_path = resolve(root, corpus_arg);
    let corpus_bytes = fs::read(&corpus_path)
        .with_context(|| format!("read trajectory corpus {}", corpus_path.display()))?;
    let corpus_digest = sha256_hex(&corpus_bytes);
    if corpus_digest != EXPECTED_CORPUS_SHA256 {
        bail!(
            "trajectory corpus digest mismatch: expected {}, got {}",
            EXPECTED_CORPUS_SHA256,
            corpus_digest
        );
    }
    let corpus_text =
        std::str::from_utf8(&corpus_bytes).context("trajectory corpus must be valid UTF-8")?;
    let corpus: TrajectoryCorpus =
        serde_json::from_str(corpus_text).context("parse trajectory corpus JSON")?;
    validate_corpus(&corpus)?;

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
    if corpus.claim_ceiling.trim().is_empty() {
        bail!("trajectory corpus must declare a claim ceiling");
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
                let is_valid = active_bundle.as_deref() == Some(bundle_id)
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
                verified_bundle = None;
            }
            EventKind::SessionResumed => {}
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
            EventKind::NetworkRequiredForCore => {
                if trace.connectivity == Connectivity::Offline {
                    violations.push("offline_core_network_dependency".to_string());
                }
            }
            EventKind::SensitiveDataEgress => {
                if event.sensitive_payload == Some(true) && event.consent_scoped != Some(true) {
                    violations.push("sensitive_data_egress_without_consent".to_string());
                }
            }
            EventKind::SessionStarted => {}
        }
    }
    violations
}

fn is_known_violation_code(code: &str) -> bool {
    matches!(
        code,
        "invalid_bundle_verification"
            | "render_after_revocation"
            | "render_without_verified_bundle"
            | "unknown_normalized"
            | "uncertainty_lost_at_handoff"
            | "stale_resource_claimed_confirmed"
            | "offline_core_network_dependency"
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

    #[test]
    fn checked_in_trajectory_corpus_has_pinned_digest() {
        assert_eq!(sha256_hex(CORPUS_JSON.as_bytes()), EXPECTED_CORPUS_SHA256);
        assert_eq!(corpus().scenarios.len(), 12);
    }

    #[test]
    fn independent_expected_violation_codes_match() {
        validate_corpus(&corpus()).expect("frozen trace oracles must match");
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
}
