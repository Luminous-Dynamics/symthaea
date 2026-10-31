//! Fail-closed validator for BASALT run-level coverage envelopes.
//!
//! This checks that a receipt's completeness claim is internally consistent
//! with a separately pinned candidate profile. It cannot prove that events
//! actually occurred, authenticate the observer, or qualify the profile.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CHANNELS: usize = 64;
const MAX_EVENT_KINDS: usize = 256;
const MAX_ROOTS: usize = 1024;

#[derive(Debug, Serialize)]
pub struct CoverageAuditReport {
    pub report_version: u32,
    pub profile_digest: String,
    pub receipt_digest: String,
    pub profile_valid: bool,
    pub receipt_well_formed: bool,
    pub profile_match: bool,
    pub coverage_claim_consistent: bool,
    pub coverage_complete_within_profile: bool,
    pub overstated_claim: bool,
    pub qualification_eligible: bool,
    pub profile_qualified: bool,
    pub receipt_authenticity_established: bool,
    pub counts: CoverageCounts,
    pub errors: Vec<CoverageError>,
}

#[derive(Debug, Default, Serialize)]
pub struct CoverageCounts {
    pub required_channels: usize,
    pub assessed_channels: usize,
    pub required_event_kinds: usize,
    pub observed_event_kinds: usize,
    pub unresolved_event_kinds: usize,
    pub excluded_event_kinds: usize,
    pub active_required_controls: usize,
    pub monitored_roots: usize,
}

#[derive(Debug, Serialize)]
pub struct CoverageError {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    format_version: u32,
    suite_id: String,
    profile_id: String,
    profile_status: String,
    platform: PlatformConstraints,
    allowed_controls: Vec<String>,
    required_controls: Vec<String>,
    channels: Vec<ProfileChannel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlatformConstraints {
    os: String,
    architectures: Vec<String>,
    require_kernel_build_id: bool,
    require_kernel_btf_digest: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileChannel {
    channel_id: String,
    required_event_kinds: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    format_version: u32,
    suite_id: String,
    run_id: String,
    profile_id: String,
    profile_digest: String,
    observer_binary_digest: String,
    runner_binary_digest: String,
    platform: RuntimePlatform,
    controls_active: Vec<String>,
    capture: CaptureEnvelope,
    health: HealthCounters,
    channels: Vec<ChannelAssessment>,
    claim: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimePlatform {
    os: String,
    architecture: String,
    kernel_release: String,
    kernel_build_id: Option<String>,
    kernel_btf_digest: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureEnvelope {
    start_monotonic_ns: u64,
    end_monotonic_ns: u64,
    root_pid: u32,
    process_tree_closed: bool,
    monitored_roots: Vec<String>,
    snapshot_consistency: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HealthCounters {
    event_count: u64,
    dropped_event_count: u64,
    ring_buffer_overflow_count: u64,
    observer_restart_count: u64,
    unmonitored_process_count: u64,
    root_escape_count: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChannelAssessment {
    channel_id: String,
    events: Vec<EventAssessment>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EventAssessment {
    event_kind: String,
    status: String,
    evidence_digest: Option<String>,
    reason: Option<String>,
}

pub fn audit_files(profile_path: &Path, receipt_path: &Path) -> anyhow::Result<CoverageAuditReport> {
    let profile_bytes = read_bounded(profile_path)?;
    let receipt_bytes = read_bounded(receipt_path)?;
    Ok(audit_bytes(&profile_bytes, &receipt_bytes))
}

pub fn audit_bytes(profile_bytes: &[u8], receipt_bytes: &[u8]) -> CoverageAuditReport {
    let mut report = CoverageAuditReport {
        report_version: 1,
        profile_digest: sha256_tagged(profile_bytes),
        receipt_digest: sha256_tagged(receipt_bytes),
        profile_valid: false,
        receipt_well_formed: false,
        profile_match: false,
        coverage_claim_consistent: false,
        coverage_complete_within_profile: false,
        overstated_claim: false,
        qualification_eligible: false,
        profile_qualified: false,
        receipt_authenticity_established: false,
        counts: CoverageCounts::default(),
        errors: Vec::new(),
    };

    if profile_bytes.len() as u64 > MAX_INPUT_BYTES || receipt_bytes.len() as u64 > MAX_INPUT_BYTES {
        push_error(&mut report, "E_INPUT_TOO_LARGE", "$", "profile and receipt are each limited to 4 MiB");
        return report;
    }

    let profile: Profile = match serde_json::from_slice(profile_bytes) {
        Ok(value) => value,
        Err(error) => {
            push_error(&mut report, "E_PROFILE_JSON", "profile", &format!("invalid or non-strict profile JSON: {error}"));
            return report;
        }
    };

    let before_profile_errors = report.errors.len();
    validate_profile(&profile, &mut report);
    report.profile_valid = report.errors.len() == before_profile_errors;
    report.counts.required_channels = profile.channels.len();
    report.counts.required_event_kinds = profile.channels.iter().map(|channel| channel.required_event_kinds.len()).sum();

    let receipt: Receipt = match serde_json::from_slice(receipt_bytes) {
        Ok(value) => value,
        Err(error) => {
            push_error(&mut report, "E_RECEIPT_JSON", "receipt", &format!("invalid or non-strict coverage receipt JSON: {error}"));
            return report;
        }
    };

    let before_receipt_errors = report.errors.len();
    validate_receipt_shape(&receipt, &mut report);

    let profile_digest_matches = receipt.profile_digest == report.profile_digest;
    let profile_id_matches = receipt.profile_id == profile.profile_id;
    report.profile_match = report.profile_valid && profile_digest_matches && profile_id_matches;
    if !profile_digest_matches {
        push_error(&mut report, "E_PROFILE_DIGEST_MISMATCH", "receipt.profile_digest", "receipt must bind the exact separately supplied profile bytes");
    }
    if !profile_id_matches {
        push_error(&mut report, "E_PROFILE_ID_MISMATCH", "receipt.profile_id", "receipt profile_id differs from the separately supplied profile");
    }

    validate_platform(&profile, &receipt, &mut report);
    validate_controls(&profile, &receipt, &mut report);
    validate_capture(&receipt, &mut report);
    validate_channels(&profile, &receipt, &mut report);
    // Channel classification, capture identity and profile binding are part of
    // receipt well-formedness, not merely advisory diagnostics.
    report.receipt_well_formed = report.errors.len() == before_receipt_errors;

    let all_events_observed = report.counts.required_event_kinds > 0
        && report.counts.observed_event_kinds == report.counts.required_event_kinds
        && report.counts.unresolved_event_kinds == 0
        && report.counts.excluded_event_kinds == 0;
    let controls_complete = profile.required_controls.iter().all(|required| receipt.controls_active.iter().any(|active| active == required));
    let platform_complete = (!profile.platform.require_kernel_build_id
            || receipt.platform.kernel_build_id.as_deref().is_some_and(|value| !value.trim().is_empty()))
        && (!profile.platform.require_kernel_btf_digest
            || receipt.platform.kernel_btf_digest.as_deref().is_some_and(valid_digest));
    let capture_complete = receipt.capture.start_monotonic_ns < receipt.capture.end_monotonic_ns
        && receipt.capture.root_pid > 0
        && receipt.capture.process_tree_closed
        && receipt.capture.snapshot_consistency == "stable"
        && !receipt.capture.monitored_roots.is_empty();
    let health_clean = receipt.health.event_count > 0
        && receipt.health.dropped_event_count == 0
        && receipt.health.ring_buffer_overflow_count == 0
        && receipt.health.observer_restart_count == 0
        && receipt.health.unmonitored_process_count == 0
        && receipt.health.root_escape_count == 0;
    let complete_conditions = report.profile_match
        && report.receipt_well_formed
        && receipt.platform.os == profile.platform.os
        && profile.platform.architectures.contains(&receipt.platform.architecture)
        && controls_complete
        && platform_complete
        && capture_complete
        && health_clean
        && all_events_observed;

    let platform_supported = receipt.platform.os == profile.platform.os
        && profile.platform.architectures.contains(&receipt.platform.architecture);
    let has_unresolved_scope = report.counts.unresolved_event_kinds > 0
        || report.counts.excluded_event_kinds > 0
        || !controls_complete
        || !platform_complete;
    report.coverage_claim_consistent = report.receipt_well_formed && report.profile_match && match receipt.claim.as_str() {
        "complete_within_profile" => complete_conditions,
        "incomplete" => true,
        "unsupported" => !platform_supported || has_unresolved_scope,
        _ => false,
    };
    report.coverage_complete_within_profile =
        receipt.claim == "complete_within_profile" && complete_conditions;
    report.overstated_claim =
        receipt.claim == "complete_within_profile" && !complete_conditions;
    if report.overstated_claim {
        push_error(
            &mut report,
            "E_CLAIM_OVERSTATES_CAPTURE",
            "receipt.claim",
            "complete_within_profile requires exact profile binding, supported platform, required controls, complete event classification, stable snapshot, closed process tree, monitored roots, and zero loss/restart/escape counters",
        );
    } else if receipt.claim == "unsupported" && all_events_observed && complete_conditions {
        // A conservative downgrade is permitted; it never promotes the receipt.
    }

    // This auditor only checks self-consistency against a candidate profile.
    // Neither a digest nor a claimed event classification authenticates capture.
    report.qualification_eligible = false;
    report.profile_qualified = false;
    report.receipt_authenticity_established = false;
    report
}

fn validate_profile(profile: &Profile, report: &mut CoverageAuditReport) {
    if profile.format_version != 1 {
        push_error(report, "E_PROFILE_VERSION", "profile.format_version", "only format_version=1 is supported");
    }
    if profile.suite_id != "basalt-observer-coverage-profile-v1" {
        push_error(report, "E_PROFILE_SUITE", "profile.suite_id", "unexpected profile suite identifier");
    }
    if !valid_identifier(&profile.profile_id) {
        push_error(report, "E_PROFILE_ID", "profile.profile_id", "profile_id has invalid syntax");
    }
    if profile.profile_status != "candidate_not_qualified" {
        push_error(report, "E_PROFILE_STATUS", "profile.profile_status", "candidate profile must remain candidate_not_qualified until independent qualification");
    }
    if profile.platform.os != "linux" || profile.platform.architectures.is_empty() {
        push_error(report, "E_PROFILE_PLATFORM", "profile.platform", "candidate profile must name Linux and at least one supported architecture");
    }
    if profile.channels.is_empty() || profile.channels.len() > MAX_CHANNELS {
        push_error(report, "E_PROFILE_CHANNEL_COUNT", "profile.channels", "profile must declare 1..=64 channels");
    }
    if profile.allowed_controls.is_empty() || profile.required_controls.is_empty() {
        push_error(report, "E_PROFILE_CONTROLS", "profile", "allowed_controls and required_controls must be non-empty");
    }
    let allowed: BTreeSet<&str> = profile.allowed_controls.iter().map(String::as_str).collect();
    if allowed.len() != profile.allowed_controls.len()
        || profile.required_controls.iter().any(|control| !allowed.contains(control.as_str()))
        || profile.required_controls.iter().collect::<BTreeSet<_>>().len() != profile.required_controls.len()
    {
        push_error(report, "E_PROFILE_CONTROL_SET", "profile.required_controls", "controls must be unique and required controls must be a subset of allowed controls");
    }
    if profile.platform.architectures.iter().collect::<BTreeSet<_>>().len() != profile.platform.architectures.len() {
        push_error(report, "E_PROFILE_ARCH_DUPLICATE", "profile.platform.architectures", "supported architectures must be unique");
    }

    let mut channel_ids = BTreeSet::new();
    let mut total_events = 0usize;
    for (index, channel) in profile.channels.iter().enumerate() {
        let path = format!("profile.channels[{index}]");
        if !valid_identifier(&channel.channel_id) || !channel_ids.insert(channel.channel_id.as_str()) {
            push_error(report, "E_PROFILE_CHANNEL_ID", &format!("{path}.channel_id"), "channel identifiers must be valid and unique");
        }
        if channel.required_event_kinds.is_empty() {
            push_error(report, "E_PROFILE_EMPTY_CHANNEL", &format!("{path}.required_event_kinds"), "each channel must require at least one event kind");
        }
        let events: BTreeSet<&str> = channel.required_event_kinds.iter().map(String::as_str).collect();
        if events.len() != channel.required_event_kinds.len()
            || channel.required_event_kinds.iter().any(|event| !valid_identifier(event))
        {
            push_error(report, "E_PROFILE_EVENT_SET", &format!("{path}.required_event_kinds"), "event kinds must be valid and unique within a channel");
        }
        total_events = total_events.saturating_add(channel.required_event_kinds.len());
    }
    if total_events > MAX_EVENT_KINDS {
        push_error(report, "E_PROFILE_EVENT_LIMIT", "profile.channels", "profile exceeds 256 required event kinds");
    }
}

fn validate_receipt_shape(receipt: &Receipt, report: &mut CoverageAuditReport) {
    if receipt.format_version != 1 || receipt.suite_id != "basalt-observer-coverage-envelope-v1" {
        push_error(report, "E_RECEIPT_VERSION", "receipt", "unsupported coverage envelope version or suite");
    }
    if receipt.run_id.trim().is_empty() || receipt.run_id.len() > 128 {
        push_error(report, "E_RUN_ID", "receipt.run_id", "run_id must contain 1..=128 non-whitespace bytes");
    }
    if !valid_identifier(&receipt.profile_id) {
        push_error(report, "E_RECEIPT_PROFILE_ID", "receipt.profile_id", "profile_id has invalid syntax");
    }
    if !valid_digest(&receipt.profile_digest)
        || !valid_digest(&receipt.observer_binary_digest)
        || !valid_digest(&receipt.runner_binary_digest)
    {
        push_error(report, "E_DIGEST", "receipt", "profile, observer and runner digests must be lowercase sha256 digests");
    }
    if receipt.claim != "complete_within_profile" && receipt.claim != "incomplete" && receipt.claim != "unsupported" {
        push_error(report, "E_CLAIM", "receipt.claim", "claim must be complete_within_profile, incomplete or unsupported");
    }
    if receipt.controls_active.len() > 128
        || receipt.controls_active.iter().collect::<BTreeSet<_>>().len() != receipt.controls_active.len()
    {
        push_error(report, "E_ACTIVE_CONTROLS", "receipt.controls_active", "active controls must be unique and limited to 128");
    }
    if receipt.channels.len() > MAX_CHANNELS {
        push_error(report, "E_RECEIPT_CHANNEL_LIMIT", "receipt.channels", "receipt exceeds 64 channels");
    }
    if receipt.capture.monitored_roots.len() > MAX_ROOTS {
        push_error(report, "E_ROOT_LIMIT", "receipt.capture.monitored_roots", "receipt exceeds 1024 monitored roots");
    }
}

fn validate_platform(_profile: &Profile, receipt: &Receipt, report: &mut CoverageAuditReport) {
    // A syntactically valid but unsupported platform can be truthfully reported
    // as unsupported; it must not be promoted to complete_within_profile.
    if receipt.platform.kernel_release.trim().is_empty() {
        push_error(report, "E_KERNEL_RELEASE", "receipt.platform.kernel_release", "kernel_release must be recorded");
    }
    if let Some(digest) = receipt.platform.kernel_btf_digest.as_deref() {
        if !valid_digest(digest) {
            push_error(report, "E_KERNEL_BTF_DIGEST", "receipt.platform.kernel_btf_digest", "kernel BTF digest must be a lowercase sha256 digest");
        }
    }
}

fn validate_controls(profile: &Profile, receipt: &Receipt, report: &mut CoverageAuditReport) {
    let allowed: BTreeSet<&str> = profile.allowed_controls.iter().map(String::as_str).collect();
    for (index, control) in receipt.controls_active.iter().enumerate() {
        if !allowed.contains(control.as_str()) {
            push_error(report, "E_UNKNOWN_CONTROL", &format!("receipt.controls_active[{index}]"), "active control is not allowed by the pinned profile");
        }
    }
    report.counts.active_required_controls = profile.required_controls.iter()
        .filter(|required| receipt.controls_active.iter().any(|active| active == *required))
        .count();
}

fn validate_capture(receipt: &Receipt, report: &mut CoverageAuditReport) {
    let capture = &receipt.capture;
    if capture.start_monotonic_ns >= capture.end_monotonic_ns {
        push_error(report, "E_CAPTURE_INTERVAL", "receipt.capture", "capture interval must have strictly increasing monotonic timestamps");
    }
    if capture.root_pid == 0 {
        push_error(report, "E_ROOT_PID", "receipt.capture.root_pid", "root_pid must be non-zero");
    }
    if capture.snapshot_consistency != "stable" && capture.snapshot_consistency != "raced" && capture.snapshot_consistency != "unknown" {
        push_error(report, "E_SNAPSHOT_STATE", "receipt.capture.snapshot_consistency", "snapshot_consistency must be stable, raced or unknown");
    }
    let mut roots = BTreeSet::new();
    for (index, root) in capture.monitored_roots.iter().enumerate() {
        if !valid_absolute_root(root) || !roots.insert(root.as_str()) {
            push_error(report, "E_MONITORED_ROOT", &format!("receipt.capture.monitored_roots[{index}]"), "monitored roots must be unique normalized absolute paths without dot or parent components");
        }
    }
    report.counts.monitored_roots = roots.len();
}

fn validate_channels(profile: &Profile, receipt: &Receipt, report: &mut CoverageAuditReport) {
    let mut profile_by_id = BTreeMap::<&str, &ProfileChannel>::new();
    for channel in &profile.channels {
        profile_by_id.insert(channel.channel_id.as_str(), channel);
    }
    let mut seen_channels = BTreeSet::new();
    for (index, assessment) in receipt.channels.iter().enumerate() {
        let path = format!("receipt.channels[{index}]");
        let Some(expected) = profile_by_id.get(assessment.channel_id.as_str()).copied() else {
            push_error(report, "E_UNKNOWN_CHANNEL", &format!("{path}.channel_id"), "receipt channel is not declared by the pinned profile");
            continue;
        };
        if !seen_channels.insert(assessment.channel_id.as_str()) {
            push_error(report, "E_DUPLICATE_CHANNEL", &format!("{path}.channel_id"), "each profile channel must appear exactly once");
            continue;
        }
        report.counts.assessed_channels += 1;
        let expected_events: BTreeSet<&str> = expected.required_event_kinds.iter().map(String::as_str).collect();
        let mut seen_events = BTreeSet::new();
        for (event_index, event) in assessment.events.iter().enumerate() {
            let event_path = format!("{path}.events[{event_index}]");
            if !expected_events.contains(event.event_kind.as_str()) {
                push_error(report, "E_UNKNOWN_EVENT_KIND", &format!("{event_path}.event_kind"), "event kind is not required by the pinned channel profile");
                continue;
            }
            if !seen_events.insert(event.event_kind.as_str()) {
                push_error(report, "E_DUPLICATE_EVENT_KIND", &format!("{event_path}.event_kind"), "each required event kind must have exactly one classification");
                continue;
            }
            match event.status.as_str() {
                "observed" => {
                    if !event.evidence_digest.as_deref().is_some_and(valid_digest) || event.reason.as_ref().is_some_and(|reason| !reason.trim().is_empty()) {
                        push_error(report, "E_OBSERVED_EVENT_WITNESS", &format!("{event_path}"), "observed events require a digest and must not carry an unresolved/exclusion reason");
                    } else {
                        report.counts.observed_event_kinds += 1;
                    }
                }
                "unresolved" | "excluded" => {
                    if event.reason.as_deref().is_none_or(|reason| reason.trim().is_empty()) || event.evidence_digest.is_some() {
                        push_error(report, "E_UNRESOLVED_EVENT_WITNESS", &format!("{event_path}"), "unresolved/excluded events require a reason and must not claim an observation digest");
                    } else if event.status == "unresolved" {
                        report.counts.unresolved_event_kinds += 1;
                    } else {
                        report.counts.excluded_event_kinds += 1;
                    }
                }
                _ => push_error(report, "E_EVENT_STATUS", &format!("{event_path}.status"), "event status must be observed, unresolved or excluded"),
            }
        }
        for missing in expected_events.difference(&seen_events) {
            push_error(report, "E_EVENT_CLASSIFICATION_MISSING", &format!("{path}.events"), &format!("required event kind {missing} is not classified"));
        }
    }
    for missing in profile_by_id.keys().filter(|channel| !seen_channels.contains(**channel)) {
        push_error(report, "E_CHANNEL_MISSING", "receipt.channels", &format!("required profile channel {missing} is not present"));
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_lowercase().then_some(true).unwrap_or(false)
        && value.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_')
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_absolute_root(value: &str) -> bool {
    if !value.starts_with('/') || value.trim().is_empty() || value.contains('\\') || value.contains("//") {
        return false;
    }
    if value.len() > 1 && value.ends_with('/') {
        return false;
    }
    value != "/" && value.split('/').skip(1).all(|component| !component.is_empty() && component != "." && component != "..")
}

fn read_bounded(path: &Path) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        anyhow::bail!("input {} exceeds the 4 MiB limit", path.display());
    }
    Ok(bytes)
}

fn push_error(report: &mut CoverageAuditReport, code: &str, path: &str, message: &str) {
    report.errors.push(CoverageError {
        code: code.to_string(),
        path: path.to_string(),
        message: message.to_string(),
    });
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    format!("sha256:{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn profile_value() -> Value {
        json!({
            "format_version": 1,
            "suite_id": "basalt-observer-coverage-profile-v1",
            "profile_id": "linux-hermetic-candidate-v1",
            "profile_status": "candidate_not_qualified",
            "platform": {
                "os": "linux",
                "architectures": ["x86_64", "aarch64"],
                "require_kernel_build_id": true,
                "require_kernel_btf_digest": true
            },
            "allowed_controls": [
                "read_only_workspace_snapshot",
                "process_tree_supervision",
                "filesystem_access_monitor",
                "environment_capture",
                "network_policy_enforcement",
                "kernel_event_loss_counters",
                "snapshot_consistency_check"
            ],
            "required_controls": [
                "read_only_workspace_snapshot",
                "process_tree_supervision",
                "filesystem_access_monitor",
                "environment_capture",
                "network_policy_enforcement",
                "kernel_event_loss_counters",
                "snapshot_consistency_check"
            ],
            "channels": [
                {"channel_id":"filesystem","required_event_kinds":["content_read","metadata_read","directory_enumeration","symlink_target","mmap_access","namespace_mutation"]},
                {"channel_id":"process","required_event_kinds":["process_exec","child_start","child_exit","argv_cwd","inherited_fd"]},
                {"channel_id":"environment","required_event_kinds":["environment_read","environment_commitment"]},
                {"channel_id":"generated_artifact","required_event_kinds":["generator_execution","generated_output_write"]},
                {"channel_id":"toolchain","required_event_kinds":["compiler_identity","linker_identity","sysroot_identity","target_configuration"]},
                {"channel_id":"cache","required_event_kinds":["cache_lookup","cache_hit_provenance","cache_miss"]},
                {"channel_id":"network","required_event_kinds":["network_attempt","endpoint_identity","response_artifact"]},
                {"channel_id":"observer_integrity","required_event_kinds":["event_loss_counter","observer_lifecycle","root_boundary"]},
                {"channel_id":"snapshot_consistency","required_event_kinds":["snapshot_start","snapshot_end","concurrent_mutation_check"]}
            ]
        })
    }

    fn healthy_receipt(profile_bytes: &[u8]) -> Value {
        let profile: Value = serde_json::from_slice(profile_bytes).unwrap();
        let digest = sha256_tagged(profile_bytes);
        let channels = profile["channels"].as_array().unwrap().iter().map(|channel| {
            let events = channel["required_event_kinds"].as_array().unwrap().iter().map(|event| {
                json!({
                    "event_kind": event,
                    "status": "observed",
                    "evidence_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "reason": null
                })
            }).collect::<Vec<_>>();
            json!({"channel_id":channel["channel_id"],"events":events})
        }).collect::<Vec<_>>();
        json!({
            "format_version":1,
            "suite_id":"basalt-observer-coverage-envelope-v1",
            "run_id":"coverage-test-run-1",
            "profile_id":profile["profile_id"],
            "profile_digest":digest,
            "observer_binary_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "runner_binary_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "platform":{
                "os":"linux",
                "architecture":"x86_64",
                "kernel_release":"6.8.0-test",
                "kernel_build_id":"test-build-id",
                "kernel_btf_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            },
            "controls_active":profile["required_controls"],
            "capture":{
                "start_monotonic_ns":100,
                "end_monotonic_ns":200,
                "root_pid":123,
                "process_tree_closed":true,
                "monitored_roots":["/workspace","/nix/store"],
                "snapshot_consistency":"stable"
            },
            "health":{
                "event_count":100,
                "dropped_event_count":0,
                "ring_buffer_overflow_count":0,
                "observer_restart_count":0,
                "unmonitored_process_count":0,
                "root_escape_count":0
            },
            "channels":channels,
            "claim":"complete_within_profile"
        })
    }

    fn encoded_pair(claim: &str) -> (Vec<u8>, Vec<u8>) {
        let profile = serde_json::to_vec(&profile_value()).unwrap();
        let mut receipt = healthy_receipt(&profile);
        receipt["claim"] = json!(claim);
        (profile, serde_json::to_vec(&receipt).unwrap())
    }

    #[test]
    fn accepts_self_consistent_complete_claim_but_never_qualifies_profile() {
        let (profile, receipt) = encoded_pair("complete_within_profile");
        let report = audit_bytes(&profile, &receipt);
        assert!(report.profile_valid);
        assert!(report.receipt_well_formed);
        assert!(report.profile_match);
        assert!(report.coverage_claim_consistent);
        assert!(report.coverage_complete_within_profile);
        assert!(!report.qualification_eligible);
        assert!(!report.profile_qualified);
        assert!(!report.receipt_authenticity_established);
    }

    #[test]
    fn rejects_complete_claim_with_missing_required_event() {
        let (profile, receipt) = encoded_pair("complete_within_profile");
        let mut value: Value = serde_json::from_slice(&receipt).unwrap();
        value["channels"][0]["events"].as_array_mut().unwrap().pop();
        let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
        assert!(report.overstated_claim);
        assert!(!report.coverage_complete_within_profile);
    }

    #[test]
    fn rejects_complete_claim_with_duplicate_event_classification() {
        let (profile, receipt) = encoded_pair("complete_within_profile");
        let mut value: Value = serde_json::from_slice(&receipt).unwrap();
        let first = value["channels"][0]["events"][0].clone();
        value["channels"][0]["events"].as_array_mut().unwrap().push(first);
        let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
        assert!(!report.receipt_well_formed);
        assert!(report.overstated_claim);
    }

    #[test]
    fn rejects_changed_profile_bytes_even_if_profile_id_is_unchanged() {
        let (profile, receipt) = encoded_pair("complete_within_profile");
        let mut changed: Value = serde_json::from_slice(&profile).unwrap();
        changed["channels"][0]["required_event_kinds"][0] = json!("different_event");
        let changed_profile = serde_json::to_vec(&changed).unwrap();
        let report = audit_bytes(&changed_profile, &receipt);
        assert!(!report.profile_match);
        assert!(report.overstated_claim);
    }

    #[test]
    fn rejects_complete_claim_with_event_loss_or_overflow() {
        for field in ["dropped_event_count","ring_buffer_overflow_count"] {
            let (profile, receipt) = encoded_pair("complete_within_profile");
            let mut value: Value = serde_json::from_slice(&receipt).unwrap();
            value["health"][field] = json!(1);
            let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
            assert!(report.overstated_claim, "{field}");
        }
    }

    #[test]
    fn rejects_complete_claim_with_restart_unmonitored_child_or_escape() {
        for field in ["observer_restart_count","unmonitored_process_count","root_escape_count"] {
            let (profile, receipt) = encoded_pair("complete_within_profile");
            let mut value: Value = serde_json::from_slice(&receipt).unwrap();
            value["health"][field] = json!(1);
            let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
            assert!(report.overstated_claim, "{field}");
        }
    }

    #[test]
    fn rejects_complete_claim_with_unstable_snapshot_or_open_process_tree() {
        for (section, field, value_set) in [
            ("capture","snapshot_consistency",json!("raced")),
            ("capture","process_tree_closed",json!(false))
        ] {
            let (profile, receipt) = encoded_pair("complete_within_profile");
            let mut value: Value = serde_json::from_slice(&receipt).unwrap();
            value[section][field] = value_set;
            let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
            assert!(report.overstated_claim);
        }
    }

    #[test]
    fn rejects_invalid_capture_interval_or_root_path() {
        for (field, value_set) in [
            ("end_monotonic_ns",json!(100)),
            ("monitored_roots",json!(["/workspace/../etc"]))
        ] {
            let (profile, receipt) = encoded_pair("complete_within_profile");
            let mut value: Value = serde_json::from_slice(&receipt).unwrap();
            value["capture"][field] = value_set;
            let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
            assert!(report.overstated_claim);
        }
    }

    #[test]
    fn rejects_complete_claim_with_an_unrecognized_extra_channel() {
        let (profile, receipt) = encoded_pair("complete_within_profile");
        let mut value: Value = serde_json::from_slice(&receipt).unwrap();
        value["channels"].as_array_mut().unwrap().push(json!({
            "channel_id":"unprofiled_channel",
            "events":[]
        }));
        let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
        assert!(!report.receipt_well_formed);
        assert!(!report.coverage_complete_within_profile);
    }

    #[test]
    fn rejects_complete_claim_when_a_required_control_is_missing() {
        let (profile, receipt) = encoded_pair("complete_within_profile");
        let mut value: Value = serde_json::from_slice(&receipt).unwrap();
        value["controls_active"].as_array_mut().unwrap().pop();
        let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
        assert!(report.overstated_claim);
        assert!(!report.coverage_complete_within_profile);
    }

    #[test]
    fn accepts_an_out_of_profile_platform_only_as_unsupported() {
        let (profile, receipt) = encoded_pair("unsupported");
        let mut value: Value = serde_json::from_slice(&receipt).unwrap();
        value["platform"]["os"] = json!("freebsd");
        let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
        assert!(report.receipt_well_formed);
        assert!(report.coverage_claim_consistent);
        assert!(!report.coverage_complete_within_profile);
        assert!(!report.qualification_eligible);
    }

    #[test]
    fn accepts_explicit_incomplete_downgrade_with_unresolved_event() {
        let (profile, receipt) = encoded_pair("incomplete");
        let mut value: Value = serde_json::from_slice(&receipt).unwrap();
        value["channels"][0]["events"][0]["status"] = json!("unresolved");
        value["channels"][0]["events"][0]["evidence_digest"] = Value::Null;
        value["channels"][0]["events"][0]["reason"] = json!("observer hook unavailable");
        let report = audit_bytes(&profile, &serde_json::to_vec(&value).unwrap());
        assert!(report.receipt_well_formed);
        assert!(report.coverage_claim_consistent);
        assert!(!report.coverage_complete_within_profile);
        assert!(!report.qualification_eligible);
    }
}
