//! Tamper-evident receipts for observable search/network exposure.
//!
//! A verified local chain is not proof of complete logging. A trustworthy
//! deployment must independently anchor the receipt commitment and enforce the
//! network boundary outside the evaluated policy process.

use crate::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const SEARCH_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_search_exposure_receipt.v1";
pub const SEARCH_RECEIPT_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_search_exposure_report.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.search_receipt.v1\0";
const ZERO_DIGEST: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
const MAX_EVENTS: usize = 250_000;
const NETWORK_POLICIES: &[&str] = &["isolated", "audited_open_web", "controlled_allowlist"];
const EVENT_KINDS: &[&str] = &[
    "search_query",
    "fetch",
    "redirect",
    "cache_hit",
    "network_request",
];
const DECISIONS: &[&str] = &["allow", "block"];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchExposureReceiptV1 {
    pub schema: String,
    pub run_id: String,
    pub generation_id: String,
    pub policy_revision_sha256: String,
    pub task_ref_sha256: String,
    pub network_policy: String,
    pub allowed_hosts: Vec<String>,
    pub expected_event_count: u64,
    pub final_event_sha256: String,
    pub events: Vec<SearchExposureEventV1>,
    pub receipt_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchExposureEventV1 {
    pub sequence: u64,
    pub event_kind: String,
    pub decision: String,
    pub request_sha256: String,
    pub host: String,
    pub response_sha256: Option<String>,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct EventHashPayloadV1<'a> {
    sequence: u64,
    event_kind: &'a str,
    decision: &'a str,
    request_sha256: &'a str,
    host: &'a str,
    response_sha256: &'a Option<String>,
    previous_event_sha256: &'a str,
}

#[derive(Debug, Clone, Serialize)]
struct ReceiptHashPayloadV1<'a> {
    schema: &'a str,
    run_id: &'a str,
    generation_id: &'a str,
    policy_revision_sha256: &'a str,
    task_ref_sha256: &'a str,
    network_policy: &'a str,
    allowed_hosts: &'a [String],
    expected_event_count: u64,
    final_event_sha256: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchExposureReceiptReportV1 {
    pub schema: String,
    pub receipt_sha256: String,
    pub run_id: String,
    pub generation_id: String,
    pub network_policy: String,
    pub expected_event_count: u64,
    pub observed_event_count: u64,
    pub event_count_matches: bool,
    pub event_chain_valid: bool,
    pub final_event_matches: bool,
    pub receipt_commitment_expected: String,
    pub receipt_commitment_recomputed: String,
    pub receipt_commitment_matches: bool,
    pub integrity_valid: bool,
    pub policy_compliant: bool,
    pub event_kind_counts: std::collections::BTreeMap<String, u64>,
    pub violations: Vec<String>,
    pub admission: String,
}

fn digest_domain_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len = u64::try_from(domain.len())
        .map_err(|_| "hash domain exceeds u64 length".to_owned())?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| "hash payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update(domain_len.to_be_bytes());
    hasher.update(domain);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_host(host: &str, label: &str) -> Result<(), String> {
    if host.is_empty()
        || host.len() > 253
        || !host.is_ascii()
        || host.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be a normalized lowercase ASCII host"));
    }
    for part in host.split('.') {
        if part.is_empty() || part.len() > 63 {
            return Err(format!("{label} contains an invalid DNS label"));
        }
        let bytes = part.as_bytes();
        if !bytes[0].is_ascii_alphanumeric()
            || !bytes[bytes.len() - 1].is_ascii_alphanumeric()
            || !bytes
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        {
            return Err(format!("{label} contains an invalid DNS label"));
        }
    }
    Ok(())
}

fn validate_receipt_shape(receipt: &SearchExposureReceiptV1) -> Result<(), String> {
    if receipt.schema != SEARCH_RECEIPT_SCHEMA {
        return Err(format!("receipt schema must be {SEARCH_RECEIPT_SCHEMA}"));
    }
    if receipt.run_id.trim().is_empty()
        || receipt.run_id.len() > 128
        || !receipt
            .run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err("run_id must be 1..=128 ASCII letters/digits or . _ : -".to_owned());
    }
    if receipt.generation_id.trim().is_empty() || receipt.generation_id.len() > 128 {
        return Err("generation_id must be 1..=128 characters".to_owned());
    }
    if !valid_digest(&receipt.policy_revision_sha256) {
        return Err("policy_revision_sha256 must be 64 lowercase hex characters".to_owned());
    }
    if !valid_digest(&receipt.task_ref_sha256) {
        return Err("task_ref_sha256 must be 64 lowercase hex characters".to_owned());
    }
    if !NETWORK_POLICIES.contains(&receipt.network_policy.as_str()) {
        return Err(format!("unsupported network_policy: {}", receipt.network_policy));
    }
    if receipt.events.len() > MAX_EVENTS {
        return Err(format!("receipt exceeds safety limit of {MAX_EVENTS} events"));
    }
    if receipt.expected_event_count > MAX_EVENTS as u64 {
        return Err(format!("expected_event_count exceeds safety limit of {MAX_EVENTS}"));
    }
    if !valid_digest(&receipt.final_event_sha256) {
        return Err("final_event_sha256 must be 64 lowercase hex characters".to_owned());
    }
    if !valid_digest(&receipt.receipt_commitment_sha256) {
        return Err("receipt_commitment_sha256 must be 64 lowercase hex characters".to_owned());
    }

    let mut allowed_hosts = BTreeSet::new();
    let mut previous_host: Option<&str> = None;
    for host in &receipt.allowed_hosts {
        validate_host(host, "allowed_hosts entry")?;
        if !allowed_hosts.insert(host.as_str()) {
            return Err(format!("duplicate allowed_hosts entry: {host}"));
        }
        if previous_host.is_some_and(|previous| previous >= host.as_str()) {
            return Err("allowed_hosts must be sorted lexicographically".to_owned());
        }
        previous_host = Some(host);
    }
    match receipt.network_policy.as_str() {
        "isolated" | "audited_open_web" if !receipt.allowed_hosts.is_empty() => {
            return Err(format!(
                "network_policy {} must not declare allowed_hosts",
                receipt.network_policy
            ));
        }
        "controlled_allowlist" if receipt.allowed_hosts.is_empty() => {
            return Err("controlled_allowlist requires at least one allowed host".to_owned());
        }
        _ => {}
    }

    for (index, event) in receipt.events.iter().enumerate() {
        if !EVENT_KINDS.contains(&event.event_kind.as_str()) {
            return Err(format!("events[{index}] has unsupported event_kind"));
        }
        if !DECISIONS.contains(&event.decision.as_str()) {
            return Err(format!("events[{index}] has unsupported decision"));
        }
        if !valid_digest(&event.request_sha256) {
            return Err(format!("events[{index}].request_sha256 must be a lowercase SHA-256 digest"));
        }
        validate_host(&event.host, &format!("events[{index}].host"))?;
        if !valid_digest(&event.previous_event_sha256) || !valid_digest(&event.event_sha256) {
            return Err(format!("events[{index}] contains an invalid chain digest"));
        }
        if let Some(response) = &event.response_sha256 {
            if !valid_digest(response) {
                return Err(format!("events[{index}].response_sha256 must be a lowercase SHA-256 digest"));
            }
        }
        match (event.decision.as_str(), event.response_sha256.is_some()) {
            ("allow", false) => {
                return Err(format!("events[{index}] allowed event must include a response digest"));
            }
            ("block", true) => {
                return Err(format!("events[{index}] blocked event must not include a response digest"));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn event_hash(event: &SearchExposureEventV1) -> Result<String, String> {
    let payload = EventHashPayloadV1 {
        sequence: event.sequence,
        event_kind: &event.event_kind,
        decision: &event.decision,
        request_sha256: &event.request_sha256,
        host: &event.host,
        response_sha256: &event.response_sha256,
        previous_event_sha256: &event.previous_event_sha256,
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize event hash payload: {error}"))?;
    digest_domain_framed(b"event.v1", &encoded)
}

pub(crate) fn receipt_hash(receipt: &SearchExposureReceiptV1) -> Result<String, String> {
    let payload = ReceiptHashPayloadV1 {
        schema: &receipt.schema,
        run_id: &receipt.run_id,
        generation_id: &receipt.generation_id,
        policy_revision_sha256: &receipt.policy_revision_sha256,
        task_ref_sha256: &receipt.task_ref_sha256,
        network_policy: &receipt.network_policy,
        allowed_hosts: &receipt.allowed_hosts,
        expected_event_count: receipt.expected_event_count,
        final_event_sha256: &receipt.final_event_sha256,
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize receipt hash payload: {error}"))?;
    digest_domain_framed(b"receipt.v1", &encoded)
}

/// Verify a receipt's chain, declared final event, run commitment, and declared
/// network policy. Only digests and hostnames are accepted; raw queries/URLs and
/// response bodies must be kept out of the receipt.
pub fn verify_search_exposure_receipt(bytes: &[u8]) -> Result<SearchExposureReceiptReportV1, String> {
    let receipt: SearchExposureReceiptV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("receipt does not match strict v1 JSON shape: {error}"))?;
    validate_receipt_shape(&receipt)?;

    let observed_event_count = u64::try_from(receipt.events.len())
        .map_err(|_| "observed event count exceeds u64".to_owned())?;
    let event_count_matches = observed_event_count == receipt.expected_event_count;
    let mut event_chain_valid = true;
    let mut expected_previous = ZERO_DIGEST.to_owned();
    let mut violations = Vec::new();
    let mut event_kind_counts = std::collections::BTreeMap::<String, u64>::new();

    for (index, event) in receipt.events.iter().enumerate() {
        let expected_sequence =
            u64::try_from(index).map_err(|_| "event index exceeds u64".to_owned())?;
        if event.sequence != expected_sequence {
            event_chain_valid = false;
            violations.push(format!(
                "events[{index}] sequence {} does not match expected {expected_sequence}",
                event.sequence
            ));
        }
        if event.previous_event_sha256 != expected_previous {
            event_chain_valid = false;
            violations.push(format!("events[{index}] previous-event digest does not match"));
        }
        let recomputed_event = event_hash(event)?;
        if recomputed_event != event.event_sha256 {
            event_chain_valid = false;
            violations.push(format!("events[{index}] event digest does not match its contents"));
        }
        expected_previous = recomputed_event;
        let count = event_kind_counts.entry(event.event_kind.clone()).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| "event-kind counter overflows u64".to_owned())?;

        match receipt.network_policy.as_str() {
            "isolated" if event.decision == "allow" => {
                violations.push(format!(
                    "events[{index}] allowed network activity violates isolated policy"
                ));
            }
            "controlled_allowlist"
                if event.decision == "allow"
                    && !receipt.allowed_hosts.iter().any(|host| host == &event.host) =>
            {
                violations.push(format!(
                    "events[{index}] allowed host {} is outside controlled allowlist",
                    event.host
                ));
            }
            _ => {}
        }
    }

    let final_event_matches = expected_previous == receipt.final_event_sha256
        && receipt
            .events
            .last()
            .map(|event| event.event_sha256.as_str())
            .unwrap_or(ZERO_DIGEST)
            == receipt.final_event_sha256;
    if !final_event_matches {
        event_chain_valid = false;
        violations.push("final_event_sha256 does not match the verified event chain".to_owned());
    }

    let receipt_commitment_recomputed = receipt_hash(&receipt)?;
    let receipt_commitment_matches =
        receipt_commitment_recomputed == receipt.receipt_commitment_sha256;
    if !event_count_matches {
        violations.push(format!(
            "expected_event_count={} but observed {} events",
            receipt.expected_event_count, observed_event_count
        ));
    }
    if !receipt_commitment_matches {
        violations.push("receipt commitment does not match the declared run context".to_owned());
    }

    let integrity_valid =
        event_count_matches && event_chain_valid && final_event_matches && receipt_commitment_matches;
    let policy_compliant = !violations.iter().any(|violation| {
        violation.contains("violates isolated policy")
            || violation.contains("outside controlled allowlist")
    });
    let admission = if !integrity_valid {
        "integrity_invalid"
    } else if !policy_compliant {
        "network_policy_violation"
    } else {
        "receipt_integrity_and_declared_policy_pass_external_anchor_required"
    };

    Ok(SearchExposureReceiptReportV1 {
        schema: SEARCH_RECEIPT_REPORT_SCHEMA.to_owned(),
        receipt_sha256: sha256_hex(bytes),
        run_id: receipt.run_id,
        generation_id: receipt.generation_id,
        network_policy: receipt.network_policy,
        expected_event_count: receipt.expected_event_count,
        observed_event_count,
        event_count_matches,
        event_chain_valid,
        final_event_matches,
        receipt_commitment_expected: receipt.receipt_commitment_sha256,
        receipt_commitment_recomputed,
        receipt_commitment_matches,
        integrity_valid,
        policy_compliant,
        event_kind_counts,
        violations,
        admission: admission.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_receipt(network_policy: &str, allowed_hosts: Vec<String>) -> SearchExposureReceiptV1 {
        let mut events = vec![
            SearchExposureEventV1 {
                sequence: 0,
                event_kind: "search_query".to_owned(),
                decision: if network_policy == "isolated" { "block" } else { "allow" }.to_owned(),
                request_sha256: sha256_hex(b"query-not-stored"),
                host: "search.example.org".to_owned(),
                response_sha256: if network_policy == "isolated" {
                    None
                } else {
                    Some(sha256_hex(b"results-not-stored"))
                },
                previous_event_sha256: ZERO_DIGEST.to_owned(),
                event_sha256: ZERO_DIGEST.to_owned(),
            },
            SearchExposureEventV1 {
                sequence: 1,
                event_kind: "fetch".to_owned(),
                decision: if network_policy == "isolated" { "block" } else { "allow" }.to_owned(),
                request_sha256: sha256_hex(b"https://docs.example.org/resource"),
                host: "docs.example.org".to_owned(),
                response_sha256: if network_policy == "isolated" {
                    None
                } else {
                    Some(sha256_hex(b"body-not-stored"))
                },
                previous_event_sha256: ZERO_DIGEST.to_owned(),
                event_sha256: ZERO_DIGEST.to_owned(),
            },
        ];
        let mut previous = ZERO_DIGEST.to_owned();
        for event in &mut events {
            event.previous_event_sha256 = previous.clone();
            event.event_sha256 = event_hash(event).unwrap();
            previous = event.event_sha256.clone();
        }
        let mut receipt = SearchExposureReceiptV1 {
            schema: SEARCH_RECEIPT_SCHEMA.to_owned(),
            run_id: "run-0001".to_owned(),
            generation_id: "refresh-g1".to_owned(),
            policy_revision_sha256: sha256_hex(b"policy-revision"),
            task_ref_sha256: sha256_hex(b"opaque-task-reference"),
            network_policy: network_policy.to_owned(),
            allowed_hosts,
            expected_event_count: events.len() as u64,
            final_event_sha256: previous,
            events,
            receipt_commitment_sha256: ZERO_DIGEST.to_owned(),
        };
        receipt.receipt_commitment_sha256 = receipt_hash(&receipt).unwrap();
        receipt
    }

    #[test]
    fn valid_isolated_receipt_has_integrity_and_policy_pass() {
        let receipt = valid_receipt("isolated", vec![]);
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let report = verify_search_exposure_receipt(&bytes).unwrap();
        assert!(report.integrity_valid);
        assert!(report.policy_compliant);
        assert_eq!(report.admission, "receipt_integrity_and_declared_policy_pass_external_anchor_required");
    }

    #[test]
    fn allowed_network_event_violates_isolated_policy() {
        let receipt = valid_receipt("audited_open_web", vec![]);
        let mut parsed = serde_json::to_value(receipt).unwrap();
        parsed["network_policy"] = json!("isolated");
        parsed["events"][0]["decision"] = json!("block");
        parsed["events"][0]["response_sha256"] = serde_json::Value::Null;
        // Re-seal the internally consistent chain to isolate the network-policy gate.
        let mut receipt: SearchExposureReceiptV1 = serde_json::from_value(parsed).unwrap();
        let mut previous = ZERO_DIGEST.to_owned();
        for event in &mut receipt.events {
            event.previous_event_sha256 = previous.clone();
            event.event_sha256 = event_hash(event).unwrap();
            previous = event.event_sha256.clone();
        }
        receipt.final_event_sha256 = previous;
        receipt.receipt_commitment_sha256 = receipt_hash(&receipt).unwrap();
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let report = verify_search_exposure_receipt(&bytes).unwrap();
        assert!(report.integrity_valid);
        assert!(!report.policy_compliant);
        assert_eq!(report.admission, "network_policy_violation");
    }

    #[test]
    fn event_mutation_is_detected() {
        let receipt = valid_receipt("audited_open_web", vec![]);
        let mut parsed = serde_json::to_value(receipt).unwrap();
        parsed["events"][0]["request_sha256"] = json!(sha256_hex(b"mutated-query"));
        let report = verify_search_exposure_receipt(&serde_json::to_vec(&parsed).unwrap()).unwrap();
        assert!(!report.integrity_valid);
        assert!(!report.event_chain_valid);
    }

    #[test]
    fn reordered_or_missing_events_are_detected() {
        let receipt = valid_receipt("audited_open_web", vec![]);
        let mut reordered = receipt.clone();
        reordered.events.swap(0, 1);
        let report = verify_search_exposure_receipt(&serde_json::to_vec(&reordered).unwrap()).unwrap();
        assert!(!report.integrity_valid);

        let mut missing = receipt;
        missing.events.pop();
        let report = verify_search_exposure_receipt(&serde_json::to_vec(&missing).unwrap()).unwrap();
        assert!(!report.event_count_matches);
        assert!(!report.integrity_valid);
    }

    #[test]
    fn controlled_allowlist_blocks_an_allowed_unlisted_host() {
        let receipt = valid_receipt("controlled_allowlist", vec!["docs.example.org".to_owned()]);
        let report = verify_search_exposure_receipt(&serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(report.integrity_valid);
        assert!(!report.policy_compliant);
        assert_eq!(report.admission, "network_policy_violation");
    }

    #[test]
    fn receipt_uses_digests_not_raw_queries_or_response_bodies() {
        let receipt = valid_receipt("audited_open_web", vec![]);
        let encoded = serde_json::to_string(&receipt).unwrap();
        assert!(!encoded.contains("query-not-stored"));
        assert!(!encoded.contains("body-not-stored"));
        assert!(!encoded.contains("https://docs.example.org/resource"));
    }
}
