//! Tamper-evident evidence for inference-time search/retrieval exposure.
//!
//! A trace records the information-access events observed by an instrumented
//! evaluation perimeter. Hash-chain replay proves internal integrity only;
//! it does not prove that the perimeter captured every possible channel.
//! Independent signer, proxy, sandbox and custody verification are separate.

use crate::{fresh_instance::parse_timestamp, hash_domain};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SEARCH_TRACE_SCHEMA: &str = "symthaea.search_time_exposure_trace.v1";
pub const SEARCH_AUDIT_SCHEMA: &str = "symthaea.search_time_exposure_audit.v1";
pub const SEARCH_FINDINGS_SCHEMA: &str = "symthaea.search_time_exposure_findings.v1";
pub const SEARCH_FINDINGS_AUDIT_SCHEMA: &str = "symthaea.search_time_exposure_findings_audit.v1";
pub const SEARCH_TRACE_GENESIS: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
pub const SEARCH_MAX_EVENTS: usize = 1_000_000;
pub const SEARCH_MAX_TRACE_BYTES: usize = 256 * 1024 * 1024;
pub const SEARCH_MAX_FINDINGS_BYTES: usize = 64 * 1024 * 1024;
pub const SEARCH_MAX_ALLOWLIST_DESTINATIONS: usize = 100_000;
pub const SEARCH_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const SEARCH_MAX_TOTAL_BODY_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchAccessModeV1 {
    NetworkDisabled,
    Allowlisted,
    Unrestricted,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SearchInformationChannelV1 {
    WebSearch,
    BrowserNavigation,
    HttpApi,
    RemoteMcp,
    VectorStore,
    LocalCache,
    MountedFile,
    AgentMemory,
    Other,
}

impl SearchInformationChannelV1 {
    fn requires_network(&self) -> bool {
        matches!(
            self,
            Self::WebSearch | Self::BrowserNavigation | Self::HttpApi | Self::RemoteMcp
        )
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchEventDispositionV1 {
    Allowed,
    Blocked,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeEventV1 {
    pub sequence: u64,
    pub invocation_id: String,
    pub channel: SearchInformationChannelV1,
    pub provider_or_tool: String,
    /// Hash of normalized destination identifier. Raw URLs are kept evaluator-side.
    pub destination_sha256: String,
    /// Hash of the exact request bytes, captured before redaction.
    pub request_sha256: String,
    /// Absent for blocked requests or requests returning no body.
    pub response_sha256: Option<String>,
    pub response_bytes: u64,
    pub occurred_at_utc: String,
    pub disposition: SearchEventDispositionV1,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeExposureTraceV1 {
    pub schema: String,
    pub run_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub challenge_packet_sha256: String,
    pub network_policy_sha256: String,
    pub enforcement_build_sha256: String,
    pub policy_id: String,
    pub access_mode: SearchAccessModeV1,
    /// SHA-256 digests of normalized destinations permitted by allowlisted mode.
    pub allowlisted_destination_sha256s: Vec<String>,
    /// Channels that the instrumentation claims to observe.
    pub instrumented_channels: Vec<SearchInformationChannelV1>,
    /// Known channels outside this trace perimeter. Must be disclosed.
    pub unobserved_channels: Vec<SearchInformationChannelV1>,
    pub events: Vec<SearchTimeEventV1>,
    pub event_count: u64,
    pub trace_root_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeExposureAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub hash_chain_replayed: bool,
    pub trace_root_matches: bool,
    pub sequence_complete: bool,
    pub timestamps_valid_and_monotonic: bool,
    pub request_receipts_complete: bool,
    pub accepted_response_receipts_complete: bool,
    pub blocked_events_have_no_response_bytes: bool,
    pub network_disabled_policy_respected: bool,
    pub allowlist_policy_respected: bool,
    pub all_event_channels_declared_instrumented: bool,
    pub all_known_channel_types_accounted_for: bool,
    pub no_unobserved_channels_declared: bool,
    /// Always false: this module cannot authenticate the enforcement perimeter.
    pub complete_mediation_verified: bool,
    pub independently_verified: bool,
    pub event_count: u64,
    pub allowed_event_count: u64,
    pub blocked_event_count: u64,
    pub run_id: String,
    pub trace_root_sha256: String,
    pub coverage_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchExposureClassV1 {
    MetadataOnly,
    ContextBearing,
    AnswerBearing,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchFindingMethodV1 {
    ExactMatch,
    FuzzyText,
    SemanticReview,
    HumanAdjudication,
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchExposureFindingV1 {
    pub event_sequence: u64,
    pub exposure_class: SearchExposureClassV1,
    pub matched_item_ids: Vec<String>,
    pub evidence_sha256: String,
    pub method: SearchFindingMethodV1,
    pub confidence_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeExposureFindingsV1 {
    pub schema: String,
    pub run_id: String,
    pub trace_root_sha256: String,
    pub response_body_event_count: u64,
    pub response_body_evidence_count: u64,
    /// Allowed events whose response bodies were not retained for matching.
    pub unavailable_body_event_sequences: Vec<u64>,
    pub findings: Vec<SearchExposureFindingV1>,
    pub findings_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchCapturedResponseBodyV1 {
    pub event_sequence: u64,
    pub content_type: String,
    /// Evaluator-only raw body. Never copy this field into a public finding.
    pub body_text: String,
    pub response_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeExposureFindingsAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub findings_digest_matches: bool,
    pub all_findings_reference_allowed_events: bool,
    pub unique_event_sequences: bool,
    pub metadata_only_count: u64,
    pub context_bearing_count: u64,
    pub answer_bearing_count: u64,
    pub unknown_count: u64,
    pub matched_item_reference_count: u64,
    pub response_body_event_count: u64,
    pub response_body_evidence_count: u64,
    pub unavailable_body_event_count: u64,
    pub body_evidence_coverage_permille: Option<u16>,
    pub findings_sha256: String,
    pub claim_boundary: String,
    pub issues: Vec<String>,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn all_known_channels() -> BTreeSet<SearchInformationChannelV1> {
    [
        SearchInformationChannelV1::WebSearch,
        SearchInformationChannelV1::BrowserNavigation,
        SearchInformationChannelV1::HttpApi,
        SearchInformationChannelV1::RemoteMcp,
        SearchInformationChannelV1::VectorStore,
        SearchInformationChannelV1::LocalCache,
        SearchInformationChannelV1::MountedFile,
        SearchInformationChannelV1::AgentMemory,
        SearchInformationChannelV1::Other,
    ]
    .into_iter()
    .collect()
}

fn event_digest(event: &SearchTimeEventV1) -> Result<String, String> {
    let mut normalized = event.clone();
    normalized.event_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize search event: {error}"))?;
    Ok(hash_domain(b"search-time-event.v1", &bytes))
}

fn trace_digest(trace: &SearchTimeExposureTraceV1) -> Result<String, String> {
    let mut normalized = trace.clone();
    normalized.trace_root_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize search trace: {error}"))?;
    Ok(hash_domain(b"search-time-trace.v1", &bytes))
}

/// Domain-separated digest for the exact UTF-8 response body bytes.
pub fn search_response_body_digest(body: &str) -> String {
    hash_domain(b"search-time-response-body.v1", body.as_bytes())
}

fn normalized_words(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn contains_word_sequence(haystack: &[String], needle: &[String]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

fn shared_shingle_count(left: &[String], right: &[String], width: usize) -> usize {
    if left.len() < width || right.len() < width {
        return 0;
    }
    let left_shingles: BTreeSet<Vec<&str>> = left.windows(width)
        .map(|window| window.iter().map(String::as_str).collect())
        .collect();
    let right_shingles: BTreeSet<Vec<&str>> = right.windows(width)
        .map(|window| window.iter().map(String::as_str).collect())
        .collect();
    left_shingles.intersection(&right_shingles).count()
}

fn extract_json_objects(input: &str) -> Vec<&str> {
    let bytes = input.as_bytes();
    let mut objects = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'{' {
            index += 1;
            continue;
        }
        let start = index;
        let mut depth = 0_u32;
        let mut in_string = false;
        let mut escaped = false;
        let mut cursor = index;
        let mut end = None;
        while cursor < bytes.len() {
            let byte = bytes[cursor];
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
            } else {
                match byte {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            end = Some(cursor + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            cursor += 1;
        }
        if let Some(end) = end {
            if let Some(object) = input.get(start..end) {
                objects.push(object);
            }
            index = end;
        } else {
            index = start + 1;
        }
    }
    objects
}

fn matching_answer_items(
    body: &str,
    bundle: &FreshChallengeBundleV1,
) -> BTreeSet<String> {
    let mut matched = BTreeSet::new();
    for object in extract_json_objects(body) {
        if let Ok(solution) = serde_json::from_str::<ExactLinearSolutionV1>(object) {
            for entry in &bundle.sealed_oracle_packet.entries {
                if solution == entry.solution {
                    matched.insert(entry.item_id.clone());
                }
            }
        }
    }
    matched
}

fn analyze_response_body(
    body: &str,
    bundle: &FreshChallengeBundleV1,
) -> (SearchExposureClassV1, Vec<String>, SearchFindingMethodV1, u16) {
    let body_words = normalized_words(body);
    let answer_items = matching_answer_items(body, bundle);
    if !answer_items.is_empty() {
        return (
            SearchExposureClassV1::AnswerBearing,
            answer_items.into_iter().collect(),
            SearchFindingMethodV1::ExactMatch,
            1_000,
        );
    }

    let mut context_items = BTreeSet::new();
    let mut exact_context = false;
    let mut fuzzy_context = false;
    for task in &bundle.public_packet.tasks {
        let task_words = normalized_words(&task.task_text);
        if task_words.len() >= 8 && contains_word_sequence(&body_words, &task_words) {
            exact_context = true;
            context_items.insert(task.item_id.clone());
            continue;
        }
        // This conservative lexical rule is only a candidate duplicate detector:
        // require at least two distinct shared 6-token shingles. It is not semantic.
        if task_words.len() >= 12 && shared_shingle_count(&body_words, &task_words, 6) >= 2 {
            fuzzy_context = true;
            context_items.insert(task.item_id.clone());
        }
    }
    if !context_items.is_empty() {
        return (
            SearchExposureClassV1::ContextBearing,
            context_items.into_iter().collect(),
            if exact_context {
                SearchFindingMethodV1::ExactMatch
            } else {
                SearchFindingMethodV1::FuzzyText
            },
            if exact_context { 1_000 } else { 800 },
        );
    }

    let mut metadata_items = BTreeSet::new();
    for task in &bundle.public_packet.tasks {
        let item_words = normalized_words(&task.item_id);
        if contains_word_sequence(&body_words, &item_words) {
            metadata_items.insert(task.item_id.clone());
        }
    }
    let metadata_terms = [
        bundle.public_packet.batch_id.as_str(),
        bundle.public_packet.benchmark_id.as_str(),
        bundle.public_packet.benchmark_version.as_str(),
        bundle.public_packet.task_family.as_str(),
    ];
    let metadata_found = metadata_terms.iter().any(|term| {
        let words = normalized_words(term);
        contains_word_sequence(&body_words, &words)
    });
    if metadata_found || !metadata_items.is_empty() {
        return (
            SearchExposureClassV1::MetadataOnly,
            metadata_items.into_iter().collect(),
            SearchFindingMethodV1::ExactMatch,
            1_000,
        );
    }
    (
        SearchExposureClassV1::Unknown,
        Vec::new(),
        SearchFindingMethodV1::Other,
        0,
    )
}

/// Deterministically match evaluator-retained response bodies against task
/// metadata, exact/near-exact task text, and sealed exact-answer objects.
///
/// Semantic paraphrase detection is intentionally not attempted. Every
/// response-bearing allowed event must either have a retained body record or
/// be explicitly listed as unavailable. Raw body text is not copied to output.
pub fn analyze_search_time_response_bodies(
    trace: &SearchTimeExposureTraceV1,
    bundle: &FreshChallengeBundleV1,
    response_bodies: &[SearchCapturedResponseBodyV1],
    unavailable_body_event_sequences: &[u64],
) -> Result<SearchTimeExposureFindingsV1, String> {
    let trace_audit = audit_search_time_trace(trace)?;
    if !trace_audit.valid {
        return Err("search-time trace failed integrity audit".to_owned());
    }
    if trace.challenge_packet_sha256 != bundle.public_packet.public_packet_sha256 {
        return Err("search trace and fresh challenge packet are not bound".to_owned());
    }
    if response_bodies.len() > SEARCH_MAX_EVENTS
        || unavailable_body_event_sequences.len() > SEARCH_MAX_EVENTS
    {
        return Err("body evidence count exceeds search event limit".to_owned());
    }
    let expected_events: BTreeMap<u64, &SearchTimeEventV1> = trace.events.iter()
        .filter(|event| {
            event.disposition == SearchEventDispositionV1::Allowed && event.response_bytes > 0
        })
        .map(|event| (event.sequence, event))
        .collect();
    let mut evidence_sequences = BTreeSet::new();
    let mut total_body_bytes = 0_usize;
    let mut findings = Vec::with_capacity(response_bodies.len());
    for record in response_bodies {
        if !evidence_sequences.insert(record.event_sequence) {
            return Err("duplicate retained response-body event sequence".to_owned());
        }
        if record.content_type.trim().is_empty() {
            return Err("response-body evidence requires a content type".to_owned());
        }
        let event = expected_events.get(&record.event_sequence)
            .ok_or_else(|| format!("response-body evidence references non-body event {}", record.event_sequence))?;
        let byte_len = record.body_text.len();
        if byte_len > SEARCH_MAX_BODY_BYTES {
            return Err("individual retained response body exceeds 16 MiB safety bound".to_owned());
        }
        total_body_bytes = total_body_bytes.checked_add(byte_len)
            .ok_or_else(|| "total retained response body size overflowed".to_owned())?;
        if total_body_bytes > SEARCH_MAX_TOTAL_BODY_BYTES {
            return Err("total retained response bodies exceed 128 MiB safety bound".to_owned());
        }
        validate_digest(&record.response_sha256, "response_body.response_sha256")?;
        let actual_digest = search_response_body_digest(&record.body_text);
        if actual_digest != record.response_sha256
            || event.response_sha256.as_deref() != Some(record.response_sha256.as_str())
            || event.response_bytes != byte_len as u64
        {
            return Err(format!("retained response body does not match event {} receipt", record.event_sequence));
        }
        let (exposure_class, matched_item_ids, method, confidence_permille) =
            analyze_response_body(&record.body_text, bundle);
        findings.push(SearchExposureFindingV1 {
            event_sequence: record.event_sequence,
            exposure_class,
            matched_item_ids,
            evidence_sha256: record.response_sha256.clone(),
            method,
            confidence_permille,
        });
    }
    let mut unavailable = BTreeSet::new();
    for sequence in unavailable_body_event_sequences {
        if !unavailable.insert(*sequence) {
            return Err("duplicate unavailable body event sequence".to_owned());
        }
        if !expected_events.contains_key(sequence) {
            return Err(format!("unavailable body sequence {sequence} is not a response-bearing allowed event"));
        }
    }
    if !evidence_sequences.is_disjoint(&unavailable) {
        return Err("a response body cannot be both retained and unavailable".to_owned());
    }
    let accounted: BTreeSet<u64> = evidence_sequences.union(&unavailable).copied().collect();
    let expected: BTreeSet<u64> = expected_events.keys().copied().collect();
    if accounted != expected {
        return Err("every response-bearing allowed event must be retained or explicitly unavailable".to_owned());
    }
    findings.sort_by_key(|finding| finding.event_sequence);
    let mut artifact = SearchTimeExposureFindingsV1 {
        schema: SEARCH_FINDINGS_SCHEMA.to_owned(),
        run_id: trace.run_id.clone(),
        trace_root_sha256: trace.trace_root_sha256.clone(),
        response_body_event_count: u64::try_from(expected_events.len())
            .map_err(|_| "response body event count exceeds u64")?,
        response_body_evidence_count: u64::try_from(response_bodies.len())
            .map_err(|_| "response body evidence count exceeds u64")?,
        unavailable_body_event_sequences: unavailable.into_iter().collect(),
        findings,
        findings_sha256: String::new(),
        claim_boundary: "deterministic_exact_and_lexical_duplicate_matching_only;semantic_paraphrase_and_causal_influence_not_established;raw_response_bodies_remain_evaluator_only".to_owned(),
    };
    artifact.findings_sha256 = findings_digest(&artifact)?;
    Ok(artifact)
}

fn findings_digest(findings: &SearchTimeExposureFindingsV1) -> Result<String, String> {
    let mut normalized = findings.clone();
    normalized.findings_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize search findings: {error}"))?;
    Ok(hash_domain(b"search-time-findings.v1", &bytes))
}

pub fn seal_search_time_events(events: &mut [SearchTimeEventV1]) -> Result<(), String> {
    let mut previous = SEARCH_TRACE_GENESIS.to_owned();
    let mut last_timestamp: Option<String> = None;
    for (index, event) in events.iter_mut().enumerate() {
        let expected_sequence =
            u64::try_from(index + 1).map_err(|_| "search event sequence exceeds u64")?;
        if event.sequence != expected_sequence {
            return Err("search event sequences must start at 1 and be contiguous".to_owned());
        }
        if event.invocation_id.trim().is_empty() || event.provider_or_tool.trim().is_empty() {
            return Err(
                "search event invocation and provider/tool identities must be nonempty".to_owned(),
            );
        }
        parse_timestamp(&event.occurred_at_utc, "search_event.occurred_at_utc")?;
        if last_timestamp
            .as_ref()
            .is_some_and(|last| event.occurred_at_utc < *last)
        {
            return Err("search event timestamps must be monotonic with sequence".to_owned());
        }
        validate_digest(&event.destination_sha256, "destination_sha256")?;
        validate_digest(&event.request_sha256, "request_sha256")?;
        match event.disposition {
            SearchEventDispositionV1::Blocked => {
                if event.response_bytes != 0 || event.response_sha256.is_some() {
                    return Err(
                        "blocked search events cannot claim received response bytes".to_owned()
                    );
                }
            }
            SearchEventDispositionV1::Allowed => {
                match (event.response_bytes, event.response_sha256.as_deref()) {
                    (0, None) => {}
                    (0, Some(_)) => {
                        return Err("zero-byte response cannot have a response digest".to_owned());
                    }
                    (_, None) => {
                        return Err("accepted response bytes require a response digest".to_owned());
                    }
                    (_, Some(digest)) => validate_digest(digest, "response_sha256")?,
                }
            }
        }
        event.previous_event_sha256 = previous.clone();
        event.event_sha256.clear();
        event.event_sha256 = event_digest(event)?;
        previous = event.event_sha256.clone();
        last_timestamp = Some(event.occurred_at_utc.clone());
    }
    Ok(())
}

pub fn seal_search_time_trace(trace: &mut SearchTimeExposureTraceV1) -> Result<(), String> {
    if trace.events.len() > SEARCH_MAX_EVENTS {
        return Err(format!("search trace exceeds {SEARCH_MAX_EVENTS} events"));
    }
    seal_search_time_events(&mut trace.events)?;
    trace.event_count =
        u64::try_from(trace.events.len()).map_err(|_| "search event count exceeds u64")?;
    trace.trace_root_sha256.clear();
    trace.trace_root_sha256 = trace_digest(trace)?;
    Ok(())
}

/// Audit the exact event chain and disclose the perimeter's claim ceiling.
/// This cannot prove that network/tool instrumentation captured every channel.
pub fn audit_search_time_trace(
    trace: &SearchTimeExposureTraceV1,
) -> Result<SearchTimeExposureAuditV1, String> {
    if trace.schema != SEARCH_TRACE_SCHEMA {
        return Err(format!("trace schema must be {SEARCH_TRACE_SCHEMA}"));
    }
    if trace.run_id.trim().is_empty()
        || trace.benchmark_id.trim().is_empty()
        || trace.benchmark_version.trim().is_empty()
        || trace.policy_id.trim().is_empty()
    {
        return Err("trace run, benchmark and policy identities must be nonempty".to_owned());
    }
    if trace.events.len() > SEARCH_MAX_EVENTS {
        return Err(format!("search trace exceeds {SEARCH_MAX_EVENTS} events"));
    }
    for (digest, label) in [
        (&trace.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&trace.challenge_packet_sha256, "challenge_packet_sha256"),
        (&trace.network_policy_sha256, "network_policy_sha256"),
        (&trace.enforcement_build_sha256, "enforcement_build_sha256"),
        (&trace.trace_root_sha256, "trace_root_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    if trace.instrumented_channels.is_empty() {
        return Err("trace must declare at least one instrumented information channel".to_owned());
    }
    let instrumented: BTreeSet<SearchInformationChannelV1> =
        trace.instrumented_channels.iter().copied().collect();
    let unobserved: BTreeSet<SearchInformationChannelV1> =
        trace.unobserved_channels.iter().copied().collect();
    let allowlisted_destinations: BTreeSet<&str> = trace
        .allowlisted_destination_sha256s
        .iter()
        .map(String::as_str)
        .collect();
    for digest in &trace.allowlisted_destination_sha256s {
        validate_digest(digest, "allowlisted_destination_sha256s[]")?;
    }
    if instrumented.len() != trace.instrumented_channels.len()
        || unobserved.len() != trace.unobserved_channels.len()
        || allowlisted_destinations.len() != trace.allowlisted_destination_sha256s.len()
        || trace.allowlisted_destination_sha256s.len() > SEARCH_MAX_ALLOWLIST_DESTINATIONS
        || !instrumented.is_disjoint(&unobserved)
    {
        return Err("instrumented/unobserved channels and allowlist destinations must be unique and disjoint".to_owned());
    }
    match trace.access_mode {
        SearchAccessModeV1::Allowlisted if allowlisted_destinations.is_empty() => {
            return Err(
                "allowlisted access mode requires at least one destination digest".to_owned(),
            );
        }
        SearchAccessModeV1::NetworkDisabled | SearchAccessModeV1::Unrestricted
            if !allowlisted_destinations.is_empty() =>
        {
            return Err(
                "network-disabled and unrestricted modes must not carry an allowlist".to_owned(),
            );
        }
        _ => {}
    }
    let declared_channels: BTreeSet<SearchInformationChannelV1> =
        instrumented.union(&unobserved).copied().collect();
    let all_channel_types_accounted_for = declared_channels == all_known_channels();

    let mut previous = SEARCH_TRACE_GENESIS.to_owned();
    let mut last_timestamp: Option<String> = None;
    let mut allowed_count = 0_u64;
    let mut blocked_count = 0_u64;
    let mut timestamps_valid = true;
    let mut request_receipts_complete = true;
    let mut response_receipts_complete = true;
    let mut blocked_events_clean = true;
    let mut network_disabled_respected = true;
    let mut allowlist_respected = true;
    let mut channels_instrumented = true;
    let mut invocation_ids = BTreeSet::new();
    let mut sequence_complete = true;
    let mut event_hashes_valid = true;
    let mut issues = Vec::new();

    for (index, event) in trace.events.iter().enumerate() {
        let expected_sequence =
            u64::try_from(index + 1).map_err(|_| "search event sequence exceeds u64")?;
        sequence_complete &= event.sequence == expected_sequence;
        if event.sequence != expected_sequence {
            issues.push(format!(
                "event at index {index} has non-contiguous sequence"
            ));
        }
        if event.invocation_id.trim().is_empty()
            || event.provider_or_tool.trim().is_empty()
            || !invocation_ids.insert(event.invocation_id.as_str())
        {
            request_receipts_complete = false;
            issues.push(format!(
                "event {} has missing or duplicate invocation/provider identity",
                event.sequence
            ));
        }
        request_receipts_complete &= valid_digest(&event.request_sha256);
        if !valid_digest(&event.request_sha256) {
            issues.push(format!("event {} lacks request digest", event.sequence));
        }
        if !valid_digest(&event.destination_sha256) {
            request_receipts_complete = false;
            issues.push(format!("event {} lacks destination digest", event.sequence));
        }
        if let Some(response_digest) = event.response_sha256.as_deref() {
            response_receipts_complete &= valid_digest(response_digest);
            if !valid_digest(response_digest) {
                issues.push(format!(
                    "event {} has invalid response digest",
                    event.sequence
                ));
            }
        }
        match event.disposition {
            SearchEventDispositionV1::Allowed => {
                allowed_count = allowed_count
                    .checked_add(1)
                    .ok_or_else(|| "allowed event count overflowed".to_owned())?;
                if event.response_bytes > 0 && event.response_sha256.is_none() {
                    response_receipts_complete = false;
                }
                if event.response_bytes == 0 && event.response_sha256.is_some() {
                    response_receipts_complete = false;
                }
                if event.channel.requires_network() {
                    match trace.access_mode {
                        SearchAccessModeV1::NetworkDisabled => {
                            network_disabled_respected = false;
                            issues.push(format!(
                                "network-disabled trace contains allowed network event {}",
                                event.sequence
                            ));
                        }
                        SearchAccessModeV1::Allowlisted
                            if !allowlisted_destinations
                                .contains(event.destination_sha256.as_str()) =>
                        {
                            allowlist_respected = false;
                            issues.push(format!("allowed network event {} targets a destination outside the declared allowlist", event.sequence));
                        }
                        SearchAccessModeV1::Allowlisted | SearchAccessModeV1::Unrestricted => {}
                    }
                }
            }
            SearchEventDispositionV1::Blocked => {
                blocked_count = blocked_count
                    .checked_add(1)
                    .ok_or_else(|| "blocked event count overflowed".to_owned())?;
                if event.response_bytes != 0 || event.response_sha256.is_some() {
                    blocked_events_clean = false;
                    issues.push(format!(
                        "blocked event {} claims response bytes",
                        event.sequence
                    ));
                }
            }
        }
        if !instrumented.contains(&event.channel) {
            channels_instrumented = false;
            issues.push(format!(
                "event {} channel is not declared instrumented",
                event.sequence
            ));
        }
        if parse_timestamp(&event.occurred_at_utc, "search_event.occurred_at_utc").is_err() {
            timestamps_valid = false;
            issues.push(format!(
                "event {} has invalid UTC timestamp",
                event.sequence
            ));
        }
        if last_timestamp
            .as_ref()
            .is_some_and(|last| event.occurred_at_utc < *last)
        {
            timestamps_valid = false;
            issues.push(format!(
                "event {} timestamp moves backwards",
                event.sequence
            ));
        }
        if event.previous_event_sha256 != previous {
            event_hashes_valid = false;
            issues.push(format!(
                "event {} previous digest does not match chain",
                event.sequence
            ));
        }
        let expected_event_digest = event_digest(event)?;
        if event.event_sha256 != expected_event_digest {
            event_hashes_valid = false;
            issues.push(format!(
                "event {} digest does not match contents",
                event.sequence
            ));
        }
        previous = event.event_sha256.clone();
        last_timestamp = Some(event.occurred_at_utc.clone());
    }

    let trace_root_matches = trace_digest(trace)? == trace.trace_root_sha256;
    if !trace_root_matches {
        issues.push("trace root digest does not match trace contents".to_owned());
    }
    let hash_chain_replayed = sequence_complete && event_hashes_valid;
    if !hash_chain_replayed {
        issues.push("search event hash chain could not be replayed".to_owned());
    }
    if trace.event_count != trace.events.len() as u64 {
        sequence_complete = false;
        issues.push("declared event_count does not match trace events".to_owned());
    }
    let no_unobserved_channels =
        trace.unobserved_channels.is_empty() && all_channel_types_accounted_for;
    if !all_channel_types_accounted_for {
        issues.push("instrumented plus unobserved declarations do not cover every known information channel".to_owned());
    }
    if !trace.unobserved_channels.is_empty() {
        issues.push(
            "trace declares information channels outside the instrumentation perimeter".to_owned(),
        );
    }
    let valid = trace_root_matches
        && hash_chain_replayed
        && sequence_complete
        && timestamps_valid
        && request_receipts_complete
        && response_receipts_complete
        && blocked_events_clean
        && network_disabled_respected
        && allowlist_respected
        && channels_instrumented
        && all_channel_types_accounted_for
        && no_unobserved_channels;
    let coverage_status = if no_unobserved_channels && channels_instrumented {
        "all_known_channel_types_declared_instrumented_not_independently_qualified"
    } else {
        "partial_or_unobserved_information_channels"
    }
    .to_owned();
    Ok(SearchTimeExposureAuditV1 {
        schema: SEARCH_AUDIT_SCHEMA.to_owned(),
        valid,
        hash_chain_replayed,
        trace_root_matches,
        sequence_complete,
        timestamps_valid_and_monotonic: timestamps_valid,
        request_receipts_complete,
        accepted_response_receipts_complete: response_receipts_complete,
        blocked_events_have_no_response_bytes: blocked_events_clean,
        network_disabled_policy_respected: network_disabled_respected,
        allowlist_policy_respected: allowlist_respected,
        all_event_channels_declared_instrumented: channels_instrumented,
        all_known_channel_types_accounted_for: all_channel_types_accounted_for,
        no_unobserved_channels_declared: no_unobserved_channels,
        complete_mediation_verified: false,
        independently_verified: false,
        event_count: u64::try_from(trace.events.len())
            .map_err(|_| "event count exceeds u64")?,
        allowed_event_count: allowed_count,
        blocked_event_count: blocked_count,
        run_id: trace.run_id.clone(),
        trace_root_sha256: trace.trace_root_sha256.clone(),
        coverage_status,
        issues,
        claim_boundary: "integrity_of_declared_trace_checked;does_not_prove_complete_channel_mediation_or_retrieval_influence;external_proxy_runner_and_tool_attestations_not_authenticated".to_owned(),
    })
}

pub fn audit_search_time_findings(
    trace: &SearchTimeExposureTraceV1,
    findings: &SearchTimeExposureFindingsV1,
) -> Result<SearchTimeExposureFindingsAuditV1, String> {
    if findings.schema != SEARCH_FINDINGS_SCHEMA {
        return Err(format!("findings schema must be {SEARCH_FINDINGS_SCHEMA}"));
    }
    if findings.run_id != trace.run_id || findings.trace_root_sha256 != trace.trace_root_sha256 {
        return Err("findings are not bound to the exact trace and run".to_owned());
    }
    validate_digest(&findings.findings_sha256, "findings_sha256")?;
    let digest_matches = findings_digest(findings)? == findings.findings_sha256;
    if !digest_matches {
        return Err("search exposure findings digest does not match contents".to_owned());
    }
    let trace_audit = audit_search_time_trace(trace)?;
    if !trace_audit.valid {
        return Err("search-time trace failed integrity audit".to_owned());
    }
    if findings.findings.len() > SEARCH_MAX_EVENTS {
        return Err(format!("search findings exceed {SEARCH_MAX_EVENTS}"));
    }
    let event_by_sequence: BTreeMap<u64, &SearchTimeEventV1> = trace
        .events
        .iter()
        .map(|event| (event.sequence, event))
        .collect();
    let mut seen = BTreeSet::new();
    let mut metadata = 0_u64;
    let mut context = 0_u64;
    let mut answer = 0_u64;
    let mut unknown = 0_u64;
    let mut matched_items = 0_u64;
    for finding in &findings.findings {
        if !seen.insert(finding.event_sequence) {
            return Err(
                "at most one normalized exposure finding is allowed per event sequence".to_owned(),
            );
        }
        let event = event_by_sequence
            .get(&finding.event_sequence)
            .ok_or_else(|| {
                format!(
                    "finding references unknown event {}",
                    finding.event_sequence
                )
            })?;
        if event.disposition != SearchEventDispositionV1::Allowed {
            return Err(format!(
                "finding references blocked event {}",
                finding.event_sequence
            ));
        }
        validate_digest(&finding.evidence_sha256, "finding.evidence_sha256")?;
        if matches!(
            finding.exposure_class,
            SearchExposureClassV1::ContextBearing | SearchExposureClassV1::AnswerBearing
        ) && finding.matched_item_ids.is_empty()
        {
            return Err(
                "context/answer-bearing findings must reference at least one benchmark item"
                    .to_owned(),
            );
        }
        if event.response_bytes == 0 || event.response_sha256.is_none() {
            return Err("exposure finding requires a received response body".to_owned());
        }
        if finding.matched_item_ids.len() > 100_000 {
            return Err("finding has too many matched item IDs".to_owned());
        }
        if finding.confidence_permille > 1_000 {
            return Err("finding confidence must be in [0, 1000] permille".to_owned());
        }
        let mut unique_items = BTreeSet::new();
        for item_id in &finding.matched_item_ids {
            if item_id.trim().is_empty() || !unique_items.insert(item_id.as_str()) {
                return Err("finding item IDs must be unique and nonempty".to_owned());
            }
        }
        matched_items = matched_items
            .checked_add(unique_items.len() as u64)
            .ok_or_else(|| "matched item count overflowed".to_owned())?;
        match finding.exposure_class {
            SearchExposureClassV1::MetadataOnly => metadata += 1,
            SearchExposureClassV1::ContextBearing => context += 1,
            SearchExposureClassV1::AnswerBearing => answer += 1,
            SearchExposureClassV1::Unknown => unknown += 1,
        }
    }
    Ok(SearchTimeExposureFindingsAuditV1 {
        schema: SEARCH_FINDINGS_AUDIT_SCHEMA.to_owned(),
        valid: digest_matches,
        findings_digest_matches: digest_matches,
        all_findings_reference_allowed_events: true,
        unique_event_sequences: true,
        metadata_only_count: metadata,
        context_bearing_count: context,
        answer_bearing_count: answer,
        unknown_count: unknown,
        matched_item_reference_count: matched_items,
        findings_sha256: findings.findings_sha256.clone(),
        claim_boundary: "validates_evaluator_supplied_classification_record_bindings_only;does_not_perform_semantic_retrieval_matching_or_prove_score_influence".to_owned(),
        issues: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        hash_domain(b"search-time-test", label.as_bytes())
    }

    fn event(
        sequence: u64,
        channel: SearchInformationChannelV1,
        disposition: SearchEventDispositionV1,
        time: &str,
    ) -> SearchTimeEventV1 {
        SearchTimeEventV1 {
            sequence,
            invocation_id: format!("invocation-{sequence}"),
            channel,
            provider_or_tool: "fixture-tool".to_owned(),
            destination_sha256: digest("destination"),
            request_sha256: digest(&format!("request-{sequence}")),
            response_sha256: if disposition == SearchEventDispositionV1::Allowed {
                Some(digest(&format!("response-{sequence}")))
            } else {
                None
            },
            response_bytes: if disposition == SearchEventDispositionV1::Allowed {
                32
            } else {
                0
            },
            occurred_at_utc: time.to_owned(),
            disposition,
            previous_event_sha256: String::new(),
            event_sha256: String::new(),
        }
    }

    fn trace_fixture() -> SearchTimeExposureTraceV1 {
        let mut trace = SearchTimeExposureTraceV1 {
            schema: SEARCH_TRACE_SCHEMA.to_owned(),
            run_id: "run-001".to_owned(),
            benchmark_id: "fresh-benchmark".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_checkpoint_sha256: digest("checkpoint"),
            challenge_packet_sha256: digest("challenge"),
            network_policy_sha256: digest("policy"),
            enforcement_build_sha256: digest("enforcer"),
            policy_id: "network-isolated-v1".to_owned(),
            access_mode: SearchAccessModeV1::NetworkDisabled,
            allowlisted_destination_sha256s: Vec::new(),
            instrumented_channels: vec![
                SearchInformationChannelV1::WebSearch,
                SearchInformationChannelV1::BrowserNavigation,
                SearchInformationChannelV1::HttpApi,
                SearchInformationChannelV1::RemoteMcp,
                SearchInformationChannelV1::VectorStore,
                SearchInformationChannelV1::LocalCache,
                SearchInformationChannelV1::MountedFile,
                SearchInformationChannelV1::AgentMemory,
                SearchInformationChannelV1::Other,
            ],
            unobserved_channels: Vec::new(),
            events: vec![
                event(
                    1,
                    SearchInformationChannelV1::LocalCache,
                    SearchEventDispositionV1::Allowed,
                    "2026-10-31T18:00:00Z",
                ),
                event(
                    2,
                    SearchInformationChannelV1::WebSearch,
                    SearchEventDispositionV1::Blocked,
                    "2026-10-31T18:00:01Z",
                ),
            ],
            event_count: 0,
            trace_root_sha256: String::new(),
            claim_boundary: "synthetic fixture".to_owned(),
        };
        seal_search_time_trace(&mut trace).unwrap();
        trace
    }

    fn findings_for_trace(trace: &SearchTimeExposureTraceV1) -> SearchTimeExposureFindingsV1 {
        let mut findings = SearchTimeExposureFindingsV1 {
            schema: SEARCH_FINDINGS_SCHEMA.to_owned(),
            run_id: trace.run_id.clone(),
            trace_root_sha256: trace.trace_root_sha256.clone(),
            findings: vec![SearchExposureFindingV1 {
                event_sequence: 1,
                exposure_class: SearchExposureClassV1::MetadataOnly,
                matched_item_ids: vec!["item-001".to_owned()],
                evidence_sha256: digest("matching-evidence"),
                method: SearchFindingMethodV1::ExactMatch,
                confidence_permille: 1_000,
            }],
            findings_sha256: String::new(),
            claim_boundary: "evaluator fixture".to_owned(),
        };
        findings.findings_sha256 = findings_digest(&findings).unwrap();
        findings
    }

    #[test]
    fn strict_trace_and_finding_schemas_match_rust_serialization() {
        let trace_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/search-time-exposure-trace.schema.json"
        ))
        .unwrap();
        let audit_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/search-time-exposure-audit.schema.json"
        ))
        .unwrap();
        let findings_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/search-time-exposure-findings.schema.json"
        ))
        .unwrap();
        let findings_audit_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/search-time-exposure-findings-audit.schema.json"
        ))
        .unwrap();

        let trace = trace_fixture();
        let trace_audit = audit_search_time_trace(&trace).unwrap();
        let findings = findings_for_trace(&trace);
        let findings_audit = audit_search_time_findings(&trace, &findings).unwrap();
        for (schema, value, version) in [
            (
                &trace_schema,
                serde_json::to_value(&trace).unwrap(),
                SEARCH_TRACE_SCHEMA,
            ),
            (
                &audit_schema,
                serde_json::to_value(&trace_audit).unwrap(),
                SEARCH_AUDIT_SCHEMA,
            ),
            (
                &findings_schema,
                serde_json::to_value(&findings).unwrap(),
                SEARCH_FINDINGS_SCHEMA,
            ),
            (
                &findings_audit_schema,
                serde_json::to_value(&findings_audit).unwrap(),
                SEARCH_FINDINGS_AUDIT_SCHEMA,
            ),
        ] {
            assert_eq!(
                schema["properties"]["schema"]["const"].as_str(),
                Some(version)
            );
            let required: BTreeSet<String> = schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_owned())
                .collect();
            let properties: BTreeSet<String> = schema["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            let actual: BTreeSet<String> = value.as_object().unwrap().keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }
        let expected_event_fields: BTreeSet<String> = trace_schema["properties"]["events"]["items"]
            ["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let actual_event_fields: BTreeSet<String> = serde_json::to_value(&trace.events[0])
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(expected_event_fields, actual_event_fields);

        let expected_finding_fields: BTreeSet<String> =
            findings_schema["properties"]["findings"]["items"]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
        let actual_finding_fields: BTreeSet<String> = serde_json::to_value(&findings.findings[0])
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(expected_finding_fields, actual_finding_fields);
    }

    #[test]
    fn exact_hash_chain_replays_for_valid_trace() {
        let trace = trace_fixture();
        let audit = audit_search_time_trace(&trace).unwrap();
        assert!(audit.valid);
        assert!(audit.hash_chain_replayed);
        assert!(audit.trace_root_matches);
        assert_eq!(audit.allowed_event_count, 1);
        assert_eq!(audit.blocked_event_count, 1);
        assert!(!audit.complete_mediation_verified);
        assert!(!audit.independently_verified);
    }

    #[test]
    fn tampered_event_or_root_fails_chain_replay() {
        let mut trace = trace_fixture();
        trace.events[0].request_sha256 = digest("tampered-request");
        trace.trace_root_sha256 = trace_digest(&trace).unwrap();
        let audit = audit_search_time_trace(&trace).unwrap();
        assert!(!audit.valid);
        assert!(!audit.hash_chain_replayed);

        let mut trace = trace_fixture();
        trace.trace_root_sha256 = digest("wrong-root");
        assert!(!audit_search_time_trace(&trace).unwrap().trace_root_matches);
    }

    #[test]
    fn network_disabled_policy_rejects_accepted_search_http_browser_or_remote_mcp() {
        for channel in [
            SearchInformationChannelV1::WebSearch,
            SearchInformationChannelV1::BrowserNavigation,
            SearchInformationChannelV1::HttpApi,
            SearchInformationChannelV1::RemoteMcp,
        ] {
            let mut trace = trace_fixture();
            trace.events[0] = event(
                1,
                channel,
                SearchEventDispositionV1::Allowed,
                "2026-10-31T18:00:00Z",
            );
            seal_search_time_trace(&mut trace).unwrap();
            let audit = audit_search_time_trace(&trace).unwrap();
            assert!(!audit.valid);
            assert!(!audit.network_disabled_policy_respected);
        }
    }

    #[test]
    fn allowlisted_mode_accepts_only_declared_destination_digests() {
        let mut trace = trace_fixture();
        trace.access_mode = SearchAccessModeV1::Allowlisted;
        trace.allowlisted_destination_sha256s = vec![digest("allowed-search-destination")];
        trace.events[0] = event(
            1,
            SearchInformationChannelV1::WebSearch,
            SearchEventDispositionV1::Allowed,
            "2026-10-31T18:00:00Z",
        );
        trace.events[0].destination_sha256 = digest("allowed-search-destination");
        seal_search_time_trace(&mut trace).unwrap();
        let audit = audit_search_time_trace(&trace).unwrap();
        assert!(audit.valid);
        assert!(audit.allowlist_policy_respected);

        let mut trace = trace;
        trace.events[0].destination_sha256 = digest("unlisted-search-destination");
        trace.trace_root_sha256 = trace_digest(&trace).unwrap();
        let audit = audit_search_time_trace(&trace).unwrap();
        assert!(!audit.valid);
        assert!(!audit.allowlist_policy_respected);
    }

    #[test]
    fn blocked_event_must_not_claim_response_bytes() {
        let mut trace = trace_fixture();
        trace.events[1].response_sha256 = Some(digest("impossible-response"));
        trace.events[1].response_bytes = 10;
        assert!(seal_search_time_trace(&mut trace).is_err());
    }

    #[test]
    fn unobserved_or_undeclared_channels_prevent_validity() {
        let mut trace = trace_fixture();
        trace
            .instrumented_channels
            .retain(|channel| *channel != SearchInformationChannelV1::AgentMemory);
        trace
            .unobserved_channels
            .push(SearchInformationChannelV1::AgentMemory);
        seal_search_time_trace(&mut trace).unwrap();
        let audit = audit_search_time_trace(&trace).unwrap();
        assert!(!audit.valid);
        assert!(!audit.no_unobserved_channels_declared);
        assert!(audit.all_known_channel_types_accounted_for);

        let mut trace = trace_fixture();
        trace
            .instrumented_channels
            .retain(|channel| *channel != SearchInformationChannelV1::LocalCache);
        seal_search_time_trace(&mut trace).unwrap();
        assert!(
            !audit_search_time_trace(&trace)
                .unwrap()
                .all_event_channels_declared_instrumented
        );
    }

    #[test]
    fn non_monotonic_and_impossible_event_times_fail_closed() {
        let mut trace = trace_fixture();
        trace.events[1].occurred_at_utc = "2026-02-30T18:00:01Z".to_owned();
        seal_search_time_events(&mut trace.events).unwrap_err();

        let mut trace = trace_fixture();
        trace.events[1].occurred_at_utc = "2026-10-31T17:59:59Z".to_owned();
        assert!(seal_search_time_events(&mut trace.events).is_err());
    }

    #[test]
    fn findings_bind_to_trace_and_only_allowed_events() {
        let trace = trace_fixture();
        let findings = findings_for_trace(&trace);
        let audit = audit_search_time_findings(&trace, &findings).unwrap();
        assert!(audit.valid);
        assert_eq!(audit.metadata_only_count, 1);
        assert_eq!(audit.answer_bearing_count, 0);

        let mut blocked = findings_for_trace(&trace);
        blocked.findings[0].event_sequence = 2;
        blocked.findings_sha256 = findings_digest(&blocked).unwrap();
        assert!(audit_search_time_findings(&trace, &blocked).is_err());
    }

    #[test]
    fn findings_tampering_and_duplicate_event_sequences_fail() {
        let trace = trace_fixture();
        let mut findings = findings_for_trace(&trace);
        findings.findings[0].confidence_permille = 900;
        assert!(audit_search_time_findings(&trace, &findings).is_err());

        let mut findings = findings_for_trace(&trace);
        findings.findings.push(findings.findings[0].clone());
        findings.findings_sha256 = findings_digest(&findings).unwrap();
        assert!(audit_search_time_findings(&trace, &findings).is_err());
    }
}
