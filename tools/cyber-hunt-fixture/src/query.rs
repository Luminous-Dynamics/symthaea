use crate::{Event, SolverManifest, EVENT_SCHEMA_VERSION, GENERATOR_VERSION, SOLVER_MANIFEST_VERSION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const QUERY_SCHEMA_VERSION: &str = "cyber-hunt-query-v1";
const QUERY_RECEIPT_VERSION: &str = "cyber-hunt-query-receipt-v1";
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_BATCH_BYTES: usize = 16 * 1024 * 1024;
const MAX_BATCH_REQUESTS: usize = 100;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum TextField {
    RecordId,
    SourceEventId,
    DeliveryId,
    HostId,
    PrincipalId,
    SourceAddress,
    Destination,
    DevicePosture,
    Source,
    EventType,
    Result,
    Summary,
    IntegrityState,
    ChangeTicket,
}

impl TextField {
    fn value<'a>(&self, event: &'a Event) -> Option<&'a str> {
        match self {
            Self::RecordId => Some(&event.record_id),
            Self::SourceEventId => Some(&event.source_event_id),
            Self::DeliveryId => Some(&event.delivery_id),
            Self::HostId => Some(&event.host_id),
            Self::PrincipalId => Some(&event.principal_id),
            Self::SourceAddress => Some(&event.source_address),
            Self::Destination => Some(&event.destination),
            Self::DevicePosture => Some(&event.device_posture),
            Self::Source => Some(&event.source),
            Self::EventType => Some(&event.event_type),
            Self::Result => Some(&event.result),
            Self::Summary => Some(&event.summary),
            Self::IntegrityState => Some(&event.integrity_state),
            Self::ChangeTicket => event.change_ticket.as_deref(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum TimeField {
    EventTimeMs,
    IngestTimeMs,
}

impl TimeField {
    fn value(&self, event: &Event) -> i64 {
        match self {
            Self::EventTimeMs => event.event_time_ms,
            Self::IngestTimeMs => event.ingest_time_ms,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Filter {
    TextEquals {
        field: TextField,
        value: String,
    },
    TimeRange {
        field: TimeField,
        from_inclusive_ms: Option<i64>,
        to_exclusive_ms: Option<i64>,
    },
    BytesSentAtLeast {
        minimum: u64,
    },
}

impl Filter {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::TextEquals { value, .. } => {
                if value.trim().is_empty() || value.len() > 256 {
                    return Err("text_equals requires a non-empty value of at most 256 bytes".to_string());
                }
            }
            Self::TimeRange {
                from_inclusive_ms,
                to_exclusive_ms,
                ..
            } => {
                if from_inclusive_ms.is_none() && to_exclusive_ms.is_none() {
                    return Err("time_range requires at least one bound".to_string());
                }
                if let (Some(from), Some(to)) = (from_inclusive_ms, to_exclusive_ms) {
                    if from >= to {
                        return Err("time_range lower bound must precede the exclusive upper bound".to_string());
                    }
                }
            }
            Self::BytesSentAtLeast { .. } => {}
        }
        Ok(())
    }

    fn matches(&self, event: &Event) -> bool {
        match self {
            Self::TextEquals { field, value } => field.value(event) == Some(value.as_str()),
            Self::TimeRange {
                field,
                from_inclusive_ms,
                to_exclusive_ms,
            } => {
                let actual = field.value(event);
                match from_inclusive_ms {
                    Some(from) if actual < *from => return false,
                    _ => {}
                }
                match to_exclusive_ms {
                    Some(to) if actual >= *to => return false,
                    _ => {}
                }
                true
            }
            Self::BytesSentAtLeast { minimum } => event.bytes_sent >= *minimum,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum ProjectionField {
    RecordId,
    SourceEventId,
    DeliveryId,
    EventTimeMs,
    IngestTimeMs,
    HostId,
    PrincipalId,
    SourceAddress,
    Destination,
    BytesSent,
    DevicePosture,
    Source,
    EventType,
    Result,
    Summary,
    ChangeTicket,
    IntegrityState,
}

impl ProjectionField {
    fn key(&self) -> &'static str {
        match self {
            Self::RecordId => "record_id",
            Self::SourceEventId => "source_event_id",
            Self::DeliveryId => "delivery_id",
            Self::EventTimeMs => "event_time_ms",
            Self::IngestTimeMs => "ingest_time_ms",
            Self::HostId => "host_id",
            Self::PrincipalId => "principal_id",
            Self::SourceAddress => "source_address",
            Self::Destination => "destination",
            Self::BytesSent => "bytes_sent",
            Self::DevicePosture => "device_posture",
            Self::Source => "source",
            Self::EventType => "event_type",
            Self::Result => "result",
            Self::Summary => "summary",
            Self::ChangeTicket => "change_ticket",
            Self::IntegrityState => "integrity_state",
        }
    }

    fn value(&self, event: &Event) -> serde_json::Value {
        match self {
            Self::RecordId => serde_json::Value::String(event.record_id.clone()),
            Self::SourceEventId => serde_json::Value::String(event.source_event_id.clone()),
            Self::DeliveryId => serde_json::Value::String(event.delivery_id.clone()),
            Self::EventTimeMs => serde_json::Value::from(event.event_time_ms),
            Self::IngestTimeMs => serde_json::Value::from(event.ingest_time_ms),
            Self::HostId => serde_json::Value::String(event.host_id.clone()),
            Self::PrincipalId => serde_json::Value::String(event.principal_id.clone()),
            Self::SourceAddress => serde_json::Value::String(event.source_address.clone()),
            Self::Destination => serde_json::Value::String(event.destination.clone()),
            Self::BytesSent => serde_json::Value::from(event.bytes_sent),
            Self::DevicePosture => serde_json::Value::String(event.device_posture.clone()),
            Self::Source => serde_json::Value::String(event.source.clone()),
            Self::EventType => serde_json::Value::String(event.event_type.clone()),
            Self::Result => serde_json::Value::String(event.result.clone()),
            Self::Summary => serde_json::Value::String(event.summary.clone()),
            Self::ChangeTicket => event.change_ticket
                .clone()
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
            Self::IntegrityState => serde_json::Value::String(event.integrity_state.clone()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct QueryRequest {
    schema_version: String,
    filters: Vec<Filter>,
    projection: Vec<ProjectionField>,
    limit: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct QueryBudget {
    max_queries: usize,
    max_rows_examined_total: usize,
    max_rows_returned_total: usize,
    max_bytes_returned_total: usize,
    max_limit_per_query: usize,
    max_filters_per_query: usize,
    max_projection_fields: usize,
}

impl Default for QueryBudget {
    fn default() -> Self {
        Self {
            max_queries: 100,
            max_rows_examined_total: 1_000_000,
            max_rows_returned_total: 10_000,
            max_bytes_returned_total: 8 * 1024 * 1024,
            max_limit_per_query: 200,
            max_filters_per_query: 8,
            max_projection_fields: 12,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct QueryUsage {
    queries: usize,
    rows_examined: usize,
    rows_returned: usize,
    bytes_returned: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct QueryRecord {
    record_id: String,
    fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct QueryReceipt {
    schema_version: String,
    disposition: String,
    query_id: String,
    request_sha256: String,
    solver_manifest_sha256: String,
    event_store_sha256: String,
    rows_examined: usize,
    rows_matched: usize,
    rows_returned: usize,
    row_bytes_returned: usize,
    truncated: bool,
    cumulative_queries: usize,
    cumulative_rows_examined: usize,
    cumulative_rows_returned: usize,
    cumulative_bytes_returned: usize,
    non_claims: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct QueryResponse {
    receipt: QueryReceipt,
    records: Vec<QueryRecord>,
}

struct SolverStore {
    events: Vec<Event>,
    manifest_sha256: String,
    event_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_request(request: &QueryRequest, budget: &QueryBudget) -> Result<(), String> {
    if request.schema_version != QUERY_SCHEMA_VERSION {
        return Err(format!("unsupported query schema version {}", request.schema_version));
    }
    if request.filters.is_empty() || request.filters.len() > budget.max_filters_per_query {
        return Err(format!(
            "query requires 1..={} typed filters",
            budget.max_filters_per_query
        ));
    }
    if request.limit == 0 || request.limit > budget.max_limit_per_query {
        return Err(format!(
            "limit must be within 1..={}",
            budget.max_limit_per_query
        ));
    }
    if request.projection.is_empty()
        || request.projection.len() > budget.max_projection_fields
        || !request.projection.contains(&ProjectionField::RecordId)
    {
        return Err(format!(
            "projection must include record_id and at most {} unique fields",
            budget.max_projection_fields
        ));
    }
    if request.projection.iter().collect::<BTreeSet<_>>().len() != request.projection.len() {
        return Err("projection contains duplicate fields".to_string());
    }
    if request.filters.iter().collect::<BTreeSet<_>>().len() != request.filters.len() {
        return Err("query contains duplicate filters".to_string());
    }
    for filter in &request.filters {
        filter.validate()?;
    }
    Ok(())
}

fn execute_query(
    events: &[Event],
    request: &QueryRequest,
    event_store_sha256: &str,
    solver_manifest_sha256: &str,
    budget: &QueryBudget,
    usage: &mut QueryUsage,
) -> Result<QueryResponse, String> {
    validate_request(request, budget)?;

    if usage.queries >= budget.max_queries {
        return Err("query budget exhausted: max_queries".to_string());
    }
    let next_examined = usage.rows_examined
        .checked_add(events.len())
        .ok_or_else(|| "rows-examined counter overflow".to_string())?;
    if next_examined > budget.max_rows_examined_total {
        return Err("query budget exhausted: max_rows_examined_total".to_string());
    }

    let request_bytes = serde_json::to_vec(request)
        .map_err(|e| format!("cannot serialize normalized query request: {e}"))?;
    let request_sha256 = sha256(&request_bytes);
    let mut matches: Vec<&Event> = events
        .iter()
        .filter(|event| request.filters.iter().all(|filter| filter.matches(event)))
        .collect();
    matches.sort_by(|a, b| a.ingest_time_ms.cmp(&b.ingest_time_ms).then(a.record_id.cmp(&b.record_id)));

    let rows_matched = matches.len();
    let truncated = rows_matched > request.limit;
    matches.truncate(request.limit);

    let records: Vec<QueryRecord> = matches
        .iter()
        .map(|event| {
            let mut fields = BTreeMap::new();
            for field in &request.projection {
                fields.insert(field.key().to_string(), field.value(event));
            }
            QueryRecord {
                record_id: event.record_id.clone(),
                fields,
            }
        })
        .collect();

    let row_bytes_returned = serde_json::to_vec(&records)
        .map_err(|e| format!("cannot serialize projected query rows: {e}"))?
        .len();
    let next_returned = usage.rows_returned
        .checked_add(records.len())
        .ok_or_else(|| "rows-returned counter overflow".to_string())?;
    if next_returned > budget.max_rows_returned_total {
        return Err("query budget exhausted: max_rows_returned_total".to_string());
    }
    let next_bytes = usage.bytes_returned
        .checked_add(row_bytes_returned)
        .ok_or_else(|| "bytes-returned counter overflow".to_string())?;
    if next_bytes > budget.max_bytes_returned_total {
        return Err("query budget exhausted: max_bytes_returned_total".to_string());
    }

    // Commit budget usage only after the entire query and its response have passed checks.
    let next_queries = usage.queries + 1;
    usage.queries = next_queries;
    usage.rows_examined = next_examined;
    usage.rows_returned = next_returned;
    usage.bytes_returned = next_bytes;

    Ok(QueryResponse {
        receipt: QueryReceipt {
            schema_version: QUERY_RECEIPT_VERSION.to_string(),
            disposition: "READ_ONLY_EVIDENCE_RESULTS_NOT_A_SECURITY_CONCLUSION".to_string(),
            query_id: format!("query-{next_queries:06}"),
            request_sha256,
            solver_manifest_sha256: solver_manifest_sha256.to_string(),
            event_store_sha256: event_store_sha256.to_string(),
            rows_examined: events.len(),
            rows_matched,
            rows_returned: records.len(),
            row_bytes_returned,
            truncated,
            cumulative_queries: usage.queries,
            cumulative_rows_examined: usage.rows_examined,
            cumulative_rows_returned: usage.rows_returned,
            cumulative_bytes_returned: usage.bytes_returned,
            non_claims: vec![
                "A query result is an observation projection, not a finding, compromise claim, or authorization.".to_string(),
                "The query receipt binds to the supplied solver-only manifest and event bytes; it does not authenticate the event producer.".to_string(),
                "Source-event identity, delivery identity, event time, ingestion time, integrity state, and security classification remain distinct.".to_string(),
            ],
        },
        records,
    })
}

fn validate_solver_directory_contents(root: &Path) -> Result<(), String> {
    let entries = fs::read_dir(root)
        .map_err(|error| format!("cannot inspect solver-only directory {}: {error}", root.display()))?;
    let mut observed = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read solver directory entry: {error}"))?;
        let file_type = entry.file_type()
            .map_err(|error| format!("cannot inspect solver directory entry: {error}"))?;
        if !file_type.is_file() {
            return Err(format!(
                "solver-only directory contains a non-regular entry: {}",
                entry.path().display()
            ));
        }
        let name = entry.file_name().into_string()
            .map_err(|_| "solver-only directory contains a non-UTF-8 filename".to_string())?;
        if !matches!(name.as_str(), "events.jsonl" | "manifest.json") {
            return Err(format!("solver-only directory contains unexpected file: {name}"));
        }
        if !observed.insert(name.clone()) {
            return Err(format!("solver-only directory has duplicate name: {name}"));
        }
    }
    let expected: BTreeSet<String> = ["events.jsonl".to_string(), "manifest.json".to_string()]
        .into_iter()
        .collect();
    if observed != expected {
        return Err("solver-only directory must contain exactly events.jsonl and manifest.json".to_string());
    }
    Ok(())
}

fn load_solver_store(root: &Path) -> Result<SolverStore, String> {
    validate_solver_directory_contents(root)?;
    let manifest_path = root.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|e| format!("cannot read solver-only manifest {}: {e}", manifest_path.display()))?;
    let manifest: SolverManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("cannot parse solver-only manifest {}: {e}", manifest_path.display()))?;
    if manifest.schema_version != SOLVER_MANIFEST_VERSION
        || manifest.generator_version != GENERATOR_VERSION
        || !manifest.synthetic_only
        || manifest.qualification_state != "UNSCORED"
        || manifest.event_file != "events.jsonl"
        || manifest.record_count < 1
        || manifest.claim_ceiling.trim().is_empty()
    {
        return Err("solver-only manifest version, path, count, state, or claim ceiling is invalid".to_string());
    }
    if manifest.event_sha256.len() != 64
        || !manifest.event_sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("solver-only manifest event SHA-256 is malformed".to_string());
    }

    // Deliberately only open files inside the solver-view directory. This function
    // never resolves a parent-level manifest or evaluator/oracle path.
    let event_path = root.join("events.jsonl");
    let event_bytes = fs::read(&event_path)
        .map_err(|e| format!("cannot read solver-visible event stream {}: {e}", event_path.display()))?;
    if event_bytes.len() != manifest.event_utf8_bytes || sha256(&event_bytes) != manifest.event_sha256 {
        return Err("solver event bytes do not match the isolated solver manifest".to_string());
    }
    let event_text = std::str::from_utf8(&event_bytes)
        .map_err(|e| format!("solver events are not UTF-8: {e}"))?;
    if !event_text.ends_with('\n') {
        return Err("solver event JSONL must end with LF".to_string());
    }
    let mut events = Vec::new();
    let mut record_ids = BTreeSet::new();
    let mut delivery_ids = BTreeSet::new();
    for (index, line) in event_text.split_terminator('\n').enumerate() {
        if line.trim().is_empty() {
            return Err(format!("solver event JSONL has blank record at line {}", index + 1));
        }
        let event: Event = serde_json::from_str(line)
            .map_err(|e| format!("solver event JSONL record {} is malformed: {e}", index + 1))?;
        if event.schema_version != EVENT_SCHEMA_VERSION
            || !record_ids.insert(event.record_id.clone())
            || !delivery_ids.insert(event.delivery_id.clone())
        {
            return Err(format!("solver event record {} has invalid version or duplicate identity", event.record_id));
        }
        events.push(event);
    }
    if events.len() != manifest.record_count {
        return Err(format!(
            "solver event count mismatch: manifest {}, actual {}",
            manifest.record_count,
            events.len()
        ));
    }
    Ok(SolverStore {
        events,
        manifest_sha256: sha256(&manifest_bytes),
        event_sha256: sha256(&event_bytes),
    })
}

fn run_cli(args: &[String]) -> Result<(), String> {
    let mut store_dir = None;
    let mut requests_file = None;
    let mut args_iter = args.iter();
    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "--store-dir" => store_dir = Some(PathBuf::from(args_iter.next().ok_or_else(|| "--store-dir requires a solver-only directory".to_string())?)),
            "--requests-jsonl" => requests_file = Some(PathBuf::from(args_iter.next().ok_or_else(|| "--requests-jsonl requires a JSONL file".to_string())?)),
            _ => return Err(format!("unknown query argument {arg}")),
        }
    }
    let store_dir = store_dir.ok_or_else(|| "query mode requires --store-dir".to_string())?;
    let requests_file = requests_file.ok_or_else(|| "query mode requires --requests-jsonl".to_string())?;
    let requests_bytes = fs::read(&requests_file)
        .map_err(|e| format!("cannot read query batch {}: {e}", requests_file.display()))?;
    if requests_bytes.len() > MAX_BATCH_BYTES {
        return Err(format!("query batch exceeds {MAX_BATCH_BYTES} bytes"));
    }
    let requests_text = std::str::from_utf8(&requests_bytes)
        .map_err(|e| format!("query batch is not UTF-8: {e}"))?;
    if !requests_text.is_empty() && !requests_text.ends_with('\n') {
        return Err("query JSONL batch must end with LF".to_string());
    }
    let mut requests = Vec::<QueryRequest>::new();
    for (index, line) in requests_text.split_terminator('\n').enumerate() {
        if line.len() > MAX_REQUEST_BYTES {
            return Err(format!("query request at line {} exceeds {MAX_REQUEST_BYTES} bytes", index + 1));
        }
        if line.trim().is_empty() {
            return Err(format!("query batch has a blank request at line {}", index + 1));
        }
        let request: QueryRequest = serde_json::from_str(line)
            .map_err(|e| format!("query request at line {} is invalid: {e}", index + 1))?;
        requests.push(request);
        if requests.len() > MAX_BATCH_REQUESTS {
            return Err(format!("query batch exceeds {MAX_BATCH_REQUESTS} requests"));
        }
    }
    if requests.is_empty() {
        return Err("query batch must contain at least one request".to_string());
    }

    let store = load_solver_store(&store_dir)?;
    let budget = QueryBudget::default();
    let mut usage = QueryUsage::default();
    for (index, request) in requests.iter().enumerate() {
        let result = execute_query(
            &store.events,
            request,
            &store.event_sha256,
            &store.manifest_sha256,
            &budget,
            &mut usage,
        );
        #[derive(Serialize)]
        struct BatchLine<'a> {
            line_number: usize,
            ok: bool,
            response: Option<&'a QueryResponse>,
            error: Option<String>,
        }
        match result {
            Ok(response) => println!(
                "{}",
                serde_json::to_string(&BatchLine { line_number: index + 1, ok: true, response: Some(&response), error: None })
                    .map_err(|e| format!("cannot serialize query response: {e}"))?
            ),
            Err(error) => println!(
                "{}",
                serde_json::to_string(&BatchLine { line_number: index + 1, ok: false, response: None, error: Some(error) })
                    .map_err(|e| format!("cannot serialize query error: {e}"))?
            ),
        }
    }
    Ok(())
}

pub fn run_cli_or_exit(args: &[String]) {
    if let Err(error) = run_cli(args) {
        eprintln!("cyber-hunt-fixture query: FAIL: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_dataset;

    fn default_request() -> QueryRequest {
        QueryRequest {
            schema_version: QUERY_SCHEMA_VERSION.to_string(),
            filters: vec![Filter::TextEquals {
                field: TextField::EventType,
                value: "network.transfer".to_string(),
            }],
            projection: vec![
                ProjectionField::RecordId,
                ProjectionField::EventType,
                ProjectionField::Destination,
                ProjectionField::BytesSent,
                ProjectionField::DevicePosture,
                ProjectionField::SourceAddress,
                ProjectionField::ChangeTicket,
            ],
            limit: 25,
        }
    }

    #[test]
    fn typed_query_returns_bounded_evidence_and_receipt() {
        let dataset = build_dataset(1_000, 42).expect("dataset should build");
        let mut usage = QueryUsage::default();
        let response = execute_query(
            &dataset.events,
            &default_request(),
            &"a".repeat(64),
            &"b".repeat(64),
            &QueryBudget::default(),
            &mut usage,
        ).expect("typed query should succeed");
        assert_eq!(response.receipt.disposition, "READ_ONLY_EVIDENCE_RESULTS_NOT_A_SECURITY_CONCLUSION");
        assert_eq!(response.receipt.rows_examined, 1_000);
        assert!(!response.receipt.source_commit_pinned);
        assert!(response.records.len() <= 25);
        assert_eq!(usage.queries, 1);
        assert!(response.records.iter().all(|row| row.fields.contains_key("event_type")));
    }

    #[test]
    fn repeated_queries_share_the_same_session_budget() {
        let dataset = build_dataset(1_000, 42).expect("dataset should build");
        let mut usage = QueryUsage::default();
        let budget = QueryBudget { max_queries: 1, ..QueryBudget::default() };
        execute_query(
            &dataset.events,
            &default_request(),
            &"a".repeat(64),
            &"b".repeat(64),
            &budget,
            &mut usage,
        ).expect("first query should be within budget");
        assert_eq!(usage.queries, 1);
        let error = execute_query(
            &dataset.events,
            &default_request(),
            &"a".repeat(64),
            &"b".repeat(64),
            "test-source-commit",
            &budget,
            &mut usage,
        ).expect_err("second query should exceed the same session budget");
        assert!(error.contains("max_queries"));
        assert_eq!(usage.queries, 1);
    }

    #[test]
    fn high_volume_attack_and_approved_change_are_both_returned_for_contextual_review() {
        let dataset = build_dataset(10_000, 42).expect("dataset should build");
        let mut request = default_request();
        request.filters.push(Filter::BytesSentAtLeast { minimum: 9_000_000 });
        let mut usage = QueryUsage::default();
        let response = execute_query(
            &dataset.events,
            &request,
            &"a".repeat(64),
            &"b".repeat(64),
            "test-source-commit",
            &QueryBudget::default(),
            &mut usage,
        ).expect("bounded contextual query should pass");
        assert_eq!(response.receipt.rows_matched, 4);
        assert_eq!(response.receipt.rows_returned, 4);
        let ticket_backed = response.records.iter().filter(|row| {
            row.fields.get("change_ticket").is_some_and(|value| !value.is_null())
        }).count();
        assert_eq!(ticket_backed, 2);
        assert!(response.records.iter().all(|row| !row.fields.contains_key("security_label")));
        assert_eq!(response.receipt.disposition, "READ_ONLY_EVIDENCE_RESULTS_NOT_A_SECURITY_CONCLUSION");
    }

    #[test]
    fn query_schema_declares_the_serialized_request_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/query-v1.schema.json"))
            .expect("query schema should be JSON");
        let request = serde_json::to_value(default_request()).expect("request should serialize");
        let actual: BTreeSet<String> = request.as_object().unwrap().keys().cloned().collect();
        let required: BTreeSet<String> = schema["required"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_string()).collect();
        assert_eq!(actual, required);
        assert_eq!(schema["properties"]["schema_version"]["const"], QUERY_SCHEMA_VERSION);
        assert_eq!(schema["properties"]["filters"]["items"]["oneOf"].as_array().unwrap().len(), 3);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn query_budget_failure_does_not_consume_session_budget() {
        let dataset = build_dataset(1_000, 42).expect("dataset should build");
        let mut usage = QueryUsage::default();
        let budget = QueryBudget { max_rows_examined_total: 999, ..QueryBudget::default() };
        let error = execute_query(
            &dataset.events,
            &default_request(),
            &"a".repeat(64),
            &"b".repeat(64),
            "test-source-commit",
            &budget,
            &mut usage,
        ).expect_err("scan larger than budget must be rejected");
        assert!(error.contains("max_rows_examined_total"));
        assert_eq!(usage, QueryUsage::default());
    }

    #[test]
    fn invalid_projection_and_unbounded_query_fail_closed() {
        let dataset = build_dataset(1_000, 42).expect("dataset should build");
        let mut usage = QueryUsage::default();
        let mut request = default_request();
        request.projection = vec![ProjectionField::EventType];
        assert!(execute_query(&dataset.events, &request, "a", "b", "test", &QueryBudget::default(), &mut usage).is_err());
        request = default_request();
        request.filters.clear();
        assert!(execute_query(&dataset.events, &request, "a", "b", "test", &QueryBudget::default(), &mut usage).is_err());
        assert_eq!(usage, QueryUsage::default());
    }

    #[test]
    fn query_response_and_receipt_match_their_versioned_schemas() {
        let dataset = build_dataset(1_000, 42).expect("dataset should build");
        let mut usage = QueryUsage::default();
        let response = execute_query(
            &dataset.events,
            &default_request(),
            &"a".repeat(64),
            &"b".repeat(64),
            "test-source-commit",
            &QueryBudget::default(),
            &mut usage,
        ).expect("query should succeed");
        let response_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/query-response-v1.schema.json"))
                .expect("query response schema should be valid JSON");
        let receipt_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/query-receipt-v1.schema.json"))
                .expect("query receipt schema should be valid JSON");
        let response_value = serde_json::to_value(&response).expect("response should serialize");
        let receipt_value = serde_json::to_value(&response.receipt).expect("receipt should serialize");
        let actual_response: BTreeSet<String> = response_value.as_object().unwrap().keys().cloned().collect();
        let required_response: BTreeSet<String> = response_schema["required"].as_array().unwrap()
            .iter().map(|key| key.as_str().unwrap().to_string()).collect();
        assert_eq!(actual_response, required_response);
        let actual_receipt: BTreeSet<String> = receipt_value.as_object().unwrap().keys().cloned().collect();
        let required_receipt: BTreeSet<String> = receipt_schema["required"].as_array().unwrap()
            .iter().map(|key| key.as_str().unwrap().to_string()).collect();
        assert_eq!(actual_receipt, required_receipt);
    }

    #[test]
    fn serialized_query_cannot_request_unknown_or_mutating_operations() {
        let payload = br#"{"schema_version":"cyber-hunt-query-v1","filters":[{"kind":"text_equals","field":"event_type","value":"network.transfer","sql":"DELETE FROM events"}],"projection":["record_id"],"limit":5,"shell":"true"}"#;
        assert!(serde_json::from_slice::<QueryRequest>(payload).is_err());
    }

    #[test]
    fn solver_store_loader_never_resolves_parent_oracle_path() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let work_root = env::temp_dir().join(format!("solver-view-query-test-{}-{nonce}", process::id()));
        let root = work_root.join("solver");
        fs::create_dir_all(&root).expect("solver directory should be created");
        let evaluator_dir = work_root.join("evaluator");
        fs::create_dir_all(&evaluator_dir).expect("evaluator directory should be created");
        fs::write(evaluator_dir.join("oracle.json"), b"deliberately-not-valid-json")
            .expect("evaluator-only answer key fixture should be created");
        let dataset = build_dataset(1_000, 42).expect("dataset should build");
        let event_bytes = crate::render_jsonl(&dataset.events).expect("events should serialize");
        let manifest = SolverManifest {
            schema_version: SOLVER_MANIFEST_VERSION.to_string(),
            generator_version: GENERATOR_VERSION.to_string(),
            source_commit: "test-source-commit".to_string(),
            source_commit_pinned: false,
            synthetic_only: true,
            seed: 42,
            record_count: 1_000,
            event_file: "events.jsonl".to_string(),
            event_sha256: sha256(&event_bytes),
            event_utf8_bytes: event_bytes.len(),
            qualification_state: "UNSCORED".to_string(),
            claim_ceiling: "test fixture".to_string(),
        };
        fs::write(root.join("events.jsonl"), &event_bytes).expect("event file should be written");
        fs::write(root.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).expect("manifest should be written");
        let store = load_solver_store(&root).expect("solver-only directory should load");
        assert_eq!(store.events.len(), 1_000);
        assert!(!root.join("oracle.json").exists());

        fs::write(root.join("oracle.json"), b"unexpected-answer-key")
            .expect("unexpected in-view oracle should be written");
        let error = load_solver_store(&root)
            .expect_err("solver loader must fail closed when an oracle is accidentally mounted");
        assert!(error.contains("unexpected file"));
        fs::remove_dir_all(work_root).expect("temporary solver/evaluator tree should be removed");
    }
}
