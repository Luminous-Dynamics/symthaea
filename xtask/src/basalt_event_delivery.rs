//! Audits per-source event delivery accounting, including tail loss at stream close.
//!
//! This tool reconciles a source-epoch ledger against serialized events and explicit
//! loss/discard ranges. It does not capture events or authenticate the ledger author.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SOURCE_EPOCHS: usize = 4_096;
const MAX_EVENTS: usize = 100_000;
const MAX_RANGES: usize = 100_000;

#[derive(Debug, Default, Serialize)]
pub struct DeliveryAuditCounts {
    pub source_epochs: usize,
    pub events: usize,
    pub attempted_sequences: u64,
    pub published_sequences: u64,
    pub lost_sequences: u64,
    pub required_discards: u64,
    pub excluded_discards: u64,
    pub unclassified_sequences: u64,
    pub overlapping_sequences: u64,
    pub counter_mismatches: usize,
    pub unclosed_sources: usize,
    pub restarted_sources: usize,
    pub unmonitored_producers: u64,
}

#[derive(Debug, Serialize)]
pub struct DeliveryAuditError {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct DeliveryAuditReport {
    pub report_version: u32,
    pub ledger_digest: String,
    pub run_id: Option<String>,
    pub ledger_well_formed: bool,
    pub sequence_accounting_reconciled: bool,
    pub counters_consistent: bool,
    pub sources_closed: bool,
    pub loss_free: bool,
    pub audit_passed: bool,
    pub qualification_eligible: bool,
    pub counts: DeliveryAuditCounts,
    pub errors: Vec<DeliveryAuditError>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeliveryLedger {
    format_version: u32,
    suite_id: String,
    run_id: String,
    observer_binary_digest: String,
    observer_configuration_digest: String,
    capture_start_monotonic_ns: u64,
    capture_end_monotonic_ns: u64,
    source_epochs: Vec<SourceEpoch>,
    events: Vec<DeliveryEvent>,
    classified_ranges: Vec<ClassifiedRange>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEpoch {
    source_id: String,
    epoch_id: String,
    partition_id: String,
    first_attempted_sequence: u64,
    last_attempted_sequence: u64,
    reported_published_count: u64,
    reported_lost_count: u64,
    reported_discarded_required_count: u64,
    reported_discarded_excluded_count: u64,
    restart_count: u64,
    unmonitored_producer_count: u64,
    producer_quiesced: bool,
    consumer_drained: bool,
    close_fence_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeliveryEvent {
    source_id: String,
    epoch_id: String,
    partition_id: String,
    sequence: u64,
    observed_at_monotonic_ns: u64,
    event_digest: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RangeDisposition {
    Lost,
    DiscardedRequired,
    DiscardedExcluded,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RangeReason {
    RingbufReservationFailureUnknown,
    PerfBufferLostSample,
    ConsumerQueueOverrun,
    ExporterDrop,
    DecoderRejection,
    ObserverRestart,
    IntentionalPolicySuppression,
    Unknown,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassifiedRange {
    source_id: String,
    epoch_id: String,
    partition_id: String,
    first_sequence: u64,
    last_sequence: u64,
    disposition: RangeDisposition,
    reason: RangeReason,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SourceEpochKey {
    source_id: String,
    epoch_id: String,
    partition_id: String,
}

impl SourceEpoch {
    fn key(&self) -> SourceEpochKey {
        SourceEpochKey {
            source_id: self.source_id.clone(),
            epoch_id: self.epoch_id.clone(),
            partition_id: self.partition_id.clone(),
        }
    }
}

impl DeliveryEvent {
    fn key(&self) -> SourceEpochKey {
        SourceEpochKey {
            source_id: self.source_id.clone(),
            epoch_id: self.epoch_id.clone(),
            partition_id: self.partition_id.clone(),
        }
    }
}

impl ClassifiedRange {
    fn key(&self) -> SourceEpochKey {
        SourceEpochKey {
            source_id: self.source_id.clone(),
            epoch_id: self.epoch_id.clone(),
            partition_id: self.partition_id.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct Span {
    first: u64,
    last: u64,
}

pub fn audit_file(path: &Path) -> anyhow::Result<DeliveryAuditReport> {
    let bytes = read_bounded(path)?;
    Ok(audit_bytes(&bytes))
}

pub fn audit_bytes(bytes: &[u8]) -> DeliveryAuditReport {
    let mut report = DeliveryAuditReport {
        report_version: 1,
        ledger_digest: sha256_tagged(bytes),
        run_id: None,
        ledger_well_formed: false,
        sequence_accounting_reconciled: false,
        counters_consistent: false,
        sources_closed: false,
        loss_free: false,
        audit_passed: false,
        qualification_eligible: false,
        counts: DeliveryAuditCounts::default(),
        errors: Vec::new(),
    };

    if bytes.len() as u64 > MAX_INPUT_BYTES {
        push_error(&mut report, "E_INPUT_TOO_LARGE", "$", "ledger exceeds the 16 MiB input limit");
        return report;
    }

    let ledger: DeliveryLedger = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(parse_error) => {
            push_error(&mut report, "E_LEDGER_JSON", "$", &format!("invalid or non-strict delivery ledger: {parse_error}"));
            return report;
        }
    };

    report.run_id = Some(ledger.run_id.clone());
    report.counts.source_epochs = ledger.source_epochs.len();
    report.counts.events = ledger.events.len();

    let structural_error_start = report.errors.len();
    if ledger.format_version != 1 || ledger.suite_id != "basalt-event-delivery-ledger-v1" {
        push_error(&mut report, "E_LEDGER_VERSION", "$", "unsupported ledger version or suite_id");
    }
    if ledger.run_id.is_empty() || ledger.run_id.len() > 128 {
        push_error(&mut report, "E_LEDGER_RUN_ID", "$.run_id", "run_id must be non-empty and at most 128 bytes");
    }
    if !valid_digest(&ledger.observer_binary_digest) || !valid_digest(&ledger.observer_configuration_digest) {
        push_error(&mut report, "E_LEDGER_DIGEST", "$", "observer binary and configuration digests must be lowercase SHA-256");
    }
    if ledger.capture_start_monotonic_ns >= ledger.capture_end_monotonic_ns {
        push_error(&mut report, "E_LEDGER_CAPTURE_INTERVAL", "$", "capture interval must be strictly increasing");
    }
    if ledger.source_epochs.is_empty() || ledger.source_epochs.len() > MAX_SOURCE_EPOCHS {
        push_error(&mut report, "E_SOURCE_EPOCH_COUNT", "$.source_epochs", "source_epochs must contain 1..=4096 entries");
    }
    if ledger.events.len() > MAX_EVENTS {
        push_error(&mut report, "E_EVENT_COUNT", "$.events", "events exceeds the 100,000-record limit");
    }
    if ledger.classified_ranges.len() > MAX_RANGES {
        push_error(&mut report, "E_RANGE_COUNT", "$.classified_ranges", "classified_ranges exceeds the 100,000-range limit");
    }

    let mut sources = BTreeMap::<SourceEpochKey, &SourceEpoch>::new();
    for (index, source) in ledger.source_epochs.iter().enumerate() {
        if source.source_id.is_empty() || source.source_id.len() > 128
            || source.epoch_id.is_empty() || source.epoch_id.len() > 128
            || source.partition_id.is_empty() || source.partition_id.len() > 128
        {
            push_error(&mut report, "E_SOURCE_IDENTITY", &format!("source_epochs[{index}]"), "source_id, epoch_id and partition_id must be non-empty and at most 128 bytes");
        }
        if source.first_attempted_sequence == 0 || source.last_attempted_sequence < source.first_attempted_sequence {
            push_error(&mut report, "E_SOURCE_SEQUENCE_RANGE", &format!("source_epochs[{index}]"), "attempted sequence range must be positive and ordered");
        }
        if !valid_digest(&source.close_fence_digest) {
            push_error(&mut report, "E_SOURCE_CLOSE_FENCE", &format!("source_epochs[{index}]"), "close_fence_digest must be lowercase SHA-256");
        }
        if sources.insert(source.key(), source).is_some() {
            push_error(&mut report, "E_SOURCE_DUPLICATE", &format!("source_epochs[{index}]"), "source_id/epoch_id/partition_id tuple must be unique");
        }
    }

    let mut events_by_source = BTreeMap::<SourceEpochKey, Vec<&DeliveryEvent>>::new();
    for (index, event) in ledger.events.iter().enumerate() {
        if !valid_digest(&event.event_digest) {
            push_error(&mut report, "E_EVENT_DIGEST", &format!("events[{index}]"), "event_digest must be lowercase SHA-256");
        }
        if event.observed_at_monotonic_ns < ledger.capture_start_monotonic_ns
            || event.observed_at_monotonic_ns > ledger.capture_end_monotonic_ns
        {
            push_error(&mut report, "E_EVENT_TIME", &format!("events[{index}]"), "event timestamp lies outside the ledger capture interval");
        }
        let key = event.key();
        if !sources.contains_key(&key) {
            push_error(&mut report, "E_EVENT_SOURCE_UNKNOWN", &format!("events[{index}]"), "event references an unknown source epoch");
        }
        events_by_source.entry(key).or_default().push(event);
    }

    let mut ranges_by_source = BTreeMap::<SourceEpochKey, Vec<&ClassifiedRange>>::new();
    for (index, range) in ledger.classified_ranges.iter().enumerate() {
        if range.first_sequence == 0 || range.last_sequence < range.first_sequence {
            push_error(&mut report, "E_RANGE_ORDER", &format!("classified_ranges[{index}]"), "range sequence bounds must be positive and ordered");
        }
        match range.disposition {
            RangeDisposition::DiscardedExcluded if range.reason != RangeReason::IntentionalPolicySuppression => {
                push_error(&mut report, "E_RANGE_EXCLUSION_REASON", &format!("classified_ranges[{index}]"), "discarded_excluded requires intentional_policy_suppression");
            }
            RangeDisposition::Lost if range.reason == RangeReason::IntentionalPolicySuppression => {
                push_error(&mut report, "E_RANGE_LOSS_REASON", &format!("classified_ranges[{index}]"), "intentional_policy_suppression must use a discard disposition");
            }
            _ => {}
        }
        let key = range.key();
        if !sources.contains_key(&key) {
            push_error(&mut report, "E_RANGE_SOURCE_UNKNOWN", &format!("classified_ranges[{index}]"), "range references an unknown source epoch");
        }
        ranges_by_source.entry(key).or_default().push(range);
    }

    let mut sequence_ok = true;
    let mut counters_ok = true;
    let mut sources_closed = true;
    let mut loss_free = true;

    for (key, source) in &sources {
        let events = events_by_source.get(key).cloned().unwrap_or_default();
        let ranges = ranges_by_source.get(key).cloned().unwrap_or_default();

        let mut spans = Vec::<Span>::new();
        let mut event_sequences = BTreeSet::new();
        for event in &events {
            if event.sequence < source.first_attempted_sequence || event.sequence > source.last_attempted_sequence {
                push_error(&mut report, "E_EVENT_OUT_OF_RANGE", "$.events", "published event sequence is outside its source epoch range");
                sequence_ok = false;
                continue;
            }
            if !event_sequences.insert(event.sequence) {
                push_error(&mut report, "E_EVENT_DUPLICATE_SEQUENCE", "$.events", "duplicate published sequence within one source epoch");
                sequence_ok = false;
            }
            spans.push(Span { first: event.sequence, last: event.sequence });
        }

        let mut lost_count = 0u64;
        let mut required_discard_count = 0u64;
        let mut excluded_discard_count = 0u64;
        for range in &ranges {
            if range.first_sequence < source.first_attempted_sequence || range.last_sequence > source.last_attempted_sequence {
                push_error(&mut report, "E_RANGE_OUT_OF_RANGE", "$.classified_ranges", "classified range lies outside its source epoch sequence bounds");
                sequence_ok = false;
                continue;
            }
            let Some(length) = range.last_sequence.checked_sub(range.first_sequence).and_then(|n| n.checked_add(1)) else {
                push_error(&mut report, "E_RANGE_LENGTH_OVERFLOW", "$.classified_ranges", "classified range length overflows u64");
                sequence_ok = false;
                continue;
            };
            match range.disposition {
                RangeDisposition::Lost => lost_count = lost_count.saturating_add(length),
                RangeDisposition::DiscardedRequired => required_discard_count = required_discard_count.saturating_add(length),
                RangeDisposition::DiscardedExcluded => excluded_discard_count = excluded_discard_count.saturating_add(length),
            }
            spans.push(Span { first: range.first_sequence, last: range.last_sequence });
        }

        let attempted_count = source.last_attempted_sequence
            .checked_sub(source.first_attempted_sequence)
            .and_then(|n| n.checked_add(1));
        if let Some(attempted) = attempted_count {
            report.counts.attempted_sequences = report.counts.attempted_sequences.saturating_add(attempted);
        } else {
            push_error(&mut report, "E_ATTEMPTED_COUNT_OVERFLOW", "$.source_epochs", "attempted sequence count overflows u64");
            sequence_ok = false;
        }

        report.counts.published_sequences = report.counts.published_sequences.saturating_add(events.len() as u64);
        report.counts.lost_sequences = report.counts.lost_sequences.saturating_add(lost_count);
        report.counts.required_discards = report.counts.required_discards.saturating_add(required_discard_count);
        report.counts.excluded_discards = report.counts.excluded_discards.saturating_add(excluded_discard_count);

        spans.sort_by_key(|span| (span.first, span.last));
        let mut expected = source.first_attempted_sequence;
        let mut completed = false;
        for span in spans {
            if span.first < expected {
                push_error(&mut report, "E_SEQUENCE_OVERLAP", "$.source_epochs", "published event or classified range overlaps an earlier sequence span");
                report.counts.overlapping_sequences = report.counts.overlapping_sequences.saturating_add(1);
                sequence_ok = false;
                continue;
            }
            if span.first > expected {
                let gap = span.first - expected;
                push_error(&mut report, "E_SEQUENCE_UNCLASSIFIED_GAP", "$.source_epochs", "one or more attempted sequences are neither published nor classified as lost/discarded");
                report.counts.unclassified_sequences = report.counts.unclassified_sequences.saturating_add(gap);
                sequence_ok = false;
            }
            if span.last < span.first || span.last > source.last_attempted_sequence {
                sequence_ok = false;
                continue;
            }
            if span.last == source.last_attempted_sequence {
                completed = true;
                break;
            }
            match span.last.checked_add(1) {
                Some(next) => expected = next,
                None => {
                    sequence_ok = false;
                    push_error(&mut report, "E_SEQUENCE_OVERFLOW", "$.source_epochs", "sequence high-water mark overflows");
                    break;
                }
            }
        }
        if !completed {
            let tail_start = expected;
            let tail_gap = source.last_attempted_sequence.saturating_sub(tail_start).saturating_add(1);
            if tail_gap > 0 {
                push_error(&mut report, "E_SEQUENCE_TAIL_UNACCOUNTED", "$.source_epochs", "final attempted sequences are missing or unclassified; a terminal high-water mark must account for tail loss");
                report.counts.unclassified_sequences = report.counts.unclassified_sequences.saturating_add(tail_gap);
                sequence_ok = false;
            }
        }

        if source.reported_published_count != events.len() as u64
            || source.reported_lost_count != lost_count
            || source.reported_discarded_required_count != required_discard_count
            || source.reported_discarded_excluded_count != excluded_discard_count
        {
            push_error(&mut report, "E_SOURCE_COUNTER_MISMATCH", "$.source_epochs", "reported published/lost/discard counters do not match independently derived event and range counts");
            report.counts.counter_mismatches += 1;
            counters_ok = false;
        }

        if !source.producer_quiesced || !source.consumer_drained || !valid_digest(&source.close_fence_digest) {
            push_error(&mut report, "E_SOURCE_NOT_CLOSED", "$.source_epochs", "source epoch requires producer quiescence, a close fence and consumer-drain acknowledgement");
            report.counts.unclosed_sources += 1;
            sources_closed = false;
        }
        if source.restart_count > 0 {
            report.counts.restarted_sources += 1;
            push_error(&mut report, "E_SOURCE_RESTART", "$.source_epochs", "source restart invalidates continuous event delivery for this epoch");
            sources_closed = false;
            loss_free = false;
        }
        if source.unmonitored_producer_count > 0 {
            report.counts.unmonitored_producers = report.counts.unmonitored_producers.saturating_add(source.unmonitored_producer_count);
            push_error(&mut report, "E_UNMONITORED_PRODUCER", "$.source_epochs", "unmonitored producers prevent a loss-free delivery claim");
            sources_closed = false;
            loss_free = false;
        }
        if lost_count > 0 || required_discard_count > 0 {
            loss_free = false;
        }
    }

    for key in events_by_source.keys() {
        if !sources.contains_key(key) {
            sequence_ok = false;
        }
    }
    for key in ranges_by_source.keys() {
        if !sources.contains_key(key) {
            sequence_ok = false;
        }
    }

    report.ledger_well_formed = !report.errors.iter().any(|e| {
        matches!(e.code.as_str(),
            "E_LEDGER_VERSION" | "E_LEDGER_RUN_ID" | "E_LEDGER_DIGEST" |
            "E_LEDGER_CAPTURE_INTERVAL" | "E_SOURCE_EPOCH_COUNT" | "E_EVENT_COUNT" |
            "E_RANGE_COUNT" | "E_SOURCE_IDENTITY" | "E_SOURCE_SEQUENCE_RANGE" |
            "E_SOURCE_CLOSE_FENCE" | "E_SOURCE_DUPLICATE" | "E_EVENT_DIGEST" |
            "E_EVENT_TIME" | "E_EVENT_SOURCE_UNKNOWN" | "E_RANGE_ORDER" |
            "E_RANGE_EXCLUSION_REASON" | "E_RANGE_LOSS_REASON" | "E_RANGE_SOURCE_UNKNOWN" |
            "E_EVENT_OUT_OF_RANGE" | "E_RANGE_OUT_OF_RANGE" | "E_RANGE_LENGTH_OVERFLOW" |
            "E_ATTEMPTED_COUNT_OVERFLOW"
        )
    });
    report.sequence_accounting_reconciled = sequence_ok;
    report.counters_consistent = counters_ok;
    report.sources_closed = sources_closed;
    report.loss_free = loss_free && report.counts.unclassified_sequences == 0 && report.counts.overlapping_sequences == 0;
    report.audit_passed = report.ledger_well_formed
        && report.sequence_accounting_reconciled
        && report.counters_consistent
        && report.sources_closed
        && report.loss_free;
    report
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("sha256:{encoded}")
}

fn read_bounded(path: &Path) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        anyhow::bail!("{} exceeds the 16 MiB input limit", path.display());
    }
    Ok(bytes)
}

fn push_error(report: &mut DeliveryAuditReport, code: &str, path: &str, message: &str) {
    report.errors.push(DeliveryAuditError {
        code: code.to_string(),
        path: path.to_string(),
        message: message.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn digest(seed: u8) -> String {
        format!("sha256:{}", format!("{:064x}", seed))
    }

    fn fixture() -> Value {
        json!({
            "format_version":1,
            "suite_id":"basalt-event-delivery-ledger-v1",
            "run_id":"run-1",
            "observer_binary_digest":digest(1),
            "observer_configuration_digest":digest(2),
            "capture_start_monotonic_ns":100,
            "capture_end_monotonic_ns":1000,
            "source_epochs":[{
                "source_id":"bpf:process-exit",
                "epoch_id":"attach-1",
                "partition_id":"cpu:0",
                "first_attempted_sequence":1,
                "last_attempted_sequence":3,
                "reported_published_count":3,
                "reported_lost_count":0,
                "reported_discarded_required_count":0,
                "reported_discarded_excluded_count":0,
                "restart_count":0,
                "unmonitored_producer_count":0,
                "producer_quiesced":true,
                "consumer_drained":true,
                "close_fence_digest":digest(3)
            }],
            "events":[
                {"source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0","sequence":1,"observed_at_monotonic_ns":200,"event_digest":digest(4)},
                {"source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0","sequence":2,"observed_at_monotonic_ns":201,"event_digest":digest(5)},
                {"source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0","sequence":3,"observed_at_monotonic_ns":202,"event_digest":digest(6)}
            ],
            "classified_ranges":[]
        })
    }

    fn audit(value: &Value) -> DeliveryAuditReport {
        audit_bytes(value.to_string().as_bytes())
    }

    fn add_lost_tail(value: &mut Value) {
        value["events"].as_array_mut().unwrap().pop();
        value["source_epochs"][0]["reported_published_count"] = json!(2);
        value["source_epochs"][0]["reported_lost_count"] = json!(1);
        value["classified_ranges"] = json!([{
            "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":3,"last_sequence":3,
            "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
        }]);
    }

    #[test]
    fn clean_closed_epoch_is_audited() {
        let report = audit(&fixture());
        assert!(report.audit_passed, "{:?}", report.errors);
        assert!(report.sequence_accounting_reconciled);
        assert!(report.counters_consistent);
        assert!(report.sources_closed);
        assert!(report.loss_free);
        assert!(!report.qualification_eligible);
    }

    #[test]
    fn final_event_loss_is_detected_by_terminal_high_water_mark() {
        let mut input = fixture();
        add_lost_tail(&mut input);
        let report = audit(&input);
        assert!(report.sequence_accounting_reconciled, "{:?}", report.errors);
        assert!(!report.loss_free);
        assert_eq!(report.counts.lost_sequences, 1);
        assert!(!report.audit_passed);
    }

    #[test]
    fn unclassified_final_sequence_is_not_hidden_by_no_later_event() {
        let mut input = fixture();
        input["events"].as_array_mut().unwrap().pop();
        input["source_epochs"][0]["reported_published_count"] = json!(2);
        let report = audit(&input);
        assert!(!report.sequence_accounting_reconciled);
        assert!(report.errors.iter().any(|e| e.code == "E_SEQUENCE_TAIL_UNACCOUNTED"));
    }

    #[test]
    fn unclassified_middle_gap_is_rejected() {
        let mut input = fixture();
        input["events"].as_array_mut().unwrap().remove(1);
        input["source_epochs"][0]["reported_published_count"] = json!(2);
        let report = audit(&input);
        assert!(!report.sequence_accounting_reconciled);
        assert!(report.errors.iter().any(|e| e.code == "E_SEQUENCE_UNCLASSIFIED_GAP"));
    }

    #[test]
    fn overlapping_loss_range_and_published_event_are_rejected() {
        let mut input = fixture();
        input["classified_ranges"] = json!([{
            "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":2,"last_sequence":2,
            "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
        }]);
        input["source_epochs"][0]["reported_lost_count"] = json!(1);
        let report = audit(&input);
        assert!(!report.sequence_accounting_reconciled);
        assert!(report.errors.iter().any(|e| e.code == "E_SEQUENCE_OVERLAP"));
    }

    #[test]
    fn reported_counters_must_match_derived_counts() {
        let mut input = fixture();
        input["source_epochs"][0]["reported_published_count"] = json!(2);
        let report = audit(&input);
        assert!(!report.counters_consistent);
        assert_eq!(report.counts.counter_mismatches, 1);
    }

    #[test]
    fn required_discard_is_accounted_but_not_loss_free() {
        let mut input = fixture();
        input["events"].as_array_mut().unwrap().pop();
        input["source_epochs"][0]["reported_published_count"] = json!(2);
        input["source_epochs"][0]["reported_discarded_required_count"] = json!(1);
        input["classified_ranges"] = json!([{
            "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":3,"last_sequence":3,
            "disposition":"discarded_required","reason":"intentional_policy_suppression"
        }]);
        let report = audit(&input);
        assert!(report.sequence_accounting_reconciled);
        assert!(!report.loss_free);
        assert!(!report.audit_passed);
    }

    #[test]
    fn explicitly_excluded_policy_suppression_is_visible_and_separate() {
        let mut input = fixture();
        input["events"].as_array_mut().unwrap().pop();
        input["source_epochs"][0]["reported_published_count"] = json!(2);
        input["source_epochs"][0]["reported_discarded_excluded_count"] = json!(1);
        input["classified_ranges"] = json!([{
            "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":3,"last_sequence":3,
            "disposition":"discarded_excluded","reason":"intentional_policy_suppression"
        }]);
        let report = audit(&input);
        assert!(report.sequence_accounting_reconciled);
        assert!(report.loss_free);
        assert!(report.audit_passed, "{:?}", report.errors);
        assert_eq!(report.counts.excluded_discards, 1);
    }

    #[test]
    fn unclosed_consumer_drain_prevents_loss_free_claim() {
        let mut input = fixture();
        input["source_epochs"][0]["consumer_drained"] = json!(false);
        let report = audit(&input);
        assert!(!report.sources_closed);
        assert!(!report.audit_passed);
    }

    #[test]
    fn observer_restart_prevents_continuous_epoch_claim() {
        let mut input = fixture();
        input["source_epochs"][0]["restart_count"] = json!(1);
        let report = audit(&input);
        assert!(!report.sources_closed);
        assert!(!report.loss_free);
    }

    #[test]
    fn unknown_event_epoch_is_rejected() {
        let mut input = fixture();
        input["events"][0]["epoch_id"] = json!("other-epoch");
        let report = audit(&input);
        assert!(!report.ledger_well_formed);
        assert!(report.errors.iter().any(|e| e.code == "E_EVENT_SOURCE_UNKNOWN"));
    }

    #[test]
    fn source_identity_must_be_unique() {
        let mut input = fixture();
        let source = input["source_epochs"][0].clone();
        input["source_epochs"].as_array_mut().unwrap().push(source);
        let report = audit(&input);
        assert!(!report.ledger_well_formed);
        assert!(report.errors.iter().any(|e| e.code == "E_SOURCE_DUPLICATE"));
    }

    #[test]
    fn source_requires_final_close_fence() {
        let mut input = fixture();
        input["source_epochs"][0]["close_fence_digest"] = json!("");
        let report = audit(&input);
        assert!(!report.ledger_well_formed);
        assert!(!report.sources_closed);
    }
}
