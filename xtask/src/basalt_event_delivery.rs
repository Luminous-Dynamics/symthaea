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
            let counter = match range.disposition {
                RangeDisposition::Lost => &mut lost_count,
                RangeDisposition::DiscardedRequired => &mut required_discard_count,
                RangeDisposition::DiscardedExcluded => &mut excluded_discard_count,
            };
            if !checked_accumulate(counter, length) {
                push_error(&mut report, "E_RANGE_COUNT_OVERFLOW", "$.classified_ranges", "per-source classified range count overflows u64");
                sequence_ok = false;
                counters_ok = false;
            }
            spans.push(Span { first: range.first_sequence, last: range.last_sequence });
        }

        let attempted_count = source.last_attempted_sequence
            .checked_sub(source.first_attempted_sequence)
            .and_then(|n| n.checked_add(1));
        if let Some(attempted) = attempted_count {
            if !checked_accumulate(&mut report.counts.attempted_sequences, attempted) {
                push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate attempted-sequence count overflows u64");
                sequence_ok = false;
                counters_ok = false;
            }
        } else {
            push_error(&mut report, "E_ATTEMPTED_COUNT_OVERFLOW", "$.source_epochs", "attempted sequence count overflows u64");
            sequence_ok = false;
        }

        if !checked_accumulate(&mut report.counts.published_sequences, events.len() as u64) {
            push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate published count overflows u64");
            sequence_ok = false;
            counters_ok = false;
        }
        if !checked_accumulate(&mut report.counts.lost_sequences, lost_count) {
            push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate lost count overflows u64");
            sequence_ok = false;
            counters_ok = false;
        }
        if !checked_accumulate(&mut report.counts.required_discards, required_discard_count) {
            push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate required-discard count overflows u64");
            sequence_ok = false;
            counters_ok = false;
        }
        if !checked_accumulate(&mut report.counts.excluded_discards, excluded_discard_count) {
            push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate excluded-discard count overflows u64");
            sequence_ok = false;
            counters_ok = false;
        }

        spans.sort_by_key(|span| (span.first, span.last));
        let mut expected = source.first_attempted_sequence;
        let mut completed = false;
        for span in spans {
            if completed {
                push_error(&mut report, "E_SEQUENCE_OVERLAP", "$.source_epochs", "an additional published event or classified range appears after the terminal sequence was already accounted");
                if !checked_accumulate(&mut report.counts.overlapping_sequences, 1) {
                    push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate overlap count overflows u64");
                }
                sequence_ok = false;
                continue;
            }
            if span.first < expected {
                push_error(&mut report, "E_SEQUENCE_OVERLAP", "$.source_epochs", "published event or classified range overlaps an earlier sequence span");
                if !checked_accumulate(&mut report.counts.overlapping_sequences, 1) {
                    push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate overlap count overflows u64");
                }
                sequence_ok = false;
                continue;
            }
            if span.first > expected {
                let gap = span.first - expected;
                push_error(&mut report, "E_SEQUENCE_UNCLASSIFIED_GAP", "$.source_epochs", "one or more attempted sequences are neither published nor classified as lost/discarded");
                if !checked_accumulate(&mut report.counts.unclassified_sequences, gap) {
                    push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate unclassified-sequence count overflows u64");
                }
                sequence_ok = false;
            }
            if span.last < span.first || span.last > source.last_attempted_sequence {
                sequence_ok = false;
                continue;
            }
            if span.last == source.last_attempted_sequence {
                completed = true;
                continue;
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
            let tail_gap = source.last_attempted_sequence.checked_sub(tail_start).and_then(|n| n.checked_add(1));
            if let Some(tail_gap) = tail_gap.filter(|count| *count > 0) {
                push_error(&mut report, "E_SEQUENCE_TAIL_UNACCOUNTED", "$.source_epochs", "final attempted sequences are missing or unclassified; a terminal high-water mark must account for tail loss");
                if !checked_accumulate(&mut report.counts.unclassified_sequences, tail_gap) {
                    push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate unclassified-sequence count overflows u64");
                }
                sequence_ok = false;
            } else if tail_gap.is_none() {
                push_error(&mut report, "E_SEQUENCE_OVERFLOW", "$.source_epochs", "terminal sequence gap length overflows u64");
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
            if !checked_accumulate(&mut report.counts.unmonitored_producers, source.unmonitored_producer_count) {
                push_error(&mut report, "E_AGGREGATE_COUNT_OVERFLOW", "$.source_epochs", "aggregate unmonitored-producer count overflows u64");
            }
            push_error(&mut report, "E_UNMONITORED_PRODUCER", "$.source_epochs", "unmonitored producers prevent a loss-free delivery claim");
            sources_closed = false;
            loss_free = false;
        }
        if lost_count > 0 || required_discard_count > 0 {
            loss_free = false;
        }
        if excluded_discard_count > 0 {
            push_error(&mut report, "E_POLICY_EXCLUSION_UNVERIFIED", "$.classified_ranges", "policy-excluded discards remain visible but cannot be treated as loss-free until a separate pinned policy evaluator approves them");
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
            "E_EVENT_TIME" | "E_EVENT_SOURCE_UNKNOWN" | "E_EVENT_DUPLICATE_SEQUENCE" | "E_RANGE_ORDER" |
            "E_RANGE_EXCLUSION_REASON" | "E_RANGE_LOSS_REASON" | "E_RANGE_SOURCE_UNKNOWN" |
            "E_RANGE_COUNT_OVERFLOW" | "E_AGGREGATE_COUNT_OVERFLOW" |
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


#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceEpochSeal {
    format_version: u32,
    suite_id: String,
    source_id: String,
    epoch_id: String,
    partition_id: String,
    observer_binary_digest: String,
    observer_configuration_digest: String,
    first_attempted_sequence: u64,
    last_attempted_sequence: u64,
    published_event_count: u64,
    published_event_chain_root: String,
    classified_range_count: u64,
    classified_range_chain_root: String,
    close_fence_digest: String,
    producer_quiescence_evidence_digest: String,
    producer_quiesced_at_monotonic_ns: u64,
    consumer_drain_evidence_digest: String,
    consumer_drained_at_monotonic_ns: u64,
    generation: u64,
    previous_seal_digest: Option<String>,
    ledger_digest: String,
    commitment_digest: String,
}

#[derive(Debug, Serialize)]
pub struct EpochSealReport {
    pub report_version: u32,
    pub ledger_digest: String,
    pub seal_commitment: Option<String>,
    pub source_id: Option<String>,
    pub epoch_id: Option<String>,
    pub partition_id: Option<String>,
    pub ledger_reconciled: bool,
    pub seal_well_formed: bool,
    pub ledger_binding_matches: bool,
    pub event_chain_matches: bool,
    pub classified_range_chain_matches: bool,
    pub source_close_claims_match: bool,
    pub commitment_matches: bool,
    pub integrity_passed: bool,
    pub signature_verified: bool,
    pub signer_authorized: bool,
    pub anti_rollback_verified: bool,
    pub close_coordinator_qualified: bool,
    pub observer_delivery_qualified: bool,
    pub qualification_eligible: bool,
    pub errors: Vec<DeliveryAuditError>,
}

pub fn audit_seal_files(ledger_path: &Path, seal_path: &Path) -> anyhow::Result<EpochSealReport> {
    let ledger_bytes = read_bounded(ledger_path)?;
    let seal_bytes = read_bounded(seal_path)?;
    Ok(audit_seal_bytes(&ledger_bytes, &seal_bytes))
}

fn audit_seal_bytes(ledger_bytes: &[u8], seal_bytes: &[u8]) -> EpochSealReport {
    let mut report = EpochSealReport {
        report_version: 1,
        ledger_digest: sha256_tagged(ledger_bytes),
        seal_commitment: None,
        source_id: None,
        epoch_id: None,
        partition_id: None,
        ledger_reconciled: false,
        seal_well_formed: false,
        ledger_binding_matches: false,
        event_chain_matches: false,
        classified_range_chain_matches: false,
        source_close_claims_match: false,
        commitment_matches: false,
        integrity_passed: false,
        signature_verified: false,
        signer_authorized: false,
        anti_rollback_verified: false,
        close_coordinator_qualified: false,
        observer_delivery_qualified: false,
        qualification_eligible: false,
        errors: Vec::new(),
    };

    let ledger_audit = audit_bytes(ledger_bytes);
    report.ledger_reconciled = ledger_audit.ledger_well_formed
        && ledger_audit.sequence_accounting_reconciled
        && ledger_audit.counters_consistent
        && ledger_audit.sources_closed;
    if !report.ledger_reconciled {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_LEDGER_NOT_RECONCILED".to_string(),
            path: "$.ledger".to_string(),
            message: "source-epoch seal requires a structurally valid, sequence-reconciled ledger with consistent counters and closed sources".to_string(),
        });
    }

    let ledger: DeliveryLedger = match serde_json::from_slice(ledger_bytes) {
        Ok(value) => value,
        Err(error) => {
            report.errors.push(DeliveryAuditError {
                code: "E_SEAL_LEDGER_JSON".to_string(),
                path: "$.ledger".to_string(),
                message: format!("cannot parse delivery ledger: {error}"),
            });
            return report;
        }
    };
    let seal: SourceEpochSeal = match serde_json::from_slice(seal_bytes) {
        Ok(value) => value,
        Err(error) => {
            report.errors.push(DeliveryAuditError {
                code: "E_SEAL_JSON".to_string(),
                path: "$.seal".to_string(),
                message: format!("invalid or non-strict source-epoch seal: {error}"),
            });
            return report;
        }
    };

    report.source_id = Some(seal.source_id.clone());
    report.epoch_id = Some(seal.epoch_id.clone());
    report.partition_id = Some(seal.partition_id.clone());
    report.seal_commitment = Some(seal.commitment_digest.clone());

    let mut seal_well_formed = true;
    if seal.format_version != 1 || seal.suite_id != "basalt-source-epoch-seal-v1" {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_VERSION".to_string(),
            path: "$.seal".to_string(),
            message: "unsupported source-epoch seal version or suite_id".to_string(),
        });
        seal_well_formed = false;
    }
    for (field, digest) in [
        ("observer_binary_digest", seal.observer_binary_digest.as_str()),
        ("observer_configuration_digest", seal.observer_configuration_digest.as_str()),
        ("published_event_chain_root", seal.published_event_chain_root.as_str()),
        ("classified_range_chain_root", seal.classified_range_chain_root.as_str()),
        ("close_fence_digest", seal.close_fence_digest.as_str()),
        ("producer_quiescence_evidence_digest", seal.producer_quiescence_evidence_digest.as_str()),
        ("consumer_drain_evidence_digest", seal.consumer_drain_evidence_digest.as_str()),
        ("ledger_digest", seal.ledger_digest.as_str()),
        ("commitment_digest", seal.commitment_digest.as_str()),
    ] {
        if !valid_digest(digest) {
            report.errors.push(DeliveryAuditError {
                code: "E_SEAL_DIGEST".to_string(),
                path: format!("$.seal.{field}"),
                message: "expected a lowercase sha256:<64-hex> digest".to_string(),
            });
            seal_well_formed = false;
        }
    }
    match (&seal.previous_seal_digest, seal.generation) {
        (None, 1) => {}
        (Some(previous), generation) if generation > 1 && valid_digest(previous) => {}
        _ => {
            report.errors.push(DeliveryAuditError {
                code: "E_SEAL_GENERATION_LINK".to_string(),
                path: "$.seal.generation".to_string(),
                message: "generation 1 must have no predecessor; later generations require a valid previous_seal_digest".to_string(),
            });
            seal_well_formed = false;
        }
    }
    if seal.first_attempted_sequence == 0
        || seal.last_attempted_sequence < seal.first_attempted_sequence
        || seal.generation == 0
        || seal.producer_quiesced_at_monotonic_ns == 0
        || seal.consumer_drained_at_monotonic_ns == 0
    {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_BOUNDS".to_string(),
            path: "$.seal".to_string(),
            message: "seal sequence bounds, generation and monotonic close timestamps must be positive and ordered".to_string(),
        });
        seal_well_formed = false;
    }

    let key = SourceEpochKey {
        source_id: seal.source_id.clone(),
        epoch_id: seal.epoch_id.clone(),
        partition_id: seal.partition_id.clone(),
    };
    let source = ledger.source_epochs.iter().find(|source| source.key() == key);
    let Some(source) = source else {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_SOURCE_UNKNOWN".to_string(),
            path: "$.seal.source_id".to_string(),
            message: "seal does not identify a source epoch in the ledger".to_string(),
        });
        report.seal_well_formed = seal_well_formed;
        return report;
    };

    let events: Vec<&DeliveryEvent> = ledger.events.iter().filter(|event| event.key() == key).collect();
    let ranges: Vec<&ClassifiedRange> = ledger.classified_ranges.iter().filter(|range| range.key() == key).collect();
    let event_root = event_chain_root(&key, &events);
    let range_root = classified_range_chain_root(&key, &ranges);

    report.ledger_binding_matches = seal.ledger_digest == report.ledger_digest
        && seal.observer_binary_digest == ledger.observer_binary_digest
        && seal.observer_configuration_digest == ledger.observer_configuration_digest
        && seal.first_attempted_sequence == source.first_attempted_sequence
        && seal.last_attempted_sequence == source.last_attempted_sequence
        && seal.published_event_count == events.len() as u64
        && seal.classified_range_count == ranges.len() as u64
        && seal.close_fence_digest == source.close_fence_digest;
    if !report.ledger_binding_matches {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_LEDGER_BINDING".to_string(),
            path: "$.seal".to_string(),
            message: "seal ledger digest, observer identity, sequence bounds, record counts or close-fence digest do not match the supplied ledger".to_string(),
        });
    }

    report.event_chain_matches = event_root.as_deref() == Some(seal.published_event_chain_root.as_str());
    if !report.event_chain_matches {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_EVENT_CHAIN".to_string(),
            path: "$.seal.published_event_chain_root".to_string(),
            message: "published-event chain root does not match the ordered source-epoch event records".to_string(),
        });
    }
    report.classified_range_chain_matches =
        range_root.as_deref() == Some(seal.classified_range_chain_root.as_str());
    if !report.classified_range_chain_matches {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_RANGE_CHAIN".to_string(),
            path: "$.seal.classified_range_chain_root".to_string(),
            message: "classified-range chain root does not match the ordered source-epoch loss/discard records".to_string(),
        });
    }

    let capture_window_valid = seal.producer_quiesced_at_monotonic_ns >= ledger.capture_start_monotonic_ns
        && seal.consumer_drained_at_monotonic_ns >= seal.producer_quiesced_at_monotonic_ns
        && seal.consumer_drained_at_monotonic_ns <= ledger.capture_end_monotonic_ns;
    report.source_close_claims_match = source.producer_quiesced
        && source.consumer_drained
        && source.restart_count == 0
        && source.unmonitored_producer_count == 0
        && capture_window_valid;
    if !report.source_close_claims_match {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_CLOSE_CLAIMS".to_string(),
            path: "$.seal".to_string(),
            message: "ledger close assertions or monotonic quiescence/drain ordering do not support this seal".to_string(),
        });
    }

    if let Some(commitment) = compute_seal_commitment(&seal) {
        report.commitment_matches = commitment == seal.commitment_digest;
    } else {
        report.commitment_matches = false;
    }
    if !report.commitment_matches {
        report.errors.push(DeliveryAuditError {
            code: "E_SEAL_COMMITMENT".to_string(),
            path: "$.seal.commitment_digest".to_string(),
            message: "domain-separated source-epoch commitment does not match the supplied seal fields".to_string(),
        });
    }

    report.seal_well_formed = seal_well_formed;
    report.integrity_passed = report.ledger_reconciled
        && report.seal_well_formed
        && report.ledger_binding_matches
        && report.event_chain_matches
        && report.classified_range_chain_matches
        && report.source_close_claims_match
        && report.commitment_matches;
    report
}

fn event_chain_root(key: &SourceEpochKey, events: &[&DeliveryEvent]) -> Option<String> {
    let mut ordered = events.to_vec();
    ordered.sort_by_key(|event| (event.sequence, event.observed_at_monotonic_ns, event.event_digest.clone()));
    let mut state = Sha256::digest(b"BASALT/EVENT-CHAIN-INIT/v1\0").to_vec();
    for event in &ordered {
        let mut leaf = b"BASALT/EVENT-LEAF/v1\0".to_vec();
        append_string(&mut leaf, &key.source_id)?;
        append_string(&mut leaf, &key.epoch_id)?;
        append_string(&mut leaf, &key.partition_id)?;
        append_u64(&mut leaf, event.sequence);
        append_u64(&mut leaf, event.observed_at_monotonic_ns);
        append_digest_bytes(&mut leaf, &event.event_digest)?;
        let leaf_digest = Sha256::digest(&leaf);
        let mut step = Sha256::new();
        step.update(b"BASALT/EVENT-CHAIN-STEP/v1\0");
        step.update(&state);
        step.update(leaf_digest);
        state = step.finalize().to_vec();
    }
    let mut root = Sha256::new();
    root.update(b"BASALT/EVENT-CHAIN-ROOT/v1\0");
    root.update(&state);
    root.update((ordered.len() as u64).to_be_bytes());
    Some(sha256_from_bytes(root.finalize().as_slice()))
}

fn classified_range_chain_root(key: &SourceEpochKey, ranges: &[&ClassifiedRange]) -> Option<String> {
    let mut ordered = ranges.to_vec();
    ordered.sort_by_key(|range| (range.first_sequence, range.last_sequence));
    let mut state = Sha256::digest(b"BASALT/RANGE-CHAIN-INIT/v1\0").to_vec();
    for range in &ordered {
        let mut leaf = b"BASALT/RANGE-LEAF/v1\0".to_vec();
        append_string(&mut leaf, &key.source_id)?;
        append_string(&mut leaf, &key.epoch_id)?;
        append_string(&mut leaf, &key.partition_id)?;
        append_u64(&mut leaf, range.first_sequence);
        append_u64(&mut leaf, range.last_sequence);
        leaf.push(match range.disposition {
            RangeDisposition::Lost => 1,
            RangeDisposition::DiscardedRequired => 2,
            RangeDisposition::DiscardedExcluded => 3,
        });
        leaf.push(match range.reason {
            RangeReason::RingbufReservationFailureUnknown => 1,
            RangeReason::PerfBufferLostSample => 2,
            RangeReason::ConsumerQueueOverrun => 3,
            RangeReason::ExporterDrop => 4,
            RangeReason::DecoderRejection => 5,
            RangeReason::ObserverRestart => 6,
            RangeReason::IntentionalPolicySuppression => 7,
            RangeReason::Unknown => 8,
        });
        let leaf_digest = Sha256::digest(&leaf);
        let mut step = Sha256::new();
        step.update(b"BASALT/RANGE-CHAIN-STEP/v1\0");
        step.update(&state);
        step.update(leaf_digest);
        state = step.finalize().to_vec();
    }
    let mut root = Sha256::new();
    root.update(b"BASALT/RANGE-CHAIN-ROOT/v1\0");
    root.update(&state);
    root.update((ordered.len() as u64).to_be_bytes());
    Some(sha256_from_bytes(root.finalize().as_slice()))
}

fn compute_seal_commitment(seal: &SourceEpochSeal) -> Option<String> {
    let mut bytes = b"BASALT/SOURCE-EPOCH-SEAL/v1\0".to_vec();
    bytes.extend_from_slice(&seal.format_version.to_be_bytes());
    append_string(&mut bytes, &seal.suite_id)?;
    append_string(&mut bytes, &seal.source_id)?;
    append_string(&mut bytes, &seal.epoch_id)?;
    append_string(&mut bytes, &seal.partition_id)?;
    append_digest_bytes(&mut bytes, &seal.observer_binary_digest)?;
    append_digest_bytes(&mut bytes, &seal.observer_configuration_digest)?;
    append_u64(&mut bytes, seal.first_attempted_sequence);
    append_u64(&mut bytes, seal.last_attempted_sequence);
    append_u64(&mut bytes, seal.published_event_count);
    append_digest_bytes(&mut bytes, &seal.published_event_chain_root)?;
    append_u64(&mut bytes, seal.classified_range_count);
    append_digest_bytes(&mut bytes, &seal.classified_range_chain_root)?;
    append_digest_bytes(&mut bytes, &seal.close_fence_digest)?;
    append_digest_bytes(&mut bytes, &seal.producer_quiescence_evidence_digest)?;
    append_u64(&mut bytes, seal.producer_quiesced_at_monotonic_ns);
    append_digest_bytes(&mut bytes, &seal.consumer_drain_evidence_digest)?;
    append_u64(&mut bytes, seal.consumer_drained_at_monotonic_ns);
    append_u64(&mut bytes, seal.generation);
    match &seal.previous_seal_digest {
        Some(previous) => {
            bytes.push(1);
            append_digest_bytes(&mut bytes, previous)?;
        }
        None => bytes.push(0),
    }
    append_digest_bytes(&mut bytes, &seal.ledger_digest)?;
    Some(sha256_tagged(&bytes))
}

fn append_string(bytes: &mut Vec<u8>, value: &str) -> Option<()> {
    let length = u32::try_from(value.len()).ok()?;
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Some(())
}

fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn append_digest_bytes(bytes: &mut Vec<u8>, value: &str) -> Option<()> {
    bytes.extend_from_slice(&decode_digest(value)?);
    Some(())
}

fn decode_digest(value: &str) -> Option<[u8; 32]> {
    let hex = value.strip_prefix("sha256:")?;
    if hex.len() != 64 {
        return None;
    }
    let mut output = [0u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        let high = hex.as_bytes()[index * 2];
        let low = hex.as_bytes()[index * 2 + 1];
        let nibble = |value: u8| match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            _ => None,
        };
        *byte = (nibble(high)? << 4) | nibble(low)?;
    }
    Some(output)
}

fn sha256_from_bytes(digest: &[u8]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("sha256:{encoded}")
}

fn checked_accumulate(total: &mut u64, increment: u64) -> bool {
    match total.checked_add(increment) {
        Some(value) => {
            *total = value;
            true
        }
        None => false,
    }
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

    fn make_seal(ledger_bytes: &[u8]) -> Vec<u8> {
        let ledger: DeliveryLedger = serde_json::from_slice(ledger_bytes).unwrap();
        let source = ledger.source_epochs.first().unwrap();
        let key = source.key();
        let events: Vec<&DeliveryEvent> = ledger.events.iter().filter(|event| event.key() == key).collect();
        let ranges: Vec<&ClassifiedRange> = ledger.classified_ranges.iter().filter(|range| range.key() == key).collect();
        let mut seal = SourceEpochSeal {
            format_version: 1,
            suite_id: "basalt-source-epoch-seal-v1".to_string(),
            source_id: key.source_id,
            epoch_id: key.epoch_id,
            partition_id: key.partition_id,
            observer_binary_digest: ledger.observer_binary_digest,
            observer_configuration_digest: ledger.observer_configuration_digest,
            first_attempted_sequence: source.first_attempted_sequence,
            last_attempted_sequence: source.last_attempted_sequence,
            published_event_count: events.len() as u64,
            published_event_chain_root: event_chain_root(&source.key(), &events).unwrap(),
            classified_range_count: ranges.len() as u64,
            classified_range_chain_root: classified_range_chain_root(&source.key(), &ranges).unwrap(),
            close_fence_digest: source.close_fence_digest.clone(),
            producer_quiescence_evidence_digest: digest(7),
            producer_quiesced_at_monotonic_ns: 800,
            consumer_drain_evidence_digest: digest(8),
            consumer_drained_at_monotonic_ns: 900,
            generation: 1,
            previous_seal_digest: None,
            ledger_digest: sha256_tagged(ledger_bytes),
            commitment_digest: digest(9),
        };
        seal.commitment_digest = compute_seal_commitment(&seal).unwrap();
        serde_json::to_vec(&seal).unwrap()
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
    fn classified_range_cannot_cover_sequences_already_published() {
        let mut input = fixture();
        input["source_epochs"][0]["reported_lost_count"] = json!(3);
        input["classified_ranges"] = json!([{
            "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":1,"last_sequence":3,
            "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
        }]);
        let report = audit(&input);
        assert!(!report.sequence_accounting_reconciled);
        assert!(report.errors.iter().any(|e| e.code == "E_SEQUENCE_OVERLAP"));
        assert!(!report.audit_passed);
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
        assert!(!report.loss_free);
        assert!(!report.audit_passed);
        assert!(report.errors.iter().any(|e| e.code == "E_POLICY_EXCLUSION_UNVERIFIED"));
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
    fn per_source_classified_range_count_overflow_is_rejected() {
        let mut input = fixture();
        input["source_epochs"][0]["last_attempted_sequence"] = json!(u64::MAX);
        input["source_epochs"][0]["reported_lost_count"] = json!(u64::MAX);
        input["source_epochs"][0]["reported_published_count"] = json!(0);
        input["events"] = json!([]);
        input["classified_ranges"] = json!([
            {
                "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
                "first_sequence":1,"last_sequence":u64::MAX,
                "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
            },
            {
                "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
                "first_sequence":1,"last_sequence":u64::MAX,
                "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
            }
        ]);
        let report = audit(&input);
        assert!(!report.ledger_well_formed);
        assert!(!report.audit_passed);
        assert!(report.errors.iter().any(|e| e.code == "E_RANGE_COUNT_OVERFLOW"));
    }

    #[test]
    fn aggregate_sequence_totals_overflow_fail_closed() {
        let mut input = fixture();
        input["events"] = json!([]);
        input["source_epochs"][0]["first_attempted_sequence"] = json!(1);
        input["source_epochs"][0]["last_attempted_sequence"] = json!(u64::MAX);
        input["source_epochs"][0]["reported_published_count"] = json!(0);
        input["source_epochs"][0]["reported_lost_count"] = json!(u64::MAX);
        input["classified_ranges"] = json!([{
            "source_id":"bpf:process-exit","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":1,"last_sequence":u64::MAX,
            "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
        }]);
        let mut second = input["source_epochs"][0].clone();
        second["source_id"] = json!("bpf:process-exit-secondary");
        input["source_epochs"] = json!([input["source_epochs"][0].clone(), second]);
        input["classified_ranges"].as_array_mut().unwrap().push(json!({
            "source_id":"bpf:process-exit-secondary","epoch_id":"attach-1","partition_id":"cpu:0",
            "first_sequence":1,"last_sequence":u64::MAX,
            "disposition":"lost","reason":"ringbuf_reservation_failure_unknown"
        }));
        let report = audit(&input);
        assert!(!report.ledger_well_formed);
        assert!(!report.audit_passed);
        assert!(report.errors.iter().any(|e| e.code == "E_AGGREGATE_COUNT_OVERFLOW"));
    }

    #[test]
    fn matching_epoch_seal_passes_integrity_but_never_qualifies_trust() {
        let ledger = fixture().to_string().into_bytes();
        let seal = make_seal(&ledger);
        let report = audit_seal_bytes(&ledger, &seal);
        assert!(report.integrity_passed, "{:?}", report.errors);
        assert!(!report.signature_verified);
        assert!(!report.signer_authorized);
        assert!(!report.anti_rollback_verified);
        assert!(!report.close_coordinator_qualified);
        assert!(!report.observer_delivery_qualified);
        assert!(!report.qualification_eligible);
    }

    #[test]
    fn byte_level_ledger_mutation_breaks_exact_seal_binding() {
        let ledger = fixture().to_string().into_bytes();
        let seal = make_seal(&ledger);
        let mut mutated = ledger.clone();
        mutated.push(b' ');
        let report = audit_seal_bytes(&mutated, &seal);
        assert!(!report.integrity_passed);
        assert!(!report.ledger_binding_matches);
        assert!(report.errors.iter().any(|error| error.code == "E_SEAL_LEDGER_BINDING"));
    }

    #[test]
    fn recomputed_ledger_digest_does_not_hide_mutated_event_chain() {
        let original = fixture().to_string().into_bytes();
        let original_seal = make_seal(&original);
        let mut seal: SourceEpochSeal = serde_json::from_slice(&original_seal).unwrap();
        let mut changed: Value = serde_json::from_slice(&original).unwrap();
        changed["events"][0]["event_digest"] = json!(digest(99));
        let changed_bytes = changed.to_string().into_bytes();
        seal.ledger_digest = sha256_tagged(&changed_bytes);
        seal.commitment_digest = compute_seal_commitment(&seal).unwrap();
        let report = audit_seal_bytes(&changed_bytes, &serde_json::to_vec(&seal).unwrap());
        assert!(report.ledger_binding_matches);
        assert!(!report.event_chain_matches);
        assert!(!report.integrity_passed);
        assert!(report.errors.iter().any(|error| error.code == "E_SEAL_EVENT_CHAIN"));
    }

    #[test]
    fn seal_commitment_tampering_is_rejected() {
        let ledger = fixture().to_string().into_bytes();
        let seal_bytes = make_seal(&ledger);
        let mut seal: Value = serde_json::from_slice(&seal_bytes).unwrap();
        seal["consumer_drain_evidence_digest"] = json!(digest(77));
        let report = audit_seal_bytes(&ledger, seal.to_string().as_bytes());
        assert!(!report.commitment_matches);
        assert!(!report.integrity_passed);
    }

    #[test]
    fn later_generation_requires_a_predecessor_digest() {
        let ledger = fixture().to_string().into_bytes();
        let seal_bytes = make_seal(&ledger);
        let mut seal: SourceEpochSeal = serde_json::from_slice(&seal_bytes).unwrap();
        seal.generation = 2;
        seal.commitment_digest = compute_seal_commitment(&seal).unwrap();
        let report = audit_seal_bytes(&ledger, &serde_json::to_vec(&seal).unwrap());
        assert!(!report.seal_well_formed);
        assert!(report.errors.iter().any(|error| error.code == "E_SEAL_GENERATION_LINK"));
    }

    #[test]
    fn consumer_drain_cannot_precede_producer_quiescence() {
        let ledger = fixture().to_string().into_bytes();
        let seal_bytes = make_seal(&ledger);
        let mut seal: SourceEpochSeal = serde_json::from_slice(&seal_bytes).unwrap();
        seal.producer_quiesced_at_monotonic_ns = 900;
        seal.consumer_drained_at_monotonic_ns = 800;
        seal.commitment_digest = compute_seal_commitment(&seal).unwrap();
        let report = audit_seal_bytes(&ledger, &serde_json::to_vec(&seal).unwrap());
        assert!(!report.source_close_claims_match);
        assert!(!report.integrity_passed);
    }

    #[test]
    fn seal_cannot_be_retargeted_to_an_unknown_source() {
        let ledger = fixture().to_string().into_bytes();
        let seal_bytes = make_seal(&ledger);
        let mut seal: SourceEpochSeal = serde_json::from_slice(&seal_bytes).unwrap();
        seal.source_id = "different-source".to_string();
        seal.commitment_digest = compute_seal_commitment(&seal).unwrap();
        let report = audit_seal_bytes(&ledger, &serde_json::to_vec(&seal).unwrap());
        assert!(!report.integrity_passed);
        assert!(report.errors.iter().any(|error| error.code == "E_SEAL_SOURCE_UNKNOWN"));
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
