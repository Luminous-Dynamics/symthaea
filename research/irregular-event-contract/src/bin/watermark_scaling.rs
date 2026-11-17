use std::hint::black_box;
use std::time::{Duration, Instant};

use symthaea_irregular_event_contract::{
    EventArrivalClassification, EventRecord, Measurement, SourceKey, SourceProgressOperation,
    SourceProgressRevision, SourceWatermarkLedger,
};

const INDEXED_REPETITIONS: usize = 51;
const SCAN_REPETITIONS: usize = 31;
const VERIFIED_REPETITIONS: usize = 15;
const EVENT_COUNTS: [usize; 3] = [4_096, 16_384, 65_536];
const SOURCE_COUNTS: [usize; 3] = [1, 4, 16];
const INTERVAL_WIDTHS: [usize; 3] = [1, 16, 256];
const LATE_RATES_BASIS_POINTS: [u16; 4] = [0, 100, 5_000, 10_000];
const PROGRESS_RECORDED_AT_US: i64 = 1_000_000;
const FIRST_ARRIVAL_AT_US: i64 = 2_000_000;
const ARRIVAL_STEP_US: i64 = 1_000_000;
const KNOWLEDGE_CUTOFFS: [i64; 2] = [3_000_000, 5_000_000];

fn percentile_micros(samples: &mut [Duration], percentile: f64) -> f64 {
    samples.sort_unstable();
    let rank = ((percentile * samples.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[rank].as_secs_f64() * 1_000_000.0
}

fn source_key(source_id: usize) -> SourceKey {
    SourceKey {
        entity_id: 1,
        modality_id: 1,
        source_id: source_id as u64,
    }
}

fn build_ledger(
    event_count: usize,
    source_count: usize,
    late_rate_basis_points: u16,
) -> Result<(SourceWatermarkLedger, Vec<SourceKey>), String> {
    if event_count == 0 || source_count == 0 || event_count % source_count != 0 {
        return Err("event_count must be nonzero and divisible by source_count".to_owned());
    }
    if late_rate_basis_points > 10_000 {
        return Err(
            "late-event rate must be in the inclusive range 0..=10000 basis points".to_owned(),
        );
    }
    let per_source_count = event_count / source_count;
    if per_source_count < *INTERVAL_WIDTHS.iter().max().unwrap_or(&1) {
        return Err("per-source event count must cover the widest benchmark interval".to_owned());
    }
    let per_source_count_i64 =
        i64::try_from(per_source_count).map_err(|_| "source event count exceeds i64".to_owned())?;
    let watermark_bound = per_source_count_i64
        .checked_mul(i64::from(late_rate_basis_points))
        .ok_or_else(|| "watermark bound multiplication overflow".to_owned())?
        / 10_000;

    let required_sources: Vec<_> = (1..=source_count).map(source_key).collect();
    let progress =
        required_sources
            .iter()
            .enumerate()
            .map(|(index, source)| SourceProgressRevision {
                progress_id: index as u64 + 1,
                source: *source,
                stream_generation: 1,
                progress_sequence: 1,
                recorded_at_us: PROGRESS_RECORDED_AT_US,
                policy_sha256: [0xA5; 32],
                operation: SourceProgressOperation::Advance {
                    exclusive_event_time_us: watermark_bound,
                },
            });
    let mut ledger = SourceWatermarkLedger::default();
    ledger
        .append_batch(progress)
        .map_err(|error| format!("append source progress: {error}"))?;

    // Four arrival transactions are deliberately distinct from event time.
    // Multiplication by 37 permutes each source's event-time keys relative to
    // receipt order for all configured power-of-two source lengths.
    let mut arrival_batches: [Vec<EventRecord>; 4] = std::array::from_fn(|_| Vec::new());
    let mut next_event_id = 1_u64;
    for source in &required_sources {
        for local_index in 0..per_source_count {
            let arrival_bucket = local_index % arrival_batches.len();
            let available_at_us = FIRST_ARRIVAL_AT_US + (arrival_bucket as i64 * ARRIVAL_STEP_US);
            let event = EventRecord {
                event_id: next_event_id,
                entity_id: source.entity_id,
                modality_id: source.modality_id,
                source_id: source.source_id,
                event_time_us: ((local_index * 37) % per_source_count) as i64,
                available_at_us,
                source_sequence: local_index as u64 + 1,
                unit_code: 1,
                measurement: Measurement::Observed {
                    scaled_value: local_index as i64,
                    decimal_scale: 0,
                },
                source_sha256: [0x5A; 32],
            };
            arrival_batches[arrival_bucket].push(event);
            next_event_id = next_event_id
                .checked_add(1)
                .ok_or_else(|| "event ID overflow".to_owned())?;
        }
    }

    for (bucket, events) in arrival_batches.into_iter().enumerate() {
        let recorded_at_us = FIRST_ARRIVAL_AT_US + (bucket as i64 * ARRIVAL_STEP_US);
        let expected_late_count = events
            .iter()
            .filter(|event| event.event_time_us < watermark_bound)
            .count();
        let classifications = ledger
            .record_event_arrival_batch(events, recorded_at_us)
            .map_err(|error| format!("record arrival bucket {bucket}: {error}"))?;
        let actual_late_count = classifications
            .iter()
            .filter(|classification| matches!(classification, EventArrivalClassification::Late(_)))
            .count();
        if actual_late_count != expected_late_count {
            return Err(format!(
                "late-arrival classification mismatch in bucket {bucket}: expected {expected_late_count}, got {actual_late_count}"
            ));
        }
    }

    Ok((ledger, required_sources))
}

fn run_case(
    event_count: usize,
    source_count: usize,
    interval_width: usize,
    late_rate_basis_points: u16,
    known_at_us: i64,
) -> Result<(), String> {
    let (ledger, required_sources) =
        build_ledger(event_count, source_count, late_rate_basis_points)?;
    let per_source_count = event_count / source_count;
    let interval_start_us = ((per_source_count - interval_width) / 2) as i64;
    let interval_end_exclusive_us = interval_start_us + interval_width as i64;
    let query = || {
        ledger.completeness(
            1,
            &required_sources,
            interval_start_us,
            interval_end_exclusive_us,
            known_at_us,
        )
    };
    let linear_scan_query = || {
        ledger.completeness_linear_scan_reference(
            1,
            &required_sources,
            interval_start_us,
            interval_end_exclusive_us,
            known_at_us,
        )
    };
    let verified_query = || {
        ledger.completeness_verified(
            1,
            &required_sources,
            interval_start_us,
            interval_end_exclusive_us,
            known_at_us,
        )
    };

    // Correctness is established before timing: indexed, linear-scan and
    // verified paths share the complete query contract and must agree exactly
    // on every semantic result field. Only the breach-selection strategy and
    // its diagnostic counters differ.
    let warm = query().map_err(|error| format!("warm indexed query: {error}"))?;
    let scanned = linear_scan_query()
        .map_err(|error| format!("warm linear-scan query: {error}"))?;
    let verified = verified_query().map_err(|error| format!("warm verified query: {error}"))?;
    let same_semantics = |left: &_, right: &_| {
        left.interval_start_us == right.interval_start_us
            && left.interval_end_exclusive_us == right.interval_end_exclusive_us
            && left.known_at_us == right.known_at_us
            && left.verdict == right.verdict
            && left.sources == right.sources
            && left.blocked_sources == right.blocked_sources
            && left.late_breaches == right.late_breaches
    };
    if !same_semantics(&warm, &scanned) || !same_semantics(&warm, verified.report()) {
        return Err(format!(
            "indexed, linear-scan and verified reports differ: events={event_count}, sources={source_count}, width={interval_width}, known_at={known_at_us}"
        ));
    }
    if warm.late_breach_linear_scan_receipts_examined != 0
        || warm.late_breach_candidates_examined > ledger.late_breaches().len()
        || scanned.late_breach_linear_scan_receipts_examined != ledger.late_breaches().len()
        || scanned.late_breach_index_buckets_examined != 0
        || scanned.late_breach_candidates_examined != 0
    {
        return Err("indexed / linear-scan instrumentation counters are inconsistent".to_owned());
    }
    if verified.ledger_sha256() != ledger.state_sha256() {
        return Err("verified report digest does not bind the current ledger snapshot".to_owned());
    }

    let mut indexed_samples = Vec::with_capacity(INDEXED_REPETITIONS);
    for _ in 0..INDEXED_REPETITIONS {
        let started = Instant::now();
        let report = black_box(query().map_err(|error| format!("indexed query: {error}"))?);
        black_box(report.verdict);
        indexed_samples.push(started.elapsed());
    }

    let mut scan_samples = Vec::with_capacity(SCAN_REPETITIONS);
    for _ in 0..SCAN_REPETITIONS {
        let started = Instant::now();
        let report = black_box(
            linear_scan_query().map_err(|error| format!("linear-scan query: {error}"))?,
        );
        black_box((report.verdict, report.late_breaches.len()));
        scan_samples.push(started.elapsed());
    }

    let mut verified_samples = Vec::with_capacity(VERIFIED_REPETITIONS);
    for _ in 0..VERIFIED_REPETITIONS {
        let started = Instant::now();
        let report =
            black_box(verified_query().map_err(|error| format!("verified query: {error}"))?);
        black_box(report.verdict);
        verified_samples.push(started.elapsed());
    }

    let mut indexed_for_percentiles = indexed_samples.clone();
    let mut scan_for_percentiles = scan_samples.clone();
    let mut verified_for_percentiles = verified_samples.clone();
    let indexed_p50 = percentile_micros(&mut indexed_for_percentiles, 0.50);
    let indexed_p95 = percentile_micros(&mut indexed_for_percentiles, 0.95);
    let indexed_p99 = percentile_micros(&mut indexed_for_percentiles, 0.99);
    let scan_p50 = percentile_micros(&mut scan_for_percentiles, 0.50);
    let scan_p95 = percentile_micros(&mut scan_for_percentiles, 0.95);
    let scan_p99 = percentile_micros(&mut scan_for_percentiles, 0.99);
    let verified_p50 = percentile_micros(&mut verified_for_percentiles, 0.50);
    let verified_max = verified_for_percentiles
        .last()
        .map(|duration| duration.as_secs_f64() * 1_000_000.0)
        .unwrap_or_default();

    println!(
        "{event_count},{source_count},{interval_width},{late_rate_basis_points},{known_at_us},{INDEXED_REPETITIONS},{SCAN_REPETITIONS},{VERIFIED_REPETITIONS},{},{},{},{},{scan_p50:.3},{scan_p95:.3},{scan_p99:.3},{indexed_p50:.3},{indexed_p95:.3},{indexed_p99:.3},{verified_p50:.3},{verified_max:.3}",
        warm.late_breach_index_buckets_examined,
        warm.late_breach_candidates_examined,
        scanned.late_breach_linear_scan_receipts_examined,
        warm.late_breaches.len()
    );
    Ok(())
}

fn main() {
    println!(
        "events,sources,interval_width_us,late_rate_basis_points,known_at_us,indexed_samples,scan_samples,verified_samples,index_buckets,indexed_receipt_candidates,scan_receipts_examined,matching_breaches,scan_p50_us,scan_p95_us,scan_p99_us,indexed_p50_us,indexed_p95_us,indexed_p99_us,verified_p50_us,verified_max_us"
    );
    for event_count in EVENT_COUNTS {
        for source_count in SOURCE_COUNTS {
            for interval_width in INTERVAL_WIDTHS {
                for late_rate_basis_points in LATE_RATES_BASIS_POINTS {
                    for known_at_us in KNOWLEDGE_CUTOFFS {
                        if let Err(error) = run_case(
                            event_count,
                            source_count,
                            interval_width,
                            late_rate_basis_points,
                            known_at_us,
                        ) {
                            eprintln!(
                                "watermark scaling benchmark failed at events={event_count}, sources={source_count}, width={interval_width}, late_rate_bp={late_rate_basis_points}, known_at={known_at_us}: {error}"
                            );
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
    }
}
