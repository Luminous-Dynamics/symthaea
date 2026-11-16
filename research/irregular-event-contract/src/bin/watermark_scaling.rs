use std::hint::black_box;
use std::time::{Duration, Instant};

use symthaea_irregular_event_contract::{
    CompletenessVerdict, EventRecord, Measurement, SourceKey, SourceProgressOperation,
    SourceProgressRevision, SourceWatermarkLedger,
};

const INDEXED_REPETITIONS: usize = 101;
const VERIFIED_REPETITIONS: usize = 15;
const EVENT_COUNTS: [usize; 3] = [1_000, 10_000, 100_000];

fn percentile_micros(samples: &mut [Duration], percentile: f64) -> f64 {
    samples.sort_unstable();
    let rank = ((percentile * samples.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[rank].as_secs_f64() * 1_000_000.0
}

fn build_ledger(event_count: usize) -> Result<SourceWatermarkLedger, String> {
    if event_count == 0 {
        return Err("event_count must be nonzero".to_owned());
    }
    let event_count_i64 =
        i64::try_from(event_count).map_err(|_| "event_count exceeds i64".to_owned())?;
    let source = SourceKey {
        entity_id: 1,
        modality_id: 1,
        source_id: 1,
    };
    let mut ledger = SourceWatermarkLedger::default();
    ledger
        .append(SourceProgressRevision {
            progress_id: 1,
            source,
            stream_generation: 1,
            progress_sequence: 1,
            recorded_at_us: 10,
            policy_sha256: [0xA5; 32],
            operation: SourceProgressOperation::Advance {
                exclusive_event_time_us: event_count_i64 + 1,
            },
        })
        .map_err(|error| format!("append source progress: {error}"))?;

    // Multiplication by 37 permutes the event-time keys for the selected
    // ledger sizes. Arrival order therefore differs from event-time order.
    let events = (0..event_count).map(|index| {
        let event_id = index as u64 + 1;
        EventRecord {
            event_id,
            entity_id: source.entity_id,
            modality_id: source.modality_id,
            source_id: source.source_id,
            event_time_us: ((index * 37) % event_count) as i64,
            available_at_us: 20,
            source_sequence: event_id,
            unit_code: 1,
            measurement: Measurement::Observed {
                scaled_value: index as i64,
                decimal_scale: 0,
            },
            source_sha256: [0x5A; 32],
        }
    });
    let classifications = ledger
        .record_event_arrival_batch(events, 20)
        .map_err(|error| format!("record event batch: {error}"))?;
    if classifications
        .iter()
        .any(|classification| !matches!(classification, symthaea_irregular_event_contract::EventArrivalClassification::Late(_)))
    {
        return Err("benchmark fixture expected every event to breach the declared watermark".to_owned());
    }
    Ok(ledger)
}

fn run_case(event_count: usize) -> Result<(), String> {
    let ledger = build_ledger(event_count)?;
    let source = SourceKey {
        entity_id: 1,
        modality_id: 1,
        source_id: 1,
    };
    let start = (event_count / 2) as i64;
    let end = start + 16;
    let query = || ledger.completeness(1, &[source], start, end, 20);
    let verified_query = || ledger.completeness_verified(1, &[source], start, end, 20);

    let warm = query().map_err(|error| format!("warm indexed query: {error}"))?;
    if warm.verdict != CompletenessVerdict::Contested || warm.late_breaches.len() != 16 {
        return Err(format!(
            "unexpected known-answer result: verdict={:?}, breaches={}, candidates={}",
            warm.verdict,
            warm.late_breaches.len(),
            warm.late_breach_candidates_examined
        ));
    }
    let verified = verified_query().map_err(|error| format!("warm verified query: {error}"))?;
    if warm != verified {
        return Err("indexed and verified completeness reports differ".to_owned());
    }

    let mut indexed_samples = Vec::with_capacity(INDEXED_REPETITIONS);
    for _ in 0..INDEXED_REPETITIONS {
        let started = Instant::now();
        let report = black_box(query().map_err(|error| format!("indexed query: {error}"))?);
        black_box(report.verdict);
        indexed_samples.push(started.elapsed());
    }

    let mut verified_samples = Vec::with_capacity(VERIFIED_REPETITIONS);
    for _ in 0..VERIFIED_REPETITIONS {
        let started = Instant::now();
        let report = black_box(
            verified_query().map_err(|error| format!("verified query: {error}"))?,
        );
        black_box(report.verdict);
        verified_samples.push(started.elapsed());
    }

    let mut indexed_for_percentiles = indexed_samples.clone();
    let mut verified_for_percentiles = verified_samples.clone();
    let indexed_p50 = percentile_micros(&mut indexed_for_percentiles, 0.50);
    let indexed_p95 = percentile_micros(&mut indexed_for_percentiles, 0.95);
    let indexed_p99 = percentile_micros(&mut indexed_for_percentiles, 0.99);
    let verified_p50 = percentile_micros(&mut verified_for_percentiles, 0.50);
    let verified_p95 = percentile_micros(&mut verified_for_percentiles, 0.95);
    let verified_p99 = percentile_micros(&mut verified_for_percentiles, 0.99);

    println!(
        "{event_count},{INDEXED_REPETITIONS},{VERIFIED_REPETITIONS},{},{},{indexed_p50:.3},{indexed_p95:.3},{indexed_p99:.3},{verified_p50:.3},{verified_p95:.3},{verified_p99:.3}",
        warm.late_breach_candidates_examined,
        warm.late_breaches.len()
    );
    Ok(())
}

fn main() {
    println!(
        "events,indexed_samples,verified_samples,index_candidates,matching_breaches,indexed_p50_us,indexed_p95_us,indexed_p99_us,verified_p50_us,verified_p95_us,verified_p99_us"
    );
    for event_count in EVENT_COUNTS {
        if let Err(error) = run_case(event_count) {
            eprintln!("watermark scaling benchmark failed at {event_count} events: {error}");
            std::process::exit(1);
        }
    }
}
