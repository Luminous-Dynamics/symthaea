//! Reproducible workload-shaped performance characterization for REA claim replay.
//!
//! This is a development benchmark binary, not a production decision path.
//! It separates raw Evidence admission, producer claim evaluation and complete
//! independent replay over identical Evidence sets. It reports distributions per workload cell;
//! it never collapses different sizes into one score.

#[allow(dead_code)]
#[path = "../../../../src/rea_compat.rs"]
mod rea_compat;

use rea_claim_replay_verifier::{
    MAX_CLAIM_JSON_BYTES, MAX_EVIDENCE_RECORDS, MAX_RESULT_JSON_BYTES,
    MAX_TOTAL_EVIDENCE_JSON_BYTES, ReplayOutcome, replay_claim_evaluation,
};
use rea_compat::claims::{
    ClaimDisposition, ClaimPredicate, ClaimRequest, ClaimScope, evaluate_claim,
};
use rea_compat::VerifiedEvidence;
use serde_json::{Value, json};
use std::hint::black_box;
use std::time::Instant;

const EVIDENCE_FIXTURE: &[u8] = include_bytes!(
    "../../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
);
const EVIDENCE_COUNTS: &[usize] = &[1, 4, 16, 64];
const RAW_RESULT_PADDING_BYTES: &[usize] = &[0, 64 * 1024, 1024 * 1024];
const DEFAULT_SAMPLES: usize = 50;
const WARMUP_SAMPLES: usize = 3;

struct Scenario {
    claim: ClaimRequest,
    verified_records: Vec<VerifiedEvidence>,
    claim_bytes: Vec<u8>,
    result_bytes: Vec<u8>,
    evidence_bytes: Vec<Vec<u8>>,
    evidence_total_bytes: usize,
    evidence_max_bytes: usize,
}

fn build_scenario(evidence_count: usize, padding_bytes: usize) -> Scenario {
    assert!((1..=MAX_EVIDENCE_RECORDS).contains(&evidence_count));

    let base = rea_compat::verify_evidence_json(EVIDENCE_FIXTURE)
        .expect("checked-in REA Evidence fixture must verify");
    let padding = "x".repeat(padding_bytes);
    let mut evidence_bytes = Vec::with_capacity(evidence_count);
    let mut verified_records = Vec::with_capacity(evidence_count);

    for index in 0..evidence_count {
        let mut value = base.value.clone();
        value["provider"]["id"] = json!(format!("benchmark-provider-{index}"));
        value["provider"]["name"] = json!(format!("Benchmark Provider {index}"));
        value["analysis_profile"]["provider"] = value["provider"].clone();
        value["analysis_profile"]["digest"] = json!(
            rea_compat::analysis_profile_digest(&value["analysis_profile"])
                .expect("analysis profile digest")
        );

        if padding_bytes == 0 {
            value["raw_result"] = Value::Null;
        } else {
            value["raw_result"] = json!({"opaque_benchmark_payload": padding.clone()});
        }
        value["evidence_id"] =
            json!(rea_compat::compute_evidence_id(&value).expect("Evidence semantic ID"));

        let bytes = serde_json::to_vec(&value).expect("serialize benchmark Evidence");
        assert!(
            bytes.len() <= rea_compat::MAX_EVIDENCE_JSON_BYTES,
            "benchmark Evidence record exceeds the production per-record limit"
        );
        let verified = rea_compat::verify_evidence_json(&bytes)
            .expect("synthetic benchmark Evidence must pass production admission");
        evidence_bytes.push(bytes);
        verified_records.push(verified);
    }

    let evidence_total_bytes = evidence_bytes.iter().map(Vec::len).sum::<usize>();
    let evidence_max_bytes = evidence_bytes.iter().map(Vec::len).max().unwrap_or(0);
    assert!(
        evidence_total_bytes <= MAX_TOTAL_EVIDENCE_JSON_BYTES,
        "benchmark Evidence set exceeds the production aggregate limit"
    );

    let claim = ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: format!("replay-performance-{evidence_count}-{padding_bytes}"),
        predicate: ClaimPredicate::ProviderReportedObservation,
        scope: ClaimScope {
            subject_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_owned(),
            operation: "inspect_artifact".to_owned(),
            platform: None,
            platform_version: None,
            environment_id: None,
        },
        asserted_value: json!({
            "entries": [{"path": "classes.dex", "size": 42}],
            "status": "complete"
        }),
        source_predicate_type: Some("rea.analysis".to_owned()),
        evidence_ids: verified_records
            .iter()
            .map(|record| record.evidence_id.clone())
            .collect(),
        assumptions: Vec::new(),
        falsification_condition:
            "A same-scope verified Evidence record has a different normalized_result".to_owned(),
    };

    let claim_bytes = serde_json::to_vec(&claim).expect("serialize benchmark claim");
    assert!(claim_bytes.len() <= MAX_CLAIM_JSON_BYTES);

    // The production evaluator runs outside the timed sections to construct a
    // valid serialized result for the independent replay input.
    let produced = evaluate_claim(&claim, &verified_records)
        .expect("production evaluator must produce the benchmark result");
    let result_bytes = serde_json::to_vec(&produced).expect("serialize benchmark result");
    assert!(result_bytes.len() <= MAX_RESULT_JSON_BYTES);

    Scenario {
        claim,
        verified_records,
        claim_bytes,
        result_bytes,
        evidence_bytes,
        evidence_total_bytes,
        evidence_max_bytes,
    }
}

fn percentile_ns(samples: &[u128], percentile: usize) -> u128 {
    assert!(!samples.is_empty());
    assert!((1..=100).contains(&percentile));
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let rank = (ordered.len() * percentile).div_ceil(100).max(1);
    ordered[rank - 1]
}

fn measure_ns<F>(warmups: usize, samples: usize, mut operation: F) -> Vec<u128>
where
    F: FnMut(),
{
    for _ in 0..warmups {
        operation();
    }

    let mut measurements = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        operation();
        measurements.push(start.elapsed().as_nanos());
    }
    measurements
}

fn distribution(samples: &[u128]) -> Value {
    json!({
        "sample_count": samples.len(),
        "p50_ns": percentile_ns(samples, 50),
        "p95_ns": percentile_ns(samples, 95),
        "p99_ns": percentile_ns(samples, 99),
        "min_ns": samples.iter().copied().min().unwrap_or(0),
        "max_ns": samples.iter().copied().max().unwrap_or(0)
    })
}

fn run_scenario(evidence_count: usize, padding_bytes: usize, sample_count: usize) -> Value {
    let scenario = build_scenario(evidence_count, padding_bytes);
    let evidence_refs: Vec<&[u8]> = scenario
        .evidence_bytes
        .iter()
        .map(|bytes| bytes.as_slice())
        .collect();

    let admission_samples = measure_ns(WARMUP_SAMPLES, sample_count, || {
        for bytes in &scenario.evidence_bytes {
            let verified = rea_compat::verify_evidence_json(bytes)
                .expect("valid benchmark Evidence must be admitted");
            black_box(verified);
        }
    });

    let producer_samples = measure_ns(WARMUP_SAMPLES, sample_count, || {
        let result = evaluate_claim(&scenario.claim, &scenario.verified_records)
            .expect("valid benchmark producer evaluation must succeed");
        assert_eq!(result.disposition, ClaimDisposition::SupportedWithinScope);
        black_box(result);
    });

    let replay_samples = measure_ns(WARMUP_SAMPLES, sample_count, || {
        let receipt = replay_claim_evaluation(
            &scenario.claim_bytes,
            &evidence_refs,
            &scenario.result_bytes,
        )
        .expect("valid benchmark replay must succeed");
        assert_eq!(receipt.outcome, ReplayOutcome::ReplayMatch);
        assert_eq!(receipt.evidence_count, evidence_count);
        black_box(receipt);
    });

    json!({
        "evidence_count": evidence_count,
        "raw_result_padding_bytes_per_record": padding_bytes,
        "claim_bytes": scenario.claim_bytes.len(),
        "result_bytes": scenario.result_bytes.len(),
        "evidence_total_bytes": scenario.evidence_total_bytes,
        "evidence_max_record_bytes": scenario.evidence_max_bytes,
        "admission": distribution(&admission_samples),
        "producer_evaluation": distribution(&producer_samples),
        "independent_replay": distribution(&replay_samples)
    })
}

fn main() {
    let sample_count = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SAMPLES);
    assert!(
        (1..=10_000).contains(&sample_count),
        "sample count must be between 1 and 10,000"
    );

    let scenarios: Vec<Value> = EVIDENCE_COUNTS
        .iter()
        .flat_map(|count| {
            RAW_RESULT_PADDING_BYTES
                .iter()
                .map(move |padding| (*count, *padding))
        })
        .map(|(count, padding)| run_scenario(count, padding, sample_count))
        .collect();

    let receipt = json!({
        "schema": "luminous.rea.claim-replay-performance.v0",
        "package_version": env!("CARGO_PKG_VERSION"),
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "target_arch": std::env::consts::ARCH,
        "target_os": std::env::consts::OS,
        "warmup_samples_per_operation": WARMUP_SAMPLES,
        "sample_count_per_operation": sample_count,
        "percentile_method": "nearest-rank",
        "timed_operations": [
            "raw Evidence parse + schema + semantic-ID verification",
            "producer typed-claim evaluation over already admitted Evidence values",
            "complete independent claim replay over raw claim/result/Evidence bytes"
        ],
        "scenarios": scenarios
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("serialize benchmark receipt")
    );
}
