use serde_json::Value;
use std::process::Command;

#[test]
fn benchmark_smoke_emits_all_workload_cells_and_three_distinct_operations() {
    let output = Command::new(env!("CARGO_BIN_EXE_rea-claim-replay-bench"))
        .arg("1")
        .output()
        .expect("run replay benchmark smoke");

    assert!(
        output.status.success(),
        "benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&output.stdout).expect("benchmark emits machine-readable JSON");

    assert_eq!(
        receipt["schema"],
        "luminous.rea.claim-replay-performance.v0"
    );
    assert_eq!(receipt["sample_count_per_operation"], 1);
    assert_eq!(receipt["warmup_samples_per_operation"], 3);
    assert_eq!(receipt["percentile_method"], "nearest-rank");
    assert_eq!(receipt["timed_operations"].as_array().unwrap().len(), 3);

    let scenarios = receipt["scenarios"]
        .as_array()
        .expect("scenarios is an array");
    assert_eq!(scenarios.len(), 12);

    let expected_counts = [1, 4, 16, 64];
    let expected_padding = [0, 65_536, 1_048_576];
    for scenario in scenarios {
        let count = scenario["evidence_count"].as_u64().unwrap() as usize;
        let padding = scenario["raw_result_padding_bytes_per_record"]
            .as_u64()
            .unwrap() as usize;
        assert!(expected_counts.contains(&count));
        assert!(expected_padding.contains(&padding));
        assert!(scenario["evidence_total_bytes"].as_u64().unwrap() > 0);
        assert!(scenario["evidence_max_record_bytes"].as_u64().unwrap() > 0);

        for operation in ["admission", "producer_evaluation", "independent_replay"] {
            assert_eq!(
                scenario[operation]["sample_count"],
                1,
                "{operation} must report the requested sample count"
            );
            for percentile in ["p50_ns", "p95_ns", "p99_ns", "min_ns", "max_ns"] {
                assert!(
                    scenario[operation][percentile].as_u64().is_some(),
                    "{operation}.{percentile} must be a nonnegative integer"
                );
            }
        }
    }
}
