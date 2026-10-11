// Conformance tests for the snapshot comparison/serialization API.
//
// These fixtures are intentionally synthetic and are unit tests of the
// comparator itself. They are not a canonical baseline, do not claim to be
// measurements of Symthaea, and must never be written to snapshots/ or used
// as evidence that a real benchmark run has not regressed.

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_snapshot(name: &str, mean: f64) -> RegressionSnapshot {
        let metric = MetricValue {
            mean,
            std_dev: 0.01,
            n: 20,
            ci_lower: mean * 0.95,
            ci_upper: mean * 1.05,
        };
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "NBack".to_string(),
            BTreeMap::from([("accuracy".to_string(), metric)]),
        );

        RegressionSnapshot {
            name: name.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_hash: None,
            config_summary: "synthetic unit-test fixture; not empirical evidence".to_string(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics,
        }
    }

    #[test]
    fn synthetic_comparison_detects_large_regression() {
        let baseline = fixture_snapshot("baseline-fixture", 0.75);
        let current = fixture_snapshot("regressed-fixture", 0.25);
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);

        assert!(report.has_critical(), "{}", report.format_summary());
        assert_eq!(report.summary.total_metrics, 1);
        assert_eq!(report.summary.critical, 1);
        assert!(report.results[0].delta_pct < 0.0);
    }

    #[test]
    fn identical_synthetic_metrics_do_not_report_regression() {
        let baseline = fixture_snapshot("baseline-fixture", 0.75);
        let current = fixture_snapshot("current-fixture", 0.75);
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);

        assert!(!report.has_regressions(), "{}", report.format_summary());
        assert_eq!(report.summary.total_metrics, 1);
        assert_eq!(report.summary.passed, 1);
    }

    #[test]
    fn snapshot_json_round_trip_preserves_schema_and_metric_values() {
        let original = fixture_snapshot("round-trip-fixture", 0.80);
        let json = original.to_json().expect("serialize fixture");
        let loaded = RegressionSnapshot::from_json(&json).expect("deserialize fixture");

        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.schema_version, original.schema_version);
        assert_eq!(
            loaded.metrics["NBack"]["accuracy"].mean,
            original.metrics["NBack"]["accuracy"].mean
        );
        assert_eq!(
            loaded.metrics["NBack"]["accuracy"].n,
            original.metrics["NBack"]["accuracy"].n
        );
    }
}
