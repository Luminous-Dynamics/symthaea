use serde_json::Value;
use symthaea_household_service_evaluator::{
    MeterMeasurement, TopologyAuditReference, WaterBalanceDisposition, WaterBalanceInput,
    evaluate_water_balance,
};

fn meter(value_l: f64, stream: &str, boundary: &str) -> MeterMeasurement {
    MeterMeasurement {
        value_l,
        stream_id: stream.to_owned(),
        measurement_boundary_id: boundary.to_owned(),
        window_start_unix_ms: 1,
        window_end_unix_ms: 86_400_001,
        calibration_qualified: true,
        calibration_record_sha256: Some("a".repeat(64)),
    }
}

#[test]
fn existing_wcont_008_fixture_is_recomputed_with_conservative_meter_bounds() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json"
    ))
    .expect("committed WCONT fixture must remain valid JSON");

    let scenario = &corpus["scenarios"][7];
    assert_eq!(scenario["id"], "WCONT-008");
    let input = &scenario["input"];
    // The fixture's meter_uncertainty_l field does not declare whether it is
    // per-meter or already aggregated. For this negative control, conservatively
    // treat it as an absolute bound for each of the four terms and sum them.
    // The fixture remains synthetic; this interpretation is not calibration data.
    let per_term_bound_l = input["meter_uncertainty_l"].as_f64().unwrap();
    let aggregate_bound_l = 4.0 * per_term_bound_l;

    let result = evaluate_water_balance(&WaterBalanceInput {
        inlet: meter(
            input["municipal_inlet_volume_l"].as_f64().unwrap(),
            "inlet-stream",
            "inlet-boundary",
        ),
        tank_storage_delta: meter(
            input["tank_storage_delta_l"].as_f64().unwrap(),
            "tank-stream",
            "tank-boundary",
        ),
        delivered_outlet: meter(
            input["outlet_meter_volume_l"].as_f64().unwrap(),
            "outlet-stream",
            "outlet-boundary",
        ),
        purge_and_overflow: meter(
            input["purge_and_overflow_meter_volume_l"].as_f64().unwrap(),
            "purge-stream",
            "purge-boundary",
        ),
        residual_absolute_error_bound_l: aggregate_bound_l,
        topology_audit: Some(TopologyAuditReference {
            report_sha256: "b".repeat(64),
        }),
    })
    .expect("synthetic known-answer input must be evaluable");

    assert_eq!(result.residual_l, -200.0);
    assert_eq!(result.residual_lower_l, -220.0);
    assert_eq!(result.residual_upper_l, -180.0);
    assert_eq!(
        result.disposition,
        WaterBalanceDisposition::ContradictionBeyondDeclaredBound
    );
}
