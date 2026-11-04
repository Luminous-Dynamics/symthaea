use serde_json::Value;
use symthaea_household_service_evaluator::{
    MeterMeasurement, TopologyAuditReference, WaterBalanceDisposition, WaterBalanceInput,
    evaluate_water_balance,
};

fn meter(value_l: f64, error_bound_l: f64, stream: &str, boundary: &str) -> MeterMeasurement {
    MeterMeasurement {
        value_l,
        absolute_error_bound_l: error_bound_l,
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
    let input = WaterBalanceInput {
        inlet: meter(
            input["municipal_inlet_volume_l"].as_f64().unwrap(),
            per_term_bound_l,
            "inlet-stream",
            "inlet-boundary",
        ),
        tank_storage_delta: meter(
            input["tank_storage_delta_l"].as_f64().unwrap(),
            per_term_bound_l,
            "tank-stream",
            "tank-boundary",
        ),
        delivered_outlet: meter(
            input["outlet_meter_volume_l"].as_f64().unwrap(),
            per_term_bound_l,
            "outlet-stream",
            "outlet-boundary",
        ),
        purge_and_overflow: meter(
            input["purge_and_overflow_meter_volume_l"].as_f64().unwrap(),
            per_term_bound_l,
            "purge-stream",
            "purge-boundary",
        ),
        topology_audit: Some(TopologyAuditReference {
            report_sha256: "b".repeat(64),
        }),
    };
    let encoded_input =
        serde_json::to_vec(&input).expect("typed evidence input must serialize deterministically");
    let decoded_input: WaterBalanceInput = serde_json::from_slice(&encoded_input)
        .expect("typed evidence input must deserialize without changing its contract");
    let result = evaluate_water_balance(&decoded_input)
        .expect("synthetic known-answer input must be evaluable");

    assert_eq!(result.residual_l, -200.0);
    assert_eq!(result.residual_lower_l, -220.0);
    assert_eq!(result.residual_upper_l, -180.0);
    assert_eq!(result.aggregate_absolute_error_bound_l, 20.0);
    assert_eq!(
        result.disposition,
        WaterBalanceDisposition::ContradictionBeyondDeclaredBound
    );
    let receipt = serde_json::to_value(&result)
        .expect("evaluation result must serialize as a machine-readable receipt");
    assert_eq!(
        receipt["disposition"],
        "contradiction_beyond_declared_bound"
    );
    assert_eq!(receipt["residual_l"], -200.0);
    assert_eq!(receipt["residual_lower_l"], -220.0);
    assert_eq!(receipt["residual_upper_l"], -180.0);
    assert_eq!(receipt["aggregate_absolute_error_bound_l"], 20.0);
}
