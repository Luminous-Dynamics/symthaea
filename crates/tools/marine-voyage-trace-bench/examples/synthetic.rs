use marine_voyage_trace_bench::{
    assess_trace, PowerReading, PowerSample, TraceOutcome, TraceSpec,
};

fn main() {
    let samples = [0_u64, 1_800_000, 3_600_000]
        .into_iter()
        .map(|elapsed_ms| PowerSample {
            elapsed_ms,
            propulsion: Some(PowerReading {
                nominal_w: 800.0,
                absolute_bound_w: 8.0,
            }),
            auxiliary: Some(PowerReading {
                nominal_w: 200.0,
                absolute_bound_w: 2.0,
            }),
        })
        .collect();

    let trace = TraceSpec {
        trace_id: "synthetic-voyage-001".to_owned(),
        raw_trace_sha256: "a".repeat(64),
        mission_id: "synthetic-mission-A".to_owned(),
        condition_digest: "synthetic-condition-digest".to_owned(),
        service_digest: "b".repeat(64),
        calibration_epoch_id: "synthetic-meter-calibration-epoch".to_owned(),
        energy_boundary_id: "synthetic-dc-input-boundary".to_owned(),
        uncertainty_method_id: "synthetic-absolute-bound-v1".to_owned(),
        expected_start_ms: 0,
        expected_end_ms: 3_600_000,
        max_gap_ms: 1_800_000,
        samples,
    };

    println!("trace_id,disposition,propulsion_wh,auxiliary_wh,total_lower_wh,total_nominal_wh,total_upper_wh,sample_count,max_gap_ms");
    match assess_trace(&trace) {
        TraceOutcome::Complete(energy) => println!(
            "{},COMPLETE,{:.3},{:.3},{:.3},{:.3},{:.3},{},{}",
            energy.trace_id,
            energy.propulsion_wh.nominal_wh,
            energy.auxiliary_wh.nominal_wh,
            energy.total_wh.lower_wh,
            energy.total_wh.nominal_wh,
            energy.total_wh.upper_wh,
            energy.sample_count,
            energy.max_observed_gap_ms
        ),
        TraceOutcome::Indeterminate(issues) => {
            println!("{},INDETERMINATE,,,,,,,{}", trace.trace_id, issues.len());
        }
        TraceOutcome::Invalid(issues) => {
            println!("{},INVALID,,,,,,,{}", trace.trace_id, issues.len());
        }
    }
}
