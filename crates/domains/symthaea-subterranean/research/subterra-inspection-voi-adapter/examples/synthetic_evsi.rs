use subterra_inspection_voi_adapter::{
    InspectionTask, evaluate_inspection_case, synthetic_decision_boundary_case,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for task in [
        InspectionTask::PipeCable,
        InspectionTask::DrawpointVolume,
        InspectionTask::SupportVerification,
    ] {
        let case = synthetic_decision_boundary_case(task);
        let report = evaluate_inspection_case(&case)?;
        let high = report
            .myopic_action_baselines
            .iter()
            .max_by(|a, b| {
                a.expected_information_gain_nats
                    .total_cmp(&b.expected_information_gain_nats)
            })
            .expect("two synthetic candidate actions");
        let decision_valuable = report
            .myopic_action_baselines
            .iter()
            .max_by(|a, b| a.gross_evsi.total_cmp(&b.gross_evsi))
            .expect("two synthetic candidate actions");

        println!("task={}", task.as_str());
        println!(
            "  no_test={} risk={:.4}",
            report.no_test.decision_id, report.no_test.expected_decision_loss
        );
        println!(
            "  entropy_greedy={} eig_nats={:.4} gross_evsi={:.4}",
            high.action_id, high.expected_information_gain_nats, high.gross_evsi
        );
        println!(
            "  myopic_evsi={} eig_nats={:.4} gross_evsi={:.4}",
            decision_valuable.action_id,
            decision_valuable.expected_information_gain_nats,
            decision_valuable.gross_evsi
        );
        println!(
            "  planner_selected={:?} planner_gross_evsi={:.4}",
            report.planner.policy.measurement_action_id, report.planner_gross_evsi
        );
        println!(
            "  robust_models={} worst_case_value={:.4} admissible={}",
            report.robust.model_count,
            report.robust.worst_case_expected_total_value,
            report.robust_policy_admissible
        );
        println!(
            "  expected_resources energy_j={:.2} duration_s={:.2} data_bytes={:.2} exposure={:.4}",
            report.planner.metrics.expected_resources.energy_j,
            report.planner.metrics.expected_resources.duration_s,
            report.planner.metrics.expected_resources.data_bytes,
            report.planner.metrics.expected_resources.exposure_units
        );
    }
    Ok(())
}
