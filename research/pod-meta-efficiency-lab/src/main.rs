use pod_meta_efficiency_lab::{run_matrix, Policy};

fn main() {
    println!(
        "task_id\tseed\tpolicy\tbudget\tspent\toperations\tscheduler_candidate_rejections\tverified_obligations\trequired_obligations\treported_complete\tcompletion_verified\tfalse_completion\tfalse_support_count\tmandatory_required_mask\tmandatory_verified_mask\tmissed_mandatory_count\trejected_candidates\trejected_candidate_cost\tredundant_operations\twasted_operation_cost\tcorrect_abstention\tcost_to_first_sufficient_proof\toracle_min_cost\tcost_regret"
    );

    for row in run_matrix(1729) {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            row.task_id,
            row.seed,
            row.policy.as_str(),
            row.budget,
            row.spent,
            row.operation_count,
            row.scheduler_candidate_rejections,
            row.verified_obligations,
            row.required_obligations,
            row.reported_complete,
            row.completion_verified,
            row.false_completion,
            row.false_support_count,
            row.mandatory_required_mask,
            row.mandatory_verified_mask,
            row.missed_mandatory_count,
            row.rejected_candidate_count,
            row.rejected_candidate_cost,
            row.redundant_operation_count,
            row.wasted_operation_cost,
            row.correct_abstention,
            row.cost_to_first_sufficient_proof.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            row.oracle_min_cost.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            row.cost_regret.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
        );
    }
}
