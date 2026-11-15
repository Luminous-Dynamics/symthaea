use fusion_search_benchmark::{evaluate_funnel, summarize, Policy, Scenario};

fn display_metric(value: Option<f64>) -> String {
    value
        .map(|metric| format!("{metric:.3}"))
        .unwrap_or_else(|| "unavailable".to_string())
}

fn display_regret(value: Option<i32>) -> String {
    value
        .map(|regret| regret.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn main() {
    println!(
        "scenario,policy,universe_coverage,records,screened,hf_attempts,feasible_recall,pareto_recall,usable_pareto_recall,best_objective_regret,hf_failed,hf_nonconverged,hf_unsupported,hf_inconclusive"
    );

    for scenario in [Scenario::Smooth, Scenario::RareFeasibleIsland] {
        for policy in [
            Policy::Exhaustive,
            Policy::LowFidelityThreshold,
            Policy::UncertaintyGuarded,
        ] {
            let records = evaluate_funnel(scenario, policy);
            let scorecard = summarize(&records, true);
            println!(
                "{},{},{:.3},{},{},{},{},{},{},{},{},{},{},{}",
                scenario.name(),
                policy.name(),
                scorecard.candidate_universe_coverage,
                scorecard.generated_records,
                scorecard.screened_candidates,
                scorecard.high_fidelity_attempts,
                display_metric(scorecard.feasible_recall),
                display_metric(scorecard.pareto_recall),
                display_metric(scorecard.usable_pareto_recall),
                display_regret(scorecard.best_objective_regret),
                scorecard.high_fidelity_failed,
                scorecard.high_fidelity_non_converged,
                scorecard.high_fidelity_unsupported,
                scorecard.high_fidelity_inconclusive,
            );
        }
    }
}
