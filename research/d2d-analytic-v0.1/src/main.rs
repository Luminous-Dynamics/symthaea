use d2d_analytic::{
    AgentModel, BranchDisposition, BranchRecord, EvaluatorWorld, ExperimentDesign, Hypothesis,
    PredictionSeal, audit_expected_branches, evaluate_sealed_prediction,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = AgentModel::new(
        "demo-hypothesis-set-v1",
        vec![
            Hypothesis::new("low-friction", 0.1)?,
            Hypothesis::new("high-friction", 0.8)?,
        ],
    )?;

    let candidates = [1, 2, 5, 10]
        .into_iter()
        .map(|steps| {
            ExperimentDesign::new(
                format!("coast-{steps}"),
                1.0,
                0.0,
                0.1,
                steps,
                0.1,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let selection = model.select_experiment(&candidates, 5)?;
    println!(
        "selected={} steps={} discrimination_score_per_cost={:.6}",
        selection.experiment.id(),
        selection.experiment.steps(),
        selection.discrimination_score_per_cost
    );

    // Sequence 10 seals predictions; sequence 11 performs the hidden-truth run.
    let seal = PredictionSeal::create(&model, &selection.experiment, 10)?;
    println!("prediction_seal_fingerprint={:016x}", seal.fingerprint());
    for prediction in seal.predictions() {
        println!(
            "predicted[{}]={:.9}",
            prediction.hypothesis_id(),
            prediction.terminal_velocity()
        );
    }

    // Evaluator truth is kept behind a separate API. In this demo it shares a
    // process for simplicity; that is explicitly not a security boundary.
    let evaluator = EvaluatorWorld::from_known_truth(0.8)?;
    let execution = evaluator.execute(&selection.experiment, 11)?;
    let result = evaluate_sealed_prediction(&seal, &execution)?;
    println!(
        "observed_terminal_velocity={:.9}",
        result.observed_terminal_velocity
    );
    println!(
        "inferred_hypothesis={} best_absolute_error={:.12}",
        result.inferred_hypothesis,
        result.best_absolute_error
    );

    // Show that missing/unsupported branches stay in the denominator.
    let expected = vec![
        "branch-1".to_string(),
        "branch-2".to_string(),
        "branch-3".to_string(),
    ];
    let observed = vec![
        BranchRecord::new("branch-1", BranchDisposition::Completed),
        BranchRecord::new("branch-2", BranchDisposition::Unsupported),
    ];
    let audit = audit_expected_branches(&expected, &observed);
    println!(
        "branch_accounting expected={} reported={} completed={} non_completed={} missing={:?}",
        audit.expected_count,
        audit.reported_count,
        audit.completed_count,
        audit.non_completed_count,
        audit.missing
    );

    println!("claim_boundary=analytic_fixture_only; not_symtropy_execution; not_a_cryptographic_commitment");
    Ok(())
}
