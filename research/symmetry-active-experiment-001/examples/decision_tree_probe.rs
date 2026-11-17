use symmetry_active_experiment_reference::{
    ActiveExperimentProblemV1, ExperimentProbeV1, QualificationBudgetV1, compile_active_experiment,
};

fn main() {
    let problem = ActiveExperimentProblemV1 {
        problem_id: "four-symmetry-hypotheses-v0.1".to_owned(),
        hypotheses: vec![
            "C2".to_owned(),
            "Cs".to_owned(),
            "D2".to_owned(),
            "C2v".to_owned(),
        ],
        probes: vec![
            ExperimentProbeV1 {
                probe_id: "orientation-probe".to_owned(),
                cost: 1,
                admissible: true,
                outcomes: vec![0, 0, 1, 1],
            },
            ExperimentProbeV1 {
                probe_id: "boundary-probe".to_owned(),
                cost: 1,
                admissible: true,
                outcomes: vec![0, 1, 1, 1],
            },
            ExperimentProbeV1 {
                probe_id: "defect-localization-probe".to_owned(),
                cost: 1,
                admissible: true,
                outcomes: vec![0, 0, 0, 1],
            },
        ],
    };
    let certificate =
        compile_active_experiment(&problem, QualificationBudgetV1::default()).unwrap();
    println!(
        "problem={}, hypotheses={}, probes={}, worst_case_cost={}, explored_subsets={}, nodes={}",
        certificate.problem_id,
        certificate.hypothesis_count,
        certificate.probe_count,
        certificate.root_worst_case_cost,
        certificate.explored_subsets,
        certificate.node_count
    );
}
