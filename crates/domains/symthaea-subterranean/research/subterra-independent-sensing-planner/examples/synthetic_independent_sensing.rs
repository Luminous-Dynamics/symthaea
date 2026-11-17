use subterra_independent_sensing_planner::{
    AcquisitionBudget, AcquisitionPolicy, ExpectedAcquisitionContext, PlannedAcquisitionOption,
    VerifierDependencyClosure, plan_independent_acquisition_portfolio,
};
use subterra_corroboration_assurance::{DependencyIdentity, InventoryDisposition};
use subterra_inspection_campaign::ResourceVector;

fn dep(dimension: &str, identity: &str) -> DependencyIdentity {
    DependencyIdentity { dimension: dimension.into(), identity: identity.into() }
}

fn option(id: &str, source: &str, deps: Vec<DependencyIdentity>, energy: u64) -> PlannedAcquisitionOption {
    PlannedAcquisitionOption {
        option_id: id.into(),
        source_id: source.into(),
        claim_id: "pipe-claim-7".into(),
        asset_id: "pipe-7".into(),
        asset_revision: "rev-12".into(),
        coordinate_frame_digest: "sha256:frame".into(),
        region_scope_digest: "sha256:region".into(),
        deployment_bundle_id: "north-corridor".into(),
        deployment_bundle_digest: "sha256:bundle".into(),
        dependency_graph_digest: "sha256:graph".into(),
        fault_domain_profile_digest: "sha256:fault-profile".into(),
        dependency_assessment_digest: "sha256:assessment".into(),
        inventory: InventoryDisposition::CompleteWithinDeclaredScope,
        claim_critical_dependencies: deps,
        resource_model_digest: "sha256:resource-model".into(),
        resource_assessment_digest: format!("sha256:resource-{id}"),
        risk_model_digest: "sha256:risk-model".into(),
        risk_assessment_digest: format!("sha256:risk-{id}"),
        resource_assessment_valid_until_unix_ms: 1_800_000_600_000,
        risk_assessment_valid_until_unix_ms: 1_800_000_600_000,
        resources: ResourceVector {
            energy_millijoules: energy,
            duration_milliseconds: 20_000,
            data_bytes: 1_024,
            exposure_milliunits: 10,
        },
        mission_interruption_upper_ppm: 100,
        robot_loss_upper_ppm: 10,
    }
}

fn main() {
    let expected = ExpectedAcquisitionContext {
        claim_id: "pipe-claim-7".into(),
        asset_id: "pipe-7".into(),
        asset_revision: "rev-12".into(),
        coordinate_frame_digest: "sha256:frame".into(),
        region_scope_digest: "sha256:region".into(),
        dependency_graph_digest: "sha256:graph".into(),
        fault_domain_profile_digest: "sha256:fault-profile".into(),
        dependency_assessment_digest: "sha256:assessment".into(),
    };
    let verifier = VerifierDependencyClosure {
        verifier_id: "independent-review-node".into(),
        inventory: InventoryDisposition::CompleteWithinDeclaredScope,
        claim_critical_dependencies: vec![
            dep("compute", "review-host"),
            dep("clock", "review-clock"),
            dep("ground-truth", "sealed-metrology"),
        ],
    };
    let options = vec![
        option("rgb-camera-a", "robot-a-camera", vec![
            dep("robot", "robot-a"),
            dep("power", "battery-bus-a"),
            dep("clock", "clock-a"),
            dep("map", "map-a"),
        ], 5_000),
        option("depth-a", "robot-a-depth", vec![
            dep("robot", "robot-a"),
            dep("power", "battery-bus-a"),
            dep("clock", "clock-a"),
            dep("map", "map-a"),
        ], 4_000),
        option("thermal-b", "robot-b-thermal", vec![
            dep("robot", "robot-b"),
            dep("power", "battery-bus-b"),
            dep("clock", "clock-b"),
            dep("map", "map-b"),
        ], 7_000),
    ];
    let report = plan_independent_acquisition_portfolio(
        options,
        &expected,
        &verifier,
        AcquisitionPolicy {
            required_independent_sources: 2,
            evaluation_time_unix_ms: 1_800_000_000_000,
            maximum_resource_assessment_age_ms: 3_600_000,
            maximum_risk_assessment_age_ms: 3_600_000,
            maximum_future_clock_skew_ms: 5_000,
            maximum_search_nodes: 10_000,
        },
        &AcquisitionBudget {
            resources: ResourceVector {
                energy_millijoules: 20_000,
                duration_milliseconds: 100_000,
                data_bytes: 100_000,
                exposure_milliunits: 100_000,
            },
            mission_interruption_upper_ppm: 500,
            robot_loss_upper_ppm: 100,
        },
    ).expect("complete verifier dependency closure required");

    println!("frontier_count={}", report.pareto_frontier.len());
    println!("conflict_count={}", report.source_conflicts.len());
    println!("explored_nodes={}", report.explored_nodes);
    println!("search_disposition={:?}", report.disposition);
    for (index, plan) in report.pareto_frontier.iter().enumerate() {
        println!("portfolio_{}={:?}", index, plan.selected_option_ids);
        println!("portfolio_{}_energy_millijoules={}", index, plan.resources.energy_millijoules);
        println!("portfolio_{}_robot_loss_upper_ppm={}", index, plan.robot_loss_upper_ppm);
    }
    println!("joint_evsi_not_summed=true");
    println!("synthetic_dependencies=true; external Mycelix signatures were not authenticated");
}
