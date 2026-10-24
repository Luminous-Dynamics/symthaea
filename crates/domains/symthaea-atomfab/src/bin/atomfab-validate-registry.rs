use std::process::ExitCode;

use symthaea_atomfab::benchmark_registry::load_embedded_registry;

fn main() -> ExitCode {
    let registry = match load_embedded_registry() {
        Ok(registry) => registry,
        Err(error) => {
            eprintln!("FAIL: cannot parse AtomFab benchmark registry: {error}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(error) = registry.validate() {
        eprintln!("FAIL: AtomFab benchmark registry validation failed: {error}");
        return ExitCode::FAILURE;
    }

    println!(
        "PASS: {} schema v{}; {} benchmark lanes; {} policy baselines; status={}",
        registry.registry_id,
        registry.schema_version,
        registry.benchmark_lanes.len(),
        registry.policy_baselines.len(),
        registry.status
    );
    for lane in &registry.benchmark_lanes {
        println!("- {} [{}]: {}", lane.id, lane.status, lane.reference.name);
    }
    println!(
        "Performance gates: median AF >= {:.2} across >= {} fixed-seed campaigns; max false-feasible accepts = {}",
        registry
            .performance_claims
            .accelerations
            .stretch_target_median_af,
        registry.performance_claims.accelerations.minimum_campaigns,
        registry
            .performance_claims
            .process_feasibility
            .maximum_false_feasible_accepts_on_frozen_adversarial_suite
    );

    ExitCode::SUCCESS
}
