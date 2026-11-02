use symthaea_dishwasher_scheduler::{
    generate_workload, illustrative_profiles, simulate, DemandPattern, Policy,
};

fn main() {
    let (quick, full) = illustrative_profiles();
    let scenarios = [
        ("sparse", DemandPattern::SparseHousehold),
        ("meal_bursts", DemandPattern::MealBursts),
        ("irregular", DemandPattern::Irregular),
    ];
    let policies = [
        ("full_load_only", Policy::FullLoadOnly { max_batch_wait_s: 10_800 }),
        ("quickcell_only", Policy::QuickCellOnly),
        ("hybrid_resource_aware", Policy::HybridResourceAware { max_batch_wait_s: 10_800 }),
    ];

    println!("scenario,policy,items,unprocessed,energy_kwh,fresh_water_l,mean_wait_min,p95_wait_min,mean_latency_min,p95_latency_min,deadline_misses,quick_cycles,full_cycles");
    for (scenario_name, pattern) in scenarios {
        let jobs = generate_workload(42, 100, pattern);
        for (policy_name, policy) in policies {
            match simulate(&jobs, &quick, &full, policy) {
                Ok(result) => println!(
                    "{},{},{},{},{:.6},{:.3},{:.2},{:.2},{:.2},{:.2},{},{},{}",
                    scenario_name,
                    policy_name,
                    result.completed_items(),
                    result.unprocessed_job_ids.len(),
                    result.total_energy_kwh,
                    result.total_fresh_water_l,
                    result.mean_wait_s / 60.0,
                    result.p95_wait_minutes(),
                    result.mean_latency_s / 60.0,
                    result.p95_latency_s as f64 / 60.0,
                    result.deadline_misses,
                    result.quick_cell_cycles,
                    result.full_load_cycles
                ),
                Err(error) => {
                    eprintln!("{scenario_name},{policy_name}: simulation rejected: {error}");
                    std::process::exit(1);
                }
            }
        }
    }
}
