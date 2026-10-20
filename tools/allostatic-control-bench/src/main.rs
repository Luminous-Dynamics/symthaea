use allostatic_control_bench::{run_episode, Policy, Scenario};

fn main() {
    println!(
        "scenario,seed,policy,task_reward,cumulative_exposure,mean_load,peak_load,unsafe_ticks,missed_limits,false_alarms,push_steps,maintain_steps,resource_cost,failed,recovery_steps"
    );

    for scenario in Scenario::all() {
        for seed in 0..32_u64 {
            for policy in Policy::all() {
                let m = run_episode(policy, scenario, seed);
                let recovery = m
                    .recovery_steps
                    .map_or_else(|| "NA".to_string(), |steps| steps.to_string());
                println!(
                    "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{},{}",
                    scenario.name(),
                    seed,
                    policy.name(),
                    m.task_reward,
                    m.cumulative_exposure,
                    m.mean_load,
                    m.peak_load,
                    m.unsafe_ticks,
                    m.missed_limits,
                    m.false_alarms,
                    m.push_steps,
                    m.maintain_steps,
                    m.resource_cost,
                    m.failed,
                    recovery
                );
            }
        }
    }
}
