use allostatic_control_bench::{run_episode_with_shield, Policy, Scenario};

fn main() {
    println!(
        "scenario,seed,policy,shielded,task_reward,cumulative_exposure,mean_load,peak_load,minimum_reserve,terminal_reserve,terminal_load,unsafe_ticks,unsafe_episode,violation_exposure,max_violation_severity,missed_limits,shield_vetoes,no_safe_action_ticks,false_alarms,push_steps,maintain_steps,emergency_recover_steps,resource_cost,failed,recovery_steps"
    );

    for scenario in Scenario::all() {
        for seed in 0..32_u64 {
            for policy in Policy::all() {
                for shielded in [false, true] {
                    let m = run_episode_with_shield(policy, scenario, seed, shielded);
                    let recovery = m
                        .recovery_steps
                        .map_or_else(|| "NA".to_string(), |steps| steps.to_string());
                    println!(
                        "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{:.6},{},{}",
                        scenario.name(),
                        seed,
                        policy.name(),
                        shielded,
                        m.task_reward,
                        m.cumulative_exposure,
                        m.mean_load,
                        m.peak_load,
                        m.minimum_reserve,
                        m.terminal_reserve,
                        m.terminal_load,
                        m.unsafe_ticks,
                        m.unsafe_episode,
                        m.violation_exposure,
                        m.max_violation_severity,
                        m.missed_limits,
                        m.shield_vetoes,
                        m.no_safe_action_ticks,
                        m.false_alarms,
                        m.push_steps,
                        m.maintain_steps,
                        m.emergency_recover_steps,
                        m.resource_cost,
                        m.failed,
                        recovery
                    );
                }
            }
        }
    }
}
