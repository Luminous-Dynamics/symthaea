use allostatic_control_bench::{run_episode_with_shield, Policy, Scenario, EPISODE_CSV_HEADER};

fn main() {
    println!("{EPISODE_CSV_HEADER}");

    for scenario in Scenario::all() {
        for seed in 0..32_u64 {
            for policy in Policy::all() {
                for shielded in [false, true] {
                    let m = run_episode_with_shield(policy, scenario, seed, shielded);
                    let recovery = m
                        .recovery_steps
                        .map_or_else(|| "NA".to_string(), |steps| steps.to_string());
                    println!(
                        "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{},{:.6},{},{}",
                        scenario.name(),
                        seed,
                        policy.name(),
                        shielded,
                        m.task_reward,
                        m.cumulative_exposure,
                        m.mean_load,
                        m.peak_load,
                        m.minimum_reserve,
                        m.mean_safe_action_options,
                        m.minimum_safe_action_options,
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
                        m.false_alarm_opportunities,
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
