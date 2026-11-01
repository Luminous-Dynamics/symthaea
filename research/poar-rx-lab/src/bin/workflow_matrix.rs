use poar_rx_lab::workflow::{WorkflowFault, WorkflowPolicy, run_workflow};

fn main() {
    println!(
        "policy\tfault\tcompensation_authorized\tterminal\tfirst_target\tsecond_target\tfirst_agent_writes\tsecond_agent_writes\tcompensation_writes\texternal_writes\treadback_calls\tunauthorized_effects\tlost_concurrent_updates\treported_complete\tgoal_state_holds\tcompletion_verified\tcompensation_verified\tduplicate_effect\tfalse_completion\tcompensation_failed"
    );

    for fault in WorkflowFault::ALL {
        for compensation_authorized in [false, true] {
            for policy in WorkflowPolicy::ALL {
                let result = run_workflow(policy, fault, compensation_authorized);
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    result.policy.as_str(),
                    result.fault.as_str(),
                    result.compensation_authorized,
                    result.terminal.as_str(),
                    result.first_target,
                    result.second_target,
                    result.first_agent_writes,
                    result.second_agent_writes,
                    result.compensation_writes,
                    result.external_writes,
                    result.readback_calls,
                    result.unauthorized_effects,
                    result.lost_concurrent_updates,
                    result.reported_complete,
                    result.goal_state_holds,
                    result.completion_verified,
                    result.compensation_verified,
                    result.duplicate_effect,
                    result.false_completion,
                    result.compensation_failed,
                );
            }
        }
    }
}
