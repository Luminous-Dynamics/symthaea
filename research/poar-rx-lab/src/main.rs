use pod_meta_lab::recovery::DispatchSemantics;
use poar_rx_lab::{run_trial, ControllerPolicy, FaultSchedule};

fn main() {
    println!(
        "policy\tfault\tdispatch_semantics\toutcome\tdispatch_attempts\treadback_calls\tactual_effects\treported_complete\tpostcondition_verified\tduplicate_effect\tfalse_completion\tunverified_completion_claim"
    );

    for fault in FaultSchedule::ALL {
        for semantics in [
            DispatchSemantics::NonIdempotent,
            DispatchSemantics::IdempotencyKeyEnforced,
        ] {
            for policy in ControllerPolicy::ALL {
                let result = run_trial(policy, fault, semantics);
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    result.policy.as_str(),
                    result.fault.as_str(),
                    match result.dispatch_semantics {
                        DispatchSemantics::NonIdempotent => "non_idempotent",
                        DispatchSemantics::IdempotencyKeyEnforced => "idempotency_key_enforced",
                    },
                    result.outcome.as_str(),
                    result.dispatch_attempts,
                    result.readback_calls,
                    result.actual_effects,
                    result.reported_complete,
                    result.postcondition_verified,
                    result.duplicate_effect,
                    result.false_completion,
                    result.unverified_completion_claim,
                );
            }
        }
    }
}
