use mel_rmi_synthetic::{
    canonical_fixture_suite_v1, run_policy_suite_v1, run_reciprocal_suite_v1,
    AgentPolicy, FixedResponsePolicy, ImitationOnlyPolicy, OneWayPromptPolicy,
    ResponsivePartnerPolicy, RuleBasedPolicy, TrialRunConfig, YokedReplayPolicy,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let code_revision = std::env::var("MEL_RMI_CODE_REVISION")?;
    let config = TrialRunConfig {
        policy_version: "canonical-fixture-policy-v1".to_owned(),
        code_revision,
    };
    let fixtures = canonical_fixture_suite_v1();
    let policies: Vec<Box<dyn AgentPolicy>> = vec![
        Box::new(RuleBasedPolicy {
            id: "rule-positive".to_owned(),
        }),
        Box::new(FixedResponsePolicy {
            id: "fixed".to_owned(),
            value: 0,
        }),
        Box::new(ImitationOnlyPolicy {
            id: "imitation".to_owned(),
        }),
        Box::new(OneWayPromptPolicy {
            id: "one-way".to_owned(),
        }),
        Box::new(YokedReplayPolicy {
            id: "yoked".to_owned(),
            source_session_id: "prior-session".to_owned(),
            replayed_feature: mel_rmi_synthetic::Feature {
                dimension: mel_rmi_synthetic::Dimension::Motif,
                value: 9,
            },
        }),
    ];

    println!(
        "record_type,task_id,policy_id,dimension,agent_targeted_delta,agent_sham_delta,agent_unrelated_delta,agent_input_sensitive,agent_target_specific,agent_oracle_rule_match,agent_qualified,partner_targeted_delta,partner_sham_delta,partner_unrelated_delta,partner_input_sensitive,partner_target_specific,reciprocal_qualified"
    );

    for policy in policies {
        let records = run_policy_suite_v1(&fixtures, policy.as_ref(), &config)?;
        for record in records {
            let contrast = record.contrast;
            println!(
                "trial,{},{},{},{},{},{},{},{},{},{},,,,,,",
                record.task.task_id,
                policy.policy_id(),
                record.task.target_dimension.as_str(),
                contrast.targeted_delta,
                contrast.sham_delta,
                contrast.unrelated_target_delta,
                contrast.input_sensitive,
                contrast.target_specific,
                contrast.oracle_rule_match,
                contrast.qualified_synthetic_contingency,
            );
        }
    }

    let reciprocal = run_reciprocal_suite_v1(
        &fixtures,
        &RuleBasedPolicy {
            id: "rule-positive".to_owned(),
        },
        &ResponsivePartnerPolicy {
            id: "responsive-partner".to_owned(),
        },
        &config,
    )?;
    for record in reciprocal {
        let agent = record.agent.contrast;
        let partner = record.partner;
        println!(
            "reciprocal,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            record.agent.task.task_id,
            record.agent.manifest.policy_id,
            record.agent.task.target_dimension.as_str(),
            agent.targeted_delta,
            agent.sham_delta,
            agent.unrelated_target_delta,
            agent.input_sensitive,
            agent.target_specific,
            agent.oracle_rule_match,
            agent.qualified_synthetic_contingency,
            partner.targeted_delta,
            partner.sham_delta,
            partner.unrelated_delta,
            partner.input_sensitive,
            partner.target_specific,
            record.qualified_synthetic_reciprocity,
        );
    }

    Ok(())
}
