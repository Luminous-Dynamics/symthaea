use mel_rmi_synthetic::{
    ActionMechanism, AgentAction, AgentPolicy, ArtifactFormatV1, ArtifactPayloadSetV1,
    ArtifactPayloadV1, Dimension, Feature, FixedResponsePolicy, HarnessError,
    ImitationOnlyPolicy, Observation, OneWayPromptPolicy, ResponsivePartnerPolicy, RuleBasedPolicy,
    TaskSpec, TrialRecordV1, TrialRunConfig, UnrelatedFeatureShortcutPolicy,
    bind_verified_artifacts, canonical_fixture_replicates_v1, run_policy_suite_v1,
    run_reciprocal_suite_v1, verify_trial_artifact_bindings,
};
use std::error::Error;

struct YokedSuitePolicy {
    source_session_id: String,
}

impl AgentPolicy for YokedSuitePolicy {
    fn policy_id(&self) -> &str {
        "yoked"
    }

    fn fork_from_frozen_state(&self, _common_random_seed: u64) -> Box<dyn AgentPolicy> {
        Box::new(Self {
            source_session_id: self.source_session_id.clone(),
        })
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        if self.source_session_id.trim().is_empty()
            || self.source_session_id == observation.session_id
        {
            return Err(HarnessError::InvalidYoke);
        }
        let value = match task.target_dimension {
            Dimension::Motif => 9,
            Dimension::Rhythm => 3,
            Dimension::Tempo => 8,
            Dimension::Role => 2,
            Dimension::Texture => 4,
        };
        Ok(AgentAction {
            event_id: format!("{}::yoked::action", observation.event_id),
            source_event_id: observation.event_id.clone(),
            feature: Feature {
                dimension: task.target_dimension,
                value,
            },
            mechanism: ActionMechanism::Yoked,
        })
    }
}

fn bind_and_verify_synthetic_artifacts(record: &mut TrialRecordV1) -> Result<(), HarnessError> {
    let inputs = std::array::from_fn(|index| {
        let event_id = record.manifest.input_event_ids[index].clone();
        ArtifactPayloadSetV1 {
            event_id: event_id.clone(),
            transaction_receipt_bytes: format!("synthetic-studio-receipt:input:{event_id}")
                .into_bytes(),
            artifacts: vec![
                ArtifactPayloadV1 {
                    format: ArtifactFormatV1::ScoreJson,
                    bytes: format!(r#"{{"event":"{event_id}","role":"input"}}"#).into_bytes(),
                },
                ArtifactPayloadV1 {
                    format: ArtifactFormatV1::Midi,
                    bytes: [b"MThd".as_slice(), event_id.as_bytes()].concat(),
                },
            ],
        }
    });
    let outputs = std::array::from_fn(|index| {
        let event_id = record.manifest.output_event_ids[index].clone();
        ArtifactPayloadSetV1 {
            event_id: event_id.clone(),
            transaction_receipt_bytes: format!("synthetic-studio-receipt:output:{event_id}")
                .into_bytes(),
            artifacts: vec![
                ArtifactPayloadV1 {
                    format: ArtifactFormatV1::ScoreJson,
                    bytes: format!(r#"{{"event":"{event_id}","role":"output"}}"#).into_bytes(),
                },
                ArtifactPayloadV1 {
                    format: ArtifactFormatV1::Midi,
                    bytes: [b"MThd".as_slice(), event_id.as_bytes()].concat(),
                },
            ],
        }
    });

    bind_verified_artifacts(
        record,
        inputs.clone(),
        outputs.clone(),
        "synthetic-byte-verifier",
        "v1",
    )?;
    verify_trial_artifact_bindings(record, &inputs, &outputs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let code_revision = std::env::var("MEL_RMI_CODE_REVISION")?;
    let config = TrialRunConfig {
        policy_version: "canonical-fixture-policy-v1".to_owned(),
        code_revision,
    };
    let fixtures = canonical_fixture_replicates_v1(32);
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
        Box::new(YokedSuitePolicy {
            source_session_id: "prior-session".to_owned(),
        }),
        Box::new(UnrelatedFeatureShortcutPolicy {
            id: "distractor-shortcut".to_owned(),
        }),
    ];

    println!(
        "record_type,task_id,policy_id,dimension,agent_targeted_delta,agent_sham_delta,agent_unrelated_delta,agent_input_sensitive,agent_target_specific,agent_oracle_rule_match,agent_qualified,partner_targeted_delta,partner_sham_delta,partner_unrelated_delta,partner_input_sensitive,partner_target_specific,partner_oracle_rule_match,reciprocal_qualified"
    );

    let mut artifact_bindings_verified = 0usize;
    for policy in policies {
        let mut records = run_policy_suite_v1(&fixtures, policy.as_ref(), &config)?;
        for record in &mut records {
            bind_and_verify_synthetic_artifacts(record)?;
            artifact_bindings_verified += 1;
            let contrast = record.contrast;
            println!(
                "trial,{},{},{},{},{},{},{},{},{},{},,,,,,,",
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
    for mut record in reciprocal {
        bind_and_verify_synthetic_artifacts(&mut record.agent)?;
        artifact_bindings_verified += 1;
        let agent = record.agent.contrast;
        let partner = record.partner;
        println!(
            "reciprocal,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
            partner.oracle_rule_match,
            record.qualified_synthetic_reciprocity,
        );
    }

    println!("artifact_verified,{artifact_bindings_verified}");
    Ok(())
}
