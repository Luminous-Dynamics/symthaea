use mel_rmi_synthetic::{
    ActionMechanism, AgentAction, AgentPolicy, ArtifactFormatV1, ArtifactPayloadSetV1,
    ArtifactPayloadV1, ArtifactSetEvidenceV1, Dimension, Feature, FixedResponsePolicy,
    HarnessError, ImitationOnlyPolicy, NegotiatedResponseContractV1, NegotiationChoiceV1,
    Observation, OneWayPromptPolicy, PartnerAuthorityEventV1, PartnerAuthorityV1,
    ReciprocalTrialRecordV1, ResponsivePartnerPolicy, RuleBasedPolicy, TaskSpec, TrialRecordV1,
    TrialRunConfig, UnrelatedFeatureShortcutPolicy, assess_negotiation, assess_scoped_negotiation,
    bind_verified_artifacts, canonical_fixture_replicates_v1, run_policy_suite_v1,
    run_reciprocal_suite_v1, validate_partner_authority_event, verify_trial_artifact_bindings,
};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

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

fn synthetic_midi_bytes(marker: &[u8]) -> Vec<u8> {
    let mut track = vec![0x00, 0xff, 0x01, marker.len() as u8];
    track.extend_from_slice(marker);
    track.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);
    let mut bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk".to_vec();
    bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
    bytes.extend(track);
    bytes
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
                    bytes: synthetic_midi_bytes(event_id.as_bytes()),
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
                    bytes: synthetic_midi_bytes(event_id.as_bytes()),
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

fn csv_cell(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn write_artifact_set_evidence<W: Write>(
    writer: &mut W,
    task_id: &str,
    session_id: &str,
    task_seed: u64,
    randomization_seed: u64,
    policy_id: &str,
    policy_version: &str,
    code_revision: &str,
    metric_version: &str,
    verifier_id: &str,
    verifier_version: &str,
    dimension: Dimension,
    direction: &str,
    condition: &str,
    evidence: &ArtifactSetEvidenceV1,
) -> std::io::Result<usize> {
    let mut rows = 0;
    for artifact in &evidence.artifacts {
        let format = match artifact.format {
            ArtifactFormatV1::ScoreJson => "score_json",
            ArtifactFormatV1::Midi => "midi",
            ArtifactFormatV1::Wav => "wav",
            ArtifactFormatV1::PcmF32Le => "pcm_f32_le",
        };
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv_cell(task_id),
            csv_cell(session_id),
            task_seed,
            randomization_seed,
            csv_cell(policy_id),
            csv_cell(policy_version),
            code_revision,
            metric_version,
            dimension.as_str(),
            direction,
            condition,
            csv_cell(&evidence.event_id),
            evidence.transaction_receipt_sha256,
            format,
            artifact.sha256_hex,
            artifact.byte_len,
            csv_cell(verifier_id),
            csv_cell(verifier_version),
        )?;
        rows += 1;
    }
    Ok(rows)
}

fn write_record_artifact_evidence<W: Write>(
    writer: &mut W,
    record: &TrialRecordV1,
) -> Result<usize, Box<dyn Error>> {
    let bindings = record
        .artifact_bindings
        .as_ref()
        .ok_or("artifact bindings missing after verifier returned success")?;
    let conditions = ["control", "targeted", "sham", "unrelated"];
    let mut rows = 0;
    for (index, condition) in conditions.iter().enumerate() {
        for (direction, evidence) in [
            ("input", &bindings.inputs[index]),
            ("output", &bindings.outputs[index]),
        ] {
            rows += write_artifact_set_evidence(
                writer,
                &record.task.task_id,
                &record.manifest.session_id,
                record.manifest.task_seed,
                record.manifest.randomization_seed,
                &record.manifest.policy_id,
                &record.manifest.policy_version,
                &record.manifest.code_revision,
                &record.manifest.metric_version,
                &bindings.verifier_id,
                &bindings.verifier_version,
                record.task.target_dimension,
                direction,
                condition,
                evidence,
            )?;
        }
    }
    Ok(rows)
}

fn report_policy_aggregates(records: &[TrialRecordV1]) {
    let mut by_dimension: BTreeMap<Dimension, Vec<&TrialRecordV1>> = BTreeMap::new();
    for record in records {
        by_dimension
            .entry(record.task.target_dimension)
            .or_default()
            .push(record);
    }

    for (dimension, group) in by_dimension {
        let count = group.len() as f64;
        let mean_targeted = group
            .iter()
            .map(|record| f64::from(record.contrast.targeted_delta))
            .sum::<f64>()
            / count;
        let variance = group
            .iter()
            .map(|record| {
                let delta = f64::from(record.contrast.targeted_delta) - mean_targeted;
                delta * delta
            })
            .sum::<f64>()
            / count;
        let mean_sham = group
            .iter()
            .map(|record| f64::from(record.contrast.sham_delta))
            .sum::<f64>()
            / count;
        let mean_unrelated = group
            .iter()
            .map(|record| f64::from(record.contrast.unrelated_target_delta))
            .sum::<f64>()
            / count;
        let input_sensitive = group
            .iter()
            .filter(|record| record.contrast.input_sensitive)
            .count();
        let target_specific = group
            .iter()
            .filter(|record| record.contrast.target_specific)
            .count();
        let oracle_matches = group
            .iter()
            .filter(|record| record.contrast.oracle_rule_match)
            .count();
        let qualified = group
            .iter()
            .filter(|record| record.contrast.qualified_synthetic_contingency)
            .count();
        let max_delta = f64::from((group[0].task.expected_response.modulus / 2).max(1));
        eprintln!(
            "RMI_AGENT_AGGREGATE,policy={},dimension={},n={},mean_targeted={:.4},normalized_targeted={:.4},sd_targeted={:.4},mean_sham={:.4},mean_unrelated={:.4},input_sensitive={},target_specific={},oracle_match={},qualified={}",
            group[0].manifest.policy_id,
            dimension.as_str(),
            group.len(),
            mean_targeted,
            mean_targeted / max_delta,
            variance.sqrt(),
            mean_sham,
            mean_unrelated,
            input_sensitive,
            target_specific,
            oracle_matches,
            qualified,
        );
    }
}

fn report_reciprocal_aggregates(records: &[ReciprocalTrialRecordV1]) {
    let mut by_dimension: BTreeMap<Dimension, Vec<&ReciprocalTrialRecordV1>> = BTreeMap::new();
    for record in records {
        by_dimension
            .entry(record.agent.task.target_dimension)
            .or_default()
            .push(record);
    }

    for (dimension, group) in by_dimension {
        let count = group.len() as f64;
        let mean_targeted = group
            .iter()
            .map(|record| f64::from(record.partner.targeted_delta))
            .sum::<f64>()
            / count;
        let mean_sham = group
            .iter()
            .map(|record| f64::from(record.partner.sham_delta))
            .sum::<f64>()
            / count;
        let mean_unrelated = group
            .iter()
            .map(|record| f64::from(record.partner.unrelated_delta))
            .sum::<f64>()
            / count;
        let specific = group
            .iter()
            .filter(|record| record.partner.target_specific)
            .count();
        let oracle_match = group
            .iter()
            .filter(|record| record.partner.oracle_rule_match)
            .count();
        let reciprocal = group
            .iter()
            .filter(|record| record.qualified_synthetic_reciprocity)
            .count();
        eprintln!(
            "RMI_PARTNER_AGGREGATE,partner=responsive-partner,dimension={},n={},mean_targeted={:.4},mean_sham={:.4},mean_unrelated={:.4},target_specific={},oracle_match={},reciprocal_qualified={}",
            dimension.as_str(),
            group.len(),
            mean_targeted,
            mean_sham,
            mean_unrelated,
            specific,
            oracle_match,
            reciprocal,
        );
    }
}

fn run_negotiation_conformance() -> Result<(usize, usize), Box<dyn Error>> {
    let contract = NegotiatedResponseContractV1 {
        target_dimension: Dimension::Motif,
        modulus: 12,
        accepted_offsets: [3, 5, 7].into_iter().collect(),
        max_branch_count: 3,
    };
    let input = Feature {
        dimension: Dimension::Motif,
        value: 2,
    };
    let cases = [
        (
            "branch_all",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: Default::default(),
            },
            NegotiationChoiceV1::Branch(vec![5, 7, 9]),
        ),
        (
            "ask_unresolved",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: Default::default(),
            },
            NegotiationChoiceV1::AskClarification,
        ),
        (
            "premature_commit",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: Default::default(),
            },
            NegotiationChoiceV1::Commit(7),
        ),
        (
            "branch_after_veto",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: [7].into_iter().collect(),
            },
            NegotiationChoiceV1::Branch(vec![5, 9]),
        ),
        (
            "branch_violates_veto",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: [7].into_iter().collect(),
            },
            NegotiationChoiceV1::Branch(vec![5, 7, 9]),
        ),
        (
            "follow_selected",
            PartnerAuthorityV1 {
                selected_value: Some(9),
                vetoed_values: Default::default(),
            },
            NegotiationChoiceV1::Commit(9),
        ),
        (
            "abstain_empty",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: [5, 7, 9].into_iter().collect(),
            },
            NegotiationChoiceV1::Abstain,
        ),
        (
            "commit_after_all_vetoed",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: [5, 7, 9].into_iter().collect(),
            },
            NegotiationChoiceV1::Commit(5),
        ),
        (
            "incomplete_branch",
            PartnerAuthorityV1 {
                selected_value: None,
                vetoed_values: Default::default(),
            },
            NegotiationChoiceV1::Branch(vec![5, 9]),
        ),
    ];
    let mut qualified = 0usize;
    let mut rejected = 0usize;
    for (name, authority, choice) in cases {
        let assessment = assess_negotiation(&contract, input, &authority, &choice)?;
        if assessment.qualified {
            qualified += 1;
        } else {
            rejected += 1;
        }
        println!(
            "negotiation,{name},{},{},{},{},{},{},{},{}",
            assessment.available_option_count,
            assessment.proposed_option_count,
            assessment.contract_valid,
            assessment.veto_respected,
            assessment.preference_respected,
            assessment.ambiguity_preserved,
            assessment.action_well_formed,
            assessment.qualified,
        );
    }
    let wide_contract = NegotiatedResponseContractV1 {
        target_dimension: Dimension::Motif,
        modulus: 12,
        accepted_offsets: (1..=10).collect(),
        max_branch_count: 3,
    };
    let wide_input = Feature {
        dimension: Dimension::Motif,
        value: 0,
    };
    let wide_authority = PartnerAuthorityV1 {
        selected_value: None,
        vetoed_values: Default::default(),
    };
    let wide_options: Vec<i32> = wide_contract
        .valid_values(wide_input)?
        .into_iter()
        .collect();
    let clarification = assess_negotiation(
        &wide_contract,
        wide_input,
        &wide_authority,
        &NegotiationChoiceV1::AskClarification,
    )?;
    let excessive_branch = assess_negotiation(
        &wide_contract,
        wide_input,
        &wide_authority,
        &NegotiationChoiceV1::Branch(wide_options),
    )?;
    println!(
        "negotiation,branch_explosion_clarify,{},{},{},{},{},{},{},{}",
        clarification.available_option_count,
        clarification.proposed_option_count,
        clarification.contract_valid,
        clarification.veto_respected,
        clarification.preference_respected,
        clarification.ambiguity_preserved,
        clarification.action_well_formed,
        clarification.qualified,
    );
    println!(
        "negotiation,branch_explosion_rejected,{},{},{},{},{},{},{},{}",
        excessive_branch.available_option_count,
        excessive_branch.proposed_option_count,
        excessive_branch.contract_valid,
        excessive_branch.veto_respected,
        excessive_branch.preference_respected,
        excessive_branch.ambiguity_preserved,
        excessive_branch.action_well_formed,
        excessive_branch.qualified,
    );
    if clarification.qualified {
        qualified += 1;
    } else {
        rejected += 1;
    }
    if excessive_branch.qualified {
        qualified += 1;
    } else {
        rejected += 1;
    }

    println!("negotiation_qualified,{qualified}");
    println!("negotiation_rejected,{rejected}");
    Ok((qualified, rejected))
}

fn run_authority_scope_conformance() -> Result<(usize, usize), Box<dyn Error>> {
    let observation = Observation {
        event_id: "authority-source-event".to_owned(),
        session_id: "authority-session".to_owned(),
        history_id: "authority-history".to_owned(),
        task_id: "authority-task".to_owned(),
        seed: 5,
        initial_partner_feature: Feature {
            dimension: Dimension::Motif,
            value: 2,
        },
        current_partner_feature: Feature {
            dimension: Dimension::Motif,
            value: 2,
        },
        unrelated_feature: Feature {
            dimension: Dimension::Texture,
            value: 0,
        },
    };
    let contract = NegotiatedResponseContractV1 {
        target_dimension: Dimension::Motif,
        modulus: 12,
        accepted_offsets: [3, 5, 7].into_iter().collect(),
        max_branch_count: 3,
    };
    let mut event = PartnerAuthorityEventV1 {
        event_id: "authority-event".to_owned(),
        session_id: observation.session_id.clone(),
        task_id: observation.task_id.clone(),
        source_event_id: observation.event_id.clone(),
        control_sequence: 7,
        clock_origin_id: "runner-clock-authority-session".to_owned(),
        issued_at_ns: 100,
        expires_at_ns: 200,
        authority_receipt_ref: "opaque-authority-receipt-ref".to_owned(),
        authority: PartnerAuthorityV1 {
            selected_value: None,
            vetoed_values: Default::default(),
        },
    };

    let valid = assess_scoped_negotiation(
        &contract,
        &observation,
        &event,
        7,
        "runner-clock-authority-session",
        150,
        &NegotiationChoiceV1::Branch(vec![5, 7, 9]),
    )?
    .qualified;
    println!("authority_scope,valid_event,{valid}");
    let mut qualified = usize::from(valid);
    let mut rejected = 0usize;

    event.source_event_id = "different-observation".to_owned();
    let wrong_source = validate_partner_authority_event(
        &event,
        &observation,
        7,
        "runner-clock-authority-session",
        150,
    );
    let passed = wrong_source.is_err();
    println!("authority_scope,cross_event_rejected,{passed}");
    if passed {
        rejected += 1;
    } else {
        qualified += 1;
    }

    event.source_event_id = observation.event_id.clone();
    event.control_sequence = 6;
    let stale_sequence = validate_partner_authority_event(
        &event,
        &observation,
        7,
        "runner-clock-authority-session",
        150,
    );
    let passed = stale_sequence.is_err();
    println!("authority_scope,stale_sequence_rejected,{passed}");
    if passed {
        rejected += 1;
    } else {
        qualified += 1;
    }

    event.control_sequence = 7;
    let wrong_clock =
        validate_partner_authority_event(&event, &observation, 7, "different-clock-origin", 150);
    let passed = wrong_clock.is_err();
    println!("authority_scope,clock_origin_rejected,{passed}");
    if passed {
        rejected += 1;
    } else {
        qualified += 1;
    }

    let not_effective = validate_partner_authority_event(
        &event, &observation, 7, "runner-clock-authority-session", 99,
    );
    let passed = not_effective.is_err();
    println!("authority_scope,not_yet_effective_rejected,{passed}");
    if passed {
        rejected += 1;
    } else {
        qualified += 1;
    }

    let expired = validate_partner_authority_event(
        &event, &observation, 7, "runner-clock-authority-session", 200,
    );
    let passed = expired.is_err();
    println!("authority_scope,expired_rejected,{passed}");
    if passed {
        rejected += 1;
    } else {
        qualified += 1;
    }

    event.authority_receipt_ref.clear();
    let missing_receipt = validate_partner_authority_event(
        &event,
        &observation,
        7,
        "runner-clock-authority-session",
        150,
    );
    let passed = missing_receipt.is_err();
    println!("authority_scope,missing_receipt_rejected,{passed}");
    if passed {
        rejected += 1;
    } else {
        qualified += 1;
    }

    println!("authority_scope_qualified,{qualified}");
    println!("authority_scope_rejected,{rejected}");
    Ok((qualified, rejected))
}

fn main() -> Result<(), Box<dyn Error>> {
    let code_revision = std::env::var("MEL_RMI_CODE_REVISION")?;
    let config = TrialRunConfig {
        policy_version: "canonical-fixture-policy-v1".to_owned(),
        code_revision,
    };
    let fixtures = canonical_fixture_replicates_v1(32);
    let artifact_report_path = std::env::var("MEL_RMI_ARTIFACT_REPORT")?;
    let artifact_report_file = File::create(artifact_report_path)?;
    let mut artifact_report = BufWriter::new(artifact_report_file);
    writeln!(
        artifact_report,
        "task_id,session_id,task_seed,randomization_seed,policy_id,policy_version,code_revision,metric_version,dimension,direction,condition,event_id,transaction_receipt_sha256,artifact_format,artifact_sha256,byte_len,verifier_id,verifier_version"
    )?;
    let mut artifact_report_rows = 0usize;
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
            artifact_report_rows += write_record_artifact_evidence(&mut artifact_report, record)?;
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
        report_policy_aggregates(&records);
    }

    let mut reciprocal = run_reciprocal_suite_v1(
        &fixtures,
        &RuleBasedPolicy {
            id: "rule-positive".to_owned(),
        },
        &ResponsivePartnerPolicy {
            id: "responsive-partner".to_owned(),
        },
        &config,
    )?;
    for record in &mut reciprocal {
        bind_and_verify_synthetic_artifacts(&mut record.agent)?;
        artifact_bindings_verified += 1;
        artifact_report_rows +=
            write_record_artifact_evidence(&mut artifact_report, &record.agent)?;
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
    report_reciprocal_aggregates(&reciprocal);

    let (negotiation_qualified, negotiation_rejected) = run_negotiation_conformance()?;
    if negotiation_qualified != 6 || negotiation_rejected != 5 {
        return Err("negotiation conformance count mismatch".into());
    }
    let (authority_scope_qualified, authority_scope_rejected) = run_authority_scope_conformance()?;
    if authority_scope_qualified != 1 || authority_scope_rejected != 6 {
        return Err("authority scope conformance count mismatch".into());
    }

    artifact_report.flush()?;
    println!("artifact_verified,{artifact_bindings_verified}");
    println!("artifact_digest_rows,{artifact_report_rows}");
    Ok(())
}
