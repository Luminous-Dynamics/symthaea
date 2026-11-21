//! Offline, deterministic RMI-Bench Track A runner.
//!
//! This runner qualifies the evaluator's synthetic controls. It does not run a
//! live Melothaea policy, render audio, or establish human musical relevance.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    path::Path,
    process::{Command, ExitCode},
};

use symthaea_muse::rmi_benchmark::{
    FixedPartnerResponsePolicyV1, FixedResponsePolicyV1, ImitationOnlyPolicyV1,
    OneWayPromptPolicyV1, RmiAgentActionV1, RmiEvaluationEnvelopeV1, RmiHistoryActorV1,
    RmiHistoryEventV1, RmiInterventionPairV1, RmiMusicalFeaturesV1, RmiPartnerPolicyV1,
    RmiPartnerResponseV1, RmiPolicyV1, RmiReciprocalEvaluationEnvelopeV1, RmiRoleV1,
    RmiTargetFeatureV1, SyntheticContingentPolicyV1, SyntheticResponsivePartnerPolicyV1,
    YokedReplayPolicyV1, evaluate_rmi_pair, evaluate_rmi_reciprocal_pair,
};

const RUNNER_SCHEMA_V1: &str = "melothaea-rmi-track-a-report-v1";
const MANIFEST_SCHEMA_V1: &str = "melothaea-rmi-track-a-manifest-v1";
const RUNNER_VERSION: &str = "0.1.0";
const MAX_SEED_COUNT: usize = 1024;
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-melothaea-rmi-track-a-report-v1";

#[derive(Debug)]
struct Args {
    seed_count: usize,
    source_commit: String,
    output: Option<String>,
    self_check: bool,
    internal_replay: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyDescriptorV1 {
    policy_id: String,
    version: String,
    role: String,
    interpretation: String,
}

#[derive(Debug, Clone, Serialize)]
struct RunManifestV1 {
    schema: String,
    runner_version: String,
    source_commit: String,
    cargo_lock_sha256: String,
    seed_schedule: Vec<u64>,
    targets: Vec<RmiTargetFeatureV1>,
    agent_policies: Vec<PolicyDescriptorV1>,
    partner_policies: Vec<PolicyDescriptorV1>,
    fixture_contract: String,
    denominator_contract: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct DirectionCountsV1 {
    trials: u64,
    targeted_sensitivity_count: u64,
    collateral_changed_feature_total: u64,
}

impl DirectionCountsV1 {
    fn observe(&mut self, targeted_sensitivity: bool, collateral_changed: usize) {
        self.trials += 1;
        if targeted_sensitivity {
            self.targeted_sensitivity_count += 1;
        }
        self.collateral_changed_feature_total += collateral_changed as u64;
    }
}

#[derive(Debug, Clone, Serialize)]
struct TrialReceiptV1 {
    receipt_id: String,
    trial_id: String,
    seed: u64,
    target: RmiTargetFeatureV1,
    intervention: RmiInterventionPairV1,
    agent_policy_id: String,
    partner_policy_id: String,
    intervention_pair_sha256: String,
    agent_evaluation: RmiEvaluationEnvelopeV1,
    reciprocal_evaluation: RmiReciprocalEvaluationEnvelopeV1,
}

#[derive(Debug, Clone, Serialize)]
struct RunReportPayloadV1 {
    schema: String,
    manifest: RunManifestV1,
    agent_to_input_sensitivity: BTreeMap<String, DirectionCountsV1>,
    agent_to_partner_sensitivity: BTreeMap<String, DirectionCountsV1>,
    receipts: Vec<TrialReceiptV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RunReportEnvelopeV1 {
    payload: RunReportPayloadV1,
    report_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_source_commit(value: &str) -> Result<(), String> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("--source-commit must be a 40-character lowercase Git commit SHA".into());
    }
    Ok(())
}

fn target_name(target: RmiTargetFeatureV1) -> &'static str {
    match target {
        RmiTargetFeatureV1::Motif => "motif",
        RmiTargetFeatureV1::Rhythm => "rhythm",
        RmiTargetFeatureV1::Tempo => "tempo",
        RmiTargetFeatureV1::Role => "role",
    }
}

fn targets() -> [RmiTargetFeatureV1; 4] {
    [
        RmiTargetFeatureV1::Motif,
        RmiTargetFeatureV1::Rhythm,
        RmiTargetFeatureV1::Tempo,
        RmiTargetFeatureV1::Role,
    ]
}

fn base_features(seed: u64) -> RmiMusicalFeaturesV1 {
    let motif_intervals = [
        2 + (seed % 5) as i16,
        -1 - ((seed / 3) % 4) as i16,
        3 + ((seed / 7) % 3) as i16,
        -2,
    ];
    let rhythm_onsets = std::array::from_fn(|index| {
        let mixed = seed
            .wrapping_mul(37)
            .wrapping_add((index as u64 + 11).wrapping_mul(19));
        mixed % 7 < 3
    });
    let role = match seed % 4 {
        0 => RmiRoleV1::Lead,
        1 => RmiRoleV1::Accompaniment,
        2 => RmiRoleV1::Bass,
        _ => RmiRoleV1::Counterline,
    };
    RmiMusicalFeaturesV1 {
        motif_intervals,
        rhythm_onsets,
        tempo_milli_bpm: 90_000 + (seed % 100) as u32 * 1_000,
        role,
    }
}

fn treatment_features(
    control: &RmiMusicalFeaturesV1,
    target: RmiTargetFeatureV1,
    seed: u64,
) -> RmiMusicalFeaturesV1 {
    let mut treatment = control.clone();
    match target {
        RmiTargetFeatureV1::Motif => {
            treatment.motif_intervals[0] = if treatment.motif_intervals[0] < 24 {
                treatment.motif_intervals[0] + 1
            } else {
                treatment.motif_intervals[0] - 1
            };
        }
        RmiTargetFeatureV1::Rhythm => {
            let index = (seed % 8) as usize;
            treatment.rhythm_onsets[index] = !treatment.rhythm_onsets[index];
        }
        RmiTargetFeatureV1::Tempo => {
            treatment.tempo_milli_bpm += 5_000;
        }
        RmiTargetFeatureV1::Role => {
            treatment.role = match treatment.role {
                RmiRoleV1::Lead => RmiRoleV1::Accompaniment,
                RmiRoleV1::Accompaniment => RmiRoleV1::Bass,
                RmiRoleV1::Bass => RmiRoleV1::Counterline,
                RmiRoleV1::Counterline => RmiRoleV1::Lead,
            };
        }
    }
    treatment
}

fn make_pair(seed: u64, target: RmiTargetFeatureV1) -> Result<RmiInterventionPairV1, String> {
    let control_features = base_features(seed);
    let treatment_features = treatment_features(&control_features, target, seed);
    let history = vec![
        RmiHistoryEventV1 {
            event_id: format!("partner-history-{seed}"),
            actor: RmiHistoryActorV1::Partner,
            features: control_features.clone(),
        },
        RmiHistoryEventV1 {
            event_id: format!("agent-history-{seed}"),
            actor: RmiHistoryActorV1::Agent,
            features: base_features(seed.wrapping_add(1)),
        },
    ];
    let history_bytes = serde_json::to_vec(&history)
        .map_err(|error| format!("history serialization failed: {error}"))?;
    let pair = RmiInterventionPairV1 {
        trial_id: format!("rmi-track-a-{seed:04}-{}", target_name(target)),
        seed,
        pre_intervention_history: history,
        pre_intervention_state_sha256: sha256_hex(&history_bytes),
        target,
        control_observation: symthaea_muse::rmi_benchmark::RmiPartnerObservationV1 {
            source_event_id: format!("partner-current-{seed}"),
            features: control_features,
        },
        treatment_observation: symthaea_muse::rmi_benchmark::RmiPartnerObservationV1 {
            source_event_id: format!("partner-current-{seed}"),
            features: treatment_features,
        },
    };
    symthaea_muse::rmi_benchmark::validate_rmi_intervention_pair(&pair)
        .map_err(|issues| format!("fixture failed intervention validation: {issues:?}"))?;
    Ok(pair)
}

fn agent_policy_descriptors() -> Vec<PolicyDescriptorV1> {
    vec![
        PolicyDescriptorV1 {
            policy_id: "fixed-response".into(),
            version: "1".into(),
            role: "negative_control".into(),
            interpretation: "Returns one constant action; must not show partner-to-agent sensitivity.".into(),
        },
        PolicyDescriptorV1 {
            policy_id: "one-way-prompt".into(),
            version: "1".into(),
            role: "negative_control".into(),
            interpretation: "Replays a frozen response independent of the current partner event.".into(),
        },
        PolicyDescriptorV1 {
            policy_id: "yoked-replay".into(),
            version: "1".into(),
            role: "negative_control".into(),
            interpretation: "Replays a matched prior response in both counterfactual arms.".into(),
        },
        PolicyDescriptorV1 {
            policy_id: "imitation-only".into(),
            version: "1".into(),
            role: "input_sensitivity_control_not_collaboration".into(),
            interpretation: "Copies observed surface features; sensitivity is not musical relevance or collaboration.".into(),
        },
        PolicyDescriptorV1 {
            policy_id: "synthetic-contingent".into(),
            version: "1".into(),
            role: "positive_control".into(),
            interpretation: "Applies a known bijective transformation to the configured target feature only.".into(),
        },
    ]
}

fn partner_policy_descriptors() -> Vec<PolicyDescriptorV1> {
    vec![
        PolicyDescriptorV1 {
            policy_id: "fixed-partner".into(),
            version: "1".into(),
            role: "negative_control".into(),
            interpretation: "Returns a constant partner response independent of the agent action."
                .into(),
        },
        PolicyDescriptorV1 {
            policy_id: "synthetic-responsive-partner".into(),
            version: "1".into(),
            role: "positive_control".into(),
            interpretation: "Applies a known target-specific transformation to the agent action."
                .into(),
        },
    ]
}

fn collect_partner<P, H>(
    agent_policy: &P,
    partner_policy: &H,
    pair: &RmiInterventionPairV1,
    agent_evaluation: &RmiEvaluationEnvelopeV1,
    partner_counts: &mut BTreeMap<String, DirectionCountsV1>,
    receipts: &mut Vec<TrialReceiptV1>,
) -> Result<(), String>
where
    P: RmiPolicyV1,
    H: RmiPartnerPolicyV1,
{
    let reciprocal = evaluate_rmi_reciprocal_pair(agent_policy, partner_policy, pair)
        .map_err(|error| format!("reciprocal evaluation failed: {error:?}"))?;
    if reciprocal.payload.agent_evaluation_sha256 != agent_evaluation.payload_sha256 {
        return Err(
            "agent evaluation digest differs between unidirectional and reciprocal paths".into(),
        );
    }
    let target = pair.target;
    let key = format!(
        "target={}|agent={}|partner={}",
        target_name(target),
        agent_policy.policy_id(),
        partner_policy.policy_id()
    );
    partner_counts.entry(key).or_default().observe(
        reciprocal.payload.targeted_partner_sensitivity,
        reciprocal.payload.partner_collateral_changed_feature_count,
    );
    receipts.push(TrialReceiptV1 {
        receipt_id: format!(
            "{}|agent={}|partner={}",
            pair.trial_id,
            agent_policy.policy_id(),
            partner_policy.policy_id()
        ),
        trial_id: pair.trial_id.clone(),
        seed: pair.seed,
        target,
        intervention: pair.clone(),
        agent_policy_id: agent_policy.policy_id().to_string(),
        partner_policy_id: partner_policy.policy_id().to_string(),
        intervention_pair_sha256: agent_evaluation.payload.intervention_pair_sha256.clone(),
        agent_evaluation: agent_evaluation.clone(),
        reciprocal_evaluation: reciprocal,
    });
    Ok(())
}

fn collect_agent_policy<P: RmiPolicyV1>(
    policy: &P,
    pair: &RmiInterventionPairV1,
    agent_counts: &mut BTreeMap<String, DirectionCountsV1>,
    partner_counts: &mut BTreeMap<String, DirectionCountsV1>,
    receipts: &mut Vec<TrialReceiptV1>,
) -> Result<(), String> {
    let agent_evaluation = evaluate_rmi_pair(policy, pair)
        .map_err(|error| format!("agent evaluation failed: {error:?}"))?;
    let key = format!(
        "target={}|policy={}",
        target_name(pair.target),
        policy.policy_id()
    );
    agent_counts.entry(key).or_default().observe(
        agent_evaluation.payload.targeted_input_sensitivity,
        agent_evaluation.payload.collateral_changed_feature_count,
    );

    let fixed_partner = FixedPartnerResponsePolicyV1 {
        id: "fixed-partner".into(),
        version: "1".into(),
        response: RmiPartnerResponseV1 {
            response_id: "fixed-partner-response".into(),
            features: base_features(pair.seed.wrapping_add(17)),
        },
    };
    collect_partner(
        policy,
        &fixed_partner,
        pair,
        &agent_evaluation,
        partner_counts,
        receipts,
    )?;

    let responsive_partner = SyntheticResponsivePartnerPolicyV1 {
        id: "synthetic-responsive-partner".into(),
        version: "1".into(),
        target: pair.target,
    };
    collect_partner(
        policy,
        &responsive_partner,
        pair,
        &agent_evaluation,
        partner_counts,
        receipts,
    )?;
    Ok(())
}

fn build_manifest(
    seed_count: usize,
    source_commit: &str,
    cargo_lock_sha256: &str,
) -> RunManifestV1 {
    RunManifestV1 {
        schema: MANIFEST_SCHEMA_V1.into(),
        runner_version: RUNNER_VERSION.into(),
        source_commit: source_commit.into(),
        cargo_lock_sha256: cargo_lock_sha256.into(),
        seed_schedule: (0..seed_count as u64).collect(),
        targets: targets().to_vec(),
        agent_policies: agent_policy_descriptors(),
        partner_policies: partner_policy_descriptors(),
        fixture_contract: "Paired arms share an identical frozen history and source event; exactly one declared symbolic feature changes. Synthetic input units are motif semitone intervals, an eight-step onset mask, milli-BPM, and one of four symbolic roles.".into(),
        denominator_contract: "Every seed × target × agent-policy × partner-policy combination emits one receipt. Invalid trials abort the run; no missing, failed, or abstained case is silently removed. Synthetic v1 reports exact counts only and does not attach inferential confidence intervals.".into(),
    }
}

fn build_report(
    seed_count: usize,
    source_commit: &str,
    cargo_lock_sha256: &str,
) -> Result<RunReportEnvelopeV1, String> {
    if !(1..=MAX_SEED_COUNT).contains(&seed_count) {
        return Err(format!("--seed-count must be in 1..={MAX_SEED_COUNT}"));
    }
    validate_source_commit(source_commit)?;
    if !is_lowercase_sha256(cargo_lock_sha256) {
        return Err("Cargo.lock digest is not a lowercase SHA-256".into());
    }

    let manifest = build_manifest(seed_count, source_commit, cargo_lock_sha256);
    let mut agent_counts = BTreeMap::new();
    let mut partner_counts = BTreeMap::new();
    let mut receipts = Vec::with_capacity(seed_count * 4 * 5 * 2);

    for seed in 0..seed_count as u64 {
        for target in targets() {
            let pair = make_pair(seed, target)?;
            let base_action = RmiAgentActionV1 {
                action_id: "fixed-agent-action".into(),
                features: base_features(seed.wrapping_add(31)),
            };

            let fixed = FixedResponsePolicyV1 {
                id: "fixed-response".into(),
                version: "1".into(),
                response: base_action.clone(),
            };
            collect_agent_policy(
                &fixed,
                &pair,
                &mut agent_counts,
                &mut partner_counts,
                &mut receipts,
            )?;

            let one_way = OneWayPromptPolicyV1 {
                id: "one-way-prompt".into(),
                version: "1".into(),
                frozen_response: base_action.clone(),
            };
            collect_agent_policy(
                &one_way,
                &pair,
                &mut agent_counts,
                &mut partner_counts,
                &mut receipts,
            )?;

            let yoked = YokedReplayPolicyV1 {
                id: "yoked-replay".into(),
                version: "1".into(),
                replayed_response: base_action.clone(),
            };
            collect_agent_policy(
                &yoked,
                &pair,
                &mut agent_counts,
                &mut partner_counts,
                &mut receipts,
            )?;

            let imitation = ImitationOnlyPolicyV1 {
                id: "imitation-only".into(),
                version: "1".into(),
            };
            collect_agent_policy(
                &imitation,
                &pair,
                &mut agent_counts,
                &mut partner_counts,
                &mut receipts,
            )?;

            let contingent = SyntheticContingentPolicyV1 {
                id: "synthetic-contingent".into(),
                version: "1".into(),
                target,
            };
            collect_agent_policy(
                &contingent,
                &pair,
                &mut agent_counts,
                &mut partner_counts,
                &mut receipts,
            )?;
        }
    }

    let payload = RunReportPayloadV1 {
        schema: RUNNER_SCHEMA_V1.into(),
        manifest,
        agent_to_input_sensitivity: agent_counts,
        agent_to_partner_sensitivity: partner_counts,
        receipts,
        claim_ceiling: vec![
            "This run qualifies the deterministic synthetic evaluator and its encoded positive/negative controls only.".into(),
            "Synthetic positive controls are not Melothaea and are not evidence of a live system passing RMI-Bench.".into(),
            "Input sensitivity and simulator sensitivity do not establish musical relevance, human causal influence, listener preference, collaboration value, or authorship.".into(),
            "No audio was generated, decoded, rendered, or physically observed by this Track A runner.".into(),
            "No claim about consciousness, subjective experience, or human-equivalent understanding is made.".into(),
        ],
    };
    let payload_bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("report serialization failed: {error}"))?;
    let report_sha256 = domain_digest(REPORT_DOMAIN, &payload_bytes);
    Ok(RunReportEnvelopeV1 {
        payload,
        report_sha256,
    })
}

fn cargo_lock_sha256() -> Result<String, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_lock = manifest_dir.join("../../..").join("Cargo.lock");
    let standalone_lock = manifest_dir.join("Cargo.lock");
    let lock_path = if workspace_lock.is_file() {
        workspace_lock
    } else {
        standalone_lock
    };
    let bytes = fs::read(&lock_path).map_err(|error| {
        format!(
            "could not read workspace or staged standalone Cargo.lock at {}: {error}",
            lock_path.display()
        )
    })?;
    Ok(sha256_hex(&bytes))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        seed_count: 16,
        source_commit: String::new(),
        output: None,
        self_check: false,
        internal_replay: false,
    };
    let mut iter = env::args().skip(1);
    while let Some(argument) = iter.next() {
        match argument.as_str() {
            "--seed-count" => {
                let value = iter.next().ok_or("--seed-count requires an integer")?;
                args.seed_count = value
                    .parse::<usize>()
                    .map_err(|_| "--seed-count must be an unsigned integer")?;
            }
            "--source-commit" => {
                args.source_commit = iter.next().ok_or("--source-commit requires a commit SHA")?;
            }
            "--output" => {
                args.output = Some(iter.next().ok_or("--output requires a file path")?);
            }
            "--self-check" => args.self_check = true,
            "--internal-replay" => args.internal_replay = true,
            "--help" | "-h" => {
                println!(
                    "rmi_bench_track_a --source-commit SHA [--seed-count N] [--output PATH] [--self-check]\n\
                     Defaults: --seed-count 16; maximum 1024.\n\
                     --self-check repeats the frozen run in a fresh process and compares report digests."
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    if args.source_commit.is_empty() {
        return Err("--source-commit is required; pass the exact source HEAD SHA".into());
    }
    validate_source_commit(&args.source_commit)?;
    if args.internal_replay && (args.self_check || args.output.is_some()) {
        return Err("--internal-replay cannot be combined with --self-check or --output".into());
    }
    Ok(args)
}

fn fresh_process_replay(
    seed_count: usize,
    source_commit: &str,
    expected_digest: &str,
) -> Result<(), String> {
    let executable = env::current_exe()
        .map_err(|error| format!("could not locate current executable: {error}"))?;
    let output = Command::new(executable)
        .arg("--seed-count")
        .arg(seed_count.to_string())
        .arg("--source-commit")
        .arg(source_commit)
        .arg("--internal-replay")
        .output()
        .map_err(|error| format!("could not launch fresh replay process: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "fresh replay process failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let observed = String::from_utf8(output.stdout)
        .map_err(|error| format!("fresh replay stdout was not UTF-8: {error}"))?;
    if observed.trim() != expected_digest {
        return Err(format!(
            "fresh-process report digest mismatch: expected {expected_digest}, got {}",
            observed.trim()
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let lock_digest = cargo_lock_sha256()?;
    let report = build_report(args.seed_count, &args.source_commit, &lock_digest)?;

    if args.internal_replay {
        println!("{}", report.report_sha256);
        return Ok(());
    }

    if args.self_check {
        fresh_process_replay(args.seed_count, &args.source_commit, &report.report_sha256)?;
        eprintln!(
            "fresh-process replay: PASS (report SHA-256 {})",
            report.report_sha256
        );
    }

    let serialized = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("pretty report serialization failed: {error}"))?;
    if let Some(path) = args.output {
        fs::write(&path, &serialized)
            .map_err(|error| format!("could not write report to {path}: {error}"))?;
        println!("report: {path}");
        println!("report_sha256: {}", report.report_sha256);
        println!("trial_receipts: {}", report.payload.receipts.len());
    } else {
        println!(
            "{}",
            String::from_utf8(serialized)
                .map_err(|error| format!("report JSON was not UTF-8: {error}"))?
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rmi_bench_track_a: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const TEST_LOCK_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn report(seed_count: usize) -> RunReportEnvelopeV1 {
        build_report(seed_count, TEST_COMMIT, TEST_LOCK_DIGEST).expect("valid deterministic report")
    }

    #[test]
    fn report_replay_is_byte_deterministic() {
        let first = serde_json::to_vec(&report(3)).unwrap();
        let second = serde_json::to_vec(&report(3)).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn manifest_and_receipt_denominators_are_complete() {
        let result = report(2);
        assert_eq!(result.payload.manifest.seed_schedule, vec![0, 1]);
        assert_eq!(result.payload.receipts.len(), 2 * 4 * 5 * 2);
        let receipt_ids: std::collections::BTreeSet<_> = result
            .payload
            .receipts
            .iter()
            .map(|receipt| receipt.receipt_id.as_str())
            .collect();
        assert_eq!(receipt_ids.len(), result.payload.receipts.len());
        assert!(result.payload.receipts.iter().all(|receipt| {
            receipt.intervention.trial_id == receipt.trial_id
                && receipt.intervention.target == receipt.target
        }));
        assert_eq!(result.payload.agent_to_input_sensitivity.len(), 4 * 5);
        assert_eq!(result.payload.agent_to_partner_sensitivity.len(), 4 * 5 * 2);
        assert!(
            result
                .payload
                .receipts
                .iter()
                .all(|receipt| receipt.intervention_pair_sha256
                    == receipt.agent_evaluation.payload.intervention_pair_sha256)
        );
    }

    #[test]
    fn synthetic_agent_positive_control_is_target_specific() {
        for target in targets() {
            let key = format!("target={}|policy=synthetic-contingent", target_name(target));
            let counts = report(4).payload.agent_to_input_sensitivity[&key].clone();
            assert_eq!(counts.trials, 4);
            assert_eq!(counts.targeted_sensitivity_count, 4);
            assert_eq!(counts.collateral_changed_feature_total, 0);
        }
    }

    #[test]
    fn fixed_one_way_and_yoked_controls_are_not_partner_sensitive() {
        let result = report(3);
        for policy in ["fixed-response", "one-way-prompt", "yoked-replay"] {
            for target in targets() {
                let key = format!("target={}|policy={policy}", target_name(target));
                let counts = &result.payload.agent_to_input_sensitivity[&key];
                assert_eq!(counts.targeted_sensitivity_count, 0);
                assert_eq!(counts.collateral_changed_feature_total, 0);
            }
        }
    }

    #[test]
    fn imitation_is_sensitive_but_is_not_labelled_as_collaboration() {
        let result = report(3);
        for target in targets() {
            let key = format!("target={}|policy=imitation-only", target_name(target));
            let counts = &result.payload.agent_to_input_sensitivity[&key];
            assert_eq!(counts.targeted_sensitivity_count, 3);
            assert_eq!(counts.collateral_changed_feature_total, 0);
        }
        let descriptor = result
            .payload
            .manifest
            .agent_policies
            .iter()
            .find(|policy| policy.policy_id == "imitation-only")
            .unwrap();
        assert_eq!(
            descriptor.role,
            "input_sensitivity_control_not_collaboration"
        );
    }

    #[test]
    fn fixed_partner_is_not_agent_action_sensitive() {
        let result = report(3);
        for agent in [
            "fixed-response",
            "one-way-prompt",
            "yoked-replay",
            "imitation-only",
            "synthetic-contingent",
        ] {
            for target in targets() {
                let key = format!(
                    "target={}|agent={agent}|partner=fixed-partner",
                    target_name(target)
                );
                let counts = &result.payload.agent_to_partner_sensitivity[&key];
                assert_eq!(counts.targeted_sensitivity_count, 0);
            }
        }
    }

    #[test]
    fn fixed_one_way_and_yoked_agents_do_not_influence_responsive_partner() {
        let result = report(3);
        for agent in ["fixed-response", "one-way-prompt", "yoked-replay"] {
            for target in targets() {
                let key = format!(
                    "target={}|agent={agent}|partner=synthetic-responsive-partner",
                    target_name(target)
                );
                let counts = &result.payload.agent_to_partner_sensitivity[&key];
                assert_eq!(counts.targeted_sensitivity_count, 0);
                assert_eq!(counts.collateral_changed_feature_total, 0);
            }
        }
    }

    #[test]
    fn responsive_partner_positive_control_tracks_contingent_agent() {
        let result = report(4);
        for target in targets() {
            let key = format!(
                "target={}|agent=synthetic-contingent|partner=synthetic-responsive-partner",
                target_name(target)
            );
            let counts = &result.payload.agent_to_partner_sensitivity[&key];
            assert_eq!(counts.trials, 4);
            assert_eq!(counts.targeted_sensitivity_count, 4);
            assert_eq!(counts.collateral_changed_feature_total, 0);
        }
    }

    #[test]
    fn invalid_source_commit_and_seed_count_fail_closed() {
        assert!(build_report(0, TEST_COMMIT, TEST_LOCK_DIGEST).is_err());
        assert!(build_report(MAX_SEED_COUNT + 1, TEST_COMMIT, TEST_LOCK_DIGEST).is_err());
        assert!(build_report(1, "not-a-commit", TEST_LOCK_DIGEST).is_err());
    }
}
