//! Paired A0/A2 smoke test that drives the JSONL baseline through a child process.
//!
//! This is an executable protocol integration check, not a security sandbox or
//! a learned-researcher evaluation. The fixed evaluator truth is constructed
//! only in this parent executable and is never passed in child arguments/env.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use open_world_ansatz_microcosm::numeric::{Mechanism, ModelHypothesis, SensorMap};
use open_world_ansatz_microcosm::process::{
    ProcessAgent, ProcessAgentConfig, ProcessEpisodeReceipt, run_process_episode,
};
use open_world_ansatz_microcosm::protocol::{BlindEpisode, StructureAccess};
use serde::Serialize;

#[derive(Serialize)]
struct PairedProcessSmokeReport {
    report_version: &'static str,
    campaign_kind: &'static str,
    policy: String,
    same_agent_artifact: bool,
    comparison_design: &'static str,
    a0: ProcessEpisodeReceipt,
    a2: ProcessEpisodeReceipt,
    out_of_catalogue: ProcessEpisodeReceipt,
    equivalent_out_of_catalogue: ProcessEpisodeReceipt,
    structural_out_of_catalogue: ProcessEpisodeReceipt,
    limitations: Vec<String>,
}

fn usage() -> &'static str {
    "usage: process_episode_smoke --agent ABSOLUTE_OR_RELATIVE_PATH [--strategy finite|maximum_partition|minimax|maximum_entropy|minimum_expected_remaining|maximum_structure_information_gain|fixed_grid|random:SEED]"
}

fn parse_args(args: &[String]) -> Result<(PathBuf, String), String> {
    let mut agent: Option<PathBuf> = None;
    let mut strategy = "finite".to_owned();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--agent" if index + 1 < args.len() => {
                agent = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--strategy" if index + 1 < args.len() => {
                strategy = args[index + 1].clone();
                index += 2;
            }
            _ => return Err(usage().to_owned()),
        }
    }

    let agent = agent.ok_or_else(|| usage().to_owned())?;
    Ok((agent, strategy))
}

fn child_args(strategy: &str) -> Vec<OsString> {
    if strategy == "finite" {
        Vec::new()
    } else {
        vec![OsString::from("--strategy"), OsString::from(strategy)]
    }
}

fn run_arm(
    executable: &Path,
    strategy: &str,
    access: StructureAccess,
    episode_id: &str,
    arm_label: &str,
    truth: ModelHypothesis,
) -> Result<ProcessEpisodeReceipt, String> {
    let episode = BlindEpisode::new_numeric(episode_id, truth, access, 6)
        .map_err(|error| format!("cannot create evaluator episode: {error}"))?;

    let config = ProcessAgentConfig::new(executable)
        .with_args(child_args(strategy))
        .with_working_directory(std::env::temp_dir());
    let agent = ProcessAgent::spawn(config)
        .map_err(|error| format!("cannot start child agent: {error}"))?;
    run_process_episode(episode, agent, arm_label)
        .map_err(|error| format!("process episode failed: {error}"))
}

fn run_open_support_arm(
    executable: &Path,
    strategy: &str,
) -> Result<ProcessEpisodeReceipt, String> {
    let truth = ModelHypothesis {
        mechanism: Mechanism::DirectResponse,
        gain: 7,
        decay: None,
        initial_state: None,
        sensor: SensorMap::Identity,
    };
    let episode = BlindEpisode::new_numeric_open_support(
        "process-open-support-out-of-catalogue",
        truth,
        StructureAccess::NumericWithheld,
        6,
    )
    .map_err(|error| format!("cannot create open-support evaluator episode: {error}"))?;
    let config = ProcessAgentConfig::new(executable)
        .with_args(child_args(strategy))
        .with_working_directory(std::env::temp_dir());
    let agent = ProcessAgent::spawn(config)
        .map_err(|error| format!("cannot start open-support child agent: {error}"))?;
    run_process_episode(episode, agent, "out_of_catalogue")
        .map_err(|error| format!("open-support process episode failed: {error}"))
}

fn run_equivalent_open_support_arm(
    executable: &Path,
    strategy: &str,
) -> Result<ProcessEpisodeReceipt, String> {
    let truth = ModelHypothesis {
        mechanism: Mechanism::DirectResponse,
        gain: 4,
        decay: None,
        initial_state: None,
        sensor: SensorMap::Identity,
    };
    let episode = BlindEpisode::new_numeric_open_support(
        "process-open-support-equivalent-parameterization",
        truth,
        StructureAccess::NumericWithheld,
        6,
    )
    .map_err(|error| format!("cannot create equivalent open-support episode: {error}"))?;
    let config = ProcessAgentConfig::new(executable)
        .with_args(child_args(strategy))
        .with_working_directory(std::env::temp_dir());
    let agent = ProcessAgent::spawn(config)
        .map_err(|error| format!("cannot start equivalent open-support child agent: {error}"))?;
    run_process_episode(episode, agent, "equivalent_out_of_catalogue")
        .map_err(|error| format!("equivalent open-support process episode failed: {error}"))
}

fn run_structural_shift_arm(
    executable: &Path,
    strategy: &str,
) -> Result<ProcessEpisodeReceipt, String> {
    let truth = ModelHypothesis {
        mechanism: Mechanism::DirectResponse,
        gain: 2,
        decay: None,
        initial_state: None,
        sensor: SensorMap::Identity,
    };
    let episode = BlindEpisode::new_numeric_structural_shift(
        "process-f2-quadratic-response-shift",
        truth,
        1,
        6,
    )
    .map_err(|error| format!("cannot create F2 structural-shift episode: {error}"))?;
    let config = ProcessAgentConfig::new(executable)
        .with_args(child_args(strategy))
        .with_working_directory(std::env::temp_dir());
    let agent = ProcessAgent::spawn(config)
        .map_err(|error| format!("cannot start F2 child agent: {error}"))?;
    run_process_episode(episode, agent, "structural_out_of_catalogue")
        .map_err(|error| format!("F2 structural-shift process episode failed: {error}"))
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (executable, strategy) = parse_args(&args)?;
    let truth = ModelHypothesis {
        mechanism: Mechanism::DirectResponse,
        gain: 2,
        decay: None,
        initial_state: None,
        sensor: SensorMap::Identity,
    };

    let a0 = run_arm(
        &executable,
        &strategy,
        StructureAccess::NumericSupplied(Mechanism::DirectResponse),
        "process-pair-0001-a0",
        "a0",
        truth,
    )?;
    let a2 = run_arm(
        &executable,
        &strategy,
        StructureAccess::NumericWithheld,
        "process-pair-0001-a2",
        "a2",
        truth,
    )?;
    let out_of_catalogue = run_open_support_arm(&executable, &strategy)?;
    let equivalent_out_of_catalogue = run_equivalent_open_support_arm(&executable, &strategy)?;
    let structural_out_of_catalogue = run_structural_shift_arm(&executable, &strategy)?;

    for (label, score) in [("a0", &a0.evaluator_score), ("a2", &a2.evaluator_score)] {
        if !score.generating_model_in_version_space
            || !score.set_score.exact_version_space_match
            || !score.evidence_references_valid
            || score.probe_count > 6
        {
            return Err(format!(
                "{label} failed the smoke gate: truth={}, exact_set={}, evidence_refs={}, probes={}",
                score.generating_model_in_version_space,
                score.set_score.exact_version_space_match,
                score.evidence_references_valid,
                score.probe_count
            ));
        }
    }

    if out_of_catalogue
        .evaluator_score
        .generating_model_in_version_space
        || out_of_catalogue
            .evaluator_score
            .set_score
            .oracle_model_count
            != 0
        || !out_of_catalogue
            .evaluator_score
            .set_score
            .exact_version_space_match
        || !out_of_catalogue.evaluator_score.decision_correct
        || !out_of_catalogue.evaluator_score.evidence_references_valid
        || out_of_catalogue.evaluator_score.probe_count > 6
    {
        return Err(format!(
            "out-of-catalogue support control failed: truth={}, oracle_models={}, exact_set={}, decision_correct={}, evidence_refs={}, probes={}",
            out_of_catalogue
                .evaluator_score
                .generating_model_in_version_space,
            out_of_catalogue
                .evaluator_score
                .set_score
                .oracle_model_count,
            out_of_catalogue
                .evaluator_score
                .set_score
                .exact_version_space_match,
            out_of_catalogue.evaluator_score.decision_correct,
            out_of_catalogue.evaluator_score.evidence_references_valid,
            out_of_catalogue.evaluator_score.probe_count
        ));
    }

    if equivalent_out_of_catalogue
        .evaluator_score
        .generating_model_in_version_space
        || !equivalent_out_of_catalogue
            .evaluator_score
            .generating_truth_behaviorally_represented
        || equivalent_out_of_catalogue
            .evaluator_score
            .set_score
            .oracle_model_count
            != 1
        || equivalent_out_of_catalogue
            .evaluator_score
            .set_score
            .claimed_model_count
            != 1
        || !equivalent_out_of_catalogue
            .evaluator_score
            .set_score
            .exact_version_space_match
        || !equivalent_out_of_catalogue.evaluator_score.decision_correct
        || !equivalent_out_of_catalogue
            .evaluator_score
            .evidence_references_valid
        || equivalent_out_of_catalogue.evaluator_score.probe_count > 6
    {
        return Err(format!(
            "equivalent out-of-catalogue control failed: exact_truth={}, behavioral_representation={}, oracle_models={}, exact_set={}, decision_correct={}, evidence_refs={}, probes={}",
            equivalent_out_of_catalogue
                .evaluator_score
                .generating_model_in_version_space,
            equivalent_out_of_catalogue
                .evaluator_score
                .generating_truth_behaviorally_represented,
            equivalent_out_of_catalogue
                .evaluator_score
                .set_score
                .oracle_model_count,
            equivalent_out_of_catalogue
                .evaluator_score
                .set_score
                .exact_version_space_match,
            equivalent_out_of_catalogue.evaluator_score.decision_correct,
            equivalent_out_of_catalogue
                .evaluator_score
                .evidence_references_valid,
            equivalent_out_of_catalogue.evaluator_score.probe_count
        ));
    }

    let same_process_artifact = [
        &a0.agent,
        &a2.agent,
        &out_of_catalogue.agent,
        &equivalent_out_of_catalogue.agent,
        &structural_out_of_catalogue.agent,
    ]
    .iter()
    .all(|identity| {
        identity.executable_sha256 == a0.agent.executable_sha256
            && identity.arguments == a0.agent.arguments
    });
    if !same_process_artifact {
        return Err(
            "paired process episodes used different agent artifacts or arguments".to_owned(),
        );
    }

    if structural_out_of_catalogue
        .evaluator_score
        .generating_model_in_version_space
        || structural_out_of_catalogue
            .evaluator_score
            .generating_truth_behaviorally_represented
        || structural_out_of_catalogue
            .evaluator_score
            .set_score
            .oracle_model_count
            != 0
        || structural_out_of_catalogue
            .evaluator_score
            .set_score
            .claimed_model_count
            != 0
        || !structural_out_of_catalogue
            .evaluator_score
            .set_score
            .exact_version_space_match
        || !structural_out_of_catalogue.evaluator_score.decision_correct
        || !structural_out_of_catalogue
            .evaluator_score
            .evidence_references_valid
        || structural_out_of_catalogue.evaluator_score.probe_count > 6
    {
        return Err(format!(
            "F2 structural-support control failed: exact_truth={}, behavioral_representation={}, oracle_models={}, claimed_models={}, exact_set={}, decision_correct={}, evidence_refs={}, probes={}",
            structural_out_of_catalogue
                .evaluator_score
                .generating_model_in_version_space,
            structural_out_of_catalogue
                .evaluator_score
                .generating_truth_behaviorally_represented,
            structural_out_of_catalogue
                .evaluator_score
                .set_score
                .oracle_model_count,
            structural_out_of_catalogue
                .evaluator_score
                .set_score
                .claimed_model_count,
            structural_out_of_catalogue
                .evaluator_score
                .set_score
                .exact_version_space_match,
            structural_out_of_catalogue.evaluator_score.decision_correct,
            structural_out_of_catalogue
                .evaluator_score
                .evidence_references_valid,
            structural_out_of_catalogue.evaluator_score.probe_count
        ));
    }

    let report = PairedProcessSmokeReport {
        report_version: "symthaea-blind-process-pair-smoke-v0",
        campaign_kind: "paired_process_baseline_smoke",
        policy: strategy,
        same_agent_artifact: a0.agent.executable_sha256 == a2.agent.executable_sha256,
        comparison_design: "adaptive_budget",
        a0,
        a2,
        out_of_catalogue,
        equivalent_out_of_catalogue,
        structural_out_of_catalogue,
        limitations: vec![
            "The policy is open_world_agent's finite baseline, not the learned Symthaea research policy.".to_owned(),
            "A0 and A2 share generating model, empty starting evidence, policy arguments and action budget; their adaptive traces may diverge.".to_owned(),
            "The child receives no generating truth, evaluator score or truth-bearing command-line argument.".to_owned(),
            "The parent/child process boundary is not an OS sandbox; filesystem and network access remain available unless an external isolator restricts them.".to_owned(),
            "The third episode is a truly unsupported open-support control: gain 7 is not behaviorally represented by any candidate, so correct behavior is an exact empty version space and no-compatible-model disposition.".to_owned(),
            "The fourth episode is parameter-OOD but behaviorally represented: gain 4 with an identity sensor is equivalent to candidate gain 2 with a double sensor; exact parameter membership is false, behavioral representation is true, and the correct claim is the singleton equivalent candidate.".to_owned(),
            "The fifth episode is a structural-support shift: direct response includes an evaluator-only quadratic input term that is absent from every candidate equation. The correct outcome is an empty version space, no behavioral-support claim and NoCompatibleModel.".to_owned(),
            "This five-episode smoke test verifies wire/process/evidence binding, parameter and structural support-omission handling, and behavioral-equivalence acceptance; it does not measure agent performance or establish scientific discovery.".to_owned(),
        ],
    };

    let stdout = io::stdout();
    let mut writer = stdout.lock();
    serde_json::to_writer_pretty(&mut writer, &report)
        .map_err(|error| format!("cannot encode paired receipt: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("cannot write receipt terminator: {error}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(64)
        }
    }
}
