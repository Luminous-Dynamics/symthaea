// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-005 structural graph and latent-identifiability qualification lane.
//!
//! Public development fixtures only: this lane is intentionally marked exposed and cannot
//! support fresh-holdout or superhuman claims.

#[cfg(feature = "reasoning_engine")]
mod qualified {
    use std::collections::HashSet;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;
    use symthaea::consciousness::counterfactual::{
        CausalDAG, CausalGraphWithLatents, IDAlgorithm,
    };
    use symthaea::intelligence::{
        AbstentionReason, CapabilityLaneBundle, CapabilityLaneDescriptor, ContaminationStatus,
        EpisodeJudgment, EvidenceRef, ExactScoringPolicy, HoldoutPolicy, ReasoningDecisionRecord,
        ReasoningDomain, ReasoningEpisode, ReasoningOutcome, ReasoningProblemRef,
        ReasoningQualificationReceipt, ResourceBudget, ResourceUsage, evaluate_episode_with_policy,
    };

    const BENCHMARK: &str = "RQ-005 causal graph semantic conformance";
    const VERSION: &str = "v1";
    const SPLIT: &str = "development";
    const CONFIGURATION: &str = "causal-graph-semantics-v1";
    const SOURCE_GROUP: &str = "rq-005-public-graph-fixtures-v1";

    struct Observation {
        actual: String,
        correct: bool,
        outcome: ReasoningOutcome,
        operation: String,
    }

    pub fn run() -> Result<(), String> {
        let subject_revision = required_env("SYMTHAEA_SUBJECT_REVISION")?;
        let output = env::var("SYMTHAEA_CAUSAL_GRAPH_LANE_BUNDLE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from("data/benchmarks/reasoning/causal-graph-semantics-lane.json")
            });

        let mut episodes = Vec::new();
        let mut receipts = Vec::new();

        run_id_case(
            &subject_revision,
            "id-markovian-direct-effect",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into()],
                vec![(0, 1)],
                vec![],
            ),
            &[0],
            &[1],
            true,
            &mut episodes,
            &mut receipts,
        )?;

        // Bow graph: direct X→Y plus latent X↔Y confounding is the canonical simple hedge.
        run_id_case(
            &subject_revision,
            "id-bow-graph-hedge",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into()],
                vec![(0, 1)],
                vec![(0, 1)],
            ),
            &[0],
            &[1],
            false,
            &mut episodes,
            &mut receipts,
        )?;

        run_dsep_case(
            &subject_revision,
            "dsep-chain-open",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into()],
                vec![(0, 1), (1, 2)],
            ),
            0,
            2,
            &[],
            false,
            &mut episodes,
            &mut receipts,
        )?;
        run_dsep_case(
            &subject_revision,
            "dsep-chain-blocked-by-middle",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into()],
                vec![(0, 1), (1, 2)],
            ),
            0,
            2,
            &[1],
            true,
            &mut episodes,
            &mut receipts,
        )?;
        run_dsep_case(
            &subject_revision,
            "dsep-fork-open",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into()],
                vec![(1, 0), (1, 2)],
            ),
            0,
            2,
            &[],
            false,
            &mut episodes,
            &mut receipts,
        )?;
        run_dsep_case(
            &subject_revision,
            "dsep-fork-blocked-by-common-cause",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into()],
                vec![(1, 0), (1, 2)],
            ),
            0,
            2,
            &[1],
            true,
            &mut episodes,
            &mut receipts,
        )?;
        run_dsep_case(
            &subject_revision,
            "dsep-collider-closed",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into()],
                vec![(0, 1), (2, 1)],
            ),
            0,
            2,
            &[],
            true,
            &mut episodes,
            &mut receipts,
        )?;
        run_dsep_case(
            &subject_revision,
            "dsep-collider-open-when-conditioned",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into()],
                vec![(0, 1), (2, 1)],
            ),
            0,
            2,
            &[1],
            false,
            &mut episodes,
            &mut receipts,
        )?;
        run_dsep_case(
            &subject_revision,
            "dsep-collider-descendant-opens",
            CausalDAG::new(
                vec!["A".into(), "B".into(), "C".into(), "D".into()],
                vec![(0, 1), (2, 1), (1, 3)],
            ),
            0,
            2,
            &[3],
            false,
            &mut episodes,
            &mut receipts,
        )?;

        let descriptor = CapabilityLaneDescriptor {
            lane_id: "causal-graph-semantics-development-v1".into(),
            domain: ReasoningDomain::Causal,
            benchmark: BENCHMARK.into(),
            benchmark_version: VERSION.into(),
            split: SPLIT.into(),
            holdout_policy: HoldoutPolicy::DevelopmentProbe,
            contamination_policy_id: SOURCE_GROUP.into(),
            contamination_status: ContaminationStatus::Exposed,
            resource_budget: ResourceBudget {
                wall_time_us: Some(1_000_000),
                deliberation_steps: Some(1),
                tool_calls: Some(0),
                model_tokens: Some(0),
            },
        };
        let bundle = CapabilityLaneBundle::new(descriptor, episodes, receipts, vec![]);
        let report = bundle
            .qualify()
            .map_err(|err| format!("graph semantic lane failed qualification: {err}"))?;

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        fs::write(
            &output,
            serde_json::to_vec_pretty(&bundle)
                .map_err(|err| format!("failed to encode lane bundle: {err}"))?,
        )
        .map_err(|err| format!("failed to write {}: {err}", output.display()))?;

        println!("RQ-005 graph semantics");
        println!("cases:          {}", report.capability.episodes);
        println!("exact correct:  {}", report.capability.exact_correct);
        println!(
            "exact accuracy: {}",
            report
                .capability
                .exact_accuracy
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".into())
        );
        println!("coverage:       {:.3}", report.capability.coverage);
        println!("bundle:         {}", output.display());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_id_case(
        subject_revision: &str,
        id: &str,
        graph: CausalGraphWithLatents,
        treatment: &[usize],
        outcome: &[usize],
        expected_identified: bool,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
    ) -> Result<(), String> {
        let input = serde_json::to_vec(&(&graph, treatment, outcome))
            .map_err(|err| format!("failed to encode ID fixture {id}: {err}"))?;
        let started = Instant::now();
        let actual = IDAlgorithm::new().identify(&graph, treatment, outcome);
        let wall_time_us = elapsed_us(started);

        let observation = match actual {
            Ok(expression) => Observation {
                actual: format!("identified:{}", expression.to_string(&graph.nodes)),
                correct: expected_identified,
                outcome: ReasoningOutcome::Asserted {
                    value: "identified".into(),
                    confidence: 0.5,
                },
                operation: "IDAlgorithm::identify => identified expression; uncalibrated confidence fixed at neutral 0.5".into(),
            },
            Err((hedge_nodes, description)) => Observation {
                actual: format!("unidentified:hedge:{hedge_nodes:?}:{description}"),
                correct: !expected_identified && !hedge_nodes.is_empty(),
                outcome: ReasoningOutcome::Abstained {
                    reason: AbstentionReason::Unidentified,
                    answerability: 0.0,
                },
                operation: format!("IDAlgorithm::identify => hedge {hedge_nodes:?}: {description}"),
            },
        };
        record(
            subject_revision,
            id,
            &input,
            if expected_identified { "identified" } else { "unidentified:hedge" },
            observation,
            wall_time_us,
            episodes,
            receipts,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_dsep_case(
        subject_revision: &str,
        id: &str,
        dag: CausalDAG,
        a: usize,
        b: usize,
        conditioned: &[usize],
        expected_separated: bool,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
    ) -> Result<(), String> {
        let input = serde_json::to_vec(&(&dag, a, b, conditioned))
            .map_err(|err| format!("failed to encode d-separation fixture {id}: {err}"))?;
        let z: HashSet<usize> = conditioned.iter().copied().collect();
        let started = Instant::now();
        let separated = dag.is_d_separated(a, b, &z);
        let wall_time_us = elapsed_us(started);
        let label = if separated { "d-separated" } else { "d-connected" };
        let observation = Observation {
            actual: label.into(),
            correct: separated == expected_separated,
            outcome: ReasoningOutcome::Asserted {
                value: label.into(),
                confidence: 0.5,
            },
            operation: format!(
                "CausalDAG::is_d_separated({a},{b},conditioned={conditioned:?}) => {label}; uncalibrated confidence fixed at neutral 0.5"
            ),
        };
        record(
            subject_revision,
            id,
            &input,
            if expected_separated { "d-separated" } else { "d-connected" },
            observation,
            wall_time_us,
            episodes,
            receipts,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        subject_revision: &str,
        id: &str,
        input: &[u8],
        expected: &str,
        observation: Observation,
        wall_time_us: u64,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
    ) -> Result<(), String> {
        let input_hash = blake3::hash(input).to_hex().to_string();
        if !observation.correct {
            eprintln!("semantic mismatch {id}: expected {expected}, got {}", observation.actual);
        }
        let asserted = matches!(&observation.outcome, ReasoningOutcome::Asserted { .. });
        let episode = ReasoningEpisode::new(
            subject_revision,
            CONFIGURATION,
            ReasoningDomain::Causal,
            ReasoningProblemRef {
                benchmark: BENCHMARK.into(),
                benchmark_version: VERSION.into(),
                split: SPLIT.into(),
                problem_id: id.into(),
                problem_hash: format!("blake3:{input_hash}"),
            },
            vec![EvidenceRef {
                id: format!("{id}:input"),
                content_hash: format!("blake3:{input_hash}"),
                provenance: "RQ-005 public graph-semantic fixture input; oracle withheld from subject call".into(),
                independence_group: Some(SOURCE_GROUP.into()),
            }],
            vec![],
            vec![ReasoningDecisionRecord {
                operation: observation.operation,
                input_refs: vec![format!("{id}:input")],
                output_refs: if asserted { vec![format!("{id}:semantic-output")] } else { vec![] },
                verifier: Some("rq-005-graph-semantic-oracle-v1".into()),
            }],
            observation.outcome,
            ResourceUsage {
                wall_time_us,
                deliberation_steps: 1,
                tool_calls: 0,
                model_tokens: 0,
            },
        )
        .map_err(|err| format!("invalid episode {id}: {err}"))?;
        let lineage = format!(
            "rq-005-graph:v1:{id}:{}",
            blake3::hash(expected.as_bytes()).to_hex()
        );
        let receipt = evaluate_episode_with_policy(
            &episode,
            &lineage,
            &EpisodeJudgment {
                exact_correct: Some(observation.correct),
                task_score: None,
            },
            ExactScoringPolicy::OutcomeSemantic,
        )
        .map_err(|err| format!("evaluation failed for {id}: {err}"))?;
        episodes.push(episode);
        receipts.push(receipt);
        Ok(())
    }

    fn elapsed_us(started: Instant) -> u64 {
        started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
    }

    fn required_env(name: &str) -> Result<String, String> {
        match env::var(name) {
            Ok(value) if !value.trim().is_empty() => Ok(value),
            _ => Err(format!("required environment variable {name} is missing")),
        }
    }
}

#[cfg(feature = "reasoning_engine")]
fn main() {
    if let Err(err) = qualified::run() {
        eprintln!("causal graph semantic qualification failed: {err}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("benchmark_causal_graph_semantics requires --features reasoning_engine");
}
