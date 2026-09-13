// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-005B qualified Markovian identification evidence lane.
//!
//! Public development/conformance evidence only. The expected expression trees are visible in
//! this runner, so this lane is `DevelopmentProbe` + `Exposed`, not fresh holdout evidence.
//!
//! Run with `--features reasoning_engine`.

#[cfg(feature = "reasoning_engine")]
mod qualified {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use serde::Serialize;
    use symthaea::consciousness::counterfactual::{
        CausalExpression, CausalGraphWithLatents, QualifiedMarkovianId,
        QualifiedMarkovianIdError, QUALIFIED_MARKOVIAN_ID_VERSION,
    };
    use symthaea::intelligence::{
        AbstentionReason, CapabilityLaneBundle, CapabilityLaneDescriptor, ContaminationStatus,
        EpisodeJudgment, ExactScoringPolicy, HoldoutPolicy, ReasoningDomain, ReasoningEpisode,
        ReasoningOutcome, ReasoningProblemRef, ReasoningQualificationReceipt, ResourceBudget,
        ResourceUsage, evaluate_episode_with_policy,
    };

    const BENCHMARK: &str = "RQ-005B qualified Markovian g-formula conformance";
    const BENCHMARK_VERSION: &str = "v1";
    const SPLIT: &str = "development";
    const CONFIGURATION_ID: &str = "qualified-markovian-id-v1";
    const CONTAMINATION_POLICY: &str = "rq-005b-public-fixtures-v1";

    #[derive(Debug)]
    enum Expected {
        Formula(CausalExpression),
        UnsupportedLatent,
        DirectedCycle,
    }

    #[derive(Serialize)]
    struct ConsoleSummary {
        subject_revision: String,
        implementation_version: &'static str,
        cases: usize,
        exact_correct: usize,
        failures: Vec<String>,
        bundle_path: String,
    }

    fn p(outcome: usize, conditioning: Vec<usize>) -> CausalExpression {
        CausalExpression::Probability {
            outcome: vec![outcome],
            conditioning,
        }
    }

    fn product(parts: Vec<CausalExpression>) -> CausalExpression {
        CausalExpression::Product(parts)
    }

    fn sum(sum_over: Vec<usize>, inner: CausalExpression) -> CausalExpression {
        CausalExpression::Sum {
            sum_over,
            inner: Box::new(inner),
        }
    }

    pub fn run() -> Result<(), String> {
        let subject_revision = env::var("SYMTHAEA_SUBJECT_REVISION")
            .map_err(|_| "SYMTHAEA_SUBJECT_REVISION is required".to_string())?;
        let output = env::var("SYMTHAEA_MARKOVIAN_ID_LANE_BUNDLE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from("data/benchmarks/reasoning/qualified-markovian-id-lane.json")
            });

        let mut episodes = Vec::new();
        let mut receipts = Vec::new();
        let mut failures = Vec::new();

        run_case(
            &subject_revision,
            "direct-effect",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into()],
                vec![(0, 1)],
                vec![],
            ),
            vec![0],
            vec![1],
            Expected::Formula(p(1, vec![0])),
            &mut episodes,
            &mut receipts,
            &mut failures,
        )?;

        run_case(
            &subject_revision,
            "observed-backdoor-confounder",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into(), "Z".into()],
                vec![(2, 0), (2, 1), (0, 1)],
                vec![],
            ),
            vec![0],
            vec![1],
            Expected::Formula(sum(
                vec![2],
                product(vec![p(2, vec![]), p(1, vec![0, 2])]),
            )),
            &mut episodes,
            &mut receipts,
            &mut failures,
        )?;

        run_case(
            &subject_revision,
            "mediated-effect",
            CausalGraphWithLatents::new(
                vec!["X".into(), "M".into(), "Y".into()],
                vec![(0, 1), (1, 2)],
                vec![],
            ),
            vec![0],
            vec![2],
            Expected::Formula(sum(
                vec![1],
                product(vec![p(1, vec![0]), p(2, vec![1])]),
            )),
            &mut episodes,
            &mut receipts,
            &mut failures,
        )?;

        run_case(
            &subject_revision,
            "two-observed-confounders",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into(), "Z1".into(), "Z2".into()],
                vec![(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)],
                vec![],
            ),
            vec![0],
            vec![1],
            Expected::Formula(sum(
                vec![2, 3],
                product(vec![p(2, vec![]), p(3, vec![]), p(1, vec![0, 2, 3])]),
            )),
            &mut episodes,
            &mut receipts,
            &mut failures,
        )?;

        run_case(
            &subject_revision,
            "latent-bow-fails-closed",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into()],
                vec![(0, 1)],
                vec![(0, 1)],
            ),
            vec![0],
            vec![1],
            Expected::UnsupportedLatent,
            &mut episodes,
            &mut receipts,
            &mut failures,
        )?;

        run_case(
            &subject_revision,
            "cycle-fails-closed",
            CausalGraphWithLatents::new(
                vec!["X".into(), "Y".into()],
                vec![(0, 1), (1, 0)],
                vec![],
            ),
            vec![0],
            vec![1],
            Expected::DirectedCycle,
            &mut episodes,
            &mut receipts,
            &mut failures,
        )?;

        let descriptor = CapabilityLaneDescriptor {
            lane_id: "qualified-markovian-id-development-v1".into(),
            domain: ReasoningDomain::Causal,
            benchmark: BENCHMARK.into(),
            benchmark_version: BENCHMARK_VERSION.into(),
            split: SPLIT.into(),
            holdout_policy: HoldoutPolicy::DevelopmentProbe,
            contamination_policy_id: CONTAMINATION_POLICY.into(),
            contamination_status: ContaminationStatus::Exposed,
            resource_budget: ResourceBudget::default(),
        };
        let bundle = CapabilityLaneBundle::new(descriptor, episodes, receipts, vec![]);
        let report = bundle
            .qualify()
            .map_err(|err| format!("Markovian ID lane failed structural qualification: {err}"))?;

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

        let summary = ConsoleSummary {
            subject_revision,
            implementation_version: QUALIFIED_MARKOVIAN_ID_VERSION,
            cases: report.capability.episodes,
            exact_correct: report.capability.exact_correct,
            failures,
            bundle_path: output.display().to_string(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode summary: {err}"))?
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        subject_revision: &str,
        id: &str,
        graph: CausalGraphWithLatents,
        treatment: Vec<usize>,
        outcome: Vec<usize>,
        expected: Expected,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
        failures: &mut Vec<String>,
    ) -> Result<(), String> {
        let input = serde_json::to_vec(&(&graph, &treatment, &outcome))
            .map_err(|err| format!("failed to encode fixture {id}: {err}"))?;
        let problem_hash = format!("blake3:{}", blake3::hash(&input).to_hex());

        let actual = QualifiedMarkovianId::new().identify(&graph, &treatment, &outcome);
        let (actual_label, correct, reasoning_outcome) = match (expected, actual) {
            (Expected::Formula(expected_expression), Ok(expression)) => {
                let expected_value = serde_json::to_value(&expected_expression)
                    .map_err(|err| format!("failed to encode expected expression {id}: {err}"))?;
                let observed_value = serde_json::to_value(&expression)
                    .map_err(|err| format!("failed to encode observed expression {id}: {err}"))?;
                let correct = observed_value == expected_value;
                let observed = expression.to_string(&graph.nodes);
                let expected = expected_expression.to_string(&graph.nodes);
                (
                    format!("formula:{observed}; expected:{expected}"),
                    correct,
                    ReasoningOutcome::Asserted {
                        value: observed,
                        confidence: 1.0,
                    },
                )
            }
            (
                Expected::UnsupportedLatent,
                Err(QualifiedMarkovianIdError::UnsupportedLatentGraph { .. }),
            ) => (
                "unidentified:unsupported-latent".into(),
                true,
                ReasoningOutcome::Abstained {
                    reason: AbstentionReason::Unidentified,
                    answerability: 0.0,
                },
            ),
            (Expected::DirectedCycle, Err(QualifiedMarkovianIdError::DirectedCycle)) => (
                "invalid:directed-cycle".into(),
                true,
                ReasoningOutcome::Abstained {
                    reason: AbstentionReason::Other("invalid-problem".into()),
                    answerability: 0.0,
                },
            ),
            (expected, Ok(expression)) => (
                format!(
                    "unexpected-formula:{}; expected={expected:?}",
                    expression.to_string(&graph.nodes)
                ),
                false,
                ReasoningOutcome::Asserted {
                    value: "unexpected-formula".into(),
                    confidence: 1.0,
                },
            ),
            (expected, Err(err)) => (
                format!("unexpected-error:{err}; expected={expected:?}"),
                false,
                ReasoningOutcome::Abstained {
                    reason: AbstentionReason::Unidentified,
                    answerability: 0.0,
                },
            ),
        };

        if !correct {
            failures.push(format!("{id}: {actual_label}"));
        }

        let episode = ReasoningEpisode::new(
            subject_revision,
            CONFIGURATION_ID,
            ReasoningDomain::Causal,
            ReasoningProblemRef {
                benchmark: BENCHMARK.into(),
                benchmark_version: BENCHMARK_VERSION.into(),
                split: SPLIT.into(),
                problem_id: id.into(),
                problem_hash,
            },
            vec![],
            vec![],
            vec![],
            reasoning_outcome,
            ResourceUsage::default(),
        )
        .map_err(|err| format!("failed to build episode {id}: {err}"))?;

        let receipt = evaluate_episode_with_policy(
            &episode,
            &format!("{CONTAMINATION_POLICY}:{id}"),
            &EpisodeJudgment {
                exact_correct: Some(correct),
                task_score: None,
            },
            ExactScoringPolicy::OutcomeSemantic,
        )
        .map_err(|err| format!("failed to evaluate episode {id}: {err}"))?;

        episodes.push(episode);
        receipts.push(receipt);
        Ok(())
    }
}

#[cfg(feature = "reasoning_engine")]
fn main() {
    if let Err(err) = qualified::run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("benchmark_qualified_markovian_id requires --features reasoning_engine");
    std::process::exit(2);
}
