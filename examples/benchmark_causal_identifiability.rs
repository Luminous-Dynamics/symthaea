// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-005 causal identifiability semantic qualification.
//!
//! This is a public development/conformance lane, not fresh holdout evidence. The oracle is
//! encoded in the runner and therefore the lane is explicitly marked `DevelopmentProbe` and
//! `Exposed`. Its purpose is to reveal semantic defects before frozen unseen qualification.
//!
//! Run with `--features reasoning_engine`.

#[cfg(feature = "reasoning_engine")]
mod qualified {
    use serde::Serialize;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;
    use symthaea::consciousness::counterfactual::{
        CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner,
        IdentificationMethod, IVEstimator, IVValidity, MediationAnalysis, MediationIdentification,
        UnidentifiedReason,
    };
    use symthaea::intelligence::{
        AbstentionReason, CapabilityLaneBundle, CapabilityLaneDescriptor, ContaminationStatus,
        EpisodeJudgment, EvidenceRef, ExactScoringPolicy, HoldoutPolicy, ReasoningDecisionRecord,
        ReasoningDomain, ReasoningEpisode, ReasoningOutcome, ReasoningProblemRef,
        ReasoningQualificationReceipt, ResourceBudget, ResourceUsage, evaluate_episode_with_policy,
    };

    const BENCHMARK: &str = "RQ-005 causal semantic conformance";
    const BENCHMARK_VERSION: &str = "v1";
    const SPLIT: &str = "development";
    const CONFIGURATION_ID: &str = "causal-semantic-conformance-v1";
    const COMMON_SOURCE: &str = "rq-005-public-fixtures-v1";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SemanticClass {
        Identified,
        IdentifiedBackdoor,
        IdentifiedZero,
        UnidentifiedDagTooLarge,
        InstrumentValid,
        InstrumentInvalid,
        MediationIdentified,
        MediationNotMediator,
        MediationExposureConfounded,
    }

    impl SemanticClass {
        fn label(self) -> &'static str {
            match self {
                Self::Identified => "identified",
                Self::IdentifiedBackdoor => "identified:backdoor",
                Self::IdentifiedZero => "identified:zero-effect",
                Self::UnidentifiedDagTooLarge => "unidentified:dag-too-large",
                Self::InstrumentValid => "instrument:valid",
                Self::InstrumentInvalid => "instrument:invalid",
                Self::MediationIdentified => "mediation:identified",
                Self::MediationNotMediator => "mediation:not-mediator",
                Self::MediationExposureConfounded => "mediation:exposure-induced-confounding",
            }
        }
    }

    #[derive(Debug)]
    struct Observation {
        actual_label: String,
        correct: bool,
        outcome: ReasoningOutcome,
        operation: String,
    }

    #[derive(Serialize)]
    struct ConsoleSummary {
        subject_revision: String,
        cases: usize,
        exact_correct: usize,
        asserted: usize,
        abstained: usize,
        failures: Vec<String>,
        bundle_path: String,
    }

    pub fn run() -> Result<(), String> {
        let subject_revision = required_env("SYMTHAEA_SUBJECT_REVISION")?;
        let output = env::var("SYMTHAEA_CAUSAL_LANE_BUNDLE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from("data/benchmarks/reasoning/causal-identifiability-lane.json")
            });

        let mut episodes = Vec::new();
        let mut receipts = Vec::new();
        let mut failures = Vec::new();
        let mut asserted = 0usize;
        let mut abstained = 0usize;

        run_query_case(
            &subject_revision,
            "direct-edge-identified",
            CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]),
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            SemanticClass::Identified,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
            &mut abstained,
        )?;

        run_query_case(
            &subject_revision,
            "measured-backdoor-confounder",
            CausalDAG::new(
                vec!["X".into(), "Y".into(), "U".into()],
                vec![(2, 0), (2, 1), (0, 1)],
            ),
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            SemanticClass::IdentifiedBackdoor,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
            &mut abstained,
        )?;

        // A complete causal DAG with no directed path X→Y establishes zero total causal effect.
        // Treating this as epistemically unidentified conflates "effect is zero" with "unknown".
        run_query_case(
            &subject_revision,
            "known-disconnected-zero-effect",
            CausalDAG::new(vec!["X".into(), "Y".into()], vec![]),
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            SemanticClass::IdentifiedZero,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
            &mut abstained,
        )?;

        // Reverse-direction query on a known X→Y DAG: Y has no causal path to X, so effect is zero.
        run_query_case(
            &subject_revision,
            "known-reverse-zero-effect",
            CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]),
            CausalQuery {
                treatment: 1,
                outcome: 0,
                conditioning: vec![],
            },
            SemanticClass::IdentifiedZero,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
            &mut abstained,
        )?;

        let mut oversized_nodes: Vec<String> = (0..21).map(|i| format!("N{i}")).collect();
        oversized_nodes[0] = "X".into();
        oversized_nodes[1] = "Y".into();
        run_query_case(
            &subject_revision,
            "oversized-dag-honest-abstention",
            CausalDAG::new(oversized_nodes, vec![(0, 1)]),
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            SemanticClass::UnidentifiedDagTooLarge,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
            &mut abstained,
        )?;

        run_iv_case(
            &subject_revision,
            "valid-instrument",
            CausalDAG::new(
                vec!["Z".into(), "X".into(), "Y".into(), "U".into()],
                vec![(0, 1), (1, 2), (3, 1), (3, 2)],
            ),
            0,
            1,
            2,
            SemanticClass::InstrumentValid,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
        )?;

        run_iv_case(
            &subject_revision,
            "invalid-instrument-direct-outcome-edge",
            CausalDAG::new(
                vec!["Z".into(), "X".into(), "Y".into()],
                vec![(0, 1), (1, 2), (0, 2)],
            ),
            0,
            1,
            2,
            SemanticClass::InstrumentInvalid,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
        )?;

        // Exclusion restriction violation through an alternate Z→W→Y path. A validity checker
        // must reject this even though the legitimate Z→X→Y path also exists.
        run_iv_case(
            &subject_revision,
            "invalid-instrument-alternate-outcome-path",
            CausalDAG::new(
                vec!["Z".into(), "X".into(), "Y".into(), "W".into()],
                vec![(0, 1), (1, 2), (0, 3), (3, 2)],
            ),
            0,
            1,
            2,
            SemanticClass::InstrumentInvalid,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
        )?;

        run_mediation_case(
            &subject_revision,
            "mediation-identified",
            CausalDAG::new(
                vec!["X".into(), "M".into(), "Y".into()],
                vec![(0, 1), (1, 2), (0, 2)],
            ),
            0,
            1,
            2,
            SemanticClass::MediationIdentified,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
        )?;

        run_mediation_case(
            &subject_revision,
            "mediation-not-a-mediator",
            CausalDAG::new(
                vec!["X".into(), "M".into(), "Y".into()],
                vec![(0, 2)],
            ),
            0,
            1,
            2,
            SemanticClass::MediationNotMediator,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
        )?;

        run_mediation_case(
            &subject_revision,
            "mediation-exposure-induced-confounding",
            CausalDAG::new(
                vec!["X".into(), "M".into(), "Y".into(), "C".into()],
                vec![(0, 3), (3, 1), (3, 2), (1, 2)],
            ),
            0,
            1,
            2,
            SemanticClass::MediationExposureConfounded,
            &mut episodes,
            &mut receipts,
            &mut failures,
            &mut asserted,
        )?;

        let descriptor = CapabilityLaneDescriptor {
            lane_id: "causal-identifiability-development-v1".into(),
            domain: ReasoningDomain::Causal,
            benchmark: BENCHMARK.into(),
            benchmark_version: BENCHMARK_VERSION.into(),
            split: SPLIT.into(),
            holdout_policy: HoldoutPolicy::DevelopmentProbe,
            contamination_policy_id: COMMON_SOURCE.into(),
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
            .map_err(|err| format!("causal lane failed structural qualification: {err}"))?;

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        let encoded = serde_json::to_vec_pretty(&bundle)
            .map_err(|err| format!("failed to encode causal lane bundle: {err}"))?;
        fs::write(&output, encoded)
            .map_err(|err| format!("failed to write {}: {err}", output.display()))?;

        let exact_correct = report.capability.exact_correct;
        let summary = ConsoleSummary {
            subject_revision,
            cases: report.capability.episodes,
            exact_correct,
            asserted,
            abstained,
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
    fn run_query_case(
        subject_revision: &str,
        id: &str,
        dag: CausalDAG,
        query: CausalQuery,
        expected: SemanticClass,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
        failures: &mut Vec<String>,
        asserted: &mut usize,
        abstained: &mut usize,
    ) -> Result<(), String> {
        let input = serde_json::to_vec(&(&dag, &query))
            .map_err(|err| format!("failed to encode query input {id}: {err}"))?;
        let started = Instant::now();
        let actual = CounterfactualReasoner::new().query(&dag, &query);
        let wall_time_us = elapsed_us(started);

        let observation = observe_query(&actual, expected);
        record_case(
            subject_revision,
            id,
            &input,
            expected,
            observation,
            wall_time_us,
            episodes,
            receipts,
            failures,
            asserted,
            abstained,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_iv_case(
        subject_revision: &str,
        id: &str,
        dag: CausalDAG,
        instrument: usize,
        treatment: usize,
        outcome: usize,
        expected: SemanticClass,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
        failures: &mut Vec<String>,
        asserted: &mut usize,
    ) -> Result<(), String> {
        let input = serde_json::to_vec(&(&dag, instrument, treatment, outcome))
            .map_err(|err| format!("failed to encode IV input {id}: {err}"))?;
        let started = Instant::now();
        let actual = IVEstimator::is_valid_instrument(&dag, instrument, treatment, outcome);
        let wall_time_us = elapsed_us(started);
        let actual_class = match actual {
            IVValidity::Valid { .. } => SemanticClass::InstrumentValid,
            IVValidity::Invalid { .. } => SemanticClass::InstrumentInvalid,
        };
        let observation = Observation {
            actual_label: actual_class.label().into(),
            correct: actual_class == expected,
            outcome: ReasoningOutcome::Asserted {
                value: actual_class.label().into(),
                confidence: 0.5,
            },
            operation: format!(
                "IVEstimator::is_valid_instrument => {} (API exposes no calibrated probability; qualification confidence fixed at neutral 0.5)",
                actual_class.label()
            ),
        };
        let mut ignored_abstentions = 0usize;
        record_case(
            subject_revision,
            id,
            &input,
            expected,
            observation,
            wall_time_us,
            episodes,
            receipts,
            failures,
            asserted,
            &mut ignored_abstentions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mediation_case(
        subject_revision: &str,
        id: &str,
        dag: CausalDAG,
        treatment: usize,
        mediator: usize,
        outcome: usize,
        expected: SemanticClass,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
        failures: &mut Vec<String>,
        asserted: &mut usize,
    ) -> Result<(), String> {
        let input = serde_json::to_vec(&(&dag, treatment, mediator, outcome))
            .map_err(|err| format!("failed to encode mediation input {id}: {err}"))?;
        let started = Instant::now();
        let actual = MediationAnalysis::new(&dag, treatment, mediator, outcome).is_identified();
        let wall_time_us = elapsed_us(started);
        let actual_class = match actual {
            MediationIdentification::Identified { .. } => SemanticClass::MediationIdentified,
            MediationIdentification::NotMediator { .. } => SemanticClass::MediationNotMediator,
            MediationIdentification::ExposureInducedConfounding { .. } => {
                SemanticClass::MediationExposureConfounded
            }
        };
        let observation = Observation {
            actual_label: actual_class.label().into(),
            correct: actual_class == expected,
            outcome: ReasoningOutcome::Asserted {
                value: actual_class.label().into(),
                confidence: 0.5,
            },
            operation: format!(
                "MediationAnalysis::is_identified => {} (API exposes no calibrated probability; qualification confidence fixed at neutral 0.5)",
                actual_class.label()
            ),
        };
        let mut ignored_abstentions = 0usize;
        record_case(
            subject_revision,
            id,
            &input,
            expected,
            observation,
            wall_time_us,
            episodes,
            receipts,
            failures,
            asserted,
            &mut ignored_abstentions,
        )
    }

    fn observe_query(actual: &CausalQueryOutcome, expected: SemanticClass) -> Observation {
        match actual {
            CausalQueryOutcome::Identified {
                estimand,
                method,
                confidence,
            } => {
                let actual_class = if expected == SemanticClass::IdentifiedZero
                    && estimand.effect.abs() <= f64::EPSILON
                    && matches!(method, IdentificationMethod::DSeparation)
                {
                    SemanticClass::IdentifiedZero
                } else if matches!(method, IdentificationMethod::BackdoorAdjustment) {
                    SemanticClass::IdentifiedBackdoor
                } else {
                    SemanticClass::Identified
                };
                let correct = match expected {
                    SemanticClass::Identified => matches!(
                        actual_class,
                        SemanticClass::Identified | SemanticClass::IdentifiedBackdoor
                    ),
                    SemanticClass::IdentifiedBackdoor => {
                        actual_class == SemanticClass::IdentifiedBackdoor
                            && estimand.adjustment_set.contains(&2)
                    }
                    SemanticClass::IdentifiedZero => actual_class == SemanticClass::IdentifiedZero,
                    _ => false,
                };
                Observation {
                    actual_label: format!(
                        "{}:{method:?}:adjust={:?}",
                        actual_class.label(),
                        estimand.adjustment_set
                    ),
                    correct,
                    outcome: ReasoningOutcome::Asserted {
                        value: actual_class.label().into(),
                        confidence: confidence.clamp(0.0, 1.0),
                    },
                    operation: format!(
                        "CounterfactualReasoner::query => Identified via {method:?}, adjustment={:?}",
                        estimand.adjustment_set
                    ),
                }
            }
            CausalQueryOutcome::Unidentified { reason, .. } => {
                let correct = expected == SemanticClass::UnidentifiedDagTooLarge
                    && matches!(reason, UnidentifiedReason::DagTooLarge { .. });
                Observation {
                    actual_label: format!("unidentified:{reason:?}"),
                    correct,
                    outcome: ReasoningOutcome::Abstained {
                        reason: AbstentionReason::Unidentified,
                        answerability: 0.0,
                    },
                    operation: format!("CounterfactualReasoner::query => Unidentified({reason:?})"),
                }
            }
            CausalQueryOutcome::AssumptionRequired {
                assumption,
                plausibility,
                ..
            } => Observation {
                actual_label: format!("assumption-required:{assumption:?}"),
                correct: false,
                outcome: ReasoningOutcome::Abstained {
                    reason: AbstentionReason::InsufficientEvidence,
                    answerability: plausibility.clamp(0.0, 1.0),
                },
                operation: format!(
                    "CounterfactualReasoner::query => AssumptionRequired({assumption:?})"
                ),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_case(
        subject_revision: &str,
        id: &str,
        input: &[u8],
        expected: SemanticClass,
        observation: Observation,
        wall_time_us: u64,
        episodes: &mut Vec<ReasoningEpisode>,
        receipts: &mut Vec<ReasoningQualificationReceipt>,
        failures: &mut Vec<String>,
        asserted: &mut usize,
        abstained: &mut usize,
    ) -> Result<(), String> {
        let input_hash = blake3::hash(input).to_hex().to_string();
        let is_asserted = matches!(&observation.outcome, ReasoningOutcome::Asserted { .. });
        if is_asserted {
            *asserted += 1;
        } else {
            *abstained += 1;
        }
        if !observation.correct {
            failures.push(format!(
                "{id}: expected {}, got {}",
                expected.label(), observation.actual_label
            ));
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
                problem_hash: format!("blake3:{input_hash}"),
            },
            vec![EvidenceRef {
                id: format!("{id}:input"),
                content_hash: format!("blake3:{input_hash}"),
                provenance: "RQ-005 public semantic fixture input; oracle withheld from subject call"
                    .into(),
                independence_group: Some(COMMON_SOURCE.into()),
            }],
            vec![],
            vec![ReasoningDecisionRecord {
                operation: observation.operation,
                input_refs: vec![format!("{id}:input")],
                output_refs: if is_asserted {
                    vec![format!("{id}:semantic-output")]
                } else {
                    vec![]
                },
                verifier: Some("rq-005-semantic-oracle-v1".into()),
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
            "rq-005:v1:{id}:{}",
            blake3::hash(expected.label().as_bytes()).to_hex()
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
        eprintln!("causal identifiability qualification failed: {err}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("benchmark_causal_identifiability requires --features reasoning_engine");
}
