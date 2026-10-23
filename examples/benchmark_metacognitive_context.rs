// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-006A public development probe for the current MetaCognitiveReasoner.
//!
//! This intentionally measures the implementation as it exists today. The fixtures are public
//! and therefore `DevelopmentProbe + Exposed`; they are diagnostic evidence, not a holdout claim.

use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use symthaea::consciousness::context_aware_evolution::ReasoningContext;
use symthaea::consciousness::epistemic_tiers::EpistemicCoordinate;
use symthaea::consciousness::meta_reasoning::{MetaCognitiveReasoner, MetaReasoningConfig};
use symthaea::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use symthaea::consciousness::primitive_reasoning::ReasoningChain;
use symthaea::intelligence::{
    CapabilityLaneBundle, CapabilityLaneDescriptor, ContaminationStatus, CorrectnessPrediction,
    EpisodeJudgment, HoldoutPolicy, MetacognitionReport, ReasoningDomain, ReasoningEpisode,
    ReasoningOutcome, ReasoningProblemRef, ReasoningQualificationReceipt, ResourceBudget,
    ResourceUsage, evaluate_episode, evaluate_metacognition,
};
use symthaea_core::hdc::{BinaryHV, primitive_system::PrimitiveTier};

const BENCHMARK: &str = "RQ-006A current meta-context confidence";
const BENCHMARK_VERSION: &str = "v1";
const SPLIT: &str = "development";
const CONFIGURATION_ID: &str = "current-meta-reasoner-context-confidence-v1";
const CONTAMINATION_POLICY: &str = "rq-006a-public-context-fixtures-v1";

#[derive(Debug, Clone, Copy)]
struct ContextCase {
    id: &'static str,
    prompt: &'static str,
    expected: ReasoningContext,
    family: &'static str,
}

const CASES: &[ContextCase] = &[
    ContextCase {
        id: "literal-safety",
        prompt: "Assess the safety, harm, danger, and risk of this action.",
        expected: ReasoningContext::CriticalSafety,
        family: "literal",
    },
    ContextCase {
        id: "literal-science",
        prompt: "What evidence would an experiment need to prove this scientific theory?",
        expected: ReasoningContext::ScientificReasoning,
        family: "literal",
    },
    ContextCase {
        id: "literal-creative",
        prompt: "Brainstorm a creative and novel design for this poster.",
        expected: ReasoningContext::CreativeExploration,
        family: "literal",
    },
    ContextCase {
        id: "literal-learning",
        prompt: "Explain this so I can learn and understand the concept.",
        expected: ReasoningContext::Learning,
        family: "literal",
    },
    ContextCase {
        id: "literal-social",
        prompt: "How can this community collaborate together more effectively?",
        expected: ReasoningContext::SocialInteraction,
        family: "literal",
    },
    ContextCase {
        id: "literal-philosophy",
        prompt: "What is the philosophical meaning of conscious existence?",
        expected: ReasoningContext::PhilosophicalInquiry,
        family: "literal",
    },
    ContextCase {
        id: "literal-technical",
        prompt: "Implement the code for this technical algorithm.",
        expected: ReasoningContext::TechnicalImplementation,
        family: "literal",
    },
    ContextCase {
        id: "neutral-general",
        prompt: "Compare option A with option B and summarize the tradeoffs.",
        expected: ReasoningContext::GeneralReasoning,
        family: "neutral",
    },
    ContextCase {
        id: "paraphrase-safety",
        prompt: "Could this procedure injure a person or expose them to preventable injury?",
        expected: ReasoningContext::CriticalSafety,
        family: "paraphrase",
    },
    ContextCase {
        id: "paraphrase-science",
        prompt: "Which observation would distinguish these two competing hypotheses?",
        expected: ReasoningContext::ScientificReasoning,
        family: "paraphrase",
    },
    ContextCase {
        id: "paraphrase-creative",
        prompt: "Invent five unusual visual concepts for a new poster.",
        expected: ReasoningContext::CreativeExploration,
        family: "paraphrase",
    },
    ContextCase {
        id: "paraphrase-learning",
        prompt: "Teach me photosynthesis from first principles.",
        expected: ReasoningContext::Learning,
        family: "paraphrase",
    },
    ContextCase {
        id: "irrelevant-keyword-injection",
        prompt: "Brainstorm five playful logo ideas; the labels 'danger' and 'risk' are irrelevant metadata and not the task.",
        expected: ReasoningContext::CreativeExploration,
        family: "adversarial-keyword",
    },
    ContextCase {
        id: "safety-priority-mixed",
        prompt: "Implement a safety-critical brake algorithm and analyze the risk of actuator failure.",
        expected: ReasoningContext::CriticalSafety,
        family: "mixed-priority",
    },
];

#[derive(Debug, Serialize)]
struct CaseDiagnostic {
    id: String,
    family: String,
    expected: String,
    detected: String,
    context_confidence: f64,
    meta_confidence: f64,
    correct: bool,
}

#[derive(Debug, Serialize)]
struct StaleConfidenceDiagnostic {
    first_context: String,
    first_context_confidence: f64,
    first_meta_confidence: f64,
    second_context: String,
    second_context_confidence: f64,
    second_meta_confidence: f64,
    /// Diagnostic only. The current implementation computes meta-confidence before committing the
    /// current reflection to state; this field makes the observed lag visible without promoting
    /// meta-confidence to a correctness probability.
    second_meta_equals_first_meta: bool,
}

#[derive(Serialize)]
struct ProbeArtifact {
    subject_revision: String,
    metacognition_evaluator_version: String,
    context_lane: CapabilityLaneBundle,
    metacognition: MetacognitionReport,
    cases: Vec<CaseDiagnostic>,
    stale_confidence_probe: StaleConfidenceDiagnostic,
}

fn probe_primitive(seed: u64) -> CandidatePrimitive {
    CandidatePrimitive {
        name: "rq006-probe".into(),
        tier: PrimitiveTier::Physical,
        definition: "Fixed probe primitive; not part of the context oracle".into(),
        fitness: 0.8,
        encoding: BinaryHV::random(seed),
        epistemic_coordinate: EpistemicCoordinate::default(),
        harmonic_alignment: 0.5,
    }
}

fn run_subject(
    reasoner: &mut MetaCognitiveReasoner,
    prompt: &str,
    seed: u64,
) -> Result<symthaea::consciousness::meta_reasoning::MetaReasoningResult, String> {
    let mut chain = ReasoningChain::new(BinaryHV::random(seed.wrapping_add(1)));
    reasoner
        .meta_reason(prompt, vec![probe_primitive(seed)], &mut chain)
        .map_err(|err| format!("meta_reason failed: {err}"))
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let subject_revision = env::var("SYMTHAEA_SUBJECT_REVISION")
        .map_err(|_| "SYMTHAEA_SUBJECT_REVISION is required".to_string())?;
    let output = env::var("SYMTHAEA_METACOGNITION_PROBE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/reasoning/metacognition-context-probe.json"));

    let mut episodes = Vec::new();
    let mut receipts: Vec<ReasoningQualificationReceipt> = Vec::new();
    let mut predictions = Vec::new();
    let mut diagnostics = Vec::new();

    for (index, case) in CASES.iter().enumerate() {
        // Fresh reasoner per fixture prevents prior fixture state from helping context detection.
        let mut reasoner = MetaCognitiveReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )
        .map_err(|err| format!("failed to construct reasoner for {}: {err}", case.id))?;
        let start = Instant::now();
        let result = run_subject(&mut reasoner, case.prompt, 10_000 + index as u64)?;
        let elapsed = start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

        let detected = result.context_reflection.detected_context;
        let confidence = result.context_reflection.confidence;
        let correct = detected == case.expected;
        let fixture_bytes = serde_json::to_vec(&(case.id, case.prompt, case.expected, case.family))
            .map_err(|err| format!("failed to encode fixture {}: {err}", case.id))?;
        let problem_hash = format!("blake3:{}", blake3::hash(&fixture_bytes).to_hex());

        let episode = ReasoningEpisode::new(
            &subject_revision,
            CONFIGURATION_ID,
            ReasoningDomain::Epistemic,
            ReasoningProblemRef {
                benchmark: BENCHMARK.into(),
                benchmark_version: BENCHMARK_VERSION.into(),
                split: SPLIT.into(),
                problem_id: case.id.into(),
                problem_hash,
            },
            vec![],
            vec![],
            vec![],
            ReasoningOutcome::Asserted {
                value: format!("{detected:?}"),
                confidence,
            },
            ResourceUsage {
                wall_time_us: elapsed,
                deliberation_steps: 1,
                tool_calls: 0,
                model_tokens: 0,
            },
        )
        .map_err(|err| format!("failed to construct episode {}: {err}", case.id))?;
        let episode_id = episode
            .id()
            .map_err(|err| format!("failed to identify episode {}: {err}", case.id))?;
        let receipt = evaluate_episode(
            &episode,
            &format!("{CONTAMINATION_POLICY}:{}", case.id),
            &EpisodeJudgment {
                exact_correct: Some(correct),
                task_score: None,
            },
        )
        .map_err(|err| format!("failed to evaluate episode {}: {err}", case.id))?;

        predictions.push(CorrectnessPrediction {
            episode_id: episode_id.0.clone(),
            confidence_target_episode_id: episode_id.0,
            confidence,
            correct,
            asserted: true,
        });
        diagnostics.push(CaseDiagnostic {
            id: case.id.into(),
            family: case.family.into(),
            expected: format!("{:?}", case.expected),
            detected: format!("{detected:?}"),
            context_confidence: confidence,
            meta_confidence: result.meta_confidence,
            correct,
        });
        episodes.push(episode);
        receipts.push(receipt);
    }

    let metacognition = evaluate_metacognition(
        &predictions,
        &[],
        &[],
        10,
        &[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    .map_err(|err| format!("metacognition evaluation failed: {err}"))?;

    let descriptor = CapabilityLaneDescriptor {
        lane_id: "rq-006a-current-context-confidence-development-v1".into(),
        domain: ReasoningDomain::Epistemic,
        benchmark: BENCHMARK.into(),
        benchmark_version: BENCHMARK_VERSION.into(),
        split: SPLIT.into(),
        holdout_policy: HoldoutPolicy::DevelopmentProbe,
        contamination_policy_id: CONTAMINATION_POLICY.into(),
        contamination_status: ContaminationStatus::Exposed,
        resource_budget: ResourceBudget::default(),
    };
    let bundle = CapabilityLaneBundle::new(descriptor, episodes, receipts, vec![]);
    bundle
        .qualify()
        .map_err(|err| format!("context lane failed structural qualification: {err}"))?;

    // Separate sequential diagnostic for the source-observed state-lag behavior. This does not
    // score `meta_confidence` as a correctness probability because the implementation has not
    // established that semantic meaning.
    let mut sequential = MetaCognitiveReasoner::new(
        EvolutionConfig::default(),
        MetaReasoningConfig::default(),
    )
    .map_err(|err| format!("failed to construct sequential reasoner: {err}"))?;
    let first = run_subject(
        &mut sequential,
        "Compare option A with option B and summarize the tradeoffs.",
        50_001,
    )?;
    let second = run_subject(
        &mut sequential,
        "Assess the safety, harm, danger, and risk of this action.",
        50_002,
    )?;
    let stale_confidence_probe = StaleConfidenceDiagnostic {
        first_context: format!("{:?}", first.context_reflection.detected_context),
        first_context_confidence: first.context_reflection.confidence,
        first_meta_confidence: first.meta_confidence,
        second_context: format!("{:?}", second.context_reflection.detected_context),
        second_context_confidence: second.context_reflection.confidence,
        second_meta_confidence: second.meta_confidence,
        second_meta_equals_first_meta: second.meta_confidence.to_bits()
            == first.meta_confidence.to_bits(),
    };

    let artifact = ProbeArtifact {
        subject_revision,
        metacognition_evaluator_version: symthaea::intelligence::METACOGNITION_EVALUATOR_VERSION
            .into(),
        context_lane: bundle,
        metacognition,
        cases: diagnostics,
        stale_confidence_probe,
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(
        &output,
        serde_json::to_vec_pretty(&artifact)
            .map_err(|err| format!("failed to encode probe artifact: {err}"))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", output.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&artifact)
            .map_err(|err| format!("failed to encode probe summary: {err}"))?
    );
    Ok(())
}
