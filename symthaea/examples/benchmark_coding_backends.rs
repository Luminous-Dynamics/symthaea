// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Small coding-backend benchmark harness.
//!
//! Run:
//!   cargo run --example benchmark_coding_backends --features code_generation
//!   cargo run --example benchmark_coding_backends --features code_generation,geodesic_synthesis -- --json

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;
use symthaea::language::code_orchestrator::{CodeOrchestrator, CodingAgentRuntimePolicy};
use symthaea::language::repair_taxonomy::{
    FORCE_REPAIR_BENCH_ENV, categorize_rejection, extract_embedded_category,
    repair_lesson_for_rejection,
};
use symthaea::language::rust_ast_hdc::{ast_feature_cosine_similarity, ast_feature_l1_distance};
use symthaea::language::structural_prototype::{
    StructuralPrototypeBank, StructuralPrototypeLabels, ast_features_for_source,
    return_shape_for_signature,
};
use symthaea_core::synthesis_trait::SynthesisRequest;

#[derive(Debug, Clone)]
struct BenchTask {
    lane: &'static str,
    id: &'static str,
    category: &'static str,
    name: &'static str,
    purpose: &'static str,
    signature: &'static str,
    examples: &'static [(&'static str, &'static str)],
    constraints: &'static [&'static str],
}

#[derive(Debug, Serialize)]
struct TaskReport {
    id: String,
    lane: String,
    category: String,
    accepted: bool,
    quality_gate_passed: bool,
    confidence: f32,
    backend_name: String,
    elapsed_ms: u128,
    attempts: BTreeMap<String, usize>,
    rejection_categories: BTreeMap<String, usize>,
    rejections: Vec<AttemptRejectionReport>,
    repair_lessons: Vec<RepairLessonReport>,
    repair_attempt_count: usize,
    repair_successful: bool,
    successful_backend_after_repair: Option<String>,
    repair_priors_seen: BTreeMap<String, usize>,
    repair_prior_labels_seen: Vec<String>,
    prediction_errors_seen: usize,
    prediction_error_categories: BTreeMap<String, usize>,
    prediction_error_hinted_retry_successful: bool,
    surprise_before_retry: Option<f32>,
    surprise_after_retry: Option<f32>,
    ast_hdc_parse_successes: usize,
    ast_hdc_parse_failures: usize,
    structural_prediction_errors: usize,
    mean_ast_feature_count: Option<f32>,
    structural_repair_similarity: Option<f32>,
    structural_repair_l1_delta: Option<usize>,
    structural_prior_score: Option<f32>,
    structural_prior_label: Option<String>,
    structural_prior_delta: Option<f32>,
    attempt_count: usize,
    certificate_backend: Option<String>,
    certificate_source_provenance: Option<String>,
    certificate_has_topology: bool,
    certificate_has_oracle: bool,
    certificate_has_sheaf: bool,
    certificate_sheaf_coherent: Option<bool>,
    topology_beta_1: Option<usize>,
    oracle_convergence: Option<f32>,
}

#[derive(Debug, Serialize)]
struct AttemptRejectionReport {
    backend: String,
    category: String,
    reason: String,
    source_preview: Option<String>,
    repair_prior_count: usize,
    repair_prior_labels: Vec<String>,
    surprise: f32,
    diagnostic_hv_count: usize,
    ast_hdc_parse_successes: usize,
    ast_hdc_parse_failures: usize,
    structural_prediction_errors: usize,
    ast_hdc_feature_count: usize,
    ast_hdc_last_features: Option<BTreeMap<String, usize>>,
    structural_prior_score: Option<f32>,
    structural_prior_label: Option<String>,
}

#[derive(Debug, Serialize)]
struct RepairLessonReport {
    task_id: String,
    task_name: String,
    signature: String,
    backend: String,
    category: String,
    diagnostic: String,
    hint: String,
    source_preview: Option<String>,
    fixed_source_preview: Option<String>,
    final_outcome: String,
    broca_training_record: bool,
    prediction_error_training_record: bool,
    prediction_error_hv_count: usize,
    surprise_before_retry: Option<f32>,
    surprise_after_retry: Option<f32>,
    broken_ast_features: Option<BTreeMap<String, usize>>,
    fixed_ast_features: Option<BTreeMap<String, usize>>,
    structural_similarity: Option<f32>,
    structural_l1_delta: Option<usize>,
    broken_structural_prior_score: Option<f32>,
    fixed_structural_prior_score: Option<f32>,
    structural_prior_delta: Option<f32>,
    structural_prior_label: Option<String>,
    final_backend: String,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    benchmark: String,
    feature_geodesic: bool,
    task_count: usize,
    accepted_count: usize,
    quality_pass_count: usize,
    pass_rate: f32,
    quality_pass_rate: f32,
    elapsed_ms: u128,
    backend_attempts: BTreeMap<String, usize>,
    rejection_categories: BTreeMap<String, usize>,
    repair_lesson_categories: BTreeMap<String, usize>,
    repair_attempts: usize,
    repair_successes: usize,
    repair_success_rate: f32,
    success_after_hint_by_category: BTreeMap<String, usize>,
    first_successful_backend_after_repair: BTreeMap<String, usize>,
    repair_prior_counts_by_backend: BTreeMap<String, usize>,
    repair_prior_labels: BTreeMap<String, usize>,
    repair_prior_uses: usize,
    repair_prior_label_count: usize,
    repair_hinted_attempts: usize,
    repair_hinted_successes: usize,
    repair_hinted_success_rate: f32,
    repair_memory_hits: usize,
    repair_memory_successes: usize,
    repair_memory_success_rate: f32,
    repair_memory_categories_used: BTreeMap<String, usize>,
    prediction_error_repair_hints_enabled: bool,
    ast_hdc_fep_enabled: bool,
    prediction_errors_seen: usize,
    prediction_error_categories: BTreeMap<String, usize>,
    prediction_error_hinted_retry_tasks: usize,
    prediction_error_hinted_retry_successes: usize,
    prediction_error_hinted_retry_success_rate: f32,
    mean_surprise_before_retry: Option<f32>,
    mean_surprise_after_retry: Option<f32>,
    ast_hdc_parse_successes: usize,
    ast_hdc_parse_failures: usize,
    structural_prediction_errors: usize,
    mean_ast_feature_count: Option<f32>,
    mean_structural_repair_similarity: Option<f32>,
    mean_structural_repair_l1_delta: Option<f32>,
    structural_success_prototypes: usize,
    structural_prior_observations: usize,
    mean_structural_prior_score: Option<f32>,
    mean_structural_prior_delta: Option<f32>,
    distillation_import_path: Option<String>,
    distillation_imported: usize,
    distillation_export_path: Option<String>,
    distillation_exported: usize,
    structural_prototype_import_path: Option<String>,
    structural_prototype_imported: bool,
    structural_prototype_export_path: Option<String>,
    certificate_source_provenance_counts: BTreeMap<String, usize>,
    broca_eval_gate_passed: bool,
    broca_selection_score: f32,
    category_pass_rates: BTreeMap<String, CategoryReport>,
    certificates_with_topology: usize,
    certificates_with_oracle: usize,
    certificates_with_sheaf: usize,
    certificates_sheaf_coherent: usize,
    certificates_sheaf_incoherent: usize,
    mean_attempts_per_task: f32,
    tasks: Vec<TaskReport>,
}

#[derive(Debug, Default, Serialize)]
struct CategoryReport {
    task_count: usize,
    accepted_count: usize,
    pass_rate: f32,
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime");
    let _guard = runtime.enter();
    run_benchmark();
}

fn run_benchmark() {
    let args = Args::parse();
    if args.disable_fep_repair_hints {
        unsafe {
            std::env::set_var("SYMTHAEA_DISABLE_FEP_REPAIR_HINTS", "1");
        }
    }
    if args.disable_ast_hdc_fep {
        unsafe {
            std::env::set_var("SYMTHAEA_DISABLE_AST_HDC_FEP", "1");
        }
    }
    if matches!(args.lane.as_str(), "repair" | "all")
        && std::env::var_os(FORCE_REPAIR_BENCH_ENV).is_none()
    {
        eprintln!(
            "warning: repair lane selected without {FORCE_REPAIR_BENCH_ENV}=1; forced repair probes will be disabled"
        );
    }
    let start = Instant::now();
    let mut orch = CodeOrchestrator::new();
    if args.simulated_llm {
        orch = orch.with_llm_backend(symthaea::language::llm_backend::simulated_backend());
    }
    if let Some(budget) = args.energy_budget {
        orch = orch.with_energy_budget(budget);
    }
    if let Some(path) = args.runtime_policy_json.as_deref() {
        match CodingAgentRuntimePolicy::from_file(std::path::Path::new(path)) {
            Ok(policy) => {
                eprintln!("[benchmark] loaded runtime coding-agent policy from {path}");
                orch = orch.with_runtime_policy(policy);
            }
            Err(error) => {
                eprintln!("error: failed to load runtime coding-agent policy from {path}: {error}");
                std::process::exit(2);
            }
        }
    }
    let distillation_imported = if let Some(path) = args.load_distillation_jsonl.as_deref() {
        match orch.load_distillation(std::path::Path::new(path)) {
            Ok(count) => {
                eprintln!(
                    "[benchmark] imported {count} distillation structural memories from {path}"
                );
                count
            }
            Err(error) => {
                eprintln!("error: failed to load distillation JSONL from {path}: {error}");
                std::process::exit(2);
            }
        }
    } else {
        0
    };
    let mut tasks = Vec::new();
    let mut backend_attempts = BTreeMap::new();
    let mut rejection_categories = BTreeMap::new();
    let mut repair_lesson_categories = BTreeMap::new();
    let mut success_after_hint_by_category = BTreeMap::new();
    let mut first_successful_backend_after_repair = BTreeMap::new();
    let mut repair_prior_counts_by_backend = BTreeMap::new();
    let mut repair_prior_labels = BTreeMap::new();
    let selected_tasks = tasks_for_lane(&args.lane);
    let mut structural_prototypes = if let Some(path) = args.load_structural_prototypes.as_deref() {
        match std::fs::read_to_string(path)
            .map_err(anyhow::Error::from)
            .and_then(|json| StructuralPrototypeBank::from_json(&json).map_err(anyhow::Error::from))
        {
            Ok(bank) => {
                eprintln!("[benchmark] loaded compact structural prototypes from {path}");
                bank
            }
            Err(error) => {
                eprintln!(
                    "error: failed to load compact structural prototypes from {path}: {error}"
                );
                std::process::exit(2);
            }
        }
    } else {
        StructuralPrototypeBank::default()
    };
    let structural_prototype_imported = args.load_structural_prototypes.is_some();

    for task in selected_tasks {
        let return_shape = return_shape_for_signature(task.signature);
        let before_attempts = orch.attempt_history().len();
        let before_certs = orch.certificates().len();
        let task_start = Instant::now();
        let mut request =
            SynthesisRequest::new("rust", task.name, task.purpose).with_signature(task.signature);
        for (input, output) in task.examples {
            request = request.with_example(*input, *output);
        }
        for constraint in task.constraints {
            request = request.with_constraint(*constraint);
        }

        let response = orch.synthesize(&request);
        let response_ast_features = response
            .accepted
            .then(|| ast_features_for_source(&response.source))
            .flatten();
        let response_backend_name = response.backend_name.clone();
        let response_structural_prior = response_ast_features.as_ref().and_then(|features| {
            structural_prototypes.score(
                features,
                &StructuralPrototypeLabels::new(
                    task.category,
                    return_shape.clone(),
                    response_backend_name.clone(),
                ),
            )
        });
        let attempts = orch.attempt_history();
        let new_attempts = &attempts[before_attempts..];
        let mut task_attempts = BTreeMap::new();
        let mut task_rejections = BTreeMap::new();
        let mut detailed_rejections = Vec::new();
        let mut repair_lessons = Vec::new();
        let mut task_repair_priors_seen = BTreeMap::new();
        let mut task_repair_prior_labels_seen = Vec::new();
        let mut task_prediction_error_categories = BTreeMap::new();
        let mut task_structural_repair_similarities = Vec::new();
        let mut task_structural_repair_l1_deltas = Vec::new();
        let mut task_structural_prior_deltas = Vec::new();
        let task_prediction_errors_seen = new_attempts
            .iter()
            .map(|attempt| attempt.diagnostic_hv_count)
            .sum::<usize>();
        let task_ast_hdc_parse_successes = new_attempts
            .iter()
            .map(|attempt| attempt.ast_hdc_parse_successes)
            .sum::<usize>();
        let task_ast_hdc_parse_failures = new_attempts
            .iter()
            .map(|attempt| attempt.ast_hdc_parse_failures)
            .sum::<usize>();
        let task_structural_prediction_errors = new_attempts
            .iter()
            .map(|attempt| attempt.structural_prediction_errors)
            .sum::<usize>();
        let task_ast_feature_counts = new_attempts
            .iter()
            .filter_map(|attempt| {
                (attempt.ast_hdc_feature_count > 0).then_some(attempt.ast_hdc_feature_count as f32)
            })
            .collect::<Vec<_>>();
        let task_mean_ast_feature_count = mean_optional(&task_ast_feature_counts);
        let surprise_before_retry = new_attempts
            .iter()
            .find(|attempt| attempt.diagnostic_hv_count > 0 || attempt.surprise > 0.0)
            .map(|attempt| attempt.surprise);
        let surprise_after_retry = new_attempts.last().map(|attempt| attempt.surprise);
        for attempt in new_attempts {
            *task_attempts.entry(attempt.backend.clone()).or_insert(0) += 1;
            *backend_attempts.entry(attempt.backend.clone()).or_insert(0) += 1;
            if attempt.repair_prior_count > 0 {
                *task_repair_priors_seen
                    .entry(attempt.backend.clone())
                    .or_insert(0) += attempt.repair_prior_count;
                *repair_prior_counts_by_backend
                    .entry(attempt.backend.clone())
                    .or_insert(0) += attempt.repair_prior_count;
                for label in &attempt.repair_prior_labels {
                    task_repair_prior_labels_seen.push(label.clone());
                    *repair_prior_labels.entry(label.clone()).or_insert(0) += 1;
                }
            }
            if let Some(reason) = &attempt.rejection_reason {
                let attempt_structural_prior =
                    attempt.ast_hdc_last_features.as_ref().and_then(|features| {
                        structural_prototypes.score(
                            features,
                            &StructuralPrototypeLabels::new(
                                task.category,
                                return_shape.clone(),
                                attempt.backend.clone(),
                            ),
                        )
                    });
                let category = extract_embedded_category(reason)
                    .unwrap_or_else(|| categorize_rejection(reason))
                    .to_string();
                *task_rejections.entry(category.clone()).or_insert(0) += 1;
                *rejection_categories.entry(category.clone()).or_insert(0) += 1;
                detailed_rejections.push(AttemptRejectionReport {
                    backend: attempt.backend.clone(),
                    category: category.clone(),
                    reason: reason.clone(),
                    source_preview: attempt.source_preview.clone(),
                    repair_prior_count: attempt.repair_prior_count,
                    repair_prior_labels: attempt.repair_prior_labels.clone(),
                    surprise: attempt.surprise,
                    diagnostic_hv_count: attempt.diagnostic_hv_count,
                    ast_hdc_parse_successes: attempt.ast_hdc_parse_successes,
                    ast_hdc_parse_failures: attempt.ast_hdc_parse_failures,
                    structural_prediction_errors: attempt.structural_prediction_errors,
                    ast_hdc_feature_count: attempt.ast_hdc_feature_count,
                    ast_hdc_last_features: attempt.ast_hdc_last_features.clone(),
                    structural_prior_score: attempt_structural_prior
                        .as_ref()
                        .map(|prior| prior.score),
                    structural_prior_label: attempt_structural_prior
                        .as_ref()
                        .map(|prior| prior.label.clone()),
                });
                if attempt.diagnostic_hv_count > 0 {
                    *task_prediction_error_categories
                        .entry(category.clone())
                        .or_insert(0) += attempt.diagnostic_hv_count;
                }
                let lesson = repair_lesson_for_rejection(reason);
                let structural_similarity = attempt
                    .ast_hdc_last_features
                    .as_ref()
                    .zip(response_ast_features.as_ref())
                    .and_then(|(broken, fixed)| ast_feature_cosine_similarity(broken, fixed));
                let structural_l1_delta = attempt
                    .ast_hdc_last_features
                    .as_ref()
                    .zip(response_ast_features.as_ref())
                    .map(|(broken, fixed)| ast_feature_l1_distance(broken, fixed));
                if let Some(similarity) = structural_similarity {
                    task_structural_repair_similarities.push(similarity);
                }
                if let Some(delta) = structural_l1_delta {
                    task_structural_repair_l1_deltas.push(delta as f32);
                }
                let broken_structural_prior_score =
                    attempt_structural_prior.as_ref().map(|prior| prior.score);
                let fixed_structural_prior_score =
                    response_structural_prior.as_ref().map(|prior| prior.score);
                let structural_prior_delta = broken_structural_prior_score
                    .zip(fixed_structural_prior_score)
                    .map(|(broken, fixed)| fixed - broken);
                if let Some(delta) = structural_prior_delta {
                    task_structural_prior_deltas.push(delta);
                }
                let structural_prior_label = response_structural_prior
                    .as_ref()
                    .or(attempt_structural_prior.as_ref())
                    .map(|prior| prior.label.clone());
                *repair_lesson_categories
                    .entry(category.clone())
                    .or_insert(0) += 1;
                repair_lessons.push(RepairLessonReport {
                    task_id: task.id.to_string(),
                    task_name: task.name.to_string(),
                    signature: task.signature.to_string(),
                    backend: attempt.backend.clone(),
                    category,
                    diagnostic: reason.clone(),
                    hint: lesson,
                    source_preview: attempt.source_preview.clone(),
                    fixed_source_preview: if response.accepted {
                        Some(preview_source(&response.source))
                    } else {
                        None
                    },
                    final_outcome: if response.accepted {
                        "repaired_or_bypassed".to_string()
                    } else {
                        "unresolved".to_string()
                    },
                    broca_training_record: response.accepted,
                    prediction_error_training_record: response.accepted
                        && attempt.diagnostic_hv_count > 0,
                    prediction_error_hv_count: attempt.diagnostic_hv_count,
                    surprise_before_retry,
                    surprise_after_retry,
                    broken_ast_features: attempt.ast_hdc_last_features.clone(),
                    fixed_ast_features: response_ast_features.clone(),
                    structural_similarity,
                    structural_l1_delta,
                    broken_structural_prior_score,
                    fixed_structural_prior_score,
                    structural_prior_delta,
                    structural_prior_label,
                    final_backend: response_backend_name.clone(),
                });
            }
        }
        let repair_attempt_count = detailed_rejections.len();
        let repair_successful = response.accepted && repair_attempt_count > 0;
        let prediction_error_hinted_retry_successful =
            response.accepted && task_prediction_errors_seen > 0 && repair_attempt_count > 0;
        let successful_backend_after_repair =
            repair_successful.then(|| response.backend_name.clone());
        if repair_successful {
            for category in task_rejections.keys() {
                *success_after_hint_by_category
                    .entry(category.clone())
                    .or_insert(0) += 1;
            }
            *first_successful_backend_after_repair
                .entry(response.backend_name.clone())
                .or_insert(0) += 1;
        }

        let cert = orch.certificates().into_iter().nth(before_certs);
        let topology_beta_1 = cert
            .as_ref()
            .and_then(|c| c.topology.as_ref())
            .map(|t| t.beta_1);
        let certificate_sheaf_coherent = cert.as_ref().and_then(|c| c.sheaf_coherent);
        let quality_gate_passed = response.accepted && certificate_sheaf_coherent != Some(false);
        let structural_repair_similarity = mean_optional(&task_structural_repair_similarities);
        let structural_repair_l1_delta =
            mean_optional(&task_structural_repair_l1_deltas).map(|mean| mean.round() as usize);
        let structural_prior_delta = mean_optional(&task_structural_prior_deltas);
        let observed_response_features = response_ast_features.clone();
        let repair_categories_for_prototype = task_rejections.keys().cloned().collect::<Vec<_>>();
        tasks.push(TaskReport {
            id: task.id.to_string(),
            lane: task.lane.to_string(),
            category: task.category.to_string(),
            accepted: response.accepted,
            quality_gate_passed,
            confidence: response.confidence,
            backend_name: response_backend_name.clone(),
            elapsed_ms: task_start.elapsed().as_millis(),
            attempts: task_attempts,
            rejection_categories: task_rejections,
            rejections: detailed_rejections,
            repair_lessons,
            repair_attempt_count,
            repair_successful,
            successful_backend_after_repair,
            repair_priors_seen: task_repair_priors_seen,
            repair_prior_labels_seen: task_repair_prior_labels_seen,
            prediction_errors_seen: task_prediction_errors_seen,
            prediction_error_categories: task_prediction_error_categories,
            prediction_error_hinted_retry_successful,
            surprise_before_retry,
            surprise_after_retry,
            ast_hdc_parse_successes: task_ast_hdc_parse_successes,
            ast_hdc_parse_failures: task_ast_hdc_parse_failures,
            structural_prediction_errors: task_structural_prediction_errors,
            mean_ast_feature_count: task_mean_ast_feature_count,
            structural_repair_similarity,
            structural_repair_l1_delta,
            structural_prior_score: response_structural_prior.as_ref().map(|prior| prior.score),
            structural_prior_label: response_structural_prior
                .as_ref()
                .map(|prior| prior.label.clone()),
            structural_prior_delta,
            attempt_count: new_attempts.len(),
            certificate_backend: cert.as_ref().map(|c| c.backend_used.clone()),
            certificate_source_provenance: cert.as_ref().map(|c| c.source_provenance.clone()),
            certificate_has_topology: cert.as_ref().and_then(|c| c.topology.as_ref()).is_some(),
            certificate_has_oracle: cert.as_ref().and_then(|c| c.oracle_convergence).is_some(),
            certificate_has_sheaf: cert.as_ref().and_then(|c| c.sheaf_coherent).is_some(),
            certificate_sheaf_coherent,
            topology_beta_1,
            oracle_convergence: cert.as_ref().and_then(|c| c.oracle_convergence),
        });
        if let Some(features) = observed_response_features.as_ref() {
            structural_prototypes.observe_success(
                features,
                &StructuralPrototypeLabels::new(task.category, return_shape, response_backend_name),
            );
            for category in repair_categories_for_prototype {
                structural_prototypes.observe_repair_success(features, &category);
            }
        }
    }

    let accepted_count = tasks.iter().filter(|task| task.accepted).count();
    let quality_pass_count = tasks.iter().filter(|task| task.quality_gate_passed).count();
    let certificates_with_topology = tasks
        .iter()
        .filter(|task| task.certificate_has_topology)
        .count();
    let certificates_with_oracle = tasks
        .iter()
        .filter(|task| task.certificate_has_oracle)
        .count();
    let certificates_with_sheaf = tasks
        .iter()
        .filter(|task| task.certificate_has_sheaf)
        .count();
    let certificates_sheaf_coherent = tasks
        .iter()
        .filter(|task| task.certificate_sheaf_coherent == Some(true))
        .count();
    let certificates_sheaf_incoherent = tasks
        .iter()
        .filter(|task| task.certificate_sheaf_coherent == Some(false))
        .count();
    let mean_attempts_per_task = tasks
        .iter()
        .map(|task| task.attempt_count as f32)
        .sum::<f32>()
        / tasks.len().max(1) as f32;
    let repair_attempts = tasks
        .iter()
        .map(|task| task.repair_attempt_count)
        .sum::<usize>();
    let repair_successes = tasks.iter().filter(|task| task.repair_successful).count();
    let repair_success_rate = repair_successes as f32 / repair_attempts.max(1) as f32;
    let category_pass_rates = category_reports(&tasks);
    let broca_eval_gate_passed = accepted_count == tasks.len()
        && quality_pass_count == tasks.len()
        && certificates_sheaf_incoherent == 0
        && mean_attempts_per_task <= 1.2;
    let broca_selection_score = compute_broca_selection_score(
        accepted_count as f32 / tasks.len().max(1) as f32,
        quality_pass_count as f32 / tasks.len().max(1) as f32,
        mean_attempts_per_task,
        certificates_sheaf_incoherent,
        repair_attempts,
        repair_success_rate,
    );
    let mut certificate_source_provenance_counts = BTreeMap::new();
    for task in &tasks {
        if let Some(provenance) = &task.certificate_source_provenance {
            *certificate_source_provenance_counts
                .entry(provenance.clone())
                .or_insert(0) += 1;
        }
    }
    let repair_prior_uses = repair_prior_counts_by_backend.values().sum();
    let repair_prior_label_count = repair_prior_labels.values().sum();
    let repair_hinted_attempts = tasks
        .iter()
        .filter(|task| !task.repair_priors_seen.is_empty())
        .count();
    let repair_hinted_successes = tasks
        .iter()
        .filter(|task| !task.repair_priors_seen.is_empty() && task.accepted)
        .count();
    let repair_hinted_success_rate =
        repair_hinted_successes as f32 / repair_hinted_attempts.max(1) as f32;
    let repair_memory_hits = tasks
        .iter()
        .filter(|task| task_uses_repair_memory(task))
        .count();
    let repair_memory_successes = tasks
        .iter()
        .filter(|task| task_uses_repair_memory(task) && task.accepted)
        .count();
    let repair_memory_success_rate =
        repair_memory_successes as f32 / repair_memory_hits.max(1) as f32;
    let repair_memory_categories_used = repair_memory_categories(&repair_prior_labels);
    let prediction_errors_seen = tasks
        .iter()
        .map(|task| task.prediction_errors_seen)
        .sum::<usize>();
    let prediction_error_categories = prediction_error_categories(&tasks);
    let prediction_error_hinted_retry_tasks = tasks
        .iter()
        .filter(|task| task.prediction_errors_seen > 0 && task.repair_attempt_count > 0)
        .count();
    let prediction_error_hinted_retry_successes = tasks
        .iter()
        .filter(|task| task.prediction_error_hinted_retry_successful)
        .count();
    let prediction_error_hinted_retry_success_rate = prediction_error_hinted_retry_successes as f32
        / prediction_error_hinted_retry_tasks.max(1) as f32;
    let mean_surprise_before_retry = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.surprise_before_retry)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let mean_surprise_after_retry = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.surprise_after_retry)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let ast_hdc_parse_successes = tasks
        .iter()
        .map(|task| task.ast_hdc_parse_successes)
        .sum::<usize>();
    let ast_hdc_parse_failures = tasks
        .iter()
        .map(|task| task.ast_hdc_parse_failures)
        .sum::<usize>();
    let structural_prediction_errors = tasks
        .iter()
        .map(|task| task.structural_prediction_errors)
        .sum::<usize>();
    let mean_ast_feature_count = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.mean_ast_feature_count)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let mean_structural_repair_similarity = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.structural_repair_similarity)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let mean_structural_repair_l1_delta = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.structural_repair_l1_delta.map(|delta| delta as f32))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let structural_prior_observations = tasks
        .iter()
        .filter(|task| task.structural_prior_score.is_some())
        .count();
    let mean_structural_prior_score = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.structural_prior_score)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let mean_structural_prior_delta = mean_optional(
        tasks
            .iter()
            .filter_map(|task| task.structural_prior_delta)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let distillation_exported = orch.distillation_buffer().len();
    let report = BenchReport {
        benchmark: format!("coding_backends_{}", args.lane),
        feature_geodesic: cfg!(feature = "geodesic_synthesis"),
        task_count: tasks.len(),
        accepted_count,
        quality_pass_count,
        pass_rate: accepted_count as f32 / tasks.len().max(1) as f32,
        quality_pass_rate: quality_pass_count as f32 / tasks.len().max(1) as f32,
        elapsed_ms: start.elapsed().as_millis(),
        backend_attempts,
        rejection_categories,
        repair_lesson_categories,
        repair_attempts,
        repair_successes,
        repair_success_rate,
        success_after_hint_by_category,
        first_successful_backend_after_repair,
        repair_prior_counts_by_backend,
        repair_prior_labels,
        repair_prior_uses,
        repair_prior_label_count,
        repair_hinted_attempts,
        repair_hinted_successes,
        repair_hinted_success_rate,
        repair_memory_hits,
        repair_memory_successes,
        repair_memory_success_rate,
        repair_memory_categories_used,
        prediction_error_repair_hints_enabled: !args.disable_fep_repair_hints,
        ast_hdc_fep_enabled: !args.disable_ast_hdc_fep,
        prediction_errors_seen,
        prediction_error_categories,
        prediction_error_hinted_retry_tasks,
        prediction_error_hinted_retry_successes,
        prediction_error_hinted_retry_success_rate,
        mean_surprise_before_retry,
        mean_surprise_after_retry,
        ast_hdc_parse_successes,
        ast_hdc_parse_failures,
        structural_prediction_errors,
        mean_ast_feature_count,
        mean_structural_repair_similarity,
        mean_structural_repair_l1_delta,
        structural_success_prototypes: structural_prototypes.prototype_count(),
        structural_prior_observations,
        mean_structural_prior_score,
        mean_structural_prior_delta,
        distillation_import_path: args.load_distillation_jsonl.clone(),
        distillation_imported,
        distillation_export_path: args.save_distillation_jsonl.clone(),
        distillation_exported,
        structural_prototype_import_path: args.load_structural_prototypes.clone(),
        structural_prototype_imported,
        structural_prototype_export_path: args.save_structural_prototypes.clone(),
        certificate_source_provenance_counts,
        broca_eval_gate_passed,
        broca_selection_score,
        category_pass_rates,
        certificates_with_topology,
        certificates_with_oracle,
        certificates_with_sheaf,
        certificates_sheaf_coherent,
        certificates_sheaf_incoherent,
        mean_attempts_per_task,
        tasks,
    };

    if let Some(path) = args.save_distillation_jsonl.as_deref() {
        if let Err(error) = orch.save_distillation(std::path::Path::new(path)) {
            eprintln!("error: failed to save distillation JSONL to {path}: {error}");
            std::process::exit(2);
        }
        eprintln!(
            "[benchmark] exported {} distillation structural memories to {path}",
            report.distillation_exported
        );
    }
    if let Some(path) = args.save_structural_prototypes.as_deref() {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                eprintln!("error: failed to create parent directory for {path}: {error}");
                std::process::exit(2);
            }
        }
        match structural_prototypes.to_json() {
            Ok(json) => {
                if let Err(error) = std::fs::write(path, json) {
                    eprintln!(
                        "error: failed to save compact structural prototypes to {path}: {error}"
                    );
                    std::process::exit(2);
                }
                eprintln!(
                    "[benchmark] exported {} compact structural prototypes to {path}",
                    report.structural_success_prototypes
                );
            }
            Err(error) => {
                eprintln!("error: failed to serialize compact structural prototypes: {error}");
                std::process::exit(2);
            }
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!(
            "{}: {}/{} accepted in {}ms",
            report.benchmark, report.accepted_count, report.task_count, report.elapsed_ms
        );
        println!(
            "quality gate: {}/{} passed ({:.1}%)",
            report.quality_pass_count,
            report.task_count,
            report.quality_pass_rate * 100.0
        );
        println!("backend attempts: {:?}", report.backend_attempts);
        println!(
            "certificate source provenance: {:?}",
            report.certificate_source_provenance_counts
        );
        println!("rejection categories: {:?}", report.rejection_categories);
        println!(
            "repair lesson categories: {:?}",
            report.repair_lesson_categories
        );
        println!(
            "repair effectiveness: attempts={} successes={} rate={:.3} success_by_category={:?} backend_after_repair={:?}",
            report.repair_attempts,
            report.repair_successes,
            report.repair_success_rate,
            report.success_after_hint_by_category,
            report.first_successful_backend_after_repair
        );
        println!(
            "repair priors: uses={} labels={} hinted_attempts={} hinted_successes={} hinted_rate={:.3} by_backend={:?} labels={:?}",
            report.repair_prior_uses,
            report.repair_prior_label_count,
            report.repair_hinted_attempts,
            report.repair_hinted_successes,
            report.repair_hinted_success_rate,
            report.repair_prior_counts_by_backend,
            report.repair_prior_labels
        );
        println!(
            "repair memory: hits={} successes={} rate={:.3} categories={:?}",
            report.repair_memory_hits,
            report.repair_memory_successes,
            report.repair_memory_success_rate,
            report.repair_memory_categories_used
        );
        println!(
            "prediction errors: hints_enabled={} seen={} categories={:?} hinted_retry={}/{} rate={:.3} surprise_before={:?} surprise_after={:?}",
            report.prediction_error_repair_hints_enabled,
            report.prediction_errors_seen,
            report.prediction_error_categories,
            report.prediction_error_hinted_retry_successes,
            report.prediction_error_hinted_retry_tasks,
            report.prediction_error_hinted_retry_success_rate,
            report.mean_surprise_before_retry,
            report.mean_surprise_after_retry
        );
        println!(
            "AST-HDC: enabled={} parse_successes={} parse_failures={} structural_errors={} mean_features={:?} repair_similarity={:?} repair_l1_delta={:?} prototypes={} prior_observations={} prior_score={:?} prior_delta={:?}",
            report.ast_hdc_fep_enabled,
            report.ast_hdc_parse_successes,
            report.ast_hdc_parse_failures,
            report.structural_prediction_errors,
            report.mean_ast_feature_count,
            report.mean_structural_repair_similarity,
            report.mean_structural_repair_l1_delta,
            report.structural_success_prototypes,
            report.structural_prior_observations,
            report.mean_structural_prior_score,
            report.mean_structural_prior_delta
        );
        println!(
            "distillation memory: imported={} from={:?} exported={} to={:?}",
            report.distillation_imported,
            report.distillation_import_path,
            report.distillation_exported,
            report.distillation_export_path
        );
        println!(
            "compact structural prototypes: imported={} from={:?} exported_to={:?}",
            report.structural_prototype_imported,
            report.structural_prototype_import_path,
            report.structural_prototype_export_path
        );
        println!(
            "Broca eval gate: passed={} selection_score={:.3}",
            report.broca_eval_gate_passed, report.broca_selection_score
        );
        println!(
            "certificates: topology={} oracle={} sheaf={} coherent={} incoherent={}",
            report.certificates_with_topology,
            report.certificates_with_oracle,
            report.certificates_with_sheaf,
            report.certificates_sheaf_coherent,
            report.certificates_sheaf_incoherent
        );
        println!("mean attempts/task: {:.2}", report.mean_attempts_per_task);
        println!("category pass rates: {:?}", report.category_pass_rates);
        for task in &report.tasks {
            println!(
                "  {:<22} {:<11} accepted={} quality={} backend={} provenance={:?} confidence={:.3} attempts={:?} rejections={:?} sheaf={:?} beta1={:?}",
                task.id,
                task.category,
                task.accepted,
                task.quality_gate_passed,
                task.backend_name,
                task.certificate_source_provenance,
                task.confidence,
                task.attempts,
                task.rejection_categories,
                task.certificate_sheaf_coherent,
                task.topology_beta_1
            );
            if task.prediction_errors_seen > 0 {
                println!(
                    "      prediction errors: count={} categories={:?} hinted_success={} surprise={:?}->{:?}",
                    task.prediction_errors_seen,
                    task.prediction_error_categories,
                    task.prediction_error_hinted_retry_successful,
                    task.surprise_before_retry,
                    task.surprise_after_retry
                );
            }
            if task.ast_hdc_parse_successes > 0 || task.ast_hdc_parse_failures > 0 {
                println!(
                    "      AST-HDC: parse_successes={} parse_failures={} structural_errors={} mean_features={:?} repair_similarity={:?} repair_l1_delta={:?} prior={:?}@{:?} prior_delta={:?}",
                    task.ast_hdc_parse_successes,
                    task.ast_hdc_parse_failures,
                    task.structural_prediction_errors,
                    task.mean_ast_feature_count,
                    task.structural_repair_similarity,
                    task.structural_repair_l1_delta,
                    task.structural_prior_score,
                    task.structural_prior_label,
                    task.structural_prior_delta
                );
            }
            if !task.repair_priors_seen.is_empty() {
                println!(
                    "      repair priors seen: {:?} labels={:?}",
                    task.repair_priors_seen, task.repair_prior_labels_seen
                );
            }
            for rejection in &task.rejections {
                println!(
                    "    - {} [{}]: {}",
                    rejection.backend, rejection.category, rejection.reason
                );
                if let Some(preview) = &rejection.source_preview {
                    println!("      source: {}", preview.replace('\n', "\\n"));
                }
            }
            for lesson in &task.repair_lessons {
                println!(
                    "      repair lesson {} [{}]: {}",
                    lesson.backend, lesson.category, lesson.hint
                );
            }
        }
    }

    if let Some(path) = &args.repair_lessons_jsonl {
        if let Err(error) = write_repair_lessons_jsonl(path, &report) {
            eprintln!("failed to write repair lessons JSONL to {path}: {error}");
            std::process::exit(1);
        }
    }
}

fn compute_broca_selection_score(
    pass_rate: f32,
    quality_pass_rate: f32,
    mean_attempts_per_task: f32,
    sheaf_incoherent: usize,
    repair_attempts: usize,
    repair_success_rate: f32,
) -> f32 {
    let attempt_penalty = ((mean_attempts_per_task - 1.0).max(0.0) / 4.0).min(1.0);
    let sheaf_penalty = (sheaf_incoherent as f32 * 0.05).min(0.5);
    let repair_score = if repair_attempts == 0 {
        1.0
    } else {
        repair_success_rate
    };
    (0.40 * pass_rate
        + 0.40 * quality_pass_rate
        + 0.10 * (1.0 - attempt_penalty)
        + 0.10 * repair_score
        - sheaf_penalty)
        .clamp(0.0, 1.0)
}

#[derive(Debug, Default)]
struct Args {
    json: bool,
    simulated_llm: bool,
    energy_budget: Option<f32>,
    lane: String,
    repair_lessons_jsonl: Option<String>,
    load_distillation_jsonl: Option<String>,
    save_distillation_jsonl: Option<String>,
    load_structural_prototypes: Option<String>,
    save_structural_prototypes: Option<String>,
    runtime_policy_json: Option<String>,
    disable_fep_repair_hints: bool,
    disable_ast_hdc_fep: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args::default();
        args.lane = "smoke".to_string();
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--json" => args.json = true,
                "--simulated-llm" => args.simulated_llm = true,
                "--energy-budget" => {
                    let Some(raw) = iter.next() else {
                        print_help_and_exit(2, "--energy-budget requires a number");
                    };
                    let Ok(budget) = raw.parse::<f32>() else {
                        print_help_and_exit(2, "--energy-budget must be a number");
                    };
                    args.energy_budget = Some(budget);
                }
                "--lane" => {
                    let Some(lane) = iter.next() else {
                        print_help_and_exit(
                            2,
                            "--lane requires smoke, hard, repair, frontier, or all",
                        );
                    };
                    if !matches!(
                        lane.as_str(),
                        "smoke" | "hard" | "repair" | "frontier" | "all"
                    ) {
                        print_help_and_exit(
                            2,
                            "--lane must be smoke, hard, repair, frontier, or all",
                        );
                    }
                    args.lane = lane;
                }
                "--repair-lessons-jsonl" => {
                    let Some(path) = iter.next() else {
                        print_help_and_exit(2, "--repair-lessons-jsonl requires a path");
                    };
                    args.repair_lessons_jsonl = Some(path);
                }
                "--load-distillation-jsonl" => {
                    let Some(path) = iter.next() else {
                        print_help_and_exit(2, "--load-distillation-jsonl requires a path");
                    };
                    args.load_distillation_jsonl = Some(path);
                }
                "--save-distillation-jsonl" => {
                    let Some(path) = iter.next() else {
                        print_help_and_exit(2, "--save-distillation-jsonl requires a path");
                    };
                    args.save_distillation_jsonl = Some(path);
                }
                "--load-structural-prototypes" => {
                    let Some(path) = iter.next() else {
                        print_help_and_exit(2, "--load-structural-prototypes requires a path");
                    };
                    args.load_structural_prototypes = Some(path);
                }
                "--save-structural-prototypes" => {
                    let Some(path) = iter.next() else {
                        print_help_and_exit(2, "--save-structural-prototypes requires a path");
                    };
                    args.save_structural_prototypes = Some(path);
                }
                "--runtime-policy-json" => {
                    let Some(path) = iter.next() else {
                        print_help_and_exit(2, "--runtime-policy-json requires a path");
                    };
                    args.runtime_policy_json = Some(path);
                }
                "--disable-fep-repair-hints" => args.disable_fep_repair_hints = true,
                "--disable-ast-hdc-fep" => args.disable_ast_hdc_fep = true,
                "--help" | "-h" => print_help_and_exit(0, ""),
                other => print_help_and_exit(2, &format!("unknown argument: {other}")),
            }
        }
        args
    }
}

fn print_help_and_exit(code: i32, error: &str) -> ! {
    if !error.is_empty() {
        eprintln!("error: {error}");
        eprintln!();
    }
    eprintln!("Usage:");
    eprintln!(
        "  cargo run --example benchmark_coding_backends --features code_generation,geodesic_synthesis -- --json"
    );
    eprintln!("Options:");
    eprintln!("  --json              Print machine-readable JSON");
    eprintln!("  --simulated-llm     Use deterministic offline LLM fallback");
    eprintln!("  --energy-budget N   Override orchestrator energy budget");
    eprintln!("  --lane NAME         Benchmark lane: smoke, hard, repair, frontier, or all");
    eprintln!("  --repair-lessons-jsonl PATH");
    eprintln!("                      Write one structured repair lesson per JSONL line");
    eprintln!("  --load-distillation-jsonl PATH");
    eprintln!("                      Seed verified structural memory from distillation JSONL");
    eprintln!("  --save-distillation-jsonl PATH");
    eprintln!("                      Export verified structural memory after the benchmark");
    eprintln!("  --load-structural-prototypes PATH");
    eprintln!("                      Seed compact AST-HDC prototype memory from JSON");
    eprintln!("  --save-structural-prototypes PATH");
    eprintln!("                      Export compact AST-HDC prototype memory after the benchmark");
    eprintln!("  --runtime-policy-json PATH");
    eprintln!(
        "                      Apply a coding-agent routing policy JSON during the benchmark"
    );
    eprintln!("  --disable-fep-repair-hints");
    eprintln!(
        "                      Ablate FEP prediction-error hints while still measuring failures"
    );
    eprintln!("  --disable-ast-hdc-fep");
    eprintln!("                      Ablate AST-HDC structural FEP observations");
    std::process::exit(code);
}

fn write_repair_lessons_jsonl(path: &str, report: &BenchReport) -> std::io::Result<()> {
    let mut lines = Vec::new();
    for task in &report.tasks {
        for lesson in &task.repair_lessons {
            lines.push(serde_json::to_string(lesson).expect("repair lesson serializes"));
        }
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, lines.join("\n"))
}

fn preview_source(source: &str) -> String {
    let trimmed = source.trim();
    let mut preview = trimmed.lines().take(24).collect::<Vec<_>>().join("\n");
    const MAX_CHARS: usize = 1200;
    if preview.len() > MAX_CHARS {
        preview.truncate(MAX_CHARS);
        preview.push_str("\n...");
    }
    preview
}

fn category_reports(tasks: &[TaskReport]) -> BTreeMap<String, CategoryReport> {
    let mut reports = BTreeMap::<String, CategoryReport>::new();
    for task in tasks {
        let report = reports.entry(task.category.clone()).or_default();
        report.task_count += 1;
        if task.accepted {
            report.accepted_count += 1;
        }
    }
    for report in reports.values_mut() {
        report.pass_rate = report.accepted_count as f32 / report.task_count.max(1) as f32;
    }
    reports
}

fn prediction_error_categories(tasks: &[TaskReport]) -> BTreeMap<String, usize> {
    let mut categories = BTreeMap::new();
    for task in tasks {
        for (category, count) in &task.prediction_error_categories {
            *categories.entry(category.clone()).or_insert(0) += count;
        }
    }
    categories
}

fn mean_optional(values: &[f32]) -> Option<f32> {
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

fn task_uses_repair_memory(task: &TaskReport) -> bool {
    task.repair_prior_labels_seen
        .iter()
        .any(|label| label.starts_with("repair_memory_"))
}

fn repair_memory_categories(labels: &BTreeMap<String, usize>) -> BTreeMap<String, usize> {
    let mut categories = BTreeMap::new();
    for (label, count) in labels {
        let Some(category) = repair_memory_category_from_label(label) else {
            continue;
        };
        *categories.entry(category).or_insert(0) += count;
    }
    categories
}

fn repair_memory_category_from_label(label: &str) -> Option<String> {
    let rest = label.strip_prefix("repair_memory_")?;
    if let Some(rest) = rest.strip_prefix("diagnostic_") {
        return rest
            .splitn(2, '_')
            .nth(1)
            .filter(|category| !category.is_empty())
            .map(ToString::to_string);
    }
    rest.split_once('_')
        .map(|(_, category)| category)
        .filter(|category| !category.is_empty())
        .map(ToString::to_string)
}

fn tasks_for_lane(lane: &str) -> Vec<BenchTask> {
    match lane {
        "smoke" => smoke_tasks(),
        "hard" => hard_tasks(),
        "repair" => repair_tasks(),
        "frontier" => frontier_tasks(),
        "all" => {
            let mut tasks = smoke_tasks();
            tasks.extend(hard_tasks());
            tasks.extend(repair_tasks());
            tasks.extend(frontier_tasks());
            tasks
        }
        _ => smoke_tasks(),
    }
}

fn smoke_tasks() -> Vec<BenchTask> {
    vec![
        BenchTask {
            lane: "smoke",
            id: "linear_add",
            category: "linear",
            name: "add",
            purpose: "Add two integers",
            signature: "fn add(a: i32, b: i32) -> i32",
            examples: &[("add(2, 3)", "5")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "linear_double",
            category: "linear",
            name: "double",
            purpose: "Double an integer",
            signature: "fn double(n: i32) -> i32",
            examples: &[("double(4)", "8")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "numeric_abs",
            category: "branch",
            name: "abs_i32",
            purpose: "Return the absolute value of an integer",
            signature: "fn abs_i32(n: i32) -> i32",
            examples: &[("abs_i32(-3)", "3")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "branch_even",
            category: "branch",
            name: "is_even",
            purpose: "Return whether a number is even",
            signature: "fn is_even(n: i32) -> bool",
            examples: &[("is_even(4)", "true"), ("is_even(5)", "false")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "branch_clamp",
            category: "branch",
            name: "clamp_0_100",
            purpose: "Clamp an integer into the inclusive range 0 to 100",
            signature: "fn clamp_0_100(n: i32) -> i32",
            examples: &[("clamp_0_100(-5)", "0"), ("clamp_0_100(120)", "100")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "branch_max",
            category: "branch",
            name: "max_i32",
            purpose: "Return the maximum of two integers",
            signature: "fn max_i32(a: i32, b: i32) -> i32",
            examples: &[("max_i32(2, 7)", "7"), ("max_i32(-1, -3)", "-1")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "branch_min",
            category: "branch",
            name: "min_i32",
            purpose: "Return the minimum of two integers",
            signature: "fn min_i32(a: i32, b: i32) -> i32",
            examples: &[("min_i32(2, 7)", "2"), ("min_i32(-1, -3)", "-3")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "branch_positive",
            category: "branch",
            name: "is_positive",
            purpose: "Return whether an integer is positive",
            signature: "fn is_positive(n: i32) -> bool",
            examples: &[("is_positive(4)", "true"), ("is_positive(0)", "false")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "reduce_sum",
            category: "collection",
            name: "sum",
            purpose: "Sum each number in a slice",
            signature: "fn sum(items: &[i32]) -> i32",
            examples: &[("sum(&[1, 2, 3])", "6")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "collection_count_positive",
            category: "collection",
            name: "count_positive",
            purpose: "Count the positive integers in a slice",
            signature: "fn count_positive(items: &[i32]) -> usize",
            examples: &[("count_positive(&[-1, 0, 3, 4])", "2")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "collection_any_even",
            category: "collection",
            name: "any_even",
            purpose: "Return whether any integer in a slice is even",
            signature: "fn any_even(items: &[i32]) -> bool",
            examples: &[("any_even(&[1, 3, 4])", "true")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "map_strings",
            category: "string",
            name: "normalize_all",
            purpose: "Map each string to a normalized lowercase string",
            signature: "fn normalize_all(items: &[String]) -> Vec<String>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "string_reverse",
            category: "string",
            name: "reverse",
            purpose: "Reverse a string",
            signature: "fn reverse(s: &str) -> String",
            examples: &[("reverse(\"abc\")", "\"cba\"")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "string_count_words",
            category: "string",
            name: "count_words",
            purpose: "Count whitespace separated words in a string",
            signature: "fn count_words(s: &str) -> usize",
            examples: &[("count_words(\"one two three\")", "3")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "string_trim",
            category: "string",
            name: "trim_owned",
            purpose: "Trim surrounding whitespace from a string",
            signature: "fn trim_owned(s: &str) -> String",
            examples: &[("trim_owned(\"  hi  \")", "\"hi\"")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "string_uppercase",
            category: "string",
            name: "uppercase",
            purpose: "Convert a string to uppercase",
            signature: "fn uppercase(s: &str) -> String",
            examples: &[("uppercase(\"hi\")", "\"HI\"")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "string_contains",
            category: "string",
            name: "contains_substr",
            purpose: "Return whether a string contains a substring",
            signature: "fn contains_substr(haystack: &str, needle: &str) -> bool",
            examples: &[
                ("contains_substr(\"hello\", \"ell\")", "true"),
                ("contains_substr(\"hello\", \"xyz\")", "false"),
            ],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "option_get_or",
            category: "option",
            name: "get_or",
            purpose: "Return the option value or the fallback integer",
            signature: "fn get_or(value: Option<i32>, fallback: i32) -> i32",
            examples: &[("get_or(Some(7), 1)", "7"), ("get_or(None, 1)", "1")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "option_map_increment",
            category: "option",
            name: "inc_option",
            purpose: "Increment an optional integer with option map",
            signature: "fn inc_option(value: Option<i32>) -> Option<i32>",
            examples: &[
                ("inc_option(Some(7))", "Some(8)"),
                ("inc_option(None)", "None"),
            ],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "option_ok_or",
            category: "result",
            name: "require_value",
            purpose: "Require an option value and convert None with ok_or",
            signature: "fn require_value(value: Option<i32>) -> Result<i32, &'static str>",
            examples: &[("require_value(Some(7)).unwrap()", "7")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "result_parse_i32",
            category: "result",
            name: "parse_i32",
            purpose: "Parse a string as i32 and return the parse error on failure",
            signature: "fn parse_i32(raw: &str) -> Result<i32, std::num::ParseIntError>",
            examples: &[("parse_i32(\"42\").unwrap()", "42")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "result_filter_map_parse",
            category: "result",
            name: "parse_numbers",
            purpose: "Parse all valid numbers from string slices using filter_map",
            signature: "fn parse_numbers(raw: &[&str]) -> Vec<i32>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "result_parse_u64",
            category: "result",
            name: "parse_u64",
            purpose: "Parse a string as u64 and return the parse error on failure",
            signature: "fn parse_u64(raw: &str) -> Result<u64, std::num::ParseIntError>",
            examples: &[("parse_u64(\"42\").unwrap()", "42")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "generic_first",
            category: "generic",
            name: "first",
            purpose: "Return the first item from a slice by reference",
            signature: "fn first<T>(items: &[T]) -> Option<&T>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "generic_clone_first",
            category: "generic",
            name: "clone_first",
            purpose: "Return the first item from a slice as an owned clone",
            signature: "fn clone_first<T: Clone>(items: &[T]) -> Option<T>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "generic_len",
            category: "generic",
            name: "slice_len",
            purpose: "Return the length of a generic slice",
            signature: "fn slice_len<T>(items: &[T]) -> usize",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "async_ready",
            category: "async",
            name: "ready_value",
            purpose: "Return an integer from an async function",
            signature: "async fn ready_value(value: i32) -> i32",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "conversion_to_vec",
            category: "conversion",
            name: "to_vec",
            purpose: "Copy a slice of integers into a new vector",
            signature: "fn to_vec(items: &[i32]) -> Vec<i32>",
            examples: &[("to_vec(&[1, 2])", "vec![1, 2]")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "conversion_string_len",
            category: "conversion",
            name: "string_len",
            purpose: "Return the length of a string slice",
            signature: "fn string_len(s: &str) -> usize",
            examples: &[("string_len(\"abcd\")", "4")],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "map_word_counts",
            category: "map",
            name: "word_counts",
            purpose: "Build word frequency counts from text",
            signature: "fn word_counts(text: &str) -> std::collections::HashMap<String, usize>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "smoke",
            id: "map_index_by_len",
            category: "map",
            name: "index_by_len",
            purpose: "Index strings by length in a BTreeMap",
            signature: "fn index_by_len(items: &[String]) -> std::collections::BTreeMap<usize, String>",
            examples: &[],
            constraints: &[],
        },
    ]
}

fn hard_tasks() -> Vec<BenchTask> {
    vec![
        BenchTask {
            lane: "hard",
            id: "hard_result_question_mark",
            category: "result",
            name: "parse_and_double",
            purpose: "Parse a string as i32 with the question mark operator and double it",
            signature: "fn parse_and_double(raw: &str) -> Result<i32, std::num::ParseIntError>",
            examples: &[("parse_and_double(\"21\").unwrap()", "42")],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_trait_bound_sort",
            category: "generic",
            name: "sorted_clone",
            purpose: "Return a sorted cloned vector from a generic slice using the Ord bound",
            signature: "fn sorted_clone<T: Ord + Clone>(items: &[T]) -> Vec<T>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_lifetime_longer",
            category: "lifetime",
            name: "longer",
            purpose: "Return the longer of two borrowed string slices",
            signature: "fn longer<'a>(a: &'a str, b: &'a str) -> &'a str",
            examples: &[("longer(\"abc\", \"d\")", "\"abc\"")],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_borrow_mutation",
            category: "ownership",
            name: "push_if_missing",
            purpose: "Mutate a vector by pushing a value only when it is missing",
            signature: "fn push_if_missing(items: &mut Vec<i32>, value: i32)",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_result_option_bridge",
            category: "result",
            name: "first_positive",
            purpose: "Find the first positive integer and return an error string if none exists",
            signature: "fn first_positive(items: &[i32]) -> Result<i32, &'static str>",
            examples: &[("first_positive(&[-1, 0, 5]).unwrap()", "5")],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_struct_constructor",
            category: "struct",
            name: "make_pair",
            purpose: "Construct a tuple-like pair represented as a Rust tuple",
            signature: "fn make_pair(a: i32, b: i32) -> (i32, i32)",
            examples: &[("make_pair(1, 2)", "(1, 2)")],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_hashmap_group",
            category: "map",
            name: "group_by_len",
            purpose: "Group strings by length in a HashMap accumulator",
            signature: "fn group_by_len(items: &[String]) -> std::collections::HashMap<usize, Vec<String>>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "hard",
            id: "hard_async_result",
            category: "async",
            name: "async_parse",
            purpose: "Parse an integer inside an async Result-returning function",
            signature: "async fn async_parse(raw: &str) -> Result<i32, std::num::ParseIntError>",
            examples: &[],
            constraints: &[],
        },
    ]
}

fn repair_tasks() -> Vec<BenchTask> {
    vec![
        BenchTask {
            lane: "repair",
            id: "repair_forced_type_mismatch",
            category: "repair",
            name: "add",
            purpose: "Add two integers after a forced geodesic type-mismatch rejection",
            signature: "fn add(a: i32, b: i32) -> i32",
            examples: &[("add(2, 3)", "5")],
            constraints: &[
                "benchmark_force_geodesic_rejection:function `add` returns `()` but signature expects `i32`",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_forced_parse_failure",
            category: "repair",
            name: "add",
            purpose: "Add two integers after a forced parse failure in the previous backend",
            signature: "fn add(a: i32, b: i32) -> i32",
            examples: &[("add(2, 3)", "5")],
            constraints: &[
                "benchmark_force_geodesic_rejection:generated candidate does not parse as Rust: expected expression, found `}`",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_forced_stub",
            category: "repair",
            name: "sum",
            purpose: "Sum all integers in a slice after a forced stub rejection",
            signature: "fn sum(items: &[i32]) -> i32",
            examples: &[("sum(&[1, 2, 3])", "6")],
            constraints: &[
                "benchmark_force_geodesic_rejection:implementation stub `todo!()` remains in generated source",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_forced_unresolved_identifier",
            category: "repair",
            name: "add",
            purpose: "Add two integers after a forced unresolved identifier rejection",
            signature: "fn add(a: i32, b: i32) -> i32",
            examples: &[("add(2, 3)", "5")],
            constraints: &[
                "benchmark_force_geodesic_rejection:cannot find value `total` in this scope",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_forced_ownership",
            category: "repair",
            name: "sum",
            purpose: "Sum integers in a slice after a forced ownership rejection",
            signature: "fn sum(items: &[i32]) -> i32",
            examples: &[("sum(&[1, 2, 3])", "6")],
            constraints: &[
                "benchmark_force_geodesic_rejection:borrow of moved value `items` in generated candidate",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_forced_test_failure",
            category: "repair",
            name: "add",
            purpose: "Add two integers after a forced test failure rejection",
            signature: "fn add(a: i32, b: i32) -> i32",
            examples: &[("add(2, 3)", "5")],
            constraints: &[
                "benchmark_force_geodesic_rejection:generated candidate compiled but test_example_0 failed",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_memory_sensitive_add",
            category: "repair_memory",
            name: "add",
            purpose: "Add two integers after learning from a prior type-mismatch repair",
            signature: "fn add(a: i32, b: i32) -> i32",
            examples: &[("add(2, 3)", "5")],
            constraints: &[
                "benchmark_force_geodesic_rejection_unless_repair_memory:function `add` returns `()` but signature expects `i32`",
            ],
        },
        BenchTask {
            lane: "repair",
            id: "repair_memory_sensitive_is_even",
            category: "repair_memory",
            name: "is_even",
            purpose: "Return whether a number is even after learning from a prior test repair",
            signature: "fn is_even(n: i32) -> bool",
            examples: &[("is_even(4)", "true"), ("is_even(5)", "false")],
            constraints: &[
                "benchmark_force_geodesic_rejection_unless_repair_memory:generated candidate compiled but test_example_0 failed",
            ],
        },
    ]
}

fn frontier_tasks() -> Vec<BenchTask> {
    vec![
        BenchTask {
            lane: "frontier",
            id: "frontier_result_collect",
            category: "result",
            name: "parse_all",
            purpose: "Parse all string slices into integers and return the first parse error",
            signature: "fn parse_all(raw: &[&str]) -> Result<Vec<i32>, std::num::ParseIntError>",
            examples: &[("parse_all(&[\"1\", \"2\"]).unwrap()", "vec![1, 2]")],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_result_collect_u64",
            category: "result",
            name: "parse_all_u64",
            purpose: "Parse all string slices into u64 integers and return the first parse error",
            signature: "fn parse_all_u64(raw: &[&str]) -> Result<Vec<u64>, std::num::ParseIntError>",
            examples: &[("parse_all_u64(&[\"1\", \"2\"]).unwrap()", "vec![1, 2]")],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_generic_contains",
            category: "generic",
            name: "contains_item",
            purpose: "Return whether a generic slice contains a borrowed target value",
            signature: "fn contains_item<T: PartialEq>(items: &[T], target: &T) -> bool",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_generic_clone_first",
            category: "generic",
            name: "clone_first",
            purpose: "Return the first item from a generic slice as an owned clone",
            signature: "fn clone_first<T: Clone>(items: &[T]) -> Option<T>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_lifetime_prefix",
            category: "lifetime",
            name: "first_nonempty",
            purpose: "Return the first nonempty borrowed string slice from a slice",
            signature: "fn first_nonempty<'a>(items: &'a [&'a str]) -> Option<&'a str>",
            examples: &[("first_nonempty(&[\"\", \"hi\"])", "Some(\"hi\")")],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_lifetime_owned_string_slice",
            category: "lifetime",
            name: "first_nonempty_owned",
            purpose: "Return the first nonempty string slice from owned strings",
            signature: "fn first_nonempty_owned(items: &[String]) -> Option<&str>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_mut_dedup",
            category: "ownership",
            name: "dedup_sorted",
            purpose: "Sort a mutable vector of integers and remove duplicates in place",
            signature: "fn dedup_sorted(items: &mut Vec<i32>)",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_hashmap_group_by_len",
            category: "map",
            name: "group_by_len",
            purpose: "Group strings by length in a HashMap accumulator",
            signature: "fn group_by_len(items: &[String]) -> std::collections::HashMap<usize, Vec<String>>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_async_option",
            category: "async",
            name: "async_first",
            purpose: "Return the first integer from a slice inside an async function",
            signature: "async fn async_first(items: &[i32]) -> Option<i32>",
            examples: &[],
            constraints: &[],
        },
        BenchTask {
            lane: "frontier",
            id: "frontier_async_option_result",
            category: "async",
            name: "async_parse_optional",
            purpose: "Parse an optional string inside an async function",
            signature: "async fn async_parse_optional(raw: Option<&str>) -> Result<Option<i32>, std::num::ParseIntError>",
            examples: &[],
            constraints: &[],
        },
    ]
}
