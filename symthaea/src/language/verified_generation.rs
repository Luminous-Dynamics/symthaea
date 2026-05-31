// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verified Code Generation — Guaranteed-Compile, Test-First Pipeline
//!
//! This is what makes Symthaea fundamentally different from an LLM:
//! **generated code is verified before returning.**
//!
//! Pipeline:
//! ```text
//! 1. Generate tests from spec (test-first)
//! 2. Generate function body
//! 3. Combine function + tests
//! 4. Compile via rustc
//! 5. If fails: auto-fix → retry (max 3)
//! 6. Run tests
//! 7. If tests fail: use failure info to adjust → retry
//! 8. Return ONLY if compiled + tests pass
//! 9. Optionally: Z3 formal verification for pure functions
//! ```
//!
//! An LLM returns code and hopes it works.
//! Symthaea returns code that is **proven to compile and pass its own tests**.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::code_executor::{CodeExecutor, ExecutionResult, TestFailure, try_auto_fix};
use super::code_generator::{CodeContext, CodeGenerator};
use super::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use super::code_parser::EntityKind;
use super::coding_prediction_error::{
    CodingPredictionError, prediction_error_categories, prediction_error_diagnostics,
    prediction_error_hints, prediction_errors_from_execution,
    structural_prediction_error_from_ast_parse, structural_prediction_error_from_prior,
};
use super::llm_backend::{GenerationParams, LLMBackend};
use super::repair_memory;
use super::repo_map::RepoMap;
use super::rust_ast_hdc::encode_rust_ast_hdc;
use super::rust_lsp::{LspPosition, RustAnalyzerClient};
use super::semantic_repair::try_semantic_ast_repair;
use super::sovereign_attestor::{ProofAttestation, SovereignAttestor};
use super::structural_prototype::{
    StructuralPriorScore, StructuralPrototypeBank, StructuralPrototypeLabels,
    ast_features_for_source, return_shape_for_signature,
};
use crate::coding_experience::CodingExperienceStore;
use crate::hdc::diagnostic_encoder::DiagnosticHDEncoder;
use crate::mind::structured_thought::EpistemicStatus;
use symthaea_core::core::ContinuousHV;
use tracing::{error, info, warn};

/// Maximum number of compile-fix-retry iterations
const MAX_COMPILE_RETRIES: usize = 3;

/// Maximum number of test-fix-retry iterations
const MAX_TEST_RETRIES: usize = 2;

/// Result of verified code generation — guaranteed properties.
#[derive(Debug, Clone)]
pub struct VerifiedCode {
    /// The generated source code (guaranteed to compile if `compiled == true`)
    pub source: String,
    /// Whether the code compiled successfully
    pub compiled: bool,
    /// Whether all generated tests passed
    pub tests_passed: bool,
    /// Number of tests that passed
    pub test_count_passed: usize,
    /// Number of tests that failed
    pub test_count_failed: usize,
    /// Number of compile-fix retries needed
    pub compile_retries: usize,
    /// Number of test-fix retries needed
    pub test_retries: usize,
    /// Whether Z3 formal verification succeeded (None = not attempted)
    pub formally_verified: Option<bool>,
    /// Cryptographic attestation of formal proof (Dilithium)
    pub attestation: Option<ProofAttestation>,
    /// Confidence assessment
    pub confidence: VerificationConfidence,
    /// Compilation errors encountered (empty if compiled successfully)
    pub compile_errors: Vec<String>,
    /// Test failures encountered (empty if all passed)
    pub test_failures: Vec<String>,
    /// Diagnostic geometries captured during repair.
    pub diagnostic_hvs: Vec<ContinuousHV>,
    /// AST-HDC observations captured during generation and repair.
    pub ast_hdc: AstHdcTrace,
}

/// Lightweight telemetry for Rust AST-HDC structural observations.
///
/// The actual hypervectors stay inside retry contexts and repair memory. This
/// trace keeps reportable counts and feature maps so benchmarks can measure
/// structural surprise without serializing high-dimensional vectors.
#[derive(Debug, Clone, Default)]
pub struct AstHdcTrace {
    pub parse_successes: usize,
    pub parse_failures: usize,
    pub structural_prediction_errors: usize,
    pub feature_observations: usize,
    pub total_feature_count: usize,
    pub last_feature_count: usize,
    pub first_features: Option<BTreeMap<String, usize>>,
    pub last_features: Option<BTreeMap<String, usize>>,
    pub structural_prior_observations: usize,
    pub last_structural_prior_score: Option<f32>,
    pub last_structural_prior_label: Option<String>,
    pub best_structural_prior_score: Option<f32>,
    pub structural_prior_delta: Option<f32>,
    pub geodesic_rejection_shadow_hits: usize,
    pub geodesic_rejection_shadow_true_positives: usize,
    pub geodesic_rejection_shadow_false_positives: usize,
    pub hard_geodesic_rejections: usize,
}

impl AstHdcTrace {
    pub fn mean_feature_count(&self) -> Option<f32> {
        (self.feature_observations > 0)
            .then(|| self.total_feature_count as f32 / self.feature_observations as f32)
    }
}

impl VerifiedCode {
    /// Whether this code meets the "guaranteed correct" bar
    pub fn is_guaranteed(&self) -> bool {
        self.compiled && self.tests_passed
    }

    /// Summary string
    pub fn summary(&self) -> String {
        if self.is_guaranteed() {
            format!(
                "VERIFIED: compiled, {}/{} tests passed{}{}",
                self.test_count_passed,
                self.test_count_passed + self.test_count_failed,
                if self.compile_retries > 0 {
                    format!(" (after {} compile fixes)", self.compile_retries)
                } else {
                    String::new()
                },
                if self.formally_verified == Some(true) {
                    " [Z3 PROVEN]"
                } else {
                    ""
                },
            )
        } else if self.compiled {
            format!("COMPILED but {} tests failed", self.test_count_failed)
        } else {
            format!(
                "FAILED to compile: {}",
                self.compile_errors
                    .first()
                    .unwrap_or(&"unknown".to_string())
            )
        }
    }
}

/// Calibrated confidence assessment for generated code.
#[derive(Debug, Clone)]
pub struct VerificationConfidence {
    /// Did it compile?
    pub compiled: bool,
    /// Did all tests pass?
    pub tests_passed: bool,
    /// Did Z3 prove correctness?
    pub formally_verified: bool,
    /// Historical success rate for this task category (0.0-1.0)
    pub category_success_rate: f32,
    /// Overall calibrated confidence (0.0-1.0)
    pub confidence: f32,
}

impl VerificationConfidence {
    fn compute(
        compiled: bool,
        tests_passed: bool,
        formally_verified: bool,
        category_rate: f32,
    ) -> Self {
        let mut confidence = 0.0f32;

        if compiled {
            confidence += 0.4; // compilation alone is 40%
        }
        if tests_passed {
            confidence += 0.35; // passing tests adds 35%
        }
        if formally_verified {
            confidence += 0.15; // Z3 proof adds 15%
        }
        // Historical category rate contributes 10%
        confidence += category_rate * 0.1;

        Self {
            compiled,
            tests_passed,
            formally_verified,
            category_success_rate: category_rate,
            confidence: confidence.min(1.0),
        }
    }
}

/// Generate code with compilation and test verification.
///
/// This is the core "better than LLM" function:
/// 1. Generate tests from the spec (test-first)
/// 2. Generate the function body
/// 3. Compile and run tests
/// 4. Auto-fix on failure, retry
/// 5. Return only verified code
///
/// # Arguments
/// * `generator` — the code generator (uses native emitter)
/// * `executor` — code executor (must have `enable_real_execution()`)
/// * `spec` — what to generate
/// * `context` — generation context (codebase memory, past examples, etc.)
pub fn generate_verified(
    generator: &CodeGenerator,
    executor: &mut CodeExecutor,
    intent: &CodeIntent,
    context: &CodeContext<'_>,
) -> VerifiedCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(generate_verified_inner(
        generator, executor, intent, context, None, None, None, None, None, None,
    ))
}

/// Generate and verify code with AST/HDC repository context available for
/// compile-error repair retries.
pub fn generate_verified_with_repo_map<'a>(
    generator: &CodeGenerator,
    executor: &mut CodeExecutor,
    intent: &CodeIntent,
    context: &CodeContext<'a>,
    repo_map: &'a RepoMap,
) -> VerifiedCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(generate_verified_inner(
        generator,
        executor,
        intent,
        context,
        Some(repo_map),
        None,
        None,
        None,
        None,
        None,
    ))
}

/// Generate and verify code with both RepoMap and LSP client available for
/// high-precision repair.
pub fn generate_verified_full<'a>(
    generator: &CodeGenerator,
    executor: &mut CodeExecutor,
    intent: &CodeIntent,
    context: &CodeContext<'a>,
    repo_map: Option<&'a RepoMap>,
    mut lsp_client: Option<&mut RustAnalyzerClient>,
    experience_store: Option<&mut CodingExperienceStore>,
    llm_backend: Option<Arc<dyn LLMBackend>>,
    #[cfg(feature = "swarm")] swarm_tx: Option<
        tokio::sync::mpsc::Sender<symthaea_swarm::SwarmMessage>,
    >,
    #[cfg(not(feature = "swarm"))] _swarm_tx: Option<()>,
    #[cfg(feature = "swarm")] swarm_proofs: Option<&[symthaea_swarm::SwarmProofMsg]>,
    #[cfg(not(feature = "swarm"))] _swarm_proofs: Option<()>,
) -> VerifiedCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(generate_verified_inner(
        generator,
        executor,
        intent,
        context,
        repo_map,
        lsp_client.as_deref_mut(),
        experience_store,
        llm_backend,
        #[cfg(feature = "swarm")]
        swarm_tx.as_ref(),
        #[cfg(not(feature = "swarm"))]
        None,
        #[cfg(feature = "swarm")]
        swarm_proofs,
        #[cfg(not(feature = "swarm"))]
        None,
    ))
}

async fn generate_verified_inner<'a>(
    generator: &CodeGenerator,
    executor: &mut CodeExecutor,
    intent: &CodeIntent,
    context: &CodeContext<'a>,
    repo_map: Option<&'a RepoMap>,
    mut lsp_client: Option<&mut RustAnalyzerClient>,
    mut experience_store: Option<&mut CodingExperienceStore>,
    llm_backend: Option<Arc<dyn LLMBackend>>,
    #[cfg(feature = "swarm")] swarm_tx: Option<
        &tokio::sync::mpsc::Sender<symthaea_swarm::SwarmMessage>,
    >,
    #[cfg(not(feature = "swarm"))] _swarm_tx: Option<()>,
    #[cfg(feature = "swarm")] swarm_proofs: Option<&[symthaea_swarm::SwarmProofMsg]>,
    #[cfg(not(feature = "swarm"))] _swarm_proofs: Option<()>,
) -> VerifiedCode {
    let temp_spec;
    let spec = match intent {
        CodeIntent::Create { spec, .. } => spec,
        CodeIntent::Solve { spec, .. } => {
            temp_spec = CodeSpec::new(&spec.language, "swe-bench-fix", &spec.description);
            &temp_spec
        }
        _ => {
            return simulated_verification_failure(String::new(), 0, 0);
        }
    };

    if !executor.supports_real_execution() {
        let reason =
            "Verified generation requires real execution; simulation mode cannot claim compilation or test verification"
                .to_string();
        return VerifiedCode {
            source: String::new(),
            compiled: false,
            tests_passed: false,
            test_count_passed: 0,
            test_count_failed: 0,
            compile_retries: 0,
            test_retries: 0,
            formally_verified: None,
            attestation: None,
            confidence: VerificationConfidence::compute(false, false, false, 0.0),
            compile_errors: vec![reason],
            test_failures: Vec::new(),
            diagnostic_hvs: Vec::new(),
            ast_hdc: AstHdcTrace::default(),
        };
    }

    // Step 1: Generate the function body (which may include inline tests)
    let generated_source = if let Some(llm) = llm_backend.as_ref() {
        let prompt = build_llm_repair_prompt(intent, context, &[]);
        let params = GenerationParams::default();
        let res = llm.generate(&prompt, &params).await;
        match res {
            Ok(src) => {
                let stripped = strip_markdown(&src);
                info!(
                    "LLM generated source ({} chars):\n{}",
                    stripped.len(),
                    stripped
                );
                stripped
            }
            Err(e) => {
                error!("LLM generation failed: {}, falling back to native", e);
                generator.generate(intent, context).source
            }
        }
    } else {
        generator.generate(intent, context).source
    };

    let mut source = generated_source;

    // Step 2: Generate additional tests ONLY if source doesn't already have them
    let test_source = if source.contains("#[cfg(test)]") || source.contains("mod tests") {
        // Source already has tests (emitter generated them from examples)
        String::new()
    } else {
        generator.generate_tests_only(spec).unwrap_or_default()
    };

    // Step 3: Compile-fix loop
    let mut compile_retries = 0;
    let mut last_compile_errors = Vec::new();
    let mut all_diagnostic_hvs = Vec::new();
    let mut seen_prediction_error_keys = HashSet::new();
    let mut ast_hdc = AstHdcTrace::default();
    let mut last_ast_hv: Option<ContinuousHV> = None;
    let structural_prototypes = structural_prototypes_from_context(context, spec);
    let structural_labels = structural_labels_for_spec(spec);
    let mut last_structural_prior: Option<StructuralPriorScore> = None;

    loop {
        let full_source = match intent {
            CodeIntent::Solve { .. } => source.clone(), // For solve, full_source is the patch
            _ => {
                if test_source.is_empty() {
                    source.clone()
                } else {
                    format!("{}\n\n{}", source, test_source)
                }
            }
        };

        let ast_observation = observe_ast_hdc(
            &source,
            compile_retries,
            &mut ast_hdc,
            &structural_prototypes,
            &structural_labels,
        );
        let shadow_hit = ast_observation.prediction_error.is_some()
            && std::env::var_os("SYMTHAEA_GEODESIC_REJECTION_SHADOW").is_some();

        if let Some(hv) = ast_observation.hv {
            last_ast_hv = Some(hv);
        }
        if let Some(prior) = ast_observation.structural_prior {
            last_structural_prior = Some(prior);
        }
        let structural_prediction_errors: Vec<_> =
            ast_observation.prediction_error.into_iter().collect();
        record_prediction_error_hvs(
            &structural_prediction_errors,
            &mut all_diagnostic_hvs,
            &mut seen_prediction_error_keys,
        );
        if ast_observation.fast_fail && compile_retries < MAX_COMPILE_RETRIES {
            compile_retries += 1;
            let retry_spec = augment_spec_with_compile_errors(
                spec,
                &prediction_error_diagnostics(&structural_prediction_errors),
                compile_retries,
            );
            let retry_context = build_compile_retry_context(
                context,
                repo_map,
                lsp_client.as_deref_mut(),
                experience_store.as_deref_mut(),
                &mut all_diagnostic_hvs,
                &prediction_error_diagnostics(&structural_prediction_errors),
                spec,
                &structural_prediction_errors,
                last_ast_hv.as_ref(),
                last_structural_prior.as_ref(),
                compile_retries,
            );
            let retry_intent = CodeIntent::Create {
                target: CodeTarget {
                    kind: EntityKind::Function,
                    name: retry_spec.name.clone(),
                    path: None,
                    language: Some(retry_spec.language.clone()),
                    hv: None,
                },
                spec: retry_spec,
            };
            source = generator.generate(&retry_intent, &retry_context).source;
            continue;
        }

        let mut result = match intent {
            CodeIntent::Solve {
                target: repo_target,
                ..
            } => {
                if let Err(e) = executor.apply_patch(&repo_target.root, &source) {
                    ExecutionResult {
                        compiled: false,
                        compile_errors: vec![format!("Patch Error: {}", e)],
                        tests_passed: 0,
                        tests_failed: 0,
                        test_output: String::new(),
                        runtime_error: None,
                        elapsed: std::time::Duration::from_millis(0),
                        simulated: false,
                        binary_path: None,
                        test_failures: Vec::new(),
                    }
                } else {
                    executor.execute_workspace_tests(&repo_target.root)
                }
            }
            _ => executor.execute_rust_with_inline_tests(&full_source),
        };

        // Evaluate shadow rejection accuracy
        if shadow_hit {
            if result.compiled {
                ast_hdc.geodesic_rejection_shadow_false_positives += 1;
                warn!(
                    "GEODESIC SHADOW FALSE POSITIVE: Candidate would have been rejected but it COMPILED."
                );
            } else {
                ast_hdc.geodesic_rejection_shadow_true_positives += 1;
                info!(
                    "GEODESIC SHADOW TRUE POSITIVE: Candidate would have been rejected and it FAILED to compile."
                );
            }
        }

        result.parse_test_failures();

        if result.simulated {
            return simulated_verification_failure(source, compile_retries, 0);
        }

        if result.compiled {
            // Compiled! Check tests
            if result.tests_failed == 0 {
                let mut formally_verified = None;
                let mut formal_gate_failure = None;
                if let Some(smtlib2_query) = spec.metadata.get("smtlib2") {
                    info!(
                        "Executing neuro-symbolic proof lookup and lemma chaining for {}",
                        spec.name
                    );
                    let bridge = crate::z3_bridge::Z3Bridge::new();
                    let proof_memory = super::proof_memory::global_proof_memory();
                    let memory = proof_memory.lock().clone();
                    let mut engine = super::proof_memory::CachedProofEngine::new(bridge, memory);

                    #[cfg(feature = "swarm")]
                    let proofs_slice = swarm_proofs.unwrap_or(&[]);
                    #[cfg(not(feature = "swarm"))]
                    let proofs_slice: &[()] = &[];

                    let (verdict, details) = engine.verify_with_cache(
                        &spec.name,
                        &source,
                        smtlib2_query,
                        0.85,
                        proofs_slice,
                    );
                    *proof_memory.lock() = engine.memory.clone();
                    let is_proven = verdict == super::proof_memory::ProofVerdict::Proven;
                    let is_refuted = verdict == super::proof_memory::ProofVerdict::Refuted;
                    formally_verified = Some(is_proven);

                    if is_refuted {
                        if let (Some(ref mut store), Some(hv)) =
                            (experience_store.as_mut(), &last_ast_hv)
                        {
                            info!(
                                "LABELING NEGATIVE PROTOTYPE: {} disproven by Z3.",
                                spec.name
                            );
                            store.store_negative_prototype(&spec.name, hv.values.clone());
                        }
                        formal_gate_failure = Some(TestFailure {
                            test_name: "FormalContractGate".into(),
                            expected: Some("SMT contract satisfiable/proven".into()),
                            actual: Some("SMT contract refuted".into()),
                            message: "Formal SMT contract was refuted; candidate cannot be accepted even if it compiles and passes tests.".into(),
                            assertion: None,
                        });
                    }

                    if is_proven && engine.solver_calls > 0 {
                        #[cfg(feature = "swarm")]
                        if let Some(tx) = swarm_tx {
                            let gossip_msg = symthaea_swarm::SwarmProofMsg {
                                node_id: uuid::Uuid::new_v4(),
                                label: format!("{}_proof", spec.name),
                                smtlib2: smtlib2_query.clone(),
                                proof_hv: last_ast_hv.clone().unwrap_or_else(|| {
                                    symthaea_core::hdc::ContinuousHV::zero(16384)
                                }),
                                verified: true,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64,
                            };
                            let _ =
                                tx.try_send(symthaea_swarm::SwarmMessage::ProofGossip(gossip_msg));
                        }
                    }
                    info!("Formal verification outcome: {}", details);
                }

                let contains_unsafe = source.contains("unsafe ") || source.contains("unsafe {");
                let is_proven = formally_verified.unwrap_or(false);

                // SOVEREIGN BLOCKADE: Hard gate for unsafe Rust (INV-13)
                // We physically block the return of any code containing unsafe blocks
                // unless they are mathematically proven safe by the SMT engine.
                if let Some(failure) = formal_gate_failure {
                    warn!(
                        "FORMAL CONTRACT GATE: Rejecting {} because SMT contract was refuted.",
                        spec.name
                    );
                    result.tests_failed = 1;
                    result.test_failures.push(failure);
                } else if contains_unsafe && !is_proven {
                    warn!(
                        "SOVEREIGN BLOCKADE: Rejecting unverified unsafe block in {}. SMT proof required for systems governance.",
                        spec.name
                    );
                    // Treat as test failure to trigger fix/retry logic
                    result.tests_failed = 1;
                    result.test_failures.push(TestFailure {
                        test_name: "SovereignSafetyGate".into(),
                        expected: Some("Proven Unsafe Block".into()),
                        actual: Some("Unverified Unsafe Block".into()),
                        message: "Memory-unsafe code detected without formal SMT proof.".into(),
                        assertion: None,
                    });
                } else {
                    // RECURSIVE ATTESTATION: only sign successful proofs.
                    let mut attestation = None;
                    if formally_verified == Some(true) {
                        // For attestation, we need the SMT query. If not in metadata, use empty.
                        let smt = spec.metadata.get("smtlib2").cloned().unwrap_or_default();

                        // Supply-chain integrity: Link formal proof to physical binary artifact (INV-14)
                        let binary_data = result
                            .binary_path
                            .as_ref()
                            .and_then(|p| std::fs::read(p).ok());

                        attestation = Some(SovereignAttestor::attest_with_process_key(
                            &spec.name,
                            &smt,
                            "Proven",
                            binary_data.as_deref(),
                        ));
                        info!(
                            "SOVEREIGN ATTESTATION: Generated signed proof and binary hash for {}.",
                            spec.name
                        );
                    }

                    return VerifiedCode {
                        source,
                        compiled: true,
                        tests_passed: true,
                        test_count_passed: result.tests_passed,
                        test_count_failed: 0,
                        compile_retries,
                        test_retries: 0,
                        formally_verified,
                        attestation,
                        confidence: VerificationConfidence::compute(true, true, is_proven, 1.0),
                        compile_errors: Vec::new(),
                        test_failures: Vec::new(),
                        diagnostic_hvs: all_diagnostic_hvs.clone(),
                        ast_hdc: ast_hdc.clone(),
                    };
                }
            }

            // If we reached here, tests failed OR the sovereign blockade was triggered
            if result.tests_failed > 0 {
                // Tests failed — try to fix based on test output
                let test_failures: Vec<String> = result
                    .test_failures
                    .iter()
                    .map(|f| {
                        format!(
                            "{}: expected={}, actual={}",
                            f.test_name,
                            f.expected.as_deref().unwrap_or("?"),
                            f.actual.as_deref().unwrap_or("?")
                        )
                    })
                    .collect();

                // Try test-fix retries
                let mut test_retries = 0;
                let mut latest_result = result.clone();
                let mut latest_test_failures = if test_failures.is_empty() {
                    vec![result.test_output.clone()]
                } else {
                    test_failures.clone()
                };
                let mut last_compile_errors = Vec::new();
                while test_retries < MAX_TEST_RETRIES {
                    test_retries += 1;

                    let retry_spec =
                        augment_spec_with_test_failures(spec, &latest_test_failures, test_retries);
                    let prediction_errors =
                        prediction_errors_from_execution(&latest_result, test_retries);
                    record_prediction_error_hvs(
                        &prediction_errors,
                        &mut all_diagnostic_hvs,
                        &mut seen_prediction_error_keys,
                    );
                    let retry_context = build_test_retry_context(
                        context,
                        &latest_test_failures,
                        &prediction_errors,
                        last_ast_hv.as_ref(),
                        last_structural_prior.as_ref(),
                        test_retries,
                    );
                    let retry_intent = CodeIntent::Create {
                        target: CodeTarget {
                            kind: EntityKind::Function,
                            name: retry_spec.name.clone(),
                            path: None,
                            language: Some(retry_spec.language.clone()),
                            hv: None,
                        },
                        spec: retry_spec,
                    };

                    let generated_retry = if let Some(llm) = llm_backend.as_ref() {
                        // retry_context.error_hints contains the test failures
                        let prompt = build_llm_repair_prompt(&retry_intent, &retry_context, &[]);
                        let params = GenerationParams::default();
                        let res = llm.generate(&prompt, &params).await;
                        match res {
                            Ok(src) => strip_markdown(&src),
                            Err(e) => {
                                error!(
                                    "LLM test-fix generation failed: {}, falling back to native",
                                    e
                                );
                                generator.generate(&retry_intent, &retry_context).source
                            }
                        }
                    } else {
                        generator.generate(&retry_intent, &retry_context).source
                    };
                    source = generated_retry;

                    let ast_observation = observe_ast_hdc(
                        &source,
                        test_retries,
                        &mut ast_hdc,
                        &structural_prototypes,
                        &structural_labels,
                    );
                    if let Some(hv) = ast_observation.hv {
                        last_ast_hv = Some(hv);
                    }
                    if let Some(prior) = ast_observation.structural_prior {
                        last_structural_prior = Some(prior);
                    }
                    let retry_structural_prediction_errors: Vec<_> =
                        ast_observation.prediction_error.into_iter().collect();
                    record_prediction_error_hvs(
                        &retry_structural_prediction_errors,
                        &mut all_diagnostic_hvs,
                        &mut seen_prediction_error_keys,
                    );
                    if ast_observation.fast_fail && test_retries < MAX_TEST_RETRIES {
                        latest_test_failures =
                            prediction_error_diagnostics(&retry_structural_prediction_errors);
                        if latest_test_failures.is_empty() {
                            latest_test_failures.push(
                                "AST-HDC structural prior rejected the candidate before rustc"
                                    .to_string(),
                            );
                        }
                        continue;
                    }

                    let mut retry_result = match intent {
                        CodeIntent::Solve {
                            target: repo_target,
                            ..
                        } => {
                            if let Err(e) = executor.apply_patch(&repo_target.root, &source) {
                                ExecutionResult {
                                    compiled: false,
                                    compile_errors: vec![format!("Patch Error: {}", e)],
                                    tests_passed: 0,
                                    tests_failed: 0,
                                    test_output: String::new(),
                                    runtime_error: None,
                                    elapsed: std::time::Duration::from_millis(0),
                                    simulated: false,
                                    binary_path: None,
                                    test_failures: Vec::new(),
                                }
                            } else {
                                executor.execute_workspace_tests(&repo_target.root)
                            }
                        }
                        _ => {
                            let retry_full_source = if test_source.is_empty() {
                                source.clone()
                            } else {
                                format!("{}\n\n{}", source, test_source)
                            };
                            executor.execute_rust_with_inline_tests(&retry_full_source)
                        }
                    };
                    retry_result.parse_test_failures();

                    if retry_result.simulated {
                        return simulated_verification_failure(
                            source,
                            compile_retries,
                            test_retries,
                        );
                    }

                    if retry_result.compiled && retry_result.tests_failed == 0 {
                        let mut formally_verified = None;
                        let mut formal_gate_failure = None;
                        if let Some(smtlib2_query) = spec.metadata.get("smtlib2") {
                            info!(
                                "Executing neuro-symbolic proof lookup on test-repair path for {}",
                                spec.name
                            );
                            let bridge = crate::z3_bridge::Z3Bridge::new();
                            let proof_memory = super::proof_memory::global_proof_memory();
                            let memory = proof_memory.lock().clone();
                            let mut engine =
                                super::proof_memory::CachedProofEngine::new(bridge, memory);

                            #[cfg(feature = "swarm")]
                            let proofs_slice = swarm_proofs.unwrap_or(&[]);
                            #[cfg(not(feature = "swarm"))]
                            let proofs_slice: &[()] = &[];
                            let (verdict, details) = engine.verify_with_cache(
                                &spec.name,
                                &source,
                                smtlib2_query,
                                0.85,
                                proofs_slice,
                            );
                            *proof_memory.lock() = engine.memory.clone();
                            let is_proven = verdict == super::proof_memory::ProofVerdict::Proven;
                            let is_refuted = verdict == super::proof_memory::ProofVerdict::Refuted;
                            formally_verified = Some(is_proven);

                            if is_refuted {
                                if let (Some(ref mut store), Some(hv)) =
                                    (experience_store.as_mut(), &last_ast_hv)
                                {
                                    info!(
                                        "LABELING NEGATIVE PROTOTYPE: {} disproven by Z3.",
                                        spec.name
                                    );
                                    store.store_negative_prototype(&spec.name, hv.values.clone());
                                }
                                formal_gate_failure = Some(TestFailure {
                                    test_name: "FormalContractGate".into(),
                                    expected: Some("SMT contract satisfiable/proven".into()),
                                    actual: Some("SMT contract refuted".into()),
                                    message: "Formal SMT contract was refuted on the repair path; candidate cannot be accepted.".into(),
                                    assertion: None,
                                });
                            }

                            if is_proven && engine.solver_calls > 0 {
                                #[cfg(feature = "swarm")]
                                if let Some(tx) = swarm_tx {
                                    let gossip_msg = symthaea_swarm::SwarmProofMsg {
                                        node_id: uuid::Uuid::new_v4(),
                                        label: format!("{}_proof", spec.name),
                                        smtlib2: smtlib2_query.clone(),
                                        proof_hv: last_ast_hv.clone().unwrap_or_else(|| {
                                            symthaea_core::hdc::ContinuousHV::zero(16384)
                                        }),
                                        verified: true,
                                        timestamp: std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_millis()
                                            as u64,
                                    };
                                    let _ = tx.try_send(symthaea_swarm::SwarmMessage::ProofGossip(
                                        gossip_msg,
                                    ));
                                }
                            }
                            info!("Formal verification outcome (retry lane): {}", details);
                        }

                        let contains_unsafe =
                            source.contains("unsafe ") || source.contains("unsafe {");
                        let is_proven = formally_verified.unwrap_or(false);
                        if let Some(failure) = formal_gate_failure {
                            warn!(
                                "FORMAL CONTRACT GATE: Rejecting repair candidate {} because SMT contract was refuted.",
                                spec.name
                            );
                            retry_result.tests_failed = 1;
                            retry_result.test_failures.push(failure);
                        } else if contains_unsafe && !is_proven {
                            warn!(
                                "SOVEREIGN BLOCKADE: Rejecting unverified unsafe block in repair candidate {}.",
                                spec.name
                            );
                            retry_result.tests_failed = 1;
                            retry_result.test_failures.push(TestFailure {
                                test_name: "SovereignSafetyGate".into(),
                                expected: Some("Proven Unsafe Block".into()),
                                actual: Some("Unverified Unsafe Block".into()),
                                message: "Memory-unsafe code detected without formal SMT proof on repair path.".into(),
                                assertion: None,
                            });
                        } else {
                            let mut attestation = None;
                            if formally_verified == Some(true) {
                                let smt = spec.metadata.get("smtlib2").cloned().unwrap_or_default();
                                let binary_data = retry_result
                                    .binary_path
                                    .as_ref()
                                    .and_then(|p| std::fs::read(p).ok());
                                attestation = Some(SovereignAttestor::attest_with_process_key(
                                    &spec.name,
                                    &smt,
                                    "Proven",
                                    binary_data.as_deref(),
                                ));
                            }

                            return VerifiedCode {
                                source,
                                compiled: true,
                                tests_passed: true,
                                test_count_passed: result.tests_passed,
                                test_count_failed: 0,
                                compile_retries,
                                test_retries: 0,
                                formally_verified,
                                attestation,
                                confidence: VerificationConfidence::compute(
                                    true, true, is_proven, 1.0,
                                ),
                                compile_errors: Vec::new(),
                                test_failures: Vec::new(),
                                diagnostic_hvs: all_diagnostic_hvs.clone(),
                                ast_hdc: ast_hdc.clone(),
                            };
                        }
                    }

                    if !retry_result.compiled {
                        last_compile_errors = retry_result.compile_errors.clone();
                        let stderr = retry_result.compile_errors.join("\n");
                        let mut prediction_errors =
                            prediction_errors_from_execution(&retry_result, compile_retries + 1);
                        prediction_errors.extend(retry_structural_prediction_errors);
                        record_prediction_error_hvs(
                            &prediction_errors,
                            &mut all_diagnostic_hvs,
                            &mut seen_prediction_error_keys,
                        );
                        let mut active_error_hints = retry_context.error_hints.clone();
                        active_error_hints.extend(prediction_error_hints(&prediction_errors));
                        if let Some(fixed) =
                            try_semantic_ast_repair(&source, &retry_result.compile_errors)
                        {
                            source = fixed;
                        } else if let Some(fixed) =
                            generator.try_auto_fix_with_hints(&source, &stderr, &active_error_hints)
                        {
                            source = fixed;
                        }
                    }

                    latest_test_failures = retry_result
                        .test_failures
                        .iter()
                        .map(|f| {
                            format!(
                                "{}: expected={}, actual={}",
                                f.test_name,
                                f.expected.as_deref().unwrap_or("?"),
                                f.actual.as_deref().unwrap_or("?")
                            )
                        })
                        .collect();
                    if latest_test_failures.is_empty() && !retry_result.test_output.is_empty() {
                        latest_test_failures.push(retry_result.test_output.clone());
                    }
                    latest_result = retry_result;
                }

                return VerifiedCode {
                    source,
                    compiled: latest_result.compiled,
                    tests_passed: false,
                    test_count_passed: latest_result.tests_passed,
                    test_count_failed: latest_result.tests_failed,
                    compile_retries,
                    test_retries,
                    formally_verified: None,
                    attestation: None,
                    confidence: VerificationConfidence::compute(
                        latest_result.compiled,
                        false,
                        false,
                        0.5,
                    ),
                    compile_errors: last_compile_errors,
                    test_failures: latest_test_failures,
                    diagnostic_hvs: all_diagnostic_hvs.clone(),
                    ast_hdc: ast_hdc.clone(),
                };
            }
        }

        // Compilation failed — try auto-fix
        last_compile_errors = result.compile_errors.clone();
        compile_retries += 1;

        if compile_retries > MAX_COMPILE_RETRIES {
            break;
        }

        // Notify LSP about the failing file so it can provide context
        if let Some(lsp) = lsp_client.as_mut() {
            let path = executor.work_dir().join("generated_test.rs");
            let _ = lsp.did_open(&path, "rust", &full_source);
        }

        // Apply auto-fix
        let stderr = result.compile_errors.join("\n");
        let mut prediction_errors = prediction_errors_from_execution(&result, compile_retries);
        prediction_errors.extend(structural_prediction_errors);
        let diagnostics = prediction_error_diagnostics(&prediction_errors);
        let categories = prediction_error_categories(&prediction_errors);
        record_prediction_error_hvs(
            &prediction_errors,
            &mut all_diagnostic_hvs,
            &mut seen_prediction_error_keys,
        );
        let mut active_error_hints = context.error_hints.clone();
        active_error_hints.extend(prediction_error_hints(&prediction_errors));
        active_error_hints.extend(repair_memory::repair_priors_for_spec_diagnostics(
            spec,
            &diagnostics,
            &categories,
            3,
        ));
        if let Some(fixed) = try_semantic_ast_repair(&source, &result.compile_errors)
            .or_else(|| generator.try_auto_fix_with_hints(&source, &stderr, &active_error_hints))
            .or_else(|| try_auto_fix(&source, &result.compile_errors))
        {
            source = fixed;
        } else {
            if repo_map.is_none() {
                break; // No fix available
            }

            let retry_spec =
                augment_spec_with_compile_errors(spec, &result.compile_errors, compile_retries);
            let retry_context = build_compile_retry_context(
                context,
                repo_map,
                lsp_client.as_deref_mut(),
                experience_store.as_deref_mut(),
                &mut all_diagnostic_hvs,
                &result.compile_errors,
                spec,
                &prediction_errors,
                last_ast_hv.as_ref(),
                last_structural_prior.as_ref(),
                compile_retries,
            );
            let retry_intent = CodeIntent::Create {
                target: CodeTarget {
                    kind: EntityKind::Function,
                    name: retry_spec.name.clone(),
                    path: None,
                    language: Some(retry_spec.language.clone()),
                    hv: None,
                },
                spec: retry_spec,
            };

            let generated = if let Some(llm) = llm_backend.as_ref() {
                // If LLM is available, use it to solve the problem if it's complex
                let prompt =
                    build_llm_repair_prompt(&retry_intent, &retry_context, &result.compile_errors);
                let params = GenerationParams::default();

                let res = llm.generate(&prompt, &params).await;

                match res {
                    Ok(src) => strip_markdown(&src),
                    Err(_) => generator.generate(&retry_intent, &retry_context).source,
                }
            } else {
                generator.generate(&retry_intent, &retry_context).source
            };

            source = generated;
        }
    }

    // Failed after all retries
    VerifiedCode {
        source,
        compiled: false,
        tests_passed: false,
        test_count_passed: 0,
        test_count_failed: 0,
        compile_retries,
        test_retries: 0,
        formally_verified: None,
        attestation: None,
        confidence: VerificationConfidence::compute(false, false, false, 0.0),
        compile_errors: last_compile_errors,
        test_failures: Vec::new(),
        diagnostic_hvs: all_diagnostic_hvs,
        ast_hdc,
    }
}

fn simulated_verification_failure(
    source: String,
    compile_retries: usize,
    test_retries: usize,
) -> VerifiedCode {
    let reason = "Verification aborted because execution was simulated; real compilation and test results are required"
        .to_string();
    VerifiedCode {
        source,
        compiled: false,
        tests_passed: false,
        test_count_passed: 0,
        test_count_failed: 0,
        compile_retries,
        test_retries,
        formally_verified: None,
        attestation: None,
        confidence: VerificationConfidence::compute(false, false, false, 0.0),
        compile_errors: vec![reason],
        test_failures: Vec::new(),
        diagnostic_hvs: Vec::new(),
        ast_hdc: AstHdcTrace::default(),
    }
}

fn generate_llm_blocking(
    llm: &dyn LLMBackend,
    prompt: &str,
    params: &GenerationParams,
) -> anyhow::Result<String> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread {
                tokio::task::block_in_place(|| handle.block_on(llm.generate(prompt, params)))
            } else {
                handle.block_on(llm.generate(prompt, params))
            }
        }
        Err(_) => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(llm.generate(prompt, params))
        }
    }
}

fn augment_spec_with_test_failures(
    spec: &CodeSpec,
    test_failures: &[String],
    retry_number: usize,
) -> CodeSpec {
    let mut retry_spec = spec.clone();
    retry_spec
        .constraints
        .extend(test_failures.iter().map(|failure| {
            format!(
                "Retry {} must satisfy this failing test feedback: {}",
                retry_number, failure
            )
        }));
    retry_spec
}

struct AstHdcObservation {
    hv: Option<ContinuousHV>,
    prediction_error: Option<CodingPredictionError>,
    structural_prior: Option<StructuralPriorScore>,
    fast_fail: bool,
}

fn observe_ast_hdc(
    source: &str,
    retry_number: usize,
    trace: &mut AstHdcTrace,
    structural_prototypes: &StructuralPrototypeBank,
    structural_labels: &StructuralPrototypeLabels,
) -> AstHdcObservation {
    if std::env::var_os("SYMTHAEA_DISABLE_AST_HDC_FEP").is_some() {
        return AstHdcObservation {
            hv: None,
            prediction_error: None,
            structural_prior: None,
            fast_fail: false,
        };
    }

    match encode_rust_ast_hdc(source, symthaea_core::hdc::unified_hv::HDC_DIMENSION) {
        Ok(encoded) => {
            let feature_count = encoded.features.values().sum();
            let structural_prior =
                structural_prototypes.score(&encoded.features, structural_labels);
            let prior_prediction_error = structural_prior
                .as_ref()
                .and_then(|prior| structural_prior_prediction_error(prior, retry_number));
            let fast_fail = prior_prediction_error.is_some()
                && std::env::var_os("SYMTHAEA_HARD_GEODESIC_REJECTION").is_some();
            if prior_prediction_error.is_some()
                && std::env::var_os("SYMTHAEA_GEODESIC_REJECTION_SHADOW").is_some()
            {
                trace.geodesic_rejection_shadow_hits += 1;
            }
            if fast_fail {
                trace.hard_geodesic_rejections += 1;
            }
            trace.parse_successes += 1;
            trace.feature_observations += 1;
            trace.total_feature_count += feature_count;
            trace.last_feature_count = feature_count;
            if trace.first_features.is_none() {
                trace.first_features = Some(encoded.features.clone());
            }
            if let Some(prior) = &structural_prior {
                trace.structural_prior_observations += 1;
                let previous = trace.last_structural_prior_score;
                trace.last_structural_prior_score = Some(prior.score);
                trace.last_structural_prior_label = Some(prior.label.clone());
                trace.best_structural_prior_score = Some(
                    trace
                        .best_structural_prior_score
                        .map(|best| best.max(prior.score))
                        .unwrap_or(prior.score),
                );
                if let Some(previous) = previous {
                    trace.structural_prior_delta = Some(prior.score - previous);
                }
            }
            trace.last_features = Some(encoded.features);
            AstHdcObservation {
                hv: Some(encoded.hv),
                prediction_error: prior_prediction_error,
                structural_prior,
                fast_fail,
            }
        }
        Err(error) => {
            trace.parse_failures += 1;
            trace.structural_prediction_errors += 1;
            AstHdcObservation {
                hv: None,
                prediction_error: Some(structural_prediction_error_from_ast_parse(
                    format!("Generated Rust did not parse as an AST before compilation: {error}"),
                    retry_number,
                )),
                structural_prior: None,
                fast_fail: std::env::var_os("SYMTHAEA_HARD_GEODESIC_REJECTION").is_some(),
            }
        }
    }
}

fn structural_prior_prediction_error(
    prior: &StructuralPriorScore,
    retry_number: usize,
) -> Option<CodingPredictionError> {
    let threshold = std::env::var("SYMTHAEA_STRUCTURAL_PRIOR_SURPRISE_THRESHOLD")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(0.25)
        .clamp(0.0, 1.0);
    if prior.score >= threshold {
        return None;
    }

    Some(structural_prediction_error_from_prior(
        format!(
            "Candidate AST similarity {:.3} is below structural prior threshold {:.3} for `{}`",
            prior.score, threshold, prior.label
        ),
        retry_number,
        1.0 - prior.score,
    ))
}

fn augment_spec_with_compile_errors(
    spec: &CodeSpec,
    compile_errors: &[String],
    retry_number: usize,
) -> CodeSpec {
    let mut retry_spec = spec.clone();
    retry_spec
        .constraints
        .extend(compile_errors.iter().map(|error| {
            format!(
                "Retry {} must resolve this compiler diagnostic: {}",
                retry_number, error
            )
        }));
    retry_spec
}

fn structural_prototypes_from_context(
    context: &CodeContext<'_>,
    spec: &CodeSpec,
) -> StructuralPrototypeBank {
    let mut bank = StructuralPrototypeBank::default();
    let labels = structural_labels_for_spec(spec);

    if let Some(template) = &context.learned_template {
        if let Some(features) = ast_features_for_source(template) {
            bank.observe_success(
                &features,
                &StructuralPrototypeLabels::new(
                    format!("template:{}", labels.category),
                    labels.return_shape.clone(),
                    "learned_template",
                ),
            );
        }
    }

    for (label, source) in &context.past_examples {
        if let Some(features) = ast_features_for_source(source) {
            bank.observe_success(
                &features,
                &StructuralPrototypeLabels::new(
                    format!("example:{}", structural_label_fragment(label)),
                    labels.return_shape.clone(),
                    "past_example",
                ),
            );
        }
    }

    bank
}

fn structural_labels_for_spec(spec: &CodeSpec) -> StructuralPrototypeLabels {
    StructuralPrototypeLabels::new(
        structural_label_fragment(&spec.purpose),
        spec.signature
            .as_deref()
            .map(return_shape_for_signature)
            .unwrap_or_else(|| "unit".to_string()),
        "candidate",
    )
}

fn structural_label_fragment(value: &str) -> String {
    let mut fragment = value
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if ch.is_whitespace() || matches!(ch, '-' | '_' | ':' | '/') {
                Some('_')
            } else {
                None
            }
        })
        .collect::<String>();
    while fragment.contains("__") {
        fragment = fragment.replace("__", "_");
    }
    let fragment = fragment.trim_matches('_');
    if fragment.is_empty() {
        "unknown".to_string()
    } else {
        fragment.chars().take(48).collect()
    }
}

fn build_compile_retry_context<'a>(
    context: &CodeContext<'a>,
    repo_map: Option<&'a RepoMap>,
    mut lsp_client: Option<&mut RustAnalyzerClient>,
    mut experience_store: Option<&mut CodingExperienceStore>,
    all_diagnostic_hvs: &mut Vec<ContinuousHV>,
    compile_errors: &[String],
    spec: &CodeSpec,
    prediction_errors: &[CodingPredictionError],
    ast_hv: Option<&ContinuousHV>,
    structural_prior: Option<&StructuralPriorScore>,
    retry_number: usize,
) -> CodeContext<'a> {
    let diagnostic_context =
        repo_map.map(|map| map.code_context_for_compile_errors(compile_errors, 5));

    let mut source_files = context.source_files.clone();
    let mut error_hints = context.error_hints.clone();
    let mut diagnostic_hvs = context.diagnostic_hvs.clone();
    let mut learned_template = context.learned_template.clone();

    error_hints.extend(prediction_error_hints(prediction_errors));
    error_hints.extend(repair_memory::repair_priors_for_spec_diagnostics(
        spec,
        &prediction_error_diagnostics(prediction_errors),
        &prediction_error_categories(prediction_errors),
        3,
    ));
    diagnostic_hvs.extend(
        prediction_errors
            .iter()
            .map(|error| error.diagnostic_hv.clone()),
    );
    if let Some(ast_hv) = ast_hv {
        diagnostic_hvs.push(ast_hv.clone());
        error_hints.push((
            format!("ast_hdc_structural_context_retry_{retry_number}"),
            "AST-HDC structural context from the previous candidate is available; preserve useful structure while repairing the compiler error."
                .to_string(),
        ));
    }
    if let Some(prior) = structural_prior {
        error_hints.push((
            format!(
                "ast_hdc_structural_prior_retry_{}_{}",
                retry_number,
                sanitize_hint_label(&prior.label)
            ),
            format!(
                "Nearest successful AST prototype is `{}` with similarity {:.3}. Move the repair toward this known-good structure while preserving the requested semantics.",
                prior.label, prior.score
            ),
        ));
    }

    // Neuro-symbolic encoding: translate diagnostics into HDC geometry
    let encoder = DiagnosticHDEncoder::default_dim();
    let structured_errors =
        crate::language::code_executor::parse_structured_errors(&compile_errors.join("\n"));
    for error in &structured_errors {
        let hv = encoder.encode_diagnostic(error);
        diagnostic_hvs.push(hv.clone());
        all_diagnostic_hvs.push(hv.clone());

        // EXPERIENTIAL RECALL: If we have an experience store, query for past fixes for this failure geometry.
        // This closes the loop: "I've felt this error before, and this code fixed it."
        if let Some(ref mut store) = experience_store {
            let diagnostic_query = hv.clone();

            if learned_template.is_none() {
                // Since we are already on a potentially multi-threaded runtime,
                // and the store uses async database calls, we use block_in_place
                // to safely wait for the result without deadlocking the executor.
                let recalled = tokio::task::block_in_place(|| {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(store.learned_template_for_diagnostic(&diagnostic_query))
                });

                if let Some(template) = recalled {
                    learned_template = Some(template);
                }
            }
        }
    }

    if let Some(repo_context) = diagnostic_context {
        for (label, snippet) in repo_context.source_files {
            if !source_files
                .iter()
                .any(|(existing_label, _)| existing_label == &label)
            {
                source_files.push((label, snippet));
            }
        }
        error_hints.extend(repo_context.error_hints);
    }

    // Enrich with LSP if available
    if let (Some(lsp), Some(map)) = (lsp_client.as_mut(), repo_map) {
        let diagnostics = map.attach_compile_errors(compile_errors);
        for diag in diagnostics {
            if let (Some(file), Some(line), Some(col)) =
                (&diag.error.file, diag.error.line, diag.error.column)
            {
                let pos = LspPosition::new(line as u32 - 1, col as u32 - 1);

                // 1. Get hover information (enrich error hints)
                if let Ok(Some(hover)) = lsp.hover(file, pos) {
                    error_hints.push((
                        format!(
                            "lsp_hover_{}",
                            diag.error.code.as_deref().unwrap_or("error")
                        ),
                        format!("LSP analysis at {file}:{line}:{col}: {}", hover.contents),
                    ));
                }

                // 2. Get definitions (enrich source snippets)
                if let Ok(locations) = lsp.goto_definition(file, pos) {
                    let lsp_context = map.code_context_for_lsp_locations(&locations);
                    for (label, snippet) in lsp_context.source_files {
                        if !source_files
                            .iter()
                            .any(|(existing_label, _)| existing_label == &label)
                        {
                            source_files.push((label, snippet));
                        }
                    }
                }
            }
        }
    }

    if repo_map.is_none() {
        error_hints.extend(compile_errors.iter().map(|error| {
            let category = categorize_rustc_diagnostic(error);
            let hint = repair_hint_for_rustc_category(category);
            (
                format!("compile_error_retry_{retry_number}_{category}"),
                format!(
                    "Fix the generated function so this compiler diagnostic is resolved: {error}. Repair hint: {hint}"
                ),
            )
        }));
    }

    CodeContext {
        memory: repo_map.map(|map| map.memory()).or(context.memory),
        context_hvs: context.context_hvs.clone(),
        source_files,
        past_examples: context.past_examples.clone(),
        mcts_plan_confidence: context.mcts_plan_confidence,
        negative_prototypes: context.negative_prototypes.clone(),
        error_hints,
        diagnostic_hvs,
        issue_text: context.issue_text.clone(),
        learned_template,
    }
}

fn record_prediction_error_hvs(
    prediction_errors: &[CodingPredictionError],
    all_diagnostic_hvs: &mut Vec<ContinuousHV>,
    seen_prediction_error_keys: &mut HashSet<String>,
) {
    for error in prediction_errors {
        if seen_prediction_error_keys.insert(error.key.clone()) {
            all_diagnostic_hvs.push(error.diagnostic_hv.clone());
        }
    }
}

fn categorize_rustc_diagnostic(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("e0308") || lower.contains("mismatched types") {
        "type_mismatch"
    } else if lower.contains("e0425") || lower.contains("not found in this scope") {
        "unresolved_identifier"
    } else if lower.contains("e0599") || lower.contains("no method named") {
        "missing_method"
    } else if lower.contains("e0369") || lower.contains("cannot calculate") {
        "invalid_operator"
    } else if lower.contains("e0507") || lower.contains("cannot move out") {
        "move_out_of_borrow"
    } else if lower.contains("e0382") || lower.contains("use of moved value") {
        "use_after_move"
    } else if lower.contains("e0596") || lower.contains("cannot borrow") {
        "borrow_mutability"
    } else if lower.contains("expected one of")
        || lower.contains("unknown start of token")
        || lower.contains("expected expression")
    {
        "parse_failure"
    } else if lower.contains("lifetime") {
        "lifetime_error"
    } else if lower.contains("trait bound") {
        "trait_bound"
    } else {
        "rustc_error"
    }
}

fn repair_hint_for_rustc_category(category: &str) -> &'static str {
    match category {
        "type_mismatch" => {
            "align the returned expression and function-call arguments with the declared signature"
        }
        "unresolved_identifier" => {
            "use an in-scope parameter/local binding or introduce the missing binding before use"
        }
        "missing_method" => {
            "call the method on the element type, not the whole collection, or choose a method available for the receiver type"
        }
        "invalid_operator" => {
            "apply the operator to scalar elements rather than collection/reference values"
        }
        "move_out_of_borrow" => {
            "borrow, clone, copy, or iterate by reference instead of moving from a borrowed value"
        }
        "use_after_move" => {
            "avoid using a value after move; borrow, clone, copy, or reorder the operations"
        }
        "borrow_mutability" => "mark the binding mutable or switch to an immutable operation",
        "parse_failure" => {
            "emit only Rust source code with balanced braces, complete signatures, and valid statement separators"
        }
        "lifetime_error" => {
            "return references derived from inputs or return owned values instead of local temporaries"
        }
        "trait_bound" => {
            "add the required trait bound or choose an implementation that does not require the missing trait"
        }
        _ => {
            "use the rustc diagnostic to repair the smallest local type, syntax, or ownership inconsistency"
        }
    }
}

fn build_test_retry_context<'a>(
    context: &CodeContext<'a>,
    test_failures: &[String],
    prediction_errors: &[CodingPredictionError],
    ast_hv: Option<&ContinuousHV>,
    structural_prior: Option<&StructuralPriorScore>,
    retry_number: usize,
) -> CodeContext<'a> {
    let mut error_hints = context.error_hints.clone();
    error_hints.extend(test_failures.iter().map(|failure| {
        (
            format!("test_failure_retry_{retry_number}"),
            format!("Fix the generated function so this failure no longer occurs: {failure}"),
        )
    }));
    error_hints.extend(prediction_error_hints(prediction_errors));
    if ast_hv.is_some() {
        error_hints.push((
            format!("ast_hdc_structural_context_retry_{retry_number}"),
            "AST-HDC structural context from the previous candidate is available; preserve useful structure while repairing the test failure."
                .to_string(),
        ));
    }
    if let Some(prior) = structural_prior {
        error_hints.push((
            format!(
                "ast_hdc_structural_prior_retry_{}_{}",
                retry_number,
                sanitize_hint_label(&prior.label)
            ),
            format!(
                "Nearest successful AST prototype is `{}` with similarity {:.3}. Adjust behavior without drifting away from this known-good structure.",
                prior.label, prior.score
            ),
        ));
    }

    CodeContext {
        memory: context.memory,
        context_hvs: context.context_hvs.clone(),
        source_files: context.source_files.clone(),
        past_examples: context.past_examples.clone(),
        mcts_plan_confidence: context.mcts_plan_confidence,
        negative_prototypes: context.negative_prototypes.clone(),
        error_hints,
        diagnostic_hvs: {
            let mut diagnostic_hvs = context.diagnostic_hvs.clone();
            diagnostic_hvs.extend(
                prediction_errors
                    .iter()
                    .map(|error| error.diagnostic_hv.clone()),
            );
            if let Some(ast_hv) = ast_hv {
                diagnostic_hvs.push(ast_hv.clone());
            }
            diagnostic_hvs
        },
        issue_text: context.issue_text.clone(),
        learned_template: context.learned_template.clone(),
    }
}

fn sanitize_hint_label(label: &str) -> String {
    let sanitized = label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('_').chars().take(48).collect()
}

/// Generate and verify a function from minimal inputs (convenience wrapper).
///
/// # Example
/// ```ignore
/// let result = generate_verified_function(
///     &generator, &mut executor,
///     "add", "Add two integers",
///     "fn add(a: i32, b: i32) -> i32",
///     &[("add(2, 3)", "5"), ("add(0, 0)", "0")],
/// );
/// assert!(result.is_guaranteed());
/// ```
pub fn generate_verified_function(
    generator: &CodeGenerator,
    executor: &mut CodeExecutor,
    name: &str,
    purpose: &str,
    signature: &str,
    examples: &[(&str, &str)],
) -> VerifiedCode {
    let spec = CodeSpec {
        language: "rust".into(),
        name: name.into(),
        purpose: purpose.into(),
        purpose_hv: None,
        signature: Some(signature.into()),
        constraints: Vec::new(),
        examples: examples
            .iter()
            .map(|(i, o)| (i.to_string(), o.to_string()))
            .collect(),
        epistemic_status: EpistemicStatus::Certain,
        metadata: HashMap::new(),
    };

    let context = CodeContext::default();
    let target =
        CodeTarget::new(spec.name.clone(), EntityKind::Function).with_language(&spec.language);
    let intent = CodeIntent::Create { target, spec };

    generate_verified(generator, executor, &intent, &context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::code_encoder::CodeHDEncoder;
    use crate::language::repo_map::RepoMap;

    fn make_generator() -> CodeGenerator {
        CodeGenerator::new(CodeHDEncoder::new(512))
    }

    #[test]
    fn test_geodesic_rejection_shadow_mode() {
        let generator = make_generator();
        let mut executor = CodeExecutor::with_real_execution();

        // Enable shadow mode
        unsafe {
            std::env::set_var("SYMTHAEA_GEODESIC_REJECTION_SHADOW", "1");
            // Set a high threshold to trigger a "shadow hit" (since the prototype bank will be dissimilar)
            std::env::set_var("SYMTHAEA_STRUCTURAL_PRIOR_SURPRISE_THRESHOLD", "0.99");
        }

        // Seed context with a past example to ensure StructuralPrototypeBank is NOT empty
        let mut context = CodeContext::default();
        context.past_examples.push((
            "add_one".to_string(),
            "pub fn add_one(x: i32) -> i32 { x + 1 }".to_string(),
        ));

        let result = generate_verified_function_with_context(
            &generator,
            &mut executor,
            "add",
            "Add two integers",
            "fn add(a: i32, b: i32) -> i32",
            &[("add(2, 3)", "5")],
            &context,
        );

        // We expect at least one shadow hit because the candidate "add(a, b)"
        // will be dissimilar enough from "add_one(x)" to fall below 0.99
        assert!(result.ast_hdc.geodesic_rejection_shadow_hits > 0);

        // Since it actually compiled, it should be a false positive
        assert!(result.ast_hdc.geodesic_rejection_shadow_false_positives > 0);

        unsafe {
            std::env::remove_var("SYMTHAEA_GEODESIC_REJECTION_SHADOW");
            std::env::remove_var("SYMTHAEA_STRUCTURAL_PRIOR_SURPRISE_THRESHOLD");
        }
    }

    /// Helper for testing with custom context
    fn generate_verified_function_with_context(
        generator: &CodeGenerator,
        executor: &mut CodeExecutor,
        name: &str,
        purpose: &str,
        signature: &str,
        examples: &[(&str, &str)],
        context: &CodeContext<'_>,
    ) -> VerifiedCode {
        let spec = CodeSpec {
            language: "rust".into(),
            name: name.into(),
            purpose: purpose.into(),
            purpose_hv: None,
            signature: Some(signature.into()),
            constraints: Vec::new(),
            examples: examples
                .iter()
                .map(|(i, o)| (i.to_string(), o.to_string()))
                .collect(),
            epistemic_status: EpistemicStatus::Certain,
            metadata: HashMap::new(),
        };

        let target =
            CodeTarget::new(spec.name.clone(), EntityKind::Function).with_language(&spec.language);
        let intent = CodeIntent::Create { target, spec };

        generate_verified(generator, executor, &intent, context)
    }

    #[test]
    fn test_verified_generation_simple_add() {
        let generator = make_generator();
        let mut executor = CodeExecutor::with_real_execution();

        let result = generate_verified_function(
            &generator,
            &mut executor,
            "add",
            "Add two integers",
            "fn add(a: i32, b: i32) -> i32",
            &[("add(2, 3)", "5")],
        );

        assert!(
            result.compiled,
            "add() should compile: {:?}",
            result.compile_errors
        );
    }

    #[test]
    fn test_verified_generation_reverse_string() {
        let generator = make_generator();
        let mut executor = CodeExecutor::with_real_execution();

        let result = generate_verified_function(
            &generator,
            &mut executor,
            "reverse_string",
            "Reverse a string",
            "fn reverse_string(s: &str) -> String",
            &[],
        );

        assert!(
            result.compiled,
            "reverse_string() should compile: {:?}",
            result.compile_errors
        );
    }

    #[test]
    fn test_verified_code_summary() {
        let verified = VerifiedCode {
            source: "fn add(a: i32, b: i32) -> i32 { a + b }".into(),
            compiled: true,
            tests_passed: true,
            test_count_passed: 3,
            test_count_failed: 0,
            compile_retries: 0,
            test_retries: 0,
            formally_verified: Some(true),
            attestation: None,
            confidence: VerificationConfidence::compute(true, true, true, 1.0),
            compile_errors: Vec::new(),
            test_failures: Vec::new(),
            diagnostic_hvs: Vec::new(),
            ast_hdc: AstHdcTrace::default(),
        };

        assert!(verified.is_guaranteed());
        assert!(verified.summary().contains("VERIFIED"));
        assert!(verified.summary().contains("Z3 PROVEN"));
        assert!(verified.confidence.confidence > 0.9);
    }

    #[test]
    fn test_compile_retry_context_uses_repo_map_diagnostics() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "src/config.rs",
            "pub struct EngineConfig {\n    pub enabled: bool,\n}\n",
        )
        .unwrap();
        let compile_errors =
            vec!["error[E0412]: cannot find type `EngineConfig` in this scope".to_string()];
        let spec = CodeSpec::new("rust", "load_config", "Load engine config");
        let result = ExecutionResult {
            compiled: false,
            compile_errors: compile_errors.clone(),
            tests_passed: 0,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: std::time::Duration::from_millis(1),
            simulated: false,
            binary_path: None,
            test_failures: Vec::new(),
        };
        let prediction_errors = prediction_errors_from_execution(&result, 1);

        let context = build_compile_retry_context(
            &CodeContext::default(),
            Some(&repo),
            None,
            None,
            &mut Vec::new(),
            &compile_errors,
            &spec,
            &prediction_errors,
            None,
            None,
            1,
        );

        assert!(context.memory.is_some());
        assert!(
            context
                .source_files
                .iter()
                .any(|(_, snippet)| snippet.contains("pub struct EngineConfig"))
        );
        assert!(
            context
                .error_hints
                .iter()
                .any(|(pattern, hint)| pattern == "compile_error_E0412"
                    && hint.contains("cannot find type"))
        );
        assert!(context.error_hints.iter().any(|(pattern, hint)| {
            pattern.starts_with("fep_prediction_error") && hint.contains("Prediction-error signal")
        }));
    }

    #[test]
    fn test_confidence_computation() {
        // Full verification
        let full = VerificationConfidence::compute(true, true, true, 1.0);
        assert!(full.confidence > 0.9);

        // Compiled but tests fail
        let partial = VerificationConfidence::compute(true, false, false, 0.5);
        assert!(partial.confidence > 0.3);
        assert!(partial.confidence < 0.6);

        // Nothing works
        let fail = VerificationConfidence::compute(false, false, false, 0.0);
        assert!(fail.confidence < 0.1);
    }

    #[test]
    fn test_verified_generation_rejects_simulated_execution() {
        let generator = make_generator();
        let mut executor = CodeExecutor::new();

        let result = generate_verified_function(
            &generator,
            &mut executor,
            "add",
            "Add two integers",
            "fn add(a: i32, b: i32) -> i32",
            &[("add(2, 3)", "5")],
        );

        assert!(!result.compiled);
        assert!(!result.tests_passed);
        assert!(
            result
                .compile_errors
                .iter()
                .any(|err| err.contains("requires real execution")),
            "expected real-execution failure, got {:?}",
            result.compile_errors
        );
    }
}

/// Build a comprehensive prompt for the LLM to perform a surgical code fix.
pub fn build_llm_repair_prompt(
    intent: &CodeIntent,
    context: &CodeContext<'_>,
    errors: &[String],
) -> String {
    let mut prompt = String::new();
    prompt.push_str("# Engineering Task: Repair Failing Code\n\n");

    match intent {
        CodeIntent::Create { spec, .. } => {
            prompt.push_str(&format!("## Objective: {}\n", spec.purpose));
            prompt.push_str(&format!(
                "## Target Signature: `{}`\n",
                spec.signature.as_deref().unwrap_or("unknown")
            ));

            if !spec.constraints.is_empty() {
                prompt.push_str("\n## Constraints:\n");
                for c in &spec.constraints {
                    prompt.push_str(&format!("- {}\n", c));
                }
            }

            if !spec.examples.is_empty() {
                prompt.push_str("\n## Required Test Cases:\n");
                for (input, output) in &spec.examples {
                    prompt.push_str(&format!("- Input: `{}` -> Expected: `{}`\n", input, output));
                }
            }
        }
        _ => prompt.push_str("## Objective: Repair existing code to pass compilation and tests.\n"),
    }

    prompt.push_str("\n## Compilation Errors Encountered:\n");
    for err in errors {
        prompt.push_str(&format!("```text\n{}\n```\n", err));
    }

    if !context.error_hints.is_empty() {
        prompt.push_str("\n## Analysis Hints:\n");
        for (_, hint) in &context.error_hints {
            prompt.push_str(&format!("- {}\n", hint));
        }
    }

    if !context.source_files.is_empty() {
        prompt.push_str("\n## Relevant Context (RepoMap/LSP):\n");
        for (label, snippet) in &context.source_files {
            prompt.push_str(&format!("### {}\n```rust\n{}\n```\n", label, snippet));
        }
    }

    prompt.push_str("\n## Instructions:\n");
    prompt.push_str("1. Analyze the errors and the provided context.\n");
    prompt.push_str("2. Fix the code so it compiles successfully.\n");
    prompt.push_str(
        "3. DO NOT include any explanatory text, only the raw source code of the implementation.\n",
    );
    prompt.push_str("4. Return ONLY the code, no markdown blocks if possible, or just the content within blocks.\n");

    prompt
}

/// Helper to strip markdown code blocks from an LLM response.
fn strip_markdown(s: &str) -> String {
    let s = s.trim();
    if s.starts_with("```") {
        // Find the first newline after the opening triple-backticks
        if let Some(newline_pos) = s.find('\n') {
            // Find the last triple-backticks
            if let Some(end_pos) = s.rfind("```") {
                if end_pos > newline_pos {
                    return s[newline_pos + 1..end_pos].trim().to_string();
                }
            }
        }
    }
    s.to_string()
}
