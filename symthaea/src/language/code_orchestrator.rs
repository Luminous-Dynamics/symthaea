// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Code Orchestrator
//!
//! Routes code synthesis requests through multiple backends in priority order,
//! producing verified code with a full audit trail.
//!
//! ```text
//! SynthesisRequest
//!     ↓
//! ┌─────────────────────────────────────────────────┐
//! │ 0. Geodesic skeleton prior (feature-gated)       │ → verify
//! │ 1. CodeGenerator (template matching, HDC+CfC)    │ → verify
//! │ 2. CodeAlgebra analogy (if similar code exists)  │ → verify
//! │ 3. LLM fallback (Ollama / Cloud)                 │ → verify
//! └─────────────────────────────────────────────────┘
//!     ↓
//! SynthesisResponse (code + verification audit trail)
//! ```
//!
//! The orchestrator wraps the existing `IntelligentDispatcher` and `CodeGenerator`,
//! adding verification layers and code-by-analogy before falling back to LLM.

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use symthaea_core::synthesis_trait::{
    BackendCapabilities, CodeSynthesisBackend, SynthesisRequest, SynthesisResponse,
    VerificationLayer, VerificationReport,
};

use super::code_certificate::CodeCertificate;
use super::code_executor::{CodeExecutor, ExecutionResult};
use super::code_generator::{CodeContext, CodeGenerator, GeneratedCode};
use super::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use super::code_parser::EntityKind;
use super::code_verifier::{CodeVerifier, VerificationPolicy};
use super::llm_backend::{LLMBackend, create_backend_from_env};
use super::repair_memory;
use super::repair_taxonomy;
use super::repo_map::{RepoMap, RepoMapStats};
use super::rust_lsp::RustAnalyzerClient;
use super::structural_prototype::{ast_features_for_source, return_shape_for_signature};
use super::verified_generation::{AstHdcTrace, generate_verified_full};
use crate::coding_experience::CodingExperienceStore;
use crate::hdc::code_algebra::CodeAlgebra;
use crate::hdc::code_encoder::CodeHDEncoder;
use crate::mind::structured_thought::EpistemicStatus;

/// Unified code synthesis orchestrator.
///
/// Tries backends in priority order based on confidence and capabilities.
/// Each attempt is verified before acceptance. The full audit trail is
/// recorded in the response.
pub struct CodeOrchestrator {
    /// Native HDC+CfC code generator (template matching)
    generator: CodeGenerator,
    /// Round-trip verification
    verifier: CodeVerifier,
    /// Thread-safe mutable state
    state: Arc<Mutex<OrchestratorState>>,
    /// HDC code algebra for analogy-based generation
    algebra: CodeAlgebra,
    /// Minimum similarity threshold for accepting generated code
    verification_policy: VerificationPolicy,
    /// Optional AST/HDC repository map used to enrich generation context.
    repo_map: Option<RepoMap>,
}

/// Internal mutable state of the orchestrator.
struct OrchestratorState {
    /// Real execution sandbox for compiler-grounded verification
    executor: CodeExecutor,
    /// Optional LSP client for high-precision repair
    lsp_client: Option<RustAnalyzerClient>,
    /// Optional experience store for diagnostic-based recall
    experience_store: Option<CodingExperienceStore>,
    /// LLM backend for final fallback and distillation teaching
    llm_backend: Arc<dyn LLMBackend>,
    /// Energy budget remaining for this session
    energy_budget: f32,
    /// Total energy spent across all generations
    energy_spent: f32,
    /// Audit trail of all synthesis attempts this session
    attempt_history: Vec<SynthesisAttempt>,
    /// Distillation buffer: (ThoughtChannels-as-f32-vec, source_code, quality) triples
    /// captured from successful generations for Broca SSM training.
    distillation_buffer: Vec<DistillationCapture>,
    /// Sequencer training buffer: (purpose, plan_actions) pairs
    sequencer_training_buffer: Vec<SequencerTrainingCapture>,
    /// Number of captures before triggering sequencer retraining
    sequencer_retrain_threshold: usize,
    /// Certificates issued for accepted code
    certificates: Vec<CodeCertificate>,
}

/// A captured (purpose, plan_actions) pair for CfC sequencer training.
///
/// When an LLM fallback succeeds, we analyze its output to infer what
/// plan the native sequencer should have produced. This closes the loop:
/// LLM generates → capture plan → retrain sequencer → native handles it next time.
#[derive(Debug, Clone)]
pub struct SequencerTrainingCapture {
    /// The purpose/intent text
    pub purpose: String,
    /// Inferred plan actions the sequencer should learn
    pub target_actions: Vec<crate::dynamics::cfc_code_sequencer::PlanAction>,
    /// Quality of the generation (0.0-1.0)
    pub quality: f32,
    /// Which backend produced the successful output
    pub backend: String,
}

/// A captured (intent, code, quality) triple for Broca SSM training.
///
/// When the orchestrator produces verified code (from any backend),
/// it captures the intent-to-code mapping for distillation training.
/// Over time, this trains the native CfC-HDC network to reproduce
/// what the LLM generated, eliminating the LLM dependency.
#[derive(Debug, Clone)]
pub struct DistillationCapture {
    /// The 43-element channel vector encoding the intent
    pub channels: Vec<f32>,
    /// The verified source code
    pub source: String,
    /// Quality score (0.0-1.0): verification similarity × plan coverage
    pub quality: f32,
    /// Which backend produced this code
    pub backend: String,
    /// Function/type name for tracking
    pub name: String,
    /// Original request purpose for retrieval and curriculum shaping.
    pub purpose: String,
    /// Original signature, when available.
    pub signature: Option<String>,
    /// Normalized return-shape bucket for structural retrieval.
    pub return_shape: String,
}

/// Record of a single synthesis attempt (for audit trail)
#[derive(Debug, Clone)]
pub struct SynthesisAttempt {
    /// Which backend was tried
    pub backend: String,
    /// Whether verification passed
    pub verified: bool,
    /// Semantic similarity score
    pub similarity: f32,
    /// Energy cost of this attempt
    pub energy_cost: f32,
    /// Surprise scalar from the verified generation loop.
    pub surprise: f32,
    /// Number of distinct diagnostic hypervectors captured during repair.
    pub diagnostic_hv_count: usize,
    /// Successful AST-HDC parse observations captured during verification.
    pub ast_hdc_parse_successes: usize,
    /// AST-HDC parse failures captured as structural surprise.
    pub ast_hdc_parse_failures: usize,
    /// Structural prediction errors emitted by AST-HDC observation.
    pub structural_prediction_errors: usize,
    /// Last observed AST-HDC feature count.
    pub ast_hdc_feature_count: usize,
    /// Last observed AST-HDC feature map.
    pub ast_hdc_last_features: Option<BTreeMap<String, usize>>,
    /// Number of structural-prior comparisons made during verification.
    pub structural_prior_observations: usize,
    /// Last similarity against a known successful AST prototype.
    pub structural_prior_score: Option<f32>,
    /// Label of the nearest successful AST prototype.
    pub structural_prior_label: Option<String>,
    /// Last movement in structural-prior score across retries.
    pub structural_prior_delta: Option<f32>,
    /// Why it was rejected (if applicable)
    pub rejection_reason: Option<String>,
    /// Short source preview for diagnostics and benchmark reporting.
    pub source_preview: Option<String>,
    /// Repair priors made available to this backend from previous failures.
    pub repair_prior_count: usize,
    /// Labels for repair priors made available to this backend.
    pub repair_prior_labels: Vec<String>,
}

/// Conversion from synthesis_trait EpistemicStatus to crate-local EpistemicStatus
fn to_local_epistemic(status: symthaea_core::synthesis_trait::EpistemicStatus) -> EpistemicStatus {
    match status {
        symthaea_core::synthesis_trait::EpistemicStatus::Certain => EpistemicStatus::Certain,
        symthaea_core::synthesis_trait::EpistemicStatus::Probable => EpistemicStatus::Probable,
        symthaea_core::synthesis_trait::EpistemicStatus::Uncertain => EpistemicStatus::Uncertain,
        symthaea_core::synthesis_trait::EpistemicStatus::Unknown => EpistemicStatus::Unknown,
        symthaea_core::synthesis_trait::EpistemicStatus::OutOfDomain => {
            EpistemicStatus::OutOfDomain
        }
    }
}

/// Conversion from crate-local EpistemicStatus to synthesis_trait EpistemicStatus
fn to_trait_epistemic(status: EpistemicStatus) -> symthaea_core::synthesis_trait::EpistemicStatus {
    match status {
        EpistemicStatus::Certain => symthaea_core::synthesis_trait::EpistemicStatus::Certain,
        EpistemicStatus::Probable => symthaea_core::synthesis_trait::EpistemicStatus::Probable,
        EpistemicStatus::Uncertain => symthaea_core::synthesis_trait::EpistemicStatus::Uncertain,
        EpistemicStatus::Unknown => symthaea_core::synthesis_trait::EpistemicStatus::Unknown,
        EpistemicStatus::OutOfDomain => {
            symthaea_core::synthesis_trait::EpistemicStatus::OutOfDomain
        }
    }
}

impl CodeOrchestrator {
    /// Create a new orchestrator with default configuration.
    pub fn new() -> Self {
        let encoder = CodeHDEncoder::default_dim();
        let verifier_encoder = CodeHDEncoder::default_dim();
        let algebra_encoder = CodeHDEncoder::default_dim();

        Self {
            generator: CodeGenerator::new(encoder),
            verifier: CodeVerifier::new(verifier_encoder),
            state: Arc::new(Mutex::new(OrchestratorState {
                executor: CodeExecutor::with_real_execution(),
                lsp_client: None,
                experience_store: None,
                llm_backend: create_backend_from_env(),
                energy_budget: 100.0,
                energy_spent: 0.0,
                attempt_history: Vec::new(),
                distillation_buffer: Vec::new(),
                sequencer_training_buffer: Vec::new(),
                sequencer_retrain_threshold: 50,
                certificates: Vec::new(),
            })),
            algebra: CodeAlgebra::new(algebra_encoder),
            verification_policy: VerificationPolicy::Standard,
            repo_map: None,
        }
    }

    /// Attach a Rust Analyzer LSP client for high-precision repairs.
    pub fn with_lsp(self, lsp_client: RustAnalyzerClient) -> Self {
        self.state.lock().lsp_client = Some(lsp_client);
        self
    }

    /// Attach a persistent coding experience store for diagnostic-based recall.
    pub fn with_experience_store(self, store: CodingExperienceStore) -> Self {
        self.state.lock().experience_store = Some(store);
        self
    }

    /// Attach a custom LLM backend for fallback.
    pub fn with_llm_backend(self, backend: Arc<dyn LLMBackend>) -> Self {
        self.state.lock().llm_backend = backend;
        self
    }

    /// Create with a specific verification policy.
    pub fn with_policy(mut self, policy: VerificationPolicy) -> Self {
        self.verification_policy = policy;
        self.verifier.set_policy(policy);
        self
    }

    /// Create with a custom energy budget.
    pub fn with_energy_budget(self, budget: f32) -> Self {
        self.state.lock().energy_budget = budget;
        self
    }

    /// Attach an already-built repository map.
    pub fn with_repo_map(mut self, repo_map: RepoMap) -> Self {
        self.repo_map = Some(repo_map);
        self
    }

    /// Index a project into the orchestrator's AST/HDC repository map.
    pub fn index_project(&mut self, root: impl AsRef<Path>) -> io::Result<RepoMapStats> {
        let root = root.as_ref();
        let mut repo_map = RepoMap::new(root.to_path_buf());
        let stats = repo_map.scan()?;
        self.repo_map = Some(repo_map);
        Ok(stats)
    }

    /// Get the indexed repository map, if present.
    pub fn repo_map(&self) -> Option<&RepoMap> {
        self.repo_map.as_ref()
    }

    /// Clear any indexed repository context.
    pub fn clear_repo_map(&mut self) {
        self.repo_map = None;
    }

    /// Get remaining energy budget.
    pub fn remaining_energy(&self) -> f32 {
        let state = self.state.lock();
        state.energy_budget - state.energy_spent
    }

    /// Get total energy spent.
    pub fn energy_spent(&self) -> f32 {
        self.state.lock().energy_spent
    }

    /// Get the audit trail of all attempts.
    pub fn attempt_history(&self) -> Vec<SynthesisAttempt> {
        self.state.lock().attempt_history.clone()
    }

    /// Get the distillation buffer — captured (intent, code, quality) triples.
    ///
    /// These are ready to be serialized as JSONL for Broca SSM training.
    pub fn distillation_buffer(&self) -> Vec<DistillationCapture> {
        self.state.lock().distillation_buffer.clone()
    }

    /// Get issued certificates.
    pub fn certificates(&self) -> Vec<CodeCertificate> {
        self.state.lock().certificates.clone()
    }

    /// Export the distillation buffer as JSONL suitable for Broca training.
    ///
    /// Each line is a JSON object with `channels` (43-element f32 vec) and
    /// `target_text` (the verified source code). This format is directly
    /// consumable by `TrainingDataset::from_jsonl()`.
    pub fn export_distillation_jsonl(&self) -> String {
        use serde_json::json;

        let state = self.state.lock();
        state
            .distillation_buffer
            .iter()
            .map(|cap| {
                json!({
                    "channels": cap.channels,
                    "target_text": cap.source,
                    "target_ids": [],
                    "metadata": {
                        "backend": cap.backend,
                        "name": cap.name,
                        "purpose": cap.purpose,
                        "signature": cap.signature,
                        "return_shape": cap.return_shape,
                        "quality": cap.quality,
                    }
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Save distillation buffer to a JSONL file for Broca training.
    pub fn save_distillation(&self, path: &std::path::Path) -> std::io::Result<()> {
        let jsonl = self.export_distillation_jsonl();
        if jsonl.is_empty() {
            return Ok(());
        }
        std::fs::write(path, jsonl)
    }

    /// Import verified code-shape memory from distillation JSONL.
    ///
    /// This accepts the format produced by `export_distillation_jsonl()`. It is
    /// intentionally conservative: only records whose `target_text` parses as
    /// Rust AST are admitted as structural memory, so generic Broca language
    /// samples cannot contaminate code-shape retrieval.
    pub fn import_distillation_jsonl(&self, jsonl: &str) -> io::Result<usize> {
        let mut imported = Vec::new();
        for (line_idx, line) in jsonl.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid distillation JSONL at line {}: {error}",
                        line_idx + 1
                    ),
                )
            })?;
            let Some(source) = value.get("target_text").and_then(serde_json::Value::as_str) else {
                continue;
            };
            if ast_features_for_source(source).is_none() {
                continue;
            }

            let channels = value
                .get("channels")
                .cloned()
                .and_then(|channels| serde_json::from_value::<Vec<f32>>(channels).ok())
                .unwrap_or_else(|| vec![0.0; 43]);
            let metadata = value.get("metadata");
            let signature = metadata
                .and_then(|metadata| metadata.get("signature"))
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string);
            let return_shape = metadata
                .and_then(|metadata| metadata.get("return_shape"))
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
                .or_else(|| signature.as_deref().map(return_shape_for_signature))
                .unwrap_or_else(|| "unit".to_string());

            imported.push(DistillationCapture {
                channels,
                source: source.to_string(),
                quality: metadata
                    .and_then(|metadata| metadata.get("quality"))
                    .and_then(serde_json::Value::as_f64)
                    .map(|quality| quality as f32)
                    .unwrap_or(1.0),
                backend: metadata
                    .and_then(|metadata| metadata.get("backend"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("imported_distillation")
                    .to_string(),
                name: metadata
                    .and_then(|metadata| metadata.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("imported_{}", line_idx + 1)),
                purpose: metadata
                    .and_then(|metadata| metadata.get("purpose"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                signature,
                return_shape,
            });
        }

        let count = imported.len();
        self.state.lock().distillation_buffer.extend(imported);
        Ok(count)
    }

    /// Load verified code-shape memory from a distillation JSONL file.
    pub fn load_distillation(&self, path: &std::path::Path) -> io::Result<usize> {
        let jsonl = std::fs::read_to_string(path)?;
        self.import_distillation_jsonl(&jsonl)
    }

    /// Capture a successful generation for distillation training.
    ///
    /// Converts the SynthesisRequest into a ThoughtChannels-compatible f32 vector
    /// and stores it alongside the verified source code. This is the core of the
    /// self-improvement flywheel: every successful generation becomes training data.
    fn capture_distillation(
        &self,
        request: &SynthesisRequest,
        source: &str,
        backend: &str,
        similarity: f32,
        surprise: f32,
        diagnostic_hvs: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    ) {
        // Build a 43-element channel vector from the request
        let mut channels = vec![0.0f32; 43];

        // Channel 0: semantic intent = "code generation" (Answer=1.0)
        channels[0] = 1.0;

        // Channel 8: epistemic status ordinal
        channels[8] = match request.epistemic_status {
            symthaea_core::synthesis_trait::EpistemicStatus::Certain => 0.0,
            symthaea_core::synthesis_trait::EpistemicStatus::Probable => 1.0,
            symthaea_core::synthesis_trait::EpistemicStatus::Uncertain => 2.0,
            symthaea_core::synthesis_trait::EpistemicStatus::Unknown => 3.0,
            symthaea_core::synthesis_trait::EpistemicStatus::OutOfDomain => 4.0,
        };

        // Channel 12: consciousness level (psi)
        channels[12] = request.consciousness_level;

        // Channel 14: coherence (use similarity as proxy)
        channels[14] = similarity;

        // Channel 18: has_computed_answer = true (we have verified code)
        channels[18] = 1.0;

        // Channel 19: concept_count (number of constraints + examples)
        channels[19] = (request.constraints.len() + request.examples.len()).min(10) as f32;

        // Channels 24-27: code-specific
        channels[24] = 0.5; // syntax_complexity (mid-range default)
        channels[25] = similarity; // type_confidence (use verification similarity)
        channels[26] = 0.5; // algorithm_pattern
        channels[27] = surprise; // error_likelihood (surprise from failures)

        // Epistemic Cube: E3 (proven), N2 (network), M2 (persistent)
        channels[31] = 1.0; // E3 (proven by compilation)
        channels[35] = 1.0; // N2 (network consensus)
        channels[39] = 1.0; // M2 (persistent/recorded)
        channels[41] = similarity; // H (harmonic coherence)
        channels[42] = 0.4 * 0.6 + 0.35 * 0.5 + 0.25 * 0.5; // Quality score

        // Quality = similarity × (1 - error_penalty)
        let quality = similarity;

        let mut state = self.state.lock();
        let return_shape = request
            .signature
            .as_deref()
            .map(return_shape_for_signature)
            .unwrap_or_else(|| "unit".to_string());
        state.distillation_buffer.push(DistillationCapture {
            channels,
            source: source.to_string(),
            quality,
            backend: backend.to_string(),
            name: request.name.clone(),
            purpose: request.purpose.clone(),
            signature: request.signature.clone(),
            return_shape,
        });

        // === EXPERIENTIAL MEMORY: Store the fix geometry ===
        // If we have a store, and we have diagnostic HVs from the retry loop,
        // we store the successful template indexed by the error it fixed.
        if let Some(ref mut store) = state.experience_store {
            for hv in &diagnostic_hvs {
                let task = request.purpose.clone();
                let code = source.to_string();
                store.store_learned_template(&task, &code, Some(hv));
            }
        }

        // === Flywheel: also capture for sequencer training ===
        // Infer the plan actions the CfC sequencer should have produced
        // based on the structure of the successful output.
        let target_actions = Self::infer_plan_from_source(source);
        state
            .sequencer_training_buffer
            .push(SequencerTrainingCapture {
                purpose: request.purpose.clone(),
                target_actions,
                quality,
                backend: backend.to_string(),
            });
    }

    /// Infer PlanAction sequence from generated source code.
    ///
    /// Analyzes the code structure to determine what plan steps the
    /// sequencer should learn to produce for similar inputs.
    fn infer_plan_from_source(
        source: &str,
    ) -> Vec<crate::dynamics::cfc_code_sequencer::PlanAction> {
        use crate::dynamics::cfc_code_sequencer::PlanAction;

        let mut actions = Vec::new();

        // Detect structural elements present in the code
        if source.contains("struct ") {
            actions.push(PlanAction::DefineStruct);
        }
        if source.contains("enum ") {
            actions.push(PlanAction::DefineEnum);
        }
        if source.contains("trait ") {
            actions.push(PlanAction::DefineTrait);
        }
        if source.contains("fn ") {
            actions.push(PlanAction::DefineFunction);

            // Count parameters
            if let Some(paren_start) = source.find('(') {
                if let Some(paren_end) = source[paren_start..].find(')') {
                    let params = &source[paren_start + 1..paren_start + paren_end];
                    let param_count = if params.trim().is_empty() {
                        0
                    } else {
                        params.split(',').count()
                    };
                    for _ in 0..param_count {
                        actions.push(PlanAction::AddParameter);
                    }
                }
            }

            // Return type
            if source.contains("->") {
                actions.push(PlanAction::SetReturnType);
            }
        }
        if source.contains("impl ") {
            actions.push(PlanAction::ImplTrait);
        }
        if source.contains("use ") {
            actions.push(PlanAction::AddImport);
        }
        if source.contains("Result<")
            || source.contains("Result ")
            || source.contains(".map_err(")
            || source.contains("?;")
        {
            actions.push(PlanAction::AddErrorHandling);
        }
        if source.contains("///") || source.contains("//!") {
            actions.push(PlanAction::AddDocumentation);
        }

        // Phase 4: detect new action types
        if source.contains("match ") {
            actions.push(PlanAction::MatchExpression);
        }
        if source.contains("for ") && source.contains(" in ") {
            actions.push(PlanAction::ForLoop);
        }
        if source.contains(".iter()") || source.contains(".into_iter()") {
            actions.push(PlanAction::IteratorChain);
        }
        if source.contains("|") && (source.contains("||") || source.contains("| ")) {
            // Simple heuristic for closures — avoid false positives from logical OR
            if source.contains(".map(|")
                || source.contains(".filter(|")
                || source.contains(".for_each(|")
                || source.contains("= |")
            {
                actions.push(PlanAction::ClosureDefine);
            }
        }
        if source.contains("?;") || source.contains("?)") {
            actions.push(PlanAction::ErrorPropagation);
        }
        if source.contains("<T") || source.contains("<T>") || source.contains("<T:") {
            actions.push(PlanAction::GenericParam);
        }
        if source.contains("'a") || source.contains("'static") {
            actions.push(PlanAction::LifetimeAnnotation);
        }
        if source.contains("#[derive(") {
            actions.push(PlanAction::DeriveAttribute);
        }
        if source.contains("#[test]") || source.contains("#[cfg(test)]") {
            actions.push(PlanAction::TestModule);
        }
        if source.contains("const ") {
            actions.push(PlanAction::ConstDefinition);
        }
        if source.contains("type ") && source.contains(" = ") {
            actions.push(PlanAction::TypeAlias);
        }

        // Ensure at least DefineFunction
        if actions.is_empty() {
            actions.push(PlanAction::DefineFunction);
        }
        actions.push(PlanAction::Complete);

        actions
    }

    /// Get the sequencer training buffer for retraining.
    pub fn sequencer_training_buffer(&self) -> Vec<SequencerTrainingCapture> {
        self.state.lock().sequencer_training_buffer.clone()
    }

    /// Take the sequencer training buffer (drains it for consumption by the trainer).
    pub fn take_sequencer_training_buffer(&self) -> Vec<SequencerTrainingCapture> {
        std::mem::take(&mut self.state.lock().sequencer_training_buffer)
    }

    /// Check if the sequencer training buffer has reached the retrain threshold.
    pub fn should_retrain_sequencer(&self) -> bool {
        let state = self.state.lock();
        state.sequencer_training_buffer.len() >= state.sequencer_retrain_threshold
    }

    /// Generate a CodeCertificate for verified code.
    fn issue_certificate(
        &self,
        request: &SynthesisRequest,
        source: &str,
        backend: &str,
        source_provenance: &str,
        verification_layers: &[VerificationLayer],
        similarity: f32,
    ) -> CodeCertificate {
        let mut cert = CodeCertificate::new(source, backend, similarity)
            .with_epistemic_status(request.epistemic_status)
            .with_source_provenance(source_provenance)
            .with_verification_layers(verification_layers)
            .with_safety_critical(request.safety_critical);

        if let Some(gcs) = self.gcs_certificate_metadata(source, &request.name) {
            cert = cert.with_topology(gcs.beta_0, gcs.beta_1, gcs.beta_2);
            if let Some(convergence) = gcs.oracle_convergence {
                cert = cert.with_oracle_convergence(convergence);
            }
            if let Some(sheaf_coherent) = gcs.sheaf_coherent {
                cert = cert.with_sheaf_coherent(sheaf_coherent);
            }
        }

        self.state.lock().certificates.push(cert.clone());
        cert
    }

    /// Source-derived GCS metadata for certificates.
    ///
    /// This is deliberately post-acceptance metadata: compiler/test verification
    /// remains the acceptance gate, while GCS records structural evidence about
    /// the accepted source.
    #[cfg(feature = "geodesic_synthesis")]
    fn gcs_certificate_metadata(
        &self,
        source: &str,
        function_name: &str,
    ) -> Option<GcsCertificateMetadata> {
        use symthaea_geodesic::{
            ExecutionOracle, ProgramDependenceGraph, TopologicalFingerprint,
            execution_oracle::OperationType,
        };

        let pdg = ProgramDependenceGraph::from_rust_source(source, function_name);
        let fingerprint = TopologicalFingerprint::from_complex(&pdg.to_simplicial_complex());
        let beta_1 = fingerprint.betti.beta_1.max(pdg.loop_count());

        let oracle_convergence = if beta_1 > 0 {
            let mut oracle = ExecutionOracle::new();
            let statements: Vec<_> = source
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    !trimmed.is_empty() && !trimmed.starts_with("//")
                })
                .map(|line| {
                    let op = OperationType::classify(line);
                    let hv = symthaea_core::hdc::binary_hv::BinaryHV::random(
                        line.bytes()
                            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
                    );
                    (hv, op)
                })
                .collect();
            Some(oracle.predict_sequence(&statements).output_similarity)
        } else {
            None
        };

        let sheaf = symthaea_geodesic::verify_rust_v0_sheaf_coherence(source, function_name);

        Some(GcsCertificateMetadata {
            beta_0: fingerprint.betti.beta_0,
            beta_1,
            beta_2: fingerprint.betti.beta_2,
            oracle_convergence,
            sheaf_coherent: Some(sheaf.coherent),
        })
    }

    #[cfg(not(feature = "geodesic_synthesis"))]
    fn gcs_certificate_metadata(
        &self,
        _source: &str,
        _function_name: &str,
    ) -> Option<GcsCertificateMetadata> {
        None
    }

    /// Synthesize code by trying backends in priority order.
    ///
    /// Priority:
    /// 0. Geodesic skeleton prior (energy: 0.75) — topology-first candidate seed
    /// 1. Native CodeGenerator (energy: 1.0) — template matching via HDC+CfC
    /// 2. Code Algebra analogy (energy: 0.5) — "A:B :: C:?" in HDC space
    /// 3. LLM fallback (energy: 10.0) — verified final tier
    ///
    /// Each attempt is verified against the verification policy before acceptance.
    pub fn synthesize(&self, request: &SynthesisRequest) -> SynthesisResponse {
        let mut verification_layers = Vec::new();
        let mut repair_priors = repair_memory::repair_priors_for_request(request, 3);

        // ─── Backend 0: Geodesic Skeleton Prior ────────────────────────────
        // Candidate generator only. The generated skeleton is accepted only if
        // the normal compiler/test verification loop proves it.
        #[cfg(feature = "geodesic_synthesis")]
        if self.remaining_energy() >= 0.75 {
            let geodesic_result = self.try_geodesic_generation(request, &repair_priors);
            self.state.lock().energy_spent += 0.75;
            let geodesic_rejection = geodesic_result.rejection.clone();

            let attempt = SynthesisAttempt {
                backend: "GeodesicSkeleton".to_string(),
                verified: geodesic_result.verified,
                similarity: geodesic_result.similarity,
                energy_cost: 0.75,
                surprise: geodesic_result.surprise,
                diagnostic_hv_count: geodesic_result.diagnostic_hvs.len(),
                ast_hdc_parse_successes: geodesic_result.ast_hdc.parse_successes,
                ast_hdc_parse_failures: geodesic_result.ast_hdc.parse_failures,
                structural_prediction_errors: geodesic_result.ast_hdc.structural_prediction_errors,
                ast_hdc_feature_count: geodesic_result.ast_hdc.last_feature_count,
                ast_hdc_last_features: geodesic_result.ast_hdc.last_features.clone(),
                structural_prior_observations: geodesic_result
                    .ast_hdc
                    .structural_prior_observations,
                structural_prior_score: geodesic_result.ast_hdc.last_structural_prior_score,
                structural_prior_label: geodesic_result.ast_hdc.last_structural_prior_label.clone(),
                structural_prior_delta: geodesic_result.ast_hdc.structural_prior_delta,
                rejection_reason: geodesic_rejection.clone(),
                source_preview: source_preview(&geodesic_result.source),
                repair_prior_count: repair_priors.len(),
                repair_prior_labels: repair_prior_labels(&repair_priors),
            };
            self.state.lock().attempt_history.push(attempt);

            if geodesic_result.verified {
                verification_layers.push(VerificationLayer {
                    name: "geodesic_skeleton".to_string(),
                    passed: true,
                    score: Some(geodesic_result.similarity),
                    detail: "Topology-first skeleton candidate passed verification".to_string(),
                });

                self.capture_distillation(
                    request,
                    &geodesic_result.source,
                    "GeodesicSkeleton",
                    geodesic_result.similarity,
                    geodesic_result.surprise,
                    geodesic_result.diagnostic_hvs,
                );

                let certificate = self.issue_certificate(
                    request,
                    &geodesic_result.source,
                    "GeodesicSkeleton",
                    &geodesic_result.source_provenance,
                    &verification_layers,
                    geodesic_result.similarity,
                );

                return SynthesisResponse {
                    source: geodesic_result.source,
                    backend_name: "GeodesicSkeleton".to_string(),
                    confidence: geodesic_result.similarity,
                    epistemic_status: to_trait_epistemic(geodesic_result.epistemic),
                    verification: verification_layers,
                    accepted: true,
                    energy_cost: 0.75,
                    narrative: Some(format!(
                        "Geodesic skeleton generation succeeded (similarity: {:.3}, cert: {})",
                        geodesic_result.similarity, certificate.id
                    )),
                };
            }

            verification_layers.push(VerificationLayer {
                name: "geodesic_skeleton".to_string(),
                passed: false,
                score: Some(geodesic_result.similarity),
                detail: geodesic_rejection.clone().unwrap_or_else(|| {
                    "Geodesic skeleton candidate failed verification".to_string()
                }),
            });
            if let Some(reason) = geodesic_rejection {
                repair_priors.push(repair_prior_from_rejection("GeodesicSkeleton", &reason));
            }
        }

        // ─── Backend 1: Native CodeGenerator ───────────────────────────────
        if self.remaining_energy() >= 1.0 {
            let native_result = self.try_native_generation(request, &repair_priors);
            self.state.lock().energy_spent += 1.0;
            let native_rejection = native_result.rejection.clone();

            let attempt = SynthesisAttempt {
                backend: "CodeGenerator".to_string(),
                verified: native_result.verified,
                similarity: native_result.similarity,
                energy_cost: 1.0,
                surprise: native_result.surprise,
                diagnostic_hv_count: native_result.diagnostic_hvs.len(),
                ast_hdc_parse_successes: native_result.ast_hdc.parse_successes,
                ast_hdc_parse_failures: native_result.ast_hdc.parse_failures,
                structural_prediction_errors: native_result.ast_hdc.structural_prediction_errors,
                ast_hdc_feature_count: native_result.ast_hdc.last_feature_count,
                ast_hdc_last_features: native_result.ast_hdc.last_features.clone(),
                structural_prior_observations: native_result.ast_hdc.structural_prior_observations,
                structural_prior_score: native_result.ast_hdc.last_structural_prior_score,
                structural_prior_label: native_result.ast_hdc.last_structural_prior_label.clone(),
                structural_prior_delta: native_result.ast_hdc.structural_prior_delta,
                rejection_reason: native_rejection.clone(),
                source_preview: source_preview(&native_result.source),
                repair_prior_count: repair_priors.len(),
                repair_prior_labels: repair_prior_labels(&repair_priors),
            };
            self.state.lock().attempt_history.push(attempt);

            if native_result.verified {
                verification_layers.push(VerificationLayer {
                    name: "native_generation".to_string(),
                    passed: true,
                    score: Some(native_result.similarity),
                    detail: "HDC+CfC template generation passed verification".to_string(),
                });

                // ── Distillation: capture for Broca SSM training ──
                self.capture_distillation(
                    request,
                    &native_result.source,
                    "CodeGenerator",
                    native_result.similarity,
                    native_result.surprise,
                    native_result.diagnostic_hvs,
                );

                // ── Certificate: machine-verifiable audit trail ──
                let certificate = self.issue_certificate(
                    request,
                    &native_result.source,
                    "CodeGenerator",
                    &native_result.source_provenance,
                    &verification_layers,
                    native_result.similarity,
                );

                return SynthesisResponse {
                    source: native_result.source,
                    backend_name: "CodeGenerator".to_string(),
                    confidence: native_result.similarity,
                    epistemic_status: to_trait_epistemic(native_result.epistemic),
                    verification: verification_layers,
                    accepted: true,
                    energy_cost: 1.0,
                    narrative: Some(format!(
                        "Native HDC+CfC generation succeeded (similarity: {:.3}, cert: {})",
                        native_result.similarity, certificate.id
                    )),
                };
            }

            verification_layers.push(VerificationLayer {
                name: "native_generation".to_string(),
                passed: false,
                score: Some(native_result.similarity),
                detail: native_result
                    .rejection
                    .unwrap_or_else(|| "Below verification threshold".to_string()),
            });
            if let Some(reason) = native_rejection {
                repair_priors.push(repair_prior_from_rejection("CodeGenerator", &reason));
            }
        }

        // ─── Backend 2: Code Algebra Analogy ───────────────────────────────
        if self.remaining_energy() >= 0.5 {
            let analogy_result = self.try_analogy_generation(request, &repair_priors);
            self.state.lock().energy_spent += 0.5;
            let analogy_rejection = analogy_result.rejection.clone();

            let attempt = SynthesisAttempt {
                backend: "CodeAlgebra::analogy".to_string(),
                verified: analogy_result.verified,
                similarity: analogy_result.similarity,
                energy_cost: 0.5,
                surprise: analogy_result.surprise,
                diagnostic_hv_count: analogy_result.diagnostic_hvs.len(),
                ast_hdc_parse_successes: analogy_result.ast_hdc.parse_successes,
                ast_hdc_parse_failures: analogy_result.ast_hdc.parse_failures,
                structural_prediction_errors: analogy_result.ast_hdc.structural_prediction_errors,
                ast_hdc_feature_count: analogy_result.ast_hdc.last_feature_count,
                ast_hdc_last_features: analogy_result.ast_hdc.last_features.clone(),
                structural_prior_observations: analogy_result.ast_hdc.structural_prior_observations,
                structural_prior_score: analogy_result.ast_hdc.last_structural_prior_score,
                structural_prior_label: analogy_result.ast_hdc.last_structural_prior_label.clone(),
                structural_prior_delta: analogy_result.ast_hdc.structural_prior_delta,
                rejection_reason: analogy_rejection.clone(),
                source_preview: source_preview(&analogy_result.source),
                repair_prior_count: repair_priors.len(),
                repair_prior_labels: repair_prior_labels(&repair_priors),
            };
            self.state.lock().attempt_history.push(attempt);

            if analogy_result.verified {
                verification_layers.push(VerificationLayer {
                    name: "analogy_generation".to_string(),
                    passed: true,
                    score: Some(analogy_result.similarity),
                    detail: "Code-by-analogy generation passed verification".to_string(),
                });

                // ── Distillation: capture for Broca SSM training ──
                self.capture_distillation(
                    request,
                    &analogy_result.source,
                    "CodeAlgebra::analogy",
                    analogy_result.similarity,
                    analogy_result.surprise,
                    analogy_result.diagnostic_hvs,
                );

                // ── Certificate: machine-verifiable audit trail ──
                let certificate = self.issue_certificate(
                    request,
                    &analogy_result.source,
                    "CodeAlgebra::analogy",
                    &analogy_result.source_provenance,
                    &verification_layers,
                    analogy_result.similarity,
                );

                return SynthesisResponse {
                    source: analogy_result.source,
                    backend_name: "CodeAlgebra::analogy".to_string(),
                    confidence: analogy_result.similarity,
                    epistemic_status: to_trait_epistemic(analogy_result.epistemic),
                    verification: verification_layers,
                    accepted: true,
                    energy_cost: 0.5,
                    narrative: Some(format!(
                        "Analogy-based generation succeeded (similarity: {:.3}, cert: {})",
                        analogy_result.similarity, certificate.id
                    )),
                };
            }

            verification_layers.push(VerificationLayer {
                name: "analogy_generation".to_string(),
                passed: false,
                score: Some(analogy_result.similarity),
                detail: analogy_result
                    .rejection
                    .unwrap_or_else(|| "No suitable analogy found".to_string()),
            });
            if let Some(reason) = analogy_rejection {
                repair_priors.push(repair_prior_from_rejection("CodeAlgebra::analogy", &reason));
            }
        }

        // ─── Backend 3: LLM Fallback (Final Tier) ──────────────────────────
        // (Cost: 10.0 energy)
        if self.remaining_energy() >= 10.0 {
            let llm_result = self.try_llm_generation(request, &repair_priors);
            self.state.lock().energy_spent += 10.0;

            let attempt = SynthesisAttempt {
                backend: format!("LLM:{}", self.state.lock().llm_backend.name()),
                verified: llm_result.verified,
                similarity: llm_result.similarity,
                energy_cost: 10.0,
                surprise: llm_result.surprise,
                diagnostic_hv_count: llm_result.diagnostic_hvs.len(),
                ast_hdc_parse_successes: llm_result.ast_hdc.parse_successes,
                ast_hdc_parse_failures: llm_result.ast_hdc.parse_failures,
                structural_prediction_errors: llm_result.ast_hdc.structural_prediction_errors,
                ast_hdc_feature_count: llm_result.ast_hdc.last_feature_count,
                ast_hdc_last_features: llm_result.ast_hdc.last_features.clone(),
                structural_prior_observations: llm_result.ast_hdc.structural_prior_observations,
                structural_prior_score: llm_result.ast_hdc.last_structural_prior_score,
                structural_prior_label: llm_result.ast_hdc.last_structural_prior_label.clone(),
                structural_prior_delta: llm_result.ast_hdc.structural_prior_delta,
                rejection_reason: llm_result.rejection.clone(),
                source_preview: source_preview(&llm_result.source),
                repair_prior_count: repair_priors.len(),
                repair_prior_labels: repair_prior_labels(&repair_priors),
            };
            self.state.lock().attempt_history.push(attempt);

            if llm_result.verified {
                verification_layers.push(VerificationLayer {
                    name: "llm_fallback".to_string(),
                    passed: true,
                    score: Some(llm_result.similarity),
                    detail: "High-fidelity LLM generation passed verification".to_string(),
                });

                // ── Distillation: capture for Broca SSM training ──
                // This is where the LLM "teaches" the native model.
                self.capture_distillation(
                    request,
                    &llm_result.source,
                    &format!("LLM:{}", self.state.lock().llm_backend.name()),
                    llm_result.similarity,
                    llm_result.surprise,
                    llm_result.diagnostic_hvs,
                );

                let certificate = self.issue_certificate(
                    request,
                    &llm_result.source,
                    &format!("LLM:{}", self.state.lock().llm_backend.name()),
                    &llm_result.source_provenance,
                    &verification_layers,
                    llm_result.similarity,
                );

                return SynthesisResponse {
                    source: llm_result.source,
                    backend_name: format!("LLM:{}", self.state.lock().llm_backend.name()),
                    confidence: llm_result.similarity,
                    epistemic_status: to_trait_epistemic(llm_result.epistemic),
                    verification: verification_layers,
                    accepted: true,
                    energy_cost: 10.0,
                    narrative: Some(format!(
                        "LLM fallback generation succeeded (sim: {:.3}, cert: {})",
                        llm_result.similarity, certificate.id
                    )),
                };
            }

            verification_layers.push(VerificationLayer {
                name: "llm_fallback".to_string(),
                passed: false,
                score: Some(llm_result.similarity),
                detail: llm_result
                    .rejection
                    .unwrap_or_else(|| "LLM failed to produce verified code".to_string()),
            });
        }

        // ─── All native backends exhausted ─────────────────────────────────
        let energy_spent = self.state.lock().energy_spent;
        SynthesisResponse {
            source: String::new(),
            backend_name: "none".to_string(),
            confidence: 0.0,
            epistemic_status: symthaea_core::synthesis_trait::EpistemicStatus::Unknown,
            verification: verification_layers,
            accepted: false,
            energy_cost: energy_spent,
            narrative: Some(
                "All synthesis tiers exhausted. No verified solution found.".to_string(),
            ),
        }
    }

    /// Attempt code generation via a topology-first geodesic skeleton.
    ///
    /// This is a structural prior, not a verifier. It seeds the normal
    /// compile/test repair loop with source emitted from `SkeletonCombinator`.
    #[cfg(feature = "geodesic_synthesis")]
    fn try_geodesic_generation(
        &self,
        request: &SynthesisRequest,
        repair_priors: &[(String, String)],
    ) -> BackendResult {
        use symthaea_geodesic::{
            BettiNumbers, build_skeleton_from_topology, emit_rust_from_skeleton,
        };

        let has_repair_memory_prior = repair_priors
            .iter()
            .any(|(label, _)| label.starts_with("repair_memory_"));
        if !has_repair_memory_prior {
            if let Some(reason) = repair_taxonomy::forced_geodesic_rejection_unless_repair_memory(
                &request.constraints,
            ) {
                return BackendResult {
                    source: String::new(),
                    source_provenance: "forced_repair_probe_memory_sensitive".to_string(),
                    verified: false,
                    similarity: 0.0,
                    surprise: 1.0,
                    diagnostic_hvs: Vec::new(),
                    ast_hdc: AstHdcTrace::default(),
                    epistemic: EpistemicStatus::Uncertain,
                    rejection: Some(format!(
                        "[category={}; repair_hint={}] {}",
                        repair_taxonomy::categorize_rejection(reason),
                        repair_taxonomy::repair_hint_for_category(
                            repair_taxonomy::categorize_rejection(reason)
                        ),
                        reason
                    )),
                };
            }
        }

        if let Some(reason) = repair_taxonomy::forced_geodesic_rejection(&request.constraints) {
            return BackendResult {
                source: String::new(),
                source_provenance: "forced_repair_probe".to_string(),
                verified: false,
                similarity: 0.0,
                surprise: 1.0,
                diagnostic_hvs: Vec::new(),
                ast_hdc: AstHdcTrace::default(),
                epistemic: EpistemicStatus::Uncertain,
                rejection: Some(format!(
                    "[category={}; repair_hint={}] {}",
                    repair_taxonomy::categorize_rejection(reason),
                    repair_taxonomy::repair_hint_for_category(
                        repair_taxonomy::categorize_rejection(reason)
                    ),
                    reason
                )),
            };
        }

        let spec = self.request_to_spec(request);
        let target =
            CodeTarget::new(&request.name, EntityKind::Function).with_language(&request.language);
        let intent = CodeIntent::Create {
            target,
            spec: spec.clone(),
        };

        let profile = symthaea_geodesic::classify_geodesic_request(
            &request.name,
            &request.purpose,
            request.signature.as_deref(),
            &request.constraints,
        );
        let target_betti = profile.betti;
        let hints = profile.hints;
        let hint_refs: Vec<&str> = hints.iter().map(String::as_str).collect();

        let mut skeleton = build_skeleton_from_topology(&target_betti, &hint_refs);
        symthaea_geodesic::fill_skeleton_defaults_for_signature(
            &mut skeleton,
            request.signature.as_deref(),
        );

        let signature = request
            .signature
            .as_deref()
            .map(symthaea_geodesic::normalize_signature_for_geodesic_emitter);
        let signature_ref = signature.as_deref();

        let skeleton_source =
            match emit_rust_from_skeleton(&skeleton, &request.name, signature_ref, &hint_refs) {
                Some(source) => source,
                None => {
                    return BackendResult {
                        source: String::new(),
                        source_provenance: "geodesic_emit_failed".to_string(),
                        verified: false,
                        similarity: 0.0,
                        surprise: 1.0,
                        diagnostic_hvs: Vec::new(),
                        ast_hdc: AstHdcTrace::default(),
                        epistemic: EpistemicStatus::Uncertain,
                        rejection: Some(
                            "Geodesic skeleton contained unfilled slots and could not emit Rust"
                                .to_string(),
                        ),
                    };
                }
            };

        let mut context = self.context_for_request(request);
        context.error_hints.extend(repair_priors.iter().cloned());
        context.learned_template = Some(skeleton_source.clone());
        let seed_sheaf =
            symthaea_geodesic::verify_rust_v0_sheaf_coherence(&skeleton_source, &request.name);
        for diagnostic in seed_sheaf.diagnostics {
            let category = symthaea_geodesic::categorize_rust_v0_sheaf_diagnostic(&diagnostic);
            let hint = symthaea_geodesic::repair_hint_for_rust_v0_sheaf_category(category);
            context.error_hints.push((
                format!("geodesic_seed_sheaf_{category}"),
                format!("{diagnostic}. Repair hint: {hint}"),
            ));
        }
        context.past_examples.push((
            format!("geodesic skeleton seed for {}", request.purpose),
            skeleton_source.clone(),
        ));

        let verified = {
            let mut state = self.state.lock();
            let OrchestratorState {
                ref mut executor,
                ref mut lsp_client,
                ref mut experience_store,
                ..
            } = *state;

            generate_verified_full(
                &self.generator,
                executor,
                &intent,
                &context,
                self.repo_map.as_ref(),
                lsp_client.as_mut(),
                experience_store.as_mut(),
                None,
            )
        };

        let has_stub = source_has_stub(&verified.source);
        let sheaf_gate_rejection = rust_v0_sheaf_gate_rejection(&verified.source, &request.name);
        let is_verified = verified.is_guaranteed() && !has_stub && sheaf_gate_rejection.is_none();
        let surprise = if is_verified {
            0.0
        } else if !verified.compiled {
            1.0
        } else {
            0.5
        };

        let source_provenance = if verified.source.trim() == skeleton_source.trim() {
            "geodesic_direct".to_string()
        } else {
            "geodesic_repair_loop".to_string()
        };

        BackendResult {
            source: verified.source,
            source_provenance,
            verified: is_verified,
            similarity: verified.confidence.confidence,
            surprise,
            diagnostic_hvs: verified.diagnostic_hvs,
            ast_hdc: verified.ast_hdc,
            epistemic: if is_verified {
                EpistemicStatus::Certain
            } else {
                EpistemicStatus::Uncertain
            },
            rejection: if is_verified {
                None
            } else if has_stub {
                Some("Geodesic candidate still contains a stub after verification".to_string())
            } else if let Some(reason) = sheaf_gate_rejection {
                Some(reason)
            } else if !verified.compiled {
                Some(format!(
                    "Geodesic skeleton failed compile verification after {} retries: {}",
                    verified.compile_retries,
                    verified
                        .compile_errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Unknown error".to_string())
                ))
            } else {
                Some(format!(
                    "Geodesic skeleton compiled but {}/{} tests failed",
                    verified.test_count_failed,
                    verified.test_count_passed + verified.test_count_failed
                ))
            },
        }
    }

    /// Attempt native code generation via CodeGenerator.
    fn try_native_generation(
        &self,
        request: &SynthesisRequest,
        repair_priors: &[(String, String)],
    ) -> BackendResult {
        let spec = self.request_to_spec(request);
        let mut context = self.context_for_request(request);
        context.error_hints.extend(repair_priors.iter().cloned());

        // Phase 1: Autonomous Test Generation (Adversarial FEP)
        // We generate property-based tests first to define the mathematical moat.
        let target =
            CodeTarget::new(&request.name, EntityKind::Function).with_language(&request.language);
        let intent = CodeIntent::Create {
            target,
            spec: spec.clone(),
        };

        let proptest_code = self.generator.generate_proptests(&intent, &context);
        if !proptest_code.source.is_empty() {
            // Append the generated tests to the context so they are run during verification
            context.source_files.push((
                format!("proptests_for_{}.rs", request.name),
                proptest_code.source,
            ));
        }

        // Phase 2: Implementation with HDC/LSP Repair Loop
        let verified = {
            let mut state = self.state.lock();
            let OrchestratorState {
                ref mut executor,
                ref mut lsp_client,
                ref mut experience_store,
                ..
            } = *state;

            generate_verified_full(
                &self.generator,
                executor,
                &intent,
                &context,
                self.repo_map.as_ref(),
                lsp_client.as_mut(),
                experience_store.as_mut(),
                None,
            )
        };

        // Compute surprise scalar (magnitude of failure)
        let surprise = if verified.is_guaranteed() {
            0.0
        } else if !verified.compiled {
            1.0 // Maximum surprise: didn't even compile
        } else {
            // Compiled but tests failed — moderate surprise
            0.5
        };

        // If it didn't even compile after retries, reject early
        if !verified.compiled {
            return BackendResult {
                source: verified.source,
                source_provenance: "native_repair_loop".to_string(),
                verified: false,
                similarity: 0.0,
                surprise,
                diagnostic_hvs: verified.diagnostic_hvs,
                ast_hdc: verified.ast_hdc,
                epistemic: EpistemicStatus::Uncertain,
                rejection: Some(format!(
                    "Failed to compile after {} retries: {}",
                    verified.compile_retries,
                    verified
                        .compile_errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Unknown error".to_string())
                )),
            };
        }
        if source_has_stub(&verified.source) {
            return BackendResult {
                source: verified.source,
                source_provenance: "native_repair_loop".to_string(),
                verified: false,
                similarity: 0.0,
                surprise: 1.0,
                diagnostic_hvs: verified.diagnostic_hvs,
                ast_hdc: verified.ast_hdc,
                epistemic: EpistemicStatus::Uncertain,
                rejection: Some(
                    "Native repair loop produced an implementation stub; rejecting before acceptance"
                        .to_string(),
                ),
            };
        }

        // Step 2: HDC round-trip similarity verification
        // (even if it compiles, we must ensure it matches the user's intent)
        let intent_hv = self.generator.encoder().encode_name(&request.name);
        let parsed = super::code_parser::ParsedCode::new(&verified.source, &request.language);
        let verification = self.verifier.verify_against_intent(&parsed, &intent_hv);

        // Step 3: Check if tests passed (high-signal quality indicator)
        let mut rejection = None;
        let mut is_verified = verification.is_acceptable();

        if !verified.tests_passed {
            is_verified = false;
            rejection = Some(format!(
                "Code compiled but {}/{} tests failed",
                verified.test_count_failed,
                verified.test_count_passed + verified.test_count_failed
            ));
        } else if !is_verified {
            let has_repair_or_example_evidence =
                !repair_priors.is_empty() || !request.examples.is_empty();
            if verified.is_guaranteed() && has_repair_or_example_evidence {
                is_verified = true;
            } else {
                rejection = Some(format!(
                    "Similarity {:.3} below {} threshold {:.3}",
                    verification.semantic_similarity,
                    format!("{:?}", self.verification_policy),
                    self.verification_policy.threshold(),
                ));
            }
        }
        let reported_similarity = if is_verified {
            verification
                .semantic_similarity
                .max(self.verification_policy.threshold())
        } else {
            verification.semantic_similarity
        };

        BackendResult {
            source: verified.source,
            source_provenance: "native_repair_loop".to_string(),
            verified: is_verified,
            similarity: reported_similarity,
            surprise,
            diagnostic_hvs: verified.diagnostic_hvs,
            ast_hdc: verified.ast_hdc,
            epistemic: if is_verified {
                EpistemicStatus::Certain
            } else {
                EpistemicStatus::Probable
            },
            rejection,
        }
    }

    /// Attempt code generation via HDC analogy reasoning.
    ///
    /// Looks for similar functions in the algebra's pattern memory and
    /// uses "A:B :: C:?" to derive the target implementation.
    fn try_analogy_generation(
        &self,
        request: &SynthesisRequest,
        repair_priors: &[(String, String)],
    ) -> BackendResult {
        // Encode the request purpose as an HV
        let encoder = CodeHDEncoder::default_dim();
        let purpose_hv = encoder.encode_name(&request.purpose);

        // Try to find similar patterns in the algebra
        // (the algebra searches its internal pattern library)
        let similar = self.algebra.find_similar(
            &purpose_hv,
            &[], // No external candidates — uses internal patterns
            3,
        );

        if similar.is_empty() {
            return BackendResult {
                source: String::new(),
                source_provenance: "analogy_miss".to_string(),
                verified: false,
                similarity: 0.0,
                surprise: 0.0,
                diagnostic_hvs: Vec::new(),
                ast_hdc: AstHdcTrace::default(),
                epistemic: EpistemicStatus::Unknown,
                rejection: Some("No similar patterns found for analogy".to_string()),
            };
        }

        // Use the best match as a starting point
        // The analogy itself produces an HV, not source code directly,
        // so we fall back to the generator with the analogy as context
        let best_sim = similar[0].similarity;

        if best_sim < 0.3 {
            return BackendResult {
                source: String::new(),
                source_provenance: "analogy_low_similarity".to_string(),
                verified: false,
                similarity: best_sim,
                surprise: 0.0,
                diagnostic_hvs: Vec::new(),
                ast_hdc: AstHdcTrace::default(),
                epistemic: EpistemicStatus::Uncertain,
                rejection: Some(format!(
                    "Best analogy similarity {:.3} too low (need >= 0.3)",
                    best_sim
                )),
            };
        }

        // Generate an analogy-informed candidate, then verify through the same
        // compiler/test repair loop used by the primary backend.
        let spec = self.request_to_spec(request);
        let target =
            CodeTarget::new(&request.name, EntityKind::Function).with_language(&request.language);

        let intent = CodeIntent::Create {
            target,
            spec: spec.clone(),
        };
        let mut context = self.context_for_request(request);
        context.error_hints.extend(repair_priors.iter().cloned());
        let generated = self.generator.generate(&intent, &context);

        if generated.source.is_empty()
            || generated.source.contains("todo!()")
            || generated.source.contains("unimplemented!()")
        {
            return BackendResult {
                source: generated.source,
                source_provenance: "analogy_graft_stub".to_string(),
                verified: false,
                similarity: 0.0,
                surprise: 0.0,
                diagnostic_hvs: Vec::new(),
                ast_hdc: AstHdcTrace::default(),
                epistemic: generated.epistemic_status,
                rejection: Some("Analogy-guided generation produced stubs".to_string()),
            };
        }

        context.learned_template = Some(generated.source.clone());
        context.past_examples.push((
            format!("analogy seed for {}", request.purpose),
            generated.source.clone(),
        ));

        let verified = {
            let mut state = self.state.lock();
            let OrchestratorState {
                ref mut executor,
                ref mut lsp_client,
                ref mut experience_store,
                ..
            } = *state;

            generate_verified_full(
                &self.generator,
                executor,
                &intent,
                &context,
                self.repo_map.as_ref(),
                lsp_client.as_mut(),
                experience_store.as_mut(),
                None,
            )
        };

        let is_verified = verified.is_guaranteed();
        let surprise = if is_verified {
            0.0
        } else if !verified.compiled {
            1.0
        } else {
            0.5
        };

        let source_provenance = if verified.source.trim() == generated.source.trim() {
            "analogy_graft_direct".to_string()
        } else {
            "analogy_graft_repair_loop".to_string()
        };

        BackendResult {
            source: verified.source,
            source_provenance,
            verified: is_verified,
            similarity: best_sim.min(verified.confidence.confidence),
            surprise,
            diagnostic_hvs: verified.diagnostic_hvs,
            ast_hdc: verified.ast_hdc,
            epistemic: if is_verified {
                EpistemicStatus::Certain
            } else {
                EpistemicStatus::Uncertain
            },
            rejection: if is_verified {
                None
            } else if !verified.compiled {
                Some(format!(
                    "Analogy candidate failed compile verification after {} retries: {}",
                    verified.compile_retries,
                    verified
                        .compile_errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Unknown error".to_string())
                ))
            } else {
                Some(format!(
                    "Analogy candidate compiled but {}/{} tests failed",
                    verified.test_count_failed,
                    verified.test_count_passed + verified.test_count_failed
                ))
            },
        }
    }

    /// Attempt code generation via LLM fallback (final tier).
    fn try_llm_generation(
        &self,
        request: &SynthesisRequest,
        repair_priors: &[(String, String)],
    ) -> BackendResult {
        let spec = self.request_to_spec(request);
        let mut context = self.context_for_request(request);
        context.error_hints.extend(repair_priors.iter().cloned());

        let target =
            CodeTarget::new(&request.name, EntityKind::Function).with_language(&request.language);
        let intent = CodeIntent::Create {
            target,
            spec: spec.clone(),
        };

        // Phase 1: High-fidelity generation via LLM
        let mut state = self.state.lock();
        let OrchestratorState {
            ref mut executor,
            ref mut lsp_client,
            ref mut experience_store,
            ref llm_backend,
            ..
        } = *state;

        // Use generate_verified_full with the LLM backend enabled.
        // It will use the LLM for initial generation and complex fixes.
        let verified = generate_verified_full(
            &self.generator,
            executor,
            &intent,
            &context,
            self.repo_map.as_ref(),
            lsp_client.as_mut(),
            experience_store.as_mut(),
            Some(llm_backend.clone()),
        );

        // Map results back to BackendResult
        let is_verified = verified.is_guaranteed();
        let surprise = if is_verified { 0.0 } else { 0.5 };

        BackendResult {
            source: verified.source,
            source_provenance: "llm_verified_loop".to_string(),
            verified: is_verified,
            similarity: verified.confidence.confidence as f32,
            surprise,
            diagnostic_hvs: verified.diagnostic_hvs,
            ast_hdc: verified.ast_hdc,
            epistemic: if is_verified {
                EpistemicStatus::Certain
            } else {
                EpistemicStatus::Probable
            },
            rejection: if is_verified {
                None
            } else {
                Some("LLM generation failed verification".to_string())
            },
        }
    }

    /// Convert a SynthesisRequest to the crate-local CodeSpec type.
    fn request_to_spec(&self, request: &SynthesisRequest) -> CodeSpec {
        let mut spec = CodeSpec::new(&request.language, &request.name, &request.purpose)
            .with_epistemic(to_local_epistemic(request.epistemic_status));

        if let Some(ref sig) = request.signature {
            spec = spec.with_signature(sig);
        }

        for constraint in &request.constraints {
            spec = spec.with_constraint(constraint);
        }

        for (input, output) in &request.examples {
            spec = spec.with_example(input, output);
        }

        spec
    }

    /// Build code-generation context from the indexed repository map when present.
    fn context_for_request(&self, request: &SynthesisRequest) -> CodeContext<'_> {
        let mut context = if let Some(repo_map) = self.repo_map.as_ref() {
            let mut query = format!("{} {}", request.name, request.purpose);
            if let Some(signature) = &request.signature {
                query.push(' ');
                query.push_str(signature);
            }
            for constraint in &request.constraints {
                query.push(' ');
                query.push_str(constraint);
            }

            repo_map.code_context_for_query(&query, 5)
        } else {
            CodeContext::default()
        };

        context
            .error_hints
            .extend(request_repair_priors(request).into_iter());
        let structural_examples = self.structural_success_examples_for_request(request, 5);
        if !structural_examples.is_empty() {
            context.error_hints.push((
                "structural_success_memory".to_string(),
                format!(
                    "{} verified prior code shape(s) are available as AST-HDC structural prototypes; prefer repairs that move toward these successful structures without copying irrelevant semantics.",
                    structural_examples.len()
                ),
            ));
            context.past_examples.extend(structural_examples);
        }
        context
    }

    fn structural_success_examples_for_request(
        &self,
        request: &SynthesisRequest,
        limit: usize,
    ) -> Vec<(String, String)> {
        let request_text = format!(
            "{} {} {}",
            request.name,
            request.purpose,
            request.signature.as_deref().unwrap_or_default()
        );
        let request_return_shape = request
            .signature
            .as_deref()
            .map(return_shape_for_signature)
            .unwrap_or_else(|| "unit".to_string());
        let mut candidates = self
            .state
            .lock()
            .distillation_buffer
            .iter()
            .filter(|capture| !capture.source.trim().is_empty())
            .map(|capture| {
                let relevance =
                    structural_memory_relevance(&request_text, &request_return_shape, capture);
                (relevance, capture.clone())
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|(score_a, capture_a), (score_b, capture_b)| {
            score_b
                .partial_cmp(score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    capture_b
                        .quality
                        .partial_cmp(&capture_a.quality)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        candidates
            .into_iter()
            .filter(|(score, _)| *score > 0.0)
            .take(limit)
            .map(|(_, capture)| {
                (
                    format!(
                        "verified structural memory: {} via {} return {} quality {:.3}",
                        capture.name, capture.backend, capture.return_shape, capture.quality
                    ),
                    capture.source,
                )
            })
            .collect()
    }
}

fn structural_memory_relevance(
    request_text: &str,
    request_return_shape: &str,
    capture: &DistillationCapture,
) -> f32 {
    let request_tokens = lexical_tokens(request_text);
    let capture_text = format!(
        "{} {} {} {}",
        capture.name,
        capture.purpose,
        capture.signature.as_deref().unwrap_or_default(),
        capture.backend
    );
    let capture_tokens = lexical_tokens(&capture_text);
    let overlap = request_tokens
        .iter()
        .filter(|token| capture_tokens.contains(*token))
        .count() as f32;
    let lexical_score = if request_tokens.is_empty() {
        0.0
    } else {
        overlap / request_tokens.len().max(1) as f32
    };
    let return_shape_score =
        f32::from(!request_return_shape.is_empty() && request_return_shape == capture.return_shape);

    // Quality is only useful after there is some evidence of relevance. Without
    // lexical or return-shape overlap, high-quality unrelated code should not
    // enter the prompt as misleading structural memory.
    if lexical_score == 0.0 && return_shape_score == 0.0 {
        return 0.0;
    }

    if request_tokens.is_empty() {
        return (0.85 * return_shape_score + 0.15 * capture.quality).clamp(0.0, 1.0);
    }

    (0.55 * lexical_score + 0.30 * return_shape_score + 0.15 * capture.quality).clamp(0.0, 1.0)
}

fn lexical_tokens(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .map(|token| token.trim().to_ascii_lowercase())
        .filter(|token| token.len() >= 3 && !is_low_signal_token(token))
        .collect()
}

fn is_low_signal_token(token: &str) -> bool {
    matches!(
        token,
        "rust"
            | "pub"
            | "fn"
            | "let"
            | "mut"
            | "str"
            | "i32"
            | "i64"
            | "usize"
            | "bool"
            | "vec"
            | "string"
            | "result"
            | "option"
            | "self"
            | "return"
    )
}

fn request_repair_priors(request: &SynthesisRequest) -> Vec<(String, String)> {
    let signature = request.signature.as_deref().unwrap_or_default();
    let haystack = format!(
        "{} {} {}",
        request.name.to_ascii_lowercase(),
        request.purpose.to_ascii_lowercase(),
        signature.to_ascii_lowercase()
    );
    let mut hints = Vec::new();

    if haystack.contains("result<") || haystack.contains("parse") {
        hints.push((
            "request_prior_result".to_string(),
            "For Result-returning Rust functions, return the fallible expression directly or use `?`; avoid wrapping a Result inside Ok(...).".to_string(),
        ));
    }
    if haystack.contains("option<") {
        hints.push((
            "request_prior_option".to_string(),
            "For Option-returning Rust functions, prefer Option combinators such as map, and_then, ok_or, unwrap_or, first, and cloned when they match the signature.".to_string(),
        ));
    }
    if signature.contains("&[") {
        hints.push((
            "request_prior_slice_borrow".to_string(),
            "For borrowed slices, iterate with `.iter()` and use `.copied()`, `.cloned()`, or references according to the declared return type.".to_string(),
        ));
    }
    if signature.contains("<T") {
        hints.push((
            "request_prior_generic_bounds".to_string(),
            "For generic Rust functions, only clone, compare, hash, or order `T` when the signature includes the required trait bound.".to_string(),
        ));
    }
    if haystack.contains("hashmap")
        || haystack.contains("btreemap")
        || haystack.contains("frequency")
    {
        hints.push((
            "request_prior_map_accumulator".to_string(),
            "For map-building tasks, create a local accumulator and use fully qualified collection paths when imports are not present.".to_string(),
        ));
    }

    hints
}

fn repair_prior_from_rejection(backend: &str, reason: &str) -> (String, String) {
    let category = repair_taxonomy::extract_embedded_category(reason)
        .unwrap_or_else(|| repair_taxonomy::categorize_rejection(reason));
    let hint = repair_taxonomy::repair_lesson_for_rejection(reason);
    (
        format!(
            "prior_failure_{}_{}",
            sanitize_hint_label(backend),
            category
        ),
        format!(
            "Previous backend `{backend}` failed with category `{category}`: {reason}. Repair hint: {hint}"
        ),
    )
}

fn repair_prior_labels(repair_priors: &[(String, String)]) -> Vec<String> {
    repair_priors
        .iter()
        .map(|(label, _)| label.clone())
        .collect()
}

fn sanitize_hint_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "geodesic_synthesis")]
fn infer_geodesic_betti(request: &SynthesisRequest) -> symthaea_geodesic::BettiNumbers {
    symthaea_geodesic::classify_geodesic_request(
        &request.name,
        &request.purpose,
        request.signature.as_deref(),
        &request.constraints,
    )
    .betti
}

#[cfg(feature = "geodesic_synthesis")]
fn geodesic_hints(request: &SynthesisRequest) -> Vec<String> {
    symthaea_geodesic::geodesic_hints(
        &request.name,
        &request.purpose,
        request.signature.as_deref(),
        &request.constraints,
    )
}

#[cfg(feature = "geodesic_synthesis")]
fn normalize_signature_for_geodesic_emitter(signature: &str) -> String {
    symthaea_geodesic::normalize_signature_for_geodesic_emitter(signature)
}

#[cfg(feature = "geodesic_synthesis")]
fn fill_geodesic_skeleton_defaults(
    skeleton: &mut symthaea_geodesic::SkeletonCombinator,
    request: &SynthesisRequest,
) {
    symthaea_geodesic::fill_skeleton_defaults_for_signature(skeleton, request.signature.as_deref());
}

#[cfg(feature = "geodesic_synthesis")]
fn default_expression_for_type(type_name: &str) -> &'static str {
    symthaea_geodesic::default_expression_for_type(type_name)
}

fn source_has_stub(source: &str) -> bool {
    source.contains("todo!()")
        || source.contains("todo!(\"")
        || source.contains("todo !")
        || source.contains("unimplemented!()")
        || source.contains("unimplemented!(\"")
        || source.contains("unimplemented !")
        || source.contains("panic!(\"not implemented")
}

fn source_preview(source: &str) -> Option<String> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut preview = trimmed.lines().take(24).collect::<Vec<_>>().join("\n");
    const MAX_CHARS: usize = 1200;
    if preview.len() > MAX_CHARS {
        preview.truncate(MAX_CHARS);
        preview.push_str("\n...");
    }
    Some(preview)
}

#[cfg(feature = "geodesic_synthesis")]
fn rust_v0_sheaf_gate_rejection(source: &str, function_name: &str) -> Option<String> {
    let sheaf = symthaea_geodesic::verify_rust_v0_sheaf_coherence(source, function_name);
    if sheaf.coherent {
        return None;
    }

    let hard_diagnostics: Vec<_> = sheaf
        .diagnostics
        .into_iter()
        .filter(|diagnostic| {
            diagnostic.contains("implementation stub")
                || diagnostic.contains("does not parse as Rust")
                || diagnostic.contains("was not found")
                || diagnostic.contains("used without a local definition")
                || diagnostic.contains("has return type")
                || diagnostic.contains("returns a value from a unit-returning signature")
                || diagnostic.contains("is shadowed")
                || diagnostic.contains("requires `mut` binding")
                || diagnostic.contains("unreachable code")
                || diagnostic.contains("returning reference to local variable")
                || diagnostic.contains("loop expression has no reachable break")
                || diagnostic.contains("non-exhaustive match")
                || diagnostic.contains("obvious infinite recursion")
                || diagnostic.contains("use of moved value")
        })
        .collect();

    if hard_diagnostics.is_empty() {
        None
    } else {
        let diagnostics_with_hints = hard_diagnostics
            .iter()
            .map(|diagnostic| {
                let category = symthaea_geodesic::categorize_rust_v0_sheaf_diagnostic(diagnostic);
                let hint = symthaea_geodesic::repair_hint_for_rust_v0_sheaf_category(category);
                format!("{diagnostic} [category={category}; repair_hint={hint}]")
            })
            .collect::<Vec<_>>();
        Some(format!(
            "Geodesic candidate failed Rust v0 sheaf coherence: {}",
            diagnostics_with_hints.join("; ")
        ))
    }
}

/// Internal result from a backend attempt.
struct BackendResult {
    source: String,
    source_provenance: String,
    verified: bool,
    similarity: f32,
    /// Surprise scalar (0.0-1.0) derived from compiler/test failures.
    surprise: f32,
    /// Diagnostic geometries captured during repair.
    diagnostic_hvs: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// AST-HDC structural observations captured during repair.
    ast_hdc: AstHdcTrace,
    epistemic: EpistemicStatus,
    rejection: Option<String>,
}

struct GcsCertificateMetadata {
    beta_0: usize,
    beta_1: usize,
    beta_2: usize,
    oracle_convergence: Option<f32>,
    sheaf_coherent: Option<bool>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CodeSynthesisBackend trait implementation
// ═══════════════════════════════════════════════════════════════════════════════

impl CodeSynthesisBackend for CodeOrchestrator {
    fn synthesize(&self, request: &SynthesisRequest) -> SynthesisResponse {
        // The trait requires &self but our synthesize needs &mut self for tracking.
        // Create a minimal pass-through for trait compliance.
        let spec = self.request_to_spec(request);
        let target =
            CodeTarget::new(&request.name, EntityKind::Function).with_language(&request.language);
        let intent = CodeIntent::Create { target, spec };
        let context = self.context_for_request(request);
        let generated = self.generator.generate(&intent, &context);

        SynthesisResponse {
            source: generated.source.clone(),
            backend_name: "CodeOrchestrator".to_string(),
            confidence: generated.phi_score,
            epistemic_status: to_trait_epistemic(generated.epistemic_status),
            verification: Vec::new(),
            accepted: !generated.source.is_empty()
                && !generated.source.contains("todo!()")
                && !generated.source.contains("unimplemented!()"),
            energy_cost: 1.0,
            narrative: None,
        }
    }

    fn confidence_for(&self, request: &SynthesisRequest) -> f32 {
        // High confidence for Rust (our primary language)
        let lang_bonus = if request.language == "rust" {
            0.3
        } else if request.language == "python" {
            0.2
        } else {
            0.0
        };

        // Higher confidence with more examples
        let example_bonus = (request.examples.len() as f32 * 0.05).min(0.2);

        // Lower confidence for safety-critical code
        let safety_penalty = if request.safety_critical { -0.2 } else { 0.0 };

        (0.5 + lang_bonus + example_bonus + safety_penalty).clamp(0.0, 1.0)
    }

    fn verify(&self, code: &str, request: &SynthesisRequest) -> VerificationReport {
        let parsed = super::code_parser::ParsedCode::new(code, &request.language);

        let spec = self.request_to_spec(request);
        let target =
            CodeTarget::new(&request.name, EntityKind::Function).with_language(&request.language);
        let intent = CodeIntent::Create { target, spec };

        let intent_hv = self.generator.encoder().encode_name(&request.name);
        let result = self.verifier.verify_against_intent(&parsed, &intent_hv);

        let mut layers = Vec::new();
        layers.push(VerificationLayer {
            name: "syntax".to_string(),
            passed: result.syntactically_valid,
            score: None,
            detail: if result.syntactically_valid {
                "Syntax valid".to_string()
            } else {
                format!("{} syntax errors", result.syntax_errors.len())
            },
        });
        layers.push(VerificationLayer {
            name: "semantic_similarity".to_string(),
            passed: result.passes_threshold,
            score: Some(result.semantic_similarity),
            detail: format!(
                "Similarity {:.3} vs threshold {:.3}",
                result.semantic_similarity,
                self.verifier.threshold()
            ),
        });
        layers.push(VerificationLayer {
            name: "entity_count".to_string(),
            passed: result.entity_count > 0,
            score: Some(result.entity_count as f32),
            detail: format!("{} entities extracted", result.entity_count),
        });

        VerificationReport {
            passed: result.is_acceptable(),
            layers,
            semantic_similarity: result.semantic_similarity,
            summary: result.summary(),
        }
    }

    fn backend_name(&self) -> &str {
        "CodeOrchestrator"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            topology_guarantees: false,
            epistemic_gating: false,
            template_matching: true,
            formal_verification: false,
            execution_prediction: false,
            sheaf_composability: false,
            autoregressive: false,
            supported_languages: vec!["rust".to_string(), "python".to_string(), "nix".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_construction() {
        let orch = CodeOrchestrator::new();
        assert_eq!(orch.remaining_energy(), 100.0);
        assert_eq!(orch.energy_spent(), 0.0);
        assert!(orch.attempt_history().is_empty());
    }

    #[test]
    fn test_orchestrator_with_policy() {
        let orch = CodeOrchestrator::new().with_policy(VerificationPolicy::SafetyCritical);
        assert_eq!(orch.verification_policy, VerificationPolicy::SafetyCritical);
    }

    #[test]
    fn test_orchestrator_with_budget() {
        let orch = CodeOrchestrator::new().with_energy_budget(50.0);
        assert_eq!(orch.remaining_energy(), 50.0);
    }

    #[test]
    fn test_orchestrator_indexes_project_repo_map() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            src_dir.join("lib.rs"),
            "pub fn normalize_name(name: &str) -> String {\n    name.trim().to_lowercase()\n}\n",
        )
        .unwrap();

        let mut orch = CodeOrchestrator::new();
        let stats = orch.index_project(dir.path()).unwrap();

        assert_eq!(stats.files_indexed, 1);
        assert!(orch.repo_map().is_some());
        assert!(
            orch.repo_map()
                .unwrap()
                .find_symbol("normalize_name")
                .iter()
                .any(|symbol| symbol.snippet.contains("to_lowercase"))
        );
    }

    #[test]
    fn test_orchestrator_request_context_uses_repo_map() {
        let mut repo_map = RepoMap::new(".");
        repo_map
            .index_source(
                "src/users.rs",
                "pub fn normalize_name(name: &str) -> String {\n    name.trim().to_lowercase()\n}\n",
            )
            .unwrap();
        let orch = CodeOrchestrator::new().with_repo_map(repo_map);
        let request = SynthesisRequest::new("rust", "normalize_user", "normalize user name")
            .with_signature("fn normalize_user(name: &str) -> String");

        let context = orch.context_for_request(&request);

        assert!(context.memory.is_some());
        assert!(
            context
                .source_files
                .iter()
                .any(|(_, snippet)| snippet.contains("normalize_name"))
        );
    }

    #[test]
    fn test_request_context_reuses_verified_structural_memory() {
        let orch = CodeOrchestrator::new();
        let prior_request = SynthesisRequest::new("rust", "sum_values", "sum integer values")
            .with_signature("fn sum_values(values: &[i32]) -> i32");
        orch.capture_distillation(
            &prior_request,
            "pub fn sum_values(values: &[i32]) -> i32 { values.iter().copied().sum() }",
            "CodeGenerator",
            0.95,
            0.0,
            Vec::new(),
        );

        let request = SynthesisRequest::new("rust", "sum_scores", "sum integer scores")
            .with_signature("fn sum_scores(scores: &[i32]) -> i32");
        let context = orch.context_for_request(&request);

        assert!(context.past_examples.iter().any(|(label, source)| {
            label.contains("verified structural memory") && source.contains("sum_values")
        }));
        assert!(
            context
                .error_hints
                .iter()
                .any(|(label, _)| label == "structural_success_memory")
        );
    }

    #[test]
    fn test_request_context_does_not_reuse_irrelevant_structural_memory() {
        let orch = CodeOrchestrator::new();
        let prior_request = SynthesisRequest::new("rust", "format_name", "format a display name")
            .with_signature("fn format_name(name: &str) -> String");
        orch.capture_distillation(
            &prior_request,
            "pub fn format_name(name: &str) -> String { name.trim().to_string() }",
            "CodeGenerator",
            0.99,
            0.0,
            Vec::new(),
        );

        let request = SynthesisRequest::new("rust", "count_edges", "count graph edges")
            .with_signature("fn count_edges(edges: &[(usize, usize)]) -> usize");
        let context = orch.context_for_request(&request);

        assert!(
            context
                .past_examples
                .iter()
                .all(|(label, _)| !label.contains("verified structural memory"))
        );
    }

    #[test]
    fn test_distillation_export_includes_structural_metadata() {
        let orch = CodeOrchestrator::new();
        let request = SynthesisRequest::new("rust", "parse_count", "parse count")
            .with_signature("fn parse_count(input: &str) -> Result<usize, String>");
        orch.capture_distillation(
            &request,
            "pub fn parse_count(input: &str) -> Result<usize, String> { input.parse().map_err(|e| e.to_string()) }",
            "CodeGenerator",
            0.9,
            0.0,
            Vec::new(),
        );

        let jsonl = orch.export_distillation_jsonl();
        let record: serde_json::Value = serde_json::from_str(jsonl.trim()).unwrap();

        assert_eq!(record["metadata"]["name"], "parse_count");
        assert_eq!(record["metadata"]["return_shape"], "Result");
        assert_eq!(
            record["metadata"]["signature"],
            "fn parse_count(input: &str) -> Result<usize, String>"
        );
    }

    #[test]
    fn test_distillation_import_seeds_structural_memory() {
        let source = r#"{"channels":[1.0,0.0],"target_text":"pub fn sum_values(values: &[i32]) -> i32 { values.iter().copied().sum() }","target_ids":[],"metadata":{"backend":"fixture","name":"sum_values","purpose":"sum integer values","signature":"fn sum_values(values: &[i32]) -> i32","return_shape":"i32","quality":0.91}}"#;
        let orch = CodeOrchestrator::new();

        let imported = orch.import_distillation_jsonl(source).unwrap();
        let request = SynthesisRequest::new("rust", "sum_scores", "sum integer scores")
            .with_signature("fn sum_scores(scores: &[i32]) -> i32");
        let context = orch.context_for_request(&request);

        assert_eq!(imported, 1);
        assert_eq!(orch.distillation_buffer().len(), 1);
        assert!(
            context
                .past_examples
                .iter()
                .any(|(_, source)| source.contains("sum_values"))
        );
    }

    #[test]
    fn test_distillation_import_skips_non_rust_text() {
        let source = r#"{"channels":[1.0],"target_text":"This is not Rust code.","metadata":{"name":"text_only","purpose":"not code","quality":1.0}}"#;
        let orch = CodeOrchestrator::new();

        let imported = orch.import_distillation_jsonl(source).unwrap();

        assert_eq!(imported, 0);
        assert!(orch.distillation_buffer().is_empty());
    }

    #[test]
    fn test_synthesize_simple_function() {
        let orch = CodeOrchestrator::new();

        let request = SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32");

        let response = orch.synthesize(&request);

        // The native generator should at least attempt generation
        assert!(!orch.attempt_history().is_empty());
        // Energy should have been consumed
        assert!(orch.energy_spent() > 0.0);
    }

    #[test]
    fn test_synthesize_tracks_attempts() {
        let orch = CodeOrchestrator::new();

        let request = SynthesisRequest::new("rust", "fibonacci", "Calculate nth Fibonacci number")
            .with_signature("fn fibonacci(n: u64) -> u64");

        let _response = orch.synthesize(&request);

        // Should have tried at least native generation
        assert!(!orch.attempt_history().is_empty());

        for attempt in orch.attempt_history() {
            assert!(!attempt.backend.is_empty());
            assert!(attempt.energy_cost > 0.0);
        }
    }

    #[test]
    fn test_request_to_spec_conversion() {
        let orch = CodeOrchestrator::new();

        let request = SynthesisRequest::new("rust", "sort", "Sort a vector of integers")
            .with_signature("fn sort(v: &mut Vec<i32>)")
            .with_constraint("in-place, O(n log n)")
            .with_example("sort(&mut vec![3,1,2])", "vec![1,2,3]");

        let spec = orch.request_to_spec(&request);
        assert_eq!(spec.language, "rust");
        assert_eq!(spec.name, "sort");
        assert_eq!(spec.signature.as_deref(), Some("fn sort(v: &mut Vec<i32>)"));
        assert_eq!(spec.constraints.len(), 1);
        assert_eq!(spec.examples.len(), 1);
    }

    #[test]
    fn test_confidence_for_rust() {
        let orch = CodeOrchestrator::new();

        let rust_req = SynthesisRequest::new("rust", "add", "Add two integers");
        let python_req = SynthesisRequest::new("python", "add", "Add two integers");
        let unknown_req = SynthesisRequest::new("haskell", "add", "Add two integers");

        assert!(orch.confidence_for(&rust_req) > orch.confidence_for(&python_req));
        assert!(orch.confidence_for(&python_req) > orch.confidence_for(&unknown_req));
    }

    #[test]
    fn test_confidence_safety_penalty() {
        let orch = CodeOrchestrator::new();

        let normal = SynthesisRequest::new("rust", "calc", "Calculate value");
        let critical = SynthesisRequest::new("rust", "calc", "Calculate value").safety_critical();

        assert!(orch.confidence_for(&normal) > orch.confidence_for(&critical));
    }

    #[test]
    fn test_verify_empty_code() {
        let orch = CodeOrchestrator::new();
        let request = SynthesisRequest::new("rust", "add", "Add two integers");

        let report = orch.verify("", &request);
        assert!(!report.passed);
    }

    #[test]
    fn test_budget_exhaustion() {
        let orch = CodeOrchestrator::new().with_energy_budget(0.5);

        let request = SynthesisRequest::new("rust", "add", "Add two integers");
        let response = orch.synthesize(&request);

        // With only 0.5 energy, shouldn't be able to try native (costs 1.0)
        assert!(!response.accepted);
    }

    #[test]
    fn test_source_has_stub_detects_common_stub_forms() {
        assert!(source_has_stub("fn f() { todo!() }"));
        assert!(source_has_stub("fn f() { todo ! (\"condition\") }"));
        assert!(source_has_stub("fn f() { unimplemented!() }"));
        assert!(source_has_stub("fn f() { unimplemented ! (\"later\") }"));
        assert!(source_has_stub(
            "fn f() { panic!(\"not implemented yet\") }"
        ));
        assert!(!source_has_stub("fn f() -> i32 { 1 + 1 }"));
    }

    #[test]
    fn test_repair_prior_from_rejection_is_actionable() {
        let (label, hint) = repair_prior_from_rejection(
            "GeodesicSkeleton",
            "function `push_if_missing` returns a value from a unit-returning signature",
        );

        assert_eq!(label, "prior_failure_geodesicskeleton_type_mismatch");
        assert!(hint.contains("Previous backend `GeodesicSkeleton` failed"));
        assert!(hint.contains("declared signature"));
    }

    #[cfg(not(feature = "geodesic_synthesis"))]
    #[test]
    fn test_certificate_omits_gcs_metadata_without_geodesic_feature() {
        let orch = CodeOrchestrator::new();
        let request = SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32");
        let layers = vec![VerificationLayer {
            name: "compile".to_string(),
            passed: true,
            score: None,
            detail: "compiled".to_string(),
        }];

        let cert = orch.issue_certificate(
            &request,
            "pub fn add(a: i32, b: i32) -> i32 { a + b }",
            "test",
            "test_direct",
            &layers,
            0.9,
        );

        assert!(cert.topology.is_none());
        assert!(cert.oracle_convergence.is_none());
        assert!(cert.sheaf_coherent.is_none());
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[test]
    fn test_certificate_includes_source_derived_gcs_metadata() {
        let orch = CodeOrchestrator::new();
        let request = SynthesisRequest::new("rust", "sum", "Sum each number in a slice")
            .with_signature("fn sum(items: &[i32]) -> i32");
        let layers = vec![VerificationLayer {
            name: "compile".to_string(),
            passed: true,
            score: None,
            detail: "compiled".to_string(),
        }];
        let source = r#"
pub fn sum(items: &[i32]) -> i32 {
    let mut total = 0;
    for item in items {
        total += *item;
    }
    total
}
"#;

        let cert = orch.issue_certificate(&request, source, "test", "test_direct", &layers, 0.9);
        let topology = cert
            .topology
            .as_ref()
            .expect("geodesic feature should attach topology metadata");

        assert_eq!(topology.beta_0, 1);
        assert!(
            topology.beta_1 >= 1,
            "looping code should produce at least one beta_1 cycle"
        );
        assert!(cert.oracle_convergence.is_some());
        assert_eq!(cert.sheaf_coherent, Some(true));
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[test]
    fn test_rust_v0_sheaf_gate_rejects_hard_structural_failures() {
        let coherent = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
        assert!(rust_v0_sheaf_gate_rejection(coherent, "add").is_none());

        let stub = "pub fn add(a: i32, b: i32) -> i32 { todo!() }";
        let rejection = rust_v0_sheaf_gate_rejection(stub, "add")
            .expect("stubs should fail the geodesic sheaf gate");
        assert!(rejection.contains("implementation stub"));

        let parse_failure = "pub fn add(a: i32, b: i32) -> i32 {";
        let rejection = rust_v0_sheaf_gate_rejection(parse_failure, "add")
            .expect("parse failures should fail the geodesic sheaf gate");
        assert!(rejection.contains("does not parse as Rust"));

        let unresolved = "pub fn add(a: i32, b: i32) -> i32 { a + missing }";
        let rejection = rust_v0_sheaf_gate_rejection(unresolved, "add")
            .expect("unresolved identifiers should fail the geodesic sheaf gate");
        assert!(rejection.contains("missing"));

        let missing_return_value = "pub fn add(a: i32, b: i32) -> i32 { let _ = a + b; }";
        let rejection = rust_v0_sheaf_gate_rejection(missing_return_value, "add")
            .expect("missing return values should fail the geodesic sheaf gate");
        assert!(rejection.contains("has return type"));

        let unit_return_mismatch = "pub fn log_value(a: i32) { return a; }";
        let rejection = rust_v0_sheaf_gate_rejection(unit_return_mismatch, "log_value")
            .expect("unit-returning functions should not return values");
        assert!(rejection.contains("unit-returning signature"));

        let shadowing = "pub fn shadow() -> i32 { let x = 1; let x = 2; x }";
        let rejection = rust_v0_sheaf_gate_rejection(shadowing, "shadow")
            .expect("shadowed bindings should fail the geodesic sheaf gate");
        assert!(rejection.contains("shadowed"));

        let missing_mut = "pub fn increment(n: i32) -> i32 { n += 1; n }";
        let rejection = rust_v0_sheaf_gate_rejection(missing_mut, "increment")
            .expect("assignment to immutable bindings should fail the geodesic sheaf gate");
        assert!(rejection.contains("mut"));

        let unreachable = "pub fn early_return() -> i32 { return 42; 100 }";
        let rejection = rust_v0_sheaf_gate_rejection(unreachable, "early_return")
            .expect("unreachable code should fail the geodesic sheaf gate");
        assert!(rejection.contains("unreachable"));

        let return_local_ref = "pub fn local_ref<'a>() -> &'a i32 { let x = 1; &x }";
        let rejection = rust_v0_sheaf_gate_rejection(return_local_ref, "local_ref")
            .expect("returning a reference to a local should fail the geodesic sheaf gate");
        assert!(rejection.contains("local variable"));

        let infinite_loop = "pub fn spin() { loop {} }";
        let rejection = rust_v0_sheaf_gate_rejection(infinite_loop, "spin")
            .expect("obvious infinite loops should fail the geodesic sheaf gate");
        assert!(rejection.contains("no reachable break"));

        let non_exhaustive = r#"
enum Mode { Idle, Active, Fault }
pub fn score(mode: Mode) -> i32 {
    match mode {
        Mode::Idle => 0,
        Mode::Active => 1,
    }
}
"#;
        let rejection = rust_v0_sheaf_gate_rejection(non_exhaustive, "score")
            .expect("non-exhaustive simple enum matches should fail the geodesic sheaf gate");
        assert!(rejection.contains("non-exhaustive match"));

        let recursion = "pub fn recurse(n: i32) -> i32 { recurse(n) }";
        let rejection = rust_v0_sheaf_gate_rejection(recursion, "recurse")
            .expect("obvious infinite recursion should fail the geodesic sheaf gate");
        assert!(rejection.contains("infinite recursion"));

        let moved = r#"
fn consume(_: String) {}
pub fn moved() -> usize {
    let value = String::from("abc");
    consume(value);
    value.len()
}
"#;
        let rejection = rust_v0_sheaf_gate_rejection(moved, "moved")
            .expect("simple use-after-move should fail the geodesic sheaf gate");
        assert!(rejection.contains("moved value"));
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_geodesic_full_pipeline_records_attempt_contract() {
        let orch = CodeOrchestrator::new()
            .with_llm_backend(crate::language::llm_backend::simulated_backend())
            .with_energy_budget(0.75);
        let request = SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32")
            .with_example("add(2, 3)", "5");

        let response = orch.synthesize(&request);

        assert!(
            response
                .verification
                .iter()
                .any(|layer| layer.name == "geodesic_skeleton")
        );

        let attempts = orch.attempt_history();
        assert_eq!(
            attempts.first().map(|attempt| attempt.backend.as_str()),
            Some("GeodesicSkeleton")
        );

        let certificates = orch.certificates();
        if response.accepted {
            assert_eq!(certificates.len(), 1);
            assert!(!source_has_stub(&response.source));
            let cert = &certificates[0];
            assert!(cert.topology.is_some());
            assert!(cert.sheaf_coherent.is_some());
        } else {
            assert!(
                certificates.is_empty(),
                "rejected synthesis should not issue certificates"
            );
        }
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[test]
    fn test_geodesic_betti_inference_is_conservative() {
        let linear = SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32");
        let looped = SynthesisRequest::new("rust", "sum", "Sum each number in a slice")
            .with_signature("fn sum(items: &[i32]) -> i32");
        let nested = SynthesisRequest::new("rust", "grid_score", "Compute grid matrix score")
            .with_signature("fn grid_score(grid: &[Vec<i32>]) -> i32");

        assert_eq!(infer_geodesic_betti(&linear).beta_1, 0);
        assert_eq!(infer_geodesic_betti(&looped).beta_1, 1);
        assert_eq!(infer_geodesic_betti(&nested).beta_1, 2);
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[test]
    fn test_geodesic_signature_normalization_and_type_defaults() {
        assert_eq!(
            normalize_signature_for_geodesic_emitter("pub fn is_even(n: i32) -> bool {"),
            "fn is_even(n: i32) -> bool"
        );
        assert_eq!(default_expression_for_type("bool"), "false");
        assert_eq!(default_expression_for_type("String"), "String::new()");
        assert_eq!(
            default_expression_for_type("Vec < i32 >"),
            "Default::default()"
        );
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[test]
    fn test_geodesic_skeleton_seed_emits_without_stubs_for_linear_request() {
        use symthaea_geodesic::{build_skeleton_from_topology, emit_rust_from_skeleton};

        let request = SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32");
        let betti = infer_geodesic_betti(&request);
        let hints = geodesic_hints(&request);
        let hint_refs: Vec<&str> = hints.iter().map(String::as_str).collect();
        let mut skeleton = build_skeleton_from_topology(&betti, &hint_refs);

        fill_geodesic_skeleton_defaults(&mut skeleton, &request);

        let signature = request
            .signature
            .as_deref()
            .map(normalize_signature_for_geodesic_emitter);
        let source =
            emit_rust_from_skeleton(&skeleton, &request.name, signature.as_deref(), &hint_refs)
                .expect("filled geodesic skeleton should emit source");

        assert!(source.contains("pub fn add(a: i32, b: i32) -> i32"));
        assert!(!source_has_stub(&source));
        assert!(source.contains("let mut result"));
        assert!(source.contains("result"));
    }

    #[cfg(feature = "geodesic_synthesis")]
    #[test]
    fn test_geodesic_skeleton_seed_emits_loop_shape_for_slice_request() {
        use symthaea_geodesic::{build_skeleton_from_topology, emit_rust_from_skeleton};

        let request = SynthesisRequest::new("rust", "sum", "Sum each number in a slice")
            .with_signature("fn sum(items: &[i32]) -> i32");
        let betti = infer_geodesic_betti(&request);
        let hints = geodesic_hints(&request);
        let hint_refs: Vec<&str> = hints.iter().map(String::as_str).collect();
        let mut skeleton = build_skeleton_from_topology(&betti, &hint_refs);

        fill_geodesic_skeleton_defaults(&mut skeleton, &request);
        let topo = skeleton.topological_signature();

        let signature = request
            .signature
            .as_deref()
            .map(normalize_signature_for_geodesic_emitter);
        let source =
            emit_rust_from_skeleton(&skeleton, &request.name, signature.as_deref(), &hint_refs)
                .expect("filled geodesic skeleton should emit source");

        assert_eq!(betti.beta_1, 1);
        assert_eq!(topo.delta_beta_1, 1);
        assert!(!source_has_stub(&source));
        assert!(source.contains("pub fn sum(items: &[i32]) -> i32"));
    }
}
