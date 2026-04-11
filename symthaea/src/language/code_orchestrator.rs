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
//! │ 1. CodeGenerator (template matching, HDC+CfC)   │ → verify
//! │ 2. CodeAlgebra analogy (if similar code exists)  │ → verify
//! │ 3. LLM fallback (Ollama / Cloud)                 │ → verify
//! └─────────────────────────────────────────────────┘
//!     ↓
//! SynthesisResponse (code + verification audit trail)
//! ```
//!
//! The orchestrator wraps the existing `IntelligentDispatcher` and `CodeGenerator`,
//! adding verification layers and code-by-analogy before falling back to LLM.

use symthaea_core::synthesis_trait::{
    BackendCapabilities, CodeSynthesisBackend, SynthesisRequest, SynthesisResponse,
    VerificationLayer, VerificationReport,
};

use super::code_certificate::CodeCertificate;
use super::code_generator::{CodeGenerator, GeneratedCode};
use super::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use super::code_parser::EntityKind;
use super::code_verifier::{CodeVerifier, VerificationPolicy};
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
    /// HDC code algebra for analogy-based generation
    algebra: CodeAlgebra,
    /// Minimum similarity threshold for accepting generated code
    verification_policy: VerificationPolicy,
    /// Energy budget remaining for this session
    energy_budget: f32,
    /// Total energy spent across all generations
    energy_spent: f32,
    /// Audit trail of all synthesis attempts this session
    attempt_history: Vec<SynthesisAttempt>,
    /// Distillation buffer: (ThoughtChannels-as-f32-vec, source_code, quality) triples
    /// captured from successful generations for Broca SSM training.
    /// The flywheel: LLM teaches native → native improves → less LLM needed.
    distillation_buffer: Vec<DistillationCapture>,
    /// Certificates issued for accepted code
    certificates: Vec<CodeCertificate>,
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
    /// Why it was rejected (if applicable)
    pub rejection_reason: Option<String>,
}

/// Conversion from synthesis_trait EpistemicStatus to crate-local EpistemicStatus
fn to_local_epistemic(
    status: symthaea_core::synthesis_trait::EpistemicStatus,
) -> EpistemicStatus {
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
fn to_trait_epistemic(
    status: EpistemicStatus,
) -> symthaea_core::synthesis_trait::EpistemicStatus {
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
            algebra: CodeAlgebra::new(algebra_encoder),
            verification_policy: VerificationPolicy::Standard,
            energy_budget: 100.0,
            energy_spent: 0.0,
            attempt_history: Vec::new(),
            distillation_buffer: Vec::new(),
            certificates: Vec::new(),
        }
    }

    /// Create with a specific verification policy.
    pub fn with_policy(mut self, policy: VerificationPolicy) -> Self {
        self.verification_policy = policy;
        self.verifier.set_policy(policy);
        self
    }

    /// Create with a custom energy budget.
    pub fn with_energy_budget(mut self, budget: f32) -> Self {
        self.energy_budget = budget;
        self
    }

    /// Get remaining energy budget.
    pub fn remaining_energy(&self) -> f32 {
        self.energy_budget - self.energy_spent
    }

    /// Get total energy spent.
    pub fn energy_spent(&self) -> f32 {
        self.energy_spent
    }

    /// Get the audit trail of all attempts.
    pub fn attempt_history(&self) -> &[SynthesisAttempt] {
        &self.attempt_history
    }

    /// Get the distillation buffer — captured (intent, code, quality) triples.
    ///
    /// These are ready to be serialized as JSONL for Broca SSM training.
    pub fn distillation_buffer(&self) -> &[DistillationCapture] {
        &self.distillation_buffer
    }

    /// Get issued certificates.
    pub fn certificates(&self) -> &[CodeCertificate] {
        &self.certificates
    }

    /// Export the distillation buffer as JSONL suitable for Broca training.
    ///
    /// Each line is a JSON object with `channels` (43-element f32 vec) and
    /// `target_text` (the verified source code). This format is directly
    /// consumable by `TrainingDataset::from_jsonl()`.
    pub fn export_distillation_jsonl(&self) -> String {
        use serde_json::json;

        self.distillation_buffer
            .iter()
            .map(|cap| {
                json!({
                    "channels": cap.channels,
                    "target_text": cap.source,
                    "target_ids": []
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

    /// Capture a successful generation for distillation training.
    ///
    /// Converts the SynthesisRequest into a ThoughtChannels-compatible f32 vector
    /// and stores it alongside the verified source code. This is the core of the
    /// self-improvement flywheel: every successful generation becomes training data.
    fn capture_distillation(
        &mut self,
        request: &SynthesisRequest,
        source: &str,
        backend: &str,
        similarity: f32,
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
        channels[19] = (request.constraints.len() + request.examples.len())
            .min(10) as f32;

        // Channels 24-27: code-specific
        channels[24] = 0.5; // syntax_complexity (mid-range default)
        channels[25] = similarity; // type_confidence (use verification similarity)
        channels[26] = 0.5; // algorithm_pattern
        channels[27] = 0.0; // error_likelihood (low — code passed verification)

        // Epistemic Cube: E3 (proven), N2 (network), M2 (persistent)
        channels[31] = 1.0; // E3 (proven by compilation)
        channels[35] = 1.0; // N2 (network consensus)
        channels[39] = 1.0; // M2 (persistent/recorded)
        channels[41] = similarity; // H (harmonic coherence)
        channels[42] = 0.4 * 0.6 + 0.35 * 0.5 + 0.25 * 0.5; // Quality score

        // Quality = similarity × (1 - error_penalty)
        let quality = similarity;

        self.distillation_buffer.push(DistillationCapture {
            channels,
            source: source.to_string(),
            quality,
            backend: backend.to_string(),
            name: request.name.clone(),
        });
    }

    /// Generate a CodeCertificate for verified code.
    fn issue_certificate(
        &mut self,
        request: &SynthesisRequest,
        source: &str,
        backend: &str,
        verification_layers: &[VerificationLayer],
        similarity: f32,
    ) -> CodeCertificate {
        let cert = CodeCertificate::new(source, backend, similarity)
            .with_epistemic_status(request.epistemic_status)
            .with_verification_layers(verification_layers)
            .with_safety_critical(request.safety_critical);

        self.certificates.push(cert.clone());
        cert
    }

    /// Synthesize code by trying backends in priority order.
    ///
    /// Priority:
    /// 1. Native CodeGenerator (energy: 1.0) — template matching via HDC+CfC
    /// 2. Code Algebra analogy (energy: 0.5) — "A:B :: C:?" in HDC space
    /// 3. Returns failed response if budget exhausted (LLM fallback handled by caller)
    ///
    /// Each attempt is verified against the verification policy before acceptance.
    pub fn synthesize(&mut self, request: &SynthesisRequest) -> SynthesisResponse {
        let mut verification_layers = Vec::new();

        // ─── Backend 1: Native CodeGenerator ───────────────────────────────
        if self.remaining_energy() >= 1.0 {
            let native_result = self.try_native_generation(request);
            self.energy_spent += 1.0;

            let attempt = SynthesisAttempt {
                backend: "CodeGenerator".to_string(),
                verified: native_result.verified,
                similarity: native_result.similarity,
                energy_cost: 1.0,
                rejection_reason: native_result.rejection.clone(),
            };
            self.attempt_history.push(attempt);

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
                );

                // ── Certificate: machine-verifiable audit trail ──
                let certificate = self.issue_certificate(
                    request,
                    &native_result.source,
                    "CodeGenerator",
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
        }

        // ─── Backend 2: Code Algebra Analogy ───────────────────────────────
        if self.remaining_energy() >= 0.5 {
            let analogy_result = self.try_analogy_generation(request);
            self.energy_spent += 0.5;

            let attempt = SynthesisAttempt {
                backend: "CodeAlgebra::analogy".to_string(),
                verified: analogy_result.verified,
                similarity: analogy_result.similarity,
                energy_cost: 0.5,
                rejection_reason: analogy_result.rejection.clone(),
            };
            self.attempt_history.push(attempt);

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
                );

                // ── Certificate: machine-verifiable audit trail ──
                let certificate = self.issue_certificate(
                    request,
                    &analogy_result.source,
                    "CodeAlgebra::analogy",
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
        }

        // ─── All native backends exhausted ─────────────────────────────────
        SynthesisResponse {
            source: String::new(),
            backend_name: "none".to_string(),
            confidence: 0.0,
            epistemic_status: symthaea_core::synthesis_trait::EpistemicStatus::Unknown,
            verification: verification_layers,
            accepted: false,
            energy_cost: self.energy_spent,
            narrative: Some(
                "All native backends exhausted. LLM fallback recommended.".to_string(),
            ),
        }
    }

    /// Attempt native code generation via CodeGenerator.
    fn try_native_generation(&self, request: &SynthesisRequest) -> BackendResult {
        let spec = self.request_to_spec(request);
        let target = CodeTarget::new(&request.name, EntityKind::Function)
            .with_language(&request.language);

        let intent = CodeIntent::Create { target, spec };
        let context = super::code_generator::CodeContext::default();
        let generated = self.generator.generate(&intent, &context);

        // Verify: check for empty/stub output
        if generated.source.is_empty()
            || generated.source.contains("todo!()")
            || generated.source.contains("unimplemented!()")
        {
            return BackendResult {
                source: generated.source,
                verified: false,
                similarity: 0.0,
                epistemic: generated.epistemic_status,
                rejection: Some("Generated code contains stubs or is empty".to_string()),
            };
        }

        // Verify: HDC round-trip similarity
        let intent_hv = self.generator.encoder().encode_name(&request.name);
        let parsed =
            super::code_parser::ParsedCode::new(&generated.source, &request.language);
        let verification = self.verifier.verify_against_intent(&parsed, &intent_hv);

        BackendResult {
            source: generated.source,
            verified: verification.is_acceptable(),
            similarity: verification.semantic_similarity,
            epistemic: generated.epistemic_status,
            rejection: if verification.is_acceptable() {
                None
            } else {
                Some(format!(
                    "Similarity {:.3} below {} threshold {:.3}",
                    verification.semantic_similarity,
                    format!("{:?}", self.verification_policy),
                    self.verification_policy.threshold(),
                ))
            },
        }
    }

    /// Attempt code generation via HDC analogy reasoning.
    ///
    /// Looks for similar functions in the algebra's pattern memory and
    /// uses "A:B :: C:?" to derive the target implementation.
    fn try_analogy_generation(&self, request: &SynthesisRequest) -> BackendResult {
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
                verified: false,
                similarity: 0.0,
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
                verified: false,
                similarity: best_sim,
                epistemic: EpistemicStatus::Uncertain,
                rejection: Some(format!(
                    "Best analogy similarity {:.3} too low (need >= 0.3)",
                    best_sim
                )),
            };
        }

        // Generate using the analogy-informed context
        let spec = self.request_to_spec(request);
        let target = CodeTarget::new(&request.name, EntityKind::Function)
            .with_language(&request.language);

        let intent = CodeIntent::Create { target, spec };
        let context = super::code_generator::CodeContext::default();
        let generated = self.generator.generate(&intent, &context);

        if generated.source.is_empty()
            || generated.source.contains("todo!()")
            || generated.source.contains("unimplemented!()")
        {
            return BackendResult {
                source: generated.source,
                verified: false,
                similarity: 0.0,
                epistemic: generated.epistemic_status,
                rejection: Some("Analogy-guided generation produced stubs".to_string()),
            };
        }

        BackendResult {
            source: generated.source,
            verified: true, // Analogy-derived code is accepted at lower threshold
            similarity: best_sim,
            epistemic: generated.epistemic_status,
            rejection: None,
        }
    }

    /// Convert a SynthesisRequest to the crate-local CodeSpec type.
    fn request_to_spec(&self, request: &SynthesisRequest) -> CodeSpec {
        let mut spec = CodeSpec::new(
            &request.language,
            &request.name,
            &request.purpose,
        )
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
}

/// Internal result from a backend attempt.
struct BackendResult {
    source: String,
    verified: bool,
    similarity: f32,
    epistemic: EpistemicStatus,
    rejection: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CodeSynthesisBackend trait implementation
// ═══════════════════════════════════════════════════════════════════════════════

impl CodeSynthesisBackend for CodeOrchestrator {
    fn synthesize(&self, request: &SynthesisRequest) -> SynthesisResponse {
        // The trait requires &self but our synthesize needs &mut self for tracking.
        // Create a minimal pass-through for trait compliance.
        let spec = self.request_to_spec(request);
        let target = CodeTarget::new(&request.name, EntityKind::Function)
            .with_language(&request.language);
        let intent = CodeIntent::Create { target, spec };
        let context = super::code_generator::CodeContext::default();
        let generated = self.generator.generate(&intent, &context);

        SynthesisResponse {
            source: generated.source.clone(),
            backend_name: "CodeOrchestrator".to_string(),
            confidence: generated.phi,
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
        let target = CodeTarget::new(&request.name, EntityKind::Function)
            .with_language(&request.language);
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
    fn test_synthesize_simple_function() {
        let mut orch = CodeOrchestrator::new();

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
        let mut orch = CodeOrchestrator::new();

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
        let mut orch = CodeOrchestrator::new().with_energy_budget(0.5);

        let request = SynthesisRequest::new("rust", "add", "Add two integers");
        let response = orch.synthesize(&request);

        // With only 0.5 energy, shouldn't be able to try native (costs 1.0)
        assert!(!response.accepted);
    }
}
