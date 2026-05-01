// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Code Synthesis Trait
//!
//! Defines the common interface for all code synthesis backends:
//! - **CodeGenerator** (template-based HDC+CfC generation)
//! - **GeodesicSynthesizer** (topology-first with Betti number constraints)
//! - **BrocaCodeSynthesis** (autoregressive CfC-HDC with epistemic gating)
//!
//! This trait lives in `symthaea-core` to avoid circular dependencies between
//! the main `symthaea` crate and sub-crates like `symthaea-geodesic`.
//!
//! # Architecture
//!
//! ```text
//! SynthesisRequest (unified spec)
//!     ↓
//! CodeSynthesisBackend::synthesize()
//!     ↓
//! SynthesisResponse (code + verification audit trail)
//! ```

use std::fmt;

/// Epistemic status of generated code — how confident the system is.
///
/// Maps to Broca's epistemic gating levels. At `Unknown` or `OutOfDomain`,
/// the system should defer to a more capable backend or decline to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicStatus {
    /// High confidence — verified pattern, known solution
    Certain,
    /// Moderate confidence — good match to known patterns
    Probable,
    /// Low confidence — partial match, may contain errors
    Uncertain,
    /// Very low confidence — outside known pattern space
    Unknown,
    /// No relevant knowledge — should not attempt generation
    OutOfDomain,
}

impl fmt::Display for EpistemicStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Certain => write!(f, "Certain"),
            Self::Probable => write!(f, "Probable"),
            Self::Uncertain => write!(f, "Uncertain"),
            Self::Unknown => write!(f, "Unknown"),
            Self::OutOfDomain => write!(f, "OutOfDomain"),
        }
    }
}

/// Topological constraint on generated code structure.
///
/// Derived from Geodesic synthesis: Betti numbers encode structural invariants
/// (connected components, loops, voids) that must hold in the output.
#[derive(Debug, Clone)]
pub struct TopologicalConstraint {
    /// β₀: number of connected components (independent modules)
    pub beta_0: Option<usize>,
    /// β₁: number of independent cycles (loops)
    pub beta_1: Option<usize>,
    /// β₂: number of enclosed voids (dead-code regions)
    pub beta_2: Option<usize>,
    /// Human-readable description of the constraint
    pub description: String,
}

/// Unified request for code synthesis across all backends.
///
/// Merges the three previously incompatible spec types:
/// - `CodeSpec` from `code_intent.rs` (CodeGenerator)
/// - `CodeSpec` from `geodesic/synthesis.rs` (Geodesic)
/// - `ThoughtChannels` from Broca
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Target programming language (e.g., "rust", "python", "nix")
    pub language: String,
    /// Name of the function/type to generate
    pub name: String,
    /// Natural language description of what the code should do
    pub purpose: String,
    /// Optional function signature (e.g., "fn add(a: i32, b: i32) -> i32")
    pub signature: Option<String>,
    /// Input/output examples as (input_desc, expected_output) pairs
    pub examples: Vec<(String, String)>,
    /// Constraints on the generated code (e.g., "must not allocate", "O(n log n)")
    pub constraints: Vec<String>,
    /// Epistemic confidence about the task
    pub epistemic_status: EpistemicStatus,
    /// Topological constraints from Geodesic analysis
    pub topology_constraints: Vec<TopologicalConstraint>,
    /// Consciousness level (Phi) at time of request — gates backend selection
    pub consciousness_level: f32,
    /// Whether the generated code involves side effects (file I/O, network, etc.)
    pub has_side_effects: bool,
    /// Whether this is safety-critical code (medical, aerospace, etc.)
    pub safety_critical: bool,
}

impl SynthesisRequest {
    /// Create a minimal synthesis request
    pub fn new(language: &str, name: &str, purpose: &str) -> Self {
        Self {
            language: language.to_string(),
            name: name.to_string(),
            purpose: purpose.to_string(),
            signature: None,
            examples: Vec::new(),
            constraints: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            topology_constraints: Vec::new(),
            consciousness_level: 0.5,
            has_side_effects: false,
            safety_critical: false,
        }
    }

    /// Set the function signature
    pub fn with_signature(mut self, sig: &str) -> Self {
        self.signature = Some(sig.to_string());
        self
    }

    /// Add an input/output example
    pub fn with_example(mut self, input: &str, output: &str) -> Self {
        self.examples.push((input.to_string(), output.to_string()));
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: &str) -> Self {
        self.constraints.push(constraint.to_string());
        self
    }

    /// Mark as safety-critical
    pub fn safety_critical(mut self) -> Self {
        self.safety_critical = true;
        self
    }
}

/// Which verification checks were applied to the generated code.
#[derive(Debug, Clone)]
pub struct VerificationLayer {
    /// Name of the verification (e.g., "syntax", "semantic_similarity", "topology", "sheaf")
    pub name: String,
    /// Whether this check passed
    pub passed: bool,
    /// Score or metric (0.0-1.0 where applicable)
    pub score: Option<f32>,
    /// Details or error message
    pub detail: String,
}

/// Response from a code synthesis backend.
#[derive(Debug, Clone)]
pub struct SynthesisResponse {
    /// The generated source code (empty string if generation failed)
    pub source: String,
    /// Which backend produced this code
    pub backend_name: String,
    /// Confidence in the generated code (0.0-1.0)
    pub confidence: f32,
    /// Epistemic status of the output
    pub epistemic_status: EpistemicStatus,
    /// Verification layers applied (audit trail)
    pub verification: Vec<VerificationLayer>,
    /// Whether the code is considered acceptable
    pub accepted: bool,
    /// Energy cost of this generation (for metabolic budgeting)
    pub energy_cost: f32,
    /// Human-readable explanation of the generation process
    pub narrative: Option<String>,
}

impl SynthesisResponse {
    /// Create a failed response
    pub fn failed(backend: &str, reason: &str) -> Self {
        Self {
            source: String::new(),
            backend_name: backend.to_string(),
            confidence: 0.0,
            epistemic_status: EpistemicStatus::Unknown,
            verification: Vec::new(),
            accepted: false,
            energy_cost: 0.0,
            narrative: Some(reason.to_string()),
        }
    }
}

/// Capabilities that a synthesis backend provides.
///
/// Used by the orchestrator to route requests to the most appropriate backend.
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    /// Can enforce topological invariants (Betti numbers) before generating
    pub topology_guarantees: bool,
    /// Provides epistemic gating at the logit level (hallucination prevention)
    pub epistemic_gating: bool,
    /// Uses template matching for known algorithm families
    pub template_matching: bool,
    /// Can formally verify properties (bounds, termination, overflow)
    pub formal_verification: bool,
    /// Can predict execution outcome without running code (CfC O(1) oracle)
    pub execution_prediction: bool,
    /// Verifies local-to-global interface coherence (sheaf theory)
    pub sheaf_composability: bool,
    /// Supports autoregressive token-by-token generation
    pub autoregressive: bool,
    /// Can generate code for these languages
    pub supported_languages: Vec<String>,
}

/// Report from verifying generated code against a synthesis request.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether the code passes verification
    pub passed: bool,
    /// Individual verification layers with results
    pub layers: Vec<VerificationLayer>,
    /// Overall semantic similarity to intent (0.0-1.0)
    pub semantic_similarity: f32,
    /// Summary message
    pub summary: String,
}

/// Unified trait for all code synthesis backends.
///
/// Each implementor has different strengths:
/// - **CodeGenerator**: Fast template matching, good for known algorithm patterns
/// - **GeodesicSynthesizer**: Topology-guaranteed structure, enforces Betti numbers before code exists
/// - **BrocaCodeSynthesis**: Autoregressive generation with epistemic/emotional gating at logit level
///
/// The orchestrator (`code_orchestrator.rs`) routes requests to backends based on
/// `confidence_for()` scores and `capabilities()`.
pub trait CodeSynthesisBackend: Send + Sync {
    /// Generate code from a unified specification.
    fn synthesize(&self, request: &SynthesisRequest) -> SynthesisResponse;

    /// How confident is this backend about handling this request? (0.0-1.0)
    ///
    /// Used by the orchestrator to select the best backend. Higher = more appropriate.
    /// Return 0.0 if the backend cannot handle this request at all.
    fn confidence_for(&self, request: &SynthesisRequest) -> f32;

    /// Verify generated code against the original request.
    ///
    /// Can be called on code from any backend — enables cross-verification
    /// (e.g., Geodesic verifying Broca's output).
    fn verify(&self, code: &str, request: &SynthesisRequest) -> VerificationReport;

    /// Name of this backend (for telemetry and audit trails).
    fn backend_name(&self) -> &str;

    /// What this backend can do (for routing decisions).
    fn capabilities(&self) -> BackendCapabilities;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_request_builder() {
        let req = SynthesisRequest::new("rust", "fibonacci", "Calculate the nth Fibonacci number")
            .with_signature("fn fibonacci(n: u64) -> u64")
            .with_example("fibonacci(0)", "0")
            .with_example("fibonacci(10)", "55")
            .with_constraint("O(n) time complexity");

        assert_eq!(req.language, "rust");
        assert_eq!(req.name, "fibonacci");
        assert_eq!(
            req.signature.as_deref(),
            Some("fn fibonacci(n: u64) -> u64")
        );
        assert_eq!(req.examples.len(), 2);
        assert_eq!(req.constraints.len(), 1);
        assert_eq!(req.epistemic_status, EpistemicStatus::Probable);
        assert!(!req.safety_critical);
    }

    #[test]
    fn test_synthesis_request_safety_critical() {
        let req = SynthesisRequest::new("rust", "dose_calc", "Calculate medication dosage")
            .safety_critical();
        assert!(req.safety_critical);
    }

    #[test]
    fn test_synthesis_response_failed() {
        let resp = SynthesisResponse::failed("test_backend", "insufficient confidence");
        assert!(!resp.accepted);
        assert!(resp.source.is_empty());
        assert_eq!(resp.confidence, 0.0);
        assert_eq!(resp.epistemic_status, EpistemicStatus::Unknown);
    }

    #[test]
    fn test_epistemic_status_display() {
        assert_eq!(format!("{}", EpistemicStatus::Certain), "Certain");
        assert_eq!(format!("{}", EpistemicStatus::OutOfDomain), "OutOfDomain");
    }

    #[test]
    fn test_backend_capabilities_default() {
        let caps = BackendCapabilities::default();
        assert!(!caps.topology_guarantees);
        assert!(!caps.epistemic_gating);
        assert!(caps.supported_languages.is_empty());
    }

    #[test]
    fn test_verification_policy_thresholds() {
        let layer = VerificationLayer {
            name: "semantic_similarity".to_string(),
            passed: true,
            score: Some(0.85),
            detail: "Above Standard threshold (0.7)".to_string(),
        };
        assert!(layer.passed);
        assert_eq!(layer.score, Some(0.85));
    }
}
