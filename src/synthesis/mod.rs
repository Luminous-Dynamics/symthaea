// Enhancement #7: Causal Program Synthesis
//
// Revolutionary approach to program synthesis using causal reasoning
// instead of correlation-based synthesis.
//
// Core Innovation: Synthesize programs that capture TRUE causal relationships
//
// Phases:
// 1. Causal Specification - Define desired causal effects
// 2. Causal Discovery - Learn current causal relationships (Enhancement #4)
// 3. Program Synthesis - Generate programs implementing specifications
// 4. Counterfactual Verification - Verify correctness via counterfactuals
// 5. Explanation Generation - Explain why programs work (Enhancement #4)

pub mod causal_spec;
pub mod synthesizer;
pub mod verifier;
pub mod adaptive;

pub use causal_spec::{
    CausalSpec, CausalPath, CausalStrength, SpecVerifier,
};
pub use synthesizer::{
    CausalProgramSynthesizer, SynthesisConfig, SynthesizedProgram,
    ProgramTemplate,
};
// SynthesisResult is already defined in this module (line 68)
pub use verifier::{
    CounterfactualVerifier, VerificationResult, VerificationConfig,
    MinimalityChecker,
};
pub use adaptive::{
    AdaptiveProgram, AdaptationStrategy, ProgramMonitor,
};

/// Synthesis error types
#[derive(Debug, Clone)]
pub enum SynthesisError {
    /// Specification is unsatisfiable
    UnsatisfiableSpec(String),

    /// Cannot find causal path
    NoPathExists(String),

    /// Verification failed
    VerificationFailed(String),

    /// Program is not minimal
    NotMinimal(String),

    /// Integration error with Enhancement #4
    CausalEngineError(String),
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsatisfiableSpec(msg) => write!(f, "Unsatisfiable specification: {}", msg),
            Self::NoPathExists(msg) => write!(f, "No causal path exists: {}", msg),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::NotMinimal(msg) => write!(f, "Program not minimal: {}", msg),
            Self::CausalEngineError(msg) => write!(f, "Causal engine error: {}", msg),
        }
    }
}

impl std::error::Error for SynthesisError {}

pub type SynthesisResult<T> = Result<T, SynthesisError>;
