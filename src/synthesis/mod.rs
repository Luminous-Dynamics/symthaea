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

// Temporarily disabled to isolate phi_exact for PyPhi validation
// pub mod causal_spec;
// pub mod synthesizer;
// pub mod verifier;
// pub mod adaptive;
// pub mod consciousness_synthesis;  // Enhancement #8: Consciousness-guided synthesis

pub mod phi_exact;  // Enhancement #8 Week 4: Exact IIT Φ via PyPhi

// pub use causal_spec::{
//     CausalSpec, CausalPath, CausalStrength, SpecVerifier,
// };
// pub use synthesizer::{
//     CausalProgramSynthesizer, SynthesisConfig, SynthesizedProgram,
//     ProgramTemplate,
// };
// // SynthesisResult is already defined in this module (line 68)
// pub use verifier::{
//     CounterfactualVerifier, VerificationResult, VerificationConfig,
//     MinimalityChecker,
// };
// pub use adaptive::{
//     AdaptiveProgram, AdaptationStrategy, ProgramMonitor,
// };
// pub use consciousness_synthesis::{
//     ConsciousnessSynthesisConfig, TopologyType, ConsciousSynthesizedProgram,
//     MultiObjectiveScores, ConsciousnessQuality,
// };

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

    /// Consciousness synthesis error (Enhancement #8)
    ConsciousnessSynthesisError(String),

    /// Φ computation timeout
    PhiComputationTimeout { candidate_id: usize, time_ms: u64 },

    /// No candidates meet consciousness constraints
    InsufficientConsciousness { min_phi: f64, best_phi: f64 },

    /// PyPhi import error (Week 4: IIT validation)
    PyPhiImportError { message: String },

    /// PyPhi computation error (Week 4: IIT validation)
    PyPhiComputationError { message: String },

    /// Topology too large for exact IIT (Week 4: IIT validation)
    PhiExactTooLarge { size: usize, recommended_max: usize },

    /// PyPhi feature not enabled (Week 4: IIT validation)
    PyPhiNotEnabled { message: String },

    /// Internal error (shouldn't happen)
    InternalError(String),

    /// Unsatisfiable specification (alias for backward compatibility)
    UnsatisfiableSpecification(String),
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsatisfiableSpec(msg) => write!(f, "Unsatisfiable specification: {}", msg),
            Self::NoPathExists(msg) => write!(f, "No causal path exists: {}", msg),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::NotMinimal(msg) => write!(f, "Program not minimal: {}", msg),
            Self::CausalEngineError(msg) => write!(f, "Causal engine error: {}", msg),
            Self::ConsciousnessSynthesisError(msg) => write!(f, "Consciousness synthesis error: {}", msg),
            Self::PhiComputationTimeout { candidate_id, time_ms } => {
                write!(f, "Φ computation timeout for candidate {}: {}ms", candidate_id, time_ms)
            }
            Self::InsufficientConsciousness { min_phi, best_phi } => {
                write!(f, "No candidates meet Φ threshold: best={:.3} < min={:.3}", best_phi, min_phi)
            }
            Self::PyPhiImportError { message } => write!(f, "PyPhi import error: {}", message),
            Self::PyPhiComputationError { message } => write!(f, "PyPhi computation error: {}", message),
            Self::PhiExactTooLarge { size, recommended_max } => {
                write!(f, "Topology too large for exact IIT: n={} (recommended max: n={})", size, recommended_max)
            }
            Self::PyPhiNotEnabled { message } => write!(f, "PyPhi not enabled: {}", message),
            Self::InternalError(msg) => write!(f, "Internal error: {}", msg),
            Self::UnsatisfiableSpecification(msg) => write!(f, "Unsatisfiable specification: {}", msg),
        }
    }
}

impl std::error::Error for SynthesisError {}

pub type SynthesisResult<T> = Result<T, SynthesisError>;
