//! Intelligence Module
//!
//! HDC-powered intelligence features:
//! - H1: Adaptive completions (learning from user patterns)
//! - H2: Semantic error explanations
//! - H3: Conflict detection
//! - H4: Auto-fix suggestions
//! - H5: Bivariate causal discovery (71.3% accuracy on Tübingen benchmark)
//! - H6: Causal consciousness integration (HSIC, attention, LTC bridge)
//! - H7: NixOS causal analysis (root cause detection, side effect prediction)
//! - H8: Multivariate DAG discovery (PC algorithm with KCIT)

pub mod adaptive;
pub mod error_explain;
pub mod conflict;
pub mod autofix;
pub mod causal_discovery;
pub mod causal_consciousness;
pub mod nixos_causal;
pub mod multivariate_causal;

pub use adaptive::{AdaptiveCompleter, UserPattern, PatternStore};
pub use error_explain::{SemanticErrorExplainer, ErrorExplanation, ErrorCategory};
pub use conflict::{ConflictDetector, Conflict, ConflictSeverity};
pub use autofix::{AutoFixer, Fix, FixSuggestion};
pub use causal_discovery::{CausalDiscoveryEngine, CausalDirection, MetaFeatures};
pub use causal_consciousness::{
    CausalConsciousness, CausalAttention, CausalLTCBridge,
    HSICTest, LiveLearningRouter, CausalAnalysisResult,
    ThresholdTuner, RandomThresholdSearch, GridSearchResult,
};
pub use nixos_causal::{
    NixOSCausalAnalyzer, RootCauseAnalysis, RootCause,
    SideEffectPrediction, CausalEdge, NixOSCausalPatterns,
};
pub use multivariate_causal::{
    PCAlgorithm, CausalDAG, Variable, DirectedEdge,
};
