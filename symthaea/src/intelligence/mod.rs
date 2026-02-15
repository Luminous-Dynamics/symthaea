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
pub mod autofix;
pub mod causal_consciousness;
pub mod causal_discovery;
pub mod conflict;
pub mod error_explain;
pub mod multivariate_causal;
pub mod nixos_causal;

pub use adaptive::{AdaptiveCompleter, PatternStore, UserPattern};
pub use autofix::{AutoFixer, Fix, FixSuggestion};
pub use causal_consciousness::{
    CausalAnalysisResult, CausalAttention, CausalConsciousness, CausalLTCBridge, GridSearchResult,
    HSICTest, LiveLearningRouter, RandomThresholdSearch, ThresholdTuner,
};
pub use causal_discovery::{CausalDirection, CausalDiscoveryEngine, MetaFeatures};
pub use conflict::{Conflict, ConflictDetector, ConflictSeverity};
pub use error_explain::{ErrorCategory, ErrorExplanation, SemanticErrorExplainer};
pub use multivariate_causal::{CausalDAG, DirectedEdge, PCAlgorithm, Variable};
pub use nixos_causal::{
    CausalEdge, NixOSCausalAnalyzer, NixOSCausalPatterns, RootCause, RootCauseAnalysis,
    SideEffectPrediction,
};
