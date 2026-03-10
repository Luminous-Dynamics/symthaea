//! Intelligence Module
//!
//! HDC-powered intelligence features:
//! - Bivariate causal discovery (71.3% accuracy on Tübingen benchmark)
//! - Causal consciousness integration (HSIC, attention, LTC bridge)

pub mod causal_consciousness;
pub mod causal_discovery;

pub use causal_consciousness::{
    CausalAnalysisResult, CausalAttention, CausalConsciousness, CausalLTCBridge, GridSearchResult,
    HSICTest, LiveLearningRouter, RandomThresholdSearch, ThresholdTuner,
};
pub use causal_discovery::{CausalDirection, CausalDiscoveryEngine, MetaFeatures};
