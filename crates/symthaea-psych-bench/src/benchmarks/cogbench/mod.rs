//! CogBench: Cognitive psychology experiments via FEP active inference.
//!
//! 7 experiments testing exploration, model-based reasoning, learning,
//! temporal discounting, and risk-taking using `ActiveInferenceAgent`.

pub mod bart;
pub mod horizon;
pub mod instrumental;
pub mod probabilistic;
pub mod restless_bandit;
pub mod temporal_discounting;
pub mod two_step;

pub use bart::BartBenchmark;
pub use horizon::HorizonBenchmark;
pub use instrumental::InstrumentalLearningBenchmark;
pub use probabilistic::ProbabilisticReasoningBenchmark;
pub use restless_bandit::RestlessBanditBenchmark;
pub use temporal_discounting::TemporalDiscountingBenchmark;
pub use two_step::TwoStepBenchmark;
