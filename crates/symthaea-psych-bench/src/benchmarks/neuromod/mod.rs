//! Neuromodulator domain benchmarks.
//!
//! Validates neuromodulator dynamics against known human psychopharmacological effects.
//! - **RewardLearning** — DA reward prediction error drives reversal learning (Schultz 1997)
//! - **YerkesDodson** — Inverted-U performance curve under varying NE (Yerkes-Dodson 1908)
//! - **AttentionNetwork** — ANT-like alerting (NE), orienting (ACh), conflict (DA) (Posner 1990)
//! - **MoodInduction** — 5-HT mood-cognition interaction and risk preference (Dayan & Huys 2009)

pub mod attention_network;
pub mod mood_induction;
pub mod pharmacological_ablation;
pub mod reward_learning;
pub mod yerkes_dodson;

pub use attention_network::AttentionNetworkBenchmark;
pub use mood_induction::MoodInductionBenchmark;
pub use pharmacological_ablation::PharmacologicalAblationBenchmark;
pub use reward_learning::RewardLearningBenchmark;
pub use yerkes_dodson::YerkesDodsonBenchmark;
