//! Neuromodulator domain benchmarks.
//!
//! Validates neuromodulator dynamics against known human psychopharmacological effects.
//! - **RewardLearning** — DA reward prediction error drives reversal learning (Schultz 1997)
//! - **YerkesDodson** — Inverted-U performance curve under varying NE (Yerkes-Dodson 1908)
//! - **AttentionNetwork** — ANT-like alerting (NE), orienting (ACh), conflict (DA) (Posner 1990)
//! - **MoodInduction** — 5-HT mood-cognition interaction and risk preference (Dayan & Huys 2009)
//! - **LiveLoopAblation** — Transmitter knockout on real CognitiveLoopService (Doya 2002)
//! - **PharmacologicalChallenge** — Agonist (0.9) + antagonist (0.1) inverted-U (Arnsten 2011)
//! - **InjectionChallenge** — PK injection + tolerance dynamics (Arnsten 2011)
//! - **AllostaticStress** — Chronic stress load accumulation and recovery (McEwen 1998)
//! - **BehavioralKnockout** — Cohen's d effect sizes for transmitter knockouts (Doya 2002)
//! - **ConsciousnessPharmacology** — Pharmacological injection → consciousness round-trip

pub mod allostatic_stress;
pub mod attention_network;
pub mod behavioral_knockout;
pub mod consciousness_pharmacology;
pub mod injection_challenge;
#[cfg(feature = "symthaea-backend")]
pub mod live_loop_ablation;
pub mod mood_induction;
pub mod pharmacological_ablation;
pub mod pharmacological_challenge;
pub mod reward_learning;
pub mod yerkes_dodson;

pub use allostatic_stress::AllostaticStressBenchmark;
pub use attention_network::AttentionNetworkBenchmark;
pub use behavioral_knockout::BehavioralKnockoutBenchmark;
pub use consciousness_pharmacology::ConsciousnessPharmacologyBenchmark;
pub use injection_challenge::InjectionChallengeBenchmark;
#[cfg(feature = "symthaea-backend")]
pub use live_loop_ablation::LiveLoopAblationBenchmark;
pub use mood_induction::MoodInductionBenchmark;
pub use pharmacological_ablation::PharmacologicalAblationBenchmark;
pub use pharmacological_challenge::PharmacologicalChallengeBenchmark;
pub use reward_learning::RewardLearningBenchmark;
pub use yerkes_dodson::YerkesDodsonBenchmark;
