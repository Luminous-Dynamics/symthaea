// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! - **DoseResponse** — Graded dose sweeps verify monotonic behavioral output (Clark 1937)
//! - **ToleranceWithdrawal** — Tolerance/withdrawal curves under sustained injection (Koob 2001)
//! - **AntagonistProfiles** — Named antagonist methods + behavioral validation (Stahl 2013)
//! - **MultiTransmitterSynergy** — Paired transmitter interaction validation (Doya 2002)

pub mod allostatic_stress;
pub mod antagonist_profiles;
pub mod attention_network;
pub mod behavioral_knockout;
pub mod consciousness_feedback;
pub mod consciousness_pharmacology;
pub mod dose_response;
pub mod injection_challenge;
#[cfg(feature = "symthaea-backend")]
pub mod live_loop_ablation;
pub mod mood_induction;
pub mod moral_oxytocin;
pub mod multi_transmitter_synergy;
pub mod pharmacological_ablation;
pub mod pharmacological_challenge;
pub mod reward_learning;
pub mod tolerance_withdrawal;
pub mod yerkes_dodson;

pub use allostatic_stress::AllostaticStressBenchmark;
pub use antagonist_profiles::AntagonistProfilesBenchmark;
pub use attention_network::AttentionNetworkBenchmark;
pub use behavioral_knockout::BehavioralKnockoutBenchmark;
pub use consciousness_feedback::ConsciousnessFeedbackBenchmark;
pub use consciousness_pharmacology::ConsciousnessPharmacologyBenchmark;
pub use dose_response::DoseResponseBenchmark;
pub use injection_challenge::InjectionChallengeBenchmark;
#[cfg(feature = "symthaea-backend")]
pub use live_loop_ablation::LiveLoopAblationBenchmark;
pub use mood_induction::MoodInductionBenchmark;
pub use moral_oxytocin::MoralOxytocinBenchmark;
pub use multi_transmitter_synergy::MultiTransmitterSynergyBenchmark;
pub use pharmacological_ablation::PharmacologicalAblationBenchmark;
pub use pharmacological_challenge::PharmacologicalChallengeBenchmark;
pub use reward_learning::RewardLearningBenchmark;
pub use tolerance_withdrawal::ToleranceWithdrawalBenchmark;
pub use yerkes_dodson::YerkesDodsonBenchmark;

#[cfg(all(test, feature = "symthaea-backend"))]
mod live_social_affect;
