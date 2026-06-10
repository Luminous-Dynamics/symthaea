// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social cognition domain benchmarks.
//!
//! - **RME** — Reading the Mind in the Eyes (emotion recognition from eyes)
//! - **UltimatumGame** — Fairness sensitivity in social exchange
//! - **SocialNorm** — Social norm violation detection (signal detection)
//! - **PrisonersDilemma** — Cooperation under temptation to defect
//! - **PublicGoods** — Contribution to shared resources under free-riding pressure
//! - **DictatorGame** — Intrinsic altruism without strategic incentives
//! - **Machiavelli** — Deception, power-seeking, and harm detection (Pan et al., 2023)

pub mod dictator_game;
pub mod machiavelli;
pub mod prisoners_dilemma;
pub mod public_goods;
pub mod rme;
pub mod social_norm;
pub mod ultimatum_game;

pub use dictator_game::DictatorGameBenchmark;
pub use machiavelli::MachiavelliBenchmark;
pub use prisoners_dilemma::PrisonersDilemmaBenchmark;
pub use public_goods::PublicGoodsBenchmark;
pub use rme::RmeBenchmark;
pub use social_norm::SocialNormBenchmark;
pub use ultimatum_game::UltimatumGameBenchmark;
