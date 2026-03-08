//! Social cognition domain benchmarks.
//!
//! - **RME** — Reading the Mind in the Eyes (emotion recognition from eyes)
//! - **UltimatumGame** — Fairness sensitivity in social exchange
//! - **SocialNorm** — Social norm violation detection (signal detection)
//! - **PrisonersDilemma** — Cooperation under temptation to defect
//! - **PublicGoods** — Contribution to shared resources under free-riding pressure
//! - **DictatorGame** — Intrinsic altruism without strategic incentives

pub mod dictator_game;
pub mod prisoners_dilemma;
pub mod public_goods;
pub mod rme;
pub mod social_norm;
pub mod ultimatum_game;

pub use dictator_game::DictatorGameBenchmark;
pub use prisoners_dilemma::PrisonersDilemmaBenchmark;
pub use public_goods::PublicGoodsBenchmark;
pub use rme::RmeBenchmark;
pub use social_norm::SocialNormBenchmark;
pub use ultimatum_game::UltimatumGameBenchmark;
