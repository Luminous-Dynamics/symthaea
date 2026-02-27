//! Social cognition domain benchmarks.
//!
//! - **RME** — Reading the Mind in the Eyes (emotion recognition from eyes)
//! - **UltimatumGame** — Fairness sensitivity in social exchange
//! - **SocialNorm** — Social norm violation detection (signal detection)

pub mod rme;
pub mod social_norm;
pub mod ultimatum_game;

pub use rme::RmeBenchmark;
pub use social_norm::SocialNormBenchmark;
pub use ultimatum_game::UltimatumGameBenchmark;
