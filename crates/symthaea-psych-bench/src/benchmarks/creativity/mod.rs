//! Creativity domain benchmarks.
//!
//! - **RemoteAssociates** — Find a word connecting three cue words (RAT)
//! - **AlternateUses** — Generate novel uses for common objects (AUT)
//! - **DivergentThinking** — Semantic distance of generated uses (Guilford 1967)

pub mod alternate_uses;
pub mod alternate_uses_divergent;
pub mod remote_associates;

pub use alternate_uses::AlternateUsesBenchmark;
pub use alternate_uses_divergent::DivergentThinkingBenchmark;
pub use remote_associates::RemoteAssociatesBenchmark;
