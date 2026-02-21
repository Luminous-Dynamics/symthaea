//! Creativity domain benchmarks.
//!
//! - **RemoteAssociates** — Find a word connecting three cue words (RAT)
//! - **AlternateUses** — Generate novel uses for common objects (AUT)

pub mod alternate_uses;
pub mod remote_associates;

pub use alternate_uses::AlternateUsesBenchmark;
pub use remote_associates::RemoteAssociatesBenchmark;
