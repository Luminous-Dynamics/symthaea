//! Language processing domain benchmarks.
//!
//! - **GardenPath** — Garden-path sentence processing (disambiguation cost)
//! - **SemanticCoherence** — Topic coherence, drift, and recovery in generated text

pub mod garden_path;
pub mod semantic_coherence;

pub use garden_path::GardenPathBenchmark;
pub use semantic_coherence::SemanticCoherenceBenchmark;
