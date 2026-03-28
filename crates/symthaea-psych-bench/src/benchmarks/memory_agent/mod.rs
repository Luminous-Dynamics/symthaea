//! MemoryAgentBench: Full memory pipeline benchmark.
//!
//! Tests WM → eviction → graduation → episodic → semantic → database
//! pipeline using `ContinuousMind` and `CognitiveLoopService`.

pub mod accurate_retrieval;
pub mod conflict_resolution;
pub mod long_range;
pub mod prospective_memory;
pub mod test_time_learning;

pub use accurate_retrieval::AccurateRetrievalBenchmark;
pub use conflict_resolution::ConflictResolutionBenchmark;
pub use long_range::LongRangeBenchmark;
pub use prospective_memory::ProspectiveMemoryBenchmark;
pub use test_time_learning::TestTimeLearningBenchmark;
