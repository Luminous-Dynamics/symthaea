//! Fluid reasoning benchmarks.

pub mod arc_abductive;
pub mod arc_analogy;
pub mod arc_chain;
pub mod arc_compositional;
pub mod arc_dataset;
pub mod arc_fewshot;
pub mod arc_fluid;
pub mod arc_noise;
pub mod arc_scaling;

pub use arc_abductive::ArcAbductiveBenchmark;
pub use arc_analogy::ArcAnalogyBenchmark;
pub use arc_chain::ArcChainBenchmark;
pub use arc_compositional::ArcCompositionalBenchmark;
pub use arc_fewshot::ArcFewShotBenchmark;
pub use arc_fluid::ArcFluidBenchmark;
pub use arc_noise::ArcNoiseBenchmark;
pub use arc_scaling::ArcScalingBenchmark;
