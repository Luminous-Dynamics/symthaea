//! Benchmark harness: traits, configuration, reporting, and baselines.

pub mod analysis;
pub mod baselines;
pub mod config;
pub mod report;
pub mod snapshot;

pub use config::{AblationConfig, AblationPreset, BenchmarkConfig};
pub use report::{BaselineComparison, BenchmarkReport, BenchmarkResult, MetricValue};
pub use snapshot::{
    RegressionReport, RegressionResult, RegressionSeverity, RegressionSnapshot, RegressionSummary,
};

/// A runnable psychological benchmark.
pub trait PsychBenchmark {
    /// Human-readable benchmark name.
    fn name(&self) -> &str;

    /// Run the benchmark with given configuration.
    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult;

    /// Run ablation study with multiple configurations.
    fn run_ablation(&self, configs: &[AblationConfig]) -> Vec<BenchmarkResult> {
        configs
            .iter()
            .map(|ac| {
                let mut bc = ac.base.clone();
                bc.label = Some(ac.name.clone());
                self.run(&bc)
            })
            .collect()
    }
}
