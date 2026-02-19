//! Run the full psychological benchmark suite and print results.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --json
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --csv

use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, TemporalDiscountingBenchmark,
    TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, NBackBenchmark, SerialRecallBenchmark,
    SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::{BenchmarkConfig, BenchmarkReport, PsychBenchmark};

fn main() {
    let output_json = std::env::args().any(|a| a == "--json");
    let output_csv = std::env::args().any(|a| a == "--csv");

    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };

    let mut report = BenchmarkReport::new();

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        // WorM
        Box::new(NBackBenchmark),
        Box::new(ChangeDetectionBenchmark),
        Box::new(SerialRecallBenchmark),
        Box::new(SpatialUpdatingBenchmark),
        Box::new(BindingBenchmark),
        // CogBench
        Box::new(ProbabilisticReasoningBenchmark),
        Box::new(HorizonBenchmark),
        Box::new(RestlessBanditBenchmark),
        Box::new(InstrumentalLearningBenchmark),
        Box::new(TwoStepBenchmark),
        Box::new(TemporalDiscountingBenchmark),
        Box::new(BartBenchmark),
        // Butlin
        Box::new(ButlinIndicatorSuite),
        // ToMBench
        Box::new(FalseBeliefBenchmark),
        Box::new(FauxPasBenchmark),
        Box::new(PersuasionBenchmark),
        Box::new(StrangeStoryBenchmark),
        Box::new(HintingBenchmark),
        // MemoryAgent
        Box::new(AccurateRetrievalBenchmark),
        Box::new(TestTimeLearningBenchmark),
        Box::new(LongRangeBenchmark),
        Box::new(ConflictResolutionBenchmark),
    ];

    eprintln!("Running {} benchmarks...", benchmarks.len());
    for bench in &benchmarks {
        eprint!("  {} ... ", bench.name());
        let result = bench.run(&config);
        eprintln!("{}ms ({} metrics)", result.elapsed_ms, result.metrics.len());
        report.add(result);
    }

    if output_json {
        println!("{}", report.to_json().expect("JSON serialization"));
    } else if output_csv {
        println!("{}", report.to_csv().expect("CSV serialization"));
    } else {
        println!("\n{}", report.summary());
    }
}
