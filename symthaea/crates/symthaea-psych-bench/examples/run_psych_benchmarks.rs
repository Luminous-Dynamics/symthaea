//! Run the full psychological benchmark suite and print results.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --json
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --csv
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --json-output /tmp/bench.json

use std::path::PathBuf;
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
    let args: Vec<String> = std::env::args().collect();
    let output_json = args.iter().any(|a| a == "--json");
    let output_csv = args.iter().any(|a| a == "--csv");
    let json_output_path: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--json-output")
        .map(|w| PathBuf::from(&w[1]));
    let filter: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--filter")
        .map(|w| w[1].to_lowercase());

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
        if let Some(ref f) = filter {
            if !bench.name().to_lowercase().contains(f) {
                continue;
            }
        }
        eprint!("  {} ... ", bench.name());
        let result = bench.run(&config);
        eprintln!("{}ms ({} metrics)", result.elapsed_ms, result.metrics.len());
        report.add(result);
    }

    // Write JSON to file if --json-output was specified
    if let Some(path) = &json_output_path {
        report
            .to_json_file(path)
            .expect("failed to write JSON output file");
        eprintln!("JSON written to {}", path.display());
    }

    if output_json {
        println!("{}", report.to_json().expect("JSON serialization"));
    } else if output_csv {
        println!("{}", report.to_csv().expect("CSV serialization"));
    } else {
        println!("\n{}", report.summary());
    }
}
