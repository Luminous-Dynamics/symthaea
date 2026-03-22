//! Coding domain benchmarks.
//!
//! - **HumanEvalMini** -- 20 HumanEval-style function synthesis problems solved via HDC (Chen et al., 2021)
//! - **BugDetection** -- Detect and classify seeded bugs in code snippets via HDC (Beller et al., 2018)
//! - **AlgorithmRecognition** -- Classify algorithm families from code structure via HDC (Alam et al., 2022)

pub mod algorithm_recognition;
pub mod bug_detection;
pub mod humaneval_mini;

pub use algorithm_recognition::AlgorithmRecognitionBenchmark;
pub use bug_detection::BugDetectionBenchmark;
pub use humaneval_mini::HumanEvalMiniBenchmark;
