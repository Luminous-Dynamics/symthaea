//! Metacognition benchmarks.
//!
//! - **Calibration** — Tests "knowing what you know": whether confidence
//!   covaries with actual accuracy across difficulty levels.

pub mod calibration;

pub use calibration::MetacognitiveCalibrationBenchmark;
