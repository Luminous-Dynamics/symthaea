//! Metacognition benchmarks.
//!
//! - **Calibration** — Tests "knowing what you know": whether confidence
//!   covaries with actual accuracy across difficulty levels.
//! - **FeelingOfKnowing** — Tests metamemory: after failed recall, can the
//!   system predict whether it would recognize the correct answer?

pub mod calibration;
pub mod feeling_of_knowing;

pub use calibration::MetacognitiveCalibrationBenchmark;
pub use feeling_of_knowing::FeelingOfKnowingBenchmark;
