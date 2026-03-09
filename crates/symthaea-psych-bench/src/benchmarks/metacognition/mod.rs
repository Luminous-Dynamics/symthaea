//! Metacognition benchmarks.
//!
//! - **Calibration** — Tests "knowing what you know": whether confidence
//!   covaries with actual accuracy across difficulty levels.
//! - **FeelingOfKnowing** — Tests metamemory: after failed recall, can the
//!   system predict whether it would recognize the correct answer?
//! - **ChangeBlindness** — Tests attention-dependent conscious perception:
//!   large scene changes go undetected when accompanied by disruption.

pub mod calibration;
pub mod change_blindness;
pub mod feeling_of_knowing;

pub use calibration::MetacognitiveCalibrationBenchmark;
pub use change_blindness::ChangeBlindnessBenchmark;
pub use feeling_of_knowing::FeelingOfKnowingBenchmark;
