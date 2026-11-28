// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metacognition benchmarks.
//!
//! - **Calibration** — Tests "knowing what you know": whether confidence
//!   covaries with actual accuracy across difficulty levels.
//! - **FeelingOfKnowing** — Tests metamemory: after failed recall, can the
//!   system predict whether it would recognize the correct answer?
//! - **ChangeBlindness** — Tests attention-dependent scene-change detection.
//! - **SelectivePrediction** — Scores answer, abstain and request-information
//!   policies using counterfactual correctness, calibration, risk and cost.

pub mod calibration;
pub mod change_blindness;
pub mod feeling_of_knowing;
pub mod selective_prediction;

pub use calibration::MetacognitiveCalibrationBenchmark;
pub use change_blindness::ChangeBlindnessBenchmark;
pub use feeling_of_knowing::FeelingOfKnowingBenchmark;
