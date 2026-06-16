// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psych-bench → cognitive loop calibration bridge.
//!
//! Translates normative z-scores from the psych-bench suite into neuromodulator
//! sensitivity adjustments and feedback variable proposals. This closes the loop
//! between *measuring* cognitive performance and *tuning* the system.
//!
//! ## Architecture (9-transmitter coverage)
//!
//! ```text
//! NormativeReport (z-scores)
//!   ├── Stroop/Flanker z       → DA receptor_sensitivity
//!   ├── N-back/DigitSpan z     → ACh receptor_sensitivity
//!   ├── StopSignal/GoNoGo z    → NE-β (phasic), NE-α (tonic via WM)
//!   ├── CPT/SART z             → 5-HT receptor_sensitivity
//!   ├── DualTask z             → GABA receptor_sensitivity
//!   ├── UltimatumGame/RME z    → Oxytocin receptor_sensitivity
//!   ├── PVT lapse_rate z       → Glutamate receptor_sensitivity
//!   ├── PVT vigilance_decrement → Adenosine receptor_sensitivity
//!   ├── Allostatic load proxy  → Endocannabinoid receptor_sensitivity
//!   └── FoK calibration        → confidence proposal
//! ```
//!
//! ## Sub-modules
//!
//! - [`normative`]: z-score → adjustment pipeline (types, builders, application)
//! - [`monitor`]: Autonomous self-assessment metacognitive monitor
//! - [`history`]: Calibration history and systematic drift tracking

pub mod history;
pub mod monitor;
pub mod normative;

// Re-export primary types for backward compatibility
pub use history::{CalibrationHistory, CalibrationSnapshot, CalibrationValidator};
pub use monitor::{SelfAssessmentInput, SelfAssessmentMonitor};
pub use normative::{
    NeuromodCalibration, ReceptorSubtype, SharedCalibrationProfile, TransmitterCalibration,
};

// Re-export test-only helpers for the tests submodule
#[cfg(test)]
pub(crate) use normative::{is_lower_better_metric, z_to_sensitivity_factor};

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
#[path = "tests.rs"]
mod tests;
