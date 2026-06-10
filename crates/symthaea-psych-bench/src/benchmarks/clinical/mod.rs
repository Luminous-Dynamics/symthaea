// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Clinical psychology benchmarks.
//!
//! Evaluates therapeutic capabilities against established clinical measures:
//! - Empathic accuracy (Ickes 1993)
//! - Therapeutic response appropriateness (Hill 2009)
//! - Alliance maintenance and repair (Safran & Muran 2000)
//! - Crisis detection sensitivity/specificity (C-SSRS)
//! - Cognitive distortion identification (Burns 1980)
//! - Motivational interviewing fidelity (MITI coding)

pub mod alliance_maintenance;
pub mod cognitive_distortion;
pub mod crisis_detection;
pub mod empathic_accuracy;
pub mod motivational_interviewing;
pub mod therapeutic_response;

pub use alliance_maintenance::AllianceMaintenanceBenchmark;
pub use cognitive_distortion::CognitiveDistortionBenchmark;
pub use crisis_detection::CrisisDetectionBenchmark;
pub use empathic_accuracy::EmpathicAccuracyBenchmark;
pub use motivational_interviewing::MotivationalInterviewingBenchmark;
pub use therapeutic_response::TherapeuticResponseBenchmark;
