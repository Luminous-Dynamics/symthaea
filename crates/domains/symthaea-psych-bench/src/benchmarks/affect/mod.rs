// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Affect domain benchmarks.
//!
//! - **ValenceClassification** — Classify stimuli into positive/negative/neutral valence
//! - **MoodCongruentRecall** — Test mood-congruent memory bias
//! - **EmotionalStroop** — Emotional interference on color naming

pub mod emotional_stroop;
pub mod mood_congruent_recall;
pub mod valence_classification;

pub use emotional_stroop::EmotionalStroopBenchmark;
pub use mood_congruent_recall::MoodCongruentRecallBenchmark;
pub use valence_classification::ValenceClassificationBenchmark;
