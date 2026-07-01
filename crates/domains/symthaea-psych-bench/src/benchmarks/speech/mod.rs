// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
pub mod categorical_perception;
pub mod phoneme_discrimination;
pub mod vot_continuum;

pub use categorical_perception::CategoricalPerceptionBenchmark;
pub use phoneme_discrimination::PhonemeDiscriminationBenchmark;
pub use vot_continuum::VotContinuumBenchmark;
