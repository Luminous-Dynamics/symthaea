// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness domain benchmarks.
//!
//! - **Blindsight** — Dissociation between forced-choice accuracy and subjective
//!   awareness report (Weiskrantz, 1986).
//! - **BinocularRivalry** — Perceptual alternation dynamics (Levelt, 1965).
//! - **PerceptualCrowding** — Flanker-induced target identification loss
//!   (Whitney & Levi, 2011).

pub mod binocular_rivalry;
pub mod blindsight;
pub mod perceptual_crowding;

pub use binocular_rivalry::BinocularRivalryBenchmark;
pub use blindsight::BlindSightBenchmark;
pub use perceptual_crowding::PerceptualCrowdingBenchmark;
