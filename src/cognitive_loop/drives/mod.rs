// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Drive subsystems: emotion contagion, curiosity, and self-reflection.

mod curiosity_drive;
mod emotion_contagion;
mod self_reflection;

pub(crate) use curiosity_drive::{CuriosityDrive, ExplorationUpdate};
pub(crate) use emotion_contagion::EmotionContagion;
pub(crate) use self_reflection::{
    AdjustmentDirection, Recommendation, RecommendationTarget, ReflectionSummary,
    ReflectionThresholds, SelfAssessment, SelfReflection,
};
