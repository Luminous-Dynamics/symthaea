// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Location & Biome Context — Providing local environmental signals for the curriculum.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BiomeContext {
    pub hardiness_zone: u8,
    pub terrain: String,
    pub annual_rainfall_mm: u32,
    pub current_season: String,
}

impl Default for BiomeContext {
    fn default() -> Self {
        Self {
            hardiness_zone: 9, // Default to a temperate/sub-tropical baseline
            terrain: "Urban/Warehouse".to_string(),
            annual_rainfall_mm: 600,
            current_season: "Spring".to_string(),
        }
    }
}

/// Provide Biome context to the application.
pub fn provide_biome_context() {
    let context = signal(BiomeContext::default());
    provide_context(context.0); // ReadSignal
    provide_context(context.1); // WriteSignal
}

pub fn use_biome() -> ReadSignal<BiomeContext> {
    expect_context::<ReadSignal<BiomeContext>>()
}

pub fn use_set_biome() -> WriteSignal<BiomeContext> {
    expect_context::<WriteSignal<BiomeContext>>()
}
