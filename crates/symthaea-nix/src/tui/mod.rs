// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Terminal UI with consciousness visualization.

pub mod app;
pub mod widgets;

pub use app::{App, ComplexityLevel, FocusPanel};
pub use widgets::{
    AlertsPanel, AlertsSnapshot, CausalExplorer, CausalLink, ConsciousnessGauge,
    ConsciousnessState, GenerationTimeline, HealthSnapshot, SystemHealth, TimelineEntry,
    WorldModelSnapshot, WorldModelView,
};
