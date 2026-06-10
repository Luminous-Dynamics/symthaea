// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TUI Widgets for NixOS Consciousness Visualization

pub mod alerts_panel;
pub mod causal_explorer;
pub mod consciousness_gauge;
pub mod generation_timeline;
pub mod system_health;
pub mod world_model_view;

pub use alerts_panel::{AlertsPanel, AlertsSnapshot};
pub use causal_explorer::{CausalExplorer, CausalLink};
pub use consciousness_gauge::{ConsciousnessGauge, ConsciousnessState};
pub use generation_timeline::{GenerationTimeline, TimelineEntry};
pub use system_health::{HealthSnapshot, SystemHealth};
pub use world_model_view::{WorldModelSnapshot, WorldModelView};
