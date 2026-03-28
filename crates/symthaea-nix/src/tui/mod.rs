//! Terminal UI with consciousness visualization.

pub mod app;
pub mod widgets;

pub use app::{App, ComplexityLevel, FocusPanel};
pub use widgets::{
    AlertsPanel, AlertsSnapshot, CausalExplorer, CausalLink, ConsciousnessGauge,
    ConsciousnessState, GenerationTimeline, HealthSnapshot, SystemHealth, TimelineEntry,
    WorldModelSnapshot, WorldModelView,
};
