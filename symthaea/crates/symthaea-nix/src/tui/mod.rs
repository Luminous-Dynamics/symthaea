//! Terminal UI with consciousness visualization.

pub mod app;
pub mod widgets;

pub use app::{App, FocusPanel};
pub use widgets::{
    CausalExplorer, CausalLink, ConsciousnessGauge, ConsciousnessState, GenerationTimeline,
    HealthSnapshot, SystemHealth, TimelineEntry, WorldModelSnapshot, WorldModelView,
};
