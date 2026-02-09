//! Terminal UI with consciousness visualization.

pub mod app;
pub mod widgets;

pub use app::{App, FocusPanel};
pub use widgets::{
    ConsciousnessGauge, ConsciousnessState,
    SystemHealth, HealthSnapshot,
    GenerationTimeline, TimelineEntry,
    WorldModelView, WorldModelSnapshot,
    CausalExplorer, CausalLink,
};
