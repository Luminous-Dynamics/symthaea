//! TUI Widgets for NixOS Consciousness Visualization

pub mod consciousness_gauge;
pub mod causal_explorer;
pub mod generation_timeline;
pub mod system_health;
pub mod world_model_view;

pub use consciousness_gauge::{ConsciousnessGauge, ConsciousnessState};
pub use causal_explorer::{CausalExplorer, CausalLink};
pub use generation_timeline::{GenerationTimeline, TimelineEntry};
pub use system_health::{SystemHealth, HealthSnapshot};
pub use world_model_view::{WorldModelView, WorldModelSnapshot};
