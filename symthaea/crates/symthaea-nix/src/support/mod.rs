//! Proactive NixOS Support Modules
//!
//! Extends symthaea-nix with consciousness-aware health monitoring,
//! predictive failure detection, post-rebuild watchdog, privacy scrubbing,
//! and a curated NixOS knowledge corpus with HDC similarity search.
//!
//! All modules reuse the existing observers, encoders, mind, and action layers.

pub mod assessment;
pub mod health_check;
pub mod knowledge;
pub mod predictive;
pub mod scrubber;
pub mod watchdog;

pub use assessment::{SupportAssessment, SupportAssessor, SupportRecommendation};
pub use health_check::{HealthAssessor, HealthCheck, HealthStatus};
pub use knowledge::{
    AnyKnowledgeMatch, DynamicKnowledgeArticle, KnowledgeArticle, KnowledgeBase, KnowledgeMatch,
};
pub use predictive::{
    AlertThresholds, Prediction, PredictiveMonitor, SavedPredictiveState, SystemTelemetry,
};
pub use scrubber::{ScrubResult, Scrubber};
pub use watchdog::{AutonomyLevel, Watchdog, WatchdogConfig, WatchdogVerdict};
