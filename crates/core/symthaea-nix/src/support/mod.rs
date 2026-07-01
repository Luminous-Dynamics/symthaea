// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
pub mod poml;
pub mod predictive;
pub mod scrubber;
pub mod watchdog;

pub use assessment::{SupportAssessment, SupportAssessor, SupportRecommendation};
pub use health_check::{HealthAssessor, HealthCheck, HealthStatus};
pub use knowledge::{
    AnyKnowledgeMatch, DynamicKnowledgeArticle, KnowledgeArticle, KnowledgeBase, KnowledgeMatch,
};
pub use poml::{
    CacheSettings, ModelHints, PomlContext, PomlFeature, PomlMetadata, PomlProcessor, PomlResult,
    PomlValue,
};
pub use predictive::{
    AlertThresholds, Prediction, PredictiveMonitor, SavedPredictiveState, SystemTelemetry,
};
pub use scrubber::{ScrubResult, Scrubber};
pub use watchdog::{AutonomyLevel, Watchdog, WatchdogConfig, WatchdogVerdict};
