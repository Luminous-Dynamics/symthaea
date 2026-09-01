// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Infrastructure Module
//!
//! Performance and reliability infrastructure for Symthaea:
//! - F1: Lazy loading with pagination
//! - F2: LRU caching for HDC encodings
//! - F3: File watching for config changes
//! - I1: Git tracking for configs
//! - I2: Flake input updater
//! - I3: Home-manager bridge
//! - J1: Prometheus metrics
//! - K1: Socket authentication
//! - K3: Dry-run sandbox
//! - L1: Resilient lock guards (poison recovery)

pub mod auth;
pub mod cache;
pub mod file_watcher;
pub mod flake_updater;
pub mod git_tracking;
pub mod home_manager;
pub mod lock_guard;
pub mod metrics;
pub mod pagination;
pub mod sandbox;
pub mod somatic_error_bridge;
pub mod task_supervisor;
pub mod thermal_bridge;

pub use auth::{AuthError, AuthProvider, AuthToken, LocalAuth, LocalPeerIdentity};
pub use cache::{CacheStats, HdcCache, LruCache};
pub use file_watcher::{ConfigWatcher, WatchEvent, WatchEventKind};
pub use flake_updater::{FlakeInput, FlakeUpdater, UpdatePreview, UpdateResult};
pub use git_tracking::{CommitInfo, GitTracker};
pub use home_manager::{HomeConfig, HomeGeneration, HomeManagerBridge, HomeResult};
pub use lock_guard::{ResilientMutex, ResilientMutexWithPain, ResilientRwLock};
pub use metrics::{MetricValue, MetricsCollector};
pub use pagination::{Page, PageRequest, Paginator};
pub use sandbox::{Sandbox, SandboxResult};
pub use somatic_error_bridge::{
    InfrastructureError, PainSender, SomaticErrorBridge, SomaticSignals,
};
pub use task_supervisor::TaskSupervisor;
pub use thermal_bridge::{ThermalBridge, ThermalLevel, ThermalSender, ThermalSignals};
