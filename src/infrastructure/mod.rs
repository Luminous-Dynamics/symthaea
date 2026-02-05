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

pub mod cache;
pub mod file_watcher;
pub mod pagination;
pub mod auth;
pub mod metrics;
pub mod git_tracking;
pub mod sandbox;
pub mod flake_updater;
pub mod home_manager;
pub mod lock_guard;

pub use cache::{LruCache, HdcCache, CacheStats};
pub use file_watcher::{ConfigWatcher, WatchEvent, WatchEventKind};
pub use pagination::{Paginator, Page, PageRequest};
pub use auth::{AuthToken, AuthProvider, AuthError};
pub use metrics::{MetricsCollector, MetricValue};
pub use git_tracking::{GitTracker, CommitInfo};
pub use sandbox::{Sandbox, SandboxResult};
pub use flake_updater::{FlakeUpdater, FlakeInput, UpdatePreview, UpdateResult};
pub use home_manager::{HomeManagerBridge, HomeConfig, HomeGeneration, HomeResult};
pub use lock_guard::{ResilientMutex, ResilientRwLock};
