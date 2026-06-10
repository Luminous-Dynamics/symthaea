// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Model Status Registry
//!
//! Provides non-panicking model loading with status tracking.
//! Ensures the perception system gracefully degrades when models
//! are unavailable rather than crashing.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// Status of a model in the registry
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// Model has not been attempted to load yet
    NotAttempted,
    /// Model is currently being loaded
    Loading,
    /// Model loaded successfully at given path
    Loaded(PathBuf),
    /// Model is unavailable (with reason)
    Unavailable(String),
    /// Model loaded but with degraded functionality
    Degraded(String),
}

impl ModelStatus {
    /// Returns true if the model can produce output (Loaded or Degraded)
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Loaded(_) | Self::Degraded(_))
    }

    /// Returns true only if the model is fully loaded
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded(_))
    }

    /// Returns the path if the model is loaded, None otherwise
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            Self::Loaded(p) => Some(p),
            _ => None,
        }
    }
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAttempted => write!(f, "Not attempted"),
            Self::Loading => write!(f, "Loading..."),
            Self::Loaded(p) => write!(f, "Loaded ({})", p.display()),
            Self::Unavailable(reason) => write!(f, "Unavailable: {reason}"),
            Self::Degraded(reason) => write!(f, "Degraded: {reason}"),
        }
    }
}

/// Errors that can occur during model loading (typed, never panics)
#[derive(Debug, Clone)]
pub enum ModelLoadError {
    /// Model file was not found at expected path
    FileNotFound(String),
    /// Model file exists but is corrupt or unreadable
    CorruptModel(String),
    /// A required dependency (runtime, library) is missing
    DependencyMissing(String),
    /// A required compile-time feature is not enabled
    FeatureDisabled(String),
    /// Network error during download or verification
    NetworkError(String),
    /// Any other error
    Other(String),
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "Model file not found: {p}"),
            Self::CorruptModel(r) => write!(f, "Corrupt model: {r}"),
            Self::DependencyMissing(d) => write!(f, "Missing dependency: {d}"),
            Self::FeatureDisabled(feat) => write!(f, "Feature '{feat}' not enabled"),
            Self::NetworkError(e) => write!(f, "Network error: {e}"),
            Self::Other(e) => write!(f, "Model load error: {e}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

// Sentinel value returned when a model name is not found in the registry.
// This avoids the need to return Option from status().
static NOT_ATTEMPTED: ModelStatus = ModelStatus::NotAttempted;

/// Registry tracking the status of all models.
///
/// Provides a central place to record which models have been attempted,
/// which succeeded, and which failed -- without ever panicking.
pub struct ModelRegistry {
    statuses: HashMap<String, ModelStatus>,
}

impl ModelRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            statuses: HashMap::new(),
        }
    }

    /// Try to load a model, returning Result instead of panicking.
    ///
    /// The loader closure is called to perform the actual loading. Its
    /// result is recorded in the registry regardless of success or failure.
    pub fn try_load(
        &mut self,
        name: &str,
        loader: impl FnOnce() -> Result<PathBuf, ModelLoadError>,
    ) -> Result<PathBuf, ModelLoadError> {
        self.statuses.insert(name.to_string(), ModelStatus::Loading);
        match loader() {
            Ok(path) => {
                self.statuses
                    .insert(name.to_string(), ModelStatus::Loaded(path.clone()));
                Ok(path)
            }
            Err(e) => {
                self.statuses
                    .insert(name.to_string(), ModelStatus::Unavailable(e.to_string()));
                Err(e)
            }
        }
    }

    /// Get the status of a model by name.
    ///
    /// Returns `ModelStatus::NotAttempted` if the model has never been
    /// registered.
    pub fn status(&self, name: &str) -> &ModelStatus {
        self.statuses.get(name).unwrap_or(&NOT_ATTEMPTED)
    }

    /// Set the status of a model directly (e.g. to mark it as Degraded).
    pub fn set_status(&mut self, name: &str, status: ModelStatus) {
        self.statuses.insert(name.to_string(), status);
    }

    /// Human-readable summary of all tracked models.
    pub fn summary(&self) -> String {
        if self.statuses.is_empty() {
            return "ModelRegistry: (empty -- no models tracked)".to_string();
        }

        let mut lines = vec![format!(
            "ModelRegistry: {}/{} available",
            self.available_count(),
            self.total_count()
        )];

        // Sort by name for deterministic output
        let mut entries: Vec<_> = self.statuses.iter().collect();
        entries.sort_by_key(|(name, _)| (*name).clone());

        for (name, status) in entries {
            lines.push(format!("  - {name}: {status}"));
        }

        lines.join("\n")
    }

    /// Number of models that are available (Loaded or Degraded).
    pub fn available_count(&self) -> usize {
        self.statuses.values().filter(|s| s.is_available()).count()
    }

    /// Total number of models tracked in the registry.
    pub fn total_count(&self) -> usize {
        self.statuses.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ModelStatus tests ----

    #[test]
    fn test_status_not_attempted() {
        let status = ModelStatus::NotAttempted;
        assert!(!status.is_available());
        assert!(!status.is_loaded());
        assert!(status.path().is_none());
        assert_eq!(format!("{}", status), "Not attempted");
    }

    #[test]
    fn test_status_loading() {
        let status = ModelStatus::Loading;
        assert!(!status.is_available());
        assert!(!status.is_loaded());
        assert!(status.path().is_none());
        assert_eq!(format!("{}", status), "Loading...");
    }

    #[test]
    fn test_status_loaded() {
        let path = PathBuf::from("/models/qwen3");
        let status = ModelStatus::Loaded(path.clone());
        assert!(status.is_available());
        assert!(status.is_loaded());
        assert_eq!(status.path(), Some(&path));
        assert!(format!("{}", status).contains("/models/qwen3"));
    }

    #[test]
    fn test_status_unavailable() {
        let status = ModelStatus::Unavailable("file missing".to_string());
        assert!(!status.is_available());
        assert!(!status.is_loaded());
        assert!(status.path().is_none());
        assert!(format!("{}", status).contains("file missing"));
    }

    #[test]
    fn test_status_degraded() {
        let status = ModelStatus::Degraded("quantized fallback".to_string());
        assert!(status.is_available());
        assert!(!status.is_loaded());
        assert!(status.path().is_none());
        assert!(format!("{}", status).contains("quantized fallback"));
    }

    // ---- ModelLoadError tests ----

    #[test]
    fn test_error_display() {
        let err = ModelLoadError::FileNotFound("/models/siglip".to_string());
        assert!(format!("{}", err).contains("/models/siglip"));

        let err = ModelLoadError::CorruptModel("bad header".to_string());
        assert!(format!("{}", err).contains("bad header"));

        let err = ModelLoadError::DependencyMissing("ort".to_string());
        assert!(format!("{}", err).contains("ort"));

        let err = ModelLoadError::FeatureDisabled("vision".to_string());
        assert!(format!("{}", err).contains("vision"));

        let err = ModelLoadError::NetworkError("timeout".to_string());
        assert!(format!("{}", err).contains("timeout"));

        let err = ModelLoadError::Other("unknown".to_string());
        assert!(format!("{}", err).contains("unknown"));
    }

    #[test]
    fn test_error_is_std_error() {
        // Verify ModelLoadError implements std::error::Error
        let err: Box<dyn std::error::Error> = Box::new(ModelLoadError::Other("test".to_string()));
        assert!(format!("{}", err).contains("test"));
    }

    // ---- ModelRegistry tests ----

    #[test]
    fn test_registry_new_is_empty() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.total_count(), 0);
        assert_eq!(registry.available_count(), 0);
    }

    #[test]
    fn test_registry_default_is_empty() {
        let registry = ModelRegistry::default();
        assert_eq!(registry.total_count(), 0);
        assert_eq!(registry.available_count(), 0);
    }

    #[test]
    fn test_status_of_unknown_model() {
        let registry = ModelRegistry::new();
        let status = registry.status("nonexistent");
        assert_eq!(*status, ModelStatus::NotAttempted);
    }

    #[test]
    fn test_successful_load_transitions() {
        let mut registry = ModelRegistry::new();
        let path = PathBuf::from("/models/qwen3-embedding");

        // Before load: NotAttempted
        assert_eq!(*registry.status("qwen3"), ModelStatus::NotAttempted);

        // Load succeeds
        let result = registry.try_load("qwen3", || Ok(path.clone()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), path);

        // After load: Loaded
        assert!(registry.status("qwen3").is_loaded());
        assert_eq!(registry.status("qwen3").path(), Some(&path));
        assert_eq!(registry.available_count(), 1);
        assert_eq!(registry.total_count(), 1);
    }

    #[test]
    fn test_failed_load_transitions() {
        let mut registry = ModelRegistry::new();

        // Load fails
        let result = registry.try_load("siglip", || {
            Err(ModelLoadError::FileNotFound(
                "/models/siglip not found".to_string(),
            ))
        });
        assert!(result.is_err());

        // After failed load: Unavailable
        let status = registry.status("siglip");
        assert!(!status.is_available());
        assert!(!status.is_loaded());
        assert!(matches!(status, ModelStatus::Unavailable(_)));
        assert_eq!(registry.available_count(), 0);
        assert_eq!(registry.total_count(), 1);
    }

    #[test]
    fn test_failure_never_panics() {
        let mut registry = ModelRegistry::new();

        // Simulate multiple failures -- none should panic
        for i in 0..10 {
            let name = format!("model-{}", i);
            let result =
                registry.try_load(&name, || Err(ModelLoadError::Other(format!("error {}", i))));
            assert!(result.is_err());
        }

        // Registry should have tracked all 10 failures
        assert_eq!(registry.total_count(), 10);
        assert_eq!(registry.available_count(), 0);

        // All should be Unavailable
        for i in 0..10 {
            let name = format!("model-{}", i);
            assert!(matches!(
                registry.status(&name),
                ModelStatus::Unavailable(_)
            ));
        }
    }

    #[test]
    fn test_loader_panic_safety() {
        // Even if a loader panics, the registry itself is safe.
        // (The panic propagates but doesn't corrupt the registry.)
        let mut registry = ModelRegistry::new();

        // First, load a good model
        let _ = registry.try_load("good", || Ok(PathBuf::from("/models/good")));
        assert_eq!(registry.available_count(), 1);

        // Panic in loader will propagate, but catch it for the test
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            registry.try_load("bad", || panic!("simulated panic"))
        }));
        assert!(result.is_err()); // panic was caught

        // The "good" model is still loaded
        assert!(registry.status("good").is_loaded());
        // "bad" was set to Loading before the panic, so it may be Loading
        // This is acceptable -- the panic interrupted the state machine
    }

    #[test]
    fn test_degraded_operation() {
        let mut registry = ModelRegistry::new();

        // Load a model, then mark it degraded
        let _ = registry.try_load("qwen3", || Ok(PathBuf::from("/models/qwen3")));
        assert!(registry.status("qwen3").is_loaded());

        registry.set_status(
            "qwen3",
            ModelStatus::Degraded("quantized to int8".to_string()),
        );
        let status = registry.status("qwen3");
        assert!(status.is_available()); // Degraded is still available
        assert!(!status.is_loaded()); // But not fully loaded
        assert!(matches!(status, ModelStatus::Degraded(_)));
    }

    #[test]
    fn test_reload_after_failure() {
        let mut registry = ModelRegistry::new();

        // First attempt fails
        let _ = registry.try_load("siglip", || {
            Err(ModelLoadError::NetworkError("timeout".to_string()))
        });
        assert!(!registry.status("siglip").is_available());

        // Second attempt succeeds (model was downloaded later)
        let path = PathBuf::from("/models/siglip");
        let result = registry.try_load("siglip", || Ok(path.clone()));
        assert!(result.is_ok());
        assert!(registry.status("siglip").is_loaded());
        assert_eq!(registry.available_count(), 1);
    }

    #[test]
    fn test_multiple_models() {
        let mut registry = ModelRegistry::new();

        // Load several models with mixed results
        let _ = registry.try_load("qwen3-embedding", || Ok(PathBuf::from("/models/qwen3")));
        let _ = registry.try_load("siglip", || {
            Err(ModelLoadError::FeatureDisabled("vision".to_string()))
        });
        let _ = registry.try_load("moondream", || {
            Err(ModelLoadError::FileNotFound("not downloaded".to_string()))
        });
        let _ = registry.try_load("bge", || Ok(PathBuf::from("/models/bge")));

        assert_eq!(registry.total_count(), 4);
        assert_eq!(registry.available_count(), 2);
    }

    #[test]
    fn test_summary_formatting() {
        let mut registry = ModelRegistry::new();

        let _ = registry.try_load("qwen3-embedding", || Ok(PathBuf::from("/models/qwen3")));
        let _ = registry.try_load("siglip", || {
            Err(ModelLoadError::FeatureDisabled("vision".to_string()))
        });

        let summary = registry.summary();
        assert!(summary.contains("1/2 available"));
        assert!(summary.contains("qwen3-embedding"));
        assert!(summary.contains("siglip"));
        assert!(summary.contains("Loaded"));
        assert!(summary.contains("Unavailable"));
    }

    #[test]
    fn test_summary_empty_registry() {
        let registry = ModelRegistry::new();
        let summary = registry.summary();
        assert!(summary.contains("empty"));
    }

    #[test]
    fn test_counts_with_mixed_statuses() {
        let mut registry = ModelRegistry::new();

        // Manually set various statuses to test counting
        registry.set_status("a", ModelStatus::Loaded(PathBuf::from("/a")));
        registry.set_status("b", ModelStatus::Degraded("low quality".to_string()));
        registry.set_status("c", ModelStatus::Unavailable("missing".to_string()));
        registry.set_status("d", ModelStatus::Loading);
        registry.set_status("e", ModelStatus::NotAttempted);

        assert_eq!(registry.total_count(), 5);
        // Only Loaded and Degraded count as available
        assert_eq!(registry.available_count(), 2);
    }

    #[test]
    fn test_feature_disabled_error_classification() {
        let err = ModelLoadError::FeatureDisabled("vision".to_string());
        let display = format!("{}", err);
        assert!(display.contains("vision"));
        assert!(display.contains("not enabled"));
    }
}
