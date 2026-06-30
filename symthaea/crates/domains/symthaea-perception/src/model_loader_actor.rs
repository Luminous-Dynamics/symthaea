// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Model Loader Actor - Async Model Loading with Actor Pattern
//!
//! This module provides async model loading through the actor model:
//! - Non-blocking model downloads from HuggingFace
//! - Status tracking via BackgroundLoader
//! - Integration with consciousness bridge (Φ feedback)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    MODEL LOADER ACTOR                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
//! │  │ Message      │───→│ Actor Loop   │───→│ ModelHub     │  │
//! │  │ Channel      │    │ (tokio)      │    │ (Download)   │  │
//! │  └──────────────┘    └──────────────┘    └──────────────┘  │
//! │                             │                               │
//! │                             ▼                               │
//! │                    ┌──────────────┐                         │
//! │                    │ Background   │                         │
//! │                    │ Loader       │                         │
//! │                    │ (Status)     │                         │
//! │                    └──────────────┘                         │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info};

use super::model_hub::{ModelHub, ModelSpec};
use super::resilience::{BackgroundLoader, ResilienceConfig};

// ============================================================================
// ACTOR MESSAGES
// ============================================================================

/// Messages that the model loader actor can receive
#[derive(Debug)]
pub enum ModelLoaderMessage {
    /// Request to load a model asynchronously
    LoadModel {
        spec: ModelSpec,
        /// Optional channel to notify when complete
        notify: Option<oneshot::Sender<ModelLoadResult>>,
    },

    /// Check if a model is loaded
    IsLoaded {
        spec: ModelSpec,
        reply: oneshot::Sender<bool>,
    },

    /// Get current loading status
    GetStatus {
        reply: oneshot::Sender<Vec<LoadingStatusSnapshot>>,
    },

    /// Preload all recommended models
    PreloadAll {
        notify: Option<oneshot::Sender<Vec<ModelLoadResult>>>,
    },

    /// Shutdown the actor
    Shutdown,
}

/// Result of a model load operation
#[derive(Debug, Clone)]
pub struct ModelLoadResult {
    pub spec: ModelSpec,
    pub success: bool,
    pub path: Option<std::path::PathBuf>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Snapshot of loading status for a model
#[derive(Debug, Clone)]
pub struct LoadingStatusSnapshot {
    pub model_name: String,
    pub progress: f32,
    pub message: String,
    pub is_complete: bool,
    pub error: Option<String>,
}

// ============================================================================
// MODEL LOADER ACTOR
// ============================================================================

/// Actor that handles async model loading
pub struct ModelLoaderActor {
    /// Model hub for downloading
    hub: ModelHub,
    /// Background loader for status tracking
    loader: BackgroundLoader,
    /// Which models are fully loaded
    loaded_models: std::collections::HashSet<ModelSpec>,
    /// Configuration
    config: ModelLoaderConfig,
}

/// Configuration for the model loader actor
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    /// Models directory path
    pub models_dir: std::path::PathBuf,
    /// Maximum concurrent downloads
    pub max_concurrent: usize,
    /// Resilience configuration
    pub resilience: ResilienceConfig,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            models_dir: std::path::PathBuf::from("models"),
            max_concurrent: 2,
            resilience: ResilienceConfig::default(),
        }
    }
}

impl ModelLoaderActor {
    /// Create a new model loader actor
    pub fn new(config: ModelLoaderConfig) -> anyhow::Result<Self> {
        let hub = ModelHub::with_dir(config.models_dir.clone())?;
        let loader = BackgroundLoader::new(config.resilience.clone());

        // Check which models are already downloaded
        let mut loaded_models = std::collections::HashSet::new();
        for spec in [
            ModelSpec::Qwen3Embedding,
            ModelSpec::SigLIP,
            ModelSpec::Moondream,
            ModelSpec::BGE,
        ] {
            if hub.is_downloaded(spec) {
                loaded_models.insert(spec);
            }
        }

        Ok(Self {
            hub,
            loader,
            loaded_models,
            config,
        })
    }

    /// Get a sender channel for this actor
    pub fn spawn(config: ModelLoaderConfig) -> anyhow::Result<mpsc::Sender<ModelLoaderMessage>> {
        let actor = Self::new(config)?;
        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            actor.run(rx).await;
        });

        Ok(tx)
    }

    /// Run the actor's message loop
    async fn run(mut self, mut rx: mpsc::Receiver<ModelLoaderMessage>) {
        info!("ModelLoaderActor: Starting");

        while let Some(msg) = rx.recv().await {
            match msg {
                ModelLoaderMessage::LoadModel { spec, notify } => {
                    self.handle_load_model(spec, notify).await;
                }

                ModelLoaderMessage::IsLoaded { spec, reply } => {
                    let is_loaded = self.loaded_models.contains(&spec);
                    let _ = reply.send(is_loaded);
                }

                ModelLoaderMessage::GetStatus { reply } => {
                    let statuses = self.get_all_statuses();
                    let _ = reply.send(statuses);
                }

                ModelLoaderMessage::PreloadAll { notify } => {
                    self.handle_preload_all(notify).await;
                }

                ModelLoaderMessage::Shutdown => {
                    info!("ModelLoaderActor: Shutting down");
                    break;
                }
            }
        }

        info!("ModelLoaderActor: Stopped");
    }

    /// Handle a model load request
    async fn handle_load_model(
        &mut self,
        spec: ModelSpec,
        notify: Option<oneshot::Sender<ModelLoadResult>>,
    ) {
        let start = std::time::Instant::now();

        // Check if already loaded
        if self.loaded_models.contains(&spec) {
            debug!("Model {:?} already loaded", spec);
            if let Some(tx) = notify {
                let _ = tx.send(ModelLoadResult {
                    spec,
                    success: true,
                    path: Some(self.hub.model_path(spec)),
                    error: None,
                    duration_ms: 0,
                });
            }
            return;
        }

        // Start tracking
        let model_name = spec.local_name().to_string();
        self.loader.start_loading(&model_name);

        // Perform the download
        info!("Loading model {:?}...", spec);
        self.loader
            .update_progress(&model_name, 0.1, "Starting download...");

        #[cfg(feature = "vision")]
        let result = {
            match self.hub.ensure_model(spec) {
                Ok(path) => {
                    self.loader
                        .update_progress(&model_name, 0.9, "Validating...");
                    self.loader.complete_loading(&model_name);
                    self.loaded_models.insert(spec);

                    ModelLoadResult {
                        spec,
                        success: true,
                        path: Some(path),
                        error: None,
                        duration_ms: start.elapsed().as_millis() as u64,
                    }
                }
                Err(e) => {
                    let error_msg = format!("{:#}", e);
                    self.loader.fail_loading(&model_name, &error_msg);

                    ModelLoadResult {
                        spec,
                        success: false,
                        path: None,
                        error: Some(error_msg),
                        duration_ms: start.elapsed().as_millis() as u64,
                    }
                }
            }
        };

        #[cfg(not(feature = "vision"))]
        let result = {
            // Without vision feature, mark as unavailable
            self.loader
                .fail_loading(&model_name, "Vision feature not enabled");
            ModelLoadResult {
                spec,
                success: false,
                path: None,
                error: Some("Vision feature not enabled".to_string()),
                duration_ms: start.elapsed().as_millis() as u64,
            }
        };

        info!(
            "Model {:?} load completed: success={}, duration={}ms",
            spec, result.success, result.duration_ms
        );

        if let Some(tx) = notify {
            let _ = tx.send(result);
        }
    }

    /// Handle preload all request
    async fn handle_preload_all(&mut self, notify: Option<oneshot::Sender<Vec<ModelLoadResult>>>) {
        let specs_to_load: Vec<ModelSpec> = [
            ModelSpec::Qwen3Embedding,
            ModelSpec::SigLIP,
            ModelSpec::Moondream,
        ]
        .into_iter()
        .filter(|spec| !self.loaded_models.contains(spec))
        .collect();

        info!("Preloading {} models...", specs_to_load.len());

        let mut results = Vec::new();
        for spec in specs_to_load {
            let (tx, rx) = oneshot::channel();
            self.handle_load_model(spec, Some(tx)).await;
            if let Ok(result) = rx.await {
                results.push(result);
            }
        }

        if let Some(tx) = notify {
            let _ = tx.send(results);
        }
    }

    /// Get all loading statuses as snapshots
    fn get_all_statuses(&self) -> Vec<LoadingStatusSnapshot> {
        self.loader
            .all_statuses()
            .into_iter()
            .map(|status| LoadingStatusSnapshot {
                model_name: status.model.clone(),
                progress: status.progress,
                message: status.message.clone(),
                is_complete: status.complete,
                error: status.error.clone(),
            })
            .collect()
    }
}

// ============================================================================
// CLIENT HANDLE
// ============================================================================

/// Handle for interacting with the model loader actor
#[derive(Clone)]
pub struct ModelLoaderHandle {
    sender: mpsc::Sender<ModelLoaderMessage>,
}

impl ModelLoaderHandle {
    /// Create a new handle from an existing sender
    pub fn new(sender: mpsc::Sender<ModelLoaderMessage>) -> Self {
        Self { sender }
    }

    /// Start a new model loader actor and return a handle
    pub fn spawn(config: ModelLoaderConfig) -> anyhow::Result<Self> {
        let sender = ModelLoaderActor::spawn(config)?;
        Ok(Self { sender })
    }

    /// Request loading a model (fire-and-forget)
    pub async fn load_model(&self, spec: ModelSpec) -> anyhow::Result<()> {
        self.sender
            .send(ModelLoaderMessage::LoadModel { spec, notify: None })
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))
    }

    /// Request loading a model and wait for result
    pub async fn load_model_and_wait(&self, spec: ModelSpec) -> anyhow::Result<ModelLoadResult> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ModelLoaderMessage::LoadModel {
                spec,
                notify: Some(tx),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Actor dropped reply channel"))
    }

    /// Check if a model is loaded
    pub async fn is_loaded(&self, spec: ModelSpec) -> anyhow::Result<bool> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ModelLoaderMessage::IsLoaded { spec, reply: tx })
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Actor dropped reply channel"))
    }

    /// Get current loading statuses
    pub async fn get_statuses(&self) -> anyhow::Result<Vec<LoadingStatusSnapshot>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ModelLoaderMessage::GetStatus { reply: tx })
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Actor dropped reply channel"))
    }

    /// Preload all models (fire-and-forget)
    pub async fn preload_all(&self) -> anyhow::Result<()> {
        self.sender
            .send(ModelLoaderMessage::PreloadAll { notify: None })
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))
    }

    /// Preload all models and wait for results
    pub async fn preload_all_and_wait(&self) -> anyhow::Result<Vec<ModelLoadResult>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ModelLoaderMessage::PreloadAll { notify: Some(tx) })
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Actor dropped reply channel"))
    }

    /// Shutdown the actor
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        self.sender
            .send(ModelLoaderMessage::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("Actor channel closed"))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ModelLoaderConfig {
            models_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let handle = ModelLoaderHandle::spawn(config).unwrap();

        // Should start successfully
        let statuses = handle.get_statuses().await.unwrap();
        assert!(statuses.is_empty()); // No loads in progress

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_is_loaded_for_missing_model() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ModelLoaderConfig {
            models_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let handle = ModelLoaderHandle::spawn(config).unwrap();

        // Model shouldn't be loaded (directory is empty)
        let is_loaded = handle.is_loaded(ModelSpec::Qwen3Embedding).await.unwrap();
        assert!(!is_loaded);

        handle.shutdown().await.unwrap();
    }
}
