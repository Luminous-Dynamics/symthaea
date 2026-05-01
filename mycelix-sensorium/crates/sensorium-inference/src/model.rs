// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model loading pipeline for Candle WASM inference.
//!
//! Architecture:
//!
//! ```text
//! Browser                          Web Worker
//! ┌─────────────────┐              ┌─────────────────────────┐
//! │ Sensorium UI       │  postMessage │ Inference Worker        │
//! │                 │ ──────────►  │                         │
//! │ "embed: text"   │              │ 1. Tokenize (tokenizers)│
//! │                 │              │ 2. Forward (candle-nn)  │
//! │ ◄────────────── │              │ 3. Mean-pool hidden     │
//! │ [f32; 384]      │  postMessage │ 4. L2-normalize         │
//! └─────────────────┘              └─────────────────────────┘
//!
//! Model loading (one-time):
//! 1. fetch("model.safetensors") → ArrayBuffer
//! 2. Cache in IndexedDB for subsequent loads
//! 3. candle_core::safetensors::load() into VarMap
//! 4. Build model graph (BertModel or MiniLM)
//! ```
//!
//! This module provides the Rust-side types and loading logic.
//! The Web Worker wrapper is in JavaScript (sensorium-inference-worker.js)
//! to handle the worker lifecycle.

/// Configuration for model loading.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// URL to fetch model weights from (safetensors format).
    pub weights_url: String,
    /// URL to fetch tokenizer.json from.
    pub tokenizer_url: String,
    /// Expected embedding dimension (e.g., 384 for MiniLM-L6-v2).
    pub embedding_dim: usize,
    /// Maximum sequence length (default: 128 tokens).
    pub max_seq_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            weights_url: "/assets/models/minilm-l6-v2/model.safetensors".into(),
            tokenizer_url: "/assets/models/minilm-l6-v2/tokenizer.json".into(),
            embedding_dim: 384,
            max_seq_len: 128,
        }
    }
}

/// Status of the model loading pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// Not yet started loading.
    Idle,
    /// Fetching model weights from URL.
    Downloading { progress_pct: f32 },
    /// Loading weights into candle tensors.
    Loading,
    /// Model is ready for inference.
    Ready,
    /// Loading failed.
    Error(String),
}

/// Message sent from the UI to the inference worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InferenceRequest {
    /// Load the model from the configured URLs.
    LoadModel,
    /// Encode a text string into an embedding vector.
    Encode { text: String, request_id: u32 },
    /// Encode a batch of texts.
    EncodeBatch { texts: Vec<String>, request_id: u32 },
}

/// Message sent from the inference worker back to the UI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InferenceResponse {
    /// Model loaded successfully.
    ModelReady { embedding_dim: usize },
    /// Model loading failed.
    ModelError { error: String },
    /// Single text embedding result.
    Embedding { request_id: u32, vector: Vec<f32> },
    /// Batch embedding result.
    EmbeddingBatch { request_id: u32, vectors: Vec<Vec<f32>> },
    /// Download progress update.
    Progress { pct: f32 },
}
