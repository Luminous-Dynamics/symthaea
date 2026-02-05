//! # Conscious Integration Pipeline
//!
//! Orchestrates the end-to-end consciousness-aware AI pipeline:
//!
//! ```text
//! User Input
//!     ↓
//! Memory Context Retrieval
//!     ↓
//! LLM Understanding (Φ-gated)
//!     ↓
//! Φ Measurement & Confidence
//!     ↓
//! Action Gating (ConsciousnessThresholds)
//!     ↓
//! Action Execution (with rollback)
//!     ↓
//! Outcome Recording (causal learning)
//!     ↓
//! Adaptive Threshold Update
//! ```
//!
//! ## Key Components
//!
//! - [`ConsciousPipeline`] - The main orchestrator
//! - [`PipelineConfig`] - Configuration for the pipeline
//! - [`PipelineResult`] - Result of pipeline execution
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::integration::{ConsciousPipeline, PipelineConfig};
//!
//! let mut pipeline = ConsciousPipeline::new(PipelineConfig::default()).await?;
//! let result = pipeline.process("Install htop on my system").await?;
//! println!("Action: {:?}, Confidence: {:?}", result.action, result.confidence);
//! ```

mod conscious_pipeline;

pub use conscious_pipeline::*;
