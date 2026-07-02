// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//!
//! # EduNet Content Generation Pipeline
//!
//! AI-assisted educational content generation using Symthaea Broca's ThoughtChannels API.
//!
//! This crate provides:
//! - **Content types**: Lessons, practice problems, assessments, flashcards
//! - **Channel presets**: ThoughtChannel configurations for educational content generation
//! - **Pipeline orchestration**: Takes a Common Core standard and produces complete teaching materials
//! - **Mock generator**: Realistic content for development without a Symthaea runtime
//!
//! ## Architecture
//!
//! The pipeline does NOT import Broca directly (it requires the full Symthaea runtime).
//! Instead, it defines data structures and orchestration logic through the [`ContentGenerator`]
//! trait. A real Broca adapter maps [`ContentChannels`] to Broca's 43-channel ThoughtChannels
//! at the integration boundary. For development, [`MockGenerator`] returns mathematically
//! correct content for Grade 3 standards.
//!
//! ## Usage
//!
//! ```rust
//! use edunet_content_gen::prelude::*;
//!
//! let pipeline = ContentPipeline::new(MockGenerator);
//! let standard = StandardInput {
//!     code: "3.OA.A.1".to_string(),
//!     description: "Interpret products of whole numbers".to_string(),
//!     grade_level: "Grade3".to_string(),
//!     domain: "Operations & Algebraic Thinking".to_string(),
//!     prerequisites: vec!["Addition within 100".to_string()],
//! };
//! let lesson = pipeline.generate_lesson(&standard).unwrap();
//! assert!(!lesson.explanation.is_empty());
//! ```

pub mod batch;
pub mod broca_adapter;
pub mod caps_content;
pub mod channels;
pub mod dedup;
pub mod ingest;
pub mod international_content;
pub mod mock;
pub mod pipeline;
pub mod retention_challenges;
pub mod templates;
pub mod translation_seeds;
pub mod types;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::channels::{ContentChannels, ContentIntent};
    pub use crate::mock::MockGenerator;
    pub use crate::pipeline::{ContentGenError, ContentGenerator, ContentPipeline, GenerationOutput, StandardInput};
    pub use crate::types::*;
}
