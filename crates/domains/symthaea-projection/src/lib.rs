// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![deny(unsafe_code)]

//! # Holographic 2.5D Projection System — Data Model
//!
//! This crate defines the canonical data model for all projection modes:
//!
//! - [`ProjectionMode::TimeWaterfall`] — depth = time (present=front, past=rear)
//! - [`ProjectionMode::StratifiedStack`] — depth = abstraction layer
//! - [`ProjectionMode::HolographicCrossSection`] — depth = evidence chain depth
//!
//! ## Core Doctrine
//!
//! > 2D shows truth.
//! > 2.5D shows how truth became itself.
//! > Holographic projection shows why the truth is not alone.
//!
//! ## Visual Grammar (Immutable)
//!
//! **Depth must always mean one of three things:**
//! - Time (Time-Waterfall)
//! - Abstraction layer (Stratified Stack)
//! - Evidence/source-chain depth (Cross-Section)
//!
//! **Never use depth as decoration.**
//!
//! ## Design Philosophy
//!
//! The dashboard is an instrument, not a representation of the mind.
//! It visualizes evidence. It must look *less certain* when data is uncertain.
//!
//! A projection is good only if it answers a question.

pub mod edge;
pub mod frame;
pub mod grammar;
pub mod layer;
pub mod node;
pub mod visual_grammar;
pub mod waterfall;

pub use edge::{EdgeType, ProjectionEdge};
pub use frame::{ProjectionFrame, ProjectionMode, SourceSystem};
pub use grammar::{ColorRole, DepthMeaning, LineStyle, MotionType, OpacityState};
pub use layer::{LayerId, LayerType, ProjectionLayer, VisibilityState};
pub use node::{NodeSemanticType, ProjectionNode};
pub use waterfall::{WaterfallBuffer, WaterfallConfig};
