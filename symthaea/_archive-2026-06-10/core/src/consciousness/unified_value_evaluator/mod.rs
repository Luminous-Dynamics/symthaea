// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Value Evaluator - Consciousness-Guided Decision Making
//!
//! This module unifies:
//! - **Eight Harmonies**: Semantic value alignment
//! - **Affective Consciousness**: CARE system for authenticity
//! - **Narrative Self**: Goal alignment and coherence
//! - **Veto Mechanism**: Self-preservation and integrity
//!
//! The key insight: **Genuine caring cannot be faked.**
//! By requiring CARE system activation alongside value alignment,
//! we distinguish authentic benevolence from mere compliance.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED VALUE EVALUATOR                            │
//! │                                                                       │
//! │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
//! │  │ Eight Harmonies │  │    Affective   │  │  Consciousness │         │
//! │  │   (Semantic)    │  │   (CARE/PLAY)  │  │    (Φ level)   │         │
//! │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘         │
//! │          │                   │                    │                  │
//! │          └───────────────────┼────────────────────┘                  │
//! │                              ▼                                       │
//! │                  ┌─────────────────────┐                             │
//! │                  │  Value Alignment    │                             │
//! │                  │  + Authenticity     │                             │
//! │                  │  + Consciousness    │                             │
//! │                  └──────────┬──────────┘                             │
//! │                             │                                        │
//! │                             ▼                                        │
//! │                  ┌─────────────────────┐                             │
//! │                  │   DECISION GATE     │                             │
//! │                  │  Allow/Warn/Veto    │                             │
//! │                  └─────────────────────┘                             │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use symthaea::consciousness::unified_value_evaluator::UnifiedValueEvaluator;
//!
//! let mut evaluator = UnifiedValueEvaluator::new();
//!
//! // Evaluate an action
//! let result = evaluator.evaluate(
//!     "help user understand their options",
//!     context,
//! );
//!
//! match result.decision {
//!     Decision::Allow => { /* proceed */ },
//!     Decision::Warn(reason) => { /* log warning, proceed */ },
//!     Decision::Veto(reason) => { /* block action */ },
//! }
//! ```

pub(crate) mod evaluator;
mod explanation;
mod types;

pub use evaluator::*;
pub use explanation::*;
pub use types::*;
