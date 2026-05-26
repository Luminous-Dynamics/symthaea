// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Integration Module
//!
//! **Modular architecture for consciousness pipeline components**
//!
//! This module provides a decomposed view of the consciousness integration:
//!
//! ## Modules
//!
//! - `config` - Integration configuration (`IntegrationConfig`)
//! - `metrics` - Consciousness metrics and optimization (`ConsciousnessMetricsReport`, `OptimizationRecommendation`)
//! - `state` - Consciousness state management (`ConsciousnessState`, `AlteredStateIndex`, `PredictiveMode`)
//! - `binding` - Object and temporal binding (`BoundObject`, `BindingLevel`, `TemporalBindingMemory`)
//! - `workspace` - Global workspace items (`WorkspaceItem`, `MetaThought`)
//! - `assessment` - Integration assessment (`IntegrationAssessment`)
//! - `pipeline` - Main consciousness pipeline (`ConsciousnessPipeline`)
//!
//! ## Integrated Subsystems
//!
//! The consciousness pipeline integrates these specialized modules:
//!
//! 1. **Metacognitive Monitoring** - Self-awareness of processing quality
//! 2. **Predictive Consciousness** - Active inference and anticipation
//! 3. **Cross-Modal Binding** - Multi-sensory integration
//! 4. **Temporal Binding** - Stream coherence with theta-phase modulation
//! 5. **Emergent Self-Model** - Recursive higher-order thought
//! 6. **Φ-Guided Optimization** - Network topology optimization
//! 7. **Feedback Dynamics** - Bidirectional processing
//! 8. **Meta-Consciousness** - Φ of Φ, strange loops
//! 9. **Temporal Consciousness** - Multi-scale time perception
//! 10. **Consciousness Creativity** - Novel idea generation
//! 11. **Phase Transitions** - Consciousness state changes
//! 12. **Epistemic Consciousness** - Belief and knowledge tracking
//! 13. **Fractal Consciousness** - Self-similar structure
//! 14. **Collective Consciousness** - Multi-agent awareness
//!
//! ## Backward Compatibility
//!
//! All types are re-exported from the parent module for compatibility:
//! ```rust,ignore
//! use symthaea::hdc::consciousness_integration::*; // Still works
//! use symthaea::hdc::consciousness::*;              // New recommended path
//! ```

// Re-export everything from the monolithic file for backward compatibility
// This allows gradual migration to the modular structure
pub use super::consciousness_integration::*;

// Future module structure (to be implemented incrementally):
//
// mod config;
// mod metrics;
// mod state;
// mod binding;
// mod workspace;
// mod assessment;
// mod pipeline;
//
// pub use config::*;
// pub use metrics::*;
// pub use state::*;
// pub use binding::*;
// pub use workspace::*;
// pub use assessment::*;
// pub use pipeline::*;
