//! # Cognitive Routers Module
//!
//! This module contains specialized routers for consciousness-aware processing.
//! Each router implements a different cognitive paradigm:
//!
//! ## Router Hierarchy
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        ROUTER ARCHITECTURE                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  BASE ROUTERS (Independent)                                              │
//! │  ├── PredictiveRouter         - World model predictions                  │
//! │  ├── CausalValidatedRouter    - Causal emergence validation              │
//! │  ├── InformationGeometricRouter - Fisher information geometry            │
//! │  ├── TopologicalConsciousnessRouter - Persistent homology                │
//! │  ├── QuantumCoherenceRouter   - Quantum-inspired coherence               │
//! │  ├── ActiveInferenceRouter    - Free energy minimization                 │
//! │  ├── PredictiveProcessingRouter - Hierarchical predictive coding         │
//! │  ├── ASTRouter                - Attention schema theory                  │
//! │  ├── GlobalWorkspaceRouter    - Global workspace theory                  │
//! │  └── ResonantConsciousnessRouter - HDC+LTC+Resonator soft routing        │
//! │                                                                          │
//! │  COMPOSITE ROUTERS (Build on base routers)                               │
//! │  ├── OscillatoryRouter        - Contains PredictiveRouter                │
//! │  └── MetaRouter               - Selects between all routers              │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! Each router can be used independently or combined through the MetaRouter:
//!
//! ```rust,ignore
//! use symthaea::consciousness::recursive_improvement::routers::*;
//!
//! // Use predictive router directly
//! let mut router = PredictiveRouter::new(PredictiveRouterConfig::default());
//! let plan = router.plan_route(&current_state);
//!
//! // Use meta-router for paradigm selection
//! let meta = MetaRouter::new(MetaRouterConfig::default());
//! let best_paradigm = meta.select_paradigm(&state, &metrics);
//! ```

// Shared types for all routers
mod types;
pub use types::*;

// Individual router implementations
mod predictive;
pub use predictive::*;

// Phase 5G: OscillatoryRouter type mismatch FIXED (December 30, 2025)
// OscillatoryState is now a proper struct with phase, frequency, phi, coherence, phase_category
// OscillatoryPhase is now an enum with Peak, Rising, Falling, Trough + methods
mod oscillatory;
pub use oscillatory::*;

mod causal_validated;
pub use causal_validated::*;

mod information_geometric;
pub use information_geometric::*;

mod topological;
pub use topological::*;

mod quantum_coherence;
pub use quantum_coherence::*;

mod active_inference;
pub use active_inference::*;

mod predictive_processing;
pub use predictive_processing::*;

mod ast;
pub use ast::*;

mod global_workspace;
pub use global_workspace::*;

mod meta;
pub use meta::*;

// Phase 5H: HDC+LTC+Resonator fusion router (December 30, 2025)
// Soft routing with smooth transitions, eliminates hard Φ threshold artifacts
mod resonant;
pub use resonant::*;
