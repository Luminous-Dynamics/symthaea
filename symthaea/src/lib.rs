// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
# Symthaea: Holographic Liquid Brain

A consciousness-first AI framework combining Hyperdimensional Computing (HDC),
Liquid Time-Constant Networks (LTC), and Integrated Information Theory (IIT/Phi).

## Overview

Symthaea implements a novel cognitive architecture where:

- **Neuron state IS a hypervector** (16,384 dimensions)
- **Phi measurement** guides consciousness-aware processing
- **Free Energy Principle** drives action selection and learning
- **Temporal dynamics** use closed-form LTC solutions for O(1) time jumps

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYMTHAEA ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                 │
│  │   PERCEPTION  │   │   COGNITION   │   │    ACTION     │                 │
│  │               │   │               │   │               │                 │
│  │ • HDC Encode  │──▶│ • LTC Dynamics│──▶│ • FEP Bridge  │                 │
│  │ • Multi-modal │   │ • Phi Measure │   │ • Motor Cmd   │                 │
│  │ • Binding     │   │ • Prediction  │   │ • Execution   │                 │
│  └───────────────┘   └───────┬───────┘   └───────────────┘                 │
│                              │                                              │
│                     ┌────────▼────────┐                                     │
│                     │  CONSCIOUSNESS  │                                     │
│                     │                 │                                     │
│                     │ • Phi (IIT)     │                                     │
│                     │ • Coherence     │                                     │
│                     │ • Flow State    │                                     │
│                     └─────────────────┘                                     │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                           SUBSYSTEMS                                        │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  HDC (symthaea_core)    LTC (unified_ltc)      FEP (consciousness)         │
│  ─────────────────────  ─────────────────────  ─────────────────────        │
│  • BinaryHV (binary)        • Euler integration   • Active Inference           │
│  • ContinuousHV (f32)   • RK4 integration     • Motor Commands             │
│  • Bind (⊗)             • Closed-form O(1)    • TD Learning                │
│  • Bundle (⊕)           • Hebbian learning    • Precision Gating           │
│  • 16,384 dimensions    • State-dependent τ   • 8 action types             │
│                                                                             │
│  SWARM (distributed)    REPL (interactive)    PHI ENGINE                   │
│  ─────────────────────  ─────────────────────  ─────────────────────        │
│  • Iroh (fast tensor)   • Voice output        • Spectral method            │
│  • Holochain (trust)    • Cognitive loop      • Tiered approximation       │
│  • Mirror neurons       • Action execution    • Resonator O(n log n)       │
│  • Swarm coherence      • IPC daemon mode     • Cached results             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### BinaryHV and ContinuousHV

Hypervectors are high-dimensional vectors (16,384D) with special algebraic properties:

```rust,ignore
use symthaea_core::hdc::unified_hv::{ContinuousHV, BinaryHV, HDC_DIMENSION};

// Create random hypervectors (nearly orthogonal)
let a = ContinuousHV::random(HDC_DIMENSION, 42);
let b = ContinuousHV::random(HDC_DIMENSION, 43);

// Binding creates associations (dissimilar to inputs)
let bound = a.bind(&b);
assert!(bound.similarity(&a).abs() < 0.1);

// Bundling creates superpositions (similar to all inputs)
let bundled = ContinuousHV::bundle(&[&a, &b]);
assert!(bundled.similarity(&a) > 0.5);
```

### HDC-LTC Unified Network

The unified architecture uses hypervectors as neuron state, enabling:
- O(1) temporal jumps via closed-form solution
- Binding-based weight application (no matrix multiply)
- Natural integration with phi measurement

```rust,ignore
use symthaea::hdc::{HdcLtcUnifiedNeuron, UnifiedConfig, ContinuousHV};

let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
let input = ContinuousHV::random_default(123);

// Closed-form evolution: O(1) regardless of dt
neuron.evolve_closed_form(1.0, &input);  // 1 second jump
neuron.evolve_closed_form(100.0, &input); // 100 second jump, same cost!
```

### Consciousness Measurement (Three Layers)

Symthaea uses a tiered measurement system for consciousness:

- **Layer 1: Ψ (Psi)** — Fast consciousness estimate (every cycle, O(1)).
  Composite soft signal from temporal coherence, voice quality, flow state,
  relational, body, and embodied subsystems. Used for runtime gating,
  memory graduation, and action selection.

- **Layer 2: Σ (Sigma)** — Synergistic integration (every N cycles, O(n²)).
  PhiR-inspired measure via information decomposition on HDC states.
  Quantifies how much the whole exceeds the sum of its parts.

- **Layer 3: Φ (Phi)** — True IIT integrated information (on demand, O(n³)).
  Spectral MIP search + Phi* (Gaussian). Used for research validation
  and deep diagnostics. See `symthaea_core::consciousness_metrics`.

```rust,ignore
use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);
let result = engine.compute_from_hvs(&node_representations);
println!("Phi: {:.4} ({})", result.value, result.category);
```

### FEP Active Inference

The Free Energy Principle drives action selection through 8 motor command types:

```rust,ignore
use symthaea::consciousness::fep_active_inference::{
    EnhancedFEPBridge, ActiveInferenceAgentConfig, MotorCommandType
};

let mut bridge = EnhancedFEPBridge::new(config, 4);
let result = bridge.cycle(phi, integration, coherence, attention);

match result.motor_command.command_type {
    MotorCommandType::AttentionShift => { /* redirect focus */ }
    MotorCommandType::ExplorationTrigger => { /* seek novelty */ }
    MotorCommandType::MemoryConsolidate => { /* strengthen memory */ }
    _ => {}
}
```

## Quick Start

```rust,ignore
use symthaea::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopService, CognitiveLoopConfig};

// Create cognitive loop with CfC backend
let config = CognitiveLoopConfig::with_cfc();
let mut service = CognitiveLoopService::new(config)?;

// Process input
let result = service.cycle("Hello, Symthaea");
println!("Psi: {:.4}, Coherence: {:.4}", result.psi, result.coherence);

// Check consciousness state
let snapshot = service.consciousness_snapshot();
if snapshot.in_flow {
    println!("System is in flow state!");
}
```

## Module Organization

### Core Stable Modules
- [`perception`]: Multi-modal sensory encoding to HDC
- [`cognitive_loop`]: Main consciousness processing loop
- [`unified_ltc`]: Liquid Time-Constant network implementation
- [`hdc_ltc_bridge`]: Bridge for using HDC-LTC in cognitive loop
- [`consciousness`]: Consciousness metrics, FEP, active inference

### Infrastructure Modules
- [`repl`]: Interactive REPL with voice output and action execution
- [`swarm`]: Distributed consciousness via Iroh + Holochain
- [`voice`]: Text-to-speech with consciousness-modulated pacing
- [`action`]: Motor cortex for command execution
- [`control_plane`]: Shared auth, protocol, and audit helpers for remote interfaces

### Memory and Learning
- [`memory`]: Hippocampal and working memory systems
- [`school`]: Learning algorithms and curriculum

## Feature Flags

- `swarm`: Enable distributed consciousness networking
- `voice`: Enable TTS voice output
- `web_research_module`: Epistemic web research capabilities
- `api_module`: REST API server
- `integration_module`: Advanced integration features

## Re-exports

Key types are re-exported at the crate root for convenience:
- [`ContinuousMind`], [`MindConfig`], [`MindState`] from [`mind`]
- [`symthaea_core`] for direct HDC primitive access
- [`phi_engine`] for consciousness measurement

*/

// ── Strict deny lints ──
// Workspace lints (Cargo.toml [workspace.lints]) handle: dbg_macro, todo,
// unimplemented, eq_op, erasing_op, invalid_regex, self_assignment,
// cast_ptr_alignment, nonsensical_open_options, unused_io_amount.
// Below: additional crate-specific denies not in workspace config.
#![deny(clippy::zero_divided_by_zero, clippy::fn_to_numeric_cast)]
// Suppress "unknown lint" when clippy version doesn't recognize newer lints.
#![allow(unknown_lints)]
// HDC code uses index-based iteration for element-wise vector operations (~1,075 sites).
#![allow(clippy::needless_range_loop)]
// Crate-level allows: each lint has 5+ occurrences across src/ (verified Feb 2026).
// Lints with <5 occurrences moved to per-function #[allow(...)] annotations.
#![allow(
    clippy::if_same_then_else,
    clippy::large_enum_variant,
    clippy::manual_clamp,
    clippy::new_ret_no_self,
    clippy::new_without_default,
    clippy::should_implement_trait,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::wrong_self_convention,
    clippy::await_holding_lock,
    clippy::cloned_ref_to_slice_refs,
    clippy::doc_overindented_list_items,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_map_or,
    // ── Style lints with many occurrences across feature-gated modules ──
    clippy::collapsible_if,
    clippy::clone_on_copy,
    clippy::derivable_impls,
    clippy::doc_lazy_continuation,
    clippy::explicit_into_iter_loop,
    clippy::field_reassign_with_default,
    clippy::len_zero,
    clippy::manual_div_ceil,
    clippy::manual_map,
    clippy::manual_range_contains,
    clippy::manual_saturating_arithmetic, // pre-1.93 name
    clippy::manual_strip,
    clippy::map_clone,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::or_fun_call,
    clippy::ptr_arg,
    clippy::push_str_single,
    clippy::redundant_closure,
    clippy::repeat_vec_with_capacity,
    clippy::single_char_add_str,
    clippy::single_element_loop,
    clippy::trim_split_whitespace,
    clippy::unnecessary_cast,
    clippy::unnecessary_lazy_evaluations,
    clippy::unnecessary_to_owned,
    clippy::useless_conversion,
    clippy::useless_format,
    // Additional lints from --all-features clippy (exact names from CI)
    clippy::double_ended_iterator_last,
    clippy::implicit_saturating_sub,
    clippy::io_other_error,
    clippy::manual_abs_diff,
    clippy::manual_pattern_char_comparison,
    clippy::manual_repeat_n,
    clippy::op_ref,
    clippy::unused_enumerate_index,
    clippy::unwrap_or_default,
    // Dead code: many fields/methods reserved for future feature wiring
    dead_code,
    unused_imports,
    unused_variables,
    unused_assignments
)]

// ============================================================================
// Feature Flag Guards
// ============================================================================

#[cfg(all(feature = "broca_lite", feature = "ssm_language"))]
compile_error!(
    "Features `broca_lite` and `ssm_language` are mutually exclusive. \
     broca_lite provides lightweight generation via symthaea-spore; \
     ssm_language provides full CfC-HDC generation via symthaea-broca. \
     Enable only one."
);

// ============================================================================
// Unified Error Types
// ============================================================================

/// Unified error types for cross-subsystem error handling.
pub mod errors;
pub use errors::{SymthaeaError, SymthaeaResult};

// ============================================================================
// Symthaea Facade (Primary Entry Point)
// ============================================================================
pub mod symthaea;
pub use symthaea::Symthaea;

// ============================================================================
// Core Modules (Stable, Verified Working)
// ============================================================================

// Perception: Multi-modal sensory processing
pub mod perception;

// Chronobiology: Time-dependent cognitive modulation
pub mod chronobiology;

// Mind orchestration system
pub mod mind;

// Local HDC module - extends symthaea_core with additional modules
pub mod hdc;

// ============================================================================
// Standalone Module Files (Generally Stable)
// ============================================================================

// Cognitive loop for conscious processing
pub mod cognitive_loop;

// Unified LTC (Liquid Time-Constant) network
pub mod unified_ltc;

// HDC-LTC Unified Network Bridge (alternative to CfC)
pub mod hdc_ltc_bridge;

// Cross-environment transport and localization policy
pub mod domain;

// Dynamics: attractor networks, temporal evolution
pub mod dynamics;

// Exploration: Surprise-driven exploration using FEP prediction errors
pub mod exploration;

// Attention: Phi-guided attention mechanisms
pub mod attention;

// Visualization: Attention debugging and interpretation tools
// Provides ASCII heatmaps, JSON export, and attention flow graphs
pub mod visualization;

// two_track and bridges modules REMOVED (Feb 2026) — superseded by cognitive_loop/cycle.rs
// which performs HDC+CfC fusion directly with 30+ subsystem imports.

// Inference: Production-ready streaming and batch inference
// Provides real-time processing with configurable latency/throughput tradeoffs
pub mod inference;

// ============================================================================
// Modules with Known Import Issues (Conditionally Compiled)
// Many of these have internal structural issues that need fixing
// ============================================================================

// Consciousness module (enabling - fixing dependencies)
pub mod consciousness;

// Memory systems (fixed and enabled)
pub mod memory;

// Brain regions (enabling - fixing errors)
pub mod brain;

// Soul module (enabled - self-contained)
pub mod soul;

// Language processing (enabled - core modules, advanced gated behind full_language)
pub mod language;

// School: anticipatory curriculum learning with CfC lookahead, mastery tracking
#[cfg(feature = "school_learning")]
pub mod school;

// State Space Models (SSM) integration modules
pub mod ssm;

// Physiology (enabled - social coherence and hormone modeling)
pub mod physiology;

// Voice (enabled - 0 errors)
pub mod voice;

// Resonant speech (enabled - 0 errors)
pub mod resonant_speech;

// Embeddings (extracted to symthaea-embeddings crate)
pub use symthaea_embeddings as embeddings;

// Benchmarks: causal validation tests (not criterion benchmarks).
// Used by examples/ for validation experiments.
#[cfg(feature = "benchmarks")]
pub mod benchmarks;

// Integration (cfg-gated for build isolation)
// ConsciousPipeline depends on ExecutionStrategy from language module
// and pulls in heavyweight nix-mind dependencies.
#[cfg(any(feature = "integration_module", feature = "nix-mind"))]
pub mod integration;

// Action (depends on consciousness module - now enabled)
pub mod action;
pub mod control_plane;

// Coding Agent: multi-step consciousness-gated coding loop
#[cfg(feature = "code_generation")]
pub mod coding_agent;
pub mod coding_experience;

// Partnership (enabled - 0 errors)
pub mod partnership;

// User state inference (enabled - 0 errors)
pub mod user_state_inference;

// Observability
#[cfg(feature = "observability_module")]
pub use symthaea_observability as observability;

// Shell (enabled - language module provides NixErrorDiagnoser)
pub mod shell;

// Experience (enabled - uses blake3 for topic hashing)
pub mod experience;

// Wisdom (enabled - 0 errors)
pub mod wisdom;

// Mycelix core types (GIS, KosmicSong, EpistemicMesh) — always compiled because
// KosmicSong is a non-optional field on CognitiveLoopService and GIS types are
// used in the core consciousness pipeline. The `mycelix` Cargo feature gates
// external dependencies (FL algorithms, hashing) not these core type definitions.
pub mod mycelix;

// Swarm Intelligence (Hybrid Iroh + Holochain Architecture)
// Uses Iroh for real-time tensor streaming (<50ms) and Holochain for trust/identity
pub mod swarm;

// Safety (enabled - with stub implementations)
pub mod safety;

// Databases (enabled - types defined in mod.rs)
pub mod databases;

// Infrastructure (enabled - 0 errors)
pub mod infrastructure;

// Intelligence (enabled - 0 errors)
pub mod intelligence;

// Physics Bridge: HDC semantic search for mathematical physics analogies + guided design exploration
#[cfg(feature = "physics-bridge")]
pub mod spark_physics_bridge;

// Causal: Causal discovery integration with cognitive loop
// Tracks (input, output) pairs and discovers causal structure for attention weighting
pub mod causal;

// Identity: MFDI (Multi-Factor Delegated Identity) integration
// Enables agents to prove identity, sign outputs, and carry reputation
#[cfg(feature = "identity")]
pub mod identity;

// Integrity: Hardware integrity & tamper detection (attestation, temporal checks, canaries)
#[cfg(feature = "integrity")]
pub mod integrity;

// GUI bridge — only needed by symthaea-gui binary
#[cfg(feature = "gui")]
pub mod gui_bridge;

// Physics: tokamak plasma encoding, C-Mod adapter, Phi-based control loops
// Extracted to crates/symthaea-physics (deps resolved: FEP via symthaea-fep, TieredPhi via symthaea-core)
#[cfg(feature = "physics")]
pub use symthaea_physics as physics;

// Cell Foundry: iPSC, IVG, SCNT, multi-scale prediction, lab controller, hydrology
#[cfg(feature = "cell-foundry")]
pub use symthaea_cell_foundry;

// Flight: HDC-LTC + FEP Active Inference quadrotor flight control
#[cfg(feature = "flight")]
pub use symthaea_flight as flight;

// Humanoid: HDC-LTC + FEP Active Inference bipedal humanoid control (DMC benchmark)
#[cfg(feature = "humanoid")]
pub use symthaea_humanoid as humanoid;

// Helicopter: HDC-LTC + FEP Active Inference SAR helicopter control
#[cfg(feature = "helicopter")]
pub use symthaea_helicopter as helicopter;

// Vehicle: HDC-LTC + FEP Active Inference autonomous vehicle control
#[cfg(feature = "vehicle")]
pub use symthaea_vehicle as vehicle;

// Manipulator: HDC-LTC + FEP Active Inference consciousness-coupled industrial arm
#[cfg(feature = "manipulator")]
pub use symthaea_manipulator as manipulator;

// AUV: HDC-LTC + FEP Active Inference autonomous underwater vehicle for commons stewardship
#[cfg(feature = "auv")]
pub use symthaea_auv as auv;

// Exoskeleton: wearable consciousness-coupled assistive device
#[cfg(feature = "exoskeleton")]
pub use symthaea_exoskeleton as exoskeleton;

// Surgical: highest-stakes consciousness-coupled surgical robot
#[cfg(feature = "surgical")]
pub use symthaea_surgical as surgical;

// Quadruped: legged locomotion with phi-gated gait selection
#[cfg(feature = "quadruped")]
pub use symthaea_quadruped as quadruped;

// Orbital: spacecraft with NASA-standard safe-mode fallback
#[cfg(feature = "orbital")]
pub use symthaea_orbital as orbital;

// Meta: Self-analysis, code quality metrics, active inference exploration, dream synthesis
#[cfg(feature = "code_generation")]
pub mod meta;

// REPL Orchestrator - Unified interactive system
// Wires together: cognitive_loop, language, action, voice, shell, observability
pub mod repl;

// API
#[cfg(feature = "api_module")]
pub mod api;

// Scientific Method: hypothesis-test-update cycle (Phase 4b math plan)
// Wires HDC encoding, Bayesian statistics, and propositional logic into a
// scientific reasoning engine.
#[cfg(feature = "scientific_method")]
pub mod scientific_method;

// Web Research: Epistemic verification and autonomous research
//
// This module provides three-level epistemic consciousness:
// 1. Base Consciousness (Phi) - Already present in Symthaea
// 2. Epistemic Consciousness - Knows what it knows via verification
// 3. Meta-Epistemic Consciousness - Self-improving verification (Meta-Phi)
//
// Submodules: types, knowledge_graph, extractor, verifier, researcher,
// integrator, meta_learning. See docs/developer/WEB_RESEARCH_INTEGRATION_GUIDE.md.
#[cfg(feature = "web_research_module")]
pub mod web_research;

pub mod knowledge;

// Z3 Formal Verification Bridge (Phase 4B — math/science plan)
// Dual-mode: subprocess z3 binary (SMTLIB2) when available, internal DPLL fallback always.
// Domain-specific verifiers for ODE solutions, Einstein condition, CSP solutions.
pub mod z3_bridge;

// Phase 5: Curriculum generator with Z3-verified lemmas. Generates formal
// lemma statements from the IMO curriculum and verifies them via z3_bridge.
// JSON-persisted catalog for human inspection + downstream proof-checker reuse.
pub mod curriculum_generator;

// Proof State: goal trees, tactic engine, proof search (Phase 5 math plan)
// ProofTree struct with goal stack, tactic application, backtracking search.
pub mod proof_state;

// ============================================================================
// Re-exports for Convenience
// ============================================================================

// Re-export key types at crate root
pub use mind::{ContinuousMind, MindConfig, MindState};

// Re-export symthaea-core for direct access to HDC primitives
pub use symthaea_core;

// Re-export symthaea-nix for conscious NixOS mind
#[cfg(feature = "nix-mind")]
pub use symthaea_nix;

// Re-export phi_engine for consciousness calculations
pub use symthaea_core::phi_engine;

// Re-export core module for primitives like ContinuousHV, HDC_DIMENSION
pub use symthaea_core::core;
