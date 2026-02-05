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
│  • HV16 (binary)        • Euler integration   • Active Inference           │
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

### HV16 and ContinuousHV

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

### Phi Measurement

Phi (Φ) quantifies integrated information - a key consciousness metric:

```rust,ignore
use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);
let phi = engine.compute_from_hvs(&node_representations);
println!("Integrated information: {:.4}", phi.phi);
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
println!("Phi: {:.4}, Coherence: {:.4}", result.phi, result.coherence);

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

// Hierarchical Cantor-LTC Network
pub mod hierarchical_cantor_ltc;

// Mind orchestration system
pub mod mind;

// Local HDC module - extends symthaea_core with additional modules
pub mod hdc;

// Minimal prelude
pub mod prelude;

// ============================================================================
// Standalone Module Files (Generally Stable)
// ============================================================================

// CfC (Closed-form Continuous-time) network
pub mod cfc;

// Cognitive loop for conscious processing
pub mod cognitive_loop;

// Unified LTC (Liquid Time-Constant) network
pub mod unified_ltc;

// Learnable LTC networks
pub mod learnable_ltc;

// HDC-LTC Unified Network Bridge (alternative to CfC)
pub mod hdc_ltc_bridge;

// Dynamics: attractor networks, temporal evolution
pub mod dynamics;

// Exploration: Surprise-driven exploration using FEP prediction errors
pub mod exploration;

// Attention: Phi-guided attention mechanisms
pub mod attention;

// Visualization: Attention debugging and interpretation tools
// Provides ASCII heatmaps, JSON export, and attention flow graphs
pub mod visualization;

// Two-Track Architecture: HDC semantics + CfC temporal
// Combines HDC for semantic meaning with CfC for temporal patterns
pub mod two_track;

// Bridges: Cross-representation translation between HDC and CfC
// Enables bidirectional semantic-temporal information flow
pub mod bridges;

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

// School: learning (enabled - with stub lookahead when full feature disabled)
pub mod school;

// Physiology (enabled - social coherence and hormone modeling)
pub mod physiology;

// Voice (enabled - 0 errors)
pub mod voice;

// Resonant speech (enabled - 0 errors)
pub mod resonant_speech;

// Embeddings (enabling - self-contained module)
pub mod embeddings;

// Benchmarks (enabled - API fixes complete)
pub mod benchmarks;

// Integration (cfg-gated - needs significant API alignment)
// The integration module expects ExecutionStrategy as enum with variants
// (Lost, Curious, Confident, Autopilot) and other API differences
#[cfg(feature = "integration_module")]
pub mod integration;

// Action (depends on consciousness module - now enabled)
pub mod action;

// Partnership (enabled - 0 errors)
pub mod partnership;

// User state inference (enabled - 0 errors)
pub mod user_state_inference;

// Observability
#[cfg(feature = "observability_module")]
pub mod observability;

// Shell (enabled - language module provides NixErrorDiagnoser)
pub mod shell;

// Experience (enabled - md5 crate added)
pub mod experience;

// Wisdom (enabled - 0 errors)
pub mod wisdom;

// Markets: HDC-based financial pattern recognition (enabled - 0 errors)
pub mod markets;

// Mycelix (enabled - GIS, Kosmic Song, Dark Spot DHT)
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

// Causal: Causal discovery integration with cognitive loop
// Tracks (input, output) pairs and discovers causal structure for attention weighting
pub mod causal;

// Substrate (enabled - 0 errors)
pub mod substrate;

// GUI bridge (enabled - 0 errors)
pub mod gui_bridge;

// Physics: Spark Engine, plasma encoding, and physical simulations
pub mod physics;

// REPL Orchestrator - Unified interactive system
// Wires together: cognitive_loop, language, action, voice, shell, observability
pub mod repl;

// API
#[cfg(feature = "api_module")]
pub mod api;

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

// ============================================================================
// Re-exports for Convenience
// ============================================================================

// Re-export key types at crate root
pub use mind::{ContinuousMind, MindConfig, MindState};

// Re-export symthaea-core for direct access to HDC primitives
pub use symthaea_core;

// Re-export phi_engine for consciousness calculations
pub use symthaea_core::phi_engine;

// Re-export core module for primitives like ContinuousHV, HDC_DIMENSION
pub use symthaea_core::core;
