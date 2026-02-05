//! Hyperdimensional Computing (HDC) Module
//!
//! This module provides the core HDC/VSA (Vector Symbolic Architecture) primitives
//! and consciousness-related computations for Symthaea. It re-exports fundamental
//! types from `symthaea-core` and adds local extensions for consciousness-specific
//! computations.
//!
//! # Overview
//!
//! Hyperdimensional Computing (HDC) is a brain-inspired computing paradigm that
//! represents information as high-dimensional vectors (hypervectors). Key properties:
//!
//! - **High dimensionality**: 16,384 dimensions (2^14) by default
//! - **Holographic**: Information distributed across all dimensions
//! - **Noise-tolerant**: Robust to corruption and partial information
//! - **Efficient**: Binary operations enable SIMD acceleration
//!
//! # Core Types
//!
//! ## Binary Hypervectors ([`HV16`])
//!
//! The primary type for efficient HDC operations. 16,384-bit vectors (2KB) with:
//! - 32x memory reduction vs `Vec<f32>`
//! - 200x faster operations via SIMD (XOR, popcount)
//! - Deterministic random generation from seeds
//!
//! ```rust,ignore
//! use symthaea::hdc::HV16;
//!
//! let a = HV16::random(42);  // Deterministic from seed
//! let b = HV16::random(43);
//!
//! // Bind: create associations (XOR)
//! let bound = a.bind(&b);
//!
//! // Bundle: create prototypes (majority vote)
//! let prototype = HV16::bundle(&[a, b]);
//!
//! // Similarity: 0.0 = opposite, 0.5 = random, 1.0 = identical
//! let sim = a.similarity(&b);
//! ```
//!
//! ## Continuous Hypervectors ([`ContinuousHV`])
//!
//! Real-valued hypervectors for applications requiring gradient computation
//! or finer-grained similarity measurements.
//!
//! ## HDC-LTC Unified Neurons ([`HdcLtcUnifiedNeuron`])
//!
//! Revolutionary architecture where neuron state IS a hypervector, enabling:
//! - O(1) temporal jumps via closed-form solution
//! - Binding-based weight application (no matrix multiply)
//! - Natural integration with phi measurement
//!
//! # Fundamental Operations
//!
//! | Operation | Symbol | Purpose | Properties |
//! |-----------|--------|---------|------------|
//! | **Bind**  | `bind()` | Create associations | Commutative, self-inverse |
//! | **Bundle** | `bundle()` | Create prototypes | Preserves similarity |
//! | **Permute** | `permute()` | Encode sequences | Order-sensitive |
//!
//! # Module Organization
//!
//! - **Core types**: [`HV16`], [`ContinuousHV`], [`BinaryHV`], [`RealHV`]
//! - **Operations**: [`simd_ops`] for accelerated computation
//! - **Encoding**: [`text_encoder`], [`semantic_encoder`], [`semantic_decoder`]
//! - **Consciousness**: [`tiered_phi`], [`phi`], [`consciousness`] submodules
//! - **Networks**: [`HdcLtcUnifiedNeuron`], [`reservoir`], [`cincinnati_ltc`]
//!
//! # Examples
//!
//! ## Encoding Text
//!
//! ```rust,ignore
//! use symthaea::hdc::{TextEncoder, TextEncoderConfig, HV16};
//!
//! let encoder = TextEncoder::new(TextEncoderConfig::default());
//! let hello = encoder.encode("hello");
//! let world = encoder.encode("world");
//!
//! // Similar words have higher similarity
//! let greeting = encoder.encode("hi");
//! assert!(hello.similarity(&greeting) > hello.similarity(&world));
//! ```
//!
//! ## Graph Encoding
//!
//! ```rust,ignore
//! use symthaea::hdc::HV16;
//!
//! // Create unique basis vectors for nodes
//! let node_a = HV16::basis(0);
//! let node_b = HV16::basis(1);
//! let node_c = HV16::basis(2);
//!
//! // Encode edges as bindings
//! let edge_ab = node_a.bind(&node_b);
//! let edge_bc = node_b.bind(&node_c);
//!
//! // Node representation = bundle of incident edges
//! let node_b_repr = HV16::bundle(&[edge_ab, edge_bc]);
//! ```
//!
//! # See Also
//!
//! - [`symthaea_core::hdc`] - Core HDC primitives
//! - [`crate::consciousness`] - Phi measurement and IIT
//! - [`crate::unified_ltc`] - Liquid Time-Constant networks

#![allow(unused_imports)]

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS FROM SYMTHAEA-CORE - Core Types
// ═══════════════════════════════════════════════════════════════════════════════

// Core HDC types from symthaea-core
pub use symthaea_core::hdc::{
    // Binary hypervectors
    binary_hv::HV16,
    // Unified hypervector types
    unified_hv::{ContinuousHV, BinaryHV, HV, HDC_DIMENSION},
    // Real-valued hypervectors
    real_hv::RealHV,
    // LTC neuron count constants
    LTC_NEURONS,
    // Primitive system
    primitive_system::{PrimitiveSystem, Primitive, PrimitiveTier},
    // Tiered Phi calculation
    TieredPhi,
    // Native similarity with PackedBipolar
    PackedBipolar, NativeSimilarityIndex,
    // Causal types
    CausalDirection,
    // HDC context for arena-based operations
    HdcContext,
};

// Re-export native_similarity module for tests
pub use symthaea_core::hdc::native_similarity;

// Re-export sensorimotor contingencies (enactivist perception)
pub use symthaea_core::hdc::sensorimotor_contingencies;

/// Convenient alias for the standard HDC dimension (16,384).
pub const HDC_DIM: usize = symthaea_core::hdc::unified_hv::HDC_DIMENSION;

// Phi engine types from symthaea-core
pub use symthaea_core::phi_engine::{
    PhiEngine, PhiMethod, PhiResult, PhiCalculator,
};

// Observability types from symthaea-core
pub use symthaea_core::observability::{
    SharedObserver, Observer, NoOpObserver, no_op_observer,
    PhiComponents, PhiMeasurementEvent,
};

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS FROM SYMTHAEA-CORE - Full Modules
// ═══════════════════════════════════════════════════════════════════════════════

// Re-export entire modules for compatibility
pub use symthaea_core::hdc::hebbian;
pub use symthaea_core::hdc::adaptive_topology;
pub use symthaea_core::hdc::sleep_and_altered_states;
pub use symthaea_core::hdc::causal_mind;
pub use symthaea_core::hdc::cross_modal_binding;
pub use symthaea_core::hdc::unified_cognitive_core;
pub use symthaea_core::hdc::collective_consciousness;
pub use symthaea_core::hdc::grounded_understanding;
pub use symthaea_core::hdc::predictive_coding;
pub use symthaea_core::hdc::simd_hv16;
pub use symthaea_core::hdc::binary_hv;
pub use symthaea_core::hdc::unified_hv;
pub use symthaea_core::hdc::real_hv;
pub use symthaea_core::hdc::primitive_system;
pub use symthaea_core::hdc::simd_ops;
pub use symthaea_core::hdc::consciousness_integration;
pub use symthaea_core::hdc::consciousness_topology_generators;
pub use symthaea_core::hdc::spectral_connectivity;
pub use symthaea_core::hdc::phi_resonant;
pub use symthaea_core::hdc::global_workspace;
pub use symthaea_core::hdc::meta_consciousness;
pub use symthaea_core::hdc::relational_consciousness;
pub use symthaea_core::hdc::text_encoder;
pub use symthaea_core::hdc::semantic_encoder;
pub use symthaea_core::hdc::semantic_decoder;
pub use symthaea_core::hdc::reservoir;
pub use symthaea_core::hdc::cincinnati_ltc;
pub use symthaea_core::hdc::cincinnati_enhanced;
pub use symthaea_core::hdc::cincinnati_advanced;
pub use symthaea_core::hdc::cincinnati_network;
pub use symthaea_core::hdc::gwt_cincinnati_integration;
pub use symthaea_core::hdc::universal_semantics;
pub use symthaea_core::hdc::tiered_phi as core_tiered_phi;
pub use symthaea_core::hdc::arithmetic_engine;
pub use symthaea_core::hdc::consciousness_persistence as core_consciousness_persistence;
pub use symthaea_core::hdc::conscious_learning as core_conscious_learning;
pub use symthaea_core::hdc::adaptive_learning_signals as core_adaptive_learning;
pub use symthaea_core::hdc::unified_understanding as core_unified_understanding;
pub use symthaea_core::hdc::full_stack_consciousness as core_full_stack;
pub use symthaea_core::hdc::unified_conscious_being as core_unified_being;
pub use symthaea_core::hdc::ecosystem_bridge as core_ecosystem_bridge;
pub use symthaea_core::hdc::predictive_encoder as core_predictive_encoder;
pub use symthaea_core::hdc::emotional_depth as core_emotional_depth;
pub use symthaea_core::hdc::cross_modal_attention_router as core_attention_router;
pub use symthaea_core::hdc::consciousness_streaming as core_streaming;
pub use symthaea_core::hdc::consciousness_metacognition as core_metacognition;
pub use symthaea_core::hdc::counterfactual_dreams as core_counterfactual;
pub use symthaea_core::hdc::self_improvement_integration as core_self_improvement;
pub use symthaea_core::hdc::consciousness_advanced_cognition as core_advanced_cognition;
pub use symthaea_core::hdc::consciousness_complete_being as core_complete_being;
pub use symthaea_core::hdc::consciousness_cross_integration as core_cross_integration;
pub use symthaea_core::hdc::consciousness_feedback_dynamics as core_feedback_dynamics;
pub use symthaea_core::hdc::hdc_ltc_unified as core_hdc_ltc_unified;

// Re-export unified HDC-LTC types (revolutionary closed-form dynamics)
pub use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNeuron, HdcLtcUnifiedNetwork,
    UnifiedConfig, UnifiedNetworkConfig,
    UnifiedActivation, UnifiedNeuronStats, UnifiedNetworkStats,
};

// Re-export from causal_mind
pub use symthaea_core::hdc::causal_mind::{CausalDiscoveryResult, CausalFeatures};

// Re-export SemanticPrime
pub use symthaea_core::hdc::universal_semantics::SemanticPrime;

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - Convenience Types (only types that exist)
// ═══════════════════════════════════════════════════════════════════════════════

// Re-exports from symthaea-core consciousness modules
pub use symthaea_core::hdc::consciousness_topology_generators::{ConsciousnessTopology, TopologyType};
pub use symthaea_core::hdc::relational_consciousness::RelationalConsciousness;
pub use symthaea_core::hdc::meta_consciousness::MetaConsciousness;
pub use symthaea_core::hdc::global_workspace::GlobalWorkspace;
pub use symthaea_core::hdc::text_encoder::{TextEncoder, TextEncoderConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Subdirectory modules (stable)
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC arithmetic operations (addition, multiplication in HDC space).
pub mod arithmetic;

/// Consciousness-specific HDC computations.
pub mod consciousness;

/// Tiered phi computation with performance tiers.
pub mod tiered_phi;

/// Phi measurement utilities.
pub mod phi;

/// HDC-LTC neurons with CfC closed-form predict_forward.
pub mod hdc_ltc_neuron;

/// Phenomenal Binding Study - Research Direction 2: HDC binding vs bundling.
pub mod phenomenal_binding_study;

/// Binding Reversibility Study - H2 Revised: Testing binding via reversibility.
pub mod binding_reversibility_study;

/// HD-LTC codec: bidirectional translation between HDC and LTC spaces.
pub mod hd_ltc_codec;

/// LTC generative core: generative thought engine using HD-LTC codec.
pub mod ltc_generative_core;

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Testing
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
pub mod proptest_hdc;
