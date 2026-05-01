// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! ## Binary Hypervectors ([`BinaryHV`])
//!
//! The primary type for efficient HDC operations. 16,384-bit vectors (2KB) with:
//! - 32x memory reduction vs `Vec<f32>`
//! - 200x faster operations via SIMD (XOR, popcount)
//! - Deterministic random generation from seeds
//!
//! ```rust,ignore
//! use symthaea::hdc::BinaryHV;
//!
//! let a = BinaryHV::random(42);  // Deterministic from seed
//! let b = BinaryHV::random(43);
//!
//! // Bind: create associations (XOR)
//! let bound = a.bind(&b);
//!
//! // Bundle: create prototypes (majority vote)
//! let prototype = BinaryHV::bundle(&[a, b]);
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
//! - **Core types**: [`BinaryHV`], [`ContinuousHV`]
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
//! use symthaea::hdc::{TextEncoder, TextEncoderConfig, BinaryHV};
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
//! use symthaea::hdc::BinaryHV;
//!
//! // Create unique basis vectors for nodes
//! let node_a = BinaryHV::basis(0);
//! let node_b = BinaryHV::basis(1);
//! let node_c = BinaryHV::basis(2);
//!
//! // Encode edges as bindings
//! let edge_ab = node_a.bind(&node_b);
//! let edge_bc = node_b.bind(&node_c);
//!
//! // Node representation = bundle of incident edges
//! let node_b_repr = BinaryHV::bundle(&[edge_ab, edge_bc]);
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
    binary_hv::BinaryHV,
    // Primitive system
    primitive_system::{Primitive, PrimitiveSystem, PrimitiveTier},
    // Unified hypervector types
    unified_hv::{ContinuousHV, HDC_DIMENSION, HV},
    // Causal types
    CausalDirection,
    // HDC context for arena-based operations
    HdcContext,
    NativeSimilarityIndex,
    // Native similarity with PackedBipolar
    PackedBipolar,
    // Tiered Phi calculation
    TieredPhi,
    // Real-valued hypervectors
    // RealHV alias available via real_hv module
    // LTC neuron count constants
    LTC_NEURONS,
};

// Re-export native_similarity module for tests
pub use symthaea_core::hdc::native_similarity;

// Re-export sensorimotor contingencies (enactivist perception)
pub use symthaea_core::hdc::sensorimotor_contingencies;

/// Convenient alias for the standard HDC dimension (16,384).
pub const HDC_DIM: usize = symthaea_core::hdc::unified_hv::HDC_DIMENSION;

// Phi engine types from symthaea-core
pub use symthaea_core::phi_engine::{PhiCalculator, PhiEngine, PhiMethod, PhiResult};

// Observability types from symthaea-core
pub use symthaea_core::observability::{
    metrics_collector, no_op_observer, DataPoint, MetricsCollector, MetricsSnapshot, NoOpObserver,
    Observer, PhiComponents, PhiMeasurementEvent, SharedObserver,
};

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS FROM SYMTHAEA-CORE - Full Modules
// ═══════════════════════════════════════════════════════════════════════════════

// Re-export entire modules for compatibility
pub use symthaea_core::hdc::adaptive_topology;
pub use symthaea_core::hdc::causal_mind;
pub use symthaea_core::hdc::collective_consciousness;
pub use symthaea_core::hdc::cross_modal_binding;
pub use symthaea_core::hdc::grounded_understanding;
pub use symthaea_core::hdc::hebbian;
pub use symthaea_core::hdc::predictive_coding;
pub use symthaea_core::hdc::sleep_and_altered_states;
pub use symthaea_core::hdc::unified_cognitive_core;
// simd_hv16 removed — all methods absorbed into BinaryHV
pub use symthaea_core::hdc::binary_hv;
pub use symthaea_core::hdc::cincinnati_advanced;
pub use symthaea_core::hdc::cincinnati_enhanced;
pub use symthaea_core::hdc::cincinnati_ltc;
pub use symthaea_core::hdc::cincinnati_network;
pub use symthaea_core::hdc::consciousness_integration;
pub use symthaea_core::hdc::consciousness_topology_generators;
pub use symthaea_core::hdc::global_workspace;
pub use symthaea_core::hdc::gwt_cincinnati_integration;
pub use symthaea_core::hdc::meta_consciousness;
pub use symthaea_core::hdc::phi_resonant;
pub use symthaea_core::hdc::primitive_system;
pub use symthaea_core::hdc::real_hv;
pub use symthaea_core::hdc::relational_consciousness;
pub use symthaea_core::hdc::reservoir;
pub use symthaea_core::hdc::semantic_decoder;
pub use symthaea_core::hdc::semantic_encoder;
pub use symthaea_core::hdc::simd_ops;
pub use symthaea_core::hdc::spectral_connectivity;
pub use symthaea_core::hdc::text_encoder;
pub use symthaea_core::hdc::unified_hv;
pub use symthaea_core::hdc::universal_semantics;
// tiered_phi is canonicalized in symthaea-core (single source of truth)
pub use symthaea_core::hdc::adaptive_learning_signals as core_adaptive_learning;
pub use symthaea_core::hdc::arithmetic_engine;
pub use symthaea_core::hdc::conscious_learning as core_conscious_learning;
pub use symthaea_core::hdc::consciousness_advanced_cognition as core_advanced_cognition;
pub use symthaea_core::hdc::consciousness_complete_being as core_complete_being;
pub use symthaea_core::hdc::consciousness_cross_integration as core_cross_integration;
pub use symthaea_core::hdc::consciousness_feedback_dynamics as core_feedback_dynamics;
pub use symthaea_core::hdc::consciousness_metacognition as core_metacognition;
pub use symthaea_core::hdc::consciousness_metacognitive;
pub use symthaea_core::hdc::consciousness_perf;
pub use symthaea_core::hdc::consciousness_persistence as core_consciousness_persistence;
pub use symthaea_core::hdc::consciousness_phi_optimization;
pub use symthaea_core::hdc::consciousness_self_awareness;
pub use symthaea_core::hdc::consciousness_streaming as core_streaming;
pub use symthaea_core::hdc::consciousness_subsystem;
pub use symthaea_core::hdc::consciousness_verifier;
pub use symthaea_core::hdc::counterfactual_dreams as core_counterfactual;
pub use symthaea_core::hdc::cross_modal_attention_router as core_attention_router;
pub use symthaea_core::hdc::ecosystem_bridge as core_ecosystem_bridge;
pub use symthaea_core::hdc::emotional_depth as core_emotional_depth;
pub use symthaea_core::hdc::full_stack_consciousness as core_full_stack;
pub use symthaea_core::hdc::integrated_information;
pub use symthaea_core::hdc::phi_feedback;
pub use symthaea_core::hdc::predictive_encoder as core_predictive_encoder;
pub use symthaea_core::hdc::self_improvement_integration as core_self_improvement;
pub use symthaea_core::hdc::semantic_bridge;
pub use symthaea_core::hdc::tiered_phi;
pub use symthaea_core::hdc::unified_conscious_being as core_unified_being;
pub use symthaea_core::hdc::unified_understanding as core_unified_understanding;

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS FROM SYMTHAEA-CORE - Mathematics
// ═══════════════════════════════════════════════════════════════════════════════

pub use symthaea_core::hdc::algebraic_structures;
pub use symthaea_core::hdc::calculus;
pub use symthaea_core::hdc::complex;
pub use symthaea_core::hdc::foundations;
pub use symthaea_core::hdc::integer;
pub use symthaea_core::hdc::math_bridge;
pub use symthaea_core::hdc::number_theory;
pub use symthaea_core::hdc::numeric_tower;
pub use symthaea_core::hdc::phi_guided_math;
pub use symthaea_core::hdc::phi_guided_search;
pub use symthaea_core::hdc::rational;
pub use symthaea_core::hdc::real_arithmetic;

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS FROM SYMTHAEA-CORE - Performance & Similarity Search
// ═══════════════════════════════════════════════════════════════════════════════

pub use symthaea_core::hdc::dynamical_system;
pub use symthaea_core::hdc::grid_encoder;
pub use symthaea_core::hdc::incremental_hv;
pub use symthaea_core::hdc::lsh_index;
pub use symthaea_core::hdc::lsh_simhash;
pub use symthaea_core::hdc::lsh_similarity;
pub use symthaea_core::hdc::parallel_hv;

pub use symthaea_core::hdc::hdc_ltc_unified as core_hdc_ltc_unified;

// Re-export unified HDC-LTC types (revolutionary closed-form dynamics)
pub use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig,
    UnifiedNetworkConfig, UnifiedNetworkStats, UnifiedNeuronStats,
};

// Re-export from causal_mind
pub use symthaea_core::hdc::causal_mind::{CausalDiscoveryResult, CausalFeatures};

// Re-export SemanticPrime
pub use symthaea_core::hdc::universal_semantics::SemanticPrime;

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - Convenience Types (only types that exist)
// ═══════════════════════════════════════════════════════════════════════════════

// Re-exports from symthaea-core consciousness modules
pub use symthaea_core::hdc::consciousness_topology_generators::{
    ConsciousnessTopology, TopologyType,
};
pub use symthaea_core::hdc::global_workspace::GlobalWorkspace;
pub use symthaea_core::hdc::meta_consciousness::MetaConsciousness;
pub use symthaea_core::hdc::relational_consciousness::RelationalConsciousness;
pub use symthaea_core::hdc::text_encoder::{TextEncoder, TextEncoderConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Subdirectory modules (stable)
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC arithmetic operations (addition, multiplication in HDC space).
pub mod arithmetic;

/// Consciousness-specific HDC computations.
pub mod consciousness;

// tiered_phi is re-exported from symthaea_core above

/// Phi measurement utilities.
pub mod phi;

/// HDC-LTC neurons with CfC closed-form predict_forward.
pub mod hdc_ltc_neuron;

/// Phenomenal Binding Study - Research Direction 2: HDC binding vs bundling.
pub mod phenomenal_binding_study;

/// Binding Reversibility Study - H2 Revised: Testing binding via reversibility.
pub mod binding_reversibility_study;

/// True IIT 3.0 exact computation: TPM, EMD, cause-effect repertoires, MIP, temporal Phi.
pub mod iit_exact;

/// Information-theoretic measures (Shannon entropy, mutual information, transfer
/// entropy) for HDC states. These underpin Phi measurement and causal analysis.
pub mod information_theory;

/// Geometric operations on the hypervector manifold (hypersphere S^{d-1}).
/// Provides geodesic distance, SLERP, exp/log maps, parallel transport,
/// Frechet mean, Riemannian gradient, and Principal Geodesic Analysis.
pub mod geometric_ops;

/// Probabilistic HDC: extends hypervectors with explicit uncertainty (mean + variance).
/// Enables Bayesian belief updating, confidence-weighted similarity, and
/// information-theoretic introspection --- the mathematical substrate for
/// Active Inference in hyperdimensional space.
pub mod probabilistic_hdc;

/// Tensor operations and geometric (Clifford) algebra for structured HDC.
/// Tensor products encode multi-way concept relationships; geometric algebra
/// provides rotation-invariant processing for spatial and attentional reasoning.
pub mod tensor_algebra;

/// Learned HDC encodings: adaptive level hypervectors, feature weighting,
/// spatial context, and prototype-based classification with iterative refinement.
pub mod learned_encoding;

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Code Understanding (Consciousness-Aware Code)
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC encoder for code AST structures.
#[cfg(feature = "code_generation")]
pub mod code_encoder;

/// HDC encoder for structured compiler diagnostics.
#[cfg(feature = "code_generation")]
pub mod diagnostic_encoder;

/// Semantic encoder with synonym-aware programming concept codebook.
#[cfg(feature = "code_generation")]
pub mod code_semantic_encoder;

/// Algebraic operations on code hypervectors (similarity, analogy, compose).
#[cfg(feature = "code_generation")]
pub mod code_algebra;

/// Project-level HDC memory for codebase indexing and retrieval.
#[cfg(feature = "code_generation")]
pub mod code_memory;

/// Bridge between BinaryHV and ContinuousHV for unified code synthesis pipeline.
#[cfg(feature = "code_generation")]
pub mod hv_bridge;

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Moral Reasoning
// ═══════════════════════════════════════════════════════════════════════════════

/// Moral algebra for compositional ethical reasoning via HDC.
///
/// Defines 7 moral primitives (AGENT, PATIENT, ACTION, INTENT, CONSENT,
/// OBLIGATION, MAGNITUDE) and 5 operators (CAUSES, VIOLATES, SATISFIES,
/// PROPORTIONAL, NEGATES) for algebraic moral reasoning.
pub mod moral_algebra;

/// Moral semantic parser for extracting ethical primitives from text.
///
/// Lightweight parser that extracts AGENT, ACTION, PATIENT, INTENT, CONSENT,
/// and MAGNITUDE from natural language scenarios. Uses pattern matching and
/// keyword analysis optimized for the ETHICS benchmark format.
pub mod moral_parser;

/// Character n-gram HDC text encoder for moral reasoning.
///
/// Deterministic trigram-binding encoder: each character is bound with a
/// positional HV, n-grams are bundled, and the result is L2-normalized.
/// (Named `moral_text_encoder` to avoid conflict with symthaea_core::hdc::text_encoder.)
pub mod moral_text_encoder;

/// Learned moral prototype classifier (Good/Neutral/Bad).
///
/// Trains 3-class HDC prototypes from labeled text using accumulate→retrain,
/// the same loop that achieved 87.6% on MNIST and 91.7% on ISOLET.
pub mod moral_prototypes;

/// Shared Eight Harmonies basis vectors and moral free energy (FEP).
pub mod harmony_basis;

/// Manifold-based moral classifier using 8D harmony basis centroids.
pub mod manifold_classifier;

/// CfC-based non-linear moral classifier using HdcLtcUnifiedNetwork.
pub mod cfc_moral_classifier;

/// Spinozist Moral Geometry — NSM-grounded affect space for moral reasoning.
pub mod spinozist_geometry;

/// Learned Moral Classifier — Spinozist features + adaptive HDC classification.
pub mod learned_moral_classifier;

/// Consciousness-driven text encoder — CfC word-by-word temporal dynamics.
pub mod consciousness_encoder;

/// Cognitive Loop Moral Classifier — full consciousness pipeline for moral features.
pub mod cognitive_moral_classifier;

/// Persistent homology on moral scenario hypervectors.
pub mod moral_topology;

/// HDC vector compression via TurboQuant (PolarQuant + QJL).
/// Compresses 16,384D ContinuousHV from 64KB to ~10KB for storage and transmission.
#[cfg(feature = "turbo-quant")]
pub mod hv_compression;

/// Glyph Codex: symbolic consciousness field basis vectors and registry.
#[cfg(feature = "glyph_codex")]
pub mod glyph_basis;

/// Embedded glyph registry data (70 entries from Primary Glyph Registry CSV).
#[cfg(feature = "glyph_codex")]
pub mod glyph_registry_data;

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Narrative Reasoning
// ═══════════════════════════════════════════════════════════════════════════════

/// Compositional narrative reasoning in hyperdimensional space.
///
/// Primitives (protagonist, setting, conflict, ...) and operators (causes,
/// transforms_into, escalates, ...) compose into scene-level hypervectors
/// that can be compared for coherence and fed into `StoryArcDynamics`.
pub mod narrative_algebra;

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL MODULES - Testing
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
pub mod proptest_hdc;
#[cfg(test)]
mod proptest_moral_algebra;
#[cfg(test)]
mod proptest_moral_topology;
