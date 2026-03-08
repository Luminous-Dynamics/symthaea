//! Mycelix Core Types
//!
//! This crate provides the fundamental types used throughout the Mycelix ecosystem:
//!
//! - **K-Vector**: 8-dimensional trust representation
//! - **E/N/M/H Classification**: Epistemic classification system
//! - **Harmonic Types**: Eight Harmonies integration for GIS v4
//! - **Moral Uncertainty**: Tripartite moral uncertainty model
//! - **WisdomEngine**: Full Stack Wisdom / Holistic Epistemics architecture with
//!   advanced features including Temporal Pattern Decay, Confidence Calibration,
//!   Counterfactual Reasoning, Active Experimentation, and Causal Discovery
//! - **Domain System**: Hierarchical knowledge organization with cross-domain discovery
//! - **Pattern Composition**: Combining patterns with synergy tracking and auto-discovery
//! - **CollectiveField**: Swarm wisdom and emergence detection
//!
//! These types are designed to be used across all Mycelix modules and integrate
//! with the Graceful Ignorance System (GIS) v4.0 "Kosmic Song".
//!
//! ## WisdomEngine: Holistic Epistemics
//!
//! The WisdomEngine provides the most advanced epistemic architecture for decentralized
//! networks, combining four components:
//!
//! 1. **The Lens** - Harmonic-weighted scoring through the Eight Harmonies
//! 2. **The Setting** - Community-specific epistemic profiles
//! 3. **The Mirror** - Diversity metrics and bias detection
//! 4. **The Hand** - Epistemic reparations for marginalized voices
//!
//! Plus emergent weight learning and multi-epistemology support.
//!
//! ## SymthaeaCausalBridge: Scientific Method for AI
//!
//! The SymthaeaCausalBridge implements a complete scientific method cycle for
//! AI pattern learning, now with fully integrated Component 9 enhancements:
//!
//! ```rust,ignore
//! use mycelix_core_types::{SymthaeaCausalBridge, SymthaeaPattern, OracleVerificationLevel};
//!
//! // Create a bridge - Component 9 enhancements are automatically initialized
//! let mut bridge = SymthaeaCausalBridge::new();
//!
//! // Learn a pattern (HYPOTHESIS phase)
//! let pattern = SymthaeaPattern::new(1, "web_dev", "use_caching", 0.8, 1000, 1);
//! bridge.on_pattern_learned(pattern);
//!
//! // Use the pattern (EXPERIMENT phase)
//! // This automatically:
//! //   - Tracks calibration (prediction vs outcome)
//! //   - Records usage for causal discovery
//! //   - Runs auto-discovery when threshold reached
//! bridge.on_pattern_used(1, true, 2000, "production deploy", OracleVerificationLevel::Audited);
//!
//! // Get a quick health report (ANALYZE phase)
//! let report = bridge.quick_health_report(3000);
//! println!("Calibration quality: {:.2}", report.calibration_quality);
//! println!("Discovered dependencies: {}", report.discovered_dependencies);
//!
//! // Find the best pattern for a domain, considering temporal decay
//! if let Some(best) = bridge.best_pattern_for_domain("web_dev", 3000) {
//!     println!("Best pattern: {} (effective rate: {:.2})",
//!         best.pattern_id, best.effective_success_rate);
//! }
//! ```
//!
//! ## Domain System: Hierarchical Knowledge Organization
//!
//! Component 10 adds a comprehensive domain hierarchy for organizing patterns
//! and enabling cross-domain discovery:
//!
//! ```rust,ignore
//! use mycelix_core_types::{SymthaeaCausalBridge, SymthaeaPattern, DomainPath};
//!
//! let mut bridge = SymthaeaCausalBridge::new();
//!
//! // Register domains using path strings (auto-creates hierarchy)
//! let web_id = bridge.register_domain("engineering/web_dev", 1000);
//! let backend_id = bridge.register_domain("engineering/backend", 1001);
//!
//! // Create patterns with domain associations
//! let mut pattern = SymthaeaPattern::new(1, "web_dev", "use_caching", 0.8, 1000, 1);
//! pattern.add_domain(web_id);
//! bridge.on_pattern_learned(pattern);
//!
//! // Query patterns by domain (with optional related domains)
//! let web_patterns = bridge.patterns_in_domain(web_id, false);
//! let eng_patterns = bridge.patterns_in_domain(
//!     bridge.domain_registry.find_by_name("engineering").unwrap(),
//!     true  // include_related: finds patterns in child domains too
//! );
//!
//! // Domain similarity enables cross-domain causal discovery
//! // Sibling domains (web_dev, backend) have similarity 0.64
//! // Parent-child domains have similarity 0.8
//! let similarity = bridge.domain_registry.similarity(web_id, backend_id);
//! ```
//!
//! ## Pattern Composition: Combining Patterns for Synergy
//!
//! Component 11 enables combining multiple patterns to achieve synergistic effects:
//!
//! ```rust,ignore
//! use mycelix_core_types::{SymthaeaCausalBridge, SymthaeaPattern, CompositionType};
//!
//! let mut bridge = SymthaeaCausalBridge::new();
//!
//! // Add some patterns
//! bridge.on_pattern_learned(SymthaeaPattern::new(1, "web", "use_caching", 0.8, 1000, 1));
//! bridge.on_pattern_learned(SymthaeaPattern::new(2, "web", "use_indexing", 0.8, 1000, 1));
//!
//! // Create a composite: caching + indexing together
//! let composite_id = bridge.create_composite(
//!     "cache_with_index",
//!     vec![1, 2],
//!     CompositionType::Parallel,  // Apply both together
//!     2000,
//!     1,
//! );
//!
//! // Track composite usage and outcomes
//! bridge.on_composite_used(composite_id, true, 3000);  // Success!
//!
//! // The synergy score reveals if combination beats individuals:
//! // synergy_score = actual_success_rate / expected_success_rate
//! // > 1.0 = synergy (combination works better!)
//! // < 1.0 = interference (patterns conflict)
//!
//! // Track which patterns are used together for automatic discovery
//! bridge.on_patterns_used_together(&[1, 2], true, 4000);
//!
//! // Discover synergy candidates automatically
//! let candidates = bridge.discover_synergies();
//!
//! // Auto-create composites from high-synergy candidates
//! let new_composites = bridge.auto_create_composites(5000, 1);
//! ```
//!
//! ## CollectiveField: Swarm Wisdom
//!
//! The CollectiveField module enables sensing of collective intelligence:
//!
//! - **Coherence**: How aligned is the group's attention?
//! - **Emergence**: Is collective insight crystallizing?
//! - **Tension**: Where are the productive growth edges?
//! - **Wisdom Density**: Has something profound emerged?

pub mod k_vector;
pub mod epistemic;
pub mod harmonic;
pub mod moral;
pub mod trust;
pub mod wisdom_engine;
pub mod collective_field;

pub use k_vector::*;
pub use epistemic::*;
pub use harmonic::*;
pub use moral::*;
pub use trust::*;
pub use wisdom_engine::*;
pub use collective_field::*;

/// Mycelix protocol version
pub const PROTOCOL_VERSION: &str = "2.0";

/// Maximum Byzantine tolerance (45% of participants)
/// From MATL specification - system cannot guarantee safety beyond this
pub const MAX_BYZANTINE_TOLERANCE: f32 = 0.45;

/// Phi threshold levels for governance
pub mod phi_thresholds {
    /// Basic proposals (routine decisions)
    pub const BASIC: f32 = 0.3;
    /// Major proposals (significant changes)
    pub const MAJOR: f32 = 0.4;
    /// Constitutional proposals (fundamental changes)
    pub const CONSTITUTIONAL: f32 = 0.6;
}

// ==============================================================================
// Python Bindings (PyO3)
// ==============================================================================
//
// When compiled with the `python` feature, this crate can be used as a Python
// extension module. Use `maturin build --features python` to build the wheel.

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module entry point for mycelix_wisdom
#[cfg(feature = "python")]
#[pymodule]
fn mycelix_wisdom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    wisdom_engine::python_bindings::register_wisdom_classes(m)?;

    // Module metadata
    m.add("__doc__", "Mycelix WisdomEngine - Pattern intelligence for AI systems")?;
    m.add("__version__", PROTOCOL_VERSION)?;

    Ok(())
}
