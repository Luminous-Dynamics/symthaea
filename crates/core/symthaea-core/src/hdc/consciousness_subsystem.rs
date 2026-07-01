// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Subsystem Trait
//!
//! Defines the `ConsciousnessSubsystem` trait for pluggable consciousness
//! processing components. This enables decomposition of the monolithic
//! `ConsciousnessPipeline` into modular, testable subsystems while maintaining
//! backward compatibility.

use std::any::Any;
use std::collections::HashMap;

use super::binary_hv::BinaryHV;
use super::consciousness_integration::ConsciousnessState;

/// Shared context for passing data between subsystems within a single cycle.
///
/// Each subsystem can publish named values that later subsystems (lower priority)
/// can read. The context is cleared at the start of each `process()` cycle.
///
/// # Examples
///
/// ```rust,ignore
/// // In a high-priority subsystem:
/// context.set("topology_embedding", my_embedding_vec);
///
/// // In a lower-priority subsystem:
/// if let Some(embedding) = context.get::<Vec<f32>>("topology_embedding") {
///     // use the embedding without recomputing
/// }
/// ```
pub struct SubsystemContext {
    data: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl SubsystemContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Store a value. Overwrites any previous value with the same key.
    pub fn set<T: Any + Send + Sync>(&mut self, key: &str, value: T) {
        self.data.insert(key.to_string(), Box::new(value));
    }

    /// Retrieve a value by key and type. Returns `None` if missing or wrong type.
    pub fn get<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.data.get(key).and_then(|v| v.downcast_ref::<T>())
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the context is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for SubsystemContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for subsystem processing failures.
#[derive(Debug, Clone)]
pub struct SubsystemError {
    /// Name of the subsystem that failed.
    pub subsystem: String,
    /// Human-readable error message.
    pub message: String,
}

impl std::fmt::Display for SubsystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "subsystem '{}': {}", self.subsystem, self.message)
    }
}

impl std::error::Error for SubsystemError {}

/// A pluggable consciousness subsystem that processes one aspect of consciousness.
///
/// Subsystems are registered with `ConsciousnessPipeline::register_subsystem()`
/// and are called during each `process()` cycle after the built-in systems.
/// They execute in priority order (higher priority first), and errors are
/// collected rather than halting the pipeline.
///
/// # Examples
///
/// ```rust,ignore
/// use symthaea_core::hdc::consciousness_subsystem::ConsciousnessSubsystem;
///
/// struct MySubsystem { enabled: bool }
///
/// impl ConsciousnessSubsystem for MySubsystem {
///     fn name(&self) -> &str { "my_subsystem" }
///     fn process_cycle(&mut self, state: &mut ConsciousnessState, inputs: &[BinaryHV])
///         -> Result<(), SubsystemError>
///     {
///         state.phi = (state.phi + 0.01).min(1.0);
///         Ok(())
///     }
///     fn is_enabled(&self) -> bool { self.enabled }
/// }
/// ```
pub trait ConsciousnessSubsystem: Send + Sync {
    /// Human-readable name of this subsystem.
    fn name(&self) -> &str;

    /// Process one cycle, mutating the consciousness state.
    ///
    /// Returns `Ok(())` on success, or a `SubsystemError` if processing fails.
    /// The pipeline collects errors and continues with remaining subsystems.
    fn process_cycle(
        &mut self,
        state: &mut ConsciousnessState,
        inputs: &[BinaryHV],
    ) -> Result<(), SubsystemError>;

    /// Whether this subsystem is currently active.
    fn is_enabled(&self) -> bool;

    /// Execution priority. Higher values run first. Default is 0.
    fn priority(&self) -> i32 {
        0
    }

    /// Called when the subsystem is registered with a pipeline.
    ///
    /// Override to perform one-time initialization.
    fn on_register(&mut self) {}

    /// Called when the pipeline is being shut down or the subsystem is removed.
    ///
    /// Override to release resources.
    fn on_shutdown(&mut self) {}

    /// Process one cycle with access to the inter-subsystem context.
    ///
    /// The default implementation ignores the context and delegates to
    /// [`process_cycle`]. Override this to publish or consume shared data
    /// between subsystems within a single cycle.
    fn process_cycle_with_context(
        &mut self,
        state: &mut ConsciousnessState,
        inputs: &[BinaryHV],
        _context: &mut SubsystemContext,
    ) -> Result<(), SubsystemError> {
        self.process_cycle(state, inputs)
    }
}
