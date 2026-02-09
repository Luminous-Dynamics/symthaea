//! Consciousness Subsystem Trait
//!
//! Defines the `ConsciousnessSubsystem` trait for pluggable consciousness
//! processing components. This enables decomposition of the monolithic
//! `ConsciousnessPipeline` into modular, testable subsystems while maintaining
//! backward compatibility.

use super::binary_hv::BinaryHV;
use super::consciousness_integration::ConsciousnessState;

/// A pluggable consciousness subsystem that processes one aspect of consciousness.
///
/// Subsystems are registered with `ConsciousnessPipeline::register_subsystem()`
/// and are called during each `process()` cycle after the built-in systems.
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
///     fn process_cycle(&mut self, state: &mut ConsciousnessState, inputs: &[BinaryHV]) {
///         state.phi = (state.phi + 0.01).min(1.0);
///     }
///     fn is_enabled(&self) -> bool { self.enabled }
/// }
/// ```
pub trait ConsciousnessSubsystem: Send + Sync {
    /// Human-readable name of this subsystem.
    fn name(&self) -> &str;

    /// Process one cycle, mutating the consciousness state.
    fn process_cycle(&mut self, state: &mut ConsciousnessState, inputs: &[BinaryHV]);

    /// Whether this subsystem is currently active.
    fn is_enabled(&self) -> bool;
}
