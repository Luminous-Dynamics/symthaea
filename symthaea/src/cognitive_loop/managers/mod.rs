//! # Cognitive Subsystem Managers
//!
//! Groups the 75+ subsystems in `CognitiveLoopService` into coherent managers,
//! each implementing the [`CognitiveSubsystem`] trait for proposal-based output.
//!
//! ## Current Managers
//!
//! | Manager | Subsystems | Interval | Science |
//! |---------|-----------|----------|---------|
//! | [`DriveManager`] | curiosity, flow, boredom, exploration | 7 | Berlyne (1960), Csikszentmihalyi (1990) |
//! | [`MemoryManager`] | episodic, semantic, resonator, coordinator | 11 | Tulving (2002), Cowan (2001) |
//! | [`LearningManager`] | FEP, dream, school, evolution | 13 | Friston (2010), Walker (2017) |
//! | [`PerceptionManager`] | attention, multi-modal, social | 19 | Posner (1980), Baron-Cohen (1995) |
//! | [`GovernanceManager`] | governance events, neuromod contagion | 37 | Schultz (1997), Zak (2012) | `mycelix` |
//!
//! ## Architecture
//!
//! Managers implement `CognitiveSubsystem::process(&CycleSnapshot) → SubsystemOutput`.
//! They run **alongside** existing inline code (dual-write bridge) until the old paths
//! are removed. The `OutputCollector` in cycle.rs integrates all manager proposals via
//! consensus averaging.
//!
//! ## WASM Readiness
//!
//! Managers use only `CycleSnapshot` (fixed-size, `#[repr(C)]`) as input and produce
//! `SubsystemOutput` (fixed-size, `#[repr(C)]`) as output. This makes them candidates
//! for future WASM boundary crossing.

pub mod drive_manager;
pub mod learning_manager;
pub mod memory_manager;
pub mod perception_manager;
pub mod swarm_manager;

#[cfg(feature = "mycelix")]
pub mod governance_manager;

pub use drive_manager::DriveManager;
pub use learning_manager::LearningManager;
pub use memory_manager::MemoryManager;
pub use perception_manager::PerceptionManager;
pub use swarm_manager::SwarmManager;

#[cfg(feature = "mycelix")]
pub use governance_manager::GovernanceManager;
