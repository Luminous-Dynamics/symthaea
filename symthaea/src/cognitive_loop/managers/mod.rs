// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! | [`SoulManager`] | value alignment, dissonance, coherence | 43 | Schwartz (1992), Festinger (1957) |
//! | [`MemoryManager`] | episodic, semantic, resonator, coordinator | 11 | Tulving (2002), Cowan (2001) |
//! | [`LearningManager`] | FEP, dream, school, evolution | 13 | Friston (2010), Walker (2017) |
//! | [`PerceptionManager`] | attention, multi-modal, social | 19 | Posner (1980), Baron-Cohen (1995) |
//! | [`GovernanceManager`] | governance events, neuromod contagion | 37 | Schultz (1997), Zak (2012) | `mycelix` |
//! | [`FabricationManager`] | Cincinnati quality, twin readings, PoGF | 47 | Aston-Jones (2005), Schultz (1997) | `advanced-manufacturing` |
//! | [`SentinelManager`] | governance anomaly, Sybil, dispatch loop | 67 | Aston-Jones & Cohen (2005) | `sentinel` |
//! | [`VisionManager`] | visual surprise, habituation, saccade urgency | 17 | Itti & Koch (2001), Rankin (2009) | `vision-manifold` |
//! | [`LanguageManager`] | Broca quality, coherence gating, consciousness cadence | 61 | Clark (2013), Hagoort (2005) | `ssm_language` |
//! | [`ReasoningManager`] | reasoning reliability, epistemic gating, prediction trend | 73 | Stanovich (2011), Koriat (2007) | `reasoning_engine` |
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
pub mod embodiment_transfer;
pub mod learning_manager;
pub mod memory_manager;
pub mod network_service_bridge;
pub mod perception_manager;
pub mod soul_manager;
pub mod swarm_consciousness;
pub mod swarm_manager;
pub mod thermodynamic_manager;

#[cfg(feature = "mycelix")]
pub mod governance_manager;

#[cfg(feature = "mesh")]
pub mod radio_dispatcher;

pub use drive_manager::DriveManager;
pub use learning_manager::LearningManager;
pub use memory_manager::MemoryManager;
pub use network_service_bridge::{
    forward_affective_state, forward_federated_round, NetworkServiceBridge,
    NetworkServiceBridgeHandle,
};
pub use perception_manager::PerceptionManager;
pub use soul_manager::{SoulManager, SoulTelemetry};
pub use swarm_manager::SwarmManager;

#[cfg(feature = "mycelix")]
pub use governance_manager::GovernanceManager;

#[cfg(feature = "mesh")]
pub use radio_dispatcher::{
    CompressedDelta, NetworkHealth, PayloadClass, PayloadClassifier, RadioTier, RoutingDecision,
    SpectrumManager, SpectrumObservation, SpectrumTelemetry,
};

#[cfg(feature = "cpg")]
pub mod cpg_manager;
#[cfg(feature = "cpg")]
pub use cpg_manager::{CpgConfig, CpgManager, CpgTelemetry, GaitPreset};

#[cfg(feature = "spectral_state")]
pub mod spectral_manager;
#[cfg(feature = "spectral_state")]
pub use spectral_manager::{SpectralManager, SpectralManagerConfig, SpectralTelemetry};

#[cfg(feature = "glyph_codex")]
pub mod glyph_manager;

#[cfg(feature = "therapeutic")]
pub mod therapeutic_dream_bridge;
#[cfg(feature = "therapeutic")]
pub mod therapeutic_manager;
#[cfg(feature = "glyph_codex")]
pub use glyph_manager::GlyphManager;

#[cfg(feature = "therapeutic")]
pub use therapeutic_dream_bridge::DreamableTherapeuticAction;
#[cfg(feature = "therapeutic")]
pub use therapeutic_manager::TherapeuticManager;

#[cfg(feature = "advanced-manufacturing")]
pub mod fabrication_manager;

#[cfg(feature = "sentinel")]
pub mod sentinel_manager;

#[cfg(feature = "neuroevolution")]
pub mod neuroevolution_manager;
#[cfg(feature = "advanced-manufacturing")]
pub use fabrication_manager::{
    FabricationEvent, FabricationEventKind, FabricationManager, FabricationTelemetry,
};

#[cfg(feature = "sentinel")]
pub(crate) use sentinel_manager::{
    SentinelEvent, SentinelManager, SentinelTelemetry, ThreatSignal, ThreatSignalKind,
};

#[cfg(feature = "neuroevolution")]
pub use neuroevolution_manager::{NeuroevolutionManager, NeuroevolutionTelemetry};

#[cfg(feature = "muse")]
pub mod music_publisher;
#[cfg(feature = "muse")]
pub use music_publisher::{MusicPublisher, UploadResult};

#[cfg(feature = "ssm_language")]
pub mod language_manager;
#[cfg(feature = "ssm_language")]
pub use language_manager::LanguageManager;

#[cfg(feature = "reasoning_engine")]
pub mod reasoning_manager;
#[cfg(feature = "reasoning_engine")]
pub use reasoning_manager::ReasoningManager;

// Sovereign Inoculation managers
#[cfg(feature = "mesh")]
pub mod time_manager;
#[cfg(feature = "mesh")]
pub use time_manager::TimeManager;

#[cfg(feature = "mesh-trust")]
pub mod trust_manager;
#[cfg(feature = "mesh-trust")]
pub use trust_manager::TrustManager;

#[cfg(feature = "social-fabric")]
pub mod social_fabric_manager;
#[cfg(feature = "social-fabric")]
pub use social_fabric_manager::SocialFabricManager;

#[cfg(feature = "circular")]
pub mod waste_manager;
#[cfg(feature = "circular")]
pub use waste_manager::WasteManager;

#[cfg(feature = "digital_metabolism")]
pub mod metabolism_manager;
#[cfg(feature = "digital_metabolism")]
pub use metabolism_manager::MetabolicState;

#[cfg(feature = "survival")]
pub mod survival_manager;
#[cfg(feature = "survival")]
pub use survival_manager::SurvivalManager;

#[cfg(feature = "vision-manifold")]
pub mod vision_manager;
#[cfg(feature = "vision-manifold")]
pub use vision_manager::VisionManager;

#[cfg(feature = "muse")]
pub mod muse_manager;
#[cfg(feature = "muse")]
pub use muse_manager::{MuseManager, MuseTelemetry};

#[cfg(feature = "hypervisor")]
pub mod hypervisor_manager;
#[cfg(feature = "hypervisor")]
pub use hypervisor_manager::{HypervisorManager, HypervisorTelemetry, SupervisoryEvent};
