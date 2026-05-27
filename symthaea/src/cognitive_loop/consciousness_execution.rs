// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Execution — extracted from CognitiveLoopService
//!
//! Groups the 5 consciousness measurement and self-model subsystems that were
//! previously individual fields on `CognitiveLoopService`:
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `consciousness_engine` | `ConsciousnessEngine` | Unified 4-layer measurement (SpectralMIP, MultiModal, EqV2, Pipeline) |
//! | `consciousness_monitors` | `ConsciousnessMonitorTier` | 7 optional analyzers (resonance, quantum, temporal, embodied, thermo, binding, HFE) |
//! | `gwt_mgr` | `GwtManager` | Global Workspace Theory workspace + flags |
//! | `self_model_tier` | `SelfModelTierManager` | Narrative/predictive/attention/meta-cognitive self-models |
//! | `master_equation` | `MasterConsciousnessEquation` | MCE: comprehensive consciousness metric C(t) |
//!
//! ## Design
//!
//! - **Zero behavior change**: Pure structural refactor, same field paths via delegation.
//! - **Public API preserved**: All 292+ accessor methods on `CognitiveLoopService` still work.
//! - **Field access**: Internal code accesses via `self.consciousness.field_name`.

use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;

use super::CognitiveLoopConfig;
use super::consciousness_engine;
use super::consciousness_monitor_tier;
use super::gwt_manager;
use super::self_model_tier;

/// Groups 5 consciousness measurement and self-model subsystems.
///
/// Extracted from `CognitiveLoopService` to reduce its field count
/// and provide a cohesive consciousness execution boundary.
pub(crate) struct ConsciousnessExecution {
    /// Unified Consciousness Engine: wraps SpectralMIP + MultiModal + EquationV2 + Pipeline
    /// into a single `measure()` call per cycle with co-prime interval scheduling.
    /// Runs **alongside** existing inline code (additive wiring — old code not removed yet).
    pub consciousness_engine: consciousness_engine::ConsciousnessEngine,

    /// Consciousness monitoring tier: resonance, quantum, temporal, embodied,
    /// thermodynamics, phenomenal binding, hierarchical free energy.
    pub consciousness_monitors: consciousness_monitor_tier::ConsciousnessMonitorTier,

    /// GWT manager: workspace + memory flag + perception counter.
    pub gwt_mgr: gwt_manager::GwtManager,

    /// Self-model subsystems: narrative, predictive, attention schema, meta-cognition.
    pub self_model_tier: self_model_tier::SelfModelTierManager,

    /// Master Consciousness Equation (MCE) — comprehensive consciousness metric.
    /// C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × weighted_sum × S × ρ(t) × M × N × Soc
    /// Runs every 10th cycle to provide richer consciousness measurement than Phi alone.
    pub master_equation: MasterConsciousnessEquation,
}

impl ConsciousnessExecution {
    /// Construct from individually built components.
    ///
    /// Called from `CognitiveLoopService::new()` after each subsystem is created.
    pub fn new(
        consciousness_engine: consciousness_engine::ConsciousnessEngine,
        consciousness_monitors: consciousness_monitor_tier::ConsciousnessMonitorTier,
        gwt_mgr: gwt_manager::GwtManager,
        self_model_tier: self_model_tier::SelfModelTierManager,
        master_equation: MasterConsciousnessEquation,
    ) -> Self {
        Self {
            consciousness_engine,
            consciousness_monitors,
            gwt_mgr,
            self_model_tier,
            master_equation,
        }
    }
}
