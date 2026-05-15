// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Scaffolded for upcoming wiring — callers not yet connected
#![allow(dead_code)]

//! # CognitiveSubsystem Trait — WASM-Ready Subsystem Interface
//!
//! Universal interface for cognitive subsystems within the cognitive loop.
//! Every subsystem (consciousness engine, ethics engine, memory manager, etc.)
//! implements this trait, enabling:
//!
//! - **Proposal-based computation**: Subsystems observe an immutable snapshot,
//!   return proposed state changes. No direct field mutation.
//! - **Attribution**: Every proposal is tagged with the subsystem's name.
//! - **WASM readiness**: `CycleSnapshot` and `SubsystemOutput` use `#[repr(C)]`
//!   with fixed-size buffers — no references, no lifetimes, no heap allocation
//!   crosses the boundary.
//! - **Hot-swap potential**: Subsystems compiled to WASM could be loaded via
//!   wasmtime and swapped at runtime without recompilation.
//!
//! ## Current Mode
//!
//! All subsystems are native Rust trait objects. The `#[repr(C)]` data structures
//! are designed for future WASM boundary crossing but incur no overhead today.
//!
//! ## Data Flow
//!
//! ```text
//! Phase A: OBSERVE  — Build CycleSnapshot from current state (immutable)
//! Phase B: COMPUTE  — Each subsystem calls process(snapshot) → SubsystemOutput
//! Phase C: INTEGRATE — Consensus-average all outputs → single state update
//! Phase D: DECAY    — EMA smoothing, clamping, homeostasis, urgency transitions
//! ```
//!
//! ## Integration with FeedbackState (Phase 2.2)
//!
//! `SubsystemOutput` proposals map directly to `FeedbackProposal`:
//! - `confidence_delta` → `FeedbackProposal::Add(delta)` on confidence collector
//! - `lr_modulation`    → `FeedbackProposal::Scale(factor)` on LR collector
//! - `exploration_delta` / `arousal_delta` → applied during Phase C integration

/// Maximum compressed state dimension for WASM boundary crossing.
///
/// Matches the default CfC `num_neurons` (256). Native subsystems can access
/// the full Vec<f32> via the `NativeSubsystemExt` extension trait.
pub const SNAPSHOT_STATE_DIM: usize = 256;

/// BinaryHV size in bytes (16,384 bits = 2,048 bytes).
///
/// This is the canonical HDC dimensionality used throughout Symthaea.
/// BinaryHV is already `#[repr(align(8))]` with `[u8; 2048]`.
pub const SNAPSHOT_HV_BYTES: usize = 2048;

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE SNAPSHOT — Immutable observation of cycle state
// ═══════════════════════════════════════════════════════════════════════════════

/// Immutable snapshot of cycle state passed to all subsystems.
///
/// This is the **sole input** to `CognitiveSubsystem::process()`. It captures
/// the complete observable state at the start of a cycle, before any subsystem
/// has mutated anything.
///
/// # WASM Safety
///
/// - `#[repr(C)]` — deterministic layout, passable as raw pointer across FFI
/// - Fixed-size arrays only — no `Vec`, no `String`, no trait objects
/// - All fields are `Copy` — no heap allocation
/// - `compressed_state` is clamped to 256 elements (full CfC hidden state)
/// - `input_hv` is the raw 2048-byte BinaryHV representation
///
/// # For Native Subsystems
///
/// Native (non-WASM) subsystems that need the full `Vec<f32>` compressed state,
/// the original `&str` input text, or other Rust-heap data can use the
/// `NativeSnapshotExt` extension, which wraps a `CycleSnapshot` with references
/// to the full data.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CycleSnapshot {
    // ── Identity ────────────────────────────────────────────────────────
    /// Current cycle number (monotonically increasing from 0).
    pub cycle_number: u64,

    // ── Core Pipeline State ─────────────────────────────────────────────
    /// Prediction confidence (0.0–1.0). Grows with accurate predictions,
    /// decays during uncertain states.
    pub prediction_confidence: f64,

    /// FEP-driven learning rate boost (1.0–3.0). Multiplicative factor
    /// applied during CfC training step.
    pub fep_lr_boost: f64,

    /// Prediction error from last cycle (cosine distance between predicted
    /// and actual CfC state). Higher = more surprise.
    pub prediction_error: f32,

    /// Conversation coherence (0.0–1.0). Smoothed EMA of semantic similarity
    /// between consecutive inputs.
    pub coherence: f32,

    // ── Consciousness Measures ──────────────────────────────────────────
    /// Unified Psi (Ψ) — integrated information measure combining spectral
    /// MIP Phi, cross-modal Phi, and body modulations.
    pub unified_psi: f64,

    /// Phi attention weight (0.0–2.0). Adaptive gating from phi_attention
    /// module; modulates perception sensitivity.
    pub phi_attention_weight: f32,

    // ── Affective State ─────────────────────────────────────────────────
    /// Emotional arousal (0.0–1.0). High arousal = urgent/excited processing.
    pub arousal: f32,

    /// Emotional valence (-1.0 to 1.0). Positive = pleasant, negative = aversive.
    pub valence: f32,

    // ── Somatic / Thermodynamic ─────────────────────────────────────────
    /// Thermodynamic load (0.0–1.0). Approaches 1.0 as computational budget
    /// is exhausted (6W metabolic limit analogy).
    pub thermodynamic_load: f32,

    /// Dissipative consciousness health (0.0–1.0). Prigogine thermodynamic
    /// model for consciousness; low values dampen learning.
    pub dissipative_health: f64,

    /// Somatic stress from infrastructure errors (0.0–1.0).
    /// Fed by SomaticErrorBridge: lock poisoning, task panics, DB failures.
    pub somatic_stress: f64,

    // ── Urgency & Scheduling ────────────────────────────────────────────
    /// Current urgency level. Determines how many subsystems run.
    /// 0 = Cruise (stable, skip expensive), 1 = Normal, 2 = Critical (run all).
    pub urgency: u8,

    /// Whether the attention budget was exceeded this cycle (subsystems
    /// should be more conservative about computation).
    pub attention_budget_exceeded: u8, // bool as u8 for repr(C)

    // ── Fixed-Size Buffers (WASM-safe) ──────────────────────────────────
    /// Compressed CfC hidden state (256 floats). This is the temporal
    /// network's internal representation of the current cognitive state.
    /// Padded with zeros if actual state dim < 256.
    pub compressed_state: [f32; SNAPSHOT_STATE_DIM],

    /// Raw BinaryHV of the current input (2048 bytes = 16,384 bits).
    /// This is the HDC encoding of the input before CfC temporal processing.
    pub input_hv: [u8; SNAPSHOT_HV_BYTES],

    // ── Cached Consciousness Scores ─────────────────────────────────────
    /// Last phenomenal binding strength (0.0–1.0). Measures phase coherence
    /// across consciousness dimensions for unified experience quality.
    pub phenomenal_binding: f64,

    /// Last harmonic field coherence (0.0–1.0). Geometric mean of all
    /// Seven Harmonics strengths.
    pub harmonic_coherence: f64,

    /// Last holographic unity score (0.0–1.0). Interference-based binding
    /// measure from the holographic analyzer.
    pub holographic_unity: f64,

    /// Last gradient magnitude from differentiable consciousness (0.0+).
    /// High gradient = consciousness bottleneck identified.
    pub gradient_magnitude: f64,

    /// Epistemic gate confidence (0.0–1.0). From epistemic quality
    /// classification; gates factual assertions.
    pub epistemic_confidence: f32,

    /// Moral score from the latest ethics evaluation (-1.0 to 1.0).
    pub moral_score: f32,

    /// Whether there are active external multimodal requests (e.g. video gen).
    /// bool as u8 for repr(C) compatibility.
    pub active_external_requests: u8,

    /// Current vision manifold HDC dimension (16384 or 65536).
    pub vision_hdc_dim: u32,

    /// Last visual Variational Free Energy.
    pub vision_free_energy: f32,
    /// Last visual model complexity.
    pub vision_complexity: f32,
    /// Last visual prediction accuracy.
    pub vision_accuracy: f32,

    // ── Padding for alignment ───────────────────────────────────────────
    /// Reserved for future fields. Ensures struct size is stable across
    /// versions for WASM ABI compatibility.
    pub _reserved: [u8; 3],
}

impl Default for CycleSnapshot {
    fn default() -> Self {
        Self {
            cycle_number: 0,
            prediction_confidence: 0.5,
            fep_lr_boost: 1.0,
            prediction_error: 0.0,
            coherence: 0.5,
            unified_psi: 0.0,
            phi_attention_weight: 1.0,
            arousal: 0.5,
            valence: 0.0,
            thermodynamic_load: 0.0,
            dissipative_health: 1.0,
            somatic_stress: 0.0,
            urgency: 1, // Normal
            attention_budget_exceeded: 0,
            compressed_state: [0.0; SNAPSHOT_STATE_DIM],
            input_hv: [0; SNAPSHOT_HV_BYTES],
            phenomenal_binding: 0.5,
            harmonic_coherence: 0.0,
            holographic_unity: 0.0,
            gradient_magnitude: 0.0,
            epistemic_confidence: 0.5,
            moral_score: 0.0,
            active_external_requests: 0,
            vision_hdc_dim: 16384,
            vision_free_energy: 0.0,
            vision_complexity: 0.0,
            vision_accuracy: 1.0,
            _reserved: [0; 3],
        }
    }
}

impl CycleSnapshot {
    /// Convert urgency byte back to CycleUrgency enum.
    #[inline]
    pub fn cycle_urgency(&self) -> super::CycleUrgency {
        match self.urgency {
            0 => super::CycleUrgency::Cruise,
            2 => super::CycleUrgency::Critical,
            _ => super::CycleUrgency::Normal,
        }
    }

    /// Whether the system is in Critical urgency (run all subsystems).
    #[inline]
    pub fn is_critical(&self) -> bool {
        self.urgency == 2
    }

    /// Whether attention budget is exceeded (should conserve computation).
    #[inline]
    pub fn budget_exceeded(&self) -> bool {
        self.attention_budget_exceeded != 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NATIVE SNAPSHOT EXTENSION — Full Rust data for native subsystems
// ═══════════════════════════════════════════════════════════════════════════════

/// Extension wrapping a `CycleSnapshot` with references to full Rust-heap data.
///
/// Native (non-WASM) subsystems receive this instead of the bare snapshot when
/// they need access to variable-length data (full compressed state, input text,
/// BinaryHV object methods, etc.).
///
/// WASM subsystems only see the `CycleSnapshot` portion.
#[derive(Debug)]
pub struct NativeSnapshot<'a> {
    /// The WASM-safe core snapshot (all fixed-size scalars and buffers).
    pub core: &'a CycleSnapshot,

    /// Full compressed state vector (may be longer than 256 if CfC dim > 256).
    pub full_compressed_state: &'a [f32],

    /// Original input text for this cycle.
    pub input_text: &'a str,

    /// Reference to the full BinaryHV object (for HDC operations like
    /// cosine similarity, binding, etc.).
    pub input_binary_hv: &'a symthaea_core::hdc::BinaryHV,

    /// CfC network output (prediction) for this cycle.
    pub prediction: &'a [f32],
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSYSTEM OUTPUT — Proposed state changes
// ═══════════════════════════════════════════════════════════════════════════════

/// Output flags bitfield constants for `SubsystemOutput::flags`.
pub mod output_flags {
    /// Subsystem requests exploration (novelty-seeking behavior).
    pub const REQUEST_EXPLORATION: u32 = 1 << 0;
    /// Subsystem requests consolidation (memory stabilization).
    pub const REQUEST_CONSOLIDATION: u32 = 1 << 1;
    /// Subsystem detected an anomaly (metacognitive signal).
    pub const ANOMALY_DETECTED: u32 = 1 << 2;
    /// Subsystem vetoes the current action (safety/ethics concern).
    pub const VETO_ACTION: u32 = 1 << 3;
    /// Subsystem suggests entering rest/recovery mode.
    pub const REQUEST_REST: u32 = 1 << 4;
    /// Subsystem has telemetry to report (check extended output).
    pub const HAS_TELEMETRY: u32 = 1 << 5;
    /// Subsystem wants to trigger a GWT broadcast.
    pub const REQUEST_BROADCAST: u32 = 1 << 6;
    /// Subsystem recommends urgency escalation.
    pub const ESCALATE_URGENCY: u32 = 1 << 7;
    /// Subsystem requests geodesic mental simulation (pathfinding).
    pub const REQUEST_GEODESIC: u32 = 1 << 8;
}

/// Proposed state changes from a single subsystem.
///
/// This is the **sole output** of `CognitiveSubsystem::process()`. All
/// proposals from all subsystems are collected and integrated via consensus
/// averaging in Phase C.
///
/// # WASM Safety
///
/// - `#[repr(C)]` — deterministic layout for FFI
/// - All fields are `Copy` — no heap allocation
/// - Proposals are *deltas*, not absolute values — safe to combine
///
/// # Integration with FeedbackState
///
/// During Phase C integration:
/// - `confidence_delta` → averaged across all subsystems, then applied additively
/// - `lr_modulation` → geometric mean across all subsystems, then applied multiplicatively
/// - `exploration_delta` → averaged, applied to exploration_urge
/// - `arousal_delta` → averaged, applied to emotion_contagion.arousal
/// - `flags` → OR'd together (any subsystem can set a flag)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SubsystemOutput {
    /// Additive proposal for prediction_confidence.
    /// Positive = "I'm more confident", negative = "I'm less confident".
    /// Averaged across all subsystems during integration.
    pub confidence_delta: f64,

    /// Multiplicative proposal for fep_lr_boost.
    /// 1.0 = no change. >1.0 = boost learning, <1.0 = dampen learning.
    /// Geometric mean across all subsystems during integration.
    pub lr_modulation: f64,

    /// Additive proposal for exploration_urge.
    /// Positive = "explore more", negative = "exploit more".
    pub exploration_delta: f64,

    /// Additive proposal for arousal level.
    /// Positive = "increase alertness", negative = "calm down".
    pub arousal_delta: f32,

    /// Additive proposal for valence.
    /// Positive = "more pleasant", negative = "more aversive".
    pub valence_delta: f32,

    /// Bitfield of boolean signals. OR'd across all subsystems.
    /// See `output_flags` module for flag constants.
    pub flags: u32,

    /// Reserved for future fields.
    pub _reserved: u32,
}

impl SubsystemOutput {
    /// An output that proposes no changes (identity element for integration).
    pub const NEUTRAL: Self = Self {
        confidence_delta: 0.0,
        lr_modulation: 1.0,
        exploration_delta: 0.0,
        arousal_delta: 0.0,
        valence_delta: 0.0,
        flags: 0,
        _reserved: 0,
    };

    /// Whether this output proposes any changes.
    #[inline]
    pub fn is_neutral(&self) -> bool {
        self.confidence_delta == 0.0
            && self.lr_modulation == 1.0
            && self.exploration_delta == 0.0
            && self.arousal_delta == 0.0
            && self.valence_delta == 0.0
            && self.flags == 0
    }

    /// Check if a specific flag is set.
    #[inline]
    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Set a specific flag.
    #[inline]
    pub fn with_flag(mut self, flag: u32) -> Self {
        self.flags |= flag;
        self
    }
}

impl Default for SubsystemOutput {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE SUBSYSTEM TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// The universal interface for cognitive subsystems.
///
/// Every subsystem in the cognitive loop implements this trait:
/// - Consciousness engine (spectral MIP, multi-modal, equation v2)
/// - Ethics engine (moral algebra, value evaluator, harmonies)
/// - Memory manager (episodic, semantic, resonator, coordinator)
/// - Learning manager (FEP, dream, school, evolution)
/// - Perception manager (attention, multi-modal, social)
/// - Drive manager (curiosity, flow, boredom, exploration)
/// - Somatic error bridge (infrastructure pain receptors)
///
/// # Design for WASM Boundary
///
/// - **Input**: `CycleSnapshot` — `#[repr(C)]`, fixed-size, passable as raw pointer
/// - **Output**: `SubsystemOutput` — `#[repr(C)]`, fixed-size, proposals only
/// - **No references cross the boundary**: state is owned, not borrowed
/// - **Checkpoint/restore**: serialize full state to bytes for migration
///
/// # Current Mode
///
/// All subsystems are native Rust trait objects called directly. The WASM-safe
/// data structures are "architecture only" — they add no runtime overhead but
/// ensure the trait can be implemented by WASM modules in the future.
///
/// # Co-prime Intervals
///
/// Each subsystem specifies its firing `interval()` in cycles. Intervals should
/// be co-prime to prevent phase-locking (multiple expensive subsystems firing
/// on the same cycle). The cognitive loop scheduler respects these intervals
/// and further modulates them by urgency level.
///
/// Common intervals: 1 (every cycle), 7, 11, 13, 19, 23, 29, 47, 97
pub trait CognitiveSubsystem: Send + Sync {
    /// Unique name for attribution, debugging, and telemetry.
    ///
    /// This string is used as the `source` field in `FeedbackProposal::AttributedProposal`
    /// and in `CycleMetadata` module timing keys.
    fn name(&self) -> &'static str;

    /// Firing interval in cycles (co-prime recommended).
    ///
    /// The subsystem's `process()` method is called every `interval()` cycles.
    /// Return 1 to run every cycle. The actual interval may be further modulated
    /// by the urgency level (e.g., doubled in Cruise mode).
    ///
    /// Guidelines for choosing intervals:
    /// - Core pipeline (HDC, CfC, predict, learn): 1 (always)
    /// - Fast feedback (affective, arousal): 1–3
    /// - Medium subsystems (consciousness measurement): 7–23
    /// - Expensive subsystems (phi validation, evolution): 47–97
    fn interval(&self) -> u32;

    /// Process one cycle step.
    ///
    /// Takes an immutable snapshot of the cycle state, returns proposed changes.
    /// The snapshot is built once in Phase A (OBSERVE) and shared with all
    /// subsystems. Proposals are collected and integrated in Phase C.
    ///
    /// # Contract
    ///
    /// - **Must not panic**: Return `SubsystemOutput::NEUTRAL` on internal errors
    /// - **Must be deterministic**: Given the same snapshot and internal state,
    ///   produce the same output (for genesis reproducibility)
    /// - **Should be fast**: Target <5ms per invocation; use `interval()` for
    ///   expensive computations
    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput;

    /// Serialize full internal state for checkpoint/migration.
    ///
    /// Returns an opaque byte buffer that can be passed to `restore()` to
    /// recreate the subsystem's exact state. Used for:
    /// - Hot-swap: serialize state, load new WASM module, restore state
    /// - Persistence: save/load cognitive state across restarts
    /// - Testing: snapshot state for deterministic replay
    ///
    /// Default: returns empty vec (stateless subsystem).
    fn checkpoint(&self) -> Vec<u8> {
        vec![]
    }

    /// Restore from serialized state.
    ///
    /// Inverse of `checkpoint()`. Must handle version mismatches gracefully
    /// (e.g., ignore unknown fields, use defaults for missing fields).
    ///
    /// Default: accepts any data (stateless subsystem).
    fn restore(&mut self, _data: &[u8]) -> Result<(), String> {
        Ok(())
    }

    /// Whether this subsystem should run at the given cycle number and urgency.
    ///
    /// Default implementation uses `interval()` with urgency-based modulation:
    /// - Critical: run at base interval
    /// - Normal: run at base interval
    /// - Cruise: run at 2× base interval (skip half)
    ///
    /// Override for custom scheduling logic (e.g., attention-budget gating).
    fn should_run(&self, cycle: u64, urgency: u8) -> bool {
        let interval = self.interval() as u64;
        if interval <= 1 {
            return true;
        }
        let effective_interval = if urgency == 0 {
            // Cruise: double the interval to conserve compute
            interval * 2
        } else {
            interval
        };
        cycle % effective_interval == 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT COLLECTOR — Aggregates SubsystemOutputs for Phase C integration
// ═══════════════════════════════════════════════════════════════════════════════

/// Collects `SubsystemOutput`s from all subsystems during Phase B,
/// then integrates them into a single state update in Phase C.
#[derive(Debug, Clone)]
pub struct OutputCollector {
    outputs: Vec<(&'static str, SubsystemOutput)>,
}

impl OutputCollector {
    /// Create a new collector with pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            outputs: Vec::with_capacity(16),
        }
    }

    /// Record an output from a named subsystem.
    pub fn record(&mut self, name: &'static str, output: SubsystemOutput) {
        if !output.is_neutral() {
            self.outputs.push((name, output));
        }
    }

    /// Number of non-neutral outputs recorded.
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Whether no outputs were recorded.
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Clear all outputs (called at cycle start).
    pub fn clear(&mut self) {
        self.outputs.clear();
    }

    /// Look up a specific manager's output by name.
    /// Returns `None` if the manager did not contribute this cycle.
    pub fn get(&self, name: &str) -> Option<&SubsystemOutput> {
        self.outputs
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, o)| o)
    }

    /// Integrate all outputs into a single state update.
    ///
    /// Integration strategy:
    /// - `confidence_delta`: averaged (consensus)
    /// - `lr_modulation`: geometric mean
    /// - `exploration_delta`: averaged
    /// - `arousal_delta`: averaged
    /// - `valence_delta`: averaged
    /// - `flags`: OR'd together
    pub fn integrate(&self) -> IntegratedOutput {
        if self.outputs.is_empty() {
            return IntegratedOutput::default();
        }

        let n = self.outputs.len() as f64;

        // Additive: consensus average
        let confidence_delta: f64 = self
            .outputs
            .iter()
            .map(|(_, o)| o.confidence_delta)
            .sum::<f64>()
            / n;
        let exploration_delta: f64 = self
            .outputs
            .iter()
            .map(|(_, o)| o.exploration_delta)
            .sum::<f64>()
            / n;
        let arousal_delta: f32 = self
            .outputs
            .iter()
            .map(|(_, o)| o.arousal_delta)
            .sum::<f32>()
            / n as f32;
        let valence_delta: f32 = self
            .outputs
            .iter()
            .map(|(_, o)| o.valence_delta)
            .sum::<f32>()
            / n as f32;

        // Multiplicative: geometric mean
        let log_sum: f64 = self
            .outputs
            .iter()
            .map(|(_, o)| o.lr_modulation.max(0.01).ln())
            .sum::<f64>();
        let lr_modulation = (log_sum / n).exp();

        // Boolean flags: OR
        let flags: u32 = self
            .outputs
            .iter()
            .map(|(_, o)| o.flags)
            .fold(0, |acc, f| acc | f);

        IntegratedOutput {
            confidence_delta,
            lr_modulation,
            exploration_delta,
            arousal_delta,
            valence_delta,
            flags,
            n_contributors: self.outputs.len(),
        }
    }

    /// Iterate over all recorded outputs for telemetry/debugging.
    pub fn outputs(&self) -> &[(&'static str, SubsystemOutput)] {
        &self.outputs
    }
}

impl Default for OutputCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of integrating all subsystem outputs for a single cycle.
#[derive(Debug, Clone, Default)]
pub struct IntegratedOutput {
    /// Consensus-averaged confidence delta.
    pub confidence_delta: f64,
    /// Geometric-mean learning rate modulation.
    pub lr_modulation: f64,
    /// Consensus-averaged exploration delta.
    pub exploration_delta: f64,
    /// Consensus-averaged arousal delta.
    pub arousal_delta: f32,
    /// Consensus-averaged valence delta.
    pub valence_delta: f32,
    /// OR'd flags from all subsystems.
    pub flags: u32,
    /// Number of subsystems that contributed non-neutral outputs.
    pub n_contributors: usize,
}

impl IntegratedOutput {
    /// Check if a specific flag is set by any subsystem.
    #[inline]
    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

impl std::fmt::Display for IntegratedOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Δconf={:+.4} ×LR={:.3} Δexpl={:+.4} Δarou={:+.3} flags=0x{:04x} ({})",
            self.confidence_delta,
            self.lr_modulation,
            self.exploration_delta,
            self.arousal_delta,
            self.flags,
            self.n_contributors,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SNAPSHOT BUILDER — Constructs CycleSnapshot from live state
// ═══════════════════════════════════════════════════════════════════════════════

impl CycleSnapshot {
    /// Build a snapshot from the cognitive loop's current state.
    ///
    /// Called once at the start of each cycle (Phase A: OBSERVE).
    /// Copies all observable state into fixed-size buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        cycle_number: u64,
        prediction_confidence: f64,
        fep_lr_boost: f64,
        prediction_error: f32,
        coherence: f32,
        unified_psi: f64,
        phi_attention_weight: f32,
        arousal: f32,
        valence: f32,
        thermodynamic_load: f32,
        dissipative_health: f64,
        somatic_stress: f64,
        urgency: super::CycleUrgency,
        attention_budget_exceeded: bool,
        compressed_state: &[f32],
        input_hv: &symthaea_core::hdc::BinaryHV,
        consciousness_cache: &super::types::ConsciousnessCache,
        quality_metrics: &super::types::QualityMetrics,
    ) -> Self {
        // Copy compressed state into fixed-size buffer
        let mut state_buf = [0.0f32; SNAPSHOT_STATE_DIM];
        let copy_len = compressed_state.len().min(SNAPSHOT_STATE_DIM);
        state_buf[..copy_len].copy_from_slice(&compressed_state[..copy_len]);

        // Copy BinaryHV bytes (tuple struct: .0 is [u8; 2048])
        let hv_buf = input_hv.0;

        let urgency_byte = match urgency {
            super::CycleUrgency::Cruise => 0,
            super::CycleUrgency::Normal => 1,
            super::CycleUrgency::Critical => 2,
        };

        Self {
            cycle_number,
            prediction_confidence,
            fep_lr_boost,
            prediction_error,
            coherence,
            unified_psi,
            phi_attention_weight,
            arousal,
            valence,
            thermodynamic_load,
            dissipative_health,
            somatic_stress,
            urgency: urgency_byte,
            attention_budget_exceeded: if attention_budget_exceeded { 1 } else { 0 },
            compressed_state: state_buf,
            input_hv: hv_buf,
            phenomenal_binding: quality_metrics.last_phenomenal_binding,
            harmonic_coherence: consciousness_cache.last_harmonic_coherence,
            holographic_unity: consciousness_cache.last_holographic_unity,
            gradient_magnitude: quality_metrics.last_gradient_magnitude,
            epistemic_confidence: quality_metrics.last_epistemic_confidence,
            moral_score: quality_metrics.last_moral_score,
            active_external_requests: 0, // TODO: Wire from multimodal coordinator
            vision_hdc_dim: quality_metrics.last_vision_hdc_dim,
            vision_free_energy: quality_metrics.last_vision_free_energy,
            vision_complexity: quality_metrics.last_vision_complexity,
            vision_accuracy: quality_metrics.last_vision_accuracy,
            _reserved: [0; 3],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PANIC ISOLATION — Subsystem health tracking and catch_unwind boundary
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum consecutive panics before a subsystem is disabled for the session.
const SUBSYSTEM_PANIC_DISABLE_THRESHOLD: u32 = 3;

/// Tracks consecutive panic counts per subsystem and disables repeat offenders.
///
/// When a subsystem panics `SUBSYSTEM_PANIC_DISABLE_THRESHOLD` times in a row,
/// it is marked as faulted and skipped for the remainder of the session.
/// A successful `process()` call resets the panic counter to zero.
#[derive(Debug, Clone)]
pub struct SubsystemHealthTracker {
    /// Map from subsystem name to consecutive panic count.
    consecutive_panics: std::collections::HashMap<&'static str, u32>,
    /// Set of subsystems that have been permanently disabled this session.
    faulted: std::collections::HashSet<&'static str>,
}

impl SubsystemHealthTracker {
    /// Create a new tracker with no recorded panics.
    pub fn new() -> Self {
        Self {
            consecutive_panics: std::collections::HashMap::new(),
            faulted: std::collections::HashSet::new(),
        }
    }

    /// Returns true if the subsystem has been disabled due to repeated panics.
    #[inline]
    pub fn is_faulted(&self, name: &str) -> bool {
        self.faulted.contains(name)
    }

    /// Record a successful process() call — resets the panic counter.
    pub fn record_success(&mut self, name: &'static str) {
        if self.consecutive_panics.contains_key(name) {
            self.consecutive_panics.remove(name);
        }
    }

    /// Record a panic — increments the counter and may disable the subsystem.
    pub fn record_panic(&mut self, name: &'static str) {
        let count = self.consecutive_panics.entry(name).or_insert(0);
        *count += 1;
        if *count >= SUBSYSTEM_PANIC_DISABLE_THRESHOLD {
            self.faulted.insert(name);
            tracing::error!(
                subsystem = name,
                consecutive_panics = *count,
                "Subsystem disabled — exceeded {} consecutive panics",
                SUBSYSTEM_PANIC_DISABLE_THRESHOLD,
            );
        }
    }

    /// Number of currently faulted subsystems.
    pub fn faulted_count(&self) -> usize {
        self.faulted.len()
    }

    /// Names of all faulted subsystems.
    pub fn faulted_names(&self) -> impl Iterator<Item = &&'static str> {
        self.faulted.iter()
    }
}

impl Default for SubsystemHealthTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a subsystem's `process()` method with panic isolation.
///
/// Wraps the call in `catch_unwind` so that a panicking sub-crate cannot crash
/// the entire cognitive loop. On panic, returns `SubsystemOutput::NEUTRAL` and
/// records the failure in the health tracker.
///
/// If the subsystem has been faulted (too many consecutive panics), the call is
/// skipped entirely and `None` is returned.
pub fn safe_process(
    subsystem: &mut dyn CognitiveSubsystem,
    snapshot: &CycleSnapshot,
    health: &mut SubsystemHealthTracker,
) -> Option<SubsystemOutput> {
    let name = subsystem.name();

    if health.is_faulted(name) {
        return None;
    }

    // SAFETY: CognitiveSubsystem implementations contain mutable state that is
    // not UnwindSafe. We use AssertUnwindSafe because:
    // 1. On panic, we return NEUTRAL (no partial output propagates).
    // 2. The subsystem may be in an inconsistent state, but the health tracker
    //    will disable it after repeated panics, preventing cascading corruption.
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| subsystem.process(snapshot)));

    match result {
        Ok(output) => {
            health.record_success(name);
            Some(output)
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            tracing::error!(
                subsystem = name,
                panic_message = %msg,
                "Subsystem panicked — returning NEUTRAL output",
            );
            health.record_panic(name);
            Some(SubsystemOutput::NEUTRAL)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify CycleSnapshot is #[repr(C)] compatible — size is stable.
    #[test]
    fn test_cycle_snapshot_size_stable() {
        let size = std::mem::size_of::<CycleSnapshot>();
        // Should be deterministic: fixed fields + [f32; 256] + [u8; 2048] + padding
        assert!(size > 3000, "CycleSnapshot too small: {} bytes", size);
        // Document the expected size for ABI compatibility
        eprintln!("CycleSnapshot size: {} bytes", size);
    }

    /// Verify SubsystemOutput is #[repr(C)] compatible.
    #[test]
    fn test_subsystem_output_size_stable() {
        let size = std::mem::size_of::<SubsystemOutput>();
        assert!(size > 0);
        eprintln!("SubsystemOutput size: {} bytes", size);
    }

    /// Verify CycleSnapshot alignment is suitable for WASM.
    #[test]
    fn test_snapshot_alignment() {
        let align = std::mem::align_of::<CycleSnapshot>();
        // Must be at least 8-byte aligned for f64 fields
        assert!(align >= 8, "CycleSnapshot alignment {} < 8", align);
    }

    /// Verify SubsystemOutput alignment.
    #[test]
    fn test_output_alignment() {
        let align = std::mem::align_of::<SubsystemOutput>();
        assert!(align >= 8, "SubsystemOutput alignment {} < 8", align);
    }

    /// Default snapshot has sane values.
    #[test]
    fn test_snapshot_defaults() {
        let snap = CycleSnapshot::default();
        assert_eq!(snap.cycle_number, 0);
        assert!((snap.prediction_confidence - 0.5).abs() < 1e-10);
        assert!((snap.fep_lr_boost - 1.0).abs() < 1e-10);
        assert_eq!(snap.urgency, 1); // Normal
        assert!(!snap.budget_exceeded());
        assert_eq!(snap.cycle_urgency(), super::super::CycleUrgency::Normal);
    }

    /// Urgency byte round-trips correctly.
    #[test]
    fn test_urgency_roundtrip() {
        let mut snap = CycleSnapshot {
            urgency: 0,
            ..Default::default()
        };
        assert_eq!(snap.cycle_urgency(), super::super::CycleUrgency::Cruise);

        snap.urgency = 1;
        assert_eq!(snap.cycle_urgency(), super::super::CycleUrgency::Normal);

        snap.urgency = 2;
        assert_eq!(snap.cycle_urgency(), super::super::CycleUrgency::Critical);
        assert!(snap.is_critical());
    }

    /// SubsystemOutput::NEUTRAL is the identity element.
    #[test]
    fn test_neutral_output() {
        let output = SubsystemOutput::NEUTRAL;
        assert!(output.is_neutral());
        assert_eq!(output.confidence_delta, 0.0);
        assert_eq!(output.lr_modulation, 1.0);
        assert_eq!(output.exploration_delta, 0.0);
        assert_eq!(output.arousal_delta, 0.0);
        assert_eq!(output.valence_delta, 0.0);
        assert_eq!(output.flags, 0);
    }

    /// Output flags work correctly.
    #[test]
    fn test_output_flags() {
        let output = SubsystemOutput::NEUTRAL
            .with_flag(output_flags::REQUEST_EXPLORATION)
            .with_flag(output_flags::ANOMALY_DETECTED);

        assert!(output.has_flag(output_flags::REQUEST_EXPLORATION));
        assert!(output.has_flag(output_flags::ANOMALY_DETECTED));
        assert!(!output.has_flag(output_flags::VETO_ACTION));
        assert!(!output.is_neutral()); // flags make it non-neutral
    }

    /// OutputCollector integration with consensus averaging.
    #[test]
    fn test_collector_integration_averaging() {
        let mut collector = OutputCollector::new();

        // Two subsystems propose different confidence deltas
        collector.record(
            "subsystem_a",
            SubsystemOutput {
                confidence_delta: 0.1,
                lr_modulation: 1.2,
                ..SubsystemOutput::NEUTRAL
            },
        );
        collector.record(
            "subsystem_b",
            SubsystemOutput {
                confidence_delta: 0.3,
                lr_modulation: 0.8,
                ..SubsystemOutput::NEUTRAL
            },
        );

        let result = collector.integrate();
        assert_eq!(result.n_contributors, 2);

        // Confidence: average of 0.1 and 0.3 = 0.2
        assert!((result.confidence_delta - 0.2).abs() < 1e-10);

        // LR: geometric mean of 1.2 and 0.8 = sqrt(0.96) ≈ 0.9798
        let expected_lr = (1.2f64 * 0.8).sqrt();
        assert!(
            (result.lr_modulation - expected_lr).abs() < 1e-10,
            "LR {} != expected {}",
            result.lr_modulation,
            expected_lr,
        );
    }

    /// Empty collector returns default integrated output.
    #[test]
    fn test_collector_empty() {
        let collector = OutputCollector::new();
        let result = collector.integrate();
        assert_eq!(result.n_contributors, 0);
        assert_eq!(result.confidence_delta, 0.0);
        assert_eq!(result.lr_modulation, 0.0); // default f64
    }

    /// Neutral outputs are filtered out.
    #[test]
    fn test_collector_filters_neutral() {
        let mut collector = OutputCollector::new();
        collector.record("neutral", SubsystemOutput::NEUTRAL);
        assert!(collector.is_empty());

        collector.record(
            "active",
            SubsystemOutput {
                confidence_delta: 0.05,
                ..SubsystemOutput::NEUTRAL
            },
        );
        assert_eq!(collector.len(), 1);
    }

    /// Flags are OR'd across subsystems.
    #[test]
    fn test_collector_flags_or() {
        let mut collector = OutputCollector::new();
        collector.record(
            "ethics",
            SubsystemOutput {
                confidence_delta: 0.01, // non-neutral to be recorded
                flags: output_flags::VETO_ACTION,
                ..SubsystemOutput::NEUTRAL
            },
        );
        collector.record(
            "curiosity",
            SubsystemOutput {
                exploration_delta: 0.1, // non-neutral
                flags: output_flags::REQUEST_EXPLORATION,
                ..SubsystemOutput::NEUTRAL
            },
        );

        let result = collector.integrate();
        assert!(result.has_flag(output_flags::VETO_ACTION));
        assert!(result.has_flag(output_flags::REQUEST_EXPLORATION));
        assert!(!result.has_flag(output_flags::REQUEST_REST));
    }

    /// Collector clear resets state.
    #[test]
    fn test_collector_clear() {
        let mut collector = OutputCollector::new();
        collector.record(
            "test",
            SubsystemOutput {
                confidence_delta: 0.1,
                ..SubsystemOutput::NEUTRAL
            },
        );
        assert_eq!(collector.len(), 1);
        collector.clear();
        assert!(collector.is_empty());
    }

    /// Display formatting for IntegratedOutput.
    #[test]
    fn test_integrated_output_display() {
        let result = IntegratedOutput {
            confidence_delta: 0.05,
            lr_modulation: 1.1,
            exploration_delta: -0.02,
            arousal_delta: 0.01,
            valence_delta: 0.0,
            flags: output_flags::REQUEST_EXPLORATION | output_flags::ANOMALY_DETECTED,
            n_contributors: 3,
        };
        let s = format!("{}", result);
        assert!(s.contains("+0.05"));
        assert!(s.contains("1.100"));
        assert!(s.contains("(3)"));
    }

    /// Mock subsystem implementing the trait.
    struct MockSubsystem {
        fire_count: u32,
    }

    impl CognitiveSubsystem for MockSubsystem {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn interval(&self) -> u32 {
            7
        }

        fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
            self.fire_count += 1;
            SubsystemOutput {
                confidence_delta: snapshot.prediction_error as f64 * 0.01,
                ..SubsystemOutput::NEUTRAL
            }
        }
    }

    /// Trait is object-safe and can be called through dyn dispatch.
    #[test]
    fn test_trait_object_safe() {
        let mut subsystem: Box<dyn CognitiveSubsystem> = Box::new(MockSubsystem { fire_count: 0 });

        assert_eq!(subsystem.name(), "mock");
        assert_eq!(subsystem.interval(), 7);

        let snapshot = CycleSnapshot {
            prediction_error: 0.5,
            ..CycleSnapshot::default()
        };

        // Should run on cycle 0, 7, 14, ...
        assert!(subsystem.should_run(0, 1));
        assert!(!subsystem.should_run(1, 1));
        assert!(subsystem.should_run(7, 1));

        // In Cruise mode, interval doubles: 0, 14, 28, ...
        assert!(subsystem.should_run(0, 0));
        assert!(!subsystem.should_run(7, 0));
        assert!(subsystem.should_run(14, 0));

        let output = subsystem.process(&snapshot);
        assert!((output.confidence_delta - 0.005).abs() < 1e-10);
    }

    /// Checkpoint/restore default implementations work.
    #[test]
    fn test_checkpoint_restore_defaults() {
        let mut subsystem = MockSubsystem { fire_count: 0 };
        let data = subsystem.checkpoint();
        assert!(data.is_empty());
        assert!(subsystem.restore(&[1, 2, 3]).is_ok());
    }

    /// should_run with interval=1 always returns true.
    #[test]
    fn test_interval_one_always_runs() {
        struct AlwaysRun;
        impl CognitiveSubsystem for AlwaysRun {
            fn name(&self) -> &'static str {
                "always"
            }
            fn interval(&self) -> u32 {
                1
            }
            fn process(&mut self, _: &CycleSnapshot) -> SubsystemOutput {
                SubsystemOutput::NEUTRAL
            }
        }

        let subsystem = AlwaysRun;
        for cycle in 0..100 {
            assert!(subsystem.should_run(cycle, 0)); // Even in Cruise
            assert!(subsystem.should_run(cycle, 1));
            assert!(subsystem.should_run(cycle, 2));
        }
    }

    // ── Panic Isolation Tests ──────────────────────────────────────────────

    /// SubsystemHealthTracker starts clean.
    #[test]
    fn test_health_tracker_initial_state() {
        let tracker = SubsystemHealthTracker::new();
        assert_eq!(tracker.faulted_count(), 0);
        assert!(!tracker.is_faulted("anything"));
    }

    /// Successful calls reset the panic counter.
    #[test]
    fn test_health_tracker_success_resets_counter() {
        let mut tracker = SubsystemHealthTracker::new();
        tracker.record_panic("test_subsystem");
        tracker.record_panic("test_subsystem");
        // Two panics, not yet faulted
        assert!(!tracker.is_faulted("test_subsystem"));
        // Success resets
        tracker.record_success("test_subsystem");
        // Two more panics shouldn't fault (counter was reset)
        tracker.record_panic("test_subsystem");
        tracker.record_panic("test_subsystem");
        assert!(!tracker.is_faulted("test_subsystem"));
    }

    /// Three consecutive panics disable the subsystem.
    #[test]
    fn test_health_tracker_faults_after_threshold() {
        let mut tracker = SubsystemHealthTracker::new();
        tracker.record_panic("flaky");
        assert!(!tracker.is_faulted("flaky"));
        tracker.record_panic("flaky");
        assert!(!tracker.is_faulted("flaky"));
        tracker.record_panic("flaky");
        assert!(tracker.is_faulted("flaky"));
        assert_eq!(tracker.faulted_count(), 1);
    }

    /// Multiple subsystems track independently.
    #[test]
    fn test_health_tracker_independent_subsystems() {
        let mut tracker = SubsystemHealthTracker::new();
        tracker.record_panic("alpha");
        tracker.record_panic("alpha");
        tracker.record_panic("alpha");
        assert!(tracker.is_faulted("alpha"));
        assert!(!tracker.is_faulted("beta"));
        assert_eq!(tracker.faulted_count(), 1);
    }

    /// safe_process catches panics and returns NEUTRAL.
    #[test]
    fn test_safe_process_catches_panic() {
        struct PanickingSubsystem;
        impl CognitiveSubsystem for PanickingSubsystem {
            fn name(&self) -> &'static str {
                "panicker"
            }
            fn interval(&self) -> u32 {
                1
            }
            fn process(&mut self, _: &CycleSnapshot) -> SubsystemOutput {
                panic!("subsystem exploded");
            }
        }

        let mut sub = PanickingSubsystem;
        let mut health = SubsystemHealthTracker::new();
        let snapshot = CycleSnapshot::default();

        let result = safe_process(&mut sub, &snapshot, &mut health);
        assert!(result.is_some());
        assert!(result.unwrap().is_neutral());
        assert!(!health.is_faulted("panicker")); // 1 panic, not yet faulted
    }

    /// safe_process skips faulted subsystems.
    #[test]
    fn test_safe_process_skips_faulted() {
        struct PanickingSubsystem;
        impl CognitiveSubsystem for PanickingSubsystem {
            fn name(&self) -> &'static str {
                "panicker"
            }
            fn interval(&self) -> u32 {
                1
            }
            fn process(&mut self, _: &CycleSnapshot) -> SubsystemOutput {
                panic!("subsystem exploded");
            }
        }

        let mut sub = PanickingSubsystem;
        let mut health = SubsystemHealthTracker::new();
        let snapshot = CycleSnapshot::default();

        // Trigger 3 panics to fault the subsystem
        for _ in 0..3 {
            let _ = safe_process(&mut sub, &snapshot, &mut health);
        }
        assert!(health.is_faulted("panicker"));

        // Now safe_process should return None (skipped)
        let result = safe_process(&mut sub, &snapshot, &mut health);
        assert!(result.is_none());
    }

    /// safe_process passes through normal output on success.
    #[test]
    fn test_safe_process_passes_output() {
        let mut sub = MockSubsystem { fire_count: 0 };
        let mut health = SubsystemHealthTracker::new();
        let snapshot = CycleSnapshot {
            prediction_error: 1.0,
            ..CycleSnapshot::default()
        };

        let result = safe_process(&mut sub, &snapshot, &mut health);
        assert!(result.is_some());
        let output = result.unwrap();
        assert!((output.confidence_delta - 0.01).abs() < 1e-10);
        assert_eq!(sub.fire_count, 1);
        assert!(!health.is_faulted("mock"));
    }
}
