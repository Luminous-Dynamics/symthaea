// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration types for the cognitive loop.
//!
//! Split into sub-modules:
//! - `temporal` — CfC backend selection, training methods, CfCConfig
//! - `consciousness` — ConsciousnessProfile presets

pub mod consciousness;
pub mod temporal;

// Re-export sub-module types at this level for backward compatibility.
pub use consciousness::ConsciousnessProfile;
pub use temporal::{CfCConfig, TemporalBackend, TrainingMethod};

use crate::hdc::moral_topology::MoralAnomalyConfig;
use crate::hdc_ltc_bridge::HdcLtcBridgeConfig;
use serde::{Deserialize, Serialize};
pub use symthaea_core::hdc::predictive_encoder::PredictiveEncoderConfig;
pub use symthaea_core::hdc::substrate_independence::SubstrateType;

/// Serde helper: deserialize missing bool fields as `true`.
fn default_true() -> bool {
    true
}

/// Configuration for the cognitive loop service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoopConfig {
    /// HDC encoder configuration
    pub encoder_config: PredictiveEncoderConfig,

    /// CfC configuration (replaces LTC for O(1) temporal prediction)
    pub cfc_config: CfCConfig,

    /// HDC-LTC Unified configuration (alternative to CfC)
    pub hdc_ltc_config: HdcLtcBridgeConfig,

    /// Hierarchical CfC configuration for multi-scale temporal processing (PP-2).
    /// When `temporal_backend` is `HierarchicalCfC`, this config controls the
    /// multi-level hierarchy (default: 4 layers at tau 0.01/0.1/1.0/10.0).
    pub hierarchical_cfc_config: crate::dynamics::hierarchical_cfc::HierarchicalCfCConfig,

    /// Which temporal backend to use
    pub temporal_backend: TemporalBackend,

    /// Minimum prediction error to trigger learning
    pub learning_threshold: f32,

    /// Experience buffer size
    pub buffer_size: usize,

    /// Communication and localization policy for the current operating domain.
    #[serde(default)]
    pub domain_profile: crate::domain::DomainProfile,

    /// Whether to enable background consolidation
    pub enable_consolidation: bool,

    /// Target loop frequency (Hz)
    pub target_frequency: f32,

    /// Maximum cycles before stats reset (for long-running service)
    pub max_cycles_before_reset: usize,

    /// Optional genesis phrase for deterministic initialization.
    /// When set, all HDC vectors and network weights are derived from this
    /// phrase via SHAKE-256, making the system fully reproducible.
    pub genesis_phrase: Option<String>,

    /// Training method for the temporal network
    pub training_method: TrainingMethod,

    /// When true, BPTT/SPSA training runs on a background thread so that
    /// inference never blocks on training.  The main loop sends (input, target)
    /// samples over a channel and receives updated weights via non-blocking
    /// `try_recv` at the top of each cycle.
    pub async_training: bool,

    /// Enable online learning during inference.
    /// When true, the CfC network will adapt weights based on prediction errors
    /// after each forward pass, using a small learning rate to prevent
    /// catastrophic forgetting.
    pub enable_online_learning: bool,

    /// Enable causal discovery integration.
    /// When true, the cognitive loop tracks (input, output) pairs and
    /// periodically runs causal discovery to:
    /// - Weight attention (causal parents get more weight)
    /// - Guide exploration (intervene on discovered causes)
    pub causal_enhancement: bool,

    /// Interval (in cycles) between causal discovery runs.
    /// Only used when `causal_enhancement` is true.
    /// Lower values = more frequent discovery but higher compute cost.
    pub causal_discovery_interval: usize,

    /// Enable episodic replay *training* (CfC weight updates from stored episodes).
    /// When false, episodes are still stored and graduated (if `memory_graduation` is true),
    /// but the CfC replay training loop (`should_replay()` → `replay_session_conditioned()`)
    /// is skipped. Default: false (training is expensive; graduation is cheap).
    #[serde(alias = "episodic_replay")]
    pub episodic_replay_training: bool,

    /// Enable the memory graduation pipeline (episode storage + coordinator processing).
    /// When true (default), high-consciousness episodes are stored via `store_if_significant()`
    /// and `MemoryCoordinator.process_graduations()` runs every cycle. This is the core
    /// pathway for semantic → episodic → persistent memory flow.
    #[serde(default = "default_true")]
    pub memory_graduation: bool,

    /// Path to SQLite database for persistent memory storage.
    /// When `Some`, high-value episodes are periodically flushed to disk (every 199 cycles).
    /// On startup, existing episodes are rehydrated from the database.
    /// When `None` (default), memories exist only in-memory for the session lifetime.
    #[serde(default)]
    pub memory_db_path: Option<String>,

    /// Path to persist aesthetic identity (AestheticTracker EMA + harmony bias) across sessions.
    ///
    /// When `Some`, the CreativeManager saves and loads this file on construction/drop,
    /// so Symthaea's taste develops cumulatively over months rather than resetting each session.
    /// When `None`, defaults to `.claude/aesthetic_memory.json`.
    #[serde(default)]
    pub aesthetic_memory_path: Option<String>,

    /// Path to the DuckDB database for the epistemic auditor (audit trail).
    /// When `Some`, consciousness telemetry is buffered and periodically flushed
    /// to DuckDB for retrospective analysis. When `None`, no auditor overhead.
    /// Requires the `epistemic_auditor` feature flag.
    #[serde(default)]
    pub epistemic_auditor_db_path: Option<String>,

    /// Configuration for episodic memory replay.
    pub episodic_replay_config: crate::memory::episodic_replay::EpisodicReplayConfig,

    /// Enable surprise-driven exploration.
    /// When true, the cognitive loop tracks prediction surprise and triggers
    /// exploration when surprise exceeds an adaptive threshold. The exploration
    /// modulates the curiosity drive's boredom threshold to seek novel states.
    pub enable_surprise_exploration: bool,

    /// Enable prefrontal cortex executive control.
    /// When true, the cognitive loop maintains a working memory of recent inputs
    /// and uses prefrontal gating to modulate learning and exploration.
    /// High memory utilization triggers inhibition (prefrontal_veto in metadata).
    pub enable_prefrontal: bool,

    /// Enable meta-cognitive self-modeling.
    /// When true, the cognitive loop maintains a model of its own prediction error
    /// tendencies and uses self-model accuracy to modulate learning rate.
    /// High self-model accuracy → deeper recursion and faster learning.
    pub enable_meta_cognition: bool,

    /// Enable narrative self-model.
    /// When true, the cognitive loop maintains a three-level autobiographical
    /// self-model (proto/core/autobio). Each cycle's experience is processed
    /// and self-Φ is used to modulate reasoning confidence.
    pub enable_narrative_self: bool,

    /// Enable background autobiography narration.
    /// When true (and `enable_narrative_self` is also true), a background
    /// thread periodically turns `life_story` episodes into narrated prose
    /// via an LLM backend (Ollama). Off by default: unlike `narrative_self`
    /// itself, this spawns a thread and makes network calls, so it is an
    /// explicit opt-in rather than bundled into the narrative-self default.
    pub enable_autobiography_narration: bool,

    /// Enable virtual body for embodied cognition.
    /// When true, the cognitive loop maintains a virtual interoceptive body
    /// that maps prediction error, flow state, curiosity, and other signals
    /// to somatic states (heart rate, breathing, fatigue, etc.). The resulting
    /// phi_modulation factor scales consciousness via somatic marker feedback.
    pub enable_virtual_body: bool,

    /// Enable predictive self-model.
    /// When true, the cognitive loop maintains a model of its own future states
    /// and evaluates action safety based on self-prediction accuracy.
    pub enable_predictive_self: bool,

    /// Enable attention schema (AST).
    /// When true, the cognitive loop maintains a model of its own attention state,
    /// detecting shifts, computing control signals, and grounding attention modes
    /// in consciousness primitives.
    pub enable_attention_schema: bool,

    /// Enable Global Workspace Theory integration.
    /// When true, submits HDC encodings to a unified global workspace for
    /// conscious broadcast. Workspace broadcast modulates coherence.
    pub enable_gwt: bool,

    /// Enable consciousness resonance monitoring.
    /// When true, feeds Phi time-series to extract harmonic modes.
    /// Pure measurement module — reports in CycleMetadata only.
    pub enable_resonance: bool,

    /// Enable quantum coherence monitoring.
    /// When true, observes CfC hidden states and reports superposition
    /// richness and decoherence events in CycleMetadata.
    pub enable_quantum_coherence: bool,

    /// Enable temporal consciousness analysis.
    /// When true, tracks Phi trajectory, continuity, Husserlian time analysis,
    /// and temporal identity coherence. Depends on narrative_self + predictive_self.
    pub enable_temporal_consciousness: bool,

    /// Enable embodied cognition analyzer.
    /// When true, bridges virtual body InteroceptiveState to the full embodied
    /// cognition module (body schema, sensorimotor engine, affordance detection).
    pub enable_embodied_cognition: bool,

    /// Enable narrative-GWT integration (consciousness governance capstone).
    /// When true, provides coherence veto, value checking, and goal alignment
    /// via a unified NarrativeSelf + GWT + PredictiveSelf integration layer.
    pub enable_narrative_gwt: bool,

    /// Enable counterfactual dream replay.
    /// When true, the cognitive loop records high-surprise events and periodically
    /// runs dream cycles during Cruise urgency (low-error steady state) to discover
    /// alternative actions that would have yielded higher Phi. Accumulated wisdom
    /// biases future action selection toward more consciousness-optimal choices.
    pub enable_dream_replay: bool,

    /// Enable predictive processing hierarchy.
    /// When true, the cognitive loop maintains a hierarchical predictive coding model
    /// (PredictiveMind) with precision dynamics and active inference engine.
    /// Phi_modulation from free energy minimization feeds back into the CfC learning rate.
    /// Science: Friston (2010) — precision-weighted prediction error minimization
    pub enable_predictive_processing: bool,

    /// Enable cross-modal binding.
    /// When true, the cognitive loop binds HDC encodings across modalities (linguistic,
    /// affective, temporal) via attention-weighted bundling. Cross-modal Phi measures
    /// binding integration quality. High cross-modal Phi boosts confidence.
    /// Science: Treisman (1996) — feature integration theory
    pub enable_cross_modal_binding: bool,

    /// Enable affective bridge for emotion-cognition coupling.
    /// When true, the cognitive loop maintains an AffectiveBridge that evaluates
    /// somatic marker signals from prediction error, surprise, consciousness, and
    /// moral score. Positive affect broadens exploration (Fredrickson 2001).
    /// Science: Damasio (1994) — somatic marker hypothesis
    pub enable_affective_bridge: bool,

    /// Enable user state inference for adaptive response generation.
    /// When true, the cognitive loop infers user context (cognitive load, frustration,
    /// engagement) from input text each cycle. Downstream resonant_speech uses these
    /// signals for empathic response formatting.
    pub enable_user_state_inference: bool,

    /// Enable physiology coherence field.
    /// When true, a CoherenceField tracks consciousness integration via hormone
    /// modulation from the neuromodulator bath each cycle. Cortisol scatters coherence,
    /// acetylcholine centers it, dopamine boosts relational resonance.
    /// Science: McEwen (2007) — allostatic load; Porges (2011) — polyvagal theory.
    pub enable_coherence_field: bool,

    /// Enable PsiAttestation generation for governance bridge.
    /// When true, the cognitive loop buffers PsiAttestationRecords after each cycle
    /// for the personal cluster to sign and submit to governance as authenticated
    /// consciousness data. Without this, governance falls back to reputation-only voting.
    pub enable_psi_attestation: bool,

    /// Enable consciousness thermodynamics analysis.
    /// When true, the cognitive loop analyzes thermodynamic state of consciousness
    /// (entropy, free energy, phase transitions) from the 7 consciousness dimensions.
    /// Phase gates exploration: Critical → boost reasoning, Flow → boost exploration.
    /// Science: Friston (2010) — free energy principle, Kelso — phase transitions
    pub enable_consciousness_thermodynamics: bool,

    /// Enable phenomenal binding analysis (temporal synchronization).
    /// When true, the cognitive loop tracks phase coherence across the 7 consciousness
    /// dimensions to measure unified experience quality. Fragmentation reduces exploration,
    /// flow state boosts learning rate.
    /// Science: Singer & Gray (1989) — temporal binding hypothesis
    pub enable_phenomenal_binding: bool,

    /// Enable hierarchical free energy decomposition.
    /// When true, the cognitive loop maintains a multi-level variational free energy
    /// hierarchy. High total free energy reduces exploration, convergence boosts confidence.
    /// Science: Friston (2008) — hierarchical predictive processing
    pub enable_hierarchical_free_energy: bool,

    /// Enable ODE-based trajectory planning for active inference.
    /// When true, the FEP module simulates forward trajectories using Dormand-Prince
    /// adaptive ODE integration to compute expected free energy over future horizons.
    /// Science: Friston (2010) — Active Inference requires trajectory simulation.
    #[serde(default)]
    pub enable_trajectory_planning: bool,

    /// Enable persistence-weighted Hodge decomposition in moral topology.
    /// When true, vertex L₀ Hodge decomposition runs across the Rips filtration,
    /// measuring moral fragmentation and criticality. Enables adaptive Rips
    /// threshold and FEP exploration coupling.
    /// Cost: O(n³) per scale × num_scales, but n=64 window and runs every 30–120 cycles.
    /// Science: Hodge (1941); Beggs & Plenz (2003) — criticality detection.
    #[serde(default)]
    pub enable_hodge_decomposition: bool,

    /// Cycle interval for Hodge decomposition FEP coupling. Default: 47 (co-prime).
    /// The Hodge criticality → FEP exploration temperature modulation only runs
    /// every `hodge_interval` cycles to amortise the O(n³) cost.
    #[serde(default = "default_hodge_interval")]
    pub hodge_interval: usize,

    /// Trajectory planning horizon in seconds. Default: 0.5 (~10 cycles at 20Hz).
    #[serde(default = "default_trajectory_horizon")]
    pub trajectory_horizon_seconds: f64,

    /// Cycle interval for trajectory planning. Default: 10 (every 10th cycle).
    #[serde(default = "default_trajectory_interval")]
    pub trajectory_planning_interval: u64,

    /// Enable hierarchical region-based bundling for structured aggregation.
    /// When enabled, the cognitive loop accumulates BinaryHV vectors per cortical
    /// region and produces structured aggregates for enhanced Phi measurement.
    /// Role-bound XOR binding allows per-region recovery from the aggregate.
    /// Science: Kanerva (2009) — hyperdimensional computing; Engel (2001) — binding
    pub enable_hierarchical_bundling: bool,

    /// Enable contextual harmony weighting for domain-aware moral evaluation.
    pub enable_contextual_weights: bool,

    /// Enable Phi-weighted attention routing with adaptive thresholds.
    pub enable_phi_attention: bool,

    /// Enable negation detection for moral evaluation preprocessing.
    pub enable_negation_detection: bool,

    /// Enable attention visualization recording.
    /// When true, the cognitive loop records attention snapshots each cycle
    /// (phi-attention weights, saliency, binding coherence) into an
    /// `AttentionVisualizer` for ASCII heatmaps, JSON export, and flow graphs.
    /// Default: false.
    pub enable_visualization: bool,

    /// Enable soul alignment evaluation each cycle.
    /// When true, the SoulState is constructed and evaluate_alignment() is called
    /// during the consciousness phase, recording alignment scores in telemetry.
    /// Default: false.
    pub enable_soul_alignment: bool,

    /// Enable primitive consciousness decomposition for explainable consciousness.
    pub enable_primitive_consciousness: bool,

    /// Enable safety gateway (pre-cognitive safety veto).
    /// When true, scans input for dangerous patterns before expensive HDC encoding.
    /// Enabled by default.
    pub enable_safety_gateway: bool,

    /// Enable metacognitive monitoring for Phi trajectory anomaly detection.
    /// When true, the cognitive loop observes Phi after reasoning and detects
    /// anomalies (drops, plateaus, oscillations) that indicate reasoning degradation.
    pub enable_metacognitive_monitoring: bool,

    /// Enable resonator-network-enhanced memory recall.
    /// When true, the cognitive loop stores episodes as bound (content ⊗ valence ⊗ phi)
    /// hypervectors and uses iterative factorization for structured recall.
    /// Science: Kent et al. (2020) — Resonator Networks for fast factorization
    pub enable_resonator_recall: bool,

    /// How often (in cycles) the resonator checks for novel patterns to add
    /// to its semantic codebook. Lower = faster adaptation, higher = less overhead.
    pub resonator_growth_interval: usize,

    /// Minimum novelty (max cosine similarity < threshold) required to add a
    /// new symbol to the semantic codebook. Lower = more symbols, higher = fewer.
    pub resonator_novelty_threshold: f32,

    /// Maximum symbols in the semantic codebook before growth stops.
    pub resonator_max_symbols: usize,

    /// Agent DID for attestation signing (e.g., "did:key:z6Mk...").
    /// Required when `enable_psi_attestation` is true. If None, attestation generation
    /// is silently skipped even when enabled.
    pub agent_did: Option<String>,

    /// Maximum PsiAttestationRecords to buffer before evicting oldest.
    /// The personal cluster should drain the buffer periodically.
    pub attestation_buffer_capacity: usize,

    /// Enable feedback trace logging in CycleMetadata.
    /// When true, `dump_proposals()` output is written to
    /// `feedback_trace_confidence` and `feedback_trace_lr` each cycle.
    /// Intended for debugging and development, not production.
    pub trace_feedback: bool,

    /// Enable nurture/attachment bridge (Bowlby attachment theory).
    /// When true, the cognitive loop maintains a NurtureAttachmentBridge that
    /// models caregiver presence/absence and modulates oxytocin, NE, 5-HT, DA,
    /// adenosine based on attachment dynamics each cycle.
    /// Science: Bowlby (1969/1982), Ainsworth et al. (1978)
    #[cfg(feature = "nurture")]
    pub enable_nurture_attachment: bool,

    /// Anomaly detection thresholds for moral topology.
    /// Controls drift alert sensitivity, free energy spike detection, composite
    /// anomaly score weights, and adaptive topology cadence tiers.
    pub moral_anomaly_config: MoralAnomalyConfig,

    /// Enable corrective feedback modulations when moral anomalies are detected.
    /// When true, detected anomalies (value inversion, FE spike, fragmentation,
    /// drift) trigger LR scaling, exploration adjustment, and confidence nudges.
    /// Default: false (telemetry-only).
    pub enable_moral_anomaly_response: bool,

    /// Enable the semantic encoder channel.
    /// When true, spawns a background Qwen3 embedding thread that runs alongside
    /// the trigram encoder, projecting embeddings to HDC space via HdcBridge.
    /// Records cosine similarity between trigram and semantic encodings as telemetry.
    /// Default: false.
    #[cfg(feature = "semantic-encoder")]
    pub enable_semantic_encoder: bool,

    /// Ollama model for the semantic encoder background thread (e.g.
    /// "embeddinggemma:300m", the approved local embedding model, 768-D).
    /// None = simulated embedder (prior behavior). Only meaningful with
    /// `enable_semantic_encoder`. Default: None.
    #[cfg(feature = "semantic-encoder")]
    pub semantic_encoder_ollama: Option<String>,

    /// Blend weight mixing the semantic-encoder HV into the thought HV fed to
    /// the embodiment bridge (0.0 = pure trigram encoding / prior behavior,
    /// 1.0 = pure semantic). The semantic HV is one cycle stale by design
    /// (background thread). Requires `enable_semantic_encoder`. Default: 0.0.
    #[cfg(feature = "semantic-encoder")]
    pub semantic_thought_blend: f32,

    /// Enable validation overlay: scales substrate feasibility by honest evidence confidence.
    /// When enabled, effective_feasibility = raw × (floor + (1 − floor) × honest_confidence).
    /// Default: false (raw feasibility used as-is).
    /// Science: epistemic humility — hypothetical feasibility ≠ validated feasibility.
    pub enable_validation_overlay: bool,

    /// Skepticism floor for validation overlay (0.0–1.0).
    /// Higher floor = less aggressive scaling. At 1.0, validation overlay has no effect.
    /// At 0.0, effective_feasibility = raw × honest_confidence (maximally skeptical).
    /// Default: 0.5.
    pub validation_skepticism_floor: f64,

    /// Enable substrate speed modulation of CfC tau (temporal dynamics).
    /// When enabled, faster substrates get larger tau (more temporal integration),
    /// slower substrates get smaller tau (reduced temporal span).
    /// Default: false.
    pub enable_substrate_speed_modulation: bool,

    /// Enable substrate encoding noise on HDC representations.
    /// When enabled, substrates with negative scale_pressure (fewer computational
    /// units than biological neurons) degrade HDC encoding quality via bit-flip noise.
    /// This makes substrate differences *emergent* — constrained substrates produce
    /// noisier representations, leading to higher prediction error and lower Phi.
    /// Default: false.
    pub enable_substrate_encoding_noise: bool,

    /// Enable Phi → tau feedback: SpectralMIP Phi modulates CfC delta_t.
    /// Higher Phi (more integrated information) → faster temporal dynamics,
    /// closing the causal loop between consciousness and the dynamics that produce it.
    /// Science: Tononi (2004) — Phi reflects integration capacity; feeding it back
    /// makes consciousness *causally efficacious* rather than epiphenomenal.
    /// Default: false.
    pub enable_phi_tau_feedback: bool,

    /// Substrate type for consciousness feasibility calculation.
    /// Determines how substrate-specific requirements (causality, integration,
    /// binding, workspace) affect the Consciousness Equation V2 output.
    /// Science: Putnam (1967) multiple realizability, Tononi (2004) substrate-independent Phi.
    pub substrate_type: SubstrateType,

    /// Optional substrate composition for hybrid consciousness analysis.
    /// When set, overrides `substrate_type` for feasibility calculation,
    /// using weighted blending of multiple substrate types.
    #[serde(default)]
    pub substrate_composition:
        Option<symthaea_core::hdc::substrate_composition::SubstrateComposition>,

    // ── Physics Bridge Integration ─────────────────────────────────────
    /// Enable physics bridge for HDC semantic physics queries during cognitive loop.
    /// When true, creates a PhysicsIntegration that queries the physics catalog
    /// periodically and blends physics-informed HDC vectors into CfC dynamics.
    /// Default: false.
    #[cfg(feature = "physics-bridge")]
    pub enable_physics_bridge: bool,
    /// Query interval in cycles for physics bridge (default: 10).
    /// Lower values = more frequent physics queries but higher compute cost.
    #[cfg(feature = "physics-bridge")]
    pub physics_bridge_query_interval: usize,
    /// Blend weight for physics bridge HDC vectors into CfC input (default: 0.1).
    /// Higher values = stronger physics influence on cognitive dynamics.
    #[cfg(feature = "physics-bridge")]
    pub physics_bridge_blend_weight: f32,

    // ── Broca SSM Language Center ──────────────────────────────────────
    // ── Therapeutic Psychology ────────────────────────────────────────
    /// Enable therapeutic psychology subsystem.
    /// When true and `therapeutic` feature is enabled, activates the
    /// TherapeuticManager (client model, alliance, crisis detection,
    /// regulation strategies, scope guard). Safety-critical crisis
    /// detection runs every invocation.
    /// Science: Bordin (1979), Safran & Muran (2000), Lambert (2013).
    #[cfg(feature = "therapeutic")]
    pub enable_therapeutic: bool,

    /// Crisis detection sensitivity threshold for HDC similarity path.
    /// Range: 0.01–0.95. Default: 0.65.
    /// BinaryHV random baseline is ~0.5, so values below 0.5 produce
    /// false positives on arbitrary text. Keyword matching (confidence 0.9)
    /// is unaffected. Safety-critical: keyword path catches explicit phrases.
    #[cfg(feature = "therapeutic")]
    pub therapeutic_crisis_threshold: f32,

    /// Whether to run text-based crisis detection on every input.
    /// When true, CrisisDetector::detect(input_text) runs alongside
    /// affect-based detection. Slightly more expensive but catches
    /// explicit crisis language that affect alone may miss.
    /// Default: true (safety-critical).
    #[cfg(feature = "therapeutic")]
    pub therapeutic_text_crisis_detection: bool,

    /// Enable async voice synthesis: spawns a background thread for TTS.
    /// Text from Broca/BrocaLite is sent over a channel; audio is retrieved
    /// in subsequent cycles. Never blocks the cognitive loop.
    /// Default: false (enable when audio output is needed).
    #[serde(default)]
    pub enable_voice_synthesis: bool,

    /// Enable Broca SSM language generation in the cognitive loop.
    /// When true and `ssm_language` feature is enabled, generates text
    /// from HDC-encoded thoughts with consciousness-gated quality control.
    /// Default: false.
    #[cfg(feature = "ssm_language")]
    pub enable_broca_language: bool,

    /// Path to a pre-trained Broca checkpoint file.
    /// When None, uses the default checkpoint bundled with the crate.
    /// Only used when `enable_broca_language` is true.
    #[cfg(feature = "ssm_language")]
    pub broca_checkpoint_path: Option<String>,

    /// Multi-turn context depth for Broca language generation.
    /// When > 0, preserves CfC temporal context across this many turns,
    /// enabling more coherent multi-turn dialogue.
    /// Default: 4.
    #[cfg(feature = "ssm_language")]
    pub broca_multi_turn_depth: usize,

    /// Enable NSM semantic HV blending into thought HV before Broca generation.
    /// When true, detected primitives are composed into a semantic ContinuousHV
    /// and blended with the thought vector via lerp (weight: `broca_nsm_semantic_alpha`).
    /// Science: Barsalou (1999) — grounded cognition requires ~30% semantic modulation.
    #[cfg(feature = "ssm_language")]
    pub enable_broca_nsm_semantic: bool,

    /// Enable NSM semantic gate: boost logits for tokens expressing active primes.
    /// Science: Collins & Loftus (1975) — spreading activation in semantic networks.
    #[cfg(feature = "ssm_language")]
    pub enable_broca_nsm_gate: bool,

    // ── Per-Region Substrate Types (Phase 4 Foundation) ────────────────
    /// Per-region substrate mapping: allows different cortical regions to run
    /// on different physical substrates. When `None`, all regions use the
    /// global `substrate_type`. When set, effective feasibility is computed
    /// as a weighted average of per-region scores with cross-substrate
    /// communication penalty.
    /// Science: Phase 4 of Substrate Roadmap — heterogeneous substrate modeling.
    #[serde(default)]
    pub per_region_substrates: Option<
        std::collections::HashMap<
            symthaea_core::hdc::substrate_independence::CorticalRegion,
            SubstrateType,
        >,
    >,

    // ── Energy Budget ─────────────────────────────────────────────────
    /// Enable substrate energy budget tracking.
    /// When true, the substrate manager tracks cumulative energy expenditure
    /// based on substrate energy_per_op and marks consciousness as non-viable
    /// when budget is exceeded.
    /// Default: false.
    pub enable_energy_budget: bool,
    /// Energy budget in joules per second (optional).
    /// When set, limits cognitive throughput based on substrate energy characteristics.
    pub energy_budget_joules_per_sec: Option<f64>,

    /// Transition smoothing alpha for substrate switches [0.0, 1.0].
    /// 1.0 = instant switch (backward compatible default).
    /// 0.1 = gradual EMA blend over ~10 cycles (set by `enable_substrate_simulation()`).
    /// Science: Bostrom (2003) gradual substrate transfer.
    pub substrate_transition_alpha: f32,

    /// Enable thermal-adaptive CfC frequency modulation.
    /// When enabled, platform thermal reports (via ThermalBridge channel)
    /// modulate the CfC delta_t as the 10th tau factor.
    /// Science: Angilletta (2009) thermal performance curves.
    /// Default: false (desktop). Set true for mobile profiles.
    pub enable_thermal_adaptation: bool,

    // ── Federation ──────────────────────────────────────────────────────
    /// Enable federated learning coordinator (requires async runtime).
    pub federation_enabled: bool,
    /// Federated sync round interval in milliseconds (default: 30000 = 30s).
    pub federation_round_interval_ms: u64,

    // ── Vision Manifold Integration ─────────────────────────────────────
    /// Enable the internal VisionBridge in the cognitive loop (default false).
    /// When true, the cognitive loop creates a VisionBridge and processes
    /// injected frames through the vision manifold each cycle.
    #[cfg(feature = "vision-manifold")]
    pub enable_vision_manifold: bool,
    /// Vision frame width (default 64). Only used when enable_vision_manifold is true.
    #[cfg(feature = "vision-manifold")]
    pub vision_frame_width: u32,
    /// Vision frame height (default 64). Only used when enable_vision_manifold is true.
    #[cfg(feature = "vision-manifold")]
    pub vision_frame_height: u32,
    /// Enable vision predictive coding hierarchy (default false).
    #[cfg(feature = "vision-manifold")]
    pub enable_vision_predictive_hierarchy: bool,
    /// Scene memory coherence threshold (default 0.7).
    #[cfg(feature = "vision-manifold")]
    pub scene_memory_coherence_threshold: f32,
    /// Scene memory error threshold (default 0.1).
    #[cfg(feature = "vision-manifold")]
    pub scene_memory_error_threshold: f32,
    /// Scene memory dampening factor (default 0.5).
    #[cfg(feature = "vision-manifold")]
    pub scene_memory_dampen_factor: f32,
    /// Enable cross-manifold predictor (vision→cognitive Hebbian mapping, default false).
    #[cfg(feature = "vision-manifold")]
    pub enable_cross_manifold_predictor: bool,

    // ── Foveation Dispatch ──────────────────────────────────────────────
    /// Enable foveation attention dispatch (requires vision-manifold feature).
    /// When true AND the `foveation` feature is compiled, foveation dispatch
    /// runs in the perception phase. Default: false.
    #[cfg(feature = "foveation")]
    pub enable_foveation: bool,
    /// Maximum foveation dispatches per cycle (attention budget).
    /// Prevents over-allocation of ventral pipeline resources. Default: 3.
    #[cfg(feature = "foveation")]
    pub foveation_max_dispatches: u8,

    // ── Sovereign Inoculation ──────────────────────────────────────────
    /// Enable sovereign mesh-time consensus (requires `mesh` feature).
    #[serde(default)]
    pub enable_mesh_time: bool,
    /// Enable sovereign name resolution (requires `mesh` feature).
    #[serde(default)]
    pub enable_name_resolution: bool,
    /// Name cache size for mesh name resolver. Default: 256.
    #[serde(default = "default_name_cache_size")]
    pub name_cache_size: usize,

    // ── Chronobiology ──────────────────────────────────────────────────
    /// Timezone offset in hours from UTC (e.g., -5.0 for CDT, +9.0 for JST).
    /// Biorhythm stores time in UTC internally; this offset is applied via
    /// `effective_hour()` for circadian phase computation.
    /// Default: 0.0 (UTC). Set to `auto_detect_timezone()` for system locale.
    #[serde(default)]
    pub timezone_offset_hours: f64,

    // ── Knowledge Engine ────────────────────────────────────────────────
    /// Enable the knowledge engine for general-purpose reasoning.
    /// When true, the cognitive loop extracts structured facts from input,
    /// encodes them as HDC vectors, stores in a temporal knowledge graph,
    /// builds causal DAG edges from extracted relations, and grows an
    /// adaptive ontology of learned primitives.
    /// Science: Kanerva (2009) HDC, Pearl (2009) Causality, Carey (2009) conceptual change.
    #[serde(default)]
    pub enable_knowledge_engine: bool,

    /// Maximum facts in the knowledge graph (default: 10,000).
    #[serde(default = "default_knowledge_graph_capacity")]
    pub knowledge_graph_capacity: usize,

    /// Maximum causal edges in the causal bridge (default: 5,000).
    #[serde(default = "default_knowledge_causal_capacity")]
    pub knowledge_causal_capacity: usize,

    /// How many top-k search results to return per query (default: 5).
    #[serde(default = "default_knowledge_search_top_k")]
    pub knowledge_search_top_k: usize,

    /// Maximum learned primitives in the adaptive ontology (default: 500).
    #[serde(default = "default_knowledge_ontology_max")]
    pub knowledge_ontology_max: usize,

    /// Path to SQLite database for persistent knowledge storage.
    /// When set, facts and causal edges are saved periodically and loaded on startup.
    #[serde(default)]
    pub knowledge_db_path: Option<String>,

    /// Enable FHE collective wisdom pool for privacy-preserving peer learning.
    /// Feature-gated behind `fhe-wisdom`.
    #[cfg(feature = "fhe-wisdom")]
    pub fhe_wisdom_enabled: bool,

    /// Minimum contributions before aggregation (threshold k).
    #[cfg(feature = "fhe-wisdom")]
    pub fhe_threshold_k: usize,

    /// Aggregation interval in cycles.
    #[cfg(feature = "fhe-wisdom")]
    pub fhe_aggregation_interval: usize,

    // ── Embodiment Bridge ────────────────────────────────────────────────
    /// Override the attention budget (microseconds) used for cycle-time enforcement.
    /// When `Some(us)`, replaces the default 50ms `ATTENTION_BUDGET_US` constant.
    /// Set to a large value (e.g., 60_000_000 = 60s) for offline telemetry / debug
    /// builds where cycles naturally exceed the real-time budget without indicating
    /// a problem. Prevents the budget-exceeded → safety-escalation doom loop.
    #[serde(default)]
    pub attention_budget_override_us: Option<u64>,

    /// Which embodiment platform to use for proprioceptive loop closure.
    /// Default: `None` (disembodied cognitive loop).
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    #[serde(default)]
    pub embodiment_platform: super::motor_bridge::EmbodimentPlatform,

    /// Blend weight for proprioceptive HV injection (0.0–1.0). Default: 0.1.
    ///
    /// Empirically optimized via 11-point sweep (0.0–1.0):
    /// weight=0.1 → Phi=0.757, weight=0.2 → ~0.62, weight=0.3 → ~0.44.
    /// Light proprioceptive feedback grounds consciousness; heavy feedback
    /// floods the CfC with prediction errors that reduce Phi.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    #[serde(default = "default_embodiment_blend")]
    pub embodiment_blend_weight: f32,

    /// Embodiment step interval in cognitive cycles. Default: 1.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    #[serde(default = "default_embodiment_interval")]
    pub embodiment_step_interval: usize,
}

#[cfg(any(
    feature = "humanoid",
    feature = "helicopter",
    feature = "flight",
    feature = "vehicle",
    feature = "auv",
    feature = "manipulator",
    feature = "exoskeleton",
    feature = "surgical",
    feature = "orbital",
    feature = "quadruped",
    feature = "subterranean",
    feature = "infrastructure",
    feature = "scavenger",
    feature = "agribot",
    feature = "biota",
    feature = "clime",
    feature = "phone"
))]
fn default_embodiment_blend() -> f32 {
    0.1 // Optimized via weight sweep: 0.1 → Phi=0.757 (was 0.2 → ~0.62)
}

#[cfg(any(
    feature = "humanoid",
    feature = "helicopter",
    feature = "flight",
    feature = "vehicle",
    feature = "auv",
    feature = "manipulator",
    feature = "exoskeleton",
    feature = "surgical",
    feature = "orbital",
    feature = "quadruped",
    feature = "subterranean",
    feature = "infrastructure",
    feature = "scavenger",
    feature = "agribot",
    feature = "biota",
    feature = "clime",
    feature = "phone"
))]
fn default_embodiment_interval() -> usize {
    1
}

impl Default for CognitiveLoopConfig {
    fn default() -> Self {
        Self {
            encoder_config: PredictiveEncoderConfig::default(),
            cfc_config: CfCConfig::default(),
            hdc_ltc_config: HdcLtcBridgeConfig {
                hdc_dim: 2048,
                adaptive_dim: Some(crate::hdc_ltc_bridge::AdaptiveDimConfig::default()),
                ..HdcLtcBridgeConfig::default()
            },
            temporal_backend: TemporalBackend::default(),
            hierarchical_cfc_config:
                crate::dynamics::hierarchical_cfc::HierarchicalCfCConfig::default(),
            learning_threshold: 0.05,
            buffer_size: 1000,
            domain_profile: crate::domain::DomainProfile::default(),
            enable_consolidation: true,
            target_frequency: 50.0, // 50 Hz
            max_cycles_before_reset: 100000,
            genesis_phrase: None,
            training_method: TrainingMethod::default(),
            async_training: true,
            enable_online_learning: false,
            causal_enhancement: false,
            causal_discovery_interval: 100,
            episodic_replay_training: true,
            memory_graduation: true,
            memory_db_path: None,
            aesthetic_memory_path: None,
            epistemic_auditor_db_path: None,
            episodic_replay_config: crate::memory::episodic_replay::EpisodicReplayConfig::default(),
            enable_surprise_exploration: true,

            enable_prefrontal: true,
            enable_meta_cognition: true,
            enable_narrative_self: true,
            enable_autobiography_narration: false,
            enable_virtual_body: true,
            enable_predictive_self: true,
            enable_attention_schema: true,
            enable_gwt: true,
            enable_resonance: true,
            enable_quantum_coherence: true,
            enable_temporal_consciousness: true,
            enable_embodied_cognition: true,
            enable_narrative_gwt: true,
            enable_dream_replay: true,
            enable_predictive_processing: true,
            enable_cross_modal_binding: true,
            enable_affective_bridge: true,
            enable_user_state_inference: true,
            enable_coherence_field: false,
            enable_consciousness_thermodynamics: true,
            enable_phenomenal_binding: true,
            enable_hierarchical_free_energy: true,
            enable_trajectory_planning: true,
            trajectory_horizon_seconds: default_trajectory_horizon(),
            trajectory_planning_interval: default_trajectory_interval(),
            enable_hierarchical_bundling: true,
            enable_contextual_weights: true,
            enable_phi_attention: true,
            enable_negation_detection: true,
            enable_visualization: true,
            enable_soul_alignment: true,
            enable_primitive_consciousness: true,
            enable_safety_gateway: true,
            enable_metacognitive_monitoring: true,
            enable_resonator_recall: true,
            resonator_growth_interval: 50,
            resonator_novelty_threshold: 0.7,
            resonator_max_symbols: 100,
            enable_psi_attestation: true,
            agent_did: Some("did:symthaea:default-agent".into()),
            attestation_buffer_capacity: 64,
            trace_feedback: false,
            #[cfg(feature = "nurture")]
            enable_nurture_attachment: false,
            moral_anomaly_config: MoralAnomalyConfig::default(),
            enable_moral_anomaly_response: false,
            #[cfg(feature = "semantic-encoder")]
            enable_semantic_encoder: false,
            #[cfg(feature = "semantic-encoder")]
            semantic_encoder_ollama: None,
            #[cfg(feature = "semantic-encoder")]
            semantic_thought_blend: 0.0,
            enable_validation_overlay: false,
            validation_skepticism_floor: 0.5,
            enable_substrate_speed_modulation: false,
            enable_substrate_encoding_noise: false,
            enable_phi_tau_feedback: true,
            substrate_type: SubstrateType::SiliconDigital,
            substrate_composition: None,
            per_region_substrates: None,
            #[cfg(feature = "physics-bridge")]
            enable_physics_bridge: false,
            #[cfg(feature = "physics-bridge")]
            physics_bridge_query_interval: 10,
            #[cfg(feature = "physics-bridge")]
            physics_bridge_blend_weight: 0.1,
            #[cfg(feature = "therapeutic")]
            enable_therapeutic: true, // On by default when feature is compiled in
            #[cfg(feature = "therapeutic")]
            therapeutic_crisis_threshold: 0.62,
            #[cfg(feature = "therapeutic")]
            therapeutic_text_crisis_detection: true, // Safety-critical: on by default
            enable_voice_synthesis: false,
            #[cfg(feature = "ssm_language")]
            enable_broca_language: true,
            #[cfg(feature = "ssm_language")]
            broca_checkpoint_path: None,
            #[cfg(feature = "ssm_language")]
            broca_multi_turn_depth: 4,
            #[cfg(feature = "ssm_language")]
            enable_broca_nsm_semantic: true,
            #[cfg(feature = "ssm_language")]
            enable_broca_nsm_gate: true,
            enable_hodge_decomposition: true,
            hodge_interval: default_hodge_interval(),
            enable_energy_budget: true,
            energy_budget_joules_per_sec: None,
            substrate_transition_alpha: super::thresholds::SUBSTRATE_TRANSITION_ALPHA_DEFAULT
                as f32,
            enable_thermal_adaptation: true,
            federation_enabled: false,
            federation_round_interval_ms: 30_000,
            #[cfg(feature = "vision-manifold")]
            enable_vision_manifold: true,
            #[cfg(feature = "vision-manifold")]
            vision_frame_width: 64,
            #[cfg(feature = "vision-manifold")]
            vision_frame_height: 64,
            #[cfg(feature = "vision-manifold")]
            enable_vision_predictive_hierarchy: true,
            #[cfg(feature = "vision-manifold")]
            scene_memory_coherence_threshold: 0.7,
            #[cfg(feature = "vision-manifold")]
            scene_memory_error_threshold: 0.1,
            #[cfg(feature = "vision-manifold")]
            scene_memory_dampen_factor: 0.5,
            #[cfg(feature = "vision-manifold")]
            enable_cross_manifold_predictor: false,
            #[cfg(feature = "foveation")]
            enable_foveation: true,
            #[cfg(feature = "foveation")]
            foveation_max_dispatches: 3,
            timezone_offset_hours: 0.0,
            enable_mesh_time: true,
            enable_name_resolution: false,
            name_cache_size: 256,
            enable_knowledge_engine: true,
            knowledge_graph_capacity: 10_000,
            knowledge_causal_capacity: 5_000,
            knowledge_search_top_k: 5,
            knowledge_ontology_max: 500,
            knowledge_db_path: None,

            #[cfg(feature = "fhe-wisdom")]
            fhe_wisdom_enabled: false,
            #[cfg(feature = "fhe-wisdom")]
            fhe_threshold_k: 3,
            #[cfg(feature = "fhe-wisdom")]
            fhe_aggregation_interval: 100,
            attention_budget_override_us: None,
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "subterranean",
                feature = "infrastructure",
                feature = "scavenger",
                feature = "agribot",
                feature = "biota",
                feature = "clime",
                feature = "phone"
            ))]
            embodiment_platform: super::motor_bridge::EmbodimentPlatform::None,
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "subterranean",
                feature = "infrastructure",
                feature = "scavenger",
                feature = "agribot",
                feature = "biota",
                feature = "clime",
                feature = "phone"
            ))]
            embodiment_blend_weight: 0.1,
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "subterranean",
                feature = "infrastructure",
                feature = "scavenger",
                feature = "agribot",
                feature = "biota",
                feature = "clime",
                feature = "phone"
            ))]
            embodiment_step_interval: 1,
        }
    }
}

fn default_federation_round_interval_ms() -> u64 {
    30_000
}
fn default_knowledge_graph_capacity() -> usize {
    10_000
}
fn default_knowledge_causal_capacity() -> usize {
    5_000
}
fn default_knowledge_search_top_k() -> usize {
    5
}
fn default_knowledge_ontology_max() -> usize {
    500
}
fn default_name_cache_size() -> usize {
    256
}

fn default_trajectory_horizon() -> f64 {
    0.5
}

fn default_trajectory_interval() -> u64 {
    10
}

fn default_hodge_interval() -> usize {
    47
}

impl CognitiveLoopConfig {
    /// Create configuration with CfC backend (default)
    pub fn with_cfc() -> Self {
        Self {
            temporal_backend: TemporalBackend::CfC,
            ..Default::default()
        }
    }

    /// Auto-detect timezone from the system locale and apply it.
    ///
    /// Call once at startup. The detected offset is stored as explicit state;
    /// subsequent timezone changes should go through `CognitiveLoopService::set_timezone()`.
    pub fn with_system_timezone(mut self) -> Self {
        self.timezone_offset_hours = crate::chronobiology::Biorhythm::detect_system_timezone();
        self
    }

    /// Override the active operating domain.
    pub fn with_domain_profile(mut self, domain_profile: crate::domain::DomainProfile) -> Self {
        self.domain_profile = domain_profile;
        self
    }

    /// Create a configuration scoped to a specific operating domain.
    pub fn for_domain(domain_profile: crate::domain::DomainProfile) -> Self {
        Self {
            domain_profile,
            ..Default::default()
        }
    }

    /// Create configuration for a platform's preferred operating domain.
    pub fn for_platform(platform: symthaea_core::embodiment::EmbodimentPlatform) -> Self {
        let capability = crate::domain::PlatformCapabilityProfile::for_platform(platform);
        #[allow(unused_mut)]
        let mut config = Self {
            domain_profile: capability.preferred_domain_profile(),
            ..Default::default()
        };
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "subterranean",
            feature = "infrastructure",
            feature = "scavenger",
            feature = "agribot",
            feature = "biota",
            feature = "clime",
            feature = "phone"
        ))]
        {
            config.embodiment_platform = platform;
        }
        config
    }

    /// Create configuration for an explicit platform/domain pairing.
    ///
    /// Unsupported combinations fall back to the platform's preferred domain.
    pub fn for_platform_domain(
        platform: symthaea_core::embodiment::EmbodimentPlatform,
        domain_profile: crate::domain::DomainProfile,
    ) -> Self {
        let capability = crate::domain::PlatformCapabilityProfile::for_platform(platform);
        let resolved_domain = if capability.supports_domain(&domain_profile.kind) {
            domain_profile
        } else {
            capability.preferred_domain_profile()
        };
        #[allow(unused_mut)]
        let mut config = Self {
            domain_profile: resolved_domain,
            ..Default::default()
        };
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "subterranean",
            feature = "infrastructure",
            feature = "scavenger",
            feature = "agribot",
            feature = "biota",
            feature = "clime",
            feature = "phone"
        ))]
        {
            config.embodiment_platform = platform;
        }
        config
    }

    /// Create configuration with HdcLtcUnified backend
    pub fn with_hdc_ltc_unified() -> Self {
        Self {
            temporal_backend: TemporalBackend::HdcLtcUnified,
            ..Default::default()
        }
    }

    /// Create configuration with HdcLtcUnified backend optimized for speed
    pub fn with_hdc_ltc_fast() -> Self {
        Self {
            temporal_backend: TemporalBackend::HdcLtcUnified,
            hdc_ltc_config: HdcLtcBridgeConfig::fast(),
            ..Default::default()
        }
    }

    /// Create configuration with HdcLtcUnified backend optimized for accuracy
    pub fn with_hdc_ltc_accurate() -> Self {
        Self {
            temporal_backend: TemporalBackend::HdcLtcUnified,
            hdc_ltc_config: HdcLtcBridgeConfig::accurate(),
            ..Default::default()
        }
    }

    /// Enable full substrate simulation: speed modulation, encoding noise,
    /// validation overlay, and energy budget tracking.
    ///
    /// When enabled, switching substrate types via `SubstrateManager::reconfigure_substrate()`
    /// produces *emergent* behavioral differences:
    /// - CfC temporal dynamics scale with substrate operation speed (tau_factor)
    /// - HDC encoding degrades on scale-constrained substrates (bit-flip + Gaussian noise)
    /// - Consciousness feasibility is tempered by honest evidence confidence
    /// - Energy budget tracks cumulative joules per substrate
    pub fn enable_substrate_simulation(&mut self) -> &mut Self {
        self.enable_substrate_speed_modulation = true;
        self.enable_substrate_encoding_noise = true;
        self.enable_validation_overlay = true;
        self.enable_energy_budget = true;
        self.substrate_transition_alpha =
            super::thresholds::SUBSTRATE_TRANSITION_ALPHA_SIMULATION as f32;
        self
    }
}

// ConsciousnessProfile is in consciousness.rs, re-exported at module level.

impl CognitiveLoopConfig {
    /// Create a configuration from a named consciousness profile.
    pub fn from_profile(profile: ConsciousnessProfile) -> Self {
        let mut config = Self::with_cfc();
        profile.apply(&mut config);
        config
    }

    /// Count the number of enabled consciousness modules.
    #[cfg(test)]
    fn count_enabled(&self) -> usize {
        let bools = [
            self.enable_virtual_body,
            self.enable_surprise_exploration,
            self.enable_prefrontal,
            self.enable_meta_cognition,
            self.enable_narrative_self,
            self.enable_predictive_self,
            self.enable_attention_schema,
            self.enable_gwt,
            self.enable_resonance,
            self.enable_quantum_coherence,
            self.enable_temporal_consciousness,
            self.enable_embodied_cognition,
            self.enable_narrative_gwt,
            self.enable_dream_replay,
            self.enable_predictive_processing,
            self.enable_cross_modal_binding,
            self.enable_affective_bridge,
            self.enable_user_state_inference,
            self.enable_coherence_field,
            self.enable_consciousness_thermodynamics,
            self.enable_phenomenal_binding,
            self.enable_hierarchical_free_energy,
            self.enable_hierarchical_bundling,
            self.enable_contextual_weights,
            self.enable_phi_attention,
            self.enable_negation_detection,
            self.enable_primitive_consciousness,
            self.enable_resonator_recall,
            self.enable_psi_attestation,
            self.causal_enhancement,
            self.episodic_replay_training,
            self.enable_soul_alignment,
        ];
        bools.iter().filter(|&&b| b).count()
    }

    /// Validate configuration for dependency issues.
    ///
    /// Returns a list of warnings for soft dependency violations — cases where
    /// a module is enabled but its upstream dependency is disabled, causing the
    /// module to operate at reduced capacity or be functionally inert.
    ///
    /// All dependencies are handled gracefully at runtime (None checks), so
    /// these are warnings, not hard errors.
    pub fn validate_dependencies(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // temporal_consciousness passes narrative_self.as_ref() and predictive_self.as_ref()
        if self.enable_temporal_consciousness {
            if !self.enable_narrative_self {
                warnings.push(
                    "enable_temporal_consciousness without enable_narrative_self: \
                     identity continuity tracking will be disabled"
                        .into(),
                );
            }
            if !self.enable_predictive_self {
                warnings.push(
                    "enable_temporal_consciousness without enable_predictive_self: \
                     action prediction trajectory will be disabled"
                        .into(),
                );
            }
        }

        // predictive_self.observe(narrative) needs narrative_self
        if self.enable_predictive_self && !self.enable_narrative_self {
            warnings.push(
                "enable_predictive_self without enable_narrative_self: \
                 self-model observation will be disabled"
                    .into(),
            );
        }

        // autobiography narrator reads narrative_self.autobio.life_story
        if self.enable_autobiography_narration && !self.enable_narrative_self {
            warnings.push(
                "enable_autobiography_narration without enable_narrative_self: \
                 there is no life_story to narrate"
                    .into(),
            );
        }

        // embodied_cognition.update_interoception() needs virtual_body state
        if self.enable_embodied_cognition && !self.enable_virtual_body {
            warnings.push(
                "enable_embodied_cognition without enable_virtual_body: \
                 body schema updates will be disabled"
                    .into(),
            );
        }

        // cross_modal_binding checks affective_bridge.is_some() for affective modality
        if self.enable_cross_modal_binding && !self.enable_affective_bridge {
            warnings.push(
                "enable_cross_modal_binding without enable_affective_bridge: \
                 affective modality will be absent from binding"
                    .into(),
            );
        }

        // predictive_processing applies affective modulation when affective_bridge present
        if self.enable_predictive_processing && !self.enable_affective_bridge {
            warnings.push(
                "enable_predictive_processing without enable_affective_bridge: \
                 precision-weighted affective modulation will be disabled"
                    .into(),
            );
        }

        // psi_attestation silently skips without agent_did
        if self.enable_psi_attestation && self.agent_did.is_none() {
            warnings.push(
                "enable_psi_attestation without agent_did: \
                 attestation records will not be generated"
                    .into(),
            );
        }

        // dream_replay benefits from surprise_exploration (records surprise events)
        if self.enable_dream_replay && !self.enable_surprise_exploration {
            warnings.push(
                "enable_dream_replay without enable_surprise_exploration: \
                 dream replay will only use prediction error, not surprise events"
                    .into(),
            );
        }

        // foveation dispatch requires vision-manifold for frame/saliency data
        #[cfg(all(feature = "foveation", feature = "vision-manifold"))]
        if self.enable_foveation && !self.enable_vision_manifold {
            warnings.push(
                "enable_foveation without enable_vision_manifold: \
                 foveation dispatch requires vision bridge for frame and saliency data"
                    .into(),
            );
        }

        // narrative_gwt is the capstone integrating narrative_self + gwt
        if self.enable_narrative_gwt {
            if !self.enable_narrative_self {
                warnings.push(
                    "enable_narrative_gwt without enable_narrative_self: \
                     narrative governance layer will lack self-model data"
                        .into(),
                );
            }
            if !self.enable_gwt {
                warnings.push(
                    "enable_narrative_gwt without enable_gwt: \
                     workspace broadcast coherence will be unavailable"
                        .into(),
                );
            }
        }

        warnings
    }

    /// Validate configuration parameter ranges.
    ///
    /// Returns `Err` if any parameter is out of its valid range. This catches
    /// hard errors (invalid values that would cause panics or incorrect behavior),
    /// unlike [`validate_dependencies`] which returns soft warnings.
    ///
    /// Checks performed:
    /// - CfC sub-config validation (dimensions, learning rate, delta_t)
    /// - `learning_threshold` in [0.0, 1.0]
    /// - `buffer_size` > 0
    /// - `target_frequency` positive and finite
    /// - `max_cycles_before_reset` > 0
    /// - `causal_discovery_interval` > 0
    /// - `resonator_novelty_threshold` in (0.0, 1.0]
    /// - `resonator_max_symbols` > 0
    /// - `attestation_buffer_capacity` > 0
    pub fn validate(&self) -> Result<(), String> {
        // Validate nested CfC config
        self.cfc_config
            .validate()
            .map_err(|e| format!("CognitiveLoopConfig: {e}"))?;

        if self.learning_threshold < 0.0
            || self.learning_threshold > 1.0
            || !self.learning_threshold.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: learning_threshold must be in [0.0, 1.0], got {}",
                self.learning_threshold
            ));
        }
        if self.buffer_size == 0 {
            return Err("CognitiveLoopConfig: buffer_size must be > 0".into());
        }
        if self.target_frequency <= 0.0 || !self.target_frequency.is_finite() {
            return Err(format!(
                "CognitiveLoopConfig: target_frequency must be positive and finite, got {}",
                self.target_frequency
            ));
        }
        if self.max_cycles_before_reset == 0 {
            return Err("CognitiveLoopConfig: max_cycles_before_reset must be > 0".into());
        }
        if self.causal_discovery_interval == 0 {
            return Err("CognitiveLoopConfig: causal_discovery_interval must be > 0".into());
        }
        if self.resonator_novelty_threshold <= 0.0
            || self.resonator_novelty_threshold > 1.0
            || !self.resonator_novelty_threshold.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: resonator_novelty_threshold must be in (0.0, 1.0], got {}",
                self.resonator_novelty_threshold
            ));
        }
        if self.resonator_max_symbols == 0 {
            return Err("CognitiveLoopConfig: resonator_max_symbols must be > 0".into());
        }
        if self.attestation_buffer_capacity == 0 {
            return Err("CognitiveLoopConfig: attestation_buffer_capacity must be > 0".into());
        }
        if self.validation_skepticism_floor < 0.0
            || self.validation_skepticism_floor > 1.0
            || !self.validation_skepticism_floor.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: validation_skepticism_floor must be in [0.0, 1.0], got {}",
                self.validation_skepticism_floor
            ));
        }
        if self.substrate_transition_alpha < 0.0
            || self.substrate_transition_alpha > 1.0
            || !self.substrate_transition_alpha.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: substrate_transition_alpha must be in [0.0, 1.0], got {}",
                self.substrate_transition_alpha
            ));
        }
        #[cfg(feature = "physics-bridge")]
        if self.physics_bridge_blend_weight < 0.0
            || self.physics_bridge_blend_weight > 1.0
            || !self.physics_bridge_blend_weight.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: physics_bridge_blend_weight must be in [0.0, 1.0], got {}",
                self.physics_bridge_blend_weight
            ));
        }
        #[cfg(feature = "therapeutic")]
        if self.therapeutic_crisis_threshold < 0.01
            || self.therapeutic_crisis_threshold > 0.95
            || !self.therapeutic_crisis_threshold.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: therapeutic_crisis_threshold must be in [0.01, 0.95], got {}",
                self.therapeutic_crisis_threshold
            ));
        }
        #[cfg(feature = "vision-manifold")]
        if self.scene_memory_coherence_threshold < 0.0
            || self.scene_memory_coherence_threshold > 1.0
            || !self.scene_memory_coherence_threshold.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: scene_memory_coherence_threshold must be in [0.0, 1.0], got {}",
                self.scene_memory_coherence_threshold
            ));
        }
        #[cfg(feature = "vision-manifold")]
        if self.scene_memory_error_threshold < 0.0
            || self.scene_memory_error_threshold > 1.0
            || !self.scene_memory_error_threshold.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: scene_memory_error_threshold must be in [0.0, 1.0], got {}",
                self.scene_memory_error_threshold
            ));
        }
        #[cfg(feature = "vision-manifold")]
        if self.scene_memory_dampen_factor < 0.0
            || self.scene_memory_dampen_factor > 1.0
            || !self.scene_memory_dampen_factor.is_finite()
        {
            return Err(format!(
                "CognitiveLoopConfig: scene_memory_dampen_factor must be in [0.0, 1.0], got {}",
                self.scene_memory_dampen_factor
            ));
        }
        // Validate moral anomaly config
        self.moral_anomaly_config
            .validate()
            .map_err(|e| format!("CognitiveLoopConfig: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::domain::DomainProfile;
    use symthaea_core::embodiment::EmbodimentPlatform;

    // ═══════════════════════════════════════════════════════════════════════
    // CfCConfig validation
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn cfc_default_validates() {
        assert!(CfCConfig::default().validate().is_ok());
    }

    #[test]
    fn cfc_zero_neurons_rejected() {
        let mut c = CfCConfig::default();
        c.num_neurons = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_zero_input_dim_rejected() {
        let mut c = CfCConfig::default();
        c.input_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_negative_lr_rejected() {
        let mut c = CfCConfig::default();
        c.learning_rate = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_zero_lr_rejected() {
        let mut c = CfCConfig::default();
        c.learning_rate = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_nan_lr_rejected() {
        let mut c = CfCConfig::default();
        c.learning_rate = f32::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_inf_lr_rejected() {
        let mut c = CfCConfig::default();
        c.learning_rate = f32::INFINITY;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_negative_delta_t_rejected() {
        let mut c = CfCConfig::default();
        c.delta_t = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_empty_horizons_rejected() {
        let mut c = CfCConfig::default();
        c.prediction_horizons = vec![];
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_negative_horizon_rejected() {
        let mut c = CfCConfig::default();
        c.prediction_horizons = vec![0.02, -0.1];
        assert!(c.validate().is_err());
    }

    #[test]
    fn cfc_lr_at_max_valid() {
        let mut c = CfCConfig::default();
        c.learning_rate = 1.0;
        assert!(c.validate().is_ok());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CognitiveLoopConfig validation
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn config_default_validates() {
        assert!(CognitiveLoopConfig::default().validate().is_ok());
    }

    #[test]
    fn config_domain_defaults_to_empty() {
        let c = CognitiveLoopConfig::default();
        assert_eq!(c.domain_profile.kind, "");
    }

    #[test]
    fn config_for_domain_overrides_default_domain() {
        let c = CognitiveLoopConfig::for_domain(DomainProfile::underwater());
        assert_eq!(c.domain_profile.kind, c.domain_profile.kind.clone());
    }

    #[test]
    fn config_for_platform_uses_platform_preferred_domain() {
        let c = CognitiveLoopConfig::for_platform(EmbodimentPlatform::Auv);
        assert_eq!(c.domain_profile.kind, c.domain_profile.kind.clone());
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "phone"
        ))]
        assert_eq!(c.embodiment_platform, EmbodimentPlatform::Auv);
    }

    #[test]
    fn unsupported_platform_domain_pair_falls_back_to_preferred_domain() {
        let c = CognitiveLoopConfig::for_platform_domain(
            EmbodimentPlatform::Auv,
            DomainProfile::deep_space(),
        );
        assert_eq!(c.domain_profile.kind, c.domain_profile.kind.clone());
    }

    #[test]
    fn config_negative_learning_threshold_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.learning_threshold = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_learning_threshold_above_one_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.learning_threshold = 1.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_buffer_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.buffer_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_frequency_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.target_frequency = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_negative_frequency_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.target_frequency = -50.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_causal_interval_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.causal_discovery_interval = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_novelty_threshold_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.resonator_novelty_threshold = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_max_symbols_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.resonator_max_symbols = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_attestation_cap_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.attestation_buffer_capacity = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_propagates_cfc_error() {
        let mut c = CognitiveLoopConfig::default();
        c.cfc_config.num_neurons = 0;
        let err = c.validate().unwrap_err();
        assert!(
            err.contains("num_neurons"),
            "error should mention CfC field: {}",
            err
        );
    }

    #[test]
    fn config_skepticism_floor_out_of_range_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.validation_skepticism_floor = 1.5;
        assert!(c.validate().is_err());
        c.validation_skepticism_floor = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_substrate_transition_alpha_out_of_range_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.substrate_transition_alpha = 2.0;
        assert!(c.validate().is_err());
        c.substrate_transition_alpha = -0.5;
        assert!(c.validate().is_err());
    }

    #[test]
    #[cfg(feature = "physics-bridge")]
    fn config_physics_blend_weight_out_of_range_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.physics_bridge_blend_weight = f32::NAN;
        assert!(c.validate().is_err());
        c.physics_bridge_blend_weight = 1.01;
        assert!(c.validate().is_err());
    }

    #[test]
    #[cfg(feature = "therapeutic")]
    fn config_therapeutic_crisis_threshold_out_of_range_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.therapeutic_crisis_threshold = 0.0;
        assert!(c.validate().is_err());
        c.therapeutic_crisis_threshold = 0.99;
        assert!(c.validate().is_err());
    }

    #[test]
    #[cfg(feature = "vision-manifold")]
    fn config_scene_memory_thresholds_out_of_range_rejected() {
        let mut c = CognitiveLoopConfig::default();
        c.scene_memory_coherence_threshold = -0.1;
        assert!(c.validate().is_err());

        let mut c = CognitiveLoopConfig::default();
        c.scene_memory_error_threshold = 1.5;
        assert!(c.validate().is_err());

        let mut c = CognitiveLoopConfig::default();
        c.scene_memory_dampen_factor = f32::INFINITY;
        assert!(c.validate().is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Factory methods
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn factory_with_cfc() {
        let c = CognitiveLoopConfig::with_cfc();
        assert_eq!(c.temporal_backend, TemporalBackend::CfC);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn factory_with_hdc_ltc_unified() {
        let c = CognitiveLoopConfig::with_hdc_ltc_unified();
        assert_eq!(c.temporal_backend, TemporalBackend::HdcLtcUnified);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn factory_with_hdc_ltc_fast() {
        let c = CognitiveLoopConfig::with_hdc_ltc_fast();
        assert_eq!(c.temporal_backend, TemporalBackend::HdcLtcUnified);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn factory_with_hdc_ltc_accurate() {
        let c = CognitiveLoopConfig::with_hdc_ltc_accurate();
        assert_eq!(c.temporal_backend, TemporalBackend::HdcLtcUnified);
        assert!(c.validate().is_ok());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ConsciousnessProfile
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn profile_minimal_enables_only_virtual_body() {
        let c = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Minimal);
        assert!(c.enable_virtual_body);
        assert_eq!(c.count_enabled(), 1);
    }

    #[test]
    fn profile_standard_subset_of_full() {
        let std = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
        let full = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
        assert!(
            std.count_enabled() < full.count_enabled(),
            "Standard {} should have fewer modules than Full {}",
            std.count_enabled(),
            full.count_enabled()
        );
    }

    #[test]
    fn profile_full_subset_of_research() {
        let full = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
        let research = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Research);
        assert!(
            full.count_enabled() < research.count_enabled(),
            "Full {} should have fewer modules than Research {}",
            full.count_enabled(),
            research.count_enabled()
        );
    }

    #[test]
    fn profile_research_enables_causal_and_episodic() {
        let c = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Research);
        assert!(c.causal_enhancement);
        assert!(c.episodic_replay_training);
        assert!(c.enable_psi_attestation);
        assert!(c.enable_user_state_inference);
    }

    #[test]
    fn profile_all_configs_validate() {
        for profile in [
            ConsciousnessProfile::Minimal,
            ConsciousnessProfile::Standard,
            ConsciousnessProfile::Full,
            ConsciousnessProfile::Research,
            ConsciousnessProfile::Mobile,
        ] {
            let c = CognitiveLoopConfig::from_profile(profile);
            assert!(
                c.validate().is_ok(),
                "{:?} profile should validate",
                profile
            );
        }
    }

    #[test]
    fn profile_mobile_power_aware_defaults() {
        let c = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Mobile);
        // Core consciousness enabled
        assert!(c.enable_virtual_body);
        assert!(c.enable_surprise_exploration);
        assert!(c.enable_prefrontal);
        assert!(c.enable_meta_cognition);
        assert!(c.enable_gwt);
        assert!(c.enable_affective_bridge);
        assert!(c.enable_narrative_self);
        // Power-aware tuning
        assert_eq!(c.target_frequency, 20.0);
        assert_eq!(c.cfc_config.num_neurons, 128);
        assert!(c.enable_energy_budget);
        assert!(c.enable_thermal_adaptation);
        // Expensive subsystems disabled
        assert!(!c.enable_dream_replay);
        assert!(!c.enable_phenomenal_binding);
        assert!(!c.enable_narrative_gwt);
        assert!(!c.enable_resonator_recall);
        assert!(!c.enable_predictive_processing);
    }

    #[test]
    fn profile_mobile_between_standard_and_full() {
        let std = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
        let mobile = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Mobile);
        let full = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
        // Mobile has affective_bridge (Standard doesn't) but not dream_replay (Full does)
        assert!(
            mobile.count_enabled() >= std.count_enabled(),
            "Mobile {} should have at least as many modules as Standard {}",
            mobile.count_enabled(),
            std.count_enabled()
        );
        assert!(
            mobile.count_enabled() < full.count_enabled(),
            "Mobile {} should have fewer modules than Full {}",
            mobile.count_enabled(),
            full.count_enabled()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Dependency validation
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn dependency_temporal_without_narrative() {
        let mut c = CognitiveLoopConfig::default();
        c.enable_temporal_consciousness = true;
        c.enable_narrative_self = false;
        let warnings = c.validate_dependencies();
        assert!(
            warnings.iter().any(|w| w.contains("narrative_self")),
            "should warn about missing narrative_self: {:?}",
            warnings
        );
    }

    #[test]
    fn dependency_temporal_without_predictive() {
        let mut c = CognitiveLoopConfig::default();
        c.enable_temporal_consciousness = true;
        c.enable_predictive_self = false;
        let warnings = c.validate_dependencies();
        assert!(warnings.iter().any(|w| w.contains("predictive_self")));
    }

    #[test]
    fn dependency_embodied_without_body() {
        let mut c = CognitiveLoopConfig::default();
        c.enable_virtual_body = false;
        c.enable_embodied_cognition = true;
        let warnings = c.validate_dependencies();
        assert!(warnings.iter().any(|w| w.contains("virtual_body")));
    }

    #[test]
    fn dependency_psi_without_did() {
        let mut c = CognitiveLoopConfig::default();
        c.enable_psi_attestation = true;
        c.agent_did = None;
        let warnings = c.validate_dependencies();
        assert!(warnings.iter().any(|w| w.contains("agent_did")));
    }

    #[test]
    fn dependency_default_no_warnings() {
        let c = CognitiveLoopConfig::default();
        let warnings = c.validate_dependencies();
        assert!(
            warnings.is_empty(),
            "default config should have no dependency warnings: {:?}",
            warnings
        );
    }

    #[test]
    fn dependency_full_profile_no_warnings() {
        let c = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
        let warnings = c.validate_dependencies();
        assert!(
            warnings.is_empty(),
            "Full profile should have no warnings: {:?}",
            warnings
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Serde round-trip
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn config_serde_roundtrip() {
        let original = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: CognitiveLoopConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.temporal_backend, original.temporal_backend);
        assert_eq!(restored.enable_gwt, original.enable_gwt);
        assert_eq!(restored.enable_prefrontal, original.enable_prefrontal);
    }

    #[test]
    fn temporal_backend_serde_roundtrip() {
        for backend in [TemporalBackend::CfC, TemporalBackend::HdcLtcUnified] {
            let json = serde_json::to_string(&backend).unwrap();
            let restored: TemporalBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, backend);
        }
    }

    #[test]
    fn training_method_serde_roundtrip() {
        for method in [
            TrainingMethod::Bptt,
            TrainingMethod::Spsa,
            TrainingMethod::BpttWithSpsaFallback,
        ] {
            let json = serde_json::to_string(&method).unwrap();
            let restored: TrainingMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, method);
        }
    }
}
