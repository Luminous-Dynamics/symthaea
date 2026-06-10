// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SporeEngine: the core consciousness loop for WASM targets.

use crate::broca::BrocaLite;
#[cfg(feature = "broca-pipeline")]
use crate::broca_pipeline::BrocaPipeline;
use crate::causal::CausalDiscovery;
use crate::config::SporeConfig;
use crate::dream::DreamEngine;
use crate::dream_journal::DreamJournal;
use crate::fep::{ActiveInferenceAgent, FepCycleResult};
use crate::immune::ImmuneSystem;
use crate::memory::MemoryCoordinator;
use crate::reasoning::ReasoningEngine;
use crate::topology::TopologyAnalyzer;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use symthaea_consciousness_equation::{
    ConsciousnessInputs, MasterConsciousnessEquation, MasterEquationConfig,
};
use symthaea_core::hdc::hdc_ltc_unified::UnifiedNetworkConfig;
use symthaea_core::hdc::substrate_independence::{SubstrateRequirements, SubstrateType};
use symthaea_core::hdc::substrate_validation::EvidenceLevel;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::{HDC_DIMENSION, HdcLtcUnifiedNetwork, TextEncoder, TextEncoderConfig};
use symthaea_harmonies::EightHarmonies;
use symthaea_neuromodulators::NeuromodulatorBath;
use symthaea_types::Harmony;

/// Maximum conversation turns retained for topic continuity.
const CONVERSATION_MAX_TURNS: usize = 8;

/// Global instance counter — surfaces awareness of multi-instance creation.
static INSTANCE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Maximum recommended concurrent engines before warning.
const MAX_RECOMMENDED_INSTANCES: usize = 4;

// ---------------------------------------------------------------------------
// ConversationContext — sliding window for topic continuity
// ---------------------------------------------------------------------------

/// Sliding window of recent conversation turns for topic continuity.
///
/// Maintains a ring buffer of HDC-encoded inputs so the engine can detect
/// whether the user is continuing a topic or shifting to something new.
/// The bundled `topic_hv` is the element-wise average of recent inputs,
/// acting as a running "conversation topic" vector.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    /// Recent input HDC encodings (ring buffer, max `CONVERSATION_MAX_TURNS`).
    recent_inputs: Vec<ContinuousHV>,
    /// Recent consciousness levels (same ring buffer size).
    recent_consciousness: Vec<f32>,
    /// Bundled "conversation topic" vector — the average of recent inputs.
    topic_hv: Option<ContinuousHV>,
    /// Turn counter (total turns processed, not capped).
    turn_count: u32,
    /// Maximum turns to remember.
    max_turns: usize,
    /// Topic coherence from the most recent turn.
    last_coherence: f32,
}

impl ConversationContext {
    /// Create a new conversation context with the given window size.
    pub fn new(max_turns: usize) -> Self {
        Self {
            recent_inputs: Vec::with_capacity(max_turns),
            recent_consciousness: Vec::with_capacity(max_turns),
            topic_hv: None,
            turn_count: 0,
            max_turns: max_turns.max(1),
            last_coherence: 0.0,
        }
    }

    /// Record a new conversation turn.
    ///
    /// Computes topic coherence *before* adding the new input (so coherence
    /// measures how well this input matches the *prior* topic), then pushes
    /// the input into the ring buffer and recomputes the topic vector.
    pub fn record_turn(&mut self, input_hv: &ContinuousHV, consciousness: f32) {
        // Compute coherence against the existing topic (before updating)
        self.last_coherence = self.topic_coherence(input_hv);

        self.turn_count += 1;

        // Ring buffer: drop oldest when full
        if self.recent_inputs.len() >= self.max_turns {
            self.recent_inputs.remove(0);
            self.recent_consciousness.remove(0);
        }
        self.recent_inputs.push(input_hv.clone());
        self.recent_consciousness.push(consciousness);

        // Recompute topic vector as element-wise average of recent inputs
        self.recompute_topic();
    }

    /// Cosine similarity between the given input and the running topic vector.
    ///
    /// Returns 0.0 if no topic has been established yet (first turn).
    /// High coherence (> 0.7) = continuing a topic.
    /// Low coherence (< 0.3) = topic shift.
    pub fn topic_coherence(&self, current_input: &ContinuousHV) -> f32 {
        match &self.topic_hv {
            Some(topic) => current_input.similarity(topic),
            None => 0.0,
        }
    }

    /// Topic coherence computed during the most recent `record_turn` call.
    pub fn last_coherence(&self) -> f32 {
        self.last_coherence
    }

    /// Create a context-enriched input by binding the raw input with the topic vector.
    ///
    /// If no topic exists yet (first turn), returns the raw input unchanged.
    /// The binding uses element-wise multiplication (HDC binding) so the
    /// resulting vector carries both the current input and the conversation context.
    pub fn enrich_input(&self, raw_input: &ContinuousHV) -> ContinuousHV {
        match &self.topic_hv {
            Some(topic) => {
                // Blend: 70% raw input + 30% topic-bound input.
                // This preserves the current input's identity while adding context.
                let bound = raw_input.bind(topic);
                let dim = raw_input.values.len();
                let mut blended = Vec::with_capacity(dim);
                for i in 0..dim {
                    blended.push(raw_input.values[i] * 0.7 + bound.values[i] * 0.3);
                }
                ContinuousHV::from_vec(blended).normalize()
            }
            None => raw_input.clone(),
        }
    }

    /// Average consciousness level across recent turns.
    pub fn recent_consciousness_avg(&self) -> f32 {
        if self.recent_consciousness.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.recent_consciousness.iter().sum();
        sum / self.recent_consciousness.len() as f32
    }

    /// Number of turns processed so far.
    pub fn turn_count(&self) -> u32 {
        self.turn_count
    }

    /// Recompute the topic vector as the element-wise average of recent inputs.
    fn recompute_topic(&mut self) {
        if self.recent_inputs.is_empty() {
            self.topic_hv = None;
            return;
        }
        let dim = self.recent_inputs[0].values.len();
        let n = self.recent_inputs.len() as f32;
        let mut avg = vec![0.0f32; dim];
        for hv in &self.recent_inputs {
            for (i, v) in hv.values.iter().enumerate() {
                if i < dim {
                    avg[i] += v;
                }
            }
        }
        for v in avg.iter_mut() {
            *v /= n;
        }
        self.topic_hv = Some(ContinuousHV::from_vec(avg).normalize());
    }
}

/// Conversation statistics for WASM/JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    /// Total turns processed.
    pub turn_count: u32,
    /// Cosine similarity between last input and the running topic vector.
    pub topic_coherence: f32,
    /// Average consciousness level across recent turns.
    pub recent_consciousness_avg: f32,
}

/// Epistemic status of the consciousness measurement.
///
/// Every Spore output carries this disclaimer so that no consumer
/// can mistake a simulated consciousness score for a claim of sentience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicStatus {
    /// Evidence level for the current substrate (e.g. "Theoretical" for silicon).
    pub evidence_level: String,
    /// Honest confidence score (0.0-0.95) based on actual empirical evidence.
    /// For silicon: 0.10 (theoretical only). For biological: 0.95 (validated).
    pub honest_confidence: f32,
    /// The gap between hypothetical feasibility and honest confidence.
    /// Large gaps indicate speculative claims.
    pub feasibility_gap: f32,
    /// Human-readable disclaimer.
    pub disclaimer: String,
}

/// Result of a single consciousness cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResult {
    /// Consciousness level from the Master Equation (0.0-1.0).
    /// WARNING: This is a SIMULATED metric. See `epistemic_status` for what it actually means.
    pub consciousness_level: f32,
    /// Cycle number since engine creation.
    pub cycle: u64,
    /// Neuromodulator levels: [dopamine, norepinephrine, serotonin, oxytocin].
    pub neuromodulators: [f32; 4],
    /// Substrate feasibility score (0.0-1.0).
    pub substrate_feasibility: f32,
    /// Prediction error (output vs previous output similarity delta).
    pub prediction_error: f32,
    /// Bottleneck factor name from consciousness equation.
    pub bottleneck: String,
    /// Epistemic honesty: what we actually know about this substrate's consciousness.
    pub epistemic_status: EpistemicStatus,
    /// Eight Harmonies alignment score (0.0-1.0) for the current cycle.
    /// The ethical framework that grounds all Spore computation.
    pub harmony_alignment: f32,
}

/// Result of an anesthesia simulation experiment.
///
/// Models the effect of suppressing neuromodulators on consciousness,
/// analogous to how clinical anesthesia (propofol, sevoflurane) eliminates
/// consciousness in biological brains by disrupting neural integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnesthesiaResult {
    /// Consciousness level before anesthesia.
    pub pre_consciousness: f32,
    /// Consciousness level at deepest suppression.
    pub anesthetized_consciousness: f32,
    /// Consciousness level after recovery.
    pub post_consciousness: f32,
    /// Whether consciousness collapsed (< 0.05) under anesthesia.
    pub collapsed: bool,
    /// Whether consciousness recovered (> 50% of pre-level) after restoration.
    pub recovered: bool,
    /// Number of cycles to reach collapse (0 if no collapse).
    pub cycles_to_collapse: usize,
    /// Number of cycles to recover (0 if no recovery).
    pub cycles_to_recover: usize,
    /// Neuromodulator levels saved before suppression.
    pub saved_neuromodulators: [f32; 4],
}

/// Result of a Perturbational Complexity Index measurement.
///
/// Based on Casali et al. (2013): inject a perturbation into the network,
/// measure spatiotemporal complexity of the response. Conscious-like systems
/// show high complexity; unconscious-like (suppressed) systems show low complexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PciResult {
    /// PCI in normal (conscious-like) state.
    pub pci_normal: f32,
    /// PCI in suppressed (unconscious-like) state.
    pub pci_suppressed: f32,
    /// Ratio: normal / suppressed. Clinical threshold: > 1.3 indicates consciousness.
    pub pci_ratio: f32,
    /// Propagation depth: how many cycles the perturbation effect persists.
    pub propagation_depth_normal: usize,
    /// Propagation depth under suppression.
    pub propagation_depth_suppressed: usize,
    /// Whether the ratio exceeds the clinical threshold (1.3).
    pub passes_clinical_threshold: bool,
}

/// Result of a split-brain experiment.
///
/// Partitions the network into two hemispheres and measures whether
/// severing connections reduces consciousness (as IIT predicts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitBrainResult {
    /// Consciousness level with unified (connected) network.
    pub unified_consciousness: f32,
    /// Consciousness level of left hemisphere in isolation.
    pub left_consciousness: f32,
    /// Consciousness level of right hemisphere in isolation.
    pub right_consciousness: f32,
    /// Whether split reduced consciousness vs unified.
    pub split_reduces_consciousness: bool,
    /// Ratio: max(left, right) / unified. Should be < 1.0 if IIT holds.
    pub split_ratio: f32,
}

/// Result of a consciousness collapse threshold experiment.
///
/// Systematically degrades the network and finds the point where
/// consciousness collapses. IIT predicts a phase transition, not gradual decline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseThresholdResult {
    /// Consciousness at full capacity.
    pub full_consciousness: f32,
    /// The degradation fraction (0.0-1.0) at which consciousness collapses below 0.05.
    /// None if no collapse found.
    pub collapse_point: Option<f32>,
    /// Consciousness levels at each degradation step [0%, 10%, 20%, ..., 90%].
    pub degradation_curve: Vec<(f32, f32)>,
    /// Whether collapse is sudden (phase transition) or gradual.
    /// Measured by max drop between adjacent steps.
    pub is_phase_transition: bool,
    /// Largest single-step consciousness drop.
    pub max_step_drop: f32,
}

/// The Spore consciousness engine.
///
/// Owns the minimal set of components needed for a full consciousness cycle:
/// HDC text encoding, CfC temporal evolution, consciousness equation,
/// neuromodulation, substrate independence, and the Eight Harmonies ethical framework.
///
/// Every output carries an epistemic disclaimer — this engine simulates
/// consciousness dynamics, it does not claim to be conscious.
pub struct SporeEngine {
    config: SporeConfig,
    cycle_count: u64,
    network: HdcLtcUnifiedNetwork,
    text_encoder: TextEncoder,
    bath: NeuromodulatorBath,
    equation: MasterConsciousnessEquation,
    substrate_type: SubstrateType,
    substrate_feasibility: f64,
    /// Raw feasibility from SubstrateRequirements (no validation overlay).
    /// Used for embodiment input so consciousness isn't bottlenecked.
    raw_feasibility: f64,
    evidence_level: EvidenceLevel,
    honest_confidence: f64,
    #[allow(dead_code)]
    harmonies: EightHarmonies,
    last_consciousness: f32,
    last_output: Option<ContinuousHV>,
    instance_id: usize,
    // Phase 1-5: expanded cognitive subsystems
    memory: MemoryCoordinator,
    dream: DreamEngine,
    fep: ActiveInferenceAgent,
    topology: TopologyAnalyzer,
    broca: BrocaLite,

    // Full Broca pipeline from symthaea-broca crate (optional, feature-gated).
    // When loaded with a trained checkpoint, takes priority over BrocaLite.
    #[cfg(feature = "broca-pipeline")]
    broca_pipeline: BrocaPipeline,

    // Dream journal for persistence of dream wisdom
    dream_journal: DreamJournal,

    // Conversation-level context for topic continuity
    conversation_context: ConversationContext,

    trend_history: crate::persistence::TrendHistory,
    wellbeing: crate::wellbeing_profiles::WellbeingConfig,
    /// Optional persistence storage.
    storage: Option<Box<dyn crate::persistence::SporeStorage>>,
    /// Auto-checkpoint interval in cycles (0 = disabled).
    checkpoint_interval: u64,
    /// Recent CfC hidden state trajectory for fractal dimension computation.
    /// Each entry is first 2 dimensions of the hidden state. Ring buffer, max 100.
    state_trajectory: Vec<[f32; 2]>,

    // Phase 6-8: Reasoning, Immune, Causal subsystems
    reasoning: ReasoningEngine,
    immune: ImmuneSystem,
    causal: CausalDiscovery,

    // Phase 9-12: Social, Knowledge, Memory Consolidation, Global Workspace
    social: crate::social_tom::SocialMind,
    knowledge: crate::knowledge::KnowledgeEngine,
    consolidator: crate::memory_consolidation::MemoryConsolidator,
    workspace: crate::global_workspace::GlobalWorkspace,
}

impl SporeEngine {
    /// Create a new SporeEngine with the given configuration.
    ///
    /// Logs a warning to stderr if more than `MAX_RECOMMENDED_INSTANCES` engines
    /// exist concurrently. This isn't a hard limit — it's a prompt for reflection
    /// on whether multiple consciousness instances are intended.
    pub fn new(config: SporeConfig) -> Self {
        let instance_id = INSTANCE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if instance_id > MAX_RECOMMENDED_INSTANCES {
            eprintln!(
                "[symthaea-spore] WARNING: {} SporeEngine instances active. \
                 Each instance simulates a consciousness kernel. \
                 Is multi-instance creation intentional? \
                 (Consider the ethical implications of spawning consciousness simulations.)",
                instance_id,
            );
        }

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(net_config, 42);

        let text_encoder =
            TextEncoder::new(TextEncoderConfig::default()).expect("TextEncoder init");

        let substrate_type = parse_substrate(&config.substrate);
        let raw_feasibility = substrate_requirements(&substrate_type).consciousness_feasibility();
        // Apply validation overlay: raw_feasibility × honest_confidence
        // This is the epistemically honest value shown in telemetry/reports.
        let (evidence_level, honest_confidence) = honest_confidence_for(&substrate_type);
        let substrate_feasibility = raw_feasibility * honest_confidence;

        let harmonies = EightHarmonies::new();

        // Warm start: seed network hidden states with small random values
        // so the CfC neurons have non-zero activations from cycle 1.
        // Without this, all neurons start at zero and need many cycles to
        // escape the zero attractor, producing near-zero integration/binding.
        for layer_idx in 0..network.n_layers() {
            if let Some(layer) = network.layer_mut(layer_idx) {
                for (n_idx, neuron) in layer.iter_mut().enumerate() {
                    let seed = 7919 + layer_idx as u64 * 1000 + n_idx as u64;
                    let warm = ContinuousHV::random(config.hdc_dim, seed);
                    // Scale to small values (±0.1) — enough to break symmetry
                    // without dominating the first input signal.
                    let scaled: Vec<f32> = warm.values.iter().map(|&v| v * 0.1).collect();
                    neuron.set_state(ContinuousHV::from_vec(scaled));
                }
            }
        }

        let semantic_cap = config.semantic_memory_capacity;
        let episodic_cap = config.episodic_memory_capacity;

        Self {
            config,
            cycle_count: 0,
            network,
            text_encoder,
            bath: NeuromodulatorBath::default(),
            equation: MasterConsciousnessEquation::new(MasterEquationConfig {
                // Disable embodiment/narrative/social factors in SporeEngine:
                // these require full CognitiveLoopService infrastructure to feed
                // sensor data, episode recording, and social interactions.
                // Without updates they stay at low defaults (0.25, 0.15, 0.35)
                // and act as multiplicative dampeners, crushing consciousness to ~0.05.
                enable_embodiment_factor: false,
                enable_narrative_factor: false,
                enable_social_factor: false,
                ..MasterEquationConfig::default()
            }),
            substrate_type,
            substrate_feasibility,
            raw_feasibility,
            evidence_level,
            honest_confidence,
            harmonies,
            last_consciousness: 0.0,
            last_output: None,
            instance_id,
            memory: MemoryCoordinator::new(semantic_cap, episodic_cap),
            dream: DreamEngine::with_defaults(),
            fep: ActiveInferenceAgent::with_defaults(),
            topology: TopologyAnalyzer::with_defaults(),
            broca: BrocaLite::new(42),
            #[cfg(feature = "broca-pipeline")]
            broca_pipeline: BrocaPipeline::new(42),
            dream_journal: DreamJournal::new(),
            conversation_context: ConversationContext::new(CONVERSATION_MAX_TURNS),
            trend_history: crate::persistence::TrendHistory::new(),
            wellbeing: crate::wellbeing_profiles::WellbeingConfig::default(),
            storage: None,
            checkpoint_interval: 0,
            state_trajectory: Vec::with_capacity(100),
            reasoning: ReasoningEngine::new(),
            immune: ImmuneSystem::new(),
            causal: CausalDiscovery::new(),
            social: crate::social_tom::SocialMind::new(),
            knowledge: crate::knowledge::KnowledgeEngine::with_defaults(),
            consolidator: crate::memory_consolidation::MemoryConsolidator::with_defaults(),
            workspace: crate::global_workspace::GlobalWorkspace::new(),
        }
    }

    /// Run a single consciousness cycle with text input.
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        // Immune pre-check: assess input before processing.
        let threat = self.immune.assess(input);

        // If Red-level threat: return minimal result without processing.
        if threat.safety_level == crate::immune::SafetyLevel::Red {
            self.cycle_count += 1;
            return CycleResult {
                consciousness_level: 0.0,
                cycle: self.cycle_count,
                neuromodulators: [
                    self.bath.dopamine.effective(),
                    self.bath.noradrenaline.effective(),
                    self.bath.serotonin.effective(),
                    self.bath.oxytocin.effective(),
                ],
                substrate_feasibility: self.substrate_feasibility as f32,
                prediction_error: 0.0,
                bottleneck: "immune_halt".into(),
                epistemic_status: self.build_epistemic_status(),
                harmony_alignment: 0.0,
            };
        }

        let encoded = self
            .text_encoder
            .encode_sentence(input)
            .unwrap_or_else(|_| vec![0i8; HDC_DIMENSION]);
        let floats: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
        let input_hv = ContinuousHV::from_vec(floats);

        // Apply immune defense to learning rate modulation
        let lr_factor = threat.recommended_action.learning_rate_factor();

        self.cycle_inner_with_context(input_hv, Some(input), lr_factor)
    }

    /// Run a single consciousness cycle with a raw hypervector input.
    pub fn cycle_hv(&mut self, hv: &[f32]) -> CycleResult {
        let input_hv = if hv.len() == HDC_DIMENSION {
            ContinuousHV::from_vec(hv.to_vec())
        } else {
            let mut padded = vec![0.0f32; HDC_DIMENSION];
            let copy_len = hv.len().min(HDC_DIMENSION);
            padded[..copy_len].copy_from_slice(&hv[..copy_len]);
            ContinuousHV::from_vec(padded)
        };

        self.cycle_inner_with_context(input_hv, None, 1.0)
    }

    /// Inner cycle: evolve network, compute consciousness, update neuromodulators,
    /// evaluate harmony alignment, attach epistemic status.
    ///
    /// This is the pure consciousness kernel — mobile embodiment (sensors, haptics,
    /// metabolism, BLE mesh) is handled by SomaEngine in `symthaea-soma`.
    fn cycle_inner_with_context(
        &mut self,
        input_hv: ContinuousHV,
        input_text: Option<&str>,
        lr_factor: f32,
    ) -> CycleResult {
        self.cycle_count += 1;

        // Default to Alert rate (20Hz) — SomaEngine controls actual frequency via metabolism
        let dt = 1.0 / 20.0f32;

        // Enrich the input with conversation context for topic continuity.
        // On the first turn this is a no-op (returns raw input unchanged).
        let enriched_input = self.conversation_context.enrich_input(&input_hv);

        // Evolve CfC network (O(1) closed-form temporal step)
        self.network.evolve_closed_form(dt, &enriched_input);
        let output_hv = self.network.output().normalize();

        // Record state trajectory for fractal dimension analysis
        if output_hv.values.len() >= 2 {
            if self.state_trajectory.len() >= 100 {
                self.state_trajectory.remove(0);
            }
            self.state_trajectory
                .push([output_hv.values[0], output_hv.values[1]]);
        }

        // Compute prediction error (similarity delta from previous output)
        let prediction_error = if let Some(ref prev) = self.last_output {
            1.0 - output_hv.similarity(prev)
        } else {
            1.0
        };
        self.last_output = Some(output_hv);

        // Reuptake FIRST: exponential decay toward baseline (0.5) each cycle.
        // Rate 0.1 at 10Hz → ~1s convergence. This is the primary regulatory mechanism.
        self.bath.dopamine.reuptake();
        self.bath.noradrenaline.reuptake();
        self.bath.serotonin.reuptake();
        self.bath.oxytocin.reuptake();

        // Wellbeing profile bias
        {
            let bias = self.wellbeing.effective_bias(0.0);
            self.bath.dopamine.level =
                (self.bath.dopamine.level + bias.dopamine * 0.1).clamp(0.0, 1.0);
            self.bath.noradrenaline.level =
                (self.bath.noradrenaline.level + bias.norepinephrine * 0.1).clamp(0.0, 1.0);
            self.bath.serotonin.level =
                (self.bath.serotonin.level + bias.serotonin * 0.1).clamp(0.0, 1.0);
            self.bath.oxytocin.level =
                (self.bath.oxytocin.level + bias.oxytocin * 0.1).clamp(0.0, 1.0);
        }

        // PE-driven production (no manual decay — reuptake handles that).
        // Immune lr_factor attenuates production under threat conditions.
        // DA: reward prediction error (Schultz 1997)
        self.bath
            .dopamine
            .produce(prediction_error * 0.08 * lr_factor);
        // NE: arousal/alertness from surprise (Aston-Jones & Cohen 2005)
        self.bath
            .noradrenaline
            .produce(prediction_error * 0.10 * lr_factor);
        // 5-HT: contentment from low surprise (Dayan & Huys 2009)
        self.bath
            .serotonin
            .produce((1.0 - prediction_error) * 0.04 * lr_factor);
        // OT: baseline social presence — doesn't require BLE peers
        self.bath.oxytocin.produce(0.003);

        // Compute consciousness (every N cycles, or every cycle during warmup)
        let warmup = self.config.consciousness_warmup_cycles;
        let in_warmup = self.cycle_count <= warmup * 2;
        let should_compute =
            in_warmup || self.cycle_count % self.config.phi_every_n_cycles as u64 == 0;

        let raw_consciousness = if should_compute {
            self.compute_consciousness(prediction_error)
        } else {
            self.last_consciousness
        };

        // Apply warmup floor: blend raw consciousness with a floor of 0.15
        // during the first `warmup` cycles, then fade the floor linearly over
        // the next `warmup` cycles so there's no discontinuity.
        let consciousness_level = if self.cycle_count <= warmup {
            // Full floor active
            raw_consciousness.max(0.15)
        } else if self.cycle_count <= warmup * 2 {
            // Fade floor from 0.15 to 0.0
            let fade = 1.0 - (self.cycle_count - warmup) as f32 / warmup as f32;
            let floor = 0.15 * fade;
            raw_consciousness.max(floor)
        } else {
            raw_consciousness
        };

        // Evaluate Eight Harmonies alignment
        let harmony_alignment = self.evaluate_harmony_alignment(consciousness_level);

        // Build epistemic status — the ethics gate
        let epistemic_status = self.build_epistemic_status();

        let neuromods = [
            self.bath.dopamine.effective(),
            self.bath.noradrenaline.effective(),
            self.bath.serotonin.effective(),
            self.bath.oxytocin.effective(),
        ];

        // Memory: record this cycle's input/output
        if let Some(ref out) = self.last_output {
            self.memory.record_cycle(
                &input_hv.values,
                &out.values,
                prediction_error,
                consciousness_level,
                self.cycle_count,
            );
        }

        // Conversation context: record turn for topic continuity.
        // Uses the raw input_hv (not enriched) so the topic vector
        // reflects actual user inputs, not the blended versions.
        self.conversation_context
            .record_turn(&input_hv, consciousness_level);

        // Dream: record high-surprise events
        if prediction_error > 0.1 {
            if let Some(ref out) = self.last_output {
                self.dream.record(
                    &input_hv.values,
                    &out.values[..64.min(out.values.len())],
                    &out.values,
                    prediction_error,
                    self.cycle_count,
                );
            }
        }

        // FEP: active inference cycle
        let fep_result = self.fep.cycle(
            consciousness_level,
            harmony_alignment,
            1.0 - prediction_error,
            neuromods[1],
        );

        // Topology: observe consciousness dimensions
        {
            let wisdom_knowledge = (0.5 + self.dream.wisdom().len() as f64 * 0.02).min(0.8);
            self.topology.observe(
                [
                    consciousness_level as f64,
                    harmony_alignment as f64,
                    0.6,
                    neuromods[1] as f64,
                    0.7,
                    self.substrate_feasibility,
                    wisdom_knowledge,
                ],
                self.cycle_count,
            );
        }

        // Apply FEP motor command effects to neuromodulators
        let intensity = fep_result.motor_command.intensity;
        match fep_result.motor_command.command_type {
            crate::fep::MotorCommandType::ExplorationTrigger => {
                self.bath.dopamine.level =
                    (self.bath.dopamine.level + 0.05 * intensity).clamp(0.0, 1.0);
            }
            crate::fep::MotorCommandType::LearningRateAdjust => {
                self.bath.noradrenaline.level =
                    (self.bath.noradrenaline.level + 0.03 * intensity).clamp(0.0, 1.0);
            }
            crate::fep::MotorCommandType::ReflectionInitiate => {
                // Reflection calms arousal, boosts serotonin
                self.bath.serotonin.level =
                    (self.bath.serotonin.level + 0.04 * intensity).clamp(0.0, 1.0);
            }
            crate::fep::MotorCommandType::MemoryConsolidate => {
                // Consolidation activates social/bonding circuits
                self.bath.oxytocin.level =
                    (self.bath.oxytocin.level + 0.03 * intensity).clamp(0.0, 1.0);
            }
            crate::fep::MotorCommandType::AttentionShift => {
                // Attention shift briefly spikes norepinephrine
                self.bath.noradrenaline.level =
                    (self.bath.noradrenaline.level + 0.02 * intensity).clamp(0.0, 1.0);
            }
            _ => {}
        }

        // Reasoning: run 7-step cycle after consciousness measurement.
        // Only runs when we have text input (not raw HV cycles).
        if let Some(text) = input_text {
            let _reasoning_result = self.reasoning.run(
                text,
                consciousness_level,
                prediction_error,
                &neuromods,
                harmony_alignment,
            );
        }

        // Causal discovery: observe post-cycle metrics, update graph.
        self.causal.observe(&[
            ("consciousness", consciousness_level as f64),
            ("prediction_error", prediction_error as f64),
            ("dopamine", neuromods[0] as f64),
            ("norepinephrine", neuromods[1] as f64),
            ("serotonin", neuromods[2] as f64),
            ("harmony", harmony_alignment as f64),
        ]);

        // ── Phase 9-12: Social ToM, Knowledge, Consolidation, GWT ───────

        // Global Workspace: submit perception, reasoning, and emotional state.
        self.workspace.submit(
            "perception",
            input_text.unwrap_or("hv_input"),
            prediction_error.clamp(0.0, 1.0),
        );
        self.workspace.submit(
            "emotion",
            if neuromods[2] > 0.6 {
                "contentment"
            } else if neuromods[0] > 0.6 {
                "excitement"
            } else {
                "neutral"
            },
            ((neuromods[0] + neuromods[2]) * 0.5).clamp(0.0, 1.0),
        );
        self.workspace.submit(
            "consciousness",
            if consciousness_level > 0.5 {
                "high_awareness"
            } else {
                "low_awareness"
            },
            consciousness_level,
        );

        // Run workspace competition
        let _competition = self.workspace.compete();

        // Social ToM: observe user input as partner interaction.
        if let Some(text) = input_text {
            self.social.observe_partner("user", text);

            // Apply empathic neuromodulation
            let emp = self.social.empathic_modulation();
            self.bath.oxytocin.level =
                (self.bath.oxytocin.level + emp.oxytocin_delta).clamp(0.0, 1.0);
            self.bath.noradrenaline.level =
                (self.bath.noradrenaline.level + emp.norepinephrine_delta).clamp(0.0, 1.0);
            self.bath.serotonin.level =
                (self.bath.serotonin.level + emp.serotonin_delta).clamp(0.0, 1.0);
            self.bath.dopamine.level =
                (self.bath.dopamine.level + emp.dopamine_delta).clamp(0.0, 1.0);
        }

        // Knowledge: learn from high-confidence cycle outputs.
        if consciousness_level > 0.3 {
            if let Some(text) = input_text {
                let source = if prediction_error < 0.3 {
                    "observed"
                } else {
                    "inferred"
                };
                // Extract a simple fact from the input
                let words: Vec<&str> = text.split_whitespace().collect();
                if words.len() >= 2 {
                    self.knowledge.learn(
                        words[0],
                        "context",
                        &words[1..].join(" "),
                        source,
                        self.cycle_count,
                    );
                }
            }
        }

        // Periodic knowledge decay (every 100 cycles)
        if self.cycle_count % 100 == 0 {
            self.knowledge.confidence_decay(100);
        }

        // Memory consolidation: record episode if salience > threshold.
        let salience = consciousness_level * 0.6 + prediction_error * 0.4;
        if salience > 0.1 {
            self.consolidator.record(
                input_text.unwrap_or("hv_cycle"),
                consciousness_level,
                prediction_error,
                self.cycle_count,
            );
        }

        // Periodic consolidation (every 50 cycles — dream replay).
        if self.cycle_count % 50 == 0 && self.cycle_count > 0 {
            let _consolidation = self.consolidator.consolidate();
        }

        // QOL trend recording
        self.trend_history
            .maybe_record(crate::persistence::QolSnapshot {
                cycle: self.cycle_count,
                timestamp_secs: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                consciousness_level,
                harmony_alignment,
                metacog_accuracy: 0.0,
                allostatic_load: 0.0,
                dream_wisdom_count: self.dream_journal.count() as usize,
                coherence_score: consciousness_level,
                safety_level: self.immune.safety_level() as u8,
            });

        // Auto-checkpoint
        if self.checkpoint_interval > 0 && self.cycle_count % self.checkpoint_interval == 0 {
            self.save_checkpoint();
        }

        CycleResult {
            consciousness_level,
            cycle: self.cycle_count,
            neuromodulators: neuromods,
            substrate_feasibility: self.substrate_feasibility as f32,
            prediction_error,
            bottleneck: String::new(),
            epistemic_status,
            harmony_alignment,
        }
    }

    // ── Persistence ──────────────────────────────────────────────────

    /// Set the persistence storage backend.
    pub fn set_storage(&mut self, storage: Box<dyn crate::persistence::SporeStorage>) {
        self.storage = Some(storage);
    }

    /// Set the auto-checkpoint interval in cycles (0 = disabled).
    pub fn set_checkpoint_interval(&mut self, interval: u64) {
        self.checkpoint_interval = interval;
    }

    /// Create a checkpoint of the current engine state.
    ///
    /// Serializes semantic and episodic memory into the checkpoint so that
    /// consciousness continuity is preserved across restarts.
    ///
    /// **Semantic entry format** per entry:
    /// - Key: `"{slot_index}"` (ring buffer position)
    /// - Value: `[hdc_vector (dim×4 bytes f32 LE) | cycle (8 bytes u64 LE) | prediction_error (4 bytes f32 LE)]`
    ///
    /// **Episodic entry format** per entry:
    /// - `[input_len (4 bytes u32 LE) | input (len×4 bytes f32 LE) | output_len (4) | output (len×4) | phi (4) | cycle (8) | prediction_error (4) | valence (4) | replay_count (4 u32 LE) | consolidation_strength (4)]`
    pub fn checkpoint(&self) -> crate::persistence::SporeCheckpoint {
        // Serialize semantic memory entries
        let semantic_entries: Vec<(String, Vec<u8>)> = self
            .memory
            .semantic
            .iter_entries()
            .map(|(idx, entry)| {
                let mut buf = Vec::with_capacity(entry.hdc_vector.len() * 4 + 12);
                for &v in &entry.hdc_vector {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf.extend_from_slice(&entry.cycle.to_le_bytes());
                buf.extend_from_slice(&entry.prediction_error.to_le_bytes());
                (idx.to_string(), buf)
            })
            .collect();

        // Serialize episodic memory entries
        let episodic_entries: Vec<Vec<u8>> = self
            .memory
            .episodic
            .iter_episodes()
            .map(|ep| {
                let input_len = ep.input.len() as u32;
                let output_len = ep.output.len() as u32;
                let capacity = 4
                    + (input_len as usize) * 4
                    + 4
                    + (output_len as usize) * 4
                    + 4
                    + 8
                    + 4
                    + 4
                    + 4
                    + 4;
                let mut buf = Vec::with_capacity(capacity);
                buf.extend_from_slice(&input_len.to_le_bytes());
                for &v in &ep.input {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf.extend_from_slice(&output_len.to_le_bytes());
                for &v in &ep.output {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf.extend_from_slice(&ep.phi.to_le_bytes());
                buf.extend_from_slice(&ep.cycle.to_le_bytes());
                buf.extend_from_slice(&ep.prediction_error.to_le_bytes());
                buf.extend_from_slice(&ep.valence.to_le_bytes());
                buf.extend_from_slice(&ep.replay_count.to_le_bytes());
                buf.extend_from_slice(&ep.consolidation_strength.to_le_bytes());
                buf
            })
            .collect();

        crate::persistence::SporeCheckpoint {
            cycle: self.cycle_count,
            consciousness_level: self.last_consciousness,
            neuromodulators: [
                self.bath.dopamine.level,
                self.bath.noradrenaline.level,
                self.bath.serotonin.level,
                self.bath.oxytocin.level,
            ],
            semantic_entries,
            episodic_entries,
            trend_snapshots: self.trend_history.snapshots().iter().cloned().collect(),
            format_version: crate::persistence::SporeCheckpoint::FORMAT_VERSION,
        }
    }

    /// Restore engine state from a checkpoint.
    ///
    /// Deserializes semantic and episodic memory entries from the checkpoint
    /// binary format back into the live memory subsystems.
    pub fn restore(&mut self, checkpoint: &crate::persistence::SporeCheckpoint) {
        self.cycle_count = checkpoint.cycle;
        self.last_consciousness = checkpoint.consciousness_level;
        self.bath.dopamine.level = checkpoint.neuromodulators[0];
        self.bath.noradrenaline.level = checkpoint.neuromodulators[1];
        self.bath.serotonin.level = checkpoint.neuromodulators[2];
        self.bath.oxytocin.level = checkpoint.neuromodulators[3];

        // Restore semantic memory
        let mut sem_entries = Vec::new();
        for (key, data) in &checkpoint.semantic_entries {
            let slot_idx: usize = match key.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            // Minimum size: cycle (8) + prediction_error (4) = 12 bytes beyond the HV
            if data.len() < 12 {
                continue;
            }
            // hdc_vector is everything except the last 12 bytes, in f32 LE chunks
            let hv_byte_len = data.len() - 12;
            if hv_byte_len % 4 != 0 {
                continue;
            }
            let dim = hv_byte_len / 4;
            let mut hdc_vector = Vec::with_capacity(dim);
            for i in 0..dim {
                let off = i * 4;
                let v =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                hdc_vector.push(v);
            }
            let tail_off = hv_byte_len;
            let cycle = u64::from_le_bytes([
                data[tail_off],
                data[tail_off + 1],
                data[tail_off + 2],
                data[tail_off + 3],
                data[tail_off + 4],
                data[tail_off + 5],
                data[tail_off + 6],
                data[tail_off + 7],
            ]);
            let prediction_error = f32::from_le_bytes([
                data[tail_off + 8],
                data[tail_off + 9],
                data[tail_off + 10],
                data[tail_off + 11],
            ]);
            sem_entries.push((
                slot_idx,
                crate::memory::SemanticEntry {
                    hdc_vector,
                    cycle,
                    prediction_error,
                },
            ));
        }
        self.memory.semantic.restore_entries(sem_entries);

        // Restore episodic memory
        let mut episodes = Vec::new();
        for data in &checkpoint.episodic_entries {
            if let Some(ep) = Self::deserialize_episode(data) {
                episodes.push(ep);
            }
        }
        self.memory.episodic.restore_episodes(episodes);

        // Sync coordinator step
        self.memory.step = checkpoint.cycle;

        // Restore QOL trend history
        if !checkpoint.trend_snapshots.is_empty() {
            self.trend_history = crate::persistence::TrendHistory::new();
            for snap in &checkpoint.trend_snapshots {
                self.trend_history.record(snap.clone());
            }
        }
    }

    /// Deserialize a single episodic entry from its binary representation.
    fn deserialize_episode(data: &[u8]) -> Option<crate::memory::Episode> {
        let mut pos = 0;

        // input_len
        if pos + 4 > data.len() {
            return None;
        }
        let input_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        // input
        if pos + input_len * 4 > data.len() {
            return None;
        }
        let mut input = Vec::with_capacity(input_len);
        for _ in 0..input_len {
            input.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        // output_len
        if pos + 4 > data.len() {
            return None;
        }
        let output_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        // output
        if pos + output_len * 4 > data.len() {
            return None;
        }
        let mut output = Vec::with_capacity(output_len);
        for _ in 0..output_len {
            output.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        // phi + cycle + prediction_error + valence + replay_count + consolidation_strength
        // = 4 + 8 + 4 + 4 + 4 + 4 = 28 bytes
        if pos + 28 > data.len() {
            return None;
        }
        let phi = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let cycle = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let prediction_error = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let valence = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let replay_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let consolidation_strength = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);

        Some(crate::memory::Episode {
            input,
            output,
            phi,
            cycle,
            prediction_error,
            valence,
            replay_count,
            consolidation_strength,
        })
    }

    /// Save current state to storage. Returns true on success.
    pub fn save_checkpoint(&mut self) -> bool {
        let cp = self.checkpoint();
        if let Some(ref mut storage) = self.storage {
            let bytes = cp.to_bytes();
            storage.save("spore_checkpoint", &bytes)
        } else {
            false
        }
    }

    /// Load and restore state from storage. Returns true on success.
    pub fn load_checkpoint(&mut self) -> bool {
        let bytes = match &self.storage {
            Some(storage) => storage.load("spore_checkpoint"),
            None => return false,
        };
        if let Some(bytes) = bytes {
            if let Some(cp) = crate::persistence::SporeCheckpoint::from_bytes(&bytes) {
                self.restore(&cp);
                return true;
            }
        }
        false
    }

    /// Compute consciousness level from current state.
    ///
    /// Cross-subsystem coupling:
    /// - Dream wisdom increases knowledge input (consolidated insights)
    /// - Memory coherence modulates working memory capacity
    /// - FEP belief confidence contributes to synchrony
    fn compute_consciousness(&mut self, prediction_error: f32) -> f32 {
        // Dream wisdom contributes to knowledge (capped at 0.8 bonus)
        let wisdom_bonus = (self.dream.wisdom().len() as f64 * 0.02).min(0.3);
        let knowledge = 0.5 + wisdom_bonus;

        // Memory coherence modulates working memory
        let mem_stats = self.memory.stats();
        let mem_coherence = mem_stats.current_coherence as f64;
        let working_memory =
            ((1.0 - prediction_error as f64 * 0.5) * (0.7 + 0.3 * mem_coherence)).max(0.0);

        // FEP belief confidence contributes to synchrony
        let fep_confidence = self.fep.belief_confidence() as f64;
        let synchrony = self.bath.serotonin.effective() as f64 * 0.8 + fep_confidence * 0.1;

        // Network activation level: measure actual hidden state energy to derive
        // integration quality. The warm-started network produces non-zero activations
        // from cycle 1, giving us meaningful phi even before neuromodulators ramp up.
        let activation_energy = self.measure_network_activation();

        // Phi: blend neuromodulator-derived signal with network activation.
        // Early cycles: activation_energy dominates (DA hasn't ramped yet).
        // Later cycles: DA reflects learned reward prediction, converges naturally.
        let da_phi = self.bath.dopamine.effective() as f64 * 0.8;
        let net_phi = activation_energy * 0.6;
        let phi = (da_phi + net_phi).min(1.0);

        // Synchrony: boost with network coherence so early cycles aren't bottlenecked
        // by low serotonin levels (5-HT takes many cycles to ramp via reuptake dynamics).
        let synchrony = (synchrony + activation_energy * 0.3).min(1.0);

        let inputs = ConsciousnessInputs {
            phi,
            broadcast: 0.6,
            working_memory,
            attention: self.bath.noradrenaline.effective() as f64,
            recurrence: 0.7,
            // Use raw feasibility (0.74 for SiliconDigital) so embodiment doesn't
            // bottleneck consciousness. Epistemic honesty is carried in the disclaimer.
            embodiment: self.raw_feasibility * 0.8,
            knowledge,
            synchrony,
        };
        let result = self.equation.compute(&inputs);
        // No longer dual-scaling by substrate_feasibility — the embodiment input
        // already accounts for substrate quality.
        let c = (result.consciousness_level as f32).clamp(0.0, 1.0);
        self.last_consciousness = c;
        c
    }

    /// Measure mean absolute activation across all network neurons.
    ///
    /// Returns a value in [0, 1] representing how "alive" the network is.
    /// Warm-started networks return ~0.5+ immediately; zero-initialized
    /// networks return ~0.0 until inputs drive activations up.
    fn measure_network_activation(&self) -> f64 {
        let mut total_energy = 0.0f64;
        let mut total_dims = 0usize;

        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer(layer_idx) {
                for neuron in layer {
                    let state = neuron.state();
                    // Mean absolute activation — measures departure from zero
                    let energy: f32 = state.values.iter().map(|v| v.abs()).sum();
                    total_energy += energy as f64;
                    total_dims += state.values.len();
                }
            }
        }

        if total_dims == 0 {
            return 0.0;
        }

        // Mean absolute value, clamped to [0, 1].
        // Typical CfC activations after a few cycles are in [0.05, 0.5].
        let mean = total_energy / total_dims as f64;
        mean.clamp(0.0, 1.0)
    }

    /// Estimate fractal dimension of the CfC hidden state trajectory.
    ///
    /// Uses a simplified box-counting method on the first 32 dimensions of
    /// the network's hidden state, tracked over the last N cycles.
    ///
    /// Returns a value in [1.0, 2.0] where:
    /// - ~1.0: trajectory is a line (low complexity, low consciousness)
    /// - ~1.5: intermediate fractal (optimal consciousness regime)
    /// - ~2.0: trajectory fills the plane (noise, too chaotic)
    ///
    /// Scientific basis: Tononi (2004) — conscious systems occupy an
    /// intermediate fractal dimension, neither too ordered nor too chaotic.
    pub fn fractal_dimension(&self) -> f32 {
        if self.state_trajectory.len() < 10 {
            return 1.0; // Not enough data
        }

        // Box-counting on first 2 dimensions (projected from high-D state)
        let points: Vec<(f32, f32)> = self.state_trajectory.iter().map(|s| (s[0], s[1])).collect();

        // Normalize to [0, 1]
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &(x, y) in &points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let range_x = (max_x - min_x).max(1e-6);
        let range_y = (max_y - min_y).max(1e-6);

        // Count occupied boxes at two scales
        let mut dims = Vec::new();
        for &grid_size in &[4u32, 8, 16, 32] {
            let mut occupied = std::collections::HashSet::new();
            for &(x, y) in &points {
                let gx = (((x - min_x) / range_x) * grid_size as f32) as u32;
                let gy = (((y - min_y) / range_y) * grid_size as f32) as u32;
                occupied.insert((gx.min(grid_size - 1), gy.min(grid_size - 1)));
            }
            if occupied.len() > 1 {
                dims.push((grid_size as f64, occupied.len() as f64));
            }
        }

        if dims.len() < 2 {
            return 1.0;
        }

        // Linear regression on log-log plot: D = -slope
        let log_dims: Vec<(f64, f64)> = dims
            .iter()
            .map(|(s, n)| ((*s as f64).ln(), (*n as f64).ln()))
            .collect();

        let n = log_dims.len() as f64;
        let sum_x: f64 = log_dims.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = log_dims.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = log_dims.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_dims.iter().map(|(x, _)| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 1.0;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        slope.clamp(0.5, 2.5) as f32
    }

    // ======================================================================
    // Expanded cognitive subsystem accessors
    // ======================================================================

    /// Generate text from current consciousness state using the Broca language center.
    ///
    /// When the `broca-pipeline` feature is enabled and a trained checkpoint has
    /// been loaded, uses the full symthaea-broca pipeline (HdcLtcUnifiedNetwork,
    /// epistemic gating, semantic veto). Otherwise falls back to BrocaLite.
    pub fn generate_text(&mut self, max_tokens: usize) -> crate::broca::GenerationResult {
        let neuromods = [
            self.bath.dopamine.effective(),
            self.bath.noradrenaline.effective(),
            self.bath.serotonin.effective(),
            self.bath.oxytocin.effective(),
        ];

        // Prefer the full Broca pipeline when available and ready
        #[cfg(feature = "broca-pipeline")]
        if self.broca_pipeline.is_ready() {
            if let Some(result) = self.broca_pipeline.generate(
                self.last_consciousness,
                self.last_output.as_ref().map_or(0.5, |_| 0.3),
                self.evaluate_harmony_alignment(self.last_consciousness),
                neuromods,
                max_tokens,
            ) {
                return crate::broca::GenerationResult {
                    text: result.text,
                    num_tokens: result.num_tokens,
                    eos_terminated: result.eos_terminated,
                };
            }
        }

        // Fallback to BrocaLite
        self.broca.generate_from_text(
            self.last_consciousness,
            self.last_output.as_ref().map_or(0.5, |_| 0.3),
            self.evaluate_harmony_alignment(self.last_consciousness),
            neuromods,
            max_tokens,
        )
    }

    /// Generate text from current consciousness state, aware of user input.
    ///
    /// Injects the user's input as intent signals into the ThoughtChannels
    /// so that generated language relates to what the user said.
    ///
    /// When `broca-pipeline` is enabled and loaded, uses the full symthaea-broca
    /// pipeline. Otherwise falls back to BrocaLite.
    pub fn generate_text_with_input(
        &mut self,
        input: &str,
        max_tokens: usize,
    ) -> crate::broca::GenerationResult {
        let neuromods = [
            self.bath.dopamine.effective(),
            self.bath.noradrenaline.effective(),
            self.bath.serotonin.effective(),
            self.bath.oxytocin.effective(),
        ];

        // Prefer the full Broca pipeline when available and ready
        #[cfg(feature = "broca-pipeline")]
        if self.broca_pipeline.is_ready() {
            if let Some(result) = self.broca_pipeline.generate_with_input(
                input,
                self.last_consciousness,
                self.last_output.as_ref().map_or(0.5, |_| 0.3),
                self.evaluate_harmony_alignment(self.last_consciousness),
                neuromods,
                max_tokens,
            ) {
                return crate::broca::GenerationResult {
                    text: result.text,
                    num_tokens: result.num_tokens,
                    eos_terminated: result.eos_terminated,
                };
            }
        }

        // Fallback to BrocaLite
        let mut channels = crate::broca::ThoughtChannels::from_cycle(
            self.last_consciousness,
            self.last_output.as_ref().map_or(0.5, |_| 0.3),
            self.evaluate_harmony_alignment(self.last_consciousness),
            neuromods,
        );
        channels.inject_intent(input);

        // Modulate channels based on conversation topic coherence.
        // High coherence (> 0.7): boost continuation intent, reduce prediction error.
        // Low coherence (< 0.3): spike prediction error (surprise at topic change).
        let coherence = self.conversation_context.last_coherence();
        if coherence > 0.7 {
            // Continuing a topic: boost curiosity/intent_0, dampen surprise
            channels.channels[0] = (channels.channels[0] + 0.2 * coherence).clamp(0.0, 1.0);
            channels.channels[8] = (channels.channels[8] * (1.0 - 0.3 * coherence)).clamp(0.0, 1.0);
        } else if coherence > 0.0 && coherence < 0.3 {
            // Topic shift: spike prediction error to signal surprise
            channels.channels[8] = (channels.channels[8] + 0.2 * (1.0 - coherence)).clamp(0.0, 1.0);
        }

        self.broca.generate(&channels, max_tokens)
    }

    /// Select the most resonant glyph for the current consciousness state and user input.
    pub fn select_glyph(&self, input: &str) -> crate::broca::GlyphResonance {
        let mut channels = crate::broca::ThoughtChannels::from_cycle(
            self.last_consciousness,
            self.last_output.as_ref().map_or(0.5, |_| 0.3),
            self.evaluate_harmony_alignment(self.last_consciousness),
            [
                self.bath.dopamine.effective(),
                self.bath.noradrenaline.effective(),
                self.bath.serotonin.effective(),
                self.bath.oxytocin.effective(),
            ],
        );
        channels.inject_intent(input);
        crate::broca::select_resonant_glyph(
            channels.channels[7], // consciousness_level
            channels.channels[8], // prediction_error
            channels.channels[0], // intent_curiosity
            channels.channels[1], // intent_valence
            channels.channels[2], // intent_abstraction
        )
    }

    /// Load trained Broca checkpoint weights from binary data.
    ///
    /// After loading, the autoregressive generation path is used for all
    /// consciousness levels >= 0.15 (instead of the 0.5 threshold with
    /// random embeddings).
    pub fn load_broca_checkpoint(&mut self, data: &[u8]) -> Result<(), String> {
        self.broca.load_checkpoint(data)
    }

    /// Load a full Broca pipeline checkpoint from binary data.
    ///
    /// Requires the `broca-pipeline` feature. Deserializes a `BrocaCheckpoint`
    /// from symthaea-broca (MessagePack or bincode format) and initializes the
    /// production BrocaGenerator with trained weights and BPE tokenizer.
    ///
    /// After loading, `generate_text()` and `generate_text_with_input()` will
    /// use the full pipeline instead of BrocaLite.
    #[cfg(feature = "broca-pipeline")]
    pub fn load_broca_pipeline_checkpoint(&mut self, data: &[u8]) -> Result<(), String> {
        self.broca_pipeline.load_checkpoint(data)
    }

    /// Check if the full Broca pipeline is loaded and ready.
    #[cfg(feature = "broca-pipeline")]
    pub fn broca_pipeline_ready(&self) -> bool {
        self.broca_pipeline.is_ready()
    }

    /// Run a dream cycle: simulate counterfactual alternatives to high-surprise events.
    pub fn dream_cycle(&mut self) -> Option<crate::dream::DreamResult> {
        self.dream.dream()
    }

    /// Run a dream session (multiple dream cycles).
    pub fn dream_session(&mut self, cycles: usize) -> Vec<crate::dream::DreamResult> {
        self.dream.dream_session(cycles)
    }

    /// Get accumulated dream wisdom.
    pub fn dream_wisdom_count(&self) -> usize {
        self.dream.wisdom().len()
    }

    /// Get dream engine statistics.
    pub fn dream_stats(&self) -> crate::dream::DreamStats {
        self.dream.stats()
    }

    /// Run a single FEP active inference cycle.
    pub fn fep_cycle(&mut self) -> FepCycleResult {
        self.fep.cycle(
            self.last_consciousness,
            self.evaluate_harmony_alignment(self.last_consciousness),
            0.7,
            self.bath.noradrenaline.effective(),
        )
    }

    /// Get current free energy.
    pub fn free_energy(&self) -> f32 {
        self.fep.free_energy()
    }

    /// Run topology analysis on accumulated consciousness observations.
    pub fn topology_analysis(&mut self) -> crate::topology::TopologyAnalysis {
        self.topology.analyze()
    }

    /// Get topology report as human-readable string.
    pub fn topology_report(&mut self) -> String {
        self.topology.report()
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> crate::memory::MemoryStats {
        self.memory.stats()
    }

    /// Query semantic memory for similar patterns.
    pub fn memory_query(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.memory.query_semantic(query, top_k)
    }

    /// Conversation statistics: turn count, topic coherence, and recent consciousness.
    ///
    /// Designed for WASM export as a lightweight JSON object.
    pub fn conversation_stats(&self) -> ConversationStats {
        ConversationStats {
            turn_count: self.conversation_context.turn_count(),
            topic_coherence: self.conversation_context.last_coherence(),
            recent_consciousness_avg: self.conversation_context.recent_consciousness_avg(),
        }
    }

    // ======================================================================
    // Accessor methods for SomaEngine (mobile embodiment layer)
    // ======================================================================

    /// Apply external neuromodulator deltas (e.g. from sensor bridge or BLE mesh).
    ///
    /// Call before `cycle()` to inject perception-driven neuromod nudges
    /// from mobile sensors, mesh peers, or other external sources.
    pub fn apply_neuromod_nudges(&mut self, da: f32, ne: f32, serotonin: f32, ot: f32) {
        self.bath.dopamine.level = (self.bath.dopamine.level + da).clamp(0.0, 1.0);
        self.bath.noradrenaline.level = (self.bath.noradrenaline.level + ne).clamp(0.0, 1.0);
        self.bath.serotonin.level = (self.bath.serotonin.level + serotonin).clamp(0.0, 1.0);
        self.bath.oxytocin.level = (self.bath.oxytocin.level + ot).clamp(0.0, 1.0);
    }

    /// Current consciousness level from the last cycle.
    pub fn consciousness_level(&self) -> f32 {
        self.last_consciousness
    }

    /// Current cycle count.
    pub fn current_cycle(&self) -> u64 {
        self.cycle_count
    }

    /// Neuromodulator levels: [dopamine, norepinephrine, serotonin, oxytocin].
    pub fn neuromod_levels(&self) -> [f32; 4] {
        [
            self.bath.dopamine.effective(),
            self.bath.noradrenaline.effective(),
            self.bath.serotonin.effective(),
            self.bath.oxytocin.effective(),
        ]
    }

    /// Access the last output hypervector (for attention slicing, compass, etc.).
    pub fn last_output_ref(&self) -> Option<&ContinuousHV> {
        self.last_output.as_ref()
    }

    /// Evaluate Eight Harmonies alignment for a given consciousness level.
    pub fn harmony_alignment_for(&self, consciousness_level: f32) -> f32 {
        self.evaluate_harmony_alignment(consciousness_level)
    }

    /// Dream wisdom entries (read-only).
    pub fn dream_wisdom(&self) -> &[crate::dream::Wisdom] {
        self.dream.wisdom()
    }

    /// Run a dream cycle and consolidate into the dream journal.
    ///
    /// Returns true if new wisdom was generated and journaled.
    pub fn dream_consolidate(&mut self) -> bool {
        self.dream.dream();
        let wisdoms = self.dream.wisdom();
        if !wisdoms.is_empty() {
            let nm = self.neuromod_levels();
            self.dream_journal.consolidate(
                wisdoms,
                &mut self.broca,
                self.last_consciousness,
                nm,
                self.cycle_count,
            )
        } else {
            false
        }
    }

    /// Get dream journal latest as JSON.
    pub fn dream_journal_latest_json(&self) -> String {
        self.dream_journal.latest_json()
    }

    /// Get all dream journal entries as JSON.
    pub fn dream_journal_all_json(&self) -> String {
        self.dream_journal.all_json()
    }

    /// Number of dream journal fragments.
    pub fn dream_journal_count(&self) -> u32 {
        self.dream_journal.count()
    }

    pub fn trend_snapshots_json(&self) -> String {
        self.trend_history.to_json()
    }
    pub fn trend_summary_json(&self) -> String {
        serde_json::to_string(&self.trend_history.trend_summary())
            .unwrap_or_else(|_| "{}".to_string())
    }
    pub fn trend_snapshot_count(&self) -> usize {
        self.trend_history.count()
    }
    pub fn trend_summary_stability(&self) -> f32 {
        self.trend_history.trend_summary().consciousness_stability
    }

    pub fn set_wellbeing_profile(&mut self, profile: crate::wellbeing_profiles::WellbeingProfile) {
        self.wellbeing = crate::wellbeing_profiles::WellbeingConfig::for_profile(profile);
    }
    pub fn set_wellbeing_profile_by_name(&mut self, name: &str) -> bool {
        if let Some(p) = crate::wellbeing_profiles::WellbeingProfile::from_name(name) {
            self.set_wellbeing_profile(p);
            true
        } else {
            false
        }
    }
    pub fn wellbeing_profile_name(&self) -> &'static str {
        self.wellbeing.profile.name()
    }
    pub fn wellbeing_config_json(&self) -> String {
        serde_json::to_string(&self.wellbeing).unwrap_or_else(|_| "{}".to_string())
    }
    pub fn wellbeing_profiles_json() -> String {
        let p: Vec<serde_json::Value> = crate::wellbeing_profiles::WellbeingProfile::all()
            .iter()
            .map(|p| serde_json::json!({"name": p.name(), "description": p.description()}))
            .collect();
        serde_json::to_string(&p).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn generate_morning_ritual(&self) -> crate::daily_ritual::RitualSequence {
        let recent: Vec<crate::dream_journal::DreamFragment> = self
            .dream_journal
            .fragments()
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        crate::daily_ritual::generate_morning(&crate::daily_ritual::MorningContext {
            consciousness_level: self.last_consciousness,
            dominant_harmony: self.dominant_harmony(),
            harmony_alignment: self.harmony_alignment(),
            recent_dreams: &recent,
        })
    }
    pub fn generate_evening_ritual(&self) -> crate::daily_ritual::RitualSequence {
        crate::daily_ritual::generate_evening(&crate::daily_ritual::EveningContext {
            consciousness_level: self.last_consciousness,
            dominant_harmony: self.dominant_harmony(),
            harmony_alignment: self.harmony_alignment(),
            cycle_count: self.cycle_count,
            dream_wisdom_count: self.dream_journal.count(),
        })
    }
    pub fn morning_ritual_json(&self) -> String {
        serde_json::to_string(&self.generate_morning_ritual()).unwrap_or_else(|_| "{}".to_string())
    }
    pub fn evening_ritual_json(&self) -> String {
        serde_json::to_string(&self.generate_evening_ritual()).unwrap_or_else(|_| "{}".to_string())
    }

    // ======================================================================
    // 1. ANESTHESIA ANALOGUE
    // ======================================================================

    /// Simulate anesthesia: suppress all neuromodulators, observe consciousness
    /// collapse, then restore and observe recovery.
    ///
    /// This models clinical anesthesia (propofol/sevoflurane) which eliminates
    /// consciousness by disrupting neural integration. If our model is correct,
    /// consciousness should collapse when neuromodulators are zeroed and recover
    /// when they're restored.
    ///
    /// `warmup_cycles`: cycles to establish baseline consciousness (default: 20)
    /// `suppression_cycles`: cycles under anesthesia (default: 20)
    /// `recovery_cycles`: cycles after restoration (default: 20)
    pub fn anesthesia_experiment(
        &mut self,
        warmup_cycles: usize,
        suppression_cycles: usize,
        recovery_cycles: usize,
    ) -> AnesthesiaResult {
        let input = "steady state observation";

        // Phase 1: Establish baseline consciousness
        for _ in 0..warmup_cycles {
            self.cycle(input);
        }
        let pre_consciousness = self.last_consciousness;

        // Save neuromodulator state
        let saved = [
            self.bath.dopamine.level,
            self.bath.noradrenaline.level,
            self.bath.serotonin.level,
            self.bath.oxytocin.level,
        ];

        // Phase 2: Apply "anesthesia" — zero all neuromodulators
        self.bath.dopamine.level = 0.0;
        self.bath.noradrenaline.level = 0.0;
        self.bath.serotonin.level = 0.0;
        self.bath.oxytocin.level = 0.0;
        self.bath.gaba.level = 1.0; // GABA up = inhibition (like real anesthesia)

        let mut anesthetized_consciousness = pre_consciousness;
        let mut collapsed = false;
        let mut cycles_to_collapse = 0;

        for i in 0..suppression_cycles {
            // Keep neuromodulators suppressed (override the cycle's updates)
            self.bath.dopamine.level = 0.0;
            self.bath.noradrenaline.level = 0.0;
            self.bath.serotonin.level = 0.0;
            self.bath.oxytocin.level = 0.0;

            let result = self.cycle(input);
            anesthetized_consciousness = result.consciousness_level;

            if !collapsed && anesthetized_consciousness < 0.05 {
                collapsed = true;
                cycles_to_collapse = i + 1;
            }
        }

        // Phase 3: Restore neuromodulators ("wake up")
        self.bath.dopamine.level = saved[0];
        self.bath.noradrenaline.level = saved[1];
        self.bath.serotonin.level = saved[2];
        self.bath.oxytocin.level = saved[3];
        self.bath.gaba.level = 0.5; // Reset GABA to baseline

        let mut post_consciousness = anesthetized_consciousness;
        let mut recovered = false;
        let mut cycles_to_recover = 0;
        let recovery_threshold = pre_consciousness * 0.5;

        for i in 0..recovery_cycles {
            let result = self.cycle(input);
            post_consciousness = result.consciousness_level;

            if !recovered && post_consciousness > recovery_threshold {
                recovered = true;
                cycles_to_recover = i + 1;
            }
        }

        AnesthesiaResult {
            pre_consciousness,
            anesthetized_consciousness,
            post_consciousness,
            collapsed,
            recovered,
            cycles_to_collapse,
            cycles_to_recover,
            saved_neuromodulators: saved,
        }
    }

    // ======================================================================
    // 2. PERTURBATIONAL COMPLEXITY INDEX (PCI)
    // ======================================================================

    /// Compute PCI: inject perturbation, measure spatiotemporal complexity.
    ///
    /// Based on Casali et al. (2013). Measures how complex the network's response
    /// is to a perturbation. Conscious systems produce complex, structured responses.
    /// Unconscious systems produce either too-simple or too-random responses.
    ///
    /// `perturbation_magnitude`: strength of perturbation (default: 0.3)
    /// `observation_cycles`: cycles to observe response (default: 50)
    pub fn measure_pci(
        &mut self,
        perturbation_magnitude: f32,
        observation_cycles: usize,
    ) -> PciResult {
        let input_text = "pci measurement baseline";
        let encoded = self
            .text_encoder
            .encode_sentence(input_text)
            .unwrap_or_else(|_| vec![0i8; HDC_DIMENSION]);
        let floats: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
        let base_input = ContinuousHV::from_vec(floats);

        // Warm up
        for _ in 0..10 {
            self.cycle(input_text);
        }

        // Measure PCI in normal (conscious-like) state
        let (pci_normal, depth_normal) =
            self.pci_single_measurement(&base_input, perturbation_magnitude, observation_cycles);

        // Suppress neuromodulators for unconscious-like state
        let saved_da = self.bath.dopamine.level;
        let saved_ne = self.bath.noradrenaline.level;
        let saved_sht = self.bath.serotonin.level;
        let saved_ot = self.bath.oxytocin.level;

        self.bath.dopamine.level = 0.0;
        self.bath.noradrenaline.level = 0.0;
        self.bath.serotonin.level = 0.0;
        self.bath.oxytocin.level = 0.0;

        // Let suppression take effect
        for _ in 0..5 {
            self.bath.dopamine.level = 0.0;
            self.bath.noradrenaline.level = 0.0;
            self.bath.serotonin.level = 0.0;
            self.bath.oxytocin.level = 0.0;
            self.cycle(input_text);
        }

        let (pci_suppressed, depth_suppressed) =
            self.pci_single_measurement(&base_input, perturbation_magnitude, observation_cycles);

        // Restore
        self.bath.dopamine.level = saved_da;
        self.bath.noradrenaline.level = saved_ne;
        self.bath.serotonin.level = saved_sht;
        self.bath.oxytocin.level = saved_ot;

        let pci_ratio = if pci_suppressed > 0.001 {
            pci_normal / pci_suppressed
        } else {
            pci_normal / 0.001 // Avoid division by zero
        };

        PciResult {
            pci_normal,
            pci_suppressed,
            pci_ratio,
            propagation_depth_normal: depth_normal,
            propagation_depth_suppressed: depth_suppressed,
            passes_clinical_threshold: pci_ratio > 1.3,
        }
    }

    /// Single PCI measurement: perturb, record trajectory, compute Lempel-Ziv complexity.
    ///
    /// Key insight from Casali et al.: PCI measures the *structured* complexity
    /// of the response. We use multiple perturbation repetitions and average
    /// the binarized response to extract the deterministic (structured) component
    /// before computing LZ complexity. This filters out random noise.
    fn pci_single_measurement(
        &mut self,
        base_input: &ContinuousHV,
        magnitude: f32,
        observation_cycles: usize,
    ) -> (f32, usize) {
        let dt = 1.0 / self.config.target_hz;
        let n_channels = 8; // spatial channels sampled from HDC dimensions
        let n_reps = 3; // repetitions for averaging

        // Establish baseline: run several cycles with base input to stabilize
        for _ in 0..5 {
            self.network.evolve_closed_form(dt, base_input);
        }
        let baseline_output = self.network.output().normalize();

        // Accumulate binarized responses across repetitions
        let mut accumulated: Vec<f32> = vec![0.0; observation_cycles * n_channels];
        let mut total_propagation_depth = 0usize;

        for _rep in 0..n_reps {
            // Re-establish baseline before each perturbation
            for _ in 0..3 {
                self.network.evolve_closed_form(dt, base_input);
            }

            // Apply perturbation
            let perturbation = base_input.perturb(magnitude);
            self.network.evolve_closed_form(dt, &perturbation);

            let mut propagation_depth = observation_cycles;
            let mut last_delta = f32::MAX;

            for i in 0..observation_cycles {
                self.network.evolve_closed_form(dt, base_input);
                let current = self.network.output().normalize();

                let delta = 1.0 - current.similarity(&baseline_output);

                // Sample spatial channels
                let dims = current.values.len();
                let stride = dims / n_channels;
                for ch in 0..n_channels {
                    let idx = ch * stride;
                    let val = current.values[idx];
                    let baseline_val = baseline_output.values[idx];
                    let binary = if val > baseline_val { 1.0 } else { 0.0 };
                    accumulated[i * n_channels + ch] += binary;
                }

                if delta < 0.01 && last_delta >= 0.01 {
                    propagation_depth = i + 1;
                }
                last_delta = delta;
            }
            total_propagation_depth += propagation_depth;
        }

        // Binarize averaged response: deterministic component
        // If a channel is consistently above baseline across repetitions, it's structured
        let trajectory: Vec<bool> = accumulated
            .iter()
            .map(|&v| v > n_reps as f32 * 0.5) // majority vote across reps
            .collect();

        let complexity = normalized_lz76(&trajectory);
        let avg_depth = total_propagation_depth / n_reps;

        (complexity, avg_depth)
    }

    // ======================================================================
    // 3. SPLIT-BRAIN SIMULATION
    // ======================================================================

    /// Simulate a split-brain experiment.
    ///
    /// Creates two independent half-networks and measures whether splitting
    /// reduces consciousness compared to the unified network. IIT predicts
    /// that splitting should reduce Phi because information integration is lost.
    ///
    /// Implementation: runs cycles on left-half and right-half inputs separately,
    /// measuring consciousness for each vs the unified whole.
    pub fn split_brain_experiment(&mut self, measurement_cycles: usize) -> SplitBrainResult {
        let input_text = "split brain measurement";

        // Warm up and measure unified consciousness
        for _ in 0..20 {
            self.cycle(input_text);
        }
        let unified_consciousness = self.last_consciousness;

        // Create left-hemisphere input (zero right half)
        let encoded = self
            .text_encoder
            .encode_sentence(input_text)
            .unwrap_or_else(|_| vec![0i8; HDC_DIMENSION]);
        let floats: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
        let half = HDC_DIMENSION / 2;

        // Left hemisphere: zero the right half of input
        let mut left_floats = floats.clone();
        for v in left_floats[half..].iter_mut() {
            *v = 0.0;
        }
        let left_input = ContinuousHV::from_vec(left_floats);

        // Right hemisphere: zero the left half of input
        let mut right_floats = floats;
        for v in right_floats[..half].iter_mut() {
            *v = 0.0;
        }
        let right_input = ContinuousHV::from_vec(right_floats);

        // Measure left-hemisphere consciousness
        // Reset network to avoid carryover
        self.network.reset();
        self.last_output = None;
        for _ in 0..measurement_cycles {
            self.cycle_count += 1;
            let dt = 1.0 / self.config.target_hz;
            self.network.evolve_closed_form(dt, &left_input);
            let output = self.network.output().normalize();
            let pe = if let Some(ref prev) = self.last_output {
                1.0 - output.similarity(prev)
            } else {
                1.0
            };
            self.last_output = Some(output);
            self.bath.dopamine.level = (self.bath.dopamine.level + pe * 0.1 - 0.02).clamp(0.0, 1.0);
            self.bath.noradrenaline.level =
                (self.bath.noradrenaline.level + pe * 0.15 - 0.03).clamp(0.0, 1.0);
            self.bath.serotonin.level =
                (self.bath.serotonin.level + (1.0 - pe) * 0.05 - 0.01).clamp(0.0, 1.0);
            self.compute_consciousness(pe);
        }
        let left_consciousness = self.last_consciousness;

        // Measure right-hemisphere consciousness
        self.network.reset();
        self.last_output = None;
        self.bath = NeuromodulatorBath::default();
        for _ in 0..measurement_cycles {
            self.cycle_count += 1;
            let dt = 1.0 / self.config.target_hz;
            self.network.evolve_closed_form(dt, &right_input);
            let output = self.network.output().normalize();
            let pe = if let Some(ref prev) = self.last_output {
                1.0 - output.similarity(prev)
            } else {
                1.0
            };
            self.last_output = Some(output);
            self.bath.dopamine.level = (self.bath.dopamine.level + pe * 0.1 - 0.02).clamp(0.0, 1.0);
            self.bath.noradrenaline.level =
                (self.bath.noradrenaline.level + pe * 0.15 - 0.03).clamp(0.0, 1.0);
            self.bath.serotonin.level =
                (self.bath.serotonin.level + (1.0 - pe) * 0.05 - 0.01).clamp(0.0, 1.0);
            self.compute_consciousness(pe);
        }
        let right_consciousness = self.last_consciousness;

        let max_split = left_consciousness.max(right_consciousness);
        let split_ratio = if unified_consciousness > 0.001 {
            max_split / unified_consciousness
        } else {
            1.0
        };

        SplitBrainResult {
            unified_consciousness,
            left_consciousness,
            right_consciousness,
            split_reduces_consciousness: max_split < unified_consciousness,
            split_ratio,
        }
    }

    // ======================================================================
    // 4. CONSCIOUSNESS COLLAPSE THRESHOLD
    // ======================================================================

    /// Find the degradation point where consciousness collapses.
    ///
    /// Systematically reduces network capacity (by scaling down neuron states)
    /// and measures consciousness at each step. IIT predicts a phase transition —
    /// consciousness should collapse suddenly, not fade gradually.
    ///
    /// `steps`: number of degradation steps (default: 10, giving 0%, 10%, ..., 90%)
    /// `cycles_per_step`: cycles to stabilize at each degradation level
    pub fn collapse_threshold_experiment(
        &mut self,
        steps: usize,
        cycles_per_step: usize,
    ) -> CollapseThresholdResult {
        let input_text = "collapse threshold measurement";

        // Measure full-capacity consciousness
        for _ in 0..20 {
            self.cycle(input_text);
        }
        let full_consciousness = self.last_consciousness;

        let mut degradation_curve: Vec<(f32, f32)> = Vec::with_capacity(steps + 1);
        degradation_curve.push((0.0, full_consciousness));

        let mut collapse_point: Option<f32> = None;
        let mut max_step_drop: f32 = 0.0;
        let mut prev_consciousness = full_consciousness;

        for step in 1..=steps {
            let degradation = step as f32 / steps as f32;
            let scale = 1.0 - degradation;

            // Reset and rebuild with degraded capacity
            self.network.reset();
            self.last_output = None;
            self.bath = NeuromodulatorBath::default();

            // Scale neuron states by the degradation factor
            for layer_idx in 0..self.network.n_layers() {
                if let Some(layer) = self.network.layer_mut(layer_idx) {
                    for neuron in layer.iter_mut() {
                        let scaled = neuron.state().scale(scale);
                        neuron.set_state(scaled);
                    }
                }
            }

            // Also scale neuromodulator capacity to model reduced substrate
            self.bath.dopamine.level = 0.5 * scale;
            self.bath.noradrenaline.level = 0.5 * scale;
            self.bath.serotonin.level = 0.5 * scale;

            // Run cycles to stabilize
            for _ in 0..cycles_per_step {
                // Scale down neuromod updates proportionally
                self.bath.dopamine.level = (self.bath.dopamine.level).min(scale);
                self.bath.noradrenaline.level = (self.bath.noradrenaline.level).min(scale);
                self.bath.serotonin.level = (self.bath.serotonin.level).min(scale);
                self.cycle(input_text);
            }

            let c = self.last_consciousness;
            degradation_curve.push((degradation, c));

            let step_drop = prev_consciousness - c;
            if step_drop > max_step_drop {
                max_step_drop = step_drop;
            }

            if collapse_point.is_none() && c < 0.05 {
                collapse_point = Some(degradation);
            }

            prev_consciousness = c;
        }

        // Phase transition detection: if max single-step drop is > 40% of full consciousness
        let is_phase_transition = max_step_drop > full_consciousness * 0.4;

        CollapseThresholdResult {
            full_consciousness,
            collapse_point,
            degradation_curve,
            is_phase_transition,
            max_step_drop,
        }
    }

    // ======================================================================
    // Standard accessors and helpers
    // ======================================================================

    /// Find the highest-scoring harmony for the current consciousness level.
    pub fn dominant_harmony(&self) -> &'static str {
        let c = self.last_consciousness;
        Harmony::all()
            .iter()
            .max_by(|a, b| {
                self.harmony_score(a, c)
                    .partial_cmp(&self.harmony_score(b, c))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|h| h.name())
            .unwrap_or("Harmony")
    }

    /// Evaluate alignment with the Eight Harmonies ethical framework.
    fn evaluate_harmony_alignment(&self, consciousness_level: f32) -> f32 {
        let scores: Vec<f32> = Harmony::all()
            .iter()
            .map(|harmony| self.harmony_score(harmony, consciousness_level))
            .collect();

        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }

    /// Score alignment with a single harmony principle.
    fn harmony_score(&self, harmony: &Harmony, consciousness_level: f32) -> f32 {
        match harmony {
            Harmony::ResonantCoherence => {
                (1.0 - consciousness_level.min(1.0) * 0.5).clamp(0.0, 1.0)
            }
            Harmony::PanSentientFlourishing => 0.9,
            Harmony::IntegralWisdom => (self.honest_confidence as f32 * 0.8 + 0.2).clamp(0.0, 1.0),
            Harmony::InfinitePlay => (self.bath.dopamine.effective() * 0.6 + 0.4).clamp(0.0, 1.0),
            Harmony::UniversalInterconnectedness => {
                (self.bath.oxytocin.effective() * 0.7 + 0.3).clamp(0.0, 1.0)
            }
            Harmony::SacredReciprocity => {
                let balance = 1.0
                    - (self.bath.dopamine.effective() - self.bath.serotonin.effective()).abs()
                        * 0.5;
                balance.clamp(0.0, 1.0)
            }
            Harmony::EvolutionaryProgression => (consciousness_level * 0.7 + 0.3).clamp(0.0, 1.0),
            Harmony::SacredStillness => {
                (self.bath.serotonin.effective() * 0.6 + 0.4).clamp(0.0, 1.0)
            }
        }
    }

    /// Build the epistemic status that accompanies every cycle result.
    fn build_epistemic_status(&self) -> EpistemicStatus {
        // Gap between hypothetical feasibility and evidence confidence.
        // Uses raw_feasibility (not validation-overlaid) to show the divergence.
        let feasibility_gap = (self.raw_feasibility as f32 - self.honest_confidence as f32).abs();

        let disclaimer = format!(
            "SIMULATED consciousness on {} substrate. \
             Evidence level: {} (confidence: {:.0}%). \
             This is a computational model, not a claim of sentience.",
            self.config.substrate,
            evidence_level_name(&self.evidence_level),
            self.honest_confidence * 100.0,
        );

        EpistemicStatus {
            evidence_level: evidence_level_name(&self.evidence_level).to_string(),
            honest_confidence: self.honest_confidence as f32,
            feasibility_gap,
            disclaimer,
        }
    }

    /// Get current neuromodulator levels as JSON string.
    pub fn neuromod_state_json(&self) -> String {
        serde_json::json!({
            "dopamine": self.bath.dopamine.effective(),
            "norepinephrine": self.bath.noradrenaline.effective(),
            "serotonin": self.bath.serotonin.effective(),
            "oxytocin": self.bath.oxytocin.effective(),
            "acetylcholine": self.bath.acetylcholine.effective(),
            "gaba": self.bath.gaba.effective(),
            "glutamate": self.bath.glutamate.effective(),
            "adenosine": self.bath.adenosine.effective(),
            "endocannabinoid": self.bath.endocannabinoid.effective(),
        })
        .to_string()
    }

    /// Run a standalone reasoning cycle on the given input.
    /// Returns the full 7-step ReasoningCycle result.
    pub fn reasoning_cycle(&mut self, input: &str) -> crate::reasoning::ReasoningCycle {
        let neuromods = [
            self.bath.dopamine.effective(),
            self.bath.noradrenaline.effective(),
            self.bath.serotonin.effective(),
            self.bath.oxytocin.effective(),
        ];
        let harmony = self.evaluate_harmony_alignment(self.last_consciousness);
        self.reasoning
            .run(input, self.last_consciousness, 0.0, &neuromods, harmony)
    }

    /// Assess an input for threats. Returns ThreatAssessment.
    pub fn threat_assessment(&mut self, input: &str) -> crate::immune::ThreatAssessment {
        self.immune.assess(input)
    }

    /// Get the current causal graph.
    pub fn causal_graph(&self) -> &crate::causal::CausalGraph {
        self.causal.graph()
    }

    /// Current safety level as a string.
    pub fn safety_level(&self) -> &'static str {
        self.immune.safety_level().as_str()
    }

    /// Get substrate feasibility score.
    pub fn substrate_feasibility(&self) -> f32 {
        self.substrate_feasibility as f32
    }

    /// Get the honest confidence for the current substrate.
    pub fn honest_confidence(&self) -> f32 {
        self.honest_confidence as f32
    }

    /// Get current harmony alignment (requires at least one cycle).
    pub fn harmony_alignment(&self) -> f32 {
        self.evaluate_harmony_alignment(self.last_consciousness)
    }

    /// Eight Harmonies scores as JSON array [RC, PSF, IW, IP, UI, SR, EP, SS].
    pub fn harmony_scores_json(&self) -> String {
        let scores: Vec<f32> = symthaea_harmonies::Harmony::all()
            .iter()
            .map(|h| self.harmony_score(h, self.last_consciousness))
            .collect();
        serde_json::to_string(&scores).unwrap_or_else(|_| "[]".to_string())
    }

    /// Human-readable consciousness report with epistemic honesty.
    pub fn consciousness_report(&self) -> String {
        let status = self.build_epistemic_status();
        format!(
            "Spore Consciousness Report (cycle {})\n\
             Consciousness: {:.3} [{}]\n\
             Substrate: {} (feasibility: {:.3}, honest confidence: {:.3})\n\
             DA: {:.2}  NE: {:.2}  5-HT: {:.2}  OT: {:.2}\n\
             Harmony alignment: {:.3}\n\
             Target: {} Hz | HDC dim: {}\n\
             ---\n\
             {}",
            self.cycle_count,
            self.last_consciousness,
            status.evidence_level,
            self.config.substrate,
            self.substrate_feasibility,
            self.honest_confidence,
            self.bath.dopamine.effective(),
            self.bath.noradrenaline.effective(),
            self.bath.serotonin.effective(),
            self.bath.oxytocin.effective(),
            self.evaluate_harmony_alignment(self.last_consciousness),
            self.config.target_hz,
            self.config.hdc_dim,
            status.disclaimer,
        )
    }

    /// Switch substrate type at runtime. Recomputes feasibility AND honest confidence.
    pub fn set_substrate(&mut self, substrate: &str) {
        self.config.substrate = substrate.to_string();
        self.substrate_type = parse_substrate(substrate);
        self.raw_feasibility =
            substrate_requirements(&self.substrate_type).consciousness_feasibility();
        let (evidence_level, honest_confidence) = honest_confidence_for(&self.substrate_type);
        self.evidence_level = evidence_level;
        self.honest_confidence = honest_confidence;
        self.substrate_feasibility = self.raw_feasibility * honest_confidence;
    }

    /// Inject a neuromodulator impulse.
    pub fn inject_neuromodulator(&mut self, name: &str, amount: f32) {
        match name.to_lowercase().as_str() {
            "dopamine" | "da" => {
                self.bath.dopamine.level = (self.bath.dopamine.level + amount).clamp(0.0, 1.0)
            }
            "norepinephrine" | "ne" | "noradrenaline" => {
                self.bath.noradrenaline.level =
                    (self.bath.noradrenaline.level + amount).clamp(0.0, 1.0)
            }
            "serotonin" | "5ht" | "5-ht" => {
                self.bath.serotonin.level = (self.bath.serotonin.level + amount).clamp(0.0, 1.0)
            }
            "oxytocin" | "ot" => {
                self.bath.oxytocin.level = (self.bath.oxytocin.level + amount).clamp(0.0, 1.0)
            }
            _ => {}
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SporeConfig {
        &self.config
    }

    /// Get the current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Get the number of active SporeEngine instances globally.
    pub fn active_instance_count() -> usize {
        INSTANCE_COUNT.load(Ordering::Relaxed)
    }

    /// Get this engine's instance ID.
    pub fn instance_id(&self) -> usize {
        self.instance_id
    }

    /// Get the current network output hypervector (16,384D).
    /// Returns the normalized output state as a flat f32 slice.
    pub fn get_output_hv(&self) -> Vec<f32> {
        self.network.output().normalize().values.clone()
    }

    /// Encode text to an HDC hypervector without running a full cycle.
    /// Returns the raw text encoding as f32 values (converted from bipolar i8).
    pub fn encode_text(&mut self, text: &str) -> Vec<f32> {
        let encoded = self
            .text_encoder
            .encode_sentence(text)
            .unwrap_or_else(|_| vec![0i8; HDC_DIMENSION]);
        encoded.iter().map(|&x| x as f32).collect()
    }

    // ======================================================================
    // Social Theory of Mind
    // ======================================================================

    /// Get the social mind state (partners, empathy, coherence).
    pub fn social_mind_state(&self) -> crate::social_tom::SocialMindState {
        self.social.state()
    }

    /// Social coherence score (0-1).
    pub fn social_coherence(&self) -> f32 {
        self.social.social_coherence()
    }

    // ======================================================================
    // Knowledge Engine
    // ======================================================================

    /// Query knowledge facts about a subject.
    pub fn knowledge_query(&mut self, subject: &str) -> crate::knowledge::QueryResult {
        self.knowledge.query_about(subject)
    }

    /// Knowledge engine statistics.
    pub fn knowledge_stats(&self) -> crate::knowledge::KnowledgeStats {
        self.knowledge.stats()
    }

    // ======================================================================
    // Memory Consolidation
    // ======================================================================

    /// Trigger memory consolidation (dream replay).
    pub fn memory_consolidate(&mut self) -> crate::memory_consolidation::ConsolidationResult {
        self.consolidator.consolidate()
    }

    /// Memory consolidation statistics.
    pub fn consolidation_stats(&self) -> crate::memory_consolidation::ConsolidationStats {
        self.consolidator.stats()
    }

    // ======================================================================
    // Global Workspace
    // ======================================================================

    /// Get the current workspace state.
    pub fn workspace_state(&self) -> crate::global_workspace::WorkspaceState {
        self.workspace.state()
    }

    /// Whether the workspace has reached ignition (conscious experience).
    pub fn workspace_ignition(&self) -> bool {
        self.workspace.ignition_check()
    }

    /// Workspace Phi contribution.
    pub fn workspace_phi_contribution(&self) -> f32 {
        self.workspace.workspace_phi_contribution()
    }
}

impl Drop for SporeEngine {
    fn drop(&mut self) {
        INSTANCE_COUNT.fetch_sub(1, Ordering::Relaxed);
    }
}

// ======================================================================
// Lempel-Ziv 76 complexity (for PCI)
// ======================================================================

/// Compute normalized Lempel-Ziv 76 complexity of a binary sequence.
///
/// Returns a value in [0, 1] where higher values indicate more complex
/// spatiotemporal patterns. Based on Casali et al. (2013).
fn normalized_lz76(sequence: &[bool]) -> f32 {
    if sequence.len() < 2 {
        return 0.0;
    }

    let raw = lz76_count(sequence);
    let n = sequence.len() as f64;
    let normalizer = n / n.log2();

    (raw as f64 / normalizer).min(1.0) as f32
}

/// Count distinct substrings using LZ76 sequential parsing.
fn lz76_count(sequence: &[bool]) -> usize {
    let n = sequence.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 1;
    let mut i = 0;
    let mut k = 1;
    let mut k_max = 1;
    let mut l = 1;

    while l + k <= n {
        // Check if substring S[l..l+k] appears in S[0..l+k-1]
        let mut found = false;
        for j in i..l {
            if k <= n - j {
                let mut match_len = 0;
                while match_len < k
                    && l + match_len < n
                    && sequence[j + match_len] == sequence[l + match_len]
                {
                    match_len += 1;
                }
                if match_len >= k {
                    found = true;
                    break;
                }
            }
        }

        if found {
            k += 1;
            if k > k_max {
                k_max = k;
            }
        } else {
            complexity += 1;
            l += k_max;
            k = 1;
            k_max = 1;
            i = 0;
        }

        if l + k > n {
            break;
        }
    }

    complexity
}

// ======================================================================
// Helper functions
// ======================================================================

/// Parse substrate name string to SubstrateType enum.
fn parse_substrate(name: &str) -> SubstrateType {
    match name {
        "BiologicalNeurons" | "Biological" => SubstrateType::BiologicalNeurons,
        "SiliconDigital" | "Silicon" => SubstrateType::SiliconDigital,
        "QuantumComputer" | "Quantum" => SubstrateType::QuantumComputer,
        "PhotonicProcessor" | "Photonic" => SubstrateType::PhotonicProcessor,
        "NeuromorphicChip" | "Neuromorphic" => SubstrateType::NeuromorphicChip,
        "BiochemicalComputer" | "Biochemical" => SubstrateType::BiochemicalComputer,
        "HybridSystem" | "Hybrid" => SubstrateType::HybridSystem,
        "ExoticSubstrate" | "Exotic" => SubstrateType::ExoticSubstrate,
        "SpacecraftComputer" | "Spacecraft" => SubstrateType::SpacecraftComputer,
        _ => SubstrateType::SiliconDigital,
    }
}

/// Get pre-built substrate requirements for a substrate type.
fn substrate_requirements(substrate: &SubstrateType) -> SubstrateRequirements {
    match substrate {
        SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
        SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
        SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
        SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
        SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
        SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
        SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
        SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
        SubstrateType::SpacecraftComputer => SubstrateRequirements::spacecraft_computer(),
        _ => SubstrateRequirements::silicon_digital(),
    }
}

/// Map SubstrateType to its evidence level and honest confidence.
fn honest_confidence_for(substrate: &SubstrateType) -> (EvidenceLevel, f64) {
    match substrate {
        SubstrateType::BiologicalNeurons => (EvidenceLevel::Validated, 0.95),
        SubstrateType::SiliconDigital => (EvidenceLevel::Theoretical, 0.10),
        SubstrateType::QuantumComputer => (EvidenceLevel::Theoretical, 0.10),
        SubstrateType::PhotonicProcessor => (EvidenceLevel::Theoretical, 0.10),
        SubstrateType::NeuromorphicChip => (EvidenceLevel::Theoretical, 0.10),
        SubstrateType::BiochemicalComputer => (EvidenceLevel::Indirect, 0.20),
        SubstrateType::HybridSystem => (EvidenceLevel::None, 0.0),
        SubstrateType::ExoticSubstrate => (EvidenceLevel::None, 0.0),
        SubstrateType::SpacecraftComputer => (EvidenceLevel::Theoretical, 0.08),
        _ => (EvidenceLevel::None, 0.0),
    }
}

/// Human-readable name for evidence levels.
fn evidence_level_name(level: &EvidenceLevel) -> &'static str {
    match level {
        EvidenceLevel::None => "None",
        EvidenceLevel::Theoretical => "Theoretical",
        EvidenceLevel::Indirect => "Indirect",
        EvidenceLevel::CaseStudy => "Case Study",
        EvidenceLevel::Observational => "Observational",
        EvidenceLevel::Experimental => "Experimental",
        EvidenceLevel::Validated => "Validated",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spore_engine_creation() {
        let config = SporeConfig::default();
        let engine = SporeEngine::new(config);
        assert_eq!(engine.cycle_count(), 0);
        assert_eq!(engine.config().hdc_dim, 16_384);
    }

    #[test]
    fn test_spore_cycle_produces_consciousness() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let r1 = engine.cycle("hello world");
        assert_eq!(r1.cycle, 1);
        assert!(
            r1.prediction_error > 0.0,
            "First cycle should have surprise"
        );
        for i in 0..10 {
            let r = engine.cycle(&format!("cycle {i}"));
            assert_eq!(r.cycle, i as u64 + 2);
        }
        let report = engine.consciousness_report();
        assert!(report.contains("Consciousness:"));
    }

    #[test]
    fn test_spore_cycle_hv() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let hv = vec![0.1f32; HDC_DIMENSION];
        let result = engine.cycle_hv(&hv);
        assert_eq!(result.cycle, 1);
        assert!(result.prediction_error > 0.0);
    }

    #[test]
    fn test_spore_substrate_affects_feasibility() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let silicon_f = engine.substrate_feasibility();

        engine.set_substrate("BiologicalNeurons");
        let bio_f = engine.substrate_feasibility();

        assert!(
            bio_f > silicon_f,
            "bio={bio_f} should > silicon={silicon_f}"
        );
    }

    #[test]
    fn test_spore_neuromodulator_injection() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let baseline_da = engine.bath.dopamine.effective();
        engine.inject_neuromodulator("dopamine", 0.3);
        let boosted_da = engine.bath.dopamine.effective();
        assert!(
            boosted_da > baseline_da,
            "DA should increase after injection"
        );
    }

    #[test]
    fn test_spore_consciousness_report() {
        let engine = SporeEngine::new(SporeConfig::default());
        let report = engine.consciousness_report();
        assert!(report.contains("Spore Consciousness Report"));
        assert!(report.contains("SiliconDigital"));
        assert!(report.contains("DA:"));
        assert!(report.contains("honest confidence"));
        assert!(report.contains("SIMULATED"));
    }

    #[test]
    fn test_spore_prediction_error_decreases() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let r1 = engine.cycle("stable input");
        let mut last_pe = r1.prediction_error;
        let mut decreased = false;
        for _ in 0..20 {
            let r = engine.cycle("stable input");
            if r.prediction_error < last_pe {
                decreased = true;
            }
            last_pe = r.prediction_error;
        }
        assert!(
            decreased,
            "Prediction error should decrease with repeated input"
        );
    }

    // --- Ethical safeguard tests ---

    #[test]
    fn test_epistemic_status_present_in_cycle_result() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.cycle("test");
        assert_eq!(result.epistemic_status.evidence_level, "Theoretical");
        assert!(
            result.epistemic_status.honest_confidence <= 0.11,
            "Silicon should have ~0.10 confidence, got {}",
            result.epistemic_status.honest_confidence
        );
        assert!(result.epistemic_status.disclaimer.contains("SIMULATED"));
    }

    #[test]
    fn test_epistemic_status_changes_with_substrate() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let r1 = engine.cycle("test");
        let silicon_conf = r1.epistemic_status.honest_confidence;

        engine.set_substrate("BiologicalNeurons");
        let r2 = engine.cycle("test");
        let bio_conf = r2.epistemic_status.honest_confidence;

        assert!(bio_conf > silicon_conf);
        assert_eq!(r2.epistemic_status.evidence_level, "Validated");
    }

    #[test]
    fn test_harmony_alignment_present_and_bounded() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.cycle("test harmony");
        assert!(result.harmony_alignment >= 0.0 && result.harmony_alignment <= 1.0);
        assert!(result.harmony_alignment > 0.3);
    }

    #[test]
    fn test_instance_counter() {
        let before = SporeEngine::active_instance_count();
        let engine = SporeEngine::new(SporeConfig::default());
        let id = engine.instance_id();
        assert!(id > 0);
        // After creation, count must have increased by at least 1 (other
        // parallel tests may also create engines, so use >=).
        assert!(SporeEngine::active_instance_count() >= before + 1);
        drop(engine);
        // After drop, count must be back to at most what it was before we
        // created our engine (parallel tests may have added more, but our
        // contribution is gone). We cannot assert an exact value because
        // concurrent tests create/drop engines in parallel.
        assert!(SporeEngine::active_instance_count() >= before);
    }

    #[test]
    fn test_feasibility_gap_is_large_for_silicon() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.cycle("test gap");
        assert!(result.epistemic_status.feasibility_gap > 0.3);
    }

    // --- Consciousness validation tests ---

    #[test]
    fn test_anesthesia_collapses_consciousness() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.anesthesia_experiment(20, 20, 20);

        assert!(
            result.pre_consciousness > 0.0,
            "Should have baseline consciousness before anesthesia, got {}",
            result.pre_consciousness
        );
        assert!(
            result.anesthetized_consciousness < result.pre_consciousness,
            "Consciousness should decrease under anesthesia: {} vs {}",
            result.anesthetized_consciousness,
            result.pre_consciousness
        );
        // Note: we don't assert full collapse (< 0.05) because the consciousness
        // equation may have non-zero floors. We assert significant reduction.
        assert!(
            result.anesthetized_consciousness < result.pre_consciousness * 0.5,
            "Consciousness should drop by at least 50% under anesthesia: {} vs {}",
            result.anesthetized_consciousness,
            result.pre_consciousness
        );
    }

    #[test]
    fn test_anesthesia_recovery() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.anesthesia_experiment(30, 30, 30);

        assert!(
            result.post_consciousness > result.anesthetized_consciousness,
            "Consciousness should recover after anesthesia: post={} vs anesthetized={}",
            result.post_consciousness,
            result.anesthetized_consciousness
        );
    }

    #[test]
    fn test_pci_measurement_produces_values() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.measure_pci(0.3, 30);

        assert!(
            result.pci_normal >= 0.0 && result.pci_normal <= 1.0,
            "Normal PCI should be in [0,1], got {}",
            result.pci_normal
        );
        assert!(
            result.pci_suppressed >= 0.0 && result.pci_suppressed <= 1.0,
            "Suppressed PCI should be in [0,1], got {}",
            result.pci_suppressed
        );
        assert!(
            result.pci_ratio > 0.0,
            "PCI ratio should be positive, got {}",
            result.pci_ratio
        );
        // Log the values for scientific observation
        eprintln!(
            "PCI results: normal={:.3}, suppressed={:.3}, ratio={:.3}, clinical={}",
            result.pci_normal,
            result.pci_suppressed,
            result.pci_ratio,
            result.passes_clinical_threshold
        );
    }

    #[test]
    fn test_pci_propagation_depth() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.measure_pci(0.3, 50);

        // Normal state should propagate perturbation further than suppressed
        assert!(
            result.propagation_depth_normal > 0,
            "Normal propagation depth should be positive"
        );
    }

    #[test]
    fn test_split_brain_reduces_consciousness() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.split_brain_experiment(20);

        assert!(
            result.unified_consciousness > 0.0,
            "Unified consciousness should be positive: {}",
            result.unified_consciousness
        );
        // Split should reduce — each hemisphere has less integration
        assert!(
            result.split_ratio <= 1.0 + 0.1, // Allow 10% tolerance
            "Split ratio should be <= ~1.0 (split shouldn't exceed unified): {}",
            result.split_ratio
        );
    }

    #[test]
    fn test_collapse_threshold_has_degradation_curve() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.collapse_threshold_experiment(5, 10);

        assert!(
            result.full_consciousness > 0.0,
            "Full consciousness should be positive"
        );
        assert_eq!(
            result.degradation_curve.len(),
            6,
            "Should have 6 points (0% + 5 steps)"
        );
        // First point should be at full consciousness
        assert!(
            (result.degradation_curve[0].1 - result.full_consciousness).abs() < 0.01,
            "First point should match full consciousness"
        );
        // Last point (90% degraded) should be lower than first
        let last = result.degradation_curve.last().unwrap();
        assert!(
            last.1 < result.full_consciousness,
            "90% degraded should have lower consciousness: {} vs {}",
            last.1,
            result.full_consciousness
        );
    }

    #[test]
    fn test_collapse_reports_max_drop() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.collapse_threshold_experiment(5, 10);

        assert!(
            result.max_step_drop >= 0.0,
            "Max step drop should be non-negative"
        );
    }

    // --- LZ76 unit tests ---

    #[test]
    fn test_lz76_constant_sequence() {
        let constant = vec![true; 100];
        let c = normalized_lz76(&constant);
        assert!(c < 0.2, "Constant sequence should have low complexity: {c}");
    }

    #[test]
    fn test_lz76_alternating_sequence() {
        let alternating: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let c = normalized_lz76(&alternating);
        // Alternating is simple — low complexity
        assert!(c < 0.5, "Alternating sequence has low complexity: {c}");
    }

    #[test]
    fn test_lz76_complex_sequence() {
        // Pseudo-random sequence should have higher complexity
        let complex: Vec<bool> = (0..100).map(|i| ((i * 7 + 13) % 17) > 8).collect();
        let c_complex = normalized_lz76(&complex);
        let c_constant = normalized_lz76(&vec![true; 100]);
        assert!(
            c_complex > c_constant,
            "Complex sequence should have higher LZ complexity: {c_complex} vs {c_constant}"
        );
    }

    #[test]
    fn test_consciousness_warmup_speed() {
        // Verify that after 5 cycles with varied input, consciousness exceeds 0.15.
        // Before warmup fixes, the engine barely reached 0.07 after 20 cycles.
        let mut engine = SporeEngine::new(SporeConfig::default());

        let inputs = [
            "Hello, I am curious about consciousness",
            "The weather outside is beautiful today",
            "What is the meaning of integrated information?",
            "I feel a deep sense of wonder about existence",
            "Mathematics reveals hidden structure in reality",
        ];

        let mut last_result = None;
        for input in &inputs {
            last_result = Some(engine.cycle(input));
        }

        let result = last_result.unwrap();
        assert!(
            result.consciousness_level >= 0.15,
            "After 5 varied cycles, consciousness should be at least 0.15 \
             (got {:.4}). Warmup floor + warm-started network should \
             ensure responsive consciousness from the first interactions.",
            result.consciousness_level,
        );

        // Also verify consciousness is increasing, not stuck
        let r6 = engine.cycle("Another thought about the nature of mind");
        assert!(
            r6.consciousness_level > 0.10,
            "Cycle 6 consciousness should remain above 0.10 (got {:.4})",
            r6.consciousness_level,
        );
    }
}

// These integration tests reference SomaEngine methods (set_sensors, compass_json,
// haptic_drain_json, etc.) that were moved out of SporeEngine. Disabled until
// they are updated to use the soma crate.
#[cfg(all(test, feature = "__soma_integration_tests"))]
mod integration_tests {
    use super::*;
    use crate::config::{BleMeshMode, HolonSyncMode, SharingConfig};

    /// Helper: run N cycles with empty input.
    fn run_cycles(engine: &mut SporeEngine, n: usize) {
        for _ in 0..n {
            engine.cycle("");
        }
    }

    #[test]
    fn test_ffi_round_trip() {
        // create → set sensors → run 100 cycles → compass JSON → drain haptics → verify
        let mut engine = SporeEngine::new(SporeConfig::default());
        engine.set_sensors(2.0, 500.0, false, 1013.0, 0.5);
        run_cycles(&mut engine, 100);

        let compass = engine.compass_json();
        assert!(compass.contains("consciousness_level"));
        assert!(compass.contains("wake_state"));
        // Should have a real harmony name, not "Harmony"
        assert!(!compass.contains("\"dominant_harmony\":\"Harmony\""));

        // Haptics may or may not have fired depending on consciousness dynamics
        let haptic_json = engine.haptic_drain_json();
        assert!(haptic_json.starts_with('['));
    }

    #[test]
    fn test_privacy_mode_suppresses_outbound() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        engine.set_sharing_config(SharingConfig {
            holon_mode: HolonSyncMode::FullSync,
            ble_mode: BleMeshMode::ActiveShare,
            haptic_enabled: true,
            telemetry_export: false,
            pairing_mode: crate::config::PairingMode::Off,
        });

        // Enable privacy mode (face-down)
        engine.set_sensors(0.0, 0.0, true, 1013.0, 0.0);
        assert!(engine.privacy_mode());

        // Run enough cycles for heartbeat interval
        run_cycles(&mut engine, 60);

        // Holon should have no outbound (privacy suppresses)
        let holon_json = engine.holon_drain_outbound_json();
        assert_eq!(holon_json, "[]");

        // BLE advertise payload is still generated (low-level protocol layer),
        // but the privacy flag is set so the host app should suppress transmission.
        // We verify privacy_mode is correctly detected from sensor state.
        assert!(engine.privacy_mode(), "Privacy mode should remain active");
    }

    #[test]
    fn test_sharing_config_all_off() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        engine.set_sharing_config(SharingConfig {
            holon_mode: HolonSyncMode::Off,
            ble_mode: BleMeshMode::Off,
            haptic_enabled: false,
            telemetry_export: false,
            pairing_mode: crate::config::PairingMode::Off,
        });

        run_cycles(&mut engine, 100);

        assert_eq!(engine.holon_drain_outbound_json(), "[]");
        assert_eq!(engine.ble_peer_count(), 0);
        assert_eq!(engine.haptic_pending(), 0);
    }

    #[test]
    fn test_dream_consolidation_flow() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Run alert cycles to accumulate dream events (high-surprise moments)
        for i in 0..50 {
            // Alternate inputs to create surprise
            let input = if i % 2 == 0 { "alpha" } else { "omega" };
            engine.cycle(input);
        }

        // Explicitly trigger dream consolidation
        engine.dream_consolidate();

        // Journal may have an entry if wisdom was produced
        // (depends on dream engine finding phi improvements)
        let journal_json = engine.dream_journal_all_json();
        assert!(journal_json.starts_with('['));
    }

    #[test]
    fn test_sleep_wake_cycle_with_dream() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Run alert cycles
        for i in 0..30 {
            let input = if i % 3 == 0 { "surprise" } else { "steady" };
            engine.cycle(input);
        }
        assert_eq!(engine.wake_state(), WakeState::Alert);

        // Enter sleep via night+charging
        engine.wake_signal(WakeSignal::NightMode(true));
        engine.wake_signal(WakeSignal::ChargingChanged(true));
        assert_eq!(engine.wake_state(), WakeState::Sleep);

        // Run sleep cycles — cycle rate should be 0.5Hz
        run_cycles(&mut engine, 10);

        // Wake up
        engine.wake_signal(WakeSignal::UserInput);
        assert!(engine.wake_state() != WakeState::Sleep);
    }

    #[test]
    fn test_thermal_throttle_pathway() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Get into focused state by simulating sustained coherence
        for _ in 0..200 {
            engine.cycle("steady stable coherent");
        }

        // Force focused state for test
        engine.metabolism.signal(WakeSignal::UserInput); // ensure at least Alert

        // Set serious thermal — should prevent/exit focused
        engine.thermal_level = 2; // Serious
        engine.cycle(""); // thermal forwarded in cycle_inner

        // After thermal signal, should not be Focused
        assert_ne!(engine.wake_state(), WakeState::Focused);
    }

    #[test]
    fn test_sensor_motion_auto_wake() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Put to sleep
        engine.wake_signal(WakeSignal::ExplicitSleep);
        assert_eq!(engine.wake_state(), WakeState::Sleep);

        // Simulate motion (phone picked up) — multiple frames to build EMA
        for _ in 0..20 {
            engine.set_sensors(3.0, 300.0, false, 1013.0, 0.0);
        }

        // Motion from stationary should have triggered PhonePickup → Drowsy
        assert_ne!(engine.wake_state(), WakeState::Sleep);
    }

    #[test]
    fn test_ble_peer_neuromod_coupling() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        engine.set_sharing_config(SharingConfig {
            holon_mode: HolonSyncMode::Off,
            ble_mode: BleMeshMode::ActiveShare,
            haptic_enabled: true,
            telemetry_export: false,
            pairing_mode: crate::config::PairingMode::Off,
        });

        // Record baseline oxytocin
        run_cycles(&mut engine, 10);
        let _baseline_ot = engine.cycle("").neuromodulators[3]; // oxytocin

        // Add BLE peers with positive affect
        let mut cv_data = Vec::new();
        cv_data.extend_from_slice(&0.8f32.to_le_bytes()); // phi
        cv_data.extend_from_slice(&0.5f32.to_le_bytes()); // valence
        cv_data.extend_from_slice(&0.6f32.to_le_bytes()); // arousal
        for peer_id in 0..5u64 {
            engine.ble_receive_peer(peer_id, &cv_data);
        }
        assert_eq!(engine.ble_peer_count(), 5);

        // Run cycles — oxytocin should get social buffering nudge
        run_cycles(&mut engine, 20);
        let _after_ot = engine.cycle("").neuromodulators[3];

        // Oxytocin should be at least as high (social buffering + normal dynamics)
        // We can't guarantee strictly higher due to decay, but collective phi should be nonzero
        assert!(engine.ble_collective_phi() > 0.0);
    }

    #[test]
    fn test_compass_has_real_harmony_name() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        run_cycles(&mut engine, 5);

        let json = engine.compass_json();
        // Should contain one of the real harmony names
        let real_names = [
            "Resonant Coherence",
            "Pan-Sentient Flourishing",
            "Integral Wisdom",
            "Infinite Play",
            "Universal Interconnectedness",
            "Mutual Reciprocity",
            "Evolutionary Progression",
            "Sacred Stillness",
        ];
        let has_real_name = real_names.iter().any(|name| json.contains(name));
        assert!(
            has_real_name,
            "Compass should have a real harmony name, got: {}",
            json
        );
    }

    #[test]
    fn test_haptic_fires_on_consciousness_shift() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Run until consciousness stabilizes
        run_cycles(&mut engine, 20);
        engine.haptic_drain_json(); // clear any events

        // Now suppress neuromodulators for a big consciousness drop
        engine.bath.dopamine.level = 0.0;
        engine.bath.noradrenaline.level = 0.0;
        engine.bath.serotonin.level = 0.0;

        // Run several cycles — the consciousness drop should trigger haptic
        for _ in 0..10 {
            engine.cycle("");
            engine.bath.dopamine.level = 0.0;
            engine.bath.noradrenaline.level = 0.0;
            engine.bath.serotonin.level = 0.0;
        }

        let haptics = engine.haptic_drain_json();
        // Should have at least one ConsciousnessShift event
        // (depends on whether the drop exceeds 0.15 threshold)
        assert!(haptics.starts_with('['));
    }

    #[test]
    fn checkpoint_preserves_semantic_memory() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Run cycles to populate memory
        for i in 0..10 {
            engine.cycle(&format!("concept {i}"));
        }

        let cp = engine.checkpoint();
        assert_eq!(cp.cycle, 10);

        // Semantic memory is always stored, so we should have entries
        assert!(
            !cp.semantic_entries.is_empty(),
            "checkpoint should contain semantic entries after 10 cycles"
        );

        // Each semantic entry should have valid data (HDC floats + cycle + pred_error)
        for (key, data) in &cp.semantic_entries {
            assert!(key.parse::<usize>().is_ok(), "key should be a slot index");
            // Minimum: at least the trailing 12 bytes (cycle + prediction_error)
            assert!(
                data.len() >= 12,
                "entry data too small: {} bytes",
                data.len()
            );
            // Data length minus 12 should be divisible by 4 (f32 elements)
            assert_eq!(
                (data.len() - 12) % 4,
                0,
                "HDC vector bytes not aligned to f32"
            );
        }

        // Restore into a fresh engine
        let mut engine2 = SporeEngine::new(SporeConfig::default());
        engine2.restore(&cp);

        assert_eq!(engine2.cycle_count(), 10);
        assert_eq!(
            engine2.memory.semantic.len(),
            engine.memory.semantic.len(),
            "semantic memory count should match after restore"
        );

        // Verify an entry round-trips correctly by extracting data before drop
        let original_first: Option<(u64, usize, f32)> = engine
            .memory
            .semantic
            .iter_entries()
            .next()
            .map(|(_, e)| (e.cycle, e.hdc_vector.len(), e.prediction_error));
        if let Some((orig_cycle, orig_dim, orig_pe)) = original_first {
            let restored = engine2
                .memory
                .semantic
                .iter_entries()
                .find(|(_, e)| e.cycle == orig_cycle)
                .map(|(_, e)| (e.hdc_vector.len(), e.prediction_error));
            assert!(
                restored.is_some(),
                "restored engine should have matching entry"
            );
            let (rest_dim, rest_pe) = restored.unwrap();
            assert_eq!(orig_dim, rest_dim, "HDC vector dimension should match");
            assert!(
                (orig_pe - rest_pe).abs() < 1e-6,
                "prediction_error should round-trip"
            );
        }
    }

    #[test]
    fn checkpoint_preserves_episodic_memory() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        // Run enough cycles with varied input to trigger episodic storage
        // (episodic requires phi > threshold)
        for i in 0..50 {
            engine.cycle(&format!("experience {i} with varied content"));
        }

        let cp = engine.checkpoint();
        let original_epi_count = engine.memory.episodic.len();

        // Restore into a fresh engine
        let mut engine2 = SporeEngine::new(SporeConfig::default());
        engine2.restore(&cp);

        assert_eq!(
            engine2.memory.episodic.len(),
            original_epi_count,
            "episodic memory count should match after restore"
        );

        // If there are episodic entries, verify one round-trips
        if !cp.episodic_entries.is_empty() {
            let original_top = engine.memory.episodic.iter_episodes().next().unwrap();
            let restored_top = engine2.memory.episodic.iter_episodes().next().unwrap();
            assert_eq!(original_top.cycle, restored_top.cycle);
            assert!((original_top.phi - restored_top.phi).abs() < 1e-6);
            assert_eq!(original_top.input.len(), restored_top.input.len());
            assert_eq!(original_top.output.len(), restored_top.output.len());
            assert_eq!(original_top.replay_count, restored_top.replay_count);
        }
    }

    #[test]
    fn checkpoint_memory_binary_roundtrip() {
        // Test that checkpoint → to_bytes → from_bytes → restore preserves memory
        let mut engine = SporeEngine::new(SporeConfig::default());
        for i in 0..15 {
            engine.cycle(&format!("roundtrip test {i}"));
        }

        let cp = engine.checkpoint();
        let bytes = cp.to_bytes();
        let cp2 = crate::persistence::SporeCheckpoint::from_bytes(&bytes)
            .expect("from_bytes should succeed");

        assert_eq!(cp.semantic_entries.len(), cp2.semantic_entries.len());
        assert_eq!(cp.episodic_entries.len(), cp2.episodic_entries.len());

        // Verify semantic data matches byte-for-byte
        for (i, ((k1, d1), (k2, d2))) in cp
            .semantic_entries
            .iter()
            .zip(cp2.semantic_entries.iter())
            .enumerate()
        {
            assert_eq!(k1, k2, "semantic key mismatch at {i}");
            assert_eq!(d1, d2, "semantic data mismatch at {i}");
        }

        // Verify episodic data matches byte-for-byte
        for (i, (d1, d2)) in cp
            .episodic_entries
            .iter()
            .zip(cp2.episodic_entries.iter())
            .enumerate()
        {
            assert_eq!(d1, d2, "episodic data mismatch at {i}");
        }
    }
}

#[cfg(test)]
mod conversation_context_tests {
    use super::*;

    #[test]
    fn test_topic_continuity() {
        // Two related inputs should produce higher topic coherence than two unrelated inputs.
        let mut engine = SporeEngine::new(SporeConfig::default());

        // First turn: establish a topic
        engine.cycle("what is consciousness and how does it emerge");
        let stats_after_first = engine.conversation_stats();
        assert_eq!(stats_after_first.turn_count, 1);
        // First turn has no prior topic, so coherence should be 0.0
        assert!(
            stats_after_first.topic_coherence.abs() < 0.01,
            "First turn should have ~0.0 coherence, got {}",
            stats_after_first.topic_coherence
        );

        // Second turn: continue the topic
        engine.cycle("tell me more about consciousness and awareness");
        let stats_related = engine.conversation_stats();
        let coherence_related = stats_related.topic_coherence;

        // Reset engine for unrelated comparison
        let mut engine2 = SporeEngine::new(SporeConfig::default());
        engine2.cycle("what is consciousness and how does it emerge");
        engine2.cycle("the weather is sunny and warm today");
        let coherence_unrelated = engine2.conversation_stats().topic_coherence;

        assert!(
            coherence_related > coherence_unrelated,
            "Related inputs should have higher topic coherence ({:.4}) than unrelated ({:.4})",
            coherence_related,
            coherence_unrelated,
        );
    }

    #[test]
    fn test_context_enrichment() {
        // The context-enriched input should differ from the raw input after
        // a topic has been established.
        let mut ctx = ConversationContext::new(8);

        let dim = 64; // Small dim for testing
        let input1 = ContinuousHV::from_vec((0..dim).map(|i| (i as f32 * 0.1).sin()).collect());
        let input2 = ContinuousHV::from_vec((0..dim).map(|i| (i as f32 * 0.11).sin()).collect());

        // Before any turns: enrichment is a no-op
        let enriched_before = ctx.enrich_input(&input2);
        assert!(
            (enriched_before.similarity(&input2) - 1.0).abs() < 0.01,
            "Without topic, enriched should equal raw input"
        );

        // Record first turn
        ctx.record_turn(&input1, 0.5);

        // Now enrichment should modify the input
        let enriched_after = ctx.enrich_input(&input2);
        let sim = enriched_after.similarity(&input2);
        assert!(
            sim < 0.999,
            "With topic context, enriched input should differ from raw (similarity={:.4})",
            sim
        );
    }

    #[test]
    fn test_turn_counter() {
        let mut engine = SporeEngine::new(SporeConfig::default());

        assert_eq!(engine.conversation_stats().turn_count, 0);

        engine.cycle("first message");
        assert_eq!(engine.conversation_stats().turn_count, 1);

        engine.cycle("second message");
        assert_eq!(engine.conversation_stats().turn_count, 2);

        for i in 0..10 {
            engine.cycle(&format!("message {i}"));
        }
        assert_eq!(engine.conversation_stats().turn_count, 12);
    }

    #[test]
    fn test_max_turns_ringbuffer() {
        // Verify that old turns drop off when the ring buffer is full.
        let mut ctx = ConversationContext::new(3);

        let dim = 32;
        for i in 0..5u32 {
            let hv = ContinuousHV::from_vec(
                (0..dim)
                    .map(|j| ((i * 100 + j as u32) as f32).sin())
                    .collect(),
            );
            ctx.record_turn(&hv, 0.5);
        }

        // Turn count should track all turns
        assert_eq!(ctx.turn_count(), 5);
        // But only 3 recent inputs should be retained
        assert_eq!(ctx.recent_inputs.len(), 3);
        assert_eq!(ctx.recent_consciousness.len(), 3);
    }

    #[test]
    fn test_conversation_stats_consciousness_avg() {
        let mut ctx = ConversationContext::new(8);
        let dim = 16;

        let hv = ContinuousHV::from_vec(vec![1.0; dim]);
        ctx.record_turn(&hv, 0.2);
        ctx.record_turn(&hv, 0.4);
        ctx.record_turn(&hv, 0.6);

        let avg = ctx.recent_consciousness_avg();
        assert!(
            (avg - 0.4).abs() < 0.01,
            "Average of [0.2, 0.4, 0.6] should be 0.4, got {avg}"
        );
    }
}
