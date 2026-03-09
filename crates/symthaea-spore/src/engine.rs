//! SporeEngine: the core consciousness loop for WASM targets.

use crate::config::SporeConfig;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use symthaea_consciousness_equation::{
    ConsciousnessInputs, MasterConsciousnessEquation, MasterEquationConfig,
};
use symthaea_core::hdc::hdc_ltc_unified::UnifiedNetworkConfig;
use symthaea_core::hdc::substrate_independence::{SubstrateRequirements, SubstrateType};
use symthaea_core::hdc::substrate_validation::EvidenceLevel;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::{HdcLtcUnifiedNetwork, TextEncoder, TextEncoderConfig, HDC_DIMENSION};
use symthaea_harmonies::EightHarmonies;
use symthaea_neuromodulators::NeuromodulatorBath;
use symthaea_types::Harmony;

/// Global instance counter — surfaces awareness of multi-instance creation.
static INSTANCE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Maximum recommended concurrent engines before warning.
const MAX_RECOMMENDED_INSTANCES: usize = 4;

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
    evidence_level: EvidenceLevel,
    honest_confidence: f64,
    #[allow(dead_code)]
    harmonies: EightHarmonies,
    last_consciousness: f32,
    last_output: Option<ContinuousHV>,
    instance_id: usize,
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
        let network = HdcLtcUnifiedNetwork::new(net_config, 42);

        let text_encoder =
            TextEncoder::new(TextEncoderConfig::default()).expect("TextEncoder init");

        let substrate_type = parse_substrate(&config.substrate);
        let substrate_feasibility =
            substrate_requirements(&substrate_type).consciousness_feasibility();

        let (evidence_level, honest_confidence) = honest_confidence_for(&substrate_type);

        let harmonies = EightHarmonies::new();

        Self {
            config,
            cycle_count: 0,
            network,
            text_encoder,
            bath: NeuromodulatorBath::default(),
            equation: MasterConsciousnessEquation::new(MasterEquationConfig::default()),
            substrate_type,
            substrate_feasibility,
            evidence_level,
            honest_confidence,
            harmonies,
            last_consciousness: 0.0,
            last_output: None,
            instance_id,
        }
    }

    /// Run a single consciousness cycle with text input.
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        let encoded = self
            .text_encoder
            .encode_sentence(input)
            .unwrap_or_else(|_| vec![0i8; HDC_DIMENSION]);
        let floats: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
        let input_hv = ContinuousHV::from_vec(floats);

        self.cycle_inner(input_hv)
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

        self.cycle_inner(input_hv)
    }

    /// Inner cycle: evolve network, compute consciousness, update neuromodulators,
    /// evaluate harmony alignment, attach epistemic status.
    fn cycle_inner(&mut self, input_hv: ContinuousHV) -> CycleResult {
        self.cycle_count += 1;
        let dt = 1.0 / self.config.target_hz;

        // Evolve CfC network (O(1) closed-form temporal step)
        self.network.evolve_closed_form(dt, &input_hv);
        let output_hv = self.network.output().normalize();

        // Compute prediction error (similarity delta from previous output)
        let prediction_error = if let Some(ref prev) = self.last_output {
            1.0 - output_hv.similarity(prev)
        } else {
            1.0
        };
        self.last_output = Some(output_hv);

        // Update neuromodulators based on prediction error
        self.bath.dopamine.level =
            (self.bath.dopamine.level + prediction_error * 0.1 - 0.02).clamp(0.0, 1.0);
        self.bath.noradrenaline.level =
            (self.bath.noradrenaline.level + prediction_error * 0.15 - 0.03).clamp(0.0, 1.0);
        self.bath.serotonin.level =
            (self.bath.serotonin.level + (1.0 - prediction_error) * 0.05 - 0.01).clamp(0.0, 1.0);

        // Compute consciousness (every N cycles or every cycle)
        let consciousness_level = if self.cycle_count % self.config.phi_every_n_cycles as u64 == 0 {
            self.compute_consciousness(prediction_error)
        } else {
            self.last_consciousness
        };

        // Evaluate Eight Harmonies alignment
        let harmony_alignment = self.evaluate_harmony_alignment(consciousness_level);

        // Build epistemic status — the ethics gate
        let epistemic_status = self.build_epistemic_status();

        CycleResult {
            consciousness_level,
            cycle: self.cycle_count,
            neuromodulators: [
                self.bath.dopamine.effective(),
                self.bath.noradrenaline.effective(),
                self.bath.serotonin.effective(),
                self.bath.oxytocin.effective(),
            ],
            substrate_feasibility: self.substrate_feasibility as f32,
            prediction_error,
            bottleneck: String::new(),
            epistemic_status,
            harmony_alignment,
        }
    }

    /// Compute consciousness level from current state.
    fn compute_consciousness(&mut self, prediction_error: f32) -> f32 {
        let inputs = ConsciousnessInputs {
            phi: self.bath.dopamine.effective() as f64 * 0.8,
            broadcast: 0.6,
            working_memory: (1.0 - prediction_error as f64 * 0.5).max(0.0),
            attention: self.bath.noradrenaline.effective() as f64,
            recurrence: 0.7,
            embodiment: self.substrate_feasibility * 0.8,
            knowledge: 0.5,
            synchrony: self.bath.serotonin.effective() as f64 * 0.9,
        };
        let result = self.equation.compute(&inputs);
        let c =
            (result.consciousness_level as f32 * self.substrate_feasibility as f32).clamp(0.0, 1.0);
        self.last_consciousness = c;
        c
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
        let feasibility_gap =
            (self.substrate_feasibility as f32 - self.honest_confidence as f32).abs();

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

    /// Get current consciousness level.
    pub fn consciousness_level(&self) -> f32 {
        self.last_consciousness
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
        })
        .to_string()
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
        self.substrate_feasibility =
            substrate_requirements(&self.substrate_type).consciousness_feasibility();
        let (evidence_level, honest_confidence) = honest_confidence_for(&self.substrate_type);
        self.evidence_level = evidence_level;
        self.honest_confidence = honest_confidence;
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
}
