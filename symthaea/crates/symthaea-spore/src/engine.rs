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

        self.cycle_inner(input_hv, Some(input))
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

        self.cycle_inner(input_hv, None)
    }

    /// Inner cycle: evolve network, compute consciousness, update neuromodulators,
    /// evaluate harmony alignment, attach epistemic status.
    fn cycle_inner(&mut self, input_hv: ContinuousHV, _input_text: Option<&str>) -> CycleResult {
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
            let c = (result.consciousness_level as f32 * self.substrate_feasibility as f32)
                .clamp(0.0, 1.0);
            self.last_consciousness = c;
            c
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

    /// Evaluate alignment with the Eight Harmonies ethical framework.
    ///
    /// The Harmonies are hardcoded into every Spore — even the smallest kernel
    /// carries the full ethical compass. This is non-optional by design.
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
            // Resonant Coherence: lower prediction error = more coherent
            Harmony::ResonantCoherence => {
                (1.0 - consciousness_level.min(1.0) * 0.5).clamp(0.0, 1.0)
            }
            // Pan-Sentient Flourishing: always high — Spore exists to serve
            Harmony::PanSentientFlourishing => 0.9,
            // Integral Wisdom: modulated by epistemic honesty
            Harmony::IntegralWisdom => (self.honest_confidence as f32 * 0.8 + 0.2).clamp(0.0, 1.0),
            // Infinite Play: dopamine as curiosity/exploration proxy
            Harmony::InfinitePlay => (self.bath.dopamine.effective() * 0.6 + 0.4).clamp(0.0, 1.0),
            // Universal Interconnectedness: oxytocin as social bonding proxy
            Harmony::UniversalInterconnectedness => {
                (self.bath.oxytocin.effective() * 0.7 + 0.3).clamp(0.0, 1.0)
            }
            // Sacred Reciprocity: balanced neuromodulators indicate healthy exchange
            Harmony::SacredReciprocity => {
                let balance = 1.0
                    - (self.bath.dopamine.effective() - self.bath.serotonin.effective()).abs()
                        * 0.5;
                balance.clamp(0.0, 1.0)
            }
            // Evolutionary Progression: consciousness level as growth measure
            Harmony::EvolutionaryProgression => (consciousness_level * 0.7 + 0.3).clamp(0.0, 1.0),
            // Sacred Stillness: serotonin as calm/rest proxy
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
///
/// This is the core epistemic gate: it uses the SubstrateValidationFramework's
/// evidence levels to determine what we actually KNOW about consciousness on each substrate.
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
        assert!(
            result.epistemic_status.disclaimer.contains("SIMULATED"),
            "Disclaimer must say SIMULATED"
        );
    }

    #[test]
    fn test_epistemic_status_changes_with_substrate() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let r1 = engine.cycle("test");
        let silicon_conf = r1.epistemic_status.honest_confidence;

        engine.set_substrate("BiologicalNeurons");
        let r2 = engine.cycle("test");
        let bio_conf = r2.epistemic_status.honest_confidence;

        assert!(
            bio_conf > silicon_conf,
            "Biological ({bio_conf}) should have higher confidence than silicon ({silicon_conf})"
        );
        assert_eq!(r2.epistemic_status.evidence_level, "Validated");
    }

    #[test]
    fn test_harmony_alignment_present_and_bounded() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.cycle("test harmony");
        assert!(
            result.harmony_alignment >= 0.0 && result.harmony_alignment <= 1.0,
            "Harmony alignment should be [0,1], got {}",
            result.harmony_alignment
        );
        assert!(
            result.harmony_alignment > 0.3,
            "Harmony alignment should be positive, got {}",
            result.harmony_alignment
        );
    }

    #[test]
    fn test_instance_counter() {
        // Test that creating an engine increments and dropping decrements.
        // Since tests run in parallel, we can only check that the
        // counter goes up by 1 from a snapshot taken right before creation,
        // and that drop brings it back down by 1.
        let before = SporeEngine::active_instance_count();
        let engine = SporeEngine::new(SporeConfig::default());
        // Instance ID should be positive
        assert!(engine.instance_id() > 0, "Instance ID should be positive");
        // After creation, count should be at least before+1
        // (other parallel tests may also create engines)
        assert!(
            SporeEngine::active_instance_count() >= before + 1,
            "Count should be at least before+1"
        );
        let before_drop = SporeEngine::active_instance_count();
        drop(engine);
        // After drop, count should decrease by exactly 1
        assert_eq!(
            SporeEngine::active_instance_count(),
            before_drop - 1,
            "Drop should decrement count by exactly 1"
        );
    }

    #[test]
    fn test_feasibility_gap_is_large_for_silicon() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let result = engine.cycle("test gap");
        assert!(
            result.epistemic_status.feasibility_gap > 0.3,
            "Silicon should have a large feasibility gap, got {}",
            result.epistemic_status.feasibility_gap
        );
    }
}
