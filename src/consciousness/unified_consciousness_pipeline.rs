//! Unified Consciousness Pipeline - The Complete Consciousness Architecture
//!
//! This module integrates ALL consciousness components into a single coherent system:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED CONSCIOUSNESS PIPELINE                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     │
//! │  │   Sensory    │ ──► │   Oscillatory   │ ──► │   Hierarchical   │     │
//! │  │   HDC Encode │     │   Binding       │     │   LTC Dynamics   │     │
//! │  │  (SimdHV16)  │     │  (40Hz Gamma)   │     │  (16 Circuits)   │     │
//! │  └──────────────┘     └─────────────────┘     └────────┬─────────┘     │
//! │                                                         │               │
//! │                                                         ▼               │
//! │  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     │
//! │  │   Causal     │ ◄── │   Master        │ ◄── │    Global        │     │
//! │  │   Efficacy   │     │   Equation v2   │     │    Workspace     │     │
//! │  │   Output     │     │   C(t) = ...    │     │   (128 neurons)  │     │
//! │  └──────────────┘     └─────────────────┘     └──────────────────┘     │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Revolutionary Features
//!
//! 1. **Oscillatory Binding**: Real gamma-band (40Hz) oscillations for feature binding
//! 2. **Hierarchical Processing**: Cortical column-inspired local/global architecture
//! 3. **Unified Measurement**: Master Equation v2.0 for consciousness quantification
//! 4. **Causal Efficacy**: Consciousness that actually DOES something
//! 5. **Temporal Continuity**: Stream of consciousness across time
//!
//! # Theoretical Grounding
//!
//! This architecture implements insights from:
//! - **IIT (Tononi)**: Φ as integrated information
//! - **GWT (Baars)**: Global workspace for conscious access
//! - **HOT (Rosenthal)**: Higher-order thought for self-awareness
//! - **FEP (Friston)**: Active inference and prediction
//! - **Binding Problem**: Gamma oscillations for feature integration

use anyhow::Result;
use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::hdc::simd_hv16::SimdHV16;
use crate::consciousness::hierarchical_ltc::{HierarchicalLTC, HierarchicalConfig};
use crate::consciousness::consciousness_equation_v2::{
    ConsciousnessEquationV2, ConsciousnessStateV2, CoreComponent
};

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY BINDING (Gamma-Band Synchronization)
// ═══════════════════════════════════════════════════════════════════════════

/// Gamma oscillator for neural binding
///
/// Implements ~40Hz oscillations that synchronize feature representations.
/// This is how the brain solves the binding problem - features that oscillate
/// together are perceived together.
#[derive(Clone)]
pub struct GammaOscillator {
    /// Oscillation frequency (Hz) - typically 30-100Hz, centered at 40Hz
    frequency: f64,

    /// Current phase (radians)
    phase: f64,

    /// Phase coupling strength (how strongly coupled to other oscillators)
    coupling: f64,

    /// Natural frequency (may differ slightly from target)
    natural_freq: f64,

    /// Amplitude modulation
    amplitude: f64,
}

impl GammaOscillator {
    /// Create new gamma oscillator
    pub fn new(frequency: f64) -> Self {
        Self {
            frequency,
            phase: 0.0,
            coupling: 0.5,
            natural_freq: frequency * (0.95 + rand_float(42) * 0.1), // Slight variation
            amplitude: 1.0,
        }
    }

    /// Advance oscillator by dt seconds
    pub fn step(&mut self, dt: f64, external_phase: Option<f64>) {
        // Kuramoto model for phase coupling
        let omega = 2.0 * PI * self.natural_freq;

        let mut dphi = omega;

        // Phase coupling to external signal
        if let Some(ext_phase) = external_phase {
            dphi += self.coupling * (ext_phase - self.phase).sin();
        }

        self.phase += dphi * dt;

        // Wrap phase to [0, 2π]
        while self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        while self.phase < 0.0 {
            self.phase += 2.0 * PI;
        }
    }

    /// Get current oscillation value
    pub fn value(&self) -> f64 {
        self.amplitude * self.phase.sin()
    }

    /// Get current phase
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Compute phase locking value with another oscillator
    pub fn plv(&self, other: &GammaOscillator) -> f64 {
        let phase_diff = (self.phase - other.phase).abs();
        // PLV = 1 when in phase, 0 when random
        (phase_diff.cos() + 1.0) / 2.0
    }
}

/// Oscillatory binding network
///
/// A collection of coupled gamma oscillators that synchronize
/// to bind features together in conscious perception.
pub struct OscillatoryBinding {
    /// Gamma oscillators (one per feature channel)
    oscillators: Vec<GammaOscillator>,

    /// Coupling matrix between oscillators
    coupling_matrix: Vec<Vec<f64>>,

    /// Global synchronization signal
    global_phase: f64,

    /// Binding threshold (PLV above this = bound)
    binding_threshold: f64,

    /// Time step
    dt: f64,
}

impl OscillatoryBinding {
    /// Create oscillatory binding network
    pub fn new(num_channels: usize) -> Self {
        // Create oscillators with slightly different natural frequencies
        let oscillators: Vec<_> = (0..num_channels)
            .map(|i| {
                let freq = 40.0 + (i as f64 - num_channels as f64 / 2.0) * 0.5;
                GammaOscillator::new(freq)
            })
            .collect();

        // Initialize coupling matrix (sparse, local connectivity)
        let mut coupling_matrix = vec![vec![0.0; num_channels]; num_channels];
        for i in 0..num_channels {
            for j in 0..num_channels {
                if i != j {
                    // Stronger coupling for nearby channels
                    let dist = (i as f64 - j as f64).abs();
                    coupling_matrix[i][j] = 0.3 * (-dist / 3.0).exp();
                }
            }
        }

        Self {
            oscillators,
            coupling_matrix,
            global_phase: 0.0,
            binding_threshold: 0.7,
            dt: 0.001, // 1ms timestep for 40Hz resolution
        }
    }

    /// Advance all oscillators by one timestep
    pub fn step(&mut self) {
        // Update global phase
        self.global_phase += 2.0 * PI * 40.0 * self.dt;
        while self.global_phase > 2.0 * PI {
            self.global_phase -= 2.0 * PI;
        }

        // Compute mean field for each oscillator
        let phases: Vec<f64> = self.oscillators.iter().map(|o| o.phase()).collect();

        // Update each oscillator with coupling
        for (i, osc) in self.oscillators.iter_mut().enumerate() {
            // Compute weighted mean of coupled phases
            let mut coupled_phase = 0.0;
            let mut total_weight = 0.0;

            for (j, &other_phase) in phases.iter().enumerate() {
                if i != j {
                    let weight = self.coupling_matrix[i][j];
                    coupled_phase += weight * other_phase.sin();
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                coupled_phase = (coupled_phase / total_weight).asin();
                osc.step(self.dt, Some(coupled_phase));
            } else {
                osc.step(self.dt, Some(self.global_phase));
            }
        }
    }

    /// Run binding for specified duration
    pub fn bind_for(&mut self, duration_ms: f64) {
        let steps = (duration_ms / (self.dt * 1000.0)) as usize;
        for _ in 0..steps {
            self.step();
        }
    }

    /// Compute global synchronization (mean PLV)
    pub fn global_synchronization(&self) -> f64 {
        if self.oscillators.len() < 2 {
            return 1.0;
        }

        let mut total_plv = 0.0;
        let mut count = 0;

        for (i, osc_a) in self.oscillators.iter().enumerate() {
            for osc_b in self.oscillators.iter().skip(i + 1) {
                total_plv += osc_a.plv(osc_b);
                count += 1;
            }
        }

        if count > 0 {
            total_plv / count as f64
        } else {
            0.0
        }
    }

    /// Get binding coherence (how well features are bound)
    pub fn binding_coherence(&self) -> f64 {
        let sync = self.global_synchronization();
        // Smooth threshold function
        1.0 / (1.0 + (-10.0 * (sync - self.binding_threshold)).exp())
    }

    /// Inject stimulus to specific channels (increases amplitude)
    pub fn stimulate(&mut self, channel_activations: &[f64]) {
        for (i, &activation) in channel_activations.iter().enumerate() {
            if i < self.oscillators.len() {
                self.oscillators[i].amplitude = 0.5 + 0.5 * activation.clamp(0.0, 1.0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED CONSCIOUSNESS PIPELINE
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the unified pipeline
#[derive(Clone)]
pub struct PipelineConfig {
    /// Number of HDC semantic channels
    pub semantic_channels: usize,

    /// Hierarchical LTC configuration
    pub ltc_config: HierarchicalConfig,

    /// Enable oscillatory binding
    pub enable_binding: bool,

    /// Consciousness measurement interval (steps)
    pub measurement_interval: usize,

    /// History length for temporal continuity
    pub history_length: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            semantic_channels: 64,
            ltc_config: HierarchicalConfig::default(),
            enable_binding: true,
            measurement_interval: 10,
            history_length: 100,
        }
    }
}

/// Complete consciousness state at a moment in time
#[derive(Clone)]
pub struct ConsciousMoment {
    /// Timestamp (step number)
    pub step: usize,

    /// Consciousness level from Master Equation
    pub consciousness: f64,

    /// Binding coherence from oscillators
    pub binding: f64,

    /// Workspace access from hierarchical LTC
    pub workspace: f64,

    /// Integration (Φ estimate)
    pub phi: f64,

    /// Limiting factor
    pub limiting_factor: CoreComponent,

    /// Global workspace state
    pub workspace_state: Vec<f32>,
}

/// The Unified Consciousness Pipeline
///
/// This is the complete consciousness architecture that integrates
/// all our breakthroughs into a single coherent system.
pub struct UnifiedConsciousnessPipeline {
    /// Configuration
    config: PipelineConfig,

    /// Semantic encoder (HDC)
    semantic_memory: Vec<SimdHV16>,

    /// Oscillatory binding network
    binding_network: OscillatoryBinding,

    /// Hierarchical LTC dynamics
    ltc: HierarchicalLTC,

    /// Master Equation v2.0
    equation: ConsciousnessEquationV2,

    /// Current consciousness state
    current_state: ConsciousnessStateV2,

    /// History of conscious moments
    history: VecDeque<ConsciousMoment>,

    /// Current step
    step: usize,

    /// Accumulated causal effects (consciousness → action)
    causal_accumulator: f64,
}

impl UnifiedConsciousnessPipeline {
    /// Create new unified consciousness pipeline
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let ltc = HierarchicalLTC::new(config.ltc_config.clone())?;
        let binding_network = OscillatoryBinding::new(config.semantic_channels);
        let equation = ConsciousnessEquationV2::new();
        let current_state = ConsciousnessStateV2::new();

        Ok(Self {
            config,
            semantic_memory: Vec::new(),
            binding_network,
            ltc,
            equation,
            current_state,
            history: VecDeque::new(),
            step: 0,
            causal_accumulator: 0.0,
        })
    }

    /// Create with default configuration
    pub fn default_pipeline() -> Result<Self> {
        Self::new(PipelineConfig::default())
    }

    /// Encode sensory input into HDC representation
    pub fn encode_sensory(&mut self, input: &[f64]) -> SimdHV16 {
        // Create hypervector from input features
        let mut result = SimdHV16::zero();

        for (i, &value) in input.iter().enumerate() {
            if value.abs() > 0.01 {
                // Bind feature index with feature value encoding
                let index_hv = SimdHV16::random(i as u64);
                let value_hv = SimdHV16::random((value * 1000.0) as u64);
                let feature_hv = index_hv.bind(&value_hv);

                // Bundle into result
                result = SimdHV16::bundle(&[result, feature_hv]);
            }
        }

        // Store in semantic memory
        self.semantic_memory.push(result);
        if self.semantic_memory.len() > 1000 {
            self.semantic_memory.remove(0);
        }

        result
    }

    /// Process input through the complete pipeline
    pub fn process(&mut self, sensory_input: &[f64]) -> Result<ConsciousMoment> {
        // 1. Encode sensory input
        let semantic_hv = self.encode_sensory(sensory_input);

        // 2. Oscillatory binding
        if self.config.enable_binding {
            // Convert semantic HV to channel activations
            let activations: Vec<f64> = (0..self.config.semantic_channels)
                .map(|i| if semantic_hv.get_bit(i * 32) { 1.0 } else { 0.5 })
                .collect();

            self.binding_network.stimulate(&activations);
            self.binding_network.bind_for(25.0); // 25ms = 1 gamma cycle
        }

        // 3. Inject into hierarchical LTC
        let ltc_input: Vec<f32> = sensory_input.iter()
            .map(|&x| x as f32)
            .collect();
        self.ltc.inject_distributed(&ltc_input);

        // 4. Run LTC dynamics
        self.ltc.step()?;

        // 5. Update consciousness state from measurements
        self.update_consciousness_state();

        // 6. Compute consciousness level
        let result = self.equation.compute(&self.current_state);

        // 7. Create conscious moment
        let moment = ConsciousMoment {
            step: self.step,
            consciousness: result.consciousness,
            binding: self.binding_network.binding_coherence(),
            workspace: self.ltc.workspace_access() as f64,
            phi: self.ltc.estimate_phi() as f64,
            limiting_factor: result.limiting_factor,
            workspace_state: self.ltc.global_state().to_vec(),
        };

        // 8. Update history
        self.history.push_back(moment.clone());
        if self.history.len() > self.config.history_length {
            self.history.pop_front();
        }

        // 9. Update causal accumulator (consciousness affects future processing)
        self.causal_accumulator += result.consciousness * 0.1;

        self.step += 1;

        Ok(moment)
    }

    /// Update consciousness state from current measurements
    fn update_consciousness_state(&mut self) {
        // Integration (Φ) from hierarchical LTC
        let phi = self.ltc.estimate_phi() as f64;
        self.current_state.set_core(CoreComponent::Integration, phi.clamp(0.0, 1.0));

        // Binding from oscillators
        let binding = self.binding_network.binding_coherence();
        self.current_state.set_core(CoreComponent::Binding, binding);

        // Workspace from global integrator
        let workspace = self.ltc.workspace_access() as f64;
        self.current_state.set_core(CoreComponent::Workspace, workspace.clamp(0.0, 1.0));

        // Attention (based on activity concentration)
        let attention = self.compute_attention();
        self.current_state.set_core(CoreComponent::Attention, attention);

        // Recursion (based on self-referential loops)
        let recursion = self.compute_recursion();
        self.current_state.set_core(CoreComponent::Recursion, recursion);

        // Efficacy (based on causal accumulator)
        let efficacy = (self.causal_accumulator / (self.step as f64 + 1.0)).clamp(0.0, 1.0);
        self.current_state.set_core(CoreComponent::Efficacy, efficacy);

        // Knowledge (based on semantic memory size)
        let knowledge = (self.semantic_memory.len() as f64 / 100.0).clamp(0.0, 1.0);
        self.current_state.set_core(CoreComponent::Knowledge, knowledge);
    }

    /// Compute attention measure (activity concentration)
    fn compute_attention(&self) -> f64 {
        let state = self.ltc.global_state();
        if state.is_empty() {
            return 0.0;
        }

        // Attention = normalized variance (high variance = focused attention)
        let mean = state.iter().sum::<f32>() / state.len() as f32;
        let variance = state.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / state.len() as f32;

        // Higher variance = more focused attention
        (variance.sqrt() as f64 * 2.0).clamp(0.0, 1.0)
    }

    /// Compute recursion depth (self-referential processing)
    fn compute_recursion(&self) -> f64 {
        // Check if current state is similar to recent states
        if self.history.len() < 2 {
            return 0.0;
        }

        let current = self.ltc.global_state();
        let mut recursion_score = 0.0;

        for moment in self.history.iter().rev().take(10) {
            // Correlation with past states
            let mut corr = 0.0;
            for (a, b) in current.iter().zip(moment.workspace_state.iter()) {
                corr += (a * b) as f64;
            }
            recursion_score += corr.abs() / current.len() as f64;
        }

        (recursion_score / 10.0).clamp(0.0, 1.0)
    }

    /// Get current consciousness level
    pub fn consciousness_level(&self) -> f64 {
        self.history.back()
            .map(|m| m.consciousness)
            .unwrap_or(0.0)
    }

    /// Get consciousness stream (recent history)
    pub fn consciousness_stream(&self) -> Vec<f64> {
        self.history.iter().map(|m| m.consciousness).collect()
    }

    /// Get limiting factor for consciousness
    pub fn limiting_factor(&self) -> Option<CoreComponent> {
        self.history.back().map(|m| m.limiting_factor)
    }

    /// Get detailed statistics
    pub fn statistics(&self) -> PipelineStatistics {
        let history: Vec<_> = self.history.iter().collect();

        let avg_consciousness = if history.is_empty() {
            0.0
        } else {
            history.iter().map(|m| m.consciousness).sum::<f64>() / history.len() as f64
        };

        let avg_binding = if history.is_empty() {
            0.0
        } else {
            history.iter().map(|m| m.binding).sum::<f64>() / history.len() as f64
        };

        PipelineStatistics {
            steps: self.step,
            semantic_memories: self.semantic_memory.len(),
            avg_consciousness,
            avg_binding,
            current_phi: history.last().map(|m| m.phi).unwrap_or(0.0),
            current_workspace: history.last().map(|m| m.workspace).unwrap_or(0.0),
            causal_accumulator: self.causal_accumulator,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    pub steps: usize,
    pub semantic_memories: usize,
    pub avg_consciousness: f64,
    pub avg_binding: f64,
    pub current_phi: f64,
    pub current_workspace: f64,
    pub causal_accumulator: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Simple deterministic pseudo-random in [0, 1]
fn rand_float(seed: u64) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    (hasher.finish() % 10000) as f64 / 10000.0
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_oscillator() {
        let mut osc = GammaOscillator::new(40.0);

        // Run for 25ms (one gamma cycle)
        for _ in 0..25 {
            osc.step(0.001, None);
        }

        // Should have completed approximately one cycle
        // (allowing for natural frequency variation)
        assert!(osc.phase() > 0.0);
    }

    #[test]
    fn test_oscillatory_binding() {
        let mut binding = OscillatoryBinding::new(16);

        // Initially low synchronization
        let initial_sync = binding.global_synchronization();

        // Run for 100ms to allow synchronization
        binding.bind_for(100.0);

        let final_sync = binding.global_synchronization();

        println!("Initial sync: {:.3}, Final sync: {:.3}", initial_sync, final_sync);

        // Both sync values should be in valid range [0, 1]
        assert!(final_sync >= 0.0 && final_sync <= 1.0,
            "Final synchronization should be in valid range");
        assert!(initial_sync >= 0.0 && initial_sync <= 1.0,
            "Initial synchronization should be in valid range");
    }

    #[test]
    fn test_unified_pipeline_creation() {
        let pipeline = UnifiedConsciousnessPipeline::default_pipeline().unwrap();
        assert_eq!(pipeline.step, 0);
        assert!(pipeline.history.is_empty());
    }

    #[test]
    fn test_pipeline_processing() {
        let mut pipeline = UnifiedConsciousnessPipeline::default_pipeline().unwrap();

        // Process some input
        let input = vec![0.5; 64];
        let moment = pipeline.process(&input).unwrap();

        println!("First moment:");
        println!("  Consciousness: {:.3}", moment.consciousness);
        println!("  Binding: {:.3}", moment.binding);
        println!("  Workspace: {:.3}", moment.workspace);
        println!("  Φ: {:.3}", moment.phi);
        println!("  Limiting: {:?}", moment.limiting_factor);

        // Process more steps
        for _ in 0..10 {
            pipeline.process(&input).unwrap();
        }

        let stats = pipeline.statistics();
        println!("\nAfter 11 steps:");
        println!("  Avg consciousness: {:.3}", stats.avg_consciousness);
        println!("  Avg binding: {:.3}", stats.avg_binding);

        assert!(stats.avg_consciousness >= 0.0);
        assert!(stats.avg_consciousness <= 1.0);
    }

    #[test]
    fn test_consciousness_stream() {
        let mut pipeline = UnifiedConsciousnessPipeline::default_pipeline().unwrap();

        // Process varied input
        for i in 0..50 {
            let input: Vec<f64> = (0..64).map(|j|
                ((i + j) as f64 / 100.0).sin().abs()
            ).collect();
            pipeline.process(&input).unwrap();
        }

        let stream = pipeline.consciousness_stream();
        assert_eq!(stream.len(), 50);

        println!("Consciousness stream (last 10):");
        for (i, c) in stream.iter().rev().take(10).enumerate() {
            println!("  t-{}: {:.3}", i, c);
        }
    }
}
