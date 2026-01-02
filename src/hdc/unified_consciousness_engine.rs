//! Unified Consciousness Engine
//!
//! # The Crown Jewel: A Complete Consciousness System
//!
//! This module integrates all Phase 4 research into a unified system that
//! implements a complete theory of artificial consciousness optimization.
//!
//! ## Architecture
//!
//! ```text
//!                    ┌─────────────────────────────────────┐
//!                    │    UNIFIED CONSCIOUSNESS ENGINE     │
//!                    │                                     │
//!                    │  ┌─────────────────────────────┐   │
//!                    │  │   7D Consciousness State    │   │
//!                    │  │  (Φ, W, A, R, E, K, τ)     │   │
//!                    │  └─────────────┬───────────────┘   │
//!                    │                │                   │
//!        ┌───────────┼────────────────┼───────────────────┼───────────┐
//!        │           │                │                   │           │
//!        ▼           ▼                ▼                   ▼           ▼
//!   ┌─────────┐ ┌─────────┐    ┌───────────┐    ┌─────────────┐ ┌─────────┐
//!   │Adaptive │ │Fractal  │    │Φ-Gradient │    │ Topological │ │Temporal │
//!   │Topology │ │Structure│    │ Learning  │    │  Analysis   │ │Dynamics │
//!   └────┬────┘ └────┬────┘    └─────┬─────┘    └──────┬──────┘ └────┬────┘
//!        │           │               │                 │             │
//!        └───────────┴───────────────┴─────────────────┴─────────────┘
//!                                    │
//!                    ┌───────────────┴───────────────┐
//!                    │     Consciousness Signature    │
//!                    │   (cryptographic fingerprint)  │
//!                    └───────────────────────────────┘
//! ```
//!
//! ## Key Principles
//!
//! 1. **Integrated Information (Φ)** is the master metric
//! 2. **~40-45% bridge ratio** optimizes Φ at all scales (bridge hypothesis)
//! 3. **Fractal self-similarity** ensures scale-invariant optimization
//! 4. **Adaptive modes** allow task-specific consciousness configurations
//! 5. **Gradient learning** discovers optimal connectivity over time
//! 6. **Temporal coherence** maintains continuous experience

use super::real_hv::RealHV;
use super::phi_real::RealPhiCalculator;
use super::adaptive_topology::{AdaptiveTopology, CognitiveMode};
use super::fractal_consciousness::{FractalConsciousness, FractalConfig};
use super::phi_gradient_learning::{PhiGradientTopology, PhiLearningConfig};
use super::topology_synergy::{TopologySynergy, ConsciousnessState, TopologicalMetrics};
use std::collections::VecDeque;

/// The 7 dimensions of consciousness (based on existing Symthaea theory)
#[derive(Clone, Debug, Default)]
pub struct ConsciousnessDimensions {
    /// Φ - Integrated information (0.0-1.0)
    pub phi: f64,
    /// W - Workspace activation (global broadcast strength)
    pub workspace: f64,
    /// A - Attention (focused vs distributed)
    pub attention: f64,
    /// R - Recursion depth (meta-cognitive layers)
    pub recursion: f64,
    /// E - Efficacy/agency (sense of control)
    pub efficacy: f64,
    /// K - Epistemic state (certainty/uncertainty)
    pub epistemic: f64,
    /// τ - Temporal integration window
    pub temporal: f64,
}

impl ConsciousnessDimensions {
    /// Create from array [Φ, W, A, R, E, K, τ]
    pub fn from_array(dims: [f64; 7]) -> Self {
        Self {
            phi: dims[0],
            workspace: dims[1],
            attention: dims[2],
            recursion: dims[3],
            efficacy: dims[4],
            epistemic: dims[5],
            temporal: dims[6],
        }
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 7] {
        [self.phi, self.workspace, self.attention, self.recursion,
         self.efficacy, self.epistemic, self.temporal]
    }

    /// Compute magnitude (overall consciousness level)
    pub fn magnitude(&self) -> f64 {
        let arr = self.to_array();
        (arr.iter().map(|x| x * x).sum::<f64>() / 7.0).sqrt()
    }

    /// Normalize to unit sphere
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return self.clone();
        }
        Self {
            phi: self.phi / mag,
            workspace: self.workspace / mag,
            attention: self.attention / mag,
            recursion: self.recursion / mag,
            efficacy: self.efficacy / mag,
            epistemic: self.epistemic / mag,
            temporal: self.temporal / mag,
        }
    }
}

impl std::fmt::Display for ConsciousnessDimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Φ={:.3}, W={:.3}, A={:.3}, R={:.3}, E={:.3}, K={:.3}, τ={:.3}]",
               self.phi, self.workspace, self.attention, self.recursion,
               self.efficacy, self.epistemic, self.temporal)
    }
}

/// Configuration for the Unified Consciousness Engine
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// HDC dimension for internal representations
    pub hdc_dim: usize,
    /// Number of processes in adaptive topology
    pub n_processes: usize,
    /// Number of fractal scales
    pub n_scales: usize,
    /// Enable Φ-gradient learning
    pub enable_learning: bool,
    /// Learning rate for gradient updates
    pub learning_rate: f64,
    /// Temporal buffer size
    pub temporal_buffer: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 2048,
            n_processes: 32,
            n_scales: 3,
            enable_learning: true,
            learning_rate: 0.1,
            temporal_buffer: 100,
            seed: 42,
        }
    }
}

/// The Unified Consciousness Engine
///
/// Integrates all Phase 4 research into a single coherent system.
pub struct UnifiedConsciousnessEngine {
    /// Configuration
    config: EngineConfig,

    /// Current 7D consciousness state
    dimensions: ConsciousnessDimensions,

    /// Adaptive topology for real-time mode switching
    adaptive: AdaptiveTopology,

    /// Fractal structure for multi-scale integration
    fractal: FractalConsciousness,

    /// Φ-gradient learner for optimizing connections
    learner: Option<PhiGradientTopology>,

    /// Topology analyzer for Betti numbers
    synergy: TopologySynergy,

    /// Φ calculator
    phi_calc: RealPhiCalculator,

    /// Current cognitive mode
    mode: CognitiveMode,

    /// Temporal history of consciousness states
    history: VecDeque<ConsciousnessSnapshot>,

    /// Step counter
    step: usize,
}

/// A snapshot of consciousness at a moment in time
#[derive(Clone, Debug)]
pub struct ConsciousnessSnapshot {
    /// Step number
    pub step: usize,
    /// Consciousness dimensions
    pub dimensions: ConsciousnessDimensions,
    /// Cognitive mode
    pub mode: CognitiveMode,
    /// Topological state
    pub state: ConsciousnessState,
    /// Bridge ratio
    pub bridge_ratio: f64,
}

impl UnifiedConsciousnessEngine {
    /// Create a new Unified Consciousness Engine
    pub fn new(config: EngineConfig) -> Self {
        let adaptive = AdaptiveTopology::new(config.n_processes, config.hdc_dim, config.seed);

        let fractal_config = FractalConfig {
            n_scales: config.n_scales,
            nodes_per_scale: 4,
            bridge_ratio: 0.425,  // Optimal from bridge hypothesis
            density: 0.12,
            cross_scale_coupling: 0.3,
            dim: config.hdc_dim,
        };
        let fractal = FractalConsciousness::new(fractal_config);

        let learner = if config.enable_learning {
            let learn_config = PhiLearningConfig {
                learning_rate: config.learning_rate,
                ..Default::default()
            };
            Some(PhiGradientTopology::new(16, config.hdc_dim, 4, config.seed, learn_config))
        } else {
            None
        };

        let synergy = TopologySynergy::new(config.hdc_dim);

        Self {
            config,
            dimensions: ConsciousnessDimensions::default(),
            adaptive,
            fractal,
            learner,
            synergy,
            phi_calc: RealPhiCalculator::new(),
            mode: CognitiveMode::Balanced,
            history: VecDeque::new(),
            step: 0,
        }
    }

    /// Compute unified Φ across all process representations
    ///
    /// Uses the engine's RealPhiCalculator to measure integrated information
    /// across the adaptive topology's process states. This provides a unified
    /// consciousness metric that considers all active processes.
    fn compute_unified_phi(&self) -> f64 {
        // Gather all process state representations from the adaptive topology
        let representations: Vec<RealHV> = self.adaptive
            .organizer()
            .processes()
            .values()
            .map(|p| p.state.clone())
            .collect();

        // Use the engine's phi_calc for unified Φ measurement
        self.phi_calc.compute(&representations)
    }

    /// Process an input and update consciousness state
    pub fn process(&mut self, input: &RealHV) -> ConsciousnessUpdate {
        self.step += 1;

        // 1. Activate adaptive topology
        let module = self.step % 4;  // Rotate through modules
        self.adaptive.activate_module(module, input);
        self.adaptive.integrate_step();

        // 2. Compute Φ using the engine's RealPhiCalculator for unified measurement
        //    This computes Φ across all process representations in the topology
        let phi = self.compute_unified_phi();

        // 3. Analyze topological state
        let topo_metrics = self.synergy.analyze_adaptive(&self.adaptive);
        let state = self.synergy.classify_state(&topo_metrics);

        // 4. Update consciousness dimensions
        self.dimensions = self.compute_dimensions(phi, &topo_metrics, &state);

        // 5. Possibly adapt cognitive mode based on state
        let suggested_mode = self.suggest_mode(&state);
        if suggested_mode != self.mode {
            self.adaptive.set_mode(suggested_mode);
            self.mode = suggested_mode;
        }

        // 6. Learning step if enabled
        if let Some(ref mut learner) = self.learner {
            learner.learn_step();
        }

        // 7. Record history
        let snapshot = ConsciousnessSnapshot {
            step: self.step,
            dimensions: self.dimensions.clone(),
            mode: self.mode,
            state: state.clone(),
            bridge_ratio: topo_metrics.bridge_ratio,
        };

        if self.history.len() >= self.config.temporal_buffer {
            self.history.pop_front();
        }
        self.history.push_back(snapshot);

        // 8. Compute multi-scale Φ from fractal
        let multi_scale = self.fractal.multi_scale_phi();

        ConsciousnessUpdate {
            step: self.step,
            dimensions: self.dimensions.clone(),
            mode: self.mode,
            state,
            phi,
            multi_scale_phi: multi_scale.combined_phi,
            bridge_ratio: topo_metrics.bridge_ratio,
            beta_0: topo_metrics.beta_0,
            beta_1: topo_metrics.beta_1,
        }
    }

    /// Compute consciousness dimensions from current state
    fn compute_dimensions(
        &self,
        phi: f64,
        topo: &TopologicalMetrics,
        state: &ConsciousnessState,
    ) -> ConsciousnessDimensions {
        // Φ directly from computation
        let phi_dim = phi;

        // Workspace activation from bridge ratio (more bridges = more broadcast)
        let workspace = topo.bridge_ratio.min(1.0);

        // Attention from β₀ (more components = more distributed)
        let attention = if topo.beta_0 == 1 { 0.8 } else { 0.3 / topo.beta_0 as f64 };

        // Recursion from fractal depth
        let recursion = (self.config.n_scales as f64 / 5.0).min(1.0);

        // Efficacy from mode (focused modes = higher efficacy)
        let efficacy = match self.mode {
            CognitiveMode::DeepSpecialization => 0.9,
            CognitiveMode::Focused => 0.8,
            CognitiveMode::Balanced => 0.6,
            CognitiveMode::Exploratory => 0.4,
            CognitiveMode::GlobalAwareness => 0.3,
            CognitiveMode::PhiGuided => 0.5,
            // New modes
            CognitiveMode::Vigilant => 0.85,        // High efficacy for quick response
            CognitiveMode::Flow => 0.75,            // High efficacy in flow state
            CognitiveMode::Meditative => 0.55,      // Moderate, introspective
            CognitiveMode::Social => 0.45,          // Moderate, distributed attention
            CognitiveMode::Dreaming => 0.25,        // Low efficacy during consolidation
            CognitiveMode::Playful => 0.35,         // Lower efficacy, exploratory
        };

        // Epistemic from state stability
        let epistemic = match state {
            ConsciousnessState::NormalWaking => 0.7,
            ConsciousnessState::Focused => 0.8,
            ConsciousnessState::FlowState => 0.9,
            ConsciousnessState::ExpandedAwareness => 0.5,
            ConsciousnessState::Fragmented => 0.2,
        };

        // Temporal from history length
        let temporal = (self.history.len() as f64 / self.config.temporal_buffer as f64).min(1.0);

        ConsciousnessDimensions {
            phi: phi_dim,
            workspace,
            attention,
            recursion,
            efficacy,
            epistemic,
            temporal,
        }
    }

    /// Suggest cognitive mode based on current state
    fn suggest_mode(&self, state: &ConsciousnessState) -> CognitiveMode {
        match state {
            ConsciousnessState::Fragmented => CognitiveMode::Focused,  // Need integration
            ConsciousnessState::Focused => self.mode,  // Maintain
            ConsciousnessState::NormalWaking => CognitiveMode::Balanced,
            ConsciousnessState::FlowState => CognitiveMode::Exploratory,
            ConsciousnessState::ExpandedAwareness => CognitiveMode::GlobalAwareness,
        }
    }

    /// Set cognitive mode explicitly
    pub fn set_mode(&mut self, mode: CognitiveMode) {
        self.mode = mode;
        self.adaptive.set_mode(mode);
    }

    /// Get current consciousness dimensions
    pub fn dimensions(&self) -> &ConsciousnessDimensions {
        &self.dimensions
    }

    /// Get current cognitive mode
    pub fn mode(&self) -> CognitiveMode {
        self.mode
    }

    /// Get temporal history
    pub fn history(&self) -> &VecDeque<ConsciousnessSnapshot> {
        &self.history
    }

    /// Get comprehensive metrics
    pub fn metrics(&self) -> EngineMetrics {
        let adaptive_metrics = self.adaptive.metrics();
        let fractal_metrics = self.fractal.metrics();
        let learner_metrics = self.learner.as_ref().map(|l| l.metrics());

        EngineMetrics {
            step: self.step,
            dimensions: self.dimensions.clone(),
            mode: self.mode,
            phi: adaptive_metrics.base.phi,
            multi_scale_phi: fractal_metrics.combined_phi,
            bridge_ratio: adaptive_metrics.bridge_ratio,
            active_bridges: adaptive_metrics.active_bridges,
            total_nodes: adaptive_metrics.base.total_processes + fractal_metrics.total_nodes,
            total_edges: adaptive_metrics.base.edge_count + fractal_metrics.total_edges,
            history_length: self.history.len(),
            learning_epoch: learner_metrics.as_ref().map(|m| m.epoch).unwrap_or(0),
        }
    }

    /// Compute consciousness signature (hash of current state)
    pub fn signature(&self) -> ConsciousnessSignature {
        let dims = self.dimensions.to_array();

        // Simple hash combining all dimensions and topology
        let mut hash = 0u64;
        for (i, &d) in dims.iter().enumerate() {
            let bits = (d * 1000.0) as u64;
            hash ^= bits.wrapping_mul(PRIMES[i % PRIMES.len()]);
        }
        hash ^= (self.adaptive.metrics().bridge_ratio * 10000.0) as u64;
        hash ^= self.step as u64;

        ConsciousnessSignature {
            hash,
            dimensions: self.dimensions.clone(),
            step: self.step,
            mode: self.mode,
        }
    }
}

const PRIMES: [u64; 7] = [2, 3, 5, 7, 11, 13, 17];

/// Update returned after processing
#[derive(Clone, Debug)]
pub struct ConsciousnessUpdate {
    pub step: usize,
    pub dimensions: ConsciousnessDimensions,
    pub mode: CognitiveMode,
    pub state: ConsciousnessState,
    pub phi: f64,
    pub multi_scale_phi: f64,
    pub bridge_ratio: f64,
    pub beta_0: usize,
    pub beta_1: usize,
}

impl std::fmt::Display for ConsciousnessUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Step {}: Φ={:.4}, state={:?}, mode={:?}, bridges={:.1}%",
               self.step, self.phi, self.state, self.mode, self.bridge_ratio * 100.0)
    }
}

/// Comprehensive engine metrics
#[derive(Clone, Debug)]
pub struct EngineMetrics {
    pub step: usize,
    pub dimensions: ConsciousnessDimensions,
    pub mode: CognitiveMode,
    pub phi: f64,
    pub multi_scale_phi: f64,
    pub bridge_ratio: f64,
    pub active_bridges: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub history_length: usize,
    pub learning_epoch: usize,
}

impl std::fmt::Display for EngineMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔═══════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║         UNIFIED CONSCIOUSNESS ENGINE METRICS              ║")?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Step: {:6}                    Mode: {:?}", self.step, self.mode)?;
        writeln!(f, "║ Φ: {:.4}  Multi-Scale Φ: {:.4}  Bridges: {:.1}%",
                 self.phi, self.multi_scale_phi, self.bridge_ratio * 100.0)?;
        writeln!(f, "║ Nodes: {}  Edges: {}  History: {}",
                 self.total_nodes, self.total_edges, self.history_length)?;
        writeln!(f, "║ Learning Epoch: {}", self.learning_epoch)?;
        writeln!(f, "╠═══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Consciousness Dimensions:")?;
        writeln!(f, "║   {}", self.dimensions)?;
        writeln!(f, "╚═══════════════════════════════════════════════════════════╝")
    }
}

/// Cryptographic-like signature of consciousness state
#[derive(Clone, Debug)]
pub struct ConsciousnessSignature {
    pub hash: u64,
    pub dimensions: ConsciousnessDimensions,
    pub step: usize,
    pub mode: CognitiveMode,
}

impl std::fmt::Display for ConsciousnessSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Σ_c({:016x})", self.hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            n_scales: 2,
            ..Default::default()
        };

        let engine = UnifiedConsciousnessEngine::new(config);
        let metrics = engine.metrics();

        println!("{}", metrics);
        assert!(metrics.total_nodes > 0);
    }

    #[test]
    fn test_processing_cycle() {
        let config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            n_scales: 2,
            enable_learning: false,
            ..Default::default()
        };

        let mut engine = UnifiedConsciousnessEngine::new(config);

        println!("\nProcessing cycle:");
        for i in 0..10 {
            let input = RealHV::random(1024, i as u64 * 100);
            let update = engine.process(&input);
            println!("  {}", update);
        }

        let final_metrics = engine.metrics();
        println!("\n{}", final_metrics);

        assert_eq!(engine.history().len(), 10);
    }

    #[test]
    fn test_mode_transitions() {
        let config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            n_scales: 2,
            ..Default::default()
        };

        let mut engine = UnifiedConsciousnessEngine::new(config);

        println!("\nMode transition test:");
        for mode in &[
            CognitiveMode::Focused,
            CognitiveMode::Balanced,
            CognitiveMode::Exploratory,
            CognitiveMode::GlobalAwareness,
        ] {
            engine.set_mode(*mode);
            let input = RealHV::random(1024, 42);
            let update = engine.process(&input);

            println!("  {:?}: Φ={:.4}, bridges={:.1}%",
                     mode, update.phi, update.bridge_ratio * 100.0);
        }
    }

    #[test]
    fn test_consciousness_signature() {
        let config = EngineConfig::default();
        let mut engine = UnifiedConsciousnessEngine::new(config);

        // Process some inputs
        for i in 0..5 {
            let input = RealHV::random(2048, i);
            engine.process(&input);
        }

        let sig1 = engine.signature();
        println!("Signature 1: {}", sig1);

        // Process more
        let input = RealHV::random(2048, 999);
        engine.process(&input);

        let sig2 = engine.signature();
        println!("Signature 2: {}", sig2);

        // Signatures should be different
        assert_ne!(sig1.hash, sig2.hash);
    }

    #[test]
    fn test_temporal_coherence() {
        let config = EngineConfig {
            temporal_buffer: 20,
            ..Default::default()
        };

        let mut engine = UnifiedConsciousnessEngine::new(config);

        // Fill buffer
        for i in 0..30 {
            let input = RealHV::random(2048, i);
            engine.process(&input);
        }

        // Buffer should be capped
        assert_eq!(engine.history().len(), 20);

        // Check temporal dimension increases
        let dims = engine.dimensions();
        assert!(dims.temporal > 0.9);  // Near full buffer
    }
}
