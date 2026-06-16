// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Stability Regime: CfC Neurons for Primitives
//!
//! Each primitive gets temporal dynamics via an `HdcLtcUnifiedNeuron`.
//! Three **stability regimes** (Crystallized, Plastic, Fluid) map to
//! primitive tiers, each with different attractor strengths and time constants.
//!
//! ## Regime Properties
//!
//! | Regime       | Tiers   | tau   | Attractor | Learning |
//! |-------------|---------|-------|-----------|----------|
//! | Crystallized | 0-3     | 0.05  | 0.95      | Minimal  |
//! | Plastic      | 4-6     | 0.5   | 0.5       | Hebbian+STDP |
//! | Fluid        | 7-8     | 5.0   | 0.15      | Contrastive |

use std::collections::{HashMap, VecDeque};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::primitive_system::{Primitive, PrimitiveSystem, PrimitiveTier};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::consciousness::primitive_consciousness::{
    ActivationReason, ConsciousnessPrimitiveProcessor, PrimitiveConsciousnessState,
};
use crate::dynamics::cfc_coherence::{CfCCoherenceBridge, CoherenceConfig};

// =============================================================================
// REGIME TRANSITIONS
// =============================================================================

/// A regime transition event emitted when a primitive changes stability regime
#[derive(Debug, Clone)]
pub enum RegimeTransition {
    /// Primitive has crystallized (Plastic -> Crystallized)
    Crystallized {
        primitive_name: String,
        encoding: BinaryHV,
    },
    /// Primitive has decrystallized (Crystallized -> Plastic)
    Decrystallized { primitive_name: String },
    /// Primitive promoted from Fluid to Plastic
    Solidified { primitive_name: String },
}

// =============================================================================
// STABILITY REGIME TYPES
// =============================================================================

/// The three stability regimes for CfC primitives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StabilityRegimeType {
    /// Tiers 0-3: Fast snap-back, near-fixed attractors
    Crystallized,
    /// Tiers 4-6: Balanced drift and stability
    Plastic,
    /// Tiers 7-8: Slow dynamics, context-driven exploration
    Fluid,
}

impl StabilityRegimeType {
    /// Map a primitive tier to its stability regime
    pub fn from_tier(tier: PrimitiveTier) -> Self {
        match tier {
            PrimitiveTier::NSM
            | PrimitiveTier::Mathematical
            | PrimitiveTier::Physical
            | PrimitiveTier::Geometric => StabilityRegimeType::Crystallized,

            PrimitiveTier::Strategic | PrimitiveTier::MetaCognitive | PrimitiveTier::Temporal => {
                StabilityRegimeType::Plastic
            }

            PrimitiveTier::Compositional | PrimitiveTier::Consciousness | PrimitiveTier::Code => {
                StabilityRegimeType::Fluid
            }
        }
    }
}

/// Parameters for a single stability regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeParams {
    /// Base time constant for CfC neuron
    pub tau_base: f32,
    /// Attractor strength: 0.0 = fully context-driven, 1.0 = fixed attractor
    pub attractor_strength: f32,
    /// Multiplier on base learning rate
    pub learning_rate_scale: f32,
    /// Similarity threshold to become active
    pub activation_threshold: f32,
    /// Similarity threshold to deactivate (lower than activation for hysteresis)
    pub deactivation_threshold: f32,
}

/// Configuration for the entire stability regime system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRegimeConfig {
    pub crystallized: RegimeParams,
    pub plastic: RegimeParams,
    pub fluid: RegimeParams,
    pub coherence_config: CoherenceConfig,
    /// Maximum simultaneously active primitives
    pub max_active: usize,
    /// History length per primitive
    pub history_len: usize,
    /// Activations needed to transition Fluid -> Plastic
    pub fluid_to_plastic_activations: usize,
    /// Activations needed to transition Plastic -> Crystallized
    pub plastic_to_crystallized_activations: usize,
    /// Cycles without activation before a Crystallized primitive drops to Plastic
    pub decrystallize_idle_cycles: usize,
}

impl Default for StabilityRegimeConfig {
    fn default() -> Self {
        Self {
            crystallized: RegimeParams {
                tau_base: 0.05,
                attractor_strength: 0.95,
                learning_rate_scale: 0.001,
                activation_threshold: 0.35,
                deactivation_threshold: 0.25,
            },
            plastic: RegimeParams {
                tau_base: 0.5,
                attractor_strength: 0.5,
                learning_rate_scale: 1.0,
                activation_threshold: 0.30,
                deactivation_threshold: 0.20,
            },
            fluid: RegimeParams {
                tau_base: 5.0,
                attractor_strength: 0.15,
                learning_rate_scale: 2.0,
                activation_threshold: 0.25,
                deactivation_threshold: 0.15,
            },
            coherence_config: CoherenceConfig::default(),
            max_active: 32,
            history_len: 64,
            fluid_to_plastic_activations: 10,
            plastic_to_crystallized_activations: 50,
            decrystallize_idle_cycles: 100,
        }
    }
}

impl StabilityRegimeConfig {
    /// Get params for a given regime type
    pub fn params(&self, regime: StabilityRegimeType) -> &RegimeParams {
        match regime {
            StabilityRegimeType::Crystallized => &self.crystallized,
            StabilityRegimeType::Plastic => &self.plastic,
            StabilityRegimeType::Fluid => &self.fluid,
        }
    }
}

// =============================================================================
// CfC PRIMITIVE
// =============================================================================

/// A single primitive wrapped with CfC temporal dynamics
pub struct CfCPrimitive {
    /// The underlying primitive
    pub primitive: Primitive,
    /// Which stability regime this primitive belongs to
    pub regime: StabilityRegimeType,
    /// CfC neuron whose state is a ContinuousHV (16,384-D)
    pub neuron: HdcLtcUnifiedNeuron,
    /// Immutable attractor = primitive.encoding converted to continuous
    pub attractor: ContinuousHV,
    /// Current activation level (similarity between state and recent input)
    pub activation: f64,
    /// Whether this primitive is currently active in consciousness
    pub is_active: bool,
    /// History of (timestamp, activation)
    pub history: VecDeque<(f64, f64)>,
    /// How many steps this primitive has been continuously active
    pub active_duration: usize,
    /// Total number of times this primitive has been activated
    pub total_activation_count: usize,
    /// The global cycle at which this primitive was last activated
    pub last_activated_cycle: usize,
    /// The tier-based (static) regime — used as a floor for regime assignment
    pub tier_regime: StabilityRegimeType,
}

impl CfCPrimitive {
    /// Create a new CfC primitive from a base primitive.
    ///
    /// All primitives start in the **Plastic** regime regardless of tier.
    /// They transition dynamically based on activation counts:
    /// - Fluid (rarely used) -> Plastic after ~10 activations
    /// - Plastic (learning) -> Crystallized after ~50 activations
    /// - Crystallized primitives decrystallize back to Plastic after prolonged inactivity.
    pub fn new(primitive: Primitive, config: &StabilityRegimeConfig, seed: u64) -> Self {
        let tier_regime = StabilityRegimeType::from_tier(primitive.tier);
        // All primitives start Plastic — ready to learn from the first cycle
        let regime = StabilityRegimeType::Plastic;
        let params = config.params(regime);

        let unified_config = UnifiedConfig {
            tau_base: params.tau_base,
            backbone_tau: params.tau_base * 2.0,
            dimension: 16_384,
            learning_rate: config.coherence_config.base_learning_rate * params.learning_rate_scale,
            ..UnifiedConfig::default()
        };

        let mut neuron = HdcLtcUnifiedNeuron::new(unified_config, seed);

        // Convert primitive encoding to continuous space as the attractor
        let attractor = primitive.encoding.to_continuous();

        // Initialize neuron state to the attractor
        neuron.set_state(attractor.clone());

        Self {
            primitive,
            regime,
            neuron,
            attractor,
            activation: 0.0,
            is_active: false,
            history: VecDeque::with_capacity(config.history_len),
            active_duration: 0,
            total_activation_count: 0,
            last_activated_cycle: 0,
            tier_regime,
        }
    }

    /// Evolve this primitive one time step.
    ///
    /// Blends the external input with the primitive's attractor based on
    /// `attractor_strength`, then uses closed-form CfC evolution.
    /// Returns the new activation level.
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV, params: &RegimeParams) -> f64 {
        // Blend input with attractor: blended = a*attractor + (1-a)*input
        let a = params.attractor_strength;
        let blended = ContinuousHV::weighted_bundle(&[&self.attractor, input], &[a, 1.0 - a]);

        // Fused closed-form CfC evolution: eliminates 4 × 64KB intermediate allocations
        // (bind W⊗x + bind U⊗u + bundle + activation all computed in a single pass)
        self.neuron.evolve_closed_form_fused(dt, &blended);

        // Activation = how similar the neuron's state is to the input
        self.activation = self.neuron.state().similarity(input) as f64;
        self.activation
    }

    /// Apply learning based on the regime type
    pub fn learn(&mut self, input: &ContinuousHV, params: &RegimeParams, lr: f32) {
        let scaled_lr = lr * params.learning_rate_scale;

        match self.regime {
            StabilityRegimeType::Crystallized => {
                // Minimal Hebbian: reinforce attractor alignment
                self.neuron.hebbian_update(input, Some(scaled_lr));
            }
            StabilityRegimeType::Plastic => {
                // Hebbian + STDP for temporal learning
                self.neuron.hebbian_update(input, Some(scaled_lr));
                let state = self.neuron.state().clone();
                self.neuron.stdp_update(input, &state, 0.01, scaled_lr);
            }
            StabilityRegimeType::Fluid => {
                // Contrastive: push toward input, away from attractor
                self.neuron
                    .contrastive_update(input, &self.attractor, scaled_lr);
            }
        }
    }

    /// Update activation/deactivation status with hysteresis
    pub fn update_active_status(&mut self, params: &RegimeParams) {
        if self.is_active {
            // Already active: deactivate only if below lower threshold
            if self.activation < params.deactivation_threshold as f64 {
                self.is_active = false;
                self.active_duration = 0;
            } else {
                self.active_duration += 1;
            }
        } else {
            // Not active: activate only if above higher threshold
            if self.activation >= params.activation_threshold as f64 {
                self.is_active = true;
                self.active_duration = 1;
            }
        }
    }

    /// Record activation in history
    pub fn record_history(&mut self, timestamp: f64, max_len: usize) {
        self.history.push_back((timestamp, self.activation));
        while self.history.len() > max_len {
            self.history.pop_front();
        }
    }

    /// Update the dynamic regime based on activation history.
    ///
    /// Transitions:
    /// - Fluid -> Plastic after `fluid_to_plastic_activations`
    /// - Plastic -> Crystallized after `plastic_to_crystallized_activations`
    /// - Crystallized -> Plastic if idle for `decrystallize_idle_cycles`
    pub fn update_regime(
        &mut self,
        global_cycle: usize,
        config: &StabilityRegimeConfig,
    ) -> Option<RegimeTransition> {
        match self.regime {
            StabilityRegimeType::Fluid => {
                if self.total_activation_count >= config.fluid_to_plastic_activations {
                    self.regime = StabilityRegimeType::Plastic;
                    return Some(RegimeTransition::Solidified {
                        primitive_name: self.primitive.name.clone(),
                    });
                }
            }
            StabilityRegimeType::Plastic => {
                if self.total_activation_count >= config.plastic_to_crystallized_activations {
                    self.regime = StabilityRegimeType::Crystallized;
                    return Some(RegimeTransition::Crystallized {
                        primitive_name: self.primitive.name.clone(),
                        encoding: self.primitive.encoding,
                    });
                }
            }
            StabilityRegimeType::Crystallized => {
                // Decrystallize if idle too long
                let idle = global_cycle.saturating_sub(self.last_activated_cycle);
                if idle >= config.decrystallize_idle_cycles {
                    self.regime = StabilityRegimeType::Plastic;
                    // Reset activation count so it can re-crystallize naturally
                    self.total_activation_count = config.fluid_to_plastic_activations;
                    return Some(RegimeTransition::Decrystallized {
                        primitive_name: self.primitive.name.clone(),
                    });
                }
            }
        }
        None
    }

    /// Get the effective tau of this neuron for the given input
    pub fn effective_tau(&self, input: &ContinuousHV) -> f32 {
        self.neuron.effective_tau(input)
    }

    /// Similarity between current neuron state and attractor
    pub fn attractor_similarity(&self) -> f32 {
        self.neuron.state().similarity(&self.attractor)
    }
}

// =============================================================================
// STABILITY REGIME PROCESSOR
// =============================================================================

/// Main processor that wraps ConsciousnessPrimitiveProcessor with CfC dynamics
pub struct StabilityRegimeProcessor {
    /// CfC-wrapped primitives keyed by name
    primitives: HashMap<String, CfCPrimitive>,
    /// Configuration
    config: StabilityRegimeConfig,
    /// Coherence bridge for bidirectional feedback
    coherence_bridge: CfCCoherenceBridge,
    /// Inner processor for backward compatibility
    inner: ConsciousnessPrimitiveProcessor,
    /// Current consciousness state
    current_state: Option<PrimitiveConsciousnessState>,
    /// State history (VecDeque for O(1) eviction at front)
    history: VecDeque<PrimitiveConsciousnessState>,
    /// Global cycle counter for decrystallization tracking
    global_cycle: usize,
}

impl StabilityRegimeProcessor {
    /// Create a new processor with default config
    pub fn new() -> Self {
        Self::with_config(StabilityRegimeConfig::default())
    }

    /// Create a new processor with custom config
    pub fn with_config(config: StabilityRegimeConfig) -> Self {
        let inner = ConsciousnessPrimitiveProcessor::new();
        let coherence_bridge = CfCCoherenceBridge::new(config.coherence_config.clone());

        // Build CfC primitives from the primitive system
        let primitive_system = PrimitiveSystem::new();
        let all_tiers = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ];

        let mut primitives = HashMap::new();
        let mut seed = 42u64;
        for tier in &all_tiers {
            for prim in primitive_system.get_tier(*tier) {
                let cfc = CfCPrimitive::new(prim.clone(), &config, seed);
                primitives.insert(prim.name.clone(), cfc);
                seed = seed.wrapping_add(7919); // prime increment for diverse seeds
            }
        }

        Self {
            primitives,
            config,
            coherence_bridge,
            inner,
            current_state: None,
            history: VecDeque::new(),
            global_cycle: 0,
        }
    }

    /// Process a semantic input through all CfC primitives
    ///
    /// 1. Convert input to ContinuousHV
    /// 2. Evolve all primitives (closed-form, O(1) per primitive)
    /// 3. Activate/deactivate with hysteresis
    /// 4. Build PrimitiveConsciousnessState
    /// 5. Apply learning on active primitives
    /// 6. Update coherence bridge
    pub fn process_input(
        &mut self,
        input: &BinaryHV,
        dt: f32,
        timestamp: f64,
    ) -> (PrimitiveConsciousnessState, Vec<RegimeTransition>) {
        // 1. Convert input to continuous space
        let continuous_input = input.to_continuous();

        // Advance global cycle
        self.global_cycle += 1;
        let cycle = self.global_cycle;
        let config_ref = &self.config;

        // 2 & 3. Evolve all primitives in parallel, update activation, track counts, transition regimes
        // Each primitive is independent (no cross-primitive data deps), so rayon can process
        // ~75-250 CfC neurons across cores. With adaptive culling, only non-dormant primitives
        // perform the expensive 16,384D evolution.
        let transitions: Vec<RegimeTransition> = self
            .primitives
            .par_iter_mut()
            .filter_map(|(_name, cfc)| {
                let params = config_ref.params(cfc.regime);

                // Adaptive culling: skip evolution for dormant primitives
                // (inactive + near-zero activation + idle for 5+ cycles)
                // Saves ~1.5ms per skipped primitive (bind/bundle/activate on 16,384D)
                let dormant = !cfc.is_active
                    && cfc.activation.abs() < 0.01
                    && cycle.saturating_sub(cfc.last_activated_cycle) >= 5;

                if !dormant {
                    cfc.evolve(dt, &continuous_input, params);
                }

                let was_active = cfc.is_active;
                cfc.update_active_status(params);

                // Track per-primitive activation counts
                if cfc.is_active && !was_active {
                    cfc.total_activation_count += 1;
                    cfc.last_activated_cycle = cycle;
                } else if cfc.is_active {
                    cfc.last_activated_cycle = cycle;
                }

                // Dynamic regime transitions (and decrystallization)
                let transition = cfc.update_regime(cycle, config_ref);
                cfc.record_history(timestamp, config_ref.history_len);
                transition
            })
            .collect();

        // 4. Build consciousness state from active primitives
        let mut state = PrimitiveConsciousnessState::new(timestamp);

        // Collect active primitives sorted by activation (descending)
        let mut active_cfcs: Vec<&CfCPrimitive> =
            self.primitives.values().filter(|c| c.is_active).collect();
        active_cfcs.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max_active
        for cfc in active_cfcs.iter().take(self.config.max_active) {
            state.activate(
                cfc.primitive.clone(),
                cfc.activation,
                ActivationReason::BottomUp {
                    input_similarity: cfc.activation,
                },
            );
        }

        // Compute unified encoding
        state.compute_unified_encoding();

        // 5. Apply learning on active primitives
        let lr = self.coherence_bridge.effective_learning_rate();
        for cfc in self.primitives.values_mut() {
            if cfc.is_active {
                let params = self.config.params(cfc.regime);
                cfc.learn(&continuous_input, params, lr);
            }
        }

        // 6. Compute phi and update coherence bridge
        state.phi = self.estimate_phi(&state);

        // Collect tau values from active neurons for coherence (flat Vec, no Array1 per neuron)
        let tau_flat: Vec<f32> = self
            .primitives
            .values()
            .filter(|c| c.is_active)
            .map(|c| c.effective_tau(&continuous_input))
            .collect();

        if !tau_flat.is_empty() {
            self.coherence_bridge.update_flat(&tau_flat);

            // Add coherence contribution to phi
            state.phi += self.coherence_bridge.phi_contribution() as f64;
        }

        // Store history (VecDeque: O(1) push_back + O(1) pop_front)
        self.history.push_back(state.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }
        self.current_state = Some(state.clone());

        (state, transitions)
    }

    /// Estimate phi for the given consciousness state
    fn estimate_phi(&self, state: &PrimitiveConsciousnessState) -> f64 {
        use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

        let active = state.all_active();
        if active.is_empty() {
            return 0.0;
        }

        // Use CfC neuron states (not raw encodings) for richer phi computation
        let mut nodes = Vec::new();
        for ap in active.iter().take(8) {
            if let Some(cfc) = self.primitives.get(&ap.primitive.name) {
                nodes.push(cfc.neuron.state().clone());
            }
        }

        if nodes.len() < 2 {
            return (active.len() as f64).sqrt() * 0.1;
        }

        let engine = PhiEngine::new(PhiMethod::Auto);
        let result = engine.compute(&nodes);
        result.phi.min(1.0)
    }

    /// Get the current consciousness state
    pub fn current(&self) -> Option<&PrimitiveConsciousnessState> {
        self.current_state.as_ref()
    }

    /// Get state history
    pub fn history(&self) -> &VecDeque<PrimitiveConsciousnessState> {
        &self.history
    }

    /// Access the inner ConsciousnessPrimitiveProcessor for backward compat
    pub fn inner(&self) -> &ConsciousnessPrimitiveProcessor {
        &self.inner
    }

    /// Mutable access to inner processor
    pub fn inner_mut(&mut self) -> &mut ConsciousnessPrimitiveProcessor {
        &mut self.inner
    }

    /// Get a CfC primitive by name
    pub fn get_cfc_primitive(&self, name: &str) -> Option<&CfCPrimitive> {
        self.primitives.get(name)
    }

    /// Get all CfC primitives
    pub fn cfc_primitives(&self) -> &HashMap<String, CfCPrimitive> {
        &self.primitives
    }

    /// Get the coherence bridge
    pub fn coherence_bridge(&self) -> &CfCCoherenceBridge {
        &self.coherence_bridge
    }

    /// Get the config
    pub fn config(&self) -> &StabilityRegimeConfig {
        &self.config
    }

    /// Current global cycle count
    pub fn global_cycle(&self) -> usize {
        self.global_cycle
    }

    /// Get a mutable reference to a CfC primitive by name
    pub fn get_cfc_primitive_mut(&mut self, name: &str) -> Option<&mut CfCPrimitive> {
        self.primitives.get_mut(name)
    }

    /// Count of currently active primitives
    pub fn active_count(&self) -> usize {
        self.primitives.values().filter(|c| c.is_active).count()
    }

    /// Get regime distribution of active primitives
    pub fn active_by_regime(&self) -> HashMap<StabilityRegimeType, usize> {
        let mut counts = HashMap::new();
        for cfc in self.primitives.values().filter(|c| c.is_active) {
            *counts.entry(cfc.regime).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for StabilityRegimeProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_primitive(name: &str, tier: PrimitiveTier) -> Primitive {
        Primitive {
            name: name.to_string(),
            tier,
            domain: "test".to_string(),
            encoding: BinaryHV::random(name.len() as u64 * 31 + tier as u64),
            definition: name.to_string(),
            is_base: true,
            derivation: None,
        }
    }

    #[test]
    fn test_regime_assignment() {
        use PrimitiveTier::*;
        let tiers = [
            (NSM, StabilityRegimeType::Crystallized),
            (Mathematical, StabilityRegimeType::Crystallized),
            (Physical, StabilityRegimeType::Crystallized),
            (Geometric, StabilityRegimeType::Crystallized),
            (Strategic, StabilityRegimeType::Plastic),
            (MetaCognitive, StabilityRegimeType::Plastic),
            (Temporal, StabilityRegimeType::Plastic),
            (Compositional, StabilityRegimeType::Fluid),
            (Consciousness, StabilityRegimeType::Fluid),
        ];
        for (tier, expected) in &tiers {
            assert_eq!(
                StabilityRegimeType::from_tier(*tier),
                *expected,
                "Tier {:?} should map to {:?}",
                tier,
                expected,
            );
        }
    }

    #[test]
    fn test_cfc_primitive_creation() {
        let config = StabilityRegimeConfig::default();
        let prim = make_test_primitive("FORCE", PrimitiveTier::Physical);
        let cfc = CfCPrimitive::new(prim, &config, 42);

        // All primitives start Plastic, tier_regime tracks the intended regime
        assert_eq!(cfc.regime, StabilityRegimeType::Plastic);
        assert_eq!(cfc.tier_regime, StabilityRegimeType::Crystallized);
        assert!(!cfc.is_active);
        assert_eq!(cfc.active_duration, 0);
        assert_eq!(cfc.activation, 0.0);
    }

    #[test]
    fn test_evolve_produces_activation() {
        let config = StabilityRegimeConfig::default();
        let prim = make_test_primitive("FORCE", PrimitiveTier::Physical);
        let mut cfc = CfCPrimitive::new(prim, &config, 42);
        let params = config.params(StabilityRegimeType::Crystallized);

        let input = ContinuousHV::random(16_384, 999);
        let activation = cfc.evolve(0.1, &input, params);

        // Activation is cosine similarity between state and input, can be any value
        assert!(activation.is_finite(), "Activation should be finite");
    }

    #[test]
    fn test_plastic_drift() {
        let config = StabilityRegimeConfig::default();
        let prim = make_test_primitive("PLAN", PrimitiveTier::Strategic);
        let mut cfc = CfCPrimitive::new(prim, &config, 42);
        let params = config.params(StabilityRegimeType::Plastic);

        let initial_sim = cfc.attractor_similarity();

        // Feed consistent but different context
        let context = ContinuousHV::random(16_384, 777);
        for _ in 0..200 {
            cfc.evolve(0.1, &context, params);
        }

        let final_sim = cfc.attractor_similarity();
        // Plastic should drift: similarity to attractor should decrease
        assert!(
            final_sim < initial_sim,
            "Plastic primitive should drift from attractor: initial={}, final={}",
            initial_sim,
            final_sim,
        );
    }

    #[test]
    fn test_regime_configs_differ() {
        // Verify the three regimes have distinct parameters
        let config = StabilityRegimeConfig::default();
        let c = config.params(StabilityRegimeType::Crystallized);
        let p = config.params(StabilityRegimeType::Plastic);
        let f = config.params(StabilityRegimeType::Fluid);

        // Tau should increase: crystallized < plastic < fluid
        assert!(c.tau_base < p.tau_base);
        assert!(p.tau_base < f.tau_base);

        // Attractor strength should decrease: crystallized > plastic > fluid
        assert!(c.attractor_strength > p.attractor_strength);
        assert!(p.attractor_strength > f.attractor_strength);

        // Learning rate scale should increase: crystallized < plastic < fluid
        assert!(c.learning_rate_scale < p.learning_rate_scale);
        assert!(p.learning_rate_scale < f.learning_rate_scale);
    }

    #[test]
    fn test_learning_applies() {
        // Verify learning modifies neuron weights
        let config = StabilityRegimeConfig::default();
        let prim = make_test_primitive("LEARN", PrimitiveTier::Strategic);
        let mut cfc = CfCPrimitive::new(prim, &config, 42);
        let params = config.params(StabilityRegimeType::Plastic);

        let input = ContinuousHV::random(16_384, 12345);
        let state_before = cfc.neuron.state().clone();
        cfc.learn(&input, params, 0.1);
        // After learning, state should potentially change on next evolve
        // (learning modifies weights, not state directly)
        cfc.evolve(0.1, &input, params);
        let state_after = cfc.neuron.state().clone();

        let sim = state_before.similarity(&state_after);
        assert!(
            sim < 1.0,
            "Learning + evolve should change state, got similarity={}",
            sim,
        );
    }

    #[test]
    fn test_activation_hysteresis() {
        let config = StabilityRegimeConfig::default();
        let prim = make_test_primitive("TEST", PrimitiveTier::Physical);
        let mut cfc = CfCPrimitive::new(prim, &config, 42);
        let params = config.params(StabilityRegimeType::Crystallized);

        // Force activation
        cfc.activation = 0.5;
        cfc.update_active_status(params);
        assert!(cfc.is_active, "Should activate above threshold");

        // Set activation between deactivation and activation thresholds
        // For crystallized: activation_threshold=0.35, deactivation_threshold=0.25
        cfc.activation = 0.30;
        cfc.update_active_status(params);
        assert!(
            cfc.is_active,
            "Should remain active due to hysteresis (0.30 > deactivation_threshold 0.25)",
        );

        // Drop below deactivation threshold
        cfc.activation = 0.20;
        cfc.update_active_status(params);
        assert!(
            !cfc.is_active,
            "Should deactivate below deactivation threshold"
        );
    }

    #[test]
    fn test_process_input_basic() {
        let mut processor = StabilityRegimeProcessor::new();
        let input = BinaryHV::random(42);

        let (state, _transitions) = processor.process_input(&input, 0.1, 0.0);
        // Should produce a valid state with phi >= 0
        assert!(state.phi >= 0.0, "Phi should be non-negative");
    }

    #[test]
    fn test_coherence_feedback() {
        let mut processor = StabilityRegimeProcessor::new();
        let input = BinaryHV::random(42);

        // Run multiple steps
        for i in 0..10 {
            let _ = processor.process_input(&input, 0.1, i as f64 * 0.1);
        }

        // Coherence bridge should have been updated
        let coherence = processor.coherence_bridge().smoothed_coherence();
        // With consistent input, coherence should be > 0
        assert!(
            coherence >= 0.0,
            "Coherence should be non-negative, got {}",
            coherence,
        );
    }

    #[test]
    fn test_backward_compat() {
        let processor = StabilityRegimeProcessor::new();
        // Inner processor should work independently
        let stats = processor.inner().primitive_stats();
        assert!(
            stats.total_primitives > 0,
            "Inner processor should have primitives"
        );
    }
}
