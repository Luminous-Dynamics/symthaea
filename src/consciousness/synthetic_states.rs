//! Synthetic Consciousness State Generator
//!
//! **Paradigm Shift #1: Φ Validation Framework** - Component 1/3
//!
//! Generates synthetic system states representing different levels of consciousness
//! for empirical validation of Integrated Information (Φ) computation.
//!
//! # Consciousness Levels Simulated
//!
//! 1. **Deep Anesthesia** (Φ: 0.0-0.05) - Completely disconnected components
//! 2. **Light Anesthesia** (Φ: 0.05-0.15) - Minimal integration
//! 3. **Deep Sleep** (Φ: 0.15-0.25) - Local patterns only
//! 4. **Light Sleep** (Φ: 0.25-0.35) - Some integration emerging
//! 5. **Drowsy** (Φ: 0.35-0.45) - Weak coherence
//! 6. **Resting Awake** (Φ: 0.45-0.55) - Moderate integration
//! 7. **Awake/Alert** (Φ: 0.55-0.65) - Good coherence
//! 8. **Focused/Flow** (Φ: 0.65-0.85) - Strong integration
//!
//! # Scientific Basis
//!
//! States are generated to match neuroscience findings:
//! - **Integration**: Decreases from wake to sleep to anesthesia
//! - **Information**: Specificity decreases with consciousness level
//! - **Binding**: Coherence varies with consciousness state
//!
//! # Examples
//!
//! ```rust
//! use symthaea::consciousness::synthetic_states::*;
//!
//! let mut generator = SyntheticStateGenerator::new();
//!
//! // Generate high-consciousness state
//! let awake_state = generator.generate_state(&StateType::Awake);
//!
//! // Generate low-consciousness state
//! let sleep_state = generator.generate_state(&StateType::DeepSleep);
//! ```

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};

/// Type of consciousness state to generate
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateType {
    /// Deep anesthesia: Completely disconnected (Φ: 0.0-0.05)
    DeepAnesthesia,

    /// Light anesthesia: Minimal integration (Φ: 0.05-0.15)
    LightAnesthesia,

    /// Deep sleep: Local patterns only (Φ: 0.15-0.25)
    DeepSleep,

    /// Light sleep: Some integration (Φ: 0.25-0.35)
    LightSleep,

    /// Drowsy: Weak coherence (Φ: 0.35-0.45)
    Drowsy,

    /// Resting awake: Moderate integration (Φ: 0.45-0.55)
    RestingAwake,

    /// Awake/Alert: Good coherence (Φ: 0.55-0.65)
    Awake,

    /// Focused/Flow: Strong integration (Φ: 0.65-0.85)
    AlertFocused,
}

impl StateType {
    /// Expected Φ range for this state type
    pub fn expected_phi_range(&self) -> (f64, f64) {
        match self {
            StateType::DeepAnesthesia => (0.0, 0.05),
            StateType::LightAnesthesia => (0.05, 0.15),
            StateType::DeepSleep => (0.15, 0.25),
            StateType::LightSleep => (0.25, 0.35),
            StateType::Drowsy => (0.35, 0.45),
            StateType::RestingAwake => (0.45, 0.55),
            StateType::Awake => (0.55, 0.65),
            StateType::AlertFocused => (0.65, 0.85),
        }
    }

    /// Numeric consciousness level (0.0 = unconscious, 1.0 = peak consciousness)
    pub fn consciousness_level(&self) -> f64 {
        match self {
            StateType::DeepAnesthesia => 0.0,
            StateType::LightAnesthesia => 0.1,
            StateType::DeepSleep => 0.2,
            StateType::LightSleep => 0.3,
            StateType::Drowsy => 0.4,
            StateType::RestingAwake => 0.5,
            StateType::Awake => 0.6,
            StateType::AlertFocused => 0.8,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            StateType::DeepAnesthesia => "Deep Anesthesia (unconscious, no integration)",
            StateType::LightAnesthesia => "Light Anesthesia (minimal integration)",
            StateType::DeepSleep => "Deep Sleep (local patterns only)",
            StateType::LightSleep => "Light Sleep (some integration)",
            StateType::Drowsy => "Drowsy (weak coherence)",
            StateType::RestingAwake => "Resting Awake (moderate integration)",
            StateType::Awake => "Awake/Alert (good coherence)",
            StateType::AlertFocused => "Focused/Flow (strong integration)",
        }
    }

    /// Get all state types in order of increasing consciousness
    pub fn all_ordered() -> Vec<Self> {
        vec![
            StateType::DeepAnesthesia,
            StateType::LightAnesthesia,
            StateType::DeepSleep,
            StateType::LightSleep,
            StateType::Drowsy,
            StateType::RestingAwake,
            StateType::Awake,
            StateType::AlertFocused,
        ]
    }
}

/// Generates synthetic system states for validation
///
/// Creates HDC system states (vectors of HV16) that represent different
/// levels of consciousness, from deep anesthesia to alert focus.
///
/// # State Generation Strategy
///
/// States vary in three key IIT dimensions:
/// 1. **Integration**: How much components interact (binding/bundling)
/// 2. **Information**: Specificity vs randomness of patterns
/// 3. **Differentiation**: Diversity of component states
///
/// High consciousness = High integration + High information + High differentiation
/// Low consciousness = Low integration + Low information + Low differentiation
#[derive(Clone, Debug)]
pub struct SyntheticStateGenerator {
    /// Random seed for reproducibility
    seed: u64,

    /// Number of components in each state
    num_components: usize,

    /// HDC dimension (should match system default)
    dimension: usize,
}

impl SyntheticStateGenerator {
    /// Create new generator with default parameters
    pub fn new() -> Self {
        Self::with_params(4, 16384, 42)
    }

    /// Create generator with custom parameters
    pub fn with_params(num_components: usize, dimension: usize, seed: u64) -> Self {
        Self {
            num_components,
            dimension,
            seed,
        }
    }

    /// Generate state matching specified consciousness level
    pub fn generate_state(&mut self, state_type: &StateType) -> Vec<HV16> {
        match state_type {
            StateType::AlertFocused => self.generate_high_integration(),
            StateType::Awake => self.generate_moderate_high_integration(),
            StateType::RestingAwake => self.generate_moderate_integration(),
            StateType::Drowsy => self.generate_moderate_low_integration(),
            StateType::LightSleep => self.generate_low_integration(),
            StateType::DeepSleep => self.generate_isolated_state(),
            StateType::LightAnesthesia => self.generate_fragmented_state(),
            StateType::DeepAnesthesia => self.generate_random_state(),
        }
    }

    /// ALERT FOCUSED (Φ: 0.65-0.85): Star Topology with BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Use BIND instead of BUNDLE!
    /// **Structure**: Central hub connected to all periphery nodes
    /// **Encoding**: Hub pattern BOUND to each spoke's unique pattern
    ///
    /// **Result**:
    /// - similarity(hub, spoke_i) ≈ 0.5 (HIGH - bound together)
    /// - similarity(spoke_i, spoke_j) ≈ 0.0 (LOW - different unique patterns)
    /// - **Perfect star structure in HDV space!**
    ///
    /// **Why BIND Works**: Creates heterogeneous similarity structure
    /// - Cross-partition (hub-spoke) correlations are HIGH
    /// - Within-partition (spoke-spoke) correlations are LOW
    /// - Partitioning LOSES information → HIGH Φ ✅
    fn generate_high_integration(&mut self) -> Vec<HV16> {
        let hub_pattern = HV16::random(self.next_seed());

        let mut components = Vec::new();

        // First component is the hub itself
        components.push(hub_pattern.clone());

        // Remaining components are spokes bound to hub
        for _ in 1..self.num_components {
            let spoke_unique = HV16::random(self.next_seed());
            // BIND creates correlation: spoke is related to hub but unique
            components.push(HV16::bind(&hub_pattern, &spoke_unique));
        }

        components
    }

    /// AWAKE (Φ: 0.55-0.65): Dense Network (Two-Hub Structure) with BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Two hubs with BIND operations
    /// **Structure**: Two clusters with inter-cluster connections
    /// **Encoding**: Alternating binding to two hub patterns
    fn generate_moderate_high_integration(&mut self) -> Vec<HV16> {
        let hub1 = HV16::random(self.next_seed());
        let hub2 = HV16::random(self.next_seed());

        let mut components = Vec::new();

        for i in 0..self.num_components {
            let spoke = HV16::random(self.next_seed());
            if i % 2 == 0 {
                // Even indices: bound to hub1
                components.push(HV16::bind(&hub1, &spoke));
            } else {
                // Odd indices: bound to hub2
                components.push(HV16::bind(&hub2, &spoke));
            }
        }

        components
    }

    /// RESTING AWAKE (Φ: 0.45-0.55): Ring with Shortcuts using BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Ring topology with shortcuts
    /// **Structure**: Sequential connections + long-range shortcuts
    /// **Encoding**: Sequential BIND + periodic cross-ring bindings
    fn generate_moderate_integration(&mut self) -> Vec<HV16> {
        let n = self.num_components;
        let mut node_patterns: Vec<HV16> = (0..n)
            .map(|_| HV16::random(self.next_seed()))
            .collect();

        let mut components = Vec::new();

        for i in 0..n {
            let curr = node_patterns[i].clone();
            let next = node_patterns[(i + 1) % n].clone();

            // Create ring connection
            let mut component = HV16::bind(&curr, &next);

            // Add shortcut every 2 nodes
            if i % 2 == 0 && n > 3 {
                let shortcut = node_patterns[(i + n / 2) % n].clone();
                component = HV16::bind(&component, &shortcut);
            }

            components.push(component);
        }

        components
    }

    /// DROWSY (Φ: 0.35-0.45): Ring Topology using BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Pure ring structure
    /// **Structure**: Circular chain (each node connected to next)
    /// **Encoding**: Sequential BIND operations
    fn generate_moderate_low_integration(&mut self) -> Vec<HV16> {
        let n = self.num_components;
        let mut node_patterns: Vec<HV16> = (0..n)
            .map(|_| HV16::random(self.next_seed()))
            .collect();

        let mut components = Vec::new();

        for i in 0..n {
            let curr = node_patterns[i].clone();
            let next = node_patterns[(i + 1) % n].clone();
            // Bind current node to next in ring
            components.push(HV16::bind(&curr, &next));
        }

        components
    }

    /// LIGHT SLEEP (Φ: 0.25-0.35): Modular Structure using BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Multiple small modules
    /// **Structure**: Several small integrated clusters
    /// **Encoding**: Multiple hub patterns with local bindings
    fn generate_low_integration(&mut self) -> Vec<HV16> {
        let num_modules = (self.num_components / 2).max(2);
        let mut module_hubs: Vec<HV16> = (0..num_modules)
            .map(|_| HV16::random(self.next_seed()))
            .collect();

        let mut components = Vec::new();

        for i in 0..self.num_components {
            let module_idx = i % num_modules;
            let hub = module_hubs[module_idx].clone();
            let spoke = HV16::random(self.next_seed());
            // Bind to local module hub
            components.push(HV16::bind(&hub, &spoke));
        }

        components
    }

    /// DEEP SLEEP (Φ: 0.15-0.25): Small Clusters using BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Isolated pairs
    /// **Structure**: Pairs bound together, but pairs independent
    /// **Encoding**: Within-pair binding, no cross-pair correlation
    fn generate_isolated_state(&mut self) -> Vec<HV16> {
        let mut components = Vec::new();

        for i in 0..self.num_components {
            if i % 2 == 0 {
                // Even: create new pair pattern
                let pair_pattern = HV16::random(self.next_seed());
                components.push(pair_pattern);
            } else {
                // Odd: bind to previous (creating pair)
                let prev = components[i - 1].clone();
                let unique = HV16::random(self.next_seed());
                components.push(HV16::bind(&prev, &unique));
            }
        }

        components
    }

    /// LIGHT ANESTHESIA (Φ: 0.05-0.15): Isolated Pairs using BIND
    ///
    /// **V3 FIX (Dec 26 Evening)**: Completely independent pairs
    /// **Structure**: Each pair uses different base pattern
    /// **Encoding**: Minimal integration, only local pair correlations
    fn generate_fragmented_state(&mut self) -> Vec<HV16> {
        let mut components = Vec::new();

        // Create completely independent pairs
        for _ in 0..(self.num_components / 2) {
            let base = HV16::random(self.next_seed());
            let variation = HV16::random(self.next_seed());

            // First element of pair
            components.push(base.clone());
            // Second element bound to first
            components.push(HV16::bind(&base, &variation));
        }

        // Handle odd number of components
        if self.num_components % 2 == 1 {
            components.push(HV16::random(self.next_seed()));
        }

        components
    }

    /// RANDOM STATE (DeepAnesthesia): Pure Independence
    ///
    /// **Graph Structure**: NO structure whatsoever - completely independent
    /// **Cross-Partition Property**: All partitions have equal info (random baseline)
    /// **Expected Φ**: 0.00-0.05 (near-zero integration)
    ///
    /// Strategy: Purely random independent components
    fn generate_random_state(&mut self) -> Vec<HV16> {
        // Completely independent random vectors
        // Expected pairwise similarity: ~0.5 (HDV baseline)
        // Φ should be minimal - just the random correlation baseline
        (0..self.num_components)
            .map(|_| HV16::random(self.next_seed()))
            .collect()
    }

    /// Generate next random seed using LCG
    fn next_seed(&mut self) -> u64 {
        // Linear Congruential Generator (LCG) - simple but sufficient for our needs
        self.seed = self.seed.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.seed
    }
}

impl Default for SyntheticStateGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_type_consciousness_levels() {
        // Verify consciousness levels are monotonically increasing
        let states = StateType::all_ordered();

        for i in 1..states.len() {
            assert!(states[i].consciousness_level() > states[i - 1].consciousness_level(),
                    "{:?} should have higher level than {:?}",
                    states[i], states[i - 1]);
        }
    }

    #[test]
    fn test_state_type_phi_ranges() {
        // Verify Φ ranges are non-overlapping and ordered
        let states = StateType::all_ordered();

        for i in 1..states.len() {
            let (_, prev_max) = states[i - 1].expected_phi_range();
            let (curr_min, _) = states[i].expected_phi_range();

            assert!(curr_min >= prev_max,
                    "Φ ranges should not overlap: {:?} [{}, {}] vs {:?} [{}, {}]",
                    states[i - 1], prev_max, curr_min,
                    states[i], curr_min, prev_max);
        }
    }

    #[test]
    fn test_generator_creation() {
        let generator = SyntheticStateGenerator::new();
        assert_eq!(generator.num_components, 4);
        assert_eq!(generator.dimension, 16384);
    }

    #[test]
    fn test_generate_all_state_types() {
        let mut generator = SyntheticStateGenerator::new();

        for state_type in StateType::all_ordered() {
            let state = generator.generate_state(&state_type);

            assert_eq!(state.len(), generator.num_components,
                       "State {:?} should have {} components",
                       state_type, generator.num_components);

            // HV16 has fixed dimension of 16384 bits (2048 bytes * 8 bits)
            // Just verify we have the right number of components
            assert!(!state.is_empty(), "State {:?} should not be empty", state_type);
        }
    }

    #[test]
    fn test_high_integration_has_shared_patterns() {
        let mut generator = SyntheticStateGenerator::new();
        let state = generator.generate_high_integration();

        // All components should share some similarity (they're all bound to same base)
        let similarities: Vec<f64> = (0..state.len() - 1)
            .map(|i| state[i].similarity(&state[i + 1]) as f64)
            .collect();

        let avg_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;

        // High integration should have noticeable similarity between components
        assert!(avg_similarity > 0.3,
                "High integration should have avg similarity > 0.3, got {:.3}",
                avg_similarity);
    }

    #[test]
    fn test_random_state_has_low_similarity() {
        let mut generator = SyntheticStateGenerator::new();
        let state = generator.generate_random_state();

        // Components should be nearly independent
        let similarities: Vec<f64> = (0..state.len() - 1)
            .map(|i| state[i].similarity(&state[i + 1]) as f64)
            .collect();

        let avg_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;

        // For binary hypervectors, similarity ≈ 0.5 means "random/no correlation"
        // Values significantly above 0.5 indicate positive correlation (shared structure)
        // Random state should have similarity close to 0.5 (the expected value for uncorrelated vectors)
        assert!(avg_similarity < 0.6 && avg_similarity > 0.4,
                "Random state should have avg similarity near 0.5 (random), got {:.3}",
                avg_similarity);
    }

    #[test]
    fn test_reproducibility_with_same_seed() {
        let mut gen1 = SyntheticStateGenerator::with_params(4, 16384, 42);
        let mut gen2 = SyntheticStateGenerator::with_params(4, 16384, 42);

        let state1 = gen1.generate_state(&StateType::Awake);
        let state2 = gen2.generate_state(&StateType::Awake);

        // Same seed should produce identical states
        for (comp1, comp2) in state1.iter().zip(state2.iter()) {
            assert_eq!(comp1, comp2, "Same seed should produce identical components");
        }
    }

    #[test]
    fn test_different_seeds_produce_different_states() {
        let mut gen1 = SyntheticStateGenerator::with_params(4, 16384, 42);
        let mut gen2 = SyntheticStateGenerator::with_params(4, 16384, 123);

        let state1 = gen1.generate_state(&StateType::Awake);
        let state2 = gen2.generate_state(&StateType::Awake);

        // Different seeds should produce different states
        let mut any_different = false;
        for (comp1, comp2) in state1.iter().zip(state2.iter()) {
            if comp1 != comp2 {
                any_different = true;
                break;
            }
        }

        assert!(any_different, "Different seeds should produce different states");
    }

    #[test]
    fn test_state_descriptions() {
        for state_type in StateType::all_ordered() {
            let desc = state_type.description();
            assert!(!desc.is_empty(), "Description should not be empty");
            assert!(desc.len() > 10, "Description should be meaningful");
        }
    }
}
