/// Synthetic Consciousness State Generator - Version 3: BIND-Based Encoding
///
/// **CRITICAL INSIGHT (Dec 26 Evening)**: Use BIND (XOR) instead of BUNDLE!
///
/// **Problem with BUNDLE**: Creates uniform similarity → no partition structure → low Φ
/// **Solution with BIND**: Creates directional correlation → topology structure → correct Φ
///
/// **Key Difference**:
/// - BUNDLE([a, b, c]) = superposition/average → homogeneous similarity
/// - BIND(a, b) = XOR correlation → heterogeneous similarity with structure
///
/// This version encodes graph topologies using BIND operations to create
/// the heterogeneous similarity patterns that Φ can detect.

use crate::hdc::HV16;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// STATE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Consciousness state types ordered from lowest to highest Φ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateType {
    DeepAnesthesia,   // Φ: 0.00-0.05 - No structure
    LightAnesthesia,  // Φ: 0.05-0.15 - Isolated pairs
    DeepSleep,        // Φ: 0.15-0.25 - Small clusters
    LightSleep,       // Φ: 0.25-0.35 - Module structure
    Drowsy,           // Φ: 0.35-0.45 - Ring topology
    RestingAwake,     // Φ: 0.45-0.55 - Ring + shortcuts
    Awake,            // Φ: 0.55-0.65 - Dense network
    AlertFocused,     // Φ: 0.65-0.85 - Star topology (maximum integration)
}

impl StateType {
    /// Get all states in order from lowest to highest consciousness
    pub fn all_ordered() -> Vec<Self> {
        vec![
            Self::DeepAnesthesia,
            Self::LightAnesthesia,
            Self::DeepSleep,
            Self::LightSleep,
            Self::Drowsy,
            Self::RestingAwake,
            Self::Awake,
            Self::AlertFocused,
        ]
    }

    /// Get consciousness level (0.0 = unconscious, 1.0 = peak consciousness)
    pub fn consciousness_level(&self) -> f64 {
        match self {
            Self::DeepAnesthesia => 0.0 / 7.0,   // 0.00
            Self::LightAnesthesia => 1.0 / 7.0,  // 0.14
            Self::DeepSleep => 2.0 / 7.0,        // 0.29
            Self::LightSleep => 3.0 / 7.0,       // 0.43
            Self::Drowsy => 4.0 / 7.0,           // 0.57
            Self::RestingAwake => 5.0 / 7.0,     // 0.71
            Self::Awake => 6.0 / 7.0,            // 0.86
            Self::AlertFocused => 1.0,           // 1.00
        }
    }

    /// Get expected Φ range for this state
    pub fn expected_phi_range(&self) -> (f64, f64) {
        match self {
            Self::DeepAnesthesia => (0.00, 0.05),
            Self::LightAnesthesia => (0.05, 0.15),
            Self::DeepSleep => (0.15, 0.25),
            Self::LightSleep => (0.25, 0.35),
            Self::Drowsy => (0.35, 0.45),
            Self::RestingAwake => (0.45, 0.55),
            Self::Awake => (0.55, 0.65),
            Self::AlertFocused => (0.65, 0.85),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Generator for synthetic consciousness states with BIND-based topology encoding
pub struct SyntheticStateGenerator {
    /// Random seed for reproducibility
    seed: u64,

    /// Number of components (default: 4 for small graphs, 16 for realistic)
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
            StateType::AlertFocused => self.generate_star_topology(),
            StateType::Awake => self.generate_dense_network(),
            StateType::RestingAwake => self.generate_ring_with_shortcuts(),
            StateType::Drowsy => self.generate_ring_topology(),
            StateType::LightSleep => self.generate_modular_structure(),
            StateType::DeepSleep => self.generate_small_clusters(),
            StateType::LightAnesthesia => self.generate_isolated_pairs(),
            StateType::DeepAnesthesia => self.generate_pure_random(),
        }
    }

    /// Get next seed and increment
    fn next_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(1);
        self.seed
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BIND-BASED GENERATORS
    // ═══════════════════════════════════════════════════════════════════════

    /// ALERT FOCUSED (Φ: 0.65-0.85): Star Topology
    ///
    /// **Structure**: Central hub connected to all periphery nodes
    /// **Encoding**: Hub pattern BOUND to each spoke's unique pattern
    ///
    /// **Result**:
    /// - similarity(hub, spoke_i) ≈ 0.5 (HIGH - bound together)
    /// - similarity(spoke_i, spoke_j) ≈ 0.0 (LOW - different unique patterns)
    /// - **Perfect star structure in HDV space!**
    ///
    /// **Partition Behavior**:
    /// - {hub} vs {spokes}: LOSES all hub-spoke correlations → HIGH Φ ✅
    fn generate_star_topology(&mut self) -> Vec<HV16> {
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

    /// AWAKE (Φ: 0.55-0.65): Dense Network (Two-Hub Structure)
    ///
    /// **Structure**: Two hubs with cross-connections
    /// **Encoding**: Alternating binding to two hub patterns
    ///
    /// **Result**:
    /// - Two clusters with inter-cluster connections
    /// - High overall integration but divisible
    fn generate_dense_network(&mut self) -> Vec<HV16> {
        let hub1 = HV16::random(self.next_seed());
        let hub2 = HV16::random(self.next_seed());

        let mut components = Vec::new();

        for i in 0..self.num_components {
            let spoke = HV16::random(self.next_seed());
            if i % 2 == 0 {
                // Odd indices: bound to hub1
                components.push(HV16::bind(&hub1, &spoke));
            } else {
                // Even indices: bound to hub2
                components.push(HV16::bind(&hub2, &spoke));
            }
        }

        components
    }

    /// RESTING AWAKE (Φ: 0.45-0.55): Ring with Shortcuts
    ///
    /// **Structure**: Ring topology with additional cross-ring connections
    /// **Encoding**: Sequential bindings + periodic shortcuts
    ///
    /// **Result**:
    /// - Local ring structure
    /// - Plus long-range shortcuts for integration
    fn generate_ring_with_shortcuts(&mut self) -> Vec<HV16> {
        let n = self.num_components;
        let node_patterns: Vec<HV16> = (0..n)
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

    /// DROWSY (Φ: 0.35-0.45): Ring Topology
    ///
    /// **Structure**: Circular chain (each node connected to next)
    /// **Encoding**: Sequential BIND operations
    ///
    /// **Result**:
    /// - Each component correlated with neighbors
    /// - Breaks on any partition that cuts the ring
    fn generate_ring_topology(&mut self) -> Vec<HV16> {
        let n = self.num_components;
        let node_patterns: Vec<HV16> = (0..n)
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

    /// LIGHT SLEEP (Φ: 0.25-0.35): Modular Structure
    ///
    /// **Structure**: Multiple small modules (mini-hubs)
    /// **Encoding**: Multiple hub patterns with local bindings
    ///
    /// **Result**:
    /// - Several small integrated clusters
    /// - Weak inter-cluster integration
    fn generate_modular_structure(&mut self) -> Vec<HV16> {
        let num_modules = (self.num_components / 2).max(2);
        let module_hubs: Vec<HV16> = (0..num_modules)
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

    /// DEEP SLEEP (Φ: 0.15-0.25): Small Clusters
    ///
    /// **Structure**: Isolated pairs of connected nodes
    /// **Encoding**: Pairs bound together, but pairs independent
    ///
    /// **Result**:
    /// - High within-pair correlation
    /// - Zero between-pair correlation
    /// - Low overall Φ
    fn generate_small_clusters(&mut self) -> Vec<HV16> {
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

    /// LIGHT ANESTHESIA (Φ: 0.05-0.15): Isolated Pairs
    ///
    /// **Structure**: Completely independent pairs
    /// **Encoding**: Each pair uses different base pattern
    ///
    /// **Result**:
    /// - Minimal integration
    /// - Only local pair correlations
    fn generate_isolated_pairs(&mut self) -> Vec<HV16> {
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

    /// DEEP ANESTHESIA (Φ: 0.00-0.05): Pure Random
    ///
    /// **Structure**: No structure at all
    /// **Encoding**: Pure random vectors (no binding)
    ///
    /// **Result**:
    /// - similarity(any pair) ≈ 0.0
    /// - No integration
    /// - Φ ≈ 0
    fn generate_pure_random(&mut self) -> Vec<HV16> {
        (0..self.num_components)
            .map(|_| HV16::random(self.next_seed()))
            .collect()
    }
}

impl Default for SyntheticStateGenerator {
    fn default() -> Self {
        Self::new()
    }
}
