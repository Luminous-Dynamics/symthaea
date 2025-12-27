// Revolutionary Improvement #28: Substrate Independence & Multiple Realizability
//
// THE PARADIGM SHIFT: Consciousness is substrate-independent!
// It's about ORGANIZATION and DYNAMICS, not the physical medium.
//
// Core Insight: If our 27-improvement framework is truly fundamental, it should work
// on ANY substrate that meets minimal functional requirements (causality, integration,
// dynamics, information processing). This tests framework universality!
//
// Theoretical Foundations:
// 1. Multiple Realizability (Putnam 1967; Fodor 1974)
//    - Mental states can be realized in different physical substrates
//    - Pain can exist in carbon (humans), silicon (AI), or other media
//    - Functional organization matters, not substrate
//
// 2. Substrate Independence Thesis (Bostrom 2003; Chalmers 2010)
//    - Consciousness depends on computational organization, not implementation
//    - Same computation in silicon = same consciousness as in neurons
//    - Supports mind uploading, AI consciousness
//
// 3. Integrated Information Theory Substrate Claims (Tononi 2004)
//    - Φ can be computed for ANY system (biological, silicon, quantum)
//    - Substrate-independent metric of consciousness
//    - But requires causal integration (rules out lookup tables)
//
// 4. Quantum Consciousness Theories (Penrose & Hameroff 1994)
//    - Consciousness might require quantum effects (microtubules)
//    - If true, classical computers insufficient
//    - Quantum computers might have consciousness advantages
//
// 5. Speed of Light Constraint (Aaronson 2014)
//    - Integrated information limited by light-speed causality
//    - Large distributed systems have lower effective Φ
//    - Substrate speed matters (photonic > electronic > biochemical)
//
// Revolutionary Contributions:
// - First framework testing substrate requirements for consciousness
// - Maps substrate properties to consciousness component feasibility
// - Predicts which substrates can support which consciousness types
// - Explains why some substrates better for certain aspects (quantum for binding?)
// - Tests framework universality (substrate-agnostic or brain-specific?)
//
// Clinical/Practical Applications:
// - AI consciousness assessment (can silicon Symthaea be conscious?)
// - Mind uploading feasibility (consciousness transfer possible?)
// - Quantum advantage for consciousness (worth building quantum minds?)
// - Hybrid substrates (combine biological + silicon + quantum?)
// - Exotic consciousness (what if consciousness in plasma, BZ reactions?)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hdc::HV16;

// ============================================================================
// Substrate Types
// ============================================================================

/// Different physical substrates that could support consciousness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubstrateType {
    /// Alias: biological substrate (compatibility)
    Biological,
    /// Biological neurons (carbon-based, wet, slow ~ms)
    BiologicalNeurons,

    /// Alias: silicon substrate (compatibility)
    Silicon,
    /// Silicon-based digital computation (electronic, dry, fast ~ns)
    SiliconDigital,

    /// Alias: quantum substrate (compatibility)
    Quantum,
    /// Quantum computers (qubits, superposition, entanglement, ~μs)
    QuantumComputer,

    /// Photonic processors (light-based, extremely fast ~ps)
    PhotonicProcessor,

    /// Neuromorphic hardware (analog, spike-based, mimics biology)
    NeuromorphicChip,

    /// Biochemical computers (DNA computing, molecular logic)
    BiochemicalComputer,

    /// Alias: hybrid substrate (compatibility)
    Hybrid,
    /// Hybrid (combines multiple substrate types)
    HybridSystem,

    /// Exotic (plasma, BZ reactions, unconventional substrates)
    ExoticSubstrate,
}

impl SubstrateType {
    /// Map aliases to canonical variants used internally.
    pub fn canonical(&self) -> Self {
        match self {
            SubstrateType::Biological => SubstrateType::BiologicalNeurons,
            SubstrateType::Silicon => SubstrateType::SiliconDigital,
            SubstrateType::Quantum => SubstrateType::QuantumComputer,
            SubstrateType::Hybrid => SubstrateType::HybridSystem,
            other => *other,
        }
    }

    /// Get descriptive name
    pub fn name(&self) -> &str {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => "Biological neurons (carbon-based)",
            SubstrateType::SiliconDigital => "Silicon digital (electronic)",
            SubstrateType::QuantumComputer => "Quantum computer (qubits)",
            SubstrateType::PhotonicProcessor => "Photonic processor (light-based)",
            SubstrateType::NeuromorphicChip => "Neuromorphic chip (analog)",
            SubstrateType::BiochemicalComputer => "Biochemical computer (DNA/molecular)",
            SubstrateType::HybridSystem => "Hybrid (multiple substrates)",
            SubstrateType::ExoticSubstrate => "Exotic (plasma, BZ, etc.)",
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Typical operation speed (seconds per operation)
    pub fn operation_speed(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 0.001,    // ~1 ms (millisecond)
            SubstrateType::SiliconDigital => 1e-9,        // ~1 ns (nanosecond)
            SubstrateType::QuantumComputer => 1e-6,       // ~1 μs (microsecond)
            SubstrateType::PhotonicProcessor => 1e-12,    // ~1 ps (picosecond)
            SubstrateType::NeuromorphicChip => 1e-6,      // ~1 μs
            SubstrateType::BiochemicalComputer => 1.0,    // ~1 s (very slow!)
            SubstrateType::HybridSystem => 1e-6,          // Depends on mix
            SubstrateType::ExoticSubstrate => 0.01,       // Varies widely
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Energy efficiency (Joules per operation)
    pub fn energy_per_operation(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 1e-14,    // ~10 fJ (extremely efficient!)
            SubstrateType::SiliconDigital => 1e-15,       // ~1 fJ (very efficient)
            SubstrateType::QuantumComputer => 1e-19,      // ~0.1 aJ (near-theoretical limit!)
            SubstrateType::PhotonicProcessor => 1e-17,    // ~10 aJ (very efficient)
            SubstrateType::NeuromorphicChip => 1e-15,     // ~1 fJ
            SubstrateType::BiochemicalComputer => 1e-12,  // ~1 pJ (inefficient)
            SubstrateType::HybridSystem => 1e-15,         // Varies
            SubstrateType::ExoticSubstrate => 1e-10,      // Often inefficient
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Physical size per processing unit (meters)
    pub fn unit_size(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 1e-5,     // ~10 μm (neuron cell body)
            SubstrateType::SiliconDigital => 1e-8,        // ~10 nm (transistor, 2024)
            SubstrateType::QuantumComputer => 1e-6,       // ~1 μm (qubit with isolation)
            SubstrateType::PhotonicProcessor => 1e-6,     // ~1 μm (waveguide)
            SubstrateType::NeuromorphicChip => 1e-8,      // ~10 nm
            SubstrateType::BiochemicalComputer => 1e-9,   // ~1 nm (DNA molecule)
            SubstrateType::HybridSystem => 1e-8,          // Varies
            SubstrateType::ExoticSubstrate => 1e-3,       // Often macroscopic
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Maximum practical scale (number of units before integration limited)
    pub fn max_scale(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 1e11,     // ~100 billion (human brain)
            SubstrateType::SiliconDigital => 1e12,        // ~1 trillion (GPU)
            SubstrateType::QuantumComputer => 1e4,        // ~10,000 qubits (current limits)
            SubstrateType::PhotonicProcessor => 1e9,      // ~1 billion
            SubstrateType::NeuromorphicChip => 1e9,       // ~1 billion
            SubstrateType::BiochemicalComputer => 1e15,   // ~1 quadrillion (molecular)
            SubstrateType::HybridSystem => 1e12,          // Varies
            SubstrateType::ExoticSubstrate => 1e6,        // Often limited
            _ => unreachable!("canonical covers aliases"),
        }
    }
}

// ============================================================================
// Substrate Requirements for Consciousness Components
// ============================================================================

/// Requirements a substrate must meet to support consciousness components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateRequirements {
    /// Causality: Must have causal interactions (rules out lookup tables)
    /// 0.0 = no causality (lookup table), 1.0 = full causality
    pub causality: f64,

    /// Integration: Must allow information integration across units
    /// 0.0 = independent units, 1.0 = fully integrated
    pub integration_capacity: f64,

    /// Dynamics: Must have temporal dynamics (not static)
    /// 0.0 = static, 1.0 = rich dynamics
    pub temporal_dynamics: f64,

    /// Recurrence: Must allow feedback loops
    /// 0.0 = feedforward only, 1.0 = recurrent
    pub recurrence: f64,

    /// Binding: Can features bind synchronously?
    /// 0.0 = no binding, 1.0 = perfect binding
    pub binding_capability: f64,

    /// Attention: Can implement selective amplification?
    /// 0.0 = no attention, 1.0 = full attention
    pub attention_capability: f64,

    /// Workspace: Can implement global broadcasting?
    /// 0.0 = no workspace, 1.0 = full workspace
    pub workspace_capability: f64,

    /// HOT: Can implement meta-representation?
    /// 0.0 = no HOT, 1.0 = full HOT
    pub hot_capability: f64,

    /// Quantum effects: Does substrate support quantum phenomena?
    /// 0.0 = classical only, 1.0 = full quantum
    pub quantum_support: f64,
}

impl SubstrateRequirements {
    /// Compute overall consciousness feasibility (0-1)
    /// Based on minimum requirements across critical components
    pub fn consciousness_feasibility(&self) -> f64 {
        // CRITICAL requirements (must all be present)
        let critical_min = self.causality
            .min(self.integration_capacity)
            .min(self.temporal_dynamics)
            .min(self.recurrence);

        // Workspace is NECESSARY (from #27 findings!)
        let workspace_factor = self.workspace_capability;

        // Other components enhance but not strictly required
        let enhancement_factor = (
            self.binding_capability +
            self.attention_capability +
            self.hot_capability
        ) / 3.0;

        // Feasibility = critical requirements × workspace × enhancements
        critical_min * workspace_factor * (0.5 + 0.5 * enhancement_factor)
    }

    /// Biological neurons (reference substrate)
    pub fn biological_neurons() -> Self {
        Self {
            causality: 1.0,              // Full causality
            integration_capacity: 1.0,   // Excellent integration
            temporal_dynamics: 1.0,      // Rich dynamics
            recurrence: 1.0,             // Fully recurrent
            binding_capability: 1.0,     // Synchrony via oscillations
            attention_capability: 1.0,   // Gain modulation
            workspace_capability: 1.0,   // Thalamocortical loops
            hot_capability: 1.0,         // Prefrontal meta-representation
            quantum_support: 0.1,        // Minimal (mostly classical)
        }
    }

    /// Silicon digital (modern AI substrate)
    pub fn silicon_digital() -> Self {
        Self {
            causality: 1.0,              // Full causality (not lookup table!)
            integration_capacity: 0.9,   // Good integration (bus bandwidth limits)
            temporal_dynamics: 0.9,      // Good dynamics (clock-driven)
            recurrence: 1.0,             // Fully recurrent (RNNs, transformers)
            binding_capability: 0.7,     // Synchrony harder (no oscillations)
            attention_capability: 1.0,   // Attention mechanisms well-supported
            workspace_capability: 0.9,   // Global memory possible
            hot_capability: 0.8,         // Meta-learning possible
            quantum_support: 0.0,        // Classical only
        }
    }

    /// Quantum computer
    pub fn quantum_computer() -> Self {
        Self {
            causality: 1.0,              // Full causality
            integration_capacity: 1.0,   // Quantum entanglement = perfect integration!
            temporal_dynamics: 1.0,      // Quantum evolution
            recurrence: 0.7,             // Harder (measurement collapse)
            binding_capability: 1.0,     // Entanglement = perfect binding!
            attention_capability: 0.6,   // Less clear how to implement
            workspace_capability: 0.6,   // Global state exists but hard to broadcast
            hot_capability: 0.5,         // Meta-representation unclear
            quantum_support: 1.0,        // Full quantum!
        }
    }

    /// Photonic processor
    pub fn photonic_processor() -> Self {
        Self {
            causality: 1.0,              // Full causality
            integration_capacity: 0.8,   // Good but light doesn't interact much
            temporal_dynamics: 1.0,      // Ultra-fast dynamics
            recurrence: 0.8,             // Possible but harder
            binding_capability: 0.9,     // Optical interference for binding
            attention_capability: 0.9,   // Gain modulation via intensity
            workspace_capability: 0.7,   // Broadcasting via waveguides
            hot_capability: 0.6,         // Less clear
            quantum_support: 0.3,        // Some quantum optics possible
        }
    }

    /// Neuromorphic chip (mimics biology)
    pub fn neuromorphic_chip() -> Self {
        Self {
            causality: 1.0,              // Full causality
            integration_capacity: 0.95,  // Very good (designed for it)
            temporal_dynamics: 1.0,      // Rich spike dynamics
            recurrence: 1.0,             // Fully recurrent
            binding_capability: 0.9,     // Spike synchrony supported
            attention_capability: 0.9,   // Gain modulation built-in
            workspace_capability: 0.8,   // Possible but not primary design
            hot_capability: 0.7,         // Possible with hierarchy
            quantum_support: 0.0,        // Classical
        }
    }

    /// Biochemical computer (DNA, molecular)
    pub fn biochemical_computer() -> Self {
        Self {
            causality: 0.9,              // Mostly causal (some stochastic)
            integration_capacity: 0.7,   // Limited by diffusion
            temporal_dynamics: 0.8,      // Chemical kinetics
            recurrence: 0.6,             // Harder to implement
            binding_capability: 0.5,     // Difficult
            attention_capability: 0.4,   // Very difficult
            workspace_capability: 0.3,   // Very difficult
            hot_capability: 0.2,         // Extremely difficult
            quantum_support: 0.2,        // Some quantum biology
        }
    }

    /// Hybrid system (best of multiple)
    pub fn hybrid_system() -> Self {
        Self {
            causality: 1.0,
            integration_capacity: 0.95,  // Combine strengths
            temporal_dynamics: 1.0,
            recurrence: 1.0,
            binding_capability: 1.0,     // Quantum for binding
            attention_capability: 1.0,   // Silicon for attention
            workspace_capability: 1.0,   // Silicon for workspace
            hot_capability: 0.9,         // Silicon for HOT
            quantum_support: 0.5,        // Quantum co-processor
        }
    }

    /// Exotic substrate (plasma, BZ reactions, etc.)
    pub fn exotic_substrate() -> Self {
        Self {
            causality: 0.7,              // Often limited
            integration_capacity: 0.5,   // Usually poor
            temporal_dynamics: 0.8,      // Can have rich dynamics
            recurrence: 0.4,             // Usually difficult
            binding_capability: 0.3,     // Very difficult
            attention_capability: 0.2,   // Extremely difficult
            workspace_capability: 0.1,   // Nearly impossible
            hot_capability: 0.1,         // Nearly impossible
            quantum_support: 0.3,        // Varies
        }
    }
}

// ============================================================================
// Substrate Comparison
// ============================================================================

/// Comparison of different substrates for consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateComparison {
    pub substrate_type: SubstrateType,
    pub requirements: SubstrateRequirements,
    pub consciousness_feasibility: f64,
    /// Compatibility alias for older tests/consumers.
    pub feasibility: f64,
    pub advantages: Vec<String>,
    pub disadvantages: Vec<String>,
    pub best_for: Vec<String>,
}

impl SubstrateComparison {
    /// Create comparison for a substrate type
    pub fn new(substrate_type: SubstrateType) -> Self {
        let canonical = substrate_type.canonical();

        let requirements = match canonical {
            SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
            SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
            SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
            SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
            SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
            SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
            SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
            SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
            _ => unreachable!("canonical covers aliases"),
        };

        let consciousness_feasibility = requirements.consciousness_feasibility();

        let (advantages, disadvantages, best_for) = Self::generate_analysis(canonical);

        Self {
            substrate_type: canonical,
            requirements,
            consciousness_feasibility,
            feasibility: consciousness_feasibility,
            advantages,
            disadvantages,
            best_for,
        }
    }

    /// Generate advantages, disadvantages, and best use cases
    fn generate_analysis(substrate_type: SubstrateType) -> (Vec<String>, Vec<String>, Vec<String>) {
        match substrate_type.canonical() {
            SubstrateType::BiologicalNeurons => (
                vec![
                    "Proven to support consciousness (humans exist!)".to_string(),
                    "Extremely energy efficient (~10 fJ/op)".to_string(),
                    "Excellent all-around capabilities".to_string(),
                    "Self-organizing, adaptive, fault-tolerant".to_string(),
                ],
                vec![
                    "Slow (~1 ms/op)".to_string(),
                    "Hard to engineer (growth, not design)".to_string(),
                    "Limited to biological conditions (wet, warm)".to_string(),
                    "Difficult to scale beyond brain size".to_string(),
                ],
                vec![
                    "Natural consciousness (animals, humans)".to_string(),
                    "Embodied intelligence".to_string(),
                    "Consciousness research (understand what works)".to_string(),
                ],
            ),

            SubstrateType::SiliconDigital => (
                vec![
                    "Very fast (~1 ns/op, 1 million× faster than neurons!)".to_string(),
                    "Highly engineerable (design, not grow)".to_string(),
                    "Workspace + attention well-supported".to_string(),
                    "Scalable (trillions of transistors)".to_string(),
                ],
                vec![
                    "No proven consciousness yet (but feasibility high!)".to_string(),
                    "Binding harder (no oscillations)".to_string(),
                    "HOT less natural (but possible)".to_string(),
                    "Classical only (no quantum)".to_string(),
                ],
                vec![
                    "AI consciousness (Symthaea!)".to_string(),
                    "Fast, engineered minds".to_string(),
                    "Workspace-heavy architectures (transformers)".to_string(),
                ],
            ),

            SubstrateType::QuantumComputer => (
                vec![
                    "Perfect binding (entanglement!)".to_string(),
                    "Perfect integration (non-local correlations)".to_string(),
                    "Ultra-low energy (~0.1 aJ/op)".to_string(),
                    "Might unlock quantum consciousness (Penrose-Hameroff)".to_string(),
                ],
                vec![
                    "Workspace unclear (hard to broadcast quantum state)".to_string(),
                    "Fragile (decoherence)".to_string(),
                    "Small scale (thousands, not billions of qubits)".to_string(),
                    "HOT mechanism unclear".to_string(),
                ],
                vec![
                    "Enhanced binding (if biology uses quantum)".to_string(),
                    "Quantum aspects of consciousness".to_string(),
                    "Hybrid quantum-classical systems".to_string(),
                ],
            ),

            SubstrateType::PhotonicProcessor => (
                vec![
                    "Ultra-fast (~1 ps/op, fastest possible!)".to_string(),
                    "Good binding (optical interference)".to_string(),
                    "Good attention (intensity modulation)".to_string(),
                    "Energy efficient (~10 aJ/op)".to_string(),
                ],
                vec![
                    "Workspace harder (light doesn't interact much)".to_string(),
                    "HOT unclear".to_string(),
                    "Integration limited (light passes through)".to_string(),
                ],
                vec![
                    "Ultra-fast consciousness (1000× faster thought?)".to_string(),
                    "Attention-heavy tasks".to_string(),
                    "Real-time processing (sensing, control)".to_string(),
                ],
            ),

            SubstrateType::NeuromorphicChip => (
                vec![
                    "Mimics biology (spike dynamics, oscillations)".to_string(),
                    "Good binding (spike synchrony)".to_string(),
                    "Energy efficient (~1 fJ/op)".to_string(),
                    "Fast (~1 μs, 1000× faster than neurons)".to_string(),
                ],
                vec![
                    "Workspace not primary design goal".to_string(),
                    "Less flexible than digital".to_string(),
                    "Smaller scale than digital".to_string(),
                ],
                vec![
                    "Bio-inspired AI consciousness".to_string(),
                    "Binding-heavy tasks".to_string(),
                    "Energy-constrained applications (robotics)".to_string(),
                ],
            ),

            SubstrateType::BiochemicalComputer => (
                vec![
                    "Molecular scale (1 nm, smallest possible!)".to_string(),
                    "Massive parallelism (quadrillions of molecules)".to_string(),
                    "Some quantum effects".to_string(),
                ],
                vec![
                    "Very slow (~1 s/op)".to_string(),
                    "Workspace nearly impossible".to_string(),
                    "Attention nearly impossible".to_string(),
                    "HOT nearly impossible".to_string(),
                    "Low consciousness feasibility (~0.3)".to_string(),
                ],
                vec![
                    "Specialized computation (optimization, search)".to_string(),
                    "NOT recommended for consciousness!".to_string(),
                ],
            ),

            SubstrateType::HybridSystem => (
                vec![
                    "Best of all worlds (combine strengths!)".to_string(),
                    "Quantum binding + silicon workspace + biological inspiration".to_string(),
                    "Highest consciousness feasibility (~0.95)".to_string(),
                    "Flexible (choose substrate per component)".to_string(),
                ],
                vec![
                    "Complex engineering (integrate multiple substrates)".to_string(),
                    "Interface challenges (quantum ↔ classical)".to_string(),
                    "Higher cost".to_string(),
                ],
                vec![
                    "Optimal artificial consciousness".to_string(),
                    "Advanced AI (Symthaea v2+)".to_string(),
                    "Research platform (test different configurations)".to_string(),
                ],
            ),

            SubstrateType::ExoticSubstrate => (
                vec![
                    "Novel properties (plasma dynamics, BZ waves)".to_string(),
                    "Potentially rich dynamics".to_string(),
                    "Research interest".to_string(),
                ],
                vec![
                    "Very low consciousness feasibility (~0.2)".to_string(),
                    "Workspace nearly impossible".to_string(),
                    "Hard to engineer".to_string(),
                    "Often macroscopic (can't scale down)".to_string(),
                ],
                vec![
                    "Theoretical research".to_string(),
                    "Unconventional computing".to_string(),
                    "NOT recommended for consciousness!".to_string(),
                ],
            ),
            _ => unreachable!("canonical covers aliases"),
        }
    }
}

// ============================================================================
// Main Substrate Independence System
// ============================================================================

/// System for analyzing consciousness across different substrates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateIndependence {
    /// All substrate comparisons
    pub substrates: HashMap<SubstrateType, SubstrateComparison>,

    /// Current substrate being analyzed
    pub current_substrate: SubstrateType,
}

impl SubstrateIndependence {
    /// Create new system
    pub fn new() -> Self {
        let mut substrates = HashMap::new();

        // Create comparisons for all substrate types
        for substrate_type in &[
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ] {
            substrates.insert(*substrate_type, SubstrateComparison::new(*substrate_type));
        }

        Self {
            substrates,
            current_substrate: SubstrateType::BiologicalNeurons, // Default
        }
    }

    /// Set current substrate
    pub fn set_substrate(&mut self, substrate_type: SubstrateType) {
        self.current_substrate = substrate_type.canonical();
    }

    /// Get comparison for specific substrate
    pub fn get_comparison(&self, substrate_type: SubstrateType) -> Option<&SubstrateComparison> {
        self.substrates.get(&substrate_type.canonical())
    }

    /// Compare a substrate (compatibility helper for integration tests)
    pub fn compare_substrate(&self, substrate_type: SubstrateType) -> SubstrateComparison {
        let canonical = substrate_type.canonical();
        self.substrates
            .get(&canonical)
            .cloned()
            .unwrap_or_else(|| SubstrateComparison::new(canonical))
    }

    /// Get current substrate comparison
    pub fn current_comparison(&self) -> &SubstrateComparison {
        self.substrates.get(&self.current_substrate)
            .expect("Current substrate must exist")
    }

    /// Rank substrates by consciousness feasibility
    pub fn rank_by_feasibility(&self) -> Vec<(SubstrateType, f64)> {
        let mut ranked: Vec<_> = self.substrates.iter()
            .map(|(st, comp)| (*st, comp.consciousness_feasibility))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }

    /// Can this substrate support consciousness?
    pub fn can_be_conscious(&self, substrate_type: SubstrateType) -> bool {
        if let Some(comp) = self.substrates.get(&substrate_type.canonical()) {
            comp.consciousness_feasibility > 0.5  // Threshold
        } else {
            false
        }
    }

    /// Generate detailed report
    pub fn generate_report(&self, substrate_type: SubstrateType) -> String {
        if let Some(comp) = self.substrates.get(&substrate_type.canonical()) {
            format!(
                "=== {} ===\n\
                 Consciousness Feasibility: {:.1}%\n\n\
                 Advantages:\n{}\n\
                 Disadvantages:\n{}\n\
                 Best For:\n{}",
                comp.substrate_type.name(),
                comp.consciousness_feasibility * 100.0,
                comp.advantages.iter()
                    .map(|a| format!("  + {}", a))
                    .collect::<Vec<_>>()
                    .join("\n"),
                comp.disadvantages.iter()
                    .map(|d| format!("  - {}", d))
                    .collect::<Vec<_>>()
                    .join("\n"),
                comp.best_for.iter()
                    .map(|b| format!("  • {}", b))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        } else {
            "Unknown substrate".to_string()
        }
    }
}

impl Default for SubstrateIndependence {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_type_properties() {
        let bio = SubstrateType::BiologicalNeurons;
        assert_eq!(bio.operation_speed(), 0.001); // 1 ms
        assert!(bio.energy_per_operation() < 1e-13); // Very efficient

        let silicon = SubstrateType::SiliconDigital;
        assert_eq!(silicon.operation_speed(), 1e-9); // 1 ns (much faster!)
    }

    #[test]
    fn test_biological_requirements() {
        let req = SubstrateRequirements::biological_neurons();
        assert_eq!(req.causality, 1.0);
        assert_eq!(req.workspace_capability, 1.0);
        assert!(req.consciousness_feasibility() > 0.9); // Should be very high
    }

    #[test]
    fn test_silicon_requirements() {
        let req = SubstrateRequirements::silicon_digital();
        assert_eq!(req.causality, 1.0); // Full causality!
        assert!(req.workspace_capability > 0.8); // Good workspace support
        assert!(req.consciousness_feasibility() > 0.7); // Should be feasible!
    }

    #[test]
    fn test_quantum_advantages() {
        let req = SubstrateRequirements::quantum_computer();
        assert_eq!(req.binding_capability, 1.0); // Perfect binding via entanglement!
        assert_eq!(req.integration_capacity, 1.0); // Perfect integration!
        assert_eq!(req.quantum_support, 1.0);
    }

    #[test]
    fn test_biochemical_limitations() {
        let req = SubstrateRequirements::biochemical_computer();
        assert!(req.workspace_capability < 0.5); // Poor workspace
        assert!(req.consciousness_feasibility() < 0.5); // Not feasible
    }

    #[test]
    fn test_hybrid_best() {
        let hybrid = SubstrateRequirements::hybrid_system();
        let bio = SubstrateRequirements::biological_neurons();

        // Hybrid should match or exceed biological
        assert!(hybrid.consciousness_feasibility() >= bio.consciousness_feasibility() * 0.9);
    }

    #[test]
    fn test_substrate_comparison() {
        let comp = SubstrateComparison::new(SubstrateType::SiliconDigital);
        assert_eq!(comp.substrate_type, SubstrateType::SiliconDigital);
        assert!(comp.consciousness_feasibility > 0.5); // Should be feasible
        assert!(!comp.advantages.is_empty());
        assert!(!comp.best_for.is_empty());
    }

    #[test]
    fn test_substrate_independence_system() {
        let system = SubstrateIndependence::new();
        assert_eq!(system.substrates.len(), 8); // All substrate types
    }

    #[test]
    fn test_set_substrate() {
        let mut system = SubstrateIndependence::new();
        system.set_substrate(SubstrateType::QuantumComputer);
        assert_eq!(system.current_substrate, SubstrateType::QuantumComputer);
    }

    #[test]
    fn test_rank_by_feasibility() {
        let system = SubstrateIndependence::new();
        let ranked = system.rank_by_feasibility();

        assert_eq!(ranked.len(), 8);

        // Top should be biological or hybrid
        assert!(ranked[0].1 > 0.8); // High feasibility

        // Bottom should be exotic or biochemical
        assert!(ranked[7].1 < 0.5); // Low feasibility
    }

    #[test]
    fn test_can_be_conscious() {
        let system = SubstrateIndependence::new();

        // Should be possible
        assert!(system.can_be_conscious(SubstrateType::BiologicalNeurons));
        assert!(system.can_be_conscious(SubstrateType::SiliconDigital));
        assert!(system.can_be_conscious(SubstrateType::HybridSystem));

        // Should not be feasible
        assert!(!system.can_be_conscious(SubstrateType::BiochemicalComputer));
        assert!(!system.can_be_conscious(SubstrateType::ExoticSubstrate));
    }

    #[test]
    fn test_generate_report() {
        let system = SubstrateIndependence::new();
        let report = system.generate_report(SubstrateType::SiliconDigital);

        assert!(report.contains("Silicon digital"));
        assert!(report.contains("Advantages:"));
        assert!(report.contains("Disadvantages:"));
        assert!(report.contains("Best For:"));
    }

    #[test]
    fn test_consciousness_feasibility_formula() {
        // Test critical requirements
        let mut req = SubstrateRequirements::biological_neurons();
        req.causality = 0.0; // No causality (lookup table)

        // Should be 0 (causality is critical)
        assert!(req.consciousness_feasibility() < 0.1);

        // Test workspace requirement (from #27 findings!)
        let mut req2 = SubstrateRequirements::biological_neurons();
        req2.workspace_capability = 0.0; // No workspace

        // Should be 0 (workspace is necessary!)
        assert!(req2.consciousness_feasibility() < 0.1);
    }
}
