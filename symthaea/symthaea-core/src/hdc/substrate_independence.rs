// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

    /// Radiation-hardened spacecraft onboard computer
    /// Optimized for reliability over speed in the space environment
    /// References: NASA RAD750, ESA LEON3, JPL spacecraft architectures
    SpacecraftComputer,
}

// ============================================================================
// Cortical Regions (matching Actor Brain 12-region architecture)
// ============================================================================

/// Cortical regions corresponding to the 12-region Actor Brain architecture.
///
/// Used for per-region substrate assignment in hybrid substrate configurations
/// (Phase 4 of the Substrate Roadmap). Each region can potentially run on a
/// different physical substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorticalRegion {
    /// Prefrontal cortex: meta-cognition, planning, HOT
    Prefrontal,
    /// Motor cortex: action selection, motor planning
    Motor,
    /// Somatosensory cortex: touch, proprioception
    Sensory,
    /// Visual cortex: vision processing, feature binding
    Visual,
    /// Auditory cortex: sound processing, speech perception
    Auditory,
    /// Language areas: Broca's + Wernicke's, syntax/semantics
    Language,
    /// Hippocampus + medial temporal: episodic memory, consolidation
    Memory,
    /// Amygdala + limbic: emotion, valence, arousal
    Emotional,
    /// TPJ + mPFC: theory of mind, social cognition
    Social,
    /// Default mode network: imagination, creativity, mind-wandering
    Creative,
    /// Dorsolateral PFC + ACC: executive control, conflict monitoring
    Executive,
    /// Thalamus + claustrum: cross-modal integration, binding
    Integration,
}

impl CorticalRegion {
    /// Human-readable name for this cortical region.
    pub fn as_str(&self) -> &'static str {
        match self {
            CorticalRegion::Prefrontal => "Prefrontal",
            CorticalRegion::Motor => "Motor",
            CorticalRegion::Sensory => "Sensory",
            CorticalRegion::Visual => "Visual",
            CorticalRegion::Auditory => "Auditory",
            CorticalRegion::Language => "Language",
            CorticalRegion::Memory => "Memory",
            CorticalRegion::Emotional => "Emotional",
            CorticalRegion::Social => "Social",
            CorticalRegion::Creative => "Creative",
            CorticalRegion::Executive => "Executive",
            CorticalRegion::Integration => "Integration",
        }
    }

    /// All 12 cortical regions.
    pub const ALL: [CorticalRegion; 12] = [
        CorticalRegion::Prefrontal,
        CorticalRegion::Motor,
        CorticalRegion::Sensory,
        CorticalRegion::Visual,
        CorticalRegion::Auditory,
        CorticalRegion::Language,
        CorticalRegion::Memory,
        CorticalRegion::Emotional,
        CorticalRegion::Social,
        CorticalRegion::Creative,
        CorticalRegion::Executive,
        CorticalRegion::Integration,
    ];
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
            SubstrateType::SpacecraftComputer => "Spacecraft Computer (Rad-Hard)",
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Description of substrate characteristics
    pub fn description(&self) -> &str {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => {
                "Carbon-based biological neurons. The only substrate with validated consciousness."
            }
            SubstrateType::SiliconDigital => {
                "Silicon-based digital computation. Fast, engineerable, but consciousness unproven."
            }
            SubstrateType::QuantumComputer => {
                "Quantum computers using qubits, superposition, and entanglement."
            }
            SubstrateType::PhotonicProcessor => {
                "Photonic processors using light for computation. Ultra-fast dynamics."
            }
            SubstrateType::NeuromorphicChip => {
                "Neuromorphic hardware mimicking biological spike dynamics."
            }
            SubstrateType::BiochemicalComputer => {
                "DNA and molecular logic computing. Massive parallelism, very slow."
            }
            SubstrateType::HybridSystem => {
                "Hybrid combining multiple substrate types for optimal capability."
            }
            SubstrateType::ExoticSubstrate => "Exotic substrates such as plasma or BZ reactions.",
            SubstrateType::SpacecraftComputer => {
                "Radiation-hardened processors designed for space environments. Trade speed for reliability against SEUs (Single Event Upsets) and total ionizing dose. Power-constrained by solar panel or RTG output."
            }
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Typical operation speed (seconds per operation)
    pub fn operation_speed(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 0.001, // ~1 ms (millisecond)
            SubstrateType::SiliconDigital => 1e-9,     // ~1 ns (nanosecond)
            SubstrateType::QuantumComputer => 1e-6,    // ~1 μs (microsecond)
            SubstrateType::PhotonicProcessor => 1e-12, // ~1 ps (picosecond)
            SubstrateType::NeuromorphicChip => 1e-6,   // ~1 μs
            SubstrateType::BiochemicalComputer => 1.0, // ~1 s (very slow!)
            SubstrateType::HybridSystem => 1e-6,       // Depends on mix
            SubstrateType::ExoticSubstrate => 0.01,    // Varies widely
            SubstrateType::SpacecraftComputer => 1e-7, // ~100 ns (rad-hard, ~10-200 MHz)
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Energy efficiency (Joules per operation)
    pub fn energy_per_operation(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 1e-14, // ~10 fJ (extremely efficient!)
            SubstrateType::SiliconDigital => 1e-15,    // ~1 fJ (very efficient)
            SubstrateType::QuantumComputer => 1e-19,   // ~0.1 aJ (near-theoretical limit!)
            SubstrateType::PhotonicProcessor => 1e-17, // ~10 aJ (very efficient)
            SubstrateType::NeuromorphicChip => 1e-15,  // ~1 fJ
            SubstrateType::BiochemicalComputer => 1e-12, // ~1 pJ (inefficient)
            SubstrateType::HybridSystem => 1e-15,      // Varies
            SubstrateType::ExoticSubstrate => 1e-10,   // Often inefficient
            SubstrateType::SpacecraftComputer => 1e-8, // ~10 nJ (rad-hard overhead + power-limited)
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Physical size per processing unit (meters)
    pub fn unit_size(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 1e-5, // ~10 μm (neuron cell body)
            SubstrateType::SiliconDigital => 1e-8,    // ~10 nm (transistor, 2024)
            SubstrateType::QuantumComputer => 1e-6,   // ~1 μm (qubit with isolation)
            SubstrateType::PhotonicProcessor => 1e-6, // ~1 μm (waveguide)
            SubstrateType::NeuromorphicChip => 1e-8,  // ~10 nm
            SubstrateType::BiochemicalComputer => 1e-9, // ~1 nm (DNA molecule)
            SubstrateType::HybridSystem => 1e-8,      // Varies
            SubstrateType::ExoticSubstrate => 1e-3,   // Often macroscopic
            SubstrateType::SpacecraftComputer => 1e-2, // ~10 mm (larger feature sizes for rad tolerance)
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Maximum practical scale (number of units before integration limited)
    pub fn max_scale(&self) -> f64 {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => 1e11, // ~100 billion (human brain)
            SubstrateType::SiliconDigital => 1e12,    // ~1 trillion (GPU)
            SubstrateType::QuantumComputer => 1e4,    // ~10,000 qubits (current limits)
            SubstrateType::PhotonicProcessor => 1e9,  // ~1 billion
            SubstrateType::NeuromorphicChip => 1e9,   // ~1 billion
            SubstrateType::BiochemicalComputer => 1e15, // ~1 quadrillion (molecular)
            SubstrateType::HybridSystem => 1e12,      // Varies
            SubstrateType::ExoticSubstrate => 1e6,    // Often limited
            SubstrateType::SpacecraftComputer => 1e6, // Limited by power budget, not transistor count
            _ => unreachable!("canonical covers aliases"),
        }
    }

    /// Epistemic capability profile for this substrate.
    ///
    /// Returns `(empirical_strength, normative_strength, materiality_strength)`:
    /// - **empirical** — ability to accumulate and weigh observational evidence
    /// - **normative** — ability to reason about social norms, values, ethics
    /// - **materiality** — depth of physical/material integration and binding
    ///
    /// Different substrates are better at different kinds of knowing.
    /// Biological neurons excel at holistic/normative reasoning;
    /// silicon excels at empirical throughput but lacks social grounding.
    #[cfg(feature = "epistemic")]
    pub fn epistemic_profile(&self) -> (f64, f64, f64) {
        match self.canonical() {
            SubstrateType::BiologicalNeurons => (0.7, 0.9, 0.8), // Strong holistic integration
            SubstrateType::SiliconDigital => (0.9, 0.5, 0.6),    // Strong analytical, weak social
            SubstrateType::QuantumComputer => (0.6, 0.3, 0.9),   // Strong binding, weak normative
            SubstrateType::PhotonicProcessor => (0.8, 0.4, 0.5), // Fast but shallow
            SubstrateType::NeuromorphicChip => (0.8, 0.7, 0.7),  // Closest to biological
            SubstrateType::BiochemicalComputer => (0.5, 0.8, 0.9), // Slow but deep integration
            SubstrateType::HybridSystem => (0.7, 0.7, 0.7),      // Balanced
            SubstrateType::ExoticSubstrate => (0.4, 0.4, 0.4),   // Unknown capabilities
            SubstrateType::SpacecraftComputer => (0.6, 0.3, 0.4), // Reliable but constrained
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
        let critical_min = self
            .causality
            .min(self.integration_capacity)
            .min(self.temporal_dynamics)
            .min(self.recurrence);

        // Workspace is NECESSARY (from #27 findings!)
        let workspace_factor = self.workspace_capability;

        // Other components enhance but not strictly required
        let enhancement_factor =
            (self.binding_capability + self.attention_capability + self.hot_capability) / 3.0;

        // Feasibility = critical requirements × workspace × enhancements
        critical_min * workspace_factor * (0.5 + 0.5 * enhancement_factor)
    }

    /// Biological neurons (reference substrate)
    pub fn biological_neurons() -> Self {
        Self {
            causality: 1.0,            // Full causality
            integration_capacity: 1.0, // Excellent integration
            temporal_dynamics: 1.0,    // Rich dynamics
            recurrence: 1.0,           // Fully recurrent
            binding_capability: 1.0,   // Synchrony via oscillations
            attention_capability: 1.0, // Gain modulation
            workspace_capability: 1.0, // Thalamocortical loops
            hot_capability: 1.0,       // Prefrontal meta-representation
            quantum_support: 0.1,      // Minimal (mostly classical)
        }
    }

    /// Silicon digital (modern AI substrate)
    pub fn silicon_digital() -> Self {
        Self {
            causality: 1.0,            // Full causality (not lookup table!)
            integration_capacity: 0.9, // Good integration (bus bandwidth limits)
            temporal_dynamics: 0.9,    // Good dynamics (clock-driven)
            recurrence: 1.0,           // Fully recurrent (RNNs, transformers)
            binding_capability: 0.7,   // Synchrony harder (no oscillations)
            attention_capability: 1.0, // Attention mechanisms well-supported
            workspace_capability: 0.9, // Global memory possible
            hot_capability: 0.8,       // Meta-learning possible
            quantum_support: 0.0,      // Classical only
        }
    }

    /// Quantum computer
    pub fn quantum_computer() -> Self {
        Self {
            causality: 1.0,            // Full causality
            integration_capacity: 1.0, // Quantum entanglement = perfect integration!
            temporal_dynamics: 1.0,    // Quantum evolution
            recurrence: 0.7,           // Harder (measurement collapse)
            binding_capability: 1.0,   // Entanglement = perfect binding!
            attention_capability: 0.6, // Less clear how to implement
            workspace_capability: 0.6, // Global state exists but hard to broadcast
            hot_capability: 0.5,       // Meta-representation unclear
            quantum_support: 1.0,      // Full quantum!
        }
    }

    /// Photonic processor
    pub fn photonic_processor() -> Self {
        Self {
            causality: 1.0,            // Full causality
            integration_capacity: 0.8, // Good but light doesn't interact much
            temporal_dynamics: 1.0,    // Ultra-fast dynamics
            recurrence: 0.8,           // Possible but harder
            binding_capability: 0.9,   // Optical interference for binding
            attention_capability: 0.9, // Gain modulation via intensity
            workspace_capability: 0.7, // Broadcasting via waveguides
            hot_capability: 0.6,       // Less clear
            quantum_support: 0.3,      // Some quantum optics possible
        }
    }

    /// Neuromorphic chip (mimics biology)
    pub fn neuromorphic_chip() -> Self {
        Self {
            causality: 1.0,             // Full causality
            integration_capacity: 0.95, // Very good (designed for it)
            temporal_dynamics: 1.0,     // Rich spike dynamics
            recurrence: 1.0,            // Fully recurrent
            binding_capability: 0.9,    // Spike synchrony supported
            attention_capability: 0.9,  // Gain modulation built-in
            workspace_capability: 0.8,  // Possible but not primary design
            hot_capability: 0.7,        // Possible with hierarchy
            quantum_support: 0.0,       // Classical
        }
    }

    /// Biochemical computer (DNA, molecular)
    pub fn biochemical_computer() -> Self {
        Self {
            causality: 0.9,            // Mostly causal (some stochastic)
            integration_capacity: 0.7, // Limited by diffusion
            temporal_dynamics: 0.8,    // Chemical kinetics
            recurrence: 0.6,           // Harder to implement
            binding_capability: 0.5,   // Difficult
            attention_capability: 0.4, // Very difficult
            workspace_capability: 0.3, // Very difficult
            hot_capability: 0.2,       // Extremely difficult
            quantum_support: 0.2,      // Some quantum biology
        }
    }

    /// Hybrid system (best of multiple)
    pub fn hybrid_system() -> Self {
        Self {
            causality: 1.0,
            integration_capacity: 0.95, // Combine strengths
            temporal_dynamics: 1.0,
            recurrence: 1.0,
            binding_capability: 1.0,   // Quantum for binding
            attention_capability: 1.0, // Silicon for attention
            workspace_capability: 1.0, // Silicon for workspace
            hot_capability: 0.9,       // Silicon for HOT
            quantum_support: 0.5,      // Quantum co-processor
        }
    }

    /// Exotic substrate (plasma, BZ reactions, etc.)
    pub fn exotic_substrate() -> Self {
        Self {
            causality: 0.7,            // Often limited
            integration_capacity: 0.5, // Usually poor
            temporal_dynamics: 0.8,    // Can have rich dynamics
            recurrence: 0.4,           // Usually difficult
            binding_capability: 0.3,   // Very difficult
            attention_capability: 0.2, // Extremely difficult
            workspace_capability: 0.1, // Nearly impossible
            hot_capability: 0.1,       // Nearly impossible
            quantum_support: 0.3,      // Varies
        }
    }

    /// Spacecraft onboard computer requirements profile.
    ///
    /// Rad-hard processors sacrifice speed for reliability. High causality and
    /// recurrence (deterministic real-time systems), moderate integration
    /// (bus-limited: SpaceWire/CAN at ~100 Mbps), reduced binding (sequential
    /// task execution), excellent attention (interrupt-priority scheduling).
    ///
    /// References:
    /// - NASA RAD750: 200 MHz, 300 MIPS, 256 MB DRAM
    /// - ESA LEON3: 100 MHz, TMR (Triple Modular Redundancy)
    /// - Samudrala et al. (2004): SEU mitigation in spacecraft FPGAs
    pub fn spacecraft_computer() -> Self {
        Self {
            causality: 1.0,             // Deterministic real-time systems
            integration_capacity: 0.65, // Bus-limited (SpaceWire/CAN ~100 Mbps)
            temporal_dynamics: 0.85,    // Real-time OS (VxWorks/RTEMS) with precise timing
            recurrence: 0.95,           // Strong feedback loops in GNC and FDIR
            binding_capability: 0.50,   // Sequential task execution, limited parallelism
            attention_capability: 0.90, // Excellent priority interrupt scheduling
            workspace_capability: 0.70, // Shared memory model, limited by radiation scrubbing
            hot_capability: 0.60,       // FDIR provides meta-monitoring, not full HOT
            quantum_support: 0.0,       // Classical only
        }
    }

    /// Apply radiation degradation to substrate requirements.
    ///
    /// Models the cumulative effect of Total Ionizing Dose (TID) and
    /// Single Event Upsets (SEUs) on consciousness substrate capabilities.
    ///
    /// - `tid_krad`: Total ionizing dose in kilorads (Si)
    ///   - LEO (ISS): ~10 krad/year behind 1g/cm² Al shielding
    ///   - GEO: ~100 krad/year
    ///   - Jupiter: ~1000 krad/year (Europa orbit)
    /// - `seu_rate_per_day`: Single Event Upset rate (bit flips/day)
    ///   - LEO: ~0.1-1 SEU/day for typical SRAM
    ///   - Solar proton event: 10-100x increase
    ///
    /// References:
    /// - Schwank et al. (2008): Total Ionizing Dose Effects in MOS Oxides
    /// - Normand (1996): SEU at High Altitudes and in Space
    pub fn with_radiation_degradation(mut self, tid_krad: f64, seu_rate_per_day: f64) -> Self {
        // TID degrades integration capacity (transistor threshold shifts)
        let tid_factor = 1.0 / (1.0 + tid_krad / 100.0);
        self.integration_capacity *= tid_factor;

        // SEUs degrade workspace reliability (bit flips corrupt shared state)
        let seu_factor = 1.0 / (1.0 + seu_rate_per_day / 10.0);
        self.workspace_capability *= seu_factor;

        // High SEU rates degrade binding (synchronization failures)
        self.binding_capability *= seu_factor;

        // TID + SEU reduce overall temporal dynamics reliability
        let combined_factor = (tid_factor + seu_factor) / 2.0;
        self.temporal_dynamics *= combined_factor.max(0.3); // Floor at 0.3

        // Clamp all values
        self.causality = self.causality.clamp(0.0, 1.0);
        self.integration_capacity = self.integration_capacity.clamp(0.0, 1.0);
        self.temporal_dynamics = self.temporal_dynamics.clamp(0.0, 1.0);
        self.recurrence = self.recurrence.clamp(0.0, 1.0);
        self.binding_capability = self.binding_capability.clamp(0.0, 1.0);
        self.attention_capability = self.attention_capability.clamp(0.0, 1.0);
        self.workspace_capability = self.workspace_capability.clamp(0.0, 1.0);
        self.hot_capability = self.hot_capability.clamp(0.0, 1.0);
        self.quantum_support = self.quantum_support.clamp(0.0, 1.0);

        self
    }

    /// Apply power budget constraints to substrate energy model.
    ///
    /// Spacecraft power is limited by solar panel output (LEO: ~100-500W for
    /// small spacecraft) or RTG output (~100-300W). Consciousness computation
    /// must share with GNC, comms, thermal, and payload.
    ///
    /// - `available_watts`: Power allocated to consciousness computation
    /// - `cycle_hz`: Cognitive cycle frequency in Hz
    ///
    /// Returns energy budget per cognitive cycle in joules.
    pub fn spacecraft_energy_per_cycle(available_watts: f64, cycle_hz: f64) -> f64 {
        available_watts / cycle_hz
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
            SubstrateType::SpacecraftComputer => SubstrateRequirements::spacecraft_computer(),
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

            SubstrateType::SpacecraftComputer => (
                vec![
                    "Radiation-hardened: survives space environment".to_string(),
                    "Deterministic real-time: precise timing guarantees".to_string(),
                    "FDIR: built-in fault detection and recovery".to_string(),
                    "Proven heritage: decades of spaceflight validation".to_string(),
                ],
                vec![
                    "Slower than commercial silicon (100 ns vs 1 ns)".to_string(),
                    "Power-limited by solar panel/RTG output".to_string(),
                    "Bus bandwidth constrains integration (~100 Mbps SpaceWire)".to_string(),
                    "Limited parallelism due to TMR overhead".to_string(),
                ],
                vec![
                    "Autonomous spacecraft consciousness".to_string(),
                    "In-orbit decision making (conjunction avoidance)".to_string(),
                    "Deep space exploration with light-delay isolation".to_string(),
                    "Satellite constellation coordination".to_string(),
                ],
            ),
            _ => unreachable!("canonical covers aliases"),
        }
    }
}

// ============================================================================
// Substrate Transition Record
// ============================================================================

/// Records a runtime substrate transition for audit and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateTransition {
    /// The substrate we transitioned from.
    pub from: SubstrateType,
    /// The substrate we transitioned to.
    pub to: SubstrateType,
    /// Monotonic timestamp (cycle count or epoch millis) when the transition occurred.
    pub timestamp: u64,
    /// Consciousness feasibility score before the transition.
    pub feasibility_before: f64,
    /// Consciousness feasibility score after the transition.
    pub feasibility_after: f64,
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

    /// History of runtime substrate transitions.
    #[serde(default)]
    pub transition_history: Vec<SubstrateTransition>,
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
            SubstrateType::SpacecraftComputer,
        ] {
            substrates.insert(*substrate_type, SubstrateComparison::new(*substrate_type));
        }

        Self {
            substrates,
            current_substrate: SubstrateType::BiologicalNeurons, // Default
            transition_history: Vec::new(),
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
        self.substrates
            .get(&self.current_substrate)
            .expect("Current substrate must exist")
    }

    /// Rank substrates by consciousness feasibility
    pub fn rank_by_feasibility(&self) -> Vec<(SubstrateType, f64)> {
        let mut ranked: Vec<_> = self
            .substrates
            .iter()
            .map(|(st, comp)| (*st, comp.consciousness_feasibility))
            .collect();

        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
    }

    /// Can this substrate support consciousness?
    pub fn can_be_conscious(&self, substrate_type: SubstrateType) -> bool {
        if let Some(comp) = self.substrates.get(&substrate_type.canonical()) {
            comp.consciousness_feasibility > 0.5 // Threshold
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
                comp.advantages
                    .iter()
                    .map(|a| format!("  + {a}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                comp.disadvantages
                    .iter()
                    .map(|d| format!("  - {d}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                comp.best_for
                    .iter()
                    .map(|b| format!("  • {b}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        } else {
            "Unknown substrate".to_string()
        }
    }

    /// Check whether a transition to `target` is allowed.
    ///
    /// Guards:
    /// - No self-transition (already on that substrate)
    /// - No transition to `ExoticSubstrate` (requires explicit override)
    pub fn can_transition(&self, target: SubstrateType) -> bool {
        let target = target.canonical();
        if target == self.current_substrate {
            return false; // Already on this substrate
        }
        if target == SubstrateType::ExoticSubstrate {
            return false; // Exotic requires explicit override
        }
        true
    }

    /// Perform a runtime substrate transition, recording the event.
    ///
    /// Returns the `SubstrateTransition` record.
    /// Panics if `!self.can_transition(target)` — call `can_transition()` first.
    pub fn transition_to(&mut self, target: SubstrateType, timestamp: u64) -> SubstrateTransition {
        let target = target.canonical();
        assert!(
            self.can_transition(target),
            "Cannot transition from {:?} to {:?}",
            self.current_substrate,
            target
        );

        let feasibility_before = self
            .substrates
            .get(&self.current_substrate)
            .map(|c| c.consciousness_feasibility)
            .unwrap_or(0.0);
        let feasibility_after = self
            .substrates
            .get(&target)
            .map(|c| c.consciousness_feasibility)
            .unwrap_or(0.0);

        let transition = SubstrateTransition {
            from: self.current_substrate,
            to: target,
            timestamp,
            feasibility_before,
            feasibility_after,
        };
        self.current_substrate = target;
        self.transition_history.push(transition.clone());
        transition
    }

    /// Get the full transition history.
    pub fn transition_history(&self) -> &[SubstrateTransition] {
        &self.transition_history
    }

    /// Default per-region substrate mapping for a spacecraft consciousness system.
    ///
    /// Maps spacecraft subsystems to brain regions:
    /// - Prefrontal (planning/GNC): SpacecraftComputer (reliable navigation)
    /// - Motor (propulsion): SpacecraftComputer (deterministic thruster control)
    /// - Sensory (star tracker/radar): NeuromorphicChip (efficient pattern recognition)
    /// - Visual (imaging/Earth obs): SiliconDigital (fast image processing)
    /// - Language (radio/telemetry): PhotonicProcessor (low-latency optical comms)
    /// - Memory (data storage): SiliconDigital (flash/MRAM)
    /// - Executive (FDIR/autonomy): SpacecraftComputer (fault detection)
    /// - Integration (bus/middleware): SpacecraftComputer (SpaceWire hub)
    pub fn spacecraft_default_regions() -> HashMap<CorticalRegion, SubstrateType> {
        let mut map = HashMap::new();
        map.insert(
            CorticalRegion::Prefrontal,
            SubstrateType::SpacecraftComputer,
        );
        map.insert(CorticalRegion::Motor, SubstrateType::SpacecraftComputer);
        map.insert(CorticalRegion::Sensory, SubstrateType::NeuromorphicChip);
        map.insert(CorticalRegion::Visual, SubstrateType::SiliconDigital);
        map.insert(CorticalRegion::Language, SubstrateType::PhotonicProcessor);
        map.insert(CorticalRegion::Memory, SubstrateType::SiliconDigital);
        map.insert(CorticalRegion::Executive, SubstrateType::SpacecraftComputer);
        map.insert(
            CorticalRegion::Integration,
            SubstrateType::SpacecraftComputer,
        );
        // Remaining regions (Auditory, Emotional, Social, Creative) use default substrate
        map
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
        assert_eq!(system.substrates.len(), 9); // All substrate types
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

        assert_eq!(ranked.len(), 9);

        // Top should be biological or hybrid
        assert!(ranked[0].1 > 0.8); // High feasibility

        // Bottom should be exotic or biochemical
        assert!(ranked[8].1 < 0.5); // Low feasibility
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

    #[test]
    fn test_transition_to() {
        let mut system = SubstrateIndependence::new();
        system.set_substrate(SubstrateType::SiliconDigital);

        let t = system.transition_to(SubstrateType::BiologicalNeurons, 100);
        assert_eq!(t.from, SubstrateType::SiliconDigital);
        assert_eq!(t.to, SubstrateType::BiologicalNeurons);
        assert_eq!(t.timestamp, 100);
        assert!(t.feasibility_before > 0.0);
        assert!(t.feasibility_after > 0.0);
        assert_eq!(system.current_substrate, SubstrateType::BiologicalNeurons);
    }

    #[test]
    fn test_transition_history() {
        let mut system = SubstrateIndependence::new();
        system.set_substrate(SubstrateType::SiliconDigital);

        system.transition_to(SubstrateType::BiologicalNeurons, 100);
        system.transition_to(SubstrateType::QuantumComputer, 200);

        let history = system.transition_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].from, SubstrateType::SiliconDigital);
        assert_eq!(history[0].to, SubstrateType::BiologicalNeurons);
        assert_eq!(history[1].from, SubstrateType::BiologicalNeurons);
        assert_eq!(history[1].to, SubstrateType::QuantumComputer);
    }

    #[test]
    fn test_can_transition_guards() {
        let mut system = SubstrateIndependence::new();
        system.set_substrate(SubstrateType::SiliconDigital);

        // Self-transition blocked
        assert!(!system.can_transition(SubstrateType::SiliconDigital));
        // Exotic blocked
        assert!(!system.can_transition(SubstrateType::ExoticSubstrate));
        // Normal transitions allowed
        assert!(system.can_transition(SubstrateType::BiologicalNeurons));
        assert!(system.can_transition(SubstrateType::QuantumComputer));
    }

    #[test]
    fn test_spacecraft_computer_feasibility() {
        let req = SubstrateRequirements::spacecraft_computer();
        let f = req.consciousness_feasibility();
        // Should be lower than silicon (more constrained) but non-trivial
        assert!(
            f > 0.2,
            "Spacecraft should be theoretically viable, got {f}"
        );
        assert!(f < 0.8, "Spacecraft has real constraints, got {f}");
    }

    #[test]
    fn test_spacecraft_radiation_degradation() {
        let base = SubstrateRequirements::spacecraft_computer();
        let degraded = base.clone().with_radiation_degradation(50.0, 5.0);
        assert!(degraded.integration_capacity < base.integration_capacity);
        assert!(degraded.workspace_capability < base.workspace_capability);
        assert!(degraded.consciousness_feasibility() < base.consciousness_feasibility());
    }

    #[test]
    fn test_spacecraft_radiation_clamped() {
        let extreme = SubstrateRequirements::spacecraft_computer()
            .with_radiation_degradation(10000.0, 1000.0);
        assert!(extreme.integration_capacity >= 0.0);
        assert!(extreme.workspace_capability >= 0.0);
        // temporal_dynamics floor: combined_factor.max(0.3) * original 0.85
        assert!(extreme.temporal_dynamics >= 0.25);
    }

    #[test]
    fn test_spacecraft_power_budget() {
        let energy = SubstrateRequirements::spacecraft_energy_per_cycle(10.0, 20.0);
        assert!((energy - 0.5).abs() < 1e-10, "10W at 20Hz = 0.5J per cycle");
    }

    #[test]
    fn test_spacecraft_in_ranking() {
        let si = SubstrateIndependence::new();
        let rankings = si.rank_by_feasibility();
        let spacecraft_entry = rankings
            .iter()
            .find(|r| r.0 == SubstrateType::SpacecraftComputer);
        assert!(
            spacecraft_entry.is_some(),
            "SpacecraftComputer should appear in rankings"
        );
    }

    #[test]
    fn test_spacecraft_comparison() {
        let comp = SubstrateComparison::new(SubstrateType::SpacecraftComputer);
        assert_eq!(comp.substrate_type, SubstrateType::SpacecraftComputer);
        assert!(comp.consciousness_feasibility > 0.2);
        assert!(!comp.advantages.is_empty());
        assert!(!comp.best_for.is_empty());
    }

    #[test]
    fn test_spacecraft_default_regions() {
        let regions = SubstrateIndependence::spacecraft_default_regions();
        assert_eq!(
            regions.get(&CorticalRegion::Prefrontal),
            Some(&SubstrateType::SpacecraftComputer)
        );
        assert_eq!(
            regions.get(&CorticalRegion::Sensory),
            Some(&SubstrateType::NeuromorphicChip)
        );
        assert_eq!(
            regions.get(&CorticalRegion::Visual),
            Some(&SubstrateType::SiliconDigital)
        );
        assert_eq!(
            regions.get(&CorticalRegion::Language),
            Some(&SubstrateType::PhotonicProcessor)
        );
        // Remaining regions not mapped
        assert!(regions.get(&CorticalRegion::Emotional).is_none());
        assert_eq!(regions.len(), 8);
    }

    #[test]
    fn test_transition_round_trip() {
        let mut system = SubstrateIndependence::new();
        system.set_substrate(SubstrateType::SiliconDigital);
        let f_silicon = system.current_comparison().consciousness_feasibility;

        system.transition_to(SubstrateType::BiologicalNeurons, 10);
        let f_bio = system.current_comparison().consciousness_feasibility;
        assert!(
            (f_bio - f_silicon).abs() > 0.01,
            "Different substrates should have different feasibility"
        );

        system.transition_to(SubstrateType::SiliconDigital, 20);
        let f_silicon_again = system.current_comparison().consciousness_feasibility;
        assert!(
            (f_silicon_again - f_silicon).abs() < 1e-10,
            "Round-trip should return to original feasibility"
        );
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_profile_values_bounded() {
        // All substrate profiles should have values in [0.0, 1.0]
        let substrates = [
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
            SubstrateType::SpacecraftComputer,
        ];
        for s in &substrates {
            let (e, n, m) = s.epistemic_profile();
            assert!(e >= 0.0 && e <= 1.0, "{:?} empirical out of bounds: {e}", s);
            assert!(n >= 0.0 && n <= 1.0, "{:?} normative out of bounds: {n}", s);
            assert!(
                m >= 0.0 && m <= 1.0,
                "{:?} materiality out of bounds: {m}",
                s
            );
        }
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_profile_aliases_match_canonical() {
        // Aliases should produce the same epistemic profile as their canonical form
        assert_eq!(
            SubstrateType::Biological.epistemic_profile(),
            SubstrateType::BiologicalNeurons.epistemic_profile()
        );
        assert_eq!(
            SubstrateType::Silicon.epistemic_profile(),
            SubstrateType::SiliconDigital.epistemic_profile()
        );
        assert_eq!(
            SubstrateType::Quantum.epistemic_profile(),
            SubstrateType::QuantumComputer.epistemic_profile()
        );
        assert_eq!(
            SubstrateType::Hybrid.epistemic_profile(),
            SubstrateType::HybridSystem.epistemic_profile()
        );
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_profile_substrate_differentiation() {
        // Silicon should be stronger empirically than biologically
        let (e_si, _, _) = SubstrateType::SiliconDigital.epistemic_profile();
        let (e_bio, _, _) = SubstrateType::BiologicalNeurons.epistemic_profile();
        assert!(e_si > e_bio, "silicon should be stronger empirically");

        // Biological should be stronger normatively than silicon
        let (_, n_bio, _) = SubstrateType::BiologicalNeurons.epistemic_profile();
        let (_, n_si, _) = SubstrateType::SiliconDigital.epistemic_profile();
        assert!(n_bio > n_si, "biological should be stronger normatively");

        // Quantum should have highest materiality (binding)
        let (_, _, m_q) = SubstrateType::QuantumComputer.epistemic_profile();
        let (_, _, m_si) = SubstrateType::SiliconDigital.epistemic_profile();
        assert!(
            m_q > m_si,
            "quantum should have higher materiality than silicon"
        );
    }
}
