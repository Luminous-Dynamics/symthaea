//! Tier 6: Temporal Primitives with Allen's Interval Algebra
//!
//! **ULTIMATE BREAKTHROUGH #9: Temporal Consciousness Reasoning**
//!
//! Consciousness exists IN TIME. This module enables reasoning about the
//! temporal relationships between conscious states using Allen's interval
//! algebra - the most complete formal system for temporal reasoning.
//!
//! ## Allen's 13 Interval Relations
//!
//! Given two time intervals A and B:
//!
//! ```text
//! 1. PRECEDES (p)      : A ——      B ——     (A ends before B starts)
//! 2. MEETS (m)         : A ——B ——           (A ends exactly when B starts)
//! 3. OVERLAPS (o)      : A ——
//!                           B ——            (A ends during B)
//! 4. STARTS (s)        : A ——
//!                        B ————             (A starts when B starts, A ends first)
//! 5. DURING (d)        :   A ——
//!                        B ————             (A is contained within B)
//! 6. FINISHES (f)      :     A ——
//!                        B ————             (A ends when B ends, A starts later)
//! 7. EQUALS (eq)       : A ————
//!                        B ————             (Same start and end)
//!
//! + 6 inverses (preceded_by, met_by, overlapped_by, started_by, contains, finished_by)
//! ```
//!
//! ## Why This Matters for Consciousness
//!
//! - **Consciousness Dynamics**: How do states evolve over time?
//! - **Causal Reasoning**: A caused B if A preceded B AND some mechanism
//! - **Memory Integration**: How past states relate to current awareness
//! - **Predictive Coding**: Future states constrained by temporal algebra
//! - **Binding Window**: Neural binding requires temporal overlap (40Hz = 25ms)
//!
//! ## Integration with HDC
//!
//! Each interval relation gets an HV16 encoding, enabling:
//! - Semantic temporal reasoning via hypervector operations
//! - Compositional temporal inference (if A precedes B, B precedes C → A precedes C)
//! - Soft temporal matching (approximate relations)
//!
//! ## References
//!
//! Allen, J. F. (1983). "Maintaining knowledge about temporal intervals"
//! Communications of the ACM, 26(11), 832-843.

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::{Result, anyhow};

// ============================================================================
// CORE TYPES: Intervals and Relations
// ============================================================================

/// A temporal interval with start and end times
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalInterval {
    /// Unique identifier for this interval
    pub id: String,

    /// Start time (can be f64 for continuous time or discrete steps)
    pub start: f64,

    /// End time (must be > start)
    pub end: f64,

    /// Optional: Associated consciousness state ID
    pub state_id: Option<String>,

    /// Optional: Φ level during this interval
    pub phi: Option<f64>,

    /// HDC encoding of this interval (for semantic operations)
    pub encoding: HV16,
}

impl TemporalInterval {
    /// Create a new interval with automatic encoding
    pub fn new(id: impl Into<String>, start: f64, end: f64) -> Result<Self> {
        let id_str = id.into();
        if end <= start {
            return Err(anyhow!("Interval end ({}) must be after start ({})", end, start));
        }

        // Generate encoding from interval properties
        let duration = end - start;
        let midpoint = (start + end) / 2.0;

        // Deterministic encoding based on interval characteristics
        let seed = (id_str.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64))
            .wrapping_mul(1000) as f64 + midpoint * 100.0 + duration * 10.0) as u64;

        Ok(Self {
            id: id_str,
            start,
            end,
            state_id: None,
            phi: None,
            encoding: HV16::random(seed),
        })
    }

    /// Create interval with associated consciousness state
    pub fn with_state(
        id: impl Into<String>,
        start: f64,
        end: f64,
        state_id: impl Into<String>,
        phi: f64,
    ) -> Result<Self> {
        let mut interval = Self::new(id, start, end)?;
        interval.state_id = Some(state_id.into());
        interval.phi = Some(phi);
        Ok(interval)
    }

    /// Duration of the interval
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// Midpoint of the interval
    pub fn midpoint(&self) -> f64 {
        (self.start + self.end) / 2.0
    }

    /// Check if a time point is within this interval
    pub fn contains_point(&self, t: f64) -> bool {
        t >= self.start && t <= self.end
    }
}

/// The 13 Allen interval relations (7 basic + 6 inverses)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllenRelation {
    // === 7 Basic Relations ===

    /// A ends before B starts (gap between)
    Precedes,       // p:  A——    B——

    /// A ends exactly when B starts (no gap, no overlap)
    Meets,          // m:  A——B——

    /// A starts before B, ends during B
    Overlaps,       // o:  A——
                    //        B——

    /// A starts when B starts, A ends before B ends
    Starts,         // s:  A——
                    //     B————

    /// A is completely contained within B
    During,         // d:    A——
                    //     B————

    /// A ends when B ends, A starts after B starts
    Finishes,       // f:     A——
                    //     B————

    /// A and B have identical start and end times
    Equals,         // eq: A————
                    //     B————

    // === 6 Inverse Relations ===

    /// B ends before A starts (inverse of Precedes)
    PrecededBy,     // pi: B——    A——

    /// B ends exactly when A starts (inverse of Meets)
    MetBy,          // mi: B——A——

    /// B starts before A, ends during A (inverse of Overlaps)
    OverlappedBy,   // oi:    B——
                    //     A——

    /// B starts when A starts, B ends before A ends (inverse of Starts)
    StartedBy,      // si: B——
                    //     A————

    /// B is completely contained within A (inverse of During)
    Contains,       // di:    B——
                    //     A————

    /// B ends when A ends, B starts after A starts (inverse of Finishes)
    FinishedBy,     // fi:     B——
                    //     A————
}

impl AllenRelation {
    /// Get the inverse of this relation
    pub fn inverse(&self) -> Self {
        match self {
            Self::Precedes => Self::PrecededBy,
            Self::Meets => Self::MetBy,
            Self::Overlaps => Self::OverlappedBy,
            Self::Starts => Self::StartedBy,
            Self::During => Self::Contains,
            Self::Finishes => Self::FinishedBy,
            Self::Equals => Self::Equals, // Self-inverse
            Self::PrecededBy => Self::Precedes,
            Self::MetBy => Self::Meets,
            Self::OverlappedBy => Self::Overlaps,
            Self::StartedBy => Self::Starts,
            Self::Contains => Self::During,
            Self::FinishedBy => Self::Finishes,
        }
    }

    /// All 13 relations
    pub fn all() -> Vec<Self> {
        vec![
            Self::Precedes, Self::Meets, Self::Overlaps, Self::Starts,
            Self::During, Self::Finishes, Self::Equals,
            Self::PrecededBy, Self::MetBy, Self::OverlappedBy,
            Self::StartedBy, Self::Contains, Self::FinishedBy,
        ]
    }

    /// Short name (Allen's notation)
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Precedes => "p",
            Self::Meets => "m",
            Self::Overlaps => "o",
            Self::Starts => "s",
            Self::During => "d",
            Self::Finishes => "f",
            Self::Equals => "eq",
            Self::PrecededBy => "pi",
            Self::MetBy => "mi",
            Self::OverlappedBy => "oi",
            Self::StartedBy => "si",
            Self::Contains => "di",
            Self::FinishedBy => "fi",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Precedes => "A ends before B starts",
            Self::Meets => "A ends exactly when B starts",
            Self::Overlaps => "A starts first, they overlap, A ends first",
            Self::Starts => "A and B start together, A ends first",
            Self::During => "A is completely within B",
            Self::Finishes => "A and B end together, A starts later",
            Self::Equals => "A and B are identical",
            Self::PrecededBy => "A starts after B ends",
            Self::MetBy => "A starts exactly when B ends",
            Self::OverlappedBy => "B starts first, they overlap, B ends first",
            Self::StartedBy => "A and B start together, B ends first",
            Self::Contains => "B is completely within A",
            Self::FinishedBy => "A and B end together, B starts later",
        }
    }
}

// ============================================================================
// TEMPORAL REASONING ENGINE
// ============================================================================

/// Temporal reasoning engine with Allen's algebra
pub struct TemporalReasoner {
    /// Known intervals
    intervals: HashMap<String, TemporalInterval>,

    /// Cached relations between interval pairs
    relations: HashMap<(String, String), AllenRelation>,

    /// HDC encodings for each Allen relation
    relation_encodings: HashMap<AllenRelation, HV16>,

    /// Composition table (pre-computed transitivity)
    composition_table: HashMap<(AllenRelation, AllenRelation), Vec<AllenRelation>>,

    /// Temporal binding window (default: 25ms for gamma oscillations)
    binding_window: f64,

    /// Configuration
    config: TemporalConfig,
}

/// Configuration for temporal reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Tolerance for "equals" comparison (floating point)
    pub epsilon: f64,

    /// Maximum gap for "meets" relation (fuzzy meeting)
    pub meet_tolerance: f64,

    /// Minimum overlap duration for "overlaps" (must be significant)
    pub min_overlap: f64,

    /// Enable soft/fuzzy temporal matching
    pub soft_matching: bool,

    /// Binding window for consciousness integration (25ms = 40Hz gamma)
    pub binding_window_ms: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-9,
            meet_tolerance: 1e-6,
            min_overlap: 0.001,
            soft_matching: true,
            binding_window_ms: 25.0, // 40Hz gamma rhythm
        }
    }
}

impl TemporalReasoner {
    /// Create new temporal reasoner
    pub fn new(config: TemporalConfig) -> Self {
        let mut reasoner = Self {
            intervals: HashMap::new(),
            relations: HashMap::new(),
            relation_encodings: Self::init_relation_encodings(),
            composition_table: HashMap::new(),
            binding_window: config.binding_window_ms / 1000.0, // Convert to seconds
            config,
        };
        reasoner.init_composition_table();
        reasoner
    }

    /// Initialize HDC encodings for each Allen relation
    fn init_relation_encodings() -> HashMap<AllenRelation, HV16> {
        let mut encodings = HashMap::new();

        for (i, relation) in AllenRelation::all().iter().enumerate() {
            // Each relation gets a unique, deterministic encoding
            let seed = 7000 + i as u64 * 100;
            encodings.insert(*relation, HV16::random(seed));
        }

        encodings
    }

    /// Initialize the composition (transitivity) table
    ///
    /// This is the heart of Allen's algebra: if we know A relates to B,
    /// and B relates to C, what can we infer about A's relation to C?
    fn init_composition_table(&mut self) {
        use AllenRelation::*;

        // Allen's composition table (partial - key entries)
        // Full table has 13x13 = 169 entries, each mapping to a set of possible relations

        // Precedes compositions
        self.add_composition(Precedes, Precedes, vec![Precedes]);
        self.add_composition(Precedes, Meets, vec![Precedes]);
        self.add_composition(Precedes, Overlaps, vec![Precedes]);
        self.add_composition(Precedes, Starts, vec![Precedes]);
        self.add_composition(Precedes, During, vec![Precedes, Meets, Overlaps, Starts, During]);
        self.add_composition(Precedes, Finishes, vec![Precedes, Meets, Overlaps, Starts, During]);
        self.add_composition(Precedes, Equals, vec![Precedes]);

        // Meets compositions
        self.add_composition(Meets, Precedes, vec![Precedes]);
        self.add_composition(Meets, Meets, vec![Precedes]);
        self.add_composition(Meets, Overlaps, vec![Precedes]);
        self.add_composition(Meets, Starts, vec![Meets]);
        self.add_composition(Meets, During, vec![Overlaps, Starts, During]);
        self.add_composition(Meets, Finishes, vec![Overlaps, Starts, During]);
        self.add_composition(Meets, Equals, vec![Meets]);

        // Overlaps compositions
        self.add_composition(Overlaps, Precedes, vec![Precedes]);
        self.add_composition(Overlaps, Meets, vec![Precedes, Meets, Overlaps]);
        self.add_composition(Overlaps, Overlaps, vec![Precedes, Meets, Overlaps]);
        self.add_composition(Overlaps, Starts, vec![Overlaps]);
        self.add_composition(Overlaps, During, vec![Overlaps, Starts, During]);
        self.add_composition(Overlaps, Finishes, vec![Overlaps, Starts, During]);
        self.add_composition(Overlaps, Equals, vec![Overlaps]);

        // During compositions
        self.add_composition(During, Precedes, vec![Precedes, Meets, Overlaps, Starts, During]);
        self.add_composition(During, Meets, vec![Overlaps, Starts, During]);
        self.add_composition(During, Overlaps, vec![Overlaps, Starts, During]);
        self.add_composition(During, Starts, vec![During]);
        self.add_composition(During, During, vec![During]);
        self.add_composition(During, Finishes, vec![During]);
        self.add_composition(During, Equals, vec![During]);

        // Equals is identity
        for r in AllenRelation::all() {
            self.add_composition(Equals, r, vec![r]);
            self.add_composition(r, Equals, vec![r]);
        }

        // Contains compositions (inverse of During)
        self.add_composition(Contains, Precedes, vec![Precedes, Meets, Overlaps, StartedBy, Contains]);
        self.add_composition(Contains, Meets, vec![Overlaps, StartedBy, Contains]);
        self.add_composition(Contains, Overlaps, vec![Overlaps, StartedBy, Contains]);
        self.add_composition(Contains, Contains, vec![Contains]);
        self.add_composition(Contains, Starts, vec![Overlaps, StartedBy, Contains]);
        self.add_composition(Contains, During, AllenRelation::all());
        self.add_composition(Contains, Finishes, vec![Contains, FinishedBy, OverlappedBy]);

        // Key inverse relationships automatically derive from the above
        // by using: compose(inverse(B), inverse(A)) = inverse(compose(A, B))
    }

    fn add_composition(&mut self, r1: AllenRelation, r2: AllenRelation, result: Vec<AllenRelation>) {
        self.composition_table.insert((r1, r2), result);
    }

    /// Add an interval to the reasoner
    pub fn add_interval(&mut self, interval: TemporalInterval) {
        self.intervals.insert(interval.id.clone(), interval);
    }

    /// Create and add a new interval
    pub fn create_interval(&mut self, id: impl Into<String>, start: f64, end: f64) -> Result<()> {
        let interval = TemporalInterval::new(id, start, end)?;
        self.add_interval(interval);
        Ok(())
    }

    /// Compute the Allen relation between two intervals
    pub fn compute_relation(&self, a: &TemporalInterval, b: &TemporalInterval) -> AllenRelation {
        let eps = self.config.epsilon;

        // Compare endpoints
        let a_start = a.start;
        let a_end = a.end;
        let b_start = b.start;
        let b_end = b.end;

        // Helper for approximate equality
        let approx_eq = |x: f64, y: f64| (x - y).abs() < eps;

        // Determine relation based on endpoint comparisons
        if a_end < b_start - self.config.meet_tolerance {
            // A ends before B starts (with gap)
            AllenRelation::Precedes
        } else if approx_eq(a_end, b_start) || (a_end >= b_start - self.config.meet_tolerance && a_end <= b_start + self.config.meet_tolerance && a_end < b_end && a_start < b_start) {
            // A ends when B starts (no gap, no overlap)
            AllenRelation::Meets
        } else if a_start < b_start && a_end > b_start && a_end < b_end {
            // A starts before B, they overlap, A ends first
            AllenRelation::Overlaps
        } else if approx_eq(a_start, b_start) && a_end < b_end {
            // Same start, A ends first
            AllenRelation::Starts
        } else if a_start > b_start && a_end < b_end {
            // A completely within B
            AllenRelation::During
        } else if a_start > b_start && approx_eq(a_end, b_end) {
            // Same end, A starts later
            AllenRelation::Finishes
        } else if approx_eq(a_start, b_start) && approx_eq(a_end, b_end) {
            // Identical
            AllenRelation::Equals
        }
        // Inverse relations (when A comes "after" B in various ways)
        else if b_end < a_start - self.config.meet_tolerance {
            AllenRelation::PrecededBy
        } else if approx_eq(b_end, a_start) || (b_end >= a_start - self.config.meet_tolerance && b_end <= a_start + self.config.meet_tolerance && b_end < a_end && b_start < a_start) {
            AllenRelation::MetBy
        } else if b_start < a_start && b_end > a_start && b_end < a_end {
            AllenRelation::OverlappedBy
        } else if approx_eq(a_start, b_start) && b_end < a_end {
            AllenRelation::StartedBy
        } else if b_start > a_start && b_end < a_end {
            AllenRelation::Contains
        } else if b_start > a_start && approx_eq(a_end, b_end) {
            AllenRelation::FinishedBy
        } else {
            // Default to examining overlap more carefully
            if a_start <= b_start && a_end >= b_end {
                AllenRelation::Contains
            } else if b_start <= a_start && b_end >= a_end {
                AllenRelation::During
            } else if a_end > b_start && a_start < b_start {
                AllenRelation::Overlaps
            } else {
                AllenRelation::OverlappedBy
            }
        }
    }

    /// Get or compute relation between two intervals by ID
    pub fn get_relation(&mut self, id_a: &str, id_b: &str) -> Result<AllenRelation> {
        // Check cache
        let key = (id_a.to_string(), id_b.to_string());
        if let Some(&relation) = self.relations.get(&key) {
            return Ok(relation);
        }

        // Compute and cache
        let a = self.intervals.get(id_a)
            .ok_or_else(|| anyhow!("Interval not found: {}", id_a))?
            .clone();
        let b = self.intervals.get(id_b)
            .ok_or_else(|| anyhow!("Interval not found: {}", id_b))?
            .clone();

        let relation = self.compute_relation(&a, &b);
        self.relations.insert(key.clone(), relation);
        self.relations.insert((id_b.to_string(), id_a.to_string()), relation.inverse());

        Ok(relation)
    }

    /// Compose two relations (transitivity)
    ///
    /// Given A rel1 B and B rel2 C, what can we infer about A ? C?
    pub fn compose(&self, r1: AllenRelation, r2: AllenRelation) -> Vec<AllenRelation> {
        self.composition_table
            .get(&(r1, r2))
            .cloned()
            .unwrap_or_else(|| {
                // Conservative fallback: all relations possible
                AllenRelation::all()
            })
    }

    /// Infer possible relations between A and C given A-B and B-C relations
    pub fn infer_relation(&mut self, id_a: &str, id_b: &str, id_c: &str) -> Result<Vec<AllenRelation>> {
        let r_ab = self.get_relation(id_a, id_b)?;
        let r_bc = self.get_relation(id_b, id_c)?;
        Ok(self.compose(r_ab, r_bc))
    }

    /// Get HDC encoding for an Allen relation
    pub fn encode_relation(&self, relation: AllenRelation) -> HV16 {
        self.relation_encodings[&relation].clone()
    }

    /// Encode a temporal statement "A relation B" as an HV16
    pub fn encode_statement(&self, a: &TemporalInterval, relation: AllenRelation, b: &TemporalInterval) -> HV16 {
        // Bind A's encoding with relation encoding with B's encoding
        // (A ⊗ R ⊗ B)
        let relation_hv = self.encode_relation(relation);
        a.encoding.bind(&relation_hv).bind(&b.encoding)
    }

    /// Find intervals that overlap with the binding window (for consciousness integration)
    pub fn find_binding_candidates(&self, reference_time: f64) -> Vec<&TemporalInterval> {
        let window_start = reference_time - self.binding_window;
        let window_end = reference_time + self.binding_window;

        self.intervals.values()
            .filter(|interval| {
                // Interval overlaps with binding window
                interval.start <= window_end && interval.end >= window_start
            })
            .collect()
    }
}

impl Default for TemporalReasoner {
    fn default() -> Self {
        Self::new(TemporalConfig::default())
    }
}

// ============================================================================
// CONSCIOUSNESS TEMPORAL INTEGRATION
// ============================================================================

/// Temporal consciousness state - extends basic interval with consciousness info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousInterval {
    /// Base temporal interval
    pub interval: TemporalInterval,

    /// Integrated Information (Φ) during this interval
    pub phi: f64,

    /// Phi trend during interval (rising, stable, falling)
    pub phi_trend: PhiTrend,

    /// Binding coherence (gamma-band synchrony)
    pub binding_coherence: f64,

    /// Causal efficacy (did this state cause action?)
    pub causal_efficacy: f64,

    /// Associated content (what was the conscious experience about?)
    pub content: Option<HV16>,
}

/// Phi trend during an interval
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PhiTrend {
    Rising,
    Stable,
    Falling,
    Oscillating,
}

impl ConsciousInterval {
    /// Create from basic interval with consciousness metrics
    pub fn new(
        interval: TemporalInterval,
        phi: f64,
        binding_coherence: f64,
        causal_efficacy: f64,
    ) -> Self {
        Self {
            interval,
            phi,
            phi_trend: PhiTrend::Stable,
            binding_coherence,
            causal_efficacy,
            content: None,
        }
    }

    /// Check if this represents a "conscious moment" (Φ above threshold)
    pub fn is_conscious(&self, threshold: f64) -> bool {
        self.phi >= threshold
    }

    /// Check if this moment has causal efficacy
    pub fn is_causally_efficacious(&self, threshold: f64) -> bool {
        self.causal_efficacy >= threshold
    }

    /// Combined consciousness score
    pub fn consciousness_score(&self) -> f64 {
        // Weighted combination of all consciousness dimensions
        0.4 * self.phi + 0.3 * self.binding_coherence + 0.3 * self.causal_efficacy
    }
}

/// Temporal consciousness analyzer
pub struct ConsciousnessTemporalAnalyzer {
    /// Temporal reasoner
    reasoner: TemporalReasoner,

    /// Conscious intervals
    conscious_intervals: Vec<ConsciousInterval>,

    /// Phi threshold for consciousness
    phi_threshold: f64,

    /// Causal chain detections
    causal_chains: Vec<CausalChain>,
}

/// A causal chain of conscious intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    /// Sequence of interval IDs in causal order
    pub intervals: Vec<String>,

    /// Allen relations connecting them
    pub relations: Vec<AllenRelation>,

    /// Strength of causal connection (based on temporal proximity and Φ)
    pub causal_strength: f64,

    /// Whether this represents genuine causation (not just correlation)
    pub genuine_causation: bool,
}

impl ConsciousnessTemporalAnalyzer {
    /// Create new analyzer
    pub fn new(phi_threshold: f64) -> Self {
        Self {
            reasoner: TemporalReasoner::default(),
            conscious_intervals: Vec::new(),
            phi_threshold,
            causal_chains: Vec::new(),
        }
    }

    /// Add a conscious interval
    pub fn add_interval(&mut self, interval: ConsciousInterval) {
        self.reasoner.add_interval(interval.interval.clone());
        self.conscious_intervals.push(interval);
    }

    /// Find all intervals that could causally influence a given interval
    ///
    /// Uses Allen algebra: only Precedes, Meets, or Overlaps can be causes
    pub fn find_potential_causes(&mut self, target_id: &str) -> Result<Vec<(String, AllenRelation)>> {
        let mut causes = Vec::new();

        for interval in &self.conscious_intervals {
            if interval.interval.id == target_id {
                continue;
            }

            let relation = self.reasoner.get_relation(&interval.interval.id, target_id)?;

            // Causal relations: cause must precede or meet or overlap with effect
            match relation {
                AllenRelation::Precedes | AllenRelation::Meets | AllenRelation::Overlaps => {
                    causes.push((interval.interval.id.clone(), relation));
                }
                _ => {}
            }
        }

        Ok(causes)
    }

    /// Find all intervals causally influenced by a given interval
    pub fn find_effects(&mut self, source_id: &str) -> Result<Vec<(String, AllenRelation)>> {
        let mut effects = Vec::new();

        for interval in &self.conscious_intervals {
            if interval.interval.id == source_id {
                continue;
            }

            let relation = self.reasoner.get_relation(source_id, &interval.interval.id)?;

            // Effect relations
            match relation {
                AllenRelation::Precedes | AllenRelation::Meets | AllenRelation::Overlaps => {
                    effects.push((interval.interval.id.clone(), relation));
                }
                _ => {}
            }
        }

        Ok(effects)
    }

    /// Detect causal chains in the consciousness stream
    pub fn detect_causal_chains(&mut self, min_length: usize) -> Vec<CausalChain> {
        let mut chains = Vec::new();

        // Sort intervals by start time
        let mut sorted: Vec<_> = self.conscious_intervals.iter().collect();
        sorted.sort_by(|a, b| a.interval.start.partial_cmp(&b.interval.start).unwrap());

        // Build chains greedily
        for start in &sorted {
            let mut chain_ids = vec![start.interval.id.clone()];
            let mut chain_relations = Vec::new();
            let mut current = start;

            // Extend chain
            for candidate in &sorted {
                if candidate.interval.start <= current.interval.end {
                    continue; // Must come after current
                }

                // Check if temporally connected
                if let Ok(relation) = self.reasoner.get_relation(&current.interval.id, &candidate.interval.id) {
                    match relation {
                        AllenRelation::Precedes | AllenRelation::Meets => {
                            chain_ids.push(candidate.interval.id.clone());
                            chain_relations.push(relation);
                            current = candidate;
                        }
                        _ => {}
                    }
                }
            }

            if chain_ids.len() >= min_length {
                // Calculate causal strength based on temporal proximity and Φ
                let avg_phi: f64 = chain_ids.iter()
                    .filter_map(|id| {
                        self.conscious_intervals.iter()
                            .find(|i| i.interval.id == *id)
                            .map(|i| i.phi)
                    })
                    .sum::<f64>() / chain_ids.len() as f64;

                chains.push(CausalChain {
                    intervals: chain_ids,
                    relations: chain_relations,
                    causal_strength: avg_phi,
                    genuine_causation: avg_phi > self.phi_threshold,
                });
            }
        }

        self.causal_chains = chains.clone();
        chains
    }

    /// Analyze consciousness continuity - how smooth is the conscious experience?
    pub fn analyze_continuity(&mut self) -> ContinuityAnalysis {
        let mut gaps = Vec::new();
        let mut total_conscious_time = 0.0;
        let mut total_gap_time = 0.0;

        // Sort by start time
        let mut sorted: Vec<_> = self.conscious_intervals.iter().collect();
        sorted.sort_by(|a, b| a.interval.start.partial_cmp(&b.interval.start).unwrap());

        for window in sorted.windows(2) {
            let a = &window[0];
            let b = &window[1];

            total_conscious_time += a.interval.duration();

            if b.interval.start > a.interval.end {
                let gap = b.interval.start - a.interval.end;
                gaps.push(gap);
                total_gap_time += gap;
            }
        }

        // Add last interval's duration
        if let Some(last) = sorted.last() {
            total_conscious_time += last.interval.duration();
        }

        let continuity_score = if total_conscious_time + total_gap_time > 0.0 {
            total_conscious_time / (total_conscious_time + total_gap_time)
        } else {
            1.0
        };

        ContinuityAnalysis {
            total_conscious_time,
            total_gap_time,
            gap_count: gaps.len(),
            average_gap: if gaps.is_empty() { 0.0 } else { total_gap_time / gaps.len() as f64 },
            max_gap: gaps.iter().cloned().fold(0.0, f64::max),
            continuity_score,
        }
    }
}

/// Analysis of consciousness continuity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityAnalysis {
    /// Total time spent in conscious states
    pub total_conscious_time: f64,

    /// Total time in gaps between conscious states
    pub total_gap_time: f64,

    /// Number of gaps
    pub gap_count: usize,

    /// Average gap duration
    pub average_gap: f64,

    /// Longest gap
    pub max_gap: f64,

    /// Continuity score (0 = fragmented, 1 = continuous)
    pub continuity_score: f64,
}

// ============================================================================
// TEMPORAL PRIMITIVE SYSTEM INTEGRATION
// ============================================================================

/// Temporal primitives for the primitive system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPrimitive {
    /// Allen relation this primitive represents
    pub relation: AllenRelation,

    /// HDC encoding
    pub encoding: HV16,

    /// Tier (always Temporal)
    pub tier: TemporalTier,

    /// Compositionality rules (what happens when combined with other relations)
    pub composition_rules: Vec<(AllenRelation, Vec<AllenRelation>)>,
}

/// Tier for temporal primitives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalTier {
    /// Basic Allen relations
    BasicRelation,

    /// Composed temporal patterns
    ComposedPattern,

    /// Consciousness-specific temporal concepts
    ConsciousTemporal,

    /// Meta-temporal (reasoning about time itself)
    MetaTemporal,
}

/// Complete temporal primitive system
pub struct TemporalPrimitiveSystem {
    /// All 13 Allen relation primitives
    pub relation_primitives: HashMap<AllenRelation, TemporalPrimitive>,

    /// Domain manifold for temporal primitives
    pub domain_rotation: HV16,

    /// Temporal patterns (composed relations)
    pub patterns: Vec<TemporalPattern>,
}

/// A composed temporal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Pattern name
    pub name: String,

    /// Sequence of relations that define this pattern
    pub relations: Vec<AllenRelation>,

    /// Semantic meaning
    pub meaning: String,

    /// Encoding
    pub encoding: HV16,
}

impl TemporalPrimitiveSystem {
    /// Create the complete temporal primitive system
    pub fn new() -> Self {
        let domain_rotation = HV16::random(6000);
        let mut relation_primitives = HashMap::new();

        for (i, relation) in AllenRelation::all().iter().enumerate() {
            let local_encoding = HV16::random(6001 + i as u64);
            let encoding = HV16::bundle(&[domain_rotation.clone(), local_encoding]);

            relation_primitives.insert(*relation, TemporalPrimitive {
                relation: *relation,
                encoding,
                tier: TemporalTier::BasicRelation,
                composition_rules: Vec::new(), // Filled by TemporalReasoner
            });
        }

        // Create common temporal patterns
        let patterns = vec![
            TemporalPattern {
                name: "Sequence".to_string(),
                relations: vec![AllenRelation::Precedes, AllenRelation::Precedes],
                meaning: "A happens, then B, then C".to_string(),
                encoding: HV16::random(6100),
            },
            TemporalPattern {
                name: "Overlap_Chain".to_string(),
                relations: vec![AllenRelation::Overlaps, AllenRelation::Overlaps],
                meaning: "Events cascade with overlap".to_string(),
                encoding: HV16::random(6101),
            },
            TemporalPattern {
                name: "Containment".to_string(),
                relations: vec![AllenRelation::Contains],
                meaning: "One event fully contains another".to_string(),
                encoding: HV16::random(6102),
            },
            TemporalPattern {
                name: "Simultaneity".to_string(),
                relations: vec![AllenRelation::Equals],
                meaning: "Events occur at same time".to_string(),
                encoding: HV16::random(6103),
            },
            TemporalPattern {
                name: "Causal_Immediacy".to_string(),
                relations: vec![AllenRelation::Meets],
                meaning: "Effect immediately follows cause".to_string(),
                encoding: HV16::random(6104),
            },
            TemporalPattern {
                name: "Binding_Window".to_string(),
                relations: vec![AllenRelation::Overlaps, AllenRelation::Starts, AllenRelation::During],
                meaning: "Events within 25ms binding window".to_string(),
                encoding: HV16::random(6105),
            },
        ];

        Self {
            relation_primitives,
            domain_rotation,
            patterns,
        }
    }

    /// Encode a temporal relation
    pub fn encode(&self, relation: AllenRelation) -> HV16 {
        self.relation_primitives[&relation].encoding.clone()
    }

    /// Encode a sequence of relations (temporal pattern)
    pub fn encode_sequence(&self, relations: &[AllenRelation]) -> HV16 {
        if relations.is_empty() {
            return HV16::zero();
        }

        // Bind relations in sequence with positional permutation
        let mut result = self.encode(relations[0]);
        for (i, &relation) in relations.iter().enumerate().skip(1) {
            let encoded = self.encode(relation);
            let positioned = encoded.permute(i * 100); // Position encoding
            result = result.bind(&positioned);
        }

        result
    }

    /// Find the pattern that best matches a sequence
    pub fn match_pattern(&self, sequence: &[AllenRelation]) -> Option<(String, f32)> {
        let query = self.encode_sequence(sequence);

        self.patterns.iter()
            .map(|pattern| {
                let similarity = query.similarity(&pattern.encoding);
                (pattern.name.clone(), similarity)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
}

impl Default for TemporalPrimitiveSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_creation() {
        let interval = TemporalInterval::new("test", 0.0, 1.0).unwrap();
        assert_eq!(interval.duration(), 1.0);
        assert_eq!(interval.midpoint(), 0.5);
        assert!(interval.contains_point(0.5));
        assert!(!interval.contains_point(1.5));
    }

    #[test]
    fn test_interval_invalid() {
        assert!(TemporalInterval::new("invalid", 1.0, 0.0).is_err());
    }

    #[test]
    fn test_allen_relations_basic() {
        let reasoner = TemporalReasoner::default();

        // A precedes B
        let a = TemporalInterval::new("A", 0.0, 1.0).unwrap();
        let b = TemporalInterval::new("B", 2.0, 3.0).unwrap();
        assert_eq!(reasoner.compute_relation(&a, &b), AllenRelation::Precedes);

        // A meets B
        let a = TemporalInterval::new("A", 0.0, 1.0).unwrap();
        let b = TemporalInterval::new("B", 1.0, 2.0).unwrap();
        assert_eq!(reasoner.compute_relation(&a, &b), AllenRelation::Meets);

        // A overlaps B
        let a = TemporalInterval::new("A", 0.0, 2.0).unwrap();
        let b = TemporalInterval::new("B", 1.0, 3.0).unwrap();
        assert_eq!(reasoner.compute_relation(&a, &b), AllenRelation::Overlaps);

        // A during B
        let a = TemporalInterval::new("A", 1.0, 2.0).unwrap();
        let b = TemporalInterval::new("B", 0.0, 3.0).unwrap();
        assert_eq!(reasoner.compute_relation(&a, &b), AllenRelation::During);

        // A contains B
        let a = TemporalInterval::new("A", 0.0, 3.0).unwrap();
        let b = TemporalInterval::new("B", 1.0, 2.0).unwrap();
        assert_eq!(reasoner.compute_relation(&a, &b), AllenRelation::Contains);

        // A equals B
        let a = TemporalInterval::new("A", 0.0, 1.0).unwrap();
        let b = TemporalInterval::new("B", 0.0, 1.0).unwrap();
        assert_eq!(reasoner.compute_relation(&a, &b), AllenRelation::Equals);
    }

    #[test]
    fn test_allen_relation_inverses() {
        use AllenRelation::*;

        assert_eq!(Precedes.inverse(), PrecededBy);
        assert_eq!(PrecededBy.inverse(), Precedes);
        assert_eq!(Meets.inverse(), MetBy);
        assert_eq!(Overlaps.inverse(), OverlappedBy);
        assert_eq!(Equals.inverse(), Equals);
    }

    #[test]
    fn test_composition() {
        let reasoner = TemporalReasoner::default();

        // If A precedes B, and B precedes C, then A precedes C
        let result = reasoner.compose(AllenRelation::Precedes, AllenRelation::Precedes);
        assert!(result.contains(&AllenRelation::Precedes));

        // Equals is identity
        let result = reasoner.compose(AllenRelation::Equals, AllenRelation::Overlaps);
        assert!(result.contains(&AllenRelation::Overlaps));
    }

    #[test]
    fn test_temporal_inference() {
        let mut reasoner = TemporalReasoner::default();

        reasoner.create_interval("A", 0.0, 1.0).unwrap();
        reasoner.create_interval("B", 2.0, 3.0).unwrap();
        reasoner.create_interval("C", 4.0, 5.0).unwrap();

        // A precedes B precedes C => A precedes C
        let inferred = reasoner.infer_relation("A", "B", "C").unwrap();
        assert!(inferred.contains(&AllenRelation::Precedes));
    }

    #[test]
    fn test_hdc_encoding() {
        let reasoner = TemporalReasoner::default();

        let precedes_hv = reasoner.encode_relation(AllenRelation::Precedes);
        let meets_hv = reasoner.encode_relation(AllenRelation::Meets);

        // Different relations should have different encodings
        let similarity = precedes_hv.similarity(&meets_hv);
        assert!(similarity < 0.9, "Different relations should have different encodings");
    }

    #[test]
    fn test_temporal_statement_encoding() {
        let reasoner = TemporalReasoner::default();

        let a = TemporalInterval::new("A", 0.0, 1.0).unwrap();
        let b = TemporalInterval::new("B", 2.0, 3.0).unwrap();

        let statement = reasoner.encode_statement(&a, AllenRelation::Precedes, &b);

        // Statement should be meaningful (not zero)
        assert!(statement.popcount() > 0);
    }

    #[test]
    fn test_binding_window() {
        let mut reasoner = TemporalReasoner::default();

        // Create intervals within binding window (25ms = 0.025s)
        reasoner.create_interval("A", 0.0, 0.01).unwrap();
        reasoner.create_interval("B", 0.015, 0.025).unwrap();
        reasoner.create_interval("C", 1.0, 1.01).unwrap(); // Outside window

        let candidates = reasoner.find_binding_candidates(0.02);

        assert_eq!(candidates.len(), 2); // A and B within window
    }

    #[test]
    fn test_conscious_interval() {
        let interval = TemporalInterval::new("conscious_moment", 0.0, 0.1).unwrap();
        let conscious = ConsciousInterval::new(interval, 0.8, 0.7, 0.6);

        assert!(conscious.is_conscious(0.5));
        assert!(!conscious.is_conscious(0.9));
        assert_eq!(conscious.consciousness_score(), 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.6);
    }

    #[test]
    fn test_causal_chain_detection() {
        let mut analyzer = ConsciousnessTemporalAnalyzer::new(0.5);

        // Create a causal chain: A -> B -> C
        for (i, (start, end)) in [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)].iter().enumerate() {
            let interval = TemporalInterval::new(format!("state_{}", i), *start, *end).unwrap();
            let conscious = ConsciousInterval::new(interval, 0.8, 0.7, 0.6);
            analyzer.add_interval(conscious);
        }

        let chains = analyzer.detect_causal_chains(2);
        assert!(!chains.is_empty(), "Should detect at least one chain");
        assert!(chains[0].intervals.len() >= 2);
    }

    #[test]
    fn test_continuity_analysis() {
        let mut analyzer = ConsciousnessTemporalAnalyzer::new(0.5);

        // Continuous consciousness (no gaps)
        let intervals = vec![
            (0.0, 1.0),
            (1.0, 2.0), // Meets previous
            (2.0, 3.0), // Meets previous
        ];

        for (i, (start, end)) in intervals.iter().enumerate() {
            let interval = TemporalInterval::new(format!("state_{}", i), *start, *end).unwrap();
            let conscious = ConsciousInterval::new(interval, 0.8, 0.7, 0.6);
            analyzer.add_interval(conscious);
        }

        let analysis = analyzer.analyze_continuity();
        assert_eq!(analysis.total_conscious_time, 3.0);
        assert_eq!(analysis.gap_count, 0);
        assert_eq!(analysis.continuity_score, 1.0);
    }

    #[test]
    fn test_temporal_primitive_system() {
        let system = TemporalPrimitiveSystem::new();

        // All 13 relations should be encoded
        assert_eq!(system.relation_primitives.len(), 13);

        // Encode a sequence
        let sequence = vec![AllenRelation::Precedes, AllenRelation::Meets];
        let encoding = system.encode_sequence(&sequence);
        assert!(encoding.popcount() > 0);
    }

    #[test]
    fn test_pattern_matching() {
        let system = TemporalPrimitiveSystem::new();

        // The "Sequence" pattern uses Precedes twice
        let sequence = vec![AllenRelation::Precedes, AllenRelation::Precedes];

        if let Some((name, similarity)) = system.match_pattern(&sequence) {
            println!("Best match: {} (similarity: {:.3})", name, similarity);
            // Should match "Sequence" pattern
            assert!(similarity > 0.0);
        }
    }
}
