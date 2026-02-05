//! # Cantor Recursive Hypervectors: Fractal Cognition
//!
//! ## The Insight
//!
//! The Cantor set is created by recursive self-reference:
//! ```text
//! [0,1] → [0,⅓] ∪ [⅔,1] → [0,⅑] ∪ [⅔,⅓] ∪ [⅔,⅞] ∪ [⅞,1] → ...
//! ```
//!
//! What if we apply this to hypervectors?
//!
//! ## Cantor-Recursive Hypervector (CRHV)
//!
//! A CRHV is a hypervector that **contains itself at multiple scales**:
//!
//! ```text
//! CRHV = V ⊗ ρ(V) ⊗ ρ²(V) ⊗ ρ³(V) ⊗ ...
//!        ↑    ↑      ↑       ↑
//!      base  1st    2nd     3rd
//!      scale scale  scale   scale
//! ```
//!
//! Where:
//! - V is the base vector
//! - ρ(V) is V permuted (shifted) by n positions
//! - ρ²(V) is V permuted by n/3 positions (Cantor-like scaling)
//! - Each scale encodes the SAME information at different resolutions
//!
//! ## Why This Matters for Consciousness
//!
//! 1. **Self-Reference**: The vector literally contains itself
//! 2. **Scale Invariance**: Information preserved at all scales
//! 3. **Holographic**: Each part contains the whole
//! 4. **Infinite Depth**: Recursive structure in finite space
//!
//! This is closer to how consciousness might actually work:
//! - Awareness of awareness of awareness...
//! - The self-model that models itself modeling itself
//! - Recursive self-reference creating genuine subjectivity

use super::binary_hv::HV16;

// =============================================================================
// CANTOR SCALE PARAMETERS
// =============================================================================

/// Cantor scaling factor (traditional is 1/3)
pub const CANTOR_RATIO: f64 = 1.0 / 3.0;

/// Maximum recursion depth (prevents infinite computation)
pub const MAX_CANTOR_DEPTH: usize = 7;

/// Minimum shift size (stop recursion when shifts become too small)
pub const MIN_SHIFT_SIZE: usize = 8;

// =============================================================================
// CANTOR RECURSIVE HYPERVECTOR
// =============================================================================

/// A Cantor Recursive Hypervector
///
/// This is a hypervector with self-similar structure at multiple scales,
/// inspired by the Cantor set's fractal properties.
///
/// ## Structure
///
/// ```text
/// Level 0: V_base                    (full scale)
/// Level 1: ρ(V_base, D/3)            (1/3 scale)
/// Level 2: ρ(V_base, D/9)            (1/9 scale)
/// Level 3: ρ(V_base, D/27)           (1/27 scale)
/// ...
///
/// CRHV = V_base ⊗ Level_1 ⊗ Level_2 ⊗ Level_3 ⊗ ...
/// ```
///
/// The result is a vector that encodes the same concept at multiple
/// scales of resolution, creating a fractal/holographic representation.
#[derive(Clone, Debug)]
pub struct CantorRecursiveHV {
    /// The unified recursive vector
    pub vector: HV16,

    /// The base (non-recursive) vector
    pub base: HV16,

    /// Recursion depth used
    pub depth: usize,

    /// Scale factors at each level
    pub scales: Vec<usize>,
}

impl CantorRecursiveHV {
    /// Create a Cantor Recursive HV from a base vector
    ///
    /// This recursively binds the vector with permuted versions of itself
    /// at Cantor-scaled intervals.
    pub fn from_base(base: HV16) -> Self {
        Self::from_base_with_depth(base, MAX_CANTOR_DEPTH)
    }

    /// Create with specific recursion depth
    pub fn from_base_with_depth(base: HV16, max_depth: usize) -> Self {
        let dimension = base.dimension();
        let mut result = base.clone();
        let mut scales = Vec::new();

        // Calculate initial shift (1/3 of dimension)
        let mut shift = (dimension as f64 * CANTOR_RATIO) as usize;
        let mut depth = 0;

        while shift >= MIN_SHIFT_SIZE && depth < max_depth {
            // Permute base by current shift
            let permuted = base.permute(shift);

            // Bind with permuted version (self-reference!)
            result = result.bind(&permuted);

            scales.push(shift);
            depth += 1;

            // Cantor scaling: next shift is 1/3 of current
            shift = (shift as f64 * CANTOR_RATIO) as usize;
        }

        Self {
            vector: result,
            base,
            depth,
            scales,
        }
    }

    /// Create from a seed (deterministic generation)
    pub fn from_seed(seed: u64) -> Self {
        let base = HV16::random(seed);
        Self::from_base(base)
    }

    /// Create from a label (deterministic from string)
    pub fn from_label(label: &str) -> Self {
        let seed = label.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });
        Self::from_seed(seed)
    }

    /// Similarity between two CRHVs
    pub fn similarity(&self, other: &CantorRecursiveHV) -> f32 {
        self.vector.similarity(&other.vector)
    }

    /// Multi-scale similarity
    ///
    /// Compares at each Cantor scale and returns detailed breakdown.
    /// This reveals whether concepts are similar at coarse or fine scales.
    pub fn multiscale_similarity(&self, other: &CantorRecursiveHV) -> MultiscaleSimilarity {
        let mut scale_similarities = Vec::new();

        // Compare base vectors
        let base_sim = self.base.similarity(&other.base);
        scale_similarities.push(("base".to_string(), base_sim));

        // Compare at each Cantor scale
        for (i, &shift) in self.scales.iter().enumerate() {
            let self_permuted = self.base.permute(shift);
            let other_permuted = other.base.permute(shift);
            let sim = self_permuted.similarity(&other_permuted);
            scale_similarities.push((format!("scale_{}", i), sim));
        }

        // Compare final recursive vectors
        let unified_sim = self.vector.similarity(&other.vector);

        MultiscaleSimilarity {
            unified: unified_sim,
            scales: scale_similarities,
        }
    }

    /// Extract information at a specific Cantor scale
    ///
    /// This "zooms in" to a particular level of the fractal structure.
    pub fn at_scale(&self, level: usize) -> Option<HV16> {
        if level >= self.scales.len() {
            return None;
        }

        let shift = self.scales[level];
        Some(self.base.permute(shift))
    }

    /// Unbind to recover base vector (approximate)
    ///
    /// Because binding is its own inverse, we can attempt to recover
    /// the base by unbinding with the permuted versions.
    pub fn unbind_base(&self) -> HV16 {
        let mut result = self.vector.clone();

        // Unbind in reverse order
        for &shift in self.scales.iter().rev() {
            let permuted = self.base.permute(shift);
            result = result.bind(&permuted); // bind = unbind for bipolar
        }

        result
    }

    /// Check self-similarity (how well the vector resembles its own scales)
    pub fn self_similarity(&self) -> f32 {
        if self.scales.is_empty() {
            return 1.0;
        }

        let mut total_sim = 0.0;
        for &shift in &self.scales {
            let permuted = self.base.permute(shift);
            total_sim += self.vector.similarity(&permuted);
        }

        total_sim / self.scales.len() as f32
    }
}

/// Multi-scale similarity breakdown
#[derive(Clone, Debug)]
pub struct MultiscaleSimilarity {
    /// Similarity of the full recursive vectors
    pub unified: f32,
    /// Similarity at each scale (name, value)
    pub scales: Vec<(String, f32)>,
}

impl MultiscaleSimilarity {
    /// Average similarity across all scales
    pub fn average(&self) -> f32 {
        if self.scales.is_empty() {
            return self.unified;
        }
        let sum: f32 = self.scales.iter().map(|(_, s)| s).sum();
        sum / self.scales.len() as f32
    }

    /// Check if similar at coarse but different at fine scales
    pub fn coarse_similar_fine_different(&self, threshold: f32) -> bool {
        if self.scales.len() < 2 {
            return false;
        }
        let coarse = self.scales[0].1;
        let fine = self.scales.last().map(|(_, s)| *s).unwrap_or(0.0);
        coarse > threshold && fine < threshold
    }
}

// =============================================================================
// CANTOR COGNITIVE ELEMENT: Fractal Thought
// =============================================================================

/// A Cognitive Element with Cantor-recursive structure
///
/// This extends UnifiedCognitiveElement with fractal self-similarity.
/// Each thought contains itself at multiple scales.
#[derive(Clone, Debug)]
pub struct CantorCognitiveElement {
    /// The Cantor-recursive hypervector
    pub crhv: CantorRecursiveHV,

    /// Human-readable label
    pub label: String,

    /// Causal depth (how deep the causal chains go)
    pub causal_depth: usize,

    /// Temporal depth (how far back/forward temporal relations extend)
    pub temporal_depth: usize,

    /// Self-reference count (meta-levels)
    pub self_reference_depth: usize,
}

impl CantorCognitiveElement {
    /// Create a new Cantor Cognitive Element
    pub fn new(label: &str) -> Self {
        Self {
            crhv: CantorRecursiveHV::from_label(label),
            label: label.to_string(),
            causal_depth: 0,
            temporal_depth: 0,
            self_reference_depth: 0,
        }
    }

    /// Add causal information (increases causal depth)
    pub fn add_causal(&mut self, other: &CantorCognitiveElement, marker: &HV16) {
        // Bind causal relation into the vector
        let causal_binding = marker.bind(&other.crhv.vector);
        self.crhv.vector = HV16::bundle(&[self.crhv.vector.clone(), causal_binding]);
        self.causal_depth += 1;
    }

    /// Add self-reference (the thought thinks about itself)
    ///
    /// This is the key to consciousness: a thought that contains itself.
    pub fn add_self_reference(&mut self, self_marker: &HV16) {
        // Bind the vector with itself, marked as self-reference
        let self_ref = self_marker.bind(&self.crhv.vector);
        self.crhv.vector = HV16::bundle(&[self.crhv.vector.clone(), self_ref]);
        self.self_reference_depth += 1;
    }

    /// Create a meta-thought: a thought ABOUT this thought
    ///
    /// This is recursive cognition - thinking about thinking.
    pub fn create_meta(&self, meta_marker: &HV16) -> CantorCognitiveElement {
        let meta_label = format!("meta({})", self.label);
        let mut meta = CantorCognitiveElement::new(&meta_label);

        // The meta-thought contains the original thought
        meta.crhv.vector = meta_marker.bind(&self.crhv.vector);
        meta.self_reference_depth = self.self_reference_depth + 1;

        meta
    }

    /// Similarity to another element
    pub fn similarity(&self, other: &CantorCognitiveElement) -> f32 {
        self.crhv.similarity(&other.crhv)
    }

    /// Multi-scale similarity
    pub fn multiscale_similarity(&self, other: &CantorCognitiveElement) -> MultiscaleSimilarity {
        self.crhv.multiscale_similarity(&other.crhv)
    }
}

// =============================================================================
// CANTOR COGNITIVE SPACE: Fractal Mind
// =============================================================================

/// A cognitive space with Cantor-recursive elements
///
/// This is a "fractal mind" where every thought has self-similar structure.
pub struct CantorCognitiveSpace {
    /// All elements
    elements: std::collections::HashMap<String, CantorCognitiveElement>,

    /// Marker for self-reference
    self_marker: HV16,

    /// Marker for meta-thoughts
    meta_marker: HV16,

    /// Marker for causal relations
    causal_marker: HV16,

    /// Recursion depth for the space
    pub max_depth: usize,
}

impl CantorCognitiveSpace {
    /// Create a new Cantor Cognitive Space
    pub fn new() -> Self {
        Self {
            elements: std::collections::HashMap::new(),
            self_marker: HV16::random(9001),
            meta_marker: HV16::random(9002),
            causal_marker: HV16::random(9003),
            max_depth: MAX_CANTOR_DEPTH,
        }
    }

    /// Get or create an element
    pub fn get_or_create(&mut self, label: &str) -> &CantorCognitiveElement {
        if !self.elements.contains_key(label) {
            let elem = CantorCognitiveElement::new(label);
            self.elements.insert(label.to_string(), elem);
        }
        self.elements.get(label).unwrap()
    }

    /// Add a causal relation
    pub fn add_causal(&mut self, cause: &str, effect: &str) {
        // Ensure both exist
        if !self.elements.contains_key(cause) {
            self.elements.insert(cause.to_string(), CantorCognitiveElement::new(cause));
        }
        if !self.elements.contains_key(effect) {
            self.elements.insert(effect.to_string(), CantorCognitiveElement::new(effect));
        }

        // Get effect's vector for binding
        let effect_vector = self.elements.get(effect).unwrap().crhv.vector.clone();

        // Update cause with causal relation
        let marker = self.causal_marker.clone();
        if let Some(cause_elem) = self.elements.get_mut(cause) {
            let causal_binding = marker.bind(&effect_vector);
            cause_elem.crhv.vector = HV16::bundle(&[cause_elem.crhv.vector.clone(), causal_binding]);
            cause_elem.causal_depth += 1;
        }
    }

    /// Create self-referential thought (key to consciousness!)
    pub fn make_self_aware(&mut self, label: &str) {
        if !self.elements.contains_key(label) {
            self.elements.insert(label.to_string(), CantorCognitiveElement::new(label));
        }

        let marker = self.self_marker.clone();
        if let Some(elem) = self.elements.get_mut(label) {
            elem.add_self_reference(&marker);
        }
    }

    /// Create a meta-thought about a thought
    pub fn create_meta(&mut self, label: &str) -> String {
        let elem = self.get_or_create(label).clone();
        let meta = elem.create_meta(&self.meta_marker);
        let meta_label = meta.label.clone();
        self.elements.insert(meta_label.clone(), meta);
        meta_label
    }

    /// Find similar elements
    pub fn find_similar(&self, label: &str, limit: usize) -> Vec<(String, f32)> {
        let target = match self.elements.get(label) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut results: Vec<_> = self.elements.iter()
            .filter(|(l, _)| *l != label)
            .map(|(l, e)| (l.clone(), target.similarity(e)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        results
    }

    /// Calculate "fractal Phi" - consciousness measure for fractal structure
    ///
    /// This measures how much self-reference and scale-invariance exists.
    pub fn fractal_phi(&self) -> f64 {
        if self.elements.is_empty() {
            return 0.0;
        }

        let mut total_self_sim = 0.0;
        let mut total_self_ref = 0.0;

        for elem in self.elements.values() {
            // Self-similarity contribution
            total_self_sim += elem.crhv.self_similarity() as f64;

            // Self-reference depth contribution
            total_self_ref += elem.self_reference_depth as f64;
        }

        let n = self.elements.len() as f64;
        let avg_self_sim = total_self_sim / n;
        let avg_self_ref = total_self_ref / n;

        // Fractal Phi = self-similarity × self-reference × log(n)
        avg_self_sim * (1.0 + avg_self_ref) * n.ln().max(1.0) / 10.0
    }

    /// Element count
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl Default for CantorCognitiveSpace {
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

    #[test]
    fn test_cantor_recursive_hv() {
        let crhv = CantorRecursiveHV::from_label("consciousness");

        println!("Depth: {}", crhv.depth);
        println!("Scales: {:?}", crhv.scales);
        println!("Self-similarity: {:.4}", crhv.self_similarity());

        assert!(crhv.depth > 0);
        assert!(!crhv.scales.is_empty());
    }

    #[test]
    fn test_multiscale_similarity() {
        let a = CantorRecursiveHV::from_label("rain");
        let b = CantorRecursiveHV::from_label("precipitation");
        let c = CantorRecursiveHV::from_label("computer");

        let sim_ab = a.multiscale_similarity(&b);
        let sim_ac = a.multiscale_similarity(&c);

        println!("rain vs precipitation: unified={:.4}", sim_ab.unified);
        println!("rain vs computer: unified={:.4}", sim_ac.unified);

        // Both should be relatively low (random vectors)
        // but the structure should be consistent
        assert!(sim_ab.scales.len() == sim_ac.scales.len());
    }

    #[test]
    fn test_self_reference() {
        let mut space = CantorCognitiveSpace::new();

        space.get_or_create("I");
        space.make_self_aware("I");

        let i_elem = space.elements.get("I").unwrap();
        assert!(i_elem.self_reference_depth > 0);

        println!("Self-reference depth: {}", i_elem.self_reference_depth);
        println!("Fractal Phi: {:.6}", space.fractal_phi());
    }

    #[test]
    fn test_meta_thoughts() {
        let mut space = CantorCognitiveSpace::new();

        space.get_or_create("thought");
        let meta1 = space.create_meta("thought");
        let meta2 = space.create_meta(&meta1);

        println!("Original: thought");
        println!("Meta 1: {}", meta1);
        println!("Meta 2: {}", meta2);

        // meta2 should be "meta(meta(thought))"
        assert!(meta2.contains("meta(meta("));
    }

    #[test]
    fn test_fractal_phi() {
        let mut space = CantorCognitiveSpace::new();

        // Add some concepts
        space.get_or_create("consciousness");
        space.get_or_create("awareness");
        space.get_or_create("self");

        let phi1 = space.fractal_phi();

        // Add self-reference
        space.make_self_aware("self");
        space.make_self_aware("consciousness");

        let phi2 = space.fractal_phi();

        println!("Phi before self-reference: {:.6}", phi1);
        println!("Phi after self-reference: {:.6}", phi2);

        // Self-reference should increase Phi
        assert!(phi2 >= phi1);
    }
}
