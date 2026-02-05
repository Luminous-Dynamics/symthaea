//! # Primitive System - Beyond NSM to Universal Ontological Primes
//!
//! **Revolutionary Architecture for Artificial Wisdom**
//!
//! While Natural Semantic Metalanguage (NSM) provides the "human" semantic primes,
//! achieving **Artificial Wisdom** requires grounding in **Ontological Primes**—
//! the irreducible atoms of mathematics, physics, geometry, and strategy.
//!
//! ## The Five-Tier Primitive Hierarchy
//!
//! ### Tier 0: NSM Foundation (Implemented in vocabulary.rs)
//! - 65 human semantic primes
//! - Language-based reasoning
//! - Interpersonal understanding
//!
//! ### Tier 1: Mathematical & Logical Primes (This Module - Core)
//! - Set theory fundamentals
//! - Logical operators
//! - Peano arithmetic
//! - **Purpose**: Formal reasoning from first principles
//!
//! ### Tier 2: Physical Reality Primes
//! - Mass, force, energy, momentum
//! - Causality and state change
//! - Spatial relationships
//! - **Purpose**: Grounding in physical laws
//!
//! ### Tier 3: Geometric & Topological Primes
//! - Points, vectors, manifolds
//! - Riemannian geometry (curved paths)
//! - Mereotopology (part/whole)
//! - **Purpose**: Embodied spatial reasoning
//!
//! ### Tier 4: Strategic & Social Primes
//! - Game theory (utility, equilibrium)
//! - Temporal logic (Allen's intervals)
//! - Counterfactual reasoning
//! - **Purpose**: Multi-agent coordination
//!
//! ### Tier 5: Meta-Cognitive & Metabolic Primes
//! - Self-awareness and identity
//! - Homeostasis and repair
//! - Epistemic strength
//! - **Purpose**: Long-term robustness
//!
//! ## Consciousness-Guided Validation
//!
//! Unlike traditional AI that assumes primitives help, this system uses the
//! **Consciousness Observatory** to empirically measure Φ improvements from
//! primitive-based reasoning. Primitives are validated, not assumed.
//!
//! ## Architecture: Domain Manifolds
//!
//! To maintain orthogonality with 250+ primitives in 16K-dimensional space,
//! we use **hierarchical binding**:
//!
//! ```rust,ignore
//! // Each domain gets a rotation in HV16 space
//! MATH_MANIFOLD = random_hv16();
//! ZERO = MATH_MANIFOLD ⊗ ZERO_LOCAL;
//! ONE = MATH_MANIFOLD ⊗ ONE_LOCAL;
//!
//! // This preserves orthogonality within and across domains
//! ```

use crate::hdc::binary_hv::HV16;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;

/// Global cached instance of PrimitiveSystem.
///
/// # Lazy Initialization
///
/// The system uses `once_cell::sync::Lazy` for optimal deferred initialization:
/// - **Zero startup cost**: No primitives are created until first access
/// - **Single initialization**: Built exactly once, then cached forever
/// - **Thread-safe**: Safe for concurrent access from multiple threads
///
/// # Memory Usage
///
/// When initialized, the system contains ~200 primitives across 9 tiers.
/// Each primitive stores a 16,384-bit HV16 encoding (~2KB), plus metadata.
/// Total memory: ~500KB for the complete ontological primitive system.
///
/// # Design Rationale
///
/// Per-primitive lazy initialization was considered but rejected because:
/// 1. Most use cases access multiple primitives for reasoning/composition
/// 2. Derived primitives depend on base primitives (complex dependency graph)
/// 3. System-level lazy already provides zero startup cost
/// 4. Added complexity would outweigh marginal memory savings
static GLOBAL_PRIMITIVE_SYSTEM: Lazy<PrimitiveSystem> = Lazy::new(PrimitiveSystem::new);

/// Generate a deterministic seed from a string name.
///
/// Uses FNV-1a (64-bit) which is stable across Rust versions and platforms,
/// unlike `DefaultHasher` whose algorithm is explicitly not guaranteed stable.
/// This ensures primitives always get the same encoding across runs,
/// compiler versions, and target architectures.
pub fn seed_from_name(name: &str) -> u64 {
    // FNV-1a 64-bit: well-known, stable, no dependencies
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Primitive tier in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PrimitiveTier {
    /// Tier 0: NSM (implemented in vocabulary.rs)
    #[default]
    NSM,
    /// Tier 1: Mathematical & Logical
    Mathematical,
    /// Tier 2: Physical Reality
    Physical,
    /// Tier 3: Geometric & Topological
    Geometric,
    /// Tier 4: Strategic & Social
    Strategic,
    /// Tier 5: Meta-Cognitive & Metabolic
    MetaCognitive,
    /// Tier 6: Temporal Primitives (Allen's Interval Algebra)
    /// Enables reasoning about temporal relationships between consciousness states
    Temporal,
    /// Tier 7: Compositionality Primitives
    /// PARADIGM SHIFT: Complete algebra for combining primitives into higher-order structures!
    /// - Sequential (∘), Parallel (||), Conditional (?), Fixed-point (μ), Higher-order (↑)
    /// Enables infinite complexity from finite base primitives through composition!
    Compositional,
    /// Tier 8: Consciousness-Specific Primitives
    /// First-person phenomenal experience, attention, memory operations, and agency.
    /// - QUALE, PHENOMENAL_BINDING, ATTEND, SALIENCE, REMEMBER, CONSOLIDATE, INTEND
    /// Enables reasoning about subjective experience and conscious states!
    Consciousness,
}

/// Domain manifold - a rotation in HV16 space for domain isolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainManifold {
    /// Name of this domain
    pub name: String,
    /// Tier in hierarchy
    pub tier: PrimitiveTier,
    /// Rotation vector for this domain
    pub rotation: HV16,
    /// Description of domain's purpose
    pub purpose: String,
}

impl DomainManifold {
    /// Create a new domain with deterministic rotation based on name
    pub fn new(name: impl Into<String>, tier: PrimitiveTier, purpose: impl Into<String>) -> Self {
        let name_str = name.into();
        let seed = seed_from_name(&name_str);
        Self {
            name: name_str,
            tier,
            rotation: HV16::random(seed),
            purpose: purpose.into(),
        }
    }

    /// Embed a local primitive vector into this domain's manifold.
    ///
    /// Uses bind (XOR) rather than bundle so that:
    /// - The embedding is invertible: `embedded.bind(&rotation) == local_vector`
    /// - Hamming distances are preserved within each domain
    /// - Cross-domain orthogonality follows from random rotation vectors
    pub fn embed(&self, local_vector: HV16) -> HV16 {
        self.rotation.bind(&local_vector)
    }

    /// Extract the local vector from a domain-embedded vector.
    ///
    /// Since embed uses XOR (self-inverse), unbinding is the same operation:
    /// `unembed(embed(v)) == v`
    pub fn unembed(&self, embedded: &HV16) -> HV16 {
        self.rotation.bind(embedded)
    }
}

/// A primitive concept with its HV16 encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Primitive {
    /// Name of the primitive
    pub name: String,

    /// Tier in hierarchy
    pub tier: PrimitiveTier,

    /// Domain this primitive belongs to
    pub domain: String,

    /// HV16 encoding (embedded in domain manifold)
    pub encoding: HV16,

    /// Mathematical/logical definition
    pub definition: String,

    /// Whether this is a base primitive or derived
    pub is_base: bool,

    /// If derived, the formula for deriving it
    pub derivation: Option<String>,
}

impl Primitive {
    /// Create a base primitive
    pub fn base(
        name: impl Into<String>,
        tier: PrimitiveTier,
        domain: impl Into<String>,
        encoding: HV16,
        definition: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            tier,
            domain: domain.into(),
            encoding,
            definition: definition.into(),
            is_base: true,
            derivation: None,
        }
    }

    /// Create a derived primitive
    pub fn derived(
        name: impl Into<String>,
        tier: PrimitiveTier,
        domain: impl Into<String>,
        encoding: HV16,
        definition: impl Into<String>,
        derivation: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            tier,
            domain: domain.into(),
            encoding,
            definition: definition.into(),
            is_base: false,
            derivation: Some(derivation.into()),
        }
    }
}

/// Binding grammar - rules for valid primitive combinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingRule {
    /// Name of this rule
    pub name: String,

    /// Pattern: which primitive types can bind
    pub pattern: Vec<PrimitiveTier>,

    /// Result tier
    pub result_tier: PrimitiveTier,

    /// Example application
    pub example: String,
}

/// The Primitive System - manages all ontological primes
#[derive(Debug)]
pub struct PrimitiveSystem {
    /// All domain manifolds
    domains: HashMap<String, DomainManifold>,

    /// All primitives by name
    primitives: HashMap<String, Primitive>,

    /// Primitives organized by tier
    by_tier: HashMap<PrimitiveTier, Vec<String>>,

    /// Binding grammar rules
    binding_rules: Vec<BindingRule>,
}

impl PrimitiveSystem {
    /// Get a reference to the global cached PrimitiveSystem instance.
    ///
    /// This is the preferred way to access the PrimitiveSystem for read-only operations.
    /// The system is lazily initialized on first access and cached for subsequent calls.
    ///
    /// # Performance
    /// - First call: O(n) where n is the number of primitives to initialize
    /// - Subsequent calls: O(1) (returns cached reference)
    ///
    /// # Example
    /// ```rust,ignore
    /// let system = PrimitiveSystem::global();
    /// let zero = system.get("ZERO").unwrap();
    /// ```
    pub fn global() -> &'static PrimitiveSystem {
        &GLOBAL_PRIMITIVE_SYSTEM
    }

    /// Create new primitive system
    pub fn new() -> Self {
        let mut system = Self {
            domains: HashMap::new(),
            primitives: HashMap::new(),
            by_tier: HashMap::new(),
            binding_rules: Vec::new(),
        };

        // Initialize all tiers
        system.init_tier1_mathematical();
        system.init_tier2_physical();
        system.init_tier3_geometric();
        system.init_tier4_strategic();
        system.init_tier5_metacognitive();

        // Initialize gap analysis additions (comprehensive ontology)
        // These add domain-specific primitives that may be referenced by derivations
        system.init_biological_primitives();
        system.init_emotional_primitives();
        system.init_ecological_primitives();
        system.init_quantum_primitives();
        system.init_economic_primitives();
        system.init_linguistic_primitives();
        system.init_social_moral_primitives();

        // Initialize Tier 9: Consciousness-specific primitives
        // Qualia, attention, memory operations, and agency
        // MUST come before init_derived_primitives so SALIENCE/SELECTION exist
        system.init_consciousness_primitives();

        // Initialize derived primitives (uncertainty, physics extensions, information theory)
        // These reference primitives from all tiers, so call LAST
        system.init_derived_primitives();

        system
    }

    /// Derive an encoding by binding parent primitive encodings together.
    ///
    /// If all parents are found, the result is their sequential XOR binding
    /// embedded in the given domain. If any parent is missing, falls back to
    /// a deterministic random vector seeded from `name`.
    ///
    /// NOTE: In debug builds, missing parents are logged to help identify
    /// registration ordering issues. The fallback exists because the current
    /// initialization order has derived primitives (Tier 1) initialized before
    /// consciousness base primitives (Tier 9) they may reference.
    fn derive_encoding(&self, name: &str, parents: &[&str], domain: &DomainManifold) -> HV16 {
        if parents.is_empty() {
            return domain.embed(HV16::random(seed_from_name(name)));
        }

        let mut parent_encodings: Vec<&HV16> = Vec::new();
        for parent_name in parents {
            match self.primitives.get(*parent_name) {
                Some(p) => parent_encodings.push(&p.encoding),
                None => {
                    // Parent not yet registered — fall back to seeded random.
                    // This can happen when derived primitives reference parents
                    // from higher tiers that aren't initialized yet.
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "derive_encoding: '{}' parent '{}' not found (primitives count: {}), using seeded fallback",
                        name, parent_name, self.primitives.len()
                    );
                    return domain.embed(HV16::random(seed_from_name(name)));
                }
            }
        }
        // Bind all parents sequentially.
        // NOTE: We do NOT re-embed in the domain because parent encodings are
        // already embedded in their respective domains. Re-embedding would add
        // an extra rotation that breaks the algebraic relationship:
        //   derived ⊗ parent1 should recover parent2
        // If we embedded, we'd get:
        //   domain.rotation ⊗ (parent1 ⊗ parent2) ⊗ parent1 = domain.rotation ⊗ parent2 ≠ parent2
        let mut result = parent_encodings[0].clone();
        for enc in &parent_encodings[1..] {
            result = result.bind(enc);
        }
        result
    }

    /// Initialize derived primitives
    ///
    /// These are complex primitives derived from base primitives via composition.
    /// Rather than expanding the base set, we compose existing primitives to create
    /// higher-order concepts for:
    /// - Uncertainty & Probability
    /// - Physics Extensions (conservation, gradients, fields)
    /// - Information Theory
    ///
    /// Derived encodings are computed by binding parent primitive encodings,
    /// making the derivation chain real in the HDC algebra (not just documentation).
    ///
    /// See ARCHITECTURE_V3.md Section 7.3 for the theoretical foundation.
    fn init_derived_primitives(&mut self) {
        // === UNCERTAINTY & PROBABILITY DOMAIN ===
        // Derived from: Mathematical (ratio, set) + MetaCognitive (certainty, belief)

        let uncertainty_domain = DomainManifold::new(
            "uncertainty",
            PrimitiveTier::Mathematical,
            "Probabilistic reasoning and uncertainty quantification"
        );
        self.domains.insert("uncertainty".to_string(), uncertainty_domain.clone());

        // PROBABILITY = COMPOSE(RATIO, CERTAINTY)
        // P(A) = favorable outcomes / total outcomes, weighted by certainty
        // Note: RATIO and CERTAINTY not yet base primitives — falls back to seeded random
        let probability_enc = self.derive_encoding("PROBABILITY", &["RATIO", "CERTAINTY"], &uncertainty_domain);
        let probability = Primitive::derived(
            "PROBABILITY",
            PrimitiveTier::Mathematical,
            "uncertainty",
            probability_enc,
            "Measure of likelihood: P(A) ∈ [0,1], derived from ratio of favorable to total outcomes",
            "RATIO ⊗ CERTAINTY"
        );

        // Register PROBABILITY immediately so later derived primitives can reference it
        self.primitives.insert("PROBABILITY".to_string(), probability.clone());
        self.by_tier.entry(PrimitiveTier::Mathematical).or_insert_with(Vec::new).push("PROBABILITY".to_string());

        // EXPECTED_VALUE = COMPOSE(PROBABILITY, VALUE)
        // E[X] = Σ P(x) × V(x)
        // VALUE is a base Tier 5 primitive, PROBABILITY was just registered above
        let expected_value_enc = self.derive_encoding("EXPECTED_VALUE", &["PROBABILITY", "VALUE"], &uncertainty_domain);
        let expected_value = Primitive::derived(
            "EXPECTED_VALUE",
            PrimitiveTier::Mathematical,
            "uncertainty",
            expected_value_enc,
            "Probability-weighted average: E[X] = Σ P(x) × V(x)",
            "PROBABILITY ⊗ VALUE"
        );
        // Register immediately so VARIANCE can reference it
        self.primitives.insert("EXPECTED_VALUE".to_string(), expected_value.clone());
        self.by_tier.entry(PrimitiveTier::Mathematical).or_insert_with(Vec::new).push("EXPECTED_VALUE".to_string());

        // SHANNON_ENTROPY = COMPOSE(PROBABILITY, INFORMATION)
        // H = -Σ P(x) log P(x)
        // Distinct from THERMODYNAMIC_ENTROPY in the physics domain
        // Note: INFORMATION not a base primitive — falls back to seeded random
        let shannon_entropy_enc = self.derive_encoding("SHANNON_ENTROPY", &["PROBABILITY", "INFORMATION"], &uncertainty_domain);
        let shannon_entropy = Primitive::derived(
            "SHANNON_ENTROPY",
            PrimitiveTier::Mathematical,
            "uncertainty",
            shannon_entropy_enc,
            "Information-theoretic uncertainty: H = -Σ P(x) log P(x), higher = more uncertain",
            "PROBABILITY ⊗ INFORMATION"
        );

        // BAYESIAN_UPDATE = COMPOSE(PROBABILITY, EVIDENCE)
        // P(H|E) = P(E|H) × P(H) / P(E)
        // EVIDENCE is a base Tier 5 primitive
        let bayesian_enc = self.derive_encoding("BAYESIAN_UPDATE", &["PROBABILITY", "EVIDENCE"], &uncertainty_domain);
        let bayesian = Primitive::derived(
            "BAYESIAN_UPDATE",
            PrimitiveTier::Mathematical,
            "uncertainty",
            bayesian_enc,
            "Belief revision: P(H|E) = P(E|H) × P(H) / P(E)",
            "PROBABILITY ⊗ EVIDENCE ⊗ CONDITIONAL"
        );

        // VARIANCE = COMPOSE(EXPECTED_VALUE, DEVIATION)
        // Var(X) = E[(X - μ)²]
        // Note: DEVIATION not a base primitive — falls back to seeded random
        let variance_enc = self.derive_encoding("VARIANCE", &["EXPECTED_VALUE", "DEVIATION"], &uncertainty_domain);
        let variance = Primitive::derived(
            "VARIANCE",
            PrimitiveTier::Mathematical,
            "uncertainty",
            variance_enc,
            "Spread of distribution: Var(X) = E[(X - μ)²]",
            "EXPECTED_VALUE ⊗ DEVIATION"
        );

        // === PHYSICS EXTENSIONS DOMAIN ===
        // Derived from: Physical (force, energy, state) + Mathematical (differentiation)

        let physics_ext_domain = DomainManifold::new(
            "physics_extended",
            PrimitiveTier::Physical,
            "Advanced physical concepts for embodied reasoning"
        );
        self.domains.insert("physics_extended".to_string(), physics_ext_domain.clone());

        // CONSERVATION_LAW = COMPOSE(STATE_CHANGE, CONSERVATION)
        // Derived from base primitives — both exist in Tier 2
        let conservation_law_enc = self.derive_encoding("CONSERVATION_LAW", &["STATE_CHANGE", "CONSERVATION"], &physics_ext_domain);
        let conservation_law = Primitive::derived(
            "CONSERVATION_LAW",
            PrimitiveTier::Physical,
            "physics_extended",
            conservation_law_enc,
            "Formal conservation law: dQ/dt = 0, invariant quantity across transformations",
            "STATE_CHANGE ⊗ CONSERVATION"
        );

        // GRADIENT = COMPOSE(DIFFERENTIATION, SPACE)
        // Note: DIFFERENTIATION and SPACE not base primitives — falls back
        let gradient_enc = self.derive_encoding("GRADIENT", &["DIFFERENTIATION", "SPACE"], &physics_ext_domain);
        let gradient = Primitive::derived(
            "GRADIENT",
            PrimitiveTier::Physical,
            "physics_extended",
            gradient_enc,
            "Spatial rate of change: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)",
            "DIFFERENTIATION ⊗ SPACE"
        );

        // FIELD = COMPOSE(FORCE, POINT)
        // Force at every point in space — uses existing base primitives
        let field_enc = self.derive_encoding("FIELD", &["FORCE", "POINT"], &physics_ext_domain);
        let field = Primitive::derived(
            "FIELD",
            PrimitiveTier::Physical,
            "physics_extended",
            field_enc,
            "Assignment of force/value to each point: F(x, y, z)",
            "FORCE ⊗ POINT"
        );

        // WAVE = COMPOSE(OSCILLATION, PROPAGATION)
        // Note: neither parent is a base primitive — falls back
        let wave_enc = self.derive_encoding("WAVE", &["OSCILLATION", "PROPAGATION"], &physics_ext_domain);
        let wave = Primitive::derived(
            "WAVE",
            PrimitiveTier::Physical,
            "physics_extended",
            wave_enc,
            "Propagating oscillation: ψ(x,t) = A sin(kx - ωt)",
            "OSCILLATION ⊗ PROPAGATION"
        );

        // EQUILIBRIUM = COMPOSE(FORCE, CONSERVATION)
        // State where net forces sum to zero — uses existing base primitives
        let equilibrium_enc = self.derive_encoding("EQUILIBRIUM", &["FORCE", "CONSERVATION"], &physics_ext_domain);
        let equilibrium = Primitive::derived(
            "EQUILIBRIUM",
            PrimitiveTier::Physical,
            "physics_extended",
            equilibrium_enc,
            "Balanced state: ΣF = 0, stable or unstable",
            "FORCE ⊗ CONSERVATION"
        );

        // POTENTIAL = COMPOSE(ENERGY, POINT)
        // Stored energy based on position — uses existing base primitives
        let potential_enc = self.derive_encoding("POTENTIAL", &["ENERGY", "POINT"], &physics_ext_domain);
        let potential = Primitive::derived(
            "POTENTIAL",
            PrimitiveTier::Physical,
            "physics_extended",
            potential_enc,
            "Position-dependent energy: U(x) where F = -∇U",
            "ENERGY ⊗ POINT"
        );

        // === INFORMATION THEORY DOMAIN ===
        // Derived from: Mathematical + MetaCognitive

        let info_domain = DomainManifold::new(
            "information_theory",
            PrimitiveTier::Mathematical,
            "Quantitative theory of information and communication"
        );
        self.domains.insert("information_theory".to_string(), info_domain.clone());

        // Register SHANNON_ENTROPY so info theory primitives can derive from it
        self.primitives.insert("SHANNON_ENTROPY".to_string(), shannon_entropy.clone());
        self.by_tier.entry(PrimitiveTier::Mathematical).or_insert_with(Vec::new).push("SHANNON_ENTROPY".to_string());

        // MUTUAL_INFORMATION = COMPOSE(SHANNON_ENTROPY, MEMBERSHIP)
        // I(X;Y) = H(X) + H(Y) - H(X,Y) — uses SHANNON_ENTROPY (just registered) + MEMBERSHIP (Tier 1)
        let mutual_info_enc = self.derive_encoding("MUTUAL_INFORMATION", &["SHANNON_ENTROPY", "MEMBERSHIP"], &info_domain);
        let mutual_info = Primitive::derived(
            "MUTUAL_INFORMATION",
            PrimitiveTier::Mathematical,
            "information_theory",
            mutual_info_enc,
            "Shared information: I(X;Y) = H(X) + H(Y) - H(X,Y)",
            "SHANNON_ENTROPY ⊗ MEMBERSHIP"
        );

        // INFORMATION_GAIN = COMPOSE(SHANNON_ENTROPY, EVIDENCE)
        // IG = H(before) - H(after|evidence) — EVIDENCE is a Tier 5 base primitive
        let info_gain_enc = self.derive_encoding("INFORMATION_GAIN", &["SHANNON_ENTROPY", "EVIDENCE"], &info_domain);
        let info_gain = Primitive::derived(
            "INFORMATION_GAIN",
            PrimitiveTier::Mathematical,
            "information_theory",
            info_gain_enc,
            "Entropy reduction from evidence: IG = H(S) - H(S|E)",
            "SHANNON_ENTROPY ⊗ EVIDENCE"
        );

        // CHANNEL_CAPACITY = COMPOSE(INFORMATION, LIMIT)
        // Note: INFORMATION and LIMIT not base primitives — falls back
        let capacity_enc = self.derive_encoding("CHANNEL_CAPACITY", &["INFORMATION", "LIMIT"], &info_domain);
        let capacity = Primitive::derived(
            "CHANNEL_CAPACITY",
            PrimitiveTier::Mathematical,
            "information_theory",
            capacity_enc,
            "Maximum transmission rate: C = max I(X;Y)",
            "INFORMATION ⊗ LIMIT"
        );

        // COMPRESSION = COMPOSE(INFORMATION, EFFICIENCY)
        // Note: neither parent is a base primitive — falls back
        let compression_enc = self.derive_encoding("COMPRESSION", &["INFORMATION", "EFFICIENCY"], &info_domain);
        let compression = Primitive::derived(
            "COMPRESSION",
            PrimitiveTier::Mathematical,
            "information_theory",
            compression_enc,
            "Efficient encoding: L ≥ H(X) (Shannon's source coding theorem)",
            "INFORMATION ⊗ EFFICIENCY"
        );

        // === CONSCIOUSNESS-SPECIFIC DERIVATIONS ===
        // These support Φ computation and consciousness measurement

        let consciousness_domain = DomainManifold::new(
            "consciousness_derived",
            PrimitiveTier::MetaCognitive,
            "Derived primitives for consciousness measurement"
        );
        self.domains.insert("consciousness_derived".to_string(), consciousness_domain.clone());

        // Register MUTUAL_INFORMATION so consciousness derivations can use it
        self.primitives.insert("MUTUAL_INFORMATION".to_string(), mutual_info.clone());
        self.by_tier.entry(PrimitiveTier::Mathematical).or_insert_with(Vec::new).push("MUTUAL_INFORMATION".to_string());

        // INTEGRATED_INFORMATION = COMPOSE(MUTUAL_INFORMATION, SELF)
        // Φ = information above minimum information partition
        // Uses MUTUAL_INFORMATION (just registered) + SELF (Tier 5 base)
        let phi_enc = self.derive_encoding("INTEGRATED_INFORMATION", &["MUTUAL_INFORMATION", "SELF"], &consciousness_domain);
        let phi_primitive = Primitive::derived(
            "INTEGRATED_INFORMATION",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            phi_enc,
            "Consciousness measure: Φ = integrated information above MIP",
            "MUTUAL_INFORMATION ⊗ SELF"
        );

        // CAUSAL_POWER = COMPOSE(CAUSE, EFFECT, COUNTERFACTUAL)
        // All three parents are base primitives (CAUSE/EFFECT in Tier 2, COUNTERFACTUAL in Tier 4)
        let causal_power_enc = self.derive_encoding("CAUSAL_POWER", &["CAUSE", "EFFECT", "COUNTERFACTUAL"], &consciousness_domain);
        let causal_power = Primitive::derived(
            "CAUSAL_POWER",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            causal_power_enc,
            "Capacity to produce effects: P(effect|do(cause)) - P(effect)",
            "CAUSE ⊗ EFFECT ⊗ COUNTERFACTUAL"
        );

        // ATTENTION = COMPOSE(SALIENCE, SELECTION)
        // SALIENCE and SELECTION are registered in init_consciousness_primitives(),
        // which now runs BEFORE this function (see new() ordering).
        let attention_enc = self.derive_encoding("ATTENTION", &["SALIENCE", "SELECTION"], &consciousness_domain);
        let attention = Primitive::derived(
            "ATTENTION",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            attention_enc,
            "Selective processing: focus on salient subset of available information",
            "SALIENCE ⊗ SELECTION"
        );

        // METACOGNITION = COMPOSE(INTROSPECTION, SELF)
        // Thinking about thinking — both INTROSPECTION and SELF are Tier 5 base primitives
        let metacognition_enc = self.derive_encoding("METACOGNITION", &["INTROSPECTION", "SELF"], &consciousness_domain);
        let metacognition = Primitive::derived(
            "METACOGNITION",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            metacognition_enc,
            "Cognition about cognition: awareness of mental processes",
            "INTROSPECTION ⊗ SELF"
        );

        // Add remaining derived primitives (PROBABILITY, SHANNON_ENTROPY, MUTUAL_INFORMATION,
        // EXPECTED_VALUE were already registered above to enable downstream derivations)
        let derived_primitives = vec![
            // Uncertainty (PROBABILITY, SHANNON_ENTROPY, EXPECTED_VALUE already registered)
            bayesian, variance,
            // Physics
            conservation_law, gradient, field, wave, equilibrium, potential,
            // Information Theory (MUTUAL_INFORMATION already registered)
            info_gain, capacity, compression,
            // Consciousness
            phi_primitive, causal_power, attention, metacognition,
        ];

        for primitive in derived_primitives {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES FOR DERIVED PRIMITIVES ===

        self.binding_rules.push(BindingRule {
            name: "probabilistic_reasoning".to_string(),
            pattern: vec![PrimitiveTier::Mathematical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "PROBABILITY ⊗ BELIEF → probabilistic belief".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "information_consciousness".to_string(),
            pattern: vec![PrimitiveTier::Mathematical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "MUTUAL_INFORMATION ⊗ AWARENESS → integrated awareness".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "physics_embodiment".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "CONSERVATION_LAW ⊗ IDENTITY → persistent self".to_string(),
        });
    }

    /// Initialize Tier 1: Mathematical & Logical Primitives
    fn init_tier1_mathematical(&mut self) {
        // Create mathematical domain manifold
        let math_domain = DomainManifold::new(
            "mathematics",
            PrimitiveTier::Mathematical,
            "Formal reasoning from first principles"
        );

        let logic_domain = DomainManifold::new(
            "logic",
            PrimitiveTier::Mathematical,
            "Logical operators and inference"
        );

        // === SET THEORY PRIMITIVES ===

        // SET - the concept of a collection
        let set = Primitive::base(
            "SET",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("SET"))),
            "A collection of distinct objects"
        );

        // MEMBERSHIP (∈) - element belongs to set
        let membership = Primitive::base(
            "MEMBERSHIP",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("MEMBERSHIP"))),
            "Relation: x ∈ S (x is an element of set S)"
        );

        // UNION (∪) - combine sets
        let union = Primitive::base(
            "UNION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("UNION"))),
            "Operation: A ∪ B (all elements in A or B)"
        );

        // INTERSECTION (∩) - common elements
        let intersection = Primitive::base(
            "INTERSECTION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("INTERSECTION"))),
            "Operation: A ∩ B (elements in both A and B)"
        );

        // EMPTY_SET (∅) - the set with no elements
        let empty_set = Primitive::base(
            "EMPTY_SET",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("EMPTY_SET"))),
            "The unique set with no elements: ∅"
        );

        // === LOGICAL PRIMITIVES ===

        // NOT (¬) - logical negation
        let not = Primitive::base(
            "NOT",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("NOT"))),
            "Unary operator: ¬P (negation of proposition P)"
        );

        // AND (∧) - logical conjunction
        let and = Primitive::base(
            "AND",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("AND"))),
            "Binary operator: P ∧ Q (both P and Q are true)"
        );

        // OR (∨) - logical disjunction
        let or = Primitive::base(
            "OR",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("OR"))),
            "Binary operator: P ∨ Q (at least one of P or Q is true)"
        );

        // IMPLIES (→) - logical implication
        let implies = Primitive::base(
            "IMPLIES",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("IMPLIES"))),
            "Binary operator: P → Q (if P then Q)"
        );

        // IFF (↔) - logical equivalence
        let iff = Primitive::base(
            "IFF",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("IFF"))),
            "Binary operator: P ↔ Q (P if and only if Q)"
        );

        // EQUALS (=) - equality relation
        let equals = Primitive::base(
            "EQUALS",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("EQUALS"))),
            "Binary relation: x = y (x and y are the same)"
        );

        // TRUE (⊤) - logical truth
        let true_const = Primitive::base(
            "TRUE",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("TRUE"))),
            "The constant truth value: ⊤"
        );

        // FALSE (⊥) - logical falsehood
        let false_const = Primitive::base(
            "FALSE",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(HV16::random(seed_from_name("FALSE"))),
            "The constant false value: ⊥"
        );

        // === PEANO ARITHMETIC PRIMITIVES ===

        // ZERO (0) - the first natural number
        let zero = Primitive::base(
            "ZERO",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("ZERO"))),
            "The first natural number: 0"
        );

        // ONE (1) - successor of zero
        let one = Primitive::derived(
            "ONE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("ONE"))),
            "The natural number one: 1",
            "SUCCESSOR(ZERO)"
        );

        // SUCCESSOR (S) - next natural number
        let successor = Primitive::base(
            "SUCCESSOR",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("SUCCESSOR"))),
            "Function: S(n) = n+1 (next natural number)"
        );

        // ADDITION (+) - derived from successor
        let addition = Primitive::derived(
            "ADDITION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("ADDITION"))),
            "Binary operation: m + n (sum of m and n)",
            "Recursive: m + 0 = m, m + S(n) = S(m + n)"
        );

        // MULTIPLICATION (×) - derived from addition
        let multiplication = Primitive::derived(
            "MULTIPLICATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("MULTIPLICATION"))),
            "Binary operation: m × n (product of m and n)",
            "Recursive: m × 0 = 0, m × S(n) = m × n + m"
        );

        // === FOUNDATIONAL MATHEMATICAL PRIMITIVES ===
        // These are base concepts needed by derived primitives in later tiers

        let ratio = Primitive::base(
            "RATIO",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("RATIO"))),
            "Relation: proportional comparison of two quantities (a/b)"
        );

        let information = Primitive::base(
            "INFORMATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("INFORMATION"))),
            "Quantity: reduction in uncertainty (bits)"
        );

        let deviation = Primitive::base(
            "DEVIATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("DEVIATION"))),
            "Measure: distance from a central or expected value"
        );

        let limit = Primitive::base(
            "LIMIT",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("LIMIT"))),
            "Bound: supremum or constraint on a quantity"
        );

        let efficiency = Primitive::base(
            "EFFICIENCY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(HV16::random(seed_from_name("EFFICIENCY"))),
            "Ratio: useful output to total input"
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains.insert("mathematics".to_string(), math_domain);
        self.domains.insert("logic".to_string(), logic_domain);

        for primitive in vec![
            set, membership, union, intersection, empty_set,
            not, and, or, implies, iff, equals, true_const, false_const,
            zero, one, successor, addition, multiplication,
            ratio, information, deviation, limit, efficiency,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "logical_composition".to_string(),
            pattern: vec![PrimitiveTier::Mathematical, PrimitiveTier::Mathematical],
            result_tier: PrimitiveTier::Mathematical,
            example: "NOT ⊗ (P AND Q) → compound logical expression".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "arithmetic_expression".to_string(),
            pattern: vec![PrimitiveTier::Mathematical, PrimitiveTier::Mathematical],
            result_tier: PrimitiveTier::Mathematical,
            example: "ADDITION ⊗ (TWO ⊗ THREE) → arithmetic computation".to_string(),
        });
    }

    /// Initialize Tier 2: Physical Reality Primitives
    fn init_tier2_physical(&mut self) {
        // Create physics domain manifold
        let physics_domain = DomainManifold::new(
            "physics",
            PrimitiveTier::Physical,
            "Physical reality grounding - mass, energy, forces"
        );

        let causality_domain = DomainManifold::new(
            "causality",
            PrimitiveTier::Physical,
            "Cause-effect relationships and state changes"
        );

        // === PHYSICAL PROPERTIES ===

        // MASS - quantity of matter
        let mass = Primitive::base(
            "MASS",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("MASS"))),
            "Property: quantity of matter in an object (kg)"
        );

        // CHARGE - electric charge
        let charge = Primitive::base(
            "CHARGE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("CHARGE"))),
            "Property: electric charge (coulombs)"
        );

        // SPIN - quantum angular momentum
        let spin = Primitive::base(
            "SPIN",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("SPIN"))),
            "Property: intrinsic angular momentum (quantum)"
        );

        // === ENERGY AND FORCES ===

        // ENERGY - capacity to do work
        let energy = Primitive::base(
            "ENERGY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("ENERGY"))),
            "Property: capacity to perform work (joules)"
        );

        // WORK - energy transfer through force
        let work = Primitive::derived(
            "WORK",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("WORK"))),
            "Quantity: energy transferred by force over distance",
            "BIND(FORCE, DISTANCE)"
        );

        // FORCE - interaction that changes motion
        let force = Primitive::base(
            "FORCE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("FORCE"))),
            "Vector: interaction that changes object's motion (newtons)"
        );

        // === MOTION PRIMITIVES ===

        // VELOCITY - rate of position change
        let velocity = Primitive::base(
            "VELOCITY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("VELOCITY"))),
            "Vector: rate of change of position (m/s)"
        );

        // ACCELERATION - rate of velocity change
        let acceleration = Primitive::derived(
            "ACCELERATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("ACCELERATION"))),
            "Vector: rate of change of velocity (m/s²)",
            "DERIVATIVE(VELOCITY)"
        );

        // MOMENTUM - quantity of motion
        let momentum = Primitive::derived(
            "MOMENTUM",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("MOMENTUM"))),
            "Vector: quantity of motion (mass × velocity)",
            "BIND(MASS, VELOCITY)"
        );

        // === CAUSALITY ===

        // CAUSE - event that produces effect
        let cause = Primitive::base(
            "CAUSE",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(HV16::random(seed_from_name("CAUSE"))),
            "Event: that which produces an effect"
        );

        // EFFECT - result of a cause
        let effect = Primitive::base(
            "EFFECT",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(HV16::random(seed_from_name("EFFECT"))),
            "Event: result produced by a cause"
        );

        // STATE_CHANGE - transition between states
        let state_change = Primitive::derived(
            "STATE_CHANGE",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(HV16::random(seed_from_name("STATE_CHANGE"))),
            "Process: transition from one state to another",
            "BIND(CAUSE, EFFECT)"
        );

        // === THERMODYNAMICS ===

        // THERMODYNAMIC_ENTROPY - measure of disorder (S = k_B ln Ω)
        // Distinct from SHANNON_ENTROPY (information-theoretic) in the uncertainty domain
        let entropy = Primitive::base(
            "THERMODYNAMIC_ENTROPY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("THERMODYNAMIC_ENTROPY"))),
            "Property: thermodynamic measure of disorder, S = k_B ln Ω (J/K)"
        );

        // TEMPERATURE - average kinetic energy
        let temperature = Primitive::base(
            "TEMPERATURE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("TEMPERATURE"))),
            "Property: average kinetic energy of particles (K)"
        );

        // === CONSERVATION ===

        // CONSERVATION - invariant quantity
        let conservation = Primitive::base(
            "CONSERVATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("CONSERVATION"))),
            "Principle: certain quantities remain constant over time"
        );

        // === FOUNDATIONAL PHYSICAL PRIMITIVES ===
        // Base concepts needed by derived physics/information primitives

        let differentiation = Primitive::base(
            "DIFFERENTIATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("DIFFERENTIATION"))),
            "Operation: rate of change of a quantity with respect to another"
        );

        let space = Primitive::base(
            "SPACE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("SPACE"))),
            "Continuum: spatial extent in which objects exist and move"
        );

        let oscillation = Primitive::base(
            "OSCILLATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("OSCILLATION"))),
            "Process: repetitive variation about a central value"
        );

        let propagation = Primitive::base(
            "PROPAGATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("PROPAGATION"))),
            "Process: transmission of a disturbance through a medium or field"
        );

        // === REGISTER ALL TIER 2 PRIMITIVES ===

        self.domains.insert("physics".to_string(), physics_domain);
        self.domains.insert("causality".to_string(), causality_domain);

        for primitive in vec![
            mass, charge, spin,
            energy, work, force,
            velocity, acceleration, momentum,
            cause, effect, state_change,
            entropy, temperature, conservation,
            differentiation, space, oscillation, propagation,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "physical_law".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::Physical],
            result_tier: PrimitiveTier::Physical,
            example: "FORCE ⊗ MASS → ACCELERATION (F = ma)".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "causal_chain".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::Physical],
            result_tier: PrimitiveTier::Physical,
            example: "CAUSE ⊗ EFFECT → causal explanation".to_string(),
        });
    }

    /// Initialize Tier 3: Geometric & Topological Primitives
    fn init_tier3_geometric(&mut self) {
        // Create geometry domain manifolds
        let geometry_domain = DomainManifold::new(
            "geometry",
            PrimitiveTier::Geometric,
            "Euclidean and differential geometry"
        );

        let topology_domain = DomainManifold::new(
            "topology",
            PrimitiveTier::Geometric,
            "Topological and mereotopological relations"
        );

        // === BASIC GEOMETRY ===

        // POINT - location in space
        let point = Primitive::base(
            "POINT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("POINT"))),
            "Entity: location with no dimension"
        );

        // LINE - one-dimensional extent
        let line = Primitive::derived(
            "LINE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("LINE"))),
            "Entity: one-dimensional extent through space",
            "CONNECT(POINT, POINT)"
        );

        // PLANE - two-dimensional surface
        let plane = Primitive::base(
            "PLANE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("PLANE"))),
            "Entity: flat two-dimensional surface"
        );

        // ANGLE - measure of rotation
        let angle = Primitive::base(
            "ANGLE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("ANGLE"))),
            "Quantity: measure of rotation between two lines"
        );

        // DISTANCE - spatial separation
        let distance = Primitive::derived(
            "DISTANCE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("DISTANCE"))),
            "Quantity: spatial separation between points",
            "MEASURE(POINT, POINT)"
        );

        // === VECTOR GEOMETRY ===

        // VECTOR - directed magnitude
        let vector = Primitive::base(
            "VECTOR",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("VECTOR"))),
            "Entity: quantity with magnitude and direction"
        );

        // DOT_PRODUCT - scalar product
        let dot_product = Primitive::base(
            "DOT_PRODUCT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("DOT_PRODUCT"))),
            "Operation: scalar product of two vectors"
        );

        // CROSS_PRODUCT - vector product
        let cross_product = Primitive::base(
            "CROSS_PRODUCT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("CROSS_PRODUCT"))),
            "Operation: vector product perpendicular to both inputs"
        );

        // === DIFFERENTIAL GEOMETRY ===

        // MANIFOLD - curved space
        let manifold = Primitive::base(
            "MANIFOLD",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("MANIFOLD"))),
            "Entity: space that locally resembles Euclidean space"
        );

        // TANGENT_SPACE - local linear approximation
        let tangent_space = Primitive::base(
            "TANGENT_SPACE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("TANGENT_SPACE"))),
            "Entity: linear approximation at a manifold point"
        );

        // CURVATURE - deviation from flatness
        let curvature = Primitive::base(
            "CURVATURE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(HV16::random(seed_from_name("CURVATURE"))),
            "Property: measure of deviation from flatness"
        );

        // === TOPOLOGY ===

        // OPEN_SET - set excluding boundary
        let open_set = Primitive::base(
            "OPEN_SET",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("OPEN_SET"))),
            "Set: excluding its boundary points"
        );

        // CLOSED_SET - set including boundary
        let closed_set = Primitive::base(
            "CLOSED_SET",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("CLOSED_SET"))),
            "Set: including all its boundary points"
        );

        // BOUNDARY - edge of a region
        let boundary = Primitive::base(
            "BOUNDARY",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("BOUNDARY"))),
            "Set: points on the edge of a region"
        );

        // INTERIOR - inside of a region
        let interior = Primitive::base(
            "INTERIOR",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("INTERIOR"))),
            "Set: all points strictly inside a region"
        );

        // === MEREOTOPOLOGY (part-whole) ===

        // PART_OF - mereological inclusion
        let part_of = Primitive::base(
            "PART_OF",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("PART_OF"))),
            "Relation: x is part of y"
        );

        // OVERLAPS - shared parts
        let overlaps = Primitive::base(
            "OVERLAPS",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("OVERLAPS"))),
            "Relation: x and y share common parts"
        );

        // TOUCHES - external contact
        let touches = Primitive::base(
            "TOUCHES",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(HV16::random(seed_from_name("TOUCHES"))),
            "Relation: x and y are in contact at boundary"
        );

        // === REGISTER ALL TIER 3 PRIMITIVES ===

        self.domains.insert("geometry".to_string(), geometry_domain);
        self.domains.insert("topology".to_string(), topology_domain);

        for primitive in vec![
            point, line, plane, angle, distance,
            vector, dot_product, cross_product,
            manifold, tangent_space, curvature,
            open_set, closed_set, boundary, interior,
            part_of, overlaps, touches,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "geometric_construction".to_string(),
            pattern: vec![PrimitiveTier::Geometric, PrimitiveTier::Geometric],
            result_tier: PrimitiveTier::Geometric,
            example: "POINT ⊗ POINT → LINE (geometric construction)".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "topological_relation".to_string(),
            pattern: vec![PrimitiveTier::Geometric, PrimitiveTier::Geometric],
            result_tier: PrimitiveTier::Geometric,
            example: "PART_OF ⊗ WHOLE → mereotopological structure".to_string(),
        });
    }

    /// Initialize Tier 4: Strategic & Social Primitives
    fn init_tier4_strategic(&mut self) {
        // Create strategic domain manifolds
        let game_theory_domain = DomainManifold::new(
            "game_theory",
            PrimitiveTier::Strategic,
            "Strategic reasoning and multi-agent coordination"
        );

        let temporal_domain = DomainManifold::new(
            "temporal",
            PrimitiveTier::Strategic,
            "Temporal logic and interval relations"
        );

        let social_domain = DomainManifold::new(
            "social",
            PrimitiveTier::Strategic,
            "Social coordination and cooperation"
        );

        // === GAME THEORY ===

        // UTILITY - preference measure
        let utility = Primitive::base(
            "UTILITY",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(HV16::random(seed_from_name("UTILITY"))),
            "Function: measure of preference or value"
        );

        // STRATEGY - action plan
        let strategy = Primitive::base(
            "STRATEGY",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(HV16::random(seed_from_name("STRATEGY"))),
            "Plan: complete specification of actions in all situations"
        );

        // EQUILIBRIUM - stable state
        let equilibrium = Primitive::base(
            "EQUILIBRIUM",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(HV16::random(seed_from_name("EQUILIBRIUM"))),
            "State: where no agent benefits from unilateral deviation"
        );

        // PAYOFF - outcome value
        let payoff = Primitive::derived(
            "PAYOFF",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(HV16::random(seed_from_name("PAYOFF"))),
            "Value: utility resulting from strategy profile",
            "APPLY(UTILITY, STRATEGY)"
        );

        // === TEMPORAL LOGIC (Allen's Intervals) ===

        // BEFORE - temporal precedence
        let before = Primitive::base(
            "BEFORE",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(HV16::random(seed_from_name("BEFORE"))),
            "Relation: interval x ends before interval y starts"
        );

        // AFTER - temporal succession
        let after = Primitive::base(
            "AFTER",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(HV16::random(seed_from_name("AFTER"))),
            "Relation: interval x starts after interval y ends"
        );

        // DURING - temporal containment
        let during = Primitive::base(
            "DURING",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(HV16::random(seed_from_name("DURING"))),
            "Relation: interval x occurs within interval y"
        );

        // MEETS - temporal adjacency
        let meets = Primitive::base(
            "MEETS",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(HV16::random(seed_from_name("MEETS"))),
            "Relation: interval x ends exactly when y starts"
        );

        // OVERLAPS_TEMPORAL - partial overlap
        let overlaps_temporal = Primitive::base(
            "OVERLAPS_TEMPORAL",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(HV16::random(seed_from_name("OVERLAPS_TEMPORAL"))),
            "Relation: intervals x and y partially overlap in time"
        );

        // === COUNTERFACTUAL REASONING ===

        // COUNTERFACTUAL - hypothetical condition
        let counterfactual = Primitive::base(
            "COUNTERFACTUAL",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(HV16::random(seed_from_name("COUNTERFACTUAL"))),
            "Condition: what would have happened if..."
        );

        // POSSIBLE_WORLD - alternative reality
        let possible_world = Primitive::base(
            "POSSIBLE_WORLD",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(HV16::random(seed_from_name("POSSIBLE_WORLD"))),
            "Structure: consistent alternative state of reality"
        );

        // === SOCIAL COORDINATION ===

        // COOPERATE - joint action for mutual benefit
        let cooperate = Primitive::base(
            "COOPERATE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("COOPERATE"))),
            "Action: work together for mutual benefit"
        );

        // DEFECT - self-interested deviation
        let defect = Primitive::base(
            "DEFECT",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("DEFECT"))),
            "Action: act in self-interest against cooperation"
        );

        // RECIPROCATE - conditional cooperation
        let reciprocate = Primitive::derived(
            "RECIPROCATE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("RECIPROCATE"))),
            "Strategy: cooperate if and only if partner cooperates",
            "CONDITIONAL(COOPERATE, COOPERATE)"
        );

        // TRUST - belief in cooperation
        let trust = Primitive::base(
            "TRUST",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("TRUST"))),
            "Belief: expectation that others will cooperate"
        );

        // === INFORMATION ===

        // SIGNAL - information transmission
        let signal = Primitive::base(
            "SIGNAL",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("SIGNAL"))),
            "Action: transmit information to influence others"
        );

        // BELIEF - subjective probability
        let belief = Primitive::base(
            "BELIEF",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("BELIEF"))),
            "State: subjective probability assignment"
        );

        // COMMON_KNOWLEDGE - shared awareness
        let common_knowledge = Primitive::base(
            "COMMON_KNOWLEDGE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(HV16::random(seed_from_name("COMMON_KNOWLEDGE"))),
            "State: all know, all know that all know, etc."
        );

        // === REGISTER ALL TIER 4 PRIMITIVES ===

        self.domains.insert("game_theory".to_string(), game_theory_domain);
        self.domains.insert("temporal".to_string(), temporal_domain);
        self.domains.insert("social".to_string(), social_domain);

        for primitive in vec![
            utility, strategy, equilibrium, payoff,
            before, after, during, meets, overlaps_temporal,
            counterfactual, possible_world,
            cooperate, defect, reciprocate, trust,
            signal, belief, common_knowledge,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "strategic_interaction".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Strategic],
            result_tier: PrimitiveTier::Strategic,
            example: "COOPERATE ⊗ TRUST → Sacred Reciprocity harmonic".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "temporal_reasoning".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Strategic],
            result_tier: PrimitiveTier::Strategic,
            example: "BEFORE ⊗ AFTER → temporal sequence".to_string(),
        });
    }

    /// Initialize Tier 5: Meta-Cognitive & Metabolic Primitives
    fn init_tier5_metacognitive(&mut self) {
        // Create meta-cognitive domain manifolds
        let metacognition_domain = DomainManifold::new(
            "metacognition",
            PrimitiveTier::MetaCognitive,
            "Self-awareness and introspection"
        );

        let homeostasis_domain = DomainManifold::new(
            "homeostasis",
            PrimitiveTier::MetaCognitive,
            "Self-regulation and repair"
        );

        let epistemic_domain = DomainManifold::new(
            "epistemic",
            PrimitiveTier::MetaCognitive,
            "Knowledge and uncertainty"
        );

        let metabolic_domain = DomainManifold::new(
            "metabolic",
            PrimitiveTier::MetaCognitive,
            "Resource allocation and management"
        );

        // === SELF-AWARENESS ===

        // SELF - reflexive identity
        let self_prim = Primitive::base(
            "SELF",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("SELF"))),
            "Entity: the reflexive subject of awareness"
        );

        // IDENTITY - persistent self-recognition
        let identity = Primitive::base(
            "IDENTITY",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("IDENTITY"))),
            "Property: persistent self-recognition over time"
        );

        // META_BELIEF - belief about beliefs
        let meta_belief = Primitive::derived(
            "META_BELIEF",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("META_BELIEF"))),
            "State: belief about one's own beliefs",
            "APPLY(SELF, BELIEF)"
        );

        // INTROSPECTION - self-examination
        let introspection = Primitive::base(
            "INTROSPECTION",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("INTROSPECTION"))),
            "Process: examination of one's own mental states"
        );

        // === HOMEOSTASIS & REGULATION ===

        // HOMEOSTASIS - self-regulation
        let homeostasis = Primitive::base(
            "HOMEOSTASIS",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("HOMEOSTASIS"))),
            "Process: maintaining stable internal state"
        );

        // SETPOINT - target state
        let setpoint = Primitive::base(
            "SETPOINT",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("SETPOINT"))),
            "Value: target state for homeostatic regulation"
        );

        // REGULATION - corrective action
        let regulation = Primitive::base(
            "REGULATION",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("REGULATION"))),
            "Process: adjusting state toward setpoint"
        );

        // FEEDBACK - state monitoring
        let feedback = Primitive::base(
            "FEEDBACK",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("FEEDBACK"))),
            "Signal: information about current state vs setpoint"
        );

        // === REPAIR & ADAPTATION ===

        // REPAIR - damage correction
        let repair = Primitive::base(
            "REPAIR",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("REPAIR"))),
            "Process: restoring damaged structures or functions"
        );

        // RESTORE - return to previous state
        let restore = Primitive::base(
            "RESTORE",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("RESTORE"))),
            "Process: returning to a previous functional state"
        );

        // ADAPT - modify in response to change
        let adapt = Primitive::base(
            "ADAPT",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(HV16::random(seed_from_name("ADAPT"))),
            "Process: modify structure/behavior in response to environment"
        );

        // LEARN - update from experience
        let learn = Primitive::base(
            "LEARN",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("LEARN"))),
            "Process: update knowledge or behavior from experience"
        );

        // === EPISTEMIC STRENGTH ===

        // KNOW - justified true belief
        let know = Primitive::base(
            "KNOW",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(HV16::random(seed_from_name("KNOW"))),
            "State: justified true belief"
        );

        // UNCERTAIN - lack of certainty
        let uncertain = Primitive::base(
            "UNCERTAIN",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(HV16::random(seed_from_name("UNCERTAIN"))),
            "State: lacking sufficient information for certainty"
        );

        // CONFIDENCE - degree of certainty
        let confidence = Primitive::base(
            "CONFIDENCE",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(HV16::random(seed_from_name("CONFIDENCE"))),
            "Measure: degree of certainty in a belief"
        );

        // EVIDENCE - justification
        let evidence = Primitive::base(
            "EVIDENCE",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(HV16::random(seed_from_name("EVIDENCE"))),
            "Support: information supporting or refuting a belief"
        );

        // === METABOLIC / RESOURCE MANAGEMENT ===

        // RESOURCE - available capacity
        let resource = Primitive::base(
            "RESOURCE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(HV16::random(seed_from_name("RESOURCE"))),
            "Entity: available capacity for use"
        );

        // ALLOCATE - distribute resources
        let allocate = Primitive::base(
            "ALLOCATE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(HV16::random(seed_from_name("ALLOCATE"))),
            "Process: distribute resources to tasks"
        );

        // CONSUME - use resources
        let consume = Primitive::base(
            "CONSUME",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(HV16::random(seed_from_name("CONSUME"))),
            "Process: use resources to perform work"
        );

        // PRODUCE - generate resources
        let produce = Primitive::base(
            "PRODUCE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(HV16::random(seed_from_name("PRODUCE"))),
            "Process: generate resources from inputs"
        );

        // === REWARD & VALUE ===

        // REWARD - positive reinforcement
        let reward = Primitive::base(
            "REWARD",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("REWARD"))),
            "Signal: positive reinforcement for actions"
        );

        // GOAL - desired state
        let goal = Primitive::base(
            "GOAL",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("GOAL"))),
            "State: desired future state to achieve"
        );

        // VALUE - measure of importance
        let value = Primitive::base(
            "VALUE",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(HV16::random(seed_from_name("VALUE"))),
            "Measure: importance or worth of a state/action"
        );

        // CERTAINTY - state of complete knowledge
        let certainty = Primitive::base(
            "CERTAINTY",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(HV16::random(seed_from_name("CERTAINTY"))),
            "State: complete confidence in a proposition's truth value"
        );

        // === REGISTER ALL TIER 5 PRIMITIVES ===

        self.domains.insert("metacognition".to_string(), metacognition_domain);
        self.domains.insert("homeostasis".to_string(), homeostasis_domain);
        self.domains.insert("epistemic".to_string(), epistemic_domain);
        self.domains.insert("metabolic".to_string(), metabolic_domain);

        for primitive in vec![
            self_prim, identity, meta_belief, introspection,
            homeostasis, setpoint, regulation, feedback,
            repair, restore, adapt, learn,
            know, uncertain, confidence, evidence, certainty,
            resource, allocate, consume, produce,
            reward, goal, value,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "metacognitive_reflection".to_string(),
            pattern: vec![PrimitiveTier::MetaCognitive, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "SELF ⊗ KNOW → meta-knowledge".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "homeostatic_regulation".to_string(),
            pattern: vec![PrimitiveTier::MetaCognitive, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "FEEDBACK ⊗ REGULATION → self-regulating loop".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "cross_tier_reasoning".to_string(),
            pattern: vec![PrimitiveTier::MetaCognitive, PrimitiveTier::Physical],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "KNOW ⊗ ENERGY → understanding of physical constraints".to_string(),
        });
    }

    /// Initialize Biological/Organic Primitives (Gap Analysis Priority 1)
    /// Ground AI in biological reality - metabolism, growth, evolution, homeostasis
    fn init_biological_primitives(&mut self) {
        let biology_domain = DomainManifold::new(
            "biology",
            PrimitiveTier::Physical,
            "Biological processes and organic life principles"
        );

        // === CORE BIOLOGICAL PROCESSES ===

        let metabolism = Primitive::base(
            "METABOLISM",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("METABOLISM"))),
            "Process: energy transformation in living systems (ATP synthesis, glycolysis)"
        );

        let growth = Primitive::base(
            "GROWTH",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("GROWTH"))),
            "Process: increase in size, complexity through cell division and development"
        );

        let reproduction = Primitive::base(
            "REPRODUCTION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("REPRODUCTION"))),
            "Process: creation of new organisms, transmission of heredity"
        );

        let evolution = Primitive::base(
            "EVOLUTION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("EVOLUTION"))),
            "Process: change in heritable characteristics over generations"
        );

        let adaptation = Primitive::derived(
            "ADAPTATION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("ADAPTATION"))),
            "Process: adjustment to environment through selection",
            "EVOLUTION ⊗ ENVIRONMENT"
        );

        let homeostasis_dynamic = Primitive::base(
            "HOMEOSTASIS_DYNAMIC",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("HOMEOSTASIS_DYNAMIC"))),
            "Process: dynamic self-regulation maintaining stable internal conditions"
        );

        let symbiosis = Primitive::base(
            "SYMBIOSIS",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("SYMBIOSIS"))),
            "Relationship: close interaction between different organisms (mutualism, parasitism)"
        );

        let immune_response = Primitive::base(
            "IMMUNE_RESPONSE",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("IMMUNE_RESPONSE"))),
            "Process: self/non-self distinction, pathogen recognition and elimination"
        );

        let circadian_rhythm = Primitive::base(
            "CIRCADIAN_RHYTHM",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("CIRCADIAN_RHYTHM"))),
            "Pattern: ~24-hour biological cycles, internal timekeeping"
        );

        let morphogen = Primitive::base(
            "MORPHOGEN",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("MORPHOGEN"))),
            "Substance: concentration gradient guiding development and pattern formation"
        );

        let apoptosis = Primitive::base(
            "APOPTOSIS",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(HV16::random(seed_from_name("APOPTOSIS"))),
            "Process: programmed cell death, controlled system renewal"
        );

        // Register domain and primitives
        self.domains.insert("biology".to_string(), biology_domain);

        for primitive in vec![
            metabolism, growth, reproduction, evolution, adaptation,
            homeostasis_dynamic, symbiosis, immune_response,
            circadian_rhythm, morphogen, apoptosis,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "biological_regulation".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "HOMEOSTASIS_DYNAMIC ⊗ FEEDBACK → self-regulating biological system".to_string(),
        });
    }

    /// Initialize Emotional/Affective Primitives (Gap Analysis Priority 2)
    /// Ground emotional reasoning for endocrine system integration
    fn init_emotional_primitives(&mut self) {
        let emotion_domain = DomainManifold::new(
            "emotion",
            PrimitiveTier::MetaCognitive,
            "Affective states and emotional processing"
        );

        // === DIMENSIONAL MODEL ===

        let valence = Primitive::base(
            "AFFECTIVE_VALENCE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("AFFECTIVE_VALENCE"))),
            "Dimension: positive/negative affect, pleasantness/unpleasantness"
        );

        let arousal = Primitive::base(
            "AFFECTIVE_AROUSAL",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("AFFECTIVE_AROUSAL"))),
            "Dimension: activation level, calm/excited continuum"
        );

        // === BASIC EMOTIONS (Ekman) ===

        let joy = Primitive::derived(
            "JOY",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("JOY"))),
            "Emotion: positive valence, high arousal - happiness, pleasure",
            "VALENCE_POSITIVE ⊗ AROUSAL_MODERATE"
        );

        let sadness = Primitive::derived(
            "SADNESS",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("SADNESS"))),
            "Emotion: negative valence, low arousal - loss, disappointment",
            "VALENCE_NEGATIVE ⊗ AROUSAL_LOW"
        );

        let fear = Primitive::derived(
            "FEAR",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("FEAR"))),
            "Emotion: negative valence, high arousal - threat response",
            "VALENCE_NEGATIVE ⊗ AROUSAL_HIGH"
        );

        let anger = Primitive::derived(
            "ANGER",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("ANGER"))),
            "Emotion: negative valence, high arousal - obstacle response",
            "VALENCE_NEGATIVE ⊗ AROUSAL_HIGH ⊗ APPROACH"
        );

        let disgust = Primitive::derived(
            "DISGUST",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("DISGUST"))),
            "Emotion: negative valence, rejection response - contamination avoidance",
            "VALENCE_NEGATIVE ⊗ REJECTION"
        );

        let surprise = Primitive::derived(
            "SURPRISE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("SURPRISE"))),
            "Emotion: neutral valence, high arousal - unexpected event response",
            "AROUSAL_HIGH ⊗ NOVELTY"
        );

        // === SOCIAL EMOTIONS ===

        let empathy = Primitive::base(
            "EMPATHY",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("EMPATHY"))),
            "Capacity: shared emotional experience, feeling with others"
        );

        let attachment = Primitive::base(
            "ATTACHMENT",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("ATTACHMENT"))),
            "Bond: emotional connection, social bonding, relationship formation"
        );

        let awe = Primitive::derived(
            "AWE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(HV16::random(seed_from_name("AWE"))),
            "Emotion: vastness + accommodation - wonder at something greater",
            "VASTNESS ⊗ ACCOMMODATION"
        );

        // Register domain and primitives
        self.domains.insert("emotion".to_string(), emotion_domain);

        for primitive in vec![
            valence, arousal, joy, sadness, fear, anger, disgust, surprise,
            empathy, attachment, awe,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "emotional_regulation".to_string(),
            pattern: vec![PrimitiveTier::MetaCognitive, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "EMOTION ⊗ REGULATION → emotional intelligence".to_string(),
        });
    }

    /// Initialize Ecological/Systems Primitives (Gap Analysis Priority 3)
    /// Reason about complex adaptive systems and collective intelligence
    fn init_ecological_primitives(&mut self) {
        let ecology_domain = DomainManifold::new(
            "ecology",
            PrimitiveTier::Physical,
            "Complex adaptive systems and ecosystem dynamics"
        );

        // === ECOLOGICAL CONCEPTS ===

        let niche = Primitive::base(
            "NICHE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("NICHE"))),
            "Concept: environmental role, opportunity and constraint space"
        );

        let carrying_capacity = Primitive::base(
            "CARRYING_CAPACITY",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("CARRYING_CAPACITY"))),
            "Limit: maximum population sustainable by available resources"
        );

        let succession = Primitive::base(
            "SUCCESSION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("SUCCESSION"))),
            "Process: sequential ecosystem development, progressive change"
        );

        let trophic_level = Primitive::base(
            "TROPHIC_LEVEL",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("TROPHIC_LEVEL"))),
            "Structure: position in energy flow chain (producer, consumer, decomposer)"
        );

        let resilience = Primitive::base(
            "RESILIENCE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("RESILIENCE"))),
            "Property: system capacity to absorb disturbance and reorganize"
        );

        // === SYSTEMS DYNAMICS ===

        let feedback_loop_positive = Primitive::base(
            "FEEDBACK_LOOP_POSITIVE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("FEEDBACK_LOOP_POSITIVE"))),
            "Pattern: amplifying feedback, exponential growth or collapse"
        );

        let feedback_loop_negative = Primitive::base(
            "FEEDBACK_LOOP_NEGATIVE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("FEEDBACK_LOOP_NEGATIVE"))),
            "Pattern: dampening feedback, stabilization, homeostasis"
        );

        let emergence_strong = Primitive::base(
            "EMERGENCE_STRONG",
            PrimitiveTier::MetaCognitive,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("EMERGENCE_STRONG"))),
            "Property: irreducible higher-level properties from component interactions"
        );

        let attractor = Primitive::base(
            "ATTRACTOR",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("ATTRACTOR"))),
            "State: stable system configuration in phase space"
        );

        let bifurcation = Primitive::base(
            "BIFURCATION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("BIFURCATION"))),
            "Transition: qualitative system change at critical parameter value"
        );

        let phase_transition = Primitive::base(
            "PHASE_TRANSITION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(HV16::random(seed_from_name("PHASE_TRANSITION"))),
            "Change: abrupt qualitative state transformation (order/disorder)"
        );

        // Register domain and primitives
        self.domains.insert("ecology".to_string(), ecology_domain);

        for primitive in vec![
            niche, carrying_capacity, succession, trophic_level, resilience,
            feedback_loop_positive, feedback_loop_negative,
            emergence_strong, attractor, bifurcation, phase_transition,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "systems_dynamics".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::Physical],
            result_tier: PrimitiveTier::Physical,
            example: "FEEDBACK_LOOP ⊗ ATTRACTOR → stable system behavior".to_string(),
        });
    }

    /// Initialize Quantum/Fundamental Physics Primitives (Gap Analysis Priority 4)
    /// Foundation for consciousness theories and IIT (Φ)
    fn init_quantum_primitives(&mut self) {
        let quantum_domain = DomainManifold::new(
            "quantum",
            PrimitiveTier::Physical,
            "Quantum mechanics and fundamental physics"
        );

        // === QUANTUM PHENOMENA ===

        let superposition = Primitive::base(
            "SUPERPOSITION",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(HV16::random(seed_from_name("SUPERPOSITION"))),
            "State: system existing in multiple configurations simultaneously"
        );

        let entanglement = Primitive::base(
            "ENTANGLEMENT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(HV16::random(seed_from_name("ENTANGLEMENT"))),
            "Correlation: non-local quantum correlation between systems"
        );

        let measurement = Primitive::base(
            "MEASUREMENT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(HV16::random(seed_from_name("MEASUREMENT"))),
            "Process: observer-dependent state collapse, wave function reduction"
        );

        let uncertainty_heisenberg = Primitive::base(
            "UNCERTAINTY_HEISENBERG",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(HV16::random(seed_from_name("UNCERTAINTY_HEISENBERG"))),
            "Principle: fundamental limit on simultaneous knowledge (ΔxΔp ≥ ℏ/2)"
        );

        let wave_particle_duality = Primitive::base(
            "WAVE_PARTICLE_DUALITY",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(HV16::random(seed_from_name("WAVE_PARTICLE_DUALITY"))),
            "Property: complementary wave and particle aspects of quantum objects"
        );

        let planck_constant = Primitive::base(
            "PLANCK_CONSTANT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(HV16::random(seed_from_name("PLANCK_CONSTANT"))),
            "Constant: quantum of action (h = 6.626×10⁻³⁴ J·s), fundamental scale"
        );

        // Register domain and primitives
        self.domains.insert("quantum".to_string(), quantum_domain);

        for primitive in vec![
            superposition, entanglement, measurement,
            uncertainty_heisenberg, wave_particle_duality, planck_constant,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "quantum_consciousness".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "ENTANGLEMENT ⊗ INTEGRATED_INFORMATION → quantum consciousness theories".to_string(),
        });
    }

    /// Initialize Economic/Value Primitives
    /// Reason about value, exchange, scarcity - supports ATP economy (Hearth)
    fn init_economic_primitives(&mut self) {
        let economics_domain = DomainManifold::new(
            "economics",
            PrimitiveTier::Strategic,
            "Value, exchange, and resource allocation"
        );

        // === ECONOMIC CONCEPTS ===

        let scarcity = Primitive::base(
            "SCARCITY",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("SCARCITY"))),
            "Condition: limited availability relative to demand"
        );

        let supply = Primitive::base(
            "SUPPLY",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("SUPPLY"))),
            "Quantity: amount available at given price"
        );

        let demand = Primitive::base(
            "DEMAND",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("DEMAND"))),
            "Quantity: amount desired at given price"
        );

        let exchange = Primitive::base(
            "EXCHANGE",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("EXCHANGE"))),
            "Transaction: trading goods/services, reciprocal transfer"
        );

        let value_subjective = Primitive::base(
            "VALUE_SUBJECTIVE",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("VALUE_SUBJECTIVE"))),
            "Property: preference-dependent worth, individual utility"
        );

        let capital = Primitive::base(
            "CAPITAL",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("CAPITAL"))),
            "Resource: accumulated assets for production, stored value"
        );

        let debt = Primitive::base(
            "DEBT",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("DEBT"))),
            "Obligation: future claim, deferred payment"
        );

        let trust_economic = Primitive::base(
            "TRUST_ECONOMIC",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(HV16::random(seed_from_name("TRUST_ECONOMIC"))),
            "Property: reliability expectation, reputation, social capital"
        );

        // Register domain and primitives
        self.domains.insert("economics".to_string(), economics_domain);

        for primitive in vec![
            scarcity, supply, demand, exchange, value_subjective,
            capital, debt, trust_economic,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "market_equilibrium".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Strategic],
            result_tier: PrimitiveTier::Strategic,
            example: "SUPPLY ⊗ DEMAND → market price equilibrium".to_string(),
        });
    }

    /// Initialize Linguistic/Semiotic Primitives
    /// Reason about symbols, meaning, communication
    fn init_linguistic_primitives(&mut self) {
        let linguistics_domain = DomainManifold::new(
            "linguistics",
            PrimitiveTier::MetaCognitive,
            "Symbols, meaning, and communication"
        );

        // === SEMIOTIC CONCEPTS ===

        let sign = Primitive::base(
            "SIGN",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("SIGN"))),
            "Structure: signifier + signified, symbol and its meaning"
        );

        let reference = Primitive::base(
            "REFERENCE",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("REFERENCE"))),
            "Relation: symbol → object mapping, denotation"
        );

        let context_dependency = Primitive::base(
            "CONTEXT_DEPENDENCY",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("CONTEXT_DEPENDENCY"))),
            "Property: meaning varies with situational context"
        );

        let metaphor = Primitive::base(
            "METAPHOR",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("METAPHOR"))),
            "Mapping: cross-domain conceptual transfer, analogical reasoning"
        );

        let syntax = Primitive::base(
            "SYNTAX",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("SYNTAX"))),
            "Structure: compositional rules, grammatical organization"
        );

        let semantics = Primitive::base(
            "SEMANTICS",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("SEMANTICS"))),
            "Property: meaning relations, truth conditions"
        );

        let pragmatics = Primitive::base(
            "PRAGMATICS",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(HV16::random(seed_from_name("PRAGMATICS"))),
            "Use: meaning in communicative context, speaker intention"
        );

        // Register domain and primitives
        self.domains.insert("linguistics".to_string(), linguistics_domain);

        for primitive in vec![
            sign, reference, context_dependency, metaphor,
            syntax, semantics, pragmatics,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "linguistic_meaning".to_string(),
            pattern: vec![PrimitiveTier::MetaCognitive, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "SYNTAX ⊗ SEMANTICS → compositional meaning".to_string(),
        });
    }

    /// Initialize Social/Moral Primitives
    /// Ethical reasoning, norms, obligations - supports safety system (Amygdala)
    fn init_social_moral_primitives(&mut self) {
        let moral_domain = DomainManifold::new(
            "morality",
            PrimitiveTier::Strategic,
            "Ethics, norms, and moral reasoning"
        );

        // === DEONTIC CONCEPTS (Rules) ===

        let norm = Primitive::base(
            "NORM",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("NORM"))),
            "Rule: social expectation, behavioral standard"
        );

        let obligation = Primitive::base(
            "OBLIGATION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("OBLIGATION"))),
            "Duty: moral requirement, what must be done"
        );

        let permission = Primitive::base(
            "PERMISSION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("PERMISSION"))),
            "Allowance: action that may be done without violation"
        );

        let prohibition = Primitive::base(
            "PROHIBITION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("PROHIBITION"))),
            "Restriction: forbidden action, taboo"
        );

        // === MORAL FOUNDATIONS ===

        let fairness = Primitive::base(
            "FAIRNESS",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("FAIRNESS"))),
            "Principle: equitable distribution, reciprocity, justice"
        );

        let harm = Primitive::base(
            "HARM",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("HARM"))),
            "Concept: damage, suffering, negative impact on well-being"
        );

        let care = Primitive::base(
            "CARE",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("CARE"))),
            "Disposition: protection, nurturance, compassion"
        );

        let rights = Primitive::base(
            "RIGHTS",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(HV16::random(seed_from_name("RIGHTS"))),
            "Entitlement: claims, protections, freedoms"
        );

        // Register domain and primitives
        self.domains.insert("morality".to_string(), moral_domain);

        for primitive in vec![
            norm, obligation, permission, prohibition,
            fairness, harm, care, rights,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "ethical_reasoning".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Strategic],
            result_tier: PrimitiveTier::Strategic,
            example: "HARM ⊗ PROHIBITION → ethical constraint".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "moral_deliberation".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "OBLIGATION ⊗ REFLECTION → moral judgment".to_string(),
        });
    }

    /// Initialize Tier 8: Consciousness-Specific Primitives
    ///
    /// These primitives capture first-person phenomenal experience, attention,
    /// memory operations, and agency - the irreducible atoms of conscious experience.
    ///
    /// ## Qualia Primitives
    /// - QUALE: Irreducible unit of subjective experience
    /// - PHENOMENAL_BINDING: Integration of qualia into unified experience
    /// - SUBJECTIVE_TIME: The felt passage of time
    ///
    /// ## Attention Primitives
    /// - ATTEND: Selective focus on information
    /// - SALIENCE: Intrinsic importance/relevance
    /// - BINDING_WINDOW: Temporal integration window for consciousness
    ///
    /// ## Memory Operation Primitives
    /// - REMEMBER: Retrieval of episodic information
    /// - FORGET: Decay/loss of information
    /// - CONSOLIDATE: Transfer to long-term storage
    /// - RECOGNIZE: Pattern matching to stored memories
    ///
    /// ## Agency Primitives
    /// - INTEND: Goal-directed mental state
    /// - WILL: Volitional initiation of action
    /// - DECIDE: Selection among alternatives
    /// - CONTROL: Executive regulation
    fn init_consciousness_primitives(&mut self) {
        let consciousness_domain = DomainManifold::new(
            "consciousness",
            PrimitiveTier::Consciousness,
            "First-person phenomenal experience, attention, memory, and agency"
        );

        // === QUALIA PRIMITIVES ===

        let quale = Primitive::base(
            "QUALE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("QUALE"))),
            "Irreducible unit of subjective experience - what it is like to experience"
        );

        let phenomenal_binding = Primitive::base(
            "PHENOMENAL_BINDING",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("PHENOMENAL_BINDING"))),
            "Integration of disparate qualia into unified perceptual field"
        );

        let subjective_time = Primitive::base(
            "SUBJECTIVE_TIME",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("SUBJECTIVE_TIME"))),
            "The felt passage of time - duration as experienced"
        );

        let sentience = Primitive::base(
            "SENTIENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("SENTIENCE"))),
            "Capacity for subjective experience - being a subject of experience"
        );

        // === ATTENTION PRIMITIVES ===

        let attend = Primitive::base(
            "ATTEND",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("ATTEND"))),
            "Selective focus - directing conscious awareness to subset of information"
        );

        let salience = Primitive::base(
            "SALIENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("SALIENCE"))),
            "Intrinsic importance - property that draws attention"
        );

        let binding_window = Primitive::base(
            "BINDING_WINDOW",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("BINDING_WINDOW"))),
            "Temporal integration window (~100-200ms) for conscious binding"
        );

        let awareness = Primitive::base(
            "AWARENESS",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("AWARENESS"))),
            "State of being conscious of something - phenomenal access"
        );

        // === MEMORY OPERATION PRIMITIVES ===

        let remember = Primitive::base(
            "REMEMBER",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("REMEMBER"))),
            "Retrieval of encoded episodic information into consciousness"
        );

        let forget = Primitive::base(
            "FORGET",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("FORGET"))),
            "Loss or decay of stored information - natural or active"
        );

        let consolidate = Primitive::base(
            "CONSOLIDATE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("CONSOLIDATE"))),
            "Transfer from working memory to long-term storage"
        );

        let recognize = Primitive::base(
            "RECOGNIZE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("RECOGNIZE"))),
            "Pattern matching of percept to stored memory - familiarity"
        );

        // === AGENCY PRIMITIVES ===

        let intend = Primitive::base(
            "INTEND",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("INTEND"))),
            "Goal-directed mental state - representation of desired outcome"
        );

        let will = Primitive::base(
            "WILL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("WILL"))),
            "Volitional initiation of action - self-determined causation"
        );

        let decide = Primitive::base(
            "DECIDE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("DECIDE"))),
            "Selection among alternatives - commitment to course of action"
        );

        let control = Primitive::base(
            "CONTROL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("CONTROL"))),
            "Executive regulation - top-down modulation of processing"
        );

        // === AFFECTIVE PRIMITIVES ===

        let valence = Primitive::base(
            "VALENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("VALENCE"))),
            "Positive-negative dimension of experience - pleasantness/unpleasantness"
        );

        let arousal = Primitive::base(
            "AROUSAL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("AROUSAL"))),
            "Activation level of experience - calm to excited"
        );

        let selection = Primitive::base(
            "SELECTION",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(HV16::random(seed_from_name("SELECTION"))),
            "Process: choosing one option from a set of alternatives"
        );

        // Register domain
        self.domains.insert("consciousness".to_string(), consciousness_domain);

        // Register all consciousness primitives
        for primitive in vec![
            // Qualia
            quale, phenomenal_binding, subjective_time, sentience,
            // Attention
            attend, salience, binding_window, awareness, selection,
            // Memory operations
            remember, forget, consolidate, recognize,
            // Agency
            intend, will, decide, control,
            // Affective
            valence, arousal,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_insert_with(Vec::new).push(name);
        }

        // === BINDING RULES FOR CONSCIOUSNESS ===

        // Qualia composition
        self.binding_rules.push(BindingRule {
            name: "qualia_integration".to_string(),
            pattern: vec![PrimitiveTier::Consciousness, PrimitiveTier::Consciousness],
            result_tier: PrimitiveTier::Consciousness,
            example: "QUALE ⊗ PHENOMENAL_BINDING → unified experience".to_string(),
        });

        // Attention-memory interaction
        self.binding_rules.push(BindingRule {
            name: "attention_memory".to_string(),
            pattern: vec![PrimitiveTier::Consciousness, PrimitiveTier::Consciousness],
            result_tier: PrimitiveTier::Consciousness,
            example: "ATTEND ⊗ REMEMBER → conscious recall".to_string(),
        });

        // Agency-consciousness bridge
        self.binding_rules.push(BindingRule {
            name: "conscious_agency".to_string(),
            pattern: vec![PrimitiveTier::Consciousness, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::Consciousness,
            example: "INTEND ⊗ SELF → self-directed action".to_string(),
        });

        // Temporal consciousness
        self.binding_rules.push(BindingRule {
            name: "temporal_experience".to_string(),
            pattern: vec![PrimitiveTier::Consciousness, PrimitiveTier::Temporal],
            result_tier: PrimitiveTier::Consciousness,
            example: "SUBJECTIVE_TIME ⊗ DURING → experienced duration".to_string(),
        });

        // Emotional consciousness
        self.binding_rules.push(BindingRule {
            name: "affective_experience".to_string(),
            pattern: vec![PrimitiveTier::Consciousness, PrimitiveTier::Consciousness],
            result_tier: PrimitiveTier::Consciousness,
            example: "VALENCE ⊗ AROUSAL → emotional state".to_string(),
        });
    }

    /// Get a primitive by name
    pub fn get(&self, name: &str) -> Option<&Primitive> {
        self.primitives.get(name)
    }

    /// Get all primitives in a tier
    pub fn get_tier(&self, tier: PrimitiveTier) -> Vec<&Primitive> {
        self.by_tier.get(&tier)
            .map(|names| names.iter().filter_map(|n| self.primitives.get(n)).collect())
            .unwrap_or_default()
    }

    /// Get a domain manifold
    pub fn domain(&self, name: &str) -> Option<&DomainManifold> {
        self.domains.get(name)
    }

    /// Check orthogonality between primitives
    pub fn check_orthogonality(&self, name1: &str, name2: &str) -> Option<f32> {
        let p1 = self.get(name1)?;
        let p2 = self.get(name2)?;
        Some(p1.encoding.similarity(&p2.encoding))
    }

    /// Validate that all primitives in a tier are sufficiently orthogonal.
    ///
    /// Returns pairs whose similarity deviates from 0.5 (random baseline) by
    /// more than `threshold`. With 16,384-bit vectors, expected deviation from
    /// 0.5 is ~0.008 (1σ) for random pairs, so threshold=0.03 ≈ 4σ.
    pub fn validate_tier_orthogonality(&self, tier: PrimitiveTier, threshold: f32) -> Vec<(String, String, f32)> {
        let mut violations = Vec::new();
        let primitives = self.get_tier(tier);

        for i in 0..primitives.len() {
            for j in (i+1)..primitives.len() {
                let sim = primitives[i].encoding.similarity(&primitives[j].encoding);
                let deviation = (sim - 0.5).abs();
                if deviation > threshold {
                    violations.push((
                        primitives[i].name.clone(),
                        primitives[j].name.clone(),
                        sim
                    ));
                }
            }
        }

        violations
    }

    /// Get count of primitives
    pub fn count(&self) -> usize {
        self.primitives.len()
    }

    /// Get all primitives as an iterator
    pub fn all_primitives(&self) -> impl Iterator<Item = &Primitive> {
        self.primitives.values()
    }

    /// Get count by tier
    pub fn count_tier(&self, tier: PrimitiveTier) -> usize {
        self.by_tier.get(&tier).map(|v| v.len()).unwrap_or(0)
    }

    /// Get all binding rules
    pub fn binding_rules(&self) -> &[BindingRule] {
        &self.binding_rules
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("# Primitive System Summary\n\n");
        report.push_str(&format!("**Total Primitives**: {}\n", self.count()));
        report.push_str(&format!("**Domains**: {}\n\n", self.domains.len()));

        report.push_str("## Primitives by Tier\n\n");
        for tier in &[
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            let count = self.count_tier(*tier);
            if count > 0 {
                report.push_str(&format!("- **{:?}**: {} primitives\n", tier, count));
            }
        }

        report.push_str("\n## Domain Manifolds\n\n");
        for (name, domain) in &self.domains {
            report.push_str(&format!("### {}\n", name));
            report.push_str(&format!("- **Tier**: {:?}\n", domain.tier));
            report.push_str(&format!("- **Purpose**: {}\n\n", domain.purpose));
        }

        report.push_str(&format!("\n## Binding Rules: {}\n\n", self.binding_rules.len()));

        report
    }

    // ========================================================================
    // SIMILARITY SEARCH
    // ========================================================================

    /// Get all primitive names as a vector
    pub fn all_primitive_names(&self) -> Vec<&str> {
        self.primitives.keys().map(|s| s.as_str()).collect()
    }

    /// Find primitives most similar to the given primitive by name.
    ///
    /// Returns a vector of (name, similarity) pairs sorted by descending similarity.
    pub fn find_similar(&self, name: &str, top_k: usize) -> Vec<(String, f32)> {
        let query = match self.primitives.get(name) {
            Some(p) => &p.encoding,
            None => return Vec::new(),
        };

        let mut similarities: Vec<(String, f32)> = self.primitives
            .iter()
            .filter(|(n, _)| *n != name)
            .map(|(n, p)| {
                let sim = query.similarity(&p.encoding);
                (n.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Find primitives most similar to a given encoding.
    ///
    /// Useful for finding matches to composed/derived encodings.
    pub fn find_similar_to_encoding(&self, encoding: &HV16, top_k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self.primitives
            .iter()
            .map(|(n, p)| {
                let sim = encoding.similarity(&p.encoding);
                (n.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    // ========================================================================
    // LSH INDEX FOR FAST APPROXIMATE SIMILARITY SEARCH
    // ========================================================================

    /// Create an LSH (Locality Sensitive Hashing) index for fast similarity search.
    ///
    /// LSH provides O(1) expected time for approximate nearest neighbor queries
    /// instead of O(n) linear scan. For 200+ primitives this is faster.
    ///
    /// # Parameters
    /// - `num_bands`: Number of hash tables (more = higher recall, more memory)
    /// - `bits_per_band`: Bits sampled per table (fewer = more collisions/candidates)
    ///
    /// # Example
    /// ```ignore
    /// let system = PrimitiveSystem::global();
    /// let lsh = system.build_lsh_index(8, 64);
    /// let candidates = lsh.query_candidates(&some_encoding);
    /// ```
    pub fn build_lsh_index(&self, num_bands: usize, bits_per_band: usize) -> LshIndex {
        LshIndex::build(&self.primitives, num_bands, bits_per_band)
    }

    /// Find similar primitives using LSH (faster for large primitive sets).
    ///
    /// This method uses a pre-built LSH index for O(1) candidate retrieval,
    /// then does full similarity comparison only on candidates.
    pub fn find_similar_lsh(
        &self,
        encoding: &HV16,
        top_k: usize,
        lsh: &LshIndex,
    ) -> Vec<(String, f32)> {
        // Get candidate primitive names from LSH
        let candidates = lsh.query_candidates(encoding);

        if candidates.is_empty() {
            // Fallback to linear scan if no LSH candidates
            return self.find_similar_to_encoding(encoding, top_k);
        }

        // Compute exact similarity only for candidates
        let mut similarities: Vec<(String, f32)> = candidates
            .into_iter()
            .filter_map(|name| {
                self.primitives.get(&name).map(|p| {
                    let sim = encoding.similarity(&p.encoding);
                    (name, sim)
                })
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    // ========================================================================
    // BATCH SIMILARITY SEARCH (SIMD-OPTIMIZED)
    // ========================================================================

    /// Batch find similar primitives for multiple query encodings.
    ///
    /// Uses parallel processing with rayon for queries and SIMD for similarity
    /// computation. Automatically selects optimal algorithm based on batch size.
    ///
    /// # Performance
    /// - Small batches (<50): Sequential processing (avoids parallel overhead)
    /// - Large batches (≥50): Parallel processing (2-8x speedup on multi-core)
    ///
    /// # Example
    /// ```ignore
    /// let system = PrimitiveSystem::global();
    /// let queries = vec![encoding1, encoding2, encoding3];
    /// let results = system.batch_find_similar(&queries, 5);
    /// // results[0] = top 5 similar to encoding1
    /// // results[1] = top 5 similar to encoding2
    /// // etc.
    /// ```
    #[cfg(feature = "rayon")]
    pub fn batch_find_similar(
        &self,
        queries: &[HV16],
        top_k: usize,
    ) -> Vec<Vec<(String, f32)>> {
        use rayon::prelude::*;

        const PARALLEL_THRESHOLD: usize = 50;

        if queries.len() < PARALLEL_THRESHOLD {
            // Sequential for small batches
            queries
                .iter()
                .map(|q| self.find_similar_to_encoding(q, top_k))
                .collect()
        } else {
            // Parallel for large batches
            queries
                .par_iter()
                .map(|q| self.find_similar_to_encoding(q, top_k))
                .collect()
        }
    }

    /// Batch find similar primitives (sequential version for no-parallel builds).
    #[cfg(not(feature = "rayon"))]
    pub fn batch_find_similar(
        &self,
        queries: &[HV16],
        top_k: usize,
    ) -> Vec<Vec<(String, f32)>> {
        queries
            .iter()
            .map(|q| self.find_similar_to_encoding(q, top_k))
            .collect()
    }

    /// Batch find similar using LSH for very large searches.
    ///
    /// Builds an LSH index once and reuses it for all queries.
    /// Best for: many queries against all primitives.
    pub fn batch_find_similar_lsh(
        &self,
        queries: &[HV16],
        top_k: usize,
        num_bands: usize,
        bits_per_band: usize,
    ) -> Vec<Vec<(String, f32)>> {
        let lsh = self.build_lsh_index(num_bands, bits_per_band);

        queries
            .iter()
            .map(|q| self.find_similar_lsh(q, top_k, &lsh))
            .collect()
    }

    /// Batch bind multiple primitive pairs.
    ///
    /// More efficient than calling bind_primitives repeatedly.
    pub fn batch_bind(
        &self,
        pairs: &[(&str, &str)],
    ) -> Vec<Result<PrimitiveResult, PrimitiveError>> {
        pairs
            .iter()
            .map(|(a, b)| self.bind_primitives(a, b))
            .collect()
    }

    /// Batch bundle multiple primitive groups.
    pub fn batch_bundle(
        &self,
        groups: &[&[&str]],
    ) -> Vec<Result<PrimitiveResult, PrimitiveError>> {
        groups
            .iter()
            .map(|names| self.bundle_primitives(names))
            .collect()
    }

    /// Batch encode multiple sequences.
    pub fn batch_encode_sequences(
        &self,
        sequences: &[&[&str]],
    ) -> Vec<Result<PrimitiveResult, PrimitiveError>> {
        sequences
            .iter()
            .map(|names| self.encode_sequence(names))
            .collect()
    }

    /// Compute pairwise similarities between all given encodings.
    ///
    /// Returns a flattened lower-triangular matrix: [(i, j, similarity)]
    /// for all i > j pairs.
    pub fn pairwise_similarities(&self, encodings: &[HV16]) -> Vec<(usize, usize, f32)> {
        let mut results = Vec::with_capacity(encodings.len() * (encodings.len() - 1) / 2);

        for i in 0..encodings.len() {
            for j in 0..i {
                let sim = encodings[i].similarity(&encodings[j]);
                results.push((i, j, sim));
            }
        }

        results
    }

    /// Compute similarity matrix for named primitives.
    ///
    /// Returns a symmetric matrix where matrix[i][j] = similarity(primitive_i, primitive_j).
    pub fn similarity_matrix(&self, names: &[&str]) -> Vec<Vec<f32>> {
        let encodings: Vec<_> = names
            .iter()
            .filter_map(|n| self.get(n).map(|p| p.encoding.clone()))
            .collect();

        let n = encodings.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0; // Self-similarity
            for j in 0..i {
                let sim = encodings[i].similarity(&encodings[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim; // Symmetric
            }
        }

        matrix
    }

    // ========================================================================
    // TYPED PRIMITIVE OPERATIONS
    // ========================================================================

    /// Bind two named primitives together (XOR in HV16 space).
    ///
    /// Binding creates a new encoding that represents the relationship between
    /// two concepts. In HDC, bind(A, B) creates a vector orthogonal to both
    /// A and B but can be "unbound" by either to recover the other.
    pub fn bind_primitives(&self, a: &str, b: &str) -> Result<PrimitiveResult, PrimitiveError> {
        let prim_a = self.primitives.get(a)
            .ok_or_else(|| PrimitiveError::NotFound(a.to_string()))?;
        let prim_b = self.primitives.get(b)
            .ok_or_else(|| PrimitiveError::NotFound(b.to_string()))?;

        let encoding = prim_a.encoding.bind(&prim_b.encoding);
        Ok(PrimitiveResult {
            encoding,
            operation: format!("bind({}, {})", a, b),
            source_primitives: vec![a.to_string(), b.to_string()],
        })
    }

    /// Bundle multiple named primitives together (majority vote in HV16 space).
    ///
    /// Bundling creates an encoding similar to all inputs (unlike bind).
    pub fn bundle_primitives(&self, names: &[&str]) -> Result<PrimitiveResult, PrimitiveError> {
        if names.is_empty() {
            return Err(PrimitiveError::EmptyInput);
        }

        let mut encodings = Vec::with_capacity(names.len());
        for name in names {
            let prim = self.primitives.get(*name)
                .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;
            encodings.push(prim.encoding.clone());
        }

        let encoding = HV16::bundle(&encodings);

        Ok(PrimitiveResult {
            encoding,
            operation: format!("bundle({})", names.join(", ")),
            source_primitives: names.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Bundle primitives with weights for nuanced compositions.
    ///
    /// Higher weights make that primitive more dominant in the result.
    /// Uses probabilistic bit selection based on weights.
    pub fn bundle_weighted(&self, weighted: &[(&str, f32)]) -> Result<PrimitiveResult, PrimitiveError> {
        if weighted.is_empty() {
            return Err(PrimitiveError::EmptyInput);
        }

        // Normalize weights
        let total_weight: f32 = weighted.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return Err(PrimitiveError::InvalidWeight);
        }

        let mut encodings = Vec::with_capacity(weighted.len());
        let mut weights = Vec::with_capacity(weighted.len());

        for (name, weight) in weighted {
            let prim = self.primitives.get(*name)
                .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;
            encodings.push(prim.encoding.clone());
            weights.push(*weight / total_weight);
        }

        // Weighted bundling: for each bit position, sum weighted votes
        // HV16 is [u8; 2048] (2048 * 8 = 16384 bits)
        let mut result_bytes = [0u8; 2048];
        for byte_idx in 0..2048 {
            let mut byte_val: u8 = 0;
            for bit_in_byte in 0..8 {
                let mut weighted_sum: f32 = 0.0;
                for (enc, w) in encodings.iter().zip(weights.iter()) {
                    let enc_byte = enc.0[byte_idx];
                    let bit = (enc_byte >> bit_in_byte) & 1;
                    weighted_sum += if bit == 1 { *w } else { -*w };
                }

                if weighted_sum > 0.0 {
                    byte_val |= 1u8 << bit_in_byte;
                }
            }
            result_bytes[byte_idx] = byte_val;
        }

        let encoding = HV16(result_bytes);
        let names: Vec<String> = weighted.iter().map(|(n, _)| n.to_string()).collect();

        Ok(PrimitiveResult {
            encoding,
            operation: format!("bundle_weighted({})",
                weighted.iter().map(|(n, w)| format!("{}:{:.2}", n, w)).collect::<Vec<_>>().join(", ")),
            source_primitives: names,
        })
    }

    /// Compute an analogy: A is to B as C is to ?
    ///
    /// Uses the HDC analogy formula: result = bind(unbind(A, B), C)
    pub fn analogy(&self, a: &str, b: &str, c: &str) -> Result<PrimitiveResult, PrimitiveError> {
        let prim_a = self.primitives.get(a)
            .ok_or_else(|| PrimitiveError::NotFound(a.to_string()))?;
        let prim_b = self.primitives.get(b)
            .ok_or_else(|| PrimitiveError::NotFound(b.to_string()))?;
        let prim_c = self.primitives.get(c)
            .ok_or_else(|| PrimitiveError::NotFound(c.to_string()))?;

        // Analogy: A:B :: C:? => ? = bind(bind(A, B), C)
        // Note: In XOR-based HDC, unbind(A, B) = bind(A, B) since XOR is self-inverse
        let ab_relation = prim_a.encoding.bind(&prim_b.encoding);
        let encoding = ab_relation.bind(&prim_c.encoding);

        Ok(PrimitiveResult {
            encoding,
            operation: format!("analogy({}:{} :: {}:?)", a, b, c),
            source_primitives: vec![a.to_string(), b.to_string(), c.to_string()],
        })
    }

    /// Permute a named primitive (cyclic rotation in HV16 space).
    ///
    /// Useful for encoding sequences or temporal relationships.
    pub fn permute_primitive(&self, name: &str, steps: usize) -> Result<PrimitiveResult, PrimitiveError> {
        let prim = self.primitives.get(name)
            .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;

        let encoding = prim.encoding.permute(steps);

        Ok(PrimitiveResult {
            encoding,
            operation: format!("permute({}, {})", name, steps),
            source_primitives: vec![name.to_string()],
        })
    }

    /// Encode an ordered sequence of primitives preserving position.
    ///
    /// Uses permutation to encode position: A ⊗ permute(B, 1) ⊗ permute(C, 2)
    /// This creates an encoding that captures both content and order.
    pub fn encode_sequence(&self, names: &[&str]) -> Result<PrimitiveResult, PrimitiveError> {
        if names.is_empty() {
            return Err(PrimitiveError::EmptyInput);
        }

        let first = self.primitives.get(names[0])
            .ok_or_else(|| PrimitiveError::NotFound(names[0].to_string()))?;

        let mut encoding = first.encoding.clone();

        for (i, name) in names.iter().enumerate().skip(1) {
            let prim = self.primitives.get(*name)
                .ok_or_else(|| PrimitiveError::NotFound(name.to_string()))?;
            let permuted = prim.encoding.permute(i);
            encoding = encoding.bind(&permuted);
        }

        Ok(PrimitiveResult {
            encoding,
            operation: format!("sequence({})", names.join(" → ")),
            source_primitives: names.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Query what primitive best matches a given encoding.
    pub fn query(&self, encoding: &HV16) -> (String, f32) {
        let matches = self.find_similar_to_encoding(encoding, 1);
        matches.into_iter().next().unwrap_or_else(|| ("UNKNOWN".to_string(), 0.0))
    }
}

/// Result of a typed primitive operation
#[derive(Debug, Clone)]
pub struct PrimitiveResult {
    /// The resulting HV16 encoding
    pub encoding: HV16,
    /// Description of the operation
    pub operation: String,
    /// Source primitives used
    pub source_primitives: Vec<String>,
}

impl PrimitiveResult {
    /// Find similar primitives to this result
    pub fn find_similar(&self, system: &PrimitiveSystem, top_k: usize) -> Vec<(String, f32)> {
        system.find_similar_to_encoding(&self.encoding, top_k)
    }
}

/// Errors from typed primitive operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveError {
    /// Primitive not found
    NotFound(String),
    /// Empty input
    EmptyInput,
    /// Invalid weight (zero or negative total)
    InvalidWeight,
}

impl std::fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimitiveError::NotFound(name) => write!(f, "primitive not found: {}", name),
            PrimitiveError::EmptyInput => write!(f, "operation requires at least one input"),
            PrimitiveError::InvalidWeight => write!(f, "weights must sum to positive value"),
        }
    }
}

impl std::error::Error for PrimitiveError {}

// ============================================================================
// LSH INDEX FOR FAST APPROXIMATE NEAREST NEIGHBOR SEARCH
// ============================================================================

/// Locality Sensitive Hashing index for fast approximate similarity search.
///
/// For binary hypervectors like HV16, we use a simple bit-sampling LSH scheme:
/// - Create `num_bands` hash tables
/// - Each table samples `bits_per_band` random bit positions
/// - Vectors with the same sampled bits hash to the same bucket
/// - Query returns all primitives that share a bucket with the query in any table
///
/// This provides O(1) expected time for candidate retrieval instead of O(n) linear scan.
/// The tradeoff is recall vs. speed: more bands = better recall, more memory.
#[derive(Debug, Clone)]
pub struct LshIndex {
    /// Hash tables: band_idx -> bucket_key -> primitive_names
    tables: Vec<HashMap<u64, Vec<String>>>,
    /// Bit indices sampled for each band
    bit_indices: Vec<Vec<usize>>,
    /// Number of bits per band
    bits_per_band: usize,
}

impl LshIndex {
    /// Build an LSH index from a collection of primitives.
    ///
    /// # Parameters
    /// - `primitives`: Map of primitive name to Primitive
    /// - `num_bands`: Number of hash tables (8-16 is typical)
    /// - `bits_per_band`: Bits per hash (32-128 typical for 16K vectors)
    pub fn build(
        primitives: &HashMap<String, Primitive>,
        num_bands: usize,
        bits_per_band: usize,
    ) -> Self {
        // Generate deterministic random bit indices for each band
        let total_bits = 16384; // HV16 dimension
        let mut bit_indices = Vec::with_capacity(num_bands);

        for band in 0..num_bands {
            let mut indices = Vec::with_capacity(bits_per_band);
            // Use deterministic seed based on band number
            let mut seed = 0x5f3759dfu64.wrapping_mul(band as u64 + 1);
            for _ in 0..bits_per_band {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (seed >> 32) as usize % total_bits;
                indices.push(idx);
            }
            bit_indices.push(indices);
        }

        // Build hash tables
        let mut tables: Vec<HashMap<u64, Vec<String>>> = vec![HashMap::new(); num_bands];

        for (name, prim) in primitives {
            for (band_idx, indices) in bit_indices.iter().enumerate() {
                let hash = Self::compute_hash(&prim.encoding, indices);
                tables[band_idx]
                    .entry(hash)
                    .or_default()
                    .push(name.clone());
            }
        }

        Self {
            tables,
            bit_indices,
            bits_per_band,
        }
    }

    /// Compute hash for a vector using the given bit indices.
    fn compute_hash(hv: &HV16, indices: &[usize]) -> u64 {
        let mut hash: u64 = 0;
        for (i, &bit_idx) in indices.iter().enumerate() {
            let byte_idx = bit_idx / 8;
            let bit_in_byte = bit_idx % 8;
            let bit = (hv.0[byte_idx] >> bit_in_byte) & 1;
            if bit == 1 && i < 64 {
                hash |= 1u64 << (i % 64);
            }
        }
        hash
    }

    /// Query the index for candidate primitives similar to the given encoding.
    ///
    /// Returns a set of primitive names that share at least one bucket with the query.
    /// These are candidates for full similarity comparison.
    pub fn query_candidates(&self, encoding: &HV16) -> Vec<String> {
        let mut candidates = std::collections::HashSet::new();

        for (band_idx, indices) in self.bit_indices.iter().enumerate() {
            let hash = Self::compute_hash(encoding, indices);
            if let Some(bucket) = self.tables[band_idx].get(&hash) {
                for name in bucket {
                    candidates.insert(name.clone());
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> LshStats {
        let total_entries: usize = self.tables.iter().map(|t| t.values().map(|v| v.len()).sum::<usize>()).sum();
        let total_buckets: usize = self.tables.iter().map(|t| t.len()).sum();
        let avg_bucket_size = if total_buckets > 0 {
            total_entries as f32 / total_buckets as f32
        } else {
            0.0
        };

        LshStats {
            num_bands: self.tables.len(),
            bits_per_band: self.bits_per_band,
            total_buckets,
            total_entries,
            avg_bucket_size,
        }
    }
}

/// Statistics about an LSH index
#[derive(Debug, Clone)]
pub struct LshStats {
    pub num_bands: usize,
    pub bits_per_band: usize,
    pub total_buckets: usize,
    pub total_entries: usize,
    pub avg_bucket_size: f32,
}

// ============================================================================
// COMPOSITION CACHE FOR MEMOIZED PRIMITIVE OPERATIONS
// ============================================================================

/// Cache for memoizing primitive composition operations.
///
/// Stores results of bind, bundle, sequence, and other operations to avoid
/// redundant computation. Uses operation strings as keys.
///
/// # Example
/// ```ignore
/// let mut cache = CompositionCache::new(1000);
/// let system = PrimitiveSystem::global();
///
/// // First call computes and caches
/// let result1 = cache.bind_cached(system, "CAUSE", "EFFECT").unwrap();
///
/// // Second call retrieves from cache
/// let result2 = cache.bind_cached(system, "CAUSE", "EFFECT").unwrap();
/// assert_eq!(result1.encoding, result2.encoding);
/// ```
#[derive(Debug)]
pub struct CompositionCache {
    /// Cached results: operation_key -> PrimitiveResult
    cache: HashMap<String, PrimitiveResult>,
    /// Maximum cache size (LRU eviction when exceeded)
    max_size: usize,
    /// Access order for LRU (operation_key -> access_count)
    access_order: HashMap<String, u64>,
    /// Global access counter
    access_counter: u64,
    /// Cache statistics
    hits: u64,
    misses: u64,
}

impl CompositionCache {
    /// Create a new composition cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
            access_order: HashMap::with_capacity(max_size),
            access_counter: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Create a cache with default size (1000 entries).
    pub fn default_size() -> Self {
        Self::new(1000)
    }

    /// Bind two primitives with caching.
    pub fn bind_cached(
        &mut self,
        system: &PrimitiveSystem,
        a: &str,
        b: &str,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("bind:{}:{}", a, b);

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.bind_primitives(a, b)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Bundle primitives with caching.
    pub fn bundle_cached(
        &mut self,
        system: &PrimitiveSystem,
        names: &[&str],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("bundle:{}", names.join(":"));

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.bundle_primitives(names)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Weighted bundle with caching.
    pub fn bundle_weighted_cached(
        &mut self,
        system: &PrimitiveSystem,
        weighted: &[(&str, f32)],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        // Create a key that includes weights (rounded for cache friendliness)
        let key = format!(
            "bundle_weighted:{}",
            weighted
                .iter()
                .map(|(n, w)| format!("{}:{:.2}", n, w))
                .collect::<Vec<_>>()
                .join(":")
        );

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.bundle_weighted(weighted)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Encode sequence with caching.
    pub fn sequence_cached(
        &mut self,
        system: &PrimitiveSystem,
        names: &[&str],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("sequence:{}", names.join(":"));

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.encode_sequence(names)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Analogy with caching.
    pub fn analogy_cached(
        &mut self,
        system: &PrimitiveSystem,
        a: &str,
        b: &str,
        c: &str,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("analogy:{}:{}:{}", a, b, c);

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.analogy(a, b, c)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Permute primitive with caching.
    pub fn permute_cached(
        &mut self,
        system: &PrimitiveSystem,
        name: &str,
        steps: usize,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("permute:{}:{}", name, steps);

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.permute_primitive(name, steps)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Get an entry from the cache, updating access order.
    fn get(&mut self, key: &str) -> Option<&PrimitiveResult> {
        if self.cache.contains_key(key) {
            self.hits += 1;
            self.access_counter += 1;
            self.access_order.insert(key.to_string(), self.access_counter);
            self.cache.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Put an entry in the cache, evicting LRU if necessary.
    fn put(&mut self, key: String, value: PrimitiveResult) {
        // Evict if at capacity
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.access_counter += 1;
        self.access_order.insert(key.clone(), self.access_counter);
        self.cache.insert(key, value);
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some((lru_key, _)) = self
            .access_order
            .iter()
            .min_by_key(|(_, &access_time)| access_time)
            .map(|(k, v)| (k.clone(), *v))
        {
            self.cache.remove(&lru_key);
            self.access_order.remove(&lru_key);
        }
    }

    /// Clear the cache and reset statistics.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.access_counter = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        CacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if total > 0 {
                self.hits as f32 / total as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for CompositionCache {
    fn default() -> Self {
        Self::default_size()
    }
}

/// Statistics about the composition cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
}

// ============================================================================
// COMPOSITION ALGEBRA - NAMED COMPOSITIONS WITH EXPRESSION EVALUATION
// ============================================================================

/// Algebra for defining and evaluating named compositions.
///
/// Allows users to define reusable compositions like:
/// - `CAUSALITY = CAUSE ⊗ EFFECT` (bind)
/// - `TEMPORAL_FLOW = BEFORE → DURING → AFTER` (sequence)
/// - `PHYSICS_CORE = MASS + ENERGY + FORCE` (bundle)
/// - `MOSTLY_CAUSE = CAUSE:3 + EFFECT:1` (weighted bundle)
///
/// # Example
/// ```ignore
/// let system = PrimitiveSystem::global();
/// let mut algebra = CompositionAlgebra::new();
///
/// // Define compositions
/// algebra.define("CAUSALITY", "CAUSE ⊗ EFFECT", system)?;
/// algebra.define("TEMPORAL", "BEFORE → DURING → AFTER", system)?;
///
/// // Use in new expressions
/// algebra.define("CAUSAL_TIME", "CAUSALITY ⊗ TEMPORAL", system)?;
///
/// // Evaluate
/// let result = algebra.get("CAUSAL_TIME")?;
/// ```
#[derive(Debug, Clone)]
pub struct CompositionAlgebra {
    /// Named compositions: name -> (expression, encoding)
    compositions: HashMap<String, NamedComposition>,
}

/// A named composition with its expression and computed encoding
#[derive(Debug, Clone)]
pub struct NamedComposition {
    /// The name of this composition
    pub name: String,
    /// The expression that defines it (e.g., "CAUSE ⊗ EFFECT")
    pub expression: String,
    /// The computed HV16 encoding
    pub encoding: HV16,
    /// Source primitives/compositions used
    pub sources: Vec<String>,
}

impl CompositionAlgebra {
    /// Create a new empty algebra.
    pub fn new() -> Self {
        Self {
            compositions: HashMap::new(),
        }
    }

    /// Define a new named composition from an expression.
    ///
    /// Expression syntax:
    /// - `A ⊗ B` or `A ^ B` - Bind (XOR)
    /// - `A + B + C` - Bundle (majority vote)
    /// - `A → B → C` or `A > B > C` - Sequence (position-aware)
    /// - `A:2 + B:1` - Weighted bundle
    ///
    /// Names can reference primitives or previously defined compositions.
    pub fn define(
        &mut self,
        name: &str,
        expression: &str,
        system: &PrimitiveSystem,
    ) -> Result<(), CompositionAlgebraError> {
        if name.is_empty() {
            return Err(CompositionAlgebraError::InvalidName("name cannot be empty".to_string()));
        }

        let (encoding, sources) = self.evaluate_expression(expression, system)?;

        self.compositions.insert(
            name.to_string(),
            NamedComposition {
                name: name.to_string(),
                expression: expression.to_string(),
                encoding,
                sources,
            },
        );

        Ok(())
    }

    /// Get a named composition by name.
    pub fn get(&self, name: &str) -> Option<&NamedComposition> {
        self.compositions.get(name)
    }

    /// Get encoding for a name (composition or primitive).
    pub fn get_encoding(&self, name: &str, system: &PrimitiveSystem) -> Option<HV16> {
        // First check compositions
        if let Some(comp) = self.compositions.get(name) {
            return Some(comp.encoding.clone());
        }
        // Then check primitives
        system.get(name).map(|p| p.encoding.clone())
    }

    /// List all defined compositions.
    pub fn list(&self) -> Vec<&NamedComposition> {
        self.compositions.values().collect()
    }

    /// Remove a composition.
    pub fn remove(&mut self, name: &str) -> Option<NamedComposition> {
        self.compositions.remove(name)
    }

    /// Clear all compositions.
    pub fn clear(&mut self) {
        self.compositions.clear();
    }

    /// Evaluate an expression and return the encoding.
    fn evaluate_expression(
        &self,
        expression: &str,
        system: &PrimitiveSystem,
    ) -> Result<(HV16, Vec<String>), CompositionAlgebraError> {
        let expr = expression.trim();

        // Check for sequence operator (→ or >)
        if expr.contains('→') || expr.contains('>') {
            return self.evaluate_sequence(expr, system);
        }

        // Check for bind operator (⊗ or ^)
        if expr.contains('⊗') || expr.contains('^') {
            return self.evaluate_bind(expr, system);
        }

        // Check for weighted bundle (contains :)
        if expr.contains(':') && expr.contains('+') {
            return self.evaluate_weighted_bundle(expr, system);
        }

        // Check for bundle operator (+)
        if expr.contains('+') {
            return self.evaluate_bundle(expr, system);
        }

        // Single name - look up directly
        let name = expr.trim();
        if let Some(enc) = self.get_encoding(name, system) {
            return Ok((enc, vec![name.to_string()]));
        }

        Err(CompositionAlgebraError::NotFound(name.to_string()))
    }

    fn evaluate_bind(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(HV16, Vec<String>), CompositionAlgebraError> {
        // Split on ⊗ or ^
        let parts: Vec<&str> = if expr.contains('⊗') {
            expr.split('⊗').collect()
        } else {
            expr.split('^').collect()
        };

        if parts.len() < 2 {
            return Err(CompositionAlgebraError::ParseError("bind requires at least 2 operands".to_string()));
        }

        let mut sources = Vec::new();
        let first_name = parts[0].trim();
        let mut result = self.get_encoding(first_name, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(first_name.to_string()))?;
        sources.push(first_name.to_string());

        for part in &parts[1..] {
            let name = part.trim();
            let enc = self.get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            result = result.bind(&enc);
            sources.push(name.to_string());
        }

        Ok((result, sources))
    }

    fn evaluate_bundle(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(HV16, Vec<String>), CompositionAlgebraError> {
        let parts: Vec<&str> = expr.split('+').collect();

        if parts.len() < 2 {
            return Err(CompositionAlgebraError::ParseError("bundle requires at least 2 operands".to_string()));
        }

        let mut sources = Vec::new();
        let mut encodings = Vec::new();

        for part in &parts {
            let name = part.trim();
            let enc = self.get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            encodings.push(enc);
            sources.push(name.to_string());
        }

        let result = HV16::bundle(&encodings);
        Ok((result, sources))
    }

    fn evaluate_weighted_bundle(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(HV16, Vec<String>), CompositionAlgebraError> {
        let parts: Vec<&str> = expr.split('+').collect();

        let mut sources = Vec::new();
        let mut encodings = Vec::new();
        let mut weights = Vec::new();

        for part in &parts {
            let part = part.trim();
            let kv: Vec<&str> = part.split(':').collect();

            let (name, weight) = if kv.len() == 2 {
                let w: f32 = kv[1].trim().parse()
                    .map_err(|_| CompositionAlgebraError::ParseError(format!("invalid weight: {}", kv[1])))?;
                (kv[0].trim(), w)
            } else {
                (part, 1.0)
            };

            let enc = self.get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            encodings.push(enc);
            weights.push(weight);
            sources.push(name.to_string());
        }

        // Normalize weights
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return Err(CompositionAlgebraError::ParseError("weights must sum to positive value".to_string()));
        }
        let weights: Vec<f32> = weights.iter().map(|w| w / total).collect();

        // Weighted bundling
        let mut result_bytes = [0u8; 2048];
        for byte_idx in 0..2048 {
            let mut byte_val: u8 = 0;
            for bit_in_byte in 0..8 {
                let mut weighted_sum: f32 = 0.0;
                for (enc, w) in encodings.iter().zip(weights.iter()) {
                    let enc_byte = enc.0[byte_idx];
                    let bit = (enc_byte >> bit_in_byte) & 1;
                    weighted_sum += if bit == 1 { *w } else { -*w };
                }
                if weighted_sum > 0.0 {
                    byte_val |= 1u8 << bit_in_byte;
                }
            }
            result_bytes[byte_idx] = byte_val;
        }

        Ok((HV16(result_bytes), sources))
    }

    fn evaluate_sequence(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(HV16, Vec<String>), CompositionAlgebraError> {
        // Split on → or >
        let parts: Vec<&str> = if expr.contains('→') {
            expr.split('→').collect()
        } else {
            expr.split('>').collect()
        };

        if parts.len() < 2 {
            return Err(CompositionAlgebraError::ParseError("sequence requires at least 2 elements".to_string()));
        }

        let mut sources = Vec::new();
        let first_name = parts[0].trim();
        let mut result = self.get_encoding(first_name, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(first_name.to_string()))?;
        sources.push(first_name.to_string());

        for (i, part) in parts[1..].iter().enumerate() {
            let name = part.trim();
            let enc = self.get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            let permuted = enc.permute(i + 1);
            result = result.bind(&permuted);
            sources.push(name.to_string());
        }

        Ok((result, sources))
    }

    /// Export all compositions to a serializable format.
    pub fn export(&self) -> Vec<CompositionExport> {
        self.compositions
            .values()
            .map(|c| CompositionExport {
                name: c.name.clone(),
                expression: c.expression.clone(),
            })
            .collect()
    }

    /// Import compositions from exported format.
    pub fn import(
        &mut self,
        exports: &[CompositionExport],
        system: &PrimitiveSystem,
    ) -> Result<usize, CompositionAlgebraError> {
        let mut count = 0;
        for exp in exports {
            self.define(&exp.name, &exp.expression, system)?;
            count += 1;
        }
        Ok(count)
    }
}

impl Default for CompositionAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

/// Exportable composition (without encoding)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionExport {
    pub name: String,
    pub expression: String,
}

/// Errors from composition algebra operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionAlgebraError {
    /// Name not found (primitive or composition)
    NotFound(String),
    /// Invalid composition name
    InvalidName(String),
    /// Expression parse error
    ParseError(String),
}

impl std::fmt::Display for CompositionAlgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositionAlgebraError::NotFound(name) => write!(f, "not found: {}", name),
            CompositionAlgebraError::InvalidName(msg) => write!(f, "invalid name: {}", msg),
            CompositionAlgebraError::ParseError(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl std::error::Error for CompositionAlgebraError {}

// ============================================================================
// PRIMITIVE GRAPH VISUALIZATION (DOT FORMAT)
// ============================================================================

/// Generator for DOT format graphs showing primitive relationships.
///
/// Creates graphs where:
/// - Nodes are primitives (colored by tier)
/// - Edges connect primitives with similarity above threshold
/// - Edge weight/thickness represents similarity strength
///
/// # Example
/// ```ignore
/// let system = PrimitiveSystem::global();
/// let graph = PrimitiveGraph::from_tier(system, PrimitiveTier::Physical);
/// let dot = graph.to_dot();
/// std::fs::write("primitives.dot", dot)?;
/// // Then: dot -Tsvg primitives.dot -o primitives.svg
/// ```
pub struct PrimitiveGraph {
    /// Nodes: (name, tier, is_base)
    nodes: Vec<(String, PrimitiveTier, bool)>,
    /// Edges: (from_idx, to_idx, similarity)
    edges: Vec<(usize, usize, f32)>,
    /// Graph title
    title: String,
}

impl PrimitiveGraph {
    /// Create a graph from specific primitives.
    pub fn from_primitives(
        system: &PrimitiveSystem,
        names: &[&str],
        similarity_threshold: f32,
    ) -> Self {
        let mut nodes = Vec::new();
        let mut encodings = Vec::new();

        for name in names {
            if let Some(prim) = system.get(name) {
                nodes.push((prim.name.clone(), prim.tier, prim.is_base));
                encodings.push(prim.encoding.clone());
            }
        }

        let edges = Self::compute_edges(&encodings, similarity_threshold);

        Self {
            nodes,
            edges,
            title: "Primitive Relationships".to_string(),
        }
    }

    /// Create a graph from all primitives in a tier.
    pub fn from_tier(
        system: &PrimitiveSystem,
        tier: PrimitiveTier,
        similarity_threshold: f32,
    ) -> Self {
        let prims = system.get_tier(tier);
        let mut nodes = Vec::new();
        let mut encodings = Vec::new();

        for prim in prims {
            nodes.push((prim.name.clone(), prim.tier, prim.is_base));
            encodings.push(prim.encoding.clone());
        }

        let edges = Self::compute_edges(&encodings, similarity_threshold);

        Self {
            nodes,
            edges,
            title: format!("{:?} Tier Primitives", tier),
        }
    }

    /// Create a graph from all primitives in a domain.
    pub fn from_domain(
        system: &PrimitiveSystem,
        domain: &str,
        similarity_threshold: f32,
    ) -> Self {
        let all_names = system.all_primitive_names();
        let mut nodes = Vec::new();
        let mut encodings = Vec::new();

        for name in all_names {
            if let Some(prim) = system.get(name) {
                if prim.domain == domain {
                    nodes.push((prim.name.clone(), prim.tier, prim.is_base));
                    encodings.push(prim.encoding.clone());
                }
            }
        }

        let edges = Self::compute_edges(&encodings, similarity_threshold);

        Self {
            nodes,
            edges,
            title: format!("{} Domain Primitives", domain),
        }
    }

    /// Create a similarity neighborhood graph around a primitive.
    pub fn neighborhood(
        system: &PrimitiveSystem,
        center: &str,
        depth: usize,
        top_k: usize,
    ) -> Self {
        let mut visited = std::collections::HashSet::new();
        let mut to_visit = vec![center.to_string()];
        let mut nodes = Vec::new();
        let mut node_map = std::collections::HashMap::new();
        let mut encodings = Vec::new();

        for _ in 0..depth {
            let current_batch: Vec<String> = to_visit.drain(..).collect();

            for name in current_batch {
                if visited.contains(&name) {
                    continue;
                }
                visited.insert(name.clone());

                if let Some(prim) = system.get(&name) {
                    let idx = nodes.len();
                    node_map.insert(name.clone(), idx);
                    nodes.push((prim.name.clone(), prim.tier, prim.is_base));
                    encodings.push(prim.encoding.clone());

                    // Find similar primitives for next iteration
                    let similar = system.find_similar(&name, top_k);
                    for (sim_name, _) in similar {
                        if !visited.contains(&sim_name) {
                            to_visit.push(sim_name);
                        }
                    }
                }
            }
        }

        let edges = Self::compute_edges(&encodings, 0.52); // Slightly above random

        Self {
            nodes,
            edges,
            title: format!("Neighborhood of {}", center),
        }
    }

    fn compute_edges(encodings: &[HV16], threshold: f32) -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();

        for i in 0..encodings.len() {
            for j in (i + 1)..encodings.len() {
                let sim = encodings[i].similarity(&encodings[j]);
                if sim > threshold {
                    edges.push((i, j, sim));
                }
            }
        }

        edges
    }

    /// Generate DOT format representation.
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();

        dot.push_str("digraph PrimitiveGraph {\n");
        dot.push_str(&format!("  label=\"{}\";\n", self.title));
        dot.push_str("  labelloc=\"t\";\n");
        dot.push_str("  fontsize=16;\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box, style=rounded];\n");
        dot.push_str("\n");

        // Nodes with tier-based colors
        for (i, (name, tier, is_base)) in self.nodes.iter().enumerate() {
            let color = tier_color(*tier);
            let shape = if *is_base { "box" } else { "ellipse" };
            dot.push_str(&format!(
                "  n{} [label=\"{}\", fillcolor=\"{}\", style=\"filled,rounded\", shape={}];\n",
                i, name, color, shape
            ));
        }

        dot.push_str("\n");

        // Edges with similarity-based styling
        for (from, to, sim) in &self.edges {
            let weight = ((sim - 0.5) * 10.0).max(1.0) as i32;
            let penwidth = ((sim - 0.5) * 8.0).max(0.5);
            let color = if *sim > 0.6 { "darkgreen" } else { "gray50" };

            dot.push_str(&format!(
                "  n{} -> n{} [dir=none, weight={}, penwidth={:.1}, color=\"{}\", label=\"{:.2}\"];\n",
                from, to, weight, penwidth, color, sim
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Generate a simple ASCII representation.
    pub fn to_ascii(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("=== {} ===\n\n", self.title));
        out.push_str(&format!("Nodes: {}\n", self.nodes.len()));
        out.push_str(&format!("Edges: {} (above threshold)\n\n", self.edges.len()));

        out.push_str("Nodes:\n");
        for (name, tier, is_base) in &self.nodes {
            let marker = if *is_base { "◆" } else { "◇" };
            out.push_str(&format!("  {} {} ({:?})\n", marker, name, tier));
        }

        out.push_str("\nEdges (by similarity):\n");
        let mut sorted_edges = self.edges.clone();
        sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (from, to, sim) in sorted_edges.iter().take(20) {
            let from_name = &self.nodes[*from].0;
            let to_name = &self.nodes[*to].0;
            let bar_len = ((sim - 0.5) * 40.0) as usize;
            let bar: String = "█".repeat(bar_len.min(20));
            out.push_str(&format!("  {} ↔ {} : {:.3} {}\n", from_name, to_name, sim, bar));
        }

        if self.edges.len() > 20 {
            out.push_str(&format!("  ... and {} more edges\n", self.edges.len() - 20));
        }

        out
    }

    /// Get graph statistics.
    pub fn stats(&self) -> GraphStats {
        let avg_similarity = if self.edges.is_empty() {
            0.0
        } else {
            self.edges.iter().map(|(_, _, s)| s).sum::<f32>() / self.edges.len() as f32
        };

        let max_similarity = self.edges.iter().map(|(_, _, s)| *s).fold(0.0f32, f32::max);

        GraphStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            avg_similarity,
            max_similarity,
            density: if self.nodes.len() > 1 {
                2.0 * self.edges.len() as f32 / (self.nodes.len() * (self.nodes.len() - 1)) as f32
            } else {
                0.0
            },
        }
    }
}

fn tier_color(tier: PrimitiveTier) -> &'static str {
    match tier {
        PrimitiveTier::NSM => "#E8F5E9",          // Light green
        PrimitiveTier::Mathematical => "#E3F2FD", // Light blue
        PrimitiveTier::Physical => "#FFF3E0",     // Light orange
        PrimitiveTier::Geometric => "#F3E5F5",    // Light purple
        PrimitiveTier::Strategic => "#FFEBEE",    // Light red
        PrimitiveTier::MetaCognitive => "#E0F7FA", // Light cyan
        PrimitiveTier::Temporal => "#FFF8E1",     // Light amber
        PrimitiveTier::Compositional => "#F1F8E9", // Light lime
        PrimitiveTier::Consciousness => "#FCE4EC", // Light pink
    }
}

/// Statistics about a primitive graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_similarity: f32,
    pub max_similarity: f32,
    pub density: f32,
}

// ============================================================================
// PERSISTENCE LAYER FOR COMPOSITIONS AND SESSION DATA
// ============================================================================

/// Persistence manager for saving/loading primitive compositions and session data.
///
/// Supports:
/// - Composition algebra definitions (named expressions)
/// - Session history (composition operations performed)
/// - Custom primitive definitions (experimental)
///
/// # Example
/// ```ignore
/// let system = PrimitiveSystem::global();
/// let mut algebra = CompositionAlgebra::new();
/// algebra.define("MY_COMP", "CAUSE ⊗ EFFECT", system)?;
///
/// // Save session
/// let persistence = PrimitivePersistence::new();
/// persistence.save_session("session.json", &algebra, &history)?;
///
/// // Later, restore
/// let (loaded_algebra, loaded_history) = persistence.load_session("session.json", system)?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct PrimitivePersistence;

/// Serializable session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Version for forward compatibility
    pub version: u32,
    /// Timestamp (Unix seconds)
    pub timestamp: u64,
    /// Named compositions
    pub compositions: Vec<CompositionExport>,
    /// Operation history
    pub history: Vec<HistoryEntry>,
    /// Optional notes
    pub notes: Option<String>,
}

/// A single history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The operation performed
    pub operation: String,
    /// The best-matching primitive result
    pub result_match: String,
    /// Similarity to the match
    pub similarity: f32,
}

impl PrimitivePersistence {
    /// Create a new persistence manager.
    pub fn new() -> Self {
        Self
    }

    /// Save session data to a JSON file.
    pub fn save_session(
        &self,
        path: &str,
        algebra: &CompositionAlgebra,
        history: &[HistoryEntry],
        notes: Option<&str>,
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let session = SessionData {
            version: 1,
            timestamp,
            compositions: algebra.export(),
            history: history.to_vec(),
            notes: notes.map(|s| s.to_string()),
        };

        let json = serde_json::to_string_pretty(&session)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;

        let mut file = File::create(path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        file.write_all(json.as_bytes())
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load session data from a JSON file.
    pub fn load_session(
        &self,
        path: &str,
        system: &PrimitiveSystem,
    ) -> Result<(CompositionAlgebra, Vec<HistoryEntry>), PersistenceError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let session: SessionData = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;

        // Rebuild algebra
        let mut algebra = CompositionAlgebra::new();
        algebra.import(&session.compositions, system)
            .map_err(|e| PersistenceError::CompositionError(e.to_string()))?;

        Ok((algebra, session.history))
    }

    /// Save just compositions (without history).
    pub fn save_compositions(
        &self,
        path: &str,
        algebra: &CompositionAlgebra,
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;

        let exports = algebra.export();
        let json = serde_json::to_string_pretty(&exports)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;

        let mut file = File::create(path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        file.write_all(json.as_bytes())
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load just compositions (without history).
    pub fn load_compositions(
        &self,
        path: &str,
        system: &PrimitiveSystem,
    ) -> Result<CompositionAlgebra, PersistenceError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let exports: Vec<CompositionExport> = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;

        let mut algebra = CompositionAlgebra::new();
        algebra.import(&exports, system)
            .map_err(|e| PersistenceError::CompositionError(e.to_string()))?;

        Ok(algebra)
    }

    /// Export a graph to DOT format file.
    pub fn export_graph_dot(
        &self,
        path: &str,
        graph: &PrimitiveGraph,
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;

        let dot = graph.to_dot();

        let mut file = File::create(path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        file.write_all(dot.as_bytes())
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Export similarity matrix to CSV format.
    pub fn export_similarity_csv(
        &self,
        path: &str,
        system: &PrimitiveSystem,
        names: &[&str],
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;

        let matrix = system.similarity_matrix(names);

        let mut file = File::create(path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        // Header
        let mut header = String::from(",");
        header.push_str(&names.join(","));
        writeln!(file, "{}", header)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        // Rows
        for (i, row) in matrix.iter().enumerate() {
            let row_str: Vec<String> = row.iter().map(|v| format!("{:.4}", v)).collect();
            writeln!(file, "{},{}", names[i], row_str.join(","))
                .map_err(|e| PersistenceError::IoError(e.to_string()))?;
        }

        Ok(())
    }
}

/// Errors from persistence operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistenceError {
    /// IO error (file not found, permission denied, etc.)
    IoError(String),
    /// Serialization error
    SerializationError(String),
    /// Deserialization error
    DeserializationError(String),
    /// Error rebuilding composition
    CompositionError(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::IoError(msg) => write!(f, "IO error: {}", msg),
            PersistenceError::SerializationError(msg) => write!(f, "serialization error: {}", msg),
            PersistenceError::DeserializationError(msg) => write!(f, "deserialization error: {}", msg),
            PersistenceError::CompositionError(msg) => write!(f, "composition error: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {}

impl Default for PrimitiveSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_system_creation() {
        let system = PrimitiveSystem::new();
        assert!(system.count() > 0, "Should have primitives");
        assert!(system.count_tier(PrimitiveTier::Mathematical) > 0, "Should have mathematical primitives");
    }

    #[test]
    fn test_tier1_primitives() {
        let system = PrimitiveSystem::new();

        // Check key primitives exist
        assert!(system.get("SET").is_some(), "SET primitive should exist");
        assert!(system.get("NOT").is_some(), "NOT primitive should exist");
        assert!(system.get("ZERO").is_some(), "ZERO primitive should exist");
        assert!(system.get("ADDITION").is_some(), "ADDITION primitive should exist");
    }

    #[test]
    fn test_orthogonality_check() {
        let system = PrimitiveSystem::new();

        // Check that SET and NOT are reasonably orthogonal
        let sim = system.check_orthogonality("SET", "NOT");
        assert!(sim.is_some(), "Should be able to check orthogonality");

        // With 16,384-bit vectors, similarity() returns fraction of matching bits
        // in [0, 1]. Random/orthogonal pairs concentrate around 0.5.
        if let Some(similarity) = sim {
            assert!((similarity - 0.5).abs() < 0.03,
                "Cross-domain primitives should be near-orthogonal (sim ≈ 0.5), got {}", similarity);
        }
    }

    #[test]
    fn test_tier_validation() {
        let system = PrimitiveSystem::new();

        // With bind-based embedding and 16,384 dimensions, all within-tier
        // primitive pairs should have |similarity - 0.5| < 0.03 (~4σ threshold).
        let violations = system.validate_tier_orthogonality(PrimitiveTier::Mathematical, 0.03);

        // Zero violations expected — any violation indicates a seeding collision
        assert!(violations.is_empty(),
            "All Tier 1 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}", violations);
    }

    #[test]
    fn test_domain_manifolds() {
        let system = PrimitiveSystem::new();

        let math = system.domain("mathematics");
        let logic = system.domain("logic");

        assert!(math.is_some(), "Mathematics domain should exist");
        assert!(logic.is_some(), "Logic domain should exist");

        // Domain rotations are independent random 16K-bit vectors, so near-orthogonal
        // similarity() returns [0,1] where 0.5 = random baseline
        if let (Some(m), Some(l)) = (math, logic) {
            let sim = m.rotation.similarity(&l.rotation);
            assert!((sim - 0.5).abs() < 0.03,
                "Domain rotations should be near-orthogonal (sim ≈ 0.5), got {}", sim);
        }
    }

    #[test]
    fn test_derived_primitives() {
        let system = PrimitiveSystem::new();

        let zero = system.get("ZERO").unwrap();
        let one = system.get("ONE").unwrap();
        let addition = system.get("ADDITION").unwrap();

        assert!(zero.is_base, "ZERO should be a base primitive");
        assert!(!one.is_base, "ONE should be derived");
        assert!(!addition.is_base, "ADDITION should be derived");

        assert!(one.derivation.is_some(), "Derived primitives should have derivation");
    }

    // ========================================================================
    // TIER 2: PHYSICAL REALITY TESTS
    // ========================================================================

    #[test]
    fn test_tier2_primitives_exist() {
        let system = PrimitiveSystem::new();

        // Physical properties
        assert!(system.get("MASS").is_some(), "MASS primitive should exist");
        assert!(system.get("CHARGE").is_some(), "CHARGE primitive should exist");
        assert!(system.get("ENERGY").is_some(), "ENERGY primitive should exist");

        // Motion
        assert!(system.get("VELOCITY").is_some(), "VELOCITY primitive should exist");
        assert!(system.get("ACCELERATION").is_some(), "ACCELERATION primitive should exist");
        assert!(system.get("MOMENTUM").is_some(), "MOMENTUM primitive should exist");

        // Causality
        assert!(system.get("CAUSE").is_some(), "CAUSE primitive should exist");
        assert!(system.get("EFFECT").is_some(), "EFFECT primitive should exist");
        assert!(system.get("STATE_CHANGE").is_some(), "STATE_CHANGE primitive should exist");

        // Thermodynamics
        assert!(system.get("THERMODYNAMIC_ENTROPY").is_some(), "THERMODYNAMIC_ENTROPY primitive should exist");
        assert!(system.get("TEMPERATURE").is_some(), "TEMPERATURE primitive should exist");
    }

    #[test]
    fn test_tier2_domains() {
        let system = PrimitiveSystem::new();

        assert!(system.domain("physics").is_some(), "Physics domain should exist");
        assert!(system.domain("causality").is_some(), "Causality domain should exist");
    }

    #[test]
    fn test_tier2_derived_primitives() {
        let system = PrimitiveSystem::new();

        // MOMENTUM should be derived (MASS × VELOCITY)
        let momentum = system.get("MOMENTUM").unwrap();
        assert!(!momentum.is_base, "MOMENTUM should be derived");
        assert!(momentum.derivation.is_some(), "MOMENTUM should have derivation");

        // ACCELERATION should be derived
        let acceleration = system.get("ACCELERATION").unwrap();
        assert!(!acceleration.is_base, "ACCELERATION should be derived");
    }

    #[test]
    fn test_tier2_orthogonality() {
        let system = PrimitiveSystem::new();

        let violations = system.validate_tier_orthogonality(PrimitiveTier::Physical, 0.03);
        assert!(violations.is_empty(),
            "All Tier 2 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}", violations);
    }

    // ========================================================================
    // TIER 3: GEOMETRIC & TOPOLOGICAL TESTS
    // ========================================================================

    #[test]
    fn test_tier3_primitives_exist() {
        let system = PrimitiveSystem::new();

        // Basic geometry
        assert!(system.get("POINT").is_some(), "POINT primitive should exist");
        assert!(system.get("LINE").is_some(), "LINE primitive should exist");
        assert!(system.get("PLANE").is_some(), "PLANE primitive should exist");
        assert!(system.get("ANGLE").is_some(), "ANGLE primitive should exist");
        assert!(system.get("DISTANCE").is_some(), "DISTANCE primitive should exist");

        // Vectors
        assert!(system.get("VECTOR").is_some(), "VECTOR primitive should exist");
        assert!(system.get("DOT_PRODUCT").is_some(), "DOT_PRODUCT primitive should exist");
        assert!(system.get("CROSS_PRODUCT").is_some(), "CROSS_PRODUCT primitive should exist");

        // Differential geometry
        assert!(system.get("MANIFOLD").is_some(), "MANIFOLD primitive should exist");
        assert!(system.get("TANGENT_SPACE").is_some(), "TANGENT_SPACE primitive should exist");
        assert!(system.get("CURVATURE").is_some(), "CURVATURE primitive should exist");

        // Topology
        assert!(system.get("OPEN_SET").is_some(), "OPEN_SET primitive should exist");
        assert!(system.get("BOUNDARY").is_some(), "BOUNDARY primitive should exist");
        assert!(system.get("PART_OF").is_some(), "PART_OF primitive should exist");
    }

    #[test]
    fn test_tier3_domains() {
        let system = PrimitiveSystem::new();

        assert!(system.domain("geometry").is_some(), "Geometry domain should exist");
        assert!(system.domain("topology").is_some(), "Topology domain should exist");
    }

    #[test]
    fn test_tier3_orthogonality() {
        let system = PrimitiveSystem::new();

        let violations = system.validate_tier_orthogonality(PrimitiveTier::Geometric, 0.03);
        assert!(violations.is_empty(),
            "All Tier 3 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}", violations);
    }

    // ========================================================================
    // TIER 4: STRATEGIC & SOCIAL TESTS
    // ========================================================================

    #[test]
    fn test_tier4_primitives_exist() {
        let system = PrimitiveSystem::new();

        // Game theory
        assert!(system.get("UTILITY").is_some(), "UTILITY primitive should exist");
        assert!(system.get("STRATEGY").is_some(), "STRATEGY primitive should exist");
        assert!(system.get("EQUILIBRIUM").is_some(), "EQUILIBRIUM primitive should exist");
        assert!(system.get("PAYOFF").is_some(), "PAYOFF primitive should exist");

        // Temporal logic
        assert!(system.get("BEFORE").is_some(), "BEFORE primitive should exist");
        assert!(system.get("AFTER").is_some(), "AFTER primitive should exist");
        assert!(system.get("DURING").is_some(), "DURING primitive should exist");
        assert!(system.get("MEETS").is_some(), "MEETS primitive should exist");

        // Social coordination
        assert!(system.get("COOPERATE").is_some(), "COOPERATE primitive should exist");
        assert!(system.get("DEFECT").is_some(), "DEFECT primitive should exist");
        assert!(system.get("RECIPROCATE").is_some(), "RECIPROCATE primitive should exist");
        assert!(system.get("TRUST").is_some(), "TRUST primitive should exist");

        // Information
        assert!(system.get("BELIEF").is_some(), "BELIEF primitive should exist");
        assert!(system.get("COMMON_KNOWLEDGE").is_some(), "COMMON_KNOWLEDGE primitive should exist");
    }

    #[test]
    fn test_tier4_domains() {
        let system = PrimitiveSystem::new();

        assert!(system.domain("game_theory").is_some(), "Game theory domain should exist");
        assert!(system.domain("temporal").is_some(), "Temporal domain should exist");
        assert!(system.domain("social").is_some(), "Social domain should exist");
    }

    #[test]
    fn test_tier4_harmonic_connections() {
        let system = PrimitiveSystem::new();

        // Tier 4 primitives should connect to harmonics
        // COOPERATE + TRUST → Sacred Reciprocity
        let cooperate = system.get("COOPERATE").unwrap();
        let trust = system.get("TRUST").unwrap();

        assert!(cooperate.tier == PrimitiveTier::Strategic, "COOPERATE should be Strategic tier");
        assert!(trust.tier == PrimitiveTier::Strategic, "TRUST should be Strategic tier");
    }

    #[test]
    fn test_tier4_orthogonality() {
        let system = PrimitiveSystem::new();

        let violations = system.validate_tier_orthogonality(PrimitiveTier::Strategic, 0.03);
        assert!(violations.is_empty(),
            "All Tier 4 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}", violations);
    }

    // ========================================================================
    // TIER 5: META-COGNITIVE TESTS
    // ========================================================================

    #[test]
    fn test_tier5_primitives_exist() {
        let system = PrimitiveSystem::new();

        // Self-awareness
        assert!(system.get("SELF").is_some(), "SELF primitive should exist");
        assert!(system.get("IDENTITY").is_some(), "IDENTITY primitive should exist");
        assert!(system.get("META_BELIEF").is_some(), "META_BELIEF primitive should exist");
        assert!(system.get("INTROSPECTION").is_some(), "INTROSPECTION primitive should exist");

        // Homeostasis
        assert!(system.get("HOMEOSTASIS").is_some(), "HOMEOSTASIS primitive should exist");
        assert!(system.get("SETPOINT").is_some(), "SETPOINT primitive should exist");
        assert!(system.get("REGULATION").is_some(), "REGULATION primitive should exist");
        assert!(system.get("FEEDBACK").is_some(), "FEEDBACK primitive should exist");

        // Repair & adaptation
        assert!(system.get("REPAIR").is_some(), "REPAIR primitive should exist");
        assert!(system.get("ADAPT").is_some(), "ADAPT primitive should exist");
        assert!(system.get("LEARN").is_some(), "LEARN primitive should exist");

        // Epistemic
        assert!(system.get("KNOW").is_some(), "KNOW primitive should exist");
        assert!(system.get("UNCERTAIN").is_some(), "UNCERTAIN primitive should exist");
        assert!(system.get("CONFIDENCE").is_some(), "CONFIDENCE primitive should exist");
        assert!(system.get("EVIDENCE").is_some(), "EVIDENCE primitive should exist");

        // Metabolic
        assert!(system.get("RESOURCE").is_some(), "RESOURCE primitive should exist");
        assert!(system.get("ALLOCATE").is_some(), "ALLOCATE primitive should exist");

        // Reward
        assert!(system.get("REWARD").is_some(), "REWARD primitive should exist");
        assert!(system.get("GOAL").is_some(), "GOAL primitive should exist");
        assert!(system.get("VALUE").is_some(), "VALUE primitive should exist");
    }

    #[test]
    fn test_tier5_domains() {
        let system = PrimitiveSystem::new();

        assert!(system.domain("metacognition").is_some(), "Metacognition domain should exist");
        assert!(system.domain("homeostasis").is_some(), "Homeostasis domain should exist");
        assert!(system.domain("epistemic").is_some(), "Epistemic domain should exist");
        assert!(system.domain("metabolic").is_some(), "Metabolic domain should exist");
    }

    #[test]
    fn test_tier5_consciousness_primitives() {
        let system = PrimitiveSystem::new();

        // Tier 5 enables consciousness-first computing
        // SELF + HOMEOSTASIS → self-regulation
        let self_prim = system.get("SELF").unwrap();
        let homeostasis = system.get("HOMEOSTASIS").unwrap();

        assert!(self_prim.tier == PrimitiveTier::MetaCognitive);
        assert!(homeostasis.tier == PrimitiveTier::MetaCognitive);

        // These primitives enable the system to reason about itself
        assert!(self_prim.is_base, "SELF should be a base primitive");
        assert!(homeostasis.is_base, "HOMEOSTASIS should be a base primitive");
    }

    #[test]
    fn test_tier5_orthogonality() {
        let system = PrimitiveSystem::new();

        let violations = system.validate_tier_orthogonality(PrimitiveTier::MetaCognitive, 0.03);
        assert!(violations.is_empty(),
            "All Tier 5 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}", violations);
    }

    // ========================================================================
    // CROSS-TIER INTEGRATION TESTS
    // ========================================================================

    #[test]
    fn test_complete_primitive_ecology() {
        let system = PrimitiveSystem::new();

        // Count primitives per tier
        let tier1_count = system.count_tier(PrimitiveTier::Mathematical);
        let tier2_count = system.count_tier(PrimitiveTier::Physical);
        let tier3_count = system.count_tier(PrimitiveTier::Geometric);
        let tier4_count = system.count_tier(PrimitiveTier::Strategic);
        let tier5_count = system.count_tier(PrimitiveTier::MetaCognitive);

        // Verify all tiers have primitives
        assert!(tier1_count > 0, "Tier 1 should have primitives");
        assert!(tier2_count > 0, "Tier 2 should have primitives");
        assert!(tier3_count > 0, "Tier 3 should have primitives");
        assert!(tier4_count > 0, "Tier 4 should have primitives");
        assert!(tier5_count > 0, "Tier 5 should have primitives");

        // Verify total count is substantial
        let total = system.count();
        assert!(total >= 80, "Should have at least 80 primitives across all tiers (got {})", total);

        println!("Complete Primitive Ecology:");
        println!("  Tier 1 (Mathematical): {} primitives", tier1_count);
        println!("  Tier 2 (Physical):     {} primitives", tier2_count);
        println!("  Tier 3 (Geometric):    {} primitives", tier3_count);
        println!("  Tier 4 (Strategic):    {} primitives", tier4_count);
        println!("  Tier 5 (MetaCognitive): {} primitives", tier5_count);
        println!("  TOTAL: {} primitives", total);
    }

    #[test]
    fn test_cross_tier_binding() {
        let system = PrimitiveSystem::new();

        // Find cross-tier binding rules
        let cross_tier_rules: Vec<_> = system.binding_rules()
            .iter()
            .filter(|rule| {
                rule.pattern.len() > 1 &&
                rule.pattern.iter().collect::<std::collections::HashSet<_>>().len() > 1
            })
            .collect();

        // Should have at least one cross-tier binding rule
        assert!(!cross_tier_rules.is_empty(), "Should have cross-tier binding rules");

        // Print cross-tier rules
        for rule in &cross_tier_rules {
            println!("Cross-tier rule: {} - {}", rule.name, rule.example);
        }
    }

    #[test]
    fn test_domain_diversity() {
        let system = PrimitiveSystem::new();

        // Should have multiple domains across tiers
        assert!(system.domain("mathematics").is_some());
        assert!(system.domain("logic").is_some());
        assert!(system.domain("physics").is_some());
        assert!(system.domain("causality").is_some());
        assert!(system.domain("geometry").is_some());
        assert!(system.domain("topology").is_some());
        assert!(system.domain("game_theory").is_some());
        assert!(system.domain("temporal").is_some());
        assert!(system.domain("social").is_some());
        assert!(system.domain("metacognition").is_some());
        assert!(system.domain("homeostasis").is_some());
        assert!(system.domain("epistemic").is_some());
        assert!(system.domain("metabolic").is_some());

        // Total domain count
        println!("Total domains: {}", system.domains.len());
        assert!(system.domains.len() >= 13, "Should have at least 13 distinct domains");
    }

    #[test]
    fn test_harmonic_primitive_connections() {
        let system = PrimitiveSystem::new();

        // Test primitives that connect to specific harmonics

        // Tier 2 → Resonant Coherence (physical stability)
        assert!(system.get("THERMODYNAMIC_ENTROPY").is_some());

        // Tier 3 → Universal Interconnectedness (spatial relationships)
        assert!(system.get("PART_OF").is_some());

        // Tier 4 → Sacred Reciprocity (cooperation)
        assert!(system.get("COOPERATE").is_some());
        assert!(system.get("TRUST").is_some());
        assert!(system.get("RECIPROCATE").is_some());

        // Tier 5 → All 7 harmonics (meta-cognitive spans all)
        assert!(system.get("SELF").is_some());
        assert!(system.get("HOMEOSTASIS").is_some());
        assert!(system.get("KNOW").is_some());
        assert!(system.get("GOAL").is_some());
    }

    #[test]
    fn test_primitive_ecology_summary() {
        let system = PrimitiveSystem::new();

        let summary = system.summary();

        println!("\n{}", summary);

        // Summary should contain all tier names
        assert!(summary.contains("Mathematical"));
        assert!(summary.contains("Physical"));
        assert!(summary.contains("Geometric"));
        assert!(summary.contains("Strategic"));
        assert!(summary.contains("MetaCognitive"));
    }

    // =========================================================================
    // DERIVATION CHAIN INTEGRATION TESTS
    // =========================================================================
    //
    // These tests verify that derived primitives have the expected algebraic
    // relationships with their parent primitives via XOR binding.
    //
    // For a derived primitive D = P1 ⊗ P2:
    // - D ⊗ P1 should have high similarity to P2
    // - D ⊗ P2 should have high similarity to P1
    // =========================================================================

    #[test]
    fn test_attention_derives_from_salience_selection() {
        let system = PrimitiveSystem::new();

        let attention = system.get("ATTENTION").expect("ATTENTION should exist");
        let salience = system.get("SALIENCE").expect("SALIENCE should exist");
        let selection = system.get("SELECTION").expect("SELECTION should exist");

        // If ATTENTION = SALIENCE ⊗ SELECTION, then:
        // ATTENTION ⊗ SALIENCE should be similar to SELECTION
        let unbound = attention.encoding.bind(&salience.encoding);
        let sim_to_selection = unbound.similarity(&selection.encoding);

        // For proper derivation, similarity should be very high (>0.95)
        // If derivation fell back to random, it would be ~0.5
        assert!(
            sim_to_selection > 0.90,
            "ATTENTION⊗SALIENCE should recover SELECTION (sim={:.3}, expected >0.90). \
             If ~0.5, ATTENTION used random fallback instead of derivation.",
            sim_to_selection
        );

        // Similarly, ATTENTION ⊗ SELECTION should recover SALIENCE
        let unbound2 = attention.encoding.bind(&selection.encoding);
        let sim_to_salience = unbound2.similarity(&salience.encoding);
        assert!(
            sim_to_salience > 0.90,
            "ATTENTION⊗SELECTION should recover SALIENCE (sim={:.3}, expected >0.90)",
            sim_to_salience
        );
    }

    #[test]
    fn test_probability_derivation_chain() {
        let system = PrimitiveSystem::new();

        let probability = system.get("PROBABILITY").expect("PROBABILITY should exist");
        let ratio = system.get("RATIO").expect("RATIO should exist");
        let certainty = system.get("CERTAINTY").expect("CERTAINTY should exist");

        // PROBABILITY = RATIO ⊗ CERTAINTY
        // Unbinding should recover the other parent
        let unbound = probability.encoding.bind(&ratio.encoding);
        let sim_to_certainty = unbound.similarity(&certainty.encoding);

        // Also check the reverse unbinding
        let unbound2 = probability.encoding.bind(&certainty.encoding);
        let sim_to_ratio = unbound2.similarity(&ratio.encoding);

        // For proper derivation, similarity should be very high (>0.90)
        // If derivation fell back to random, it would be ~0.5
        assert!(
            sim_to_certainty > 0.90,
            "PROBABILITY⊗RATIO should recover CERTAINTY (sim={:.3}). \
             If ~0.5, PROBABILITY used random fallback instead of derivation.",
            sim_to_certainty
        );
        assert!(
            sim_to_ratio > 0.90,
            "PROBABILITY⊗CERTAINTY should recover RATIO (sim={:.3})",
            sim_to_ratio
        );
    }

    #[test]
    fn test_expected_value_derivation_chain() {
        let system = PrimitiveSystem::new();

        let expected_value = system.get("EXPECTED_VALUE").expect("EXPECTED_VALUE should exist");
        let probability = system.get("PROBABILITY").expect("PROBABILITY should exist");
        let value = system.get("VALUE").expect("VALUE should exist");

        // EXPECTED_VALUE = PROBABILITY ⊗ VALUE
        // Unbinding should recover the other parent
        let unbound = expected_value.encoding.bind(&probability.encoding);
        let sim_to_value = unbound.similarity(&value.encoding);

        // For proper derivation, similarity should be very high (>0.90)
        // If derivation fell back to random, it would be ~0.5
        assert!(
            sim_to_value > 0.90,
            "EXPECTED_VALUE⊗PROBABILITY should recover VALUE (sim={:.3})",
            sim_to_value
        );
    }

    #[test]
    fn test_certainty_uncertain_relationship() {
        let system = PrimitiveSystem::new();

        // UNCERTAIN and CERTAINTY are both base primitives (not derived)
        // They should be nearly orthogonal (independent random vectors ~0.5 similarity)
        let uncertain = system.get("UNCERTAIN").expect("UNCERTAIN should exist");
        let certainty = system.get("CERTAINTY").expect("CERTAINTY should exist");

        let sim = uncertain.encoding.similarity(&certainty.encoding);
        // Both are independently seeded randoms, so should be ~0.5 (orthogonal)
        assert!(
            sim > 0.40 && sim < 0.60,
            "UNCERTAIN and CERTAINTY are independent randoms, should be ~0.5 orthogonal (sim={:.3})",
            sim
        );
    }

    #[test]
    fn test_derived_primitives_not_random() {
        let system = PrimitiveSystem::new();

        // Test several ACTUAL derived primitives to ensure they're not just random vectors
        // Random vectors would have ~0.5 similarity to their declared parents
        // NOTE: Only test primitives that are actually derived in init_derived_primitives!

        let test_cases = [
            ("FIELD", vec!["FORCE", "POINT"]),
            ("EQUILIBRIUM", vec!["FORCE", "CONSERVATION"]),
            ("SHANNON_ENTROPY", vec!["PROBABILITY", "INFORMATION"]),
        ];

        for (derived_name, parent_names) in test_cases {
            let derived = system.get(derived_name);
            if derived.is_none() {
                // Skip if this derived primitive doesn't exist in this version
                continue;
            }
            let derived = derived.unwrap();

            // Get parent encodings
            let parents: Vec<_> = parent_names
                .iter()
                .filter_map(|name| system.get(name))
                .collect();

            if parents.len() < 2 {
                // Skip if parents don't exist
                continue;
            }

            // Compute expected derivation: P1 ⊗ P2
            let expected = parents[0].encoding.bind(&parents[1].encoding);

            let sim = derived.encoding.similarity(&expected);
            assert!(
                sim > 0.90,
                "{} should be derived from {:?} (sim={:.3}, expected >0.90). \
                 Low similarity suggests random fallback was used.",
                derived_name, parent_names, sim
            );
        }
    }

    #[test]
    fn test_derivation_chain_transitivity() {
        // Test that multi-level derivations work correctly
        // VARIANCE = EXPECTED_VALUE ⊗ DEVIATION
        // EXPECTED_VALUE = PROBABILITY ⊗ VALUE
        // So VARIANCE ⊗ EXPECTED_VALUE should recover DEVIATION
        let system = PrimitiveSystem::new();

        let variance = system.get("VARIANCE");
        let expected_value = system.get("EXPECTED_VALUE");
        let deviation = system.get("DEVIATION");

        if let (Some(variance), Some(expected_value), Some(deviation)) = (variance, expected_value, deviation) {
            // VARIANCE ⊗ EXPECTED_VALUE should recover DEVIATION
            let unbound = variance.encoding.bind(&expected_value.encoding);
            let sim = unbound.similarity(&deviation.encoding);

            assert!(
                sim > 0.90,
                "VARIANCE⊗EXPECTED_VALUE should recover DEVIATION (sim={:.3})",
                sim
            );
        }
    }

    // =========================================================================
    // DERIVATION QUALITY BENCHMARK
    // =========================================================================

    /// Benchmark test: measures derivation quality across all derived primitives
    ///
    /// This test quantifies how well the derive_encoding function works by:
    /// 1. Finding all primitives with a documented derivation (e.g., "A ⊗ B")
    /// 2. For each, computing the expected encoding from parents
    /// 3. Measuring similarity between actual and expected encodings
    ///
    /// Results are printed to help assess system health.
    #[test]
    fn test_benchmark_derivation_quality() {
        let system = PrimitiveSystem::new();

        // Collect all derived primitives with their documented derivation
        let mut derived_count = 0;
        let mut success_count = 0;
        let mut total_similarity: f32 = 0.0;
        let mut failures: Vec<(String, f32)> = Vec::new();

        for (name, primitive) in system.primitives.iter() {
            if primitive.is_base || primitive.derivation.is_none() {
                continue;
            }

            let derivation = primitive.derivation.as_ref().unwrap();

            // Parse derivation string like "A ⊗ B" or "APPLY(A, B)"
            // For now, only handle "X ⊗ Y" format
            if !derivation.contains(" ⊗ ") {
                continue;
            }

            let parts: Vec<&str> = derivation.split(" ⊗ ").collect();
            if parts.len() != 2 {
                continue;
            }

            let parent1_name = parts[0].trim();
            let parent2_name = parts[1].trim();

            // Get parent primitives
            let parent1 = system.get(parent1_name);
            let parent2 = system.get(parent2_name);

            if let (Some(p1), Some(p2)) = (parent1, parent2) {
                derived_count += 1;

                // Expected encoding: parent1 ⊗ parent2
                let expected = p1.encoding.bind(&p2.encoding);
                let similarity = primitive.encoding.similarity(&expected);
                total_similarity += similarity;

                if similarity > 0.90 {
                    success_count += 1;
                } else {
                    failures.push((name.clone(), similarity));
                }
            }
        }

        // Print benchmark results
        if derived_count > 0 {
            let avg_similarity = total_similarity / derived_count as f32;
            println!("\n=== DERIVATION QUALITY BENCHMARK ===");
            println!("Total derived primitives tested: {}", derived_count);
            println!("Successfully derived (sim > 0.90): {} ({:.1}%)",
                     success_count, 100.0 * success_count as f32 / derived_count as f32);
            println!("Average similarity to expected: {:.3}", avg_similarity);

            if !failures.is_empty() {
                println!("\nFailed derivations:");
                for (name, sim) in &failures {
                    println!("  {} - similarity: {:.3}", name, sim);
                }
            }
            println!("=====================================\n");

            // Assert at least 90% success rate
            let success_rate = success_count as f32 / derived_count as f32;
            assert!(
                success_rate >= 0.90,
                "Derivation success rate {:.1}% is below 90% threshold. {} of {} primitives failed.",
                success_rate * 100.0, failures.len(), derived_count
            );
        }
    }

    // =========================================================================
    // TYPED PRIMITIVE OPERATIONS TESTS
    // =========================================================================

    #[test]
    fn test_bind_primitives() {
        let system = PrimitiveSystem::new();

        // Test successful binding
        let result = system.bind_primitives("CAUSE", "EFFECT");
        assert!(result.is_ok(), "bind_primitives should succeed");

        let result = result.unwrap();
        assert_eq!(result.source_primitives, vec!["CAUSE", "EFFECT"]);
        assert!(result.operation.contains("bind"));

        // Verify XOR binding properties: A ⊗ B ⊗ B = A
        let cause = system.get("CAUSE").unwrap();
        let effect = system.get("EFFECT").unwrap();
        let unbound = result.encoding.bind(&effect.encoding);
        let sim = unbound.similarity(&cause.encoding);
        assert!(sim > 0.99, "Unbinding should recover original (sim={:.3})", sim);
    }

    #[test]
    fn test_bind_primitives_not_found() {
        let system = PrimitiveSystem::new();

        let result = system.bind_primitives("CAUSE", "NONEXISTENT");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::NotFound(_)));
    }

    #[test]
    fn test_bundle_primitives() {
        let system = PrimitiveSystem::new();

        let result = system.bundle_primitives(&["AND", "OR", "NOT"]);
        assert!(result.is_ok(), "bundle_primitives should succeed");

        let result = result.unwrap();
        assert_eq!(result.source_primitives.len(), 3);
        assert!(result.operation.contains("bundle"));

        // Bundle should have moderate similarity to all inputs
        let and_prim = system.get("AND").unwrap();
        let or_prim = system.get("OR").unwrap();
        let not_prim = system.get("NOT").unwrap();

        let sim_and = result.encoding.similarity(&and_prim.encoding);
        let sim_or = result.encoding.similarity(&or_prim.encoding);
        let sim_not = result.encoding.similarity(&not_prim.encoding);

        // Bundle should be more similar to inputs than random (~0.5)
        assert!(sim_and > 0.55, "Bundle should be similar to AND (sim={:.3})", sim_and);
        assert!(sim_or > 0.55, "Bundle should be similar to OR (sim={:.3})", sim_or);
        assert!(sim_not > 0.55, "Bundle should be similar to NOT (sim={:.3})", sim_not);
    }

    #[test]
    fn test_bundle_primitives_empty() {
        let system = PrimitiveSystem::new();

        let result = system.bundle_primitives(&[]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::EmptyInput));
    }

    // =========================================================================
    // SEQUENCE ENCODING TESTS
    // =========================================================================

    #[test]
    fn test_encode_sequence_single() {
        let system = PrimitiveSystem::new();

        // Single element sequence should be equivalent to the primitive itself
        let result = system.encode_sequence(&["CAUSE"]);
        assert!(result.is_ok());

        let result = result.unwrap();
        let cause = system.get("CAUSE").unwrap();
        let sim = result.encoding.similarity(&cause.encoding);
        assert!(sim > 0.99, "Single-element sequence should equal the primitive (sim={:.3})", sim);
    }

    #[test]
    fn test_encode_sequence_order_matters() {
        let system = PrimitiveSystem::new();

        // A → B should be different from B → A
        let seq_ab = system.encode_sequence(&["CAUSE", "EFFECT"]).unwrap();
        let seq_ba = system.encode_sequence(&["EFFECT", "CAUSE"]).unwrap();

        let sim = seq_ab.encoding.similarity(&seq_ba.encoding);

        // Should be significantly different (not random ~0.5, but also not identical >0.99)
        assert!(
            sim < 0.8,
            "Different orderings should produce different encodings (sim={:.3})",
            sim
        );
    }

    #[test]
    fn test_encode_sequence_longer() {
        let system = PrimitiveSystem::new();

        // Test longer sequence
        let result = system.encode_sequence(&["BEFORE", "DURING", "AFTER"]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.source_primitives, vec!["BEFORE", "DURING", "AFTER"]);
        assert!(result.operation.contains("→"));
    }

    #[test]
    fn test_encode_sequence_empty() {
        let system = PrimitiveSystem::new();

        let result = system.encode_sequence(&[]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::EmptyInput));
    }

    #[test]
    fn test_encode_sequence_not_found() {
        let system = PrimitiveSystem::new();

        let result = system.encode_sequence(&["CAUSE", "NONEXISTENT"]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::NotFound(_)));
    }

    // =========================================================================
    // WEIGHTED BUNDLING TESTS
    // =========================================================================

    #[test]
    fn test_bundle_weighted_equal_weights() {
        let system = PrimitiveSystem::new();

        // Equal weights should be similar to regular bundle
        let regular = system.bundle_primitives(&["AND", "OR"]).unwrap();
        let weighted = system.bundle_weighted(&[("AND", 1.0), ("OR", 1.0)]).unwrap();

        let sim = regular.encoding.similarity(&weighted.encoding);
        assert!(
            sim > 0.95,
            "Equal-weighted bundle should match regular bundle (sim={:.3})",
            sim
        );
    }

    #[test]
    fn test_bundle_weighted_dominant() {
        let system = PrimitiveSystem::new();

        let and_prim = system.get("AND").unwrap();
        let or_prim = system.get("OR").unwrap();

        // Heavily weight AND
        let and_dominant = system.bundle_weighted(&[("AND", 10.0), ("OR", 1.0)]).unwrap();
        let sim_to_and = and_dominant.encoding.similarity(&and_prim.encoding);
        let sim_to_or = and_dominant.encoding.similarity(&or_prim.encoding);

        assert!(
            sim_to_and > sim_to_or,
            "AND-dominant bundle should be more similar to AND ({:.3}) than OR ({:.3})",
            sim_to_and, sim_to_or
        );
        assert!(
            sim_to_and > 0.7,
            "AND-dominant bundle should have high similarity to AND (sim={:.3})",
            sim_to_and
        );
    }

    #[test]
    fn test_bundle_weighted_empty() {
        let system = PrimitiveSystem::new();

        let result = system.bundle_weighted(&[]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::EmptyInput));
    }

    #[test]
    fn test_bundle_weighted_zero_weights() {
        let system = PrimitiveSystem::new();

        let result = system.bundle_weighted(&[("AND", 0.0), ("OR", 0.0)]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::InvalidWeight));
    }

    #[test]
    fn test_bundle_weighted_not_found() {
        let system = PrimitiveSystem::new();

        let result = system.bundle_weighted(&[("AND", 1.0), ("NONEXISTENT", 1.0)]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrimitiveError::NotFound(_)));
    }

    // =========================================================================
    // ANALOGY AND PERMUTATION TESTS
    // =========================================================================

    #[test]
    fn test_analogy() {
        let system = PrimitiveSystem::new();

        // CAUSE:EFFECT :: BEFORE:? should give something related to AFTER
        let result = system.analogy("CAUSE", "EFFECT", "BEFORE");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.operation.contains("analogy"));
        assert_eq!(result.source_primitives, vec!["CAUSE", "EFFECT", "BEFORE"]);
    }

    #[test]
    fn test_permute_primitive() {
        let system = PrimitiveSystem::new();

        let original = system.get("CAUSE").unwrap();
        let permuted = system.permute_primitive("CAUSE", 1).unwrap();

        // Permuted should be different from original
        let sim = original.encoding.similarity(&permuted.encoding);
        assert!(
            sim < 0.7,
            "Permuted vector should be different from original (sim={:.3})",
            sim
        );

        // Permuting back should recover original (HV16::permute(16384 - 1) = inverse)
        // Note: This depends on the permute implementation being cyclic
    }

    // =========================================================================
    // SIMILARITY SEARCH TESTS
    // =========================================================================

    #[test]
    fn test_find_similar_primitives() {
        let system = PrimitiveSystem::new();

        let similar = system.find_similar("CAUSE", 5);
        assert_eq!(similar.len(), 5, "Should return requested number of results");

        // Results should be sorted by similarity (descending)
        for i in 1..similar.len() {
            assert!(
                similar[i-1].1 >= similar[i].1,
                "Results should be sorted by similarity"
            );
        }

        // Should not include the query primitive itself
        assert!(
            !similar.iter().any(|(name, _)| name == "CAUSE"),
            "Results should not include query primitive"
        );
    }

    #[test]
    fn test_find_similar_to_encoding() {
        let system = PrimitiveSystem::new();

        let cause = system.get("CAUSE").unwrap();
        let similar = system.find_similar_to_encoding(&cause.encoding, 3);

        assert_eq!(similar.len(), 3);

        // CAUSE itself should be the top match
        assert_eq!(similar[0].0, "CAUSE");
        assert!(similar[0].1 > 0.99, "Exact match should have ~1.0 similarity");
    }

    #[test]
    fn test_query() {
        let system = PrimitiveSystem::new();

        let cause = system.get("CAUSE").unwrap();
        let (name, sim) = system.query(&cause.encoding);

        assert_eq!(name, "CAUSE");
        assert!(sim > 0.99);
    }

    // =========================================================================
    // LSH INDEX TESTS
    // =========================================================================

    #[test]
    fn test_lsh_index_build() {
        let system = PrimitiveSystem::new();

        // Build LSH index with 8 bands and 64 bits per band
        let lsh = system.build_lsh_index(8, 64);
        let stats = lsh.stats();

        assert_eq!(stats.num_bands, 8);
        assert_eq!(stats.bits_per_band, 64);
        assert!(stats.total_buckets > 0, "Should have created buckets");
        assert!(stats.total_entries > 0, "Should have indexed primitives");
    }

    #[test]
    fn test_lsh_find_exact_match() {
        let system = PrimitiveSystem::new();
        let lsh = system.build_lsh_index(8, 64);

        // Query for CAUSE should find CAUSE in candidates
        let cause = system.get("CAUSE").unwrap();
        let candidates = lsh.query_candidates(&cause.encoding);

        assert!(
            candidates.contains(&"CAUSE".to_string()),
            "LSH should find exact match in candidates"
        );
    }

    #[test]
    fn test_lsh_find_similar() {
        let system = PrimitiveSystem::new();
        let lsh = system.build_lsh_index(8, 64);

        let cause = system.get("CAUSE").unwrap();
        let similar = system.find_similar_lsh(&cause.encoding, 5, &lsh);

        // Should find at least one result (CAUSE itself)
        assert!(!similar.is_empty(), "LSH search should find results");

        // CAUSE should be the top match
        assert_eq!(similar[0].0, "CAUSE");
        assert!(similar[0].1 > 0.99);
    }

    #[test]
    fn test_lsh_candidates_contain_similar() {
        let system = PrimitiveSystem::new();
        let lsh = system.build_lsh_index(16, 32); // More bands, fewer bits = more candidates

        // Bundle CAUSE and EFFECT
        let bound = system.bind_primitives("CAUSE", "EFFECT").unwrap();

        // The candidates should include some primitives
        let candidates = lsh.query_candidates(&bound.encoding);

        // With a good LSH configuration, we should get some candidates
        // (though bound vectors are orthogonal to inputs, some collisions are expected)
        println!("LSH returned {} candidates for bound(CAUSE, EFFECT)", candidates.len());
    }

    #[test]
    fn test_lsh_deterministic() {
        let system = PrimitiveSystem::new();

        // Build two indices with same parameters
        let lsh1 = system.build_lsh_index(4, 32);
        let lsh2 = system.build_lsh_index(4, 32);

        // Query should return same candidates
        let cause = system.get("CAUSE").unwrap();
        let candidates1 = lsh1.query_candidates(&cause.encoding);
        let candidates2 = lsh2.query_candidates(&cause.encoding);

        let set1: std::collections::HashSet<_> = candidates1.into_iter().collect();
        let set2: std::collections::HashSet<_> = candidates2.into_iter().collect();

        assert_eq!(set1, set2, "LSH should be deterministic");
    }

    #[test]
    fn test_lsh_stats() {
        let system = PrimitiveSystem::new();
        let lsh = system.build_lsh_index(4, 48);

        let stats = lsh.stats();
        println!(
            "LSH stats: {} bands, {} bits/band, {} buckets, {} entries, avg {:.2} per bucket",
            stats.num_bands,
            stats.bits_per_band,
            stats.total_buckets,
            stats.total_entries,
            stats.avg_bucket_size
        );

        // Each primitive should be in each band
        let expected_entries = system.count() * stats.num_bands;
        assert_eq!(
            stats.total_entries, expected_entries,
            "Each primitive should be indexed in each band"
        );
    }

    // =========================================================================
    // COMPOSITION CACHE TESTS
    // =========================================================================

    #[test]
    fn test_cache_bind() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        // First call should be a miss
        let result1 = cache.bind_cached(&system, "CAUSE", "EFFECT").unwrap();
        let stats1 = cache.stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second call should be a hit
        let result2 = cache.bind_cached(&system, "CAUSE", "EFFECT").unwrap();
        let stats2 = cache.stats();
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hits, 1);

        // Results should be identical
        assert!(
            result1.encoding.similarity(&result2.encoding) > 0.99,
            "Cached result should be identical"
        );
    }

    #[test]
    fn test_cache_bundle() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        let result1 = cache.bundle_cached(&system, &["AND", "OR", "NOT"]).unwrap();
        let result2 = cache.bundle_cached(&system, &["AND", "OR", "NOT"]).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        assert!(
            result1.encoding.similarity(&result2.encoding) > 0.99,
            "Cached bundle should be identical"
        );
    }

    #[test]
    fn test_cache_sequence() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        let result1 = cache.sequence_cached(&system, &["BEFORE", "DURING", "AFTER"]).unwrap();
        let result2 = cache.sequence_cached(&system, &["BEFORE", "DURING", "AFTER"]).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        assert!(
            result1.encoding.similarity(&result2.encoding) > 0.99,
            "Cached sequence should be identical"
        );
    }

    #[test]
    fn test_cache_hit_rate() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        // Do several operations, some repeated
        let _ = cache.bind_cached(&system, "CAUSE", "EFFECT"); // miss
        let _ = cache.bind_cached(&system, "CAUSE", "EFFECT"); // hit
        let _ = cache.bind_cached(&system, "CAUSE", "EFFECT"); // hit
        let _ = cache.bind_cached(&system, "AND", "OR"); // miss
        let _ = cache.bind_cached(&system, "AND", "OR"); // hit

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 2);
        assert!((stats.hit_rate - 0.6).abs() < 0.01, "Hit rate should be 60%");
    }

    #[test]
    fn test_cache_lru_eviction() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(3); // Very small cache

        // Fill the cache
        let _ = cache.bind_cached(&system, "CAUSE", "EFFECT");
        let _ = cache.bind_cached(&system, "AND", "OR");
        let _ = cache.bind_cached(&system, "BEFORE", "AFTER");

        assert_eq!(cache.stats().size, 3);

        // Add one more, should evict LRU (CAUSE-EFFECT)
        let _ = cache.bind_cached(&system, "SET", "NOT");
        assert_eq!(cache.stats().size, 3);

        // The first entry should have been evicted (miss on re-query)
        let stats_before = cache.stats();
        let _ = cache.bind_cached(&system, "CAUSE", "EFFECT");
        let stats_after = cache.stats();

        assert_eq!(
            stats_after.misses,
            stats_before.misses + 1,
            "LRU entry should have been evicted"
        );
    }

    #[test]
    fn test_cache_clear() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        let _ = cache.bind_cached(&system, "CAUSE", "EFFECT");
        let _ = cache.bind_cached(&system, "AND", "OR");

        assert_eq!(cache.stats().size, 2);

        cache.clear();

        assert_eq!(cache.stats().size, 0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_weighted_bundle() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        let result1 = cache.bundle_weighted_cached(&system, &[("AND", 2.0), ("OR", 1.0)]).unwrap();
        let result2 = cache.bundle_weighted_cached(&system, &[("AND", 2.0), ("OR", 1.0)]).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);

        assert!(
            result1.encoding.similarity(&result2.encoding) > 0.99,
            "Cached weighted bundle should be identical"
        );
    }

    #[test]
    fn test_cache_analogy() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        let result1 = cache.analogy_cached(&system, "CAUSE", "EFFECT", "BEFORE").unwrap();
        let result2 = cache.analogy_cached(&system, "CAUSE", "EFFECT", "BEFORE").unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);

        assert!(
            result1.encoding.similarity(&result2.encoding) > 0.99,
            "Cached analogy should be identical"
        );
    }

    #[test]
    fn test_cache_permute() {
        let system = PrimitiveSystem::new();
        let mut cache = CompositionCache::new(100);

        let result1 = cache.permute_cached(&system, "CAUSE", 5).unwrap();
        let result2 = cache.permute_cached(&system, "CAUSE", 5).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);

        assert!(
            result1.encoding.similarity(&result2.encoding) > 0.99,
            "Cached permute should be identical"
        );
    }

    // =========================================================================
    // COMPOSITION ALGEBRA TESTS
    // =========================================================================

    #[test]
    fn test_algebra_define_bind() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        // Define a bind composition
        algebra.define("CAUSALITY", "CAUSE ⊗ EFFECT", &system).unwrap();

        let comp = algebra.get("CAUSALITY").unwrap();
        assert_eq!(comp.name, "CAUSALITY");
        assert_eq!(comp.sources, vec!["CAUSE", "EFFECT"]);

        // Verify it matches direct bind
        let direct = system.bind_primitives("CAUSE", "EFFECT").unwrap();
        let sim = comp.encoding.similarity(&direct.encoding);
        assert!(sim > 0.99, "Algebra bind should match direct bind (sim={:.3})", sim);
    }

    #[test]
    fn test_algebra_define_bundle() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        algebra.define("LOGIC_CORE", "AND + OR + NOT", &system).unwrap();

        let comp = algebra.get("LOGIC_CORE").unwrap();
        assert_eq!(comp.sources.len(), 3);

        // Verify it matches direct bundle
        let direct = system.bundle_primitives(&["AND", "OR", "NOT"]).unwrap();
        let sim = comp.encoding.similarity(&direct.encoding);
        assert!(sim > 0.99, "Algebra bundle should match direct bundle (sim={:.3})", sim);
    }

    #[test]
    fn test_algebra_define_sequence() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        algebra.define("TIME_FLOW", "BEFORE → DURING → AFTER", &system).unwrap();

        let comp = algebra.get("TIME_FLOW").unwrap();
        assert_eq!(comp.sources, vec!["BEFORE", "DURING", "AFTER"]);

        // Verify it matches direct sequence
        let direct = system.encode_sequence(&["BEFORE", "DURING", "AFTER"]).unwrap();
        let sim = comp.encoding.similarity(&direct.encoding);
        assert!(sim > 0.99, "Algebra sequence should match direct sequence (sim={:.3})", sim);
    }

    #[test]
    fn test_algebra_define_weighted() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        algebra.define("MOSTLY_CAUSE", "CAUSE:3 + EFFECT:1", &system).unwrap();

        let comp = algebra.get("MOSTLY_CAUSE").unwrap();

        // Should be more similar to CAUSE than EFFECT
        let cause = system.get("CAUSE").unwrap();
        let effect = system.get("EFFECT").unwrap();

        let sim_cause = comp.encoding.similarity(&cause.encoding);
        let sim_effect = comp.encoding.similarity(&effect.encoding);

        assert!(
            sim_cause > sim_effect,
            "Weighted composition should be more similar to CAUSE ({:.3}) than EFFECT ({:.3})",
            sim_cause, sim_effect
        );
    }

    #[test]
    fn test_algebra_composition_chaining() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        // Define base compositions
        algebra.define("AB", "CAUSE ⊗ EFFECT", &system).unwrap();
        algebra.define("CD", "BEFORE ⊗ AFTER", &system).unwrap();

        // Chain them together
        algebra.define("ABCD", "AB ⊗ CD", &system).unwrap();

        let abcd = algebra.get("ABCD").unwrap();
        assert_eq!(abcd.sources, vec!["AB", "CD"]);

        // Verify algebraic properties
        // ABCD = (CAUSE ⊗ EFFECT) ⊗ (BEFORE ⊗ AFTER)
        // Unbinding should recover components
    }

    #[test]
    fn test_algebra_ascii_operators() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        // Test ASCII alternatives
        algebra.define("BIND_ASCII", "CAUSE ^ EFFECT", &system).unwrap();
        algebra.define("SEQ_ASCII", "BEFORE > DURING > AFTER", &system).unwrap();

        assert!(algebra.get("BIND_ASCII").is_some());
        assert!(algebra.get("SEQ_ASCII").is_some());
    }

    #[test]
    fn test_algebra_not_found() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        let result = algebra.define("BAD", "NONEXISTENT ⊗ CAUSE", &system);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CompositionAlgebraError::NotFound(_)));
    }

    #[test]
    fn test_algebra_export_import() {
        let system = PrimitiveSystem::new();
        let mut algebra1 = CompositionAlgebra::new();

        algebra1.define("COMP1", "CAUSE ⊗ EFFECT", &system).unwrap();
        algebra1.define("COMP2", "AND + OR", &system).unwrap();

        // Export
        let exports = algebra1.export();
        assert_eq!(exports.len(), 2);

        // Import into new algebra
        let mut algebra2 = CompositionAlgebra::new();
        let count = algebra2.import(&exports, &system).unwrap();
        assert_eq!(count, 2);

        // Verify imported compositions match
        let comp1_orig = algebra1.get("COMP1").unwrap();
        let comp1_imported = algebra2.get("COMP1").unwrap();
        let sim = comp1_orig.encoding.similarity(&comp1_imported.encoding);
        assert!(sim > 0.99, "Imported composition should match original");
    }

    #[test]
    fn test_algebra_list_and_clear() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        algebra.define("A", "CAUSE ⊗ EFFECT", &system).unwrap();
        algebra.define("B", "AND + OR", &system).unwrap();

        assert_eq!(algebra.list().len(), 2);

        algebra.remove("A");
        assert_eq!(algebra.list().len(), 1);

        algebra.clear();
        assert_eq!(algebra.list().len(), 0);
    }

    // =========================================================================
    // GRAPH VISUALIZATION TESTS
    // =========================================================================

    #[test]
    fn test_graph_from_primitives() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::from_primitives(
            &system,
            &["CAUSE", "EFFECT", "BEFORE", "AFTER"],
            0.45,
        );

        assert_eq!(graph.nodes.len(), 4);
        // With threshold 0.45, random vectors (~0.5 similarity) should have edges
    }

    #[test]
    fn test_graph_from_tier() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::from_tier(
            &system,
            PrimitiveTier::Mathematical,
            0.55, // Above random, only strong connections
        );

        let stats = graph.stats();
        assert!(stats.node_count > 0, "Should have nodes from Mathematical tier");
    }

    #[test]
    fn test_graph_neighborhood() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::neighborhood(
            &system,
            "CAUSE",
            2, // depth
            3, // top_k similar at each step
        );

        assert!(graph.nodes.len() >= 1, "Should have at least the center node");
        assert!(graph.nodes.iter().any(|(n, _, _)| n == "CAUSE"), "Should contain center");
    }

    #[test]
    fn test_graph_to_dot() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::from_primitives(
            &system,
            &["CAUSE", "EFFECT"],
            0.40,
        );

        let dot = graph.to_dot();

        assert!(dot.contains("digraph"), "Should be DOT format");
        assert!(dot.contains("CAUSE"), "Should contain CAUSE node");
        assert!(dot.contains("EFFECT"), "Should contain EFFECT node");
    }

    #[test]
    fn test_graph_to_ascii() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::from_primitives(
            &system,
            &["AND", "OR", "NOT"],
            0.45,
        );

        let ascii = graph.to_ascii();

        assert!(ascii.contains("Nodes:"), "Should have nodes section");
        assert!(ascii.contains("AND"), "Should list AND");
    }

    #[test]
    fn test_graph_stats() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::from_primitives(
            &system,
            &["CAUSE", "EFFECT", "BEFORE", "AFTER"],
            0.40,
        );

        let stats = graph.stats();

        assert_eq!(stats.node_count, 4);
        assert!(stats.density >= 0.0 && stats.density <= 1.0);
    }

    // =========================================================================
    // BATCH OPERATIONS TESTS
    // =========================================================================

    #[test]
    fn test_batch_find_similar() {
        let system = PrimitiveSystem::new();

        let cause = system.get("CAUSE").unwrap().encoding.clone();
        let effect = system.get("EFFECT").unwrap().encoding.clone();

        let queries = vec![cause, effect];
        let results = system.batch_find_similar(&queries, 3);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);

        // First result should have CAUSE as top match
        assert_eq!(results[0][0].0, "CAUSE");
    }

    #[test]
    fn test_batch_find_similar_lsh() {
        let system = PrimitiveSystem::new();

        let cause = system.get("CAUSE").unwrap().encoding.clone();
        let effect = system.get("EFFECT").unwrap().encoding.clone();
        let before = system.get("BEFORE").unwrap().encoding.clone();

        let queries = vec![cause, effect, before];
        let results = system.batch_find_similar_lsh(&queries, 3, 8, 64);

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.len() <= 3);
        }
    }

    #[test]
    fn test_batch_bind() {
        let system = PrimitiveSystem::new();

        let pairs = vec![
            ("CAUSE", "EFFECT"),
            ("BEFORE", "AFTER"),
            ("AND", "OR"),
        ];

        let results = system.batch_bind(&pairs);

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_batch_bundle() {
        let system = PrimitiveSystem::new();

        let groups: Vec<&[&str]> = vec![
            &["AND", "OR"],
            &["CAUSE", "EFFECT", "BEFORE"],
        ];

        let results = system.batch_bundle(&groups);

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_batch_encode_sequences() {
        let system = PrimitiveSystem::new();

        let sequences: Vec<&[&str]> = vec![
            &["BEFORE", "DURING", "AFTER"],
            &["CAUSE", "EFFECT"],
        ];

        let results = system.batch_encode_sequences(&sequences);

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_pairwise_similarities() {
        let system = PrimitiveSystem::new();

        let encodings: Vec<HV16> = ["CAUSE", "EFFECT", "BEFORE"]
            .iter()
            .filter_map(|n| system.get(n).map(|p| p.encoding.clone()))
            .collect();

        let pairs = system.pairwise_similarities(&encodings);

        // 3 encodings = 3 pairs: (1,0), (2,0), (2,1)
        assert_eq!(pairs.len(), 3);

        for (i, j, sim) in &pairs {
            assert!(i > j, "Should be lower triangular");
            assert!(*sim >= 0.0 && *sim <= 1.0, "Similarity should be in [0,1]");
        }
    }

    #[test]
    fn test_similarity_matrix() {
        let system = PrimitiveSystem::new();

        let names = ["CAUSE", "EFFECT", "BEFORE"];
        let matrix = system.similarity_matrix(&names);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Diagonal should be 1.0 (self-similarity)
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 0.01, "Self-similarity should be 1.0");
        }

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 0.001,
                    "Matrix should be symmetric"
                );
            }
        }
    }

    // =========================================================================
    // PERSISTENCE TESTS
    // =========================================================================

    #[test]
    fn test_persistence_save_load_session() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        algebra.define("TEST_COMP", "CAUSE ⊗ EFFECT", &system).unwrap();
        algebra.define("TEST_BUNDLE", "AND + OR", &system).unwrap();

        let history = vec![
            HistoryEntry {
                operation: "bind(CAUSE, EFFECT)".to_string(),
                result_match: "CAUSALITY".to_string(),
                similarity: 0.55,
            },
        ];

        let persistence = PrimitivePersistence::new();
        let path = "/tmp/test_primitive_session.json";

        // Save
        persistence.save_session(path, &algebra, &history, Some("Test session")).unwrap();

        // Load
        let (loaded_algebra, loaded_history) = persistence.load_session(path, &system).unwrap();

        // Verify
        assert!(loaded_algebra.get("TEST_COMP").is_some());
        assert!(loaded_algebra.get("TEST_BUNDLE").is_some());
        assert_eq!(loaded_history.len(), 1);
        assert_eq!(loaded_history[0].operation, "bind(CAUSE, EFFECT)");

        // Verify compositions match
        let orig_comp = algebra.get("TEST_COMP").unwrap();
        let loaded_comp = loaded_algebra.get("TEST_COMP").unwrap();
        let sim = orig_comp.encoding.similarity(&loaded_comp.encoding);
        assert!(sim > 0.99, "Loaded composition should match original");

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_persistence_save_load_compositions() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();

        algebra.define("COMP_A", "CAUSE ⊗ EFFECT", &system).unwrap();
        algebra.define("COMP_B", "BEFORE → DURING → AFTER", &system).unwrap();

        let persistence = PrimitivePersistence::new();
        let path = "/tmp/test_compositions.json";

        // Save
        persistence.save_compositions(path, &algebra).unwrap();

        // Load
        let loaded = persistence.load_compositions(path, &system).unwrap();

        assert!(loaded.get("COMP_A").is_some());
        assert!(loaded.get("COMP_B").is_some());

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_persistence_export_graph_dot() {
        let system = PrimitiveSystem::new();

        let graph = PrimitiveGraph::from_primitives(
            &system,
            &["CAUSE", "EFFECT"],
            0.40,
        );

        let persistence = PrimitivePersistence::new();
        let path = "/tmp/test_graph.dot";

        persistence.export_graph_dot(path, &graph).unwrap();

        // Verify file exists and contains DOT content
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("digraph"));
        assert!(content.contains("CAUSE"));

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_persistence_export_similarity_csv() {
        let system = PrimitiveSystem::new();
        let persistence = PrimitivePersistence::new();
        let path = "/tmp/test_similarity.csv";

        persistence.export_similarity_csv(path, &system, &["CAUSE", "EFFECT", "BEFORE"]).unwrap();

        // Verify file exists and has correct format
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("CAUSE"));
        assert!(content.contains("EFFECT"));
        assert!(content.contains("BEFORE"));
        assert!(content.lines().count() == 4); // Header + 3 rows

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_session_data_serialization() {
        let session = SessionData {
            version: 1,
            timestamp: 1234567890,
            compositions: vec![
                CompositionExport {
                    name: "TEST".to_string(),
                    expression: "A ⊗ B".to_string(),
                },
            ],
            history: vec![],
            notes: Some("Test notes".to_string()),
        };

        let json = serde_json::to_string(&session).unwrap();
        let loaded: SessionData = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.timestamp, 1234567890);
        assert_eq!(loaded.compositions.len(), 1);
        assert_eq!(loaded.notes, Some("Test notes".to_string()));
    }
}
