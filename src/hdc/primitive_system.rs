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
//! ```rust
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

/// Generate a deterministic seed from a string name
/// This ensures primitives always get the same encoding across runs
fn seed_from_name(name: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

/// Primitive tier in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveTier {
    /// Tier 0: NSM (implemented in vocabulary.rs)
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

    /// Embed a local primitive vector into this domain's manifold
    pub fn embed(&self, local_vector: HV16) -> HV16 {
        HV16::bundle(&[self.rotation.clone(), local_vector])
    }

    /// Check if a vector belongs to this domain (via similarity)
    pub fn contains(&self, vector: &HV16, threshold: f32) -> bool {
        self.rotation.similarity(vector) > threshold
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

        // Initialize derived primitives (uncertainty, physics extensions, information theory)
        system.init_derived_primitives();

        system
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
        let probability = Primitive::derived(
            "PROBABILITY",
            PrimitiveTier::Mathematical,
            "uncertainty",
            uncertainty_domain.embed(HV16::random(seed_from_name("PROBABILITY"))),
            "Measure of likelihood: P(A) ∈ [0,1], derived from ratio of favorable to total outcomes",
            "RATIO ⊗ CERTAINTY"
        );

        // EXPECTED_VALUE = COMPOSE(PROBABILITY, VALUE)
        // E[X] = Σ P(x) × V(x)
        let expected_value = Primitive::derived(
            "EXPECTED_VALUE",
            PrimitiveTier::Mathematical,
            "uncertainty",
            uncertainty_domain.embed(HV16::random(seed_from_name("EXPECTED_VALUE"))),
            "Probability-weighted average: E[X] = Σ P(x) × V(x)",
            "PROBABILITY ⊗ VALUE"
        );

        // ENTROPY = COMPOSE(PROBABILITY, INFORMATION)
        // H = -Σ P(x) log P(x)
        let entropy = Primitive::derived(
            "ENTROPY",
            PrimitiveTier::Mathematical,
            "uncertainty",
            uncertainty_domain.embed(HV16::random(seed_from_name("ENTROPY"))),
            "Measure of uncertainty: H = -Σ P(x) log P(x), higher = more uncertain",
            "PROBABILITY ⊗ INFORMATION"
        );

        // BAYESIAN_UPDATE = COMPOSE(PROBABILITY, EVIDENCE)
        // P(H|E) = P(E|H) × P(H) / P(E)
        let bayesian = Primitive::derived(
            "BAYESIAN_UPDATE",
            PrimitiveTier::Mathematical,
            "uncertainty",
            uncertainty_domain.embed(HV16::random(seed_from_name("BAYESIAN_UPDATE"))),
            "Belief revision: P(H|E) = P(E|H) × P(H) / P(E)",
            "PROBABILITY ⊗ EVIDENCE ⊗ CONDITIONAL"
        );

        // VARIANCE = COMPOSE(EXPECTED_VALUE, DEVIATION)
        // Var(X) = E[(X - μ)²]
        let variance = Primitive::derived(
            "VARIANCE",
            PrimitiveTier::Mathematical,
            "uncertainty",
            uncertainty_domain.embed(HV16::random(seed_from_name("VARIANCE"))),
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

        // CONSERVATION = COMPOSE(STATE_CHANGE, INVARIANT)
        // Quantity that remains constant through transformation
        let conservation = Primitive::derived(
            "CONSERVATION",
            PrimitiveTier::Physical,
            "physics_extended",
            physics_ext_domain.embed(HV16::random(seed_from_name("CONSERVATION"))),
            "Invariant quantity across transformations: dQ/dt = 0",
            "STATE_CHANGE ⊗ INVARIANT"
        );

        // GRADIENT = COMPOSE(DIFFERENTIATION, SPACE)
        // Rate of change in space: ∇f
        let gradient = Primitive::derived(
            "GRADIENT",
            PrimitiveTier::Physical,
            "physics_extended",
            physics_ext_domain.embed(HV16::random(seed_from_name("GRADIENT"))),
            "Spatial rate of change: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)",
            "DIFFERENTIATION ⊗ SPACE"
        );

        // FIELD = COMPOSE(SPACE, FORCE)
        // Force at every point in space
        let field = Primitive::derived(
            "FIELD",
            PrimitiveTier::Physical,
            "physics_extended",
            physics_ext_domain.embed(HV16::random(seed_from_name("FIELD"))),
            "Assignment of force/value to each point: F(x, y, z)",
            "SPACE ⊗ FORCE"
        );

        // WAVE = COMPOSE(OSCILLATION, PROPAGATION)
        // Traveling disturbance through medium
        let wave = Primitive::derived(
            "WAVE",
            PrimitiveTier::Physical,
            "physics_extended",
            physics_ext_domain.embed(HV16::random(seed_from_name("WAVE"))),
            "Propagating oscillation: ψ(x,t) = A sin(kx - ωt)",
            "OSCILLATION ⊗ PROPAGATION"
        );

        // EQUILIBRIUM = COMPOSE(FORCE, BALANCE)
        // State where net forces sum to zero
        let equilibrium = Primitive::derived(
            "EQUILIBRIUM",
            PrimitiveTier::Physical,
            "physics_extended",
            physics_ext_domain.embed(HV16::random(seed_from_name("EQUILIBRIUM"))),
            "Balanced state: ΣF = 0, stable or unstable",
            "FORCE ⊗ BALANCE"
        );

        // POTENTIAL = COMPOSE(ENERGY, POSITION)
        // Stored energy based on position/configuration
        let potential = Primitive::derived(
            "POTENTIAL",
            PrimitiveTier::Physical,
            "physics_extended",
            physics_ext_domain.embed(HV16::random(seed_from_name("POTENTIAL"))),
            "Position-dependent energy: U(x) where F = -∇U",
            "ENERGY ⊗ POSITION"
        );

        // === INFORMATION THEORY DOMAIN ===
        // Derived from: Mathematical + MetaCognitive

        let info_domain = DomainManifold::new(
            "information_theory",
            PrimitiveTier::Mathematical,
            "Quantitative theory of information and communication"
        );
        self.domains.insert("information_theory".to_string(), info_domain.clone());

        // MUTUAL_INFORMATION = COMPOSE(ENTROPY, BINDING)
        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        let mutual_info = Primitive::derived(
            "MUTUAL_INFORMATION",
            PrimitiveTier::Mathematical,
            "information_theory",
            info_domain.embed(HV16::random(seed_from_name("MUTUAL_INFORMATION"))),
            "Shared information: I(X;Y) = H(X) + H(Y) - H(X,Y)",
            "ENTROPY ⊗ BINDING"
        );

        // INFORMATION_GAIN = COMPOSE(ENTROPY, REDUCTION)
        // IG = H(before) - H(after|evidence)
        let info_gain = Primitive::derived(
            "INFORMATION_GAIN",
            PrimitiveTier::Mathematical,
            "information_theory",
            info_domain.embed(HV16::random(seed_from_name("INFORMATION_GAIN"))),
            "Entropy reduction from evidence: IG = H(S) - H(S|E)",
            "ENTROPY ⊗ REDUCTION"
        );

        // CHANNEL_CAPACITY = COMPOSE(INFORMATION, LIMIT)
        // Maximum mutual information: C = max I(X;Y)
        let capacity = Primitive::derived(
            "CHANNEL_CAPACITY",
            PrimitiveTier::Mathematical,
            "information_theory",
            info_domain.embed(HV16::random(seed_from_name("CHANNEL_CAPACITY"))),
            "Maximum transmission rate: C = max I(X;Y)",
            "INFORMATION ⊗ LIMIT"
        );

        // COMPRESSION = COMPOSE(INFORMATION, EFFICIENCY)
        // Representing information with minimum bits
        let compression = Primitive::derived(
            "COMPRESSION",
            PrimitiveTier::Mathematical,
            "information_theory",
            info_domain.embed(HV16::random(seed_from_name("COMPRESSION"))),
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

        // INTEGRATED_INFORMATION = COMPOSE(MUTUAL_INFORMATION, IRREDUCIBILITY)
        // Φ = information above minimum information partition
        let phi_primitive = Primitive::derived(
            "INTEGRATED_INFORMATION",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            consciousness_domain.embed(HV16::random(seed_from_name("INTEGRATED_INFORMATION"))),
            "Consciousness measure: Φ = integrated information above MIP",
            "MUTUAL_INFORMATION ⊗ IRREDUCIBILITY"
        );

        // CAUSAL_POWER = COMPOSE(CAUSE, EFFECT, COUNTERFACTUAL)
        // Ability to make a difference
        let causal_power = Primitive::derived(
            "CAUSAL_POWER",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            consciousness_domain.embed(HV16::random(seed_from_name("CAUSAL_POWER"))),
            "Capacity to produce effects: P(effect|do(cause)) - P(effect)",
            "CAUSE ⊗ EFFECT ⊗ COUNTERFACTUAL"
        );

        // ATTENTION = COMPOSE(SALIENCE, SELECTION)
        // Selective focus on subset of information
        let attention = Primitive::derived(
            "ATTENTION",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            consciousness_domain.embed(HV16::random(seed_from_name("ATTENTION"))),
            "Selective processing: focus on salient subset of available information",
            "SALIENCE ⊗ SELECTION"
        );

        // METACOGNITION = COMPOSE(COGNITION, SELF)
        // Thinking about thinking
        let metacognition = Primitive::derived(
            "METACOGNITION",
            PrimitiveTier::MetaCognitive,
            "consciousness_derived",
            consciousness_domain.embed(HV16::random(seed_from_name("METACOGNITION"))),
            "Cognition about cognition: awareness of mental processes",
            "COGNITION ⊗ SELF"
        );

        // Add all derived primitives
        let derived_primitives = vec![
            // Uncertainty
            probability, expected_value, entropy, bayesian, variance,
            // Physics
            conservation, gradient, field, wave, equilibrium, potential,
            // Information Theory
            mutual_info, info_gain, capacity, compression,
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
            example: "CONSERVATION ⊗ IDENTITY → persistent self".to_string(),
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

        // === REGISTER ALL PRIMITIVES ===

        self.domains.insert("mathematics".to_string(), math_domain);
        self.domains.insert("logic".to_string(), logic_domain);

        for primitive in vec![
            set, membership, union, intersection, empty_set,
            not, and, or, implies, iff, equals, true_const, false_const,
            zero, one, successor, addition, multiplication,
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

        // ENTROPY - measure of disorder
        let entropy = Primitive::base(
            "ENTROPY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(HV16::random(seed_from_name("ENTROPY"))),
            "Property: measure of disorder or randomness (J/K)"
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

        // === REGISTER ALL TIER 2 PRIMITIVES ===

        self.domains.insert("physics".to_string(), physics_domain);
        self.domains.insert("causality".to_string(), causality_domain);

        for primitive in vec![
            mass, charge, spin,
            energy, work, force,
            velocity, acceleration, momentum,
            cause, effect, state_change,
            entropy, temperature, conservation,
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

        // === REGISTER ALL TIER 5 PRIMITIVES ===

        self.domains.insert("metacognition".to_string(), metacognition_domain);
        self.domains.insert("homeostasis".to_string(), homeostasis_domain);
        self.domains.insert("epistemic".to_string(), epistemic_domain);
        self.domains.insert("metabolic".to_string(), metabolic_domain);

        for primitive in vec![
            self_prim, identity, meta_belief, introspection,
            homeostasis, setpoint, regulation, feedback,
            repair, restore, adapt, learn,
            know, uncertain, confidence, evidence,
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

    /// Validate that all primitives in a tier are sufficiently orthogonal
    pub fn validate_tier_orthogonality(&self, tier: PrimitiveTier, threshold: f32) -> Vec<(String, String, f32)> {
        let mut violations = Vec::new();
        let primitives = self.get_tier(tier);

        for i in 0..primitives.len() {
            for j in (i+1)..primitives.len() {
                let sim = primitives[i].encoding.similarity(&primitives[j].encoding);
                if sim > threshold {
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
}

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

        // They should be in different domains, so should be fairly orthogonal
        if let Some(similarity) = sim {
            assert!(similarity < 0.7, "Different domain primitives should be fairly orthogonal");
        }
    }

    #[test]
    fn test_tier_validation() {
        let system = PrimitiveSystem::new();

        // Check that Tier 1 primitives are sufficiently orthogonal
        let violations = system.validate_tier_orthogonality(PrimitiveTier::Mathematical, 0.9);

        // Should have few or no violations
        assert!(violations.len() < system.count_tier(PrimitiveTier::Mathematical) / 2,
            "Most primitives should be orthogonal");
    }

    #[test]
    fn test_domain_manifolds() {
        let system = PrimitiveSystem::new();

        let math = system.domain("mathematics");
        let logic = system.domain("logic");

        assert!(math.is_some(), "Mathematics domain should exist");
        assert!(logic.is_some(), "Logic domain should exist");

        // Domains should have different rotations
        if let (Some(m), Some(l)) = (math, logic) {
            let sim = m.rotation.similarity(&l.rotation);
            assert!(sim < 0.8, "Different domains should have different rotations");
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
        assert!(system.get("ENTROPY").is_some(), "ENTROPY primitive should exist");
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

        // Check that Tier 2 primitives are sufficiently orthogonal
        let violations = system.validate_tier_orthogonality(PrimitiveTier::Physical, 0.9);

        // Should have few violations
        assert!(violations.len() < system.count_tier(PrimitiveTier::Physical) / 2,
            "Most Tier 2 primitives should be orthogonal");
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

        // Check that Tier 3 primitives are sufficiently orthogonal
        let violations = system.validate_tier_orthogonality(PrimitiveTier::Geometric, 0.9);

        assert!(violations.len() < system.count_tier(PrimitiveTier::Geometric) / 2,
            "Most Tier 3 primitives should be orthogonal");
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

        let violations = system.validate_tier_orthogonality(PrimitiveTier::Strategic, 0.9);

        assert!(violations.len() < system.count_tier(PrimitiveTier::Strategic) / 2,
            "Most Tier 4 primitives should be orthogonal");
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

        let violations = system.validate_tier_orthogonality(PrimitiveTier::MetaCognitive, 0.9);

        assert!(violations.len() < system.count_tier(PrimitiveTier::MetaCognitive) / 2,
            "Most Tier 5 primitives should be orthogonal");
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
        assert!(system.get("ENTROPY").is_some());

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
}
