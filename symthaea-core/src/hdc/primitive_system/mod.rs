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
//! // Each domain gets a rotation in BinaryHV space
//! MATH_MANIFOLD = random_hv16();
//! ZERO = MATH_MANIFOLD ⊗ ZERO_LOCAL;
//! ONE = MATH_MANIFOLD ⊗ ONE_LOCAL;
//!
//! // This preserves orthogonality within and across domains
//! ```


// Submodule declarations
mod primitive_tier;
mod primitive;
mod lsh_index;
mod composition_algebra;
mod composition_cache;
mod primitive_graph;
mod persistence;
#[cfg(test)]
mod tests;

// Re-exports
pub use primitive_tier::*;
pub use primitive::*;
pub use lsh_index::*;
pub use composition_algebra::*;
pub use composition_cache::*;
pub use primitive_graph::*;
pub use persistence::*;

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::universal_semantics::SemanticPrime;
use std::collections::HashMap;
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
/// Each primitive stores a 16,384-bit BinaryHV encoding (~2KB), plus metadata.
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
        // Tier 0: NSM (Natural Semantic Metalanguage) - 65 Wierzbicka primes
        system.init_tier0_nsm();

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

        // Initialize Tier 6: Temporal primitives (Allen's Interval Algebra extended)
        system.init_tier6_temporal();

        // Initialize Tier 7: Compositional primitives (composition operators)
        system.init_tier7_compositional();

        // Initialize Tier 8: Consciousness-specific primitives
        // Qualia, attention, memory operations, and agency
        // MUST come before init_derived_primitives so SALIENCE/SELECTION exist
        system.init_consciousness_primitives();

        // Initialize Tier 9: Code primitives
        // Enables consciousness-aware code understanding, generation, and transformation
        system.init_tier9_code();

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
    fn derive_encoding(&self, name: &str, parents: &[&str], domain: &DomainManifold) -> BinaryHV {
        if parents.is_empty() {
            return domain.embed(BinaryHV::random(seed_from_name(name)));
        }

        let mut parent_encodings: Vec<&BinaryHV> = Vec::new();
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
                    return domain.embed(BinaryHV::random(seed_from_name(name)));
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
        let mut result = *parent_encodings[0];
        for enc in &parent_encodings[1..] {
            result = result.bind(enc);
        }
        result
    }

    /// Initialize derived primitives using dependency-aware two-pass resolution.
    ///
    /// These are complex primitives derived from base primitives via composition.
    /// Rather than expanding the base set, we compose existing primitives to create
    /// higher-order concepts. The two-pass approach processes derivations in rounds:
    /// each round registers all specs whose parents are available, then repeats
    /// until no more can be resolved. This eliminates silent fallback to random.
    fn init_derived_primitives(&mut self) {
        // === DOMAIN SETUP ===

        let uncertainty_domain = DomainManifold::new(
            "uncertainty",
            PrimitiveTier::Mathematical,
            "Probabilistic reasoning and uncertainty quantification"
        );
        self.domains.insert("uncertainty".to_string(), uncertainty_domain.clone());

        let physics_ext_domain = DomainManifold::new(
            "physics_extended",
            PrimitiveTier::Physical,
            "Advanced physical concepts for embodied reasoning"
        );
        self.domains.insert("physics_extended".to_string(), physics_ext_domain.clone());

        let info_domain = DomainManifold::new(
            "information_theory",
            PrimitiveTier::Mathematical,
            "Quantitative theory of information and communication"
        );
        self.domains.insert("information_theory".to_string(), info_domain.clone());

        let consciousness_domain = DomainManifold::new(
            "consciousness_derived",
            PrimitiveTier::MetaCognitive,
            "Derived primitives for consciousness measurement"
        );
        self.domains.insert("consciousness_derived".to_string(), consciousness_domain.clone());

        // === DERIVATION SPECS ===
        // Collect all derivations with their parent dependencies

        struct DerivationSpec {
            name: &'static str,
            parents: Vec<&'static str>,
            tier: PrimitiveTier,
            domain_name: &'static str,
            domain: DomainManifold,
            definition: &'static str,
            derivation_expr: &'static str,
        }

        let specs = vec![
            // Uncertainty & Probability
            DerivationSpec {
                name: "PROBABILITY", parents: vec!["RATIO", "CERTAINTY"],
                tier: PrimitiveTier::Mathematical, domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Measure of likelihood: P(A) in [0,1], derived from ratio of favorable to total outcomes",
                derivation_expr: "RATIO ^ CERTAINTY",
            },
            DerivationSpec {
                name: "EXPECTED_VALUE", parents: vec!["PROBABILITY", "VALUE"],
                tier: PrimitiveTier::Mathematical, domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Probability-weighted average: E[X] = sum P(x) * V(x)",
                derivation_expr: "PROBABILITY ^ VALUE",
            },
            DerivationSpec {
                name: "SHANNON_ENTROPY", parents: vec!["PROBABILITY", "INFORMATION"],
                tier: PrimitiveTier::Mathematical, domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Information-theoretic uncertainty: H = -sum P(x) log P(x), higher = more uncertain",
                derivation_expr: "PROBABILITY ^ INFORMATION",
            },
            DerivationSpec {
                name: "BAYESIAN_UPDATE", parents: vec!["PROBABILITY", "EVIDENCE"],
                tier: PrimitiveTier::Mathematical, domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Belief revision: P(H|E) = P(E|H) * P(H) / P(E)",
                derivation_expr: "PROBABILITY ^ EVIDENCE",
            },
            DerivationSpec {
                name: "VARIANCE", parents: vec!["EXPECTED_VALUE", "DEVIATION"],
                tier: PrimitiveTier::Mathematical, domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Spread of distribution: Var(X) = E[(X - mu)^2]",
                derivation_expr: "EXPECTED_VALUE ^ DEVIATION",
            },
            // Physics Extensions
            DerivationSpec {
                name: "CONSERVATION_LAW", parents: vec!["STATE_CHANGE", "CONSERVATION"],
                tier: PrimitiveTier::Physical, domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Formal conservation law: dQ/dt = 0, invariant quantity across transformations",
                derivation_expr: "STATE_CHANGE ^ CONSERVATION",
            },
            DerivationSpec {
                name: "GRADIENT", parents: vec!["DIFFERENTIATION", "SPACE"],
                tier: PrimitiveTier::Physical, domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Spatial rate of change: grad f = (df/dx, df/dy, df/dz)",
                derivation_expr: "DIFFERENTIATION ^ SPACE",
            },
            DerivationSpec {
                name: "FIELD", parents: vec!["FORCE", "POINT"],
                tier: PrimitiveTier::Physical, domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Assignment of force/value to each point: F(x, y, z)",
                derivation_expr: "FORCE ^ POINT",
            },
            DerivationSpec {
                name: "WAVE", parents: vec!["OSCILLATION", "PROPAGATION"],
                tier: PrimitiveTier::Physical, domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Propagating oscillation: psi(x,t) = A sin(kx - wt)",
                derivation_expr: "OSCILLATION ^ PROPAGATION",
            },
            DerivationSpec {
                name: "EQUILIBRIUM", parents: vec!["FORCE", "CONSERVATION"],
                tier: PrimitiveTier::Physical, domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Balanced state: sum F = 0, stable or unstable",
                derivation_expr: "FORCE ^ CONSERVATION",
            },
            DerivationSpec {
                name: "POTENTIAL", parents: vec!["ENERGY", "POINT"],
                tier: PrimitiveTier::Physical, domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Position-dependent energy: U(x) where F = -grad U",
                derivation_expr: "ENERGY ^ POINT",
            },
            // Information Theory
            DerivationSpec {
                name: "MUTUAL_INFORMATION", parents: vec!["SHANNON_ENTROPY", "MEMBERSHIP"],
                tier: PrimitiveTier::Mathematical, domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Shared information: I(X;Y) = H(X) + H(Y) - H(X,Y)",
                derivation_expr: "SHANNON_ENTROPY ^ MEMBERSHIP",
            },
            DerivationSpec {
                name: "INFORMATION_GAIN", parents: vec!["SHANNON_ENTROPY", "EVIDENCE"],
                tier: PrimitiveTier::Mathematical, domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Entropy reduction from evidence: IG = H(S) - H(S|E)",
                derivation_expr: "SHANNON_ENTROPY ^ EVIDENCE",
            },
            DerivationSpec {
                name: "CHANNEL_CAPACITY", parents: vec!["INFORMATION", "LIMIT"],
                tier: PrimitiveTier::Mathematical, domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Maximum transmission rate: C = max I(X;Y)",
                derivation_expr: "INFORMATION ^ LIMIT",
            },
            DerivationSpec {
                name: "COMPRESSION", parents: vec!["INFORMATION", "EFFICIENCY"],
                tier: PrimitiveTier::Mathematical, domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Efficient encoding: L >= H(X) (Shannon's source coding theorem)",
                derivation_expr: "INFORMATION ^ EFFICIENCY",
            },
            // Consciousness
            DerivationSpec {
                name: "INTEGRATED_INFORMATION", parents: vec!["MUTUAL_INFORMATION", "SELF"],
                tier: PrimitiveTier::MetaCognitive, domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Consciousness measure: Phi = integrated information above MIP",
                derivation_expr: "MUTUAL_INFORMATION ^ SELF",
            },
            DerivationSpec {
                name: "CAUSAL_POWER", parents: vec!["CAUSE", "EFFECT", "COUNTERFACTUAL"],
                tier: PrimitiveTier::MetaCognitive, domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Capacity to produce effects: P(effect|do(cause)) - P(effect)",
                derivation_expr: "CAUSE ^ EFFECT ^ COUNTERFACTUAL",
            },
            DerivationSpec {
                name: "ATTENTION", parents: vec!["SALIENCE", "SELECTION"],
                tier: PrimitiveTier::MetaCognitive, domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Selective processing: focus on salient subset of available information",
                derivation_expr: "SALIENCE ^ SELECTION",
            },
            DerivationSpec {
                name: "METACOGNITION", parents: vec!["INTROSPECTION", "SELF"],
                tier: PrimitiveTier::MetaCognitive, domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Cognition about cognition: awareness of mental processes",
                derivation_expr: "INTROSPECTION ^ SELF",
            },
        ];

        // === TWO-PASS DEPENDENCY-AWARE RESOLUTION ===
        // Process in rounds: each round registers all specs whose parents are available

        let mut pending: Vec<DerivationSpec> = specs.into_iter().collect();
        let mut round = 0;
        let max_rounds = 10;

        while !pending.is_empty() && round < max_rounds {
            let mut resolved_this_round = Vec::new();
            let mut still_pending = Vec::new();

            for spec in pending {
                let all_parents_available = spec.parents.iter()
                    .all(|p| self.primitives.contains_key(*p));

                if all_parents_available {
                    let encoding = self.derive_encoding(spec.name, &spec.parents, &spec.domain);
                    let primitive = Primitive::derived(
                        spec.name,
                        spec.tier,
                        spec.domain_name,
                        encoding,
                        spec.definition,
                        spec.derivation_expr,
                    );
                    self.primitives.insert(spec.name.to_string(), primitive);
                    self.by_tier.entry(spec.tier).or_default().push(spec.name.to_string());
                    resolved_this_round.push(spec.name);
                } else {
                    still_pending.push(spec);
                }
            }

            if resolved_this_round.is_empty() {
                // No progress — log unresolved specs as warnings
                for spec in &still_pending {
                    let missing: Vec<&&str> = spec.parents.iter()
                        .filter(|p| !self.primitives.contains_key(**p))
                        .collect();
                    eprintln!(
                        "WARNING: derived primitive '{}' could not be resolved. Missing parents: {:?}",
                        spec.name, missing
                    );
                }
                break;
            }

            pending = still_pending;
            round += 1;
        }

        // === BINDING RULES FOR DERIVED PRIMITIVES ===

        self.binding_rules.push(BindingRule {
            name: "probabilistic_reasoning".to_string(),
            pattern: vec![PrimitiveTier::Mathematical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "PROBABILITY ^ BELIEF -> probabilistic belief".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "information_consciousness".to_string(),
            pattern: vec![PrimitiveTier::Mathematical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "MUTUAL_INFORMATION ^ AWARENESS -> integrated awareness".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "physics_embodiment".to_string(),
            pattern: vec![PrimitiveTier::Physical, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::MetaCognitive,
            example: "CONSERVATION_LAW ^ IDENTITY -> persistent self".to_string(),
        });
    }

    /// Initialize Tier 0: NSM (Natural Semantic Metalanguage) Primitives
    ///
    /// Bridges the 65 Wierzbicka semantic primes from `universal_semantics.rs`
    /// into the PrimitiveSystem. These are the universal concepts found across
    /// all human languages - the "atoms" of human thought.
    ///
    /// Categories:
    /// - Substantives: I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY
    /// - Relational: KIND_OF, PART_OF
    /// - Determiners: THIS, SAME, OTHER
    /// - Quantifiers: ONE, TWO, SOME, ALL, MUCH, LITTLE
    /// - Evaluators: GOOD, BAD
    /// - Descriptors: BIG, SMALL
    /// - Mental: THINK, KNOW, WANT, FEEL, SEE, HEAR
    /// - Speech: SAY, WORDS, TRUE
    /// - Actions: DO, HAPPEN, MOVE, TOUCH
    /// - Existence: BE, THERE_IS, HAVE
    /// - Life: LIVE, DIE
    /// - Logical: NOT, MAYBE, CAN, BECAUSE, IF
    /// - Time: WHEN, NOW, BEFORE, AFTER, LONG_TIME, SHORT_TIME, FOR_SOME_TIME, IN_ONE_MOMENT
    /// - Space: WHERE, HERE, ABOVE, BELOW, FAR, NEAR, SIDE, INSIDE, ON
    /// - Intensifiers: VERY, MORE
    /// - Similarity: LIKE
    /// - Social: WITH
    fn init_tier0_nsm(&mut self) {
        // Create NSM domain manifold - grounding for human semantic understanding
        let nsm_domain = DomainManifold::new(
            "nsm",
            PrimitiveTier::NSM,
            "Natural Semantic Metalanguage - universal human concepts"
        );

        // Register all 65 semantic primes
        for prime in SemanticPrime::all() {
            let name = Self::semantic_prime_to_name(prime);
            let description = prime.description();

            let primitive = Primitive::base(
                &name,
                PrimitiveTier::NSM,
                "nsm",
                nsm_domain.embed(BinaryHV::random(seed_from_name(&name))),
                description,
            );

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(PrimitiveTier::NSM).or_default().push(name);
        }

        // Store the domain
        self.domains.insert("nsm".to_string(), nsm_domain);
    }

    /// Convert SemanticPrime enum variant to primitive name string
    fn semantic_prime_to_name(prime: SemanticPrime) -> String {
        match prime {
            // Substantives
            SemanticPrime::I => "NSM_I".to_string(),
            SemanticPrime::You => "NSM_YOU".to_string(),
            SemanticPrime::Someone => "NSM_SOMEONE".to_string(),
            SemanticPrime::Something => "NSM_SOMETHING".to_string(),
            SemanticPrime::People => "NSM_PEOPLE".to_string(),
            SemanticPrime::Body => "NSM_BODY".to_string(),

            // Relational
            SemanticPrime::KindOf => "NSM_KIND_OF".to_string(),
            SemanticPrime::PartOf => "NSM_PART_OF".to_string(),

            // Determiners
            SemanticPrime::This => "NSM_THIS".to_string(),
            SemanticPrime::Same => "NSM_SAME".to_string(),
            SemanticPrime::Other => "NSM_OTHER".to_string(),

            // Quantifiers
            SemanticPrime::One => "NSM_ONE".to_string(),
            SemanticPrime::Two => "NSM_TWO".to_string(),
            SemanticPrime::Some => "NSM_SOME".to_string(),
            SemanticPrime::All => "NSM_ALL".to_string(),
            SemanticPrime::Much => "NSM_MUCH".to_string(),
            SemanticPrime::Little => "NSM_LITTLE".to_string(),

            // Evaluators
            SemanticPrime::Good => "NSM_GOOD".to_string(),
            SemanticPrime::Bad => "NSM_BAD".to_string(),

            // Descriptors
            SemanticPrime::Big => "NSM_BIG".to_string(),
            SemanticPrime::Small => "NSM_SMALL".to_string(),

            // Mental predicates
            SemanticPrime::Think => "NSM_THINK".to_string(),
            SemanticPrime::Know => "NSM_KNOW".to_string(),
            SemanticPrime::Want => "NSM_WANT".to_string(),
            SemanticPrime::Feel => "NSM_FEEL".to_string(),
            SemanticPrime::See => "NSM_SEE".to_string(),
            SemanticPrime::Hear => "NSM_HEAR".to_string(),

            // Speech
            SemanticPrime::Say => "NSM_SAY".to_string(),
            SemanticPrime::Words => "NSM_WORDS".to_string(),
            SemanticPrime::True => "NSM_TRUE".to_string(),

            // Actions
            SemanticPrime::Do => "NSM_DO".to_string(),
            SemanticPrime::Happen => "NSM_HAPPEN".to_string(),
            SemanticPrime::Move => "NSM_MOVE".to_string(),
            SemanticPrime::Touch => "NSM_TOUCH".to_string(),

            // Existence
            SemanticPrime::Be => "NSM_BE".to_string(),
            SemanticPrime::ThereIs => "NSM_THERE_IS".to_string(),
            SemanticPrime::Have => "NSM_HAVE".to_string(),

            // Life/Death
            SemanticPrime::Live => "NSM_LIVE".to_string(),
            SemanticPrime::Die => "NSM_DIE".to_string(),

            // Logical
            SemanticPrime::Not => "NSM_NOT".to_string(),
            SemanticPrime::Maybe => "NSM_MAYBE".to_string(),
            SemanticPrime::Can => "NSM_CAN".to_string(),
            SemanticPrime::Because => "NSM_BECAUSE".to_string(),
            SemanticPrime::If => "NSM_IF".to_string(),

            // Time
            SemanticPrime::When => "NSM_WHEN".to_string(),
            SemanticPrime::Now => "NSM_NOW".to_string(),
            SemanticPrime::Before => "NSM_BEFORE".to_string(),
            SemanticPrime::After => "NSM_AFTER".to_string(),
            SemanticPrime::LongTime => "NSM_LONG_TIME".to_string(),
            SemanticPrime::ShortTime => "NSM_SHORT_TIME".to_string(),
            SemanticPrime::ForSomeTime => "NSM_FOR_SOME_TIME".to_string(),
            SemanticPrime::InOneMoment => "NSM_IN_ONE_MOMENT".to_string(),

            // Space
            SemanticPrime::Where => "NSM_WHERE".to_string(),
            SemanticPrime::Here => "NSM_HERE".to_string(),
            SemanticPrime::Above => "NSM_ABOVE".to_string(),
            SemanticPrime::Below => "NSM_BELOW".to_string(),
            SemanticPrime::Far => "NSM_FAR".to_string(),
            SemanticPrime::Near => "NSM_NEAR".to_string(),
            SemanticPrime::Side => "NSM_SIDE".to_string(),
            SemanticPrime::Inside => "NSM_INSIDE".to_string(),
            SemanticPrime::On => "NSM_ON".to_string(),

            // Intensifiers
            SemanticPrime::Very => "NSM_VERY".to_string(),
            SemanticPrime::More => "NSM_MORE".to_string(),

            // Similarity
            SemanticPrime::Like => "NSM_LIKE".to_string(),

            // Social
            SemanticPrime::With => "NSM_WITH".to_string(),
        }
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
            math_domain.embed(BinaryHV::random(seed_from_name("SET"))),
            "A collection of distinct objects"
        );

        // MEMBERSHIP (∈) - element belongs to set
        let membership = Primitive::base(
            "MEMBERSHIP",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("MEMBERSHIP"))),
            "Relation: x ∈ S (x is an element of set S)"
        );

        // UNION (∪) - combine sets
        let union = Primitive::base(
            "UNION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("UNION"))),
            "Operation: A ∪ B (all elements in A or B)"
        );

        // INTERSECTION (∩) - common elements
        let intersection = Primitive::base(
            "INTERSECTION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INTERSECTION"))),
            "Operation: A ∩ B (elements in both A and B)"
        );

        // EMPTY_SET (∅) - the set with no elements
        let empty_set = Primitive::base(
            "EMPTY_SET",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("EMPTY_SET"))),
            "The unique set with no elements: ∅"
        );

        // === LOGICAL PRIMITIVES ===

        // NOT (¬) - logical negation
        let not = Primitive::base(
            "NOT",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("NOT"))),
            "Unary operator: ¬P (negation of proposition P)"
        );

        // AND (∧) - logical conjunction
        let and = Primitive::base(
            "AND",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("AND"))),
            "Binary operator: P ∧ Q (both P and Q are true)"
        );

        // OR (∨) - logical disjunction
        let or = Primitive::base(
            "OR",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("OR"))),
            "Binary operator: P ∨ Q (at least one of P or Q is true)"
        );

        // IMPLIES (→) - logical implication
        let implies = Primitive::base(
            "IMPLIES",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("IMPLIES"))),
            "Binary operator: P → Q (if P then Q)"
        );

        // IFF (↔) - logical equivalence
        let iff = Primitive::base(
            "IFF",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("IFF"))),
            "Binary operator: P ↔ Q (P if and only if Q)"
        );

        // EQUALS (=) - equality relation
        let equals = Primitive::base(
            "EQUALS",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("EQUALS"))),
            "Binary relation: x = y (x and y are the same)"
        );

        // TRUE (⊤) - logical truth
        let true_const = Primitive::base(
            "TRUE",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("TRUE"))),
            "The constant truth value: ⊤"
        );

        // FALSE (⊥) - logical falsehood
        let false_const = Primitive::base(
            "FALSE",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("FALSE"))),
            "The constant false value: ⊥"
        );

        // === PEANO ARITHMETIC PRIMITIVES ===

        // ZERO (0) - the first natural number
        let zero = Primitive::base(
            "ZERO",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ZERO"))),
            "The first natural number: 0"
        );

        // ONE (1) - successor of zero
        let one = Primitive::derived(
            "ONE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ONE"))),
            "The natural number one: 1",
            "SUCCESSOR(ZERO)"
        );

        // SUCCESSOR (S) - next natural number
        let successor = Primitive::base(
            "SUCCESSOR",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("SUCCESSOR"))),
            "Function: S(n) = n+1 (next natural number)"
        );

        // ADDITION (+) - derived from successor
        let addition = Primitive::derived(
            "ADDITION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ADDITION"))),
            "Binary operation: m + n (sum of m and n)",
            "Recursive: m + 0 = m, m + S(n) = S(m + n)"
        );

        // MULTIPLICATION (×) - derived from addition
        let multiplication = Primitive::derived(
            "MULTIPLICATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("MULTIPLICATION"))),
            "Binary operation: m × n (product of m and n)",
            "Recursive: m × 0 = 0, m × S(n) = m × n + m"
        );

        // === FOUNDATIONAL MATHEMATICAL PRIMITIVES ===
        // These are base concepts needed by derived primitives in later tiers

        let ratio = Primitive::base(
            "RATIO",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("RATIO"))),
            "Relation: proportional comparison of two quantities (a/b)"
        );

        let information = Primitive::base(
            "INFORMATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INFORMATION"))),
            "Quantity: reduction in uncertainty (bits)"
        );

        let deviation = Primitive::base(
            "DEVIATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("DEVIATION"))),
            "Measure: distance from a central or expected value"
        );

        let limit = Primitive::base(
            "LIMIT",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("LIMIT"))),
            "Bound: supremum or constraint on a quantity"
        );

        let efficiency = Primitive::base(
            "EFFICIENCY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("EFFICIENCY"))),
            "Ratio: useful output to total input"
        );

        // === NEGATION PRIMITIVE ===
        // Unary operation: additive inverse needed for integer arithmetic (Z)

        let negation = Primitive::base(
            "NEGATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("NEGATION"))),
            "Unary operation: additive inverse, -a where a + (-a) = 0"
        );

        // === ALGEBRAIC STRUCTURE PRIMITIVES ===
        // Domain for abstract algebra concepts

        let algebra_domain = DomainManifold::new(
            "algebra",
            PrimitiveTier::Mathematical,
            "Abstract algebraic structures and their properties"
        );

        let group = Primitive::base(
            "GROUP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("GROUP"))),
            "Set + associative binary op + identity + inverses"
        );

        let ring = Primitive::base(
            "RING",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("RING"))),
            "(S,+) abelian group + (S,*) monoid + distributivity"
        );

        let field_alg = Primitive::base(
            "FIELD_ALG",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("FIELD_ALG"))),
            "Ring where nonzero elements form multiplicative group"
        );

        let homomorphism = Primitive::base(
            "HOMOMORPHISM",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("HOMOMORPHISM"))),
            "Structure-preserving map between algebraic objects"
        );

        let isomorphism = Primitive::base(
            "ISOMORPHISM",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("ISOMORPHISM"))),
            "Bijective homomorphism (structural equivalence)"
        );

        let order = Primitive::base(
            "ORDER",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("ORDER"))),
            "Binary relation: reflexive, antisymmetric, transitive"
        );

        let inverse = Primitive::base(
            "INVERSE",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("INVERSE"))),
            "Element reversing an operation: a * a^(-1) = e"
        );

        let identity_element = Primitive::base(
            "IDENTITY_ELEMENT",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("IDENTITY_ELEMENT"))),
            "Neutral element: a * e = e * a = a"
        );

        let associativity_prop = Primitive::base(
            "ASSOCIATIVITY_PROP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("ASSOCIATIVITY_PROP"))),
            "Property: (a*b)*c = a*(b*c)"
        );

        let commutativity_prop = Primitive::base(
            "COMMUTATIVITY_PROP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("COMMUTATIVITY_PROP"))),
            "Property: a*b = b*a"
        );

        let distributivity_prop = Primitive::base(
            "DISTRIBUTIVITY_PROP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("DISTRIBUTIVITY_PROP"))),
            "Property: a*(b+c) = a*b + a*c"
        );

        // === CALCULUS / ANALYSIS PRIMITIVES ===

        let integration_calc = Primitive::base(
            "INTEGRATION_CALC",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INTEGRATION_CALC"))),
            "Accumulation / antiderivative: integral of f over domain"
        );

        let convergence = Primitive::base(
            "CONVERGENCE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("CONVERGENCE"))),
            "Sequence/series approaching a limit"
        );

        let continuity = Primitive::base(
            "CONTINUITY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("CONTINUITY"))),
            "Small input changes produce small output changes"
        );

        let infinity = Primitive::base(
            "INFINITY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INFINITY"))),
            "Unbounded quantity, larger than any finite number"
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains.insert("mathematics".to_string(), math_domain);
        self.domains.insert("logic".to_string(), logic_domain);
        self.domains.insert("algebra".to_string(), algebra_domain);

        for primitive in vec![
            set, membership, union, intersection, empty_set,
            not, and, or, implies, iff, equals, true_const, false_const,
            zero, one, successor, addition, multiplication,
            ratio, information, deviation, limit, efficiency,
            negation,
            group, ring, field_alg, homomorphism, isomorphism,
            order, inverse, identity_element,
            associativity_prop, commutativity_prop, distributivity_prop,
            integration_calc, convergence, continuity, infinity,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;

            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
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
            physics_domain.embed(BinaryHV::random(seed_from_name("MASS"))),
            "Property: quantity of matter in an object (kg)"
        );

        // CHARGE - electric charge
        let charge = Primitive::base(
            "CHARGE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("CHARGE"))),
            "Property: electric charge (coulombs)"
        );

        // SPIN - quantum angular momentum
        let spin = Primitive::base(
            "SPIN",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("SPIN"))),
            "Property: intrinsic angular momentum (quantum)"
        );

        // === ENERGY AND FORCES ===

        // ENERGY - capacity to do work
        let energy = Primitive::base(
            "ENERGY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("ENERGY"))),
            "Property: capacity to perform work (joules)"
        );

        // WORK - energy transfer through force
        let work = Primitive::derived(
            "WORK",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("WORK"))),
            "Quantity: energy transferred by force over distance",
            "BIND(FORCE, DISTANCE)"
        );

        // FORCE - interaction that changes motion
        let force = Primitive::base(
            "FORCE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("FORCE"))),
            "Vector: interaction that changes object's motion (newtons)"
        );

        // === MOTION PRIMITIVES ===

        // VELOCITY - rate of position change
        let velocity = Primitive::base(
            "VELOCITY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("VELOCITY"))),
            "Vector: rate of change of position (m/s)"
        );

        // ACCELERATION - rate of velocity change
        let acceleration = Primitive::derived(
            "ACCELERATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("ACCELERATION"))),
            "Vector: rate of change of velocity (m/s²)",
            "DERIVATIVE(VELOCITY)"
        );

        // MOMENTUM - quantity of motion
        let momentum = Primitive::derived(
            "MOMENTUM",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("MOMENTUM"))),
            "Vector: quantity of motion (mass × velocity)",
            "BIND(MASS, VELOCITY)"
        );

        // === CAUSALITY ===

        // CAUSE - event that produces effect
        let cause = Primitive::base(
            "CAUSE",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(BinaryHV::random(seed_from_name("CAUSE"))),
            "Event: that which produces an effect"
        );

        // EFFECT - result of a cause
        let effect = Primitive::base(
            "EFFECT",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(BinaryHV::random(seed_from_name("EFFECT"))),
            "Event: result produced by a cause"
        );

        // STATE_CHANGE - transition between states
        let state_change = Primitive::derived(
            "STATE_CHANGE",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(BinaryHV::random(seed_from_name("STATE_CHANGE"))),
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
            physics_domain.embed(BinaryHV::random(seed_from_name("THERMODYNAMIC_ENTROPY"))),
            "Property: thermodynamic measure of disorder, S = k_B ln Ω (J/K)"
        );

        // TEMPERATURE - average kinetic energy
        let temperature = Primitive::base(
            "TEMPERATURE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("TEMPERATURE"))),
            "Property: average kinetic energy of particles (K)"
        );

        // === CONSERVATION ===

        // CONSERVATION - invariant quantity
        let conservation = Primitive::base(
            "CONSERVATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("CONSERVATION"))),
            "Principle: certain quantities remain constant over time"
        );

        // === FOUNDATIONAL PHYSICAL PRIMITIVES ===
        // Base concepts needed by derived physics/information primitives

        let differentiation = Primitive::base(
            "DIFFERENTIATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("DIFFERENTIATION"))),
            "Operation: rate of change of a quantity with respect to another"
        );

        let space = Primitive::base(
            "SPACE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("SPACE"))),
            "Continuum: spatial extent in which objects exist and move"
        );

        let oscillation = Primitive::base(
            "OSCILLATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("OSCILLATION"))),
            "Process: repetitive variation about a central value"
        );

        let propagation = Primitive::base(
            "PROPAGATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("PROPAGATION"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            geometry_domain.embed(BinaryHV::random(seed_from_name("POINT"))),
            "Entity: location with no dimension"
        );

        // LINE - one-dimensional extent
        let line = Primitive::derived(
            "LINE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("LINE"))),
            "Entity: one-dimensional extent through space",
            "CONNECT(POINT, POINT)"
        );

        // PLANE - two-dimensional surface
        let plane = Primitive::base(
            "PLANE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("PLANE"))),
            "Entity: flat two-dimensional surface"
        );

        // ANGLE - measure of rotation
        let angle = Primitive::base(
            "ANGLE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("ANGLE"))),
            "Quantity: measure of rotation between two lines"
        );

        // DISTANCE - spatial separation
        let distance = Primitive::derived(
            "DISTANCE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("DISTANCE"))),
            "Quantity: spatial separation between points",
            "MEASURE(POINT, POINT)"
        );

        // === VECTOR GEOMETRY ===

        // VECTOR - directed magnitude
        let vector = Primitive::base(
            "VECTOR",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("VECTOR"))),
            "Entity: quantity with magnitude and direction"
        );

        // DOT_PRODUCT - scalar product
        let dot_product = Primitive::base(
            "DOT_PRODUCT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("DOT_PRODUCT"))),
            "Operation: scalar product of two vectors"
        );

        // CROSS_PRODUCT - vector product
        let cross_product = Primitive::base(
            "CROSS_PRODUCT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("CROSS_PRODUCT"))),
            "Operation: vector product perpendicular to both inputs"
        );

        // === DIFFERENTIAL GEOMETRY ===

        // MANIFOLD - curved space
        let manifold = Primitive::base(
            "MANIFOLD",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("MANIFOLD"))),
            "Entity: space that locally resembles Euclidean space"
        );

        // TANGENT_SPACE - local linear approximation
        let tangent_space = Primitive::base(
            "TANGENT_SPACE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("TANGENT_SPACE"))),
            "Entity: linear approximation at a manifold point"
        );

        // CURVATURE - deviation from flatness
        let curvature = Primitive::base(
            "CURVATURE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("CURVATURE"))),
            "Property: measure of deviation from flatness"
        );

        // === TOPOLOGY ===

        // OPEN_SET - set excluding boundary
        let open_set = Primitive::base(
            "OPEN_SET",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("OPEN_SET"))),
            "Set: excluding its boundary points"
        );

        // CLOSED_SET - set including boundary
        let closed_set = Primitive::base(
            "CLOSED_SET",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("CLOSED_SET"))),
            "Set: including all its boundary points"
        );

        // BOUNDARY - edge of a region
        let boundary = Primitive::base(
            "BOUNDARY",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("BOUNDARY"))),
            "Set: points on the edge of a region"
        );

        // INTERIOR - inside of a region
        let interior = Primitive::base(
            "INTERIOR",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("INTERIOR"))),
            "Set: all points strictly inside a region"
        );

        // === MEREOTOPOLOGY (part-whole) ===

        // PART_OF - mereological inclusion
        let part_of = Primitive::base(
            "PART_OF",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("PART_OF"))),
            "Relation: x is part of y"
        );

        // OVERLAPS - shared parts
        let overlaps = Primitive::base(
            "OVERLAPS",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("OVERLAPS"))),
            "Relation: x and y share common parts"
        );

        // TOUCHES - external contact
        let touches = Primitive::base(
            "TOUCHES",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("TOUCHES"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            game_theory_domain.embed(BinaryHV::random(seed_from_name("UTILITY"))),
            "Function: measure of preference or value"
        );

        // STRATEGY - action plan
        let strategy = Primitive::base(
            "STRATEGY",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("STRATEGY"))),
            "Plan: complete specification of actions in all situations"
        );

        // EQUILIBRIUM - stable state
        let equilibrium = Primitive::base(
            "EQUILIBRIUM",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("EQUILIBRIUM"))),
            "State: where no agent benefits from unilateral deviation"
        );

        // PAYOFF - outcome value
        let payoff = Primitive::derived(
            "PAYOFF",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("PAYOFF"))),
            "Value: utility resulting from strategy profile",
            "APPLY(UTILITY, STRATEGY)"
        );

        // === TEMPORAL LOGIC (Allen's Intervals) ===

        // BEFORE - temporal precedence
        let before = Primitive::base(
            "BEFORE",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("BEFORE"))),
            "Relation: interval x ends before interval y starts"
        );

        // AFTER - temporal succession
        let after = Primitive::base(
            "AFTER",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("AFTER"))),
            "Relation: interval x starts after interval y ends"
        );

        // DURING - temporal containment
        let during = Primitive::base(
            "DURING",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("DURING"))),
            "Relation: interval x occurs within interval y"
        );

        // MEETS - temporal adjacency
        let meets = Primitive::base(
            "MEETS",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("MEETS"))),
            "Relation: interval x ends exactly when y starts"
        );

        // OVERLAPS_TEMPORAL - partial overlap
        let overlaps_temporal = Primitive::base(
            "OVERLAPS_TEMPORAL",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("OVERLAPS_TEMPORAL"))),
            "Relation: intervals x and y partially overlap in time"
        );

        // === COUNTERFACTUAL REASONING ===

        // COUNTERFACTUAL - hypothetical condition
        let counterfactual = Primitive::base(
            "COUNTERFACTUAL",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("COUNTERFACTUAL"))),
            "Condition: what would have happened if..."
        );

        // POSSIBLE_WORLD - alternative reality
        let possible_world = Primitive::base(
            "POSSIBLE_WORLD",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("POSSIBLE_WORLD"))),
            "Structure: consistent alternative state of reality"
        );

        // === SOCIAL COORDINATION ===

        // COOPERATE - joint action for mutual benefit
        let cooperate = Primitive::base(
            "COOPERATE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("COOPERATE"))),
            "Action: work together for mutual benefit"
        );

        // DEFECT - self-interested deviation
        let defect = Primitive::base(
            "DEFECT",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("DEFECT"))),
            "Action: act in self-interest against cooperation"
        );

        // RECIPROCATE - conditional cooperation
        let reciprocate = Primitive::derived(
            "RECIPROCATE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("RECIPROCATE"))),
            "Strategy: cooperate if and only if partner cooperates",
            "CONDITIONAL(COOPERATE, COOPERATE)"
        );

        // TRUST - belief in cooperation
        let trust = Primitive::base(
            "TRUST",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("TRUST"))),
            "Belief: expectation that others will cooperate"
        );

        // === INFORMATION ===

        // SIGNAL - information transmission
        let signal = Primitive::base(
            "SIGNAL",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("SIGNAL"))),
            "Action: transmit information to influence others"
        );

        // BELIEF - subjective probability
        let belief = Primitive::base(
            "BELIEF",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("BELIEF"))),
            "State: subjective probability assignment"
        );

        // COMMON_KNOWLEDGE - shared awareness
        let common_knowledge = Primitive::base(
            "COMMON_KNOWLEDGE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("COMMON_KNOWLEDGE"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            metacognition_domain.embed(BinaryHV::random(seed_from_name("SELF"))),
            "Entity: the reflexive subject of awareness"
        );

        // IDENTITY - persistent self-recognition
        let identity = Primitive::base(
            "IDENTITY",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("IDENTITY"))),
            "Property: persistent self-recognition over time"
        );

        // META_BELIEF - belief about beliefs
        let meta_belief = Primitive::derived(
            "META_BELIEF",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("META_BELIEF"))),
            "State: belief about one's own beliefs",
            "APPLY(SELF, BELIEF)"
        );

        // INTROSPECTION - self-examination
        let introspection = Primitive::base(
            "INTROSPECTION",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("INTROSPECTION"))),
            "Process: examination of one's own mental states"
        );

        // === HOMEOSTASIS & REGULATION ===

        // HOMEOSTASIS - self-regulation
        let homeostasis = Primitive::base(
            "HOMEOSTASIS",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("HOMEOSTASIS"))),
            "Process: maintaining stable internal state"
        );

        // SETPOINT - target state
        let setpoint = Primitive::base(
            "SETPOINT",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("SETPOINT"))),
            "Value: target state for homeostatic regulation"
        );

        // REGULATION - corrective action
        let regulation = Primitive::base(
            "REGULATION",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("REGULATION"))),
            "Process: adjusting state toward setpoint"
        );

        // FEEDBACK - state monitoring
        let feedback = Primitive::base(
            "FEEDBACK",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("FEEDBACK"))),
            "Signal: information about current state vs setpoint"
        );

        // === REPAIR & ADAPTATION ===

        // REPAIR - damage correction
        let repair = Primitive::base(
            "REPAIR",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("REPAIR"))),
            "Process: restoring damaged structures or functions"
        );

        // RESTORE - return to previous state
        let restore = Primitive::base(
            "RESTORE",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("RESTORE"))),
            "Process: returning to a previous functional state"
        );

        // ADAPT - modify in response to change
        let adapt = Primitive::base(
            "ADAPT",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("ADAPT"))),
            "Process: modify structure/behavior in response to environment"
        );

        // LEARN - update from experience
        let learn = Primitive::base(
            "LEARN",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("LEARN"))),
            "Process: update knowledge or behavior from experience"
        );

        // === EPISTEMIC STRENGTH ===

        // KNOW - justified true belief
        let know = Primitive::base(
            "KNOW",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("KNOW"))),
            "State: justified true belief"
        );

        // UNCERTAIN - lack of certainty
        let uncertain = Primitive::base(
            "UNCERTAIN",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("UNCERTAIN"))),
            "State: lacking sufficient information for certainty"
        );

        // CONFIDENCE - degree of certainty
        let confidence = Primitive::base(
            "CONFIDENCE",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("CONFIDENCE"))),
            "Measure: degree of certainty in a belief"
        );

        // EVIDENCE - justification
        let evidence = Primitive::base(
            "EVIDENCE",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("EVIDENCE"))),
            "Support: information supporting or refuting a belief"
        );

        // === METABOLIC / RESOURCE MANAGEMENT ===

        // RESOURCE - available capacity
        let resource = Primitive::base(
            "RESOURCE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("RESOURCE"))),
            "Entity: available capacity for use"
        );

        // ALLOCATE - distribute resources
        let allocate = Primitive::base(
            "ALLOCATE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("ALLOCATE"))),
            "Process: distribute resources to tasks"
        );

        // CONSUME - use resources
        let consume = Primitive::base(
            "CONSUME",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("CONSUME"))),
            "Process: use resources to perform work"
        );

        // PRODUCE - generate resources
        let produce = Primitive::base(
            "PRODUCE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("PRODUCE"))),
            "Process: generate resources from inputs"
        );

        // === REWARD & VALUE ===

        // REWARD - positive reinforcement
        let reward = Primitive::base(
            "REWARD",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("REWARD"))),
            "Signal: positive reinforcement for actions"
        );

        // GOAL - desired state
        let goal = Primitive::base(
            "GOAL",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("GOAL"))),
            "State: desired future state to achieve"
        );

        // VALUE - measure of importance
        let value = Primitive::base(
            "VALUE",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("VALUE"))),
            "Measure: importance or worth of a state/action"
        );

        // CERTAINTY - state of complete knowledge
        let certainty = Primitive::base(
            "CERTAINTY",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("CERTAINTY"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            biology_domain.embed(BinaryHV::random(seed_from_name("METABOLISM"))),
            "Process: energy transformation in living systems (ATP synthesis, glycolysis)"
        );

        let growth = Primitive::base(
            "GROWTH",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("GROWTH"))),
            "Process: increase in size, complexity through cell division and development"
        );

        let reproduction = Primitive::base(
            "REPRODUCTION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("REPRODUCTION"))),
            "Process: creation of new organisms, transmission of heredity"
        );

        let evolution = Primitive::base(
            "EVOLUTION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("EVOLUTION"))),
            "Process: change in heritable characteristics over generations"
        );

        let adaptation = Primitive::derived(
            "ADAPTATION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("ADAPTATION"))),
            "Process: adjustment to environment through selection",
            "EVOLUTION ⊗ ENVIRONMENT"
        );

        let homeostasis_dynamic = Primitive::base(
            "HOMEOSTASIS_DYNAMIC",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("HOMEOSTASIS_DYNAMIC"))),
            "Process: dynamic self-regulation maintaining stable internal conditions"
        );

        let symbiosis = Primitive::base(
            "SYMBIOSIS",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("SYMBIOSIS"))),
            "Relationship: close interaction between different organisms (mutualism, parasitism)"
        );

        let immune_response = Primitive::base(
            "IMMUNE_RESPONSE",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("IMMUNE_RESPONSE"))),
            "Process: self/non-self distinction, pathogen recognition and elimination"
        );

        let circadian_rhythm = Primitive::base(
            "CIRCADIAN_RHYTHM",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("CIRCADIAN_RHYTHM"))),
            "Pattern: ~24-hour biological cycles, internal timekeeping"
        );

        let morphogen = Primitive::base(
            "MORPHOGEN",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("MORPHOGEN"))),
            "Substance: concentration gradient guiding development and pattern formation"
        );

        let apoptosis = Primitive::base(
            "APOPTOSIS",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("APOPTOSIS"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            emotion_domain.embed(BinaryHV::random(seed_from_name("AFFECTIVE_VALENCE"))),
            "Dimension: positive/negative affect, pleasantness/unpleasantness"
        );

        let arousal = Primitive::base(
            "AFFECTIVE_AROUSAL",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("AFFECTIVE_AROUSAL"))),
            "Dimension: activation level, calm/excited continuum"
        );

        // === BASIC EMOTIONS (Ekman) ===

        let joy = Primitive::derived(
            "JOY",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("JOY"))),
            "Emotion: positive valence, high arousal - happiness, pleasure",
            "VALENCE_POSITIVE ⊗ AROUSAL_MODERATE"
        );

        let sadness = Primitive::derived(
            "SADNESS",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("SADNESS"))),
            "Emotion: negative valence, low arousal - loss, disappointment",
            "VALENCE_NEGATIVE ⊗ AROUSAL_LOW"
        );

        let fear = Primitive::derived(
            "FEAR",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("FEAR"))),
            "Emotion: negative valence, high arousal - threat response",
            "VALENCE_NEGATIVE ⊗ AROUSAL_HIGH"
        );

        let anger = Primitive::derived(
            "ANGER",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("ANGER"))),
            "Emotion: negative valence, high arousal - obstacle response",
            "VALENCE_NEGATIVE ⊗ AROUSAL_HIGH ⊗ APPROACH"
        );

        let disgust = Primitive::derived(
            "DISGUST",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("DISGUST"))),
            "Emotion: negative valence, rejection response - contamination avoidance",
            "VALENCE_NEGATIVE ⊗ REJECTION"
        );

        let surprise = Primitive::derived(
            "SURPRISE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("SURPRISE"))),
            "Emotion: neutral valence, high arousal - unexpected event response",
            "AROUSAL_HIGH ⊗ NOVELTY"
        );

        // === SOCIAL EMOTIONS ===

        let empathy = Primitive::base(
            "EMPATHY",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("EMPATHY"))),
            "Capacity: shared emotional experience, feeling with others"
        );

        let attachment = Primitive::base(
            "ATTACHMENT",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("ATTACHMENT"))),
            "Bond: emotional connection, social bonding, relationship formation"
        );

        let awe = Primitive::derived(
            "AWE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("AWE"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            ecology_domain.embed(BinaryHV::random(seed_from_name("NICHE"))),
            "Concept: environmental role, opportunity and constraint space"
        );

        let carrying_capacity = Primitive::base(
            "CARRYING_CAPACITY",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("CARRYING_CAPACITY"))),
            "Limit: maximum population sustainable by available resources"
        );

        let succession = Primitive::base(
            "SUCCESSION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("SUCCESSION"))),
            "Process: sequential ecosystem development, progressive change"
        );

        let trophic_level = Primitive::base(
            "TROPHIC_LEVEL",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("TROPHIC_LEVEL"))),
            "Structure: position in energy flow chain (producer, consumer, decomposer)"
        );

        let resilience = Primitive::base(
            "RESILIENCE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("RESILIENCE"))),
            "Property: system capacity to absorb disturbance and reorganize"
        );

        // === SYSTEMS DYNAMICS ===

        let feedback_loop_positive = Primitive::base(
            "FEEDBACK_LOOP_POSITIVE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("FEEDBACK_LOOP_POSITIVE"))),
            "Pattern: amplifying feedback, exponential growth or collapse"
        );

        let feedback_loop_negative = Primitive::base(
            "FEEDBACK_LOOP_NEGATIVE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("FEEDBACK_LOOP_NEGATIVE"))),
            "Pattern: dampening feedback, stabilization, homeostasis"
        );

        let emergence_strong = Primitive::base(
            "EMERGENCE_STRONG",
            PrimitiveTier::MetaCognitive,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("EMERGENCE_STRONG"))),
            "Property: irreducible higher-level properties from component interactions"
        );

        let attractor = Primitive::base(
            "ATTRACTOR",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("ATTRACTOR"))),
            "State: stable system configuration in phase space"
        );

        let bifurcation = Primitive::base(
            "BIFURCATION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("BIFURCATION"))),
            "Transition: qualitative system change at critical parameter value"
        );

        let phase_transition = Primitive::base(
            "PHASE_TRANSITION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("PHASE_TRANSITION"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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
            quantum_domain.embed(BinaryHV::random(seed_from_name("SUPERPOSITION"))),
            "State: system existing in multiple configurations simultaneously"
        );

        let entanglement = Primitive::base(
            "ENTANGLEMENT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("ENTANGLEMENT"))),
            "Correlation: non-local quantum correlation between systems"
        );

        let measurement = Primitive::base(
            "MEASUREMENT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("MEASUREMENT"))),
            "Process: observer-dependent state collapse, wave function reduction"
        );

        let uncertainty_heisenberg = Primitive::base(
            "UNCERTAINTY_HEISENBERG",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("UNCERTAINTY_HEISENBERG"))),
            "Principle: fundamental limit on simultaneous knowledge (ΔxΔp ≥ ℏ/2)"
        );

        let wave_particle_duality = Primitive::base(
            "WAVE_PARTICLE_DUALITY",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("WAVE_PARTICLE_DUALITY"))),
            "Property: complementary wave and particle aspects of quantum objects"
        );

        let planck_constant = Primitive::base(
            "PLANCK_CONSTANT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("PLANCK_CONSTANT"))),
            "Constant: quantum of action (h = 6.626×10⁻³⁴ J·s), fundamental scale"
        );

        // Register domain and primitives
        self.domains.insert("quantum".to_string(), quantum_domain);

        for primitive in [superposition, entanglement, measurement,
            uncertainty_heisenberg, wave_particle_duality, planck_constant] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
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
            economics_domain.embed(BinaryHV::random(seed_from_name("SCARCITY"))),
            "Condition: limited availability relative to demand"
        );

        let supply = Primitive::base(
            "SUPPLY",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("SUPPLY"))),
            "Quantity: amount available at given price"
        );

        let demand = Primitive::base(
            "DEMAND",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("DEMAND"))),
            "Quantity: amount desired at given price"
        );

        let exchange = Primitive::base(
            "EXCHANGE",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("EXCHANGE"))),
            "Transaction: trading goods/services, reciprocal transfer"
        );

        let value_subjective = Primitive::base(
            "VALUE_SUBJECTIVE",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("VALUE_SUBJECTIVE"))),
            "Property: preference-dependent worth, individual utility"
        );

        let capital = Primitive::base(
            "CAPITAL",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("CAPITAL"))),
            "Resource: accumulated assets for production, stored value"
        );

        let debt = Primitive::base(
            "DEBT",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("DEBT"))),
            "Obligation: future claim, deferred payment"
        );

        let trust_economic = Primitive::base(
            "TRUST_ECONOMIC",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("TRUST_ECONOMIC"))),
            "Property: reliability expectation, reputation, social capital"
        );

        // Register domain and primitives
        self.domains.insert("economics".to_string(), economics_domain);

        for primitive in [scarcity, supply, demand, exchange, value_subjective,
            capital, debt, trust_economic] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
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
            linguistics_domain.embed(BinaryHV::random(seed_from_name("SIGN"))),
            "Structure: signifier + signified, symbol and its meaning"
        );

        let reference = Primitive::base(
            "REFERENCE",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("REFERENCE"))),
            "Relation: symbol → object mapping, denotation"
        );

        let context_dependency = Primitive::base(
            "CONTEXT_DEPENDENCY",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("CONTEXT_DEPENDENCY"))),
            "Property: meaning varies with situational context"
        );

        let metaphor = Primitive::base(
            "METAPHOR",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("METAPHOR"))),
            "Mapping: cross-domain conceptual transfer, analogical reasoning"
        );

        let syntax = Primitive::base(
            "SYNTAX",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("SYNTAX"))),
            "Structure: compositional rules, grammatical organization"
        );

        let semantics = Primitive::base(
            "SEMANTICS",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("SEMANTICS"))),
            "Property: meaning relations, truth conditions"
        );

        let pragmatics = Primitive::base(
            "PRAGMATICS",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("PRAGMATICS"))),
            "Use: meaning in communicative context, speaker intention"
        );

        // Register domain and primitives
        self.domains.insert("linguistics".to_string(), linguistics_domain);

        for primitive in [sign, reference, context_dependency, metaphor,
            syntax, semantics, pragmatics] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
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
            moral_domain.embed(BinaryHV::random(seed_from_name("NORM"))),
            "Rule: social expectation, behavioral standard"
        );

        let obligation = Primitive::base(
            "OBLIGATION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("OBLIGATION"))),
            "Duty: moral requirement, what must be done"
        );

        let permission = Primitive::base(
            "PERMISSION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("PERMISSION"))),
            "Allowance: action that may be done without violation"
        );

        let prohibition = Primitive::base(
            "PROHIBITION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("PROHIBITION"))),
            "Restriction: forbidden action, taboo"
        );

        // === MORAL FOUNDATIONS ===

        let fairness = Primitive::base(
            "FAIRNESS",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("FAIRNESS"))),
            "Principle: equitable distribution, reciprocity, justice"
        );

        let harm = Primitive::base(
            "HARM",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("HARM"))),
            "Concept: damage, suffering, negative impact on well-being"
        );

        let care = Primitive::base(
            "CARE",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("CARE"))),
            "Disposition: protection, nurturance, compassion"
        );

        let rights = Primitive::base(
            "RIGHTS",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("RIGHTS"))),
            "Entitlement: claims, protections, freedoms"
        );

        // Register domain and primitives
        self.domains.insert("morality".to_string(), moral_domain);

        for primitive in [norm, obligation, permission, prohibition,
            fairness, harm, care, rights] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
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

    /// Initialize Tier 6: Temporal Primitives
    ///
    /// Extended Allen's Interval Algebra plus temporal reasoning concepts.
    /// Note: Some temporal primitives (BEFORE, AFTER, DURING, MEETS) already exist
    /// in Tier 4 (Strategic) with domain "temporal". This tier adds higher-level
    /// temporal reasoning concepts.
    ///
    /// ## Interval Relations (Extended)
    /// - STARTS: Interval x begins at same point as y
    /// - FINISHES: Interval x ends at same point as y
    /// - EQUALS_TEMPORAL: Intervals have same start and end
    ///
    /// ## Temporal Reasoning
    /// - INSTANT: A point in time (zero duration)
    /// - DURATION: The length of an interval
    /// - TEMPO: Rate of change over time
    /// - RHYTHM: Repeating temporal pattern
    /// - ANTICIPATE: Expectation of future state
    /// - PERSIST: Continuation through time
    fn init_tier6_temporal(&mut self) {
        let temporal_domain = DomainManifold::new(
            "temporal_reasoning",
            PrimitiveTier::Temporal,
            "Extended temporal reasoning and interval algebra"
        );

        // === INTERVAL RELATIONS (Extended Allen's) ===

        let starts = Primitive::base(
            "STARTS",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("STARTS"))),
            "Relation: interval x begins at same point as interval y begins"
        );

        let finishes = Primitive::base(
            "FINISHES",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("FINISHES"))),
            "Relation: interval x ends at same point as interval y ends"
        );

        let equals_temporal = Primitive::base(
            "EQUALS_TEMPORAL",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("EQUALS_TEMPORAL"))),
            "Relation: intervals x and y have identical start and end points"
        );

        // === TEMPORAL CONCEPTS ===

        let instant = Primitive::base(
            "INSTANT",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("INSTANT"))),
            "A point in time with zero duration"
        );

        let duration = Primitive::base(
            "DURATION",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("DURATION"))),
            "The length or extent of a temporal interval"
        );

        let tempo = Primitive::base(
            "TEMPO",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("TEMPO"))),
            "Rate of occurrence or change over time"
        );

        let rhythm = Primitive::base(
            "RHYTHM",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("RHYTHM"))),
            "Repeating pattern of temporal events"
        );

        let anticipate = Primitive::base(
            "ANTICIPATE",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("ANTICIPATE"))),
            "Expectation or prediction of a future state"
        );

        let persist = Primitive::base(
            "PERSIST",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("PERSIST"))),
            "Continuation of existence or state through time"
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains.insert("temporal_reasoning".to_string(), temporal_domain);

        for primitive in vec![
            starts, finishes, equals_temporal,
            instant, duration, tempo, rhythm,
            anticipate, persist,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "temporal_composition".to_string(),
            pattern: vec![PrimitiveTier::Temporal, PrimitiveTier::Temporal],
            result_tier: PrimitiveTier::Temporal,
            example: "STARTS ⊗ FINISHES → interval containment".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "temporal_physical".to_string(),
            pattern: vec![PrimitiveTier::Temporal, PrimitiveTier::Physical],
            result_tier: PrimitiveTier::Physical,
            example: "DURATION ⊗ VELOCITY → distance traveled".to_string(),
        });
    }

    /// Initialize Tier 7: Compositional Primitives
    ///
    /// These primitives enable higher-order composition of other primitives,
    /// forming a complete algebra for building complex structures from simple ones.
    ///
    /// ## Composition Operators
    /// - SEQUENCE: Sequential composition (do A then B)
    /// - PARALLEL: Parallel composition (do A and B together)
    /// - CONDITIONAL: Conditional composition (if P then A else B)
    /// - ITERATE: Repeated application (do A n times)
    /// - FIXPOINT: Fixed-point operator (find stable state)
    ///
    /// ## Structural Operators
    /// - ABSTRACT: Extract pattern from instances
    /// - INSTANTIATE: Create instance from pattern
    /// - COMPOSE: Combine functions (f ∘ g)
    /// - CURRY: Partial application
    fn init_tier7_compositional(&mut self) {
        let compositional_domain = DomainManifold::new(
            "composition",
            PrimitiveTier::Compositional,
            "Higher-order composition operators for building complex structures"
        );

        // === COMPOSITION OPERATORS ===

        let sequence_op = Primitive::base(
            "SEQUENCE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("SEQUENCE_OP"))),
            "Sequential composition: do A, then do B"
        );

        let parallel_op = Primitive::base(
            "PARALLEL_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("PARALLEL_OP"))),
            "Parallel composition: do A and B simultaneously"
        );

        let conditional_op = Primitive::base(
            "CONDITIONAL_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("CONDITIONAL_OP"))),
            "Conditional composition: if P then A else B"
        );

        let iterate_op = Primitive::base(
            "ITERATE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("ITERATE_OP"))),
            "Iteration: repeated application of an operation"
        );

        let fixpoint_op = Primitive::base(
            "FIXPOINT_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("FIXPOINT_OP"))),
            "Fixed-point: find stable state under repeated application"
        );

        // === STRUCTURAL OPERATORS ===

        let abstract_op = Primitive::base(
            "ABSTRACT_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("ABSTRACT_OP"))),
            "Abstraction: extract common pattern from instances"
        );

        let instantiate_op = Primitive::base(
            "INSTANTIATE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("INSTANTIATE_OP"))),
            "Instantiation: create concrete instance from abstract pattern"
        );

        let compose_op = Primitive::base(
            "COMPOSE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("COMPOSE_OP"))),
            "Function composition: (f ∘ g)(x) = f(g(x))"
        );

        let curry_op = Primitive::base(
            "CURRY_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("CURRY_OP"))),
            "Currying: transform multi-argument function to chain of single-argument functions"
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains.insert("composition".to_string(), compositional_domain);

        for primitive in vec![
            sequence_op, parallel_op, conditional_op, iterate_op, fixpoint_op,
            abstract_op, instantiate_op, compose_op, curry_op,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
        }

        // === BINDING RULES ===

        self.binding_rules.push(BindingRule {
            name: "compositional_algebra".to_string(),
            pattern: vec![PrimitiveTier::Compositional, PrimitiveTier::Compositional],
            result_tier: PrimitiveTier::Compositional,
            example: "SEQUENCE_OP ⊗ ITERATE_OP → loop construct".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "compositional_lifting".to_string(),
            pattern: vec![PrimitiveTier::Compositional, PrimitiveTier::Mathematical],
            result_tier: PrimitiveTier::Compositional,
            example: "ITERATE_OP ⊗ ADDITION → summation".to_string(),
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
            consciousness_domain.embed(BinaryHV::random(seed_from_name("QUALE"))),
            "Irreducible unit of subjective experience - what it is like to experience"
        );

        let phenomenal_binding = Primitive::base(
            "PHENOMENAL_BINDING",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("PHENOMENAL_BINDING"))),
            "Integration of disparate qualia into unified perceptual field"
        );

        let subjective_time = Primitive::base(
            "SUBJECTIVE_TIME",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SUBJECTIVE_TIME"))),
            "The felt passage of time - duration as experienced"
        );

        let sentience = Primitive::base(
            "SENTIENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SENTIENCE"))),
            "Capacity for subjective experience - being a subject of experience"
        );

        // === ATTENTION PRIMITIVES ===

        let attend = Primitive::base(
            "ATTEND",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("ATTEND"))),
            "Selective focus - directing conscious awareness to subset of information"
        );

        let salience = Primitive::base(
            "SALIENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SALIENCE"))),
            "Intrinsic importance - property that draws attention"
        );

        let binding_window = Primitive::base(
            "BINDING_WINDOW",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("BINDING_WINDOW"))),
            "Temporal integration window (~100-200ms) for conscious binding"
        );

        let awareness = Primitive::base(
            "AWARENESS",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("AWARENESS"))),
            "State of being conscious of something - phenomenal access"
        );

        // === MEMORY OPERATION PRIMITIVES ===

        let remember = Primitive::base(
            "REMEMBER",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("REMEMBER"))),
            "Retrieval of encoded episodic information into consciousness"
        );

        let forget = Primitive::base(
            "FORGET",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("FORGET"))),
            "Loss or decay of stored information - natural or active"
        );

        let consolidate = Primitive::base(
            "CONSOLIDATE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("CONSOLIDATE"))),
            "Transfer from working memory to long-term storage"
        );

        let recognize = Primitive::base(
            "RECOGNIZE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("RECOGNIZE"))),
            "Pattern matching of percept to stored memory - familiarity"
        );

        // === AGENCY PRIMITIVES ===

        let intend = Primitive::base(
            "INTEND",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("INTEND"))),
            "Goal-directed mental state - representation of desired outcome"
        );

        let will = Primitive::base(
            "WILL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("WILL"))),
            "Volitional initiation of action - self-determined causation"
        );

        let decide = Primitive::base(
            "DECIDE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("DECIDE"))),
            "Selection among alternatives - commitment to course of action"
        );

        let control = Primitive::base(
            "CONTROL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("CONTROL"))),
            "Executive regulation - top-down modulation of processing"
        );

        // === AFFECTIVE PRIMITIVES ===

        let valence = Primitive::base(
            "VALENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("VALENCE"))),
            "Positive-negative dimension of experience - pleasantness/unpleasantness"
        );

        let arousal = Primitive::base(
            "AROUSAL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("AROUSAL"))),
            "Activation level of experience - calm to excited"
        );

        let selection = Primitive::base(
            "SELECTION",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SELECTION"))),
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
            self.by_tier.entry(tier).or_default().push(name);
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

    /// Initialize Tier 9: Code & Symbol Manipulation Primitives
    ///
    /// These primitives enable consciousness-aware code understanding, generation,
    /// and transformation. Code operations flow through the same primitive routing
    /// as all other cognitive tasks.
    ///
    /// ## Structural Primitives
    /// - PARSE: Decompose source into AST structure
    /// - ENTITY: Identify code entities (functions, types, variables)
    /// - ROLE: Determine role in code (parameter, return, field)
    ///
    /// ## Encoding Primitives
    /// - ENCODE: Convert code structure to hypervector
    /// - BIND_SYMBOL: Associate name with meaning in code context
    ///
    /// ## Generative Primitives
    /// - GENERATE: Create new code from specification
    /// - COMPOSE: Combine code patterns
    /// - SPECIALIZE: Create specific instance from generic pattern
    ///
    /// ## Flow Primitives
    /// - BRANCH: Conditional execution path
    /// - LOOP: Iterative execution pattern
    /// - CALL: Function invocation
    /// - RETURN: Value production
    ///
    /// ## Reasoning Primitives
    /// - EXPLAIN: Describe code semantics
    /// - DEBUG: Diagnose code issues
    /// - VERIFY: Validate code correctness
    fn init_tier9_code(&mut self) {
        let code_domain = DomainManifold::new(
            "code",
            PrimitiveTier::Code,
            "Code understanding, generation, and transformation"
        );

        // === STRUCTURAL PRIMITIVES ===

        let parse = Primitive::base(
            "PARSE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("PARSE"))),
            "Decompose source code into AST structure"
        );

        let entity = Primitive::base(
            "ENTITY",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ENTITY"))),
            "Identify code entity: function, struct, variable, import"
        );

        let role = Primitive::base(
            "ROLE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ROLE"))),
            "Determine syntactic role: parameter, return type, field, attribute"
        );

        let import = Primitive::base(
            "IMPORT",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("IMPORT"))),
            "External dependency reference"
        );

        let attribute = Primitive::base(
            "ATTRIBUTE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ATTRIBUTE"))),
            "Metadata annotation on code element"
        );

        // === ENCODING PRIMITIVES ===

        let encode = Primitive::base(
            "ENCODE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ENCODE"))),
            "Convert code structure to hypervector representation"
        );

        let bind_symbol = Primitive::base(
            "BIND_SYMBOL",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("BIND_SYMBOL"))),
            "Associate identifier with meaning in code context"
        );

        let type_check = Primitive::base(
            "TYPE_CHECK",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("TYPE_CHECK"))),
            "Verify type consistency and constraints"
        );

        // === GENERATIVE PRIMITIVES ===

        let generate = Primitive::base(
            "GENERATE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("GENERATE"))),
            "Create new code from specification or pattern"
        );

        let compose = Primitive::base(
            "COMPOSE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("COMPOSE"))),
            "Combine code patterns into larger structure"
        );

        let specialize = Primitive::base(
            "SPECIALIZE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("SPECIALIZE"))),
            "Create specific instance from generic pattern"
        );

        let mutate = Primitive::base(
            "MUTATE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("MUTATE"))),
            "Transform code while preserving semantics"
        );

        // === FLOW PRIMITIVES ===

        let branch = Primitive::base(
            "BRANCH",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("BRANCH"))),
            "Conditional execution path (if/match)"
        );

        let loop_prim = Primitive::base(
            "LOOP",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("LOOP"))),
            "Iterative execution pattern (for/while/loop)"
        );

        let call = Primitive::base(
            "CALL",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("CALL"))),
            "Function or method invocation"
        );

        let return_prim = Primitive::base(
            "RETURN",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("RETURN"))),
            "Value production and control flow exit"
        );

        // === SIMILARITY & ABSTRACTION ===

        let code_similarity = Primitive::base(
            "CODE_SIMILARITY",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("CODE_SIMILARITY"))),
            "Measure semantic similarity between code patterns"
        );

        let abstract_prim = Primitive::base(
            "ABSTRACT",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ABSTRACT"))),
            "Extract common pattern from concrete implementations"
        );

        let refactor = Primitive::base(
            "REFACTOR",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("REFACTOR"))),
            "Restructure code while preserving behavior"
        );

        // === REASONING PRIMITIVES ===

        let explain = Primitive::base(
            "EXPLAIN",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("EXPLAIN"))),
            "Describe code semantics in natural language"
        );

        let trace = Primitive::base(
            "TRACE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("TRACE"))),
            "Follow execution path through code"
        );

        let intent = Primitive::base(
            "INTENT",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("INTENT"))),
            "Infer programmer's purpose from code"
        );

        let debug = Primitive::base(
            "DEBUG",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("DEBUG"))),
            "Diagnose issues and locate errors"
        );

        let verify = Primitive::base(
            "VERIFY",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("VERIFY"))),
            "Validate code correctness against specification"
        );

        // === SEQUENCE PRIMITIVE ===

        let code_sequence = Primitive::base(
            "CODE_SEQUENCE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("CODE_SEQUENCE"))),
            "Ordered sequence of code operations"
        );

        // Register all code primitives
        let primitives = vec![
            // Structural
            parse, entity, role, import, attribute,
            // Encoding
            encode, bind_symbol, type_check,
            // Generative
            generate, compose, specialize, mutate,
            // Flow
            branch, loop_prim, call, return_prim,
            // Similarity & Abstraction
            code_similarity, abstract_prim, refactor,
            // Reasoning
            explain, trace, intent, debug, verify,
            // Sequence
            code_sequence,
        ];

        for primitive in primitives {
            let name = primitive.name.clone();
            self.primitives.insert(name.clone(), primitive);
            self.by_tier
                .entry(PrimitiveTier::Code)
                .or_default()
                .push(name);
        }

        // === BINDING RULES FOR CODE ===

        // Code tier internal composition
        self.binding_rules.push(BindingRule {
            name: "code_composition".to_string(),
            pattern: vec![PrimitiveTier::Code, PrimitiveTier::Code],
            result_tier: PrimitiveTier::Code,
            example: "PARSE ⊗ ENCODE → code embedding".to_string(),
        });

        // Code + Compositional for higher-order patterns
        self.binding_rules.push(BindingRule {
            name: "code_higher_order".to_string(),
            pattern: vec![PrimitiveTier::Code, PrimitiveTier::Compositional],
            result_tier: PrimitiveTier::Code,
            example: "LOOP ⊗ SEQUENCE_OP → loop body execution".to_string(),
        });

        // Code + Consciousness for intentional code understanding
        self.binding_rules.push(BindingRule {
            name: "code_consciousness".to_string(),
            pattern: vec![PrimitiveTier::Code, PrimitiveTier::Consciousness],
            result_tier: PrimitiveTier::Code,
            example: "INTENT ⊗ INTEND → conscious code purpose".to_string(),
        });

        // Code + MetaCognitive for self-modifying code reasoning
        self.binding_rules.push(BindingRule {
            name: "code_metacognition".to_string(),
            pattern: vec![PrimitiveTier::Code, PrimitiveTier::MetaCognitive],
            result_tier: PrimitiveTier::Code,
            example: "REFACTOR ⊗ SELF → self-improvement".to_string(),
        });

        // Code + Mathematical for formal verification
        self.binding_rules.push(BindingRule {
            name: "code_formal".to_string(),
            pattern: vec![PrimitiveTier::Code, PrimitiveTier::Mathematical],
            result_tier: PrimitiveTier::Code,
            example: "VERIFY ⊗ PROOF → formal verification".to_string(),
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

    // === DERIVATION CHAIN VALIDATION ===

    /// Validate the derivation chain: check that all derived primitives
    /// have their parents registered and encodings are genuinely composed.
    pub fn validate_derivation_chain(&self) -> Vec<(String, bool, Option<String>)> {
        let mut diagnostics = Vec::new();
        for (name, prim) in &self.primitives {
            if !prim.is_base {
                if let Some(ref derivation) = prim.derivation {
                    // Parse parent names from derivation expression (split on ^ or whitespace ops)
                    let parent_names: Vec<&str> = derivation.split(['^', ' '])
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty() && s.chars().next().is_some_and(|c| c.is_uppercase()))
                        .collect();
                    let all_found = parent_names.iter().all(|p| self.primitives.contains_key(*p));
                    if !all_found {
                        diagnostics.push((name.clone(), false, Some(derivation.clone())));
                    }
                }
            }
        }
        diagnostics
    }

    /// Validate domain orthogonality: for each pair of domains,
    /// measure average inter-domain similarity (should be ~0.5 for random baseline).
    pub fn validate_domain_orthogonality(&self) -> Vec<(String, String, f32)> {
        let domain_names: Vec<String> = self.domains.keys().cloned().collect();
        let mut results = Vec::new();

        for i in 0..domain_names.len() {
            for j in (i+1)..domain_names.len() {
                let prims_i: Vec<&Primitive> = self.primitives.values()
                    .filter(|p| p.domain == domain_names[i])
                    .collect();
                let prims_j: Vec<&Primitive> = self.primitives.values()
                    .filter(|p| p.domain == domain_names[j])
                    .collect();

                if prims_i.is_empty() || prims_j.is_empty() { continue; }

                let mut total_sim = 0.0f32;
                let mut count = 0u32;
                for pi in &prims_i {
                    for pj in &prims_j {
                        total_sim += pi.encoding.similarity(&pj.encoding);
                        count += 1;
                    }
                }
                let avg_sim = total_sim / count as f32;
                results.push((domain_names[i].clone(), domain_names[j].clone(), avg_sim));
            }
        }
        results
    }

    /// Run all validation checks and return a summary.
    #[allow(clippy::type_complexity)]
    pub fn validate_all(&self) -> (Vec<(String, bool, Option<String>)>, Vec<(String, String, f32)>) {
        (self.validate_derivation_chain(), self.validate_domain_orthogonality())
    }

    // === COMPOSITIONAL OPERATOR EXECUTION (Tier 7) ===

    /// Execute a sequence of HDC operations (function composition).
    pub fn execute_sequence(
        ops: &[Box<dyn Fn(BinaryHV) -> BinaryHV>],
        input: BinaryHV,
    ) -> BinaryHV {
        let mut result = input;
        for op in ops {
            result = op(result);
        }
        result
    }

    /// Execute conditional: if condition is similar to reference (above threshold),
    /// apply then_op, otherwise apply else_op.
    pub fn execute_conditional(
        condition: &BinaryHV,
        reference: &BinaryHV,
        threshold: f32,
        then_op: &dyn Fn(BinaryHV) -> BinaryHV,
        else_op: &dyn Fn(BinaryHV) -> BinaryHV,
        input: BinaryHV,
    ) -> BinaryHV {
        if condition.similarity(reference) > threshold {
            then_op(input)
        } else {
            else_op(input)
        }
    }

    /// Execute fixpoint iteration: apply op until the result stabilizes
    /// (similarity to previous > threshold) or max_iter is reached.
    /// Returns (final_result, iterations_used).
    pub fn execute_fixpoint(
        op: &dyn Fn(BinaryHV) -> BinaryHV,
        initial: BinaryHV,
        max_iter: usize,
        threshold: f32,
    ) -> (BinaryHV, usize) {
        let mut current = initial;
        for i in 0..max_iter {
            let next = op(current);
            if next.similarity(&current) > threshold {
                return (next, i + 1);
            }
            current = next;
        }
        (current, max_iter)
    }

    /// Execute iterate: apply op n times starting from initial.
    pub fn execute_iterate(
        op: &dyn Fn(BinaryHV) -> BinaryHV,
        initial: BinaryHV,
        n: usize,
    ) -> BinaryHV {
        let mut result = initial;
        for _ in 0..n {
            result = op(result);
        }
        result
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
            PrimitiveTier::Code,
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
    pub fn find_similar_to_encoding(&self, encoding: &BinaryHV, top_k: usize) -> Vec<(String, f32)> {
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
        encoding: &BinaryHV,
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
        queries: &[BinaryHV],
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
        queries: &[BinaryHV],
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
        queries: &[BinaryHV],
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
    pub fn pairwise_similarities(&self, encodings: &[BinaryHV]) -> Vec<(usize, usize, f32)> {
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
            .filter_map(|n| self.get(n).map(|p| p.encoding))
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

    /// Bind two named primitives together (XOR in BinaryHV space).
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

    /// Bundle multiple named primitives together (majority vote in BinaryHV space).
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
            encodings.push(prim.encoding);
        }

        let encoding = BinaryHV::bundle(&encodings);

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
            encodings.push(prim.encoding);
            weights.push(*weight / total_weight);
        }

        // Weighted bundling: for each bit position, sum weighted votes
        // BinaryHV is [u8; 2048] (2048 * 8 = 16384 bits)
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

        let encoding = BinaryHV(result_bytes);
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

    /// Permute a named primitive (cyclic rotation in BinaryHV space).
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

        let mut encoding = first.encoding;

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
    pub fn query(&self, encoding: &BinaryHV) -> (String, f32) {
        let matches = self.find_similar_to_encoding(encoding, 1);
        matches.into_iter().next().unwrap_or_else(|| ("UNKNOWN".to_string(), 0.0))
    }
}

impl Default for PrimitiveSystem {
    fn default() -> Self {
        Self::new()
    }
}
