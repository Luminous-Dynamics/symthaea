// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 0-5 initialization for the Primitive System.
//!
//! This module contains the init methods for the foundational tiers:
//! - Tier 0: NSM (Natural Semantic Metalanguage)
//! - Tier 1: Mathematical & Logical Primitives
//! - Tier 2: Physical Reality Primitives
//! - Tier 3: Geometric & Topological Primitives
//! - Tier 4: Strategic & Social Primitives
//! - Tier 5: Meta-Cognitive & Metabolic Primitives

use super::{
    BindingRule, DomainManifold, Primitive, PrimitiveSystem, PrimitiveTier, seed_from_name,
};
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::universal_semantics::SemanticPrime;

impl PrimitiveSystem {
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
    pub(super) fn init_tier0_nsm(&mut self) {
        // Create NSM domain manifold - grounding for human semantic understanding
        let nsm_domain = DomainManifold::new(
            "nsm",
            PrimitiveTier::NSM,
            "Natural Semantic Metalanguage - universal human concepts",
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
            self.by_tier
                .entry(PrimitiveTier::NSM)
                .or_default()
                .push(name);
        }

        // Store the domain
        self.domains.insert("nsm".to_string(), nsm_domain);
    }

    /// Convert SemanticPrime enum variant to primitive name string
    pub(super) fn semantic_prime_to_name(prime: SemanticPrime) -> String {
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
    pub(super) fn init_tier1_mathematical(&mut self) {
        // Create mathematical domain manifold
        let math_domain = DomainManifold::new(
            "mathematics",
            PrimitiveTier::Mathematical,
            "Formal reasoning from first principles",
        );

        let logic_domain = DomainManifold::new(
            "logic",
            PrimitiveTier::Mathematical,
            "Logical operators and inference",
        );

        // === SET THEORY PRIMITIVES ===

        // SET - the concept of a collection
        let set = Primitive::base(
            "SET",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("SET"))),
            "A collection of distinct objects",
        );

        // MEMBERSHIP (∈) - element belongs to set
        let membership = Primitive::base(
            "MEMBERSHIP",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("MEMBERSHIP"))),
            "Relation: x ∈ S (x is an element of set S)",
        );

        // UNION (∪) - combine sets
        let union = Primitive::base(
            "UNION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("UNION"))),
            "Operation: A ∪ B (all elements in A or B)",
        );

        // INTERSECTION (∩) - common elements
        let intersection = Primitive::base(
            "INTERSECTION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INTERSECTION"))),
            "Operation: A ∩ B (elements in both A and B)",
        );

        // EMPTY_SET (∅) - the set with no elements
        let empty_set = Primitive::base(
            "EMPTY_SET",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("EMPTY_SET"))),
            "The unique set with no elements: ∅",
        );

        // === LOGICAL PRIMITIVES ===

        // NOT (¬) - logical negation
        let not = Primitive::base(
            "NOT",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("NOT"))),
            "Unary operator: ¬P (negation of proposition P)",
        );

        // AND (∧) - logical conjunction
        let and = Primitive::base(
            "AND",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("AND"))),
            "Binary operator: P ∧ Q (both P and Q are true)",
        );

        // OR (∨) - logical disjunction
        let or = Primitive::base(
            "OR",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("OR"))),
            "Binary operator: P ∨ Q (at least one of P or Q is true)",
        );

        // IMPLIES (→) - logical implication
        let implies = Primitive::base(
            "IMPLIES",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("IMPLIES"))),
            "Binary operator: P → Q (if P then Q)",
        );

        // IFF (↔) - logical equivalence
        let iff = Primitive::base(
            "IFF",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("IFF"))),
            "Binary operator: P ↔ Q (P if and only if Q)",
        );

        // EQUALS (=) - equality relation
        let equals = Primitive::base(
            "EQUALS",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("EQUALS"))),
            "Binary relation: x = y (x and y are the same)",
        );

        // TRUE (⊤) - logical truth
        let true_const = Primitive::base(
            "TRUE",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("TRUE"))),
            "The constant truth value: ⊤",
        );

        // FALSE (⊥) - logical falsehood
        let false_const = Primitive::base(
            "FALSE",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("FALSE"))),
            "The constant false value: ⊥",
        );

        // === PEANO ARITHMETIC PRIMITIVES ===

        // ZERO (0) - the first natural number
        let zero = Primitive::base(
            "ZERO",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ZERO"))),
            "The first natural number: 0",
        );

        // ONE (1) - successor of zero
        let one = Primitive::derived(
            "ONE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ONE"))),
            "The natural number one: 1",
            "SUCCESSOR(ZERO)",
        );

        // SUCCESSOR (S) - next natural number
        let successor = Primitive::base(
            "SUCCESSOR",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("SUCCESSOR"))),
            "Function: S(n) = n+1 (next natural number)",
        );

        // ADDITION (+) - derived from successor
        let addition = Primitive::derived(
            "ADDITION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ADDITION"))),
            "Binary operation: m + n (sum of m and n)",
            "Recursive: m + 0 = m, m + S(n) = S(m + n)",
        );

        // MULTIPLICATION (×) - derived from addition
        let multiplication = Primitive::derived(
            "MULTIPLICATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("MULTIPLICATION"))),
            "Binary operation: m × n (product of m and n)",
            "Recursive: m × 0 = 0, m × S(n) = m × n + m",
        );

        // === FOUNDATIONAL MATHEMATICAL PRIMITIVES ===
        // These are base concepts needed by derived primitives in later tiers

        let ratio = Primitive::base(
            "RATIO",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("RATIO"))),
            "Relation: proportional comparison of two quantities (a/b)",
        );

        let information = Primitive::base(
            "INFORMATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INFORMATION"))),
            "Quantity: reduction in uncertainty (bits)",
        );

        let deviation = Primitive::base(
            "DEVIATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("DEVIATION"))),
            "Measure: distance from a central or expected value",
        );

        let limit = Primitive::base(
            "LIMIT",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("LIMIT"))),
            "Bound: supremum or constraint on a quantity",
        );

        let efficiency = Primitive::base(
            "EFFICIENCY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("EFFICIENCY"))),
            "Ratio: useful output to total input",
        );

        // === NEGATION PRIMITIVE ===
        // Unary operation: additive inverse needed for integer arithmetic (Z)

        let negation = Primitive::base(
            "NEGATION",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("NEGATION"))),
            "Unary operation: additive inverse, -a where a + (-a) = 0",
        );

        // === ALGEBRAIC STRUCTURE PRIMITIVES ===
        // Domain for abstract algebra concepts

        let algebra_domain = DomainManifold::new(
            "algebra",
            PrimitiveTier::Mathematical,
            "Abstract algebraic structures and their properties",
        );

        let group = Primitive::base(
            "GROUP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("GROUP"))),
            "Set + associative binary op + identity + inverses",
        );

        let ring = Primitive::base(
            "RING",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("RING"))),
            "(S,+) abelian group + (S,*) monoid + distributivity",
        );

        let field_alg = Primitive::base(
            "FIELD_ALG",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("FIELD_ALG"))),
            "Ring where nonzero elements form multiplicative group",
        );

        let homomorphism = Primitive::base(
            "HOMOMORPHISM",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("HOMOMORPHISM"))),
            "Structure-preserving map between algebraic objects",
        );

        let isomorphism = Primitive::base(
            "ISOMORPHISM",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("ISOMORPHISM"))),
            "Bijective homomorphism (structural equivalence)",
        );

        let order = Primitive::base(
            "ORDER",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("ORDER"))),
            "Binary relation: reflexive, antisymmetric, transitive",
        );

        let inverse = Primitive::base(
            "INVERSE",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("INVERSE"))),
            "Element reversing an operation: a * a^(-1) = e",
        );

        let identity_element = Primitive::base(
            "IDENTITY_ELEMENT",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("IDENTITY_ELEMENT"))),
            "Neutral element: a * e = e * a = a",
        );

        let associativity_prop = Primitive::base(
            "ASSOCIATIVITY_PROP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("ASSOCIATIVITY_PROP"))),
            "Property: (a*b)*c = a*(b*c)",
        );

        let commutativity_prop = Primitive::base(
            "COMMUTATIVITY_PROP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("COMMUTATIVITY_PROP"))),
            "Property: a*b = b*a",
        );

        let distributivity_prop = Primitive::base(
            "DISTRIBUTIVITY_PROP",
            PrimitiveTier::Mathematical,
            "algebra",
            algebra_domain.embed(BinaryHV::random(seed_from_name("DISTRIBUTIVITY_PROP"))),
            "Property: a*(b+c) = a*b + a*c",
        );

        // === CALCULUS / ANALYSIS PRIMITIVES ===

        let integration_calc = Primitive::base(
            "INTEGRATION_CALC",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INTEGRATION_CALC"))),
            "Accumulation / antiderivative: integral of f over domain",
        );

        let convergence = Primitive::base(
            "CONVERGENCE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("CONVERGENCE"))),
            "Sequence/series approaching a limit",
        );

        let continuity = Primitive::base(
            "CONTINUITY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("CONTINUITY"))),
            "Small input changes produce small output changes",
        );

        let infinity = Primitive::base(
            "INFINITY",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INFINITY"))),
            "Unbounded quantity, larger than any finite number",
        );

        // === LINEAR ALGEBRA PRIMITIVES ===

        let matrix = Primitive::base(
            "MATRIX",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("MATRIX"))),
            "Rectangular array of numbers (rows × columns)",
        );

        let vector = Primitive::base(
            "VECTOR",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("VECTOR"))),
            "Ordered sequence of numbers in a linear space",
        );

        let determinant = Primitive::derived(
            "DETERMINANT",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("DETERMINANT"))),
            "Scalar: signed volume scaling factor of a matrix",
            "MATRIX → scalar (product of LU diagonal × sign)",
        );

        let eigenvalue = Primitive::derived(
            "EIGENVALUE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("EIGENVALUE"))),
            "Scalar λ where Av = λv for eigenvector v",
            "DETERMINANT(A - λI) = 0",
        );

        let transpose = Primitive::base(
            "TRANSPOSE",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("TRANSPOSE"))),
            "Matrix reflection: swap rows and columns (A^T)",
        );

        // === ROOT FINDING PRIMITIVE ===

        let root = Primitive::base(
            "ROOT",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("ROOT"))),
            "Value x where f(x) = 0 (zero of a function)",
        );

        // === NUMERICAL INTEGRATION PRIMITIVE ===

        let integral_numeric = Primitive::derived(
            "INTEGRAL_NUMERIC",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("INTEGRAL_NUMERIC"))),
            "Numerical approximation of definite integral ∫[a,b] f(x) dx",
            "Quadrature: Simpson / Gauss-Legendre / Adaptive",
        );

        // === LOGIC PRIMITIVES (Phase 3) ===

        let forall = Primitive::base(
            "FORALL",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("FORALL"))),
            "Universal quantifier: ∀x. P(x) — for all x, P holds",
        );

        let exists = Primitive::base(
            "EXISTS",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("EXISTS"))),
            "Existential quantifier: ∃x. P(x) — there exists x such that P holds",
        );

        let satisfies = Primitive::base(
            "SATISFIES",
            PrimitiveTier::Mathematical,
            "logic",
            logic_domain.embed(BinaryHV::random(seed_from_name("SATISFIES"))),
            "Satisfaction relation: assignment makes formula true",
        );

        let constraint = Primitive::base(
            "CONSTRAINT",
            PrimitiveTier::Mathematical,
            "mathematics",
            math_domain.embed(BinaryHV::random(seed_from_name("CONSTRAINT"))),
            "Restriction on variable values in a CSP",
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains.insert("mathematics".to_string(), math_domain);
        self.domains.insert("logic".to_string(), logic_domain);
        self.domains.insert("algebra".to_string(), algebra_domain);

        for primitive in vec![
            set,
            membership,
            union,
            intersection,
            empty_set,
            not,
            and,
            or,
            implies,
            iff,
            equals,
            true_const,
            false_const,
            zero,
            one,
            successor,
            addition,
            multiplication,
            ratio,
            information,
            deviation,
            limit,
            efficiency,
            negation,
            group,
            ring,
            field_alg,
            homomorphism,
            isomorphism,
            order,
            inverse,
            identity_element,
            associativity_prop,
            commutativity_prop,
            distributivity_prop,
            integration_calc,
            convergence,
            continuity,
            infinity,
            matrix,
            vector,
            determinant,
            eigenvalue,
            transpose,
            root,
            integral_numeric,
            forall,
            exists,
            satisfies,
            constraint,
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
    pub(super) fn init_tier2_physical(&mut self) {
        // Create physics domain manifold
        let physics_domain = DomainManifold::new(
            "physics",
            PrimitiveTier::Physical,
            "Physical reality grounding - mass, energy, forces",
        );

        let causality_domain = DomainManifold::new(
            "causality",
            PrimitiveTier::Physical,
            "Cause-effect relationships and state changes",
        );

        // === PHYSICAL PROPERTIES ===

        // MASS - quantity of matter
        let mass = Primitive::base(
            "MASS",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("MASS"))),
            "Property: quantity of matter in an object (kg)",
        );

        // CHARGE - electric charge
        let charge = Primitive::base(
            "CHARGE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("CHARGE"))),
            "Property: electric charge (coulombs)",
        );

        // SPIN - quantum angular momentum
        let spin = Primitive::base(
            "SPIN",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("SPIN"))),
            "Property: intrinsic angular momentum (quantum)",
        );

        // === ENERGY AND FORCES ===

        // ENERGY - capacity to do work
        let energy = Primitive::base(
            "ENERGY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("ENERGY"))),
            "Property: capacity to perform work (joules)",
        );

        // WORK - energy transfer through force
        let work = Primitive::derived(
            "WORK",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("WORK"))),
            "Quantity: energy transferred by force over distance",
            "BIND(FORCE, DISTANCE)",
        );

        // FORCE - interaction that changes motion
        let force = Primitive::base(
            "FORCE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("FORCE"))),
            "Vector: interaction that changes object's motion (newtons)",
        );

        // === MOTION PRIMITIVES ===

        // VELOCITY - rate of position change
        let velocity = Primitive::base(
            "VELOCITY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("VELOCITY"))),
            "Vector: rate of change of position (m/s)",
        );

        // ACCELERATION - rate of velocity change
        let acceleration = Primitive::derived(
            "ACCELERATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("ACCELERATION"))),
            "Vector: rate of change of velocity (m/s²)",
            "DERIVATIVE(VELOCITY)",
        );

        // MOMENTUM - quantity of motion
        let momentum = Primitive::derived(
            "MOMENTUM",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("MOMENTUM"))),
            "Vector: quantity of motion (mass × velocity)",
            "BIND(MASS, VELOCITY)",
        );

        // === CAUSALITY ===

        // CAUSE - event that produces effect
        let cause = Primitive::base(
            "CAUSE",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(BinaryHV::random(seed_from_name("CAUSE"))),
            "Event: that which produces an effect",
        );

        // EFFECT - result of a cause
        let effect = Primitive::base(
            "EFFECT",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(BinaryHV::random(seed_from_name("EFFECT"))),
            "Event: result produced by a cause",
        );

        // STATE_CHANGE - transition between states
        let state_change = Primitive::derived(
            "STATE_CHANGE",
            PrimitiveTier::Physical,
            "causality",
            causality_domain.embed(BinaryHV::random(seed_from_name("STATE_CHANGE"))),
            "Process: transition from one state to another",
            "BIND(CAUSE, EFFECT)",
        );

        // === THERMODYNAMICS ===

        // THERMODYNAMIC_ENTROPY - measure of disorder (S = k_B ln Ω)
        // Distinct from SHANNON_ENTROPY (information-theoretic) in the uncertainty domain
        let entropy = Primitive::base(
            "THERMODYNAMIC_ENTROPY",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("THERMODYNAMIC_ENTROPY"))),
            "Property: thermodynamic measure of disorder, S = k_B ln Ω (J/K)",
        );

        // TEMPERATURE - average kinetic energy
        let temperature = Primitive::base(
            "TEMPERATURE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("TEMPERATURE"))),
            "Property: average kinetic energy of particles (K)",
        );

        // === CONSERVATION ===

        // CONSERVATION - invariant quantity
        let conservation = Primitive::base(
            "CONSERVATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("CONSERVATION"))),
            "Principle: certain quantities remain constant over time",
        );

        // === FOUNDATIONAL PHYSICAL PRIMITIVES ===
        // Base concepts needed by derived physics/information primitives

        let differentiation = Primitive::base(
            "DIFFERENTIATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("DIFFERENTIATION"))),
            "Operation: rate of change of a quantity with respect to another",
        );

        let space = Primitive::base(
            "SPACE",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("SPACE"))),
            "Continuum: spatial extent in which objects exist and move",
        );

        let oscillation = Primitive::base(
            "OSCILLATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("OSCILLATION"))),
            "Process: repetitive variation about a central value",
        );

        let propagation = Primitive::base(
            "PROPAGATION",
            PrimitiveTier::Physical,
            "physics",
            physics_domain.embed(BinaryHV::random(seed_from_name("PROPAGATION"))),
            "Process: transmission of a disturbance through a medium or field",
        );

        // === REGISTER ALL TIER 2 PRIMITIVES ===

        self.domains.insert("physics".to_string(), physics_domain);
        self.domains
            .insert("causality".to_string(), causality_domain);

        for primitive in vec![
            mass,
            charge,
            spin,
            energy,
            work,
            force,
            velocity,
            acceleration,
            momentum,
            cause,
            effect,
            state_change,
            entropy,
            temperature,
            conservation,
            differentiation,
            space,
            oscillation,
            propagation,
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
    pub(super) fn init_tier3_geometric(&mut self) {
        // Create geometry domain manifolds
        let geometry_domain = DomainManifold::new(
            "geometry",
            PrimitiveTier::Geometric,
            "Euclidean and differential geometry",
        );

        let topology_domain = DomainManifold::new(
            "topology",
            PrimitiveTier::Geometric,
            "Topological and mereotopological relations",
        );

        // === BASIC GEOMETRY ===

        // POINT - location in space
        let point = Primitive::base(
            "POINT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("POINT"))),
            "Entity: location with no dimension",
        );

        // LINE - one-dimensional extent
        let line = Primitive::derived(
            "LINE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("LINE"))),
            "Entity: one-dimensional extent through space",
            "CONNECT(POINT, POINT)",
        );

        // PLANE - two-dimensional surface
        let plane = Primitive::base(
            "PLANE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("PLANE"))),
            "Entity: flat two-dimensional surface",
        );

        // ANGLE - measure of rotation
        let angle = Primitive::base(
            "ANGLE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("ANGLE"))),
            "Quantity: measure of rotation between two lines",
        );

        // DISTANCE - spatial separation
        let distance = Primitive::derived(
            "DISTANCE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("DISTANCE"))),
            "Quantity: spatial separation between points",
            "MEASURE(POINT, POINT)",
        );

        // === VECTOR GEOMETRY ===

        // VECTOR - directed magnitude
        let vector = Primitive::base(
            "VECTOR",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("VECTOR"))),
            "Entity: quantity with magnitude and direction",
        );

        // DOT_PRODUCT - scalar product
        let dot_product = Primitive::base(
            "DOT_PRODUCT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("DOT_PRODUCT"))),
            "Operation: scalar product of two vectors",
        );

        // CROSS_PRODUCT - vector product
        let cross_product = Primitive::base(
            "CROSS_PRODUCT",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("CROSS_PRODUCT"))),
            "Operation: vector product perpendicular to both inputs",
        );

        // === DIFFERENTIAL GEOMETRY ===

        // MANIFOLD - curved space
        let manifold = Primitive::base(
            "MANIFOLD",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("MANIFOLD"))),
            "Entity: space that locally resembles Euclidean space",
        );

        // TANGENT_SPACE - local linear approximation
        let tangent_space = Primitive::base(
            "TANGENT_SPACE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("TANGENT_SPACE"))),
            "Entity: linear approximation at a manifold point",
        );

        // CURVATURE - deviation from flatness
        let curvature = Primitive::base(
            "CURVATURE",
            PrimitiveTier::Geometric,
            "geometry",
            geometry_domain.embed(BinaryHV::random(seed_from_name("CURVATURE"))),
            "Property: measure of deviation from flatness",
        );

        // === TOPOLOGY ===

        // OPEN_SET - set excluding boundary
        let open_set = Primitive::base(
            "OPEN_SET",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("OPEN_SET"))),
            "Set: excluding its boundary points",
        );

        // CLOSED_SET - set including boundary
        let closed_set = Primitive::base(
            "CLOSED_SET",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("CLOSED_SET"))),
            "Set: including all its boundary points",
        );

        // BOUNDARY - edge of a region
        let boundary = Primitive::base(
            "BOUNDARY",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("BOUNDARY"))),
            "Set: points on the edge of a region",
        );

        // INTERIOR - inside of a region
        let interior = Primitive::base(
            "INTERIOR",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("INTERIOR"))),
            "Set: all points strictly inside a region",
        );

        // === MEREOTOPOLOGY (part-whole) ===

        // PART_OF - mereological inclusion
        let part_of = Primitive::base(
            "PART_OF",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("PART_OF"))),
            "Relation: x is part of y",
        );

        // OVERLAPS - shared parts
        let overlaps = Primitive::base(
            "OVERLAPS",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("OVERLAPS"))),
            "Relation: x and y share common parts",
        );

        // TOUCHES - external contact
        let touches = Primitive::base(
            "TOUCHES",
            PrimitiveTier::Geometric,
            "topology",
            topology_domain.embed(BinaryHV::random(seed_from_name("TOUCHES"))),
            "Relation: x and y are in contact at boundary",
        );

        // === REGISTER ALL TIER 3 PRIMITIVES ===

        self.domains.insert("geometry".to_string(), geometry_domain);
        self.domains.insert("topology".to_string(), topology_domain);

        for primitive in vec![
            point,
            line,
            plane,
            angle,
            distance,
            vector,
            dot_product,
            cross_product,
            manifold,
            tangent_space,
            curvature,
            open_set,
            closed_set,
            boundary,
            interior,
            part_of,
            overlaps,
            touches,
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
    pub(super) fn init_tier4_strategic(&mut self) {
        // Create strategic domain manifolds
        let game_theory_domain = DomainManifold::new(
            "game_theory",
            PrimitiveTier::Strategic,
            "Strategic reasoning and multi-agent coordination",
        );

        let temporal_domain = DomainManifold::new(
            "temporal",
            PrimitiveTier::Strategic,
            "Temporal logic and interval relations",
        );

        let social_domain = DomainManifold::new(
            "social",
            PrimitiveTier::Strategic,
            "Social coordination and cooperation",
        );

        // === GAME THEORY ===

        // UTILITY - preference measure
        let utility = Primitive::base(
            "UTILITY",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("UTILITY"))),
            "Function: measure of preference or value",
        );

        // STRATEGY - action plan
        let strategy = Primitive::base(
            "STRATEGY",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("STRATEGY"))),
            "Plan: complete specification of actions in all situations",
        );

        // EQUILIBRIUM - stable state
        let equilibrium = Primitive::base(
            "EQUILIBRIUM",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("EQUILIBRIUM"))),
            "State: where no agent benefits from unilateral deviation",
        );

        // PAYOFF - outcome value
        let payoff = Primitive::derived(
            "PAYOFF",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("PAYOFF"))),
            "Value: utility resulting from strategy profile",
            "APPLY(UTILITY, STRATEGY)",
        );

        // === TEMPORAL LOGIC (Allen's Intervals) ===

        // BEFORE - temporal precedence
        let before = Primitive::base(
            "BEFORE",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("BEFORE"))),
            "Relation: interval x ends before interval y starts",
        );

        // AFTER - temporal succession
        let after = Primitive::base(
            "AFTER",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("AFTER"))),
            "Relation: interval x starts after interval y ends",
        );

        // DURING - temporal containment
        let during = Primitive::base(
            "DURING",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("DURING"))),
            "Relation: interval x occurs within interval y",
        );

        // MEETS - temporal adjacency
        let meets = Primitive::base(
            "MEETS",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("MEETS"))),
            "Relation: interval x ends exactly when y starts",
        );

        // OVERLAPS_TEMPORAL - partial overlap
        let overlaps_temporal = Primitive::base(
            "OVERLAPS_TEMPORAL",
            PrimitiveTier::Strategic,
            "temporal",
            temporal_domain.embed(BinaryHV::random(seed_from_name("OVERLAPS_TEMPORAL"))),
            "Relation: intervals x and y partially overlap in time",
        );

        // === COUNTERFACTUAL REASONING ===

        // COUNTERFACTUAL - hypothetical condition
        let counterfactual = Primitive::base(
            "COUNTERFACTUAL",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("COUNTERFACTUAL"))),
            "Condition: what would have happened if...",
        );

        // POSSIBLE_WORLD - alternative reality
        let possible_world = Primitive::base(
            "POSSIBLE_WORLD",
            PrimitiveTier::Strategic,
            "game_theory",
            game_theory_domain.embed(BinaryHV::random(seed_from_name("POSSIBLE_WORLD"))),
            "Structure: consistent alternative state of reality",
        );

        // === SOCIAL COORDINATION ===

        // COOPERATE - joint action for mutual benefit
        let cooperate = Primitive::base(
            "COOPERATE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("COOPERATE"))),
            "Action: work together for mutual benefit",
        );

        // DEFECT - self-interested deviation
        let defect = Primitive::base(
            "DEFECT",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("DEFECT"))),
            "Action: act in self-interest against cooperation",
        );

        // RECIPROCATE - conditional cooperation
        let reciprocate = Primitive::derived(
            "RECIPROCATE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("RECIPROCATE"))),
            "Strategy: cooperate if and only if partner cooperates",
            "CONDITIONAL(COOPERATE, COOPERATE)",
        );

        // TRUST - belief in cooperation
        let trust = Primitive::base(
            "TRUST",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("TRUST"))),
            "Belief: expectation that others will cooperate",
        );

        // === INFORMATION ===

        // SIGNAL - information transmission
        let signal = Primitive::base(
            "SIGNAL",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("SIGNAL"))),
            "Action: transmit information to influence others",
        );

        // BELIEF - subjective probability
        let belief = Primitive::base(
            "BELIEF",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("BELIEF"))),
            "State: subjective probability assignment",
        );

        // COMMON_KNOWLEDGE - shared awareness
        let common_knowledge = Primitive::base(
            "COMMON_KNOWLEDGE",
            PrimitiveTier::Strategic,
            "social",
            social_domain.embed(BinaryHV::random(seed_from_name("COMMON_KNOWLEDGE"))),
            "State: all know, all know that all know, etc.",
        );

        // === REGISTER ALL TIER 4 PRIMITIVES ===

        self.domains
            .insert("game_theory".to_string(), game_theory_domain);
        self.domains.insert("temporal".to_string(), temporal_domain);
        self.domains.insert("social".to_string(), social_domain);

        for primitive in vec![
            utility,
            strategy,
            equilibrium,
            payoff,
            before,
            after,
            during,
            meets,
            overlaps_temporal,
            counterfactual,
            possible_world,
            cooperate,
            defect,
            reciprocate,
            trust,
            signal,
            belief,
            common_knowledge,
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
            example: "COOPERATE ⊗ TRUST → Mutual Reciprocity harmonic".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "temporal_reasoning".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Strategic],
            result_tier: PrimitiveTier::Strategic,
            example: "BEFORE ⊗ AFTER → temporal sequence".to_string(),
        });
    }

    /// Initialize Tier 5: Meta-Cognitive & Metabolic Primitives
    pub(super) fn init_tier5_metacognitive(&mut self) {
        // Create meta-cognitive domain manifolds
        let metacognition_domain = DomainManifold::new(
            "metacognition",
            PrimitiveTier::MetaCognitive,
            "Self-awareness and introspection",
        );

        let homeostasis_domain = DomainManifold::new(
            "homeostasis",
            PrimitiveTier::MetaCognitive,
            "Self-regulation and repair",
        );

        let epistemic_domain = DomainManifold::new(
            "epistemic",
            PrimitiveTier::MetaCognitive,
            "Knowledge and uncertainty",
        );

        let metabolic_domain = DomainManifold::new(
            "metabolic",
            PrimitiveTier::MetaCognitive,
            "Resource allocation and management",
        );

        // === SELF-AWARENESS ===

        // SELF - reflexive identity
        let self_prim = Primitive::base(
            "SELF",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("SELF"))),
            "Entity: the reflexive subject of awareness",
        );

        // IDENTITY - persistent self-recognition
        let identity = Primitive::base(
            "IDENTITY",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("IDENTITY"))),
            "Property: persistent self-recognition over time",
        );

        // META_BELIEF - belief about beliefs
        let meta_belief = Primitive::derived(
            "META_BELIEF",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("META_BELIEF"))),
            "State: belief about one's own beliefs",
            "APPLY(SELF, BELIEF)",
        );

        // INTROSPECTION - self-examination
        let introspection = Primitive::base(
            "INTROSPECTION",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("INTROSPECTION"))),
            "Process: examination of one's own mental states",
        );

        // === HOMEOSTASIS & REGULATION ===

        let homeostasis = Primitive::base(
            "HOMEOSTASIS",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("HOMEOSTASIS"))),
            "Process: maintaining stable internal state",
        );

        let setpoint = Primitive::base(
            "SETPOINT",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("SETPOINT"))),
            "Value: target state for homeostatic regulation",
        );

        let regulation = Primitive::base(
            "REGULATION",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("REGULATION"))),
            "Process: adjusting state toward setpoint",
        );

        let feedback = Primitive::base(
            "FEEDBACK",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("FEEDBACK"))),
            "Signal: information about current state vs setpoint",
        );

        // === REPAIR & ADAPTATION ===

        let repair = Primitive::base(
            "REPAIR",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("REPAIR"))),
            "Process: restoring damaged structures or functions",
        );

        let restore = Primitive::base(
            "RESTORE",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("RESTORE"))),
            "Process: returning to a previous functional state",
        );

        let adapt = Primitive::base(
            "ADAPT",
            PrimitiveTier::MetaCognitive,
            "homeostasis",
            homeostasis_domain.embed(BinaryHV::random(seed_from_name("ADAPT"))),
            "Process: modify structure/behavior in response to environment",
        );

        let learn = Primitive::base(
            "LEARN",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("LEARN"))),
            "Process: update knowledge or behavior from experience",
        );

        // === EPISTEMIC STRENGTH ===

        let know = Primitive::base(
            "KNOW",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("KNOW"))),
            "State: justified true belief",
        );

        let uncertain = Primitive::base(
            "UNCERTAIN",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("UNCERTAIN"))),
            "State: lacking sufficient information for certainty",
        );

        let confidence = Primitive::base(
            "CONFIDENCE",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("CONFIDENCE"))),
            "Measure: degree of certainty in a belief",
        );

        let evidence = Primitive::base(
            "EVIDENCE",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("EVIDENCE"))),
            "Support: information supporting or refuting a belief",
        );

        // === METABOLIC / RESOURCE MANAGEMENT ===

        let resource = Primitive::base(
            "RESOURCE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("RESOURCE"))),
            "Entity: available capacity for use",
        );

        let allocate = Primitive::base(
            "ALLOCATE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("ALLOCATE"))),
            "Process: distribute resources to tasks",
        );

        let consume = Primitive::base(
            "CONSUME",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("CONSUME"))),
            "Process: use resources to perform work",
        );

        let produce = Primitive::base(
            "PRODUCE",
            PrimitiveTier::MetaCognitive,
            "metabolic",
            metabolic_domain.embed(BinaryHV::random(seed_from_name("PRODUCE"))),
            "Process: generate resources from inputs",
        );

        // === REWARD & VALUE ===

        let reward = Primitive::base(
            "REWARD",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("REWARD"))),
            "Signal: positive reinforcement for actions",
        );

        let goal = Primitive::base(
            "GOAL",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("GOAL"))),
            "State: desired future state to achieve",
        );

        let value = Primitive::base(
            "VALUE",
            PrimitiveTier::MetaCognitive,
            "metacognition",
            metacognition_domain.embed(BinaryHV::random(seed_from_name("VALUE"))),
            "Measure: importance or worth of a state/action",
        );

        let certainty = Primitive::base(
            "CERTAINTY",
            PrimitiveTier::MetaCognitive,
            "epistemic",
            epistemic_domain.embed(BinaryHV::random(seed_from_name("CERTAINTY"))),
            "State: complete confidence in a proposition's truth value",
        );

        // === REGISTER ALL TIER 5 PRIMITIVES ===

        self.domains
            .insert("metacognition".to_string(), metacognition_domain);
        self.domains
            .insert("homeostasis".to_string(), homeostasis_domain);
        self.domains
            .insert("epistemic".to_string(), epistemic_domain);
        self.domains
            .insert("metabolic".to_string(), metabolic_domain);

        for primitive in vec![
            self_prim,
            identity,
            meta_belief,
            introspection,
            homeostasis,
            setpoint,
            regulation,
            feedback,
            repair,
            restore,
            adapt,
            learn,
            know,
            uncertain,
            confidence,
            evidence,
            certainty,
            resource,
            allocate,
            consume,
            produce,
            reward,
            goal,
            value,
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
}
