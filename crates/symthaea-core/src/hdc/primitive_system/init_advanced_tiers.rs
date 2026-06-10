// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced tier initialization (Tiers 6-9) for the Primitive System.
//!
//! This module contains initialization for:
//! - Tier 6: Temporal Primitives (Allen's Interval Algebra extended)
//! - Tier 7: Compositional Primitives (composition operators)
//! - Tier 8: Consciousness-Specific Primitives
//! - Tier 9: Code & Symbol Manipulation Primitives

use super::{
    BindingRule, DomainManifold, Primitive, PrimitiveSystem, PrimitiveTier, seed_from_name,
};
use crate::hdc::binary_hv::BinaryHV;

impl PrimitiveSystem {
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
    pub(super) fn init_tier6_temporal(&mut self) {
        let temporal_domain = DomainManifold::new(
            "temporal_reasoning",
            PrimitiveTier::Temporal,
            "Extended temporal reasoning and interval algebra",
        );

        // === INTERVAL RELATIONS (Extended Allen's) ===

        let starts = Primitive::base(
            "STARTS",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("STARTS"))),
            "Relation: interval x begins at same point as interval y begins",
        );

        let finishes = Primitive::base(
            "FINISHES",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("FINISHES"))),
            "Relation: interval x ends at same point as interval y ends",
        );

        let equals_temporal = Primitive::base(
            "EQUALS_TEMPORAL",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("EQUALS_TEMPORAL"))),
            "Relation: intervals x and y have identical start and end points",
        );

        // === TEMPORAL CONCEPTS ===

        let instant = Primitive::base(
            "INSTANT",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("INSTANT"))),
            "A point in time with zero duration",
        );

        let duration = Primitive::base(
            "DURATION",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("DURATION"))),
            "The length or extent of a temporal interval",
        );

        let tempo = Primitive::base(
            "TEMPO",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("TEMPO"))),
            "Rate of occurrence or change over time",
        );

        let rhythm = Primitive::base(
            "RHYTHM",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("RHYTHM"))),
            "Repeating pattern of temporal events",
        );

        let anticipate = Primitive::base(
            "ANTICIPATE",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("ANTICIPATE"))),
            "Expectation or prediction of a future state",
        );

        let persist = Primitive::base(
            "PERSIST",
            PrimitiveTier::Temporal,
            "temporal_reasoning",
            temporal_domain.embed(BinaryHV::random(seed_from_name("PERSIST"))),
            "Continuation of existence or state through time",
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains
            .insert("temporal_reasoning".to_string(), temporal_domain);

        for primitive in vec![
            starts,
            finishes,
            equals_temporal,
            instant,
            duration,
            tempo,
            rhythm,
            anticipate,
            persist,
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
    /// - COMPOSE: Combine functions (f . g)
    /// - CURRY: Partial application
    pub(super) fn init_tier7_compositional(&mut self) {
        let compositional_domain = DomainManifold::new(
            "composition",
            PrimitiveTier::Compositional,
            "Higher-order composition operators for building complex structures",
        );

        // === COMPOSITION OPERATORS ===

        let sequence_op = Primitive::base(
            "SEQUENCE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("SEQUENCE_OP"))),
            "Sequential composition: do A, then do B",
        );

        let parallel_op = Primitive::base(
            "PARALLEL_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("PARALLEL_OP"))),
            "Parallel composition: do A and B simultaneously",
        );

        let conditional_op = Primitive::base(
            "CONDITIONAL_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("CONDITIONAL_OP"))),
            "Conditional composition: if P then A else B",
        );

        let iterate_op = Primitive::base(
            "ITERATE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("ITERATE_OP"))),
            "Iteration: repeated application of an operation",
        );

        let fixpoint_op = Primitive::base(
            "FIXPOINT_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("FIXPOINT_OP"))),
            "Fixed-point: find stable state under repeated application",
        );

        // === STRUCTURAL OPERATORS ===

        let abstract_op = Primitive::base(
            "ABSTRACT_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("ABSTRACT_OP"))),
            "Abstraction: extract common pattern from instances",
        );

        let instantiate_op = Primitive::base(
            "INSTANTIATE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("INSTANTIATE_OP"))),
            "Instantiation: create concrete instance from abstract pattern",
        );

        let compose_op = Primitive::base(
            "COMPOSE_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("COMPOSE_OP"))),
            "Function composition: (f . g)(x) = f(g(x))",
        );

        let curry_op = Primitive::base(
            "CURRY_OP",
            PrimitiveTier::Compositional,
            "composition",
            compositional_domain.embed(BinaryHV::random(seed_from_name("CURRY_OP"))),
            "Currying: transform multi-argument function to chain of single-argument functions",
        );

        // === REGISTER ALL PRIMITIVES ===

        self.domains
            .insert("composition".to_string(), compositional_domain);

        for primitive in vec![
            sequence_op,
            parallel_op,
            conditional_op,
            iterate_op,
            fixpoint_op,
            abstract_op,
            instantiate_op,
            compose_op,
            curry_op,
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
    pub(super) fn init_consciousness_primitives(&mut self) {
        let consciousness_domain = DomainManifold::new(
            "consciousness",
            PrimitiveTier::Consciousness,
            "First-person phenomenal experience, attention, memory, and agency",
        );

        // === QUALIA PRIMITIVES ===

        let quale = Primitive::base(
            "QUALE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("QUALE"))),
            "Irreducible unit of subjective experience - what it is like to experience",
        );

        let phenomenal_binding = Primitive::base(
            "PHENOMENAL_BINDING",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("PHENOMENAL_BINDING"))),
            "Integration of disparate qualia into unified perceptual field",
        );

        let subjective_time = Primitive::base(
            "SUBJECTIVE_TIME",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SUBJECTIVE_TIME"))),
            "The felt passage of time - duration as experienced",
        );

        let sentience = Primitive::base(
            "SENTIENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SENTIENCE"))),
            "Capacity for subjective experience - being a subject of experience",
        );

        // === ATTENTION PRIMITIVES ===

        let attend = Primitive::base(
            "ATTEND",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("ATTEND"))),
            "Selective focus - directing conscious awareness to subset of information",
        );

        let salience = Primitive::base(
            "SALIENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SALIENCE"))),
            "Intrinsic importance - property that draws attention",
        );

        let binding_window = Primitive::base(
            "BINDING_WINDOW",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("BINDING_WINDOW"))),
            "Temporal integration window (~100-200ms) for conscious binding",
        );

        let awareness = Primitive::base(
            "AWARENESS",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("AWARENESS"))),
            "State of being conscious of something - phenomenal access",
        );

        // === MEMORY OPERATION PRIMITIVES ===

        let remember = Primitive::base(
            "REMEMBER",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("REMEMBER"))),
            "Retrieval of encoded episodic information into consciousness",
        );

        let forget = Primitive::base(
            "FORGET",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("FORGET"))),
            "Loss or decay of stored information - natural or active",
        );

        let consolidate = Primitive::base(
            "CONSOLIDATE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("CONSOLIDATE"))),
            "Transfer from working memory to long-term storage",
        );

        let recognize = Primitive::base(
            "RECOGNIZE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("RECOGNIZE"))),
            "Pattern matching of percept to stored memory - familiarity",
        );

        // === AGENCY PRIMITIVES ===

        let intend = Primitive::base(
            "INTEND",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("INTEND"))),
            "Goal-directed mental state - representation of desired outcome",
        );

        let will = Primitive::base(
            "WILL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("WILL"))),
            "Volitional initiation of action - self-determined causation",
        );

        let decide = Primitive::base(
            "DECIDE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("DECIDE"))),
            "Selection among alternatives - commitment to course of action",
        );

        let control = Primitive::base(
            "CONTROL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("CONTROL"))),
            "Executive regulation - top-down modulation of processing",
        );

        // === AFFECTIVE PRIMITIVES ===

        let valence = Primitive::base(
            "VALENCE",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("VALENCE"))),
            "Positive-negative dimension of experience - pleasantness/unpleasantness",
        );

        let arousal = Primitive::base(
            "AROUSAL",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("AROUSAL"))),
            "Activation level of experience - calm to excited",
        );

        let selection = Primitive::base(
            "SELECTION",
            PrimitiveTier::Consciousness,
            "consciousness",
            consciousness_domain.embed(BinaryHV::random(seed_from_name("SELECTION"))),
            "Process: choosing one option from a set of alternatives",
        );

        // Register domain
        self.domains
            .insert("consciousness".to_string(), consciousness_domain);

        // Register all consciousness primitives
        for primitive in vec![
            // Qualia
            quale,
            phenomenal_binding,
            subjective_time,
            sentience,
            // Attention
            attend,
            salience,
            binding_window,
            awareness,
            selection,
            // Memory operations
            remember,
            forget,
            consolidate,
            recognize,
            // Agency
            intend,
            will,
            decide,
            control,
            // Affective
            valence,
            arousal,
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
    pub(super) fn init_tier9_code(&mut self) {
        let code_domain = DomainManifold::new(
            "code",
            PrimitiveTier::Code,
            "Code understanding, generation, and transformation",
        );

        // === STRUCTURAL PRIMITIVES ===

        let parse = Primitive::base(
            "PARSE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("PARSE"))),
            "Decompose source code into AST structure",
        );

        let entity = Primitive::base(
            "ENTITY",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ENTITY"))),
            "Identify code entity: function, struct, variable, import",
        );

        let role = Primitive::base(
            "ROLE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ROLE"))),
            "Determine syntactic role: parameter, return type, field, attribute",
        );

        let import = Primitive::base(
            "IMPORT",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("IMPORT"))),
            "External dependency reference",
        );

        let attribute = Primitive::base(
            "ATTRIBUTE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ATTRIBUTE"))),
            "Metadata annotation on code element",
        );

        // === ENCODING PRIMITIVES ===

        let encode = Primitive::base(
            "ENCODE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ENCODE"))),
            "Convert code structure to hypervector representation",
        );

        let bind_symbol = Primitive::base(
            "BIND_SYMBOL",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("BIND_SYMBOL"))),
            "Associate identifier with meaning in code context",
        );

        let type_check = Primitive::base(
            "TYPE_CHECK",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("TYPE_CHECK"))),
            "Verify type consistency and constraints",
        );

        // === GENERATIVE PRIMITIVES ===

        let generate = Primitive::base(
            "GENERATE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("GENERATE"))),
            "Create new code from specification or pattern",
        );

        let compose = Primitive::base(
            "COMPOSE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("COMPOSE"))),
            "Combine code patterns into larger structure",
        );

        let specialize = Primitive::base(
            "SPECIALIZE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("SPECIALIZE"))),
            "Create specific instance from generic pattern",
        );

        let mutate = Primitive::base(
            "MUTATE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("MUTATE"))),
            "Transform code while preserving semantics",
        );

        // === FLOW PRIMITIVES ===

        let branch = Primitive::base(
            "BRANCH",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("BRANCH"))),
            "Conditional execution path (if/match)",
        );

        let loop_prim = Primitive::base(
            "LOOP",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("LOOP"))),
            "Iterative execution pattern (for/while/loop)",
        );

        let call = Primitive::base(
            "CALL",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("CALL"))),
            "Function or method invocation",
        );

        let return_prim = Primitive::base(
            "RETURN",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("RETURN"))),
            "Value production and control flow exit",
        );

        // === SIMILARITY & ABSTRACTION ===

        let code_similarity = Primitive::base(
            "CODE_SIMILARITY",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("CODE_SIMILARITY"))),
            "Measure semantic similarity between code patterns",
        );

        let abstract_prim = Primitive::base(
            "ABSTRACT",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("ABSTRACT"))),
            "Extract common pattern from concrete implementations",
        );

        let refactor = Primitive::base(
            "REFACTOR",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("REFACTOR"))),
            "Restructure code while preserving behavior",
        );

        // === REASONING PRIMITIVES ===

        let explain = Primitive::base(
            "EXPLAIN",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("EXPLAIN"))),
            "Describe code semantics in natural language",
        );

        let trace = Primitive::base(
            "TRACE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("TRACE"))),
            "Follow execution path through code",
        );

        let intent = Primitive::base(
            "INTENT",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("INTENT"))),
            "Infer programmer's purpose from code",
        );

        let debug = Primitive::base(
            "DEBUG",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("DEBUG"))),
            "Diagnose issues and locate errors",
        );

        let verify = Primitive::base(
            "VERIFY",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("VERIFY"))),
            "Validate code correctness against specification",
        );

        // === SEQUENCE PRIMITIVE ===

        let code_sequence = Primitive::base(
            "CODE_SEQUENCE",
            PrimitiveTier::Code,
            "code",
            code_domain.embed(BinaryHV::random(seed_from_name("CODE_SEQUENCE"))),
            "Ordered sequence of code operations",
        );

        // Register all code primitives
        let primitives = vec![
            // Structural
            parse,
            entity,
            role,
            import,
            attribute,
            // Encoding
            encode,
            bind_symbol,
            type_check,
            // Generative
            generate,
            compose,
            specialize,
            mutate,
            // Flow
            branch,
            loop_prim,
            call,
            return_prim,
            // Similarity & Abstraction
            code_similarity,
            abstract_prim,
            refactor,
            // Reasoning
            explain,
            trace,
            intent,
            debug,
            verify,
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
}
