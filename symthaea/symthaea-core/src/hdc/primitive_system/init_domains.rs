// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Domain-specific primitive initialization (gap analysis additions).
//!
//! This module contains initialization for domain-specific primitives
//! identified through ontological gap analysis:
//! - Biological/Organic primitives
//! - Emotional/Affective primitives
//! - Ecological/Systems primitives
//! - Quantum/Fundamental Physics primitives
//! - Economic/Value primitives
//! - Linguistic/Semiotic primitives
//! - Social/Moral primitives
//! - Institutional/Geopolitical primitives

use super::{
    BindingRule, DomainManifold, Primitive, PrimitiveSystem, PrimitiveTier, seed_from_name,
};
use crate::hdc::binary_hv::BinaryHV;

impl PrimitiveSystem {
    /// Initialize Biological/Organic Primitives (Gap Analysis Priority 1)
    /// Ground AI in biological reality - metabolism, growth, evolution, homeostasis
    pub(super) fn init_biological_primitives(&mut self) {
        let biology_domain = DomainManifold::new(
            "biology",
            PrimitiveTier::Physical,
            "Biological processes and organic life principles",
        );

        // === CORE BIOLOGICAL PROCESSES ===

        let metabolism = Primitive::base(
            "METABOLISM",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("METABOLISM"))),
            "Process: energy transformation in living systems (ATP synthesis, glycolysis)",
        );

        let growth = Primitive::base(
            "GROWTH",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("GROWTH"))),
            "Process: increase in size, complexity through cell division and development",
        );

        let reproduction = Primitive::base(
            "REPRODUCTION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("REPRODUCTION"))),
            "Process: creation of new organisms, transmission of heredity",
        );

        let evolution = Primitive::base(
            "EVOLUTION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("EVOLUTION"))),
            "Process: change in heritable characteristics over generations",
        );

        let adaptation = Primitive::derived(
            "ADAPTATION",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("ADAPTATION"))),
            "Process: adjustment to environment through selection",
            "EVOLUTION ⊗ ENVIRONMENT",
        );

        let homeostasis_dynamic = Primitive::base(
            "HOMEOSTASIS_DYNAMIC",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("HOMEOSTASIS_DYNAMIC"))),
            "Process: dynamic self-regulation maintaining stable internal conditions",
        );

        let symbiosis = Primitive::base(
            "SYMBIOSIS",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("SYMBIOSIS"))),
            "Relationship: close interaction between different organisms (mutualism, parasitism)",
        );

        let immune_response = Primitive::base(
            "IMMUNE_RESPONSE",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("IMMUNE_RESPONSE"))),
            "Process: self/non-self distinction, pathogen recognition and elimination",
        );

        let circadian_rhythm = Primitive::base(
            "CIRCADIAN_RHYTHM",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("CIRCADIAN_RHYTHM"))),
            "Pattern: ~24-hour biological cycles, internal timekeeping",
        );

        let morphogen = Primitive::base(
            "MORPHOGEN",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("MORPHOGEN"))),
            "Substance: concentration gradient guiding development and pattern formation",
        );

        let apoptosis = Primitive::base(
            "APOPTOSIS",
            PrimitiveTier::Physical,
            "biology",
            biology_domain.embed(BinaryHV::random(seed_from_name("APOPTOSIS"))),
            "Process: programmed cell death, controlled system renewal",
        );

        // Register domain and primitives
        self.domains.insert("biology".to_string(), biology_domain);

        for primitive in vec![
            metabolism,
            growth,
            reproduction,
            evolution,
            adaptation,
            homeostasis_dynamic,
            symbiosis,
            immune_response,
            circadian_rhythm,
            morphogen,
            apoptosis,
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
            example: "HOMEOSTASIS_DYNAMIC ⊗ FEEDBACK → self-regulating biological system"
                .to_string(),
        });
    }

    /// Initialize Emotional/Affective Primitives (Gap Analysis Priority 2)
    /// Ground emotional reasoning for endocrine system integration
    pub(super) fn init_emotional_primitives(&mut self) {
        let emotion_domain = DomainManifold::new(
            "emotion",
            PrimitiveTier::MetaCognitive,
            "Affective states and emotional processing",
        );

        // === DIMENSIONAL MODEL ===

        let valence = Primitive::base(
            "AFFECTIVE_VALENCE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("AFFECTIVE_VALENCE"))),
            "Dimension: positive/negative affect, pleasantness/unpleasantness",
        );

        let arousal = Primitive::base(
            "AFFECTIVE_AROUSAL",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("AFFECTIVE_AROUSAL"))),
            "Dimension: activation level, calm/excited continuum",
        );

        // === BASIC EMOTIONS (Ekman) ===

        let joy = Primitive::derived(
            "JOY",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("JOY"))),
            "Emotion: positive valence, high arousal - happiness, pleasure",
            "VALENCE_POSITIVE ⊗ AROUSAL_MODERATE",
        );

        let sadness = Primitive::derived(
            "SADNESS",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("SADNESS"))),
            "Emotion: negative valence, low arousal - loss, disappointment",
            "VALENCE_NEGATIVE ⊗ AROUSAL_LOW",
        );

        let fear = Primitive::derived(
            "FEAR",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("FEAR"))),
            "Emotion: negative valence, high arousal - threat response",
            "VALENCE_NEGATIVE ⊗ AROUSAL_HIGH",
        );

        let anger = Primitive::derived(
            "ANGER",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("ANGER"))),
            "Emotion: negative valence, high arousal - obstacle response",
            "VALENCE_NEGATIVE ⊗ AROUSAL_HIGH ⊗ APPROACH",
        );

        let disgust = Primitive::derived(
            "DISGUST",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("DISGUST"))),
            "Emotion: negative valence, rejection response - contamination avoidance",
            "VALENCE_NEGATIVE ⊗ REJECTION",
        );

        let surprise = Primitive::derived(
            "SURPRISE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("SURPRISE"))),
            "Emotion: neutral valence, high arousal - unexpected event response",
            "AROUSAL_HIGH ⊗ NOVELTY",
        );

        // === SOCIAL EMOTIONS ===

        let empathy = Primitive::base(
            "EMPATHY",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("EMPATHY"))),
            "Capacity: shared emotional experience, feeling with others",
        );

        let attachment = Primitive::base(
            "ATTACHMENT",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("ATTACHMENT"))),
            "Bond: emotional connection, social bonding, relationship formation",
        );

        let awe = Primitive::derived(
            "AWE",
            PrimitiveTier::MetaCognitive,
            "emotion",
            emotion_domain.embed(BinaryHV::random(seed_from_name("AWE"))),
            "Emotion: vastness + accommodation - wonder at something greater",
            "VASTNESS ⊗ ACCOMMODATION",
        );

        // Register domain and primitives
        self.domains.insert("emotion".to_string(), emotion_domain);

        for primitive in vec![
            valence, arousal, joy, sadness, fear, anger, disgust, surprise, empathy, attachment,
            awe,
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
    pub(super) fn init_ecological_primitives(&mut self) {
        let ecology_domain = DomainManifold::new(
            "ecology",
            PrimitiveTier::Physical,
            "Complex adaptive systems and ecosystem dynamics",
        );

        // === ECOLOGICAL CONCEPTS ===

        let niche = Primitive::base(
            "NICHE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("NICHE"))),
            "Concept: environmental role, opportunity and constraint space",
        );

        let carrying_capacity = Primitive::base(
            "CARRYING_CAPACITY",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("CARRYING_CAPACITY"))),
            "Limit: maximum population sustainable by available resources",
        );

        let succession = Primitive::base(
            "SUCCESSION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("SUCCESSION"))),
            "Process: sequential ecosystem development, progressive change",
        );

        let trophic_level = Primitive::base(
            "TROPHIC_LEVEL",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("TROPHIC_LEVEL"))),
            "Structure: position in energy flow chain (producer, consumer, decomposer)",
        );

        let resilience = Primitive::base(
            "RESILIENCE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("RESILIENCE"))),
            "Property: system capacity to absorb disturbance and reorganize",
        );

        // === SYSTEMS DYNAMICS ===

        let feedback_loop_positive = Primitive::base(
            "FEEDBACK_LOOP_POSITIVE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("FEEDBACK_LOOP_POSITIVE"))),
            "Pattern: amplifying feedback, exponential growth or collapse",
        );

        let feedback_loop_negative = Primitive::base(
            "FEEDBACK_LOOP_NEGATIVE",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("FEEDBACK_LOOP_NEGATIVE"))),
            "Pattern: dampening feedback, stabilization, homeostasis",
        );

        let emergence_strong = Primitive::base(
            "EMERGENCE_STRONG",
            PrimitiveTier::MetaCognitive,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("EMERGENCE_STRONG"))),
            "Property: irreducible higher-level properties from component interactions",
        );

        let attractor = Primitive::base(
            "ATTRACTOR",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("ATTRACTOR"))),
            "State: stable system configuration in phase space",
        );

        let bifurcation = Primitive::base(
            "BIFURCATION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("BIFURCATION"))),
            "Transition: qualitative system change at critical parameter value",
        );

        let phase_transition = Primitive::base(
            "PHASE_TRANSITION",
            PrimitiveTier::Physical,
            "ecology",
            ecology_domain.embed(BinaryHV::random(seed_from_name("PHASE_TRANSITION"))),
            "Change: abrupt qualitative state transformation (order/disorder)",
        );

        // Register domain and primitives
        self.domains.insert("ecology".to_string(), ecology_domain);

        for primitive in vec![
            niche,
            carrying_capacity,
            succession,
            trophic_level,
            resilience,
            feedback_loop_positive,
            feedback_loop_negative,
            emergence_strong,
            attractor,
            bifurcation,
            phase_transition,
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
    /// Foundation for consciousness theories and IIT (Phi)
    pub(super) fn init_quantum_primitives(&mut self) {
        let quantum_domain = DomainManifold::new(
            "quantum",
            PrimitiveTier::Physical,
            "Quantum mechanics and fundamental physics",
        );

        // === QUANTUM PHENOMENA ===

        let superposition = Primitive::base(
            "SUPERPOSITION",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("SUPERPOSITION"))),
            "State: system existing in multiple configurations simultaneously",
        );

        let entanglement = Primitive::base(
            "ENTANGLEMENT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("ENTANGLEMENT"))),
            "Correlation: non-local quantum correlation between systems",
        );

        let measurement = Primitive::base(
            "MEASUREMENT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("MEASUREMENT"))),
            "Process: observer-dependent state collapse, wave function reduction",
        );

        let uncertainty_heisenberg = Primitive::base(
            "UNCERTAINTY_HEISENBERG",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("UNCERTAINTY_HEISENBERG"))),
            "Principle: fundamental limit on simultaneous knowledge (DxDp >= h/2)",
        );

        let wave_particle_duality = Primitive::base(
            "WAVE_PARTICLE_DUALITY",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("WAVE_PARTICLE_DUALITY"))),
            "Property: complementary wave and particle aspects of quantum objects",
        );

        let planck_constant = Primitive::base(
            "PLANCK_CONSTANT",
            PrimitiveTier::Physical,
            "quantum",
            quantum_domain.embed(BinaryHV::random(seed_from_name("PLANCK_CONSTANT"))),
            "Constant: quantum of action (h = 6.626e-34 J*s), fundamental scale",
        );

        // Register domain and primitives
        self.domains.insert("quantum".to_string(), quantum_domain);

        for primitive in [
            superposition,
            entanglement,
            measurement,
            uncertainty_heisenberg,
            wave_particle_duality,
            planck_constant,
        ] {
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
            example: "ENTANGLEMENT ⊗ INTEGRATED_INFORMATION → quantum consciousness theories"
                .to_string(),
        });
    }

    /// Initialize Economic/Value Primitives
    /// Reason about value, exchange, scarcity - supports ATP economy (Hearth)
    pub(super) fn init_economic_primitives(&mut self) {
        let economics_domain = DomainManifold::new(
            "economics",
            PrimitiveTier::Strategic,
            "Value, exchange, and resource allocation",
        );

        // === ECONOMIC CONCEPTS ===

        let scarcity = Primitive::base(
            "SCARCITY",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("SCARCITY"))),
            "Condition: limited availability relative to demand",
        );

        let supply = Primitive::base(
            "SUPPLY",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("SUPPLY"))),
            "Quantity: amount available at given price",
        );

        let demand = Primitive::base(
            "DEMAND",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("DEMAND"))),
            "Quantity: amount desired at given price",
        );

        let exchange = Primitive::base(
            "EXCHANGE",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("EXCHANGE"))),
            "Transaction: trading goods/services, reciprocal transfer",
        );

        let value_subjective = Primitive::base(
            "VALUE_SUBJECTIVE",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("VALUE_SUBJECTIVE"))),
            "Property: preference-dependent worth, individual utility",
        );

        let capital = Primitive::base(
            "CAPITAL",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("CAPITAL"))),
            "Resource: accumulated assets for production, stored value",
        );

        let debt = Primitive::base(
            "DEBT",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("DEBT"))),
            "Obligation: future claim, deferred payment",
        );

        let trust_economic = Primitive::base(
            "TRUST_ECONOMIC",
            PrimitiveTier::Strategic,
            "economics",
            economics_domain.embed(BinaryHV::random(seed_from_name("TRUST_ECONOMIC"))),
            "Property: reliability expectation, reputation, social capital",
        );

        // Register domain and primitives
        self.domains
            .insert("economics".to_string(), economics_domain);

        for primitive in [
            scarcity,
            supply,
            demand,
            exchange,
            value_subjective,
            capital,
            debt,
            trust_economic,
        ] {
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
    pub(super) fn init_linguistic_primitives(&mut self) {
        let linguistics_domain = DomainManifold::new(
            "linguistics",
            PrimitiveTier::MetaCognitive,
            "Symbols, meaning, and communication",
        );

        // === SEMIOTIC CONCEPTS ===

        let sign = Primitive::base(
            "SIGN",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("SIGN"))),
            "Structure: signifier + signified, symbol and its meaning",
        );

        let reference = Primitive::base(
            "REFERENCE",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("REFERENCE"))),
            "Relation: symbol -> object mapping, denotation",
        );

        let context_dependency = Primitive::base(
            "CONTEXT_DEPENDENCY",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("CONTEXT_DEPENDENCY"))),
            "Property: meaning varies with situational context",
        );

        let metaphor = Primitive::base(
            "METAPHOR",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("METAPHOR"))),
            "Mapping: cross-domain conceptual transfer, analogical reasoning",
        );

        let syntax = Primitive::base(
            "SYNTAX",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("SYNTAX"))),
            "Structure: compositional rules, grammatical organization",
        );

        let semantics = Primitive::base(
            "SEMANTICS",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("SEMANTICS"))),
            "Property: meaning relations, truth conditions",
        );

        let pragmatics = Primitive::base(
            "PRAGMATICS",
            PrimitiveTier::MetaCognitive,
            "linguistics",
            linguistics_domain.embed(BinaryHV::random(seed_from_name("PRAGMATICS"))),
            "Use: meaning in communicative context, speaker intention",
        );

        // Register domain and primitives
        self.domains
            .insert("linguistics".to_string(), linguistics_domain);

        for primitive in [
            sign,
            reference,
            context_dependency,
            metaphor,
            syntax,
            semantics,
            pragmatics,
        ] {
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
    pub(super) fn init_social_moral_primitives(&mut self) {
        let moral_domain = DomainManifold::new(
            "morality",
            PrimitiveTier::Strategic,
            "Ethics, norms, and moral reasoning",
        );

        // === DEONTIC CONCEPTS (Rules) ===

        let norm = Primitive::base(
            "NORM",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("NORM"))),
            "Rule: social expectation, behavioral standard",
        );

        let obligation = Primitive::base(
            "OBLIGATION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("OBLIGATION"))),
            "Duty: moral requirement, what must be done",
        );

        let permission = Primitive::base(
            "PERMISSION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("PERMISSION"))),
            "Allowance: action that may be done without violation",
        );

        let prohibition = Primitive::base(
            "PROHIBITION",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("PROHIBITION"))),
            "Restriction: forbidden action, taboo",
        );

        // === MORAL FOUNDATIONS ===

        let fairness = Primitive::base(
            "FAIRNESS",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("FAIRNESS"))),
            "Principle: equitable distribution, reciprocity, justice",
        );

        let harm = Primitive::base(
            "HARM",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("HARM"))),
            "Concept: damage, suffering, negative impact on well-being",
        );

        let care = Primitive::base(
            "CARE",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("CARE"))),
            "Disposition: protection, nurturance, compassion",
        );

        let rights = Primitive::base(
            "RIGHTS",
            PrimitiveTier::Strategic,
            "morality",
            moral_domain.embed(BinaryHV::random(seed_from_name("RIGHTS"))),
            "Entitlement: claims, protections, freedoms",
        );

        // Register domain and primitives
        self.domains.insert("morality".to_string(), moral_domain);

        for primitive in [
            norm,
            obligation,
            permission,
            prohibition,
            fairness,
            harm,
            care,
            rights,
        ] {
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

    /// Initialize Institutional/Geopolitical Primitives
    ///
    /// Models sociological entities (nation-states, institutions, legal systems) as
    /// **environmental constraints and causal objects**, NOT as foundational or supreme
    /// primitives. Nation-states are composite hypervectors formed by binding underlying
    /// sociological concepts — they are "drivers, not kernel."
    ///
    /// ## Design Philosophy
    ///
    /// Nation-states are shared human conventions with very real physical consequences.
    /// They are modeled with the same causal rigor as physical phenomena: a firewall
    /// deterministically blocks packets, sanctions freeze assets at the banking layer,
    /// export controls physically prevent shipments.
    ///
    /// ## Base Primitives (irreducible atoms of institutional reality)
    ///
    /// - AUTHORITY: Recognized capacity to direct action within a scope
    /// - LEGITIMACY: Socially constructed basis for authority acceptance
    /// - SOVEREIGNTY: Exclusive authority claim over a territory
    /// - ENFORCEMENT: Capacity to impose consequences for rule violations
    /// - JURISDICTION: Scope within which authority applies (spatial/temporal/domain)
    /// - POPULATION: Collective of agents subject to a governance structure
    /// - MONOPOLY: Exclusive control over a resource or capability
    /// - COMPLIANCE: Conformity with rules imposed by an authority
    /// - TREATY: Binding agreement between sovereign entities
    /// - SANCTION: Punitive constraint imposed by authority on target
    ///
    /// ## Derived Composites (in init_derived)
    ///
    /// - TERRITORY = SPACE ⊗ BOUNDARY ⊗ SOVEREIGNTY
    /// - INSTITUTION = NORM ⊗ AUTHORITY ⊗ PERSIST
    /// - LAW = NORM ⊗ ENFORCEMENT ⊗ JURISDICTION
    /// - TAXATION = OBLIGATION ⊗ AUTHORITY ⊗ EXCHANGE
    /// - REGULATION = CONSTRAINT ⊗ LAW ⊗ COMPLIANCE
    /// - FIAT_CURRENCY = VALUE_SUBJECTIVE ⊗ AUTHORITY ⊗ TRUST_ECONOMIC ⊗ MONOPOLY
    /// - NATION_STATE = SOVEREIGNTY ⊗ INSTITUTION ⊗ ENFORCEMENT ⊗ POPULATION
    /// - FAILED_STATE = NATION_STATE with LEGITIMACY → 0 (decomposition analysis)
    /// - BORDER_DISPUTE = SOVEREIGNTY ⊗ OVERLAPS (topological overlap of claims)
    pub(super) fn init_institutional_primitives(&mut self) {
        let institutional_domain = DomainManifold::new(
            "institutional",
            PrimitiveTier::Strategic,
            "Institutional structures, geopolitical entities, and legal systems",
        );

        // === AUTHORITY & LEGITIMACY ===
        // The two irreducible components of institutional power.
        // Authority without legitimacy = tyranny. Legitimacy without authority = aspiration.

        let authority = Primitive::base(
            "AUTHORITY",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("AUTHORITY"))),
            "Capacity: recognized power to direct action within a scope",
        );

        let legitimacy = Primitive::base(
            "LEGITIMACY",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("LEGITIMACY"))),
            "Property: socially constructed basis for authority acceptance (consent, tradition, charisma, rational-legal)",
        );

        // === SOVEREIGNTY & JURISDICTION ===
        // Spatial and domain-scoped authority claims.

        let sovereignty = Primitive::base(
            "SOVEREIGNTY",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("SOVEREIGNTY"))),
            "Claim: exclusive supreme authority over a bounded domain (Westphalian principle)",
        );

        let jurisdiction = Primitive::base(
            "JURISDICTION",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("JURISDICTION"))),
            "Scope: spatial, temporal, or domain bounds within which authority applies",
        );

        // === ENFORCEMENT & COMPLIANCE ===
        // The coercive and cooperative dimensions of institutional power.

        let enforcement = Primitive::base(
            "ENFORCEMENT",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("ENFORCEMENT"))),
            "Capacity: ability to impose consequences for rule violations (fines, imprisonment, sanctions)",
        );

        let compliance = Primitive::base(
            "COMPLIANCE",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("COMPLIANCE"))),
            "State: conformity with rules imposed by an authority (GDPR, HIPAA, FATF)",
        );

        // === COLLECTIVE ENTITIES ===

        let population = Primitive::base(
            "POPULATION",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("POPULATION"))),
            "Collective: agents subject to a common governance structure",
        );

        let monopoly = Primitive::base(
            "MONOPOLY",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("MONOPOLY"))),
            "Structure: exclusive control over a resource, capability, or market",
        );

        // === INTER-INSTITUTIONAL RELATIONS ===

        let treaty = Primitive::base(
            "TREATY",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("TREATY"))),
            "Agreement: binding compact between sovereign entities (mutual obligations)",
        );

        let sanction = Primitive::base(
            "SANCTION",
            PrimitiveTier::Strategic,
            "institutional",
            institutional_domain.embed(BinaryHV::random(seed_from_name("SANCTION"))),
            "Constraint: punitive measure imposed by authority on target (trade, financial, diplomatic)",
        );

        // Register domain and primitives
        self.domains
            .insert("institutional".to_string(), institutional_domain);

        for primitive in [
            authority,
            legitimacy,
            sovereignty,
            jurisdiction,
            enforcement,
            compliance,
            population,
            monopoly,
            treaty,
            sanction,
        ] {
            let name = primitive.name.clone();
            let tier = primitive.tier;
            self.primitives.insert(name.clone(), primitive);
            self.by_tier.entry(tier).or_default().push(name);
        }

        // Binding rules
        self.binding_rules.push(BindingRule {
            name: "institutional_structure".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Strategic],
            result_tier: PrimitiveTier::Strategic,
            example: "AUTHORITY ⊗ LEGITIMACY → legitimate governance".to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "institutional_constraint".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Physical],
            result_tier: PrimitiveTier::Strategic,
            example: "SOVEREIGNTY ⊗ SPACE → territorial claim (causal constraint on physical infrastructure)"
                .to_string(),
        });

        self.binding_rules.push(BindingRule {
            name: "institutional_geometric".to_string(),
            pattern: vec![PrimitiveTier::Strategic, PrimitiveTier::Geometric],
            result_tier: PrimitiveTier::Strategic,
            example: "JURISDICTION ⊗ BOUNDARY → jurisdictional boundary (regulatory zone)"
                .to_string(),
        });
    }
}
