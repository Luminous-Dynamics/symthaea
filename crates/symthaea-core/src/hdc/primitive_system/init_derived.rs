// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Derived primitive initialization for the Primitive System.
//!
//! Contains `init_derived_primitives` which creates complex primitives
//! by composing base primitives via two-pass dependency-aware resolution.

use super::{BindingRule, DomainManifold, Primitive, PrimitiveSystem, PrimitiveTier};

impl PrimitiveSystem {
    /// Initialize derived primitives using dependency-aware two-pass resolution.
    ///
    /// These are complex primitives derived from base primitives via composition.
    /// Rather than expanding the base set, we compose existing primitives to create
    /// higher-order concepts. The two-pass approach processes derivations in rounds:
    /// each round registers all specs whose parents are available, then repeats
    /// until no more can be resolved. This eliminates silent fallback to random.
    pub(super) fn init_derived_primitives(&mut self) {
        // === DOMAIN SETUP ===

        let uncertainty_domain = DomainManifold::new(
            "uncertainty",
            PrimitiveTier::Mathematical,
            "Probabilistic reasoning and uncertainty quantification",
        );
        self.domains
            .insert("uncertainty".to_string(), uncertainty_domain.clone());

        let physics_ext_domain = DomainManifold::new(
            "physics_extended",
            PrimitiveTier::Physical,
            "Advanced physical concepts for embodied reasoning",
        );
        self.domains
            .insert("physics_extended".to_string(), physics_ext_domain.clone());

        let info_domain = DomainManifold::new(
            "information_theory",
            PrimitiveTier::Mathematical,
            "Quantitative theory of information and communication",
        );
        self.domains
            .insert("information_theory".to_string(), info_domain.clone());

        let consciousness_domain = DomainManifold::new(
            "consciousness_derived",
            PrimitiveTier::MetaCognitive,
            "Derived primitives for consciousness measurement",
        );
        self.domains.insert(
            "consciousness_derived".to_string(),
            consciousness_domain.clone(),
        );

        // Institutional domain — for derived geopolitical/sociological composites.
        // Uses the existing "institutional" domain registered by init_institutional_primitives.
        // We create a local clone for the DerivationSpec closures below.
        let institutional_domain =
            self.domains
                .get("institutional")
                .cloned()
                .unwrap_or_else(|| {
                    DomainManifold::new(
                        "institutional",
                        PrimitiveTier::Strategic,
                        "Institutional structures, geopolitical entities, and legal systems",
                    )
                });

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
                name: "PROBABILITY",
                parents: vec!["RATIO", "CERTAINTY"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Measure of likelihood: P(A) in [0,1], derived from ratio of favorable to total outcomes",
                derivation_expr: "RATIO ^ CERTAINTY",
            },
            DerivationSpec {
                name: "EXPECTED_VALUE",
                parents: vec!["PROBABILITY", "VALUE"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Probability-weighted average: E[X] = sum P(x) * V(x)",
                derivation_expr: "PROBABILITY ^ VALUE",
            },
            DerivationSpec {
                name: "SHANNON_ENTROPY",
                parents: vec!["PROBABILITY", "INFORMATION"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Information-theoretic uncertainty: H = -sum P(x) log P(x), higher = more uncertain",
                derivation_expr: "PROBABILITY ^ INFORMATION",
            },
            DerivationSpec {
                name: "BAYESIAN_UPDATE",
                parents: vec!["PROBABILITY", "EVIDENCE"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Belief revision: P(H|E) = P(E|H) * P(H) / P(E)",
                derivation_expr: "PROBABILITY ^ EVIDENCE",
            },
            DerivationSpec {
                name: "VARIANCE",
                parents: vec!["EXPECTED_VALUE", "DEVIATION"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "uncertainty",
                domain: uncertainty_domain.clone(),
                definition: "Spread of distribution: Var(X) = E[(X - mu)^2]",
                derivation_expr: "EXPECTED_VALUE ^ DEVIATION",
            },
            // Physics Extensions
            DerivationSpec {
                name: "CONSERVATION_LAW",
                parents: vec!["STATE_CHANGE", "CONSERVATION"],
                tier: PrimitiveTier::Physical,
                domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Formal conservation law: dQ/dt = 0, invariant quantity across transformations",
                derivation_expr: "STATE_CHANGE ^ CONSERVATION",
            },
            DerivationSpec {
                name: "GRADIENT",
                parents: vec!["DIFFERENTIATION", "SPACE"],
                tier: PrimitiveTier::Physical,
                domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Spatial rate of change: grad f = (df/dx, df/dy, df/dz)",
                derivation_expr: "DIFFERENTIATION ^ SPACE",
            },
            DerivationSpec {
                name: "FIELD",
                parents: vec!["FORCE", "POINT"],
                tier: PrimitiveTier::Physical,
                domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Assignment of force/value to each point: F(x, y, z)",
                derivation_expr: "FORCE ^ POINT",
            },
            DerivationSpec {
                name: "WAVE",
                parents: vec!["OSCILLATION", "PROPAGATION"],
                tier: PrimitiveTier::Physical,
                domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Propagating oscillation: psi(x,t) = A sin(kx - wt)",
                derivation_expr: "OSCILLATION ^ PROPAGATION",
            },
            DerivationSpec {
                name: "EQUILIBRIUM",
                parents: vec!["FORCE", "CONSERVATION"],
                tier: PrimitiveTier::Physical,
                domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Balanced state: sum F = 0, stable or unstable",
                derivation_expr: "FORCE ^ CONSERVATION",
            },
            DerivationSpec {
                name: "POTENTIAL",
                parents: vec!["ENERGY", "POINT"],
                tier: PrimitiveTier::Physical,
                domain_name: "physics_extended",
                domain: physics_ext_domain.clone(),
                definition: "Position-dependent energy: U(x) where F = -grad U",
                derivation_expr: "ENERGY ^ POINT",
            },
            // Information Theory
            DerivationSpec {
                name: "MUTUAL_INFORMATION",
                parents: vec!["SHANNON_ENTROPY", "MEMBERSHIP"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Shared information: I(X;Y) = H(X) + H(Y) - H(X,Y)",
                derivation_expr: "SHANNON_ENTROPY ^ MEMBERSHIP",
            },
            DerivationSpec {
                name: "INFORMATION_GAIN",
                parents: vec!["SHANNON_ENTROPY", "EVIDENCE"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Entropy reduction from evidence: IG = H(S) - H(S|E)",
                derivation_expr: "SHANNON_ENTROPY ^ EVIDENCE",
            },
            DerivationSpec {
                name: "CHANNEL_CAPACITY",
                parents: vec!["INFORMATION", "LIMIT"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Maximum transmission rate: C = max I(X;Y)",
                derivation_expr: "INFORMATION ^ LIMIT",
            },
            DerivationSpec {
                name: "COMPRESSION",
                parents: vec!["INFORMATION", "EFFICIENCY"],
                tier: PrimitiveTier::Mathematical,
                domain_name: "information_theory",
                domain: info_domain.clone(),
                definition: "Efficient encoding: L >= H(X) (Shannon's source coding theorem)",
                derivation_expr: "INFORMATION ^ EFFICIENCY",
            },
            // Consciousness
            DerivationSpec {
                name: "INTEGRATED_INFORMATION",
                parents: vec!["MUTUAL_INFORMATION", "SELF"],
                tier: PrimitiveTier::MetaCognitive,
                domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Consciousness measure: Phi = integrated information above MIP",
                derivation_expr: "MUTUAL_INFORMATION ^ SELF",
            },
            DerivationSpec {
                name: "CAUSAL_POWER",
                parents: vec!["CAUSE", "EFFECT", "COUNTERFACTUAL"],
                tier: PrimitiveTier::MetaCognitive,
                domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Capacity to produce effects: P(effect|do(cause)) - P(effect)",
                derivation_expr: "CAUSE ^ EFFECT ^ COUNTERFACTUAL",
            },
            DerivationSpec {
                name: "ATTENTION",
                parents: vec!["SALIENCE", "SELECTION"],
                tier: PrimitiveTier::MetaCognitive,
                domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Selective processing: focus on salient subset of available information",
                derivation_expr: "SALIENCE ^ SELECTION",
            },
            DerivationSpec {
                name: "METACOGNITION",
                parents: vec!["INTROSPECTION", "SELF"],
                tier: PrimitiveTier::MetaCognitive,
                domain_name: "consciousness_derived",
                domain: consciousness_domain.clone(),
                definition: "Cognition about cognition: awareness of mental processes",
                derivation_expr: "INTROSPECTION ^ SELF",
            },
            // === Institutional/Geopolitical Composites ===
            // Nation-states and institutions are composite concepts, NOT irreducible primes.
            // They decompose into sociological atoms — removing a component yields meaningful
            // analysis (e.g., remove LEGITIMACY from AUTHORITY → revolution scenario).
            DerivationSpec {
                name: "TERRITORY",
                parents: vec!["SPACE", "BOUNDARY", "SOVEREIGNTY"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Bounded spatial region under sovereign authority claim",
                derivation_expr: "SPACE ^ BOUNDARY ^ SOVEREIGNTY",
            },
            DerivationSpec {
                name: "INSTITUTION",
                parents: vec!["NORM", "AUTHORITY", "PERSIST"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Persistent normative structure with recognized authority (survives agent turnover)",
                derivation_expr: "NORM ^ AUTHORITY ^ PERSIST",
            },
            DerivationSpec {
                name: "LAW",
                parents: vec!["NORM", "ENFORCEMENT", "JURISDICTION"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Enforceable norm within a jurisdiction (codified obligation with consequences)",
                derivation_expr: "NORM ^ ENFORCEMENT ^ JURISDICTION",
            },
            DerivationSpec {
                name: "TAXATION",
                parents: vec!["OBLIGATION", "AUTHORITY", "EXCHANGE"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Compulsory transfer of value from population to authority",
                derivation_expr: "OBLIGATION ^ AUTHORITY ^ EXCHANGE",
            },
            DerivationSpec {
                name: "REGULATION",
                parents: vec!["LAW", "COMPLIANCE"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Rules governing conduct within a domain, requiring conformity (GDPR, HIPAA, FATF)",
                derivation_expr: "LAW ^ COMPLIANCE",
            },
            DerivationSpec {
                name: "FIAT_CURRENCY",
                parents: vec![
                    "VALUE_SUBJECTIVE",
                    "AUTHORITY",
                    "TRUST_ECONOMIC",
                    "MONOPOLY",
                ],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Medium of exchange backed by sovereign authority (not intrinsic value)",
                derivation_expr: "VALUE_SUBJECTIVE ^ AUTHORITY ^ TRUST_ECONOMIC ^ MONOPOLY",
            },
            DerivationSpec {
                name: "NATION_STATE",
                parents: vec!["SOVEREIGNTY", "INSTITUTION", "ENFORCEMENT", "POPULATION"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Sovereign institutional structure with enforcement over a population (Westphalian entity)",
                derivation_expr: "SOVEREIGNTY ^ INSTITUTION ^ ENFORCEMENT ^ POPULATION",
            },
            DerivationSpec {
                name: "DIPLOMATIC_RELATION",
                parents: vec!["TREATY", "RECIPROCATE", "SOVEREIGNTY"],
                tier: PrimitiveTier::Strategic,
                domain_name: "institutional",
                domain: institutional_domain.clone(),
                definition: "Mutual recognition and negotiated cooperation between sovereign entities",
                derivation_expr: "TREATY ^ RECIPROCATE ^ SOVEREIGNTY",
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
                let all_parents_available = spec
                    .parents
                    .iter()
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
                    self.by_tier
                        .entry(spec.tier)
                        .or_default()
                        .push(spec.name.to_string());
                    resolved_this_round.push(spec.name);
                } else {
                    still_pending.push(spec);
                }
            }

            if resolved_this_round.is_empty() {
                // No progress -- log unresolved specs as warnings
                for spec in &still_pending {
                    let missing: Vec<&&str> = spec
                        .parents
                        .iter()
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
}
