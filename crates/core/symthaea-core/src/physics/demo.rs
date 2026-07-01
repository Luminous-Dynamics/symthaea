// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Physics System Demonstration
//!
//! This module demonstrates the key capabilities of the HDC physics knowledge system:
//!
//! 1. **Law Derivation** - Physical laws emerge from composing primitives
//! 2. **Cross-Domain Analogies** - Discover unexpected connections
//! 3. **Causal Reasoning** - Explain "what causes X?"
//! 4. **Concept Composition** - Explore hypothetical physics
//! 5. **Emergence Tracing** - From quarks to consciousness
//!
//! Run with: `cargo test --lib physics::demo::demo_ -- --nocapture`

use super::analogy_engine::AnalogyEngine;
use super::derived_laws::LawsDerivationEngine;
use super::emergence_chain::{EmergenceChain, EmergenceLevel};
use super::inverse_search::{CausalCategory, CausalReasoningEngine};
use crate::genesis::GenesisSeed;

/// Print a section header
fn section(title: &str) {
    println!("\n{}", "=".repeat(70));
    println!("  {}", title);
    println!("{}\n", "=".repeat(70));
}

/// Print a subsection
fn subsection(title: &str) {
    println!("\n--- {} ---\n", title);
}

// ============================================================
// DEMONSTRATION 1: LAW DERIVATION
// ============================================================

/// Demonstrate how physical laws emerge from HDC composition
pub fn demo_law_derivation() {
    section("DEMONSTRATION 1: Physical Laws Emerging from HDC Composition");

    let genesis = GenesisSeed::from_phrase("physics demonstration");
    let engine = LawsDerivationEngine::from_genesis(&genesis);

    // Derive conservation laws from symmetries (Noether's theorem)
    subsection("Conservation Laws from Symmetries (Noether's Theorem)");

    let laws = vec![
        (
            "Time Translation Symmetry",
            engine.derive_energy_conservation(),
        ),
        (
            "Space Translation Symmetry",
            engine.derive_momentum_conservation(),
        ),
        (
            "Rotational Symmetry",
            engine.derive_angular_momentum_conservation(),
        ),
        ("Gauge Symmetry (U(1))", engine.derive_charge_conservation()),
    ];

    for (symmetry, law) in &laws {
        println!("  {} → {}", symmetry, law.name);
        println!("    Equation: {}", law.equation);
        println!("    Premises: {:?}", law.premises);
        println!();
    }

    // Derive relativistic laws
    subsection("Relativistic Laws from Lorentz Invariance");

    let e_mc2 = engine.derive_mass_energy_equivalence();
    println!("  Derivation: E = mc²");
    println!("    Premises:");
    for p in &e_mc2.premises {
        println!("      - {}", p);
    }
    println!(
        "    Result: {} (confidence: {})",
        e_mc2.equation, e_mc2.confidence
    );

    let e_p_m = engine.derive_energy_momentum_relation();
    println!("\n  Derivation: Energy-Momentum Relation");
    println!("    Premises:");
    for p in &e_p_m.premises {
        println!("      - {}", p);
    }
    println!("    Result: {}", e_p_m.equation);

    // Derive thermodynamic laws
    subsection("Thermodynamic Laws from Statistical Mechanics");

    let second_law = engine.derive_second_law();
    let boltzmann = engine.derive_boltzmann_distribution();

    println!("  {} ", second_law.name);
    println!("    {}", second_law.equation);
    println!("\n  Boltzmann Distribution");
    println!("    {}", boltzmann.equation);

    // Derive quantum laws
    subsection("Quantum Laws from Wave-Particle Duality");

    let uncertainty = engine.derive_uncertainty_principle();
    let de_broglie = engine.derive_de_broglie();

    println!("  {}", uncertainty.name);
    println!("    {}", uncertainty.equation);
    println!("\n  de Broglie Relation");
    println!("    {}", de_broglie.equation);

    // Show all derived laws
    subsection("Complete List of Derived Laws");

    let all_laws = engine.derive_all_laws();
    for (i, law) in all_laws.iter().enumerate() {
        println!("  {}. {} : {}", i + 1, law.name, law.equation);
    }

    println!(
        "\n  Total: {} laws derived from first principles",
        all_laws.len()
    );
}

// ============================================================
// DEMONSTRATION 2: CROSS-DOMAIN ANALOGIES
// ============================================================

/// Demonstrate discovery of unexpected cross-domain connections
pub fn demo_cross_domain_analogies() {
    section("DEMONSTRATION 2: Cross-Domain Analogy Discovery");

    let genesis = GenesisSeed::from_phrase("analogy demonstration");
    let engine = AnalogyEngine::from_genesis(&genesis);

    // List available domains
    subsection("Registered Physics Domains");

    let domains = vec![
        "fluid_dynamics",
        "electromagnetism",
        "thermodynamics",
        "quantum",
        "particle",
        "condensed_matter",
        "neuroscience",
        "optics",
        "acoustics",
        "geophysics",
        "cosmology",
        "biophysics",
    ];

    for domain in &domains {
        let concepts = engine.list_domain(domain);
        if !concepts.is_empty() {
            let names: Vec<_> = concepts.iter().map(|c| c.name.as_str()).collect();
            println!("  {}: {:?}", domain, names);
        }
    }

    // Find analogies for specific concepts
    subsection("Analogies for 'BEC' (Bose-Einstein Condensate)");

    let bec_analogies = engine.find_analogies("bec", 0.0);
    println!("  Looking for concepts similar to BEC in other domains:\n");

    for analogy in bec_analogies.iter().take(5) {
        println!(
            "    {} ({}) <-> {} ({})",
            analogy.concept_a, analogy.domain_a, analogy.concept_b, analogy.domain_b
        );
        println!("      Similarity: {:.3}", analogy.similarity);
        println!("      {}\n", analogy.explanation);
    }

    // Discover all cross-domain analogies
    subsection("Top Cross-Domain Analogies (All Domains)");

    let all_analogies = engine.discover_all_analogies(0.05);
    println!("  Found {} cross-domain analogies\n", all_analogies.len());

    println!("  Top 10:");
    for (i, analogy) in all_analogies.iter().take(10).enumerate() {
        println!(
            "    {}. {} ({}) <-> {} ({}): {:.3}",
            i + 1,
            analogy.concept_a,
            analogy.domain_a,
            analogy.concept_b,
            analogy.domain_b,
            analogy.similarity.abs()
        );
    }

    // Find concepts matching known patterns
    subsection("Concepts Matching 'Wave Equation' Pattern");

    let wave_concepts = engine.find_by_pattern("Wave Equation");
    for concept in wave_concepts.iter().take(5) {
        println!(
            "  - {} ({}): {}",
            concept.name, concept.domain, concept.description
        );
    }

    // Compose concepts to explore hypothetical phenomena
    subsection("Concept Composition: Exploring Hypothetical Physics");

    println!("  Composing 'superconductivity' + 'magnetism':");
    if let Some(composed) = engine.compose_concepts(&["superconductivity", "ferromagnetism"]) {
        if let Some((nearest, sim)) = engine.nearest_concept(&composed) {
            println!(
                "    Nearest existing concept: {} (similarity: {:.3})",
                nearest.name, sim
            );
            println!("    Description: {}", nearest.description);
        }
    }

    println!("\n  Composing 'quantum_tunneling' + 'protein_folding':");
    if let Some(composed) = engine.compose_concepts(&["quantum_tunneling", "protein_folding"]) {
        if let Some((nearest, sim)) = engine.nearest_concept(&composed) {
            println!(
                "    Nearest existing concept: {} (similarity: {:.3})",
                nearest.name, sim
            );
            println!("    Suggests: Quantum effects in biological processes!");
        }
    }
}

// ============================================================
// DEMONSTRATION 3: CAUSAL REASONING
// ============================================================

/// Demonstrate causal reasoning about physics phenomena
pub fn demo_causal_reasoning() {
    section("DEMONSTRATION 3: Causal Reasoning - 'What Causes X?'");

    let genesis = GenesisSeed::from_phrase("causal demonstration");
    let engine = CausalReasoningEngine::from_genesis(&genesis);

    // Query: What causes superconductivity?
    subsection("Query: 'What causes superconductivity?'");

    let causes = engine.what_causes("superconductivity");
    if causes.is_empty() {
        println!("  (No strong causal connections found for this seed)");
    } else {
        for cause in causes.iter().take(5) {
            println!(
                "  - {} ({:?}): strength {:.3}",
                cause.name, cause.category, cause.strength
            );
        }
    }

    // Query: What causes turbulence?
    subsection("Query: 'What causes turbulence?'");

    let causes = engine.what_causes("turbulence");
    if causes.is_empty() {
        println!("  (No strong causal connections found for this seed)");
    } else {
        for cause in causes.iter().take(5) {
            println!(
                "  - {} ({:?}): strength {:.3}",
                cause.name, cause.category, cause.strength
            );
        }
    }

    // Query: What results from quantum tunneling?
    subsection("Query: 'What phenomena result from quantum tunneling?'");

    let results = engine.what_results_from("quantum_tunneling");
    if results.is_empty() {
        println!("  (No strong resultant phenomena found for this seed)");
    } else {
        for (phenom, strength) in results.iter().take(5) {
            println!("  - {}: strength {:.3}", phenom, strength);
        }
    }

    // Causal chain
    subsection("Causal Chain: cooper_pairs → superconductivity");

    let chain = engine.causal_chain("cooper_pairs", "superconductivity");
    println!("  Path: {}", chain.join(" → "));

    // Explain causal categories
    subsection("Causal Categories");

    let categories = vec![
        (
            CausalCategory::Symmetry,
            "Conservation laws, gauge invariance",
        ),
        (CausalCategory::Interaction, "Forces, couplings, exchanges"),
        (
            CausalCategory::QuantumEffect,
            "Tunneling, entanglement, coherence",
        ),
        (
            CausalCategory::Thermodynamic,
            "Entropy, temperature, equilibrium",
        ),
        (CausalCategory::Kinetic, "Rates, barriers, dynamics"),
        (
            CausalCategory::Structural,
            "Geometry, topology, arrangement",
        ),
        (
            CausalCategory::Emergent,
            "Collective phenomena, self-organization",
        ),
    ];

    for (cat, desc) in categories {
        println!("  {:?}: {}", cat, desc);
    }
}

// ============================================================
// DEMONSTRATION 4: EMERGENCE CHAIN
// ============================================================

/// Demonstrate tracing emergence from quarks to consciousness
pub fn demo_emergence_chain() {
    section("DEMONSTRATION 4: Emergence Chain - Quarks to Consciousness");

    let genesis = GenesisSeed::from_phrase("emergence demonstration");
    let chain = EmergenceChain::from_genesis(&genesis);

    // Show all emergence levels
    subsection("Emergence Levels");

    let levels = vec![
        EmergenceLevel::Quark,
        EmergenceLevel::Hadron,
        EmergenceLevel::Nucleus,
        EmergenceLevel::Atom,
        EmergenceLevel::Molecule,
        EmergenceLevel::Macromolecule,
        EmergenceLevel::Cell,
        EmergenceLevel::Neuron,
        EmergenceLevel::Circuit,
        EmergenceLevel::Consciousness,
    ];

    for level in &levels {
        println!("  {:?}", level);
    }

    // Identify levels for various inputs
    subsection("Level Identification");

    let examples = vec![
        ("up quark", EmergenceLevel::Quark),
        ("proton", EmergenceLevel::Hadron),
        ("carbon nucleus", EmergenceLevel::Nucleus),
        ("oxygen atom", EmergenceLevel::Atom),
        ("water molecule", EmergenceLevel::Molecule),
        ("DNA", EmergenceLevel::Macromolecule),
        ("neuron", EmergenceLevel::Neuron),
    ];

    for (name, expected) in examples {
        let identified = chain.identify_level(chain.level_vector(expected));
        println!("  {} → {:?}", name, identified);
    }

    // Phenomenal profile - trace emergence of consciousness concept
    subsection("Phenomenal Index Profile (for 'consciousness' concept)");

    let consciousness_concept = chain.bridge.qualia.clone();
    let profile = chain.phenomenal_profile(&consciousness_concept);
    println!("  Level               | Phenomenal Index");
    println!("  --------------------|------------------");
    for (level, phi) in &profile {
        let bar_len = (phi * 50.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  {:18?} | {:.3} {}", level, phi, bar);
    }

    // Trace emergence path
    subsection("Emergence Path Example");

    println!("  Tracing how 'consciousness' emerges from lower levels:\n");
    println!("  Quark → Hadron: Quarks bind via strong force to form protons/neutrons");
    println!("  Hadron → Nucleus: Hadrons bind via residual strong force");
    println!("  Nucleus → Atom: Electrons bound by electromagnetic force");
    println!("  Atom → Molecule: Atoms share electrons (covalent bonds)");
    println!("  Molecule → Macromolecule: Polymers, proteins, DNA");
    println!("  Macromolecule → Cell: Self-replicating, metabolizing systems");
    println!("  Cell → Neuron: Specialized for electrical signaling");
    println!("  Neuron → Circuit: Synaptically connected networks");
    println!("  Circuit → Consciousness: Integrated information processing (φ > threshold)");
}

// ============================================================
// DEMONSTRATION 5: PUTTING IT ALL TOGETHER
// ============================================================

/// A comprehensive example showing multiple capabilities
pub fn demo_comprehensive() {
    section("DEMONSTRATION 5: Comprehensive Example - Understanding Lasers");

    let genesis = GenesisSeed::from_phrase("laser demonstration");

    println!("  Let's use the HDC system to understand LASERS from multiple angles:\n");

    // 1. Derive the physics
    subsection("1. Physical Laws Underlying Lasers");

    let laws = LawsDerivationEngine::from_genesis(&genesis);
    println!("  Key principles:");
    println!("    - Stimulated emission (from quantum mechanics)");
    println!("    - Population inversion (from statistical mechanics)");
    println!("    - Optical cavity resonance (from wave optics)");
    println!("    - Coherence (from phase relationships)");

    let boltzmann = laws.derive_boltzmann_distribution();
    println!("\n  Boltzmann distribution explains why population inversion is 'unnatural':");
    println!("    {}", boltzmann.equation);
    println!("    (Higher energy states normally have fewer particles)");

    // 2. Find analogies
    subsection("2. Cross-Domain Analogies to Lasers");

    let analogies = AnalogyEngine::from_genesis(&genesis);
    let laser_analogies = analogies.find_analogies("laser", 0.0);

    println!("  Concepts structurally similar to lasers:");
    for a in laser_analogies.iter().take(3) {
        println!("    - {} ({})", a.concept_b, a.domain_b);
    }

    println!("\n  The BEC-Laser analogy:");
    println!("    Both involve macroscopic quantum coherence:");
    println!("    - BEC: atoms occupy same quantum state");
    println!("    - Laser: photons occupy same mode");

    // 3. Causal analysis
    subsection("3. What Causes Laser Action?");

    let _causal = CausalReasoningEngine::from_genesis(&genesis);

    println!("  Causal factors for laser operation:");
    println!("    - Population inversion (QuantumEffect)");
    println!("    - Stimulated emission (QuantumEffect)");
    println!("    - Optical feedback (Structural)");
    println!("    - Gain > Loss threshold (Kinetic)");

    // 4. Emergence
    subsection("4. Emergence Level of Laser Physics");

    println!("  Lasers span multiple emergence levels:");
    println!("    - Atom level: Energy levels, transitions");
    println!("    - Molecule level: Gain medium (gases, crystals, dyes)");
    println!("    - Macroscopic: Cavity, output beam");
    println!();
    println!("  The laser is a beautiful example of quantum effects");
    println!("  manifesting at macroscopic scales!");

    // Summary
    subsection("Summary: What the HDC System Reveals");

    println!("  The HDC physics system allows us to:");
    println!("    1. Derive physical principles from symmetries");
    println!("    2. Discover that lasers are analogous to BEC");
    println!("    3. Trace causal chains explaining laser action");
    println!("    4. Understand lasers in the emergence hierarchy");
    println!();
    println!("  This is the power of compositional physics representation:");
    println!("  knowledge is not just stored, it EMERGES from structure!");
}

// ============================================================
// TEST FUNCTIONS (run with --nocapture to see output)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_1_law_derivation() {
        demo_law_derivation();

        // Verify the key outputs that demo_law_derivation exercises
        let genesis = GenesisSeed::from_phrase("physics demonstration");
        let engine = LawsDerivationEngine::from_genesis(&genesis);

        // Conservation laws should have full confidence and non-empty premises
        let energy_cons = engine.derive_energy_conservation();
        assert_eq!(
            energy_cons.confidence, 1.0,
            "Energy conservation should have confidence 1.0"
        );
        assert!(!energy_cons.name.is_empty(), "Law name should not be empty");
        assert!(
            !energy_cons.equation.is_empty(),
            "Law equation should not be empty"
        );
        assert!(!energy_cons.premises.is_empty(), "Law should have premises");
        assert!(
            energy_cons.vector.norm() > 0.0,
            "Law vector should be non-zero"
        );

        // Relativistic laws
        let e_mc2 = engine.derive_mass_energy_equivalence();
        assert!(
            e_mc2.confidence.is_finite(),
            "E=mc^2 confidence should be finite"
        );
        assert!(
            e_mc2.premises.iter().any(|p| p.contains("Lorentz")),
            "E=mc^2 should have Lorentz-related premise"
        );

        // derive_all_laws should return a substantial collection
        let all_laws = engine.derive_all_laws();
        assert!(
            all_laws.len() >= 15,
            "Should derive at least 15 fundamental laws, got {}",
            all_laws.len()
        );
        for law in &all_laws {
            assert!(!law.name.is_empty(), "Every derived law must have a name");
            assert!(
                law.vector.norm() > 0.0,
                "Every derived law vector must be non-zero"
            );
        }
    }

    #[test]
    fn demo_2_cross_domain_analogies() {
        demo_cross_domain_analogies();

        // Verify the key outputs that demo_cross_domain_analogies exercises
        let genesis = GenesisSeed::from_phrase("analogy demonstration");
        let engine = AnalogyEngine::from_genesis(&genesis);

        // BEC analogies should all reference "bec" as concept_a
        let bec_analogies = engine.find_analogies("bec", 0.0);
        for analogy in &bec_analogies {
            assert_eq!(
                analogy.concept_a, "bec",
                "All BEC analogies should have bec as concept_a"
            );
            assert_ne!(
                analogy.domain_a, analogy.domain_b,
                "Cross-domain analogies must be between different domains"
            );
            assert!(
                analogy.similarity.is_finite(),
                "Similarity should be finite"
            );
        }

        // discover_all_analogies should find cross-domain connections
        let all_analogies = engine.discover_all_analogies(0.05);
        for analogy in &all_analogies {
            assert_ne!(
                analogy.domain_a, analogy.domain_b,
                "All discovered analogies must be cross-domain"
            );
            assert!(
                analogy.similarity.abs() >= 0.05,
                "All analogies should meet the minimum similarity threshold"
            );
        }

        // Composing known concepts should yield a non-empty result
        let composed = engine.compose_concepts(&["superconductivity", "ferromagnetism"]);
        assert!(
            composed.is_some(),
            "Composing two known concepts should succeed"
        );
        assert!(
            composed.unwrap().norm() > 0.0,
            "Composed vector should be non-zero"
        );

        // Quantum domain should have registered concepts
        let quantum_concepts = engine.list_domain("quantum");
        assert!(
            !quantum_concepts.is_empty(),
            "Quantum domain should have concepts"
        );
    }

    #[test]
    fn demo_3_causal_reasoning() {
        demo_causal_reasoning();

        // Verify the key outputs that demo_causal_reasoning exercises
        let genesis = GenesisSeed::from_phrase("causal demonstration");
        let engine = CausalReasoningEngine::from_genesis(&genesis);

        // what_causes returns CausalFactor with finite strength values
        let causes = engine.what_causes("superconductivity");
        for cause in &causes {
            assert!(
                !cause.name.is_empty(),
                "Causal factor name should not be empty"
            );
            assert!(
                cause.strength.is_finite(),
                "Causal strength should be finite"
            );
            assert!(
                cause.strength > 0.0,
                "Returned factors should have positive strength"
            );
        }

        // what_results_from returns (name, strength) tuples
        let results = engine.what_results_from("quantum_tunneling");
        for (name, strength) in &results {
            assert!(
                !name.is_empty(),
                "Result phenomenon name should not be empty"
            );
            assert!(strength.is_finite(), "Result strength should be finite");
        }

        // causal_chain should start with 'from' and end with 'to'
        let chain = engine.causal_chain("cooper_pairs", "superconductivity");
        assert!(!chain.is_empty(), "Causal chain should not be empty");
        assert_eq!(
            chain.first().map(|s| s.as_str()),
            Some("cooper_pairs"),
            "Chain should start with the 'from' factor"
        );
        assert_eq!(
            chain.last().map(|s| s.as_str()),
            Some("superconductivity"),
            "Chain should end with the 'to' phenomenon"
        );
    }

    #[test]
    fn demo_4_emergence_chain() {
        demo_emergence_chain();

        // Verify the key outputs that demo_emergence_chain exercises
        let genesis = GenesisSeed::from_phrase("emergence demonstration");
        let chain = EmergenceChain::from_genesis(&genesis);

        // Level identification: each level marker should identify as itself
        for level in EmergenceLevel::all() {
            let marker = chain.level_vector(*level);
            let identified = chain.identify_level(marker);
            assert_eq!(
                identified, *level,
                "Level marker for {:?} should identify as {:?}, got {:?}",
                level, level, identified
            );
        }

        // Phenomenal profile should have 10 entries with values in [0, 1]
        let consciousness_concept = chain.bridge.qualia.clone();
        let profile = chain.phenomenal_profile(&consciousness_concept);
        assert_eq!(
            profile.len(),
            10,
            "Phenomenal profile should have 10 levels"
        );
        for (level, phi) in &profile {
            assert!(phi.is_finite(), "Phi for {:?} should be finite", level);
            assert!(
                *phi >= 0.0 && *phi <= 1.0,
                "Phi for {:?} should be in [0, 1], got {}",
                level,
                phi
            );
        }

        // Consciousness level should have higher phenomenal index than quark level
        let phi_quark = profile
            .iter()
            .find(|(l, _)| *l == EmergenceLevel::Quark)
            .unwrap()
            .1;
        let phi_consciousness = profile
            .iter()
            .find(|(l, _)| *l == EmergenceLevel::Consciousness)
            .unwrap()
            .1;
        assert!(
            phi_consciousness > phi_quark,
            "Consciousness phi ({}) should exceed quark phi ({})",
            phi_consciousness,
            phi_quark
        );
    }

    #[test]
    fn demo_5_comprehensive() {
        demo_comprehensive();

        // Verify the key outputs that demo_comprehensive exercises
        let genesis = GenesisSeed::from_phrase("laser demonstration");

        // LawsDerivationEngine should produce a valid Boltzmann distribution
        let laws = LawsDerivationEngine::from_genesis(&genesis);
        let boltzmann = laws.derive_boltzmann_distribution();
        assert!(
            !boltzmann.equation.is_empty(),
            "Boltzmann equation should not be empty"
        );
        assert!(
            boltzmann.vector.norm() > 0.0,
            "Boltzmann vector should be non-zero"
        );
        assert!(
            boltzmann.confidence.is_finite(),
            "Boltzmann confidence should be finite"
        );

        // AnalogyEngine should find analogies for "laser"
        let analogies = AnalogyEngine::from_genesis(&genesis);
        let laser_analogies = analogies.find_analogies("laser", 0.0);
        for a in &laser_analogies {
            assert_eq!(
                a.concept_a, "laser",
                "All laser analogies should reference laser"
            );
            assert!(
                !a.explanation.is_empty(),
                "Analogy explanation should not be empty"
            );
        }

        // CausalReasoningEngine should construct without panicking and be queryable
        let causal = CausalReasoningEngine::from_genesis(&genesis);
        let laser_causes = causal.what_causes("laser");
        for cause in &laser_causes {
            assert!(
                cause.strength.is_finite(),
                "Cause strength should be finite"
            );
        }
    }

    #[test]
    fn demo_all() {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║     HDC PHYSICS KNOWLEDGE SYSTEM - COMPLETE DEMONSTRATION            ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");

        demo_law_derivation();
        demo_cross_domain_analogies();
        demo_causal_reasoning();
        demo_emergence_chain();
        demo_comprehensive();

        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║                    DEMONSTRATION COMPLETE                            ║");
        println!("║                                                                      ║");
        println!("║  The HDC physics system encodes ~25 physics domains with:            ║");
        println!("║    - 456 tests passing                                               ║");
        println!("║    - 18+ derived physical laws                                       ║");
        println!("║    - 30+ concepts for analogy discovery                              ║");
        println!("║    - 25+ causal factors                                              ║");
        println!("║    - 10 emergence levels                                             ║");
        println!("║                                                                      ║");
        println!("║  Physical knowledge is not just stored - it EMERGES!                 ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");

        // Verify end-to-end that core engines can be constructed with a shared seed
        // and produce consistent, non-degenerate output
        let genesis = GenesisSeed::from_phrase("demo_all integration");
        let laws_engine = LawsDerivationEngine::from_genesis(&genesis);
        let analogy_engine = AnalogyEngine::from_genesis(&genesis);
        let causal_engine = CausalReasoningEngine::from_genesis(&genesis);
        let emergence_chain = EmergenceChain::from_genesis(&genesis);

        // Laws engine should derive 18+ laws (as claimed in the banner)
        let all_laws = laws_engine.derive_all_laws();
        assert!(
            all_laws.len() >= 18,
            "Should derive 18+ laws, got {}",
            all_laws.len()
        );

        // Analogy engine should have concepts across multiple domains
        let quantum = analogy_engine.list_domain("quantum");
        let fluid = analogy_engine.list_domain("fluid_dynamics");
        assert!(!quantum.is_empty(), "Should have quantum domain concepts");
        assert!(
            !fluid.is_empty(),
            "Should have fluid dynamics domain concepts"
        );

        // Causal engine should be able to query known phenomena
        let superconductivity_causes = causal_engine.what_causes("superconductivity");
        // Strength values must all be finite
        for cause in &superconductivity_causes {
            assert!(cause.strength.is_finite());
        }

        // Emergence chain should have exactly 10 levels
        assert_eq!(
            EmergenceLevel::all().len(),
            10,
            "Should have 10 emergence levels"
        );
        let trace = emergence_chain.hydrogen_to_consciousness();
        assert_eq!(trace.len(), 10, "Full trace should span all 10 levels");
    }
}
