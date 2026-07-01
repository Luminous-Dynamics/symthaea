// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Genesis Physics Demo: From Quarks to Chemistry
//!
//! This demo showcases the complete physics hierarchy built from first principles:
//!
//! ```text
//! Layer 4: H₂O, NaCl, C₆H₁₂O₆ ─────────────────────────────────── CHEMISTRY
//!              ↑
//! Layer 3: H, He, C, N, O, Fe... ──────────────────────────────── ELEMENTS
//!              ↑
//! Layer 2: Proton, Neutron ────────────────────────────────────── HADRONS
//!              ↑
//! Layer 1: Up, Down, Electron, Photon... ──────────────────────── STANDARD MODEL
//!              ↑
//! Layer 0: Genesis Seed ("E=mc²") ─────────────────────────────── AXIOM
//! ```
//!
//! ## Key Insights
//!
//! 1. **Compositional**: Elements are *built* from protons/neutrons/electrons, not memorized
//! 2. **Deterministic**: Same genesis seed → identical universe
//! 3. **Emergent Properties**: Isotopes, ions, molecules emerge from composition
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example genesis_physics_demo --release
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::chemistry::{BondType, Chemistry};
use symthaea_core::physics::{
    EnergyScale, Hadrons, NuclearPhysics, PeriodicTable, PhysicsConsciousnessBridge, StandardModel,
};

fn main() {
    println!("\n");
    println!("================================================================");
    println!("   GENESIS PHYSICS: FROM QUARKS TO CHEMISTRY");
    println!("   Building the universe from first principles");
    println!("================================================================\n");

    // Create our universe from a genesis phrase
    let genesis = GenesisSeed::from_phrase("E=mc²");
    println!("Genesis Seed: \"E=mc²\"\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 1: STANDARD MODEL");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let model = StandardModel::from_genesis(&genesis);

    // Show quark relationships
    let up_down = model.up_quark.similarity(&model.down_quark);
    let up_charm = model.up_quark.similarity(&model.charm_quark);
    println!("Quark Relationships:");
    println!("  Up ↔ Down (same generation):   {:.4}", up_down);
    println!("  Up ↔ Charm (cross generation): {:.4}", up_charm);
    println!("  → Same-generation quarks are more related\n");

    // Antiparticle demonstration
    let positron = model.antiparticle(&model.electron);
    let double = model.antiparticle(&positron);
    println!("Antiparticle Transform:");
    println!(
        "  Electron → Positron similarity: {:.4}",
        model.electron.similarity(&positron)
    );
    println!(
        "  Positron → Electron similarity: {:.4} (recovered!)",
        model.electron.similarity(&double)
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 2: HADRONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let hadrons = Hadrons::from_model(&model, &genesis);

    println!("Hadron Composition:");
    println!("  Proton  = Bundle(Up, Up, Down)");
    println!("  Neutron = Bundle(Up, Down, Down)");
    println!();

    let p_n_sim = hadrons.proton.similarity(&hadrons.neutron);
    println!("Nucleon Relationship:");
    println!(
        "  Proton ↔ Neutron: {:.4} (siblings, differ by one quark)",
        p_n_sim
    );

    // Meson composition
    let pion_plus_minus = hadrons.pion_plus.similarity(&hadrons.pion_minus);
    println!("\nMeson Structure:");
    println!(
        "  π+ (Up·anti-Down) ↔ π- (Down·anti-Up): {:.4}",
        pion_plus_minus
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 3: ELEMENTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let table = PeriodicTable::from_model(&model, &hadrons, &genesis);

    println!("Element Composition:");
    println!("  Carbon-12 = Bundle(6×Proton, 6×Neutron, 6×Electron) ⊗ Binding\n");

    // Isotope similarity
    let c12 = table.element(6).unwrap();
    let c14 = table.isotope(6, 8, &hadrons); // Carbon-14: 6 protons, 8 neutrons

    println!("Isotope Similarity (Carbon-12 vs Carbon-14):");
    println!("  Similarity: {:.4}", c12.vector.similarity(&c14));
    println!("  → Isotopes are highly similar (same element, different mass)\n");

    // Ion similarity
    let na = table.element(11).unwrap();
    let na_plus = table.ion(11, 1, &hadrons);
    let cl = table.element(17).unwrap();
    let cl_minus = table.ion(17, -1, &hadrons);

    println!("Ion Formation:");
    println!(
        "  Na ↔ Na⁺: {:.4} (lost 1 electron)",
        na.vector.similarity(&na_plus)
    );
    println!(
        "  Cl ↔ Cl⁻: {:.4} (gained 1 electron)",
        cl.vector.similarity(&cl_minus)
    );

    // Noble gas family
    let he = table.element(2).unwrap();
    let ne = table.element(10).unwrap();
    let ar = table.element(18).unwrap();
    let li = table.element(3).unwrap();

    println!("\nNoble Gas Family (Group 18):");
    println!("  He ↔ Ne: {:.4}", he.vector.similarity(&ne.vector));
    println!("  He ↔ Ar: {:.4}", he.vector.similarity(&ar.vector));
    println!(
        "  He ↔ Li: {:.4} (non-noble for comparison)",
        he.vector.similarity(&li.vector)
    );
    println!("  → Noble gases share 'full shell' character\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 4: CHEMISTRY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let chemistry = Chemistry::from_genesis(&genesis, &table);

    // Water molecule
    let water = chemistry.water(&table);
    let h = table.element(1).unwrap();
    let o = table.element(8).unwrap();

    println!("Molecule Composition:");
    println!("  H₂O = Bundle(2×H, 1×O) ⊗ covalent_bond\n");

    println!("Water Analysis:");
    println!(
        "  H₂O ↔ H: {:.4} (contains hydrogen)",
        water.vector.similarity(&h.vector)
    );
    println!(
        "  H₂O ↔ O: {:.4} (contains oxygen)",
        water.vector.similarity(&o.vector)
    );

    // Salt formation (ionic) - using compose_molecule
    let salt = chemistry.compose_molecule(
        "Sodium Chloride",
        "NaCl",
        &[(na.vector.clone(), 1), (cl.vector.clone(), 1)],
        &[BondType::Ionic],
    );

    println!("\nSalt (NaCl) - Ionic Compound:");
    println!("  NaCl ↔ Na: {:.4}", salt.vector.similarity(&na.vector));
    println!("  NaCl ↔ Cl: {:.4}", salt.vector.similarity(&cl.vector));

    // Glucose (organic)
    let glucose = chemistry.compose_molecule(
        "Glucose",
        "C6H12O6",
        &[
            (c12.vector.clone(), 6),
            (h.vector.clone(), 12),
            (o.vector.clone(), 6),
        ],
        &[BondType::Single], // Simplified - glucose has single bonds
    );

    println!("\nGlucose (C₆H₁₂O₆) - Organic:");
    println!(
        "  Glucose ↔ Carbon: {:.4}",
        glucose.vector.similarity(&c12.vector)
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("NUCLEAR PHYSICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let nuclear = NuclearPhysics::from_genesis(&genesis, &hadrons, &table);

    println!("Binding Energy Curve:");
    for mass_num in [2, 4, 12, 56, 238] {
        let be = nuclear.binding_energy_per_nucleon(mass_num);
        let name = match mass_num {
            2 => "H-2 (deuterium)",
            4 => "He-4 (alpha)",
            12 => "C-12",
            56 => "Fe-56 (PEAK)",
            238 => "U-238",
            _ => "",
        };
        println!("  A={:3}: {:.2} MeV/nucleon  {}", mass_num, be, name);
    }

    println!("\nFusion/Fission Energy:");
    let fusion_q = nuclear.fusion_q_value(2, 3, 4); // D-T fusion → He-4
    let fission_q = nuclear.fission_q_value(235, 140, 95); // U-235 fission
    println!("  D + T → He-4 + n: Q = {:.1} MeV (fusion)", fusion_q);
    println!("  U-235 → Ba + Kr: Q = {:.1} MeV (fission)", fission_q);

    println!("\nEnergy Scale Comparison:");
    println!(
        "  Chemical (1 eV):     {:.0}× baseline",
        EnergyScale::Chemical.density_ratio()
    );
    println!(
        "  Isomeric (100 keV):  {:.0}× baseline",
        EnergyScale::Isomeric.density_ratio()
    );
    println!(
        "  Nuclear (1 MeV):     {:.0}× baseline",
        EnergyScale::Nuclear.density_ratio()
    );
    println!(
        "  Relativistic (1 GeV): {:.0}× baseline",
        EnergyScale::Relativistic.density_ratio()
    );

    // Nuclear isomers
    println!("\nNuclear Isomers (metastable excited states):");
    if let Some(hf) = nuclear.get_isomer(72, 178) {
        println!(
            "  Hf-178m2: {:.0} keV excitation, {:.1} year half-life",
            hf.excitation_kev,
            hf.half_life_s.unwrap() / (365.25 * 24.0 * 3600.0)
        );
    }
    if let Some(tc) = nuclear.get_isomer(43, 99) {
        println!(
            "  Tc-99m:   {:.0} keV excitation, {:.1} hour half-life (medical imaging)",
            tc.excitation_kev,
            tc.half_life_s.unwrap() / 3600.0
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CONSCIOUSNESS BRIDGE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let bridge = PhysicsConsciousnessBridge::from_genesis(&genesis);

    println!("Phenomenal Index (physics concepts):");
    println!("  (Range: -1 = functional, +1 = phenomenal)\n");

    let electron_phen = bridge.estimate_phenomenal_index(&model.electron);
    let water_phen = bridge.estimate_phenomenal_index(&water.vector);
    let proton_phen = bridge.estimate_phenomenal_index(&hadrons.proton);

    println!("  Electron: {:+.4}", electron_phen);
    println!("  Proton:   {:+.4}", proton_phen);
    println!("  Water:    {:+.4}", water_phen);

    println!("\nCompositional Emergence Analysis:");
    let analysis = bridge.compositional_phenomenal_analysis(&model, &hadrons, &table, &chemistry);
    println!("  Level 0 (Quarks):    {:+.4}", analysis.level_0_quarks);
    println!("  Level 1 (Hadrons):   {:+.4}", analysis.level_1_hadrons);
    println!("  Level 2 (Atoms):     {:+.4}", analysis.level_2_atoms);
    println!("  Level 3 (Molecules): {:+.4}", analysis.level_3_molecules);
    println!(
        "  Emergence gradient:  {:+.4}/level",
        analysis.emergence_gradient()
    );

    // Embodied qualia test
    let embodied = bridge.embody_qualia(&model.electron);
    let embodied_phen = bridge.estimate_phenomenal_index(&embodied);
    println!("\nEmbodiment Effect:");
    println!("  Bare electron:    {:+.4}", electron_phen);
    println!("  Embodied qualia:  {:+.4}", embodied_phen);
    println!("  Δ phenomenal:     {:+.4}", embodied_phen - electron_phen);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DETERMINISM VERIFICATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create another universe with same seed
    let genesis2 = GenesisSeed::from_phrase("E=mc²");
    let model2 = StandardModel::from_genesis(&genesis2);
    let hadrons2 = Hadrons::from_model(&model2, &genesis2);
    let table2 = PeriodicTable::from_model(&model2, &hadrons2, &genesis2);

    let c1 = table.element(6).unwrap();
    let c2 = table2.element(6).unwrap();

    println!("Same Seed Test:");
    println!(
        "  Carbon (universe 1) ↔ Carbon (universe 2): {:.6}",
        c1.vector.similarity(&c2.vector)
    );
    println!("  → Identical universes from same genesis!\n");

    // Different seed comparison
    let genesis3 = GenesisSeed::from_phrase("Quantum Universe");
    let model3 = StandardModel::from_genesis(&genesis3);
    let hadrons3 = Hadrons::from_model(&model3, &genesis3);
    let table3 = PeriodicTable::from_model(&model3, &hadrons3, &genesis3);
    let c3 = table3.element(6).unwrap();

    println!("Different Seed Test:");
    println!(
        "  Carbon (E=mc²) ↔ Carbon (Quantum Universe): {:.4}",
        c1.vector.similarity(&c3.vector)
    );
    println!("  → Different universes have different physics!\n");

    println!("================================================================");
    println!("   DEMO COMPLETE: Physics hierarchy fully operational");
    println!("================================================================\n");
}
