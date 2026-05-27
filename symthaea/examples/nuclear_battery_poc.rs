// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Nuclear Battery Proof of Concept
//!
//! Demonstrates the energy storage potential of nuclear isomers compared to
//! chemical batteries, using Symthaea's compositional physics framework.
//!
//! ## The Promise of Nuclear Batteries
//!
//! Nuclear isomers store energy in excited nuclear states:
//! - Hf-178m2: 2.4 MeV excitation, 31-year half-life
//! - Could theoretically store 100,000× more energy than chemical bonds
//!
//! ## Energy Hierarchy
//!
//! ```text
//! Storage Type          Energy Density    Example
//! ──────────────────────────────────────────────────
//! Chemical              ~1 eV/bond        Li-ion battery
//! Nuclear Isomer        ~100 keV-MeV      Hf-178m2
//! Nuclear Fission       ~200 MeV/atom     U-235
//! Antimatter            ~938 MeV/nucleon  p + anti-p
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example nuclear_battery_poc --release
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{EnergyScale, Hadrons, NuclearPhysics, PeriodicTable, StandardModel};

/// Battery specifications
struct BatterySpec {
    name: &'static str,
    energy_ev: f64,
    mass_kg: f64,
    description: &'static str,
}

impl BatterySpec {
    fn energy_density_wh_kg(&self) -> f64 {
        // Convert eV to Wh (1 eV = 1.6e-19 J, 1 Wh = 3600 J)
        let joules = self.energy_ev * 1.6e-19;
        let wh = joules / 3600.0;
        wh / self.mass_kg
    }

    #[allow(dead_code)]
    fn energy_density_j_kg(&self) -> f64 {
        let joules = self.energy_ev * 1.6e-19;
        joules / self.mass_kg
    }
}

fn main() {
    println!("\n");
    println!("================================================================");
    println!("   NUCLEAR BATTERY PROOF OF CONCEPT");
    println!("   Energy storage across physics scales");
    println!("================================================================\n");

    // Create our physics universe
    let genesis = GenesisSeed::from_phrase("Nuclear Battery Research");
    let model = StandardModel::from_genesis(&genesis);
    let hadrons = Hadrons::from_model(&model, &genesis);
    let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
    let nuclear = NuclearPhysics::from_genesis(&genesis, &hadrons, &table);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ENERGY SCALE COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Physical Scales:");
    for scale in [
        EnergyScale::Chemical,
        EnergyScale::Isomeric,
        EnergyScale::Nuclear,
        EnergyScale::Relativistic,
    ] {
        println!(
            "  {:12} {:12.0} eV ({:.0}× chemical)",
            format!("{:?}:", scale),
            scale.typical_ev(),
            scale.density_ratio()
        );
    }

    // Vector representations
    println!("\nVector Space Encodings:");
    let chem_vec = nuclear.encode_energy(EnergyScale::Chemical, 1.0);
    let iso_vec = nuclear.encode_energy(EnergyScale::Isomeric, 100_000.0);
    let nuc_vec = nuclear.encode_energy(EnergyScale::Nuclear, 1_000_000.0);

    println!(
        "  Chemical ↔ Isomeric: {:.4}",
        chem_vec.similarity(&iso_vec)
    );
    println!(
        "  Chemical ↔ Nuclear:  {:.4}",
        chem_vec.similarity(&nuc_vec)
    );
    println!("  Isomeric ↔ Nuclear:  {:.4}", iso_vec.similarity(&nuc_vec));
    println!("  → Energy scales form distinct but related concepts\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("NUCLEAR ISOMER CANDIDATES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Known High-Energy Isomers:\n");

    // Hafnium-178m2
    if let Some(hf) = nuclear.get_isomer(72, 178) {
        let half_life_years = hf.half_life_s.unwrap() / (365.25 * 24.0 * 3600.0);
        let energy_mev = hf.excitation_kev / 1000.0;
        println!("  Hf-178m2 (Hafnium-178 metastable state 2):");
        println!(
            "    Excitation energy: {:.2} MeV ({:.0} keV)",
            energy_mev, hf.excitation_kev
        );
        println!("    Half-life: {:.1} years", half_life_years);
        println!("    Status: Most studied isomer for energy storage");
        println!("    Challenge: Triggered release mechanism\n");
    }

    // Tantalum-180m
    if let Some(ta) = nuclear.get_isomer(73, 180) {
        let half_life_years = ta.half_life_s.unwrap() / (365.25 * 24.0 * 3600.0);
        println!("  Ta-180m (Tantalum-180 metastable):");
        println!("    Excitation energy: {:.0} keV", ta.excitation_kev);
        println!(
            "    Half-life: >10^{:.0} years (practically stable)",
            (half_life_years).log10()
        );
        println!("    Status: Only naturally-occurring nuclear isomer");
        println!("    Significance: Proof that nuclear batteries are stable\n");
    }

    // Technetium-99m
    if let Some(tc) = nuclear.get_isomer(43, 99) {
        let half_life_hours = tc.half_life_s.unwrap() / 3600.0;
        println!("  Tc-99m (Technetium-99 metastable):");
        println!("    Excitation energy: {:.0} keV", tc.excitation_kev);
        println!("    Half-life: {:.1} hours", half_life_hours);
        println!("    Status: Most widely used medical isotope");
        println!("    Application: 30+ million diagnostic procedures/year\n");
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BATTERY COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Compare different battery technologies
    let batteries = vec![
        BatterySpec {
            name: "Li-ion Cell",
            // ~3.7V * 200 Ah/kg = 740 Wh/kg → ~2.7e25 eV/kg
            energy_ev: 2.7e25,
            mass_kg: 1.0,
            description: "Standard lithium-ion battery",
        },
        BatterySpec {
            name: "Gasoline",
            // ~46 MJ/kg → ~2.9e26 eV/kg
            energy_ev: 2.9e26,
            mass_kg: 1.0,
            description: "Chemical fuel (combustion)",
        },
        BatterySpec {
            name: "Hf-178m2",
            // 2.4 MeV per nucleus, ~1.7e24 atoms/kg (178 g/mol)
            energy_ev: 2.4e6 * 1.7e24,
            mass_kg: 1.0,
            description: "Nuclear isomer battery (theoretical)",
        },
        BatterySpec {
            name: "U-235 Fission",
            // ~200 MeV per fission, ~2.6e24 atoms/kg
            energy_ev: 200e6 * 2.6e24,
            mass_kg: 1.0,
            description: "Nuclear reactor fuel",
        },
    ];

    println!("Energy Density Comparison (per kg):\n");
    println!(
        "{:<16} {:>15} {:>10} Type",
        "Battery", "Energy (Wh/kg)", "Relative"
    );
    println!("{}", "-".repeat(70));

    let baseline = batteries[0].energy_density_wh_kg();
    for bat in &batteries {
        let density = bat.energy_density_wh_kg();
        let relative = density / baseline;
        println!(
            "{:<16} {:>15.2e} {:>10.0}× {}",
            bat.name, density, relative, bat.description
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("VECTOR SPACE ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Stability concept similarity
    println!("Stability Analysis:\n");
    let hf_isomer = nuclear.get_isomer(72, 178).unwrap();
    let ta_isomer = nuclear.get_isomer(73, 180).unwrap();
    let tc_isomer = nuclear.get_isomer(43, 99).unwrap();

    println!("Isomer Vector Relationships:");
    println!(
        "  Hf-178m2 ↔ Ta-180m: {:.4}",
        hf_isomer.vector.similarity(&ta_isomer.vector)
    );
    println!(
        "  Hf-178m2 ↔ Tc-99m:  {:.4}",
        hf_isomer.vector.similarity(&tc_isomer.vector)
    );
    println!(
        "  Ta-180m ↔ Tc-99m:   {:.4}",
        ta_isomer.vector.similarity(&tc_isomer.vector)
    );

    println!("\nStability Concept Correlation:");
    println!(
        "  Hf-178m2 ↔ Stable:     {:.4}",
        hf_isomer.vector.similarity(&nuclear.stable)
    );
    println!(
        "  Hf-178m2 ↔ Metastable: {:.4}",
        hf_isomer.vector.similarity(&nuclear.metastable)
    );
    println!(
        "  Hf-178m2 ↔ Radioactive:{:.4}",
        hf_isomer.vector.similarity(&nuclear.radioactive)
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("THEORETICAL NUCLEAR BATTERY DESIGN");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Hf-178m2 Nuclear Battery Concept:\n");

    // Calculate theoretical battery specs
    let hf_excitation_ev = 2.446e6; // 2.446 MeV in eV
    let hf_atoms_per_gram = 6.022e23 / 178.0; // Avogadro / atomic mass
    let total_energy_ev = hf_excitation_ev * hf_atoms_per_gram;
    let total_energy_j = total_energy_ev * 1.6e-19;
    let total_energy_wh = total_energy_j / 3600.0;

    println!("  Specifications (1 gram Hf-178m2):");
    println!(
        "    Stored energy:     {:.2e} MeV",
        hf_excitation_ev * hf_atoms_per_gram / 1e6
    );
    println!("    Stored energy:     {:.2} kWh", total_energy_wh / 1000.0);
    println!(
        "    Energy density:    {:.2e} Wh/kg",
        total_energy_wh * 1000.0
    );
    println!(
        "    vs Li-ion:         {:.0}× improvement",
        total_energy_wh * 1000.0 / 250.0
    );

    println!("\n  Challenges:");
    println!("    1. Triggered release: Need controlled de-excitation mechanism");
    println!("    2. Manufacturing:     Producing pure Hf-178m2 is difficult");
    println!("    3. Gamma shielding:   2.4 MeV gamma rays require heavy shielding");
    println!("    4. Conversion:        Converting gamma to electricity efficiently");

    println!("\n  Potential Applications:");
    println!("    - Deep space probes (decade-long power source)");
    println!("    - Remote sensors (no maintenance required)");
    println!("    - Medical implants (compact, long-lasting)");
    println!("    - Emergency power (dormant until needed)");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FUSION/FISSION COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Compare energy release mechanisms
    println!("Nuclear Reaction Energy:\n");

    let dt_fusion = nuclear.fusion_q_value(2, 3, 4);
    let u235_fission = nuclear.fission_q_value(235, 140, 95);

    println!("  D-T Fusion (D + T → He-4 + n):");
    println!("    Q-value: {:.1} MeV per reaction", dt_fusion);
    println!("    Status:  Under development (ITER, NIF)");

    println!("\n  U-235 Fission (n + U-235 → fragments + 2-3n):");
    println!("    Q-value: {:.1} MeV per reaction", u235_fission);
    println!("    Status:  Mature technology (nuclear power plants)");

    println!("\n  Hf-178m2 Isomeric Transition:");
    println!("    Q-value: 2.4 MeV per nucleus");
    println!("    Status:  Theoretical (triggered release unproven)");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BINDING ENERGY INSIGHT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Why Fe-56 is special (most stable nucleus):\n");

    println!("  Mass Number     BE/nucleon (MeV)    Element");
    println!("  ─────────────────────────────────────────────");
    for a in [2, 4, 12, 26, 56, 62, 92, 238] {
        let be = nuclear.binding_energy_per_nucleon(a);
        let name = match a {
            2 => "H-2",
            4 => "He-4",
            12 => "C-12",
            26 => "Fe-56 (approach)",
            56 => "Fe-56 (PEAK)",
            62 => "Ni-62 (close second)",
            92 => "U-92",
            238 => "U-238",
            _ => "",
        };
        println!("  {:>10}      {:>10.2}          {}", a, be, name);
    }

    println!("\n  Insight: Fusion (light → heavy) and fission (heavy → light)");
    println!("           both release energy by moving TOWARD iron-56!");

    println!("\n================================================================");
    println!("   NUCLEAR BATTERY PoC COMPLETE");
    println!("================================================================\n");
}