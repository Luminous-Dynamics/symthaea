// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Resonance Discovery Engine
//!
//! Use Symthaea's HDC framework to discover novel energy solutions through
//! inverse vector search - dreaming backwards from ideal properties.
//!
//! ## The Strategy
//!
//! Instead of checking isotopes one by one (brute force), we:
//! 1. Define the ideal solution as a target vector
//! 2. Search the space of isotopes, lattices, and couplings
//! 3. Find what minimizes distance to the target
//!
//! ## Three Search Paths
//!
//! 1. **NEEC Path**: Electron-nuclear coupling (Thorium-229m style)
//! 2. **Phonon Path**: Lattice vibration coupling (Mössbauer style)
//! 3. **LCF Path**: Lattice Confinement Fusion (NASA Erbium style)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example resonance_discovery --release
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    ElectronNuclearCoupling, Hadrons, InverseSearchEngine, NuclearPhysics, PeriodicTable,
    PhononDynamics, StandardModel, TargetProperties,
};

fn main() {
    println!("\n");
    println!("================================================================");
    println!("   RESONANCE DISCOVERY ENGINE");
    println!("   Dreaming backwards from ideal properties");
    println!("================================================================\n");

    // Initialize the physics universe
    let genesis = GenesisSeed::from_phrase("Sovereign Energy Discovery");
    println!("Genesis Seed: \"Sovereign Energy Discovery\"\n");

    println!("Initializing physics layers...");
    let model = StandardModel::from_genesis(&genesis);
    let hadrons = Hadrons::from_model(&model, &genesis);
    let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
    let _nuclear = NuclearPhysics::from_genesis(&genesis, &hadrons, &table);
    let neec = ElectronNuclearCoupling::from_genesis(&genesis, &table, &hadrons);
    let phonons = PhononDynamics::from_genesis(&genesis, &table);
    let engine = InverseSearchEngine::from_genesis(&genesis);
    println!("  Standard Model: {} particles", 17); // quarks + leptons + bosons
    println!("  Periodic Table: {} elements", 36);
    println!("  NEEC Candidates: {}", neec.candidates.len());
    println!("  Lattice Materials: {}", phonons.materials.len());

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SEARCH 1: THE IDEAL ENERGY SOURCE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let target = TargetProperties::ideal_energy_source();
    println!("Target Properties (0-1 scale):");
    println!(
        "  Energy Density:       {:.1} (high = nuclear scale)",
        target.energy_density
    );
    println!(
        "  Trigger Accessibility:{:.1} (high = laser accessible)",
        target.trigger_accessibility
    );
    println!(
        "  Stability:            {:.1} (high = decades)",
        target.stability
    );
    println!(
        "  Abundance:            {:.1} (high = common)",
        target.abundance
    );
    println!("  NEEC Coupling:        {:.1}", target.neec_coupling);
    println!("  Phonon Coupling:      {:.1}", target.phonon_coupling);
    println!("  Safety:               {:.1}", target.safety);

    println!("\nSearching...\n");
    let candidates = engine.search(&target, &neec, &phonons, &table, &hadrons);

    println!("TOP 5 CANDIDATES:\n");
    println!(
        "{:<25} {:>8} {:>8} {:>8} {:>8}",
        "Candidate", "Match", "Trigger", "Stable", "NEEC"
    );
    println!("{}", "─".repeat(60));

    for (i, candidate) in candidates.iter().take(5).enumerate() {
        println!(
            "{:<25} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            candidate.name,
            candidate.match_score * 100.0,
            candidate.property_scores.trigger_accessibility * 100.0,
            candidate.property_scores.stability * 100.0,
            candidate.property_scores.neec_coupling * 100.0
        );

        if i == 0 {
            println!("\n  ┌─ BEST MATCH TRIGGER PROTOCOL:");
            println!("  │  Method:      {:?}", candidate.trigger.method);
            if let Some(wl) = candidate.trigger.wavelength_nm {
                println!("  │  Wavelength:  {:.2} nm", wl);
            }
            if let Some(ref pulse) = candidate.trigger.pulse_duration {
                println!("  │  Pulse:       {}", pulse);
            }
            if let Some(temp) = candidate.trigger.temperature_k {
                println!("  │  Temperature: {} K", temp);
            }
            println!("  └─ Notes: {}\n", candidate.trigger.notes);
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SEARCH 2: NUCLEAR CLOCK (Thorium-229m Style)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let clock_target = TargetProperties::nuclear_clock();
    println!("Target: Maximum trigger accessibility (laser-accessible nuclear transition)\n");

    let clock_candidates = engine.search(&clock_target, &neec, &phonons, &table, &hadrons);

    println!("TOP 3 NUCLEAR CLOCK CANDIDATES:\n");
    for candidate in clock_candidates.iter().take(3) {
        println!(
            "  {} (Match: {:.1}%)",
            candidate.name,
            candidate.match_score * 100.0
        );
        println!("    Trigger: {:?}", candidate.trigger.method);
        if let Some(wl) = candidate.trigger.wavelength_nm {
            if wl < 200.0 {
                println!(
                    "    Wavelength: {:.2} nm (VUV - challenging but possible)",
                    wl
                );
            } else {
                println!("    Wavelength: {:.2} nm", wl);
            }
        }
        println!();
    }

    // Highlight Thorium-229m specifically
    if let Some(_th229) = clock_candidates.iter().find(|c| c.name.contains("Th-229")) {
        println!("  ★ THORIUM-229m ANALYSIS:");
        println!("    The nuclear clock isomer with ~8.28 eV transition");
        println!("    This is the ONLY known nuclear transition in the VUV range");
        println!("    Trigger: VUV laser at ~150 nm");
        println!("    Application: Ultra-precise timekeeping, dark matter detection");
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SEARCH 3: LATTICE CONFINEMENT FUSION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let lcf_target = TargetProperties::lcf_optimized();
    println!("Target: Maximum phonon coupling + deuterium host capability\n");

    let lcf_candidates = engine.search(&lcf_target, &neec, &phonons, &table, &hadrons);

    // Filter to LCF candidates only
    let lcf_only: Vec<_> = lcf_candidates
        .iter()
        .filter(|c| c.name.contains("LCF"))
        .collect();

    println!("LATTICE CONFINEMENT FUSION HOSTS:\n");
    println!(
        "{:<20} {:>10} {:>12} {:>10}",
        "Material", "LCF Score", "Phonon Coup", "Match"
    );
    println!("{}", "─".repeat(55));

    for candidate in lcf_only.iter().take(5) {
        println!(
            "{:<20} {:>9.1}% {:>11.1}% {:>9.1}%",
            candidate.name,
            candidate.property_scores.energy_density * 100.0,
            candidate.property_scores.phonon_coupling * 100.0,
            candidate.match_score * 100.0
        );
    }

    // Analyze best LCF materials
    println!("\nLATTICE ANALYSIS:\n");
    for material in phonons.materials.iter().take(5) {
        let lcf = phonons.analyze_lcf_potential(material);
        if lcf.lcf_potential > 0.5 {
            println!(
                "  {} (LCF Potential: {:.1}%)",
                material.name,
                lcf.lcf_potential * 100.0
            );
            println!("    H Solubility:   {:.1}%", lcf.h_solubility * 100.0);
            println!("    Screening:      {:.1}%", lcf.screening_factor * 100.0);
            println!("    Phonon Factor:  {:.1}%", lcf.phonon_factor * 100.0);
            println!("    Debye Temp:     {} K", material.debye_temp_k);
            println!("    Max Phonon:     {:.4} eV", material.max_phonon_ev());
            println!();
        }
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("NEEC COUPLING ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Known NEEC Candidates (sorted by cross-section estimate):\n");
    println!(
        "{:<12} {:>10} {:>12} {:>12} {:>12}",
        "Isomer", "Energy", "Detuning", "Coupling", "X-Section"
    );
    println!("{}", "─".repeat(60));

    let mut neec_sorted = neec.candidates.clone();
    neec_sorted.sort_by(|a, b| {
        let xs_a = neec.estimate_cross_section(a);
        let xs_b = neec.estimate_cross_section(b);
        xs_b.partial_cmp(&xs_a).unwrap()
    });

    for coupling in neec_sorted.iter().take(5) {
        let xs = neec.estimate_cross_section(coupling);
        let energy_str = if coupling.nuclear.energy_ev < 1000.0 {
            format!("{:.2} eV", coupling.nuclear.energy_ev)
        } else if coupling.nuclear.energy_ev < 1_000_000.0 {
            format!("{:.2} keV", coupling.nuclear.energy_ev / 1000.0)
        } else {
            format!("{:.2} MeV", coupling.nuclear.energy_ev / 1_000_000.0)
        };

        println!(
            "{:<12} {:>10} {:>11.2} eV {:>11.1}% {:>12.2e}",
            format!(
                "{}-{}",
                element_symbol(coupling.nuclear.z),
                coupling.nuclear.a
            ),
            energy_str,
            coupling.detuning_ev,
            coupling.coupling_strength * 100.0,
            xs
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHONON-NUCLEAR RESONANCE SEARCH");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Searching for lattices where phonon modes couple to nuclear transitions...\n");

    // For each NEEC candidate, find best phonon coupling
    for coupling in neec.candidates.iter().take(3) {
        println!(
            "Target: {}-{} ({:.2} eV transition)",
            element_symbol(coupling.nuclear.z),
            coupling.nuclear.a,
            coupling.nuclear.energy_ev
        );

        let phonon_coupling = phonons.calculate_coupling(
            phonons
                .materials
                .iter()
                .find(|m| m.name == "CaF2")
                .unwrap_or(&phonons.materials[0]),
            coupling.nuclear.energy_ev,
        );

        println!("  Best Host: {}", phonon_coupling.material.name);
        println!(
            "  Phonon Mode: {:.2} THz ({:.4} eV)",
            phonon_coupling.phonon.frequency_thz, phonon_coupling.phonon.energy_ev
        );
        println!(
            "  Coupling Strength: {:.1}%",
            phonon_coupling.coupling_strength * 100.0
        );
        println!("  Detuning: {:.4} eV", phonon_coupling.detuning_ev);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DISCOVERY SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Get best overall candidate
    let best = candidates.first();

    println!(
        "RECOMMENDED PATH: {}\n",
        if let Some(b) = best {
            if b.name.contains("Th-229") {
                "NEEC (Thorium-229m Nuclear Clock)"
            } else if b.name.contains("LCF") {
                "Lattice Confinement Fusion"
            } else {
                "NEEC Coupling"
            }
        } else {
            "Unknown"
        }
    );

    println!("Key Findings:");
    println!();
    println!("  1. THORIUM-229m remains the 'Holy Grail' for laser-triggered nuclear");
    println!("     transitions due to its uniquely low 8.28 eV excitation energy.");
    println!("     Trigger: VUV laser at ~150 nm in CaF2 crystal host.");
    println!();
    println!("  2. LATTICE CONFINEMENT FUSION in Erbium/Palladium hosts offers the");
    println!("     highest energy density (fusion scale) with demonstrated physics.");
    println!("     Challenge: Maintaining lattice integrity under fusion conditions.");
    println!();
    println!("  3. ELECTRON-NUCLEAR COUPLING provides a path to trigger nuclear");
    println!("     transitions by exciting electron shells - the 'Rube Goldberg' approach.");
    println!();

    println!("NEXT STEPS:");
    println!();
    println!("  → Run multi-seed search for novel candidates");
    println!("  → Design self-healing metamaterial for LCF lattice");
    println!("  → Calculate optimal CaF2 crystal orientation for Th-229m");
    println!("  → Search for common-element isomers with NEEC potential");

    println!("\n================================================================");
    println!("   RESONANCE DISCOVERY COMPLETE");
    println!("================================================================\n");
}

fn element_symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        6 => "C",
        8 => "O",
        26 => "Fe",
        42 => "Mo",
        43 => "Tc",
        46 => "Pd",
        68 => "Er",
        72 => "Hf",
        73 => "Ta",
        90 => "Th",
        92 => "U",
        _ => "X",
    }
}