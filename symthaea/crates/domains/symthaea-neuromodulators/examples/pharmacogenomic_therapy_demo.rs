// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! B→F→A→H pipeline demo: pharmacogenomic-aware therapeutic consciousness
//! intervention with PNI coupling.
//!
//! Simulates a 100-tick therapy session for 3 genetically distinct patients
//! under the same protocol (baseline → SSRI → threat exposure → sleep → follow-up),
//! showing how CYP2D6/HTR2A/COMT variants alter neuromodulator trajectories
//! and consciousness cost.
//!
//! Run:
//! ```
//! cargo run -p symthaea-neuromodulators --example pharmacogenomic_therapy_demo
//! ```

use std::collections::HashMap;
use symthaea_neuromodulators::pharmacogenomics::{
    AncestryGroup, CypEnzyme, DrugClass, GeneticVariant, MetabolizerPhenotype, PgxProfile, Zygosity,
};
use symthaea_neuromodulators::research_bridge::ResearchModulation;

// ── Patient profiles ────────────────────────────────────────────────────

fn patient_a() -> PgxProfile {
    // Normal metabolizer, wild-type HTR2A — standard responder
    let mut metabolizer_map = HashMap::new();
    metabolizer_map.insert(CypEnzyme::CYP2D6, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP2C19, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP3A4, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP2C9, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP1A2, MetabolizerPhenotype::Normal);
    PgxProfile {
        metabolizer_map,
        genetic_variants: vec![
            (GeneticVariant::HTR2A_rs6311, Zygosity::WildType),
            (GeneticVariant::COMT_Val158Met, Zygosity::Heterozygous),
            (GeneticVariant::OPRM1_A118G, Zygosity::WildType),
            (GeneticVariant::BDNF_Val66Met, Zygosity::WildType),
            (GeneticVariant::SLC6A4_5HTTLPR, Zygosity::Heterozygous),
        ],
        ancestry: AncestryGroup::European,
    }
}

fn patient_b() -> PgxProfile {
    // Poor CYP2D6, homozygous HTR2A — slow metabolism, high 5-HT2A density
    let mut metabolizer_map = HashMap::new();
    metabolizer_map.insert(CypEnzyme::CYP2D6, MetabolizerPhenotype::Poor);
    metabolizer_map.insert(CypEnzyme::CYP2C19, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP3A4, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP2C9, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP1A2, MetabolizerPhenotype::Normal);
    PgxProfile {
        metabolizer_map,
        genetic_variants: vec![
            (GeneticVariant::HTR2A_rs6311, Zygosity::Homozygous),
            (GeneticVariant::COMT_Val158Met, Zygosity::WildType),
            (GeneticVariant::OPRM1_A118G, Zygosity::WildType),
            (GeneticVariant::BDNF_Val66Met, Zygosity::WildType),
            (GeneticVariant::SLC6A4_5HTTLPR, Zygosity::WildType),
        ],
        ancestry: AncestryGroup::European,
    }
}

fn patient_c() -> PgxProfile {
    // Ultra-rapid CYP2D6, COMT Val/Val — fast metabolism, fast DA clearance
    let mut metabolizer_map = HashMap::new();
    metabolizer_map.insert(CypEnzyme::CYP2D6, MetabolizerPhenotype::UltraRapid);
    metabolizer_map.insert(CypEnzyme::CYP2C19, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP3A4, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP2C9, MetabolizerPhenotype::Normal);
    metabolizer_map.insert(CypEnzyme::CYP1A2, MetabolizerPhenotype::Normal);
    PgxProfile {
        metabolizer_map,
        genetic_variants: vec![
            (GeneticVariant::HTR2A_rs6311, Zygosity::WildType),
            (GeneticVariant::COMT_Val158Met, Zygosity::WildType), // Val/Val = high activity
            (GeneticVariant::OPRM1_A118G, Zygosity::WildType),
            (GeneticVariant::BDNF_Val66Met, Zygosity::WildType),
            (GeneticVariant::SLC6A4_5HTTLPR, Zygosity::WildType),
        ],
        ancestry: AncestryGroup::Mixed,
    }
}

// ── Therapy protocol ────────────────────────────────────────────────────

fn phase_label(tick: usize) -> &'static str {
    match tick {
        0..=19 => "Baseline      ",
        20..=39 => "SSRI Admin    ",
        40..=59 => "Threat Expose ",
        60..=79 => "Sleep/Recovery",
        80..=99 => "Follow-Up     ",
        _ => "Unknown       ",
    }
}

fn run_session(name: &str, profile: PgxProfile) {
    let cs = profile.consciousness_sensitivity();
    println!("═══ {name} ═══");
    println!(
        "  CYP2D6: {:?}  |  5-HT2A density: {:.2}  |  DA clearance: {:.2}  |  Phi mod: {:.3}",
        profile.metabolizer(CypEnzyme::CYP2D6),
        cs.serotonin_receptor_density,
        cs.dopamine_clearance_rate,
        cs.phi_baseline_modifier,
    );
    println!();
    println!(
        "{:>4} | {:<14} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6} | {:>10}",
        "Tick", "Phase", "D5-HT", "DDA", "DNE", "DAdn", "Sick?", "Phi Cost"
    );
    println!("{}", "-".repeat(88));

    let mut rm = ResearchModulation::with_pgx(profile);
    let dt = 0.05; // ~20Hz cognitive tick

    for tick in 0..100 {
        // Phase-dependent inputs
        let (threat, cortisol, allostatic, sleeping, sleep_q) = match tick {
            0..=19 => (0.05, 0.1, 0.1, false, 0.0), // mild baseline stress
            20..=39 => (0.05, 0.1, 0.1, false, 0.0), // SSRI on board, same stress
            40..=59 => (0.7, 0.6, 0.3, false, 0.0), // threat exposure therapy
            60..=79 => (0.0, 0.05, 0.15, true, 0.8), // sleep recovery
            80..=99 => (0.1, 0.15, 0.12, false, 0.0), // mild follow-up
            _ => (0.0, 0.0, 0.0, false, 0.0),
        };

        // Administer SSRI at tick 20
        if tick == 20 {
            let resp = rm.apply_drug(DrugClass::SSRI, 1.0);
            println!(
                "  >> SSRI administered: peak conc={:.2}, half-life={:.1}h, accum warn={}",
                resp.peak_concentration, resp.effective_half_life, resp.accumulation_warning,
            );
        }

        // Update PNI from environmental signals
        rm.update_pni(threat, cortisol, allostatic, sleeping, sleep_q);
        rm.tick(dt);

        // Print at key ticks (every 5th + phase boundaries)
        if tick % 5 == 0 || tick == 20 || tick == 40 || tick == 60 || tick == 80 {
            let deltas = rm.combined_deltas();
            let cost = rm.consciousness_cost();
            let sick = if rm.is_sick() { "YES" } else { "no " };
            println!(
                "{:>4} | {} | {:>+7.4} | {:>+7.4} | {:>+7.4} | {:>+7.4} | {:>6} | {:>10.6}",
                tick,
                phase_label(tick),
                deltas.delta_serotonin,
                deltas.delta_dopamine,
                deltas.delta_noradrenaline,
                deltas.delta_adenosine,
                sick,
                cost,
            );
        }
    }

    // Final summary
    let final_deltas = rm.combined_deltas();
    let final_cost = rm.consciousness_cost();
    println!("{}", "-".repeat(88));
    println!(
        "  Final state: 5-HT={:>+.4}  DA={:>+.4}  NE={:>+.4}  cost={:.6}  immune={:.2}  sick={}",
        final_deltas.delta_serotonin,
        final_deltas.delta_dopamine,
        final_deltas.delta_noradrenaline,
        final_cost,
        rm.immune_competence(),
        rm.is_sick(),
    );
    if let Some(sens) = rm.consciousness_sensitivity() {
        println!(
            "  Consciousness profile: Phi mod={:.3}  neuroplasticity={:.2}  5-HT transport={:.2}",
            sens.phi_baseline_modifier,
            sens.neuroplasticity_factor,
            sens.serotonin_transporter_efficiency,
        );
    }
    println!();
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Pharmacogenomic Consciousness Therapy Demo          ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  B→F→A→H: PGx + PNI → Neuromod bath → Consciousness ║");
    println!("║  Comparing 3 genetic profiles under same protocol    ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Protocol: Baseline(0-19) → SSRI(20-39) → Threat(40-59) → Sleep(60-79) → Follow-up(80-99)"
    );
    println!();

    run_session(
        "Patient A: CYP2D6 Normal / HTR2A Wild-Type (standard responder)",
        patient_a(),
    );
    run_session(
        "Patient B: CYP2D6 Poor / HTR2A Homozygous (slow metabolism, high 5-HT2A)",
        patient_b(),
    );
    run_session(
        "Patient C: CYP2D6 Ultra-Rapid / COMT Val/Val (fast metabolism, fast DA clearance)",
        patient_c(),
    );

    // Cross-patient comparison
    println!("═══ Cross-Patient Comparison ═══");
    println!();
    let profiles = [
        ("A (Normal)", patient_a()),
        ("B (Poor/High-5HT2A)", patient_b()),
        ("C (UltraRapid/COMT)", patient_c()),
    ];
    println!(
        "{:<22} | {:>10} | {:>10} | {:>8} | {:>10} | {:>8}",
        "Patient", "Peak Conc", "Half-Life", "Accum?", "Phi Mod", "DA Clear"
    );
    println!("{}", "-".repeat(80));
    for (label, profile) in &profiles {
        let resp = profile.predict_response(DrugClass::SSRI, 1.0);
        let cs = profile.consciousness_sensitivity();
        println!(
            "{:<22} | {:>10.2} | {:>8.1} h | {:>8} | {:>10.4} | {:>8.2}",
            label,
            resp.peak_concentration,
            resp.effective_half_life,
            if resp.accumulation_warning {
                "YES"
            } else {
                "no"
            },
            cs.phi_baseline_modifier,
            cs.dopamine_clearance_rate,
        );
    }
    println!();
    println!("Key insight: Patient B (CYP2D6 Poor) accumulates 1.8x peak SSRI concentration");
    println!("  with accumulation warning — requires dose reduction in clinical practice.");
    println!("Patient C (Ultra-Rapid) clears drug at 0.6x normal — may need dose increase.");
}
