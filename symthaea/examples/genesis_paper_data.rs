// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Paper Data Generator
//!
//! Runs real simulations and outputs CSV data for the stewardship paper figures.
//! Three outputs: heterozygosity decay, attachment trajectories, CfC vs simulation.
//!
//! Run with: `cargo run --example genesis_paper_data --features population,nurture`

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::io::Write;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_nurture::{AttachmentSystem, CaregiverAction};
use symthaea_population::{
    Allele, BiologicalSex, BreedingStrategy, Genotype, Individual, Locus, Pedigree, PedigreeEntry,
    Population, PopulationSimulator, PopulationTrajectoryPredictor, encode_individual,
    encode_locus, heterozygosity_after_generations, observed_heterozygosity,
};

/// Create a founder population with balanced allelic diversity at each locus.
fn make_founder_population(
    n_females: usize,
    n_males: usize,
    loci: &[Locus],
    rng: &mut impl Rng,
) -> (Population, Pedigree) {
    let mut individuals = Vec::new();
    let mut pedigree = Pedigree::new();
    let n = n_females + n_males;

    for i in 0..n {
        let sex = if i < n_females {
            BiologicalSex::Female
        } else {
            BiologicalSex::Male
        };

        let genotypes: Vec<Genotype> = loci
            .iter()
            .map(|l| {
                let a_id = rng.gen_range(0..10);
                let b_id = rng.gen_range(0..10);
                Genotype {
                    locus_id: l.id,
                    allele_a: Allele::neutral(a_id),
                    allele_b: Allele::neutral(b_id),
                }
            })
            .collect();

        let genome_hv = if !loci.is_empty() {
            encode_individual(loci, &genotypes)
        } else {
            ContinuousHV::zero(HDC_DIMENSION)
        };

        let id = i as u64;
        individuals.push(Individual {
            id,
            sex,
            genotypes,
            genome_hv,
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        });
        pedigree.add_entry(PedigreeEntry {
            individual_id: id,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
    }

    let div_hv = ContinuousHV::zero(HDC_DIMENSION);
    (
        Population {
            individuals,
            generation: 0,
            founding_size: n,
            diversity_hv: div_hv,
        },
        pedigree,
    )
}

/// Create loci with deterministic HDC encodings.
fn make_loci(n: usize) -> Vec<Locus> {
    (0..n)
        .map(|i| Locus {
            id: i as u64,
            name: format!("locus_{i}"),
            chromosome: 1,
            position: i as u64 * 1000,
            hv: encode_locus(i as u64),
        })
        .collect()
}

/// Compute mean observed heterozygosity across all loci in a population.
fn mean_heterozygosity(pop: &Population) -> f64 {
    if pop.individuals.is_empty() {
        return 0.0;
    }
    let n_loci = pop.individuals[0].genotypes.len();
    if n_loci == 0 {
        return 0.0;
    }
    let total: f64 = (0..n_loci)
        .map(|li| {
            let genos: Vec<Genotype> = pop
                .individuals
                .iter()
                .filter_map(|ind| ind.genotypes.get(li).cloned())
                .collect();
            observed_heterozygosity(&genos)
        })
        .sum();
    total / n_loci as f64
}

// ============================================================================
// Data Generator 1: Heterozygosity Decay at Three Population Sizes
// ============================================================================

fn generate_het_decay_data() {
    println!("--- Generating heterozygosity decay data ---");

    let n_loci = 20;
    let generations = 50;
    let loci = make_loci(n_loci);

    // Three effective population sizes: Ne ~ 50, 200, 500
    // Using balanced sex ratios so Ne ~ N
    let configs: [(usize, usize, &str); 3] = [
        (25, 25, "ne50"),    // Ne ~ 50
        (100, 100, "ne200"), // Ne ~ 200
        (250, 250, "ne500"), // Ne ~ 500
    ];

    let mut results: Vec<Vec<f64>> = Vec::new();
    let mut h0_values: Vec<f64> = Vec::new();

    for (idx, &(n_f, n_m, label)) in configs.iter().enumerate() {
        let target_size = n_f + n_m;
        let ne = target_size as f64;
        println!(
            "  Running {label} (N={target_size}, Ne~{ne:.0}) for {generations} generations..."
        );

        let mut rng = StdRng::seed_from_u64(42 + idx as u64);
        let (pop, mut ped) = make_founder_population(n_f, n_m, &loci, &mut rng);

        let h0 = mean_heterozygosity(&pop);
        h0_values.push(h0);
        println!("    Initial H(0) = {h0:.4}");

        let sim = PopulationSimulator {
            strategy: BreedingStrategy::Random,
            target_population_size: target_size,
            mutation_rate: 0.0,
            selection_enabled: false,
        };

        let result = sim.run(&pop, &mut ped, &loci, generations, &mut rng);

        let het_series: Vec<f64> = result
            .generations
            .iter()
            .map(|snap| snap.heterozygosity)
            .collect();

        let h_final = het_series.last().copied().unwrap_or(0.0);
        println!("    Final H({generations}) = {h_final:.4}");

        results.push(het_series);
    }

    // Write CSV
    let path = "papers/data/het_decay.csv";
    let mut file = fs::File::create(path).expect("create het_decay.csv");
    writeln!(
        file,
        "generation,ne50_sim,ne200_sim,ne500_sim,ne50_analytical,ne200_analytical,ne500_analytical"
    )
    .unwrap();

    let ne_values = [50.0, 200.0, 500.0];

    for r#gen in 0..=generations {
        let ne50_sim = results[0][r#gen as usize];
        let ne200_sim = results[1][r#gen as usize];
        let ne500_sim = results[2][r#gen as usize];

        let ne50_ana = heterozygosity_after_generations(h0_values[0], ne_values[0], r#gen);
        let ne200_ana = heterozygosity_after_generations(h0_values[1], ne_values[1], r#gen);
        let ne500_ana = heterozygosity_after_generations(h0_values[2], ne_values[2], r#gen);

        writeln!(
            file,
            "{r#gen},{ne50_sim:.6},{ne200_sim:.6},{ne500_sim:.6},{ne50_ana:.6},{ne200_ana:.6},{ne500_ana:.6}"
        )
        .unwrap();
    }

    println!("  Written: {path}");
    println!();
}

// ============================================================================
// Data Generator 2: Attachment Style Formation Trajectories
// ============================================================================

fn generate_attachment_trajectories() {
    println!("--- Generating attachment style trajectories ---");

    let n_interactions = 1000;
    let sample_interval = 10;

    let profiles: [(&str, CaregiverAction); 4] = [
        ("reliable", CaregiverAction::reliable()),
        ("inconsistent", CaregiverAction::inconsistent()),
        ("unresponsive", CaregiverAction::unresponsive()),
        ("frightening", CaregiverAction::frightening()),
    ];

    let mut trajectories: Vec<Vec<f64>> = Vec::new();

    for (label, action) in &profiles {
        println!("  Running {label} caregiver for {n_interactions} interactions...");
        let mut sys = AttachmentSystem::new();

        let mut security_series = Vec::new();

        for i in 0..n_interactions {
            sys.process_interaction(action);
            sys.advance_age(0.05); // ~4 years total over 1000 interactions

            if i % sample_interval == 0 {
                security_series.push(sys.security_score);
            }
        }

        println!(
            "    Final style: {:?}, security: {:.4}",
            sys.style, sys.security_score
        );
        trajectories.push(security_series);
    }

    // Write CSV
    let path = "papers/data/attachment_trajectories.csv";
    let mut file = fs::File::create(path).expect("create attachment_trajectories.csv");
    writeln!(
        file,
        "interaction,reliable_security,inconsistent_security,unresponsive_security,frightening_security"
    )
    .unwrap();

    let n_samples = trajectories[0].len();
    for i in 0..n_samples {
        let interaction = i * sample_interval;
        writeln!(
            file,
            "{interaction},{:.6},{:.6},{:.6},{:.6}",
            trajectories[0][i], trajectories[1][i], trajectories[2][i], trajectories[3][i]
        )
        .unwrap();
    }

    println!("  Written: {path}");
    println!();
}

// ============================================================================
// Data Generator 3: CfC Prediction vs Forward Simulation
// ============================================================================

fn generate_cfc_vs_sim() {
    println!("--- Generating CfC prediction vs simulation data ---");

    let n_loci = 20;
    let generations = 30;
    let target_size = 100; // Ne ~ 100
    let ne = target_size as f64;
    let loci = make_loci(n_loci);

    let mut rng = StdRng::seed_from_u64(777);
    let (pop, mut ped) = make_founder_population(target_size / 2, target_size / 2, &loci, &mut rng);

    let h0 = mean_heterozygosity(&pop);
    println!("  Initial H(0) = {h0:.4}");

    // Run forward simulation
    println!("  Running forward simulation (Ne~{ne:.0}, {generations} generations)...");
    let sim = PopulationSimulator {
        strategy: BreedingStrategy::Random,
        target_population_size: target_size,
        mutation_rate: 0.0,
        selection_enabled: false,
    };
    let sim_result = sim.run(&pop, &mut ped, &loci, generations, &mut rng);

    // Encode initial state for CfC predictor
    let initial_state = PopulationTrajectoryPredictor::encode_population_state(&pop);
    let mut predictor = PopulationTrajectoryPredictor::new();

    // Write CSV
    let path = "papers/data/cfc_vs_sim.csv";
    let mut file = fs::File::create(path).expect("create cfc_vs_sim.csv");
    writeln!(
        file,
        "generation,simulation_het,cfc_predicted_het,analytical_het"
    )
    .unwrap();

    for r#gen in 0..=generations {
        let sim_het = sim_result.generations[r#gen as usize].heterozygosity;

        // CfC prediction: predict from initial state to this generation
        predictor.reset();
        let predicted_state = predictor.predict_at_generation(&initial_state, r#gen);
        let cfc_het = PopulationTrajectoryPredictor::decode_heterozygosity(&predicted_state);

        // Analytical prediction
        let ana_het = heterozygosity_after_generations(h0, ne, r#gen);

        writeln!(file, "{r#gen},{sim_het:.6},{cfc_het:.6},{ana_het:.6}").unwrap();

        if r#gen % 10 == 0 {
            println!(
                "    Gen {r#gen:3}: sim={sim_het:.4} cfc={cfc_het:.4} analytical={ana_het:.4}"
            );
        }
    }

    println!("  Written: {path}");
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    fs::create_dir_all("papers/data").expect("create data dir");

    println!("=== Genesis Paper Data Generator ===");
    println!();

    generate_het_decay_data();
    generate_attachment_trajectories();
    generate_cfc_vs_sim();

    println!("Done! CSV files in papers/data/");
}
