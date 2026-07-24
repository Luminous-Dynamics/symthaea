use symthaea_quantum_comp::{
    HdqfBaselineDisposition, HdqfInstanceFamily, HdqfProblemConfig, HdqfSearchBudget,
    ResonatorConfig, ResonatorInitialization, ResonatorTermination, Wagner4Config,
    exhaustive_factorization, generate_hdqf_instance, resonator_factorization,
    wagner4_factorization,
};

fn config(dimension: usize, factors: usize, size: usize, seed: u64) -> HdqfProblemConfig {
    HdqfProblemConfig {
        dimension,
        factor_count: factors,
        codebook_size: size,
        epsilon: 0.0,
        family: HdqfInstanceFamily::Random,
        correlation_rate: 0.9,
        adversarial_margin_bits: 2,
        seed,
    }
}

#[test]
fn complete_wagner4_scan_matches_exhaustive_multiplicity() {
    let instance = generate_hdqf_instance(&config(16, 4, 8, 71)).unwrap();
    let exhaustive = exhaustive_factorization(&instance, HdqfSearchBudget::unlimited()).unwrap();
    let wagner = wagner4_factorization(
        &instance,
        Wagner4Config {
            low_bits: 2,
            alpha_trials: 4,
            alpha_seed: 99,
            max_intermediate_entries: usize::MAX,
        },
    )
    .unwrap();

    assert_eq!(wagner.disposition, HdqfBaselineDisposition::Completed);
    assert!(wagner.complete_alpha_scan);
    assert_eq!(wagner.exact_solution_count, exhaustive.optimal_count);
    assert_eq!(wagner.returned_indices, exhaustive.returned_indices);
    assert_eq!(wagner.returned_planted, exhaustive.returned_planted);
    assert_eq!(wagner.metrics.solution_candidates_verified, 1);
}

#[test]
fn partial_wagner4_scan_never_claims_exact_multiplicity() {
    let instance = generate_hdqf_instance(&config(16, 4, 8, 72)).unwrap();
    let wagner = wagner4_factorization(
        &instance,
        Wagner4Config {
            low_bits: 3,
            alpha_trials: 1,
            alpha_seed: 5,
            max_intermediate_entries: usize::MAX,
        },
    )
    .unwrap();
    assert!(!wagner.complete_alpha_scan);
    assert_eq!(wagner.exact_solution_count, None);
    assert_eq!(wagner.metrics.alpha_buckets_examined, 1);
}

#[test]
fn wagner4_reports_applicability_and_resource_boundaries() {
    let dense = generate_hdqf_instance(&config(6, 4, 4, 73)).unwrap();
    let dense_report = wagner4_factorization(
        &dense,
        Wagner4Config {
            low_bits: 2,
            alpha_trials: 4,
            alpha_seed: 0,
            max_intermediate_entries: usize::MAX,
        },
    )
    .unwrap();
    assert!(dense_report.asymptotic_assumptions_satisfied);

    let censored = wagner4_factorization(
        &dense,
        Wagner4Config {
            low_bits: 2,
            alpha_trials: 4,
            alpha_seed: 0,
            max_intermediate_entries: 0,
        },
    )
    .unwrap();
    assert_eq!(
        censored.disposition,
        HdqfBaselineDisposition::ResourceCensored
    );
    assert_eq!(censored.metrics.partial_pairs_generated, 0);

    let three_factor = generate_hdqf_instance(&config(16, 3, 4, 74)).unwrap();
    let not_applicable =
        wagner4_factorization(&three_factor, Wagner4Config::for_instance(&three_factor, 0))
            .unwrap();
    assert_eq!(
        not_applicable.disposition,
        HdqfBaselineDisposition::NotApplicable
    );
}

#[test]
fn resonator_is_deterministic_and_never_beats_exhaustive_ground_truth() {
    let instance = generate_hdqf_instance(&config(256, 3, 4, 80)).unwrap();
    let settings = ResonatorConfig {
        max_iterations: 64,
        restarts: 4,
        initialization: ResonatorInitialization::CodebookSuperposition,
        seed: 123,
        stop_on_exact: true,
    };
    let a = resonator_factorization(&instance, settings).unwrap();
    let b = resonator_factorization(&instance, settings).unwrap();
    let exhaustive = exhaustive_factorization(&instance, HdqfSearchBudget::unlimited()).unwrap();

    assert_eq!(a, b);
    assert!(a.achieved_hamming_distance >= exhaustive.best_hamming_distance.unwrap());
    assert!(a.metrics.cleanup_calls > 0);
    assert!(a.metrics.codebook_dot_products > 0);
    assert!(a.metrics.readout_comparisons > 0);
}

#[test]
fn resonator_recovers_an_easy_high_dimensional_factorization() {
    let mut cfg = config(1024, 3, 3, 81);
    cfg.family = HdqfInstanceFamily::PlantedUnique;
    let instance = generate_hdqf_instance(&cfg).unwrap();
    let report = resonator_factorization(
        &instance,
        ResonatorConfig {
            max_iterations: 64,
            restarts: 8,
            initialization: ResonatorInitialization::CodebookSuperposition,
            seed: 456,
            stop_on_exact: true,
        },
    )
    .unwrap();

    assert_eq!(report.achieved_hamming_distance, 0);
    assert_eq!(report.termination, ResonatorTermination::ExactSolution);
    assert!(report.returned_planted);
}

#[test]
fn resonator_handles_noisy_targets_without_relabeling_them_exact() {
    let mut cfg = config(128, 3, 4, 82);
    cfg.epsilon = 0.10;
    let instance = generate_hdqf_instance(&cfg).unwrap();
    let settings = ResonatorConfig {
        max_iterations: 8,
        restarts: 2,
        initialization: ResonatorInitialization::Random,
        seed: 789,
        stop_on_exact: false,
    };
    let report = resonator_factorization(&instance, settings).unwrap();
    assert!(report.achieved_hamming_distance <= instance.dimension());
    assert!(report.metrics.iterations <= settings.max_iterations * settings.restarts);
}
