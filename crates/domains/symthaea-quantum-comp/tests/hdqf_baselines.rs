use symthaea_quantum_comp::{
    HdqfBaselineDisposition, HdqfInstanceFamily, HdqfProblemConfig, HdqfSearchBudget,
    exhaustive_factorization, generate_hdqf_instance, meet_in_the_middle_exact,
};

fn config(family: HdqfInstanceFamily, seed: u64) -> HdqfProblemConfig {
    HdqfProblemConfig {
        dimension: 32,
        factor_count: 3,
        codebook_size: 4,
        epsilon: 0.0,
        family,
        correlation_rate: 0.9,
        adversarial_margin_bits: 3,
        seed,
    }
}

#[test]
fn all_preregistered_families_are_deterministic_and_replayable() {
    for family in [
        HdqfInstanceFamily::PlantedUnique,
        HdqfInstanceFamily::Random,
        HdqfInstanceFamily::CollisionRich,
        HdqfInstanceFamily::Correlated,
        HdqfInstanceFamily::Adversarial,
    ] {
        let a = generate_hdqf_instance(&config(family, 41)).unwrap();
        let b = generate_hdqf_instance(&config(family, 41)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.canonical_text(), b.canonical_text());
        assert_eq!(
            a.reproducibility_fingerprint(),
            b.reproducibility_fingerprint()
        );
        assert_eq!(
            a.product_for_indices(&a.planted_indices).unwrap(),
            a.clean_target
        );
        assert!(a.canonical_text().starts_with("HDQF_INSTANCE_V1\n"));
    }
}

#[test]
fn planted_unique_and_collision_rich_have_expected_multiplicity() {
    let unique = generate_hdqf_instance(&config(HdqfInstanceFamily::PlantedUnique, 7)).unwrap();
    let unique_report = exhaustive_factorization(&unique, HdqfSearchBudget::unlimited()).unwrap();
    assert_eq!(
        unique_report.disposition,
        HdqfBaselineDisposition::Completed
    );
    assert_eq!(unique_report.best_hamming_distance, Some(0));
    assert_eq!(unique_report.optimal_count, Some(1));
    assert_eq!(unique_report.returned_planted, Some(true));

    let collision = generate_hdqf_instance(&config(HdqfInstanceFamily::CollisionRich, 7)).unwrap();
    let collision_report =
        exhaustive_factorization(&collision, HdqfSearchBudget::unlimited()).unwrap();
    assert_eq!(collision_report.best_hamming_distance, Some(0));
    assert!(collision_report.optimal_count.unwrap() >= 2);
}

#[test]
fn exact_mitm_matches_exhaustive_for_multiple_factor_counts() {
    for factor_count in [2, 3, 4] {
        let mut cfg = config(HdqfInstanceFamily::Random, 100 + factor_count as u64);
        cfg.factor_count = factor_count;
        if factor_count == 4 {
            cfg.dimension = 130;
        }
        let instance = generate_hdqf_instance(&cfg).unwrap();
        let exhaustive =
            exhaustive_factorization(&instance, HdqfSearchBudget::unlimited()).unwrap();
        let mitm = meet_in_the_middle_exact(&instance, HdqfSearchBudget::unlimited()).unwrap();

        assert_eq!(mitm.disposition, HdqfBaselineDisposition::Completed);
        assert_eq!(mitm.best_hamming_distance, exhaustive.best_hamming_distance);
        assert_eq!(mitm.optimal_count, exhaustive.optimal_count);
        assert_eq!(mitm.returned_indices, exhaustive.returned_indices);
        assert_eq!(mitm.returned_planted, exhaustive.returned_planted);
        assert_eq!(mitm.metrics.solution_candidates_verified, 1);
        assert!(
            mitm.metrics.partial_products_generated
                < exhaustive.metrics.candidate_products_evaluated
                || factor_count == 2
        );
    }
}

#[test]
fn noisy_objective_is_exhaustive_only_until_noise_aware_mitm_exists() {
    let mut cfg = config(HdqfInstanceFamily::Random, 88);
    cfg.dimension = 64;
    cfg.epsilon = 0.10;
    let instance = generate_hdqf_instance(&cfg).unwrap();
    assert!(instance.is_noisy());
    assert_eq!(instance.epsilon(), 0.10);

    let exhaustive = exhaustive_factorization(&instance, HdqfSearchBudget::unlimited()).unwrap();
    assert_eq!(exhaustive.disposition, HdqfBaselineDisposition::Completed);
    assert!(exhaustive.best_hamming_distance.is_some());

    let mitm = meet_in_the_middle_exact(&instance, HdqfSearchBudget::unlimited()).unwrap();
    assert_eq!(mitm.disposition, HdqfBaselineDisposition::NotApplicable);
    assert!(mitm.reason.unwrap().contains("minimum-distance"));
}

#[test]
fn adversarial_family_injects_the_declared_near_collision() {
    let cfg = config(HdqfInstanceFamily::Adversarial, 19);
    let instance = generate_hdqf_instance(&cfg).unwrap();
    let mut alternative = instance.planted_indices.clone();
    alternative[0] = 1;
    let alternative_product = instance.product_for_indices(&alternative).unwrap();
    assert_eq!(
        alternative_product
            .hamming_distance(&instance.clean_target)
            .unwrap(),
        cfg.adversarial_margin_bits
    );
}

#[test]
fn preregistered_resource_ceilings_censor_before_work() {
    let instance = generate_hdqf_instance(&config(HdqfInstanceFamily::Random, 5)).unwrap();
    let tiny_budget = HdqfSearchBudget {
        max_candidate_products: 1,
        max_stored_partial_tuples: 1,
    };

    let exhaustive = exhaustive_factorization(&instance, tiny_budget).unwrap();
    assert_eq!(
        exhaustive.disposition,
        HdqfBaselineDisposition::ResourceCensored
    );
    assert_eq!(exhaustive.metrics.candidate_products_evaluated, 0);

    let mitm = meet_in_the_middle_exact(&instance, tiny_budget).unwrap();
    assert_eq!(mitm.disposition, HdqfBaselineDisposition::ResourceCensored);
    assert_eq!(mitm.metrics.partial_products_generated, 0);
}
