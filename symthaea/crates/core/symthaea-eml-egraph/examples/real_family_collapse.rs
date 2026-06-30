use symthaea_core::hdc::abstract_thought::expr_canonical_string;
use symthaea_core::hdc::conjecture_engine::{
    ConjectureEngine, ObservedSequence, RegressorConfig, observe_balmer_series,
    observe_bell_numbers, observe_blackbody_peak, observe_catalan, observe_central_binomial_limit,
    observe_derangement_ratio, observe_fibonacci_ratios, observe_hydrogen_energy_levels,
    observe_inverse_square_law, observe_kepler_third_law, observe_partitions,
    observe_prime_counting, observe_prime_gaps, observe_quantum_harmonic_oscillator,
    observe_relativistic_kinetic_energy, observe_stefan_boltzmann,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;
use symthaea_eml_egraph::compare_current_vs_egg_collapse;

struct FamilyCase {
    name: &'static str,
    observations: Vec<ObservedSequence>,
}

struct FamilySummary {
    name: &'static str,
    observations: usize,
    conjectures: usize,
    candidates: usize,
    current_buckets: usize,
    egg_classes: usize,
    egg_only_merges: usize,
    unsupported: usize,
}

fn default_engine() -> ConjectureEngine {
    let mut engine = ConjectureEngine::with_config(RegressorConfig {
        population_size: 120,
        generations: 60,
        max_depth: 4,
        max_complexity: 12,
        lambda: 0.001,
        tournament_size: 5,
        mutation_rate: 0.3,
        seed: 42,
        ..RegressorConfig::default()
    });
    engine.enable_abstract_thought();
    engine
}

fn family_cases() -> Vec<FamilyCase> {
    vec![
        FamilyCase {
            name: "combinatorics_growth",
            observations: vec![
                observe_fibonacci_ratios(16),
                observe_partitions(12),
                observe_catalan(12),
                observe_bell_numbers(10),
                observe_central_binomial_limit(18),
            ],
        },
        FamilyCase {
            name: "number_theory",
            observations: vec![
                observe_prime_gaps(120),
                observe_prime_counting(60),
                observe_derangement_ratio(10),
            ],
        },
        FamilyCase {
            name: "physics_closed_form",
            observations: vec![
                observe_hydrogen_energy_levels(8),
                observe_quantum_harmonic_oscillator(8),
                observe_kepler_third_law(8),
                observe_stefan_boltzmann(8),
                observe_inverse_square_law(10),
            ],
        },
        FamilyCase {
            name: "physics_transcendental",
            observations: vec![
                observe_blackbody_peak(12),
                observe_balmer_series(8),
                observe_relativistic_kinetic_energy(12),
            ],
        },
    ]
}

fn summarize_case(case: &FamilyCase) -> FamilySummary {
    let mut engine = default_engine();
    for obs in &case.observations {
        engine.observe(obs.clone());
    }

    engine.generate_conjectures(5);
    engine.verify_numerical();
    engine.reflect(&PrimitiveSystem::new());

    let at = engine
        .abstract_thought
        .as_ref()
        .expect("abstract thought enabled");
    let patterns: Vec<_> = at
        .dynamic_grammar
        .candidates
        .iter()
        .map(|candidate| candidate.pattern.clone())
        .collect();
    let report = compare_current_vs_egg_collapse(&patterns);
    let current_bucket_count = {
        let mut buckets = std::collections::BTreeSet::new();
        for pattern in &patterns {
            buckets.insert(expr_canonical_string(pattern));
        }
        buckets.len()
    };

    println!("family: {}", case.name);
    println!("  observations: {}", engine.observations.len());
    println!("  conjectures: {}", engine.conjectures.len());
    println!("  dynamic grammar candidates: {}", patterns.len());
    println!("  current canonical buckets: {}", current_bucket_count);
    println!("  egg equivalence classes: {}", report.classes.len());
    println!(
        "  egg-only merged classes: {}",
        report
            .classes
            .iter()
            .filter(|class| class.current_canonical_buckets.len() > 1)
            .count()
    );
    println!("  unsupported candidate indices: {:?}", report.unsupported);

    for (class_idx, class) in report.classes.iter().enumerate() {
        if class.current_canonical_buckets.len() <= 1 {
            continue;
        }
        println!("  egg-only merge class {class_idx}");
        println!("    members: {:?}", class.egg_class.members);
        println!("    egg canonical: {}", class.egg_class.egg_canonical);
        for bucket in &class.current_canonical_buckets {
            println!("    current {:?} -> {}", bucket.members, bucket.canonical);
        }
    }

    if !report.unsupported.is_empty() {
        println!("  unsupported patterns");
        for &idx in &report.unsupported {
            let pattern = &patterns[idx];
            println!("    [{idx}] {pattern}");
        }
    }
    println!();

    FamilySummary {
        name: case.name,
        observations: engine.observations.len(),
        conjectures: engine.conjectures.len(),
        candidates: patterns.len(),
        current_buckets: current_bucket_count,
        egg_classes: report.classes.len(),
        egg_only_merges: report
            .classes
            .iter()
            .filter(|class| class.current_canonical_buckets.len() > 1)
            .count(),
        unsupported: report.unsupported.len(),
    }
}

fn main() {
    let summaries: Vec<_> = family_cases().iter().map(summarize_case).collect();

    println!("summary");
    for summary in &summaries {
        println!(
            "  {}: obs={} conjectures={} candidates={} current_buckets={} egg_classes={} egg_only_merges={} unsupported={}",
            summary.name,
            summary.observations,
            summary.conjectures,
            summary.candidates,
            summary.current_buckets,
            summary.egg_classes,
            summary.egg_only_merges,
            summary.unsupported,
        );
    }
}
