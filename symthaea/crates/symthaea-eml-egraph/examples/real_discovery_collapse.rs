use symthaea_core::hdc::abstract_thought::expr_canonical_string;
use symthaea_core::hdc::conjecture_engine::{
    observe_bell_numbers, observe_catalan, observe_fibonacci_ratios, observe_partitions,
    ConjectureEngine, RegressorConfig,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;
use symthaea_eml_egraph::compare_current_vs_egg_collapse;

fn main() {
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

    engine.observe(observe_fibonacci_ratios(16));
    engine.observe(observe_partitions(12));
    engine.observe(observe_catalan(12));
    engine.observe(observe_bell_numbers(10));

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
    let egg_class_count = report.classes.len();
    let egg_only_merges = report
        .classes
        .iter()
        .filter(|class| class.current_canonical_buckets.len() > 1)
        .count();

    println!("real discovery offline egg collapse");
    println!("observations: {}", engine.observations.len());
    println!("conjectures: {}", engine.conjectures.len());
    println!("dynamic grammar candidates: {}", patterns.len());
    println!("current canonical buckets: {}", current_bucket_count);
    println!("egg equivalence classes: {}", egg_class_count);
    println!("egg-only merged classes: {}", egg_only_merges);
    println!("unsupported candidate indices: {:?}", report.unsupported);
    println!();

    if !report.unsupported.is_empty() {
        println!("unsupported real candidates");
        for &idx in &report.unsupported {
            let pattern = &patterns[idx];
            println!("  [{idx}] pattern: {pattern}");
            println!(
                "      current canonical: {}",
                expr_canonical_string(pattern)
            );
        }
        println!();
    }

    for (class_idx, class) in report.classes.iter().enumerate() {
        if class.current_canonical_buckets.len() <= 1 {
            continue;
        }
        println!("egg-only merge class {class_idx}");
        println!("  candidate members: {:?}", class.egg_class.members);
        println!("  egg canonical: {}", class.egg_class.egg_canonical);
        println!("  current canonical buckets:");
        for bucket in &class.current_canonical_buckets {
            println!("    {:?} -> {}", bucket.members, bucket.canonical);
        }
        println!();
    }
}
