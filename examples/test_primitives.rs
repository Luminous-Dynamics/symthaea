// Quick test of principled causal primitives
use symthaea::benchmarks::{
    TuebingenAdapter,
    discover_by_principled_hdc,
    discover_by_phi,
    discover_by_unified_primitives,
    discover_information_theoretic,
    discover_majority_voting,
};

fn main() {
    println!("Testing Principled Symthaea Causal Primitives\n");

    // Load dataset
    let tuebingen_path = "benchmarks/external/tuebingen";
    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };
    println!("Loaded {} pairs\n", adapter.len());

    // Test Principled HDC
    println!("Testing Principled HDC (Functional Complexity + Independence)...");
    let results_phdc = adapter.run(discover_by_principled_hdc);
    println!("  Principled HDC: {:.1}% ({}/{})",
             results_phdc.accuracy() * 100.0,
             results_phdc.correct,
             results_phdc.total);

    // Test Phi-based
    println!("Testing Phi-based (Effective Information)...");
    let results_phi = adapter.run(discover_by_phi);
    println!("  Phi-based: {:.1}% ({}/{})",
             results_phi.accuracy() * 100.0,
             results_phi.correct,
             results_phi.total);

    // Test Unified Primitives
    println!("Testing Unified Symthaea Primitives...");
    let results_unified = adapter.run(discover_by_unified_primitives);
    println!("  Unified Primitives: {:.1}% ({}/{})",
             results_unified.accuracy() * 100.0,
             results_unified.correct,
             results_unified.total);

    // Compare with previous best
    println!("\nComparing with previous methods...");
    let results_info = adapter.run(discover_information_theoretic);
    println!("  Info-Theoretic: {:.1}% ({}/{})",
             results_info.accuracy() * 100.0,
             results_info.correct,
             results_info.total);

    let results_majority = adapter.run(discover_majority_voting);
    println!("  Majority Voting: {:.1}% ({}/{})",
             results_majority.accuracy() * 100.0,
             results_majority.correct,
             results_majority.total);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       SUMMARY                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let methods = vec![
        ("Principled HDC", results_phdc.accuracy()),
        ("Phi-based (EI)", results_phi.accuracy()),
        ("Unified Primitives", results_unified.accuracy()),
        ("Info-Theoretic", results_info.accuracy()),
        ("Majority Voting", results_majority.accuracy()),
    ];

    println!("  Method                   Accuracy    vs Random");
    println!("  ─────────────────────────────────────────────────");
    for (name, acc) in &methods {
        let delta = (acc - 0.5) * 100.0;
        let marker = if *acc > 0.6 { "✓" } else if *acc > 0.5 { "~" } else { "✗" };
        println!("  {:22} {:5.1}%      {:+5.1}%  {}", name, acc * 100.0, delta, marker);
    }

    let best = methods.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("\n  Best: {} ({:.1}%)", best.0, best.1 * 100.0);
    println!("\n  Done!");
}
