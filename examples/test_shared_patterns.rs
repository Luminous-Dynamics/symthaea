/// Test Shared Pattern Ratio Approach
///
/// This test verifies that our shared pattern ratio approach actually creates
/// the expected similarity relationships.

use symthaea::hdc::HV16;

fn main() {
    println!("🧪 Testing Shared Pattern Ratio Hypothesis");
    println!("═════════════════════════════════════════\n");

    // Test 1: 80% shared (4 shared + 1 unique)
    println!("Test 1: 80% Shared Patterns (AlertFocused)");
    let global1 = HV16::random(42);
    let global2 = HV16::random(43);
    let global3 = HV16::random(44);
    let global4 = HV16::random(45);

    let comp_a = HV16::bundle(&[
        global1.clone(),
        global2.clone(),
        global3.clone(),
        global4.clone(),
        HV16::random(100),  // unique_a
    ]);

    let comp_b = HV16::bundle(&[
        global1.clone(),
        global2.clone(),
        global3.clone(),
        global4.clone(),
        HV16::random(200),  // unique_b
    ]);

    let similarity_80 = comp_a.similarity(&comp_b);
    println!("  Similarity between components: {:.4}", similarity_80);
    println!("  Expected: ~0.80");
    println!();

    // Test 2: 0% shared (pure random)
    println!("Test 2: 0% Shared Patterns (DeepAnesthesia)");
    let random_a = HV16::random(300);
    let random_b = HV16::random(400);

    let similarity_0 = random_a.similarity(&random_b);
    println!("  Similarity between components: {:.4}", similarity_0);
    println!("  Expected: ~0.00");
    println!();

    // Test 3: 25% shared (1 shared + 3 unique)
    println!("Test 3: 25% Shared Patterns (LightAnesthesia)");
    let shared_one = HV16::random(500);

    let comp_c = HV16::bundle(&[
        shared_one.clone(),
        HV16::random(600),
        HV16::random(601),
        HV16::random(602),
    ]);

    let comp_d = HV16::bundle(&[
        shared_one.clone(),
        HV16::random(700),
        HV16::random(701),
        HV16::random(702),
    ]);

    let similarity_25 = comp_c.similarity(&comp_d);
    println!("  Similarity between components: {:.4}", similarity_25);
    println!("  Expected: ~0.25");
    println!();

    // Verification
    println!("═════════════════════════════════════════");
    println!("HYPOTHESIS VERIFICATION:");
    println!("  More shared patterns → Higher similarity?");

    let hypothesis_holds = similarity_80 > similarity_25 && similarity_25 > similarity_0;

    if hypothesis_holds {
        println!("  ✅ CONFIRMED: {} > {} > {}", similarity_80, similarity_25, similarity_0);
    } else {
        println!("  ❌ REJECTED: {} > {} > {} is FALSE", similarity_80, similarity_25, similarity_0);
        println!("  Actual order: 80%: {}, 25%: {}, 0%: {}",
                 similarity_80, similarity_25, similarity_0);
    }
}
