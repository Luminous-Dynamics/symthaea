// Tier 3 Quick Test - Minimal validation of 3 new topologies
// Optimized for fast compilation (<2 min)

use symthaea::hdc::{
    HDC_DIMENSION,
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    tiered_phi::global_phi,
};

fn main() {
    println!("\n🚀 Tier 3 Quick Validation Test");
    println!("{}", "=".repeat(60));
    println!("Testing 3 new topologies with minimal samples");
    println!("Dimension: {}", HDC_DIMENSION);
    println!("Samples: 3 (for speed)\n");

    let n_nodes = 8;
    let n_samples = 3;
    let real_calc = RealPhiCalculator::new();

    // Test 1: Fractal Network
    println!("1️⃣ Fractal Network (Sierpiński-inspired)");
    let mut fractal_real_phis = Vec::new();
    let mut fractal_binary_phis = Vec::new();

    for seed in 0..n_samples {
        let topology = ConsciousnessTopology::fractal(n_nodes, HDC_DIMENSION, seed as u64);

        // RealHV Φ
        let real_phi = real_calc.compute(&topology.node_representations);
        fractal_real_phis.push(real_phi);

        // Binary Φ
        let binary_components: Vec<_> = topology.node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed as u64))
            .collect();
        let binary_phi = global_phi(&binary_components);
        fractal_binary_phis.push(binary_phi);

        println!("   Sample {}: RealΦ = {:.4}, BinaryΦ = {:.4}", seed, real_phi, binary_phi);
    }

    let fractal_real_mean = fractal_real_phis.iter().sum::<f64>() / n_samples as f64;
    let fractal_binary_mean = fractal_binary_phis.iter().sum::<f64>() / n_samples as f64;
    println!("   Mean: RealΦ = {:.4}, BinaryΦ = {:.4}\n", fractal_real_mean, fractal_binary_mean);

    // Test 2: Hypercube 3D
    println!("2️⃣ Hypercube 3D (Cube, 8 vertices)");
    let mut cube_real_phis = Vec::new();
    let mut cube_binary_phis = Vec::new();

    for seed in 0..n_samples {
        let topology = ConsciousnessTopology::hypercube(3, HDC_DIMENSION, seed as u64);

        let real_phi = real_calc.compute(&topology.node_representations);
        cube_real_phis.push(real_phi);

        let binary_components: Vec<_> = topology.node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed as u64))
            .collect();
        let binary_phi = global_phi(&binary_components);
        cube_binary_phis.push(binary_phi);

        println!("   Sample {}: RealΦ = {:.4}, BinaryΦ = {:.4}", seed, real_phi, binary_phi);
    }

    let cube_real_mean = cube_real_phis.iter().sum::<f64>() / n_samples as f64;
    let cube_binary_mean = cube_binary_phis.iter().sum::<f64>() / n_samples as f64;
    println!("   Mean: RealΦ = {:.4}, BinaryΦ = {:.4}\n", cube_real_mean, cube_binary_mean);

    // Test 3: Hypercube 4D
    println!("3️⃣ Hypercube 4D (Tesseract, 16 vertices)");
    let mut tesseract_real_phis = Vec::new();
    let mut tesseract_binary_phis = Vec::new();

    for seed in 0..n_samples {
        let topology = ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed as u64);

        let real_phi = real_calc.compute(&topology.node_representations);
        tesseract_real_phis.push(real_phi);

        let binary_components: Vec<_> = topology.node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed as u64))
            .collect();
        let binary_phi = global_phi(&binary_components);
        tesseract_binary_phis.push(binary_phi);

        println!("   Sample {}: RealΦ = {:.4}, BinaryΦ = {:.4}", seed, real_phi, binary_phi);
    }

    let tesseract_real_mean = tesseract_real_phis.iter().sum::<f64>() / n_samples as f64;
    let tesseract_binary_mean = tesseract_binary_phis.iter().sum::<f64>() / n_samples as f64;
    println!("   Mean: RealΦ = {:.4}, BinaryΦ = {:.4}\n", tesseract_real_mean, tesseract_binary_mean);

    // Test 4: Quantum (1:1:1 equal)
    println!("4️⃣ Quantum Network (1:1:1 equal superposition)");
    let mut quantum_equal_real_phis = Vec::new();
    let mut quantum_equal_binary_phis = Vec::new();

    for seed in 0..n_samples {
        let topology = ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (1.0, 1.0, 1.0), seed as u64);

        let real_phi = real_calc.compute(&topology.node_representations);
        quantum_equal_real_phis.push(real_phi);

        let binary_components: Vec<_> = topology.node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed as u64))
            .collect();
        let binary_phi = global_phi(&binary_components);
        quantum_equal_binary_phis.push(binary_phi);

        println!("   Sample {}: RealΦ = {:.4}, BinaryΦ = {:.4}", seed, real_phi, binary_phi);
    }

    let quantum_equal_real_mean = quantum_equal_real_phis.iter().sum::<f64>() / n_samples as f64;
    let quantum_equal_binary_mean = quantum_equal_binary_phis.iter().sum::<f64>() / n_samples as f64;
    println!("   Mean: RealΦ = {:.4}, BinaryΦ = {:.4}\n", quantum_equal_real_mean, quantum_equal_binary_mean);

    // Test 5: Quantum (3:1:1 Ring-biased)
    println!("5️⃣ Quantum Network (3:1:1 Ring-biased)");
    let mut quantum_ring_real_phis = Vec::new();
    let mut quantum_ring_binary_phis = Vec::new();

    for seed in 0..n_samples {
        let topology = ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (3.0, 1.0, 1.0), seed as u64);

        let real_phi = real_calc.compute(&topology.node_representations);
        quantum_ring_real_phis.push(real_phi);

        let binary_components: Vec<_> = topology.node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed as u64))
            .collect();
        let binary_phi = global_phi(&binary_components);
        quantum_ring_binary_phis.push(binary_phi);

        println!("   Sample {}: RealΦ = {:.4}, BinaryΦ = {:.4}", seed, real_phi, binary_phi);
    }

    let quantum_ring_real_mean = quantum_ring_real_phis.iter().sum::<f64>() / n_samples as f64;
    let quantum_ring_binary_mean = quantum_ring_binary_phis.iter().sum::<f64>() / n_samples as f64;
    println!("   Mean: RealΦ = {:.4}, BinaryΦ = {:.4}\n", quantum_ring_real_mean, quantum_ring_binary_mean);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("📊 TIER 3 QUICK TEST SUMMARY");
    println!("{}", "=".repeat(60));
    println!("\nRealHV Φ Results:");
    println!("  Fractal:           {:.4}", fractal_real_mean);
    println!("  Hypercube 3D:      {:.4}", cube_real_mean);
    println!("  Hypercube 4D:      {:.4}", tesseract_real_mean);
    println!("  Quantum (1:1:1):   {:.4}", quantum_equal_real_mean);
    println!("  Quantum (3:1:1):   {:.4}", quantum_ring_real_mean);

    println!("\nBinary Φ Results:");
    println!("  Fractal:           {:.4}", fractal_binary_mean);
    println!("  Hypercube 3D:      {:.4}", cube_binary_mean);
    println!("  Hypercube 4D:      {:.4}", tesseract_binary_mean);
    println!("  Quantum (1:1:1):   {:.4}", quantum_equal_binary_mean);
    println!("  Quantum (3:1:1):   {:.4}", quantum_ring_binary_mean);

    // Analysis
    println!("\n{}", "=".repeat(60));
    println!("🔬 HYPOTHESIS VALIDATION");
    println!("{}", "=".repeat(60));

    // Reference values from Tier 1 & 2
    let ring_phi = 0.4954;
    let torus_phi = 0.4954;

    println!("\n1. Dimensional Invariance (Ring={:.4}, Torus={:.4}):", ring_phi, torus_phi);
    println!("   Hypercube 3D: {:.4} (Δ = {:.2}%)",
        cube_real_mean,
        ((cube_real_mean - ring_phi) / ring_phi * 100.0));
    println!("   Hypercube 4D: {:.4} (Δ = {:.2}%)",
        tesseract_real_mean,
        ((tesseract_real_mean - ring_phi) / ring_phi * 100.0));

    if (cube_real_mean - ring_phi).abs() < 0.001 && (tesseract_real_mean - ring_phi).abs() < 0.001 {
        println!("   ✅ CONFIRMED: Dimensional invariance extends to 3D and 4D!");
    } else {
        println!("   ❌ REJECTED: Dimensional invariance breaks beyond 2D");
    }

    println!("\n2. Self-Similarity (Binary Tree=0.4712):");
    println!("   Fractal: {:.4} (Δ = {:.2}%)",
        fractal_real_mean,
        ((fractal_real_mean - 0.4712) / 0.4712 * 100.0));

    if fractal_real_mean > 0.4712 {
        println!("   ✅ Fractal outperforms Binary Tree (cross-scale links help)");
    } else {
        println!("   ❌ Fractal doesn't beat Binary Tree");
    }

    println!("\n3. Quantum Superposition:");
    let expected_equal = (0.4954 + 0.4553 + 0.4358) / 3.0; // Ring + Star + Random
    println!("   Quantum (1:1:1): {:.4} vs expected {:.4} (Δ = {:.2}%)",
        quantum_equal_real_mean,
        expected_equal,
        ((quantum_equal_real_mean - expected_equal) / expected_equal * 100.0));

    if (quantum_equal_real_mean - expected_equal).abs() < 0.005 {
        println!("   ✅ Linear combination confirmed (no emergent benefit)");
    } else if quantum_equal_real_mean > expected_equal + 0.005 {
        println!("   🌟 EMERGENT BENEFIT detected! Φ exceeds linear combination!");
    } else {
        println!("   ⚠️  Below expected - possible destructive interference");
    }

    println!("\n{}", "=".repeat(60));
    println!("✅ Tier 3 Quick Test Complete!");
    println!("{}", "=".repeat(60));
}

// Helper function: Probabilistic binarization
fn real_hv_to_hv16_probabilistic(real_hv: &symthaea::hdc::real_hv::RealHV, seed: u64) -> symthaea::hdc::binary_hv::HV16 {
    use symthaea::hdc::binary_hv::HV16;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    // Start with zero HV
    let mut result = HV16::zero();

    // Set bits probabilistically based on sigmoid
    for i in 0..real_hv.values.len() {
        let value = real_hv.values[i];
        // Sigmoid: p = 1 / (1 + exp(-value))
        let prob = 1.0 / (1.0 + (-value).exp());
        let random: f32 = rng.gen();
        if random < prob {
            // Set bit i using basis vector (which has only bit i set)
            let basis = HV16::basis(i);
            result = result.bind(&basis); // XOR to set the bit
        }
    }

    result
}
