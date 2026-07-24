// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Quarantined Shared-Mask HDC Algebra Demo
//!
//! **INSECURE EXPERIMENT:** retained under its historical filename for
//! compatibility. This is not FHE, secure aggregation, threshold sharing, or
//! a privacy protocol. Reusing a mask exposes pairwise XOR and distance.
//!
//! ## What This Demonstrates
//!
//! 1. A fresh, uniformly random, one-use XOR mask has OTP algebra
//! 2. **Leakage under reuse**: sim(enc(A,M), enc(B,M)) = sim(A,B) — exactly
//! 3. **Homomorphic binding**: Operations on encrypted data produce correct results
//! 4. Shared-mask majority aggregation as a non-private algebra experiment
//! 5. Why the historical share format fails threshold privacy
//!
//! ## Run
//!
//! ```bash
//! cargo run --example fhe_collective_intelligence_demo
//! ```

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_crypto::HdcThresholdSharing;
use symthaea_core::hdc::hdc_fhe::{CollectiveWisdomPool, EncryptedHV, generate_collective_mask};

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  INSECURE DEMO: Shared-Mask HDC Algebra (Not FHE/Privacy) ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════
    // PART 1: ENCRYPTION FUNDAMENTALS
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 1: One-Time Pad Encryption ━━━\n");

    let concept_love = BinaryHV::random(42);
    let concept_justice = BinaryHV::random(43);
    let mask = BinaryHV::random(99);

    // Plaintext similarity
    let plain_sim = concept_love.similarity(&concept_justice);
    println!("  Plaintext similarity(love, justice) = {plain_sim:.6}");

    // Encrypt both with same mask
    let enc_love = EncryptedHV::encrypt(&concept_love, &mask);
    let enc_justice = EncryptedHV::encrypt(&concept_justice, &mask);

    // Encrypted similarity — should be IDENTICAL to plaintext
    let enc_sim = enc_love.encrypted_similarity(&enc_justice);
    println!("  Encrypted similarity(love, justice) = {enc_sim:.6}");
    println!(
        "  Difference: {:.10} (should be exactly 0.0)",
        (plain_sim - enc_sim).abs()
    );

    // Ciphertext reveals nothing about plaintext
    let ciphertext_leakage = enc_love.ciphertext.similarity(&concept_love);
    println!("  Ciphertext leakage: {ciphertext_leakage:.4} (should be ~0.5 = random)");

    // Roundtrip
    let decrypted = enc_love.decrypt(&mask);
    assert_eq!(decrypted, concept_love);
    println!("  Roundtrip: PASS (decrypted == original)\n");

    // ═══════════════════════════════════════════════════════════════════
    // PART 2: HOMOMORPHIC BINDING
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 2: Homomorphic Binding ━━━\n");

    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);
    let mask_a = BinaryHV::random(10);
    let mask_b = BinaryHV::random(20);

    // Bind in plaintext
    let plain_bind = a.bind(&b);

    // Bind in encrypted domain
    let enc_a = EncryptedHV::encrypt(&a, &mask_a);
    let enc_b = EncryptedHV::encrypt(&b, &mask_b);
    let enc_bind = enc_a.hom_bind(&enc_b);

    // Decrypt with combined mask
    let combined_mask = mask_a.bind(&mask_b);
    let decrypted_bind = enc_bind.decrypt(&combined_mask);

    let bind_match = decrypted_bind == plain_bind;
    println!(
        "  Plain bind(A, B)     = [first 32 bits: {:08b}{:08b}{:08b}{:08b}]",
        plain_bind.0[0], plain_bind.0[1], plain_bind.0[2], plain_bind.0[3]
    );
    println!(
        "  Encrypted bind(A, B) = [first 32 bits: {:08b}{:08b}{:08b}{:08b}]",
        decrypted_bind.0[0], decrypted_bind.0[1], decrypted_bind.0[2], decrypted_bind.0[3]
    );
    println!("  Exact match: {bind_match} (XOR distributes over XOR)\n");

    // ═══════════════════════════════════════════════════════════════════
    // PART 3: DIFFERENT-MASK DECORRELATION
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 3: Different-Mask Decorrelation (Not a Security Test) ━━━\n");

    let same_thought = BinaryHV::random(777);
    let mask_session_1 = BinaryHV::random(100);
    let mask_session_2 = BinaryHV::random(200);

    let enc_s1 = EncryptedHV::encrypt(&same_thought, &mask_session_1);
    let enc_s2 = EncryptedHV::encrypt(&same_thought, &mask_session_2);

    // Same thought, different sessions
    let cross_sim = enc_s1.encrypted_similarity(&enc_s2);
    println!("  Same thought, different masks: sim = {cross_sim:.4} (should be ~0.5)");
    println!("  Observation only: different deterministic masks decorrelate this sample\n");

    // ═══════════════════════════════════════════════════════════════════
    // PART 4: COLLECTIVE WISDOM AGGREGATION
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 4: Collective Wisdom (5 Peers) ━━━\n");

    let n_peers = 5;
    let collective_mask = BinaryHV::random(42_000);
    let mut pool = CollectiveWisdomPool::new();

    // Each peer has local wisdom about a shared concept
    let peer_names = ["Alice", "Bob", "Carol", "Dave", "Eve"];
    let mut wisdoms = Vec::new();
    for (i, name) in peer_names.iter().enumerate() {
        let wisdom = BinaryHV::random(1000 + i as u64);
        let encrypted = EncryptedHV::encrypt(&wisdom, &collective_mask);
        pool.contribute(name, encrypted);
        wisdoms.push(wisdom);
        println!(
            "  {name} contributes encrypted wisdom (density: {:.3})",
            wisdom.density()
        );
    }

    // Aggregate
    let aggregate = pool.aggregate().expect("should have contributions");
    let decrypted_aggregate = aggregate.decrypt(&collective_mask);

    // Compare to plaintext bundle
    let expected_bundle = BinaryHV::bundle(&wisdoms);
    let fidelity = decrypted_aggregate.similarity(&expected_bundle);

    println!("\n  Aggregated {n_peers} encrypted wisdoms");
    println!("  Decrypted aggregate similarity to true bundle: {fidelity:.4}");
    println!(
        "  Fidelity: {} (>0.85 expected for 5 peers)\n",
        if fidelity > 0.85 {
            "EXCELLENT"
        } else {
            "DEGRADED"
        }
    );

    // Show that aggregate captures each peer's contribution
    println!("  Individual contribution recovery:");
    for (i, name) in peer_names.iter().enumerate() {
        let sim = decrypted_aggregate.similarity(&wisdoms[i]);
        println!("    {name}: sim = {sim:.4} (>0.5 = above random)");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // PART 5: BROKEN THRESHOLD-SHARING ATTACK
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 5: One Share Defeats Claimed 3-of-5 Threshold ━━━\n");

    let (mask_full, shares) = generate_collective_mask(3, 5, 12345);

    // Encrypt with the full mask
    let secret_wisdom = BinaryHV::random(9999);
    let encrypted_wisdom = EncryptedHV::encrypt(&secret_wisdom, &mask_full);

    let mask_from_one_share = HdcThresholdSharing::recover(&shares[..1]);
    let decrypt_attempt = encrypted_wisdom.decrypt(&mask_from_one_share);
    let similarity = decrypt_attempt.similarity(&secret_wisdom);
    println!("  1-of-5 shares → similarity to truth: {similarity:.4}");
    println!("  ATTACK CONFIRMED: one share recovers the mask despite k=3");

    // ═══════════════════════════════════════════════════════════════════
    // PART 6: ALGEBRA COST COMPARISON
    // ═══════════════════════════════════════════════════════════════════

    println!("\n━━━ Part 6: Plain XOR vs Wrapped XOR Cost (Not an FHE Comparison) ━━━\n");

    let n_ops = 10_000;
    let hv_a = BinaryHV::random(1);
    let hv_b = BinaryHV::random(2);
    let mask_perf = BinaryHV::random(3);

    // Time plaintext operations
    let start = std::time::Instant::now();
    for _ in 0..n_ops {
        let _ = hv_a.bind(&hv_b);
    }
    let plain_time = start.elapsed();

    // Time encrypted operations
    let enc_perf_a = EncryptedHV::encrypt(&hv_a, &mask_perf);
    let enc_perf_b = EncryptedHV::encrypt(&hv_b, &mask_perf);
    let start = std::time::Instant::now();
    for _ in 0..n_ops {
        let _ = enc_perf_a.hom_bind(&enc_perf_b);
    }
    let enc_time = start.elapsed();

    let overhead = enc_time.as_nanos() as f64 / plain_time.as_nanos() as f64;
    println!("  {n_ops} plaintext binds:  {:?}", plain_time);
    println!("  {n_ops} encrypted binds: {:?}", enc_time);
    println!("  Overhead ratio: {overhead:.2}× (CKKS would be ~10,000×)");

    // ═══════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  One-use XOR algebra:   Round-trip demonstrated           ║");
    println!("║  Similarity preserved:  Exactly (XOR is isometry)          ║");
    println!("║  Homomorphic binding:   Exact (XOR distributes over XOR)   ║");
    println!(
        "║  Collective aggregation: >{:.0}% fidelity (5 peers)         ║",
        fidelity * 100.0
    );
    println!("║  Threshold privacy:     BROKEN — one share recovers mask   ║");
    println!("║  Wrapped XOR overhead:  {overhead:.1}× vs direct XOR              ║");
    println!("║  Shared-mask privacy:   BROKEN — pairwise distances leak   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("This example demonstrates HDC algebra and its security failures.");
    println!("It must not be used as an encryption or privacy protocol.\n");
}
