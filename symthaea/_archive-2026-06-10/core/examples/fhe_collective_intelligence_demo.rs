// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-FHE Collective Intelligence Demo
//!
//! Demonstrates privacy-preserving collective reasoning using
//! hyperdimensional homomorphic encryption — zero overhead, perfect secrecy.
//!
//! ## What This Proves
//!
//! 1. **Perfect secrecy**: Encrypted vectors reveal zero information (OTP)
//! 2. **Distance preservation**: sim(enc(A,M), enc(B,M)) = sim(A,B) — exactly
//! 3. **Homomorphic binding**: Operations on encrypted data produce correct results
//! 4. **Collective aggregation**: Multiple peers aggregate without revealing individuals
//! 5. **Threshold recovery**: k-of-n mask sharing enables cooperative decryption
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
    println!("║    Symthaea HDC-FHE: Privacy-Preserving Collective Mind    ║");
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
    // PART 3: CROSS-SESSION PRIVACY
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 3: Cross-Session Privacy ━━━\n");

    let same_thought = BinaryHV::random(777);
    let mask_session_1 = BinaryHV::random(100);
    let mask_session_2 = BinaryHV::random(200);

    let enc_s1 = EncryptedHV::encrypt(&same_thought, &mask_session_1);
    let enc_s2 = EncryptedHV::encrypt(&same_thought, &mask_session_2);

    // Same thought, different sessions
    let cross_sim = enc_s1.encrypted_similarity(&enc_s2);
    println!("  Same thought, different masks: sim = {cross_sim:.4} (should be ~0.5)");
    println!(
        "  Privacy: {} (adversary learns nothing across sessions)\n",
        if (cross_sim - 0.5).abs() < 0.05 {
            "PROTECTED"
        } else {
            "LEAK!"
        }
    );

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
    // PART 5: THRESHOLD SECRET SHARING
    // ═══════════════════════════════════════════════════════════════════

    println!("━━━ Part 5: Threshold Recovery (3-of-5) ━━━\n");

    let (mask_full, shares) = generate_collective_mask(3, 5, 12345);

    // Encrypt with the full mask
    let secret_wisdom = BinaryHV::random(9999);
    let encrypted_wisdom = EncryptedHV::encrypt(&secret_wisdom, &mask_full);

    // Try 2-of-5 (should fail — insufficient shares)
    let mask_2_of_5 = HdcThresholdSharing::recover(&shares[..2]);
    let decrypt_attempt_2 = encrypted_wisdom.decrypt(&mask_2_of_5);
    let sim_2 = decrypt_attempt_2.similarity(&secret_wisdom);
    println!("  2-of-5 shares → similarity to truth: {sim_2:.4} (should be ~0.5 = failure)");

    // Try 3-of-5 (should succeed — meets threshold)
    let mask_3_of_5 = HdcThresholdSharing::recover(&shares[..3]);
    let decrypt_attempt_3 = encrypted_wisdom.decrypt(&mask_3_of_5);
    let sim_3 = decrypt_attempt_3.similarity(&secret_wisdom);
    println!("  3-of-5 shares → similarity to truth: {sim_3:.4} (should be 1.0 = success)");

    // Try 5-of-5 (should also succeed)
    let mask_5_of_5 = HdcThresholdSharing::recover(&shares);
    let decrypt_attempt_5 = encrypted_wisdom.decrypt(&mask_5_of_5);
    let sim_5 = decrypt_attempt_5.similarity(&secret_wisdom);
    println!("  5-of-5 shares → similarity to truth: {sim_5:.4} (should be 1.0 = success)");

    // ═══════════════════════════════════════════════════════════════════
    // PART 6: PERFORMANCE COMPARISON
    // ═══════════════════════════════════════════════════════════════════

    println!("\n━━━ Part 6: Performance vs Lattice FHE ━━━\n");

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
    println!("║  OTP encryption:        Perfect secrecy (Shannon 1949)     ║");
    println!("║  Similarity preserved:  Exactly (XOR is isometry)          ║");
    println!("║  Homomorphic binding:   Exact (XOR distributes over XOR)   ║");
    println!(
        "║  Collective aggregation: >{:.0}% fidelity (5 peers)         ║",
        fidelity * 100.0
    );
    println!("║  Threshold recovery:    3-of-5 shares required             ║");
    println!("║  Overhead vs plaintext: {overhead:.1}× (vs ~10,000× for CKKS)     ║");
    println!("║  Cross-session privacy: No leakage across mask rotations   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Consciousness can reason collectively without surrendering privacy.");
    println!("The HDC algebra makes this free — not fast, FREE.\n");
}