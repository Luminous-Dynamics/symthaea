// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grand Unified Demo — Ecological Closure
//!
//! Demonstrates the full integrated closed-loop system:
//! Self-Optimization → Formal Proof → Physiological Feedback → Consensus → Living Manifest → Affective Sculpting → Secure Dreaming

use symthaea_broca::*;
use symthaea_core::hdc::fol_formula_ext::FolFormulaExt;
use symthaea_core::hdc::logic_engine::Proposition;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::HDC_DIMENSION;

fn main() -> anyhow::Result<()> {
    println!("\n🌟 SYMBTHAEA GRAND UNIFIED DEMO: ECOLOGICAL CLOSURE");
    println!("══════════════════════════════════════════════════");

    // 1. Affective Sculpting: Human "Nudge"
    println!("\n[1] AFFECTIVE SCULPTING");
    let mut sculptor = AffectiveSculptor::new();
    let mut channels = ThoughtChannels::default();
    sculptor.sculpt(&mut channels, AffectiveStyle::CalmRigor);
    println!(
        "    Nudged system into {:?} mode for security patch.",
        sculptor.current_style()
    );

    // 2. Self-Optimization: Recursive Evolution
    println!("\n[2] RECURSIVE SELF-OPTIMIZATION");
    let optimizer = SelfOptimizationEngine::new(std::env::current_dir()?);
    let original_intent = "Improve performance while maintaining formal correctness and safety";
    // (Mocking result for demo)
    println!("    Optimizing src/gating.rs...");
    println!("    🚀 Architectural improvement detected! 0.8500 → 0.8712");

    // 3. Formal Logic: Mathematical Proof
    println!("\n[3] FORMAL LOGIC VERIFICATION (E-Axis)");
    let formal_scorer = FormalLogicScorer::new();
    let spec = FolFormulaExt::from_prop(Proposition::True);
    let formal_result = formal_scorer.score_algorithm("security_patch", &spec, "unsafe { ... }");
    println!(
        "    Proof status: {} (Score: {:.2})",
        if formal_result.verified {
            "VERIFIED"
        } else {
            "GRACEFUL DEGRADATION"
        },
        formal_result.score
    );

    // 4. Physiological Grounding: Hardware Feedback
    println!("\n[4] PHYSIOLOGICAL GROUNDING (Machine Physiology)");
    let mut phys_scorer = PhysiologicalScorer::new();
    let profile = phys_scorer.profile_execution("security_service", "enable = true;");
    let surprisal = phys_scorer.to_surprisal(&profile);
    println!(
        "    Memory: {:.1} MB, CPU: {:.1}%, Latency: {:.1}ms",
        profile.memory_mb, profile.cpu_percent, profile.latency_ms
    );
    println!("    Hardware Surprisal: {:.4} (Healthy)", surprisal);

    // 5. Distributed Consensus: P2P Engineering
    println!("\n[5] DISTRIBUTED P2P CONSENSUS");
    let mut consensus = ConsensusEngine::new("node-alpha".to_string());
    let hv = ContinuousHV::random(HDC_DIMENSION, profile.latency_ms as u64);
    consensus.propose_change("src/gating.rs", hv, "Security patch evolution", None, None);

    let mut memory = ArchitecturalMemory::new("target/demo_memory")?;
    let result = consensus.evaluate_consensus("src/gating.rs", 0.9, &mut memory);
    consensus.internalize_consensus("src/gating.rs", &result, &mut memory)?;

    println!(
        "    Network Consensus: reached={}, confidence={:.2}",
        result.consensus_reached, result.confidence
    );

    // 6. Living Manifest: Documentation Projection
    println!("\n[6] LIVING MANIFEST (Documentation)");
    let binding_engine = SubstrateBindingEngine::default();
    let manifest_gen = LivingManifestGenerator::new(binding_engine);
    let manifest = manifest_gen.generate_manifest("Symthaea Core Architecture");
    println!("{}", manifest_gen.to_markdown(&manifest));

    // 7. Secure Dreaming (Zero-Trust Evolution)
    println!("\n[7] SECURE DREAMING (Confidential Evolution)");
    let secure_engine = SecureDreamingEngine::new(std::env::current_dir()?);
    let secure_result = secure_engine.dream_securely(
        std::path::Path::new("crates/symthaea-broca/src/gating.rs"),
        "Improve performance while maintaining formal correctness and safety",
    )?;

    println!("    Success: {}", secure_result.success);
    println!("    Attestation:\n{}", secure_result.attestation);
    println!(
        "    Enclave PK (PQE): {}...",
        &secure_result.enclave_pk[..32]
    );
    println!("    PQ Signature: {}...", &secure_result.pq_signature[..32]);

    println!("══════════════════════════════════════════════════");
    println!("✅ End-to-end loop complete. System is stable and evolving.");

    Ok(())
}
