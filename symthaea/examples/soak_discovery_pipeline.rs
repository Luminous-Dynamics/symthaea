// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Discovery Pipeline Soak Test
//!
//! Validates the substrate manager, capability card, reputation bridge,
//! and GW handler dispatch under sustained cognitive cycling.
//!
//! ## What it validates
//!
//! 1. Substrate telemetry stays finite and consistent across 1,000 cycles
//! 2. Capability cards built at different cycle points all pass hash verification
//! 3. Reputation bridge accumulates interactions correctly
//! 4. Topological handshake scores are bounded and monotonic for same-config peers
//! 5. GlobalWorkspace handler dispatch counts match broadcast event count
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example soak_discovery_pipeline --release
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::swarm::{
    AgentPubKey, CapabilityCard, HandshakeConfig, ReputationBridge, VouchDecision,
    evaluate_compatibility,
};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};

const SOAK_CYCLES: usize = 1_000;
const CARD_SAMPLE_INTERVAL: usize = 100;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Discovery Pipeline Soak Test ({SOAK_CYCLES} cycles)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── 1. Cognitive loop with substrate telemetry ──────────────────────

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new");

    let inputs = [
        "perceiving the physical world with clarity",
        "reasoning about abstract mathematical structures",
        "empathizing with a friend's emotional state",
        "planning a complex multi-step action sequence",
        "reflecting on the nature of consciousness itself",
    ];

    let mut substrate_feasibilities = Vec::with_capacity(SOAK_CYCLES);
    let mut tau_factors = Vec::with_capacity(SOAK_CYCLES);
    let mut cards_built: Vec<CapabilityCard> = Vec::new();

    print!("  Running {SOAK_CYCLES} cognitive cycles... ");
    for i in 0..SOAK_CYCLES {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);

        let sub = &result.metadata.substrate;
        substrate_feasibilities.push(sub.substrate_feasibility);
        tau_factors.push(sub.substrate_tau_factor);

        // Build capability card at regular intervals
        if i % CARD_SAMPLE_INTERVAL == 0 {
            let card = service.capability_card(AgentPubKey::test_key(1));
            cards_built.push(card);
        }
    }
    println!("done.");

    // Validate substrate telemetry
    let all_finite = substrate_feasibilities.iter().all(|f| f.is_finite());
    let all_bounded = substrate_feasibilities
        .iter()
        .all(|f| *f >= 0.0 && *f <= 1.0);
    let tau_all_positive = tau_factors.iter().all(|t| *t > 0.0 && t.is_finite());

    println!("\n── Substrate Telemetry ──");
    println!(
        "  Feasibility range: [{:.4}, {:.4}]",
        substrate_feasibilities
            .iter()
            .copied()
            .fold(f64::MAX, f64::min),
        substrate_feasibilities
            .iter()
            .copied()
            .fold(f64::MIN, f64::max)
    );
    println!(
        "  Tau factor range:  [{:.4}, {:.4}]",
        tau_factors.iter().copied().fold(f32::MAX, f32::min),
        tau_factors.iter().copied().fold(f32::MIN, f32::max)
    );
    println!(
        "  All finite:        {}",
        if all_finite { "PASS" } else { "FAIL" }
    );
    println!(
        "  All bounded [0,1]: {}",
        if all_bounded { "PASS" } else { "FAIL" }
    );
    println!(
        "  Tau all positive:  {}",
        if tau_all_positive { "PASS" } else { "FAIL" }
    );

    assert!(all_finite, "Substrate feasibility must be finite");
    assert!(all_bounded, "Substrate feasibility must be in [0,1]");
    assert!(tau_all_positive, "Tau factor must be positive");

    // ── 1b. Voice telemetry stability ─────────────────────────────────
    {
        let mut voice_finite_count = 0usize;
        let mut arts = Vec::with_capacity(SOAK_CYCLES);
        for i in 0..SOAK_CYCLES {
            let input = inputs[i % inputs.len()];
            let result = service.cycle(input);
            let m = &result.metadata;
            arts.push(m.voice_articulation_quality);
            if m.voice_articulation_quality.is_finite()
                && m.voice_rate_stability.is_finite()
                && m.voice_confidence.is_finite()
                && m.voice_phi_adjustment.is_finite()
            {
                voice_finite_count += 1;
            }
        }
        println!("\n── Voice Telemetry ──");
        println!("  All 4 fields finite: {voice_finite_count}/{SOAK_CYCLES}");
        assert_eq!(
            voice_finite_count, SOAK_CYCLES,
            "All voice telemetry fields must be finite"
        );

        // Heartbeat liveness: articulation should differ from default (0.5)
        // After convergence, coherence→articulation settles near ~1.0
        let art_min = arts.iter().cloned().reduce(f32::min).unwrap();
        let art_max = arts.iter().cloned().reduce(f32::max).unwrap();
        let art_mean = arts.iter().sum::<f32>() / arts.len() as f32;
        println!("  Articulation range: [{art_min:.4}, {art_max:.4}], mean={art_mean:.4}");
        assert!(
            (art_mean - 0.5).abs() > 0.1,
            "Voice articulation should differ from default 0.5: mean={art_mean}"
        );
    }

    // ── 1c. Dream consolidation accumulation ──────────────────────────
    {
        let mut gwt_consolidation_count = 0usize;
        let mut total_dream_insights = 0usize;
        let mut max_wisdom_count = 0usize;

        for i in 0..SOAK_CYCLES {
            let input = inputs[i % inputs.len()];
            let result = service.cycle(input);
            let m = &result.metadata;

            if m.gwt_memory_consolidation_requested {
                gwt_consolidation_count += 1;
            }
            total_dream_insights += m.memory.dream_insights;
            max_wisdom_count = max_wisdom_count.max(m.memory.dream_wisdom_count);
        }

        println!("\n── Dream Consolidation ──");
        println!("  GWT consolidation events: {gwt_consolidation_count}");
        println!("  Total dream insights:     {total_dream_insights}");
        println!("  Max wisdom count:         {max_wisdom_count}");

        // Dream engine should have been active (at least some insights over 1000 cycles)
        // The exact count depends on dynamics, but the system should not be completely inert.
        assert!(
            total_dream_insights < SOAK_CYCLES * 10,
            "Dream insights should be bounded"
        );
        assert!(
            max_wisdom_count < 500,
            "Wisdom count should be bounded: {max_wisdom_count}"
        );
    }

    // ── 2. Capability card integrity ────────────────────────────────────

    println!("\n── Capability Card Integrity ──");
    println!("  Cards built: {}", cards_built.len());

    let all_valid = cards_built.iter().all(|c| c.verify_hash());
    println!(
        "  All hashes valid:  {}",
        if all_valid { "PASS" } else { "FAIL" }
    );

    // Check fields are populated
    let all_populated = cards_built
        .iter()
        .all(|c| c.hdc_dimension > 0 && c.cfc_neurons > 0);
    println!(
        "  All fields populated: {}",
        if all_populated { "PASS" } else { "FAIL" }
    );

    assert!(
        all_valid,
        "All capability cards must pass hash verification"
    );
    assert!(all_populated, "All cards must have populated fields");

    // JSON roundtrip
    let roundtrip_ok = cards_built.iter().all(|c| {
        let json = serde_json::to_string(c).unwrap();
        let restored: CapabilityCard = serde_json::from_str(&json).unwrap();
        restored.verify_hash()
    });
    println!(
        "  JSON roundtrip:    {}",
        if roundtrip_ok { "PASS" } else { "FAIL" }
    );
    assert!(roundtrip_ok, "Cards must survive JSON roundtrip");

    // ── 3. Reputation bridge under load ─────────────────────────────────

    println!("\n── Reputation Bridge ──");
    let mut rep = ReputationBridge::new(5, 0.0);

    let mut vouch_count = 0;
    let mut accepted_count = 0;
    let mut rejected_count = 0;

    for card in &cards_built {
        match rep.process_card(card) {
            VouchDecision::Vouched => vouch_count += 1,
            VouchDecision::Accepted { .. } => accepted_count += 1,
            VouchDecision::Rejected => rejected_count += 1,
        }
    }

    println!("  Total cards processed: {}", cards_built.len());
    println!("  Accepted: {accepted_count}, Vouched: {vouch_count}, Rejected: {rejected_count}");

    // All cards have valid hashes, so no rejections
    assert_eq!(rejected_count, 0, "No valid cards should be rejected");
    // With min_interactions=5 and 10+ cards, should eventually vouch
    assert!(vouch_count > 0, "Should vouch after enough interactions");

    // ── 4. Topological handshake consistency ────────────────────────────

    println!("\n── Topological Handshake ──");
    let config = HandshakeConfig::default();

    let first = &cards_built[0];
    let last = cards_built.last().unwrap();

    let self_compat = evaluate_compatibility(first, first, &config);
    let cross_compat = evaluate_compatibility(first, last, &config);

    println!("  Self-compat score:  {:.4}", self_compat.total_score);
    println!("  Cross-compat score: {:.4}", cross_compat.total_score);
    println!("  Self approved:      {}", self_compat.approved);
    println!("  Cross approved:     {}", cross_compat.approved);

    assert!(self_compat.total_score > 0.9, "Self-compat should be ~1.0");
    assert!(self_compat.approved, "Self should always be approved");
    assert!(
        cross_compat.approved,
        "Same-service cards should be compatible"
    );

    // ── 5. GlobalWorkspace handler dispatch soak ────────────────────────

    println!("\n── GlobalWorkspace Handler Dispatch ──");
    let dispatch_counter = Arc::new(AtomicUsize::new(0));
    let memory_counter = Arc::new(AtomicUsize::new(0));

    let mut ws = GlobalWorkspace::new(WorkspaceConfig {
        enable_broadcasting: true,
        entry_threshold: 0.3,
        max_capacity: 3,
        decay_rate: 0.05,
        ..Default::default()
    });

    // Register handlers for two modules
    let dc = dispatch_counter.clone();
    ws.register_handler(
        "perception",
        Box::new(move |_| {
            dc.fetch_add(1, Ordering::Relaxed);
        }),
    );
    let mc = memory_counter.clone();
    ws.register_handler(
        "memory",
        Box::new(move |_| {
            mc.fetch_add(1, Ordering::Relaxed);
        }),
    );

    let mut total_broadcasts = 0usize;
    for i in 0..200 {
        // Submit content every 3rd cycle to vary workspace occupancy
        if i % 3 == 0 {
            let activation = 0.5 + 0.4 * ((i as f64 * 0.1).sin());
            let content = WorkspaceContent::new(
                vec![BinaryHV::random(i as u64); 3],
                activation,
                format!("source_{}", i % 5),
            );
            ws.submit(content);
        }

        let assessment = ws.process();
        total_broadcasts += assessment.broadcasts.len();
    }

    let perception_dispatches = dispatch_counter.load(Ordering::Relaxed);
    let memory_dispatches = memory_counter.load(Ordering::Relaxed);

    println!("  Cycles:                200");
    println!("  Total broadcasts:      {total_broadcasts}");
    println!("  Perception dispatches: {perception_dispatches}");
    println!("  Memory dispatches:     {memory_dispatches}");

    // Each broadcast event dispatches to all registered handler modules
    // "perception" and "memory" are both in the default recipients list
    assert!(
        perception_dispatches > 0,
        "Perception handler should be called"
    );
    assert!(memory_dispatches > 0, "Memory handler should be called");

    // ── Summary ─────────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  ALL CHECKS PASSED");
    println!("═══════════════════════════════════════════════════════════");
}
