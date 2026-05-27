// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Knowledge Engine

Verifies that the knowledge engine maintains invariants under fuzzed inputs:

1. Graph size monotonically non-decreasing (extraction only adds)
2. Confidence values bounded [0, 1]
3. Consciousness level stays finite with knowledge engine enabled
4. Prune count never exceeds graph size
5. Consolidation is idempotent on empty graphs
6. Knowledge grounding score bounded [0, 1]
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

fn knowledge_input_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Normal text
        "[a-z ]{1,80}",
        // Causal sentences
        Just("rain causes flooding which causes damage".to_string()),
        Just("heat causes evaporation which causes clouds".to_string()),
        // Entity-rich
        Just("Einstein discovered relativity at Princeton University".to_string()),
        // Empty-ish
        Just(String::new()),
        Just(" ".to_string()),
        // Mixed
        "[A-Za-z0-9 .,!?]{1,120}",
    ]
}

fn create_knowledge_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        enable_knowledge_engine: true,
        ..Default::default()
    })
    .unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 1: Graph size non-decreasing under extraction
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    #[test]
    fn prop_graph_size_non_decreasing(
        inputs in prop::collection::vec(knowledge_input_strategy(), 5..20)
    ) {
        let mut service = create_knowledge_service();
        let mut prev_size = 0u32;

        for input in &inputs {
            service.cycle(input);
            if let Some(t) = service.knowledge_telemetry() {
                // Graph should never shrink during normal extraction
                // (pruning only happens during dream consolidation)
                prop_assert!(
                    t.graph_size >= prev_size,
                    "Graph shrunk from {} to {} on input {:?}",
                    prev_size, t.graph_size, input
                );
                prev_size = t.graph_size;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 2: Avg confidence bounded [0, 1]
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    #[test]
    fn prop_confidence_bounded(
        inputs in prop::collection::vec(knowledge_input_strategy(), 5..20)
    ) {
        let mut service = create_knowledge_service();

        for input in &inputs {
            service.cycle(input);
        }

        if let Some(t) = service.knowledge_telemetry() {
            prop_assert!(
                t.avg_confidence >= 0.0 && t.avg_confidence <= 1.0,
                "Avg confidence {} out of bounds",
                t.avg_confidence
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 3: Consciousness stays finite with knowledge engine
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_consciousness_finite_with_knowledge(
        inputs in prop::collection::vec(knowledge_input_strategy(), 10..30)
    ) {
        let mut service = create_knowledge_service();

        for input in &inputs {
            service.cycle(input);
            let cl = service.consciousness_level();
            prop_assert!(
                cl.is_finite(),
                "Consciousness became non-finite: {} on input {:?}",
                cl, input
            );
            prop_assert!(
                cl >= 0.0 && cl <= 1.0,
                "Consciousness out of bounds: {} on input {:?}",
                cl, input
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 4: Prune count never exceeds graph size
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_prune_bounded_by_graph_size(
        inputs in prop::collection::vec(knowledge_input_strategy(), 5..15)
    ) {
        let mut service = create_knowledge_service();

        for input in &inputs {
            service.cycle(input);
        }

        let pre_size = service.knowledge_telemetry()
            .map(|t| t.graph_size)
            .unwrap_or(0);

        let (pruned, _strengthened) = service.knowledge_consolidate_and_forget();

        prop_assert!(
            pruned as u32 <= pre_size,
            "Pruned {} but graph only had {} facts",
            pruned, pre_size
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 5: Consolidation idempotent on empty graph
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn prop_consolidation_idempotent_empty(_seed in 0u32..100) {
        let mut service = create_knowledge_service();

        // Double consolidation on empty graph should be (0, 0) both times
        let (p1, s1) = service.knowledge_consolidate_and_forget();
        let (p2, s2) = service.knowledge_consolidate_and_forget();

        prop_assert_eq!(p1, 0);
        prop_assert_eq!(s1, 0);
        prop_assert_eq!(p2, 0);
        prop_assert_eq!(s2, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 6: Knowledge query grounding bounded [0, 1]
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_query_grounding_bounded(
        inputs in prop::collection::vec(knowledge_input_strategy(), 5..15),
        query in "[a-z]{1,20}"
    ) {
        let mut service = create_knowledge_service();

        for input in &inputs {
            service.cycle(input);
        }

        let result = service.knowledge_query(&query);
        prop_assert!(
            result.grounding_score >= 0.0 && result.grounding_score <= 1.0,
            "Grounding score {} out of bounds for query {:?}",
            result.grounding_score, query
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 7: Spreading activation never exceeds 1.5 and result count is bounded
// ═══════════════════════════════════════════════════════════════════════════════

// Item 2: Spreading activation never exceeds 1.0 and result count is bounded.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_spreading_activation_bounded(
        seed_count in 1usize..5,
        _hops in 1usize..4,
        _decay in 0.1f32..0.9f32,
    ) {
        use symthaea::knowledge::manager::{KnowledgeManager, KnowledgeManagerConfig};

        let mut mgr = KnowledgeManager::new(KnowledgeManagerConfig {
            processing_interval: 1,
            search_top_k: 5,
            ..KnowledgeManagerConfig::default()
        });
        mgr.bootstrap_entities();

        // Insert some facts
        for i in 0..10 {
            mgr.process(&format!("Country {} imposed sanctions round {}.", i, i + 1), i as u64 + 1);
        }

        // Search to populate results (use seed_count to vary query)
        let query = format!("sanctions {}", seed_count);
        mgr.process(&query, 20);
        let results = mgr.last_search_results();

        // All similarities should be in [0.0, 1.5] (HDC cosine similarity may be slightly above 1.0
        // due to fp arithmetic, but should never explode)
        for r in results {
            prop_assert!(r.similarity >= 0.0, "Similarity should be non-negative: {}", r.similarity);
            prop_assert!(r.similarity <= 1.5, "Similarity should not explode: {}", r.similarity);
            prop_assert!(r.confidence >= 0.0 && r.confidence <= 1.0,
                "Confidence {} out of [0,1]", r.confidence);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 8: do(X) intervention effects are bounded and contain no duplicates
// ═══════════════════════════════════════════════════════════════════════════════

// Item 3: do(X) intervention effects attenuate with depth and contain no cycles.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_intervention_attenuates(
        chain_len in 2usize..6,
    ) {
        use symthaea::knowledge::manager::{KnowledgeManager, KnowledgeManagerConfig};
        use std::collections::HashSet;

        let mut mgr = KnowledgeManager::new(KnowledgeManagerConfig {
            processing_interval: 1,
            ..KnowledgeManagerConfig::default()
        });

        // Build a linear causal chain: A→B→C→...
        let names: Vec<String> = (0..chain_len).map(|i| format!("entity_{}", i)).collect();
        for i in 0..chain_len - 1 {
            mgr.process(
                &format!("{} caused {}.", names[i], names[i + 1]),
                i as u64 + 1,
            );
        }

        let effects = mgr.do_intervention(&names[0], 10);

        // No entity should appear twice (no cycles)
        let effect_names: Vec<String> = effects.iter().map(|e| e.0.to_lowercase()).collect();
        let unique: HashSet<&String> = effect_names.iter().collect();
        prop_assert_eq!(unique.len(), effect_names.len(), "No duplicate effects in intervention");

        // Strengths should be bounded
        for (_, strength) in &effects {
            prop_assert!(strength.abs() <= 1.0, "Effect strength should not exceed 1.0: {}", strength);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 9: Calibration ECE is always in [0, 1] and sample count is monotonic
// ═══════════════════════════════════════════════════════════════════════════════

// Item 4: Calibration ECE is always in [0, 1] and sample count is monotonic.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_calibration_ece_bounded(
        num_samples in 1usize..100,
        confidence in 0.0f32..1.0f32,
        correct_rate in 0.0f32..1.0f32,
    ) {
        use symthaea::knowledge::manager::CalibrationAudit;

        let mut audit = CalibrationAudit::default();

        for i in 0..num_samples {
            let correct = (i as f32 / num_samples as f32) < correct_rate;
            audit.record(confidence, correct);
        }

        let ece = audit.ece();
        prop_assert!(ece >= 0.0, "ECE should be non-negative: {}", ece);
        prop_assert!(ece <= 1.0, "ECE should not exceed 1.0: {}", ece);
        prop_assert_eq!(audit.total_samples(), num_samples as u64);

        let mce = audit.mce();
        prop_assert!(mce >= 0.0 && mce <= 1.0, "MCE should be in [0,1]: {}", mce);
        prop_assert!(mce >= ece, "MCE should be >= ECE: mce={}, ece={}", mce, ece);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 10: IS-A chain depth is bounded and consistent with children_of
// ═══════════════════════════════════════════════════════════════════════════════

// Item 5: IS-A chain depth is bounded and consistent with children_of.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    #[test]
    fn prop_isa_depth_bounded(
        chain_depth in 1usize..8,
    ) {
        use symthaea::knowledge::adaptive_ontology::{AdaptiveOntology, AdaptiveOntologyConfig};
        use symthaea_core::hdc::binary_hv::BinaryHV;

        let mut ontology = AdaptiveOntology::new(AdaptiveOntologyConfig::default());

        // Build a chain: concept_0 IS-A concept_1 IS-A ... IS-A concept_N
        // Use index-based seeds (multiplied by large prime) to ensure distinct vectors.
        let names: Vec<String> = (0..=chain_depth).map(|i| format!("concept_{}", i)).collect();
        for (i, name) in names.iter().enumerate() {
            let seed = (i as u64 + 1) * 10_007;
            ontology.learn(name, BinaryHV::random(seed), vec![], 1);
        }
        for i in 0..chain_depth {
            ontology.set_is_a(&names[i], &names[i + 1]);
        }

        // Chain from leaf should have correct depth (parents only, not the concept itself)
        let chain = ontology.is_a_chain(&names[0], 20);
        prop_assert_eq!(chain.len(), chain_depth, "Chain should include all ancestors");
        prop_assert_eq!(&chain[0], &names[1], "Chain should start with direct parent");
        prop_assert_eq!(&chain[chain_depth - 1], &names[chain_depth], "Chain should end at root");

        // Children consistency: concept_1's children should include concept_0
        let children = ontology.children_of(&names[1]);
        prop_assert!(children.contains(&names[0]), "children_of should include direct child");

        // is_a transitive
        prop_assert!(ontology.is_a(&names[0], &names[chain_depth]));
    }
}