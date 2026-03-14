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
