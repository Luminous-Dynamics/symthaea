// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for PrimitiveSystem across modules.
//!
//! Tests that PrimitiveSystem works correctly when used by:
//! - bootstrapping.rs (cognitive bootstrapping for reasoning tasks)
//! - arithmetic_engine.rs (Peano arithmetic via HDC)
//!
//! Also tests cross-feature integration: LSH + batch ops, composition algebra + cache,
//! persistence round-trip, and graph generation from composed primitives.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::{
    CompositionAlgebra, CompositionCache, HistoryEntry, PrimitiveGraph, PrimitivePersistence,
    PrimitiveSystem, PrimitiveTier,
};

// ============================================================================
// Bootstrapping Integration
// ============================================================================

mod bootstrapping_integration {
    use super::*;
    use symthaea_core::hdc::bootstrapping::{PrimitiveBootstrapper, ReasoningCategory};

    #[test]
    fn bootstrapper_loads_from_global_system() {
        let bootstrapper = PrimitiveBootstrapper::new();
        let system = PrimitiveSystem::global();

        // Bootstrapper should have access to the same total primitives
        assert_eq!(bootstrapper.total_primitives(), system.count());
    }

    #[test]
    fn bootstrapper_categories_with_primitives_match_system() {
        let bootstrapper = PrimitiveBootstrapper::new();
        let system = PrimitiveSystem::global();

        // These categories are known to have matching primitives in the system
        let known_populated = [
            ReasoningCategory::LogicalDeduction,      // AND, OR, NOT
            ReasoningCategory::MathematicalReasoning, // ZERO, ONE, SUCCESSOR, ADDITION
            ReasoningCategory::CausalInference,       // CAUSE, EFFECT, STATE_CHANGE
            ReasoningCategory::TemporalReasoning,     // BEFORE, AFTER
        ];

        for category in &known_populated {
            let prims = bootstrapper.primitives_for_category(*category);
            // Each primitive from bootstrapper should be findable in system
            for (name, hv) in prims {
                let lookup = system.get(name).unwrap_or_else(|| {
                    panic!(
                        "Bootstrapper references '{}' but system.get() returned None",
                        name
                    )
                });
                assert_eq!(
                    lookup.encoding.similarity(hv),
                    1.0,
                    "Encoding mismatch for primitive '{}'",
                    name
                );
            }
        }
    }

    #[test]
    fn bootstrapped_working_memory_is_distinct_per_subject() {
        let bootstrapper = PrimitiveBootstrapper::new();
        let question_hv = BinaryHV::random(42);

        let math_wm = bootstrapper.bootstrap_working_memory("mathematics", &question_hv);
        let ethics_wm = bootstrapper.bootstrap_working_memory("ethics", &question_hv);
        let physics_wm = bootstrapper.bootstrap_working_memory("physics", &question_hv);

        // Different subjects should yield different working memories
        let math_ethics_sim = math_wm.similarity(&ethics_wm);
        let math_physics_sim = math_wm.similarity(&physics_wm);

        // They share the question_hv so they won't be orthogonal,
        // but they should not be identical either
        assert!(math_ethics_sim < 1.0, "Math and ethics WM should differ");
        assert!(math_physics_sim < 1.0, "Math and physics WM should differ");
    }

    #[test]
    fn bootstrapper_can_use_composition_features() {
        let system = PrimitiveSystem::global();
        let bootstrapper = PrimitiveBootstrapper::new();

        // Get logical primitives from bootstrapper
        let logical = bootstrapper.primitives_for_category(ReasoningCategory::LogicalDeduction);
        let names: Vec<&str> = logical.iter().map(|(n, _)| n.as_str()).collect();

        // Use the new composition features with bootstrapper-selected primitives
        if names.len() >= 2 {
            let result = system.bind_primitives(names[0], names[1]);
            assert!(
                result.is_ok(),
                "Should be able to bind bootstrapper primitives"
            );

            let bound = result.unwrap();
            let similar = bound.find_similar(system, 3);
            assert!(
                !similar.is_empty(),
                "Composition should have similar primitives"
            );
        }

        if names.len() >= 3 {
            let seq_names: Vec<&str> = names.iter().take(3).copied().collect();
            let seq = system.encode_sequence(&seq_names);
            assert!(
                seq.is_ok(),
                "Should encode sequence of bootstrapper primitives"
            );
        }
    }
}

// ============================================================================
// Arithmetic Engine Integration
// ============================================================================

mod arithmetic_integration {
    use super::*;
    use symthaea_core::hdc::arithmetic_engine::{ArithmeticEngine, HdcNumber};

    #[test]
    fn arithmetic_engine_uses_primitive_system() {
        let mut engine = ArithmeticEngine::new();

        let zero = engine.number(0);
        assert_eq!(zero.value, 0);

        let one = engine.number(1);
        assert_eq!(one.value, 1);

        let five = engine.number(5);
        assert_eq!(five.value, 5);

        // Different numbers should have different encodings
        assert!(
            zero.encoding.similarity(&one.encoding) < 0.9,
            "Zero and one should have distinct encodings"
        );
    }

    #[test]
    fn arithmetic_zero_matches_system_primitive() {
        let system = PrimitiveSystem::global();

        let zero_prim = system.get("ZERO").expect("ZERO must exist");
        let zero_num = HdcNumber::zero(system);

        assert_eq!(
            zero_prim.encoding.similarity(&zero_num.encoding),
            1.0,
            "ZERO primitive and zero HdcNumber should have identical encoding"
        );
    }

    #[test]
    fn arithmetic_addition_produces_correct_result() {
        let mut engine = ArithmeticEngine::new();

        let result = engine.add(3, 4);
        assert_eq!(result.result.value, 7, "3 + 4 should equal 7");
        assert!(result.total_phi >= 0.0, "Phi should be non-negative");
        assert!(!result.proof.is_empty(), "Should have proof steps");
    }

    #[test]
    fn arithmetic_multiplication_produces_correct_result() {
        let mut engine = ArithmeticEngine::new();

        let result = engine.multiply(3, 4);
        assert_eq!(result.result.value, 12, "3 * 4 should equal 12");

        let prim_names: Vec<String> = result
            .proof
            .iter()
            .flat_map(|step| step.primitives_used.iter().cloned())
            .collect();

        assert!(
            prim_names
                .iter()
                .any(|n| n == "SUCCESSOR" || n == "ADDITION" || n == "MULTIPLICATION"),
            "Proof should reference system primitives, got: {:?}",
            prim_names
        );
    }

    #[test]
    fn arithmetic_number_construction_is_deterministic() {
        let system = PrimitiveSystem::global();

        // Build number 3 via manual composition: S(S(S(ZERO)))
        let zero = system.get("ZERO").unwrap();
        let succ = system.get("SUCCESSOR").unwrap();

        let one = succ.encoding.bind(&zero.encoding);
        let two = succ.encoding.bind(&one);
        let three_manual = succ.encoding.bind(&two);

        // Build via ArithmeticEngine
        let mut engine = ArithmeticEngine::new();
        let three_arith = engine.number(3);

        // They should produce the same encoding
        assert_eq!(
            three_arith.encoding.similarity(&three_manual),
            1.0,
            "Arithmetic engine and manual composition should produce identical number encodings"
        );
    }
}

// ============================================================================
// Cross-Feature Integration: LSH + Batch + Composition
// ============================================================================

mod cross_feature_integration {
    use super::*;

    #[test]
    fn lsh_finds_self_for_known_primitives() {
        let system = PrimitiveSystem::global();

        // Build LSH index with generous parameters
        let lsh = system.build_lsh_index(32, 8);

        // For primitives, LSH should find self as top match
        // Use well-known primitives
        let test_names = ["ZERO", "SET", "AND", "MASS", "POINT"];
        let mut found_self = 0;

        for name in &test_names {
            if let Some(prim) = system.get(name) {
                let lsh_result = system.find_similar_lsh(&prim.encoding, 5, &lsh);
                if lsh_result.iter().any(|(n, _)| n == *name) {
                    found_self += 1;
                }
            }
        }

        // LSH is approximate; with 32 bands we expect good recall
        assert!(
            found_self >= 3,
            "LSH should find self-match for most primitives ({}/{})",
            found_self,
            test_names.len()
        );
    }

    #[test]
    fn batch_bind_matches_single_bind() {
        let system = PrimitiveSystem::global();

        let pairs = [("CAUSE", "EFFECT"), ("AND", "OR"), ("BEFORE", "AFTER")];

        let batch_results = system.batch_bind(&pairs);

        for (i, (a, b)) in pairs.iter().enumerate() {
            let single = system.bind_primitives(a, b);
            match (&batch_results[i], &single) {
                (Ok(batch_r), Ok(single_r)) => {
                    assert_eq!(
                        batch_r.encoding.similarity(&single_r.encoding),
                        1.0,
                        "Batch bind should match single bind for {} ^ {}",
                        a,
                        b
                    );
                }
                (Err(_), Err(_)) => {}
                _ => panic!(
                    "Batch and single should agree on success/failure for {} ^ {}",
                    a, b
                ),
            }
        }
    }

    #[test]
    fn composition_algebra_matches_direct_bind() {
        let system = PrimitiveSystem::global();
        let mut algebra = CompositionAlgebra::new();

        let result = algebra.define("CAUSATION", "CAUSE ^ EFFECT", system);
        assert!(
            result.is_ok(),
            "Should define CAUSATION: {:?}",
            result.err()
        );

        let direct = system.bind_primitives("CAUSE", "EFFECT").unwrap();

        let algebra_enc = algebra.get("CAUSATION").unwrap();
        assert_eq!(
            algebra_enc.encoding.similarity(&direct.encoding),
            1.0,
            "Algebra and direct bind should produce identical encodings"
        );
    }

    #[test]
    fn composition_cache_hit_returns_same_result() {
        let system = PrimitiveSystem::global();
        let mut cache = CompositionCache::new(100);

        // First call (miss)
        let cached_1 = cache.bind_cached(system, "AND", "OR").unwrap();

        // Direct call
        let direct = system.bind_primitives("AND", "OR").unwrap();

        assert_eq!(
            cached_1.encoding.similarity(&direct.encoding),
            1.0,
            "Cached and direct should match"
        );

        // Second call (hit)
        let cached_2 = cache.bind_cached(system, "AND", "OR").unwrap();
        assert_eq!(
            cached_2.encoding.similarity(&cached_1.encoding),
            1.0,
            "Cache hit should return same result"
        );

        let stats = cache.stats();
        assert_eq!(stats.hits, 1, "Should have 1 cache hit");
        assert_eq!(stats.misses, 1, "Should have 1 cache miss");
    }

    #[test]
    fn similarity_matrix_is_symmetric_with_unit_diagonal() {
        let system = PrimitiveSystem::global();

        let names = ["AND", "OR", "NOT", "SET"];
        let matrix = system.similarity_matrix(&names);

        assert_eq!(matrix.len(), 4);

        // Diagonal should be 1.0
        for i in 0..4 {
            assert!(
                (matrix[i][i] - 1.0).abs() < 0.001,
                "Diagonal [{},{}] should be 1.0, got {}",
                i,
                i,
                matrix[i][i]
            );
        }

        // Symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 0.001,
                    "Matrix should be symmetric: [{},{}]={} != [{},{}]={}",
                    i,
                    j,
                    matrix[i][j],
                    j,
                    i,
                    matrix[j][i]
                );
            }
        }
    }

    #[test]
    fn graph_from_tier() {
        let system = PrimitiveSystem::global();

        let graph = PrimitiveGraph::from_tier(system, PrimitiveTier::Mathematical, 0.4);

        let stats = graph.stats();
        assert!(stats.node_count > 0, "Graph should have nodes");

        let dot = graph.to_dot();
        assert!(
            dot.contains("digraph"),
            "DOT output should contain 'digraph'"
        );
    }

    #[test]
    fn persistence_round_trip() {
        let system = PrimitiveSystem::global();

        let mut algebra = CompositionAlgebra::new();
        algebra
            .define("TEST_COMP", "CAUSE ^ EFFECT", system)
            .unwrap();

        let history = vec![HistoryEntry {
            operation: "bind CAUSE EFFECT".to_string(),
            result_match: "CAUSE".to_string(),
            similarity: 0.52,
        }];

        let persistence = PrimitivePersistence::new();
        let tmp_path = std::env::temp_dir().join("symthaea_integration_test.json");
        let tmp_str = tmp_path.to_str().unwrap();

        persistence
            .save_session(tmp_str, &algebra, &history, Some("Integration test"))
            .unwrap();
        let (loaded_algebra, loaded_history) = persistence.load_session(tmp_str, system).unwrap();

        assert_eq!(loaded_algebra.list().len(), 1);
        assert!(loaded_algebra.get("TEST_COMP").is_some());
        assert_eq!(loaded_history.len(), 1);
        assert_eq!(loaded_history[0].operation, "bind CAUSE EFFECT");

        // Encoding should match
        let orig_enc = algebra.get("TEST_COMP").unwrap();
        let loaded_enc = loaded_algebra.get("TEST_COMP").unwrap();
        assert_eq!(
            orig_enc.encoding.similarity(&loaded_enc.encoding),
            1.0,
            "Loaded composition encoding should match original"
        );

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn algebra_export_import_preserves_compositions() {
        let system = PrimitiveSystem::global();

        let mut original = CompositionAlgebra::new();
        original.define("A", "CAUSE ^ EFFECT", system).unwrap();
        original.define("B", "AND + OR + NOT", system).unwrap();

        let exported = original.export();
        assert_eq!(exported.len(), 2);

        let mut imported = CompositionAlgebra::new();
        let result = imported.import(&exported, system);
        assert!(result.is_ok());

        assert_eq!(imported.list().len(), 2);

        let orig_a = original.get("A").unwrap();
        let imp_a = imported.get("A").unwrap();
        assert_eq!(
            orig_a.encoding.similarity(&imp_a.encoding),
            1.0,
            "Imported composition should have same encoding"
        );
    }

    #[test]
    fn composed_primitives_are_queryable() {
        let system = PrimitiveSystem::global();

        let bound = system.bind_primitives("CAUSE", "EFFECT").unwrap();
        let (closest_name, similarity) = system.query(&bound.encoding);

        assert!(
            !closest_name.is_empty(),
            "Query should return a primitive name"
        );
        assert!(
            similarity > 0.0 && similarity <= 1.0,
            "Similarity should be in (0, 1]"
        );
    }

    #[test]
    fn pairwise_similarities_consistent() {
        let system = PrimitiveSystem::global();

        let encodings: Vec<BinaryHV> = ["CAUSE", "EFFECT", "BEFORE", "AFTER"]
            .iter()
            .filter_map(|n| system.get(n))
            .map(|p| p.encoding)
            .collect();

        let pairs = system.pairwise_similarities(&encodings);

        let n = encodings.len();
        assert_eq!(pairs.len(), n * (n - 1) / 2);

        for (i, j, sim) in &pairs {
            assert!(
                *sim >= 0.0 && *sim <= 1.0,
                "Similarity between {} and {} should be in [0,1], got {}",
                i,
                j,
                sim
            );
        }
    }
}

// ============================================================================
// Global Primitive System Consistency
// ============================================================================

mod system_consistency {
    use super::*;

    #[test]
    fn global_returns_same_instance() {
        let system1 = PrimitiveSystem::global();
        let system2 = PrimitiveSystem::global();

        assert_eq!(system1.count(), system2.count());

        let p1 = system1.get("ZERO").unwrap();
        let p2 = system2.get("ZERO").unwrap();
        assert_eq!(p1.encoding.similarity(&p2.encoding), 1.0);
    }

    #[test]
    fn initialized_tiers_have_primitives() {
        let system = PrimitiveSystem::global();

        // These tiers are known to be initialized by the system
        let initialized_tiers = [
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
        ];

        for tier in &initialized_tiers {
            let count = system.count_tier(*tier);
            assert!(
                count > 0,
                "Tier {:?} should have at least one primitive, got 0",
                tier
            );
        }
    }

    #[test]
    fn primitive_encodings_are_deterministic() {
        let sys1 = PrimitiveSystem::new();
        let sys2 = PrimitiveSystem::new();

        let test_names = [
            "ZERO",
            "ONE",
            "SUCCESSOR",
            "AND",
            "OR",
            "CAUSE",
            "EFFECT",
            "MASS",
            "ENERGY",
            "POINT",
            "SELF",
        ];

        for name in &test_names {
            if let (Some(p1), Some(p2)) = (sys1.get(name), sys2.get(name)) {
                assert_eq!(
                    p1.encoding.similarity(&p2.encoding),
                    1.0,
                    "Primitive '{}' should have deterministic encoding",
                    name
                );
            }
        }
    }

    #[test]
    fn no_duplicate_primitive_names() {
        let system = PrimitiveSystem::global();
        let names = system.all_primitive_names();

        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(*name), "Duplicate primitive name: '{}'", name);
        }
    }

    #[test]
    fn mathematical_primitives_near_orthogonal() {
        let system = PrimitiveSystem::global();

        // Base primitives with different seeds should be near-orthogonal (~0.5 similarity)
        let violations = system.validate_tier_orthogonality(PrimitiveTier::Mathematical, 0.65);
        let tier_count = system.count_tier(PrimitiveTier::Mathematical);

        if tier_count >= 2 {
            let max_pairs = tier_count * (tier_count - 1) / 2;
            let violation_rate = if max_pairs > 0 {
                violations.len() as f32 / max_pairs as f32
            } else {
                0.0
            };

            assert!(
                violation_rate < 0.15,
                "Mathematical tier should have <15% pairs above 0.65 similarity (got {:.1}%, {} violations out of {} pairs)",
                violation_rate * 100.0,
                violations.len(),
                max_pairs
            );
        }
    }

    #[test]
    fn system_has_expected_primitive_count() {
        let system = PrimitiveSystem::global();
        // The system summary in prior sessions says ~201 primitives
        assert!(
            system.count() >= 100,
            "System should have at least 100 primitives, got {}",
            system.count()
        );
    }
}
