// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Generation curriculum for Symthaea's School system
//!
//! Defines progressive learning objectives for code generation skill,
//! from simple arithmetic function bodies to cross-language synthesis
//! and self-correcting compilation loops.

use super::curriculum::Curriculum;
use super::objective::{Difficulty, Domain, LearningObjective};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const CODEGEN_DOMAIN: &str = "CodeGeneration";
const CURRICULUM_ID: &str = "code-generation";
const ADVANCED_CURRICULUM_ID: &str = "code-generation-advanced";

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 1: FOUNDATIONS (Beginner)
// ═══════════════════════════════════════════════════════════════════════════════

/// IDs for Tier 1 objectives (no prerequisites).
const TIER1_IDS: &[&str] = &[
    "codegen_simple_arithmetic",
    "codegen_string_ops",
    "codegen_boolean_checks",
    "codegen_basic_collections",
];

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 2: COMPOSITION (Intermediate)
// ═══════════════════════════════════════════════════════════════════════════════

/// IDs for Tier 2 objectives (require all of Tier 1).
const TIER2_IDS: &[&str] = &[
    "codegen_composed_chains",
    "codegen_struct_impl",
    "codegen_error_handling",
    "codegen_closures_hof",
    "codegen_test_generation",
];

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 3: ADVANCED
// ═══════════════════════════════════════════════════════════════════════════════

/// IDs for Tier 3 objectives (require all of Tier 2).
const TIER3_IDS: &[&str] = &[
    "codegen_algorithm_sorting",
    "codegen_algorithm_search",
    "codegen_algorithm_dp",
    "codegen_code_modification",
    "codegen_multi_entity",
    "codegen_import_inference",
];

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 4: MASTERY (Expert)
// ═══════════════════════════════════════════════════════════════════════════════

/// IDs for Tier 4 objectives (require all of Tier 3).
const TIER4_IDS: &[&str] = &[
    "codegen_graph_algorithms",
    "codegen_concurrent_code",
    "codegen_generic_abstractions",
    "codegen_self_correction",
    "codegen_cross_language",
];

// ═══════════════════════════════════════════════════════════════════════════════
// PRIMARY CURRICULUM
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the core code generation curriculum with ~20 objectives across 4 tiers.
///
/// Tier progression:
/// 1. Foundations (Beginner) — single-function body generation
/// 2. Composition (Intermediate) — multi-part patterns, error handling, tests
/// 3. Advanced — algorithms, structural transforms, multi-entity systems
/// 4. Mastery (Expert) — graphs, concurrency, generics, self-correction, cross-language
pub fn code_generation_curriculum() -> Curriculum {
    Curriculum::new(CURRICULUM_ID, "Code Generation")
        .with_description(
            "Progressive curriculum for code generation skill, from simple function \
             bodies through algorithmic synthesis to self-correcting cross-language output",
        )
        .with_tags(&["code", "generation", "synthesis", "programming"])
        // ── Tier 1: Foundations ──────────────────────────────────────────────
        .with_objective(
            LearningObjective::new("codegen_simple_arithmetic", "Simple Arithmetic Functions")
                .with_description(
                    "Generate simple arithmetic function bodies (add, subtract, multiply)",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["code", "generation", "arithmetic", "functions"])
                .with_estimated_minutes(15),
        )
        .with_objective(
            LearningObjective::new("codegen_string_ops", "String Manipulation Functions")
                .with_description(
                    "Generate string manipulation functions (reverse, uppercase, trim)",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["code", "generation", "string", "text"])
                .with_estimated_minutes(15),
        )
        .with_objective(
            LearningObjective::new("codegen_boolean_checks", "Boolean Predicate Functions")
                .with_description(
                    "Generate boolean predicate functions (is_even, is_positive, is_empty)",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["code", "generation", "boolean", "predicates"])
                .with_estimated_minutes(15),
        )
        .with_objective(
            LearningObjective::new("codegen_basic_collections", "Basic Collection Operations")
                .with_description("Generate basic collection operations (sort, filter, sum)")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["code", "generation", "collections", "vec"])
                .with_estimated_minutes(20),
        )
        // ── Tier 2: Composition ─────────────────────────────────────────────
        .with_objective(
            LearningObjective::new("codegen_composed_chains", "Composed Iterator Chains")
                .with_description("Generate composed iterator chains (filter+map+collect)")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["code", "generation", "iterators", "composition"])
                .with_estimated_minutes(30),
        )
        .with_objective(
            LearningObjective::new("codegen_struct_impl", "Struct Definitions with Impl Blocks")
                .with_description("Generate struct definitions with impl blocks and methods")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["code", "generation", "struct", "impl", "methods"])
                .with_estimated_minutes(30),
        )
        .with_objective(
            LearningObjective::new("codegen_error_handling", "Error Handling Patterns")
                .with_description("Generate functions with Result/Option error handling")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["code", "generation", "error", "result", "option"])
                .with_estimated_minutes(30),
        )
        .with_objective(
            LearningObjective::new(
                "codegen_closures_hof",
                "Higher-Order Functions and Closures",
            )
            .with_description("Generate higher-order functions and closure patterns")
            .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
            .with_difficulty(Difficulty::Intermediate)
            .with_prerequisites(TIER1_IDS)
            .with_tags(&["code", "generation", "closures", "higher-order"])
            .with_estimated_minutes(30),
        )
        .with_objective(
            LearningObjective::new("codegen_test_generation", "Test Assertion Generation")
                .with_description("Generate meaningful test assertions from function specs")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["code", "generation", "testing", "assertions"])
                .with_estimated_minutes(30),
        )
        // ── Tier 3: Advanced ────────────────────────────────────────────────
        .with_objective(
            LearningObjective::new(
                "codegen_algorithm_sorting",
                "Sorting Algorithm Implementations",
            )
            .with_description("Generate sorting algorithm implementations")
            .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
            .with_difficulty(Difficulty::Advanced)
            .with_prerequisites(TIER2_IDS)
            .with_tags(&["code", "generation", "algorithms", "sorting"])
            .with_estimated_minutes(45),
        )
        .with_objective(
            LearningObjective::new(
                "codegen_algorithm_search",
                "Search Algorithm Implementations",
            )
            .with_description("Generate search algorithm implementations")
            .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
            .with_difficulty(Difficulty::Advanced)
            .with_prerequisites(TIER2_IDS)
            .with_tags(&["code", "generation", "algorithms", "search"])
            .with_estimated_minutes(45),
        )
        .with_objective(
            LearningObjective::new("codegen_algorithm_dp", "Dynamic Programming Solutions")
                .with_description("Generate dynamic programming solutions")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(TIER2_IDS)
                .with_tags(&["code", "generation", "algorithms", "dynamic-programming"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new(
                "codegen_code_modification",
                "Structural Code Transformations",
            )
            .with_description("Apply structural transformations to existing code")
            .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
            .with_difficulty(Difficulty::Advanced)
            .with_prerequisites(TIER2_IDS)
            .with_tags(&["code", "generation", "refactoring", "transformation"])
            .with_estimated_minutes(45),
        )
        .with_objective(
            LearningObjective::new("codegen_multi_entity", "Multi-Type System Generation")
                .with_description("Generate complex multi-type systems (struct+trait+impl)")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(TIER2_IDS)
                .with_tags(&["code", "generation", "traits", "systems", "architecture"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("codegen_import_inference", "Import Statement Inference")
                .with_description("Correctly infer and add required import statements")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(TIER2_IDS)
                .with_tags(&["code", "generation", "imports", "modules"])
                .with_estimated_minutes(30),
        )
        // ── Tier 4: Mastery ─────────────────────────────────────────────────
        .with_objective(
            LearningObjective::new("codegen_graph_algorithms", "Graph Algorithms")
                .with_description("Generate graph traversal and shortest-path algorithms")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(TIER3_IDS)
                .with_tags(&[
                    "code",
                    "generation",
                    "graphs",
                    "algorithms",
                    "bfs",
                    "dijkstra",
                ])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("codegen_concurrent_code", "Concurrent and Async Code")
                .with_description("Generate concurrent/async code with proper synchronization")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(TIER3_IDS)
                .with_tags(&["code", "generation", "async", "concurrency", "sync"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new(
                "codegen_generic_abstractions",
                "Generic and Trait-Bounded Abstractions",
            )
            .with_description("Generate generic functions and trait-bounded abstractions")
            .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
            .with_difficulty(Difficulty::Expert)
            .with_prerequisites(TIER3_IDS)
            .with_tags(&[
                "code",
                "generation",
                "generics",
                "trait-bounds",
                "abstractions",
            ])
            .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("codegen_self_correction", "Self-Correcting Compilation")
                .with_description("Detect and fix compilation errors without external help")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(TIER3_IDS)
                .with_tags(&[
                    "code",
                    "generation",
                    "self-correction",
                    "compilation",
                    "debugging",
                ])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("codegen_cross_language", "Cross-Language Generation")
                .with_description("Generate equivalent code across Rust, Python, and Nix")
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(TIER3_IDS)
                .with_tags(&[
                    "code",
                    "generation",
                    "cross-language",
                    "rust",
                    "python",
                    "nix",
                ])
                .with_estimated_minutes(75),
        )
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED CURRICULUM
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the advanced code generation curriculum (requires `code-generation`).
///
/// Focuses on SSM distillation, test-first synthesis, property-based testing,
/// deep code understanding, and optimization passes.
pub fn code_generation_advanced_curriculum() -> Curriculum {
    Curriculum::new(ADVANCED_CURRICULUM_ID, "Advanced Code Generation")
        .with_description(
            "Expert-level code generation covering SSM distillation targets, \
             test-first synthesis, property-based test generation, deep code \
             understanding, and automated optimization",
        )
        .with_prerequisite_curriculum(CURRICULUM_ID)
        .with_tags(&["code", "generation", "advanced", "expert"])
        .with_objective(
            LearningObjective::new(
                "codegen_ssm_distillation",
                "SSM Distillation Target Generation",
            )
            .with_description(
                "Generate training targets for SSM language-model distillation \
                 by producing structurally valid code from HDC-encoded thought vectors",
            )
            .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
            .with_difficulty(Difficulty::Expert)
            .with_tags(&["code", "generation", "ssm", "distillation", "broca"])
            .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new("codegen_test_first_synthesis", "Test-First Code Synthesis")
                .with_description(
                    "Synthesize function implementations from test suites alone, \
                 iterating until all assertions pass without seeing a reference solution",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_tags(&["code", "generation", "tdd", "synthesis", "test-driven"])
                .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new("codegen_property_testing", "Property-Based Test Generation")
                .with_description(
                    "Generate proptest/quickcheck strategies and property assertions \
                 that discover edge cases in arbitrary implementations",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_tags(&["code", "generation", "proptest", "quickcheck", "properties"])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("codegen_deep_understanding", "Deep Code Understanding")
                .with_description(
                    "Analyze complex codebases to infer invariants, data flow, \
                 and architectural intent before generating compatible extensions",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_tags(&[
                    "code",
                    "understanding",
                    "analysis",
                    "architecture",
                    "invariants",
                ])
                .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new("codegen_optimization", "Automated Code Optimization")
                .with_description(
                    "Apply performance optimizations (vectorization, cache locality, \
                 allocation elimination) to generated code while preserving semantics",
                )
                .with_domain(Domain::Custom(CODEGEN_DOMAIN.into()))
                .with_difficulty(Difficulty::Expert)
                .with_tags(&["code", "generation", "optimization", "performance", "simd"])
                .with_estimated_minutes(90),
        )
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_code_generation_objective_count() {
        let curriculum = code_generation_curriculum();
        // 4 (Tier 1) + 5 (Tier 2) + 6 (Tier 3) + 5 (Tier 4) = 20
        assert_eq!(curriculum.objectives.len(), 20);
    }

    #[test]
    fn test_advanced_curriculum_objective_count() {
        let curriculum = code_generation_advanced_curriculum();
        assert_eq!(curriculum.objectives.len(), 5);
    }

    #[test]
    fn test_all_ids_unique() {
        let curriculum = code_generation_curriculum();
        let ids: HashSet<&str> = curriculum
            .objectives
            .iter()
            .map(|o| o.id.as_str())
            .collect();
        assert_eq!(
            ids.len(),
            curriculum.objectives.len(),
            "duplicate objective IDs found"
        );
    }

    #[test]
    fn test_advanced_ids_unique() {
        let curriculum = code_generation_advanced_curriculum();
        let ids: HashSet<&str> = curriculum
            .objectives
            .iter()
            .map(|o| o.id.as_str())
            .collect();
        assert_eq!(
            ids.len(),
            curriculum.objectives.len(),
            "duplicate objective IDs found"
        );
    }

    #[test]
    fn test_no_id_collision_across_curricula() {
        let base = code_generation_curriculum();
        let adv = code_generation_advanced_curriculum();
        let base_ids: HashSet<&str> = base.objectives.iter().map(|o| o.id.as_str()).collect();
        let adv_ids: HashSet<&str> = adv.objectives.iter().map(|o| o.id.as_str()).collect();
        let overlap: Vec<&&str> = base_ids.intersection(&adv_ids).collect();
        assert!(
            overlap.is_empty(),
            "ID collision across curricula: {:?}",
            overlap
        );
    }

    #[test]
    fn test_prerequisites_valid() {
        let curriculum = code_generation_curriculum();
        let all_ids: HashSet<&str> = curriculum
            .objectives
            .iter()
            .map(|o| o.id.as_str())
            .collect();

        for obj in &curriculum.objectives {
            for prereq in &obj.prerequisites {
                assert!(
                    all_ids.contains(prereq.as_str()),
                    "objective '{}' has unknown prerequisite '{}'",
                    obj.id,
                    prereq,
                );
            }
        }
    }

    #[test]
    fn test_tier1_has_no_prerequisites() {
        let curriculum = code_generation_curriculum();
        let tier1_set: HashSet<&str> = TIER1_IDS.iter().copied().collect();

        for obj in &curriculum.objectives {
            if tier1_set.contains(obj.id.as_str()) {
                assert!(
                    obj.prerequisites.is_empty(),
                    "Tier 1 objective '{}' should have no prerequisites but has {:?}",
                    obj.id,
                    obj.prerequisites,
                );
            }
        }
    }

    #[test]
    fn test_tier2_requires_tier1() {
        let curriculum = code_generation_curriculum();
        let tier1_set: HashSet<&str> = TIER1_IDS.iter().copied().collect();
        let tier2_set: HashSet<&str> = TIER2_IDS.iter().copied().collect();

        for obj in &curriculum.objectives {
            if tier2_set.contains(obj.id.as_str()) {
                let prereq_set: HashSet<&str> =
                    obj.prerequisites.iter().map(|s| s.as_str()).collect();
                assert!(
                    tier1_set.is_subset(&prereq_set),
                    "Tier 2 objective '{}' should require all Tier 1 IDs",
                    obj.id,
                );
            }
        }
    }

    #[test]
    fn test_tier3_requires_tier2() {
        let curriculum = code_generation_curriculum();
        let tier2_set: HashSet<&str> = TIER2_IDS.iter().copied().collect();
        let tier3_set: HashSet<&str> = TIER3_IDS.iter().copied().collect();

        for obj in &curriculum.objectives {
            if tier3_set.contains(obj.id.as_str()) {
                let prereq_set: HashSet<&str> =
                    obj.prerequisites.iter().map(|s| s.as_str()).collect();
                assert!(
                    tier2_set.is_subset(&prereq_set),
                    "Tier 3 objective '{}' should require all Tier 2 IDs",
                    obj.id,
                );
            }
        }
    }

    #[test]
    fn test_tier4_requires_tier3() {
        let curriculum = code_generation_curriculum();
        let tier3_set: HashSet<&str> = TIER3_IDS.iter().copied().collect();
        let tier4_set: HashSet<&str> = TIER4_IDS.iter().copied().collect();

        for obj in &curriculum.objectives {
            if tier4_set.contains(obj.id.as_str()) {
                let prereq_set: HashSet<&str> =
                    obj.prerequisites.iter().map(|s| s.as_str()).collect();
                assert!(
                    tier3_set.is_subset(&prereq_set),
                    "Tier 4 objective '{}' should require all Tier 3 IDs",
                    obj.id,
                );
            }
        }
    }

    #[test]
    fn test_all_objectives_have_codegen_domain() {
        let base = code_generation_curriculum();
        let adv = code_generation_advanced_curriculum();

        for obj in base.objectives.iter().chain(adv.objectives.iter()) {
            assert_eq!(
                obj.domain,
                Domain::Custom(CODEGEN_DOMAIN.into()),
                "objective '{}' should be in CodeGeneration domain",
                obj.id,
            );
        }
    }

    #[test]
    fn test_difficulty_progression() {
        let curriculum = code_generation_curriculum();
        let tier1_set: HashSet<&str> = TIER1_IDS.iter().copied().collect();
        let tier2_set: HashSet<&str> = TIER2_IDS.iter().copied().collect();
        let tier3_set: HashSet<&str> = TIER3_IDS.iter().copied().collect();
        let tier4_set: HashSet<&str> = TIER4_IDS.iter().copied().collect();

        for obj in &curriculum.objectives {
            let id = obj.id.as_str();
            if tier1_set.contains(id) {
                assert_eq!(
                    obj.difficulty,
                    Difficulty::Beginner,
                    "{id} should be Beginner"
                );
            } else if tier2_set.contains(id) {
                assert_eq!(
                    obj.difficulty,
                    Difficulty::Intermediate,
                    "{id} should be Intermediate"
                );
            } else if tier3_set.contains(id) {
                assert_eq!(
                    obj.difficulty,
                    Difficulty::Advanced,
                    "{id} should be Advanced"
                );
            } else if tier4_set.contains(id) {
                assert_eq!(obj.difficulty, Difficulty::Expert, "{id} should be Expert");
            }
        }
    }

    #[test]
    fn test_advanced_requires_base_curriculum() {
        let adv = code_generation_advanced_curriculum();
        assert!(
            adv.prerequisite_curricula
                .contains(&CURRICULUM_ID.to_string()),
            "advanced curriculum should require base code-generation curriculum",
        );
    }

    #[test]
    fn test_all_objectives_have_estimated_time() {
        let base = code_generation_curriculum();
        let adv = code_generation_advanced_curriculum();

        for obj in base.objectives.iter().chain(adv.objectives.iter()) {
            assert!(
                obj.estimated_minutes > 0,
                "objective '{}' should have a positive estimated_minutes",
                obj.id,
            );
        }
    }

    #[test]
    fn test_all_objectives_have_tags() {
        let base = code_generation_curriculum();
        let adv = code_generation_advanced_curriculum();

        for obj in base.objectives.iter().chain(adv.objectives.iter()) {
            assert!(
                !obj.tags.is_empty(),
                "objective '{}' should have at least one tag",
                obj.id,
            );
        }
    }

    #[test]
    fn test_curriculum_metadata() {
        let base = code_generation_curriculum();
        assert_eq!(base.id, CURRICULUM_ID);
        assert!(!base.name.is_empty());
        assert!(!base.description.is_empty());
        assert!(!base.tags.is_empty());
    }
}
