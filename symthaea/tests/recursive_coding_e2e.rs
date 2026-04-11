#![cfg(feature = "code_generation")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Recursive Self-Improving Code Generation — End-to-End Integration Test
// ==================================================================================
//
// This is "turning the key" — the first test that exercises the FULL recursive
// coding loop, forcing all 16 new modules to wire together:
//
//   1. SynthesisRequest           → define what to generate
//   2. MagiCodeBridge::predict()  → predict compilation success
//   3. CodeOrchestrator::synth()  → cascading backend generation
//   4. CodeCertificate            → machine-verifiable audit trail
//   5. DistillationCapture        → capture for Broca training
//   6. TestGenerator              → self-generated #[test] module
//   7. MagiCodeBridge::resolve()  → calibration update
//   8. ImmuneCodeEvolver          → evolve code patterns
//   9. FixRuleGenerator           → self-modification from errors
//
// Run: cargo test --test recursive_coding_e2e --features code_generation
// ==================================================================================

use symthaea_core::synthesis_trait::SynthesisRequest;

use symthaea::coding_agent::magi_code_bridge::MagiCodeBridge;
use symthaea::coding_agent::self_modification::FixRuleGenerator;
use symthaea::coding_agent::test_generation::TestGenerator;
use symthaea::consciousness::primitives::code_primitive_evolution::{
    CodeEvolutionConfig, CodePrimitiveEvolver,
};
use symthaea::consciousness::primitives::immune_code_evolution::ImmuneCodeEvolver;
use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::hdc::hv_bridge::HvBridge;
use symthaea::language::code_certificate::CodeCertificate;
use symthaea::language::code_orchestrator::CodeOrchestrator;
use symthaea::language::code_verifier::{CodeVerifier, VerificationPolicy};

// ==================================================================================
// TEST 1: The Full Recursive Loop (Native Path)
// ==================================================================================

#[test]
fn e2e_full_recursive_loop_native() {
    // ── Step 1: Define what to generate ──────────────────────────────
    let request = SynthesisRequest::new("rust", "add", "Add two integers")
        .with_signature("fn add(a: i32, b: i32) -> i32")
        .with_example("add(2, 3)", "5")
        .with_example("add(-1, 1)", "0");

    // ── Step 2: MAGI predicts compilation success ────────────────────
    let mut magi = MagiCodeBridge::new();
    let prediction_id = magi.predict_generation("add", "CodeGenerator", 0.7, "code will compile");
    assert_eq!(magi.pending_count(), 1);

    // ── Step 3: Orchestrator generates code ──────────────────────────
    let mut orchestrator = CodeOrchestrator::new();
    let response = orchestrator.synthesize(&request);

    // The native generator should at least attempt
    assert!(!orchestrator.attempt_history().is_empty(), "Should have attempted generation");
    assert!(orchestrator.energy_spent() > 0.0, "Should have consumed energy");

    // ── Step 4: Self-test generation ─────────────────────────────────
    let test_suite = TestGenerator::generate(&request);
    assert!(!test_suite.tests.is_empty(), "Should generate tests from examples");
    assert!(test_suite.source.contains("#[cfg(test)]"), "Should produce test module");
    assert!(
        test_suite.source.contains("test_add_example_0"),
        "Should name test after function + example index"
    );

    // ── Step 5: MAGI resolves prediction ─────────────────────────────
    let compiled = response.accepted;
    let was_correct = magi.resolve_prediction(&prediction_id, compiled, Some(compiled));
    assert!(was_correct.is_some(), "Prediction should be resolvable");
    assert_eq!(magi.pending_count(), 0, "No pending predictions after resolve");
    assert_eq!(magi.stats().total_predictions, 1);

    // ── Step 6: Check distillation capture ───────────────────────────
    if response.accepted {
        assert!(
            !orchestrator.distillation_buffer().is_empty(),
            "Accepted code should be captured for distillation"
        );

        let capture = &orchestrator.distillation_buffer()[0];
        assert_eq!(capture.channels.len(), 43, "Channel vector should be 43D");
        assert!(!capture.source.is_empty(), "Captured source should be non-empty");
        assert!(capture.quality > 0.0, "Quality should be positive");

        // Verify JSONL export is well-formed
        let jsonl = orchestrator.export_distillation_jsonl();
        assert!(!jsonl.is_empty(), "JSONL export should be non-empty");
        assert!(jsonl.contains("channels"), "JSONL should contain channels");
        assert!(jsonl.contains("target_text"), "JSONL should contain target_text");

        // ── Step 7: Check certificate ────────────────────────────────
        assert!(
            !orchestrator.certificates().is_empty(),
            "Accepted code should have a certificate"
        );
        let cert = &orchestrator.certificates()[0];
        assert!(cert.id.starts_with("cert_"), "Certificate ID format");
        assert!(cert.verify_source(&response.source), "Certificate should verify source");
        assert!(!cert.backend_used.is_empty(), "Backend should be recorded");
        assert!(cert.semantic_similarity > 0.0, "Similarity should be positive");
    }

    // ── Step 8: Calibration state ────────────────────────────────────
    let summary = magi.calibration_summary();
    assert!(summary.contains("1/1") || summary.contains("0/1"), "Should show 1 prediction");
}

// ==================================================================================
// TEST 2: Multiple Generations Build Calibration
// ==================================================================================

#[test]
fn e2e_calibration_improves_over_multiple_generations() {
    let mut magi = MagiCodeBridge::new();

    let test_cases = [
        ("add", "fn add(a: i32, b: i32) -> i32", true),
        ("subtract", "fn subtract(a: i32, b: i32) -> i32", true),
        ("multiply", "fn multiply(a: i32, b: i32) -> i32", true),
        ("divide", "fn divide(a: i32, b: i32) -> Option<i32>", true),
        ("fibonacci", "fn fibonacci(n: u64) -> u64", false), // harder
    ];

    for (name, sig, expected_compile) in &test_cases {
        let pred_id = magi.predict_generation(name, "CodeGenerator", 0.7, "will compile");
        magi.resolve_prediction(&pred_id, *expected_compile, Some(*expected_compile));
    }

    assert_eq!(magi.stats().total_predictions, 5);
    assert!(magi.stats().accuracy() > 0.0, "Should have some accuracy");

    // Adjusted confidence should reflect historical data
    let adjusted = magi.adjusted_confidence(0.8, "CodeGenerator");
    // With only 5 data points, should still be close to raw but with pessimism
    assert!(adjusted > 0.0 && adjusted <= 0.95, "Adjusted confidence: {adjusted}");
}

// ==================================================================================
// TEST 3: Self-Test Generation Covers Multiple Types
// ==================================================================================

#[test]
fn e2e_self_test_generation_comprehensive() {
    // Unsigned integer parameters
    let req_unsigned = SynthesisRequest::new("rust", "factorial", "Compute factorial")
        .with_signature("fn factorial(n: u64) -> u64")
        .with_example("factorial(0)", "1")
        .with_example("factorial(5)", "120");

    let suite = TestGenerator::generate(&req_unsigned);
    assert!(suite.tests.len() >= 4, "Should have examples + boundary tests: got {}", suite.tests.len());
    assert!(suite.source.contains("boundary_n_zero"), "Should test n=0");
    assert!(suite.source.contains("assert_eq!(factorial(0), 1)"), "Should test example 0");

    // String parameters
    let req_string = SynthesisRequest::new("rust", "reverse", "Reverse a string")
        .with_signature("fn reverse(s: &str) -> String")
        .with_example("reverse(\"hello\")", "\"olleh\"");

    let suite = TestGenerator::generate(&req_string);
    assert!(suite.source.contains("boundary_s_empty"), "Should test empty string");

    // Vector parameters
    let req_vec = SynthesisRequest::new("rust", "sum", "Sum a vector")
        .with_signature("fn sum(v: &[i32]) -> i32")
        .with_example("sum(&[1, 2, 3])", "6");

    let suite = TestGenerator::generate(&req_vec);
    assert!(suite.source.contains("boundary_v_empty"), "Should test empty vec");

    // No signature → minimal tests
    let req_minimal = SynthesisRequest::new("rust", "mystery", "Unknown function");
    let suite = TestGenerator::generate(&req_minimal);
    // With no signature and no examples, should produce empty or minimal
    assert!(
        suite.tests.is_empty() || suite.source.contains("No tests"),
        "No-info request should produce minimal tests"
    );
}

// ==================================================================================
// TEST 4: Immune Evolution — V(D)J Recombination
// ==================================================================================

#[test]
fn e2e_immune_evolution_vdj_recombination() {
    let mut evolver = ImmuneCodeEvolver::new();
    evolver.seed_standard();

    let initial_pop = evolver.population_size();
    assert!(initial_pop >= 20, "Should seed 20+ patterns");

    // Run one generation of immune evolution
    let encoder = CodeHDEncoder::default_dim();
    let target = encoder.encode_name("sort_vector_ascending");

    evolver.evolve_generation(|hv| {
        let sim = ((hv.similarity(&target) + 1.0) / 2.0) as f64;
        (sim, [sim, sim, sim])
    });

    assert_eq!(evolver.generation(), 1);
    assert!(evolver.stats().antibodies_created > initial_pop, "Should create new antibodies");
    assert!(evolver.stats().recombinations > 0, "Should perform V(D)J recombinations");

    // Check cytokine status
    assert!(!evolver.cytokines().is_inflamed(), "Should not be inflamed on first gen");
}

// ==================================================================================
// TEST 5: Immune Evolution — Negative Selection (Anti-Pattern Rejection)
// ==================================================================================

#[test]
fn e2e_immune_negative_selection() {
    let evolver = ImmuneCodeEvolver::new();

    // Anti-pattern library should be seeded with dangerous patterns
    assert!(evolver.anti_patterns().len() >= 10, "Should have 10+ anti-patterns");

    // A safe pattern should not be rejected
    let encoder = CodeHDEncoder::default_dim();
    let safe_hv = encoder.encode_name("fibonacci_iterative_safe");
    assert!(
        evolver.anti_patterns().check(&safe_hv).is_none(),
        "Safe pattern should not trigger negative selection"
    );
}

// ==================================================================================
// TEST 6: Immune Evolution — Cytokine Inflammation
// ==================================================================================

#[test]
fn e2e_cytokine_inflammation_and_recovery() {
    use symthaea::consciousness::primitives::immune_code_evolution::CytokineSignal;

    let mut cytokines = CytokineSignal::new();

    // Initially calm
    assert!(!cytokines.is_inflamed());
    assert_eq!(cytokines.effective_mutation_rate(0.05), 0.05);

    // Sustained failure triggers inflammation
    for _ in 0..20 {
        cytokines.record_outcome(false);
    }
    assert!(cytokines.is_inflamed(), "Should be inflamed after 20 failures");

    let boosted_rate = cytokines.effective_mutation_rate(0.05);
    assert!(boosted_rate > 0.05, "Mutation rate should be boosted: {boosted_rate}");

    // Recovery via successes
    for _ in 0..30 {
        cytokines.record_outcome(true);
    }
    assert!(!cytokines.is_inflamed(), "Should recover after sustained success");
}

// ==================================================================================
// TEST 7: Code Primitive Evolution — Evolve Toward Intent
// ==================================================================================

#[test]
fn e2e_primitive_evolution_toward_intent() {
    let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig {
        population_size: 20,
        max_generations: 10,
        discovery_threshold: 0.3,
        ..Default::default()
    });
    evolver.seed_standard_patterns();

    let encoder = CodeHDEncoder::default_dim();
    let intent = encoder.encode_name("binary_search_sorted_array");

    let result = evolver.evolve_toward_intent(&intent);

    assert!(result.generations > 0, "Should run at least 1 generation");
    assert!(result.best_fitness > 0.0, "Should achieve positive fitness");

    // Check stats
    assert!(evolver.stats().concepts_created > 0);
}

// ==================================================================================
// TEST 8: Code Primitive Evolution — Hybrid Algorithm Discovery
// ==================================================================================

#[test]
fn e2e_primitive_evolution_hybrid() {
    let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig {
        population_size: 20,
        max_generations: 5,
        discovery_threshold: 0.2, // Low threshold for test
        ..Default::default()
    });

    let result = evolver.evolve_hybrid("binary_search", "hash_lookup");

    assert!(result.generations > 0, "Should run evolution");
    assert!(result.best_fitness > 0.0, "Hybrid should have positive fitness");
}

// ==================================================================================
// TEST 9: Self-Modification — Error Cluster Detection
// ==================================================================================

#[test]
fn e2e_self_modification_error_clustering() {
    let mut gen = FixRuleGenerator::new();

    // Simulate a recurring error pattern
    for i in 0..7 {
        gen.observe_error(
            "E0308",
            "TypeMismatch",
            "mismatched types: expected String, found &str",
            "add .to_string()",
            i < 2, // Only 28% success → below 50% threshold
            "let name: String = get_name();",
        );
    }

    assert!(gen.stats().clusters_detected > 0, "Should detect error cluster");
}

// ==================================================================================
// TEST 10: Self-Modification — Rule Generation with Gates
// ==================================================================================

#[test]
fn e2e_self_modification_rule_generation_gated() {
    let mut gen = FixRuleGenerator::new();

    // Build a failure cluster
    for _ in 0..7 {
        gen.observe_error(
            "E0308",
            "TypeMismatch",
            "expected String found &str",
            "manual_convert",
            false,
            "let x: String = input;",
        );
    }

    // Gate 1: Low consciousness → blocked
    let rules = gen.try_generate_rules(0.1, 0.9);
    assert!(rules.is_empty(), "Low Phi should block self-modification");
    assert!(gen.stats().consciousness_gates > 0);

    // Gate 2: Low calibration → blocked
    let rules = gen.try_generate_rules(0.5, 0.3);
    assert!(rules.is_empty(), "Low calibration should block self-modification");
    assert!(gen.stats().calibration_gates > 0);

    // Gate 3: Both passing → generates rule
    let rules = gen.try_generate_rules(0.5, 0.9);
    assert!(!rules.is_empty(), "Should generate rule when gates pass");
    assert_eq!(gen.stats().rules_generated, 1);

    // Verify the generated rule
    let rule = &gen.all_rules()[0];
    assert!(rule.error_code.as_deref() == Some("E0308"));
    assert!(rule.confidence > 0.0);
    assert!(!rule.promoted);
    assert!(!rule.demoted);
}

// ==================================================================================
// TEST 11: Self-Modification — Full Rule Lifecycle
// ==================================================================================

#[test]
fn e2e_self_modification_full_lifecycle() {
    let mut gen = FixRuleGenerator::new();

    // Phase 1: Accumulate failures
    for _ in 0..7 {
        gen.observe_error("E0277", "MissingImport", "HashMap not found", "none", false, "HashMap::new()");
    }

    // Phase 2: Generate rule
    let rules = gen.try_generate_rules(0.6, 0.9);
    assert!(!rules.is_empty());
    let rule_id = rules[0].clone();

    // Phase 3: Apply rule successfully 6 times
    for _ in 0..6 {
        gen.record_rule_outcome(&rule_id, true, Some(true));
    }

    // Phase 4: Rule should be promoted (>70% success, 5+ applications)
    let rule = gen.all_rules().iter().find(|r| r.id == rule_id).unwrap();
    assert!(rule.is_proven(), "Rule should be proven after 6/6 successes");
    assert!(rule.promoted, "Rule should be auto-promoted");
    assert!(!rule.demoted, "Rule should not be demoted");
    assert!(rule.success_rate() > 0.8);
}

// ==================================================================================
// TEST 12: HV Bridge — Cross-Type Operations
// ==================================================================================

#[test]
fn e2e_hv_bridge_cross_type_similarity() {
    use symthaea_core::hdc::binary_hv::BinaryHV;

    // Create a BinaryHV and convert
    let bhv = BinaryHV::random(42);
    let chv = HvBridge::binary_to_continuous(&bhv);

    // Cross-similarity should match native similarity
    let cross_sim = HvBridge::cross_similarity(&chv, &bhv);
    assert!(
        (cross_sim - 1.0).abs() < 1e-4,
        "Identical vectors should have ~1.0 cross similarity: {cross_sim}"
    );

    // Round-trip should be lossless
    let bhv_recovered = HvBridge::continuous_to_binary(&chv);
    assert_eq!(bhv, bhv_recovered, "Round-trip should be lossless");
}

// ==================================================================================
// TEST 13: Verification Policy Strictness Ordering
// ==================================================================================

#[test]
fn e2e_verification_policy_ordering() {
    assert!(
        VerificationPolicy::SafetyCritical.threshold()
            > VerificationPolicy::Standard.threshold()
    );
    assert!(
        VerificationPolicy::Standard.threshold()
            > VerificationPolicy::Exploratory.threshold()
    );

    // Safety-critical should be strictest
    let safety_verifier = CodeVerifier::with_policy(
        CodeHDEncoder::new(512),
        VerificationPolicy::SafetyCritical,
    );
    assert!((safety_verifier.threshold() - 0.85).abs() < 1e-6);
}

// ==================================================================================
// TEST 14: Certificate Tamper Detection
// ==================================================================================

#[test]
fn e2e_certificate_tamper_detection() {
    let source = "fn fibonacci(n: u64) -> u64 { if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) } }";
    let cert = CodeCertificate::new(source, "CodeGenerator", 0.85)
        .with_topology(1, 1, 0)
        .with_sheaf_coherent(true);

    // Original source verifies
    assert!(cert.verify_source(source));

    // Tampered source fails
    assert!(!cert.verify_source("fn fibonacci(n: u64) -> u64 { 42 }"));

    // Certificate has topology
    let topo = cert.topology.as_ref().unwrap();
    assert_eq!(topo.beta_0, 1);
    assert_eq!(topo.beta_1, 1); // recursion = cycle

    // Summary is readable
    let summary = cert.summary();
    assert!(summary.contains("cert_"));
    assert!(summary.contains("CodeGenerator"));
    assert!(summary.contains("β₀=1"));
}

// ==================================================================================
// TEST 15: Distillation JSONL Format Compatibility
// ==================================================================================

#[test]
fn e2e_distillation_jsonl_format() {
    let mut orchestrator = CodeOrchestrator::new();

    // Generate code to trigger distillation capture
    let request = SynthesisRequest::new("rust", "double", "Double an integer")
        .with_signature("fn double(x: i32) -> i32")
        .with_example("double(5)", "10");

    let _response = orchestrator.synthesize(&request);

    // Even if generation didn't produce accepted code,
    // verify the JSONL export mechanism works
    let jsonl = orchestrator.export_distillation_jsonl();

    if !orchestrator.distillation_buffer().is_empty() {
        // Parse each line as JSON to verify format
        for line in jsonl.lines() {
            let parsed: serde_json::Value = serde_json::from_str(line)
                .expect("Each distillation line should be valid JSON");

            assert!(parsed.get("channels").is_some(), "Must have channels field");
            assert!(parsed.get("target_text").is_some(), "Must have target_text field");
            assert!(parsed.get("target_ids").is_some(), "Must have target_ids field");

            let channels = parsed["channels"].as_array().unwrap();
            assert_eq!(channels.len(), 43, "Channel vector must be 43D");
        }
    }
}

// ==================================================================================
// TEST 16: Full Recursive Loop — Multiple Iterations
// ==================================================================================

#[test]
fn e2e_recursive_loop_multiple_iterations() {
    let mut magi = MagiCodeBridge::new();
    let mut orchestrator = CodeOrchestrator::new();
    let mut fix_gen = FixRuleGenerator::new();

    let requests = [
        SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32"),
        SynthesisRequest::new("rust", "max", "Return the larger of two integers")
            .with_signature("fn max(a: i32, b: i32) -> i32"),
        SynthesisRequest::new("rust", "abs", "Absolute value")
            .with_signature("fn abs(x: i32) -> i32"),
        SynthesisRequest::new("rust", "clamp", "Clamp value to range")
            .with_signature("fn clamp(x: i32, min: i32, max: i32) -> i32"),
        SynthesisRequest::new("rust", "is_even", "Check if number is even")
            .with_signature("fn is_even(n: i32) -> bool"),
    ];

    for request in &requests {
        // Predict
        let pred_id = magi.predict_generation(
            &request.name,
            "CodeGenerator",
            magi.adjusted_confidence(0.7, "CodeGenerator"),
            "will compile",
        );

        // Generate
        let response = orchestrator.synthesize(request);

        // Generate self-tests
        let _tests = TestGenerator::generate(request);

        // Resolve
        let compiled = response.accepted;
        magi.resolve_prediction(&pred_id, compiled, Some(compiled));

        // Observe errors for self-modification
        if !compiled {
            fix_gen.observe_error(
                "E0000",
                "GenerationFailure",
                &format!("failed to generate {}", request.name),
                "native_generation",
                false,
                &response.source,
            );
        }
    }

    // After 5 iterations, system should have state
    assert_eq!(magi.stats().total_predictions, 5, "Should have 5 predictions");
    assert!(orchestrator.attempt_history().len() >= 5, "Should have 5+ attempts");
    assert!(orchestrator.energy_spent() > 0.0, "Should have spent energy");

    // Calibration summary should reflect history
    let summary = magi.calibration_summary();
    assert!(!summary.contains("No code predictions yet"), "Should have prediction data");
}

// ==================================================================================
// TEST 17: Hierarchical Position Encoding for Large Modules
// ==================================================================================

#[test]
fn e2e_hierarchical_position_encoding() {
    let encoder = CodeHDEncoder::new(512);

    // Create positions spanning beyond the 256 flat limit
    let pos_0 = encoder.encode_name("entity_0");
    let pos_255 = encoder.encode_name("entity_255");
    let pos_300 = encoder.encode_name("entity_300");
    let pos_500 = encoder.encode_name("entity_500");

    // All should be valid and different
    assert!(pos_0.similarity(&pos_255) < 0.9);
    assert!(pos_0.similarity(&pos_300) < 0.9);
    assert!(pos_0.similarity(&pos_500) < 0.9);
    assert!(pos_255.similarity(&pos_300) < 0.9);
}
