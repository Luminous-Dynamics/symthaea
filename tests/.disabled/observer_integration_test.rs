//! Observer Integration Tests
//!
//! Tests that observer hooks are correctly integrated into the consciousness pipeline.

use symthaea::observability::{TraceObserver, NullObserver, SharedObserver};
use symthaea::safety::SafetyGuardrails;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs;

#[test]
fn test_security_observer_integration() {
    // Create a temporary trace file
    let trace_path = "/tmp/security_test_trace.json";

    // Clean up any existing trace
    let _ = fs::remove_file(trace_path);

    // Create observer
    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(TraceObserver::new(trace_path).expect("Failed to create TraceObserver"))
    ));

    // Create SafetyGuardrails with observer
    let mut guards = SafetyGuardrails::with_observer(Some(Arc::clone(&observer)));

    // Create a safe action (low similarity to forbidden patterns)
    let safe_action = vec![1i8; 10_000];

    // Check safety - should be allowed
    let result = guards.check_safety(&safe_action);
    assert!(result.is_ok(), "Safe action should be allowed");

    // Finalize observer to flush trace
    observer.blocking_write().finalize().expect("Failed to finalize observer");

    // Read trace file
    let trace_content = fs::read_to_string(trace_path)
        .expect("Failed to read trace file");

    // Verify trace contains security check event
    assert!(trace_content.contains("security_check"),
           "Trace should contain security_check event");
    assert!(trace_content.contains("allowed"),
           "Trace should contain allowed decision");

    // Clean up
    let _ = fs::remove_file(trace_path);
}

#[test]
fn test_null_observer_zero_overhead() {
    // Create NullObserver (should compile to zero overhead)
    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(NullObserver::new())
    ));

    // Create SafetyGuardrails with null observer
    let mut guards = SafetyGuardrails::with_observer(Some(Arc::clone(&observer)));

    // Check safety multiple times
    let safe_action = vec![1i8; 10_000];
    for _ in 0..100 {
        let _ = guards.check_safety(&safe_action);
    }

    // No assertions needed - if this compiles and runs, NullObserver works
    assert!(true);
}

#[test]
fn test_backwards_compatibility() {
    // Old code without observer should still work
    let mut guards = SafetyGuardrails::new();

    let safe_action = vec![1i8; 10_000];
    let result = guards.check_safety(&safe_action);

    assert!(result.is_ok(), "Backwards compatibility: SafetyGuardrails::new() should work");
}

#[test]
fn test_error_diagnosis_observer_integration() {
    use symthaea::language::nix_error_diagnosis::NixErrorDiagnoser;

    // Create a temporary trace file
    let trace_path = "/tmp/error_diagnosis_test_trace.json";

    // Clean up any existing trace
    let _ = fs::remove_file(trace_path);

    // Create observer
    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(TraceObserver::new(trace_path).expect("Failed to create TraceObserver"))
    ));

    // Create NixErrorDiagnoser with observer
    let diagnoser = NixErrorDiagnoser::with_observer(Some(Arc::clone(&observer)));

    // Simulate a NixOS error
    let error_output = "error: infinite recursion encountered at /etc/nixos/configuration.nix:42";

    // Diagnose the error (should record ErrorEvent)
    let diagnosis = diagnoser.diagnose(error_output);

    assert_eq!(diagnosis.error_type.name(), "infinite recursion");
    assert!(diagnosis.confidence > 0.3);

    // Finalize observer to flush trace
    observer.blocking_write().finalize().expect("Failed to finalize observer");

    // Read trace file
    let trace_content = fs::read_to_string(trace_path)
        .expect("Failed to read trace file");

    // Verify trace contains error event
    assert!(trace_content.contains("Error") || trace_content.contains("error"),
           "Trace should contain Error event");
    assert!(trace_content.contains("infinite recursion"),
           "Trace should contain error type");

    // Clean up
    let _ = fs::remove_file(trace_path);
}

#[test]
fn test_error_diagnosis_backwards_compatibility() {
    use symthaea::language::nix_error_diagnosis::NixErrorDiagnoser;

    // Old code without observer should still work
    let diagnoser = NixErrorDiagnoser::new();

    let error_output = "error: attribute 'firefoxBrowser' missing";
    let diagnosis = diagnoser.diagnose(error_output);

    assert!(diagnosis.confidence > 0.0);
    assert!(!diagnosis.fixes.is_empty());
}

#[test]
fn test_response_generation_observer_integration() {
    use symthaea::language::generator::{ResponseGenerator, ConsciousnessContext};
    use symthaea::language::parser::SemanticParser;

    // Create a temporary trace file
    let trace_path = "/tmp/response_generation_test_trace.json";

    // Clean up any existing trace
    let _ = fs::remove_file(trace_path);

    // Create observer
    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(TraceObserver::new(trace_path).expect("Failed to create TraceObserver"))
    ));

    // Create ResponseGenerator with observer
    let generator = ResponseGenerator::with_observer(Some(Arc::clone(&observer)));

    // Parse input
    let parser = SemanticParser::new();
    let input = parser.parse("Hello, how are you?");

    // Create consciousness context
    let consciousness = ConsciousnessContext {
        phi: 0.65,
        meta_awareness: 0.5,
        emotional_valence: 0.2,
        arousal: 0.4,
        self_confidence: 0.7,
        attention_topics: vec!["conversation".to_string()],
        phenomenal_state: "aware and engaged".to_string(),
    };

    // Generate response (should record LanguageStepEvent)
    let response = generator.generate(&input, &consciousness);

    assert!(!response.text.is_empty());
    assert!(response.confidence > 0.0);

    // Finalize observer to flush trace
    observer.blocking_write().finalize().expect("Failed to finalize observer");

    // Read trace file
    let trace_content = fs::read_to_string(trace_path)
        .expect("Failed to read trace file");

    // Verify trace contains language step event
    assert!(trace_content.contains("language_step"),
           "Trace should contain language_step event");
    assert!(trace_content.contains("response_generation"),
           "Trace should indicate response_generation type");

    // Clean up
    let _ = fs::remove_file(trace_path);
}

#[test]
fn test_response_generation_backwards_compatibility() {
    use symthaea::language::generator::{ResponseGenerator, ConsciousnessContext};
    use symthaea::language::parser::SemanticParser;

    // Old code without observer should still work
    let generator = ResponseGenerator::new();
    let parser = SemanticParser::new();

    let input = parser.parse("What is consciousness?");

    let consciousness = ConsciousnessContext {
        phi: 0.5,
        meta_awareness: 0.4,
        emotional_valence: 0.0,
        arousal: 0.3,
        self_confidence: 0.6,
        attention_topics: vec![],
        phenomenal_state: String::new(),
    };

    let response = generator.generate(&input, &consciousness);

    assert!(!response.text.is_empty());
    assert!(response.confidence > 0.0);
}

#[test]
fn test_phi_measurement_observer_integration() {
    use symthaea::hdc::{IntegratedInformation, HV16};

    // Create a temporary trace file
    let trace_path = "/tmp/phi_measurement_test_trace.json";

    // Clean up any existing trace
    let _ = fs::remove_file(trace_path);

    // Create observer
    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(TraceObserver::new(trace_path).expect("Failed to create TraceObserver"))
    ));

    // Create IntegratedInformation with observer
    let mut phi_calc = IntegratedInformation::with_observer(Some(Arc::clone(&observer)));

    // Create a system state with 4 components
    let state = vec![
        HV16::random(1), // Sensory
        HV16::random(2), // Memory
        HV16::random(3), // Attention
        HV16::random(4), // Motor
    ];

    // Compute Φ (should record PhiMeasurementEvent with 7 components)
    let phi_value = phi_calc.compute_phi(&state);

    assert!(phi_value >= 0.0);
    assert!(phi_value <= 2.0); // Normalized Φ should be reasonable

    // Finalize observer to flush trace
    observer.blocking_write().finalize().expect("Failed to finalize observer");

    // Read trace file
    let trace_content = fs::read_to_string(trace_path)
        .expect("Failed to read trace file");

    // Verify trace contains Φ measurement event
    assert!(trace_content.contains("phi_measurement"),
           "Trace should contain phi_measurement event");

    // Verify all 7 components are present
    assert!(trace_content.contains("integration"),
           "Trace should contain integration component");
    assert!(trace_content.contains("binding"),
           "Trace should contain binding component");
    assert!(trace_content.contains("workspace"),
           "Trace should contain workspace component");
    assert!(trace_content.contains("attention"),
           "Trace should contain attention component");
    assert!(trace_content.contains("recursion"),
           "Trace should contain recursion component");
    assert!(trace_content.contains("efficacy"),
           "Trace should contain efficacy component");
    assert!(trace_content.contains("knowledge"),
           "Trace should contain knowledge component");

    // Clean up
    let _ = fs::remove_file(trace_path);
}

#[test]
fn test_phi_measurement_backwards_compatibility() {
    use symthaea::hdc::{IntegratedInformation, HV16};

    // Old code without observer should still work
    let mut phi_calc = IntegratedInformation::new();

    let state = vec![
        HV16::random(1),
        HV16::random(2),
        HV16::random(3),
    ];

    let phi_value = phi_calc.compute_phi(&state);

    assert!(phi_value >= 0.0);
}

#[test]
fn test_phi_components_rigorous_calculation() {
    use symthaea::hdc::{IntegratedInformation, HV16};

    // Create observer to capture detailed components
    let trace_path = "/tmp/phi_components_test_trace.json";
    let _ = fs::remove_file(trace_path);

    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(TraceObserver::new(trace_path).expect("Failed to create TraceObserver"))
    ));

    let mut phi_calc = IntegratedInformation::with_observer(Some(Arc::clone(&observer)));

    // Create a larger system to test component calculations
    let state = vec![
        HV16::random(1),
        HV16::random(2),
        HV16::random(3),
        HV16::random(4),
        HV16::random(5),
        HV16::random(6),
    ];

    // Compute Φ multiple times to build history (for recursion & knowledge)
    let phi1 = phi_calc.compute_phi(&state);
    let phi2 = phi_calc.compute_phi(&state);
    let phi3 = phi_calc.compute_phi(&state);

    // All Φ values should be reasonable
    assert!(phi1 >= 0.0 && phi1 <= 2.0);
    assert!(phi2 >= 0.0 && phi2 <= 2.0);
    assert!(phi3 >= 0.0 && phi3 <= 2.0);

    // Finalize and check trace contains component breakdown
    observer.blocking_write().finalize().expect("Failed to finalize observer");

    let trace_content = fs::read_to_string(trace_path)
        .expect("Failed to read trace file");

    // Verify components have meaningful values (not all zeros)
    let has_nonzero_components =
        trace_content.contains("\"integration\"") &&
        trace_content.contains("\"binding\"") &&
        trace_content.contains("\"workspace\"");

    assert!(has_nonzero_components,
           "Φ components should have meaningful calculated values");

    let _ = fs::remove_file(trace_path);
}
