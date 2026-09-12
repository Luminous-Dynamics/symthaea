#![cfg(feature = "symthaea-backend")]

//! Component-level AST-1 evidence for the production `AttentionSchema`.
//!
//! These tests establish two narrow properties without promoting a Butlin tier:
//! 1. the schema makes prospective predictions about its own future attention state;
//! 2. an intervention on the schema's modeled controllability causally changes
//!    its downstream competition bias under identical attention input, while a
//!    matched perturbation to a non-control self-model field does not.
//!
//! Full `CognitiveLoopService` integration, behavioral consequence, sham at the
//! system level, replication, and consciousness claims remain out of scope.

use symthaea::consciousness::attention_schema::{
    AttentionMode, AttentionSchema,
};
use symthaea::hdc::binary_hv::BinaryHV;

fn prepared_schema() -> (AttentionSchema, BinaryHV) {
    let target = BinaryHV::random(0xA571_0001);
    let mut schema = AttentionSchema::new();
    schema.update(target, 0.70);

    // Freeze a common, interpretable self-model state before cloning arms.
    schema.self_model.subjective_character.presence = 0.80;
    schema.self_model.subjective_character.controllability = 0.80;
    schema.self_model.subjective_character.effort = 0.30;
    schema.self_model.subjective_character.clarity = 0.80;

    (schema, target)
}

#[test]
fn modeled_controllability_causally_changes_competition_bias_under_identical_input() {
    let (schema, target) = prepared_schema();
    let mut baseline = schema.clone();
    let mut targeted = schema.clone();
    let mut sham = schema;

    // Targeted AST intervention: degrade only the schema's estimate of its
    // ability to control attention. Keep the attended content and incoming
    // salience identical.
    targeted.self_model.subjective_character.controllability = 0.20;

    // Matched sham: perturb a self-model field by the same absolute amount,
    // but one that is not used by compute_control_signal().
    sham.self_model.subjective_character.effort = 0.90;

    let baseline_update = baseline.update(target, 0.70);
    let targeted_update = targeted.update(target, 0.70);
    let sham_update = sham.update(target, 0.70);

    let baseline_bias = baseline.get_competition_bias(&target);
    let targeted_bias = targeted.get_competition_bias(&target);
    let sham_bias = sham.get_competition_bias(&target);

    assert!(
        targeted_update.control_signal + 0.10 < baseline_update.control_signal,
        "targeted control signal did not fall enough: baseline={} targeted={}",
        baseline_update.control_signal,
        targeted_update.control_signal,
    );
    assert!(
        targeted_bias + 0.05 < baseline_bias,
        "targeted competition bias did not fall enough: baseline={} targeted={}",
        baseline_bias,
        targeted_bias,
    );
    assert!(
        (sham_update.control_signal - baseline_update.control_signal).abs() < 1e-6,
        "matched sham unexpectedly changed control signal: baseline={} sham={}",
        baseline_update.control_signal,
        sham_update.control_signal,
    );
    assert!(
        (sham_bias - baseline_bias).abs() < 1e-6,
        "matched sham unexpectedly changed competition bias: baseline={} sham={}",
        baseline_bias,
        sham_bias,
    );
}

#[test]
fn restoring_modeled_controllability_rescues_control_output() {
    let (schema, target) = prepared_schema();
    let mut reference = schema.clone();
    let mut perturbed = schema;

    perturbed.self_model.subjective_character.controllability = 0.20;

    let reference_first = reference.update(target, 0.70);
    let perturbed_first = perturbed.update(target, 0.70);
    assert!(perturbed_first.control_signal < reference_first.control_signal);

    // Rescue only the targeted self-model coordinate. Both arms now receive the
    // same next input and salience again.
    perturbed.self_model.subjective_character.controllability =
        reference.self_model.subjective_character.controllability;

    let reference_second = reference.update(target, 0.70);
    let rescued_second = perturbed.update(target, 0.70);

    let reference_bias = reference.get_competition_bias(&target);
    let rescued_bias = perturbed.get_competition_bias(&target);

    assert!(
        (rescued_second.control_signal - reference_second.control_signal).abs() < 1e-6,
        "rescue did not restore control signal: reference={} rescued={}",
        reference_second.control_signal,
        rescued_second.control_signal,
    );
    assert!(
        (rescued_bias - reference_bias).abs() < 1e-6,
        "rescue did not restore competition bias: reference={} rescued={}",
        reference_bias,
        rescued_bias,
    );
}

#[test]
fn schema_prospectively_predicts_shift_after_attention_decay() {
    let target = BinaryHV::random(0xA571_1001);
    let novel = BinaryHV::random(0xA571_1002);
    let mut schema = AttentionSchema::new();

    schema.update(target, 0.70);

    // Maintain the same target until the schema's own forward model predicts
    // a future scanning/shift state due to decaying intensity.
    let mut predicted_shift = false;
    for _ in 0..16 {
        let update = schema.update(target, 0.70);
        predicted_shift = update
            .predictions
            .iter()
            .any(|state| state.mode == AttentionMode::Scanning);
        if predicted_shift {
            break;
        }
    }

    assert!(predicted_shift, "schema never generated a prospective shift prediction");

    let correct_before = schema.stats().predictions_correct;
    let validated_before = schema.stats().predictions_validated;

    // Outcome is revealed only now.
    let outcome = schema.update(novel, 0.70);
    assert!(outcome.is_shift, "held-out novel target did not produce a shift");
    assert_eq!(schema.stats().predictions_validated, validated_before + 1);
    assert_eq!(
        schema.stats().predictions_correct,
        correct_before + 1,
        "the prospective scanning prediction was not validated as correct",
    );
}

#[test]
fn component_scope_does_not_imply_ast1_functional_support() {
    // This is intentionally a contract test in prose form: the component
    // theorem proves schema prediction + causal control output, but no full-loop
    // behavioral consequence or independent replication is present here.
    const CLAIM_SCOPE: &str = "component-only; no AST-1 tier promotion; no consciousness claim";
    assert!(CLAIM_SCOPE.contains("no AST-1 tier promotion"));
    assert!(CLAIM_SCOPE.contains("no consciousness claim"));
}
