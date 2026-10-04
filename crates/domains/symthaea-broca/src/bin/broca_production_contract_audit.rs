// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic audit of Broca's typed speech-production contracts.
//!
//! This is an invariant/evidence probe, not a naturalness benchmark. It exercises the
//! full supported intent × epistemic matrix without stochastic generation.

use std::{fs, path::PathBuf, process};

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea_broca::{
    ContentBindingStatus, LinguisticBindingStatus, LinguisticFrame, PhonemeSlot,
    PhonologicalPlan, SpeechDeliveryObservation, SpeechDeliveryReceipt, SpeechPlan,
    StructuredDecoder, SyllableStress, ThoughtChannels,
};
use symthaea_core::genesis::GenesisSeed;

const INTENTS: [&str; 8] = [
    "analyze", "create", "explain", "question", "answer", "reflect", "relate", "unknown",
];
const EPISTEMIC_LEVELS: [f32; 4] = [0.0, 1.0, 2.0, 3.0];

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    evidence_level: &'static str,
    matrix_cases: usize,
    passed_cases: usize,
    speech_plan_validation_checks: usize,
    acoustic_receipt_checks: usize,
    linguistic_frame_checks: usize,
    phonological_binding_checks: usize,
    semantic_delivery_checks: usize,
    lexical_provenance_checks: usize,
    deterministic_grounding_checks: usize,
    failures: Vec<AuditFailure>,
    cases: Vec<AuditCase>,
}

#[derive(Debug, Serialize)]
struct AuditFailure {
    case_id: String,
    invariant: String,
    detail: String,
}

#[derive(Debug, Serialize)]
struct AuditCase {
    case_id: String,
    speech_plan_valid: bool,
    acoustic_receipt_valid: bool,
    acoustic_complete_passed: bool,
    acoustic_receipt_lineage_valid: bool,
    intent: String,
    epistemic_input: f32,
    plan_epistemic: String,
    clause_mode: String,
    formulation_strategy: String,
    linguistic_ready: bool,
    linguistic_lineage_valid: bool,
    phonological_lineage_valid: bool,
    linguistic_lexical_binding_valid: bool,
    content_binding: String,
    role_only_rejected_segments: bool,
    phonological_binding_succeeded: bool,
    phonological_persistence_validated: bool,
    lexical_binding_succeeded: bool,
    lexical_missing_provenance_rejected: bool,
    semantic_exact_passed: bool,
    semantic_receipt_lineage_valid: bool,
    semantic_mismatch_rejected: bool,
    semantic_partial_gate_rejected: bool,
    linguistic_to_phonological_preserved_intent: bool,
    grounding_deterministic: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("broca-production-contract-audit: {error:#}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let json_out = parse_json_out();
    let genesis = GenesisSeed::from_phrase("broca-production-contract-audit-v2");
    let decoder = StructuredDecoder::new(&genesis);

    let mut report = AuditReport {
        schema_version: 2,
        evidence_level: "deterministic-contract-audit",
        matrix_cases: 0,
        passed_cases: 0,
        speech_plan_validation_checks: 0,
        acoustic_receipt_checks: 0,
        linguistic_frame_checks: 0,
        phonological_binding_checks: 0,
        semantic_delivery_checks: 0,
        lexical_provenance_checks: 0,
        deterministic_grounding_checks: 0,
        failures: Vec::new(),
        cases: Vec::new(),
    };

    for (intent_index, intent_name) in INTENTS.iter().enumerate() {
        for epistemic in EPISTEMIC_LEVELS {
            report.matrix_cases += 1;
            let case_id = case_id(intent_index, epistemic);

            let mut channels = ThoughtChannels::with_intent(intent_index);
            channels.set_epistemic(epistemic);
            let readout = decoder.decode(&channels);

            if readout.intent != *intent_name {
                fail(
                    &mut report,
                    &case_id,
                    "intent_decode",
                    format!("expected {intent_name}, got {}", readout.intent),
                );
            }

            let speech_plan = SpeechPlan::from_readout(&channels, &readout);
            let speech_plan_valid = speech_plan.validate().is_ok();
            report.speech_plan_validation_checks += 1;
            if !speech_plan_valid {
                fail(
                    &mut report,
                    &case_id,
                    "speech_plan_validation",
                    "generated speech plan failed persisted-state validation".to_string(),
                );
            }
            let linguistic = LinguisticFrame::from_speech_plan(&speech_plan);
            let linguistic_ready = linguistic.ready_for_phonology();
            let linguistic_valid = linguistic.validate().is_ok();
            let linguistic_lineage_valid =
                linguistic.validate_against_plan(&speech_plan).is_ok();

            report.linguistic_frame_checks += 3;
            if !linguistic_lineage_valid {
                fail(
                    &mut report,
                    &case_id,
                    "linguistic_frame_lineage",
                    "linguistic frame did not validate against its source speech plan".to_string(),
                );
            }
            if linguistic_ready != linguistic_valid {
                fail(
                    &mut report,
                    &case_id,
                    "linguistic_ready_matches_validation",
                    format!("ready={linguistic_ready}, valid={linguistic_valid}"),
                );
            }

            let linguistic_lexical_binding_valid = if linguistic_ready {
                let mut bound = linguistic.clone();
                bound
                    .bind_lexical_provenance(format!("lexicalizer::{intent_name}"))
                    .is_ok()
                    && bound.binding_status == LinguisticBindingStatus::LexicallyBound
                    && bound.lexical_provenance.is_some()
                    && bound.validate().is_ok()
            } else {
                true
            };

            if linguistic_ready {
                report.lexical_provenance_checks += 1;
                if !linguistic_lexical_binding_valid {
                    fail(
                        &mut report,
                        &case_id,
                        "linguistic_lexical_provenance",
                        "valid linguistic frame did not accept and preserve provenance".to_string(),
                    );
                }
            }

            let mut phonological = PhonologicalPlan::from_linguistic_frame(&linguistic);
            let linguistic_to_phonological_preserved_intent =
                phonological.source_intent == linguistic.source_intent;
            let phonological_lineage_before_binding =
                phonological.validate_against_frame(&linguistic).is_ok();
            if !phonological_lineage_before_binding {
                fail(
                    &mut report,
                    &case_id,
                    "phonological_plan_lineage",
                    "phonological plan did not validate against its source linguistic frame".to_string(),
                );
            }

            let role_only_error = phonological
                .bind_segments(
                    sample_segments(false),
                    ContentBindingStatus::RoleStructureOnly,
                )
                .expect_err("role-only binding must reject segments");
            let role_only_rejected = matches!(
                role_only_error,
                symthaea_broca::PhonologicalPlanError::RoleOnlyWithSegments
            );
            report.phonological_binding_checks += 2;
            if !role_only_rejected {
                fail(
                    &mut report,
                    &case_id,
                    "role_only_segment_rejection",
                    format!("unexpected error: {role_only_error}"),
                );
            }

            phonological
                .bind_segments(
                    sample_segments(speech_plan.focus_role.is_some()),
                    ContentBindingStatus::PhonologicallyBound,
                )
                .with_context(|| format!("phonological binding failed for {case_id}"))?;
            let phonological_binding_succeeded =
                phonological.content_binding == ContentBindingStatus::PhonologicallyBound
                    && phonological.ready_for_realization();
            let phonological_persistence_validated = phonological.validate().is_ok();

            let lexical_missing_provenance = PhonologicalPlan::from_speech_plan(&speech_plan)
                .bind_segments(
                    sample_segments(true),
                    ContentBindingStatus::LexicallyBound,
                )
                .expect_err("lexical binding must require provenance");
            let lexical_missing_provenance_rejected = matches!(
                lexical_missing_provenance,
                symthaea_broca::PhonologicalPlanError::LexicalBindingWithoutProvenance
            );
            report.lexical_provenance_checks += 1;

            let mut lexical = PhonologicalPlan::from_speech_plan(&speech_plan);
            lexical
                .bind_lexical_segments(
                    sample_segments(speech_plan.focus_role.is_some()),
                    format!("lexeme::{intent_name}"),
                )
                .with_context(|| format!("lexical binding failed for {case_id}"))?;
            let lexical_binding_succeeded =
                lexical.content_binding == ContentBindingStatus::LexicallyBound
                    && lexical.lexical_provenance.is_some()
                    && lexical.validate().is_ok();

            if !lexical_missing_provenance_rejected {
                fail(
                    &mut report,
                    &case_id,
                    "phonological_lexical_provenance",
                    format!("unexpected error: {lexical_missing_provenance}"),
                );
            }

            let sensory_target = symthaea_broca::SpeechSensoryTarget::from_plan(&speech_plan);
            let acoustic_receipt = symthaea_broca::SpeechFeedbackReceipt::new(
                &speech_plan,
                symthaea_broca::SpeechSensoryObservation {
                    pitch_range: Some(sensory_target.pitch_range),
                    prominence: Some(sensory_target.prominence),
                    rate: Some(sensory_target.rate),
                    pause_weight: Some(sensory_target.pause_weight),
                },
            );
            let acoustic_receipt_valid = acoustic_receipt.validate().is_ok();
            let acoustic_complete_passed = acoustic_receipt.error.passes_complete(0.0);
            let acoustic_receipt_lineage_valid =
                acoustic_receipt.validate_against_plan(&speech_plan).is_ok();
            report.acoustic_receipt_checks += 2;
            if !acoustic_complete_passed {
                fail(
                    &mut report,
                    &case_id,
                    "acoustic_complete_gate",
                    "exact acoustic observation did not satisfy the complete zero-error gate".to_string(),
                );
            }
            if !acoustic_receipt_valid {
                fail(
                    &mut report,
                    &case_id,
                    "acoustic_receipt_validation",
                    "fresh acoustic receipt failed persisted-state validation".to_string(),
                );
            }
            if !acoustic_receipt_lineage_valid {
                fail(
                    &mut report,
                    &case_id,
                    "acoustic_receipt_lineage",
                    "acoustic receipt did not validate against its source speech plan".to_string(),
                );
            }

            let delivery_target =
                symthaea_broca::SpeechDeliveryTarget::from_plan(&speech_plan);
            let exact_observation = SpeechDeliveryObservation {
                intent: Some(delivery_target.intent.clone()),
                clause_mode: Some(delivery_target.clause_mode),
                epistemic_delivery: Some(delivery_target.epistemic_delivery),
                focus: Some(match &delivery_target.focus_role {
                    Some(role) => symthaea_broca::ObservedFocus::Role(role.clone()),
                    None => symthaea_broca::ObservedFocus::None,
                }),
            };
            let exact_receipt = SpeechDeliveryReceipt::new(&speech_plan, exact_observation);
            let semantic_exact_passed = exact_receipt.error.passes();
            let semantic_receipt_lineage_valid =
                exact_receipt.validate_against_plan(&speech_plan).is_ok();
            if !semantic_receipt_lineage_valid {
                fail(
                    &mut report,
                    &case_id,
                    "semantic_receipt_lineage",
                    "semantic receipt did not validate against its source speech plan".to_string(),
                );
            }

            let mismatching_clause = if speech_plan.clause_mode
                == symthaea_broca::ClauseMode::Question
            {
                symthaea_broca::ClauseMode::Statement
            } else {
                symthaea_broca::ClauseMode::Question
            };
            let mismatch_receipt = SpeechDeliveryReceipt::new(
                &speech_plan,
                SpeechDeliveryObservation {
                    clause_mode: Some(mismatching_clause),
                    ..Default::default()
                },
            );
            let semantic_mismatch_rejected =
                mismatch_receipt.error.has_mismatch() && !mismatch_receipt.error.passes();

            let partial_receipt = SpeechDeliveryReceipt::new(
                &speech_plan,
                SpeechDeliveryObservation {
                    intent: Some(delivery_target.intent.clone()),
                    ..Default::default()
                },
            );
            let semantic_partial_gate_rejected = !partial_receipt.error.passes();
            report.semantic_delivery_checks += 4;

            if !semantic_exact_passed
                || !semantic_mismatch_rejected
                || !semantic_partial_gate_rejected
            {
                fail(
                    &mut report,
                    &case_id,
                    "semantic_delivery_gates",
                    format!(
                        "exact_pass={semantic_exact_passed}, mismatch_rejected={semantic_mismatch_rejected}, partial_gate_rejected={semantic_partial_gate_rejected}"
                    ),
                );
            }

            let surface_a = speech_plan.grounding_surface();
            let surface_b = speech_plan.grounding_surface();
            let grounding_deterministic = surface_a == surface_b;
            report.deterministic_grounding_checks += 1;
            if !grounding_deterministic {
                fail(
                    &mut report,
                    &case_id,
                    "speech_plan_grounding_determinism",
                    "grounding surfaces differed across repeated calls".to_string(),
                );
            }

            let case_ok = speech_plan_valid
                && acoustic_receipt_valid
                && acoustic_complete_passed
                && acoustic_receipt_lineage_valid
                && linguistic_valid
                && linguistic_lineage_valid
                && phonological_lineage_before_binding
                && linguistic_to_phonological_preserved_intent
                && linguistic_lexical_binding_valid
                && role_only_rejected
                && phonological_binding_succeeded
                && phonological_persistence_validated
                && lexical_binding_succeeded
                && lexical_missing_provenance_rejected
                && semantic_exact_passed
                && semantic_receipt_lineage_valid
                && semantic_mismatch_rejected
                && semantic_partial_gate_rejected
                && grounding_deterministic;

            if case_ok {
                report.passed_cases += 1;
            }

            report.cases.push(AuditCase {
                case_id,
                speech_plan_valid,
                acoustic_receipt_valid,
                acoustic_complete_passed,
                acoustic_receipt_lineage_valid,
                intent: (*intent_name).to_string(),
                epistemic_input: epistemic,
                plan_epistemic: format!("{:?}", speech_plan.epistemic_delivery),
                clause_mode: format!("{:?}", speech_plan.clause_mode),
                formulation_strategy: format!("{:?}", linguistic.strategy),
                linguistic_ready,
                linguistic_lineage_valid,
                phonological_lineage_valid: phonological_lineage_before_binding,
                linguistic_lexical_binding_valid,
                content_binding: format!("{:?}", lexical.content_binding),
                role_only_rejected_segments: role_only_rejected,
                phonological_binding_succeeded,
                phonological_persistence_validated,
                lexical_binding_succeeded,
                lexical_missing_provenance_rejected,
                semantic_exact_passed,
                semantic_receipt_lineage_valid,
                semantic_mismatch_rejected,
                semantic_partial_gate_rejected,
                linguistic_to_phonological_preserved_intent,
                grounding_deterministic,
            });
        }
    }

    if !report.failures.is_empty() || report.passed_cases != report.matrix_cases {
        write_report(json_out.as_ref(), &report)?;
        anyhow::bail!(
            "contract audit failed: {}/{} matrix cases passed; {} invariant failures",
            report.passed_cases,
            report.matrix_cases,
            report.failures.len()
        );
    }

    write_report(json_out.as_ref(), &report)?;
    println!(
        "broca production contract audit: {}/{} cases passed",
        report.passed_cases, report.matrix_cases
    );
    Ok(())
}

fn sample_segments(focused: bool) -> Vec<PhonemeSlot> {
    vec![
        PhonemeSlot::new("B", 0, SyllableStress::Primary, true, focused, false),
        PhonemeSlot::new("AE", 0, SyllableStress::Primary, false, focused, true),
    ]
}

fn case_id(intent_index: usize, epistemic: f32) -> String {
    format!("intent-{intent_index}-epistemic-{epistemic:.1}")
}

fn fail(report: &mut AuditReport, case_id: &str, invariant: &str, detail: String) {
    report.failures.push(AuditFailure {
        case_id: case_id.to_string(),
        invariant: invariant.to_string(),
        detail,
    });
}

fn parse_json_out() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--json-out" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn write_report(path: Option<&PathBuf>, report: &AuditReport) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(report)?)?;
    }
    Ok(())
}
