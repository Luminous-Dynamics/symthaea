// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic audit of the typed Broca speech-production contracts.
//!
//! This is a contract/invariant probe, not a naturalness benchmark. It exercises the
//! complete supported intent/epistemic matrix without invoking a stochastic generator.

use std::{fs, path::PathBuf, process};

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea_broca::{
    ContentBindingStatus, PhonemeSlot, PhonologicalPlan, SpeechDeliveryObservation,
    SpeechDeliveryReceipt, SpeechPlan, SyllableStress, StructuredDecoder, ThoughtChannels,
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
    intent: String,
    epistemic_input: f32,
    plan_epistemic: String,
    clause_mode: String,
    content_binding: String,
    role_only_rejected_segments: bool,
    phonological_binding_succeeded: bool,
    lexical_binding_succeeded: bool,
    lexical_missing_provenance_rejected: bool,
    semantic_exact_passed: bool,
    semantic_mismatch_rejected: bool,
    semantic_partial_gate_rejected: bool,
    phonological_persistence_validated: bool,
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
    let genesis = GenesisSeed::from_phrase("broca-production-contract-audit-v1");
    let decoder = StructuredDecoder::new(&genesis);

    let mut report = AuditReport {
        schema_version: 1,
        evidence_level: "deterministic-contract-audit",
        matrix_cases: 0,
        passed_cases: 0,
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

            let mut channels = ThoughtChannels::with_intent(intent_index);
            channels.set_epistemic(epistemic);
            let readout = decoder.decode(&channels);
            if readout.intent != *intent_name {
                fail(
                    &mut report,
                    case_id(intent_index, epistemic),
                    "intent_decode",
                    format!("expected {intent_name}, got {}", readout.intent),
                );
            }

            let speech_plan = SpeechPlan::from_readout(&channels, &readout);
            let mut phonological = PhonologicalPlan::from_speech_plan(&speech_plan);

            let role_only_error = phonological
                .bind_segments(
                    sample_segments(false),
                    ContentBindingStatus::RoleStructureOnly,
                )
                .expect_err("role-only binding must reject segments");
            let role_only_rejected =
                matches!(role_only_error, symthaea_broca::PhonologicalPlanError::RoleOnlyWithSegments);
            report.phonological_binding_checks += 1;
            if !role_only_rejected {
                fail(
                    &mut report,
                    case_id(intent_index, epistemic),
                    "role_only_segment_rejection",
                    format!("unexpected error: {role_only_error}"),
                );
            }

            phonological
                .bind_segments(
                    sample_segments(false),
                    ContentBindingStatus::PhonologicallyBound,
                )
                .with_context(|| format!("phonological binding failed for {intent_name}/{epistemic}"))?;
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
            if !lexical_missing_provenance_rejected {
                fail(
                    &mut report,
                    case_id(intent_index, epistemic),
                    "lexical_provenance_required",
                    format!("unexpected error: {lexical_missing_provenance}"),
                );
            }

            let mut lexical = PhonologicalPlan::from_speech_plan(&speech_plan);
            lexical
                .bind_lexical_segments(sample_segments(true), format!("lexeme::{intent_name}"))
                .with_context(|| format!("lexical binding failed for {intent_name}/{epistemic}"))?;
            let lexical_binding_succeeded =
                lexical.content_binding == ContentBindingStatus::LexicallyBound
                    && lexical.lexical_provenance.is_some();

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
            report.semantic_delivery_checks += 3;
            if !semantic_exact_passed || !semantic_mismatch_rejected {
                fail(
                    &mut report,
                    case_id(intent_index, epistemic),
                    "semantic_delivery",
                    format!(
                        "exact_pass={semantic_exact_passed}, mismatch_rejected={semantic_mismatch_rejected}"
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
                    case_id(intent_index, epistemic),
                    "speech_plan_grounding_determinism",
                    "grounding surfaces differed across repeated calls".to_string(),
                );
            }

            let case_ok = role_only_rejected
                && phonological_binding_succeeded
                && lexical_binding_succeeded
                && lexical_missing_provenance_rejected
                && semantic_exact_passed
                && semantic_mismatch_rejected
                && grounding_deterministic;

            if case_ok {
                report.passed_cases += 1;
            }

            report.cases.push(AuditCase {
                case_id: case_id(intent_index, epistemic),
                intent: (*intent_name).to_string(),
                epistemic_input: epistemic,
                plan_epistemic: format!("{:?}", speech_plan.epistemic_delivery),
                clause_mode: format!("{:?}", speech_plan.clause_mode),
                content_binding: format!("{:?}", lexical.content_binding),
                role_only_rejected_segments: role_only_rejected,
                phonological_binding_succeeded,
                lexical_binding_succeeded,
                lexical_missing_provenance_rejected,
                semantic_exact_passed,
                semantic_mismatch_rejected,
                semantic_partial_gate_rejected,
                phonological_persistence_validated,
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
        PhonemeSlot::new(
            "B",
            0,
            SyllableStress::Primary,
            true,
            focused,
            false,
        ),
        PhonemeSlot::new(
            "AE",
            0,
            SyllableStress::Primary,
            false,
            focused,
            true,
        ),
    ]
}

fn case_id(intent_index: usize, epistemic: f32) -> String {
    format!("intent-{intent_index}-epistemic-{epistemic:.1}")
}

fn fail(report: &mut AuditReport, case_id: String, invariant: &str, detail: String) {
    report.failures.push(AuditFailure {
        case_id,
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
