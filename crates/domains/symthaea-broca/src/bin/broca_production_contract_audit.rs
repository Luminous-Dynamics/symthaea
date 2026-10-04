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
                    && lexical.lexical_provenance.is_some()
                    && lexical.validate().is_ok();

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
            if !semantic_exact_passed
                || !semantic_mismatch_rejected
                || !semantic_partial_gate_rejected
                || !phonological_persistence_validated
                || !phonological_binding_succeeded
                || !lexical_binding_succeeded
                || !lexical_missing_provenance_rejected
            {
                fail(
                    &mut report,
                    case_id(intent_index, epistemic),
                    "semantic_delivery",
                    format!(
                        "exact_pass={semantic_exact_passed}, mismatch_rejected={semantic_mismatch_rejected}, partial_gate_rejected={semantic_partial_gate_rejected}, phonological_valid={phonological_persistence_validated}, lexical_valid={lexical_binding_succeeded}, lexical_missing_provenance_rejected={lexical_missing_provenance_rejected}"
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
                && semantic_partial_gate_rejected
                && phonological_persistence_validated
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