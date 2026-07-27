// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic Muse Analyst v1.
//!
//! The native score and grammar plan are authoritative. Audio features and
//! learned embeddings may later verify or predict perception, but cannot
//! rewrite facts emitted here.

use std::collections::{BTreeMap, BTreeSet};

use symthaea_muse_protocol::{
    AnalystCheck, AnalystCheckStatus, AnalystDisposition, AnalystEscalation, AnalystPieceBundle,
    AnalystStructuralSummary, AudioIntegrityEvidence, CadenceTrace, ComposerStructuralTrace,
    CulturalReviewSummary, EvidenceBasis, EvidenceSource, EvidenceSourceEnvelope, EvidenceStatus,
    GrammarProvenance, ListenCompositionBundle, ListenPerformanceBundle,
    MotifEvidenceReconciliation, MotifEvidenceReconciliationEntry, MotifEvidenceRelationship,
    MotifOccurrenceTrace, MotifRealizationAnalysis, ObligationState, ObligationTransitionTrace,
    RequestedRealizedIntent, StructuralSpanTrace, TraceVerificationIssue, TraceVerificationReport,
    VerificationStatus,
};
use symthaea_music_theory::{
    AssertedObligationState, GrammarStructuralTrace, ScoreEventRef, VoiceRole,
};
use symthaea_music_theory::{GrammarPlanEvidence, MusicalIntent};

pub const ANALYST_ENGINE_VERSION: &str = "muse-analyst-symbolic-v2";

fn basis(status: EvidenceStatus, method: &str, limitations: &[&str]) -> EvidenceBasis {
    EvidenceBasis {
        status,
        source_method: method.to_string(),
        confidence: None,
        limitations: limitations.iter().map(|item| (*item).to_string()).collect(),
    }
}

fn observed_check(
    code: &str,
    label: &str,
    passed: bool,
    expected: impl Into<String>,
    observed: impl Into<String>,
) -> AnalystCheck {
    AnalystCheck {
        code: code.to_string(),
        label: label.to_string(),
        status: if passed {
            AnalystCheckStatus::Pass
        } else {
            AnalystCheckStatus::Fail
        },
        expected: expected.into(),
        observed: observed.into(),
        basis: basis(
            EvidenceStatus::Observed,
            ANALYST_ENGINE_VERSION,
            &["This check establishes symbolic compliance, not artistic quality."],
        ),
    }
}

fn plan_checks(plan: Option<&GrammarPlanEvidence>) -> Vec<AnalystCheck> {
    let Some(plan) = plan else {
        return vec![AnalystCheck {
            code: "grammar-plan-available".into(),
            label: "Grammar plan retained".into(),
            status: AnalystCheckStatus::InsufficientEvidence,
            expected: "an explicit grammar-owned plan".into(),
            observed: "no plan retained for this candidate".into(),
            basis: basis(
                EvidenceStatus::Observed,
                ANALYST_ENGINE_VERSION,
                &["A score cannot prove which planner decisions created it."],
            ),
        }];
    };
    match plan {
        GrammarPlanEvidence::PeriodSentence(plan) => vec![observed_check(
            "dedicated-period-sentence-engine",
            "Period/Sentence grammar owns phrase and closure obligations",
            plan.section_count > 0
                && !plan.cadence_strategy.is_empty()
                && !plan.ending_strategy.is_empty(),
            "nonempty form, cadence strategy, return obligation, and ending strategy",
            format!(
                "{} sections; {}; {}",
                plan.section_count, plan.cadence_strategy, plan.ending_strategy
            ),
        )],
        GrammarPlanEvidence::Contrapuntal(plan) => vec![observed_check(
            "dedicated-contrapuntal-engine",
            "Contrapuntal grammar owns imitative development",
            plan.countersubject
                && plan.inversion
                && plan.augmentation
                && plan.stretto
                && plan.controlled_dissonance
                && plan.final_combination,
            "subject/answer, countersubject, inversion, augmentation, stretto, dissonance control, final combination",
            format!(
                "engine {}; answer {}; entries {:?}",
                plan.engine_version, plan.answer_class, plan.subject_entry_order
            ),
        )],
        GrammarPlanEvidence::GrooveCycle(plan) => plan
            .obligations
            .iter()
            .map(|obligation| {
                observed_check(
                    &format!("grammar-obligation-{}", obligation.code),
                    "Groove-cycle obligation",
                    obligation.fulfilled,
                    "fulfilled",
                    &obligation.evidence,
                )
            })
            .collect(),
        GrammarPlanEvidence::AdditiveProcess(plan) => {
            let peak = plan.prefix_lengths.iter().copied().max().unwrap_or(0);
            let peak_index = plan
                .prefix_lengths
                .iter()
                .position(|length| *length == peak)
                .unwrap_or(0);
            let audible = !plan.prefix_lengths.is_empty()
                && plan.prefix_lengths[..=peak_index]
                    .windows(2)
                    .all(|window| window[0] <= window[1])
                && plan.prefix_lengths[peak_index..]
                    .windows(2)
                    .all(|window| window[0] >= window[1]);
            vec![observed_check(
                "grammar-obligation-audible-process",
                "Additive rule is structurally audible",
                audible,
                "nondecreasing growth followed by nonincreasing release",
                format!("prefix lengths {:?}", plan.prefix_lengths),
            )]
        }
        GrammarPlanEvidence::ModalArc(plan) => vec![observed_check(
            "grammar-obligation-ordered-modal-stages",
            "Modal stages remain ordered",
            plan.alap_end.beats() < plan.jor_end.beats()
                && plan.jor_end.beats() < plan.jhala_end.beats(),
            "exposition end < pulse end < intensification end",
            format!(
                "{:.2} < {:.2} < {:.2} beats",
                plan.alap_end.beats(),
                plan.jor_end.beats(),
                plan.jhala_end.beats()
            ),
        )],
        GrammarPlanEvidence::CallResponse(plan) => vec![
            // Label scoped precisely to what's actually checked (fixed
            // 2026-07-26, an independent code-review workflow caught the
            // previous wording overclaiming): `call_starts`/
            // `response_starts` record only the FIRST chorus's bar
            // template (every chorus reuses the same relative [0,4,8]/
            // [2,6,10] offsets by construction, so this is the real
            // template every chorus follows) -- this check verifies that
            // template's temporal ordering, NOT that each chorus's
            // ACTUAL emitted response notes differ from that chorus's
            // call (see `call_response.rs`'s own
            // `a_three_chorus_piece_genuinely_varies_its_response_across_
            // choruses` test for a real-notes check of that, which this
            // Analyst check does not have score access to perform).
            observed_check(
                "grammar-obligation-call-answered-by-response",
                "The call/response bar template (used by every chorus) is temporally ordered: each response strictly follows its call",
                plan.call_starts.len() == plan.response_starts.len()
                    && plan
                        .call_starts
                        .iter()
                        .zip(plan.response_starts.iter())
                        .all(|(&call, &response)| response > call),
                "each call bar strictly precedes its paired response bar",
                format!(
                    "{} choruses; call bars {:?}; response bars {:?}",
                    plan.choruses, plan.call_starts, plan.response_starts
                ),
            ),
            // Added 2026-07-26 after a real listening session found 3
            // seeds at `bars: 4` realizing as identical 36-bar pieces --
            // traced to `chorus_count` never having been wired to
            // `intent.bars` at all. This check independently re-derives
            // the SAME div_ceil quantization policy the engine claims to
            // use (not just trusting the reported fields agree with each
            // other) and confirms the realized length never falls short
            // of what was requested nor exceeds the next whole chorus.
            observed_check(
                "grammar-obligation-duration-quantization-policy",
                "Realized length rounds the request up to the nearest whole chorus, never down or past it",
                plan.bars_per_chorus > 0
                    && plan.choruses == plan.requested_bars.max(1).div_ceil(plan.bars_per_chorus)
                    && plan.realized_bars == plan.choruses * plan.bars_per_chorus,
                "realized_bars = ceil(requested_bars / bars_per_chorus) * bars_per_chorus",
                format!(
                    "requested {} bars -> realized {} bars ({} chorus(es) of {})",
                    plan.requested_bars, plan.realized_bars, plan.choruses, plan.bars_per_chorus
                ),
            ),
            // Added 2026-07-26, the user's own follow-up once duration was
            // fixed ("do not generate identical chorus templates back-to-
            // back"): a multi-chorus piece must use MORE than one distinct
            // ChorusRole across its trajectory. Re-derives this from the
            // actual `trajectory` data rather than trusting the engine's
            // own claim that it develops -- a single-chorus piece has
            // nothing to develop across, so it's exempt.
            //
            // Known limitation (confirmed by an independent code-review
            // workflow, 2026-07-26): this only inspects the DECLARED role
            // labels in `plan.trajectory`, not the actual emitted score
            // notes -- it would still pass if `response_for` had a bug
            // that made every chorus emit identical response material
            // regardless of its labeled role, since `trajectory_for`
            // picks the labels independently, before `response_for` is
            // ever called. This Analyst check has no access to the real
            // `Score`'s notes to check that directly (unlike
            // `call_response.rs`'s own
            // `a_three_chorus_piece_genuinely_varies_its_response_across_
            // choruses` test, which does). Verifies label diversity, not
            // realized-material diversity -- disclosed, not silently
            // overclaimed.
            observed_check(
                "grammar-obligation-chorus-trajectory-develops",
                "A multi-chorus piece DECLARES more than one chorus role (label-level; does not verify the emitted notes differ)",
                plan.choruses <= 1 || {
                    let distinct: std::collections::HashSet<_> = plan.trajectory.iter().collect();
                    distinct.len() > 1
                },
                "trajectory.len() > 1 distinct role(s) whenever choruses > 1",
                format!(
                    "{} choruses; trajectory {:?}",
                    plan.choruses, plan.trajectory
                ),
            ),
        ],
        GrammarPlanEvidence::JazzChorus(plan) => vec![
            // Same reasoning as call_response's identical check: re-derives
            // the div_ceil quantization policy independently rather than
            // trusting the reported fields agree with each other.
            observed_check(
                "grammar-obligation-duration-quantization-policy",
                "Realized length rounds the request up to the nearest whole chorus, never down or past it",
                plan.bars_per_chorus > 0
                    && plan.choruses == plan.requested_bars.max(1).div_ceil(plan.bars_per_chorus)
                    && plan.realized_bars == plan.choruses * plan.bars_per_chorus,
                "realized_bars = ceil(requested_bars / bars_per_chorus) * bars_per_chorus",
                format!(
                    "requested {} bars -> realized {} bars ({} chorus(es) of {})",
                    plan.requested_bars, plan.realized_bars, plan.choruses, plan.bars_per_chorus
                ),
            ),
            // Label-level only, same disclosed limitation as
            // call_response's identical check: this doesn't verify the
            // emitted notes actually differ (see jazz_chorus.rs's own
            // multi_chorus_pieces_genuinely_vary_across_seeds test for a
            // real-notes check of that).
            observed_check(
                "grammar-obligation-chorus-trajectory-develops",
                "A multi-chorus piece DECLARES more than one chorus role (label-level; does not verify the emitted notes differ)",
                plan.choruses <= 1 || {
                    let distinct: std::collections::HashSet<_> = plan.trajectory.iter().collect();
                    distinct.len() > 1
                },
                "trajectory.len() > 1 distinct role(s) whenever choruses > 1",
                format!(
                    "{} choruses; trajectory {:?}",
                    plan.choruses, plan.trajectory
                ),
            ),
        ],
        GrammarPlanEvidence::Compatibility {
            family,
            form_available,
        } => vec![AnalystCheck {
            code: "dedicated-family-engine".into(),
            label: "Dedicated grammar engine".into(),
            status: AnalystCheckStatus::InsufficientEvidence,
            expected: format!("dedicated {family:?} planning evidence"),
            observed: format!("compatibility composer; form retained: {form_available}"),
            basis: basis(
                EvidenceStatus::Observed,
                ANALYST_ENGINE_VERSION,
                &[
                    "Compatibility output may be valid music but cannot establish a dedicated grammar.",
                ],
            ),
        }],
    }
}

fn normalized_position(beats: f64, total: f64) -> f64 {
    (beats / total.max(1e-9)).clamp(0.0, 1.0)
}

fn trace_event_id(reference: ScoreEventRef) -> String {
    let role = match reference.role {
        VoiceRole::Melody => "melody",
        VoiceRole::Harmony => "harmony",
        VoiceRole::Bass => "bass",
        VoiceRole::CounterMelody => "counter",
    };
    format!("score-{role}-{}", reference.role_index)
}

fn assertion_envelope(
    record_id: &str,
    score_sha256: &str,
    created_at_unix_ms: u64,
) -> EvidenceSourceEnvelope {
    EvidenceSourceEnvelope {
        record_id: format!("composer-assertion:{record_id}"),
        schema_version: 1,
        source: EvidenceSource::ComposerAssertion,
        verification_status: VerificationStatus::Unchecked,
        producer: "symthaea-music-theory/grammar-owner".into(),
        producer_version: symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION.into(),
        artifact_sha256: score_sha256.into(),
        created_at_unix_ms,
        dependency_record_ids: Vec::new(),
        uncertainty: None,
        limitations: vec![
            "This is a composer's structural assertion; independent symbolic verification is required."
                .into(),
        ],
    }
}

fn obligation_state(state: AssertedObligationState) -> ObligationState {
    match state {
        AssertedObligationState::Created => ObligationState::Created,
        AssertedObligationState::Reinforced => ObligationState::Reinforced,
        AssertedObligationState::Deferred => ObligationState::Deferred,
        AssertedObligationState::Transformed => ObligationState::Transformed,
        AssertedObligationState::Fulfilled => ObligationState::Fulfilled,
        AssertedObligationState::Abandoned => ObligationState::Abandoned,
        AssertedObligationState::Unresolved => ObligationState::Unresolved,
    }
}

/// Translate the theory-native `(voice, index)` trace into immutable public
/// score-event IDs. No claim is promoted here; the result remains unchecked.
pub fn composer_trace_from_theory(
    trace: &GrammarStructuralTrace,
    score_sha256: &str,
    created_at_unix_ms: u64,
    motif_family: Option<(&str, u32)>,
) -> ComposerStructuralTrace {
    let (motif_family_id, motif_family_version) = motif_family.unwrap_or(("motif-primary", 1));
    ComposerStructuralTrace {
        trace_schema_version: 1,
        structures: trace
            .structures
            .iter()
            .map(|item| StructuralSpanTrace {
                id: item.id.clone(),
                kind: item.kind.clone(),
                parent_id: item.parent_id.clone(),
                start_event_id: trace_event_id(item.start),
                end_event_id: trace_event_id(item.end),
                assertion: assertion_envelope(&item.id, score_sha256, created_at_unix_ms),
            })
            .collect(),
        motif_occurrences: trace
            .motif_occurrences
            .iter()
            .map(|item| MotifOccurrenceTrace {
                motif_family_id: motif_family_id.into(),
                motif_family_version,
                occurrence_id: item.occurrence_id.clone(),
                score_event_ids: item.events.iter().copied().map(trace_event_id).collect(),
                voice_or_layer: "Melody".into(),
                formal_region_id: item.formal_region_id.clone(),
                transformation_chain: vec![item.transformation.clone()],
                claimed_preserved_invariants: item.preserved_invariants.clone(),
                changed_dimensions: item.changed_dimensions.clone(),
                literal_distance: item.literal_distance,
                structural_distance: item.structural_distance,
                role_binding: item.role_binding.clone(),
                originating_decision_id: item.decision_id.clone(),
                assertion: assertion_envelope(
                    &item.occurrence_id,
                    score_sha256,
                    created_at_unix_ms,
                ),
            })
            .collect(),
        cadences: trace
            .cadences
            .iter()
            .map(|item| CadenceTrace {
                cadence_id: item.cadence_id.clone(),
                proposed_type: item.proposed_type.clone(),
                grammar_owner: item.decision_id.clone(),
                preparation_event_ids: item
                    .preparation
                    .iter()
                    .copied()
                    .map(trace_event_id)
                    .collect(),
                arrival_event_id: trace_event_id(item.arrival),
                harmonic_evidence_event_ids: Vec::new(),
                melodic_evidence_event_ids: vec![trace_event_id(item.arrival)],
                altered_downstream: false,
                fulfils_obligation_id: item.fulfils_obligation_id.clone(),
                assertion: assertion_envelope(&item.cadence_id, score_sha256, created_at_unix_ms),
            })
            .collect(),
        obligation_transitions: trace
            .obligation_transitions
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let record_id = format!("{}:{index}", item.obligation_id);
                ObligationTransitionTrace {
                    obligation_id: item.obligation_id.clone(),
                    from: item.from.map(obligation_state),
                    to: obligation_state(item.to),
                    score_event_ids: item.evidence.iter().copied().map(trace_event_id).collect(),
                    responsible_pass: item.responsible_pass.clone(),
                    transformation: item.transformation.clone(),
                    assertion: assertion_envelope(&record_id, score_sha256, created_at_unix_ms),
                }
            })
            .collect(),
    }
}

/// Analyze one native composition bundle with its retained grammar evidence.
pub fn analyze_piece(
    composition: &ListenCompositionBundle,
    performance: Option<&ListenPerformanceBundle>,
    provenance: &GrammarProvenance,
    plan: Option<&GrammarPlanEvidence>,
    intent: &MusicalIntent,
) -> AnalystPieceBundle {
    let total = composition.duration_beats.max(1e-9);
    let phrase_start_positions: Vec<f64> = composition
        .phrases
        .iter()
        .map(|phrase| normalized_position(phrase.start.beats, total))
        .collect();
    let cadence_positions: Vec<f64> = composition
        .cadences
        .iter()
        .map(|cadence| normalized_position(cadence.at.beats, total))
        .collect();
    let climax_positions: Vec<f64> = composition
        .notes
        .iter()
        .filter(|note| note.emphasis == "climax")
        .map(|note| normalized_position(note.onset.beats, total))
        .collect();
    let phrase_recurrence_intervals = phrase_start_positions
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();

    let motif = composition.motif_definitions.first().map(|definition| {
        let occurrences: Vec<_> = composition
            .motif_occurrences
            .iter()
            .filter(|occurrence| occurrence.motif_id == definition.id)
            .collect();
        let positions: Vec<_> = occurrences
            .iter()
            .map(|occurrence| normalized_position(occurrence.start.beats, total))
            .collect();
        let best = occurrences
            .iter()
            .map(|occurrence| occurrence.similarity)
            .max_by(f32::total_cmp);
        let mean = (!occurrences.is_empty()).then(|| {
            occurrences
                .iter()
                .map(|occurrence| occurrence.similarity)
                .sum::<f32>()
                / occurrences.len() as f32
        });
        let transformations: BTreeSet<_> = occurrences
            .iter()
            .map(|occurrence| occurrence.transformation.clone())
            .collect();
        MotifRealizationAnalysis {
            motif_id: definition.id.clone(),
            occurrence_count: occurrences.len(),
            transformations: transformations.into_iter().collect(),
            best_symbolic_similarity: best,
            mean_symbolic_similarity: mean,
            appears_in_final_fifth: positions.iter().any(|position| *position >= 0.8),
            omitted_or_below_threshold: occurrences.is_empty(),
            canonical_installed_exactly_at_ingress: None,
            final_literal_occurrence_count: occurrences
                .iter()
                .filter(|occurrence| occurrence.similarity >= 0.999)
                .count(),
            final_contract_valid_occurrence_count: occurrences.len(),
            occurrence_positions: positions,
            basis: basis(
                EvidenceStatus::Inferred,
                "recipe-motif-score-window-scan-v1",
                &[
                    "Symbolic similarity is not a listener-recognition probability.",
                    "Occurrences below the conservative scan threshold are not reported.",
                ],
            ),
        }
    });
    let motif_positions = motif
        .as_ref()
        .map(|report| report.occurrence_positions.clone())
        .unwrap_or_default();
    let ending_has_motif_return = motif_positions.iter().any(|position| *position >= 0.8);
    let ending_has_cadential_marker = cadence_positions.iter().any(|position| *position >= 0.8);

    let register_min = composition.notes.iter().map(|note| note.midi).min();
    let register_max = composition.notes.iter().map(|note| note.midi).max();
    let mut checks = vec![observed_check(
        "score-nonempty",
        "Score contains realizable events",
        !composition.notes.is_empty(),
        "at least one symbolic note",
        format!("{} notes", composition.notes.len()),
    )];
    checks.extend(plan_checks(plan));
    if let Some(motif) = &motif {
        checks.push(AnalystCheck {
            code: "motif-realized".into(),
            label: "Supplied motif appears in the score".into(),
            status: if motif.omitted_or_below_threshold {
                AnalystCheckStatus::InsufficientEvidence
            } else {
                AnalystCheckStatus::Pass
            },
            expected: "at least one transformation-aware score occurrence".into(),
            observed: format!(
                "{} occurrence(s), transformations {:?}",
                motif.occurrence_count, motif.transformations
            ),
            basis: motif.basis.clone(),
        });
    }
    if let Some(performance) = performance {
        let mapped: BTreeSet<_> = performance
            .notes
            .iter()
            .filter_map(|note| note.source_note_id.as_deref())
            .collect();
        let expected: BTreeSet<_> = composition
            .notes
            .iter()
            .map(|note| note.id.as_str())
            .collect();
        let missing = expected.difference(&mapped).count();
        let unknown = mapped.difference(&expected).count();
        checks.push(observed_check(
            "renderer-source-mapping",
            "Renderer preserves source-event traceability",
            missing == 0 && unknown == 0,
            format!("all {} source IDs mapped exactly", expected.len()),
            format!(
                "{} unique mapped IDs; {missing} missing and {unknown} unknown",
                mapped.len()
            ),
        ));
    }

    let mut uncertainty = 0.05_f32;
    let mut escalations = Vec::new();
    if plan.is_none() {
        uncertainty += 0.30;
        escalations.push(AnalystEscalation {
            code: "missing-grammar-plan".into(),
            reason: "The score does not retain enough evidence to audit grammar causality.".into(),
            required_reviewer: "composer engineer".into(),
        });
    }
    if motif
        .as_ref()
        .is_some_and(|report| report.omitted_or_below_threshold)
    {
        uncertainty += 0.25;
        escalations.push(AnalystEscalation {
            code: "motif-below-threshold".into(),
            reason: "The supplied motif was not found by the conservative symbolic scan.".into(),
            required_reviewer: "motif reviewer".into(),
        });
    }
    if checks
        .iter()
        .any(|check| check.status == AnalystCheckStatus::Fail)
    {
        uncertainty += 0.30;
        escalations.push(AnalystEscalation {
            code: "symbolic-compliance-failure".into(),
            reason: "At least one deterministic grammar or renderer obligation failed.".into(),
            required_reviewer: "composer engineer".into(),
        });
    }
    if provenance.culturally_qualified {
        uncertainty = uncertainty.max(0.50);
        escalations.push(AnalystEscalation {
            code: "cultural-review-required".into(),
            reason:
                "Structural validity cannot establish cultural authenticity or respectful practice."
                    .into(),
            required_reviewer: "qualified tradition bearer or musician".into(),
        });
    }
    uncertainty = uncertainty.clamp(0.0, 1.0);
    let disposition = if escalations.is_empty() {
        AnalystDisposition::RoutineEvidenceComplete
    } else {
        AnalystDisposition::HumanReview
    };

    AnalystPieceBundle {
        analyzer_version: ANALYST_ENGINE_VERSION.into(),
        grammar_family: provenance.family.clone(),
        phrase_grammar: provenance.phrase_grammar.clone(),
        harmonic_syntax: provenance.harmonic_syntax.clone(),
        performance_dialect: provenance.performance_dialect.clone(),
        plan_kind: provenance.plan_kind.clone(),
        culturally_qualified: provenance.culturally_qualified,
        cultural_review: CulturalReviewSummary::default(),
        requested_realized: RequestedRealizedIntent {
            requested_valence: intent.valence,
            requested_arousal: intent.arousal,
            requested_energy: intent.energy,
            requested_bars: intent.bars,
            requested_tonic: intent.tonic.name().to_string(),
            realized_tempo_bpm: composition
                .tempo_map
                .first()
                .map_or(0.0, |point| point.bpm),
            realized_duration_beats: composition.duration_beats,
            realized_duration_seconds: composition.duration_seconds,
            realized_note_count: composition.notes.len(),
            realized_onset_density_per_second: composition.notes.len() as f64
                / composition.duration_seconds.max(1e-9),
            realized_register_min_midi: register_min,
            realized_register_max_midi: register_max,
        },
        structural: AnalystStructuralSummary {
            phrase_start_positions,
            cadence_positions,
            climax_positions,
            motif_positions,
            phrase_recurrence_intervals,
            ending_has_cadential_marker,
            ending_has_motif_return,
        },
        motif,
        checks,
        uncertainty,
        disposition,
        escalations,
        composer_trace: None,
        trace_verification: None,
        motif_reconciliation: None,
        audio_integrity: None,
        external_measurements: Vec::new(),
        limitations: vec![
            "Analyst v1 reports deterministic symbolic and renderer evidence; it does not score beauty, meaning, or replay value.".into(),
            "Motif recognition remains a human-calibrated prediction target, never a symbolic fact.".into(),
            "Cross-piece causal claims require matched-premise metadata and a comparison bundle.".into(),
        ],
    }
}

/// Inspect the actual rendered WAV for transport/integrity failures. This is
/// intentionally independent of compositional checks and uses conservative,
/// versioned thresholds suitable for triage rather than aesthetic scoring.
pub fn analyze_audio_integrity(
    wav: &[u8],
    audio_sha256: &str,
    created_at_unix_ms: u64,
) -> AudioIntegrityEvidence {
    const VERSION: &str = "muse-audio-integrity-v1";
    let mut issues = Vec::new();
    let reader = hound::WavReader::new(std::io::Cursor::new(wav));
    let Ok(mut reader) = reader else {
        return AudioIntegrityEvidence {
            analyzer_version: VERSION.into(),
            sample_rate_hz: 0,
            channels: 0,
            frame_count: 0,
            true_peak: 0.0,
            dc_offset: 0.0,
            clipping_sample_count: 0,
            near_silence_fraction: 1.0,
            first_difference_rms: 0.0,
            second_difference_rms: 0.0,
            impulse_outlier_count: 0,
            high_frequency_proxy_ratio: 0.0,
            issues: vec!["wav-decode-failed".into()],
            evidence: EvidenceSourceEnvelope {
                record_id: format!("audio-integrity:{audio_sha256}"),
                schema_version: 1,
                source: EvidenceSource::AudioMeasured,
                verification_status: VerificationStatus::InsufficientEvidence,
                producer: "symthaea-muse/audio-integrity".into(),
                producer_version: VERSION.into(),
                artifact_sha256: audio_sha256.into(),
                created_at_unix_ms,
                dependency_record_ids: Vec::new(),
                uncertainty: Some(1.0),
                limitations: vec!["The WAV container could not be decoded.".into()],
            },
        };
    };
    let spec = reader.spec();
    let scale = if spec.sample_format == hound::SampleFormat::Int {
        ((1_u64 << spec.bits_per_sample.saturating_sub(1)) - 1).max(1) as f32
    } else {
        1.0
    };
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .filter_map(Result::ok)
            .map(|sample| sample as f32 / scale)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
    };
    let channels = usize::from(spec.channels.max(1));
    let frame_count = samples.len() / channels;
    let true_peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    let dc_offset = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f32>() / samples.len() as f32
    };
    let clipping_sample_count = samples
        .iter()
        .filter(|sample| sample.abs() >= 0.999)
        .count();
    let near_silence_fraction = if samples.is_empty() {
        1.0
    } else {
        samples.iter().filter(|sample| sample.abs() < 1e-4).count() as f32 / samples.len() as f32
    };
    let mut first = Vec::with_capacity(samples.len());
    let mut second = Vec::with_capacity(samples.len());
    for channel in 0..channels {
        let lane: Vec<_> = samples
            .iter()
            .skip(channel)
            .step_by(channels)
            .copied()
            .collect();
        let lane_first: Vec<_> = lane.windows(2).map(|pair| pair[1] - pair[0]).collect();
        second.extend(lane_first.windows(2).map(|pair| pair[1] - pair[0]));
        first.extend(lane_first);
    }
    let rms = |values: &[f32]| {
        if values.is_empty() {
            0.0
        } else {
            (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32).sqrt()
        }
    };
    let first_difference_rms = rms(&first);
    let second_difference_rms = rms(&second);
    let impulse_threshold = (second_difference_rms * 12.0).max(0.2);
    let impulse_outlier_count = second
        .iter()
        .filter(|value| value.abs() > impulse_threshold)
        .count();
    let signal_rms = rms(&samples);
    let high_frequency_proxy_ratio =
        (first_difference_rms / (2.0 * signal_rms).max(1e-9)).clamp(0.0, 1.0);

    if samples.is_empty() {
        issues.push("audio-empty".into());
    }
    if clipping_sample_count > 0 {
        issues.push("clipping-detected".into());
    }
    if dc_offset.abs() > 0.02 {
        issues.push("dc-offset-elevated".into());
    }
    if impulse_outlier_count > (frame_count / 20_000).max(2) {
        issues.push("broadband-impulse-outliers".into());
    }
    if true_peak < 1e-4 {
        issues.push("audio-effectively-silent".into());
    }
    let status = if samples.is_empty() {
        VerificationStatus::InsufficientEvidence
    } else if issues.is_empty() {
        VerificationStatus::Verified
    } else {
        VerificationStatus::Discrepancy
    };
    AudioIntegrityEvidence {
        analyzer_version: VERSION.into(),
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        frame_count,
        true_peak,
        dc_offset,
        clipping_sample_count,
        near_silence_fraction,
        first_difference_rms,
        second_difference_rms,
        impulse_outlier_count,
        high_frequency_proxy_ratio,
        issues,
        evidence: EvidenceSourceEnvelope {
            record_id: format!("audio-integrity:{audio_sha256}"),
            schema_version: 1,
            source: EvidenceSource::AudioMeasured,
            verification_status: status,
            producer: "symthaea-muse/audio-integrity".into(),
            producer_version: VERSION.into(),
            artifact_sha256: audio_sha256.into(),
            created_at_unix_ms,
            dependency_record_ids: Vec::new(),
            uncertainty: None,
            limitations: vec![
                "High-frequency energy is a sample-difference proxy, not a calibrated spectral estimate."
                    .into(),
                "Threshold crossings triage likely defects and do not prove perceptual audibility."
                    .into(),
            ],
        },
    }
}

/// Independently verify composer claims against exact score-event IDs.
pub fn verify_composer_trace(
    trace: &ComposerStructuralTrace,
    composition: &ListenCompositionBundle,
    score_sha256: &str,
    created_at_unix_ms: u64,
) -> TraceVerificationReport {
    const VERSION: &str = "muse-structural-trace-verifier-v1";
    let events: BTreeMap<_, _> = composition
        .notes
        .iter()
        .map(|event| (event.id.as_str(), event))
        .collect();
    let structures: BTreeSet<_> = trace
        .structures
        .iter()
        .map(|item| item.id.as_str())
        .collect();
    let obligations: BTreeSet<_> = trace
        .obligation_transitions
        .iter()
        .map(|item| item.obligation_id.as_str())
        .collect();
    let motif_families: BTreeSet<_> = composition
        .motif_definitions
        .iter()
        .map(|definition| definition.id.as_str())
        .collect();
    let mut issues = Vec::new();
    let mut records = BTreeSet::new();
    fn check_assertion(
        records: &mut BTreeSet<String>,
        issues: &mut Vec<TraceVerificationIssue>,
        id: &str,
        source: &EvidenceSourceEnvelope,
    ) {
        records.insert(id.to_string());
        if source.source != EvidenceSource::ComposerAssertion {
            issues.push(TraceVerificationIssue {
                code: "invalid-assertion-source".into(),
                record_id: id.into(),
                message: "A direct trace must begin as a composer assertion.".into(),
            });
        }
    }

    for span in &trace.structures {
        check_assertion(&mut records, &mut issues, &span.id, &span.assertion);
        match (
            events.get(span.start_event_id.as_str()),
            events.get(span.end_event_id.as_str()),
        ) {
            (Some(start), Some(end)) if start.onset.beats <= end.onset.beats => {}
            (Some(_), Some(_)) => issues.push(TraceVerificationIssue {
                code: "structural-span-reversed".into(),
                record_id: span.id.clone(),
                message: "Structural span begins after it ends.".into(),
            }),
            _ => issues.push(TraceVerificationIssue {
                code: "structural-span-event-missing".into(),
                record_id: span.id.clone(),
                message: "Structural span references a missing event.".into(),
            }),
        }
        if span
            .parent_id
            .as_deref()
            .is_some_and(|parent| !structures.contains(parent))
        {
            issues.push(TraceVerificationIssue {
                code: "structural-parent-missing".into(),
                record_id: span.id.clone(),
                message: "Structural span references a missing parent.".into(),
            });
        }
    }
    for left in 0..trace.structures.len() {
        for right in left + 1..trace.structures.len() {
            let (a, b) = (&trace.structures[left], &trace.structures[right]);
            if a.kind != b.kind || a.parent_id != b.parent_id {
                continue;
            }
            let Some(a_start) = events.get(a.start_event_id.as_str()) else {
                continue;
            };
            let Some(a_end) = events.get(a.end_event_id.as_str()) else {
                continue;
            };
            let Some(b_start) = events.get(b.start_event_id.as_str()) else {
                continue;
            };
            let Some(b_end) = events.get(b.end_event_id.as_str()) else {
                continue;
            };
            if a_start.onset.beats < b_end.onset.beats && b_start.onset.beats < a_end.onset.beats {
                issues.push(TraceVerificationIssue {
                    code: "structural-sibling-overlap".into(),
                    record_id: format!("{}+{}", a.id, b.id),
                    message: "Sibling structural spans of the same kind overlap.".into(),
                });
            }
        }
    }

    for occurrence in &trace.motif_occurrences {
        check_assertion(
            &mut records,
            &mut issues,
            &occurrence.occurrence_id,
            &occurrence.assertion,
        );
        if !motif_families.contains(occurrence.motif_family_id.as_str()) {
            issues.push(TraceVerificationIssue {
                code: "motif-family-mismatch".into(),
                record_id: occurrence.occurrence_id.clone(),
                message: "Composer motif assertion does not name an emitted motif family.".into(),
            });
        }
        let occurrence_events: Vec<_> = occurrence
            .score_event_ids
            .iter()
            .filter_map(|id| events.get(id.as_str()).copied())
            .collect();
        if occurrence.score_event_ids.is_empty()
            || occurrence_events.len() != occurrence.score_event_ids.len()
        {
            issues.push(TraceVerificationIssue {
                code: "motif-event-missing".into(),
                record_id: occurrence.occurrence_id.clone(),
                message: "Motif claim is empty or references a missing event.".into(),
            });
        }
        if occurrence.formal_region_id != "piece"
            && !structures.contains(occurrence.formal_region_id.as_str())
        {
            issues.push(TraceVerificationIssue {
                code: "motif-region-missing".into(),
                record_id: occurrence.occurrence_id.clone(),
                message: "Motif claim references an unknown formal region.".into(),
            });
        }
        if occurrence
            .transformation_chain
            .iter()
            .any(|item| item == "literal")
            && occurrence.literal_distance > 0.05
        {
            issues.push(TraceVerificationIssue {
                code: "literal-distance-discrepancy".into(),
                record_id: occurrence.occurrence_id.clone(),
                message: "Literal transformation conflicts with measured distance.".into(),
            });
        }
        if !(0.0..=1.0).contains(&occurrence.structural_distance) {
            issues.push(TraceVerificationIssue {
                code: "motif-distance-out-of-range".into(),
                record_id: occurrence.occurrence_id.clone(),
                message: "Motif structural distance is outside [0, 1].".into(),
            });
        }
        if occurrence_events.windows(2).any(|pair| {
            pair[0].voice_role != pair[1].voice_role || pair[0].onset.beats > pair[1].onset.beats
        }) {
            issues.push(TraceVerificationIssue {
                code: "motif-event-order-discrepancy".into(),
                record_id: occurrence.occurrence_id.clone(),
                message: "Motif events are not an ordered sequence in one score voice.".into(),
            });
        }
        if let Some(definition) = composition
            .motif_definitions
            .iter()
            .find(|definition| definition.id == occurrence.motif_family_id)
            && definition.degrees.len() == occurrence_events.len()
            && definition.durations_beats.len() == occurrence_events.len()
            && occurrence_events.len() >= 2
        {
            let expected_contour: Vec<_> = definition
                .degrees
                .windows(2)
                .map(|pair| (pair[1] - pair[0]).signum())
                .collect();
            let actual_contour: Vec<_> = occurrence_events
                .windows(2)
                .map(|pair| (i32::from(pair[1].midi) - i32::from(pair[0].midi)).signum())
                .collect();
            let contour_similarity = expected_contour
                .iter()
                .zip(&actual_contour)
                .filter(|(expected, actual)| expected == actual)
                .count() as f64
                / expected_contour.len().max(1) as f64;
            let expected_total = definition.durations_beats.iter().sum::<f64>().max(1e-9);
            let actual_total = occurrence_events
                .iter()
                .map(|event| event.duration_beats)
                .sum::<f64>()
                .max(1e-9);
            let rhythm_similarity = 1.0
                - occurrence_events
                    .iter()
                    .zip(&definition.durations_beats)
                    .map(|(actual, expected)| {
                        (actual.duration_beats / actual_total - expected / expected_total).abs()
                    })
                    .sum::<f64>()
                    / 2.0;
            let measured_distance =
                (1.0 - (0.65 * contour_similarity + 0.35 * rhythm_similarity)) as f32;
            if (measured_distance - occurrence.structural_distance).abs() > 0.3 {
                issues.push(TraceVerificationIssue {
                    code: "motif-structural-distance-discrepancy".into(),
                    record_id: occurrence.occurrence_id.clone(),
                    message: format!(
                        "Claimed structural distance {:.3} disagrees with independent {:.3}.",
                        occurrence.structural_distance, measured_distance
                    ),
                });
            }
        }
    }

    for cadence in &trace.cadences {
        check_assertion(
            &mut records,
            &mut issues,
            &cadence.cadence_id,
            &cadence.assertion,
        );
        match events.get(cadence.arrival_event_id.as_str()) {
            Some(event) if event.emphasis == "cadential" => {}
            Some(_) => issues.push(TraceVerificationIssue {
                code: "cadence-arrival-not-marked".into(),
                record_id: cadence.cadence_id.clone(),
                message: "Claimed arrival is not cadential in the score.".into(),
            }),
            None => issues.push(TraceVerificationIssue {
                code: "cadence-arrival-missing".into(),
                record_id: cadence.cadence_id.clone(),
                message: "Claimed cadence arrival event is absent.".into(),
            }),
        }
        if cadence
            .preparation_event_ids
            .iter()
            .chain(&cadence.harmonic_evidence_event_ids)
            .chain(&cadence.melodic_evidence_event_ids)
            .any(|id| !events.contains_key(id.as_str()))
        {
            issues.push(TraceVerificationIssue {
                code: "cadence-evidence-event-missing".into(),
                record_id: cadence.cadence_id.clone(),
                message: "Cadence evidence references a missing event.".into(),
            });
        }
        if cadence
            .fulfils_obligation_id
            .as_deref()
            .is_some_and(|id| !obligations.contains(id))
        {
            issues.push(TraceVerificationIssue {
                code: "cadence-obligation-missing".into(),
                record_id: cadence.cadence_id.clone(),
                message: "Cadence references an unknown obligation.".into(),
            });
        }
    }

    let mut states = BTreeMap::<&str, ObligationState>::new();
    for transition in &trace.obligation_transitions {
        let id = format!("{}:{:?}", transition.obligation_id, transition.to);
        check_assertion(&mut records, &mut issues, &id, &transition.assertion);
        if transition.from.as_ref() != states.get(transition.obligation_id.as_str()) {
            issues.push(TraceVerificationIssue {
                code: "obligation-lifecycle-discontinuity".into(),
                record_id: id.clone(),
                message: "Obligation transition does not continue the prior state.".into(),
            });
        }
        if transition.responsible_pass.trim().is_empty() {
            issues.push(TraceVerificationIssue {
                code: "obligation-pass-missing".into(),
                record_id: id.clone(),
                message: "Obligation transition has no responsible pass.".into(),
            });
        }
        if transition
            .score_event_ids
            .iter()
            .any(|event| !events.contains_key(event.as_str()))
            || (transition.to == ObligationState::Fulfilled
                && transition.score_event_ids.is_empty())
        {
            issues.push(TraceVerificationIssue {
                code: "obligation-evidence-missing".into(),
                record_id: id,
                message: "Obligation transition lacks valid score evidence.".into(),
            });
        }
        states.insert(transition.obligation_id.as_str(), transition.to.clone());
    }
    let failed: BTreeSet<_> = issues.iter().map(|item| item.record_id.as_str()).collect();
    let checked_records = records.len();
    TraceVerificationReport {
        verifier_version: VERSION.into(),
        source_trace_schema_version: trace.trace_schema_version,
        checked_records,
        verified_records: checked_records.saturating_sub(failed.len()),
        evidence: EvidenceSourceEnvelope {
            record_id: format!("trace-verification-{score_sha256}"),
            schema_version: 1,
            source: EvidenceSource::SymbolicallyVerified,
            verification_status: if issues.is_empty() {
                VerificationStatus::Verified
            } else {
                VerificationStatus::Discrepancy
            },
            producer: "Muse Analyst".into(),
            producer_version: VERSION.into(),
            artifact_sha256: score_sha256.into(),
            created_at_unix_ms,
            dependency_record_ids: records.into_iter().collect(),
            uncertainty: None,
            limitations: vec![
                "Verifier v1 checks references, lifecycle, and annotations; it does not yet re-derive harmonic cadence type.".into(),
            ],
        },
        issues,
    }
}

/// Reconcile composer assertions with the independent score-window scanner.
/// Structural validity and independent recovery are deliberately separate:
/// a valid assertion can remain unmatched, and the scanner can find material
/// the grammar owner did not explicitly report.
pub fn reconcile_motif_evidence(
    trace: &ComposerStructuralTrace,
    composition: &ListenCompositionBundle,
    verification: &TraceVerificationReport,
) -> MotifEvidenceReconciliation {
    const VERSION: &str = "motif-assertion-inference-reconciler-v1";
    let rejected_ids: BTreeSet<_> = verification
        .issues
        .iter()
        .filter(|issue| issue.code.starts_with("motif-") || issue.code.starts_with("literal-"))
        .map(|issue| issue.record_id.as_str())
        .collect();
    let mut used_inferred = BTreeSet::<usize>::new();
    let mut entries = Vec::new();

    for asserted in &trace.motif_occurrences {
        if rejected_ids.contains(asserted.occurrence_id.as_str()) {
            entries.push(MotifEvidenceReconciliationEntry {
                relationship: MotifEvidenceRelationship::Rejected,
                asserted_occurrence_id: Some(asserted.occurrence_id.clone()),
                inferred_occurrence_id: None,
                shared_score_event_count: 0,
                reason: "The independent trace verifier rejected this assertion.".into(),
            });
            continue;
        }
        let asserted_events: BTreeSet<_> = asserted.score_event_ids.iter().collect();
        let mut candidates: Vec<_> = composition
            .motif_occurrences
            .iter()
            .enumerate()
            .filter(|(index, _)| !used_inferred.contains(index))
            .map(|(index, inferred)| {
                let overlap = inferred
                    .source_note_ids
                    .iter()
                    .filter(|id| asserted_events.contains(id))
                    .count();
                (index, inferred, overlap)
            })
            .filter(|(_, _, overlap)| *overlap > 0)
            .collect();
        candidates.sort_by(|left, right| right.2.cmp(&left.2));
        let Some((index, inferred, overlap)) = candidates.first().copied() else {
            entries.push(MotifEvidenceReconciliationEntry {
                relationship: MotifEvidenceRelationship::AssertedNotRecovered,
                asserted_occurrence_id: Some(asserted.occurrence_id.clone()),
                inferred_occurrence_id: None,
                shared_score_event_count: 0,
                reason: "The assertion is structurally valid but the conservative scanner did not recover an overlapping occurrence.".into(),
            });
            continue;
        };
        let ambiguous = candidates
            .get(1)
            .is_some_and(|candidate| candidate.2 == overlap);
        if ambiguous {
            entries.push(MotifEvidenceReconciliationEntry {
                relationship: MotifEvidenceRelationship::Ambiguous,
                asserted_occurrence_id: Some(asserted.occurrence_id.clone()),
                inferred_occurrence_id: Some(inferred.id.clone()),
                shared_score_event_count: overlap,
                reason: "Multiple inferred occurrences overlap the assertion equally.".into(),
            });
        } else {
            used_inferred.insert(index);
            entries.push(MotifEvidenceReconciliationEntry {
                relationship: MotifEvidenceRelationship::AssertedAndMatched,
                asserted_occurrence_id: Some(asserted.occurrence_id.clone()),
                inferred_occurrence_id: Some(inferred.id.clone()),
                shared_score_event_count: overlap,
                reason: "Assertion and independent scan share score-event identities.".into(),
            });
        }
    }
    for (index, inferred) in composition.motif_occurrences.iter().enumerate() {
        if !used_inferred.contains(&index) {
            entries.push(MotifEvidenceReconciliationEntry {
                relationship: MotifEvidenceRelationship::InferredNotAsserted,
                asserted_occurrence_id: None,
                inferred_occurrence_id: Some(inferred.id.clone()),
                shared_score_event_count: 0,
                reason: "The independent scanner found a contract-valid occurrence not explicitly asserted by the grammar owner.".into(),
            });
        }
    }
    let count = |relationship: MotifEvidenceRelationship| {
        entries
            .iter()
            .filter(|entry| entry.relationship == relationship)
            .count()
    };
    MotifEvidenceReconciliation {
        reconciler_version: VERSION.into(),
        asserted_and_matched: count(MotifEvidenceRelationship::AssertedAndMatched),
        asserted_not_recovered: count(MotifEvidenceRelationship::AssertedNotRecovered),
        inferred_not_asserted: count(MotifEvidenceRelationship::InferredNotAsserted),
        ambiguous: count(MotifEvidenceRelationship::Ambiguous),
        rejected: count(MotifEvidenceRelationship::Rejected),
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::{
        CadenceTrace, ListenCompositionBundle, MeterPoint, MotifDefinition, MotifOccurrence,
        MotifOccurrenceTrace, MusicalTime, ObligationTransitionTrace, PerformanceVoiceSummary,
        PerformedNoteEvent, StructuralSpanTrace, SymbolicNoteEvent, TempoPoint,
    };
    use symthaea_music_theory::{GrammarFamily, HarmonicSyntax, PerformanceDialect, PhraseGrammar};

    fn time(beats: f64) -> MusicalTime {
        MusicalTime {
            tick: (beats * 480.0) as u64,
            beats,
            seconds: beats / 2.0,
        }
    }

    #[test]
    fn escalates_culturally_qualified_work_even_when_symbolically_valid() {
        let definition_basis = basis(EvidenceStatus::Observed, "test", &[]);
        let composition = ListenCompositionBundle {
            ticks_per_beat: 480,
            duration_ticks: 1920,
            duration_beats: 4.0,
            duration_seconds: 2.0,
            form_kind: "test".into(),
            tempo_map: vec![TempoPoint {
                at: time(0.0),
                bpm: 120.0,
            }],
            meter_map: vec![MeterPoint {
                at: time(0.0),
                numerator: 4,
                denominator: 4,
            }],
            sections: vec![],
            phrases: vec![],
            notes: vec![SymbolicNoteEvent {
                id: "n".into(),
                midi: 60,
                pitch_name: "C4".into(),
                onset: time(0.0),
                duration_ticks: 480,
                duration_beats: 1.0,
                duration_seconds: 0.5,
                velocity: 0.7,
                voice_role: "Melody".into(),
                emphasis: "normal".into(),
                section_intensity: 1.0,
            }],
            motif_definitions: vec![MotifDefinition {
                id: "m".into(),
                label: "m".into(),
                degrees: vec![1, 2],
                durations_beats: vec![1.0, 1.0],
                basis: definition_basis.clone(),
            }],
            motif_occurrences: vec![MotifOccurrence {
                id: "o".into(),
                motif_id: "m".into(),
                start: time(0.0),
                end: time(2.0),
                transformation: "transposed".into(),
                similarity: 0.9,
                source_note_ids: vec!["n".into()],
                basis: definition_basis,
            }],
            cadences: vec![],
            sonorities: vec![],
            orchestration: vec![],
            resonance: None,
        };
        let profile = symthaea_music_theory::GrammarProfile {
            family: GrammarFamily::RagaModalArc,
            phrase: PhraseGrammar::ModalUnfolding,
            harmony: HarmonicSyntax::DronePitchHierarchy,
            performance: PerformanceDialect::DroneElastic,
            supported_intent_axes: &[],
        };
        let provenance = GrammarProvenance {
            family: "raga_modal_arc".into(),
            phrase_grammar: "modal_unfolding".into(),
            harmonic_syntax: "drone_pitch_hierarchy".into(),
            performance_dialect: "drone_elastic".into(),
            plan_kind: "modal_arc".into(),
            obligations: vec![],
            supported_intent_axes: vec![],
            culturally_qualified: true,
            performance_features: None,
        };
        let plan = GrammarPlanEvidence::Compatibility {
            family: profile.family,
            form_available: true,
        };
        let report = analyze_piece(
            &composition,
            None,
            &provenance,
            Some(&plan),
            &MusicalIntent::default(),
        );
        assert_eq!(report.disposition, AnalystDisposition::HumanReview);
        assert!(
            report
                .escalations
                .iter()
                .any(|item| item.code == "cultural-review-required")
        );
    }

    fn assertion(id: &str) -> EvidenceSourceEnvelope {
        EvidenceSourceEnvelope {
            record_id: id.into(),
            schema_version: 1,
            source: EvidenceSource::ComposerAssertion,
            verification_status: VerificationStatus::Unchecked,
            producer: "fixture-composer".into(),
            producer_version: "1".into(),
            artifact_sha256: "score".into(),
            created_at_unix_ms: 0,
            dependency_record_ids: vec![],
            uncertainty: None,
            limitations: vec![],
        }
    }

    fn trace_fixture() -> (ListenCompositionBundle, ComposerStructuralTrace) {
        let note = |id: &str, beat: f64, emphasis: &str| SymbolicNoteEvent {
            id: id.into(),
            midi: 60 + beat as u8,
            pitch_name: "C4".into(),
            onset: time(beat),
            duration_ticks: 480,
            duration_beats: 1.0,
            duration_seconds: 0.5,
            velocity: 0.7,
            voice_role: "Melody".into(),
            emphasis: emphasis.into(),
            section_intensity: 1.0,
        };
        let composition = ListenCompositionBundle {
            ticks_per_beat: 480,
            duration_ticks: 1920,
            duration_beats: 4.0,
            duration_seconds: 2.0,
            form_kind: "fixture".into(),
            tempo_map: vec![TempoPoint {
                at: time(0.0),
                bpm: 120.0,
            }],
            meter_map: vec![MeterPoint {
                at: time(0.0),
                numerator: 4,
                denominator: 4,
            }],
            sections: vec![],
            phrases: vec![],
            notes: vec![
                note("n0", 0.0, "phrase-start"),
                note("n1", 1.0, "cadential"),
                note("n2", 2.0, "normal"),
            ],
            motif_definitions: vec![MotifDefinition {
                id: "motif-a".into(),
                label: "fixture motif".into(),
                degrees: vec![1, 2],
                durations_beats: vec![1.0, 1.0],
                basis: basis(EvidenceStatus::Observed, "fixture", &[]),
            }],
            motif_occurrences: vec![],
            cadences: vec![],
            sonorities: vec![],
            orchestration: vec![],
            resonance: None,
        };
        let trace = ComposerStructuralTrace {
            trace_schema_version: 1,
            structures: vec![StructuralSpanTrace {
                id: "phrase-a".into(),
                kind: "phrase".into(),
                parent_id: None,
                start_event_id: "n0".into(),
                end_event_id: "n2".into(),
                assertion: assertion("phrase-a"),
            }],
            motif_occurrences: vec![MotifOccurrenceTrace {
                motif_family_id: "motif-a".into(),
                motif_family_version: 1,
                occurrence_id: "motif-a-1".into(),
                score_event_ids: vec!["n0".into(), "n1".into()],
                voice_or_layer: "melody".into(),
                formal_region_id: "phrase-a".into(),
                transformation_chain: vec!["literal".into()],
                claimed_preserved_invariants: vec!["intervals".into()],
                changed_dimensions: vec![],
                literal_distance: 0.0,
                structural_distance: 0.0,
                role_binding: Some("opening".into()),
                originating_decision_id: "decision-1".into(),
                assertion: assertion("motif-a-1"),
            }],
            cadences: vec![CadenceTrace {
                cadence_id: "cadence-a".into(),
                proposed_type: "authentic".into(),
                grammar_owner: "period_sentence".into(),
                preparation_event_ids: vec!["n0".into()],
                arrival_event_id: "n1".into(),
                harmonic_evidence_event_ids: vec!["n0".into(), "n1".into()],
                melodic_evidence_event_ids: vec!["n1".into()],
                altered_downstream: false,
                fulfils_obligation_id: Some("obligation-a".into()),
                assertion: assertion("cadence-a"),
            }],
            obligation_transitions: vec![
                ObligationTransitionTrace {
                    obligation_id: "obligation-a".into(),
                    from: None,
                    to: ObligationState::Created,
                    score_event_ids: vec!["n0".into()],
                    responsible_pass: "period-planner".into(),
                    transformation: None,
                    assertion: assertion("obligation-a-created"),
                },
                ObligationTransitionTrace {
                    obligation_id: "obligation-a".into(),
                    from: Some(ObligationState::Created),
                    to: ObligationState::Fulfilled,
                    score_event_ids: vec!["n1".into()],
                    responsible_pass: "cadence-realizer".into(),
                    transformation: None,
                    assertion: assertion("obligation-a-fulfilled"),
                },
            ],
        };
        (composition, trace)
    }

    #[test]
    fn independent_trace_verifier_accepts_consistent_assertions() {
        let (composition, trace) = trace_fixture();
        let report = verify_composer_trace(&trace, &composition, "score", 0);
        assert!(report.issues.is_empty());
        assert_eq!(
            report.evidence.verification_status,
            VerificationStatus::Verified
        );
    }

    #[test]
    fn motif_reconciliation_does_not_conflate_validity_with_recovery() {
        let (mut composition, trace) = trace_fixture();
        let verification = verify_composer_trace(&trace, &composition, "score", 0);
        let unmatched = reconcile_motif_evidence(&trace, &composition, &verification);
        assert_eq!(unmatched.asserted_not_recovered, 1);
        assert_eq!(unmatched.asserted_and_matched, 0);

        composition.motif_occurrences.push(MotifOccurrence {
            id: "scan-1".into(),
            motif_id: "motif-a".into(),
            start: time(0.0),
            end: time(2.0),
            transformation: "literal".into(),
            similarity: 1.0,
            source_note_ids: vec!["n0".into(), "n1".into()],
            basis: basis(EvidenceStatus::Inferred, "fixture-scan", &[]),
        });
        let matched = reconcile_motif_evidence(&trace, &composition, &verification);
        assert_eq!(matched.asserted_and_matched, 1);
        assert_eq!(matched.asserted_not_recovered, 0);
        assert_eq!(matched.inferred_not_asserted, 0);
    }

    #[test]
    fn adversarial_trace_corruptions_are_detected() {
        let (composition, trace) = trace_fixture();
        let corruptions: Vec<(&str, Box<dyn Fn(&mut ComposerStructuralTrace)>)> = vec![
            (
                "motif-event-missing",
                Box::new(|trace| trace.motif_occurrences[0].score_event_ids[0] = "absent".into()),
            ),
            (
                "motif-family-mismatch",
                Box::new(|trace| {
                    trace.motif_occurrences[0].motif_family_id = "undeclared-family".into()
                }),
            ),
            (
                "literal-distance-discrepancy",
                Box::new(|trace| trace.motif_occurrences[0].literal_distance = 0.8),
            ),
            (
                "cadence-arrival-not-marked",
                Box::new(|trace| trace.cadences[0].arrival_event_id = "n2".into()),
            ),
            (
                "obligation-evidence-missing",
                Box::new(|trace| trace.obligation_transitions[1].score_event_ids.clear()),
            ),
            (
                "structural-span-event-missing",
                Box::new(|trace| trace.structures[0].end_event_id = "absent".into()),
            ),
            (
                "invalid-assertion-source",
                Box::new(|trace| {
                    trace.structures[0].assertion.source = EvidenceSource::ModelPrediction
                }),
            ),
            (
                "structural-span-reversed",
                Box::new(|trace| {
                    trace.structures[0].start_event_id = "n2".into();
                    trace.structures[0].end_event_id = "n0".into();
                }),
            ),
            (
                "structural-parent-missing",
                Box::new(|trace| trace.structures[0].parent_id = Some("absent".into())),
            ),
            (
                "motif-region-missing",
                Box::new(|trace| trace.motif_occurrences[0].formal_region_id = "absent".into()),
            ),
            (
                "motif-event-order-discrepancy",
                Box::new(|trace| trace.motif_occurrences[0].score_event_ids.reverse()),
            ),
            (
                "cadence-evidence-event-missing",
                Box::new(|trace| trace.cadences[0].preparation_event_ids = vec!["absent".into()]),
            ),
            (
                "cadence-obligation-missing",
                Box::new(|trace| trace.cadences[0].fulfils_obligation_id = Some("absent".into())),
            ),
            (
                "obligation-lifecycle-discontinuity",
                Box::new(|trace| trace.obligation_transitions[1].from = None),
            ),
            (
                "obligation-pass-missing",
                Box::new(|trace| trace.obligation_transitions[1].responsible_pass.clear()),
            ),
        ];
        for (expected, corrupt) in corruptions {
            let mut candidate = trace.clone();
            corrupt(&mut candidate);
            let report = verify_composer_trace(&candidate, &composition, "score", 0);
            assert!(
                report.issues.iter().any(|issue| issue.code == expected),
                "corruption {expected} escaped verification: {:?}",
                report.issues
            );
            assert_eq!(
                report.evidence.verification_status,
                VerificationStatus::Discrepancy
            );
        }
    }

    #[test]
    fn renderer_note_loss_is_detected_by_source_ids() {
        let (composition, _) = trace_fixture();
        let performance = ListenPerformanceBundle {
            duration_seconds: 2.0,
            mapping_method: "fixture".into(),
            voices: vec![PerformanceVoiceSummary {
                id: "melody".into(),
                name: "Melody".into(),
                instrument: "piano".into(),
                note_count: 1,
            }],
            notes: vec![PerformedNoteEvent {
                id: "p0".into(),
                voice_id: "melody".into(),
                source_note_id: Some("n0".into()),
                start_seconds: 0.0,
                duration_seconds: 0.5,
                frequency_hz: 261.6,
                velocity: 0.7,
                onset_deviation_seconds: Some(0.0),
                duration_deviation_seconds: Some(0.0),
            }],
        };
        let provenance = GrammarProvenance {
            family: "period_sentence".into(),
            phrase_grammar: "period".into(),
            harmonic_syntax: "functional".into(),
            performance_dialect: "classical_rubato".into(),
            plan_kind: "compatibility".into(),
            obligations: vec![],
            supported_intent_axes: vec![],
            culturally_qualified: false,
            performance_features: None,
        };
        let report = analyze_piece(
            &composition,
            Some(&performance),
            &provenance,
            None,
            &MusicalIntent::default(),
        );
        let renderer = report
            .checks
            .iter()
            .find(|check| check.code == "renderer-source-mapping")
            .unwrap();
        assert_eq!(renderer.status, AnalystCheckStatus::Fail);
        assert!(renderer.observed.contains("2 missing"));
    }

    fn wav_fixture(samples: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for sample in samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn audio_integrity_corruption_matrix_detects_known_failures() {
        let clean: Vec<_> = (0..48_000)
            .map(|index| (index as f32 * std::f32::consts::TAU * 440.0 / 48_000.0).sin() * 0.2)
            .collect();
        let report = analyze_audio_integrity(&wav_fixture(&clean), "clean", 0);
        assert!(report.issues.is_empty(), "{:?}", report.issues);

        let silent = analyze_audio_integrity(&wav_fixture(&vec![0.0; 2048]), "silent", 0);
        assert!(
            silent
                .issues
                .iter()
                .any(|issue| issue == "audio-effectively-silent")
        );

        let mut clipped = clean.clone();
        clipped[100] = 1.0;
        let clipped = analyze_audio_integrity(&wav_fixture(&clipped), "clipped", 0);
        assert!(
            clipped
                .issues
                .iter()
                .any(|issue| issue == "clipping-detected")
        );

        let mut impulses = vec![0.0; 48_000];
        for index in (1000..20_000).step_by(1000) {
            impulses[index] = 0.9;
        }
        let impulses = analyze_audio_integrity(&wav_fixture(&impulses), "impulses", 0);
        assert!(
            impulses
                .issues
                .iter()
                .any(|issue| issue == "broadband-impulse-outliers"),
            "{:?}",
            impulses.issues
        );
    }

    #[test]
    fn all_flagship_composer_traces_survive_independent_symbolic_verification() {
        use symthaea_music_theory::{Emphasis, Style, VoiceRole, compose_with_grammar_plan};

        for style in [
            Style::Classical,
            Style::AfroCuban,
            Style::Minimalism,
            Style::HindustaniInspired,
        ] {
            let intent = MusicalIntent {
                bars: 8,
                seed: 11,
                ..MusicalIntent::default()
            };
            let spec = style.spec();
            let realized = compose_with_grammar_plan(style.grammar_profile(), &intent, &spec);
            let mut notes = Vec::new();
            for role in [
                VoiceRole::Bass,
                VoiceRole::Harmony,
                VoiceRole::CounterMelody,
                VoiceRole::Melody,
            ] {
                let role_slug = match role {
                    VoiceRole::Melody => "melody",
                    VoiceRole::Harmony => "harmony",
                    VoiceRole::Bass => "bass",
                    VoiceRole::CounterMelody => "counter",
                };
                let role_label = format!("{role:?}");
                for (index, note) in realized.score.voice(role).into_iter().enumerate() {
                    notes.push(SymbolicNoteEvent {
                        id: format!("score-{role_slug}-{index}"),
                        midi: note.pitch.midi(),
                        pitch_name: format!("{:?}", note.pitch),
                        onset: time(note.onset.beats()),
                        duration_ticks: (note.duration.beats() * 480.0) as u64,
                        duration_beats: note.duration.beats(),
                        duration_seconds: note.duration.seconds(realized.score.tempo_bpm as f64),
                        velocity: note.velocity,
                        voice_role: role_label.clone(),
                        emphasis: match note.emphasis {
                            Emphasis::Normal => "normal",
                            Emphasis::Climax => "climax",
                            Emphasis::Cadential => "cadential",
                            Emphasis::PhraseStart => "phrase-start",
                        }
                        .into(),
                        section_intensity: note.section_intensity,
                    });
                }
            }
            let motif = spec.motif(intent.arousal, intent.seed);
            let pitched: Vec<_> = motif
                .notes
                .iter()
                .filter_map(|note| note.degree.map(|degree| (degree, note.duration.beats())))
                .collect();
            let composition = ListenCompositionBundle {
                ticks_per_beat: 480,
                duration_ticks: (realized.score.total_beats.beats() * 480.0) as u64,
                duration_beats: realized.score.total_beats.beats(),
                duration_seconds: realized.score.seconds(),
                form_kind: "fixture".into(),
                tempo_map: vec![TempoPoint {
                    at: time(0.0),
                    bpm: realized.score.tempo_bpm,
                }],
                meter_map: vec![MeterPoint {
                    at: time(0.0),
                    numerator: realized.score.meter,
                    denominator: 4,
                }],
                sections: vec![],
                phrases: vec![],
                notes,
                motif_definitions: vec![MotifDefinition {
                    id: "motif-primary".into(),
                    label: "primary".into(),
                    degrees: pitched.iter().map(|item| item.0).collect(),
                    durations_beats: pitched.iter().map(|item| item.1).collect(),
                    basis: basis(EvidenceStatus::Observed, "fixture", &[]),
                }],
                motif_occurrences: vec![],
                cadences: vec![],
                sonorities: vec![],
                orchestration: vec![],
                resonance: None,
            };
            let public_trace = composer_trace_from_theory(&realized.trace, "score", 0, None);
            let verification = verify_composer_trace(&public_trace, &composition, "score", 0);
            assert!(
                verification.issues.is_empty(),
                "{style:?}: {:?}",
                verification.issues
            );
        }
    }
}
