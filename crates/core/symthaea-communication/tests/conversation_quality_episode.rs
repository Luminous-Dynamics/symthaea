// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/conversation_quality_episode.rs"]
mod conversation_quality_episode;

use conversation_quality_episode::*;

fn c(ch: char) -> String {
    format!("blake3:{}", ch.to_string().repeat(64))
}

fn component(
    kind: ConversationComponentKindV1,
    sha: &str,
    commitment_char: char,
    evidence_state: ComponentEvidenceStateV1,
) -> ComponentReceiptRefV1 {
    ComponentReceiptRefV1::new(
        kind,
        sha,
        format!("schema.{kind:?}.v1").to_ascii_lowercase(),
        c(commitment_char),
        evidence_state,
    )
    .unwrap()
}

fn manifest_with_states(
    voice: ComponentEvidenceStateV1,
    correction: ComponentEvidenceStateV1,
    persona: ComponentEvidenceStateV1,
) -> ConversationQualityEpisodeManifestV1 {
    ConversationQualityEpisodeManifestV1::new(
        "episode-001",
        "v1",
        "fixture:conversation-001",
        "evaluators:set-001",
        vec![
            component(
                ConversationComponentKindV1::FullDuplexVoice,
                "1111111111111111111111111111111111111111",
                '1',
                voice,
            ),
            component(
                ConversationComponentKindV1::Correction,
                "2222222222222222222222222222222222222222",
                '2',
                correction,
            ),
            component(
                ConversationComponentKindV1::PersonaContinuity,
                "3333333333333333333333333333333333333333",
                '3',
                persona,
            ),
        ],
    )
    .unwrap()
}

fn latency(
    value: Option<u64>,
    budget: LatencyBudgetFindingV1,
) -> LatencyFindingV1 {
    LatencyFindingV1::new(value, budget).unwrap()
}

fn voice(state: ComponentEvidenceStateV1) -> VoiceEpisodeScorecardV1 {
    VoiceEpisodeScorecardV1 {
        evidence_state: state,
        interruption_semantic: SemanticFindingV1::Satisfied,
        interruption_latency: latency(Some(80_000_000), LatencyBudgetFindingV1::WithinBudget),
        stop_semantic: SemanticFindingV1::Satisfied,
        stop_latency: latency(Some(40_000_000), LatencyBudgetFindingV1::WithinBudget),
        slowdown_semantic: SemanticFindingV1::Satisfied,
        slowdown_latency: latency(Some(100_000_000), LatencyBudgetFindingV1::WithinBudget),
        response_latency: latency(Some(250_000_000), LatencyBudgetFindingV1::NotEvaluated),
        backchannel_semantic: SemanticFindingV1::Satisfied,
        incomplete_trace_count: 0,
        runtime_rejection_count: 0,
    }
}

fn correction(state: ComponentEvidenceStateV1) -> CorrectionEpisodeScorecardV1 {
    CorrectionEpisodeScorecardV1 {
        evidence_state: state,
        first_compliant_turn_delta: Some(1),
        first_compliant_latency_ns: Some(500_000_000),
        recurrence_before_adaptation: 0,
        recurrence_after_adaptation: 0,
        unrelated_context_spillover: 0,
        ambiguous_applicable_observations: 0,
        applicable_observation_count: 4,
    }
}

fn persona(state: ComponentEvidenceStateV1) -> PersonaEpisodeScorecardV1 {
    PersonaEpisodeScorecardV1 {
        evidence_state: state,
        critical_violation_count: 0,
        noncritical_violation_count: 0,
        global_band_violation_count: 0,
        context_band_violation_count: 0,
        material_disagreement_group_count: 0,
        unobserved_required_facet_count: 0,
        observed_facet_count: 4,
        profile_facet_count: 4,
        observed_context_count: 3,
    }
}

#[test]
fn complete_evidence_does_not_mean_zero_failures() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let mut voice = voice(ComponentEvidenceStateV1::Established);
    voice.stop_semantic = SemanticFindingV1::Violated;
    voice.stop_latency = latency(Some(900_000_000), LatencyBudgetFindingV1::Exceeded);

    let receipt = ConversationQualityEpisodeReceiptV1::new(
        &manifest,
        voice,
        correction(ComponentEvidenceStateV1::Established),
        persona(ComponentEvidenceStateV1::Established),
    )
    .unwrap();

    assert_eq!(receipt.evidence_coverage, EpisodeEvidenceCoverageV1::Complete);
    assert_eq!(receipt.voice.stop_semantic, SemanticFindingV1::Violated);
    assert_eq!(
        receipt.voice.stop_latency.budget_finding,
        LatencyBudgetFindingV1::Exceeded
    );
}

#[test]
fn recurrence_after_fast_adaptation_remains_visible() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let mut correction = correction(ComponentEvidenceStateV1::Established);
    correction.first_compliant_turn_delta = Some(1);
    correction.recurrence_after_adaptation = 2;

    let receipt = ConversationQualityEpisodeReceiptV1::new(
        &manifest,
        voice(ComponentEvidenceStateV1::Established),
        correction,
        persona(ComponentEvidenceStateV1::Established),
    )
    .unwrap();
    assert_eq!(receipt.correction.first_compliant_turn_delta, Some(1));
    assert_eq!(receipt.correction.recurrence_after_adaptation, 2);
    assert_eq!(receipt.evidence_coverage, EpisodeEvidenceCoverageV1::Complete);
}

#[test]
fn missing_component_evidence_makes_episode_partial_not_failed() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::NotEstablished,
        ComponentEvidenceStateV1::Established,
    );
    let mut correction = correction(ComponentEvidenceStateV1::NotEstablished);
    correction.first_compliant_turn_delta = None;
    correction.first_compliant_latency_ns = None;
    correction.applicable_observation_count = 0;

    let receipt = ConversationQualityEpisodeReceiptV1::new(
        &manifest,
        voice(ComponentEvidenceStateV1::Established),
        correction,
        persona(ComponentEvidenceStateV1::Established),
    )
    .unwrap();
    assert_eq!(receipt.evidence_coverage, EpisodeEvidenceCoverageV1::Partial);
}

#[test]
fn all_missing_evidence_is_not_established() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::NotEstablished,
        ComponentEvidenceStateV1::NotEstablished,
        ComponentEvidenceStateV1::NotEstablished,
    );
    let mut v = voice(ComponentEvidenceStateV1::NotEstablished);
    v.interruption_semantic = SemanticFindingV1::NotEstablished;
    v.stop_semantic = SemanticFindingV1::NotEstablished;
    v.slowdown_semantic = SemanticFindingV1::NotEstablished;
    v.backchannel_semantic = SemanticFindingV1::NotEstablished;
    v.interruption_latency = latency(None, LatencyBudgetFindingV1::NotEstablished);
    v.stop_latency = latency(None, LatencyBudgetFindingV1::NotEstablished);
    v.slowdown_latency = latency(None, LatencyBudgetFindingV1::NotEstablished);
    v.response_latency = latency(None, LatencyBudgetFindingV1::NotEstablished);
    let mut c = correction(ComponentEvidenceStateV1::NotEstablished);
    c.first_compliant_turn_delta = None;
    c.first_compliant_latency_ns = None;
    c.applicable_observation_count = 0;
    let mut p = persona(ComponentEvidenceStateV1::NotEstablished);
    p.observed_facet_count = 0;
    p.observed_context_count = 0;
    p.unobserved_required_facet_count = p.profile_facet_count;

    let receipt = ConversationQualityEpisodeReceiptV1::new(&manifest, v, c, p).unwrap();
    assert_eq!(
        receipt.evidence_coverage,
        EpisodeEvidenceCoverageV1::NotEstablished
    );
}

#[test]
fn uncertainty_is_not_averaged_away() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Uncertain,
    );
    let receipt = ConversationQualityEpisodeReceiptV1::new(
        &manifest,
        voice(ComponentEvidenceStateV1::Established),
        correction(ComponentEvidenceStateV1::Established),
        persona(ComponentEvidenceStateV1::Uncertain),
    )
    .unwrap();
    assert_eq!(receipt.evidence_coverage, EpisodeEvidenceCoverageV1::Uncertain);
}

#[test]
fn invalid_component_evidence_dominates_coverage_only() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Invalid,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let receipt = ConversationQualityEpisodeReceiptV1::new(
        &manifest,
        voice(ComponentEvidenceStateV1::Invalid),
        correction(ComponentEvidenceStateV1::Established),
        persona(ComponentEvidenceStateV1::Established),
    )
    .unwrap();
    assert_eq!(receipt.evidence_coverage, EpisodeEvidenceCoverageV1::Invalid);
}

#[test]
fn manifest_identity_is_component_order_invariant() {
    let a = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let b = ConversationQualityEpisodeManifestV1::new(
        "episode-001",
        "v1",
        "fixture:conversation-001",
        "evaluators:set-001",
        vec![
            component(
                ConversationComponentKindV1::PersonaContinuity,
                "3333333333333333333333333333333333333333",
                '3',
                ComponentEvidenceStateV1::Established,
            ),
            component(
                ConversationComponentKindV1::FullDuplexVoice,
                "1111111111111111111111111111111111111111",
                '1',
                ComponentEvidenceStateV1::Established,
            ),
            component(
                ConversationComponentKindV1::Correction,
                "2222222222222222222222222222222222222222",
                '2',
                ComponentEvidenceStateV1::Established,
            ),
        ],
    )
    .unwrap();
    assert_eq!(a.commitment, b.commitment);
}

#[test]
fn component_subject_substitution_changes_manifest_identity() {
    let a = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let b = ConversationQualityEpisodeManifestV1::new(
        "episode-001",
        "v1",
        "fixture:conversation-001",
        "evaluators:set-001",
        vec![
            component(
                ConversationComponentKindV1::FullDuplexVoice,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                '1',
                ComponentEvidenceStateV1::Established,
            ),
            component(
                ConversationComponentKindV1::Correction,
                "2222222222222222222222222222222222222222",
                '2',
                ComponentEvidenceStateV1::Established,
            ),
            component(
                ConversationComponentKindV1::PersonaContinuity,
                "3333333333333333333333333333333333333333",
                '3',
                ComponentEvidenceStateV1::Established,
            ),
        ],
    )
    .unwrap();
    assert_ne!(a.commitment, b.commitment);
}

#[test]
fn duplicate_or_missing_components_fail_closed() {
    let voice = component(
        ConversationComponentKindV1::FullDuplexVoice,
        "1111111111111111111111111111111111111111",
        '1',
        ComponentEvidenceStateV1::Established,
    );
    let duplicate = voice.clone();
    let correction = component(
        ConversationComponentKindV1::Correction,
        "2222222222222222222222222222222222222222",
        '2',
        ComponentEvidenceStateV1::Established,
    );
    assert_eq!(
        ConversationQualityEpisodeManifestV1::new(
            "episode",
            "v1",
            "fixture:x",
            "eval:x",
            vec![voice.clone(), duplicate, correction.clone()],
        )
        .unwrap_err(),
        ConversationQualityEpisodeErrorV1::DuplicateComponentKind
    );
    assert_eq!(
        ConversationQualityEpisodeManifestV1::new(
            "episode",
            "v1",
            "fixture:x",
            "eval:x",
            vec![voice, correction],
        )
        .unwrap_err(),
        ConversationQualityEpisodeErrorV1::MissingRequiredComponent
    );
}

#[test]
fn missing_latency_cannot_claim_within_budget() {
    assert_eq!(
        LatencyFindingV1::new(None, LatencyBudgetFindingV1::WithinBudget).unwrap_err(),
        ConversationQualityEpisodeErrorV1::InconsistentLatencyEvidence
    );
    assert!(LatencyFindingV1::new(None, LatencyBudgetFindingV1::NotEstablished).is_ok());
}

#[test]
fn evidence_state_must_match_frozen_component_reference() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    assert_eq!(
        ConversationQualityEpisodeReceiptV1::new(
            &manifest,
            voice(ComponentEvidenceStateV1::Partial),
            correction(ComponentEvidenceStateV1::Established),
            persona(ComponentEvidenceStateV1::Established),
        )
        .unwrap_err(),
        ConversationQualityEpisodeErrorV1::EvidenceStateMismatch
    );
}

#[test]
fn tampered_manifest_and_receipt_commitments_are_rejected() {
    let mut manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    manifest.commitment = c('f');
    assert_eq!(
        ConversationQualityEpisodeReceiptV1::new(
            &manifest,
            voice(ComponentEvidenceStateV1::Established),
            correction(ComponentEvidenceStateV1::Established),
            persona(ComponentEvidenceStateV1::Established),
        )
        .unwrap_err(),
        ConversationQualityEpisodeErrorV1::ManifestCommitmentMismatch
    );

    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let mut receipt = ConversationQualityEpisodeReceiptV1::new(
        &manifest,
        voice(ComponentEvidenceStateV1::Established),
        correction(ComponentEvidenceStateV1::Established),
        persona(ComponentEvidenceStateV1::Established),
    )
    .unwrap();
    receipt.receipt_commitment = c('e');
    assert_eq!(
        receipt.validate(&manifest).unwrap_err(),
        ConversationQualityEpisodeErrorV1::ReceiptCommitmentMismatch
    );
}

#[test]
fn correction_or_persona_consistency_is_checked() {
    let manifest = manifest_with_states(
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
        ComponentEvidenceStateV1::Established,
    );
    let mut cscore = correction(ComponentEvidenceStateV1::Established);
    cscore.applicable_observation_count = 0;
    assert_eq!(
        ConversationQualityEpisodeReceiptV1::new(
            &manifest,
            voice(ComponentEvidenceStateV1::Established),
            cscore,
            persona(ComponentEvidenceStateV1::Established),
        )
        .unwrap_err(),
        ConversationQualityEpisodeErrorV1::InconsistentCorrectionEvidence
    );

    let mut pscore = persona(ComponentEvidenceStateV1::Established);
    pscore.observed_facet_count = 5;
    pscore.profile_facet_count = 4;
    assert_eq!(
        ConversationQualityEpisodeReceiptV1::new(
            &manifest,
            voice(ComponentEvidenceStateV1::Established),
            correction(ComponentEvidenceStateV1::Established),
            pscore,
        )
        .unwrap_err(),
        ConversationQualityEpisodeErrorV1::InconsistentPersonaCoverage
    );
}
