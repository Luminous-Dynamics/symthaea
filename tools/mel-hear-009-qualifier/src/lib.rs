//! Matched-condition validator for MEL-HEAR-009.
//!
//! This module consumes references to existing identity, annotation,
//! materialization and budget evidence. It does not reimplement those owners.
//! Its purpose is to validate the matched experimental design and expose
//! per-condition outcomes without making a population-inference claim.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const DIGEST_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionKind {
    OracleSymbolic,
    AudioDerivedSymbolic,
    RawAudio,
    AudioPlusPredictedStructure,
    NoEvidenceControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetProperty {
    ChordQuality,
    RootPitchClass,
    Voicing,
    RelativeIntervalSequence,
    MelodicContour,
    RhythmicRatio,
    PulseRate,
    InstrumentFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceDisposition {
    Same,
    Different,
    Ambiguous,
    NotEstimable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Prediction {
    Same,
    Different,
    Ambiguous,
    NotEstimable,
    Abstain,
    Unparseable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStage {
    LocalContextVerified,
    ClientRequestBound,
    ProviderInputAttested,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Panel {
    pub schema_version: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub panel_id: String,
    pub split: String,
    pub item_id: String,
    pub pair_id: String,
    pub triad_sha256: String,
    pub source_work_lineage_sha256: String,
    pub performer_cohort_manifest_sha256: String,
    pub target_property: TargetProperty,
    pub question_sha256: String,
    pub reference_disposition: ReferenceDisposition,
    pub reference_manifest_sha256: String,
    pub annotation_manifest_sha256: String,
    pub audio_materialization_report_sha256: String,
    pub audio_evidence_manifest_sha256: String,
    pub sampling_design_manifest_sha256: String,
    pub matched_budget_profile_sha256: String,
    pub conditions: Vec<Condition>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub kind: ConditionKind,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub system_prompt_sha256: String,
    pub adapter_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub reasoner_model_identity_sha256: Option<String>,
    pub budget_profile_sha256: String,
    pub execution_receipt_sha256: String,
    pub input_context_sha256: String,
    pub evidence_stage: EvidenceStage,
    pub audio_evidence_manifest_sha256: Option<String>,
    pub oracle_symbolic_reference_sha256: Option<String>,
    pub predicted_symbolic_representation_sha256: Option<String>,
    pub transcription_report_sha256: Option<String>,
    pub alignment_report_sha256: Option<String>,
    pub reference_visibility_audit_sha256: String,
    pub reference_label_exposed: bool,
    pub prediction: Prediction,
    pub prediction_confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConditionMetrics {
    pub kind: ConditionKind,
    pub scorable: bool,
    pub correct: usize,
    pub incorrect: usize,
    pub answered: usize,
    pub abstained_or_ambiguous: usize,
    pub unparseable: usize,
    pub accuracy: Option<f64>,
    pub coverage: Option<f64>,
    pub prediction: Prediction,
    pub prediction_confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PerceptionReasoningDiagnostics {
    pub oracle_correct_audio_derived_incorrect: bool,
    pub oracle_incorrect_audio_derived_correct: bool,
    pub raw_audio_correct_fusion_incorrect: bool,
    pub raw_audio_incorrect_fusion_correct: bool,
    pub no_evidence_control_correct: Option<bool>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AuditReport {
    pub schema: String,
    pub panel_id: String,
    pub split: String,
    pub item_id: String,
    pub target_property: TargetProperty,
    pub reference_scorable: bool,
    pub condition_count: usize,
    pub shared_budget_profile_sha256: String,
    pub shared_audio_evidence_manifest_sha256: String,
    pub reasoner_identity_matched: bool,
    pub raw_audio_model_identity_matched: bool,
    pub conditions: Vec<ConditionMetrics>,
    pub diagnostics: PerceptionReasoningDiagnostics,
}

fn is_sha256(value: &str) -> bool {
    value.len() == DIGEST_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, name: &str) -> Result<(), String> {
    if is_sha256(value) {
        Ok(())
    } else {
        Err(format!("{name} must be a lowercase SHA-256 digest"))
    }
}

fn validate_panel_header(panel: &Panel) -> Result<(), String> {
    if panel.schema_version != "mel-hear-009-perception-reasoning-panel-v0" {
        return Err("unsupported panel schema version".to_owned());
    }
    if panel.benchmark_id != "MEL-PERCEPTION-REASONING-GAP" {
        return Err("unexpected benchmark identifier".to_owned());
    }
    if panel.benchmark_version.trim().is_empty()
        || panel.panel_id.trim().is_empty()
        || panel.item_id.trim().is_empty()
        || panel.pair_id.trim().is_empty()
    {
        return Err("benchmark version, panel, item and pair identifiers are required".to_owned());
    }
    if !matches!(panel.split.as_str(), "train" | "development" | "test") {
        return Err("split must be train, development or test".to_owned());
    }

    for (value, name) in [
        (&panel.triad_sha256, "triad_sha256"),
        (
            &panel.source_work_lineage_sha256,
            "source_work_lineage_sha256",
        ),
        (
            &panel.performer_cohort_manifest_sha256,
            "performer_cohort_manifest_sha256",
        ),
        (&panel.question_sha256, "question_sha256"),
        (
            &panel.reference_manifest_sha256,
            "reference_manifest_sha256",
        ),
        (
            &panel.annotation_manifest_sha256,
            "annotation_manifest_sha256",
        ),
        (
            &panel.audio_materialization_report_sha256,
            "audio_materialization_report_sha256",
        ),
        (
            &panel.audio_evidence_manifest_sha256,
            "audio_evidence_manifest_sha256",
        ),
        (
            &panel.sampling_design_manifest_sha256,
            "sampling_design_manifest_sha256",
        ),
        (
            &panel.matched_budget_profile_sha256,
            "matched_budget_profile_sha256",
        ),
    ] {
        require_sha256(value, name)?;
    }
    Ok(())
}

fn validate_condition_common(condition: &Condition, budget_sha: &str) -> Result<(), String> {
    for (value, name) in [
        (&condition.model_identity_sha256, "model_identity_sha256"),
        (
            &condition.runtime_identity_sha256,
            "runtime_identity_sha256",
        ),
        (&condition.system_prompt_sha256, "system_prompt_sha256"),
        (
            &condition.adapter_identity_sha256,
            "adapter_identity_sha256",
        ),
        (&condition.decoding_config_sha256, "decoding_config_sha256"),
        (&condition.question_sha256, "condition.question_sha256"),
        (
            &condition.budget_profile_sha256,
            "condition.budget_profile_sha256",
        ),
        (
            &condition.execution_receipt_sha256,
            "execution_receipt_sha256",
        ),
        (&condition.input_context_sha256, "input_context_sha256"),
        (
            &condition.reference_visibility_audit_sha256,
            "reference_visibility_audit_sha256",
        ),
    ] {
        require_sha256(value, name)?;
    }
    if condition.budget_profile_sha256 != budget_sha {
        return Err(
            "condition budget profile differs from the panel's frozen matched profile".to_owned(),
        );
    }
    if condition.reference_label_exposed {
        return Err("reference labels must not be exposed to any model condition".to_owned());
    }
    if condition.evidence_stage != EvidenceStage::LocalContextVerified {
        return Err(
            "v0 accepts only local-context evidence; client/provider attestations require the trusted stage verifier"
                .to_owned(),
        );
    }
    if let Some(confidence) = condition.prediction_confidence {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err("prediction_confidence must be finite and between zero and one".to_owned());
        }
    }
    for (value, name) in [
        (
            condition.reasoner_model_identity_sha256.as_ref(),
            "reasoner_model_identity_sha256",
        ),
        (
            condition.audio_evidence_manifest_sha256.as_ref(),
            "audio_evidence_manifest_sha256",
        ),
        (
            condition.oracle_symbolic_reference_sha256.as_ref(),
            "oracle_symbolic_reference_sha256",
        ),
        (
            condition.predicted_symbolic_representation_sha256.as_ref(),
            "predicted_symbolic_representation_sha256",
        ),
        (
            condition.transcription_report_sha256.as_ref(),
            "transcription_report_sha256",
        ),
        (
            condition.alignment_report_sha256.as_ref(),
            "alignment_report_sha256",
        ),
    ] {
        if let Some(value) = value {
            require_sha256(value, name)?;
        }
    }
    Ok(())
}

fn require_absent(condition: &Condition, kind: ConditionKind) -> Result<(), String> {
    if condition.audio_evidence_manifest_sha256.is_some()
        || condition.oracle_symbolic_reference_sha256.is_some()
        || condition.predicted_symbolic_representation_sha256.is_some()
        || condition.transcription_report_sha256.is_some()
        || condition.alignment_report_sha256.is_some()
    {
        return Err(format!(
            "{kind:?} condition contains evidence it is not permitted to receive"
        ));
    }
    Ok(())
}

fn validate_condition_kind(
    condition: &Condition,
    panel_audio_manifest_sha: &str,
) -> Result<(), String> {
    match condition.kind {
        ConditionKind::OracleSymbolic => {
            if condition.oracle_symbolic_reference_sha256.is_none()
                || condition.audio_evidence_manifest_sha256.is_some()
                || condition.predicted_symbolic_representation_sha256.is_some()
                || condition.transcription_report_sha256.is_some()
                || condition.alignment_report_sha256.is_some()
                || condition.reasoner_model_identity_sha256.is_none()
            {
                return Err(
                    "oracle_symbolic requires only an oracle structure reference and a fixed reasoning head"
                        .to_owned(),
                );
            }
        }
        ConditionKind::AudioDerivedSymbolic => {
            if condition.audio_evidence_manifest_sha256.as_deref() != Some(panel_audio_manifest_sha)
                || condition.oracle_symbolic_reference_sha256.is_some()
                || condition.predicted_symbolic_representation_sha256.is_none()
                || condition.transcription_report_sha256.is_none()
                || condition.alignment_report_sha256.is_none()
                || condition.reasoner_model_identity_sha256.is_none()
            {
                return Err(
                    "audio_derived_symbolic requires exact audio evidence, predicted structure, transcription/alignment receipts and a reasoning head"
                        .to_owned(),
                );
            }
        }
        ConditionKind::RawAudio => {
            if condition.audio_evidence_manifest_sha256.as_deref() != Some(panel_audio_manifest_sha)
                || condition.oracle_symbolic_reference_sha256.is_some()
                || condition.predicted_symbolic_representation_sha256.is_some()
                || condition.transcription_report_sha256.is_some()
                || condition.alignment_report_sha256.is_some()
                || condition.reasoner_model_identity_sha256.is_some()
            {
                return Err(
                    "raw_audio requires the exact audio evidence and no symbolic representation"
                        .to_owned(),
                );
            }
        }
        ConditionKind::AudioPlusPredictedStructure => {
            if condition.audio_evidence_manifest_sha256.as_deref() != Some(panel_audio_manifest_sha)
                || condition.oracle_symbolic_reference_sha256.is_some()
                || condition.predicted_symbolic_representation_sha256.is_none()
                || condition.transcription_report_sha256.is_none()
                || condition.alignment_report_sha256.is_none()
                || condition.reasoner_model_identity_sha256.is_some()
            {
                return Err(
                    "audio_plus_predicted_structure requires exact audio and model-derived structure, never oracle structure"
                        .to_owned(),
                );
            }
        }
        ConditionKind::NoEvidenceControl => require_absent(condition, condition.kind)?,
    }
    Ok(())
}

fn score_condition(condition: &Condition, reference: ReferenceDisposition) -> ConditionMetrics {
    let scorable = matches!(
        reference,
        ReferenceDisposition::Same | ReferenceDisposition::Different
    );
    let answered = if matches!(
        condition.prediction,
        Prediction::Same | Prediction::Different
    ) {
        1
    } else {
        0
    };
    let abstained_or_ambiguous = if matches!(
        condition.prediction,
        Prediction::Ambiguous | Prediction::NotEstimable | Prediction::Abstain
    ) {
        1
    } else {
        0
    };
    let unparseable = if condition.prediction == Prediction::Unparseable {
        1
    } else {
        0
    };
    let correct = if scorable
        && matches!(
            (condition.prediction, reference),
            (Prediction::Same, ReferenceDisposition::Same)
                | (Prediction::Different, ReferenceDisposition::Different)
        ) {
        1
    } else {
        0
    };
    let incorrect = if scorable { 1 - correct } else { 0 };
    ConditionMetrics {
        kind: condition.kind,
        scorable,
        correct,
        incorrect,
        answered,
        abstained_or_ambiguous,
        unparseable,
        accuracy: scorable.then_some(correct as f64),
        coverage: scorable.then_some(answered as f64),
        prediction: condition.prediction,
        prediction_confidence: condition.prediction_confidence,
    }
}

pub fn audit_panel(panel: &Panel) -> Result<AuditReport, String> {
    validate_panel_header(panel)?;
    if panel.conditions.len() != 5 {
        return Err("v0 requires exactly five matched conditions".to_owned());
    }

    let mut by_kind = BTreeMap::new();
    for condition in &panel.conditions {
        validate_condition_common(condition, &panel.matched_budget_profile_sha256)?;
        if condition.question_sha256 != panel.question_sha256 {
            return Err(
                "condition question digest differs from the panel's canonical question".to_owned(),
            );
        }
        validate_condition_kind(condition, &panel.audio_evidence_manifest_sha256)?;
        if by_kind.insert(condition.kind, condition).is_some() {
            return Err("condition kinds must be unique".to_owned());
        }
    }
    for kind in [
        ConditionKind::OracleSymbolic,
        ConditionKind::AudioDerivedSymbolic,
        ConditionKind::RawAudio,
        ConditionKind::AudioPlusPredictedStructure,
        ConditionKind::NoEvidenceControl,
    ] {
        if !by_kind.contains_key(&kind) {
            return Err("panel is missing a required matched condition".to_owned());
        }
    }

    let oracle = by_kind[&ConditionKind::OracleSymbolic];
    let audio_derived = by_kind[&ConditionKind::AudioDerivedSymbolic];
    let raw_audio = by_kind[&ConditionKind::RawAudio];
    let fusion = by_kind[&ConditionKind::AudioPlusPredictedStructure];
    let no_evidence = by_kind[&ConditionKind::NoEvidenceControl];

    let reasoner_identity_matched =
        oracle.reasoner_model_identity_sha256 == audio_derived.reasoner_model_identity_sha256;
    if !reasoner_identity_matched {
        return Err(
            "oracle and audio-derived symbolic conditions must use the same downstream reasoning head"
                .to_owned(),
        );
    }

    let raw_audio_model_identity_matched = raw_audio.model_identity_sha256
        == fusion.model_identity_sha256
        && raw_audio.model_identity_sha256 == no_evidence.model_identity_sha256
        && raw_audio.runtime_identity_sha256 == fusion.runtime_identity_sha256
        && raw_audio.runtime_identity_sha256 == no_evidence.runtime_identity_sha256
        && raw_audio.system_prompt_sha256 == fusion.system_prompt_sha256
        && raw_audio.system_prompt_sha256 == no_evidence.system_prompt_sha256
        && raw_audio.adapter_identity_sha256 == fusion.adapter_identity_sha256
        && raw_audio.adapter_identity_sha256 == no_evidence.adapter_identity_sha256
        && raw_audio.decoding_config_sha256 == fusion.decoding_config_sha256
        && raw_audio.decoding_config_sha256 == no_evidence.decoding_config_sha256;
    if !raw_audio_model_identity_matched {
        return Err(
            "raw, fusion and no-evidence controls must share model, runtime, prompt, adapter and decoding identities"
                .to_owned(),
        );
    }

    let mut conditions = panel
        .conditions
        .iter()
        .map(|condition| score_condition(condition, panel.reference_disposition))
        .collect::<Vec<_>>();
    conditions.sort_by_key(|metrics| metrics.kind);

    let metric = |kind| {
        conditions
            .iter()
            .find(|condition| condition.kind == kind)
            .expect("all required condition kinds were validated")
    };
    let scorable = matches!(
        panel.reference_disposition,
        ReferenceDisposition::Same | ReferenceDisposition::Different
    );
    let oracle_correct = scorable && metric(ConditionKind::OracleSymbolic).correct == 1;
    let audio_derived_correct =
        scorable && metric(ConditionKind::AudioDerivedSymbolic).correct == 1;
    let raw_correct = scorable && metric(ConditionKind::RawAudio).correct == 1;
    let fusion_correct =
        scorable && metric(ConditionKind::AudioPlusPredictedStructure).correct == 1;
    let no_evidence_correct =
        scorable.then(|| metric(ConditionKind::NoEvidenceControl).correct == 1);

    Ok(AuditReport {
        schema: "mel-hear-009-perception-reasoning-report-v0".to_owned(),
        panel_id: panel.panel_id.clone(),
        split: panel.split.clone(),
        item_id: panel.item_id.clone(),
        target_property: panel.target_property,
        reference_scorable: scorable,
        condition_count: conditions.len(),
        shared_budget_profile_sha256: panel.matched_budget_profile_sha256.clone(),
        shared_audio_evidence_manifest_sha256: panel.audio_evidence_manifest_sha256.clone(),
        reasoner_identity_matched,
        raw_audio_model_identity_matched,
        diagnostics: PerceptionReasoningDiagnostics {
            oracle_correct_audio_derived_incorrect: scorable
                && oracle_correct
                && !audio_derived_correct,
            oracle_incorrect_audio_derived_correct: scorable
                && !oracle_correct
                && audio_derived_correct,
            raw_audio_correct_fusion_incorrect: scorable && raw_correct && !fusion_correct,
            raw_audio_incorrect_fusion_correct: scorable && !raw_correct && fusion_correct,
            no_evidence_control_correct: no_evidence_correct,
        },
        conditions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Panel {
        serde_json::from_str(include_str!(
            "../../../benchmarks/mel/registry/mel-hear-009-perception-reasoning-synthetic-v0.json"
        ))
        .expect("retained synthetic panel must parse")
    }

    #[test]
    fn matched_panel_reproduces_expected_error_decomposition() {
        let report = audit_panel(&fixture()).expect("known-answer panel must qualify");
        assert!(report.reference_scorable);
        assert_eq!(report.condition_count, 5);
        assert!(report.reasoner_identity_matched);
        assert!(report.raw_audio_model_identity_matched);
        assert!(report.diagnostics.oracle_correct_audio_derived_incorrect);
        assert!(!report.diagnostics.oracle_incorrect_audio_derived_correct);
        assert!(!report.diagnostics.raw_audio_correct_fusion_incorrect);
        assert!(!report.diagnostics.raw_audio_incorrect_fusion_correct);
        assert_eq!(report.diagnostics.no_evidence_control_correct, Some(false));
        assert_eq!(
            report
                .conditions
                .iter()
                .find(|condition| condition.kind == ConditionKind::AudioDerivedSymbolic)
                .unwrap()
                .incorrect,
            1
        );
    }

    #[test]
    fn condition_order_does_not_change_report() {
        let panel = fixture();
        let original = audit_panel(&panel).unwrap();
        let mut reordered = panel;
        reordered.conditions.reverse();
        assert_eq!(original, audit_panel(&reordered).unwrap());
    }

    #[test]
    fn target_label_leakage_is_rejected() {
        let mut panel = fixture();
        panel.conditions[0].reference_label_exposed = true;
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn client_or_provider_input_claims_require_external_trusted_stage_verification() {
        let mut panel = fixture();
        panel.conditions[0].evidence_stage = EvidenceStage::ProviderInputAttested;
        assert!(audit_panel(&panel).is_err());
        panel.conditions[0].evidence_stage = EvidenceStage::ClientRequestBound;
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn missing_condition_is_rejected() {
        let mut panel = fixture();
        panel.conditions.pop();
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn mismatched_matched_budget_profile_is_rejected() {
        let mut panel = fixture();
        panel.conditions[1].budget_profile_sha256 =
            "abababababababababababababababababababababababababababababababab".to_owned();
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn condition_question_substitution_is_rejected() {
        let mut panel = fixture();
        panel.conditions[1].question_sha256 =
            "abababababababababababababababababababababababababababababababab".to_owned();
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn oracle_score_cannot_leak_into_raw_audio_condition() {
        let mut panel = fixture();
        panel.conditions[2].oracle_symbolic_reference_sha256 =
            Some("abababababababababababababababababababababababababababababababab".to_owned());
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn oracle_and_audio_derived_must_share_the_same_reasoner() {
        let mut panel = fixture();
        panel.conditions[1].reasoner_model_identity_sha256 =
            Some("abababababababababababababababababababababababababababababababab".to_owned());
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn raw_audio_fusion_and_no_evidence_must_share_model_identity() {
        let mut panel = fixture();
        panel.conditions[3].model_identity_sha256 =
            "abababababababababababababababababababababababababababababababab".to_owned();
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn audio_derived_requires_transcription_and_alignment_receipts() {
        let mut panel = fixture();
        panel.conditions[1].alignment_report_sha256 = None;
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn audio_conditions_must_share_the_same_materialized_evidence() {
        let mut panel = fixture();
        panel.conditions[2].audio_evidence_manifest_sha256 =
            Some("abababababababababababababababababababababababababababababababab".to_owned());
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn abstention_and_parse_failure_remain_in_the_primary_denominator() {
        let mut panel = fixture();
        panel.conditions[0].prediction = Prediction::Abstain;
        panel.conditions[0].prediction_confidence = None;
        let report = audit_panel(&panel).unwrap();
        let oracle = report
            .conditions
            .iter()
            .find(|condition| condition.kind == ConditionKind::OracleSymbolic)
            .unwrap();
        assert_eq!(oracle.incorrect, 1);
        assert_eq!(oracle.answered, 0);
        assert_eq!(oracle.accuracy, Some(0.0));
        assert_eq!(oracle.coverage, Some(0.0));
    }

    #[test]
    fn ambiguous_reference_is_not_scored_as_a_binary_error() {
        let mut panel = fixture();
        panel.reference_disposition = ReferenceDisposition::Ambiguous;
        let report = audit_panel(&panel).unwrap();
        assert!(!report.reference_scorable);
        assert!(
            report
                .conditions
                .iter()
                .all(|condition| { condition.accuracy.is_none() && condition.coverage.is_none() })
        );
        assert_eq!(report.diagnostics.no_evidence_control_correct, None);
    }
}
